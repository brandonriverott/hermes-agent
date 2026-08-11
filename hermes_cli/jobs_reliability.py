"""Provider-isolated production evidence adapter for the Jobs dispatcher.

The provider runner owns the external builder/test/reviewer sessions.  This
module owns the common, deterministic boundary: validate provider identity,
materialize bounded receipts, bind them to the observed candidate commit, and
return provider-neutral readback claims to the reliability graph.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_cli import jobs_assurance
from hermes_cli import jobs_execution
from hermes_cli import jobs_exec
from hermes_cli import jobs_receipts


_PROVIDERS = frozenset({"claude", "codex"})
_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class PhaseResult:
    """Sanitized result of fresh build/test and review sessions."""

    status: str
    commit: Optional[str]
    tests: tuple[Mapping[str, object], ...]
    review: Mapping[str, object]
    executor_exit_digest: str
    output_capture_digest: str
    failure_reason_code: Optional[str] = None
    http_status: Optional[int] = None
    safety_gate: bool = False
    response_change_digest: Optional[str] = None
    result_delta_digest: Optional[str] = None


PhaseRunner = Callable[[str, object], PhaseResult]

_DATA_DIR = Path(__file__).resolve().parent / "data"
_BUILD_SCHEMA = _DATA_DIR / "jobs-build-result.v1.schema.json"
_REVIEW_SCHEMA = _DATA_DIR / "jobs-review-result.v1.schema.json"
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM")
_MAX_PHASE_BYTES = 64 * 1024


def _remote_repository(repository: Path) -> Path:
    """Map a Mac worktree to its explicitly configured PC checkout."""
    raw = os.environ.get("HERMES_JOBS_PC_REPO_MAP", "").strip()
    if not raw:
        return Path(repository)
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise jobs_execution.AdapterError("remote repository mapping is invalid") from exc
    if not isinstance(mapping, dict):
        raise jobs_execution.AdapterError("remote repository mapping is invalid")
    target = mapping.get(str(repository))
    if not isinstance(target, str) or not Path(target).is_absolute():
        raise jobs_execution.AdapterError("remote repository mapping is unavailable")
    return Path(target)


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    timed_out: bool = False


ProcessRunner = Callable[..., ProcessResult]


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout.strip()


def _read_json(path: Path) -> Optional[dict]:
    try:
        if path.stat().st_size > _MAX_PHASE_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    structured = value.get("structured_output")
    return structured if isinstance(structured, dict) else value


def _sanitize_phase_file(path: Path) -> bytes:
    text = jobs_execution.bounded_text(path)
    if text is None:
        return b""
    body = jobs_execution.capped_utf8(jobs_exec.redact_secrets(text))
    temporary = path.with_name(path.name + ".sanitizing")
    temporary.write_bytes(body)
    temporary.chmod(0o600)
    os.replace(temporary, path)
    return body


def _default_process_runner(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    prompt: bytes,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> ProcessResult:
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                list(argv),
                cwd=str(cwd),
                env=dict(env),
                input=prompt,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        return ProcessResult(completed.returncode)
    except subprocess.TimeoutExpired:
        return ProcessResult(124, timed_out=True)
    except OSError as exc:
        raise jobs_execution.AdapterError("provider process could not start") from exc


def _phase_env(provider: str, lane_root: Path, handoff: Path) -> dict[str, str]:
    env = {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}
    if "PATH" not in env:
        env["PATH"] = os.defpath
    user_bins = [Path.home() / ".hermes" / "node" / "bin", Path.home() / ".local" / "bin"]
    prefixes = [str(path) for path in user_bins if path.is_dir()]
    if prefixes:
        env["PATH"] = os.pathsep.join([env["PATH"], *prefixes])
    executable = shutil.which(provider, path=env["PATH"])
    if executable is None:
        raise jobs_execution.AdapterError("selected provider executable is unavailable")
    private = handoff / "private"
    for child in (private, private / "home", private / "hermes", private / "tmp"):
        child.mkdir(parents=True, exist_ok=True, mode=0o700)
        child.chmod(0o700)
    env.update(
        {
            "HOME": str(private / "home"),
            "HERMES_HOME": str(private / "hermes"),
            "TMPDIR": str(private / "tmp"),
        }
    )
    if provider == "codex":
        env["CODEX_HOME"] = str(lane_root / "auth")
    else:
        env["CLAUDE_CONFIG_DIR"] = str(lane_root / "auth")
    return env


def _codex_git_metadata_dirs(worktree: Path) -> tuple[Path, ...]:
    """Return only this isolated worktree's Git metadata write locations.

    Codex's workspace sandbox permits the checkout itself, but Git stores the
    worktree index and branch ref outside it.  Granting these two resolved
    metadata directories lets the worker commit its own branch without giving
    it write access to arbitrary source paths or another lane's workspace.
    """
    locations: list[Path] = []
    for command in ("--git-dir", "--git-common-dir"):
        raw = Path(_git(worktree, "rev-parse", command))
        location = raw if raw.is_absolute() else (worktree / raw)
        location = location.resolve()
        if not location.is_dir():
            raise jobs_execution.AdapterError("selected worktree Git metadata is unavailable")
        if location not in locations:
            locations.append(location)
    return tuple(locations)


def _provider_command(
    provider: str,
    *,
    phase: str,
    context: object,
    result_path: Path,
) -> list[str]:
    schema = _BUILD_SCHEMA if phase == "build" else _REVIEW_SCHEMA
    if provider == "codex":
        command = [
            "codex",
            "exec",
            "--model",
            str(getattr(context, "model")),
            "--sandbox",
            "workspace-write" if phase == "build" else "read-only",
            "--cd",
            str(getattr(context, "worktree")),
            "--ephemeral",
            "--json",
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(result_path),
        ]
        for metadata_dir in _codex_git_metadata_dirs(Path(getattr(context, "worktree"))):
            command.extend(["--add-dir", str(metadata_dir)])
        return [*command, "-"]
    return [
        "claude",
        "-p",
        "--model",
        str(getattr(context, "model")),
        "--effort",
        str(getattr(context, "effort")),
        "--permission-mode",
        "acceptEdits" if phase == "build" else "plan",
        "--safe-mode",
        "--no-session-persistence",
        "--output-format",
        "json",
        "--json-schema",
        schema.read_text(encoding="utf-8"),
    ]


def _build_prompt(context: object) -> bytes:
    return (
        "Complete the Job below in this isolated worktree. Run the relevant "
        "tests, commit the finished change, and return only the required "
        "structured result. Never claim a passing test that you did not run.\n\n"
        + str(getattr(context, "goal"))
    ).encode("utf-8")


def _review_prompt(context: object, commit: str) -> bytes:
    contract = jobs_assurance.AssuranceContract.from_mapping(
        getattr(context, "assurance_contract", None)
    )
    return (
        "Independently review the candidate diff against its approved base. "
        "Do not modify files and do not rely on the builder summary as evidence. "
        "Run the named critical-user-journey checks yourself in the candidate "
        "environment and capture artifact digests. Return UNABLE_TO_VERIFY when "
        "the required evidence cannot be independently obtained. Return PASS only "
        "when there are no unresolved correctness, security, evidence, user-outcome, "
        "or maintainability findings. For money, permissions, or state risks, "
        "explicitly verify every named consumer and egress path and assess the "
        "declared rollback behavior. knowledge_closure must be null unless an "
        "actual Vault Steward disposition receipt is available.\n"
        f"BASE={getattr(context, 'base_commit')}\nCANDIDATE={commit}\n"
        "ASSURANCE_CONTRACT="
        + json.dumps(contract.to_mapping(), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


class LocalProviderPhaseRunner:
    """Run fresh local builder/test and reviewer sessions in one physical lane."""

    def __init__(self, *, process_runner: ProcessRunner = _default_process_runner):
        self._process_runner = process_runner

    def __call__(self, provider: str, context: object) -> PhaseResult:
        lane_id = str(getattr(context, "lane_id", ""))
        expected_host = "mac" if platform.system().lower() == "darwin" else "pc"
        if f"-{expected_host}-" not in lane_id:
            raise jobs_execution.AdapterError(
                "selected physical lane is not local to this runtime"
            )
        lane_root = Path(getattr(context, "lane_root"))
        worktree = Path(getattr(context, "worktree"))
        handoff = lane_root / "handoffs" / str(getattr(context, "attempt_id"))
        handoff.mkdir(parents=True, exist_ok=False, mode=0o700)
        handoff.chmod(0o700)
        env = _phase_env(provider, lane_root, handoff)

        _git(
            Path(getattr(context, "repository")),
            "worktree",
            "add",
            "-b",
            str(getattr(context, "branch")),
            str(worktree),
            str(getattr(context, "base_commit")),
        )
        build_result_path = handoff / "build-result.json"
        build_stdout = handoff / "build-stdout.json"
        build_stderr = handoff / "build-stderr.log"
        build = self._process_runner(
            _provider_command(
                provider,
                phase="build",
                context=context,
                result_path=build_result_path,
            ),
            cwd=worktree,
            env=env,
            prompt=_build_prompt(context),
            stdout_path=build_stdout,
            stderr_path=build_stderr,
            timeout=max(60, int(getattr(context, "max_turns", 120)) * 30),
        )
        reported = _read_json(
            build_result_path if provider == "codex" else build_stdout
        )
        captured = b"".join(
            _sanitize_phase_file(path)
            for path in (build_stdout, build_stderr)
            if path.is_file()
        )
        exit_digest = _sha256(
            f"{provider}:build:{build.returncode}:{build.timed_out}".encode()
        )
        output_digest = _sha256(captured)
        if (
            build.returncode != 0
            or build.timed_out
            or not isinstance(reported, dict)
            or reported.get("outcome") != "succeeded"
        ):
            return PhaseResult(
                status="failed",
                commit=None,
                tests=(),
                review={},
                executor_exit_digest=exit_digest,
                output_capture_digest=output_digest,
                failure_reason_code=(
                    "PROCESS_TIMEOUT" if build.timed_out else "PROCESS_CRASHED"
                ),
            )

        commit = _git(worktree, "rev-parse", "HEAD")
        branch = _git(worktree, "rev-parse", "--abbrev-ref", "HEAD")
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                str(worktree),
                "merge-base",
                "--is-ancestor",
                str(getattr(context, "base_commit")),
                commit,
            ],
            check=False,
            capture_output=True,
            timeout=120,
        )
        if (
            branch != getattr(context, "branch")
            or commit == getattr(context, "base_commit")
            or ancestry.returncode != 0
        ):
            return PhaseResult(
                status="failed",
                commit=commit,
                tests=(),
                review={},
                executor_exit_digest=exit_digest,
                output_capture_digest=output_digest,
                failure_reason_code="NO_CANDIDATE_COMMIT",
            )
        tests = reported.get("tests")
        if not isinstance(tests, list) or not tests:
            return PhaseResult(
                status="failed",
                commit=commit,
                tests=(),
                review={},
                executor_exit_digest=exit_digest,
                output_capture_digest=output_digest,
                failure_reason_code="TEST_EVIDENCE_MISSING",
            )

        review_result_path = handoff / "review-result.json"
        review_stdout = handoff / "review-stdout.json"
        review_stderr = handoff / "review-stderr.log"
        review_run = self._process_runner(
            _provider_command(
                provider,
                phase="review",
                context=context,
                result_path=review_result_path,
            ),
            cwd=worktree,
            env=env,
            prompt=_review_prompt(context, commit),
            stdout_path=review_stdout,
            stderr_path=review_stderr,
            timeout=max(60, int(getattr(context, "max_turns", 120)) * 15),
        )
        review = _read_json(
            review_result_path if provider == "codex" else review_stdout
        )
        for path in (review_stdout, review_stderr):
            if path.is_file():
                _sanitize_phase_file(path)
        if review_run.returncode != 0 or review_run.timed_out or review is None:
            return PhaseResult(
                status="failed",
                commit=commit,
                tests=tuple(dict(item) for item in tests if isinstance(item, dict)),
                review={},
                executor_exit_digest=exit_digest,
                output_capture_digest=output_digest,
                failure_reason_code="REVIEWER_PROCESS_FAILED",
            )
        return PhaseResult(
            status="succeeded",
            commit=commit,
            tests=tuple(dict(item) for item in tests if isinstance(item, dict)),
            review=review,
            executor_exit_digest=exit_digest,
            output_capture_digest=output_digest,
        )


_SSH_HOST_RE = re.compile(r"\A[A-Za-z0-9_.:@-]{1,255}\Z")


class SSHProviderPhaseRunner:
    """Execute one provider pipeline in the immutable runtime on the PC."""

    def __init__(
        self,
        *,
        host: str,
        runtime_root: Path,
        lane_root: Path,
        ssh_binary: str = "ssh",
        subprocess_run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        if _SSH_HOST_RE.fullmatch(host) is None:
            raise ValueError("remote Jobs host is invalid")
        for value, label in ((runtime_root, "runtime"), (lane_root, "lane")):
            if not Path(value).is_absolute():
                raise ValueError(f"remote {label} root must be absolute")
        self.host = host
        self.runtime_root = Path(runtime_root)
        self.lane_root = Path(lane_root)
        self.ssh_binary = ssh_binary
        self._subprocess_run = subprocess_run

    def preflight(
        self,
        *,
        provider: str,
        repository: Path,
        base_commit: str,
        branch: str,
        lane_id: str,
    ) -> bool:
        payload = {
            "schema_version": 1,
            "provider": provider,
            "repository": str(_remote_repository(repository)),
            "base_commit": base_commit,
            "branch": branch,
            "lane_root": str(self.lane_root / lane_id),
            "lane_id": lane_id,
        }
        script = self.runtime_root / "scripts" / "jobs_remote_worker.py"
        try:
            completed = self._subprocess_run(
                [self.ssh_binary, self.host, "python3", str(script), "preflight"],
                input=json.dumps(payload, sort_keys=True).encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=45,
            )
            return completed.returncode == 0 and completed.stdout == b'{"ok":true}'
        except (OSError, subprocess.SubprocessError):
            return False

    def __call__(self, provider: str, context: object) -> PhaseResult:
        remote_lane = self.lane_root / str(getattr(context, "lane_id"))
        payload = {
            "schema_version": 1,
            "provider": provider,
            "context": {
                "job_id": str(getattr(context, "job_id")),
                "job_number": int(getattr(context, "job_number")),
                "job_name": str(getattr(context, "job_name")),
                "goal": str(getattr(context, "goal")),
                "attempt_id": str(getattr(context, "attempt_id")),
                "ordinal": int(getattr(context, "ordinal")),
                "repository": str(
                    _remote_repository(Path(getattr(context, "repository")))
                ),
                "base_commit": str(getattr(context, "base_commit")),
                "branch": str(getattr(context, "branch")),
                "worktree": str(
                    remote_lane
                    / "worktrees"
                    / f"{getattr(context, 'job_id')}-{getattr(context, 'ordinal')}"
                ),
                "lane_root": str(remote_lane),
                "requested_lane": str(getattr(context, "requested_lane")),
                "lane_id": str(getattr(context, "lane_id")),
                "executor": str(getattr(context, "executor")),
                "specialist": str(getattr(context, "specialist")),
                "model": str(getattr(context, "model")),
                "effort": str(getattr(context, "effort")),
                "max_turns": int(getattr(context, "max_turns")),
                "assurance_contract": dict(
                    getattr(context, "assurance_contract")
                ),
            },
        }
        script = self.runtime_root / "scripts" / "jobs_remote_worker.py"
        try:
            completed = self._subprocess_run(
                [self.ssh_binary, self.host, "python3", str(script), "execute"],
                input=json.dumps(payload, sort_keys=True).encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=max(300, int(getattr(context, "max_turns", 120)) * 60),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise jobs_execution.AdapterError(
                "remote provider transport is unavailable"
            ) from exc
        if completed.returncode != 0 or len(completed.stdout) > _MAX_PHASE_BYTES:
            raise jobs_execution.AdapterError("remote provider execution failed")
        try:
            raw = json.loads(completed.stdout.decode("utf-8"))
            if not isinstance(raw, dict) or set(raw) != {
                "status",
                "commit",
                "tests",
                "review",
                "executor_exit_digest",
                "output_capture_digest",
                "failure_reason_code",
                "http_status",
                "safety_gate",
                "response_change_digest",
                "result_delta_digest",
            }:
                raise ValueError("shape")
            return PhaseResult(
                status=str(raw["status"]),
                commit=raw["commit"] if isinstance(raw["commit"], str) else None,
                tests=tuple(
                    dict(item) for item in raw["tests"] if isinstance(item, dict)
                ),
                review=dict(raw["review"]) if isinstance(raw["review"], dict) else {},
                executor_exit_digest=str(raw["executor_exit_digest"]),
                output_capture_digest=str(raw["output_capture_digest"]),
                failure_reason_code=(
                    raw["failure_reason_code"]
                    if isinstance(raw["failure_reason_code"], str)
                    else None
                ),
                http_status=(
                    raw["http_status"] if isinstance(raw["http_status"], int) else None
                ),
                safety_gate=raw["safety_gate"] is True,
                response_change_digest=(
                    raw["response_change_digest"]
                    if isinstance(raw["response_change_digest"], str)
                    else None
                ),
                result_delta_digest=(
                    raw["result_delta_digest"]
                    if isinstance(raw["result_delta_digest"], str)
                    else None
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise jobs_execution.AdapterError(
                "remote provider result is malformed"
            ) from exc


class FleetProviderPhaseRunner:
    """Select physical transport only; never select or substitute a provider."""

    def __init__(self, *, local: PhaseRunner, remote_pc: Optional[PhaseRunner]):
        self.local = local
        self.remote_pc = remote_pc

    def __call__(self, provider: str, context: object) -> PhaseResult:
        lane_id = str(getattr(context, "lane_id", ""))
        system = platform.system().lower()
        local_host = "mac" if system == "darwin" else "pc"
        if f"-{local_host}-" in lane_id:
            return self.local(provider, context)
        if "-pc-" in lane_id and self.remote_pc is not None:
            return self.remote_pc(provider, context)
        raise jobs_execution.AdapterError(
            "selected physical lane has no installed transport"
        )


def _remote_runner_from_environment() -> Optional[SSHProviderPhaseRunner]:
    host = os.environ.get("HERMES_JOBS_PC_HOST", "").strip()
    runtime = os.environ.get("HERMES_JOBS_PC_RUNTIME_ROOT", "").strip()
    lanes = os.environ.get("HERMES_JOBS_PC_LANE_ROOT", "").strip()
    if not (host and runtime and lanes):
        return None
    try:
        return SSHProviderPhaseRunner(
            host=host, runtime_root=Path(runtime), lane_root=Path(lanes)
        )
    except ValueError:
        return None


def production_phase_runner_from_environment() -> FleetProviderPhaseRunner:
    """Build the exact local/PC transport registry from explicit activation env."""

    return FleetProviderPhaseRunner(
        local=LocalProviderPhaseRunner(),
        remote_pc=_remote_runner_from_environment(),
    )


def production_remote_preflight(
    *,
    provider: str,
    repository: Path,
    base_commit: str,
    branch: str,
    lane_id: str,
) -> bool:
    runner = _remote_runner_from_environment()
    if runner is None:
        return False
    return runner.preflight(
        provider=provider,
        repository=repository,
        base_commit=base_commit,
        branch=branch,
        lane_id=lane_id,
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(body) > jobs_execution.MAX_PRESERVED_BYTES:
        raise jobs_execution.AdapterError("evidence receipt exceeds the byte cap")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(body)
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def _gate_envelope(context: object, *, kind: str, commit: str) -> dict[str, object]:
    review_attempt = max(0, int(getattr(context, "ordinal")) - 1)
    return {
        "job": str(getattr(context, "job_id")),
        "attempt": review_attempt,
        "commit": commit,
        "base": str(getattr(context, "base_commit")),
        "kind": kind,
        "stamped_at": "dispatcher-observed",
    }


class ProductionReliabilityAdapter:
    """One immutable provider identity backed by one explicit phase runner."""

    def __init__(self, executor: str, *, phase_runner: PhaseRunner):
        if executor not in _PROVIDERS:
            raise ValueError("executor must be claude or codex")
        if not callable(phase_runner):
            raise TypeError("phase_runner must be callable")
        self.executor = executor
        self._phase_runner = phase_runner

    def __call__(self, context: object) -> jobs_execution.ReliabilityExecution:
        observed = str(getattr(context, "executor", ""))
        lane_id = str(getattr(context, "lane_id", ""))
        if observed != self.executor or not lane_id.startswith(self.executor + "-"):
            raise jobs_execution.AdapterError(
                "selected provider identity does not match its reliability adapter"
            )
        lane_root = getattr(context, "lane_root", None)
        if lane_root is None:
            raise jobs_execution.AdapterError("selected lane has no isolated root")

        result = self._phase_runner(self.executor, context)
        if not isinstance(result, PhaseResult):
            raise jobs_execution.AdapterError("provider returned an invalid phase result")
        commit = result.commit
        if result.status != "succeeded" or commit is None:
            return jobs_execution.ReliabilityExecution(
                status=result.status,
                commit=commit,
                worktree=Path(getattr(context, "worktree")),
                artifacts=(),
                executor_exit_digest=result.executor_exit_digest,
                output_capture_digest=result.output_capture_digest,
                failure_reason_code=result.failure_reason_code,
                http_status=result.http_status,
                safety_gate=result.safety_gate,
                response_change_digest=result.response_change_digest,
                result_delta_digest=result.result_delta_digest,
            )
        if _SHA_RE.fullmatch(commit) is None:
            raise jobs_execution.AdapterError("provider candidate commit is malformed")
        if not result.tests:
            raise jobs_execution.AdapterError("provider supplied no test evidence")
        try:
            jobs_assurance.validate_review_result(result.review)
            jobs_assurance.OutcomeEvidence.from_mapping(
                result.review.get("outcome_evidence")
            )
            closure_value = result.review.get("knowledge_closure")
            if closure_value is not None:
                jobs_assurance.KnowledgeClosureReceipt.from_mapping(closure_value)
        except (
            ValueError,
            jobs_assurance.InvalidOutcomeEvidence,
            jobs_assurance.InvalidClosureReceipt,
        ) as exc:
            raise jobs_execution.AdapterError(
                "provider review evidence is invalid"
            ) from exc

        evidence_dir = Path(lane_root) / "receipts" / str(
            getattr(context, "attempt_id")
        )
        tests_doc: dict[str, object] = {
            "tests": [dict(item) for item in result.tests],
            "skipped": [],
            "verdict_hint": "clean",
            "_gate": _gate_envelope(context, kind="tests", commit=commit),
        }
        review_doc = dict(result.review)
        review_doc["_gate"] = _gate_envelope(
            context, kind="review", commit=commit
        )
        output_doc: dict[str, object] = {
            "schema_version": 1,
            "executor": self.executor,
            "status": result.status,
            "commit": commit,
            "executor_exit_digest": result.executor_exit_digest,
            "output_capture_digest": result.output_capture_digest,
        }
        tests_path = evidence_dir / "tests.json"
        review_attempt = max(0, int(getattr(context, "ordinal")) - 1)
        review_path = evidence_dir / f"review-{review_attempt}.json"
        output_path = evidence_dir / "output.json"
        _write_json(tests_path, tests_doc)
        _write_json(review_path, review_doc)
        _write_json(output_path, output_doc)

        return jobs_execution.ReliabilityExecution(
            status="succeeded",
            commit=commit,
            worktree=Path(getattr(context, "worktree")),
            artifacts=(
                jobs_execution.claim_artifact(output_path, name="output"),
                jobs_execution.claim_artifact(tests_path, name="tests"),
            ),
            executor_exit_digest=result.executor_exit_digest,
            output_capture_digest=result.output_capture_digest,
        )


def production_gate(context: object, execution: jobs_execution.ReliabilityExecution):
    """Re-evaluate bound test/review receipts with the authoritative gate."""

    from hermes_cli import jobs_run
    from scripts import jobs_evidence_gate as evidence_gate

    lane_root = getattr(context, "lane_root", None)
    if lane_root is None or execution.commit is None:
        raise jobs_execution.AdapterError("gate input lacks lane or commit identity")
    evidence_dir = Path(lane_root) / "receipts" / str(
        getattr(context, "attempt_id")
    )
    evidence_gate.settle(
        evidence_dir,
        job=str(getattr(context, "job_id")),
        branch=str(getattr(context, "branch")),
        base=str(getattr(context, "base_commit")),
        commit=execution.commit,
        attempt=max(0, int(getattr(context, "ordinal")) - 1),
        rounds=int(getattr(context, "ordinal")),
        build_exit=0,
        effort=str(getattr(context, "effort", "max")),
    )
    normalized = evidence_gate.graph_settlement(
        evidence_dir,
        job=str(getattr(context, "job_id")),
        base=str(getattr(context, "base_commit")),
    )
    return jobs_run.adapt_gate_settlement(
        normalized,
        job_dir=evidence_dir,
        review_attempt=max(0, int(getattr(context, "ordinal")) - 1),
    )


def production_completion_gate(context: object, gate: object):
    """Authorize graph completion only from a successful identity-bound gate."""

    from hermes_cli import jobs_dispatch

    if (
        getattr(gate, "action_outcome", None) != "succeeded"
        or getattr(gate, "identity_verified", None) is not True
        or getattr(gate, "commit", None) is None
    ):
        raise jobs_execution.AdapterError("completion gate evidence is not successful")
    material = {
        "job_id": str(getattr(context, "job_id")),
        "attempt_id": str(getattr(context, "attempt_id")),
        "commit": str(getattr(gate, "commit")),
        "executor": str(getattr(context, "executor")),
        "lane_id": str(getattr(context, "lane_id")),
    }
    gate_digest = jobs_receipts.digest_bytes(
        jobs_receipts.canonical_json_bytes(material)
    )
    completion_digest = jobs_receipts.digest_bytes(
        jobs_receipts.canonical_json_bytes(
            {**material, "decision": "graph-completion-authorized"}
        )
    )
    return jobs_dispatch.ActivationDecision(
        status="PASS",
        reason_code="OK",
        activation_gate_digest=gate_digest,
        completion_receipt_digest=completion_digest,
    )


def _review_receipt(context: object) -> dict:
    lane_root = getattr(context, "lane_root", None)
    if lane_root is None:
        raise jobs_execution.AdapterError("outcome gate lacks lane identity")
    evidence_dir = Path(lane_root) / "receipts" / str(
        getattr(context, "attempt_id")
    )
    review_attempt = max(0, int(getattr(context, "ordinal")) - 1)
    path = evidence_dir / f"review-{review_attempt}.json"
    if path.is_symlink():
        raise jobs_execution.AdapterError("review receipt must not be a symlink")
    review = _read_json(path)
    if review is None:
        raise jobs_execution.AdapterError("review receipt is unavailable")
    return review


def production_outcome_gate(context: object, _execution: object):
    """Verify the named user journey from the independent reviewer receipt."""

    contract = jobs_assurance.AssuranceContract.from_mapping(
        getattr(context, "assurance_contract", None)
    )
    review = _review_receipt(context)
    validated_review = jobs_assurance.validate_review_result(review)
    evidence = jobs_assurance.OutcomeEvidence.from_mapping(
        review.get("outcome_evidence")
    )
    decision = jobs_assurance.verify_outcome(contract, evidence)
    if not validated_review.authorizes_completion and decision.status == "PASS":
        return jobs_assurance.OutcomeDecision(
            "BLOCKED", "REVIEW_NOT_PASSING", evidence.digest
        )
    return decision


def production_knowledge_closure_gate(context: object, _execution: object):
    """Require a Vault Steward closure receipt when durable knowledge changed."""

    contract = jobs_assurance.AssuranceContract.from_mapping(
        getattr(context, "assurance_contract", None)
    )
    review = _review_receipt(context)
    closure_value = review.get("knowledge_closure")
    if contract.knowledge_closure_required:
        if closure_value is None:
            raise jobs_execution.AdapterError(
                "required Vault Steward closure receipt is missing"
            )
        return jobs_assurance.KnowledgeClosureReceipt.from_mapping(closure_value)
    if closure_value is not None:
        return jobs_assurance.KnowledgeClosureReceipt.from_mapping(closure_value)
    digest = jobs_receipts.digest_bytes(
        jobs_receipts.canonical_json_bytes(
            {
                "job_id": str(getattr(context, "job_id")),
                "attempt_id": str(getattr(context, "attempt_id")),
                "contract_digest": contract.digest,
                "disposition": "NOT_APPLICABLE",
            }
        )
    )
    return jobs_assurance.KnowledgeClosureReceipt.from_mapping(
        {
            "disposition": "NOT_APPLICABLE",
            "canonical_note_path": None,
            "retrieval_confirmed": False,
            "steward_receipt_digest": digest,
        }
    )


__all__ = [
    "LocalProviderPhaseRunner",
    "FleetProviderPhaseRunner",
    "PhaseResult",
    "ProcessResult",
    "ProductionReliabilityAdapter",
    "SSHProviderPhaseRunner",
    "production_phase_runner_from_environment",
    "production_remote_preflight",
    "production_completion_gate",
    "production_gate",
    "production_knowledge_closure_gate",
    "production_outcome_gate",
]
