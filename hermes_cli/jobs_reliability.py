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


PhaseRunner = Callable[[str, object], PhaseResult]

_DATA_DIR = Path(__file__).resolve().parent / "data"
_BUILD_SCHEMA = _DATA_DIR / "jobs-build-result.v1.schema.json"
_REVIEW_SCHEMA = _DATA_DIR / "jobs-review-result.v1.schema.json"
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM")
_MAX_PHASE_BYTES = 64 * 1024


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


def _provider_command(
    provider: str,
    *,
    phase: str,
    context: object,
    result_path: Path,
) -> list[str]:
    schema = _BUILD_SCHEMA if phase == "build" else _REVIEW_SCHEMA
    if provider == "codex":
        return [
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
            "-",
        ]
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
    return (
        "Independently review the candidate diff against its approved base. "
        "Do not modify files. Return PASS only when there are no unresolved "
        "correctness, security, evidence, or maintainability findings.\n"
        f"BASE={getattr(context, 'base_commit')}\nCANDIDATE={commit}\n"
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
    return {
        "job": str(getattr(context, "job_id")),
        "attempt": 0,
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
            )
        if _SHA_RE.fullmatch(commit) is None:
            raise jobs_execution.AdapterError("provider candidate commit is malformed")
        if not result.tests:
            raise jobs_execution.AdapterError("provider supplied no test evidence")
        if result.review.get("verdict") not in {"PASS", "NEEDS_CHANGES"}:
            raise jobs_execution.AdapterError("provider review verdict is invalid")

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
        review_path = evidence_dir / "review-0.json"
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


def unavailable_phase_runner(_provider: str, _context: object) -> PhaseResult:
    """Fail closed until the concrete subprocess phase runner is installed."""

    raise jobs_execution.AdapterError("production provider runner is unavailable")


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
        attempt=0,
        rounds=1,
        build_exit=0,
        effort=str(getattr(context, "effort", "max")),
    )
    normalized = evidence_gate.graph_settlement(
        evidence_dir,
        job=str(getattr(context, "job_id")),
        base=str(getattr(context, "base_commit")),
    )
    return jobs_run.adapt_gate_settlement(
        normalized, job_dir=evidence_dir, review_attempt=0
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


__all__ = [
    "LocalProviderPhaseRunner",
    "PhaseResult",
    "ProcessResult",
    "ProductionReliabilityAdapter",
    "production_completion_gate",
    "production_gate",
]
