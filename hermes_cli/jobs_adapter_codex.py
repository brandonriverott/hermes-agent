"""Codex CLI execution adapter — one Job envelope in, one classified result out.

The adapter is the Codex mirror of :mod:`hermes_cli.jobs_adapter_claude`: it is
the only place in Jobs that launches the Codex CLI, it creates an isolated Git
worktree at the approved base commit, and it reads back what actually happened
from Git rather than from the model's self-report. It deliberately does **not**
touch the Jobs store.

Provider isolation is structural, not conventional:

- The adapter's argv is exactly ``codex exec --model gpt-5.6-sol ...``. There is
  no ``claude`` executable, no injected Claude worker, and no fallback path.
- The environment allowlist never inherits ``ANTHROPIC_API_KEY``,
  ``CLAUDE_API_KEY``, or the live Jobs DB path; instead the worker gets private
  ``HOME``/``HERMES_HOME``/``TMPDIR`` plus the selected lane's ``CODEX_HOME``
  auth root.
- ``spec.model`` must equal the canonical Codex model and the envelope
  specialist must be the Codex specialist before a single provider process
  starts. A mismatch is an :class:`AdapterError`, never a substitution.

Timeout classification follows the provider-neutral contract
(:func:`~hermes_cli.jobs_exec.classify_worker_result`): a wall-clock kill is
``interrupted``/``infrastructure`` — identical to Claude — so downstream
receipts and graph transitions read the same for both providers.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from hermes_cli import jobs_exec as jx
from hermes_cli import jobs_identity as ji
from hermes_cli import jobs_skills as jskills
from hermes_cli.jobs_contract import JobEnvelope
from hermes_cli.jobs_execution import (
    MAX_PRESERVED_BYTES,
    AdapterError,
    bounded_text,
    capped_utf8,
)

# Only these variables are inherited from the parent. Everything else the
# worker needs is constructed below — inheriting the parent environment would
# hand a builder every credential this process happens to hold (Claude keys
# included), and inheriting HOME or HERMES_HOME would point it at the
# operator's live Jobs store.
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM")

# Built per attempt, under the attempt's own private root, and never inherited.
_PRIVATE_ENV = ("HOME", "HERMES_HOME", "TMPDIR")

# The lane-scoped auth root. Set explicitly from the selected lane; never
# derived from the operator's shell.
CODEX_HOME_ENV = "CODEX_HOME"

# Every variable this module constructs per attempt. A caller may not supply
# one: each names something about *this* attempt's isolation, so accepting an
# override would let an "extra" re-point the containment the allowlist exists
# to build.
_CONSTRUCTED_ENV = _PRIVATE_ENV + (CODEX_HOME_ENV, jskills.SKILLS_ENV)

# The receipt keeps a tail, not a transcript. Read a little more than the
# redactor will keep so truncation happens on a boundary we control.
_OUTPUT_TAIL_BYTES = 4000

# A result file is a handful of JSON fields; a huge one is a worker misbehaving.
_MAX_RESULT_BYTES = 64 * 1024

# The schema file ships with the release and bounds what Codex may emit.
_SCHEMA_FILE = Path(__file__).resolve().parent / "data" / "jobs-codex-result.v1.schema.json"

_GIT_TIMEOUT = 120


@dataclass(frozen=True)
class AdapterResult:
    """What the adapter observed. Terminal status plus the evidence behind it."""

    status: str
    failure_class: Optional[str]
    repository: str
    branch: str
    worktree: str
    commit: Optional[str]
    receipt: dict

    def to_dict(self) -> dict:
        return asdict(self)


def branch_name(envelope: JobEnvelope) -> str:
    """The branch this attempt owns: stable, unique, and capability-free."""
    return f"jobs/{envelope.number}-{envelope.attempt_id}"


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git(cwd, *args, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
    )


def _git_out(cwd, *args) -> Optional[str]:
    """Stdout of a git command, or ``None`` if the command failed."""
    try:
        proc = _git(cwd, *args, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _codex_version() -> str:
    """The installed Codex CLI version, or ``unknown`` when it cannot be read.

    Bounded and non-fatal: a version string is evidence metadata, not a gate.
    The adapter never fabricates a version when the binary is absent.
    """
    try:
        proc = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    return first or "unknown"


def preflight(spec: jx.ExecutionSpec) -> None:
    """Refuse now if the repository or the exact base commit is not real."""
    repo = Path(spec.repo_path)
    if not repo.is_dir():
        raise AdapterError(f"repository path is not a directory: {repo}")
    if _git_out(repo, "rev-parse", "--git-dir") is None:
        raise AdapterError(f"repository path is not a git repository: {repo}")
    proc = _git(repo, "cat-file", "-e", f"{spec.base_commit}^{{commit}}", check=False)
    if proc.returncode != 0:
        raise AdapterError(
            f"base commit {spec.base_commit} is not a commit in {repo}"
        )


def preflight_skills(skills, *, name: str, goal: str) -> jskills.SkillPlan:
    """Decide — and prove — this attempt's skill attachments. Reads, never writes."""
    try:
        return jskills.plan(skills, text=f"{name}\n{goal}")
    except jskills.SkillAttachmentError as exc:
        raise AdapterError(f"skill attachment refused: {exc}")


# ---------------------------------------------------------------------------
# Worker launch
# ---------------------------------------------------------------------------


def _private_env_root(handoff: Path):
    """Create this attempt's private ``(HOME, HERMES_HOME, TMPDIR)`` directories."""
    paths = tuple(handoff / name for name in ("home", "hermes", "tmp"))
    for path in paths:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
    return paths


def _contained(path: Path, root: Path) -> bool:
    """True when ``path`` really resolves inside ``root``. Symlinks included."""
    try:
        resolved, base = path.resolve(), root.resolve()
    except OSError:
        return False
    return resolved == base or base in resolved.parents


def _provider_neutral_path(raw: str) -> str:
    """Remove Claude-specific executable roots from a Codex worker PATH."""
    parts = [
        part
        for part in raw.split(os.pathsep)
        if part and "claude" not in part.casefold()
    ]
    if not parts:
        raise AdapterError("no provider-neutral PATH is available to the worker")
    return os.pathsep.join(parts)


def _worker_env(
    *,
    codex_home: Path,
    private: Sequence[Path],
    root: Path,
    extra_env: Optional[Mapping[str, str]],
    skills_dir: Optional[Path] = None,
) -> dict:
    """The worker's complete environment: allowlist + private identity + lane root.

    Every private path is re-checked against the attempt's root here,
    immediately before the launch that would use it. ``CODEX_HOME`` is the one
    lane fact this attempt tells the worker: where its provider auth lives.
    Claude-shaped keys are refused outright rather than filtered, because a
    Codex worker must never carry Claude credentials.
    """
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    if "PATH" not in env:
        raise AdapterError("no PATH is available to give the worker")
    env["PATH"] = _provider_neutral_path(env["PATH"])
    for name, path in zip(_PRIVATE_ENV, private):
        if not _contained(path, root):
            raise AdapterError(
                f"{name} for this attempt does not resolve inside its private "
                f"root {root}"
            )
        env[name] = str(path.resolve())
    env[CODEX_HOME_ENV] = str(codex_home.resolve())
    for key, value in (extra_env or {}).items():
        try:
            jx.reject_secret_key(key)
        except jx.UnsafeReceipt:
            raise AdapterError(
                f"secret-shaped environment key {key!r} is refused for a Codex worker"
            )
        if str(key) in _CONSTRUCTED_ENV:
            raise AdapterError(
                f"{key} is constructed per attempt and cannot be supplied"
            )
        env[str(key)] = str(value)
    if skills_dir is not None:
        if not _contained(skills_dir, root):
            raise AdapterError(
                f"the skill attachment directory {skills_dir} does not resolve "
                f"inside this attempt's workspace root {root}"
            )
        env[jskills.SKILLS_ENV] = str(skills_dir)
    return env


def _read_tail(path: Path) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > _OUTPUT_TAIL_BYTES:
                fh.seek(size - _OUTPUT_TAIL_BYTES)
            raw = fh.read(_OUTPUT_TAIL_BYTES)
    except OSError:
        return ""
    return raw.decode("utf-8", "replace")


def _read_result(path: Path) -> Optional[dict]:
    try:
        if path.stat().st_size > _MAX_RESULT_BYTES:
            return None
        parsed = jx.loads_strict(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _write_private(path: Path, body: bytes) -> None:
    """Replace ``path`` with ``body``, owner-only, leaving no other copy behind."""
    tmp = path.with_name(path.name + ".sanitizing")
    try:
        with open(
            os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb"
        ) as fh:
            fh.write(body)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    except OSError:
        for victim in (tmp, path):
            try:
                victim.unlink()
            except OSError:
                pass


def _sanitize_log(path: Path) -> None:
    text = bounded_text(path)
    if text is None:
        return
    _write_private(path, capped_utf8(jx.redact_secrets(text)))


def _sanitize_result(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size <= _MAX_RESULT_BYTES:
        try:
            parsed = jx.loads_strict(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            parsed = None
        if parsed is not None:
            encoded = json.dumps(
                jx.redact_json(parsed), ensure_ascii=False, sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
            if len(encoded) <= MAX_PRESERVED_BYTES:
                _write_private(path, encoded)
                return
    _sanitize_log(path)


def _launch(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict,
    log_path: Path,
    prompt: bytes,
    wall_clock_seconds: int,
    heartbeat_seconds: int,
    on_heartbeat: Optional[Callable[[], None]],
) -> tuple[Optional[int], bool]:
    """Run Codex to completion, beating on cadence. Returns (rc, timed_out).

    The prompt rides on stdin (the trailing ``-`` in the argv contract), so a
    chatty model can never fill a pipe the parent has to drain. Output goes to
    a file rather than a pipe for the same reason the Claude adapter does: a
    provider that never stops talking must not deadlock the parent that is
    supposed to be timing it out.
    """
    with log_path.open("wb") as log:
        try:
            proc = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=env,
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            raise AdapterError(f"cannot launch the Codex CLI: {exc}") from exc

        stdin = proc.stdin
        if stdin is not None:
            try:
                stdin.write(prompt)
                stdin.close()
            except (BrokenPipeError, OSError):
                pass

    start = time.monotonic()
    deadline = start + wall_clock_seconds
    next_beat = start + heartbeat_seconds
    while True:
        now = time.monotonic()
        if now >= deadline:
            _kill_process(proc)
            return (None, True)
        try:
            rc = proc.wait(timeout=max(0.05, min(deadline, next_beat) - now))
            return (rc, False)
        except subprocess.TimeoutExpired:
            if time.monotonic() >= next_beat:
                if on_heartbeat is not None:
                    try:
                        on_heartbeat()
                    except Exception:
                        pass
                next_beat = time.monotonic() + heartbeat_seconds


def _kill_process(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.SubprocessError:
        pass


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def run_codex_attempt(
    envelope: JobEnvelope,
    spec: jx.ExecutionSpec,
    *,
    codex_home: Optional[str] = None,
    workspace_root,
    wall_clock_seconds: int,
    heartbeat_seconds: Optional[int] = None,
    on_heartbeat: Optional[Callable[[], None]] = None,
    extra_env: Optional[Mapping[str, str]] = None,
    worker_command: Optional[Sequence[str]] = None,
) -> AdapterResult:
    """Execute one Codex attempt in an isolated worktree and report what happened.

    ``codex_home`` is the selected lane's auth root; without it the adapter
    cannot honestly run (there is no implicit Codex login).  ``worker_command``
    is accepted for legacy-seam compatibility but never used — Codex's launch
    contract is fixed, and injecting a worker would violate provider truth.
    """
    if not codex_home:
        raise AdapterError(
            "no Codex lane auth root was supplied; refusing to run without CODEX_HOME"
        )
    if spec.model != ji.CODEX_MODEL:
        raise AdapterError("refusing to run Codex with a non-Codex model")
    if envelope.specialist != ji.CODEX_SPECIALIST:
        raise AdapterError("refusing to run Codex for a non-Codex specialist")

    plan = preflight_skills(envelope.skills, name=envelope.name, goal=envelope.goal)
    preflight(spec)

    if heartbeat_seconds is None:
        heartbeat_seconds = jx.heartbeat_interval(
            jx.lease_for(wall_clock_seconds=wall_clock_seconds)
        )
    if not isinstance(heartbeat_seconds, int) or heartbeat_seconds <= 0:
        heartbeat_seconds = 60

    repo = Path(spec.repo_path)
    branch = branch_name(envelope)
    root = Path(workspace_root)
    worktree = root / f"{envelope.job_id}-{envelope.attempt_id}"
    handoff = root / f"{envelope.job_id}-{envelope.attempt_id}.handoff"

    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        handoff.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        os.chmod(handoff, 0o700)
    except OSError as exc:
        raise AdapterError(f"cannot prepare workspace under {root}: {exc}")

    proc = _git(
        repo, "worktree", "add", "-b", branch, str(worktree), spec.base_commit,
        check=False,
    )
    if proc.returncode != 0:
        raise AdapterError(
            f"cannot create worktree for {envelope.attempt_id} on {branch}"
        )

    goal_path = handoff / "goal.txt"
    goal_path.write_bytes(envelope.goal.encode("utf-8"))
    os.chmod(goal_path, 0o600)
    result_path = handoff / "result.json"
    log_path = handoff / "output.log"

    if not _SCHEMA_FILE.is_file():
        raise AdapterError("the Codex result schema is missing from this release")
    schema_path = str(_SCHEMA_FILE)

    command = [
        "codex", "exec",
        "--model", spec.model,
        "--sandbox", "workspace-write",
        "--cd", str(worktree),
        "--json",
        "--output-schema", schema_path,
        "--output-last-message", str(result_path),
        "-",
    ]

    try:
        private = _private_env_root(handoff)
    except OSError as exc:
        raise AdapterError(f"cannot prepare private worker paths: {exc}")

    skills_dir = jskills.attachment_dir(worktree) if plan.attachments else None
    worker_env = _worker_env(
        codex_home=Path(codex_home),
        private=private,
        root=root,
        extra_env=extra_env,
        skills_dir=skills_dir,
    )

    try:
        jskills.stage(worktree, plan.attachments)
    except jskills.SkillAttachmentError as exc:
        raise AdapterError(f"cannot attach skills to {worktree}: {exc}")

    started = time.monotonic()
    exit_code, timed_out = _launch(
        command,
        cwd=worktree,
        env=worker_env,
        log_path=log_path,
        prompt=envelope.goal.encode("utf-8"),
        wall_clock_seconds=wall_clock_seconds,
        heartbeat_seconds=heartbeat_seconds,
        on_heartbeat=on_heartbeat,
    )
    duration = round(time.monotonic() - started, 3)

    stripped = jskills.strip(worktree)

    _sanitize_log(log_path)
    _sanitize_result(result_path)

    reported = _read_result(result_path)
    status, failure_class = jx.classify_worker_result(
        exit_code=exit_code,
        result=reported,
        timed_out=timed_out,
    )

    branch_head = _git_out(repo, "rev-parse", "--verify", f"refs/heads/{branch}")
    worktree_head = _git_out(worktree, "rev-parse", "HEAD")
    worktree_branch = _git_out(worktree, "rev-parse", "--abbrev-ref", "HEAD")

    problems = _evidence_problems(
        repo,
        branch=branch,
        base=spec.base_commit,
        branch_head=branch_head,
        worktree_head=worktree_head,
        worktree_branch=worktree_branch,
    )
    if problems and status == "succeeded":
        status, failure_class = ("failed", "infrastructure")

    commit = None if problems else branch_head
    diff_empty = None
    if commit is not None:
        changed = _git_out(repo, "diff", "--name-only", spec.base_commit, commit)
        if changed is None:
            problems.append(
                f"cannot read the diff between {spec.base_commit} and {commit}"
            )
            if status == "succeeded":
                status, failure_class = ("failed", "infrastructure")
        else:
            paths = changed.splitlines()
            diff_empty = not [p for p in paths if not jskills.is_attachment_path(p)]
            if diff_empty and status == "succeeded":
                status, failure_class = ("failed", "implementation")
                problems.append(
                    "worker reported success but the branch has no change from "
                    f"the approved base {spec.base_commit}"
                )

    receipt = jx.sanitize_receipt(
        {
            "schema_version": 1,
            "kind": "jobs-codex-attempt",
            "job_id": envelope.job_id,
            "attempt_id": envelope.attempt_id,
            "ordinal": envelope.ordinal,
            "specialist": envelope.specialist,
            "executor": ji.CODEX_EXECUTOR,
            "executor_version": _codex_version(),
            "model": spec.model,
            "effort": spec.effort,
            "max_turns": spec.max_turns,
            "repository": str(repo),
            "base_commit": spec.base_commit,
            "branch": branch,
            "worktree": str(worktree),
            "handoff_dir": str(handoff),
            "commit": commit,
            "diff_empty": diff_empty,
            "status": status,
            "failure_class": failure_class,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_seconds": duration,
            "evidence_problems": problems,
            "observed": {
                "branch_head": branch_head,
                "worktree_head": worktree_head,
                "worktree_branch": worktree_branch,
            },
            "worker_reported": {
                "outcome": _as_text(reported, "outcome"),
                "failure_class": _as_text(reported, "failure_class"),
            },
            "output_tail": _read_tail(log_path),
        }
    )

    return AdapterResult(
        status=status,
        failure_class=failure_class,
        repository=str(repo),
        branch=branch,
        worktree=str(worktree),
        commit=commit,
        receipt=receipt,
    )


def _as_text(reported, key) -> Optional[str]:
    if not isinstance(reported, Mapping):
        return None
    value = reported.get(key)
    return value if isinstance(value, str) else None


def _evidence_problems(
    repo: Path,
    *,
    branch: str,
    base: str,
    branch_head: Optional[str],
    worktree_head: Optional[str],
    worktree_branch: Optional[str],
) -> list:
    """Every reason the observed git state cannot back a successful attempt."""
    problems = []
    if branch_head is None:
        problems.append(f"branch {branch} does not exist in the repository")
    if worktree_branch != branch:
        problems.append(
            f"worktree is on branch {worktree_branch!r}, not the attempt's {branch!r}"
        )
    if worktree_head is None:
        problems.append("worktree has no resolvable HEAD")
    elif branch_head is not None and worktree_head != branch_head:
        problems.append(
            f"branch {branch} head {branch_head} does not match worktree HEAD "
            f"{worktree_head}"
        )
    if branch_head is not None:
        ancestor = _git(
            repo, "merge-base", "--is-ancestor", base, branch_head, check=False
        )
        if ancestor.returncode != 0:
            problems.append(
                f"commit {branch_head} does not descend from the approved base {base}"
            )
    return problems
