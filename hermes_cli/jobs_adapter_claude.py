"""Claude execution adapter — one Job envelope in, one classified result out.

The adapter is the only place in Jobs V3 that touches the outside world: it
creates an isolated Git worktree at the approved base commit, launches an
injected worker executable inside it, and reads back what actually happened. It
deliberately does **not** touch the Jobs store. It returns an
:class:`AdapterResult`; the orchestrator (:mod:`hermes_cli.jobs_run`) is the only
thing that writes attempts, receipts, or custody. Keeping the mutation authority
out of here is what makes "the adapter crashed" a recoverable state rather than a
half-written one.

Three properties are load-bearing:

- **Isolation.** Work happens in `git worktree` at ``metadata.execution``'s exact
  base commit, on a branch named for the attempt. The repository's own working
  tree, HEAD, and index are never touched, so a run cannot disturb whatever the
  operator has checked out.
- **Evidence is observed, never accepted.** The commit and branch that end up on
  the attempt come from asking Git, not from the worker's self-report. A worker
  that leaves its branch, or lands a commit with no ancestry to the approved
  base, produces evidence problems and loses its success.
- **Nothing sensitive travels.** The claim token is never passed to the worker in
  argv or environment, the worker gets an environment allowlist rather than the
  parent's inheritance, and the receipt is redacted and size-capped before it
  leaves this module.
- **Lent material is never delivered.** Skill attachments (see
  :mod:`hermes_cli.jobs_skills`) are staged into the worktree read-only, removed
  again before the first piece of evidence is read, and discounted from the diff
  if a builder committed them anyway.

ponytail: worktrees are kept after the run, success or failure. They are the
cheapest possible forensic artifact, they cost one directory, and the branch is
durable in the repository either way. Prune them from the workspace root on a
schedule if they ever add up.
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
from hermes_cli import jobs_receipts
from hermes_cli import jobs_skills as jskills
from hermes_cli.jobs_contract import JobEnvelope

# Only these variables are inherited. Everything else the worker needs is
# *constructed* below — inheriting the parent environment would hand a builder
# every credential this process happens to hold, and inheriting HOME or
# HERMES_HOME specifically would point it at the operator's live Jobs store.
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM")

# Built per attempt, under the attempt's own private root, and never inherited.
_PRIVATE_ENV = ("HOME", "HERMES_HOME", "TMPDIR")

# Every variable this module constructs per attempt. A caller may not supply one:
# each names something about *this* attempt's isolation, so accepting an override
# would let an "extra" re-point the containment the allowlist exists to build.
_CONSTRUCTED_ENV = _PRIVATE_ENV + (jskills.SKILLS_ENV,)

# The receipt keeps a tail, not a transcript. Read a little more than the
# redactor will keep so truncation happens on a boundary we control.
_OUTPUT_TAIL_BYTES = 4000

# A result file is a handful of JSON fields; a huge one is a worker misbehaving.
_MAX_RESULT_BYTES = 64 * 1024

# What a preserved worker file is allowed to weigh once the run is over. The
# worker writes its stdout straight to disk (a pipe would deadlock the parent),
# so the file is unbounded and unredacted *while* the run happens; this is the
# ceiling it is rewritten to before anything is preserved.
MAX_PRESERVED_BYTES = 64 * 1024

_GIT_TIMEOUT = 120


class AdapterError(RuntimeError):
    """Preflight refused: the repository or base identity is not usable.

    Raised before any attempt work begins, so the caller can hand custody back
    without having burned an attempt on a workspace that could never exist.
    """


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


@dataclass(frozen=True)
class ArtifactClaim:
    """One executor-authored byte claim for dispatcher readback.

    The executor captures the digest immediately after writing.  The dispatcher
    reads the path independently before REVIEWING, so a later rewrite is
    observable rather than silently becoming the new expected value.
    """

    name: str
    path: Path
    digest: str
    size: int


@dataclass(frozen=True)
class ReliabilityExecution:
    """Provider-neutral evidence returned to the reliability dispatcher."""

    status: str
    commit: Optional[str]
    worktree: Path
    artifacts: tuple[ArtifactClaim, ...]
    executor_exit_digest: str
    output_capture_digest: str
    failure_reason_code: Optional[str] = None
    http_status: Optional[int] = None
    safety_gate: bool = False


def claim_artifact(path: Path, *, name: str) -> ArtifactClaim:
    """Capture a bounded identity claim without granting it authority."""

    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("artifact name must not be empty")
    source = Path(path)
    data = source.read_bytes()
    return ArtifactClaim(
        name=clean_name,
        path=source,
        digest=jobs_receipts.digest_bytes(data),
        size=len(data),
    )


def branch_name(envelope: JobEnvelope) -> str:
    """The branch this attempt owns: stable, unique, and capability-free.

    Keyed on the attempt id, so a retry of the same Job never reuses a branch and
    two attempts can never race for one ref.
    """
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


def preflight(spec: jx.ExecutionSpec) -> None:
    """Refuse now if the repository or the exact base commit is not real.

    Checked before an attempt is started, because "the repo moved" and "that sha
    is not in this repository" are operator problems, not build failures, and
    recording them as a failed attempt would misattribute them.
    """
    repo = Path(spec.repo_path)
    if not repo.is_dir():
        raise AdapterError(f"repository path is not a directory: {repo}")
    if _git_out(repo, "rev-parse", "--git-dir") is None:
        raise AdapterError(f"repository path is not a git repository: {repo}")
    # ``^{commit}`` also rejects a sha that resolves to a tree or a blob.
    proc = _git(repo, "cat-file", "-e", f"{spec.base_commit}^{{commit}}", check=False)
    if proc.returncode != 0:
        raise AdapterError(
            f"base commit {spec.base_commit} is not a commit in {repo}"
        )


def preflight_skills(skills, *, name: str, goal: str) -> jskills.SkillPlan:
    """Decide — and prove — this attempt's skill attachments. Reads, never writes.

    Public and separate from :func:`run_claude_attempt` so the orchestrator can
    refuse a Job whose *declared* skills cannot be honoured while it still holds
    nothing but a claim: no attempt started, no worktree built, no provider paid.
    A declaration that traverses, that resolves outside every store this machine
    keeps skills in, or that names nothing on it raises :class:`AdapterError`
    here.

    ``skills=None`` is unset, and the Job's own words then select the defaults; a
    default this machine does not have is recorded, not raised.

    The adapter calls this again on the way in. Re-resolving costs a few file
    reads and buys something worth more than that: the bytes that get attached
    are the bytes that were just hashed, rather than a copy of a copy.
    """
    try:
        return jskills.plan(skills, text=f"{name}\n{goal}")
    except jskills.SkillAttachmentError as exc:
        raise AdapterError(f"skill attachment refused: {exc}")


# ---------------------------------------------------------------------------
# Worker launch
# ---------------------------------------------------------------------------


def _private_env_root(handoff: Path):
    """Create this attempt's private ``(HOME, HERMES_HOME, TMPDIR)`` directories.

    Three owner-only directories inside the attempt's own handoff root. They are
    created rather than borrowed because a worker that inherits the operator's
    ``HOME`` reads the operator's credentials, and one that inherits
    ``HERMES_HOME`` resolves — and can write — the live Jobs database it is
    supposed to be running under the supervision of.
    """
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


def _worker_env(
    extra_env: Optional[Mapping[str, str]],
    *,
    private: Sequence[Path],
    root: Path,
    skills_dir: Optional[Path] = None,
) -> dict:
    """The worker's complete environment: an allowlist plus a private identity.

    Every private path is re-checked against the attempt's root here, immediately
    before the launch that would use it. A path that does not resolve inside the
    root — a symlink out, a caller-supplied override, a root that moved — raises
    :class:`AdapterError`, and it raises *before* the worker exists rather than
    after it has already read somebody's credentials.

    ``skills_dir`` is the one *additive* fact this attempt tells the worker: where
    its read-only reference material is. It rides in the environment because the
    other two channels are closed — the goal is stored verbatim and must not be
    rewritten to mention it, and the worker's argv is a fixed contract.
    """
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    if "PATH" not in env:
        raise AdapterError("no PATH is available to give the worker")
    for name, path in zip(_PRIVATE_ENV, private):
        if not _contained(path, root):
            raise AdapterError(
                f"{name} for this attempt does not resolve inside its private "
                f"root {root}"
            )
        env[name] = str(path.resolve())
    for key, value in (extra_env or {}).items():
        # An "extra" is for a config path or a feature flag. Refuse to let it
        # become the hole the allowlist exists to close — including by
        # re-pointing the isolation this function just built.
        jx.reject_secret_key(key)
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
    """Replace ``path`` with ``body``, owner-only, leaving no other copy behind.

    The temporary carries the *cleaned* bytes and ``os.replace`` unlinks the
    original in the same step, so at no point does a second file hold the raw
    material this is scrubbing.
    """
    tmp = path.with_name(path.name + ".sanitizing")
    try:
        with open(
            os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb"
        ) as fh:
            fh.write(body)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    except OSError:
        # The scrub failed, so the original is still raw. Losing the evidence
        # beats preserving material that was never allowed to be preserved.
        for victim in (tmp, path):
            try:
                victim.unlink()
            except OSError:
                pass


def _bounded_text(path: Path) -> Optional[str]:
    """The last :data:`MAX_PRESERVED_BYTES` of ``path`` as valid UTF-8, or ``None``.

    A tail can start mid-character *and* mid-secret, so the first partial line is
    dropped: half of a token no longer matches the pattern that would have
    redacted the whole of it, and a half-token is still a leak.
    """
    marker = f"{jx.TRUNCATED}\n"
    budget = MAX_PRESERVED_BYTES - len(marker.encode("utf-8"))
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > budget:
                fh.seek(size - budget)
            raw = fh.read(budget)
    except OSError:
        return None
    text = raw.decode("utf-8", "replace")
    if size > budget:
        _, sep, rest = text.partition("\n")
        text = marker + (rest if sep else "")
    return text


def _capped_utf8(text: str) -> bytes:
    """``text`` as at most :data:`MAX_PRESERVED_BYTES` of valid UTF-8, tail kept."""
    body = text.encode("utf-8")
    if len(body) <= MAX_PRESERVED_BYTES:
        return body
    body = body[-MAX_PRESERVED_BYTES:]
    # Redaction can lengthen a line ("[redacted]" is longer than some of what it
    # replaces), so the recut tail may start mid-character. Drop the fragment.
    while body and (body[0] & 0xC0) == 0x80:
        body = body[1:]
    return body


def _sanitize_log(path: Path) -> None:
    """Rewrite a worker's raw stdout capture as bounded, redacted, owner-only."""
    text = _bounded_text(path)
    if text is None:
        return
    _write_private(path, _capped_utf8(jx.redact_secrets(text)))


def _sanitize_result(path: Path) -> None:
    """Rewrite the worker's result file as redacted standard JSON.

    Parsed and re-emitted rather than string-scrubbed, so the preserved file is
    the same JSON the receipt was classified from — and so a secret hidden in a
    nested field is reached by the same recursive policy receipts use.
    """
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
    # Not JSON (or absurdly large): keep it as scrubbed, bounded text rather than
    # deleting the one artifact that might explain what the worker was doing.
    _sanitize_log(path)


def _launch(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict,
    log_path: Path,
    wall_clock_seconds: int,
    heartbeat_seconds: int,
    on_heartbeat: Optional[Callable[[], None]],
):
    """Run the worker to completion, beating on cadence. Returns (rc, timed_out).

    No threads: the wait itself is the timer. ``proc.wait(timeout=…)`` wakes at
    whichever comes first — the next heartbeat or the wall clock — so custody
    upkeep costs one event per interval and nothing per poll. Output goes to a
    file rather than a pipe so a chatty worker can never fill a buffer and
    deadlock the parent that is supposed to be timing it out.
    """
    with log_path.open("wb") as log:
        try:
            proc = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Its own session: a timeout kills the worker *and* anything it
                # spawned, instead of orphaning children onto the workspace.
                start_new_session=True,
            )
        except (OSError, ValueError):
            return (None, False)

    start = time.monotonic()
    deadline = start + wall_clock_seconds
    next_beat = start + heartbeat_seconds
    while True:
        now = time.monotonic()
        if now >= deadline:
            _kill(proc)
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
                        # Custody upkeep is not the result. A lost lease is
                        # caught by the finish call, which fails closed itself;
                        # killing a live worker over it would waste the run.
                        pass
                next_beat = time.monotonic() + heartbeat_seconds


def _kill(proc: subprocess.Popen) -> None:
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


def run_claude_attempt(
    envelope: JobEnvelope,
    spec: jx.ExecutionSpec,
    *,
    worker_command: Sequence[str],
    workspace_root,
    wall_clock_seconds: int,
    heartbeat_seconds: Optional[int] = None,
    on_heartbeat: Optional[Callable[[], None]] = None,
    extra_env: Optional[Mapping[str, str]] = None,
) -> AdapterResult:
    """Execute one attempt in an isolated worktree and report what happened.

    Raises :class:`AdapterError` only for conditions that mean the attempt could
    never start (bad repository, bad base, unusable workspace). Everything that
    happens *after* the worktree exists — a crash, a timeout, a lie about the
    outcome, unusable git evidence — comes back as a terminal
    :class:`AdapterResult`, because those are results, not adapter faults.
    """
    if not worker_command:
        raise AdapterError("no worker command was injected")
    # Before the worktree and before the launch: a Job whose declared skills
    # cannot be honoured is refused while this has still cost nothing.
    plan = preflight_skills(envelope.skills, name=envelope.name, goal=envelope.goal)
    preflight(spec)

    if heartbeat_seconds is None:
        heartbeat_seconds = jx.heartbeat_interval(
            jx.lease_for(wall_clock_seconds=wall_clock_seconds)
        )

    repo = Path(spec.repo_path)
    branch = branch_name(envelope)
    root = Path(workspace_root)
    worktree = root / f"{envelope.job_id}-{envelope.attempt_id}"
    handoff = root / f"{envelope.job_id}-{envelope.attempt_id}.handoff"

    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        handoff.mkdir(parents=True, exist_ok=True, mode=0o700)
        # mkdir's mode is masked by the umask; chmod is not. The goal text lives
        # in the handoff directory, so owner-only is a requirement, not a hint.
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

    # The goal lives beside the worktree, never inside it: a spec file dropped
    # into the tree would show up as an uncommitted change, or worse, get
    # committed as part of the work.
    goal_path = handoff / "goal.txt"
    goal_path.write_bytes(envelope.goal.encode("utf-8"))
    os.chmod(goal_path, 0o600)
    result_path = handoff / "result.json"
    log_path = handoff / "output.log"

    command = [
        *worker_command,
        "--worktree", str(worktree),
        "--goal-file", str(goal_path),
        "--result-file", str(result_path),
        "--model", spec.model,
        "--effort", spec.effort,
        "--max-turns", str(spec.max_turns),
    ]

    # Built and containment-checked before the launch: an unsafe private root is
    # a refusal, never a worker that runs with the operator's identity.
    try:
        private = _private_env_root(handoff)
    except OSError as exc:
        raise AdapterError(f"cannot prepare private worker paths: {exc}")
    skills_dir = jskills.attachment_dir(worktree) if plan.attachments else None
    worker_env = _worker_env(
        extra_env, private=private, root=root, skills_dir=skills_dir
    )

    # Reference material goes in last, once everything that could still refuse
    # has refused, and it is the first thing taken out again after the run.
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
        wall_clock_seconds=wall_clock_seconds,
        heartbeat_seconds=heartbeat_seconds,
        on_heartbeat=on_heartbeat,
    )
    duration = round(time.monotonic() - started, 3)

    # Before a single piece of evidence below is read, on every path — success,
    # failure, and timeout alike. Reference material that survives into the diff
    # stops being reference material and becomes work nobody did.
    stripped = jskills.strip(worktree)

    # Before anything is read from them and before they become the run's durable
    # forensic artifact: the worker wrote these files, so their contents are
    # untrusted and unbounded until this point.
    _sanitize_log(log_path)
    _sanitize_result(result_path)

    reported = _read_result(result_path)
    status, failure_class = jx.classify_worker_result(
        exit_code=exit_code, result=reported, timed_out=timed_out
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
        # A success nobody can corroborate is not a success.
        status, failure_class = ("failed", "infrastructure")

    commit = None if problems else branch_head
    diff_empty = None
    committed_attachments: list = []
    if commit is not None:
        changed = _git_out(repo, "diff", "--name-only", spec.base_commit, commit)
        if changed is None:
            problems.append(f"cannot read the diff between {spec.base_commit} and {commit}")
            if status == "succeeded":
                status, failure_class = ("failed", "infrastructure")
        else:
            # A builder that ran ``git add -A`` can commit the attachments it was
            # lent. They are still not work: they are discounted here, so a run
            # whose only change is borrowed reference material reads as the empty
            # diff it really is, and the fact is kept in the receipt.
            paths = changed.splitlines()
            committed_attachments = [p for p in paths if jskills.is_attachment_path(p)]
            diff_empty = not [p for p in paths if not jskills.is_attachment_path(p)]
            if diff_empty and status == "succeeded":
                # The worker exited zero and said it worked, and the repository
                # says nothing happened. The Git evidence is intact here — there
                # is simply no work in it — so this is the builder failing at the
                # job, not the infrastructure failing at the builder.
                status, failure_class = ("failed", "implementation")
                problems.append(
                    "worker reported success but the branch has no change from "
                    f"the approved base {spec.base_commit}"
                )

    receipt = jx.sanitize_receipt(
        {
            "schema_version": 1,
            "kind": "jobs-claude-attempt",
            "job_id": envelope.job_id,
            "attempt_id": envelope.attempt_id,
            "ordinal": envelope.ordinal,
            "specialist": envelope.specialist,
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
            # What reference material this attempt ran with, and proof of which
            # bytes it was. ``committed`` is normally empty; anything in it is a
            # builder that committed material it was only lent.
            "skills": {
                "source": plan.source,
                "requested": list(plan.requested),
                "attached": [
                    {"name": a.name, "sha256": a.sha256, "bytes": a.size}
                    for a in plan.attachments
                ],
                "unavailable": list(plan.unavailable),
                "stripped": stripped,
                "committed": committed_attachments,
            },
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
            # Only the two fields the contract defines are copied out of the
            # worker's file. Echoing the whole thing would let a worker choose
            # what lands in durable evidence.
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
