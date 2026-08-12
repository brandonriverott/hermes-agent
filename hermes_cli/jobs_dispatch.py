"""Provider-free Jobs dispatch decisions.

This module begins at the control-plane choke point that stranded Job 71: the
legacy ``KEY=value`` execution header embedded in a Job's verbatim goal.  The
live shell runner tokenizes values with ``\\S+`` and therefore discards a valid
repository path when it contains spaces.  Parsing lives here so it is a pure,
tested contract rather than transient shell behavior.

The parser remains pure.  The reliability dispatcher below it is the explicit
control-plane seam: it records an immutable preflight before custody, then
advances a signed attempt graph only from read-back evidence.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_execution
from hermes_cli import jobs_executors
from hermes_cli import jobs_graph as graph
from hermes_cli import jobs_harness as harness
from hermes_cli import jobs_handoffs
from hermes_cli import jobs_loop
from hermes_cli import jobs_lanes
from hermes_cli import jobs_receipts
from hermes_cli import jobs_run


LEGACY_KEYS = frozenset(
    {
        "REPO_PATH",
        "BASE_COMMIT",
        "MODEL",
        "EFFORT",
        "MAX_TURNS",
        "WORKSPACE_KIND",
        "NO_REROUTE",
    }
)
EFFORT_TIERS = frozenset({"low", "medium", "high", "xhigh", "max"})
WORKSPACE_KINDS = frozenset({"worktree"})
MAX_TURNS_CEILING = 500

_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_DIGEST_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_HEADER_KEY_RE = re.compile(r"\A[A-Z][A-Z0-9_]*\Z")


class LegacyMetadataError(ValueError):
    """A typed refusal that a dispatcher can persist and display."""

    def __init__(self, code: str, key: str, detail: str):
        super().__init__(detail)
        self.code = str(code)
        self.key = str(key)
        self.detail = str(detail)


@dataclass(frozen=True)
class LegacyExecutionHeader:
    """Validated execution inputs recovered from an existing Job goal."""

    repo_path: str
    base_commit: str
    model: str
    effort: str
    max_turns: int
    workspace_kind: str
    no_reroute: bool
    source_digest: str


def _refuse(code: str, key: str, detail: str) -> None:
    raise LegacyMetadataError(code, key, detail)


def _header_values(goal: str) -> dict[str, str]:
    """Read the contiguous goal header without tokenizing its values."""

    values: dict[str, str] = {}
    started = False

    for raw_line in goal.splitlines():
        if not raw_line.strip():
            if started:
                break
            continue

        raw_key, marker, raw_value = raw_line.partition("=")
        key = raw_key.strip()
        if not marker:
            if started:
                break
            continue

        if key not in LEGACY_KEYS:
            if _HEADER_KEY_RE.fullmatch(key):
                _refuse("unknown_key", "UNKNOWN", "unsupported metadata key")
            if started:
                break
            continue

        started = True
        if key in values:
            _refuse("duplicate_key", key, f"duplicate metadata key {key}")

        value = raw_value.strip()
        if not value:
            _refuse("blank_value", key, f"metadata key {key} is blank")
        values[key] = value

    return values


def _required(values: dict[str, str], key: str) -> str:
    try:
        return values[key]
    except KeyError:
        _refuse("missing_key", key, f"required metadata key {key} is missing")
        raise AssertionError("unreachable")


def _max_turns(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        _refuse("invalid_integer", "MAX_TURNS", "MAX_TURNS must be an integer")
        raise AssertionError("unreachable")
    if value <= 0 or value > MAX_TURNS_CEILING:
        _refuse(
            "out_of_bounds",
            "MAX_TURNS",
            f"MAX_TURNS must be in 1..{MAX_TURNS_CEILING}",
        )
    return value


def _truthy(raw: str) -> bool:
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def parse_legacy_execution_header(goal: str) -> LegacyExecutionHeader:
    """Parse and validate the legacy execution header from ``goal``.

    The complete trimmed right-hand side is the value.  In particular,
    ``REPO_PATH=/Users/brandon/Documents/Kava Bar Scan`` remains that complete
    absolute path rather than being discarded or truncated at the first space.
    The original goal is never rewritten.
    """

    source_goal = str(goal or "")
    values = _header_values(source_goal)

    repo_path = _required(values, "REPO_PATH")
    if not Path(repo_path).is_absolute():
        _refuse(
            "invalid_repo_path",
            "REPO_PATH",
            "REPO_PATH must be absolute",
        )

    base_commit = _required(values, "BASE_COMMIT")
    if not _SHA_RE.fullmatch(base_commit):
        _refuse(
            "invalid_base_commit",
            "BASE_COMMIT",
            "BASE_COMMIT must be a full lowercase 40-character commit SHA",
        )

    effort = values.get("EFFORT", "max")
    if effort not in EFFORT_TIERS:
        _refuse(
            "unsupported_value",
            "EFFORT",
            "unsupported EFFORT value",
        )

    workspace_kind = values.get("WORKSPACE_KIND", "worktree")
    if workspace_kind not in WORKSPACE_KINDS:
        _refuse(
            "unsupported_value",
            "WORKSPACE_KIND",
            "unsupported WORKSPACE_KIND value",
        )

    return LegacyExecutionHeader(
        repo_path=repo_path,
        base_commit=base_commit,
        model=_required(values, "MODEL"),
        effort=effort,
        max_turns=_max_turns(values.get("MAX_TURNS", "120")),
        workspace_kind=workspace_kind,
        no_reroute=_truthy(values.get("NO_REROUTE", "false")),
        source_digest=hashlib.sha256(source_goal.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class DispatchSpec:
    """Immutable, provider-free inputs approved for one dispatch decision."""

    repository: Path
    base_commit: str
    branch: str
    output_parents: tuple[Path, ...]
    scoped_memory_paths: tuple[Path, ...]
    requested_lane: str
    lane_id: str
    executor: str
    specialist: str
    model: str
    worktree_parent: Optional[Path] = None
    lane_root: Optional[Path] = None
    effort: str = "max"
    max_turns: int = 120


@dataclass(frozen=True)
class DispatchContext:
    """Capability-free attempt identity passed to executors and gates."""

    job_id: str
    job_number: int
    attempt_id: str
    ordinal: int
    repository: Path
    base_commit: str
    branch: str
    worktree: Path
    requested_lane: str
    lane_id: str
    executor: str
    specialist: str
    model: str
    job_name: str = "Job"
    goal: str = "unspecified"
    lane_root: Optional[Path] = None
    effort: str = "max"
    max_turns: int = 120
    prior_handoff: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class ActivationDecision:
    """Evidence returned by the existing activation/ship policy seam."""

    status: str
    reason_code: str
    activation_gate_digest: str
    completion_receipt_digest: str
    completion_handoff: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class DispatchResult:
    claimed: bool
    reason: str
    job_id: str
    attempt_id: Optional[str]
    state: Optional[str]
    failure_class: Optional[str] = None
    retry_action: Optional[str] = None
    action_outcome: str = "failed"
    commit: Optional[str] = None
    journal_error: Optional[str] = None


@dataclass(frozen=True)
class LaneCleanupResult:
    status: str
    reason_code: str
    evidence_digest: str


Executor = Callable[[DispatchContext], jobs_execution.ReliabilityExecution]
Gate = Callable[
    [DispatchContext, jobs_execution.ReliabilityExecution], jobs_run.GateEvidence
]
ActivationGate = Callable[
    [DispatchContext, jobs_run.GateEvidence], ActivationDecision
]


def _digest(value: object) -> str:
    return jobs_receipts.digest_bytes(jobs_receipts.canonical_json_bytes(value))


def _execution_spec_digest(spec: DispatchSpec) -> str:
    return _digest(
        {
            "repository": str(spec.repository),
            "base_commit": spec.base_commit,
            "branch": spec.branch,
            "output_parents": [str(path) for path in spec.output_parents],
            "scoped_memory_paths": [
                str(path) for path in spec.scoped_memory_paths
            ],
            "requested_lane": spec.requested_lane,
            "lane_id": spec.lane_id,
            "executor": spec.executor,
            "specialist": spec.specialist,
            "model": spec.model,
            "worktree_parent": (
                None
                if spec.worktree_parent is None
                else str(spec.worktree_parent)
            ),
            "lane_root": None if spec.lane_root is None else str(spec.lane_root),
        }
    )


def _observed_commit(worktree: Path) -> Optional[str]:
    """Read the candidate identity from Git, never from provider narration."""

    try:
        observed = subprocess.run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return observed if _SHA_RE.fullmatch(observed) else None


def _latest_prior_handoff(conn, job_id: str) -> Optional[Mapping[str, object]]:
    """Load only the latest persisted rejection/blocker for correction context."""

    attempts = jdb.get_attempts(conn, job_id)
    for attempt in reversed(attempts):
        handoff = jdb.latest_handoff(conn, str(attempt["id"]))
        if handoff is None or handoff.get("to_phase") not in {"FAILED", "BLOCKED"}:
            continue
        if handoff.get("outcome") not in {"rejected", "blocked"}:
            continue
        return dict(handoff)
    return None


def _handoff_facts(evidence: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {"label": key, "result": "observed", "digest": digest}
        for key, digest in list(evidence.items())[:8]
    ]


def _provider_handoff_value(
    value: object,
    *,
    summary: str,
    next_action: str,
    issues: list[dict[str, str]] | None = None,
) -> tuple[str, str, list[dict[str, str]]]:
    if isinstance(value, Mapping):
        safe_summary = value.get("summary")
        safe_next = value.get("next_action")
        if isinstance(safe_summary, str) and safe_summary.strip():
            summary = safe_summary
        if isinstance(safe_next, str) and safe_next.strip():
            next_action = safe_next
        raw_issues = value.get("issues")
        if issues is None and isinstance(raw_issues, list):
            issues = [
                {
                    "requirement": str(item.get("requirement", "review requirement")),
                    "finding": str(item.get("finding", "unresolved finding")),
                    "required_fix": str(item.get("required_fix", next_action)),
                }
                for item in raw_issues
                if isinstance(item, Mapping)
            ]
    return summary, next_action, issues or []


def _valid_execution_handoff(value: object, *, executor: str, kind: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    provider = executor.capitalize()
    expected = {
        "builder": (f"{provider} Builder", f"{provider} Tester"),
        "tester": (f"{provider} Tester", "Independent Reviewer"),
        "reviewer": ("Independent Reviewer", "Hermes"),
    }[kind]
    if kind == "reviewer" and value.get("verdict") != "PASS":
        expected = (expected[0], f"{provider} Builder")
    if (
        value.get("speaker_role") != expected[0]
        or value.get("speaker_executor") != executor
        or value.get("next_owner_role") != expected[1]
        or not isinstance(value.get("summary"), str)
        or not isinstance(value.get("next_action"), str)
    ):
        return False
    if kind == "reviewer":
        verdict = value.get("verdict")
        issues = value.get("issues")
        if verdict not in {"PASS", "NEEDS_CHANGES", "UNABLE_TO_VERIFY"}:
            return False
        if not isinstance(issues, list):
            return False
        if verdict == "PASS" and issues:
            return False
        if verdict != "PASS" and not issues:
            return False
    return True


def _transition_handoff(
    *,
    job_id: str,
    attempt_id: str,
    initiator_id: str,
    executor: str,
    source: Optional[str],
    target: str,
    evidence: Mapping[str, str],
    commit: str,
    created_at: int,
    provider_handoff: object = None,
    failure_class: Optional[str] = None,
    blocker_code: Optional[str] = None,
) -> dict[str, object]:
    """Create the smallest canonical handoff for one graph edge."""

    provider_name = executor.capitalize()
    if target == "QUEUED":
        role, speaker_executor, owner = "Hermes", "hermes", f"{provider_name} Builder"
        outcome, summary, next_action, issues = (
            "started",
            "Hermes queued the claimed attempt.",
            "Assign the attempt to its selected lane.",
            [],
        )
    elif target == "ASSIGNED":
        role, speaker_executor, owner = "Hermes", "hermes", f"{provider_name} Builder"
        outcome, summary, next_action, issues = (
            "handed_off",
            "Hermes assigned the attempt after preflight and custody evidence.",
            "Start the isolated worktree attempt.",
            [],
        )
    elif target == "BUILDING":
        role, speaker_executor, owner = "Hermes", "hermes", f"{provider_name} Builder"
        outcome, summary, next_action, issues = (
            "started",
            "Hermes started the isolated attempt; this is an internal signed edge.",
            "Collect executor evidence.",
            [],
        )
    elif target == "EVIDENCE_COLLECTING":
        role, speaker_executor, owner = f"{provider_name} Builder", executor, f"{provider_name} Tester"
        outcome, summary, next_action, issues = (
            "handed_off",
            "The builder returned the observed candidate and evidence.",
            "Independently verify the recorded tests and artifacts.",
            [],
        )
        summary, next_action, issues = _provider_handoff_value(
            provider_handoff, summary=summary, next_action=next_action
        )
    elif target == "REVIEWING":
        role, speaker_executor, owner = f"{provider_name} Tester", executor, "Independent Reviewer"
        outcome, summary, next_action, issues = (
            "handed_off",
            "The tester verified the bounded artifact readbacks.",
            "Review the exact observed candidate and evidence.",
            [],
        )
        summary, next_action, issues = _provider_handoff_value(
            provider_handoff, summary=summary, next_action=next_action
        )
    elif target == "VERIFIED":
        role, speaker_executor, owner = "Independent Reviewer", executor, "Hermes"
        outcome, summary, next_action, issues = (
            "passed",
            "The independent reviewer approved the observed candidate.",
            "Evaluate the explicit activation gate.",
            [],
        )
        summary, next_action, issues = _provider_handoff_value(
            provider_handoff, summary=summary, next_action=next_action
        )
    elif target == "BLOCKED":
        role, speaker_executor, owner = "Hermes", "hermes", "Brandon"
        outcome, summary, next_action, issues = (
            "blocked",
            f"Hermes blocked the attempt: {blocker_code or 'evidence required'}.",
            "Choose one of the bounded recovery options.",
            [],
        )
    elif target == "COMPLETED":
        role, speaker_executor, owner = "Hermes", "hermes", None
        outcome, summary, next_action, issues = (
            "completed",
            "Hermes activated the explicitly authorized candidate.",
            "Keep the bounded rollback path available.",
            [],
        )
        summary, next_action, issues = _provider_handoff_value(
            provider_handoff, summary=summary, next_action=next_action
        )
    else:  # FAILED
        role, speaker_executor, owner = "Hermes", "hermes", f"{provider_name} Builder"
        outcome, summary, next_action, issues = (
            "rejected",
            f"Hermes rejected the attempt: {failure_class or 'failure'}.",
            "Correct the recorded failure and submit changed evidence.",
            [{
                "requirement": "Reliability evidence",
                "finding": failure_class or "attempt failed",
                "required_fix": "Provide changed, independently readable evidence.",
            }],
        )
        summary, next_action, provider_issues = _provider_handoff_value(
            provider_handoff, summary=summary, next_action=next_action, issues=issues
        )
        issues = provider_issues or issues
    raw = {
        "summary": summary,
        "evidence_summary": _handoff_facts(evidence),
        "next_action": next_action,
        "issues": issues,
        "decision_request": None,
    }
    if target == "BLOCKED":
        raw["decision_request"] = {
            "question": "How should Hermes recover this blocked attempt?",
            "options": [
                {"id": "human_fix", "label": "Fix the blocker", "consequence": "Resume only after new evidence is recorded."},
                {"id": "leave_blocked", "label": "Keep it blocked", "consequence": "No provider work or completion occurs."},
            ],
            "recommendation": "human_fix",
            "recommendation_reason": "The blocker is bounded and recoverable without guessing.",
            "blocked_scope": "This Jobs attempt only.",
            "safe_state": "No unverified candidate is activated.",
            "next_owner_role": "Brandon",
        }
    artifact = {"kind": "commit", "value": commit}
    return jobs_handoffs.normalize_handoff(
        raw,
        job_id=job_id,
        attempt_id=attempt_id,
        speaker_id=initiator_id,
        speaker_role=role,
        speaker_executor=speaker_executor,
        from_phase="ATTEMPT_CREATED" if source is None and target == "QUEUED" else str(source),
        to_phase=target,
        next_owner_role=owner,
        outcome=outcome,
        artifact_identity=artifact,
        transition_evidence=dict(evidence),
        created_at=created_at,
    )


def resolve_dispatch_executor(
    *,
    job: object,
    spec: DispatchSpec,
    executor_registry: jobs_executors.ExecutorRegistry,
    lane_decision: Optional[jobs_lanes.LaneDecision] = None,
) -> Executor:
    """Validate all persisted/selected identity before selecting a callable."""

    identity = executor_registry.require_job_identity(job)
    expected = (
        identity.requested_lane,
        identity.executor,
        identity.specialist,
        identity.model,
    )
    observed = (
        spec.requested_lane,
        spec.executor,
        spec.specialist,
        spec.model,
    )
    if observed != expected:
        raise jobs_executors.UnsupportedExecutor(
            "dispatch identity contradicts the persisted Job identity"
        )
    if lane_decision is not None:
        policy = jobs_lanes.load_lane_registry()
        jobs_lanes.validate_lane_decision(policy, lane_decision)
        if (
            lane_decision.status != "SELECTED"
            or lane_decision.lane_id != spec.lane_id
            or lane_decision.executor != identity.executor
            or lane_decision.model != identity.model
        ):
            raise jobs_executors.UnsupportedExecutor(
                "selected lane contradicts the persisted Job identity"
            )
    return executor_registry.require_reliability(identity.executor)


def _receipt_id(attempt_id: str, target_state: str) -> str:
    material = _digest(
        {"attempt_id": attempt_id, "target_state": target_state}
    ).removeprefix("sha256:")
    return "r_" + material[:24]


def _preflight_reason(decision: harness.PreflightDecision) -> str:
    for check in decision.checks:
        if check.status != "PASS":
            return check.code
    return "PREFLIGHT_BLOCKED"


def _registered_lane_child(root: Path, category: str, raw: str) -> Path:
    root_resolved = root.resolve(strict=True)
    category_path = root_resolved / category
    raw_path = Path(raw)
    if category_path.is_symlink() or raw_path.is_symlink():
        raise ValueError("lane cleanup category is a symlink")
    category_root = category_path.resolve(strict=True)
    candidate = raw_path.resolve(strict=False)
    if candidate == category_root:
        raise ValueError("lane cleanup target must be attempt-scoped")
    candidate.relative_to(category_root)
    return candidate


def cleanup_lane_attempt(
    *,
    conn,
    placement_id: str,
    repository: Path,
    now: int,
) -> LaneCleanupResult:
    """Remove only registered attempt paths, then release capacity on readback."""

    placement = jdb.get_lane_placement(conn, placement_id)
    if placement is None:
        raise ValueError(f"no such lane placement: {placement_id!r}")
    if placement["state"] != "VERIFYING":
        raise jdb.InvalidTransition("lane cleanup requires VERIFYING placement")
    if not all(
        placement[name]
        for name in (
            "attempt_id",
            "lane_root",
            "registered_worktree",
            "registered_handoff",
        )
    ):
        raise ValueError("lane cleanup requires registered attempt paths")

    root = Path(placement["lane_root"])
    auth_path = root / "auth"
    auth_existed = auth_path.is_dir() and not auth_path.is_symlink()
    try:
        if root.is_symlink():
            raise ValueError("lane root is a symlink")
        if not auth_existed:
            raise ValueError("lane auth boundary is absent or unsafe")
        worktree = _registered_lane_child(
            root, "worktrees", placement["registered_worktree"]
        )
        handoff = _registered_lane_child(
            root, "handoffs", placement["registered_handoff"]
        )
        if worktree.exists() or worktree.is_symlink():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(Path(repository).resolve(strict=True)),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        if handoff.is_symlink():
            handoff.unlink()
        elif handoff.is_dir():
            shutil.rmtree(handoff)
        elif handoff.exists():
            handoff.unlink()
        if worktree.exists() or worktree.is_symlink() or handoff.exists():
            raise OSError("registered cleanup target remains after removal")
        if auth_existed and (not auth_path.is_dir() or auth_path.is_symlink()):
            raise OSError("lane auth boundary changed during cleanup")
        evidence = _digest(
            {
                "placement_id": placement_id,
                "attempt_id": placement["attempt_id"],
                "worktree_absent": True,
                "handoff_absent": True,
                "auth_boundary_unchanged": auth_existed,
            }
        )
        jdb.transition_lane_placement(
            conn,
            placement_id=placement_id,
            target_state="IDLE",
            reason_code="CLEANUP_VERIFIED",
            evidence_digest=evidence,
            now=now,
        )
        return LaneCleanupResult("PASS", "CLEANUP_VERIFIED", evidence)
    except Exception as exc:  # noqa: BLE001 - fail closed and persist no secrets
        evidence = _digest(
            {
                "placement_id": placement_id,
                "attempt_id": placement["attempt_id"],
                "error_type": type(exc).__name__,
            }
        )
        current = jdb.get_lane_placement(conn, placement_id)
        if current is not None and current["state"] == "VERIFYING":
            jdb.transition_lane_placement(
                conn,
                placement_id=placement_id,
                target_state="FAILED",
                reason_code="CLEANUP_FAILED",
                evidence_digest=evidence,
                now=now,
            )
        return LaneCleanupResult("BLOCKED", "CLEANUP_FAILED", evidence)


def dispatch_once(
    *,
    conn,
    job_id: str,
    spec: DispatchSpec,
    probes: harness.PreflightProbes,
    signer: harness.ReceiptSigner,
    verifier: graph.ReceiptVerifier,
    worker_id: str,
    executor_registry: jobs_executors.ExecutorRegistry,
    gate: Gate,
    activation_gate: ActivationGate,
    observed_at: str,
    now: int,
    lease_seconds: int = 1800,
    failure_journal_path: Optional[Path] = None,
    lane_decision: Optional[jobs_lanes.LaneDecision] = None,
) -> DispatchResult:
    """Run one evidence-authorized dispatch without provider assumptions.

    Custody is taken only after the aggregate preflight receipt is durably
    recorded as PASS.  The claim token remains in this frame and is never part
    of :class:`DispatchContext`.
    """

    job = jdb.get_job(conn, job_id)
    if job is None:
        raise ValueError(f"no such job: {job_id!r}")
    if _SHA_RE.fullmatch(spec.base_commit) is None:
        raise ValueError("dispatch base_commit must be a full lowercase SHA")
    selected_executor = resolve_dispatch_executor(
        job=job,
        spec=spec,
        executor_registry=executor_registry,
        lane_decision=lane_decision,
    )
    identity = executor_registry.require_job_identity(job)
    if lane_decision is not None:
        if spec.lane_root is None:
            raise ValueError("lane-aware dispatch requires lane_root")
        lane_root = Path(spec.lane_root)
        if lane_root.is_symlink() or not lane_root.is_dir():
            raise ValueError("lane_root must be an existing non-symlink directory")
        if Path(spec.worktree_parent or "").resolve(strict=False) != (
            lane_root / "worktrees"
        ).resolve(strict=True):
            raise ValueError("lane worktree_parent must be lane_root/worktrees")

    snapshot = harness.PreflightSnapshot(
        job_id=job.id,
        execution_spec_digest=_execution_spec_digest(spec),
        repository=Path(spec.repository),
        base_commit=spec.base_commit,
        branch=spec.branch,
        output_parents=tuple(Path(path) for path in spec.output_parents),
        scoped_memory_paths=tuple(
            Path(path) for path in spec.scoped_memory_paths
        ),
        lane_id=spec.lane_id,
        executor=spec.executor,
        model=spec.model,
        observed_at=observed_at,
        expected_job_revision=job.revision,
    )
    preflight = harness.preflight_and_record(conn, snapshot, probes, signer)
    if preflight.status != "PASS":
        return DispatchResult(
            claimed=False,
            reason=_preflight_reason(preflight),
            job_id=job.id,
            attempt_id=None,
            state=None,
            failure_class=preflight.failure_class,
        )

    placement_id: Optional[str] = None
    if lane_decision is None:
        claim = jdb.claim_job(
            conn,
            worker=worker_id,
            specialist=job.specialist,
            job=job.id,
            lease_seconds=lease_seconds,
            now=now,
        )
        if claim is None:
            return DispatchResult(
                claimed=False,
                reason="NOTHING_ELIGIBLE",
                job_id=job.id,
                attempt_id=None,
                state=None,
            )
        claim_token = claim.claim_token
    else:
        placed = jdb.claim_selected_lane(
            conn,
            job_id=job.id,
            decision=lane_decision,
            preflight_id=preflight.preflight_id,
            expected_revision=job.revision,
            lease_seconds=lease_seconds,
            now=now,
        )
        if not placed.claimed:
            return DispatchResult(
                claimed=False,
                reason=placed.reason_code,
                job_id=job.id,
                attempt_id=None,
                state=None,
            )
        claim_token = placed.claim_token
        placement_id = placed.placement_id
        if claim_token is None or placement_id is None:
            raise jdb.GraphConflict("claimed lane placement lacks custody identity")
    ordinal = len(jdb.get_attempts(conn, job.id)) + 1
    worktree_parent = (
        Path(spec.worktree_parent)
        if spec.worktree_parent is not None
        else (
            Path(spec.output_parents[0])
            if spec.output_parents
            else Path(spec.repository).parent / ".hermes-jobs-worktrees"
        )
    )
    planned_worktree = worktree_parent / f"{job.id}-{ordinal}"
    try:
        attempt_id = jdb.start_attempt(
            conn,
            job.id,
            claim_token=claim_token,
            specialist=identity.specialist,
            repository=str(spec.repository),
            base_commit=spec.base_commit,
            branch=spec.branch,
            worktree=str(planned_worktree),
            now=now,
        )
        if placement_id is not None:
            lane_root = Path(spec.lane_root)
            jdb.bind_lane_attempt(
                conn,
                placement_id=placement_id,
                attempt_id=attempt_id,
                lane_root=lane_root,
                worktree=planned_worktree,
                handoff=lane_root / "handoffs" / attempt_id,
                now=now,
            )
    except Exception:
        if placement_id is not None:
            placement = jdb.get_lane_placement(conn, placement_id)
            if placement is not None and placement["state"] == "ASSIGNED":
                jdb.transition_lane_placement(
                    conn,
                    placement_id=placement_id,
                    target_state="FAILED",
                    reason_code="ATTEMPT_START_FAILED",
                    now=now,
                )
        jdb.release_claim(
            conn, job.id, claim_token=claim_token, now=now
        )
        raise
    attempt = jdb.get_attempt(conn, attempt_id)
    context = DispatchContext(
        job_id=job.id,
        job_number=job.number,
        attempt_id=attempt_id,
        ordinal=int(attempt["ordinal"]),
        repository=Path(spec.repository),
        base_commit=spec.base_commit,
        branch=spec.branch,
        worktree=planned_worktree,
        requested_lane=identity.requested_lane,
        lane_id=spec.lane_id,
        executor=identity.executor,
        specialist=identity.specialist,
        model=identity.model,
        job_name=job.name,
        goal=job.goal,
        lane_root=None if spec.lane_root is None else Path(spec.lane_root),
        effort=spec.effort,
        max_turns=spec.max_turns,
        prior_handoff=_latest_prior_handoff(conn, job.id),
    )
    sequence = 0

    def advance(
        target_state: str,
        evidence: Mapping[str, str],
        *,
        commit: str,
        failure_class: Optional[str] = None,
        blocker_code: Optional[str] = None,
        provider_handoff: object = None,
    ) -> graph.TransitionRecord:
        nonlocal sequence
        latest = jdb.latest_transition(conn, attempt_id)
        source_state = None if latest is None else latest["target_state"]
        sequence += 1
        receipt_id = _receipt_id(attempt_id, target_state)
        request = graph.TransitionRequest(
            job_id=job.id,
            attempt_id=attempt_id,
            source_state=source_state,
            target_state=target_state,
            initiator_type="system",
            initiator_id=worker_id,
            expected_job_revision=jdb.get_job(conn, job.id).revision,
            evidence=dict(evidence),
            failure_class=failure_class,
            blocker_code=blocker_code,
            commit=commit,
            receipt_id=receipt_id,
            component="jobs_dispatch",
            component_version="1",
            idempotency_key=(
                f"dispatch:{source_state or 'NONE'}:{target_state}"
            ),
            created_at=int(now) + sequence,
            handoff=_transition_handoff(
                job_id=job.id,
                attempt_id=attempt_id,
                initiator_id=worker_id,
                executor=identity.executor,
                source=source_state,
                target=target_state,
                evidence=evidence,
                commit=commit,
                created_at=int(now) + sequence,
                provider_handoff=provider_handoff,
                failure_class=failure_class,
                blocker_code=blocker_code,
            ),
        )
        envelope = signer.sign(
            {
                "job_id": job.id,
                "attempt_id": attempt_id,
                "commit": commit,
                "state": target_state,
                "evidence": dict(evidence),
                "handoff": request.handoff,
                "timestamp": observed_at,
                "transitioned_by": "jobs_dispatch/1",
            },
            receipt_id=receipt_id,
        )
        return graph.transition_attempt(
            conn, request, envelope=envelope, verifier=verifier
        )

    def fail(
        signal: jobs_loop.FailureSignal,
        *,
        evidence_digest: str,
        commit: str,
        force_blocked: bool = False,
        provider_handoff: object = None,
        journal_error_override: Optional[str] = None,
    ) -> DispatchResult:
        decision = jobs_loop.classify_failure(signal)
        reported_reason = decision.reason_code
        retry_action: Optional[str] = None
        recovery = decision.recovery_decision
        target = "FAILED"
        blocker_code: Optional[str] = None
        user_blocker_codes = {
            "AUTH_REQUIRED",
            "CREDENTIALS_EXPIRED",
            "MISSING_LANE_LOGIN",
            "SSH_AUTH_FAILED",
            "TOKEN_EXPIRED",
            "UNAUTHORIZED",
            "APPROVAL_REQUIRED",
            "IRREVERSIBLE_ACTION",
            "SAFETY_GATE",
        }
        if (
            decision.failure_class in {"AUTH_INFRA", "SAFETY_GATE"}
            and decision.reason_code in user_blocker_codes
        ):
            target = "BLOCKED"
            retry_action = "HUMAN_ACTION"
            blocker_code = decision.reason_code
        elif decision.failure_class in {"PROVIDER", "INFRA_FAILURE"}:
            attempts = jdb.get_attempts(conn, job.id)
            chain_id = attempts[0]["id"]
            history_rows = jdb.list_retry_evidence(
                conn, job.id, chain_id=chain_id
            )
            retry = jobs_loop.decide_retry(
                [
                    jobs_loop.RetryRecord(
                        ordinal=int(row["ordinal"]),
                        evidence_digest=str(row["evidence_digest"]),
                        decision=str(row["decision"]),
                    )
                    for row in history_rows
                ],
                evidence_digest=evidence_digest,
                failure_class=decision.failure_class,
                prior_failure_digest=(
                    str(history_rows[-1]["evidence_digest"])
                    if history_rows
                    else evidence_digest
                ),
                changed_evidence_digest=evidence_digest,
                what_changed=(
                    "first observed failure"
                    if not history_rows
                    else "new attempt evidence digest"
                ),
            )
            stored = jdb.record_retry_decision(
                conn,
                jdb.RetryEvidenceWrite(
                    job_id=job.id,
                    chain_id=chain_id,
                    attempt_id=attempt_id,
                    parent_attempt_id=attempt["parent_attempt_id"],
                    ordinal=int(attempt["ordinal"]),
                    evidence_digest=evidence_digest,
                    decision=retry.action,
                    reason_code=retry.reason_code,
                    created_at=int(now) + sequence + 1,
                ),
            )
            retry_action = str(stored["decision"])
            recovery = retry_action
            if retry_action != "RETRY":
                target = "BLOCKED"
                blocker_code = str(stored["reason_code"])
                reported_reason = blocker_code
        if force_blocked:
            # Only bounded evidence blockers may surface as a user decision;
            # unknown codes remain diagnostic failures, never user-facing BLOCKED.
            if decision.reason_code in {
                "READBACK_MISSING",
                "READBACK_FAILED",
                "READBACK_SIZE_MISMATCH",
                "READBACK_DIGEST_MISMATCH",
                "DUPLICATE_ARTIFACT_CLAIM",
                "TEST_EVIDENCE_MISSING",
                "HANDOFF_INCOMPLETE",
            }:
                target = "BLOCKED"
                retry_action = retry_action or "HUMAN_ACTION"
                recovery = retry_action
                blocker_code = "EVIDENCE_REQUIRED"
            else:
                target = "FAILED"
                blocker_code = None

        if placement_id is not None:
            placement = jdb.get_lane_placement(conn, placement_id)
            if placement is not None and placement["state"] in {
                "ASSIGNED",
                "BUILDING",
                "VERIFYING",
            }:
                jdb.transition_lane_placement(
                    conn,
                    placement_id=placement_id,
                    target_state="FAILED",
                    reason_code=decision.reason_code,
                    evidence_digest=evidence_digest,
                    now=int(now) + sequence,
                )
        advance(
            target,
            {"failure": evidence_digest},
            commit=commit,
            failure_class=decision.failure_class,
            blocker_code=blocker_code,
            provider_handoff=(provider_handoff if provider_handoff is not None else {
                "summary": f"Failure evidence is {evidence_digest}.",
                "next_action": "Resolve the blocker and provide changed evidence.",
            }),
        )
        journal_error = journal_error_override
        try:
            if journal_error is None:
                jobs_loop.append_failure_journal(
                jobs_loop.FailureJournalRecord(
                    job_id=job.id,
                    attempt_id=attempt_id,
                    failure_class=decision.failure_class,
                    reason_code=decision.reason_code,
                    recovery_decision=recovery,
                    evidence_digests=(evidence_digest,),
                    observed_at=int(now) + sequence,
                    component_versions={"jobs_dispatch": "1"},
                ),
                path=failure_journal_path,
                )
        except (OSError, ValueError) as exc:
            journal_error = type(exc).__name__
        return DispatchResult(
            claimed=True,
            reason=(
                "JOURNAL_WRITE_FAILED"
                if journal_error is not None
                else reported_reason
            ),
            job_id=job.id,
            attempt_id=attempt_id,
            state=target,
            failure_class=(
                "INFRA_FAILURE"
                if journal_error is not None
                else decision.failure_class
            ),
            retry_action=retry_action,
            commit=commit,
            journal_error=journal_error,
        )

    advance(
        "QUEUED",
        {"attempt_created": _digest({"attempt_id": attempt_id})},
        commit=spec.base_commit,
    )
    advance(
        "ASSIGNED",
        {
            "preflight": _digest(
                {
                    "preflight_id": preflight.preflight_id,
                    "receipt_id": preflight.receipt_id,
                }
            ),
            "route": _digest(
                {
                    "lane_id": spec.lane_id,
                    "executor": spec.executor,
                    "model": spec.model,
                    "placement_id": placement_id,
                }
            ),
            "lane_health": _digest(
                (
                    {
                        "health_id": jdb.get_lane_placement(
                            conn, placement_id
                        )["health_id"],
                        "placement_id": placement_id,
                    }
                    if placement_id is not None
                    else [check.evidence_digest for check in preflight.checks]
                )
            ),
            "claim": _digest({"job_id": job.id, "worker_id": worker_id}),
            "worktree": _digest(
                {"path": str(planned_worktree), "branch": spec.branch}
            ),
            "attempt_started": _digest(
                {"attempt_id": attempt_id, "ordinal": int(attempt["ordinal"])}
            ),
        },
        commit=spec.base_commit,
    )
    advance(
        "BUILDING",
        {
            "claim": _digest(
                {"job_id": job.id, "worker_id": worker_id, "now": int(now)}
            ),
            "worktree": _digest(
                {"path": str(planned_worktree), "branch": spec.branch}
            ),
            "attempt_started": _digest(
                {"attempt_id": attempt_id, "ordinal": int(attempt["ordinal"])}
            ),
        },
        commit=spec.base_commit,
    )
    if placement_id is not None:
        jdb.transition_lane_placement(
            conn,
            placement_id=placement_id,
            target_state="BUILDING",
            reason_code="EXECUTOR_STARTED",
            now=int(now) + sequence + 1,
        )

    try:
        execution = selected_executor(context)
    except Exception as exc:  # noqa: BLE001 - the attempt must become evidence
        evidence_digest = _digest(
            {"stage": "executor", "exception": type(exc).__name__}
        )
        return fail(
            jobs_loop.FailureSignal(
                reason_code="PROCESS_CRASHED", stage="executor"
            ),
            evidence_digest=evidence_digest,
            commit=spec.base_commit,
        )
    if not isinstance(execution, jobs_execution.ReliabilityExecution):
        return fail(
            jobs_loop.FailureSignal(
                reason_code="PROCESS_CRASHED", stage="executor"
            ),
            evidence_digest=_digest(
                {"stage": "executor", "result": "invalid"}
            ),
            commit=spec.base_commit,
        )
    observed_candidate = _observed_commit(execution.worktree)
    candidate = observed_candidate or execution.commit or spec.base_commit
    for kind, value in (
        ("builder", execution.builder_handoff),
        ("tester", execution.tester_handoff),
        ("reviewer", execution.reviewer_handoff),
    ):
        if value is not None and not _valid_execution_handoff(
            value, executor=identity.executor, kind=kind
        ):
            return fail(
                jobs_loop.FailureSignal(
                    reason_code="HANDOFF_INCOMPLETE", stage="handoff"
                ),
                evidence_digest=_digest({"malformed": f"{kind}_handoff"}),
                commit=candidate if _SHA_RE.fullmatch(candidate) else spec.base_commit,
                force_blocked=True,
            )
    if execution.status != "succeeded":
        return fail(
            jobs_loop.FailureSignal(
                http_status=execution.http_status,
                reason_code=execution.failure_reason_code,
                safety_gate=execution.safety_gate,
                stage="executor",
            ),
            evidence_digest=_digest(
                {
                    "executor_exit": execution.executor_exit_digest,
                    "output_capture": execution.output_capture_digest,
                }
            ),
            commit=candidate,
            provider_handoff=execution.reviewer_handoff or execution.builder_handoff,
        )
    if execution.builder_handoff is None:
        return fail(
            jobs_loop.FailureSignal(
                reason_code="HANDOFF_INCOMPLETE", stage="handoff"
            ),
            evidence_digest=_digest({"missing": "builder_handoff"}),
            commit=spec.base_commit,
            force_blocked=True,
        )
    if observed_candidate is None:
        return fail(
            jobs_loop.FailureSignal(
                reason_code="NO_CANDIDATE_COMMIT", stage="executor"
            ),
            evidence_digest=_digest({"observed_commit": None}),
            commit=spec.base_commit,
        )
    if execution.commit not in {None, observed_candidate}:
        return fail(
            jobs_loop.FailureSignal(
                reason_code="NO_CANDIDATE_COMMIT", stage="executor"
            ),
            evidence_digest=_digest(
                {
                    "reported_commit": execution.commit,
                    "observed_commit": observed_candidate,
                }
            ),
            commit=spec.base_commit,
        )
    if _SHA_RE.fullmatch(candidate) is None or candidate == spec.base_commit:
        return fail(
            jobs_loop.FailureSignal(
                reason_code="NO_CANDIDATE_COMMIT", stage="executor"
            ),
            evidence_digest=_digest(
                {"commit": candidate, "base_commit": spec.base_commit}
            ),
            commit=spec.base_commit,
        )

    advance(
        "EVIDENCE_COLLECTING",
        {
            "executor_exit": execution.executor_exit_digest,
            "output_capture": execution.output_capture_digest,
            "builder_handoff": _digest(execution.builder_handoff),
        },
        commit=candidate,
        provider_handoff=execution.builder_handoff,
    )
    if not execution.artifacts:
        return fail(
            jobs_loop.FailureSignal(
                reason_code="READBACK_MISSING", stage="readback"
            ),
            evidence_digest=_digest({"artifacts": []}),
            commit=candidate,
            force_blocked=True,
        )
    readbacks = []
    seen_names = set()
    for claim_item in execution.artifacts:
        if claim_item.name in seen_names:
            return fail(
                jobs_loop.FailureSignal(
                    reason_code="DUPLICATE_ARTIFACT_CLAIM", stage="readback"
                ),
                evidence_digest=_digest({"name": claim_item.name}),
                commit=candidate,
                force_blocked=True,
            )
        seen_names.add(claim_item.name)
        readback = jobs_loop.verify_readback(
            claim_item.path,
            expected_digest=claim_item.digest,
            expected_size=claim_item.size,
        )
        readbacks.append(readback)
        if readback.status != "PASS":
            return fail(
                jobs_loop.FailureSignal(
                    reason_code=readback.reason_code, stage="readback"
                ),
                evidence_digest=_digest(
                    {
                        "name": claim_item.name,
                        "expected": claim_item.digest,
                        "observed": readback.digest,
                        "reason": readback.reason_code,
                    }
                ),
                commit=candidate,
                force_blocked=True,
            )
    tests_claim = next(
        (item for item in execution.artifacts if item.name == "tests"), None
    )
    if tests_claim is None:
        return fail(
            jobs_loop.FailureSignal(
                reason_code="TEST_EVIDENCE_MISSING", stage="testing"
            ),
            evidence_digest=_digest(
                {"artifact_names": sorted(seen_names)}
            ),
            commit=candidate,
            force_blocked=True,
        )
    if execution.tester_handoff is None:
        return fail(
            jobs_loop.FailureSignal(
                reason_code="HANDOFF_INCOMPLETE", stage="handoff"
            ),
            evidence_digest=_digest({"missing": "tester_handoff"}),
            commit=candidate,
            force_blocked=True,
        )
    advance(
        "REVIEWING",
        {
            "tests": tests_claim.digest,
            "readback": _digest(
                [
                    {
                        "path": item.path,
                        "size": item.size,
                        "digest": item.digest,
                        "status": item.status,
                    }
                    for item in readbacks
                ]
            ),
            "tester_handoff": _digest(execution.tester_handoff),
        },
        commit=candidate,
        provider_handoff=execution.tester_handoff,
    )

    try:
        gate_evidence = gate(context, execution)
    except Exception as exc:  # noqa: BLE001 - gate refusal has no graph authority
        return DispatchResult(
            claimed=True,
            reason="GATE_EVIDENCE_INVALID",
            job_id=job.id,
            attempt_id=attempt_id,
            state="REVIEWING",
            failure_class="INFRA_FAILURE",
            commit=candidate,
            journal_error=type(exc).__name__,
        )
    if (
        not isinstance(gate_evidence, jobs_run.GateEvidence)
        or gate_evidence.identity_verified is not True
    ):
        reason = (
            gate_evidence.reason_code
            if isinstance(gate_evidence, jobs_run.GateEvidence)
            else "GATE_EVIDENCE_INVALID"
        )
        return DispatchResult(
            claimed=True,
            reason=reason,
            job_id=job.id,
            attempt_id=attempt_id,
            state="REVIEWING",
            failure_class="INFRA_FAILURE",
            commit=candidate,
        )
    if gate_evidence.action_outcome != "succeeded":
        return fail(
            jobs_loop.FailureSignal(
                reason_code=gate_evidence.reason_code, stage="review"
            ),
            evidence_digest=_digest(
                {
                    "gate": gate_evidence.reason_code,
                    "commit": gate_evidence.commit,
                }
            ),
            commit=candidate,
            force_blocked=False,
            provider_handoff=execution.reviewer_handoff,
        )
    if (
        gate_evidence.commit != candidate
        or gate_evidence.themis_review_digest is None
        or gate_evidence.receipt_verification_digest is None
        or not _DIGEST_RE.fullmatch(str(gate_evidence.themis_review_digest))
        or not _DIGEST_RE.fullmatch(
            str(gate_evidence.receipt_verification_digest)
        )
    ):
        return fail(
            jobs_loop.FailureSignal(
                reason_code="GATE_IDENTITY_MISMATCH", stage="review"
            ),
            evidence_digest=_digest(
                {
                    "gate_commit": gate_evidence.commit,
                    "candidate": candidate,
                    "themis_review": gate_evidence.themis_review_digest,
                    "receipt_verification": gate_evidence.receipt_verification_digest,
                }
            ),
            commit=candidate,
            provider_handoff=execution.reviewer_handoff,
        )
    if execution.reviewer_handoff is None:
        return fail(
            jobs_loop.FailureSignal(
                reason_code="HANDOFF_INCOMPLETE", stage="handoff"
            ),
            evidence_digest=_digest({"missing": "reviewer_handoff"}),
            commit=candidate,
            force_blocked=True,
        )
    if execution.reviewer_handoff.get("verdict") != "PASS":
        verdict = str(execution.reviewer_handoff.get("verdict"))
        return fail(
            jobs_loop.FailureSignal(
                reason_code=(
                    "REVIEW_NEEDS_CHANGES"
                    if verdict == "NEEDS_CHANGES"
                    else "REVIEW_UNABLE_TO_VERIFY"
                ),
                stage="review",
            ),
            evidence_digest=_digest(
                {
                    "reviewer_handoff": execution.reviewer_handoff,
                    "verdict": verdict,
                }
            ),
            commit=candidate,
            provider_handoff=execution.reviewer_handoff,
        )
    # Activation is evaluated while still REVIEWING so any callback refusal
    # can settle a signed terminal edge without leaving VERIFIED custody.
    try:
        activation = activation_gate(context, gate_evidence)
    except Exception as exc:  # noqa: BLE001 - preserve callback identity
        return fail(
            jobs_loop.FailureSignal(
                reason_code="ACTIVATION_GATE_EXCEPTION", safety_gate=True
            ),
            evidence_digest=_digest(
                {"activation": "exception", "type": type(exc).__name__}
            ),
            commit=candidate,
            force_blocked=True,
            journal_error_override=type(exc).__name__,
        )
    if not isinstance(activation, ActivationDecision):
        return fail(
            jobs_loop.FailureSignal(
                reason_code="ACTIVATION_GATE_INVALID", safety_gate=True
            ),
            evidence_digest=_digest({"activation": "invalid"}),
            commit=candidate,
            force_blocked=True,
        )
    if activation.status != "PASS":
        return fail(
            jobs_loop.FailureSignal(
                reason_code=activation.reason_code, safety_gate=True
            ),
            evidence_digest=activation.activation_gate_digest,
            commit=candidate,
            force_blocked=True,
        )
    required_completion_fields = {
        "summary",
        "next_action",
        "observed_candidate",
        "environment",
        "gate_digest",
        "rollback",
    }
    if (
        not isinstance(activation.completion_handoff, Mapping)
        or not required_completion_fields.issubset(activation.completion_handoff)
        or any(
            not isinstance(activation.completion_handoff.get(field), str)
            or not activation.completion_handoff.get(field).strip()
            or len(activation.completion_handoff.get(field)) > 700
            for field in required_completion_fields
        )
        or _SHA_RE.fullmatch(
            str(activation.completion_handoff.get("observed_candidate"))
        ) is None
        or not _DIGEST_RE.fullmatch(
            str(activation.completion_handoff.get("gate_digest"))
        )
        or str(activation.completion_handoff.get("observed_candidate")) != candidate
        or not _DIGEST_RE.fullmatch(str(activation.activation_gate_digest))
        or not _DIGEST_RE.fullmatch(str(activation.completion_receipt_digest))
    ):
        return fail(
            jobs_loop.FailureSignal(
                reason_code="ACTIVATION_HANDOFF_INCOMPLETE", safety_gate=True
            ),
            evidence_digest=_digest({"activation": "completion_handoff_invalid"}),
            commit=candidate,
            force_blocked=True,
        )
    advance(
        "VERIFIED",
        {
            "themis_review": gate_evidence.themis_review_digest,
            "receipt_verification": (
                gate_evidence.receipt_verification_digest
            ),
            "reviewer_handoff": _digest(execution.reviewer_handoff),
        },
        commit=candidate,
        provider_handoff=execution.reviewer_handoff,
    )

    if placement_id is not None:
        sequence += 1
        jdb.transition_lane_placement(
            conn,
            placement_id=placement_id,
            target_state="VERIFYING",
            reason_code="EVIDENCE_VERIFIED",
            now=int(now) + sequence,
        )
        sequence += 1
        cleanup = cleanup_lane_attempt(
            conn=conn,
            placement_id=placement_id,
            repository=spec.repository,
            now=int(now) + sequence,
        )
        if cleanup.status != "PASS":
            advance(
                "FAILED",
                {"failure": cleanup.evidence_digest},
                commit=candidate,
                failure_class="INFRA_FAILURE",
            )
            return DispatchResult(
                claimed=True,
                reason=cleanup.reason_code,
                job_id=job.id,
                attempt_id=attempt_id,
                state="FAILED",
                failure_class="INFRA_FAILURE",
                retry_action="HUMAN_ACTION",
                commit=candidate,
            )

    advance(
        "COMPLETED",
        {
            "activation_gate": activation.activation_gate_digest,
            "completion_receipt": activation.completion_receipt_digest,
            "completion_handoff": _digest(activation.completion_handoff),
        },
        commit=candidate,
        provider_handoff=activation.completion_handoff,
    )
    return DispatchResult(
        claimed=True,
        reason="COMPLETED",
        job_id=job.id,
        attempt_id=attempt_id,
        state="COMPLETED",
        action_outcome="succeeded",
        commit=candidate,
    )


__all__ = [
    "ActivationDecision",
    "DispatchContext",
    "DispatchResult",
    "DispatchSpec",
    "LaneCleanupResult",
    "LegacyExecutionHeader",
    "LegacyMetadataError",
    "dispatch_once",
    "cleanup_lane_attempt",
    "parse_legacy_execution_header",
]
