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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_cli import jobs_adapter_claude as adapter
from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_graph as graph
from hermes_cli import jobs_harness as harness
from hermes_cli import jobs_loop
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
                _refuse("unknown_key", key, f"unsupported metadata key {key}")
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
            f"REPO_PATH must be absolute: {repo_path!r}",
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
            f"unsupported EFFORT value {effort!r}",
        )

    workspace_kind = values.get("WORKSPACE_KIND", "worktree")
    if workspace_kind not in WORKSPACE_KINDS:
        _refuse(
            "unsupported_value",
            "WORKSPACE_KIND",
            f"unsupported WORKSPACE_KIND value {workspace_kind!r}",
        )

    return LegacyExecutionHeader(
        repo_path=repo_path,
        base_commit=base_commit,
        model=values.get("MODEL", "claude-opus-5"),
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
    lane_id: str
    executor: str
    model: str
    worktree_parent: Optional[Path] = None


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
    lane_id: str
    executor: str
    model: str


@dataclass(frozen=True)
class ActivationDecision:
    """Evidence returned by the existing activation/ship policy seam."""

    status: str
    reason_code: str
    activation_gate_digest: str
    completion_receipt_digest: str


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


Executor = Callable[[DispatchContext], adapter.ReliabilityExecution]
Gate = Callable[
    [DispatchContext, adapter.ReliabilityExecution], jobs_run.GateEvidence
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
            "lane_id": spec.lane_id,
            "executor": spec.executor,
            "model": spec.model,
            "worktree_parent": (
                None
                if spec.worktree_parent is None
                else str(spec.worktree_parent)
            ),
        }
    )


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


def dispatch_once(
    *,
    conn,
    job_id: str,
    spec: DispatchSpec,
    probes: harness.PreflightProbes,
    signer: harness.ReceiptSigner,
    verifier: graph.ReceiptVerifier,
    worker_id: str,
    executor: Executor,
    gate: Gate,
    activation_gate: ActivationGate,
    observed_at: str,
    now: int,
    lease_seconds: int = 1800,
    failure_journal_path: Optional[Path] = None,
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
            specialist=job.specialist or spec.executor,
            repository=str(spec.repository),
            base_commit=spec.base_commit,
            branch=spec.branch,
            worktree=str(planned_worktree),
            now=now,
        )
    except Exception:
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
        lane_id=spec.lane_id,
        executor=spec.executor,
        model=spec.model,
    )
    sequence = 0

    def advance(
        target_state: str,
        evidence: Mapping[str, str],
        *,
        commit: str,
        failure_class: Optional[str] = None,
        blocker_code: Optional[str] = None,
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
        )
        envelope = signer.sign(
            {
                "job_id": job.id,
                "attempt_id": attempt_id,
                "commit": commit,
                "state": target_state,
                "evidence": dict(evidence),
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
    ) -> DispatchResult:
        decision = jobs_loop.classify_failure(signal)
        retry_action: Optional[str] = None
        recovery = decision.recovery_decision
        target = "FAILED"
        blocker_code: Optional[str] = None
        if decision.failure_class in {"AUTH_INFRA", "SAFETY_GATE"}:
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
        if force_blocked:
            target = "BLOCKED"
            retry_action = retry_action or "HUMAN_ACTION"
            recovery = retry_action
            blocker_code = decision.reason_code

        advance(
            target,
            {"failure": evidence_digest},
            commit=commit,
            failure_class=decision.failure_class,
            blocker_code=blocker_code,
        )
        journal_error = None
        try:
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
                else decision.reason_code
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
                }
            ),
            "lane_health": _digest(
                [check.evidence_digest for check in preflight.checks]
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

    try:
        execution = executor(context)
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
    if not isinstance(execution, adapter.ReliabilityExecution):
        return fail(
            jobs_loop.FailureSignal(
                reason_code="PROCESS_CRASHED", stage="executor"
            ),
            evidence_digest=_digest(
                {"stage": "executor", "result": "invalid"}
            ),
            commit=spec.base_commit,
        )
    candidate = execution.commit or spec.base_commit
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
        },
        commit=candidate,
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
        },
        commit=candidate,
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
        )
    if (
        gate_evidence.commit != candidate
        or gate_evidence.themis_review_digest is None
        or gate_evidence.receipt_verification_digest is None
    ):
        return DispatchResult(
            claimed=True,
            reason="GATE_IDENTITY_MISMATCH",
            job_id=job.id,
            attempt_id=attempt_id,
            state="REVIEWING",
            failure_class="INFRA_FAILURE",
            commit=candidate,
        )
    advance(
        "VERIFIED",
        {
            "themis_review": gate_evidence.themis_review_digest,
            "receipt_verification": (
                gate_evidence.receipt_verification_digest
            ),
        },
        commit=candidate,
    )

    activation = activation_gate(context, gate_evidence)
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
    advance(
        "COMPLETED",
        {
            "activation_gate": activation.activation_gate_digest,
            "completion_receipt": activation.completion_receipt_digest,
        },
        commit=candidate,
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
    "LegacyExecutionHeader",
    "LegacyMetadataError",
    "dispatch_once",
    "parse_legacy_execution_header",
]
