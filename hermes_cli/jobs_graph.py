"""Evidence-authorized, append-only attempt graph for the Jobs ledger."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from hermes_cli import jobs_db
from hermes_cli import jobs_receipts


GraphState: TypeAlias = Literal[
    "QUEUED",
    "ASSIGNED",
    "BUILDING",
    "EVIDENCE_COLLECTING",
    "REVIEWING",
    "VERIFIED",
    "COMPLETED",
    "FAILED",
    "BLOCKED",
    "CANCELLED",
]

LEGAL = {
    None: frozenset({"QUEUED"}),
    "QUEUED": frozenset({"ASSIGNED", "FAILED", "BLOCKED", "CANCELLED"}),
    "ASSIGNED": frozenset({"BUILDING", "FAILED", "BLOCKED", "CANCELLED"}),
    "BUILDING": frozenset(
        {"EVIDENCE_COLLECTING", "FAILED", "BLOCKED", "CANCELLED"}
    ),
    "EVIDENCE_COLLECTING": frozenset(
        {"REVIEWING", "FAILED", "BLOCKED", "CANCELLED"}
    ),
    "REVIEWING": frozenset({"VERIFIED", "FAILED", "BLOCKED", "CANCELLED"}),
    "VERIFIED": frozenset({"COMPLETED", "FAILED", "BLOCKED", "CANCELLED"}),
    "COMPLETED": frozenset(),
    "FAILED": frozenset(),
    "BLOCKED": frozenset(),
    "CANCELLED": frozenset(),
}

REQUIRED_EVIDENCE = {
    (None, "QUEUED"): frozenset({"attempt_created"}),
    ("QUEUED", "ASSIGNED"): frozenset({"preflight", "route", "lane_health"}),
    ("ASSIGNED", "BUILDING"): frozenset({"claim", "worktree", "attempt_started"}),
    ("BUILDING", "EVIDENCE_COLLECTING"): frozenset(
        {"executor_exit", "output_capture"}
    ),
    ("EVIDENCE_COLLECTING", "REVIEWING"): frozenset({"tests", "readback"}),
    ("REVIEWING", "VERIFIED"): frozenset(
        {"themis_review", "receipt_verification"}
    ),
    ("VERIFIED", "COMPLETED"): frozenset(
        {"activation_gate", "completion_receipt"}
    ),
}

TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "BLOCKED", "CANCELLED"})
_DIGEST = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")


class IllegalGraphTransition(ValueError):
    """Raised when a requested state edge is not legal."""


class MissingTransitionEvidence(ValueError):
    """Raised when an otherwise legal edge lacks its required evidence."""


StaleJobRevision = jobs_db.StaleJobRevision


@dataclass(frozen=True)
class TransitionRequest:
    job_id: str
    attempt_id: str
    source_state: GraphState | None
    target_state: GraphState
    initiator_type: str
    initiator_id: str
    expected_job_revision: int
    evidence: Mapping[str, str]
    failure_class: str | None
    blocker_code: str | None
    commit: str
    receipt_id: str
    component: str
    component_version: str
    idempotency_key: str
    created_at: int


@dataclass(frozen=True)
class TransitionRecord:
    id: int
    job_id: str
    attempt_id: str
    source_state: GraphState | None
    target_state: GraphState
    initiator_type: str
    initiator_id: str
    expected_job_revision: int
    evidence: Mapping[str, str]
    failure_class: str | None
    blocker_code: str | None
    receipt_id: str
    component: str
    component_version: str
    idempotency_key: str
    created_at: int


@dataclass(frozen=True)
class ReceiptVerifier:
    trusted_keys: Mapping[str, Ed25519PublicKey]
    revoked_key_ids: Collection[str] = ()

    def verify(
        self,
        envelope: Mapping[str, object],
        *,
        expected: Mapping[str, str],
    ) -> None:
        jobs_receipts.verify_receipt(
            envelope,
            trusted_keys=self.trusted_keys,
            expected=expected,
            revoked_key_ids=self.revoked_key_ids,
        )


def _write(request: TransitionRequest) -> jobs_db.TransitionWrite:
    return jobs_db.TransitionWrite(
        job_id=request.job_id,
        attempt_id=request.attempt_id,
        source_state=request.source_state,
        target_state=request.target_state,
        initiator_type=request.initiator_type,
        initiator_id=request.initiator_id,
        expected_job_revision=request.expected_job_revision,
        evidence=dict(request.evidence),
        failure_class=request.failure_class,
        blocker_code=request.blocker_code,
        receipt_id=request.receipt_id,
        component=request.component,
        component_version=request.component_version,
        idempotency_key=request.idempotency_key,
        created_at=request.created_at,
        commit=request.commit,
    )


def _record(row: Mapping[str, object]) -> TransitionRecord:
    return TransitionRecord(
        id=int(row["id"]),
        job_id=str(row["job_id"]),
        attempt_id=str(row["attempt_id"]),
        source_state=row["source_state"],
        target_state=row["target_state"],
        initiator_type=str(row["initiator_type"]),
        initiator_id=str(row["initiator_id"]),
        expected_job_revision=int(row["expected_job_revision"]),
        evidence=dict(row["evidence"]),
        failure_class=row["failure_class"],
        blocker_code=row["blocker_code"],
        receipt_id=str(row["receipt_id"]),
        component=str(row["component"]),
        component_version=str(row["component_version"]),
        idempotency_key=str(row["idempotency_key"]),
        created_at=int(row["created_at"]),
    )


def _load_record(conn, transition_id: int) -> TransitionRecord:
    row = conn.execute(
        "SELECT * FROM job_attempt_transitions WHERE id = ?", (transition_id,)
    ).fetchone()
    if row is None:
        raise jobs_db.GraphConflict("accepted transition row disappeared")
    parsed = dict(row)
    parsed["evidence"] = jobs_receipts.loads_canonical(parsed.pop("evidence_json"))
    return _record(parsed)


def _validate_evidence(request: TransitionRequest) -> None:
    evidence = dict(request.evidence)
    if not all(
        isinstance(name, str)
        and name
        and isinstance(digest, str)
        and _DIGEST.fullmatch(digest)
        for name, digest in evidence.items()
    ):
        raise MissingTransitionEvidence("transition evidence must be named sha256 digests")
    required = REQUIRED_EVIDENCE.get(
        (request.source_state, request.target_state), frozenset()
    )
    missing = required - set(evidence)
    if missing:
        raise MissingTransitionEvidence(
            "missing transition evidence: " + ", ".join(sorted(missing))
        )
    if request.target_state in {"FAILED", "BLOCKED", "CANCELLED"}:
        if not evidence:
            raise MissingTransitionEvidence("terminal transition requires evidence")
        if request.target_state == "FAILED" and not request.failure_class:
            raise MissingTransitionEvidence("FAILED requires a failure class")
        if request.target_state in {"BLOCKED", "CANCELLED"} and not request.blocker_code:
            raise MissingTransitionEvidence(
                f"{request.target_state} requires a blocker or cancellation reason"
            )


def _verify_envelope(
    request: TransitionRequest,
    envelope: Mapping[str, object],
    verifier: ReceiptVerifier,
) -> None:
    verifier.verify(
        envelope,
        expected={
            "job_id": request.job_id,
            "attempt_id": request.attempt_id,
            "commit": request.commit,
            "state": request.target_state,
        },
    )
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        raise jobs_receipts.ReceiptVerificationError("receipt payload is missing")
    if payload.get("evidence") != dict(request.evidence):
        raise jobs_receipts.ReceiptVerificationError(
            "receipt evidence identity mismatch"
        )
    if payload.get("transitioned_by") != (
        f"{request.component}/{request.component_version}"
    ):
        raise jobs_receipts.ReceiptVerificationError(
            "receipt component identity mismatch"
        )


def transition_attempt(
    conn,
    request: TransitionRequest,
    *,
    envelope: Mapping[str, object],
    verifier: ReceiptVerifier,
) -> TransitionRecord:
    """Validate and append one graph edge, receipt, event, and public revision."""

    job = jobs_db.get_job(conn, request.job_id)
    if job is None:
        raise ValueError(f"no such job: {request.job_id!r}")
    attempt = jobs_db.get_attempt(conn, request.attempt_id)
    if attempt is None:
        raise ValueError(f"no such attempt: {request.attempt_id!r}")
    if attempt["job_id"] != request.job_id:
        raise ValueError("attempt belongs to a different job")
    if _COMMIT.fullmatch(request.commit) is None:
        raise ValueError("transition commit must be a full lowercase SHA")
    if attempt["commit"] is not None and attempt["commit"] != request.commit:
        raise jobs_db.GraphConflict("transition commit contradicts attempt identity")

    existing = conn.execute(
        "SELECT id FROM job_attempt_transitions "
        "WHERE attempt_id = ? AND idempotency_key = ?",
        (request.attempt_id, request.idempotency_key),
    ).fetchone()
    if existing is not None:
        transition_id = jobs_db.record_transition(conn, _write(request), envelope)
        _verify_envelope(request, envelope, verifier)
        return _load_record(conn, transition_id)

    if job.revision != request.expected_job_revision:
        raise StaleJobRevision(
            f"job {job.id!r} revision is {job.revision}, "
            f"expected {request.expected_job_revision}"
        )
    latest = jobs_db.latest_transition(conn, request.attempt_id)
    current_state = None if latest is None else latest["target_state"]
    if request.source_state != current_state:
        raise IllegalGraphTransition(
            f"source state {request.source_state!r} does not match {current_state!r}"
        )
    if request.target_state not in LEGAL.get(current_state, frozenset()):
        raise IllegalGraphTransition(
            f"illegal Job graph transition {current_state!r} -> {request.target_state!r}"
        )
    if attempt["status"] != "running":
        raise jobs_db.GraphConflict("new graph edge requires a running attempt")
    _validate_evidence(request)
    _verify_envelope(request, envelope, verifier)
    transition_id = jobs_db.record_transition(conn, _write(request), envelope)
    return _load_record(conn, transition_id)


def _projection_transition(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "id": row["id"],
        "source_state": row["source_state"],
        "target_state": row["target_state"],
        "evidence": dict(row["evidence"]),
        "failure_class": row["failure_class"],
        "blocker_code": row["blocker_code"],
        "receipt_id": row["receipt_id"],
        "component": row["component"],
        "component_version": row["component_version"],
        "created_at": row["created_at"],
    }


def compute_work_control(conn, *, now: int) -> dict[str, object]:
    """Return a deterministic, read-only projection of every attempt path."""

    projected_jobs = []
    for job in jobs_db.list_jobs(conn):
        projected_attempts = []
        attempts = jobs_db.get_attempts(conn, job.id)
        for attempt in attempts:
            transitions = jobs_db.list_transitions(conn, job.id)
            attempt_transitions = [
                row for row in transitions if row["attempt_id"] == attempt["id"]
            ]
            projected_attempts.append(
                {
                    "attempt_id": attempt["id"],
                    "ordinal": attempt["ordinal"],
                    "status": attempt["status"],
                    "state": (
                        attempt_transitions[-1]["target_state"]
                        if attempt_transitions
                        else "QUEUED"
                    ),
                    "transitions": [
                        _projection_transition(row) for row in attempt_transitions
                    ],
                }
            )
        state = projected_attempts[-1]["state"] if projected_attempts else "QUEUED"
        projected_jobs.append(
            {
                "job_id": job.id,
                "number": job.number,
                "name": job.name,
                "status": job.status,
                "step": job.step,
                "revision": job.revision,
                "state": state,
                "attempts": projected_attempts,
            }
        )
    return {"schema_version": 1, "generated_at": int(now), "jobs": projected_jobs}


__all__ = [
    "GraphState",
    "IllegalGraphTransition",
    "LEGAL",
    "MissingTransitionEvidence",
    "REQUIRED_EVIDENCE",
    "ReceiptVerifier",
    "StaleJobRevision",
    "TransitionRecord",
    "TransitionRequest",
    "compute_work_control",
    "transition_attempt",
]
