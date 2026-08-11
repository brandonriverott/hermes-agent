"""Read-only weekly reliability scorecard for the Jobs control plane."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
import json

from hermes_cli import jobs_graph


@dataclass(frozen=True)
class JobObservation:
    job_id: str
    started: bool
    settled: bool
    outcome_verified: bool = False
    false_complete: bool = False
    false_blocked: bool = False
    retry_without_progress: int = 0
    missing_evidence: int = 0
    post_green_failure: int = 0
    failure_signatures: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReliabilityScorecard:
    period_start: int
    period_end: int
    jobs_started: int
    jobs_settled: int
    outcomes_verified: int
    verification_rate: float
    false_complete: int
    false_blocked: int
    retry_without_progress: int
    missing_evidence: int
    post_green_failures: int
    guard_candidates: tuple[tuple[str, int], ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "period_start": self.period_start,
            "period_end": self.period_end,
            "jobs_started": self.jobs_started,
            "jobs_settled": self.jobs_settled,
            "outcomes_verified": self.outcomes_verified,
            "verification_rate": self.verification_rate,
            "false_complete": self.false_complete,
            "false_blocked": self.false_blocked,
            "retry_without_progress": self.retry_without_progress,
            "missing_evidence": self.missing_evidence,
            "post_green_failures": self.post_green_failures,
            "guard_candidates": [
                {"signature": signature, "count": count}
                for signature, count in self.guard_candidates
            ],
        }


def build_scorecard(
    observations: Iterable[JobObservation],
    *,
    period_start: int,
    period_end: int,
    guard_threshold: int = 2,
) -> ReliabilityScorecard:
    if period_end <= period_start:
        raise ValueError("period_end must be after period_start")
    if guard_threshold <= 0:
        raise ValueError("guard_threshold must be positive")
    rows = tuple(observations)
    started = sum(row.started for row in rows)
    settled = sum(row.settled for row in rows)
    verified = sum(row.outcome_verified and row.settled for row in rows)
    signatures = Counter(
        signature for row in rows for signature in set(row.failure_signatures)
    )
    candidates = tuple(
        sorted(
            (
                (signature, count)
                for signature, count in signatures.items()
                if count >= guard_threshold
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return ReliabilityScorecard(
        period_start=period_start,
        period_end=period_end,
        jobs_started=started,
        jobs_settled=settled,
        outcomes_verified=verified,
        verification_rate=(verified / settled if settled else 0.0),
        false_complete=sum(row.false_complete for row in rows),
        false_blocked=sum(row.false_blocked for row in rows),
        retry_without_progress=sum(row.retry_without_progress for row in rows),
        missing_evidence=sum(row.missing_evidence for row in rows),
        post_green_failures=sum(row.post_green_failure for row in rows),
        guard_candidates=candidates,
    )


def observations_from_connection(
    conn, *, period_start: int, period_end: int
) -> tuple[JobObservation, ...]:
    """Derive reliability observations from append-only Jobs evidence."""

    if period_end <= period_start:
        raise ValueError("period_end must be after period_start")
    job_ids = {
        str(row[0])
        for row in conn.execute(
            "SELECT id FROM jobs WHERE created_at >= ? AND created_at < ? "
            "UNION SELECT job_id FROM job_attempts WHERE started_at >= ? AND started_at < ? "
            "UNION SELECT job_id FROM job_attempt_transitions WHERE created_at >= ? AND created_at < ? "
            "UNION SELECT job_id FROM job_retry_evidence WHERE created_at >= ? AND created_at < ?",
            (period_start, period_end) * 4,
        )
    }
    retry_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(job_retry_evidence)")
    }
    observations: list[JobObservation] = []
    for job_id in sorted(job_ids):
        job = conn.execute(
            "SELECT created_at FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        attempts_started = conn.execute(
            "SELECT COUNT(*) FROM job_attempts WHERE job_id = ? "
            "AND started_at >= ? AND started_at < ?",
            (job_id, period_start, period_end),
        ).fetchone()[0]
        transitions = list(
            conn.execute(
                "SELECT * FROM job_attempt_transitions WHERE job_id = ? "
                "AND created_at < ? ORDER BY id",
                (job_id, period_end),
            )
        )
        period_transitions = [
            row for row in transitions if int(row["created_at"]) >= period_start
        ]
        latest_state = str(transitions[-1]["target_state"]) if transitions else None
        completed = [
            row for row in period_transitions if row["target_state"] == "COMPLETED"
        ]
        verified_by_attempt: dict[str, list[int]] = {}
        code_verified_by_attempt: dict[str, list[int]] = {}
        for row in transitions:
            attempt_id = str(row["attempt_id"])
            if row["target_state"] == "OUTCOME_VERIFIED":
                verified_by_attempt.setdefault(attempt_id, []).append(int(row["id"]))
            if row["target_state"] == "VERIFIED":
                code_verified_by_attempt.setdefault(attempt_id, []).append(int(row["id"]))

        false_complete = any(
            not any(
                verified_id < int(row["id"])
                for verified_id in verified_by_attempt.get(str(row["attempt_id"]), [])
            )
            for row in completed
        )
        incidents = []
        for event in conn.execute(
            "SELECT data FROM job_events WHERE job_id = ? AND kind = ? "
            "AND created_at >= ? AND created_at < ?",
            (job_id, "reliability_incident", period_start, period_end),
        ):
            try:
                incident = json.loads(event["data"] or "{}")
            except (TypeError, ValueError):
                continue
            if isinstance(incident, dict):
                incidents.append(incident)
        false_blocked = any(
            incident.get("classification") == "FALSE_BLOCKED"
            for incident in incidents
        )

        missing_evidence = 0
        signatures: set[str] = set()
        signatures.update(
            str(incident["signature"])
            for incident in incidents
            if isinstance(incident.get("signature"), str)
        )
        post_green_attempts: set[str] = set()
        for row in period_transitions:
            try:
                evidence = json.loads(row["evidence_json"])
            except (TypeError, ValueError):
                evidence = None
            required = jobs_graph.REQUIRED_EVIDENCE.get(
                (row["source_state"], row["target_state"]), frozenset()
            )
            if not isinstance(evidence, dict):
                missing_evidence += max(1, len(required))
            else:
                missing_evidence += len(required - set(evidence))
            if row["blocker_code"]:
                signatures.add(str(row["blocker_code"]))
            elif row["failure_class"]:
                signatures.add(str(row["failure_class"]))
            if row["target_state"] in {"FAILED", "BLOCKED", "CANCELLED"}:
                attempt_id = str(row["attempt_id"])
                if any(
                    verified_id < int(row["id"])
                    for verified_id in code_verified_by_attempt.get(attempt_id, [])
                ):
                    post_green_attempts.add(attempt_id)

        retries = list(
            conn.execute(
                "SELECT * FROM job_retry_evidence WHERE job_id = ? "
                "AND created_at >= ? AND created_at < ? ORDER BY id",
                (job_id, period_start, period_end),
            )
        )
        retry_without_progress = 0
        for row in retries:
            reason = str(row["reason_code"])
            signatures.add(reason)
            missing_progress = (
                not {"prior_failure_digest", "response_change_digest", "result_delta_digest"}
                <= retry_columns
                or any(
                    row[name] is None
                    for name in (
                        "prior_failure_digest",
                        "response_change_digest",
                        "result_delta_digest",
                    )
                    if name in retry_columns
                )
            )
            if missing_progress or reason.startswith("RETRY_MISSING_") or reason in {
                "RETRY_EVIDENCE_NOT_DISTINCT",
                "RETRY_REJECTED_NO_NEW_EVIDENCE",
            }:
                retry_without_progress += 1
                signatures.add("RETRY_WITHOUT_PROGRESS")
        if false_complete:
            signatures.add("MISSING_OUTCOME_PROOF")
        observations.append(
            JobObservation(
                job_id=job_id,
                started=(
                    (job is not None and period_start <= int(job["created_at"]) < period_end)
                    or attempts_started > 0
                ),
                settled=latest_state in jobs_graph.TERMINAL_STATES,
                outcome_verified=bool(verified_by_attempt),
                false_complete=false_complete,
                false_blocked=false_blocked,
                retry_without_progress=retry_without_progress,
                missing_evidence=missing_evidence,
                post_green_failure=len(post_green_attempts),
                failure_signatures=tuple(sorted(signatures)),
            )
        )
    return tuple(observations)


def scorecard_from_connection(
    conn,
    *,
    period_start: int,
    period_end: int,
    guard_threshold: int = 2,
) -> ReliabilityScorecard:
    return build_scorecard(
        observations_from_connection(
            conn, period_start=period_start, period_end=period_end
        ),
        period_start=period_start,
        period_end=period_end,
        guard_threshold=guard_threshold,
    )


__all__ = [
    "JobObservation",
    "ReliabilityScorecard",
    "build_scorecard",
    "observations_from_connection",
    "scorecard_from_connection",
]
