"""Transactional, ordered notification outbox for the Jobs ledger.

Every user-visible milestone is written inside the same SQLite write
transaction as the Job creation or graph transition that produced it, so a
state cannot commit without its notification intent and notification intent
cannot exist for a state that did not commit.

Rows are immutable intents plus delivery state:

- ``id`` is the durable per-Job order; delivery never overtakes an earlier
  undelivered milestone of the same Job.
- ``notification_id`` is stable and derived from Job, graph revision, and
  milestone, so retries and restarts keep one identity.
- Uniqueness on ``(job_id, job_revision, milestone)`` makes transition
  retries idempotent.
- Claims are short leases with compare-and-set updates; an expired claim
  returns the row to the queue, and a later milestone waits for the earliest
  undelivered one.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Mapping, Optional

MILESTONE_QUEUED = "queued"
MILESTONE_ASSIGNED = "assigned"
MILESTONE_BUILDING = "building"
MILESTONE_TESTING = "testing"
MILESTONE_REVIEW = "review"
MILESTONE_CORRECTING = "correcting"
MILESTONE_NEEDS_YOU = "needs-you"
MILESTONE_FAILURE = "failure"
MILESTONE_FINISHED = "finished"

ALL_MILESTONES = frozenset(
    {
        MILESTONE_QUEUED,
        MILESTONE_ASSIGNED,
        MILESTONE_BUILDING,
        MILESTONE_TESTING,
        MILESTONE_REVIEW,
        MILESTONE_CORRECTING,
        MILESTONE_NEEDS_YOU,
        MILESTONE_FAILURE,
        MILESTONE_FINISHED,
    }
)

MAX_PAYLOAD_CHARS = 4096
MAX_ERROR_CHARS = 512


class NotificationClaimError(ValueError):
    """Raised when a claim compare-and-set cannot take the requested row."""


@dataclass(frozen=True)
class NotificationRecord:
    id: int
    notification_id: str
    job_id: str
    attempt_id: Optional[str]
    job_revision: int
    milestone: str
    payload: Mapping[str, object]
    transition_id: Optional[int]
    created_at: int
    next_attempt_at: int
    claim_owner: Optional[str]
    claim_expires_at: Optional[int]
    delivery_attempts: int
    last_error: Optional[str]
    delivered_at: Optional[int]


def milestone_for_state(
    target_state: str,
    *,
    attempt_terminal_failure: Optional[int] = None,
) -> Optional[str]:
    """The user milestone for a graph target state, or ``None``.

    ``QUEUED`` milestones are emitted at Job creation, not per attempt;
    ``VERIFIED`` and ``CANCELLED`` are durable graph states with no chat
    milestone.  A ``FAILED`` attempt is ``failure`` only when it was declared
    terminal; otherwise it is ``correcting`` (a retry/re-review is coming).
    """
    if target_state == "ASSIGNED":
        return MILESTONE_ASSIGNED
    if target_state == "BUILDING":
        return MILESTONE_BUILDING
    if target_state == "EVIDENCE_COLLECTING":
        return MILESTONE_TESTING
    if target_state == "REVIEWING":
        return MILESTONE_REVIEW
    if target_state == "BLOCKED":
        return MILESTONE_NEEDS_YOU
    if target_state == "COMPLETED":
        return MILESTONE_FINISHED
    if target_state == "FAILED":
        return (
            MILESTONE_FAILURE
            if attempt_terminal_failure == 1
            else MILESTONE_CORRECTING
        )
    return None


def _record(row: sqlite3.Row) -> NotificationRecord:
    return NotificationRecord(
        id=int(row["id"]),
        notification_id=str(row["notification_id"]),
        job_id=str(row["job_id"]),
        attempt_id=row["attempt_id"],
        job_revision=int(row["job_revision"]),
        milestone=str(row["milestone"]),
        payload=json.loads(row["payload_json"]),
        transition_id=row["transition_id"],
        created_at=int(row["created_at"]),
        next_attempt_at=int(row["next_attempt_at"]),
        claim_owner=row["claim_owner"],
        claim_expires_at=row["claim_expires_at"],
        delivery_attempts=int(row["delivery_attempts"]),
        last_error=row["last_error"],
        delivered_at=row["delivered_at"],
    )


def _fetch(conn: sqlite3.Connection, notification_id: str) -> NotificationRecord:
    row = conn.execute(
        "SELECT * FROM job_notifications WHERE notification_id = ?",
        (notification_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("notification row disappeared")
    return _record(row)


def enqueue_locked(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    attempt_id: Optional[str],
    job_revision: int,
    milestone: str,
    payload: Mapping[str, object],
    transition_id: Optional[int] = None,
    now: Optional[int] = None,
) -> NotificationRecord:
    """Insert one notification intent.  Caller holds a write transaction.

    The stable ``notification_id`` is derived from Job, graph revision, and
    milestone, and the ``(job_id, job_revision, milestone)`` unique index makes
    a repeated enqueue return the existing row instead of duplicating it.
    """
    if milestone not in ALL_MILESTONES:
        raise ValueError(f"unknown notification milestone: {milestone!r}")
    now = int(now) if now is not None else _now()
    payload_json = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
    if len(payload_json) > MAX_PAYLOAD_CHARS:
        raise ValueError("notification payload exceeds size bound")
    notification_id = f"n:{job_id}:{job_revision}:{milestone}"
    conn.execute(
        "INSERT OR IGNORE INTO job_notifications "
        "(notification_id, job_id, attempt_id, job_revision, milestone, "
        " payload_json, transition_id, created_at, next_attempt_at, "
        " delivery_attempts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
        (
            notification_id,
            job_id,
            attempt_id,
            job_revision,
            milestone,
            payload_json,
            transition_id,
            now,
            now,
        ),
    )
    return _fetch(conn, notification_id)


def list_notifications(
    conn: sqlite3.Connection,
    job_id: Optional[str] = None,
) -> list[NotificationRecord]:
    """All outbox rows for one Job (or the whole outbox), in durable order."""
    if job_id is None:
        rows = conn.execute(
            "SELECT * FROM job_notifications ORDER BY id ASC"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM job_notifications WHERE job_id = ? ORDER BY id ASC",
            (job_id,),
        ).fetchall()
    return [_record(row) for row in rows]


def claim_due(
    conn: sqlite3.Connection,
    *,
    owner: str,
    now: int,
    lease_seconds: int = 60,
) -> Optional[NotificationRecord]:
    """Claim the earliest due pending row, respecting per-Job order.

    A row is claimable when its next attempt time has arrived, it is not
    delivered, and no earlier undelivered row exists for the same Job — a
    later milestone can never overtake an earlier undelivered one.  The
    selection and the compare-and-set update run in one IMMEDIATE transaction,
    so two workers cannot claim the same row.
    """
    if not owner:
        raise ValueError("claim owner must not be empty")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    from hermes_cli.sqlite_util import write_txn

    with write_txn(conn):
        row = conn.execute(
            "SELECT * FROM job_notifications n "
            "WHERE n.next_attempt_at <= ? "
            "AND n.delivered_at IS NULL "
            "AND (n.claim_expires_at IS NULL OR n.claim_expires_at <= ?) "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM job_notifications earlier "
            "  WHERE earlier.job_id = n.job_id "
            "  AND earlier.id < n.id "
            "  AND earlier.delivered_at IS NULL"
            ") "
            "ORDER BY n.id ASC LIMIT 1",
            (now, now),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE job_notifications "
            "SET claim_owner = ?, claim_expires_at = ?, "
            "    delivery_attempts = delivery_attempts + 1 "
            "WHERE notification_id = ?",
            (owner, now + lease_seconds, row["notification_id"]),
        )
        return _fetch(conn, row["notification_id"])


def renew_claim(
    conn: sqlite3.Connection,
    notification_id: str,
    *,
    owner: str,
    now: int,
    lease_seconds: int = 60,
) -> NotificationRecord:
    """Extend the lease on a row the owner already holds."""
    if not owner:
        raise ValueError("claim owner must not be empty")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    from hermes_cli.sqlite_util import write_txn

    with write_txn(conn):
        cursor = conn.execute(
            "UPDATE job_notifications SET claim_expires_at = ? "
            "WHERE notification_id = ? AND claim_owner = ? "
            "AND delivered_at IS NULL",
            (now + lease_seconds, notification_id, owner),
        )
        if cursor.rowcount == 0:
            raise NotificationClaimError(
                f"notification {notification_id!r} is not owned by {owner!r}"
            )
        return _fetch(conn, notification_id)


def release_claim(
    conn: sqlite3.Connection,
    notification_id: str,
    *,
    owner: str,
    now: int,
    retry_at: int,
    error: Optional[str] = None,
) -> NotificationRecord:
    """Release a claim for a bounded retry; the row stays pending until then."""
    if retry_at < now:
        raise ValueError("retry_at must not be in the past")
    if error is not None and len(error) > MAX_ERROR_CHARS:
        raise ValueError("claim error exceeds size bound")
    from hermes_cli.sqlite_util import write_txn

    with write_txn(conn):
        cursor = conn.execute(
            "UPDATE job_notifications "
            "SET claim_owner = NULL, claim_expires_at = NULL, "
            "    next_attempt_at = ?, last_error = ? "
            "WHERE notification_id = ? AND claim_owner = ? "
            "AND delivered_at IS NULL",
            (retry_at, error, notification_id, owner),
        )
        if cursor.rowcount == 0:
            raise NotificationClaimError(
                f"notification {notification_id!r} is not owned by {owner!r}"
            )
        return _fetch(conn, notification_id)


def acknowledge(
    conn: sqlite3.Connection,
    notification_id: str,
    *,
    owner: str,
    now: int,
) -> NotificationRecord:
    """Mark a row delivered; only the current claim owner may do so."""
    if not owner:
        raise ValueError("claim owner must not be empty")
    from hermes_cli.sqlite_util import write_txn

    with write_txn(conn):
        cursor = conn.execute(
            "UPDATE job_notifications "
            "SET delivered_at = ?, claim_owner = NULL, claim_expires_at = NULL "
            "WHERE notification_id = ? AND claim_owner = ? "
            "AND delivered_at IS NULL",
            (now, notification_id, owner),
        )
        if cursor.rowcount == 0:
            raise NotificationClaimError(
                f"notification {notification_id!r} is not owned by {owner!r}"
            )
        return _fetch(conn, notification_id)


def _now() -> int:
    import time

    return int(time.time())


__all__ = [
    "ALL_MILESTONES",
    "MAX_ERROR_CHARS",
    "MAX_PAYLOAD_CHARS",
    "MILESTONE_ASSIGNED",
    "MILESTONE_BUILDING",
    "MILESTONE_CORRECTING",
    "MILESTONE_FAILURE",
    "MILESTONE_FINISHED",
    "MILESTONE_NEEDS_YOU",
    "MILESTONE_QUEUED",
    "MILESTONE_REVIEW",
    "MILESTONE_TESTING",
    "NotificationClaimError",
    "NotificationRecord",
    "acknowledge",
    "claim_due",
    "enqueue_locked",
    "list_notifications",
    "milestone_for_state",
    "release_claim",
    "renew_claim",
]
