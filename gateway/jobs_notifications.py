"""Passive exact-origin delivery of the durable Jobs notification outbox.

Task 7 consumer.  Every user-visible milestone written to
``job_notifications`` by the Jobs ledger is delivered, one due row at a
time, to the exact stored origin session in the Hermes ``SessionDB`` —
never to Build Feed, the home channel, the most recent session, or any
other fallback chat.

Contract:

- Claim one due row with the existing lease/CAS semantics
  (:func:`hermes_cli.jobs_notifications.claim_due`).
- Resolve the exact stored origin and, where the contract permits, a
  *unique* live compression continuation of it.
- Append a structured system message to the resolved session, carrying
  the stable ``notification_id`` as ``platform_message_id`` (the
  destination idempotency key).  A crash between append and acknowledge
  therefore cannot duplicate the visible message after restart.
- Acknowledge only after the append succeeds.
- Unavailable, archived, busy, zero-continuation, and ambiguous origins
  stay pending with a bounded retry via ``release_claim``.
- An unknown platform parks the row in a durable, visible blocked state
  via ``block_notification`` — it is surfaced by health/status commands,
  never silently retried forever.

This worker is passive: it never wakes an LLM turn.  Task 8 owns the
user-facing phase wording (rendered from the bounded payload by
:func:`hermes_cli.jobs_notifications.render_milestone_message`). Historical
heartbeat rows remain deliverable, but this worker never creates a time-only
heartbeat intent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable, Mapping, Optional

logger = logging.getLogger(__name__)

DEFAULT_OWNER = "gateway-jobs-notifier"
DEFAULT_RETRY_SECONDS = 30


def _is_known_platform(name: object) -> bool:
    """True when *name* is a built-in or registered plugin platform.

    ``gateway.config.Platform`` accepts built-in members and dynamically
    registered plugin platforms; an arbitrary string raises ``ValueError``.
    """
    from gateway.config import Platform

    if not isinstance(name, str) or not name.strip():
        return False
    try:
        Platform(name.strip().lower())
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _origin_for(conn, job_id: str) -> Optional[dict]:
    """Read the exact stored origin JSON for a Job, or ``None``."""
    row = conn.execute(
        "SELECT origin FROM job_origins WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        return None
    try:
        origin = json.loads(row["origin"])
    except (TypeError, json.JSONDecodeError):
        return None
    return origin if isinstance(origin, dict) else None


def _resolve_target_session_id(
    session_db,
    origin: Mapping[str, object],
    *,
    busy_check: Optional[Callable[[str], bool]] = None,
) -> Optional[str]:
    """Resolve the exact session to append to, or ``None`` for pending.

    Rules:
    - The stored ``session_id`` is authoritative; an empty/missing value is
      unavailable (pending).
    - A missing row, an archived row, or a busy row stays pending.
    - A session ended with ``compression`` may be followed to its *unique*
      live continuation; zero or multiple continuations stay pending.
    - Any other ended session is unavailable (pending) — the exact origin is
      gone and no fallback is allowed.
    """
    session_id = str(origin.get("session_id") or "").strip()
    if not session_id:
        return None
    session = session_db.get_session(session_id)
    if session is None:
        return None
    if session.get("archived"):
        return None
    if busy_check is not None and busy_check(session_id):
        return None
    if session.get("ended_at") is not None:
        if session.get("end_reason") != "compression":
            return None
        child = session_db.find_live_compression_child(session_id)
        if child is None:
            return None
        child_id = child["id"]
        if busy_check is not None and busy_check(child_id):
            return None
        return child_id
    return session_id


def _is_re_review(conn, record) -> bool:
    """True when a ``review`` milestone follows a delivered ``correcting``.

    A later ``REVIEWING`` transition after a correction renders as a
    re-review while retaining the stable ``review`` milestone type.  The
    flag is derived from the durable delivered history, never from file
    mtimes or logs.
    """
    from hermes_cli import jobs_notifications as jn

    if record.milestone != jn.MILESTONE_REVIEW:
        return False
    row = conn.execute(
        "SELECT milestone FROM job_notifications "
        "WHERE job_id = ? AND id < ? AND delivered_at IS NOT NULL "
        "AND milestone != ? "
        "ORDER BY id DESC LIMIT 1",
        (record.job_id, record.id, jn.MILESTONE_HEARTBEAT),
    ).fetchone()
    return row is not None and row["milestone"] == jn.MILESTONE_CORRECTING


def _format_message(conn, record, *, re_review: bool) -> str:
    """User-facing lifecycle message for a milestone (Task 8 wording)."""
    from hermes_cli import jobs_notifications as jn

    binding = _transition_binding(conn, record)
    return jn.render_milestone_message(
        record,
        re_review=re_review,
        transition_evidence=_trusted_transition_evidence(conn, record),
        expected_job_id=None if binding is None else binding["job_id"],
        expected_attempt_id=None if binding is None else binding["attempt_id"],
        expected_target_state=None if binding is None else binding["target_state"],
        expected_speaker_id=None if binding is None else binding["speaker_id"],
    )


def _trusted_transition_evidence(conn, record) -> Optional[dict[str, str]]:
    """Load evidence from durable transition storage or durable payload context.

    Handoff facts themselves are never used as evidence. A transition row is
    preferred and must belong to this Job; payload evidence is accepted only
    as canonical persisted context for legacy/test rows that have no reference.
    """
    from hermes_cli import jobs_db as jdb, jobs_handoffs

    raw = None
    if record.transition_id is not None:
        row = conn.execute(
            "SELECT job_id, evidence_json FROM job_attempt_transitions WHERE id = ?",
            (record.transition_id,),
        ).fetchone()
        if row is None or str(row["job_id"]) != record.job_id:
            return None
        raw = row["evidence_json"]
    else:
        payload = record.payload
        raw = payload.get("evidence_json")
        if raw is None:
            raw = payload.get("transition_evidence")
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            return jdb._stored_transition_evidence(raw)
        if type(raw) is dict:
            return jdb._stored_transition_evidence(jdb._reliability_json(raw))
    except (
        jobs_handoffs.HandoffValidationError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ):
        return None
    return None


def _validated_handoff(
    conn,
    record,
    evidence: Optional[Mapping[str, str]],
    binding: Optional[Mapping[str, object]],
) -> Optional[dict]:
    """Return a handoff only when strict validation succeeds with trusted evidence."""
    from hermes_cli import jobs_handoffs, jobs_notifications as jn

    if "handoff" not in record.payload or binding is None:
        return None
    target_state = record.payload.get("target_state")
    try:
        handoff = jobs_handoffs.validate_persisted_handoff(
            record.payload["handoff"],
            target_state=target_state if isinstance(target_state, str) else None,
            transition_evidence=evidence,
        )
        expected_milestone = jn.milestone_for_state(binding["target_state"])
        milestone_matches = (
            record.milestone in {jn.MILESTONE_CORRECTING, jn.MILESTONE_FAILURE}
            if binding["target_state"] == "FAILED"
            else expected_milestone is not None
            and record.milestone == expected_milestone
        )
        if (
            handoff["job_id"] != binding["job_id"]
            or handoff["attempt_id"] != binding["attempt_id"]
            or handoff["to_phase"] != binding["target_state"]
            or handoff["speaker_id"] != binding["speaker_id"]
            or not milestone_matches
        ):
            raise jobs_handoffs.HandoffValidationError(
                "handoff identity does not match trusted transition"
            )
        return handoff
    except (jobs_handoffs.HandoffValidationError, TypeError, ValueError):
        return None


def _transition_binding(conn, record) -> Optional[dict[str, object]]:
    """Return trusted identity fields for the notification's durable edge."""
    if record.transition_id is None:
        return None
    row = conn.execute(
        "SELECT job_id, attempt_id, target_state, initiator_id "
        "FROM job_attempt_transitions WHERE id = ?",
        (record.transition_id,),
    ).fetchone()
    if row is None or str(row["job_id"]) != record.job_id:
        return None
    if row["attempt_id"] != record.attempt_id:
        return None
    return {
        "job_id": str(row["job_id"]),
        "attempt_id": row["attempt_id"],
        "target_state": str(row["target_state"]),
        "speaker_id": str(row["initiator_id"]),
    }


def _display_metadata(
    origin: Mapping[str, object],
    record,
    target_session_id: str,
    *,
    handoff: Optional[Mapping[str, object]] = None,
) -> dict:
    metadata = {
        "kind": "jobs_update",
        "notification_id": record.notification_id,
        "job_id": record.job_id,
        "milestone": record.milestone,
        "job_revision": record.job_revision,
        "platform": str(origin.get("platform") or ""),
        "chat_id": str(origin.get("chat_id") or ""),
        "chat_type": str(origin.get("chat_type") or ""),
        "thread_id": str(origin.get("thread_id") or ""),
        "user_id": str(origin.get("user_id") or ""),
        "profile": str(origin.get("profile") or ""),
        "session_id": target_session_id,
    }
    if handoff is not None:
        metadata.update({
            "speaker_role": handoff["speaker_role"],
            "speaker_executor": handoff["speaker_executor"],
            "next_owner_role": handoff["next_owner_role"],
            "outcome": handoff["outcome"],
        })
    return metadata


def _release_for_retry(
    conn,
    record,
    *,
    owner: str,
    now: int,
    retry_seconds: int,
    error: str,
) -> None:
    from hermes_cli import jobs_notifications as jn

    jn.release_claim(
        conn,
        record.notification_id,
        owner=owner,
        now=now,
        retry_at=now + retry_seconds,
        error=error[: jn.MAX_ERROR_CHARS],
    )


def _deliver_one(
    conn,
    session_db,
    record,
    *,
    owner: str,
    now: int,
    busy_check: Optional[Callable[[str], bool]],
    retry_seconds: int,
) -> None:
    """Deliver one claimed row to its exact origin (caller owns the claim)."""
    from hermes_cli import jobs_notifications as jn

    origin = _origin_for(conn, record.job_id)
    if origin is None:
        _release_for_retry(
            conn,
            record,
            owner=owner,
            now=now,
            retry_seconds=retry_seconds,
            error="origin-missing",
        )
        return

    platform = origin.get("platform")
    if not _is_known_platform(platform):
        jn.block_notification(
            conn,
            record.notification_id,
            owner=owner,
            now=now,
            reason=f"unknown-platform:{platform}",
        )
        return

    target_session_id = _resolve_target_session_id(
        session_db, origin, busy_check=busy_check
    )
    if target_session_id is None:
        _release_for_retry(
            conn,
            record,
            owner=owner,
            now=now,
            retry_seconds=retry_seconds,
            error="origin-unavailable-or-ambiguous",
        )
        return

    # Destination idempotency: a prior append that crashed before acknowledge
    # already carries this notification_id as platform_message_id.
    if session_db.has_platform_message_id(target_session_id, record.notification_id):
        jn.acknowledge(conn, record.notification_id, owner=owner, now=now)
        return

    evidence = _trusted_transition_evidence(conn, record)
    binding = _transition_binding(conn, record)
    handoff = _validated_handoff(conn, record, evidence, binding)
    session_db.append_message(
        target_session_id,
        role="system",
        content=jn.render_milestone_message(
            record,
            re_review=_is_re_review(conn, record),
            transition_evidence=evidence,
            expected_job_id=None if binding is None else binding["job_id"],
            expected_attempt_id=(None if binding is None else binding["attempt_id"]),
            expected_target_state=(
                None if binding is None else binding["target_state"]
            ),
            expected_speaker_id=(None if binding is None else binding["speaker_id"]),
        ),
        platform_message_id=record.notification_id,
        display_kind="jobs_update",
        display_metadata=_display_metadata(
            origin,
            record,
            target_session_id,
            handoff=handoff,
        ),
    )
    # Acknowledge only after the append succeeded.
    jn.acknowledge(conn, record.notification_id, owner=owner, now=now)


def deliver_due_notification_once(
    session_db,
    *,
    jobs_path=None,
    owner: str = DEFAULT_OWNER,
    now: Optional[int] = None,
    busy_check: Optional[Callable[[str], bool]] = None,
    retry_seconds: int = DEFAULT_RETRY_SECONDS,
) -> Optional[str]:
    """Claim one due outbox row and deliver it to its exact origin.

    Returns the ``notification_id`` handled, or ``None`` when nothing was
    due.  Never raises for a per-row failure: unexpected delivery errors
    release the row with a bounded retry so the watcher loop survives.
    """
    from hermes_cli import jobs_db as jdb
    from hermes_cli import jobs_notifications as jn

    now = int(now) if now is not None else int(time.time())
    conn = jdb.connect(jobs_path)
    try:
        record = jn.claim_due(conn, owner=owner, now=now)
        if record is None:
            return None
        try:
            _deliver_one(
                conn,
                session_db,
                record,
                owner=owner,
                now=now,
                busy_check=busy_check,
                retry_seconds=retry_seconds,
            )
        except Exception as exc:
            logger.warning(
                "jobs notifications: delivery failed for %s: %s",
                record.notification_id,
                type(exc).__name__,
            )
            try:
                _release_for_retry(
                    conn,
                    record,
                    owner=owner,
                    now=now,
                    retry_seconds=retry_seconds,
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                logger.exception(
                    "jobs notifications: failed to release claim for %s",
                    record.notification_id,
                )
        return record.notification_id
    finally:
        conn.close()


class GatewayJobsNotificationsMixin:
    """Bounded, passive outbox consumer wired into :class:`GatewayRunner`.

    One due row per tick, SQLite work pushed to a thread, failures isolated
    per tick, and a 1-second-slice sleep so shutdown stays snappy — the same
    shape as the kanban notifier watcher.
    """

    async def _jobs_notifications_watcher(self, interval: float = 10.0) -> None:
        """Poll the durable Jobs outbox and deliver one due row per tick."""
        try:
            from hermes_cli import jobs_db as _jdb  # noqa: F401
        except Exception:
            logger.warning("jobs notifications: jobs_db not importable; disabled")
            return

        await asyncio.sleep(5)  # let the gateway finish wiring adapters

        while getattr(self, "_running", False):
            try:
                await asyncio.to_thread(self._deliver_one_jobs_notification)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("jobs notifications: unexpected watcher error")

            slept = 0.0
            while slept < interval and getattr(self, "_running", False):
                await asyncio.sleep(min(1.0, interval - slept))
                slept += 1.0

    def _deliver_one_jobs_notification(self) -> None:
        session_db = getattr(getattr(self, "_session_db", None), "_db", None)
        if session_db is None:
            return
        deliver_due_notification_once(
            session_db,
            busy_check=self._jobs_session_is_busy,
        )

    def _jobs_session_is_busy(self, session_id: str) -> bool:
        """A session is busy when the gateway is mid-turn on it."""
        store = getattr(self, "session_store", None)
        if store is None:
            return False
        try:
            entry = store.lookup_by_session_id(session_id)
        except Exception:
            return False
        return bool(entry and getattr(entry, "active_turn_token", None))


__all__ = [
    "DEFAULT_OWNER",
    "DEFAULT_RETRY_SECONDS",
    "GatewayJobsNotificationsMixin",
    "deliver_due_notification_once",
]
