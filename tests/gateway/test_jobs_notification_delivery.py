"""Passive exact-origin gateway delivery of the durable Jobs notification outbox.

Task 7 contract under test:

- One due row is claimed at a time; a structured system message is appended to
  the exact stored SessionDB session; the row is acknowledged only after the
  append succeeds.
- ``notification_id`` is the destination idempotency key
  (``platform_message_id``), so a crash between append and acknowledge cannot
  duplicate a visible message after restart.
- A unique live compression continuation of the stored origin may receive the
  message; zero or multiple continuations, archived sessions, unavailable
  sessions, and busy sessions all stay pending with a bounded retry.
- An unknown platform becomes a durable, visible BLOCKED state with a bounded
  safe reason — it is never silently retried forever.
- Delivery never falls back to Build Feed, home channel, recent session, or a
  decoy chat; only the exact stored origin (or its unique continuation) is used.
"""

import json
import time
from types import SimpleNamespace

import pytest

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_handoffs
from hermes_cli import jobs_notifications as jn
from hermes_state import SessionDB
from gateway.jobs_notifications import (
    _transition_binding,
    _validated_handoff,
    deliver_due_notification_once,
)

TEST_DIGEST = "sha256:" + "a" * 64


@pytest.fixture
def jobs_path(tmp_path):
    return tmp_path / "jobs.db"


@pytest.fixture
def session_db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


def _origin(**overrides):
    origin = {
        "platform": "telegram",
        "chat_id": "chat-1",
        "session_id": "session-1",
        "chat_type": "dm",
        "thread_id": "thread-1",
        "user_id": "user-1",
        "profile": "default",
    }
    origin.update(overrides)
    return origin


def _job(conn, *, origin=None):
    """Create a Job with an exact origin; returns its ``j_`` id."""
    return jdb.create_job(
        conn,
        name="job",
        goal="goal",
        requested_lane="codex",
        origin=_origin() if origin is None else origin,
    )


def _messages(session_db, session_id):
    return session_db.get_messages(session_id)


def _handoff(job_id):
    return jobs_handoffs.normalize_handoff(
        {
            "summary": "Builder completed the scoped change.",
            "evidence_summary": [
                {"label": "Focused tests", "result": "42 passed", "digest": TEST_DIGEST}
            ],
            "next_action": "Review the committed change.",
            "issues": [],
            "decision_request": None,
        },
        job_id=job_id,
        attempt_id="attempt-1",
        speaker_id="builder-1",
        speaker_role="builder",
        speaker_executor="claude",
        from_phase="BUILDING",
        to_phase="EVIDENCE_COLLECTING",
        next_owner_role="reviewer",
        outcome="handed_off",
        artifact_identity=None,
        transition_evidence={"tests": TEST_DIGEST},
        created_at=1,
    )


def _deliver(session_db, jobs_path, *, now=None, busy_check=None):
    return deliver_due_notification_once(
        session_db,
        jobs_path=jobs_path,
        now=now if now is not None else int(time.time()),
        busy_check=busy_check,
    )


def test_delivers_every_message_only_to_exact_origin(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        job_a = _job(conn, origin=_origin(session_id="sess-a", chat_id="chat-a"))
        job_b = _job(conn, origin=_origin(session_id="sess-b", chat_id="chat-b"))
    finally:
        conn.close()

    session_db.create_session("sess-a", source="telegram")
    session_db.create_session("sess-b", source="telegram")
    session_db.create_session("decoy", source="telegram")

    delivered = {_deliver(session_db, jobs_path) for _ in range(2)}

    a_msgs = _messages(session_db, "sess-a")
    b_msgs = _messages(session_db, "sess-b")
    assert len(a_msgs) == 1
    assert len(b_msgs) == 1
    assert _messages(session_db, "decoy") == []
    # The stable notification id is the platform_message_id on the append.
    assert a_msgs[0]["platform_message_id"] == f"n:{job_a}:1:queued"
    assert b_msgs[0]["platform_message_id"] == f"n:{job_b}:1:queued"
    # Both rows are acknowledged.
    conn = jdb.connect(jobs_path)
    try:
        for row in jn.list_notifications(conn):
            assert row.delivered_at is not None
    finally:
        conn.close()


def test_ordered_delivery_across_milestones(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        job_id = _job(conn)
        # Both milestones due at the same instant: only the per-Job order gate
        # (earlier undelivered row blocks later rows) decides delivery order.
        # ``enqueue_locked`` requires the caller to hold a write transaction.
        from hermes_cli.sqlite_util import write_txn

        with write_txn(conn):
            jn.enqueue_locked(
                conn,
                job_id=job_id,
                attempt_id=None,
                job_revision=1,
                milestone=jn.MILESTONE_ASSIGNED,
                payload={"number": 1, "name": "job"},
                now=int(time.time()),
            )
    finally:
        conn.close()
    session_db.create_session("session-1", source="telegram")

    # First delivery only moves the earliest (queued) row.
    assert _deliver(session_db, jobs_path) is not None
    conn = jdb.connect(jobs_path)
    try:
        rows = jn.list_notifications(conn, job_id=job_id)
        assert rows[0].milestone == "queued"
        assert rows[0].delivered_at is not None
        assert rows[1].milestone == "assigned"
        assert rows[1].delivered_at is None
    finally:
        conn.close()
    assert len(_messages(session_db, "session-1")) == 1

    # Second delivery moves the next milestone.
    assert _deliver(session_db, jobs_path) is not None
    conn = jdb.connect(jobs_path)
    try:
        rows = jn.list_notifications(conn, job_id=job_id)
        assert rows[1].delivered_at is not None
    finally:
        conn.close()
    msgs = _messages(session_db, "session-1")
    assert len(msgs) == 2
    assert msgs[0]["platform_message_id"].endswith(":queued")
    assert msgs[1]["platform_message_id"].endswith(":assigned")


def test_payload_only_handoff_fails_closed_without_durable_transition_binding(
    jobs_path, session_db
):
    conn = jdb.connect(jobs_path)
    try:
        job_id = _job(conn)
        from hermes_cli.sqlite_util import write_txn

        handoff = _handoff(job_id)
        with write_txn(conn):
            jn.enqueue_locked(
                conn,
                job_id=job_id,
                attempt_id=None,
                job_revision=2,
                milestone=jn.MILESTONE_TESTING,
                payload={
                    "number": 1,
                    "name": "job",
                    "target_state": "EVIDENCE_COLLECTING",
                    "handoff": handoff,
                    "evidence_json": {"tests": TEST_DIGEST},
                },
                now=int(time.time()),
            )
    finally:
        conn.close()
    session_db.create_session("session-1", source="telegram")
    assert _deliver(session_db, jobs_path) is not None  # legacy queued
    assert _deliver(session_db, jobs_path) is not None  # substantive testing
    msg = _messages(session_db, "session-1")[-1]
    assert msg["content"].startswith("Hermes could not explain this handoff")
    assert "builder → reviewer" not in msg["content"]
    metadata = (
        msg["display_metadata"]
        if isinstance(msg["display_metadata"], dict)
        else json.loads(msg["display_metadata"])
    )
    assert "speaker_role" not in metadata
    assert "speaker_executor" not in metadata
    assert "next_owner_role" not in metadata
    assert "outcome" not in metadata


def test_missing_trusted_evidence_is_bounded_diagnostic_without_claimed_owner(
    jobs_path, session_db
):
    conn = jdb.connect(jobs_path)
    try:
        job_id = _job(conn)
        from hermes_cli.sqlite_util import write_txn

        with write_txn(conn):
            jn.enqueue_locked(
                conn,
                job_id=job_id,
                attempt_id=None,
                job_revision=2,
                milestone=jn.MILESTONE_TESTING,
                payload={
                    "number": 1,
                    "name": "job",
                    "target_state": "EVIDENCE_COLLECTING",
                    "handoff": {"summary": "do not echo this"},
                },
                now=int(time.time()),
            )
    finally:
        conn.close()
    session_db.create_session("session-1", source="telegram")
    assert _deliver(session_db, jobs_path) is not None
    assert _deliver(session_db, jobs_path) is not None
    msg = _messages(session_db, "session-1")[-1]
    assert msg["content"].startswith("Hermes could not explain this handoff")
    assert "do not echo this" not in msg["content"]
    metadata = msg["display_metadata"]
    assert "speaker_role" not in metadata
    assert "speaker_executor" not in metadata
    assert "next_owner_role" not in metadata
    assert "outcome" not in metadata


@pytest.mark.parametrize(
    "field, value",
    [
        ("job_id", "foreign-job"),
        ("attempt_id", "foreign-attempt"),
        ("to_phase", "REVIEWING"),
        ("speaker_id", "foreign-speaker"),
    ],
)
def test_gateway_rejects_handoff_foreign_to_durable_transition(field, value):
    job_id = "job-1"
    handoff = _handoff(job_id)
    handoff[field] = value
    record = SimpleNamespace(
        job_id=job_id,
        attempt_id="attempt-1",
        transition_id=7,
        milestone=jn.MILESTONE_TESTING,
        payload={"target_state": "EVIDENCE_COLLECTING", "handoff": handoff},
    )

    class _Conn:
        def execute(self, *_args):
            return SimpleNamespace(
                fetchone=lambda: {
                    "job_id": job_id,
                    "attempt_id": "attempt-1",
                    "target_state": "EVIDENCE_COLLECTING",
                    "initiator_id": "builder-1",
                }
            )

    binding = _transition_binding(_Conn(), record)
    assert binding is not None
    assert (
        _validated_handoff(
            _Conn(),
            record,
            {"tests": TEST_DIGEST},
            binding,
        )
        is None
    )


def test_restart_idempotency_does_not_duplicate_visible_message(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        _job(conn)
    finally:
        conn.close()
    session_db.create_session("session-1", source="telegram")

    # Simulate a crash between append and acknowledge: the worker claims the
    # row, appends the visible message carrying the notification_id as
    # platform_message_id, then dies before acknowledging.  The lease later
    # expires, returning the row to the queue.
    conn = jdb.connect(jobs_path)
    try:
        record = jn.claim_due(
            conn,
            owner="gateway-jobs-notifier",
            now=int(time.time()),
        )
        assert record is not None
        session_db.append_message(
            "session-1",
            role="system",
            content="jobs:queued (pre-crash)",
            platform_message_id=record.notification_id,
            display_kind="jobs_update",
        )
        # Crash: the claim is never acknowledged nor released.  The lease
        # expires and the row becomes claimable again.
        jn.release_claim(
            conn,
            record.notification_id,
            owner="gateway-jobs-notifier",
            now=int(time.time()),
            retry_at=int(time.time()),
            error="crash-after-append",
        )
    finally:
        conn.close()
    assert len(_messages(session_db, "session-1")) == 1

    # Re-run: must acknowledge WITHOUT appending a second visible message.
    assert _deliver(session_db, jobs_path) is not None
    assert len(_messages(session_db, "session-1")) == 1
    conn = jdb.connect(jobs_path)
    try:
        assert jn.list_notifications(conn)[0].delivered_at is not None
    finally:
        conn.close()


def test_unique_compression_continuation_receives(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        _job(conn, origin=_origin(session_id="sess-parent"))
    finally:
        conn.close()

    session_db.create_session("sess-parent", source="telegram")
    session_db.end_session("sess-parent", end_reason="compression")
    session_db.create_session(
        "sess-tip",
        source="telegram",
        parent_session_id="sess-parent",
    )

    assert _deliver(session_db, jobs_path) is not None
    # The unique live continuation receives the message; the parent does not.
    assert len(_messages(session_db, "sess-tip")) == 1
    assert _messages(session_db, "sess-parent") == []


def test_zero_or_multiple_continuations_remain_pending(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        _job(conn, origin=_origin(session_id="sess-zero"))
        _job(conn, origin=_origin(session_id="sess-multi", chat_id="chat-multi"))
    finally:
        conn.close()

    session_db.create_session("sess-zero", source="telegram")
    session_db.end_session("sess-zero", end_reason="compression")
    session_db.create_session("sess-multi", source="telegram")
    session_db.end_session("sess-multi", end_reason="compression")
    session_db.create_session(
        "sess-multi-child-1",
        source="telegram",
        parent_session_id="sess-multi",
    )
    session_db.create_session(
        "sess-multi-child-2",
        source="telegram",
        parent_session_id="sess-multi",
    )

    assert _deliver(session_db, jobs_path) is not None
    assert _deliver(session_db, jobs_path) is not None

    for sid in (
        "sess-zero",
        "sess-multi",
        "sess-multi-child-1",
        "sess-multi-child-2",
    ):
        assert _messages(session_db, sid) == []

    conn = jdb.connect(jobs_path)
    try:
        rows = jn.list_notifications(conn)
        assert len(rows) == 2
        assert all(row.delivered_at is None for row in rows)
        assert all(row.delivery_attempts >= 1 for row in rows)
        # Bounded retry: next attempt is in the future, not immediate.
        now = int(time.time())
        assert all(row.next_attempt_at > now for row in rows)
    finally:
        conn.close()


def test_archived_origin_remains_pending(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        _job(conn, origin=_origin(session_id="sess-archived"))
    finally:
        conn.close()
    session_db.create_session("sess-archived", source="telegram")
    session_db.set_session_archived("sess-archived", True)

    assert _deliver(session_db, jobs_path) is not None
    assert _messages(session_db, "sess-archived") == []
    conn = jdb.connect(jobs_path)
    try:
        row = jn.list_notifications(conn)[0]
        assert row.delivered_at is None
        assert row.next_attempt_at > int(time.time())
    finally:
        conn.close()


def test_unavailable_session_remains_pending(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        _job(conn, origin=_origin(session_id="sess-never-created"))
    finally:
        conn.close()
    session_db.create_session("decoy", source="telegram")

    assert _deliver(session_db, jobs_path) is not None
    assert _messages(session_db, "decoy") == []
    conn = jdb.connect(jobs_path)
    try:
        row = jn.list_notifications(conn)[0]
        assert row.delivered_at is None
        assert row.next_attempt_at > int(time.time())
    finally:
        conn.close()


def test_busy_origin_remains_pending(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        _job(conn, origin=_origin(session_id="sess-busy"))
    finally:
        conn.close()
    session_db.create_session("sess-busy", source="telegram")

    assert (
        _deliver(session_db, jobs_path, busy_check=lambda sid: sid == "sess-busy")
        is not None
    )
    assert _messages(session_db, "sess-busy") == []
    conn = jdb.connect(jobs_path)
    try:
        row = jn.list_notifications(conn)[0]
        assert row.delivered_at is None
        assert row.next_attempt_at > int(time.time())
    finally:
        conn.close()


def test_unknown_platform_becomes_visible_blocked_state(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        _job(conn, origin=_origin(platform="carrier_pigeon", session_id="sess-x"))
    finally:
        conn.close()
    session_db.create_session("sess-x", source="telegram")

    assert _deliver(session_db, jobs_path) is not None
    assert _messages(session_db, "sess-x") == []

    conn = jdb.connect(jobs_path)
    try:
        rows = jn.list_notifications(conn)
        assert len(rows) == 1
        row = rows[0]
        assert row.delivered_at is None
        # Durable visible BLOCKED state with a bounded safe reason.
        assert row.blocked_reason is not None
        assert "unknown-platform" in row.blocked_reason
        assert "carrier_pigeon" in row.blocked_reason
        assert len(row.blocked_reason) <= jn.MAX_ERROR_CHARS
    finally:
        conn.close()

    # It is never claimed again (no silent retry forever).
    conn = jdb.connect(jobs_path)
    try:
        assert (
            jn.claim_due(
                conn,
                owner="gateway-jobs-notifier",
                now=int(time.time()) + 3600,
            )
            is None
        )
    finally:
        conn.close()


def test_never_writes_to_build_feed_home_or_recent_decoy(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        _job(conn, origin=_origin(session_id="sess-missing"))
    finally:
        conn.close()
    # Decoys that must never receive the message even though the origin is gone.
    session_db.create_session("build-feed", source="telegram")
    session_db.create_session("home-channel", source="telegram")
    session_db.create_session("recent-session", source="telegram")
    session_db.create_session("decoy", source="telegram")

    assert _deliver(session_db, jobs_path) is not None

    for sid in ("build-feed", "home-channel", "recent-session", "decoy"):
        assert _messages(session_db, sid) == []
    conn = jdb.connect(jobs_path)
    try:
        assert jn.list_notifications(conn)[0].delivered_at is None
    finally:
        conn.close()


def test_message_is_structured_and_preserves_origin_metadata(jobs_path, session_db):
    conn = jdb.connect(jobs_path)
    try:
        job_id = _job(conn)
    finally:
        conn.close()
    session_db.create_session("session-1", source="telegram")

    assert _deliver(session_db, jobs_path) is not None
    msgs = _messages(session_db, "session-1")
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg["role"] == "system"
    # Task 8 wording: concise user-facing lifecycle message, structured and
    # stable (the raw job id lives in display_metadata, not the chat text).
    assert msg["content"] == "Legacy Jobs update — Queued — job (#1) accepted"
    assert msg["display_kind"] == "jobs_update"
    meta = (
        json.loads(msg["display_metadata"])
        if isinstance(msg["display_metadata"], str)
        else (msg["display_metadata"] or {})
    )
    assert meta["platform"] == "telegram"
    assert meta["chat_id"] == "chat-1"
    assert meta["thread_id"] == "thread-1"
    assert meta["session_id"] == "session-1"
    assert meta["profile"] == "default"
    assert meta["job_id"] == job_id
