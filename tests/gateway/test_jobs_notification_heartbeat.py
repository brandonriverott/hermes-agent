"""Task 8: one ten-minute heartbeat per stagnant phase episode.

Clock-controlled through the real gateway worker: a fake ``now`` drives
``deliver_due_notification_once``, which enqueues due heartbeats before
claiming rows.  A real-time base keeps the claim gate satisfiable while all
heartbeat decisions stay relative to the delivered-at timestamp.

- No heartbeat before 600 seconds.
- Exactly one heartbeat at 600 seconds without a phase change.
- No second heartbeat at 1,200 seconds in the same stagnant episode.
- A phase change resets heartbeat eligibility.
- A pending earlier milestone never lets a heartbeat overtake it.
- A restart does not create a duplicate heartbeat.

Progress is derived only from the durable outbox (delivered milestone rows),
never from file mtimes or logs.
"""

import time

import pytest

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_notifications as jn
from hermes_cli.sqlite_util import write_txn
from hermes_state import SessionDB
from gateway.jobs_notifications import deliver_due_notification_once


@pytest.fixture
def jobs_path(tmp_path):
    return tmp_path / "jobs.db"


@pytest.fixture
def session_db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


@pytest.fixture
def base():
    # A large real-ish timestamp so claim_due's next_attempt_at gate is
    # satisfied for rows created with the real clock; all heartbeat
    # assertions are relative offsets from this base.
    return int(time.time()) + 60


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


def _create_job(jobs_path, *, name="landing page"):
    conn = jdb.connect(jobs_path)
    try:
        return jdb.create_job(
            conn,
            name=name,
            goal="goal",
            requested_lane="codex",
            origin=_origin(),
        )
    finally:
        conn.close()


def _enqueue_milestone(jobs_path, job_id, milestone, *, revision, now):
    conn = jdb.connect(jobs_path)
    try:
        with write_txn(conn):
            jn.enqueue_locked(
                conn,
                job_id=job_id,
                attempt_id=None,
                job_revision=revision,
                milestone=milestone,
                payload={"number": 1, "name": "landing page"},
                now=now,
            )
    finally:
        conn.close()


def _deliver(session_db, jobs_path, *, now):
    return deliver_due_notification_once(
        session_db, jobs_path=jobs_path, now=now
    )


def _contents(session_db, session_id="session-1"):
    return [m["content"] for m in session_db.get_messages(session_id)]


def test_no_heartbeat_before_ten_minutes(jobs_path, session_db, base):
    _create_job(jobs_path)
    session_db.create_session("session-1", source="telegram")

    # Deliver the queued milestone at the base instant.
    assert _deliver(session_db, jobs_path, now=base) is not None
    assert len(_contents(session_db)) == 1

    # 599 seconds later there is still no heartbeat.
    assert _deliver(session_db, jobs_path, now=base + 599) is None
    assert len(_contents(session_db)) == 1
    conn = jdb.connect(jobs_path)
    try:
        assert all(
            r.milestone != jn.MILESTONE_HEARTBEAT
            for r in jn.list_notifications(conn)
        )
    finally:
        conn.close()


def test_exactly_one_heartbeat_at_ten_minutes(jobs_path, session_db, base):
    _create_job(jobs_path)
    session_db.create_session("session-1", source="telegram")
    assert _deliver(session_db, jobs_path, now=base) is not None

    assert _deliver(session_db, jobs_path, now=base + 600) is not None
    contents = _contents(session_db)
    assert len(contents) == 2
    assert contents[1].startswith("Still working")
    assert contents[1].endswith("(still queued)")

    conn = jdb.connect(jobs_path)
    try:
        heartbeats = [
            r
            for r in jn.list_notifications(conn)
            if r.milestone == jn.MILESTONE_HEARTBEAT
        ]
        assert len(heartbeats) == 1
        assert heartbeats[0].delivered_at == base + 600
    finally:
        conn.close()


def test_no_second_heartbeat_in_same_stagnant_episode(
    jobs_path, session_db, base
):
    _create_job(jobs_path)
    session_db.create_session("session-1", source="telegram")
    assert _deliver(session_db, jobs_path, now=base) is not None
    assert _deliver(session_db, jobs_path, now=base + 600) is not None
    assert len(_contents(session_db)) == 2

    # Same phase at 1,200 seconds total: still only the one heartbeat.
    assert _deliver(session_db, jobs_path, now=base + 1200) is None
    assert len(_contents(session_db)) == 2
    conn = jdb.connect(jobs_path)
    try:
        heartbeats = [
            r
            for r in jn.list_notifications(conn)
            if r.milestone == jn.MILESTONE_HEARTBEAT
        ]
        assert len(heartbeats) == 1
    finally:
        conn.close()


def test_phase_change_resets_heartbeat_eligibility(
    jobs_path, session_db, base
):
    job_id = _create_job(jobs_path)
    session_db.create_session("session-1", source="telegram")
    assert _deliver(session_db, jobs_path, now=base) is not None  # queued
    assert (
        _deliver(session_db, jobs_path, now=base + 600) is not None
    )  # heartbeat
    assert len(_contents(session_db)) == 2

    # Phase change at base+1300.
    _enqueue_milestone(
        jobs_path,
        job_id,
        jn.MILESTONE_ASSIGNED,
        revision=2,
        now=base + 1300,
    )
    assert _deliver(session_db, jobs_path, now=base + 1300) is not None
    assert _contents(session_db)[-1].startswith("Assigned")

    # 600 seconds later in the new phase a NEW heartbeat fires once.
    assert _deliver(session_db, jobs_path, now=base + 1900) is not None
    contents = _contents(session_db)
    heartbeats = [c for c in contents if c.startswith("Still working")]
    assert len(heartbeats) == 2
    assert any(c.endswith("(still assigned)") for c in contents)

    conn = jdb.connect(jobs_path)
    try:
        hb = [
            r
            for r in jn.list_notifications(conn)
            if r.milestone == jn.MILESTONE_HEARTBEAT
        ]
        assert len(hb) == 2
        assert {r.job_revision for r in hb} == {1, 2}
    finally:
        conn.close()


def test_pending_earlier_milestone_prevents_heartbeat_overtaking(
    jobs_path, session_db, base
):
    job_id = _create_job(jobs_path)
    session_db.create_session("session-1", source="telegram")
    assert _deliver(session_db, jobs_path, now=base) is not None  # queued

    # A later milestone is enqueued at base+100 and becomes due, but the
    # worker does not run again until base+700 — by then the queued-phase
    # heartbeat is also due.  Strict per-job ordering must deliver the
    # pending earlier milestone FIRST; the heartbeat must not overtake it.
    _enqueue_milestone(
        jobs_path,
        job_id,
        jn.MILESTONE_ASSIGNED,
        revision=2,
        now=base + 100,
    )

    assert _deliver(session_db, jobs_path, now=base + 700) is not None
    contents = _contents(session_db)
    assert contents[-1].startswith("Assigned")
    assert not any(c.startswith("Still working") for c in contents)

    # Next tick: the heartbeat delivers after the pending milestone.
    assert _deliver(session_db, jobs_path, now=base + 700) is not None
    contents = _contents(session_db)
    assert any(c.startswith("Still working") for c in contents)
    assert contents.index(
        next(c for c in contents if c.startswith("Assigned"))
    ) < contents.index(
        next(c for c in contents if c.startswith("Still working"))
    )


def test_restart_does_not_duplicate_heartbeat(jobs_path, session_db, base):
    _create_job(jobs_path)
    session_db.create_session("session-1", source="telegram")
    assert _deliver(session_db, jobs_path, now=base) is not None
    assert _deliver(session_db, jobs_path, now=base + 600) is not None
    assert len(_contents(session_db)) == 2

    # A fresh worker run (restart) over the same durable stores must not
    # enqueue or deliver a second heartbeat for the delivered episode.
    assert _deliver(session_db, jobs_path, now=base + 700) is None
    assert len(_contents(session_db)) == 2
    conn = jdb.connect(jobs_path)
    try:
        heartbeats = [
            r
            for r in jn.list_notifications(conn)
            if r.milestone == jn.MILESTONE_HEARTBEAT
        ]
        assert len(heartbeats) == 1
        assert heartbeats[0].delivered_at == base + 600
    finally:
        conn.close()
