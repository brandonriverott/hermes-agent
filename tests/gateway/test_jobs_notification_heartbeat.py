"""Heartbeat delivery compatibility and suppression tests."""

import time

import pytest

from gateway.jobs_notifications import deliver_due_notification_once
from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_notifications as jn
from hermes_cli.sqlite_util import write_txn
from hermes_state import SessionDB


@pytest.fixture
def jobs_path(tmp_path):
    return tmp_path / "jobs.db"


@pytest.fixture
def session_db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


def _origin(**overrides):
    value = {
        "platform": "telegram",
        "chat_id": "chat-1",
        "session_id": "session-1",
        "chat_type": "dm",
        "thread_id": "thread-1",
        "user_id": "user-1",
        "profile": "default",
    }
    value.update(overrides)
    return value


def _create_job(jobs_path):
    conn = jdb.connect(jobs_path)
    try:
        return jdb.create_job(
            conn,
            name="landing page",
            goal="goal",
            requested_lane="codex",
            origin=_origin(),
        )
    finally:
        conn.close()


def _enqueue_legacy_heartbeat(jobs_path, job_id, now):
    conn = jdb.connect(jobs_path)
    try:
        with write_txn(conn):
            jn.enqueue_locked(
                conn,
                job_id=job_id,
                attempt_id=None,
                job_revision=1,
                milestone=jn.MILESTONE_HEARTBEAT,
                payload={"number": 1, "name": "landing page", "phase": "queued"},
                now=now,
            )
    finally:
        conn.close()


def _deliver(session_db, jobs_path, now):
    return deliver_due_notification_once(session_db, jobs_path=jobs_path, now=now)


def test_elapsed_time_does_not_enqueue_time_only_heartbeat(jobs_path, session_db):
    base = int(time.time()) + 60
    _create_job(jobs_path)
    session_db.create_session("session-1", source="telegram")
    assert _deliver(session_db, jobs_path, base) is not None
    assert _deliver(session_db, jobs_path, base + 600) is None
    conn = jdb.connect(jobs_path)
    try:
        assert [
            r
            for r in jn.list_notifications(conn)
            if r.milestone == jn.MILESTONE_HEARTBEAT
        ] == []
    finally:
        conn.close()


def test_historical_heartbeat_row_is_delivered_once(jobs_path, session_db):
    base = int(time.time()) + 60
    job_id = _create_job(jobs_path)
    _enqueue_legacy_heartbeat(jobs_path, job_id, base)
    session_db.create_session("session-1", source="telegram")
    assert _deliver(session_db, jobs_path, base) is not None
    # The queued row is ordered first; heartbeat is readable on the next pass.
    assert _deliver(session_db, jobs_path, base) is not None
    assert _deliver(session_db, jobs_path, base + 600) is None
    contents = [m["content"] for m in session_db.get_messages("session-1")]
    assert contents[-1].startswith("Legacy Jobs update — Still working")
    assert len([c for c in contents if "Still working" in c]) == 1
