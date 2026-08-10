"""Compatibility checks for the reviewed live Jobs ledger shape.

Every database in this module is temporary.  The live Jobs database is never
opened, mutated, migrated, or copied.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import jobs_db as jdb


def _rows(conn, table):
    return [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def _authoritative_snapshot(conn):
    return {
        table: _rows(conn, table)
        for table in (
            "jobs",
            "job_events",
            "job_attempts",
            "job_receipts",
            "job_origins",
        )
    }


@pytest.fixture
def live_shape_db(tmp_path):
    path = tmp_path / "jobs.db"
    conn = jdb.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS job_origins ("
        "job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE, "
        "origin TEXT NOT NULL, created_at INTEGER NOT NULL)"
    )
    kat_id = jdb.create_job(
        conn,
        name="Historical KAT build",
        goal="preserve me",
        specialist="kat-builder",
    )
    with jdb.write_txn(conn):
        conn.execute(
            "INSERT INTO job_attempts "
            "(id, job_id, specialist, status, failure_class, repository, branch, "
            "worktree, commit_sha, started_at, finished_at, created_at, ordinal, "
            "base_commit, terminal_failure) "
            "VALUES ('a_kat_history', ?, 'kat-builder', 'failed', "
            "'implementation', '/repo with spaces', 'jobs/legacy', NULL, NULL, "
            "1, 2, 1, 1, ?, 1)",
            (kat_id, "a" * 40),
        )
        conn.execute(
            "INSERT INTO job_receipts "
            "(id, job_id, attempt_id, data, idempotency_key, created_at) "
            "VALUES ('rcpt_kat_history', ?, 'a_kat_history', ?, 'legacy-final', 2)",
            (kat_id, json.dumps({"kind": "legacy-kat-failure"})),
        )
        conn.execute(
            "INSERT INTO job_origins (job_id, origin, created_at) VALUES (?, ?, 1)",
            (kat_id, json.dumps({"platform": "api_server", "chat_id": "room-92"})),
        )
    conn.close()
    jdb._INITIALIZED_PATHS.discard(str(path.resolve()))
    return path, kat_id


def test_additive_initialization_preserves_live_ledger_rows(live_shape_db):
    path, _ = live_shape_db
    conn = jdb.connect(path)
    before = _authoritative_snapshot(conn)
    conn.close()
    jdb._INITIALIZED_PATHS.discard(str(path.resolve()))

    reopened = jdb.connect(path)
    after = _authoritative_snapshot(reopened)

    assert after == before
    reopened.close()


def test_historical_kat_lane_attempt_and_receipt_are_not_reassigned(live_shape_db):
    path, kat_id = live_shape_db
    conn = jdb.connect(path)

    job = jdb.get_job(conn, kat_id)
    attempt = jdb.get_attempt(conn, "a_kat_history")
    receipt = conn.execute(
        "SELECT data FROM job_receipts WHERE id = 'rcpt_kat_history'"
    ).fetchone()

    assert job is not None and job.specialist == "kat-builder"
    assert attempt is not None and attempt["specialist"] == "kat-builder"
    assert json.loads(receipt["data"])["kind"] == "legacy-kat-failure"
    conn.close()


def test_create_job_persists_a_bounded_origin_in_the_creation_transaction(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(
        conn,
        name="Origin-aware",
        goal="g",
        origin={
            "platform": "api_server",
            "chat_id": "room-92",
            "session_id": "session-71",
            "credentials": "must-not-be-stored",
        },
    )

    row = conn.execute(
        "SELECT origin FROM job_origins WHERE job_id = ?", (jid,)
    ).fetchone()
    assert json.loads(row["origin"]) == {
        "platform": "api_server",
        "chat_id": "room-92",
        "session_id": "session-71",
        "chat_type": "",
        "thread_id": "",
        "user_id": "",
        "profile": "",
    }
    conn.close()


def test_reassignment_is_explicit_evidenced_and_refuses_active_custody(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(
        conn, name="Explicit move", goal="g", specialist="claude-builder"
    )
    moved = jdb.reassign_specialist(
        conn,
        jid,
        specialist="codex-builder",
        reason="operator approved a configured executor",
    )
    assert moved.specialist == "codex-builder"
    event = [item for item in jdb.get_events(conn, jid) if item["kind"] == "job_reassigned"]
    assert event[-1]["data"] == {
        "from": "claude-builder",
        "to": "codex-builder",
        "reason": "operator approved a configured executor",
    }

    claim = jdb.claim_job(
        conn,
        worker="codex-pc-1",
        specialist="codex-builder",
        job=jid,
        lease_seconds=60,
    )
    assert claim is not None
    with pytest.raises(jdb.InvalidTransition, match="claimed by"):
        jdb.reassign_specialist(
            conn,
            jid,
            specialist="claude-builder",
            reason="must release first",
        )
    conn.close()


def test_unbuilt_job_cannot_claim_finished_complete(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(conn, name="Never built", goal="g")

    with pytest.raises(jdb.InvalidTransition, match="no attempt"):
        jdb.transition(conn, jid, status="finished", step="complete")

    assert jdb.get_job(conn, jid).status == "working"
    conn.close()
