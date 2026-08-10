"""Compatibility checks for the reviewed live Jobs ledger shape.

Every database in this module is temporary.  The live Jobs database is never
opened, mutated, migrated, or copied.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import jobs_db as jdb
from hermes_cli.jobs_identity import effective_identity, resolve_requested_lane


_IDENTITY_COLUMNS = ("requested_lane", "executor", "model")
_HISTORICAL_TABLES = (
    "jobs",
    "job_events",
    "job_attempts",
    "job_receipts",
    "job_origins",
)


def _table_columns(conn, table):
    return tuple(row["name"] for row in conn.execute(f"PRAGMA table_info({table})"))


def _snapshot(conn, columns_by_table):
    result = {}
    for table, columns in columns_by_table.items():
        selected = ", ".join(f'"{column}"' for column in columns)
        result[table] = [
            tuple(row)
            for row in conn.execute(
                f"SELECT {selected} FROM {table} ORDER BY rowid"
            )
        ]
    return result


@pytest.fixture
def live_shape_db(tmp_path):
    path = tmp_path / "jobs.db"
    conn = jdb.connect(path)
    ids = {}
    with jdb.write_txn(conn):
        for number, (slug, specialist) in enumerate(
            (
                ("kat", "kat-builder"),
                ("gpt", "gpt-builder"),
                ("claude", "claude-builder"),
            ),
            start=1,
        ):
            jid = f"j_{slug}_history"
            ids[slug] = jid
            conn.execute(
                "INSERT INTO jobs "
                "(id, number, name, goal, status, step, requested_lane, executor, "
                "specialist, model, routing_reason, created_at, updated_at, "
                "last_heartbeat_at, revision, skills) "
                "VALUES (?, ?, ?, ?, 'working', 'routing', NULL, NULL, ?, NULL, "
                "NULL, 1, 1, NULL, 1, NULL)",
                (
                    jid,
                    number,
                    f"Historical {slug} build",
                    f"preserve {slug}",
                    specialist,
                ),
            )
            conn.execute(
                "INSERT INTO job_events "
                "(job_id, kind, data, idempotency_key, created_at) "
                "VALUES (?, 'job_created', ?, NULL, 1)",
                (jid, json.dumps({"specialist": specialist})),
            )
        for ordinal, (slug, specialist) in enumerate(
            (
                ("kat", "kat-builder"),
                ("gpt", "gpt-builder"),
                ("claude", "claude-builder"),
            ),
            start=1,
        ):
            attempt_id = f"a_{slug}_history"
            conn.execute(
                "INSERT INTO job_attempts "
                "(id, job_id, specialist, status, failure_class, repository, "
                "branch, worktree, commit_sha, started_at, finished_at, "
                "created_at, ordinal, base_commit, terminal_failure) "
                "VALUES (?, ?, ?, 'failed', 'implementation', ?, ?, NULL, NULL, "
                "1, 2, 1, 1, ?, 1)",
                (
                    attempt_id,
                    ids[slug],
                    specialist,
                    f"/repo/{slug}",
                    f"jobs/{slug}",
                    chr(96 + ordinal) * 40,
                ),
            )
            conn.execute(
                "INSERT INTO job_receipts "
                "(id, job_id, attempt_id, data, idempotency_key, created_at) "
                "VALUES (?, ?, ?, ?, ?, 2)",
                (
                    f"rcpt_{slug}_history",
                    ids[slug],
                    attempt_id,
                    json.dumps({"kind": f"legacy-{slug}-failure"}),
                    f"legacy-{slug}-final",
                ),
            )
            conn.execute(
                "INSERT INTO job_origins (job_id, origin, created_at) "
                "VALUES (?, ?, 1)",
                (
                    ids[slug],
                    json.dumps(
                        {"platform": "api_server", "chat_id": f"room-{slug}"}
                    ),
                ),
            )

    # Turn the temporary database into the real pre-identity on-disk shape.
    # This is deliberately verified before current code is allowed to reopen it.
    for column in _IDENTITY_COLUMNS:
        conn.execute(f"ALTER TABLE jobs DROP COLUMN {column}")
    conn.commit()
    columns_by_table = {
        table: _table_columns(conn, table) for table in _HISTORICAL_TABLES
    }
    assert not set(_IDENTITY_COLUMNS) & set(columns_by_table["jobs"])
    before = _snapshot(conn, columns_by_table)
    conn.close()
    jdb._INITIALIZED_PATHS.discard(str(path.resolve()))
    return path, ids, columns_by_table, before


def test_additive_initialization_preserves_live_ledger_rows(live_shape_db):
    path, ids, columns_by_table, before = live_shape_db
    assert not set(_IDENTITY_COLUMNS) & set(columns_by_table["jobs"])

    reopened = jdb.connect(path)
    migrated = {
        row["name"]: row for row in reopened.execute("PRAGMA table_info(jobs)")
    }
    for column in _IDENTITY_COLUMNS:
        assert migrated[column]["type"] == "TEXT"
        assert migrated[column]["notnull"] == 0
        assert migrated[column]["dflt_value"] is None
    assert [
        tuple(row)
        for row in reopened.execute(
            "SELECT requested_lane, executor, model FROM jobs ORDER BY number"
        )
    ] == [(None, None, None)] * len(ids)
    after = _snapshot(reopened, columns_by_table)
    assert after == before
    reopened.close()


def test_historical_kat_lane_attempt_and_receipt_are_not_reassigned(live_shape_db):
    path, ids, _, _ = live_shape_db
    conn = jdb.connect(path)

    job = jdb.get_job(conn, ids["kat"])
    attempt = jdb.get_attempt(conn, "a_kat_history")
    receipt = conn.execute(
        "SELECT data FROM job_receipts WHERE id = 'rcpt_kat_history'"
    ).fetchone()

    assert job is not None and job.specialist == "kat-builder"
    assert attempt is not None and attempt["specialist"] == "kat-builder"
    assert json.loads(receipt["data"])["kind"] == "legacy-kat-failure"
    conn.close()


def test_historical_gpt_builder_row_normalizes_without_rewriting_history(
    live_shape_db,
):
    path, ids, _, _ = live_shape_db
    conn = jdb.connect(path)
    jid = ids["gpt"]

    job = jdb.get_job(conn, jid)
    assert job.specialist == "gpt-builder"
    assert job.requested_lane is None
    assert job.executor is None
    assert job.model is None
    assert effective_identity(job) == resolve_requested_lane("codex")
    row = conn.execute(
        "SELECT requested_lane, executor, specialist, model FROM jobs WHERE id = ?",
        (jid,),
    ).fetchone()
    assert tuple(row) == (None, None, "gpt-builder", None)
    conn.close()


def test_new_job_persists_the_complete_canonical_identity_atomically(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(
        conn,
        name="Canonical Codex build",
        goal="build it",
        requested_lane="codex",
    )

    job = jdb.get_job(conn, jid)
    assert (
        job.requested_lane,
        job.executor,
        job.specialist,
        job.model,
    ) == ("codex", "codex", "codex-builder", "gpt-5.6-sol")
    created = jdb.get_events(conn, jid)[0]
    assert created["data"] == {
        "number": 1,
        "name": "Canonical Codex build",
        "requested_lane": "codex",
        "executor": "codex",
        "specialist": "codex-builder",
        "model": "gpt-5.6-sol",
        "routing_reason": None,
        "skills": None,
    }
    conn.close()


def test_identity_insert_rolls_back_with_the_creation_event(tmp_path, monkeypatch):
    conn = jdb.connect(tmp_path / "jobs.db")

    def fail_event(*args, **kwargs):
        raise RuntimeError("event write failed")

    monkeypatch.setattr(jdb, "_append_event_locked", fail_event)
    with pytest.raises(RuntimeError, match="event write failed"):
        jdb.create_job(
            conn,
            name="Must roll back",
            goal="g",
            requested_lane="claude",
        )

    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    conn.close()


def test_idempotent_intake_persists_lane_and_flags_a_lane_conflict(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    first = jdb.create_or_get_job(
        conn,
        source_type="cron",
        source_key="tick-1",
        name="Scheduled",
        goal="g",
        requested_lane="claude",
    )
    replay = jdb.create_or_get_job(
        conn,
        source_type="cron",
        source_key="tick-1",
        name="Scheduled",
        goal="g",
        requested_lane="codex",
    )

    job = jdb.get_job(conn, first.job_id)
    assert first.created is True
    assert replay.created is False and replay.conflict is True
    assert effective_identity(job) == resolve_requested_lane("claude")
    conn.close()


def test_create_job_persists_a_bounded_origin_in_the_creation_transaction(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(
        conn,
        name="Origin-aware",
        goal="g",
        requested_lane="claude",
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


def _identity_dict(identity):
    return {
        "requested_lane": identity.requested_lane,
        "executor": identity.executor,
        "specialist": identity.specialist,
        "model": identity.model,
    }


@pytest.mark.parametrize(
    ("source_lane", "target_specialist", "target_lane"),
    [
        ("claude", "codex-builder", "codex"),
        ("codex", "claude-builder", "claude"),
    ],
)
def test_reassignment_atomically_moves_the_full_canonical_identity(
    tmp_path, source_lane, target_specialist, target_lane
):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(
        conn, name="Explicit move", goal="g", requested_lane=source_lane
    )
    reason = f"operator moved {source_lane} to {target_lane}"
    moved = jdb.reassign_specialist(
        conn,
        jid,
        specialist=target_specialist,
        reason=reason,
    )
    expected = resolve_requested_lane(target_lane)
    assert _identity_dict(moved) == _identity_dict(expected)
    assert effective_identity(moved) == expected
    event = [item for item in jdb.get_events(conn, jid) if item["kind"] == "job_reassigned"]
    assert event[-1]["data"] == {
        "from": _identity_dict(resolve_requested_lane(source_lane)),
        "to": _identity_dict(expected),
        "reason": reason,
    }
    conn.close()


def test_reassignment_rejects_the_historical_gpt_builder_alias(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(
        conn, name="Alias move", goal="g", requested_lane="claude"
    )
    before = _snapshot(
        conn,
        {
            "jobs": _table_columns(conn, "jobs"),
            "job_events": _table_columns(conn, "job_events"),
        },
    )
    with pytest.raises(ValueError):
        jdb.reassign_specialist(
            conn,
            jid,
            specialist="gpt-builder",
            reason="aliases are read-only",
        )
    after = _snapshot(
        conn,
        {
            "jobs": _table_columns(conn, "jobs"),
            "job_events": _table_columns(conn, "job_events"),
        },
    )
    assert after == before
    conn.close()


@pytest.mark.parametrize(
    ("requested_lane", "specialist"),
    [
        (None, None),
        (None, "claude-builder"),
        ("gpt", None),
        ("kat", None),
        ("unknown", None),
        ("codex", "gpt-builder"),
        ("claude", "kat-builder"),
        ("claude", "codex-builder"),
    ],
)
def test_current_creation_rejects_missing_alias_or_contradictory_identity_without_rows(
    tmp_path, requested_lane, specialist
):
    conn = jdb.connect(tmp_path / "jobs.db")
    before = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("jobs", "job_events", "job_sources")
    }

    with pytest.raises(ValueError):
        jdb.create_job(
            conn,
            name="Refuse",
            goal="g",
            requested_lane=requested_lane,
            specialist=specialist,
        )

    after = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("jobs", "job_events", "job_sources")
    }
    assert after == before
    conn.close()


def test_idempotent_intake_requires_identity_before_source_lookup_or_write(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    first = jdb.create_or_get_job(
        conn,
        source_type="schedule",
        source_key="one",
        name="Canonical",
        goal="g",
        requested_lane="claude",
    )
    before = _snapshot(
        conn,
        {
            table: _table_columns(conn, table)
            for table in ("jobs", "job_events", "job_sources")
        },
    )

    with pytest.raises(ValueError):
        jdb.create_or_get_job(
            conn,
            source_type="schedule",
            source_key="one",
            name="Canonical",
            goal="g",
        )

    assert first.created is True
    assert _snapshot(
        conn,
        {
            table: _table_columns(conn, table)
            for table in ("jobs", "job_events", "job_sources")
        },
    ) == before
    conn.close()


@pytest.mark.parametrize(
    ("requested_lane", "specialist"),
    [
        (None, None),
        ("gpt", None),
        ("kat", None),
        ("unknown", None),
        ("codex", "gpt-builder"),
        ("claude", "kat-builder"),
    ],
)
def test_idempotent_intake_rejects_noncanonical_new_identity_without_rows(
    tmp_path, requested_lane, specialist
):
    conn = jdb.connect(tmp_path / "jobs.db")

    with pytest.raises(ValueError):
        jdb.create_or_get_job(
            conn,
            source_type="schedule",
            source_key="invalid",
            name="Refuse",
            goal="g",
            requested_lane=requested_lane,
            specialist=specialist,
        )

    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM job_sources").fetchone()[0] == 0
    conn.close()


def test_reassignment_rolls_back_identity_when_its_event_fails(
    tmp_path, monkeypatch
):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(
        conn, name="Atomic move", goal="g", requested_lane="claude"
    )

    def fail_event(*args, **kwargs):
        raise RuntimeError("event write failed")

    monkeypatch.setattr(jdb, "_append_event_locked", fail_event)
    with pytest.raises(RuntimeError, match="event write failed"):
        jdb.reassign_specialist(
            conn,
            jid,
            specialist="codex-builder",
            reason="all or nothing",
        )
    assert effective_identity(jdb.get_job(conn, jid)) == resolve_requested_lane(
        "claude"
    )
    conn.close()


@pytest.mark.parametrize("specialist", [None, "", "kat-builder", "unknown-builder"])
def test_reassignment_rejects_retired_or_unknown_targets(tmp_path, specialist):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(
        conn, name="Stay canonical", goal="g", requested_lane="claude"
    )
    before = jdb.get_job(conn, jid)
    with pytest.raises(ValueError):
        jdb.reassign_specialist(
            conn,
            jid,
            specialist=specialist,
            reason="must reject this target",
        )
    after = jdb.get_job(conn, jid)
    assert _identity_dict(after) == _identity_dict(before)
    assert not [
        event
        for event in jdb.get_events(conn, jid)
        if event["kind"] == "job_reassigned"
    ]
    conn.close()


def test_reassignment_refuses_active_custody(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(
        conn, name="Claimed", goal="g", requested_lane="codex"
    )

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
    assert effective_identity(jdb.get_job(conn, jid)) == resolve_requested_lane(
        "codex"
    )
    conn.close()


def test_unbuilt_job_cannot_claim_finished_complete(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    jid = jdb.create_job(
        conn, name="Never built", goal="g", requested_lane="claude"
    )

    with pytest.raises(jdb.InvalidTransition, match="no attempt"):
        jdb.transition(conn, jid, status="finished", step="complete")

    assert jdb.get_job(conn, jid).status == "working"
    conn.close()
