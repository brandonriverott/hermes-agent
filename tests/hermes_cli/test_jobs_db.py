"""Tests for the independent Jobs store (hermes_cli/jobs_db).

The Jobs kernel is a NEW control-plane persistence layer that lives beside the
Kanban modules but shares no code and no database with them. These tests pin
the schema, identity, lifecycle, attempts, receipts, and heartbeat contracts
from the approved Jobs Core V1 plan, and prove the store never touches
``kanban.db``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading

import pytest

from hermes_cli import jobs_db as jdb


REQUIRED_TABLES = {
    "jobs",
    "job_events",
    "job_attempts",
    "job_receipts",
    "job_correlations",
}


@pytest.fixture
def conn(tmp_path):
    c = jdb.connect(db_path=tmp_path / "jobs.db")
    try:
        yield c
    finally:
        c.close()


def _table_names(c) -> set[str]:
    return {
        row["name"]
        for row in c.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _claimed(
    c,
    *,
    name="Build",
    goal="g",
    specialist=None,
    worker="w1",
    lease_seconds=600,
    now=None,
):
    """Create a Job and take custody of it; return ``(job_id, claim_token)``.

    V2 supersedes V1's tokenless attempt calling: running an attempt means
    holding a current, unexpired claim, so every attempt test starts here.
    """
    jid = jdb.create_job(c, name=name, goal=goal, specialist=specialist)
    claim = jdb.claim_job(
        c,
        worker=worker,
        specialist=specialist,
        job=jid,
        lease_seconds=lease_seconds,
        now=now,
    )
    return claim.job.id, claim.claim_token


# ---------------------------------------------------------------------------
# Task 1 — schema
# ---------------------------------------------------------------------------


def test_schema_creates_five_required_tables(conn):
    assert REQUIRED_TABLES <= _table_names(conn)


def test_foreign_keys_enabled(conn):
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_schema_creation_is_idempotent(tmp_path):
    path = tmp_path / "jobs.db"
    c1 = jdb.connect(db_path=path)
    try:
        # Re-running the raw schema script must not raise (CREATE ... IF NOT
        # EXISTS) and must not change the table set.
        c1.executescript(jdb.SCHEMA_SQL)
        assert REQUIRED_TABLES <= _table_names(c1)
    finally:
        c1.close()

    # A second independent connection to the same file sees the same schema.
    c2 = jdb.connect(db_path=path)
    try:
        assert REQUIRED_TABLES <= _table_names(c2)
    finally:
        c2.close()


def test_connect_isolated_from_kanban_db(tmp_path, monkeypatch):
    """Opening the Jobs DB must never create or touch a kanban.db sibling."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Default path resolves under the active Hermes home.
    assert jdb.jobs_db_path() == home / "jobs.db"

    c = jdb.connect()
    try:
        c.execute("SELECT 1")
    finally:
        c.close()

    # jobs.db exists; nothing kanban-shaped was created anywhere under home.
    assert (home / "jobs.db").exists()
    for suffix in ("", "-wal", "-shm"):
        assert not (home / f"kanban.db{suffix}").exists()
    kanban_litter = [
        p.name for p in home.rglob("*") if p.name.startswith("kanban.db")
    ]
    assert kanban_litter == []


# ---------------------------------------------------------------------------
# Task 2 — creation and identity
# ---------------------------------------------------------------------------


def test_create_job_identity_and_initial_state(conn):
    goal = "Ship the thing.\n\n  Keep — verbatim: café \U0001f680 & <tags>."
    jid = jdb.create_job(
        conn,
        name="Ship the thing",
        goal=goal,
        specialist="claude-builder",
        routing_reason="repo scope matches claude lane",
    )
    assert isinstance(jid, str) and jid.startswith("j_")

    job = jdb.get_job(conn, jid)
    assert job is not None
    assert job.id == jid
    assert job.number == 1
    assert job.name == "Ship the thing"
    # Goal is stored byte-for-byte verbatim — no strip, no rewrite.
    assert job.goal == goal
    assert job.status == "working"
    assert job.step == "routing"
    assert job.specialist == "claude-builder"
    assert job.routing_reason == "repo scope matches claude lane"
    assert job.last_heartbeat_at is None


def test_create_job_defaults_optional_fields(conn):
    jid = jdb.create_job(conn, name="Minimal", goal="do it")
    job = jdb.get_job(conn, jid)
    assert job.specialist is None
    assert job.routing_reason is None
    assert job.correlations == []


def test_create_job_appends_job_created_event(conn):
    jid = jdb.create_job(conn, name="X", goal="y")
    events = jdb.get_events(conn, jid)
    assert len(events) == 1
    assert events[0]["kind"] == "job_created"


def test_numbers_are_permanent_and_monotonic(conn):
    j1 = jdb.create_job(conn, name="A", goal="a")
    j2 = jdb.create_job(conn, name="B", goal="b")
    j3 = jdb.create_job(conn, name="C", goal="c")
    assert [jdb.get_job(conn, j).number for j in (j1, j2, j3)] == [1, 2, 3]


def test_empty_name_or_goal_rejected(conn):
    with pytest.raises(ValueError):
        jdb.create_job(conn, name="   ", goal="has goal")
    with pytest.raises(ValueError):
        jdb.create_job(conn, name="has name", goal="")


def test_historical_correlations_recorded(conn):
    jid = jdb.create_job(
        conn, name="Backlog", goal="port card", correlations=["card-123", "card-456"]
    )
    job = jdb.get_job(conn, jid)
    assert sorted(job.correlations) == ["card-123", "card-456"]


def test_resolve_by_number_and_label(conn):
    jid = jdb.create_job(conn, name="Findable", goal="g")
    job = jdb.get_job(conn, jid)
    assert jdb.get_job(conn, str(job.number)).id == jid
    assert jdb.get_job(conn, job.number).id == jid
    assert jdb.get_job(conn, f"Job #{job.number}").id == jid
    assert jdb.get_job(conn, f"#{job.number}").id == jid
    assert jdb.get_job(conn, "j_deadbeef") is None
    assert jdb.get_job(conn, "999") is None


def test_concurrent_connections_never_share_a_number(tmp_path):
    """Two independent connections must allocate distinct, unique numbers."""
    path = tmp_path / "jobs.db"
    # Pre-create the schema once so worker threads only race the INSERT, not
    # schema init.
    jdb.connect(db_path=path).close()

    numbers: list[int] = []
    errors: list[Exception] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        c = jdb.connect(db_path=path)
        try:
            barrier.wait()
            jid = jdb.create_job(c, name=name, goal="race")
            n = jdb.get_job(c, jid).number
            with lock:
                numbers.append(n)
        except Exception as exc:  # pragma: no cover - only on real defect
            with lock:
                errors.append(exc)
        finally:
            c.close()

    threads = [threading.Thread(target=worker, args=(f"job-{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert sorted(numbers) == [1, 2]

    verify = jdb.connect(db_path=path)
    try:
        rows = verify.execute("SELECT number FROM jobs ORDER BY number").fetchall()
        assert [r["number"] for r in rows] == [1, 2]
    finally:
        verify.close()


# ---------------------------------------------------------------------------
# Task 3 — transitions and append-only events
# ---------------------------------------------------------------------------


def _job_in(conn, status: str) -> str:
    """Create a job and move it into ``status`` via a valid path."""
    jid = jdb.create_job(conn, name="T", goal="g")
    if status == "working":
        return jid
    jdb.transition(conn, jid, status=status, step="building")
    return jid


_ALL_PAIRS = [
    (a, b) for a in jdb.PUBLIC_STATUSES for b in jdb.PUBLIC_STATUSES
]
_VALID_PAIRS = [(a, b) for (a, b) in _ALL_PAIRS if b in jdb.VALID_TRANSITIONS[a]]
_INVALID_PAIRS = [(a, b) for (a, b) in _ALL_PAIRS if b not in jdb.VALID_TRANSITIONS[a]]


@pytest.mark.parametrize("start,target", _VALID_PAIRS)
def test_valid_transitions_succeed(conn, start, target):
    jid = _job_in(conn, start)
    jdb.transition(conn, jid, status=target, step="reviewing")
    assert jdb.get_job(conn, jid).status == target
    assert jdb.get_job(conn, jid).step == "reviewing"


@pytest.mark.parametrize("start,target", _INVALID_PAIRS)
def test_invalid_transitions_rejected(conn, start, target):
    jid = _job_in(conn, start)
    before = jdb.get_events(conn, jid)
    with pytest.raises(jdb.InvalidTransition):
        jdb.transition(conn, jid, status=target, step="building")
    # Rejected before any write: status unchanged and no event appended.
    assert jdb.get_job(conn, jid).status == start
    assert jdb.get_events(conn, jid) == before


def test_finished_is_terminal(conn):
    jid = _job_in(conn, "finished")
    for target in ("working", "needs_you"):
        with pytest.raises(jdb.InvalidTransition):
            jdb.transition(conn, jid, status=target, step="building")
    assert jdb.get_job(conn, jid).status == "finished"


def test_transition_rejects_bad_status_or_step(conn):
    jid = jdb.create_job(conn, name="T", goal="g")
    with pytest.raises(ValueError):
        jdb.transition(conn, jid, status="bogus", step="building")
    with pytest.raises(ValueError):
        jdb.transition(conn, jid, status="working", step="bogus")


def test_transition_appends_ordered_event(conn):
    jid = jdb.create_job(conn, name="T", goal="g")
    jdb.transition(
        conn, jid, status="needs_you", step="waiting_for_login", reason="token expired"
    )
    kinds = [e["kind"] for e in jdb.get_events(conn, jid)]
    assert kinds == ["job_created", "job_transition"]
    ev = jdb.get_events(conn, jid)[-1]
    assert ev["data"]["from"] == "working"
    assert ev["data"]["to"] == "needs_you"
    assert ev["data"]["step"] == "waiting_for_login"
    assert ev["data"]["reason"] == "token expired"


def test_set_step_updates_step_and_appends_event(conn):
    jid = jdb.create_job(conn, name="T", goal="g")
    jdb.set_step(conn, jid, "building")
    job = jdb.get_job(conn, jid)
    assert job.status == "working"  # step change never alters public status
    assert job.step == "building"
    kinds = [e["kind"] for e in jdb.get_events(conn, jid)]
    assert kinds == ["job_created", "job_step"]


def test_set_step_rejects_unknown_step(conn):
    jid = jdb.create_job(conn, name="T", goal="g")
    with pytest.raises(ValueError):
        jdb.set_step(conn, jid, "teleporting")


# ---------------------------------------------------------------------------
# Task 4 — attempts and correction lineage
# ---------------------------------------------------------------------------


def test_start_and_finish_attempt_records_fields(conn):
    jid, token = _claimed(conn, specialist="claude-builder")
    aid = jdb.start_attempt(
        conn,
        jid,
        claim_token=token,
        specialist="claude-builder",
        repository="/repo",
        branch="feat/x",
        worktree="/wt/x",
        commit="abc123",
    )
    assert isinstance(aid, str) and aid.startswith("a_")

    att = jdb.get_attempt(conn, aid)
    assert att["job_id"] == jid
    assert att["specialist"] == "claude-builder"
    assert att["repository"] == "/repo"
    assert att["branch"] == "feat/x"
    assert att["worktree"] == "/wt/x"
    assert att["commit"] == "abc123"
    assert att["status"] == "running"
    assert att["started_at"] is not None
    assert att["finished_at"] is None
    assert att["parent_attempt_id"] is None

    jdb.finish_attempt(
        conn, aid, status="failed", failure_class="reviewer_rejection",
        claim_token=token,
    )
    att = jdb.get_attempt(conn, aid)
    assert att["status"] == "failed"
    assert att["failure_class"] == "reviewer_rejection"
    assert att["finished_at"] is not None

    kinds = [e["kind"] for e in jdb.get_events(conn, jid)]
    assert kinds[:3] == ["job_created", "claim_acquired", "attempt_started"]
    assert "attempt_finished" in kinds


def test_correction_attempt_keeps_same_job(conn):
    jid, token = _claimed(conn, specialist="claude-builder")
    original_number = jdb.get_job(conn, jid).number
    a1 = jdb.start_attempt(conn, jid, claim_token=token, specialist="claude-builder")
    jdb.finish_attempt(
        conn, a1, status="failed", failure_class="reviewer_rejection",
        claim_token=token,
    )

    # A reviewer correction is a child attempt on the SAME job, not a new job.
    # The rejection outcome cleared custody, so the corrector re-claims it.
    claim2 = jdb.claim_job(
        conn, worker="w2", specialist="claude-builder", job=jid, lease_seconds=600
    )
    a2 = jdb.start_attempt(
        conn, jid, claim_token=claim2.claim_token, specialist="claude-builder",
        parent_attempt_id=a1,
    )
    att2 = jdb.get_attempt(conn, a2)
    assert att2["job_id"] == jid
    assert att2["parent_attempt_id"] == a1
    assert jdb.get_job(conn, jid).number == original_number
    assert len(jdb.get_attempts(conn, jid)) == 2


def test_parent_attempt_from_another_job_rejected(conn):
    j1, t1 = _claimed(conn, name="One", specialist="claude-builder")
    a1 = jdb.start_attempt(conn, j1, claim_token=t1, specialist="claude-builder")
    j2, t2 = _claimed(conn, name="Two", specialist="claude-builder", worker="w2")
    with pytest.raises(ValueError):
        jdb.start_attempt(
            conn, j2, claim_token=t2, specialist="claude-builder", parent_attempt_id=a1
        )
    # The rejected start left no attempt on job two.
    assert jdb.get_attempts(conn, j2) == []


def test_start_attempt_unknown_job_rejected(conn):
    with pytest.raises(ValueError):
        jdb.start_attempt(conn, "j_deadbeef", specialist="x")


# ---------------------------------------------------------------------------
# Task 5 — receipts and idempotency
# ---------------------------------------------------------------------------


def test_receipt_stores_and_retrieves_structured_json(conn):
    jid, token = _claimed(conn, specialist="claude-builder")
    aid = jdb.start_attempt(conn, jid, claim_token=token, specialist="claude-builder")
    payload = {
        "changed_files": ["a.py", "b/c.py"],
        "commands": [{"cmd": "pytest", "exit": 0}],
        "reviewer_verdict": "pass",
        "activation": "built",
        "unicode": "café \U0001f680",
    }
    rid = jdb.add_receipt(conn, jid, attempt_id=aid, data=payload, idempotency_key="run-1")
    assert isinstance(rid, str) and rid.startswith("r_")

    receipts = jdb.get_receipts(conn, jid)
    assert len(receipts) == 1
    assert receipts[0]["data"] == payload
    assert receipts[0]["attempt_id"] == aid
    # A mutating write appended an event.
    assert "receipt_added" in [e["kind"] for e in jdb.get_events(conn, jid)]


def test_repeated_receipt_idempotency_key_returns_existing(conn):
    jid = jdb.create_job(conn, name="Build", goal="g")
    rid1 = jdb.add_receipt(conn, jid, data={"n": 1}, idempotency_key="k")
    rid2 = jdb.add_receipt(conn, jid, data={"n": 2}, idempotency_key="k")
    assert rid1 == rid2
    receipts = jdb.get_receipts(conn, jid)
    assert len(receipts) == 1
    # First write wins; the duplicate does not overwrite or duplicate.
    assert receipts[0]["data"] == {"n": 1}
    # No duplicate event either.
    assert [e["kind"] for e in jdb.get_events(conn, jid)].count("receipt_added") == 1


def test_same_receipt_key_allowed_on_different_job(conn):
    j1 = jdb.create_job(conn, name="One", goal="g")
    j2 = jdb.create_job(conn, name="Two", goal="g")
    r1 = jdb.add_receipt(conn, j1, data={"x": 1}, idempotency_key="shared")
    r2 = jdb.add_receipt(conn, j2, data={"x": 2}, idempotency_key="shared")
    assert r1 != r2
    assert len(jdb.get_receipts(conn, j1)) == 1
    assert len(jdb.get_receipts(conn, j2)) == 1


def test_repeated_event_idempotency_key_returns_existing(conn):
    jid = jdb.create_job(conn, name="X", goal="g")
    e1 = jdb.append_event(conn, jid, "note", data={"a": 1}, idempotency_key="once")
    e2 = jdb.append_event(conn, jid, "note", data={"a": 2}, idempotency_key="once")
    assert e1["id"] == e2["id"]
    notes = [e for e in jdb.get_events(conn, jid) if e["kind"] == "note"]
    assert len(notes) == 1
    assert notes[0]["data"] == {"a": 1}


def test_same_event_key_allowed_on_different_job(conn):
    j1 = jdb.create_job(conn, name="One", goal="g")
    j2 = jdb.create_job(conn, name="Two", goal="g")
    jdb.append_event(conn, j1, "note", idempotency_key="shared")
    jdb.append_event(conn, j2, "note", idempotency_key="shared")
    assert [e for e in jdb.get_events(conn, j1) if e["kind"] == "note"]
    assert [e for e in jdb.get_events(conn, j2) if e["kind"] == "note"]


def test_receipt_unique_constraint_is_a_real_backstop(conn):
    import sqlite3 as _sqlite3

    jid = jdb.create_job(conn, name="X", goal="g")
    jdb.add_receipt(conn, jid, data={"n": 1}, idempotency_key="k")
    # Bypassing the helper, a raw duplicate (job_id, key) must be refused by the DB.
    with pytest.raises(_sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO job_receipts (id, job_id, attempt_id, data, idempotency_key, created_at) "
            "VALUES ('r_dup', ?, NULL, '{}', 'k', 0)",
            (jid,),
        )


# ---------------------------------------------------------------------------
# Task 6 — heartbeat and stale projection
# ---------------------------------------------------------------------------


def test_heartbeat_sets_timestamp(conn):
    jid = jdb.create_job(conn, name="Build", goal="g")
    assert jdb.get_job(conn, jid).last_heartbeat_at is None
    jdb.heartbeat(conn, jid, at=1000)
    assert jdb.get_job(conn, jid).last_heartbeat_at == 1000
    jdb.heartbeat(conn, jid, at=2000)
    assert jdb.get_job(conn, jid).last_heartbeat_at == 2000


def test_stale_projection_working_over_threshold(conn):
    jid = jdb.create_job(conn, name="Build", goal="g")
    jdb.heartbeat(conn, jid, at=1000)
    # now (1000+61) exceeds heartbeat + threshold(60): stale.
    proj = jdb.projection(conn, jid, now=1061, stale_threshold=60)
    assert proj["stale"] is True
    # within the window: not stale.
    proj = jdb.projection(conn, jid, now=1050, stale_threshold=60)
    assert proj["stale"] is False


def test_stale_requires_heartbeat_and_positive_threshold(conn):
    jid = jdb.create_job(conn, name="Build", goal="g")
    # No heartbeat yet: never stale.
    assert jdb.projection(conn, jid, now=10_000, stale_threshold=60)["stale"] is False
    jdb.heartbeat(conn, jid, at=1000)
    # No threshold supplied: never stale.
    assert jdb.projection(conn, jid, now=10_000)["stale"] is False
    # Non-positive threshold: never stale.
    assert jdb.projection(conn, jid, now=10_000, stale_threshold=0)["stale"] is False
    assert jdb.projection(conn, jid, now=10_000, stale_threshold=-5)["stale"] is False


@pytest.mark.parametrize("status", ["needs_you", "finished"])
def test_non_working_never_projects_stale(conn, status):
    jid = _job_in(conn, status)
    jdb.heartbeat(conn, jid, at=1000)
    proj = jdb.projection(conn, jid, now=1_000_000, stale_threshold=60)
    assert proj["stale"] is False


def test_projection_performs_no_mutation(conn):
    jid = jdb.create_job(conn, name="Build", goal="g")
    jdb.heartbeat(conn, jid, at=1000)

    def snapshot():
        job = jdb.get_job(conn, jid)
        n_events = len(jdb.get_events(conn, jid))
        return (job.status, job.step, job.updated_at, job.last_heartbeat_at, n_events)

    before = snapshot()
    for _ in range(3):
        jdb.projection(conn, jid, now=9_999_999, stale_threshold=60)
    assert snapshot() == before


# ---------------------------------------------------------------------------
# BLOCKER 1 — operation-level idempotency (transition + set_step)
# ---------------------------------------------------------------------------


def test_transition_repeated_key_same_payload_is_noop(conn):
    """Re-issuing the exact same keyed transition writes no second event."""
    jid = jdb.create_job(conn, name="T", goal="g")
    j1 = jdb.transition(conn, jid, status="needs_you", step="building", idempotency_key="k")
    j2 = jdb.transition(conn, jid, status="needs_you", step="building", idempotency_key="k")
    assert j1.status == j2.status == "needs_you"
    trans = [e for e in jdb.get_events(conn, jid) if e["kind"] == "job_transition"]
    assert len(trans) == 1


def test_transition_repeated_key_different_payload_is_noop(conn):
    """A seen key makes the WHOLE repeated op a no-op before any state change.

    Reproduces the blocking defect: first transition working->needs_you with key
    ``k``; a second transition with the same key toward ``finished`` must not
    mutate state and must not diverge state from the append-only ledger.
    """
    jid = jdb.create_job(conn, name="T", goal="g")
    jdb.transition(conn, jid, status="needs_you", step="building", idempotency_key="k")
    jdb.transition(conn, jid, status="finished", step="complete", idempotency_key="k")
    job = jdb.get_job(conn, jid)
    # State stays exactly as the FIRST operation left it.
    assert job.status == "needs_you"
    assert job.step == "building"
    trans = [e for e in jdb.get_events(conn, jid) if e["kind"] == "job_transition"]
    assert len(trans) == 1
    assert trans[0]["data"]["to"] == "needs_you"


def test_set_step_repeated_key_different_payload_is_noop(conn):
    jid = jdb.create_job(conn, name="T", goal="g")
    jdb.set_step(conn, jid, "building", idempotency_key="s")
    jdb.set_step(conn, jid, "testing", idempotency_key="s")  # same key, new payload
    job = jdb.get_job(conn, jid)
    assert job.step == "building"  # first op wins; second is a full no-op
    steps = [e for e in jdb.get_events(conn, jid) if e["kind"] == "job_step"]
    assert len(steps) == 1


# ---------------------------------------------------------------------------
# BLOCKER 2 — transition validation is serialized (no stale ``from``)
# ---------------------------------------------------------------------------


def test_concurrent_transitions_never_emit_stale_from(tmp_path):
    """Two connections transition the same job; every event's ``from`` matches
    the actual serialized predecessor — never a stale pre-transaction value.

    Deterministic assertion (an invariant, not a fixed winner) with a barrier
    and busy-timeout so it cannot deadlock.
    """
    path = tmp_path / "jobs.db"
    c0 = jdb.connect(db_path=path)
    jid = jdb.create_job(c0, name="race", goal="g")
    c0.close()

    barrier = threading.Barrier(2)
    results: dict[str, tuple] = {}
    lock = threading.Lock()

    def worker(name: str, target: str) -> None:
        c = jdb.connect(db_path=path)
        try:
            barrier.wait()
            try:
                jdb.transition(c, jid, status=target, step="building")
                outcome = ("ok", None)
            except jdb.InvalidTransition as exc:  # legal loser after a terminal move
                outcome = ("rejected", str(exc))
            with lock:
                results[name] = outcome
        except Exception as exc:  # pragma: no cover - only on a real defect
            with lock:
                results[name] = ("error", repr(exc))
        finally:
            c.close()

    threads = [
        threading.Thread(target=worker, args=("a", "needs_you")),
        threading.Thread(target=worker, args=("b", "finished")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    verify = jdb.connect(db_path=path)
    try:
        events = [e for e in jdb.get_events(verify, jid) if e["kind"] == "job_transition"]
        final = jdb.get_job(verify, jid)
    finally:
        verify.close()

    # No unexpected errors, both threads ran to completion (no deadlock).
    assert set(results) == {"a", "b"}
    assert all(r[0] != "error" for r in results.values()), results
    # Every emitted transition chains from the real predecessor starting at
    # "working"; two events both claiming from="working" would break this.
    prev = "working"
    for e in events:
        assert e["data"]["from"] == prev, (e, events)
        prev = e["data"]["to"]
    assert final.status == prev
    assert len(events) >= 1


# ---------------------------------------------------------------------------
# BLOCKER 3 — finished jobs are terminal for new attempts
# ---------------------------------------------------------------------------


def test_start_attempt_rejected_on_finished_job(conn):
    jid = jdb.create_job(conn, name="Build", goal="g")
    jdb.transition(conn, jid, status="finished", step="failed")
    events_before = jdb.get_events(conn, jid)
    with pytest.raises(jdb.InvalidTransition):
        jdb.start_attempt(conn, jid, specialist="claude-builder")
    # No running attempt and no event were created by the rejected start.
    assert jdb.get_attempts(conn, jid) == []
    assert jdb.get_events(conn, jid) == events_before


def test_finished_to_finished_still_valid(conn):
    """Terminal finished->finished lifecycle semantics are preserved."""
    jid = jdb.create_job(conn, name="Build", goal="g")
    jdb.transition(conn, jid, status="finished", step="failed")
    jdb.transition(conn, jid, status="finished", step="failed", reason="re-verified")
    assert jdb.get_job(conn, jid).status == "finished"


# ---------------------------------------------------------------------------
# BLOCKER 4 — permanent numbers are never reused
# ---------------------------------------------------------------------------


def test_deleted_number_is_never_reused(conn):
    """A row-level delete must not let a later job reclaim the freed number."""
    j1 = jdb.create_job(conn, name="A", goal="a")
    j2 = jdb.create_job(conn, name="B", goal="b")
    assert jdb.get_job(conn, j1).number == 1
    assert jdb.get_job(conn, j2).number == 2

    conn.execute("DELETE FROM jobs WHERE id = ?", (j2,))
    conn.commit()

    j3 = jdb.create_job(conn, name="C", goal="c")
    assert jdb.get_job(conn, j3).number == 3  # NOT 2


def test_number_allocator_survives_schema_reinit(conn):
    """Re-running the schema script must never reset the monotonic allocator."""
    jdb.create_job(conn, name="A", goal="a")
    jdb.create_job(conn, name="B", goal="b")
    conn.executescript(jdb.SCHEMA_SQL)  # idempotent re-init
    j3 = jdb.create_job(conn, name="C", goal="c")
    assert jdb.get_job(conn, j3).number == 3


def test_opening_early_schema_initializes_allocator(tmp_path):
    """An early V1 DB with jobs but no allocator must seed it from MAX(number)."""
    import sqlite3

    path = tmp_path / "jobs.db"
    raw = sqlite3.connect(str(path))
    try:
        raw.executescript(
            "CREATE TABLE jobs ("
            " id TEXT PRIMARY KEY, number INTEGER NOT NULL UNIQUE, name TEXT NOT NULL,"
            " goal TEXT NOT NULL, status TEXT NOT NULL, step TEXT NOT NULL,"
            " specialist TEXT, routing_reason TEXT, created_at INTEGER NOT NULL,"
            " updated_at INTEGER NOT NULL, last_heartbeat_at INTEGER);"
        )
        raw.execute(
            "INSERT INTO jobs (id, number, name, goal, status, step, created_at, updated_at)"
            " VALUES ('j_old1', 1, 'A', 'a', 'working', 'routing', 0, 0)"
        )
        raw.execute(
            "INSERT INTO jobs (id, number, name, goal, status, step, created_at, updated_at)"
            " VALUES ('j_old2', 2, 'B', 'b', 'finished', 'complete', 0, 0)"
        )
        raw.commit()
    finally:
        raw.close()

    conn = jdb.connect(db_path=path)
    try:
        j3 = jdb.create_job(conn, name="C", goal="c")
        # Continues past the existing max; never reissues 1 or 2.
        assert jdb.get_job(conn, j3).number == 3
        numbers = [r["number"] for r in conn.execute("SELECT number FROM jobs ORDER BY number")]
        assert numbers == [1, 2, 3]
    finally:
        conn.close()


# ===========================================================================
# Jobs Execution V2 — Task 1: schema migration and revision
# ===========================================================================

V2_JOB_COLUMNS = {
    "revision",
    "claimed_by",
    "claim_token",
    "claim_acquired_at",
    "lease_expires_at",
    "current_attempt_id",
}


def _columns(c, table: str) -> set[str]:
    return {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}


def test_v2_jobs_table_has_custody_columns(conn):
    assert V2_JOB_COLUMNS <= _columns(conn, "jobs")


def test_v2_attempts_table_has_ordinal(conn):
    assert "ordinal" in _columns(conn, "job_attempts")


def test_v2_source_registry_table_exists(conn):
    assert "job_sources" in _table_names(conn)


def test_v2_claim_token_never_in_public_job_dict(conn):
    """The claim token is a capability — it must never ride on the Job dict."""
    jid = jdb.create_job(conn, name="Secret", goal="g")
    assert "claim_token" not in jdb.get_job(conn, jid).to_dict()


def test_create_job_starts_at_revision_one(conn):
    jid = jdb.create_job(conn, name="R", goal="g")
    assert jdb.get_job(conn, jid).revision == 1


def test_state_changing_writes_increment_revision(conn):
    jid = jdb.create_job(conn, name="R", goal="g")
    r0 = jdb.get_job(conn, jid).revision
    jdb.transition(conn, jid, status="needs_you", step="building")
    r1 = jdb.get_job(conn, jid).revision
    assert r1 == r0 + 1
    jdb.set_step(conn, jid, "testing")
    r2 = jdb.get_job(conn, jid).revision
    assert r2 == r1 + 1
    jdb.heartbeat(conn, jid, at=123)
    r3 = jdb.get_job(conn, jid).revision
    assert r3 == r2 + 1
    jdb.add_receipt(conn, jid, data={"n": 1})
    r4 = jdb.get_job(conn, jid).revision
    assert r4 == r3 + 1


def test_idempotent_transition_does_not_bump_revision(conn):
    jid = jdb.create_job(conn, name="R", goal="g")
    jdb.transition(conn, jid, status="needs_you", step="building", idempotency_key="k")
    r1 = jdb.get_job(conn, jid).revision
    # Same key toward a different target is a whole-op no-op: revision unchanged.
    jdb.transition(conn, jid, status="finished", step="complete", idempotency_key="k")
    assert jdb.get_job(conn, jid).revision == r1


# ---------------------------------------------------------------------------
# BLOCKER — the ledger is part of the Job aggregate: a public append_event()
# that writes an event must advance the Job's revision exactly once, or an
# observer treats a changed Job as unchanged.
# ---------------------------------------------------------------------------


def test_public_append_event_adds_one_event_and_bumps_revision_once(conn):
    jid = jdb.create_job(conn, name="R", goal="g")
    before = jdb.get_job(conn, jid).revision
    rows_before = len(jdb.get_events(conn, jid))
    jdb.append_event(conn, jid, "review_probe", data={"probe": True})
    assert len(jdb.get_events(conn, jid)) == rows_before + 1
    assert jdb.get_job(conn, jid).revision == before + 1


def test_append_event_failure_leaves_ledger_and_revision_untouched(conn, monkeypatch):
    """Forced failure mid-append: neither the event nor the bump may survive."""
    jid = jdb.create_job(conn, name="R", goal="g")
    before = jdb.get_job(conn, jid).revision
    rows_before = len(jdb.get_events(conn, jid))

    real = jdb._append_event_locked

    def boom(*args, **kwargs):
        real(*args, **kwargs)  # the row really is inserted...
        raise RuntimeError("forced failure")  # ...then the transaction dies

    monkeypatch.setattr(jdb, "_append_event_locked", boom)
    with pytest.raises(RuntimeError):
        jdb.append_event(conn, jid, "review_probe", data={"probe": True})
    monkeypatch.undo()

    assert len(jdb.get_events(conn, jid)) == rows_before
    assert jdb.get_job(conn, jid).revision == before


def test_repeated_event_idempotency_key_does_not_bump_revision(conn):
    jid = jdb.create_job(conn, name="R", goal="g")
    jdb.append_event(conn, jid, "note", data={"a": 1}, idempotency_key="once")
    r1 = jdb.get_job(conn, jid).revision
    e2 = jdb.append_event(conn, jid, "note", data={"a": 2}, idempotency_key="once")
    # Replay is a whole-operation no-op: same event back, revision unchanged.
    assert e2["data"] == {"a": 1}
    assert jdb.get_job(conn, jid).revision == r1


def test_read_paths_do_not_bump_revision(conn):
    jid, _token = _claimed(conn)
    jdb.add_receipt(conn, jid, data={"n": 1})
    before = jdb.get_job(conn, jid).revision
    jdb.get_job(conn, jid)
    jdb.get_events(conn, jid)
    jdb.list_jobs(conn)
    jdb.get_receipts(conn, jid)
    jdb.get_attempts(conn, jid)
    assert jdb.get_job(conn, jid).revision == before


def test_attempt_paths_bump_revision_exactly_once_each(conn):
    """Sibling mutators fold their own bump — appending events must not double it."""
    jid, token = _claimed(conn)
    r0 = jdb.get_job(conn, jid).revision
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    r1 = jdb.get_job(conn, jid).revision
    assert r1 == r0 + 1
    jdb.finish_attempt(conn, aid, status="succeeded", claim_token=token)
    assert jdb.get_job(conn, jid).revision == r1 + 1


def test_concurrent_event_appends_lose_no_event_and_no_revision(tmp_path):
    """Eight writers append concurrently: every event lands and every append
    advances the revision — a lost update would leave the revision short."""
    path = tmp_path / "jobs.db"
    c0 = jdb.connect(db_path=path)
    jid = jdb.create_job(c0, name="race", goal="g")
    r0 = jdb.get_job(c0, jid).revision
    rows0 = len(jdb.get_events(c0, jid))
    c0.close()

    writers = 8
    barrier = threading.Barrier(writers)
    errors: list = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        c = jdb.connect(db_path=path)
        try:
            barrier.wait()
            jdb.append_event(c, jid, "note", data={"n": n})
        except Exception as exc:  # pragma: no cover - only on a real defect
            with lock:
                errors.append(repr(exc))
        finally:
            c.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    verify = jdb.connect(db_path=path)
    try:
        events = jdb.get_events(verify, jid)
        final = jdb.get_job(verify, jid)
    finally:
        verify.close()

    assert errors == []
    notes = [e for e in events if e["kind"] == "note"]
    assert sorted(e["data"]["n"] for e in notes) == list(range(writers))
    assert len(events) == rows0 + writers
    assert final.revision == r0 + writers


def test_v2_migration_adds_columns_to_early_v1_db(tmp_path):
    """An early V1 DB (no custody columns) migrates without losing existing jobs."""
    import sqlite3

    path = tmp_path / "jobs.db"
    raw = sqlite3.connect(str(path))
    try:
        raw.executescript(
            "CREATE TABLE jobs ("
            " id TEXT PRIMARY KEY, number INTEGER NOT NULL UNIQUE, name TEXT NOT NULL,"
            " goal TEXT NOT NULL, status TEXT NOT NULL, step TEXT NOT NULL,"
            " specialist TEXT, routing_reason TEXT, created_at INTEGER NOT NULL,"
            " updated_at INTEGER NOT NULL, last_heartbeat_at INTEGER);"
        )
        raw.execute(
            "INSERT INTO jobs (id, number, name, goal, status, step, created_at, updated_at)"
            " VALUES ('j_old1', 1, 'A', 'a', 'working', 'routing', 0, 0)"
        )
        raw.commit()
    finally:
        raw.close()

    conn = jdb.connect(db_path=path)
    try:
        assert V2_JOB_COLUMNS <= _columns(conn, "jobs")
        assert "ordinal" in _columns(conn, "job_attempts")
        assert "job_sources" in _table_names(conn)
        # Existing job preserved and readable.
        job = jdb.get_job(conn, 1)
        assert job is not None and job.name == "A"
        # A new job continues the number line and gets a revision.
        jid = jdb.create_job(conn, name="B", goal="b")
        assert jdb.get_job(conn, jid).number == 2
        assert jdb.get_job(conn, jid).revision == 1
    finally:
        conn.close()


# --- Early V1 migration: deterministic repair before new unique indexes -----

# The exact pre-V2 schema, so these tests exercise a real historical database
# rather than a V2 database with columns removed.
_EARLY_V1_SQL = """
CREATE TABLE jobs (
    id TEXT PRIMARY KEY, number INTEGER NOT NULL UNIQUE, name TEXT NOT NULL,
    goal TEXT NOT NULL, status TEXT NOT NULL, step TEXT NOT NULL,
    specialist TEXT, routing_reason TEXT, created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL, last_heartbeat_at INTEGER);
CREATE TABLE job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, data TEXT, idempotency_key TEXT,
    created_at INTEGER NOT NULL);
CREATE TABLE job_attempts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    parent_attempt_id TEXT REFERENCES job_attempts(id), specialist TEXT,
    status TEXT NOT NULL, failure_class TEXT, repository TEXT, branch TEXT,
    worktree TEXT, commit_sha TEXT, started_at INTEGER, finished_at INTEGER,
    created_at INTEGER NOT NULL);
CREATE TABLE job_receipts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    attempt_id TEXT REFERENCES job_attempts(id), data TEXT NOT NULL,
    idempotency_key TEXT, created_at INTEGER NOT NULL);
CREATE TABLE job_correlations (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kanban_id TEXT NOT NULL, created_at INTEGER NOT NULL,
    PRIMARY KEY (job_id, kanban_id));
"""


def _seed_early_v1(path, attempts):
    """Write an early V1 DB holding one Job plus its historical attempts.

    Each attempt is ``(id, status, created_at)`` with an optional trailing
    ``parent_attempt_id``. Early V1 has no ``ordinal`` column at all, so the
    parent link is the *only* durable record of execution order these rows
    carry — which is exactly what the migration has to reconstruct from.
    """
    import sqlite3

    raw = sqlite3.connect(str(path))
    try:
        raw.executescript(_EARLY_V1_SQL)
        raw.execute(
            "INSERT INTO jobs (id, number, name, goal, status, step, created_at,"
            " updated_at) VALUES ('j_old1', 1, 'A', 'a', 'working', 'building', 0, 0)"
        )
        for entry in attempts:
            aid, status, created = entry[:3]
            parent = entry[3] if len(entry) > 3 else None
            raw.execute(
                "INSERT INTO job_attempts (id, job_id, parent_attempt_id, status,"
                " created_at, started_at) VALUES (?, 'j_old1', ?, ?, ?, ?)",
                (aid, parent, status, created, created),
            )
        raw.commit()
    finally:
        raw.close()


def test_early_v1_db_with_duplicate_running_attempts_opens(tmp_path):
    """Impossible duplicate-running history must not make the DB unopenable."""
    path = tmp_path / "jobs.db"
    _seed_early_v1(path, [("a_r1", "running", 10), ("a_r2", "running", 20)])
    conn = jdb.connect(db_path=path)  # must not raise IntegrityError
    try:
        assert jdb.get_job(conn, 1) is not None
    finally:
        conn.close()


def test_migration_repairs_duplicate_running_and_preserves_every_row(tmp_path):
    """Duplicates are marked interrupted, never deleted.

    These three rows carry no parent lineage, so nothing durable says which
    worker was still the live one. The repair therefore closes *every* running
    attempt rather than crowning the newest by ``(created_at, id)``: a random
    id and a wall clock cannot hand one worker custody of a Job.
    """
    path = tmp_path / "jobs.db"
    _seed_early_v1(
        path,
        [("a_r1", "running", 10), ("a_r2", "running", 20), ("a_done", "succeeded", 5)],
    )
    conn = jdb.connect(db_path=path)
    try:
        atts = {a["id"]: a for a in jdb.get_attempts(conn, 1)}
        assert set(atts) == {"a_r1", "a_r2", "a_done"}  # every row preserved
        for aid in ("a_r1", "a_r2"):
            assert atts[aid]["status"] == "interrupted"
            assert atts[aid]["failure_class"] == "infrastructure"
            assert atts[aid]["finished_at"] is not None
        assert atts["a_done"]["status"] == "succeeded"  # untouched
        # The repair is recorded in the append-only ledger, once per attempt.
        kinds = [e["kind"] for e in jdb.get_events(conn, 1)]
        assert kinds.count("attempt_finished") == 2
    finally:
        conn.close()


def test_migration_reconstructs_ordinals_from_the_parent_chain(tmp_path):
    """Permanent ordinals come from the lineage, then new attempts continue it.

    The clock and the ids both order these rows ``a_a, a_b, a_c``; the parent
    chain says ``a_c, a_a, a_b``. Only the chain is a record of what actually
    followed what, so only the chain may assign the permanent positions.
    """
    path = tmp_path / "jobs.db"
    _seed_early_v1(
        path,
        [
            ("a_c", "succeeded", 30),
            ("a_a", "succeeded", 10, "a_c"),
            ("a_b", "failed", 20, "a_a"),
        ],
    )
    conn = jdb.connect(db_path=path)
    try:
        assert jdb.get_attempt(conn, "a_c")["ordinal"] == 1
        assert jdb.get_attempt(conn, "a_a")["ordinal"] == 2
        assert jdb.get_attempt(conn, "a_b")["ordinal"] == 3
        # The next real attempt continues the line instead of restarting at 1.
        claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
        aid = jdb.start_attempt(conn, claim.job.id, claim_token=claim.claim_token)
        assert jdb.get_attempt(conn, aid)["ordinal"] == 4
    finally:
        conn.close()


def test_attempt_ordinal_uniqueness_is_database_enforced(conn):
    import sqlite3 as _sqlite3

    jdb.create_job(conn, name="A", goal="g")
    claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
    aid = jdb.start_attempt(conn, claim.job.id, claim_token=claim.claim_token)
    ordinal = jdb.get_attempt(conn, aid)["ordinal"]
    with pytest.raises(_sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO job_attempts (id, job_id, status, created_at, ordinal)"
            " VALUES ('a_dupe', ?, 'failed', 0, ?)",
            (claim.job.id, ordinal),
        )


def test_reopening_a_migrated_db_is_idempotent(tmp_path):
    """A second open repairs nothing, duplicates no migration event, changes nothing."""
    path = tmp_path / "jobs.db"
    _seed_early_v1(
        path, [("a_r1", "running", 10), ("a_r2", "running", 20), ("a_x", "failed", 5)]
    )
    conn = jdb.connect(db_path=path)
    try:
        first_attempts = jdb.get_attempts(conn, 1)
        first_events = jdb.get_events(conn, 1)
    finally:
        conn.close()

    # A fresh process would re-run the migration: clear the per-path init cache.
    jdb._INITIALIZED_PATHS.discard(str(path.resolve()))
    conn = jdb.connect(db_path=path)
    try:
        assert jdb.get_attempts(conn, 1) == first_attempts
        assert jdb.get_events(conn, 1) == first_events
    finally:
        conn.close()


def test_early_v1_settled_attempt_backfills_terminal_failure_as_false(tmp_path):
    """Early V1 rows predate the give-up flag entirely, so ``False`` is history.

    The flag landed in the same change that made the attempt outcome atomic, so
    a settled row carrying no attributed ``job_transition`` was written by code
    that had no flag to record. That absence is evidence, not a default.
    """
    path = tmp_path / "jobs.db"
    _seed_early_v1(path, [("a_x", "failed", 5)])
    conn = jdb.connect(db_path=path)
    try:
        stored = conn.execute(
            "SELECT terminal_failure FROM job_attempts WHERE id = 'a_x'"
        ).fetchone()["terminal_failure"]
        assert stored == 0
        # ...so a replay claiming the Job was given up on conflicts, and a
        # replay of the recorded outcome is still a clean read.
        row = conn.execute(
            "SELECT * FROM job_attempts WHERE id = 'a_x'"
        ).fetchone()
        evidence = dict(commit=None, branch=None, worktree=None, repository=None)
        assert jdb._replay_conflict(
            conn, row, status="failed", failure_class=None,
            terminal_failure=True, evidence=evidence,
        ) is not None
        assert jdb._replay_conflict(
            conn, row, status="failed", failure_class=None,
            terminal_failure=False, evidence=evidence,
        ) is None
    finally:
        conn.close()


# --- Pre-column V2 migration: reconstructing settled terminal outcomes -------

# The exact schema of candidate 3627baca7 — the last build that *applied*
# ``terminal_failure`` to the Job outcome without persisting it on the attempt.
# Frozen deliberately: these tests must keep exercising that real history even
# as the live schema moves on, so the fixture is a copy, never an import.
_PRE_COLUMN_V2_SQL = """
CREATE TABLE jobs (
    id TEXT PRIMARY KEY, number INTEGER NOT NULL UNIQUE, name TEXT NOT NULL,
    goal TEXT NOT NULL, status TEXT NOT NULL, step TEXT NOT NULL,
    specialist TEXT, routing_reason TEXT, created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL, last_heartbeat_at INTEGER,
    revision INTEGER NOT NULL DEFAULT 0, claimed_by TEXT, claim_token TEXT,
    claim_acquired_at INTEGER, lease_expires_at INTEGER,
    current_attempt_id TEXT);
CREATE TABLE job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, data TEXT, idempotency_key TEXT,
    created_at INTEGER NOT NULL);
CREATE TABLE job_attempts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    parent_attempt_id TEXT REFERENCES job_attempts(id), specialist TEXT,
    status TEXT NOT NULL, failure_class TEXT, repository TEXT, branch TEXT,
    worktree TEXT, commit_sha TEXT, started_at INTEGER, finished_at INTEGER,
    created_at INTEGER NOT NULL, ordinal INTEGER);
CREATE TABLE job_receipts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    attempt_id TEXT REFERENCES job_attempts(id), data TEXT NOT NULL,
    idempotency_key TEXT, created_at INTEGER NOT NULL);
CREATE TABLE job_correlations (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kanban_id TEXT NOT NULL, created_at INTEGER NOT NULL,
    PRIMARY KEY (job_id, kanban_id));
CREATE TABLE job_sources (
    source_type TEXT NOT NULL, source_key TEXT NOT NULL,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    created_at INTEGER NOT NULL, PRIMARY KEY (source_type, source_key));
CREATE TABLE job_number_seq (id INTEGER PRIMARY KEY CHECK (id = 1),
    last INTEGER NOT NULL);
CREATE UNIQUE INDEX idx_job_attempts_one_running
    ON job_attempts(job_id) WHERE status = 'running';
CREATE UNIQUE INDEX idx_job_attempts_ordinal ON job_attempts(job_id, ordinal);
"""


def _seed_pre_column_v2(
    path, attempts, *, job, extra_events=(), schema=None, terminal_flags=None
):
    """Write a pre-column V2 DB: one Job plus settled attempts and their lineage.

    Each attempt is ``(id, failure_class, created_at, outcome)`` — always a
    ``failed`` attempt, since that is the only status the give-up flag was ever
    valid on. ``outcome`` is the ``(status, step)`` that attempt atomically moved
    the Job to, which is exactly what candidate 3627baca7 recorded in the
    ``job_transition`` event it wrote in the same transaction as the attempt row.
    That event is the only durable trace the dropped flag ever left. ``None``
    writes no such event (an attempt settled before outcomes were atomic).

    Two optional trailing fields carry the execution lineage the migration has
    to read: ``parent_attempt_id`` (default none) and an explicit ``ordinal``
    (default the position in the list). They exist so a test can hold the
    lineage fixed while varying ``created_at`` and the attempt ids — the two
    things that must never define execution order.

    ``job`` is the Job's ``(status, step)`` *today* — set independently so a
    migration that reads the live Job row instead of the attempt's own event is
    caught. ``extra_events`` are ``(kind, data)`` pairs appended afterwards, for
    the later direct transitions that make an attribution ambiguous. ``schema``
    overrides the DDL, for the one case that needs the pre-index variant.

    ``terminal_flags`` is ``{attempt_id: 0|1}`` and needs the
    :data:`_MIGRATED_V2_SQL` schema: it persists give-up flags an *earlier*
    build already wrote, which is the only way an already-migrated database
    arrives carrying settled flags on history the present code cannot place.
    """
    import sqlite3

    raw = sqlite3.connect(str(path))
    try:
        raw.executescript(schema or _PRE_COLUMN_V2_SQL)
        raw.execute(
            "INSERT INTO jobs (id, number, name, goal, status, step, created_at,"
            " updated_at, revision) VALUES ('j_v2', 1, 'A', 'a', ?, ?, 0, 0, 9)",
            job,
        )
        raw.execute("INSERT INTO job_number_seq (id, last) VALUES (1, 1)")
        for position, entry in enumerate(attempts, 1):
            aid, failure_class, created, outcome = entry[:4]
            parent = entry[4] if len(entry) > 4 else None
            ordinal = entry[5] if len(entry) > 5 else position
            raw.execute(
                "INSERT INTO job_attempts (id, job_id, parent_attempt_id, status,"
                " failure_class, started_at, finished_at, created_at, ordinal)"
                " VALUES (?, 'j_v2', ?, 'failed', ?, ?, ?, ?, ?)",
                (aid, parent, failure_class, created, created, created, ordinal),
            )
            events = [
                ("attempt_started", {"attempt_id": aid, "ordinal": ordinal}),
                (
                    "attempt_finished",
                    {
                        "attempt_id": aid,
                        "status": "failed",
                        "failure_class": failure_class,
                    },
                ),
            ]
            if outcome is not None:
                events.append(
                    (
                        "job_transition",
                        {
                            "from": "working",
                            "to": outcome[0],
                            "step": outcome[1],
                            "reason": f"attempt {aid} failed",
                        },
                    )
                )
            for kind, data in events:
                raw.execute(
                    "INSERT INTO job_events (job_id, kind, data, created_at)"
                    " VALUES ('j_v2', ?, ?, ?)",
                    (kind, json.dumps(data, ensure_ascii=False, sort_keys=True), created),
                )
        for kind, data in extra_events:
            raw.execute(
                "INSERT INTO job_events (job_id, kind, data, created_at)"
                " VALUES ('j_v2', ?, ?, 999)",
                (kind, json.dumps(data, ensure_ascii=False, sort_keys=True)),
            )
        for aid, flag in (terminal_flags or {}).items():
            raw.execute(
                "UPDATE job_attempts SET terminal_failure = ? WHERE id = ?",
                (flag, aid),
            )
        raw.commit()
    finally:
        raw.close()


# The same schema once the give-up column exists — an *already-migrated*
# database, which is the only state that can hold persisted terminal flags on
# history the present-day validator refuses to place. A rejected predecessor
# whose lineage check trusted a contiguous run of ordinals wrote exactly these
# rows, and that build's flags are still on disk after upgrading.
_MIGRATED_V2_SQL = _PRE_COLUMN_V2_SQL.replace(
    "created_at INTEGER NOT NULL, ordinal INTEGER);",
    "created_at INTEGER NOT NULL, ordinal INTEGER, terminal_failure INTEGER);",
)
assert "terminal_failure INTEGER);" in _MIGRATED_V2_SQL

# The refusal a replay earns when the Job's lineage cannot place the settled
# attempt. It is checked *before* any stored column, so it is the reason every
# malformed-history replay fails closed — whatever those columns happen to hold.
_NO_LINEAGE_AUTHORITY = "the Job's execution history cannot place this attempt"


def _stored_terminal_failure(conn, attempt_id):
    return conn.execute(
        "SELECT terminal_failure FROM job_attempts WHERE id = ?", (attempt_id,)
    ).fetchone()["terminal_failure"]


def _snapshot(conn):
    """Every row of every table, so an unrelated mutation cannot hide."""
    tables = [
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]
    return {
        t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t}").fetchall()]
        for t in tables
    }


def test_pre_column_v2_settled_terminal_failure_migrates_true(tmp_path):
    """A settled give-up must survive the migration, or the truth is lost.

    Candidate 3627baca7 finished this attempt with ``terminal_failure=True`` and
    that flag is why the Job is ``finished/failed``. Backfilling it as false
    would reject the one truthful replay and accept a materially false one.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [("a_term", "implementation", 10, ("finished", "failed"))],
        job=("finished", "failed"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _stored_terminal_failure(conn, "a_term") == 1
        before = _snapshot(conn)

        # The truthful replay is a clean read of the settled record.
        replay = jdb.finish_attempt(
            conn, "a_term", status="failed", failure_class="implementation",
            claim_token="stale", terminal_failure=True,
        )
        assert replay["status"] == "failed"
        assert _snapshot(conn) == before  # idempotent: no mutation, no event

        # The materially false replay is refused, still without mutating.
        with pytest.raises(jdb.InvalidTransition):
            jdb.finish_attempt(
                conn, "a_term", status="failed", failure_class="implementation",
                claim_token="stale", terminal_failure=False,
            )
        assert _snapshot(conn) == before
        job = jdb.get_job(conn, 1)
        assert (job.status, job.step) == ("finished", "failed")
    finally:
        conn.close()


def test_pre_column_v2_settled_nonterminal_failure_migrates_false(tmp_path):
    """The same failure class that did *not* give up must migrate as false."""
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [("a_soft", "implementation", 10, ("working", "correcting"))],
        job=("working", "correcting"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _stored_terminal_failure(conn, "a_soft") == 0
        before = _snapshot(conn)
        replay = jdb.finish_attempt(
            conn, "a_soft", status="failed", failure_class="implementation",
            claim_token="stale", terminal_failure=False,
        )
        assert replay["status"] == "failed"
        assert _snapshot(conn) == before
        with pytest.raises(jdb.InvalidTransition):
            jdb.finish_attempt(
                conn, "a_soft", status="failed", failure_class="implementation",
                claim_token="stale", terminal_failure=True,
            )
        assert _snapshot(conn) == before
    finally:
        conn.close()


def test_pre_column_v2_earlier_failure_before_a_later_attempt_is_never_terminal(
    tmp_path,
):
    """Giving up ends the Job, so an attempt with a successor never gave up.

    The Job reads ``finished/failed`` today because of the *second* attempt. A
    migration that attributed the live Job state to every settled row would
    falsify the first one as terminal.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_first", "implementation", 10, ("working", "correcting")),
            ("a_last", "implementation", 20, ("finished", "failed"), "a_first"),
        ],
        job=("finished", "failed"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _stored_terminal_failure(conn, "a_first") == 0
        assert _stored_terminal_failure(conn, "a_last") == 1
        # ...so the earlier attempt only replays as the correction pass it was.
        with pytest.raises(jdb.InvalidTransition):
            jdb.finish_attempt(
                conn, "a_first", status="failed", failure_class="implementation",
                claim_token="stale", terminal_failure=True,
            )
    finally:
        conn.close()


def test_pre_column_v2_reads_the_attempt_event_not_the_present_job_state(tmp_path):
    """A later direct transition must not retro-falsify a settled attempt.

    The attempt's own atomic ``job_transition`` says it handed the Job back for
    correction; something moved the Job on afterwards. Only the attempt's event
    describes what the attempt did.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [("a_soft", "implementation", 10, ("working", "correcting"))],
        job=("finished", "failed"),
        extra_events=[
            (
                "job_transition",
                {
                    "from": "working",
                    "to": "finished",
                    "step": "failed",
                    "reason": "operator closed the job by hand",
                },
            )
        ],
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _stored_terminal_failure(conn, "a_soft") == 0
    finally:
        conn.close()


def test_pre_column_v2_flag_insensitive_failure_class_stays_unknown(tmp_path):
    """Where both flag values produced one outcome, history cannot say which.

    ``reviewer_rejection`` returns the Job for correction whether or not the
    caller gave up, so nothing durable recorded the flag. That row stays unknown
    and *no* replay may settle it — inventing either answer is fabrication.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [("a_rej", "reviewer_rejection", 10, ("working", "correcting"))],
        job=("working", "correcting"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _stored_terminal_failure(conn, "a_rej") is None
        before = _snapshot(conn)
        for guess in (True, False):
            with pytest.raises(jdb.InvalidTransition) as excinfo:
                jdb.finish_attempt(
                    conn, "a_rej", status="failed",
                    failure_class="reviewer_rejection",
                    claim_token="stale", terminal_failure=guess,
                )
            assert "unknown legacy history" in str(excinfo.value)
        assert _snapshot(conn) == before
    finally:
        conn.close()


def test_pre_column_v2_contradictory_attributed_transitions_stay_unknown(tmp_path):
    """Two events claiming the same attempt disagree, so the attribution fails."""
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [("a_x", "implementation", 10, ("working", "correcting"))],
        job=("finished", "failed"),
        extra_events=[
            (
                "job_transition",
                {
                    "from": "working",
                    "to": "finished",
                    "step": "failed",
                    "reason": "attempt a_x failed",
                },
            )
        ],
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _stored_terminal_failure(conn, "a_x") is None
        with pytest.raises(jdb.InvalidTransition):
            jdb.finish_attempt(
                conn, "a_x", status="failed", failure_class="implementation",
                claim_token="stale", terminal_failure=False,
            )
    finally:
        conn.close()


# --- Pre-column V2 migration: successorship comes from execution lineage ----


def test_pre_column_v2_same_second_attempts_follow_ordinals_not_ids(tmp_path):
    """Two attempts can share a second, and the later one can sort first by id.

    Attempt ids are random, so their lexical order is noise. Here the terminal
    successor's id (``a_a_last``) sorts *before* its parent's (``a_z_first``)
    and both were created in the same second, so any ``(created_at, id)`` test
    for "was this attempt followed by another?" answers backwards: it sees a
    successor for the row that has none and falsifies the settled give-up.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_z_first", "implementation", 1000, ("working", "correcting")),
            ("a_a_last", "implementation", 1000, ("finished", "failed"), "a_z_first"),
        ],
        job=("finished", "failed"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _stored_terminal_failure(conn, "a_z_first") == 0
        assert _stored_terminal_failure(conn, "a_a_last") == 1
        before = _snapshot(conn)

        # The truthful replay of the give-up is a clean read of the record...
        replay = jdb.finish_attempt(
            conn, "a_a_last", status="failed", failure_class="implementation",
            claim_token="stale", terminal_failure=True,
        )
        assert replay["status"] == "failed"
        assert _snapshot(conn) == before

        # ...and the materially false one is refused, still without mutating.
        with pytest.raises(jdb.InvalidTransition):
            jdb.finish_attempt(
                conn, "a_a_last", status="failed", failure_class="implementation",
                claim_token="stale", terminal_failure=False,
            )
        assert _snapshot(conn) == before
    finally:
        conn.close()


def test_pre_column_v2_same_second_chain_follows_ordinals_not_ids(tmp_path):
    """A whole same-second chain resolves by ordinal/parent, not by id order.

    All three attempts share one second and their ids descend as the chain
    ascends, so lexical order is the exact reverse of execution order.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_zzz", "implementation", 1000, ("working", "correcting")),
            ("a_mmm", "implementation", 1000, ("working", "correcting"), "a_zzz"),
            ("a_aaa", "implementation", 1000, ("finished", "failed"), "a_mmm"),
        ],
        job=("finished", "failed"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _stored_terminal_failure(conn, "a_zzz") == 0
        assert _stored_terminal_failure(conn, "a_mmm") == 0
        assert _stored_terminal_failure(conn, "a_aaa") == 1
    finally:
        conn.close()


def test_pre_column_v2_distinct_timestamps_still_follow_lineage(tmp_path):
    """Distinct timestamps change nothing: the ordinal/parent chain still rules.

    The clock agrees with the lineage here, and the ids still disagree with
    both, so this pins that the correction reads lineage rather than swapping
    one incidental tiebreak for another.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_zzz", "implementation", 10, ("working", "correcting")),
            ("a_aaa", "implementation", 20, ("finished", "failed"), "a_zzz"),
        ],
        job=("finished", "failed"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _stored_terminal_failure(conn, "a_zzz") == 0
        assert _stored_terminal_failure(conn, "a_aaa") == 1
    finally:
        conn.close()


@pytest.mark.parametrize(
    "attempts",
    [
        pytest.param(
            [
                ("a_one", "implementation", 10, ("working", "correcting")),
                ("a_two", "implementation", 20, ("finished", "failed"), "a_gone"),
            ],
            id="parent-is-not-an-attempt-of-this-job",
        ),
        pytest.param(
            [
                ("a_one", "implementation", 10, ("working", "correcting")),
                ("a_two", "implementation", 20, ("working", "correcting"), "a_one"),
                ("a_three", "implementation", 30, ("finished", "failed"), "a_one"),
            ],
            id="two-attempts-claim-the-same-parent",
        ),
        pytest.param(
            [
                ("a_one", "implementation", 10, ("working", "correcting"), "a_two"),
                ("a_two", "implementation", 20, ("finished", "failed"), "a_one"),
            ],
            id="parent-chain-is-cyclic",
        ),
        pytest.param(
            [
                ("a_one", "implementation", 10, ("working", "correcting"), None, 1),
                ("a_three", "implementation", 30, ("finished", "failed"), "a_one", 3),
            ],
            id="ordinals-are-gapped-so-a-row-is-missing",
        ),
    ],
)
def test_pre_column_v2_malformed_lineage_stays_unknown(tmp_path, attempts):
    """Lineage that cannot be trusted leaves every row unknown, never guessed.

    A dangling parent, a forked or cyclic chain, and a gap in the permanent
    ordinals all mean the durable record of execution order is incomplete or
    self-contradictory. Successorship is then unattributable, so the whole Job's
    settled rows stay NULL and no replay may settle them — refused on the
    lineage itself, before a stored column is ever consulted.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(path, attempts, job=("finished", "failed"))
    conn = jdb.connect(db_path=path)
    try:
        for entry in attempts:
            assert _stored_terminal_failure(conn, entry[0]) is None
        before = _snapshot(conn)
        for guess in (True, False):
            with pytest.raises(jdb.InvalidTransition) as excinfo:
                jdb.finish_attempt(
                    conn, attempts[-1][0], status="failed",
                    failure_class="implementation",
                    claim_token="stale", terminal_failure=guess,
                )
            assert _NO_LINEAGE_AUTHORITY in str(excinfo.value)
        assert _snapshot(conn) == before
    finally:
        conn.close()


def test_fresh_v2_attempts_store_an_explicit_terminal_failure_value(conn):
    """Nullable is only for unreconstructible history — new writes are explicit."""
    for terminal, expected in ((True, 1), (False, 0)):
        jdb.create_job(conn, name="A", goal="g")
        claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
        aid = jdb.start_attempt(conn, claim.job.id, claim_token=claim.claim_token)
        # A running attempt has no terminal outcome yet, so it is not "false".
        assert _stored_terminal_failure(conn, aid) is None
        jdb.finish_attempt(
            conn, aid, status="failed", failure_class="implementation",
            claim_token=claim.claim_token, terminal_failure=terminal,
        )
        assert _stored_terminal_failure(conn, aid) == expected


def test_pre_column_v2_migration_is_repeatable_and_changes_nothing_twice(tmp_path):
    """Re-opening must reach the same verdicts without appending or rewriting.

    Includes a row the evidence cannot settle, so the migration keeps reopening
    its write transaction — which still has to leave the database byte-identical.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_soft", "implementation", 10, ("working", "correcting")),
            ("a_rej", "reviewer_rejection", 20, ("working", "correcting"), "a_soft"),
        ],
        job=("working", "correcting"),
    )
    conn = jdb.connect(db_path=path)
    try:
        first = _snapshot(conn)
        assert _stored_terminal_failure(conn, "a_soft") == 0
        assert _stored_terminal_failure(conn, "a_rej") is None
    finally:
        conn.close()

    # A fresh process would re-run the migration: clear the per-path init cache.
    jdb._INITIALIZED_PATHS.discard(str(path.resolve()))
    conn = jdb.connect(db_path=path)
    try:
        assert _snapshot(conn) == first
    finally:
        conn.close()


# --- Legacy ordinals are reconstructed from parent lineage, never the clock -

# The pre-column schema as it looked *before* the per-Job ordinal unique index
# existed — the state a migration that never reached index creation leaves
# behind, and the only way one Job can hold two attempts claiming one ordinal.
_PRE_INDEX_V2_SQL = _PRE_COLUMN_V2_SQL.replace(
    "CREATE UNIQUE INDEX idx_job_attempts_ordinal ON job_attempts(job_id, ordinal);",
    "",
)
assert "idx_job_attempts_ordinal" not in _PRE_INDEX_V2_SQL


def _seed_null_ordinal_chain(path, ids, *, created=None):
    """A pre-column V2 DB whose real chain never had its ordinals written.

    ``ids`` are the attempts in true execution order: each is the child of the
    one before it, every ``ordinal`` is NULL, and only the last one gave up on
    the Job. ``created`` overrides the per-attempt clock (one shared second by
    default) so a test can vary the two things that must never define execution
    order — the random attempt ids and the timestamps — while the durable
    lineage stays fixed.
    """
    stamps = list(created) if created else [1000] * len(ids)
    entries = []
    parent = None
    for position, (aid, stamp) in enumerate(zip(ids, stamps), 1):
        outcome = ("finished", "failed") if position == len(ids) else (
            "working", "correcting"
        )
        entries.append((aid, "implementation", stamp, outcome, parent, None))
        parent = aid
    _seed_pre_column_v2(path, entries, job=("finished", "failed"))


def _migrated_chain(conn, job_id="j_v2"):
    """``(id, ordinal, terminal_failure, parent)`` for one Job, in read order."""
    return [
        (r["id"], r["ordinal"], r["terminal_failure"], r["parent_attempt_id"])
        for r in conn.execute(
            "SELECT id, ordinal, terminal_failure, parent_attempt_id "
            "FROM job_attempts WHERE job_id = ? "
            "ORDER BY ordinal IS NULL ASC, ordinal ASC, id ASC",
            (job_id,),
        ).fetchall()
    ]


def _replay_verdict(conn, attempt_id, terminal_failure):
    """Replay a settled failure and report ``(accepted?, database unmoved?)``.

    A replay is either a clean read of the settled record or a refusal; either
    way the database must not move, so both halves are reported together.
    """
    before = _snapshot(conn)
    try:
        jdb.finish_attempt(
            conn, attempt_id, status="failed", failure_class="implementation",
            claim_token="stale", terminal_failure=terminal_failure,
        )
        accepted = "ACCEPTED"
    except jdb.InvalidTransition:
        accepted = "REJECTED"
    return accepted, _snapshot(conn) == before


def _reopen(path):
    """Reconnect as a fresh process would, re-running the whole migration."""
    jdb._INITIALIZED_PATHS.discard(str(path.resolve()))
    return jdb.connect(db_path=path)


def test_equivalent_legacy_databases_migrate_identically_either_way_ids_sort(
    tmp_path,
):
    """The reviewer's two databases: one lineage, opposite id order, one verdict.

    Both hold the identical same-second execution history with NULL ordinals —
    a root that handed the Job back for correction, and its child that gave up.
    The only difference is which random ``a_`` id happens to sort first. Filling
    the missing ordinals from ``(created_at, id)`` made the first database
    authoritative and left the second permanently unknown, so a random id
    decided whether a settled give-up could be replayed at all.
    """
    verdicts = []
    for ids in (("a_first", "a_second"), ("a_z_first", "a_a_last")):
        path = tmp_path / f"{ids[0]}.db"
        _seed_null_ordinal_chain(path, ids)
        conn = jdb.connect(db_path=path)
        try:
            root, leaf = ids
            verdicts.append(
                {
                    "root": (
                        jdb.get_attempt(conn, root)["ordinal"],
                        _stored_terminal_failure(conn, root),
                    ),
                    "leaf": (
                        jdb.get_attempt(conn, leaf)["ordinal"],
                        _stored_terminal_failure(conn, leaf),
                    ),
                    "truthful_replay": _replay_verdict(conn, leaf, True),
                    "false_replay": _replay_verdict(conn, leaf, False),
                }
            )
        finally:
            conn.close()

    assert verdicts[0] == verdicts[1]
    assert verdicts[0] == {
        "root": (1, 0),  # followed by another attempt, so it never gave up
        "leaf": (2, 1),  # the settled give-up survives the migration
        "truthful_replay": ("ACCEPTED", True),
        "false_replay": ("REJECTED", True),
    }


def test_legacy_null_ordinals_come_from_parents_not_same_second_ids(tmp_path):
    """A whole same-second chain whose ids descend as execution ascends."""
    path = tmp_path / "jobs.db"
    _seed_null_ordinal_chain(path, ("a_zzz", "a_mmm", "a_aaa"))
    conn = jdb.connect(db_path=path)
    try:
        assert _migrated_chain(conn) == [
            ("a_zzz", 1, 0, None),
            ("a_mmm", 2, 0, "a_zzz"),
            ("a_aaa", 3, 1, "a_mmm"),
        ]
    finally:
        conn.close()


def test_legacy_null_ordinals_do_not_depend_on_the_order_rows_come_back(tmp_path):
    """The chain is rebuilt from the graph, not from however SQLite hands it over.

    These rows are inserted leaf first, so an unordered ``SELECT`` returns the
    execution chain backwards. Reading positions off the result set — or off any
    ``ORDER BY`` that is not the ordinal itself — would number it in reverse.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_leaf", "implementation", 1000, ("finished", "failed"), "a_mid", None),
            ("a_mid", "implementation", 1000, ("working", "correcting"), "a_root", None),
            ("a_root", "implementation", 1000, ("working", "correcting"), None, None),
        ],
        job=("finished", "failed"),
    )
    conn = jdb.connect(db_path=path)
    try:
        # Insertion order really is the reverse of execution order.
        assert [
            r["id"] for r in conn.execute("SELECT id FROM job_attempts").fetchall()
        ] == ["a_leaf", "a_mid", "a_root"]
        assert _migrated_chain(conn) == [
            ("a_root", 1, 0, None),
            ("a_mid", 2, 0, "a_root"),
            ("a_leaf", 3, 1, "a_mid"),
        ]
    finally:
        conn.close()


def test_legacy_null_ordinals_ignore_a_clock_that_disagrees_with_the_chain(tmp_path):
    """Distinct timestamps cannot move a chain position, in either direction.

    The root here carries the *later* ``created_at`` — a clock step, an NTP
    correction, a caller passing its own ``now``. The parent link is unmoved,
    so the reconstructed positions are unmoved.
    """
    path = tmp_path / "jobs.db"
    _seed_null_ordinal_chain(
        path, ("a_root", "a_mid", "a_leaf"), created=[3000, 1000, 2000]
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _migrated_chain(conn) == [
            ("a_root", 1, 0, None),
            ("a_mid", 2, 0, "a_root"),
            ("a_leaf", 3, 1, "a_mid"),
        ]
    finally:
        conn.close()


def test_legacy_mixed_known_and_missing_ordinals_fill_only_the_gaps(tmp_path):
    """A known ordinal that agrees with the chain is kept, never rewritten."""
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), None, None),
            ("a_two", "implementation", 1000, ("working", "correcting"), "a_one", 2),
            ("a_three", "implementation", 1000, ("finished", "failed"), "a_two", None),
        ],
        job=("finished", "failed"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _migrated_chain(conn) == [
            ("a_one", 1, 0, None),
            ("a_two", 2, 0, "a_one"),
            ("a_three", 3, 1, "a_two"),
        ]
    finally:
        conn.close()


def test_legacy_ordinal_contradicting_the_chain_leaves_the_unknowns_null(tmp_path):
    """A known ordinal that disagrees with the chain settles nothing at all.

    ``a_two`` is the chain's second attempt but its stored ordinal says third.
    One of the two records is wrong and the row cannot say which, so the
    migration rewrites neither: the known value stands untouched, the unknown
    rows stay unknown, and every replay on that Job fails closed — the
    contradiction denies the whole Job's lineage any authority to place a row.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), None, None),
            ("a_two", "implementation", 1000, ("working", "correcting"), "a_one", 3),
            ("a_three", "implementation", 1000, ("finished", "failed"), "a_two", None),
        ],
        job=("finished", "failed"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _migrated_chain(conn) == [
            ("a_two", 3, None, "a_one"),  # known history, left exactly as found
            ("a_one", None, None, None),
            ("a_three", None, None, "a_two"),
        ]
        before = _snapshot(conn)
        for guess in (True, False):
            with pytest.raises(jdb.InvalidTransition) as excinfo:
                jdb.finish_attempt(
                    conn, "a_three", status="failed",
                    failure_class="implementation",
                    claim_token="stale", terminal_failure=guess,
                )
            assert _NO_LINEAGE_AUTHORITY in str(excinfo.value)
        assert _snapshot(conn) == before
    finally:
        conn.close()

    conn = _reopen(path)
    try:
        assert _snapshot(conn) == before  # repeatable: the second open changes nothing
    finally:
        conn.close()


def _seed_foreign_job_attempt(path, attempt_id):
    """Add a second Job with one attempt, so a cross-Job parent can be seeded."""
    import sqlite3

    raw = sqlite3.connect(str(path))
    try:
        raw.execute(
            "INSERT INTO jobs (id, number, name, goal, status, step, created_at,"
            " updated_at, revision) VALUES ('j_other', 2, 'B', 'b', 'working',"
            " 'building', 0, 0, 0)"
        )
        raw.execute(
            "INSERT INTO job_attempts (id, job_id, status, created_at, ordinal)"
            " VALUES (?, 'j_other', 'succeeded', 1000, 1)",
            (attempt_id,),
        )
        raw.commit()
    finally:
        raw.close()


@pytest.mark.parametrize(
    "attempts, foreign_attempt",
    [
        pytest.param(
            [
                ("a_one", "implementation", 1000, ("working", "correcting"), None, None),
                ("a_two", "implementation", 1000, ("working", "correcting"), "a_one", None),
                ("a_three", "implementation", 1000, ("finished", "failed"), "a_one", None),
            ],
            None,
            id="fork-two-attempts-continue-one",
        ),
        pytest.param(
            [
                ("a_one", "implementation", 1000, ("working", "correcting"), "a_two", None),
                ("a_two", "implementation", 1000, ("finished", "failed"), "a_one", None),
            ],
            None,
            id="cycle-so-there-is-no-root",
        ),
        pytest.param(
            [
                ("a_one", "implementation", 1000, ("working", "correcting"), None, None),
                ("a_two", "implementation", 1000, ("finished", "failed"), "a_two", None),
            ],
            None,
            id="self-parent",
        ),
        pytest.param(
            [
                ("a_one", "implementation", 1000, ("working", "correcting"), None, None),
                ("a_two", "implementation", 1000, ("finished", "failed"), "a_gone", None),
            ],
            None,
            id="dangling-parent",
        ),
        pytest.param(
            [
                ("a_one", "implementation", 1000, ("working", "correcting"), None, None),
                (
                    "a_two", "implementation", 1000, ("finished", "failed"),
                    "a_elsewhere", None,
                ),
            ],
            "a_elsewhere",
            id="parent-belongs-to-another-job",
        ),
        pytest.param(
            [
                ("a_one", "implementation", 1000, ("working", "correcting"), None, None),
                ("a_two", "implementation", 1000, ("finished", "failed"), None, None),
            ],
            None,
            id="two-roots-so-the-graph-is-disconnected",
        ),
    ],
)
def test_legacy_unreconstructible_lineage_leaves_every_ordinal_null(
    tmp_path, attempts, foreign_attempt
):
    """No single chain covers these attempts, so no position is knowable.

    Each shape is a different way the durable record of "what followed what"
    contradicts itself. Guessing an order from the clock or the random ids is
    exactly the fabrication this migration must refuse, so every ordinal stays
    NULL, every settled outcome stays unknown, both replay values are refused
    for want of a lineage that can place the row, and re-opening reaches the
    same verdict without appending or rewriting anything.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(path, attempts, job=("finished", "failed"))
    if foreign_attempt is not None:
        _seed_foreign_job_attempt(path, foreign_attempt)

    conn = jdb.connect(db_path=path)
    try:
        assert _migrated_chain(conn) == sorted(
            (entry[0], None, None, entry[4]) for entry in attempts
        )
        before = _snapshot(conn)
        for guess in (True, False):
            with pytest.raises(jdb.InvalidTransition) as excinfo:
                jdb.finish_attempt(
                    conn, attempts[-1][0], status="failed",
                    failure_class="implementation",
                    claim_token="stale", terminal_failure=guess,
                )
            assert _NO_LINEAGE_AUTHORITY in str(excinfo.value)
        assert _snapshot(conn) == before
    finally:
        conn.close()

    conn = _reopen(path)
    try:
        assert _snapshot(conn) == before
    finally:
        conn.close()


# --- Known contiguous ordinals never vouch for the parent graph -------------

# The pre-column schema as it looked before the one-running partial index — the
# only state in which one Job can hold two ``running`` rows *and* the permanent
# ordinals that make them look like a settled chain.
_PRE_RUNNING_INDEX_V2_SQL = _PRE_COLUMN_V2_SQL.replace(
    "CREATE UNIQUE INDEX idx_job_attempts_one_running\n"
    "    ON job_attempts(job_id) WHERE status = 'running';",
    "",
)
assert "idx_job_attempts_one_running" not in _PRE_RUNNING_INDEX_V2_SQL


def _seed_known_ordinal_running(path, attempts):
    """A pre-column V2 DB whose ``running`` rows already carry their ordinals.

    ``attempts`` are ``(id, parent_attempt_id, ordinal)``, all left ``running``
    on one Job and all sharing one second — the wreck two interrupted workers
    leave behind, on a database far enough along to have numbered them.
    """
    import sqlite3

    raw = sqlite3.connect(str(path))
    try:
        raw.executescript(_PRE_RUNNING_INDEX_V2_SQL)
        raw.execute(
            "INSERT INTO jobs (id, number, name, goal, status, step, created_at,"
            " updated_at, revision) VALUES ('j_v2', 1, 'A', 'a', 'working',"
            " 'building', 0, 0, 9)"
        )
        raw.execute("INSERT INTO job_number_seq (id, last) VALUES (1, 1)")
        for aid, parent, ordinal in attempts:
            raw.execute(
                "INSERT INTO job_attempts (id, job_id, parent_attempt_id, status,"
                " started_at, created_at, ordinal)"
                " VALUES (?, 'j_v2', ?, 'running', 1000, 1000, ?)",
                (aid, parent, ordinal),
            )
        raw.commit()
    finally:
        raw.close()


# Every shape below carries a *complete* run of permanent ordinals ``1..N`` —
# the one thing a contiguity check accepts on sight — over a parent graph that
# is not one execution history.
_KNOWN_ORDINAL_MALFORMED = [
    pytest.param(
        [
            ("a_root_one", "implementation", 10, ("working", "correcting"), None, 1),
            ("a_root_two", "implementation", 20, ("finished", "failed"), None, 2),
        ],
        None,
        id="two-parentless-roots-numbered-one-and-two",
    ),
    pytest.param(
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), None, 1),
            ("a_two", "implementation", 1000, ("working", "correcting"), "a_one", 2),
            ("a_three", "implementation", 1000, ("finished", "failed"), "a_one", 3),
        ],
        None,
        id="fork-two-attempts-continue-one",
    ),
    pytest.param(
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), "a_two", 1),
            ("a_two", "implementation", 1000, ("finished", "failed"), "a_one", 2),
        ],
        None,
        id="cycle-so-there-is-no-root",
    ),
    pytest.param(
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), None, 1),
            ("a_two", "implementation", 1000, ("finished", "failed"), "a_two", 2),
        ],
        None,
        id="self-parent",
    ),
    pytest.param(
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), None, 1),
            ("a_two", "implementation", 1000, ("finished", "failed"), "a_gone", 2),
        ],
        None,
        id="dangling-parent",
    ),
    pytest.param(
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), None, 1),
            (
                "a_two", "implementation", 1000, ("finished", "failed"),
                "a_elsewhere", 2,
            ),
        ],
        "a_elsewhere",
        id="parent-belongs-to-another-job",
    ),
    pytest.param(
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), None, 1),
            ("a_two", "implementation", 1000, ("working", "correcting"), "a_one", 2),
            ("a_three", "implementation", 1000, ("working", "correcting"), None, 3),
            ("a_four", "implementation", 1000, ("finished", "failed"), "a_three", 4),
        ],
        None,
        id="two-chains-so-a-component-sits-off-the-lineage",
    ),
    pytest.param(
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), None, 1),
            ("a_two", "implementation", 1000, ("working", "correcting"), "a_one", 3),
            ("a_three", "implementation", 1000, ("finished", "failed"), "a_two", 2),
        ],
        None,
        id="ordinals-contradict-the-order-the-chain-proves",
    ),
]


@pytest.mark.parametrize("attempts, foreign_attempt", _KNOWN_ORDINAL_MALFORMED)
def test_known_contiguous_ordinals_do_not_settle_a_malformed_lineage(
    tmp_path, attempts, foreign_attempt
):
    """A complete run of ordinals is not a history. The parent graph decides.

    ``sorted(ordinals) == 1..N`` only says the numbers are complete; it says
    nothing about whether the rows carrying them are one execution chain. Two
    parentless roots numbered 1 and 2 are two disconnected histories that
    happen to be numbered, and a fork, a cycle, a dangling or cross-Job parent
    and an off-chain component are all just as unplaceable with the numbers
    filled in as without them. So every settled outcome stays unknown, both
    replay values are refused without moving the database, the permanent
    ordinals are left exactly as found, and re-opening reaches the same verdict.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(path, attempts, job=("finished", "failed"))
    if foreign_attempt is not None:
        _seed_foreign_job_attempt(path, foreign_attempt)

    conn = jdb.connect(db_path=path)
    try:
        assert jdb._validated_ordinals(conn, "j_v2") is None
        # Every row survives with its stored ordinal untouched; the give-up
        # flag is the only column the migration could have written, and did not.
        assert _migrated_chain(conn) == sorted(
            ((entry[0], entry[5], None, entry[4]) for entry in attempts),
            key=lambda row: row[1],
        )
        before = _snapshot(conn)
        for guess in (True, False):
            with pytest.raises(jdb.InvalidTransition) as excinfo:
                jdb.finish_attempt(
                    conn, attempts[-1][0], status="failed",
                    failure_class="implementation",
                    claim_token="stale", terminal_failure=guess,
                )
            assert _NO_LINEAGE_AUTHORITY in str(excinfo.value)
        assert _snapshot(conn) == before
    finally:
        conn.close()

    conn = _reopen(path)
    try:
        assert _snapshot(conn) == before
    finally:
        conn.close()


def test_known_contiguous_duplicate_running_roots_crown_no_winner(tmp_path):
    """Numbered duplicate-running history still names no live attempt.

    Both rows are parentless roots, so nothing says which worker still holds
    the Job — the ordinals only say the two rows were numbered, and crowning
    the higher one hands custody to whichever wreck happened to be numbered
    last. Every running attempt is closed with its own repair evidence, every
    row and permanent ordinal survives, the one-running index can finally be
    created, and re-opening repairs nothing twice.
    """
    path = tmp_path / "jobs.db"
    _seed_known_ordinal_running(
        path, [("a_root_one", None, 1), ("a_root_two", None, 2)]
    )
    conn = jdb.connect(db_path=path)
    try:
        atts = {a["id"]: a for a in jdb.get_attempts(conn, 1)}
        assert set(atts) == {"a_root_one", "a_root_two"}  # every row preserved
        assert [a["status"] for a in atts.values()] == ["interrupted"] * 2
        assert {a["failure_class"] for a in atts.values()} == {"infrastructure"}
        # Permanent identity is never rewritten to make the repair tidy.
        assert (atts["a_root_one"]["ordinal"], atts["a_root_two"]["ordinal"]) == (1, 2)
        finished = [
            e for e in jdb.get_events(conn, 1) if e["kind"] == "attempt_finished"
        ]
        assert {e["data"]["attempt_id"] for e in finished} == set(atts)
        assert {e["data"]["reason"] for e in finished} == {"migration_repair"}
        assert "idx_job_attempts_one_running" in {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        before = _snapshot(conn)
    finally:
        conn.close()

    conn = _reopen(path)
    try:
        assert _snapshot(conn) == before
    finally:
        conn.close()


def test_start_attempt_refuses_known_contiguous_multiple_roots(tmp_path):
    """Nothing may be appended to a numbered history that is not one chain.

    The ordinals run 1..2, so a contiguity check would hand the next attempt
    "3" — a permanent position asserting it follows attempts that no lineage
    can place. It fails closed, writing neither an attempt nor an event.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_root_one", "implementation", 10, ("working", "correcting"), None, 1),
            ("a_root_two", "implementation", 20, ("working", "correcting"), None, 2),
        ],
        job=("working", "correcting"),
    )
    conn = jdb.connect(db_path=path)
    try:
        claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
        before = _snapshot(conn)
        with pytest.raises(jdb.InvalidTransition) as excinfo:
            jdb.start_attempt(conn, claim.job.id, claim_token=claim.claim_token)
        assert "legacy" in str(excinfo.value)
        assert _snapshot(conn) == before  # no attempt row, no event, no revision
    finally:
        conn.close()


def test_known_contiguous_linear_chain_stays_authoritative(tmp_path):
    """The same known ordinals over a real chain keep every bit of authority.

    Nothing here is fail-closed: the graph proves one history, the stored
    ordinals agree with it position for position, so the settled outcomes
    reconstruct and the Job goes on numbering monotonically from its leaf.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), None, 1),
            ("a_two", "implementation", 1000, ("working", "correcting"), "a_one", 2),
            ("a_three", "implementation", 1000, ("working", "correcting"), "a_two", 3),
        ],
        job=("working", "correcting"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _migrated_chain(conn) == [
            ("a_one", 1, 0, None),
            ("a_two", 2, 0, "a_one"),
            ("a_three", 3, 0, "a_two"),
        ]
        claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
        aid = jdb.start_attempt(conn, claim.job.id, claim_token=claim.claim_token)
        att = jdb.get_attempt(conn, aid)
        assert (att["ordinal"], att["parent_attempt_id"]) == (4, "a_three")
    finally:
        conn.close()


# --- A persisted give-up flag is evidence, never replay authority -----------


def _db_bytes(path):
    """Every byte SQLite holds for this database, journal files included."""
    return {p.name: p.read_bytes() for p in sorted(path.parent.glob(path.name + "*"))}


def _predecessor_flags(attempts):
    """The give-up flags a lineage-trusting predecessor left on these rows.

    That build read a complete run of ordinals as a history, so every attempt
    but the highest-numbered one looked followed-by-another and settled as 0,
    and the last one carried the give-up that moved the Job to
    ``finished``/``failed`` — a stored 1. On the two-parentless-root database
    that is exactly the ``0``/``1`` pair the reviewer read back.
    """
    last = max(attempts, key=lambda entry: entry[5])[0]
    return {entry[0]: (1 if entry[0] == last else 0) for entry in attempts}


@pytest.mark.parametrize("attempts, foreign_attempt", _KNOWN_ORDINAL_MALFORMED)
def test_persisted_give_up_flags_never_settle_a_malformed_lineage(
    tmp_path, attempts, foreign_attempt
):
    """A stored flag records a belief; only lineage confers replay authority.

    This is the already-migrated database, not a pre-column one: a predecessor
    that mistook a run of ordinals for a history persisted an explicit 0 or 1
    on every one of these rows, and upgrading does not (and must not) erase
    them. Under the present validator the graph proves nothing, so no attempt
    can be placed at its stored position, and *both* replay values are refused
    on every row — including the row whose stored flag matches the guess
    exactly. Anything less would let a discredited build's belief hand back a
    settled result the current record cannot vouch for.

    The flags are left exactly as found. They are the evidence of what those
    rows were; rewriting them to manufacture an answer would destroy it.
    """
    path = tmp_path / "jobs.db"
    flags = _predecessor_flags(attempts)
    _seed_pre_column_v2(
        path, attempts, job=("finished", "failed"),
        schema=_MIGRATED_V2_SQL, terminal_flags=flags,
    )
    if foreign_attempt is not None:
        _seed_foreign_job_attempt(path, foreign_attempt)

    conn = jdb.connect(db_path=path)
    try:
        assert jdb._validated_ordinals(conn, "j_v2") is None
        # The predecessor's flags survived the upgrade — this really is the
        # state where a matching stored value is available to be trusted.
        assert {
            entry[0]: _stored_terminal_failure(conn, entry[0]) for entry in attempts
        } == flags
        assert set(flags.values()) == {0, 1}
        before, before_bytes = _snapshot(conn), _db_bytes(path)
        for entry in attempts:
            for guess in (True, False):
                with pytest.raises(jdb.InvalidTransition) as excinfo:
                    jdb.finish_attempt(
                        conn, entry[0], status="failed",
                        failure_class="implementation",
                        claim_token="stale", terminal_failure=guess,
                    )
                assert _NO_LINEAGE_AUTHORITY in str(excinfo.value)
        # Atomic refusal: not a row, not a revision, not an event, not a byte.
        assert _snapshot(conn) == before
        assert _db_bytes(path) == before_bytes
    finally:
        conn.close()

    conn = _reopen(path)
    try:
        assert _snapshot(conn) == before
    finally:
        conn.close()


def test_a_placed_attempt_replays_only_the_flag_it_actually_settled_with(tmp_path):
    """The same persisted flags, over a real chain, keep every bit of authority.

    Nothing here is fail-closed. One root, each attempt continuing the last,
    stored ordinals agreeing position for position — so the lineage places both
    rows and the settled record may finally speak: the truthful replay is a
    clean read of it, and the opposite value conflicts on the flag itself
    rather than on the lineage, without moving the database.
    """
    path = tmp_path / "jobs.db"
    attempts = [
        ("a_one", "implementation", 10, ("working", "correcting"), None, 1),
        ("a_two", "implementation", 20, ("finished", "failed"), "a_one", 2),
    ]
    _seed_pre_column_v2(
        path, attempts, job=("finished", "failed"),
        schema=_MIGRATED_V2_SQL, terminal_flags={"a_one": 0, "a_two": 1},
    )
    conn = jdb.connect(db_path=path)
    try:
        assert jdb._validated_ordinals(conn, "j_v2") == {"a_one": 1, "a_two": 2}
        before = _snapshot(conn)
        for aid, settled in (("a_one", False), ("a_two", True)):
            with pytest.raises(jdb.InvalidTransition) as excinfo:
                jdb.finish_attempt(
                    conn, aid, status="failed", failure_class="implementation",
                    claim_token="stale", terminal_failure=not settled,
                )
            assert "terminal_failure" in str(excinfo.value)
            assert _NO_LINEAGE_AUTHORITY not in str(excinfo.value)
            assert _snapshot(conn) == before  # a conflict moves nothing either
            replay = jdb.finish_attempt(
                conn, aid, status="failed", failure_class="implementation",
                claim_token="stale", terminal_failure=settled,
            )
            assert replay["id"] == aid
            assert _snapshot(conn) == before  # idempotent: a read, not a write
    finally:
        conn.close()


def test_reconstructed_reversed_id_chain_replays_idempotently(tmp_path):
    """Positions the migration *derived* carry the same authority as stored ones.

    The ids run backwards against the lineage and both attempts share one
    second, so nothing but the parent graph can order them. The migration fills
    the ordinals from that graph, and the rows are then placed exactly as if
    they had been numbered all along: the truthful replay stays idempotent and
    the false one is refused on the flag, never on the lineage.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_zzz", "implementation", 1000, ("working", "correcting"), None, None),
            ("a_aaa", "implementation", 1000, ("finished", "failed"), "a_zzz", None),
        ],
        job=("finished", "failed"),
    )
    conn = jdb.connect(db_path=path)
    try:
        assert _migrated_chain(conn) == [
            ("a_zzz", 1, 0, None),
            ("a_aaa", 2, 1, "a_zzz"),
        ]
        before = _snapshot(conn)
        for _ in range(2):  # replaying twice is still one settled record
            replay = jdb.finish_attempt(
                conn, "a_aaa", status="failed", failure_class="implementation",
                claim_token="stale", terminal_failure=True,
            )
            assert (replay["id"], replay["ordinal"]) == ("a_aaa", 2)
            assert _stored_terminal_failure(conn, "a_aaa") == 1
            assert _snapshot(conn) == before
        with pytest.raises(jdb.InvalidTransition) as excinfo:
            jdb.finish_attempt(
                conn, "a_aaa", status="failed", failure_class="implementation",
                claim_token="stale", terminal_failure=False,
            )
        assert _NO_LINEAGE_AUTHORITY not in str(excinfo.value)
        assert _snapshot(conn) == before
    finally:
        conn.close()


def test_legacy_duplicate_known_ordinals_fail_closed_and_the_db_still_opens(tmp_path):
    """Two attempts claiming one position cannot both be right, so neither moves.

    Rewriting a permanent ordinal to make a chain fit would destroy the only
    record of what the row actually was, so the duplicates stand and the
    unknown row stays unknown. The cost is named: this database can never gain
    the per-Job ordinal unique index, because creating it would require
    deleting or rewriting one of the two conflicting rows. Every row survives,
    the database still opens, and :func:`start_attempt` refuses to append a new
    permanent position after history this incoherent.
    """
    path = tmp_path / "jobs.db"
    _seed_pre_column_v2(
        path,
        [
            ("a_one", "implementation", 1000, ("working", "correcting"), None, 1),
            ("a_two", "implementation", 1000, ("working", "correcting"), "a_one", 1),
            ("a_three", "implementation", 1000, ("finished", "failed"), "a_two", None),
        ],
        job=("finished", "failed"),
        schema=_PRE_INDEX_V2_SQL,
    )
    conn = jdb.connect(db_path=path)  # must not raise IntegrityError
    try:
        assert _migrated_chain(conn) == [
            ("a_one", 1, None, None),
            ("a_two", 1, None, "a_one"),
            ("a_three", None, None, "a_two"),
        ]
        # The one-running invariant is still database-enforced regardless.
        indexes = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert "idx_job_attempts_one_running" in indexes
        assert "idx_job_attempts_ordinal" not in indexes
        before = _snapshot(conn)
    finally:
        conn.close()

    conn = _reopen(path)
    try:
        assert _snapshot(conn) == before
    finally:
        conn.close()


def test_legacy_duplicate_running_is_settled_by_lineage_not_by_the_clock(tmp_path):
    """Valid lineage names the live attempt, and it is not the newest row.

    The leaf carries the *earlier* ``created_at`` and the first-sorting id, so
    keeping "the newest by ``(created_at, id)``" would interrupt the attempt
    the chain says is current and leave a superseded one holding the slot.
    """
    path = tmp_path / "jobs.db"
    _seed_early_v1(
        path,
        [("a_z_root", "running", 2000), ("a_a_leaf", "running", 1000, "a_z_root")],
    )
    conn = jdb.connect(db_path=path)
    try:
        atts = {a["id"]: a for a in jdb.get_attempts(conn, 1)}
        assert set(atts) == {"a_z_root", "a_a_leaf"}  # every row preserved
        assert (atts["a_a_leaf"]["ordinal"], atts["a_a_leaf"]["status"]) == (
            2, "running",
        )
        assert (atts["a_z_root"]["ordinal"], atts["a_z_root"]["status"]) == (
            1, "interrupted",
        )
        assert atts["a_z_root"]["failure_class"] == "infrastructure"
        kinds = [e["kind"] for e in jdb.get_events(conn, 1)]
        assert kinds.count("attempt_finished") == 1
        before = _snapshot(conn)
    finally:
        conn.close()

    conn = _reopen(path)
    try:
        assert _snapshot(conn) == before
    finally:
        conn.close()


def test_legacy_ambiguous_duplicate_running_grants_no_worker_the_slot(tmp_path):
    """With no lineage, neither running attempt may be crowned by its id.

    The same two-worker wreck is migrated twice, differing only in which random
    id sorts first. Both times every running attempt is interrupted with its
    own repair evidence, every row survives, and the one-running unique index
    is still created — the invariant is restored without picking a winner.
    """
    verdicts = []
    for ids in (("a_first", "a_second"), ("a_z_first", "a_a_last")):
        path = tmp_path / f"{ids[0]}.db"
        _seed_early_v1(path, [(ids[0], "running", 1000), (ids[1], "running", 1000)])
        conn = jdb.connect(db_path=path)
        try:
            atts = {a["id"]: a for a in jdb.get_attempts(conn, 1)}
            assert set(atts) == set(ids)  # every row preserved
            verdicts.append(
                {
                    ids.index(aid): (
                        a["status"], a["failure_class"], a["ordinal"],
                    )
                    for aid, a in atts.items()
                }
            )
            assert "idx_job_attempts_one_running" in {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                ).fetchall()
            }
            # Exact repair evidence, one keyed event per closed attempt.
            finished = [
                e for e in jdb.get_events(conn, 1) if e["kind"] == "attempt_finished"
            ]
            assert {e["data"]["attempt_id"] for e in finished} == set(ids)
            assert {e["data"]["reason"] for e in finished} == {"migration_repair"}
        finally:
            conn.close()

    assert verdicts[0] == verdicts[1]
    assert verdicts[0] == {
        0: ("interrupted", "infrastructure", None),
        1: ("interrupted", "infrastructure", None),
    }


def test_start_attempt_refuses_a_job_whose_legacy_positions_are_unknown(tmp_path):
    """No new permanent ordinal may be appended after unknown history.

    Two parentless legacy attempts leave no chain to reconstruct, so positions
    1 and 2 were never established. Handing the next attempt "3" would assert
    an execution order the database cannot support, so the start fails closed
    and writes nothing at all.
    """
    path = tmp_path / "jobs.db"
    _seed_early_v1(path, [("a_x", "succeeded", 10), ("a_y", "succeeded", 20)])
    conn = jdb.connect(db_path=path)
    try:
        claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
        before = _snapshot(conn)
        with pytest.raises(jdb.InvalidTransition) as excinfo:
            jdb.start_attempt(conn, claim.job.id, claim_token=claim.claim_token)
        assert "legacy" in str(excinfo.value)
        assert _snapshot(conn) == before  # no attempt row, no event, no revision
    finally:
        conn.close()


def test_start_attempt_still_continues_a_reconstructible_legacy_history(tmp_path):
    """A legacy history the chain *can* place keeps allocating monotonically."""
    path = tmp_path / "jobs.db"
    _seed_early_v1(
        path, [("a_root", "failed", 10), ("a_leaf", "failed", 20, "a_root")]
    )
    conn = jdb.connect(db_path=path)
    try:
        claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
        aid = jdb.start_attempt(
            conn, claim.job.id, claim_token=claim.claim_token,
            parent_attempt_id="a_leaf",
        )
        assert jdb.get_attempt(conn, aid)["ordinal"] == 3
    finally:
        conn.close()


def test_start_attempt_continues_the_chain_when_no_parent_is_named(conn):
    """A retry that names no parent still continues the one execution chain.

    Nothing else keeps the Job at one root. A second parentless attempt would
    be a second history that merely happens to be numbered after the first, and
    since the lineage — not the numbering — is what grants replay authority and
    the next permanent position, that Job could never name its leaf again.
    """
    jid, token = _claimed(conn, name="A")
    ids = []
    for round_ in range(3):
        if round_:
            token = jdb.claim_job(
                conn, worker=f"w{round_}", job=jid, lease_seconds=600
            ).claim_token
        aid = jdb.start_attempt(conn, jid, claim_token=token)
        jdb.finish_attempt(
            conn, aid, status="failed", failure_class="implementation",
            claim_token=token,
        )
        ids.append(aid)
    assert [
        (a["id"], a["ordinal"], a["parent_attempt_id"])
        for a in jdb.get_attempts(conn, jid)
    ] == [
        (ids[0], 1, None),
        (ids[1], 2, ids[0]),
        (ids[2], 3, ids[1]),
    ]
    assert jdb._lineage_leaf(conn, jid) == ids[2]


def test_start_attempt_refuses_a_parent_that_is_not_the_lineage_leaf(conn):
    """A new attempt may only continue the latest one, never fork the history.

    Naming a superseded parent would write the fork by hand — history no
    lineage could place afterwards, which would cost the Job its own settled
    outcomes. Rejected outright, with nothing written.
    """
    jid, token = _claimed(conn, name="A")
    first = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(
        conn, first, status="failed", failure_class="implementation",
        claim_token=token,
    )
    token = jdb.claim_job(
        conn, worker="w2", job=jid, lease_seconds=600
    ).claim_token
    second = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(
        conn, second, status="failed", failure_class="implementation",
        claim_token=token,
    )
    token = jdb.claim_job(
        conn, worker="w3", job=jid, lease_seconds=600
    ).claim_token

    before = _snapshot(conn)
    with pytest.raises(ValueError):
        jdb.start_attempt(
            conn, jid, claim_token=token, parent_attempt_id=first
        )
    assert _snapshot(conn) == before


# ===========================================================================
# Jobs Execution V2 — Task 2: atomic claims
# ===========================================================================


def test_claim_acquires_oldest_eligible_job_first(conn):
    j1 = jdb.create_job(conn, name="A", goal="a")
    jdb.create_job(conn, name="B", goal="b")
    claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
    assert claim is not None
    assert claim.job.id == j1  # oldest permanent number wins
    assert claim.job.claimed_by == "w1"
    assert claim.job.lease_expires_at is not None
    # The token is a real capability returned only here.
    assert isinstance(claim.claim_token, str) and len(claim.claim_token) >= 16


def test_claim_token_absent_from_list_show_events(conn):
    claim = None
    jdb.create_job(conn, name="A", goal="a")
    claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
    jid = claim.job.id
    # Never on the public dict.
    assert "claim_token" not in jdb.get_job(conn, jid).to_dict()
    assert all("claim_token" not in j.to_dict() for j in jdb.list_jobs(conn))
    # Never in the event ledger.
    blob = json.dumps(jdb.get_events(conn, jid))
    assert claim.claim_token not in blob


def test_claim_repr_and_str_never_carry_the_token(conn):
    """The capability must not ride on the ordinary debug representation.

    ``repr()`` and ``str()`` land in tracebacks, log lines, and a bare
    ``print(claim)``; the token is handed over exactly once, through the
    explicit field the custody flow reads.
    """
    jdb.create_job(conn, name="A", goal="a")
    claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
    token = claim.claim_token
    assert token  # still returned programmatically
    assert token not in repr(claim)
    assert token not in str(claim)
    # The Job it carries is safe to show, so the repr stays useful.
    assert claim.job.id in repr(claim)
    # ...and legitimate custody still works off the field.
    aid = jdb.start_attempt(conn, claim.job.id, claim_token=claim.claim_token)
    assert jdb.get_attempt(conn, aid)["status"] == "running"


def test_claim_skips_needs_you_and_finished(conn):
    jdb.create_job(conn, name="A", goal="a")  # will go needs_you
    jdb.transition(conn, 1, status="needs_you", step="waiting_for_decision")
    jdb.create_job(conn, name="B", goal="b")  # will go finished
    jdb.transition(conn, 2, status="finished", step="failed")
    assert jdb.claim_job(conn, worker="w1", lease_seconds=60) is None


def test_claim_requires_executable_step(conn):
    jid = jdb.create_job(conn, name="A", goal="a")
    # Working but parked on a non-executable step: not claimable.
    jdb.set_step(conn, jid, "complete")
    assert jdb.claim_job(conn, worker="w1", lease_seconds=60) is None


def test_claim_matches_specialist(conn):
    jdb.create_job(conn, name="A", goal="a", specialist="claude-builder")
    jdb.create_job(conn, name="B", goal="b", specialist="codex-builder")
    claim = jdb.claim_job(conn, worker="cx", specialist="codex-builder", lease_seconds=60)
    assert claim is not None and claim.job.number == 2  # skips the claude one


def test_claim_unassigned_routing_work(conn):
    jdb.create_job(conn, name="Routed", goal="a", specialist="claude-builder")
    jid = jdb.create_job(conn, name="Unrouted", goal="b")  # specialist is None
    claim = jdb.claim_job(conn, worker="router", lease_seconds=60)  # specialist=None
    assert claim is not None and claim.job.id == jid
    # A specialist request must not pick up unassigned routing work.
    jdb.create_job(conn, name="AlsoUnrouted", goal="c")
    assert jdb.claim_job(conn, worker="cx", specialist="codex-builder", lease_seconds=60) is None


def test_claim_never_steals_an_active_claim(conn):
    jdb.create_job(conn, name="A", goal="a")
    first = jdb.claim_job(conn, worker="w1", lease_seconds=60)
    assert first is not None
    # No other eligible job, and the held claim is not expired → nothing to give.
    assert jdb.claim_job(conn, worker="w2", lease_seconds=60) is None


def test_claim_explicit_job_by_number(conn):
    jdb.create_job(conn, name="A", goal="a")
    j2 = jdb.create_job(conn, name="B", goal="b")
    claim = jdb.claim_job(conn, worker="w1", job=2, lease_seconds=60)
    assert claim is not None and claim.job.id == j2


def test_claim_rejects_bad_lease(conn):
    jdb.create_job(conn, name="A", goal="a")
    for bad in (0, -5, jdb.MAX_LEASE_SECONDS + 1):
        with pytest.raises(ValueError):
            jdb.claim_job(conn, worker="w1", lease_seconds=bad)


def test_concurrent_claim_gives_one_job_to_one_worker(tmp_path):
    """Two workers race for the single eligible job; exactly one wins it."""
    path = tmp_path / "jobs.db"
    c0 = jdb.connect(db_path=path)
    jdb.create_job(c0, name="only", goal="g")
    c0.close()

    barrier = threading.Barrier(2)
    results: dict[str, object] = {}
    lock = threading.Lock()

    def worker(name: str) -> None:
        c = jdb.connect(db_path=path)
        try:
            barrier.wait()
            claim = jdb.claim_job(c, worker=name, lease_seconds=60)
            with lock:
                results[name] = claim
        except Exception as exc:  # pragma: no cover - only on real defect
            with lock:
                results[name] = exc
        finally:
            c.close()

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert all(not isinstance(r, Exception) for r in results.values()), results
    winners = [r for r in results.values() if r is not None]
    assert len(winners) == 1  # exactly one claim succeeded
    # And the DB shows a single claim on the one job.
    verify = jdb.connect(db_path=path)
    try:
        rows = verify.execute(
            "SELECT claimed_by FROM jobs WHERE claim_token IS NOT NULL"
        ).fetchall()
        assert len(rows) == 1
    finally:
        verify.close()


# ===========================================================================
# Jobs Execution V2 — Task 3: attempt custody
# ===========================================================================


def test_attempt_ordinals_are_permanent_and_per_job(conn):
    j1, t1 = _claimed(conn, name="A")
    a1 = jdb.start_attempt(conn, j1, claim_token=t1)
    jdb.finish_attempt(
        conn, a1, status="failed", failure_class="implementation", claim_token=t1
    )
    # A recoverable implementation failure hands the Job back for correction.
    c2 = jdb.claim_job(conn, worker="w2", job=j1, lease_seconds=600)
    a2 = jdb.start_attempt(
        conn, j1, claim_token=c2.claim_token, parent_attempt_id=a1
    )
    assert jdb.get_attempt(conn, a1)["ordinal"] == 1
    assert jdb.get_attempt(conn, a2)["ordinal"] == 2
    # A different job restarts ordinals at 1.
    j2, t2 = _claimed(conn, name="B", worker="w3")
    b1 = jdb.start_attempt(conn, j2, claim_token=t2)
    assert jdb.get_attempt(conn, b1)["ordinal"] == 1


def _chain(c, monkeypatch, entries):
    """Run a real correction chain and return the job id.

    ``entries`` are ``(attempt_id, created_at)`` in true execution order. Each
    attempt is started with that exact id and clock, then finished as a reviewer
    rejection so the next one is its child — a genuine lineage with permanent
    ordinals ``1..N``, written through the public API only. Pinning both the id
    and the clock is the point: those are the two things that must never decide
    the order the history is read back in.
    """
    queue = [aid for aid, _ in entries]
    monkeypatch.setattr(jdb, "_new_attempt_id", lambda: queue.pop(0))
    jid = jdb.create_job(c, name="Build", goal="g", specialist="claude-builder")
    parent = None
    for _, at in entries:
        claim = jdb.claim_job(
            c, worker="w1", specialist="claude-builder", job=jid,
            lease_seconds=600, now=at,
        )
        aid = jdb.start_attempt(
            c, jid, claim_token=claim.claim_token, specialist="claude-builder",
            parent_attempt_id=parent, now=at,
        )
        jdb.finish_attempt(
            c, aid, status="failed", failure_class="reviewer_rejection",
            claim_token=claim.claim_token, now=at,
        )
        parent = aid
    return jid


def _read_order(c, jid):
    """The public history as ``(id, ordinal, parent)`` in the order returned."""
    return [
        (a["id"], a["ordinal"], a["parent_attempt_id"])
        for a in jdb.get_attempts(c, jid)
    ]


def test_get_attempts_same_second_reversed_ids_follow_ordinals(conn, monkeypatch):
    """Two attempts in one second, the successor's id sorting first.

    ``a_`` ids are random, so their lexical order is noise. Ordering the public
    history by ``(created_at, id)`` hands a reader the correction *before* the
    attempt it corrects — the chain reads backwards, and every consumer that
    trusts position (an adapter, a reviewer, recovery) is misled.
    """
    jid = _chain(conn, monkeypatch, [("a_z_first", 1000), ("a_a_last", 1000)])
    assert _read_order(conn, jid) == [
        ("a_z_first", 1, None),
        ("a_a_last", 2, "a_z_first"),
    ]
    # The whole reproduction lives inside one second: nothing but the ordinal
    # could have put these rows in execution order.
    assert {a["created_at"] for a in jdb.get_attempts(conn, jid)} == {1000}


def test_get_attempts_three_same_second_adversarial_ids_follow_ordinals(
    conn, monkeypatch
):
    """A whole same-second chain whose ids descend as execution ascends."""
    jid = _chain(
        conn, monkeypatch, [("a_zzz", 1000), ("a_mmm", 1000), ("a_aaa", 1000)]
    )
    assert _read_order(conn, jid) == [
        ("a_zzz", 1, None),
        ("a_mmm", 2, "a_zzz"),
        ("a_aaa", 3, "a_mmm"),
    ]


def test_get_attempts_ignore_a_clock_that_disagrees_with_the_lineage(
    conn, monkeypatch
):
    """Distinct timestamps cannot override the ordinal, in either direction.

    The ids agree with execution order here, so the clock is the only thing
    disagreeing. A correction can genuinely carry an earlier ``created_at`` — a
    clock step, an NTP correction, a caller passing its own ``now`` — and the
    permanent ordinal still decides.
    """
    jid = _chain(conn, monkeypatch, [("a_first", 3000), ("a_second", 1000)])
    assert _read_order(conn, jid) == [
        ("a_first", 1, None),
        ("a_second", 2, "a_first"),
    ]


def test_get_attempts_put_unknown_ordinals_last_and_never_interleaved(
    conn, monkeypatch
):
    """A row with no ordinal has no execution position, and cannot claim one.

    This is the legacy row whose place the migration could not prove from the
    parent chain and deliberately left unknown. It is real evidence, so it is
    never dropped; it just cannot be placed in a sequence it has no place in.
    Such rows sort after every known ordinal, in a stable id order that is a
    tiebreak and nothing more: the ``None`` ordinal each one carries is what
    tells the reader its execution position is unknown.
    """
    jid = _chain(
        conn,
        monkeypatch,
        [("a_one", 1000), ("a_two", 1000), ("a_zzz_lost", 1000), ("a_aaa_lost", 1000)],
    )
    conn.execute(
        "UPDATE job_attempts SET ordinal = NULL "
        "WHERE id IN ('a_zzz_lost', 'a_aaa_lost')"
    )
    assert _read_order(conn, jid) == [
        ("a_one", 1, None),
        ("a_two", 2, "a_one"),
        ("a_aaa_lost", None, "a_zzz_lost"),
        ("a_zzz_lost", None, "a_two"),
    ]


def test_only_one_running_attempt_enforced_by_db(conn):
    jid, token = _claimed(conn, name="A")
    jdb.start_attempt(conn, jid, claim_token=token)
    with pytest.raises(jdb.InvalidTransition):
        jdb.start_attempt(conn, jid, claim_token=token)  # refused by the DB
    assert len(jdb.get_attempts(conn, jid)) == 1


def test_start_sets_and_finish_clears_current_attempt(conn):
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    assert jdb.get_job(conn, jid).current_attempt_id == aid
    jdb.finish_attempt(conn, aid, status="succeeded", claim_token=token)
    assert jdb.get_job(conn, jid).current_attempt_id is None


def test_attempt_start_requires_valid_claim_token(conn):
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    assert jdb.get_attempt(conn, aid)["status"] == "running"
    # A max-turns stop hands the same Job back for another correction pass.
    jdb.finish_attempt(
        conn, aid, status="failed", failure_class="max_turns", claim_token=token
    )
    jdb.claim_job(conn, worker="w2", job=jid, lease_seconds=600)
    # A wrong token fails closed — no new attempt.
    with pytest.raises(jdb.InvalidClaim):
        jdb.start_attempt(conn, jid, claim_token="not-the-token")
    assert len(jdb.get_attempts(conn, jid)) == 1


def test_attempt_finish_wrong_token_fails_closed(conn):
    jdb.create_job(conn, name="A", goal="g")
    claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
    jid = claim.job.id
    aid = jdb.start_attempt(conn, jid, claim_token=claim.claim_token)
    with pytest.raises(jdb.InvalidClaim):
        jdb.finish_attempt(conn, aid, status="succeeded", claim_token="wrong")
    # Unchanged: still running.
    assert jdb.get_attempt(conn, aid)["status"] == "running"


def test_attempt_start_on_claimed_job_requires_token(conn):
    # A claimed Job is under exclusive custody: starting an attempt WITHOUT the
    # token must fail closed. Otherwise a tokenless caller could seize the single
    # running-attempt slot and lock the real claimholder out of its own Job.
    jdb.create_job(conn, name="A", goal="g")
    claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
    jid = claim.job.id
    with pytest.raises(jdb.InvalidClaim):
        jdb.start_attempt(conn, jid)  # no token on a claimed Job
    assert jdb.get_attempts(conn, jid) == []  # nothing written
    # The real claimholder still succeeds with its token.
    aid = jdb.start_attempt(conn, jid, claim_token=claim.claim_token)
    assert jdb.get_attempt(conn, aid)["status"] == "running"


def test_attempt_finish_on_claimed_job_requires_token(conn):
    jdb.create_job(conn, name="A", goal="g")
    claim = jdb.claim_job(conn, worker="w1", lease_seconds=60)
    jid = claim.job.id
    aid = jdb.start_attempt(conn, jid, claim_token=claim.claim_token)
    with pytest.raises(jdb.InvalidClaim):
        jdb.finish_attempt(conn, aid, status="succeeded")  # no token on a claimed Job
    assert jdb.get_attempt(conn, aid)["status"] == "running"  # unchanged


def test_attempt_start_requires_a_claim_even_on_an_unclaimed_job(conn):
    # V2 supersedes V1's tokenless attempt calling: running work on a Job means
    # holding custody of it, so an unclaimed Job cannot start an attempt at all.
    jid = jdb.create_job(conn, name="A", goal="g")
    with pytest.raises(jdb.InvalidClaim):
        jdb.start_attempt(conn, jid)
    assert jdb.get_attempts(conn, jid) == []


def test_a_lapsed_worker_cannot_finish_after_recovery_reclaimed_its_attempt(conn):
    """The classic stale writer: recovery already ruled, the worker comes back."""
    jid, token = _claimed(conn, name="A", lease_seconds=60, now=1000)
    aid = jdb.start_attempt(conn, jid, claim_token=token, now=1010)
    jdb.recover_expired_claims(conn, now=2000)
    with pytest.raises((jdb.InvalidClaim, jdb.InvalidTransition)):
        jdb.finish_attempt(conn, aid, status="succeeded", claim_token=token, now=2100)
    att = jdb.get_attempt(conn, aid)
    assert (att["status"], att["failure_class"]) == ("interrupted", "infrastructure")


def test_expired_claim_token_cannot_start_an_attempt(conn):
    jid, token = _claimed(conn, name="A", lease_seconds=60, now=1000)  # expires 1060
    with pytest.raises(jdb.InvalidClaim):
        jdb.start_attempt(conn, jid, claim_token=token, now=2000)
    assert jdb.get_attempts(conn, jid) == []  # nothing written


def test_expired_claim_token_cannot_finish_an_attempt(conn):
    jid, token = _claimed(conn, name="A", lease_seconds=60, now=1000)  # expires 1060
    aid = jdb.start_attempt(conn, jid, claim_token=token, now=1010)
    with pytest.raises(jdb.InvalidClaim):
        jdb.finish_attempt(conn, aid, status="succeeded", claim_token=token, now=2000)
    assert jdb.get_attempt(conn, aid)["status"] == "running"  # unchanged


def test_expired_claim_token_cannot_release(conn):
    # Recovery — not the lapsed worker — owns an expired claim.
    jid, token = _claimed(conn, name="A", lease_seconds=60, now=1000)
    with pytest.raises(jdb.InvalidClaim):
        jdb.release_claim(conn, jid, claim_token=token, now=2000)
    assert jdb.get_job(conn, jid).claimed_by == "w1"  # unchanged


def test_finish_rejects_non_terminal_status(conn):
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    with pytest.raises(ValueError):
        jdb.finish_attempt(conn, aid, status="running", claim_token=token)
    with pytest.raises(ValueError):
        jdb.finish_attempt(conn, aid, status="bogus", claim_token=token)


def test_finish_rejects_unknown_failure_class(conn):
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    with pytest.raises(ValueError):
        jdb.finish_attempt(
            conn, aid, status="failed", failure_class="cosmic_rays", claim_token=token
        )


@pytest.mark.parametrize("st", list(jdb.TERMINAL_ATTEMPT_STATUSES))
def test_all_terminal_statuses_accepted(conn, st):
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    fc = None if st in ("succeeded", "cancelled") else "implementation"
    jdb.finish_attempt(conn, aid, status=st, failure_class=fc, claim_token=token)
    assert jdb.get_attempt(conn, aid)["status"] == st


# ===========================================================================
# Jobs Execution V2 — Task 4: lease heartbeat and release
# ===========================================================================


def _claim(conn, **kw):
    jdb.create_job(conn, name=kw.pop("name", "A"), goal=kw.pop("goal", "g"),
                   specialist=kw.pop("specialist", None))
    return jdb.claim_job(conn, worker=kw.pop("worker", "w1"),
                         lease_seconds=kw.pop("lease_seconds", 60), now=kw.pop("now", 1000))


def test_claim_heartbeat_extends_lease_and_freshness(conn):
    claim = _claim(conn, now=1000, lease_seconds=60)
    jid = claim.job.id
    assert claim.job.lease_expires_at == 1060
    # Beat within the window (1050 <= 1060) extends deterministically.
    updated = jdb.claim_heartbeat(conn, jid, claim_token=claim.claim_token,
                                  lease_seconds=120, now=1050)
    assert updated.lease_expires_at == 1170
    assert updated.last_heartbeat_at == 1050
    # Public freshness reflects the beat.
    assert jdb.projection(conn, jid, now=1060, stale_threshold=60)["stale"] is False


def test_claim_heartbeat_wrong_token_fails_closed(conn):
    claim = _claim(conn, now=1000, lease_seconds=60)
    jid = claim.job.id
    before = jdb.get_job(conn, jid)
    with pytest.raises(jdb.InvalidClaim):
        jdb.claim_heartbeat(conn, jid, claim_token="wrong", lease_seconds=60, now=1010)
    after = jdb.get_job(conn, jid)
    # No mutation on a bad token.
    assert (after.lease_expires_at, after.revision) == (before.lease_expires_at, before.revision)


def test_claim_heartbeat_expired_lease_fails_closed(conn):
    claim = _claim(conn, now=1000, lease_seconds=60)  # expires at 1060
    jid = claim.job.id
    with pytest.raises(jdb.InvalidClaim):
        jdb.claim_heartbeat(conn, jid, claim_token=claim.claim_token,
                            lease_seconds=60, now=2000)  # 2000 > 1060


def test_claim_heartbeat_rejects_bad_lease(conn):
    claim = _claim(conn, now=1000, lease_seconds=60)
    jid = claim.job.id
    for bad in (0, -1, jdb.MAX_LEASE_SECONDS + 1):
        with pytest.raises(ValueError):
            jdb.claim_heartbeat(conn, jid, claim_token=claim.claim_token,
                                lease_seconds=bad, now=1010)


def test_release_returns_job_to_working_and_clears_custody(conn):
    claim = _claim(conn, now=1000, lease_seconds=60)
    jid = claim.job.id
    jdb.set_step(conn, jid, "building")
    released = jdb.release_claim(conn, jid, claim_token=claim.claim_token, now=1010)
    assert released.status == "working"
    assert released.claimed_by is None
    assert released.lease_expires_at is None
    assert released.current_attempt_id is None
    # A released Job is immediately claimable again.
    again = jdb.claim_job(conn, worker="w2", lease_seconds=60, now=1020)
    assert again is not None and again.job.id == jid


def test_release_wrong_token_fails_closed(conn):
    claim = _claim(conn, now=1000, lease_seconds=60)
    jid = claim.job.id
    with pytest.raises(jdb.InvalidClaim):
        jdb.release_claim(conn, jid, claim_token="nope", now=1010)
    assert jdb.get_job(conn, jid).claimed_by == "w1"


def test_release_closes_a_running_attempt_as_cancelled(conn):
    """Release must never strand a running attempt no one can ever finish."""
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.release_claim(conn, jid, claim_token=token)
    att = jdb.get_attempt(conn, aid)
    assert att["status"] == "cancelled"
    assert att["finished_at"] is not None
    # The next worker can actually run the Job instead of hitting the
    # one-running index forever.
    again = jdb.claim_job(conn, worker="w2", job=jid, lease_seconds=600)
    assert again is not None
    a2 = jdb.start_attempt(conn, jid, claim_token=again.claim_token)
    assert jdb.get_attempt(conn, a2)["status"] == "running"


@pytest.mark.parametrize(
    "status,step",
    [("needs_you", "waiting_for_decision"), ("finished", "complete")],
)
def test_transition_refuses_to_orphan_a_running_attempt(conn, status, step):
    """A general transition may not silently clear custody out from under a run."""
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    before = jdb.get_job(conn, jid)
    with pytest.raises(jdb.InvalidTransition):
        jdb.transition(conn, jid, status=status, step=step)
    after = jdb.get_job(conn, jid)
    assert (after.status, after.step, after.revision) == (
        before.status, before.step, before.revision
    )  # no mutation
    assert jdb.get_attempt(conn, aid)["status"] == "running"
    assert after.claimed_by == "w1"


def test_finish_is_terminal_and_conflicting_replay_fails_closed(conn):
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(
        conn, aid, status="failed", failure_class="implementation", claim_token=token
    )
    before_events = jdb.get_events(conn, jid)
    with pytest.raises(jdb.InvalidTransition):
        jdb.finish_attempt(conn, aid, status="succeeded", claim_token=token)
    att = jdb.get_attempt(conn, aid)
    assert (att["status"], att["failure_class"]) == ("failed", "implementation")
    assert jdb.get_events(conn, jid) == before_events  # no mutation at all


def test_identical_finish_replay_returns_existing_without_duplicate_events(conn):
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    first = jdb.finish_attempt(
        conn, aid, status="succeeded", claim_token=token, commit="abc123"
    )
    events = jdb.get_events(conn, jid)
    job = jdb.get_job(conn, jid)
    # A replay is a read: it returns the settled result even though finishing
    # already cleared the custody the replay's token refers to.
    again = jdb.finish_attempt(conn, aid, status="succeeded", claim_token=token)
    assert again == first
    assert jdb.get_events(conn, jid) == events
    assert [e["kind"] for e in events].count("attempt_finished") == 1
    assert jdb.get_job(conn, jid).to_dict() == job.to_dict()


def test_replay_with_conflicting_terminal_failure_fails_closed(conn):
    """``terminal_failure`` picks the Job outcome, so it must gate the replay.

    Same attempt status and failure class, but the replay asks for the
    give-up outcome (``finished``/``failed``) the settled call never applied.
    Returning "success" there would tell the caller its request landed.
    """
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(
        conn, aid, status="failed", failure_class="implementation",
        claim_token=token, terminal_failure=False,
    )
    settled = jdb.get_job(conn, jid)
    assert (settled.status, settled.step) == ("working", "correcting")
    with pytest.raises(jdb.InvalidTransition):
        jdb.finish_attempt(
            conn, aid, status="failed", failure_class="implementation",
            claim_token=token, terminal_failure=True,
        )
    assert jdb.get_job(conn, jid).to_dict() == settled.to_dict()


@pytest.mark.parametrize(
    "evidence,first,conflicting",
    [
        ("commit", "c0ffee", "deadbee"),
        ("branch", "feat/a", "feat/b"),
        ("repository", "repo-a", "repo-b"),
        ("worktree", "/tmp/wt-a", "/tmp/wt-b"),
    ],
)
def test_replay_with_conflicting_evidence_fails_closed(
    conn, evidence, first, conflicting
):
    """Execution evidence is part of the terminal result, not decoration."""
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(
        conn, aid, status="succeeded", claim_token=token, **{evidence: first}
    )
    with pytest.raises(jdb.InvalidTransition):
        jdb.finish_attempt(
            conn, aid, status="succeeded", claim_token=token,
            **{evidence: conflicting},
        )
    assert jdb.get_attempt(conn, aid)[evidence] == first


def test_identical_replay_with_full_evidence_stays_idempotent(conn):
    """The exact same terminal request is still a read of the settled result."""
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    outcome = dict(
        status="failed", failure_class="implementation", terminal_failure=True,
        commit="c0ffee", branch="feat/a", worktree="/tmp/wt", repository="repo",
    )
    first = jdb.finish_attempt(conn, aid, claim_token=token, **outcome)
    events = jdb.get_events(conn, jid)
    job = jdb.get_job(conn, jid)
    assert (job.status, job.step) == ("finished", "failed")

    again = jdb.finish_attempt(conn, aid, claim_token=token, **outcome)
    assert again == first
    assert jdb.get_events(conn, jid) == events
    assert [e["kind"] for e in events].count("attempt_finished") == 1
    assert jdb.get_job(conn, jid).to_dict() == job.to_dict()


def test_conflicting_replay_appends_no_event_and_leaves_state_unchanged(conn):
    """A rejected replay must not move the attempt, Job, events, or revision."""
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(
        conn, aid, status="failed", failure_class="implementation",
        claim_token=token, commit="first", terminal_failure=False,
    )
    attempt = jdb.get_attempt(conn, aid)
    events = jdb.get_events(conn, jid)
    job = jdb.get_job(conn, jid)

    with pytest.raises(jdb.InvalidTransition):
        jdb.finish_attempt(
            conn, aid, status="failed", failure_class="implementation",
            claim_token=token, commit="conflicting", terminal_failure=True,
        )
    assert jdb.get_attempt(conn, aid) == attempt
    assert jdb.get_events(conn, jid) == events
    assert jdb.get_job(conn, jid).to_dict() == job.to_dict()  # revision included


# Every terminal outcome moves the same Job atomically and clears custody.
# (attempt status, failure class) -> (public status, step)
_OUTCOMES = [
    ("succeeded", None, "finished", "complete"),
    ("review_rejected", "reviewer_rejection", "working", "correcting"),
    ("failed", "reviewer_rejection", "working", "correcting"),
    ("failed", "authentication", "needs_you", "waiting_for_login"),
    ("failed", "irreversible_action", "needs_you", "waiting_for_decision"),
    ("failed", "provider_billing", "needs_you", "waiting_for_decision"),
    ("failed", "infrastructure", "working", "routing"),
    ("interrupted", "infrastructure", "working", "routing"),
    ("cancelled", None, "working", "routing"),
    ("failed", "max_turns", "working", "correcting"),
    ("failed", "implementation", "working", "correcting"),
    ("failed", None, "working", "correcting"),
]


@pytest.mark.parametrize("att_status,failure_class,job_status,job_step", _OUTCOMES)
def test_terminal_outcome_moves_job_and_clears_custody_atomically(
    conn, att_status, failure_class, job_status, job_step
):
    goal = "Original goal stays put \U0001f680"
    jid, token = _claimed(conn, name="Outcome", goal=goal)
    number = jdb.get_job(conn, jid).number
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(
        conn, aid, status=att_status, failure_class=failure_class, claim_token=token
    )
    job = jdb.get_job(conn, jid)
    assert (job.status, job.step) == (job_status, job_step)
    # Custody is gone in the same write — no active claim survives an outcome.
    assert job.claimed_by is None
    assert job.lease_expires_at is None and job.claim_acquired_at is None
    assert job.current_attempt_id is None
    # One Job per goal: identity and goal are never rewritten by an outcome.
    assert job.goal == goal and job.number == number
    kinds = [e["kind"] for e in jdb.get_events(conn, jid)]
    assert kinds[-2:] == ["attempt_finished", "job_transition"]


def test_explicit_terminal_implementation_failure_finishes_the_job(conn):
    """finished/failed is an explicit option, never inferred from the topic."""
    jid, token = _claimed(conn, name="Dead end")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(
        conn, aid, status="failed", failure_class="implementation",
        claim_token=token, terminal_failure=True,
    )
    job = jdb.get_job(conn, jid)
    assert (job.status, job.step) == ("finished", "failed")
    assert job.claimed_by is None and job.current_attempt_id is None


def test_terminal_failure_flag_requires_a_failed_attempt(conn):
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    with pytest.raises(ValueError):
        jdb.finish_attempt(
            conn, aid, status="succeeded", claim_token=token, terminal_failure=True
        )
    assert jdb.get_attempt(conn, aid)["status"] == "running"


def test_succeeded_attempt_rejects_a_failure_class(conn):
    jid, token = _claimed(conn, name="A")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    with pytest.raises(ValueError):
        jdb.finish_attempt(
            conn, aid, status="succeeded", failure_class="implementation",
            claim_token=token,
        )
    assert jdb.get_attempt(conn, aid)["status"] == "running"


def test_success_outcome_finishes_and_clears_custody(conn):
    goal = "Do the exact thing — verbatim."
    jid, token = _claimed(conn, name="Win", goal=goal)
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(conn, aid, status="succeeded", claim_token=token)
    job = jdb.get_job(conn, jid)
    assert job.status == "finished" and job.step == "complete"
    assert job.claimed_by is None and job.current_attempt_id is None
    assert job.goal == goal  # verified success never rewrites the goal


def test_auth_failure_pauses_same_job_without_rewriting_goal(conn):
    goal = "Original goal stays put."
    jid, token = _claimed(conn, name="Auth", goal=goal)
    number = jdb.get_job(conn, jid).number
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(
        conn, aid, status="failed", failure_class="authentication", claim_token=token
    )
    job = jdb.get_job(conn, jid)
    assert job.status == "needs_you" and job.step == "waiting_for_login"
    assert job.claimed_by is None  # custody cleared on pause
    assert job.goal == goal and job.number == number  # same Job, same goal


def test_reviewer_rejection_keeps_job_working_correcting_with_findings(conn):
    jid, token = _claimed(conn, name="Rev")
    aid = jdb.start_attempt(conn, jid, claim_token=token)
    jdb.finish_attempt(conn, aid, status="review_rejected",
                       failure_class="reviewer_rejection", claim_token=token)
    jdb.add_receipt(conn, jid, data={"reviewer": "themis", "findings": ["fix x"]})
    job = jdb.get_job(conn, jid)
    assert job.status == "working" and job.step == "correcting"
    assert any(r["data"].get("findings") for r in jdb.get_receipts(conn, jid))


# ===========================================================================
# Jobs Execution V2 — Task 5: interrupted recovery
# ===========================================================================


def test_recover_only_recovers_expired_claims(conn):
    jdb.create_job(conn, name="A", goal="g")
    jdb.create_job(conn, name="B", goal="g")
    jdb.claim_job(conn, worker="w1", job=1, lease_seconds=60, now=1000)  # expires 1060
    c2 = jdb.claim_job(conn, worker="w2", job=2, lease_seconds=60, now=1000)
    # Keep job 2's lease alive past recovery time (1050 + 3600 = 4650 > 2000).
    jdb.claim_heartbeat(conn, 2, claim_token=c2.claim_token, lease_seconds=3600, now=1050)

    recovered = jdb.recover_expired_claims(conn, now=2000)
    assert recovered == [jdb.get_job(conn, 1).id]
    assert jdb.get_job(conn, 1).claimed_by is None
    assert jdb.get_job(conn, 2).claimed_by == "w2"  # unexpired claim untouched


def test_recover_interrupts_running_attempt_as_infrastructure(conn):
    jdb.create_job(conn, name="A", goal="g")
    c = jdb.claim_job(conn, worker="w1", lease_seconds=60, now=1000)
    jid = c.job.id
    aid = jdb.start_attempt(conn, jid, claim_token=c.claim_token, now=1010)
    jdb.recover_expired_claims(conn, now=2000)
    att = jdb.get_attempt(conn, aid)
    assert att["status"] == "interrupted"
    assert att["failure_class"] == "infrastructure"
    job = jdb.get_job(conn, jid)
    assert job.status == "working"  # recovery never produces needs_you
    assert job.current_attempt_id is None
    assert job.claimed_by is None


def test_recover_returns_to_routing_and_preserves_goal(conn):
    jdb.create_job(conn, name="A", goal="verbatim goal \U0001f680")
    c = jdb.claim_job(conn, worker="w1", lease_seconds=60, now=1000)
    jid = c.job.id
    number = c.job.number
    jdb.set_step(conn, jid, "building")
    jdb.recover_expired_claims(conn, now=2000)
    job = jdb.get_job(conn, jid)
    assert job.step == "routing"
    assert job.goal == "verbatim goal \U0001f680"  # identity preserved
    assert job.number == number


@pytest.mark.parametrize("step", ["correcting", "reviewing"])
def test_recover_preserves_correction_review_step(conn, step):
    jdb.create_job(conn, name="A", goal="g")
    c = jdb.claim_job(conn, worker="w1", lease_seconds=60, now=1000)
    jid = c.job.id
    jdb.set_step(conn, jid, step)
    jdb.recover_expired_claims(conn, now=2000)
    assert jdb.get_job(conn, jid).step == step


def test_recover_is_idempotent(conn):
    jdb.create_job(conn, name="A", goal="g")
    c = jdb.claim_job(conn, worker="w1", lease_seconds=60, now=1000)
    jid = c.job.id
    jdb.start_attempt(conn, jid, claim_token=c.claim_token, now=1010)
    first = jdb.recover_expired_claims(conn, now=2000)
    assert first == [jid]
    events_after = jdb.get_events(conn, jid)
    attempts_after = jdb.get_attempts(conn, jid)
    # A second run recovers nothing and duplicates nothing.
    second = jdb.recover_expired_claims(conn, now=2000)
    assert second == []
    assert jdb.get_events(conn, jid) == events_after
    assert jdb.get_attempts(conn, jid) == attempts_after
    assert [e["kind"] for e in events_after].count("claim_recovered") == 1


def test_recover_does_not_touch_unclaimed_jobs(conn):
    jdb.create_job(conn, name="A", goal="g")  # never claimed
    before = jdb.get_events(conn, 1)
    assert jdb.recover_expired_claims(conn, now=10_000_000) == []
    assert jdb.get_events(conn, 1) == before


# ===========================================================================
# Jobs Execution V2 — Task 6: idempotent source intake
# ===========================================================================


def test_intake_creates_then_returns_same_job(conn):
    r1 = jdb.create_or_get_job(
        conn, source_type="cron", source_key="daily-report",
        name="Daily report", goal="compile the report",
    )
    assert r1.created is True and r1.conflict is False
    r2 = jdb.create_or_get_job(
        conn, source_type="cron", source_key="daily-report",
        name="Daily report", goal="compile the report",
    )
    assert r2.created is False and r2.conflict is False
    assert r2.job_id == r1.job_id
    # Exactly one Job exists for the source key.
    assert len(jdb.list_jobs(conn)) == 1


def test_intake_changed_payload_is_a_conflict_and_preserves_original(conn):
    r1 = jdb.create_or_get_job(
        conn, source_type="kanban", source_key="card-9",
        name="Original name", goal="original goal",
    )
    r2 = jdb.create_or_get_job(
        conn, source_type="kanban", source_key="card-9",
        name="Rewritten name", goal="rewritten goal",
    )
    assert r2.job_id == r1.job_id
    assert r2.created is False
    assert r2.conflict is True
    # The original Job is never rewritten.
    job = jdb.get_job(conn, r1.job_id)
    assert job.name == "Original name"
    assert job.goal == "original goal"


def test_intake_different_source_types_reuse_the_same_key(conn):
    r1 = jdb.create_or_get_job(
        conn, source_type="cron", source_key="shared-1", name="A", goal="a",
    )
    r2 = jdb.create_or_get_job(
        conn, source_type="kanban", source_key="shared-1", name="B", goal="b",
    )
    assert r1.created and r2.created
    assert r1.job_id != r2.job_id


def test_intake_requires_source_fields(conn):
    with pytest.raises(ValueError):
        jdb.create_or_get_job(conn, source_type="", source_key="k", name="N", goal="g")
    with pytest.raises(ValueError):
        jdb.create_or_get_job(conn, source_type="cron", source_key="  ", name="N", goal="g")


def test_intake_on_migrated_early_v1_db(tmp_path):
    """Intake works after an early V1 DB gains the source registry on open."""
    import sqlite3

    path = tmp_path / "jobs.db"
    raw = sqlite3.connect(str(path))
    try:
        raw.executescript(
            "CREATE TABLE jobs ("
            " id TEXT PRIMARY KEY, number INTEGER NOT NULL UNIQUE, name TEXT NOT NULL,"
            " goal TEXT NOT NULL, status TEXT NOT NULL, step TEXT NOT NULL,"
            " specialist TEXT, routing_reason TEXT, created_at INTEGER NOT NULL,"
            " updated_at INTEGER NOT NULL, last_heartbeat_at INTEGER);"
        )
        raw.commit()
    finally:
        raw.close()

    conn = jdb.connect(db_path=path)
    try:
        r1 = jdb.create_or_get_job(
            conn, source_type="cron", source_key="daily-1", name="Daily", goal="run",
        )
        assert r1.created is True
        r2 = jdb.create_or_get_job(
            conn, source_type="cron", source_key="daily-1", name="Daily", goal="run",
        )
        assert r2.created is False and r2.job_id == r1.job_id
    finally:
        conn.close()


def test_concurrent_same_source_creates_exactly_one_job(tmp_path):
    """Two intakes of the same (type, key) create exactly one Job."""
    path = tmp_path / "jobs.db"
    jdb.connect(db_path=path).close()  # pre-create schema

    barrier = threading.Barrier(2)
    results: dict[str, object] = {}
    lock = threading.Lock()

    def worker(name: str) -> None:
        c = jdb.connect(db_path=path)
        try:
            barrier.wait()
            res = jdb.create_or_get_job(
                c, source_type="cron", source_key="same", name="X", goal="g",
            )
            with lock:
                results[name] = res
        except Exception as exc:  # pragma: no cover - only on real defect
            with lock:
                results[name] = exc
        finally:
            c.close()

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert all(not isinstance(r, Exception) for r in results.values()), results
    job_ids = {r.job_id for r in results.values()}
    assert len(job_ids) == 1  # both saw the same Job
    assert sum(1 for r in results.values() if r.created) == 1  # created exactly once

    verify = jdb.connect(db_path=path)
    try:
        assert verify.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 1
    finally:
        verify.close()


# ---------------------------------------------------------------------------
# Atomic finalization — the attempt and its final receipt are one outcome
# ---------------------------------------------------------------------------


def _final(attempt_id, status="succeeded", failure_class=None, **extra):
    return {
        "attempt_id": attempt_id,
        "status": status,
        "failure_class": failure_class,
        **extra,
    }


def test_settle_attempt_writes_the_attempt_and_its_final_receipt_together(conn):
    jid, token = _claimed(conn, specialist="claude-builder")
    aid = jdb.start_attempt(conn, jid, claim_token=token, specialist="claude-builder")

    settled = jdb.settle_attempt(
        conn, aid, status="succeeded", claim_token=token,
        commit="c" * 40, branch="jobs/1-x", worktree="/tmp/wt",
        receipt=_final(aid),
    )

    assert settled["attempt"]["status"] == "succeeded"
    receipts = jdb.get_receipts(conn, jid)
    assert len(receipts) == 1
    assert receipts[0]["id"] == settled["receipt_id"]
    assert receipts[0]["idempotency_key"] == f"{aid}:final"
    assert receipts[0]["attempt_id"] == aid
    assert jdb.get_job(conn, jid).status == "finished"


def test_settle_attempt_refuses_a_receipt_that_contradicts_the_attempt(conn):
    jid, token = _claimed(conn, specialist="claude-builder")
    aid = jdb.start_attempt(conn, jid, claim_token=token, specialist="claude-builder")

    with pytest.raises(ValueError, match="disagrees with attempt"):
        jdb.settle_attempt(
            conn, aid, status="failed", failure_class="implementation",
            claim_token=token, receipt=_final(aid, status="succeeded"),
        )

    # Nothing moved: the attempt is still running and still claimable custody.
    assert jdb.get_attempt(conn, aid)["status"] == "running"
    assert jdb.get_receipts(conn, jid) == []
    assert jdb.get_job(conn, jid).status == "working"


def test_a_failed_receipt_insert_rolls_back_the_whole_settlement(conn, monkeypatch):
    jid, token = _claimed(conn, specialist="claude-builder")
    aid = jdb.start_attempt(conn, jid, claim_token=token, specialist="claude-builder")
    before = jdb.get_job(conn, jid).revision

    def refuse():
        raise sqlite3.IntegrityError("receipt storage refused the write")

    monkeypatch.setattr(jdb, "_new_receipt_id", refuse)
    with pytest.raises(sqlite3.IntegrityError):
        jdb.settle_attempt(
            conn, aid, status="succeeded", claim_token=token, receipt=_final(aid)
        )

    assert jdb.get_attempt(conn, aid)["status"] == "running"
    assert jdb.get_receipts(conn, jid) == []
    job = jdb.get_job(conn, jid)
    assert (job.status, job.revision) == ("working", before)
    assert job.claimed_by == "w1"
    kinds = [e["kind"] for e in jdb.get_events(conn, jid)]
    assert "attempt_finished" not in kinds
    assert "receipt_added" not in kinds


def test_settling_the_same_outcome_twice_is_an_exact_replay(conn):
    jid, token = _claimed(conn, specialist="claude-builder")
    aid = jdb.start_attempt(conn, jid, claim_token=token, specialist="claude-builder")
    first = jdb.settle_attempt(
        conn, aid, status="succeeded", claim_token=token, receipt=_final(aid)
    )
    revision = jdb.get_job(conn, jid).revision

    second = jdb.settle_attempt(
        conn, aid, status="succeeded", claim_token=token, receipt=_final(aid)
    )

    assert second == first
    assert len(jdb.get_receipts(conn, jid)) == 1
    assert jdb.get_job(conn, jid).revision == revision


def test_a_replay_whose_receipt_content_differs_fails_closed(conn):
    jid, token = _claimed(conn, specialist="claude-builder")
    aid = jdb.start_attempt(conn, jid, claim_token=token, specialist="claude-builder")
    jdb.settle_attempt(
        conn, aid, status="succeeded", claim_token=token, receipt=_final(aid)
    )
    revision = jdb.get_job(conn, jid).revision

    with pytest.raises(ValueError, match="final receipt"):
        jdb.settle_attempt(
            conn, aid, status="succeeded", claim_token=token,
            receipt=_final(aid, extra_claim="hostile"),
        )

    assert len(jdb.get_receipts(conn, jid)) == 1
    assert jdb.get_job(conn, jid).revision == revision


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_add_receipt_refuses_a_number_json_cannot_spell(conn, bad):
    """The store refuses too — at any depth, and without leaving a write behind.

    A caller reaching this function directly has skipped the CLI's strict parse,
    and Python's encoder would otherwise write a bare ``NaN`` that no other
    language's JSON reader can load back.
    """
    jid, _ = _claimed(conn, specialist="claude-builder")
    revision = jdb.get_job(conn, jid).revision

    with pytest.raises(ValueError):
        jdb.add_receipt(conn, jid, data={"cost": {"totals": [bad]}})

    assert jdb.get_receipts(conn, jid) == []
    assert jdb.get_job(conn, jid).revision == revision
    assert "receipt_added" not in [e["kind"] for e in jdb.get_events(conn, jid)]


def test_add_receipt_refuses_the_final_receipt_namespace(conn):
    jid, token = _claimed(conn, specialist="claude-builder")
    aid = jdb.start_attempt(conn, jid, claim_token=token, specialist="claude-builder")
    with pytest.raises(ValueError, match="final"):
        jdb.add_receipt(
            conn, jid, data={"status": "succeeded"}, attempt_id=aid,
            idempotency_key=f"{aid}:final",
        )
    assert jdb.get_receipts(conn, jid) == []


def test_recovery_gives_the_interrupted_attempt_its_own_final_receipt(conn):
    jid, token = _claimed(conn, specialist="claude-builder", lease_seconds=60, now=1000)
    aid = jdb.start_attempt(
        conn, jid, claim_token=token, specialist="claude-builder", now=1000
    )

    assert jdb.recover_expired_claims(conn, now=5000) == [jid]

    receipts = jdb.get_receipts(conn, jid)
    assert len(receipts) == 1
    assert receipts[0]["idempotency_key"] == f"{aid}:final"
    assert receipts[0]["attempt_id"] == aid
    assert receipts[0]["data"]["status"] == "interrupted"
    assert receipts[0]["data"]["failure_class"] == "infrastructure"


def test_jobs_store_initializes_outside_kanban(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    conn = jdb.connect(tmp_path / "candidate-jobs.db")
    try:
        job_id = jdb.create_job(conn, name="candidate", goal="prove isolation")
        job = jdb.get_job(conn, job_id)
        assert job.id == job_id
        assert (job.status, job.step) == ("working", "routing")
        assert not (tmp_path / "profile" / "kanban.db").exists()
    finally:
        conn.close()
