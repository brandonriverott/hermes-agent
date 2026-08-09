from dataclasses import replace
import sqlite3

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_receipts as receipts


@pytest.fixture
def conn(tmp_path):
    connection = jdb.connect(tmp_path / "jobs.db")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def signing_key():
    return Ed25519PrivateKey.generate()


def _signed(signing_key, *, receipt_id, job_id, attempt_id, state):
    return receipts.sign_receipt(
        {
            "job_id": job_id,
            "attempt_id": attempt_id,
            "commit": "c" * 40,
            "state": state,
            "evidence": {},
            "timestamp": "2026-08-09T00:00:00Z",
            "transitioned_by": "jobs_graph/1",
        },
        receipt_id=receipt_id,
        key_id="lane:test:v1",
        private_key=signing_key,
    )


def _attempt(conn):
    job_id = jdb.create_job(conn, name="job", goal="goal")
    claim = jdb.claim_job(conn, worker="worker", lease_seconds=60, job=job_id, now=1)
    attempt_id = jdb.start_attempt(
        conn,
        job_id,
        claim_token=claim.claim_token,
        specialist="worker",
        base_commit="b" * 40,
        commit="c" * 40,
        now=2,
    )
    return job_id, attempt_id


def _transition(conn, signing_key, *, idempotency_key="transition:queued"):
    job_id, attempt_id = _attempt(conn)
    revision = jdb.get_job(conn, job_id).revision
    write = jdb.TransitionWrite(
        job_id=job_id,
        attempt_id=attempt_id,
        source_state=None,
        target_state="QUEUED",
        initiator_type="system",
        initiator_id="test",
        expected_job_revision=revision,
        evidence={"attempt_created": "sha256:" + "a" * 64},
        failure_class=None,
        blocker_code=None,
        receipt_id="r_transition",
        component="jobs_graph",
        component_version="1",
        idempotency_key=idempotency_key,
        created_at=3,
    )
    envelope = _signed(
        signing_key,
        receipt_id=write.receipt_id,
        job_id=job_id,
        attempt_id=attempt_id,
        state="QUEUED",
    )
    return write, envelope


def test_reliability_schema_is_additive_and_idempotent(tmp_path):
    path = tmp_path / "jobs.db"
    first = jdb.connect(path)
    job_id = jdb.create_job(first, name="kept", goal="kept")
    first.close()
    jdb._INITIALIZED_PATHS.discard(str(path.resolve()))

    second = jdb.connect(path)
    try:
        assert jdb.get_job(second, job_id).goal == "kept"
        tables = {
            row[0]
            for row in second.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {
            "job_preflights",
            "job_attempt_transitions",
            "job_retry_evidence",
        } <= tables
    finally:
        second.close()


def test_transition_and_receipt_rollback_together(conn, signing_key):
    transition_write, signed_envelope = _transition(conn, signing_key)
    conn.execute("DROP TABLE job_attempt_transitions")
    conn.commit()

    with pytest.raises(sqlite3.OperationalError):
        jdb.record_transition(conn, transition_write, signed_envelope)

    assert conn.execute("SELECT COUNT(*) FROM job_receipts").fetchone()[0] == 0


def test_preflight_revision_cas_and_idempotency_are_fail_closed(conn, signing_key):
    job_id = jdb.create_job(conn, name="job", goal="goal")
    revision = jdb.get_job(conn, job_id).revision
    record = jdb.PreflightRecord(
        id="p_1",
        job_id=job_id,
        attempt_id=None,
        execution_spec_digest="sha256:" + "e" * 64,
        expected_job_revision=revision,
        status="PASS",
        failure_class=None,
        checks=({"name": "file_paths", "status": "PASS"},),
        receipt_id="r_preflight",
        idempotency_key="preflight:one",
        created_at=1,
    )
    envelope = _signed(
        signing_key,
        receipt_id=record.receipt_id,
        job_id=job_id,
        attempt_id=None,
        state="PASS",
    )

    assert jdb.record_preflight(conn, record, envelope) == "p_1"
    assert jdb.record_preflight(conn, record, envelope) == "p_1"
    assert conn.execute("SELECT COUNT(*) FROM job_preflights").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM job_receipts").fetchone()[0] == 1

    with pytest.raises(jdb.GraphConflict):
        jdb.record_preflight(conn, replace(record, status="BLOCKED"), envelope)
    with pytest.raises(jdb.StaleJobRevision):
        jdb.record_preflight(
            conn,
            replace(
                record,
                id="p_2",
                idempotency_key="preflight:two",
                expected_job_revision=revision - 1,
                receipt_id="r_preflight_2",
            ),
            replace_envelope(envelope, receipt_id="r_preflight_2"),
        )
    assert conn.execute("SELECT COUNT(*) FROM job_preflights").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM job_receipts").fetchone()[0] == 1


def replace_envelope(envelope, *, receipt_id):
    updated = dict(envelope)
    updated["receipt_id"] = receipt_id
    return updated


def test_transition_idempotency_requires_an_exact_material_match(conn, signing_key):
    write, envelope = _transition(conn, signing_key)

    transition_id = jdb.record_transition(conn, write, envelope)

    assert jdb.record_transition(conn, write, envelope) == transition_id
    assert jdb.latest_transition(conn, write.attempt_id)["target_state"] == "QUEUED"
    assert [row["id"] for row in jdb.list_transitions(conn, write.job_id)] == [
        transition_id
    ]
    with pytest.raises(jdb.GraphConflict):
        jdb.record_transition(conn, replace(write, target_state="ASSIGNED"), envelope)
    assert conn.execute("SELECT COUNT(*) FROM job_attempt_transitions").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM job_receipts").fetchone()[0] == 1


def test_retry_evidence_is_persisted_before_duplicate_evidence_is_rejected(conn):
    job_id, attempt_id = _attempt(conn)
    write = jdb.RetryEvidenceWrite(
        job_id=job_id,
        chain_id=attempt_id,
        attempt_id=attempt_id,
        parent_attempt_id=None,
        ordinal=1,
        evidence_digest="sha256:" + "d" * 64,
        decision="RETRY",
        reason_code="NEW_EVIDENCE",
        created_at=4,
    )

    accepted = jdb.record_retry_decision(conn, write)
    duplicate = jdb.record_retry_decision(conn, write)

    assert accepted["decision"] == "RETRY"
    assert (duplicate["decision"], duplicate["reason_code"]) == (
        "BLOCKED",
        "RETRY_REJECTED_NO_NEW_EVIDENCE",
    )
    assert conn.execute("SELECT COUNT(*) FROM job_retry_evidence").fetchone()[0] == 1


def test_retry_ordinal_cannot_be_reused_with_different_evidence(conn):
    job_id, attempt_id = _attempt(conn)
    original = jdb.RetryEvidenceWrite(
        job_id=job_id,
        chain_id=attempt_id,
        attempt_id=attempt_id,
        parent_attempt_id=None,
        ordinal=1,
        evidence_digest="sha256:" + "d" * 64,
        decision="RETRY",
        reason_code="NEW_EVIDENCE",
        created_at=4,
    )
    jdb.record_retry_decision(conn, original)

    with pytest.raises(jdb.GraphConflict):
        jdb.record_retry_decision(
            conn,
            replace(original, evidence_digest="sha256:" + "e" * 64),
        )
