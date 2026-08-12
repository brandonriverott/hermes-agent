from dataclasses import replace
import sqlite3

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_handoffs as handoffs
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


def _signed(
    signing_key,
    *,
    receipt_id,
    job_id,
    attempt_id,
    state,
    evidence=None,
    handoff=None,
):
    return receipts.sign_receipt(
        {
            "job_id": job_id,
            "attempt_id": attempt_id,
            "commit": "c" * 40,
            "state": state,
            "evidence": {} if evidence is None else dict(evidence),
            "handoff": handoff,
            "timestamp": "2026-08-09T00:00:00Z",
            "transitioned_by": "jobs_graph/1",
        },
        receipt_id=receipt_id,
        key_id="lane:test:v1",
        private_key=signing_key,
    )


def _attempt(conn):
    job_id = jdb.create_job(conn, requested_lane="claude", name="job", goal="goal")
    claim = jdb.claim_job(
        conn,
        specialist="claude-builder",
        worker="worker",
        lease_seconds=60,
        job=job_id,
        now=1,
    )
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
    transition_handoff = handoffs.normalize_handoff(
        {
            "summary": "Attempt entered the Job graph.",
            "evidence_summary": [
                {
                    "label": "Attempt created",
                    "result": "Recorded.",
                    "digest": write.evidence["attempt_created"],
                }
            ],
            "next_action": "Route this bounded Job attempt.",
            "issues": [],
            "decision_request": None,
        },
        job_id=job_id,
        attempt_id=attempt_id,
        speaker_id="test",
        speaker_role="system",
        speaker_executor="hermes",
        from_phase="ATTEMPT_CREATED",
        to_phase="QUEUED",
        next_owner_role="router",
        outcome="started",
        artifact_identity=None,
        transition_evidence=dict(write.evidence),
        created_at=write.created_at,
    )
    write = replace(write, handoff=transition_handoff)
    envelope = _signed(
        signing_key,
        receipt_id=write.receipt_id,
        job_id=job_id,
        attempt_id=attempt_id,
        state="QUEUED",
        evidence=write.evidence,
        handoff=write.handoff,
    )
    return write, envelope


def test_reliability_schema_is_additive_and_idempotent(tmp_path):
    path = tmp_path / "jobs.db"
    first = jdb.connect(path)
    job_id = jdb.create_job(first, requested_lane="claude", name="kept", goal="kept")
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


def test_transition_schema_adds_nullable_handoff_without_rewriting_history(conn):
    columns = {
        row["name"]: row
        for row in conn.execute("PRAGMA table_info(job_attempt_transitions)")
    }

    assert columns["handoff_json"]["notnull"] == 0


def test_transition_round_trip_carries_exact_validated_handoff(conn, signing_key):
    write, envelope = _transition(conn, signing_key)

    transition_id = jdb.record_transition(conn, write, envelope)

    row = jdb.latest_transition(conn, write.attempt_id)
    assert row["id"] == transition_id
    assert row["handoff"] == write.handoff
    assert jdb.list_transitions(conn, write.job_id)[0]["handoff"] == write.handoff
    assert jdb.latest_handoff(conn, write.attempt_id) == write.handoff
    stored = conn.execute(
        "SELECT handoff_json FROM job_attempt_transitions WHERE id = ?",
        (transition_id,),
    ).fetchone()["handoff_json"]
    assert stored.encode("utf-8") == receipts.canonical_json_bytes(write.handoff)


def test_record_transition_rejects_signed_handoff_mismatch_before_mutation(
    conn, signing_key
):
    write, _ = _transition(conn, signing_key)
    changed = dict(write.handoff)
    changed["summary"] = "Different signed facts."
    envelope = _signed(
        signing_key,
        receipt_id=write.receipt_id,
        job_id=write.job_id,
        attempt_id=write.attempt_id,
        state=write.target_state,
        evidence=write.evidence,
        handoff=changed,
    )
    revision = jdb.get_job(conn, write.job_id).revision

    with pytest.raises(
        receipts.ReceiptVerificationError,
        match="receipt handoff identity mismatch",
    ):
        jdb.record_transition(conn, write, envelope)

    assert jdb.get_job(conn, write.job_id).revision == revision
    assert jdb.list_transitions(conn, write.job_id) == []
    assert jdb.get_receipts(conn, write.job_id) == []


def test_historical_transition_migrates_to_null_handoff_without_inference(
    tmp_path, signing_key
):
    path = tmp_path / "historical.db"
    first = jdb.connect(path)
    write, envelope = _transition(first, signing_key)
    jdb.record_transition(first, write, envelope)
    first.execute("ALTER TABLE job_attempt_transitions DROP COLUMN handoff_json")
    first.commit()
    first.close()
    jdb._INITIALIZED_PATHS.discard(str(path.resolve()))

    reopened = jdb.connect(path)
    try:
        raw = reopened.execute(
            "SELECT handoff_json FROM job_attempt_transitions"
        ).fetchone()
        assert raw["handoff_json"] is None
        assert jdb.latest_handoff(reopened, write.attempt_id) is None
        assert jdb.latest_transition(reopened, write.attempt_id)["handoff"] is None
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("handoff_json", "{", "persisted handoff.*canonical JSON"),
        (
            "handoff_json",
            receipts.canonical_json_bytes({"unknown": "field"}).decode("utf-8"),
            "persisted handoff.*unknown",
        ),
        ("evidence_json", "{", "transition evidence.*canonical JSON"),
        ("evidence_json", "[]", "transition evidence.*exact JSON object"),
        (
            "evidence_json",
            receipts.canonical_json_bytes({
                "attempt_created": "sha256:" + "f" * 64
            }).decode("utf-8"),
            "evidence_summary.*not bound",
        ),
    ],
)
def test_latest_handoff_rejects_malformed_persisted_handoff_or_evidence(
    conn, signing_key, column, value, match
):
    write, envelope = _transition(conn, signing_key)
    jdb.record_transition(conn, write, envelope)
    conn.execute(
        f"UPDATE job_attempt_transitions SET {column} = ? WHERE attempt_id = ?",
        (value, write.attempt_id),
    )
    conn.commit()

    with pytest.raises(handoffs.HandoffValidationError, match=match):
        jdb.latest_handoff(conn, write.attempt_id)


def test_transition_and_receipt_rollback_together(conn, signing_key):
    transition_write, signed_envelope = _transition(conn, signing_key)
    conn.execute("DROP TABLE job_attempt_transitions")
    conn.commit()

    with pytest.raises(sqlite3.OperationalError):
        jdb.record_transition(conn, transition_write, signed_envelope)

    assert conn.execute("SELECT COUNT(*) FROM job_receipts").fetchone()[0] == 0


def test_preflight_revision_cas_and_idempotency_are_fail_closed(conn, signing_key):
    job_id = jdb.create_job(conn, requested_lane="claude", name="job", goal="goal")
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
        jdb.record_transition(
            conn, replace(write, initiator_type="different"), envelope
        )
    changed_handoff = dict(write.handoff)
    changed_handoff["summary"] = "Different canonical facts."
    changed_envelope = _signed(
        signing_key,
        receipt_id=write.receipt_id,
        job_id=write.job_id,
        attempt_id=write.attempt_id,
        state=write.target_state,
        evidence=write.evidence,
        handoff=changed_handoff,
    )
    with pytest.raises(jdb.GraphConflict):
        jdb.record_transition(
            conn,
            replace(write, handoff=changed_handoff),
            changed_envelope,
        )
    assert (
        conn.execute("SELECT COUNT(*) FROM job_attempt_transitions").fetchone()[0] == 1
    )
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
