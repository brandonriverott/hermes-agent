"""End-to-end Jobs handoffs: ledger, outbox, and exact-origin SessionDB."""

from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.jobs_notifications import deliver_due_notification_once
from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_handoffs
from hermes_cli import jobs_notifications as jn
from hermes_cli import jobs_receipts
from hermes_state import SessionDB


DIGEST = "sha256:" + "a" * 64
COMMIT = "c" * 40
DELIVERY_NOW = 4_000_000_000


@pytest.fixture
def jobs_path(tmp_path):
    return tmp_path / "jobs.db"


@pytest.fixture
def session_db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


def _origin(session_id="origin", chat_id="chat-origin"):
    return {
        "platform": "telegram",
        "chat_id": chat_id,
        "session_id": session_id,
        "chat_type": "dm",
        "thread_id": "thread-origin",
        "user_id": "user-origin",
        "profile": "default",
    }


def _signed(key, *, job_id, attempt_id, state, handoff, receipt_id):
    return jobs_receipts.sign_receipt(
        {
            "job_id": job_id,
            "attempt_id": attempt_id,
            "commit": COMMIT,
            "state": state,
            "evidence": {"tests": DIGEST},
            "handoff": handoff,
            "timestamp": "2026-08-12T00:00:00Z",
            "transitioned_by": "jobs_graph/1",
        },
        receipt_id=receipt_id,
        key_id="lane:test:v1",
        private_key=key,
    )


def _write(
    conn,
    key,
    *,
    job_id,
    attempt_id,
    source,
    target,
    speaker_id,
    speaker_role,
    outcome,
    at,
    summary,
    next_action,
    issues=(),
    decision_request=None,
):
    revision = jdb.get_job(conn, job_id).revision
    raw = {
        "summary": summary,
        "evidence_summary": [
            {"label": "Trusted test evidence", "result": "42 passed", "digest": DIGEST}
        ],
        "next_action": next_action,
        "issues": [dict(issue) for issue in issues],
        "decision_request": decision_request,
    }
    handoff = jobs_handoffs.normalize_handoff(
        raw,
        job_id=job_id,
        attempt_id=attempt_id,
        speaker_id=speaker_id,
        speaker_role=speaker_role,
        speaker_executor="hermes",
        from_phase="ATTEMPT_CREATED" if source is None else source,
        to_phase=target,
        next_owner_role=None if outcome == "completed" else "next-owner",
        outcome=outcome,
        artifact_identity=(
            {"kind": "commit", "value": COMMIT} if outcome == "completed" else None
        ),
        transition_evidence={"tests": DIGEST},
        created_at=at,
    )
    write = jdb.TransitionWrite(
        job_id=job_id,
        attempt_id=attempt_id,
        source_state=source,
        target_state=target,
        initiator_type="system",
        initiator_id=speaker_id,
        expected_job_revision=revision,
        evidence={"tests": DIGEST},
        failure_class="reviewer_rejection" if outcome == "rejected" else None,
        blocker_code="AUTH_LOGIN_REQUIRED" if target == "BLOCKED" else None,
        receipt_id=f"receipt:{job_id}:{at}:{target}",
        component="jobs_graph",
        component_version="1",
        idempotency_key=f"transition:{job_id}:{at}:{target}",
        created_at=at,
        commit=COMMIT,
        handoff=handoff,
    )
    return write, _signed(
        key,
        job_id=job_id,
        attempt_id=attempt_id,
        state=target,
        handoff=handoff,
        receipt_id=write.receipt_id,
    )


def _record(conn, key, **kwargs):
    write, envelope = _write(conn, key, **kwargs)
    return jdb.record_transition(conn, write, envelope)


def _start(conn, job_id, *, worker, at):
    claim = jdb.claim_job(
        conn,
        worker=worker,
        specialist="codex-builder",
        lease_seconds=600,
        job=job_id,
        now=at,
    )
    assert claim is not None
    return jdb.start_attempt(
        conn,
        job_id,
        claim_token=claim.claim_token,
        specialist=worker,
        base_commit="b" * 40,
        commit=COMMIT,
        now=at + 1,
    )


def _drain(session_db, jobs_path, *, now=DELIVERY_NOW):
    while deliver_due_notification_once(session_db, jobs_path=jobs_path, now=now):
        pass


def test_full_handoff_conversation_is_substantive_and_exact_origin(
    jobs_path, session_db
):
    key = Ed25519PrivateKey.generate()
    conn = jdb.connect(jobs_path)
    try:
        job_id = jdb.create_job(
            conn,
            name="Responsive card",
            goal="Fix the card layout",
            requested_lane="codex",
            origin=_origin(),
        )
        attempt = _start(conn, job_id, worker="builder-1", at=101)
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="QUEUED",
            target="ASSIGNED",
            speaker_id="router-1",
            speaker_role="router",
            outcome="handed_off",
            at=110,
            summary="Router assigned the exact codex lane.",
            next_action="Builder starts the scoped card fix.",
        )
        # BUILDING is an internal signed edge and must not become a visible duplicate.
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="ASSIGNED",
            target="BUILDING",
            speaker_id="builder-1",
            speaker_role="builder",
            outcome="started",
            at=120,
            summary="Builder started the card fix.",
            next_action="Collect trusted test evidence.",
        )
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="BUILDING",
            target="EVIDENCE_COLLECTING",
            speaker_id="builder-1",
            speaker_role="builder",
            outcome="handed_off",
            at=130,
            summary="Builder finished the card fix and attached trusted evidence.",
            next_action="Reviewer checks the rendered width.",
        )
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="EVIDENCE_COLLECTING",
            target="REVIEWING",
            speaker_id="reviewer-1",
            speaker_role="reviewer",
            outcome="handed_off",
            at=140,
            summary="Reviewer began checking the responsive card.",
            next_action="Reviewer records the decision.",
        )
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="REVIEWING",
            target="FAILED",
            speaker_id="reviewer-1",
            speaker_role="reviewer",
            outcome="rejected",
            at=150,
            summary="Reviewer rejected the card implementation.",
            next_action="Builder changes the card width and retries review.",
            issues=(
                {
                    "requirement": "The card must fit the responsive layout.",
                    "finding": "The card is 320px wide on the narrow viewport.",
                    "required_fix": "Change the fixed width to the responsive card width.",
                },
            ),
        )
        retry_attempt = _start(conn, job_id, worker="builder-2", at=160)
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=retry_attempt,
            source=None,
            target="QUEUED",
            speaker_id="retry-router",
            speaker_role="router",
            outcome="started",
            at=170,
            summary="Retry queued for the same Job.",
            next_action="Builder applies the reviewer change.",
        )
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=retry_attempt,
            source="QUEUED",
            target="ASSIGNED",
            speaker_id="router-2",
            speaker_role="router",
            outcome="handed_off",
            at=180,
            summary="Retry assigned to the builder.",
            next_action="Builder starts the correction.",
        )
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=retry_attempt,
            source="ASSIGNED",
            target="BUILDING",
            speaker_id="builder-2",
            speaker_role="builder",
            outcome="started",
            at=190,
            summary="Builder changed the card width.",
            next_action="Collect the corrected evidence.",
        )
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=retry_attempt,
            source="BUILDING",
            target="EVIDENCE_COLLECTING",
            speaker_id="builder-2",
            speaker_role="builder",
            outcome="handed_off",
            at=200,
            summary="Builder attached corrected responsive evidence.",
            next_action="Reviewer re-checks the card.",
        )
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=retry_attempt,
            source="EVIDENCE_COLLECTING",
            target="REVIEWING",
            speaker_id="reviewer-1",
            speaker_role="reviewer",
            outcome="handed_off",
            at=210,
            summary="Reviewer re-opened the corrected card.",
            next_action="Reviewer approves the corrected width.",
        )
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=retry_attempt,
            source="REVIEWING",
            target="VERIFIED",
            speaker_id="reviewer-1",
            speaker_role="reviewer",
            outcome="passed",
            at=220,
            summary="Reviewer approved the corrected card width.",
            next_action="Complete the Job.",
        )
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=retry_attempt,
            source="VERIFIED",
            target="COMPLETED",
            speaker_id="builder-2",
            speaker_role="builder",
            outcome="completed",
            at=230,
            summary="The verified card fix is complete.",
            next_action="No further action is required.",
        )
    finally:
        conn.close()

    session_db.create_session("origin", source="telegram")
    session_db.create_session("decoy", source="telegram")
    _drain(session_db, jobs_path)
    messages = session_db.get_messages("origin")
    assert session_db.get_messages("decoy") == []
    conn = jdb.connect(jobs_path)
    try:
        rows = jn.list_notifications(conn, job_id=job_id)
        assert [row.milestone for row in rows] == [
            "queued",
            "assigned",
            "testing",
            "review",
            "correcting",
            "assigned",
            "testing",
            "review",
            "review-approved",
            "finished",
        ]
        assert all(row.delivered_at is not None for row in rows)
        assert all(row.transition_id is not None for row in rows[1:])
        transitions = conn.execute(
            "SELECT evidence_json FROM job_attempt_transitions WHERE job_id = ?",
            (job_id,),
        ).fetchall()
        assert transitions and all(
            DIGEST in row["evidence_json"] for row in transitions
        )
    finally:
        conn.close()
    contents = [message["content"] for message in messages]
    assert contents[0].startswith("Legacy Jobs update — Queued")
    substantive = contents[1:]
    assert all("Legacy Jobs update" not in content for content in substantive)
    assert len(substantive) == len(set(substantive))
    assert all(
        "Evidence: Trusted test evidence: 42 passed" in content
        for content in substantive
    )
    assert any(
        "router started" in content.lower() or "router" in content.lower()
        for content in substantive
    )
    assert any(
        "builder" in content.lower() and "evidence" in content.lower()
        for content in substantive
    )
    rejection = next(content for content in substantive if "320px" in content)
    assert "reviewer" in rejection.lower()
    assert "Change the fixed width" in rejection
    assert any("re-opened the corrected card" in content for content in substantive)
    assert any("reviewer approved" in content.lower() for content in substantive)
    assert any("Job complete" in content for content in substantive)


def test_auth_decision_is_complete_and_incomplete_request_fails_closed(
    jobs_path, session_db
):
    key = Ed25519PrivateKey.generate()
    conn = jdb.connect(jobs_path)
    try:
        job_id = jdb.create_job(
            conn,
            name="Login",
            goal="Run provider",
            requested_lane="codex",
            origin=_origin(),
        )
        attempt = _start(conn, job_id, worker="builder", at=2)
        decision = {
            "question": "OpenAI provider on the codex lane needs authentication. Which option should Hermes use?",
            "options": [
                {
                    "id": "login",
                    "label": "Log in to OpenAI",
                    "consequence": "The codex lane resumes this Job.",
                },
                {
                    "id": "stop",
                    "label": "Stop this Job",
                    "consequence": "The Job remains safely blocked.",
                },
            ],
            "recommendation": "login",
            "recommendation_reason": "Login is the smallest safe action for this provider and lane.",
            "blocked_scope": "Only Job #1 on provider OpenAI / lane codex is blocked.",
            "safe_state": "No provider call or workspace mutation occurred.",
            "next_owner_role": "builder",
        }
        _record(
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="ASSIGNED",
            target="BLOCKED",
            speaker_id="hermes",
            speaker_role="hermes",
            outcome="blocked",
            at=10,
            summary="OpenAI provider authentication blocked the codex lane.",
            next_action="Next owner: builder after the user chooses an option.",
            decision_request=decision,
        )
        bad_job = jdb.create_job(
            conn,
            name="Bad decision",
            goal="fail closed",
            requested_lane="codex",
            origin=_origin("bad", "chat-bad"),
        )
        bad_attempt = _start(conn, bad_job, worker="builder", at=21)
        write, envelope = _write(
            conn,
            key,
            job_id=bad_job,
            attempt_id=bad_attempt,
            source="ASSIGNED",
            target="BLOCKED",
            speaker_id="hermes",
            speaker_role="hermes",
            outcome="blocked",
            at=22,
            summary="Incomplete decision.",
            next_action="Choose safely.",
            decision_request=decision,
        )
        malformed = dict(write.handoff)
        malformed["decision_request"] = None
        write = replace(write, handoff=malformed)
        # The canonical handoff validator is the durable trust boundary.
        with pytest.raises(jobs_handoffs.HandoffValidationError):
            jdb.record_transition(conn, write, envelope)
    finally:
        conn.close()
    session_db.create_session("origin", source="telegram")
    _drain(session_db, jobs_path)
    content = session_db.get_messages("origin")[-1]["content"]
    assert "OpenAI provider" in content and "codex lane" in content
    assert "1. login" in content and "2. stop" in content
    assert "The codex lane resumes this Job" in content
    assert "The Job remains safely blocked" in content
    assert "Recommendation: login" in content
    assert "Only Job #1" in content
    assert "No provider call" in content
    assert "Next owner: builder" in content
    assert "Incomplete decision" not in content


def test_unavailable_origin_preserves_job_order_and_never_uses_decoy(
    jobs_path, session_db
):
    conn = jdb.connect(jobs_path)
    try:
        job_id = jdb.create_job(
            conn,
            name="Unavailable",
            goal="ordered",
            requested_lane="codex",
            origin=_origin("missing"),
        )
        job = jdb.get_job(conn, job_id)
        from hermes_cli.sqlite_util import write_txn

        with write_txn(conn):
            jn.enqueue_locked(
                conn,
                job_id=job_id,
                attempt_id=None,
                job_revision=2,
                milestone=jn.MILESTONE_ASSIGNED,
                payload={"number": job.number, "name": job.name},
                now=2,
            )
    finally:
        conn.close()
    session_db.create_session("decoy", source="telegram")
    first = deliver_due_notification_once(
        session_db, jobs_path=jobs_path, now=DELIVERY_NOW
    )
    assert first is not None
    assert session_db.get_messages("decoy") == []
    conn = jdb.connect(jobs_path)
    try:
        rows = jn.list_notifications(conn, job_id=job_id)
        assert [row.milestone for row in rows] == ["queued", "assigned"]
        assert rows[0].delivered_at is None
        assert rows[0].delivery_attempts == 1
        assert rows[0].next_attempt_at > DELIVERY_NOW
        assert rows[1].delivery_attempts == 0
        retry_at = rows[0].next_attempt_at
    finally:
        conn.close()

    assert (
        deliver_due_notification_once(session_db, jobs_path=jobs_path, now=retry_at)
        is not None
    )
    assert session_db.get_messages("decoy") == []
    conn = jdb.connect(jobs_path)
    try:
        rows = jn.list_notifications(conn, job_id=job_id)
        assert rows[0].delivery_attempts == 2
        assert rows[0].delivered_at is None
        assert rows[1].delivery_attempts == 0
    finally:
        conn.close()
