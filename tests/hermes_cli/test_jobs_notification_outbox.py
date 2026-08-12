"""Transactional, ordered notification outbox behavior for the Jobs ledger.

Every user-visible milestone is written inside the same SQLite write
transaction as the Job creation or graph transition that produced it, so a
state cannot commit without its notification intent and notification intent
cannot exist for a state that did not commit.  Ordering is the outbox row
order; delivery may never overtake an earlier undelivered milestone of the
same Job.
"""

import sqlite3
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_handoffs as handoffs
from hermes_cli import jobs_notifications as jn
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
    evidence,
    handoff,
):
    return receipts.sign_receipt(
        {
            "job_id": job_id,
            "attempt_id": attempt_id,
            "commit": "c" * 40,
            "state": state,
            "evidence": dict(evidence),
            "handoff": handoff,
            "timestamp": "2026-08-09T00:00:00Z",
            "transitioned_by": "jobs_graph/1",
        },
        receipt_id=receipt_id,
        key_id="lane:test:v1",
        private_key=signing_key,
    )


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


def _queued_job(conn, *, origin=None):
    return jdb.create_job(
        conn,
        name="job",
        goal="goal",
        requested_lane="codex",
        origin=_origin() if origin is None else origin,
    )


def _attempt(conn, job_id):
    claim = jdb.claim_job(
        conn,
        specialist="codex-builder",
        worker="worker",
        lease_seconds=60,
        job=job_id,
        now=1,
    )
    return jdb.start_attempt(
        conn,
        job_id,
        claim_token=claim.claim_token,
        specialist="worker",
        base_commit="b" * 40,
        commit="c" * 40,
        now=2,
    )


def _transition(
    conn,
    signing_key,
    *,
    job_id,
    attempt_id,
    source,
    target,
    key,
    created_at,
    failure_class=None,
    blocker_code=None,
    receipt_id=None,
):
    revision = jdb.get_job(conn, job_id).revision
    write = jdb.TransitionWrite(
        job_id=job_id,
        attempt_id=attempt_id,
        source_state=source,
        target_state=target,
        initiator_type="system",
        initiator_id="test",
        expected_job_revision=revision,
        evidence={"attempt_created": "sha256:" + "a" * 64},
        failure_class=failure_class,
        blocker_code=blocker_code,
        receipt_id=receipt_id or f"r_{key}",
        component="jobs_graph",
        component_version="1",
        idempotency_key=key,
        created_at=created_at,
    )
    outcomes = {
        "QUEUED": "started",
        "ASSIGNED": "handed_off",
        "BUILDING": "started",
        "EVIDENCE_COLLECTING": "handed_off",
        "REVIEWING": "handed_off",
        "VERIFIED": "passed",
        "COMPLETED": "completed",
        "FAILED": "rejected",
        "BLOCKED": "blocked",
    }
    outcome = outcomes[target]
    digest = next(iter(write.evidence.values()))
    transition_handoff = handoffs.normalize_handoff(
        {
            "summary": f"Transitioned to {target}.",
            "evidence_summary": [
                {
                    "label": "Transition evidence",
                    "result": "Recorded.",
                    "digest": digest,
                }
            ],
            "next_action": "Continue the bounded Job workflow.",
            "issues": (
                [
                    {
                        "requirement": "The transition must succeed.",
                        "finding": "The attempt failed.",
                        "required_fix": "Correct the failure before retrying.",
                    }
                ]
                if outcome == "rejected"
                else []
            ),
            "decision_request": (
                {
                    "question": "How should this Job continue?",
                    "options": [
                        {
                            "id": "retry",
                            "label": "Retry",
                            "consequence": "Retry after the blocker is cleared.",
                        },
                        {
                            "id": "stop",
                            "label": "Stop",
                            "consequence": "Leave the Job blocked.",
                        },
                    ],
                    "recommendation": "retry",
                    "recommendation_reason": "The bounded retry is safe.",
                    "blocked_scope": "Only this Job is blocked.",
                    "safe_state": "No additional mutation occurred.",
                    "next_owner_role": "builder",
                }
                if outcome == "blocked"
                else None
            ),
        },
        job_id=job_id,
        attempt_id=attempt_id,
        speaker_id="test",
        speaker_role="system",
        speaker_executor="hermes",
        from_phase="ATTEMPT_CREATED" if source is None else source,
        to_phase=target,
        next_owner_role=None if outcome == "completed" else "next-worker",
        outcome=outcome,
        artifact_identity=None,
        transition_evidence=dict(write.evidence),
        created_at=created_at,
    )
    write = replace(write, handoff=transition_handoff)
    envelope = _signed(
        signing_key,
        receipt_id=write.receipt_id,
        job_id=job_id,
        attempt_id=attempt_id,
        state=target,
        evidence=write.evidence,
        handoff=write.handoff,
    )
    return write, envelope


def _record_queued(conn, signing_key, job_id, attempt_id, *, created_at):
    write, envelope = _transition(
        conn,
        signing_key,
        job_id=job_id,
        attempt_id=attempt_id,
        source=None,
        target="QUEUED",
        key=f"t:queued:{attempt_id}",
        created_at=created_at,
    )
    jdb.record_transition(conn, write, envelope)


def test_create_job_with_exact_origin_atomically_queues_one_notification(
    conn,
):
    job_id = _queued_job(conn)

    rows = jn.list_notifications(conn, job_id=job_id)
    assert len(rows) == 1
    row = rows[0]
    assert row.milestone == "queued"
    assert row.job_revision == 1
    assert row.attempt_id is None
    assert row.transition_id is None
    assert row.delivered_at is None
    assert row.payload["number"] == jdb.get_job(conn, job_id).number
    assert row.payload["requested_lane"] == "codex"
    # The immutable origin reference lives on the Job; the outbox row is
    # reachable from it without copying any origin bytes.
    origin_row = conn.execute(
        "SELECT origin FROM job_origins WHERE job_id = ?", (job_id,)
    ).fetchone()
    assert origin_row is not None
    assert row.job_id == job_id


def test_originless_operator_job_creates_no_notification(conn):
    job_id = jdb.create_job(
        conn,
        name="operator",
        goal="goal",
        requested_lane="claude",
    )
    assert jn.list_notifications(conn, job_id=job_id) == []
    assert jn.list_notifications(conn) == []


def test_each_defined_phase_transition_creates_one_ordered_intent(conn, signing_key):
    job_id = _queued_job(conn)
    attempt_id = _attempt(conn, job_id)

    steps = [
        (None, "QUEUED", "t:queued", 10, None, None),
        ("QUEUED", "ASSIGNED", "t:assigned", 11, None, None),
        ("ASSIGNED", "BUILDING", "t:building", 12, None, None),
        ("BUILDING", "EVIDENCE_COLLECTING", "t:testing", 13, None, None),
        ("EVIDENCE_COLLECTING", "REVIEWING", "t:review", 14, None, None),
        ("REVIEWING", "VERIFIED", "t:verified", 15, None, None),
        ("VERIFIED", "COMPLETED", "t:finished", 16, None, None),
    ]
    visible_handoffs = []
    for source, target, key, at, failure_class, blocker_code in steps:
        write, envelope = _transition(
            conn,
            signing_key,
            job_id=job_id,
            attempt_id=attempt_id,
            source=source,
            target=target,
            key=key,
            created_at=at,
        )
        jdb.record_transition(conn, write, envelope)
        if jn.milestone_for_state(target) is not None:
            visible_handoffs.append(write.handoff)

    rows = jn.list_notifications(conn, job_id=job_id)
    # ``queued`` from creation, then one ordered intent per user milestone.
    # BUILDING is a signed internal edge, not a second visible update.
    assert [r.milestone for r in rows] == [
        "queued",
        "assigned",
        "testing",
        "review",
        "review-approved",
        "finished",
    ]
    assert [row.payload["handoff"] for row in rows[1:]] == visible_handoffs
    assert [r.id for r in rows] == sorted(r.id for r in rows)
    assert all(r.delivered_at is None for r in rows)


def test_blocked_and_terminal_failure_milestones(conn, signing_key):
    job_id = _queued_job(conn)
    attempt_id = _attempt(conn, job_id)
    _record_queued(conn, signing_key, job_id, attempt_id, created_at=9)

    write, envelope = _transition(
        conn,
        signing_key,
        job_id=job_id,
        attempt_id=attempt_id,
        source="QUEUED",
        target="BLOCKED",
        key="t:blocked",
        created_at=10,
        blocker_code="AUTH_LOGIN_REQUIRED",
    )
    jdb.record_transition(conn, write, envelope)
    assert [r.milestone for r in jn.list_notifications(conn, job_id=job_id)] == [
        "queued",
        "needs-you",
    ]

    failed_job = _queued_job(
        conn, origin=_origin(chat_id="chat-2", session_id="session-2")
    )
    terminal_attempt = _attempt(conn, failed_job)
    _record_queued(conn, signing_key, failed_job, terminal_attempt, created_at=19)
    conn.execute(
        "UPDATE job_attempts SET terminal_failure = 1 WHERE id = ?",
        (terminal_attempt,),
    )
    conn.commit()
    write, envelope = _transition(
        conn,
        signing_key,
        job_id=failed_job,
        attempt_id=terminal_attempt,
        source="QUEUED",
        target="FAILED",
        key="t:terminal-failure",
        created_at=20,
        failure_class="implementation",
    )
    jdb.record_transition(conn, write, envelope)
    assert [r.milestone for r in jn.list_notifications(conn, job_id=failed_job)] == [
        "queued",
        "failure",
    ]


def test_nonterminal_failure_creates_correcting_milestone(conn, signing_key):
    job_id = _queued_job(conn)
    attempt_id = _attempt(conn, job_id)
    _record_queued(conn, signing_key, job_id, attempt_id, created_at=9)

    write, envelope = _transition(
        conn,
        signing_key,
        job_id=job_id,
        attempt_id=attempt_id,
        source="QUEUED",
        target="FAILED",
        key="t:correcting",
        created_at=10,
        failure_class="reviewer_rejection",
    )
    jdb.record_transition(conn, write, envelope)
    assert [r.milestone for r in jn.list_notifications(conn, job_id=job_id)] == [
        "queued",
        "correcting",
    ]


def test_duplicate_transition_key_does_not_duplicate_messages(conn, signing_key):
    job_id = _queued_job(conn)
    attempt_id = _attempt(conn, job_id)
    _record_queued(conn, signing_key, job_id, attempt_id, created_at=9)
    write, envelope = _transition(
        conn,
        signing_key,
        job_id=job_id,
        attempt_id=attempt_id,
        source="QUEUED",
        target="ASSIGNED",
        key="t:same",
        created_at=10,
    )

    first_id = jdb.record_transition(conn, write, envelope)
    second_id = jdb.record_transition(conn, write, envelope)

    assert first_id == second_id
    rows = jn.list_notifications(conn, job_id=job_id)
    assert [r.milestone for r in rows] == ["queued", "assigned"]
    assert len(rows) == 2
    assert rows[-1].payload["handoff"] == write.handoff


def test_building_is_internal_and_verified_is_review_approved():
    assert jn.milestone_for_state("BUILDING") is None
    assert jn.milestone_for_state("VERIFIED") == "review-approved"


def test_outbox_insert_failure_rolls_back_job_creation(conn):
    conn.execute("DROP TABLE job_notifications")
    conn.commit()

    with pytest.raises(sqlite3.OperationalError):
        _queued_job(conn)

    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM job_origins").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 0


def test_outbox_insert_failure_rolls_back_transition_write(conn, signing_key):
    job_id = _queued_job(conn)
    attempt_id = _attempt(conn, job_id)
    _record_queued(conn, signing_key, job_id, attempt_id, created_at=9)
    revision_before = jdb.get_job(conn, job_id).revision
    transition_count = conn.execute(
        "SELECT COUNT(*) FROM job_attempt_transitions"
    ).fetchone()[0]
    receipt_count = conn.execute("SELECT COUNT(*) FROM job_receipts").fetchone()[0]
    write, envelope = _transition(
        conn,
        signing_key,
        job_id=job_id,
        attempt_id=attempt_id,
        source="QUEUED",
        target="ASSIGNED",
        key="t:rollback",
        created_at=10,
    )

    conn.execute("DROP TABLE job_notifications")
    conn.commit()

    with pytest.raises(sqlite3.OperationalError):
        jdb.record_transition(conn, write, envelope)

    assert (
        conn.execute("SELECT COUNT(*) FROM job_attempt_transitions").fetchone()[0]
        == transition_count
    )
    assert (
        conn.execute("SELECT COUNT(*) FROM job_receipts").fetchone()[0] == receipt_count
    )
    assert jdb.get_job(conn, job_id).revision == revision_before


def test_conflicting_outbox_intent_rolls_back_transition_and_handoff(conn, signing_key):
    job_id = _queued_job(conn)
    attempt_id = _attempt(conn, job_id)
    _record_queued(conn, signing_key, job_id, attempt_id, created_at=9)
    revision_before = jdb.get_job(conn, job_id).revision
    transition_count = conn.execute(
        "SELECT COUNT(*) FROM job_attempt_transitions"
    ).fetchone()[0]
    write, envelope = _transition(
        conn,
        signing_key,
        job_id=job_id,
        attempt_id=attempt_id,
        source="QUEUED",
        target="ASSIGNED",
        key="t:conflicting-outbox",
        created_at=10,
    )
    jn.enqueue_locked(
        conn,
        job_id=job_id,
        attempt_id=attempt_id,
        job_revision=write.expected_job_revision + 1,
        milestone=jn.MILESTONE_ASSIGNED,
        transition_id=None,
        payload={"handoff": {"summary": "different"}},
        now=10,
    )
    conn.commit()

    with pytest.raises(jdb.GraphConflict, match="notification intent"):
        jdb.record_transition(conn, write, envelope)

    assert (
        conn.execute("SELECT COUNT(*) FROM job_attempt_transitions").fetchone()[0]
        == transition_count
    )
    assert jdb.get_job(conn, job_id).revision == revision_before
    assert jdb.latest_handoff(conn, attempt_id)["to_phase"] == "QUEUED"


def test_pending_rows_claim_retry_acknowledge_in_order_per_job(
    conn, tmp_path, signing_key
):
    import time

    job_id = _queued_job(conn)
    attempt_id = _attempt(conn, job_id)
    _record_queued(conn, signing_key, job_id, attempt_id, created_at=9)
    steps = [
        ("QUEUED", "ASSIGNED", "t:assigned", 10),
        ("ASSIGNED", "BUILDING", "t:building", 11),
        ("BUILDING", "EVIDENCE_COLLECTING", "t:testing", 12),
    ]
    for source, target, key, at in steps:
        write, envelope = _transition(
            conn,
            signing_key,
            job_id=job_id,
            attempt_id=attempt_id,
            source=source,
            target=target,
            key=key,
            created_at=at,
        )
        jdb.record_transition(conn, write, envelope)

    base = int(time.time())
    other = jdb.connect(tmp_path / "jobs.db")
    try:
        queued, assigned, testing = jn.list_notifications(conn, job_id=job_id)
        assert [r.milestone for r in (queued, assigned, testing)] == [
            "queued",
            "assigned",
            "testing",
        ]

        # Worker A claims the earliest pending row.
        claimed = jn.claim_due(conn, owner="A", now=base, lease_seconds=30)
        assert claimed.notification_id == queued.notification_id
        assert claimed.delivery_attempts == 1
        assert claimed.claim_expires_at == base + 30

        # Worker B cannot overtake: the earlier row is claimed, and even if it
        # were due, a later milestone must wait for the earlier one.
        assert jn.claim_due(other, owner="B", now=base, lease_seconds=30) is None

        # A delivers and acknowledges; only then does the next row become
        # claimable — and only to whoever claims it first.
        jn.acknowledge(conn, queued.notification_id, owner="A", now=base + 1)
        with pytest.raises(jn.NotificationClaimError):
            jn.acknowledge(conn, queued.notification_id, owner="B", now=base + 1)

        second = jn.claim_due(other, owner="B", now=base + 2, lease_seconds=30)
        assert second.notification_id == assigned.notification_id
        assert jn.claim_due(conn, owner="A", now=base + 2, lease_seconds=30) is None

        # B releases with a bounded retry delay; the row is not claimable
        # before its next attempt time, then returns to the queue in order.
        jn.release_claim(
            other,
            assigned.notification_id,
            owner="B",
            now=base + 3,
            retry_at=base + 100,
            error="destination_busy",
        )
        assert jn.claim_due(conn, owner="A", now=base + 3, lease_seconds=30) is None
        retried = jn.claim_due(conn, owner="A", now=base + 101, lease_seconds=30)
        assert retried.notification_id == assigned.notification_id
        assert retried.delivery_attempts == 2
        assert retried.last_error == "destination_busy"

        # Acknowledging the second row unblocks the third.
        jn.acknowledge(conn, assigned.notification_id, owner="A", now=base + 102)
        third = jn.claim_due(other, owner="B", now=base + 103, lease_seconds=30)
        assert third.notification_id == testing.notification_id
        jn.acknowledge(other, testing.notification_id, owner="B", now=base + 104)

        assert jn.claim_due(conn, owner="A", now=base + 105, lease_seconds=30) is None
        assert jn.claim_due(other, owner="B", now=base + 105, lease_seconds=30) is None
        delivered = jn.list_notifications(conn, job_id=job_id)
        assert [r.delivered_at for r in delivered] == [
            base + 1,
            base + 102,
            base + 104,
        ]
    finally:
        other.close()


def test_unavailable_older_job_does_not_starve_a_new_job_notification(conn):
    """Delivery order is strict within a Job, never a global backlog lock."""
    import time

    now = int(time.time())
    older = _queued_job(conn)
    newer = _queued_job(conn)
    older_row = jn.list_notifications(conn, job_id=older)[0]
    newer_row = jn.list_notifications(conn, job_id=newer)[0]

    claimed = jn.claim_due(conn, owner="notifier", now=now + 1)

    assert claimed.notification_id == newer_row.notification_id
    assert claimed.notification_id != older_row.notification_id
