from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_graph as graph
from hermes_cli import jobs_handoffs as handoffs
from hermes_cli import jobs_receipts as receipts


LEGAL = {
    None: {"QUEUED"},
    "QUEUED": {"ASSIGNED", "FAILED", "BLOCKED", "CANCELLED"},
    "ASSIGNED": {"BUILDING", "FAILED", "BLOCKED", "CANCELLED"},
    "BUILDING": {
        "EVIDENCE_COLLECTING",
        "FAILED",
        "BLOCKED",
        "CANCELLED",
    },
    "EVIDENCE_COLLECTING": {
        "REVIEWING",
        "FAILED",
        "BLOCKED",
        "CANCELLED",
    },
    "REVIEWING": {"VERIFIED", "FAILED", "BLOCKED", "CANCELLED"},
    "VERIFIED": {"COMPLETED", "FAILED", "BLOCKED", "CANCELLED"},
    "COMPLETED": set(),
    "FAILED": set(),
    "BLOCKED": set(),
    "CANCELLED": set(),
}

REQUIRED = {
    (None, "QUEUED"): {"attempt_created"},
    ("QUEUED", "ASSIGNED"): {"preflight", "route", "lane_health"},
    ("ASSIGNED", "BUILDING"): {"claim", "worktree", "attempt_started"},
    ("BUILDING", "EVIDENCE_COLLECTING"): {"executor_exit", "output_capture"},
    ("EVIDENCE_COLLECTING", "REVIEWING"): {"tests", "readback"},
    ("REVIEWING", "VERIFIED"): {"themis_review", "receipt_verification"},
    ("VERIFIED", "COMPLETED"): {"activation_gate", "completion_receipt"},
}

FORWARD = [
    "QUEUED",
    "ASSIGNED",
    "BUILDING",
    "EVIDENCE_COLLECTING",
    "REVIEWING",
    "VERIFIED",
    "COMPLETED",
]
STATES = tuple(state for state in LEGAL if state is not None)
ILLEGAL_CASES = tuple(
    (source, target)
    for source, allowed in LEGAL.items()
    for target in STATES
    if target not in allowed
)


class GraphRig:
    def __init__(self, conn):
        self.conn = conn
        self.private_key = Ed25519PrivateKey.generate()
        self.verifier = graph.ReceiptVerifier(
            trusted_keys={"lane:test:v1": self.private_key.public_key()}
        )
        self.job_id = jdb.create_job(
            conn, requested_lane="claude", name="graph", goal="goal"
        )
        claim = jdb.claim_job(
            conn,
            specialist="claude-builder",
            worker="worker",
            lease_seconds=60,
            job=self.job_id,
            now=1,
        )
        self.attempt_id = jdb.start_attempt(
            conn,
            self.job_id,
            claim_token=claim.claim_token,
            specialist="worker",
            base_commit="b" * 40,
            commit="c" * 40,
            now=2,
        )
        self.counter = 0

    def latest_state(self):
        latest = jdb.latest_transition(self.conn, self.attempt_id)
        return None if latest is None else latest["target_state"]

    def request(self, target, *, expected_job_revision=None, evidence=None):
        source = self.latest_state()
        self.counter += 1
        if evidence is None:
            names = REQUIRED.get((source, target), {"terminal_evidence"})
            evidence = {
                name: "sha256:" + f"{self.counter:064x}"[-64:] for name in names
            }
        failure_class = "TASK_FAILURE" if target == "FAILED" else None
        blocker_code = None
        if target == "BLOCKED":
            blocker_code = "BLOCKED_BY_TEST"
        elif target == "CANCELLED":
            blocker_code = "CANCELLED_BY_TEST"
        idempotency_key = f"{source or 'NONE'}:{target}:{self.counter}"
        created_at = 10 + self.counter
        request = graph.TransitionRequest(
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            source_state=source,
            target_state=target,
            initiator_type="system",
            initiator_id="test",
            expected_job_revision=(
                jdb.get_job(self.conn, self.job_id).revision
                if expected_job_revision is None
                else expected_job_revision
            ),
            evidence=evidence,
            failure_class=failure_class,
            blocker_code=blocker_code,
            commit="c" * 40,
            receipt_id="r_" + f"{self.counter:024x}"[-24:],
            component="jobs_graph",
            component_version="1",
            idempotency_key=idempotency_key,
            created_at=created_at,
        )

        if target not in graph.LEGAL.get(source, frozenset()) or target == "CANCELLED":
            return request
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
        digest = next(iter(evidence.values()))
        raw = {
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
        }
        handoff = handoffs.normalize_handoff(
            raw,
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            speaker_id="test",
            speaker_role="system",
            speaker_executor="hermes",
            from_phase="ATTEMPT_CREATED" if source is None else source,
            to_phase=target,
            next_owner_role=None if outcome == "completed" else "next-worker",
            outcome=outcome,
            artifact_identity=None,
            transition_evidence=dict(evidence),
            created_at=created_at,
        )
        return replace(request, handoff=handoff)

    def sign(
        self,
        request,
        *,
        private_key=None,
        key_id="lane:test:v1",
        handoff=None,
    ):
        return receipts.sign_receipt(
            {
                "job_id": request.job_id,
                "attempt_id": request.attempt_id,
                "commit": request.commit,
                "state": request.target_state,
                "evidence": dict(request.evidence),
                "handoff": request.handoff if handoff is None else handoff,
                "timestamp": "2026-08-09T00:00:00Z",
                "transitioned_by": "jobs_graph/1",
            },
            receipt_id=request.receipt_id,
            key_id=key_id,
            private_key=private_key or self.private_key,
        )

    def transition(self, target):
        request = self.request(target)
        return graph.transition_attempt(
            self.conn,
            request,
            envelope=self.sign(request),
            verifier=self.verifier,
        )

    def advance(self, target):
        if target is None:
            return
        if target in {"FAILED", "BLOCKED", "CANCELLED"}:
            self.transition("QUEUED")
            self.transition(target)
            return
        for state in FORWARD:
            if self.latest_state() == state:
                if state == target:
                    return
                continue
            self.transition(state)
            if state == target:
                return

    def rows(self):
        return jdb.list_transitions(self.conn, self.job_id)

    def receipt_count(self):
        return len(jdb.get_receipts(self.conn, self.job_id))


@pytest.fixture
def graph_rig(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    try:
        yield GraphRig(conn)
    finally:
        conn.close()


def test_legal_forward_path_is_append_only_and_maps_public_completion(graph_rig):
    graph_rig.advance("COMPLETED")

    assert [row["target_state"] for row in graph_rig.rows()] == FORWARD
    assert graph_rig.receipt_count() == len(FORWARD)
    job = jdb.get_job(graph_rig.conn, graph_rig.job_id)
    assert (job.status, job.step) == ("finished", "complete")


@pytest.mark.parametrize(("source", "target"), ILLEGAL_CASES)
def test_every_illegal_edge_fails_without_mutation(graph_rig, source, target):
    graph_rig.advance(source)
    before_rows = graph_rig.rows()
    before_receipts = graph_rig.receipt_count()
    request = graph_rig.request(target)

    with pytest.raises(graph.IllegalGraphTransition):
        graph.transition_attempt(
            graph_rig.conn,
            request,
            envelope=graph_rig.sign(request),
            verifier=graph_rig.verifier,
        )

    assert graph_rig.rows() == before_rows
    assert graph_rig.receipt_count() == before_receipts


def test_completed_attempt_cannot_transition(graph_rig):
    graph_rig.advance("COMPLETED")
    before = graph_rig.rows()
    with pytest.raises(graph.IllegalGraphTransition):
        graph_rig.transition("BUILDING")
    assert graph_rig.rows() == before


def test_stale_revision_cannot_mutate(graph_rig):
    graph_rig.advance("QUEUED")
    stale = jdb.get_job(graph_rig.conn, graph_rig.job_id).revision
    jdb.heartbeat(graph_rig.conn, graph_rig.job_id, at=50)
    request = graph_rig.request("ASSIGNED", expected_job_revision=stale)

    with pytest.raises(graph.StaleJobRevision):
        graph.transition_attempt(
            graph_rig.conn,
            request,
            envelope=graph_rig.sign(request),
            verifier=graph_rig.verifier,
        )
    assert graph_rig.latest_state() == "QUEUED"


def test_missing_required_evidence_fails_without_receipt(graph_rig):
    graph_rig.advance("QUEUED")
    request = graph_rig.request(
        "ASSIGNED", evidence={"preflight": "sha256:" + "a" * 64}
    )
    before = graph_rig.receipt_count()

    with pytest.raises(graph.MissingTransitionEvidence):
        graph.transition_attempt(
            graph_rig.conn,
            request,
            envelope=graph_rig.sign(request),
            verifier=graph_rig.verifier,
        )
    assert graph_rig.receipt_count() == before


@pytest.mark.parametrize("failure", ["unknown", "invalid", "revoked"])
def test_untrusted_receipts_fail_without_transition(graph_rig, failure):
    request = graph_rig.request("QUEUED")
    envelope = graph_rig.sign(request)
    verifier = graph_rig.verifier
    if failure == "unknown":
        verifier = graph.ReceiptVerifier(trusted_keys={})
    elif failure == "invalid":
        envelope["signing"]["signature"] = "base64:AAAA"
    else:
        verifier = graph.ReceiptVerifier(
            trusted_keys={"lane:test:v1": graph_rig.private_key.public_key()},
            revoked_key_ids={"lane:test:v1"},
        )

    with pytest.raises(receipts.ReceiptVerificationError):
        graph.transition_attempt(
            graph_rig.conn,
            request,
            envelope=envelope,
            verifier=verifier,
        )
    assert graph_rig.rows() == []
    assert graph_rig.receipt_count() == 0


def test_receipt_evidence_must_match_transition_evidence(graph_rig):
    request = graph_rig.request("QUEUED")
    envelope = graph_rig.sign(request)
    envelope["payload"]["evidence"] = {"attempt_created": "sha256:" + "f" * 64}

    with pytest.raises(receipts.ReceiptVerificationError):
        graph.transition_attempt(
            graph_rig.conn,
            request,
            envelope=envelope,
            verifier=graph_rig.verifier,
        )


def test_missing_handoff_fails_before_revision_or_transition_mutation(graph_rig):
    request = replace(graph_rig.request("QUEUED"), handoff=None)
    revision = jdb.get_job(graph_rig.conn, graph_rig.job_id).revision

    with pytest.raises(handoffs.HandoffValidationError, match="requires a handoff"):
        graph.transition_attempt(
            graph_rig.conn,
            request,
            envelope=graph_rig.sign(request),
            verifier=graph_rig.verifier,
        )

    assert jdb.get_job(graph_rig.conn, graph_rig.job_id).revision == revision
    assert graph_rig.rows() == []
    assert graph_rig.receipt_count() == 0


def test_signed_handoff_must_match_request_handoff(graph_rig):
    request = graph_rig.request("QUEUED")
    changed = dict(request.handoff)
    changed["summary"] = "Different signed facts."

    with pytest.raises(
        receipts.ReceiptVerificationError,
        match="receipt handoff identity mismatch",
    ):
        graph.transition_attempt(
            graph_rig.conn,
            request,
            envelope=graph_rig.sign(request, handoff=changed),
            verifier=graph_rig.verifier,
        )

    assert graph_rig.rows() == []
    assert graph_rig.receipt_count() == 0


def test_contradictory_handoff_fails_before_envelope_trust_or_mutation(graph_rig):
    request = graph_rig.request("QUEUED")
    changed = dict(request.handoff)
    changed["job_id"] = "different-job"
    request = replace(request, handoff=changed)

    with pytest.raises(handoffs.HandoffValidationError, match="job identity"):
        graph.transition_attempt(
            graph_rig.conn,
            request,
            envelope=graph_rig.sign(request),
            verifier=graph.ReceiptVerifier(trusted_keys={}),
        )

    assert graph_rig.rows() == []
    assert graph_rig.receipt_count() == 0


def test_idempotent_replay_requires_exact_facts_and_receipt(graph_rig):
    request = graph_rig.request("QUEUED")
    envelope = graph_rig.sign(request)
    first = graph.transition_attempt(
        graph_rig.conn,
        request,
        envelope=envelope,
        verifier=graph_rig.verifier,
    )

    repeated = graph.transition_attempt(
        graph_rig.conn,
        request,
        envelope=envelope,
        verifier=graph_rig.verifier,
    )
    assert repeated == first
    with pytest.raises(jdb.GraphConflict):
        graph.transition_attempt(
            graph_rig.conn,
            replace(request, initiator_type="other"),
            envelope=envelope,
            verifier=graph_rig.verifier,
        )
    assert len(graph_rig.rows()) == 1
    assert graph_rig.receipt_count() == 1


def test_blocked_attempt_maps_to_needs_you_without_claim_leak(graph_rig):
    graph_rig.advance("BLOCKED")

    job = jdb.get_job(graph_rig.conn, graph_rig.job_id)
    assert (job.status, job.step) == ("needs_you", "waiting_for_decision")
    assert job.claimed_by is None


def test_retryable_failed_edge_closes_attempt_before_next_attempt(graph_rig):
    graph_rig.advance("FAILED")

    first = jdb.get_attempt(graph_rig.conn, graph_rig.attempt_id)
    assert first["status"] == "failed"
    claim = jdb.claim_job(
        graph_rig.conn,
        specialist="claude-builder",
        worker="retry-worker",
        lease_seconds=60,
        job=graph_rig.job_id,
        now=100,
    )
    second_id = jdb.start_attempt(
        graph_rig.conn,
        graph_rig.job_id,
        claim_token=claim.claim_token,
        specialist="retry-worker",
        base_commit="b" * 40,
        now=101,
    )

    assert jdb.get_attempt(graph_rig.conn, second_id)["ordinal"] == 2


def test_completed_edge_closes_attempt_as_succeeded(graph_rig):
    graph_rig.advance("COMPLETED")

    assert jdb.get_attempt(graph_rig.conn, graph_rig.attempt_id)["status"] == (
        "succeeded"
    )


def test_projection_is_read_only_deterministic_and_preserves_history(tmp_path):
    path = tmp_path / "jobs.db"
    first = jdb.connect(path)
    rig = GraphRig(first)
    rig.advance("COMPLETED")
    before_revision = jdb.get_job(first, rig.job_id).revision
    first_projection = graph.compute_work_control(first, now=100)
    assert jdb.get_job(first, rig.job_id).revision == before_revision
    first.close()
    jdb._INITIALIZED_PATHS.discard(str(path.resolve()))

    second = jdb.connect(path)
    try:
        second_projection = graph.compute_work_control(second, now=100)
    finally:
        second.close()

    assert second_projection == first_projection
    job = first_projection["jobs"][0]
    assert job["state"] == "COMPLETED"
    assert job["status"] == "finished"
    assert [row["target_state"] for row in job["attempts"][0]["transitions"]] == FORWARD
    assert "claim_token" not in receipts.canonical_json_bytes(first_projection).decode()
