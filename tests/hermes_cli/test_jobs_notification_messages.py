"""Task 8: concise, stable, user-facing lifecycle message rendering.

The renderer is a pure function over the bounded outbox payload: it maps
every durable milestone to exactly one deterministic user-facing message and
never leaks raw goal text, stdout, stack traces, tokens, or full review text
into a chat.  ``re_review`` is an explicit flag — the stable ``review``
milestone type renders as re-review only when the delivery worker knows a
correcting milestone preceded it.
"""

import pytest

from hermes_cli import jobs_notifications as jn
from hermes_cli import jobs_handoffs
from hermes_cli.jobs_notifications import NotificationRecord


def _record(milestone, **payload_overrides):
    payload = {"number": 3, "name": "landing page"}
    payload.update(payload_overrides)
    return NotificationRecord(
        id=1,
        notification_id=f"n:job-1:1:{milestone}",
        job_id="job-1",
        attempt_id=None,
        job_revision=1,
        milestone=milestone,
        payload=payload,
        transition_id=None,
        created_at=0,
        next_attempt_at=0,
        claim_owner=None,
        claim_expires_at=None,
        delivery_attempts=0,
        last_error=None,
        delivered_at=None,
        blocked_reason=None,
    )


def test_exact_wording_per_milestone():
    expected = {
        jn.MILESTONE_QUEUED: "Legacy Jobs update — Queued — landing page (#3) accepted",
        jn.MILESTONE_ASSIGNED: "Legacy Jobs update — Assigned — landing page (#3) started",
        jn.MILESTONE_BUILDING: "Legacy Jobs update — Building — landing page (#3) in progress",
        jn.MILESTONE_TESTING: (
            "Legacy Jobs update — Testing — landing page (#3) running tests and evidence"
        ),
        jn.MILESTONE_REVIEW: "Legacy Jobs update — Review — landing page (#3) under review",
        jn.MILESTONE_CORRECTING: (
            "Legacy Jobs update — Correcting — landing page (#3) fixing findings"
        ),
        jn.MILESTONE_NEEDS_YOU: (
            "Legacy Jobs update — Needs you — landing page (#3) requires your decision"
        ),
        jn.MILESTONE_FAILURE: (
            "Legacy Jobs update — Failed — landing page (#3) ended unsuccessfully"
        ),
        jn.MILESTONE_FINISHED: "Legacy Jobs update — Finished — landing page (#3) complete",
    }
    for milestone, wording in expected.items():
        assert jn.render_milestone_message(_record(milestone)) == wording


def test_re_review_wording_only_applies_to_review():
    record = _record(jn.MILESTONE_REVIEW)
    assert (
        jn.render_milestone_message(record)
        == "Legacy Jobs update — Review — landing page (#3) under review"
    )
    assert (
        jn.render_milestone_message(record, re_review=True)
        == "Legacy Jobs update — Re-review — landing page (#3) back under review"
    )
    # The flag must never change another milestone's wording.
    corrected = _record(jn.MILESTONE_CORRECTING)
    assert (
        jn.render_milestone_message(corrected, re_review=True)
        == "Legacy Jobs update — Correcting — landing page (#3) fixing findings"
    )


def test_heartbeat_wording_identifies_the_stagnant_phase():
    record = _record(jn.MILESTONE_HEARTBEAT, phase=jn.MILESTONE_BUILDING)
    assert (
        jn.render_milestone_message(record)
        == "Legacy Jobs update — Still working — landing page (#3) (still building)"
    )


def test_message_never_leaks_goal_stdout_stack_tokens_or_review_text():
    forbidden = (
        "goal",
        "stdout",
        "traceback",
        "stack trace",
        "tokens",
        "review text",
    )
    # Even a hostile payload carrying sensitive-looking fields cannot leak
    # them into the user-facing message: rendering reads only number/name/
    # phase and ignores everything else.
    hostile = _record(
        jn.MILESTONE_FAILURE,
        goal="top secret goal",
        stdout="stdout dump",
        stack_trace="traceback lines",
        tokens_used=12345,
        review_text="full review body",
        failure_class="provider-timeout",
    )
    text = jn.render_milestone_message(hostile)
    for token in forbidden:
        assert token not in text.lower()


def test_name_falls_back_to_job_id():
    record = _record(jn.MILESTONE_BUILDING, name=None, number=None)
    text = jn.render_milestone_message(record)
    assert "job-1" in text
    assert "landing page" not in text


def _handoff(
    *, outcome="handed_off", from_phase="BUILDING", to_phase="EVIDENCE_COLLECTING"
):
    digest = "sha256:" + "a" * 64
    return jobs_handoffs.normalize_handoff(
        {
            "summary": "Builder completed the scoped change.",
            "evidence_summary": [
                {"label": "Focused tests", "result": "42 passed", "digest": digest}
            ],
            "next_action": "Review the committed change.",
            "issues": [],
            "decision_request": None,
        },
        job_id="job-1",
        attempt_id="attempt-1",
        speaker_id="builder-1",
        speaker_role="builder",
        speaker_executor="claude",
        from_phase=from_phase,
        to_phase=to_phase,
        next_owner_role="reviewer",
        outcome=outcome,
        artifact_identity=None,
        transition_evidence={"tests": digest},
        created_at=1,
    )


def test_valid_handoff_renders_one_substantive_agent_message():
    digest = "sha256:" + "a" * 64
    record = _record(
        jn.MILESTONE_TESTING,
        target_state="EVIDENCE_COLLECTING",
        handoff=_handoff(),
    )
    text = jn.render_milestone_message(
        record,
        transition_evidence={"tests": digest},
        expected_job_id="job-1",
        expected_attempt_id="attempt-1",
        expected_target_state="EVIDENCE_COLLECTING",
        expected_speaker_id="builder-1",
    )
    assert text.startswith("builder → reviewer\nBuilder completed")
    assert "Legacy Jobs update" not in text


def test_malformed_handoff_fails_closed_without_echoing_facts():
    record = _record(
        jn.MILESTONE_REVIEW,
        target_state="VERIFIED",
        handoff={"summary": "secret outcome", "speaker_role": "builder"},
    )
    text = jn.render_milestone_message(record, transition_evidence={})
    assert text == "Hermes could not explain this handoff — Job #3 is current state."
    assert "secret outcome" not in text


def test_malformed_diagnostic_does_not_echo_untrusted_state_or_milestone():
    record = _record(
        "token=supersecret",
        target_state="authorization: leaked",
        handoff={"summary": "do not echo"},
    )
    text = jn.render_milestone_message(record)
    assert text == "Hermes could not explain this handoff — Job #3 is current state."
    assert "supersecret" not in text
    assert "leaked" not in text


def test_review_approved_milestone_is_exported_and_maps_verified():
    assert jn.MILESTONE_REVIEW_APPROVED in jn.ALL_MILESTONES
    assert jn.milestone_for_state("VERIFIED") == jn.MILESTONE_REVIEW_APPROVED


@pytest.mark.parametrize(
    "override",
    [
        {"job_id": "job-foreign"},
        {"attempt_id": "attempt-foreign"},
        {"to_phase": "REVIEWING"},
        {"speaker_id": "reviewer-foreign"},
    ],
)
def test_handoff_identity_tampering_fails_closed(override):
    digest = "sha256:" + "a" * 64
    handoff = _handoff()
    if "job_id" in override:
        handoff["job_id"] = override["job_id"]
    elif "attempt_id" in override:
        handoff["attempt_id"] = override["attempt_id"]
    elif "to_phase" in override:
        handoff["to_phase"] = override["to_phase"]
    else:
        handoff["speaker_id"] = override["speaker_id"]
    record = _record(
        jn.MILESTONE_TESTING,
        target_state="EVIDENCE_COLLECTING",
        handoff=handoff,
    )
    text = jn.render_milestone_message(
        record,
        transition_evidence={"tests": digest},
        expected_job_id="job-1",
        expected_attempt_id="attempt-1",
        expected_target_state="EVIDENCE_COLLECTING",
        expected_speaker_id="builder-1",
    )
    assert text.startswith("Hermes could not explain this handoff")
    assert "builder → reviewer" not in text
