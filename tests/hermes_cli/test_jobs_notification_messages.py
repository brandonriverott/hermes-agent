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
        jn.MILESTONE_QUEUED: "Queued — landing page (#3) accepted",
        jn.MILESTONE_ASSIGNED: "Assigned — landing page (#3) started",
        jn.MILESTONE_BUILDING: "Building — landing page (#3) in progress",
        jn.MILESTONE_TESTING: (
            "Testing — landing page (#3) running tests and evidence"
        ),
        jn.MILESTONE_REVIEW: "Review — landing page (#3) under review",
        jn.MILESTONE_CORRECTING: (
            "Correcting — landing page (#3) fixing findings"
        ),
        jn.MILESTONE_NEEDS_YOU: (
            "Needs you — landing page (#3) requires your decision"
        ),
        jn.MILESTONE_FAILURE: (
            "Failed — landing page (#3) ended unsuccessfully"
        ),
        jn.MILESTONE_FINISHED: "Finished — landing page (#3) complete",
    }
    for milestone, wording in expected.items():
        assert jn.render_milestone_message(_record(milestone)) == wording


def test_re_review_wording_only_applies_to_review():
    record = _record(jn.MILESTONE_REVIEW)
    assert (
        jn.render_milestone_message(record)
        == "Review — landing page (#3) under review"
    )
    assert (
        jn.render_milestone_message(record, re_review=True)
        == "Re-review — landing page (#3) back under review"
    )
    # The flag must never change another milestone's wording.
    corrected = _record(jn.MILESTONE_CORRECTING)
    assert (
        jn.render_milestone_message(corrected, re_review=True)
        == "Correcting — landing page (#3) fixing findings"
    )


def test_heartbeat_wording_identifies_the_stagnant_phase():
    record = _record(jn.MILESTONE_HEARTBEAT, phase=jn.MILESTONE_BUILDING)
    assert (
        jn.render_milestone_message(record)
        == "Still working — landing page (#3) (still building)"
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
