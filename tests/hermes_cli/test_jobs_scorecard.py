from hermes_cli import jobs_scorecard
from hermes_cli import jobs_db


def test_weekly_scorecard_distinguishes_settled_from_verified_and_false_green():
    observations = [
        jobs_scorecard.JobObservation(
            job_id="verified", started=True, settled=True, outcome_verified=True
        ),
        jobs_scorecard.JobObservation(
            job_id="false-complete",
            started=True,
            settled=True,
            false_complete=True,
            missing_evidence=2,
            failure_signatures=("MISSING_OUTCOME_PROOF",),
        ),
        jobs_scorecard.JobObservation(
            job_id="retry-spin",
            started=True,
            settled=False,
            retry_without_progress=1,
            failure_signatures=("RETRY_WITHOUT_PROGRESS",),
        ),
    ]

    report = jobs_scorecard.build_scorecard(
        observations, period_start=100, period_end=200
    )

    assert report.jobs_started == 3
    assert report.jobs_settled == 2
    assert report.outcomes_verified == 1
    assert report.verification_rate == 0.5
    assert report.false_complete == 1
    assert report.retry_without_progress == 1
    assert report.missing_evidence == 2


def test_recurring_failure_signatures_become_guard_candidates():
    observations = [
        jobs_scorecard.JobObservation(
            job_id="a",
            started=True,
            settled=False,
            failure_signatures=("STALE_EXECUTED_COPY",),
        ),
        jobs_scorecard.JobObservation(
            job_id="b",
            started=True,
            settled=False,
            failure_signatures=("STALE_EXECUTED_COPY", "AUTH_EXPIRED"),
        ),
    ]

    report = jobs_scorecard.build_scorecard(
        observations, period_start=100, period_end=200, guard_threshold=2
    )

    assert report.guard_candidates == (("STALE_EXECUTED_COPY", 2),)
    assert report.to_mapping()["guard_candidates"] == [
        {"signature": "STALE_EXECUTED_COPY", "count": 2}
    ]


def test_false_blocked_requires_an_explicit_incident_not_a_recovery_guess(tmp_path):
    conn = jobs_db.connect(tmp_path / "jobs.db")
    job_id = jobs_db.create_job(conn, name="case", goal="case", requested_lane="claude")
    jobs_db.append_event(
        conn,
        job_id,
        "reliability_incident",
        data={"classification": "FALSE_BLOCKED", "signature": "BAD_AUTH_CLASSIFIER"},
    )

    report = jobs_scorecard.scorecard_from_connection(
        conn, period_start=0, period_end=9_999_999_999
    )

    assert report.false_blocked == 1
    conn.close()
