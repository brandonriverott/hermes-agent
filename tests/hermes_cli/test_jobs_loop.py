import json
import os

import pytest

from hermes_cli import jobs_loop as loop


def _progress(seed: str):
    return {
        "prior_failure_digest": "sha256:" + "a" * 64,
        "response_change_digest": "sha256:" + ("b" if seed == "1" else "c") * 64,
        "result_delta_digest": "sha256:" + ("d" if seed == "1" else "e") * 64,
    }


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        (loop.FailureSignal(http_status=401), "AUTH_INFRA"),
        (loop.FailureSignal(reason_code="TOKEN_EXPIRED"), "AUTH_INFRA"),
        (loop.FailureSignal(reason_code="SSH_AUTH_FAILED"), "AUTH_INFRA"),
        (loop.FailureSignal(http_status=429), "PROVIDER"),
        (loop.FailureSignal(reason_code="BILLING_REFUSED"), "PROVIDER"),
        (
            loop.FailureSignal(
                reason_code="APPROVAL_REQUIRED", safety_gate=True
            ),
            "SAFETY_GATE",
        ),
        (loop.FailureSignal(reason_code="DISK_FULL"), "INFRA_FAILURE"),
        (
            loop.FailureSignal(reason_code="TESTS_FAILED", stage="testing"),
            "TASK_FAILURE",
        ),
    ],
)
def test_failure_signatures_map_to_one_primary_class(signal, expected):
    assert loop.classify_failure(signal).failure_class == expected


def test_authentication_has_precedence_over_safety_and_task_signals():
    decision = loop.classify_failure(
        loop.FailureSignal(
            http_status=401,
            reason_code="TESTS_FAILED",
            safety_gate=True,
            stage="testing",
        )
    )
    assert (decision.failure_class, decision.recovery_decision) == (
        "AUTH_INFRA",
        "HUMAN_ACTION",
    )


def test_identical_evidence_is_not_retried():
    history = [
        loop.RetryRecord(
            ordinal=1, evidence_digest="sha256:same", decision="RETRY"
        )
    ]
    decision = loop.decide_retry(
        history,
        evidence_digest="sha256:same",
        failure_class="PROVIDER",
        **_progress("1"),
    )
    assert (decision.action, decision.reason_code) == (
        "BLOCKED",
        "RETRY_REJECTED_NO_NEW_EVIDENCE",
    )


def test_fourth_execution_is_blocked():
    history = [
        loop.RetryRecord(
            ordinal=i, evidence_digest=f"sha256:{i}", decision="RETRY"
        )
        for i in (1, 2, 3)
    ]
    decision = loop.decide_retry(
        history,
        evidence_digest="sha256:4",
        failure_class="PROVIDER",
        **_progress("1"),
    )
    assert (decision.action, decision.reason_code) == ("BLOCKED", "RETRY_LIMIT")


def test_retry_backoff_is_bounded_and_task_correction_is_immediate():
    first = loop.decide_retry(
        [], evidence_digest="sha256:1", failure_class="PROVIDER", **_progress("1")
    )
    second = loop.decide_retry(
        [loop.RetryRecord(1, "sha256:1", "RETRY")],
        evidence_digest="sha256:2",
        failure_class="INFRA_FAILURE",
        **_progress("2"),
    )
    task = loop.decide_retry(
        [], evidence_digest="sha256:task", failure_class="TASK_FAILURE", **_progress("1")
    )
    assert (first.action, first.backoff_seconds) == ("RETRY", 30)
    assert (second.action, second.backoff_seconds) == ("RETRY", 60)
    assert (task.action, task.backoff_seconds) == ("RETRY", 0)


@pytest.mark.parametrize("failure_class", ["AUTH_INFRA", "SAFETY_GATE"])
def test_authentication_and_safety_never_retry(failure_class):
    decision = loop.decide_retry(
        [], evidence_digest="sha256:new", failure_class=failure_class
    )
    assert (decision.action, decision.backoff_seconds) == ("HUMAN_ACTION", 0)


def test_retry_without_changed_response_and_new_result_is_blocked():
    decision = loop.decide_retry(
        [], evidence_digest="sha256:new", failure_class="PROVIDER"
    )

    assert (decision.action, decision.reason_code) == (
        "BLOCKED",
        "RETRY_MISSING_PRIOR_FAILURE",
    )


def test_readback_mismatch_blocks_verification(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"changed")
    result = loop.verify_readback(
        artifact, expected_digest="sha256:" + "0" * 64
    )
    assert (result.status, result.reason_code) == (
        "BLOCKED",
        "READBACK_DIGEST_MISMATCH",
    )


def test_readback_verifies_digest_and_size(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"verified")
    expected = loop.digest_bytes(b"verified")

    passed = loop.verify_readback(
        artifact, expected_digest=expected, expected_size=len(b"verified")
    )
    wrong_size = loop.verify_readback(
        artifact, expected_digest=expected, expected_size=1
    )

    assert (passed.status, passed.reason_code, passed.size) == (
        "PASS",
        "OK",
        len(b"verified"),
    )
    assert wrong_size.reason_code == "READBACK_SIZE_MISMATCH"


def test_failure_journal_is_sanitized_and_one_line(tmp_path):
    path = tmp_path / "failure-journal.jsonl"
    record = loop.FailureJournalRecord(
        job_id="j_1",
        attempt_id="a_1",
        failure_class="AUTH_INFRA",
        reason_code="TOKEN_EXPIRED",
        recovery_decision="HUMAN_ACTION",
        evidence_digests=("sha256:" + "a" * 64,),
        observed_at=1,
        component_versions={"jobs_loop": "1"},
    )

    digest = loop.append_failure_journal(record, path=path)

    body = path.read_text(encoding="utf-8")
    assert body.count("\n") == 1
    assert "token" not in body.lower()
    assert digest.startswith("sha256:")
    assert json.loads(body)["reason_code"] == "AUTH_EXPIRED"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


def test_failure_journal_defaults_to_profile_home(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    record = loop.FailureJournalRecord(
        job_id="j_1",
        attempt_id="a_1",
        failure_class="TASK_FAILURE",
        reason_code="TESTS_FAILED",
        recovery_decision="CORRECT",
        evidence_digests=(),
        observed_at=1,
        component_versions={"jobs_loop": "1"},
    )

    loop.append_failure_journal(record)

    assert (home / "logs" / "failure-journal.jsonl").is_file()


def test_failure_journal_rejects_secret_bearing_values(tmp_path):
    record = loop.FailureJournalRecord(
        job_id="j_1",
        attempt_id="a_1",
        failure_class="PROVIDER",
        reason_code="PROVIDER_OUTAGE",
        recovery_decision="RETRY",
        evidence_digests=(),
        observed_at=1,
        component_versions={"provider": "Bearer secret"},
    )
    with pytest.raises(loop.UnsafeJournalRecord):
        loop.append_failure_journal(record, path=tmp_path / "journal.jsonl")
