from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_harness as harness
from hermes_cli import jobs_receipts as receipts


@pytest.fixture
def conn(tmp_path):
    connection = jdb.connect(tmp_path / "jobs.db")
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def passing_snapshot(conn, tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    memory = tmp_path / "memory.md"
    memory.write_text("scoped", encoding="utf-8")
    job_id = jdb.create_job(conn, requested_lane="claude", name="candidate", goal="goal")
    return harness.PreflightSnapshot(
        job_id=job_id,
        execution_spec_digest="sha256:" + "e" * 64,
        repository=repository,
        base_commit="b" * 40,
        branch="candidate",
        output_parents=(output,),
        scoped_memory_paths=(memory,),
        lane_id="lane:test",
        executor="claude",
        model="claude-opus-5",
        observed_at="2026-08-09T00:00:00Z",
        expected_job_revision=jdb.get_job(conn, job_id).revision,
    )


@pytest.fixture
def probes():
    def passed(_probe_input):
        return harness.ProbeResult(status="PASS", code="OK", safe_detail={})

    return harness.PreflightProbes(**{name: passed for name in harness.CHECK_NAMES})


@pytest.fixture
def signer():
    return harness.ReceiptSigner(
        key_id="lane:test:v1", private_key=Ed25519PrivateKey.generate()
    )


@pytest.mark.parametrize("failed_name", harness.CHECK_NAMES)
def test_any_failed_check_blocks_without_claim(
    conn, passing_snapshot, probes, signer, failed_name
):
    failed = probes.with_failure(
        failed_name, code=f"{failed_name.upper()}_FAILED"
    )
    before = jdb.get_job(conn, passing_snapshot.job_id)

    decision = harness.preflight_and_record(conn, passing_snapshot, failed, signer)

    after = jdb.get_job(conn, passing_snapshot.job_id)
    assert decision.status == "BLOCKED"
    assert [check.name for check in decision.checks] == list(harness.CHECK_NAMES)
    assert after.claimed_by is None
    assert after.revision == before.revision


def test_passing_preflight_records_once_without_claiming(
    conn, passing_snapshot, probes, signer
):
    decision = harness.preflight_and_record(conn, passing_snapshot, probes, signer)
    repeated = harness.preflight_and_record(conn, passing_snapshot, probes, signer)

    assert decision == repeated
    assert decision.status == "PASS"
    assert all(check.status == "PASS" for check in decision.checks)
    assert jdb.get_job(conn, passing_snapshot.job_id).claimed_by is None
    assert conn.execute("SELECT COUNT(*) FROM job_preflights").fetchone()[0] == 1
    stored = jdb.get_receipts(conn, passing_snapshot.job_id)
    assert len(stored) == 1
    receipts.verify_receipt(
        stored[0]["data"],
        trusted_keys={signer.key_id: signer.private_key.public_key()},
        expected={
            "job_id": passing_snapshot.job_id,
            "execution_spec_digest": passing_snapshot.execution_spec_digest,
        },
    )


@pytest.mark.parametrize(
    "reason_code", ["SSH_AUTH_FAILED", "AUTH_REQUIRED", "TOKEN_EXPIRED"]
)
def test_authentication_failures_are_never_classified_as_task_failures(
    conn, passing_snapshot, probes, signer, reason_code
):
    decision = harness.preflight_and_record(
        conn,
        passing_snapshot,
        probes.with_failure("auth_check", code=reason_code),
        signer,
    )

    assert decision.failure_class == "AUTH_INFRA"


def test_every_probe_runs_once_in_fixed_order_when_one_raises(passing_snapshot):
    calls = []

    def probe(name):
        def run(_probe_input):
            calls.append(name)
            if name == "permissions":
                raise RuntimeError("secret token=must-not-leak")
            return harness.ProbeResult(status="PASS", code="OK", safe_detail={})

        return run

    probes = harness.PreflightProbes(
        **{name: probe(name) for name in harness.CHECK_NAMES}
    )

    decision = harness.evaluate_preflight(passing_snapshot, probes)

    assert calls == list(harness.CHECK_NAMES)
    failed = next(check for check in decision.checks if check.name == "permissions")
    assert (failed.status, failed.code, failed.safe_detail) == (
        "BLOCKED",
        "PERMISSIONS_PROBE_ERROR",
        {},
    )


def test_safe_detail_rejects_secret_keys_and_oversized_values(
    passing_snapshot, probes
):
    secret = probes.with_result(
        "auth_check",
        harness.ProbeResult(
            status="PASS", code="OK", safe_detail={"token": "never-store"}
        ),
    )
    oversized = probes.with_result(
        "resource_budget",
        harness.ProbeResult(
            status="PASS", code="OK", safe_detail={"reason": "x" * 5000}
        ),
    )

    secret_check = next(
        check
        for check in harness.evaluate_preflight(passing_snapshot, secret).checks
        if check.name == "auth_check"
    )
    oversized_check = next(
        check
        for check in harness.evaluate_preflight(passing_snapshot, oversized).checks
        if check.name == "resource_budget"
    )
    assert (secret_check.status, secret_check.code) == (
        "BLOCKED",
        "UNSAFE_DETAIL",
    )
    assert (oversized_check.status, oversized_check.code) == (
        "BLOCKED",
        "SAFE_DETAIL_TOO_LARGE",
    )


def test_memory_store_rejects_a_directory_scope_even_when_probe_passes(
    passing_snapshot, probes, tmp_path
):
    broad = tmp_path / "vault"
    broad.mkdir()
    snapshot = replace(passing_snapshot, scoped_memory_paths=(broad,))

    decision = harness.evaluate_preflight(snapshot, probes)

    check = next(item for item in decision.checks if item.name == "memory_store")
    assert (check.status, check.code) == ("BLOCKED", "MEMORY_SCOPE_TOO_BROAD")
