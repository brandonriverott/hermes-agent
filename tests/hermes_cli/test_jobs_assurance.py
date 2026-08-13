import pytest
from pathlib import Path

from hermes_cli import jobs_assurance as assurance
from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_harness as harness


def _contract(**overrides):
    value = {
        "critical_user_journey": "Customer changes plan and sees the new price",
        "success_metric": "Displayed and charged totals match the approved quote",
        "outcome_mode": "runtime",
        "verification_steps": [
            "Create a test customer",
            "Change the plan",
            "Read back the invoice and account state",
        ],
        "max_attempts": 3,
        "wall_clock_budget_seconds": 3600,
        "risk_domains": ["money", "permissions", "state"],
        "consumers": ["checkout", "billing", "account history"],
        "egress_paths": ["invoice API", "webhook", "customer UI"],
        "rollback_behavior": "Restore the prior plan and void the new invoice",
        "knowledge_closure_required": True,
    }
    value.update(overrides)
    return value


def _snapshot():
    return harness.PreflightSnapshot(
        job_id="j_contract",
        execution_spec_digest="sha256:" + "a" * 64,
        repository=Path("/repo"),
        base_commit="b" * 40,
        branch="jobs/contract",
        output_parents=(Path("/repo/out"),),
        scoped_memory_paths=(Path("/repo/context.md"),),
        lane_id="claude-mac-1",
        executor="claude",
        model="claude-opus-4-1",
        observed_at="2026-08-11T00:00:00Z",
        expected_job_revision=1,
    )


def _probes():
    def passed(_value):
        return harness.ProbeResult("PASS", "OK", {})

    return harness.PreflightProbes(
        file_paths=passed,
        permissions=passed,
        memory_store=passed,
        worktree=passed,
        tool_routing=passed,
        auth_check=passed,
        resource_budget=passed,
    )


def test_contract_names_user_outcome_budget_and_risk_surface():
    contract = assurance.AssuranceContract.from_mapping(_contract())

    assert contract.critical_user_journey.startswith("Customer changes plan")
    assert contract.max_attempts == 3
    assert contract.risk_domains == ("money", "permissions", "state")
    assert contract.to_mapping() == _contract()
    assert contract.digest.startswith("sha256:")


@pytest.mark.parametrize(
    "field,value",
    [
        ("critical_user_journey", ""),
        ("success_metric", ""),
        ("verification_steps", []),
        ("max_attempts", 0),
        ("wall_clock_budget_seconds", 0),
    ],
)
def test_contract_refuses_missing_definition_of_done(field, value):
    with pytest.raises(assurance.InvalidAssuranceContract, match=field):
        assurance.AssuranceContract.from_mapping(_contract(**{field: value}))


def test_contract_refuses_unknown_fields_instead_of_ignoring_typos():
    with pytest.raises(assurance.InvalidAssuranceContract, match="unknown fields"):
        assurance.AssuranceContract.from_mapping(_contract(max_atempts=3))


@pytest.mark.parametrize("field", ["consumers", "egress_paths", "rollback_behavior"])
def test_high_risk_contract_requires_all_consumers_egress_and_rollback(field):
    empty = "" if field == "rollback_behavior" else []
    with pytest.raises(assurance.InvalidAssuranceContract, match=field):
        assurance.AssuranceContract.from_mapping(_contract(**{field: empty}))


def test_not_applicable_outcome_requires_a_specific_reason():
    with pytest.raises(
        assurance.InvalidAssuranceContract, match="not_applicable_reason"
    ):
        assurance.AssuranceContract.from_mapping(
            _contract(outcome_mode="not_applicable")
        )


def test_retry_requires_prior_failure_response_change_and_result_delta():
    missing_change = assurance.assess_retry_progress(
        prior_failure="sha256:" + "1" * 64,
        response_change=None,
        result_delta="sha256:" + "3" * 64,
    )
    progress = assurance.assess_retry_progress(
        prior_failure="sha256:" + "1" * 64,
        response_change="sha256:" + "2" * 64,
        result_delta="sha256:" + "3" * 64,
    )

    assert (missing_change.action, missing_change.reason_code) == (
        "BLOCKED",
        "RETRY_MISSING_RESPONSE_CHANGE",
    )
    assert (progress.action, progress.reason_code) == ("RETRY", "PROGRESS_PROVED")


@pytest.mark.parametrize("attempts,elapsed", [(3, 1), (1, 3600)])
def test_budget_ceiling_forces_stop_decision(attempts, elapsed):
    contract = assurance.AssuranceContract.from_mapping(_contract())
    decision = assurance.decide_continuation(
        contract, attempts_started=attempts, elapsed_seconds=elapsed
    )
    assert (decision.action, decision.reason_code) == (
        "STOP",
        "BUDGET_EXHAUSTED",
    )


def test_unable_to_verify_is_a_first_class_review_verdict():
    review = assurance.validate_review_result(
        {
            "verdict": "UNABLE_TO_VERIFY",
            "findings": ["Staging credentials are unavailable"],
            "checks_run": ["Read the candidate diff"],
        }
    )
    assert review.verdict == "UNABLE_TO_VERIFY"
    assert review.authorizes_completion is False


def test_outcome_evidence_must_match_contract_and_prove_observed_behavior():
    contract = assurance.AssuranceContract.from_mapping(_contract())
    evidence = assurance.OutcomeEvidence.from_mapping(
        {
            "critical_user_journey": contract.critical_user_journey,
            "verdict": "PASS",
            "environment": "staging",
            "checks_run": list(contract.verification_steps),
            "observed_behavior": "Quote, invoice, and UI all displayed $29.00",
            "artifact_digests": ["sha256:" + "4" * 64],
            "observed_at": 1786446000,
        }
    )
    assert assurance.verify_outcome(contract, evidence).status == "PASS"

    wrong = assurance.OutcomeEvidence.from_mapping(
        {**evidence.to_mapping(), "critical_user_journey": "Different journey"}
    )
    assert assurance.verify_outcome(contract, wrong).reason_code == "JOURNEY_MISMATCH"


def test_knowledge_closure_receipt_requires_canonical_update_and_retrieval():
    with pytest.raises(assurance.InvalidClosureReceipt, match="retrieval_confirmed"):
        assurance.KnowledgeClosureReceipt.from_mapping(
            {
                "disposition": "ACCEPTED",
                "canonical_note_path": "_brain/workflow-reliability.md",
                "retrieval_confirmed": False,
                "steward_receipt_digest": "sha256:" + "5" * 64,
            }
        )

    receipt = assurance.KnowledgeClosureReceipt.from_mapping(
        {
            "disposition": "ACCEPTED",
            "canonical_note_path": "_brain/workflow-reliability.md",
            "retrieval_confirmed": True,
            "steward_receipt_digest": "sha256:" + "5" * 64,
        }
    )
    assert receipt.closed is True


def test_job_persists_contract_and_creation_event_digest(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    contract = _contract()
    job_id = jdb.create_job(
        conn,
        requested_lane="claude",
        name="priced plan",
        goal="implement it",
        assurance_contract=contract,
    )

    job = jdb.get_job(conn, job_id)
    created = jdb.get_events(conn, job_id)[0]

    assert job.assurance_contract == contract
    assert created["data"]["assurance_contract_digest"].startswith("sha256:")
    conn.close()


def test_preflight_blocks_missing_assurance_contract():
    decision = harness.evaluate_preflight(_snapshot(), _probes())
    outcome_check = next(
        check for check in decision.checks if check.name == "outcome_contract"
    )
    assert (outcome_check.status, outcome_check.code) == (
        "BLOCKED",
        "OUTCOME_CONTRACT_MISSING",
    )


def test_preflight_accepts_complete_assurance_contract():
    snapshot = assurance.attach_to_preflight(_snapshot(), _contract())
    decision = harness.evaluate_preflight(snapshot, _probes())
    outcome_check = next(
        check for check in decision.checks if check.name == "outcome_contract"
    )
    assert (outcome_check.status, outcome_check.code) == ("PASS", "OK")
