"""Provider-free proof of the complete Jobs reliability sequence."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_adapter_claude as adapter
from hermes_cli import jobs_assurance as assurance
from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_dispatch as dispatch
from hermes_cli import jobs_executors
from hermes_cli import jobs_graph as graph
from hermes_cli import jobs_harness as harness
from hermes_cli import jobs_lanes
from hermes_cli import jobs_receipts as receipts
from hermes_cli import jobs_run
from hermes_cli import jobs_scorecard


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import jobs_evidence_gate as evidence_gate  # noqa: E402


def _digest(value: bytes) -> str:
    return receipts.digest_bytes(value)


class ReliabilityRig:
    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path
        self.home = tmp_path / "hermes-home"
        self.home.mkdir()
        monkeypatch.setenv("HOME", str(self.home))
        monkeypatch.setenv("HERMES_HOME", str(self.home))
        self.conn = jdb.connect(tmp_path / "jobs.db")

        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(self.repo)],
            check=True,
            capture_output=True,
        )
        for key, value in (
            ("user.email", "reliability@example.invalid"),
            ("user.name", "Reliability Test"),
            ("commit.gpgsign", "false"),
        ):
            subprocess.run(
                ["git", "-C", str(self.repo), "config", key, value],
                check=True,
            )
        (self.repo / "README.md").write_text("base\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "README.md"], check=True
        )
        subprocess.run(
            ["git", "-C", str(self.repo), "commit", "-qm", "base"],
            check=True,
        )
        self.base = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        self.output = tmp_path / "output"
        self.workspace = tmp_path / "workspaces"
        self.gate_dir = tmp_path / "gate"
        self.output.mkdir()
        self.workspace.mkdir()
        self.gate_dir.mkdir()
        self.memory = tmp_path / "memory.md"
        self.memory.write_text("scoped", encoding="utf-8")
        self.job_id = jdb.create_job(
            self.conn,
            name="reliability e2e",
            goal="prove the sequence",
            requested_lane="claude",
            assurance_contract={
                "critical_user_journey": "Open the built artifact and confirm its content",
                "success_metric": "Artifact contains the independently expected value",
                "outcome_mode": "integration",
                "verification_steps": ["Read artifact.txt from the candidate worktree"],
                "max_attempts": 3,
                "wall_clock_budget_seconds": 3600,
                "risk_domains": ["none"],
                "consumers": [],
                "egress_paths": [],
                "rollback_behavior": "not applicable",
                "knowledge_closure_required": False,
            },
        )

        def passed(_probe_input):
            return harness.ProbeResult("PASS", "OK", {})

        self.probes = harness.PreflightProbes(
            **{name: passed for name in harness.CHECK_NAMES}
        )
        private_key = Ed25519PrivateKey.generate()
        self.signer = harness.ReceiptSigner("lane:e2e:v1", private_key)
        self.verifier = graph.ReceiptVerifier(
            trusted_keys={self.signer.key_id: private_key.public_key()}
        )
        self.spec = dispatch.DispatchSpec(
            repository=self.repo,
            base_commit=self.base,
            branch="jobs/reliability-e2e",
            output_parents=(self.output,),
            scoped_memory_paths=(self.memory,),
            requested_lane="claude",
            lane_id="lane:e2e",
            executor="claude",
            specialist="claude-builder",
            model="claude-opus-5",
            worktree_parent=self.workspace,
        )
        self.clock = 100

    def close(self):
        self.conn.close()

    def run(
        self,
        executor,
        *,
        probes=None,
        gate_callback=None,
        activation_callback=None,
        outcome_callback=None,
        closure_callback=None,
    ):
        self.clock += 100
        lane_kwargs = {}
        if hasattr(self, "lane_decision"):
            lane_kwargs["lane_decision"] = self.lane_decision
        return dispatch.dispatch_once(
            conn=self.conn,
            job_id=self.job_id,
            spec=self.spec,
            probes=self.probes if probes is None else probes,
            signer=self.signer,
            verifier=self.verifier,
            worker_id="worker:e2e",
            executor_registry=jobs_executors.registry.with_reliability_adapters(
                {"claude": executor}
            ),
            gate=(
                (lambda _context, _execution: None)
                if gate_callback is None
                else gate_callback
            ),
            activation_gate=(
                (lambda _context, _gate: None)
                if activation_callback is None
                else activation_callback
            ),
            outcome_gate=(
                (lambda _context, _execution: None)
                if outcome_callback is None
                else outcome_callback
            ),
            closure_gate=(
                (lambda _context, _execution: None)
                if closure_callback is None
                else closure_callback
            ),
            observed_at="2026-08-09T00:00:00Z",
            now=self.clock,
            lease_seconds=60,
            failure_journal_path=self.home / "logs" / "failure.jsonl",
            **lane_kwargs,
        )


@pytest.fixture
def reliability_rig(tmp_path, monkeypatch):
    rig = ReliabilityRig(tmp_path, monkeypatch)
    try:
        yield rig
    finally:
        rig.close()


def _passing_executor(rig):
    def execute(context):
        provider = context.executor.capitalize()
        subprocess.run(
            [
                "git",
                "-C",
                str(rig.repo),
                "worktree",
                "add",
                "-q",
                "-b",
                context.branch,
                str(context.worktree),
                context.base_commit,
            ],
            check=True,
        )
        artifact = context.worktree / "artifact.txt"
        artifact.write_text("verified output\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(context.worktree), "add", "artifact.txt"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(context.worktree), "commit", "-qm", "work"],
            check=True,
        )
        candidate = subprocess.run(
            ["git", "-C", str(context.worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        tests_path = rig.gate_dir / "tests.json"
        tests_path.write_text(
            json.dumps(
                {
                    "tests": [
                        {
                            "cmd": "fake focused suite",
                            "result": "pass",
                            "evidence": "1 passed",
                        }
                    ],
                    "skipped": [],
                    "verdict_hint": "clean",
                }
            ),
            encoding="utf-8",
        )
        evidence_gate.stamp_receipt(
            tests_path,
            kind="tests",
            job="reliability-e2e",
            attempt=0,
            commit=candidate,
            base=context.base_commit,
        )
        review_path = rig.gate_dir / "review-0.json"
        review_path.write_text(
            json.dumps(
                {
                    "verdict": "PASS",
                    "findings": [],
                    "checks_run": ["read artifact"],
                    "checks_skipped": [],
                }
            ),
            encoding="utf-8",
        )
        evidence_gate.stamp_receipt(
            review_path,
            kind="review",
            job="reliability-e2e",
            attempt=0,
            commit=candidate,
            base=context.base_commit,
        )
        output_claim = adapter.claim_artifact(artifact, name="output")
        tests_claim = adapter.claim_artifact(tests_path, name="tests")
        return adapter.ReliabilityExecution(
            status="succeeded",
            commit=candidate,
            worktree=context.worktree,
            artifacts=(output_claim, tests_claim),
            executor_exit_digest=_digest(b"exit:0"),
            output_capture_digest=output_claim.digest,
            builder_handoff={
                "speaker_role": f"{provider} Builder",
                "speaker_executor": context.executor,
                "summary": "I built the candidate.",
                "next_action": "Verify the recorded artifacts.",
                "next_owner_role": f"{provider} Tester",
            },
            tester_handoff={
                "speaker_role": f"{provider} Tester",
                "speaker_executor": context.executor,
                "summary": "I ran the recorded checks.",
                "next_action": "Review the observed candidate.",
                "next_owner_role": "Independent Reviewer",
            },
            reviewer_handoff={
                "speaker_role": "Independent Reviewer",
                "speaker_executor": context.executor,
                "summary": "I independently reviewed the candidate.",
                "next_action": "Authorize completion.",
                "next_owner_role": "Hermes",
                "verdict": "PASS",
                "issues": [],
            },
        )

    return execute


def _authoritative_gate(rig):
    def settle(context, execution):
        evidence_gate.settle(
            rig.gate_dir,
            job="reliability-e2e",
            branch=context.branch,
            base=context.base_commit,
            commit=execution.commit,
            attempt=0,
            rounds=1,
            build_exit=0,
            effort="max",
        )
        normalized = evidence_gate.graph_settlement(
            rig.gate_dir,
            job="reliability-e2e",
            base=context.base_commit,
        )
        return jobs_run.adapt_gate_settlement(
            normalized, job_dir=rig.gate_dir, review_attempt=0
        )

    return settle


def _passing_activation(_context, _gate):
    return dispatch.ActivationDecision(
        status="PASS",
        reason_code="OK",
        activation_gate_digest=_digest(b"fake activation policy passed"),
        completion_receipt_digest=_digest(b"fake completion receipt"),
        completion_handoff={
            "summary": "Hermes authorized the observed candidate.",
            "next_action": "Keep rollback available.",
            "observed_candidate": _gate.commit,
            "environment": "lane:e2e",
            "gate_digest": _digest(b"fake activation policy passed"),
            "rollback": "bounded rollback available",
        },
    )


def _passing_outcome(context, _execution):
    contract = assurance.AssuranceContract.from_mapping(context.assurance_contract)
    evidence = assurance.OutcomeEvidence.from_mapping(
        {
            "critical_user_journey": contract.critical_user_journey,
            "verdict": "PASS",
            "environment": "isolated integration worktree",
            "checks_run": list(contract.verification_steps),
            "observed_behavior": "artifact.txt contained verified output",
            "artifact_digests": [_digest(b"verified output\n")],
            "observed_at": 1786233600,
        }
    )
    return assurance.verify_outcome(contract, evidence)


def _not_applicable_closure(_context, _execution):
    return assurance.KnowledgeClosureReceipt.from_mapping(
        {
            "disposition": "NOT_APPLICABLE",
            "canonical_note_path": None,
            "retrieval_confirmed": False,
            "steward_receipt_digest": _digest(b"no durable knowledge change"),
        }
    )


def test_provider_free_happy_path_reaches_exact_completed_graph(reliability_rig):
    result = reliability_rig.run(
        _passing_executor(reliability_rig),
        gate_callback=_authoritative_gate(reliability_rig),
        activation_callback=_passing_activation,
        outcome_callback=_passing_outcome,
        closure_callback=_not_applicable_closure,
    )

    transitions = jdb.list_transitions(
        reliability_rig.conn, reliability_rig.job_id
    )
    assert [row["target_state"] for row in transitions] == [
        "QUEUED",
        "ASSIGNED",
        "BUILDING",
        "EVIDENCE_COLLECTING",
        "REVIEWING",
        "VERIFIED",
        "OUTCOME_VERIFYING",
        "OUTCOME_VERIFIED",
        "COMPLETED",
    ]
    projection = graph.compute_work_control(reliability_rig.conn, now=999)
    assert projection["jobs"][0]["state"] == "COMPLETED"
    # 2026-08-14: a COMPLETED graph edge no longer implies a shipped Job. The
    # public state parks for the operator; only `hermes jobs activate` — which
    # proves merge, deployment and migration — reaches finished/complete.
    assert projection["jobs"][0]["status"] == "needs_you"
    assert "claim_token" not in json.dumps(projection)
    assert result.action_outcome == "succeeded"
    attempt = jdb.get_attempt(reliability_rig.conn, result.attempt_id)
    assert (attempt["status"], attempt["commit"]) == (
        "succeeded",
        result.commit,
    )
    assert jdb.get_job(reliability_rig.conn, reliability_rig.job_id).claimed_by is None
    report = jobs_scorecard.scorecard_from_connection(
        reliability_rig.conn, period_start=0, period_end=999
    )
    assert (
        report.jobs_started,
        report.jobs_settled,
        report.outcomes_verified,
        report.false_complete,
    ) == (1, 1, 1, 0)


def test_nonpassing_reviewer_handoff_never_reaches_verified(reliability_rig):
    passing = _passing_executor(reliability_rig)

    def executor(context):
        execution = passing(context)
        return replace(
            execution,
            reviewer_handoff={
                "speaker_role": "Independent Reviewer",
                "speaker_executor": "claude",
                "summary": "I found a concrete defect.",
                "next_action": "Correct the defect and rerun review.",
                "next_owner_role": "Claude Builder",
                "verdict": "NEEDS_CHANGES",
                "issues": [
                    {
                        "requirement": "Candidate correctness",
                        "finding": "The candidate has a blocking defect.",
                        "required_fix": "Correct the defect.",
                    }
                ],
            },
        )

    result = reliability_rig.run(
        executor,
        gate_callback=_authoritative_gate(reliability_rig),
        activation_callback=_passing_activation,
    )
    states = [
        row["target_state"]
        for row in jdb.list_transitions(reliability_rig.conn, reliability_rig.job_id)
    ]

    assert result.state == "FAILED"
    assert result.reason == "REVIEW_NEEDS_CHANGES"
    assert "VERIFIED" not in states
    assert "COMPLETED" not in states


def test_activation_callback_exception_settles_failed_without_stuck_verified(
    reliability_rig,
):
    def raises(_context, _gate):
        raise RuntimeError("activation callback exploded")

    result = reliability_rig.run(
        _passing_executor(reliability_rig),
        gate_callback=_authoritative_gate(reliability_rig),
        activation_callback=raises,
    )
    states = [
        row["target_state"]
        for row in jdb.list_transitions(reliability_rig.conn, reliability_rig.job_id)
    ]

    assert result.state == "FAILED"
    assert result.journal_error == "RuntimeError"
    assert states[-1] == "FAILED"
    assert "VERIFIED" not in states
    assert "COMPLETED" not in states
    assert jdb.get_job(reliability_rig.conn, reliability_rig.job_id).claimed_by is None


def test_malformed_activation_receipts_fail_before_verified(reliability_rig):
    def malformed(context, gate):
        return replace(
            _passing_activation(context, gate),
            activation_gate_digest="not-a-digest",
        )

    result = reliability_rig.run(
        _passing_executor(reliability_rig),
        gate_callback=_authoritative_gate(reliability_rig),
        activation_callback=malformed,
    )
    states = [
        row["target_state"]
        for row in jdb.list_transitions(reliability_rig.conn, reliability_rig.job_id)
    ]

    assert result.state == "FAILED"
    assert result.reason == "ACTIVATION_HANDOFF_INCOMPLETE"
    assert "VERIFIED" not in states
    assert "COMPLETED" not in states


def test_lane_aware_happy_path_cleans_registered_worktree_before_idle(
    reliability_rig,
):
    registry = jobs_lanes.load_lane_registry()
    lane = next(item for item in registry.lanes if item.id == "claude-mac-1")
    lane_root = reliability_rig.root / "lanes" / lane.id
    for name in ("auth", "worktrees", "handoffs", "receipts", "health"):
        (lane_root / name).mkdir(parents=True, exist_ok=True)
    auth_sentinel = lane_root / "auth" / "operator-auth-state"
    auth_sentinel.write_text("do not remove", encoding="utf-8")
    # Mac execution is only legal after durable evidence that every matching
    # PC seat failed at the infrastructure layer.  Missing PC health is not a
    # safe failover signal.
    for slot in range(1, 4):
        jdb.record_lane_health(
            reliability_rig.conn,
            jobs_lanes.LaneHealth(
                lane_id=f"claude-pc-{slot}",
                state="BLOCKED",
                status="BLOCKED",
                failure_class="INFRA_FAILURE",
                reason_code="HOST_UNREACHABLE",
                observed_at=reliability_rig.clock,
                expires_at=reliability_rig.clock + 1_000,
                executor_version="test-1.0",
                available_capacity=0,
                safe_detail={},
            ),
        )
    jdb.record_lane_health(
        reliability_rig.conn,
        jobs_lanes.LaneHealth(
            lane_id=lane.id,
            state="IDLE",
            status="PASS",
            failure_class=None,
            reason_code="OK",
            observed_at=reliability_rig.clock,
            expires_at=reliability_rig.clock + 1_000,
            executor_version="test-1.0",
            available_capacity=1,
            safe_detail={},
        ),
    )
    health, active_load = jdb.lane_routing_snapshot(
        reliability_rig.conn,
        registry=registry,
        now=reliability_rig.clock,
    )
    reliability_rig.lane_decision = jobs_lanes.select_lane(
        registry,
        health,
        jobs_lanes.RoutingRequest(
            job_id=reliability_rig.job_id,
            executor="claude",
            model="claude-opus-5",
        ),
        active_load=active_load,
        now=reliability_rig.clock,
    )
    reliability_rig.spec = replace(
        reliability_rig.spec,
        lane_id=lane.id,
        executor="claude",
        model="claude-opus-5",
        worktree_parent=lane_root / "worktrees",
        lane_root=lane_root,
    )

    result = reliability_rig.run(
        _passing_executor(reliability_rig),
        gate_callback=_authoritative_gate(reliability_rig),
        activation_callback=_passing_activation,
        outcome_callback=_passing_outcome,
        closure_callback=_not_applicable_closure,
    )
    placement = jdb.latest_lane_placement_for_attempt(
        reliability_rig.conn, result.attempt_id
    )

    assert result.state == "COMPLETED"
    assert (placement["state"], placement["state_reason_code"]) == (
        "IDLE",
        "CLEANUP_VERIFIED",
    )
    assert not Path(placement["registered_worktree"]).exists()
    assert auth_sentinel.read_text(encoding="utf-8") == "do not remove"
    assert jdb.active_lane_load(reliability_rig.conn) == {}


def test_unable_to_verify_outcome_blocks_completion(reliability_rig):
    def unable(context, _execution):
        contract = assurance.AssuranceContract.from_mapping(context.assurance_contract)
        evidence = assurance.OutcomeEvidence.from_mapping(
            {
                "critical_user_journey": contract.critical_user_journey,
                "verdict": "UNABLE_TO_VERIFY",
                "environment": "staging",
                "checks_run": list(contract.verification_steps),
                "observed_behavior": "Staging was unreachable",
                "artifact_digests": [_digest(b"staging unavailable")],
                "observed_at": 1786233600,
            }
        )
        return assurance.verify_outcome(contract, evidence)

    result = reliability_rig.run(
        _passing_executor(reliability_rig),
        gate_callback=_authoritative_gate(reliability_rig),
        outcome_callback=unable,
        closure_callback=_not_applicable_closure,
        activation_callback=_passing_activation,
    )

    states = [
        row["target_state"]
        for row in jdb.list_transitions(reliability_rig.conn, reliability_rig.job_id)
    ]
    assert result.state == "BLOCKED"
    assert states[-2:] == ["OUTCOME_VERIFYING", "BLOCKED"]
    assert "COMPLETED" not in states


def test_ssh_preflight_block_has_no_claim_or_attempt(reliability_rig):
    calls = []
    blocked = reliability_rig.probes.with_failure(
        "auth_check", code="SSH_AUTH_FAILED"
    )

    result = reliability_rig.run(
        lambda _context: calls.append("executor"), probes=blocked
    )

    assert (result.claimed, result.reason) == (False, "SSH_AUTH_FAILED")
    assert jdb.get_attempts(reliability_rig.conn, reliability_rig.job_id) == []
    assert jdb.get_job(reliability_rig.conn, reliability_rig.job_id).claimed_by is None
    assert calls == []


def test_readback_mismatch_blocks_before_reviewing(reliability_rig):
    def tampering_executor(context):
        context.worktree.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(context.worktree)], check=True)
        subprocess.run(["git", "-C", str(context.worktree), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(context.worktree), "config", "user.name", "Test"], check=True)
        (context.worktree / "README").write_text("base", encoding="utf-8")
        subprocess.run(["git", "-C", str(context.worktree), "add", "README"], check=True)
        subprocess.run(["git", "-C", str(context.worktree), "commit", "-qm", "base"], check=True)
        tests_path = context.worktree / "tests.json"
        tests_path.write_bytes(b"claimed")
        claim = adapter.claim_artifact(tests_path, name="tests")
        tests_path.write_bytes(b"changed")
        return adapter.ReliabilityExecution(
            status="succeeded",
            commit=subprocess.run(["git", "-C", str(context.worktree), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
            worktree=context.worktree,
            artifacts=(claim,),
            executor_exit_digest=_digest(b"exit:0"),
            output_capture_digest=_digest(b"capture"),
            builder_handoff={
                "speaker_role": "Claude Builder",
                "speaker_executor": "claude",
                "summary": "I built the candidate.",
                "next_action": "Verify the recorded artifacts.",
                "next_owner_role": "Claude Tester",
            },
        )

    result = reliability_rig.run(tampering_executor)
    states = [
        row["target_state"]
        for row in jdb.list_transitions(
            reliability_rig.conn, reliability_rig.job_id
        )
    ]

    assert result.reason == "READBACK_DIGEST_MISMATCH"
    assert states[-1] == "BLOCKED"
    assert "REVIEWING" not in states


def test_forged_done_never_reaches_verified(reliability_rig):
    def executor(context):
        context.worktree.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(context.worktree)], check=True)
        subprocess.run(["git", "-C", str(context.worktree), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(context.worktree), "config", "user.name", "Test"], check=True)
        (context.worktree / "README").write_text("base", encoding="utf-8")
        subprocess.run(["git", "-C", str(context.worktree), "add", "README"], check=True)
        subprocess.run(["git", "-C", str(context.worktree), "commit", "-qm", "base"], check=True)
        tests_path = context.worktree / "tests.json"
        tests_path.write_bytes(b"tests passed")
        (reliability_rig.gate_dir / "done.json").write_text(
            json.dumps(
                {
                    "outcome": "succeeded",
                    "claude_exit": 0,
                    "commit": "c" * 40,
                    "branch": "jobs/reliability-e2e",
                    "review_verdict": "PASS",
                    "review_rounds": 1,
                    "findings": 0,
                    "effort": "max",
                }
            ),
            encoding="utf-8",
        )
        return adapter.ReliabilityExecution(
            status="succeeded",
            commit=subprocess.run(["git", "-C", str(context.worktree), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
            worktree=context.worktree,
            artifacts=(adapter.claim_artifact(tests_path, name="tests"),),
            executor_exit_digest=_digest(b"exit:0"),
            output_capture_digest=_digest(b"capture"),
            builder_handoff={
                "speaker_role": "Claude Builder",
                "speaker_executor": "claude",
                "summary": "I built the candidate.",
                "next_action": "Verify the recorded artifacts.",
                "next_owner_role": "Claude Tester",
            },
            tester_handoff={
                "speaker_role": "Claude Tester",
                "speaker_executor": "claude",
                "summary": "I ran the recorded checks.",
                "next_action": "Review the observed candidate.",
                "next_owner_role": "Independent Reviewer",
            },
            reviewer_handoff={
                "speaker_role": "Independent Reviewer",
                "speaker_executor": "claude",
                "summary": "I reviewed the candidate.",
                "next_action": "Authorize completion.",
                "next_owner_role": "Hermes",
                "verdict": "PASS",
                "issues": [],
            },
        )

    def forged_gate(context, _execution):
        normalized = evidence_gate.graph_settlement(
            reliability_rig.gate_dir,
            job="reliability-e2e",
            base=context.base_commit,
        )
        return jobs_run.adapt_gate_settlement(
            normalized, job_dir=reliability_rig.gate_dir, review_attempt=0
        )

    result = reliability_rig.run(executor, gate_callback=forged_gate)
    states = [
        row["target_state"]
        for row in jdb.list_transitions(
            reliability_rig.conn, reliability_rig.job_id
        )
    ]

    assert result.action_outcome == "failed"
    assert states[-1] == "REVIEWING"
    assert "VERIFIED" not in states


def _provider_failure(evidence: bytes, *, proves_progress: bool = True):
    digest = _digest(evidence)

    def execute(context):
        return adapter.ReliabilityExecution(
            status="failed",
            commit=None,
            worktree=context.worktree,
            artifacts=(),
            executor_exit_digest=digest,
            output_capture_digest=digest,
            failure_reason_code="PROVIDER_OUTAGE",
            response_change_digest=(
                _digest(b"changed response:" + evidence) if proves_progress else None
            ),
            result_delta_digest=(
                _digest(b"new measured result:" + evidence) if proves_progress else None
            ),
        )

    return execute


def test_identical_provider_evidence_is_not_retried(reliability_rig):
    first = reliability_rig.run(_provider_failure(b"same provider evidence"))
    second = reliability_rig.run(_provider_failure(b"same provider evidence"))

    assert (first.state, first.retry_action) == ("FAILED", "RETRY")
    assert second.state == "BLOCKED"
    assert second.reason == "RETRY_REJECTED_NO_NEW_EVIDENCE"


def test_provider_retry_without_progress_proof_settles_blocked(reliability_rig):
    result = reliability_rig.run(
        _provider_failure(b"unexplained retry", proves_progress=False)
    )

    assert (result.state, result.retry_action, result.reason) == (
        "BLOCKED",
        "BLOCKED",
        "RETRY_MISSING_RESPONSE_CHANGE",
    )
    report = jobs_scorecard.scorecard_from_connection(
        reliability_rig.conn, period_start=0, period_end=999
    )
    assert report.retry_without_progress == 1


def test_contract_attempt_ceiling_blocks_the_third_failed_execution(reliability_rig):
    results = [
        reliability_rig.run(_provider_failure(f"evidence-{index}".encode()))
        for index in range(1, 4)
    ]

    assert [(item.state, item.retry_action) for item in results[:2]] == [
        ("FAILED", "RETRY"),
        ("FAILED", "RETRY"),
    ]
    assert results[2].state == "BLOCKED"
    assert results[2].reason == "BUDGET_EXHAUSTED"
