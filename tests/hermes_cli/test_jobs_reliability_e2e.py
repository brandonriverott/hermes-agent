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
from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_dispatch as dispatch
from hermes_cli import jobs_graph as graph
from hermes_cli import jobs_harness as harness
from hermes_cli import jobs_lanes
from hermes_cli import jobs_receipts as receipts
from hermes_cli import jobs_run


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
            self.conn, name="reliability e2e", goal="prove the sequence"
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
            lane_id="lane:e2e",
            executor="fake",
            model="fake-model",
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
            executor=executor,
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
    )


def test_provider_free_happy_path_reaches_exact_completed_graph(reliability_rig):
    result = reliability_rig.run(
        _passing_executor(reliability_rig),
        gate_callback=_authoritative_gate(reliability_rig),
        activation_callback=_passing_activation,
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
        "COMPLETED",
    ]
    projection = graph.compute_work_control(reliability_rig.conn, now=999)
    assert projection["jobs"][0]["state"] == "COMPLETED"
    assert projection["jobs"][0]["status"] == "finished"
    assert "claim_token" not in json.dumps(projection)
    assert result.action_outcome == "succeeded"
    attempt = jdb.get_attempt(reliability_rig.conn, result.attempt_id)
    assert (attempt["status"], attempt["commit"]) == (
        "succeeded",
        result.commit,
    )
    assert jdb.get_job(reliability_rig.conn, reliability_rig.job_id).claimed_by is None


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
        tests_path = context.worktree / "tests.json"
        tests_path.write_bytes(b"claimed")
        claim = adapter.claim_artifact(tests_path, name="tests")
        tests_path.write_bytes(b"changed")
        return adapter.ReliabilityExecution(
            status="succeeded",
            commit="c" * 40,
            worktree=context.worktree,
            artifacts=(claim,),
            executor_exit_digest=_digest(b"exit:0"),
            output_capture_digest=_digest(b"capture"),
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
            commit="c" * 40,
            worktree=context.worktree,
            artifacts=(adapter.claim_artifact(tests_path, name="tests"),),
            executor_exit_digest=_digest(b"exit:0"),
            output_capture_digest=_digest(b"capture"),
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


def _provider_failure(evidence: bytes):
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
        )

    return execute


def test_identical_provider_evidence_is_not_retried(reliability_rig):
    first = reliability_rig.run(_provider_failure(b"same provider evidence"))
    second = reliability_rig.run(_provider_failure(b"same provider evidence"))

    assert (first.state, first.retry_action) == ("FAILED", "RETRY")
    assert second.state == "BLOCKED"
    assert second.reason == "RETRY_REJECTED_NO_NEW_EVIDENCE"


def test_fourth_execution_is_blocked_by_retry_limit(reliability_rig):
    results = [
        reliability_rig.run(_provider_failure(f"evidence-{index}".encode()))
        for index in range(1, 5)
    ]

    assert [(item.state, item.retry_action) for item in results[:3]] == [
        ("FAILED", "RETRY"),
        ("FAILED", "RETRY"),
        ("FAILED", "RETRY"),
    ]
    assert results[3].state == "BLOCKED"
    assert results[3].reason == "RETRY_LIMIT"
