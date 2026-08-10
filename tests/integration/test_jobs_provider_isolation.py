"""Provider isolation mirror: a Claude Job never touches the Codex adapter.

Task 9 companion to ``test_jobs_create_codex_to_origin_chat.py``.  Uses the
same temp-store, real-handler, real-dispatcher, real-delivery rig, but the
Job is created with ``lane=claude``: only the Claude reliability adapter may
run, the Codex reliability adapter must never be invoked, and the origin
chat receives the ordered lifecycle updates through the passive worker.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_adapter_claude as adapter
from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_dispatch as dispatch
from hermes_cli import jobs_executors
from hermes_cli import jobs_graph as graph
from hermes_cli import jobs_harness as harness
from hermes_cli import jobs_lanes
from hermes_cli import jobs_notifications as jn
from hermes_cli import jobs_receipts as receipts
from hermes_cli import jobs_run
from hermes_cli import jobs_tool
from hermes_state import SessionDB
from gateway.jobs_notifications import deliver_due_notification_once

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import jobs_evidence_gate as evidence_gate  # noqa: E402


def _digest(value: bytes) -> str:
    return receipts.digest_bytes(value)


JOB_NAME = "claude-origin-e2e"


class ClaudeOriginRig:
    """Same temp-store rig as the Codex flow, with ``lane=claude``."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path
        self.home = tmp_path / "hermes-home"
        self.home.mkdir()
        monkeypatch.setenv("HOME", str(self.home))
        monkeypatch.setenv("HERMES_HOME", str(self.home))
        self.jobs_path = self.home / "jobs.db"
        self.conn = jdb.connect(self.jobs_path)
        self.session_db = SessionDB(db_path=self.home / "state.db")

        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(self.repo)],
            check=True,
            capture_output=True,
        )
        for key, value in (
            ("user.email", "integration@example.invalid"),
            ("user.name", "Integration Test"),
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

        def passed(_probe_input):
            return harness.ProbeResult("PASS", "OK", {})

        self.probes = harness.PreflightProbes(
            **{name: passed for name in harness.CHECK_NAMES}
        )
        private_key = Ed25519PrivateKey.generate()
        self.signer = harness.ReceiptSigner("lane:integration:v1", private_key)
        self.verifier = graph.ReceiptVerifier(
            trusted_keys={self.signer.key_id: private_key.public_key()}
        )
        self.clock = 100
        self.lane_decision = None

    def close(self):
        self.conn.close()

    def create_via_handler(self, monkeypatch, *, origin, lane="claude", **updates):
        monkeypatch.setattr(jobs_tool, "capture_exact_origin", lambda: origin)
        args = {
            "name": "Ship exact-origin updates",
            "goal": "Keep this body exactly.\n\nTRAILING=true\n",
            "repo_path": str(self.repo),
            "lane": lane,
        }
        args.update(updates)
        result = json.loads(jobs_tool.jobs_create_handler(args))
        assert "error" not in result, result
        self.job_id = result["job_id"]
        self.requested_lane = result["lane"]
        return result

    def select_claude_lane(self):
        registry = jobs_lanes.load_lane_registry()
        lane = next(item for item in registry.lanes if item.id == "claude-mac-1")
        lane_root = self.root / "lanes" / lane.id
        for name in ("auth", "worktrees", "handoffs", "receipts", "health"):
            (lane_root / name).mkdir(parents=True, exist_ok=True)
        jdb.record_lane_health(
            self.conn,
            jobs_lanes.LaneHealth(
                lane_id=lane.id,
                state="IDLE",
                status="PASS",
                failure_class=None,
                reason_code="OK",
                observed_at=self.clock,
                expires_at=self.clock + 1_000,
                executor_version="test-1.0",
                available_capacity=1,
                safe_detail={},
            ),
        )
        health, active_load = jdb.lane_routing_snapshot(
            self.conn,
            registry=registry,
            now=self.clock,
        )
        self.lane_decision = jobs_lanes.select_lane(
            registry,
            health,
            jobs_lanes.RoutingRequest(
                job_id=self.job_id,
                executor="claude",
                model="claude-opus-5",
            ),
            active_load=active_load,
            now=self.clock,
        )
        self.spec = dispatch.DispatchSpec(
            repository=self.repo,
            base_commit=self.base,
            branch="jobs/claude-origin-e2e",
            output_parents=(self.output,),
            scoped_memory_paths=(self.memory,),
            requested_lane="claude",
            lane_id=lane.id,
            executor="claude",
            specialist="claude-builder",
            model="claude-opus-5",
            worktree_parent=lane_root / "worktrees",
            lane_root=lane_root,
        )
        return lane

    def run(
        self,
        claude_executor,
        *,
        codex_executor=None,
        gate_callback=None,
        activation_callback=None,
    ):
        self.clock += 100
        lane_kwargs = {}
        if self.lane_decision is not None:
            lane_kwargs["lane_decision"] = self.lane_decision
        return dispatch.dispatch_once(
            conn=self.conn,
            job_id=self.job_id,
            spec=self.spec,
            probes=self.probes,
            signer=self.signer,
            verifier=self.verifier,
            worker_id="worker:integration",
            executor_registry=jobs_executors.registry.with_reliability_adapters(
                {
                    "claude": claude_executor,
                    "codex": codex_executor or _never_codex(),
                }
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
            observed_at="2026-08-10T00:00:00Z",
            now=self.clock,
            lease_seconds=60,
            failure_journal_path=self.home / "logs" / "failure.jsonl",
            **lane_kwargs,
        )

    def deliver_all(self, *, now=None):
        handled = []
        while True:
            result = deliver_due_notification_once(
                self.session_db,
                jobs_path=self.jobs_path,
                owner="worker:integration",
                now=now if now is not None else int(time.time()),
            )
            if result is None:
                break
            handled.append(result)
        return handled

    def milestone_names(self):
        return [row.milestone for row in jn.list_notifications(self.conn)]

    def job_row(self):
        return jdb.get_job(self.conn, self.job_id)


def _never_codex():
    def execute(context):
        raise AssertionError("Codex reliability adapter must never run")

    return execute


def _recording_claude(rig, calls):
    def execute(context):
        assert context.executor == "claude"
        calls.append(("executor", context.executor))
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
            job=JOB_NAME,
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
            job=JOB_NAME,
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
            job=JOB_NAME,
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
            job=JOB_NAME,
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


def _origin(**overrides):
    origin = {
        "platform": "telegram",
        "chat_id": "chat-origin",
        "session_id": "sess-origin",
        "chat_type": "dm",
        "thread_id": "thread-1",
        "user_id": "user-1",
        "profile": "default",
    }
    origin.update(overrides)
    return origin


def _messages(session_db, session_id):
    return session_db.get_messages(session_id)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    item = ClaudeOriginRig(tmp_path, monkeypatch)
    try:
        yield item
    finally:
        item.close()


def test_claude_job_whole_flow_never_touches_codex_and_reaches_origin(
    rig, monkeypatch
):
    """A Claude Job runs only the Claude adapter and delivers to its origin."""
    claude_calls = []
    codex_calls = []

    created = rig.create_via_handler(monkeypatch, origin=_origin())
    assert created["lane"] == "claude"
    assert created["executor"] == "claude"
    assert created["specialist"] == "claude-builder"
    assert created["model"] == "claude-opus-5"

    assert rig.milestone_names() == ["queued"]

    lane = rig.select_claude_lane()
    assert lane.id.startswith("claude-")

    result = rig.run(
        _recording_claude(rig, claude_calls),
        codex_executor=_recording_codex(codex_calls),
        gate_callback=_authoritative_gate(rig),
        activation_callback=_passing_activation,
    )

    assert result.action_outcome == "succeeded"
    assert result.state == "COMPLETED"
    assert claude_calls == [("executor", "claude")]
    assert codex_calls == []  # Claude Job never calls the Codex adapter.

    assert rig.milestone_names() == [
        "queued",
        "assigned",
        "building",
        "testing",
        "review",
        "finished",
    ]

    job = rig.job_row()
    assert (job.executor, job.model) == ("claude", "claude-opus-5")
    placement = jdb.latest_lane_placement_for_attempt(rig.conn, result.attempt_id)
    assert placement["lane_id"].startswith("claude-")

    rig.session_db.create_session("sess-origin", source="telegram")
    rig.session_db.create_session("decoy", source="telegram")
    handled = rig.deliver_all()
    assert len(handled) == 6

    origin_messages = _messages(rig.session_db, "sess-origin")
    assert [m["content"] for m in origin_messages] == [
        "Queued — Ship exact-origin updates (#1) accepted",
        "Assigned — Ship exact-origin updates (#1) started",
        "Building — Ship exact-origin updates (#1) in progress",
        "Testing — Ship exact-origin updates (#1) running tests and evidence",
        "Review — Ship exact-origin updates (#1) under review",
        "Finished — Ship exact-origin updates (#1) complete",
    ]
    assert [m["platform_message_id"] for m in origin_messages] == handled
    assert _messages(rig.session_db, "decoy") == []

    # Restart idempotency: a fresh worker pass appends nothing.
    assert rig.deliver_all(now=int(time.time()) + 1) == []
    assert len(_messages(rig.session_db, "sess-origin")) == 6


def _recording_codex(calls):
    def execute(context):
        calls.append(context.executor)
        raise AssertionError("Codex adapter invoked for a Claude Job")

    return execute
