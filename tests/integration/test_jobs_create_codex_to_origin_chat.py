"""Whole-workflow integration: real jobs_create(lane=codex) -> exact-origin chat.

Task 9 contract under test, using only temp stores and fake health/adapter
boundaries (no real Codex process, no live gateway, cron, Tailscale,
packaging, or activation):

- The real plugin handler creates a canonical Codex Job at an exact Hermes
  origin; the durable queued outbox row is created transactionally.
- The real dispatcher/runtime claims a real lane decision (``codex-*``),
  runs only the Codex reliability adapter, transitions through the real
  test/review graph, and records the lifecycle milestones in durable order.
- The Claude reliability adapter/preflight/runner is never invoked for the
  Codex Job.
- The passive gateway worker delivers exactly the ordered
  queued/assigned/building/testing/review/finished messages to the exact
  origin session; decoy and Build Feed/home/recent/fallback sessions receive
  zero messages.
- ``notification_id`` remains the destination idempotency key across worker
  restarts.
- An unavailable origin stays pending; an unknown platform becomes durably
  blocked.
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
from hermes_cli import jobs_assurance as assurance
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


def _passing_outcome(context, _execution):
    contract = assurance.AssuranceContract.from_mapping(context.assurance_contract)
    return assurance.verify_outcome(
        contract,
        assurance.OutcomeEvidence.from_mapping({
            "critical_user_journey": contract.critical_user_journey,
            "verdict": "PASS",
            "environment": "isolated integration worktree",
            "checks_run": list(contract.verification_steps),
            "observed_behavior": "requested change reached the exact origin",
            "artifact_digests": [_digest(b"verified exact-origin delivery")],
            "observed_at": 1786233600,
        }),
    )


def _not_applicable_closure(_context, _execution):
    return assurance.KnowledgeClosureReceipt.from_mapping({
        "disposition": "NOT_APPLICABLE",
        "canonical_note_path": None,
        "retrieval_confirmed": False,
        "steward_receipt_digest": _digest(b"no durable knowledge change"),
    })


JOB_NAME = "codex-origin-e2e"
ASSURANCE = {
    "critical_user_journey": "requested change reaches the exact origin",
    "success_metric": "verified change and one exact-origin completion",
    "outcome_mode": "integration",
    "verification_steps": ["run the full provider and delivery flow"],
    "max_attempts": 3,
    "wall_clock_budget_seconds": 3600,
    "risk_domains": ["none"],
    "consumers": [],
    "egress_paths": [],
    "rollback_behavior": "discard the test repository",
    "knowledge_closure_required": False,
}


class CodexOriginRig:
    """Temp home, real plugin handler, real Jobs DB, real lane policy, SessionDB.

    Only the external process boundary (the reliability adapters), signing
    keys, and health probes are faked.
    """

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

    # -- real plugin handler -------------------------------------------------
    def create_via_handler(self, monkeypatch, *, origin, lane="codex", **updates):
        """Call the real jobs_create plugin handler with a captured origin."""
        monkeypatch.setattr(jobs_tool, "capture_exact_origin", lambda: origin)
        args = {
            "name": "Ship exact-origin updates",
            "goal": "Keep this body exactly.\n\nTRAILING=true\n",
            "repo_path": str(self.repo),
            "lane": lane,
            "assurance": ASSURANCE,
        }
        args.update(updates)
        result = json.loads(jobs_tool.jobs_create_handler(args))
        assert "error" not in result, result
        self.job_id = result["job_id"]
        self.requested_lane = result["lane"]
        return result

    # -- real lane policy ----------------------------------------------------
    def select_codex_lane(self):
        registry = jobs_lanes.load_lane_registry()
        lane = next(item for item in registry.lanes if item.id == "codex-mac-1")
        lane_root = self.root / "lanes" / lane.id
        for name in ("auth", "worktrees", "handoffs", "receipts", "health"):
            (lane_root / name).mkdir(parents=True, exist_ok=True)
        auth_sentinel = lane_root / "auth" / "operator-auth-state"
        auth_sentinel.write_text("do not remove", encoding="utf-8")
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
                executor="codex",
                model="gpt-5.6-sol",
            ),
            active_load=active_load,
            now=self.clock,
        )
        self.spec = dispatch.DispatchSpec(
            repository=self.repo,
            base_commit=self.base,
            branch="jobs/codex-origin-e2e",
            output_parents=(self.output,),
            scoped_memory_paths=(self.memory,),
            requested_lane="codex",
            lane_id=lane.id,
            executor="codex",
            specialist="codex-builder",
            model="gpt-5.6-sol",
            worktree_parent=lane_root / "worktrees",
            lane_root=lane_root,
        )
        return lane, lane_root

    # -- real dispatcher -----------------------------------------------------
    def run(
        self,
        codex_executor,
        *,
        claude_executor=None,
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
                    "codex": codex_executor,
                    "claude": claude_executor or _never_claude(),
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
            outcome_gate=_passing_outcome,
            closure_gate=_not_applicable_closure,
            observed_at="2026-08-10T00:00:00Z",
            now=self.clock,
            lease_seconds=60,
            failure_journal_path=self.home / "logs" / "failure.jsonl",
            **lane_kwargs,
        )

    # -- real passive delivery worker ----------------------------------------
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


def _never_claude():
    def execute(context):
        raise AssertionError("Claude reliability adapter must never run")

    return execute


def _recording_codex(rig, calls):
    def execute(context):
        assert context.executor == "codex"
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
    item = CodexOriginRig(tmp_path, monkeypatch)
    try:
        yield item
    finally:
        item.close()


def test_codex_job_whole_flow_reaches_exact_origin_chat_only(rig, monkeypatch):
    """Real plugin handler -> real dispatch -> real delivery to exact origin."""
    claude_calls = []
    codex_calls = []

    # 1. Real plugin handler: jobs_create(lane=codex, exact Hermes origin).
    created = rig.create_via_handler(monkeypatch, origin=_origin())
    assert created["lane"] == "codex"
    assert created["executor"] == "codex"
    assert created["specialist"] == "codex-builder"
    assert created["model"] == "gpt-5.6-sol"

    # 2. Queued outbox row exists (created transactionally with the Job).
    assert rig.milestone_names() == ["queued"]
    queued = jn.list_notifications(rig.conn)[0]
    assert queued.job_id == rig.job_id

    # 3. Real lane policy selects a codex-* seat.
    lane, lane_root = rig.select_codex_lane()
    assert lane.id.startswith("codex-")
    assert rig.lane_decision.status == "SELECTED"
    assert rig.lane_decision.lane_id == lane.id

    # 4. Real dispatcher runs the fake/controlled Codex adapter only.
    result = rig.run(
        _recording_codex(rig, codex_calls),
        claude_executor=_recording_claude(claude_calls),
        gate_callback=_authoritative_gate(rig),
        activation_callback=_passing_activation,
    )

    assert result.action_outcome == "succeeded"
    assert result.state == "COMPLETED"
    assert codex_calls == [("executor", "codex")]
    assert claude_calls == []  # Codex job never calls the Claude adapter.

    # 5. Ordered lifecycle milestones from the durable outbox.
    assert rig.milestone_names() == [
        "queued",
        "assigned",
        "building",
        "testing",
        "review",
        "finished",
    ]

    # 6. Durable identity is unchanged through the signed receipt.
    job = rig.job_row()
    assert (job.executor, job.model) == ("codex", "gpt-5.6-sol")
    placement = jdb.latest_lane_placement_for_attempt(rig.conn, result.attempt_id)
    assert placement["lane_id"].startswith("codex-")
    assert (placement["executor"], placement["model"]) == (
        "codex",
        "gpt-5.6-sol",
    )
    receipts_rows = jdb.get_receipts(rig.conn, rig.job_id)
    assert receipts_rows
    assert all(row["job_id"] == rig.job_id for row in receipts_rows)

    # 7. Passive delivery to the exact origin session.
    rig.session_db.create_session("sess-origin", source="telegram")
    rig.session_db.create_session("decoy", source="telegram")
    rig.session_db.create_session("build-feed", source="telegram")
    rig.session_db.create_session("home-channel", source="telegram")
    rig.session_db.create_session("recent-session", source="telegram")

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
    assert all(m["platform_message_id"] for m in origin_messages)
    assert [m["platform_message_id"] for m in origin_messages] == handled

    # 8. No Build Feed / home / recent / decoy / fallback target gets anything.
    for decoy in ("decoy", "build-feed", "home-channel", "recent-session"):
        assert _messages(rig.session_db, decoy) == []


def test_worker_restart_never_duplicates_visible_delivery(rig, monkeypatch):
    """notification_id remains the destination idempotency key."""
    created = rig.create_via_handler(monkeypatch, origin=_origin())
    rig.select_codex_lane()
    result = rig.run(
        _recording_codex(rig, []),
        gate_callback=_authoritative_gate(rig),
        activation_callback=_passing_activation,
    )
    assert result.action_outcome == "succeeded"

    rig.session_db.create_session("sess-origin", source="telegram")
    rig.session_db.create_session("decoy", source="telegram")

    first = rig.deliver_all(now=int(time.time()))
    assert len(first) == 6
    origin_messages = _messages(rig.session_db, "sess-origin")
    assert len(origin_messages) == 6

    # A fresh worker pass (restart) must not re-append anything.
    second = rig.deliver_all(now=int(time.time()) + 1)
    assert second == []
    assert len(_messages(rig.session_db, "sess-origin")) == 6
    assert len(_messages(rig.session_db, "decoy")) == 0


def test_unavailable_origin_stays_pending_and_unknown_platform_blocked(
    rig, monkeypatch
):
    """Unavailable origin: pending; unknown platform: durable blocked."""
    created = rig.create_via_handler(monkeypatch, origin=_origin())
    rig.select_codex_lane()
    result = rig.run(
        _recording_codex(rig, []),
        gate_callback=_authoritative_gate(rig),
        activation_callback=_passing_activation,
    )
    assert result.action_outcome == "succeeded"

    # Origin session is never created: delivery must stay pending.
    ghost = rig.create_via_handler(
        monkeypatch, origin=_origin(session_id="ghost", chat_id="chat-ghost")
    )
    # Unknown platform: must become durably blocked, never retried forever.
    pigeon = rig.create_via_handler(
        monkeypatch, origin=_origin(platform="carrier_pigeon", session_id="pigeon")
    )

    rig.session_db.create_session("sess-origin", source="telegram")
    now = int(time.time())
    rig.deliver_all(now=now)

    rows = {row.notification_id: row for row in jn.list_notifications(rig.conn)}

    # The ghost (unavailable origin) row is pending, not delivered/blocked.
    ghost_rows = [
        row
        for row in rows.values()
        if row.job_id == ghost["job_id"] and row.milestone == "queued"
    ]
    assert len(ghost_rows) == 1
    assert ghost_rows[0].delivered_at is None
    assert ghost_rows[0].blocked_reason is None

    # The unknown platform row is durably blocked with a safe reason.
    pigeon_rows = [
        row
        for row in rows.values()
        if row.job_id == pigeon["job_id"] and row.milestone == "queued"
    ]
    assert len(pigeon_rows) == 1
    assert pigeon_rows[0].blocked_reason == "unknown-platform:carrier_pigeon"
    assert pigeon_rows[0].delivered_at is None

    # Decoy session still received nothing.
    rig.session_db.create_session("decoy", source="telegram")
    assert _messages(rig.session_db, "decoy") == []


def _recording_claude(calls):
    def execute(context):
        calls.append(context.executor)
        raise AssertionError("Claude adapter invoked for a Codex Job")

    return execute
