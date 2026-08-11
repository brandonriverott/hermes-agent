"""Production reliability adapters must be installed and provider-isolated."""

from __future__ import annotations

import json
import subprocess
import base64
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from hermes_cli import jobs_dispatch as dispatch
from hermes_cli import jobs_execution
from hermes_cli import jobs_executors
from hermes_cli import jobs_identity
from hermes_cli import jobs_reliability
from hermes_cli import jobs_receipts
from hermes_cli import jobs_runtime
from hermes_cli import jobs_lanes


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha


def _context(tmp_path: Path, provider: str) -> dispatch.DispatchContext:
    repo, base = _repo(tmp_path)
    identity = jobs_identity.resolve_requested_lane(provider)
    lane_root = tmp_path / f"{provider}-mac-1"
    for child in ("auth", "worktrees", "handoffs", "receipts", "health"):
        (lane_root / child).mkdir(parents=True, exist_ok=True)
    return dispatch.DispatchContext(
        job_id="j_test",
        job_number=1,
        job_name="production adapter",
        goal="make one safe change",
        attempt_id="a_test",
        ordinal=1,
        repository=repo,
        base_commit=base,
        branch="jobs/j_test",
        worktree=lane_root / "worktrees" / "j_test-1",
        lane_root=lane_root,
        requested_lane=identity.requested_lane,
        lane_id=f"{provider}-mac-1",
        executor=identity.executor,
        specialist=identity.specialist,
        model=identity.model,
        effort="high" if provider == "codex" else "max",
        max_turns=120,
    )


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_production_registry_installs_exact_reliability_adapter(provider):
    registry = jobs_executors.production_registry()
    adapter = registry.require_reliability(provider)
    assert callable(adapter)
    assert getattr(adapter, "executor", None) == provider


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_reliability_adapter_preserves_provider_and_materializes_evidence(
    tmp_path, provider
):
    context = _context(tmp_path, provider)
    calls: list[str] = []

    def phase_runner(actual_provider, actual_context):
        calls.append(actual_provider)
        assert actual_provider == provider
        subprocess.run(
            [
                "git", "-C", str(actual_context.repository), "worktree", "add",
                "-q", "-b", actual_context.branch, str(actual_context.worktree),
                actual_context.base_commit,
            ],
            check=True,
        )
        (actual_context.worktree / "result.txt").write_text("done\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(actual_context.worktree), "add", "result.txt"], check=True)
        subprocess.run(["git", "-C", str(actual_context.worktree), "commit", "-qm", "work"], check=True)
        commit = subprocess.run(
            ["git", "-C", str(actual_context.worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return jobs_reliability.PhaseResult(
            status="succeeded",
            commit=commit,
            tests=({"cmd": "focused", "result": "pass", "evidence": "1 passed"},),
            review={"verdict": "PASS", "findings": [], "checks_run": ["diff"]},
            executor_exit_digest="sha256:" + "1" * 64,
            output_capture_digest="sha256:" + "2" * 64,
        )

    adapter = jobs_reliability.ProductionReliabilityAdapter(
        provider, phase_runner=phase_runner
    )
    execution = adapter(context)

    assert calls == [provider]
    assert isinstance(execution, jobs_execution.ReliabilityExecution)
    assert execution.status == "succeeded"
    assert {claim.name for claim in execution.artifacts} == {"output", "tests"}
    evidence_dir = context.lane_root / "receipts" / context.attempt_id
    tests_doc = json.loads((evidence_dir / "tests.json").read_text())
    review_doc = json.loads((evidence_dir / "review-0.json").read_text())
    assert tests_doc["_gate"]["commit"] == execution.commit
    assert review_doc["_gate"]["commit"] == execution.commit
    gate = jobs_reliability.production_gate(context, execution)
    assert gate.action_outcome == "succeeded", gate
    assert gate.identity_verified is True
    completion = jobs_reliability.production_completion_gate(context, gate)
    assert completion.status == "PASS"


def test_provider_mismatch_refuses_before_phase_runner(tmp_path):
    context = _context(tmp_path, "codex")
    calls = []
    adapter = jobs_reliability.ProductionReliabilityAdapter(
        "claude", phase_runner=lambda *args: calls.append(args)
    )
    with pytest.raises(jobs_execution.AdapterError, match="identity"):
        adapter(context)
    assert calls == []


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_local_phase_runner_uses_two_fresh_same_provider_sessions(
    tmp_path, monkeypatch, provider
):
    context = _context(tmp_path, provider)
    calls: list[list[str]] = []
    monkeypatch.setattr(jobs_reliability.shutil, "which", lambda *args, **kwargs: f"/bin/{provider}")

    def process_runner(
        argv,
        *,
        cwd,
        env,
        prompt,
        stdout_path,
        stderr_path,
        timeout,
    ):
        calls.append(list(argv))
        stderr_path.write_text("token=sk-in-test-secret-0123456789\n", encoding="utf-8")
        is_review = len(calls) == 2
        if not is_review:
            (cwd / "built.txt").write_text("candidate\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(cwd), "add", "built.txt"], check=True)
            subprocess.run(["git", "-C", str(cwd), "commit", "-qm", "candidate"], check=True)
            payload = {
                "outcome": "succeeded",
                "tests": [
                    {"cmd": "focused", "result": "pass", "evidence": "1 passed"}
                ],
            }
        else:
            payload = {"verdict": "PASS", "findings": [], "checks_run": ["diff"]}
        target = stdout_path
        if provider == "codex":
            marker = "--output-last-message"
            target = Path(argv[argv.index(marker) + 1])
        target.write_text(
            json.dumps(
                {"structured_output": payload} if provider == "claude" else payload
            ),
            encoding="utf-8",
        )
        stdout_path.touch(exist_ok=True)
        return jobs_reliability.ProcessResult(0)

    result = jobs_reliability.LocalProviderPhaseRunner(
        process_runner=process_runner
    )(provider, context)

    assert result.status == "succeeded"
    assert result.review["verdict"] == "PASS"
    assert len(calls) == 2
    assert all(call[0] == provider for call in calls)
    other = "claude" if provider == "codex" else "codex"
    assert all(other not in call for call in calls)
    if provider == "codex":
        assert "workspace-write" in calls[0]
        assert "read-only" in calls[1]
    else:
        assert "acceptEdits" in calls[0]
        assert "plan" in calls[1]
    handoff = context.lane_root / "handoffs" / context.attempt_id
    assert "sk-in-test-secret" not in (handoff / "build-stderr.log").read_text()


def test_local_phase_runner_refuses_remote_physical_lane_before_provider(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, "codex")
    context = replace(context, lane_id="codex-pc-1")
    calls = []
    with pytest.raises(jobs_execution.AdapterError, match="not local"):
        jobs_reliability.LocalProviderPhaseRunner(
            process_runner=lambda *args, **kwargs: calls.append(args)
        )("codex", context)
    assert calls == []


def test_runtime_loads_only_selected_lane_signer(tmp_path):
    base = tmp_path / "lanes"
    selected = base / "codex-mac-1"
    (selected / "auth").mkdir(parents=True)
    selected.chmod(0o700)
    (selected / "auth").chmod(0o700)
    for child in ("worktrees", "handoffs", "receipts", "health"):
        (selected / child).mkdir()
    key = Ed25519PrivateKey.generate()
    key_path = selected / "auth" / "receipt-signing-key.pem"
    key_path.write_bytes(jobs_receipts.private_key_pem(key))
    key_path.chmod(0o600)
    (selected / "lane-config.json").write_text(
        json.dumps(
            {
                "key_id": "lane:codex-mac-1:v1",
                "public_key": "base64:"
                + base64.b64encode(
                    key.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ).decode("ascii"),
            }
        ),
        encoding="utf-8",
    )

    assert jobs_runtime._selected_lane_root(base, "codex-mac-1") == selected
    signer, verifier = jobs_runtime.load_lane_crypto(selected)
    envelope = signer.sign({"ok": True}, receipt_id="r_test")
    assert verifier.verify(envelope, expected={}) is None


def test_runtime_production_preflight_refuses_non_idle_health(tmp_path):
    context = _context(tmp_path, "codex")
    health = jobs_lanes.LaneHealth(
        lane_id="codex-mac-1",
        state="BUILDING",
        status="PASS",
        failure_class=None,
        reason_code="CAPACITY_FULL",
        observed_at=100,
        expires_at=220,
        executor_version="0.146.0",
        available_capacity=0,
        safe_detail={},
    )
    probes = jobs_runtime.production_preflight_probes(
        lane_root=context.lane_root, lane_health=health
    )
    result = probes.auth_check("codex-mac-1")
    assert (result.status, result.code) == ("BLOCKED", "AUTH_REQUIRED")
