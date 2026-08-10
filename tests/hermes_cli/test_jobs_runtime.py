"""Task 5 focused behavioral tests for ``jobs_runtime.dispatch_job_once``.

These tests reuse the real ReliabilityRig repo/DB/signing/evidence helpers
from ``test_jobs_reliability_e2e`` instead of duplicating any machinery.
"""

from __future__ import annotations

import pytest

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_dispatch as dispatch
from hermes_cli import jobs_executors
from hermes_cli import jobs_identity as ji
from hermes_cli import jobs_lanes
from hermes_cli import jobs_runtime as rt

from tests.hermes_cli.test_jobs_reliability_e2e import (
    _authoritative_gate,
    _passing_activation,
    _passing_executor,
    reliability_rig,
)


# ---------------------------------------------------------------------------
# Health observation builders (thin, no machinery)
# ---------------------------------------------------------------------------
def _idle_health(now: int, lane_id: str) -> jobs_lanes.LaneHealth:
    return jobs_lanes.LaneHealth(
        lane_id=lane_id,
        state="IDLE",
        status="PASS",
        failure_class=None,
        reason_code="OK",
        observed_at=now,
        expires_at=now + 300,
        executor_version="test-1.0",
        available_capacity=1,
        safe_detail={},
    )


def _blocked_health(
    now: int, lane_id: str, reason_code: str = "HOST_UNREACHABLE"
) -> jobs_lanes.LaneHealth:
    return jobs_lanes.LaneHealth(
        lane_id=lane_id,
        state="BLOCKED",
        status="BLOCKED",
        failure_class="INFRA_FAILURE",
        reason_code=reason_code,
        observed_at=now,
        expires_at=now + 300,
        executor_version="test-1.0",
        available_capacity=0,
        safe_detail={},
    )


def _busy_health(now: int, lane_id: str) -> jobs_lanes.LaneHealth:
    return jobs_lanes.LaneHealth(
        lane_id=lane_id,
        state="BUILDING",
        status="PASS",
        failure_class=None,
        reason_code="ACTIVE_LEASE",
        observed_at=now,
        expires_at=now + 300,
        executor_version="test-1.0",
        available_capacity=0,
        safe_detail={},
    )


def _dispatch(
    rig,
    *,
    job_id: str,
    lane_health,
    registry,
    gate=None,
):
    return rt.dispatch_job_once(
        conn=rig.conn,
        job_id=job_id,
        repo_path=rig.repo,
        base_commit=rig.base,
        branch="jobs/reliability-e2e",
        output_parents=(rig.output,),
        scoped_memory_paths=(rig.memory,),
        lane_root=rig.root / "lanes",
        probes=rig.probes,
        signer=rig.signer,
        verifier=rig.verifier,
        worker_id="worker:task5",
        executor_registry=registry,
        gate=gate if gate is not None else _authoritative_gate(rig),
        activation_gate=_passing_activation,
        observed_at="2026-08-09T00:00:00Z",
        now=rig.clock,
        lane_health=lane_health,
        lease_seconds=60,
        failure_journal_path=rig.home / "logs" / "failure.jsonl",
    )


# ---------------------------------------------------------------------------
# Selected + fallback + queued routing
# ---------------------------------------------------------------------------
def test_selected_codex_pc_path(reliability_rig):
    rig = reliability_rig
    job_id = jdb.create_job(
        rig.conn, name="task5 codex", goal="build", requested_lane="codex"
    )
    result = _dispatch(
        rig,
        job_id=job_id,
        lane_health=[_idle_health(rig.clock, "codex-pc-1")],
        registry=jobs_executors.registry.with_reliability_adapters(
            {"codex": _passing_executor(rig)}
        ),
    )
    assert result["claimed"] is True
    assert result["state"] == "COMPLETED"
    assert result["lane_id"] == "codex-pc-1"
    states = [
        row["target_state"]
        for row in jdb.list_transitions(rig.conn, job_id)
    ]
    assert states[-1] == "COMPLETED"


def test_codex_pc_blocked_falls_back_to_mac(reliability_rig):
    rig = reliability_rig
    job_id = jdb.create_job(
        rig.conn, name="task5 fallback", goal="build", requested_lane="codex"
    )
    lane_health = [
        _blocked_health(rig.clock, lane)
        for lane in ("codex-pc-1", "codex-pc-2", "codex-pc-3")
    ]
    lane_health.append(_idle_health(rig.clock, "codex-mac-1"))
    result = _dispatch(
        rig,
        job_id=job_id,
        lane_health=lane_health,
        registry=jobs_executors.registry.with_reliability_adapters(
            {"codex": _passing_executor(rig)}
        ),
    )
    assert result["claimed"] is True
    assert result["state"] == "COMPLETED"
    assert result["lane_id"] == "codex-mac-1"


def test_no_codex_seat_queued_with_zero_claude_calls(reliability_rig):
    rig = reliability_rig

    def claude_must_not_run(_context):
        raise AssertionError("claude adapter must not be called")

    job_id = jdb.create_job(
        rig.conn, name="task5 queued", goal="build", requested_lane="codex"
    )
    lane_health = [
        _blocked_health(rig.clock, lane)
        for lane in ("codex-pc-1", "codex-pc-2", "codex-pc-3")
    ]
    lane_health.extend(
        _busy_health(rig.clock, lane)
        for lane in ("codex-mac-1", "codex-mac-2", "codex-mac-3")
    )
    result = _dispatch(
        rig,
        job_id=job_id,
        lane_health=lane_health,
        registry=jobs_executors.registry.with_reliability_adapters(
            {"claude": claude_must_not_run, "codex": _passing_executor(rig)}
        ),
    )
    assert result["claimed"] is False
    assert result["state"] == "QUEUED"
    assert result["reason"] == "CAPACITY_FULL"
    assert jdb.get_attempts(rig.conn, job_id) == []


def test_selected_claude_mirror_path(reliability_rig):
    rig = reliability_rig
    result = _dispatch(
        rig,
        job_id=rig.job_id,
        lane_health=[_idle_health(rig.clock, "claude-pc-1")],
        registry=jobs_executors.registry.with_reliability_adapters(
            {"claude": _passing_executor(rig)}
        ),
    )
    assert result["claimed"] is True
    assert result["state"] == "COMPLETED"
    assert result["lane_id"] == "claude-pc-1"


# ---------------------------------------------------------------------------
# Fail-before-dispatch guarantees
# ---------------------------------------------------------------------------
def test_header_model_disagreement_fails_before_dispatch(reliability_rig):
    rig = reliability_rig
    goal = (
        "REPO_PATH=/tmp/task5-repo\n"
        "BASE_COMMIT=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "MODEL=claude-opus-5\n"
        "build something"
    )
    job_id = jdb.create_job(
        rig.conn, name="task5 header", goal=goal, requested_lane="codex"
    )
    with pytest.raises(dispatch.LegacyMetadataError):
        _dispatch(
            rig,
            job_id=job_id,
            lane_health=[],
            registry=jobs_executors.registry,
        )
    assert jdb.get_attempts(rig.conn, job_id) == []


def test_missing_adapter_fails_before_dispatch(reliability_rig):
    rig = reliability_rig
    job_id = jdb.create_job(
        rig.conn, name="task5 adapter", goal="build", requested_lane="codex"
    )
    with pytest.raises(jobs_executors.UnsupportedExecutor):
        _dispatch(
            rig,
            job_id=job_id,
            lane_health=[_idle_health(rig.clock, "codex-pc-1")],
            registry=jobs_executors.registry.with_reliability_adapters(
                {"claude": _passing_executor(rig)}
            ),
        )
    assert jdb.get_attempts(rig.conn, job_id) == []


def test_unknown_identity_fails_before_dispatch(reliability_rig):
    rig = reliability_rig
    job_id = jdb.create_job(
        rig.conn, name="task5 identity", goal="build", requested_lane="codex"
    )
    rig.conn.execute(
        "UPDATE jobs SET specialist = ? WHERE id = ?",
        ("unknown-builder", job_id),
    )
    rig.conn.commit()
    with pytest.raises(ji.UnsupportedJobLane):
        _dispatch(
            rig,
            job_id=job_id,
            lane_health=[],
            registry=jobs_executors.registry,
        )
    assert jdb.get_attempts(rig.conn, job_id) == []
