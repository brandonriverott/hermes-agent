"""Canonical lane-aware dispatcher runtime — the single production entry point.

``dispatch_job_once`` is the one function production calls. It loads the
persisted identity, rejects header-model disagreement before any provider
work, records fresh lane-health observations, selects a seat under the v1
policy, and delegates the actual evidence-authorized dispatch to the existing
provider-free :func:`hermes_cli.jobs_dispatch.dispatch_once`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_dispatch as dispatch
from hermes_cli import jobs_executors
from hermes_cli import jobs_identity as ji
from hermes_cli import jobs_lanes


# ---------------------------------------------------------------------------
# Bounded public result
# ---------------------------------------------------------------------------
def _bounded_result(
    *,
    claimed: bool,
    reason: str,
    job_id: str,
    attempt_id: Optional[str] = None,
    state: Optional[str] = None,
    lane_id: Optional[str] = None,
    failure_class: Optional[str] = None,
    retry_action: Optional[str] = None,
    commit: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "claimed": claimed,
        "reason": reason,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "state": state,
        "lane_id": lane_id,
        "failure_class": failure_class,
        "retry_action": retry_action,
        "commit": commit,
    }


# ---------------------------------------------------------------------------
# Identity enforcement
# ---------------------------------------------------------------------------
def _load_and_check_identity(
    conn,
    job_id: str,
) -> tuple[jdb.Job, ji.JobIdentity]:
    job = jdb.get_job(conn, job_id)
    if job is None:
        raise ValueError(f"no such job: {job_id!r}")
    return job, ji.resolve_canonical_specialist(job.specialist)


def _validate_legacy_header_against_identity(
    job: jdb.Job,
    identity: ji.JobIdentity,
) -> None:
    """Fail before dispatch when a legacy header contradicts the persisted identity."""
    try:
        header = dispatch.parse_legacy_execution_header(job.goal)
    except dispatch.LegacyMetadataError:
        # No parseable header or a malformed one: nothing to disagree with.
        return
    if header.model != identity.model:
        raise dispatch.LegacyMetadataError(
            code="MODEL_DISAGREES",
            key="MODEL",
            detail=(
                f"goal MODEL header '{header.model}' disagrees with "
                f"persisted identity model '{identity.model}'"
            ),
        )


# ---------------------------------------------------------------------------
# DispatchSpec construction
# ---------------------------------------------------------------------------
def _build_spec(
    *,
    job: jdb.Job,
    identity: ji.JobIdentity,
    lane_decision: jobs_lanes.LaneDecision,
    repo_path: Path,
    base_commit: str,
    branch: str,
    output_parents: Sequence[Path],
    scoped_memory_paths: Sequence[Path],
    lane_root: Path,
) -> dispatch.DispatchSpec:
    lane_root_path = Path(lane_root)
    lane_root_path.mkdir(parents=True, exist_ok=True)
    for name in ("auth", "worktrees", "handoffs", "receipts", "health"):
        (lane_root_path / name).mkdir(parents=True, exist_ok=True)
    worktree_parent = lane_root_path / "worktrees"
    return dispatch.DispatchSpec(
        repository=Path(repo_path),
        base_commit=base_commit,
        branch=branch,
        output_parents=tuple(Path(path) for path in output_parents),
        scoped_memory_paths=tuple(Path(path) for path in scoped_memory_paths),
        requested_lane=identity.requested_lane,
        lane_id=lane_decision.lane_id or "",
        executor=identity.executor,
        specialist=identity.specialist,
        model=identity.model,
        worktree_parent=worktree_parent,
        lane_root=lane_root_path,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def dispatch_job_once(
    *,
    conn,
    job_id: str,
    repo_path: Path,
    base_commit: str,
    branch: str,
    output_parents: Sequence[Path],
    scoped_memory_paths: Sequence[Path],
    lane_root: Path,
    probes: Any,
    signer: Any,
    verifier: Any,
    worker_id: str,
    executor_registry: jobs_executors.ExecutorRegistry,
    gate: Callable[..., Any],
    activation_gate: Callable[..., Any],
    observed_at: str,
    now: int,
    lane_health: Sequence[jobs_lanes.LaneHealth],
    lease_seconds: int = 1800,
    failure_journal_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Load identity → persist health → select lane → dispatch → bounded result.

    This is the single production entry point for the Jobs reliability
    dispatcher. It contains no provider-selection branch.
    """
    # 1. Load and validate persisted identity (unknown specialist fails here).
    job, identity = _load_and_check_identity(conn, job_id)

    # 2. Reject legacy header-model disagreement BEFORE any provider work.
    _validate_legacy_header_against_identity(job, identity)

    # 3. Load the lane registry (throws if invalid).
    registry = jobs_lanes.load_lane_registry()

    # 4. Persist the fresh health observations so the claim path never races.
    for health in lane_health:
        jdb.record_lane_health(conn, health)

    # 5. Select a lane from the durable evidence snapshot (same source the
    #    claim path re-reads, so the decision is reproducible).
    health, active_load = jdb.lane_routing_snapshot(
        conn, registry=registry, now=now
    )
    decision = jobs_lanes.select_lane(
        registry,
        health,
        jobs_lanes.RoutingRequest(
            job_id=job.id,
            executor=identity.executor,
            model=identity.model,
        ),
        active_load=active_load,
        now=now,
    )

    # 6. Handle non-SELECTED decisions truthfully (QUEUED/BLOCKED, no dispatch).
    if decision.status != "SELECTED":
        return _bounded_result(
            claimed=False,
            reason=decision.reason_code,
            job_id=job.id,
            state=decision.status,
        )
    if decision.lane_id is None:
        raise ValueError("selected lane decision has no lane_id")

    # 7. Build the immutable spec and dispatch via the provider-true seam.
    spec = _build_spec(
        job=job,
        identity=identity,
        lane_decision=decision,
        repo_path=repo_path,
        base_commit=base_commit,
        branch=branch,
        output_parents=output_parents,
        scoped_memory_paths=scoped_memory_paths,
        lane_root=lane_root,
    )
    result = dispatch.dispatch_once(
        conn=conn,
        job_id=job.id,
        spec=spec,
        probes=probes,
        signer=signer,
        verifier=verifier,
        worker_id=worker_id,
        executor_registry=executor_registry,
        gate=gate,
        activation_gate=activation_gate,
        observed_at=observed_at,
        now=now,
        lease_seconds=lease_seconds,
        failure_journal_path=failure_journal_path,
        lane_decision=decision,
    )

    # 8. Return the bounded JSON result.
    return _bounded_result(
        claimed=result.claimed,
        reason=result.reason,
        job_id=result.job_id,
        attempt_id=result.attempt_id,
        state=result.state,
        lane_id=decision.lane_id,
        failure_class=result.failure_class,
        retry_action=result.retry_action,
        commit=result.commit,
    )
