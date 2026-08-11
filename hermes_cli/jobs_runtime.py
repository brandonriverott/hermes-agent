"""Canonical lane-aware dispatcher runtime — the single production entry point.

``dispatch_job_once`` is the one function production calls. It loads the
persisted identity, rejects header-model disagreement before any provider
work, records fresh lane-health observations, selects a seat under the v1
policy, and delegates the actual evidence-authorized dispatch to the existing
provider-free :func:`hermes_cli.jobs_dispatch.dispatch_once`.
"""

from __future__ import annotations

from pathlib import Path
import base64
import json
import shutil
import subprocess
from cryptography.hazmat.primitives import serialization
from typing import Any, Callable, Mapping, Optional, Sequence

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_dispatch as dispatch
from hermes_cli import jobs_executors
from hermes_cli import jobs_identity as ji
from hermes_cli import jobs_lanes
from hermes_cli import jobs_graph
from hermes_cli import jobs_harness
from hermes_cli import jobs_receipts


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
    try:
        header = dispatch.parse_legacy_execution_header(job.goal)
    except dispatch.LegacyMetadataError:
        header = None
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
        effort=(
            header.effort
            if header is not None
            else ("high" if identity.executor == "codex" else "max")
        ),
        max_turns=120 if header is None else header.max_turns,
    )


def _selected_lane_root(base: Path, lane_id: str) -> Path:
    """Resolve an explicit lane root, preserving the Task 5 test seam."""

    root = Path(base)
    child = root / lane_id
    return child if child.is_dir() and not child.is_symlink() else root


def load_lane_crypto(lane_root: Path):
    """Load the selected lane's private signer and matching public verifier."""

    root = Path(lane_root)
    try:
        config = json.loads((root / "lane-config.json").read_text(encoding="utf-8"))
        key_id = config["key_id"]
        configured_public = base64.b64decode(
            str(config["public_key"]).removeprefix("base64:"), validate=True
        )
        private_key = jobs_receipts.load_private_key(
            root / "auth" / "receipt-signing-key.pem"
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("selected lane signing identity is unavailable") from exc
    if not isinstance(key_id, str) or not key_id.startswith("lane:"):
        raise ValueError("selected lane signing identity is invalid")
    observed_public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if configured_public != observed_public:
        raise ValueError("selected lane signing identity is inconsistent")
    signer = jobs_harness.ReceiptSigner(key_id, private_key)
    verifier = jobs_graph.ReceiptVerifier(
        trusted_keys={key_id: private_key.public_key()}
    )
    return signer, verifier


def production_preflight_probes(
    *,
    lane_root: Path,
    lane_health: jobs_lanes.LaneHealth,
    remote_repository_ok: Optional[bool] = None,
) -> jobs_harness.PreflightProbes:
    """Concrete, bounded checks run immediately before custody is acquired."""

    root = Path(lane_root)

    def result(passed: bool, code: str, **detail):
        return jobs_harness.ProbeResult(
            "PASS" if passed else "BLOCKED",
            "OK" if passed else code,
            detail if passed else {},
        )

    def file_paths(value):
        repository, outputs = value
        repository_ok = (
            remote_repository_ok
            if remote_repository_ok is not None
            else Path(repository).is_dir()
        )
        okay = repository_ok and all(Path(path).is_dir() for path in outputs)
        return result(okay, "FILE_PATHS_INVALID", repository=str(repository))

    def permissions(_value):
        required = tuple(root / name for name in ("auth", "worktrees", "handoffs", "receipts", "health"))
        okay = root.is_dir() and not root.is_symlink() and all(
            path.is_dir() and not path.is_symlink() for path in required
        )
        return result(okay, "LANE_PERMISSIONS_INVALID", path=str(root))

    def memory_store(paths):
        okay = all(Path(path).is_file() for path in paths)
        return result(okay, "MEMORY_SCOPE_INVALID", paths=[str(path) for path in paths])

    def worktree(value):
        repository, base, branch = value
        if remote_repository_ok is not None:
            return result(
                remote_repository_ok,
                "WORKTREE_PRECONDITION_FAILED",
                branch=str(branch),
                base_commit=str(base),
            )
        checked = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "-e", f"{base}^{{commit}}"],
            check=False,
            capture_output=True,
            timeout=120,
        )
        branch_check = subprocess.run(
            ["git", "-C", str(repository), "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
            check=False,
            capture_output=True,
            timeout=120,
        )
        return result(
            checked.returncode == 0 and branch_check.returncode != 0,
            "WORKTREE_PRECONDITION_FAILED",
            branch=str(branch),
            base_commit=str(base),
        )

    def tool_routing(value):
        lane_id, executor, model = value
        okay = (
            lane_id == lane_health.lane_id
            and lane_health.status == "PASS"
            and lane_health.state == "IDLE"
            and lane_id.startswith(str(executor) + "-")
            and bool(model)
        )
        return result(okay, "TOOL_ROUTING_INVALID", lane_id=str(lane_id), executor=str(executor), model=str(model))

    def auth_check(_value):
        return result(
            lane_health.status == "PASS" and lane_health.state == "IDLE",
            "AUTH_REQUIRED",
            authenticated=True,
        )

    def resource_budget(_value):
        try:
            free = shutil.disk_usage(root).free
        except OSError:
            free = 0
        return result(free >= 1024 * 1024 * 1024, "DISK_BUDGET_LOW", free_bytes=free, required_bytes=1024 * 1024 * 1024)

    return jobs_harness.PreflightProbes(
        file_paths=file_paths,
        permissions=permissions,
        memory_store=memory_store,
        worktree=worktree,
        tool_routing=tool_routing,
        auth_check=auth_check,
        resource_budget=resource_budget,
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
    executor_registry: Optional[jobs_executors.ExecutorRegistry],
    gate: Optional[Callable[..., Any]],
    activation_gate: Optional[Callable[..., Any]],
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

    selected_root = _selected_lane_root(Path(lane_root), decision.lane_id)
    selected_health = next(
        (item for item in health if item.lane_id == decision.lane_id), None
    )
    if selected_health is None:
        raise ValueError("selected lane has no durable health evidence")
    if signer is None or verifier is None:
        signer, verifier = load_lane_crypto(selected_root)
    if probes is None:
        remote_repository_ok = None
        if "-pc-" in decision.lane_id:
            from hermes_cli import jobs_reliability

            remote_repository_ok = jobs_reliability.production_remote_preflight(
                provider=identity.executor,
                repository=Path(repo_path),
                base_commit=base_commit,
                branch=branch,
                lane_id=decision.lane_id,
            )
        probes = production_preflight_probes(
            lane_root=selected_root,
            lane_health=selected_health,
            remote_repository_ok=remote_repository_ok,
        )
    if executor_registry is None:
        executor_registry = jobs_executors.production_registry()
    if gate is None or activation_gate is None:
        from hermes_cli import jobs_reliability

        gate = gate or jobs_reliability.production_gate
        activation_gate = (
            activation_gate or jobs_reliability.production_completion_gate
        )

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
        lane_root=selected_root,
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
