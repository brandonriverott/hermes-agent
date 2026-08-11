"""Opt-in gateway owner for the canonical Jobs reliability dispatcher."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)


def dispatch_configuration() -> Optional[Path]:
    """Return the explicit lane base, or ``None`` unless dispatch is enabled."""

    if os.environ.get("HERMES_JOBS_DISPATCH") != "1":
        return None
    raw = os.environ.get("HERMES_JOBS_LANE_ROOT", "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser()
    if not root.is_absolute():
        return None
    return root.resolve(strict=False)


def collect_local_lane_health(*, lane_root: Path, now: int):
    """Collect fresh health only for seats physically hosted by this process."""

    from hermes_cli import jobs_lanes
    from scripts import jobs_lane_health

    system = platform.system().lower()
    host = "mac" if system == "darwin" else "pc"
    registry = jobs_lanes.load_lane_registry()
    selected = [lane for lane in registry.lanes if lane.host_id == host]
    return [
        jobs_lanes.evaluate_lane_health(
            lane,
            root=Path(lane_root),
            probes=jobs_lane_health.collect_lane_probes(
                lane,
                root=Path(lane_root),
                registry=registry,
                observed_at=int(now),
            ),
            now=int(now),
            ttl_seconds=registry.health_ttl_seconds,
        )
        for lane in selected
    ]


def remote_configuration() -> Optional[tuple[str, Path, Path]]:
    host = os.environ.get("HERMES_JOBS_PC_HOST", "").strip()
    runtime = os.environ.get("HERMES_JOBS_PC_RUNTIME_ROOT", "").strip()
    lanes = os.environ.get("HERMES_JOBS_PC_LANE_ROOT", "").strip()
    if not (host and runtime and lanes):
        return None
    if not Path(runtime).is_absolute() or not Path(lanes).is_absolute():
        return None
    return host, Path(runtime), Path(lanes)


def collect_remote_lane_health(
    *,
    now: int,
    configuration: tuple[str, Path, Path],
    subprocess_run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
):
    """Read the PC's immutable health report over argv-only Tailscale SSH."""

    from hermes_cli import jobs_lanes

    host, runtime, lanes = configuration
    script = runtime / "scripts" / "jobs_lane_health.py"
    try:
        completed = subprocess_run(
            [
                "ssh", host, "python3", str(script), "--host", "pc", "--root",
                str(lanes), "--json",
            ],
            capture_output=True,
            check=False,
            timeout=45,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode not in {0, 2} or len(completed.stdout) > 128 * 1024:
        return []
    try:
        raw = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    registry = jobs_lanes.load_lane_registry()
    if (
        not isinstance(raw, dict)
        or raw.get("policy_version") != registry.policy_version
        or raw.get("policy_digest") != jobs_lanes.registry_digest(registry)
        or not isinstance(raw.get("lanes"), list)
    ):
        return []
    allowed = {lane.id for lane in registry.lanes if lane.host_id == "pc"}
    health = []
    try:
        for item in raw["lanes"]:
            if not isinstance(item, dict) or item.get("lane_id") not in allowed:
                return []
            health.append(jobs_lanes.LaneHealth(**item))
    except (TypeError, ValueError):
        return []
    return health if {item.lane_id for item in health} == allowed else []


def collect_fleet_lane_health(*, lane_root: Path, now: int):
    local = list(collect_local_lane_health(lane_root=lane_root, now=now))
    remote = remote_configuration()
    if remote is not None:
        local.extend(collect_remote_lane_health(now=now, configuration=remote))
    return local


def dispatch_due_job_once(
    *,
    jobs_path=None,
    lane_root: Path,
    now: Optional[int] = None,
    worker_id: str = "gateway-jobs-dispatcher",
    dispatcher: Optional[Callable[..., dict]] = None,
    health_collector: Callable[..., Sequence[object]] = collect_fleet_lane_health,
) -> Optional[dict]:
    """Dispatch the oldest valid unclaimed Job once, or return ``None``."""

    from hermes_cli import jobs_db as jdb
    from hermes_cli import jobs_dispatch
    from hermes_cli import jobs_runtime

    dispatch = dispatcher or jobs_runtime.dispatch_job_once
    instant = int(time.time()) if now is None else int(now)
    root = Path(lane_root)
    canary_job_id = os.environ.get("HERMES_JOBS_CANARY_ID", "").strip()
    conn = jdb.connect(jobs_path)
    try:
        jdb.recover_expired_claims(conn, now=instant)
        health = tuple(health_collector(lane_root=root, now=instant))
        for job in jdb.list_jobs(conn, status="working"):
            if canary_job_id and job.id != canary_job_id:
                continue
            if (
                job.step not in jdb.EXECUTABLE_STEPS
                or job.claimed_by is not None
                or job.current_attempt_id is not None
            ):
                continue
            try:
                header = jobs_dispatch.parse_legacy_execution_header(job.goal)
            except jobs_dispatch.LegacyMetadataError:
                continue
            attempt_index = len(jdb.get_attempts(conn, job.id))
            return dispatch(
                conn=conn,
                job_id=job.id,
                repo_path=Path(header.repo_path),
                base_commit=header.base_commit,
                branch=f"jobs/{job.id}/attempt-{attempt_index}",
                output_parents=(root,),
                scoped_memory_paths=(),
                lane_root=root,
                probes=None,
                signer=None,
                verifier=None,
                worker_id=worker_id,
                executor_registry=None,
                gate=None,
                activation_gate=None,
                observed_at=datetime.fromtimestamp(
                    instant, tz=timezone.utc
                ).isoformat().replace("+00:00", "Z"),
                now=instant,
                lane_health=health,
                failure_journal_path=root / "failure-journal.jsonl",
            )
        return None
    finally:
        conn.close()


class GatewayJobsDispatcherMixin:
    """One supervised, bounded, explicitly enabled dispatcher loop."""

    async def _jobs_dispatcher_watcher(self, interval: float = 10.0) -> None:
        root = dispatch_configuration()
        if root is None:
            logger.info("jobs dispatcher: disabled (explicit opt-in absent)")
            return
        await asyncio.sleep(5)
        while getattr(self, "_running", False):
            try:
                await asyncio.to_thread(
                    dispatch_due_job_once,
                    lane_root=root,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("jobs dispatcher: one tick failed")
            slept = 0.0
            while slept < interval and getattr(self, "_running", False):
                await asyncio.sleep(min(1.0, interval - slept))
                slept += 1.0


__all__ = [
    "GatewayJobsDispatcherMixin",
    "collect_local_lane_health",
    "collect_fleet_lane_health",
    "collect_remote_lane_health",
    "dispatch_configuration",
    "dispatch_due_job_once",
    "remote_configuration",
]
