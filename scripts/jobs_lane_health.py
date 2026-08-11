#!/usr/bin/env python3
"""Report truthful, sanitized health for isolated Jobs builder lanes."""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cryptography.hazmat.primitives import serialization

from hermes_cli import jobs_lanes, jobs_receipts


_CONFIG_KEYS = {
    "schema_version",
    "policy_version",
    "policy_digest",
    "lane_id",
    "host_id",
    "executor",
    "slot",
    "model_patterns",
    "key_id",
    "public_key",
    "directories",
}
_DIRECTORIES = {
    "auth": "auth",
    "worktrees": "worktrees",
    "handoffs": "handoffs",
    "receipts": "receipts",
    "health": "health",
}
_ALLOWED_ENVIRONMENT = {
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "SYSTEMROOT",
    "WINDIR",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "PATHEXT",
    "COMSPEC",
}
_VERSION_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+){1,3}(?:[-+][A-Za-z0-9.-]+)?)")
_MIN_FREE_DISK_BYTES = 1024 * 1024 * 1024
_MIN_AVAILABLE_MEMORY_BYTES = 512 * 1024 * 1024


class HealthCommandError(ValueError):
    """Invalid operator input or malformed non-secret lane configuration."""


def _probe(
    passed: bool,
    reason_code: str,
    *,
    failure_class: str | None = None,
    safe_detail: dict[str, jobs_lanes.JsonValue] | None = None,
) -> jobs_lanes.ProbeResult:
    return jobs_lanes.ProbeResult(
        passed=passed,
        reason_code=reason_code,
        failure_class=failure_class,
        safe_detail=safe_detail or {},
    )


def _allowed_environment(
    *, lane: jobs_lanes.LaneDefinition, lane_dir: Path
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _ALLOWED_ENVIRONMENT
    }
    if "PATH" not in environment:
        environment["PATH"] = os.defpath
    user_bins = [Path.home() / ".hermes" / "node" / "bin", Path.home() / ".local" / "bin"]
    prefixes = [str(path) for path in user_bins if path.is_dir()]
    if prefixes:
        environment["PATH"] = os.pathsep.join([environment["PATH"], *prefixes])
    auth_dir = str(lane_dir / "auth")
    if lane.executor == "claude":
        environment["CLAUDE_CONFIG_DIR"] = auth_dir
    else:
        environment["CODEX_HOME"] = auth_dir
    return environment


def _run_private(
    command: Sequence[str], *, environment: dict[str, str], cwd: Path
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _load_config(
    lane: jobs_lanes.LaneDefinition,
    lane_dir: Path,
    registry: jobs_lanes.LaneRegistry,
) -> tuple[jobs_lanes.ProbeResult, dict[str, object] | None]:
    config_path = lane_dir / "lane-config.json"
    if config_path.is_symlink():
        return _probe(False, "PATH_SYMLINK", failure_class="INFRA_FAILURE"), None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _probe(False, "LANE_CONFIG_MISSING", failure_class="INFRA_FAILURE"), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return _probe(False, "LANE_CONFIG_INVALID", failure_class="INFRA_FAILURE"), None
    if not isinstance(raw, dict) or set(raw) != _CONFIG_KEYS:
        return _probe(False, "LANE_CONFIG_INVALID", failure_class="INFRA_FAILURE"), None
    expected_identity = {
        "schema_version": 1,
        "policy_version": registry.policy_version,
        "policy_digest": jobs_lanes.registry_digest(registry),
        "lane_id": lane.id,
        "host_id": lane.host_id,
        "executor": lane.executor,
        "slot": lane.slot,
        "model_patterns": list(lane.model_patterns),
    }
    if any(raw.get(key) != value for key, value in expected_identity.items()):
        return _probe(False, "LANE_CONFIG_MISMATCH", failure_class="INFRA_FAILURE"), None
    if raw.get("directories") != _DIRECTORIES:
        return _probe(False, "LANE_CONFIG_INVALID", failure_class="INFRA_FAILURE"), None
    key_id = raw.get("key_id")
    public_key = raw.get("public_key")
    if not isinstance(key_id, str) or not key_id.startswith(f"lane:{lane.id}:"):
        return _probe(False, "LANE_CONFIG_INVALID", failure_class="INFRA_FAILURE"), None
    if not isinstance(public_key, str) or not public_key.startswith("base64:"):
        return _probe(False, "LANE_CONFIG_INVALID", failure_class="INFRA_FAILURE"), None
    try:
        decoded_key = base64.b64decode(public_key.removeprefix("base64:"), validate=True)
    except ValueError:
        return _probe(False, "LANE_CONFIG_INVALID", failure_class="INFRA_FAILURE"), None
    if len(decoded_key) != 32:
        return _probe(False, "LANE_CONFIG_INVALID", failure_class="INFRA_FAILURE"), None
    return _probe(True, "OK"), raw


def _probe_permissions(lane_dir: Path) -> jobs_lanes.ProbeResult:
    if os.name != "posix":
        return _probe(
            True,
            "OK",
            safe_detail={"permission_boundary": "platform_signer_check"},
        )
    expected_modes = {
        lane_dir: 0o700,
        lane_dir / "lane-config.json": 0o600,
        **{lane_dir / name: 0o700 for name in _DIRECTORIES.values()},
    }
    for path, expected_mode in expected_modes.items():
        if path.is_symlink():
            return _probe(False, "PATH_SYMLINK", failure_class="INFRA_FAILURE")
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            return _probe(False, "LANE_LAYOUT_INCOMPLETE", failure_class="INFRA_FAILURE")
        if mode != expected_mode:
            return _probe(False, "LANE_PERMISSIONS_INVALID", failure_class="INFRA_FAILURE")
    return _probe(True, "OK")


def _probe_executable(
    lane: jobs_lanes.LaneDefinition,
    *,
    lane_dir: Path,
    environment: dict[str, str],
) -> jobs_lanes.ProbeResult:
    executable = shutil.which(lane.executor, path=environment.get("PATH"))
    if executable is None:
        return _probe(False, "EXECUTOR_NOT_FOUND", failure_class="INFRA_FAILURE")
    completed = _run_private(
        [executable, "--version"], environment=environment, cwd=lane_dir
    )
    if completed is None or completed.returncode != 0:
        return _probe(False, "EXECUTOR_VERSION_FAILED", failure_class="INFRA_FAILURE")
    match = _VERSION_PATTERN.search(completed.stdout + "\n" + completed.stderr)
    if match is None:
        return _probe(
            False,
            "EXECUTOR_VERSION_UNPARSEABLE",
            failure_class="INFRA_FAILURE",
        )
    return _probe(
        True,
        "OK",
        safe_detail={"executor_version": match.group(1)},
    )


def _executor_auth_hint_exists(lane: jobs_lanes.LaneDefinition, lane_dir: Path) -> bool:
    auth_dir = lane_dir / "auth"
    candidates = (
        (".credentials.json", ".claude.json", "credentials.json")
        if lane.executor == "claude"
        else ("auth.json",)
    )
    return any((auth_dir / name).is_file() for name in candidates)


def _probe_auth(
    lane: jobs_lanes.LaneDefinition,
    *,
    lane_dir: Path,
    environment: dict[str, str],
) -> jobs_lanes.ProbeResult:
    if not _executor_auth_hint_exists(lane, lane_dir):
        return _probe(False, "AUTH_REQUIRED", failure_class="AUTH_INFRA")
    executable = shutil.which(lane.executor, path=environment.get("PATH"))
    if executable is None:
        return _probe(False, "AUTH_REQUIRED", failure_class="AUTH_INFRA")
    command = (
        [executable, "auth", "status", "--json"]
        if lane.executor == "claude"
        else [executable, "login", "status"]
    )
    completed = _run_private(command, environment=environment, cwd=lane_dir)
    if completed is None:
        return _probe(False, "AUTH_CHECK_FAILED", failure_class="AUTH_INFRA")
    private_output = completed.stdout + "\n" + completed.stderr
    if "expired" in private_output.lower():
        return _probe(False, "TOKEN_EXPIRED", failure_class="AUTH_INFRA")
    if lane.executor == "claude":
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            authenticated = any(
                payload.get(key) is True
                for key in ("loggedIn", "authenticated", "isAuthenticated")
            )
            status = payload.get("status")
            if isinstance(status, str) and status.lower() in {
                "authenticated",
                "logged_in",
                "logged-in",
            }:
                authenticated = True
            if completed.returncode == 0 and authenticated:
                return _probe(True, "OK")
    elif completed.returncode == 0 and re.search(
        r"\b(logged in|authenticated)\b", private_output, re.IGNORECASE
    ):
        return _probe(True, "OK")
    return _probe(False, "AUTH_REQUIRED", failure_class="AUTH_INFRA")


def _probe_signer(
    lane_dir: Path, config: dict[str, object] | None
) -> jobs_lanes.ProbeResult:
    if config is None:
        return _probe(False, "SIGNER_UNAVAILABLE", failure_class="AUTH_INFRA")
    if os.name != "posix":
        return _probe(
            False,
            "SIGNING_KEY_UNPROTECTED",
            failure_class="AUTH_INFRA",
        )
    key_path = lane_dir / "auth" / "receipt-signing-key.pem"
    try:
        private_key = jobs_receipts.load_private_key(key_path)
        observed_public = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        configured_public = base64.b64decode(
            str(config["public_key"]).removeprefix("base64:"), validate=True
        )
    except (OSError, ValueError, jobs_receipts.SigningKeyProtectionError):
        return _probe(
            False,
            "SIGNING_KEY_UNPROTECTED",
            failure_class="AUTH_INFRA",
        )
    if observed_public != configured_public:
        return _probe(False, "SIGNING_KEY_MISMATCH", failure_class="AUTH_INFRA")
    return _probe(True, "OK", safe_detail={"key_id": str(config["key_id"])})


def _probe_resources(lane_dir: Path) -> jobs_lanes.ProbeResult:
    try:
        disk_free = shutil.disk_usage(lane_dir).free
    except OSError:
        return _probe(False, "RESOURCE_PROBE_FAILED", failure_class="INFRA_FAILURE")
    if disk_free < _MIN_FREE_DISK_BYTES:
        return _probe(
            False,
            "DISK_BUDGET_LOW",
            failure_class="INFRA_FAILURE",
            safe_detail={"disk_free_bytes": disk_free},
        )
    try:
        import psutil

        memory_available = int(psutil.virtual_memory().available)
    except ImportError:
        memory_available = _linux_mem_available(Path("/proc/meminfo"))
    except (OSError, ValueError):
        memory_available = None
    if memory_available is None:
        return _probe(False, "RESOURCE_PROBE_FAILED", failure_class="INFRA_FAILURE")
    if memory_available < _MIN_AVAILABLE_MEMORY_BYTES:
        return _probe(
            False,
            "MEMORY_BUDGET_LOW",
            failure_class="INFRA_FAILURE",
            safe_detail={"memory_available_bytes": memory_available},
        )
    return _probe(
        True,
        "OK",
        safe_detail={
            "disk_free_bytes": disk_free,
            "memory_available_bytes": memory_available,
        },
    )


def _linux_mem_available(path: Path) -> int | None:
    """Read Linux's kernel-reported available memory without optional psutil."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            key, value, unit = line.split(maxsplit=2)
            if key == "MemAvailable:" and unit == "kB":
                return int(value) * 1024
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def _probe_git(environment: dict[str, str]) -> jobs_lanes.ProbeResult:
    executable = shutil.which("git", path=environment.get("PATH"))
    if executable is None:
        return _probe(False, "GIT_NOT_FOUND", failure_class="INFRA_FAILURE")
    completed = _run_private(
        [executable, "--version"],
        environment=environment,
        cwd=REPO_ROOT,
    )
    if completed is None or completed.returncode != 0:
        return _probe(False, "GIT_UNAVAILABLE", failure_class="INFRA_FAILURE")
    return _probe(True, "OK", safe_detail={"git_available": True})


def _probe_host(host_id: str) -> jobs_lanes.ProbeResult:
    system = platform.system().lower()
    local_match = (host_id == "mac" and system == "darwin") or (
        host_id == "pc" and system in {"linux", "windows"}
    )
    if not local_match:
        return _probe(
            False,
            "HOST_UNREACHABLE",
            failure_class="INFRA_FAILURE",
            safe_detail={"local_host_match": False},
        )
    return _probe(True, "OK", safe_detail={"local_host_match": True})


def _probe_lease(lane_dir: Path) -> jobs_lanes.LeaseObservation:
    lease_path = lane_dir / "health" / "lease.json"
    if not lease_path.exists():
        return jobs_lanes.LeaseObservation(state="IDLE", reason_code="OK")
    if lease_path.is_symlink():
        return jobs_lanes.LeaseObservation(state="FAILED", reason_code="CLEANUP_FAILED")
    try:
        raw = json.loads(lease_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return jobs_lanes.LeaseObservation(state="FAILED", reason_code="CLEANUP_FAILED")
    if not isinstance(raw, dict):
        return jobs_lanes.LeaseObservation(state="FAILED", reason_code="CLEANUP_FAILED")
    state = raw.get("state")
    reason_code = raw.get("reason_code")
    if state not in {"IDLE", "ASSIGNED", "BUILDING", "VERIFYING", "FAILED"}:
        return jobs_lanes.LeaseObservation(state="FAILED", reason_code="CLEANUP_FAILED")
    if not isinstance(reason_code, str) or not reason_code:
        return jobs_lanes.LeaseObservation(state="FAILED", reason_code="CLEANUP_FAILED")
    return jobs_lanes.LeaseObservation(state=state, reason_code=reason_code)


def _unprovisioned_probes(observed_at: int) -> jobs_lanes.LaneProbes:
    skipped = _probe(False, "NOT_EVALUATED", failure_class="INFRA_FAILURE")
    return jobs_lanes.LaneProbes(
        observed_at=observed_at,
        configuration=skipped,
        permissions=skipped,
        executable=skipped,
        auth=skipped,
        signer=skipped,
        resources=skipped,
        git=skipped,
        host=skipped,
        lease=jobs_lanes.LeaseObservation(state="IDLE", reason_code="OK"),
    )


def collect_lane_probes(
    lane: jobs_lanes.LaneDefinition,
    *,
    root: Path,
    registry: jobs_lanes.LaneRegistry,
    observed_at: int,
) -> jobs_lanes.LaneProbes:
    lane_dir = root / lane.id
    if not lane_dir.is_dir() or lane_dir.is_symlink():
        return _unprovisioned_probes(observed_at)
    configuration, config = _load_config(lane, lane_dir, registry)
    environment = _allowed_environment(lane=lane, lane_dir=lane_dir)
    return jobs_lanes.LaneProbes(
        observed_at=observed_at,
        configuration=configuration,
        permissions=_probe_permissions(lane_dir),
        executable=_probe_executable(
            lane, lane_dir=lane_dir, environment=environment
        ),
        auth=_probe_auth(lane, lane_dir=lane_dir, environment=environment),
        signer=_probe_signer(lane_dir, config),
        resources=_probe_resources(lane_dir),
        git=_probe_git(environment),
        host=_probe_host(lane.host_id),
        lease=_probe_lease(lane_dir),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--host", choices=("pc", "mac"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lane", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _select_lanes(
    registry: jobs_lanes.LaneRegistry, *, host: str, requested: Sequence[str]
) -> list[jobs_lanes.LaneDefinition]:
    by_id = {lane.id: lane for lane in registry.lanes}
    if requested:
        selected = []
        for lane_id in requested:
            lane = by_id.get(lane_id)
            if lane is None:
                raise HealthCommandError(f"unknown lane: {lane_id}")
            if lane.host_id != host:
                raise HealthCommandError(
                    f"lane {lane_id} does not belong to host {host}"
                )
            if lane not in selected:
                selected.append(lane)
        return selected
    return [lane for lane in registry.lanes if lane.host_id == host]


def _display_text(health: Sequence[jobs_lanes.LaneHealth]) -> str:
    return "\n".join(
        f"{item.lane_id} {item.status} {item.state} {item.reason_code}"
        for item in health
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        registry = jobs_lanes.load_lane_registry(args.registry)
        selected = _select_lanes(
            registry, host=args.host, requested=tuple(args.lane)
        )
        observed_at = int(time.time())
        root = args.root.expanduser().resolve(strict=False)
        health = [
            jobs_lanes.evaluate_lane_health(
                lane,
                root=root,
                probes=collect_lane_probes(
                    lane,
                    root=root,
                    registry=registry,
                    observed_at=observed_at,
                ),
                now=observed_at,
                ttl_seconds=registry.health_ttl_seconds,
            )
            for lane in selected
        ]
        if args.json:
            payload = {
                "schema_version": 1,
                "policy_version": registry.policy_version,
                "policy_digest": jobs_lanes.registry_digest(registry),
                "observed_at": observed_at,
                "lanes": [asdict(item) for item in health],
            }
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(_display_text(health))
        return 0 if all(item.status == "PASS" for item in health) else 2
    except (HealthCommandError, jobs_lanes.InvalidLaneRegistry, OSError) as exc:
        print(f"lane health error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
