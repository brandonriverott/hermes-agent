#!/usr/bin/env python3
"""Plan or create isolated, non-secret Jobs builder-lane directories."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_lanes, jobs_receipts
from hermes_constants import get_hermes_home


_SUBDIRECTORIES = {
    "auth": "auth",
    "worktrees": "worktrees",
    "handoffs": "handoffs",
    "receipts": "receipts",
    "health": "health",
}
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
_UNSAFE_ROOT_CHARACTERS = set("$%*?[]{}")


class ProvisionInputError(ValueError):
    """Invalid root, config, or command combination."""


class ProvisionCollisionError(RuntimeError):
    """An existing target cannot be proven to be this lane."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = tuple(paths)
        super().__init__("lane target collision")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--host", choices=("pc", "mac"), required=True)
    parser.add_argument("--root")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("provision", "rollback"), default="provision")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def _config_root(config_path: Path, host_id: str) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise ProvisionInputError("PyYAML is required to read lane root config") from exc
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ProvisionInputError(f"cannot load lane config: {config_path}") from exc
    value: object = raw
    for key in ("jobs", "lanes", "roots", host_id):
        if not isinstance(value, dict) or key not in value:
            raise ProvisionInputError(
                f"missing config value: jobs.lanes.roots.{host_id}"
            )
        value = value[key]
    if not isinstance(value, str) or not value.strip():
        raise ProvisionInputError(
            f"invalid config value: jobs.lanes.roots.{host_id}"
        )
    return value


def _resolve_root(args: argparse.Namespace) -> Path:
    raw = args.root
    if raw is None:
        config_path = args.config or (get_hermes_home() / "config.yaml")
        raw = _config_root(config_path, args.host)
    if not isinstance(raw, str) or not raw:
        raise ProvisionInputError("unsafe lane root: empty")
    if any(character in raw for character in _UNSAFE_ROOT_CHARACTERS):
        raise ProvisionInputError("unsafe lane root: unresolved or globbed path")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise ProvisionInputError("unsafe lane root: path must be absolute")
    resolved = candidate.resolve(strict=False)
    home = Path.home().resolve()
    if resolved == Path(resolved.anchor) or resolved == home:
        raise ProvisionInputError("unsafe lane root: broad target")
    if os.path.lexists(candidate) and candidate.is_symlink():
        raise ProvisionInputError("unsafe lane root: symlink")
    return resolved


def _lane_config(
    lane: jobs_lanes.LaneDefinition,
    registry: jobs_lanes.LaneRegistry,
    *,
    public_key: bytes,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy_version": registry.policy_version,
        "policy_digest": jobs_lanes.registry_digest(registry),
        "lane_id": lane.id,
        "host_id": lane.host_id,
        "executor": lane.executor,
        "slot": lane.slot,
        "model_patterns": list(lane.model_patterns),
        "key_id": f"lane:{lane.id}:v1",
        "public_key": "base64:" + base64.b64encode(public_key).decode("ascii"),
        "directories": dict(_SUBDIRECTORIES),
    }


def _write_private_file(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _stage_lane(
    stage_root: Path,
    lane: jobs_lanes.LaneDefinition,
    registry: jobs_lanes.LaneRegistry,
) -> Path:
    lane_dir = stage_root / lane.id
    lane_dir.mkdir(mode=0o700)
    if os.name == "posix":
        lane_dir.chmod(0o700)
    for name in _SUBDIRECTORIES.values():
        directory = lane_dir / name
        directory.mkdir(mode=0o700)
        if os.name == "posix":
            directory.chmod(0o700)

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if os.name == "posix":
        _write_private_file(
            lane_dir / "auth" / "receipt-signing-key.pem",
            jobs_receipts.private_key_pem(private_key),
        )
    config_bytes = (
        json.dumps(
            _lane_config(lane, registry, public_key=public_key),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    _write_private_file(lane_dir / "lane-config.json", config_bytes)
    return lane_dir


def _contains_symlink(root: Path) -> bool:
    if root.is_symlink():
        return True
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in (*directory_names, *file_names):
            if (current / name).is_symlink():
                return True
    return False


def _expected_identity(
    lane: jobs_lanes.LaneDefinition, registry: jobs_lanes.LaneRegistry
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy_version": registry.policy_version,
        "policy_digest": jobs_lanes.registry_digest(registry),
        "lane_id": lane.id,
        "host_id": lane.host_id,
        "executor": lane.executor,
        "slot": lane.slot,
        "model_patterns": list(lane.model_patterns),
        "directories": _SUBDIRECTORIES,
    }


def _validate_existing_lane(
    target: Path,
    lane: jobs_lanes.LaneDefinition,
    registry: jobs_lanes.LaneRegistry,
) -> bool:
    if not os.path.lexists(target):
        return False
    if target.is_symlink() or not target.is_dir() or _contains_symlink(target):
        raise ProvisionCollisionError((target,))
    config_path = target / "lane-config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProvisionCollisionError((target,)) from exc
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ProvisionCollisionError((target,))
    if any(config.get(key) != value for key, value in _expected_identity(lane, registry).items()):
        raise ProvisionCollisionError((target,))
    key_id = config.get("key_id")
    encoded_public = config.get("public_key")
    if not isinstance(key_id, str) or key_id != f"lane:{lane.id}:v1":
        raise ProvisionCollisionError((target,))
    if not isinstance(encoded_public, str) or not encoded_public.startswith("base64:"):
        raise ProvisionCollisionError((target,))
    try:
        configured_public = base64.b64decode(
            encoded_public.removeprefix("base64:"), validate=True
        )
    except ValueError as exc:
        raise ProvisionCollisionError((target,)) from exc
    if len(configured_public) != 32:
        raise ProvisionCollisionError((target,))

    expected_paths = {
        target / "lane-config.json",
        *(target / name for name in _SUBDIRECTORIES.values()),
    }
    if not all(path.exists() for path in expected_paths):
        raise ProvisionCollisionError((target,))
    if os.name == "posix":
        expected_modes = {
            target: 0o700,
            target / "lane-config.json": 0o600,
            **{target / name: 0o700 for name in _SUBDIRECTORIES.values()},
            target / "auth" / "receipt-signing-key.pem": 0o600,
        }
        for path, expected_mode in expected_modes.items():
            try:
                observed_mode = stat.S_IMODE(path.stat().st_mode)
            except OSError as exc:
                raise ProvisionCollisionError((target,)) from exc
            if observed_mode != expected_mode:
                raise ProvisionCollisionError((target,))
        try:
            private_key = jobs_receipts.load_private_key(
                target / "auth" / "receipt-signing-key.pem"
            )
        except (OSError, ValueError, jobs_receipts.SigningKeyProtectionError) as exc:
            raise ProvisionCollisionError((target,)) from exc
        observed_public = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        if observed_public != configured_public:
            raise ProvisionCollisionError((target,))
    return True


def _validate_plan(
    root: Path,
    lanes: Sequence[jobs_lanes.LaneDefinition],
    registry: jobs_lanes.LaneRegistry,
) -> tuple[list[jobs_lanes.LaneDefinition], list[jobs_lanes.LaneDefinition]]:
    existing: list[jobs_lanes.LaneDefinition] = []
    missing: list[jobs_lanes.LaneDefinition] = []
    collisions: list[Path] = []
    for lane in lanes:
        target = root / lane.id
        try:
            if _validate_existing_lane(target, lane, registry):
                existing.append(lane)
            else:
                missing.append(lane)
        except ProvisionCollisionError:
            collisions.append(target)
    if collisions:
        raise ProvisionCollisionError(collisions)
    return existing, missing


def _apply_plan(
    root: Path,
    missing: Sequence[jobs_lanes.LaneDefinition],
    registry: jobs_lanes.LaneRegistry,
) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ProvisionInputError("lane root must be an existing real directory")
    if not missing:
        return
    stage_root = Path(tempfile.mkdtemp(prefix=".jobs-lanes-stage-", dir=root))
    if os.name == "posix":
        stage_root.chmod(0o700)
    moved: list[Path] = []
    try:
        for lane in missing:
            _stage_lane(stage_root, lane, registry)
        for lane in missing:
            source = stage_root / lane.id
            target = root / lane.id
            source.rename(target)
            moved.append(target)
        stage_root.rmdir()
    except Exception:
        for target in moved:
            shutil.rmtree(target, ignore_errors=True)
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def _json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _rollback_inventory(
    root: Path,
    lanes: Sequence[jobs_lanes.LaneDefinition],
    registry: jobs_lanes.LaneRegistry,
) -> list[str]:
    inventory = []
    collisions = []
    for lane in lanes:
        target = root / lane.id
        if not os.path.lexists(target):
            continue
        try:
            _validate_existing_lane(target, lane, registry)
        except ProvisionCollisionError:
            collisions.append(target)
        else:
            inventory.append(str(target))
    if collisions:
        raise ProvisionCollisionError(collisions)
    return inventory


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args.mode == "rollback" and args.apply:
            raise ProvisionInputError(
                "rollback is inventory-only; review exact paths before removal"
            )
        root = _resolve_root(args)
        registry = jobs_lanes.load_lane_registry(args.registry)
        lanes = tuple(lane for lane in registry.lanes if lane.host_id == args.host)
        if args.mode == "rollback":
            inventory = _rollback_inventory(root, lanes, registry)
            _json(
                {
                    "schema_version": 1,
                    "status": "DRY_RUN",
                    "mode": "rollback",
                    "host_id": args.host,
                    "root": str(root),
                    "planned_lanes": inventory,
                    "changed": [],
                }
            )
            return 0

        _, missing = _validate_plan(root, lanes, registry)
        planned = [lane.id for lane in lanes]
        changed = [lane.id for lane in missing]
        if args.apply:
            _apply_plan(root, missing, registry)
        _json(
            {
                "schema_version": 1,
                "status": "APPLIED" if args.apply else "DRY_RUN",
                "mode": "provision",
                "host_id": args.host,
                "root": str(root),
                "policy_version": registry.policy_version,
                "policy_digest": jobs_lanes.registry_digest(registry),
                "planned_lanes": planned,
                "changed": changed,
            }
        )
        return 0
    except ProvisionCollisionError as exc:
        _json(
            {
                "schema_version": 1,
                "status": "BLOCKED",
                "reason_code": "PATH_COLLISION",
                "collision_paths": [str(path) for path in exc.paths],
            }
        )
        return 2
    except (ProvisionInputError, jobs_lanes.InvalidLaneRegistry, OSError) as exc:
        print(f"lane provisioning error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
