"""Versioned policy, health, and routing for isolated Jobs builder lanes."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PureWindowsPath
from typing import Literal, Mapping


HostId = Literal["pc", "mac"]
Executor = Literal["claude", "codex"]


class InvalidLaneRegistry(ValueError):
    """Raised when a lane registry violates its strict schema or policy."""


@dataclass(frozen=True)
class HostPolicy:
    id: HostId
    preference: int
    root_config_key: str


@dataclass(frozen=True)
class FallbackPolicy:
    name: str
    enabled: bool
    preserve_executor: bool
    preserve_model: bool
    from_host: HostId
    to_host: HostId
    eligible_reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class LaneDefinition:
    id: str
    host_id: HostId
    executor: Executor
    slot: int
    model_patterns: tuple[str, ...]


@dataclass(frozen=True)
class LaneRegistry:
    schema_version: int
    policy_version: str
    health_ttl_seconds: int
    hosts: tuple[HostPolicy, ...]
    fallback: FallbackPolicy
    lanes: tuple[LaneDefinition, ...]
    source_bytes: bytes


_TOP_LEVEL_KEYS = {
    "schema_version",
    "policy_version",
    "health_ttl_seconds",
    "hosts",
    "fallback",
    "lanes",
}
_HOST_KEYS = {"id", "preference", "root_config_key"}
_FALLBACK_KEYS = {
    "name",
    "enabled",
    "preserve_executor",
    "preserve_model",
    "from_host",
    "to_host",
    "eligible_reason_codes",
}
_LANE_KEYS = {"id", "host_id", "executor", "slot", "model_patterns"}
_HOST_IDS = {"pc", "mac"}
_EXECUTORS = {"claude", "codex"}
_SECRET_FIELD = re.compile(r"token|secret|password|credential|cookie", re.IGNORECASE)


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidLaneRegistry(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise InvalidLaneRegistry(f"invalid JSON constant: {value}")


def _reject_sensitive_content(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _SECRET_FIELD.search(key):
                raise InvalidLaneRegistry(f"secret field is forbidden: {key}")
            _reject_sensitive_content(item)
        return
    if isinstance(value, list):
        for item in value:
            _reject_sensitive_content(item)
        return
    if isinstance(value, str) and (
        Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ):
        raise InvalidLaneRegistry(f"absolute path is forbidden: {value}")


def _reject_unknown_keys(
    value: Mapping[str, object], allowed: set[str], where: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise InvalidLaneRegistry(f"unknown {where} field: {unknown[0]}")
    missing = sorted(allowed - set(value))
    if missing:
        raise InvalidLaneRegistry(f"missing {where} field: {missing[0]}")


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidLaneRegistry(f"{where} must be a non-empty string")
    if value != value.strip():
        raise InvalidLaneRegistry(f"{where} must not have surrounding whitespace")
    return value


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidLaneRegistry(f"{where} must be an integer")
    return value


def _boolean(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidLaneRegistry(f"{where} must be boolean")
    return value


def _string_tuple(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise InvalidLaneRegistry(f"{where} must be a non-empty list")
    result = tuple(_string(item, where) for item in value)
    if len(set(result)) != len(result):
        raise InvalidLaneRegistry(f"duplicate value in {where}")
    return result


def _host_id(value: object, where: str) -> HostId:
    host_id = _string(value, where)
    if host_id not in _HOST_IDS:
        raise InvalidLaneRegistry(f"{where} must be pc or mac")
    return host_id  # type: ignore[return-value]


def _executor(value: object, where: str) -> Executor:
    executor = _string(value, where)
    if executor not in _EXECUTORS:
        raise InvalidLaneRegistry(f"{where} must be claude or codex")
    return executor  # type: ignore[return-value]


def _parse_registry(source_bytes: bytes) -> tuple[dict[str, object], LaneRegistry]:
    try:
        raw = json.loads(
            source_bytes,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_constant,
        )
    except InvalidLaneRegistry:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidLaneRegistry("lane registry is not valid UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise InvalidLaneRegistry("lane registry must be an object")

    _reject_sensitive_content(raw)
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "registry")

    schema_version = _integer(raw["schema_version"], "schema_version")
    if schema_version != 1:
        raise InvalidLaneRegistry(f"unsupported schema_version: {schema_version}")
    policy_version = _string(raw["policy_version"], "policy_version")
    if policy_version != "jobs-lanes.v1":
        raise InvalidLaneRegistry("policy_version must be jobs-lanes.v1")
    health_ttl_seconds = _integer(
        raw["health_ttl_seconds"], "health_ttl_seconds"
    )
    if health_ttl_seconds <= 0:
        raise InvalidLaneRegistry("health_ttl_seconds must be positive")

    raw_hosts = raw["hosts"]
    if not isinstance(raw_hosts, list):
        raise InvalidLaneRegistry("hosts must be a list")
    hosts: list[HostPolicy] = []
    for index, item in enumerate(raw_hosts):
        if not isinstance(item, dict):
            raise InvalidLaneRegistry(f"hosts[{index}] must be an object")
        _reject_unknown_keys(item, _HOST_KEYS, f"hosts[{index}]")
        preference = _integer(item["preference"], f"hosts[{index}].preference")
        if preference < 0:
            raise InvalidLaneRegistry("host preference must not be negative")
        hosts.append(
            HostPolicy(
                id=_host_id(item["id"], f"hosts[{index}].id"),
                preference=preference,
                root_config_key=_string(
                    item["root_config_key"], f"hosts[{index}].root_config_key"
                ),
            )
        )
    if {host.id for host in hosts} != _HOST_IDS or len(hosts) != 2:
        raise InvalidLaneRegistry("hosts must declare pc and mac exactly once")
    if len({host.preference for host in hosts}) != len(hosts):
        raise InvalidLaneRegistry("host preferences must be unique")

    raw_fallback = raw["fallback"]
    if not isinstance(raw_fallback, dict):
        raise InvalidLaneRegistry("fallback must be an object")
    _reject_unknown_keys(raw_fallback, _FALLBACK_KEYS, "fallback")
    fallback = FallbackPolicy(
        name=_string(raw_fallback["name"], "fallback.name"),
        enabled=_boolean(raw_fallback["enabled"], "fallback.enabled"),
        preserve_executor=_boolean(
            raw_fallback["preserve_executor"], "fallback.preserve_executor"
        ),
        preserve_model=_boolean(
            raw_fallback["preserve_model"], "fallback.preserve_model"
        ),
        from_host=_host_id(raw_fallback["from_host"], "fallback.from_host"),
        to_host=_host_id(raw_fallback["to_host"], "fallback.to_host"),
        eligible_reason_codes=_string_tuple(
            raw_fallback["eligible_reason_codes"],
            "fallback.eligible_reason_codes",
        ),
    )
    if fallback.name != "mac_when_pc_unavailable":
        raise InvalidLaneRegistry(
            "fallback.name must be mac_when_pc_unavailable"
        )
    if fallback.from_host == fallback.to_host:
        raise InvalidLaneRegistry("fallback hosts must be different")
    if not fallback.preserve_executor or not fallback.preserve_model:
        raise InvalidLaneRegistry("fallback must preserve executor and model")

    raw_lanes = raw["lanes"]
    if not isinstance(raw_lanes, list):
        raise InvalidLaneRegistry("lanes must be a list")
    lanes: list[LaneDefinition] = []
    for index, item in enumerate(raw_lanes):
        if not isinstance(item, dict):
            raise InvalidLaneRegistry(f"lanes[{index}] must be an object")
        _reject_unknown_keys(item, _LANE_KEYS, f"lanes[{index}]")
        host_id = _host_id(item["host_id"], f"lanes[{index}].host_id")
        executor = _executor(item["executor"], f"lanes[{index}].executor")
        slot = _integer(item["slot"], f"lanes[{index}].slot")
        if slot < 1 or slot > 3:
            raise InvalidLaneRegistry("lane slot maximum 3 and minimum 1")
        lane_id = _string(item["id"], f"lanes[{index}].id")
        lanes.append(
            LaneDefinition(
                id=lane_id,
                host_id=host_id,
                executor=executor,
                slot=slot,
                model_patterns=_string_tuple(
                    item["model_patterns"], f"lanes[{index}].model_patterns"
                ),
            )
        )

    lane_ids = [lane.id for lane in lanes]
    if len(set(lane_ids)) != len(lane_ids):
        raise InvalidLaneRegistry("duplicate lane id")
    seats = [(lane.host_id, lane.executor, lane.slot) for lane in lanes]
    if len(set(seats)) != len(seats):
        raise InvalidLaneRegistry("duplicate lane seat")
    counts = Counter((lane.host_id, lane.executor) for lane in lanes)
    required_pairs = {
        ("pc", "claude"),
        ("pc", "codex"),
        ("mac", "claude"),
        ("mac", "codex"),
    }
    if set(counts) != required_pairs or any(count != 3 for count in counts.values()):
        raise InvalidLaneRegistry(
            "registry must declare exactly three seats per host and executor; maximum 3"
        )

    return raw, LaneRegistry(
        schema_version=schema_version,
        policy_version=policy_version,
        health_ttl_seconds=health_ttl_seconds,
        hosts=tuple(hosts),
        fallback=fallback,
        lanes=tuple(lanes),
        source_bytes=source_bytes,
    )


def load_lane_registry(path: Path | None = None) -> LaneRegistry:
    """Load and strictly validate the non-secret builder-lane policy."""

    try:
        if path is None:
            source_bytes = (
                resources.files("hermes_cli")
                .joinpath("data", "jobs-lanes.v1.json")
                .read_bytes()
            )
        else:
            source_bytes = Path(path).read_bytes()
    except OSError as exc:
        raise InvalidLaneRegistry(f"cannot load lane registry: {exc}") from exc
    _, registry = _parse_registry(source_bytes)
    return registry


def registry_digest(registry: LaneRegistry) -> str:
    """Return a SHA-256 digest of the registry's canonical JSON value."""

    if not isinstance(registry, LaneRegistry):
        raise TypeError("registry must be a LaneRegistry")
    raw, _ = _parse_registry(registry.source_bytes)
    canonical = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()
