"""Versioned policy, health, and routing for isolated Jobs builder lanes."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from importlib import resources
from pathlib import Path, PureWindowsPath
from typing import Literal, Mapping, Sequence, TypeAlias


HostId = Literal["pc", "mac"]
Executor = Literal["claude", "codex"]
JsonScalar: TypeAlias = str | int | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
LaneState = Literal[
    "UNPROVISIONED",
    "AUTH_REQUIRED",
    "BLOCKED",
    "IDLE",
    "ASSIGNED",
    "BUILDING",
    "VERIFYING",
    "FAILED",
]


class InvalidLaneRegistry(ValueError):
    """Raised when a lane registry violates its strict schema or policy."""


class InvalidLaneDecision(ValueError):
    """Raised when a supplied routing decision is not bound to its registry."""


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


@dataclass(frozen=True)
class ProbeResult:
    """One sanitized result from an immutable lane-health snapshot."""

    passed: bool
    reason_code: str
    failure_class: str | None = None
    safe_detail: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class LeaseObservation:
    """Current attempt custody for one lane."""

    state: Literal["IDLE", "ASSIGNED", "BUILDING", "VERIFYING", "FAILED"]
    reason_code: str


@dataclass(frozen=True)
class LaneProbes:
    """All checks collected against one observation timestamp."""

    observed_at: int
    configuration: ProbeResult
    permissions: ProbeResult
    executable: ProbeResult
    auth: ProbeResult
    signer: ProbeResult
    resources: ProbeResult
    git: ProbeResult
    host: ProbeResult
    lease: LeaseObservation


@dataclass(frozen=True)
class LaneHealth:
    lane_id: str
    state: LaneState
    status: Literal["PASS", "BLOCKED"]
    failure_class: str | None
    reason_code: str
    observed_at: int
    expires_at: int
    executor_version: str | None
    available_capacity: int
    safe_detail: Mapping[str, JsonValue]


@dataclass(frozen=True)
class RoutingRequest:
    job_id: str
    executor: Executor
    model: str


@dataclass(frozen=True)
class LaneDecision:
    status: Literal["SELECTED", "QUEUED", "BLOCKED"]
    lane_id: str | None
    host_id: str | None
    executor: str
    model: str
    policy_version: str
    policy_digest: str
    fallback_applied: bool
    reason_code: str
    considered_lane_ids: tuple[str, ...]


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


def validate_lane_decision(
    registry: LaneRegistry, decision: LaneDecision
) -> LaneDefinition:
    """Bind one selected decision to the immutable policy that produced it."""

    if not isinstance(registry, LaneRegistry):
        raise TypeError("registry must be a LaneRegistry")
    if not isinstance(decision, LaneDecision):
        raise InvalidLaneDecision("lane decision has an invalid contract")
    if decision.status != "SELECTED" or decision.lane_id is None:
        raise InvalidLaneDecision("lane decision is not selected")
    if (
        decision.policy_version != registry.policy_version
        or decision.policy_digest != registry_digest(registry)
    ):
        raise InvalidLaneDecision("lane decision policy is not current")
    lane = next((item for item in registry.lanes if item.id == decision.lane_id), None)
    if lane is None:
        raise InvalidLaneDecision("lane decision references an unknown seat")
    if (
        decision.host_id != lane.host_id
        or decision.executor != lane.executor
        or not _model_matches(lane, decision.model)
    ):
        raise InvalidLaneDecision("lane decision contradicts its registered seat")

    host_preference = {host.id: host.preference for host in registry.hosts}
    matching = sorted(
        (
            item
            for item in registry.lanes
            if item.executor == decision.executor
            and _model_matches(item, decision.model)
        ),
        key=lambda item: (host_preference[item.host_id], item.id),
    )
    if decision.considered_lane_ids != tuple(item.id for item in matching):
        raise InvalidLaneDecision("lane decision candidate set is not canonical")
    if lane.host_id == "pc":
        expected = (False, "PC_LANE_SELECTED")
    else:
        expected = (True, "MAC_FALLBACK_SELECTED")
    if (decision.fallback_applied, decision.reason_code) != expected:
        raise InvalidLaneDecision("lane decision selection reason is inconsistent")
    return lane


def _sanitize_safe_value(value: object, *, depth: int = 0) -> JsonValue | None:
    if depth > 4:
        return None
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return None
    if isinstance(value, str):
        cleaned = "".join(character for character in value if character.isprintable())
        return cleaned[:256]
    if isinstance(value, list):
        result: list[JsonValue] = []
        for item in value[:32]:
            safe_item = _sanitize_safe_value(item, depth=depth + 1)
            if safe_item is not None:
                result.append(safe_item)
        return result
    if isinstance(value, Mapping):
        result_dict: dict[str, JsonValue] = {}
        for key, item in list(value.items())[:32]:
            if not isinstance(key, str) or _SECRET_FIELD.search(key):
                continue
            safe_item = _sanitize_safe_value(item, depth=depth + 1)
            if safe_item is not None:
                result_dict[key] = safe_item
        return result_dict
    return None


def _safe_probe_details(
    lane_dir: Path, probes: LaneProbes
) -> tuple[dict[str, JsonValue], str | None]:
    details: dict[str, JsonValue] = {"lane_dir": str(lane_dir)}
    executor_version: str | None = None
    for name in (
        "configuration",
        "permissions",
        "executable",
        "auth",
        "signer",
        "resources",
        "git",
        "host",
    ):
        probe = getattr(probes, name)
        safe = _sanitize_safe_value(probe.safe_detail)
        if isinstance(safe, dict) and safe:
            details[name] = safe
        if name == "executable" and isinstance(safe, dict):
            version = safe.get("executor_version")
            if isinstance(version, str) and version:
                executor_version = version
    return details, executor_version


def _lane_health(
    lane: LaneDefinition,
    probes: LaneProbes,
    *,
    ttl_seconds: int,
    state: LaneState,
    status: Literal["PASS", "BLOCKED"],
    failure_class: str | None,
    reason_code: str,
    lane_dir: Path,
) -> LaneHealth:
    safe_detail, executor_version = _safe_probe_details(lane_dir, probes)
    return LaneHealth(
        lane_id=lane.id,
        state=state,
        status=status,
        failure_class=failure_class,
        reason_code=reason_code,
        observed_at=probes.observed_at,
        expires_at=probes.observed_at + ttl_seconds,
        executor_version=executor_version,
        available_capacity=1 if state == "IDLE" and status == "PASS" else 0,
        safe_detail=safe_detail,
    )


def evaluate_lane_health(
    lane: LaneDefinition,
    *,
    root: Path,
    probes: LaneProbes,
    now: int,
    ttl_seconds: int,
) -> LaneHealth:
    """Reduce ordered probe evidence without allowing later checks to mask failure."""

    if not isinstance(lane, LaneDefinition):
        raise TypeError("lane must be a LaneDefinition")
    if isinstance(now, bool) or not isinstance(now, int):
        raise TypeError("now must be an integer Unix timestamp")
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
        raise TypeError("ttl_seconds must be an integer")
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    if isinstance(probes.observed_at, bool) or not isinstance(probes.observed_at, int):
        raise TypeError("probes.observed_at must be an integer Unix timestamp")

    lane_dir = Path(root) / lane.id
    if not lane_dir.is_dir() or lane_dir.is_symlink():
        return _lane_health(
            lane,
            probes,
            ttl_seconds=ttl_seconds,
            state="UNPROVISIONED",
            status="BLOCKED",
            failure_class="INFRA_FAILURE",
            reason_code="LANE_UNPROVISIONED",
            lane_dir=lane_dir,
        )
    if probes.observed_at > now:
        return _lane_health(
            lane,
            probes,
            ttl_seconds=ttl_seconds,
            state="BLOCKED",
            status="BLOCKED",
            failure_class="INFRA_FAILURE",
            reason_code="HEALTH_CLOCK_SKEW",
            lane_dir=lane_dir,
        )
    if now > probes.observed_at + ttl_seconds:
        return _lane_health(
            lane,
            probes,
            ttl_seconds=ttl_seconds,
            state="BLOCKED",
            status="BLOCKED",
            failure_class="INFRA_FAILURE",
            reason_code="HEALTH_STALE",
            lane_dir=lane_dir,
        )

    ordered_probes = (
        ("configuration", probes.configuration),
        ("permissions", probes.permissions),
        ("executable", probes.executable),
        ("auth", probes.auth),
        ("signer", probes.signer),
        ("resources", probes.resources),
        ("git", probes.git),
        ("host", probes.host),
    )
    for name, probe in ordered_probes:
        if probe.passed:
            continue
        is_auth_required = name == "auth" and probe.reason_code in {
            "AUTH_REQUIRED",
            "TOKEN_EXPIRED",
        }
        default_failure = (
            "AUTH_INFRA"
            if name in {"auth", "signer"}
            or probe.reason_code == "SSH_AUTH_FAILED"
            else "INFRA_FAILURE"
        )
        return _lane_health(
            lane,
            probes,
            ttl_seconds=ttl_seconds,
            state="AUTH_REQUIRED" if is_auth_required else "BLOCKED",
            status="BLOCKED",
            failure_class=probe.failure_class or default_failure,
            reason_code=probe.reason_code,
            lane_dir=lane_dir,
        )

    lease = probes.lease
    if lease.state == "FAILED":
        return _lane_health(
            lane,
            probes,
            ttl_seconds=ttl_seconds,
            state="FAILED",
            status="BLOCKED",
            failure_class="INFRA_FAILURE",
            reason_code=lease.reason_code,
            lane_dir=lane_dir,
        )
    if lease.state in {"ASSIGNED", "BUILDING", "VERIFYING"}:
        return _lane_health(
            lane,
            probes,
            ttl_seconds=ttl_seconds,
            state=lease.state,
            status="PASS",
            failure_class=None,
            reason_code=lease.reason_code,
            lane_dir=lane_dir,
        )
    if lease.state != "IDLE":
        raise ValueError(f"unsupported lease state: {lease.state}")
    return _lane_health(
        lane,
        probes,
        ttl_seconds=ttl_seconds,
        state="IDLE",
        status="PASS",
        failure_class=None,
        reason_code=lease.reason_code,
        lane_dir=lane_dir,
    )


def _model_matches(lane: LaneDefinition, model: str) -> bool:
    return any(fnmatchcase(model, pattern) for pattern in lane.model_patterns)


def _unavailable_reason(
    health: LaneHealth | None, *, active_load: int, now: int
) -> str:
    if health is None:
        return "HEALTH_FAILED"
    if health.observed_at > now or now > health.expires_at:
        return "HEALTH_FAILED"
    if active_load > 0:
        return "CAPACITY_FULL"
    if health.status == "PASS" and health.state == "IDLE":
        return "READY" if health.available_capacity > 0 else "CAPACITY_FULL"
    if health.state in {"ASSIGNED", "BUILDING", "VERIFYING"}:
        return "CAPACITY_FULL"
    if health.state == "AUTH_REQUIRED":
        return "AUTH_REQUIRED"
    if health.reason_code in {
        "HOST_UNREACHABLE",
        "SSH_AUTH_FAILED",
        "AUTH_REQUIRED",
        "HEALTH_FAILED",
        "CAPACITY_FULL",
    }:
        return health.reason_code
    return "HEALTH_FAILED"


def _decision(
    registry: LaneRegistry,
    request: RoutingRequest,
    considered: tuple[str, ...],
    *,
    status: Literal["SELECTED", "QUEUED", "BLOCKED"],
    reason_code: str,
    lane: LaneDefinition | None = None,
    fallback_applied: bool = False,
) -> LaneDecision:
    return LaneDecision(
        status=status,
        lane_id=lane.id if lane is not None else None,
        host_id=lane.host_id if lane is not None else None,
        executor=request.executor,
        model=request.model,
        policy_version=registry.policy_version,
        policy_digest=registry_digest(registry),
        fallback_applied=fallback_applied,
        reason_code=reason_code,
        considered_lane_ids=considered,
    )


def select_lane(
    registry: LaneRegistry,
    health: Sequence[LaneHealth],
    request: RoutingRequest,
    *,
    active_load: Mapping[str, int],
    now: int | None = None,
) -> LaneDecision:
    """Select one healthy seat with deterministic, policy-bound host fallback."""

    if not isinstance(registry, LaneRegistry):
        raise TypeError("registry must be a LaneRegistry")
    if not isinstance(request, RoutingRequest):
        raise TypeError("request must be a RoutingRequest")
    if not request.job_id.strip():
        raise ValueError("routing request job_id must be non-empty")
    if request.executor not in _EXECUTORS:
        raise ValueError("routing request executor is invalid")
    if not request.model.strip():
        raise ValueError("routing request model must be non-empty")
    now_value = int(time.time()) if now is None else now
    if isinstance(now_value, bool) or not isinstance(now_value, int):
        raise TypeError("now must be an integer Unix timestamp")

    lane_by_id = {lane.id: lane for lane in registry.lanes}
    health_by_id: dict[str, LaneHealth] = {}
    for item in health:
        if item.lane_id not in lane_by_id:
            raise ValueError(f"health references unknown lane: {item.lane_id}")
        if item.lane_id in health_by_id:
            raise ValueError(f"duplicate lane health: {item.lane_id}")
        health_by_id[item.lane_id] = item
    for lane_id, load in active_load.items():
        if lane_id not in lane_by_id:
            raise ValueError(f"active load references unknown lane: {lane_id}")
        if isinstance(load, bool) or not isinstance(load, int) or load < 0:
            raise ValueError(f"active load must be a non-negative integer: {lane_id}")

    matching = [
        lane
        for lane in registry.lanes
        if lane.executor == request.executor and _model_matches(lane, request.model)
    ]
    host_preference = {host.id: host.preference for host in registry.hosts}
    matching.sort(key=lambda lane: (host_preference[lane.host_id], lane.id))
    considered = tuple(lane.id for lane in matching)
    if not matching:
        return _decision(
            registry,
            request,
            considered,
            status="BLOCKED",
            reason_code="NO_MATCHING_LANE",
        )

    def reason(lane: LaneDefinition) -> str:
        return _unavailable_reason(
            health_by_id.get(lane.id),
            active_load=active_load.get(lane.id, 0),
            now=now_value,
        )

    def ready(lanes: list[LaneDefinition]) -> list[LaneDefinition]:
        candidates = [lane for lane in lanes if reason(lane) == "READY"]
        candidates.sort(key=lambda lane: (active_load.get(lane.id, 0), lane.id))
        return candidates

    pc_lanes = [lane for lane in matching if lane.host_id == "pc"]
    mac_lanes = [lane for lane in matching if lane.host_id == "mac"]
    ready_pc = ready(pc_lanes)
    if ready_pc:
        return _decision(
            registry,
            request,
            considered,
            status="SELECTED",
            reason_code="PC_LANE_SELECTED",
            lane=ready_pc[0],
        )

    fallback = registry.fallback
    pc_reasons = tuple(reason(lane) for lane in pc_lanes)
    fallback_authorized = (
        bool(pc_lanes)
        and fallback.enabled
        and fallback.preserve_executor
        and fallback.preserve_model
        and fallback.from_host == "pc"
        and fallback.to_host == "mac"
        and all(code in fallback.eligible_reason_codes for code in pc_reasons)
    )
    if not fallback_authorized:
        return _decision(
            registry,
            request,
            considered,
            status="BLOCKED",
            reason_code="PC_FALLBACK_NOT_AUTHORIZED",
        )
    if not mac_lanes:
        return _decision(
            registry,
            request,
            considered,
            status="BLOCKED",
            reason_code="NO_MATCHING_MAC_LANE",
        )

    ready_mac = ready(mac_lanes)
    if ready_mac:
        return _decision(
            registry,
            request,
            considered,
            status="SELECTED",
            reason_code="MAC_FALLBACK_SELECTED",
            lane=ready_mac[0],
            fallback_applied=True,
        )
    mac_reasons = tuple(reason(lane) for lane in mac_lanes)
    if "CAPACITY_FULL" in mac_reasons:
        return _decision(
            registry,
            request,
            considered,
            status="QUEUED",
            reason_code="CAPACITY_FULL",
        )
    return _decision(
        registry,
        request,
        considered,
        status="BLOCKED",
        reason_code="NO_HEALTHY_MAC_LANE",
    )
