"""Provider-free contracts for the versioned Jobs lane registry."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import replace

import pytest

from hermes_cli import jobs_lanes


def _registry_manifest() -> dict[str, object]:
    lanes = []
    for host_id in ("pc", "mac"):
        for executor, model_pattern in (
            ("claude", "claude-*"),
            ("codex", "gpt-*"),
        ):
            for slot in range(1, 4):
                lanes.append(
                    {
                        "id": f"{executor}-{host_id}-{slot}",
                        "host_id": host_id,
                        "executor": executor,
                        "slot": slot,
                        "model_patterns": [model_pattern],
                    }
                )
    return {
        "schema_version": 1,
        "policy_version": "jobs-lanes.v1",
        "health_ttl_seconds": 120,
        "hosts": [
            {
                "id": "pc",
                "preference": 0,
                "root_config_key": "jobs.lanes.roots.pc",
            },
            {
                "id": "mac",
                "preference": 1,
                "root_config_key": "jobs.lanes.roots.mac",
            },
        ],
        "fallback": {
            "name": "mac_when_pc_unavailable",
            "enabled": True,
            "preserve_executor": True,
            "preserve_model": True,
            "from_host": "pc",
            "to_host": "mac",
            "eligible_reason_codes": [
                "HOST_UNREACHABLE",
                "SSH_AUTH_FAILED",
                "AUTH_REQUIRED",
                "HEALTH_FAILED",
                "CAPACITY_FULL",
            ],
        },
        "lanes": lanes,
    }


def _write_registry(tmp_path, *, mutate=None):
    manifest = copy.deepcopy(_registry_manifest())
    if mutate is not None:
        mutate(manifest)
    path = tmp_path / "jobs-lanes.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_registry_declares_exactly_three_executor_seats_per_host():
    registry = jobs_lanes.load_lane_registry()

    counts = Counter((lane.host_id, lane.executor) for lane in registry.lanes)

    assert counts == {
        ("pc", "claude"): 3,
        ("pc", "codex"): 3,
        ("mac", "claude"): 3,
        ("mac", "codex"): 3,
    }
    assert len({lane.id for lane in registry.lanes}) == 12


def test_registry_rejects_a_fourth_seat(tmp_path):
    def add_fourth(manifest):
        manifest["lanes"].append(
            {
                "id": "claude-pc-4",
                "host_id": "pc",
                "executor": "claude",
                "slot": 4,
                "model_patterns": ["claude-*"],
            }
        )

    path = _write_registry(tmp_path, mutate=add_fourth)

    with pytest.raises(jobs_lanes.InvalidLaneRegistry, match="maximum 3"):
        jobs_lanes.load_lane_registry(path)


def test_registry_contains_no_absolute_paths_or_secret_fields():
    registry = jobs_lanes.load_lane_registry()
    raw = registry.source_bytes.decode("utf-8")

    assert "/Users/" not in raw
    assert "/home/" not in raw
    assert "token" not in raw.lower()
    assert "secret" not in raw.lower()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest.update({"unexpected": True}), "unknown registry field"),
        (
            lambda manifest: manifest["lanes"].append(
                copy.deepcopy(manifest["lanes"][0])
            ),
            "duplicate lane id",
        ),
        (
            lambda manifest: manifest["lanes"][1].update(
                {"host_id": "pc", "executor": "claude", "slot": 1}
            ),
            "duplicate lane seat",
        ),
        (
            lambda manifest: manifest["hosts"][0].update(
                {"root_config_key": "/Users/brandon/jobs/lanes"}
            ),
            "absolute path",
        ),
        (
            lambda manifest: manifest["fallback"].update(
                {"credential": "not-even-a-real-value"}
            ),
            "secret field",
        ),
    ],
)
def test_registry_rejects_invalid_or_sensitive_shapes(tmp_path, mutation, message):
    path = _write_registry(tmp_path, mutate=mutation)

    with pytest.raises(jobs_lanes.InvalidLaneRegistry, match=message):
        jobs_lanes.load_lane_registry(path)


def test_registry_digest_uses_canonical_json_not_file_whitespace(tmp_path):
    compact_path = _write_registry(tmp_path)
    pretty_path = tmp_path / "pretty.json"
    pretty_path.write_text(
        json.dumps(_registry_manifest(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    compact = jobs_lanes.load_lane_registry(compact_path)
    pretty = jobs_lanes.load_lane_registry(pretty_path)

    assert jobs_lanes.registry_digest(compact) == jobs_lanes.registry_digest(pretty)
    assert jobs_lanes.registry_digest(compact).startswith("sha256:")


def _passing_probe():
    return jobs_lanes.ProbeResult(passed=True, reason_code="OK")


def _passing_probes(now):
    return jobs_lanes.LaneProbes(
        observed_at=now,
        configuration=_passing_probe(),
        permissions=_passing_probe(),
        executable=jobs_lanes.ProbeResult(
            passed=True,
            reason_code="OK",
            safe_detail={"executor_version": "test-1.0"},
        ),
        auth=_passing_probe(),
        signer=_passing_probe(),
        resources=_passing_probe(),
        git=_passing_probe(),
        host=_passing_probe(),
        lease=jobs_lanes.LeaseObservation(state="IDLE", reason_code="OK"),
    )


def _replace_probe(probes, name, **changes):
    return replace(probes, **{name: replace(getattr(probes, name), **changes)})


def _replace_lease(probes, **changes):
    return replace(probes, lease=replace(probes.lease, **changes))


def _evaluate_health(tmp_path, *, mutation="all_pass", now=1_000):
    lane = jobs_lanes.load_lane_registry().lanes[0]
    lane_dir = tmp_path / lane.id
    lane_dir.mkdir()
    probes = _passing_probes(now)
    if mutation == "missing_directory":
        lane_dir.rmdir()
    elif mutation == "missing_auth":
        probes = _replace_probe(
            probes,
            "auth",
            passed=False,
            reason_code="AUTH_REQUIRED",
            failure_class="AUTH_INFRA",
        )
    elif mutation == "expired_auth":
        probes = _replace_probe(
            probes,
            "auth",
            passed=False,
            reason_code="TOKEN_EXPIRED",
            failure_class="AUTH_INFRA",
        )
    elif mutation == "ssh_denied":
        probes = _replace_probe(
            probes,
            "host",
            passed=False,
            reason_code="SSH_AUTH_FAILED",
            failure_class="AUTH_INFRA",
        )
    elif mutation == "host_unreachable":
        probes = _replace_probe(
            probes,
            "host",
            passed=False,
            reason_code="HOST_UNREACHABLE",
            failure_class="INFRA_FAILURE",
        )
    elif mutation == "signer_unprotected":
        probes = _replace_probe(
            probes,
            "signer",
            passed=False,
            reason_code="SIGNING_KEY_UNPROTECTED",
            failure_class="AUTH_INFRA",
        )
    elif mutation == "disk_low":
        probes = _replace_probe(
            probes,
            "resources",
            passed=False,
            reason_code="DISK_BUDGET_LOW",
            failure_class="INFRA_FAILURE",
        )
    elif mutation == "lease_active":
        probes = _replace_lease(
            probes, state="ASSIGNED", reason_code="ACTIVE_LEASE"
        )
    return jobs_lanes.evaluate_lane_health(
        lane,
        root=tmp_path,
        probes=probes,
        now=now,
        ttl_seconds=120,
    )


@pytest.mark.parametrize(
    ("mutation", "state", "failure_class", "reason"),
    [
        ("missing_directory", "UNPROVISIONED", "INFRA_FAILURE", "LANE_UNPROVISIONED"),
        ("missing_auth", "AUTH_REQUIRED", "AUTH_INFRA", "AUTH_REQUIRED"),
        ("expired_auth", "AUTH_REQUIRED", "AUTH_INFRA", "TOKEN_EXPIRED"),
        ("ssh_denied", "BLOCKED", "AUTH_INFRA", "SSH_AUTH_FAILED"),
        ("host_unreachable", "BLOCKED", "INFRA_FAILURE", "HOST_UNREACHABLE"),
        (
            "signer_unprotected",
            "BLOCKED",
            "AUTH_INFRA",
            "SIGNING_KEY_UNPROTECTED",
        ),
        ("disk_low", "BLOCKED", "INFRA_FAILURE", "DISK_BUDGET_LOW"),
        ("lease_active", "ASSIGNED", None, "ACTIVE_LEASE"),
        ("all_pass", "IDLE", None, "OK"),
    ],
)
def test_health_reducer_never_overstates_readiness(
    tmp_path, mutation, state, failure_class, reason
):
    result = _evaluate_health(tmp_path, mutation=mutation)

    assert (result.state, result.failure_class, result.reason_code) == (
        state,
        failure_class,
        reason,
    )
    assert result.available_capacity == (1 if state == "IDLE" else 0)


def test_failed_earlier_probe_cannot_be_overwritten_by_later_pass(tmp_path):
    lane = jobs_lanes.load_lane_registry().lanes[0]
    (tmp_path / lane.id).mkdir()
    probes = _replace_probe(
        _passing_probes(1_000),
        "auth",
        passed=False,
        reason_code="AUTH_REQUIRED",
        failure_class="AUTH_INFRA",
    )

    result = jobs_lanes.evaluate_lane_health(
        lane, root=tmp_path, probes=probes, now=1_000, ttl_seconds=120
    )

    assert (result.state, result.reason_code) == ("AUTH_REQUIRED", "AUTH_REQUIRED")


def test_stale_health_observation_is_blocked(tmp_path):
    lane = jobs_lanes.load_lane_registry().lanes[0]
    (tmp_path / lane.id).mkdir()

    result = jobs_lanes.evaluate_lane_health(
        lane,
        root=tmp_path,
        probes=_passing_probes(1_000),
        now=1_121,
        ttl_seconds=120,
    )

    assert (result.state, result.status, result.reason_code) == (
        "BLOCKED",
        "BLOCKED",
        "HEALTH_STALE",
    )


@pytest.mark.parametrize(
    ("lease_state", "reason_code", "expected_state", "expected_failure"),
    [
        ("BUILDING", "ACTIVE_BUILD", "BUILDING", None),
        ("VERIFYING", "VERIFYING_EVIDENCE", "VERIFYING", None),
        ("FAILED", "CLEANUP_FAILED", "FAILED", "INFRA_FAILURE"),
    ],
)
def test_lane_custody_state_controls_capacity(
    tmp_path, lease_state, reason_code, expected_state, expected_failure
):
    lane = jobs_lanes.load_lane_registry().lanes[0]
    (tmp_path / lane.id).mkdir()
    probes = _replace_lease(
        _passing_probes(1_000), state=lease_state, reason_code=reason_code
    )

    result = jobs_lanes.evaluate_lane_health(
        lane, root=tmp_path, probes=probes, now=1_000, ttl_seconds=120
    )

    assert (result.state, result.failure_class, result.available_capacity) == (
        expected_state,
        expected_failure,
        0,
    )


def test_ssh_denial_wins_even_when_network_reachability_passes(tmp_path):
    lane = jobs_lanes.load_lane_registry().lanes[0]
    (tmp_path / lane.id).mkdir()
    probes = _replace_probe(
        _passing_probes(1_000),
        "host",
        passed=False,
        reason_code="SSH_AUTH_FAILED",
        failure_class="AUTH_INFRA",
        safe_detail={"network_reachable": True},
    )

    result = jobs_lanes.evaluate_lane_health(
        lane, root=tmp_path, probes=probes, now=1_000, ttl_seconds=120
    )

    assert (result.state, result.failure_class, result.reason_code) == (
        "BLOCKED",
        "AUTH_INFRA",
        "SSH_AUTH_FAILED",
    )


ROUTING_NOW = 2_000


def _idle_health(lane):
    return jobs_lanes.LaneHealth(
        lane_id=lane.id,
        state="IDLE",
        status="PASS",
        failure_class=None,
        reason_code="OK",
        observed_at=ROUTING_NOW,
        expires_at=ROUTING_NOW + 120,
        executor_version="test-1.0",
        available_capacity=1,
        safe_detail={},
    )


def _routing_health(registry):
    return [_idle_health(lane) for lane in registry.lanes]


def _block_host(health, registry, host_id, reason_code):
    lane_by_id = {lane.id: lane for lane in registry.lanes}
    failure_class = "AUTH_INFRA" if reason_code in {
        "SSH_AUTH_FAILED",
        "AUTH_REQUIRED",
    } else "INFRA_FAILURE"
    state = "AUTH_REQUIRED" if reason_code == "AUTH_REQUIRED" else "BLOCKED"
    result = []
    for item in health:
        if lane_by_id[item.lane_id].host_id == host_id:
            result.append(
                replace(
                    item,
                    state=state,
                    status="BLOCKED",
                    failure_class=failure_class,
                    reason_code=reason_code,
                    available_capacity=0,
                )
            )
        else:
            result.append(item)
    return result


def _routing_request(executor="claude", model="claude-opus-5"):
    return jobs_lanes.RoutingRequest(
        job_id="j_route", executor=executor, model=model
    )


def test_healthy_matching_pc_lane_wins():
    registry = jobs_lanes.load_lane_registry()

    decision = jobs_lanes.select_lane(
        registry,
        _routing_health(registry),
        _routing_request(),
        active_load={},
        now=ROUTING_NOW,
    )

    assert decision.lane_id == "claude-pc-1"
    assert decision.host_id == "pc"
    assert decision.fallback_applied is False


@pytest.mark.parametrize(
    "pc_reason",
    [
        "HOST_UNREACHABLE",
        "SSH_AUTH_FAILED",
        "AUTH_REQUIRED",
        "HEALTH_FAILED",
        "CAPACITY_FULL",
    ],
)
def test_policy_allows_matching_mac_when_pc_is_unavailable(pc_reason):
    registry = jobs_lanes.load_lane_registry()
    health = _block_host(_routing_health(registry), registry, "pc", pc_reason)

    decision = jobs_lanes.select_lane(
        registry,
        health,
        _routing_request(),
        active_load={},
        now=ROUTING_NOW,
    )

    assert decision.lane_id == "claude-mac-1"
    assert decision.host_id == "mac"
    assert decision.fallback_applied is True
    assert decision.policy_version == "jobs-lanes.v1"
    assert (decision.executor, decision.model) == ("claude", "claude-opus-5")


def test_fallback_never_changes_executor_or_model(tmp_path):
    def remove_matching_mac(manifest):
        for lane in manifest["lanes"]:
            if lane["host_id"] == "mac" and lane["executor"] == "claude":
                lane["model_patterns"] = ["claude-haiku-*"]

    registry = jobs_lanes.load_lane_registry(
        _write_registry(tmp_path, mutate=remove_matching_mac)
    )
    health = _block_host(_routing_health(registry), registry, "pc", "SSH_AUTH_FAILED")

    decision = jobs_lanes.select_lane(
        registry,
        health,
        _routing_request(),
        active_load={},
        now=ROUTING_NOW,
    )

    assert (decision.status, decision.reason_code) == (
        "BLOCKED",
        "NO_MATCHING_MAC_LANE",
    )
    assert decision.lane_id is None
    assert (decision.executor, decision.model) == ("claude", "claude-opus-5")


def test_three_occupied_pc_seats_force_policy_authorized_mac_fallback():
    registry = jobs_lanes.load_lane_registry()
    active_load = {
        f"claude-pc-{slot}": 1
        for slot in range(1, 4)
    }

    decision = jobs_lanes.select_lane(
        registry,
        _routing_health(registry),
        _routing_request(),
        active_load=active_load,
        now=ROUTING_NOW,
    )

    assert decision.lane_id == "claude-mac-1"
    assert decision.fallback_applied is True
    assert decision.reason_code == "MAC_FALLBACK_SELECTED"


def test_all_matching_seats_busy_keeps_job_queued():
    registry = jobs_lanes.load_lane_registry()
    active_load = {
        lane.id: 1
        for lane in registry.lanes
        if lane.executor == "claude"
    }

    decision = jobs_lanes.select_lane(
        registry,
        _routing_health(registry),
        _routing_request(),
        active_load=active_load,
        now=ROUTING_NOW,
    )

    assert (decision.status, decision.reason_code, decision.lane_id) == (
        "QUEUED",
        "CAPACITY_FULL",
        None,
    )


def test_tie_chooses_lowest_active_load_then_lane_id():
    registry = jobs_lanes.load_lane_registry()
    active_load = {"codex-pc-1": 1, "codex-pc-2": 0, "codex-pc-3": 0}

    decision = jobs_lanes.select_lane(
        registry,
        _routing_health(registry),
        _routing_request(executor="codex", model="gpt-5.6-sol"),
        active_load=active_load,
        now=ROUTING_NOW,
    )

    assert decision.lane_id == "codex-pc-2"


def test_expired_health_cannot_be_selected():
    registry = jobs_lanes.load_lane_registry()
    health = [replace(item, expires_at=ROUTING_NOW - 1) for item in _routing_health(registry)]

    decision = jobs_lanes.select_lane(
        registry,
        health,
        _routing_request(),
        active_load={},
        now=ROUTING_NOW,
    )

    assert decision.status == "BLOCKED"
    assert decision.lane_id is None
    assert decision.reason_code == "NO_HEALTHY_MAC_LANE"


def test_duplicate_health_observation_fails_closed():
    registry = jobs_lanes.load_lane_registry()
    health = _routing_health(registry)
    health.append(health[0])

    with pytest.raises(ValueError, match="duplicate lane health"):
        jobs_lanes.select_lane(
            registry,
            health,
            _routing_request(),
            active_load={},
            now=ROUTING_NOW,
        )
