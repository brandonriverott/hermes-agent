"""Provider-free contracts for the versioned Jobs lane registry."""

from __future__ import annotations

import copy
import json
from collections import Counter

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
