"""Behavior tests for dry-run-first builder-lane provisioning."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "jobs_provision_lanes.py"


def _provision(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--host",
            "mac",
            "--root",
            str(root),
            *arguments,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_writes_nothing(tmp_path):
    root = tmp_path / "lanes"
    root.mkdir()
    result = _provision(root, "--dry-run")

    assert result.returncode == 0
    assert list(root.iterdir()) == []
    payload = json.loads(result.stdout)
    assert payload["mode"] == "provision"
    assert len(payload["planned_lanes"]) == 6
    assert payload["changed"] == payload["planned_lanes"]


def test_apply_creates_non_secret_layout_with_private_auth(tmp_path):
    root = tmp_path / "lanes"
    root.mkdir()
    result = _provision(root, "--apply")

    assert result.returncode == 0
    lane = root / "claude-mac-1"
    assert (lane / "lane-config.json").is_file()
    assert set(path.name for path in lane.iterdir()) == {
        "lane-config.json",
        "auth",
        "worktrees",
        "handoffs",
        "receipts",
        "health",
    }
    if os.name == "posix":
        assert stat.S_IMODE(lane.stat().st_mode) == 0o700
        assert stat.S_IMODE((lane / "auth").stat().st_mode) == 0o700
        assert stat.S_IMODE((lane / "lane-config.json").stat().st_mode) == 0o600
        assert stat.S_IMODE(
            (lane / "auth" / "receipt-signing-key.pem").stat().st_mode
        ) == 0o600
    assert not list(root.rglob("*token*"))
    assert "PRIVATE KEY" not in result.stdout


def test_each_lane_gets_a_distinct_signing_key(tmp_path):
    root = tmp_path / "lanes"
    root.mkdir()
    result = _provision(root, "--apply")
    assert result.returncode == 0

    keys = [
        (lane / "auth" / "receipt-signing-key.pem").read_bytes()
        for lane in sorted(root.iterdir())
    ]
    public_keys = [
        json.loads((lane / "lane-config.json").read_text(encoding="utf-8"))[
            "public_key"
        ]
        for lane in sorted(root.iterdir())
    ]

    assert len(keys) == len(set(keys)) == 6
    assert len(public_keys) == len(set(public_keys)) == 6


def test_second_apply_is_idempotent_and_preserves_keys(tmp_path):
    root = tmp_path / "lanes"
    root.mkdir()
    first = _provision(root, "--apply")
    key_path = root / "codex-mac-1" / "auth" / "receipt-signing-key.pem"
    key_digest = hashlib.sha256(key_path.read_bytes()).hexdigest()

    second = _provision(root, "--apply")

    assert first.returncode == second.returncode == 0
    assert json.loads(second.stdout)["changed"] == []
    assert hashlib.sha256(key_path.read_bytes()).hexdigest() == key_digest


def test_path_collision_blocks_all_sibling_writes(tmp_path):
    root = tmp_path / "lanes"
    root.mkdir()
    collision = root / "claude-mac-2"
    collision.write_text("occupied", encoding="utf-8")

    result = _provision(root, "--apply")

    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "BLOCKED"
    assert payload["reason_code"] == "PATH_COLLISION"
    assert not (root / "claude-mac-1").exists()
    assert collision.read_text(encoding="utf-8") == "occupied"


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
def test_symlinked_lane_path_is_a_collision(tmp_path):
    root = tmp_path / "lanes"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "codex-mac-3").symlink_to(outside, target_is_directory=True)

    result = _provision(root, "--apply")

    assert result.returncode == 2
    assert json.loads(result.stdout)["reason_code"] == "PATH_COLLISION"
    assert list(outside.iterdir()) == []
    assert not (root / "claude-mac-1").exists()


def test_rollback_dry_run_lists_exact_lane_directories_without_removing(tmp_path):
    root = tmp_path / "lanes"
    root.mkdir()
    assert _provision(root, "--apply").returncode == 0

    result = _provision(root, "--mode", "rollback", "--dry-run")

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "rollback"
    assert len(payload["planned_lanes"]) == 6
    assert all(Path(path).parent == root for path in payload["planned_lanes"])
    assert all(Path(path).is_dir() for path in payload["planned_lanes"])


@pytest.mark.parametrize("unsafe_root", ["/", "$HOME/jobs", "lanes/*"])
def test_unsafe_or_unresolved_root_is_rejected(tmp_path, unsafe_root):
    result = _provision(Path(unsafe_root), "--dry-run")

    assert result.returncode == 1
    assert "unsafe lane root" in result.stderr
