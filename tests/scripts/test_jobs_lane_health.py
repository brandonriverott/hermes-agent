"""Behavior tests for the sanitized builder-lane health command."""

from __future__ import annotations

import base64
import builtins
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_lanes, jobs_receipts


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "jobs_lane_health.py"


def _health_module():
    spec = importlib.util.spec_from_file_location("jobs_lane_health_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_linux_memory_fallback_keeps_health_check_dependency_free(tmp_path, monkeypatch):
    module = _health_module()
    monkeypatch.setattr(module.sys, "platform", "linux")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 99 kB\nMemAvailable: 1048576 kB\n", encoding="utf-8")
    assert module._linux_mem_available(meminfo) == 1024 * 1024 * 1024

    original_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    lane = tmp_path / "lane"
    lane.mkdir()
    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    monkeypatch.setattr(module, "_linux_mem_available", lambda _path: 1024 * 1024 * 1024)
    result = module._probe_resources(lane)
    assert result.passed is True
    assert result.reason_code == "OK"


def test_macos_memory_fallback_keeps_health_check_dependency_free(
    tmp_path, monkeypatch,
):
    module = _health_module()
    monkeypatch.setattr(module.sys, "platform", "darwin")
    vm_stat = subprocess.CompletedProcess(
        args=["vm_stat"],
        returncode=0,
        stdout=(
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free: 1000.\n"
            "Pages inactive: 64000.\n"
            "Pages speculative: 1000.\n"
            "Pages purgeable: 0.\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(module.shutil, "which", lambda name, **_kw: f"/usr/bin/{name}")
    monkeypatch.setattr(module, "_run_private", lambda *_a, **_kw: vm_stat)
    assert module._macos_mem_available() == 66_000 * 16_384

    original_import = builtins.__import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    lane = tmp_path / "lane"
    lane.mkdir()
    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    result = module._probe_resources(lane)
    assert result.passed is True
    assert result.reason_code == "OK"


def test_git_probe_does_not_require_immutable_runtime_to_be_a_git_checkout(
    tmp_path, monkeypatch
):
    module = _health_module()
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    result = module._probe_git({"PATH": os.environ["PATH"]})

    assert result.passed is True
    assert result.reason_code == "OK"


def _provision_test_lane(root: Path, lane_id: str) -> Path:
    lane = next(
        item for item in jobs_lanes.load_lane_registry().lanes if item.id == lane_id
    )
    lane_dir = root / lane.id
    lane_dir.mkdir(mode=0o700)
    for name in ("auth", "worktrees", "handoffs", "receipts", "health"):
        (lane_dir / name).mkdir(mode=0o700)
    private_key = Ed25519PrivateKey.generate()
    key_path = lane_dir / "auth" / "receipt-signing-key.pem"
    key_path.write_bytes(jobs_receipts.private_key_pem(private_key))
    key_path.chmod(0o600)
    auth_hint = ".credentials.json" if lane.executor == "claude" else "auth.json"
    auth_hint_path = lane_dir / "auth" / auth_hint
    auth_hint_path.write_text("{}", encoding="utf-8")
    auth_hint_path.chmod(0o600)
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    config = {
        "schema_version": 1,
        "policy_version": "jobs-lanes.v1",
        "policy_digest": jobs_lanes.registry_digest(jobs_lanes.load_lane_registry()),
        "lane_id": lane.id,
        "host_id": lane.host_id,
        "executor": lane.executor,
        "slot": lane.slot,
        "model_patterns": list(lane.model_patterns),
        "key_id": f"lane:{lane.id}:v1",
        "public_key": "base64:" + base64.b64encode(public_key).decode("ascii"),
        "directories": {
            "auth": "auth",
            "worktrees": "worktrees",
            "handoffs": "handoffs",
            "receipts": "receipts",
            "health": "health",
        },
    }
    config_path = lane_dir / "lane-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_path.chmod(0o600)
    return lane_dir


def _write_fake_executor(
    bin_dir: Path, executor: str, expected_auth: Path, *, authenticated: bool = False
) -> None:
    if executor == "claude":
        auth_variable = "CLAUDE_CONFIG_DIR"
        auth_command = '"$1" = "auth" ] && [ "$2" = "status"'
        blocked_output = '{"loggedIn":false,"detail":"canary-secret-value"}'
        pass_output = '{"loggedIn":true}'
    else:
        auth_variable = "CODEX_HOME"
        auth_command = '"$1" = "login" ] && [ "$2" = "status"'
        blocked_output = "Not logged in: canary-secret-value"
        pass_output = "Logged in"
    expected_output = pass_output if authenticated else blocked_output
    expected_exit = 0 if authenticated else 1
    script = f"""#!/bin/sh
if [ \"$1\" = \"--version\" ]; then
  echo \"{executor} test-1.0\"
  exit 0
fi
if [ {auth_command} ]; then
  if [ \"${{{auth_variable}}}\" = \"{expected_auth}\" ]; then
    echo '{expected_output}'
    exit {expected_exit}
  fi
  echo '{pass_output}'
  exit 0
fi
exit 1
"""
    path = bin_dir / executor
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.mark.parametrize(
    ("lane_id", "executor_home_variable"),
    [
        ("claude-mac-1", "CLAUDE_CONFIG_DIR"),
        ("codex-mac-1", "CODEX_HOME"),
    ],
)
def test_health_cli_isolates_auth_and_redacts_private_output(
    tmp_path, lane_id, executor_home_variable
):
    root = tmp_path / "lanes"
    root.mkdir()
    lane_dir = _provision_test_lane(root, lane_id)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executor = lane_id.split("-", 1)[0]
    _write_fake_executor(bin_dir, executor, lane_dir / "auth")
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(bin_dir), env.get("PATH", "")))
    env[executor_home_variable] = "must-not-be-trusted"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--host",
            "mac",
            "--root",
            str(root),
            "--lane",
            lane_id,
            "--json",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "canary-secret-value" not in completed.stdout
    assert "canary-secret-value" not in completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["lanes"][0]["state"] == "AUTH_REQUIRED"
    assert payload["lanes"][0]["reason_code"] == "AUTH_REQUIRED"


def test_health_cli_returns_zero_only_for_a_fresh_idle_lane(tmp_path):
    host = "mac" if sys.platform == "darwin" else "pc"
    lane_id = f"claude-{host}-1"
    root = tmp_path / "lanes"
    root.mkdir()
    lane_dir = _provision_test_lane(root, lane_id)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_executor(
        bin_dir, "claude", lane_dir / "auth", authenticated=True
    )
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(bin_dir), env.get("PATH", "")))

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--host",
            host,
            "--root",
            str(root),
            "--lane",
            lane_id,
            "--json",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["lanes"][0]["status"] == "PASS"
    assert payload["lanes"][0]["state"] == "IDLE"
    assert payload["lanes"][0]["available_capacity"] == 1


def test_health_cli_preserves_user_for_macos_credential_lookup(tmp_path):
    host = "mac" if sys.platform == "darwin" else "pc"
    lane_id = f"claude-{host}-1"
    root = tmp_path / "lanes"
    root.mkdir()
    lane_dir = _provision_test_lane(root, lane_id)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executor = bin_dir / "claude"
    executor.write_text(
        """#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "claude test-1.0"
  exit 0
fi
if [ "$1" = "auth" ] && [ "$2" = "status" ]; then
  if [ -n "${USER:-}" ]; then
    echo '{"loggedIn":true}'
    exit 0
  fi
  echo '{"loggedIn":false}'
  exit 1
fi
exit 1
""",
        encoding="utf-8",
    )
    executor.chmod(executor.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join((str(bin_dir), env.get("PATH", "")))
    env["USER"] = "lane-operator"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--host",
            host,
            "--root",
            str(root),
            "--lane",
            lane_id,
            "--json",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["lanes"][0]["status"] == "PASS"
    assert payload["lanes"][0]["state"] == "IDLE"


def test_health_cli_rejects_lane_from_another_host(tmp_path):
    root = tmp_path / "lanes"
    root.mkdir()

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--host",
            "mac",
            "--root",
            str(root),
            "--lane",
            "claude-pc-1",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "does not belong to host mac" in completed.stderr
