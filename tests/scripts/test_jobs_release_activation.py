"""End-to-end script tests for release activation, parity, and rollback.

Task 10 scope: the release CLI tools are exercised end to end against
temporary git source trees and simulated Mac/PC roots only. No live Hermes
directory, cron, gateway restart, Tailscale connection, or real activation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVATE = REPO_ROOT / "scripts" / "jobs-release-activate.py"
PARITY = REPO_ROOT / "scripts" / "jobs-runtime-parity.py"
ROLLBACK = REPO_ROOT / "scripts" / "jobs-release-rollback.py"


def _release_relpaths() -> list[str]:
    from hermes_cli import jobs_release

    return jobs_release.resolve_release_files(REPO_ROOT)


def _make_source(tmp_path: Path) -> Path:
    from hermes_cli import jobs_release

    root = tmp_path / "src"
    for rel in jobs_release.resolve_release_files(REPO_ROOT):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"fake {rel}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=root, check=True)
    return root


def _run(script: Path, *args: str, expect: int = 0) -> subprocess.CompletedProcess:
    res = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
    )
    assert res.returncode == expect, (
        f"exit {res.returncode} != {expect}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    return res


def _bundle_dir(out: Path) -> Path:
    bundles = list((out / "releases").glob("*/jobs-release-v1"))
    assert len(bundles) == 1, bundles
    return bundles[0]


def test_activate_dry_run_default_no_mutation(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "out"
    mac = tmp_path / "mac"
    mac.mkdir()

    res = _run(
        ACTIVATE,
        "--source-root", str(src),
        "--output-root", str(out),
        "--root", str(mac),
    )

    assert "dry-run" in res.stdout
    assert "status: ok" in res.stdout
    assert list(mac.iterdir()) == []  # target root untouched

    bundle = _bundle_dir(out)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_sha"]
    assert manifest["created_at"]
    assert len(manifest["files"]) == len(_release_relpaths())
    assert all(entry["sha256"] for entry in manifest["files"])

    receipts = list((out / "activation-receipts").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["dry_run"] is True
    assert receipt["host_kind"] == "mac"


def test_activate_apply_installs_and_backs_up_mac_and_pc(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "out"
    mac = tmp_path / "mac"
    mac.mkdir()
    pc = tmp_path / "pc"
    pc.mkdir()

    res = _run(
        ACTIVATE,
        "--source-root", str(src),
        "--output-root", str(out),
        "--root", str(mac),
        "--pc-root", str(pc),
        "--apply",
    )

    assert "mode: apply" in res.stdout
    for rel in _release_relpaths():
        assert (mac / rel).exists(), rel
        assert (pc / rel).exists(), rel

    mac_receipts = list((out / "activation-receipts").glob("mac-*.json"))
    pc_receipts = list((out / "activation-receipts").glob("pc-*.json"))
    assert len(mac_receipts) == 1
    assert len(pc_receipts) == 1


def test_activate_refuses_dirty_source(tmp_path):
    src = _make_source(tmp_path)
    (src / "hermes_cli" / "jobs_identity.py").write_text("dirty\n", encoding="utf-8")
    out = tmp_path / "out"
    mac = tmp_path / "mac"
    mac.mkdir()

    res = _run(
        ACTIVATE,
        "--source-root", str(src),
        "--output-root", str(out),
        "--root", str(mac),
        expect=2,
    )
    assert "refused" in res.stderr
    assert list(mac.iterdir()) == []


def test_parity_script_identical_then_refuses_drift(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "out"
    mac = tmp_path / "mac"
    mac.mkdir()
    pc = tmp_path / "pc"
    pc.mkdir()
    _run(
        ACTIVATE,
        "--source-root", str(src),
        "--output-root", str(out),
        "--root", str(mac),
        "--pc-root", str(pc),
        "--apply",
    )
    bundle = _bundle_dir(out)

    res = _run(
        PARITY,
        "--bundle", str(bundle),
        "--mac-root", str(mac),
        "--pc-root", str(pc),
    )
    assert "parity: ok" in res.stdout

    rel = _release_relpaths()[0]
    (pc / rel).write_bytes(b"tampered\n")
    _run(
        PARITY,
        "--bundle", str(bundle),
        "--mac-root", str(mac),
        "--pc-root", str(pc),
        expect=2,
    )


def test_rollback_script_restores_prior_state(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "out"
    mac = tmp_path / "mac"
    rel = _release_relpaths()[0]
    p = mac / rel
    p.parent.mkdir(parents=True)
    p.write_text("original\n", encoding="utf-8")

    _run(
        ACTIVATE,
        "--source-root", str(src),
        "--output-root", str(out),
        "--root", str(mac),
        "--apply",
    )
    assert (mac / rel).read_text(encoding="utf-8") != "original\n"

    receipt = next((out / "activation-receipts").glob("mac-*.json"))
    (mac / rel).write_bytes(b"drifted after activation\n")

    res = _run(ROLLBACK, "--root", str(mac), "--receipt", str(receipt))
    assert "rollback: ok" in res.stdout
    assert (mac / rel).read_text(encoding="utf-8") == "original\n"
    assert list(receipt.parent.glob("rollback-*.json"))


def test_rollback_script_refuses_root_mismatch(tmp_path):
    src = _make_source(tmp_path)
    out = tmp_path / "out"
    mac = tmp_path / "mac"
    mac.mkdir()
    pc = tmp_path / "pc"
    pc.mkdir()
    _run(
        ACTIVATE,
        "--source-root", str(src),
        "--output-root", str(out),
        "--root", str(mac),
        "--apply",
    )
    receipt = next((out / "activation-receipts").glob("mac-*.json"))
    _run(ROLLBACK, "--root", str(pc), "--receipt", str(receipt), expect=2)
