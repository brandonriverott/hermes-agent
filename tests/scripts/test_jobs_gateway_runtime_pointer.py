"""Tests for the persistent gateway runtime pointer used during Jobs activation."""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
POINTER = REPO_ROOT / "scripts" / "jobs-gateway-runtime-pointer.py"


def _write_plist(path: Path, working_directory: str) -> None:
    path.write_bytes(
        plistlib.dumps(
            {
                "Label": "ai.hermes.gateway",
                "WorkingDirectory": working_directory,
                "ProgramArguments": ["/venv/bin/python", "-m", "hermes_cli.main"],
            }
        )
    )


def _run(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(POINTER), *args], capture_output=True, text=True
    )
    assert result.returncode == expect, result.stderr
    return result


def test_dry_run_does_not_mutate_gateway_plist(tmp_path: Path) -> None:
    plist = tmp_path / "ai.hermes.gateway.plist"
    _write_plist(plist, "/old/runtime")
    original = plist.read_bytes()

    result = _run("--plist", str(plist), "--runtime-root", "/immutable/runtime")

    assert "dry-run" in result.stdout
    assert plist.read_bytes() == original


def test_apply_updates_only_working_directory_and_records_backup(tmp_path: Path) -> None:
    plist = tmp_path / "ai.hermes.gateway.plist"
    _write_plist(plist, "/old/runtime")
    backup = tmp_path / "backup.plist"

    _run(
        "--plist", str(plist),
        "--runtime-root", "/immutable/runtime",
        "--backup", str(backup),
        "--apply",
    )

    updated = plistlib.loads(plist.read_bytes())
    original = plistlib.loads(backup.read_bytes())
    assert updated["WorkingDirectory"] == "/immutable/runtime"
    assert updated["ProgramArguments"] == original["ProgramArguments"]
    assert original["WorkingDirectory"] == "/old/runtime"
