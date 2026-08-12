"""Provider-free canary tests against an exact temporary runtime copy."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from hermes_cli import jobs_release


REPO_ROOT = Path(__file__).resolve().parents[2]
CANARY = REPO_ROOT / "scripts" / "jobs-handoff-canary.py"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _clean_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    archive = subprocess.run(
        ["git", "archive", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout
    archive_path = tmp_path / "source.tar"
    archive_path.write_bytes(archive)
    with tarfile.open(archive_path) as tar:
        tar.extractall(source)
    # Include this worktree's two owned files even when the test runs before
    # the release commit exists.
    for rel in ("hermes_cli/jobs_release.py", "scripts/jobs-handoff-canary.py"):
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, target)
    _git("init", "-q", "-b", "main", cwd=source)
    _git("config", "user.email", "canary@example.invalid", cwd=source)
    _git("config", "user.name", "Canary", cwd=source)
    _git("add", "-A", cwd=source)
    _git("commit", "-q", "-m", "canary source", cwd=source)
    return source


def _runtime_from_bundle(bundle: Path, tmp_path: Path) -> Path:
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True)
    with tarfile.open(bundle / jobs_release.RUNTIME_ARCHIVE_NAME, "r:gz") as archive:
        archive.extractall(runtime)
    manifest = jobs_release.verify_bundle(bundle)
    for entry in manifest["runtime_files"]:
        path = runtime / entry["relpath"]
        path.chmod(entry["mode"])
    jobs_release.verify_runtime_root(runtime, bundle)
    return runtime


def _bundle(tmp_path: Path):
    source = _clean_source(tmp_path)
    bundle, manifest = jobs_release.build_bundle(source, tmp_path / "release")
    jobs_release.verify_bundle(bundle)
    return bundle, manifest


def _run(
    runtime: Path, bundle: Path, receipt: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CANARY),
            "--runtime-root",
            str(runtime),
            "--bundle",
            str(bundle),
            "--receipt-out",
            str(receipt),
        ],
        capture_output=True,
        text=True,
    )


def test_canary_refuses_missing_or_mutable_runtime(tmp_path):
    bundle, _ = _bundle(tmp_path)
    runtime = _runtime_from_bundle(bundle, tmp_path)
    (runtime / "hermes_cli" / "jobs_handoffs.py").unlink()
    refused = _run(runtime, bundle, tmp_path / "missing.json")
    assert refused.returncode == 2
    assert "refused" in refused.stderr.lower()

    runtime = _runtime_from_bundle(bundle, tmp_path / "second")
    target = runtime / "hermes_cli" / "jobs_handoffs.py"
    target.write_bytes(target.read_bytes() + b"\nmutable\n")
    refused = _run(runtime, bundle, tmp_path / "mutable.json")
    assert refused.returncode == 2
    assert "runtime root" in refused.stderr.lower()


def test_canary_passes_signed_conversation_exact_origin_and_bounded_receipt(tmp_path):
    bundle, manifest = _bundle(tmp_path)
    runtime = _runtime_from_bundle(bundle, tmp_path)
    receipt_path = tmp_path / "receipt.json"
    result = _run(runtime, bundle, receipt_path)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["verdict"] == "PASS"
    assert receipt["release_sha"] == manifest["release_sha"]
    assert receipt["duplicate_notification_ids"] == 0
    assert receipt["generic_status_messages"] == 0
    assert receipt["decoy_message_count"] == 0
    assert receipt["delivered_count"] == receipt["notification_count"]
    assert receipt["transition_count"] >= 16
    assert len(receipt_path.read_bytes()) < 16 * 1024
