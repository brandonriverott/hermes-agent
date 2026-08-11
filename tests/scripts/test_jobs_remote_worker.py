"""Remote worker entry point refuses unsafe identity before provider work."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "jobs_remote_worker.py"


def _fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lane = tmp_path / "lanes" / "codex-pc-1"
    for child in ("auth", "worktrees", "handoffs", "receipts", "health"):
        (lane / child).mkdir(parents=True, exist_ok=True)
    return repo, base, lane


def test_remote_preflight_accepts_exact_repo_commit_and_lane(tmp_path):
    repo, base, lane = _fixture(tmp_path)
    payload = {
        "schema_version": 1,
        "provider": "codex",
        "repository": str(repo),
        "base_commit": base,
        "branch": "jobs/j_test",
        "lane_root": str(lane),
        "lane_id": "codex-pc-1",
    }
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "preflight"],
        input=json.dumps(payload).encode(),
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0
    assert completed.stdout == b'{"ok":true}'


def test_remote_preflight_rejects_provider_lane_mismatch(tmp_path):
    repo, base, lane = _fixture(tmp_path)
    payload = {
        "schema_version": 1,
        "provider": "claude",
        "repository": str(repo),
        "base_commit": base,
        "branch": "jobs/j_test",
        "lane_root": str(lane),
        "lane_id": "codex-pc-1",
    }
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "preflight"],
        input=json.dumps(payload).encode(),
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""
