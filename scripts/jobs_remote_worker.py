#!/usr/bin/env python3
"""Run and clean one provider pipeline inside an immutable remote runtime."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_cli import jobs_reliability  # noqa: E402


_CONTEXT_KEYS = {
    "job_id",
    "job_number",
    "job_name",
    "goal",
    "attempt_id",
    "ordinal",
    "repository",
    "base_commit",
    "branch",
    "worktree",
    "lane_root",
    "requested_lane",
    "lane_id",
    "executor",
    "specialist",
    "model",
    "effort",
    "max_turns",
}


def _atomic_lease(root: Path, *, state: str, reason_code: str) -> None:
    path = root / "health" / "lease.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps({"state": state, "reason_code": reason_code}, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _contained(path: Path, root: Path, category: str) -> bool:
    try:
        path.resolve(strict=False).relative_to((root / category).resolve(strict=True))
    except (OSError, ValueError):
        return False
    return path.resolve(strict=False) != (root / category).resolve(strict=True)


def _cleanup(context: SimpleNamespace) -> bool:
    lane_root = Path(context.lane_root)
    worktree = Path(context.worktree)
    handoff = lane_root / "handoffs" / context.attempt_id
    if not _contained(worktree, lane_root, "worktrees") or not _contained(
        handoff, lane_root, "handoffs"
    ):
        return False
    try:
        if worktree.exists():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(context.repository),
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        if handoff.is_dir() and not handoff.is_symlink():
            shutil.rmtree(handoff)
        return not worktree.exists() and not handoff.exists()
    except (OSError, subprocess.SubprocessError):
        return False


def execute() -> int:
    try:
        raw = json.loads(sys.stdin.buffer.read(128 * 1024).decode("utf-8"))
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "provider",
            "context",
        }:
            raise ValueError("request shape")
        if raw["schema_version"] != 1 or raw["provider"] not in {"claude", "codex"}:
            raise ValueError("request identity")
        values = raw["context"]
        if not isinstance(values, dict) or set(values) != _CONTEXT_KEYS:
            raise ValueError("context shape")
        context = SimpleNamespace(**values)
        jobs_reliability._require_provider_context(raw["provider"], context, host="pc")
        lane_root = Path(context.lane_root)
        if not lane_root.is_dir() or lane_root.is_symlink():
            raise ValueError("lane root")
        _atomic_lease(
            lane_root, state="BUILDING", reason_code="REMOTE_EXECUTOR_STARTED"
        )
        result = jobs_reliability.LocalProviderPhaseRunner()(raw["provider"], context)
        if not _cleanup(context):
            _atomic_lease(lane_root, state="FAILED", reason_code="CLEANUP_FAILED")
            result = jobs_reliability.PhaseResult(
                status="failed",
                commit=result.commit,
                tests=result.tests,
                review=result.review,
                executor_exit_digest=result.executor_exit_digest,
                output_capture_digest=result.output_capture_digest,
                build_handoff=result.build_handoff,
                critical_user_journey=result.critical_user_journey,
                review_handoff=result.review_handoff,
                failure_reason_code="CLEANUP_FAILED",
                http_status=result.http_status,
                safety_gate=result.safety_gate,
            )
        else:
            _atomic_lease(lane_root, state="IDLE", reason_code="OK")
        sys.stdout.write(
            json.dumps(asdict(result), sort_keys=True, separators=(",", ":"))
        )
        return 0
    except Exception as exc:
        sys.stderr.write(f"remote worker refused: {type(exc).__name__}\n")
        return 2


def preflight() -> int:
    try:
        raw = json.loads(sys.stdin.buffer.read(64 * 1024).decode("utf-8"))
        expected = {
            "schema_version",
            "provider",
            "repository",
            "base_commit",
            "branch",
            "lane_root",
            "lane_id",
        }
        if (
            not isinstance(raw, dict)
            or set(raw) != expected
            or raw["schema_version"] != 1
        ):
            raise ValueError("request shape")
        provider = raw["provider"]
        lane_id = raw["lane_id"]
        if provider not in {"claude", "codex"} or not str(lane_id).startswith(
            provider + "-pc-"
        ):
            raise ValueError("provider identity")
        repository = Path(raw["repository"])
        lane_root = Path(raw["lane_root"])
        if not repository.is_dir() or not lane_root.is_dir() or lane_root.is_symlink():
            raise ValueError("path")
        for child in ("auth", "worktrees", "handoffs", "receipts", "health"):
            path = lane_root / child
            if not path.is_dir() or path.is_symlink():
                raise ValueError("lane layout")
        base = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "cat-file",
                "-e",
                f"{raw['base_commit']}^{{commit}}",
            ],
            check=False,
            capture_output=True,
            timeout=120,
        )
        branch = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "show-ref",
                "--verify",
                "--quiet",
                f"refs/heads/{raw['branch']}",
            ],
            check=False,
            capture_output=True,
            timeout=120,
        )
        if base.returncode != 0 or branch.returncode == 0:
            raise ValueError("git precondition")
        sys.stdout.write('{"ok":true}')
        return 0
    except Exception as exc:
        sys.stderr.write(f"remote preflight refused: {type(exc).__name__}\n")
        return 2


if __name__ == "__main__":
    if sys.argv[1:] == ["execute"]:
        raise SystemExit(execute())
    if sys.argv[1:] == ["preflight"]:
        raise SystemExit(preflight())
    raise SystemExit(2)
