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
    category_root = root / category
    if root.is_symlink() or category_root.is_symlink() or not category_root.is_dir():
        return False
    try:
        path.resolve(strict=False).relative_to(category_root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return path.resolve(strict=False) != category_root.resolve(strict=True)


def _validated_lane_root(path: Path, lane_id: str) -> Path:
    lane = jobs_reliability._lane_identity(lane_id)
    if lane is None or lane.group("host") != "pc":
        raise ValueError("lane identity")
    if not path.is_dir() or path.is_symlink():
        raise ValueError("lane root")
    resolved = path.resolve(strict=True)
    if resolved.name != lane_id:
        raise ValueError("lane path")
    for child in ("auth", "worktrees", "handoffs", "receipts", "health"):
        child_path = resolved / child
        if not child_path.is_dir() or child_path.is_symlink():
            raise ValueError("lane layout")
    configured = os.environ.get("HERMES_JOBS_PC_LANE_ROOT", "").strip()
    if configured:
        configured_root = Path(configured).resolve(strict=True)
        if (
            not resolved.is_relative_to(configured_root)
            or resolved.parent != configured_root
        ):
            raise ValueError("configured lane path")
    return resolved


def _validated_context_paths(context: SimpleNamespace) -> Path:
    lane_root = _validated_lane_root(Path(context.lane_root), context.lane_id)
    attempt_id = jobs_reliability._safe_path_component(
        context.attempt_id, label="handoff"
    )
    jobs_reliability._safe_path_component(context.job_id, label="job")
    handoff = lane_root / "handoffs" / attempt_id
    if not _contained(Path(context.worktree), lane_root, "worktrees") or not _contained(
        handoff, lane_root, "handoffs"
    ):
        raise ValueError("worker path")
    return lane_root


def _cleanup_without_masking(context: SimpleNamespace) -> bool:
    try:
        return _cleanup(context)
    except Exception:
        return False


def _settle_failed_without_masking(root: Path, *, cleanup_ok: bool) -> None:
    try:
        _atomic_lease(
            root,
            state="FAILED",
            reason_code=(
                "REMOTE_EXECUTOR_FAILED"
                if cleanup_ok
                else "REMOTE_EXECUTOR_FAILED_CLEANUP_FAILED"
            ),
        )
    except Exception:
        pass


def _cleanup(context: SimpleNamespace) -> bool:
    lane_root = Path(context.lane_root)
    worktree = Path(context.worktree)
    handoff = lane_root / "handoffs" / context.attempt_id
    if not _contained(worktree, lane_root, "worktrees") or not _contained(
        handoff, lane_root, "handoffs"
    ):
        return False
    worktree_removed = not worktree.exists()
    handoff_removed = not handoff.exists()
    try:
        if worktree.exists():
            try:
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
            except (OSError, subprocess.SubprocessError):
                pass
            worktree_removed = not worktree.exists()
        if handoff.is_dir() and not handoff.is_symlink():
            try:
                shutil.rmtree(handoff)
            except OSError:
                pass
            handoff_removed = not handoff.exists()
        return worktree_removed and handoff_removed
    except (OSError, subprocess.SubprocessError):
        return False


def execute() -> int:
    context: SimpleNamespace | None = None
    lane_root: Path | None = None
    building = False
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
        lane_root = _validated_context_paths(context)
        _atomic_lease(
            lane_root, state="BUILDING", reason_code="REMOTE_EXECUTOR_STARTED"
        )
        building = True
        result = jobs_reliability.LocalProviderPhaseRunner()(raw["provider"], context)
        cleanup_ok = _cleanup_without_masking(context)
        if not cleanup_ok:
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
        sys.stdout.write(
            json.dumps(asdict(result), sort_keys=True, separators=(",", ":"))
        )
        if cleanup_ok:
            _atomic_lease(lane_root, state="IDLE", reason_code="OK")
        building = False
        return 0
    except Exception as exc:
        if building and context is not None and lane_root is not None:
            cleanup_ok = _cleanup_without_masking(context)
            _settle_failed_without_masking(lane_root, cleanup_ok=cleanup_ok)
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
        lane = jobs_reliability._lane_identity(lane_id)
        if (
            provider not in {"claude", "codex"}
            or lane is None
            or lane.group("provider") != provider
            or lane.group("host") != "pc"
        ):
            raise ValueError("provider identity")
        repository = Path(raw["repository"])
        lane_root = _validated_lane_root(Path(raw["lane_root"]), lane_id)
        if not repository.is_dir():
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
