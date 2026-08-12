"""Privacy-safe Live Agents projection over existing Hermes history.

This module intentionally owns no runtime, scheduler, database, or usage ledger.
It exposes stable identifiers and presentation-safe fields, never workspace
paths, raw event payloads, prompts, credentials, or worker command arguments.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from hermes_cli import kanban_db, profiles

router = APIRouter()
_PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:\\|/(?:Users|home|private|var|tmp)/)[^\s\"'`]+")
_SECRET = re.compile(
    r"\b(?:api[_-]?key|token|password|secret|authorization)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


def _safe_name(value: Any) -> str:
    return Path(str(value or "artifact").replace("\\", "/")).name[:255]


def _safe_text(value: Any) -> str:
    text = _SECRET.sub("[redacted]", str(value or ""))
    return _PRIVATE_PATH.sub("[private path]", " ".join(text.split()))[:500]


def _milliseconds(value: Any) -> int | None:
    if not isinstance(value, (int, float)):
        return None
    # Kanban timestamps are Unix seconds. Keep already-normalized fixtures safe.
    return int(value if value > 10_000_000_000 else value * 1000)


def _worker_identity(value: Any) -> str:
    """Return a stable opaque worker key without releasing an assignee name."""
    raw = str(value or "").strip()
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"kanban-worker-{digest}"


_EVENT_LABELS = {
    "assigned": "Task assigned to a worker profile.",
    "blocked": "Task needs attention.",
    "claimed": "Worker claimed the task.",
    "completed": "Worker completed the task.",
    "gave_up": "Worker stopped after repeated failures.",
    "heartbeat": "Worker heartbeat received.",
    "reclaimed": "Worker claim was released.",
    "review_requested": "Task entered review.",
    "unblocked": "Task returned to the queue.",
    "worker_started": "Worker process started.",
}


def _event_log_lines(events: list[Any], run_id: int | None) -> list[str]:
    """Project event kinds only; payloads may contain prompts and stay sealed."""
    lines: list[str] = []
    for event in events:
        event_run_id = getattr(event, "run_id", None)
        if event_run_id is not None and event_run_id != run_id:
            continue
        label = _EVENT_LABELS.get(str(getattr(event, "kind", "")))
        if label:
            lines.append(label)
    return lines[-80:]


def _status_activity(status: Any) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"running", "active", "working", "in_progress"}:
        return "Worker is running."
    if normalized in {"done", "completed", "finished"}:
        return "Worker finished."
    if normalized in {"blocked", "failed", "error"}:
        return "Worker needs attention."
    return "Worker state changed."


def _runs_for_board(slug: str) -> list[dict[str, Any]]:
    if not kanban_db.board_exists(slug):
        return []
    conn = kanban_db.connect(board=slug)
    try:
        result: list[dict[str, Any]] = []
        for task in kanban_db.list_tasks(conn, include_archived=True):
            if not task.assignee:
                continue
            attachments = [
                {"id": item.id, "name": _safe_name(item.filename), "kind": item.content_type or "file"}
                for item in kanban_db.list_attachments(conn, task.id)
            ]
            events = kanban_db.list_events(conn, task.id)
            task_event_log = _event_log_lines(events, None)
            runs = kanban_db.list_runs(conn, task.id)
            if not runs:
                result.append(
                    {
                        "id": f"task:{task.id}",
                        "task_id": task.id,
                        "title": _safe_text(task.title),
                        "identity_key": _worker_identity(task.assignee),
                        "board": slug,
                        "status": task.status,
                        "started_at": _milliseconds(task.started_at),
                        "updated_at": _milliseconds(task.completed_at or task.started_at or task.created_at),
                        "ended_at": _milliseconds(task.completed_at),
                        "latest_activity": task_event_log[-1] if task_event_log else _status_activity(task.status),
                        "log": task_event_log,
                        "artifacts": attachments,
                    }
                )
                continue
            for run in runs:
                event_log = _event_log_lines(events, run.id)
                result.append(
                    {
                        "id": str(run.id),
                        "task_id": task.id,
                        "title": _safe_text(task.title),
                        "identity_key": _worker_identity(run.profile or task.assignee),
                        "board": slug,
                        "status": run.status or task.status,
                        "started_at": _milliseconds(run.started_at),
                        "updated_at": _milliseconds(run.last_heartbeat_at or run.ended_at or run.started_at),
                        "ended_at": _milliseconds(run.ended_at),
                        "latest_activity": event_log[-1] if event_log else _status_activity(run.status),
                        "log": event_log,
                        "artifacts": attachments,
                    }
                )
        return result
    finally:
        conn.close()


def _profile_summaries() -> list[dict[str, Any]]:
    return [
        {
            "name": _safe_text(getattr(profile, "name", "")),
            "description": _safe_text(getattr(profile, "description", "")),
            "gateway_running": bool(getattr(profile, "gateway_running", False)),
        }
        for profile in profiles.list_profiles()
        if _safe_text(getattr(profile, "name", ""))
    ]


@router.get("/snapshot")
def snapshot() -> dict[str, Any]:
    """Return presentation-safe profile and Kanban evidence."""
    runs: list[dict[str, Any]] = []
    for board in kanban_db.list_boards(include_archived=True):
        slug = str(board.get("slug") or "")
        if slug:
            runs.extend(_runs_for_board(slug))
    return {"profiles": _profile_summaries(), "runs": runs}


class SteerBody(BaseModel):
    task_id: str
    text: str


class TerminateBody(BaseModel):
    task_id: str
    reason: str | None = None


def _resolve_board(board: str | None) -> str | None:
    if board is None or board == "":
        return None
    try:
        normalized = kanban_db._normalize_board_slug(board)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if normalized and normalized != kanban_db.DEFAULT_BOARD and not kanban_db.board_exists(normalized):
        raise HTTPException(status_code=404, detail=f"board {normalized!r} does not exist")
    return normalized


@router.post("/runs/{run_id}/steer")
def steer_run(
    run_id: int,
    payload: SteerBody,
    board: str | None = Query(None),
) -> dict[str, Any]:
    """Append an operator note only to the exact active Kanban run."""
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    resolved_board = _resolve_board(board)
    conn = kanban_db.connect(board=resolved_board)
    try:
        run = kanban_db.get_run(conn, run_id)
        task = kanban_db.get_task(conn, payload.task_id)
        if run is None or task is None or run.task_id != payload.task_id:
            raise HTTPException(status_code=404, detail="run target not found")
        if (
            run.ended_at is not None
            or task.status != "running"
            or getattr(task, "current_run_id", None) != run_id
        ):
            raise HTTPException(status_code=409, detail="run target is no longer active")
        kanban_db.add_comment(
            conn,
            payload.task_id,
            author="desktop-live-agents",
            body=payload.text.strip(),
        )
        return {"ok": True, "run_id": run_id, "task_id": payload.task_id}
    finally:
        conn.close()


@router.post("/runs/{run_id}/terminate")
def terminate_run(
    run_id: int,
    payload: TerminateBody,
    board: str | None = Query(None),
) -> dict[str, Any]:
    """Stop only the worker currently owning the exact requested run."""
    resolved_board = _resolve_board(board)
    conn = kanban_db.connect(board=resolved_board)
    try:
        run = kanban_db.get_run(conn, run_id)
        task = kanban_db.get_task(conn, payload.task_id)
        if run is None or task is None or run.task_id != payload.task_id:
            raise HTTPException(status_code=404, detail="run target not found")
        if (
            run.ended_at is not None
            or task.status != "running"
            or getattr(task, "current_run_id", None) != run_id
        ):
            raise HTTPException(status_code=409, detail="run target is no longer active")
        reason = (payload.reason or "Stopped from Live Agents").strip()
        if not kanban_db.reclaim_task(conn, payload.task_id, reason=reason):
            raise HTTPException(status_code=409, detail="worker could not be stopped")
        return {"ok": True, "run_id": run_id, "task_id": payload.task_id}
    finally:
        conn.close()
