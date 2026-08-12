"""Focused tests for the Live Agents plugin backend."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_plugin():
    plugin_file = Path(__file__).parents[2] / "plugins" / "live-agents" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_live_agents_plugin_test", plugin_file)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_projects_permanent_profiles_without_private_paths(monkeypatch):
    module = _load_plugin()
    profile = type(
        "Profile",
        (),
        {
            "name": "argus",
            "description": "Researches durable questions.",
            "gateway_running": False,
            "path": Path("/private/sentinel"),
        },
    )()

    monkeypatch.setattr(module.kanban_db, "list_boards", lambda **_kwargs: [])
    monkeypatch.setattr(module.profiles, "list_profiles", lambda: [profile])

    result = module.snapshot()

    assert result["profiles"] == [
        {
            "name": "argus",
            "description": "Researches durable questions.",
            "gateway_running": False,
        }
    ]
    assert "/private/sentinel" not in repr(result)


def test_snapshot_projects_safe_event_activity_without_raw_worker_log(monkeypatch):
    module = _load_plugin()
    task = SimpleNamespace(
        id="t1",
        title="Ship safely",
        assignee="builder",
        status="running",
        started_at=10,
        completed_at=None,
        created_at=9,
    )
    run = SimpleNamespace(
        id=42,
        profile="builder",
        status="running",
        started_at=10,
        last_heartbeat_at=11,
        ended_at=None,
        summary=None,
        error=None,
        outcome=None,
    )
    attachment = SimpleNamespace(id=3, filename="/Users/example/private/report.txt", content_type="text/plain")
    conn = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(module.kanban_db, "board_exists", lambda _slug: True)
    monkeypatch.setattr(module.kanban_db, "connect", lambda **_kwargs: conn)
    monkeypatch.setattr(module.kanban_db, "list_tasks", lambda *_args, **_kwargs: [task])
    monkeypatch.setattr(module.kanban_db, "list_attachments", lambda *_args, **_kwargs: [attachment])
    monkeypatch.setattr(module.kanban_db, "list_runs", lambda *_args, **_kwargs: [run])
    monkeypatch.setattr(
        module.kanban_db,
        "list_events",
        lambda *_args, **_kwargs: [
            SimpleNamespace(kind="claimed", run_id=42, created_at=10),
            SimpleNamespace(kind="worker_started", run_id=42, created_at=11),
        ],
    )
    monkeypatch.setattr(
        module.kanban_db,
        "read_worker_log",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("raw worker log must not be read")),
    )

    result = module._runs_for_board("main")

    assert result[0]["log"] == ["Worker claimed the task.", "Worker process started."]
    assert result[0]["latest_activity"] == "Worker process started."
    assert result[0]["artifacts"] == [{"id": 3, "name": "report.txt", "kind": "text/plain"}]
    assert "PROMPT_SENTINEL" not in repr(result)
    assert "/Users/example" not in repr(result)


def test_snapshot_uses_an_opaque_worker_identity_without_exposing_assignee(monkeypatch):
    module = _load_plugin()
    task = SimpleNamespace(
        id="t1",
        title="Ship safely",
        assignee="ASSIGNEE_SENTINEL",
        status="running",
        started_at=10,
        completed_at=None,
        created_at=9,
    )
    conn = SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(module.kanban_db, "board_exists", lambda _slug: True)
    monkeypatch.setattr(module.kanban_db, "connect", lambda **_kwargs: conn)
    monkeypatch.setattr(module.kanban_db, "list_tasks", lambda *_args, **_kwargs: [task])
    monkeypatch.setattr(module.kanban_db, "list_attachments", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module.kanban_db, "list_events", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module.kanban_db, "list_runs", lambda *_args, **_kwargs: [])

    result = module._runs_for_board("main")

    assert "assignee" not in result[0]
    assert result[0]["identity_key"].startswith("kanban-worker-")
    assert "ASSIGNEE_SENTINEL" not in repr(result)


def test_steer_run_rejects_stale_target_and_adds_comment_for_exact_active_run(monkeypatch):
    module = _load_plugin()
    conn = SimpleNamespace(close=lambda: None)
    run = SimpleNamespace(id=42, task_id="t1", ended_at=None)
    task = SimpleNamespace(id="t1", status="running", current_run_id=42)
    comments = []

    monkeypatch.setattr(module.kanban_db, "board_exists", lambda _slug: True)
    monkeypatch.setattr(module.kanban_db, "connect", lambda **_kwargs: conn)
    monkeypatch.setattr(module.kanban_db, "get_run", lambda *_args: run)
    monkeypatch.setattr(module.kanban_db, "get_task", lambda *_args: task)
    monkeypatch.setattr(module.kanban_db, "add_comment", lambda _conn, task_id, author, body: comments.append((task_id, author, body)))

    result = module.steer_run(42, module.SteerBody(task_id="t1", text="Check the handoff"), board="main")

    assert result == {"ok": True, "run_id": 42, "task_id": "t1"}
    assert comments == [("t1", "desktop-live-agents", "Check the handoff")]

    task.current_run_id = 99
    try:
        module.steer_run(42, module.SteerBody(task_id="t1", text="stale"), board="main")
    except module.HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("stale run target was accepted")


def test_terminate_run_reclaims_only_the_exact_active_target(monkeypatch):
    module = _load_plugin()
    conn = SimpleNamespace(close=lambda: None)
    run = SimpleNamespace(id=42, task_id="t1", ended_at=None)
    task = SimpleNamespace(id="t1", status="running", current_run_id=42)
    reclaimed = []

    monkeypatch.setattr(module.kanban_db, "board_exists", lambda _slug: True)
    monkeypatch.setattr(module.kanban_db, "connect", lambda **_kwargs: conn)
    monkeypatch.setattr(module.kanban_db, "get_run", lambda *_args: run)
    monkeypatch.setattr(module.kanban_db, "get_task", lambda *_args: task)
    monkeypatch.setattr(
        module.kanban_db,
        "reclaim_task",
        lambda _conn, task_id, reason: reclaimed.append((task_id, reason)) or True,
    )

    result = module.terminate_run(
        42,
        module.TerminateBody(task_id="t1", reason="Stopped from Live Agents"),
        board="main",
    )

    assert result == {"ok": True, "run_id": 42, "task_id": "t1"}
    assert reclaimed == [("t1", "Stopped from Live Agents")]
