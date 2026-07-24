"""Regression coverage for the manual application-routed atomic Review hold.

The environment refusal prevents accidental use by normally dispatched
workers.  It is intentionally not treated as a security boundary against a
malicious process running as the same unsandboxed OS user.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _event_payloads(conn, task_id: str) -> list[dict]:
    return [
        json.loads(row["payload"])
        for row in conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'manual_review_requested' "
            "ORDER BY id",
            (task_id,),
        ).fetchall()
    ]


@pytest.mark.parametrize(
    ("starting_status", "starting_kind"),
    [
        ("triage", None),
        ("todo", "dependency"),
        ("scheduled", None),
        ("ready", None),
        ("blocked", None),
        ("blocked", "capability"),
        ("review", "capability"),
    ],
)
def test_manual_review_directly_parks_every_non_destructive_lane(
    kanban_home,
    starting_status,
    starting_kind,
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="source", assignee="default")
        conn.execute(
            "UPDATE tasks SET status = ?, block_kind = ?, "
            "block_recurrences = 1 WHERE id = ?",
            (starting_status, starting_kind, task_id),
        )
        # This trigger proves the operator hold never bounces through Ready.
        conn.execute(
            "CREATE TRIGGER reject_transient_ready BEFORE UPDATE OF status ON tasks "
            "WHEN OLD.status != 'ready' AND NEW.status = 'ready' "
            "BEGIN SELECT RAISE(ABORT, 'transient ready forbidden'); END"
        )
        conn.commit()

        assert kb.manual_review_task(
            conn,
            task_id,
            reason="file the read-only review card",
            actor="operator",
        )

        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.block_kind == "needs_input"
        assert task.block_recurrences == 1
        assert task.current_run_id is None
        payload = _event_payloads(conn, task_id)[-1]
        assert payload == {
            "actor": "operator",
            "reason": "file the read-only review card",
            "previous_status": starting_status,
            "status": "review",
            "previous_block_kind": starting_kind,
            "block_kind": "needs_input",
            "preserved": None,
        }


def test_manual_review_closes_active_run_and_clears_claim_fields(
    kanban_home,
    monkeypatch,
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="running source", assignee="default")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None
        kb._set_worker_pid(conn, task_id, 12345)
        terminations = []
        monkeypatch.setattr(
            kb,
            "_terminate_reclaimed_worker",
            lambda pid, lock, signal_fn=None: terminations.append((pid, lock))
            or {
                "prev_pid": pid,
                "host_local": True,
                "termination_attempted": True,
                "terminated": True,
                "sigkill": False,
            },
        )

        assert kb.manual_review_task(conn, task_id, reason="handoff")

        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.block_kind == "needs_input"
        assert task.claim_lock is None
        assert task.claim_expires is None
        assert task.worker_pid is None
        assert task.current_run_id is None
        assert terminations == [(12345, claimed.claim_lock)]
        run = conn.execute(
            "SELECT status, outcome, summary, ended_at FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert run["status"] == "review"
        assert run["outcome"] == "manual_review"
        assert run["summary"] == "handoff"
        assert run["ended_at"] is not None
        event = conn.execute(
            "SELECT run_id FROM task_events "
            "WHERE task_id = ? AND kind = 'manual_review_requested'",
            (task_id,),
        ).fetchone()
        assert event["run_id"] == run_id


def test_manual_review_refuses_to_detach_running_worker_that_survives(
    kanban_home,
    monkeypatch,
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="live source", assignee="default")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None
        kb._set_worker_pid(conn, task_id, 12345)
        monkeypatch.setattr(
            kb,
            "_terminate_reclaimed_worker",
            lambda *args, **kwargs: {
                "prev_pid": 12345,
                "host_local": True,
                "termination_attempted": True,
                "terminated": False,
                "sigkill": True,
            },
        )
        monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid == 12345)

        assert not kb.manual_review_task(conn, task_id, reason="handoff")

        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.current_run_id == run_id
        assert task.claim_lock == claimed.claim_lock
        assert task.worker_pid == 12345
        run = conn.execute(
            "SELECT ended_at FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert run["ended_at"] is None
        assert _event_payloads(conn, task_id) == []


def test_manual_review_refuses_active_claim_before_pid_is_published(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="spawning source", assignee="default")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None
        assert claimed.claim_lock is not None
        assert claimed.worker_pid is None

        assert not kb.manual_review_task(conn, task_id, reason="handoff")

        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.current_run_id == run_id
        assert task.claim_lock == claimed.claim_lock
        assert task.worker_pid is None
        run = conn.execute(
            "SELECT ended_at FROM task_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert run["ended_at"] is None
        assert _event_payloads(conn, task_id) == []


def test_set_worker_pid_cas_rejects_stale_run(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="new owner", assignee="default")
        first = kb.claim_task(conn, task_id)
        assert first is not None and first.current_run_id is not None
        run_1 = first.current_run_id
        assert kb.block_task(
            conn,
            task_id,
            kind="transient",
            expected_run_id=run_1,
        )
        ok, refusal = kb.promote_task(conn, task_id, actor="operator")
        assert ok, refusal
        second = kb.claim_task(conn, task_id)
        assert second is not None and second.current_run_id is not None
        run_2 = second.current_run_id

        assert not kb._set_worker_pid(
            conn,
            task_id,
            12345,
            expected_run_id=run_1,
        )

        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.current_run_id == run_2
        assert task.worker_pid is None
        spawned = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'spawned'",
            (task_id,),
        ).fetchone()
        assert spawned is None


def test_dispatch_terminates_spawn_when_pid_publication_loses_cas(
    kanban_home,
    monkeypatch,
):
    terminations = []
    monkeypatch.setattr(
        kb,
        "_terminate_reclaimed_worker",
        lambda pid, lock, signal_fn=None: terminations.append((pid, lock))
        or {
            "prev_pid": pid,
            "host_local": True,
            "termination_attempted": True,
            "terminated": True,
            "sigkill": False,
        },
    )

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="instant stop", assignee="default")
        spawned_claims = []

        def stop_before_pid_publish(task, workspace):
            spawned_claims.append(task.claim_lock)
            assert task.current_run_id is not None
            assert kb.block_task(
                conn,
                task.id,
                reason="worker stopped before pid publish",
                kind="transient",
                expected_run_id=task.current_run_id,
            )
            return 12345

        result = kb.dispatch_once(conn, spawn_fn=stop_before_pid_publish)

        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.worker_pid is None
        assert result.spawned == []
        assert terminations == [(12345, spawned_claims[0])]
        spawned = conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'spawned'",
            (task_id,),
        ).fetchone()
        assert spawned is None


def test_manual_review_bypasses_needs_input_recurrence_breaker(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="retry source", assignee="default")
        assert kb.claim_task(conn, task_id) is not None
        assert kb.block_task(
            conn,
            task_id,
            reason="first worker stop",
            kind="needs_input",
        )
        first = kb.get_task(conn, task_id)
        assert first.status == "review"
        assert first.block_recurrences == 1

        ok, refusal = kb.promote_task(conn, task_id, actor="operator")
        assert ok, refusal
        assert kb.get_task(conn, task_id).status == "ready"

        assert kb.manual_review_task(
            conn,
            task_id,
            reason="operator review handoff",
        )
        held = kb.get_task(conn, task_id)
        assert held.status == "review"
        assert held.block_kind == "needs_input"
        assert held.block_recurrences == 1


@pytest.mark.parametrize("terminal_status", ["done", "archived"])
def test_manual_review_preserves_terminal_source_but_records_reason(
    kanban_home,
    terminal_status,
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="terminal source", assignee="default")
        conn.execute(
            "UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?",
            (terminal_status, int(time.time()), task_id),
        )
        conn.commit()

        assert kb.manual_review_task(conn, task_id, reason="review the result")
        task = kb.get_task(conn, task_id)
        assert task.status == terminal_status
        assert task.block_kind is None
        payload = _event_payloads(conn, task_id)[-1]
        assert payload["preserved"] == "terminal"
        assert payload["reason"] == "review the result"


def test_manual_review_preserves_legitimate_destructive_red_block(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="delete production data")
        conn.execute(
            "UPDATE tasks SET status = 'blocked', block_kind = 'destructive', "
            "block_recurrences = 1 WHERE id = ?",
            (task_id,),
        )
        conn.commit()

        assert kb.manual_review_task(conn, task_id, reason="review only")
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert task.block_kind == "destructive"
        assert task.block_recurrences == 1
        payload = _event_payloads(conn, task_id)[-1]
        assert payload["preserved"] == "destructive"


def test_manual_review_is_idempotent_for_already_review(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="held source", assignee="default")
        conn.execute(
            "UPDATE tasks SET status = 'review', block_kind = 'needs_input' "
            "WHERE id = ?",
            (task_id,),
        )
        conn.commit()

        assert kb.manual_review_task(conn, task_id, reason="first")
        assert kb.manual_review_task(conn, task_id, reason="second")
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.block_kind == "needs_input"
        assert [p["reason"] for p in _event_payloads(conn, task_id)] == [
            "first",
            "second",
        ]


def test_manual_review_returns_false_for_unknown_task(kanban_home):
    with kb.connect() as conn:
        assert not kb.manual_review_task(conn, "t_missing", reason="review")


def test_review_cli_parks_source_with_positional_reason(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="cli source", assignee="default")

    output = kc.run_slash(
        f"review {task_id} 'Parked for read-only Hermes review (reversible).'"
    )

    assert "review" in output.lower()
    assert "non-red" in output.lower()
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.block_kind == "needs_input"
        assert _event_payloads(conn, task_id)[-1]["reason"] == (
            "Parked for read-only Hermes review (reversible)."
        )


@pytest.mark.parametrize(
    "worker_env",
    ["HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID"],
)
def test_review_cli_is_operator_only_and_cannot_close_newer_run(
    kanban_home,
    monkeypatch,
    worker_env,
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="owned source", assignee="default")
        first = kb.claim_task(conn, task_id)
        assert first is not None and first.current_run_id is not None
        run_1 = first.current_run_id
        assert kb.block_task(
            conn,
            task_id,
            reason="retry",
            kind="transient",
            expected_run_id=run_1,
        )
        ok, refusal = kb.promote_task(conn, task_id, actor="operator")
        assert ok, refusal
        second = kb.claim_task(conn, task_id)
        assert second is not None and second.current_run_id is not None
        run_2 = second.current_run_id
        assert run_2 != run_1

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setenv(
        worker_env,
        task_id if worker_env == "HERMES_KANBAN_TASK" else str(run_1),
    )
    output = kc.run_slash(f"review {task_id} 'stale worker attempt'")

    assert "operator-only" in output.lower()
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.current_run_id == run_2
        run = conn.execute(
            "SELECT ended_at FROM task_runs WHERE id = ?",
            (run_2,),
        ).fetchone()
        assert run["ended_at"] is None
        assert _event_payloads(conn, task_id) == []
