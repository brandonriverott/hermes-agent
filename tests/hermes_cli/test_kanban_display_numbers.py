"""Regression coverage for permanent, board-local Kanban display numbers."""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from plugins.kanban.dashboard import plugin_api
from tools import kanban_tools


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _reset_display_number_migration(conn):
    """Make an initialized temporary DB resemble a pre-numbering board."""
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET display_number = NULL")
        conn.execute("DELETE FROM kanban_metadata WHERE key LIKE 'display_number_%'")


def test_create_assigns_first_permanent_display_number(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Numbered task")
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.display_number == 1


def test_migration_backfills_done_then_active_with_deterministic_ties_and_preserves_fields(
    kanban_home,
):
    with kb.connect() as conn:
        done_later = kb.create_task(conn, title="Done later", body="keep body", assignee="alice")
        done_tie_b = kb.create_task(conn, title="Done tie B", assignee="bob")
        done_tie_a = kb.create_task(conn, title="Done tie A", assignee="carol")
        active_tie_b = kb.create_task(conn, title="Active tie B", assignee="dave")
        active_tie_a = kb.create_task(conn, title="Active tie A", assignee="erin")
        archived = kb.create_task(conn, title="Old archived", assignee="frank")
        cancelled = kb.create_task(conn, title="Old cancelled", assignee="grace")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='done', completed_at=?, created_at=? WHERE id=?",
                (30, 9, done_later),
            )
            conn.execute(
                "UPDATE tasks SET status='done', completed_at=?, created_at=? WHERE id=?",
                (10, 5, done_tie_b),
            )
            conn.execute(
                "UPDATE tasks SET status='done', completed_at=?, created_at=? WHERE id=?",
                (10, 5, done_tie_a),
            )
            conn.execute("UPDATE tasks SET created_at=? WHERE id=?", (40, active_tie_b))
            conn.execute("UPDATE tasks SET created_at=? WHERE id=?", (40, active_tie_a))
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (archived,))
            conn.execute("UPDATE tasks SET status='cancelled' WHERE id=?", (cancelled,))
        before = {
            row["id"]: dict(row)
            for row in conn.execute("SELECT * FROM tasks").fetchall()
        }
        _reset_display_number_migration(conn)

    kb.init_db()

    with kb.connect() as conn:
        after = {
            row["id"]: dict(row)
            for row in conn.execute("SELECT * FROM tasks").fetchall()
        }
        next_task = kb.create_task(conn, title="Created after migration")
        next_number = kb.get_task(conn, next_task).display_number

    done_order = sorted((done_tie_a, done_tie_b), key=lambda task_id: task_id)
    active_order = sorted((active_tie_a, active_tie_b), key=lambda task_id: task_id)
    expected = {
        done_order[0]: 1,
        done_order[1]: 2,
        done_later: 3,
        active_order[0]: 4,
        active_order[1]: 5,
    }
    assert {task_id: after[task_id]["display_number"] for task_id in expected} == expected
    assert after[archived]["display_number"] is None
    assert after[cancelled]["display_number"] is None
    assert next_number == 6
    for task_id, prior in before.items():
        assert {
            key: value for key, value in after[task_id].items() if key != "display_number"
        } == {
            key: value for key, value in prior.items() if key != "display_number"
        }


def test_migration_is_safe_inside_callers_existing_write_transaction(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Legacy task")
        _reset_display_number_migration(conn)

        with kb.write_txn(conn):
            kb._migrate_display_numbers(conn)

        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.display_number == 1


def test_concurrent_creates_receive_unique_contiguous_numbers(kanban_home):
    total = 24

    def create_one(index):
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title=f"Concurrent {index}")
            task = kb.get_task(conn, task_id)
            assert task is not None
            return task.display_number

    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as executor:
        numbers = list(executor.map(create_one, range(total)))

    assert sorted(numbers) == list(range(1, total + 1))


def test_numbers_are_board_local_and_never_reused_after_archive(kanban_home):
    kb.create_board("second")
    with kb.connect() as default_conn:
        first = kb.create_task(default_conn, title="Default one")
        kb.archive_task(default_conn, first)
        second = kb.create_task(default_conn, title="Default two")
        assert kb.get_task(default_conn, first).display_number == 1
        assert kb.get_task(default_conn, second).display_number == 2
    with kb.connect(board="second") as other_conn:
        other = kb.create_task(other_conn, title="Second-board one")
        assert kb.get_task(other_conn, other).display_number == 1


def test_cli_json_and_worker_context_format_number_without_changing_title(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Keep this descriptive title", assignee="alice")
        task = kb.get_task(conn, task_id)
        context = kb.build_worker_context(conn, task_id)

    assert task is not None
    assert task.title == "Keep this descriptive title"
    assert "#1 — Keep this descriptive title" in kc._fmt_task_line(task)
    assert kc._task_to_dict(task)["display_number"] == 1
    assert plugin_api._task_dict(task)["display_number"] == 1
    assert f"# Kanban task #1 — Keep this descriptive title ({task_id})" in context


def test_cli_and_tool_surfaces_expose_display_number_without_rewriting_title(kanban_home):
    created = kc.run_slash("create 'Keep this descriptive title' --assignee alice")
    task_id = next(token for token in created.split() if token.startswith("t_"))

    assert "#1 — Keep this descriptive title" in kc.run_slash("list")
    assert f"Task {task_id}: #1 — Keep this descriptive title" in kc.run_slash(
        f"show {task_id}"
    )

    cli_show = json.loads(kc.run_slash(f"show {task_id} --json"))
    tool_show = json.loads(kanban_tools._handle_show({"task_id": task_id}))

    assert cli_show["task"]["display_number"] == 1
    assert cli_show["task"]["title"] == "Keep this descriptive title"
    assert tool_show["task"]["display_number"] == 1
    assert tool_show["task"]["title"] == "Keep this descriptive title"


def test_cli_and_worker_context_remain_compatible_for_unnumbered_legacy_task(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Legacy archived", assignee="alice")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='archived', display_number=NULL WHERE id=?",
                (task_id,),
            )
        task = kb.get_task(conn, task_id)
        context = kb.build_worker_context(conn, task_id)

    assert task is not None
    assert task.display_number is None
    assert "#None" not in kc._fmt_task_line(task)
    assert f"Task {task_id}: Legacy archived" in kc.run_slash(f"show {task_id}")
    assert json.loads(kc.run_slash(f"show {task_id} --json"))["task"]["display_number"] is None
    assert f"# Kanban task Legacy archived ({task_id})" in context
