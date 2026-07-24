"""Run-ownership CAS coverage for external Kanban workers."""

from __future__ import annotations

import json
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


def test_claim_json_exposes_current_run_id(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="external worker", assignee="worker")
    finally:
        conn.close()

    payload = json.loads(kc.run_slash(f"claim {task_id} --json"))

    assert payload["task_id"] == task_id
    assert payload["status"] == "running"
    assert isinstance(payload["current_run_id"], int)
    assert payload["current_run_id"] > 0
    assert Path(payload["workspace"]).is_absolute()

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert payload["current_run_id"] == task.current_run_id
    finally:
        conn.close()


def _event_snapshot(conn, task_id):
    return [
        (event.id, event.kind, event.run_id, event.payload)
        for event in kb.list_events(conn, task_id)
    ]


def _run_snapshot(conn, task_id):
    return [
        (run.id, run.status, run.outcome, run.summary, run.ended_at)
        for run in kb.list_runs(conn, task_id)
    ]


def test_complete_expected_run_id_rejects_null_without_mutation(kanban_home):
    """Ownership failure wins before any completion validation side effect."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="never claimed", assignee="worker")
        before_events = _event_snapshot(conn, task_id)
        before_runs = _run_snapshot(conn, task_id)

        assert not kb.complete_task(
            conn,
            task_id,
            summary="late worker claims a phantom child",
            created_cards=["t_deadbeefcafe"],
            expected_run_id=999_999,
        )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.current_run_id is None
        assert task.result is None
        assert _event_snapshot(conn, task_id) == before_events
        assert _run_snapshot(conn, task_id) == before_runs
    finally:
        conn.close()


def test_cli_complete_run_cas_rejects_stale_and_accepts_current(kanban_home):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="retry completion", assignee="worker")
    finally:
        conn.close()

    first = json.loads(kc.run_slash(f"claim {task_id} --json"))
    run1 = first["current_run_id"]

    conn = kb.connect()
    try:
        assert kb.reclaim_task(conn, task_id, reason="retry with new owner")
        reclaimed_events = _event_snapshot(conn, task_id)
        reclaimed_runs = _run_snapshot(conn, task_id)
    finally:
        conn.close()

    stale_while_unowned = kc.run_slash(
        f"complete {task_id} --summary 'late after reclaim' "
        f"--expected-run-id {run1}"
    )
    assert "cannot complete" in stale_while_unowned.lower()

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.current_run_id is None
        assert _event_snapshot(conn, task_id) == reclaimed_events
        assert _run_snapshot(conn, task_id) == reclaimed_runs
    finally:
        conn.close()

    second = json.loads(kc.run_slash(f"claim {task_id} --json"))
    run2 = second["current_run_id"]
    assert run2 != run1

    conn = kb.connect()
    try:
        before_events = _event_snapshot(conn, task_id)
        before_runs = _run_snapshot(conn, task_id)
    finally:
        conn.close()

    stale = kc.run_slash(
        f"complete {task_id} --summary 'stale close' --expected-run-id {run1}"
    )
    assert "cannot complete" in stale.lower()

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == run2
        assert _event_snapshot(conn, task_id) == before_events
        assert _run_snapshot(conn, task_id) == before_runs
    finally:
        conn.close()

    current = kc.run_slash(
        f"complete {task_id} --summary 'current close' --expected-run-id {run2}"
    )
    assert current == f"Completed {task_id}"

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.current_run_id is None
        assert [run.outcome for run in kb.list_runs(conn, task_id)] == [
            "reclaimed",
            "completed",
        ]
    finally:
        conn.close()


@pytest.mark.parametrize("action", ["complete", "block"])
def test_worker_env_run_id_cannot_be_overridden_by_explicit_newer_token(
    kanban_home,
    monkeypatch,
    action,
):
    """A stale worker cannot impersonate the newer run via a CLI flag."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="owned twice", assignee="worker")
    finally:
        conn.close()

    first = json.loads(kc.run_slash(f"claim {task_id} --json"))
    run1 = first["current_run_id"]
    conn = kb.connect()
    try:
        assert kb.reclaim_task(conn, task_id, reason="new owner")
    finally:
        conn.close()
    second = json.loads(kc.run_slash(f"claim {task_id} --json"))
    run2 = second["current_run_id"]
    assert run2 != run1

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run1))
    conn = kb.connect()
    try:
        before_events = _event_snapshot(conn, task_id)
        before_runs = _run_snapshot(conn, task_id)
        before_comments = kb.list_comments(conn, task_id)
    finally:
        conn.close()

    if action == "complete":
        out = kc.run_slash(
            f"complete {task_id} --summary 'stale override' "
            f"--expected-run-id {run2}"
        )
    else:
        out = kc.run_slash(
            f"block {task_id} 'stale override' --kind destructive "
            f"--expected-run-id {run2}"
        )

    assert "conflict" in out.lower() or "worker" in out.lower()
    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == run2
        assert task.block_kind is None
        assert kb.list_comments(conn, task_id) == before_comments
        assert _event_snapshot(conn, task_id) == before_events
        assert _run_snapshot(conn, task_id) == before_runs
    finally:
        conn.close()


def test_worker_context_cannot_mutate_a_different_task_without_cas(
    kanban_home,
    monkeypatch,
):
    """Worker context never falls back to the unguarded operator contract."""
    conn = kb.connect()
    try:
        owned = kb.create_task(conn, title="owned", assignee="worker")
        target = kb.create_task(conn, title="other owner", assignee="worker")
    finally:
        conn.close()
    owned_claim = json.loads(kc.run_slash(f"claim {owned} --json"))
    target_claim = json.loads(kc.run_slash(f"claim {target} --json"))

    monkeypatch.setenv("HERMES_KANBAN_TASK", owned)
    monkeypatch.setenv(
        "HERMES_KANBAN_RUN_ID", str(owned_claim["current_run_id"]),
    )
    out = kc.run_slash(f"complete {target} --summary 'cross-task close'")

    assert "worker" in out.lower()
    conn = kb.connect()
    try:
        task = kb.get_task(conn, target)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == target_claim["current_run_id"]
    finally:
        conn.close()


def test_worker_context_cannot_comment_or_schedule_a_different_task(
    kanban_home,
    monkeypatch,
):
    """Ownership refusal happens before schedule's audit-comment write."""
    conn = kb.connect()
    try:
        owned = kb.create_task(conn, title="owned schedule", assignee="worker")
        target = kb.create_task(conn, title="other schedule", assignee="worker")
    finally:
        conn.close()
    owned_claim = json.loads(kc.run_slash(f"claim {owned} --json"))
    target_claim = json.loads(kc.run_slash(f"claim {target} --json"))
    monkeypatch.setenv("HERMES_KANBAN_TASK", owned)
    monkeypatch.setenv(
        "HERMES_KANBAN_RUN_ID", str(owned_claim["current_run_id"]),
    )
    conn = kb.connect()
    try:
        before_events = _event_snapshot(conn, target)
        before_runs = _run_snapshot(conn, target)
        before_comments = kb.list_comments(conn, target)
    finally:
        conn.close()

    out = kc.run_slash(f"schedule {target} 'stale cross-task note'")

    assert "worker" in out.lower()
    conn = kb.connect()
    try:
        task = kb.get_task(conn, target)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == target_claim["current_run_id"]
        assert kb.list_comments(conn, target) == before_comments
        assert _event_snapshot(conn, target) == before_events
        assert _run_snapshot(conn, target) == before_runs
    finally:
        conn.close()


def test_stale_worker_schedule_rejection_has_zero_side_effects(
    kanban_home,
    monkeypatch,
):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="rescheduled owner", assignee="worker")
    finally:
        conn.close()
    first = json.loads(kc.run_slash(f"claim {task_id} --json"))
    run1 = first["current_run_id"]
    conn = kb.connect()
    try:
        assert kb.reclaim_task(conn, task_id, reason="new schedule owner")
    finally:
        conn.close()
    second = json.loads(kc.run_slash(f"claim {task_id} --json"))
    run2 = second["current_run_id"]
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run1))
    conn = kb.connect()
    try:
        before_events = _event_snapshot(conn, task_id)
        before_runs = _run_snapshot(conn, task_id)
        before_comments = kb.list_comments(conn, task_id)
    finally:
        conn.close()

    out = kc.run_slash(f"schedule {task_id} 'late stale schedule'")

    assert "cannot schedule" in out.lower()
    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == run2
        assert kb.list_comments(conn, task_id) == before_comments
        assert _event_snapshot(conn, task_id) == before_events
        assert _run_snapshot(conn, task_id) == before_runs
    finally:
        conn.close()


def test_cli_block_run_cas_rejects_stale_without_comment_or_repark(
    kanban_home,
):
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="retry block", assignee="worker")
    finally:
        conn.close()

    first = json.loads(kc.run_slash(f"claim {task_id} --json"))
    run1 = first["current_run_id"]

    conn = kb.connect()
    try:
        assert kb.reclaim_task(conn, task_id, reason="retry with new owner")
        reclaimed_events = _event_snapshot(conn, task_id)
        reclaimed_runs = _run_snapshot(conn, task_id)
        reclaimed_comments = kb.list_comments(conn, task_id)
    finally:
        conn.close()

    stale_while_unowned = kc.run_slash(
        f"block {task_id} 'late after reclaim' "
        f"--kind destructive --expected-run-id {run1}"
    )
    assert "cannot block" in stale_while_unowned.lower()

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.current_run_id is None
        assert task.block_kind is None
        assert kb.list_comments(conn, task_id) == reclaimed_comments
        assert _event_snapshot(conn, task_id) == reclaimed_events
        assert _run_snapshot(conn, task_id) == reclaimed_runs
    finally:
        conn.close()

    second = json.loads(kc.run_slash(f"claim {task_id} --json"))
    run2 = second["current_run_id"]
    assert run2 != run1

    conn = kb.connect()
    try:
        before_events = _event_snapshot(conn, task_id)
        before_runs = _run_snapshot(conn, task_id)
        before_comments = kb.list_comments(conn, task_id)
    finally:
        conn.close()

    stale = kc.run_slash(
        f"block {task_id} 'irreversible action approval' "
        f"--kind destructive --expected-run-id {run1}"
    )
    assert "cannot block" in stale.lower()

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == run2
        assert task.block_kind is None
        assert kb.list_comments(conn, task_id) == before_comments
        assert _event_snapshot(conn, task_id) == before_events
        assert _run_snapshot(conn, task_id) == before_runs
    finally:
        conn.close()

    current = kc.run_slash(
        f"block {task_id} 'irreversible action approval' "
        f"--kind destructive --expected-run-id {run2}"
    )
    assert current == f"Blocked {task_id}: irreversible action approval"

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "blocked"
        assert task.block_kind == "destructive"
        assert task.current_run_id is None
        comments = kb.list_comments(conn, task_id)
        assert [comment.body for comment in comments] == [
            "BLOCKED: irreversible action approval"
        ]
    finally:
        conn.close()


def test_cli_block_without_run_token_preserves_legacy_comment_behavior(
    kanban_home,
):
    """Omitting the CAS token keeps the pre-existing manual CLI contract."""
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="manual note", assignee="worker")
        assert kb.complete_task(conn, task_id, summary="already done")
    finally:
        conn.close()

    out = kc.run_slash(f"block {task_id} 'manual follow-up note'")
    assert "cannot block" in out.lower()

    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert [comment.body for comment in kb.list_comments(conn, task_id)] == [
            "REVIEW: manual follow-up note"
        ]
    finally:
        conn.close()
