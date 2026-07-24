"""Focused regressions for non-red review visibility in the Kanban CLI."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

from hermes_cli import kanban as cli


def test_text_stats_prints_review_count(monkeypatch, capsys):
    stats = {
        "by_status": {"review": 3, "blocked": 1},
        "by_assignee": {},
        "oldest_ready_age_seconds": None,
    }
    monkeypatch.setattr(cli.kb, "connect_closing", lambda: contextlib.nullcontext(object()))
    monkeypatch.setattr(cli.kb, "board_stats", lambda _conn: stats)

    assert cli._cmd_stats(SimpleNamespace(json=False)) == 0

    output = capsys.readouterr().out
    assert "review" in output
    assert "3" in output


def test_task_json_exposes_destructive_block_kind_for_alert_consumers():
    task = cli.kb.Task(
        id="t_delete",
        title="Exact destructive action",
        body=None,
        assignee="cc-builder",
        status="blocked",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        block_kind="destructive",
    )

    assert cli._task_to_dict(task)["block_kind"] == "destructive"


def test_cli_initial_blocked_is_explicitly_typed_destructive(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    cli.kb.init_db()

    payload = json.loads(
        cli.run_slash("create 'Exact destructive action' --initial-status blocked --json")
    )

    assert payload["status"] == "blocked"
    assert payload["block_kind"] == "destructive"


def test_non_destructive_cli_stop_reports_review_not_blocked(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    cli.kb.init_db()

    with cli.kb.connect_closing() as conn:
        task_id = cli.kb.create_task(
            conn,
            title="Need a reversible product decision",
            assignee="worker",
        )

    output = cli.run_slash(
        f"block {task_id} 'Choose A or B' --kind needs_input"
    )

    assert "→ review (non-red attention)" in output
    assert f"Blocked {task_id}" not in output
    with cli.kb.connect_closing() as conn:
        assert cli.kb.get_task(conn, task_id).status == "review"
        assert any(
            comment.body == "REVIEW: Choose A or B"
            for comment in cli.kb.list_comments(conn, task_id)
        )


def test_dispatch_telemetry_calls_retry_breaker_result_triage(monkeypatch, capsys):
    result = SimpleNamespace(
        reclaimed=0,
        crashed=[],
        timed_out=[],
        stale=[],
        auto_blocked=["t_retry"],
        promoted=0,
        spawned=[],
        skipped_unassigned=[],
        skipped_nonspawnable=[],
        skipped_per_profile_capped=[],
        auto_assigned_default=[],
    )
    monkeypatch.setattr(
        cli.kb,
        "connect_closing",
        lambda: contextlib.nullcontext(object()),
    )
    monkeypatch.setattr(cli.kb, "dispatch_once", lambda _conn, **_kwargs: result)

    args = SimpleNamespace(
        dry_run=False,
        max=None,
        failure_limit=2,
        json=True,
    )
    assert cli._cmd_dispatch(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sent_to_triage"] == ["t_retry"]
    assert "auto_blocked" not in payload

    args.json = False
    assert cli._cmd_dispatch(args) == 0
    assert "Sent to triage: 1" in capsys.readouterr().out
