"""Accepted task plans use the real persisted goal loop without an explicit /goal."""
import threading
from pathlib import Path

import pytest

from hermes_cli import goals
from hermes_cli.goal_autostart import start_goal_if_absent


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text("goals:\n  max_turns: 7\n")
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def start(session="build"):
    return start_goal_if_absent(session, request="Build both parts and verify them",
                                outcome="Build the requested parts", verification="Both parts pass their tests")


def test_plan_arms_real_loop_and_respects_budget(home, monkeypatch):
    assert start() == "started"
    manager = goals.GoalManager("build")
    assert manager.is_active()
    assert manager.state.max_turns == 7
    assert manager.state.goal == "Build both parts and verify them"
    assert "Both parts" in manager.state.contract.verification
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: ("continue", "second part unfinished", False, None, False))
    decision = manager.evaluate_after_turn("Continuing the build", background_processes=[])
    assert decision["should_continue"] is True
    manager.refresh()
    assert manager.state.turns_used == 1
    assert start() == "existing"
    assert goals.load_goal("build").turns_used == 1


@pytest.mark.parametrize("status", ["active", "paused", "cleared", "done", "malformed"])
def test_existing_records_never_replaced(home, status):
    db = goals._get_session_db()
    raw = "broken record" if status == "malformed" else goals.GoalState(
        goal="Keep original work", status=status, turns_used=6, max_turns=7).to_json()
    db.state_meta.set("goal:build", raw)
    assert start() == "existing"
    assert db.state_meta.get("goal:build") == raw


def test_unavailable_storage_reports_failure(home, monkeypatch):
    monkeypatch.setattr(goals, "_get_session_db", lambda: None)
    with pytest.raises(RuntimeError, match="could not open"):
        start()


@pytest.mark.parametrize("command", ["pause", "stop"])
def test_user_control_preserves_a_malformed_goal_and_reports_no_goal(home, command):
    from hermes_cli.goal_command import dispatch_goal_command

    meta = goals._get_session_db().state_meta
    meta.set("goal:build", "broken record")
    result = dispatch_goal_command(goals.GoalManager("build"), command, authorize_gate=lambda: None)
    assert not result.error
    assert "No " in result.output
    assert meta.get("goal:build") == "broken record"


def test_compression_keeps_contract_and_progress(home):
    start()
    manager = goals.GoalManager("build")
    manager.pause(reason="user-paused")
    assert goals.migrate_goal_to_session("build", "compressed", reason="test")
    assert start("compressed") == "existing"
    assert goals.load_goal("compressed").status == "paused"


def test_concurrent_plans_create_once(home):
    from concurrent.futures import ThreadPoolExecutor
    barrier = threading.Barrier(2)
    def submit():
        barrier.wait(timeout=10)
        return start()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit) for _ in range(2)]
        assert sorted(f.result(timeout=20) for f in futures) == ["existing", "started"]


@pytest.mark.parametrize("command", ["pause", "stop"])
@pytest.mark.parametrize("plan_first", [False, True])
def test_control_from_a_preexisting_manager_wins_either_order(home, command, plan_first):
    from hermes_cli.goal_command import dispatch_goal_command

    manager = goals.GoalManager("build")
    if plan_first:
        assert start() == "started"
    result = dispatch_goal_command(manager, command, authorize_gate=lambda: None)
    assert not result.error
    assert start() == "existing"
    assert not goals.GoalManager("build").is_active()
    manager.set("Explicit replacement")
    assert goals.GoalManager("build").is_active()


def test_stop_before_first_plan_survives_compression(home):
    from hermes_cli.goal_command import dispatch_goal_command

    manager = goals.GoalManager("parent")
    dispatch_goal_command(manager, "stop", authorize_gate=lambda: None)
    goals.migrate_goal_to_session("parent", "child")
    assert start("child") == "existing"
    assert not goals.GoalManager("child").is_active()


@pytest.mark.parametrize("command", ["pause", "stop"])
def test_parent_control_follows_a_completed_rotation_but_not_a_replacement(home, command):
    from hermes_cli.goal_command import dispatch_goal_command

    assert start("parent") == "started"
    parent = goals.GoalManager("parent")
    assert goals.migrate_goal_to_session("parent", "child")
    dispatch_goal_command(parent, command, authorize_gate=lambda: None)
    assert not goals.GoalManager("child").is_active()
    goals.GoalManager("child").set("A later explicit goal")
    dispatch_goal_command(parent, command, authorize_gate=lambda: None)
    assert goals.GoalManager("child").is_active()


def test_pause_during_rotation_is_carried_to_child(home, monkeypatch):
    from hermes_cli.goal_command import dispatch_goal_command

    assert start("parent") == "started"
    repo_type = type(goals._get_session_db().state_meta)
    move = repo_type.move_if_current
    def interrupt_move(repo, *args):
        monkeypatch.setattr(repo_type, "move_if_current", move)
        dispatch_goal_command(goals.GoalManager("parent"), "pause", authorize_gate=lambda: None)
        return move(repo, *args)

    monkeypatch.setattr(repo_type, "move_if_current", interrupt_move)
    assert goals.migrate_goal_to_session("parent", "child")
    assert goals.load_goal("child").status == "paused"


def test_failed_archive_rolls_back_child_insert(home):
    import sqlite3
    from contextlib import closing

    assert start("parent") == "started"
    with closing(sqlite3.connect(home / "state.db")) as db:
        db.execute("CREATE TRIGGER fail_archive BEFORE UPDATE ON state_meta "
                   "WHEN old.key = 'goal:parent' BEGIN SELECT RAISE(ABORT, 'interrupted'); END")
    assert not goals.migrate_goal_to_session("parent", "child")
    assert goals.load_goal("child") is None
    assert goals.GoalManager("parent").is_active()


@pytest.mark.parametrize("command", ["pause", "stop"])
def test_control_while_judge_is_running_prevents_a_stale_continuation(home, monkeypatch, command):
    from concurrent.futures import ThreadPoolExecutor
    from hermes_cli.goal_command import dispatch_goal_command

    assert start() == "started"
    entered, release = threading.Event(), threading.Event()
    def judge(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return "continue", "unchecked work", False, None, False

    monkeypatch.setattr(goals, "judge_goal", judge)
    manager = goals.GoalManager("build")
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(manager.evaluate_after_turn, "Draft")
        try:
            assert entered.wait(10)
            dispatch_goal_command(manager, command, authorize_gate=lambda: None)
        finally:
            release.set()
        assert result.result(timeout=10)["should_continue"] is False
    assert not goals.GoalManager("build").is_active()
