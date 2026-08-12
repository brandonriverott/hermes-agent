"""Regression coverage for dispatcher worker-skill preflight.

A card must not consume worker attempts when its requested skill cannot be
loaded by the assigned Hermes profile.  The live #1638 incident reached the
worker twice with ``--skills tea-time-hub`` even though that identifier names
only a category directory (there is no ``tea-time-hub/SKILL.md``).
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture()
def isolated_board(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    profile_home = tmp_path / "profiles" / "builder"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (profile_home / "skills").mkdir()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "builder"
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.resolve_profile_env",
        lambda name: str(profile_home),
    )
    kb.init_db()
    return profile_home


def test_create_rejects_skill_unavailable_to_assignee(isolated_board):
    with kb.connect_closing() as conn:
        with pytest.raises(
            ValueError,
            match=r"unknown or unavailable skill\(s\).*missing-skill",
        ):
            kb.create_task(
                conn,
                title="cannot launch",
                assignee="builder",
                skills=["missing-skill"],
            )

        count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 0


@pytest.mark.parametrize(
    ("assignee", "jobs_lane"),
    [("cc-builder", "claude"), ("codex-builder", "codex")],
)
def test_create_rejects_legacy_builder_alias_without_launchable_profile(
    isolated_board,
    assignee,
    jobs_lane,
):
    with kb.connect_closing() as conn:
        with pytest.raises(
            ValueError,
            match=rf"{assignee}.*not a launchable Kanban profile.*Jobs lane {jobs_lane}",
        ):
            kb.create_task(
                conn,
                title="must use executable jobs lane",
                assignee=assignee,
            )

        count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 0


def test_dispatcher_blocks_legacy_invalid_skill_without_spawning_or_attempt(
    isolated_board,
):
    spawned = []

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="legacy card", assignee="builder")
        # Simulate a card written before create-time validation existed (or an
        # out-of-band writer that bypassed the public creation surface).
        conn.execute(
            "UPDATE tasks SET skills = ? WHERE id = ?",
            (json.dumps(["missing-skill"]), task_id),
        )
        conn.commit()

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *args, **kwargs: spawned.append((args, kwargs)),
        )

        task = kb.get_task(conn, task_id)
        run_count = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        blocked_events = [
            event
            for event in kb.list_events(conn, task_id=task_id)
            if event.kind == "blocked"
        ]

    assert spawned == []
    assert task is not None
    assert task.status == "blocked"
    assert task.consecutive_failures == 0
    assert run_count == 0
    assert result.preflight_failed == [
        (task_id, "unknown or unavailable skill(s) for @builder: missing-skill")
    ]
    assert blocked_events[-1].payload["kind"] == "capability"
    assert "missing-skill" in blocked_events[-1].payload["reason"]


def test_dispatcher_blocks_existing_legacy_builder_alias_without_attempt(
    isolated_board,
):
    spawned = []

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="old bad route", assignee="builder")
        # Simulate a legacy/out-of-band card created before this preflight.
        conn.execute(
            "UPDATE tasks SET assignee = 'cc-builder' WHERE id = ?",
            (task_id,),
        )
        conn.commit()

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *args, **kwargs: spawned.append((args, kwargs)),
        )

        task = kb.get_task(conn, task_id)
        run_count = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
        ).fetchone()[0]

    assert spawned == []
    assert task is not None
    assert task.status == "blocked"
    assert task.consecutive_failures == 0
    assert run_count == 0
    assert result.skipped_nonspawnable == []
    assert result.preflight_failed == [
        (
            task_id,
            "@cc-builder is not a launchable Kanban profile; "
            "create this work in Jobs lane claude",
        )
    ]


def test_dispatcher_launches_when_attached_skill_exists(isolated_board):
    skill_dir = isolated_board / "skills" / "real-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: real-skill\ndescription: test fixture\n---\n\n# Real\n",
        encoding="utf-8",
    )

    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="launchable",
            assignee="builder",
            skills=["real-skill"],
        )
        result = kb.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: 43210)
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.status == "running"
    assert result.preflight_failed == []
    assert result.spawned[0][0] == task_id


def test_create_rejects_folder_alias_when_declared_skill_is_disabled(
    isolated_board,
):
    skill_dir = isolated_board / "skills" / "folder-alias"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: disabled-name\ndescription: test fixture\n---\n",
        encoding="utf-8",
    )
    (isolated_board / "config.yaml").write_text(
        "skills:\n  disabled:\n    - disabled-name\n",
        encoding="utf-8",
    )

    with kb.connect_closing() as conn:
        with pytest.raises(ValueError, match="folder-alias"):
            kb.create_task(
                conn,
                title="disabled alias",
                assignee="builder",
                skills=["folder-alias"],
            )


def test_create_rejects_platform_incompatible_skill(isolated_board):
    skill_dir = isolated_board / "skills" / "windows-only"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: windows-only\ndescription: test fixture\n"
        "platforms: [windows]\n---\n",
        encoding="utf-8",
    )

    with kb.connect_closing() as conn:
        with pytest.raises(ValueError, match="windows-only"):
            kb.create_task(
                conn,
                title="wrong platform",
                assignee="builder",
                skills=["windows-only"],
            )


def test_create_rejects_ambiguous_bare_skill_but_accepts_categorized_path(
    isolated_board,
):
    for category in ("alpha", "beta"):
        skill_dir = isolated_board / "skills" / category / "shared"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: shared\ndescription: test fixture\n---\n",
            encoding="utf-8",
        )

    with kb.connect_closing() as conn:
        with pytest.raises(ValueError, match="shared"):
            kb.create_task(
                conn,
                title="ambiguous",
                assignee="builder",
                skills=["shared"],
            )

        task_id = kb.create_task(
            conn,
            title="disambiguated",
            assignee="builder",
            skills=["alpha/shared"],
        )

    assert task_id
