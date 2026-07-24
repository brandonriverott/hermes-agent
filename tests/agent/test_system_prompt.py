"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(
        cwd=None, skip_soul=False, context_length=None,
        allow_install_tree_fallback=False,
    ):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


def _init_code_repo(path):
    """A git repo that actually holds code — the coding posture requires a source
    file (or manifest), not a bare ``.git`` (a prose/notes repo stays general)."""
    import subprocess

    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    (path / "main.py").write_text("print('hi')\n")


class TestCodingContextBlock:
    def test_injected_when_active(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        stable = _stable_prompt(agent)
        assert "coding agent" in stable
        assert "Workspace" in stable

    def test_absent_when_off(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        # Drive the real path: force the resolved mode to "off" via config.
        with patch("agent.coding_context._coding_mode", return_value="off"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable

    def test_absent_without_tools(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=[], platform="cli")
        assert "coding agent" not in _stable_prompt(agent)


class TestTelegramRichMessagesHint:
    """Verify that TELEGRAM_RICH_MESSAGES_HINT is conditionally included."""

    def test_base_hint_without_rich_messages(self, monkeypatch):
        """When rich_messages is False (default), only the base hint is used."""
        agent = _make_agent(platform="telegram")
        # Mock config to return rich_messages: false (default)
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "platforms": {"telegram": {"extra": {"rich_messages": False}}}
            }
            stable = _stable_prompt(agent)
        # Base hint should be present
        assert "Standard Markdown is automatically converted" in stable
        # Rich-messages extension should NOT be present
        assert "lean into it" not in stable
        assert "task lists" not in stable

    def test_rich_hint_with_rich_messages_enabled(self, monkeypatch):
        """When rich_messages is True, the rich-messages extension is appended."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {
                "platforms": {"telegram": {"extra": {"rich_messages": True}}}
            }
            stable = _stable_prompt(agent)
        # Base hint should be present
        assert "Standard Markdown is automatically converted" in stable
        # Rich-messages extension should be present
        assert "lean into it" in stable
        assert "task lists" in stable
        assert "math/formulas" in stable

    def test_base_hint_without_config(self, monkeypatch):
        """When config has no telegram section, only base hint is used."""
        agent = _make_agent(platform="telegram")
        with patch("hermes_cli.config.load_config_readonly") as mock_cfg:
            mock_cfg.return_value = {}
            stable = _stable_prompt(agent)
        assert "Standard Markdown is automatically converted" in stable
        assert "lean into it" not in stable


class TestSkillsProgressiveConfigIntegration:
    """The *real* system-prompt construction path must READ
    ``skills.progressive`` from config and render the startup skills block
    progressively.

    The rendering unit tests in tests/agent/test_prompt_builder.py call
    ``build_skills_system_prompt(progressive=True, ...)`` directly — they prove
    the renderer works but not that a live session ever turns progressive on.
    These tests drive ``build_system_prompt_parts`` (the path run_agent uses to
    assemble the stable prompt) so the config→prompt wiring is exercised end to
    end: ``skills.progressive`` is used in real sessions, not merely declared.
    """

    @pytest.fixture(autouse=True)
    def _clear_skills_cache(self):
        # The skills index is cached in-process and on disk; clear both so each
        # case renders from its own isolated HERMES_HOME fixture.
        from agent.prompt_builder import clear_skills_system_prompt_cache

        clear_skills_system_prompt_cache(clear_snapshot=True)
        yield
        clear_skills_system_prompt_cache(clear_snapshot=True)

    def _write_skill(self, home, category, name, description):
        skill_dir = home / "skills" / category / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n"
        )

    def _stable_with_config(self, agent, config):
        # Patch load_config_readonly (the symbol system_prompt reads for
        # skills.progressive) and neutralize the coding-posture demotion so the
        # only thing shaping the skills block is the progressive config.
        with (
            patch("run_agent.load_soul_md", return_value=""),
            patch("run_agent.build_nous_subscription_prompt", return_value=""),
            patch("run_agent.build_environment_hints", return_value=""),
            patch("run_agent.build_context_files_prompt", return_value=""),
            patch(
                "agent.coding_context.coding_compact_skill_categories",
                return_value=frozenset(),
            ),
            patch("hermes_cli.config.load_config_readonly", return_value=config),
        ):
            return build_system_prompt_parts(agent)["stable"]

    def _agent_with_skills(self):
        return _make_agent(
            valid_tool_names=["skills_list", "skill_view"], platform=""
        )

    def test_progressive_config_drives_real_session_prompt(
        self, monkeypatch, tmp_path
    ):
        """With ``skills.progressive.enabled`` set in config, the real prompt
        path renders progressively: priority-category skills keep descriptions,
        every other skill collapses to a compact per-category coverage count
        (neither its description nor its name is printed), and the progressive
        note appears."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._write_skill(
            tmp_path, "coding", "review", "Review pull requests carefully"
        )
        self._write_skill(
            tmp_path,
            "trivia",
            "obscure-widget-forge",
            "Forges obscure widgets from telemetry",
        )

        config = {
            "skills": {
                "progressive": {
                    "enabled": True,
                    "priority_categories": ["coding"],
                }
            }
        }
        stable = self._stable_with_config(self._agent_with_skills(), config)

        # Progressive mode is active on the real path.
        assert "Progressive index" in stable
        # Priority-category skill keeps its one-line description.
        assert "review: Review pull requests carefully" in stable
        # Non-priority skill: both its description and its name are dropped; it
        # is rediscoverable via skills_list(query=...) + skill_view(name). Its
        # category coverage still appears.
        assert "Forges obscure widgets from telemetry" not in stable
        assert "obscure-widget-forge" not in stable
        assert "trivia:" in stable

    def test_priority_skills_from_config_keep_descriptions(
        self, monkeypatch, tmp_path
    ):
        """``skills.progressive.priority_skills`` names are honored by the real
        path — a named skill keeps its description even outside a priority
        category."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._write_skill(tmp_path, "misc", "kept", "Keep this description")
        self._write_skill(
            tmp_path, "misc", "zz-nonpriority-slug", "Drop this description"
        )

        config = {
            "skills": {
                "progressive": {
                    "enabled": True,
                    "priority_skills": ["kept"],
                }
            }
        }
        stable = self._stable_with_config(self._agent_with_skills(), config)

        assert "Progressive index" in stable
        assert "kept: Keep this description" in stable
        assert "Drop this description" not in stable
        # Non-priority skill name is omitted too; rediscovered via skills_list.
        assert "zz-nonpriority-slug" not in stable

    def test_legacy_full_catalog_when_progressive_disabled_in_config(
        self, monkeypatch, tmp_path
    ):
        """Default/absent progressive config leaves the legacy full catalog
        untouched: every skill keeps its description and there is no
        progressive note. Backward-compatible baseline on the real path."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._write_skill(
            tmp_path, "coding", "review", "Review pull requests carefully"
        )
        self._write_skill(
            tmp_path,
            "trivia",
            "obscure-widget-forge",
            "Forges obscure widgets from telemetry",
        )

        config = {"skills": {"progressive": {"enabled": False}}}
        stable = self._stable_with_config(self._agent_with_skills(), config)

        assert "Progressive index" not in stable
        assert "review: Review pull requests carefully" in stable
        assert (
            "obscure-widget-forge: Forges obscure widgets from telemetry" in stable
        )
