"""Regression tests for macOS TCC-protected broad file searches.

Every filesystem fixture lives under pytest's temporary directory.  The tests
never traverse the real home directory or any live macOS-protected location.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tools.file_operations as file_ops_mod
from hermes_cli.config import DEFAULT_CONFIG
from tools.file_operations import ExecuteResult, ShellFileOperations


PROTECTED_HOME_PATHS = (
    "Library/Containers",
    "Library/Group Containers",
    "Library/Mail",
    "Library/Messages",
    "Library/Calendars",
    "Library/Reminders",
    "Library/Mobile Documents",
    "Library/CloudStorage",
    "Pictures/Photos Library.photoslibrary",
)


class LocalShellEnvironment:
    """Minimal real shell backend for temp-fixture integration tests."""

    cwd = "/"

    def execute(self, command, cwd=None, timeout=None, stdin_data=None):
        completed = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=cwd or self.cwd,
            input=stdin_data,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {"output": completed.stdout, "returncode": completed.returncode}


@pytest.fixture
def synthetic_home(tmp_path, monkeypatch):
    home = tmp_path / "synthetic home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(file_ops_mod.sys, "platform", "darwin")

    visible = home / "Documents" / "visible.txt"
    visible.parent.mkdir()
    visible.write_text("needle\n", encoding="utf-8")

    protected = []
    for relative in PROTECTED_HOME_PATHS:
        candidate = home / relative / "protected.txt"
        candidate.parent.mkdir(parents=True)
        candidate.write_text("needle\n", encoding="utf-8")
        protected.append(candidate)

    return home, visible, protected


@pytest.fixture
def real_file_ops():
    return ShellFileOperations(LocalShellEnvironment())


@pytest.mark.parametrize("relative_path", PROTECTED_HOME_PATHS)
def test_home_search_excludes_every_required_protected_path(
    synthetic_home, relative_path
):
    home, _, _ = synthetic_home

    excludes = file_ops_mod._macos_tcc_excluded_globs_for_root(str(home))

    assert f"{relative_path}/**" in excludes


def test_library_search_uses_root_relative_globs_for_paths_with_spaces(synthetic_home):
    home, _, _ = synthetic_home

    excludes = file_ops_mod._macos_tcc_excluded_globs_for_root(str(home / "Library"))

    assert "Group Containers/**" in excludes
    assert "Mobile Documents/**" in excludes
    assert all(not item.startswith("Library/") for item in excludes)


def test_direct_protected_root_is_treated_as_explicit_path_intent(synthetic_home):
    home, _, _ = synthetic_home

    excludes = file_ops_mod._macos_tcc_excluded_globs_for_root(
        str(home / "Library" / "Containers")
    )

    assert excludes == []


@pytest.mark.parametrize("backend", ["rg", "fallback"])
def test_default_file_name_search_omits_protected_fixtures(
    synthetic_home, real_file_ops, monkeypatch, backend
):
    home, visible, _ = synthetic_home
    if backend == "rg" and shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    monkeypatch.setattr(
        real_file_ops,
        "_has_command",
        lambda command: command == ("rg" if backend == "rg" else "find"),
    )

    result = real_file_ops.search("*.txt", path=str(home), target="files", limit=50)

    assert result.error is None
    assert result.files == [str(visible)]


@pytest.mark.parametrize("backend", ["rg", "fallback"])
def test_default_content_search_omits_protected_fixtures(
    synthetic_home, real_file_ops, monkeypatch, backend
):
    home, visible, _ = synthetic_home
    if backend == "rg" and shutil.which("rg") is None:
        pytest.skip("ripgrep is not installed")
    monkeypatch.setattr(
        real_file_ops,
        "_has_command",
        lambda command: command == ("rg" if backend == "rg" else "grep"),
    )

    result = real_file_ops.search("needle", path=str(home), target="content", limit=50)

    assert result.error is None
    assert [match.path for match in result.matches] == [str(visible)]


def test_per_call_opt_in_includes_protected_fixtures(
    synthetic_home, real_file_ops, monkeypatch
):
    home, visible, protected = synthetic_home
    monkeypatch.setattr(real_file_ops, "_has_command", lambda command: command == "find")

    result = real_file_ops.search(
        "*.txt",
        path=str(home),
        target="files",
        limit=50,
        include_tcc_paths=True,
    )

    assert result.error is None
    assert set(result.files) == {str(visible), *(str(path) for path in protected)}


def test_config_opt_in_includes_protected_fixtures(
    synthetic_home, real_file_ops, monkeypatch
):
    home, visible, protected = synthetic_home
    monkeypatch.setattr(real_file_ops, "_has_command", lambda command: command == "grep")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"agent": {"search": {"include_tcc_paths": True}}},
    )

    result = real_file_ops.search("needle", path=str(home), target="content", limit=50)

    assert result.error is None
    assert {match.path for match in result.matches} == {
        str(visible),
        *(str(path) for path in protected),
    }


def test_default_config_keeps_tcc_paths_excluded():
    assert DEFAULT_CONFIG["agent"]["search"]["include_tcc_paths"] is False


def test_linux_does_not_compute_macos_exclusions(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(file_ops_mod.sys, "platform", "linux")

    assert file_ops_mod._macos_tcc_excluded_globs_for_root(str(tmp_path)) == []


def test_rg_file_search_quotes_protected_globs_with_spaces(monkeypatch):
    ops = ShellFileOperations(MagicMock(cwd="/tmp"))
    commands = []

    def fake_exec(command, *args, **kwargs):
        commands.append(command)
        return ExecuteResult(stdout="", exit_code=1)

    monkeypatch.setattr(ops, "_exec", fake_exec)

    ops._search_files_rg(
        "*.txt",
        "/tmp/home",
        limit=10,
        offset=0,
        macos_tcc_excludes=[
            "Library/Group Containers/**",
            "Library/Mobile Documents/**",
        ],
    )

    assert commands
    assert all("--glob '!Library/Group Containers/**'" in cmd for cmd in commands)
    assert all("--glob '!Library/Mobile Documents/**'" in cmd for cmd in commands)


def test_rg_content_search_adds_all_protected_globs(monkeypatch):
    ops = ShellFileOperations(MagicMock(cwd="/tmp"))
    commands = []

    def fake_exec(command, *args, **kwargs):
        commands.append(command)
        return ExecuteResult(stdout="", exit_code=1)

    monkeypatch.setattr(ops, "_exec", fake_exec)
    exclusions = [f"{path}/**" for path in PROTECTED_HOME_PATHS]

    ops._search_with_rg(
        "needle",
        "/tmp/home",
        None,
        10,
        0,
        "content",
        0,
        exclusions,
    )

    assert len(commands) == 1
    for glob in exclusions:
        assert f"--glob '!{glob}'" in commands[0]


def test_linux_rg_file_command_is_byte_equivalent(monkeypatch):
    monkeypatch.setattr(file_ops_mod.sys, "platform", "linux")
    ops = ShellFileOperations(MagicMock(cwd="/tmp"))
    commands = []

    def fake_exec(command, *args, **kwargs):
        commands.append(command)
        return ExecuteResult(stdout="/tmp/repo/a.py\n", exit_code=0)

    monkeypatch.setattr(ops, "_exec", fake_exec)

    ops._search_files_rg("*.py", "/tmp/repo", limit=10, offset=2)

    assert commands == [
        "rg --files --sortr=modified -g '*.py' '/tmp/repo' 2>/dev/null | head -n 12"
    ]


def test_linux_rg_content_command_is_byte_equivalent(monkeypatch):
    monkeypatch.setattr(file_ops_mod.sys, "platform", "linux")
    ops = ShellFileOperations(MagicMock(cwd="/tmp"))
    commands = []

    def fake_exec(command, *args, **kwargs):
        commands.append(command)
        return ExecuteResult(stdout="", exit_code=1)

    monkeypatch.setattr(ops, "_exec", fake_exec)

    ops._search_with_rg(
        "needle", "/tmp/repo", "*.py", 10, 2, "content", 0
    )

    assert commands == [
        "set -o pipefail; rg --line-number --no-heading --with-filename "
        "--glob '*.py' 'needle' '/tmp/repo' | head -n 12"
    ]


def test_linux_grep_command_is_byte_equivalent(monkeypatch):
    monkeypatch.setattr(file_ops_mod.sys, "platform", "linux")
    ops = ShellFileOperations(MagicMock(cwd="/tmp"))
    commands = []

    def fake_exec(command, *args, **kwargs):
        commands.append(command)
        return ExecuteResult(stdout="", exit_code=1)

    monkeypatch.setattr(ops, "_exec", fake_exec)

    ops._search_with_grep(
        "needle", "/tmp/repo", "*.py", 10, 2, "content", 0
    )

    assert commands == [
        "set -o pipefail; grep -rnH --exclude-dir='.*' --include '*.py' "
        "'needle' '/tmp/repo' | head -n 12"
    ]


def test_linux_find_command_is_byte_equivalent(monkeypatch):
    monkeypatch.setattr(file_ops_mod.sys, "platform", "linux")
    ops = ShellFileOperations(MagicMock(cwd="/tmp"))
    commands = []

    def fake_exec(command, *args, **kwargs):
        commands.append(command)
        if command.startswith("find "):
            return ExecuteResult(stdout="1.0 /tmp/repo/a.py\n", exit_code=0)
        return ExecuteResult(stdout="", exit_code=0)

    monkeypatch.setattr(ops, "_exec", fake_exec)
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "find")

    ops._search_files("*.py", "/tmp/repo", limit=10, offset=2)

    assert commands == [
        "find '/tmp/repo' -not -path '*/.*' -type f -name '*.py' "
        "-printf '%T@ %p\\n' 2>/dev/null | sort -rn | tail -n +3 | head -n 10"
    ]
