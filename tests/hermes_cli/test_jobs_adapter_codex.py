"""Real-seam tests for the Codex CLI execution adapter.

No live Codex CLI. Every test builds a real temporary git repository and injects
a fake ``codex`` process boundary so the launch contract, isolation, evidence
readback, and fail-closed paths are the ones that would run in production.

Provider isolation: a Codex route must never invoke anything Claude, never
inherit Claude credentials, and never lose its `gpt-5.6-sol` model.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import jobs_adapter_codex as codex
from hermes_cli import jobs_exec as jx
from hermes_cli import jobs_executors as executors
from hermes_cli.jobs_contract import JobEnvelope


TOKEN = "claim-token-must-never-appear-anywhere"
CLAUDE_SECRET = "sk-claude-secret-1234567890abcdef"

# Real subprocess.Popen preserved for pass-through of non-codex calls (git, etc.)
_REAL_POPEN = subprocess.Popen


def _codex_only_popen(capture, scenario, rc=0):
    """Return a fake_popen that intercepts only ``codex exec`` launches.

    Everything else (git, version probes, etc.) passes through to the real
    Popen so the repository and worktree infrastructure works correctly.
    """
    def fake_popen(command, **kwargs):
        if isinstance(command, (list, tuple)) and len(command) >= 2 \
                and command[0] == "codex" and command[1] == "exec":
            capture["argv"] = list(command)
            capture["env"] = dict(kwargs.get("env", {}))
            capture["cwd"] = str(kwargs.get("cwd"))
            if "write_result" in scenario and scenario["write_result"] is not None:
                result_path = None
                for i, arg in enumerate(command):
                    if arg == "--output-last-message" and i + 1 < len(command):
                        result_path = Path(command[i + 1])
                if result_path is not None:
                    result_path.write_text(json.dumps(scenario["write_result"]))
            if "commit_file" in scenario:
                cwd = Path(kwargs.get("cwd"))
                target = cwd / scenario["commit_file"]
                target.write_text(scenario.get("commit_body", "codex was here\n"))
                _REAL_POPEN(["git", "-C", str(cwd), "add", scenario["commit_file"]],
                             stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).wait()
                _REAL_POPEN(
                    ["git", "-C", str(cwd), "-c", "user.email=c@w.invalid",
                     "-c", "user.name=Codex-Fake", "-c", "commit.gpgsign=false",
                     "commit", "-qm", "codex fake commit"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).wait()
            proc = FakeCodexProcess(scenario, capture)
            proc._rc = rc
            return proc
        return _REAL_POPEN(command, **kwargs)
    return fake_popen


def _install_fake_codex(monkeypatch, capture, scenario, rc=0):
    """Replace subprocess.Popen with a fake that intercepts only codex exec."""
    monkeypatch.setattr(
        codex.subprocess, "Popen",
        _codex_only_popen(capture, scenario, rc),
    )


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check, capture_output=True, text=True,
    )


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)],
                   check=True, capture_output=True)
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Codex Adapter Test")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "README.md").write_text("base\n")
    _git(path, "add", "README.md")
    _git(path, "commit", "-qm", "base")
    base = _git(path, "rev-parse", "HEAD").stdout.strip()
    return path, base


def _envelope(**overrides):
    fields = dict(
        job_id="j_codex01",
        number=9,
        name="Ship the Codex thing",
        goal="Do the Codex work.\nSol only.\n",
        attempt_id="a_codex01",
        ordinal=1,
        claim_token=TOKEN,
        specialist="codex-builder",
        skills=None,
    )
    fields.update(overrides)
    return JobEnvelope(**fields)


def _spec(repo_path, base, **overrides):
    return jx.ExecutionSpec(
        model=overrides.pop("model", "gpt-5.6-sol"),
        effort=overrides.pop("effort", "high"),
        repo_path=str(repo_path),
        base_commit=base,
        max_turns=overrides.pop("max_turns", 30),
        workspace_kind=overrides.pop("workspace_kind", "worktree"),
    )


# ---------------------------------------------------------------------------
# Fake codex process: record argv + env + stdin, then act out a scenario.
# ---------------------------------------------------------------------------


class _StdinRecorder:
    def __init__(self, capture):
        self._capture = capture

    def write(self, data):
        self._capture["stdin"] = self._capture.get("stdin", b"") + data

    def close(self):
        pass

    def flush(self):
        pass


class FakeCodexProcess:
    """Mimic subprocess.Popen enough for the adapter's launch helper."""

    def __init__(self, scenario, capture):
        self._scenario = scenario
        self._capture = capture
        self.pid = 99999
        self._rc = 0
        self.stdin = _StdinRecorder(capture)

    def wait(self, timeout=None):
        if "timeout" in self._scenario:
            raise subprocess.TimeoutExpired(cmd="codex", timeout=timeout)
        return self._rc

    def kill(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def communicate(self, input=None, timeout=None):
        return (b"", b"")


# ---------------------------------------------------------------------------
# Requirement: exact launch contract — argv contains codex exec + model + cd
# ---------------------------------------------------------------------------

def test_codex_launch_contract_includes_exact_flag_set(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(monkeypatch, capture, {"write_result": {"outcome": "failed"}})

    codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    argv = capture["argv"]
    assert argv[0:2] == ["codex", "exec"]
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "gpt-5.6-sol"
    assert "--sandbox" in argv
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"
    assert "--cd" in argv
    assert "--json" in argv
    assert "-" in argv  # stdin marker
    assert "--output-last-message" in argv
    assert "--output-schema" in argv
    schema_path = argv[argv.index("--output-schema") + 1]
    assert schema_path.endswith("jobs-codex-result.v1.schema.json")


def test_codex_prompt_bytes_arrive_on_stdin_not_argv(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(monkeypatch, capture, {"write_result": {"outcome": "failed"}})

    envelope = _envelope(goal="Do the real work.\nMake the change.\n")
    codex.run_codex_attempt(
        envelope,
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    assert envelope.goal.encode("utf-8") in capture["stdin"]
    assert TOKEN.encode("utf-8") not in capture["stdin"]


def test_codex_env_sets_private_home_and_codex_home(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    codex_home = tmp_path / "lane_codex_home"
    codex_home.mkdir()
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(monkeypatch, capture, {"write_result": {"outcome": "failed"}})

    codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(codex_home),
    )

    env = capture["env"]
    assert env["CODEX_HOME"] == str(codex_home)
    assert env["HOME"] != os.environ.get("HOME", "")
    assert env["HERMES_HOME"] != os.environ.get("HERMES_HOME", "")
    assert env["HOME"] != str(Path.home())
    assert env["HERMES_HOME"] != str(Path.home() / ".hermes")


def test_codex_env_does_not_leak_claude_credentials(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("ANTHROPIC_API_KEY", CLAUDE_SECRET)
    monkeypatch.setenv("CLAUDE_API_KEY", CLAUDE_SECRET)
    capture: dict = {}
    _install_fake_codex(monkeypatch, capture, {"write_result": {"outcome": "failed"}})

    codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    env = capture["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert "CLAUDE_API_KEY" not in env
    assert CLAUDE_SECRET not in json.dumps(env)


def test_codex_env_neither_argv_nor_env_contains_word_claude(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(monkeypatch, capture, {"write_result": {"outcome": "failed"}})

    codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    blob = " ".join(capture["argv"]) + " " + json.dumps(capture["env"])
    assert "claude" not in blob.lower()


def test_codex_env_does_not_inherit_live_jobs_db_path(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_JOBS_DB", "/Users/brandon/.hermes/jobs/jobs.db")
    capture: dict = {}
    _install_fake_codex(monkeypatch, capture, {"write_result": {"outcome": "failed"}})

    codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    assert "HERMES_JOBS_DB" not in capture["env"]
    assert "/tmp/leaked" not in json.dumps(capture["env"])
    assert capture["env"]["HERMES_HOME"] != str(Path.home() / ".hermes")


# ---------------------------------------------------------------------------
# Requirement: wrong model / wrong executor rejected before provider invocation
# ---------------------------------------------------------------------------

def test_codex_wrong_model_is_a_typed_refusal(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(monkeypatch, capture, {"write_result": {"outcome": "failed"}})

    with pytest.raises(codex.AdapterError):
        codex.run_codex_attempt(
            _envelope(),
            _spec(path, base, model="claude-opus-5"),
            workspace_root=workspace,
            wall_clock_seconds=300,
            codex_home=str(tmp_path / "codex_home"),
        )
    assert capture == {}  # no provider invocation happened


def test_codex_wrong_executor_name_is_a_typed_refusal(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(monkeypatch, capture, {"write_result": {"outcome": "failed"}})

    with pytest.raises(codex.AdapterError):
        codex.run_codex_attempt(
            _envelope(specialist="claude-builder"),
            _spec(path, base),
            workspace_root=workspace,
            wall_clock_seconds=300,
            codex_home=str(tmp_path / "codex_home"),
        )
    assert capture == {}


# ---------------------------------------------------------------------------
# Requirement: missing binary / auth failure / timeout / nonzero exit
# ---------------------------------------------------------------------------

def test_codex_missing_binary_fails_closed(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def boom(*args, **kwargs):
        raise OSError("No such file or directory: 'codex'")

    monkeypatch.setattr(codex.subprocess, "Popen", lambda *a, **k: boom(*a, **k) if (a and isinstance(a[0], (list, tuple)) and len(a[0]) >= 2 and a[0][0] == "codex" and a[0][1] == "exec") else _REAL_POPEN(*a, **k))

    with pytest.raises(codex.AdapterError):
        codex.run_codex_attempt(
            _envelope(),
            _spec(path, base),
            workspace_root=workspace,
            wall_clock_seconds=300,
            codex_home=str(tmp_path / "codex_home"),
        )


def test_codex_auth_failure_becomes_failed_attempt(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(monkeypatch, capture, {
        "write_result": {"outcome": "failed", "failure_class": "infrastructure"},
    })

    result = codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    assert result.status == "failed"
    assert result.failure_class == "infrastructure"


def test_codex_timeout_is_interrupted_infrastructure(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}

    class TimeoutPopen:
        def __init__(self, **kw):
            self.pid = 1234
            self.stdin = _StdinRecorder(capture)
            capture["argv"] = list(kw.get("args", []))

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="codex", timeout=timeout)

        def kill(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(codex.subprocess, "Popen", lambda *a, **k: TimeoutPopen(**k) if (a and isinstance(a[0], (list, tuple)) and len(a[0]) >= 2 and a[0][0] == "codex" and a[0][1] == "exec") else _REAL_POPEN(*a, **k))
    monkeypatch.setattr(codex, "_kill_process", lambda proc: None)

    result = codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=5,
        codex_home=str(tmp_path / "codex_home"),
    )

    assert result.status == "interrupted"
    assert result.failure_class == "infrastructure"


def test_codex_nonzero_exit_is_failed_implementation(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(
        monkeypatch, capture,
        {"write_result": {"outcome": "failed"}},
        rc=17,
    )

    result = codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    assert result.status == "failed"
    assert result.failure_class == "implementation"


# ---------------------------------------------------------------------------
# Requirement: independently observed git evidence
# ---------------------------------------------------------------------------

def test_codex_success_evidence_observed_from_git_not_self_report(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(
        monkeypatch, capture,
        {"write_result": {"outcome": "succeeded"}, "commit_file": "answer.txt"},
    )

    result = codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    assert result.status == "succeeded"
    assert result.failure_class is None
    assert result.commit is not None
    real_head = _git(path, "rev-parse", "refs/heads/jobs/9-a_codex01").stdout.strip()
    assert result.commit == real_head
    assert result.receipt["kind"] == "jobs-codex-attempt"
    assert result.receipt["model"] == "gpt-5.6-sol"
    assert result.receipt["executor"] == "codex"
    assert result.receipt["executor_version"] is not None


def test_codex_empty_diff_with_succeeded_report_is_failed_implementation(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(
        monkeypatch, capture,
        {"write_result": {"outcome": "succeeded"}},
    )

    result = codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    assert result.status == "failed"
    assert result.failure_class == "implementation"


def test_codex_malformed_result_json_does_not_make_success(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}

    def fake_popen(command, **kwargs):
        if isinstance(command, (list, tuple)) and len(command) >= 2 \
                and command[0] == "codex" and command[1] == "exec":
            capture["argv"] = list(command)
            capture["env"] = dict(kwargs.get("env", {}))
            for i, arg in enumerate(command):
                if arg == "--output-last-message" and i + 1 < len(command):
                    Path(command[i + 1]).write_text("{not valid json")
            return FakeCodexProcess({"write_result": None}, capture)
        return _REAL_POPEN(command, **kwargs)

    monkeypatch.setattr(codex.subprocess, "Popen", fake_popen)

    result = codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    assert result.status == "failed"
    assert result.failure_class == "infrastructure"


def test_codex_rejects_secret_shaped_extra_env(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    _install_fake_codex(monkeypatch, capture, {"write_result": {"outcome": "failed"}})

    with pytest.raises(codex.AdapterError):
        codex.run_codex_attempt(
            _envelope(),
            _spec(path, base),
            workspace_root=workspace,
            wall_clock_seconds=300,
            codex_home=str(tmp_path / "codex_home"),
            extra_env={"ANTHROPIC_API_KEY": "sk-leak-12345"},
        )
    assert capture == {}


def test_codex_receipt_redacts_secret_in_output_log(repo, tmp_path, monkeypatch):
    path, base = repo
    workspace = tmp_path / "ws"
    workspace.mkdir()
    capture: dict = {}
    secret = "sk-in-log-abcdef0123"
    handoff = workspace / "j_codex01-a_codex01.handoff"

    class SecretPopen(FakeCodexProcess):
        def __init__(self, scenario, cap):
            super().__init__(scenario, cap)
            self._rc = 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(command, **kwargs):
        if isinstance(command, (list, tuple)) and len(command) >= 2 \
                and command[0] == "codex" and command[1] == "exec":
            capture["argv"] = list(command)
            capture["env"] = dict(kwargs.get("env", {}))
            result_path = None
            for i, arg in enumerate(command):
                if arg == "--output-last-message" and i + 1 < len(command):
                    result_path = Path(command[i + 1])
            if result_path is not None:
                result_path.write_text(json.dumps({
                    "outcome": "failed", "failure_class": "implementation",
                    "reason": secret,
                }))
            log = handoff / "output.log"
            if log.exists():
                log.write_text("running... " + secret + " ...done\n")
            return SecretPopen({}, capture)
        return _REAL_POPEN(command, **kwargs)

    monkeypatch.setattr(codex.subprocess, "Popen", fake_popen)

    result = codex.run_codex_attempt(
        _envelope(),
        _spec(path, base),
        workspace_root=workspace,
        wall_clock_seconds=300,
        codex_home=str(tmp_path / "codex_home"),
    )

    serialized = json.dumps(result.receipt)
    assert secret not in serialized
    assert "[redacted]" in serialized


# ---------------------------------------------------------------------------
# Requirement: registry gives Codex a real legacy adapter now
# ---------------------------------------------------------------------------

def test_production_registry_installs_codex_legacy_adapter():
    production = executors.production_registry()
    adapter = production.require_legacy("codex")
    assert adapter.name == "codex"
    assert callable(adapter.preflight)
    assert callable(adapter.preflight_skills)
    assert callable(adapter.run_attempt)


def test_codex_adapter_is_bound_only_to_codex_identity():
    production = executors.production_registry()
    adapter = production.require_legacy("codex")
    assert adapter.name == "codex"
    assert adapter.name != "claude"
