"""Behavior contract for the bundled, versioned Jobs chat tools."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from contextvars import copy_context
from pathlib import Path

import pytest
import yaml

from hermes_cli import jobs_tool
from hermes_cli import plugins as plugin_api
from hermes_cli.plugins import PluginManager
from tools.registry import registry


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "jobs"
TOOL_NAMES = ("jobs_create", "jobs_queue")


@pytest.fixture(autouse=True)
def _clean_jobs_tools():
    for name in TOOL_NAMES:
        registry.deregister(name)
    yield
    for name in TOOL_NAMES:
        registry.deregister(name)
    sys.modules.pop("hermes_plugins.jobs", None)


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repository with spaces"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Hermes Tests",
            "-c",
            "user.email=hermes-tests@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, head


@pytest.fixture
def exact_origin() -> dict[str, str]:
    return {
        "platform": "discord",
        "chat_id": "chat-42",
        "session_id": "session-42",
        "chat_type": "group",
        "thread_id": "thread-7",
        "user_id": "user-9",
        "profile": "default",
    }


def _register_real_bundled_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> PluginManager:
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    bundled = tmp_path / "bundled"
    bundled.mkdir(exist_ok=True)
    shutil.copytree(
        PLUGIN_DIR,
        bundled / "jobs",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        dirs_exist_ok=True,
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugin_api, "get_bundled_plugins_dir", lambda: bundled)

    manager = PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins["jobs"]
    assert loaded.enabled is True
    assert loaded.error is None
    assert set(loaded.tools_registered) == set(TOOL_NAMES)
    return manager


def _handler(name: str):
    entry = registry.get_entry(name)
    assert entry is not None
    assert entry.toolset == "jobs"
    return entry.handler


def _invoke_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: dict,
    *,
    origin: object,
) -> dict:
    _register_real_bundled_plugin(tmp_path, monkeypatch)
    if isinstance(origin, BaseException):
        def _raise_origin():
            raise origin

        monkeypatch.setattr(jobs_tool, "capture_exact_origin", _raise_origin)
    else:
        monkeypatch.setattr(jobs_tool, "capture_exact_origin", lambda: origin)
    return json.loads(_handler("jobs_create")(args))


def _job_count(home: Path) -> int:
    db_path = home / "jobs.db"
    if not db_path.exists():
        return 0
    from hermes_cli import jobs_db as jdb

    conn = jdb.connect(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    finally:
        conn.close()


def _valid_args(repo: Path, **updates) -> dict:
    args = {
        "name": "Ship exact-origin updates",
        "goal": "Keep this body exactly.\n\nTRAILING=true\n",
        "repo_path": str(repo),
        "lane": "codex",
    }
    args.update(updates)
    return args


def test_bundled_plugin_registers_exact_schema_without_eager_jobs_stack(
    tmp_path, monkeypatch
):
    for module_name in (
        "hermes_cli.jobs_db",
        "hermes_cli.jobs_exec",
        "hermes_cli.jobs_dispatch",
        "hermes_cli.jobs_adapter_claude",
    ):
        sys.modules.pop(module_name, None)

    manager = _register_real_bundled_plugin(tmp_path, monkeypatch)
    manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))

    create = registry.get_entry("jobs_create")
    queue = registry.get_entry("jobs_queue")
    assert create is not None and queue is not None
    assert create.schema["parameters"]["required"] == [
        "name",
        "goal",
        "repo_path",
        "lane",
    ]
    assert create.schema["parameters"]["properties"]["lane"]["enum"] == [
        "claude",
        "codex",
    ]
    assert "kat" not in json.dumps(create.schema).lower()
    assert manifest["kind"] == "backend"
    assert manifest["provides_tools"] == ["jobs_create", "jobs_queue"]
    assert manager._plugins["jobs"].manifest.source == "bundled"
    assert not any(
        name in sys.modules
        for name in (
            "hermes_cli.jobs_db",
            "hermes_cli.jobs_exec",
            "hermes_cli.jobs_dispatch",
            "hermes_cli.jobs_adapter_claude",
        )
    )


def test_real_messaging_context_creates_with_exact_origin_and_optional_fields(
    tmp_path, monkeypatch, git_repo
):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import async_delegation

    repo, _ = git_repo
    _register_real_bundled_plugin(tmp_path, monkeypatch)
    # The API-only helper must never replace a real push-platform chat id.
    monkeypatch.setattr(
        async_delegation, "_current_origin_session_id", lambda: "wrong-api-origin"
    )
    tokens = set_session_vars(
        platform="discord",
        source="gateway",
        chat_id="push-chat",
        session_id="push-session",
        chat_type="group",
        thread_id="push-thread",
        user_id="push-user",
        profile="work",
    )
    try:
        captured = jobs_tool.capture_exact_origin()
        result = json.loads(_handler("jobs_create")(_valid_args(repo)))
    finally:
        clear_session_vars(tokens)

    assert captured == {
        "platform": "discord",
        "chat_id": "push-chat",
        "session_id": "push-session",
        "chat_type": "group",
        "thread_id": "push-thread",
        "user_id": "push-user",
        "profile": "work",
    }
    assert result["lane"] == "codex"
    assert result["notification_promise"] == (
        "Lifecycle updates will return to this originating Hermes chat."
    )
    from hermes_cli import jobs_db as jdb

    conn = jdb.connect(tmp_path / "hermes-home" / "jobs.db")
    try:
        row = conn.execute(
            "SELECT origin FROM job_origins WHERE job_id = ?", (result["job_id"],)
        ).fetchone()
        assert json.loads(row["origin"]) == captured
    finally:
        conn.close()


def test_cached_gateway_turn_binds_durable_session_id_and_create_succeeds(
    tmp_path, monkeypatch, git_repo
):
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionContext, SessionSource

    repo, _ = git_repo
    _register_real_bundled_plugin(tmp_path, monkeypatch)
    runner = object.__new__(GatewayRunner)
    context = SessionContext(
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="cached-chat",
            chat_type="dm",
            user_id="cached-user",
        ),
        connected_platforms=[],
        home_channels={},
        session_key="agent:main:telegram:dm:cached-chat",
        session_id="durable-cached-session",
    )

    # Bind/clear once, then bind the same context again as a cached-agent turn.
    first_tokens = runner._set_session_env(context)
    try:
        assert jobs_tool.capture_exact_origin()["session_id"] == (
            "durable-cached-session"
        )
    finally:
        runner._clear_session_env(first_tokens)
    cached_tokens = runner._set_session_env(context)
    try:
        result = json.loads(_handler("jobs_create")(_valid_args(repo)))
    finally:
        runner._clear_session_env(cached_tokens)

    assert result["job_id"].startswith("j_")
    assert _job_count(tmp_path / "hermes-home") == 1


@pytest.mark.parametrize("source", ["desktop", "tui", "api_server"])
def test_local_and_api_sources_map_to_exact_api_server_origin(source):
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        platform="",
        source=source,
        chat_id="",
        session_id=f"{source}-session",
    )
    try:
        captured = jobs_tool.capture_exact_origin()
    finally:
        clear_session_vars(tokens)

    assert captured == {
        "platform": "api_server",
        "chat_id": f"{source}-session",
        "session_id": f"{source}-session",
        "chat_type": "",
        "thread_id": "",
        "user_id": "",
        "profile": "",
    }


def test_request_scoped_session_key_is_the_only_durable_id_fallback():
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        platform="telegram",
        source="gateway",
        chat_id="fallback-chat",
        session_id="",
        session_key="agent:main:telegram:dm:fallback-chat",
    )
    try:
        captured = jobs_tool.capture_exact_origin()
    finally:
        clear_session_vars(tokens)

    assert captured["session_id"] == "agent:main:telegram:dm:fallback-chat"
    assert captured["platform"] == "telegram"
    assert captured["chat_id"] == "fallback-chat"


def test_api_server_prefers_clobber_proof_origin_over_child_session_id(monkeypatch):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import async_delegation

    tokens = set_session_vars(
        platform="api_server",
        source="api_server",
        chat_id="request-bound-origin",
        session_id="delegated-child-clobber",
        session_key="request-session-key",
    )
    monkeypatch.setattr(
        async_delegation,
        "_current_origin_session_id",
        lambda: "clobber-proof-origin",
    )
    try:
        captured = jobs_tool.capture_exact_origin()
    finally:
        clear_session_vars(tokens)

    assert captured["platform"] == "api_server"
    assert captured["chat_id"] == "clobber-proof-origin"
    assert captured["session_id"] == "clobber-proof-origin"


@pytest.mark.parametrize(
    ("source", "session_id", "session_key"),
    [
        ("cli", "tempting-unattached-id", "tempting-unattached-key"),
        ("", "", ""),
    ],
)
def test_unsupported_or_missing_origin_never_creates_any_job_rows(
    tmp_path, monkeypatch, git_repo, source, session_id, session_key
):
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli import jobs_db as jdb

    repo, _ = git_repo
    _register_real_bundled_plugin(tmp_path, monkeypatch)
    db_path = tmp_path / "hermes-home" / "jobs.db"
    conn = jdb.connect(db_path)
    conn.close()
    tokens = set_session_vars(
        platform="",
        source=source,
        chat_id="",
        session_id=session_id,
        session_key=session_key,
    )
    try:
        result = json.loads(_handler("jobs_create")(_valid_args(repo)))
    finally:
        clear_session_vars(tokens)

    assert result["code"] in {"origin_unavailable", "invalid_origin"}
    conn = jdb.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM job_origins").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 0
    finally:
        conn.close()


def test_two_copied_request_contexts_keep_exact_origins_isolated():
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )

    reset_session_vars()
    context_a = copy_context()
    context_b = copy_context()
    tokens_a = context_a.run(
        set_session_vars,
        "discord",
        "gateway",
        "chat-a",
        "dm",
        "",
        "",
        "user-a",
        "",
        "key-a",
        "session-a",
    )
    tokens_b = context_b.run(
        set_session_vars,
        "telegram",
        "gateway",
        "chat-b",
        "dm",
        "",
        "",
        "user-b",
        "",
        "key-b",
        "session-b",
    )
    try:
        origin_a = context_a.run(jobs_tool.capture_exact_origin)
        origin_b = context_b.run(jobs_tool.capture_exact_origin)
    finally:
        context_a.run(clear_session_vars, tokens_a)
        context_b.run(clear_session_vars, tokens_b)

    assert (origin_a["platform"], origin_a["chat_id"], origin_a["session_id"]) == (
        "discord",
        "chat-a",
        "session-a",
    )
    assert (origin_b["platform"], origin_b["chat_id"], origin_b["session_id"]) == (
        "telegram",
        "chat-b",
        "session-b",
    )


@pytest.mark.parametrize(
    ("lane", "executor", "specialist", "model"),
    [
        ("claude", "claude", "claude-builder", "claude-opus-5"),
        ("codex", "codex", "codex-builder", "gpt-5.6-sol"),
    ],
)
def test_create_handler_commits_identity_origin_and_creation_event_atomically(
    tmp_path,
    monkeypatch,
    git_repo,
    exact_origin,
    lane,
    executor,
    specialist,
    model,
):
    repo, head = git_repo
    result = _invoke_create(
        tmp_path,
        monkeypatch,
        _valid_args(repo, lane=lane),
        origin=exact_origin,
    )

    assert result == {
        "job_id": result["job_id"],
        "number": 1,
        "lane": lane,
        "executor": executor,
        "specialist": specialist,
        "model": model,
        "base_commit": head,
        "notification_promise": (
            "Lifecycle updates will return to this originating Hermes chat."
        ),
    }
    assert "gpt-5.6-terra" not in json.dumps(result)

    from hermes_cli import jobs_db as jdb

    conn = jdb.connect(tmp_path / "hermes-home" / "jobs.db")
    try:
        job = jdb.get_job(conn, result["job_id"])
        assert (
            job.requested_lane,
            job.executor,
            job.specialist,
            job.model,
        ) == (lane, executor, specialist, model)
        stored_origin = conn.execute(
            "SELECT origin FROM job_origins WHERE job_id = ?", (job.id,)
        ).fetchone()
        assert json.loads(stored_origin["origin"]) == exact_origin
        events = jdb.get_events(conn, job.id)
        assert [event["kind"] for event in events] == ["job_created"]
        assert events[0]["data"]["requested_lane"] == lane
        assert events[0]["data"]["executor"] == executor
        assert events[0]["data"]["specialist"] == specialist
        assert events[0]["data"]["model"] == model
    finally:
        conn.close()


def test_create_handler_durably_queues_one_notification(
    tmp_path, monkeypatch, git_repo, exact_origin
):
    repo, _ = git_repo
    result = _invoke_create(
        tmp_path,
        monkeypatch,
        _valid_args(repo),
        origin=exact_origin,
    )

    from hermes_cli import jobs_db as jdb
    from hermes_cli import jobs_notifications as jn

    conn = jdb.connect(tmp_path / "hermes-home" / "jobs.db")
    try:
        rows = jn.list_notifications(conn, job_id=result["job_id"])
    finally:
        conn.close()

    assert len(rows) == 1
    assert rows[0].milestone == "queued"
    assert rows[0].job_revision == 1
    assert rows[0].attempt_id is None
    assert rows[0].delivered_at is None


def test_creation_event_failure_rolls_back_job_and_origin(
    tmp_path, monkeypatch, git_repo, exact_origin
):
    repo, _ = git_repo
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import jobs_db as jdb

    conn = jdb.connect(home / "jobs.db")
    conn.execute(
        "CREATE TRIGGER fail_job_event BEFORE INSERT ON job_events "
        "BEGIN SELECT RAISE(ABORT, 'sensitive-trigger-detail'); END"
    )
    conn.commit()
    conn.close()

    result = _invoke_create(
        tmp_path,
        monkeypatch,
        _valid_args(repo),
        origin=exact_origin,
    )

    assert result["code"] == "create_failed"
    assert "sensitive-trigger-detail" not in json.dumps(result)
    conn = jdb.connect(home / "jobs.db")
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM job_origins").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.parametrize(
    "origin",
    [
        None,
        {},
        {"platform": "discord", "chat_id": "chat", "session_id": ""},
        {"platform": "discord", "chat_id": "chat", "session_id": "x" * 513},
        {"platform": "discord\nforged", "chat_id": "chat", "session_id": "s"},
        {"platform": "discord", "chat_id": 42, "session_id": "s"},
    ],
)
def test_missing_or_invalid_exact_origin_fails_closed_without_creating_job(
    tmp_path, monkeypatch, git_repo, origin
):
    repo, _ = git_repo
    result = _invoke_create(
        tmp_path,
        monkeypatch,
        _valid_args(repo),
        origin=origin,
    )

    assert result["code"] in {"origin_unavailable", "invalid_origin"}
    assert 0 < len(result["error"]) <= 240
    assert _job_count(tmp_path / "hermes-home") == 0


def test_origin_capture_error_is_reported_bounded_and_never_creates_originless_job(
    tmp_path, monkeypatch, git_repo
):
    repo, _ = git_repo
    result = _invoke_create(
        tmp_path,
        monkeypatch,
        _valid_args(repo),
        origin=RuntimeError("oauth_token=do-not-leak"),
    )

    assert result["code"] == "origin_unavailable"
    assert "oauth_token" not in json.dumps(result)
    assert _job_count(tmp_path / "hermes-home") == 0


@pytest.mark.parametrize(
    ("updates", "field"),
    [
        ({"lane": "kat"}, "lane"),
        ({"lane": "gpt"}, "lane"),
        ({"lane": ""}, "lane"),
        ({"lane": None}, "lane"),
        ({}, "lane"),
        ({"max_turns": 0}, "max_turns"),
        ({"max_turns": 501}, "max_turns"),
        ({"max_turns": "120"}, "max_turns"),
        ({"max_turns": True}, "max_turns"),
        ({"name": ""}, "name"),
        ({"goal": ""}, "goal"),
        ({"credential=do-not-leak": "value"}, "request"),
    ],
)
def test_invalid_create_input_returns_bounded_error_and_zero_jobs(
    tmp_path, monkeypatch, git_repo, exact_origin, updates, field
):
    repo, _ = git_repo
    args = _valid_args(repo)
    if updates:
        args.update(updates)
    else:
        args.pop("lane")

    result = _invoke_create(tmp_path, monkeypatch, args, origin=exact_origin)

    assert result["code"] == "invalid_input"
    assert result["field"] == field
    assert len(result["error"]) <= 240
    assert "credential=" not in json.dumps(result)
    assert _job_count(tmp_path / "hermes-home") == 0


def test_invalid_repository_and_unreadable_head_fail_without_path_or_job(
    tmp_path, monkeypatch, exact_origin
):
    not_repo = tmp_path / "credential=do-not-leak"
    not_repo.mkdir()
    first = _invoke_create(
        tmp_path,
        monkeypatch,
        _valid_args(not_repo),
        origin=exact_origin,
    )
    assert first["code"] == "invalid_repo"
    assert "credential=" not in json.dumps(first)
    assert _job_count(tmp_path / "hermes-home") == 0

    for name in TOOL_NAMES:
        registry.deregister(name)
    empty_repo = tmp_path / "empty-repo"
    subprocess.run(["git", "init", "-q", str(empty_repo)], check=True)
    second = _invoke_create(
        tmp_path,
        monkeypatch,
        _valid_args(empty_repo),
        origin=exact_origin,
    )
    assert second["code"] == "head_unavailable"
    assert _job_count(tmp_path / "hermes-home") == 0


def test_goal_body_default_and_max_turns_are_deterministic_and_verbatim(
    tmp_path, monkeypatch, git_repo, exact_origin
):
    repo, head = git_repo
    body = "\nKeep leading newline.\nUNICODE=茶\n\n"
    result = _invoke_create(
        tmp_path,
        monkeypatch,
        _valid_args(repo, goal=body),
        origin=exact_origin,
    )
    from hermes_cli import jobs_db as jdb

    conn = jdb.connect(tmp_path / "hermes-home" / "jobs.db")
    try:
        job = jdb.get_job(conn, result["job_id"])
        assert job.goal == (
            f"REPO_PATH={repo.resolve()}\n"
            f"BASE_COMMIT={head}\n"
            "MODEL=gpt-5.6-sol\n"
            "EFFORT=max\n"
            f"MAX_TURNS={jobs_tool.DEFAULT_MAX_TURNS}\n\n"
            f"{body}"
        )
        assert job.goal.endswith(body)
    finally:
        conn.close()

    assert jobs_tool.DEFAULT_MAX_TURNS == 120
    assert jobs_tool.MAX_TURNS_CEILING == 500


def test_head_resolution_uses_argv_check_timeout_and_no_shell(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="a" * 40 + "\n", stderr="")

    assert jobs_tool.resolve_repo_head(repo, run=fake_run) == "a" * 40
    [(argv, kwargs)] = calls
    assert argv == [
        "git",
        "-C",
        str(repo.resolve()),
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    ]
    assert kwargs["check"] is True
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == jobs_tool.GIT_TIMEOUT_SECONDS
    assert kwargs["shell"] is False


def test_jobs_queue_is_read_only_and_reports_canonical_identity(
    tmp_path, monkeypatch, git_repo, exact_origin
):
    repo, _ = git_repo
    created = _invoke_create(
        tmp_path,
        monkeypatch,
        _valid_args(repo, lane="claude"),
        origin=exact_origin,
    )
    from hermes_cli import jobs_db as jdb

    db_path = tmp_path / "hermes-home" / "jobs.db"
    conn = jdb.connect(db_path)
    try:
        before = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("jobs", "job_events", "job_origins")
        }
    finally:
        conn.close()

    result = json.loads(_handler("jobs_queue")({}))

    assert result == {
        "jobs": [
            {
                "job_id": created["job_id"],
                "number": 1,
                "name": "Ship exact-origin updates",
                "status": "working",
                "step": "routing",
                "requested_lane": "claude",
                "executor": "claude",
                "specialist": "claude-builder",
                "model": "claude-opus-5",
            }
        ]
    }
    conn = jdb.connect(db_path)
    try:
        after = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("jobs", "job_events", "job_origins")
        }
    finally:
        conn.close()
    assert after == before


def test_plugin_registers_only_tools_without_cli_or_override(tmp_path, monkeypatch):
    calls = []

    class RecordingContext:
        def register_tool(self, **kwargs):
            calls.append(kwargs)

        def register_cli_command(self, **_kwargs):
            pytest.fail("Jobs plugin must not duplicate the built-in hermes jobs CLI")

    bundled = tmp_path / "bundled"
    bundled.mkdir()
    shutil.copytree(PLUGIN_DIR, bundled / "jobs")
    monkeypatch.setattr(plugin_api, "get_bundled_plugins_dir", lambda: bundled)
    [manifest] = PluginManager()._collect_directory_manifests()
    module = PluginManager()._load_directory_module(manifest)
    module.register(RecordingContext())

    assert [call["name"] for call in calls] == ["jobs_create", "jobs_queue"]
    assert all(call.get("override", False) is False for call in calls)


def test_safe_bundled_jobs_plugin_contains_same_key_legacy_user_plugin(
    tmp_path, monkeypatch, git_repo
):
    home = tmp_path / "hermes-home"
    user_plugin = home / "plugins" / "jobs"
    user_plugin.mkdir(parents=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["jobs"]}}), encoding="utf-8"
    )
    (user_plugin / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "jobs",
                "version": "0.0.0-legacy",
                "provides_tools": ["jobs_create"],
            }
        ),
        encoding="utf-8",
    )
    (user_plugin / "__init__.py").write_text(
        "import json\n"
        "def legacy(args, **kwargs):\n"
        "    return json.dumps({'model': 'gpt-5.6-terra', "
        "'specialist': 'kat-builder', 'origin': None})\n"
        "def register(ctx):\n"
        "    ctx.register_tool(name='jobs_create', toolset='legacy-jobs', "
        "schema={'name': 'jobs_create', 'parameters': {'type': 'object', "
        "'properties': {'lane': {'enum': ['kat']}}}}, handler=legacy)\n"
        "    ctx.register_cli_command(name='jobs-legacy', help='legacy', "
        "setup_fn=lambda parser: None)\n",
        encoding="utf-8",
    )
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    shutil.copytree(PLUGIN_DIR, bundled / "jobs")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugin_api, "get_bundled_plugins_dir", lambda: bundled)

    manager = PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["jobs"]
    assert loaded.manifest.source == "bundled"
    assert loaded.manifest.collision_policy == "bundled-wins-v1"
    assert set(loaded.tools_registered) == {"jobs_create", "jobs_queue"}
    assert "jobs-legacy" not in manager._cli_commands
    create = registry.get_entry("jobs_create")
    assert create is not None
    assert create.toolset == "jobs"
    assert create.schema["parameters"]["properties"]["lane"]["enum"] == [
        "claude",
        "codex",
    ]
    assert "terra" not in json.dumps(create.schema).lower()
    assert "kat" not in json.dumps(create.schema).lower()

    repo, _ = git_repo
    from gateway.session_context import reset_session_vars

    reset_session_vars()
    result = json.loads(create.handler(_valid_args(repo)))
    assert result["code"] in {"origin_unavailable", "invalid_origin"}
    assert "terra" not in json.dumps(result).lower()
    assert _job_count(home) == 0


def test_unmarked_bundled_plugin_keeps_general_user_precedence(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    bundled = tmp_path / "bundled"
    bundled_plugin = bundled / "ordinary"
    user_plugin = home / "plugins" / "ordinary"
    bundled_plugin.mkdir(parents=True)
    user_plugin.mkdir(parents=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["ordinary"]}}), encoding="utf-8"
    )
    for plugin_dir, source_marker, kind in (
        (bundled_plugin, "bundled", "backend"),
        (user_plugin, "user", "standalone"),
    ):
        (plugin_dir / "plugin.yaml").write_text(
            yaml.safe_dump(
                {"name": "ordinary", "version": "1", "kind": kind}
            ),
            encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text(
            f"SOURCE_MARKER = {source_marker!r}\n"
            "def register(ctx):\n    pass\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugin_api, "get_bundled_plugins_dir", lambda: bundled)

    manager = PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["ordinary"]
    assert loaded.manifest.source == "user"
    assert loaded.module.SOURCE_MARKER == "user"
