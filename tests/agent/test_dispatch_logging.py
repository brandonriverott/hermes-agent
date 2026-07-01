from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import agent.cc_prompt_capture as cc_prompt_capture
import agent.chat_completion_helpers as cch
import agent.dispatch_logging as dispatch_logging
from agent.copilot_acp_client import CopilotACPClient


def _read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_llm_dispatch_records_resolved_model_without_payload(monkeypatch, tmp_path):
    log_path = tmp_path / "dispatch.jsonl"
    monkeypatch.setattr(dispatch_logging, "dispatch_log_path", lambda: log_path)
    monkeypatch.setenv("HERMES_PROFILE", "ATHENA")
    agent = SimpleNamespace(
        provider="openrouter",
        api_mode="chat_completions",
        model="agent/default-model",
        base_url="https://openrouter.ai/api/v1?api_key=secret",
        session_id="session-1",
        platform="cli",
    )
    client = SimpleNamespace(base_url="https://override.example/v1?token=secret")

    dispatch_logging.log_llm_dispatch(
        agent,
        {
            "model": "resolved/model",
            "messages": [{"role": "user", "content": "secret prompt text"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "api_key": "sk-should-not-be-logged",
            "extra_body": {"provider": {"sort": "latency"}},
        },
        dispatch_path="chat.completions.create",
        client=client,
        stream=False,
    )

    [event] = _read_events(log_path)
    assert event["kind"] == "llm_dispatch"
    assert event["seat"] == "ATHENA"
    assert event["provider"] == "openrouter"
    assert event["model"] == "resolved/model"
    assert event["source"] == "unknown"
    assert event["config_claimed"] == "agent/default-model"
    assert event["agent_model"] == "agent/default-model"
    assert len(event["prompt_hash"]) == 8
    assert event["base_url"] == "https://override.example/v1"
    assert event["base_url_host"] == "override.example"
    assert "api_key" not in event["request_keys"]
    assert event["message_count"] == 1
    assert event["tool_count"] == 1
    assert event["extra_body_keys"] == ["provider"]
    assert "secret prompt text" not in log_path.read_text(encoding="utf-8")
    assert "sk-should-not-be-logged" not in log_path.read_text(encoding="utf-8")


def test_record_dispatch_is_fail_open(monkeypatch, tmp_path):
    bad_path = tmp_path / "dispatch.jsonl"
    bad_path.mkdir()
    monkeypatch.setattr(dispatch_logging, "dispatch_log_path", lambda: bad_path)

    dispatch_logging.record_dispatch({"kind": "will-not-raise"})


def test_interruptible_api_call_logs_before_openai_dispatch(monkeypatch, tmp_path):
    log_path = tmp_path / "dispatch.jsonl"
    monkeypatch.setattr(dispatch_logging, "dispatch_log_path", lambda: log_path)
    monkeypatch.setenv("HERMES_PROFILE", "default")

    class FakeCreate:
        def __init__(self):
            self.event_seen_in_create = None

        def create(self, **kwargs):
            assert log_path.exists(), "dispatch log must exist before provider call"
            self.event_seen_in_create = _read_events(log_path)[-1]
            return SimpleNamespace(choices=[])

    fake_create = FakeCreate()
    fake_client = SimpleNamespace(
        base_url="https://provider.example/v1?token=secret",
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create.create)),
    )

    class Agent:
        provider = "openrouter"
        api_mode = "chat_completions"
        model = "configured/model"
        base_url = "https://fallback.example/v1"
        session_id = "session-1"
        platform = "cli"
        _interrupt_requested = False

        def _create_request_openai_client(self, *, reason, api_kwargs):
            assert reason == "chat_completion_request"
            return fake_client

        def _touch_activity(self, _message):
            pass

        def _close_request_openai_client(self, _client, *, reason):
            pass

        def _abort_request_openai_client(self, _client, *, reason):
            pass

        def _compute_non_stream_stale_timeout(self, _api_kwargs):
            return 5.0

    response = cch.interruptible_api_call(
        Agent(),
        {"model": "resolved/model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.choices == []
    event = fake_create.event_seen_in_create
    assert event is not None
    assert event["dispatch_path"] == "chat.completions.create"
    assert event["seat"] == "default"
    assert event["provider"] == "openrouter"
    assert event["model"] == "resolved/model"
    assert event["config_claimed"] == "configured/model"
    assert len(event["prompt_hash"]) == 8
    assert event["base_url"] == "https://provider.example/v1"


def test_copilot_acp_logs_claude_code_invocation_before_popen(monkeypatch, tmp_path):
    log_path = tmp_path / "dispatch.jsonl"
    capture_dir = tmp_path / "cc_prompts"
    monkeypatch.setattr(dispatch_logging, "dispatch_log_path", lambda: log_path)
    monkeypatch.setattr(cc_prompt_capture, "cc_prompts_dir", lambda: capture_dir)
    monkeypatch.setenv("HERMES_PROFILE", "cc-builder")
    client = CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="claude",
        acp_args=["--acp", "--stdio", "--api-key", "secret", "--token=secret"],
        acp_cwd=str(tmp_path),
    )

    def fake_popen(cmd, **kwargs):
        assert cmd[:3] == ["claude", "--acp", "--stdio"]
        assert log_path.exists(), "dispatch log must exist before ACP process spawn"
        event = _read_events(log_path)[-1]
        assert event["kind"] == "claude_code_dispatch"
        assert event["dispatch_path"] == "acp.subprocess.Popen"
        assert event["seat"] == "cc-builder"
        assert event["provider"] == "claude-code"
        assert event["model"] == "claude-code-default"
        assert event["source"] == "default"
        assert event["config_claimed"] is None
        assert len(event["prompt_hash"]) == 8
        assert event["prompt_chars"] == len("hello from test")
        assert event["argv"] == [
            "claude",
            "--acp",
            "--stdio",
            "--api-key",
            "<redacted>",
            "--token=<redacted>",
        ]
        capture_path = capture_dir / f"{event['prompt_hash']}.json"
        assert capture_path.exists(), "cc-builder prompt capture must exist before ACP process spawn"
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        assert capture["prompt_hash"] == event["prompt_hash"]
        assert capture["prompt"] == "hello from test"
        raise FileNotFoundError("claude not found")

    monkeypatch.setattr("agent.copilot_acp_client.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
        client._run_prompt("hello from test", timeout_seconds=1)
