from __future__ import annotations

import json
from types import SimpleNamespace

import agent.cc_prompt_capture as cc_prompt_capture
import agent.dispatch_logging as dispatch_logging


def _capture_dir(tmp_path):
    return tmp_path / "logs" / "cc_prompts"


def _read_capture(tmp_path, prompt_hash):
    path = _capture_dir(tmp_path) / f"{prompt_hash}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_capture_cc_prompt_redacts_fake_anthropic_key(monkeypatch, tmp_path):
    monkeypatch.setattr(cc_prompt_capture, "cc_prompts_dir", lambda: _capture_dir(tmp_path))

    cc_prompt_capture.capture_cc_prompt(
        "please use sk-ant-FAKE123 only in this fake prompt",
        "abc123ef",
        {"seat": "cc-builder", "session_id": "session-1", "dispatch_path": "acp.subprocess.Popen"},
    )

    payload = _read_capture(tmp_path, "abc123ef")
    assert payload["prompt_hash"] == "abc123ef"
    assert payload["seat"] == "cc-builder"
    assert payload["session_id"] == "session-1"
    assert payload["capture_error"] is False
    assert payload["redaction_count"] == 1
    assert "sk-ant-FAKE123" not in payload["prompt"]
    assert "[REDACTED:anthropic_key]" in payload["prompt"]


def test_capture_cc_prompt_redacts_postgres_credential_string(monkeypatch, tmp_path):
    monkeypatch.setattr(cc_prompt_capture, "cc_prompts_dir", lambda: _capture_dir(tmp_path))

    cc_prompt_capture.capture_cc_prompt(
        "connect with postgres://user:pass@db.example/app for this test",
        "feedbabe",
        {"seat": "cc-builder"},
    )

    payload = _read_capture(tmp_path, "feedbabe")
    assert payload["capture_error"] is False
    assert payload["redaction_count"] == 1
    assert "postgres://user:pass@db.example/app" not in payload["prompt"]
    assert "[REDACTED:connection_string]" in payload["prompt"]


def test_capture_cc_prompt_truncates_at_configurable_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(cc_prompt_capture, "cc_prompts_dir", lambda: _capture_dir(tmp_path))
    monkeypatch.setenv("CC_PROMPT_CAP", "10")

    cc_prompt_capture.capture_cc_prompt("abcdefghijklmnop", "capped01", {"seat": "cc-builder"})

    payload = _read_capture(tmp_path, "capped01")
    assert payload["prompt_chars_original"] == 16
    assert payload["prompt_chars_stored"] == 10
    assert payload["truncated"] is True
    assert payload["prompt"] == "abcdefghij"


def test_capture_cc_prompt_redaction_failure_writes_metadata_only(monkeypatch, tmp_path):
    monkeypatch.setattr(cc_prompt_capture, "cc_prompts_dir", lambda: _capture_dir(tmp_path))

    def fail_redaction(_prompt_text):
        raise RuntimeError("redactor failed")

    monkeypatch.setattr(cc_prompt_capture, "_redact_prompt_text", fail_redaction)

    cc_prompt_capture.capture_cc_prompt("do not write this body", "err00001", {"seat": "cc-builder"})

    payload = _read_capture(tmp_path, "err00001")
    assert payload["capture_error"] is True
    assert payload["prompt_chars_original"] == len("do not write this body")
    assert payload["prompt_chars_stored"] == 0
    assert payload["redaction_count"] == 0
    assert "prompt" not in payload


def test_capture_cc_prompt_links_file_to_passed_prompt_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(cc_prompt_capture, "cc_prompts_dir", lambda: _capture_dir(tmp_path))

    cc_prompt_capture.capture_cc_prompt("hello", "deadbeef", {"seat": "cc-builder"})

    capture_path = _capture_dir(tmp_path) / "deadbeef.json"
    payload = json.loads(capture_path.read_text(encoding="utf-8"))
    assert capture_path.exists()
    assert payload["prompt_hash"] == "deadbeef"


def test_provider_llm_dispatch_does_not_create_cc_prompt_capture(monkeypatch, tmp_path):
    capture_dir = _capture_dir(tmp_path)
    monkeypatch.setattr(cc_prompt_capture, "cc_prompts_dir", lambda: capture_dir)
    monkeypatch.setattr(dispatch_logging, "dispatch_log_path", lambda: tmp_path / "dispatch.jsonl")

    agent = SimpleNamespace(
        provider="openai",
        api_mode="chat_completions",
        model="configured/provider-model",
        base_url="https://provider.example/v1",
        session_id="session-1",
        platform="cli",
    )
    dispatch_logging.log_llm_dispatch(
        agent,
        {"model": "resolved/provider-model", "messages": [{"role": "user", "content": "hash only"}]},
        dispatch_path="chat.completions.create",
    )

    assert (tmp_path / "dispatch.jsonl").exists()
    assert not capture_dir.exists()
