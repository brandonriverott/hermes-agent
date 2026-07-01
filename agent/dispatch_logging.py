"""Fail-open dispatch logging for outbound model/agent calls.

The log is intentionally tiny and payload-safe: it records where Hermes is
about to dispatch work, never prompt text, API keys, headers, or full request
bodies.  This module must never break the hot path — every public helper
swallows its own errors.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DISPATCH_LOG_PATH = _PROJECT_ROOT / "logs" / "dispatch.jsonl"

_SECRET_ARG_MARKERS = (
    "api-key",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)

_MODEL_ENV_VARS = (
    "HERMES_INFERENCE_MODEL",
    "HERMES_MODEL",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    "CLAUDE_CODE_MODEL",
    "HERMES_COPILOT_ACP_MODEL",
)


def dispatch_log_path() -> Path:
    """Return the project-local dispatch JSONL path."""

    return _DISPATCH_LOG_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_url(raw: Any) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme and parsed.netloc:
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    except Exception:
        pass
    return value[:300]


def _safe_host(raw: Any) -> str | None:
    if raw is None:
        return None
    try:
        parsed = urlsplit(str(raw))
        if parsed.netloc:
            return parsed.netloc.lower()
    except Exception:
        return None
    return None


def _compact_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _len_if_list(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else None


def _char_count(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return len(str(value))


def _prompt_hash_from_value(value: Any) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        rendered = str(value or "")
    return hashlib.sha256(rendered.encode("utf-8", errors="replace")).hexdigest()[:8]


def prompt_hash_from_value(value: Any) -> str:
    """Return the prompt hash format used in dispatch logs."""

    return _prompt_hash_from_value(value)


def _prompt_hash_for_kwargs(api_kwargs: Mapping[str, Any]) -> str:
    prompt_parts: dict[str, Any] = {}
    for key in ("messages", "input", "prompt"):
        if key in api_kwargs:
            prompt_parts[key] = api_kwargs.get(key)
    return _prompt_hash_from_value(prompt_parts)


def _env_model_source(model: Any) -> str | None:
    model_text = _compact_str(model)
    if not model_text:
        return None
    for name in _MODEL_ENV_VARS:
        if os.environ.get(name, "").strip() == model_text:
            return f"env:{name}"
    return None


def _model_source(model: Any, config_model: Any) -> str:
    env_source = _env_model_source(model)
    if env_source:
        return env_source
    model_text = _compact_str(model)
    config_text = _compact_str(config_model)
    if model_text and config_text and model_text == config_text:
        return "config"
    return "unknown"


def _dispatch_seat(agent: Any = None) -> str:
    for attr in (
        "seat",
        "agent_seat",
        "_dispatch_seat",
        "profile_name",
        "_profile_name",
        "agent_identity",
        "_agent_identity",
    ):
        value = _compact_str(getattr(agent, attr, None)) if agent is not None else None
        if value:
            return value
    for name in ("HERMES_AGENT_SEAT", "HERMES_PROFILE", "HERMES_KANBAN_ASSIGNEE"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        from hermes_cli.profiles import get_active_profile_name

        value = _compact_str(get_active_profile_name())
        if value:
            return value
    except Exception:
        pass
    return "unknown"


def _normalize_provider(provider: Any, *, api_mode: Any = None, base_url: Any = None) -> str:
    raw_provider = str(provider or "").strip().lower()
    raw_mode = str(api_mode or "").strip().lower()
    host = (_safe_host(base_url) or "").lower()

    if raw_mode == "codex_responses" or "codex" in raw_provider:
        return "openai-codex"
    if raw_mode == "anthropic_messages" or "anthropic" in raw_provider or "anthropic" in host:
        return "anthropic"
    if "bedrock" in raw_provider:
        return "bedrock"
    if raw_provider == "moa":
        return "moa"
    if "openrouter" in raw_provider or "openrouter" in host:
        return "openrouter"
    if "nvidia" in raw_provider or "nvidia" in host:
        return "nvidia"
    if "deepseek" in raw_provider or "deepseek" in host:
        return "deepseek"
    if "zai" in raw_provider or "bigmodel" in raw_provider or "bigmodel" in host or "z.ai" in host:
        return "zai"
    if raw_provider in {"openai-api", "openai", "openai-compatible"}:
        return "openai"
    return raw_provider or "unknown"


def _payload_shape(api_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    messages = api_kwargs.get("messages")
    input_payload = api_kwargs.get("input")
    tools = api_kwargs.get("tools")
    extra_body = api_kwargs.get("extra_body")
    shape = {
        "request_keys": sorted(str(k) for k in api_kwargs.keys() if not _looks_secret(str(k))),
        "message_count": _len_if_list(messages),
        "input_count": _len_if_list(input_payload),
        "tool_count": _len_if_list(tools),
        "message_chars": _char_count(messages) if messages is not None else None,
        "input_chars": _char_count(input_payload) if input_payload is not None else None,
        "tool_chars": _char_count(tools) if tools is not None else None,
    }
    if isinstance(extra_body, Mapping):
        shape["extra_body_keys"] = sorted(str(k) for k in extra_body.keys())
    return {k: v for k, v in shape.items() if v is not None}


def _looks_secret(value: str) -> bool:
    lower = value.lower()
    return any(marker in lower for marker in _SECRET_ARG_MARKERS)


def _redact_arg_value(arg: Any) -> str:
    text = str(arg)
    if not _looks_secret(text):
        return text
    if "=" in text:
        key, _sep, _value = text.partition("=")
        return f"{key}=<redacted>"
    if text.startswith("-"):
        return text
    return "<redacted>"


def _redact_argv(args: list[Any]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for arg in args:
        text = str(arg)
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        redacted.append(_redact_arg_value(text))
        if _looks_secret(text) and "=" not in text:
            redact_next = True
    return redacted


def _extract_model_from_argv(args: list[Any]) -> str | None:
    for idx, arg in enumerate(args):
        text = str(arg)
        if text in {"--model", "-m"} and idx + 1 < len(args):
            return _compact_str(args[idx + 1])
        if text.startswith("--model="):
            return _compact_str(text.partition("=")[2])
    return None


def _client_base_url(client: Any) -> str | None:
    for attr in ("base_url", "_base_url"):
        value = getattr(client, attr, None)
        if value:
            return str(value)
    return None


def record_dispatch(event: Mapping[str, Any]) -> None:
    """Append one dispatch event as JSONL, swallowing all failures."""

    try:
        payload = dict(event)
        payload.setdefault("ts", _now_iso())
        path = dispatch_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
            handle.write("\n")
    except Exception:
        return


def log_llm_dispatch(
    agent: Any,
    api_kwargs: Mapping[str, Any],
    *,
    dispatch_path: str,
    client: Any = None,
    stream: bool | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Log an outbound LLM provider dispatch immediately before the call."""

    try:
        request_model = api_kwargs.get("model") if isinstance(api_kwargs, Mapping) else None
        configured_model = getattr(agent, "model", None)
        client_base_url = _client_base_url(client)
        base_url = client_base_url or getattr(agent, "base_url", None)
        api_mode = getattr(agent, "api_mode", None)
        provider = _normalize_provider(getattr(agent, "provider", None), api_mode=api_mode, base_url=base_url)
        event: dict[str, Any] = {
            "kind": "llm_dispatch",
            "dispatch_path": dispatch_path,
            "seat": _dispatch_seat(agent),
            "provider": provider,
            "api_mode": api_mode,
            "model": request_model or configured_model or "unknown",
            "source": _model_source(request_model, configured_model),
            "config_claimed": configured_model,
            "agent_model": configured_model,
            "base_url": _safe_url(base_url),
            "base_url_host": _safe_host(base_url),
            "prompt_hash": _prompt_hash_for_kwargs(api_kwargs),
            "session_id": getattr(agent, "session_id", None),
            "platform": getattr(agent, "platform", None),
            "stream": bool(api_kwargs.get("stream")) if stream is None else bool(stream),
        }
        event.update(_payload_shape(api_kwargs))
        if extra:
            event["extra"] = dict(extra)
        record_dispatch({k: v for k, v in event.items() if v not in (None, "")})
    except Exception:
        return


def log_acp_subprocess_dispatch(
    *,
    command: str,
    args: list[str],
    cwd: str | None,
    prompt_text: str | None = None,
    model: str | None = None,
    prompt_hash: str | None = None,
) -> None:
    """Log an ACP/Claude Code subprocess invocation before ``Popen``."""

    try:
        command_name = Path(str(command)).name
        argv = [str(command)] + [str(arg) for arg in args]
        is_claude = "claude" in command_name.lower()
        explicit_model = _extract_model_from_argv(args)
        env_model = next((os.environ.get(name, "").strip() for name in _MODEL_ENV_VARS if os.environ.get(name, "").strip()), "")
        resolved_model = explicit_model or _compact_str(model) or env_model or (
            "claude-code-default" if is_claude else "acp-default"
        )
        resolved_prompt_hash = _compact_str(prompt_hash) or _prompt_hash_from_value(prompt_text or "")
        if explicit_model:
            source = _env_model_source(explicit_model) or "config"
        elif _compact_str(model):
            source = _env_model_source(model) or "config"
        elif env_model:
            source = _env_model_source(env_model) or "unknown"
        else:
            source = "default"
        record_dispatch(
            {
                "kind": "claude_code_dispatch" if is_claude else "acp_subprocess_dispatch",
                "dispatch_path": "acp.subprocess.Popen",
                "seat": _dispatch_seat(),
                "provider": "claude-code" if is_claude else "acp-subprocess",
                "model": resolved_model,
                "source": source,
                "config_claimed": _compact_str(model),
                "prompt_hash": resolved_prompt_hash,
                "command": str(command),
                "command_name": command_name,
                "argv": _redact_argv(argv),
                "cwd": cwd,
                "prompt_chars": len(prompt_text) if isinstance(prompt_text, str) else None,
            }
        )
    except Exception:
        return


__all__ = [
    "dispatch_log_path",
    "log_acp_subprocess_dispatch",
    "log_llm_dispatch",
    "prompt_hash_from_value",
    "record_dispatch",
]
