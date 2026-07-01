"""Fail-open full-prompt capture for Claude Code / cc-builder handoffs.

Provider LLM calls intentionally remain hash-only in dispatch.jsonl.  This module
is only for ACP/Claude Code subprocess prompts and must never affect dispatch:
all public entry points swallow their own errors, redact before writing, and
write metadata-only records if redaction cannot be confirmed.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CC_PROMPTS_DIR = _PROJECT_ROOT / "logs" / "cc_prompts"
_DEFAULT_PROMPT_CAP = 100_000

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?-----END [^-\n]*PRIVATE KEY-----",
    re.DOTALL | re.IGNORECASE,
)
_AUTHORIZATION_RE = re.compile(r"(?im)^(\s*Authorization\s*:\s*)[^\r\n]+")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{6,}")
_DATABASE_URL_RE = re.compile(
    r"(?i)\b(DATABASE_URL\s*=\s*)(?:['\"])?[^\s'\"<>]+"
)
_CONNECTION_STRING_RE = re.compile(
    r"(?i)\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss)://"
    r"[^\s'\"<>/@:]+:[^\s'\"<>/@]+@[^\s'\"<>]+"
)
_ANTHROPIC_KEY_RE = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{6,}\b")
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{6,}\b")
_GITHUB_TOKEN_RE = re.compile(r"\bgh[op]_[A-Za-z0-9_]{6,}\b")
_AWS_ACCESS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{8,}\b")
_SECRET_KV_RE = re.compile(
    r"(?im)\b(?P<key>(?:[A-Z0-9_.-]+[_\-.])?"
    r"(?:api[_-]?key|access[_-]?token|token|secret|password|passwd|pwd|key)"
    r"(?:[_\-.][A-Z0-9_.-]+)?)\s*=\s*"
    r"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s#\r\n]+)",
    re.IGNORECASE,
)
_HASH_RE = re.compile(r"[^A-Za-z0-9_.-]")


def cc_prompts_dir() -> Path:
    """Return the project-local directory for captured Claude Code prompts."""

    return _CC_PROMPTS_DIR


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _prompt_cap() -> int:
    raw = os.environ.get("CC_PROMPT_CAP", "").strip()
    if not raw:
        return _DEFAULT_PROMPT_CAP
    try:
        return max(0, int(raw))
    except Exception:
        return _DEFAULT_PROMPT_CAP


def _safe_prompt_hash(prompt_hash: Any) -> str | None:
    text = str(prompt_hash or "").strip()
    if not text:
        return None
    return _HASH_RE.sub("_", text)[:128] or None


def _default_seat() -> str:
    for name in ("HERMES_AGENT_SEAT", "HERMES_PROFILE", "HERMES_KANBAN_ASSIGNEE"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return "unknown"


def _default_session_id() -> str | None:
    for name in ("HERMES_SESSION_ID", "HERMES_SESSION_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _subn(text: str, pattern: re.Pattern[str], replacement: str | Callable[[re.Match[str]], str]) -> tuple[str, int]:
    return pattern.subn(replacement, text)


def _redact_secret_kv(match: re.Match[str]) -> str:
    return f"{match.group('key')}=[REDACTED:secret]"


def _redact_prompt_text(prompt_text: str) -> tuple[str, int, bool]:
    """Return redacted text, redaction count, and whether scanning completed."""

    redacted = prompt_text
    total = 0
    replacements: tuple[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]], ...] = (
        (_PRIVATE_KEY_RE, "[REDACTED:private_key]"),
        (_AUTHORIZATION_RE, r"\1[REDACTED:authorization]"),
        (_BEARER_RE, "[REDACTED:bearer_token]"),
        (_DATABASE_URL_RE, r"\1[REDACTED:connection_string]"),
        (_CONNECTION_STRING_RE, "[REDACTED:connection_string]"),
        (_ANTHROPIC_KEY_RE, "[REDACTED:anthropic_key]"),
        (_OPENAI_KEY_RE, "[REDACTED:openai_key]"),
        (_GITHUB_TOKEN_RE, "[REDACTED:github_token]"),
        (_AWS_ACCESS_KEY_RE, "[REDACTED:aws_access_key]"),
        (_SECRET_KV_RE, _redact_secret_kv),
    )
    for pattern, replacement in replacements:
        redacted, count = _subn(redacted, pattern, replacement)
        total += count
    return redacted, total, True


def capture_cc_prompt(
    prompt_text: str | None,
    prompt_hash: str,
    meta: Mapping[str, Any] | None = None,
) -> None:
    """Capture a redacted Claude Code handoff prompt, swallowing all failures."""

    try:
        safe_hash = _safe_prompt_hash(prompt_hash)
        if not safe_hash:
            return

        metadata = dict(meta or {})
        path = cc_prompts_dir() / f"{safe_hash}.json"
        if path.exists():
            return

        prompt_is_text = isinstance(prompt_text, str)
        original_prompt = prompt_text if prompt_is_text else ""
        original_chars = len(original_prompt)
        redaction_count = 0
        capture_error = False
        truncated = False
        stored_prompt = ""

        if not prompt_is_text:
            capture_error = True
        else:
            try:
                redacted_prompt, redaction_count, scanned = _redact_prompt_text(original_prompt)
                if not scanned:
                    capture_error = True
                else:
                    cap = _prompt_cap()
                    truncated = len(redacted_prompt) > cap
                    stored_prompt = redacted_prompt[:cap] if truncated else redacted_prompt
            except Exception:
                capture_error = True

        payload: dict[str, Any] = {
            "ts": _now_iso(),
            "seat": metadata.get("seat") or _default_seat(),
            "session_id": metadata.get("session_id") or _default_session_id(),
            "dispatch_path": metadata.get("dispatch_path") or "acp.subprocess.Popen",
            "prompt_hash": safe_hash,
            "prompt_chars_original": original_chars,
            "prompt_chars_stored": len(stored_prompt) if not capture_error else 0,
            "truncated": bool(truncated and not capture_error),
            "redaction_count": redaction_count if not capture_error else 0,
            "capture_error": bool(capture_error),
        }
        for key, value in metadata.items():
            if key not in payload and value not in (None, ""):
                payload[key] = value
        if not capture_error:
            payload["prompt"] = stored_prompt

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True, default=str)
                handle.write("\n")
        except FileExistsError:
            return
    except Exception:
        return


__all__ = [
    "capture_cc_prompt",
    "cc_prompts_dir",
]
