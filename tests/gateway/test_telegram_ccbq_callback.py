"""Tests for the ccbq: (cc-builder quarantine Approve/Deny) callback branch.

The buttons are emitted by the external cc-builder watchdog; the adapter's job
is: authorize the tapper, hand the raw callback_data to the external
cc-helper/telegram_approval.py module, and reflect the verdict in the message.
These tests exercise the REAL wiring (importlib load of a helper file) via the
CC_TELEGRAM_APPROVAL_HELPER override.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from tests.gateway.test_telegram_approval_buttons import _ensure_telegram_mock, _make_adapter

_ensure_telegram_mock()


FAKE_HELPER_OK = """
CALLS = []

def handle_telegram_callback(callback_data):
    CALLS.append(callback_data)
    return {"ok": True, "event_id": "ev123", "action": "approve",
            "message": "approved; artifact copied for review only"}
"""

FAKE_HELPER_DENY = """
def handle_telegram_callback(callback_data):
    return {"ok": True, "event_id": "ev123", "action": "deny",
            "message": "denied; quarantined artifact left untouched"}
"""


def _make_query(data: str):
    query = MagicMock()
    query.data = data
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    query.from_user = SimpleNamespace(id=381798010, first_name="Brandon")
    query.message = SimpleNamespace(chat_id=381798010, chat=SimpleNamespace(type="private"), message_thread_id=None)
    return query


def _make_update(query):
    return SimpleNamespace(callback_query=query)


@pytest.mark.asyncio
async def test_ccbq_approve_authorized(tmp_path, monkeypatch):
    helper = tmp_path / "telegram_approval.py"
    helper.write_text(FAKE_HELPER_OK)
    monkeypatch.setenv("CC_TELEGRAM_APPROVAL_HELPER", str(helper))

    adapter = _make_adapter()
    query = _make_query("ccbq:ev123:approve")
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_callback_query(_make_update(query), None)

    query.answer.assert_awaited_once()
    assert "Approved" in query.answer.await_args.kwargs.get("text", query.answer.await_args.args[0] if query.answer.await_args.args else "")
    query.edit_message_text.assert_awaited_once()
    assert query.edit_message_text.await_args.kwargs.get("reply_markup") is None
    assert "ev123" in query.edit_message_text.await_args.kwargs.get("text", "")


@pytest.mark.asyncio
async def test_ccbq_deny_authorized(tmp_path, monkeypatch):
    helper = tmp_path / "telegram_approval.py"
    helper.write_text(FAKE_HELPER_DENY)
    monkeypatch.setenv("CC_TELEGRAM_APPROVAL_HELPER", str(helper))

    adapter = _make_adapter()
    query = _make_query("ccbq:ev123:deny")
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_callback_query(_make_update(query), None)

    text = query.answer.await_args.kwargs.get("text", "")
    assert "Denied" in text


@pytest.mark.asyncio
async def test_ccbq_unauthorized_never_touches_helper(tmp_path, monkeypatch):
    helper = tmp_path / "telegram_approval.py"
    helper.write_text("def handle_telegram_callback(d):\n    raise AssertionError('must not be called')\n")
    monkeypatch.setenv("CC_TELEGRAM_APPROVAL_HELPER", str(helper))

    adapter = _make_adapter()
    query = _make_query("ccbq:ev123:approve")
    with patch.object(adapter, "_is_callback_user_authorized", return_value=False):
        await adapter._handle_callback_query(_make_update(query), None)

    text = query.answer.await_args.kwargs.get("text", "")
    assert "not authorized" in text
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_ccbq_missing_helper_reports_failure(monkeypatch):
    monkeypatch.setenv("CC_TELEGRAM_APPROVAL_HELPER", "/nonexistent/telegram_approval.py")

    adapter = _make_adapter()
    query = _make_query("ccbq:ev123:approve")
    with patch.object(adapter, "_is_callback_user_authorized", return_value=True):
        await adapter._handle_callback_query(_make_update(query), None)

    text = query.answer.await_args.kwargs.get("text", "")
    assert "Could not record verdict" in text
