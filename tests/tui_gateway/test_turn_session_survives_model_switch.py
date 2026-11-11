"""A model switch applied at turn start must not blank the turn's session identity.

Bug (2026-11-11): ``_prepare_turn_input`` binds the turn's session context, then applies a
pending model switch, whose commit calls ``_persist_live_session_system_prompt``. That helper
re-bound the session and cleared it in a ``finally``; ``clear_session_vars`` is not nestable
and sets every var to ``""``, so the rest of the turn ran with no ``HERMES_SESSION_KEY`` or
``HERMES_UI_SESSION_ID``. Tool hooks keyed on them (the plan-mode plugin) then failed closed.
"""
from __future__ import annotations

import contextvars

from gateway.session_context import get_session_env
from tui_gateway import server


class _FakeAgent:
    model = "m"
    provider = "p"
    session_id = "durable-id"
    _cached_system_prompt = None

    def __init__(self):
        self.seen_key = None

        class _DB:
            def update_system_prompt(self, session_id, prompt):
                pass

        self._session_db = _DB()

    def _build_system_prompt(self, system_message=None):
        self.seen_key = get_session_env("HERMES_SESSION_KEY")
        return "prompt"


def _run_in_fresh_context(fn):
    return contextvars.Context().run(fn)


def test_persist_system_prompt_inside_a_turn_keeps_the_turn_identity():
    agent = _FakeAgent()
    session = {"agent": agent, "session_key": "turn-key"}

    def turn():
        tokens = server._set_session_context("turn-key", ui_session_id="tab-1")
        try:
            server._persist_live_session_system_prompt(session)
            return (get_session_env("HERMES_SESSION_KEY"), get_session_env("HERMES_UI_SESSION_ID"))
        finally:
            server._clear_session_context(tokens)

    assert _run_in_fresh_context(turn) == ("turn-key", "tab-1")
    assert agent.seen_key == "turn-key"


def test_persist_system_prompt_off_turn_still_binds_and_clears():
    agent = _FakeAgent()
    session = {"agent": agent, "session_key": "rpc-key"}

    def rpc():
        server._persist_live_session_system_prompt(session)
        return get_session_env("HERMES_SESSION_KEY", "unset")

    assert _run_in_fresh_context(rpc) == ""
    assert agent.seen_key == "rpc-key"


def test_a_different_bound_session_is_not_reused():
    agent = _FakeAgent()
    session = {"agent": agent, "session_key": "other-key"}

    def turn():
        tokens = server._set_session_context("turn-key", ui_session_id="tab-1")
        try:
            server._persist_live_session_system_prompt(session)
        finally:
            server._clear_session_context(tokens)

    _run_in_fresh_context(turn)
    assert agent.seen_key == "other-key"


def test_bot_capability_refresh_inside_a_turn_keeps_the_turn_identity(monkeypatch):
    import tools.bot_mode_probe as probe

    agent = _FakeAgent()
    agent._session_title_hint = "Bot Chat"
    rebuilt = _FakeAgent()
    session = {"agent": agent, "session_key": "turn-key", "bot_caps_seen": "old"}
    monkeypatch.setattr(probe, "capability_fingerprint", lambda home: "new")
    monkeypatch.setattr(server, "_rebuild_session_agent", lambda *a, **k: rebuilt)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)

    def turn():
        tokens = server._set_session_context("turn-key", ui_session_id="tab-1")
        try:
            server._sync_bot_capabilities("tab-1", session)
            return (get_session_env("HERMES_SESSION_KEY"), get_session_env("HERMES_UI_SESSION_ID"))
        finally:
            server._clear_session_context(tokens)

    assert _run_in_fresh_context(turn) == ("turn-key", "tab-1")
    assert getattr(rebuilt, "_session_title_hint", "") == "Bot Chat"
