"""Tests for the dependency-free adapter envelope (hermes_cli/jobs_contract).

The envelope is the minimum, JSON-safe, immutable value object every future
specialist adapter will accept. These pin the required-field and
secret-rejection contract now so adapters land against a stable boundary — no
adapter, engine, or Kanban coupling is exercised or implied here.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli.jobs_contract import JobEnvelope


def _valid(**over):
    base = dict(
        job_id="j_1",
        number=7,
        name="Ship it",
        goal="do the thing",
        specialist="claude-builder",
        routing_reason="repo scope",
        attempt_id="a_1",
        ordinal=1,
        claim_token="tok-abc",
        repository="/repo",
        branch="feat/x",
        worktree="/wt",
        commit="abc123",
        parent_attempt_id=None,
        metadata={"lane": "build"},
    )
    base.update(over)
    return JobEnvelope(**base)


def test_envelope_roundtrips_json_safely():
    env = _valid()
    d = env.to_dict()
    assert d["number"] == 7 and d["ordinal"] == 1
    parsed = json.loads(env.to_json())
    assert parsed["job_id"] == "j_1"
    assert parsed["metadata"]["lane"] == "build"
    assert parsed["parent_attempt_id"] is None


def test_repr_redacts_the_claim_token():
    """Any adapter debug log or traceback holding an envelope must be safe."""
    env = _valid(claim_token="SUPER-SECRET-CAPABILITY")
    assert "SUPER-SECRET-CAPABILITY" not in repr(env)
    assert "j_1" in repr(env)  # everything non-secret still debuggable


def test_ordinary_serialization_omits_the_claim_token():
    env = _valid(claim_token="SUPER-SECRET-CAPABILITY")
    assert "claim_token" not in env.to_dict()
    assert "SUPER-SECRET-CAPABILITY" not in env.to_json()
    assert "SUPER-SECRET-CAPABILITY" not in str(env)


def test_transport_serialization_carries_the_claim_token():
    """The envelope IS the adapter's credential — but only when asked."""
    env = _valid(claim_token="tok-abc")
    assert env.to_dict(include_claim_token=True)["claim_token"] == "tok-abc"
    assert json.loads(env.to_json(include_claim_token=True))["claim_token"] == "tok-abc"
    # The field itself is still readable; only the serializations are redacted.
    assert env.claim_token == "tok-abc"


def test_envelope_is_immutable():
    env = _valid()
    with pytest.raises(Exception):
        env.number = 9  # frozen dataclass


def test_envelope_metadata_copy_is_defensive():
    src = {"lane": "build"}
    env = _valid(metadata=src)
    src["lane"] = "mutated"
    assert env.metadata["lane"] == "build"  # envelope kept its own copy


@pytest.mark.parametrize("missing", ["job_id", "name", "goal", "attempt_id", "claim_token"])
def test_envelope_requires_core_string_fields(missing):
    with pytest.raises(ValueError):
        _valid(**{missing: ""})


@pytest.mark.parametrize("field_name", ["number", "ordinal"])
def test_envelope_requires_integer_fields(field_name):
    with pytest.raises(ValueError):
        _valid(**{field_name: None})
    with pytest.raises(ValueError):
        _valid(**{field_name: "1"})


@pytest.mark.parametrize(
    "key",
    ["password", "API_KEY", "Authorization", "db_secret", "auth_token", "MY_TOKEN"],
)
def test_envelope_rejects_secret_metadata_keys(key):
    with pytest.raises(ValueError):
        _valid(metadata={key: "x"})


def test_envelope_does_not_scan_or_rewrite_goal():
    # A goal that literally mentions a secret word is fine — the policy applies
    # to metadata KEYS only; the goal is never scanned or rewritten.
    goal = "reset the admin password and rotate the api_key"
    env = _valid(goal=goal)
    assert env.goal == goal


def test_envelope_allows_benign_metadata():
    env = _valid(metadata={"lane": "build", "attempt": 2})
    assert env.metadata["lane"] == "build" and env.metadata["attempt"] == 2


@pytest.mark.parametrize(
    "meta",
    [
        {"outer": {"api_key": "x"}},
        {"a": {"b": {"c": {"authorization": "x"}}}},
        {"list": [{"ok": 1}, {"db_secret": "x"}]},
    ],
)
def test_envelope_rejects_nested_secret_metadata_keys(meta):
    # A secret key buried inside a nested dict (or a dict inside a list) must not
    # slip past the policy — validation recurses through the whole structure.
    with pytest.raises(ValueError):
        _valid(metadata=meta)


def test_envelope_metadata_deep_copy_is_defensive():
    src = {"outer": {"lane": "build"}}
    env = _valid(metadata=src)
    src["outer"]["lane"] = "mutated"  # mutate a NESTED dict after construction
    assert env.metadata["outer"]["lane"] == "build"  # envelope kept a deep copy


def test_envelope_metadata_is_deeply_immutable():
    env = _valid(metadata={"outer": {"lane": "build"}})
    with pytest.raises(Exception):
        env.metadata["outer"]["lane"] = "x"  # nested mapping is read-only too


def test_envelope_rejects_non_json_safe_metadata_values():
    with pytest.raises(ValueError):
        _valid(metadata={"bad": {1, 2, 3}})  # a set is not JSON-safe
