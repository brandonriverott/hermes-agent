"""Pure tests for the canonical Jobs executor identity contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.jobs_identity import (
    JobIdentity,
    UnsupportedJobLane,
    effective_identity,
    resolve_requested_lane,
)


@pytest.mark.parametrize(
    ("lane", "expected"),
    [
        (
            "claude",
            JobIdentity(
                requested_lane="claude",
                executor="claude",
                specialist="claude-builder",
                model="claude-opus-5",
            ),
        ),
        (
            "codex",
            JobIdentity(
                requested_lane="codex",
                executor="codex",
                specialist="codex-builder",
                model="gpt-5.6-sol",
            ),
        ),
    ],
)
def test_requested_lane_resolves_to_one_frozen_identity(lane, expected):
    identity = resolve_requested_lane(lane)
    assert identity == expected
    with pytest.raises(Exception):
        identity.model = "different"  # type: ignore[misc]


@pytest.mark.parametrize(
    "lane", [None, "", "   ", "kat", "kat-builder", "gpt", "unknown"]
)
def test_requested_lane_rejects_every_non_public_lane(lane):
    with pytest.raises(UnsupportedJobLane):
        resolve_requested_lane(lane)


@pytest.mark.parametrize("specialist", ["gpt-builder", "codex-builder"])
def test_historical_and_canonical_codex_specialists_normalize(specialist):
    job = SimpleNamespace(
        requested_lane=None,
        executor=None,
        specialist=specialist,
        model=None,
    )
    assert effective_identity(job) == resolve_requested_lane("codex")


def test_historical_claude_specialist_normalizes():
    job = SimpleNamespace(
        requested_lane=None,
        executor=None,
        specialist="claude-builder",
        model=None,
    )
    assert effective_identity(job) == resolve_requested_lane("claude")


@pytest.mark.parametrize("specialist", [None, "kat-builder", "graph-engineer"])
def test_effective_identity_has_no_implicit_claude_or_kat_route(specialist):
    job = SimpleNamespace(
        requested_lane=None,
        executor=None,
        specialist=specialist,
        model=None,
    )
    with pytest.raises(UnsupportedJobLane):
        effective_identity(job)


def test_effective_identity_rejects_contradictory_persisted_fields():
    job = SimpleNamespace(
        requested_lane="codex",
        executor="claude",
        specialist="codex-builder",
        model="gpt-5.6-sol",
    )
    with pytest.raises(UnsupportedJobLane, match="contradictory"):
        effective_identity(job)
