"""Canonical, provider-explicit identity for Jobs execution lanes.

The public choice (``claude`` or ``codex``) resolves once into an immutable
four-field identity.  Physical capacity seats are deliberately absent: a
``lane_id`` such as ``codex-pc-1`` is a placement, not the user's request.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CLAUDE_REQUESTED_LANE = "claude"
CLAUDE_EXECUTOR = "claude"
CLAUDE_SPECIALIST = "claude-builder"
CLAUDE_MODEL = "claude-opus-5"

CODEX_REQUESTED_LANE = "codex"
CODEX_EXECUTOR = "codex"
CODEX_SPECIALIST = "codex-builder"
CODEX_MODEL = "gpt-5.6-sol"

# Read-only migration alias.  New writes always use ``codex-builder``.
LEGACY_GPT_SPECIALIST = "gpt-builder"

REQUESTED_LANES = (CLAUDE_REQUESTED_LANE, CODEX_REQUESTED_LANE)


class UnsupportedJobLane(ValueError):
    """A requested or persisted identity has no approved Jobs lane."""


@dataclass(frozen=True)
class JobIdentity:
    requested_lane: str
    executor: str
    specialist: str
    model: str


_BY_LANE = {
    CLAUDE_REQUESTED_LANE: JobIdentity(
        requested_lane=CLAUDE_REQUESTED_LANE,
        executor=CLAUDE_EXECUTOR,
        specialist=CLAUDE_SPECIALIST,
        model=CLAUDE_MODEL,
    ),
    CODEX_REQUESTED_LANE: JobIdentity(
        requested_lane=CODEX_REQUESTED_LANE,
        executor=CODEX_EXECUTOR,
        specialist=CODEX_SPECIALIST,
        model=CODEX_MODEL,
    ),
}

_LANE_BY_EXECUTOR = {identity.executor: lane for lane, identity in _BY_LANE.items()}
_LANE_BY_SPECIALIST = {
    CLAUDE_SPECIALIST: CLAUDE_REQUESTED_LANE,
    CODEX_SPECIALIST: CODEX_REQUESTED_LANE,
    LEGACY_GPT_SPECIALIST: CODEX_REQUESTED_LANE,
}
_LANE_BY_MODEL = {identity.model: lane for lane, identity in _BY_LANE.items()}


def resolve_requested_lane(requested_lane: object) -> JobIdentity:
    """Resolve the only two supported public lane names, or fail closed."""
    if not isinstance(requested_lane, str) or not requested_lane.strip():
        raise UnsupportedJobLane("job lane must be exactly 'claude' or 'codex'")
    lane = requested_lane.strip()
    try:
        return _BY_LANE[lane]
    except KeyError as exc:
        raise UnsupportedJobLane(
            f"unsupported job lane {requested_lane!r}; expected 'claude' or 'codex'"
        ) from exc


def _field(job: object, name: str) -> Any:
    if isinstance(job, Mapping):
        return job.get(name)
    return getattr(job, name, None)


def effective_identity(job: object) -> JobIdentity:
    """Normalize a persisted Job identity without rewriting historical rows.

    Nullable identity columns identify pre-migration history.  A historical
    ``claude-builder``, ``codex-builder``, or ``gpt-builder`` specialist can
    therefore supply the missing identity.  KAT, an unassigned row, an unknown
    value, or fields that disagree all fail closed; none silently become Claude.
    """
    values = {
        "requested_lane": _field(job, "requested_lane"),
        "executor": _field(job, "executor"),
        "specialist": _field(job, "specialist"),
        "model": _field(job, "model"),
    }
    candidates = []
    for field, lookup in (
        ("requested_lane", _BY_LANE),
        ("executor", _LANE_BY_EXECUTOR),
        ("specialist", _LANE_BY_SPECIALIST),
        ("model", _LANE_BY_MODEL),
    ):
        value = values[field]
        if value is None:
            continue
        if not isinstance(value, str) or value not in lookup:
            raise UnsupportedJobLane(
                f"unsupported persisted Job {field} {value!r}"
            )
        lane = value if field == "requested_lane" else lookup[value]
        candidates.append(lane)

    if not candidates:
        raise UnsupportedJobLane("Job has no explicit supported execution identity")
    if len(set(candidates)) != 1:
        raise UnsupportedJobLane("persisted Job identity fields are contradictory")

    identity = _BY_LANE[candidates[0]]
    expected = {
        "requested_lane": identity.requested_lane,
        "executor": identity.executor,
        "specialist": identity.specialist,
        "model": identity.model,
    }
    for field, value in values.items():
        if value is None:
            continue
        if field == "specialist" and value == LEGACY_GPT_SPECIALIST:
            continue
        if value != expected[field]:
            raise UnsupportedJobLane("persisted Job identity fields are contradictory")
    return identity
