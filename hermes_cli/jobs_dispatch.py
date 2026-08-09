"""Provider-free Jobs dispatch decisions.

This module begins at the control-plane choke point that stranded Job 71: the
legacy ``KEY=value`` execution header embedded in a Job's verbatim goal.  The
live shell runner tokenizes values with ``\\S+`` and therefore discards a valid
repository path when it contains spaces.  Parsing lives here so it is a pure,
tested contract rather than transient shell behavior.

No function in this module reads the Jobs database, probes a host, or starts a
worker.  Those orchestration seams are layered on after the compatibility
metadata is trustworthy.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


LEGACY_KEYS = frozenset(
    {
        "REPO_PATH",
        "BASE_COMMIT",
        "MODEL",
        "EFFORT",
        "MAX_TURNS",
        "WORKSPACE_KIND",
        "NO_REROUTE",
    }
)
EFFORT_TIERS = frozenset({"low", "medium", "high", "xhigh", "max"})
WORKSPACE_KINDS = frozenset({"worktree"})
MAX_TURNS_CEILING = 500

_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_HEADER_KEY_RE = re.compile(r"\A[A-Z][A-Z0-9_]*\Z")


class LegacyMetadataError(ValueError):
    """A typed refusal that a dispatcher can persist and display."""

    def __init__(self, code: str, key: str, detail: str):
        super().__init__(detail)
        self.code = str(code)
        self.key = str(key)
        self.detail = str(detail)


@dataclass(frozen=True)
class LegacyExecutionHeader:
    """Validated execution inputs recovered from an existing Job goal."""

    repo_path: str
    base_commit: str
    model: str
    effort: str
    max_turns: int
    workspace_kind: str
    no_reroute: bool
    source_digest: str


def _refuse(code: str, key: str, detail: str) -> None:
    raise LegacyMetadataError(code, key, detail)


def _header_values(goal: str) -> dict[str, str]:
    """Read the contiguous goal header without tokenizing its values."""

    values: dict[str, str] = {}
    started = False

    for raw_line in goal.splitlines():
        if not raw_line.strip():
            if started:
                break
            continue

        raw_key, marker, raw_value = raw_line.partition("=")
        key = raw_key.strip()
        if not marker:
            if started:
                break
            continue

        if key not in LEGACY_KEYS:
            if _HEADER_KEY_RE.fullmatch(key):
                _refuse("unknown_key", key, f"unsupported metadata key {key}")
            if started:
                break
            continue

        started = True
        if key in values:
            _refuse("duplicate_key", key, f"duplicate metadata key {key}")

        value = raw_value.strip()
        if not value:
            _refuse("blank_value", key, f"metadata key {key} is blank")
        values[key] = value

    return values


def _required(values: dict[str, str], key: str) -> str:
    try:
        return values[key]
    except KeyError:
        _refuse("missing_key", key, f"required metadata key {key} is missing")
        raise AssertionError("unreachable")


def _max_turns(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        _refuse("invalid_integer", "MAX_TURNS", "MAX_TURNS must be an integer")
        raise AssertionError("unreachable")
    if value <= 0 or value > MAX_TURNS_CEILING:
        _refuse(
            "out_of_bounds",
            "MAX_TURNS",
            f"MAX_TURNS must be in 1..{MAX_TURNS_CEILING}",
        )
    return value


def _truthy(raw: str) -> bool:
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def parse_legacy_execution_header(goal: str) -> LegacyExecutionHeader:
    """Parse and validate the legacy execution header from ``goal``.

    The complete trimmed right-hand side is the value.  In particular,
    ``REPO_PATH=/Users/brandon/Documents/Kava Bar Scan`` remains that complete
    absolute path rather than being discarded or truncated at the first space.
    The original goal is never rewritten.
    """

    source_goal = str(goal or "")
    values = _header_values(source_goal)

    repo_path = _required(values, "REPO_PATH")
    if not Path(repo_path).is_absolute():
        _refuse(
            "invalid_repo_path",
            "REPO_PATH",
            f"REPO_PATH must be absolute: {repo_path!r}",
        )

    base_commit = _required(values, "BASE_COMMIT")
    if not _SHA_RE.fullmatch(base_commit):
        _refuse(
            "invalid_base_commit",
            "BASE_COMMIT",
            "BASE_COMMIT must be a full lowercase 40-character commit SHA",
        )

    effort = values.get("EFFORT", "max")
    if effort not in EFFORT_TIERS:
        _refuse(
            "unsupported_value",
            "EFFORT",
            f"unsupported EFFORT value {effort!r}",
        )

    workspace_kind = values.get("WORKSPACE_KIND", "worktree")
    if workspace_kind not in WORKSPACE_KINDS:
        _refuse(
            "unsupported_value",
            "WORKSPACE_KIND",
            f"unsupported WORKSPACE_KIND value {workspace_kind!r}",
        )

    return LegacyExecutionHeader(
        repo_path=repo_path,
        base_commit=base_commit,
        model=values.get("MODEL", "claude-opus-5"),
        effort=effort,
        max_turns=_max_turns(values.get("MAX_TURNS", "120")),
        workspace_kind=workspace_kind,
        no_reroute=_truthy(values.get("NO_REROUTE", "false")),
        source_digest=hashlib.sha256(source_goal.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "LegacyExecutionHeader",
    "LegacyMetadataError",
    "parse_legacy_execution_header",
]
