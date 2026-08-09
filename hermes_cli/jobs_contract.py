"""Dependency-free adapter envelope for the Jobs control plane.

:class:`JobEnvelope` is the minimum, JSON-safe, immutable value object every
future specialist adapter (Claude, Codex, Graph Engineer, Loop, Vault Steward,
scheduled-job runner, reviewer) accepts. Pinning it now gives adapters a stable
boundary to land against.

This module deliberately imports **nothing** from the Jobs store, Kanban, or any
engine — it is a pure contract with no I/O and no side effects. It carries the
custody ``claim_token`` because the envelope *is* the adapter's working
credential for the Job it was handed.

Two guarantees adapters rely on:

- **Immutable + defensively copied.** The dataclass is frozen, and ``metadata``
  is deep-copied into a read-only structure at construction, so a caller can
  neither rebind a field nor mutate a nested value it passed in (or later reads
  back). Lists become tuples; nested dicts become read-only mappings.
- **Secrets rejected by policy.** A metadata *key* containing ``password``,
  ``token``, ``secret``, ``api_key``, or ``authorization`` (case-insensitive,
  substring, at any nesting depth) is rejected — secrets do not belong in a
  serialized job envelope. The policy applies to metadata keys ONLY; the
  original goal is never scanned or rewritten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

# Case-insensitive substring markers. A key that *contains* any of these is a
# secret by policy — catches ``API_KEY``, ``db_secret``, ``auth_token``,
# ``MY_TOKEN``, ``Authorization``, etc., not just exact matches.
_SECRET_KEY_MARKERS = ("password", "token", "secret", "api_key", "authorization")

_REQUIRED_STRINGS = ("job_id", "name", "goal", "attempt_id", "claim_token")
_REQUIRED_INTS = ("number", "ordinal")


def _reject_secret_key(key: str) -> None:
    low = key.lower()
    for marker in _SECRET_KEY_MARKERS:
        if marker in low:
            raise ValueError(
                f"metadata key {key!r} matches secret-policy marker {marker!r}; "
                "secrets must not travel in a job envelope"
            )


def _freeze(value: Any) -> Any:
    """Recursively deep-copy ``value`` into a read-only, JSON-safe structure.

    Dicts become :class:`MappingProxyType` (validating every key against the
    secret policy on the way down); lists/tuples become tuples; scalars pass
    through. Anything else (a set, an object, bytes) is not JSON-safe and is
    rejected — the envelope must always serialize.
    """
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"metadata keys must be strings, got {type(k).__name__}"
                )
            _reject_secret_key(k)
            out[k] = _freeze(v)
        return MappingProxyType(out)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    # bool is a subclass of int, so it rides through the scalar branch.
    if value is None or isinstance(value, (str, int, float)):
        return value
    raise ValueError(
        f"metadata value is not JSON-safe: {type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze`: plain dicts/lists for JSON serialization."""
    if isinstance(value, MappingProxyType):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class JobEnvelope:
    """Immutable, JSON-safe job hand-off passed to every specialist adapter.

    Required, non-empty: ``job_id``, ``name``, ``goal``, ``attempt_id``,
    ``claim_token``. Required integers: ``number`` (permanent Job number),
    ``ordinal`` (per-Job attempt ordinal). Everything else is optional and
    present only when the Job has it. ``metadata`` is frozen deeply.
    """

    job_id: str
    number: int
    name: str
    goal: str
    attempt_id: str
    ordinal: int
    # repr=False: an adapter debug log, a traceback, or a plain ``print(env)``
    # must never carry the capability. Read the field deliberately instead.
    claim_token: str = field(repr=False)
    specialist: Optional[str] = None
    routing_reason: Optional[str] = None
    repository: Optional[str] = None
    branch: Optional[str] = None
    worktree: Optional[str] = None
    commit: Optional[str] = None
    parent_attempt_id: Optional[str] = None
    # The skills this Job declared, frozen to a tuple. ``None`` is *unset* and is
    # not the same as an empty tuple: unset lets the adapter's default rules
    # choose, empty is an explicit "attach nothing". Names only — resolving one
    # to a file is the adapter's job, not this contract's.
    skills: Optional[Sequence[str]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _REQUIRED_STRINGS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in _REQUIRED_INTS:
            value = getattr(self, name)
            # Reject bool explicitly (it is an int subclass) and non-ints.
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.skills is not None:
            if isinstance(self.skills, str) or not isinstance(self.skills, (list, tuple)):
                raise ValueError("skills must be a list of names, or None")
            for skill in self.skills:
                if not isinstance(skill, str) or not skill.strip():
                    raise ValueError("every skill must be a non-empty string")
            object.__setattr__(self, "skills", tuple(self.skills))
        meta = self.metadata if self.metadata is not None else {}
        if not isinstance(meta, Mapping):
            raise ValueError("metadata must be a JSON object (dict)")
        # Frozen dataclass: bypass the field lock to store the frozen copy.
        object.__setattr__(self, "metadata", _freeze(meta))

    def to_dict(self, *, include_claim_token: bool = False) -> dict:
        """A plain, JSON-safe dict — metadata thawed to plain dicts/lists.

        The claim token is **omitted** unless explicitly requested, so ordinary
        debug or diagnostic serialization cannot leak the capability. The key is
        left out rather than stubbed: a ``KeyError`` at the one call site that
        forgot to ask is a far better failure than a token in a log line.
        ``include_claim_token=True`` is the deliberate adapter serialization
        that actually transports the credential to the engine.
        """
        out = {
            "job_id": self.job_id,
            "number": self.number,
            "name": self.name,
            "goal": self.goal,
            "specialist": self.specialist,
            "routing_reason": self.routing_reason,
            "attempt_id": self.attempt_id,
            "ordinal": self.ordinal,
            "repository": self.repository,
            "branch": self.branch,
            "worktree": self.worktree,
            "commit": self.commit,
            "parent_attempt_id": self.parent_attempt_id,
            "skills": None if self.skills is None else list(self.skills),
            "metadata": _thaw(self.metadata),
        }
        if include_claim_token:
            out["claim_token"] = self.claim_token
        return out

    def to_json(self, *, include_claim_token: bool = False) -> str:
        """Deterministic JSON (sorted keys, Unicode preserved), token redacted."""
        return json.dumps(
            self.to_dict(include_claim_token=include_claim_token),
            ensure_ascii=False,
            sort_keys=True,
        )
