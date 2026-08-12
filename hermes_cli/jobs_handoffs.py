"""Pure canonical contract and conversational renderer for Job handoffs."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence


SCHEMA_VERSION = 1
MAX_HANDOFF_JSON_CHARS = 3000
MAX_SUMMARY_CHARS = 700
MAX_NEXT_ACTION_CHARS = 500
MAX_EVIDENCE_ITEMS = 8
MAX_ISSUES = 8
OUTCOMES = frozenset({
    "started",
    "handed_off",
    "passed",
    "rejected",
    "blocked",
    "activated",
    "completed",
})


class HandoffValidationError(ValueError):
    """Raised when a substantive handoff violates its canonical contract."""


_DIGEST = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")
_MAX_IDENTITY_CHARS = 80
_SECRET_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"authorization\s*:",
        r"bearer\s+",
        r"token\s*=\s*",
        r"api[\s_-]*key",
        r"private\s+key",
        r"(?<!\w)sk\s*-\s*",
    )
)
_RAW_FIELDS = frozenset({
    "summary",
    "evidence_summary",
    "next_action",
    "issues",
    "decision_request",
})
_EVIDENCE_FIELDS = frozenset({"label", "result", "digest"})
_ISSUE_FIELDS = frozenset({"requirement", "finding", "required_fix"})
_ARTIFACT_FIELDS = frozenset({"kind", "value"})
_DECISION_FIELDS = frozenset({
    "question",
    "options",
    "recommendation",
    "recommendation_reason",
    "blocked_scope",
    "safe_state",
    "next_owner_role",
})
_OPTION_FIELDS = frozenset({"id", "label", "consequence"})
_CANONICAL_FIELDS = (
    "schema_version",
    "job_id",
    "attempt_id",
    "speaker_id",
    "speaker_role",
    "speaker_executor",
    "from_phase",
    "to_phase",
    "next_owner_role",
    "summary",
    "evidence_summary",
    "outcome",
    "next_action",
    "issues",
    "artifact_identity",
    "decision_request",
    "created_at",
)
_CANONICAL_FIELD_SET = frozenset(_CANONICAL_FIELDS)
_ACTIVE_PHASES = frozenset({
    "QUEUED",
    "ASSIGNED",
    "BUILDING",
    "EVIDENCE_COLLECTING",
    "REVIEWING",
    "VERIFIED",
})
_ALLOWED_TRANSITIONS = frozenset({
    ("INTAKE", "QUEUED", "started"),
    ("ATTEMPT_CREATED", "QUEUED", "started"),
    ("QUEUED", "ASSIGNED", "handed_off"),
    ("ASSIGNED", "BUILDING", "started"),
    ("BUILDING", "EVIDENCE_COLLECTING", "handed_off"),
    ("EVIDENCE_COLLECTING", "REVIEWING", "handed_off"),
    ("REVIEWING", "VERIFIED", "passed"),
    ("VERIFIED", "COMPLETED", "completed"),
    ("VERIFIED", "COMPLETED", "activated"),
    *((phase, "FAILED", "rejected") for phase in _ACTIVE_PHASES),
    *((phase, "BLOCKED", "blocked") for phase in _ACTIVE_PHASES),
})


def _exact_mapping(
    value: object,
    expected: frozenset[str],
    *,
    field: str,
    required: frozenset[str] | None = None,
) -> Mapping[str, object]:
    if type(value) is not dict:
        raise HandoffValidationError(f"{field} must use exact JSON object types")
    actual = set(value)
    unknown = actual - expected
    if unknown:
        raise HandoffValidationError(f"{field} has unknown keys")
    missing = (expected if required is None else required) - actual
    if missing:
        raise HandoffValidationError(
            f"{field} is missing keys: {', '.join(sorted(missing))}"
        )
    return value


def _require_exact_json_tree(value: object, *, field: str) -> None:
    ancestors: set[int] = set()
    pending: list[tuple[object, bool]] = [(value, False)]
    while pending:
        item, leaving = pending.pop()
        if leaving:
            ancestors.remove(id(item))
            continue
        if item is None or type(item) in {str, int}:
            continue
        if type(item) not in {dict, list}:
            raise HandoffValidationError(f"{field} must use exact JSON built-in types")
        identity = id(item)
        if identity in ancestors:
            raise HandoffValidationError(f"{field} must be an acyclic JSON tree")
        ancestors.add(identity)
        pending.append((item, True))
        if type(item) is list:
            pending.extend((child, False) for child in reversed(item))
            continue
        children = []
        for key, child in item.items():
            if type(key) is not str:
                raise HandoffValidationError(f"{field} must use exact JSON string keys")
            children.append(child)
        pending.extend((child, False) for child in reversed(children))


def _screen_secret_text(value: str, *, field: str) -> None:
    screening_form = unicodedata.normalize("NFKC", value).casefold()
    if any(pattern.search(screening_form) for pattern in _SECRET_PATTERNS):
        raise HandoffValidationError(f"{field} contains a forbidden secret marker")


def _screen_json_strings(value: object, *, field: str) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is str:
            _screen_secret_text(item, field=field)
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is dict:
            pending.extend(item)
            pending.extend(item.values())


def _bounded_text(
    value: object,
    *,
    field: str,
    max_chars: int,
    narrative: bool = False,
) -> str:
    if type(value) is not str or not value.strip():
        raise HandoffValidationError(f"{field} must be a non-empty string")
    value = unicodedata.normalize("NFC", value)
    if len(value) > max_chars:
        raise HandoffValidationError(f"{field} exceeds its {max_chars}-character limit")
    _screen_secret_text(value, field=field)
    for character in value:
        category = unicodedata.category(character)
        if category in {"Cf", "Cs"}:
            raise HandoffValidationError(f"{field} contains unsafe Unicode")
        if category != "Cc":
            continue
        if narrative and character in {"\n", "\t"}:
            continue
        raise HandoffValidationError(f"{field} contains a forbidden control character")
    return value


def _require_persisted_nfc(value: object) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is str:
            if unicodedata.normalize("NFC", item) != item:
                raise HandoffValidationError(
                    "persisted handoff strings must be NFC-normalized"
                )
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is dict:
            pending.extend(item)
            pending.extend(item.values())


def _normalize_transition_evidence(value: object) -> dict[str, str]:
    if type(value) is not dict:
        raise HandoffValidationError(
            "transition_evidence must use an exact JSON object"
        )
    normalized: dict[str, str] = {}
    for key, digest in value.items():
        normalized_key = _bounded_text(
            key,
            field="transition_evidence key",
            max_chars=120,
        )
        if normalized_key in normalized:
            raise HandoffValidationError(
                "transition_evidence contains duplicate normalized keys"
            )
        if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
            raise HandoffValidationError(
                "transition_evidence values must be canonical sha256 digests"
            )
        normalized[normalized_key] = digest
    return normalized


def _normalize_evidence(
    value: object, *, transition_evidence: Mapping[str, str]
) -> list[dict[str, str]]:
    if type(value) is not list:
        raise HandoffValidationError("evidence_summary must use an exact JSON list")
    if len(value) > MAX_EVIDENCE_ITEMS:
        raise HandoffValidationError(
            f"evidence_summary may contain at most {MAX_EVIDENCE_ITEMS} items"
        )
    if transition_evidence and not value:
        raise HandoffValidationError(
            "evidence_summary requires at least one fact when transition evidence exists"
        )
    allowed_digests = set(transition_evidence.values())
    normalized = []
    for index, item in enumerate(value):
        fact = _exact_mapping(
            item, _EVIDENCE_FIELDS, field=f"evidence_summary[{index}]"
        )
        digest = fact["digest"]
        if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
            raise HandoffValidationError(
                f"evidence_summary[{index}].digest must be a canonical sha256 digest"
            )
        if digest not in allowed_digests:
            raise HandoffValidationError(
                f"evidence_summary[{index}].digest is not bound to transition_evidence"
            )
        normalized.append({
            "label": _bounded_text(
                fact["label"],
                field=f"evidence_summary[{index}].label",
                max_chars=120,
                narrative=True,
            ),
            "result": _bounded_text(
                fact["result"],
                field=f"evidence_summary[{index}].result",
                max_chars=300,
                narrative=True,
            ),
            "digest": digest,
        })
    return normalized


def _normalize_issues(value: object) -> list[dict[str, str]]:
    if type(value) is not list:
        raise HandoffValidationError("issues must use an exact JSON list")
    if len(value) > MAX_ISSUES:
        raise HandoffValidationError(f"issues may contain at most {MAX_ISSUES} items")
    normalized = []
    for index, item in enumerate(value):
        issue = _exact_mapping(item, _ISSUE_FIELDS, field=f"issues[{index}]")
        normalized.append({
            field: _bounded_text(
                issue[field],
                field=f"issues[{index}].{field}",
                max_chars=500,
                narrative=True,
            )
            for field in ("requirement", "finding", "required_fix")
        })
    return normalized


def _normalize_decision(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    decision = _exact_mapping(value, _DECISION_FIELDS, field="decision_request")
    options = decision["options"]
    if type(options) is not list:
        raise HandoffValidationError(
            "decision_request.options must use an exact JSON list"
        )
    if len(options) not in {2, 3}:
        raise HandoffValidationError(
            "decision_request.options must contain two or three options"
        )
    normalized_options = []
    for index, option in enumerate(options):
        exact = _exact_mapping(
            option,
            _OPTION_FIELDS,
            field=f"decision_request.options[{index}]",
        )
        normalized_options.append({
            "id": _bounded_text(
                exact["id"],
                field=f"decision_request.options[{index}].id",
                max_chars=80,
            ),
            "label": _bounded_text(
                exact["label"],
                field=f"decision_request.options[{index}].label",
                max_chars=700,
                narrative=True,
            ),
            "consequence": _bounded_text(
                exact["consequence"],
                field=f"decision_request.options[{index}].consequence",
                max_chars=700,
                narrative=True,
            ),
        })
    option_ids = [option["id"] for option in normalized_options]
    if len(set(option_ids)) != len(option_ids):
        raise HandoffValidationError("decision_request.options ids must be unique")
    recommendation = _bounded_text(
        decision["recommendation"],
        field="decision_request.recommendation",
        max_chars=80,
    )
    if recommendation not in option_ids:
        raise HandoffValidationError(
            "decision_request.recommendation must name a supplied option id"
        )
    return {
        "question": _bounded_text(
            decision["question"],
            field="decision_request.question",
            max_chars=700,
            narrative=True,
        ),
        "options": normalized_options,
        "recommendation": recommendation,
        "recommendation_reason": _bounded_text(
            decision["recommendation_reason"],
            field="decision_request.recommendation_reason",
            max_chars=700,
            narrative=True,
        ),
        "blocked_scope": _bounded_text(
            decision["blocked_scope"],
            field="decision_request.blocked_scope",
            max_chars=700,
            narrative=True,
        ),
        "safe_state": _bounded_text(
            decision["safe_state"],
            field="decision_request.safe_state",
            max_chars=700,
            narrative=True,
        ),
        "next_owner_role": _bounded_text(
            decision["next_owner_role"],
            field="decision_request.next_owner_role",
            max_chars=80,
        ),
    }


def _normalize_artifact(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    artifact = _exact_mapping(value, _ARTIFACT_FIELDS, field="artifact_identity")
    kind = _bounded_text(artifact["kind"], field="artifact_identity.kind", max_chars=80)
    if kind not in {"commit", "release", "deployment", "artifact"}:
        raise HandoffValidationError("artifact_identity.kind is unsupported")
    value_text = _bounded_text(
        artifact["value"], field="artifact_identity.value", max_chars=200
    )
    if kind == "commit" and _COMMIT.fullmatch(value_text) is None:
        raise HandoffValidationError(
            "artifact_identity commit value must be a full lowercase SHA"
        )
    return {"kind": kind, "value": value_text}


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise HandoffValidationError("handoff must be canonical JSON data") from exc


def _enforce_json_budget(receipt: dict) -> dict:
    if len(_canonical_json(receipt)) > MAX_HANDOFF_JSON_CHARS:
        raise HandoffValidationError(
            f"canonical handoff exceeds {MAX_HANDOFF_JSON_CHARS} characters"
        )
    return receipt


def _validate_transition_edge(
    *,
    attempt_id: str | None,
    from_phase: str,
    to_phase: str,
    outcome: str,
) -> None:
    edge = (from_phase, to_phase, outcome)
    if edge not in _ALLOWED_TRANSITIONS:
        raise HandoffValidationError("handoff phase and outcome are incompatible")
    if edge == ("INTAKE", "QUEUED", "started") and attempt_id is not None:
        raise HandoffValidationError("intake handoff requires attempt_id=None")


def normalize_handoff(
    raw: Mapping[str, object],
    *,
    job_id: str,
    attempt_id: str | None,
    speaker_id: str,
    speaker_role: str,
    speaker_executor: str,
    from_phase: str,
    to_phase: str,
    next_owner_role: str | None,
    outcome: str,
    artifact_identity: Mapping[str, object] | None,
    transition_evidence: Mapping[str, str],
    created_at: int,
) -> dict:
    """Bind caller-supplied handoff facts to one transition identity."""

    _require_exact_json_tree(raw, field="handoff")
    _screen_json_strings(raw, field="handoff")
    if artifact_identity is not None:
        _require_exact_json_tree(artifact_identity, field="artifact_identity")
        _screen_json_strings(artifact_identity, field="artifact_identity")
    _require_exact_json_tree(transition_evidence, field="transition_evidence")
    _screen_json_strings(transition_evidence, field="transition_evidence")
    transition_evidence = _normalize_transition_evidence(transition_evidence)
    raw = _exact_mapping(
        raw,
        _RAW_FIELDS,
        field="handoff",
        required=frozenset({"summary", "evidence_summary", "next_action"}),
    )
    if type(outcome) is not str:
        raise HandoffValidationError("unsupported handoff outcome")
    _screen_secret_text(outcome, field="outcome")
    if outcome not in OUTCOMES:
        raise HandoffValidationError("unsupported handoff outcome")

    job_id = _bounded_text(job_id, field="job_id", max_chars=_MAX_IDENTITY_CHARS)
    speaker_id = _bounded_text(
        speaker_id, field="speaker_id", max_chars=_MAX_IDENTITY_CHARS
    )
    speaker_role = _bounded_text(
        speaker_role, field="speaker_role", max_chars=_MAX_IDENTITY_CHARS
    )
    speaker_executor = _bounded_text(
        speaker_executor, field="speaker_executor", max_chars=_MAX_IDENTITY_CHARS
    )
    from_phase = _bounded_text(
        from_phase, field="from_phase", max_chars=_MAX_IDENTITY_CHARS
    )
    to_phase = _bounded_text(to_phase, field="to_phase", max_chars=_MAX_IDENTITY_CHARS)
    if attempt_id is None:
        if (from_phase, to_phase) != ("INTAKE", "QUEUED"):
            raise HandoffValidationError(
                "attempt_id may be None only for INTAKE -> QUEUED"
            )
    else:
        attempt_id = _bounded_text(
            attempt_id, field="attempt_id", max_chars=_MAX_IDENTITY_CHARS
        )
    _validate_transition_edge(
        attempt_id=attempt_id,
        from_phase=from_phase,
        to_phase=to_phase,
        outcome=outcome,
    )

    if next_owner_role is None:
        if outcome != "completed":
            raise HandoffValidationError(
                "next_owner_role is required for a continuing outcome"
            )
    else:
        next_owner_role = _bounded_text(
            next_owner_role,
            field="next_owner_role",
            max_chars=_MAX_IDENTITY_CHARS,
        )
    if type(created_at) is not int:
        raise HandoffValidationError("created_at must be an integer")

    issues = _normalize_issues(raw.get("issues", []))
    if outcome == "rejected" and not issues:
        raise HandoffValidationError(
            "rejected handoff requires at least one issues entry"
        )
    decision_request = _normalize_decision(raw.get("decision_request"))
    if outcome == "blocked":
        if decision_request is None:
            raise HandoffValidationError(
                "blocked handoff requires a complete decision_request"
            )
    elif decision_request is not None:
        raise HandoffValidationError(
            "decision_request is allowed only for a blocked handoff"
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "speaker_id": speaker_id,
        "speaker_role": speaker_role,
        "speaker_executor": speaker_executor,
        "from_phase": from_phase,
        "to_phase": to_phase,
        "next_owner_role": next_owner_role,
        "summary": _bounded_text(
            raw["summary"],
            field="summary",
            max_chars=MAX_SUMMARY_CHARS,
            narrative=True,
        ),
        "evidence_summary": _normalize_evidence(
            raw["evidence_summary"], transition_evidence=transition_evidence
        ),
        "outcome": outcome,
        "next_action": _bounded_text(
            raw["next_action"],
            field="next_action",
            max_chars=MAX_NEXT_ACTION_CHARS,
            narrative=True,
        ),
        "issues": issues,
        "artifact_identity": _normalize_artifact(artifact_identity),
        "decision_request": decision_request,
        "created_at": created_at,
    }
    return _enforce_json_budget(receipt)


def validate_persisted_handoff(
    value: object,
    *,
    target_state: str | None = None,
    transition_evidence: Mapping[str, str] | None = None,
) -> dict:
    """Strictly revalidate a canonical handoff already loaded from storage."""

    if transition_evidence is None:
        raise HandoffValidationError(
            "persisted handoff requires trusted transition_evidence"
        )
    _require_exact_json_tree(value, field="persisted handoff")
    _screen_json_strings(value, field="persisted handoff")
    persisted = _exact_mapping(value, _CANONICAL_FIELD_SET, field="persisted handoff")
    _require_persisted_nfc(value)
    schema_version = persisted["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise HandoffValidationError("unsupported schema_version")
    if target_state is not None:
        target_state = _bounded_text(
            target_state,
            field="target_state",
            max_chars=_MAX_IDENTITY_CHARS,
        )
        if persisted["to_phase"] != target_state:
            raise HandoffValidationError(
                "persisted handoff target_state does not match to_phase"
            )

    evidence_summary = persisted["evidence_summary"]

    raw = {
        "summary": persisted["summary"],
        "evidence_summary": evidence_summary,
        "next_action": persisted["next_action"],
        "issues": persisted["issues"],
        "decision_request": persisted["decision_request"],
    }
    normalized = normalize_handoff(
        raw,
        job_id=persisted["job_id"],
        attempt_id=persisted["attempt_id"],
        speaker_id=persisted["speaker_id"],
        speaker_role=persisted["speaker_role"],
        speaker_executor=persisted["speaker_executor"],
        from_phase=persisted["from_phase"],
        to_phase=persisted["to_phase"],
        next_owner_role=persisted["next_owner_role"],
        outcome=persisted["outcome"],
        artifact_identity=persisted["artifact_identity"],
        transition_evidence=transition_evidence,
        created_at=persisted["created_at"],
    )
    if normalized != dict(persisted):
        raise HandoffValidationError("persisted handoff is not canonical")
    return normalized


def intake_handoff(
    *,
    job_id: str,
    speaker_id: str,
    speaker_role: str,
    speaker_executor: str,
    next_owner_role: str,
    summary: str,
    evidence_summary: Sequence[Mapping[str, object]],
    next_action: str,
    transition_evidence: Mapping[str, str],
    created_at: int,
) -> dict:
    """Build the canonical handoff for a newly accepted Job."""

    return normalize_handoff(
        {
            "summary": summary,
            "evidence_summary": [dict(fact) for fact in evidence_summary],
            "next_action": next_action,
            "issues": [],
            "decision_request": None,
        },
        job_id=job_id,
        attempt_id=None,
        speaker_id=speaker_id,
        speaker_role=speaker_role,
        speaker_executor=speaker_executor,
        from_phase="INTAKE",
        to_phase="QUEUED",
        next_owner_role=next_owner_role,
        outcome="started",
        artifact_identity=None,
        transition_evidence=transition_evidence,
        created_at=created_at,
    )


def blocker_handoff(
    *,
    job_id: str,
    attempt_id: str,
    speaker_id: str,
    speaker_role: str,
    speaker_executor: str,
    from_phase: str,
    decision_owner_role: str,
    post_decision_owner_role: str,
    summary: str,
    evidence_summary: Sequence[Mapping[str, object]],
    next_action: str,
    question: str,
    options: Sequence[Mapping[str, object]],
    recommendation: str,
    recommendation_reason: str,
    blocked_scope: str,
    safe_state: str,
    transition_evidence: Mapping[str, str],
    created_at: int,
    issues: Sequence[Mapping[str, object]] = (),
    artifact_identity: Mapping[str, object] | None = None,
) -> dict:
    """Build a blocker for one decision owner and a distinct follow-up owner."""

    return normalize_handoff(
        {
            "summary": summary,
            "evidence_summary": [dict(fact) for fact in evidence_summary],
            "next_action": next_action,
            "issues": [dict(issue) for issue in issues],
            "decision_request": {
                "question": question,
                "options": [dict(option) for option in options],
                "recommendation": recommendation,
                "recommendation_reason": recommendation_reason,
                "blocked_scope": blocked_scope,
                "safe_state": safe_state,
                "next_owner_role": post_decision_owner_role,
            },
        },
        job_id=job_id,
        attempt_id=attempt_id,
        speaker_id=speaker_id,
        speaker_role=speaker_role,
        speaker_executor=speaker_executor,
        from_phase=from_phase,
        to_phase="BLOCKED",
        next_owner_role=decision_owner_role,
        outcome="blocked",
        artifact_identity=artifact_identity,
        transition_evidence=transition_evidence,
        created_at=created_at,
    )


def _render_text(value: str) -> str:
    """Keep user-controlled narrative on one structural renderer line."""

    return (
        value
        .replace("\n", " ⏎ ")
        .replace("\t", " ⇥ ")
        .replace("\u2028", " ⏎ ")
        .replace("\u2029", " ⏎ ")
    )


def render_handoff(
    receipt: object,
    *,
    transition_evidence: Mapping[str, str] | None = None,
) -> str:
    """Render one validated canonical handoff without consulting other state."""

    handoff = validate_persisted_handoff(
        receipt, transition_evidence=transition_evidence
    )
    outcome = handoff["outcome"]
    speaker_role = _render_text(handoff["speaker_role"])
    next_owner_role = (
        None
        if handoff["next_owner_role"] is None
        else _render_text(handoff["next_owner_role"])
    )
    if outcome == "started":
        heading = f"{speaker_role} started"
    elif outcome == "passed":
        heading = f"{speaker_role} approved"
    elif outcome == "rejected":
        heading = f"{speaker_role} sent it back → {next_owner_role}"
    elif outcome == "blocked":
        heading = "Hermes needs your decision"
    elif outcome == "completed":
        heading = "Job complete"
    else:
        heading = f"{speaker_role} → {next_owner_role}"

    lines = [heading, _render_text(handoff["summary"])]
    for fact in handoff["evidence_summary"]:
        lines.append(
            f"Evidence: {_render_text(fact['label'])}: {_render_text(fact['result'])}"
        )
    for issue in handoff["issues"]:
        lines.extend((
            f"Issue — {_render_text(issue['requirement'])}: "
            f"{_render_text(issue['finding'])}",
            f"Required fix: {_render_text(issue['required_fix'])}",
        ))
    decision = handoff["decision_request"]
    if decision is not None:
        lines.append(_render_text(decision["question"]))
        for index, option in enumerate(decision["options"], start=1):
            lines.append(
                f"{index}. {_render_text(option['id'])} — "
                f"{_render_text(option['label'])}: "
                f"{_render_text(option['consequence'])}"
            )
        lines.extend((
            f"Recommendation: {_render_text(decision['recommendation'])} — "
            f"{_render_text(decision['recommendation_reason'])}",
            f"Blocked: {_render_text(decision['blocked_scope'])}",
            f"Safe now: {_render_text(decision['safe_state'])}",
        ))
    lines.append(f"Next: {_render_text(handoff['next_action'])}")
    return "\n".join(lines)
