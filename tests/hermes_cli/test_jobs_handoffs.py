"""Behavior tests for canonical substantive Job handoffs."""

from __future__ import annotations

from copy import deepcopy

import pytest

from hermes_cli.jobs_handoffs import (
    HandoffValidationError,
    MAX_EVIDENCE_ITEMS,
    MAX_HANDOFF_JSON_CHARS,
    MAX_ISSUES,
    MAX_NEXT_ACTION_CHARS,
    MAX_SUMMARY_CHARS,
    OUTCOMES,
    blocker_handoff,
    intake_handoff,
    normalize_handoff,
    render_handoff,
    validate_persisted_handoff,
)


TEST_DIGEST = "sha256:" + "a" * 64
COMMIT = "b" * 40
ACTIVE_PHASES = (
    "QUEUED",
    "ASSIGNED",
    "BUILDING",
    "EVIDENCE_COLLECTING",
    "REVIEWING",
    "VERIFIED",
)
ALLOWED_TRANSITIONS = (
    ("INTAKE", "QUEUED", "started"),
    ("ATTEMPT_CREATED", "QUEUED", "started"),
    ("QUEUED", "ASSIGNED", "handed_off"),
    ("ASSIGNED", "BUILDING", "started"),
    ("BUILDING", "EVIDENCE_COLLECTING", "handed_off"),
    ("EVIDENCE_COLLECTING", "REVIEWING", "handed_off"),
    ("REVIEWING", "VERIFIED", "passed"),
    ("OUTCOME_VERIFIED", "COMPLETED", "completed"),
    ("OUTCOME_VERIFIED", "COMPLETED", "activated"),
    *((phase, "FAILED", "rejected") for phase in ACTIVE_PHASES),
    *((phase, "BLOCKED", "blocked") for phase in ACTIVE_PHASES),
)


def _raw(**overrides):
    raw = {
        "summary": "Builder completed the scoped change.",
        "evidence_summary": [
            {
                "label": "Focused tests",
                "result": "42 tests passed",
                "digest": TEST_DIGEST,
            }
        ],
        "next_action": "Review the committed change.",
        "issues": [],
        "decision_request": None,
    }
    raw.update(overrides)
    return raw


def _normalize(raw=None, **overrides):
    values = {
        "job_id": "job-7",
        "attempt_id": "attempt-2",
        "speaker_id": "worker-3",
        "speaker_role": "builder",
        "speaker_executor": "claude",
        "from_phase": "BUILDING",
        "to_phase": "EVIDENCE_COLLECTING",
        "next_owner_role": "reviewer",
        "outcome": "handed_off",
        "artifact_identity": {"kind": "commit", "value": COMMIT},
        "transition_evidence": {"tests": TEST_DIGEST},
        "created_at": 1_786_500_000,
    }
    values.update(overrides)
    compatible_edges = {
        "started": ("ASSIGNED", "BUILDING"),
        "handed_off": ("BUILDING", "EVIDENCE_COLLECTING"),
        "passed": ("REVIEWING", "VERIFIED"),
        "rejected": ("REVIEWING", "FAILED"),
        "blocked": ("BUILDING", "BLOCKED"),
        "activated": ("OUTCOME_VERIFIED", "COMPLETED"),
        "completed": ("OUTCOME_VERIFIED", "COMPLETED"),
    }
    compatible_edge = (
        compatible_edges.get(values["outcome"])
        if type(values["outcome"]) is str
        else None
    )
    if compatible_edge is not None:
        if "from_phase" not in overrides:
            values["from_phase"] = compatible_edge[0]
        if "to_phase" not in overrides:
            values["to_phase"] = compatible_edge[1]
    return normalize_handoff(_raw() if raw is None else raw, **values)


def _decision(**overrides):
    decision = {
        "question": "Which recovery path should Hermes take?",
        "options": [
            {
                "id": "retry",
                "label": "Retry safely",
                "consequence": "Re-run the bounded preflight.",
            },
            {
                "id": "stop",
                "label": "Stop here",
                "consequence": "Leave the Job blocked without changing state.",
            },
        ],
        "recommendation": "retry",
        "recommendation_reason": "The prior attempt made no durable change.",
        "blocked_scope": "Only this Job is blocked.",
        "safe_state": "The repository remains at the verified commit.",
        "next_owner_role": "operator",
    }
    decision.update(overrides)
    return decision


def _raw_for_outcome(outcome):
    issue = {
        "requirement": "The transition must be safe.",
        "finding": "Review found a blocking defect.",
        "required_fix": "Correct the defect before retrying.",
    }
    return _raw(
        issues=[issue] if outcome == "rejected" else [],
        decision_request=_decision() if outcome == "blocked" else None,
    )


def test_valid_builder_handoff_binds_transition_and_artifact_identity():
    receipt = _normalize()

    assert list(receipt) == [
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
    ]
    assert receipt["schema_version"] == 1
    assert receipt["job_id"] == "job-7"
    assert receipt["attempt_id"] == "attempt-2"
    assert receipt["speaker_id"] == "worker-3"
    assert receipt["speaker_role"] == "builder"
    assert receipt["speaker_executor"] == "claude"
    assert receipt["from_phase"] == "BUILDING"
    assert receipt["to_phase"] == "EVIDENCE_COLLECTING"
    assert receipt["next_owner_role"] == "reviewer"
    assert receipt["evidence_summary"][0]["digest"] == TEST_DIGEST
    assert receipt["artifact_identity"] == {
        "kind": "commit",
        "value": COMMIT,
    }


def test_outcomes_are_the_closed_contract_set():
    assert OUTCOMES == frozenset({
        "started",
        "handed_off",
        "passed",
        "rejected",
        "blocked",
        "activated",
        "completed",
    })
    with pytest.raises(HandoffValidationError, match="outcome"):
        _normalize(outcome="failed")


@pytest.mark.parametrize(("from_phase", "to_phase", "outcome"), ALLOWED_TRANSITIONS)
def test_every_allowed_transition_edge_normalizes(from_phase, to_phase, outcome):
    receipt = _normalize(
        _raw_for_outcome(outcome),
        attempt_id=None if from_phase == "INTAKE" else "attempt-2",
        from_phase=from_phase,
        to_phase=to_phase,
        outcome=outcome,
        next_owner_role=None if outcome == "completed" else "next-owner",
    )

    assert (receipt["from_phase"], receipt["to_phase"], receipt["outcome"]) == (
        from_phase,
        to_phase,
        outcome,
    )


@pytest.mark.parametrize(
    ("from_phase", "to_phase", "outcome"),
    [
        ("NOT_A_PHASE", "VERIFIED", "passed"),
        ("COMPLETED", "VERIFIED", "passed"),
        ("QUEUED", "VERIFIED", "passed"),
        ("ASSIGNED", "EVIDENCE_COLLECTING", "handed_off"),
        ("BUILDING", "ASSIGNED", "handed_off"),
        ("COMPLETED", "BLOCKED", "blocked"),
        ("FAILED", "BLOCKED", "blocked"),
        ("BLOCKED", "FAILED", "rejected"),
        ("REVIEWING", "COMPLETED", "completed"),
        ("QUEUED", "CANCELLED", "handed_off"),
    ],
)
def test_unknown_skipped_reversed_and_terminal_source_edges_are_rejected(
    from_phase, to_phase, outcome
):
    with pytest.raises(HandoffValidationError, match="phase.*outcome"):
        _normalize(
            _raw_for_outcome(outcome),
            from_phase=from_phase,
            to_phase=to_phase,
            outcome=outcome,
            next_owner_role=None if outcome == "completed" else "next-owner",
        )


def test_persisted_validation_and_renderer_reject_an_impossible_source_edge():
    receipt = _normalize(
        outcome="passed",
        from_phase="REVIEWING",
        to_phase="VERIFIED",
    )
    receipt["from_phase"] = "COMPLETED"

    with pytest.raises(HandoffValidationError, match="phase.*outcome"):
        validate_persisted_handoff(
            receipt,
            target_state="VERIFIED",
            transition_evidence={"tests": TEST_DIGEST},
        )
    with pytest.raises(HandoffValidationError, match="phase.*outcome"):
        render_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


@pytest.mark.parametrize(
    ("to_phase", "outcome"),
    [
        ("BLOCKED", "completed"),
        ("COMPLETED", "blocked"),
        ("VERIFIED", "handed_off"),
        ("UNKNOWN", "started"),
    ],
)
def test_phase_rejects_an_incompatible_outcome(to_phase, outcome):
    next_owner_role = None if outcome == "completed" else "reviewer"
    raw = _raw(decision_request=_decision() if outcome == "blocked" else None)
    with pytest.raises(HandoffValidationError, match="phase.*outcome"):
        _normalize(
            raw,
            to_phase=to_phase,
            outcome=outcome,
            next_owner_role=next_owner_role,
        )


def test_persisted_target_state_cannot_authorize_a_contradictory_outcome():
    receipt = _normalize(
        outcome="completed",
        from_phase="OUTCOME_VERIFIED",
        to_phase="COMPLETED",
        next_owner_role=None,
    )
    receipt["to_phase"] = "BLOCKED"

    with pytest.raises(HandoffValidationError, match="phase.*outcome"):
        validate_persisted_handoff(
            receipt,
            target_state="BLOCKED",
            transition_evidence={"tests": TEST_DIGEST},
        )


def test_renderer_refuses_a_contradictory_completed_receipt():
    receipt = _normalize(
        outcome="completed",
        from_phase="OUTCOME_VERIFIED",
        to_phase="COMPLETED",
        next_owner_role=None,
    )
    receipt["to_phase"] = "BLOCKED"

    with pytest.raises(HandoffValidationError, match="phase.*outcome"):
        render_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


def test_attempt_id_none_is_legal_for_intake_to_queued():
    receipt = _normalize(
        attempt_id=None,
        from_phase="INTAKE",
        to_phase="QUEUED",
        outcome="started",
    )
    assert receipt["attempt_id"] is None


def test_intake_to_queued_rejects_a_nonnull_attempt_id():
    with pytest.raises(HandoffValidationError, match="attempt_id"):
        _normalize(
            attempt_id="attempt-too-early",
            from_phase="INTAKE",
            to_phase="QUEUED",
            outcome="started",
        )


@pytest.mark.parametrize(
    ("from_phase", "to_phase"),
    [("QUEUED", "ASSIGNED"), ("INTAKE", "ASSIGNED"), ("BUILDING", "QUEUED")],
)
def test_attempt_id_none_is_rejected_for_attempt_transitions(from_phase, to_phase):
    with pytest.raises(HandoffValidationError, match="attempt_id"):
        _normalize(
            attempt_id=None,
            from_phase=from_phase,
            to_phase=to_phase,
        )


def test_attempt_transition_rejects_an_empty_attempt_id():
    with pytest.raises(HandoffValidationError, match="attempt_id"):
        _normalize(attempt_id="")


def test_unknown_top_level_raw_key_is_rejected():
    with pytest.raises(HandoffValidationError, match="unknown"):
        _normalize(_raw(hostile="must not survive"))


@pytest.mark.parametrize(
    ("raw", "overrides"),
    [
        (
            _raw(
                evidence_summary=[
                    {
                        "label": "Focused tests",
                        "result": "42 tests passed",
                        "digest": TEST_DIGEST,
                        "log": "raw output",
                    }
                ]
            ),
            {},
        ),
        (
            _raw(
                issues=[
                    {
                        "requirement": "Tests pass",
                        "finding": "One failed",
                        "required_fix": "Fix the failing assertion.",
                        "reviewer_instruction": "hidden",
                    }
                ]
            ),
            {},
        ),
        (
            _raw(decision_request=_decision(internal_notes="hidden")),
            {"outcome": "blocked", "next_owner_role": "operator"},
        ),
        (
            _raw(
                decision_request=_decision(
                    options=[
                        {
                            "id": "retry",
                            "label": "Retry safely",
                            "consequence": "Re-run the bounded preflight.",
                            "prompt": "hidden",
                        },
                        {
                            "id": "stop",
                            "label": "Stop here",
                            "consequence": "Leave the Job blocked.",
                        },
                    ]
                )
            ),
            {"outcome": "blocked", "next_owner_role": "operator"},
        ),
    ],
)
def test_unknown_nested_key_is_rejected(raw, overrides):
    with pytest.raises(HandoffValidationError, match="unknown"):
        _normalize(raw, **overrides)


def test_unknown_artifact_key_is_rejected():
    with pytest.raises(HandoffValidationError, match="unknown"):
        _normalize(
            artifact_identity={
                "kind": "commit",
                "value": COMMIT,
                "branch": "hidden",
            }
        )


@pytest.mark.parametrize(
    "digest",
    ["a" * 64, "sha256:" + "A" * 64, "sha256:" + "a" * 63, "sha256:xyz"],
)
def test_evidence_digest_must_be_canonical_sha256(digest):
    evidence = [{"label": "Focused tests", "result": "passed", "digest": digest}]
    with pytest.raises(HandoffValidationError, match="digest"):
        _normalize(_raw(evidence_summary=evidence))


def test_evidence_digest_must_be_bound_to_transition_evidence():
    evidence = [
        {
            "label": "Focused tests",
            "result": "passed",
            "digest": "sha256:" + "c" * 64,
        }
    ]
    with pytest.raises(HandoffValidationError, match="transition_evidence"):
        _normalize(_raw(evidence_summary=evidence))


@pytest.mark.parametrize(
    "transition_evidence",
    [
        {"tests": TEST_DIGEST, "unused": "not-a-digest"},
        {"": TEST_DIGEST, "tests": TEST_DIGEST},
        {"x" * 121: TEST_DIGEST, "tests": TEST_DIGEST},
    ],
)
def test_all_transition_evidence_entries_are_validated(transition_evidence):
    with pytest.raises(HandoffValidationError, match="transition_evidence"):
        _normalize(transition_evidence=transition_evidence)


def test_optional_raw_fields_are_filled_with_canonical_defaults():
    receipt = _normalize({
        "summary": "Builder completed the scoped change.",
        "evidence_summary": [
            {
                "label": "Focused tests",
                "result": "42 tests passed",
                "digest": TEST_DIGEST,
            }
        ],
        "next_action": "Review the committed change.",
    })
    assert receipt["issues"] == []
    assert receipt["decision_request"] is None


@pytest.mark.parametrize("missing", ["summary", "evidence_summary", "next_action"])
def test_required_raw_field_cannot_be_omitted(missing):
    raw = _raw()
    del raw[missing]
    with pytest.raises(HandoffValidationError, match=missing):
        _normalize(raw)


@pytest.mark.parametrize(
    "overrides",
    [
        {"job_id": ""},
        {"speaker_id": " "},
        {"speaker_role": ""},
        {"speaker_executor": ""},
        {"from_phase": ""},
        {"to_phase": ""},
        {"next_owner_role": ""},
    ],
)
def test_identity_phase_and_owner_fields_must_be_nonempty(overrides):
    field = next(iter(overrides))
    with pytest.raises(HandoffValidationError, match=field):
        _normalize(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"job_id": "j" * 81},
        {"attempt_id": "a" * 81},
        {"speaker_id": "s" * 81},
        {"speaker_role": "r" * 81},
        {"speaker_executor": "e" * 81},
        {"from_phase": "F" * 81},
        {"to_phase": "T" * 81},
        {"next_owner_role": "o" * 81},
    ],
)
def test_identity_phase_and_owner_fields_are_bounded(overrides):
    field = next(iter(overrides))
    with pytest.raises(HandoffValidationError, match=field):
        _normalize(**overrides)


@pytest.mark.parametrize("created_at", [True, False, 1.5, "1786500000", None])
def test_created_at_must_be_an_integer_not_bool(created_at):
    with pytest.raises(HandoffValidationError, match="created_at"):
        _normalize(created_at=created_at)


class _LyingList(list):
    def __len__(self):
        return 0


class _LyingStr(str):
    def strip(self):
        return "safe"

    def casefold(self):
        return "safe"

    def __len__(self):
        return 1

    def __iter__(self):
        return iter(())


class _IntSubclass(int):
    pass


def test_list_subclass_cannot_bypass_the_evidence_count_bound():
    fact = {
        "label": "Focused tests",
        "result": "passed",
        "digest": TEST_DIGEST,
    }
    facts = _LyingList([dict(fact) for _ in range(MAX_EVIDENCE_ITEMS + 1)])

    with pytest.raises(HandoffValidationError, match="exact JSON"):
        _normalize(_raw(evidence_summary=facts))


@pytest.mark.parametrize(
    "hostile",
    [
        _LyingStr("Authorization: hidden"),
        _LyingStr("hidden\x00control"),
        _LyingStr("x" * (MAX_SUMMARY_CHARS + 1)),
    ],
)
def test_string_subclass_cannot_bypass_text_safety(hostile):
    with pytest.raises(HandoffValidationError, match="exact JSON"):
        _normalize(_raw(summary=hostile))


@pytest.mark.parametrize(
    "overrides",
    [{"outcome": []}, {"artifact_identity": {"kind": [], "value": "x"}}],
)
def test_unhashable_enum_values_raise_handoff_validation_error(overrides):
    with pytest.raises(HandoffValidationError):
        _normalize(**overrides)


@pytest.mark.parametrize(
    ("field", "factory"),
    [
        (
            "evidence_summary",
            lambda receipt: _LyingList(receipt["evidence_summary"]),
        ),
        ("summary", lambda receipt: _LyingStr(receipt["summary"])),
        ("created_at", lambda receipt: _IntSubclass(receipt["created_at"])),
    ],
)
def test_persisted_handoff_rejects_json_scalar_and_container_subclasses(field, factory):
    receipt = _normalize()
    receipt[field] = factory(receipt)

    with pytest.raises(HandoffValidationError, match="exact JSON"):
        validate_persisted_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


def test_cyclic_values_raise_handoff_validation_error_at_both_boundaries():
    raw = _raw()
    raw["issues"] = [raw]
    with pytest.raises(HandoffValidationError, match="JSON"):
        _normalize(raw)

    receipt = _normalize()
    receipt["issues"] = [receipt]
    with pytest.raises(HandoffValidationError, match="JSON"):
        validate_persisted_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


def test_completed_handoff_may_have_no_next_owner():
    receipt = _normalize(
        outcome="completed",
        from_phase="OUTCOME_VERIFIED",
        to_phase="COMPLETED",
        next_owner_role=None,
    )
    assert receipt["next_owner_role"] is None


def test_continuing_handoff_requires_a_next_owner():
    with pytest.raises(HandoffValidationError, match="next_owner_role"):
        _normalize(next_owner_role=None)


@pytest.mark.parametrize(
    ("field", "limit"),
    [("summary", MAX_SUMMARY_CHARS), ("next_action", MAX_NEXT_ACTION_CHARS)],
)
def test_summary_and_next_action_are_nonempty_and_bounded(field, limit):
    assert _normalize(_raw(**{field: "x" * limit}))[field] == "x" * limit
    with pytest.raises(HandoffValidationError, match=field):
        _normalize(_raw(**{field: "x" * (limit + 1)}))
    with pytest.raises(HandoffValidationError, match=field):
        _normalize(_raw(**{field: " "}))


def test_evidence_count_is_bounded_and_required_when_transition_has_evidence():
    fact = {
        "label": "Focused tests",
        "result": "passed",
        "digest": TEST_DIGEST,
    }
    assert (
        len(
            _normalize(
                _raw(evidence_summary=[dict(fact) for _ in range(MAX_EVIDENCE_ITEMS)])
            )["evidence_summary"]
        )
        == MAX_EVIDENCE_ITEMS
    )
    with pytest.raises(HandoffValidationError, match="evidence_summary"):
        _normalize(
            _raw(evidence_summary=[dict(fact) for _ in range(MAX_EVIDENCE_ITEMS + 1)])
        )
    with pytest.raises(HandoffValidationError, match="evidence_summary"):
        _normalize(_raw(evidence_summary=[]))


@pytest.mark.parametrize(
    ("field", "value"),
    [("label", "x" * 121), ("result", "x" * 301), ("label", " ")],
)
def test_evidence_fact_text_is_nonempty_and_bounded(field, value):
    fact = {"label": "Focused tests", "result": "passed", "digest": TEST_DIGEST}
    fact[field] = value
    with pytest.raises(HandoffValidationError, match=field):
        _normalize(_raw(evidence_summary=[fact]))


def test_issue_count_and_text_are_bounded():
    issue = {
        "requirement": "Tests pass",
        "finding": "One test failed.",
        "required_fix": "Correct the failing behavior.",
    }
    assert (
        len(_normalize(_raw(issues=[dict(issue) for _ in range(MAX_ISSUES)]))["issues"])
        == MAX_ISSUES
    )
    with pytest.raises(HandoffValidationError, match="issues"):
        _normalize(_raw(issues=[dict(issue) for _ in range(MAX_ISSUES + 1)]))
    for field in issue:
        oversized = dict(issue)
        oversized[field] = "x" * 501
        with pytest.raises(HandoffValidationError, match=field):
            _normalize(_raw(issues=[oversized]))


def test_rejected_outcome_requires_a_complete_issue_with_required_fix():
    with pytest.raises(HandoffValidationError, match="issues"):
        _normalize(_raw(issues=[]), outcome="rejected")

    missing_fix = {"requirement": "Tests pass", "finding": "One failed."}
    with pytest.raises(HandoffValidationError, match="required_fix"):
        _normalize(_raw(issues=[missing_fix]), outcome="rejected")

    incomplete_fix = {
        "requirement": "Tests pass",
        "finding": "One failed.",
        "required_fix": " ",
    }
    with pytest.raises(HandoffValidationError, match="required_fix"):
        _normalize(_raw(issues=[incomplete_fix]), outcome="rejected")


@pytest.mark.parametrize("kind", ["commit", "release", "deployment", "artifact"])
def test_accepted_artifact_kinds_roundtrip_exactly(kind):
    value = COMMIT if kind == "commit" else "release-2026-08-12"
    receipt = _normalize(artifact_identity={"kind": kind, "value": value})
    assert receipt["artifact_identity"] == {"kind": kind, "value": value}


@pytest.mark.parametrize(
    "artifact",
    [
        {"kind": "branch", "value": "main"},
        {"kind": "commit", "value": "A" * 40},
        {"kind": "commit", "value": "a" * 39},
        {"kind": "artifact", "value": "x" * 201},
        {"kind": "artifact", "value": " "},
    ],
)
def test_invalid_artifact_identity_is_rejected(artifact):
    with pytest.raises(HandoffValidationError, match="artifact_identity"):
        _normalize(artifact_identity=artifact)


def test_canonical_json_character_budget_is_enforced():
    assert MAX_HANDOFF_JSON_CHARS == 3000
    facts = [
        {
            "label": "l" * 120,
            "result": "r" * 300,
            "digest": TEST_DIGEST,
        }
        for _ in range(MAX_EVIDENCE_ITEMS)
    ]
    with pytest.raises(HandoffValidationError, match="3000"):
        _normalize(
            _raw(
                summary="s" * MAX_SUMMARY_CHARS,
                evidence_summary=facts,
                next_action="n" * MAX_NEXT_ACTION_CHARS,
            )
        )


def test_blocked_outcome_accepts_a_complete_bounded_decision():
    decision = _decision()
    receipt = _normalize(
        _raw(decision_request=decision),
        outcome="blocked",
        next_owner_role="operator",
        from_phase="BUILDING",
        to_phase="BLOCKED",
    )
    assert receipt["decision_request"] == decision


@pytest.mark.parametrize(
    "missing",
    [
        "question",
        "options",
        "recommendation",
        "recommendation_reason",
        "blocked_scope",
        "safe_state",
        "next_owner_role",
    ],
)
def test_blocked_outcome_rejects_each_missing_decision_field(missing):
    decision = _decision()
    del decision[missing]
    with pytest.raises(HandoffValidationError, match=missing):
        _normalize(
            _raw(decision_request=decision),
            outcome="blocked",
            next_owner_role="operator",
        )


def test_blocked_outcome_requires_decision_data():
    with pytest.raises(HandoffValidationError, match="decision_request"):
        _normalize(
            _raw(decision_request=None),
            outcome="blocked",
            next_owner_role="operator",
        )


def test_decision_recommendation_must_name_a_supplied_option():
    with pytest.raises(HandoffValidationError, match="recommendation"):
        _normalize(
            _raw(decision_request=_decision(recommendation="not-an-option")),
            outcome="blocked",
            next_owner_role="operator",
        )


@pytest.mark.parametrize("option_count", [0, 1, 4])
def test_decision_requires_two_or_three_options(option_count):
    option = {
        "id": "retry",
        "label": "Retry safely",
        "consequence": "Re-run the bounded preflight.",
    }
    with pytest.raises(HandoffValidationError, match="options"):
        _normalize(
            _raw(
                decision_request=_decision(
                    options=[dict(option) for _ in range(option_count)]
                )
            ),
            outcome="blocked",
            next_owner_role="operator",
        )


def test_decision_option_ids_must_be_unique():
    duplicate = {
        "id": "retry",
        "label": "Try another way",
        "consequence": "Use the same identifier ambiguously.",
    }
    with pytest.raises(HandoffValidationError, match="unique"):
        _normalize(
            _raw(
                decision_request=_decision(
                    options=[_decision()["options"][0], duplicate]
                )
            ),
            outcome="blocked",
            next_owner_role="operator",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", " "),
        ("question", "x" * 701),
        ("recommendation", "x" * 81),
        ("recommendation_reason", "x" * 701),
        ("blocked_scope", "x" * 701),
        ("safe_state", "x" * 701),
        ("next_owner_role", "x" * 81),
    ],
)
def test_decision_fields_are_nonempty_and_bounded(field, value):
    with pytest.raises(HandoffValidationError, match=field):
        _normalize(
            _raw(decision_request=_decision(**{field: value})),
            outcome="blocked",
            next_owner_role="operator",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "x" * 81),
        ("label", "x" * 701),
        ("consequence", "x" * 701),
        ("consequence", " "),
    ],
)
def test_decision_option_fields_are_nonempty_and_bounded(field, value):
    option = dict(_decision()["options"][0])
    option[field] = value
    options = [option, _decision()["options"][1]]
    with pytest.raises(HandoffValidationError, match=field):
        _normalize(
            _raw(decision_request=_decision(options=options)),
            outcome="blocked",
            next_owner_role="operator",
        )


def test_blocked_handoff_distinguishes_decision_and_post_decision_owners():
    receipt = _normalize(
        _raw(decision_request=_decision(next_owner_role="Hermes Dispatcher")),
        outcome="blocked",
        next_owner_role="Brandon",
    )

    assert receipt["next_owner_role"] == "Brandon"
    assert receipt["decision_request"]["next_owner_role"] == "Hermes Dispatcher"


@pytest.mark.parametrize("decision_owner_role", [None, ""])
def test_blocked_handoff_rejects_missing_or_invalid_decision_owner(
    decision_owner_role,
):
    with pytest.raises(HandoffValidationError, match="next_owner_role"):
        _normalize(
            _raw(decision_request=_decision(next_owner_role="Hermes Dispatcher")),
            outcome="blocked",
            next_owner_role=decision_owner_role,
        )


@pytest.mark.parametrize("post_decision_owner_role", [None, ""])
def test_blocked_handoff_rejects_missing_or_invalid_post_decision_owner(
    post_decision_owner_role,
):
    decision = _decision(next_owner_role=post_decision_owner_role)
    if post_decision_owner_role is None:
        del decision["next_owner_role"]
    with pytest.raises(HandoffValidationError, match="next_owner_role"):
        _normalize(
            _raw(decision_request=decision),
            outcome="blocked",
            next_owner_role="Brandon",
        )


@pytest.mark.parametrize(
    "outcome",
    ["started", "handed_off", "passed", "rejected", "activated", "completed"],
)
def test_nonblocked_outcome_rejects_decision_data(outcome):
    issue = {
        "requirement": "Tests pass",
        "finding": "One failed.",
        "required_fix": "Fix it.",
    }
    next_owner = None if outcome == "completed" else "operator"
    with pytest.raises(HandoffValidationError, match="decision_request"):
        _normalize(
            _raw(
                issues=[issue] if outcome == "rejected" else [],
                decision_request=_decision(),
            ),
            outcome=outcome,
            next_owner_role=next_owner,
        )


@pytest.mark.parametrize(
    "marker",
    [
        "Authorization: hidden",
        "BEARER hidden",
        "prefix Token=hidden",
        "OPENAI_API_KEY",
        "private KEY material",
        "sk-project-secret",
    ],
)
def test_secret_markers_are_rejected_case_insensitively(marker):
    with pytest.raises(HandoffValidationError, match="secret"):
        _normalize(_raw(summary=f"Build result: {marker}"))


@pytest.mark.parametrize(
    "hostile",
    [
        "Bearer\tsecret-value",
        "api_\nkey=secret-value",
        "ＡＰＩ＿ＫＥＹ＝secret-value",
        "authorization \n : secret-value",
        "token \t = secret-value",
        "private\nkey material",
        "ｓｋ－secret-value",
    ],
)
def test_nested_secret_markers_reject_whitespace_and_nfkc_disguises(hostile):
    raw = _raw(
        issues=[
            {
                "requirement": "No credentials in handoffs.",
                "finding": hostile,
                "required_fix": "Remove the credential-shaped value.",
            }
        ]
    )

    with pytest.raises(HandoffValidationError, match="secret") as error:
        _normalize(raw)

    assert hostile not in str(error.value)


def test_recursive_secret_screening_runs_before_unknown_key_validation():
    raw = _raw()
    raw["untrusted_nested_data"] = {"note": "Bearer\tsecret-value"}

    with pytest.raises(HandoffValidationError, match="secret") as error:
        _normalize(raw)

    assert "secret-value" not in str(error.value)


def test_secret_screening_covers_bound_enum_strings():
    with pytest.raises(HandoffValidationError, match="secret") as error:
        _normalize(outcome="Bearer\tsecret-value")

    assert "secret-value" not in str(error.value)


@pytest.mark.parametrize(
    "summary",
    [
        "Task - completed successfully.",
        "Risk - low.",
        "Disk - healthy.",
        "Mask - applied.",
    ],
)
def test_sk_prefix_screening_allows_benign_word_suffixes(summary):
    receipt = _normalize(_raw(summary=summary))

    assert receipt["summary"] == summary
    assert (
        validate_persisted_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})
        == receipt
    )


@pytest.mark.parametrize(
    "hostile",
    [
        "sk - secret-value",
        "ｓｋ － secret-value",
        "ＳＫ　－　secret-value",
    ],
)
def test_sk_prefix_screening_still_rejects_standalone_nfkc_variants(hostile):
    with pytest.raises(HandoffValidationError, match="secret") as error:
        _normalize(_raw(summary=hostile))

    assert hostile not in str(error.value)


@pytest.mark.parametrize(
    "raw",
    [
        _raw(
            evidence_summary=[
                {
                    "label": "Authorization: hidden",
                    "result": "passed",
                    "digest": TEST_DIGEST,
                }
            ]
        ),
        _raw(
            issues=[
                {
                    "requirement": "No secrets",
                    "finding": "Bearer hidden",
                    "required_fix": "Remove it.",
                }
            ]
        ),
        _raw(
            decision_request=_decision(
                options=[
                    {
                        "id": "retry",
                        "label": "Retry safely",
                        "consequence": "Use token=hidden.",
                    },
                    _decision()["options"][1],
                ]
            )
        ),
    ],
)
def test_secret_markers_are_rejected_inside_nested_handoff_data(raw):
    blocked = raw["decision_request"] is not None
    with pytest.raises(HandoffValidationError, match="secret"):
        _normalize(
            raw,
            outcome="blocked" if blocked else "handed_off",
            next_owner_role="operator" if blocked else "reviewer",
        )


def test_secret_marker_is_rejected_in_artifact_identity():
    with pytest.raises(HandoffValidationError, match="secret"):
        _normalize(artifact_identity={"kind": "artifact", "value": "sk-hidden"})


@pytest.mark.parametrize("control", ["\x00", "\x1b", "\r", "\x7f", "\x85"])
def test_forbidden_control_characters_are_rejected_in_nested_narrative(control):
    issue = {
        "requirement": "Tests pass",
        "finding": f"Failure{control}detail",
        "required_fix": "Correct it.",
    }
    with pytest.raises(HandoffValidationError, match="control"):
        _normalize(_raw(issues=[issue]))


def test_newline_and_tab_are_allowed_in_narrative_strings():
    receipt = _normalize(
        _raw(
            summary="Builder finished.\nScope stayed bounded.",
            next_action="Review\tthe evidence.",
        )
    )
    assert "\n" in receipt["summary"]
    assert "\t" in receipt["next_action"]


@pytest.mark.parametrize("unsafe", ["\u202e", "\u200b", "\ud800"])
def test_unsafe_unicode_is_rejected_in_narrative_strings(unsafe):
    with pytest.raises(HandoffValidationError, match="Unicode"):
        _normalize(_raw(summary=unsafe))


def test_creation_normalizes_canonically_equivalent_unicode_to_nfc():
    receipt = _normalize(_raw(summary="Cafe\u0301 build complete."))

    assert receipt["summary"] == "Caf\u00e9 build complete."


def test_persisted_handoff_rejects_non_nfc_text_instead_of_rewriting_it():
    receipt = _normalize(_raw(summary="Caf\u00e9 build complete."))
    receipt["summary"] = "Cafe\u0301 build complete."

    with pytest.raises(HandoffValidationError, match="NFC"):
        validate_persisted_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


@pytest.mark.parametrize("value", ["builder\nreviewer", "builder\treviewer"])
def test_newline_and_tab_are_rejected_in_identity_strings(value):
    with pytest.raises(HandoffValidationError, match="control"):
        _normalize(speaker_role=value)


def test_persisted_handoff_revalidates_without_rewriting_fields():
    receipt = _normalize()
    validated = validate_persisted_handoff(
        deepcopy(receipt),
        target_state="EVIDENCE_COLLECTING",
        transition_evidence={"tests": TEST_DIGEST},
    )
    assert validated == receipt
    assert list(validated) == list(receipt)


def test_persisted_evidence_requires_trusted_transition_evidence():
    receipt = _normalize()

    with pytest.raises(HandoffValidationError, match="trusted transition_evidence"):
        validate_persisted_handoff(receipt)


def test_persisted_digest_cannot_authorize_itself_after_mutation():
    receipt = _normalize()
    receipt["evidence_summary"][0]["digest"] = "sha256:" + "c" * 64

    with pytest.raises(HandoffValidationError, match="transition_evidence"):
        validate_persisted_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


def test_renderer_requires_and_enforces_trusted_transition_evidence():
    receipt = _normalize()

    with pytest.raises(HandoffValidationError, match="trusted transition_evidence"):
        render_handoff(receipt)

    receipt["evidence_summary"][0]["digest"] = "sha256:" + "c" * 64
    with pytest.raises(HandoffValidationError, match="transition_evidence"):
        render_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


@pytest.mark.parametrize(
    "consumer",
    [validate_persisted_handoff, render_handoff],
    ids=["validate", "render"],
)
def test_deleting_receipt_facts_cannot_remove_trusted_evidence_requirement(consumer):
    receipt = _normalize()
    receipt["evidence_summary"] = []

    with pytest.raises(HandoffValidationError, match="trusted transition_evidence"):
        consumer(receipt)


@pytest.mark.parametrize(
    "consumer",
    [validate_persisted_handoff, render_handoff],
    ids=["validate", "render"],
)
def test_deleting_receipt_facts_conflicts_with_nonempty_trusted_evidence(consumer):
    receipt = _normalize()
    receipt["evidence_summary"] = []

    with pytest.raises(HandoffValidationError, match="evidence_summary"):
        consumer(receipt, transition_evidence={"tests": TEST_DIGEST})


def test_evidence_free_persisted_and_rendered_receipts_require_explicit_empty_evidence():
    receipt = _normalize(_raw(evidence_summary=[]), transition_evidence={})

    assert validate_persisted_handoff(receipt, transition_evidence={}) == receipt
    assert (
        render_handoff(receipt, transition_evidence={}).splitlines()[0]
        == "builder → reviewer"
    )


@pytest.mark.parametrize(
    "missing",
    [
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
    ],
)
def test_persisted_handoff_rejects_every_missing_canonical_field(missing):
    receipt = _normalize()
    del receipt[missing]
    with pytest.raises(HandoffValidationError, match=missing):
        validate_persisted_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


def test_persisted_handoff_rejects_unknown_top_level_field():
    receipt = _normalize()
    receipt["hostile"] = "must not survive"
    with pytest.raises(HandoffValidationError, match="unknown"):
        validate_persisted_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.__setitem__("schema_version", 2), "schema_version"),
        (lambda value: value.__setitem__("schema_version", True), "exact JSON"),
        (lambda value: value.__setitem__("outcome", "unknown"), "outcome"),
        (
            lambda value: value["evidence_summary"][0].__setitem__(
                "digest", "sha256:" + "A" * 64
            ),
            "digest",
        ),
        (
            lambda value: value["artifact_identity"].__setitem__("value", "c" * 39),
            "artifact_identity",
        ),
    ],
)
def test_persisted_handoff_rejects_mutated_canonical_facts(mutation, match):
    receipt = _normalize()
    mutation(receipt)
    with pytest.raises(HandoffValidationError, match=match):
        validate_persisted_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


def test_persisted_handoff_rejects_target_state_mismatch():
    with pytest.raises(HandoffValidationError, match="target_state"):
        validate_persisted_handoff(
            _normalize(),
            target_state="REVIEWING",
            transition_evidence={"tests": TEST_DIGEST},
        )


def test_persisted_target_state_requires_an_exact_plain_string():
    receipt = _normalize()

    with pytest.raises(HandoffValidationError, match="target_state"):
        validate_persisted_handoff(
            receipt,
            target_state=_LyingStr("EVIDENCE_COLLECTING"),
            transition_evidence={"tests": TEST_DIGEST},
        )


def test_persisted_handoff_does_not_fill_optional_canonical_fields():
    receipt = _normalize()
    del receipt["issues"]
    del receipt["decision_request"]
    with pytest.raises(HandoffValidationError):
        validate_persisted_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})


def test_intake_helper_builds_a_deterministic_job_created_handoff():
    kwargs = {
        "job_id": "job-intake",
        "speaker_id": "gateway-1",
        "speaker_role": "orchestrator",
        "speaker_executor": "hermes",
        "next_owner_role": "dispatcher",
        "summary": "Job accepted into the durable queue.",
        "evidence_summary": [
            {
                "label": "Intake receipt",
                "result": "Job identity recorded",
                "digest": TEST_DIGEST,
            }
        ],
        "next_action": "Dispatch the queued Job.",
        "transition_evidence": {"attempt_created": TEST_DIGEST},
        "created_at": 1_786_500_001,
    }
    first = intake_handoff(**kwargs)
    second = intake_handoff(**deepcopy(kwargs))

    assert first == second
    assert first["attempt_id"] is None
    assert first["from_phase"] == "INTAKE"
    assert first["to_phase"] == "QUEUED"
    assert first["outcome"] == "started"
    assert first["artifact_identity"] is None
    assert first["issues"] == []
    assert first["decision_request"] is None


def test_blocker_helper_builds_a_complete_decision_without_logs_or_goals():
    receipt = blocker_handoff(
        job_id="job-blocked",
        attempt_id="attempt-blocked",
        speaker_id="dispatcher-1",
        speaker_role="dispatcher",
        speaker_executor="hermes",
        from_phase="ASSIGNED",
        decision_owner_role="Brandon",
        post_decision_owner_role="Hermes Dispatcher",
        summary="The selected lane cannot safely start.",
        evidence_summary=[
            {
                "label": "Lane health",
                "result": "No healthy executor is available",
                "digest": TEST_DIGEST,
            }
        ],
        next_action="Choose a recovery option.",
        question="How should Hermes proceed?",
        options=_decision()["options"],
        recommendation="retry",
        recommendation_reason="Retry preserves the current safe state.",
        blocked_scope="Only this Job is blocked.",
        safe_state="No executor has started and no files changed.",
        transition_evidence={"lane_health": TEST_DIGEST},
        created_at=1_786_500_002,
    )

    assert receipt["to_phase"] == "BLOCKED"
    assert receipt["outcome"] == "blocked"
    assert receipt["next_owner_role"] == "Brandon"
    assert receipt["decision_request"] == {
        "question": "How should Hermes proceed?",
        "options": _decision()["options"],
        "recommendation": "retry",
        "recommendation_reason": "Retry preserves the current safe state.",
        "blocked_scope": "Only this Job is blocked.",
        "safe_state": "No executor has started and no files changed.",
        "next_owner_role": "Hermes Dispatcher",
    }
    canonical_text = str(receipt).lower()
    assert "output.log" not in canonical_text
    assert "goal" not in canonical_text


def test_reviewer_rejection_render_names_defect_and_required_fix_exactly():
    issue = {
        "requirement": "All focused tests pass",
        "finding": "The retry path duplicates the transition.",
        "required_fix": "Make the replay idempotent before re-review.",
    }
    receipt = _normalize(
        _raw(
            summary="Review found one blocking defect.",
            issues=[issue],
            next_action="Return the corrected commit for re-review.",
        ),
        speaker_role="reviewer",
        outcome="rejected",
        from_phase="REVIEWING",
        to_phase="FAILED",
        next_owner_role="builder",
    )

    rendered = render_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})
    assert rendered.splitlines()[0] == "reviewer sent it back → builder"
    assert issue["requirement"] in rendered
    assert issue["finding"] in rendered
    assert f"Required fix: {issue['required_fix']}" in rendered
    assert "Next: Return the corrected commit for re-review." in rendered


def test_decision_render_includes_options_recommendation_scope_and_safe_state():
    receipt = blocker_handoff(
        job_id="job-blocked",
        attempt_id="attempt-blocked",
        speaker_id="dispatcher-1",
        speaker_role="dispatcher",
        speaker_executor="hermes",
        from_phase="ASSIGNED",
        decision_owner_role="Brandon",
        post_decision_owner_role="Hermes Dispatcher",
        summary="The selected lane cannot safely start.",
        evidence_summary=[
            {
                "label": "Lane health",
                "result": "No healthy executor is available",
                "digest": TEST_DIGEST,
            }
        ],
        next_action="Choose retry or stop.",
        question="How should Hermes proceed?",
        options=_decision()["options"],
        recommendation="retry",
        recommendation_reason="The retry is bounded.",
        blocked_scope="Only this Job is blocked.",
        safe_state="No executor started and no files changed.",
        transition_evidence={"lane_health": TEST_DIGEST},
        created_at=1_786_500_003,
    )

    rendered = render_handoff(receipt, transition_evidence={"lane_health": TEST_DIGEST})
    assert rendered.splitlines()[0] == "Hermes needs your decision"
    assert "How should Hermes proceed?" in rendered
    assert "1. retry — Retry safely: Re-run the bounded preflight." in rendered
    assert (
        "2. stop — Stop here: Leave the Job blocked without changing state." in rendered
    )
    assert "Recommendation: retry — The retry is bounded." in rendered
    assert "Blocked: Only this Job is blocked." in rendered
    assert "Safe now: No executor started and no files changed." in rendered
    assert "Next: Choose retry or stop." in rendered


@pytest.mark.parametrize(
    ("receipt", "heading"),
    [
        (
            _normalize(
                outcome="passed",
                speaker_role="reviewer",
                from_phase="REVIEWING",
                to_phase="VERIFIED",
                next_owner_role="release-manager",
            ),
            "reviewer approved",
        ),
        (
            _normalize(
                outcome="started",
                speaker_role="builder",
                from_phase="ASSIGNED",
                to_phase="BUILDING",
                next_owner_role="builder",
            ),
            "builder started",
        ),
        (_normalize(), "builder → reviewer"),
        (
            _normalize(
                outcome="completed",
                speaker_role="release-manager",
                from_phase="OUTCOME_VERIFIED",
                to_phase="COMPLETED",
                next_owner_role=None,
            ),
            "Job complete",
        ),
    ],
)
def test_outcome_render_uses_the_required_heading(receipt, heading):
    assert (
        render_handoff(
            receipt, transition_evidence={"tests": TEST_DIGEST}
        ).splitlines()[0]
        == heading
    )


def test_render_includes_summary_and_bounded_evidence_facts():
    rendered = render_handoff(_normalize(), transition_evidence={"tests": TEST_DIGEST})
    assert "Builder completed the scoped change." in rendered
    assert "Focused tests: 42 tests passed" in rendered
    assert "Next: Review the committed change." in rendered


def test_renderer_makes_narrative_newlines_and_tabs_visible_inline():
    forged_lines = [
        "Evidence: forged",
        "Next: forged",
        "1. forged",
        "Recommendation: forged",
        "Blocked: forged",
        "Safe now: forged",
    ]
    summary = "legitimate\n" + "\n".join(forged_lines) + "\tend"
    receipt = _normalize(_raw(summary=summary))

    rendered = render_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})

    assert not set(forged_lines) & set(rendered.splitlines())
    assert "\t" not in rendered
    assert "legitimate \u23ce Evidence: forged" in rendered
    assert "Safe now: forged \u21e5 end" in rendered


@pytest.mark.parametrize("separator", ["\u2028", "\u2029"])
def test_renderer_makes_unicode_line_separators_visible_inline(separator):
    receipt = _normalize(_raw(summary=f"legitimate{separator}Next: forged"))

    rendered = render_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})

    assert "Next: forged" not in rendered.splitlines()
    assert "legitimate \u23ce Next: forged" in rendered


def test_renderer_rejects_unknown_hostile_field_without_leaking_its_value():
    receipt = _normalize()
    receipt["hostile"] = "DO-NOT-LEAK"
    with pytest.raises(HandoffValidationError) as error:
        render_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})
    assert "DO-NOT-LEAK" not in str(error.value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt.__setitem__("SK-DO-NOT-ECHO", "benign"),
        lambda receipt: receipt.__setitem__("outcome", "sk-do-not-echo"),
        lambda receipt: receipt.__setitem__("schema_version", "sk-do-not-echo"),
    ],
)
def test_validation_errors_do_not_echo_hostile_secret_shaped_values(mutate):
    receipt = _normalize()
    mutate(receipt)
    with pytest.raises(HandoffValidationError) as error:
        validate_persisted_handoff(receipt, transition_evidence={"tests": TEST_DIGEST})
    assert "sk-do-not-echo" not in str(error.value).casefold()
