"""Provider-free tests for the versioned Jobs model-routing policy."""

from __future__ import annotations

import json

import pytest

from hermes_cli import jobs_model_policy as policy_module


def _traits(**overrides):
    values = {
        "work_kind": "implementation",
        "scope": "multi_file",
        "risk": "routine",
        "requirements_complete": True,
    }
    values.update(overrides)
    return policy_module.RoutingTraits(**values)


def _manifest():
    return {
        "schema_version": 1,
        "policy_version": "test.1",
        "source_kind": "operator_policy",
        "models": [
            {
                "id": "claude-fable-5",
                "executor": "claude",
                "policy_role": "design_fidelity",
            },
            {
                "id": "claude-opus-5",
                "executor": "claude",
                "policy_role": "high_risk_generalist",
            },
            {
                "id": "gpt-5.6-sol",
                "executor": "codex",
                "policy_role": "specified_implementation",
            },
        ],
        "rules": [
            {
                "id": "critical-opus",
                "select": "claude-opus-5",
                "risk_in": ["critical"],
                "any_true": [
                    "architectural_judgment",
                    "security",
                    "money",
                    "schema_or_migration",
                    "irreversible",
                ],
            },
            {
                "id": "design-fidelity-fable",
                "select": "claude-fable-5",
                "work_kind_in": ["design", "specification"],
                "any_true": ["visual_fidelity"],
            },
            {
                "id": "specified-build-sol",
                "select": "gpt-5.6-sol",
                "work_kind_in": [
                    "implementation",
                    "refactor",
                    "testing",
                    "debugging",
                ],
                "requirements_complete": True,
            },
            {
                "id": "unknown-opus",
                "select": "claude-opus-5",
                "default": True,
            },
        ],
        "unverified_facts": [
            "benchmarks",
            "pricing",
            "provider_capacity",
            "quota",
        ],
    }


def _write_manifest(tmp_path, data):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_visual_fidelity_selects_fable_unless_critical():
    policy = policy_module.load_model_policy()

    result = policy_module.select_model(
        _traits(visual_fidelity=True),
        policy,
    )

    assert result.selected_model == "claude-fable-5"
    assert result.executor == "claude"
    assert result.rule_id == "design-fidelity-fable"
    assert result.fit_status == "available"


def test_critical_risk_precedes_visual_fidelity_and_selects_opus():
    result = policy_module.select_model(
        _traits(risk="critical", visual_fidelity=True),
        policy_module.load_model_policy(),
    )

    assert result.selected_model == "claude-opus-5"
    assert result.rule_id == "critical-opus"


@pytest.mark.parametrize(
    "trait",
    [
        "architectural_judgment",
        "security",
        "money",
        "schema_or_migration",
        "irreversible",
    ],
)
def test_each_elevated_risk_trait_selects_opus(trait):
    result = policy_module.select_model(
        _traits(**{trait: True}),
        policy_module.load_model_policy(),
    )

    assert result.selected_model == "claude-opus-5"
    assert result.rule_id == "critical-opus"


def test_well_specified_implementation_selects_sol_but_marks_executor_unavailable():
    result = policy_module.select_model(
        _traits(),
        policy_module.load_model_policy(),
    )

    assert result.selected_model == "gpt-5.6-sol"
    assert result.executor == "codex"
    assert result.fit_status == "best_fit_unavailable"
    assert result.next_valid_model == "claude-opus-5"
    assert "no_jobs_codex_executor" in result.reasons


def test_real_injected_codex_executor_makes_sol_available():
    result = policy_module.select_model(
        _traits(),
        policy_module.load_model_policy(),
        executor_availability={"claude": True, "codex": True},
    )

    assert result.selected_model == "gpt-5.6-sol"
    assert result.fit_status == "available"
    assert result.next_valid_model is None


def test_every_decision_considers_each_model_once():
    result = policy_module.select_model(
        _traits(),
        policy_module.load_model_policy(),
    )

    assert [alternative.model for alternative in result.alternatives] == [
        "claude-fable-5",
        "claude-opus-5",
        "gpt-5.6-sol",
    ]
    assert len({alternative.model for alternative in result.alternatives}) == 3
    assert all(alternative.disposition for alternative in result.alternatives)


def test_missing_structured_traits_are_named_unknowns_and_default_to_opus():
    result = policy_module.select_model(
        policy_module.RoutingTraits(),
        policy_module.load_model_policy(),
    )

    assert result.selected_model == "claude-opus-5"
    assert result.rule_id == "unknown-opus"
    assert "work_kind" in result.unknowns
    assert "scope" in result.unknowns
    assert "risk" in result.unknowns
    assert "requirements_complete" in result.unknowns


def test_explicit_supported_model_is_recorded_as_an_operator_override():
    result = policy_module.select_model(
        _traits(explicit_model="claude-fable-5"),
        policy_module.load_model_policy(),
    )

    assert result.selected_model == "claude-fable-5"
    assert result.rule_id == "operator-override"
    assert "explicit_model" in result.reasons


def test_explicit_unknown_model_fails_closed():
    with pytest.raises(policy_module.ModelPolicyError, match="unknown explicit model"):
        policy_module.select_model(
            _traits(explicit_model="kat-coder"),
            policy_module.load_model_policy(),
        )


def test_policy_digest_is_stable_for_the_same_manifest(tmp_path):
    path = _write_manifest(tmp_path, _manifest())

    first = policy_module.load_model_policy(path)
    second = policy_module.load_model_policy(path)

    assert first.digest == second.digest
    assert len(first.digest) == 64


@pytest.mark.parametrize(
    "field",
    ["benchmark", "price", "quota", "provider_capacity"],
)
def test_policy_refuses_unverified_factual_fields(tmp_path, field):
    manifest = _manifest()
    manifest["models"][0][field] = "invented"
    path = _write_manifest(tmp_path, manifest)

    with pytest.raises(policy_module.ModelPolicyError, match=field):
        policy_module.load_model_policy(path)


def test_policy_refuses_unknown_selected_model(tmp_path):
    manifest = _manifest()
    manifest["rules"][0]["select"] = "unregistered-model"

    with pytest.raises(policy_module.ModelPolicyError, match="unregistered-model"):
        policy_module.load_model_policy(_write_manifest(tmp_path, manifest))


def test_policy_refuses_duplicate_rule_ids(tmp_path):
    manifest = _manifest()
    manifest["rules"][1]["id"] = manifest["rules"][0]["id"]

    with pytest.raises(policy_module.ModelPolicyError, match="duplicate rule"):
        policy_module.load_model_policy(_write_manifest(tmp_path, manifest))


def test_routing_selection_is_frozen():
    result = policy_module.select_model(
        _traits(),
        policy_module.load_model_policy(),
    )

    with pytest.raises(Exception):
        result.selected_model = "claude-opus-5"  # type: ignore[misc]
