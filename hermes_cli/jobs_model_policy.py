"""Versioned, inspectable operator policy for Jobs model fit selection.

The policy intentionally describes operator preferences only. Executor
availability is supplied by the caller from observed runtime capabilities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Sequence


class ModelPolicyError(ValueError):
    """Raised when a model policy or requested override is invalid."""


@dataclass(frozen=True)
class RoutingTraits:
    """Structured facts used by the operator-authored routing policy."""

    work_kind: Optional[str] = None
    scope: Optional[str] = None
    risk: Optional[str] = None
    requirements_complete: Optional[bool] = None
    visual_fidelity: bool = False
    architectural_judgment: bool = False
    security: bool = False
    money: bool = False
    schema_or_migration: bool = False
    irreversible: bool = False
    explicit_model: Optional[str] = None


@dataclass(frozen=True)
class ModelReference:
    id: str
    executor: str
    policy_role: str


@dataclass(frozen=True)
class PolicyRule:
    id: str
    select: str
    risk_in: tuple[str, ...] = ()
    work_kind_in: tuple[str, ...] = ()
    any_true: tuple[str, ...] = ()
    requirements_complete: Optional[bool] = None
    default: bool = False


@dataclass(frozen=True)
class ModelPolicy:
    schema_version: int
    policy_version: str
    source_kind: str
    models: tuple[ModelReference, ...]
    rules: tuple[PolicyRule, ...]
    unverified_facts: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class ModelAlternative:
    model: str
    executor: str
    disposition: str


@dataclass(frozen=True)
class ModelSelection:
    policy_version: str
    policy_digest: str
    selected_model: str
    executor: str
    rule_id: str
    fit_status: str
    reasons: tuple[str, ...]
    alternatives: tuple[ModelAlternative, ...]
    unknowns: tuple[str, ...]
    next_valid_model: Optional[str]


_TOP_LEVEL_KEYS = {
    "schema_version",
    "policy_version",
    "source_kind",
    "models",
    "rules",
    "unverified_facts",
}
_MODEL_KEYS = {"id", "executor", "policy_role"}
_RULE_KEYS = {
    "id",
    "select",
    "risk_in",
    "work_kind_in",
    "any_true",
    "requirements_complete",
    "default",
}
_BOOL_TRAITS = {
    field.name
    for field in fields(RoutingTraits)
    if field.name
    in {
        "visual_fidelity",
        "architectural_judgment",
        "security",
        "money",
        "schema_or_migration",
        "irreversible",
    }
}
_DEFAULT_POLICY_PATH = Path(__file__).with_name("data") / "jobs-model-policy.v1.json"
_DEFAULT_EXECUTOR_AVAILABILITY = MappingProxyType({"claude": True, "codex": False})


def _reject_unknown_keys(value: Mapping[str, object], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ModelPolicyError(f"unknown {where} field: {unknown[0]}")


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelPolicyError(f"{where} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ModelPolicyError(f"{where} must be a list")
    result = tuple(_string(item, where) for item in value)
    if len(set(result)) != len(result):
        raise ModelPolicyError(f"duplicate value in {where}")
    return result


def _canonical_digest(data: Mapping[str, object]) -> str:
    payload = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_model_policy(path: Optional[Path] = None) -> ModelPolicy:
    """Load and strictly validate a versioned operator model policy."""

    policy_path = Path(path) if path is not None else _DEFAULT_POLICY_PATH
    try:
        raw = json.loads(
            policy_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ModelPolicyError(f"invalid JSON constant: {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPolicyError(f"cannot load model policy: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelPolicyError("model policy must be an object")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "policy")
    missing = sorted(_TOP_LEVEL_KEYS - set(raw))
    if missing:
        raise ModelPolicyError(f"missing policy field: {missing[0]}")

    schema_version = raw["schema_version"]
    if schema_version != 1:
        raise ModelPolicyError(f"unsupported schema_version: {schema_version}")
    policy_version = _string(raw["policy_version"], "policy_version")
    source_kind = _string(raw["source_kind"], "source_kind")
    if source_kind != "operator_policy":
        raise ModelPolicyError("source_kind must be operator_policy")

    raw_models = raw["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise ModelPolicyError("models must be a non-empty list")
    models: list[ModelReference] = []
    for index, item in enumerate(raw_models):
        if not isinstance(item, dict):
            raise ModelPolicyError(f"models[{index}] must be an object")
        _reject_unknown_keys(item, _MODEL_KEYS, f"models[{index}]")
        missing_model = sorted(_MODEL_KEYS - set(item))
        if missing_model:
            raise ModelPolicyError(f"missing models[{index}] field: {missing_model[0]}")
        models.append(
            ModelReference(
                id=_string(item["id"], f"models[{index}].id"),
                executor=_string(item["executor"], f"models[{index}].executor"),
                policy_role=_string(item["policy_role"], f"models[{index}].policy_role"),
            )
        )
    model_ids = [model.id for model in models]
    if len(set(model_ids)) != len(model_ids):
        raise ModelPolicyError("duplicate model id")

    raw_rules = raw["rules"]
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ModelPolicyError("rules must be a non-empty list")
    rules: list[PolicyRule] = []
    for index, item in enumerate(raw_rules):
        if not isinstance(item, dict):
            raise ModelPolicyError(f"rules[{index}] must be an object")
        _reject_unknown_keys(item, _RULE_KEYS, f"rules[{index}]")
        for required in ("id", "select"):
            if required not in item:
                raise ModelPolicyError(f"missing rules[{index}] field: {required}")
        selected = _string(item["select"], f"rules[{index}].select")
        if selected not in model_ids:
            raise ModelPolicyError(f"unknown selected model: {selected}")
        requirements_complete = item.get("requirements_complete")
        if requirements_complete is not None and not isinstance(requirements_complete, bool):
            raise ModelPolicyError(
                f"rules[{index}].requirements_complete must be boolean"
            )
        default = item.get("default", False)
        if not isinstance(default, bool):
            raise ModelPolicyError(f"rules[{index}].default must be boolean")
        any_true = _string_tuple(item.get("any_true"), f"rules[{index}].any_true")
        invalid_traits = sorted(set(any_true) - _BOOL_TRAITS)
        if invalid_traits:
            raise ModelPolicyError(f"unknown boolean trait: {invalid_traits[0]}")
        rules.append(
            PolicyRule(
                id=_string(item["id"], f"rules[{index}].id"),
                select=selected,
                risk_in=_string_tuple(item.get("risk_in"), f"rules[{index}].risk_in"),
                work_kind_in=_string_tuple(
                    item.get("work_kind_in"), f"rules[{index}].work_kind_in"
                ),
                any_true=any_true,
                requirements_complete=requirements_complete,
                default=default,
            )
        )
    rule_ids = [rule.id for rule in rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise ModelPolicyError("duplicate rule id")
    default_rules = [rule for rule in rules if rule.default]
    if len(default_rules) != 1 or rules[-1] != default_rules[0]:
        raise ModelPolicyError("policy requires exactly one final default rule")

    unverified_facts = _string_tuple(raw["unverified_facts"], "unverified_facts")
    return ModelPolicy(
        schema_version=1,
        policy_version=policy_version,
        source_kind=source_kind,
        models=tuple(models),
        rules=tuple(rules),
        unverified_facts=unverified_facts,
        digest=_canonical_digest(raw),
    )


def _rule_matches(rule: PolicyRule, traits: RoutingTraits) -> bool:
    if rule.default:
        return True
    conditions: list[bool] = []
    if rule.risk_in:
        conditions.append(traits.risk in rule.risk_in)
    if rule.work_kind_in:
        conditions.append(traits.work_kind in rule.work_kind_in)
    if rule.any_true:
        conditions.append(any(bool(getattr(traits, name)) for name in rule.any_true))
    if rule.requirements_complete is not None:
        conditions.append(traits.requirements_complete is rule.requirements_complete)
    return any(conditions) if rule.any_true and len(conditions) > 1 else all(conditions)


def _unknown_traits(traits: RoutingTraits) -> tuple[str, ...]:
    names = ("work_kind", "scope", "risk", "requirements_complete")
    return tuple(name for name in names if getattr(traits, name) in (None, ""))


def _alternative_disposition(
    model: ModelReference,
    selected_model: str,
    availability: Mapping[str, bool],
) -> str:
    if model.id == selected_model:
        return "selected_best_fit" if availability.get(model.executor, False) else "selected_unavailable"
    if not availability.get(model.executor, False):
        return "executor_unavailable"
    return "considered_not_selected"


def select_model(
    traits: RoutingTraits,
    policy: ModelPolicy,
    *,
    executor_availability: Optional[Mapping[str, bool]] = None,
) -> ModelSelection:
    """Select policy best fit without silently replacing an unavailable model."""

    availability = (
        dict(_DEFAULT_EXECUTOR_AVAILABILITY)
        if executor_availability is None
        else dict(executor_availability)
    )
    by_id = {model.id: model for model in policy.models}
    if traits.explicit_model:
        if traits.explicit_model not in by_id:
            raise ModelPolicyError(f"unknown explicit model: {traits.explicit_model}")
        selected_model = traits.explicit_model
        rule_id = "operator-override"
        reasons = ("explicit_model",)
    else:
        rule = next((item for item in policy.rules if _rule_matches(item, traits)), None)
        if rule is None:  # Defensive; strict loading requires a final default.
            raise ModelPolicyError("no model policy rule matched")
        selected_model = rule.select
        rule_id = rule.id
        reasons = (f"policy_rule:{rule.id}",)

    selected = by_id[selected_model]
    is_available = bool(availability.get(selected.executor, False))
    if is_available:
        fit_status = "available"
        next_valid_model = None
    else:
        fit_status = "best_fit_unavailable"
        reasons += (f"no_jobs_{selected.executor}_executor",)
        next_valid_model = next(
            (
                model.id
                for model in policy.models
                if model.id != selected.id
                and availability.get(model.executor, False)
                and model.id == "claude-opus-5"
            ),
            None,
        )
        if next_valid_model is None:
            next_valid_model = next(
                (
                    model.id
                    for model in policy.models
                    if model.id != selected.id and availability.get(model.executor, False)
                ),
                None,
            )

    alternatives = tuple(
        ModelAlternative(
            model=model.id,
            executor=model.executor,
            disposition=_alternative_disposition(model, selected_model, availability),
        )
        for model in policy.models
    )
    return ModelSelection(
        policy_version=policy.policy_version,
        policy_digest=policy.digest,
        selected_model=selected.id,
        executor=selected.executor,
        rule_id=rule_id,
        fit_status=fit_status,
        reasons=reasons,
        alternatives=alternatives,
        unknowns=_unknown_traits(traits),
        next_valid_model=next_valid_model,
    )
