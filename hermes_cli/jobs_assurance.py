"""Deterministic outcome, risk, retry, and closure contracts for Jobs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from hermes_cli import jobs_receipts


_OUTCOME_MODES = frozenset({"runtime", "visual", "integration", "not_applicable"})
_RISK_DOMAINS = frozenset({"money", "permissions", "state", "none"})
_REVIEW_VERDICTS = frozenset({"PASS", "NEEDS_CHANGES", "UNABLE_TO_VERIFY"})
_DIGEST = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_MAX_TEXT = 4096
_MAX_ITEMS = 64
_CONTRACT_FIELDS = frozenset(
    {
        "critical_user_journey",
        "success_metric",
        "outcome_mode",
        "verification_steps",
        "max_attempts",
        "wall_clock_budget_seconds",
        "risk_domains",
        "consumers",
        "egress_paths",
        "rollback_behavior",
        "knowledge_closure_required",
        "not_applicable_reason",
    }
)


class InvalidAssuranceContract(ValueError):
    """Raised when a Job has no complete, bounded definition of done."""


class InvalidOutcomeEvidence(ValueError):
    """Raised when outcome evidence is malformed or unbounded."""


class InvalidClosureReceipt(ValueError):
    """Raised when durable-knowledge closure is incomplete."""


def _text(value: object, field: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TEXT:
        raise InvalidAssuranceContract(f"{field} must be non-empty bounded text")
    return value.strip()


def _strings(value: object, field: str, *, required: bool = True) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise InvalidAssuranceContract(f"{field} must be a list")
    items = tuple(_text(item, field) for item in value)
    if required and not items:
        raise InvalidAssuranceContract(f"{field} must not be empty")
    if len(items) > _MAX_ITEMS or len(set(items)) != len(items):
        raise InvalidAssuranceContract(f"{field} is invalid")
    return tuple(item for item in items if item is not None)


def _positive_int(value: object, field: str, *, ceiling: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
        raise InvalidAssuranceContract(f"{field} must be in 1..{ceiling}")
    return value


@dataclass(frozen=True)
class AssuranceContract:
    critical_user_journey: str
    success_metric: str
    outcome_mode: str
    verification_steps: tuple[str, ...]
    max_attempts: int
    wall_clock_budget_seconds: int
    risk_domains: tuple[str, ...]
    consumers: tuple[str, ...]
    egress_paths: tuple[str, ...]
    rollback_behavior: str
    knowledge_closure_required: bool
    not_applicable_reason: str | None = None

    @classmethod
    def from_mapping(cls, value: object) -> "AssuranceContract":
        if not isinstance(value, Mapping):
            raise InvalidAssuranceContract("assurance contract must be an object")
        unknown = set(value) - _CONTRACT_FIELDS
        if unknown:
            raise InvalidAssuranceContract("assurance contract has unknown fields")
        mode = _text(value.get("outcome_mode"), "outcome_mode")
        if mode not in _OUTCOME_MODES:
            raise InvalidAssuranceContract("outcome_mode is unsupported")
        risks = _strings(value.get("risk_domains"), "risk_domains")
        if any(item not in _RISK_DOMAINS for item in risks):
            raise InvalidAssuranceContract("risk_domains contains an unsupported value")
        if "none" in risks and len(risks) != 1:
            raise InvalidAssuranceContract("risk_domains cannot mix none with risks")
        high_risk = any(item != "none" for item in risks)
        consumers = _strings(
            value.get("consumers", []), "consumers", required=high_risk
        )
        egress_paths = _strings(
            value.get("egress_paths", []), "egress_paths", required=high_risk
        )
        rollback = _text(
            value.get("rollback_behavior"),
            "rollback_behavior",
            required=high_risk,
        )
        if rollback is None:
            rollback = "not applicable"
        not_applicable = _text(
            value.get("not_applicable_reason"),
            "not_applicable_reason",
            required=mode == "not_applicable",
        )
        closure = value.get("knowledge_closure_required")
        if not isinstance(closure, bool):
            raise InvalidAssuranceContract(
                "knowledge_closure_required must be a boolean"
            )
        return cls(
            critical_user_journey=_text(
                value.get("critical_user_journey"), "critical_user_journey"
            ),
            success_metric=_text(value.get("success_metric"), "success_metric"),
            outcome_mode=mode,
            verification_steps=_strings(
                value.get("verification_steps"), "verification_steps"
            ),
            max_attempts=_positive_int(
                value.get("max_attempts"), "max_attempts", ceiling=20
            ),
            wall_clock_budget_seconds=_positive_int(
                value.get("wall_clock_budget_seconds"),
                "wall_clock_budget_seconds",
                ceiling=604_800,
            ),
            risk_domains=risks,
            consumers=consumers,
            egress_paths=egress_paths,
            rollback_behavior=rollback,
            knowledge_closure_required=closure,
            not_applicable_reason=not_applicable,
        )

    def to_mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
            "critical_user_journey": self.critical_user_journey,
            "success_metric": self.success_metric,
            "outcome_mode": self.outcome_mode,
            "verification_steps": list(self.verification_steps),
            "max_attempts": self.max_attempts,
            "wall_clock_budget_seconds": self.wall_clock_budget_seconds,
            "risk_domains": list(self.risk_domains),
            "consumers": list(self.consumers),
            "egress_paths": list(self.egress_paths),
            "rollback_behavior": self.rollback_behavior,
            "knowledge_closure_required": self.knowledge_closure_required,
        }
        if self.not_applicable_reason is not None:
            value["not_applicable_reason"] = self.not_applicable_reason
        return value

    @property
    def digest(self) -> str:
        return jobs_receipts.digest_bytes(
            jobs_receipts.canonical_json_bytes(self.to_mapping())
        )


@dataclass(frozen=True)
class RetryProgressDecision:
    action: Literal["RETRY", "BLOCKED"]
    reason_code: str


def assess_retry_progress(
    *, prior_failure: str | None, response_change: str | None, result_delta: str | None
) -> RetryProgressDecision:
    for value, reason in (
        (prior_failure, "RETRY_MISSING_PRIOR_FAILURE"),
        (response_change, "RETRY_MISSING_RESPONSE_CHANGE"),
        (result_delta, "RETRY_MISSING_RESULT_DELTA"),
    ):
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            return RetryProgressDecision("BLOCKED", reason)
    if len({prior_failure, response_change, result_delta}) != 3:
        return RetryProgressDecision("BLOCKED", "RETRY_EVIDENCE_NOT_DISTINCT")
    return RetryProgressDecision("RETRY", "PROGRESS_PROVED")


@dataclass(frozen=True)
class ContinuationDecision:
    action: Literal["CONTINUE", "STOP"]
    reason_code: str


def decide_continuation(
    contract: AssuranceContract, *, attempts_started: int, elapsed_seconds: int
) -> ContinuationDecision:
    if attempts_started >= contract.max_attempts or elapsed_seconds >= contract.wall_clock_budget_seconds:
        return ContinuationDecision("STOP", "BUDGET_EXHAUSTED")
    return ContinuationDecision("CONTINUE", "BUDGET_AVAILABLE")


def attach_to_preflight(snapshot, value: object):
    """Return a preflight snapshot carrying one validated canonical contract."""
    contract = AssuranceContract.from_mapping(value)
    return replace(snapshot, assurance_contract=contract.to_mapping())


@dataclass(frozen=True)
class ReviewResult:
    verdict: str
    findings: tuple[str, ...]
    checks_run: tuple[str, ...]

    @property
    def authorizes_completion(self) -> bool:
        return self.verdict == "PASS"


def validate_review_result(value: object) -> ReviewResult:
    if not isinstance(value, Mapping):
        raise ValueError("review result must be an object")
    verdict = value.get("verdict")
    if verdict not in _REVIEW_VERDICTS:
        raise ValueError("review verdict is invalid")
    try:
        findings = _strings(value.get("findings", []), "findings", required=False)
        checks = _strings(value.get("checks_run"), "checks_run")
    except InvalidAssuranceContract as exc:
        raise ValueError(str(exc)) from exc
    if verdict != "PASS" and not findings:
        raise ValueError("non-passing review requires findings")
    return ReviewResult(str(verdict), findings, checks)


@dataclass(frozen=True)
class OutcomeEvidence:
    critical_user_journey: str
    verdict: str
    environment: str
    checks_run: tuple[str, ...]
    observed_behavior: str
    artifact_digests: tuple[str, ...]
    observed_at: int

    @classmethod
    def from_mapping(cls, value: object) -> "OutcomeEvidence":
        if not isinstance(value, Mapping):
            raise InvalidOutcomeEvidence("outcome evidence must be an object")
        try:
            journey = _text(value.get("critical_user_journey"), "critical_user_journey")
            environment = _text(value.get("environment"), "environment")
            behavior = _text(value.get("observed_behavior"), "observed_behavior")
            checks = _strings(value.get("checks_run"), "checks_run")
            digests = _strings(value.get("artifact_digests"), "artifact_digests")
        except InvalidAssuranceContract as exc:
            raise InvalidOutcomeEvidence(str(exc)) from exc
        verdict = value.get("verdict")
        if verdict not in {"PASS", "FAIL", "UNABLE_TO_VERIFY"}:
            raise InvalidOutcomeEvidence("verdict is invalid")
        if not all(_DIGEST.fullmatch(item) for item in digests):
            raise InvalidOutcomeEvidence("artifact_digests are invalid")
        observed_at = value.get("observed_at")
        if isinstance(observed_at, bool) or not isinstance(observed_at, int) or observed_at < 0:
            raise InvalidOutcomeEvidence("observed_at is invalid")
        return cls(journey, str(verdict), environment, checks, behavior, digests, observed_at)

    def to_mapping(self) -> dict[str, object]:
        return {
            "critical_user_journey": self.critical_user_journey,
            "verdict": self.verdict,
            "environment": self.environment,
            "checks_run": list(self.checks_run),
            "observed_behavior": self.observed_behavior,
            "artifact_digests": list(self.artifact_digests),
            "observed_at": self.observed_at,
        }

    @property
    def digest(self) -> str:
        return jobs_receipts.digest_bytes(
            jobs_receipts.canonical_json_bytes(self.to_mapping())
        )


@dataclass(frozen=True)
class OutcomeDecision:
    status: Literal["PASS", "BLOCKED"]
    reason_code: str
    evidence_digest: str


def verify_outcome(
    contract: AssuranceContract, evidence: OutcomeEvidence
) -> OutcomeDecision:
    if evidence.critical_user_journey != contract.critical_user_journey:
        return OutcomeDecision("BLOCKED", "JOURNEY_MISMATCH", evidence.digest)
    if evidence.verdict != "PASS":
        return OutcomeDecision("BLOCKED", "OUTCOME_NOT_PROVED", evidence.digest)
    missing = set(contract.verification_steps) - set(evidence.checks_run)
    if missing:
        return OutcomeDecision("BLOCKED", "VERIFICATION_STEPS_MISSING", evidence.digest)
    return OutcomeDecision("PASS", "OK", evidence.digest)


@dataclass(frozen=True)
class KnowledgeClosureReceipt:
    disposition: str
    canonical_note_path: str | None
    retrieval_confirmed: bool
    steward_receipt_digest: str
    rationale: str | None = None

    @classmethod
    def from_mapping(cls, value: object) -> "KnowledgeClosureReceipt":
        if not isinstance(value, Mapping):
            raise InvalidClosureReceipt("closure receipt must be an object")
        disposition = value.get("disposition")
        if disposition not in {"ACCEPTED", "REJECTED", "NOT_APPLICABLE"}:
            raise InvalidClosureReceipt("disposition is invalid")
        path = value.get("canonical_note_path")
        if path is not None and (
            not isinstance(path, str) or not path.strip() or len(path) > _MAX_TEXT
        ):
            raise InvalidClosureReceipt("canonical_note_path is invalid")
        confirmed = value.get("retrieval_confirmed")
        if not isinstance(confirmed, bool):
            raise InvalidClosureReceipt("retrieval_confirmed must be a boolean")
        digest = value.get("steward_receipt_digest")
        if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
            raise InvalidClosureReceipt("steward_receipt_digest is invalid")
        rationale = value.get("rationale")
        if rationale is not None and (
            not isinstance(rationale, str)
            or not rationale.strip()
            or len(rationale) > _MAX_TEXT
        ):
            raise InvalidClosureReceipt("rationale is invalid")
        if disposition == "ACCEPTED" and not path:
            raise InvalidClosureReceipt("canonical_note_path is required")
        if disposition == "ACCEPTED" and not confirmed:
            raise InvalidClosureReceipt("retrieval_confirmed must be true")
        if disposition == "REJECTED" and not rationale:
            raise InvalidClosureReceipt("rationale is required")
        return cls(str(disposition), path, confirmed, digest, rationale)

    @property
    def closed(self) -> bool:
        return self.disposition in {"ACCEPTED", "REJECTED", "NOT_APPLICABLE"}

    def to_mapping(self) -> dict[str, object]:
        value: dict[str, object] = {
            "disposition": self.disposition,
            "canonical_note_path": self.canonical_note_path,
            "retrieval_confirmed": self.retrieval_confirmed,
            "steward_receipt_digest": self.steward_receipt_digest,
        }
        if self.rationale is not None:
            value["rationale"] = self.rationale
        return value

    @property
    def digest(self) -> str:
        return jobs_receipts.digest_bytes(
            jobs_receipts.canonical_json_bytes(self.to_mapping())
        )


__all__ = [
    "AssuranceContract",
    "ContinuationDecision",
    "InvalidAssuranceContract",
    "InvalidClosureReceipt",
    "InvalidOutcomeEvidence",
    "KnowledgeClosureReceipt",
    "OutcomeDecision",
    "OutcomeEvidence",
    "ReviewResult",
    "RetryProgressDecision",
    "assess_retry_progress",
    "attach_to_preflight",
    "decide_continuation",
    "validate_review_result",
    "verify_outcome",
]
