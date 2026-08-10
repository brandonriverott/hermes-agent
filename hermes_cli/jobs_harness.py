"""Atomic, fail-closed preflight decisions for Jobs execution."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Mapping

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_db
from hermes_cli import jobs_receipts
from hermes_cli.jobs_receipts import JsonValue


CHECK_NAMES = (
    "file_paths",
    "permissions",
    "memory_store",
    "worktree",
    "tool_routing",
    "auth_check",
    "resource_budget",
)

_DETAIL_KEYS = frozenset(
    {
        "path",
        "paths",
        "repository",
        "branch",
        "base_commit",
        "lane_id",
        "executor",
        "model",
        "version",
        "free_bytes",
        "required_bytes",
        "reason",
        "permission",
        "authenticated",
        "expires_at",
        "host",
        "status",
    }
)
_DETAIL_LIMIT = 4096
_REASON_CODE = re.compile(r"\A[A-Z][A-Z0-9_]{0,63}\Z")
_SECRET_MARKERS = (
    "authorization:",
    "bearer ",
    "token=",
    "api_key",
    "private key",
    "sk-",
)


@dataclass(frozen=True)
class ProbeResult:
    status: Literal["PASS", "BLOCKED"]
    code: str
    safe_detail: Mapping[str, JsonValue]


Probe = Callable[[object], ProbeResult]


@dataclass(frozen=True)
class PreflightProbes:
    file_paths: Probe
    permissions: Probe
    memory_store: Probe
    worktree: Probe
    tool_routing: Probe
    auth_check: Probe
    resource_budget: Probe

    def with_result(self, name: str, result: ProbeResult) -> "PreflightProbes":
        if name not in CHECK_NAMES:
            raise ValueError(f"unknown preflight check: {name}")

        def fixed(_probe_input: object) -> ProbeResult:
            return result

        return replace(self, **{name: fixed})

    def with_failure(self, name: str, *, code: str) -> "PreflightProbes":
        return self.with_result(
            name,
            ProbeResult(status="BLOCKED", code=code, safe_detail={}),
        )


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Literal["PASS", "BLOCKED"]
    code: str
    observed_at: str
    evidence_digest: str
    safe_detail: Mapping[str, JsonValue]


@dataclass(frozen=True)
class PreflightSnapshot:
    job_id: str
    execution_spec_digest: str
    repository: Path
    base_commit: str
    branch: str
    output_parents: tuple[Path, ...]
    scoped_memory_paths: tuple[Path, ...]
    lane_id: str
    executor: str
    model: str
    observed_at: str
    expected_job_revision: int


@dataclass(frozen=True)
class PreflightDecision:
    preflight_id: str
    status: Literal["PASS", "BLOCKED"]
    failure_class: str | None
    checks: tuple[CheckResult, ...]
    receipt_id: str


@dataclass(frozen=True)
class ReceiptSigner:
    key_id: str
    private_key: Ed25519PrivateKey

    def sign(
        self, payload: Mapping[str, JsonValue], *, receipt_id: str
    ) -> dict[str, JsonValue]:
        return jobs_receipts.sign_receipt(
            payload,
            receipt_id=receipt_id,
            key_id=self.key_id,
            private_key=self.private_key,
        )


def _probe_input(name: str, snapshot: PreflightSnapshot) -> object:
    if name in {"file_paths", "permissions"}:
        return (snapshot.repository, snapshot.output_parents)
    if name == "memory_store":
        return snapshot.scoped_memory_paths
    if name == "worktree":
        return (snapshot.repository, snapshot.base_commit, snapshot.branch)
    if name == "tool_routing":
        return (snapshot.lane_id, snapshot.executor, snapshot.model)
    if name == "auth_check":
        return snapshot.lane_id
    return snapshot.repository


def _contains_secret(value: object) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in _SECRET_MARKERS)
    if isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_secret(item) for item in value.values())
    return False


def _bounded_probe_result(name: str, result: object) -> ProbeResult:
    if not isinstance(result, ProbeResult):
        return ProbeResult("BLOCKED", "INVALID_PROBE_RESULT", {})
    if result.status not in {"PASS", "BLOCKED"}:
        return ProbeResult("BLOCKED", "INVALID_PROBE_RESULT", {})
    if not isinstance(result.code, str) or _REASON_CODE.fullmatch(result.code) is None:
        return ProbeResult("BLOCKED", "INVALID_REASON_CODE", {})
    if not isinstance(result.safe_detail, Mapping):
        return ProbeResult("BLOCKED", "UNSAFE_DETAIL", {})
    detail = dict(result.safe_detail)
    if not set(detail) <= _DETAIL_KEYS or _contains_secret(detail):
        return ProbeResult("BLOCKED", "UNSAFE_DETAIL", {})
    try:
        size = len(jobs_receipts.canonical_json_bytes(detail))
    except jobs_receipts.CanonicalJSONError:
        return ProbeResult("BLOCKED", "UNSAFE_DETAIL", {})
    if size > _DETAIL_LIMIT:
        return ProbeResult("BLOCKED", "SAFE_DETAIL_TOO_LARGE", {})
    return ProbeResult(result.status, result.code, detail)


def _check_result(
    name: str, result: ProbeResult, observed_at: str
) -> CheckResult:
    body = {
        "name": name,
        "status": result.status,
        "code": result.code,
        "observed_at": observed_at,
        "safe_detail": dict(result.safe_detail),
    }
    return CheckResult(
        name=name,
        status=result.status,
        code=result.code,
        observed_at=observed_at,
        evidence_digest=jobs_receipts.digest_bytes(
            jobs_receipts.canonical_json_bytes(body)
        ),
        safe_detail=dict(result.safe_detail),
    )


def _memory_scope_is_broad(paths: tuple[Path, ...]) -> bool:
    for path in paths:
        try:
            if Path(path).resolve().is_dir():
                return True
        except OSError:
            continue
    return False


def classify_preflight_failure(checks: tuple[CheckResult, ...]) -> str:
    blocked = tuple(check for check in checks if check.status != "PASS")
    if any(
        check.name == "auth_check"
        or check.code in {"SSH_AUTH_FAILED", "AUTH_REQUIRED", "TOKEN_EXPIRED"}
        for check in blocked
    ):
        return "AUTH_INFRA"
    if any(
        check.code in {"APPROVAL_REQUIRED", "MEMORY_SCOPE_TOO_BROAD"}
        for check in blocked
    ):
        return "SAFETY_GATE"
    if any(
        check.name
        in {"file_paths", "permissions", "worktree", "tool_routing", "resource_budget"}
        for check in blocked
    ):
        return "INFRA_FAILURE"
    return "TASK_FAILURE"


def _stable_id(prefix: str, value: object) -> str:
    digest = jobs_receipts.digest_bytes(
        jobs_receipts.canonical_json_bytes(value)
    ).removeprefix("sha256:")
    return f"{prefix}_{digest[:24]}"


def evaluate_preflight(
    snapshot: PreflightSnapshot, probes: PreflightProbes
) -> PreflightDecision:
    """Evaluate all seven probes exactly once against one immutable snapshot."""

    checks = []
    for name in CHECK_NAMES:
        probe = getattr(probes, name)
        try:
            raw_result = probe(_probe_input(name, snapshot))
        except Exception:
            raw_result = ProbeResult(
                status="BLOCKED",
                code=f"{name.upper()}_PROBE_ERROR",
                safe_detail={},
            )
        result = _bounded_probe_result(name, raw_result)
        if name == "memory_store" and _memory_scope_is_broad(
            snapshot.scoped_memory_paths
        ):
            result = ProbeResult(
                status="BLOCKED",
                code="MEMORY_SCOPE_TOO_BROAD",
                safe_detail={},
            )
        checks.append(_check_result(name, result, snapshot.observed_at))

    ordered_checks = tuple(checks)
    status: Literal["PASS", "BLOCKED"] = (
        "PASS" if all(item.status == "PASS" for item in ordered_checks) else "BLOCKED"
    )
    failure_class = (
        None if status == "PASS" else classify_preflight_failure(ordered_checks)
    )
    identity = {
        "job_id": snapshot.job_id,
        "execution_spec_digest": snapshot.execution_spec_digest,
        "expected_job_revision": snapshot.expected_job_revision,
        "checks": [
            {"name": item.name, "evidence_digest": item.evidence_digest}
            for item in ordered_checks
        ],
    }
    preflight_id = _stable_id("p", identity)
    receipt_id = _stable_id("r", {"preflight_id": preflight_id})
    return PreflightDecision(
        preflight_id=preflight_id,
        status=status,
        failure_class=failure_class,
        checks=ordered_checks,
        receipt_id=receipt_id,
    )


def _epoch_seconds(value: str) -> int:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("preflight observed_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("preflight observed_at must include a timezone")
    return int(parsed.timestamp())


def _check_dict(check: CheckResult) -> dict[str, JsonValue]:
    return {
        "name": check.name,
        "status": check.status,
        "code": check.code,
        "observed_at": check.observed_at,
        "evidence_digest": check.evidence_digest,
        "safe_detail": dict(check.safe_detail),
    }


def preflight_and_record(
    conn,
    snapshot: PreflightSnapshot,
    probes: PreflightProbes,
    signer: ReceiptSigner,
) -> PreflightDecision:
    """Evaluate, sign, and persist exactly one aggregate decision; never claim."""

    decision = evaluate_preflight(snapshot, probes)
    payload: dict[str, JsonValue] = {
        "job_id": snapshot.job_id,
        "attempt_id": None,
        "execution_spec_digest": snapshot.execution_spec_digest,
        "status": decision.status,
        "check_evidence": [
            {"name": check.name, "evidence_digest": check.evidence_digest}
            for check in decision.checks
        ],
        "timestamp": snapshot.observed_at,
        "transitioned_by": "jobs_harness/1",
    }
    envelope = signer.sign(payload, receipt_id=decision.receipt_id)
    record = jobs_db.PreflightRecord(
        id=decision.preflight_id,
        job_id=snapshot.job_id,
        attempt_id=None,
        execution_spec_digest=snapshot.execution_spec_digest,
        expected_job_revision=snapshot.expected_job_revision,
        status=decision.status,
        failure_class=decision.failure_class,
        checks=tuple(_check_dict(check) for check in decision.checks),
        receipt_id=decision.receipt_id,
        idempotency_key=f"preflight:{decision.preflight_id}",
        created_at=_epoch_seconds(snapshot.observed_at),
    )
    jobs_db.record_preflight(conn, record, envelope)
    return decision


__all__ = [
    "CHECK_NAMES",
    "CheckResult",
    "PreflightDecision",
    "PreflightProbes",
    "PreflightSnapshot",
    "ProbeResult",
    "ReceiptSigner",
    "classify_preflight_failure",
    "evaluate_preflight",
    "preflight_and_record",
]
