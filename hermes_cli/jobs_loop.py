"""Failure classification, evidence readback, retry, and feedback journal."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from hermes_cli import jobs_receipts
from hermes_constants import get_hermes_home


FAILURE_CLASSES = frozenset({
    "AUTH_INFRA",
    "PROVIDER",
    "SAFETY_GATE",
    "TASK_FAILURE",
    "INFRA_FAILURE",
})
NON_RETRYABLE = frozenset({"AUTH_INFRA", "SAFETY_GATE"})

_AUTH_REASONS = frozenset({
    "AUTH_REQUIRED",
    "CREDENTIALS_EXPIRED",
    "MISSING_LANE_LOGIN",
    "SSH_AUTH_FAILED",
    "TOKEN_EXPIRED",
    "UNAUTHORIZED",
})
_SAFETY_REASONS = frozenset({"APPROVAL_REQUIRED", "IRREVERSIBLE_ACTION", "SAFETY_GATE"})
_PROVIDER_REASONS = frozenset({
    "BILLING_REFUSED",
    "MODEL_UNAVAILABLE",
    "PROVIDER_CAPACITY",
    "PROVIDER_OUTAGE",
    "RATE_LIMITED",
})
_INFRA_REASONS = frozenset({
    "DISK_FULL",
    "FILESYSTEM_BRIDGE_UNAVAILABLE",
    "MEMORY_EXHAUSTED",
    "NETWORK_TRANSPORT_FAILED",
    "PROCESS_CRASHED",
    "PROCESS_TIMEOUT",
    "WORKER_INFRASTRUCTURE_FAILED",
    "REVIEWER_PROCESS_FAILED",
    "CLEANUP_FAILED",
    "RESULT_CLEANUP_FAILED",
    "HANDOFF_INCOMPLETE",
})
_REASON_CODE = re.compile(r"\A[A-Z][A-Z0-9_]{0,63}\Z")
_DIGEST = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_SECRET_MARKERS = (
    "authorization:",
    "bearer ",
    "token=",
    "api_key",
    "private key",
    "sk-",
)


@dataclass(frozen=True)
class FailureSignal:
    http_status: int | None = None
    reason_code: str | None = None
    safety_gate: bool = False
    stage: str | None = None


@dataclass(frozen=True)
class FailureDecision:
    failure_class: str
    reason_code: str
    recovery_decision: str


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_backoff_seconds: int = 30
    max_backoff_seconds: int = 300


@dataclass(frozen=True)
class RetryRecord:
    ordinal: int
    evidence_digest: str
    decision: str


@dataclass(frozen=True)
class RetryDecision:
    action: Literal["RETRY", "BLOCKED", "HUMAN_ACTION"]
    reason_code: str
    backoff_seconds: int


@dataclass(frozen=True)
class ReadbackResult:
    path: str
    size: int | None
    digest: str | None
    status: Literal["PASS", "BLOCKED"]
    reason_code: str


@dataclass(frozen=True)
class FailureJournalRecord:
    job_id: str
    attempt_id: str
    failure_class: str
    reason_code: str
    recovery_decision: str
    evidence_digests: tuple[str, ...]
    observed_at: int
    component_versions: Mapping[str, str]


class UnsafeJournalRecord(ValueError):
    """Raised when a journal record is unbounded or may contain a secret."""


def _valid_reason(reason_code: str | None) -> str | None:
    if isinstance(reason_code, str) and _REASON_CODE.fullmatch(reason_code):
        return reason_code
    return None


def classify_failure(signal: FailureSignal) -> FailureDecision:
    """Map every signal to one class using deterministic precedence."""

    reason = _valid_reason(signal.reason_code)
    if signal.http_status == 401 or reason in _AUTH_REASONS:
        primary_reason = reason if reason in _AUTH_REASONS else "HTTP_401_AUTH_FAILED"
        return FailureDecision("AUTH_INFRA", primary_reason, "HUMAN_ACTION")
    if signal.safety_gate or reason in _SAFETY_REASONS:
        return FailureDecision(
            "SAFETY_GATE",
            reason if reason is not None else "SAFETY_GATE",
            "HUMAN_ACTION",
        )
    if (
        signal.http_status in {402, 429}
        or (signal.http_status is not None and 500 <= signal.http_status <= 599)
        or reason in _PROVIDER_REASONS
    ):
        if reason in _PROVIDER_REASONS:
            primary_reason = reason
        elif signal.http_status is not None:
            primary_reason = f"HTTP_{signal.http_status}_PROVIDER"
        else:
            primary_reason = "PROVIDER_FAILURE"
        return FailureDecision("PROVIDER", primary_reason, "RETRY")
    if reason in _INFRA_REASONS:
        return FailureDecision("INFRA_FAILURE", reason, "RETRY")
    return FailureDecision("TASK_FAILURE", reason or "TASK_FAILED", "CORRECT")


def decide_retry(
    history: Sequence[RetryRecord],
    *,
    evidence_digest: str,
    failure_class: str,
    prior_failure_digest: str | None = None,
    changed_evidence_digest: str | None = None,
    what_changed: str | None = None,
    policy: RetryPolicy = RetryPolicy(),
) -> RetryDecision:
    """Return a deterministic decision without mutating attempt state."""

    if policy.max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if failure_class in NON_RETRYABLE:
        return RetryDecision("HUMAN_ACTION", f"{failure_class}_REQUIRES_HUMAN", 0)
    if failure_class not in {"PROVIDER", "INFRA_FAILURE", "TASK_FAILURE"}:
        return RetryDecision("BLOCKED", "UNSUPPORTED_FAILURE_CLASS", 0)
    # Keep the original API usable for old callers, while requiring the
    # stronger three-part correction proof whenever a dispatcher supplies it.
    if any(
        value is not None
        for value in (prior_failure_digest, changed_evidence_digest, what_changed)
    ) and (
        not prior_failure_digest
        or not changed_evidence_digest
        or not isinstance(what_changed, str)
        or not what_changed.strip()
    ):
        return RetryDecision("BLOCKED", "RETRY_REQUIRES_CHANGED_EVIDENCE", 0)
    if (
        changed_evidence_digest is not None
        and not _DIGEST.fullmatch(changed_evidence_digest)
    ) or (
        prior_failure_digest is not None
        and not _DIGEST.fullmatch(prior_failure_digest)
    ):
        return RetryDecision("BLOCKED", "RETRY_REQUIRES_CHANGED_EVIDENCE", 0)
    if any(
        item.evidence_digest in {evidence_digest, changed_evidence_digest}
        for item in history
    ):
        return RetryDecision("BLOCKED", "RETRY_REJECTED_NO_NEW_EVIDENCE", 0)
    if len(history) >= policy.max_attempts:
        return RetryDecision("BLOCKED", "RETRY_LIMIT", 0)
    attempt_count = len(history) + 1
    if failure_class == "TASK_FAILURE":
        backoff = 0
    else:
        backoff = min(
            policy.base_backoff_seconds * (2 ** (attempt_count - 1)),
            policy.max_backoff_seconds,
        )
    return RetryDecision("RETRY", "NEW_EVIDENCE", backoff)


def digest_bytes(data: bytes) -> str:
    return jobs_receipts.digest_bytes(data)


def verify_readback(
    path: Path,
    *,
    expected_digest: str,
    expected_size: int | None = None,
) -> ReadbackResult:
    """Read the claimed bytes and compare their digest and optional size."""

    resolved = str(Path(path))
    if _DIGEST.fullmatch(expected_digest) is None:
        return ReadbackResult(
            resolved, None, None, "BLOCKED", "INVALID_EXPECTED_DIGEST"
        )
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return ReadbackResult(resolved, None, None, "BLOCKED", "READBACK_MISSING")
    except OSError:
        return ReadbackResult(resolved, None, None, "BLOCKED", "READBACK_FAILED")
    observed_digest = digest_bytes(data)
    size = len(data)
    if expected_size is not None and size != expected_size:
        return ReadbackResult(
            resolved,
            size,
            observed_digest,
            "BLOCKED",
            "READBACK_SIZE_MISMATCH",
        )
    if observed_digest != expected_digest:
        return ReadbackResult(
            resolved,
            size,
            observed_digest,
            "BLOCKED",
            "READBACK_DIGEST_MISMATCH",
        )
    return ReadbackResult(resolved, size, observed_digest, "PASS", "OK")


def _contains_secret(value: object) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in _SECRET_MARKERS)
    if isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_secret(item) for item in value.values())
    return False


def _journal_reason(reason_code: str) -> str:
    aliases = {"TOKEN_EXPIRED": "AUTH_EXPIRED"}
    return aliases.get(reason_code, reason_code)


def _journal_body(record: FailureJournalRecord) -> bytes:
    reason_code = _journal_reason(record.reason_code)
    if record.failure_class not in FAILURE_CLASSES:
        raise UnsafeJournalRecord("unknown failure class")
    if _REASON_CODE.fullmatch(reason_code) is None:
        raise UnsafeJournalRecord("invalid reason code")
    if record.recovery_decision not in {
        "BLOCKED",
        "CORRECT",
        "HUMAN_ACTION",
        "RETRY",
    }:
        raise UnsafeJournalRecord("invalid recovery decision")
    if not all(_DIGEST.fullmatch(value) for value in record.evidence_digests):
        raise UnsafeJournalRecord("invalid evidence digest")
    if not all(
        isinstance(key, str)
        and isinstance(value, str)
        and 0 < len(key) <= 64
        and 0 < len(value) <= 128
        for key, value in record.component_versions.items()
    ):
        raise UnsafeJournalRecord("invalid component version")
    body = {
        "schema_version": 1,
        "job_id": record.job_id,
        "attempt_id": record.attempt_id,
        "failure_class": record.failure_class,
        "reason_code": reason_code,
        "recovery_decision": record.recovery_decision,
        "evidence_digests": list(record.evidence_digests),
        "observed_at": record.observed_at,
        "component_versions": dict(record.component_versions),
    }
    if _contains_secret(body):
        raise UnsafeJournalRecord("journal record may contain a secret")
    encoded = jobs_receipts.canonical_json_bytes(body)
    if len(encoded) > 16_384:
        raise UnsafeJournalRecord("journal record is too large")
    return encoded + b"\n"


def append_failure_journal(
    record: FailureJournalRecord, *, path: Path | None = None
) -> str:
    """Append one sanitized canonical record with one durable OS write."""

    destination = (
        get_hermes_home() / "logs" / "failure-journal.jsonl"
        if path is None
        else Path(path)
    )
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    body = _journal_body(record)
    descriptor = os.open(
        destination,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        written = os.write(descriptor, body)
        if written != len(body):
            raise OSError("short failure journal write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest_bytes(body)


__all__ = [
    "FAILURE_CLASSES",
    "NON_RETRYABLE",
    "FailureDecision",
    "FailureJournalRecord",
    "FailureSignal",
    "ReadbackResult",
    "RetryDecision",
    "RetryPolicy",
    "RetryRecord",
    "UnsafeJournalRecord",
    "append_failure_journal",
    "classify_failure",
    "decide_retry",
    "digest_bytes",
    "verify_readback",
]
