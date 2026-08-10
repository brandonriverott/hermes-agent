"""Pure execution contract for Jobs V3 — routing, metadata, evidence hygiene.

Everything in this module is a total function of its arguments: no database, no
clock, no filesystem, no subprocess. That is deliberate. The decisions made here
(which specialist runs a Job, whether the operator-approved execution parameters
are well-formed, what a worker's exit actually means, what may be written into a
durable receipt) are the ones that most need to be provable in isolation, and a
function that cannot touch the world cannot half-apply a decision.

Four contracts live here:

- :func:`route` — ``specialist -> (specialist, model, effort, reason)``. Claude
  Opus 5 is the only *executable* target in this slice; the managed Codex
  decision is represented and tested but marked non-executable until its adapter
  exists, so an unbuilt lane fails closed instead of silently running on Claude.
- :func:`validate_execution` — the ``metadata.execution`` sub-contract. The
  accepted :class:`~hermes_cli.jobs_contract.JobEnvelope` schema is not widened;
  model, effort, turn budget, repository, exact base commit, and workspace kind
  ride inside ``metadata``, which the envelope already freezes and already
  screens for secret-shaped keys.
- :func:`classify_worker_result` — a worker's self-report reconciled against the
  process's own exit. Every disagreement resolves to the *worse* reading, so an
  adapter can never report a success the operating system contradicts.
- :func:`sanitize_receipt` — receipts are durable and get rendered by future
  UIs, so a secret-shaped key is rejected outright and secret-shaped text is
  redacted, recursively, before anything reaches the store. Its strict sibling
  :func:`screen_settlement` refuses instead of redacting, for the two inputs a
  settlement turns into permanent authority.
"""

from __future__ import annotations

import json
import math
import re
import secrets
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_identity as ji

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

# ``max`` exists on ``claude --effort`` only; the managed gateway tops out at
# ``xhigh`` (see the V3 recon). Both live in one tuple because this is the set a
# validated ``metadata.execution`` may name, not a per-provider capability list.
EFFORT_TIERS = ("low", "medium", "high", "xhigh", "max")

WORKSPACE_KINDS = ("worktree",)

# A turn budget is a cost ceiling as much as a control: refuse an absurd one.
MAX_TURNS_CEILING = 500


class UnsupportedRouting(ValueError):
    """The Job cannot be executed by any adapter that exists in this slice."""


class UnsafeReceipt(ValueError):
    """Receipt material that must never reach durable evidence."""


@dataclass(frozen=True)
class RoutingDecision:
    """Which specialist runs this Job, on what model, at what effort, and why.

    ``executable`` is the fail-closed flag: a decision may be perfectly valid
    and still have no adapter behind it. The orchestrator refuses to run those
    rather than substituting a lane the operator did not choose.
    """

    specialist: str
    model: str
    effort: str
    reason: str
    executable: bool


def route(*, specialist: Optional[str]) -> RoutingDecision:
    """Decide the execution lane for a Job. Pure, total, and deterministic.

    The specialist must be an explicit current canonical name. Historical
    aliases normalize only while reading persisted identity and are never an
    active route; neither an absent nor malformed specialist lands on Claude.

    ponytail: the Job's step is deliberately not an input. Nothing in this slice
    routes differently for ``correcting`` vs ``building``; the reviewer lane that
    would is a later slice. A parameter that cannot change the answer is a
    parameter that lies about how the decision is made.
    """
    if not isinstance(specialist, str) or not specialist.strip():
        raise UnsupportedRouting(
            "specialist must explicitly name claude-builder or codex-builder"
        )
    name = specialist.strip()
    if name == ji.CLAUDE_SPECIALIST:
        return RoutingDecision(
            specialist=ji.CLAUDE_SPECIALIST,
            model=ji.CLAUDE_MODEL,
            effort="max",
            reason="job assigned to the Claude builder",
            executable=True,
        )
    if name == ji.CODEX_SPECIALIST:
        return RoutingDecision(
            specialist=ji.CODEX_SPECIALIST,
            model=ji.CODEX_MODEL,
            effort="high",
            reason="job assigned to the Codex builder; its adapter is not built in this slice",
            executable=False,
        )
    raise UnsupportedRouting("no execution lane for the supplied specialist")


# ---------------------------------------------------------------------------
# metadata.execution
# ---------------------------------------------------------------------------

_EXECUTION_STRINGS = ("model", "effort", "repo_path", "base_commit", "workspace_kind")
_EXECUTION_KEYS = frozenset(_EXECUTION_STRINGS) | {"max_turns"}

_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class ExecutionSpec:
    """The validated, operator-approved execution parameters for one attempt."""

    model: str
    effort: str
    max_turns: int
    repo_path: str
    base_commit: str
    workspace_kind: str


def validate_execution(metadata: Optional[Mapping[str, Any]]) -> ExecutionSpec:
    """Validate ``metadata['execution']`` into an :class:`ExecutionSpec`.

    Fails closed on anything it cannot vouch for. Two rules are stricter than
    they look and are the point of the function:

    - ``base_commit`` must be a full 40-character lowercase SHA. ``HEAD``, a tag,
      or an abbreviation names *whatever the repository happens to be right now*,
      which is not an identity a receipt can be checked against later.
    - An unrecognised key is rejected rather than ignored. A caller passing
      ``sandbox_profile`` believes it is being honoured; silently dropping it
      would run the attempt under conditions nobody approved.
    """
    if not isinstance(metadata, Mapping):
        raise ValueError("execution metadata must be a JSON object")
    execution = metadata.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("metadata.execution must be a JSON object")

    unknown = sorted(set(execution) - _EXECUTION_KEYS)
    if unknown:
        raise ValueError("metadata.execution contains unsupported keys")

    values = {}
    for name in _EXECUTION_STRINGS:
        value = execution.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata.execution.{name} must be a non-empty string")
        values[name] = value.strip()

    max_turns = execution.get("max_turns")
    if isinstance(max_turns, bool) or not isinstance(max_turns, int):
        raise ValueError("metadata.execution.max_turns must be an integer")
    if max_turns <= 0 or max_turns > MAX_TURNS_CEILING:
        raise ValueError(
            f"metadata.execution.max_turns must be in 1..{MAX_TURNS_CEILING}"
        )

    if values["effort"] not in EFFORT_TIERS:
        raise ValueError(
            f"metadata.execution.effort must be one of {', '.join(EFFORT_TIERS)}"
        )
    if values["workspace_kind"] not in WORKSPACE_KINDS:
        raise ValueError(
            "metadata.execution.workspace_kind must be one of "
            f"{', '.join(WORKSPACE_KINDS)}"
        )
    if not _SHA_RE.match(values["base_commit"]):
        raise ValueError(
            "metadata.execution.base_commit must be a full 40-character "
            "lowercase commit sha"
        )
    if not values["repo_path"].startswith("/"):
        raise ValueError("metadata.execution.repo_path must be an absolute path")

    return ExecutionSpec(max_turns=max_turns, **values)


def require_agreement(decision: RoutingDecision, spec: ExecutionSpec) -> None:
    """Refuse to execute unless the route and the approved metadata agree.

    Two independent refusals. The routed lane must actually have an adapter, and
    the approved model must be the model that lane runs — a Job routed to Claude
    carrying ``gpt-5.6-sol`` in its approved metadata is a contradiction, and
    running *either* of the two answers would be running something nobody
    approved. Effort is deliberately not compared: the tier is the operator's
    call within the lane, and :func:`validate_execution` has already bounded it.
    """
    if not decision.executable:
        raise UnsupportedRouting(
            f"specialist {decision.specialist!r} has no execution adapter in "
            "this slice; refusing to run it on another lane"
        )
    if spec.model != decision.model:
        raise UnsupportedRouting(
            f"approved model {spec.model!r} does not match the routed lane "
            f"{decision.specialist!r} (model {decision.model!r})"
        )


# ---------------------------------------------------------------------------
# Worker outcome classification
# ---------------------------------------------------------------------------

_WORKER_OUTCOMES = ("succeeded", "failed")


def classify_worker_result(
    *,
    exit_code: Optional[int],
    result: Any,
    timed_out: bool,
) -> Tuple[str, Optional[str]]:
    """Reconcile a worker's self-report with its exit into a terminal status.

    Returns ``(attempt_status, failure_class)`` — always a member of
    :data:`~hermes_cli.jobs_db.TERMINAL_ATTEMPT_STATUSES` and either ``None`` or
    a member of :data:`~hermes_cli.jobs_db.FAILURE_CLASSES`.

    The rule for a disagreement is one line: **the worse reading wins.** A
    worker claiming success under a non-zero exit is reporting something the
    kernel contradicts, and a worker claiming failure under a zero exit is
    reporting something only it can see. Both resolve to a failure. The opposite
    convention — trusting the self-report — is how a crashed run gets recorded as
    a green build.

    A missing, unparseable, or non-conforming result is infrastructure: the
    adapter genuinely does not know what happened, and "don't know" is never
    allowed to read as "fine".
    """
    if timed_out:
        return ("interrupted", "infrastructure")
    if not isinstance(result, Mapping):
        return ("failed", "infrastructure")
    outcome = result.get("outcome")
    if not isinstance(outcome, str) or outcome not in _WORKER_OUTCOMES:
        return ("failed", "infrastructure")
    if outcome == "succeeded":
        if exit_code != 0:
            return ("failed", "infrastructure")
        return ("succeeded", None)
    failure_class = result.get("failure_class")
    if not isinstance(failure_class, str) or failure_class not in jdb.FAILURE_CLASSES:
        failure_class = "implementation"
    return ("failed", failure_class)


# ---------------------------------------------------------------------------
# Receipt hygiene
# ---------------------------------------------------------------------------

REDACTED = "[redacted]"
TRUNCATED = "…[truncated]"

# A receipt is evidence, not a log archive. Both ceilings exist so a builder
# cannot turn durable storage into a place to dump a transcript.
MAX_TEXT_CHARS = 2000
MAX_RECEIPT_BYTES = 64 * 1024

# Same policy as the envelope's metadata screen: a key *containing* any of these
# is a secret by name, at any depth.
_SECRET_KEY_MARKERS = ("password", "token", "secret", "api_key", "api-key",
                       "authorization", "credential")

# Secret-shaped *values*. Deliberately a short, high-confidence list: a broad
# regex that redacts ordinary text makes evidence useless, which is its own kind
# of failure. Order matters — the assignment form runs first so the whole
# ``KEY=value`` pair goes, not just the value's provider prefix.
_SECRET_TEXT_PATTERNS = (
    re.compile(r"(?i)\b[\w.-]*(?:api[_-]?key|secret|password|token|credential)"
               r"[\w.-]*\s*[=:]\s*\S+"),
    re.compile(r"(?i)\bauthorization\b\s*[:=]\s*\S+(?:\s+\S+)?"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}=*"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def reject_secret_key(key: str) -> None:
    """Raise if ``key`` is named like a secret. Public: the adapter screens
    worker environment names with the same policy receipts are screened with."""
    low = key.lower()
    for marker in _SECRET_KEY_MARKERS:
        if marker in low:
            raise UnsafeReceipt(
                f"receipt key {key!r} matches secret-policy marker {marker!r}; "
                "secrets must not be written to durable evidence"
            )


def redact_secrets(text: str) -> str:
    """Replace every secret-shaped run in ``text``. Length is left alone.

    Split out from :func:`redact_text` because the two ceilings serve different
    things: a receipt field is capped at :data:`MAX_TEXT_CHARS` because it is
    durable database evidence, while a preserved worker log is capped in bytes
    on disk. Both need the same scrubbing, and only one of them wants 2 KB.
    """
    for pattern in _SECRET_TEXT_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact_text(text: str) -> str:
    """Replace secret-shaped runs in ``text`` and cap its length."""
    text = redact_secrets(text)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + TRUNCATED
    return text


def redact_json(value: Any) -> Any:
    """A best-effort recursive scrub of untrusted JSON. Never raises.

    :func:`sanitize_receipt` refuses a secret-named key outright, because a
    caller that writes one meant to write a secret. This is for a file some
    worker produced: refusing there would only mean throwing away the evidence,
    so a secret-named key keeps its name and loses its value.
    """
    if isinstance(value, Mapping):
        out = {}
        for key, sub in value.items():
            name = key if isinstance(key, str) else str(key)
            try:
                reject_secret_key(name)
            except UnsafeReceipt:
                out[name] = REDACTED
                continue
            out[name] = redact_json(sub)
        return out
    if isinstance(value, (list, tuple)):
        return [redact_json(v) for v in value]
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (int, float)):
        return value
    return redact_secrets(str(value))


def _carries_capability(text: str, capabilities: Sequence[str]) -> bool:
    """True if ``text`` is, or contains, one of the live custody capabilities.

    :func:`secrets.compare_digest` covers the equality case because that is the
    comparison an attacker gets to time. The containment scan after it is not
    constant-time and does not need to be: it only ever runs on material that is
    about to be refused anyway, and the alternative — letting a capability ride
    into durable evidence inside a longer string — is the exact failure this
    exists to prevent. Nothing here echoes ``text`` or the capability.
    """
    for capability in capabilities:
        if text.isascii() and secrets.compare_digest(text, capability):
            return True
        if capability in text:
            return True
    return False


def _reject_capability(text: str, capabilities: Sequence[str], where: str) -> None:
    if capabilities and _carries_capability(text, capabilities):
        raise UnsafeReceipt(
            f"settlement {where} carries the live custody capability; a "
            "capability is never stored, replayed, or printed"
        )


def _reject_constant(name: str):
    raise ValueError(f"{name} is not a standard JSON value")


def loads_strict(text: str):
    """``json.loads`` that refuses ``NaN``/``Infinity``/``-Infinity``.

    Python's decoder accepts those three JavaScript spellings by default, so a
    payload carrying one parses here and then cannot be re-emitted as standard
    JSON — any consumer that is not Python chokes on it. Refusing at the read
    seam keeps a non-JSON number from ever reaching a decision or a store.
    """
    return json.loads(text, parse_constant=_reject_constant)


def _clean(value: Any, *, strict: bool = False, capabilities: Sequence[str] = ()) -> Any:
    if isinstance(value, Mapping):
        out = {}
        for key, sub in value.items():
            if not isinstance(key, str):
                raise UnsafeReceipt(
                    f"receipt keys must be strings, got {type(key).__name__}"
                )
            reject_secret_key(key)
            if strict:
                _reject_capability(key, capabilities, "key")
            out[key] = _clean(sub, strict=strict, capabilities=capabilities)
        return out
    if isinstance(value, (list, tuple)):
        return [_clean(v, strict=strict, capabilities=capabilities) for v in value]
    if isinstance(value, str):
        if strict:
            _reject_capability(value, capabilities, "value")
            if redact_secrets(value) != value:
                # Redacting here would settle the attempt, move the Job, and
                # write the ledger on the strength of material the caller was
                # never allowed to hand a settlement. Refusing leaves all of it
                # untouched and repairable.
                raise UnsafeReceipt(
                    "settlement input carries secret-shaped material; an "
                    "authoritative input is refused, never redacted"
                )
        return redact_text(value)
    if isinstance(value, float) and not math.isfinite(value):
        # NaN and the infinities are Python floats with no JSON spelling. Storing
        # one makes the receipt unreadable to every non-Python consumer, so it is
        # refused here rather than discovered at render time.
        raise UnsafeReceipt(f"receipt value is not standard JSON: {value!r}")
    # bool rides through the scalar branch as an int subclass, which is correct.
    if value is None or isinstance(value, (int, float)):
        return value
    raise UnsafeReceipt(f"receipt value is not JSON-safe: {type(value).__name__}")


def sanitize_receipt(data: Any) -> dict:
    """Return a deep, redacted, size-bounded copy of ``data``, or refuse.

    Rejection and redaction split along a line worth stating: a *key* named like
    a secret means the caller intended to store one, so the write is refused
    outright. A *value* that merely looks like a secret is usually an accident in
    a log tail, so it is redacted and the surrounding evidence survives. The
    input is never mutated — a caller that also writes its own copy of the
    payload somewhere must not silently get the redacted one.

    This is the *ordinary* screen: worker logs, result files, and receipts an
    adapter assembles from untrusted output. The two authoritative settlement
    inputs get :func:`screen_settlement` instead.
    """
    return _screened(data, strict=False)


def screen_settlement(data: Any, *, capabilities: Sequence[str] = ()) -> dict:
    """The strict screen for a settlement's authoritative inputs, or refuse.

    Same recursion, same bounds, same standard-JSON rules — one different
    verdict, and it is the whole point. A settlement's response envelope and
    final receipt are what the store will hand back as *what happened*, forever.
    Quietly redacting hostile material there would still settle the attempt, move
    the Job, write the ledger, and pin an answer built from input that was never
    allowed in. So here secret-shaped material is refused rather than scrubbed,
    and anything carrying a live custody capability is refused outright.

    Redaction stays correct one layer out, where a preserved worker log or a
    result file is evidence about a run rather than the run's verdict — throwing
    those away would destroy the only record of what a worker did.
    """
    return _screened(data, strict=True, capabilities=tuple(capabilities))


def _screened(data: Any, *, strict: bool, capabilities: Sequence[str] = ()) -> dict:
    if not isinstance(data, Mapping):
        raise UnsafeReceipt("receipt data must be a JSON object (dict)")
    cleaned = _clean(data, strict=strict, capabilities=capabilities)
    encoded = json.dumps(cleaned, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise UnsafeReceipt(
            f"receipt exceeds {MAX_RECEIPT_BYTES} bytes; store a summary, not a log"
        )
    return cleaned


# ---------------------------------------------------------------------------
# Lease and heartbeat cadence
# ---------------------------------------------------------------------------

# Floor and cap together bound the ledger: a lease can never buy more than
# MAX_BEATS_PER_LEASE claim_heartbeat rows, and a tight polling loop can never
# beat more often than once every MIN_HEARTBEAT_SECONDS.
MIN_HEARTBEAT_SECONDS = 15
MAX_BEATS_PER_LEASE = 4

# Wall clock plus margin: the lease has to outlive the worker, or custody
# recovery starts fighting a process that is still running.
LEASE_MARGIN_SECONDS = 300


def heartbeat_interval(lease_seconds: int) -> int:
    """Seconds between claim heartbeats for a lease of ``lease_seconds``.

    A quarter of the lease, floored, so custody is refreshed twice over before
    it could expire while never producing more than four events per lease.
    """
    return max(MIN_HEARTBEAT_SECONDS, math.ceil(int(lease_seconds) / MAX_BEATS_PER_LEASE))


def lease_for(*, wall_clock_seconds: int) -> int:
    """A lease sized to the worker's wall clock plus margin, clamped to core's cap.

    Sizing the lease to the run is what keeps a dead adapter from parking a Job
    for the full :data:`~hermes_cli.jobs_db.MAX_LEASE_SECONDS` after it died.
    """
    if isinstance(wall_clock_seconds, bool) or not isinstance(wall_clock_seconds, int):
        raise ValueError("wall_clock_seconds must be a positive integer")
    if wall_clock_seconds <= 0:
        raise ValueError("wall_clock_seconds must be a positive integer")
    return min(jdb.MAX_LEASE_SECONDS, wall_clock_seconds + LEASE_MARGIN_SECONDS)
