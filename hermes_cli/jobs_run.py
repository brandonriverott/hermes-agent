"""``hermes jobs run-once`` — claim one Job, run one attempt, settle it.

This is the orchestration seam, and the only place in V3 that both holds a
custody capability and writes to the Jobs store. One invocation does exactly one
of everything: recover whatever died, claim one eligible Job, start one attempt,
call one adapter, finish that attempt, and write one final receipt. Then it
returns. No daemon, no fleet, no queue — a caller that wants more work calls
again.

The order is the design:

1. **Recover first.** Expired custody is cleared before anything is claimed.
   Core deliberately never steals an expired-but-uncleared claim, so without a
   recovery caller one dead worker parks a Job forever. Doing it here means the
   recovery caller ships with the runner instead of being somebody's cron job to
   remember.
2. **Validate before custody.** Metadata and the selected candidate's complete
   persisted identity are checked inside atomic claim selection, before its
   custody update or event. Repository/base and skill preflight then run while
   custody is held but before an attempt starts; refusal releases the claim
   without burning an attempt.
3. **Once an attempt starts, it settles.** Everything before ``start_attempt``
   can hand custody back without creating an attempt; everything after it is a
   result, including a workspace that could not be built and a receipt that
   could not be sanitized. A terminal attempt hidden behind ``attempt_id=null``
   is a lie the caller cannot detect.
4. **The attempt is the verdict, settled with its evidence.**
   ``settle_attempt`` writes the terminal status, the Job outcome, the custody
   clear, and the final receipt in *one* transaction. Two transactions would
   leave a crash window in which an attempt is permanently settled, has no
   evidence, and is no longer running for recovery to repair.

The claim token lives only in this function's frame. It is not written to a file,
not passed to the worker, not logged, and not present in the returned result.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_exec as jx
from hermes_cli import jobs_execution
from hermes_cli import jobs_executors
from hermes_cli import jobs_identity
from hermes_cli import jobs_loop
from hermes_cli.jobs_contract import JobEnvelope

# The Claude lane's wall clock. The lease is sized from this, not the other way
# round, so a dead runner's claim expires shortly after the run could have ended.
DEFAULT_WALL_CLOCK_SECONDS = 1500
MAX_PUBLIC_ERROR_BYTES = 512


@dataclass(frozen=True)
class GateEvidence:
    """Graph-safe evidence derived from the authoritative gate adapter."""

    action_outcome: str
    identity_verified: bool
    reason_code: str
    commit: Optional[str]
    themis_review_digest: Optional[str]
    receipt_verification_digest: Optional[str]


def adapt_gate_settlement(
    result: Mapping[str, Any], *, job_dir, review_attempt: int
) -> GateEvidence:
    """Turn verified gate output into graph evidence digests.

    ``done.json`` is intentionally absent from this interface.  The caller must
    pass the normalized result of ``jobs_evidence_gate.graph_settlement``; only
    a result whose identity was authoritatively re-verified can become a
    successful graph decision.
    """

    reason_code = str(result.get("reason_code") or "GATE_EVIDENCE_INVALID")
    identity_verified = result.get("identity_verified") is True
    action_outcome = str(result.get("action_outcome") or "failed")
    commit = result.get("commit")
    clean_commit = commit if isinstance(commit, str) else None
    if action_outcome != "succeeded" or not identity_verified:
        return GateEvidence(
            action_outcome="failed",
            identity_verified=identity_verified,
            reason_code=reason_code,
            commit=clean_commit,
            themis_review_digest=None,
            receipt_verification_digest=None,
        )

    directory = Path(job_dir)
    try:
        review = (directory / f"review-{int(review_attempt)}.json").read_bytes()
        gate_receipt = (directory / "gate.json").read_bytes()
    except (OSError, ValueError, TypeError):
        return GateEvidence(
            action_outcome="failed",
            identity_verified=False,
            reason_code="GATE_EVIDENCE_READBACK_FAILED",
            commit=clean_commit,
            themis_review_digest=None,
            receipt_verification_digest=None,
        )
    return GateEvidence(
        action_outcome="succeeded",
        identity_verified=True,
        reason_code="OK",
        commit=clean_commit,
        themis_review_digest=jobs_loop.digest_bytes(review),
        receipt_verification_digest=jobs_loop.digest_bytes(gate_receipt),
    )


def run_once(
    *,
    worker_command: Sequence[str],
    workspace_root,
    worker_id: str,
    execution: Optional[Mapping[str, Any]],
    executor_registry: Optional[jobs_executors.ExecutorRegistry] = None,
    specialist: Optional[str] = None,
    job: Optional[str] = None,
    wall_clock_seconds: int = DEFAULT_WALL_CLOCK_SECONDS,
    heartbeat_seconds: Optional[int] = None,
    extra_env: Optional[Mapping[str, str]] = None,
    request_id: Optional[str] = None,
    now: Optional[int] = None,
) -> dict:
    """Run at most one Job attempt. Returns a truthful machine-readable result.

    ``now`` pins the custody clock for deterministic tests; everything derived
    from it advances with real elapsed time, so a long run cannot expire its own
    lease by holding a frozen clock.

    ``request_id`` is the caller's durable name for *this* request, and it names
    it across the whole Jobs database rather than within one Job. It is reserved
    before any Job is claimed, bound to the attempt that runs it, and settled
    with that attempt's response — so a caller that lost the answer (a killed
    pipe, a dropped SSH session, a crashed supervisor) asks again with the same
    id and is handed the identical settled result, while a second runner racing
    for the same id is turned away with ``request_in_progress`` instead of
    running the work twice.
    """
    lease_seconds = jx.lease_for(wall_clock_seconds=wall_clock_seconds)
    base_now = int(time.time()) if now is None else int(now)
    started = time.monotonic()
    registry = (
        jobs_executors.production_registry()
        if executor_registry is None
        else executor_registry
    )
    if not isinstance(registry, jobs_executors.ExecutorRegistry):
        raise TypeError("executor_registry must be an ExecutorRegistry")

    def clock() -> int:
        return base_now + int(time.monotonic() - started)

    with jdb.connect_closing() as conn:
        if request_id is not None:
            replayed = jdb.find_run_response(conn, request_id)
            if replayed is not None:
                # Answered from the request's own row, before recovery and before
                # any claim: a settled replay must observe the world, never touch
                # it. The store validates the response against the Job, attempt,
                # and final receipt it names before handing it over.
                return _public(replayed)

        # Neither of these reads the Job, so both are decided before custody is
        # taken: a refusal that happens after a claim still leaves a claim event
        # and a revision bump behind, and "we refused to run" should leave the
        # store exactly as it found it.
        try:
            spec = jx.validate_execution(execution)
        except ValueError as exc:
            return _refused(
                "invalid_execution_metadata", recovered=[], error=str(exc),
            )
        requested_identity = None
        if specialist is not None:
            # claim_job matches this specialist and no other, so the routing
            # answer is already knowable and a Job need never be picked up only
            # to be put straight back down.
            try:
                requested_identity = jobs_identity.resolve_canonical_specialist(
                    specialist
                )
                registry.require(requested_identity.executor)
                if spec.model != requested_identity.model:
                    raise jobs_executors.UnsupportedExecutor(
                        "approved model contradicts the requested executor"
                    )
                registry.require_legacy(requested_identity.executor)
            except (
                jobs_identity.UnsupportedJobLane,
                jobs_executors.UnsupportedExecutor,
            ) as exc:
                return _refused(
                    "unsupported_routing", recovered=[], error=str(exc),
                )

        def validate_candidate(candidate):
            identity = registry.require_job_identity(candidate)
            if requested_identity is not None and identity != requested_identity:
                raise jobs_executors.UnsupportedExecutor(
                    "candidate Job identity contradicts the selected specialist"
                )
            if spec.model != identity.model:
                raise jobs_executors.UnsupportedExecutor(
                    "approved model contradicts the persisted Job identity"
                )
            registry.require_legacy(identity.executor)

        # Recovery first, ownership second. An abandoned request is bound to an
        # attempt, and it is *this* pass that makes that attempt terminal, so
        # reserving before recovering would look at a corpse still marked
        # running and refuse a request nobody is running — forever.
        recovered = [
            jdb.get_job(conn, jid).number
            for jid in jdb.recover_expired_claims(conn, now=base_now)
        ]

        def release_request() -> None:
            """Free the reservation when this run turns out to start no attempt."""
            if request_id is not None:
                jdb.release_request(conn, request_id, owner=worker_id)

        if request_id is not None:
            # Global, exclusive, and taken before the claim: a second runner is
            # turned away here rather than after it has taken custody of some
            # other Job, so one request id can only ever produce one claim.
            try:
                reservation = jdb.reserve_request(
                    conn, request_id, owner=worker_id,
                    lease_seconds=lease_seconds, now=base_now,
                )
            except jdb.RequestInProgress as exc:
                return _refused(
                    "request_in_progress", recovered=recovered, error=str(exc),
                )
            if reservation.response is not None:
                # Reconciled from an attempt this run's recovery just settled, or
                # settled by a racing runner between the read above and here.
                return _public(reservation.response)

        try:
            claim = jdb.claim_job(
                conn,
                worker=worker_id,
                specialist=specialist,
                job=job,
                lease_seconds=lease_seconds,
                now=base_now,
                candidate_guard=validate_candidate,
            )
        except jobs_executors.UnsupportedExecutor as exc:
            release_request()
            return _refused(
                "unsupported_routing", recovered=recovered, error=str(exc),
            )
        if claim is None:
            release_request()
            return _refused("nothing_eligible", recovered=recovered)

        token = claim.claim_token
        held = claim.job

        def refuse(reason: str, exc: BaseException) -> dict:
            try:
                jdb.release_claim(conn, held.id, claim_token=token, now=clock())
            except jdb.InvalidClaim:
                # The lease already lapsed — recovery owns this claim now, and
                # it will clear it on the next run. Losing custody must not
                # replace the reason we were refusing for; that reason is the
                # only thing that tells the operator what to fix.
                pass
            release_request()
            return _refused(reason, recovered=recovered, job=held, error=str(exc))

        try:
            identity = registry.require_job_identity(held)
            if requested_identity is not None and identity != requested_identity:
                raise jobs_executors.UnsupportedExecutor(
                    "claimed Job identity contradicts the selected specialist"
                )
            if spec.model != identity.model:
                raise jobs_executors.UnsupportedExecutor(
                    "approved model contradicts the persisted Job identity"
                )
            selected_adapter = registry.require_legacy(identity.executor)
            route_reason = f"Job persisted with canonical {identity.executor} executor"
        except jobs_executors.UnsupportedExecutor as exc:
            return refuse("unsupported_routing", exc)

        try:
            selected_adapter.preflight(spec)
        except jobs_execution.AdapterError as exc:
            return refuse("preflight_failed", exc)

        try:
            # Declared skills are an instruction, so a Job that names one nobody
            # can resolve is handed straight back rather than run without it. It
            # lands here, beside the other refusals, so it costs no attempt and
            # no provider spend.
            selected_adapter.preflight_skills(
                held.skills, name=held.name, goal=held.goal
            )
        except jobs_execution.AdapterError as exc:
            return refuse("skill_attachment_failed", exc)

        attempt_id = jdb.start_attempt(
            conn,
            held.id,
            claim_token=token,
            specialist=identity.specialist,
            repository=spec.repo_path,
            # Persisted now, while it is still an input. After the run nothing
            # left behind can reconstruct which commit this was allowed to
            # build on, and every later evidence check hangs off it.
            base_commit=spec.base_commit,
            request_id=request_id,
            request_owner=worker_id,
            now=clock(),
        )
        attempt = jdb.get_attempt(conn, attempt_id)

        envelope = JobEnvelope(
            job_id=held.id,
            number=held.number,
            name=held.name,
            goal=held.goal,
            attempt_id=attempt_id,
            ordinal=attempt["ordinal"],
            claim_token=token,
            specialist=identity.specialist,
            routing_reason=route_reason,
            repository=spec.repo_path,
            skills=held.skills,
            # Exactly what was approved, never the caller's raw input.
            metadata={"execution": asdict(spec)},
        )

        def beat() -> None:
            jdb.claim_heartbeat(
                conn, held.id, claim_token=token,
                lease_seconds=lease_seconds, now=clock(),
            )

        def settle(
            outcome_status, failure_class, receipt, *,
            reason: str = "completed", error: Optional[str] = None, **evidence,
        ) -> dict:
            """Close the attempt, write its final receipt, and pin the answer.

            The parts of the answer that only this frame knows — the routing
            decision, why the run ended the way it did, and the error text if it
            ended badly — are sanitized here and settled *with* the attempt, into
            typed columns, in the same write. Everything else in the record is the
            settled truth itself. The response returned below is then rendered from
            that record, and so is any later replay of the same request: one
            envelope, read twice, so the two cannot disagree.

            ``base_commit`` is passed as an assertion, not a value to store: the
            attempt already carries the approved base, and a settlement that
            disagrees with it fails closed rather than settling a run against a
            base nobody approved.

            Sanitizing the receipt before the call is deliberate. Non-finite,
            malformed, or secret-shaped material is refused or redacted while the
            attempt is still running and the store is still untouched. The
            narrative goes in raw on purpose: the store screens it with the same
            strict policy it screens every other settlement input with, so there
            is one place that decides what may become durable evidence.
            """
            done = jdb.settle_attempt(
                conn,
                attempt_id,
                status=outcome_status,
                failure_class=failure_class,
                claim_token=token,
                receipt=receipt,
                request_id=request_id,
                request_owner=worker_id,
                request_data=jx.sanitize_receipt({
                    "ran": True, "recovered": recovered,
                }),
                base_commit=spec.base_commit,
                route=route_reason,
                reason=reason,
                error=error,
                now=clock(),
                **evidence,
            )
            return _public(done["run_record"])

        def settle_broken(reason: str, exc: BaseException) -> dict:
            """Something broke after the attempt started. That is a real result.

            Handing custody back here would leave a terminal attempt with no
            evidence and a machine-readable answer that never mentions it — the
            Job looks untouched while its history says otherwise. The attempt
            started, so it settles, and the caller gets its real identity.
            """
            return settle(
                "failed",
                "infrastructure",
                jx.sanitize_receipt({
                    "schema_version": 1,
                    "kind": "jobs-attempt-final",
                    "source": reason,
                    "job_id": held.id,
                    "attempt_id": attempt_id,
                    "ordinal": attempt["ordinal"],
                    "specialist": identity.specialist,
                    "repository": spec.repo_path,
                    "base_commit": spec.base_commit,
                    "status": "failed",
                    "failure_class": "infrastructure",
                    "exception": type(exc).__name__,
                    # ``error`` is not written here: the settlement stamps the
                    # authoritative one, and two spellings of the same failure in
                    # one document is exactly the contradiction replay rejects.
                }),
                reason=reason,
                error=str(exc),
                repository=spec.repo_path,
            )

        try:
            outcome = selected_adapter.run_attempt(
                envelope,
                spec,
                worker_command=worker_command,
                workspace_root=workspace_root,
                wall_clock_seconds=wall_clock_seconds,
                heartbeat_seconds=heartbeat_seconds,
                on_heartbeat=beat,
                extra_env=extra_env,
            )
        except Exception as exc:  # noqa: BLE001 - the attempt still has to settle
            return settle_broken("adapter_error", exc)

        try:
            receipt = jx.sanitize_receipt(outcome.receipt)
        except jx.UnsafeReceipt as exc:
            # The run happened; only its evidence is unusable. Settle the attempt
            # with a receipt that says exactly that, rather than pretending the
            # attempt never ran.
            return settle_broken("unsafe_receipt", exc)

        # The attempt is the verdict and the receipt is evidence about it. The
        # store enforces that they agree; this call is where a disagreement means
        # *this function* built one of the two from the wrong place, and it fails
        # before the attempt settles rather than publishing both.
        return settle(
            outcome.status,
            outcome.failure_class,
            receipt,
            commit=outcome.commit,
            branch=outcome.branch,
            worktree=outcome.worktree,
            repository=outcome.repository,
        )


def canonical_dispatch_once(
    *,
    lane_root: Path,
    worker_id: str,
    job: Optional[str] = None,
    now: Optional[int] = None,
) -> dict:
    """Run one Job through the graph/review/watcher-owned dispatcher.

    The legacy ``run_once`` adapter is intentionally builder-only.  When the
    gateway has Jobs dispatch enabled, callers must use this seam so the
    builder result cannot be mistaken for a completed business workflow.
    """

    from gateway.jobs_dispatcher import dispatch_due_job_once

    result = dispatch_due_job_once(
        lane_root=Path(lane_root),
        now=now,
        worker_id=worker_id,
        job_id=job,
    )
    if result is None:
        return _refused("nothing_eligible", recovered=[])

    job_id = result.get("job_id")
    with jdb.connect_closing() as conn:
        persisted = jdb.get_job(conn, job_id) if job_id else None
        attempt = (
            jdb.get_attempt(conn, result.get("attempt_id"))
            if result.get("attempt_id")
            else None
        )
    if not result.get("claimed"):
        return _refused(
            str(result.get("reason") or "dispatch_refused"),
            recovered=[],
            job=persisted,
        )
    if persisted is None or attempt is None:
        return _refused(
            "dispatch_evidence_missing", recovered=[], job=persisted,
        )
    return _public({
        "ran": True,
        "reason": result.get("reason"),
        "recovered": [],
        "job_id": persisted.id,
        "job_number": persisted.number,
        "job_label": persisted.label,
        "attempt_id": attempt["id"],
        "ordinal": attempt.get("ordinal"),
        "status": attempt.get("status"),
        "failure_class": result.get("failure_class") or attempt.get("failure_class"),
        "repository": attempt.get("repository"),
        "base_commit": attempt.get("base_commit"),
        "commit": result.get("commit") or attempt.get("commit"),
        "branch": attempt.get("branch"),
        "worktree": attempt.get("worktree"),
        "receipt_id": attempt.get("receipt_id"),
        "job_status": persisted.status,
        "job_step": persisted.step,
        "routing_reason": persisted.routing_reason,
    })


def _refused(
    reason: str, *, recovered: list, job=None, error: Optional[str] = None
) -> dict:
    """The answer for a run that never started an attempt. Nothing settled."""
    return _public({
        "ran": False,
        "reason": reason,
        "recovered": recovered,
        "error": error,
        "job_id": None if job is None else job.id,
        "job_number": None if job is None else job.number,
        "job_label": None if job is None else job.label,
    })


def _public(record: Mapping[str, Any]) -> dict:
    """The machine-readable shape of one run. Every key is always present.

    The single renderer, and the only place the public shape is written down. A
    settled run's first response and every later replay of that request both come
    through here from the one envelope stored with the settlement, so there is no
    second construction of the answer that could disagree with the first.
    """
    return {
        "ran": record.get("ran", True),
        "reason": record.get("reason"),
        "recovered": list(record.get("recovered") or []),
        "job": None if record.get("job_id") is None else {
            "id": record["job_id"],
            "number": record.get("job_number"),
            "label": record.get("job_label"),
        },
        "attempt_id": record.get("attempt_id"),
        "ordinal": record.get("ordinal"),
        "status": record.get("status"),
        "failure_class": record.get("failure_class"),
        "repository": record.get("repository"),
        "base_commit": record.get("base_commit"),
        "commit": record.get("commit"),
        "branch": record.get("branch"),
        "worktree": record.get("worktree"),
        "receipt_id": record.get("receipt_id"),
        "job_status": record.get("job_status"),
        "job_step": record.get("job_step"),
        "routing_reason": record.get("routing_reason"),
        "error": _public_error(record.get("error")),
    }


def _public_error(value: object) -> Optional[str]:
    """Redact and byte-cap every error crossing the public run boundary."""

    if value is None:
        return None
    cleaned = jx.redact_secrets(str(value))
    encoded = cleaned.encode("utf-8")
    if len(encoded) <= MAX_PUBLIC_ERROR_BYTES:
        return cleaned
    marker = "…[truncated]"
    budget = MAX_PUBLIC_ERROR_BYTES - len(marker.encode("utf-8"))
    prefix = encoded[:budget]
    while prefix:
        try:
            return prefix.decode("utf-8") + marker
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return marker
