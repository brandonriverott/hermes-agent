"""``hermes jobs`` CLI — the standalone Jobs control plane.

Jobs Core V1 exposes a tiny, human-readable command surface over the
independent ``$HERMES_HOME/jobs.db`` store (see :mod:`hermes_cli.jobs_db`).
The public lifecycle is exactly ``working`` / ``needs_you`` / ``finished``;
internal execution detail lives as steps, append-only events, attempts, and
receipts on one persistent Job.

This is a footprint-ladder rung-2 capability: a CLI command with zero
model-tool schema cost. It shares no code and no database with Kanban.

V1 commands::

    hermes jobs create <name> --goal-file <path> --lane claude|codex
                              [--routing-reason <text>] [--skills <name>] [--json]
    hermes jobs list [--status working|needs_you|finished] [--json]
    hermes jobs show <id-or-number> [--json]
    hermes jobs transition <id-or-number> --status <status> --step <step>
                              [--reason <text>] [--json]
    hermes jobs heartbeat <id-or-number> [--at <epoch>] [--json]
    hermes jobs events <id-or-number> [--json]
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shlex
import stat
import sys
import time
from pathlib import Path

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_exec as jx
from hermes_cli import jobs_identity as ji
from hermes_cli import jobs_run as jrun


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser(
    parent_subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Attach the ``jobs`` subcommand tree. Returns the top parser."""
    parser = parent_subparsers.add_parser(
        "jobs",
        help="Jobs control plane (independent of Kanban)",
        description=(
            "Manage Jobs — the small work-control plane with a three-status "
            "public lifecycle (working / needs_you / finished). A single Job "
            "owns the original goal verbatim, every attempt, append-only "
            "events, and durable receipts. State is per-profile and shares no "
            "database with Kanban."
        ),
    )
    sub = parser.add_subparsers(dest="jobs_action")

    p_create = sub.add_parser("create", help="Create a new job from a goal file")
    p_create.add_argument("name", help="Plain-English job name")
    p_create.add_argument(
        "--goal-file",
        required=True,
        metavar="PATH",
        help="Path to a UTF-8 file whose contents become the verbatim goal",
    )
    p_create.add_argument(
        "--lane",
        required=True,
        choices=ji.REQUESTED_LANES,
        help="Requested executor lane",
    )
    p_create.add_argument(
        "--routing-reason", default=None, help="Why this specialist was chosen"
    )
    p_create.add_argument(
        "--correlation",
        action="append",
        default=None,
        metavar="KANBAN_ID",
        help="Historical Kanban card id (repeatable)",
    )
    p_create.add_argument(
        "--skills",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Skill to hand the builder as read-only reference material "
            "(repeatable, or comma-separated). Omit and the Job's own words "
            "select the defaults; pass an empty value for explicitly none."
        ),
    )
    p_create.add_argument("--json", action="store_true", help="Emit JSON")

    p_list = sub.add_parser("list", aliases=["ls"], help="List jobs")
    p_list.add_argument(
        "--status", choices=list(jdb.PUBLIC_STATUSES), default=None,
        help="Filter by public status",
    )
    p_list.add_argument("--json", action="store_true", help="Emit JSON")

    p_show = sub.add_parser("show", help="Show a job's details")
    p_show.add_argument("job", help="Job id or number (e.g. j_ab12, 7, 'Job #7')")
    p_show.add_argument("--json", action="store_true", help="Emit JSON")

    p_tr = sub.add_parser("transition", help="Move a job to a new status/step")
    p_tr.add_argument("job", help="Job id or number")
    p_tr.add_argument(
        "--status", required=True, choices=list(jdb.PUBLIC_STATUSES),
        help="Target public status",
    )
    p_tr.add_argument(
        "--step", required=True, choices=list(jdb.JOB_STEPS),
        help="Current internal step",
    )
    p_tr.add_argument("--reason", default=None, help="Human reason for the move")
    p_tr.add_argument("--json", action="store_true", help="Emit JSON")

    p_hb = sub.add_parser("heartbeat", help="Record a liveness heartbeat")
    p_hb.add_argument("job", help="Job id or number")
    p_hb.add_argument(
        "--at", type=int, default=None, metavar="EPOCH",
        help="UTC epoch seconds (defaults to now)",
    )
    p_hb.add_argument("--json", action="store_true", help="Emit JSON")

    p_ev = sub.add_parser("events", help="Show a job's append-only event ledger")
    p_ev.add_argument("job", help="Job id or number")
    p_ev.add_argument("--json", action="store_true", help="Emit JSON")

    # --- V2 execution custody (internal adapter surface) --------------------

    p_claim = sub.add_parser("claim", help="Claim an eligible job (custody)")
    p_claim.add_argument("--worker", required=True, help="Claiming worker id")
    p_claim.add_argument(
        "--specialist", default=None,
        help="Match this specialist; omit for unassigned routing work",
    )
    p_claim.add_argument(
        "--job", default=None, help="Claim a specific job id or number",
    )
    p_claim.add_argument(
        "--lease-seconds", type=int, required=True, metavar="N",
        help="Lease duration in seconds (1..3600)",
    )
    p_claim.add_argument(
        "--token-out", required=True, metavar="PATH",
        help="Write the claim token to this file (0600); it is never printed",
    )
    p_claim.add_argument(
        "--at", type=int, default=None, metavar="EPOCH",
        help="UTC epoch seconds (defaults to now)",
    )
    p_claim.add_argument("--json", action="store_true", help="Emit JSON")

    # allow_abbrev=False so a bare ``--claim-token`` is NOT silently accepted as
    # an abbreviation of ``--claim-token-file`` — tokens only ever come from a
    # file, never a command-line value.
    p_as = sub.add_parser(
        "attempt-start", help="Start an execution attempt", allow_abbrev=False
    )
    p_as.add_argument("job", help="Job id or number")
    p_as.add_argument(
        "--claim-token-file", required=True, metavar="PATH",
        help="File holding the claim token (never passed on the command line)",
    )
    p_as.add_argument("--parent", default=None, help="Parent attempt id")
    p_as.add_argument("--specialist", default=None)
    p_as.add_argument("--repository", default=None)
    p_as.add_argument("--branch", default=None)
    p_as.add_argument("--worktree", default=None)
    p_as.add_argument("--commit", default=None)
    p_as.add_argument("--json", action="store_true", help="Emit JSON")

    p_af = sub.add_parser(
        "attempt-finish", help="Finish an execution attempt", allow_abbrev=False
    )
    p_af.add_argument("attempt", help="Attempt id")
    p_af.add_argument(
        "--claim-token-file", required=True, metavar="PATH",
        help="File holding the claim token (never passed on the command line)",
    )
    p_af.add_argument(
        "--status", required=True, choices=list(jdb.TERMINAL_ATTEMPT_STATUSES),
        help="Terminal attempt status",
    )
    p_af.add_argument(
        "--failure-class", default=None, choices=list(jdb.FAILURE_CLASSES),
        help="Failure classification",
    )
    p_af.add_argument(
        "--terminal-failure", action="store_true",
        help=(
            "Give up on this Job for good: move it to finished/failed. Only "
            "valid with --status failed; never inferred from the goal text."
        ),
    )
    p_af.add_argument("--commit", default=None)
    p_af.add_argument("--branch", default=None)
    p_af.add_argument("--worktree", default=None)
    p_af.add_argument("--repository", default=None)
    p_af.add_argument("--json", action="store_true", help="Emit JSON")

    p_hb2 = sub.add_parser(
        "claim-heartbeat", help="Extend a claim lease", allow_abbrev=False
    )
    p_hb2.add_argument("job", help="Job id or number")
    p_hb2.add_argument(
        "--claim-token-file", required=True, metavar="PATH",
        help="File holding the claim token (never passed on the command line)",
    )
    p_hb2.add_argument(
        "--lease-seconds", type=int, required=True, metavar="N",
        help="New lease duration in seconds (1..3600)",
    )
    p_hb2.add_argument(
        "--at", type=int, default=None, metavar="EPOCH",
        help="UTC epoch seconds (defaults to now)",
    )
    p_hb2.add_argument("--json", action="store_true", help="Emit JSON")

    p_rec = sub.add_parser("recover-expired", help="Recover expired claims")
    p_rec.add_argument(
        "--at", type=int, required=True, metavar="EPOCH",
        help="UTC epoch seconds used as the recovery clock",
    )
    p_rec.add_argument("--json", action="store_true", help="Emit JSON")

    # --- V3 adapter surface -------------------------------------------------
    # Thin public wiring over already-tested jobs_db functions. Without these an
    # adapter cannot store evidence, read its own history, or hand a Job back.

    p_ra = sub.add_parser("receipt-add", help="Store a run receipt on a job")
    p_ra.add_argument("job", help="Job id or number")
    p_ra.add_argument(
        "--data-file", required=True, metavar="PATH",
        help="Path to a UTF-8 JSON object holding the receipt payload",
    )
    p_ra.add_argument("--attempt", default=None, help="Attempt this receipt documents")
    p_ra.add_argument(
        "--idempotency-key", default=None,
        help="Job-scoped key; a repeat returns the existing receipt",
    )
    p_ra.add_argument("--json", action="store_true", help="Emit JSON")

    p_rl = sub.add_parser("receipts", help="List a job's receipts")
    p_rl.add_argument("job", help="Job id or number")
    p_rl.add_argument("--json", action="store_true", help="Emit JSON")

    p_at = sub.add_parser("attempts", help="List a job's attempts in execution order")
    p_at.add_argument("job", help="Job id or number")
    p_at.add_argument("--json", action="store_true", help="Emit JSON")

    p_st = sub.add_parser("status", help="Read a job's projection, including staleness")
    p_st.add_argument("job", help="Job id or number")
    p_st.add_argument(
        "--stale-threshold", type=float, default=None, metavar="N",
        help="Seconds since the last heartbeat after which a working job is stale",
    )
    p_st.add_argument(
        "--at", type=int, default=None, metavar="EPOCH",
        help="UTC epoch seconds used as the staleness clock (defaults to now)",
    )
    p_st.add_argument("--json", action="store_true", help="Emit JSON")

    p_rel = sub.add_parser(
        "release", help="Hand a claimed job back to working", allow_abbrev=False
    )
    p_rel.add_argument("job", help="Job id or number")
    p_rel.add_argument(
        "--claim-token-file", required=True, metavar="PATH",
        help="File holding the claim token (never passed on the command line)",
    )
    p_rel.add_argument(
        "--at", type=int, default=None, metavar="EPOCH",
        help="UTC epoch seconds (defaults to now)",
    )
    p_rel.add_argument("--json", action="store_true", help="Emit JSON")

    p_ro = sub.add_parser(
        "run-once", help="Claim, run, and settle exactly one job attempt"
    )
    p_ro.add_argument(
        "--worker", required=True, metavar="CMD",
        help="Worker executable to launch (shell-style words, no shell)",
    )
    p_ro.add_argument(
        "--workspace-root", required=True, metavar="PATH",
        help="Directory that will hold per-attempt worktrees and handoffs",
    )
    p_ro.add_argument(
        "--execution-file", required=True, metavar="PATH",
        help="JSON file holding the approved {\"execution\": {...}} metadata",
    )
    p_ro.add_argument("--worker-id", required=True, help="Identity of this runner")
    p_ro.add_argument(
        "--specialist", default=None,
        help="Claim only this specialist's jobs; omit for unassigned work",
    )
    p_ro.add_argument("--job", default=None, help="Run a specific job id or number")
    p_ro.add_argument(
        "--request-id", default=None, metavar="ID",
        help="Durable caller identity; repeating it returns the already-settled "
             "result instead of running the work a second time",
    )
    p_ro.add_argument(
        "--wall-clock-seconds", type=int, default=jrun.DEFAULT_WALL_CLOCK_SECONDS,
        metavar="N", help="Hard limit on the worker's run time",
    )
    p_ro.add_argument(
        "--at", type=int, default=None, metavar="EPOCH",
        help="UTC epoch seconds used as the custody clock (defaults to now)",
    )
    p_ro.add_argument("--json", action="store_true", help="Emit JSON")

    p_in = sub.add_parser("intake", help="Idempotent source-keyed job intake")
    p_in.add_argument("name", help="Plain-English job name")
    p_in.add_argument("--source-type", required=True, help="Source type (e.g. cron)")
    p_in.add_argument("--source-key", required=True, help="Source key (unique per type)")
    p_in.add_argument(
        "--goal-file", required=True, metavar="PATH",
        help="Path to a UTF-8 file whose contents become the verbatim goal",
    )
    p_in.add_argument(
        "--lane",
        required=True,
        choices=ji.REQUESTED_LANES,
        help="Requested executor lane",
    )
    p_in.add_argument("--routing-reason", default=None)
    p_in.add_argument(
        "--correlation", action="append", default=None, metavar="KANBAN_ID",
        help="Historical Kanban card id (repeatable)",
    )
    p_in.add_argument("--json", action="store_true", help="Emit JSON")

    parser.set_defaults(_jobs_parser=parser)
    return parser


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def jobs_command(args: argparse.Namespace) -> int:
    """Entry point from ``hermes jobs …`` argparse dispatch."""
    action = getattr(args, "jobs_action", None)
    if not action:
        parser = getattr(args, "_jobs_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print(
                "usage: hermes jobs <action> [options]\n"
                "Run 'hermes jobs --help' for the full list.",
                file=sys.stderr,
            )
        return 0

    handlers = {
        "create": _cmd_create,
        "list": _cmd_list,
        "ls": _cmd_list,
        "show": _cmd_show,
        "transition": _cmd_transition,
        "heartbeat": _cmd_heartbeat,
        "events": _cmd_events,
        "claim": _cmd_claim,
        "attempt-start": _cmd_attempt_start,
        "attempt-finish": _cmd_attempt_finish,
        "claim-heartbeat": _cmd_claim_heartbeat,
        "recover-expired": _cmd_recover_expired,
        "intake": _cmd_intake,
        "receipt-add": _cmd_receipt_add,
        "receipts": _cmd_receipts,
        "attempts": _cmd_attempts,
        "status": _cmd_status,
        "release": _cmd_release,
        "run-once": _cmd_run_once,
    }
    handler = handlers.get(action)
    if handler is None:
        print(f"Unknown jobs action: {action}", file=sys.stderr)
        return 1
    return handler(args)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _emit_json(obj) -> None:
    """Deterministic JSON: sorted keys, Unicode preserved, no env leakage."""
    print(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2))


def _print_job(job: jdb.Job) -> None:
    print(f"{job.label}  [{job.id}]")
    print(f"  name:       {job.name}")
    print(f"  status:     {job.status}")
    print(f"  step:       {job.step}")
    if job.specialist:
        print(f"  specialist: {job.specialist}")
    if job.routing_reason:
        print(f"  routing:    {job.routing_reason}")
    if job.last_heartbeat_at is not None:
        print(f"  heartbeat:  {job.last_heartbeat_at}")
    if job.skills is not None:
        print(f"  skills:     {', '.join(job.skills) or '(none)'}")
    if job.correlations:
        print(f"  kanban:     {', '.join(job.correlations)}")
    print("  goal:")
    for line in job.goal.splitlines() or [""]:
        print(f"    {line}")


def _resolve_or_report(conn, ident):
    job = jdb.get_job(conn, ident)
    if job is None:
        print(f"jobs: no such job: {ident}", file=sys.stderr)
    return job


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _cmd_create(args: argparse.Namespace) -> int:
    path = Path(args.goal_file)
    try:
        # Read bytes and decode explicitly so the goal is preserved verbatim —
        # text-mode reads would apply universal-newline translation.
        goal = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        print(f"jobs: goal file not found: {path}", file=sys.stderr)
        return 2
    except UnicodeDecodeError as exc:
        print(f"jobs: goal file is not valid UTF-8: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"jobs: cannot read goal file: {exc}", file=sys.stderr)
        return 2

    try:
        with jdb.connect_closing() as conn:
            jid = jdb.create_job(
                conn,
                name=args.name,
                goal=goal,
                requested_lane=args.lane,
                routing_reason=args.routing_reason,
                correlations=args.correlation,
                skills=args.skills,
            )
            job = jdb.get_job(conn, jid)
    except ValueError as exc:
        print(f"jobs: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _emit_json(job.to_dict())
    else:
        print(f"Created {job.label} ({job.id})")
        _print_job(job)
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    with jdb.connect_closing() as conn:
        jobs = jdb.list_jobs(conn, status=getattr(args, "status", None))
    if args.json:
        _emit_json([j.to_dict() for j in jobs])
        return 0
    if not jobs:
        print("No jobs yet. Create one with `hermes jobs create <name> --goal-file <path>`.")
        return 0
    for j in jobs:
        print(f"{j.label:<9} {j.status:<10} {j.step:<20} {j.name}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    with jdb.connect_closing() as conn:
        job = _resolve_or_report(conn, args.job)
        if job is None:
            return 1
        if args.json:
            _emit_json(job.to_dict())
        else:
            _print_job(job)
    return 0


def _cmd_transition(args: argparse.Namespace) -> int:
    with jdb.connect_closing() as conn:
        job = _resolve_or_report(conn, args.job)
        if job is None:
            return 1
        try:
            updated = jdb.transition(
                conn, job.id, status=args.status, step=args.step, reason=args.reason
            )
        except jdb.InvalidTransition as exc:
            print(f"jobs: {exc}", file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"jobs: {exc}", file=sys.stderr)
            return 2
        if args.json:
            _emit_json(updated.to_dict())
        else:
            print(f"{updated.label}: {job.status} -> {updated.status} ({updated.step})")
    return 0


def _cmd_heartbeat(args: argparse.Namespace) -> int:
    with jdb.connect_closing() as conn:
        job = _resolve_or_report(conn, args.job)
        if job is None:
            return 1
        updated = jdb.heartbeat(conn, job.id, at=args.at)
        if args.json:
            _emit_json(updated.to_dict())
        else:
            print(f"{updated.label}: heartbeat at {updated.last_heartbeat_at}")
    return 0


def _cmd_events(args: argparse.Namespace) -> int:
    with jdb.connect_closing() as conn:
        job = _resolve_or_report(conn, args.job)
        if job is None:
            return 1
        events = jdb.get_events(conn, job.id)
        if args.json:
            _emit_json(events)
        else:
            for e in events:
                print(f"[{e['created_at']}] {e['kind']}")
    return 0


# ---------------------------------------------------------------------------
# V2 custody token files — the token is a capability, never argv/stdout/stderr
# ---------------------------------------------------------------------------


# A claim token is ~43 characters of urlsafe base64; a bounded read keeps a
# decoy file from being slurped into memory before it is rejected.
_MAX_TOKEN_BYTES = 4096


def _read_token_file(path_str: str) -> str:
    """Read a claim token from a strictly owner-private regular file.

    Tokens are never accepted as command-line values (they would leak into
    process listings and shell history), and the file carrying one has to be as
    protected as the capability inside it. Every check here is a hard failure,
    never a warning: a token another account can read is already compromised,
    and a symlink means the path is not the file the caller believes it is.

    ``O_NOFOLLOW`` refuses the symlink at open time and makes the rest race-free
    — the descriptor being ``fstat``-ed is the descriptor being read, so nothing
    can be swapped underneath between the check and the read. ``O_NONBLOCK``
    keeps a FIFO from hanging the process before it can be rejected.
    """
    path = Path(path_str)
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        raise _CliError(f"claim token file not found: {path}")
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise _CliError(
                f"claim token file is a symlink, refusing to read: {path}"
            )
        raise _CliError(f"cannot read claim token file: {exc}")
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _CliError(f"claim token file is not a regular file: {path}")
        if st.st_uid != os.getuid():
            raise _CliError(
                f"claim token file is not owned by this user: {path}"
            )
        mode = stat.S_IMODE(st.st_mode)
        if mode & 0o077:
            raise _CliError(
                f"claim token file {path} is group/other-accessible "
                f"(mode {oct(mode)}); chmod 600 it"
            )
        raw = os.read(fd, _MAX_TOKEN_BYTES)
    finally:
        os.close(fd)
    token = raw.decode("utf-8", "replace").strip()
    if not token:
        raise _CliError(f"claim token file is empty: {path}")
    return token


def _write_token_file(path_str: str, token: str) -> None:
    """Create ``path`` exclusively at 0600 and write the capability into it.

    ``O_CREAT|O_EXCL`` never follows a symlink and never truncates an existing
    file, so neither a pre-planted symlink nor a decoy can redirect or capture
    the token. A stale token file is the operator's to remove deliberately —
    this will not silently overwrite one. ``fchmod`` pins 0600 on the descriptor
    regardless of the process umask.
    """
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise _CliError(
            f"claim token file already exists, refusing to overwrite: {path}"
        )
    except OSError as exc:
        raise _CliError(f"cannot write claim token file: {exc}")
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)


class _CliError(Exception):
    """Internal: a handled CLI error that maps to exit code 2 + a stderr line."""


# ---------------------------------------------------------------------------
# V2 custody handlers
# ---------------------------------------------------------------------------


def _cmd_claim(args: argparse.Namespace) -> int:
    try:
        with jdb.connect_closing() as conn:
            claim = jdb.claim_job(
                conn,
                worker=args.worker,
                specialist=args.specialist,
                job=args.job,
                lease_seconds=args.lease_seconds,
                now=args.at,
            )
            if claim is None:
                if args.json:
                    _emit_json({"claimed": False, "job": None})
                else:
                    print("Nothing eligible to claim.")
                return 0
            # Persist the capability to the requested file ONLY — never stdout.
            try:
                _write_token_file(args.token_out, claim.claim_token)
            except (OSError, _CliError) as exc:
                # claim_job() has already committed. Undo exactly this claim,
                # using the token still in memory, so a claim nobody can ever
                # hold is not left parked on the Job until its lease expires.
                jdb.release_claim(
                    conn, claim.job.id, claim_token=claim.claim_token, now=args.at
                )
                raise _CliError(
                    f"{exc}; released the claim on {claim.job.label}"
                )
            job_dict = claim.job.to_dict()
        if args.json:
            _emit_json({"claimed": True, "job": job_dict})
        else:
            print(f"Claimed {claim.job.label} (token written to {args.token_out})")
        return 0
    except (ValueError, OSError, _CliError) as exc:
        print(f"jobs: {exc}", file=sys.stderr)
        return 2


def _cmd_attempt_start(args: argparse.Namespace) -> int:
    try:
        token = _read_token_file(args.claim_token_file)
        with jdb.connect_closing() as conn:
            job = _resolve_or_report(conn, args.job)
            if job is None:
                return 1
            aid = jdb.start_attempt(
                conn,
                job.id,
                claim_token=token,
                specialist=args.specialist,
                parent_attempt_id=args.parent,
                repository=args.repository,
                branch=args.branch,
                worktree=args.worktree,
                commit=args.commit,
            )
            attempt = jdb.get_attempt(conn, aid)
        if args.json:
            _emit_json({"attempt": attempt})
        else:
            print(f"Started attempt {aid} (#{attempt['ordinal']}) on {job.label}")
        return 0
    except (jdb.InvalidClaim, jdb.InvalidTransition, ValueError, _CliError) as exc:
        print(f"jobs: {exc}", file=sys.stderr)
        return 2


def _cmd_attempt_finish(args: argparse.Namespace) -> int:
    try:
        token = _read_token_file(args.claim_token_file)
        with jdb.connect_closing() as conn:
            if jdb.get_attempt(conn, args.attempt) is None:
                print(f"jobs: no such attempt: {args.attempt}", file=sys.stderr)
                return 1
            attempt = jdb.finish_attempt(
                conn,
                args.attempt,
                status=args.status,
                failure_class=args.failure_class,
                claim_token=token,
                commit=args.commit,
                branch=args.branch,
                worktree=args.worktree,
                repository=args.repository,
                terminal_failure=args.terminal_failure,
            )
        if args.json:
            _emit_json({"attempt": attempt})
        else:
            print(f"Finished attempt {args.attempt}: {attempt['status']}")
        return 0
    except (jdb.InvalidClaim, jdb.InvalidTransition, ValueError, _CliError) as exc:
        print(f"jobs: {exc}", file=sys.stderr)
        return 2


def _cmd_claim_heartbeat(args: argparse.Namespace) -> int:
    try:
        token = _read_token_file(args.claim_token_file)
        with jdb.connect_closing() as conn:
            job = _resolve_or_report(conn, args.job)
            if job is None:
                return 1
            updated = jdb.claim_heartbeat(
                conn,
                job.id,
                claim_token=token,
                lease_seconds=args.lease_seconds,
                now=args.at,
            )
            job_dict = updated.to_dict()
        if args.json:
            _emit_json({"job": job_dict})
        else:
            print(f"{updated.label}: lease extended to {updated.lease_expires_at}")
        return 0
    except (jdb.InvalidClaim, ValueError, _CliError) as exc:
        print(f"jobs: {exc}", file=sys.stderr)
        return 2


def _cmd_recover_expired(args: argparse.Namespace) -> int:
    with jdb.connect_closing() as conn:
        recovered = jdb.recover_expired_claims(conn, now=args.at)
        numbers = [jdb.get_job(conn, jid).number for jid in recovered]
    if args.json:
        _emit_json({"recovered": numbers, "count": len(numbers)})
    else:
        print(f"Recovered {len(numbers)} expired claim(s): {numbers}")
    return 0


def _cmd_intake(args: argparse.Namespace) -> int:
    path = Path(args.goal_file)
    try:
        goal = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        print(f"jobs: goal file not found: {path}", file=sys.stderr)
        return 2
    except UnicodeDecodeError as exc:
        print(f"jobs: goal file is not valid UTF-8: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"jobs: cannot read goal file: {exc}", file=sys.stderr)
        return 2

    try:
        with jdb.connect_closing() as conn:
            result = jdb.create_or_get_job(
                conn,
                source_type=args.source_type,
                source_key=args.source_key,
                name=args.name,
                goal=goal,
                requested_lane=args.lane,
                routing_reason=args.routing_reason,
                correlations=args.correlation,
            )
            job_dict = jdb.get_job(conn, result.job_id).to_dict()
    except ValueError as exc:
        print(f"jobs: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _emit_json(
            {
                "job": job_dict,
                "created": result.created,
                "conflict": result.conflict,
            }
        )
    else:
        state = "created" if result.created else ("conflict" if result.conflict else "existing")
        print(f"Intake {state}: {job_dict['label']} ({job_dict['id']})")
    return 0


# ---------------------------------------------------------------------------
# V3 adapter surface — receipts, attempts, projection, release, run-once
# ---------------------------------------------------------------------------


def _read_json_file(path_str: str, what: str) -> object:
    """Read a UTF-8 **standard**-JSON file, raising :class:`_CliError` on anything else.

    ``NaN`` and the infinities are refused here rather than at the store: they
    parse in Python and then have no JSON spelling on the way out, so a payload
    carrying one would be accepted, persisted, and only fail when some other
    reader tried to make sense of it.
    """
    path = Path(path_str)
    try:
        raw = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise _CliError(f"{what} file not found: {path}")
    except UnicodeDecodeError as exc:
        raise _CliError(f"{what} file is not valid UTF-8: {exc}")
    except OSError as exc:
        raise _CliError(f"cannot read {what} file: {exc}")
    try:
        return jx.loads_strict(raw)
    except ValueError as exc:
        raise _CliError(f"{what} file is not valid JSON: {exc}")


def _cmd_receipt_add(args: argparse.Namespace) -> int:
    try:
        data = _read_json_file(args.data_file, "receipt data")
        # Sanitized at the write seam, not just by well-behaved callers: this is
        # the last point before the payload becomes durable, renderable evidence.
        clean = jx.sanitize_receipt(data)
        with jdb.connect_closing() as conn:
            rid = jdb.add_receipt(
                conn,
                args.job,
                data=clean,
                attempt_id=args.attempt,
                idempotency_key=args.idempotency_key,
            )
            stored = next(
                r for r in jdb.get_receipts(conn, args.job) if r["id"] == rid
            )
    except (jx.UnsafeReceipt, ValueError, _CliError) as exc:
        print(f"jobs: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _emit_json({"receipt": stored})
    else:
        print(f"Stored receipt {rid} on {args.job}")
    return 0


def _cmd_receipts(args: argparse.Namespace) -> int:
    with jdb.connect_closing() as conn:
        job = _resolve_or_report(conn, args.job)
        if job is None:
            return 1
        receipts = jdb.get_receipts(conn, job.id)
    if args.json:
        _emit_json(receipts)
    else:
        for r in receipts:
            print(f"[{r['created_at']}] {r['id']}  attempt={r['attempt_id']}")
    return 0


def _cmd_attempts(args: argparse.Namespace) -> int:
    with jdb.connect_closing() as conn:
        job = _resolve_or_report(conn, args.job)
        if job is None:
            return 1
        attempts = jdb.get_attempts(conn, job.id)
    if args.json:
        _emit_json(attempts)
    else:
        for a in attempts:
            ordinal = "?" if a["ordinal"] is None else a["ordinal"]
            print(f"#{ordinal:<3} {a['id']}  {a['status']:<15} {a['failure_class'] or ''}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    now = int(time.time()) if args.at is None else int(args.at)
    with jdb.connect_closing() as conn:
        job = _resolve_or_report(conn, args.job)
        if job is None:
            return 1
        proj = jdb.projection(
            conn, job.id, now=now, stale_threshold=args.stale_threshold
        )
    if args.json:
        _emit_json(proj)
    else:
        print(f"{proj['label']}  {proj['status']}/{proj['step']}"
              f"{'  STALE' if proj['stale'] else ''}")
    return 0


def _cmd_release(args: argparse.Namespace) -> int:
    try:
        token = _read_token_file(args.claim_token_file)
        with jdb.connect_closing() as conn:
            job = _resolve_or_report(conn, args.job)
            if job is None:
                return 1
            updated = jdb.release_claim(
                conn, job.id, claim_token=token, now=args.at
            )
            job_dict = updated.to_dict()
    except (jdb.InvalidClaim, ValueError, _CliError) as exc:
        print(f"jobs: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _emit_json({"job": job_dict})
    else:
        print(f"{updated.label}: claim released, back to {updated.status}")
    return 0


def _cmd_run_once(args: argparse.Namespace) -> int:
    try:
        if os.environ.get("HERMES_JOBS_DISPATCH") == "1":
            lane_root = os.environ.get("HERMES_JOBS_LANE_ROOT", "").strip()
            if not lane_root or not Path(lane_root).is_absolute():
                raise _CliError(
                    "Jobs dispatch is enabled but HERMES_JOBS_LANE_ROOT is invalid"
                )
            result = jrun.canonical_dispatch_once(
                lane_root=Path(lane_root), worker_id=args.worker_id,
                job=args.job, now=args.at,
            )
        else:
            execution = _read_json_file(args.execution_file, "execution metadata")
            # shlex, never a shell: the worker command is words, not a script.
            worker_command = shlex.split(args.worker)
            if not worker_command:
                raise _CliError("--worker must name an executable")
            result = jrun.run_once(
                worker_command=worker_command,
                workspace_root=args.workspace_root,
                worker_id=args.worker_id,
                execution=execution,
                specialist=args.specialist,
                job=args.job,
                wall_clock_seconds=args.wall_clock_seconds,
                request_id=args.request_id,
                now=args.at,
            )
    except (ValueError, OSError, _CliError) as exc:
        print(f"jobs: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _emit_json(result)
    else:
        if not result["ran"]:
            print(f"Nothing run: {result['reason']}"
                  + (f" ({result['error']})" if result["error"] else ""))
        else:
            print(f"{result['job']['label']} attempt #{result['ordinal']}: "
                  f"{result['status']} -> {result['job_status']}/{result['job_step']}")
    # An empty poll is a normal, successful nothing. Every other non-run is a
    # refusal — an unbuilt lane, metadata nobody approved, a repository that is
    # not there — and a caller looping on exit status must not read one of those
    # as work completed.
    if not result["ran"] and result["reason"] != "nothing_eligible":
        print(f"jobs: {result['reason']}: {result['error'] or ''}", file=sys.stderr)
        return 2
    return 0
