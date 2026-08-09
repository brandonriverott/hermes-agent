"""Independent Jobs control-plane store (``$HERMES_HOME/jobs.db``).

Jobs Core V1: a small, standalone persistence kernel that replaces the Kanban
board's role as the work-control plane while sharing **none** of its code or
database. A single **Job** owns the original goal verbatim, a permanent
human-readable number, every execution attempt, append-only events, durable
run receipts, a truthful heartbeat, and optional historical Kanban correlation
identifiers.

The public lifecycle is deliberately tiny — ``working``, ``needs_you``,
``finished`` — while internal execution detail (routing, building, reviewing,
testing, correcting, verifying, waiting, complete, failed) is carried as a
plain-text *step* plus append-only events, never as board columns requiring
manual movement.

Design constraints honored here:

- **Profile-safe path:** the DB resolves under the active Hermes home via
  ``get_hermes_home()`` (default filename ``jobs.db``). Tests pass an explicit
  ``db_path``.
- **No Kanban coupling:** this module imports nothing from ``kanban`` /
  ``kanban_db`` / ``kanban_swarm`` / ``tools.kanban_tools`` and never opens
  ``kanban.db``.
- **Atomic events:** every mutating operation appends a ``job_events`` row in
  the same IMMEDIATE write transaction.
- **Idempotency:** event and receipt writes accept an optional job-scoped
  idempotency key; repeating the same key returns the existing row instead of
  duplicating it.

Authoritative design: ``jobs-control-plane-design-2026-08-01.md``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hermes_cli.sqlite_util import add_column_if_missing, write_txn
from hermes_constants import get_hermes_home

# ---------------------------------------------------------------------------
# Public lifecycle contract
# ---------------------------------------------------------------------------

PUBLIC_STATUSES = ("working", "needs_you", "finished")

VALID_TRANSITIONS: Dict[str, set] = {
    "working": {"working", "needs_you", "finished"},
    "needs_you": {"needs_you", "working", "finished"},
    "finished": {"finished"},
}

JOB_STEPS = (
    "routing",
    "building",
    "reviewing",
    "testing",
    "correcting",
    "verifying",
    "waiting_for_login",
    "waiting_for_decision",
    "complete",
    "failed",
)

# ---------------------------------------------------------------------------
# V2 execution-custody contract
# ---------------------------------------------------------------------------

# Steps an automatic claim may select. Never ``waiting_*``/``complete``/
# ``failed`` — those are not executable work.
EXECUTABLE_STEPS = (
    "routing",
    "building",
    "correcting",
    "reviewing",
    "testing",
    "verifying",
)

# The only statuses an attempt may finish in.
TERMINAL_ATTEMPT_STATUSES = (
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
    "review_rejected",
)

# Approved failure taxonomy (``None`` is also allowed, e.g. a success).
FAILURE_CLASSES = (
    "authentication",
    "provider_billing",
    "infrastructure",
    "max_turns",
    "reviewer_rejection",
    "implementation",
    "irreversible_action",
)

# Reaching either of these public statuses ends the current custody lease, so
# a ``transition`` into them atomically clears the Job's claim block.
_CUSTODY_CLEARING_STATUSES = frozenset({"needs_you", "finished"})

# On recovery, keep the Job on a genuine correction/review step; anything else
# rewinds to routing so the router re-dispatches from a clean point.
_RECOVER_PRESERVE_STEPS = frozenset({"correcting", "reviewing"})

# A lease must be positive and is capped by a conservative constant so a bad
# caller can never park a claim for an unbounded time.
MAX_LEASE_SECONDS = 3600


class InvalidTransition(ValueError):
    """Raised when a lifecycle rule forbids an operation.

    Covers an illegal public status transition and starting a new execution
    attempt on a job that has already ``finished`` (finished is terminal for
    new attempts).
    """


class InvalidClaim(ValueError):
    """Raised when a custody operation presents a wrong/missing/expired token.

    Custody operations fail *closed*: on any token mismatch, absence, or lease
    expiry the operation raises before mutating anything, so a worker that has
    silently lost its claim can never keep writing to the Job.
    """


class RequestInProgress(ValueError):
    """Raised when a caller request id is already owned by a live runner.

    Soft and recoverable: nothing is wrong, somebody else is simply running this
    request right now. The caller is told to wait, not that the store is broken.
    """


class ReplayIntegrity(ValueError):
    """Raised when stored replay state contradicts the world it describes.

    Hard and fail-closed. Every path that could return or extend a stored
    response checks it against the immutable request id and the live Job,
    attempt, and final receipt it names first, so cross-Job, cross-attempt,
    cross-request, dangling, and contradictory data raise *before* anything is
    returned and before any settlement mutates state. Re-running the work on a
    contradiction would be a second execution the caller never asked for, and
    returning it anyway would be a lie, so this raises instead.
    """


class GraphConflict(ValueError):
    """Raised when an idempotency key is replayed with contradictory facts."""


class StaleJobRevision(ValueError):
    """Raised when a reliability write loses its Job revision comparison."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def jobs_db_path() -> Path:
    """The per-profile Jobs DB path (``$HERMES_HOME/jobs.db``).

    Profile-aware: ``get_hermes_home()`` already points at the active profile's
    home. Tests pass an explicit ``db_path`` to :func:`connect`.
    """
    return get_hermes_home() / "jobs.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    number            INTEGER NOT NULL UNIQUE,
    name              TEXT NOT NULL,
    goal              TEXT NOT NULL,
    status            TEXT NOT NULL,
    step              TEXT NOT NULL,
    specialist        TEXT,
    routing_reason    TEXT,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL,
    last_heartbeat_at INTEGER,
    -- V2 execution custody. ``revision`` is a monotonic per-job counter bumped
    -- by every state-changing write in the same transaction (a compare-and-set
    -- token for adapters). The claim block holds at most one active custody
    -- lease; ``claim_token`` is an opaque capability that is NEVER surfaced by
    -- list/show/events or carried on the public Job object.
    revision          INTEGER NOT NULL DEFAULT 0,
    claimed_by        TEXT,
    claim_token       TEXT,
    claim_acquired_at INTEGER,
    lease_expires_at  INTEGER,
    current_attempt_id TEXT,
    -- Declared skill attachment: a JSON array of skill names the builder is
    -- handed as read-only reference material (see :mod:`hermes_cli.jobs_skills`).
    -- NULL is *unset* and means "nobody has an opinion, let the default rules
    -- table choose"; an empty array is an opinion and means "attach nothing".
    -- Names only — whether one resolves to a file on this machine is a runtime
    -- question, and a Job outlives the machine it was created on.
    skills            TEXT
);

CREATE TABLE IF NOT EXISTS job_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,
    data            TEXT,
    idempotency_key TEXT,
    created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_events_job
    ON job_events(job_id, id);

-- Job-scoped idempotency: the same key may recur on a *different* job, but a
-- repeat on the same job returns the existing event. SQLite treats NULLs as
-- distinct, so un-keyed events coexist freely.
CREATE UNIQUE INDEX IF NOT EXISTS idx_job_events_idem
    ON job_events(job_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS job_attempts (
    id                TEXT PRIMARY KEY,
    job_id            TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    parent_attempt_id TEXT REFERENCES job_attempts(id),
    specialist        TEXT,
    status            TEXT NOT NULL,
    failure_class     TEXT,
    repository        TEXT,
    branch            TEXT,
    worktree          TEXT,
    commit_sha        TEXT,
    started_at        INTEGER,
    finished_at       INTEGER,
    created_at        INTEGER NOT NULL,
    -- Permanent per-Job ordinal (1, 2, 3, ...), unique within its Job.
    ordinal           INTEGER,
    -- The exact commit this attempt was approved to start from, written once at
    -- :func:`start_attempt` and never rewritten. It is the only piece of
    -- execution evidence no later observation can reconstruct — a branch, a
    -- worktree, and a head commit are all still readable after the fact, but
    -- "what was this allowed to build on" exists nowhere else. Replay treats a
    -- NULL here as legacy history and refuses to answer from it. Bounded so a
    -- rewritten row cannot hide a novel in an evidence column.
    base_commit       TEXT
        CHECK (base_commit IS NULL OR length(base_commit) BETWEEN 7 AND 64),
    -- The caller's explicit give-up flag, recorded because it selects the Job
    -- outcome and therefore has to be part of terminal replay equivalence.
    -- Every settled attempt written under this schema carries an explicit 0/1;
    -- a running attempt has no terminal outcome yet, and NULL on a settled row
    -- means legacy history nobody can reconstruct (see
    -- :func:`_legacy_terminal_failure`), which makes replay fail closed.
    terminal_failure  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_job_attempts_job
    ON job_attempts(job_id, id);

CREATE TABLE IF NOT EXISTS job_receipts (
    id              TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    attempt_id      TEXT REFERENCES job_attempts(id),
    data            TEXT NOT NULL,
    idempotency_key TEXT,
    created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_receipts_job
    ON job_receipts(job_id, id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_job_receipts_idem
    ON job_receipts(job_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- Caller request identity, and the ONLY place replay authority lives.
--
-- ``request_id`` is the primary key of the whole Jobs database, not of a Job:
-- two Jobs and two concurrent runners cannot independently own or settle one
-- caller request. Events are evidence and carry no authority at all, so no
-- public append can create, replace, or repoint a row here.
--
-- Deliberately free of foreign keys. ``job_id``/``attempt_id``/``receipt_id``
-- are checked against the live rows every time a response is read
-- (:func:`_validated_response_locked`), which is a stronger check than a
-- constraint — it also catches a row that is merely *pointed at the wrong*
-- live job, attempt, or receipt — and it keeps a dangling row readable enough
-- to fail closed with a truthful reason instead of unopenable.
CREATE TABLE IF NOT EXISTS job_run_requests (
    request_id       TEXT PRIMARY KEY,
    -- Who holds it. A live owner is never stolen; see :func:`reserve_request`.
    owner            TEXT NOT NULL,
    -- Only meaningful while unbound: once a request is bound to an attempt,
    -- liveness is the attempt's own claim lease, not a second clock.
    lease_expires_at INTEGER,
    job_id           TEXT,
    attempt_id       TEXT,
    receipt_id       TEXT,
    -- Canonical, screened, bounded standard JSON. NULL until settlement.
    response         TEXT,
    -- SHA-256 of the canonical bytes this settlement wrote, for the response
    -- and for the final receipt it names. Corruption detection, not
    -- authentication: anybody who can rewrite a row can rewrite its digest, and
    -- introducing a key to stop that would put a new durable secret in the
    -- schema. What these catch is the single-row damage no relational check can
    -- see — a torn write, a careless UPDATE, a restored-from-elsewhere row —
    -- because the fields a caller narrates (why a run ended, its error text)
    -- have no second source to be checked against. Everything that *does* have
    -- one is checked against the Job/attempt/receipt themselves, not against
    -- these.
    response_digest  TEXT,
    receipt_digest   TEXT,
    -- The run's narrative outcome, typed and bounded, written once with the
    -- settlement. These three are the only material facts about a run with no
    -- second source: which lane decided it, why it ended, and what the error
    -- was. Keeping them here rather than only inside ``response`` is what makes
    -- them authoritative — the response envelope and the final receipt are two
    -- redundant renderings of these columns, so rewriting any one of the three
    -- rows contradicts the other two.
    route            TEXT CHECK (route IS NULL OR length(route) <= 200),
    reason           TEXT CHECK (reason IS NULL OR length(reason) BETWEEN 1 AND 64),
    error            TEXT CHECK (error IS NULL OR length(error) <= 4096),
    created_at       INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_run_requests_attempt
    ON job_run_requests(attempt_id);

CREATE TABLE IF NOT EXISTS job_correlations (
    job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    kanban_id   TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (job_id, kanban_id)
);

-- Exact return address captured atomically with Job creation.  Only a bounded
-- allowlist is stored; credentials and other ambient session data never enter
-- the Jobs ledger.
CREATE TABLE IF NOT EXISTS job_origins (
    job_id      TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    origin      TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);

-- Idempotent source registry for scheduled jobs and later migration. The
-- ``(source_type, source_key)`` pair is globally unique, so replaying the same
-- scheduled tick or migration row returns the existing Job instead of creating
-- a duplicate. Different source types may reuse the same key.
CREATE TABLE IF NOT EXISTS job_sources (
    source_type TEXT NOT NULL,
    source_key  TEXT NOT NULL,
    job_id      TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (source_type, source_key)
);

CREATE TABLE IF NOT EXISTS job_preflights (
    id                    TEXT PRIMARY KEY,
    job_id                TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    attempt_id            TEXT REFERENCES job_attempts(id),
    execution_spec_digest TEXT NOT NULL,
    expected_job_revision INTEGER NOT NULL,
    status                TEXT NOT NULL CHECK (status IN ('PASS', 'BLOCKED')),
    failure_class         TEXT,
    checks_json           TEXT NOT NULL,
    receipt_id            TEXT NOT NULL REFERENCES job_receipts(id),
    idempotency_key       TEXT NOT NULL,
    created_at            INTEGER NOT NULL,
    UNIQUE (job_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS job_attempt_transitions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id                TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    attempt_id            TEXT NOT NULL REFERENCES job_attempts(id) ON DELETE CASCADE,
    source_state          TEXT,
    target_state          TEXT NOT NULL,
    initiator_type        TEXT NOT NULL,
    initiator_id          TEXT NOT NULL,
    expected_job_revision INTEGER NOT NULL,
    evidence_json         TEXT NOT NULL,
    failure_class         TEXT,
    blocker_code          TEXT,
    receipt_id            TEXT NOT NULL REFERENCES job_receipts(id),
    component             TEXT NOT NULL,
    component_version     TEXT NOT NULL,
    idempotency_key       TEXT NOT NULL,
    created_at            INTEGER NOT NULL,
    UNIQUE (attempt_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_job_attempt_transitions_job
    ON job_attempt_transitions(job_id, attempt_id, id);

CREATE TABLE IF NOT EXISTS job_retry_evidence (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id                TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    chain_id              TEXT NOT NULL,
    attempt_id            TEXT NOT NULL REFERENCES job_attempts(id) ON DELETE CASCADE,
    parent_attempt_id     TEXT REFERENCES job_attempts(id),
    ordinal               INTEGER NOT NULL,
    evidence_digest       TEXT NOT NULL,
    decision              TEXT NOT NULL,
    reason_code           TEXT NOT NULL,
    created_at            INTEGER NOT NULL,
    UNIQUE (chain_id, ordinal),
    UNIQUE (chain_id, evidence_digest)
);

-- Durable monotonic number allocator. A single row (id = 1) holds the highest
-- job number ever issued, independent of the jobs table so deleting a Job row
-- can never free its number for reuse. Seeded from any existing MAX(number) so
-- an early V1 database (created before this allocator existed) migrates safely;
-- INSERT OR IGNORE keeps re-running the schema from ever resetting it.
CREATE TABLE IF NOT EXISTS job_number_seq (
    id    INTEGER PRIMARY KEY CHECK (id = 1),
    last  INTEGER NOT NULL
);

INSERT OR IGNORE INTO job_number_seq (id, last)
    VALUES (1, (SELECT COALESCE(MAX(number), 0) FROM jobs));
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_INITIALIZED_PATHS: set[str] = set()

_BUSY_TIMEOUT_MS = 5000


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the independent Jobs DB.

    Mirrors the projects store: WAL with DELETE fallback for network
    filesystems, foreign keys ON, a configured busy timeout so IMMEDIATE
    writers wait for each other instead of failing instantly, and idempotent
    schema init cached per resolved path per process.
    """
    path = db_path if db_path is not None else jobs_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    conn = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_MS / 1000)
    try:
        conn.row_factory = sqlite3.Row
        from hermes_state import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="jobs.db")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        if resolved not in _INITIALIZED_PATHS:
            conn.executescript(SCHEMA_SQL)
            _migrate(conn)
            _INITIALIZED_PATHS.add(resolved)
    except Exception:
        conn.close()
        raise
    return conn


# The V2 custody columns, added to a pre-V2 ``jobs`` table in place. ``CREATE
# TABLE IF NOT EXISTS`` in ``SCHEMA_SQL`` never alters an existing table, so an
# early V1 database keeps its old column set until this runs.
_JOBS_V2_COLUMNS = (
    ("revision", "revision INTEGER NOT NULL DEFAULT 0"),
    ("claimed_by", "claimed_by TEXT"),
    ("claim_token", "claim_token TEXT"),
    ("claim_acquired_at", "claim_acquired_at INTEGER"),
    ("lease_expires_at", "lease_expires_at INTEGER"),
    ("current_attempt_id", "current_attempt_id TEXT"),
)


# Indexes deliberately kept OUT of ``SCHEMA_SQL``: an early V1 database can hold
# history that violates them (several ``running`` attempts on one Job, NULL
# ordinals), and creating them before the repair below would raise
# ``IntegrityError`` at open time and make that database permanently unopenable.

# The database is the backstop for "at most one running attempt per Job": a
# partial unique index makes a second concurrent ``running`` insert fail rather
# than relying on application-level checks alone. The repair always restores
# that invariant, so this one is created strictly.
_ONE_RUNNING_INDEX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_job_attempts_one_running "
    "ON job_attempts(job_id) WHERE status = 'running'"
)

# Permanent per-Job attempt ordinals, database-enforced. SQLite treats NULLs as
# distinct, so a legacy row whose position stayed unknown cannot block it.
_ORDINAL_INDEX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_job_attempts_ordinal "
    "ON job_attempts(job_id, ordinal)"
)


def _chain_positions(rows: List[sqlite3.Row]) -> Optional[Dict[str, int]]:
    """One Job's attempts as permanent positions ``1..N``, from parents alone.

    ``rows`` are every attempt of a single Job, in any order — the caller must
    not assume the query returned them in any meaningful one. The durable
    record of execution order a row carries is its explicit
    ``parent_attempt_id``, so that graph has to *prove* the order: one linear
    chain covering every attempt, exactly one root, and every other attempt
    continuing exactly one predecessor of the same Job. Stage one of
    :func:`_validated_lineage`, which is the only way callers reach it.

    ``None`` means it proves nothing — a self-parent, a parent that is not an
    attempt of this Job (dangling, or another Job's), a fork, a cycle, two
    roots, a component sitting off the chain. Any of those and the real order is
    unknowable from the durable record; the caller must leave it unknown rather
    than reach for ``created_at`` or the random ``a_`` id, neither of which is
    execution order.
    """
    parents = {r["id"]: r["parent_attempt_id"] for r in rows}
    children: Dict[str, str] = {}
    root: Optional[str] = None
    for aid, parent in parents.items():
        if parent is None:
            if root is not None:
                return None  # two roots: these attempts are not one chain
            root = aid
            continue
        if parent == aid or parent not in parents:
            return None  # a self-parent, or a parent outside this Job
        if parent in children:
            return None  # a fork: two attempts continue one attempt
        children[parent] = aid
    if root is None:
        return None  # every attempt has a parent, so the graph is a cycle
    # Each attempt records one parent and is claimed by at most one child, so
    # walking from the parentless root reaches every node at most once.
    positions: Dict[str, int] = {}
    node: Optional[str] = root
    while node is not None:
        positions[node] = len(positions) + 1
        node = children.get(node)
    if len(positions) != len(parents):
        return None  # a cycle or a disconnected component sits off the chain
    return positions


def _validated_lineage(rows: List[sqlite3.Row]) -> Optional[Dict[str, int]]:
    """One Job's attempts as trustworthy positions ``1..N``, or ``None``.

    The single authority on execution order, for every path — reconstruction,
    successorship, the leaf, the duplicate-running repair, the next permanent
    ordinal. The parent graph is validated *first* (:func:`_chain_positions`),
    and only the order it proves may then be compared against what the rows
    claim. A stored ordinal never gets to vouch for itself: a complete run of
    ``1..N`` says the numbers are all there, not that the rows carrying them
    are one history. Two parentless roots numbered 1 and 2 are two disconnected
    histories that happen to be numbered — as unplaceable as the same two rows
    with no numbers at all.

    A stored ordinal that disagrees with its proven position is a contradiction
    between two durable records, and the rows cannot say which one is wrong, so
    neither is trusted. (That also catches a duplicate stored ordinal: chain
    positions are distinct, so two rows sharing one value cannot both match.)

    ``None`` means the durable record cannot say, and the caller must fail
    closed — no successorship, no leaf, no replay authority, no new position.
    """
    positions = _chain_positions(rows)
    if positions is None:
        return None
    if any(
        r["ordinal"] is not None and r["ordinal"] != positions[r["id"]] for r in rows
    ):
        return None
    return positions


def _reconstruct_ordinals_locked(conn: sqlite3.Connection) -> None:
    """Recover missing permanent ordinals from lineage. Caller holds the txn.

    Per Job, and only for the Jobs actually missing one: take the positions the
    Job's lineage proves (:func:`_validated_lineage`) and write them into the
    NULL ordinals. Nothing else may supply the order — a timestamp and a random
    id are metadata, and letting either decide would make two otherwise
    identical databases migrate to different histories.

    Fails closed, and only ever fills a blank: a Job whose lineage cannot be
    validated leaves every unknown row NULL, and an ordinal already issued is
    permanent identity that is never rewritten to make a chain fit.

    Repeatable: a Job it cannot settle is re-examined to the same answer on
    every later open, writes nothing, and appends no event.
    """
    job_ids = [
        r["job_id"]
        for r in conn.execute(
            "SELECT DISTINCT job_id FROM job_attempts WHERE ordinal IS NULL"
        ).fetchall()
    ]
    for job_id in job_ids:
        rows = conn.execute(
            "SELECT id, parent_attempt_id, ordinal FROM job_attempts WHERE job_id = ?",
            (job_id,),
        ).fetchall()
        positions = _validated_lineage(rows)
        if positions is None:
            continue
        for r in rows:
            if r["ordinal"] is None:
                conn.execute(
                    "UPDATE job_attempts SET ordinal = ? WHERE id = ?",
                    (positions[r["id"]], r["id"]),
                )


def _validated_ordinals(
    conn: sqlite3.Connection, job_id: str
) -> Optional[Dict[str, int]]:
    """This Job's attempts as ``{id: ordinal}``, or ``None`` if order is unknown.

    The stored, database-backed read of :func:`_validated_lineage`: the parent
    graph proves the order, and every stored ordinal has to agree with it.
    Neither ``created_at`` nor the attempt ``id`` is order: two attempts can
    start within one second, and the ``a_`` id is random, so its lexical order
    is noise that happens to look decisive.

    ``None`` means the Job's lineage cannot be trusted and every caller must
    fail closed rather than guess.
    """
    rows = conn.execute(
        "SELECT id, ordinal, parent_attempt_id FROM job_attempts WHERE job_id = ?",
        (job_id,),
    ).fetchall()
    return _validated_lineage(rows)


def _lineage_has_successor(
    conn: sqlite3.Connection, row: sqlite3.Row
) -> Optional[bool]:
    """Was this attempt followed by another, per the Job's durable lineage?

    ``None`` means the record of what came after this attempt is incomplete —
    see :func:`_validated_ordinals` — and the caller must fail closed.
    """
    ordinals = _validated_ordinals(conn, row["job_id"])
    if ordinals is None:
        return None
    return ordinals[row["id"]] < len(ordinals)


def _lineage_leaf(conn: sqlite3.Connection, job_id: str) -> Optional[str]:
    """The attempt the Job's durable lineage says came last, or ``None``.

    ``None`` means the lineage cannot name the current attempt, and nobody may
    then pick one: a wall clock or a random id would be inventing the answer.
    """
    ordinals = _validated_ordinals(conn, job_id)
    if not ordinals:
        return None
    return next(aid for aid, o in ordinals.items() if o == len(ordinals))


def _repair_duplicate_running_locked(conn: sqlite3.Connection, now: int) -> None:
    """Close impossible duplicate ``running`` history. Caller holds the write txn.

    An early V1 database predates the one-running index and may hold several
    ``running`` attempts for one Job (interrupted workers that never closed
    them). Only the Job's durable lineage may say which of them is still live:
    the leaf of the validated chain keeps the slot, and every other running
    attempt is marked ``interrupted``/``infrastructure``.

    When the lineage cannot say — no chain to reconstruct, or one that
    contradicts itself — no attempt is crowned and *every* running attempt is
    closed. Handing one worker continued custody of a Job on the strength of a
    wall clock or a random attempt id is exactly the fabrication this migration
    refuses. Nothing is ever deleted, so no execution evidence is lost, and the
    Job stops pointing at an attempt this pass just closed. Each repair appends
    one attempt-scoped keyed event, so a re-run can neither repair nor record
    anything twice, and the one-running unique index can then be created.
    """
    rows = conn.execute(
        "SELECT id, job_id FROM job_attempts WHERE status = 'running' "
        "AND job_id IN (SELECT job_id FROM job_attempts WHERE status = 'running' "
        "GROUP BY job_id HAVING COUNT(*) > 1)"
    ).fetchall()
    by_job: Dict[str, List[str]] = {}
    for r in rows:
        by_job.setdefault(r["job_id"], []).append(r["id"])
    for job_id in sorted(by_job):
        current = _lineage_leaf(conn, job_id)
        # Sorted only so the repair events land in a stable order. *Which*
        # attempts are closed was decided above, by lineage and nothing else.
        for aid in sorted(by_job[job_id]):
            if aid == current:
                continue
            conn.execute(
                "UPDATE job_attempts SET status = 'interrupted', "
                "failure_class = 'infrastructure', finished_at = ? WHERE id = ?",
                (now, aid),
            )
            conn.execute(
                "UPDATE jobs SET current_attempt_id = NULL "
                "WHERE id = ? AND current_attempt_id = ?",
                (job_id, aid),
            )
            _append_event_locked(
                conn,
                job_id,
                "attempt_finished",
                data={
                    "attempt_id": aid,
                    "status": "interrupted",
                    "failure_class": "infrastructure",
                    "reason": "migration_repair",
                },
                idempotency_key=f"migration:interrupt:{aid}",
                now=now,
            )


def _legacy_terminal_failure(
    conn: sqlite3.Connection, row: sqlite3.Row
) -> Optional[int]:
    """The give-up flag a settled pre-column attempt truly finished with.

    ``None`` means the durable history genuinely cannot say. That row stays
    unknown so a replay fails closed, rather than inheriting a fabricated
    ``False`` that would reject the truthful replay and accept a false one.

    The evidence, in order:

    1. Only a ``failed`` attempt can carry the flag — :func:`finish_attempt`
       rejects it on every other terminal status — so those are a truthful 0.
    2. An attempt followed by a later attempt on the same Job cannot have been
       terminal: giving up moved the Job to ``finished``, which refuses both new
       attempts and any further transition. "Later" is read from the durable
       execution lineage — see :func:`_lineage_has_successor` — never from the
       clock or the random attempt id, and a lineage that cannot say leaves the
       row unknown.
    3. The flag is only *observable* where it changes the outcome
       :func:`_outcome_for` selects. For ``authentication``, ``infrastructure``,
       ``reviewer_rejection`` and the rest, both values produce the identical
       Job outcome, so nothing durable records which one was passed → unknown.
    4. Otherwise the attempt's own ``job_transition`` event decides. It is
       written in the same transaction as the outcome, is append-only, and names
       the attempt in its reason. The present-day Job row is deliberately *not*
       consulted — later transitions may have moved it since. Attributed events
       that disagree make the attribution ambiguous → unknown.
    5. No attributed event at all means the row predates the flag entirely: it
       landed in the same change that made the attempt outcome atomic, so a row
       settled without one was written by code that had no give-up flag to
       record. 0 is that row's history, not a default.
    """
    if row["status"] != "failed":
        return 0
    had_successor = _lineage_has_successor(conn, row)
    if had_successor is None:
        return None
    if had_successor:
        return 0
    failure_class = row["failure_class"]
    terminal = _outcome_for("failed", failure_class, terminal_failure=True)
    nonterminal = _outcome_for("failed", failure_class, terminal_failure=False)
    if terminal == nonterminal:
        return None
    attributed = set()
    for event in conn.execute(
        "SELECT data FROM job_events WHERE job_id = ? AND kind = 'job_transition'",
        (row["job_id"],),
    ).fetchall():
        try:
            data = json.loads(event["data"] or "{}")
        except ValueError:
            continue
        if data.get("reason") != f"attempt {row['id']} failed":
            continue
        attributed.add((data.get("to"), data.get("step")))
    if not attributed:
        return 0
    if attributed == {terminal}:
        return 1
    if attributed == {nonterminal}:
        return 0
    return None


def _backfill_terminal_failure_locked(conn: sqlite3.Connection) -> None:
    """Reconstruct every unknown settled give-up flag. Caller holds the txn.

    Runs after the duplicate-running repair so a row this pass closed is judged
    on the status it actually ended with. Rows the evidence cannot settle keep
    their NULL and are re-examined — to the same answer — on the next open, so
    the pass is repeatable, appends no event, and rewrites no other column.
    """
    rows = conn.execute(
        "SELECT id, job_id, status, failure_class FROM job_attempts "
        "WHERE terminal_failure IS NULL AND status != 'running'"
    ).fetchall()
    for row in rows:
        value = _legacy_terminal_failure(conn, row)
        if value is not None:
            conn.execute(
                "UPDATE job_attempts SET terminal_failure = ? WHERE id = ?",
                (value, row["id"]),
            )


def _migrate(conn: sqlite3.Connection) -> None:
    """Add V2 columns, repair pre-V2 history, then database-enforce V2 uniqueness.

    ``add_column_if_missing`` swallows the duplicate-column error a concurrent
    migrator (or the fresh ``SCHEMA_SQL`` above) may have run first, so this is
    safe to call on a brand-new DB, an early V1 DB, and a fully migrated one.

    Repair strictly precedes index creation — see :data:`_ONE_RUNNING_INDEX_SQL`.
    Idempotent: every repair re-runs to the same answer and appends no duplicate
    event. A database still holding unreconstructible legacy rows re-opens the
    (empty) write transaction on each connect, which changes nothing.
    """
    for name, ddl in _JOBS_V2_COLUMNS:
        add_column_if_missing(conn, "jobs", name, ddl)
    # Skill attachment. Nullable, and every pre-existing Job arrives NULL, which
    # is exactly the truth about them: nobody declared any skills, so the default
    # rules table decides for them the same way it does for a new Job.
    add_column_if_missing(conn, "jobs", "skills", "skills TEXT")
    add_column_if_missing(conn, "job_attempts", "ordinal", "ordinal INTEGER")
    # Nullable on purpose. A blanket ``DEFAULT 0`` here would *falsify* every
    # settled row a pre-column V2 build finished with an explicit give-up, so
    # existing rows arrive as unknown and are reconstructed from real evidence.
    # (A database already migrated by a build that used the NOT NULL DEFAULT 0
    # column keeps those zeros: the falsification is lossy and cannot be undone
    # from the row alone. No released build ever wrote that column.)
    add_column_if_missing(
        conn, "job_attempts", "terminal_failure", "terminal_failure INTEGER"
    )
    # A response stored before these existed cannot be re-sealed from the row
    # alone, so it arrives NULL and :func:`_validated_response_locked` refuses
    # it rather than inventing a seal for bytes nobody can vouch for.
    add_column_if_missing(
        conn, "job_run_requests", "response_digest", "response_digest TEXT"
    )
    add_column_if_missing(
        conn, "job_run_requests", "receipt_digest", "receipt_digest TEXT"
    )
    # Deterministic and lossless: every legacy row arrives NULL and stays NULL.
    # Nothing is inferred from a receipt, an event, or a repository on disk — an
    # attempt whose approved base nobody recorded is legacy history, and
    # :func:`_v3_authority_locked` refuses to grant it replay authority rather
    # than promoting a guess into evidence. The CHECK constraints ride along
    # with ADD COLUMN, so a rewritten row is bounded on the way in.
    for table, name, ddl in (
        (
            "job_attempts", "base_commit",
            "base_commit TEXT CHECK "
            "(base_commit IS NULL OR length(base_commit) BETWEEN 7 AND 64)",
        ),
        (
            "job_run_requests", "route",
            "route TEXT CHECK (route IS NULL OR length(route) <= 200)",
        ),
        (
            "job_run_requests", "reason",
            "reason TEXT CHECK (reason IS NULL OR length(reason) BETWEEN 1 AND 64)",
        ),
        (
            "job_run_requests", "error",
            "error TEXT CHECK (error IS NULL OR length(error) <= 4096)",
        ),
    ):
        add_column_if_missing(conn, table, name, ddl)

    duplicate_running = conn.execute(
        "SELECT 1 FROM job_attempts WHERE status = 'running' "
        "GROUP BY job_id HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()
    missing_ordinal = conn.execute(
        "SELECT 1 FROM job_attempts WHERE ordinal IS NULL LIMIT 1"
    ).fetchone()
    unknown_terminal = conn.execute(
        "SELECT 1 FROM job_attempts WHERE terminal_failure IS NULL "
        "AND status != 'running' LIMIT 1"
    ).fetchone()
    if (
        duplicate_running is not None
        or missing_ordinal is not None
        or unknown_terminal is not None
    ):
        now = _now()
        with write_txn(conn):
            # Order matters: the duplicate-running repair reads the lineage the
            # reconstruction just completed, and the terminal-failure pass reads
            # the status the repair left behind.
            _reconstruct_ordinals_locked(conn)
            _repair_duplicate_running_locked(conn, now)
            _backfill_terminal_failure_locked(conn)

    conn.execute(_ONE_RUNNING_INDEX_SQL)
    try:
        conn.execute(_ORDINAL_INDEX_SQL)
    except sqlite3.IntegrityError:
        # A legacy database already holding two attempts on one ordinal. The
        # repair may not rewrite a permanent ordinal to make them unique, and
        # deleting a row would destroy execution evidence, so this database
        # simply cannot carry the index. It still opens with every row intact,
        # and :func:`_next_ordinal_locked` refuses to extend any Job whose
        # positions are not a clean 1..N, so nothing is written on top of the
        # contradiction.
        pass


@contextlib.contextmanager
def connect_closing(db_path: Optional[Path] = None):
    """Open a Jobs DB connection and guarantee it is closed on exit.

    sqlite3's connection context manager only commits/rollbacks; it does NOT
    close the file descriptor. Long-lived processes route many operations
    through ``connect()``; without closing, FDs to ``jobs.db`` accumulate.
    Mirrors ``projects_db.connect_closing`` / ``kanban_db.connect_closing``.
    """
    conn = connect(db_path=db_path)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now() -> int:
    return int(time.time())


def _new_job_id() -> str:
    return "j_" + secrets.token_hex(4)


def _new_attempt_id() -> str:
    return "a_" + secrets.token_hex(4)


def _new_receipt_id() -> str:
    return "r_" + secrets.token_hex(4)


def _new_claim_token() -> str:
    """An opaque, unguessable custody capability (never derived from job id)."""
    return secrets.token_urlsafe(32)


def _bump_revision_locked(
    conn: sqlite3.Connection, job_id: str, now: int
) -> None:
    """Increment a Job's monotonic revision. Caller holds the write txn.

    Used by mutators that don't already ``UPDATE jobs`` (attempt/receipt
    writes); lifecycle mutators fold ``revision = revision + 1`` into their own
    row update instead, so revision advances exactly once per state change.
    """
    conn.execute(
        "UPDATE jobs SET revision = revision + 1, updated_at = ? WHERE id = ?",
        (now, job_id),
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreflightRecord:
    id: str
    job_id: str
    attempt_id: Optional[str]
    execution_spec_digest: str
    expected_job_revision: int
    status: str
    failure_class: Optional[str]
    checks: Sequence[Mapping[str, object]]
    receipt_id: str
    idempotency_key: str
    created_at: int


@dataclass(frozen=True)
class TransitionWrite:
    job_id: str
    attempt_id: str
    source_state: Optional[str]
    target_state: str
    initiator_type: str
    initiator_id: str
    expected_job_revision: int
    evidence: Mapping[str, str]
    failure_class: Optional[str]
    blocker_code: Optional[str]
    receipt_id: str
    component: str
    component_version: str
    idempotency_key: str
    created_at: int


@dataclass
class Job:
    id: str
    number: int
    name: str
    goal: str
    status: str
    step: str
    created_at: int
    updated_at: int
    specialist: Optional[str] = None
    routing_reason: Optional[str] = None
    last_heartbeat_at: Optional[int] = None
    revision: int = 0
    claimed_by: Optional[str] = None
    claim_acquired_at: Optional[int] = None
    lease_expires_at: Optional[int] = None
    current_attempt_id: Optional[str] = None
    # ``None`` is unset (the default rules table may choose); ``[]`` is an
    # explicit "attach nothing". The two are not interchangeable.
    skills: Optional[List[str]] = None
    correlations: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Public human label, e.g. ``Job #7``."""
        return f"Job #{self.number}"

    def to_dict(self) -> dict:
        # ``claim_token`` is deliberately absent: it is the custody capability
        # and must never leak through list/show/events/JSON. ``claimed_by`` and
        # the lease timestamps are safe, non-secret custody visibility.
        return {
            "id": self.id,
            "number": self.number,
            "label": self.label,
            "name": self.name,
            "goal": self.goal,
            "status": self.status,
            "step": self.step,
            "specialist": self.specialist,
            "routing_reason": self.routing_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "revision": self.revision,
            "claimed_by": self.claimed_by,
            "claim_acquired_at": self.claim_acquired_at,
            "lease_expires_at": self.lease_expires_at,
            "current_attempt_id": self.current_attempt_id,
            "skills": None if self.skills is None else list(self.skills),
            "correlations": list(self.correlations),
        }


def _decode_skills(row: sqlite3.Row) -> Optional[List[str]]:
    """The declared skill names on a Job row, or ``None`` when it declared none.

    Anything that is not a JSON array of non-empty strings reads as ``None``.
    Only :func:`create_job` ever writes this column, and it writes a validated
    array, so a value this cannot parse is a row somebody rewrote by hand — and
    "unset" is the conservative reading of it: the Job attaches whatever its own
    words justify rather than whatever the malformed value happened to contain.
    """
    if "skills" not in row.keys() or row["skills"] is None:
        return None
    try:
        parsed = json.loads(row["skills"])
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    return [s for s in parsed if isinstance(s, str) and s]


def _job_from_row(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        number=row["number"],
        name=row["name"],
        goal=row["goal"],
        status=row["status"],
        step=row["step"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        specialist=row["specialist"],
        routing_reason=row["routing_reason"],
        last_heartbeat_at=row["last_heartbeat_at"],
        revision=row["revision"],
        claimed_by=row["claimed_by"],
        claim_acquired_at=row["claim_acquired_at"],
        lease_expires_at=row["lease_expires_at"],
        current_attempt_id=row["current_attempt_id"],
        skills=_decode_skills(row),
    )


# ---------------------------------------------------------------------------
# Events (append-only) — the atomic-write primitive every mutation uses
# ---------------------------------------------------------------------------


def _append_event_locked(
    conn: sqlite3.Connection,
    job_id: str,
    kind: str,
    *,
    data: Optional[dict] = None,
    idempotency_key: Optional[str] = None,
    now: Optional[int] = None,
) -> dict:
    """Append one event. Caller already holds a write transaction.

    When ``idempotency_key`` repeats for the same job, the existing event is
    returned unchanged and no new row is written.
    """
    if idempotency_key is not None:
        existing = conn.execute(
            "SELECT * FROM job_events WHERE job_id = ? AND idempotency_key = ?",
            (job_id, idempotency_key),
        ).fetchone()
        if existing is not None:
            return _event_to_dict(existing)
    now = _now() if now is None else now
    payload = None if data is None else json.dumps(data, ensure_ascii=False, sort_keys=True)
    cur = conn.execute(
        "INSERT INTO job_events (job_id, kind, data, idempotency_key, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (job_id, kind, payload, idempotency_key, now),
    )
    row = conn.execute(
        "SELECT * FROM job_events WHERE id = ?", (cur.lastrowid,)
    ).fetchone()
    return _event_to_dict(row)


def _event_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "kind": row["kind"],
        "data": json.loads(row["data"]) if row["data"] is not None else None,
        "idempotency_key": row["idempotency_key"],
        "created_at": row["created_at"],
    }


def _event_key_seen(
    conn: sqlite3.Connection, job_id: str, idempotency_key: str
) -> bool:
    """True when this job already has an event with ``idempotency_key``.

    Used by state-changing operations to make a repeated job-scoped key a
    whole-operation no-op *before* any mutation — keeping the job row and its
    append-only ledger from ever disagreeing.
    """
    return (
        conn.execute(
            "SELECT 1 FROM job_events WHERE job_id = ? AND idempotency_key = ? LIMIT 1",
            (job_id, idempotency_key),
        ).fetchone()
        is not None
    )


def append_event(
    conn: sqlite3.Connection,
    job_id: str,
    kind: str,
    *,
    data: Optional[dict] = None,
    idempotency_key: Optional[str] = None,
) -> dict:
    """Append an append-only event to a job in its own write transaction.

    The ledger is part of the Job aggregate, so writing an event advances the
    Job's revision exactly once, in the same transaction as the event — an
    observer polling ``revision`` must never read a changed Job as unchanged.
    A repeated ``idempotency_key`` is a whole-operation no-op: the recorded
    event comes back and the revision stays put.

    Sibling mutators bump their own revision (folded into their row update, or
    via :func:`_bump_revision_locked`), so the bump lives here rather than in
    ``_append_event_locked`` — putting it there would double-bump all of them.
    """
    if get_job(conn, job_id) is None:
        raise ValueError(f"no such job: {job_id}")
    now = _now()
    with write_txn(conn):
        if idempotency_key is not None and _event_key_seen(
            conn, job_id, idempotency_key
        ):
            return _append_event_locked(
                conn, job_id, kind, data=data, idempotency_key=idempotency_key
            )
        _bump_revision_locked(conn, job_id, now)
        return _append_event_locked(
            conn, job_id, kind, data=data, idempotency_key=idempotency_key, now=now
        )


def get_events(conn: sqlite3.Connection, id_or_number) -> List[dict]:
    """All events for a job in insertion order (append-only ledger)."""
    job = get_job(conn, id_or_number)
    if job is None:
        return []
    rows = conn.execute(
        "SELECT * FROM job_events WHERE job_id = ? ORDER BY id ASC", (job.id,)
    ).fetchall()
    return [_event_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Job creation + lookup
# ---------------------------------------------------------------------------


def create_job(
    conn: sqlite3.Connection,
    *,
    name: str,
    goal: str,
    specialist: Optional[str] = None,
    routing_reason: Optional[str] = None,
    correlations: Optional[Iterable[str]] = None,
    skills=None,
    origin: Optional[Mapping[str, object]] = None,
) -> str:
    """Create a Job and return its opaque ``j_`` id.

    ``goal`` is stored verbatim — never stripped or rewritten. ``name`` is the
    plain-English handle and must be non-empty. The job starts in
    ``working``/``routing``; a ``job_created`` event is appended atomically in
    the same transaction.

    ``skills`` is the optional declaration of which skills this Job's builder is
    handed as reference material (see :mod:`hermes_cli.jobs_skills`). It is
    validated *here*, at the write seam, so a traversal attempt or an absurd
    declaration is refused seconds after somebody typed it rather than minutes
    into a run nobody is watching. ``None`` leaves it unset.
    """
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("job name must not be empty")
    if goal is None or goal == "":
        raise ValueError("job goal must not be empty")

    with write_txn(conn):
        return _create_job_locked(
            conn,
            name=clean_name,
            goal=goal,
            specialist=specialist,
            routing_reason=routing_reason,
            correlations=correlations,
            skills=skills,
            origin=origin,
        )


def _create_job_locked(
    conn: sqlite3.Connection,
    *,
    name: str,
    goal: str,
    specialist: Optional[str],
    routing_reason: Optional[str],
    correlations: Optional[Iterable[str]],
    now: Optional[int] = None,
    skills=None,
    origin: Optional[Mapping[str, object]] = None,
) -> str:
    """Insert one Job (and its ``job_created`` event). Caller holds the txn.

    Shared by :func:`create_job` and :func:`create_or_get_job` so both allocate
    a number, insert the row at ``revision = 1``, and append the creation event
    inside a single IMMEDIATE transaction. ``name`` is assumed pre-validated.
    """
    jid = _new_job_id()
    now = _now() if now is None else now
    corr: List[str] = []
    for c in correlations or []:
        c = str(c).strip()
        if c and c not in corr:
            corr.append(c)
    # Imported here rather than at module scope: jobs_skills reaches the
    # filesystem and this store deliberately does not, so the coupling stays at
    # the one seam that validates a declaration.
    from hermes_cli import jobs_skills as jskills

    declared = None if skills is None else list(jskills.parse_declared_skills(skills))
    skills_json = None if declared is None else json.dumps(declared, ensure_ascii=False)

    # Allocate through the durable monotonic sequence, never MAX(number)+1: a
    # deleted Job row must not free its number for reuse. Atomic with the INSERT
    # because both run inside this one IMMEDIATE transaction.
    conn.execute("UPDATE job_number_seq SET last = last + 1 WHERE id = 1")
    number = conn.execute(
        "SELECT last FROM job_number_seq WHERE id = 1"
    ).fetchone()["last"]
    conn.execute(
        "INSERT INTO jobs "
        "(id, number, name, goal, status, step, specialist, routing_reason, "
        " created_at, updated_at, last_heartbeat_at, revision, skills) "
        "VALUES (?, ?, ?, ?, 'working', 'routing', ?, ?, ?, ?, NULL, 1, ?)",
        (jid, number, name, goal, specialist, routing_reason, now, now,
         skills_json),
    )
    for kanban_id in corr:
        conn.execute(
            "INSERT OR IGNORE INTO job_correlations "
            "(job_id, kanban_id, created_at) VALUES (?, ?, ?)",
            (jid, kanban_id, now),
        )
    if origin:
        platform = str(origin.get("platform") or "").strip()
        chat_id = str(origin.get("chat_id") or "").strip()
        if platform and chat_id:
            allowed = {
                key: str(origin.get(key) or "")[:512]
                for key in (
                    "platform",
                    "chat_id",
                    "session_id",
                    "chat_type",
                    "thread_id",
                    "user_id",
                    "profile",
                )
            }
            conn.execute(
                "INSERT INTO job_origins (job_id, origin, created_at) "
                "VALUES (?, ?, ?)",
                (jid, json.dumps(allowed, ensure_ascii=False, sort_keys=True), now),
            )
    _append_event_locked(
        conn,
        jid,
        "job_created",
        data={
            "number": number,
            "name": name,
            "specialist": specialist,
            "routing_reason": routing_reason,
            "skills": declared,
        },
        now=now,
    )
    return jid


def _resolve_number(ident) -> Optional[int]:
    """Parse an integer job number from ``7``, ``"7"``, ``"#7"``, ``"Job #7"``."""
    if isinstance(ident, bool):
        return None
    if isinstance(ident, int):
        return ident
    s = str(ident).strip()
    if not s:
        return None
    low = s.lower()
    if low.startswith("job #"):
        s = s[5:].strip()
    elif low.startswith("job#"):
        s = s[4:].strip()
    elif s.startswith("#"):
        s = s[1:].strip()
    if s.isdigit():
        return int(s)
    return None


def get_job(conn: sqlite3.Connection, id_or_number) -> Optional[Job]:
    """Resolve a job by opaque id first, then by integer/public ``Job #N``."""
    if isinstance(id_or_number, str) and id_or_number.startswith("j_"):
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (id_or_number,)
        ).fetchone()
        if row is not None:
            return _attach_correlations(conn, _job_from_row(row))
        return None

    number = _resolve_number(id_or_number)
    if number is not None:
        row = conn.execute(
            "SELECT * FROM jobs WHERE number = ?", (number,)
        ).fetchone()
        if row is not None:
            return _attach_correlations(conn, _job_from_row(row))

    # Fall back to an exact id match for non-``j_`` opaque ids.
    if isinstance(id_or_number, str):
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (id_or_number,)
        ).fetchone()
        if row is not None:
            return _attach_correlations(conn, _job_from_row(row))
    return None


def _attach_correlations(conn: sqlite3.Connection, job: Job) -> Job:
    rows = conn.execute(
        "SELECT kanban_id FROM job_correlations WHERE job_id = ? ORDER BY created_at ASC, kanban_id ASC",
        (job.id,),
    ).fetchall()
    job.correlations = [r["kanban_id"] for r in rows]
    return job


def list_jobs(
    conn: sqlite3.Connection, *, status: Optional[str] = None
) -> List[Job]:
    """All jobs, newest number last. Optionally filter by public status."""
    if status is not None and status not in PUBLIC_STATUSES:
        raise ValueError(f"invalid status filter: {status!r}")
    sql = "SELECT * FROM jobs"
    params: tuple = ()
    if status is not None:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY number ASC"
    rows = conn.execute(sql, params).fetchall()
    return [_attach_correlations(conn, _job_from_row(r)) for r in rows]


# ---------------------------------------------------------------------------
# Lifecycle transitions and step updates
# ---------------------------------------------------------------------------


def transition(
    conn: sqlite3.Connection,
    id_or_number,
    *,
    status: str,
    step: str,
    reason: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> Job:
    """Move a job to a new public ``status`` and ``step`` atomically.

    The current status is read and the public transition (``VALID_TRANSITIONS``)
    is validated *inside* the IMMEDIATE write transaction, so the event's
    ``from`` value always matches the actual serialized predecessor even under
    concurrent writers — never a stale pre-transaction read. A repeated
    job-scoped ``idempotency_key`` makes the whole operation a no-op before any
    mutation, keeping the job row and its append-only ledger in agreement. A
    ``job_transition`` event is appended in the same transaction as the update.
    """
    if status not in PUBLIC_STATUSES:
        raise ValueError(f"invalid status: {status!r} (expected one of {PUBLIC_STATUSES})")
    if step not in JOB_STEPS:
        raise ValueError(f"invalid step: {step!r} (expected one of {JOB_STEPS})")

    now = _now()
    with write_txn(conn):
        job = get_job(conn, id_or_number)
        if job is None:
            raise ValueError(f"no such job: {id_or_number!r}")
        # Operation-level idempotency: a seen key no-ops the whole op, before any
        # state change, returning the state the first operation left behind.
        if idempotency_key is not None and _event_key_seen(conn, job.id, idempotency_key):
            return job
        if status not in VALID_TRANSITIONS[job.status]:
            raise InvalidTransition(
                f"cannot transition job {job.label} from {job.status!r} to {status!r}"
            )
        if status == "finished" and step == "complete":
            latest = conn.execute(
                "SELECT id FROM job_attempts WHERE job_id = ? "
                "ORDER BY ordinal DESC LIMIT 1",
                (job.id,),
            ).fetchone()
            if latest is None:
                raise InvalidTransition(
                    f"SHIP GATE: job {job.label} has no attempt, so nothing was "
                    "built, merged, deployed, or verified live"
                )
        if status in _CUSTODY_CLEARING_STATUSES:
            # A general transition must never silently orphan a running attempt.
            # Clearing custody underneath one would strand it ``running`` with an
            # invalidated token — unfinishable — and the one-running index would
            # then block the Job from ever executing again. A general transition
            # carries no outcome, so the attempt has to be closed first through
            # the custody API, where the outcome is explicit.
            running = conn.execute(
                "SELECT id FROM job_attempts WHERE job_id = ? AND status = 'running'",
                (job.id,),
            ).fetchone()
            if running is not None:
                raise InvalidTransition(
                    f"job {job.label} has running attempt {running['id']}: finish "
                    f"it through finish_attempt before moving to {status!r}"
                )
            # Reaching needs_you/finished ends the active lease: clear the whole
            # claim block in the same write so no stale custody survives a pause
            # or completion. A no-op when the Job was never claimed.
            conn.execute(
                "UPDATE jobs SET status = ?, step = ?, updated_at = ?, "
                "revision = revision + 1, claimed_by = NULL, claim_token = NULL, "
                "claim_acquired_at = NULL, lease_expires_at = NULL, "
                "current_attempt_id = NULL WHERE id = ?",
                (status, step, now, job.id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET status = ?, step = ?, updated_at = ?, "
                "revision = revision + 1 WHERE id = ?",
                (status, step, now, job.id),
            )
        _append_event_locked(
            conn,
            job.id,
            "job_transition",
            data={
                "from": job.status,
                "to": status,
                "step": step,
                "reason": reason,
            },
            idempotency_key=idempotency_key,
            now=now,
        )
    return get_job(conn, job.id)


def reassign_specialist(
    conn: sqlite3.Connection,
    id_or_number,
    *,
    specialist: Optional[str],
    reason: str,
    idempotency_key: Optional[str] = None,
) -> Job:
    """Move an unclaimed Job to another lane with durable evidence."""

    if specialist is not None:
        if not isinstance(specialist, str) or not specialist.strip():
            raise ValueError("specialist must be a non-empty name or None")
        specialist = specialist.strip()
    if not str(reason or "").strip():
        raise ValueError("reassigning a Job's lane requires a reason")

    now = _now()
    with write_txn(conn):
        job = get_job(conn, id_or_number)
        if job is None:
            raise ValueError(f"no such job: {id_or_number!r}")
        if idempotency_key is not None and _event_key_seen(conn, job.id, idempotency_key):
            return job
        if job.status == "finished":
            raise InvalidTransition(
                f"job {job.label} is finished; its lane is a matter of record now"
            )
        if job.specialist == specialist:
            return job
        running = conn.execute(
            "SELECT id FROM job_attempts WHERE job_id = ? AND status = 'running'",
            (job.id,),
        ).fetchone()
        if running is not None:
            raise InvalidTransition(
                f"job {job.label} has running attempt {running['id']}: finish it "
                "before moving the Job to another lane"
            )
        held = conn.execute(
            "SELECT claim_token IS NOT NULL FROM jobs WHERE id = ?", (job.id,)
        ).fetchone()
        if held is not None and held[0]:
            raise InvalidTransition(
                f"job {job.label} is claimed by {job.claimed_by!r}: release the "
                "claim before moving the Job to another lane"
            )
        conn.execute(
            "UPDATE jobs SET specialist = ?, updated_at = ?, revision = revision + 1 "
            "WHERE id = ?",
            (specialist, now, job.id),
        )
        _append_event_locked(
            conn,
            job.id,
            "job_reassigned",
            data={"from": job.specialist, "to": specialist, "reason": reason},
            idempotency_key=idempotency_key,
            now=now,
        )
    return get_job(conn, job.id)


def set_step(
    conn: sqlite3.Connection,
    id_or_number,
    step: str,
    *,
    reason: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> Job:
    """Update a job's internal ``step`` without changing its public status.

    Steps are internal execution detail, never board columns. The current step
    is read inside the IMMEDIATE transaction so the event's ``from`` is the real
    serialized predecessor, and a repeated job-scoped ``idempotency_key`` no-ops
    the whole operation before any mutation. A ``job_step`` event is appended in
    the same transaction.
    """
    if step not in JOB_STEPS:
        raise ValueError(f"invalid step: {step!r} (expected one of {JOB_STEPS})")

    now = _now()
    with write_txn(conn):
        job = get_job(conn, id_or_number)
        if job is None:
            raise ValueError(f"no such job: {id_or_number!r}")
        if idempotency_key is not None and _event_key_seen(conn, job.id, idempotency_key):
            return job
        conn.execute(
            "UPDATE jobs SET step = ?, updated_at = ?, revision = revision + 1 "
            "WHERE id = ?",
            (step, now, job.id),
        )
        _append_event_locked(
            conn,
            job.id,
            "job_step",
            data={"from": job.step, "to": step, "reason": reason},
            idempotency_key=idempotency_key,
            now=now,
        )
    return get_job(conn, job.id)


# ---------------------------------------------------------------------------
# Attempts and correction lineage
# ---------------------------------------------------------------------------


def _attempt_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "parent_attempt_id": row["parent_attempt_id"],
        "ordinal": row["ordinal"],
        "specialist": row["specialist"],
        "status": row["status"],
        "failure_class": row["failure_class"],
        "repository": row["repository"],
        "base_commit": row["base_commit"],
        "branch": row["branch"],
        "worktree": row["worktree"],
        "commit": row["commit_sha"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "created_at": row["created_at"],
    }


def get_attempt(conn: sqlite3.Connection, attempt_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM job_attempts WHERE id = ?", (attempt_id,)
    ).fetchone()
    return _attempt_to_dict(row) if row is not None else None


def get_attempts(conn: sqlite3.Connection, id_or_number) -> List[dict]:
    """All attempts for a job in execution order (correction chain included).

    Order is the permanent per-Job ``ordinal`` — the same durable lineage
    :func:`_validated_ordinals` reads, and the only thing on the row that *is*
    execution order. Neither ``created_at`` nor ``id`` may decide it: two
    attempts can start within one second, and the ``a_`` id is random, so a
    correction whose id happens to sort first would be handed to a reader
    before the attempt it corrects.

    An attempt with no ordinal has no execution position to return. That is a
    legacy row whose place :func:`_reconstruct_ordinals_locked` could not prove
    from the parent chain, left unknown on purpose rather than guessed. It is
    still evidence and is never dropped, but it cannot be interleaved into a
    sequence it has no place in. Those rows sort after every known ordinal, by
    id purely so the result is stable — their ``ordinal`` comes back ``None``,
    which is how the caller knows the position is unknown rather than last.
    """
    job = get_job(conn, id_or_number)
    if job is None:
        return []
    rows = conn.execute(
        "SELECT * FROM job_attempts WHERE job_id = ? "
        "ORDER BY ordinal IS NULL ASC, ordinal ASC, id ASC",
        (job.id,),
    ).fetchall()
    return [_attempt_to_dict(r) for r in rows]


def _next_ordinal_locked(conn: sqlite3.Connection, job: Job) -> int:
    """The Job's next permanent position, or refuse. Caller holds the write txn.

    A permanent ordinal only means something on top of a history the Job's
    lineage can actually prove — see :func:`_validated_lineage`, the same
    authority every other V2 path reads. A complete run of ``1..N`` is not that
    proof: rows nobody can place can still carry a full set of numbers, and
    appending ``N+1`` there would assert an order the database cannot support.
    So would a row still holding a NULL, which the reconstruction pass left
    unnumbered on purpose. Either fails closed, with no attempt row and no event.

    Fresh Jobs and reconstructed legacy ones are unaffected: allocation here,
    on top of a validated chain, is what keeps them exactly ``1..N``.
    """
    rows = conn.execute(
        "SELECT id, ordinal, parent_attempt_id FROM job_attempts WHERE job_id = ?",
        (job.id,),
    ).fetchall()
    if not rows:
        return 1
    positions = _validated_lineage(rows)
    if positions is None or any(r["ordinal"] is None for r in rows):
        raise InvalidTransition(
            f"job {job.label} holds legacy attempts whose execution order could "
            "not be reconstructed; refusing to append a permanent ordinal after "
            "unknown history"
        )
    return len(rows) + 1


def start_attempt(
    conn: sqlite3.Connection,
    id_or_number,
    *,
    specialist: Optional[str] = None,
    parent_attempt_id: Optional[str] = None,
    claim_token: Optional[str] = None,
    repository: Optional[str] = None,
    base_commit: Optional[str] = None,
    branch: Optional[str] = None,
    worktree: Optional[str] = None,
    commit: Optional[str] = None,
    request_id: Optional[str] = None,
    request_owner: Optional[str] = None,
    now: Optional[int] = None,
) -> str:
    """Begin an execution attempt on a job and return its ``a_`` id.

    The attempt gets a permanent per-Job ``ordinal`` and becomes the Job's
    ``current_attempt_id``. It also continues the Job's execution lineage: it
    is recorded as the child of the attempt the lineage says came last, so the
    whole retry/review/correction chain stays one linear history under one
    root. That is not decoration — the lineage is what later grants replay
    authority and the next permanent position (:func:`_validated_lineage`), and
    a second parentless attempt would leave the Job with two histories that
    merely happen to be numbered in sequence, and no way to name either one.

    ``parent_attempt_id`` may state that link explicitly, and then it must be
    exactly the Job's latest attempt; naming a superseded one would fork the
    history by hand (``ValueError``). It must belong to the **same** job, which
    keeps the chain on one Job.

    Custody is mandatory: starting an attempt requires the Job's current,
    unexpired claim token. Missing, wrong, or expired all fail closed
    (``InvalidClaim``) with no write, so nobody — not even a caller that once
    held the claim — can seize the single running-attempt slot out from under
    the live claimholder. This supersedes Core V1's tokenless attempt calling.

    A ``finished`` job is terminal for new attempts (``InvalidTransition``). So
    is a Job still holding legacy attempts whose execution order the migration
    could not reconstruct — see :func:`_next_ordinal_locked`. The database's
    one-running partial unique index is the hard backstop: a second concurrent
    running attempt raises ``InvalidTransition`` rather than slipping through.

    ``base_commit`` is the approved starting point, recorded here because this
    is the moment it is still an *input* — after the run it is unreconstructible
    from anything the attempt leaves behind. It is written once and no later
    call may change it (:func:`settle_attempt` refuses a contradicting one), so
    it is the immutable anchor every replay check hangs off.

    ``request_id`` binds a reserved caller request to this exact Job, claim, and
    attempt, in this transaction. A binding that cannot be taken — unreserved,
    owned by somebody else, already bound elsewhere, already settled — raises
    and the attempt is not started at all.

    Every rejection leaves neither an attempt nor an event behind.
    """
    aid = _new_attempt_id()
    now = _now() if now is None else int(now)
    with write_txn(conn):
        job = get_job(conn, id_or_number)
        if job is None:
            raise ValueError(f"no such job: {id_or_number!r}")
        if job.status == "finished":
            raise InvalidTransition(
                f"cannot start a new attempt on finished job {job.label}"
            )
        if parent_attempt_id is not None:
            parent = get_attempt(conn, parent_attempt_id)
            if parent is None:
                raise ValueError(f"no such parent attempt: {parent_attempt_id!r}")
            if parent["job_id"] != job.id:
                raise ValueError(
                    f"parent attempt {parent_attempt_id!r} belongs to a different job"
                )
        _require_claim_token_locked(
            conn, job.id, claim_token, op="start_attempt", now=now
        )
        ordinal = _next_ordinal_locked(conn, job)
        leaf = _lineage_leaf(conn, job.id)
        if parent_attempt_id is None:
            parent_attempt_id = leaf
        elif parent_attempt_id != leaf:
            raise ValueError(
                f"parent attempt {parent_attempt_id!r} is not the latest attempt "
                f"of job {job.label}"
            )
        try:
            conn.execute(
                "INSERT INTO job_attempts "
                "(id, job_id, parent_attempt_id, specialist, status, failure_class, "
                " repository, base_commit, branch, worktree, commit_sha, started_at, "
                " finished_at, created_at, ordinal) "
                "VALUES (?, ?, ?, ?, 'running', NULL, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (
                    aid,
                    job.id,
                    parent_attempt_id,
                    specialist,
                    repository,
                    base_commit,
                    branch,
                    worktree,
                    commit,
                    now,
                    now,
                    ordinal,
                ),
            )
        except sqlite3.IntegrityError as exc:
            # The only enforceable unique here is the one-running partial index;
            # a random ``a_`` id and a pre-validated parent rule out the others.
            raise InvalidTransition(
                f"job {job.label} already has a running attempt"
            ) from exc
        conn.execute(
            "UPDATE jobs SET current_attempt_id = ?, updated_at = ?, "
            "revision = revision + 1 WHERE id = ?",
            (aid, now, job.id),
        )
        if request_id is not None:
            _bind_request_locked(
                conn, _clean_request_id(request_id),
                owner=_clean_owner(request_owner),
                job_id=job.id, attempt_id=aid, now=now,
            )
        _append_event_locked(
            conn,
            job.id,
            "attempt_started",
            data={
                "attempt_id": aid,
                "ordinal": ordinal,
                "parent_attempt_id": parent_attempt_id,
                "specialist": specialist,
            },
            now=now,
        )
    return aid


def _require_claim_token_locked(
    conn: sqlite3.Connection,
    job_id: str,
    claim_token: Optional[str],
    *,
    op: str,
    now: int,
) -> None:
    """Raise ``InvalidClaim`` unless ``claim_token`` is a live, unexpired claim.

    Three ways to fail, all closed and all before any mutation: no claim at all
    (a cleared token reads as ``None`` and never matches), the wrong token, or a
    lease that has already run out. Expiry matters as much as the token itself —
    once a lease lapses the worker has lost custody even though the row still
    carries its token, and :func:`recover_expired_claims`, not the worker, owns
    that claim. Constant-time compare avoids leaking the token through timing.
    Caller holds the write txn.
    """
    row = conn.execute(
        "SELECT claim_token, lease_expires_at FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    current = None if row is None else row["claim_token"]
    if current is None or not secrets.compare_digest(current, str(claim_token or "")):
        raise InvalidClaim(f"{op}: wrong or missing claim token")
    expires = row["lease_expires_at"]
    if expires is None or now > expires:
        raise InvalidClaim(f"{op}: claim lease expired")


def _outcome_for(
    status: str, failure_class: Optional[str], *, terminal_failure: bool
) -> tuple:
    """Map a terminal attempt result to the Job's ``(status, step)``.

    Deterministic and total — every terminal attempt has exactly one Job
    outcome, and the first matching rule wins. Nothing here reads the goal or
    any topic text: the classification comes only from the attempt's declared
    status and failure class, plus the caller's explicit ``terminal_failure``.
    """
    if status == "succeeded":
        return ("finished", "complete")
    if status == "review_rejected" or failure_class == "reviewer_rejection":
        # Reviewer rejection keeps the same Job working and returns it to the
        # builder for correction — it is never a blocker for Brandon.
        return ("working", "correcting")
    if failure_class == "authentication":
        return ("needs_you", "waiting_for_login")
    if failure_class in ("irreversible_action", "provider_billing"):
        # A provider/billing outage is a real operator blocker, not a retry.
        return ("needs_you", "waiting_for_decision")
    if status in ("cancelled", "interrupted") or failure_class == "infrastructure":
        # Infrastructure interruption/cancellation is nobody's fault: hand the
        # Job back to the router from a clean point.
        return ("working", "routing")
    if terminal_failure:
        return ("finished", "failed")
    # max_turns and recoverable implementation failure both come back for
    # another correction pass on the same Job.
    return ("working", "correcting")


# Caller keyword -> settled column, for terminal replay equivalence. Evidence
# is part of the result, not decoration: a replay that reports a different
# commit is asking for a different outcome than the one on record.
_REPLAY_EVIDENCE_COLUMNS = (
    ("commit", "commit_sha"),
    ("branch", "branch"),
    ("worktree", "worktree"),
    ("repository", "repository"),
)


def _replay_conflict(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    status: str,
    failure_class: Optional[str],
    terminal_failure: bool,
    evidence: Dict[str, Optional[str]],
) -> Optional[str]:
    """First materially conflicting field, or ``None`` for an identical replay.

    **Lineage authority comes first.** Handing back a settled result is an
    assertion about the Job's history — "this attempt, at this position, ended
    exactly this way" — so the same authority every other path answers to
    (:func:`_validated_ordinals`, re-read here inside the caller's write
    transaction from every attempt of the Job, in no particular order) has to
    place this attempt before any stored column is allowed to speak. A Job
    whose lineage is unknown, malformed or self-contradictory places nothing,
    and an attempt whose permanent ordinal is missing or disagrees with the
    position its lineage proves is not placed either. Then *no* replay value
    may be accepted, and both are refused without touching a row: a persisted
    give-up flag is a record of what some earlier build believed, and a flag
    that happens to match the caller's guess is a coincidence, not authority.
    The stored flags stay exactly as found — they are the evidence of what the
    row was, and rewriting them to make an answer available would destroy it.

    Once the attempt is placed, "material" is every caller-supplied input that
    defines the terminal result or the Job outcome it produces: the attempt
    status, the failure class, the explicit ``terminal_failure`` give-up flag,
    and the execution evidence. ``None`` evidence is *not supplied* — the write
    path leaves that column untouched, so an omitted value requests no change
    and cannot conflict.

    A NULL ``terminal_failure`` is legacy history the migration could not
    reconstruct (see :func:`_legacy_terminal_failure`). Neither answer is
    evidence there, so *every* replay conflicts: failing closed keeps a
    caller's guess from silently becoming the settled record.
    """
    ordinals = _validated_ordinals(conn, row["job_id"])
    if ordinals is None or ordinals.get(row["id"]) != row["ordinal"]:
        return (
            "lineage: the Job's execution history cannot place this attempt "
            "at its stored position"
        )
    settled_terminal = row["terminal_failure"]
    if settled_terminal is None:
        return "terminal_failure: the settled value is unknown legacy history"
    for name, settled, requested in (
        ("status", row["status"], status),
        ("failure_class", row["failure_class"], failure_class),
        ("terminal_failure", bool(settled_terminal), bool(terminal_failure)),
    ):
        if settled != requested:
            return f"{name} {settled!r} != {requested!r}"
    for name, column in _REPLAY_EVIDENCE_COLUMNS:
        requested = evidence[name]
        if requested is not None and row[column] != requested:
            return f"{name} {row[column]!r} != {requested!r}"
    return None


def _validate_finish_arguments(
    status: str, failure_class: Optional[str], *, terminal_failure: bool
) -> str:
    """Screen a terminal result before any lock is taken. Returns the status.

    Shared by :func:`finish_attempt` and :func:`settle_attempt` so the two ways
    to close an attempt cannot drift into accepting different results.
    """
    status = str(status or "").strip()
    if not status:
        raise ValueError("attempt finish status must not be empty")
    if status not in TERMINAL_ATTEMPT_STATUSES:
        raise ValueError(
            f"invalid attempt status: {status!r} "
            f"(expected one of {TERMINAL_ATTEMPT_STATUSES})"
        )
    if failure_class is not None and failure_class not in FAILURE_CLASSES:
        raise ValueError(
            f"invalid failure_class: {failure_class!r} "
            f"(expected one of {FAILURE_CLASSES} or None)"
        )
    if status == "succeeded" and failure_class is not None:
        raise ValueError("a succeeded attempt must not carry a failure_class")
    if terminal_failure and status != "failed":
        raise ValueError(
            "terminal_failure is only valid on a 'failed' attempt, "
            f"got {status!r}"
        )
    return status


def finish_attempt(
    conn: sqlite3.Connection,
    attempt_id: str,
    *,
    status: str,
    failure_class: Optional[str] = None,
    claim_token: Optional[str] = None,
    commit: Optional[str] = None,
    branch: Optional[str] = None,
    worktree: Optional[str] = None,
    repository: Optional[str] = None,
    terminal_failure: bool = False,
    now: Optional[int] = None,
) -> dict:
    """Close an attempt with a constrained terminal status and evidence.

    ``status`` must be one of :data:`TERMINAL_ATTEMPT_STATUSES` and
    ``failure_class`` one of :data:`FAILURE_CLASSES` (or ``None``). Custody
    mirrors :func:`start_attempt`: the Job's current, unexpired claim token is
    required, and missing/wrong/expired all fail closed (``InvalidClaim``).

    **The outcome is atomic.** In one transaction this closes the attempt,
    moves the Job to the public status/step :func:`_outcome_for` maps the result
    to, clears the whole claim block, and appends ``attempt_finished`` plus the
    matching ``job_transition``. A caller can never observe a "successful"
    attempt on a Job that is still ``working`` under an active claim.

    ``terminal_failure`` is the explicit opt-in for the one outcome that cannot
    be inferred — giving up on a Job for good (``finished``/``failed``). It is
    only valid on a ``failed`` attempt and is never derived from topic text.

    **Finishing is terminal.** Replaying the identical outcome returns the
    settled record with no duplicate event and no mutation; any *conflicting*
    outcome raises :class:`InvalidTransition` without touching a thing, so a
    stale or concurrent worker can never rewrite a result or its lineage.
    "Identical" is judged on every caller-supplied field that defines the
    result or the Job outcome — status, failure class, ``terminal_failure`` and
    the execution evidence — and only for an attempt the Job's validated
    lineage can place at its stored position. A Job whose history cannot say
    where this attempt sits has no settled result to hand back, so every replay
    of it fails closed. See :func:`_replay_conflict`.
    """
    status = _validate_finish_arguments(
        status, failure_class, terminal_failure=terminal_failure
    )
    if get_attempt(conn, attempt_id) is None:
        raise ValueError(f"no such attempt: {attempt_id!r}")

    now = _now() if now is None else int(now)
    with write_txn(conn):
        # Re-read inside the write lock: the attempt may have been closed by a
        # concurrent worker, by release, or by recovery since the pre-check.
        row = conn.execute(
            "SELECT * FROM job_attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"no such attempt: {attempt_id!r}")
        if row["status"] != "running":
            # Already terminal. An identical replay is a read of the settled
            # result — deliberately checked *before* custody, since finishing
            # cleared the very claim the replay's token refers to. Anything that
            # differs materially, or an attempt the Job's lineage cannot place,
            # fails closed rather than handing back a "success" for an outcome
            # that was never applied. This is the only settled-attempt replay
            # entry path: release_claim and recover_expired_claims reach
            # _finish_attempt_locked only for a row still 'running'.
            conflict = _replay_conflict(
                conn,
                row,
                status=status,
                failure_class=failure_class,
                terminal_failure=terminal_failure,
                evidence={
                    "commit": commit,
                    "branch": branch,
                    "worktree": worktree,
                    "repository": repository,
                },
            )
            if conflict is not None:
                raise InvalidTransition(
                    f"attempt {attempt_id} already finished; the replay "
                    f"conflicts on {conflict}"
                )
            return _attempt_to_dict(row)
        # The interruption path in recover_expired_claims calls
        # _finish_attempt_locked directly and is system-authoritative, so it is
        # unaffected by this check.
        _require_claim_token_locked(
            conn, row["job_id"], claim_token, op="finish_attempt", now=now
        )
        _finish_attempt_locked(
            conn,
            _attempt_to_dict(row),
            status=status,
            failure_class=failure_class,
            terminal_failure=terminal_failure,
            now=now,
            commit=commit,
            branch=branch,
            worktree=worktree,
            repository=repository,
            outcome=_outcome_for(
                status, failure_class, terminal_failure=terminal_failure
            ),
        )
    return get_attempt(conn, attempt_id)


def _finish_attempt_locked(
    conn: sqlite3.Connection,
    attempt: dict,
    *,
    status: str,
    failure_class: Optional[str],
    now: int,
    terminal_failure: bool = False,
    commit: Optional[str] = None,
    branch: Optional[str] = None,
    worktree: Optional[str] = None,
    repository: Optional[str] = None,
    outcome: Optional[tuple] = None,
    receipt: Optional[dict] = None,
    v3_narrative: Optional[dict] = None,
) -> None:
    """Close ``attempt``, optionally applying its Job outcome. Caller holds the txn.

    Shared by :func:`finish_attempt`, :func:`release_claim`, and
    :func:`recover_expired_claims` so all three write the same terminal attempt
    record and event. ``outcome`` is the ``(status, step)`` the Job moves to,
    and passing it also clears the whole claim block in the same write — the
    caller supplies it only when the attempt's result *is* the Job's outcome.
    Release and recovery pass ``None`` because they own the Job move themselves.
    Status/failure-class are assumed pre-validated by the caller.

    ``terminal_failure`` is recorded on the row so a later replay can be judged
    against the outcome that was actually requested. Release and recovery never
    give up on a Job, so their default of ``False`` is the truthful record.

    ``receipt`` is the attempt's final receipt. When supplied it is written in
    this same transaction, so a terminal attempt and the evidence for it either
    both exist or neither does.

    ``v3_narrative`` is ``{"request_id", "route", "reason", "error"}`` for a
    settlement that owns a caller request. Its presence is what makes the stored
    receipt the exact versioned V3 shape, stamped from the row this call just
    wrote rather than from the running row it started with.
    """
    sets = [
        "status = ?", "failure_class = ?", "finished_at = ?",
        "terminal_failure = ?",
    ]
    params: List[Any] = [status, failure_class, now, 1 if terminal_failure else 0]
    for col, val in (
        ("commit_sha", commit),
        ("branch", branch),
        ("worktree", worktree),
        ("repository", repository),
    ):
        if val is not None:
            sets.append(f"{col} = ?")
            params.append(val)
    params.append(attempt["id"])
    conn.execute(
        f"UPDATE job_attempts SET {', '.join(sets)} WHERE id = ?", params
    )
    if outcome is None:
        # Bump revision and detach this attempt from the Job if it was the
        # current one — a CASE keeps a concurrently-started newer attempt's
        # pointer intact.
        conn.execute(
            "UPDATE jobs SET revision = revision + 1, updated_at = ?, "
            "current_attempt_id = CASE WHEN current_attempt_id = ? THEN NULL "
            "ELSE current_attempt_id END WHERE id = ?",
            (now, attempt["id"], attempt["job_id"]),
        )
    else:
        job_status, job_step = outcome
        previous = conn.execute(
            "SELECT status FROM jobs WHERE id = ?", (attempt["job_id"],)
        ).fetchone()["status"]
        conn.execute(
            "UPDATE jobs SET revision = revision + 1, updated_at = ?, "
            "status = ?, step = ?, current_attempt_id = NULL, claimed_by = NULL, "
            "claim_token = NULL, claim_acquired_at = NULL, lease_expires_at = NULL "
            "WHERE id = ?",
            (now, job_status, job_step, attempt["job_id"]),
        )
    _append_event_locked(
        conn,
        attempt["job_id"],
        "attempt_finished",
        data={
            "attempt_id": attempt["id"],
            "status": status,
            "failure_class": failure_class,
        },
        now=now,
    )
    if outcome is not None:
        _append_event_locked(
            conn,
            attempt["job_id"],
            "job_transition",
            data={
                "from": previous,
                "to": job_status,
                "step": job_step,
                "reason": f"attempt {attempt['id']} {status}",
            },
            now=now,
        )
    if receipt is not None:
        # Judged against the terminal result just written, not the running row
        # this call started from — so the evidence columns in the receipt are
        # the ones the UPDATE above landed, re-read rather than assumed.
        settled = _attempt_to_dict(
            conn.execute(
                "SELECT * FROM job_attempts WHERE id = ?", (attempt["id"],)
            ).fetchone()
        )
        _add_final_receipt_locked(
            conn,
            settled,
            receipt,
            now=now,
            authoritative=(
                None if v3_narrative is None
                else _v3_receipt_envelope(settled, **v3_narrative)
            ),
        )


# ---------------------------------------------------------------------------
# The final receipt — one per terminal attempt, written by finalization only
# ---------------------------------------------------------------------------

# Every attempt owns exactly one key in this namespace. It is reserved: a public
# caller that could reserve it first would decide what a run "was" before the run
# happened, and the settled attempt would arrive to find its own evidence taken.
_FINAL_RECEIPT_SUFFIX = ":final"


def final_receipt_key(attempt_id: str) -> str:
    """The one idempotency key that may document a terminal attempt."""
    return f"{attempt_id}{_FINAL_RECEIPT_SUFFIX}"


def is_final_receipt_key(idempotency_key: Optional[str]) -> bool:
    return isinstance(idempotency_key, str) and idempotency_key.endswith(
        _FINAL_RECEIPT_SUFFIX
    )


def _canonical(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)


# The V3 final receipt is an exact, versioned shape. Every key below is
# mandatory: a receipt missing one cannot authorize a replay, because "the field
# was not there" and "the field agreed" are the same answer to a validator that
# only checks what it finds, and that is precisely the hole a forgery uses.
# ``evidence`` is the execution tuple repeated inside the same document, so a
# partial rewrite of the top-level keys contradicts the copy beside them.
V3_RECEIPT_SCHEMA_VERSION = 3
V3_RECEIPT_KIND = "jobs-attempt-final"
V3_RECEIPT_EVIDENCE_KEYS = (
    "repository", "base_commit", "branch", "worktree", "commit",
)
V3_RECEIPT_REQUIRED_KEYS = (
    "schema_version", "kind", "request_id", "job_id", "attempt_id", "ordinal",
    "repository", "base_commit", "branch", "worktree", "commit", "status",
    "failure_class", "route", "reason", "error", "evidence",
)


def _v3_receipt_envelope(
    attempt: dict,
    *,
    request_id: str,
    route: Optional[str],
    reason: Optional[str],
    error: Optional[str],
) -> dict:
    """The authoritative half of a V3 final receipt, built from settled state.

    Composed here rather than accepted from the caller so the receipt cannot be
    assembled from the wrong place: every value is read off the attempt this
    settlement just wrote, or off the request narrative settled with it.
    """
    evidence = {k: attempt.get(k) for k in V3_RECEIPT_EVIDENCE_KEYS}
    return {
        "schema_version": V3_RECEIPT_SCHEMA_VERSION,
        "kind": V3_RECEIPT_KIND,
        "request_id": request_id,
        "job_id": attempt["job_id"],
        "attempt_id": attempt["id"],
        "ordinal": attempt["ordinal"],
        "status": attempt["status"],
        "failure_class": attempt["failure_class"],
        "route": route,
        "reason": reason,
        "error": error,
        **evidence,
        "evidence": evidence,
    }


def _add_final_receipt_locked(
    conn: sqlite3.Connection,
    attempt: dict,
    receipt: dict,
    *,
    now: int,
    authoritative: Optional[dict] = None,
) -> str:
    """Write (or exactly re-read) an attempt's final receipt. Caller holds the txn.

    Two agreements are enforced before anything is stored, both of them the
    difference between evidence and fiction:

    - The receipt must report the same terminal status and failure class as the
      attempt it documents. The attempt is authoritative; a receipt that says
      otherwise means the caller built one of the two from the wrong place.
    - A repeat must match the stored receipt byte for byte. Idempotent replay is
      a *read* of a settled outcome, so a differing payload is a second,
      contradictory claim about the same attempt and is refused without mutation.

    ``authoritative`` is the V3 envelope (:func:`_v3_receipt_envelope`). When it
    is supplied the caller's receipt keeps only the keys the envelope does not
    own — a caller that *contradicts* an authoritative value is refused outright
    rather than silently overwritten, because a disagreement there means one of
    the two documents was built from the wrong state and publishing either would
    be a lie. Everything else the caller collected (adapter observations, output
    tails, worker self-reports) rides along untouched as ordinary evidence.
    """
    if not isinstance(receipt, dict):
        raise ValueError("final receipt data must be a JSON object (dict)")
    if authoritative is not None:
        for key, expected in authoritative.items():
            if key in ("schema_version", "kind", "evidence"):
                # Version/kind are stamped by definition, and ``evidence`` is
                # this module's own restatement — never the caller's to supply.
                continue
            if key in receipt and receipt[key] != expected:
                raise ValueError(
                    f"final receipt disagrees with attempt {attempt['id']} on "
                    f"{key}: receipt says {receipt[key]!r}, the store says "
                    f"{expected!r}"
                )
        receipt = {**receipt, **authoritative}
    if (receipt.get("status"), receipt.get("failure_class")) != (
        attempt["status"], attempt["failure_class"],
    ):
        raise ValueError(
            f"final receipt disagrees with attempt {attempt['id']}: receipt says "
            f"{receipt.get('status')!r}/{receipt.get('failure_class')!r}, attempt "
            f"is {attempt['status']!r}/{attempt['failure_class']!r}"
        )
    key = final_receipt_key(attempt["id"])
    encoded = _canonical(receipt)
    existing = conn.execute(
        "SELECT id, data FROM job_receipts WHERE job_id = ? AND idempotency_key = ?",
        (attempt["job_id"], key),
    ).fetchone()
    if existing is not None:
        if existing["data"] != encoded:
            raise ValueError(
                f"attempt {attempt['id']} already has a final receipt with "
                "different content; refusing to replace settled evidence"
            )
        return existing["id"]
    rid = _new_receipt_id()
    conn.execute(
        "INSERT INTO job_receipts "
        "(id, job_id, attempt_id, data, idempotency_key, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (rid, attempt["job_id"], attempt["id"], encoded, key, now),
    )
    _append_event_locked(
        conn,
        attempt["job_id"],
        "receipt_added",
        data={"receipt_id": rid, "attempt_id": attempt["id"]},
        now=now,
    )
    return rid


def settle_attempt(
    conn: sqlite3.Connection,
    attempt_id: str,
    *,
    status: str,
    receipt: dict,
    failure_class: Optional[str] = None,
    claim_token: Optional[str] = None,
    commit: Optional[str] = None,
    branch: Optional[str] = None,
    worktree: Optional[str] = None,
    repository: Optional[str] = None,
    base_commit: Optional[str] = None,
    route: Optional[str] = None,
    reason: Optional[str] = None,
    error: Optional[str] = None,
    terminal_failure: bool = False,
    request_id: Optional[str] = None,
    request_owner: Optional[str] = None,
    request_data: Optional[dict] = None,
    now: Optional[int] = None,
) -> dict:
    """Finish an attempt **and** write its final receipt in one transaction.

    :func:`finish_attempt` plus :func:`add_receipt` is two transactions, and the
    gap between them is a real crash window: a process that dies inside it leaves
    a permanently settled attempt with no evidence and nothing left running for
    recovery to repair. This is the same settlement, indivisible — if the receipt
    cannot be stored, the attempt did not finish, the Job did not move, custody
    did not clear, and no event was appended.

    Returns ``{"attempt": <settled attempt>, "receipt_id": <r_ id>}``.

    ``request_id`` settles the caller's reserved, bound request in this same
    transaction, so a caller that loses its output after the commit can find the
    settled result again instead of re-running the work. Two checks come first,
    both before any mutation: the request must be owned by ``request_owner`` and
    bound to exactly this Job and attempt, and both authoritative inputs — the
    ``receipt`` and ``request_data`` — must survive :func:`_screen_settlement`.
    A request already settled here returns its stored response and writes nothing.

    The returned ``run_record`` is that response: the caller's ``request_data``
    merged with the settled truth this transaction just wrote, then screened,
    bounded, sealed, and read back out of the column. The caller renders its
    answer from it, so the answer it gives now and the answer a replay gives
    later are the same bytes decoded twice rather than built twice.

    ``base_commit`` is not written here — :func:`start_attempt` already recorded
    the approved base and it is immutable. Passing one is an *assertion*, and a
    settlement asserting a different base than the attempt started from is a
    contradiction that fails closed with nothing written.

    ``route``/``reason``/``error`` are the run's narrative outcome: the routing
    decision, why the run ended, and the sanitized error if it ended badly. They
    settle into typed columns on the caller's request row in this same
    transaction and are restated in the final receipt, so neither document is
    the only place they exist.
    """
    _validate_finish_arguments(
        status, failure_class, terminal_failure=terminal_failure
    )
    if get_attempt(conn, attempt_id) is None:
        raise ValueError(f"no such attempt: {attempt_id!r}")

    now = _now() if now is None else int(now)
    with write_txn(conn):
        row = conn.execute(
            "SELECT * FROM job_attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"no such attempt: {attempt_id!r}")
        if base_commit is not None and row["base_commit"] != base_commit:
            raise InvalidTransition(
                f"attempt {attempt_id} started from base {row['base_commit']!r}, "
                f"not {base_commit!r}; the approved base is immutable"
            )
        # Every authoritative input, screened first, against the live capability,
        # while the store is still untouched. A refusal here rolls back a
        # transaction that has written nothing: hostile material is refused
        # before the attempt settles, not redacted into evidence after it has.
        capabilities = _capabilities_locked(conn, row["job_id"], claim_token)
        receipt = _screen_settlement(receipt, capabilities)
        if request_data is not None:
            request_data = _screen_settlement(dict(request_data), capabilities)
        # The narrative is settlement material too — it lands in typed columns
        # and in the final receipt — so it passes the same screen. A key that was
        # never supplied stays absent rather than becoming an empty string a
        # bounds constraint would have to argue with.
        narrative = _screen_settlement(
            {k: v for k, v in (
                ("route", route), ("reason", reason), ("error", error),
            ) if v is not None},
            capabilities,
        )
        narrative = {k: narrative.get(k) for k in ("route", "reason", "error")}
        v3_narrative = None if request_id is None else {
            "request_id": _clean_request_id(request_id), **narrative,
        }
        if request_id is not None:
            request_row = _owned_request_locked(
                conn, request_id, owner=request_owner,
                job_id=row["job_id"], attempt_id=attempt_id,
            )
            if request_row["response"] is not None:
                # Already settled under this request. An idempotent repeat is a
                # read of the answer that exists, never a second one.
                return {
                    "attempt": _attempt_to_dict(row),
                    "receipt_id": request_row["receipt_id"],
                    "run_record": _validated_response_locked(conn, request_row),
                }
        if row["status"] != "running":
            conflict = _replay_conflict(
                conn,
                row,
                status=status,
                failure_class=failure_class,
                terminal_failure=terminal_failure,
                evidence={
                    "commit": commit, "branch": branch,
                    "worktree": worktree, "repository": repository,
                },
            )
            if conflict is not None:
                raise InvalidTransition(
                    f"attempt {attempt_id} already finished; the replay "
                    f"conflicts on {conflict}"
                )
            settled = _attempt_to_dict(row)
            had_receipt = conn.execute(
                "SELECT 1 FROM job_receipts WHERE job_id = ? AND idempotency_key = ?",
                (row["job_id"], final_receipt_key(attempt_id)),
            ).fetchone() is not None
            receipt_id = _add_final_receipt_locked(
                conn, settled, receipt, now=now,
                authoritative=(
                    None if v3_narrative is None
                    else _v3_receipt_envelope(settled, **v3_narrative)
                ),
            )
            if not had_receipt:
                # An attempt that settled before this path existed can still be
                # given its evidence, and that is a mutation, so the revision has
                # to move. An identical replay of an already-documented attempt
                # writes nothing and leaves it exactly where it was.
                _bump_revision_locked(conn, row["job_id"], now)
            record = _run_record(
                conn, settled, receipt_id,
                request_id=request_id, request_data=request_data,
                narrative=narrative,
            )
            if request_id is not None:
                record = _store_response_locked(
                    conn, request_id, owner=request_owner,
                    job_id=row["job_id"], attempt_id=attempt_id,
                    receipt_id=receipt_id, record=record, now=now,
                    capabilities=capabilities, narrative=v3_narrative,
                )
            return {
                "attempt": settled,
                "receipt_id": receipt_id,
                "run_record": record,
            }
        _require_claim_token_locked(
            conn, row["job_id"], claim_token, op="settle_attempt", now=now
        )
        _finish_attempt_locked(
            conn,
            _attempt_to_dict(row),
            status=status,
            failure_class=failure_class,
            terminal_failure=terminal_failure,
            now=now,
            commit=commit,
            branch=branch,
            worktree=worktree,
            repository=repository,
            outcome=_outcome_for(
                status, failure_class, terminal_failure=terminal_failure
            ),
            receipt=receipt,
            v3_narrative=v3_narrative,
        )
        settled = _attempt_to_dict(
            conn.execute(
                "SELECT * FROM job_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
        )
        receipt_id = conn.execute(
            "SELECT id FROM job_receipts WHERE job_id = ? AND idempotency_key = ?",
            (row["job_id"], final_receipt_key(attempt_id)),
        ).fetchone()["id"]
        record = _run_record(
            conn, settled, receipt_id,
            request_id=request_id, request_data=request_data,
            narrative=narrative,
        )
        if request_id is not None:
            # Stored, then read back out of the column: an idempotent repeat
            # hands back the response this settlement wrote, never a fresher one.
            record = _store_response_locked(
                conn, request_id, owner=request_owner,
                job_id=row["job_id"], attempt_id=attempt_id,
                receipt_id=receipt_id, record=record, now=now,
                capabilities=capabilities, narrative=v3_narrative,
            )
        return {
            "attempt": settled, "receipt_id": receipt_id, "run_record": record,
        }


def _run_record(
    conn: sqlite3.Connection,
    settled: dict,
    receipt_id: str,
    *,
    request_id: Optional[str],
    request_data: Optional[dict],
    narrative: Optional[dict] = None,
) -> dict:
    """What this settlement was: the caller's data plus the truth just written.

    Composed inside the settling transaction, from the settled attempt and the
    Job as this settlement left it, so everything a caller needs to describe the
    run is fixed at the moment it becomes true. A description assembled afterwards
    from separate reads is a different thing: the Job can move on, and then the
    same request answered twice answers differently.

    Capability-free by construction — the claim token is not read here and no
    column carrying one is copied in.

    The authoritative keys are written *last* and unconditionally. A caller's
    ``request_data`` may carry anything it likes, but where it overlaps with a
    fact the store just settled, the store wins — otherwise the envelope a
    replay is validated against could be seeded with a value that never came
    from the attempt.
    """
    job = get_job(conn, settled["job_id"])
    narrative = narrative or {}
    return {
        **(request_data or {}),
        "request_id": None if request_id is None else str(request_id),
        "job_id": job.id,
        "job_number": job.number,
        "job_label": job.label,
        "job_status": job.status,
        "job_step": job.step,
        "attempt_id": settled["id"],
        "ordinal": settled["ordinal"],
        "status": settled["status"],
        "failure_class": settled["failure_class"],
        "repository": settled["repository"],
        "base_commit": settled["base_commit"],
        "commit": settled["commit"],
        "branch": settled["branch"],
        "worktree": settled["worktree"],
        "receipt_id": receipt_id,
        # The narrative this settlement is about to persist in typed columns,
        # not whatever the caller's ``request_data`` happened to say.
        **({} if not narrative else {
            "route": narrative["route"],
            "routing_reason": narrative["route"],
            "reason": narrative["reason"],
            "error": narrative["error"],
        }),
    }


# ---------------------------------------------------------------------------
# Caller request ownership — the protected replay seam
#
# One globally unique row per request id, reserved before any work, bound to the
# Job/claim/attempt that work runs on, and settled with the response in the same
# transaction as the attempt. Ordinary event insertion cannot reach any of it:
# the ledger is public evidence, and evidence never decides what a request was.
# ---------------------------------------------------------------------------


@dataclass
class RunRequest:
    """One caller request id and everything it owns."""

    request_id: str
    owner: str
    job_id: Optional[str]
    attempt_id: Optional[str]
    receipt_id: Optional[str]
    # The settled response, already validated against the world it names, or
    # ``None`` while the request is still only reserved.
    response: Optional[dict]


def _request_from_row(row: sqlite3.Row, response: Optional[dict]) -> RunRequest:
    return RunRequest(
        request_id=row["request_id"],
        owner=row["owner"],
        job_id=row["job_id"],
        attempt_id=row["attempt_id"],
        receipt_id=row["receipt_id"],
        response=response,
    )


def _request_row(conn: sqlite3.Connection, request_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM job_run_requests WHERE request_id = ?", (request_id,)
    ).fetchone()


def _clean_request_id(request_id) -> str:
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    return request_id


def _clean_owner(owner) -> str:
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("request owner must be a non-empty string")
    return owner


def _screen_settlement(data: dict, capabilities: Sequence[str]) -> dict:
    """The one screen both authoritative settlement inputs pass.

    Imported here rather than at module scope because ``jobs_exec`` imports this
    module; the cycle only exists on paper, and resolving it at call time keeps
    a single implementation of the receipt policy instead of a second copy that
    could drift from it.

    Recursive: secret- and capability-named keys are refused at any depth, so is
    a non-finite number, a non-string key, and anything with no JSON spelling;
    secret-*shaped* values and anything carrying a live capability are refused
    outright rather than redacted; text is capped; and the whole payload is
    bounded. ``allow_nan=False`` in :func:`_canonical` is the last gate — what
    goes in the column is standard JSON or nothing.
    """
    from hermes_cli import jobs_exec

    return jobs_exec.screen_settlement(data, capabilities=capabilities)


def _capabilities_locked(
    conn: sqlite3.Connection, job_id: str, *extra
) -> Tuple[str, ...]:
    """Every capability string this settlement must not be allowed to store.

    The Job's live claim token plus whatever the caller presented as one. Both,
    because a caller holding a token the row no longer has is still holding a
    capability, and a settlement is not the place to publish either.
    """
    row = conn.execute(
        "SELECT claim_token FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    found = (None if row is None else row["claim_token"], *extra)
    return tuple(c for c in found if isinstance(c, str) and c)


def _digest(text: str) -> str:
    """SHA-256 of exactly the bytes stored. See ``response_digest`` in the schema."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _loads_response(encoded: str):
    from hermes_cli import jobs_exec

    return jobs_exec.loads_strict(encoded)


def _validated_response_locked(
    conn: sqlite3.Connection, row: sqlite3.Row
) -> dict:
    """The stored response, checked against everything it claims to be.

    A response is only an answer if the world still agrees with it, and matching
    row ids are not that agreement — they are the cheapest part of a forgery. So
    three things are checked here, in order, and any one of them fails closed:

    - **The seal.** The response bytes and the final receipt's bytes still hash
      to what this settlement stored, which catches the single-row damage no
      relational check can see.
    - **The response against the world.** Every field that is re-derivable —
      the request, Job, attempt and receipt it names, the Job's permanent number,
      the attempt's ordinal, terminal status, failure class, and its
      commit/branch/worktree/repository evidence — is re-read from the live rows
      and compared. Nothing is believed because the row it sits in points at the
      right attempt.
    - **The receipt against the world.** It is parsed as strict standard JSON
      and held to the same attempt: a receipt that reports a different outcome,
      position, or evidence than the attempt it documents is a contradiction,
      and a contradiction is never quietly repaired into an answer.

    The Job's mutable state (status, step, label) is deliberately not re-derived:
    the Job moves on after a run, and a response that changed with it would stop
    being the answer to the request that was actually asked. Everything a caller
    *narrates* — why the run ended, its error text, its routing decision — is not
    left to the seal either: it settled into typed columns on this very row and
    is restated in the final receipt, so it is checked three ways round.
    """
    request_id = row["request_id"]

    def fail(why: str):
        raise ReplayIntegrity(
            f"stored response for request {request_id!r} {why}; refusing to "
            "return it"
        )

    if row["job_id"] is None or row["attempt_id"] is None or row["receipt_id"] is None:
        fail("is not bound to a job, attempt, and final receipt")
    if row["response_digest"] is None:
        fail("was stored without a seal, so its bytes cannot be vouched for")
    if not secrets.compare_digest(_digest(row["response"]), row["response_digest"]):
        fail("no longer matches the seal this settlement wrote")
    try:
        response = _loads_response(row["response"])
    except ValueError as exc:
        fail(f"is not standard JSON ({exc})")
    if not isinstance(response, dict):
        fail("is not a JSON object")

    attempt = get_attempt(conn, row["attempt_id"])
    if attempt is None:
        fail("names an attempt that does not exist")
    if attempt["job_id"] != row["job_id"]:
        fail("names an attempt belonging to a different job")
    if attempt["status"] == "running":
        fail("documents an attempt that has not settled")
    job = get_job(conn, row["job_id"])
    if job is None:
        fail("names a job that does not exist")

    receipt = conn.execute(
        "SELECT * FROM job_receipts WHERE id = ?", (row["receipt_id"],)
    ).fetchone()
    if receipt is None:
        fail("names a receipt that does not exist")
    if (receipt["job_id"], receipt["attempt_id"]) != (
        row["job_id"], row["attempt_id"],
    ):
        fail("names a receipt belonging to a different job or attempt")
    if receipt["idempotency_key"] != final_receipt_key(row["attempt_id"]):
        fail("names a receipt that is not that attempt's final receipt")
    if row["receipt_digest"] is None:
        fail("names a final receipt that was never sealed")
    if not secrets.compare_digest(_digest(receipt["data"]), row["receipt_digest"]):
        fail("names a final receipt that no longer matches its seal")

    authority = _v3_authority(row, job, attempt, fail)
    for field_name, expected in authority.items():
        if field_name not in response or response[field_name] != expected:
            fail(
                f"disagrees on {field_name}: says {response.get(field_name)!r}, "
                f"the store says {expected!r}"
            )
    if attempt["status"] == "succeeded" and not attempt["commit"]:
        fail("documents a succeeded attempt with no commit evidence")

    _validate_final_receipt(receipt["data"], attempt, authority, fail)
    return response


def _v3_authority(
    row: sqlite3.Row, job: "Job", attempt: dict, fail
) -> Dict[str, Any]:
    """Every field a V3 replay is *required* to reproduce, read from live rows.

    Mandatory, not best-effort: a validator that only checks the keys it happens
    to find treats "absent" and "agreed" as the same answer, and that is the hole
    a forged envelope walks through. So this is the whole set, and a response or
    receipt missing any one of them fails closed.

    An attempt with no ``base_commit`` never went through the V3 start path — it
    is legacy history the migration deliberately did not invent a base for — and
    legacy history cannot authorize a V3 replay.
    """
    if not attempt["base_commit"]:
        fail(
            "names an attempt with no approved base commit, which is legacy "
            "history and carries no V3 replay authority"
        )
    if not attempt["repository"]:
        fail("names an attempt with no repository evidence")
    if not row["reason"]:
        fail("was settled without a recorded outcome reason")
    return {
        "request_id": row["request_id"],
        "job_id": row["job_id"],
        "attempt_id": row["attempt_id"],
        "receipt_id": row["receipt_id"],
        "job_number": job.number,
        "ordinal": attempt["ordinal"],
        "status": attempt["status"],
        "failure_class": attempt["failure_class"],
        "repository": attempt["repository"],
        "base_commit": attempt["base_commit"],
        "branch": attempt["branch"],
        "worktree": attempt["worktree"],
        "commit": attempt["commit"],
        "route": row["route"],
        "routing_reason": row["route"],
        "reason": row["reason"],
        "error": row["error"],
    }


def _validate_final_receipt(
    encoded: str, attempt: dict, authority: Dict[str, Any], fail
) -> None:
    """Hold a final receipt to the exact, versioned V3 shape, or fail closed.

    Owning the reserved final-receipt key for the right attempt says the row is
    *addressed* correctly; it says nothing about what it reports. So the whole
    of :data:`V3_RECEIPT_REQUIRED_KEYS` must be present — a receipt that simply
    omits its commit or its base is not a lenient receipt, it is an unfalsifiable
    one — and every one of those keys must match the live attempt and the
    request's own typed narrative columns.

    ``evidence`` is the same execution tuple restated inside the document, so a
    rewrite that fixes up the top-level keys still has to fix up the copy beside
    them, and both still have to agree with the attempt.

    Extra keys are welcome and untouched: the adapter's observations, output
    tail, and worker self-report are evidence *about* the run, and none of them
    can decide what the run was.
    """
    try:
        data = _loads_response(encoded)
    except ValueError as exc:
        fail(f"names a final receipt that is not standard JSON ({exc})")
    if not isinstance(data, dict):
        fail("names a final receipt that is not a JSON object")
    missing = [k for k in V3_RECEIPT_REQUIRED_KEYS if k not in data]
    if missing:
        fail(
            "names a final receipt missing required V3 key(s): "
            + ", ".join(sorted(missing))
        )
    if data["schema_version"] != V3_RECEIPT_SCHEMA_VERSION:
        fail(
            f"names a final receipt of schema_version {data['schema_version']!r}, "
            f"which cannot authorize a V3 replay"
        )
    if data["kind"] != V3_RECEIPT_KIND:
        fail(f"names a final receipt of kind {data['kind']!r}")
    expected = {
        **{k: authority[k] for k in (
            "request_id", "job_id", "attempt_id", "ordinal", "status",
            "failure_class", "repository", "base_commit", "branch", "worktree",
            "commit", "route", "reason", "error",
        )},
        "evidence": {k: attempt[k] for k in V3_RECEIPT_EVIDENCE_KEYS},
    }
    for field_name, want in expected.items():
        if data[field_name] != want:
            fail(
                f"names a final receipt that disagrees on {field_name}: says "
                f"{data[field_name]!r}, the store says {want!r}"
            )


def find_run_response(
    conn: sqlite3.Connection, request_id: str
) -> Optional[dict]:
    """The settled response for ``request_id``, or ``None`` if it never settled.

    A pure read of the protected row — no event is scanned and none is trusted.
    Raises :class:`ReplayIntegrity` rather than returning a response the store
    itself contradicts, so a replay is either exact or refused.
    """
    row = _request_row(conn, _clean_request_id(request_id))
    if row is None or row["response"] is None:
        return None
    return _validated_response_locked(conn, row)


def reserve_request(
    conn: sqlite3.Connection,
    request_id: str,
    *,
    owner: str,
    lease_seconds: int,
    now: Optional[int] = None,
) -> RunRequest:
    """Take global ownership of one caller request id. One transaction.

    The primary key is the whole mechanism: the first writer owns the request
    for the entire database, so two Jobs and two concurrent runners cannot both
    decide what it was. Four outcomes, all deterministic and all idempotent:

    - **Free** — reserved for ``owner`` under a lease.
    - **Settled** — the response already exists; it comes straight back
      (validated) and nothing is written.
    - **Live** — somebody is running it: bound to an attempt that is still
      running, or reserved by another owner whose lease has not lapsed.
      :class:`RequestInProgress`, and no theft.
    - **Abandoned after binding** — bound to an attempt that is no longer
      running, which means the existing lease/attempt recovery path already
      settled the runner's work. The request is reconciled to that truth rather
      than executed a second time or deadlocked forever.

    An *unbound* reservation whose lease lapsed is taken over freely: nothing
    was ever bound to it, so no attempt, worker, or receipt exists behind it and
    a takeover cannot duplicate execution.
    """
    request_id = _clean_request_id(request_id)
    owner = _clean_owner(owner)
    lease = _validate_lease(lease_seconds)
    now = _now() if now is None else int(now)
    with write_txn(conn):
        row = _request_row(conn, request_id)
        if row is not None and row["response"] is not None:
            return _request_from_row(row, _validated_response_locked(conn, row))
        if row is not None and row["attempt_id"] is not None:
            return _reconcile_request_locked(conn, row, now=now)
        if (
            row is not None
            and row["owner"] != owner
            and row["lease_expires_at"] is not None
            and row["lease_expires_at"] >= now
        ):
            raise RequestInProgress(
                f"request {request_id!r} is already reserved by "
                f"{row['owner']!r} until {row['lease_expires_at']}"
            )
        expires = now + lease
        if row is None:
            conn.execute(
                "INSERT INTO job_run_requests "
                "(request_id, owner, lease_expires_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (request_id, owner, expires, now, now),
            )
        else:
            conn.execute(
                "UPDATE job_run_requests SET owner = ?, lease_expires_at = ?, "
                "updated_at = ? WHERE request_id = ?",
                (owner, expires, now, request_id),
            )
        return _request_from_row(_request_row(conn, request_id), None)


def _reconcile_request_locked(
    conn: sqlite3.Connection, row: sqlite3.Row, *, now: int
) -> RunRequest:
    """Finish a request whose runner died after binding. Caller holds the txn.

    The attempt it was bound to is terminal, so the work is over and its outcome
    is on record — settled normally, or made ``interrupted`` by the same claim
    recovery that repairs every other abandoned attempt. Reading the answer off
    that attempt is what keeps the request from either running twice or waiting
    forever for a process that is gone.

    ``recovered`` is empty and ``reason`` is ``attempt_reconciled``: this
    describes the *request's* run, and that run recovered nothing — it died. The
    caller's incidental recovery pass is a separate fact and is not backdated
    into an answer that has to replay identically forever.
    """
    request_id = row["request_id"]
    attempt = get_attempt(conn, row["attempt_id"])
    if attempt is None:
        raise ReplayIntegrity(
            f"request {request_id!r} is bound to attempt {row['attempt_id']!r}, "
            "which does not exist"
        )
    if row["job_id"] is not None and attempt["job_id"] != row["job_id"]:
        raise ReplayIntegrity(
            f"request {request_id!r} is bound to an attempt on a different job"
        )
    if attempt["status"] == "running":
        raise RequestInProgress(
            f"request {request_id!r} is being run right now by {row['owner']!r} "
            f"as attempt {attempt['id']}"
        )
    receipt = conn.execute(
        "SELECT id FROM job_receipts WHERE job_id = ? AND idempotency_key = ?",
        (attempt["job_id"], final_receipt_key(attempt["id"])),
    ).fetchone()
    if receipt is None:
        raise ReplayIntegrity(
            f"request {request_id!r} is bound to terminal attempt "
            f"{attempt['id']} with no final receipt"
        )
    narrative = {"request_id": request_id, **RECOVERED_NARRATIVE}
    record = _run_record(
        conn, attempt, receipt["id"],
        request_id=request_id,
        request_data={"ran": True, "recovered": []},
        narrative=narrative,
    )
    response = _store_response_locked(
        conn, request_id,
        owner=row["owner"],
        job_id=attempt["job_id"],
        attempt_id=attempt["id"],
        receipt_id=receipt["id"],
        record=record,
        now=now,
        narrative=narrative,
    )
    return _request_from_row(_request_row(conn, request_id), response)


def release_request(
    conn: sqlite3.Connection, request_id: str, *, owner: str
) -> bool:
    """Give back a reservation nothing was ever bound to. Idempotent.

    Only an unbound, unsettled reservation held by ``owner`` is released, so a
    run that decided not to start (nothing eligible, an unbuilt lane, a
    repository that is not there) frees the id immediately instead of parking it
    for a whole lease. A bound or settled request is never touched.
    """
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM job_run_requests WHERE request_id = ? AND owner = ? "
            "AND attempt_id IS NULL AND response IS NULL",
            (_clean_request_id(request_id), _clean_owner(owner)),
        )
    return cur.rowcount > 0


def _bind_request_locked(
    conn: sqlite3.Connection,
    request_id: str,
    *,
    owner: str,
    job_id: str,
    attempt_id: str,
    now: int,
) -> None:
    """Point a reservation at the Job and attempt it is about to run on.

    Called inside :func:`start_attempt`'s transaction, so the attempt and its
    ownership become true together: a binding that cannot be written takes the
    attempt down with it and leaves nothing behind to repair.
    """
    row = _request_row(conn, request_id)
    if row is None:
        raise ReplayIntegrity(f"request {request_id!r} was never reserved")
    if row["owner"] != owner:
        raise ReplayIntegrity(
            f"request {request_id!r} is owned by {row['owner']!r}, not {owner!r}"
        )
    if row["response"] is not None:
        raise ReplayIntegrity(
            f"request {request_id!r} already settled; it cannot start new work"
        )
    if row["attempt_id"] is not None and row["attempt_id"] != attempt_id:
        raise ReplayIntegrity(
            f"request {request_id!r} is already bound to attempt "
            f"{row['attempt_id']!r}"
        )
    conn.execute(
        "UPDATE job_run_requests SET job_id = ?, attempt_id = ?, updated_at = ? "
        "WHERE request_id = ?",
        (job_id, attempt_id, now, request_id),
    )


def _owned_request_locked(
    conn: sqlite3.Connection,
    request_id: str,
    *,
    owner: Optional[str],
    job_id: str,
    attempt_id: str,
) -> sqlite3.Row:
    """The request row this settlement is allowed to write, or raise.

    Read before the settlement touches anything, so a contradiction that was
    already in the store — a request owned by somebody else, bound to another
    Job or another attempt, or settled under a different attempt entirely — is
    refused while the attempt is still running and the Job is still untouched.
    """
    request_id = _clean_request_id(request_id)
    row = _request_row(conn, request_id)
    if row is None:
        raise ReplayIntegrity(f"request {request_id!r} was never reserved")
    if owner is None or row["owner"] != owner:
        raise ReplayIntegrity(
            f"request {request_id!r} is owned by {row['owner']!r}, not {owner!r}"
        )
    if row["attempt_id"] != attempt_id or row["job_id"] != job_id:
        raise ReplayIntegrity(
            f"request {request_id!r} is bound to job {row['job_id']!r} / attempt "
            f"{row['attempt_id']!r}, not {job_id!r} / {attempt_id!r}"
        )
    return row


def _store_response_locked(
    conn: sqlite3.Connection,
    request_id: str,
    *,
    owner: str,
    job_id: str,
    attempt_id: str,
    receipt_id: str,
    record: dict,
    now: int,
    capabilities: Sequence[str] = (),
    narrative: Optional[dict] = None,
) -> dict:
    """Persist one terminal response. The only writer of replay authority.

    Every response in the database went through :func:`_screen_settlement` here,
    including one written by a direct database caller: there is no second write
    path to bypass, so a secret, a capability, a non-finite number, or an
    oversized envelope cannot be stored by anybody, however they arrive.

    The response and the final receipt are sealed together, in this transaction,
    with the digests of exactly the bytes stored — so a later single-row change
    to either one is visible even when the row still names all the right ids.

    The run's narrative — route, reason, error — lands in typed, bounded columns
    beside the envelope rather than only inside it. That is what stops replay
    from having to trust a blob: the envelope and the final receipt both restate
    these three, and both are checked against the columns.

    Write-once, enforced in the UPDATE itself (``response IS NULL``). A settled
    request is authoritative forever, so there is no supported call — this one
    included — that can reach in and change what a run was.

    Returns the stored bytes decoded — not the in-memory ``record`` — so the
    answer the caller gives now is literally the bytes a replay will decode
    later, and the two cannot drift.
    """
    # The Job's own column plus whatever the settling caller was holding: by the
    # time this runs the settlement has already handed custody back and cleared
    # the column, so reading it alone would screen against nothing.
    encoded = _canonical(_screen_settlement(
        record, _capabilities_locked(conn, job_id, *capabilities)
    ))
    receipt = conn.execute(
        "SELECT data FROM job_receipts WHERE id = ?", (receipt_id,)
    ).fetchone()
    if receipt is None:
        raise ReplayIntegrity(
            f"request {request_id!r} cannot settle on final receipt "
            f"{receipt_id!r}, which does not exist"
        )
    narrative = narrative or {}
    cur = conn.execute(
        "UPDATE job_run_requests SET owner = ?, job_id = ?, attempt_id = ?, "
        "receipt_id = ?, response = ?, response_digest = ?, receipt_digest = ?, "
        "route = ?, reason = ?, error = ?, "
        "lease_expires_at = NULL, updated_at = ? WHERE request_id = ? "
        "AND response IS NULL",
        (owner, job_id, attempt_id, receipt_id, encoded, _digest(encoded),
         _digest(receipt["data"]), narrative.get("route"),
         narrative.get("reason"), narrative.get("error"), now, request_id),
    )
    if cur.rowcount != 1:
        raise ReplayIntegrity(
            f"request {request_id!r} is already settled; its response and its "
            "narrative are permanent"
        )
    return _loads_response(encoded)


# ---------------------------------------------------------------------------
# Receipts (durable, structured run evidence)
# ---------------------------------------------------------------------------


def _receipt_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "attempt_id": row["attempt_id"],
        "data": json.loads(row["data"]),
        "idempotency_key": row["idempotency_key"],
        "created_at": row["created_at"],
    }


def add_receipt(
    conn: sqlite3.Connection,
    id_or_number,
    *,
    data: dict,
    attempt_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> str:
    """Store a structured JSON run receipt on a job; return its ``r_`` id.

    V1 stores and retrieves receipts verbatim — it does not interpret
    provider-specific schemas. A repeated job-scoped ``idempotency_key``
    returns the existing receipt id (first write wins) and appends no duplicate
    row or event; the same key on a different job is allowed.

    The final-receipt namespace (see :func:`final_receipt_key`) is refused here.
    It belongs to :func:`settle_attempt`, which writes it in the same transaction
    as the attempt it documents. A caller that could take that key first would be
    deciding what a run "was" before the run had finished, and first-write-wins
    would then hand the settled attempt somebody else's story as its own.
    """
    if not isinstance(data, dict):
        raise ValueError("receipt data must be a JSON object (dict)")
    if is_final_receipt_key(idempotency_key):
        raise ValueError(
            f"idempotency key {idempotency_key!r} is in the final-receipt "
            "namespace, which only attempt settlement may write"
        )
    job = get_job(conn, id_or_number)
    if job is None:
        raise ValueError(f"no such job: {id_or_number!r}")
    if attempt_id is not None:
        attempt = get_attempt(conn, attempt_id)
        if attempt is None:
            raise ValueError(f"no such attempt: {attempt_id!r}")
        if attempt["job_id"] != job.id:
            raise ValueError(
                f"attempt {attempt_id!r} belongs to a different job"
            )

    now = _now()
    with write_txn(conn):
        if idempotency_key is not None:
            existing = conn.execute(
                "SELECT id FROM job_receipts WHERE job_id = ? AND idempotency_key = ?",
                (job.id, idempotency_key),
            ).fetchone()
            if existing is not None:
                return existing["id"]
        rid = _new_receipt_id()
        conn.execute(
            "INSERT INTO job_receipts "
            "(id, job_id, attempt_id, data, idempotency_key, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                rid,
                job.id,
                attempt_id,
                # allow_nan=False: NaN and the infinities have no JSON spelling,
                # so storing one writes a row no non-Python reader can parse.
                _canonical(data),
                idempotency_key,
                now,
            ),
        )
        _bump_revision_locked(conn, job.id, now)
        _append_event_locked(
            conn,
            job.id,
            "receipt_added",
            data={"receipt_id": rid, "attempt_id": attempt_id},
            now=now,
        )
    return rid


def get_receipts(conn: sqlite3.Connection, id_or_number) -> List[dict]:
    """All receipts for a job in insertion order, with parsed JSON payloads.

    The tiebreak is ``rowid``, not ``id``. Two receipts written inside the same
    second share a ``created_at``, and the ``r_`` id is random — ordering on it
    would hand back the second receipt first roughly half the time, which is not
    the insertion order this function promises. ``rowid`` is monotonic in
    insertion for this table, so it is the only column that actually knows.
    """
    job = get_job(conn, id_or_number)
    if job is None:
        return []
    rows = conn.execute(
        "SELECT * FROM job_receipts WHERE job_id = ? ORDER BY created_at ASC, rowid ASC",
        (job.id,),
    ).fetchall()
    return [_receipt_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Reliability decisions (signed, append-only, and transactionally paired)
# ---------------------------------------------------------------------------


def _reliability_json(value: object) -> str:
    from hermes_cli import jobs_receipts

    return jobs_receipts.canonical_json_bytes(value).decode("utf-8")


def _receipt_blob(envelope: Mapping[str, object], receipt_id: str) -> str:
    if not isinstance(envelope, Mapping):
        raise ValueError("receipt envelope must be a mapping")
    if envelope.get("receipt_id") != receipt_id:
        raise GraphConflict("receipt envelope id contradicts the reliability record")
    return _reliability_json(dict(envelope))


def _require_job_revision_locked(
    conn: sqlite3.Connection, job_id: str, expected_revision: int
) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise ValueError(f"no such job: {job_id!r}")
    if row["revision"] != expected_revision:
        raise StaleJobRevision(
            f"job {job_id!r} revision is {row['revision']}, expected {expected_revision}"
        )
    return row


def _require_attempt_job_locked(
    conn: sqlite3.Connection, attempt_id: Optional[str], job_id: str
) -> None:
    if attempt_id is None:
        return
    row = conn.execute(
        "SELECT job_id FROM job_attempts WHERE id = ?", (attempt_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"no such attempt: {attempt_id!r}")
    if row["job_id"] != job_id:
        raise ValueError(f"attempt {attempt_id!r} belongs to a different job")


def _insert_reliability_receipt_locked(
    conn: sqlite3.Connection,
    *,
    receipt_id: str,
    job_id: str,
    attempt_id: Optional[str],
    blob: str,
    created_at: int,
) -> None:
    existing = conn.execute(
        "SELECT * FROM job_receipts WHERE id = ?", (receipt_id,)
    ).fetchone()
    idempotency_key = f"reliability:{receipt_id}"
    if existing is not None:
        if (
            existing["job_id"] == job_id
            and existing["attempt_id"] == attempt_id
            and existing["data"] == blob
            and existing["idempotency_key"] == idempotency_key
        ):
            return
        raise GraphConflict(f"receipt {receipt_id!r} already stores different facts")
    conn.execute(
        "INSERT INTO job_receipts "
        "(id, job_id, attempt_id, data, idempotency_key, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (receipt_id, job_id, attempt_id, blob, idempotency_key, created_at),
    )


def _preflight_material(row: sqlite3.Row) -> tuple:
    return tuple(
        row[name]
        for name in (
            "job_id",
            "attempt_id",
            "execution_spec_digest",
            "expected_job_revision",
            "status",
            "failure_class",
            "checks_json",
            "receipt_id",
            "idempotency_key",
            "created_at",
        )
    )


def record_preflight(
    conn: sqlite3.Connection,
    record: PreflightRecord,
    envelope: Mapping[str, object],
) -> str:
    """Persist one complete preflight and its signed receipt atomically."""

    if record.status not in {"PASS", "BLOCKED"}:
        raise ValueError("preflight status must be PASS or BLOCKED")
    if not record.idempotency_key:
        raise ValueError("preflight idempotency_key must not be empty")
    checks_json = _reliability_json(list(record.checks))
    receipt_blob = _receipt_blob(envelope, record.receipt_id)
    expected_material = (
        record.job_id,
        record.attempt_id,
        record.execution_spec_digest,
        record.expected_job_revision,
        record.status,
        record.failure_class,
        checks_json,
        record.receipt_id,
        record.idempotency_key,
        record.created_at,
    )
    with write_txn(conn):
        _require_job_revision_locked(
            conn, record.job_id, record.expected_job_revision
        )
        _require_attempt_job_locked(conn, record.attempt_id, record.job_id)
        existing = conn.execute(
            "SELECT * FROM job_preflights WHERE job_id = ? AND idempotency_key = ?",
            (record.job_id, record.idempotency_key),
        ).fetchone()
        if existing is not None:
            stored_receipt = conn.execute(
                "SELECT data FROM job_receipts WHERE id = ?", (existing["receipt_id"],)
            ).fetchone()
            if (
                _preflight_material(existing) == expected_material
                and stored_receipt is not None
                and stored_receipt["data"] == receipt_blob
            ):
                return existing["id"]
            raise GraphConflict("preflight idempotency key stores different facts")
        _insert_reliability_receipt_locked(
            conn,
            receipt_id=record.receipt_id,
            job_id=record.job_id,
            attempt_id=record.attempt_id,
            blob=receipt_blob,
            created_at=record.created_at,
        )
        conn.execute(
            "INSERT INTO job_preflights "
            "(id, job_id, attempt_id, execution_spec_digest, expected_job_revision, "
            " status, failure_class, checks_json, receipt_id, idempotency_key, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.job_id,
                record.attempt_id,
                record.execution_spec_digest,
                record.expected_job_revision,
                record.status,
                record.failure_class,
                checks_json,
                record.receipt_id,
                record.idempotency_key,
                record.created_at,
            ),
        )
    return record.id


def _transition_material(row: sqlite3.Row) -> tuple:
    return tuple(
        row[name]
        for name in (
            "job_id",
            "attempt_id",
            "source_state",
            "target_state",
            "initiator_type",
            "initiator_id",
            "expected_job_revision",
            "evidence_json",
            "failure_class",
            "blocker_code",
            "receipt_id",
            "component",
            "component_version",
            "idempotency_key",
            "created_at",
        )
    )


def record_transition(
    conn: sqlite3.Connection,
    request: TransitionWrite,
    envelope: Mapping[str, object],
) -> int:
    """Persist one detailed-state transition and receipt in one transaction."""

    if not request.idempotency_key:
        raise ValueError("transition idempotency_key must not be empty")
    evidence_json = _reliability_json(dict(request.evidence))
    receipt_blob = _receipt_blob(envelope, request.receipt_id)
    expected_material = (
        request.job_id,
        request.attempt_id,
        request.source_state,
        request.target_state,
        request.initiator_type,
        request.initiator_id,
        request.expected_job_revision,
        evidence_json,
        request.failure_class,
        request.blocker_code,
        request.receipt_id,
        request.component,
        request.component_version,
        request.idempotency_key,
        request.created_at,
    )
    with write_txn(conn):
        existing = conn.execute(
            "SELECT * FROM job_attempt_transitions "
            "WHERE attempt_id = ? AND idempotency_key = ?",
            (request.attempt_id, request.idempotency_key),
        ).fetchone()
        if existing is not None:
            stored_receipt = conn.execute(
                "SELECT data FROM job_receipts WHERE id = ?", (existing["receipt_id"],)
            ).fetchone()
            if (
                _transition_material(existing) == expected_material
                and stored_receipt is not None
                and stored_receipt["data"] == receipt_blob
            ):
                return int(existing["id"])
            raise GraphConflict("transition idempotency key stores different facts")
        _require_job_revision_locked(
            conn, request.job_id, request.expected_job_revision
        )
        _require_attempt_job_locked(conn, request.attempt_id, request.job_id)
        _insert_reliability_receipt_locked(
            conn,
            receipt_id=request.receipt_id,
            job_id=request.job_id,
            attempt_id=request.attempt_id,
            blob=receipt_blob,
            created_at=request.created_at,
        )
        cursor = conn.execute(
            "INSERT INTO job_attempt_transitions "
            "(job_id, attempt_id, source_state, target_state, initiator_type, "
            " initiator_id, expected_job_revision, evidence_json, failure_class, "
            " blocker_code, receipt_id, component, component_version, "
            " idempotency_key, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            expected_material,
        )
        transition_id = int(cursor.lastrowid)
    return transition_id


def _transition_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "attempt_id": row["attempt_id"],
        "source_state": row["source_state"],
        "target_state": row["target_state"],
        "initiator_type": row["initiator_type"],
        "initiator_id": row["initiator_id"],
        "expected_job_revision": row["expected_job_revision"],
        "evidence": json.loads(row["evidence_json"]),
        "failure_class": row["failure_class"],
        "blocker_code": row["blocker_code"],
        "receipt_id": row["receipt_id"],
        "component": row["component"],
        "component_version": row["component_version"],
        "idempotency_key": row["idempotency_key"],
        "created_at": row["created_at"],
    }


def latest_transition(
    conn: sqlite3.Connection, attempt_id: str
) -> Optional[dict]:
    row = conn.execute(
        "SELECT * FROM job_attempt_transitions WHERE attempt_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (attempt_id,),
    ).fetchone()
    return None if row is None else _transition_to_dict(row)


def list_transitions(conn: sqlite3.Connection, job_id: str) -> List[dict]:
    rows = conn.execute(
        "SELECT * FROM job_attempt_transitions WHERE job_id = ? ORDER BY id ASC",
        (job_id,),
    ).fetchall()
    return [_transition_to_dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Heartbeat and truthful stale projection
# ---------------------------------------------------------------------------


def heartbeat(
    conn: sqlite3.Connection, id_or_number, *, at: Optional[int] = None
) -> Job:
    """Record a liveness heartbeat on a job.

    Updates ``last_heartbeat_at`` and appends a ``heartbeat`` event in the same
    transaction (the append-only invariant covers every mutating write).

    ponytail: V1 has no live poller, so per-beat events cost nothing here. If a
    future phase adds a high-frequency dispatcher, throttle or omit the
    heartbeat event to keep the ledger quiet.
    """
    job = get_job(conn, id_or_number)
    if job is None:
        raise ValueError(f"no such job: {id_or_number!r}")
    ts = _now() if at is None else int(at)
    with write_txn(conn):
        conn.execute(
            "UPDATE jobs SET last_heartbeat_at = ?, updated_at = ?, "
            "revision = revision + 1 WHERE id = ?",
            (ts, ts, job.id),
        )
        _append_event_locked(conn, job.id, "heartbeat", data={"at": ts}, now=ts)
    return get_job(conn, job.id)


def is_stale(
    status: str,
    last_heartbeat_at: Optional[int],
    *,
    now: int,
    stale_threshold: Optional[float],
) -> bool:
    """Pure staleness rule — true only for a working job whose heartbeat aged out.

    Stale is reported only when the public status is ``working``, a heartbeat
    exists, the caller supplies a positive threshold, and ``now`` exceeds the
    heartbeat plus the threshold. Everything else (no heartbeat, no/zero/
    negative threshold, non-working status) is not stale.
    """
    if status != "working":
        return False
    if last_heartbeat_at is None:
        return False
    if stale_threshold is None or stale_threshold <= 0:
        return False
    return now > last_heartbeat_at + stale_threshold


def projection(
    conn: sqlite3.Connection,
    id_or_number,
    *,
    now: int,
    stale_threshold: Optional[float] = None,
) -> Optional[dict]:
    """Read-only status projection for a job, including truthful ``stale``.

    Never mutates the job; a stale projection does not claim the job is
    actively running, it only reports that a ``working`` job's heartbeat has
    aged past the caller's threshold.
    """
    job = get_job(conn, id_or_number)
    if job is None:
        return None
    out = job.to_dict()
    out["stale"] = is_stale(
        job.status, job.last_heartbeat_at, now=now, stale_threshold=stale_threshold
    )
    return out


# ---------------------------------------------------------------------------
# Execution custody — claims and leases
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """The result of a successful claim: a Job plus its custody capability.

    ``claim_token`` is returned here ONCE, and only here — it never rides on a
    ``Job`` object, a projection, an event, or any list/show output. The caller
    keeps it (a permission-checked file in the CLI) to authorize attempt
    start/finish and heartbeat on this Job.
    """

    job: Job
    # repr=False, exactly as on ``JobEnvelope``: a traceback, a debug log, or a
    # bare ``print(claim)`` must never carry the capability. ``str()`` of a
    # dataclass falls through to ``__repr__``, so this closes both. Read the
    # field deliberately instead.
    claim_token: str = field(repr=False)


def _validate_lease(lease_seconds) -> int:
    """A lease must be a positive integer bounded by ``MAX_LEASE_SECONDS``."""
    if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int):
        raise ValueError("lease_seconds must be a positive integer")
    if lease_seconds <= 0 or lease_seconds > MAX_LEASE_SECONDS:
        raise ValueError(
            f"lease_seconds must be in 1..{MAX_LEASE_SECONDS}, got {lease_seconds}"
        )
    return lease_seconds


def _select_claimable_locked(conn, *, specialist, job) -> Optional[sqlite3.Row]:
    """Pick one claimable Job row, or ``None``. Caller holds the write txn.

    Eligibility: public status ``working``, an executable step, and no active
    claim (``claim_token IS NULL`` — an expired-but-uncleared claim is never
    silently stolen; :func:`recover_expired_claims` must clear it first).
    ``specialist=None`` means unassigned routing work (``specialist IS NULL``);
    a named specialist matches only its own Jobs. Selection is oldest permanent
    number first, unless an explicit ``job`` identity is requested.
    """
    placeholders = ",".join("?" * len(EXECUTABLE_STEPS))
    clauses = [
        "status = 'working'",
        "claim_token IS NULL",
        f"step IN ({placeholders})",
    ]
    params: List[Any] = list(EXECUTABLE_STEPS)
    if specialist is None:
        clauses.append("specialist IS NULL")
    else:
        clauses.append("specialist = ?")
        params.append(specialist)
    if job is not None:
        target = get_job(conn, job)
        if target is None:
            raise ValueError(f"no such job: {job!r}")
        clauses.append("id = ?")
        params.append(target.id)
    sql = (
        "SELECT * FROM jobs WHERE "
        + " AND ".join(clauses)
        + " ORDER BY number ASC LIMIT 1"
    )
    return conn.execute(sql, params).fetchone()


def claim_job(
    conn: sqlite3.Connection,
    *,
    worker: str,
    lease_seconds: int,
    specialist: Optional[str] = None,
    job=None,
    now: Optional[int] = None,
) -> Optional[Claim]:
    """Atomically claim one eligible Job and return its :class:`Claim`.

    The whole select-and-claim runs in one IMMEDIATE transaction, so two racing
    workers can never both win the same Job: the second serializes behind the
    first and re-reads the now-claimed row as ineligible. Returns ``None`` when
    nothing eligible is available (never an error — the caller polls).
    """
    worker = str(worker or "").strip()
    if not worker:
        raise ValueError("claim worker must not be empty")
    _validate_lease(lease_seconds)
    now = _now() if now is None else int(now)
    token = _new_claim_token()

    with write_txn(conn):
        row = _select_claimable_locked(conn, specialist=specialist, job=job)
        if row is None:
            return None
        jid = row["id"]
        expires = now + lease_seconds
        conn.execute(
            "UPDATE jobs SET claimed_by = ?, claim_token = ?, "
            "claim_acquired_at = ?, lease_expires_at = ?, last_heartbeat_at = ?, "
            "updated_at = ?, revision = revision + 1 WHERE id = ?",
            (worker, token, now, expires, now, now, jid),
        )
        _append_event_locked(
            conn,
            jid,
            "claim_acquired",
            data={"worker": worker, "lease_expires_at": expires},
            now=now,
        )
        claimed = get_job(conn, jid)
    return Claim(job=claimed, claim_token=token)


def claim_heartbeat(
    conn: sqlite3.Connection,
    id_or_number,
    *,
    claim_token: str,
    lease_seconds: int,
    now: Optional[int] = None,
) -> Job:
    """Extend a live claim's lease and refresh the Job heartbeat, atomically.

    Fails closed (``InvalidClaim``, no mutation) on a wrong/missing token or an
    already-expired lease — a worker that has lost its claim can never revive
    it; recovery owns the expired claim. ``now`` is explicit for deterministic
    tests. Appends a ``claim_heartbeat`` event and bumps the revision.
    """
    _validate_lease(lease_seconds)
    now = _now() if now is None else int(now)
    with write_txn(conn):
        job = get_job(conn, id_or_number)
        if job is None:
            raise ValueError(f"no such job: {id_or_number!r}")
        _require_claim_token_locked(
            conn, job.id, claim_token, op="claim_heartbeat", now=now
        )
        expires = now + lease_seconds
        conn.execute(
            "UPDATE jobs SET lease_expires_at = ?, last_heartbeat_at = ?, "
            "updated_at = ?, revision = revision + 1 WHERE id = ?",
            (expires, now, now, job.id),
        )
        _append_event_locked(
            conn,
            job.id,
            "claim_heartbeat",
            data={"lease_expires_at": expires},
            now=now,
        )
    return get_job(conn, job.id)


def release_claim(
    conn: sqlite3.Connection,
    id_or_number,
    *,
    claim_token: str,
    now: Optional[int] = None,
) -> Job:
    """Hand a valid claim back to ``working`` for deterministic retry.

    Requires the live claim token (fails closed otherwise). Any running attempt
    is closed as ``cancelled`` in the same transaction *before* custody clears:
    releasing underneath a running attempt would strand it with an invalidated
    token, and the one-running index would then block the Job from ever
    executing again. Clears the whole claim block so the Job is immediately
    re-claimable, keeps the current step (the worker chose to hand off
    mid-flight), appends ``claim_released``, and bumps the revision.
    """
    now = _now() if now is None else int(now)
    with write_txn(conn):
        job = get_job(conn, id_or_number)
        if job is None:
            raise ValueError(f"no such job: {id_or_number!r}")
        _require_claim_token_locked(
            conn, job.id, claim_token, op="release_claim", now=now
        )
        running = conn.execute(
            "SELECT * FROM job_attempts WHERE job_id = ? AND status = 'running'",
            (job.id,),
        ).fetchone()
        cancelled_id = None
        if running is not None:
            # The Job move belongs to release, so no outcome is applied here.
            _finish_attempt_locked(
                conn,
                _attempt_to_dict(running),
                status="cancelled",
                failure_class=None,
                now=now,
            )
            cancelled_id = running["id"]
        conn.execute(
            "UPDATE jobs SET status = 'working', updated_at = ?, "
            "revision = revision + 1, claimed_by = NULL, claim_token = NULL, "
            "claim_acquired_at = NULL, lease_expires_at = NULL, "
            "current_attempt_id = NULL WHERE id = ?",
            (now, job.id),
        )
        _append_event_locked(
            conn,
            job.id,
            "claim_released",
            data={"worker": job.claimed_by, "cancelled_attempt_id": cancelled_id},
            now=now,
        )
    return get_job(conn, job.id)


# The one narrative a recovered attempt ever gets. Fixed, because the receipt
# recovery writes and the response :func:`_reconcile_request_locked` writes later
# have to agree, and they are produced by two different calls at two different
# times. A constant is the only way they cannot drift.
RECOVERED_NARRATIVE = {"route": None, "reason": "attempt_reconciled", "error": None}


def _recovery_receipt(attempt: dict, worker: Optional[str], now: int) -> dict:
    """The final receipt for an attempt whose runner died mid-flight.

    Everything in it is observed from the store — the worker left no report, and
    inventing one would be the opposite of evidence.
    """
    return {
        "schema_version": 1,
        "kind": "jobs-attempt-final",
        "source": "claim_recovery",
        "job_id": attempt["job_id"],
        "attempt_id": attempt["id"],
        "ordinal": attempt["ordinal"],
        "specialist": attempt["specialist"],
        "repository": attempt["repository"],
        "status": "interrupted",
        "failure_class": "infrastructure",
        "previous_worker": worker,
        "recovered_at": now,
    }


def _recovery_narrative_locked(
    conn: sqlite3.Connection, attempt_id: str
) -> Optional[dict]:
    """The V3 stamp for a recovered attempt, or ``None`` if it owns no request.

    Recovery is the *only* writer of a final receipt for an attempt nobody was
    there to witness, so it is also the only chance to give that receipt the
    exact V3 shape a later replay demands. An attempt with no bound request, or
    one that never recorded an approved base, gets the plain legacy receipt —
    inventing V3 authority for history that never had it is the thing the
    migration is careful not to do.
    """
    row = conn.execute(
        "SELECT request_id FROM job_run_requests "
        "WHERE attempt_id = ? AND response IS NULL",
        (attempt_id,),
    ).fetchone()
    if row is None:
        return None
    return {"request_id": row["request_id"], **RECOVERED_NARRATIVE}


def recover_expired_claims(
    conn: sqlite3.Connection, *, now: int
) -> List[str]:
    """Reclaim every Job whose lease has expired. Deterministic + idempotent.

    For each expired claim, in its own transaction and oldest number first:
    finish any running attempt as ``interrupted``/``infrastructure``, clear the
    whole claim block, keep the Job ``working`` (never ``needs_you``), rewind
    the step to ``routing`` unless it is a genuine correction/review step, and
    append exactly one ``claim_recovered`` event. The Job's goal and identity
    are never touched.

    Idempotent by construction: recovery clears ``claim_token``, so a second run
    finds no expired claims and writes nothing. Returns the recovered Job ids.
    """
    now = int(now)
    candidates = conn.execute(
        "SELECT id FROM jobs WHERE claim_token IS NOT NULL "
        "AND lease_expires_at IS NOT NULL AND lease_expires_at < ? "
        "ORDER BY number ASC",
        (now,),
    ).fetchall()

    recovered: List[str] = []
    for c in candidates:
        jid = c["id"]
        with write_txn(conn):
            # Re-check inside the write lock: another writer may have released
            # or heartbeat-extended this claim between the select and here.
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
            if row is None or row["claim_token"] is None:
                continue
            if row["lease_expires_at"] is None or row["lease_expires_at"] >= now:
                continue

            running = conn.execute(
                "SELECT * FROM job_attempts WHERE job_id = ? AND status = 'running'",
                (jid,),
            ).fetchone()
            interrupted_id = None
            if running is not None:
                attempt = _attempt_to_dict(running)
                _finish_attempt_locked(
                    conn,
                    attempt,
                    status="interrupted",
                    failure_class="infrastructure",
                    now=now,
                    # Recovery makes an attempt terminal, so recovery owes it a
                    # final receipt. Without one the only attempts with no
                    # evidence would be exactly the ones nobody was there to
                    # witness. Authored here because the worker is gone.
                    receipt=_recovery_receipt(attempt, row["claimed_by"], now),
                    v3_narrative=_recovery_narrative_locked(conn, attempt["id"]),
                )
                interrupted_id = running["id"]

            new_step = (
                row["step"] if row["step"] in _RECOVER_PRESERVE_STEPS else "routing"
            )
            conn.execute(
                "UPDATE jobs SET status = 'working', step = ?, updated_at = ?, "
                "revision = revision + 1, claimed_by = NULL, claim_token = NULL, "
                "claim_acquired_at = NULL, lease_expires_at = NULL, "
                "current_attempt_id = NULL WHERE id = ?",
                (new_step, now, jid),
            )
            _append_event_locked(
                conn,
                jid,
                "claim_recovered",
                data={
                    "previous_worker": row["claimed_by"],
                    "interrupted_attempt_id": interrupted_id,
                    "step": new_step,
                },
                now=now,
            )
            recovered.append(jid)
    return recovered


# ---------------------------------------------------------------------------
# Idempotent source intake — scheduled jobs and later migration
# ---------------------------------------------------------------------------


@dataclass
class IntakeResult:
    """Outcome of :func:`create_or_get_job`.

    ``created`` is True only when this call inserted a new Job. ``conflict`` is
    True when the source key already existed but the incoming name/goal differ
    from the stored Job — the original is preserved and the caller is told so
    (an explicit conflict indicator rather than a silent rewrite).
    """

    job_id: str
    created: bool
    conflict: bool


def create_or_get_job(
    conn: sqlite3.Connection,
    *,
    source_type: str,
    source_key: str,
    name: str,
    goal: str,
    specialist: Optional[str] = None,
    routing_reason: Optional[str] = None,
    correlations: Optional[Iterable[str]] = None,
    now: Optional[int] = None,
) -> IntakeResult:
    """Create one Job for a source key, or return the existing one.

    ``(source_type, source_key)`` is globally unique, so replaying the same
    scheduled tick or migration row returns the original Job instead of
    duplicating it. A changed name/goal on replay never rewrites the original —
    it comes back as ``conflict=True``. Different source types may reuse a key.

    Check-and-create run in one IMMEDIATE transaction, so two racing intakes of
    the same key create exactly one Job (the second serializes behind the first
    and reads the now-existing source row); the ``job_sources`` primary key is
    the hard backstop.
    """
    source_type = str(source_type or "").strip()
    source_key = str(source_key or "").strip()
    if not source_type:
        raise ValueError("source_type must not be empty")
    if not source_key:
        raise ValueError("source_key must not be empty")
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("job name must not be empty")
    if goal is None or goal == "":
        raise ValueError("job goal must not be empty")
    now = _now() if now is None else int(now)

    with write_txn(conn):
        existing = conn.execute(
            "SELECT job_id FROM job_sources WHERE source_type = ? AND source_key = ?",
            (source_type, source_key),
        ).fetchone()
        if existing is not None:
            jid = existing["job_id"]
            job = get_job(conn, jid)
            conflict = job is not None and (
                job.name != clean_name or job.goal != goal
            )
            return IntakeResult(job_id=jid, created=False, conflict=conflict)

        jid = _create_job_locked(
            conn,
            name=clean_name,
            goal=goal,
            specialist=specialist,
            routing_reason=routing_reason,
            correlations=correlations,
            now=now,
        )
        conn.execute(
            "INSERT INTO job_sources (source_type, source_key, job_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (source_type, source_key, jid, now),
        )
    return IntakeResult(job_id=jid, created=True, conflict=False)
