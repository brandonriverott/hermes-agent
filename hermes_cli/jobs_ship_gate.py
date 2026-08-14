"""The Jobs completion gate — one truth table, vendored into both engines.

Two implementations write ``~/.hermes/jobs.db``: the out-of-tree plugin at
``~/.hermes/plugins/jobs/`` (which serves ``hermes jobs``) and the release
engine at ``hermes_cli/`` (which the gateway runs).  From 2026-08-09 to
2026-08-14 they disagreed about what ``complete`` means, and the ungated one
was the one running: 25 Jobs were stamped complete without a merged commit,
and the release engine relabelled a sealed verdict.

This module is the answer to "how do the two stay in agreement after the next
release cut".  It is the **only** place the truth table lives, and it is
vendored **byte-identical** into exactly two locations:

    ~/.hermes/plugins/jobs/jobs_ship_gate.py
    <release>/hermes_cli/jobs_ship_gate.py

``jobs-flow-selftest.sh`` compares the SHA-256 of those two files on every run
and fails loudly on divergence, then drives the same refusal battery through
both engines.  A release cut that carries a changed gate and leaves the plugin
behind (or the reverse) turns the selftest red within six hours.

Deliberately dependency-free: standard library only, no imports from either
engine, no logging, no I/O beyond ``git`` subprocesses and reads on the caller's
open sqlite3 connection.  That is what makes verbatim vendoring possible.

Every check returns a **refusal string or ``None``** rather than raising, so
each engine raises its own ``InvalidTransition`` and neither has to import the
other's exception type.  The strings are the operator-facing contract: they are
quoted in evidence bundles and asserted by the canaries, so treat a wording
change as a contract change.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from typing import Optional

# Bumped whenever the truth table itself changes. Both engines report it, and
# the dual-engine selftest asserts they agree — a mismatch means one side is
# running a stale vendored copy.
CONTRACT_VERSION = 1

# Ship evidence, not a note. Only the activation writer may mint this kind.
RESERVED_RECEIPT_KIND = "jobs-activation"

# Returned by :func:`seal_check` when the caller is re-recording the state the
# Job is already in. Not a refusal: a second ``jobs activate`` stays harmless.
SEAL_NOOP = "__seal_noop__"

_COMPLETE = ("finished", "complete")


# ── evidence helpers ─────────────────────────────────────────────────────────


def matching_activation_receipt(
    conn: sqlite3.Connection,
    job_id: str,
    attempt_id: str,
    commit_sha: str,
) -> Optional[dict]:
    """Return durable merged/deployed/live evidence for exactly ``commit_sha``.

    Attempt receipts prove a build happened.  They cannot prove activation: a
    local Git merge says nothing about deployment, live health, or whether a
    migration was applied.  The explicit ship step records that separate fact
    as a ``jobs-activation`` receipt.  Read newest-first so a corrected receipt
    can supersede an earlier incomplete one without mutating history.
    """
    rows = conn.execute(
        "SELECT attempt_id, data FROM job_receipts "
        "WHERE job_id = ? ORDER BY rowid DESC",
        (job_id,),
    ).fetchall()
    for row in rows:
        try:
            data = json.loads(row["data"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if data.get("kind") != RESERVED_RECEIPT_KIND:
            continue
        if data.get("schema_version") != 1:
            continue
        if str(row["attempt_id"] or "") != str(attempt_id):
            continue
        if str(data.get("job_id") or "") != str(job_id):
            continue
        if str(data.get("attempt_id") or "") != str(attempt_id):
            continue
        if str(data.get("commit") or "") != str(commit_sha):
            continue
        if not all(
            data.get(key) is True
            for key in ("merged", "deployed", "live_verified")
        ):
            continue
        if data.get("migrations") not in ("applied", "not_required"):
            continue
        return data
    return None


def commit_changes_require_migration(
    repository: str, base_commit: str, commit_sha: str
) -> Optional[bool]:
    """Whether the observed build diff contains SQL/migration files.

    ``None`` means the repository evidence is unavailable.  Completion still
    requires the activation receipt to state ``applied`` or ``not_required``;
    when Git evidence *is* available, a migration-bearing build may only use
    ``applied``.
    """
    if not repository or not os.path.isdir(repository):
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", repository, "diff", "--name-only", base_commit, commit_sha],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    files = [line.strip().lower() for line in proc.stdout.splitlines() if line.strip()]
    return any(
        path.endswith(".sql") or "/migrations/" in f"/{path}"
        for path in files
    )


def commit_is_on_main(repository: str, commit_sha: str) -> Optional[bool]:
    """Bounded local merge proof; ``None`` means proof was unavailable."""
    if not repository or not os.path.isdir(repository):
        return None
    try:
        has_main = subprocess.run(
            ["git", "-C", repository, "rev-parse", "--verify", "-q", "main"],
            capture_output=True,
            timeout=10,
        ).returncode == 0
        if not has_main:
            return None
        proc = subprocess.run(
            ["git", "-C", repository, "merge-base", "--is-ancestor", commit_sha, "main"],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def latest_attempt(conn: sqlite3.Connection, job_id: str):
    """The highest-ordinal attempt row, or ``None``.

    Highest ordinal, never "most recent success": an earlier success cannot
    authorize a Job whose latest run failed.
    """
    return conn.execute(
        "SELECT id, status, commit_sha, base_commit, repository "
        "FROM job_attempts WHERE job_id = ? "
        "ORDER BY ordinal DESC LIMIT 1",
        (job_id,),
    ).fetchone()


# ── the truth table ──────────────────────────────────────────────────────────


def complete_refusal(
    conn: sqlite3.Connection,
    job_id: str,
    label: str,
    number,
    *,
    is_on_main=None,
    requires_migration=None,
    activation_receipt=None,
) -> Optional[str]:
    """Why this Job may not move to finished/complete, or ``None`` to allow it.

    Brandon's 100% ship rule: complete == MERGED AND DEPLOYED AND LIVE.  Six
    checks, ANDed, every one fail-closed — unverifiable evidence refuses exactly
    like disproved evidence, because "the repo was missing" is not a merge proof.

    The three probes are injectable so each engine can pass its own
    module-level delegates.  That is not indirection for its own sake: the
    canaries prove the unverifiable-evidence branches by monkeypatching those
    names, and a gate whose fail-closed paths cannot be exercised by a test is
    a gate nobody is watching.
    """
    is_on_main = is_on_main or commit_is_on_main
    requires_migration = requires_migration or commit_changes_require_migration
    activation_receipt = activation_receipt or matching_activation_receipt
    ev = latest_attempt(conn, job_id)
    if ev is None:
        return (
            f"SHIP GATE: job {label} has no attempt, so nothing was "
            f"built, merged, deployed or verified live. complete "
            f"requires a succeeded, evidenced build followed by "
            f"activation; close unbuilt work as finished/failed."
        )
    if ev["status"] != "succeeded":
        return (
            f"SHIP GATE: job {label}'s latest attempt "
            f"{ev['id']} is {ev['status']}; an earlier success cannot "
            f"authorize complete. Resolve or rerun the latest attempt."
        )
    if not ev["commit_sha"]:
        return (
            f"SHIP GATE: job {label}'s succeeded attempt "
            f"{ev['id']} carries no commit evidence, so there is "
            f"nothing to prove merged, deployed or live."
        )
    repo = ev["repository"] or ""
    merged = is_on_main(str(repo), str(ev["commit_sha"]))
    if merged is None:
        return (
            f"SHIP GATE: cannot verify that job {label} commit "
            f"{ev['commit_sha'][:12]} is on main in {repo or '(missing repo)'}."
        )
    if not merged:
        return (
            f"SHIP GATE: job {label} commit "
            f"{ev['commit_sha'][:9]} is not on main in {repo}. "
            f"complete means merged, deployed, and live — merge "
            f"it first, then record activation evidence."
        )
    activation = activation_receipt(
        conn, job_id, ev["id"], ev["commit_sha"]
    )
    if activation is None:
        return (
            f"SHIP GATE: job {label} needs a matching "
            f"jobs-activation receipt before complete. Do not "
            f"hand-write it — run: hermes jobs activate "
            f"{number} --migrations applied|not_required. "
            f"The receipt it writes is exactly: "
            f'{{"schema_version": 1, "kind": "jobs-activation", '
            f'"job_id": "{job_id}", "attempt_id": "{ev["id"]}", '
            f'"commit": "{ev["commit_sha"]}", "merged": true, '
            f'"deployed": true, "live_verified": true, '
            f'"migrations": "applied|not_required"}} '
            f"bound to attempt {ev['id']}. Every one of those "
            f"keys is checked; omitting schema_version or job_id "
            f"is the usual reason this refusal repeats."
        )
    needs_migration = requires_migration(
        str(ev["repository"] or ""),
        str(ev["base_commit"] or ""),
        str(ev["commit_sha"]),
    )
    if needs_migration is None:
        return (
            f"SHIP GATE: cannot verify job {label}'s migration "
            f"diff for commit {ev['commit_sha'][:12]}; refusing complete."
        )
    if needs_migration is True and activation.get("migrations") != "applied":
        return (
            f"SHIP GATE: job {label} changes SQL/migration files; "
            f"its matching activation receipt must say "
            f"migrations='applied' before complete."
        )
    return None


def seal_check(
    job_status: str, job_step: str, label: str, *, status: str, step: str
) -> Optional[str]:
    """Guard a *shipped* verdict against rewriting.

    ``None`` allows the move, :data:`SEAL_NOOP` means "already there, do
    nothing", anything else is the refusal text.

    Only ``complete`` is sealed, because it is the only step that asserts
    something about the world.  The gate is consulted on the way *in*; without
    this, nothing guarded the way *out*, and a shipped Job could be quietly
    rewritten to finished/failed — or, as the release engine actually did on a
    scratch reproduction, to finished/cancelled.
    """
    if (job_status, job_step) != _COMPLETE:
        return None
    if (status, step) == _COMPLETE:
        return SEAL_NOOP
    return (
        f"job {label} is complete, and a shipped Job's verdict is "
        f"sealed: it cannot be rewritten to {status}/{step}"
    )


def step_refusal(job_status: str, job_step: str, label: str, step: str) -> Optional[str]:
    """Guard ``set_step`` in both directions. ``None`` allows the move.

    ``set_step`` never consults the ship gate, so it must not be able to reach
    ``complete`` at all — that was the back door *in* (a zero-attempt Job could
    be stepped straight to complete and then sealed itself).  ``transition`` is
    the only door, and it is gated.  The way *out* is the same seal as
    :func:`seal_check`.
    """
    if step == "complete":
        return (
            f"job {label}: set_step cannot reach 'complete' — it never "
            f"consults the ship gate. Use: hermes jobs activate "
            f"<job> --migrations applied|not_required, which records the "
            f"merge, deployment and migration evidence and transitions in "
            f"one operation."
        )
    if (job_status, job_step) == _COMPLETE:
        return (
            f"job {label} is complete, and a shipped Job's verdict is "
            f"sealed: its step cannot be rewritten to {step!r}"
        )
    return None


def settlement_outcome(outcome: tuple) -> tuple:
    """Settlement never grants complete. Park a would-be ship for the operator.

    Brandon's 100% ship rule, settlement edition: a succeeded attempt proves a
    build, never a deployment.  Any outcome that would be (finished, complete)
    becomes needs_you/waiting_for_decision so the alert relay puts the
    merge/deploy question in front of a human.  Only the explicit activation
    step — which passes :func:`complete_refusal` — moves a Job to complete.

    Deterministic on ``outcome`` alone, so terminal replay equivalence holds.
    """
    if tuple(outcome) != _COMPLETE:
        return outcome
    return ("needs_you", "waiting_for_decision")


def reserved_kind_refusal(data) -> Optional[str]:
    """Refuse a hand-written ``jobs-activation`` receipt. ``None`` allows it.

    The gate trusts any receipt whose fields match, and the general receipt
    doors accept arbitrary JSON — so "deployed, live_verified" was a thing any
    caller could simply assert about a build it never deployed.  Both the
    ad-hoc door (``add_receipt``) and the settlement door
    (``_add_final_receipt_locked``) must call this; only the activation writer
    is exempt.
    """
    if isinstance(data, dict) and data.get("kind") == RESERVED_RECEIPT_KIND:
        return (
            "the 'jobs-activation' receipt kind is reserved: it is ship "
            "evidence, not a note. Run 'hermes jobs activate <job> "
            "--migrations applied|not_required' so the merge, deployment and "
            "migration checks and the receipt are one operation."
        )
    return None
