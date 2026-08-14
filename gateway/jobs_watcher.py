"""Jobs wake watcher — post a finished job's verdict back into its origin session.

Modeled directly on :mod:`gateway.kanban_watchers`: a polling background loop
that does all its SQLite work inside ``asyncio.to_thread`` (so the WAL lock
never blocks the event loop), resolves the owning adapter through
``_authorization_adapter`` (the same chokepoint the authorization path uses, so
a secondary profile's job is never announced by the default profile's bot), and
delivers through :func:`gateway.wake.deliver_wake` — which already covers both
push-capable adapters and the stateless API server that backs the desktop UI.

The Jobs runner itself is out-of-tree (``~/.hermes/plugins/jobs``) and owns
``jobs.db``; this module only ever READS it. Column names are discovered with
``PRAGMA table_info`` and every field is optional, so a schema tweak on the
runner side degrades to "that field is missing from the message" instead of
crashing the watcher.

Delivery bookkeeping lives in a SEPARATE database (``jobs_wake.db``, next to
``jobs.db``) so we never write to the runner's schema. A job id is CLAIMED with
``INSERT OR IGNORE`` *before* delivery: the claim is the dedup mechanism, so
exactly one gateway announces a job, a restart never re-announces, and a failed
delivery releases the claim so a later tick retries.

That same sidecar holds the origin envelopes recorded at create time (see
``plugins/jobs-origin``) and a one-shot SEED marker: on a ledger that has never
been seeded, every already-terminal job is marked delivered instead of
announced, so shipping this watcher onto a machine with months of finished jobs
does not replay that history into the chats that created them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("gateway.run")

# Job states that mean "the runner is done with this job and a human should
# hear about it". Anything else is still in flight.
TERMINAL_STATES = ("finished", "needs_you")

# Key the origin envelope is filed under inside the job's ``correlations``
# blob. Written by the out-of-tree ``jobs_create`` handler via
# :func:`capture_job_origin`, read back by the watcher.
ORIGIN_KEY = "hermes_origin"

# Where a Job with no originating chat is announced. Cron rotations and CLI
# runs never bind a conversation; this is the one surface that exists to
# receive them, so they surface instead of disappearing. Not a guess at a
# human chat — see :func:`_unrouted_feed_origin`.
UNROUTED_FEED_SESSION_ID = "build_feed_live"

# Candidate column names, in priority order. The Jobs runner is out-of-tree, so
# we probe rather than assume; a missing field is simply left out of the
# announcement.
_STATE_COLUMNS = ("status", "state")
_STEP_COLUMNS = ("step",)
_ORIGIN_COLUMNS = ("origin", "correlations", "metadata", "meta")
_NAME_COLUMNS = ("name", "title")
_OUTCOME_COLUMNS = ("outcome", "result")
_VERDICT_COLUMNS = ("review_verdict", "verdict")
_ROUNDS_COLUMNS = ("rounds", "round_count", "review_rounds")
_FILES_COLUMNS = ("changed_files", "files_changed")
# JSON blobs to look inside when the field has no column of its own.
_BLOB_COLUMNS = ("result", "summary", "report", "correlations", "metadata", "meta")

# How many changed-file names to name explicitly before eliding the rest.
# Three is enough to recognise what was touched. A 15-path wall is exactly the
# "random coding shit" Brandon asked to keep out of the handoff; any SQL file
# that actually needs running is named in the numbered steps, not down here.
_MAX_LISTED_FILES = 3

# Ceiling on any single probed text field in the announcement. The runner is
# out-of-tree, so a column we *hope* holds a one-word outcome may hold a whole
# report — ``result`` is both an outcome candidate and a blob candidate.
_MAX_FIELD_CHARS = 160

# Marker row in the sidecar's ``meta`` table recording that the ledger has been
# seeded once. Its absence — not an empty ``delivered`` table — is what means
# "first run".
_SEEDED_KEY = "ledger_seeded_at"


def jobs_db_path() -> Path:
    """Path of the Jobs runner database (``~/.hermes/jobs.db``).

    ``HERMES_JOBS_DB`` overrides it — same escape hatch shape as
    ``HERMES_KANBAN_DB``, and what the tests point at a fixture.
    """
    override = os.environ.get("HERMES_JOBS_DB", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "jobs.db"


def wake_db_path() -> Path:
    """Path of our own delivery-ledger database, a sibling of ``jobs.db``."""
    return jobs_db_path().with_name("jobs_wake.db")


def capture_job_origin() -> Optional[Dict[str, Any]]:
    """Snapshot the originating session identity for a job being created.

    Called at job-create time by ``plugins/jobs-origin`` (which files the result
    with :func:`record_job_origin`, keyed by the new job id, without touching
    the runner's schema), and equally usable from the out-of-tree
    ``jobs_create`` handler itself for installs that would rather write it into
    the job's ``correlations`` under :data:`ORIGIN_KEY`. The watcher reads
    either. Returns ``None`` outside a gateway session (TUI/CLI runs bind no
    session platform), and a job with no origin is skipped silently.

    Mirrors ``tools/cronjob_tools.py::_origin_from_env`` (the cron
    ``deliver="origin"`` capture) field for field, with one addition: the raw
    ``session_id`` needed by :func:`gateway.wake.deliver_wake` to self-post into
    an API-server session. That id is read via
    ``tools.async_delegation._current_origin_session_id`` rather than
    ``HERMES_SESSION_ID`` directly, because constructing a child agent clobbers
    ``HERMES_SESSION_ID`` with the subagent's internal id — waking that would
    land in a session nobody is reading.
    """
    from gateway.session_context import get_session_env

    platform = get_session_env("HERMES_SESSION_PLATFORM")
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID")

    try:
        from tools.async_delegation import _current_origin_session_id

        session_id = _current_origin_session_id() or ""
    except Exception:  # pragma: no cover - defensive; helper is in-tree
        session_id = ""
    if not session_id:
        session_id = get_session_env("HERMES_SESSION_ID") or ""

    # Local Desktop/TUI sessions intentionally do not bind a messaging
    # platform/chat tuple.  Their durable session id is the exact return
    # address consumed by the API-server/Desktop delivery path.
    source = (get_session_env("HERMES_SESSION_SOURCE") or "").lower()
    if not session_id:
        session_id = get_session_env("HERMES_SESSION_KEY") or ""
    if not (platform and chat_id) and source in {"desktop", "tui", "api_server"}:
        platform = "api_server"
        chat_id = session_id
    if not (platform and chat_id and session_id):
        return None

    return {
        "platform": platform,
        "chat_id": chat_id,
        "session_id": session_id,
        "chat_type": get_session_env("HERMES_SESSION_CHAT_TYPE") or "",
        "thread_id": get_session_env("HERMES_SESSION_THREAD_ID") or "",
        "user_id": get_session_env("HERMES_SESSION_USER_ID") or "",
        "profile": get_session_env("HERMES_SESSION_PROFILE") or "",
    }


def _first_present(columns: set, candidates) -> str:
    for name in candidates:
        if name in columns:
            return name
    return ""


def _as_json_dict(raw: Any) -> Dict[str, Any]:
    """Best-effort decode of a JSON-object column. Non-objects yield ``{}``."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return {}
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _scalar(value: Any, limit: int = _MAX_FIELD_CHARS) -> Optional[str]:
    """Coerce a probed field to a short single-value string, or ``None``.

    A structured value is REJECTED rather than stringified: the module's
    promise is that a schema difference degrades to "that field is missing
    from the message", not "the whole JSON report lands on the header line".
    Anything else is truncated to ``limit``.
    """
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text[0] in "{[":
        try:
            if isinstance(json.loads(text), (dict, list)):
                return None
        except (ValueError, TypeError):
            pass  # Not JSON after all — a message that merely starts with "{".
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _valid_origin(candidate: Any) -> Optional[Dict[str, Any]]:
    """An origin envelope is usable only with a platform AND a chat id."""
    if not isinstance(candidate, dict):
        return None
    platform = str(candidate.get("platform") or "").strip()
    chat_id = str(candidate.get("chat_id") or "").strip()
    return candidate if (platform and chat_id) else None


def _extract_origin(row: sqlite3.Row, columns: set) -> Optional[Dict[str, Any]]:
    """Pull the origin envelope out of whichever column carries it.

    Accepts both a dedicated ``origin`` column holding the envelope directly and
    a ``correlations``/``metadata`` blob holding it under :data:`ORIGIN_KEY`.
    Returns ``None`` when no column carries one — the caller then falls back to
    the sidecar record written at create time.
    """
    for name in _ORIGIN_COLUMNS:
        if name not in columns:
            continue
        blob = _as_json_dict(row[name])
        candidate = blob.get(ORIGIN_KEY)
        if not isinstance(candidate, dict) and name == "origin":
            # A dedicated `origin` column stores the envelope unwrapped.
            candidate = blob
        found = _valid_origin(candidate)
        if found:
            return found
    return None


def _field(row: sqlite3.Row, columns: set, candidates, blobs: Dict[str, Any]) -> Any:
    """Read a field from its own column, falling back to the JSON blobs."""
    name = _first_present(columns, candidates)
    if name:
        value = row[name]
        if value not in (None, ""):
            return value
    for key in candidates:
        for blob in blobs.values():
            value = blob.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _file_list(value: Any) -> List[str]:
    """Normalise a changed-files field (JSON list, or a delimited string)."""
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        separator = "\n" if "\n" in text else ","
        return [part.strip() for part in text.split(separator) if part.strip()]
    return []


def _tests_are_clean(tests: Any) -> bool:
    """Accept only the bridge's explicit ``command -> pass`` evidence shape."""
    if not isinstance(tests, list) or not tests:
        return False
    return all(
        isinstance(test, str)
        and re.search(r"\s->\s*pass\s*$", test, flags=re.IGNORECASE)
        for test in tests
    )


def _latest_by_job(rows, *, ordinal_key: str) -> Dict[str, sqlite3.Row]:
    latest: Dict[str, sqlite3.Row] = {}
    for row in rows:
        job_id = str(row["job_id"])
        prior = latest.get(job_id)
        if prior is None or int(row[ordinal_key] or 0) > int(prior[ordinal_key] or 0):
            latest[job_id] = row
    return latest


def _reported_list(report: Dict[str, Any], key: str) -> List[str]:
    value = report.get(key)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def _git_files(repository: Any, base: Any, commit: Any) -> List[str]:
    repo = str(repository or "").strip()
    base_sha = str(base or "").strip()
    commit_sha = str(commit or "").strip()
    if not (repo and base_sha and commit_sha and Path(repo).is_dir()):
        return []
    try:
        proc = subprocess.run(
            ["git", "-C", repo, "diff", "--name-only", base_sha, commit_sha],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return proc.stdout.splitlines() if proc.returncode == 0 else []


def _merged_into_main(repository: Any, commit: Any) -> Optional[bool]:
    repo = str(repository or "").strip()
    commit_sha = str(commit or "").strip()
    if not (repo and commit_sha and Path(repo).is_dir()):
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", repo, "merge-base", "--is-ancestor", commit_sha, "main"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def _migration_handoff(
    repository: Any, commit: Any, files: List[str]
) -> tuple[List[str], List[str], List[str]]:
    """Classify migration/rollback files and read bounded operator comments."""
    candidates = [
        str(path)
        for path in files
        if str(path).lower().endswith(".sql")
        or "/migrations/" in f"/{str(path).lower()}"
    ]
    rollbacks = [
        path for path in candidates
        if "rollback" in Path(path).name.lower()
    ]
    migrations = [path for path in candidates if path not in rollbacks]

    repo = str(repository or "").strip()
    sha = str(commit or "").strip()
    if not (repo and sha and Path(repo).is_dir()):
        return migrations, rollbacks, []

    instructions: List[str] = []
    keywords = (
        "gated", "approval", "run once", "non-pooler", "never ",
        "do not ", "production", "live ",
    )
    for path in migrations:
        try:
            proc = subprocess.run(
                ["git", "-C", repo, "show", f"{sha}:{path}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0:
            continue
        for raw in proc.stdout.splitlines()[:30]:
            stripped = raw.strip()
            if not stripped.startswith("--"):
                if stripped:
                    break
                continue
            comment = stripped[2:].strip()
            if comment and any(word in comment.lower() for word in keywords):
                text = _scalar(comment, 400)
                if text and text not in instructions:
                    instructions.append(text)
            if len(instructions) >= 4:
                break
    return migrations, rollbacks, instructions


def _append_desktop_message(
    session_id: str,
    message: str,
    *,
    state_db: Optional[Path] = None,
    job_id: str = "",
    delivery_key: str = "",
) -> str:
    """Append one passive assistant handoff to an exact desktop session.

    A terminal Job already has a deterministic action packet.  Starting a new
    agent turn merely to paraphrase it adds latency and created duplicate user
    prompts when the HTTP self-post timed out after committing.  One SQLite
    SessionDB enforces transcript/compression guards.  If compression already
    rotated the originating chat, follow only its unique live continuation;
    intentionally closed or ambiguous sessions fail closed and are retried.
    """
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    session_id = str(session_id or "").strip()
    if not session_id:
        raise ValueError("desktop Jobs handoff has no origin session id")
    db_path = Path(state_db) if state_db is not None else get_hermes_home() / "state.db"
    state = SessionDB(db_path=db_path)
    display_metadata = {
        "source": "jobs",
        "job_id": str(job_id or ""),
        "delivery_key": str(delivery_key or ""),
        "notification_kind": "terminal",
    }
    try:
        tip_append = getattr(state, "append_message_to_compression_tip", None)
        if tip_append is not None:
            _row_id, target = tip_append(
                session_id,
                "assistant",
                message,
                timestamp=time.time(),
                display_kind="job_notification",
                display_metadata=display_metadata,
            )
            return target
        # 2026-08-14: this module evolved against the mutable checkout, whose
        # SessionDB grew ``append_message_to_compression_tip``. The release
        # branch's SessionDB has not; committing the watcher into a release
        # (which is what keeps it alive across cuts) therefore met an engine
        # missing that method, and every delivery raised AttributeError. Do the
        # same job with the primitives both trees do have: follow the
        # compression lineage, refuse anything not live, append once.
        #
        # The narrow race the richer method retries — compression rotating the
        # row between resolution and append — is not retried here. It does not
        # need to be: the caller releases the delivery claim on any exception
        # and the next tick, ten seconds later, resolves the new tip.
        target = state.get_compression_tip(session_id) or session_id
        row = state.get_session(target)
        if row is None:
            raise ValueError(f"origin session does not exist: {session_id}")
        if row.get("ended_at") is not None:
            raise ValueError(
                f"origin session is not live: {target} "
                f"({row.get('end_reason') or 'ended'})"
            )
        state.append_message(
            target,
            "assistant",
            message,
            timestamp=time.time(),
            display_kind="job_notification",
            display_metadata=display_metadata,
        )
        return target
    finally:
        state.close()


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _plain_check_summary(tests: List[str], review: str) -> str:
    """What the tests and the reviewer said, in words Brandon reads once."""
    parts = []
    if tests:
        if _tests_are_clean(tests):
            parts.append(f"{_count(len(tests), 'check')} ran and passed")
        else:
            parts.append(
                f"{_count(len(tests), 'check')} ran, but not all of them passed"
            )
    else:
        parts.append("no test results came back")
    if review == "PASS":
        parts.append("the reviewer approved it")
    elif review:
        parts.append("the reviewer did not approve it")
    else:
        parts.append("the reviewer never signed off")
    return ", and ".join(parts)


# Failure classes are internal vocabulary. Brandon gets the situation instead.
_PLAIN_FAILURE = {
    "authentication": (
        "the builder got logged out partway through",
        'Log the builder back in, then reply here: "Retry Job #{n}."',
    ),
    "provider_billing": (
        "the AI provider cut it off over billing or usage limits",
        'Sort out the billing or limit, then reply here: "Retry Job #{n}." '
        "I won't retry this one on my own.",
    ),
    "infrastructure": (
        "the build machine wasn't reachable",
        'Get the build machine back up, then reply here: "Retry Job #{n}."',
    ),
    "max_turns": (
        "it ran out of room before it could finish the work",
        'Reply here: "Split Job #{n} into smaller pieces and retry."',
    ),
    "reviewer_rejection": (
        "the reviewer turned the work down",
        'Reply here: "Show me why the reviewer rejected Job #{n}."',
    ),
    "irreversible_action": (
        "finishing it would have needed a permanent change nobody approved",
        "Tell me yes or no on that change. I won't retry Job #{n} on my own.",
    ),
    "implementation": (
        "the build itself broke",
        'Reply here: "Show me what broke on Job #{n}, then retry."',
    ),
}


def format_job_wake(job: Dict[str, Any]) -> str:
    """Render a truthful, actionable terminal handoff for one Job."""
    state = str(job.get("state") or "").strip()
    step = str(job.get("step") or "").strip()
    attempt_status = str(job.get("attempt_status") or "").strip()
    successful = attempt_status == "succeeded"
    review = str(job.get("verdict") or "").strip()
    tests = job.get("tests") or []
    verified = successful and review == "PASS" and _tests_are_clean(tests)
    activation = job.get("activation")
    activation_evidence = (
        isinstance(activation, dict)
        and activation.get("merged") is True
        and activation.get("deployed") is True
        and activation.get("live_verified") is True
        and activation.get("migrations") in ("applied", "not_required")
    )
    activated = (
        state == "finished" and step == "complete" and activation_evidence
    )
    name = str(job.get("name") or job.get("id") or "?")
    number = job.get("number")
    ident = f"#{number}" if number not in (None, "") else str(job.get("id") or "?")
    merged = job.get("merged")
    migrations = job.get("migrations") or []
    rollbacks = job.get("rollbacks") or []
    instructions = job.get("migration_instructions") or []

    lines = [f"Job {ident} — {name}"]
    if job.get("unrouted"):
        lines.append(
            "(This job wasn't started from a chat, so it's reporting here.)"
        )

    # ── What happened ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("What happened:")
    summary = str(job.get("summary") or "").strip()
    if successful:
        opening = "I built it"
        if summary:
            opening = f"I built it — {summary.rstrip('.')}"
        if activated:
            # It's live and you confirmed it. Leading with a missing test
            # record here reads as doubt about something already working.
            lines.append(f"{opening}, and it's been shipped.")
            if not (tests and review):
                lines.append(
                    "(I don't have the test or review record for this one — "
                    "it was built before those were kept.)"
                )
        else:
            lines.append(
                f"{opening}. Then {_plain_check_summary(tests, review)}."
            )
    else:
        reason, _action = _PLAIN_FAILURE.get(
            str(job.get("failure_class") or ""),
            ("it didn't finish", ""),
        )
        lines.append(f"It didn't get built — {reason}.")

    # ── Where it stands ────────────────────────────────────────────────────
    lines.append("")
    lines.append("Where it stands:")
    if activated:
        db_text = (
            "The database change has been applied."
            if activation.get("migrations") == "applied"
            else "No database change was needed."
        )
        lines.append(f"It's merged, deployed, and running live. {db_text}")
    elif state == "finished" and step == "complete":
        # Local-main ancestry is the only thing the gate proves here. Calling
        # that "live" is the false promise that started all of this.
        lines.append(
            "The code is merged, but I can't confirm it actually deployed or "
            "that it's working live."
        )
    elif not successful:
        lines.append("Nothing was merged or deployed. Your app is untouched.")
    else:
        merge_text = (
            "The code is merged"
            if merged is True
            else "Nothing has been merged"
        )
        db_text = (
            " The database change has NOT been run yet."
            if migrations
            else " No database change is needed."
        )
        lines.append(
            f"{merge_text}, but it is not deployed and not confirmed live.{db_text}"
        )

    # ── What you need to do ────────────────────────────────────────────────
    lines.append("")
    lines.append("What you need to do:")
    if activated:
        lines.append("Nothing — this one is done. Just keep an eye on it.")
    elif not successful:
        _reason, action = _PLAIN_FAILURE.get(
            str(job.get("failure_class") or ""),
            ("", f'Reply here: "Show me what happened to Job #{number}."'),
        )
        lines.append("Don't merge or deploy this one.")
        lines.append(action.format(n=number))
    elif not verified:
        lines.append(
            "Don't merge, deploy, or run any SQL for this one yet — I don't "
            "have proof it's safe."
        )
        lines.append(
            f'Reply here: "Recheck the tests and review for Job #{number}."'
        )
    else:
        steps: List[str] = []
        if merged is not True:
            steps.append(f'Reply here: "Merge Job #{number} only."')
        for path in migrations:
            step_text = f"Run this one database change yourself: {path}"
            if instructions:
                step_text += " — " + " ".join(instructions)
            steps.append(step_text)
        if state == "finished" and step == "complete":
            steps.append(
                f'Reply here: "Confirm Job #{number} deployed and check it live."'
            )
        else:
            steps.append(f'Reply here: "Deploy and live-verify Job #{number}."')
        for index, text in enumerate(steps, start=1):
            lines.append(f"{index}. {text}")
        if rollbacks:
            lines.append(
                "Ignore " + ", ".join(rollbacks)
                + " — that's the undo file, only if we have to undo this."
            )

    # ── Details (everything technical lives below this line) ───────────────
    detail: List[str] = []
    branch = str(job.get("branch") or "").strip()
    commit = str(job.get("commit") or "").strip()
    if branch or commit:
        ref = branch or "(branch unavailable)"
        if commit:
            ref += f" @ {commit[:12]}"
        detail.append(f"branch {ref}")
    repository = str(job.get("repository") or "").strip()
    if repository:
        detail.append(f"repo {repository}")
    files = job.get("files") or []
    if files:
        shown = ", ".join(files[:_MAX_LISTED_FILES])
        extra = len(files) - _MAX_LISTED_FILES
        if extra > 0:
            shown = f"{shown}, +{extra} more"
        detail.append(f"{_count(len(files), 'file')} changed: {shown}")
    if tests:
        detail.append("checks: " + "; ".join(str(test) for test in tests[:3]))
    if review:
        # Keep the reviewer's own verdict on the record even when the summary
        # above no longer repeats it (a shipped job leads with "shipped").
        detail.append(
            "review: the reviewer approved it" if review == "PASS"
            else f"review: {review}"
        )
    rounds = job.get("rounds")
    if rounds not in (None, ""):
        detail.append(f"review rounds: {rounds}")
    if detail:
        lines.append("")
        lines.append("Details:")
        lines.extend(detail)

    return "\n".join(lines)


def _ensure_wake_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS delivered ("
        " job_id TEXT PRIMARY KEY,"
        " delivered_at REAL NOT NULL)"
    )
    delivered_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(delivered)")
    }
    if "delivery_episode" not in delivered_columns:
        conn.execute("ALTER TABLE delivered ADD COLUMN delivery_episode TEXT")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta ("
        " key TEXT PRIMARY KEY,"
        " value TEXT NOT NULL)"
    )
    # Origin envelopes captured at job-create time. Written here rather than
    # into jobs.db so the runner's schema is never touched and a Jobs plugin
    # upgrade cannot clobber the record.
    # ponytail: grows one small row per created job and is never pruned; add a
    # retention sweep if a machine ever creates jobs fast enough to notice.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS origins ("
        " job_id TEXT PRIMARY KEY,"
        " origin TEXT NOT NULL,"
        " created_at REAL NOT NULL)"
    )
    conn.commit()


def _open_wake_db() -> sqlite3.Connection:
    path = wake_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    _ensure_wake_db(conn)
    return conn


def claim_job(job_id: str, delivery_episode: str) -> bool:
    """Durably claim one terminal episode. ``True`` only for the winner.

    The job id remains the primary key, while an atomic conditional upsert lets
    a later retry/terminal episode replace the earlier one exactly once.  A
    plain job-id dedup suppressed every notification after a Job's first
    ``needs_you`` state.
    """
    conn = _open_wake_db()
    try:
        cur = conn.execute(
            "INSERT INTO delivered (job_id, delivered_at, delivery_episode) "
            "VALUES (?, ?, ?) ON CONFLICT(job_id) DO UPDATE SET "
            "delivered_at=excluded.delivered_at, "
            "delivery_episode=excluded.delivery_episode "
            "WHERE COALESCE(delivered.delivery_episode, '') "
            "<> excluded.delivery_episode",
            (job_id, time.time(), str(delivery_episode)),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def record_job_origin(job_id: str, origin: Dict[str, Any]) -> None:
    """File a job's origin envelope in the sidecar, keyed by job id.

    Called at create time by ``plugins/jobs-origin`` with the dict returned by
    :func:`capture_job_origin`. Ignores anything without a platform and a chat
    id — an unusable envelope is the same as no origin, and the watcher skips
    origin-less jobs silently.
    """
    if not job_id or _valid_origin(origin) is None:
        return
    conn = _open_wake_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO origins (job_id, origin, created_at)"
            " VALUES (?, ?, ?)",
            (str(job_id), json.dumps(origin), time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def release_job(job_id: str, delivery_episode: str) -> None:
    """Undo this episode's failed claim without deleting a newer claim."""
    conn = _open_wake_db()
    try:
        conn.execute(
            "DELETE FROM delivered WHERE job_id = ? AND delivery_episode = ?",
            (job_id, str(delivery_episode)),
        )
        conn.commit()
    finally:
        conn.close()


def _pending(terminal_episodes: Dict[str, str]):
    """Seed-or-filter, in ONE wake-db round trip per tick.

    Returns ``(fresh_ids, origins_by_id)``:

    * first run ever (no :data:`_SEEDED_KEY` marker) — every currently-terminal
      id is written to ``delivered`` and nothing is returned, so deploying the
      watcher onto an established machine announces nothing retroactively;
    * afterwards — the delivered set is read once and the ids are filtered in
      memory, so the expensive per-job work (``SELECT *``, JSON decoding, and a
      ``claim_job`` connection each) is paid only for genuinely new jobs
      instead of for every job ever finished, every tick.
    """
    conn = _open_wake_db()
    try:
        seeded = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (_SEEDED_KEY,)
        ).fetchone()
        if seeded is None:
            now = time.time()
            conn.executemany(
                "INSERT INTO delivered "
                "(job_id, delivered_at, delivery_episode) VALUES (?, ?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET "
                "delivered_at=excluded.delivered_at, "
                "delivery_episode=excluded.delivery_episode",
                [
                    (job_id, now, episode)
                    for job_id, episode in terminal_episodes.items()
                ],
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (_SEEDED_KEY, repr(now)),
            )
            conn.commit()
            if terminal_episodes:
                logger.info(
                    "jobs watcher: first run — seeded %d already-terminal job(s) "
                    "as delivered; only jobs finishing from now on are announced",
                    len(terminal_episodes),
                )
            return [], {}

        delivered = {
            str(row[0]): (None if row[1] is None else str(row[1]))
            for row in conn.execute(
                "SELECT job_id, delivery_episode FROM delivered"
            )
        }
        # Schema upgrade from the old job-id-only ledger: those rows already
        # mean "this historical terminal state was delivered".  Bind them to
        # the currently observed episode without returning them, otherwise the
        # first restart after ALTER would replay every old terminal Job.
        legacy = [
            (episode, job_id)
            for job_id, episode in terminal_episodes.items()
            if job_id in delivered and not delivered[job_id]
        ]
        if legacy:
            conn.executemany(
                "UPDATE delivered SET delivery_episode = ? "
                "WHERE job_id = ? AND delivery_episode IS NULL",
                legacy,
            )
            conn.commit()
            for episode, job_id in legacy:
                delivered[job_id] = episode
        fresh = [
            job_id
            for job_id, episode in terminal_episodes.items()
            if delivered.get(job_id) != episode
        ]
        if not fresh:
            return [], {}
        placeholders = ",".join("?" for _ in fresh)
        origins: Dict[str, Dict[str, Any]] = {}
        for row in conn.execute(
            f"SELECT job_id, origin FROM origins WHERE job_id IN ({placeholders})",
            fresh,
        ):
            candidate = _valid_origin(_as_json_dict(row[1]))
            if candidate:
                origins[str(row[0])] = candidate
        return fresh, origins
    finally:
        conn.close()


def _unrouted_feed_origin() -> Optional[Dict[str, Any]]:
    """The dedicated Jobs feed, for jobs that were never born in a chat.

    Cron rotations (``agent-training-rotation.sh``) and CLI-queued work carry
    no conversation to answer. Dropping them silently is the original bug in a
    new costume: Brandon still finds out nothing. Announcing them in the
    newest desktop chat is the opposite failure — a result disclosed into an
    unrelated conversation. The Build Feed is neither: a purpose-built surface
    that exists only to receive this. Missing feed => the old skip-silently
    behavior stands, because there is still nowhere honest to report.
    """
    from hermes_constants import get_hermes_home

    db = get_hermes_home() / "state.db"
    if not db.exists():
        return None
    from hermes_state import SessionDB

    try:
        state = SessionDB(db_path=db)
    except Exception:
        return None
    try:
        # get_compression_tip echoes an unknown id back, so existence is a
        # separate question from "which row in the chain is current".
        if state.get_session(UNROUTED_FEED_SESSION_ID) is None:
            return None
        tip = state.get_compression_tip(UNROUTED_FEED_SESSION_ID)
        tip_row = state.get_session(tip) if tip else None
    except Exception:
        return None
    finally:
        state.close()
    # A closed feed is not a mailbox: appending would raise every tick.
    if not tip_row or tip_row.get("ended_at") is not None:
        return None
    return {
        "platform": "api_server",
        "chat_id": UNROUTED_FEED_SESSION_ID,
        "session_id": UNROUTED_FEED_SESSION_ID,
        "chat_type": "", "thread_id": "", "user_id": "", "profile": "",
    }


def collect_terminal_jobs() -> List[Dict[str, Any]]:
    """Read every unannounced terminal job that carries an origin.

    Runs in a worker thread (``asyncio.to_thread``). Returns normalised dicts;
    jobs without an origin are dropped here, silently, and are never claimed —
    the whole point being that a job created outside a gateway session has
    nowhere to be announced.
    """
    path = jobs_db_path()
    if not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    latest_attempts: Dict[str, sqlite3.Row] = {}
    receipts_by_job: Dict[str, List[sqlite3.Row]] = {}
    try:
        columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(jobs)")
        }
        if not columns:
            return []
        state_column = _first_present(columns, _STATE_COLUMNS)
        if not state_column or "id" not in columns:
            logger.debug(
                "jobs watcher: %s has no usable jobs schema (columns=%s)",
                path, sorted(columns),
            )
            return []
        placeholders = ",".join("?" for _ in TERMINAL_STATES)
        # Scalar id sweep first. The steady-state tick then costs one id column
        # over the terminal rows — not SELECT * plus a JSON decode of up to six
        # blobs on every job the runner has ever finished.
        episode_columns = ["id", state_column]
        if "step" in columns:
            episode_columns.append("step")
        if "revision" in columns:
            episode_columns.append("revision")
        if "updated_at" in columns:
            episode_columns.append("updated_at")
        terminal_rows = conn.execute(
            f"SELECT {', '.join(episode_columns)} FROM jobs "
            f"WHERE {state_column} IN ({placeholders})",
            TERMINAL_STATES,
        ).fetchall()
        terminal_ids = [str(row["id"]) for row in terminal_rows]

        tables = {
            str(table_row[0])
            for table_row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        event_anchors: Dict[str, str] = {}
        if terminal_ids and "job_events" in tables:
            event_placeholders = ",".join("?" for _ in terminal_ids)
            for event_row in conn.execute(
                f"SELECT job_id, MAX(id) AS anchor FROM job_events "
                f"WHERE job_id IN ({event_placeholders}) AND kind IN "
                f"('job_transition', 'job_step', 'attempt_finished') "
                f"GROUP BY job_id",
                terminal_ids,
            ):
                if event_row["anchor"] is not None:
                    event_anchors[str(event_row["job_id"])] = str(
                        event_row["anchor"]
                    )
        terminal_episodes: Dict[str, str] = {}
        for terminal_row in terminal_rows:
            job_id = str(terminal_row["id"])
            state_value = str(terminal_row[state_column] or "")
            step_value = (
                str(terminal_row["step"] or "") if "step" in columns else ""
            )
            anchor = event_anchors.get(job_id)
            if anchor is None and "revision" in columns:
                anchor = str(terminal_row["revision"] or "")
            if anchor is None and "updated_at" in columns:
                anchor = str(terminal_row["updated_at"] or "")
            terminal_episodes[job_id] = (
                f"{state_value}:{step_value}:{anchor or ''}"
            )
        fresh_ids, sidecar_origins = _pending(terminal_episodes)
        if not fresh_ids:
            return []
        id_placeholders = ",".join("?" for _ in fresh_ids)
        rows = conn.execute(
            f"SELECT * FROM jobs WHERE id IN ({id_placeholders})", fresh_ids
        ).fetchall()
        stored_origins: Dict[str, Dict[str, Any]] = {}
        has_origins = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='job_origins'"
        ).fetchone()
        if has_origins:
            for origin_row in conn.execute(
                f"SELECT job_id, origin FROM job_origins "
                f"WHERE job_id IN ({id_placeholders})",
                fresh_ids,
            ):
                candidate = _valid_origin(_as_json_dict(origin_row[1]))
                if candidate:
                    stored_origins[str(origin_row[0])] = candidate

        if "job_attempts" in tables:
            attempt_columns = {
                str(info[1]) for info in conn.execute("PRAGMA table_info(job_attempts)")
            }
            if {"job_id", "ordinal"} <= attempt_columns:
                attempt_rows = conn.execute(
                    f"SELECT * FROM job_attempts "
                    f"WHERE job_id IN ({id_placeholders})",
                    fresh_ids,
                ).fetchall()
                latest_attempts = _latest_by_job(
                    attempt_rows, ordinal_key="ordinal"
                )
        if "job_receipts" in tables:
            receipt_columns = {
                str(info[1]) for info in conn.execute("PRAGMA table_info(job_receipts)")
            }
            if {"job_id", "data", "created_at"} <= receipt_columns:
                receipt_rows = conn.execute(
                    f"SELECT rowid AS receipt_rowid, * FROM job_receipts "
                    f"WHERE job_id IN ({id_placeholders}) ORDER BY rowid",
                    fresh_ids,
                ).fetchall()
                for receipt_row in receipt_rows:
                    receipts_by_job.setdefault(
                        str(receipt_row["job_id"]), []
                    ).append(receipt_row)
    except sqlite3.DatabaseError as exc:
        logger.debug("jobs watcher: cannot read %s: %s", path, exc)
        return []
    finally:
        conn.close()

    jobs: List[Dict[str, Any]] = []
    for row in rows:
        job_id = str(row["id"])
        # Runner-side capture wins; otherwise use the sidecar record written at
        # create time.  Never guess a return address: "newest desktop chat" is
        # unrelated to the chat that commissioned the Job and can disclose the
        # result into the wrong conversation.  Origin-less Jobs remain visible
        # in the queue/status surfaces until their source explicitly binds one.
        origin = (
            _extract_origin(row, columns)
            or stored_origins.get(job_id)
            or sidecar_origins.get(job_id)
        )
        unrouted = False
        if not origin:
            origin = _unrouted_feed_origin()
            if not origin:
                continue
            unrouted = True
        blobs = {
            name: _as_json_dict(row[name])
            for name in _BLOB_COLUMNS
            if name in columns
        }
        attempt = latest_attempts.get(job_id)
        attempt_id = (
            str(attempt["id"] or "")
            if attempt is not None and "id" in attempt.keys()
            else ""
        )
        attempt_commit = (
            str(attempt["commit_sha"] or "")
            if attempt is not None and "commit_sha" in attempt.keys()
            else ""
        )

        # Attempt evidence and activation evidence are distinct receipts.  A
        # later deployment receipt must not hide the earlier VULCAN/THEMIS
        # report that explains what was tested and reviewed.
        receipt: Dict[str, Any] = {}
        activation: Optional[Dict[str, Any]] = None
        for receipt_row in reversed(receipts_by_job.get(job_id, [])):
            data = _as_json_dict(receipt_row["data"])
            receipt_attempt = str(receipt_row["attempt_id"] or "")
            if (
                activation is None
                and data.get("schema_version") == 1
                and data.get("kind") == "jobs-activation"
                and str(data.get("job_id") or "") == job_id
                and str(data.get("attempt_id") or "") == attempt_id
                and attempt_id
                and attempt_commit
                and str(data.get("commit") or "") == attempt_commit
                and receipt_attempt == attempt_id
                and all(data.get(key) is True for key in (
                    "merged", "deployed", "live_verified"
                ))
                and data.get("migrations") in ("applied", "not_required")
            ):
                activation = data
            if (
                not receipt
                and isinstance(data.get("worker_reported"), dict)
                and (not attempt_id or receipt_attempt == attempt_id)
            ):
                receipt = data
        reported = receipt.get("worker_reported")
        if not isinstance(reported, dict):
            reported = {}

        def attempt_value(name: str):
            return attempt[name] if attempt is not None and name in attempt.keys() else None

        repository = attempt_value("repository") or receipt.get("repository")
        branch = attempt_value("branch") or receipt.get("branch")
        commit = attempt_value("commit_sha") or receipt.get("commit")
        base_commit = attempt_value("base_commit") or receipt.get("base_commit")
        files = _reported_list(reported, "changes")
        if not files:
            files = _file_list(_field(row, columns, _FILES_COLUMNS, blobs))
        if not files:
            files = _git_files(repository, base_commit, commit)
        migrations, rollbacks, migration_instructions = _migration_handoff(
            repository, commit, files
        )
        verdict = _scalar(reported.get("review_verdict"), 64) or _scalar(
            _field(row, columns, _VERDICT_COLUMNS, blobs)
        )
        rounds = reported.get("review_rounds")
        if rounds in (None, ""):
            rounds = _scalar(_field(row, columns, _ROUNDS_COLUMNS, blobs), 32)
        jobs.append({
            "id": job_id,
            "number": row["number"] if "number" in columns else None,
            "state": str(row[state_column] or ""),
            "step": str(row["step"] or "") if "step" in columns else "",
            "name": _scalar(_field(row, columns, _NAME_COLUMNS, blobs)),
            "outcome": _scalar(reported.get("outcome"), 64)
            or _scalar(_field(row, columns, _OUTCOME_COLUMNS, blobs)),
            "summary": _scalar(reported.get("summary"), 1000),
            "verdict": verdict,
            "rounds": rounds,
            "tests": _reported_list(reported, "tests_run"),
            "files": files,
            "migrations": migrations,
            "rollbacks": rollbacks,
            "migration_instructions": migration_instructions,
            "attempt_status": attempt_value("status"),
            "failure_class": attempt_value("failure_class"),
            "repository": repository,
            "branch": branch,
            "commit": commit,
            "base_commit": base_commit,
            "merged": (
                True if activation is not None
                else _merged_into_main(repository, commit)
            ),
            "activation": activation,
            "delivery_episode": terminal_episodes[job_id],
            "origin": origin,
            "unrouted": unrouted,
        })
    return jobs


class GatewayJobsWatcherMixin:
    """Jobs wake loop for GatewayRunner."""

    async def _jobs_wake_watcher(self, interval: float = 10.0) -> None:
        """Announce terminal jobs back into the session that created them.

        One message per job, exactly once. Same loop shape as
        ``_kanban_notifier_watcher``: an initial delay so adapters finish
        wiring, all DB work in ``asyncio.to_thread``, per-tick failures logged
        and swallowed, and a sliced sleep so shutdown stays snappy.
        """
        from gateway.config import Platform as _Platform

        await asyncio.sleep(5)

        while self._running:
            try:
                jobs = await asyncio.to_thread(collect_terminal_jobs)
                for job in jobs:
                    origin = job["origin"]
                    platform_str = str(origin.get("platform") or "").lower()
                    try:
                        plat = _Platform(platform_str)
                    except ValueError:
                        # Unknown platform string — claim it so we don't
                        # re-evaluate the same dead row every tick.
                        await asyncio.to_thread(
                            claim_job, job["id"], job["delivery_episode"]
                        )
                        logger.debug(
                            "jobs watcher: job %s has unknown platform %r; skipping",
                            job["id"], platform_str,
                        )
                        continue
                    profile = str(origin.get("profile") or "").strip() or None
                    adapter = self._authorization_adapter(plat, profile)
                    if adapter is None:
                        # Adapter not connected (yet). Leave the job unclaimed
                        # so a later tick — or the gateway that DOES own this
                        # profile — announces it.
                        logger.debug(
                            "jobs watcher: no adapter for %s (profile=%s); "
                            "deferring job %s",
                            platform_str, profile or "default", job["id"],
                        )
                        continue

                    # Claim BEFORE delivering: the claim is the dedup, and it
                    # must be durable before the message can possibly be seen.
                    if not await asyncio.to_thread(
                        claim_job, job["id"], job["delivery_episode"]
                    ):
                        continue

                    try:
                        await self._deliver_job_wake(adapter, plat, job)
                        logger.info(
                            "jobs watcher: announced job %s (%s) into %s/%s",
                            job["id"], job["state"], platform_str,
                            origin.get("chat_id"),
                        )
                    except Exception as exc:
                        # Release the claim so the next tick retries instead of
                        # losing the announcement forever.
                        await asyncio.to_thread(
                            release_job, job["id"], job["delivery_episode"]
                        )
                        logger.warning(
                            "jobs watcher: delivery failed for job %s: %s",
                            job["id"], exc, exc_info=True,
                        )
            except Exception as exc:
                logger.warning("jobs watcher tick failed: %s", exc)

            slept = 0.0
            while slept < interval and self._running:
                await asyncio.sleep(min(1.0, interval - slept))
                slept += 1.0

    async def _deliver_job_wake(self, adapter, plat, job: Dict[str, Any]) -> None:
        """Deliver one job announcement. Raises so the caller can release."""
        from gateway.session import SessionSource
        from gateway.wake import deliver_wake

        origin = job["origin"]
        message = format_job_wake(job)
        if str(getattr(plat, "value", plat)).lower() == "api_server":
            await asyncio.to_thread(
                _append_desktop_message,
                str(origin.get("session_id") or origin.get("chat_id") or ""),
                message,
                job_id=str(job.get("id") or ""),
                delivery_key=str(job.get("delivery_episode") or ""),
            )
            return
        source = SessionSource(
            platform=plat,
            chat_id=str(origin.get("chat_id") or ""),
            chat_type=str(origin.get("chat_type") or "") or "group",
            thread_id=str(origin.get("thread_id") or "") or None,
            user_id=str(origin.get("user_id") or "") or None,
            profile=str(origin.get("profile") or "") or None,
        )
        # deliver_wake picks the strategy off the adapter's
        # supports_async_delivery flag: synthetic MessageEvent for push-capable
        # platforms, /v1/chat/completions self-post (needs the raw session id)
        # for the API server behind the desktop UI.
        await deliver_wake(
            adapter,
            text=message,
            session_id=str(origin.get("session_id") or ""),
            source=source,
        )
