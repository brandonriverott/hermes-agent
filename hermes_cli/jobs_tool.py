"""Lazy, exact-origin chat tools for the independent Jobs control plane.

Importing this module is intentionally cheap.  Plugin discovery needs the two
schemas and handlers, but the Jobs database and gateway context are imported
only when a handler is actually called.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hermes_cli import jobs_assurance
from hermes_cli import jobs_identity as ji


logger = logging.getLogger(__name__)

DEFAULT_MAX_TURNS = 120
MAX_TURNS_CEILING = 500
GIT_TIMEOUT_SECONDS = 5.0
MAX_ORIGIN_FIELD_CHARS = 512
MAX_NAME_CHARS = 240
MAX_GOAL_CHARS = 262_144
MAX_REPO_PATH_CHARS = 4096
MAX_QUEUE_JOBS = 100

_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")
_ORIGIN_FIELDS = (
    "platform",
    "chat_id",
    "session_id",
    "chat_type",
    "thread_id",
    "user_id",
    "profile",
)
_REQUIRED_ORIGIN_FIELDS = frozenset({"platform", "chat_id", "session_id"})
_CREATE_FIELDS = frozenset(
    {"name", "goal", "repo_path", "lane", "max_turns", "assurance"}
)
_API_SESSION_SOURCES = frozenset({"desktop", "tui", "api_server"})


class OriginCaptureUnavailable(RuntimeError):
    """The current tool request has no trustworthy delivery origin."""


class RepoHeadUnavailable(RuntimeError):
    """The requested repository does not expose a readable commit HEAD."""


class _InputRefusal(ValueError):
    def __init__(self, field: str):
        super().__init__(field)
        self.field = field


def _json_result(payload: Mapping[str, object]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)


def _error(code: str, message: str, *, field: str | None = None) -> str:
    payload: dict[str, object] = {
        "error": str(message)[:240],
        "code": code,
    }
    if field is not None:
        payload["field"] = field
    return _json_result(payload)


def _has_control(value: str) -> bool:
    return any(ord(char) < 32 or ord(char) == 127 for char in value)


def _required_text(
    args: Mapping[str, object],
    field: str,
    *,
    maximum: int,
    controls_allowed: bool = False,
) -> str:
    value = args.get(field)
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise _InputRefusal(field)
    if not value.strip():
        raise _InputRefusal(field)
    if not controls_allowed and _has_control(value):
        raise _InputRefusal(field)
    return value


def _validate_create_args(
    args: object,
) -> tuple[str, str, Path, ji.JobIdentity, int, dict[str, object]]:
    if not isinstance(args, Mapping):
        raise _InputRefusal("request")
    if any(key not in _CREATE_FIELDS for key in args):
        # Argument names are caller-controlled too.  Never echo an unknown key
        # into the tool result: it may itself contain a credential or prompt
        # framing text.
        raise _InputRefusal("request")

    name = _required_text(args, "name", maximum=MAX_NAME_CHARS)
    goal = _required_text(
        args,
        "goal",
        maximum=MAX_GOAL_CHARS,
        controls_allowed=True,
    )
    if "\x00" in goal:
        raise _InputRefusal("goal")
    raw_repo = _required_text(args, "repo_path", maximum=MAX_REPO_PATH_CHARS)

    try:
        identity = ji.resolve_requested_lane(args.get("lane"))
    except ji.UnsupportedJobLane as exc:
        raise _InputRefusal("lane") from exc

    max_turns = args.get("max_turns", DEFAULT_MAX_TURNS)
    if (
        isinstance(max_turns, bool)
        or not isinstance(max_turns, int)
        or max_turns < 1
        or max_turns > MAX_TURNS_CEILING
    ):
        raise _InputRefusal("max_turns")

    repo = Path(raw_repo)
    if not repo.is_absolute():
        raise _InputRefusal("repo_path")
    try:
        resolved = repo.resolve(strict=True)
        is_directory = resolved.is_dir()
        has_git_marker = (resolved / ".git").exists()
    except OSError as exc:
        raise _InputRefusal("repo_path") from exc
    if not is_directory or not has_git_marker:
        raise _InputRefusal("repo_path")
    try:
        assurance = jobs_assurance.AssuranceContract.from_mapping(
            args.get("assurance")
        )
    except jobs_assurance.InvalidAssuranceContract as exc:
        raise _InputRefusal("assurance") from exc
    return name.strip(), goal, resolved, identity, max_turns, assurance.to_mapping()


def resolve_repo_head(
    repo: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    """Resolve one full commit SHA with argv execution and a hard timeout."""
    argv = [
        "git",
        "-C",
        str(repo.resolve()),
        "rev-parse",
        "--verify",
        "HEAD^{commit}",
    ]
    try:
        completed = run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepoHeadUnavailable("repository HEAD is unavailable") from exc
    head = str(completed.stdout or "").strip()
    if not _SHA_RE.fullmatch(head):
        raise RepoHeadUnavailable("repository HEAD is unavailable")
    return head


def _normalized_session_value(value: object) -> object:
    """Trim a safe context value while leaving invalid values rejectable."""
    if not isinstance(value, str):
        return value
    if len(value) > MAX_ORIGIN_FIELD_CHARS or _has_control(value):
        return value
    return value.strip()


def _clobber_proof_api_origin_session_id() -> object:
    """Return the API request's stable session id when that helper is usable.

    The helper is deliberately optional: Desktop/TUI and older API binders do
    not present the raw ``platform=api_server`` shape it gates on.  Those paths
    safely fall back to their request-scoped durable session id/session key.
    """
    try:
        from tools.async_delegation import _current_origin_session_id

        return _current_origin_session_id()
    except Exception:
        return ""


def capture_exact_origin() -> dict[str, object]:
    """Capture the request-local Hermes chat identity, or fail closed."""
    try:
        from gateway.session_context import get_session_env
    except Exception as exc:
        raise OriginCaptureUnavailable("request origin helper unavailable") from exc

    try:
        captured: dict[str, object] = {
            "platform": get_session_env("HERMES_SESSION_PLATFORM", ""),
            "source": get_session_env("HERMES_SESSION_SOURCE", ""),
            "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
            "session_id": get_session_env("HERMES_SESSION_ID", ""),
            "session_key": get_session_env("HERMES_SESSION_KEY", ""),
            "chat_type": get_session_env("HERMES_SESSION_CHAT_TYPE", ""),
            "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", ""),
            "user_id": get_session_env("HERMES_SESSION_USER_ID", ""),
            "profile": get_session_env("HERMES_SESSION_PROFILE", ""),
        }
    except Exception as exc:
        raise OriginCaptureUnavailable("request origin helper failed") from exc

    normalized = {
        field: _normalized_session_value(value)
        for field, value in captured.items()
    }
    platform = normalized["platform"]
    source = normalized["source"]
    chat_id = normalized["chat_id"]
    session_id = normalized["session_id"]
    session_key = normalized["session_key"]

    # Desktop, TUI, and API-server requests are durable Hermes sessions but
    # have no push-platform chat id.  Normalize only those explicitly trusted
    # request sources into the established API-session address shape.  CLI,
    # cron, and unknown sources remain originless and are rejected downstream.
    if platform == "" and source in _API_SESSION_SOURCES:
        platform = "api_server"

    if platform == "api_server":
        stable_api_origin = _normalized_session_value(
            _clobber_proof_api_origin_session_id()
        )
        if isinstance(stable_api_origin, str) and stable_api_origin:
            chat_id = stable_api_origin
            session_id = stable_api_origin
        else:
            durable_id = session_id or session_key
            session_id = durable_id
            if not chat_id:
                chat_id = durable_id
    elif isinstance(platform, str) and platform:
        # Older/cached push turns may expose the request-scoped routing key
        # before the durable id is rebound.  It is scoped to this exact turn,
        # unlike a process-global environment fallback.
        session_id = session_id or session_key

    return {
        "platform": platform,
        "chat_id": chat_id,
        "session_id": session_id,
        "chat_type": normalized["chat_type"],
        "thread_id": normalized["thread_id"],
        "user_id": normalized["user_id"],
        "profile": normalized["profile"],
    }


def _validated_origin(raw: object) -> dict[str, str]:
    if raw is None or raw == {}:
        raise OriginCaptureUnavailable("no request origin is available")
    if not isinstance(raw, Mapping):
        raise ValueError("invalid request origin")

    origin: dict[str, str] = {}
    for field in _ORIGIN_FIELDS:
        value = raw.get(field, "")
        if not isinstance(value, str):
            raise ValueError("invalid request origin")
        if len(value) > MAX_ORIGIN_FIELD_CHARS or _has_control(value):
            raise ValueError("invalid request origin")
        if value != value.strip():
            raise ValueError("invalid request origin")
        if field in _REQUIRED_ORIGIN_FIELDS and not value:
            raise ValueError("invalid request origin")
        origin[field] = value
    return origin


def _execution_goal(
    *,
    repo: Path,
    base_commit: str,
    model: str,
    max_turns: int,
    body: str,
) -> str:
    header = (
        f"REPO_PATH={repo}\n"
        f"BASE_COMMIT={base_commit}\n"
        f"MODEL={model}\n"
        "EFFORT=max\n"
        f"MAX_TURNS={max_turns}\n\n"
    )
    return header + body


def jobs_create_handler(args: object, **_request_context: Any) -> str:
    """Validate and atomically create a canonical, exactly-originated Job."""
    try:
        name, body, repo, identity, max_turns, assurance = _validate_create_args(args)
    except _InputRefusal as exc:
        code = "invalid_repo" if exc.field == "repo_path" else "invalid_input"
        message = (
            "Repository must be an absolute Git worktree."
            if code == "invalid_repo"
            else "Invalid jobs_create request."
        )
        return _error(code, message, field=exc.field)

    try:
        base_commit = resolve_repo_head(repo)
    except RepoHeadUnavailable:
        return _error(
            "head_unavailable",
            "Repository HEAD could not be resolved.",
            field="repo_path",
        )

    try:
        raw_origin = capture_exact_origin()
    except Exception:
        return _error(
            "origin_unavailable",
            "This Jobs tool requires an attached Hermes chat session.",
        )
    try:
        origin = _validated_origin(raw_origin)
    except OriginCaptureUnavailable:
        return _error(
            "origin_unavailable",
            "This Jobs tool requires an attached Hermes chat session.",
        )
    except Exception:
        return _error(
            "invalid_origin",
            "The current Hermes chat origin is invalid.",
        )

    goal = _execution_goal(
        repo=repo,
        base_commit=base_commit,
        model=identity.model,
        max_turns=max_turns,
        body=body,
    )
    try:
        from hermes_cli import jobs_db as jdb

        with jdb.connect_closing() as conn:
            job_id = jdb.create_job(
                conn,
                name=name,
                goal=goal,
                requested_lane=identity.requested_lane,
                origin=origin,
                assurance_contract=assurance,
            )
            job = jdb.get_job(conn, job_id)
            if job is None:  # defensive: never return an unverified write
                raise RuntimeError("created Job could not be read back")
    except Exception as exc:
        logger.error("jobs_create persistence failed (%s)", type(exc).__name__)
        return _error("create_failed", "The Job could not be created.")

    return _json_result(
        {
            "job_id": job.id,
            "number": job.number,
            "lane": job.requested_lane,
            "executor": job.executor,
            "specialist": job.specialist,
            "model": job.model,
            "base_commit": base_commit,
            "notification_promise": (
                "Lifecycle updates will return to this originating Hermes chat."
            ),
        }
    )


def jobs_queue_handler(args: object, **_request_context: Any) -> str:
    """Return a bounded, mutation-free Jobs queue projection."""
    if not isinstance(args, Mapping) or args:
        return _error("invalid_input", "jobs_queue does not accept arguments.")
    try:
        from hermes_cli import jobs_db as jdb

        with jdb.connect_closing() as conn:
            jobs = jdb.list_jobs(conn)[-MAX_QUEUE_JOBS:]
    except Exception as exc:
        logger.error("jobs_queue read failed (%s)", type(exc).__name__)
        return _error("queue_unavailable", "The Jobs queue is unavailable.")

    return _json_result(
        {
            "jobs": [
                {
                    "job_id": job.id,
                    "number": job.number,
                    "name": job.name,
                    "status": job.status,
                    "step": job.step,
                    "requested_lane": job.requested_lane,
                    "executor": job.executor,
                    "specialist": job.specialist,
                    "model": job.model,
                }
                for job in jobs
            ]
        }
    )


JOBS_CREATE_SCHEMA = {
    "name": "jobs_create",
    "description": (
        "Create a durable background Job in the current repository and route "
        "it explicitly to the Claude or Codex executor. The originating Hermes "
        "chat receives lifecycle updates."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_NAME_CHARS,
                "description": "Short plain-English Job name.",
            },
            "goal": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_GOAL_CHARS,
                "description": "Complete implementation goal, preserved verbatim.",
            },
            "repo_path": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_REPO_PATH_CHARS,
                "description": "Absolute path to the Git worktree to modify.",
            },
            "lane": {
                "type": "string",
                "enum": list(ji.REQUESTED_LANES),
                "description": "Explicit executor lane.",
            },
            "max_turns": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TURNS_CEILING,
                "default": DEFAULT_MAX_TURNS,
                "description": "Maximum builder turns (defaults to 120).",
            },
            "assurance": {
                "type": "object",
                "additionalProperties": False,
                "description": "Definition-of-done, critical journey, risk, and stop contract.",
                "properties": {
                    "critical_user_journey": {"type": "string", "minLength": 1, "maxLength": 4096},
                    "success_metric": {"type": "string", "minLength": 1, "maxLength": 4096},
                    "outcome_mode": {"type": "string", "enum": ["runtime", "visual", "integration", "not_applicable"]},
                    "verification_steps": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"type": "string", "minLength": 1, "maxLength": 4096}},
                    "max_attempts": {"type": "integer", "minimum": 1, "maximum": 20},
                    "wall_clock_budget_seconds": {"type": "integer", "minimum": 1, "maximum": 604800},
                    "risk_domains": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "string", "enum": ["money", "permissions", "state", "none"]}},
                    "consumers": {"type": "array", "maxItems": 64, "items": {"type": "string", "minLength": 1, "maxLength": 4096}},
                    "egress_paths": {"type": "array", "maxItems": 64, "items": {"type": "string", "minLength": 1, "maxLength": 4096}},
                    "rollback_behavior": {"type": "string", "minLength": 1, "maxLength": 4096},
                    "knowledge_closure_required": {"type": "boolean"},
                    "not_applicable_reason": {"type": ["string", "null"], "maxLength": 4096}
                },
                "required": ["critical_user_journey", "success_metric", "outcome_mode", "verification_steps", "max_attempts", "wall_clock_budget_seconds", "risk_domains", "consumers", "egress_paths", "rollback_behavior", "knowledge_closure_required"]
            },
        },
        "required": ["name", "goal", "repo_path", "lane", "assurance"],
    },
}


JOBS_QUEUE_SCHEMA = {
    "name": "jobs_queue",
    "description": "Read the bounded durable Jobs queue and canonical executor identity.",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    },
}


TOOL_REGISTRATIONS = (
    ("jobs_create", JOBS_CREATE_SCHEMA, jobs_create_handler, "🧰"),
    ("jobs_queue", JOBS_QUEUE_SCHEMA, jobs_queue_handler, "📋"),
)
