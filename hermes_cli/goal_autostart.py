"""Opt-in plugin bridge from an accepted task plan to the native goal loop."""
from __future__ import annotations

import time

from hermes_cli import goals
from hermes_cli.config import load_config_readonly


def start_goal_if_absent(session_id: str, *, request: str, outcome: str,
                         verification: str) -> str:
    """Arm once, preserving every existing state (including pause/clear/done).

    Call only after a user-authorized plan has been accepted, in the owning
    profile and native session scope. This schedules no turn and runs no shell
    gates. The normal frontend goal loop owns judging, budgets and Stop.
    Returns ``started`` or ``existing``; persistence failures raise.
    """
    if not all(isinstance(v, str) and v.strip()
               for v in (session_id, request, outcome, verification)):
        raise ValueError("A native session, original request, outcome and verification are required")
    config = load_config_readonly().get("goals") or {}
    limit = int(config.get("max_turns", goals.DEFAULT_MAX_TURNS))
    if limit <= 0:
        raise ValueError("goals.max_turns must be positive")
    state = goals.GoalState(
        goal=request, status="active", max_turns=limit, created_at=time.time(),
        contract=goals.GoalContract(
            outcome=outcome,
            verification=verification + "\nCheck the entire original request against actual artifacts and task-specific tests. "
            "A checkpoint or one passed subtask is not overall acceptance.",
            constraints="Preserve existing work and inspect original jobs before any retry. Keep existing models, "
            "permissions, approval gates and review/retry budgets. Do not duplicate working jobs.",
            boundaries="A goal grants no new authority. Stop or pause remains authoritative. "
            "Do not merge, publish, delete or spend beyond the user's authorization.",
            stop_when="The whole requested outcome is verified; or report INCOMPLETE with the concrete "
            "missing access, required user decision or exhausted existing budget. Never just promise to continue.",
        ),
    )
    db = goals._get_session_db()
    if db is None:
        raise RuntimeError("Native completion tracking could not open the session database")
    # An atomic insert also preserves malformed records and a concurrent explicit
    # goal command. A read-then-set would overwrite pause or reset a spent budget.
    inserted = db.state_meta.insert_if_absent(goals._meta_key(session_id), state.to_json())
    return "started" if inserted else "existing"
