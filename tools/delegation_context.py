"""Thread-safe runtime identity for in-process delegated subagents.

Delegate children share their parent worker's process and therefore inherit
process-wide environment variables such as ``HERMES_KANBAN_TASK``.  A
ContextVar distinguishes the child execution scope without mutating global
environment state or leaking across parallel child threads.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_IS_DELEGATED_SUBAGENT: ContextVar[bool] = ContextVar(
    "hermes_is_delegated_subagent",
    default=False,
)


def is_delegated_subagent() -> bool:
    """Return whether the current execution context is a delegate child."""
    return _IS_DELEGATED_SUBAGENT.get()


@contextmanager
def delegated_subagent_scope() -> Iterator[None]:
    """Mark the current context as a delegated child for the call duration."""
    token = _IS_DELEGATED_SUBAGENT.set(True)
    try:
        yield
    finally:
        _IS_DELEGATED_SUBAGENT.reset(token)
