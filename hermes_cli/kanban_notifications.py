"""Shared Kanban notification routing helpers.

The structured Kanban tool and the ``hermes kanban create`` CLI are both used
from live Hermes conversations.  They must preserve the same origin route so
later lifecycle events can be delivered back to the chat that created a card.
"""

from __future__ import annotations

import logging
import os
from typing import Any


logger = logging.getLogger(__name__)


def auto_subscribe_origin(conn: Any, task_id: str) -> bool:
    """Best-effort subscribe the current persistent chat/TUI session.

    Returns ``False`` for unattached CLI/cron processes and on bookkeeping
    failures.  Notification setup must never roll back task creation.
    """
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
        if not cfg_get(
            cfg, "kanban", "auto_subscribe_on_create", default=True
        ):
            return False
    except Exception:
        # A broken/unavailable config keeps the user-friendly default.
        pass

    platform = ""
    chat_id = ""
    try:
        from gateway.session_context import get_session_env

        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
        if not platform or not chat_id:
            # Desktop/TUI sessions have no messaging adapter, but their child
            # shell commands inherit this durable local-session routing key.
            session_key = (
                get_session_env("HERMES_SESSION_KEY", "")
                or os.environ.get("HERMES_SESSION_KEY", "")
            )
            if not session_key:
                return False
            platform = "tui"
            chat_id = session_key

        thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "") or None
        user_id = get_session_env("HERMES_SESSION_USER_ID", "") or None
        chat_type = get_session_env("HERMES_SESSION_CHAT_TYPE", "") or None
        message_id = get_session_env("HERMES_SESSION_MESSAGE_ID", "") or ""
        notifier_profile = (
            get_session_env("HERMES_SESSION_PROFILE", "")
            or os.environ.get("HERMES_PROFILE")
        )
        if not notifier_profile:
            try:
                from hermes_cli.profiles import get_active_profile_name

                notifier_profile = get_active_profile_name() or "default"
            except Exception:
                notifier_profile = "default"

        delivery_metadata: dict[str, Any] = {}
        if thread_id:
            delivery_metadata["thread_id"] = thread_id
        if chat_type:
            delivery_metadata["chat_type"] = chat_type
        if (
            platform.lower() == "telegram"
            and thread_id
            and (chat_type or "").lower() in {"dm", "direct", "private"}
        ):
            delivery_metadata["telegram_dm_topic_reply_fallback"] = True
            if str(thread_id) not in {"", "1"}:
                delivery_metadata["direct_messages_topic_id"] = str(thread_id)
            if message_id:
                delivery_metadata["telegram_reply_to_message_id"] = str(
                    message_id
                )

        from hermes_cli import kanban_db as kb

        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=chat_id,
            chat_type=chat_type,
            thread_id=thread_id,
            user_id=user_id,
            notifier_profile=notifier_profile,
            delivery_metadata=delivery_metadata or None,
        )
        return True
    except Exception as exc:
        logger.warning(
            "kanban auto-subscribe failed: %r (platform=%r key_set=%r)",
            exc,
            platform,
            bool(chat_id),
        )
        return False
