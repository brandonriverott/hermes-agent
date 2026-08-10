"""Bundled Jobs tool registration; implementation imports remain lazy."""

from __future__ import annotations


def register(ctx) -> None:
    from hermes_cli import jobs_tool

    for name, schema, handler, emoji in jobs_tool.TOOL_REGISTRATIONS:
        ctx.register_tool(
            name=name,
            toolset="jobs",
            schema=schema,
            handler=handler,
            emoji=emoji,
            override=False,
        )
