"""Immutable, fail-closed executor identities and adapter selection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Optional, TypeAlias

from hermes_cli import jobs_identity as ji
from hermes_cli.jobs_execution import ReliabilityExecution


ReliabilityExecutor: TypeAlias = Callable[[object], ReliabilityExecution]


class UnsupportedExecutor(ValueError):
    """An execution identity or required provider adapter is unsupported."""


@dataclass(frozen=True)
class LegacyExecutorAdapter:
    """Temporary adapter contract for the legacy ``run-once`` seam."""

    name: str
    preflight: Callable[[object], None]
    preflight_skills: Callable[..., object]
    run_attempt: Callable[..., object]


@dataclass(frozen=True)
class ExecutorDefinition:
    """One canonical executor identity plus explicitly installed callables."""

    name: str
    identity: ji.JobIdentity
    reliability: Optional[ReliabilityExecutor] = None
    legacy: Optional[LegacyExecutorAdapter] = None


class ExecutorRegistry:
    """An immutable registry whose only identities are Claude and Codex."""

    def __init__(self, entries: Mapping[str, ExecutorDefinition]):
        copied = dict(entries)
        expected = set(ji.REQUESTED_LANES)
        if set(copied) != expected:
            raise ValueError("executor registry must contain exactly claude and codex")
        for name, entry in copied.items():
            canonical = ji.resolve_requested_lane(name)
            if (
                not isinstance(entry, ExecutorDefinition)
                or entry.name != name
                or entry.identity != canonical
            ):
                raise ValueError("invalid canonical executor definition")
        self._entries = MappingProxyType(copied)

    @property
    def entries(self) -> Mapping[str, ExecutorDefinition]:
        return self._entries

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def require(self, name: object) -> ExecutorDefinition:
        if not isinstance(name, str) or name not in self._entries:
            raise UnsupportedExecutor(
                "unsupported executor identity; expected claude or codex"
            )
        return self._entries[name]

    def require_job_identity(self, job: object) -> ji.JobIdentity:
        """Normalize one persisted identity, wrapping every refusal uniformly."""

        try:
            identity = ji.effective_identity(job)
            definition = self.require(identity.executor)
        except (ji.UnsupportedJobLane, UnsupportedExecutor) as exc:
            raise UnsupportedExecutor(
                "persisted Job execution identity is unsupported"
            ) from exc
        if definition.identity != identity:
            raise UnsupportedExecutor("persisted Job identity is not canonical")
        return identity

    def require_reliability(self, name: object) -> ReliabilityExecutor:
        definition = self.require(name)
        if definition.reliability is None:
            raise UnsupportedExecutor(
                "selected executor has no installed reliability adapter"
            )
        return definition.reliability

    def require_legacy(self, name: object) -> LegacyExecutorAdapter:
        definition = self.require(name)
        if definition.legacy is None:
            raise UnsupportedExecutor(
                "selected executor has no installed legacy adapter"
            )
        return definition.legacy

    def with_reliability_adapters(
        self, adapters: Mapping[str, ReliabilityExecutor]
    ) -> "ExecutorRegistry":
        entries = dict(self._entries)
        for name, adapter in adapters.items():
            definition = self.require(name)
            if not callable(adapter):
                raise TypeError("reliability adapter must be callable")
            entries[name] = replace(definition, reliability=adapter)
        return ExecutorRegistry(entries)

    def with_legacy_adapters(
        self, adapters: Mapping[str, LegacyExecutorAdapter]
    ) -> "ExecutorRegistry":
        entries = dict(self._entries)
        for name, adapter in adapters.items():
            definition = self.require(name)
            if not isinstance(adapter, LegacyExecutorAdapter):
                raise TypeError("legacy adapter has the wrong contract")
            if adapter.name != name:
                raise UnsupportedExecutor(
                    "legacy adapter identity does not match its executor"
                )
            entries[name] = replace(definition, legacy=adapter)
        return ExecutorRegistry(entries)


registry = ExecutorRegistry(
    {
        lane: ExecutorDefinition(
            name=lane,
            identity=ji.resolve_requested_lane(lane),
        )
        for lane in ji.REQUESTED_LANES
    }
)


def production_registry() -> ExecutorRegistry:
    """Install the adapters that exist in this release slice.

    Both Claude and Codex are registered with their explicit legacy adapters.
    No provider substitution: a Codex call cannot land on the Claude adapter
    and vice versa. Every identity is resolved before any provider call.
    """

    from hermes_cli import jobs_adapter_claude as claude
    from hermes_cli import jobs_adapter_codex as codex

    return registry.with_legacy_adapters(
        {
            ji.CLAUDE_EXECUTOR: LegacyExecutorAdapter(
                name=ji.CLAUDE_EXECUTOR,
                preflight=claude.preflight,
                preflight_skills=claude.preflight_skills,
                run_attempt=claude.run_claude_attempt,
            ),
            ji.CODEX_EXECUTOR: LegacyExecutorAdapter(
                name=ji.CODEX_EXECUTOR,
                preflight=codex.preflight,
                preflight_skills=codex.preflight_skills,
                run_attempt=codex.run_codex_attempt,
            ),
        }
    )
