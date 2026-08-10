"""Provider-neutral Jobs execution contracts and fail-closed selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import jobs_adapter_claude as claude_adapter
from hermes_cli import jobs_execution as execution
from hermes_cli import jobs_executors as executors
from hermes_cli import jobs_identity as identity


def test_evidence_contract_and_bounded_helpers_are_provider_neutral(tmp_path):
    artifact_path = tmp_path / "artifact.txt"
    artifact_path.write_text("provider-neutral evidence\n", encoding="utf-8")

    claim = execution.claim_artifact(artifact_path, name="tests")
    result = execution.ReliabilityExecution(
        status="succeeded",
        commit="a" * 40,
        worktree=tmp_path,
        artifacts=(claim,),
        executor_exit_digest="sha256:" + "b" * 64,
        output_capture_digest="sha256:" + "c" * 64,
    )

    assert isinstance(claim, execution.ArtifactClaim)
    assert result.artifacts == (claim,)
    assert claim.size == len(b"provider-neutral evidence\n")

    oversized = tmp_path / "oversized.log"
    oversized.write_text("first line\n" + ("x" * 512), encoding="utf-8")
    bounded = execution.bounded_text(oversized, maximum=96)
    assert bounded is not None
    assert len(bounded.encode("utf-8")) <= 96
    assert "truncated" in bounded
    assert len(execution.capped_utf8("snowman: \N{SNOWMAN}" * 100, maximum=47)) <= 47


def test_claude_adapter_reexports_the_neutral_contract_for_compatibility():
    assert claude_adapter.ArtifactClaim is execution.ArtifactClaim
    assert claude_adapter.ReliabilityExecution is execution.ReliabilityExecution
    assert claude_adapter.claim_artifact is execution.claim_artifact
    assert claude_adapter.MAX_PRESERVED_BYTES == execution.MAX_PRESERVED_BYTES


def test_immutable_registry_contains_exactly_claude_and_codex_identities():
    assert executors.registry.names == ("claude", "codex")
    assert executors.registry.require("claude").identity == (
        identity.resolve_requested_lane("claude")
    )
    assert executors.registry.require("codex").identity == (
        identity.resolve_requested_lane("codex")
    )

    with pytest.raises(TypeError):
        executors.registry.entries["kat"] = executors.registry.require("claude")


@pytest.mark.parametrize(
    "name",
    [None, "", "unknown", "missing", "kat", "kat-builder", "gpt-builder"],
)
def test_registry_has_no_default_alias_or_unknown_executor(name):
    with pytest.raises(executors.UnsupportedExecutor) as exc:
        executors.registry.require(name)
    if name:
        assert str(name) not in str(exc.value)


def test_registry_refusal_never_reflects_a_secret_shaped_executor():
    secret = "authorization=Bearer registry-secret-123456789"
    with pytest.raises(executors.UnsupportedExecutor) as exc:
        executors.registry.require(secret)

    assert secret not in str(exc.value)


def test_injected_reliability_adapters_select_only_the_exact_executor():
    calls: list[str] = []
    selected = executors.registry.with_reliability_adapters(
        {
            "claude": lambda _context: calls.append("claude"),
            "codex": lambda _context: calls.append("codex"),
        }
    )

    selected.require_reliability("codex")(object())
    assert calls == ["codex"]
    calls.clear()
    selected.require_reliability("claude")(object())
    assert calls == ["claude"]


def test_task_three_production_registry_has_no_fake_codex_adapter():
    production = executors.production_registry()

    assert production.require_legacy("claude").name == "claude"
    with pytest.raises(executors.UnsupportedExecutor, match="no installed legacy adapter"):
        production.require_legacy("codex")


def test_historical_gpt_alias_normalizes_only_at_the_identity_boundary():
    historical = SimpleNamespace(
        requested_lane=None,
        executor=None,
        specialist="gpt-builder",
        model=None,
    )

    normalized = executors.registry.require_job_identity(historical)

    assert normalized == identity.resolve_requested_lane("codex")
    with pytest.raises(executors.UnsupportedExecutor):
        executors.registry.require("gpt-builder")
