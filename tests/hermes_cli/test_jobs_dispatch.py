"""Provider-free tests for the Jobs dispatch control plane.

The first regression is Job 71, which the live shell dispatcher silently skips
because its valid repository path contains spaces.  These tests deliberately
exercise only the durable parsing contract; no Jobs database, provider, worker,
or live repository is touched.
"""

from __future__ import annotations

from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_cli import jobs_dispatch as dispatch
from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_executors as executors
from hermes_cli import jobs_graph as graph
from hermes_cli import jobs_harness as harness
from hermes_cli import jobs_identity as identity
from hermes_cli import jobs_lanes


JOB_71_REPO_PATH = "/Users/brandon/Documents/Kava Bar Scan"
JOB_71_BASE_COMMIT = "d15ef113de74f3d1bdf901f29b5e4eb9c51e475c"
JOB_71_GOAL = f"""REPO_PATH={JOB_71_REPO_PATH}
BASE_COMMIT={JOB_71_BASE_COMMIT}
MODEL=claude-opus-5
MAX_TURNS=120

Build the Room 92 V3 reconstruction.
"""


def test_job_71_repo_path_with_spaces_survives_legacy_parsing():
    header = dispatch.parse_legacy_execution_header(JOB_71_GOAL)

    assert header.repo_path == JOB_71_REPO_PATH


def test_job_71_base_commit_survives_legacy_parsing():
    header = dispatch.parse_legacy_execution_header(JOB_71_GOAL)

    assert header.base_commit == JOB_71_BASE_COMMIT


def test_job_71_model_and_turn_limit_survive_legacy_parsing():
    header = dispatch.parse_legacy_execution_header(JOB_71_GOAL)

    assert header.model == "claude-opus-5"
    assert header.max_turns == 120


def test_legacy_non_identity_defaults_are_explicit_and_stable():
    header = dispatch.parse_legacy_execution_header(
        f"REPO_PATH={JOB_71_REPO_PATH}\n"
        f"BASE_COMMIT={JOB_71_BASE_COMMIT}\n"
        "MODEL=claude-opus-5\n"
    )

    assert header.model == "claude-opus-5"
    assert header.effort == "max"
    assert header.max_turns == 120
    assert header.workspace_kind == "worktree"


@pytest.mark.parametrize(
    ("goal", "code", "key"),
    [
        (
            f"REPO_PATH=/a\nREPO_PATH=/b\nBASE_COMMIT={JOB_71_BASE_COMMIT}",
            "duplicate_key",
            "REPO_PATH",
        ),
        (
            f"REPO_PATH=   \nBASE_COMMIT={JOB_71_BASE_COMMIT}",
            "blank_value",
            "REPO_PATH",
        ),
        (
            f"REPO_PATH=/repo\nSECRET_MODE=yes\nBASE_COMMIT={JOB_71_BASE_COMMIT}",
            "unknown_key",
            "UNKNOWN",
        ),
    ],
)
def test_legacy_header_refusals_are_typed(goal, code, key):
    with pytest.raises(dispatch.LegacyMetadataError) as exc:
        dispatch.parse_legacy_execution_header(goal)

    assert exc.value.code == code
    assert exc.value.key == key
    assert exc.value.detail


def test_goal_body_key_value_text_is_not_mistaken_for_header_metadata():
    goal = (
        f"REPO_PATH={JOB_71_REPO_PATH}\n"
        f"BASE_COMMIT={JOB_71_BASE_COMMIT}\n"
        "MODEL=claude-opus-5\n\n"
        "Document EXAMPLE=value in the implementation notes.\n"
    )

    header = dispatch.parse_legacy_execution_header(goal)

    assert header.repo_path == JOB_71_REPO_PATH


def test_source_digest_binds_the_header_to_the_verbatim_goal():
    first = dispatch.parse_legacy_execution_header(JOB_71_GOAL)
    second = dispatch.parse_legacy_execution_header(JOB_71_GOAL + "More detail.\n")

    assert first.source_digest != second.source_digest


@pytest.mark.parametrize(
    ("goal", "code", "key"),
    [
        (f"BASE_COMMIT={JOB_71_BASE_COMMIT}", "missing_key", "REPO_PATH"),
        (f"REPO_PATH=/repo", "missing_key", "BASE_COMMIT"),
        (
            f"REPO_PATH=/repo\nBASE_COMMIT={JOB_71_BASE_COMMIT}",
            "missing_key",
            "MODEL",
        ),
        (
            f"REPO_PATH=relative/repo\nBASE_COMMIT={JOB_71_BASE_COMMIT}",
            "invalid_repo_path",
            "REPO_PATH",
        ),
        (
            "REPO_PATH=/repo\nBASE_COMMIT=HEAD",
            "invalid_base_commit",
            "BASE_COMMIT",
        ),
        (
            f"REPO_PATH=/repo\nBASE_COMMIT={JOB_71_BASE_COMMIT}\n"
            "MODEL=claude-opus-5\nMAX_TURNS=zero",
            "invalid_integer",
            "MAX_TURNS",
        ),
        (
            f"REPO_PATH=/repo\nBASE_COMMIT={JOB_71_BASE_COMMIT}\n"
            "MODEL=claude-opus-5\nMAX_TURNS=501",
            "out_of_bounds",
            "MAX_TURNS",
        ),
        (
            f"REPO_PATH=/repo\nBASE_COMMIT={JOB_71_BASE_COMMIT}\nEFFORT=impossible",
            "unsupported_value",
            "EFFORT",
        ),
        (
            f"REPO_PATH=/repo\nBASE_COMMIT={JOB_71_BASE_COMMIT}\nWORKSPACE_KIND=in_place",
            "unsupported_value",
            "WORKSPACE_KIND",
        ),
    ],
)
def test_invalid_legacy_execution_values_have_visible_refusal_codes(
    goal, code, key
):
    with pytest.raises(dispatch.LegacyMetadataError) as exc:
        dispatch.parse_legacy_execution_header(goal)

    assert exc.value.code == code
    assert exc.value.key == key


def _canonical_job(lane: str):
    resolved = identity.resolve_requested_lane(lane)
    return SimpleNamespace(
        requested_lane=resolved.requested_lane,
        executor=resolved.executor,
        specialist=resolved.specialist,
        model=resolved.model,
    )


def _canonical_spec(lane: str) -> dispatch.DispatchSpec:
    resolved = identity.resolve_requested_lane(lane)
    return dispatch.DispatchSpec(
        repository=Path("/tmp/provider-neutral-repo"),
        base_commit="a" * 40,
        branch="jobs/provider-neutral",
        output_parents=(),
        scoped_memory_paths=(),
        requested_lane=resolved.requested_lane,
        lane_id=f"{lane}-pc-1",
        executor=resolved.executor,
        specialist=resolved.specialist,
        model=resolved.model,
    )


def _selected_lane(lane: str) -> jobs_lanes.LaneDecision:
    resolved = identity.resolve_requested_lane(lane)
    registry = jobs_lanes.load_lane_registry()
    lane_id = f"{lane}-pc-1"
    definition = next(item for item in registry.lanes if item.id == lane_id)
    return jobs_lanes.LaneDecision(
        status="SELECTED",
        lane_id=lane_id,
        host_id=definition.host_id,
        executor=resolved.executor,
        model=resolved.model,
        policy_version=registry.policy_version,
        policy_digest=jobs_lanes.registry_digest(registry),
        fallback_applied=False,
        reason_code="PC_LANE_SELECTED",
        considered_lane_ids=tuple(
            item.id
            for item in registry.lanes
            if item.executor == resolved.executor
            and any(fnmatchcase(resolved.model, pattern) for pattern in item.model_patterns)
        ),
    )


@pytest.mark.parametrize("lane", ["claude", "codex"])
def test_dispatch_resolves_only_the_persisted_executor(lane):
    calls: list[str] = []
    registry = executors.registry.with_reliability_adapters(
        {
            "claude": lambda _context: calls.append("claude"),
            "codex": lambda _context: calls.append("codex"),
        }
    )

    selected = dispatch.resolve_dispatch_executor(
        job=_canonical_job(lane),
        spec=_canonical_spec(lane),
        executor_registry=registry,
        lane_decision=_selected_lane(lane),
    )
    selected(object())

    assert calls == [lane]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("requested_lane", "codex"),
        ("executor", "codex"),
        ("specialist", "codex-builder"),
        ("model", "gpt-5.6-sol"),
    ],
)
def test_dispatch_identity_mismatch_fails_before_any_provider_callable(
    field, bad_value
):
    calls: list[str] = []
    registry = executors.registry.with_reliability_adapters(
        {
            "claude": lambda _context: calls.append("claude"),
            "codex": lambda _context: calls.append("codex"),
        }
    )
    spec = replace(_canonical_spec("claude"), **{field: bad_value})

    with pytest.raises(executors.UnsupportedExecutor):
        dispatch.resolve_dispatch_executor(
            job=_canonical_job("claude"),
            spec=spec,
            executor_registry=registry,
            lane_decision=_selected_lane("claude"),
        )

    assert calls == []


def test_selected_lane_mismatch_fails_before_any_provider_callable():
    calls: list[str] = []
    registry = executors.registry.with_reliability_adapters(
        {
            "claude": lambda _context: calls.append("claude"),
            "codex": lambda _context: calls.append("codex"),
        }
    )
    wrong_lane = replace(_selected_lane("claude"), lane_id="claude-pc-2")

    with pytest.raises(executors.UnsupportedExecutor):
        dispatch.resolve_dispatch_executor(
            job=_canonical_job("claude"),
            spec=_canonical_spec("claude"),
            executor_registry=registry,
            lane_decision=wrong_lane,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("decision_changes", "spec_changes"),
    [
        ({"lane_id": "codex-pc-1"}, {"lane_id": "codex-pc-1"}),
        ({"host_id": "mac"}, {}),
        ({"policy_version": "forged-policy"}, {}),
        ({"policy_digest": "sha256:" + ("0" * 64)}, {}),
    ],
)
def test_forged_lane_decision_fails_before_preflight_custody_or_provider(
    tmp_path, decision_changes, spec_changes
):
    conn = jdb.connect(tmp_path / "jobs.db")
    try:
        job_id = jdb.create_job(
            conn,
            name="forged lane",
            goal="must not launch",
            requested_lane="claude",
        )
        repository = tmp_path / "repo"
        output = tmp_path / "output"
        memory = tmp_path / "memory.md"
        repository.mkdir()
        output.mkdir()
        memory.write_text("scoped", encoding="utf-8")
        probe_calls: list[str] = []

        def passed(_probe_input):
            probe_calls.append("probe")
            return harness.ProbeResult("PASS", "OK", {})

        probes = harness.PreflightProbes(
            **{name: passed for name in harness.CHECK_NAMES}
        )
        private_key = Ed25519PrivateKey.generate()
        signer = harness.ReceiptSigner("lane:test:v1", private_key)
        verifier = graph.ReceiptVerifier(
            trusted_keys={signer.key_id: private_key.public_key()}
        )
        provider_calls: list[str] = []
        decision = replace(_selected_lane("claude"), **decision_changes)
        base_spec = dispatch.DispatchSpec(
            repository=repository,
            base_commit="b" * 40,
            branch="jobs/test",
            output_parents=(output,),
            scoped_memory_paths=(memory,),
            requested_lane="claude",
            lane_id="claude-pc-1",
            executor="claude",
            specialist="claude-builder",
            model="claude-opus-5",
        )
        spec = replace(base_spec, **spec_changes)
        tables = (
            "jobs",
            "job_events",
            "job_preflights",
            "job_receipts",
            "job_attempts",
            "job_lane_placements",
        )

        def snapshot():
            return {
                table: [
                    dict(row)
                    for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")
                ]
                for table in tables
            }

        before = snapshot()
        with pytest.raises(jobs_lanes.InvalidLaneDecision):
            dispatch.dispatch_once(
                conn=conn,
                job_id=job_id,
                spec=spec,
                probes=probes,
                signer=signer,
                verifier=verifier,
                worker_id="worker:test",
                executor_registry=executors.registry.with_reliability_adapters(
                    {
                        "claude": lambda _context: provider_calls.append("claude"),
                        "codex": lambda _context: provider_calls.append("codex"),
                    }
                ),
                gate=lambda _context, _execution: None,
                activation_gate=lambda _context, _gate: None,
                observed_at="2026-08-09T00:00:00Z",
                now=1,
                lane_decision=decision,
            )

        assert snapshot() == before
        assert probe_calls == []
        assert provider_calls == []
    finally:
        conn.close()


def test_dispatch_does_not_claim_when_preflight_blocks(tmp_path):
    conn = jdb.connect(tmp_path / "jobs.db")
    try:
        job_id = jdb.create_job(
            conn,
            name="blocked",
            goal="do not launch",
            requested_lane="claude",
        )
        repository = tmp_path / "repo"
        output = tmp_path / "output"
        memory = tmp_path / "memory.md"
        repository.mkdir()
        output.mkdir()
        memory.write_text("scoped", encoding="utf-8")

        def passed(_probe_input):
            return harness.ProbeResult("PASS", "OK", {})

        probes = harness.PreflightProbes(
            **{name: passed for name in harness.CHECK_NAMES}
        ).with_failure("auth_check", code="SSH_AUTH_FAILED")
        private_key = Ed25519PrivateKey.generate()
        signer = harness.ReceiptSigner("lane:test:v1", private_key)
        verifier = graph.ReceiptVerifier(
            trusted_keys={signer.key_id: private_key.public_key()}
        )
        worker_calls = []

        result = dispatch.dispatch_once(
            conn=conn,
            job_id=job_id,
            spec=dispatch.DispatchSpec(
                repository=repository,
                base_commit="b" * 40,
                branch="jobs/test",
                output_parents=(output,),
                scoped_memory_paths=(memory,),
                requested_lane="claude",
                lane_id="lane:test",
                executor="claude",
                specialist="claude-builder",
                model="claude-opus-5",
            ),
            probes=probes,
            signer=signer,
            verifier=verifier,
            worker_id="worker:test",
            executor_registry=executors.registry.with_reliability_adapters(
                {"claude": lambda _context: worker_calls.append("launched")}
            ),
            gate=lambda _context, _execution: None,
            activation_gate=lambda _context, _gate: None,
            observed_at="2026-08-09T00:00:00Z",
            now=1,
        )

        job = jdb.get_job(conn, job_id)
        assert result.claimed is False
        assert result.reason == "SSH_AUTH_FAILED"
        assert job.claimed_by is None
        assert jdb.get_attempts(conn, job_id) == []
        assert worker_calls == []
    finally:
        conn.close()
