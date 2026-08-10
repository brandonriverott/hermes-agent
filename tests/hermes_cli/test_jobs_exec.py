"""Unit tests for the pure Jobs V3 execution contract (hermes_cli/jobs_exec).

Everything here is I/O-free by construction: routing, execution-metadata
validation, worker-outcome classification, receipt sanitization, and the
heartbeat cadence policy. If a test in this file needs a database, a clock, or
a filesystem, the module under test has grown a dependency it must not have.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_exec as jx


# ---------------------------------------------------------------------------
# route() — pure, deterministic, fails closed
# ---------------------------------------------------------------------------


def test_route_unassigned_job_goes_to_the_claude_builder():
    d = jx.route(specialist=None)
    assert d.specialist == "claude-builder"
    assert d.model == "claude-opus-5"
    assert d.effort == "max"
    assert d.executable is True
    assert d.reason


def test_route_named_claude_builder_lands_on_the_same_lane_as_unassigned():
    named = jx.route(specialist="claude-builder")
    unassigned = jx.route(specialist=None)
    assert (named.specialist, named.model, named.effort, named.executable) == (
        unassigned.specialist,
        unassigned.model,
        unassigned.effort,
        unassigned.executable,
    )
    # The lane is the same; the explanation is not, and should not be.
    assert named.reason != unassigned.reason


def test_route_is_deterministic_across_calls():
    assert jx.route(specialist=None) == jx.route(specialist=None)
    assert jx.route(specialist="gpt-builder") == jx.route(specialist="gpt-builder")


def test_route_managed_gpt_is_represented_but_not_executable():
    d = jx.route(specialist="gpt-builder")
    assert d.specialist == "gpt-builder"
    assert d.model == "gpt-5.6-sol"
    assert d.effort in jx.EFFORT_TIERS
    # Represented and tested, unreachable until its adapter exists.
    assert d.executable is False
    assert "adapter" in d.reason.lower()


def test_route_rejects_an_unknown_specialist():
    with pytest.raises(jx.UnsupportedRouting):
        jx.route(specialist="graph-engineer")


def test_route_rejects_a_blank_specialist_rather_than_defaulting():
    # Empty string is a malformed assignment, not "unassigned".
    with pytest.raises(jx.UnsupportedRouting):
        jx.route(specialist="   ")


def test_routing_decision_is_immutable():
    d = jx.route(specialist=None)
    with pytest.raises(Exception):
        d.model = "something-else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validate_execution() — the metadata.execution sub-contract
# ---------------------------------------------------------------------------


BASE = "a" * 40


def _meta(**overrides):
    execution = {
        "model": "claude-opus-5",
        "effort": "max",
        "max_turns": 90,
        "repo_path": "/tmp/repo",
        "base_commit": BASE,
        "workspace_kind": "worktree",
    }
    execution.update(overrides)
    return {"execution": execution}


def test_validate_execution_returns_the_frozen_spec():
    spec = jx.validate_execution(_meta())
    assert spec.model == "claude-opus-5"
    assert spec.effort == "max"
    assert spec.max_turns == 90
    assert spec.repo_path == "/tmp/repo"
    assert spec.base_commit == BASE
    assert spec.workspace_kind == "worktree"
    with pytest.raises(Exception):
        spec.max_turns = 1  # type: ignore[misc]


def test_validate_execution_requires_the_execution_block():
    with pytest.raises(ValueError):
        jx.validate_execution({})
    with pytest.raises(ValueError):
        jx.validate_execution({"execution": "claude"})
    with pytest.raises(ValueError):
        jx.validate_execution(None)


@pytest.mark.parametrize(
    "field", ["model", "effort", "repo_path", "base_commit", "workspace_kind"]
)
def test_validate_execution_rejects_a_missing_required_string(field):
    meta = _meta()
    del meta["execution"][field]
    with pytest.raises(ValueError):
        jx.validate_execution(meta)


@pytest.mark.parametrize(
    "field", ["model", "effort", "repo_path", "base_commit", "workspace_kind"]
)
def test_validate_execution_rejects_a_blank_required_string(field):
    with pytest.raises(ValueError):
        jx.validate_execution(_meta(**{field: "  "}))


@pytest.mark.parametrize("bad", [0, -1, "90", 1.5, True, None])
def test_validate_execution_rejects_a_non_positive_int_max_turns(bad):
    with pytest.raises(ValueError):
        jx.validate_execution(_meta(max_turns=bad))


def test_validate_execution_caps_max_turns():
    with pytest.raises(ValueError):
        jx.validate_execution(_meta(max_turns=jx.MAX_TURNS_CEILING + 1))
    assert jx.validate_execution(_meta(max_turns=jx.MAX_TURNS_CEILING)).max_turns


@pytest.mark.parametrize(
    "bad",
    [
        "HEAD",
        "abc123",              # abbreviated
        "A" * 40,              # upper-case hex is not the canonical form
        "z" * 40,              # not hex
        "a" * 39,
        "a" * 41,
    ],
)
def test_validate_execution_requires_a_full_lowercase_sha(bad):
    with pytest.raises(ValueError):
        jx.validate_execution(_meta(base_commit=bad))


def test_validate_execution_requires_an_absolute_repo_path():
    with pytest.raises(ValueError):
        jx.validate_execution(_meta(repo_path="relative/repo"))


def test_validate_execution_rejects_an_unknown_effort_tier():
    with pytest.raises(ValueError):
        jx.validate_execution(_meta(effort="ludicrous"))


def test_validate_execution_only_supports_the_worktree_workspace_kind():
    with pytest.raises(ValueError):
        jx.validate_execution(_meta(workspace_kind="in_place"))


def test_validate_execution_rejects_unknown_execution_keys():
    # An unrecognised key is a contract drift, not a harmless extra.
    meta = _meta()
    meta["execution"]["sandbox_profile"] = "/tmp/p.sb"
    with pytest.raises(ValueError):
        jx.validate_execution(meta)


# ---------------------------------------------------------------------------
# The routed decision and the approved metadata have to agree
# ---------------------------------------------------------------------------


def test_execution_spec_must_match_the_routed_model():
    spec = jx.validate_execution(_meta(model="gpt-5.6-sol"))
    with pytest.raises(jx.UnsupportedRouting):
        jx.require_agreement(jx.route(specialist=None), spec)


def test_execution_spec_matching_the_routed_model_agrees():
    spec = jx.validate_execution(_meta())
    jx.require_agreement(jx.route(specialist=None), spec)  # no raise


def test_agreement_refuses_a_specialist_with_no_adapter():
    spec = jx.validate_execution(_meta(model="gpt-5.6-sol", effort="high"))
    with pytest.raises(jx.UnsupportedRouting):
        jx.require_agreement(jx.route(specialist="gpt-builder"), spec)


# ---------------------------------------------------------------------------
# classify_worker_result() — fails closed on every contradiction
# ---------------------------------------------------------------------------


def test_classify_clean_success():
    assert jx.classify_worker_result(
        exit_code=0, result={"outcome": "succeeded"}, timed_out=False
    ) == ("succeeded", None)


def test_classify_timeout_is_an_infrastructure_interruption():
    assert jx.classify_worker_result(
        exit_code=None, result={"outcome": "succeeded"}, timed_out=True
    ) == ("interrupted", "infrastructure")


def test_classify_missing_result_file_is_infrastructure():
    assert jx.classify_worker_result(
        exit_code=0, result=None, timed_out=False
    ) == ("failed", "infrastructure")


def test_classify_refuses_success_claimed_on_a_nonzero_exit():
    # The sharpest contradiction: never report a success the process denies.
    assert jx.classify_worker_result(
        exit_code=1, result={"outcome": "succeeded"}, timed_out=False
    ) == ("failed", "infrastructure")


def test_classify_trusts_a_failure_claimed_on_a_zero_exit():
    # The worse of the two signals wins, so a lying exit code cannot pass.
    assert jx.classify_worker_result(
        exit_code=0,
        result={"outcome": "failed", "failure_class": "max_turns"},
        timed_out=False,
    ) == ("failed", "max_turns")


@pytest.mark.parametrize("cls", list(jdb.FAILURE_CLASSES))
def test_classify_passes_through_every_approved_failure_class(cls):
    status, failure_class = jx.classify_worker_result(
        exit_code=1, result={"outcome": "failed", "failure_class": cls}, timed_out=False
    )
    assert status == "failed"
    assert failure_class == cls


def test_classify_downgrades_an_unapproved_failure_class():
    assert jx.classify_worker_result(
        exit_code=1,
        result={"outcome": "failed", "failure_class": "vibes"},
        timed_out=False,
    ) == ("failed", "implementation")


def test_classify_defaults_a_classless_failure_to_implementation():
    assert jx.classify_worker_result(
        exit_code=3, result={"outcome": "failed"}, timed_out=False
    ) == ("failed", "implementation")


@pytest.mark.parametrize("junk", ["succeeded ", "SUCCEEDED", "done", 1, None, {}])
def test_classify_rejects_an_unrecognised_outcome(junk):
    assert jx.classify_worker_result(
        exit_code=0, result={"outcome": junk}, timed_out=False
    ) == ("failed", "infrastructure")


def test_classify_rejects_a_non_mapping_result():
    assert jx.classify_worker_result(
        exit_code=0, result=["succeeded"], timed_out=False
    ) == ("failed", "infrastructure")


def test_every_classification_is_a_terminal_attempt_status():
    for exit_code, result, timed_out in (
        (0, {"outcome": "succeeded"}, False),
        (1, {"outcome": "failed"}, False),
        (None, None, True),
        (0, None, False),
    ):
        status, failure_class = jx.classify_worker_result(
            exit_code=exit_code, result=result, timed_out=timed_out
        )
        assert status in jdb.TERMINAL_ATTEMPT_STATUSES
        assert failure_class is None or failure_class in jdb.FAILURE_CLASSES


# ---------------------------------------------------------------------------
# sanitize_receipt() — secrets never reach durable evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["password", "API_KEY", "db_secret", "auth_token", "Authorization", "x-api-key"],
)
def test_sanitize_rejects_a_secret_like_key_at_the_top_level(key):
    with pytest.raises(jx.UnsafeReceipt):
        jx.sanitize_receipt({key: "value"})


def test_sanitize_rejects_a_secret_like_key_at_any_depth():
    with pytest.raises(jx.UnsafeReceipt):
        jx.sanitize_receipt({"a": {"b": [{"c": {"MY_TOKEN": "x"}}]}})


def test_sanitize_keeps_ordinary_evidence_intact():
    data = {
        "schema_version": 1,
        "outcome": "succeeded",
        "commit": "b" * 40,
        "diff_empty": False,
        "files": ["a.py", "b.py"],
        "nested": {"turns": 3},
    }
    assert jx.sanitize_receipt(data) == data


@pytest.mark.parametrize(
    "text",
    [
        "sk-" + "ant-api03-" + "A" * 30,
        "ghp_" + "A" * 36,
        "github_pat_" + "A" * 30,
        "AKIA" + "IOSFODNN7EXAMPLE",
        "xoxb-" + "1" * 10 + "-" + "2" * 10 + "-" + "a" * 24,
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345",
        "OPENAI_API_KEY=" + "sk-proj-" + "abcdefghijklmnop",
    ],
)
def test_sanitize_redacts_secret_shaped_text(text):
    out = jx.sanitize_receipt({"log": text})
    assert jx.REDACTED in out["log"]
    for fragment in ("sk-" + "ant-api03-" + "A" * 30, "ghp_" + "A" * 36, "AKIA" + "IOSFODNN7EXAMPLE", "xoxb-" + "1" * 10 + "-" + "2" * 10 + "-" + "a" * 24):
        assert fragment not in out["log"]


def test_sanitize_redacts_inside_nested_lists_and_dicts():
    out = jx.sanitize_receipt({"tail": [{"line": "ghp_" + "A" * 36 + "B" * 36}]})
    assert out["tail"][0]["line"] == jx.REDACTED


def test_sanitize_truncates_a_long_log_tail():
    out = jx.sanitize_receipt({"stdout": "x" * (jx.MAX_TEXT_CHARS + 500)})
    assert len(out["stdout"]) <= jx.MAX_TEXT_CHARS + len(jx.TRUNCATED)
    assert out["stdout"].endswith(jx.TRUNCATED)


def test_sanitize_rejects_an_oversized_receipt():
    with pytest.raises(jx.UnsafeReceipt):
        jx.sanitize_receipt({f"k{i}": "y" * 900 for i in range(200)})


def test_sanitize_requires_a_json_object():
    with pytest.raises(jx.UnsafeReceipt):
        jx.sanitize_receipt(["not", "an", "object"])


def test_sanitize_rejects_a_value_that_cannot_be_serialized():
    with pytest.raises(jx.UnsafeReceipt):
        jx.sanitize_receipt({"when": {1, 2, 3}})


def test_sanitize_rejects_a_non_string_key():
    with pytest.raises(jx.UnsafeReceipt):
        jx.sanitize_receipt({1: "one"})


def test_sanitize_does_not_mutate_its_input():
    data = {"log": "ghp_" + "A" * 36 + "C" * 36, "nested": {"n": 1}}
    before = {"log": data["log"], "nested": dict(data["nested"])}
    jx.sanitize_receipt(data)
    assert data == before


# ---------------------------------------------------------------------------
# heartbeat cadence — alive without flooding the ledger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lease", [60, 120, 600, 1800, jdb.MAX_LEASE_SECONDS])
def test_heartbeat_interval_beats_well_before_half_the_lease(lease):
    interval = jx.heartbeat_interval(lease)
    assert 0 < interval <= lease / 2


@pytest.mark.parametrize("lease", [60, 120, 600, 1800, jdb.MAX_LEASE_SECONDS])
def test_heartbeat_interval_bounds_the_events_one_lease_can_produce(lease):
    # The ledger-flooding guard: a lease can never buy more than a handful of
    # claim_heartbeat rows, no matter how tightly the adapter polls.
    assert lease / jx.heartbeat_interval(lease) <= jx.MAX_BEATS_PER_LEASE


def test_heartbeat_interval_never_drops_below_the_floor():
    # Even an absurdly short lease cannot make the adapter beat every second.
    assert jx.heartbeat_interval(1) >= jx.MIN_HEARTBEAT_SECONDS
    assert jx.heartbeat_interval(4) >= jx.MIN_HEARTBEAT_SECONDS


def test_lease_for_covers_the_wall_clock_plus_margin_and_stays_legal():
    lease = jx.lease_for(wall_clock_seconds=1500)
    assert lease > 1500
    assert lease <= jdb.MAX_LEASE_SECONDS


def test_lease_for_clamps_to_the_core_maximum():
    assert jx.lease_for(wall_clock_seconds=99999) == jdb.MAX_LEASE_SECONDS


def test_lease_for_rejects_a_non_positive_wall_clock():
    with pytest.raises(ValueError):
        jx.lease_for(wall_clock_seconds=0)


# ---------------------------------------------------------------------------
# Standard JSON only — NaN and Infinity are not JSON values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_never_reaches_a_receipt(bad):
    with pytest.raises(jx.UnsafeReceipt):
        jx.sanitize_receipt({"duration_seconds": bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_number_is_refused_at_any_depth(bad):
    with pytest.raises(jx.UnsafeReceipt):
        jx.sanitize_receipt({"observed": {"samples": [1, {"x": bad}]}})


def test_a_sanitized_receipt_always_encodes_as_standard_json():
    clean = jx.sanitize_receipt({"a": 1, "b": [1.5, None, "x"], "c": {"d": True}})
    json.dumps(clean, allow_nan=False)  # no raise


@pytest.mark.parametrize(
    "text", ['{"x": NaN}', '{"x": Infinity}', '{"x": [-Infinity]}',
             '{"x": {"y": NaN}}'],
)
def test_loads_strict_refuses_javascript_style_constants(text):
    with pytest.raises(ValueError):
        jx.loads_strict(text)


def test_loads_strict_accepts_ordinary_json():
    assert jx.loads_strict('{"x": [1, 2.5, null, true]}') == {
        "x": [1, 2.5, None, True]
    }


# ---------------------------------------------------------------------------
# Redaction without the receipt's text cap — preserved files need both halves
# ---------------------------------------------------------------------------


def test_redact_secrets_scrubs_without_capping_length():
    body = "an ordinary log line\n" * 5000
    assert len(jx.redact_secrets(body)) == len(body)


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdefghijklmnop",
        "ghp_" + "A" * 36,
        "AKIA" + "IOSFODNN7EXAMPLE",
        "api_key=hunter2",
        "Authorization: Bearer abcdefghijkl",
    ],
)
def test_redact_secrets_removes_every_secret_shape(secret):
    assert secret not in jx.redact_secrets(f"prefix {secret} suffix")


def test_redact_text_still_caps_what_redact_secrets_leaves():
    capped = jx.redact_text("x" * (jx.MAX_TEXT_CHARS + 500))
    assert capped.endswith(jx.TRUNCATED)
    assert len(capped) == jx.MAX_TEXT_CHARS + len(jx.TRUNCATED)
