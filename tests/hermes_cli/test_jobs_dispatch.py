"""Provider-free tests for the Jobs dispatch control plane.

The first regression is Job 71, which the live shell dispatcher silently skips
because its valid repository path contains spaces.  These tests deliberately
exercise only the durable parsing contract; no Jobs database, provider, worker,
or live repository is touched.
"""

from __future__ import annotations

import pytest

from hermes_cli import jobs_dispatch as dispatch


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


def test_legacy_header_defaults_are_explicit_and_stable():
    header = dispatch.parse_legacy_execution_header(
        f"REPO_PATH={JOB_71_REPO_PATH}\nBASE_COMMIT={JOB_71_BASE_COMMIT}\n"
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
            "SECRET_MODE",
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
        f"BASE_COMMIT={JOB_71_BASE_COMMIT}\n\n"
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
            f"REPO_PATH=/repo\nBASE_COMMIT={JOB_71_BASE_COMMIT}\nMAX_TURNS=zero",
            "invalid_integer",
            "MAX_TURNS",
        ),
        (
            f"REPO_PATH=/repo\nBASE_COMMIT={JOB_71_BASE_COMMIT}\nMAX_TURNS=501",
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
