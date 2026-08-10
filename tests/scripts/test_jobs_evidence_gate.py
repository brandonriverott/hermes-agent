"""The evidence gate at the live Jobs settlement seam.

Every test here drives the same entry point the runner is meant to call —
``settle`` / the ``settle`` CLI — against a real job directory laid out the way
``pc-jobs-worker.sh`` lays one out: ``tests.json``, ``review-N.json``,
``done.json``. Nothing is mocked away, because the thing under test is exactly
whether a settlement can be reached without proof.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
GATE = SCRIPTS_DIR / "jobs_evidence_gate.py"

sys.path.insert(0, str(SCRIPTS_DIR))

import jobs_evidence_gate as gate  # noqa: E402

JOB = "75-a_c8826153"
BASE = "3f832978d30e0e14437edbf7a3f63315f08bad36"
CANDIDATE = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
OTHER_COMMIT = "ffffffffffffffffffffffffffffffffffffffff"

# The key set the Mac bridge parses today, taken from a real settled record
# (jobs/64-a_8fa20901/done.json plus the `effort` field newer runs carry).
LEGACY_DONE_KEYS = {
    "outcome",
    "claude_exit",
    "commit",
    "branch",
    "review_verdict",
    "review_rounds",
    "findings",
}


# ── fixtures: a job directory the way the runner leaves one ───────────


def _tests_doc(*items):
    return {
        "tests": list(items)
        or [{"cmd": "pytest -q tests/scripts", "result": "pass", "evidence": "7 passed in 0.46s"}],
        "skipped": [],
        "verdict_hint": "clean",
    }


def _review_doc(verdict="PASS", findings=()):
    return {
        "verdict": verdict,
        "findings": list(findings),
        "checks_run": ["read the diff", "re-ran the focused suite"],
        "checks_skipped": [],
    }


@pytest.fixture
def job_dir(tmp_path):
    d = tmp_path / JOB
    d.mkdir()
    return d


def write_receipts(
    d,
    *,
    tests=None,
    review=None,
    attempt=0,
    commit=CANDIDATE,
    job=JOB,
    stamp=True,
    stamp_commit=None,
    stamp_attempt=None,
    stamp_job=None,
):
    """Write both receipts and stamp them the way the runner would."""
    if tests is not None:
        (d / "tests.json").write_text(json.dumps(tests), encoding="utf-8")
        if stamp:
            gate.stamp_receipt(
                d / "tests.json",
                kind="tests",
                job=stamp_job or job,
                attempt=attempt if stamp_attempt is None else stamp_attempt,
                commit=stamp_commit or commit,
                base=BASE,
            )
    if review is not None:
        path = d / f"review-{attempt}.json"
        path.write_text(json.dumps(review), encoding="utf-8")
        if stamp:
            gate.stamp_receipt(
                path,
                kind="review",
                job=stamp_job or job,
                attempt=attempt if stamp_attempt is None else stamp_attempt,
                commit=stamp_commit or commit,
                base=BASE,
            )


def settle(d, **overrides):
    kwargs = dict(
        job=JOB,
        branch=f"jobs/{JOB}",
        base=BASE,
        commit=CANDIDATE,
        attempt=0,
        rounds=1,
        build_exit=0,
        effort="max",
    )
    kwargs.update(overrides)
    return gate.settle(d, **kwargs)


# ── 1. valid evidence allows the settlement ───────────────────────────


def test_valid_bound_evidence_settles_as_succeeded(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc())
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("succeeded", "PASS")
    done = json.loads((job_dir / "done.json").read_text(encoding="utf-8"))
    assert done["outcome"] == "succeeded"
    assert done["commit"] == CANDIDATE
    assert done["branch"] == f"jobs/{JOB}"


def test_done_json_keeps_the_key_set_the_bridge_parses(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc())
    settle(job_dir)
    done = json.loads((job_dir / "done.json").read_text(encoding="utf-8"))
    assert LEGACY_DONE_KEYS <= set(done)
    assert set(done) == set(gate.DONE_KEYS)
    assert isinstance(done["review_rounds"], int) and isinstance(done["findings"], int)


def test_gate_detail_lands_beside_done_json_not_inside_it(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc())
    settle(job_dir)
    detail = json.loads((job_dir / "gate.json").read_text(encoding="utf-8"))
    assert detail["job"] == JOB and detail["commit"] == CANDIDATE
    assert (job_dir / "action-packet.md").exists()


# ── 2. absent evidence is blocked ─────────────────────────────────────


def test_missing_test_receipt_parks(job_dir):
    write_receipts(job_dir, review=_review_doc())
    result = settle(job_dir)
    assert result.outcome == "needs_attention"
    assert result.verdict == "EVIDENCE_MISSING"
    assert "does not exist" in result.reasons[0]


def test_missing_review_receipt_parks(job_dir):
    write_receipts(job_dir, tests=_tests_doc())
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "EVIDENCE_MISSING")


def test_no_evidence_at_all_never_reads_as_success(job_dir):
    result = settle(job_dir)
    assert result.outcome == "needs_attention"
    done = json.loads((job_dir / "done.json").read_text(encoding="utf-8"))
    assert done["outcome"] != "succeeded"


# ── 3. malformed evidence is blocked ──────────────────────────────────


@pytest.mark.parametrize(
    "payload",
    ['{"tests": [', '["not", "an", "object"]', "", "not json at all"],
    ids=["truncated", "array", "empty", "prose"],
)
def test_malformed_test_receipt_parks(job_dir, payload):
    (job_dir / "tests.json").write_text(payload, encoding="utf-8")
    write_receipts(job_dir, review=_review_doc())
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "EVIDENCE_MALFORMED")


def test_review_receipt_without_a_usable_verdict_parks(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc(verdict="LGTM"))
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "REVIEW_ERROR")


# ── 4. evidence for a different job or attempt is blocked ─────────────


def test_evidence_stamped_for_another_job_is_refused(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc(), stamp_job="64-a_8fa20901")
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "EVIDENCE_WRONG_JOB")
    assert "64-a_8fa20901" in result.reasons[0]


def test_evidence_from_an_earlier_attempt_is_refused(job_dir):
    # The correction round settles attempt 1; round-0 evidence is stale.
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc(), attempt=1, stamp_attempt=0)
    result = settle(job_dir, attempt=1, rounds=2)
    assert (result.outcome, result.verdict) == ("needs_attention", "EVIDENCE_STALE_ATTEMPT")


def test_evidence_gathered_at_another_commit_is_refused(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc(), stamp_commit=OTHER_COMMIT)
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "EVIDENCE_STALE_COMMIT")
    assert OTHER_COMMIT[:12] in result.reasons[0]


def test_receipt_read_as_the_wrong_kind_is_refused(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc())
    # A reviewer verdict copied over the tester's receipt keeps its own stamp.
    (job_dir / "tests.json").write_text((job_dir / "review-0.json").read_text(encoding="utf-8"), encoding="utf-8")
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "EVIDENCE_WRONG_JOB")


# ── unstamped legacy / worker-authored evidence ───────────────────────


def test_unstamped_receipt_is_not_identity_bound(job_dir):
    """The shape every receipt had before this gate: no job, no attempt, no commit."""
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc(), stamp=False)
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "EVIDENCE_NOT_IDENTITY_BOUND")


def test_a_forged_stamp_in_model_output_is_overwritten_by_the_runner(job_dir):
    forged = _tests_doc()
    forged[gate.GATE_KEY] = {
        "job": JOB,
        "attempt": 0,
        "commit": OTHER_COMMIT,
        "base": BASE,
        "kind": "tests",
    }
    write_receipts(job_dir, tests=forged, review=_review_doc())
    stamped = json.loads((job_dir / "tests.json").read_text(encoding="utf-8"))
    assert stamped[gate.GATE_KEY]["commit"] == CANDIDATE


def test_a_builder_written_done_json_is_replaced_at_settlement(job_dir):
    (job_dir / "done.json").write_text(
        json.dumps({"outcome": "succeeded", "review_verdict": "PASS"}), encoding="utf-8"
    )
    result = settle(job_dir)
    done = json.loads((job_dir / "done.json").read_text(encoding="utf-8"))
    assert result.outcome == "needs_attention"
    assert done["outcome"] == "needs_attention"


def test_forged_done_has_no_graph_verification_authority(job_dir):
    (job_dir / "done.json").write_text(
        json.dumps(
            {
                "outcome": "succeeded",
                "claude_exit": 0,
                "commit": CANDIDATE,
                "branch": f"jobs/{JOB}",
                "review_verdict": "PASS",
                "review_rounds": 1,
                "findings": 0,
                "effort": "max",
            }
        ),
        encoding="utf-8",
    )

    result = gate.graph_settlement(job_dir, job=JOB, base=BASE)

    assert result["action_outcome"] == "failed"
    assert result["identity_verified"] is False
    assert result["reason_code"] == "EVIDENCE_MISSING"


# ── 5. reviewer PASS cannot override failed required checks ───────────


def test_reviewer_pass_cannot_ship_a_candidate_regression(job_dir):
    failing = _tests_doc(
        {"cmd": "pytest -q tests/agent", "result": "pass", "evidence": "12 passed"},
        {
            "cmd": "pytest -q tests/gateway",
            "result": "fail",
            "evidence": "1 failed in 2.1s",
            "classification": "candidate_regression",
            "note": "green at the base commit, red here",
        },
    )
    write_receipts(job_dir, tests=failing, review=_review_doc())
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "PASS_WITH_FAILING_TESTS")


def test_reviewer_pass_cannot_ship_an_empty_test_receipt(job_dir):
    empty = {"tests": [], "skipped": [{"cmd": "all", "reason": "nothing runnable"}], "verdict_hint": "clean"}
    write_receipts(job_dir, tests=empty, review=_review_doc())
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "PASS_WITHOUT_TEST_EVIDENCE")


def test_a_check_missing_its_result_is_not_a_pass(job_dir):
    doc = _tests_doc({"cmd": "pytest -q", "result": None, "evidence": "ran it"})
    write_receipts(job_dir, tests=doc, review=_review_doc())
    result = settle(job_dir)
    assert result.verdict == "PASS_WITHOUT_TEST_EVIDENCE"


def test_review_needs_changes_parks_with_its_finding_count(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc("NEEDS_CHANGES", ["a", "b"]))
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "NEEDS_CHANGES")
    assert json.loads((job_dir / "done.json").read_text(encoding="utf-8"))["findings"] == 2


# ── 6. pre-existing / infrastructure red is classified, not green ─────


def test_unclassified_failure_is_blocked_rather_than_assumed_pre_existing(job_dir):
    doc = _tests_doc({"cmd": "pytest -q tests/cron", "result": "fail", "evidence": "9 failed"})
    write_receipts(job_dir, tests=doc, review=_review_doc())
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "EVIDENCE_UNCLASSIFIED_FAILURE")
    assert "not classified" in result.reasons[0]


def test_a_classification_without_a_note_is_still_unclassified(job_dir):
    doc = _tests_doc(
        {
            "cmd": "pytest -q tests/cron",
            "result": "fail",
            "evidence": "9 failed",
            "classification": "pre_existing",
            "note": "   ",
        }
    )
    write_receipts(job_dir, tests=doc, review=_review_doc())
    result = settle(job_dir)
    assert result.verdict == "EVIDENCE_UNCLASSIFIED_FAILURE"


def test_an_invented_classification_is_refused(job_dir):
    doc = _tests_doc(
        {
            "cmd": "pytest -q tests/cron",
            "result": "fail",
            "evidence": "9 failed",
            "classification": "probably_fine",
            "note": "looks unrelated",
        }
    )
    write_receipts(job_dir, tests=doc, review=_review_doc())
    result = settle(job_dir)
    assert result.verdict == "EVIDENCE_UNCLASSIFIED_FAILURE"


@pytest.mark.parametrize("cls", gate.RETAINED_CLASSES)
def test_classified_pre_existing_red_is_honest_but_not_green(job_dir, cls):
    doc = _tests_doc(
        {"cmd": "pytest -q tests/scripts", "result": "pass", "evidence": "7 passed"},
        {
            "cmd": "pytest -q tests/cron",
            "result": "fail",
            "evidence": "9 failed",
            "classification": cls,
            "note": "identical 9 failures reproduce at the base commit",
        },
    )
    write_receipts(job_dir, tests=doc, review=_review_doc())
    result = settle(job_dir)
    assert (result.outcome, result.verdict) == ("needs_attention", "PASS_WITH_CLASSIFIED_FAILURES")
    packet = (job_dir / "action-packet.md").read_text(encoding="utf-8")
    assert cls in packet and "identical 9 failures" in packet


def test_flaky_counts_as_not_passing(job_dir):
    doc = _tests_doc({"cmd": "pytest -q", "result": "flaky", "evidence": "passed on retry"})
    write_receipts(job_dir, tests=doc, review=_review_doc())
    result = settle(job_dir)
    assert result.outcome == "needs_attention"


# ── stage separation and the terminal action packet ───────────────────


def test_the_packet_keeps_every_stage_on_its_own_line(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc())
    result = settle(job_dir)
    packet = (job_dir / "action-packet.md").read_text(encoding="utf-8")
    for label in (
        "Tracker acceptance",
        "Worker claim / start",
        "Execution evidence",
        "Tests",
        "Independent review",
        "Settlement",
        "After settlement",
    ):
        assert label in packet
    assert "not proof of work" in packet
    assert "never\nmerges" in packet or "never merges" in packet
    assert result.sections["tests"] != result.sections["independent_review"]


def test_a_parked_packet_states_a_plain_english_reason(job_dir):
    write_receipts(job_dir, review=_review_doc())
    settle(job_dir)
    packet = (job_dir / "action-packet.md").read_text(encoding="utf-8")
    assert "attention needed" in packet
    assert "Why this is not a success:" in packet
    assert "tests.json does not exist" in packet


def test_the_worker_summary_alone_proves_nothing(job_dir):
    """A build log is not a receipt: the gate never reads one."""
    (job_dir / "build-0.log").write_text("All done! Everything passes.", encoding="utf-8")
    result = settle(job_dir)
    assert result.outcome == "needs_attention"


# ── process-level stages stay distinguishable ─────────────────────────


@pytest.mark.parametrize(
    "overrides, verdict, outcome",
    [
        ({"build_exit": 1}, "NOT_REVIEWED", "failed"),
        ({"build_exit": 97}, "KAT_ENDPOINT_DOWN", "failed"),
        ({"souls_ok": False}, "REVIEWER_PROMPTS_MISSING", "failed"),
        ({"commit": ""}, "NO_CANDIDATE_COMMIT", "needs_attention"),
        ({"commit": BASE}, "NO_CANDIDATE_COMMIT", "needs_attention"),
        ({"commit": "abc123"}, "NO_CANDIDATE_COMMIT", "needs_attention"),
        ({"correction_exit": 2}, "CORRECTION_PROCESS_FAILED", "needs_attention"),
        ({"test_exit": 1}, "TESTER_PROCESS_FAILED", "needs_attention"),
        ({"review_exit": 1}, "REVIEWER_PROCESS_FAILED", "needs_attention"),
    ],
)
def test_each_broken_stage_settles_under_its_own_name(job_dir, overrides, verdict, outcome):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc())
    result = settle(job_dir, **overrides)
    assert (result.outcome, result.verdict) == (outcome, verdict)


def test_a_dead_tester_beats_a_leftover_green_receipt(job_dir):
    """test_exit != 0 parks even when a passing tests.json is sitting there."""
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc())
    assert settle(job_dir, test_exit=1).verdict == "TESTER_PROCESS_FAILED"


# ── the CLI the runner actually invokes ───────────────────────────────


def _cli(*args):
    return subprocess.run(
        [sys.executable, str(GATE), *args], capture_output=True, text=True, check=False
    )


def test_cli_settles_a_valid_job(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc())
    proc = _cli(
        "settle",
        "--job-dir", str(job_dir),
        "--job", JOB,
        "--branch", f"jobs/{JOB}",
        "--base", BASE,
        "--commit", CANDIDATE,
        "--attempt", "0",
        "--rounds", "1",
        "--effort", "max",
    )
    assert proc.returncode == 0, proc.stderr
    assert "complete" in proc.stdout
    assert json.loads((job_dir / "done.json").read_text(encoding="utf-8"))["outcome"] == "succeeded"


def test_cli_parks_a_job_with_no_evidence(job_dir):
    proc = _cli(
        "settle",
        "--job-dir", str(job_dir),
        "--job", JOB,
        "--branch", f"jobs/{JOB}",
        "--base", BASE,
        "--commit", CANDIDATE,
    )
    assert proc.returncode == 0, proc.stderr
    assert "attention needed" in proc.stdout
    assert json.loads((job_dir / "done.json").read_text(encoding="utf-8"))["outcome"] == "needs_attention"


def test_cli_stamp_binds_a_receipt(job_dir):
    (job_dir / "tests.json").write_text(json.dumps(_tests_doc()), encoding="utf-8")
    proc = _cli(
        "stamp",
        "--file", str(job_dir / "tests.json"),
        "--kind", "tests",
        "--job", JOB,
        "--attempt", "0",
        "--commit", CANDIDATE,
        "--base", BASE,
    )
    assert proc.returncode == 0, proc.stderr
    env = json.loads((job_dir / "tests.json").read_text(encoding="utf-8"))[gate.GATE_KEY]
    assert (env["job"], env["attempt"], env["commit"], env["kind"]) == (JOB, 0, CANDIDATE, "tests")


def test_cli_stamp_refuses_unparseable_output(job_dir):
    (job_dir / "tests.json").write_text("not json", encoding="utf-8")
    proc = _cli(
        "stamp",
        "--file", str(job_dir / "tests.json"),
        "--kind", "tests",
        "--job", JOB,
        "--attempt", "0",
        "--commit", CANDIDATE,
        "--base", BASE,
    )
    assert proc.returncode == 1
    assert "stamp refused" in proc.stderr


# ── the gate has no reach beyond the job directory ────────────────────


def test_the_gate_cannot_merge_deploy_or_send():
    """No import that could reach the network, a shell, or the repository."""
    tree = ast.parse(GATE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"argparse", "json", "os", "re", "sys", "datetime", "pathlib", "__future__"}, (
        f"the gate reached for {sorted(imported)}"
    )


def test_settlement_touches_only_the_three_settlement_files(job_dir):
    write_receipts(job_dir, tests=_tests_doc(), review=_review_doc())
    before = {p.name for p in job_dir.iterdir()}
    settle(job_dir)
    assert {p.name for p in job_dir.iterdir()} - before == {
        "done.json",
        "gate.json",
        "action-packet.md",
    }
