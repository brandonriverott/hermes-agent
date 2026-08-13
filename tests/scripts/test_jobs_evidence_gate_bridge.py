"""Regression tests for the Jobs gate -> PC bridge -> action packet seam.

These tests copy the bridge that actually executes on Brandon's Mac and drive
its local-fallback path with a harmless stub worker.  The stub writes the same
job-directory artifacts as the GPU worker; no network, live Jobs database,
business system, or third-party account is touched.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
GATE = SCRIPTS_DIR / "jobs_evidence_gate.py"
BRIDGE_PATCH = SCRIPTS_DIR / "jobs_evidence_gate.bridge.patch"
LIVE_BRIDGE = Path(
    os.environ.get("JOBS_PC_BRIDGE")
    or "/Users/brandon/.hermes/scripts/jobs-pc-bridge.sh"
)

sys.path.insert(0, str(SCRIPTS_DIR))
import jobs_evidence_gate as gate  # noqa: E402


JOB = "99-a_bridgecanary"
BASE = "1" * 40
CANDIDATE = "2" * 40
GREEN_TESTS = {
    "tests": [
        {
            "cmd": "pytest -q tests/scripts",
            "result": "pass",
            "evidence": "4 passed in 0.10s",
        }
    ],
    "skipped": [],
    "verdict_hint": "clean",
}
PASS_REVIEW = {
    "verdict": "PASS",
    "findings": [],
    "checks_run": ["read the diff"],
    "checks_skipped": [],
}


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")


def _write_receipts(
    job_dir: Path,
    *,
    tests: dict = GREEN_TESTS,
    review: dict = PASS_REVIEW,
    job: str = JOB,
    attempt: int = 0,
    commit: str = CANDIDATE,
    base: str = BASE,
) -> None:
    _write_json(job_dir / "tests.json", tests)
    gate.stamp_receipt(
        job_dir / "tests.json",
        kind="tests",
        job=job,
        attempt=attempt,
        commit=commit,
        base=base,
    )
    _write_json(job_dir / f"review-{attempt}.json", review)
    gate.stamp_receipt(
        job_dir / f"review-{attempt}.json",
        kind="review",
        job=job,
        attempt=attempt,
        commit=commit,
        base=base,
    )


def _settle(
    job_dir: Path,
    *,
    tests: dict = GREEN_TESTS,
    review: dict = PASS_REVIEW,
    commit: str = CANDIDATE,
    base: str = BASE,
) -> None:
    _write_receipts(
        job_dir,
        tests=tests,
        review=review,
        commit=commit,
        base=base,
    )
    gate.settle(
        job_dir,
        job=JOB,
        branch=f"jobs/{JOB}",
        base=base,
        commit=commit,
        attempt=0,
        rounds=1,
        build_exit=0,
        effort="max",
    )


def _verify(job_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(GATE),
            "bridge-verify",
            "--job-dir",
            str(job_dir),
            "--job",
            JOB,
            "--base",
            BASE,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_forged_green_done_without_gate_cannot_make_a_success_action(tmp_path):
    job_dir = tmp_path / JOB
    job_dir.mkdir()
    _write_json(
        job_dir / "done.json",
        {
            "outcome": "succeeded",
            "claude_exit": 0,
            "commit": CANDIDATE,
            "branch": f"jobs/{JOB}",
            "review_verdict": "PASS",
            "review_rounds": 1,
            "findings": 0,
            "effort": "max",
        },
    )

    proc = _verify(job_dir)

    assert proc.returncode != 0
    assert "EVIDENCE_MISSING" in proc.stderr
    assert "gate.json does not exist" in proc.stderr


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda _doc: "{not json", "EVIDENCE_MALFORMED"),
        (
            lambda doc: json.dumps({**doc, "job": "98-a_other"}),
            "EVIDENCE_WRONG_JOB",
        ),
        (
            lambda doc: json.dumps({**doc, "attempt": 1}),
            "EVIDENCE_STALE_ATTEMPT",
        ),
        (
            lambda doc: json.dumps({**doc, "commit": "f" * 40}),
            "EVIDENCE_STALE_COMMIT",
        ),
    ],
    ids=["malformed", "wrong-job", "wrong-attempt", "wrong-commit"],
)
def test_malformed_or_identity_mismatched_gate_cannot_make_success(
    tmp_path, mutation, expected
):
    job_dir = tmp_path / JOB
    job_dir.mkdir()
    _settle(job_dir)
    original = json.loads((job_dir / "gate.json").read_text(encoding="utf-8"))
    (job_dir / "gate.json").write_text(mutation(original), encoding="utf-8")

    proc = _verify(job_dir)

    assert proc.returncode != 0
    assert expected in proc.stderr


def test_valid_identity_bound_gate_produces_success_action(tmp_path):
    job_dir = tmp_path / JOB
    job_dir.mkdir()
    _settle(job_dir)

    proc = _verify(job_dir)

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["action_outcome"] == "succeeded"
    assert result["gate_outcome"] == "succeeded"
    assert result["commit"] == CANDIDATE
    assert result["review_verdict"] == "PASS"
    assert result["reason"] == "all required checks passed and review returned PASS"


def test_failed_required_check_stays_blocked_when_review_says_pass(tmp_path):
    job_dir = tmp_path / JOB
    job_dir.mkdir()
    failing = {
        "tests": [
            {
                "cmd": "pytest -q tests/scripts",
                "result": "fail",
                "evidence": "1 failed in 0.10s",
                "classification": "candidate_regression",
                "note": "green at the base commit, red at the candidate",
            }
        ],
        "skipped": [],
        "verdict_hint": "failures",
    }
    _settle(job_dir, tests=failing, review=PASS_REVIEW)

    proc = _verify(job_dir)

    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["action_outcome"] == "failed"
    assert result["gate_outcome"] == "needs_attention"
    assert result["review_verdict"] == "PASS_WITH_FAILING_TESTS"
    assert "fails because of this candidate" in result["reason"]


def test_bridge_rechecks_claimed_success_when_required_check_turns_red(tmp_path):
    job_dir = tmp_path / JOB
    job_dir.mkdir()
    _settle(job_dir)
    failing = {
        "tests": [
            {
                "cmd": "pytest -q tests/scripts",
                "result": "fail",
                "evidence": "1 failed in 0.10s",
                "classification": "candidate_regression",
                "note": "green at the base commit, red at the candidate",
            }
        ],
        "skipped": [],
        "verdict_hint": "failures",
    }
    _write_json(job_dir / "tests.json", failing)
    gate.stamp_receipt(
        job_dir / "tests.json",
        kind="tests",
        job=JOB,
        attempt=0,
        commit=CANDIDATE,
        base=BASE,
    )

    proc = _verify(job_dir)

    assert proc.returncode != 0
    assert "PASS_WITH_FAILING_TESTS" in proc.stderr
    assert "fails because of this candidate" in proc.stderr


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
    )
    return proc.stdout.strip()


@pytest.fixture
def bridge_rig(tmp_path):
    if not LIVE_BRIDGE.is_file():
        pytest.skip(f"live bridge is not installed at {LIVE_BRIDGE}")

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "base",
    )
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "candidate")
    (repo / "seed.txt").write_text("seed\ncandidate\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "-m",
        "candidate",
    )
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")

    local_jobs = tmp_path / "jobs"
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    profiles = tmp_path / "profiles"
    for seat in ("vulcan", "themis"):
        (profiles / seat).mkdir(parents=True)
        (profiles / seat / "SOUL.md").write_text(f"{seat} test prompt\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    ssh.chmod(0o755)

    worker = tmp_path / "stub-worker.sh"
    worker.write_text(
        "#!/usr/bin/env bash\n"
        'key="$2"; repo="$3"\n'
        'job_dir="$JOBS_ROOT/$key"\n'
        'mkdir -p "$job_dir"\n'
        'cp "$BRIDGE_FIXTURE_DIR"/* "$job_dir"/ 2>/dev/null || true\n'
        'git -C "$repo" branch -f "jobs/$key" "$BRIDGE_CANDIDATE"\n'
        'echo RAN_SYNC\n',
        encoding="utf-8",
    )
    worker.chmod(0o755)

    bridge = tmp_path / "jobs-pc-bridge.sh"
    shutil.copy2(LIVE_BRIDGE, bridge)
    if BRIDGE_PATCH.is_file() and os.environ.get("BRIDGE_TEST_UNPATCHED") != "1":
        check = subprocess.run(
            ["git", "apply", "--check", "-p1", str(BRIDGE_PATCH)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            subprocess.run(
                ["git", "apply", "-p1", str(BRIDGE_PATCH)],
                cwd=tmp_path,
                check=True,
                capture_output=True,
                text=True,
            )
    source = bridge.read_text(encoding="utf-8")
    source = source.replace(
        'LOCAL_JOBS_ROOT="$HOME/.hermes/jobs-local"',
        f'LOCAL_JOBS_ROOT="{local_jobs}"',
    )
    source = source.replace(
        'WORKER_LOCAL="$HOME/.hermes/scripts/pc-jobs-worker.sh"',
        f'WORKER_LOCAL="{worker}"',
    )
    source = source.replace(
        'PROFILE_ROOT="/Users/brandon/.hermes/profiles"',
        f'PROFILE_ROOT="{profiles}"',
    )
    source = source.replace(
        'VAULT_PROPOSALS="/Users/brandon/Desktop/Tea Time Hub Vault/_inbox/proposals"',
        f'VAULT_PROPOSALS="{tmp_path / "no-vault"}"',
    )
    bridge.write_text(source, encoding="utf-8")
    bridge.chmod(0o755)

    goal = tmp_path / "goal.md"
    goal.write_text("Harmless bridge canary.\n", encoding="utf-8")
    return {
        "bridge": bridge,
        "repo": repo,
        "base": base,
        "candidate": candidate,
        "fixture_dir": fixture_dir,
        "goal": goal,
        "bin_dir": bin_dir,
        "tmp_path": tmp_path,
    }


def _run_bridge(rig) -> dict:
    result = rig["tmp_path"] / "result.json"
    env = {
        **os.environ,
        "PATH": f"{rig['bin_dir']}{os.pathsep}{os.environ['PATH']}",
        "HERMES_JOB_NUMBER": "99",
        "HERMES_ATTEMPT_ID": "a_bridgecanary",
        "BRIDGE_FIXTURE_DIR": str(rig["fixture_dir"]),
        "BRIDGE_CANDIDATE": rig["candidate"],
        "JOBS_BRIDGE_GATE": os.environ.get("JOBS_BRIDGE_GATE") or str(GATE),
    }
    proc = subprocess.run(
        [
            "bash",
            str(rig["bridge"]),
            "--worktree",
            str(rig["repo"]),
            "--goal-file",
            str(rig["goal"]),
            "--result-file",
            str(result),
            "--model",
            "claude-opus-5",
            "--effort",
            "max",
            "--max-turns",
            "5",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert result.is_file(), f"no result.json\nstdout={proc.stdout}\nstderr={proc.stderr}"
    return json.loads(result.read_text(encoding="utf-8"))


def test_real_bridge_rejects_forged_done_without_valid_gate(bridge_rig):
    _write_json(
        bridge_rig["fixture_dir"] / "done.json",
        {
            "outcome": "succeeded",
            "claude_exit": 0,
            "commit": bridge_rig["candidate"],
            "branch": f"jobs/{JOB}",
            "review_verdict": "PASS",
            "review_rounds": 1,
            "findings": 0,
            "effort": "max",
        },
    )

    result = _run_bridge(bridge_rig)

    assert result["outcome"] == "failed"
    assert "EVIDENCE_MISSING" in result["summary"]
    assert "gate.json does not exist" in result["summary"]


def test_real_bridge_accepts_valid_identity_bound_gate(bridge_rig):
    # The gate identity follows the real bridge key; the commit follows Git.
    _settle(
        bridge_rig["fixture_dir"],
        commit=bridge_rig["candidate"],
        base=bridge_rig["base"],
    )

    result = _run_bridge(bridge_rig)

    assert result["outcome"] == "succeeded", result
    assert result["review_verdict"] == "PASS"


@pytest.mark.parametrize(
    "bad_gate",
    ["malformed", "wrong-job", "wrong-attempt", "wrong-commit"],
)
def test_real_bridge_rejects_malformed_or_mismatched_gate(bridge_rig, bad_gate):
    _settle(
        bridge_rig["fixture_dir"],
        commit=bridge_rig["candidate"],
        base=bridge_rig["base"],
    )
    gate_path = bridge_rig["fixture_dir"] / "gate.json"
    if bad_gate == "malformed":
        gate_path.write_text("{not json", encoding="utf-8")
        expected = "EVIDENCE_MALFORMED"
    else:
        doc = json.loads(gate_path.read_text(encoding="utf-8"))
        if bad_gate == "wrong-job":
            _write_json(gate_path, {**doc, "job": "98-a_other"})
            expected = "EVIDENCE_WRONG_JOB"
        elif bad_gate == "wrong-attempt":
            _write_json(gate_path, {**doc, "attempt": 1})
            expected = "EVIDENCE_STALE_ATTEMPT"
        else:
            _write_json(gate_path, {**doc, "commit": "f" * 40})
            expected = "EVIDENCE_STALE_COMMIT"

    result = _run_bridge(bridge_rig)

    assert result["outcome"] == "failed", result
    assert expected in result["summary"]


def test_real_bridge_keeps_failed_checks_blocked_despite_reviewer_pass(bridge_rig):
    failing = {
        "tests": [
            {
                "cmd": "pytest -q tests/scripts",
                "result": "fail",
                "evidence": "1 failed in 0.10s",
                "classification": "candidate_regression",
                "note": "green at the base commit, red at the candidate",
            }
        ],
        "skipped": [],
        "verdict_hint": "failures",
    }
    _settle(
        bridge_rig["fixture_dir"],
        tests=failing,
        review=PASS_REVIEW,
        commit=bridge_rig["candidate"],
        base=bridge_rig["base"],
    )

    result = _run_bridge(bridge_rig)

    assert result["outcome"] == "failed", result
    assert result["review_verdict"] == "PASS_WITH_FAILING_TESTS"
    assert "fails because of this candidate" in result["summary"]


def test_bridge_patch_is_mapped_to_the_installed_bridge(tmp_path):
    if not LIVE_BRIDGE.is_file():
        pytest.skip("installed PC bridge is available only on the PC runner")
    assert LIVE_BRIDGE.is_file(), f"installed bridge missing: {LIVE_BRIDGE}"
    assert BRIDGE_PATCH.is_file(), "canonical bridge patch is missing"
    bridge = tmp_path / "jobs-pc-bridge.sh"
    shutil.copy2(LIVE_BRIDGE, bridge)
    forward = subprocess.run(
        ["git", "apply", "--check", "-p1", str(BRIDGE_PATCH)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    reverse = subprocess.run(
        ["git", "apply", "--reverse", "--check", "-p1", str(BRIDGE_PATCH)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert forward.returncode == 0 or reverse.returncode == 0, (
        "installed bridge is neither the documented pre-patch source nor the "
        f"patched source\nforward={forward.stderr}\nreverse={reverse.stderr}"
    )
