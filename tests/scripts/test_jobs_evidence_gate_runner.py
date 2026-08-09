"""End-to-end proof that the gate governs the real Jobs runner.

The unit tests in ``test_jobs_evidence_gate.py`` drive the gate directly. These
drive ``pc-jobs-worker.sh`` itself — the installed runner, copied to a scratch
directory and patched with ``scripts/jobs_evidence_gate.runner.patch`` — over a
throwaway git repo, with a stub standing in for the ``claude`` CLI.

Nothing here touches the installed runner, the live ``~/jobs`` tree, or any
network. The runner lives outside this repository, so these tests skip when it
is not installed; that skip *is* the activation boundary, and the reason the
gate cannot be called live until it is deployed beside the runner. Point
``PC_JOBS_WORKER`` at a copy to run them anywhere.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "jobs_evidence_gate.py"
PATCH = REPO_ROOT / "scripts" / "jobs_evidence_gate.runner.patch"

RUNNER = Path(
    os.environ.get("PC_JOBS_WORKER") or Path.home() / ".local" / "bin" / "pc-jobs-worker.sh"
)

pytestmark = pytest.mark.skipif(
    not RUNNER.is_file() or shutil.which("git") is None,
    reason=f"the Jobs runner is not installed at {RUNNER} (set PC_JOBS_WORKER)",
)

KEY = "99-a_gatetest"

GREEN_TESTS = {
    "tests": [{"cmd": "pytest -q", "result": "pass", "evidence": "3 passed in 0.10s"}],
    "skipped": [],
    "verdict_hint": "clean",
}
PASS_REVIEW = {"verdict": "PASS", "findings": [], "checks_run": ["read the diff"], "checks_skipped": []}


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"},
    )


@pytest.fixture
def rig(tmp_path):
    """A patched runner, an empty repo, a job dir and a scriptable claude stub."""
    runner = tmp_path / "pc-jobs-worker.sh"
    shutil.copy2(RUNNER, runner)
    subprocess.run(
        ["git", "apply", "-p1", str(PATCH)], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    runner.chmod(0o755)

    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git(
        "-c", "user.name=t", "-c", "user.email=t@e", "-c", "commit.gpgsign=false",
        "commit", "-q", "-m", "seed", cwd=repo,
    )
    base = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()

    jobs_root = tmp_path / "jobs"
    job_dir = jobs_root / KEY
    job_dir.mkdir(parents=True)
    (job_dir / "goal.md").write_text("Add a line to seed.txt.\n", encoding="utf-8")
    (job_dir / "soul-vulcan.md").write_text("You gather evidence.\n", encoding="utf-8")
    (job_dir / "soul-themis.md").write_text("You review.\n", encoding="utf-8")

    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    return {
        "runner": runner,
        "repo": repo,
        "base": base,
        "jobs_root": jobs_root,
        "job_dir": job_dir,
        "stub_dir": stub_dir,
        "bin": tmp_path / "bin",
    }


def install_claude_stub(rig, *, builder_sh, replies):
    """A fake `claude` that edits on call 1 and replays canned JSON after."""
    bin_dir = rig["bin"]
    bin_dir.mkdir(exist_ok=True)
    for index, reply in enumerate(replies, start=2):
        (rig["stub_dir"] / f"{index}.json").write_text(json.dumps(reply), encoding="utf-8")
    stub = bin_dir / "claude"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'state="$STUB_DIR/count"\n'
        'n=$(cat "$state" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$state"\n'
        'if [ "$n" = "1" ]; then\n'
        f"{builder_sh}\n"
        "  exit 0\n"
        "fi\n"
        'cat "$STUB_DIR/$n.json" 2>/dev/null || echo "{}"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def run_job(rig):
    env = {
        **os.environ,
        "JOBS_ROOT": str(rig["jobs_root"]),
        "JOBS_GATE": str(GATE),
        "NO_TMUX": "1",
        "STUB_DIR": str(rig["stub_dir"]),
        "PATH": f"{rig['bin']}{os.pathsep}{os.environ['PATH']}",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "HOME": str(rig["runner"].parent),
    }
    proc = subprocess.run(
        [
            "bash", str(rig["runner"]), "spawn", KEY,
            str(rig["repo"]), rig["base"], "claude-opus-5", "5", "max", "600",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    done_path = rig["job_dir"] / "done.json"
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else None
    return proc, done


def test_green_evidence_settles_as_succeeded_through_the_real_runner(rig):
    install_claude_stub(
        rig,
        builder_sh='  echo "built" >> seed.txt',
        replies=[GREEN_TESTS, PASS_REVIEW],
    )
    proc, done = run_job(rig)
    assert done is not None, f"no done.json\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert done["outcome"] == "succeeded", done
    assert done["review_verdict"] == "PASS"
    assert done["commit"] and done["commit"] != rig["base"]

    # The runner stamped both receipts, and the stamps name this job, this
    # attempt, and the commit that was settled.
    for name, kind in (("tests.json", "tests"), ("review-0.json", "review")):
        env = json.loads((rig["job_dir"] / name).read_text(encoding="utf-8"))["_gate"]
        assert (env["job"], env["attempt"], env["kind"]) == (KEY, 0, kind)
        assert env["commit"] == done["commit"]

    packet = (rig["job_dir"] / "action-packet.md").read_text(encoding="utf-8")
    assert "Independent review" in packet and "After settlement" in packet


def test_a_tester_that_moves_head_cannot_certify_the_candidate(rig):
    """Evidence gathered at a commit that is not the candidate parks the job.

    VULCAN runs with Bash, so it can commit. Before the gate, its receipt had
    no commit on it and settlement could not tell.
    """
    install_claude_stub(
        rig,
        builder_sh='  echo "built" >> seed.txt',
        replies=[GREEN_TESTS, PASS_REVIEW],
    )
    # Make the tester (call 2) move HEAD before it answers.
    stub = rig["bin"] / "claude"
    stub.write_text(
        stub.read_text(encoding="utf-8").replace(
            'cat "$STUB_DIR/$n.json"',
            'if [ "$n" = "2" ]; then\n'
            '  echo "tester scratch" > zz-vulcan-scratch.txt\n'
            "  git add -A >/dev/null 2>&1\n"
            '  git -c user.name=v -c user.email=v@e -c commit.gpgsign=false commit -q -m "tester" >/dev/null 2>&1\n'
            "fi\n"
            'cat "$STUB_DIR/$n.json"',
        ),
        encoding="utf-8",
    )
    proc, done = run_job(rig)
    assert done is not None, f"no done.json\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert done["outcome"] == "needs_attention", done
    assert done["review_verdict"] == "EVIDENCE_STALE_COMMIT", done
    packet = (rig["job_dir"] / "action-packet.md").read_text(encoding="utf-8")
    assert "Why this is not a success:" in packet


def test_a_builder_planted_done_json_does_not_survive_settlement(rig):
    """The builder can write $job_dir. Only the gate decides what settles."""
    planted = '{"outcome":"succeeded","commit":"planted","review_verdict":"PASS"}'
    install_claude_stub(
        rig,
        builder_sh=(
            f"  printf '%s' '{planted}' > \"$STUB_DIR/../jobs/{KEY}/done.json\"\n"
            f'  touch "$STUB_DIR/planted.flag"\n'
            '  echo "built" >> seed.txt'
        ),
        replies=[GREEN_TESTS, PASS_REVIEW],
    )
    proc, done = run_job(rig)
    # The flag proves the builder really reached the job directory and wrote it.
    assert (rig["stub_dir"] / "planted.flag").exists(), proc.stderr
    assert done is not None, f"no done.json\nstdout={proc.stdout}\nstderr={proc.stderr}"
    assert done["commit"] != "planted", "the gate accepted a builder-authored settlement"
    assert done["commit"] and done["commit"] != rig["base"]
    assert (rig["job_dir"] / "gate.json").exists()


def test_the_patch_still_applies_to_the_installed_runner(tmp_path):
    """Drift detector: the runner changed under the activation patch."""
    shutil.copy2(RUNNER, tmp_path / "pc-jobs-worker.sh")
    proc = subprocess.run(
        ["git", "apply", "--check", "-p1", str(PATCH)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"{PATCH.name} no longer applies to {RUNNER}; re-cut it before activating.\n{proc.stderr}"
    )
