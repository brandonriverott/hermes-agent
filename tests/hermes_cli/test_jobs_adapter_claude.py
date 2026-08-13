"""Real-seam tests for the Claude execution adapter (hermes_cli/jobs_adapter_claude).

No mocks and no provider. Every test builds a **real** temporary Git repository
and runs a **real** injected worker process, so the isolation, evidence, and
fail-closed behaviour under test are the ones that would run in production —
only the model call is replaced by a deterministic fake executable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_cli import jobs_adapter_claude as adapter
from hermes_cli import jobs_exec as jx
from hermes_cli.jobs_contract import JobEnvelope


TOKEN = "claim-token-must-never-appear-anywhere"


def test_kill_uses_process_kill_on_windows(monkeypatch):
    class Process:
        pid = 7

        def __init__(self):
            self.killed = False

        def kill(self):
            self.killed = True

        def wait(self, timeout):
            assert timeout == 10

    proc = Process()
    monkeypatch.setattr(adapter.os, "name", "nt")
    monkeypatch.setattr(
        adapter.os,
        "killpg",
        lambda *_: pytest.fail("Windows must not call os.killpg"),
    )

    adapter._kill(proc)

    assert proc.killed


# ---------------------------------------------------------------------------
# Real git repository + real injected worker
# ---------------------------------------------------------------------------


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A real git repository with one commit. Returns (path, base_sha)."""
    path = tmp_path / "repo"
    path.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(path)], check=True, capture_output=True
    )
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Jobs V3 Test")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "README.md").write_text("base\n")
    _git(path, "add", "README.md")
    _git(path, "commit", "-qm", "base")
    base = _git(path, "rev-parse", "HEAD").stdout.strip()
    return path, base


WORKER_HEADER = textwrap.dedent(
    """
    import argparse, json, os, subprocess, sys, time
    p = argparse.ArgumentParser()
    p.add_argument("--worktree", required=True)
    p.add_argument("--goal-file", required=True)
    p.add_argument("--result-file", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--effort", required=True)
    p.add_argument("--max-turns", required=True)
    a = p.parse_args()
    wt = a.worktree

    def git(*args):
        subprocess.run(["git", "-C", wt, *args], check=True,
                       capture_output=True, text=True)

    def result(**kw):
        with open(a.result_file, "w") as fh:
            json.dump(kw, fh)

    def commit(name="worker.txt", text="worker was here\\n"):
        with open(os.path.join(wt, name), "w") as fh:
            fh.write(text)
        git("add", name)
        git("-c", "user.email=w@example.invalid", "-c", "user.name=Worker",
            "-c", "commit.gpgsign=false", "commit", "-qm", "worker commit")

    def record(path):
        with open(path, "w") as fh:
            json.dump({"argv": sys.argv, "env": dict(os.environ),
                       "cwd": os.getcwd(),
                       "goal": open(a.goal_file).read()}, fh)
    """
).strip()


def _worker(tmp_path, body: str, name="worker.py"):
    """Write an injected fake worker and return its command list."""
    script = tmp_path / name
    script.write_text(WORKER_HEADER + "\n" + textwrap.dedent(body).strip() + "\n")
    return [sys.executable, str(script)]


def _envelope(**overrides):
    fields = dict(
        job_id="j_test01",
        number=7,
        name="Ship the thing",
        goal="Do the work.\nCarefully.\n",
        attempt_id="a_test01",
        ordinal=1,
        claim_token=TOKEN,
        specialist="claude-builder",
    )
    fields.update(overrides)
    return JobEnvelope(**fields)


def _spec(repo_path, base, **overrides):
    execution = {
        "model": "claude-opus-5",
        "effort": "max",
        "max_turns": 12,
        "repo_path": str(repo_path),
        "base_commit": base,
        "workspace_kind": "worktree",
    }
    execution.update(overrides)
    return jx.validate_execution({"execution": execution})


def _run(tmp_path, repo_path, base, worker, *, envelope=None, **kwargs):
    kwargs.setdefault("wall_clock_seconds", 60)
    return adapter.run_claude_attempt(
        _envelope() if envelope is None else envelope,
        _spec(repo_path, base),
        worker_command=worker,
        workspace_root=tmp_path / "workspaces",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Preflight — missing repository or base identity fails closed
# ---------------------------------------------------------------------------


def test_preflight_accepts_a_real_repo_at_a_real_base(repo):
    path, base = repo
    adapter.preflight(_spec(path, base))  # no raise


def test_preflight_rejects_a_path_that_is_not_a_git_repo(tmp_path, repo):
    _, base = repo
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    with pytest.raises(adapter.AdapterError):
        adapter.preflight(_spec(plain, base))


def test_preflight_rejects_a_missing_repo_path(tmp_path, repo):
    _, base = repo
    with pytest.raises(adapter.AdapterError):
        adapter.preflight(_spec(tmp_path / "nope", base))


def test_preflight_rejects_a_base_commit_the_repo_does_not_have(repo):
    path, _ = repo
    with pytest.raises(adapter.AdapterError):
        adapter.preflight(_spec(path, "b" * 40))


def test_preflight_rejects_a_sha_that_is_not_a_commit(repo):
    path, base = repo
    blob = _git(path, "rev-parse", f"{base}:README.md").stdout.strip()
    with pytest.raises(adapter.AdapterError):
        adapter.preflight(_spec(path, blob))


# ---------------------------------------------------------------------------
# Happy path — isolated worktree, real commit, truthful evidence
# ---------------------------------------------------------------------------


def test_successful_run_captures_branch_worktree_and_commit_evidence(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        commit()
        result(outcome="succeeded")
    """)
    res = _run(tmp_path, path, base, worker)

    assert res.status == "succeeded"
    assert res.failure_class is None
    assert res.repository == str(path)
    assert res.branch and res.branch != "main"
    assert Path(res.worktree).is_dir()
    assert res.commit and res.commit != base
    # The commit named in the receipt is really on the branch, in this repo.
    on_branch = _git(path, "rev-parse", "--verify", res.branch).stdout.strip()
    assert on_branch == res.commit
    assert _git(path, "merge-base", "--is-ancestor", base, res.commit, check=False).returncode == 0
    assert res.receipt["diff_empty"] is False


def test_the_run_is_isolated_from_the_repository_working_tree(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        commit()
        result(outcome="succeeded")
    """)
    before_head = _git(path, "rev-parse", "HEAD").stdout.strip()
    res = _run(tmp_path, path, base, worker)

    assert _git(path, "rev-parse", "HEAD").stdout.strip() == before_head
    assert _git(path, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    assert _git(path, "status", "--porcelain").stdout == ""
    assert not (path / "worker.txt").exists()
    assert (Path(res.worktree) / "worker.txt").exists()


def test_the_worker_runs_inside_the_worktree_and_reads_the_goal_verbatim(tmp_path, repo):
    path, base = repo
    record = tmp_path / "record.json"
    worker = _worker(tmp_path, f"""
        record({str(record)!r})
        result(outcome="succeeded")
    """)
    res = _run(tmp_path, path, base, worker)
    seen = json.loads(record.read_text())

    assert Path(seen["cwd"]).resolve() == Path(res.worktree).resolve()
    assert seen["goal"] == _envelope().goal
    assert "--model" in seen["argv"] and "claude-opus-5" in seen["argv"]
    assert "--effort" in seen["argv"] and "max" in seen["argv"]
    assert "12" in seen["argv"]


def test_a_zero_diff_success_is_recorded_explicitly(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        result(outcome="succeeded")
    """)
    res = _run(tmp_path, path, base, worker)

    # A run that changed nothing did not do the work, whatever it says.
    assert (res.status, res.failure_class) == ("failed", "implementation")
    assert res.commit == base
    assert res.receipt["diff_empty"] is True


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def test_a_declared_worker_failure_is_carried_through(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        result(outcome="failed", failure_class="max_turns")
        sys.exit(1)
    """)
    res = _run(tmp_path, path, base, worker)
    assert (res.status, res.failure_class) == ("failed", "max_turns")


def test_success_claimed_on_a_nonzero_exit_is_refused(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        commit()
        result(outcome="succeeded")
        sys.exit(9)
    """)
    res = _run(tmp_path, path, base, worker)
    assert (res.status, res.failure_class) == ("failed", "infrastructure")


def test_a_worker_that_writes_no_result_is_infrastructure(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        pass
    """)
    res = _run(tmp_path, path, base, worker)
    assert (res.status, res.failure_class) == ("failed", "infrastructure")


def test_a_worker_that_writes_junk_json_is_infrastructure(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        open(a.result_file, "w").write("{not json")
    """)
    res = _run(tmp_path, path, base, worker)
    assert (res.status, res.failure_class) == ("failed", "infrastructure")


def test_a_worker_that_never_exits_is_killed_and_interrupted(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        result(outcome="succeeded")
        time.sleep(300)
    """)
    res = _run(tmp_path, path, base, worker, wall_clock_seconds=2)

    assert (res.status, res.failure_class) == ("interrupted", "infrastructure")
    assert res.receipt["timed_out"] is True


def test_a_worker_that_cannot_be_executed_is_infrastructure(tmp_path, repo):
    path, base = repo
    res = _run(tmp_path, path, base, [str(tmp_path / "no-such-binary")])
    assert (res.status, res.failure_class) == ("failed", "infrastructure")


# ---------------------------------------------------------------------------
# Evidence integrity — the adapter never takes the worker's word for git state
# ---------------------------------------------------------------------------


def test_a_worker_that_leaves_the_declared_branch_fails_closed(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        git("checkout", "-q", "-b", "somewhere-else")
        commit()
        result(outcome="succeeded")
    """)
    res = _run(tmp_path, path, base, worker)

    assert (res.status, res.failure_class) == ("failed", "infrastructure")
    assert "branch" in " ".join(res.receipt["evidence_problems"]).lower()


def test_a_head_that_does_not_descend_from_the_base_fails_closed(tmp_path, repo):
    path, base = repo
    # An orphan commit is a valid commit with no ancestry to the approved base —
    # exactly the evidence a receipt must never accept.
    worker = _worker(tmp_path, """
        git("checkout", "-q", "--orphan", "detached-history")
        git("rm", "-q", "-rf", ".")
        commit()
        git("branch", "-f", os.environ["JOBS_TEST_BRANCH"], "HEAD")
        git("checkout", "-q", os.environ["JOBS_TEST_BRANCH"])
        result(outcome="succeeded")
    """)
    branch = adapter.branch_name(_envelope())
    res = _run(
        tmp_path, path, base, worker, extra_env={"JOBS_TEST_BRANCH": branch}
    )

    assert (res.status, res.failure_class) == ("failed", "infrastructure")
    # Specifically the ancestry check, not merely "something looked odd".
    assert any(
        "does not descend from the approved base" in p
        for p in res.receipt["evidence_problems"]
    ), res.receipt["evidence_problems"]
    # And no commit reaches the attempt row from an unprovable history.
    assert res.commit is None


def test_evidence_problems_are_absent_on_a_clean_run(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        commit()
        result(outcome="succeeded")
    """)
    res = _run(tmp_path, path, base, worker)
    assert res.receipt["evidence_problems"] == []


# ---------------------------------------------------------------------------
# Capability and secret containment
# ---------------------------------------------------------------------------


def test_the_claim_token_never_reaches_the_worker(tmp_path, repo):
    path, base = repo
    record = tmp_path / "record.json"
    worker = _worker(tmp_path, f"""
        record({str(record)!r})
        result(outcome="succeeded")
    """)
    res = _run(tmp_path, path, base, worker)
    seen = json.loads(record.read_text())

    assert TOKEN not in json.dumps(seen["argv"])
    assert TOKEN not in json.dumps(seen["env"])
    assert TOKEN not in json.dumps(res.receipt)
    assert TOKEN not in json.dumps(res.to_dict())


def test_the_worker_environment_is_an_allowlist_not_an_inheritance(tmp_path, repo, monkeypatch):
    path, base = repo
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-be-inherited")
    monkeypatch.setenv("SOME_PRIVATE_THING", "hunter2")
    record = tmp_path / "record.json"
    worker = _worker(tmp_path, f"""
        record({str(record)!r})
        result(outcome="succeeded")
    """)
    _run(tmp_path, path, base, worker)
    env = json.loads(record.read_text())["env"]

    assert "ANTHROPIC_API_KEY" not in env
    assert "SOME_PRIVATE_THING" not in env
    assert env.get("PATH")


def test_a_secret_shaped_log_tail_is_redacted_in_the_receipt(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        print("leaking ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA now")
        sys.stdout.flush()
        result(outcome="failed", failure_class="implementation")
        sys.exit(1)
    """)
    res = _run(tmp_path, path, base, worker)

    blob = json.dumps(res.receipt)
    assert "ghp_AAAA" not in blob
    assert jx.REDACTED in res.receipt["output_tail"]


def test_a_huge_log_tail_cannot_bloat_the_receipt(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        for _ in range(20000):
            print("x" * 200)
        commit()
        result(outcome="succeeded")
    """)
    res = _run(tmp_path, path, base, worker)

    assert res.status == "succeeded"
    encoded = json.dumps(res.receipt, ensure_ascii=False)
    assert len(encoded.encode("utf-8")) <= jx.MAX_RECEIPT_BYTES


# ---------------------------------------------------------------------------
# Custody: heartbeats out, no Jobs mutation in
# ---------------------------------------------------------------------------


def test_the_adapter_heartbeats_through_the_injected_callback(tmp_path, repo):
    path, base = repo
    beats = []
    worker = _worker(tmp_path, """
        time.sleep(1.2)
        commit()
        result(outcome="succeeded")
    """)
    res = _run(
        tmp_path, path, base, worker,
        on_heartbeat=lambda: beats.append(1),
        heartbeat_seconds=1,
    )
    assert res.status == "succeeded"
    assert 1 <= len(beats) <= 3


def test_heartbeats_are_bounded_by_the_interval_not_the_poll_rate(tmp_path, repo):
    path, base = repo
    beats = []
    worker = _worker(tmp_path, """
        time.sleep(1.5)
        commit()
        result(outcome="succeeded")
    """)
    _run(
        tmp_path, path, base, worker,
        on_heartbeat=lambda: beats.append(1),
        heartbeat_seconds=60,
    )
    # A 1.5s run under a 60s cadence must not produce a single event.
    assert beats == []


def test_a_failing_heartbeat_does_not_break_the_run(tmp_path, repo):
    path, base = repo

    def boom():
        raise RuntimeError("custody lost")

    worker = _worker(tmp_path, """
        time.sleep(1.2)
        commit()
        result(outcome="succeeded")
    """)
    res = _run(
        tmp_path, path, base, worker, on_heartbeat=boom, heartbeat_seconds=1
    )
    # The heartbeat is custody upkeep, not the result. A lost lease is caught by
    # the finish call, which fails closed on its own.
    assert res.status == "succeeded"


def test_the_adapter_does_not_touch_the_jobs_database(tmp_path, repo, monkeypatch):
    from hermes_cli import jobs_db as jdb

    path, base = repo
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with jdb.connect_closing() as conn:
        jid = jdb.create_job(conn, requested_lane="claude", name="untouched", goal="stay put")
        before = jdb.get_job(conn, jid).to_dict()
        events_before = len(jdb.get_events(conn, jid))

    worker = _worker(tmp_path, """
        commit()
        result(outcome="succeeded")
    """)
    _run(tmp_path, path, base, worker)

    with jdb.connect_closing() as conn:
        assert jdb.get_job(conn, jid).to_dict() == before
        assert len(jdb.get_events(conn, jid)) == events_before
        assert jdb.get_attempts(conn, jid) == []
        assert jdb.get_receipts(conn, jid) == []


# ---------------------------------------------------------------------------
# Workspace hygiene
# ---------------------------------------------------------------------------


def test_the_branch_name_is_derived_from_the_attempt_and_is_stable(tmp_path, repo):
    env = _envelope()
    assert adapter.branch_name(env) == adapter.branch_name(env)
    assert env.attempt_id in adapter.branch_name(env)
    assert TOKEN not in adapter.branch_name(env)


def test_two_attempts_on_one_job_get_separate_worktrees(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        commit()
        result(outcome="succeeded")
    """)
    first = _run(tmp_path, path, base, worker)
    second = adapter.run_claude_attempt(
        _envelope(attempt_id="a_test02", ordinal=2),
        _spec(path, base),
        worker_command=worker,
        workspace_root=tmp_path / "workspaces",
        wall_clock_seconds=60,
    )
    assert first.worktree != second.worktree
    assert first.branch != second.branch
    assert second.status == "succeeded"


def test_the_handoff_directory_is_owner_private(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, """
        result(outcome="succeeded")
    """)
    res = _run(tmp_path, path, base, worker)
    handoff = Path(res.receipt["handoff_dir"])
    assert handoff.is_dir()
    assert (os.stat(handoff).st_mode & 0o077) == 0


# ---------------------------------------------------------------------------
# A success with nothing to show for it is not a success
# ---------------------------------------------------------------------------


def test_a_zero_exit_success_with_no_change_is_an_implementation_failure(
    tmp_path, repo
):
    path, base = repo
    worker = _worker(tmp_path, 'result(outcome="succeeded")')
    res = _run(tmp_path, path, base, worker)

    assert (res.status, res.failure_class) == ("failed", "implementation")
    assert res.receipt["diff_empty"] is True
    assert any("no change" in p for p in res.receipt["evidence_problems"])


def test_a_real_change_still_succeeds(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, 'commit()\nresult(outcome="succeeded")')
    res = _run(tmp_path, path, base, worker)
    assert (res.status, res.failure_class) == ("succeeded", None)
    assert res.receipt["diff_empty"] is False


# ---------------------------------------------------------------------------
# Preserved worker files carry no secrets and cannot grow without bound
# ---------------------------------------------------------------------------

# Synthetic, never-real fragments. Each matches a different redaction shape so
# one surviving pattern cannot hide behind another.
LEAKED = (
    "sk-hostilesynthetic0123456789",
    "ghp_HOSTILESYNTHETIC0123456789AB",
    "AKIAHOSTILESYNTHET42",
    "api_key=hostile-synthetic-value",
    "Authorization: Bearer hostilesyntheticbearervalue",
)

LEAKY_WORKER = """
for line in {leaked!r}:
    print(line)
sys.stdout.flush()
commit()
result(outcome="succeeded", note={note!r}, nested={{"deep": [{note!r}]}})
""".format(leaked=list(LEAKED), note=LEAKED[3])


def _preserved_files(res):
    handoff = Path(res.receipt["handoff_dir"])
    return sorted(p for p in handoff.rglob("*") if p.is_file())


def test_preserved_worker_files_hold_none_of_the_leaked_secrets(tmp_path, repo):
    path, base = repo
    res = _run(tmp_path, path, base, _worker(tmp_path, LEAKY_WORKER))

    files = _preserved_files(res)
    assert files, "the run must leave its evidence behind"
    for path_obj in files:
        body = path_obj.read_bytes().decode("utf-8")
        for secret in LEAKED:
            assert secret not in body, f"{secret!r} survived in {path_obj}"


def test_no_unredacted_duplicate_survives_anywhere_in_the_workspace(tmp_path, repo):
    path, base = repo
    res = _run(tmp_path, path, base, _worker(tmp_path, LEAKY_WORKER))

    root = Path(res.receipt["handoff_dir"]).parent
    for path_obj in root.rglob("*"):
        if not path_obj.is_file():
            continue
        raw = path_obj.read_bytes()
        for secret in LEAKED:
            assert secret.encode() not in raw, f"{secret!r} survived in {path_obj}"


def test_preserved_worker_files_are_owner_only(tmp_path, repo):
    path, base = repo
    res = _run(tmp_path, path, base, _worker(tmp_path, LEAKY_WORKER))
    for path_obj in _preserved_files(res):
        assert (os.stat(path_obj).st_mode & 0o077) == 0, path_obj


def test_a_chatty_worker_cannot_grow_the_preserved_log_without_bound(tmp_path, repo):
    path, base = repo
    worker = _worker(
        tmp_path,
        'sys.stdout.write("noise line that goes on and on\\n" * 200000)\n'
        'sys.stdout.flush()\n'
        'commit()\n'
        'result(outcome="succeeded")',
    )
    res = _run(tmp_path, path, base, worker)
    log = Path(res.receipt["handoff_dir"]) / "output.log"
    assert log.stat().st_size <= adapter.MAX_PRESERVED_BYTES
    log.read_bytes().decode("utf-8")  # strict UTF-8, no raise


def test_a_preserved_result_file_stays_parseable_standard_json(tmp_path, repo):
    path, base = repo
    res = _run(tmp_path, path, base, _worker(tmp_path, LEAKY_WORKER))
    stored = json.loads(
        (Path(res.receipt["handoff_dir"]) / "result.json").read_text(encoding="utf-8")
    )
    assert stored["outcome"] == "succeeded"


def test_a_scrub_that_cannot_be_written_takes_the_raw_file_with_it(
    tmp_path, monkeypatch
):
    """A failed sanitize must not leave the unredacted original in place.

    The whole point of the rewrite is that the raw bytes stop existing. If the
    replace fails and we only clean up the temporary, the file we were called to
    scrub survives untouched — a leak produced by the leak defence.
    """
    log = tmp_path / "output.log"
    log.write_text("\n".join(LEAKED) + "\n", encoding="utf-8")

    def no_space(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(adapter.os, "replace", no_space)
    adapter._sanitize_log(log)

    assert list(tmp_path.glob("output.log*")) == []  # original and temporary, both gone


def test_a_worker_result_with_non_finite_numbers_never_reads_as_success(
    tmp_path, repo
):
    path, base = repo
    worker = _worker(
        tmp_path,
        'commit()\n'
        'open(a.result_file, "w").write(\'{"outcome": "succeeded", "cost": NaN}\')',
    )
    res = _run(tmp_path, path, base, worker)
    assert (res.status, res.failure_class) == ("failed", "infrastructure")


# ---------------------------------------------------------------------------
# Skill attachment — lent to the builder, never delivered to the reviewer
# ---------------------------------------------------------------------------

SKILL_BODY = "# jobs-demo\nRead this before you build.\n"
SKILL_NAME = "software-development/jobs-demo"
STAGED_AS = "software-development-jobs-demo.md"


@pytest.fixture
def installed_skill(tmp_path, monkeypatch):
    """One real skill installed under a temporary HERMES_HOME."""
    home = tmp_path / "hermes_home"
    skill = home / "skills" / "software-development" / "jobs-demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(SKILL_BODY, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


# Plain strings, not f-strings: the worker body is full of braces and the report
# path arrives through the environment the way the other probes here do.
REPORT_ATTACHMENTS = """
    d = os.path.join(wt, ".cc-skill-attachments")
    seen = {}
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            seen[fn] = {"body": open(p).read(),
                        "mode": oct(os.stat(p).st_mode & 0o777)}
    with open(os.environ["JOBS_TEST_SKILL_REPORT"], "w") as fh:
        json.dump({"present": os.path.isdir(d), "files": seen,
                   "dir_mode": oct(os.stat(d).st_mode & 0o777) if os.path.isdir(d) else None,
                   "pointer": os.environ.get("HERMES_JOB_SKILLS")}, fh)
    commit()
    result(outcome="succeeded")
"""

COMMIT_EVERYTHING = """
    with open(os.path.join(wt, "real-work.txt"), "w") as fh:
        fh.write("actual work\\n")
    git("add", "-A")
    git("-c", "user.email=w@example.invalid", "-c", "user.name=Worker",
        "-c", "commit.gpgsign=false", "commit", "-qm", "everything")
    result(outcome="succeeded")
"""

COMMIT_ONLY_THE_LOAN = """
    git("add", "-A")
    git("-c", "user.email=w@example.invalid", "-c", "user.name=Worker",
        "-c", "commit.gpgsign=false", "commit", "-qm", "only the loan")
    result(outcome="succeeded")
"""


def _with_report(tmp_path, body):
    """(worker, report_path) for a worker that writes a JSON probe."""
    report = tmp_path / "skill-report.json"
    return _worker(tmp_path, body, name="skillworker.py"), report


def test_a_declared_skill_is_readable_and_read_only_inside_the_worktree(
    tmp_path, repo, installed_skill
):
    path, base = repo
    worker, report = _with_report(tmp_path, REPORT_ATTACHMENTS)
    res = _run(
        tmp_path, path, base, worker,
        envelope=_envelope(skills=[SKILL_NAME]),
        extra_env={"JOBS_TEST_SKILL_REPORT": str(report)},
    )
    seen = json.loads(report.read_text())

    assert res.status == "succeeded"
    assert seen["present"] is True
    assert list(seen["files"]) == [STAGED_AS]
    # Byte-for-byte what the skill said, and not writable by the builder.
    assert seen["files"][STAGED_AS]["body"] == SKILL_BODY
    assert seen["files"][STAGED_AS]["mode"] == "0o444"
    assert seen["dir_mode"] == "0o500"
    # And the builder was told where to look.
    assert seen["pointer"] == str(Path(res.worktree) / ".cc-skill-attachments")


def test_the_receipt_records_which_bytes_were_attached(
    tmp_path, repo, installed_skill
):
    path, base = repo
    worker = _worker(tmp_path, 'commit()\nresult(outcome="succeeded")')
    res = _run(
        tmp_path, path, base, worker, envelope=_envelope(skills=[SKILL_NAME])
    )
    skills = res.receipt["skills"]

    assert skills["source"] == "declared"
    assert skills["requested"] == [SKILL_NAME]
    assert skills["attached"] == [{
        "name": SKILL_NAME,
        "sha256": hashlib.sha256(SKILL_BODY.encode("utf-8")).hexdigest(),
        "bytes": len(SKILL_BODY.encode("utf-8")),
    }]
    assert skills["unavailable"] == []
    assert skills["stripped"] is True
    assert skills["committed"] == []


def test_attachments_are_stripped_from_the_worktree_after_the_run(
    tmp_path, repo, installed_skill
):
    path, base = repo
    worker = _worker(tmp_path, 'commit()\nresult(outcome="succeeded")')
    res = _run(
        tmp_path, path, base, worker, envelope=_envelope(skills=[SKILL_NAME])
    )
    attachments = Path(res.worktree) / ".cc-skill-attachments"

    assert res.status == "succeeded"
    assert not attachments.exists()
    # Nothing left of it anywhere in the preserved workspace, either.
    root = Path(res.receipt["handoff_dir"]).parent
    assert [p for p in root.rglob("*jobs-demo*")] == []


def test_attachments_a_builder_committed_are_not_counted_as_work(
    tmp_path, repo, installed_skill
):
    """``git add -A`` must not turn lent reference material into a deliverable."""
    path, base = repo
    worker = _worker(tmp_path, COMMIT_ONLY_THE_LOAN, name="greedy.py")
    res = _run(
        tmp_path, path, base, worker, envelope=_envelope(skills=[SKILL_NAME])
    )

    # The only thing on the branch is material this attempt lent it, so the
    # attempt did no work — and a success with no work is not a success.
    assert res.receipt["diff_empty"] is True
    assert (res.status, res.failure_class) == ("failed", "implementation")
    assert res.receipt["skills"]["committed"] == [
        f".cc-skill-attachments/{STAGED_AS}"
    ]


def test_real_work_still_counts_when_attachments_ride_along(
    tmp_path, repo, installed_skill
):
    path, base = repo
    worker = _worker(tmp_path, COMMIT_EVERYTHING, name="greedy2.py")
    res = _run(
        tmp_path, path, base, worker, envelope=_envelope(skills=[SKILL_NAME])
    )

    assert (res.status, res.failure_class) == ("succeeded", None)
    assert res.receipt["diff_empty"] is False
    assert res.receipt["skills"]["committed"] == [
        f".cc-skill-attachments/{STAGED_AS}"
    ]


@pytest.mark.parametrize(
    "declared",
    [["../../etc/passwd"], ["security/../../../etc"], ["no-such-skill-anywhere"]],
)
def test_a_declared_skill_that_cannot_be_honoured_refuses_before_anything_runs(
    tmp_path, repo, declared
):
    path, base = repo
    launched = tmp_path / "launched.log"
    worker = _worker(
        tmp_path,
        'open(os.environ["JOBS_TEST_LAUNCH_LOG"], "w").write("ran")\n'
        'commit()\nresult(outcome="succeeded")',
        name="never.py",
    )
    with pytest.raises(adapter.AdapterError) as exc:
        _run(
            tmp_path, path, base, worker,
            envelope=_envelope(skills=declared),
            extra_env={"JOBS_TEST_LAUNCH_LOG": str(launched)},
        )

    assert "skill attachment refused" in str(exc.value)
    # Not merely "the worker did not succeed": nothing was built or spent at all.
    assert not launched.exists()
    assert not (tmp_path / "workspaces").exists()


def test_defaults_fire_when_a_job_declares_no_skills(tmp_path, repo):
    """No declaration and a recognisable work type: the rules table decides."""
    path, base = repo
    worker, report = _with_report(tmp_path, REPORT_ATTACHMENTS)
    res = _run(
        tmp_path, path, base, worker,
        envelope=_envelope(goal="Add regression tests for the parser.\n"),
        extra_env={"JOBS_TEST_SKILL_REPORT": str(report)},
    )
    seen = json.loads(report.read_text())

    assert res.receipt["skills"]["source"] == "default"
    assert [a["name"] for a in res.receipt["skills"]["attached"]] == [
        "software-development/test-driven-development"
    ]
    assert list(seen["files"]) == ["software-development-test-driven-development.md"]
    assert seen["files"]["software-development-test-driven-development.md"]["body"]


def test_a_job_with_no_skills_and_no_keywords_attaches_nothing(tmp_path, repo):
    path, base = repo
    worker, report = _with_report(tmp_path, REPORT_ATTACHMENTS)
    res = _run(
        tmp_path, path, base, worker,
        extra_env={"JOBS_TEST_SKILL_REPORT": str(report)},
    )
    seen = json.loads(report.read_text())

    assert res.status == "succeeded"
    assert res.receipt["skills"] == {
        "source": "none",
        "requested": [],
        "attached": [],
        "unavailable": [],
        "stripped": True,
        "committed": [],
    }
    # The worktree is exactly what it would have been before this feature.
    assert seen["present"] is False
    assert seen["pointer"] is None


def test_an_explicitly_empty_declaration_overrides_the_defaults(tmp_path, repo):
    path, base = repo
    worker, report = _with_report(tmp_path, REPORT_ATTACHMENTS)
    res = _run(
        tmp_path, path, base, worker,
        envelope=_envelope(
            goal="Add regression tests for the parser.\n", skills=[]
        ),
        extra_env={"JOBS_TEST_SKILL_REPORT": str(report)},
    )
    assert res.receipt["skills"]["source"] == "none"
    assert res.receipt["skills"]["attached"] == []
    assert json.loads(report.read_text())["present"] is False


def test_a_caller_cannot_supply_the_skills_pointer_itself(tmp_path, repo):
    path, base = repo
    worker = _worker(tmp_path, 'commit()\nresult(outcome="succeeded")')
    with pytest.raises(adapter.AdapterError):
        _run(
            tmp_path, path, base, worker,
            extra_env={"HERMES_JOB_SKILLS": "/somewhere/else"},
        )
