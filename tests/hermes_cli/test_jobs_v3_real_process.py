"""Exactly-once proved with real operating-system processes, not threads.

Threads share a process, an interpreter, and a SQLite connection cache. They can
show that a lock is held; they cannot show what happens when the thing holding
custody is *killed*. Everything here runs ``hermes jobs run-once`` in independent
child processes against a temporary Git repository and a temporary Jobs database,
and proves the three properties that only real processes can:

- four concurrent runners over four different Jobs and one request id produce one
  owner, one claim, one attempt, one worker launch, one receipt, and one response;
- a runner SIGKILLed after it durably bound its request but before it launched a
  worker recovers exactly once, launches nothing, and does not deadlock;
- a runner SIGKILLed after its worker really finished but before the settlement
  committed recovers exactly once, keeps its worktree evidence, and replays exact.

The two crash windows are opened deterministically by putting a ``git`` shim on
the child's ``PATH``. That is the child's own environment, not a hook in the
product: the adapter genuinely resolves ``git`` from ``PATH``, so a shim that
blocks on one exact subcommand parks the run inside the window and holds it there
until the parent has observed durable state and killed the process.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import jobs_db as jdb

from tests.hermes_cli.test_jobs_run_once import (  # noqa: F401 - pytest fixtures
    T0,
    _execution,
    home,
    repo,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LATER = T0 + 100_000  # comfortably past any lease this suite hands out
POLL = 0.02
PATIENCE = 120


# ---------------------------------------------------------------------------
# The child process: the real CLI, in its own interpreter, under its own HOME
# ---------------------------------------------------------------------------


DRIVER = '''
import argparse, os, sys

from hermes_cli import jobs as jobs_cli
from hermes_cli import jobs_db as jdb

# Containment, asserted by the child itself and before it touches anything: the
# environment was set before this interpreter started, so if the resolver ever
# stopped honouring it the child fails here rather than writing a live database.
_home = os.environ["HERMES_HOME"]
_db = str(jdb.jobs_db_path())
assert _db.startswith(_home + os.sep), (_db, _home)
assert os.environ["HOME"].startswith(os.environ["JOBS_TEST_TMP"]), os.environ["HOME"]

parser = argparse.ArgumentParser(prog="hermes")
sub = parser.add_subparsers(dest="command")
jobs_cli.build_parser(sub)
sys.exit(jobs_cli.jobs_command(parser.parse_args(sys.argv[1:])))
'''


GIT_SHIM = '''#!{python}
"""A git that parks on one exact subcommand, and is otherwise the real git."""
import os, sys, time

BLOCK_ON = {block!r}
MARKER = {marker!r}
REAL = {real!r}

argv = sys.argv[1:]
if BLOCK_ON and BLOCK_ON in " ".join(argv):
    with open(MARKER, "w") as fh:
        fh.write(str(os.getpid()))
    # Park until the run-once process that invoked us is killed. Watching for
    # reparenting means this leaves nothing sleeping behind when it is.
    parent = os.getppid()
    deadline = time.time() + 300
    while os.getppid() == parent and time.time() < deadline:
        time.sleep(0.02)
    sys.exit(1)
os.execv(REAL, [REAL] + argv)
'''


WORKER = '''#!{python}
"""Commit one file, then optionally wait to be released."""
import argparse, json, os, subprocess, sys, time

LOG = {log!r}
DONE = {done!r}
RELEASE = {release!r}

p = argparse.ArgumentParser()
for f in ("worktree", "goal-file", "result-file", "model", "effort", "max-turns"):
    p.add_argument("--" + f, required=True)
a = p.parse_args()

with open(LOG, "a") as fh:
    fh.write("%s %s\\n" % (os.getpid(), a.worktree))

with open(os.path.join(a.worktree, "worker.txt"), "w") as fh:
    fh.write("done\\n")
env = dict(os.environ, GIT_AUTHOR_NAME="W", GIT_AUTHOR_EMAIL="w@x.invalid",
           GIT_COMMITTER_NAME="W", GIT_COMMITTER_EMAIL="w@x.invalid")
subprocess.run(["git", "-C", a.worktree, "add", "worker.txt"], check=True, env=env)
subprocess.run(["git", "-C", a.worktree, "-c", "commit.gpgsign=false",
                "commit", "-qm", "work"], check=True, env=env)
json.dump({{"outcome": "succeeded"}}, open(a.result_file, "w"))
if DONE:
    with open(DONE, "w") as fh:
        fh.write("done")
if RELEASE:
    deadline = time.time() + 300
    while not os.path.exists(RELEASE) and time.time() < deadline:
        time.sleep(0.02)
'''


def _executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _worker_script(tmp_path, name, *, log, done=None, release=None) -> Path:
    return _executable(
        tmp_path / name,
        WORKER.format(
            python=sys.executable, log=str(log),
            done=None if done is None else str(done),
            release=None if release is None else str(release),
        ),
    )


def _git_shim(tmp_path, name, *, block_on, marker) -> Path:
    """A PATH directory whose ``git`` parks on ``block_on``. Returns the dir."""
    shim_dir = tmp_path / name
    shim_dir.mkdir()
    real = shutil.which("git")
    assert real, "these tests need a real git"
    _executable(
        shim_dir / "git",
        GIT_SHIM.format(
            python=sys.executable, block=block_on, marker=str(marker), real=real,
        ),
    )
    return shim_dir


def _child_env(home_dir, tmp_path, *, path_prefix=None) -> dict:
    """The child's whole environment. HOME and HERMES_HOME are the temp root."""
    path = os.environ.get("PATH", "/usr/bin:/bin")
    if path_prefix is not None:
        path = f"{path_prefix}{os.pathsep}{path}"
    return {
        "PATH": path,
        "HOME": str(home_dir),
        "HERMES_HOME": str(home_dir),
        "JOBS_TEST_TMP": str(tmp_path),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": os.environ.get("LANG", "C"),
        "TMPDIR": str(tmp_path / "childtmp"),
    }


def _spawn(driver, args, *, env, stdout) -> subprocess.Popen:
    (Path(env["TMPDIR"])).mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [sys.executable, str(driver), *args],
        env=env, stdout=stdout, stderr=subprocess.STDOUT,
    )


def _run_once_args(*, execution_file, workspace_root, worker, worker_id,
                   request_id=None, job=None, at=T0) -> list:
    args = [
        "jobs", "run-once",
        "--worker", str(worker),
        "--workspace-root", str(workspace_root),
        "--execution-file", str(execution_file),
        "--worker-id", worker_id,
        "--wall-clock-seconds", "60",
        "--at", str(at),
        "--json",
    ]
    if request_id is not None:
        args += ["--request-id", request_id]
    if job is not None:
        args += ["--job", str(job)]
    return args


def _wait(predicate, what):
    deadline = time.monotonic() + PATIENCE
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(POLL)
    raise AssertionError(f"timed out waiting for {what}")


# ---------------------------------------------------------------------------
# Read-only observation of the child's store
# ---------------------------------------------------------------------------


def _rows(statement, params=()):
    with sqlite3.connect(jdb.jobs_db_path()) as raw:
        raw.row_factory = sqlite3.Row
        return raw.execute(statement, params).fetchall()


def _counters() -> dict:
    """Every side effect a run can leave, counted."""
    return {
        name: _rows(f"SELECT COUNT(*) AS n FROM {name}")[0]["n"]
        for name in ("jobs", "job_events", "job_attempts", "job_receipts",
                     "job_run_requests")
    } | {
        "revisions": [
            (r["id"], r["revision"])
            for r in _rows("SELECT id, revision FROM jobs ORDER BY number")
        ],
    }


def _launch_lines(log: Path) -> list:
    if not log.exists():
        return []
    return [line for line in log.read_text().splitlines() if line.strip()]


def _payload(out: Path) -> dict:
    return json.loads(out.read_text())


@pytest.fixture
def rig(home, tmp_path, repo):
    """Everything four child processes need, already on disk."""
    path, base = repo
    execution_file = tmp_path / "execution.json"
    execution_file.write_text(json.dumps(_execution(path, base)))
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER)
    return {
        "repo": path, "base": base, "execution_file": execution_file,
        "driver": driver, "workspace_root": tmp_path / "ws",
        "log": tmp_path / "launches.log",
    }


def _make_jobs(count):
    with jdb.connect_closing() as conn:
        return [
            jdb.get_job(conn, jdb.create_job(conn, name=f"J{i}", goal="do it"))
            for i in range(count)
        ]


# ---------------------------------------------------------------------------
# Four processes, four Jobs, one request id
# ---------------------------------------------------------------------------


def test_four_real_processes_on_one_request_id_settle_it_exactly_once(
    home, tmp_path, rig
):
    jobs = _make_jobs(4)
    release = tmp_path / "release"
    worker = _worker_script(tmp_path, "worker.py", log=rig["log"], release=release)
    env = _child_env(home, tmp_path)

    outs = [tmp_path / f"out-{i}.json" for i in range(4)]
    handles = []
    procs = []
    for i, job in enumerate(jobs):
        fh = open(outs[i], "wb")
        handles.append(fh)
        procs.append(_spawn(
            rig["driver"],
            _run_once_args(
                execution_file=rig["execution_file"],
                workspace_root=rig["workspace_root"], worker=worker,
                worker_id=f"runner-{i}", request_id="one-request", job=job.number,
            ),
            env=env, stdout=fh,
        ))

    # The winner is parked inside its worker until released, so both waits are
    # states the race is guaranteed to reach — and reaching them means the losers
    # lost before the winner settled anything.
    _wait(lambda: len(_launch_lines(rig["log"])) == 1, "the one worker launch")
    _wait(lambda: sum(p.poll() is not None for p in procs) == 3, "three losers")
    release.write_text("go")
    codes = [p.wait(timeout=PATIENCE) for p in procs]
    for fh in handles:
        fh.close()

    winners = [i for i, code in enumerate(codes) if code == 0]
    assert len(winners) == 1, [
        (c, outs[i].read_text()) for i, c in enumerate(codes)
    ]
    won = _payload(outs[winners[0]])
    assert won["ran"] is True and won["status"] == "succeeded"
    assert all(code != 0 for i, code in enumerate(codes) if i != winners[0])
    losers = [outs[i].read_text() for i, c in enumerate(codes) if c != 0]
    assert all("request_in_progress" in text for text in losers), losers

    # One of everything, across all four Jobs.
    assert len(_launch_lines(rig["log"])) == 1
    assert _rows("SELECT COUNT(*) AS n FROM job_attempts")[0]["n"] == 1
    assert _rows("SELECT COUNT(*) AS n FROM job_receipts")[0]["n"] == 1
    assert _rows("SELECT COUNT(*) AS n FROM job_run_requests")[0]["n"] == 1
    assert [r["request_id"] for r in _rows(
        "SELECT request_id FROM job_run_requests")] == ["one-request"]
    kinds = [r["kind"] for r in _rows("SELECT kind FROM job_events")]
    assert kinds.count("claim_acquired") == 1
    assert kinds.count("attempt_started") == 1

    # A fifth process, later, replays the identical answer and changes nothing.
    before = _counters()
    replay_out = tmp_path / "replay.json"
    with open(replay_out, "wb") as fh:
        code = _spawn(
            rig["driver"],
            _run_once_args(
                execution_file=rig["execution_file"],
                workspace_root=rig["workspace_root"], worker=worker,
                worker_id="runner-late", request_id="one-request", at=T0 + 5,
            ),
            env=env, stdout=fh,
        ).wait(timeout=PATIENCE)
    assert code == 0
    assert _payload(replay_out) == won
    assert _counters() == before
    assert len(_launch_lines(rig["log"])) == 1


# ---------------------------------------------------------------------------
# Crash window one: bound, then killed, before any worker existed
# ---------------------------------------------------------------------------


def _bound_request(request_id):
    rows = _rows(
        "SELECT * FROM job_run_requests WHERE request_id = ? "
        "AND attempt_id IS NOT NULL",
        (request_id,),
    )
    return rows[0] if rows else None


@pytest.mark.live_system_guard_bypass
def test_a_real_process_killed_after_binding_recovers_exactly_once(
    home, tmp_path, rig
):
    job = _make_jobs(1)[0]
    marker = tmp_path / "git-parked"
    shim = _git_shim(tmp_path, "shim", block_on="worktree add", marker=marker)
    worker = _worker_script(tmp_path, "worker.py", log=rig["log"])
    env = _child_env(home, tmp_path, path_prefix=shim)

    out = tmp_path / "victim.json"
    with open(out, "wb") as fh:
        victim = _spawn(
            rig["driver"],
            _run_once_args(
                execution_file=rig["execution_file"],
                workspace_root=rig["workspace_root"], worker=worker,
                worker_id="doomed", request_id="died-early", job=job.number,
            ),
            env=env, stdout=fh,
        )

        # Durable bound state, read from the store by the parent — not inferred
        # from a log line and not announced by the child.
        bound = _wait(lambda: _bound_request("died-early"), "the request binding")
        _wait(marker.exists, "the run to reach the worktree step")
        attempt = _rows(
            "SELECT * FROM job_attempts WHERE id = ?", (bound["attempt_id"],)
        )[0]
        assert attempt["status"] == "running"
        assert attempt["base_commit"] == rig["base"]

        victim.kill()
        assert victim.wait(timeout=PATIENCE) != 0
    assert _launch_lines(rig["log"]) == []  # it died before any worker existed

    # A fresh process, after the lease has lapsed. Recovery settles the corpse
    # once, launches nothing, and answers the request instead of deadlocking.
    first_out = tmp_path / "first.json"
    with open(first_out, "wb") as fh:
        code = _spawn(
            rig["driver"],
            _run_once_args(
                execution_file=rig["execution_file"],
                workspace_root=rig["workspace_root"], worker=worker,
                worker_id="fresh", request_id="died-early", at=LATER,
            ),
            env=_child_env(home, tmp_path), stdout=fh,
        ).wait(timeout=PATIENCE)
    assert code == 0
    first = _payload(first_out)
    assert first["ran"] is True
    assert (first["status"], first["failure_class"]) == (
        "interrupted", "infrastructure"
    )
    assert first["attempt_id"] == bound["attempt_id"]
    assert first["base_commit"] == rig["base"]
    assert _launch_lines(rig["log"]) == []
    assert _rows("SELECT COUNT(*) AS n FROM job_attempts")[0]["n"] == 1
    assert _rows("SELECT COUNT(*) AS n FROM job_receipts")[0]["n"] == 1

    before = _counters()
    second_out = tmp_path / "second.json"
    with open(second_out, "wb") as fh:
        code = _spawn(
            rig["driver"],
            _run_once_args(
                execution_file=rig["execution_file"],
                workspace_root=rig["workspace_root"], worker=worker,
                worker_id="fresh", request_id="died-early", at=LATER + 100,
            ),
            env=_child_env(home, tmp_path), stdout=fh,
        ).wait(timeout=PATIENCE)
    assert code == 0
    assert _payload(second_out) == first
    assert _counters() == before
    assert _launch_lines(rig["log"]) == []


# ---------------------------------------------------------------------------
# Crash window two: the worker really finished, the settlement never committed
# ---------------------------------------------------------------------------


@pytest.mark.live_system_guard_bypass
def test_a_real_process_killed_after_its_worker_finished_recovers_exactly_once(
    home, tmp_path, rig
):
    job = _make_jobs(1)[0]
    marker = tmp_path / "git-parked"
    done = tmp_path / "worker-done"
    # Parked on the ancestry check, which the adapter only reaches once the
    # worker has exited and its result file has been read.
    shim = _git_shim(tmp_path, "shim", block_on="merge-base", marker=marker)
    worker = _worker_script(tmp_path, "worker.py", log=rig["log"], done=done)
    env = _child_env(home, tmp_path, path_prefix=shim)

    out = tmp_path / "victim.json"
    with open(out, "wb") as fh:
        victim = _spawn(
            rig["driver"],
            _run_once_args(
                execution_file=rig["execution_file"],
                workspace_root=rig["workspace_root"], worker=worker,
                worker_id="doomed", request_id="died-late", job=job.number,
            ),
            env=env, stdout=fh,
        )
        _wait(done.exists, "the worker to finish")
        _wait(marker.exists, "the run to reach the evidence step")
        bound = _wait(lambda: _bound_request("died-late"), "the request binding")
        victim.kill()
        assert victim.wait(timeout=PATIENCE) != 0

    launches = _launch_lines(rig["log"])
    assert len(launches) == 1
    worktree = Path(launches[0].split(" ", 1)[1])
    assert (worktree / "worker.txt").exists()  # evidence survives the crash
    assert _rows(
        "SELECT status FROM job_attempts WHERE id = ?", (bound["attempt_id"],)
    )[0]["status"] == "running"
    assert _rows("SELECT COUNT(*) AS n FROM job_receipts")[0]["n"] == 0

    first_out = tmp_path / "first.json"
    with open(first_out, "wb") as fh:
        code = _spawn(
            rig["driver"],
            _run_once_args(
                execution_file=rig["execution_file"],
                workspace_root=rig["workspace_root"], worker=worker,
                worker_id="fresh", request_id="died-late", at=LATER,
            ),
            env=_child_env(home, tmp_path), stdout=fh,
        ).wait(timeout=PATIENCE)
    assert code == 0
    first = _payload(first_out)
    assert first["ran"] is True
    assert (first["status"], first["failure_class"]) == (
        "interrupted", "infrastructure"
    )
    assert first["attempt_id"] == bound["attempt_id"]
    assert _launch_lines(rig["log"]) == launches  # no second execution
    assert (worktree / "worker.txt").exists()
    assert _rows("SELECT COUNT(*) AS n FROM job_attempts")[0]["n"] == 1
    assert _rows("SELECT COUNT(*) AS n FROM job_receipts")[0]["n"] == 1

    before = _counters()
    second_out = tmp_path / "second.json"
    with open(second_out, "wb") as fh:
        code = _spawn(
            rig["driver"],
            _run_once_args(
                execution_file=rig["execution_file"],
                workspace_root=rig["workspace_root"], worker=worker,
                worker_id="fresh", request_id="died-late", at=LATER + 100,
            ),
            env=_child_env(home, tmp_path), stdout=fh,
        ).wait(timeout=PATIENCE)
    assert code == 0
    assert _payload(second_out) == first
    assert _counters() == before
    assert _launch_lines(rig["log"]) == launches


# ---------------------------------------------------------------------------
# The spawned worker's own environment, observed from a real child process
# ---------------------------------------------------------------------------


ENV_WORKER = '''#!{python}
import argparse, json, os, sys
p = argparse.ArgumentParser()
for f in ("worktree", "goal-file", "result-file", "model", "effort", "max-turns"):
    p.add_argument("--" + f, required=True)
a = p.parse_args()
json.dump(dict(os.environ), open({report!r}, "w"))
json.dump({{"outcome": "failed", "failure_class": "implementation"}},
          open(a.result_file, "w"))
sys.exit(1)
'''


def test_a_real_worker_process_never_sees_an_operator_path(home, tmp_path, rig):
    _make_jobs(1)
    report = tmp_path / "env.json"
    worker = _executable(
        tmp_path / "envworker.py",
        ENV_WORKER.format(python=sys.executable, report=str(report)),
    )
    out = tmp_path / "out.json"
    with open(out, "wb") as fh:
        code = _spawn(
            rig["driver"],
            _run_once_args(
                execution_file=rig["execution_file"],
                workspace_root=rig["workspace_root"], worker=worker,
                worker_id="env-runner",
            ),
            env=_child_env(home, tmp_path), stdout=fh,
        ).wait(timeout=PATIENCE)
    assert code == 0

    seen = json.loads(report.read_text())
    private = rig["workspace_root"].resolve()
    for key in ("HOME", "HERMES_HOME", "TMPDIR"):
        resolved = Path(seen[key]).resolve()
        assert private in resolved.parents, (key, seen[key])
    assert seen["HOME"] != str(home) and seen["HERMES_HOME"] != str(home)
    # The operator's live Hermes home appears nowhere in the child's environment.
    live = str(Path("~/.hermes").expanduser())
    assert not [v for v in seen.values() if live in v], seen
    assert seen["PATH"]
