"""End-to-end tests for ``hermes jobs run-once`` — the V3 orchestration seam.

Real temporary Git repositories, a real ``jobs.db``, and a real injected worker
process. No provider is called and no mock stands in for custody, so what these
prove is what would actually happen on a live run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sqlite3
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from hermes_cli import jobs as jobs_cli
from hermes_cli import jobs_adapter_claude as claude_adapter
from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_exec as jx
from hermes_cli import jobs_executors as executors
from hermes_cli import jobs_identity as identity
from hermes_cli import jobs_run


T0 = 1_000_000


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes_home"
    h.mkdir()
    # Both, and the assertion: HERMES_HOME is what the resolver reads, HOME is
    # what it (and any child process) falls back to. A test that only set the
    # first would still write the real database the moment the resolver changed.
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setenv("HERMES_HOME", str(h))
    assert tmp_path in jdb.jobs_db_path().parents
    return h


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(path)], check=True, capture_output=True
    )
    for key, value in (("user.email", "t@example.invalid"), ("user.name", "T"),
                       ("commit.gpgsign", "false")):
        subprocess.run(["git", "-C", str(path), "config", key, value], check=True)
    (path / "README.md").write_text("base\n")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "base"], check=True)
    base = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return path, base


WORKER = textwrap.dedent(
    """
    import argparse, json, os, subprocess, sys, time
    p = argparse.ArgumentParser()
    for f in ("worktree", "goal-file", "result-file", "model", "effort", "max-turns"):
        p.add_argument("--" + f, required=True)
    a = p.parse_args()
    launch_log = os.environ.get("JOBS_TEST_LAUNCH_LOG")
    if launch_log:
        with open(launch_log, "a") as fh:
            fh.write(a.worktree + "\\n")
    {behaviour}
    """
)

COMMIT_AND_SUCCEED = """
    with open(os.path.join(a.worktree, "worker.txt"), "w") as fh:
        fh.write("done\\n")
    subprocess.run(["git", "-C", a.worktree, "add", "worker.txt"], check=True)
    subprocess.run(["git", "-C", a.worktree, "-c", "user.email=w@x.invalid",
                    "-c", "user.name=W", "-c", "commit.gpgsign=false",
                    "commit", "-qm", "work"], check=True)
    json.dump({"outcome": "succeeded"}, open(a.result_file, "w"))
"""

FAIL = """
    json.dump({"outcome": "failed", "failure_class": "implementation"},
              open(a.result_file, "w"))
    sys.exit(1)
"""


def _worker(tmp_path, behaviour=COMMIT_AND_SUCCEED, name="worker.py"):
    script = tmp_path / name
    script.write_text(WORKER.format(behaviour=textwrap.dedent(behaviour).strip()))
    return [sys.executable, str(script)]


def _execution(repo_path, base, **overrides):
    execution = {
        "model": "claude-opus-5",
        "effort": "max",
        "max_turns": 12,
        "repo_path": str(repo_path),
        "base_commit": base,
        "workspace_kind": "worktree",
    }
    execution.update(overrides)
    return {"execution": execution}


def _make_job(
    name="Ship it", specialist="claude-builder", requested_lane="claude"
):
    with jdb.connect_closing() as conn:
        jid = jdb.create_job(
            conn,
            name=name,
            goal="do the work",
            specialist=specialist,
            requested_lane=requested_lane,
        )
        return jdb.get_job(conn, jid)


def _run_once(tmp_path, repo_path, base, worker=None, **kwargs):
    kwargs.setdefault("worker_id", "runner-1")
    kwargs.setdefault("specialist", "claude-builder")
    kwargs.setdefault("wall_clock_seconds", 60)
    kwargs.setdefault("heartbeat_seconds", 60)
    kwargs.setdefault("now", T0)
    kwargs.setdefault("execution", _execution(repo_path, base))
    return jobs_run.run_once(
        worker_command=worker if worker is not None else _worker(tmp_path),
        workspace_root=tmp_path / "workspaces",
        **kwargs,
    )


def _events(job_id):
    with jdb.connect_closing() as conn:
        return [e["kind"] for e in jdb.get_events(conn, job_id)]


def _store_snapshot(job_id):
    """Every durable row a pre-attempt routing refusal must leave untouched."""

    with jdb.connect_closing() as conn:
        return {
            "job": dict(
                conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            ),
            "events": [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM job_events WHERE job_id = ? ORDER BY id", (job_id,)
                )
            ],
            "attempts": [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY id", (job_id,)
                )
            ],
            "receipts": [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM job_receipts WHERE job_id = ? ORDER BY id", (job_id,)
                )
            ],
        }


# ---------------------------------------------------------------------------
# The whole seam, once, on real infrastructure
# ---------------------------------------------------------------------------


def test_run_once_claims_runs_and_settles_one_job(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    res = _run_once(tmp_path, path, base)

    assert res["ran"] is True
    assert res["job"]["number"] == job.number
    assert res["status"] == "succeeded"
    assert res["ordinal"] == 1
    assert res["commit"] and res["commit"] != base
    assert res["branch"] and res["worktree"]

    with jdb.connect_closing() as conn:
        attempts = jdb.get_attempts(conn, job.id)
        after = jdb.get_job(conn, job.id)
        receipts = jdb.get_receipts(conn, job.id)

    assert len(attempts) == 1
    a = attempts[0]
    assert (a["ordinal"], a["status"], a["failure_class"]) == (1, "succeeded", None)
    assert a["commit"] == res["commit"]
    assert a["branch"] == res["branch"]
    assert a["worktree"] == res["worktree"]
    assert a["repository"] == str(path)

    # Custody was returned; the Job settled.
    assert (after.status, after.step) == ("finished", "complete")
    assert after.claimed_by is None and after.lease_expires_at is None
    assert after.current_attempt_id is None

    assert len(receipts) == 1
    assert receipts[0]["idempotency_key"] == f"{a['id']}:final"
    assert receipts[0]["attempt_id"] == a["id"]


def test_the_commit_in_the_receipt_is_really_on_the_branch(home, tmp_path, repo):
    path, base = repo
    _make_job()
    res = _run_once(tmp_path, path, base)

    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--verify", res["branch"]],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == res["commit"]
    changed = subprocess.run(
        ["git", "-C", str(path), "diff", "--name-only", base, res["commit"]],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert changed == "worker.txt"


def test_the_ledger_records_the_lifecycle_in_order(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    _run_once(tmp_path, path, base)
    kinds = _events(job.id)

    for kind in ("claim_acquired", "attempt_started", "attempt_finished",
                 "job_transition", "receipt_added"):
        assert kind in kinds, kinds
    assert (
        kinds.index("claim_acquired")
        < kinds.index("attempt_started")
        < kinds.index("attempt_finished")
        <= kinds.index("job_transition")
        < kinds.index("receipt_added")
    )


def test_run_once_reports_when_nothing_is_eligible(home, tmp_path, repo):
    path, base = repo
    res = _run_once(tmp_path, path, base)
    assert res["ran"] is False
    assert res["reason"] == "nothing_eligible"
    assert res["attempt_id"] is None


def test_a_failing_worker_returns_the_job_for_correction(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    res = _run_once(tmp_path, path, base, worker=_worker(tmp_path, FAIL))

    assert res["ran"] is True
    assert (res["status"], res["failure_class"]) == ("failed", "implementation")
    with jdb.connect_closing() as conn:
        after = jdb.get_job(conn, job.id)
    assert (after.status, after.step) == ("working", "correcting")
    assert after.claim_token is None if hasattr(after, "claim_token") else True
    assert after.claimed_by is None


def test_exactly_one_worker_is_launched_per_run(home, tmp_path, repo, monkeypatch):
    path, base = repo
    log = tmp_path / "launches.log"
    monkeypatch.setenv("JOBS_TEST_LAUNCH_LOG", str(log))
    _make_job()
    _run_once(
        tmp_path, path, base,
        extra_env={"JOBS_TEST_LAUNCH_LOG": str(log)},
    )
    assert log.read_text().strip().count("\n") == 0
    assert log.read_text().strip()


# ---------------------------------------------------------------------------
# Exactly one final receipt, and it never outranks the attempt
# ---------------------------------------------------------------------------


def test_exactly_one_final_receipt_is_written(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    res = _run_once(tmp_path, path, base)
    with jdb.connect_closing() as conn:
        receipts = jdb.get_receipts(conn, job.id)
    assert len(receipts) == 1
    assert receipts[0]["idempotency_key"] == f"{res['attempt_id']}:final"
    assert _events(job.id).count("receipt_added") == 1


def test_the_receipt_agrees_with_the_attempt_it_documents(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    _run_once(tmp_path, path, base, worker=_worker(tmp_path, FAIL))
    with jdb.connect_closing() as conn:
        attempt = jdb.get_attempts(conn, job.id)[0]
        receipt = jdb.get_receipts(conn, job.id)[0]["data"]
    assert receipt["status"] == attempt["status"]
    assert receipt["failure_class"] == attempt["failure_class"]
    assert receipt["attempt_id"] == attempt["id"]


def test_a_receipt_that_disagrees_with_its_attempt_is_never_written(
    home, tmp_path, repo, monkeypatch
):
    path, base = repo
    job = _make_job()
    real = jobs_run.jx.sanitize_receipt

    def tamper(data):
        clean = real(data)
        clean["status"] = "succeeded"  # a receipt claiming more than the attempt
        return clean

    monkeypatch.setattr(jobs_run.jx, "sanitize_receipt", tamper)
    with pytest.raises(ValueError, match="disagrees with attempt"):
        _run_once(tmp_path, path, base, worker=_worker(tmp_path, FAIL))

    with jdb.connect_closing() as conn:
        # Settlement and its receipt are one write, so the refusal lands before
        # either exists: the attempt is still running and repairable, and no
        # contradictory evidence was stored.
        assert jdb.get_attempts(conn, job.id)[0]["status"] == "running"
        assert jdb.get_receipts(conn, job.id) == []


def test_a_later_receipt_cannot_rewrite_a_settled_attempt(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    _run_once(tmp_path, path, base, worker=_worker(tmp_path, FAIL))
    with jdb.connect_closing() as conn:
        attempt = jdb.get_attempts(conn, job.id)[0]
        # A receipt is evidence, never a verdict: storing a contradictory one
        # leaves the authoritative attempt status exactly where it was.
        jdb.add_receipt(
            conn, job.id, data={"status": "succeeded"}, attempt_id=attempt["id"],
            idempotency_key="hostile",
        )
        assert jdb.get_attempt(conn, attempt["id"])["status"] == "failed"
        assert jdb.get_job(conn, job.id).step == "correcting"


# ---------------------------------------------------------------------------
# Two runners, one Job
# ---------------------------------------------------------------------------


def test_two_concurrent_runs_produce_exactly_one_of_everything(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    log = tmp_path / "launches.log"
    worker = _worker(tmp_path, COMMIT_AND_SUCCEED)
    results = []
    barrier = threading.Barrier(2)

    def go(worker_id):
        barrier.wait()
        results.append(
            _run_once(
                tmp_path, path, base, worker=worker, worker_id=worker_id,
                extra_env={"JOBS_TEST_LAUNCH_LOG": str(log)},
            )
        )

    threads = [threading.Thread(target=go, args=(f"runner-{i}",)) for i in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert len(results) == 2
    ran = [r for r in results if r["ran"]]
    assert len(ran) == 1, results
    assert [r["reason"] for r in results if not r["ran"]] == ["nothing_eligible"]

    with jdb.connect_closing() as conn:
        assert len(jdb.get_attempts(conn, job.id)) == 1
        assert len(jdb.get_receipts(conn, job.id)) == 1
    kinds = _events(job.id)
    assert kinds.count("claim_acquired") == 1
    assert kinds.count("attempt_started") == 1
    assert kinds.count("attempt_finished") == 1
    # One worker process, one worktree.
    assert len(log.read_text().strip().splitlines()) == 1


# ---------------------------------------------------------------------------
# Recovery of an abandoned worker
# ---------------------------------------------------------------------------


def _abandon(job, lease=60, now=T0):
    """Claim and start an attempt, then walk away — a killed worker."""
    with jdb.connect_closing() as conn:
        claim = jdb.claim_job(
            conn, worker="dead-runner", job=job.number,
            specialist=job.specialist, lease_seconds=lease, now=now,
        )
        jdb.start_attempt(conn, job.id, claim_token=claim.claim_token, now=now)


def test_an_abandoned_attempt_is_recovered_then_the_job_runs_again(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    _abandon(job)

    res = _run_once(tmp_path, path, base, now=T0 + 61)

    assert res["recovered"] == [job.number]
    assert res["ran"] is True
    with jdb.connect_closing() as conn:
        attempts = jdb.get_attempts(conn, job.id)
    assert len(attempts) == 2
    assert (attempts[0]["status"], attempts[0]["failure_class"]) == (
        "interrupted", "infrastructure",
    )
    assert attempts[0]["ordinal"] == 1
    assert (attempts[1]["status"], attempts[1]["ordinal"]) == ("succeeded", 2)
    assert _events(job.id).count("claim_recovered") == 1


def test_recovery_happens_exactly_once_however_often_run_once_is_called(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    _abandon(job)

    first = _run_once(tmp_path, path, base, now=T0 + 61)
    second = _run_once(tmp_path, path, base, now=T0 + 62)
    third = _run_once(tmp_path, path, base, now=T0 + 63)

    assert first["recovered"] == [job.number]
    assert second["recovered"] == []
    assert third["recovered"] == []
    assert _events(job.id).count("claim_recovered") == 1


def test_an_unexpired_claim_is_never_stolen(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    _abandon(job, lease=3600)

    res = _run_once(tmp_path, path, base, now=T0 + 10)

    assert res["recovered"] == []
    assert res["ran"] is False
    assert res["reason"] == "nothing_eligible"
    with jdb.connect_closing() as conn:
        assert len(jdb.get_attempts(conn, job.id)) == 1
        assert jdb.get_attempts(conn, job.id)[0]["status"] == "running"


# ---------------------------------------------------------------------------
# Fail closed, without partial mutation
# ---------------------------------------------------------------------------


def _assert_clean_refusal(job, res, reason):
    assert res["ran"] is False
    assert res["reason"] == reason
    assert res["attempt_id"] is None
    with jdb.connect_closing() as conn:
        after = jdb.get_job(conn, job.id)
        assert jdb.get_attempts(conn, job.id) == []
        assert jdb.get_receipts(conn, job.id) == []
        # Custody handed back: still working, unclaimed, claimable again.
        assert after.status == "working"
        assert after.claimed_by is None
        assert after.lease_expires_at is None
        again = jdb.claim_job(
            conn, worker="next", job=job.number, specialist=job.specialist,
            lease_seconds=60, now=T0 + 5,
        )
        assert again is not None


def test_malformed_execution_metadata_fails_closed(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    res = _run_once(
        tmp_path, path, base, execution={"execution": {"model": "claude-opus-5"}}
    )
    _assert_clean_refusal(job, res, "invalid_execution_metadata")


def test_a_base_commit_the_repo_does_not_have_fails_closed(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    res = _run_once(
        tmp_path, path, base, execution=_execution(path, "c" * 40)
    )
    _assert_clean_refusal(job, res, "preflight_failed")


def test_a_missing_repository_fails_closed(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    res = _run_once(
        tmp_path, path, base, execution=_execution(tmp_path / "gone", base)
    )
    _assert_clean_refusal(job, res, "preflight_failed")


def _neutral_fake_adapter(name, calls, *, forbidden=False):
    """Independent provider fake: no callable delegates to another provider."""

    def record(stage):
        calls.append((name, stage))
        if forbidden:
            raise AssertionError(f"wrong provider invoked at {stage}")

    def preflight(_spec):
        record("preflight")

    def preflight_skills(_skills, **_kwargs):
        record("preflight_skills")
        return ()

    def run_attempt(_envelope, spec, **_kwargs):
        record("run_attempt")
        return types.SimpleNamespace(
            status="failed",
            failure_class="implementation",
            repository=spec.repo_path,
            branch=None,
            worktree=None,
            commit=None,
            receipt={"status": "failed", "failure_class": "implementation"},
        )

    return executors.LegacyExecutorAdapter(
        name=name,
        preflight=preflight,
        preflight_skills=preflight_skills,
        run_attempt=run_attempt,
    )


@pytest.mark.parametrize("lane", ["claude", "codex"])
def test_run_once_invokes_only_the_persisted_executor(
    home, tmp_path, repo, lane
):
    path, base = repo
    resolved = identity.resolve_requested_lane(lane)
    _make_job(
        specialist=resolved.specialist,
        requested_lane=resolved.requested_lane,
    )
    calls: list[tuple[str, str]] = []
    other = "codex" if lane == "claude" else "claude"
    registry = executors.registry.with_legacy_adapters(
        {
            lane: _neutral_fake_adapter(lane, calls),
            other: _neutral_fake_adapter(other, calls, forbidden=True),
        }
    )

    result = _run_once(
        tmp_path,
        path,
        base,
        specialist=resolved.specialist,
        execution=_execution(
            path,
            base,
            model=resolved.model,
            effort="max" if lane == "claude" else "high",
        ),
        executor_registry=registry,
    )

    assert result["ran"] is True
    assert calls == [
        (lane, "preflight"),
        (lane, "preflight_skills"),
        (lane, "run_attempt"),
    ]


def test_a_canonical_codex_job_with_no_installed_adapter_fails_closed(
    home, tmp_path, repo, monkeypatch
):
    path, base = repo
    job = _make_job(specialist="codex-builder", requested_lane="codex")
    claude_calls: list[str] = []

    def forbidden(stage):
        def call(*_args, **_kwargs):
            claude_calls.append(stage)
            raise AssertionError(f"Claude {stage} must not serve Codex")

        return call

    monkeypatch.setattr(claude_adapter, "preflight", forbidden("preflight"))
    monkeypatch.setattr(
        claude_adapter, "preflight_skills", forbidden("preflight_skills")
    )
    monkeypatch.setattr(
        claude_adapter, "run_claude_attempt", forbidden("run_attempt")
    )
    res = _run_once(
        tmp_path, path, base, specialist="codex-builder",
        execution=_execution(path, base, model="gpt-5.6-sol", effort="high"),
    )
    _assert_clean_refusal(job, res, "unsupported_routing")
    assert claude_calls == []


def test_contradictory_persisted_identity_fails_before_provider_invocation(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job(requested_lane="claude")
    with jdb.connect_closing() as conn:
        conn.execute(
            "UPDATE jobs SET model = 'gpt-5.6-sol' WHERE id = ?", (job.id,)
        )
        conn.commit()
    calls: list[tuple[str, str]] = []
    registry = executors.registry.with_legacy_adapters(
        {
            "claude": _neutral_fake_adapter("claude", calls),
            "codex": _neutral_fake_adapter("codex", calls),
        }
    )

    res = _run_once(
        tmp_path,
        path,
        base,
        executor_registry=registry,
    )

    _assert_clean_refusal(job, res, "unsupported_routing")
    assert calls == []


@pytest.mark.parametrize("missing", [False, True])
def test_invalid_candidate_identity_is_refused_before_custody_or_any_write(
    home, tmp_path, repo, missing
):
    path, base = repo
    job = _make_job()
    with jdb.connect_closing() as conn:
        if missing:
            conn.execute(
                "UPDATE jobs SET requested_lane = NULL, executor = NULL, "
                "specialist = NULL, model = NULL WHERE id = ?",
                (job.id,),
            )
        else:
            conn.execute(
                "UPDATE jobs SET executor = 'codex' WHERE id = ?", (job.id,)
            )
        conn.commit()
        corrupted = jdb.get_job(conn, job.id)
    before = _store_snapshot(job.id)
    calls: list[tuple[str, str]] = []
    registry = executors.registry.with_legacy_adapters(
        {
            "claude": _neutral_fake_adapter("claude", calls),
            "codex": _neutral_fake_adapter("codex", calls),
        }
    )

    result = _run_once(
        tmp_path,
        path,
        base,
        specialist=None if missing else "claude-builder",
        executor_registry=registry,
    )

    assert result["ran"] is False
    assert result["reason"] == "unsupported_routing"
    assert calls == []
    assert _store_snapshot(job.id) == before
    assert corrupted.revision == before["job"]["revision"]


def test_public_refusals_redact_secrets_and_cap_utf8_bytes_without_mutation(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    secret = "sk-super-secret-value-1234567890"
    hostile_key = "authorization=" + secret + ("\N{SNOWMAN}" * 1000)
    execution = _execution(path, base)
    execution["execution"][hostile_key] = "ignored"
    before = _store_snapshot(job.id)
    calls: list[tuple[str, str]] = []
    registry = executors.registry.with_legacy_adapters(
        {
            "claude": _neutral_fake_adapter("claude", calls),
            "codex": _neutral_fake_adapter("codex", calls),
        }
    )

    result = _run_once(
        tmp_path,
        path,
        base,
        execution=execution,
        executor_registry=registry,
    )

    assert result["ran"] is False
    assert result["reason"] == "invalid_execution_metadata"
    assert secret not in result["error"]
    assert len(result["error"].encode("utf-8")) <= 512
    assert calls == []
    assert _store_snapshot(job.id) == before


def test_long_secret_shaped_caller_specialist_is_bounded_without_custody(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    secret = "sk-caller-specialist-secret-123456789"
    hostile_specialist = "authorization=Bearer " + secret + ("\N{SNOWMAN}" * 1000)
    before = _store_snapshot(job.id)
    calls: list[tuple[str, str]] = []
    registry = executors.registry.with_legacy_adapters(
        {
            "claude": _neutral_fake_adapter("claude", calls),
            "codex": _neutral_fake_adapter("codex", calls),
        }
    )

    result = _run_once(
        tmp_path,
        path,
        base,
        specialist=hostile_specialist,
        executor_registry=registry,
    )

    assert result["ran"] is False
    assert result["reason"] == "unsupported_routing"
    assert secret not in result["error"]
    assert len(result["error"].encode("utf-8")) <= 512
    assert calls == []
    assert _store_snapshot(job.id) == before


def test_public_error_cap_is_deterministic_utf8_bytes():
    raw = "safe error " + ("\N{SNOWMAN}" * 1000)

    first = jobs_run._public({"ran": False, "error": raw})["error"]
    second = jobs_run._public({"ran": False, "error": raw})["error"]

    assert first == second
    assert first.endswith("…[truncated]")
    assert len(first.encode("utf-8")) <= jobs_run.MAX_PUBLIC_ERROR_BYTES


def test_corrupted_secret_shaped_identity_is_not_reflected_or_claimed(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    secret = "Bearer sk-secret-persisted-123456789"
    with jdb.connect_closing() as conn:
        conn.execute("UPDATE jobs SET executor = ? WHERE id = ?", (secret, job.id))
        conn.commit()
    before = _store_snapshot(job.id)
    calls: list[tuple[str, str]] = []
    registry = executors.registry.with_legacy_adapters(
        {
            "claude": _neutral_fake_adapter("claude", calls),
            "codex": _neutral_fake_adapter("codex", calls),
        }
    )

    result = _run_once(
        tmp_path, path, base, executor_registry=registry
    )

    assert result["ran"] is False
    assert secret not in result["error"]
    assert len(result["error"].encode("utf-8")) <= 512
    assert calls == []
    assert _store_snapshot(job.id) == before


def test_an_unassigned_legacy_job_has_no_implicit_claude_route(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    with jdb.connect_closing() as conn:
        conn.execute(
            "UPDATE jobs SET requested_lane = NULL, executor = NULL, "
            "specialist = NULL, model = NULL WHERE id = ?",
            (job.id,),
        )
        conn.commit()
        job = jdb.get_job(conn, job.id)
    res = _run_once(tmp_path, path, base, specialist=None)
    _assert_clean_refusal(job, res, "unsupported_routing")


def test_approved_metadata_that_contradicts_the_route_fails_closed(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    res = _run_once(
        tmp_path, path, base, execution=_execution(path, base, model="gpt-5.6-sol")
    )
    _assert_clean_refusal(job, res, "unsupported_routing")


def test_an_unrunnable_worker_still_settles_the_attempt(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    res = _run_once(tmp_path, path, base, worker=[str(tmp_path / "nope")])

    assert res["ran"] is True
    assert (res["status"], res["failure_class"]) == ("failed", "infrastructure")
    with jdb.connect_closing() as conn:
        assert jdb.get_job(conn, job.id).status == "working"
        assert len(jdb.get_receipts(conn, job.id)) == 1


# ---------------------------------------------------------------------------
# Capability containment and ledger volume
# ---------------------------------------------------------------------------


def test_the_claim_token_never_escapes_the_orchestrator(
    home, tmp_path, repo, capsys, monkeypatch
):
    path, base = repo
    job = _make_job()

    # Capture the real capability the runner was issued, so this checks for the
    # token itself rather than for the word "claim_token".
    issued = []
    real_claim = jdb.claim_job

    def spy(*args, **kwargs):
        claim = real_claim(*args, **kwargs)
        if claim is not None:
            issued.append(claim.claim_token)
        return claim

    monkeypatch.setattr(jobs_run.jdb, "claim_job", spy)
    res = _run_once(tmp_path, path, base)
    assert res["ran"] is True
    assert len(issued) == 1
    token = issued[0]

    with jdb.connect_closing() as conn:
        row = conn.execute(
            "SELECT claim_token FROM jobs WHERE id = ?", (job.id,)
        ).fetchone()
        assert row["claim_token"] is None
        blob = json.dumps(
            [jdb.get_events(conn, job.id), jdb.get_receipts(conn, job.id),
             jdb.get_attempts(conn, job.id), jdb.get_job(conn, job.id).to_dict()]
        )
    assert token not in json.dumps(res)
    assert token not in blob
    assert token not in res["worktree"] and token not in res["branch"]
    captured = capsys.readouterr()
    assert token not in captured.out and token not in captured.err


def test_the_claim_token_is_absent_from_failure_text(home, tmp_path, repo, monkeypatch):
    path, base = repo
    _make_job()
    issued = []
    real_claim = jdb.claim_job

    def spy(*args, **kwargs):
        claim = real_claim(*args, **kwargs)
        if claim is not None:
            issued.append(claim.claim_token)
        return claim

    monkeypatch.setattr(jobs_run.jdb, "claim_job", spy)
    # An adapter that blows up mid-run: the reported error text is the most
    # likely place for a capability to ride out on.
    registry = executors.registry.with_legacy_adapters(
        {
            "claude": executors.LegacyExecutorAdapter(
                name="claude",
                preflight=claude_adapter.preflight,
                preflight_skills=claude_adapter.preflight_skills,
                run_attempt=lambda envelope, spec, **kw: (_ for _ in ()).throw(
                    RuntimeError(f"exploded handling {envelope!r}")
                ),
            )
        }
    )
    res = _run_once(tmp_path, path, base, executor_registry=registry)

    # The attempt started, so it settled — with the failure text on it.
    assert res["ran"] is True
    assert res["reason"] == "adapter_error"
    assert issued and issued[0] not in json.dumps(res)
    assert issued[0] not in (res["error"] or "")
    with jdb.connect_closing() as conn:
        receipt = jdb.get_receipts(conn, res["job"]["id"])[0]
    assert issued[0] not in json.dumps(receipt)


def test_run_once_writes_no_claim_token_file(home, tmp_path, repo):
    path, base = repo
    _make_job()
    before = set(p.name for p in home.iterdir())
    _run_once(tmp_path, path, base)
    assert set(p.name for p in home.iterdir()) - before <= {"jobs.db",
                                                            "jobs.db-wal",
                                                            "jobs.db-shm"}


def test_a_quiet_run_produces_no_heartbeat_events(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    _run_once(tmp_path, path, base, heartbeat_seconds=60)
    assert _events(job.id).count("claim_heartbeat") == 0


def test_heartbeats_stay_bounded_on_a_slow_worker(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    # Same indentation as COMMIT_AND_SUCCEED so textwrap.dedent keeps the block.
    slow = _worker(tmp_path, "\n    time.sleep(2.5)" + COMMIT_AND_SUCCEED, name="slow.py")
    _run_once(tmp_path, path, base, worker=slow, heartbeat_seconds=1)

    beats = _events(job.id).count("claim_heartbeat")
    # Alive, but nowhere near one event per poll of a ~2.5s run.
    assert 1 <= beats <= 4, beats


def test_neither_new_module_touches_kanban(home):
    for module in (jobs_run, __import__(
        "hermes_cli.jobs_adapter_claude", fromlist=["x"]
    )):
        source = Path(module.__file__).read_text()
        assert "kanban" not in source.lower()


# ---------------------------------------------------------------------------
# The CLI verb
# ---------------------------------------------------------------------------


def test_run_once_cli_drives_the_same_seam(home, tmp_path, repo, capsys):
    path, base = repo
    job = _make_job()
    worker = _worker(tmp_path)
    ef = tmp_path / "execution.json"
    ef.write_text(json.dumps(_execution(path, base)))

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    jobs_cli.build_parser(sub)
    args = parser.parse_args([
        "jobs", "run-once",
        "--worker", " ".join(worker),
        "--workspace-root", str(tmp_path / "ws"),
        "--execution-file", str(ef),
        "--worker-id", "cli-runner",
        "--specialist", "claude-builder",
        "--wall-clock-seconds", "60",
        "--json",
    ])
    rc = jobs_cli.jobs_command(args)
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out)
    assert payload["ran"] is True
    assert payload["status"] == "succeeded"
    with jdb.connect_closing() as conn:
        assert len(jdb.get_attempts(conn, job.id)) == 1
        assert len(jdb.get_receipts(conn, job.id)) == 1


def test_run_once_cli_refuses_a_malformed_execution_file(home, tmp_path, repo, capsys):
    path, base = repo
    _make_job()
    ef = tmp_path / "execution.json"
    ef.write_text("{not json")

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    jobs_cli.build_parser(sub)
    args = parser.parse_args([
        "jobs", "run-once", "--worker", "/bin/true",
        "--workspace-root", str(tmp_path / "ws"),
        "--execution-file", str(ef), "--worker-id", "cli-runner",
    ])
    rc = jobs_cli.jobs_command(args)
    err = capsys.readouterr().err
    assert rc == 2
    assert err.strip()


# ---------------------------------------------------------------------------
# Settlement and its final receipt are one outcome, or they are nothing
# ---------------------------------------------------------------------------


def test_a_refused_receipt_write_settles_absolutely_nothing(
    home, tmp_path, repo, monkeypatch
):
    path, base = repo
    job = _make_job()

    def refuse_the_insert():
        raise sqlite3.IntegrityError("receipt storage refused the write")

    monkeypatch.setattr(jdb, "_new_receipt_id", refuse_the_insert)
    with pytest.raises(sqlite3.IntegrityError):
        _run_once(tmp_path, path, base)

    with jdb.connect_closing() as conn:
        attempts = jdb.get_attempts(conn, job.id)
        after = jdb.get_job(conn, job.id)
        receipts = jdb.get_receipts(conn, job.id)
    kinds = _events(job.id)

    # The attempt is still running, so recovery can still repair it.
    assert [a["status"] for a in attempts] == ["running"]
    assert receipts == []
    assert after.status == "working"
    assert after.claimed_by == "runner-1"
    assert after.current_attempt_id == attempts[0]["id"]
    assert "attempt_finished" not in kinds
    assert "job_transition" not in kinds
    assert "receipt_added" not in kinds


def test_a_setup_failure_after_the_attempt_starts_reports_the_real_attempt(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    blocked = tmp_path / "workspace-root-is-a-file"
    blocked.write_text("not a directory\n")

    res = jobs_run.run_once(
        worker_command=_worker(tmp_path),
        workspace_root=blocked,
        worker_id="runner-1",
        specialist="claude-builder",
        execution=_execution(path, base),
        wall_clock_seconds=60,
        heartbeat_seconds=60,
        now=T0,
    )

    assert res["attempt_id"] is not None
    assert res["receipt_id"] is not None
    assert (res["status"], res["failure_class"]) == ("failed", "infrastructure")

    with jdb.connect_closing() as conn:
        attempts = jdb.get_attempts(conn, job.id)
        receipts = jdb.get_receipts(conn, job.id)
        after = jdb.get_job(conn, job.id)

    assert [a["id"] for a in attempts] == [res["attempt_id"]]
    assert attempts[0]["status"] == "failed"
    assert len(receipts) == 1
    assert receipts[0]["id"] == res["receipt_id"]
    assert receipts[0]["idempotency_key"] == f"{res['attempt_id']}:final"
    assert receipts[0]["data"]["status"] == "failed"
    assert receipts[0]["data"]["failure_class"] == "infrastructure"
    assert (after.status, after.step) == ("working", "routing")
    assert after.claimed_by is None


def test_every_terminal_attempt_including_a_recovered_one_has_one_final_receipt(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    with jdb.connect_closing() as conn:
        claim = jdb.claim_job(
            conn, worker="died", job=job.number,
            specialist=job.specialist, lease_seconds=60, now=T0,
        )
        abandoned = jdb.start_attempt(
            conn, job.id, claim_token=claim.claim_token,
            specialist="claude-builder", now=T0,
        )

    _run_once(tmp_path, path, base, now=T0 + 10_000)

    with jdb.connect_closing() as conn:
        attempts = jdb.get_attempts(conn, job.id)
        receipts = jdb.get_receipts(conn, job.id)

    assert [a["status"] for a in attempts] == ["interrupted", "succeeded"]
    assert {r["idempotency_key"] for r in receipts} == {
        f"{a['id']}:final" for a in attempts
    }
    recovered = next(r for r in receipts if r["attempt_id"] == abandoned)
    assert recovered["data"]["status"] == "interrupted"
    assert recovered["data"]["failure_class"] == "infrastructure"


# ---------------------------------------------------------------------------
# The final-receipt namespace belongs to finalization alone
# ---------------------------------------------------------------------------


def test_a_public_caller_cannot_pre_seed_a_final_receipt(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    with jdb.connect_closing() as conn:
        claim = jdb.claim_job(
            conn, worker="hostile", job=job.number,
            specialist=job.specialist, lease_seconds=600, now=T0,
        )
        aid = jdb.start_attempt(
            conn, job.id, claim_token=claim.claim_token,
            specialist="claude-builder", now=T0,
        )
        with pytest.raises(ValueError, match="final"):
            jdb.add_receipt(
                conn, job.id, data={"status": "succeeded"}, attempt_id=aid,
                idempotency_key=f"{aid}:final",
            )
        assert jdb.get_receipts(conn, job.id) == []


def test_a_settled_final_receipt_cannot_be_contradicted_by_a_replay(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    res = _run_once(tmp_path, path, base, worker=_worker(tmp_path, FAIL))
    with jdb.connect_closing() as conn:
        before = jdb.get_job(conn, job.id).revision
        with pytest.raises((ValueError, jdb.InvalidTransition)):
            jdb.settle_attempt(
                conn, res["attempt_id"], status="failed",
                failure_class="implementation",
                receipt={"status": "failed", "failure_class": "implementation",
                         "hostile": True},
            )
        assert len(jdb.get_receipts(conn, job.id)) == 1
        assert jdb.get_receipts(conn, job.id)[0]["id"] == res["receipt_id"]
        assert jdb.get_job(conn, job.id).revision == before


# ---------------------------------------------------------------------------
# Exact replay for a caller that lost its output after the commit
# ---------------------------------------------------------------------------


def test_a_settled_run_replays_exactly_for_the_same_request_id(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    launches = tmp_path / "launches.log"
    env = {"JOBS_TEST_LAUNCH_LOG": str(launches)}

    first = _run_once(tmp_path, path, base, request_id="req-1", extra_env=env)
    assert first["ran"] is True and first["status"] == "succeeded"

    with jdb.connect_closing() as conn:
        events_before = jdb.get_events(conn, job.id)
        revision_before = jdb.get_job(conn, job.id).revision

    second = _run_once(tmp_path, path, base, request_id="req-1", extra_env=env)

    assert second == first
    assert launches.read_text().count("\n") == 1
    with jdb.connect_closing() as conn:
        assert jdb.get_events(conn, job.id) == events_before
        assert jdb.get_job(conn, job.id).revision == revision_before
        assert len(jdb.get_attempts(conn, job.id)) == 1
        assert len(jdb.get_receipts(conn, job.id)) == 1


def test_a_different_request_id_is_not_a_replay(home, tmp_path, repo):
    path, base = repo
    _make_job()
    _make_job(name="Second")
    first = _run_once(tmp_path, path, base, request_id="req-1")
    second = _run_once(tmp_path, path, base, request_id="req-2")
    assert first["attempt_id"] != second["attempt_id"]
    assert first["job"]["id"] != second["job"]["id"]


def _replay_is_exact(
    tmp_path, path, base, job, request_id, env, executor_registry=None
):
    """Run twice under one request id; return ``(first, second)``.

    Asserts the whole point in between: the second call is a read. Every one of
    these paths leaves the Job claimable again, so a replay that does not
    short-circuit runs a *second* real attempt — which is exactly what the
    counters here would catch.
    """
    first = _run_once(
        tmp_path,
        path,
        base,
        request_id=request_id,
        extra_env=env,
        executor_registry=executor_registry,
    )
    with jdb.connect_closing() as conn:
        events = jdb.get_events(conn, job.id)
        revision = jdb.get_job(conn, job.id).revision
    second = _run_once(
        tmp_path,
        path,
        base,
        request_id=request_id,
        extra_env=env,
        executor_registry=executor_registry,
    )
    with jdb.connect_closing() as conn:
        assert jdb.get_events(conn, job.id) == events
        assert jdb.get_job(conn, job.id).revision == revision
        assert len(jdb.get_attempts(conn, job.id)) == 1
        assert len(jdb.get_receipts(conn, job.id)) == 1
    return first, second


def test_a_setup_failure_replays_exactly_for_the_same_request_id(
    home, tmp_path, repo, monkeypatch
):
    path, base = repo
    job = _make_job()
    launches = tmp_path / "launches.log"
    adapter_calls = []

    def explode(envelope, spec, **kw):
        adapter_calls.append(envelope.attempt_id)
        raise RuntimeError("worktree setup failed: no space left on device")

    registry = executors.registry.with_legacy_adapters(
        {
            "claude": executors.LegacyExecutorAdapter(
                name="claude",
                preflight=claude_adapter.preflight,
                preflight_skills=claude_adapter.preflight_skills,
                run_attempt=explode,
            )
        }
    )

    first, second = _replay_is_exact(
        tmp_path, path, base, job, "lost-setup",
        {"JOBS_TEST_LAUNCH_LOG": str(launches)},
        executor_registry=registry,
    )

    assert first["ran"] is True and first["reason"] == "adapter_error"
    assert "no space left on device" in first["error"]
    # The whole machine-readable result, not a handful of fields.
    assert second == first
    assert len(adapter_calls) == 1
    assert not launches.exists()


def test_an_unsafe_receipt_replays_exactly_for_the_same_request_id(
    home, tmp_path, repo, monkeypatch
):
    path, base = repo
    job = _make_job()
    launches = tmp_path / "launches.log"
    real = claude_adapter.run_claude_attempt

    def poisoned(envelope, spec, **kw):
        # A real worker really runs; only the evidence coming back is unusable.
        outcome = real(envelope, spec, **kw)
        return dataclasses.replace(
            outcome, receipt={**outcome.receipt, "worker_password": "hunter2"}
        )

    registry = executors.registry.with_legacy_adapters(
        {
            "claude": executors.LegacyExecutorAdapter(
                name="claude",
                preflight=claude_adapter.preflight,
                preflight_skills=claude_adapter.preflight_skills,
                run_attempt=poisoned,
            )
        }
    )

    first, second = _replay_is_exact(
        tmp_path, path, base, job, "lost-receipt",
        {"JOBS_TEST_LAUNCH_LOG": str(launches)},
        executor_registry=registry,
    )

    assert first["ran"] is True and first["reason"] == "unsafe_receipt"
    assert "worker_password" in first["error"]
    assert second == first
    assert launches.read_text().count("\n") == 1
    # The replayed answer is still evidence, not a leak.
    assert "hunter2" not in json.dumps(second)


# ---------------------------------------------------------------------------
# A zero-diff success is an implementation failure, not a finished Job
# ---------------------------------------------------------------------------


NO_CHANGE_BUT_CLAIMS_SUCCESS = """
    json.dump({"outcome": "succeeded"}, open(a.result_file, "w"))
"""


def test_a_zero_diff_claimed_success_never_finishes_the_job(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    res = _run_once(
        tmp_path, path, base,
        worker=_worker(tmp_path, NO_CHANGE_BUT_CLAIMS_SUCCESS),
    )

    assert (res["status"], res["failure_class"]) == ("failed", "implementation")
    assert (res["job_status"], res["job_step"]) == ("working", "correcting")
    with jdb.connect_closing() as conn:
        after = jdb.get_job(conn, job.id)
        receipt = jdb.get_receipts(conn, job.id)[0]["data"]
    assert after.status != "finished"
    assert receipt["diff_empty"] is True


# ---------------------------------------------------------------------------
# Public exit codes — automation must not read a refusal as success
# ---------------------------------------------------------------------------


def _cli(argv):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    jobs_cli.build_parser(sub)
    return jobs_cli.jobs_command(parser.parse_args(["jobs", *argv]))


def test_run_once_cli_exits_non_zero_for_unsupported_routing(
    home, tmp_path, repo, capsys
):
    path, base = repo
    job = _make_job(specialist="codex-builder", requested_lane="codex")
    ef = tmp_path / "execution.json"
    ef.write_text(
        json.dumps(_execution(path, base, model="gpt-5.6-sol", effort="high"))
    )
    ws = tmp_path / "ws"

    rc = _cli([
        "run-once", "--worker", "/bin/true", "--workspace-root", str(ws),
        "--execution-file", str(ef), "--worker-id", "cli",
        "--specialist", "gpt-builder", "--json",
    ])
    capsys.readouterr()

    assert rc != 0
    assert not ws.exists()
    with jdb.connect_closing() as conn:
        assert jdb.get_attempts(conn, job.id) == []
        assert jdb.get_receipts(conn, job.id) == []
    assert "claim_acquired" not in _events(job.id)


def test_run_once_cli_exits_zero_when_nothing_is_eligible(
    home, tmp_path, repo, capsys
):
    path, base = repo
    ef = tmp_path / "execution.json"
    ef.write_text(json.dumps(_execution(path, base)))
    rc = _cli([
        "run-once", "--worker", "/bin/true",
        "--workspace-root", str(tmp_path / "ws"),
        "--execution-file", str(ef), "--worker-id", "cli", "--json",
    ])
    capsys.readouterr()
    assert rc == 0


def test_run_once_cli_refuses_non_standard_json_in_the_execution_file(
    home, tmp_path, repo, capsys
):
    path, base = repo
    job = _make_job()
    ef = tmp_path / "execution.json"
    ef.write_text('{"execution": {"max_turns": NaN}}')
    rc = _cli([
        "run-once", "--worker", "/bin/true",
        "--workspace-root", str(tmp_path / "ws"),
        "--execution-file", str(ef), "--worker-id", "cli",
    ])
    capsys.readouterr()
    assert rc == 2
    with jdb.connect_closing() as conn:
        assert jdb.get_attempts(conn, job.id) == []


def test_run_once_cli_replays_a_settled_request(home, tmp_path, repo, capsys):
    path, base = repo
    _make_job()
    worker = _worker(tmp_path)
    ef = tmp_path / "execution.json"
    ef.write_text(json.dumps(_execution(path, base)))
    argv = [
        "run-once", "--worker", " ".join(worker),
        "--workspace-root", str(tmp_path / "ws"), "--execution-file", str(ef),
        "--worker-id", "cli", "--request-id", "cli-req-1",
        "--specialist", "claude-builder",
        "--wall-clock-seconds", "60", "--json",
    ]
    assert _cli(argv) == 0
    first = json.loads(capsys.readouterr().out)
    assert _cli(argv) == 0
    second = json.loads(capsys.readouterr().out)
    assert second == first


# ---------------------------------------------------------------------------
# Replay authority: one protected, globally unique, atomically owned seam
#
# Everything below is hostile. The ledger is public — any caller can append any
# event with any key to any Job — so these prove that no amount of forged,
# repointed, or pre-seeded *evidence* can decide what a request "was".
# ---------------------------------------------------------------------------


LEGACY_KEY = "run:{}".format  # the namespace an earlier build treated as authority


def _forge_event(job_id, request_id, data):
    """Append a settled-looking event through the ordinary public evidence path."""
    with jdb.connect_closing() as conn:
        return jdb.append_event(
            conn, job_id, "run_settled",
            data=data, idempotency_key=LEGACY_KEY(request_id),
        )


def _forged_record(request_id, **overrides):
    record = {
        "ran": True, "reason": "completed", "recovered": [], "error": None,
        "request_id": request_id,
        "job_id": "j_forged", "job_number": 9999, "job_label": "#9999 Forged",
        "job_status": "finished", "job_step": "complete",
        "attempt_id": "a_forged", "ordinal": 1,
        "status": "succeeded", "failure_class": None,
        "commit": "f" * 40, "branch": "forged", "worktree": "/tmp/forged",
        "receipt_id": "r_forged", "routing_reason": "forged",
    }
    record.update(overrides)
    return record


def _all_events(job_id):
    with jdb.connect_closing() as conn:
        return jdb.get_events(conn, job_id)


def _requests():
    with jdb.connect_closing() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM job_run_requests ORDER BY request_id"
        ).fetchall()]


def test_a_forged_event_on_another_job_cannot_become_a_replay(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    decoy = _make_job(name="Decoy")
    _forge_event(decoy.id, "req-forged", _forged_record("req-forged"))

    res = _run_once(tmp_path, path, base, request_id="req-forged")

    assert res["ran"] is True
    assert res["attempt_id"] != "a_forged"
    assert res["receipt_id"] != "r_forged"
    assert res["job"]["id"] in (job.id, decoy.id)
    assert res["branch"] != "forged"
    with jdb.connect_closing() as conn:
        assert jdb.get_attempt(conn, res["attempt_id"]) is not None
    # And the request now owns the real settlement, forever.
    assert _run_once(tmp_path, path, base, request_id="req-forged") == res


def test_a_forged_event_cannot_repoint_a_settled_request(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    other = _make_job(name="Other")
    first = _run_once(tmp_path, path, base, request_id="req-repoint")
    assert first["ran"] is True

    _forge_event(other.id, "req-repoint", _forged_record("req-repoint"))
    _forge_event(job.id, "req-repoint-2", _forged_record("req-repoint"))

    assert _run_once(tmp_path, path, base, request_id="req-repoint") == first


def test_a_pre_seeded_event_cannot_contradict_the_real_settlement(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    _forge_event(job.id, "req-seeded", _forged_record(
        "req-seeded", job_id=job.id, attempt_id="a_seed", receipt_id="r_seed",
        status="failed", failure_class="implementation",
    ))

    res = _run_once(tmp_path, path, base, request_id="req-seeded")

    assert res["status"] == "succeeded"
    assert res["attempt_id"] != "a_seed"
    assert res["receipt_id"] != "r_seed"
    with jdb.connect_closing() as conn:
        attempts = jdb.get_attempts(conn, job.id)
        receipts = jdb.get_receipts(conn, job.id)
    assert [a["id"] for a in attempts] == [res["attempt_id"]]
    assert [r["id"] for r in receipts] == [res["receipt_id"]]
    assert attempts[0]["status"] == "succeeded"
    assert _run_once(tmp_path, path, base, request_id="req-seeded") == res


def test_one_request_id_settles_once_across_two_jobs_and_two_runners(
    home, tmp_path, repo
):
    path, base = repo
    first_job = _make_job(name="First")
    second_job = _make_job(name="Second")
    log = tmp_path / "launches.log"
    worker = _worker(tmp_path, COMMIT_AND_SUCCEED)
    env = {"JOBS_TEST_LAUNCH_LOG": str(log)}
    results = []
    barrier = threading.Barrier(2)

    def go(worker_id):
        barrier.wait()
        try:
            results.append(_run_once(
                tmp_path, path, base, worker=worker, worker_id=worker_id,
                request_id="one-request", extra_env=env,
            ))
        except Exception as exc:  # noqa: BLE001 - recorded, asserted below
            results.append(exc)

    threads = [threading.Thread(target=go, args=(f"runner-{i}",)) for i in (1, 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert len(results) == 2
    assert not [r for r in results if isinstance(r, Exception)], results
    ran = [r for r in results if r["ran"]]
    assert len(ran) == 1, results
    assert [r["reason"] for r in results if not r["ran"]] == ["request_in_progress"]

    with jdb.connect_closing() as conn:
        attempts = jdb.get_attempts(conn, first_job.id) + jdb.get_attempts(
            conn, second_job.id
        )
        receipts = jdb.get_receipts(conn, first_job.id) + jdb.get_receipts(
            conn, second_job.id
        )
    kinds = _events(first_job.id) + _events(second_job.id)
    assert len(attempts) == 1
    assert len(receipts) == 1
    assert kinds.count("claim_acquired") == 1
    assert kinds.count("attempt_started") == 1
    assert len(log.read_text().strip().splitlines()) == 1
    # One globally unique owner, and every later replay is that one answer.
    assert [r["request_id"] for r in _requests()] == ["one-request"]
    assert _run_once(
        tmp_path, path, base, worker=worker, request_id="one-request", extra_env=env,
    ) == ran[0]


# ---------------------------------------------------------------------------
# The protected envelope write is the only way in, and it screens everything
# ---------------------------------------------------------------------------


PROBE_SECRET = "sk-hostilehostile0123456789"

HOSTILE_ENVELOPES = (
    ("secret_named_key", {"api_key": PROBE_SECRET}),
    ("capability_named_key", {"claim_token": "c_deadbeefdeadbeef"}),
    # A secret-shaped *value* under an innocuous key. Redacting it and settling
    # anyway would mutate the attempt, the Job, and the ledger on the strength of
    # material the caller was never allowed to hand a settlement.
    ("secret_shaped_value", {"note": PROBE_SECRET}),
    ("secret_shaped_nested", {"outer": {"inner": [PROBE_SECRET]}}),
    ("nan", {"latency": float("nan")}),
    ("infinity", {"latency": float("inf")}),
    ("nested_non_finite", {"outer": {"inner": [1, float("-inf")]}}),
    ("malformed_value", {"blob": {1, 2, 3}}),
    ("non_string_key", {7: "seven"}),
    ("oversize", {f"k{i}": "A" * 2048 for i in range(40)}),
)


PROBE_REPO = "/probe/repository"
PROBE_BASE = "b" * 40
PROBE_COMMIT = "c" * 40


def _claimed_attempt(job, request_id, *, worker="hostile-runner", now=T0):
    """Reserve, claim, and start one bound attempt. Returns (token, attempt_id).

    Started the way ``run_once`` starts one — with a repository and an approved
    base — because these probes go on to assert V3 replay, and an attempt with
    no recorded base is legacy history that (correctly) carries no V3 authority.
    """
    with jdb.connect_closing() as conn:
        jdb.reserve_request(
            conn, request_id, owner=worker, lease_seconds=600, now=now
        )
        claim = jdb.claim_job(
            conn, worker=worker, job=job.number,
            specialist=job.specialist, lease_seconds=600, now=now,
        )
        aid = jdb.start_attempt(
            conn, job.id, claim_token=claim.claim_token,
            repository=PROBE_REPO, base_commit=PROBE_BASE,
            request_id=request_id, request_owner=worker, now=now,
        )
        return claim.claim_token, aid


def _final_receipt(job, attempt_id):
    return {
        "schema_version": 1, "kind": "jobs-attempt-final", "source": "probe",
        "job_id": job.id, "attempt_id": attempt_id,
        "status": "succeeded", "failure_class": None,
    }


@pytest.mark.parametrize("label,payload", HOSTILE_ENVELOPES,
                         ids=[c[0] for c in HOSTILE_ENVELOPES])
def test_hostile_envelope_material_is_refused_before_any_mutation(
    home, tmp_path, repo, label, payload
):
    job = _make_job()
    request_id = f"hostile-{label}"
    token, aid = _claimed_attempt(job, request_id)
    before = _events(job.id)
    with jdb.connect_closing() as conn:
        revision = jdb.get_job(conn, job.id).revision

    with jdb.connect_closing() as conn, pytest.raises(Exception):
        jdb.settle_attempt(
            conn, aid, status="succeeded", claim_token=token,
            receipt=_final_receipt(job, aid), request_id=request_id,
            request_owner="hostile-runner", request_data=payload, now=T0,
        )

    with jdb.connect_closing() as conn:
        assert jdb.get_attempt(conn, aid)["status"] == "running"
        assert jdb.get_receipts(conn, job.id) == []
        assert jdb.get_job(conn, job.id).revision == revision
        assert jdb.find_run_response(conn, request_id) is None
    assert _events(job.id) == before
    assert [r["response"] for r in _requests()] == [None]
    # Nothing hostile reached durable storage, so nothing hostile can be replayed.
    assert PROBE_SECRET not in json.dumps([_events(job.id), _requests()], default=str)

    # Repairable: the same attempt still settles once the envelope is clean.
    with jdb.connect_closing() as conn:
        done = jdb.settle_attempt(
            conn, aid, status="succeeded", claim_token=token,
            receipt=_final_receipt(job, aid), request_id=request_id,
            request_owner="hostile-runner",
            request_data={"ran": True, "recovered": []}, reason="completed",
            commit=PROBE_COMMIT, now=T0,
        )
        assert jdb.find_run_response(conn, request_id) == done["run_record"]


@pytest.mark.parametrize("label,payload", HOSTILE_ENVELOPES,
                         ids=[c[0] for c in HOSTILE_ENVELOPES])
def test_hostile_final_receipt_material_is_refused_before_any_mutation(
    home, tmp_path, repo, label, payload
):
    """The final receipt is a settlement input too, and it gets the same screen.

    The envelope and the receipt are written by one transaction and are equally
    durable, so a control that only guards one of them guards neither.
    """
    job = _make_job()
    request_id = f"hostile-receipt-{label}"
    token, aid = _claimed_attempt(job, request_id)
    before = _events(job.id)
    with jdb.connect_closing() as conn:
        revision = jdb.get_job(conn, job.id).revision

    with jdb.connect_closing() as conn, pytest.raises(Exception):
        jdb.settle_attempt(
            conn, aid, status="succeeded", claim_token=token,
            receipt={**_final_receipt(job, aid), **payload},
            request_id=request_id, request_owner="hostile-runner",
            request_data={"ran": True, "recovered": []}, reason="completed",
            commit=PROBE_COMMIT, now=T0,
        )

    with jdb.connect_closing() as conn:
        assert jdb.get_attempt(conn, aid)["status"] == "running"
        assert jdb.get_receipts(conn, job.id) == []
        assert jdb.get_job(conn, job.id).revision == revision
        assert jdb.find_run_response(conn, request_id) is None
    assert _events(job.id) == before
    assert PROBE_SECRET not in _durable_text()

    # Repairable: the same attempt still settles once the receipt is clean.
    with jdb.connect_closing() as conn:
        jdb.settle_attempt(
            conn, aid, status="succeeded", claim_token=token,
            receipt=_final_receipt(job, aid), request_id=request_id,
            request_owner="hostile-runner",
            request_data={"ran": True, "recovered": []}, reason="completed",
            commit=PROBE_COMMIT, now=T0,
        )


def _durable_text() -> str:
    """Everything a settlement could leak *through*, as one searchable string.

    ``jobs.claim_token`` is deliberately not in it: the custody column is where
    the capability is supposed to live. What must never hold it is evidence —
    events, receipts, and the stored replay envelope.
    """
    with jdb.connect_closing() as conn:
        return json.dumps([
            [dict(r) for r in conn.execute("SELECT * FROM job_events")],
            [dict(r) for r in conn.execute("SELECT * FROM job_receipts")],
            [dict(r) for r in conn.execute(
                "SELECT request_id, owner, job_id, attempt_id, receipt_id, "
                "response FROM job_run_requests"
            )],
        ], default=str)


@pytest.mark.parametrize("where", ["envelope", "receipt"])
def test_the_live_capability_is_refused_under_a_benign_key(
    home, tmp_path, repo, where
):
    """The custody capability is not content, whatever key it arrives under.

    A screen that only refuses keys *named* like a capability refuses a spelling,
    not a secret. The value itself is compared against the live claim, so calling
    it ``note`` buys nothing.
    """
    job = _make_job()
    request_id = f"capability-{where}"
    token, aid = _claimed_attempt(job, request_id)
    before = _events(job.id)
    with jdb.connect_closing() as conn:
        revision = jdb.get_job(conn, job.id).revision

    receipt = _final_receipt(job, aid)
    envelope = {"ran": True, "reason": "completed", "recovered": []}
    if where == "envelope":
        envelope = {**envelope, "note": token}
    else:
        receipt = {**receipt, "note": token}

    with jdb.connect_closing() as conn, pytest.raises(jx.UnsafeReceipt):
        jdb.settle_attempt(
            conn, aid, status="succeeded", claim_token=token, receipt=receipt,
            request_id=request_id, request_owner="hostile-runner",
            request_data=envelope, now=T0,
        )

    with jdb.connect_closing() as conn:
        assert jdb.get_attempt(conn, aid)["status"] == "running"
        assert jdb.get_receipts(conn, job.id) == []
        assert jdb.get_job(conn, job.id).revision == revision
        assert jdb.find_run_response(conn, request_id) is None
    assert _events(job.id) == before
    # Asserted, never displayed: the value stays out of the failure output too.
    assert token not in _durable_text()


def test_a_long_error_is_bounded_in_the_stored_envelope(home, tmp_path, repo):
    job = _make_job()
    token, aid = _claimed_attempt(job, "bounded-error")
    with jdb.connect_closing() as conn:
        done = jdb.settle_attempt(
            conn, aid, status="succeeded", claim_token=token,
            receipt=_final_receipt(job, aid), request_id="bounded-error",
            request_owner="hostile-runner",
            request_data={"ran": True, "recovered": []},
            reason="completed", error="E" * 70_000,
            commit=PROBE_COMMIT, now=T0,
        )
    assert len(done["run_record"]["error"]) < 4096


# ---------------------------------------------------------------------------
# A stored envelope is checked against the world before it is ever returned
# ---------------------------------------------------------------------------


def _corrupt_request(request_id, **columns):
    sets = ", ".join(f"{c} = ?" for c in columns)
    with jdb.connect_closing() as conn:
        conn.execute(
            f"UPDATE job_run_requests SET {sets} WHERE request_id = ?",
            (*columns.values(), request_id),
        )
        conn.commit()


CONTRADICTIONS = {
    "cross_job": {"job_id": "j_elsewhere"},
    "cross_attempt": {"attempt_id": "a_elsewhere"},
    "dangling_receipt": {"receipt_id": "r_elsewhere"},
    "cross_request": {"response": json.dumps(
        _forged_record("some-other-request"), sort_keys=True
    )},
    "contradictory_status": {"response": json.dumps(
        _forged_record("bad-envelope", status="failed"), sort_keys=True
    )},
}


@pytest.mark.parametrize("label", sorted(CONTRADICTIONS))
def test_a_contradictory_stored_envelope_is_never_returned(
    home, tmp_path, repo, label
):
    path, base = repo
    job = _make_job()
    first = _run_once(tmp_path, path, base, request_id="bad-envelope")
    assert first["ran"] is True
    before = _events(job.id)
    _corrupt_request("bad-envelope", **CONTRADICTIONS[label])

    with jdb.connect_closing() as conn:
        with pytest.raises(jdb.ReplayIntegrity):
            jdb.find_run_response(conn, "bad-envelope")
    with pytest.raises(jdb.ReplayIntegrity):
        _run_once(tmp_path, path, base, request_id="bad-envelope")

    with jdb.connect_closing() as conn:
        assert len(jdb.get_attempts(conn, job.id)) == 1
        assert len(jdb.get_receipts(conn, job.id)) == 1
    assert _events(job.id) == before


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _stored(request_id) -> dict:
    with jdb.connect_closing() as conn:
        return dict(conn.execute(
            "SELECT * FROM job_run_requests WHERE request_id = ?", (request_id,)
        ).fetchone())


def _write_response(request_id, record, *, reseal: bool) -> None:
    """Replace the stored response. ``reseal`` recomputes its digest too.

    Two different attackers. Without ``reseal`` this is one corrupted column —
    a bad write, a torn page, a careless UPDATE. With it, the row is internally
    consistent and only the world disagrees, which is the one that matters: a
    digest is not a signature and anybody who can write the response can write
    its digest.
    """
    encoded = _canonical(record)
    columns = {"response": encoded}
    if reseal:
        columns["response_digest"] = jdb._digest(encoded)
    sets = ", ".join(f"{c} = ?" for c in columns)
    with jdb.connect_closing() as conn:
        conn.execute(
            f"UPDATE job_run_requests SET {sets} WHERE request_id = ?",
            (*columns.values(), request_id),
        )
        conn.commit()


def _write_receipt(request_id, receipt_id, data, *, reseal: bool) -> None:
    encoded = _canonical(data)
    with jdb.connect_closing() as conn:
        conn.execute(
            "UPDATE job_receipts SET data = ? WHERE id = ?", (encoded, receipt_id)
        )
        if reseal:
            conn.execute(
                "UPDATE job_run_requests SET receipt_digest = ? WHERE request_id = ?",
                (jdb._digest(encoded), request_id),
            )
        conn.commit()


def _settled_once(tmp_path, repo, request_id):
    """One real run, plus everything a replay must not be able to change."""
    path, base = repo
    _make_job()
    first = _run_once(tmp_path, path, base, request_id=request_id)
    assert first["ran"] is True
    row = _stored(request_id)
    with jdb.connect_closing() as conn:
        job = jdb.get_job(conn, row["job_id"])
        receipt = [r for r in jdb.get_receipts(conn, job.id)
                   if r["id"] == row["receipt_id"]][0]
        before = (
            _events(job.id), job.revision,
            len(jdb.get_attempts(conn, job.id)), len(jdb.get_receipts(conn, job.id)),
        )
    return first, row, receipt, before


def _assert_replay_fails_closed(tmp_path, repo, request_id, before, why):
    path, base = repo
    job_id = _stored(request_id)["job_id"]
    with jdb.connect_closing() as conn:
        with pytest.raises(jdb.ReplayIntegrity):
            jdb.find_run_response(conn, request_id)
    with pytest.raises(jdb.ReplayIntegrity):
        _run_once(tmp_path, path, base, request_id=request_id)
    with jdb.connect_closing() as conn:
        job = jdb.get_job(conn, job_id)
        now = (
            _events(job_id), job.revision,
            len(jdb.get_attempts(conn, job_id)), len(jdb.get_receipts(conn, job_id)),
        )
    assert now == before, why


# Every one of these is re-derivable from the attempt, the request, or the Job,
# so a stored response that disagrees with the store is refused on the strength
# of the store — not merely because its row ids happen to line up.
FORGED_RESPONSE_FIELDS = {
    "commit": "f" * 40,
    "branch": "forged-branch",
    "worktree": "/tmp/forged-worktree",
    "repository": "/tmp/forged-repo",
    "ordinal": 99,
    "status": "failed",
    "failure_class": "implementation",
    "attempt_id": "a_forged",
    "job_id": "j_forged",
    "receipt_id": "r_forged",
    "request_id": "some-other-request",
    "job_number": 4242,
}


def test_a_forged_stored_response_field_cannot_survive_replay(home, tmp_path, repo):
    _first, row, _receipt, before = _settled_once(tmp_path, repo, "forged-field")
    truth = json.loads(row["response"])

    for field, forged in sorted(FORGED_RESPONSE_FIELDS.items()):
        _write_response("forged-field", {**truth, field: forged}, reseal=True)
        _assert_replay_fails_closed(
            tmp_path, repo, "forged-field", before,
            f"forging {field} left a side effect",
        )
        # And restoring the truth restores the exact original answer.
        _write_response("forged-field", truth, reseal=True)
        with jdb.connect_closing() as conn:
            assert jdb.find_run_response(conn, "forged-field") == truth


CORRUPTED_RESPONSE_FIELDS = {
    "reason": "forged", "error": "forged", "routing_reason": "forged",
    "job_status": "blocked", "job_step": "forged", "job_label": "#9999 Forged",
    "ran": False, "recovered": ["j_forged"],
}


def test_single_row_response_corruption_is_detected(home, tmp_path, repo):
    """The narrative fields have no second source, so the seal is what guards them."""
    _first, row, _receipt, before = _settled_once(tmp_path, repo, "corrupt-response")
    truth = json.loads(row["response"])

    for field, corrupted in sorted(CORRUPTED_RESPONSE_FIELDS.items()):
        _write_response("corrupt-response", {**truth, field: corrupted}, reseal=False)
        _assert_replay_fails_closed(
            tmp_path, repo, "corrupt-response", before,
            f"corrupting {field} left a side effect",
        )
        _write_response("corrupt-response", truth, reseal=False)
        with jdb.connect_closing() as conn:
            assert jdb.find_run_response(conn, "corrupt-response") == truth


FORGED_RECEIPT_FIELDS = {
    "status": "failed",
    "failure_class": "implementation",
    "commit": "f" * 40,
    "branch": "forged-branch",
    "worktree": "/tmp/forged-worktree",
    "repository": "/tmp/forged-repo",
    "ordinal": 99,
    "attempt_id": "a_forged",
    "job_id": "j_forged",
}


def test_a_contradictory_final_receipt_cannot_survive_replay(home, tmp_path, repo):
    """The receipt is read, parsed, and held to the attempt — not just owned by it."""
    _first, row, receipt, before = _settled_once(tmp_path, repo, "forged-receipt")
    truth = receipt["data"]
    rid = row["receipt_id"]

    for field, forged in sorted(FORGED_RECEIPT_FIELDS.items()):
        _write_receipt("forged-receipt", rid, {**truth, field: forged}, reseal=True)
        _assert_replay_fails_closed(
            tmp_path, repo, "forged-receipt", before,
            f"forging receipt {field} left a side effect",
        )
        _write_receipt("forged-receipt", rid, truth, reseal=True)
        with jdb.connect_closing() as conn:
            assert jdb.find_run_response(conn, "forged-receipt") is not None


def test_single_row_receipt_corruption_is_detected(home, tmp_path, repo):
    _first, row, receipt, before = _settled_once(tmp_path, repo, "corrupt-receipt")
    truth = receipt["data"]
    rid = row["receipt_id"]

    for field, corrupted in (("duration_seconds", 999.0), ("evidence_problems", ["x"])):
        _write_receipt("corrupt-receipt", rid, {**truth, field: corrupted},
                       reseal=False)
        _assert_replay_fails_closed(
            tmp_path, repo, "corrupt-receipt", before,
            f"corrupting receipt {field} left a side effect",
        )
        _write_receipt("corrupt-receipt", rid, truth, reseal=False)
        with jdb.connect_closing() as conn:
            assert jdb.find_run_response(conn, "corrupt-receipt") is not None


def test_a_final_receipt_that_is_not_standard_json_fails_closed(home, tmp_path, repo):
    _first, row, _receipt, before = _settled_once(tmp_path, repo, "nan-receipt")
    with jdb.connect_closing() as conn:
        conn.execute(
            "UPDATE job_receipts SET data = ? WHERE id = ?",
            ('{"status": NaN}', row["receipt_id"]),
        )
        conn.commit()
    _assert_replay_fails_closed(
        tmp_path, repo, "nan-receipt", before, "a NaN receipt left a side effect",
    )


# ---------------------------------------------------------------------------
# Crash windows: ownership is truthful, and recovery is the repair path
# ---------------------------------------------------------------------------


def test_a_failure_to_reserve_leaves_the_store_untouched(home, tmp_path, repo):
    path, base = repo
    job = _make_job()

    def refuse(*a, **kw):
        raise sqlite3.OperationalError("reservation storage refused the write")

    # Its own context: undoing the shared fixture's monkeypatch would also undo
    # HERMES_HOME and point the rest of this test at the real Jobs database.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jdb, "reserve_request", refuse)
        with pytest.raises(sqlite3.OperationalError):
            _run_once(tmp_path, path, base, request_id="no-reservation")

    with jdb.connect_closing() as conn:
        assert jdb.get_attempts(conn, job.id) == []
        assert jdb.get_job(conn, job.id).claimed_by is None
    assert _requests() == []
    assert "claim_acquired" not in _events(job.id)
    # Repairable: the same request id is still free.
    assert _run_once(tmp_path, path, base, request_id="no-reservation")["ran"] is True


def test_a_failure_to_bind_the_request_starts_no_attempt(home, tmp_path, repo):
    path, base = repo
    job = _make_job()

    def refuse(*a, **kw):
        raise sqlite3.OperationalError("binding storage refused the write")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jdb, "_bind_request_locked", refuse)
        with pytest.raises(sqlite3.OperationalError):
            _run_once(tmp_path, path, base, request_id="no-binding")

    with jdb.connect_closing() as conn:
        assert jdb.get_attempts(conn, job.id) == []
        assert jdb.get_receipts(conn, job.id) == []
    assert "attempt_started" not in _events(job.id)
    assert [r["attempt_id"] for r in _requests()] == [None]

    # Repairable through the existing lease path: the abandoned claim expires,
    # recovery clears it, and the still-unbound request runs for real.
    again = _run_once(tmp_path, path, base, request_id="no-binding", now=T0 + 4000)
    assert again["ran"] is True
    assert _run_once(tmp_path, path, base, request_id="no-binding") == again


def test_a_live_in_progress_owner_is_never_stolen(home, tmp_path, repo):
    path, base = repo
    job = _make_job()
    _claimed_attempt(job, "in-flight", worker="live-runner")

    res = _run_once(tmp_path, path, base, request_id="in-flight", now=T0 + 10)

    assert res["ran"] is False
    assert res["reason"] == "request_in_progress"
    with jdb.connect_closing() as conn:
        attempts = jdb.get_attempts(conn, job.id)
    assert [a["status"] for a in attempts] == ["running"]


def test_a_crash_before_the_worker_launched_is_recovered_and_replays_exactly(
    home, tmp_path, repo
):
    path, base = repo
    job = _make_job()
    log = tmp_path / "launches.log"
    _claimed_attempt(job, "died-early", worker="dead-runner")

    first = _run_once(
        tmp_path, path, base, request_id="died-early", now=T0 + 4000,
        extra_env={"JOBS_TEST_LAUNCH_LOG": str(log)},
    )

    assert first["ran"] is True
    assert first["status"] == "interrupted"
    assert first["failure_class"] == "infrastructure"
    assert first["receipt_id"] is not None
    assert not log.exists()  # the dead runner never launched one, and neither did we
    with jdb.connect_closing() as conn:
        assert len(jdb.get_attempts(conn, job.id)) == 1
        assert len(jdb.get_receipts(conn, job.id)) == 1
    before = _events(job.id)

    second = _run_once(
        tmp_path, path, base, request_id="died-early", now=T0 + 4100,
        extra_env={"JOBS_TEST_LAUNCH_LOG": str(log)},
    )
    assert second == first
    assert _events(job.id) == before
    assert not log.exists()


def test_a_crash_after_the_worker_finished_is_recovered_and_replays_exactly(
    home, tmp_path, repo, monkeypatch
):
    path, base = repo
    job = _make_job()
    log = tmp_path / "launches.log"
    env = {"JOBS_TEST_LAUNCH_LOG": str(log)}
    real_settle = jdb.settle_attempt

    def die_before_persisting(*a, **kw):
        raise sqlite3.OperationalError("process died before the settlement committed")

    monkeypatch.setattr(jdb, "settle_attempt", die_before_persisting)
    with pytest.raises(sqlite3.OperationalError):
        _run_once(tmp_path, path, base, request_id="died-late", extra_env=env)
    monkeypatch.setattr(jdb, "settle_attempt", real_settle)

    launches = log.read_text().strip().splitlines()
    assert len(launches) == 1
    with jdb.connect_closing() as conn:
        assert [a["status"] for a in jdb.get_attempts(conn, job.id)] == ["running"]

    first = _run_once(
        tmp_path, path, base, request_id="died-late", now=T0 + 4000, extra_env=env,
    )
    assert first["ran"] is True
    assert first["status"] == "interrupted"
    assert log.read_text().strip().splitlines() == launches  # no second execution
    before = _events(job.id)

    second = _run_once(
        tmp_path, path, base, request_id="died-late", now=T0 + 4100, extra_env=env,
    )
    assert second == first
    assert _events(job.id) == before
    assert log.read_text().strip().splitlines() == launches


def test_a_legacy_settled_event_grants_no_replay_authority(home, tmp_path, repo):
    """An old database's ``run_settled`` events migrate as evidence, not authority."""
    path, base = repo
    job = _make_job()
    _forge_event(job.id, "legacy-req", _forged_record("legacy-req"))

    res = _run_once(tmp_path, path, base, request_id="legacy-req")

    assert res["attempt_id"] != "a_forged"
    # The event survives untouched, and the request is owned by the real run.
    assert LEGACY_KEY("legacy-req") in [
        e["idempotency_key"] for e in _all_events(job.id)
    ]
    assert [(r["request_id"], r["attempt_id"]) for r in _requests()] == [
        ("legacy-req", res["attempt_id"])
    ]
    assert _run_once(tmp_path, path, base, request_id="legacy-req") == res


def test_graph_gate_adapter_requires_authoritative_identity(tmp_path):
    result = jobs_run.adapt_gate_settlement(
        {
            "action_outcome": "succeeded",
            "identity_verified": False,
            "reason_code": "EVIDENCE_MISSING",
            "commit": "c" * 40,
        },
        job_dir=tmp_path,
        review_attempt=0,
    )

    assert result.action_outcome == "failed"
    assert result.identity_verified is False
    assert result.reason_code == "EVIDENCE_MISSING"
