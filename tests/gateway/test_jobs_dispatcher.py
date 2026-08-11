"""The gateway owns one bounded, opt-in Jobs dispatcher loop."""

from __future__ import annotations

from pathlib import Path

from hermes_cli import jobs_db as jdb
from hermes_cli import jobs_lanes
from gateway import jobs_dispatcher


def _health(now: int) -> list[jobs_lanes.LaneHealth]:
    return [
        jobs_lanes.LaneHealth(
            lane_id="codex-mac-1",
            state="IDLE",
            status="PASS",
            failure_class=None,
            reason_code="OK",
            observed_at=now,
            expires_at=now + 120,
            executor_version="0.146.0",
            available_capacity=1,
            safe_detail={},
        )
    ]


def test_dispatch_one_uses_persisted_header_and_production_dependencies(
    tmp_path
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = "a" * 40
    db_path = tmp_path / "jobs.db"
    conn = jdb.connect(db_path)
    job_id = jdb.create_job(
        conn,
        name="gateway dispatch",
        requested_lane="codex",
        goal=(
            f"REPO_PATH={repo}\nBASE_COMMIT={base}\nMODEL=gpt-5.6-sol\n"
            "EFFORT=high\nMAX_TURNS=120\nWORKSPACE_KIND=worktree\n"
            "NO_REROUTE=true\n\nchange one file"
        ),
    )
    conn.close()
    calls = []

    def dispatch(**kwargs):
        calls.append(kwargs)
        return {"claimed": False, "reason": "CAPACITY_FULL", "job_id": job_id}

    result = jobs_dispatcher.dispatch_due_job_once(
        jobs_path=db_path,
        lane_root=tmp_path / "lanes",
        now=100,
        dispatcher=dispatch,
        health_collector=lambda **kwargs: _health(100),
    )

    assert result["job_id"] == job_id
    assert len(calls) == 1
    call = calls[0]
    assert call["repo_path"] == repo
    assert call["base_commit"] == base
    assert call["branch"] == f"jobs/{job_id}"
    assert call["executor_registry"] is None
    assert call["gate"] is None
    assert call["activation_gate"] is None
    assert call["signer"] is None and call["verifier"] is None


def test_dispatcher_is_disabled_without_exact_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_JOBS_DISPATCH", raising=False)
    monkeypatch.setenv("HERMES_JOBS_LANE_ROOT", str(tmp_path / "lanes"))
    assert jobs_dispatcher.dispatch_configuration() is None
    monkeypatch.setenv("HERMES_JOBS_DISPATCH", "true")
    assert jobs_dispatcher.dispatch_configuration() is None
    monkeypatch.setenv("HERMES_JOBS_DISPATCH", "1")
    assert jobs_dispatcher.dispatch_configuration() == tmp_path / "lanes"


def test_invalid_legacy_job_does_not_block_later_valid_job(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    db_path = tmp_path / "jobs.db"
    conn = jdb.connect(db_path)
    jdb.create_job(
        conn, name="old invalid", requested_lane="codex", goal="no header"
    )
    valid = jdb.create_job(
        conn,
        name="valid",
        requested_lane="codex",
        goal=(
            f"REPO_PATH={repo}\nBASE_COMMIT={'b' * 40}\nMODEL=gpt-5.6-sol\n"
            "MAX_TURNS=120\n\nvalid body"
        ),
    )
    conn.close()
    seen = []
    jobs_dispatcher.dispatch_due_job_once(
        jobs_path=db_path,
        lane_root=tmp_path / "lanes",
        now=100,
        dispatcher=lambda **kwargs: seen.append(kwargs) or {"job_id": kwargs["job_id"]},
        health_collector=lambda **kwargs: _health(100),
    )
    assert [item["job_id"] for item in seen] == [valid]
