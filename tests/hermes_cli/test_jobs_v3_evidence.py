"""V3 replay evidence must be relational, mandatory, and impossible to forge.

Every test here attacks the same claim: a settled V3 request replays because the
store still *agrees with itself*, not because one row says so. The authoritative
facts live in typed columns on the attempt and the request; the response envelope
and the final receipt are two redundant renderings of them, and any single row
rewritten — payload and digest together — contradicts the other two.

The child-process isolation tests belong here for the same reason: HOME and
HERMES_HOME are the two paths that decide which Jobs database a worker resolves,
so leaving either inherited would make every guarantee above conditional on the
operator's environment.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from hermes_cli import jobs_adapter_claude as adapter
from hermes_cli import jobs_db as jdb

from tests.hermes_cli.test_jobs_run_once import (  # noqa: F401 - pytest fixtures
    COMMIT_AND_SUCCEED,
    T0,
    _execution,
    _make_job,
    _run_once,
    _worker,
    home,
    repo,
)


# ---------------------------------------------------------------------------
# Helpers: the raw store, because forging is the point
# ---------------------------------------------------------------------------


def _sql(statement, params=()):
    with sqlite3.connect(jdb.jobs_db_path()) as raw:
        raw.row_factory = sqlite3.Row
        cur = raw.execute(statement, params)
        rows = cur.fetchall()
    return rows


def _request(request_id="req-1"):
    return _sql(
        "SELECT * FROM job_run_requests WHERE request_id = ?", (request_id,)
    )[0]


def _receipt_row(receipt_id):
    return _sql("SELECT * FROM job_receipts WHERE id = ?", (receipt_id,))[0]


def _digest(text: str) -> str:
    return jdb._digest(text)


def _canonical(data: dict) -> str:
    return jdb._canonical(data)


def _rewrite_response(request_id, mutate):
    """Rewrite a stored response *and* re-seal it, the way real damage would."""
    row = _request(request_id)
    payload = json.loads(row["response"])
    mutate(payload)
    encoded = _canonical(payload)
    _sql(
        "UPDATE job_run_requests SET response = ?, response_digest = ? "
        "WHERE request_id = ?",
        (encoded, _digest(encoded), request_id),
    )


def _rewrite_receipt(request_id, mutate):
    """Rewrite a final receipt *and* the request row's seal for it."""
    row = _request(request_id)
    receipt = _receipt_row(row["receipt_id"])
    payload = json.loads(receipt["data"])
    mutate(payload)
    encoded = _canonical(payload)
    _sql("UPDATE job_receipts SET data = ? WHERE id = ?", (encoded, receipt["id"]))
    _sql(
        "UPDATE job_run_requests SET receipt_digest = ? WHERE request_id = ?",
        (_digest(encoded), request_id),
    )


def _settled(tmp_path, repo_pair, request_id="req-1"):
    path, base = repo_pair
    job = _make_job()
    first = _run_once(tmp_path, path, base, request_id=request_id)
    assert first["ran"] is True and first["status"] == "succeeded"
    return job, first


def _replay(tmp_path, repo_pair, request_id="req-1"):
    path, base = repo_pair
    return _run_once(tmp_path, path, base, request_id=request_id, now=T0 + 5)


def _side_effects(job_id):
    with jdb.connect_closing() as conn:
        return (
            [e["kind"] for e in jdb.get_events(conn, job_id)],
            len(jdb.get_receipts(conn, job_id)),
            jdb.get_job(conn, job_id).revision,
            [a["status"] for a in jdb.get_attempts(conn, job_id)],
        )


# ---------------------------------------------------------------------------
# The approved base commit is attempt evidence, not a runtime argument
# ---------------------------------------------------------------------------


def test_the_approved_base_commit_is_persisted_and_returned(home, tmp_path, repo):
    path, base = repo
    job, res = _settled(tmp_path, repo)

    assert res["base_commit"] == base
    assert res["repository"] == str(path)

    with jdb.connect_closing() as conn:
        attempt = jdb.get_attempt(conn, res["attempt_id"])
    assert attempt["base_commit"] == base

    receipt = json.loads(_receipt_row(_request()["receipt_id"])["data"])
    assert receipt["base_commit"] == base
    assert receipt["schema_version"] == jdb.V3_RECEIPT_SCHEMA_VERSION
    assert receipt["evidence"]["base_commit"] == base

    assert _replay(tmp_path, repo) == res


def test_a_settled_base_commit_cannot_be_rewritten_by_a_supported_api(
    home, tmp_path, repo
):
    path, base = repo
    job, res = _settled(tmp_path, repo)
    before = _side_effects(job.id)

    with jdb.connect_closing() as conn:
        with pytest.raises((jdb.InvalidTransition, ValueError)):
            jdb.settle_attempt(
                conn,
                res["attempt_id"],
                status="succeeded",
                receipt={"status": "succeeded", "failure_class": None},
                base_commit="0" * 40,
            )
    assert _side_effects(job.id) == before


def test_a_legacy_attempt_without_a_base_commit_grants_no_replay_authority(
    home, tmp_path, repo
):
    job, res = _settled(tmp_path, repo)
    _sql(
        "UPDATE job_attempts SET base_commit = NULL WHERE id = ?",
        (res["attempt_id"],),
    )
    before = _side_effects(job.id)

    with jdb.connect_closing() as conn:
        with pytest.raises(jdb.ReplayIntegrity):
            jdb.find_run_response(conn, "req-1")
    assert _side_effects(job.id) == before


def test_the_migration_invents_no_base_commit_for_legacy_attempts(home, tmp_path):
    """Re-opening a database that predates the column leaves it NULL."""
    with jdb.connect_closing() as conn:
        jid = jdb.create_job(conn, name="Legacy", goal="g")
        claim = jdb.claim_job(conn, worker="w", lease_seconds=60, now=T0)
        aid = jdb.start_attempt(conn, jid, claim_token=claim.claim_token, now=T0)
    _sql("UPDATE job_attempts SET base_commit = NULL WHERE id = ?", (aid,))

    jdb._INITIALIZED_PATHS.discard(str(jdb.jobs_db_path()))
    with jdb.connect_closing() as conn:
        assert jdb.get_attempt(conn, aid)["base_commit"] is None


# ---------------------------------------------------------------------------
# Every required replay field is mandatory, not best-effort
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "repository", "base_commit", "route", "reason", "error", "branch",
        "worktree", "commit", "ordinal", "status", "failure_class",
        "job_id", "attempt_id", "receipt_id", "request_id",
    ],
)
def test_a_missing_response_field_fails_closed(home, tmp_path, repo, field):
    job, res = _settled(tmp_path, repo)
    before = _side_effects(job.id)

    _rewrite_response("req-1", lambda payload: payload.pop(field, None))

    with jdb.connect_closing() as conn:
        with pytest.raises(jdb.ReplayIntegrity):
            jdb.find_run_response(conn, "req-1")
    assert _side_effects(job.id) == before


@pytest.mark.parametrize(
    "field,forged",
    [
        ("repository", "/tmp/elsewhere"),
        ("base_commit", "0" * 40),
        ("branch", "jobs/forged"),
        ("worktree", "/tmp/forged"),
        ("commit", "1" * 40),
        ("ordinal", 99),
        ("status", "failed"),
        ("failure_class", "implementation"),
        ("route", "forged route"),
        ("reason", "forged"),
        ("error", "forged error"),
    ],
)
def test_a_contradicted_response_field_fails_closed(
    home, tmp_path, repo, field, forged
):
    """The stored row and its seal rewritten together still lose to the store."""
    job, res = _settled(tmp_path, repo)
    before = _side_effects(job.id)

    _rewrite_response("req-1", lambda payload: payload.__setitem__(field, forged))

    with jdb.connect_closing() as conn:
        with pytest.raises(jdb.ReplayIntegrity):
            jdb.find_run_response(conn, "req-1")
    assert _side_effects(job.id) == before


# ---------------------------------------------------------------------------
# The final receipt shape is exact and versioned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(jdb.V3_RECEIPT_REQUIRED_KEYS))
def test_a_missing_final_receipt_key_fails_closed(home, tmp_path, repo, key):
    job, res = _settled(tmp_path, repo)
    before = _side_effects(job.id)

    _rewrite_receipt("req-1", lambda payload: payload.pop(key, None))

    with jdb.connect_closing() as conn:
        with pytest.raises(jdb.ReplayIntegrity):
            jdb.find_run_response(conn, "req-1")
    assert _side_effects(job.id) == before


@pytest.mark.parametrize(
    "key,forged",
    [
        ("schema_version", 1),
        ("kind", "jobs-claude-attempt"),
        ("repository", "/tmp/elsewhere"),
        ("base_commit", "0" * 40),
        ("branch", "jobs/forged"),
        ("worktree", "/tmp/forged"),
        ("commit", "1" * 40),
        ("ordinal", 99),
        ("status", "failed"),
        ("failure_class", "implementation"),
        ("route", "forged route"),
        ("reason", "forged"),
        ("error", "forged error"),
        ("job_id", "j_forged"),
        ("attempt_id", "a_forged"),
        ("request_id", "forged-request"),
        ("evidence", {"repository": "/tmp/elsewhere"}),
    ],
)
def test_a_contradicted_final_receipt_key_fails_closed(
    home, tmp_path, repo, key, forged
):
    """Receipt payload and its seal rewritten together, attempt/request intact."""
    job, res = _settled(tmp_path, repo)
    before = _side_effects(job.id)

    _rewrite_receipt("req-1", lambda payload: payload.__setitem__(key, forged))

    with jdb.connect_closing() as conn:
        with pytest.raises(jdb.ReplayIntegrity):
            jdb.find_run_response(conn, "req-1")
    assert _side_effects(job.id) == before


def test_a_legacy_non_v3_receipt_is_still_accepted_as_ordinary_evidence(
    home, tmp_path, repo
):
    """Compatibility is kept for the ledger; it is authority that is refused."""
    job, res = _settled(tmp_path, repo)
    with jdb.connect_closing() as conn:
        rid = jdb.add_receipt(
            conn, job.id, data={"schema_version": 1, "kind": "note", "any": "shape"},
        )
        assert rid
        assert len(jdb.get_receipts(conn, job.id)) == 2
        # ...and the V3 replay is unaffected by the extra evidence beside it.
        assert jdb.find_run_response(conn, "req-1")["attempt_id"] == res["attempt_id"]


# ---------------------------------------------------------------------------
# Narrative outcome evidence is typed and relational, not a blob plus a digest
# ---------------------------------------------------------------------------


def test_route_reason_and_error_live_in_typed_request_columns(home, tmp_path, repo):
    job, res = _settled(tmp_path, repo)
    row = _request()

    assert row["reason"] == "completed"
    assert row["route"] == res["routing_reason"] and row["route"]
    assert row["error"] is None
    assert res["reason"] == "completed"


def test_a_rewritten_response_blob_cannot_outvote_the_typed_columns(
    home, tmp_path, repo
):
    """Response row + digest rewritten; attempt and receipt untouched."""
    job, res = _settled(tmp_path, repo)
    before = _side_effects(job.id)

    _rewrite_response("req-1", lambda payload: payload.update(
        {"reason": "invented", "error": "invented", "route": "invented"}
    ))

    with jdb.connect_closing() as conn:
        with pytest.raises(jdb.ReplayIntegrity):
            jdb.find_run_response(conn, "req-1")
    assert _side_effects(job.id) == before


def test_a_settled_response_cannot_be_re_stored(home, tmp_path, repo):
    """No supported API rewrites settled authoritative request fields."""
    job, res = _settled(tmp_path, repo)
    row = _request()
    with jdb.connect_closing() as conn:
        with jdb.write_txn(conn):
            with pytest.raises(jdb.ReplayIntegrity):
                jdb._store_response_locked(
                    conn, "req-1", owner=row["owner"], job_id=row["job_id"],
                    attempt_id=row["attempt_id"], receipt_id=row["receipt_id"],
                    record={"reason": "second", "error": None}, now=T0 + 9,
                )
    assert _request()["reason"] == "completed"


def test_bounds_are_enforced_on_the_typed_narrative_columns(home, tmp_path, repo):
    job, res = _settled(tmp_path, repo)
    with pytest.raises(sqlite3.IntegrityError):
        _sql(
            "UPDATE job_run_requests SET error = ? WHERE request_id = ?",
            ("x" * 5000, "req-1"),
        )


# ---------------------------------------------------------------------------
# Child-process isolation: HOME and HERMES_HOME are constructed, never inherited
# ---------------------------------------------------------------------------


REPORT_ENV = """
    json.dump({"outcome": "failed", "failure_class": "implementation",
               "env": {k: os.environ.get(k) for k in
                       ("HOME", "HERMES_HOME", "PATH", "TMPDIR")}},
              open(os.environ["JOBS_TEST_ENV_REPORT"], "w"))
    json.dump({"outcome": "failed", "failure_class": "implementation"},
              open(a.result_file, "w"))
    sys.exit(1)
"""


def test_the_worker_gets_private_contained_home_and_hermes_home(
    home, tmp_path, repo, monkeypatch
):
    path, base = repo
    _make_job()
    report = tmp_path / "env.json"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))

    res = _run_once(
        tmp_path, path, base,
        worker=_worker(tmp_path, REPORT_ENV, name="envworker.py"),
        extra_env={"JOBS_TEST_ENV_REPORT": str(report)},
    )
    assert res["ran"] is True

    seen = json.loads(report.read_text())["env"]
    private = (tmp_path / "workspaces").resolve()
    for key in ("HOME", "HERMES_HOME", "TMPDIR"):
        assert seen[key], f"{key} was not passed to the worker"
        resolved = type(tmp_path)(seen[key]).resolve()
        assert private in resolved.parents or resolved == private
    assert seen["HOME"] != str(home) and seen["HERMES_HOME"] != str(home)
    assert seen["PATH"]


def test_an_uncontained_private_root_fails_closed_before_the_worker_launches(
    home, tmp_path, repo, monkeypatch
):
    path, base = repo
    log = tmp_path / "launches.log"
    escape = tmp_path / "outside"
    escape.mkdir()
    monkeypatch.setattr(
        adapter, "_private_env_root", lambda handoff: (escape, escape, escape)
    )
    _make_job()

    res = _run_once(
        tmp_path, path, base, extra_env={"JOBS_TEST_LAUNCH_LOG": str(log)},
    )
    assert res["ran"] is True
    assert res["status"] == "failed" and res["failure_class"] == "infrastructure"
    assert not log.exists()


def test_extra_env_cannot_un_isolate_the_worker(home, tmp_path, repo):
    path, base = repo
    log = tmp_path / "launches.log"
    _make_job()

    res = _run_once(
        tmp_path, path, base,
        extra_env={"HERMES_HOME": str(home), "JOBS_TEST_LAUNCH_LOG": str(log)},
    )
    assert res["ran"] is True
    assert res["status"] == "failed" and res["failure_class"] == "infrastructure"
    assert not log.exists()


def test_the_operator_environment_is_not_inherited(home, tmp_path):
    assert "HOME" not in adapter._ENV_ALLOWLIST
    assert "HERMES_HOME" not in adapter._ENV_ALLOWLIST
    assert "PATH" in adapter._ENV_ALLOWLIST
