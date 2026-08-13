"""Tests for the Jobs V3 public CLI verbs (receipts, release, attempts, status).

These are the capabilities adapters need and V2 left unreachable: a worker could
not store a receipt, list attempts, read a stale projection, or hand a Job back.
Every test drives the real argparse tree against a real temporary ``jobs.db``.
"""

from __future__ import annotations

import argparse
import json
import os

import pytest

from hermes_cli import jobs as jobs_cli
from hermes_cli import jobs_db as jdb


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes_home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


# capsys buffers until something reads it, so a setup command's output would
# otherwise be prepended to the JSON the next assertion parses. Draining on
# every call keeps each _run's output its own.
_CAPSYS = {}


@pytest.fixture(autouse=True)
def _drain(capsys):
    _CAPSYS["fixture"] = capsys
    yield
    _CAPSYS.pop("fixture", None)


def _run(argv, capsys=None):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    jobs_cli.build_parser(sub)
    args = parser.parse_args(["jobs", *argv])
    rc = jobs_cli.jobs_command(args)
    cap = _CAPSYS["fixture"].readouterr()
    return rc, cap.out, cap.err


def _job(home, name="Ship it", goal="do the thing", lane="claude"):
    gf = home / "goal.txt"
    gf.write_text(goal)
    assurance = home / "assurance.json"
    assurance.write_text(json.dumps({
        "critical_user_journey": "operator observes the requested result",
        "success_metric": "critical journey passes",
        "outcome_mode": "integration",
        "verification_steps": ["exercise critical journey"],
        "max_attempts": 3,
        "wall_clock_budget_seconds": 3600,
        "risk_domains": ["none"],
        "consumers": [],
        "egress_paths": [],
        "rollback_behavior": "not applicable",
        "knowledge_closure_required": False,
    }))
    argv = [
        "create",
        name,
        "--goal-file",
        str(gf),
        "--assurance-file",
        str(assurance),
        "--lane",
        lane,
        "--json",
    ]
    _run(argv)
    with jdb.connect_closing() as conn:
        return jdb.list_jobs(conn)[-1]


def _claim(home, job, token_path=None, lease=600):
    token_path = token_path or (home / "token")
    rc, _, _ = _run(
        ["claim", "--worker", "w1", "--job", str(job.number),
         "--specialist", job.specialist,
         "--lease-seconds", str(lease), "--token-out", str(token_path)]
    )
    assert rc == 0
    return token_path


def _start(home, job, token_path):
    rc, out, _ = _run(
        ["attempt-start", str(job.number), "--claim-token-file", str(token_path),
         "--json"]
    )
    assert rc == 0
    with jdb.connect_closing() as conn:
        return jdb.get_attempts(conn, job.id)[-1]["id"]


# ---------------------------------------------------------------------------
# receipt-add / receipts
# ---------------------------------------------------------------------------


def test_receipt_add_stores_a_receipt_against_an_attempt(home, capsys):
    job = _job(home)
    token = _claim(home, job)
    attempt = _start(home, job, token)
    df = home / "receipt.json"
    df.write_text(json.dumps({"outcome": "succeeded", "commit": "a" * 40}))

    rc, out, _ = _run(
        ["receipt-add", str(job.number), "--data-file", str(df),
         "--attempt", attempt, "--idempotency-key", f"{attempt}:progress",
         "--json"],
        capsys,
    )
    assert rc == 0
    payload = json.loads(out)
    assert payload["receipt"]["attempt_id"] == attempt
    assert payload["receipt"]["data"]["commit"] == "a" * 40


def test_receipts_lists_what_was_stored(home, capsys):
    job = _job(home)
    df = home / "receipt.json"
    df.write_text(json.dumps({"note": "first"}))
    _run(["receipt-add", str(job.number), "--data-file", str(df)])
    df.write_text(json.dumps({"note": "second"}))
    _run(["receipt-add", str(job.number), "--data-file", str(df)])

    rc, out, _ = _run(["receipts", str(job.number), "--json"], capsys)
    assert rc == 0
    notes = [r["data"]["note"] for r in json.loads(out)]
    assert notes == ["first", "second"]


def test_receipt_add_is_idempotent_on_a_repeated_key(home, capsys):
    job = _job(home)
    df = home / "receipt.json"
    df.write_text(json.dumps({"note": "only"}))
    rc, out, _ = _run(
        ["receipt-add", str(job.number), "--data-file", str(df),
         "--idempotency-key", "k1", "--json"], capsys,
    )
    first = json.loads(out)["receipt"]["id"]
    rc2, out2, _ = _run(
        ["receipt-add", str(job.number), "--data-file", str(df),
         "--idempotency-key", "k1", "--json"], capsys,
    )
    assert (rc, rc2) == (0, 0)
    assert json.loads(out2)["receipt"]["id"] == first
    with jdb.connect_closing() as conn:
        assert len(jdb.get_receipts(conn, job.id)) == 1


def test_receipt_add_rejects_a_secret_shaped_key(home, capsys):
    job = _job(home)
    df = home / "receipt.json"
    df.write_text(json.dumps({"env": {"ANTHROPIC_API_KEY": "sk-ant-nope"}}))
    rc, _, err = _run(
        ["receipt-add", str(job.number), "--data-file", str(df)], capsys
    )
    assert rc == 2
    assert "secret" in err.lower()
    with jdb.connect_closing() as conn:
        assert jdb.get_receipts(conn, job.id) == []


def test_receipt_add_redacts_secret_shaped_text(home, capsys):
    job = _job(home)
    df = home / "receipt.json"
    df.write_text(json.dumps({"tail": "leaked ghp_" + "D" * 36}))
    rc, out, _ = _run(
        ["receipt-add", str(job.number), "--data-file", str(df), "--json"], capsys
    )
    assert rc == 0
    assert "ghp_D" not in out
    with jdb.connect_closing() as conn:
        stored = jdb.get_receipts(conn, job.id)[0]["data"]["tail"]
    assert "ghp_D" not in stored


def test_receipt_add_rejects_a_non_object_payload(home, capsys):
    job = _job(home)
    df = home / "receipt.json"
    df.write_text(json.dumps(["not", "an", "object"]))
    rc, _, err = _run(
        ["receipt-add", str(job.number), "--data-file", str(df)], capsys
    )
    assert rc == 2
    assert err.strip()


def test_receipt_add_rejects_invalid_json(home, capsys):
    job = _job(home)
    df = home / "receipt.json"
    df.write_text("{not json")
    rc, _, err = _run(
        ["receipt-add", str(job.number), "--data-file", str(df)], capsys
    )
    assert rc == 2
    assert err.strip()


def test_receipt_add_rejects_an_attempt_from_another_job(home, capsys):
    first = _job(home, name="first")
    second = _job(home, name="second")
    token = _claim(home, first)
    attempt = _start(home, first, token)
    df = home / "receipt.json"
    df.write_text(json.dumps({"note": "x"}))

    rc, _, err = _run(
        ["receipt-add", str(second.number), "--data-file", str(df),
         "--attempt", attempt], capsys,
    )
    assert rc == 2
    assert err.strip()


def test_receipts_on_an_unknown_job_reports_not_found(home, capsys):
    rc, _, err = _run(["receipts", "404", "--json"], capsys)
    assert rc == 1
    assert "no such job" in err


# ---------------------------------------------------------------------------
# attempts
# ---------------------------------------------------------------------------


def test_attempts_lists_in_execution_order(home, capsys):
    job = _job(home)
    token = _claim(home, job)
    first = _start(home, job, token)
    _run(["attempt-finish", first, "--claim-token-file", str(token),
          "--status", "failed", "--failure-class", "implementation"])
    token2 = _claim(home, job, token_path=home / "token2")
    second = _start(home, job, token2)

    rc, out, _ = _run(["attempts", str(job.number), "--json"], capsys)
    assert rc == 0
    rows = json.loads(out)
    assert [r["id"] for r in rows] == [first, second]
    assert [r["ordinal"] for r in rows] == [1, 2]


def test_attempts_never_exposes_a_claim_token(home, capsys):
    job = _job(home)
    token = _claim(home, job)
    _start(home, job, token)
    secret = (home / "token").read_text().strip()

    rc, out, _ = _run(["attempts", str(job.number), "--json"], capsys)
    assert rc == 0
    assert secret not in out


def test_attempts_on_an_unknown_job_reports_not_found(home, capsys):
    rc, _, err = _run(["attempts", "404"], capsys)
    assert rc == 1
    assert "no such job" in err


# ---------------------------------------------------------------------------
# status (stale projection)
# ---------------------------------------------------------------------------


def test_status_reports_a_fresh_working_job_as_not_stale(home, capsys):
    job = _job(home)
    _run(["heartbeat", str(job.number), "--at", "1000"])
    rc, out, _ = _run(
        ["status", str(job.number), "--at", "1010", "--stale-threshold", "60",
         "--json"], capsys,
    )
    assert rc == 0
    assert json.loads(out)["stale"] is False


def test_status_reports_an_aged_out_heartbeat_as_stale(home, capsys):
    job = _job(home)
    _run(["heartbeat", str(job.number), "--at", "1000"])
    rc, out, _ = _run(
        ["status", str(job.number), "--at", "5000", "--stale-threshold", "60",
         "--json"], capsys,
    )
    assert rc == 0
    assert json.loads(out)["stale"] is True


def test_status_without_a_threshold_never_claims_stale(home, capsys):
    job = _job(home)
    _run(["heartbeat", str(job.number), "--at", "1000"])
    rc, out, _ = _run(["status", str(job.number), "--at", "999999", "--json"], capsys)
    assert rc == 0
    assert json.loads(out)["stale"] is False


def test_status_never_exposes_a_claim_token(home, capsys):
    job = _job(home)
    _claim(home, job)
    secret = (home / "token").read_text().strip()
    rc, out, _ = _run(["status", str(job.number), "--json"], capsys)
    assert rc == 0
    assert secret not in out
    assert "claim_token" not in out


def test_status_on_an_unknown_job_reports_not_found(home, capsys):
    rc, _, err = _run(["status", "404"], capsys)
    assert rc == 1
    assert "no such job" in err


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


def test_release_hands_a_claimed_job_back_and_makes_it_claimable(home, capsys):
    job = _job(home)
    token = _claim(home, job)

    rc, out, _ = _run(
        ["release", str(job.number), "--claim-token-file", str(token), "--json"],
        capsys,
    )
    assert rc == 0
    with jdb.connect_closing() as conn:
        after = jdb.get_job(conn, job.id)
        assert after.status == "working"
        assert after.claimed_by is None
        assert after.lease_expires_at is None
        kinds = [e["kind"] for e in jdb.get_events(conn, job.id)]
    assert "claim_released" in kinds

    # Claimable again, by a different worker.
    rc2, _, _ = _run(
        ["claim", "--worker", "w2", "--job", str(job.number),
         "--lease-seconds", "600", "--token-out", str(home / "token-b")]
    )
    assert rc2 == 0


def test_release_cancels_a_running_attempt(home, capsys):
    job = _job(home)
    token = _claim(home, job)
    attempt = _start(home, job, token)

    rc, _, _ = _run(
        ["release", str(job.number), "--claim-token-file", str(token)], capsys
    )
    assert rc == 0
    with jdb.connect_closing() as conn:
        assert jdb.get_attempt(conn, attempt)["status"] == "cancelled"


def test_release_fails_closed_on_a_wrong_token(home, capsys):
    job = _job(home)
    _claim(home, job)
    wrong = home / "wrong-token"
    wrong.write_text("not-the-token")
    os.chmod(wrong, 0o600)

    rc, _, err = _run(
        ["release", str(job.number), "--claim-token-file", str(wrong)], capsys
    )
    assert rc == 2
    assert "claim token" in err.lower()
    with jdb.connect_closing() as conn:
        assert jdb.get_job(conn, job.id).claimed_by == "w1"


def test_release_fails_closed_on_an_expired_lease(home, capsys):
    job = _job(home)
    token = _claim(home, job, lease=60)
    with jdb.connect_closing() as conn:
        expires = jdb.get_job(conn, job.id).lease_expires_at

    rc, _, err = _run(
        ["release", str(job.number), "--claim-token-file", str(token),
         "--at", str(expires + 1)], capsys,
    )
    assert rc == 2
    assert "expired" in err.lower()


def test_release_refuses_a_group_readable_token_file(home, capsys):
    job = _job(home)
    token = _claim(home, job)
    os.chmod(token, 0o644)
    rc, _, err = _run(
        ["release", str(job.number), "--claim-token-file", str(token)], capsys
    )
    assert rc == 2
    assert "accessible" in err.lower()


def test_release_does_not_accept_a_token_on_the_command_line(home):
    job = _job(home)
    _claim(home, job)
    with pytest.raises(SystemExit):
        _run(["release", str(job.number), "--claim-token", "whatever"])


def test_release_on_an_unknown_job_reports_not_found(home, capsys):
    token = home / "t"
    token.write_text("x")
    os.chmod(token, 0o600)
    rc, _, err = _run(
        ["release", "404", "--claim-token-file", str(token)], capsys
    )
    assert rc == 1
    assert "no such job" in err


# ---------------------------------------------------------------------------
# The new verbs are wired into the same parser as the accepted ones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verb", ["receipt-add", "receipts", "release", "attempts", "status", "run-once"]
)
def test_every_new_verb_is_reachable_through_the_public_parser(verb):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    jobs_cli.build_parser(sub)
    with pytest.raises(SystemExit):
        parser.parse_args(["jobs", verb, "--help"])


def test_the_accepted_verbs_still_dispatch(home, capsys):
    job = _job(home)
    for argv in (["list", "--json"], ["show", str(job.number), "--json"],
                 ["events", str(job.number), "--json"]):
        rc, _, _ = _run(argv, capsys)
        assert rc == 0


# ---------------------------------------------------------------------------
# The final-receipt namespace is not a public namespace
# ---------------------------------------------------------------------------


def test_receipt_add_refuses_the_final_receipt_namespace(home, capsys):
    job = _job(home)
    token = _claim(home, job)
    aid = _start(home, job, token)
    df = home / "receipt.json"
    df.write_text(json.dumps({"status": "succeeded"}))

    rc, _, err = _run([
        "receipt-add", str(job.number), "--data-file", str(df),
        "--attempt", aid, "--idempotency-key", f"{aid}:final",
    ])

    assert rc == 2
    assert "final" in err
    with jdb.connect_closing() as conn:
        assert jdb.get_receipts(conn, job.id) == []


def test_receipt_add_still_accepts_an_ordinary_key(home, capsys):
    job = _job(home)
    df = home / "receipt.json"
    df.write_text(json.dumps({"note": "ordinary"}))
    rc, _, _ = _run([
        "receipt-add", str(job.number), "--data-file", str(df),
        "--idempotency-key", "run-1",
    ])
    assert rc == 0


# ---------------------------------------------------------------------------
# Public JSON in and out is standard JSON
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    ['{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}',
     '{"nested": {"deep": [NaN]}}'],
)
def test_receipt_add_refuses_non_standard_json(home, payload):
    job = _job(home)
    df = home / "receipt.json"
    df.write_text(payload)
    rc, _, err = _run([
        "receipt-add", str(job.number), "--data-file", str(df),
    ])
    assert rc == 2
    assert err.strip()
    with jdb.connect_closing() as conn:
        assert jdb.get_receipts(conn, job.id) == []
