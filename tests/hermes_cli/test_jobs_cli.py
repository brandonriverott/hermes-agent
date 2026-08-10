"""Integration tests for the ``hermes jobs`` CLI (hermes_cli/jobs).

These drive the real argparse tree and dispatch against a real ``jobs.db``
under a temporary ``HERMES_HOME`` — no mocks. They pin the byte-verbatim goal
contract, id/number/``Job #N`` resolution, JSON determinism + no env leakage,
and that the command is wired into the top-level CLI without disturbing the
existing Kanban registration.
"""

from __future__ import annotations

import argparse
import json

import pytest

from hermes_cli import jobs as jobs_cli
from hermes_cli import jobs_db as jdb


GOAL_BYTES = (
    "Ship it! “curly quotes”, café, \U0001f680.\n"
    "\n"
    "Second paragraph.\r\n"
    "Tabbed:\tend.\n"
).encode("utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fresh, isolated temporary HERMES_HOME for each test."""
    h = tmp_path / "hermes_home"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _run(argv, capsys=None):
    """Build the jobs subparser, parse argv, dispatch. Returns (rc, out, err).

    ``capsys.readouterr()`` clears the capture buffers, so this helper reads
    stdout AND stderr in one shot and hands both back — a later
    ``capsys.readouterr()`` in the test would otherwise see nothing.
    """
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    jobs_cli.build_parser(sub)
    args = parser.parse_args(["jobs", *argv])
    rc = jobs_cli.jobs_command(args)
    if capsys is None:
        return rc, "", ""
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _goal_file(home, data: bytes = GOAL_BYTES):
    p = home / "goal.txt"
    p.write_bytes(data)
    return p


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_stores_goal_file_verbatim(home, capsys):
    gf = _goal_file(home)
    rc, out, _ = _run(
        ["create", "Ship the thing", "--goal-file", str(gf),
         "--lane", "claude", "--routing-reason", "repo scope"],
        capsys,
    )
    assert rc == 0
    assert "Job #1" in out

    with jdb.connect_closing() as conn:
        job = jdb.get_job(conn, 1)
        assert job is not None
        # Byte-for-byte: CRLF, tabs, and Unicode all survive intact.
        assert job.goal.encode("utf-8") == GOAL_BYTES
        assert job.name == "Ship the thing"
        assert job.requested_lane == "claude"
        assert job.executor == "claude"
        assert job.specialist == "claude-builder"
        assert job.model == "claude-opus-5"
        assert job.routing_reason == "repo scope"
        assert job.status == "working" and job.step == "routing"


def test_create_requires_goal_file_not_positional(home):
    """A long goal cannot be smuggled in as a shell argument in V1."""
    # Missing --goal-file is an argparse error.
    with pytest.raises(SystemExit):
        _run(["create", "Just a name", "--lane", "claude"])
    # An extra positional (the "goal") is rejected — only <name> is positional.
    with pytest.raises(SystemExit):
        _run(["create", "name", "the whole long goal text here", "--lane", "claude"])


def test_create_requires_a_supported_explicit_lane(home):
    goal = str(_goal_file(home, b"g"))
    with pytest.raises(SystemExit):
        _run(["create", "Missing lane", "--goal-file", goal])
    with pytest.raises(SystemExit):
        _run(["create", "Retired KAT", "--goal-file", goal, "--lane", "kat"])
    with pytest.raises(SystemExit):
        _run(
            [
                "create",
                "Unrestricted specialist",
                "--goal-file",
                goal,
                "--specialist",
                "kat-builder",
            ]
        )


def test_create_missing_goal_file_reports_error(home, capsys):
    rc, _, err = _run(
        ["create", "X", "--goal-file", str(home / "nope.txt"), "--lane", "claude"],
        capsys,
    )
    assert rc == 2
    assert "goal file not found" in err


def test_create_records_correlations(home, capsys):
    gf = _goal_file(home, b"port this")
    rc, _, _ = _run(
        ["create", "Backlog", "--goal-file", str(gf),
         "--lane", "codex",
         "--correlation", "card-1", "--correlation", "card-2"],
        capsys,
    )
    assert rc == 0
    with jdb.connect_closing() as conn:
        assert sorted(jdb.get_job(conn, 1).correlations) == ["card-1", "card-2"]


# ---------------------------------------------------------------------------
# list / show resolution
# ---------------------------------------------------------------------------


def test_list_and_status_filter(home, capsys):
    _run(["create", "A", "--goal-file", str(_goal_file(home, b"a")),
          "--lane", "claude"], capsys)
    _run(["create", "B", "--goal-file", str(_goal_file(home, b"b")),
          "--lane", "claude"], capsys)
    with jdb.connect_closing() as conn:
        jdb.transition(conn, 2, status="needs_you", step="waiting_for_decision")

    rc, out, _ = _run(["list"], capsys)
    assert rc == 0
    assert "Job #1" in out and "Job #2" in out

    rc, out, _ = _run(["list", "--status", "needs_you"], capsys)
    assert rc == 0
    assert "Job #2" in out and "Job #1" not in out


def test_show_resolves_id_number_and_label(home, capsys):
    _run(["create", "Findable", "--goal-file", str(_goal_file(home, b"g")),
          "--lane", "claude"], capsys)
    with jdb.connect_closing() as conn:
        jid = jdb.get_job(conn, 1).id

    for ident in (jid, "1", "#1", "Job #1"):
        rc, out, _ = _run(["show", ident], capsys)
        assert rc == 0, ident
        assert "Findable" in out

    rc, _, err = _run(["show", "999"], capsys)
    assert rc == 1
    assert "no such job" in err


# ---------------------------------------------------------------------------
# transition / heartbeat / events
# ---------------------------------------------------------------------------


def test_transition_via_cli(home, capsys):
    _run(["create", "T", "--goal-file", str(_goal_file(home, b"g")),
          "--lane", "claude"], capsys)
    rc, out, _ = _run(
        ["transition", "1", "--status", "finished", "--step", "failed",
         "--reason", "done"],
        capsys,
    )
    assert rc == 0
    with jdb.connect_closing() as conn:
        assert jdb.get_job(conn, 1).status == "finished"


def test_transition_invalid_reports_nonzero(home, capsys):
    _run(["create", "T", "--goal-file", str(_goal_file(home, b"g")),
          "--lane", "claude"], capsys)
    _run(["transition", "1", "--status", "finished", "--step", "failed"], capsys)
    rc, _, err = _run(["transition", "1", "--status", "working", "--step", "building"], capsys)
    assert rc == 2
    assert "cannot transition" in err


def test_heartbeat_via_cli(home, capsys):
    _run(["create", "T", "--goal-file", str(_goal_file(home, b"g")),
          "--lane", "claude"], capsys)
    rc, out, _ = _run(["heartbeat", "1", "--at", "1234"], capsys)
    assert rc == 0
    with jdb.connect_closing() as conn:
        assert jdb.get_job(conn, 1).last_heartbeat_at == 1234


def test_events_via_cli(home, capsys):
    _run(["create", "T", "--goal-file", str(_goal_file(home, b"g")),
          "--lane", "claude"], capsys)
    _run(["heartbeat", "1", "--at", "5"], capsys)
    rc, out, _ = _run(["events", "1"], capsys)
    assert rc == 0
    assert "job_created" in out and "heartbeat" in out


# ---------------------------------------------------------------------------
# JSON output: deterministic + no environment leakage
# ---------------------------------------------------------------------------


def test_json_is_deterministic_and_leaks_no_env(home, capsys):
    gf = _goal_file(home)
    rc, out, _ = _run(
        ["create", "Ship", "--goal-file", str(gf), "--lane", "claude", "--json"],
        capsys,
    )
    assert rc == 0
    loaded = json.loads(out)
    # Canonical sorted form == the raw output (proves deterministic formatting).
    canonical = json.dumps(loaded, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    assert out == canonical
    # No environment / filesystem home path leaked into the payload.
    assert str(home) not in out
    assert "HERMES_HOME" not in out
    # Goal round-trips through JSON verbatim.
    assert loaded["goal"].encode("utf-8") == GOAL_BYTES


def test_show_json_matches_store(home, capsys):
    _run(["create", "J", "--goal-file", str(_goal_file(home, b"g")),
          "--lane", "claude"], capsys)
    rc, out, _ = _run(["show", "1", "--json"], capsys)
    assert rc == 0
    loaded = json.loads(out)
    assert loaded["number"] == 1
    assert loaded["label"] == "Job #1"
    assert loaded["status"] == "working"


# ---------------------------------------------------------------------------
# Top-level CLI wiring
# ---------------------------------------------------------------------------


def test_cmd_jobs_wired_into_main(home, capsys):
    """`hermes jobs` dispatches through main.cmd_jobs to jobs_command."""
    from hermes_cli.main import cmd_jobs

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    jobs_cli.build_parser(sub)
    args = parser.parse_args(["jobs", "list"])
    rc = cmd_jobs(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "No jobs yet" in out


# ---------------------------------------------------------------------------
# Jobs Execution V2 — custody CLI (claim / attempt / heartbeat / recover / intake)
# ---------------------------------------------------------------------------


import os
import stat


def _mk_job(home, capsys, name="A", goal=b"g", lane="claude"):
    argv = [
        "create",
        name,
        "--goal-file",
        str(_goal_file(home, goal)),
        "--lane",
        lane,
    ]
    _run(argv, capsys)


def test_claim_writes_token_to_file_never_stdout(home, capsys):
    _mk_job(home, capsys)
    tok = home / "tok"
    rc, out, err = _run(
        ["claim", "--worker", "w1", "--specialist", "claude-builder",
         "--lease-seconds", "60",
         "--token-out", str(tok), "--json"],
        capsys,
    )
    assert rc == 0
    payload = json.loads(out)
    assert payload["claimed"] is True
    assert payload["job"]["number"] == 1
    # The token is written to the file, never printed anywhere.
    assert tok.exists()
    token = tok.read_text().strip()
    assert token and token not in out and token not in err
    # 0600 permissions on the capability file.
    assert stat.S_IMODE(os.stat(tok).st_mode) == 0o600
    with jdb.connect_closing() as conn:
        assert jdb.get_job(conn, 1).claimed_by == "w1"


def test_claim_nothing_eligible_reports_not_claimed(home, capsys):
    tok = home / "tok"
    rc, out, _ = _run(
        ["claim", "--worker", "w1", "--specialist", "claude-builder",
         "--lease-seconds", "60",
         "--token-out", str(tok), "--json"],
        capsys,
    )
    assert rc == 0
    assert json.loads(out)["claimed"] is False
    assert not tok.exists()  # no token written when nothing was claimed


def test_attempt_start_finish_via_cli(home, capsys):
    _mk_job(home, capsys)
    tok = home / "tok"
    _run(["claim", "--worker", "w1", "--specialist", "claude-builder",
          "--lease-seconds", "60",
          "--token-out", str(tok), "--json"], capsys)

    rc, out, _ = _run(
        ["attempt-start", "1", "--claim-token-file", str(tok),
         "--specialist", "claude-builder", "--json"],
        capsys,
    )
    assert rc == 0
    att = json.loads(out)["attempt"]
    assert att["status"] == "running" and att["ordinal"] == 1
    aid = att["id"]
    token = tok.read_text().strip()
    assert token not in out  # token never echoed

    rc, out, _ = _run(
        ["attempt-finish", aid, "--claim-token-file", str(tok),
         "--status", "succeeded", "--json"],
        capsys,
    )
    assert rc == 0
    assert json.loads(out)["attempt"]["status"] == "succeeded"
    # The outcome moved the Job and cleared custody in the same write.
    with jdb.connect_closing() as conn:
        job = jdb.get_job(conn, 1)
    assert (job.status, job.step) == ("finished", "complete")
    assert job.claimed_by is None and job.current_attempt_id is None


def test_cli_attempt_history_reads_back_in_ordinal_order(home, capsys, monkeypatch):
    """The history a CLI surface reads back is the order the CLI reported.

    Real commands, real db: two attempts started in the same second, the
    correction drawing the id that sorts first. The ordinals the CLI printed as
    it ran are execution order by definition, so the public read has to hand the
    same sequence back — otherwise anything rendering this history (`show`, a
    future `attempts`/`history` view, an adapter) prints the chain backwards.
    """
    monkeypatch.setattr(jdb, "_now", lambda: 1000)
    ids = iter(["a_z_first", "a_a_last"])
    monkeypatch.setattr(jdb, "_new_attempt_id", lambda: next(ids))

    _mk_job(home, capsys)
    reported = []

    tok = home / "tok"
    _run(["claim", "--worker", "w1", "--specialist", "claude-builder",
          "--lease-seconds", "600", "--token-out", str(tok), "--json"], capsys)
    _, out, _ = _run(
        ["attempt-start", "1", "--claim-token-file", str(tok), "--json"], capsys
    )
    first = json.loads(out)["attempt"]
    reported.append((first["id"], first["ordinal"]))
    _run(["attempt-finish", first["id"], "--claim-token-file", str(tok),
          "--status", "failed", "--failure-class", "reviewer_rejection",
          "--json"], capsys)

    # The rejection cleared custody, so the corrector re-claims and continues.
    tok2 = home / "tok2"
    _run(["claim", "--worker", "w2", "--specialist", "claude-builder",
          "--lease-seconds", "600", "--token-out", str(tok2), "--json"], capsys)
    rc, out, _ = _run(
        ["attempt-start", "1", "--claim-token-file", str(tok2),
         "--parent", first["id"], "--json"],
        capsys,
    )
    assert rc == 0
    second = json.loads(out)["attempt"]
    reported.append((second["id"], second["ordinal"]))

    assert reported == [("a_z_first", 1), ("a_a_last", 2)]
    with jdb.connect_closing() as conn:
        history = jdb.get_attempts(conn, 1)
    assert [(a["id"], a["ordinal"]) for a in history] == reported
    assert {a["created_at"] for a in history} == {1000}  # one second, both


def test_attempt_finish_terminal_failure_via_cli(home, capsys):
    """finished/failed is reachable only via the explicit flag."""
    _mk_job(home, capsys)
    tok = home / "tok"
    _run(["claim", "--worker", "w1", "--specialist", "claude-builder",
          "--lease-seconds", "60",
          "--token-out", str(tok), "--json"], capsys)
    rc, out, _ = _run(
        ["attempt-start", "1", "--claim-token-file", str(tok), "--json"], capsys
    )
    aid = json.loads(out)["attempt"]["id"]
    rc, out, _ = _run(
        ["attempt-finish", aid, "--claim-token-file", str(tok), "--status", "failed",
         "--failure-class", "implementation", "--terminal-failure", "--json"],
        capsys,
    )
    assert rc == 0
    with jdb.connect_closing() as conn:
        job = jdb.get_job(conn, 1)
    assert (job.status, job.step) == ("finished", "failed")


def test_attempt_finish_conflicting_replay_fails_closed_via_cli(home, capsys):
    _mk_job(home, capsys)
    tok = home / "tok"
    _run(["claim", "--worker", "w1", "--specialist", "claude-builder",
          "--lease-seconds", "60",
          "--token-out", str(tok), "--json"], capsys)
    rc, out, _ = _run(
        ["attempt-start", "1", "--claim-token-file", str(tok), "--json"], capsys
    )
    aid = json.loads(out)["attempt"]["id"]
    _run(["attempt-finish", aid, "--claim-token-file", str(tok),
          "--status", "failed", "--failure-class", "max_turns", "--json"], capsys)
    rc, _, err = _run(
        ["attempt-finish", aid, "--claim-token-file", str(tok),
         "--status", "succeeded", "--json"],
        capsys,
    )
    assert rc == 2 and "already finished" in err
    with jdb.connect_closing() as conn:
        assert jdb.get_attempt(conn, aid)["status"] == "failed"


def test_attempt_start_rejects_inline_token_arg(home, capsys):
    """A token can never be passed on the command line — only via a file."""
    _mk_job(home, capsys)
    with pytest.raises(SystemExit):
        _run(["attempt-start", "1", "--claim-token", "abc"])


def test_attempt_finish_wrong_token_file_fails_closed(home, capsys):
    _mk_job(home, capsys)
    tok = home / "tok"
    _run(["claim", "--worker", "w1", "--specialist", "claude-builder",
          "--lease-seconds", "60",
          "--token-out", str(tok), "--json"], capsys)
    rc, out, _ = _run(
        ["attempt-start", "1", "--claim-token-file", str(tok), "--json"], capsys
    )
    aid = json.loads(out)["attempt"]["id"]
    bad = home / "bad"
    bad.write_text("not-the-token")
    rc, _, err = _run(
        ["attempt-finish", aid, "--claim-token-file", str(bad),
         "--status", "succeeded", "--json"],
        capsys,
    )
    assert rc == 2
    with jdb.connect_closing() as conn:
        assert jdb.get_attempt(conn, aid)["status"] == "running"


def test_claim_heartbeat_via_cli(home, capsys):
    _mk_job(home, capsys)
    tok = home / "tok"
    _run(["claim", "--worker", "w1", "--specialist", "claude-builder",
          "--lease-seconds", "60", "--at", "1000",
          "--token-out", str(tok), "--json"], capsys)
    rc, out, _ = _run(
        ["claim-heartbeat", "1", "--claim-token-file", str(tok),
         "--lease-seconds", "120", "--at", "1050", "--json"],
        capsys,
    )
    assert rc == 0
    assert json.loads(out)["job"]["lease_expires_at"] == 1170


def test_recover_expired_via_cli(home, capsys):
    _mk_job(home, capsys)
    tok = home / "tok"
    _run(["claim", "--worker", "w1", "--specialist", "claude-builder",
          "--lease-seconds", "60", "--at", "1000",
          "--token-out", str(tok), "--json"], capsys)
    rc, out, _ = _run(["recover-expired", "--at", "5000", "--json"], capsys)
    assert rc == 0
    assert json.loads(out)["count"] == 1
    with jdb.connect_closing() as conn:
        assert jdb.get_job(conn, 1).claimed_by is None


def test_intake_via_cli_is_idempotent(home, capsys):
    gf = _goal_file(home, b"run the thing")
    rc, out, _ = _run(
        ["intake", "Scheduled", "--source-type", "cron", "--source-key", "d1",
         "--goal-file", str(gf), "--lane", "claude", "--json"],
        capsys,
    )
    assert rc == 0
    first = json.loads(out)
    assert first["created"] is True

    rc, out, _ = _run(
        ["intake", "Scheduled", "--source-type", "cron", "--source-key", "d1",
         "--goal-file", str(gf), "--lane", "claude", "--json"],
        capsys,
    )
    second = json.loads(out)
    assert second["created"] is False and second["conflict"] is False
    assert second["job"]["id"] == first["job"]["id"]


# --- Token files are capability files: strict, not advisory ----------------


def _claimed_attempt(home, capsys, tok):
    """Claim job 1 into ``tok`` and start an attempt; return the attempt id."""
    _mk_job(home, capsys)
    _run(["claim", "--worker", "w1", "--specialist", "claude-builder",
          "--lease-seconds", "60",
          "--token-out", str(tok), "--json"], capsys)
    _, out, _ = _run(
        ["attempt-start", "1", "--claim-token-file", str(tok), "--json"], capsys
    )
    return json.loads(out)["attempt"]["id"]


def test_token_read_rejects_group_or_other_accessible_file(home, capsys):
    tok = home / "tok"
    aid = _claimed_attempt(home, capsys, tok)
    os.chmod(tok, 0o644)  # any group/other bit is a hard failure, not a warning
    rc, _, err = _run(
        ["attempt-finish", aid, "--claim-token-file", str(tok),
         "--status", "succeeded", "--json"],
        capsys,
    )
    assert rc == 2 and "group/other" in err
    with jdb.connect_closing() as conn:
        assert jdb.get_attempt(conn, aid)["status"] == "running"


def test_token_read_rejects_a_symlink(home, capsys):
    tok = home / "tok"
    aid = _claimed_attempt(home, capsys, tok)
    link = home / "link"
    link.symlink_to(tok)
    rc, _, err = _run(
        ["attempt-finish", aid, "--claim-token-file", str(link),
         "--status", "succeeded", "--json"],
        capsys,
    )
    assert rc == 2 and "symlink" in err


def test_token_read_rejects_a_non_regular_file(home, capsys):
    tok = home / "tok"
    aid = _claimed_attempt(home, capsys, tok)
    fifo = home / "fifo"
    os.mkfifo(fifo, 0o600)
    rc, _, err = _run(
        ["attempt-finish", aid, "--claim-token-file", str(fifo),
         "--status", "succeeded", "--json"],
        capsys,
    )
    assert rc == 2 and "regular file" in err


def test_token_read_rejects_a_file_owned_by_someone_else(home, capsys, monkeypatch):
    tok = home / "tok"
    aid = _claimed_attempt(home, capsys, tok)
    # Can't chown without privileges, so move *this* process's identity instead.
    monkeypatch.setattr(os, "getuid", lambda: os.stat(tok).st_uid + 1)
    rc, _, err = _run(
        ["attempt-finish", aid, "--claim-token-file", str(tok),
         "--status", "succeeded", "--json"],
        capsys,
    )
    assert rc == 2 and "owned" in err


def test_token_write_refuses_to_follow_a_symlink(home, capsys):
    _mk_job(home, capsys)
    target = home / "attacker_target"
    target.write_text("")
    link = home / "tok"
    link.symlink_to(target)
    rc, _, err = _run(
        ["claim", "--worker", "w1", "--specialist", "claude-builder",
         "--lease-seconds", "60",
         "--token-out", str(link), "--json"],
        capsys,
    )
    assert rc == 2
    assert target.read_text() == ""  # the capability never reached the target
    with jdb.connect_closing() as conn:
        assert jdb.get_job(conn, 1).claimed_by is None  # and the claim was undone


def test_token_write_refuses_to_overwrite_an_existing_file(home, capsys):
    _mk_job(home, capsys)
    tok = home / "tok"
    tok.write_text("decoy")
    rc, _, err = _run(
        ["claim", "--worker", "w1", "--specialist", "claude-builder",
         "--lease-seconds", "60",
         "--token-out", str(tok), "--json"],
        capsys,
    )
    assert rc == 2
    assert tok.read_text() == "decoy"


def test_claim_is_released_when_the_token_file_cannot_be_written(home, capsys):
    """A DB claim no one can hold must not sit on the Job until it expires."""
    _mk_job(home, capsys)
    blocked = home / "blocked"
    blocked.mkdir()  # os.open() for writing fails with EISDIR
    rc, _, err = _run(
        ["claim", "--worker", "w1", "--specialist", "claude-builder",
         "--lease-seconds", "60",
         "--token-out", str(blocked), "--json"],
        capsys,
    )
    assert rc == 2
    with jdb.connect_closing() as conn:
        job = jdb.get_job(conn, 1)
        kinds = [e["kind"] for e in jdb.get_events(conn, 1)]
    assert job.claimed_by is None and job.lease_expires_at is None
    assert kinds[-1] == "claim_released"  # the exact claim was undone
    # And the Job is immediately claimable again.
    ok = home / "tok2"
    rc, out, _ = _run(
        ["claim", "--worker", "w2", "--specialist", "claude-builder",
         "--lease-seconds", "60",
         "--token-out", str(ok), "--json"],
        capsys,
    )
    assert rc == 0 and json.loads(out)["claimed"] is True


def test_token_never_surfaces_in_show_or_events(home, capsys):
    _mk_job(home, capsys)
    tok = home / "tok"
    _run(["claim", "--worker", "w1", "--specialist", "claude-builder",
          "--lease-seconds", "60",
          "--token-out", str(tok), "--json"], capsys)
    token = tok.read_text().strip()

    rc, show_out, _ = _run(["show", "1", "--json"], capsys)
    rc2, ev_out, _ = _run(["events", "1", "--json"], capsys)
    assert token not in show_out
    assert token not in ev_out
    assert "claim_token" not in show_out
