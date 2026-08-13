from __future__ import annotations

import argparse
import json

from hermes_cli import jobs as jobs_cli
from hermes_cli import jobs_run


def test_run_once_uses_canonical_dispatch_when_enabled(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HERMES_JOBS_DISPATCH", "1")
    lane_root = tmp_path / "lanes"
    lane_root.mkdir()
    monkeypatch.setenv("HERMES_JOBS_LANE_ROOT", str(lane_root))

    calls = {}

    def canonical(**kwargs):
        calls.update(kwargs)
        return {
            "ran": True,
            "job": {"id": "j_1", "number": 1, "label": "Job #1"},
            "attempt_id": "a_1",
            "ordinal": 1,
            "status": "succeeded",
            "job_status": "finished",
            "job_step": "complete",
            "error": None,
        }

    monkeypatch.setattr(jobs_run, "canonical_dispatch_once", canonical)
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    jobs_cli.build_parser(sub)
    args = parser.parse_args([
        "jobs",
        "run-once",
        "--worker",
        "/bin/false",
        "--workspace-root",
        str(tmp_path / "ws"),
        "--execution-file",
        str(tmp_path / "missing.json"),
        "--worker-id",
        "runner",
        "--job",
        "1",
        "--json",
    ])

    assert jobs_cli.jobs_command(args) == 0
    assert json.loads(capsys.readouterr().out)["job"]["number"] == 1
    assert calls == {
        "lane_root": lane_root,
        "worker_id": "runner",
        "job": "1",
        "now": None,
    }
