#!/usr/bin/env python3
"""Evidence gate for the live Hermes Jobs pipeline (``pc-jobs-worker.sh``).

The Jobs runner settles every build by writing ``<job_dir>/done.json``. Until
now that settlement read two receipts — VULCAN's ``tests.json`` and THEMIS's
``review-N.json`` — that carried **no identity at all**: no job key, no attempt
number, no candidate commit. Nothing downstream could tell evidence for this
job from evidence left over in the directory by an earlier run, an earlier
attempt, or a builder with a Bash tool and a motive.

This module is that missing check. It is the only thing allowed to decide
``outcome`` — the runner hands it the process exit codes and the candidate
identity, and it answers with a settlement plus a plain-English reason.

Two rules it never bends:

* **Fail closed.** Missing, malformed, unstamped, stale or mismatched evidence
  parks the job in ``needs_attention``. There is no input — including a
  reviewer that said PASS — that turns an absence of evidence into a success.
* **Keep the stages apart.** Tracker acceptance, worker claim/start, execution
  evidence, tests, independent review, settlement, and any later
  merge/deploy/live verification are rendered as separate lines with separate
  outcomes. A queued row or a builder's own summary is never proof of anything.

Identity binding works by stamping. The runner — not the model — writes a
``_gate`` envelope onto each receipt the moment it is produced, from shell
variables the model cannot reach::

    "_gate": {"job": "<key>", "attempt": 0, "commit": "<40-hex HEAD>",
              "base": "<40-hex>", "kind": "tests", "stamped_at": "<iso8601>"}

At settlement the gate requires that envelope to name *this* job, *this*
attempt and *this* candidate commit. Any model-authored ``_gate`` in the raw
output is overwritten at stamp time, so a forged envelope cannot survive.

This file performs no network, subprocess, or VCS action. It reads a job
directory and writes three files into it: ``done.json`` (unchanged key set, for
the Mac bridge), ``gate.json`` (the full decision), and ``action-packet.md``
(what a human reads). It never merges, pushes, deploys, or sends.

Activation is a separate, explicit step — see
``scripts/jobs_evidence_gate_activation.md``. Nothing here is wired into the
live runner by this commit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

GATE_KEY = "_gate"
RECEIPT_KINDS = ("tests", "review")
TEST_RESULTS = ("pass", "fail", "flaky")

# A failing check has to say *why* it is failing. "It was already broken" is a
# legitimate answer, but only when someone writes it down — an unexplained red
# is the shape every "succeeded" build with failing tests has arrived in.
FAILURE_CLASSES = {
    "candidate_regression": "broken by this candidate",
    "pre_existing": "already failing at the base commit",
    "infrastructure": "host or environment problem, not the code",
    "prerequisite_unavailable": "a prerequisite for the check was unavailable",
}
# Classified-but-not-ours failures are honest, and still not green.
RETAINED_CLASSES = ("pre_existing", "infrastructure", "prerequisite_unavailable")

# done.json's key set is the Mac bridge's contract. Adding to it is a
# compatibility break; the gate's own detail goes to gate.json instead.
DONE_KEYS = (
    "outcome",
    "claude_exit",
    "commit",
    "branch",
    "review_verdict",
    "review_rounds",
    "findings",
    "effort",
)

_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")


class EvidenceError(Exception):
    """Evidence that cannot be trusted, with the verdict it settles as."""

    def __init__(self, reason: str, verdict: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.verdict = verdict


# ── receipts ──────────────────────────────────────────────────────────


def _read_json(path: Path, verdict_on_missing: str = "EVIDENCE_MISSING") -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise EvidenceError(f"{path.name} does not exist.", verdict_on_missing) from None
    except OSError as exc:
        raise EvidenceError(f"{path.name} could not be read ({exc}).", "EVIDENCE_MALFORMED") from None
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise EvidenceError(f"{path.name} is not valid JSON ({exc}).", "EVIDENCE_MALFORMED") from None
    if not isinstance(doc, dict):
        raise EvidenceError(f"{path.name} is not a JSON object.", "EVIDENCE_MALFORMED")
    return doc


def stamp_receipt(path, *, kind: str, job: str, attempt: int, commit: str, base: str) -> dict:
    """Bind a receipt to the job, attempt and commit that produced it.

    Called by the runner immediately after a receipt is written, from values
    the model never sees. Any ``_gate`` the model emitted is replaced.
    """
    if kind not in RECEIPT_KINDS:
        raise ValueError(f"unknown receipt kind {kind!r}")
    path = Path(path)
    doc = _read_json(path)
    doc[GATE_KEY] = {
        "job": job,
        "attempt": int(attempt),
        "commit": commit,
        "base": base,
        "kind": kind,
        "stamped_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(path, doc)
    return doc


def load_receipt(path, *, kind: str, job: str, attempt: int, commit: str) -> dict:
    """Read a receipt and prove it belongs to this job, attempt and candidate."""
    path = Path(path)
    doc = _read_json(path)
    env = doc.get(GATE_KEY)
    if not isinstance(env, dict):
        raise EvidenceError(
            f"{path.name} carries no identity stamp, so there is nothing tying it to "
            f"this job or this candidate.",
            "EVIDENCE_NOT_IDENTITY_BOUND",
        )
    for field in ("job", "attempt", "commit", "kind"):
        if env.get(field) in (None, ""):
            raise EvidenceError(
                f"{path.name}'s identity stamp is missing its {field}.",
                "EVIDENCE_NOT_IDENTITY_BOUND",
            )
    if env.get("kind") != kind:
        raise EvidenceError(
            f"{path.name} is stamped as {env.get('kind')!r} evidence but was read as "
            f"{kind!r} evidence.",
            "EVIDENCE_WRONG_JOB",
        )
    if env.get("job") != job:
        raise EvidenceError(
            f"{path.name} is evidence for job {env.get('job')!r}, but this is job {job!r}.",
            "EVIDENCE_WRONG_JOB",
        )
    if env.get("attempt") != attempt:
        raise EvidenceError(
            f"{path.name} is evidence from attempt {env.get('attempt')}, but attempt "
            f"{attempt} is being settled.",
            "EVIDENCE_STALE_ATTEMPT",
        )
    if env.get("commit") != commit:
        raise EvidenceError(
            f"{path.name} was gathered at commit {_short(env.get('commit'))}, but the "
            f"candidate being settled is {_short(commit)}.",
            "EVIDENCE_STALE_COMMIT",
        )
    return doc


def _short(sha) -> str:
    text = str(sha or "")
    return text[:12] if text else "(none)"


# ── the test receipt's own truth table ─────────────────────────────────


class TestEvidence:
    """What VULCAN's receipt actually supports, with failures classified."""

    def __init__(self, verdict, reasons, lines, passed, failures):
        self.verdict = verdict  # None => nothing blocking
        self.reasons = reasons
        self.lines = lines
        self.passed = passed
        self.failures = failures


def classify_tests(doc: dict) -> TestEvidence:
    raw = doc.get("tests")
    tests = list(raw) if isinstance(raw, list) else []
    usable = bool(tests) and all(
        isinstance(item, dict)
        and str(item.get("cmd") or "").strip()
        and item.get("result") in TEST_RESULTS
        and str(item.get("evidence") or "").strip()
        for item in tests
    )
    if not usable:
        return TestEvidence(
            "PASS_WITHOUT_TEST_EVIDENCE",
            ["The test receipt names no check that both ran and reported a result."],
            [],
            [],
            [],
        )

    passed = [t for t in tests if t["result"] == "pass"]
    failures = [t for t in tests if t["result"] != "pass"]

    unclassified = []
    for item in failures:
        cls = item.get("classification")
        note = str(item.get("note") or "").strip()
        if cls not in FAILURE_CLASSES or not note:
            unclassified.append(item)
    if unclassified:
        return TestEvidence(
            "EVIDENCE_UNCLASSIFIED_FAILURE",
            [
                f"Check {item['cmd']!r} reported {item['result']} and was not classified "
                f"(needs one of {', '.join(sorted(FAILURE_CLASSES))} plus a note)."
                for item in unclassified
            ],
            [],
            passed,
            failures,
        )

    lines = [
        f"{item['cmd']} — {item['result']}, {item['classification']} "
        f"({FAILURE_CLASSES[item['classification']]}): {item['note']}"
        for item in failures
    ]
    regressions = [t for t in failures if t["classification"] == "candidate_regression"]
    if regressions:
        return TestEvidence(
            "PASS_WITH_FAILING_TESTS",
            [
                f"Check {item['cmd']!r} fails because of this candidate."
                for item in regressions
            ],
            lines,
            passed,
            failures,
        )
    if failures:
        return TestEvidence(
            "PASS_WITH_CLASSIFIED_FAILURES",
            [
                f"Check {item['cmd']!r} is red and classified {item['classification']} — "
                f"honest, but not a green run."
                for item in failures
            ],
            lines,
            passed,
            failures,
        )
    return TestEvidence(None, [], [], passed, [])


# ── the settlement ────────────────────────────────────────────────────


class GateResult:
    def __init__(self, outcome, verdict, reasons, sections, findings):
        self.outcome = outcome
        self.verdict = verdict
        self.reasons = reasons
        self.sections = sections
        self.findings = findings

    @property
    def succeeded(self) -> bool:
        return self.outcome == "succeeded"


def evaluate(
    job_dir,
    *,
    job: str,
    branch: str,
    base: str,
    commit: str,
    attempt: int,
    rounds: int,
    build_exit: int,
    correction_exit: int = 0,
    test_exit: int = 0,
    review_exit: int = 0,
    souls_ok: bool = True,
    review_required: bool = True,
) -> GateResult:
    """Decide the settlement for one Jobs attempt. Never raises on bad evidence."""
    job_dir = Path(job_dir)
    sections = {
        "tracker_acceptance": "accepted and routed to a worker — a queued row, not proof of work",
        "worker_claim": f"claimed as {branch} from base {_short(base)}",
        "execution_evidence": "no candidate commit",
        "tests": "not reached",
        "independent_review": "not reached",
        "settlement": "pending",
        "post_settlement": (
            "merge / deploy / live verification: not performed — this gate never "
            "merges, deploys, or sends"
        ),
    }
    reasons = []

    def park(verdict, why, outcome="needs_attention"):
        sections["settlement"] = f"{outcome} — {verdict}"
        return GateResult(outcome, verdict, why, sections, 0)

    # Process-level failures first: no evidence claim can outrank a stage that
    # never ran. Verdict strings here match what the runner already emits.
    if not souls_ok:
        return park(
            "REVIEWER_PROMPTS_MISSING",
            ["The sealed reviewer instructions were missing or altered, so no review could be trusted."],
            outcome="failed",
        )
    if build_exit == 97:
        return park("KAT_ENDPOINT_DOWN", ["The local build model endpoint was unreachable."], outcome="failed")
    if build_exit != 0:
        return park(
            "NOT_REVIEWED",
            [f"The builder process exited {build_exit}; nothing was reviewed."],
            outcome="failed",
        )

    if not commit or not _HEX40.match(str(commit)) or commit == base:
        return park(
            "NO_CANDIDATE_COMMIT",
            ["The builder left no commit past the base, so there is nothing to certify."],
        )
    sections["execution_evidence"] = f"candidate {_short(commit)} on {branch}, past base {_short(base)}"

    if correction_exit != 0:
        return park("CORRECTION_PROCESS_FAILED", [f"The correction round exited {correction_exit}."])
    if test_exit != 0:
        return park("TESTER_PROCESS_FAILED", [f"The tester process exited {test_exit}; its partial output is not evidence."])
    if review_exit != 0:
        return park("REVIEWER_PROCESS_FAILED", [f"The reviewer process exited {review_exit}; silence is not approval."])

    # Evidence, each class read and bound separately.
    try:
        tests_doc = load_receipt(
            job_dir / "tests.json", kind="tests", job=job, attempt=attempt, commit=commit
        )
    except EvidenceError as exc:
        sections["tests"] = f"BLOCKED — {exc.reason}"
        return park(exc.verdict, [exc.reason])

    findings = 0
    if review_required:
        try:
            review_doc = load_receipt(
                job_dir / f"review-{attempt}.json",
                kind="review",
                job=job,
                attempt=attempt,
                commit=commit,
            )
        except EvidenceError as exc:
            sections["tests"] = "evidence accepted, pending review"
            sections["independent_review"] = f"BLOCKED — {exc.reason}"
            return park(exc.verdict, [exc.reason])
        review_verdict = review_doc.get("verdict")
        found = review_doc.get("findings")
        findings = len(found) if isinstance(found, list) else 0
        if review_verdict not in ("PASS", "NEEDS_CHANGES"):
            sections["independent_review"] = f"BLOCKED — verdict {review_verdict!r} is not a verdict"
            result = park("REVIEW_ERROR", ["The independent reviewer did not return a usable verdict."])
            result.findings = findings
            return result
        sections["independent_review"] = f"{review_verdict} on attempt {attempt}, {findings} finding(s)"
        if review_verdict != "PASS":
            result = park("NEEDS_CHANGES", [f"The independent review returned {findings} unresolved finding(s)."])
            result.findings = findings
            return result
    else:
        sections["independent_review"] = "not required by this workflow"

    evidence = classify_tests(tests_doc)
    ran = len(evidence.passed) + len(evidence.failures)
    if evidence.verdict is not None:
        sections["tests"] = f"BLOCKED — {ran} check(s), {len(evidence.failures)} not passing"
        if evidence.lines:
            sections["tests"] += "\n    " + "\n    ".join(evidence.lines)
        # A reviewer's PASS does not reach across this line.
        result = park(evidence.verdict, evidence.reasons)
        result.findings = findings
        return result

    sections["tests"] = f"{ran} check(s) named, all passing"
    sections["settlement"] = "succeeded — PASS"
    return GateResult("succeeded", "PASS", reasons, sections, findings)


# ── output ────────────────────────────────────────────────────────────

_SECTION_LABELS = (
    ("tracker_acceptance", "Tracker acceptance"),
    ("worker_claim", "Worker claim / start"),
    ("execution_evidence", "Execution evidence"),
    ("tests", "Tests"),
    ("independent_review", "Independent review"),
    ("settlement", "Settlement"),
    ("post_settlement", "After settlement"),
)


def render_packet(result: GateResult, *, job: str) -> str:
    headline = "complete" if result.succeeded else "attention needed"
    out = [f"Job {job} — {headline}", ""]
    width = max(len(label) for _, label in _SECTION_LABELS) + 1
    for key, label in _SECTION_LABELS:
        out.append(f"{(label + ':').ljust(width)} {result.sections[key]}")
    if not result.succeeded:
        out += ["", "Why this is not a success:"]
        out += [f"  - {reason}" for reason in result.reasons] or ["  - No reason was recorded, which is itself disqualifying."]
    return "\n".join(out) + "\n"


def _atomic_json(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def settle(job_dir, *, effort: str = "medium", **kwargs) -> GateResult:
    """Evaluate and write done.json, gate.json and action-packet.md."""
    job_dir = Path(job_dir)
    result = evaluate(job_dir, **kwargs)
    done = {
        "outcome": result.outcome,
        "claude_exit": kwargs["build_exit"],
        "commit": kwargs["commit"],
        "branch": kwargs["branch"],
        "review_verdict": result.verdict,
        "review_rounds": kwargs["rounds"],
        "findings": result.findings,
        "effort": effort,
    }
    packet = render_packet(result, job=kwargs["job"])
    # gate.json first: it is what the supervisor uses to tell a settled build
    # from a directory a builder wrote done.json into by hand.
    _atomic_json(
        job_dir / "gate.json",
        {
            "job": kwargs["job"],
            "attempt": kwargs["attempt"],
            "commit": kwargs["commit"],
            "base": kwargs["base"],
            "outcome": result.outcome,
            "verdict": result.verdict,
            "reasons": result.reasons,
            "sections": result.sections,
            "settled_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _atomic_text(job_dir / "action-packet.md", packet)
    _atomic_json(job_dir / "done.json", done)
    return result


# ── CLI (the seam the runner calls) ───────────────────────────────────


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evidence gate for the Hermes Jobs pipeline.")
    sub = parser.add_subparsers(dest="mode", required=True)

    stamp = sub.add_parser("stamp", help="bind a receipt to this job/attempt/commit")
    stamp.add_argument("--file", required=True)
    stamp.add_argument("--kind", required=True, choices=RECEIPT_KINDS)
    stamp.add_argument("--job", required=True)
    stamp.add_argument("--attempt", required=True, type=int)
    stamp.add_argument("--commit", required=True)
    stamp.add_argument("--base", required=True)

    gate = sub.add_parser("settle", help="decide the outcome and write done.json")
    gate.add_argument("--job-dir", required=True)
    gate.add_argument("--job", required=True)
    gate.add_argument("--branch", required=True)
    gate.add_argument("--base", required=True)
    gate.add_argument("--commit", default="")
    gate.add_argument("--attempt", type=int, default=0)
    gate.add_argument("--rounds", type=int, default=0)
    gate.add_argument("--build-exit", type=int, default=0)
    gate.add_argument("--correction-exit", type=int, default=0)
    gate.add_argument("--test-exit", type=int, default=0)
    gate.add_argument("--review-exit", type=int, default=0)
    gate.add_argument("--effort", default="medium")
    gate.add_argument("--souls-ok", type=int, default=1)
    gate.add_argument("--no-review-required", action="store_true")

    args = parser.parse_args(argv)

    if args.mode == "stamp":
        try:
            stamp_receipt(
                args.file,
                kind=args.kind,
                job=args.job,
                attempt=args.attempt,
                commit=args.commit,
                base=args.base,
            )
        except EvidenceError as exc:
            print(f"stamp refused: {exc.reason}", file=sys.stderr)
            return 1
        return 0

    result = settle(
        args.job_dir,
        job=args.job,
        branch=args.branch,
        base=args.base,
        commit=args.commit,
        attempt=args.attempt,
        rounds=args.rounds,
        build_exit=args.build_exit,
        correction_exit=args.correction_exit,
        test_exit=args.test_exit,
        review_exit=args.review_exit,
        souls_ok=bool(args.souls_ok),
        review_required=not args.no_review_required,
        effort=args.effort,
    )
    sys.stdout.write(render_packet(result, job=args.job))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
