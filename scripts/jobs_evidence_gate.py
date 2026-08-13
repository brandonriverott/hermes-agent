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

This file performs no network, subprocess, or VCS action. Settlement reads a
job directory and writes three files into it: ``done.json`` (unchanged key set,
for compatibility), ``gate.json`` (the full decision), and
``action-packet.md`` (what a human reads). ``bridge-verify`` is read-only: it
validates those files and independently re-evaluates every claimed success from
the stamped tester/reviewer receipts before the Mac bridge emits an action. It
never merges, pushes, deploys, or sends.

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


def _require_nonempty_string(doc: dict, field: str, filename: str) -> str:
    value = doc.get(field)
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(
            f"{filename} is missing a non-empty {field}.",
            "EVIDENCE_MALFORMED",
        )
    return value


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
        if review_verdict not in ("PASS", "NEEDS_CHANGES", "UNABLE_TO_VERIFY"):
            sections["independent_review"] = f"BLOCKED — verdict {review_verdict!r} is not a verdict"
            result = park("REVIEW_ERROR", ["The independent reviewer did not return a usable verdict."])
            result.findings = findings
            return result
        sections["independent_review"] = f"{review_verdict} on attempt {attempt}, {findings} finding(s)"
        if review_verdict != "PASS":
            result = park(
                review_verdict,
                [
                    "The independent reviewer could not verify the required evidence."
                    if review_verdict == "UNABLE_TO_VERIFY"
                    else f"The independent review returned {findings} unresolved finding(s)."
                ],
            )
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


def verify_bridge_settlement(job_dir, *, job: str, base: str) -> dict:
    """Return normalized action data only when gate output is authoritative.

    ``done.json`` remains the compatibility document, but it cannot authorize a
    successful action.  Identity and outcome come from ``gate.json``; every
    claimed success is then recomputed from the exact stamped receipts.  This
    closes the bridge bypass where a builder wrote a green-looking ``done.json``
    before the runner reached its gate.

    A valid *blocked* gate is returned as ``action_outcome=failed`` with its
    first plain-English reason. Invalid or absent gate evidence raises
    :class:`EvidenceError`, which the bridge renders as an infrastructure
    failure instead of an action packet that could be mistaken for success.
    """
    job_dir = Path(job_dir)
    gate_doc = _read_json(job_dir / "gate.json")
    done = _read_json(job_dir / "done.json")

    gate_job = _require_nonempty_string(gate_doc, "job", "gate.json")
    if gate_job != job:
        raise EvidenceError(
            f"gate.json is evidence for job {gate_job!r}, but the bridge is handling job {job!r}.",
            "EVIDENCE_WRONG_JOB",
        )

    gate_base = _require_nonempty_string(gate_doc, "base", "gate.json")
    if gate_base != base:
        raise EvidenceError(
            f"gate.json was settled from base {_short(gate_base)}, but the bridge started from {_short(base)}.",
            "EVIDENCE_STALE_COMMIT",
        )

    gate_commit = gate_doc.get("commit")
    if not isinstance(gate_commit, str):
        raise EvidenceError("gate.json commit is not a string.", "EVIDENCE_MALFORMED")
    done_commit = done.get("commit")
    if not isinstance(done_commit, str):
        raise EvidenceError("done.json commit is not a string.", "EVIDENCE_MALFORMED")
    if gate_commit != done_commit:
        raise EvidenceError(
            f"gate.json names commit {_short(gate_commit)}, but done.json names {_short(done_commit)}.",
            "EVIDENCE_STALE_COMMIT",
        )

    attempt = gate_doc.get("attempt")
    rounds = done.get("review_rounds")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise EvidenceError("gate.json attempt is not a non-negative integer.", "EVIDENCE_MALFORMED")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
        raise EvidenceError("done.json review_rounds is not a non-negative integer.", "EVIDENCE_MALFORMED")
    expected_attempt = max(0, rounds - 1)
    if attempt != expected_attempt:
        raise EvidenceError(
            f"gate.json settles review attempt {attempt}, but done.json records {rounds} review round(s).",
            "EVIDENCE_STALE_ATTEMPT",
        )

    gate_outcome = gate_doc.get("outcome")
    done_outcome = done.get("outcome")
    if gate_outcome not in ("succeeded", "needs_attention", "failed"):
        raise EvidenceError(
            f"gate.json outcome {gate_outcome!r} is not a settlement outcome.",
            "EVIDENCE_MALFORMED",
        )
    if done_outcome != gate_outcome:
        raise EvidenceError(
            f"gate.json settles {gate_outcome!r}, but done.json reports {done_outcome!r}.",
            "EVIDENCE_MALFORMED",
        )

    verdict = _require_nonempty_string(gate_doc, "verdict", "gate.json")
    if done.get("review_verdict") != verdict:
        raise EvidenceError(
            f"gate.json verdict {verdict!r} does not match done.json verdict {done.get('review_verdict')!r}.",
            "EVIDENCE_MALFORMED",
        )
    branch = _require_nonempty_string(done, "branch", "done.json")
    expected_branches = {
        f"jobs/{job}",
        f"jobs/{job}/attempt-{attempt}",
    }
    if branch not in expected_branches:
        raise EvidenceError(
            f"done.json names branch {branch!r}, but this settlement owns the Job's attempt branch.",
            "EVIDENCE_WRONG_JOB",
        )
    build_exit = done.get("claude_exit")
    if isinstance(build_exit, bool) or not isinstance(build_exit, int):
        raise EvidenceError("done.json claude_exit is not an integer.", "EVIDENCE_MALFORMED")

    reasons = gate_doc.get("reasons")
    if not isinstance(reasons, list) or not all(
        isinstance(reason, str) and reason.strip() for reason in reasons
    ):
        raise EvidenceError(
            "gate.json reasons must be a list of non-empty strings.",
            "EVIDENCE_MALFORMED",
        )
    sections = gate_doc.get("sections")
    required_sections = {key for key, _label in _SECTION_LABELS}
    if (
        not isinstance(sections, dict)
        or set(sections) != required_sections
        or not all(isinstance(value, str) and value.strip() for value in sections.values())
    ):
        raise EvidenceError(
            "gate.json does not contain the complete stage-by-stage settlement record.",
            "EVIDENCE_MALFORMED",
        )
    _require_nonempty_string(gate_doc, "settled_at", "gate.json")

    # A claimed success has one more burden than a blocked result: prove it
    # again from the current receipts. This is deliberately evaluate(), not
    # settle(), so bridge verification cannot rewrite job artifacts.
    if gate_outcome == "succeeded":
        if not _HEX40.match(gate_commit) or gate_commit == base:
            raise EvidenceError(
                "gate.json claims success without a candidate commit past the approved base.",
                "NO_CANDIDATE_COMMIT",
            )
        rechecked = evaluate(
            job_dir,
            job=job,
            branch=branch,
            base=base,
            commit=gate_commit,
            attempt=attempt,
            rounds=rounds,
            build_exit=build_exit,
            correction_exit=0,
            test_exit=0,
            review_exit=0,
            souls_ok=True,
            review_required=True,
        )
        if not rechecked.succeeded:
            reason = rechecked.reasons[0] if rechecked.reasons else "The success could not be reproduced from its receipts."
            raise EvidenceError(reason, rechecked.verdict)
        if verdict != rechecked.verdict:
            raise EvidenceError(
                f"gate.json verdict {verdict!r} does not match the receipt-derived verdict {rechecked.verdict!r}.",
                "EVIDENCE_MALFORMED",
            )
        reason = "all required checks passed and review returned PASS"
    else:
        if not reasons:
            raise EvidenceError(
                "gate.json blocks settlement but records no plain-English reason.",
                "EVIDENCE_MALFORMED",
            )
        reason = reasons[0]

    return {
        "action_outcome": "succeeded" if gate_outcome == "succeeded" else "failed",
        "gate_outcome": gate_outcome,
        "commit": gate_commit,
        "branch": branch,
        "review_verdict": verdict,
        "review_rounds": rounds,
        "reason": reason,
        "stage_record": sections,
    }


def graph_settlement(job_dir, *, job: str, base: str) -> dict:
    """Normalize bridge verification for the reliability graph.

    The graph never reads ``done.json`` directly.  It receives a successful
    decision only from :func:`verify_bridge_settlement`, which binds the gate,
    compatibility document, tests, review, Job, attempt, base, and candidate
    commit.  Invalid evidence is data here rather than an exception so the
    dispatcher can record a typed terminal edge without accidentally treating
    an unreadable gate as a crash outside the attempt.
    """

    try:
        verified = verify_bridge_settlement(job_dir, job=job, base=base)
    except EvidenceError as exc:
        return {
            "action_outcome": "failed",
            "identity_verified": False,
            "reason_code": exc.verdict,
            "reason": exc.reason,
            "commit": None,
        }
    return {
        **verified,
        "identity_verified": True,
        "reason_code": "OK" if verified["action_outcome"] == "succeeded" else verified["review_verdict"],
    }


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

    bridge = sub.add_parser(
        "bridge-verify",
        help="verify gate output and emit normalized bridge action data",
    )
    bridge.add_argument("--job-dir", required=True)
    bridge.add_argument("--job", required=True)
    bridge.add_argument("--base", required=True)

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

    if args.mode == "bridge-verify":
        try:
            decision = verify_bridge_settlement(
                args.job_dir,
                job=args.job,
                base=args.base,
            )
        except EvidenceError as exc:
            print(f"{exc.verdict}: {exc.reason}", file=sys.stderr)
            return 2
        sys.stdout.write(json.dumps(decision, separators=(",", ":")) + "\n")
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
