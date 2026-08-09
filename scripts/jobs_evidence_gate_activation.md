# Activating the Jobs evidence gate

**Status: not active.** This commit adds the gate, its tests, and the runner
patch. Nothing in the live Jobs pipeline calls it yet. Until the two deploy
steps below are performed and verified, `pc-jobs-worker.sh` still settles jobs
with its own inline truth table and no identity-bound evidence.

## What the gate is

`scripts/jobs_evidence_gate.py` decides one thing: whether a Jobs attempt may
settle as complete. It replaces the truth table that currently lives inline at
the bottom of `pc-jobs-worker.sh`'s `inner` mode.

It refuses to settle `succeeded` unless, for **this** job key, **this** attempt
number and **this** candidate commit, it can read:

- a tester receipt (`tests.json`) naming at least one check that ran, with a
  result and decisive output, all of them passing; and
- an independent-review receipt (`review-N.json`) with verdict `PASS`.

Anything else — a missing file, unparseable JSON, an unstamped receipt, a
receipt stamped for another job, another attempt or another commit, a red check
with no classification, a red check classified `candidate_regression`, a
reviewer that crashed — settles `needs_attention` with a plain-English reason in
`action-packet.md`. There is no input that converts an absence of evidence into
a success, including a reviewer that said PASS.

Red checks that are genuinely not the candidate's fault are still not green.
They must be classified `pre_existing`, `infrastructure` or
`prerequisite_unavailable` **with a note**, and they settle
`needs_attention` / `PASS_WITH_CLASSIFIED_FAILURES` — honest, and Brandon's
call, which is the only honest place for that call to be made.

## How identity binding works

Receipts had no identity at all: `tests.json` was `{tests, skipped,
verdict_hint}` and `review-N.json` was `{verdict, findings, checks_run,
checks_skipped}`. No job key, no attempt, no commit. Settlement could not tell
this attempt's evidence from a leftover file.

The patched runner stamps each receipt the moment it is produced, from shell
variables the model never sees:

```json
"_gate": {"job": "75-a_c8826153", "attempt": 0,
          "commit": "<40-hex HEAD>", "base": "<40-hex>",
          "kind": "tests", "stamped_at": "<iso8601>"}
```

`HEAD` is read *after* the seat has run, not before. VULCAN and THEMIS both hold
Bash, so either can commit; evidence gathered at a commit that is not the
candidate now fails the gate instead of certifying it. Any `_gate` the model
emitted in its own output is overwritten at stamp time.

## Deploy steps (both, or neither)

`pc-jobs-worker.sh` is a release gate: the Mac and the PC must run
byte-identical bytes. They drifted 84 lines apart once, and the drift was
exactly where failing tests became a "succeeded" build. The gate inherits that
rule — a stamped receipt from one machine must mean the same thing on the other.

1. **Copy the gate beside the runner on both machines:**

   ```sh
   install -m 0755 scripts/jobs_evidence_gate.py ~/.local/bin/jobs_evidence_gate.py
   ```

   Beside the runner, *not* inside the worktree: a builder holds Write and Bash
   over everything under `$job_dir/wt`, so a gate living there would be a gate
   the candidate could edit — the same exposure that put the reviewer souls
   under a hash seal.

2. **Apply the runner patch on both machines:**

   ```sh
   cd ~/.local/bin
   cp pc-jobs-worker.sh pc-jobs-worker.sh.bak-pregate-$(date +%Y%m%d)
   git apply -p1 /path/to/hermes-agent/scripts/jobs_evidence_gate.runner.patch
   bash -n pc-jobs-worker.sh
   shasum -a 256 pc-jobs-worker.sh jobs_evidence_gate.py   # must match on both
   ```

3. **Verify with a canary job** before trusting it on real work: one job whose
   tester returns a green receipt should settle `succeeded`; one whose evidence
   is removed before settlement should settle `needs_attention` with
   `EVIDENCE_MISSING`.

The patch is verified by `tests/scripts/test_jobs_evidence_gate_runner.py`,
which copies the installed runner into a scratch directory, applies this patch,
and drives a real build with a stubbed `claude`. Those tests skip when the
runner is not installed — that skip is the activation boundary.

## What the patch changes

| Location | Change |
|---|---|
| top of file | `GATE=` path and the `gate_stamp` helper |
| `run_tests` | takes the attempt number; stamps `tests.json` |
| `run_review` | stamps `review-N.json` |
| `spawn` | refuses to start when the gate is not deployed |
| `supervise` | the "did it settle?" guard reads `gate.json`, not `done.json` |
| `inner`, soul-seal failure | settles through the gate with `--souls-ok 0` |
| `inner`, tail | the inline truth table and `printf … done.json` become one gate call |

## Compatibility

`done.json` keeps exactly the key set it has today — `outcome`, `claude_exit`,
`commit`, `branch`, `review_verdict`, `review_rounds`, `findings`, `effort` —
so the Mac bridge needs no change. Existing verdict strings are reused where
they already fit (`PASS`, `NEEDS_CHANGES`, `REVIEW_ERROR`, `NOT_REVIEWED`,
`PASS_WITH_FAILING_TESTS`, `PASS_WITHOUT_TEST_EVIDENCE`,
`TESTER_PROCESS_FAILED`, `REVIEWER_PROCESS_FAILED`,
`CORRECTION_PROCESS_FAILED`, `REVIEWER_PROMPTS_MISSING`, `KAT_ENDPOINT_DOWN`).
New strings appear only for states the runner could not previously express:
`EVIDENCE_MISSING`, `EVIDENCE_MALFORMED`, `EVIDENCE_NOT_IDENTITY_BOUND`,
`EVIDENCE_WRONG_JOB`, `EVIDENCE_STALE_ATTEMPT`, `EVIDENCE_STALE_COMMIT`,
`EVIDENCE_UNCLASSIFIED_FAILURE`, `PASS_WITH_CLASSIFIED_FAILURES`,
`NO_CANDIDATE_COMMIT`. All of them carry `outcome: needs_attention`, which the
bridge already renders as a failed attempt.

Two new files land in the job directory beside `done.json`: `gate.json` (the
full decision, and the marker `supervise` uses to tell a settled build from a
directory something else wrote a file into) and `action-packet.md` (the
plain-English packet).

Historical job records are not touched, read, or rewritten by any of this.

## Known gaps, stated plainly

- **VULCAN does not emit `classification` / `note` yet.** Until
  `soul-vulcan.md` asks for them, any job with a red check settles
  `EVIDENCE_UNCLASSIFIED_FAILURE` rather than
  `PASS_WITH_CLASSIFIED_FAILURES`. Both park, so the failure direction is safe,
  but the reason will be less useful than it should be. Update the soul in the
  same deploy.
- **The gate trusts the runner's stamp.** A stamp is only as good as the
  process that writes it; anything that can edit `~/.local/bin` can defeat this
  layer, exactly as it can defeat the soul seal today.
- **The deployed gate is not hash-pinned** the way the souls are. The runner
  itself has never been either.
- **Nothing here verifies merge, deploy, or live behaviour.** The packet says
  so on its own line, every time.
