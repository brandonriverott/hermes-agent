# Activating the Jobs evidence gate

**Status: activation requires three synchronized targets.** This commit adds
the gate, runner patch, bridge patch, and their tests. The evidence gate is not
live until the gate and runner are installed together on every worker host and
the Mac terminal bridge is installed from its mapped patch. A runner-only
activation is insufficient: an unpatched bridge still trusts builder-writable
`done.json` as a terminal success signal.

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

## Deploy steps (all targets, or roll back all changed targets)

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

3. **Patch the Mac terminal bridge:**

   `scripts/jobs_evidence_gate.bridge.patch` is mapped to the installed
   `/Users/brandon/.hermes/scripts/jobs-pc-bridge.sh`. Apply it to a scratch
   copy first, run `bash -n`, then atomically replace the installed bridge. The
   bridge invokes the gate at the absolute installed Mac path because the Jobs
   adapter deliberately runs it under a private `HOME`.

4. **Verify with bridge canaries** before trusting it on real work: one job whose
   tester returns a green receipt should settle `succeeded`; one whose evidence
   is removed before settlement should settle `needs_attention` with
   `EVIDENCE_MISSING`. Also exercise a mismatched identity and a failed required
   check paired with reviewer `PASS`; both must produce failed action results
   with their blocking reason.

The runner patch is verified by
`tests/scripts/test_jobs_evidence_gate_runner.py`, which copies the installed
runner into a scratch directory, applies this patch, and drives a real build
with a stubbed `claude`. The bridge patch is verified by
`tests/scripts/test_jobs_evidence_gate_bridge.py`, which copies the installed
bridge, applies or recognizes the mapped patch, and drives its real local
fallback/action-result seam with synthetic evidence. These tests skip when the
corresponding live source is unavailable; such a skip is an unproven activation
boundary, not a green result.

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
| Mac bridge, local fallback | validates `gate.json`, `done.json`, and the bound receipts before action output |
| Mac bridge, remote poll | waits for the supervisor boundary and `gate.json`; `done.json` alone is ungated |
| Mac bridge, remote readback | copies the complete settlement evidence and validates it locally before fetch/merge |

## Compatibility

`done.json` keeps exactly the key set it has today — `outcome`, `claude_exit`,
`commit`, `branch`, `review_verdict`, `review_rounds`, `findings`, `effort` —
for display and compatibility. It is not settlement authority. The patched Mac
bridge requires identity-bound `gate.json` and independently re-evaluates every
claimed success from the stamped receipts before it can emit `succeeded`.
Existing verdict strings are reused where they already fit (`PASS`,
`NEEDS_CHANGES`, `REVIEW_ERROR`, `NOT_REVIEWED`,
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
- **Bridge verification is read-only.** It does not mint or repair evidence;
  missing or malformed gate output becomes a failed action result with a
  blocking reason. Historical job directories are left untouched.
- **Nothing here verifies merge, deploy, or live behaviour.** The packet says
  so on its own line, every time.

## Release, parity, and rollback (Task 10 tooling)

The Jobs runtime ships as a **versioned immutable bundle**: a content-addressed
directory plus a JSON manifest. This is the only supported way to change the
installed Jobs plugin/runtime/policy set.

Artifact format (`<output-root>/releases/<release-sha>/jobs-release-v1/`):

- `manifest.json` — `schema_version`, `release_sha` (full git HEAD of the
  source tree), `created_at` / `created_by` (creation metadata), `source`
  (absolute source root), `files` (exact list with `relpath`, SHA-256
  `sha256`, `size` per file), `bundle_digest` (stable SHA-256 over release SHA
  plus the canonical file list), and `manifest_digest` (integrity digest over
  metadata and content identity). Rebuilding the same SHA at a later time
  preserves `bundle_digest`;
- `files/<relpath>` — byte-identical copies of every release file.

Refusals (all of them safe, none silent):

- **dirty source tree** — uncommitted changes or untracked files;
- **missing manifest files** — a bundle that does not contain every file its
  manifest lists;
- **digest mismatch** — any bundle file (or installed target) whose SHA-256
  differs from the manifest;
- **target drift** — in the dry-run validation gate, install check, parity,
  and rollback: a target whose bytes differ from the expected state is
  refused, never clobbered;
- **live Hermes root** — refused by default. The only exception is a known
  `mac` or `pc` profile bound to the exact release SHA, stable bundle digest,
  declared versioned target, backup/receipt directories, preflight evidence,
  and literal acknowledgement. There is no generic force flag.

Build, dry-run activate (default; refuses drift, writes nothing):

```sh
python scripts/jobs-release-activate.py \
  --source-root . \
  --output-root ~/jobs-releases \
  --root <mac-root> \
  [--pc-root <pc-root>]
```

Apply to simulated roots (backup + install + receipt; real live activation is
Task 11 only):

```sh
python scripts/jobs-release-activate.py \
  --source-root . \
  --output-root ~/jobs-releases \
  --root <mac-root> --pc-root <pc-root> \
  --apply
```

Prepare one read-only live-profile preflight (run locally on each target):

```sh
python scripts/jobs-release-activate.py \
  --source-root . \
  --output-root <scratch-release-root> \
  --live-profile mac \
  --write-live-preflight <scratch-preflight.json>
```

After an explicit Task 11 GO, apply exactly that preflight. Replace `mac` with
`pc` on the PC; the target is derived as
`~/.hermes/releases/hermes-agent-<full-release-sha>` and cannot be overridden:

```sh
python scripts/jobs-release-activate.py \
  --source-root . \
  --output-root <scratch-release-root> \
  --live-profile mac \
  --expected-release-sha <full-release-sha> \
  --acknowledge-live-activation ACTIVATE-JOBS-LIVE:mac:<full-release-sha> \
  --preflight <scratch-preflight.json> \
  --apply
```

Verify byte-identical install across Mac and PC roots:

```sh
python scripts/jobs-runtime-parity.py \
  --bundle ~/jobs-releases/releases/<release-sha>/jobs-release-v1 \
  --mac-root <mac-root> --pc-root <pc-root>
```

Roll back to the prior recorded state — the exact rollback command is the
receipt path printed by the activation tool:

```sh
python scripts/jobs-release-rollback.py \
  --root <mac-root> \
  --receipt <path printed by activation output>
```

Rollback restores only the targets recorded in the receipt: files that
existed before activation are restored from backup, files that did not are
removed, and rollback evidence is written next to the receipt.

### Activation gate (Task 11, stated now)

Real activation must not proceed when any critical Jobs/gateway test is
skipped — a skip is an unproven activation boundary, not a green result. If a
live canary fails after activation, roll back immediately with the command
above and retain the rollback evidence.
