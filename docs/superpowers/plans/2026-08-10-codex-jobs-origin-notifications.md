# Codex Jobs and Exact-Origin Notifications Implementation Plan

> **Required execution skills:** Use `superpowers:test-driven-development` for every behavior and `superpowers:subagent-driven-development` with the assigned AI engineer. Do not write production code until the named test has been run and failed for the expected missing behavior.

**Goal:** Ship one immutable Hermes Jobs runtime where explicit Claude/Codex intake invokes only the selected provider and every durable phase update returns exactly once, in order, to the originating Hermes session.

**Architecture:** Keep the existing Jobs SQLite ledger, signed graph, lane policy, and evidence gates. Add typed Job identity, an explicit executor registry, a real Codex adapter, a canonical dispatcher entry point, and a transition-coupled notification outbox. Port the useful exact-origin session primitives from `fix/jobs-wake-watcher`, but replace its terminal-only sidecar with the Jobs outbox. Package tool, runtime, watcher, and policies in one release; use a release-parity gate before activation.

**Stack:** Python 3, SQLite, Hermes plugin API, gateway session store, `codex exec`, Claude CLI adapter, Git worktrees, Tailscale SSH, canonical `scripts/run_tests.sh` runner.

**Design:** `docs/superpowers/specs/2026-08-10-codex-jobs-origin-notifications-design.md`

**Worktree:** `/Users/brandon/.hermes/hermes-agent/.worktrees/full-rollout-20260809`

## Execution Rules

- Work only in the rollout worktree. Do not edit `/Users/brandon/.hermes/hermes-agent` or `/Users/brandon/.hermes/plugins/jobs` during implementation.
- Other work exists in shared checkouts. Do not revert or overwrite it.
- Use `apply_patch` for file edits.
- Every task follows RED, GREEN, REFACTOR and ends with a focused commit.
- Run tests only through `scripts/run_tests.sh`; never invoke `pytest` directly.
- Every test sets a temporary `HERMES_HOME`; no test opens the live Jobs DB or live sessions.
- Prefer behavior and invariant tests. Do not assert against source text.
- Keep the live Jobs dispatcher/progress cron entries paused until Task 11.
- Never default an unknown route to Claude.
- Do not rerun or alter the completed Ponytail audit in this implementation.

## Task 1: Canonical Job Identity and Additive Migration

**Files:**

- Create: `hermes_cli/jobs_identity.py`
- Modify: `hermes_cli/jobs_db.py`
- Modify: `hermes_cli/jobs_exec.py`
- Modify: `hermes_cli/jobs.py`
- Modify: `tests/hermes_cli/test_jobs_live_schema_compat.py`
- Create: `tests/hermes_cli/test_jobs_identity.py`
- Modify: `tests/hermes_cli/test_jobs_exec.py`
- Modify: `tests/hermes_cli/test_jobs_cli.py`

### Step 1: Write failing identity tests

Add tests that specify the only accepted new-work mappings:

```python
assert resolve_requested_lane("claude") == JobIdentity(
    requested_lane="claude",
    executor="claude",
    specialist="claude-builder",
    model="claude-opus-5",
)
assert resolve_requested_lane("codex") == JobIdentity(
    requested_lane="codex",
    executor="codex",
    specialist="codex-builder",
    model="gpt-5.6-sol",
)
for value in (None, "", "kat", "kat-builder", "gpt", "unknown"):
    with pytest.raises(UnsupportedJobLane):
        resolve_requested_lane(value)
```

Add an existing-row compatibility test where `gpt-builder` normalizes to canonical Codex identity, while a historical `kat-builder` Job/attempt/receipt remains unchanged and readable. Add a database test proving a new Job stores `requested_lane`, `executor`, `specialist`, and `model` in its creation transaction.

Run:

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_identity.py tests/hermes_cli/test_jobs_live_schema_compat.py tests/hermes_cli/test_jobs_exec.py -q
```

Expected RED: import of `hermes_cli.jobs_identity` fails, identity columns are absent, and current GPT refusal assertions contradict the new contract.

### Step 2: Implement the smallest identity layer

Create a frozen `JobIdentity` value object and `resolve_requested_lane`. Add nullable identity columns through the existing additive schema initializer. Extend `Job`, row conversion, `create_job`, and idempotent intake to persist identity.

Add `effective_identity(job)` for recognized historical rows:

- `claude-builder` becomes Claude/Opus when identity columns are empty;
- `gpt-builder` becomes the migration alias for Codex/Sol;
- `codex-builder` becomes Codex/Sol;
- `kat-builder` remains historical and non-executable for new work;
- unknown/ambiguous values raise instead of defaulting.

Update `jobs_exec.route` so `codex-builder` is canonical and `gpt-builder` is a legacy alias. Remove the intentional “GPT has no adapter” routing decision, but do not invoke an adapter yet.

Change `hermes jobs create` to require `--lane {claude,codex}` instead of unrestricted `--specialist`. Preserve `--specialist` only on read/maintenance commands needed for historical records; new creation and reassignment reject KAT.

### Step 3: Run GREEN and regressions

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_identity.py tests/hermes_cli/test_jobs_live_schema_compat.py tests/hermes_cli/test_jobs_exec.py tests/hermes_cli/test_jobs_cli.py -q
```

Expected GREEN: canonical values persist; live-shaped DB initialization is additive; historical KAT records are byte-for-byte preserved; unknown/KAT new intake fails.

### Step 4: Refactor and commit

Keep provider strings in `jobs_identity.py`; other modules import constants/value objects rather than duplicating maps.

```bash
git add hermes_cli/jobs_identity.py hermes_cli/jobs_db.py hermes_cli/jobs_exec.py hermes_cli/jobs.py tests/hermes_cli/test_jobs_identity.py tests/hermes_cli/test_jobs_live_schema_compat.py tests/hermes_cli/test_jobs_exec.py tests/hermes_cli/test_jobs_cli.py
git commit -m "feat(jobs): persist canonical executor identity"
```

## Task 2: Versioned `jobs_create` Plugin and Exact-Origin Intake

**Files:**

- Create: `hermes_cli/jobs_tool.py`
- Create: `plugins/jobs/__init__.py`
- Create: `plugins/jobs/plugin.yaml`
- Create: `tests/plugins/test_jobs_tool.py`
- Modify: `tests/hermes_cli/test_plugins.py`

### Step 1: Write failing tool-contract tests

Use a temporary repository and Jobs DB. Register the bundled plugin through the real plugin manager and inspect the registered tool schema. Assert:

```python
assert lane_schema["enum"] == ["claude", "codex"]
assert "lane" in create_schema["parameters"]["required"]
assert "kat" not in json.dumps(create_schema).lower()
```

Call the real handler with injected request origin and assert the Job, identity, origin, and creation event commit together. Call with a missing origin and assert an error response plus zero new Jobs. Call with `lane="codex"` and assert `gpt-5.6-sol`; assert no `gpt-5.6-terra` is written. Task 6 extends this same test to require the queued outbox row once the outbox exists.

Run:

```bash
scripts/run_tests.sh tests/plugins/test_jobs_tool.py tests/hermes_cli/test_plugins.py -q
```

Expected RED: bundled Jobs plugin and `jobs_tool` do not exist.

### Step 2: Implement a lazy bundled plugin

Move schema and handler logic into `hermes_cli/jobs_tool.py`. Keep `plugins/jobs/__init__.py` as a lazy registration shim. The handler must:

1. validate required fields and lane;
2. resolve repository HEAD without a shell;
3. capture request-scoped origin before opening the write transaction;
4. fail without creating a Job when exact origin is missing;
5. call the typed database intake once with canonical identity;
6. return Job number, requested lane, executor, model, base commit, and notification promise.

Do not swallow origin-capture exceptions. Return a bounded error class without credentials or stack traces.

### Step 3: Run GREEN and plugin regressions

```bash
scripts/run_tests.sh tests/plugins/test_jobs_tool.py tests/hermes_cli/test_plugins.py tests/hermes_cli/test_plugin_cli_registration.py -q
```

### Step 4: Commit

```bash
git add hermes_cli/jobs_tool.py plugins/jobs tests/plugins/test_jobs_tool.py tests/hermes_cli/test_plugins.py
git commit -m "feat(jobs): ship versioned Claude and Codex intake"
```

## Task 3: Provider-Neutral Execution Contract and Fail-Closed Registry

**Files:**

- Create: `hermes_cli/jobs_execution.py`
- Create: `hermes_cli/jobs_executors.py`
- Modify: `hermes_cli/jobs_adapter_claude.py`
- Modify: `hermes_cli/jobs_dispatch.py`
- Modify: `hermes_cli/jobs_run.py`
- Create: `tests/hermes_cli/test_jobs_executors.py`
- Modify: `tests/hermes_cli/test_jobs_dispatch.py`
- Modify: `tests/hermes_cli/test_jobs_run_once.py`

### Step 1: Write failing registry tests

Specify a registry with no fallback:

```python
assert registry.require("claude").name == "claude"
assert registry.require("codex").name == "codex"
with pytest.raises(UnsupportedExecutor):
    registry.require("kat")
with pytest.raises(UnsupportedExecutor):
    registry.require("missing")
```

Inject spies and prove a Codex context cannot call the Claude callable and a Claude context cannot call Codex. Add mismatch tests for executor, model, specialist, and selected lane.

Run:

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_executors.py tests/hermes_cli/test_jobs_dispatch.py tests/hermes_cli/test_jobs_run_once.py -q
```

Expected RED: provider-neutral module and explicit registry are missing; `jobs_run` imports Claude as the only adapter.

### Step 2: Extract neutral types and add explicit selection

Move `ArtifactClaim`, `ReliabilityExecution`, and bounded artifact helpers from the Claude module into `jobs_execution.py`. Keep import-compatible re-exports if existing callers need them.

Create an immutable adapter registry keyed only by `claude` and `codex`. `require()` raises a typed exception for every unknown key. Update dispatch/run seams to receive or resolve an adapter by the persisted executor. Delete provider defaults and any `LANES.get(value, "claude")` behavior from active Python paths.

### Step 3: Run GREEN and commit

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_executors.py tests/hermes_cli/test_jobs_dispatch.py tests/hermes_cli/test_jobs_run_once.py tests/hermes_cli/test_jobs_adapter_claude.py -q
git add hermes_cli/jobs_execution.py hermes_cli/jobs_executors.py hermes_cli/jobs_adapter_claude.py hermes_cli/jobs_dispatch.py hermes_cli/jobs_run.py tests/hermes_cli/test_jobs_executors.py tests/hermes_cli/test_jobs_dispatch.py tests/hermes_cli/test_jobs_run_once.py
git commit -m "refactor(jobs): select executors without provider fallback"
```

## Task 4: Real Codex CLI Adapter

**Files:**

- Create: `hermes_cli/jobs_adapter_codex.py`
- Create: `hermes_cli/data/jobs-codex-result.v1.schema.json`
- Create: `tests/hermes_cli/test_jobs_adapter_codex.py`
- Modify: `tests/hermes_cli/test_jobs_executors.py`

### Step 1: Write failing adapter tests

Inject the subprocess boundary; use a real temporary Git repository/worktree. Assert the exact launch contract includes:

```text
codex exec --model gpt-5.6-sol --sandbox workspace-write --cd <attempt-worktree> --json --output-schema <release-schema> --output-last-message <attempt-file> -
```

Assert prompt bytes arrive on stdin, `CODEX_HOME` equals the selected lane auth root, `HOME`/`HERMES_HOME`/`TMPDIR` are private attempt directories, and neither argv nor environment contains `claude`, Claude auth, the claim token, or the live Jobs DB path.

Cover successful commit/evidence readback, wrong model, wrong executor, missing binary, auth failure, timeout, cancellation, nonzero exit, malformed JSONL, oversized output, unsafe result path, missing test artifact, branch escape, and no candidate commit.

Run:

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_adapter_codex.py -q
```

Expected RED: `jobs_adapter_codex` is missing.

### Step 2: Implement bounded Codex execution

Implement `preflight_codex` and `run_codex_attempt` against the neutral contract. Build argv as a list, pass the complete prompt on stdin, set `start_new_session=True`, enforce wall clock with TERM then KILL, cap/redact preserved output, and inspect Git/artifact bytes independently.

The output schema permits only the bounded result fields needed by the evidence pipeline. The adapter records actual Codex version, requested model, lane, process exit digest, output capture digest, candidate commit, and claimed artifact digests. It never accepts a self-reported provider/model as authority.

### Step 3: Run GREEN and commit

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_adapter_codex.py tests/hermes_cli/test_jobs_executors.py -q
git add hermes_cli/jobs_adapter_codex.py hermes_cli/data/jobs-codex-result.v1.schema.json tests/hermes_cli/test_jobs_adapter_codex.py tests/hermes_cli/test_jobs_executors.py
git commit -m "feat(jobs): execute Codex jobs with lane-scoped auth"
```

## Task 5: Canonical Lane-Aware Dispatcher Runtime

**Files:**

- Create: `hermes_cli/jobs_runtime.py`
- Modify: `hermes_cli/jobs.py`
- Modify: `hermes_cli/jobs_dispatch.py`
- Modify: `hermes_cli/jobs_model_policy.py`
- Modify: `scripts/jobs_lane_health.py`
- Create: `scripts/jobs-dispatch-once.sh`
- Create: `tests/hermes_cli/test_jobs_runtime.py`
- Create: `tests/hermes_cli/test_jobs_codex_dispatch_integration.py`
- Modify: `tests/hermes_cli/test_jobs_reliability_e2e.py`

### Step 1: Write failing runtime integration tests

Use a real temporary repository, Jobs DB, versioned model policy, 12-seat lane registry, fake health probes, real dispatcher, and fake external provider process. Assert:

- Codex chooses a healthy `codex-pc-*` seat first;
- eligible PC failure falls back only to `codex-mac-*` with the same model;
- no Codex seat queues/blocks visibly rather than invoking Claude;
- Claude follows the mirror rule;
- persisted Job identity, lane placement, attempt, transitions, and receipt agree;
- legacy header `MODEL` disagreement fails before provider invocation;
- missing adapter and unknown specialist fail closed;
- success, timeout, provider failure, and capacity-full paths settle truthfully.

Run:

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_runtime.py tests/hermes_cli/test_jobs_codex_dispatch_integration.py tests/hermes_cli/test_jobs_reliability_e2e.py -q
```

Expected RED: no production runtime calls the lane-aware reliability dispatcher or Codex adapter.

### Step 2: Implement one production entry point

Add `dispatch_job_once()` in `jobs_runtime.py`. It must load persisted identity, parse only repository/base/effort/turn metadata from the goal, reject identity contradictions, load versioned policy/lanes, probe health, select one lane, construct a `DispatchSpec`, require the exact adapter, call `jobs_dispatch.dispatch_once`, and return a bounded JSON result.

Expose `hermes jobs dispatch-once --job ... --worker-id ... --json`. The release shell wrapper calls only this command; it contains no provider-selection branch. Keep legacy shell runner paused and remove it from activation instructions.

### Step 3: Run GREEN and commit

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_runtime.py tests/hermes_cli/test_jobs_codex_dispatch_integration.py tests/hermes_cli/test_jobs_reliability_e2e.py tests/hermes_cli/test_jobs_lanes.py tests/hermes_cli/test_jobs_model_policy.py -q
git add hermes_cli/jobs_runtime.py hermes_cli/jobs.py hermes_cli/jobs_dispatch.py hermes_cli/jobs_model_policy.py scripts/jobs_lane_health.py scripts/jobs-dispatch-once.sh tests/hermes_cli/test_jobs_runtime.py tests/hermes_cli/test_jobs_codex_dispatch_integration.py tests/hermes_cli/test_jobs_reliability_e2e.py
git commit -m "feat(jobs): dispatch provider-true work across twelve lanes"
```

## Task 6: Atomic Notification Outbox

**Files:**

- Create: `hermes_cli/jobs_notifications.py`
- Modify: `hermes_cli/jobs_db.py`
- Modify: `hermes_cli/jobs_graph.py`
- Create: `tests/hermes_cli/test_jobs_notification_outbox.py`
- Modify: `tests/hermes_cli/test_jobs_graph_db.py`
- Modify: `tests/plugins/test_jobs_tool.py`

### Step 1: Write failing outbox tests

Use temporary DBs and the real graph writer. Assert:

- Job plus exact origin creates one `queued` row in the same transaction;
- the real `jobs_create` handler returns only after that queued row is durable;
- originless operator Job creates none;
- `ASSIGNED`, `BUILDING`, `EVIDENCE_COLLECTING`, `REVIEWING`, `BLOCKED`, terminal `FAILED`, and `COMPLETED` create the specified milestones;
- nonterminal `FAILED` creates `correcting`, and a later `REVIEWING` payload renders re-review;
- repeated graph idempotency key creates no second notification;
- forced outbox insert failure rolls back the graph transition and Job revision;
- claim expiry, retry scheduling, acknowledgement, and ordering behave across two SQLite connections;
- a later row for one Job cannot be claimed while its earlier row is pending;
- payload and error fields enforce size/secret bounds.

Run:

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_notification_outbox.py tests/hermes_cli/test_jobs_graph_db.py -q
```

Expected RED: notification table/API do not exist.

### Step 2: Implement transaction-coupled outbox

Add additive tables/indexes for immutable notification intent and delivery state. Implement locked enqueue helpers called from `_create_job_locked` and `record_transition` before their transaction exits. Stable IDs use the Job ID, resulting revision, and milestone. Snapshot or immutably reference the exact `job_origins` row.

Add APIs to list/claim due rows, renew/release claims, acknowledge delivered/already-present rows, schedule bounded retry, surface blocked-origin rows, and query health. Claims use leases and compare-and-set updates.

### Step 3: Run GREEN and commit

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_notification_outbox.py tests/hermes_cli/test_jobs_graph_db.py tests/hermes_cli/test_jobs_db.py -q
git add hermes_cli/jobs_notifications.py hermes_cli/jobs_db.py hermes_cli/jobs_graph.py tests/hermes_cli/test_jobs_notification_outbox.py tests/hermes_cli/test_jobs_graph_db.py tests/plugins/test_jobs_tool.py
git commit -m "feat(jobs): persist ordered chat updates with transitions"
```

## Task 7: Exact-Origin Gateway Delivery

**Reference:** Read `bf757ec5b` from `/Users/brandon/.hermes/hermes-agent/.worktrees/jobs-wake-watcher` for existing exact-session, compression-continuation, and gateway lifecycle behavior. Port only needed behavior with `apply_patch`; do not cherry-pick unrelated API/gateway changes or reuse its separate `jobs_wake.db` sidecar.

**Files:**

- Create: `gateway/jobs_notifications.py`
- Modify: `gateway/run.py`
- Modify: `gateway/wake.py`
- Modify: `gateway/session.py`
- Modify: `gateway/platforms/base.py`
- Modify: `gateway/platforms/api_server.py`
- Create: `tests/gateway/test_jobs_notification_delivery.py`
- Modify: `tests/gateway/test_wake_delivery.py`
- Modify: `tests/gateway/test_api_server.py`

### Step 1: Write failing real-store delivery tests

Build real temporary Jobs and Hermes state/session databases. Create an origin session and a decoy. Insert ordered outbox rows and run the real worker. Assert:

- all messages appear only in the exact origin session;
- stable `notification_id` prevents duplicate visible append after crash/restart;
- unique compression continuation receives the message;
- zero or multiple continuations leave it pending;
- archived, unavailable, and busy sessions leave it pending;
- messaging origin retains exact conversation/thread/session identifiers;
- no path writes Build Feed, home channel, recent session, or decoy;
- unknown platform becomes visible blocked delivery state rather than silently delivered/skipped.

Run:

```bash
scripts/run_tests.sh tests/gateway/test_jobs_notification_delivery.py tests/gateway/test_wake_delivery.py tests/gateway/test_api_server.py -q
```

Expected RED: release branch has no Jobs notification worker or exact-origin append API.

### Step 2: Port origin primitives and consume the outbox

Create a small gateway worker that claims one due row, resolves only its stored origin or unique compression tip, and appends a structured system message with `notification_id` as destination idempotency metadata. A destination lookup returning unavailable/ambiguous releases the claim with retry; it never selects a fallback.

Wire worker start/stop into `GatewayRunner` with bounded polling and clean shutdown. Capture request-scoped origin in a small reusable helper that `jobs_tool` can call. Keep the worker passive: it does not invoke an LLM turn.

### Step 3: Run GREEN and commit

```bash
scripts/run_tests.sh tests/gateway/test_jobs_notification_delivery.py tests/gateway/test_wake_delivery.py tests/gateway/test_api_server.py tests/gateway/test_shutdown.py -q
git add gateway/jobs_notifications.py gateway/run.py gateway/wake.py gateway/session.py gateway/platforms/base.py gateway/platforms/api_server.py tests/gateway/test_jobs_notification_delivery.py tests/gateway/test_wake_delivery.py tests/gateway/test_api_server.py
git commit -m "feat(gateway): deliver Jobs updates to exact origin"
```

## Task 8: Message Rendering and Ten-Minute Heartbeat

**Files:**

- Modify: `hermes_cli/jobs_notifications.py`
- Modify: `gateway/jobs_notifications.py`
- Create: `tests/hermes_cli/test_jobs_notification_messages.py`
- Create: `tests/gateway/test_jobs_notification_heartbeat.py`

### Step 1: Write failing clock-controlled tests

Assert concise deterministic messages for queued, assigned, building, testing, review, re-review, correcting, needs-you, failure, and finished. Ensure raw goal, stdout, stack trace, tokens, and full review text never appear.

With a fake clock:

- no heartbeat at 599 seconds;
- exactly one heartbeat at 600 seconds without phase change;
- none at 1,200 seconds in the same stagnant episode;
- a phase change resets eligibility;
- pending earlier milestone prevents heartbeat overtaking it;
- restart does not create a duplicate heartbeat.

Run:

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_notification_messages.py tests/gateway/test_jobs_notification_heartbeat.py -q
```

Expected RED: renderer/heartbeat function is absent.

### Step 2: Implement and commit

Make rendering a pure function over the bounded outbox payload. Add `enqueue_due_heartbeats(conn, now=...)` with a unique phase-episode key. Invoke it from the gateway worker before claiming due rows.

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_notification_messages.py tests/gateway/test_jobs_notification_heartbeat.py tests/hermes_cli/test_jobs_notification_outbox.py -q
git add hermes_cli/jobs_notifications.py gateway/jobs_notifications.py tests/hermes_cli/test_jobs_notification_messages.py tests/gateway/test_jobs_notification_heartbeat.py
git commit -m "feat(jobs): report phases and one stalled heartbeat"
```

## Task 9: Whole Workflow Integration

**Files:**

- Create: `tests/integration/test_jobs_create_codex_to_origin_chat.py`
- Create: `tests/integration/test_jobs_provider_isolation.py`
- Modify: `tests/hermes_cli/test_jobs_reliability_e2e.py`

### Step 1: Write the failing end-to-end test

Use a temporary Hermes home, temporary Git repository, real plugin handler, real Jobs DB, real lane/model policies, real dispatcher/runtime, real test/review graph, and real gateway notification worker. Fake only the external Codex/Claude process boundary and signing key material.

The Codex success test must observe this sequence:

```python
assert milestones == [
    "queued", "assigned", "building", "testing", "review", "finished"
]
assert decoy_messages == []
assert every_receipt_executor == "codex"
assert every_receipt_model == "gpt-5.6-sol"
assert selected_lane.startswith("codex-")
assert claude_process_calls == []
```

Add failure/retry/review-correction, gateway restart, worker restart, capacity fallback, archived-origin retry, and compression-continuation cases. Mirror provider isolation for Claude.

Run:

```bash
scripts/run_tests.sh tests/integration/test_jobs_create_codex_to_origin_chat.py tests/integration/test_jobs_provider_isolation.py -q
```

Expected RED: at least one integration seam remains unwired.

### Step 2: Make only integration fixes

Patch imports/wiring/configuration only. If the test exposes missing behavior, return to the owning unit test first, make it RED, implement there, then rerun integration.

### Step 3: Commit

```bash
scripts/run_tests.sh tests/integration/test_jobs_create_codex_to_origin_chat.py tests/integration/test_jobs_provider_isolation.py tests/hermes_cli/test_jobs_reliability_e2e.py -q
git add tests/integration/test_jobs_create_codex_to_origin_chat.py tests/integration/test_jobs_provider_isolation.py tests/hermes_cli/test_jobs_reliability_e2e.py
git commit -m "test(jobs): prove provider and exact-chat workflow"
```

## Task 10: Immutable Runtime Packaging, Parity, and Rollback

**Files:**

- Create: `hermes_cli/jobs_release.py`
- Create: `scripts/jobs-release-activate.py`
- Create: `scripts/jobs-runtime-parity.py`
- Create: `scripts/jobs-release-rollback.py`
- Create: `tests/hermes_cli/test_jobs_release.py`
- Create: `tests/scripts/test_jobs_release_activation.py`
- Modify: `scripts/jobs_evidence_gate_activation.md`

### Step 1: Write failing filesystem-level tests

Against temporary Mac/PC roots, assert activation:

- refuses a dirty or uncommitted source;
- creates a content-addressed immutable release;
- backs up plugin, cron, wrappers, gateway launch config, and DB before mutation;
- installs the versioned Jobs plugin from the release;
- writes launch/wrapper paths containing the exact release SHA;
- keeps legacy runner/progress watchers paused;
- enables only canonical dispatcher/outbox worker after parity passes;
- refuses when any installed runtime hash differs;
- rolls back only the exact recorded targets and retains evidence.

Run:

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_release.py tests/scripts/test_jobs_release_activation.py -q
```

Expected RED: release/parity modules do not exist and current activation runbook permits manual skew.

### Step 2: Implement safe activation primitives

Use explicit resolved paths, atomic rename/copy, JSON manifest with SHA-256 for plugin/runtime/policy files, and a signed activation receipt. Never use broad globs or destructive recursive targets. PC installation uses the configured Tailscale SSH alias and the same release SHA; credentials remain in lane auth roots, not the release.

Update the runbook to reject skipped critical tests and document exact rollback commands produced by the activation tool.

### Step 3: Run GREEN and commit

```bash
scripts/run_tests.sh tests/hermes_cli/test_jobs_release.py tests/scripts/test_jobs_release_activation.py -q
git add hermes_cli/jobs_release.py scripts/jobs-release-activate.py scripts/jobs-runtime-parity.py scripts/jobs-release-rollback.py tests/hermes_cli/test_jobs_release.py tests/scripts/test_jobs_release_activation.py scripts/jobs_evidence_gate_activation.md
git commit -m "feat(jobs): activate one parity-checked runtime"
```

## Task 11: Verification, Immutable Activation, and Live Canary

### Step 1: Run all focused Jobs/gateway suites with retries disabled

```bash
HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh \
  tests/hermes_cli/test_jobs_identity.py \
  tests/plugins/test_jobs_tool.py \
  tests/hermes_cli/test_jobs_executors.py \
  tests/hermes_cli/test_jobs_adapter_codex.py \
  tests/hermes_cli/test_jobs_runtime.py \
  tests/hermes_cli/test_jobs_codex_dispatch_integration.py \
  tests/hermes_cli/test_jobs_notification_outbox.py \
  tests/hermes_cli/test_jobs_notification_messages.py \
  tests/gateway/test_jobs_notification_delivery.py \
  tests/gateway/test_jobs_notification_heartbeat.py \
  tests/integration/test_jobs_create_codex_to_origin_chat.py \
  tests/integration/test_jobs_provider_isolation.py -q
```

Require zero failures, retries, and skipped critical tests.

### Step 2: Run the canonical full suite

```bash
HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh
```

Record pass/fail counts, elapsed time, and exact HEAD. Do not activate on any failure.

### Step 3: Request the test-engineer final audit

Provide design, plan, full diff, focused results, and full-suite result. Fix every critical/high finding using a fresh RED test. Re-run affected suites and the full suite.

### Step 4: Commit verification-only fixes and push

```bash
git status --short
git log --oneline --decorate -12
git push brandon codex/hermes-full-rollout-20260809
```

### Step 5: Create and activate the immutable release

Run the tested activation command in dry-run mode, inspect exact targets, then activate Mac and Tailscale PC with the same commit SHA. Run runtime parity before resuming the canonical dispatcher/outbox worker. Preserve generated backup and rollback receipt paths.

### Step 6: Run one bounded real Codex canary

Create a disposable Git repository, a dedicated origin Hermes session, and a decoy session. Through the installed `jobs_create`, submit one `lane="codex"` Job capped at ten minutes and a fixed turn budget. The Job makes one deterministic file change and runs one deterministic test.

Verify from durable evidence:

- actual process is Codex; no Claude invocation;
- executor `codex`, model `gpt-5.6-sol`, lane `codex-pc-*` or allowed `codex-mac-*` fallback;
- signed transition/receipt chain passes;
- origin receives ordered queued, assigned, building, testing, review, and finished exactly once;
- decoy and Build Feed receive zero;
- no pending/blocked outbox rows remain for the canary;
- installed runtime parity still passes after completion.

If the canary fails, pause intake/dispatcher, preserve evidence, run the recorded rollback, and report the exact failed invariant. Do not retry provider work automatically.

### Step 7: Final rollout receipt

Write a concise release receipt containing commit/release SHA, test counts, audit result, Mac/PC parity hashes, canary Job/attempt/receipt IDs, milestone IDs, latency, backup path, and rollback command. Exclude credentials, raw logs, and full prompts.
