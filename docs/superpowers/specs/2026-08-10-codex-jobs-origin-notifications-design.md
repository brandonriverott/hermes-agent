# Hermes Jobs: Real Codex Routing and Exact-Origin Notifications

**Date:** 2026-08-10  
**Status:** Approved for implementation  
**Owner:** Hermes Jobs rollout  
**Release branch:** `codex/hermes-full-rollout-20260809`

## Outcome

Hermes Jobs must execute explicitly selected Claude work with Claude and explicitly selected Codex work with Codex. A Job created from a Hermes chat must send concise, ordered progress updates back to that exact originating chat from queueing through completion. No provider substitution, unrelated-chat fallback, inferred progress, or mutable out-of-tree runtime is allowed.

The rollout is complete only when tool intake, routing, execution adapters, durable lifecycle/outbox state, gateway delivery, tests, and installed runtime all come from one immutable release and a bounded live Codex canary passes over the configured Tailscale path.

## Scope

This design covers:

- `jobs_create` intake for `claude` and `codex`;
- retirement of KAT for new Jobs while retaining readable history;
- durable executor/model identity from intake through receipts;
- a real Codex CLI adapter;
- PC-first placement with same-executor Mac fallback;
- ordered lifecycle notifications to the exact originating Hermes session;
- one ten-minute stalled-phase heartbeat;
- retry, deduplication, restart recovery, and compression-continuation handling;
- immutable release packaging, activation, rollback, and end-to-end proof.

This design does not redesign Kanban, replace the existing evidence/review gates, send raw worker logs to chats, reroute an unavailable executor to another provider, or revive KAT.

## Current Failure Modes Being Removed

The existing environment has several incompatible implementations:

- the live out-of-tree Jobs plugin offers `codex` but writes `codex-builder` with `gpt-5.6-terra`;
- the immutable release recognizes only legacy `gpt-builder`, deliberately refuses it as non-executable, and has no Codex adapter;
- the legacy PC worker invokes Claude for every non-KAT route;
- lifecycle progress is inferred from remote marker-file modification times;
- terminal and intermediate notifications use separate stores and watchers;
- some notification paths fall back to Build Feed or another session;
- active plugin, gateway watcher, scripts, and release do not share one versioned artifact;
- dispatcher and progress cron entries are paused.

Enabling intake before these conflicts are removed would accept Jobs that cannot truthfully execute as Codex. Activation therefore remains blocked until the complete release passes all gates in this document.

## Canonical Job Identity

One explicit provider identity is persisted at intake and remains invariant for the Job attempt.

| User-facing lane | Persisted executor | Canonical specialist | Default model | Eligible lanes |
|---|---|---|---|---|
| `claude` | `claude` | `claude-builder` | `claude-opus-5` | `claude-pc-*`, then `claude-mac-*` |
| `codex` | `codex` | `codex-builder` | `gpt-5.6-sol` | `codex-pc-*`, then `codex-mac-*` |

Rules:

1. `jobs_create.lane` is required and accepts exactly `claude` or `codex`.
2. Missing or unknown lane input fails closed. There is no implicit Claude default.
3. `executor`, `specialist`, and `model` are resolved before insert and stored on the Job, not reconstructed from free-form goal text.
4. A lane decision must match all three stored values. Any mismatch is a visible routing failure before provider invocation.
5. PC-to-Mac fallback may change only host and seat. It must preserve executor and model.
6. Claude unavailability never invokes Codex, and Codex unavailability never invokes Claude.
7. `gpt-builder` is accepted only as a migration alias for existing Jobs and normalizes to `codex-builder` plus `executor=codex`. New intake never writes it.
8. Existing `kat-builder` rows remain readable and auditable. All new creation and reassignment entry points reject KAT.
9. Receipts record requested lane, resolved executor, specialist, model, physical lane, host, adapter kind, and executor version so provider identity is independently auditable.

## Component Architecture

### 1. In-tree Jobs plugin

The Jobs tool becomes a bundled, versioned plugin under the release source tree. It owns the public tool schema and converts user intent into a typed intake request. The out-of-tree live plugin is replaced during activation, not edited in place as an independent source of truth.

For chat-created Jobs, the handler obtains request-scoped origin context before database mutation. If exact origin cannot be captured, the tool returns an error and creates no Job. Operator CLI creation may remain originless, but originless Jobs do not produce chat notifications.

### 2. Jobs database and graph

The Jobs database remains the sole lifecycle authority. The existing evidence-authorized graph remains canonical:

`QUEUED`, `ASSIGNED`, `BUILDING`, `EVIDENCE_COLLECTING`, `REVIEWING`, `VERIFIED`, then `COMPLETED`, `FAILED`, `BLOCKED`, or `CANCELLED`.

The Job row gains durable requested lane, executor, and model fields. Every lifecycle transition that produces a user-visible milestone inserts its notification-outbox row inside the same SQLite `IMMEDIATE` transaction as the transition. State cannot commit without its notification intent, and notification intent cannot exist for a state that did not commit.

### 3. Executor adapters

Claude and Codex adapters implement one provider-neutral reliability-executor contract. Each adapter:

- receives an immutable attempt context;
- verifies the context executor/model matches itself;
- uses only its own provider binary;
- creates or uses the registered attempt worktree and handoff roots;
- receives a lane-scoped authentication home;
- applies a bounded wall clock and output cap;
- observes Git and test artifacts independently of model claims;
- returns provider-neutral evidence for the existing test and review gates;
- never writes Jobs state directly.

The Codex adapter invokes `codex exec` non-interactively with the persisted `gpt-5.6-sol` model, the approved effort, JSONL output, and the attempt worktree as its working directory. Its environment uses the selected lane's `CODEX_HOME`; it must not invoke `claude`, inherit Claude credentials, or silently use another model.

### 4. Dispatcher

The evidence-authorized dispatcher is the only production dispatch path. It selects a healthy lane from the versioned 12-seat registry, verifies the selected executor/model against the Job identity, claims custody, invokes the matching adapter from an explicit adapter registry, advances the graph from read-back evidence, and releases capacity only after cleanup verification.

No dictionary lookup may default an unknown specialist to Claude. Unknown, missing, or unavailable adapters produce a durable blocked/failure result and a same-chat update.

### 5. Gateway notification worker

The gateway runs one outbox consumer built from the same immutable release. It reads ordered pending rows, routes them only to the stored origin, and appends a structured Hermes system update carrying the stable notification ID.

Legacy terminal-only watcher state, marker-file progress inference, and Build Feed fallback are removed from the active path after migration. They may remain disabled for rollback inspection but cannot consume events concurrently.

## Durable Data Contract

### Job identity fields

The `jobs` table stores:

- `requested_lane`: `claude` or `codex` for new Jobs;
- `executor`: `claude` or `codex`;
- `model`: exact approved model ID;
- existing `specialist`: canonical role name.

For historical rows, migration derives identity only from recognized legacy values. Ambiguous rows remain blocked for operator review; they are never guessed as Claude.

### Origin record

`job_origins` is immutable after creation and stores bounded strings for:

- platform;
- chat ID;
- session ID;
- thread ID where applicable;
- chat type;
- profile;
- user ID where supplied.

Chat tool intake requires platform, chat ID, and session ID. Messaging adapters additionally retain their exact thread/conversation identifiers. Origin values are never replaced with a default feed.

### Notification outbox

Each outbox row contains:

- stable `notification_id` derived from Job, graph revision, and milestone;
- Job ID and attempt ID when present;
- graph transition ID and Job revision;
- immutable origin reference;
- milestone and bounded JSON payload;
- creation time and next-attempt time;
- claim owner and claim expiry;
- delivery-attempt count and last bounded error class;
- delivered time.

Uniqueness on `(job_id, job_revision, milestone)` makes transition retries idempotent. Delivery uses a short lease, so process death returns the row to pending. Later milestones for one Job cannot overtake an earlier undelivered milestone.

Destination append accepts `notification_id` as its idempotency key. A retry that finds that ID already present marks the outbox row delivered without appending a second visible message. This closes the crash window between visible append and outbox acknowledgement for Hermes-owned sessions.

## Milestone Contract

Messages are concise and contain Job number/name, current phase, and only the next useful fact. They do not include raw stdout, prompts, stack traces, credentials, or full review reports.

| Graph event | User milestone | Example meaning |
|---|---|---|
| Job plus origin created | `queued` | Job accepted for the selected provider |
| `ASSIGNED` | `assigned` | Provider, host, and lane selected |
| `BUILDING` | `building` | Worker started implementation |
| `EVIDENCE_COLLECTING` | `testing` | Build finished; tests/evidence are running |
| `REVIEWING` | `review` | Tests passed far enough to enter review |
| Nonterminal attempt `FAILED` | `correcting` | Findings or failed evidence returned before a retry/re-review |
| `BLOCKED` | `needs-you` | Operator decision or external action required |
| Terminal attempt `FAILED` | `failure` | Job ended unsuccessfully with a bounded reason |
| `COMPLETED` | `finished` | Evidence, review, and activation gate passed |
| No phase change for 600 seconds | `heartbeat` | Job is alive in the same phase |

`VERIFIED` is retained in the durable graph but does not need a separate chat message; the next user-relevant update is completion or activation failure. A later `REVIEWING` transition after a correcting milestone is rendered as re-review while retaining the stable `review` milestone type.

One heartbeat is emitted per stagnant phase episode. A phase change resets heartbeat eligibility. Heartbeats do not repeat every ten minutes while the phase remains unchanged.

## Exact-Origin Delivery Rules

1. A notification targets only the origin stored with its Job.
2. Desktop, TUI, and API origins append to the exact session.
3. If that session was compressed, delivery may follow only a unique live compression continuation.
4. Zero or multiple continuation candidates leave the notification pending.
5. Archived, unavailable, or temporarily busy sessions leave the notification pending for retry.
6. Messaging-platform delivery must preserve the stored conversation and thread identifiers rather than reconstructing a broader destination.
7. No failure path may deliver to Build Feed, a home channel, the most recent session, or another chat.
8. Originless operator Jobs have no notification rows and remain visible through Jobs status commands.

## Failure, Retry, and Recovery

- Provider binary missing, auth missing, model mismatch, unsafe lane root, or adapter mismatch fails before provider spend and becomes a visible blocked/failure milestone.
- Provider timeout or nonzero exit becomes durable failed attempt evidence. It cannot be reported as completion.
- Delivery retries use capped exponential backoff with deterministic jitter and keep the same notification ID.
- Retryable destination errors remain pending. Invalid/unknown origin formats become blocked notification records surfaced by health/status commands; they are not silently discarded.
- Gateway restart scans expired delivery claims and resumes in order.
- Dispatcher restart relies on existing custody leases and attempt recovery. It does not reconstruct phase from files.
- Activation refuses if the live plugin, dispatcher, adapter, gateway watcher, and versioned data files do not match the release manifest.

## Migration and Activation

Activation is one reversible cutover:

1. Build and test a new immutable release from this branch.
2. Back up the live Jobs database, plugin directory, cron definition, wrappers, and gateway launch configuration.
3. Run schema migration in a transaction and validate historical rows.
4. Replace the out-of-tree Jobs plugin with the bundled release plugin or a release-pinned wrapper; do not maintain two writable implementations.
5. Point both Mac and Tailscale PC workers at the same release contract and provider-specific adapter entry point.
6. Point the gateway notification worker at the release outbox consumer.
7. Keep legacy runner/progress watchers paused; enable only the canonical dispatcher and outbox worker.
8. Run installed-runtime parity checks before accepting intake.
9. Run one bounded live Codex canary.
10. Enable normal intake only after the canary and exact-origin assertions pass.

Rollback pauses intake and new workers, restores the backed-up launcher/plugin/cron configuration, and restores the database backup only if the migration itself must be reversed. Immutable release directories and canary evidence remain for diagnosis.

## Test and Release Gates

Implementation follows strict RED, GREEN, REFACTOR cycles. Each behavior begins with a test that fails for the missing behavior, then the smallest production change, then the canonical targeted test command.

Required automated proof:

- tool schema exposes only Claude and Codex;
- intake persists canonical identity and exact origin atomically;
- KAT and missing/unknown lanes fail closed;
- Codex route can invoke only the Codex adapter and Claude route only Claude;
- same-executor/model PC-to-Mac fallback works and cross-provider fallback fails;
- every milestone is written atomically, ordered, deduplicated, retryable, and restart-safe;
- exact session and unique compression continuation receive updates; decoy sessions receive none;
- archived/ambiguous origins remain pending with no Build Feed fallback;
- heartbeat fires once after ten stagnant minutes and resets on phase change;
- runtime release hashes match the installed plugin, adapters, dispatcher, gateway worker, and policy files;
- targeted suites pass, then `scripts/run_tests.sh` passes with zero failures, retries, or skipped critical tests.

The bounded live canary submits one deterministic Job through installed `jobs_create` with `lane=codex`, routes it over Tailscale to a real Codex worker, makes one deterministic file change, runs the normal test/review gates, and returns ordered updates to its origin chat. A decoy chat must receive zero messages. Database and signed receipt evidence must show `executor=codex`, `model=gpt-5.6-sol`, a `codex-*` lane, and no Claude process invocation. The canary is capped at one Job, one lane, ten minutes, and a fixed turn budget.

## Acceptance Criteria

The rollout is accepted only when all conditions hold:

1. `jobs_create` offers Claude and Codex, not KAT.
2. One real Codex Job completes through the Codex CLI; queuing it as Claude is neither required nor allowed.
3. Provider identity is unchanged from intake through lane placement and signed receipt.
4. Origin chat receives queued, assigned, building, testing, review, any correction/needs-you/failure event that occurs, and finished, each exactly once and in order.
5. Decoy and Build Feed sessions receive nothing.
6. Restart and unavailable-origin tests retain pending messages without duplication or reroute.
7. Active runtime is entirely release-pinned and parity-verified.
8. Canonical full tests and the bounded live canary pass.
