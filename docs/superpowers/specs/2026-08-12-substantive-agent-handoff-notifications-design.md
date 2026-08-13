# Hermes Jobs: Substantive Agent Handoffs in the Origin Chat

**Date:** 2026-08-12

**Status:** Design approved; pending written-spec review

**Owner:** Hermes Jobs reliability workflow

**Release branch:** `codex/kanban-auto-updates-worker-preflight-20260812`

## Outcome

Every meaningful Hermes Job handoff must return a useful, conversational update to the exact chat that created the Job. The message must explain who acted, what happened, what evidence supports the result, who owns the next step, and—when Brandon must decide—exactly what decision is needed.

Agents remain responsible for the substance of their own handoffs. One durable Hermes watcher remains responsible for reliable chat delivery. Generic labels such as “Building,” “Correcting,” or “Needs you” may remain as audit metadata, but they must not be the primary visible update.

## User Experience Contract

The originating Hermes chat receives a normal message after each material handoff. Each message is labeled with the speaking agent or role and reads like that agent reporting back, not like a status badge.

Examples:

> **Claude Builder → Codex Tester**
>
> I finished the mobile card-width fix. I changed the dashboard container and added a 390 px regression test. Typecheck and the focused UI suite pass. Codex is now testing the real drag journey.

> **Codex Tester → Independent Reviewer**
>
> The focused suite passed: 18 tests, 0 failures. I also verified the card stays inside a 390 px viewport while dragging. The exact commit tested was `abc1234`. I sent that evidence to independent review.

> **Independent Reviewer → Claude Builder**
>
> I rejected this build because the card still overlaps the filter bar at 320 px. Typecheck passed, but the critical user journey failed. Claude is rebuilding the responsive spacing and must return with new visual evidence.

> **Hermes needs your decision**
>
> Supplier-document upload is ready, but production cannot activate until R2 credentials are added. Choose: (1) add the production credentials now, or (2) leave upload disabled and ship the rest. I recommend option 1 because option 2 leaves the new intake flow unavailable. Nothing has been deployed, and the Job is safely blocked.

The visible chat must not contain repeated sequences such as “Assigned,” “Building,” and “Correcting” with no explanation. A watcher heartbeat is allowed only when it contains new, worker-supplied progress or a newly detected problem. Elapsed time alone is not a substantive update.

## Scope

This design covers:

- substantive updates for builder, tester, reviewer, correction, decision, activation, and completion handoffs;
- one structured handoff receipt shared between agents;
- durable, exact-origin delivery by the existing Hermes notification watcher;
- explicit decision requests with options, recommendation, impact, and blocked scope;
- ordering, deduplication, restart recovery, bounded message rendering, and secret filtering;
- compatibility with existing Jobs graph states and evidence gates;
- focused automated proof and a bounded installed-runtime chat canary.

This design does not replace the Jobs state machine, weaken evidence or independent-review gates, expose private prompts or raw logs, give locked workers unrestricted chat or vault access, redesign the Hermes chat interface, change provider routing, or create direct worker-to-user network access.

## Chosen Architecture

The system uses a hybrid model:

1. **The acting agent authors the substance.** At a material transition, it returns a structured handoff receipt describing completed work, evidence, verdict, issues, and next action.
2. **Hermes validates and stores the receipt.** A graph transition that requires a visible update cannot silently discard an invalid or missing handoff.
3. **The existing watcher delivers it.** The durable outbox sends an agent-labeled conversational message to the exact originating chat, with the same ordering and retry guarantees already used for lifecycle notifications.
4. **The next agent receives the same bounded receipt.** This lets builders, testers, and reviewers communicate through durable evidence without giving every worker direct access to the full user chat or shared brain.

This preserves the agent’s voice while keeping delivery independent of a worker process. A worker can crash after committing its handoff and Hermes can still deliver it. A worker cannot claim progress merely by keeping a process alive.

## Components and Responsibilities

### 1. Handoff receipt

A handoff receipt is a versioned, bounded JSON object attached to a graph transition. Its required fields are:

- `schema_version`;
- `job_id`, `attempt_id`, and graph transition/revision identity;
- `speaker_id`, `speaker_role`, and `speaker_executor`;
- `from_phase` and `to_phase`;
- `next_owner_role` when work continues;
- `summary`: what the agent actually did or determined;
- `evidence_summary`: bounded results tied to hashes or artifact references already accepted by the graph;
- `outcome`: one of `started`, `handed_off`, `passed`, `rejected`, `blocked`, `activated`, or `completed`;
- `next_action`: the concrete next step;
- `issues`: bounded findings when work failed, was rejected, or needs correction;
- `artifact_identity`: exact commit, release, deployment, or other artifact under review when applicable;
- creation time and producer identity.

The receipt references durable evidence; it does not embed raw stdout, full prompts, stack traces, credentials, environment dumps, or unrestricted model prose. Field lengths and list counts are capped. Control characters and known secret patterns are rejected or redacted before persistence.

### 2. Decision request

Any transition that asks Brandon to act must also include a `decision_request` object with:

- one concrete question;
- two or three actionable options when alternatives exist;
- the agent’s recommendation and reason;
- the consequence of each option;
- what is blocked and what remains safe or unchanged;
- the next owner after Brandon responds.

“Requires your decision,” by itself, is invalid. If an agent cannot state the question and impact, the Job remains in internal triage and Hermes reports that the handoff is incomplete; it does not send a fake user-decision request.

### 3. Transition validator

The Jobs graph defines which transitions require which receipt fields:

| Transition purpose | Required substance |
|---|---|
| assignment/start | assigned agent, understood goal, first concrete action |
| builder to testing | work completed, artifact identity, checks run, next tester |
| testing to review | test results, critical user journey result, tested artifact, next reviewer |
| review pass | verdict, evidence reviewed, remaining gate/next owner |
| review reject/correction | exact defects, failed requirement or journey, required fix, correction owner |
| re-review | what changed since rejection, new evidence, reviewer destination |
| blocked/needs Brandon | complete decision request and safe current state |
| activation/live verification | deployed artifact, environment, health and user-journey result, rollback readiness |
| completion | verified business outcome, exact live artifact when applicable, unresolved caveats |

A success, handoff, review, decision, activation, or completion transition missing required substance fails atomically: the graph remains at its prior state and no success-like chat message is queued. Existing recovery logic may settle the attempt from independently observed failure evidence, but that produces a Hermes system diagnostic—not invented agent dialogue—and must never be rendered as confident progress.

### 4. Agent handoff context

The next worker receives the validated receipt plus the minimum evidence references needed for its role. It does not rely on the previous worker’s free-form summary alone. Reviewers receive independently captured test and artifact evidence and may return `unable_to_verify`; they do not inherit or reveal sealed review instructions.

Locked builders continue to receive scoped attachments only. This feature does not grant them full vault, profile, credential, or unrelated-chat access.

### 5. Durable notification outbox

The graph transition and its handoff-notification intent commit in the same database transaction. The outbox payload stores the validated receipt or an immutable reference to it. Its stable identity binds:

- Job and attempt;
- graph transition/revision;
- handoff schema version;
- speaker and next owner;
- exact originating chat;
- exact artifact/evidence references.

The watcher claims, renders, and delivers each row in Job order. A retry uses the same stable notification ID. If the chat append succeeds but acknowledgement fails, destination idempotency prevents a duplicate visible message.

### 6. Message renderer

The renderer turns the receipt into a concise conversational message while preserving facts and agent attribution. It may adjust grammar and layout but cannot invent evidence, findings, owners, options, or recommendations.

Visible messages use these headings:

- `<Agent> started`;
- `<Agent> → <Next agent>`;
- `<Reviewer> approved` or `<Reviewer> sent it back`;
- `Hermes needs your decision`;
- `<Agent> verified it live`;
- `Job complete`.

The old milestone is retained as structured metadata for filters, diagnostics, and compatibility. It is not emitted as an additional chat line when a substantive handoff covers the same transition.

### 7. Watcher fallback behavior

The watcher may render only persisted, validated facts. If a legacy or malformed event lacks a usable handoff:

- it emits one explicit bounded warning such as “Hermes could not explain this handoff because the agent did not provide the required receipt”;
- it identifies the Job and current safe state;
- it routes the Job to internal correction or operator triage;
- it never manufactures agent dialogue or marks work complete;
- repeated checks of the same missing receipt do not create repeated chat messages.

## Data Flow

1. A worker performs its scoped action and returns evidence plus a structured handoff.
2. Hermes independently checks the transition evidence and validates the handoff schema.
3. In one transaction, Hermes advances the Job graph, stores the handoff, and inserts the exact-origin notification intent.
4. The next agent receives the bounded handoff and evidence references.
5. The gateway watcher claims the outbox row and renders the agent-labeled message.
6. The watcher appends it to the originating Hermes chat using the stable notification ID.
7. Delivery acknowledgement closes the outbox row; a crash or temporary destination failure resumes from the durable lease without duplication.

The user-facing report therefore follows proven work rather than predicting it. “Claude is testing” is emitted only after testing custody is actually accepted, and “sent back to be fixed” names the reviewer finding and confirmed correction owner.

## Ordering and Noise Rules

1. Handoffs for one Job are delivered in graph order.
2. A later handoff cannot overtake an undelivered earlier handoff.
3. One transition produces at most one primary visible message.
4. Delivery retry never creates a second visible copy.
5. Repeated graph polling, lease renewal, or unchanged phase does not create chat output.
6. A progress heartbeat requires a newly persisted progress fact, changed risk, or changed estimate supplied by the active owner.
7. Repeated correction cycles remain visible, but every cycle must say what failed and what changed; identical correction text is rejected as retry-without-progress.
8. System diagnostics remain available in Jobs status and logs without being copied into the user chat.

## Failure and Recovery

- **Worker crashes before handoff:** no substantive claim is emitted. Hermes reports the verified failure or missing handoff once and follows normal retry policy.
- **Worker returns prose without required fields:** the intended transition is rejected atomically and the Job cannot masquerade as handed off.
- **Evidence and narrative conflict:** evidence wins; the transition is rejected and the discrepancy is recorded for review.
- **Reviewer cannot verify:** the message says exactly what evidence is missing and who must provide it.
- **Origin chat unavailable:** the outbox remains pending. There is no fallback to Build Feed, another session, or the most recent chat.
- **Gateway restarts:** expired delivery leases are reclaimed and stable IDs preserve order and deduplication.
- **Renderer fails:** the receipt remains durable and pending; a bounded renderer error is exposed through Jobs health diagnostics.
- **Decision reply is ambiguous:** the Job remains blocked and asks one clarified question. It does not infer approval.
- **Sensitive content is detected:** the unsafe field is rejected/redacted, the full raw value is not stored in the notification payload, and the producer must submit a safe summary.

## Compatibility and Migration

Existing lifecycle outbox rows remain readable. During the compatibility window:

- a new structured handoff renders as the primary conversational update;
- a legacy event without a handoff renders at most one clearly labeled legacy-status message;
- the watcher never emits both a generic milestone and substantive message for the same transition;
- no historical Job state, origin, evidence, or notification ID is rewritten merely to adopt the new format;
- new Jobs use the handoff schema from their first material transition;
- completion cannot use legacy generic rendering once the feature is activated for that Job.

Activation is release-pinned and reversible. It must not edit live Job records, credentials, profiles, provider routing, or unrelated gateway behavior.

## Automated Proof

Implementation follows test-driven development. Focused tests must prove:

1. Builder, tester, and reviewer handoffs contain the required substantive fields.
2. The exact originating chat receives the rendered message and a decoy chat receives none.
3. Reviewer rejection names the failed requirement and required correction.
4. The correction owner receives the same bounded handoff that the user sees.
5. Re-review states what changed since the prior rejection.
6. A decision request without a question, options/available action, recommendation, impact, blocked scope, or next owner is rejected.
7. `unable_to_verify` remains honest and cannot render as approval.
8. Missing or malformed handoffs fail closed and create one non-spamming diagnostic message.
9. Repeated polling, leases, identical retries, and gateway restarts do not duplicate chat messages.
10. Ordered handoffs survive transient destination failure and restart.
11. Secret-like content and oversized fields are not persisted or displayed.
12. Raw logs, prompts, credentials, and sealed reviewer instructions never appear in the rendered message.
13. Existing legacy outbox rows remain readable without duplicate generic-plus-substantive output.
14. Normal Jobs routing, retry, testing, review, evidence, and completion behavior remains unchanged.

After focused unit and integration suites pass, the affected broad Jobs and gateway suites must pass. The installed immutable candidate then runs a bounded exact-origin canary covering:

`builder handoff → tester handoff → reviewer rejection → corrected build → re-review → verified completion`

The canary must show substantive messages once and in order, zero messages in a decoy chat, exact artifact identity at testing/review/completion, and no generic status spam.

## Acceptance Criteria

The feature is complete only when:

1. Every material handoff produces one substantive, agent-attributed message in the Job’s originating Hermes chat.
2. Each message says what happened, what evidence exists, who owns the next step, and what that step is.
3. Review rejection and correction messages identify the exact defect and required fix.
4. A request for Brandon contains the complete decision question, actionable choices, recommendation, consequences, blocked scope, and safe current state.
5. Generic lifecycle labels are metadata, not repetitive primary chat messages.
6. Agents share the same durable, bounded handoff; reviewers remain independent and can say `unable_to_verify`.
7. Missing evidence or handoff substance cannot produce a false progress, approval, or completion message.
8. Exact-origin delivery is ordered, idempotent, restart-safe, and has no fallback destination.
9. No private prompt, credential, raw log, sealed instruction, or unrestricted memory enters the chat payload.
10. Focused tests, affected broad suites, installed-runtime parity checks, and the bounded live canary all pass before activation.

## Rollback

Rollback restores the prior immutable gateway release and pauses new handoff-schema activation. Durable handoff receipts and outbox rows remain preserved for audit. Rollback must not delete Job history, retry a business action, change provider routing, or deliver pending messages to a different destination. Before reactivation, the candidate must reconcile any pending rows by stable notification ID and prove that no transition will be displayed twice.
