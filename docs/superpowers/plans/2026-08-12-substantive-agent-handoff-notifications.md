# Substantive Agent Handoff Notifications Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every material Hermes Job handoff produce one evidence-bound, agent-attributed message in the exact originating chat, including complete reviewer corrections and actionable user-decision requests.

**Architecture:** Workers return bounded handoff content in their strict result envelopes; Hermes normalizes and signs that content into each graph transition, stores it atomically with the outbox intent, and passes the same receipt to the next worker. The existing gateway watcher remains the only delivery owner and renders one substantive message per transition, while generic milestones remain metadata and elapsed-time heartbeats stop producing chat spam.

**Tech Stack:** Python 3.11+, SQLite, dataclasses, JSON Schema 2020-12, Ed25519 transition receipts, pytest, Hermes `SessionDB`, immutable Jobs release tooling.

---

## File and Responsibility Map

- Create `hermes_cli/jobs_handoffs.py`: canonical handoff shape, bounds, secret screening, transition-specific validation, deterministic system receipts, and conversational rendering.
- Modify `hermes_cli/data/jobs-build-result.v1.schema.json`: require builder summary and critical-user-journey evidence.
- Modify `hermes_cli/data/jobs-review-result.v1.schema.json`: require reviewer summary and next action while preserving sealed independent review.
- Modify `hermes_cli/jobs_execution.py`: carry validated builder, tester, and reviewer handoffs across the provider-neutral adapter boundary.
- Modify `hermes_cli/jobs_reliability.py`: prompt for handoffs, validate provider output, give the independent reviewer bounded builder evidence, and materialize handoff artifacts.
- Modify `scripts/jobs_remote_worker.py`: preserve handoff fields across the Mac/PC transport without exposing logs or credentials.
- Modify `hermes_cli/jobs_graph.py`: bind the handoff to the signed transition request and reject missing or contradictory handoffs before mutation.
- Modify `hermes_cli/jobs_db.py`: migrate and persist `handoff_json`, include it in idempotency material, and enqueue the same validated receipt atomically.
- Modify `hermes_cli/jobs_dispatch.py`: create factual dispatcher/start receipts, attach worker receipts to phase transitions, return reviewer findings to correction attempts, and build complete blocker decisions.
- Modify `hermes_cli/jobs_loop.py`: classify a missing or invalid handoff as an infrastructure-contract failure eligible only for evidence-changing retry.
- Modify `hermes_cli/jobs_notifications.py`: retain lifecycle milestones as metadata, add reviewer-approval metadata, render substantive receipts, and remove elapsed-time-only heartbeat generation.
- Modify `gateway/jobs_notifications.py`: deliver the handoff rendering once to the exact origin and expose speaker/next-owner metadata.
- Modify `hermes_cli/jobs_release.py`: include the new contract module in every immutable Mac/PC release.
- Add focused tests under `tests/hermes_cli/` and `tests/gateway/` for validation, persistence, inter-agent feedback, exact-origin delivery, decisions, deduplication, restart recovery, and noise suppression.

## Task 1: Canonical Handoff Contract and Renderer

**Files:**
- Create: `hermes_cli/jobs_handoffs.py`
- Create: `tests/hermes_cli/test_jobs_handoffs.py`

- [ ] **Step 1: Write failing tests for a valid evidence-bound handoff**

```python
import pytest

from hermes_cli import jobs_handoffs as handoffs

DIGEST = "sha256:" + "a" * 64


def _base_raw():
    return {
        "summary": "I finished the responsive card fix.",
        "evidence_summary": [
            {"label": "Focused UI suite", "result": "18 passed", "digest": DIGEST}
        ],
        "next_action": "Claude Tester will verify dragging at 390 px.",
        "issues": [],
        "decision_request": None,
    }


def _normalize(raw=None, **identity_overrides):
    identity = {
        "job_id": "j_test",
        "attempt_id": "a_test",
        "speaker_id": "claude-mac-1",
        "speaker_role": "Claude Builder",
        "speaker_executor": "claude",
        "from_phase": "BUILDING",
        "to_phase": "EVIDENCE_COLLECTING",
        "next_owner_role": "Claude Tester",
        "outcome": "handed_off",
        "artifact_identity": {"kind": "commit", "value": "b" * 40},
        "transition_evidence": {"executor_exit": DIGEST},
        "created_at": 1_786_560_000,
    }
    identity.update(identity_overrides)
    return handoffs.normalize_handoff(_base_raw() if raw is None else raw, **identity)


def test_normalize_handoff_binds_identity_evidence_and_next_owner():
    result = handoffs.normalize_handoff(
        _base_raw(),
        job_id="j_test",
        attempt_id="a_test",
        speaker_id="claude-mac-1",
        speaker_role="Claude Builder",
        speaker_executor="claude",
        from_phase="BUILDING",
        to_phase="EVIDENCE_COLLECTING",
        next_owner_role="Claude Tester",
        outcome="handed_off",
        artifact_identity={"kind": "commit", "value": "b" * 40},
        transition_evidence={"executor_exit": DIGEST},
        created_at=1_786_560_000,
    )

    assert result["schema_version"] == 1
    assert result["job_id"] == "j_test"
    assert result["speaker_role"] == "Claude Builder"
    assert result["next_owner_role"] == "Claude Tester"
    assert result["evidence_summary"][0]["digest"] == DIGEST
```

- [ ] **Step 2: Run the focused test and verify the missing module is the failure**

Run: `uv run --extra dev pytest -q tests/hermes_cli/test_jobs_handoffs.py::test_normalize_handoff_binds_identity_evidence_and_next_owner`

Expected: FAIL during import because `hermes_cli.jobs_handoffs` does not exist.

- [ ] **Step 3: Implement the bounded canonical contract**

Create constants, a typed exception, normalizers, and one canonical JSON-size check:

```python
SCHEMA_VERSION = 1
MAX_HANDOFF_JSON_CHARS = 3_000
MAX_SUMMARY_CHARS = 700
MAX_NEXT_ACTION_CHARS = 500
MAX_EVIDENCE_ITEMS = 8
MAX_ISSUES = 8
OUTCOMES = frozenset(
    {"started", "handed_off", "passed", "rejected", "blocked", "activated", "completed"}
)


class HandoffValidationError(ValueError):
    pass


def normalize_handoff(raw, *, job_id, attempt_id, speaker_id, speaker_role,
                      speaker_executor, from_phase, to_phase, next_owner_role,
                      outcome, artifact_identity, transition_evidence, created_at):
    summary = _bounded_text(raw.get("summary"), "summary", MAX_SUMMARY_CHARS)
    next_action = _bounded_text(
        raw.get("next_action"), "next_action", MAX_NEXT_ACTION_CHARS
    )
    evidence_summary = _normalize_evidence(
        raw.get("evidence_summary"), transition_evidence
    )
    issues = _normalize_issues(raw.get("issues"))
    decision = _normalize_decision(raw.get("decision_request"))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "job_id": _identity(job_id, "job_id"),
        "attempt_id": _identity(attempt_id, "attempt_id"),
        "speaker_id": _identity(speaker_id, "speaker_id"),
        "speaker_role": _bounded_text(speaker_role, "speaker_role", 80),
        "speaker_executor": _bounded_text(speaker_executor, "speaker_executor", 32),
        "from_phase": _bounded_text(from_phase, "from_phase", 32),
        "to_phase": _bounded_text(to_phase, "to_phase", 32),
        "next_owner_role": _optional_text(next_owner_role, 80),
        "summary": summary,
        "evidence_summary": evidence_summary,
        "outcome": _outcome(outcome),
        "next_action": next_action,
        "issues": issues,
        "artifact_identity": _normalize_artifact(artifact_identity),
        "decision_request": decision,
        "created_at": _timestamp(created_at),
    }
    _validate_transition_requirements(receipt)
    _reject_secrets(receipt)
    if len(_canonical_json(receipt)) > MAX_HANDOFF_JSON_CHARS:
        raise HandoffValidationError("handoff exceeds size bound")
    return receipt
```

The helper rules are exact: SHA-256 evidence digests must occur in `transition_evidence`; completion may omit `next_owner_role`; rejection requires at least one issue with `requirement`, `finding`, and `required_fix`; blocked outcomes require a decision; all other continuing outcomes require a next owner. `attempt_id=None` is legal only for the deterministic intake receipt whose edge is `INTAKE → QUEUED`; every attempt transition requires a non-empty attempt ID.

- [ ] **Step 4: Add failing validation tests for decisions, reviewer findings, bounds, and secrets**

```python
def _complete_decision():
    return {
        "question": "Should Hermes reconnect the Codex lane now?",
        "options": [
            {"id": "reconnect", "label": "Reconnect now", "consequence": "Hermes retries after authentication passes."},
            {"id": "keep_blocked", "label": "Keep it blocked", "consequence": "No provider work runs."},
        ],
        "recommendation": "reconnect",
        "recommendation_reason": "The Job cannot safely run without valid lane authentication.",
        "blocked_scope": "Only this Codex Job is blocked.",
        "safe_state": "No code was merged, pushed, deployed, or retried.",
        "next_owner_role": "Hermes Dispatcher",
    }


def _blocked_raw():
    raw = _base_raw()
    raw["summary"] = "Codex authentication is unavailable."
    raw["next_action"] = "Wait for Brandon's choice."
    raw["decision_request"] = _complete_decision()
    return raw


@pytest.mark.parametrize("missing", ["question", "options", "recommendation", "blocked_scope", "safe_state", "next_owner_role"])
def test_blocked_handoff_rejects_incomplete_decision(missing):
    raw = _blocked_raw()
    raw["decision_request"].pop(missing)
    with pytest.raises(handoffs.HandoffValidationError):
        _normalize(
            raw,
            speaker_id="hermes-dispatcher",
            speaker_role="Hermes",
            speaker_executor="system",
            from_phase="ASSIGNED",
            to_phase="BLOCKED",
            next_owner_role="Brandon",
            outcome="blocked",
        )


def test_rejection_requires_exact_finding_and_required_fix():
    raw = _base_raw()
    raw["issues"] = [{"requirement": "mobile drag", "finding": "overflows"}]
    with pytest.raises(handoffs.HandoffValidationError, match="required_fix"):
        _normalize(
            raw,
            speaker_role="Independent Reviewer",
            from_phase="REVIEWING",
            to_phase="FAILED",
            next_owner_role="Claude Builder",
            outcome="rejected",
        )


@pytest.mark.parametrize("unsafe", ["Authorization: Bearer hidden", "token=hidden", "sk-secret-value"])
def test_handoff_rejects_secret_markers(unsafe):
    raw = _base_raw()
    raw["summary"] = unsafe
    with pytest.raises(handoffs.HandoffValidationError, match="secret"):
        _normalize(raw)
```

- [ ] **Step 5: Implement complete decision and issue validation**

`decision_request.options` is a list of two or three objects with `id`, `label`, and `consequence`. Require one recommendation that names a supplied option ID, a concrete `blocked_scope`, a concrete `safe_state`, and a `next_owner_role`. Reject control characters, nested unknown keys, secret markers, and any string outside its cap.

- [ ] **Step 6: Add and implement deterministic conversational rendering**

```python
def test_render_reviewer_rejection_is_substantive():
    raw = _base_raw()
    raw["summary"] = "I sent the build back because the mobile card still overflows."
    raw["issues"] = [{
        "requirement": "Cards must fit a 320 px viewport",
        "finding": "The card overflows at 320 px",
        "required_fix": "Reduce the card container width and rerun the drag journey",
    }]
    receipt = _normalize(
        raw,
        speaker_role="Independent Reviewer",
        from_phase="REVIEWING",
        to_phase="FAILED",
        next_owner_role="Claude Builder",
        outcome="rejected",
    )
    text = handoffs.render_handoff(receipt)
    assert text.startswith("Independent Reviewer → Claude Builder")
    assert "overflows at 320 px" in text
    assert "reduce the card container width" in text
    assert "Next:" in text


def test_render_decision_contains_options_recommendation_and_safe_state():
    text = handoffs.render_handoff(
        _normalize(
            _blocked_raw(),
            speaker_id="hermes-dispatcher",
            speaker_role="Hermes",
            speaker_executor="system",
            from_phase="ASSIGNED",
            to_phase="BLOCKED",
            next_owner_role="Brandon",
            outcome="blocked",
        )
    )
    assert "Hermes needs your decision" in text
    assert "1." in text and "2." in text
    assert "Recommendation:" in text
    assert "Blocked:" in text
    assert "Safe now:" in text
```

Implement `render_handoff(receipt) -> str` as a pure formatter. It uses only normalized receipt fields and never reads a Job goal, raw log, prompt, stack trace, credential, or sealed reviewer instruction.

- [ ] **Step 7: Run the contract suite**

Run: `uv run --extra dev pytest -q tests/hermes_cli/test_jobs_handoffs.py`

Expected: PASS with every validation and renderer test green.

- [ ] **Step 8: Commit the contract**

```bash
git add hermes_cli/jobs_handoffs.py tests/hermes_cli/test_jobs_handoffs.py
git commit -m "feat(jobs): define substantive handoff contract"
```

## Task 2: Worker-Authored Builder and Reviewer Handoffs

**Files:**
- Modify: `hermes_cli/data/jobs-build-result.v1.schema.json`
- Modify: `hermes_cli/data/jobs-review-result.v1.schema.json`
- Modify: `hermes_cli/jobs_execution.py:30-43`
- Modify: `hermes_cli/jobs_reliability.py:32-52, 246-261, 270-422, 661-745`
- Modify: `scripts/jobs_remote_worker.py:74-104`
- Test: `tests/hermes_cli/test_jobs_production_reliability.py`

- [ ] **Step 1: Write failing strict-schema tests for substantive output**

```python
def load_phase_schema(filename):
    path = Path(jobs_reliability.__file__).resolve().parent / "data" / filename
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_schema_requires_handoff_and_critical_user_journey():
    schema = load_phase_schema("jobs-build-result.v1.schema.json")
    assert {"handoff", "critical_user_journey"} <= set(schema["required"])
    assert schema["properties"]["critical_user_journey"]["properties"]["result"]["enum"] == ["pass", "fail", "not_applicable"]


def test_review_schema_requires_handoff_and_unable_to_verify_verdict():
    schema = load_phase_schema("jobs-review-result.v1.schema.json")
    assert "handoff" in schema["required"]
    assert schema["properties"]["verdict"]["enum"] == ["PASS", "NEEDS_CHANGES", "UNABLE_TO_VERIFY"]
```

- [ ] **Step 2: Run schema tests and confirm they fail on the absent fields**

Run: `uv run --extra dev pytest -q tests/hermes_cli/test_jobs_production_reliability.py -k 'schema_requires or strict_response'`

Expected: FAIL for missing `handoff`, missing `critical_user_journey`, and missing `UNABLE_TO_VERIFY`.

- [ ] **Step 3: Extend both strict JSON schemas**

The build result adds required objects:

```json
"critical_user_journey": {
  "type": "object",
  "additionalProperties": false,
  "required": ["name", "result", "evidence"],
  "properties": {
    "name": {"type": "string", "minLength": 1, "maxLength": 256},
    "result": {"type": "string", "enum": ["pass", "fail", "not_applicable"]},
    "evidence": {"type": "string", "minLength": 1, "maxLength": 1024}
  }
},
"handoff": {
  "type": "object",
  "additionalProperties": false,
  "required": ["summary", "next_action"],
  "properties": {
    "summary": {"type": "string", "minLength": 1, "maxLength": 700},
    "next_action": {"type": "string", "minLength": 1, "maxLength": 500}
  }
}
```

The review result keeps `findings` authoritative, adds `UNABLE_TO_VERIFY`, and adds the same strict `handoff` object. Every object property remains present in `required` so Codex strict-response mode accepts both schemas. Runtime validation requires at least one concrete missing-evidence finding for `UNABLE_TO_VERIFY` and at least one concrete defect for `NEEDS_CHANGES`; only `PASS` may carry an empty findings list.

- [ ] **Step 4: Write failing tests proving the reviewer receives bounded evidence, not the builder’s authority**

```python
def test_reviewer_prompt_contains_bounded_builder_handoff_and_independence_rule(tmp_path):
    context = _context(tmp_path, "codex")
    prompt = jobs_reliability._review_prompt(
        context,
        "c" * 40,
        builder_handoff={"summary": "Changed card width", "next_action": "Verify 390 px drag"},
        journey={"name": "mobile drag", "result": "pass", "evidence": "Playwright screenshot"},
    ).decode()
    assert "UNTRUSTED BUILDER HANDOFF" in prompt
    assert "independently verify" in prompt.lower()
    assert "Changed card width" in prompt
    assert context.goal not in prompt
```

- [ ] **Step 5: Carry handoffs through the provider-neutral result types**

Add `Mapping` to the imports from `typing`, then add fields without raw output:

```python
@dataclass(frozen=True)
class ReliabilityExecution:
    status: str
    commit: Optional[str]
    worktree: Path
    artifacts: tuple[ArtifactClaim, ...]
    executor_exit_digest: str
    output_capture_digest: str
    builder_handoff: Optional[Mapping[str, object]] = None
    tester_handoff: Optional[Mapping[str, object]] = None
    reviewer_handoff: Optional[Mapping[str, object]] = None
    failure_reason_code: Optional[str] = None
    http_status: Optional[int] = None
    safety_gate: bool = False
```

Add `build_handoff`, `critical_user_journey`, and `review_handoff` to `PhaseResult`. Update `_build_prompt` to demand the critical journey and first-person handoff. Update `_review_prompt` to label builder content untrusted and preserve the sealed independent verdict.

- [ ] **Step 6: Normalize provider results before returning them**

After the build response is read, reject missing/invalid handoff or journey with `HANDOFF_INCOMPLETE`. Generate `tester_handoff` strictly from the command/result/evidence records and journey object. After review, normalize its summary and turn `findings` into structured issues. Return `UNABLE_TO_VERIFY` as a non-passing reviewer handoff rather than approval.

- [ ] **Step 7: Preserve the fields over SSH and cleanup replacement**

`scripts/jobs_remote_worker.py` already uses `asdict(result)`. When replacing a result after cleanup failure, copy `build_handoff`, `critical_user_journey`, and `review_handoff` into the new `PhaseResult`. Update `SSHProviderPhaseRunner` deserialization to require mappings for those fields and refuse malformed transport data.

- [ ] **Step 8: Run provider and transport tests**

Run: `uv run --extra dev pytest -q tests/hermes_cli/test_jobs_production_reliability.py`

Expected: PASS for both Claude and Codex local sessions, Mac/PC transport, strict schemas, provider isolation, secret redaction, and handoff preservation.

- [ ] **Step 9: Commit provider handoffs**

```bash
git add hermes_cli/data/jobs-build-result.v1.schema.json hermes_cli/data/jobs-review-result.v1.schema.json hermes_cli/jobs_execution.py hermes_cli/jobs_reliability.py scripts/jobs_remote_worker.py tests/hermes_cli/test_jobs_production_reliability.py
git commit -m "feat(jobs): capture builder and reviewer handoffs"
```

## Task 3: Sign and Persist Handoffs Atomically

**Files:**
- Modify: `hermes_cli/jobs_db.py:376-394, 936-1008, 1130-1146, 3802-4057`
- Modify: `hermes_cli/jobs_graph.py:79-117, 138-177, 191-301`
- Modify: `tests/hermes_cli/test_jobs_graph.py`
- Modify: `tests/hermes_cli/test_jobs_graph_db.py`
- Modify: `tests/hermes_cli/test_jobs_notification_outbox.py`

- [ ] **Step 1: Write failing graph tests for missing and contradictory handoffs**

```python
def test_material_transition_without_handoff_fails_before_database_mutation(graph_rig):
    request = graph_rig.request("QUEUED", handoff=None)
    envelope = graph_rig.sign(request)
    revision = jdb.get_job(graph_rig.conn, graph_rig.job_id).revision
    with pytest.raises(handoffs.HandoffValidationError):
        graph.transition_attempt(graph_rig.conn, request, envelope=envelope, verifier=graph_rig.verifier)
    assert jdb.get_job(graph_rig.conn, graph_rig.job_id).revision == revision
    assert jdb.list_transitions(graph_rig.conn, graph_rig.job_id) == []


def test_signed_envelope_handoff_must_equal_transition_handoff(graph_rig):
    signed_request = graph_rig.request("QUEUED", handoff=graph_rig.valid_handoff("QUEUED"))
    envelope = graph_rig.sign(signed_request)
    changed = dict(signed_request.handoff)
    changed["summary"] = "different facts"
    request = replace(signed_request, handoff=changed)
    with pytest.raises(receipts.ReceiptVerificationError, match="handoff identity mismatch"):
        graph.transition_attempt(graph_rig.conn, request, envelope=envelope, verifier=graph_rig.verifier)
```

Extend the existing `GraphRig.request` with a `handoff` keyword, add `GraphRig.valid_handoff(target)` using `jobs_handoffs.normalize_handoff`, and add `"handoff": request.handoff` to `GraphRig.sign`. The tests already import `replace`; add `jobs_handoffs as handoffs` to their imports.

- [ ] **Step 2: Run the graph tests and verify they fail for the missing field**

Run: `uv run --extra dev pytest -q tests/hermes_cli/test_jobs_graph.py tests/hermes_cli/test_jobs_graph_db.py -k handoff`

Expected: FAIL because `TransitionRequest` and transition rows do not carry handoffs.

- [ ] **Step 3: Add the additive database migration and write contract**

Add `handoff_json TEXT` to `job_attempt_transitions` in `SCHEMA_SQL` and:

```python
add_column_if_missing(
    conn,
    "job_attempt_transitions",
    "handoff_json",
    "handoff_json TEXT",
)
```

Add `handoff: Optional[Mapping[str, object]] = None` to `TransitionWrite`. Include canonical `handoff_json` in `_transition_material`, the `INSERT`, `_transition_to_dict`, and `list_transitions`. Existing historical rows remain `NULL`; no history is rewritten or inferred.

Add the bounded retrieval seam used by correction attempts:

```python
def latest_handoff(conn: sqlite3.Connection, job_id: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT handoff_json FROM job_attempt_transitions "
        "WHERE job_id = ? AND handoff_json IS NOT NULL "
        "ORDER BY id DESC LIMIT 1",
        (job_id,),
    ).fetchone()
    if row is None:
        return None
    value = json.loads(row["handoff_json"])
    return jobs_handoffs.validate_persisted_handoff(value)
```

The helper returns only the canonical bounded handoff and never joins raw receipts, prompts, logs, credentials, or Job goals.

- [ ] **Step 4: Bind the same handoff into the signed graph envelope**

Add `handoff` to `TransitionRequest`, `TransitionRecord`, `_write`, and `_record`. Before `_verify_envelope`, call `jobs_handoffs.validate_persisted_handoff(request.handoff, target_state=request.target_state)`. In `_verify_envelope`, require:

```python
if payload.get("handoff") != request.handoff:
    raise jobs_receipts.ReceiptVerificationError("receipt handoff identity mismatch")
```

- [ ] **Step 5: Enqueue the validated handoff in the same transaction**

For every user-visible transition, add `"handoff": request.handoff` to `payload`. Add a `review-approved` milestone for `VERIFIED`, so reviewer approval is no longer silent. Creation enqueues a deterministic Hermes intake receipt built by `jobs_handoffs.intake_handoff(...)`. `BUILDING` remains a signed graph transition with a handoff receipt but does not enqueue a second visible message: `ASSIGNED` already represents the same dispatcher-to-builder custody transfer, and two chat lines would recreate the screenshot’s noise.

- [ ] **Step 6: Prove transaction rollback and idempotency include handoff bytes**

```python
def test_outbox_failure_rolls_back_transition_and_handoff(conn, signing_key):
    write, envelope = _transition(conn, signing_key, handoff=_valid_test_handoff())
    conn.execute("DROP TABLE job_notifications")
    conn.commit()
    with pytest.raises(sqlite3.OperationalError):
        jdb.record_transition(conn, write, envelope)
    assert conn.execute("SELECT COUNT(*) FROM job_attempt_transitions").fetchone()[0] == 0


def test_same_transition_key_with_different_handoff_conflicts(conn, signing_key):
    first, first_envelope = _transition(
        conn, signing_key, key="same-key", handoff=_valid_test_handoff(summary="first")
    )
    jdb.record_transition(conn, first, first_envelope)
    second = replace(first, handoff=_valid_test_handoff(summary="changed"))
    second_envelope = _signed(
        signing_key,
        receipt_id=second.receipt_id,
        job_id=second.job_id,
        attempt_id=second.attempt_id,
        state=second.target_state,
        handoff=second.handoff,
    )
    with pytest.raises(jdb.GraphConflict):
        jdb.record_transition(conn, second, second_envelope)
```

Extend the existing outbox-test `_transition` and `_signed` helpers with a `handoff` keyword, and define `_valid_test_handoff(summary="test handoff")` once using the canonical normalizer. This keeps every new fixture on the same production validation path.

- [ ] **Step 7: Run graph and outbox tests**

Run: `uv run --extra dev pytest -q tests/hermes_cli/test_jobs_graph.py tests/hermes_cli/test_jobs_graph_db.py tests/hermes_cli/test_jobs_notification_outbox.py`

Expected: PASS with atomic state, receipt, handoff, and notification persistence.

- [ ] **Step 8: Commit signed persistence**

```bash
git add hermes_cli/jobs_db.py hermes_cli/jobs_graph.py tests/hermes_cli/test_jobs_graph.py tests/hermes_cli/test_jobs_graph_db.py tests/hermes_cli/test_jobs_notification_outbox.py
git commit -m "feat(jobs): bind handoffs to graph transitions"
```

## Task 4: Dispatcher Handoffs, Corrections, and Decisions

**Files:**
- Modify: `hermes_cli/jobs_dispatch.py:224-246, 677-724, 726-844, 846-1172`
- Modify: `hermes_cli/jobs_loop.py:20-50, 130-160`
- Modify: `hermes_cli/jobs_reliability.py:246-261`
- Modify: `tests/hermes_cli/test_jobs_dispatch.py`
- Modify: `tests/hermes_cli/test_jobs_reliability_e2e.py`

- [ ] **Step 1: Write a failing happy-path test for all speaking roles**

```python
def test_happy_path_records_substantive_handoff_for_every_material_owner(reliability_rig):
    result = reliability_rig.run(
        _passing_executor(reliability_rig),
        gate_callback=_authoritative_gate(reliability_rig),
        activation_callback=_passing_activation,
    )
    transitions = jdb.list_transitions(reliability_rig.conn, result.job_id)
    by_state = {row["target_state"]: row["handoff"] for row in transitions}
    assert by_state["ASSIGNED"]["speaker_role"] == "Hermes Dispatcher"
    assert by_state["BUILDING"]["next_owner_role"] == "Claude Builder"
    assert by_state["EVIDENCE_COLLECTING"]["speaker_role"] == "Claude Builder"
    assert by_state["REVIEWING"]["speaker_role"] == "Claude Tester"
    assert by_state["VERIFIED"]["speaker_role"] == "Independent Reviewer"
    assert by_state["COMPLETED"]["outcome"] == "completed"
```

- [ ] **Step 2: Add handoff inputs to `advance` and the signed envelope**

Change the nested dispatcher helper to:

```python
def advance(target_state, evidence, *, commit, handoff,
            failure_class=None, blocker_code=None):
    request = graph.TransitionRequest(
        # existing immutable fields stay unchanged
        handoff=dict(handoff),
    )
    envelope = signer.sign(
        {
            "job_id": job.id,
            "attempt_id": attempt_id,
            "commit": commit,
            "state": target_state,
            "evidence": dict(evidence),
            "handoff": dict(handoff),
            "timestamp": observed_at,
            "transitioned_by": "jobs_dispatch/1",
        },
        receipt_id=receipt_id,
    )
```

Use deterministic Hermes receipts for `QUEUED`, `ASSIGNED`, and `BUILDING`. The visible `ASSIGNED` receipt is written only after claim, selected lane, worktree registration, and attempt-start evidence exist; it says Hermes handed custody to the named builder and names that builder’s first action. The `BUILDING` receipt records internal phase readiness without claiming that a provider process is alive and without creating another chat message.

- [ ] **Step 3: Attach worker-authored build, test, and review receipts**

Use `execution.builder_handoff` for `EVIDENCE_COLLECTING`, `execution.tester_handoff` for `REVIEWING`, and `execution.reviewer_handoff` for `VERIFIED` or review failure. Derive visible roles from the persisted executor (`Claude Builder` / `Claude Tester` or `Codex Builder` / `Codex Tester`); never label a Claude session as Codex or the reverse. Bind the exact candidate commit as `artifact_identity`; never use a self-reported commit without the existing Git observation. Merge every digest named by the handoff into that transition’s evidence map, so the canonical validator can prove each user-facing statement is tied to signed graph evidence.

Extend `ActivationDecision` with `completion_handoff: Optional[Mapping[str, object]]`. `production_completion_gate` supplies a Hermes activation receipt whose summary names only the observed candidate, environment/gate result, and rollback readiness. `COMPLETED` rejects a missing completion handoff; it never infers “live” from tests or review alone.

- [ ] **Step 4: Feed reviewer findings into the next correction attempt**

Add `prior_handoff: Optional[Mapping[str, object]] = None` to `DispatchContext`. Before executor invocation, load the most recent persisted rejection/blocked handoff for the Job. `_build_prompt` appends only that bounded JSON under `PREVIOUS VERIFIED HANDOFF`; it does not attach the full chat, vault, sealed reviewer prompt, or raw logs.

```python
def test_correction_builder_receives_exact_prior_reviewer_findings(reliability_rig):
    base_executor = _passing_executor(reliability_rig)

    def rejected_executor(context):
        execution = base_executor(context)
        rejection = handoffs.normalize_handoff(
            {
                "summary": "I sent the build back after independent review.",
                "evidence_summary": [{
                    "label": "Independent review",
                    "result": "needs changes",
                    "digest": _digest(b"review rejection"),
                }],
                "next_action": "Claude Builder must correct the 320 px overflow.",
                "issues": [{
                    "requirement": "Card must fit a 320 px viewport",
                    "finding": "card overflows at 320 px",
                    "required_fix": "reduce the container width and rerun drag proof",
                }],
                "decision_request": None,
            },
            job_id=context.job_id,
            attempt_id=context.attempt_id,
            speaker_id="independent-reviewer",
            speaker_role="Independent Reviewer",
            speaker_executor=context.executor,
            from_phase="REVIEWING",
            to_phase="FAILED",
            next_owner_role="Claude Builder",
            outcome="rejected",
            artifact_identity={"kind": "commit", "value": execution.commit},
            transition_evidence={"review": _digest(b"review rejection")},
            created_at=reliability_rig.clock,
        )
        return replace(execution, reviewer_handoff=rejection)

    def rejected_gate(_context, execution):
        return jobs_run.GateEvidence(
            action_outcome="failed",
            identity_verified=True,
            reason_code="REVIEW_NEEDS_CHANGES",
            commit=execution.commit,
            themis_review_digest=None,
            receipt_verification_digest=None,
        )

    first = reliability_rig.run(rejected_executor, gate_callback=rejected_gate)
    assert first.state == "FAILED"
    captured = {}

    def correction_executor(context):
        captured["prior_handoff"] = context.prior_handoff
        return _provider_failure(b"correction probe")(context)

    reliability_rig.run(correction_executor)
    assert captured["prior_handoff"]["issues"][0]["finding"] == "card overflows at 320 px"
    assert "sealed" not in json.dumps(captured["prior_handoff"]).lower()
```

Add imports for `replace`, `jobs_handoffs as handoffs`, and `jobs_run` where this test lives. Extend the existing `_passing_executor` so its returned `ReliabilityExecution` contains valid builder, tester, and reviewer handoffs; extend `_passing_activation` so `ActivationDecision` contains a completion handoff.

- [ ] **Step 5: Make retry-without-progress visible and non-looping**

Add `HANDOFF_INCOMPLETE` to infrastructure reason classification. For any retry, the correction handoff must include the prior failure evidence digest, the changed evidence digest, and what changed. Existing `decide_retry` blocks an identical digest; add an assertion that identical handoff plus identical evidence settles as `RETRY_REJECTED_NO_NEW_EVIDENCE` rather than “working.”

- [ ] **Step 6: Build complete deterministic decision requests**

Create `jobs_handoffs.blocker_handoff(...)` mappings for every current human-action class:

- auth/login: reconnect the exact provider lane or keep the Job blocked;
- safety/approval: approve the named bounded action or keep it blocked;
- exhausted evidence-changing retries: narrow/redesign the Job or stop it.

Each mapping supplies two options, a recommendation, consequences, `blocked_scope`, `safe_state`, and next owner. Unknown blocker codes cannot transition to user-facing `BLOCKED`; they settle through `HANDOFF_INCOMPLETE` diagnostic evidence.

- [ ] **Step 7: Prove reviewer rejection and `UNABLE_TO_VERIFY` remain honest**

```python
@pytest.mark.parametrize("verdict", ["NEEDS_CHANGES", "UNABLE_TO_VERIFY"])
def test_nonpassing_review_never_emits_approval(reliability_rig, verdict):
    result = reliability_rig.run(
        _review_result_executor(reliability_rig, verdict),
        gate_callback=_nonpassing_gate(verdict),
    )
    last = jdb.list_transitions(reliability_rig.conn, result.job_id)[-1]
    assert last["target_state"] in {"FAILED", "BLOCKED"}
    assert last["handoff"]["outcome"] != "passed"
    assert last["handoff"]["issues"]
```

Define `_review_result_executor(rig, verdict)` beside `_passing_executor`: it returns a normal candidate plus a normalized reviewer handoff whose issue is either the exact required correction or the exact missing evidence. Define the gate helper exactly as:

```python
def _nonpassing_gate(verdict):
    def gate(_context, execution):
        return jobs_run.GateEvidence(
            action_outcome="failed",
            identity_verified=True,
            reason_code=(
                "REVIEW_NEEDS_CHANGES"
                if verdict == "NEEDS_CHANGES"
                else "REVIEW_UNABLE_TO_VERIFY"
            ),
            commit=execution.commit,
            themis_review_digest=None,
            receipt_verification_digest=None,
        )
    return gate
```

- [ ] **Step 8: Run dispatcher and reliability end-to-end tests**

Run: `uv run --extra dev pytest -q tests/hermes_cli/test_jobs_dispatch.py tests/hermes_cli/test_jobs_reliability_e2e.py`

Expected: PASS for happy path, review rejection, correction, no-progress retry refusal, user decisions, failure taxonomy, lane cleanup, and provider isolation.

- [ ] **Step 9: Commit dispatcher handoffs**

```bash
git add hermes_cli/jobs_dispatch.py hermes_cli/jobs_loop.py hermes_cli/jobs_reliability.py tests/hermes_cli/test_jobs_dispatch.py tests/hermes_cli/test_jobs_reliability_e2e.py
git commit -m "feat(jobs): route substantive handoffs between agents"
```

## Task 5: Replace Generic Chat Spam with Substantive Watcher Messages

**Files:**
- Modify: `hermes_cli/jobs_notifications.py:28-58, 86-116, 400-530`
- Modify: `gateway/jobs_notifications.py:118-163, 187-319`
- Modify: `tests/hermes_cli/test_jobs_notification_messages.py`
- Modify: `tests/gateway/test_jobs_notification_delivery.py`
- Modify: `tests/gateway/test_jobs_notification_heartbeat.py`

- [ ] **Step 1: Replace generic-wording assertions with substantive rendering assertions**

```python
def _normalized_message_handoff(*, role="Claude Builder", outcome="handed_off"):
    digest = "sha256:" + "a" * 64
    return handoffs.normalize_handoff(
        {
            "summary": "I finished the card-width fix.",
            "evidence_summary": [{
                "label": "Focused UI suite", "result": "18 passed", "digest": digest
            }],
            "next_action": "Claude Tester will verify the 390 px drag journey.",
            "issues": [],
            "decision_request": None,
        },
        job_id="job-1",
        attempt_id="attempt-1",
        speaker_id="claude-mac-1",
        speaker_role=role,
        speaker_executor="claude",
        from_phase="BUILDING",
        to_phase="EVIDENCE_COLLECTING",
        next_owner_role="Claude Tester",
        outcome=outcome,
        artifact_identity={"kind": "commit", "value": "b" * 40},
        transition_evidence={"tests": digest},
        created_at=1_786_560_000,
    )


def test_new_handoff_payload_renders_one_agent_message_not_status_labels():
    record = _record(
        jn.MILESTONE_TESTING,
        handoff=_normalized_message_handoff(),
    )
    text = jn.render_milestone_message(record)
    assert text.startswith("Claude Builder → Claude Tester")
    assert "18 passed" in text
    assert not text.startswith(("Assigned —", "Building —", "Testing —", "Correcting —"))


def test_reviewer_approval_has_its_own_substantive_message():
    digest = "sha256:" + "c" * 64
    approval = handoffs.normalize_handoff(
        {
            "summary": "I independently approved the exact candidate commit.",
            "evidence_summary": [{
                "label": "Independent review", "result": "pass", "digest": digest
            }],
            "next_action": "Hermes will run the activation gate.",
            "issues": [],
            "decision_request": None,
        },
        job_id="job-1",
        attempt_id="attempt-1",
        speaker_id="independent-reviewer",
        speaker_role="Independent Reviewer",
        speaker_executor="codex",
        from_phase="REVIEWING",
        to_phase="VERIFIED",
        next_owner_role="Hermes Activation Gate",
        outcome="passed",
        artifact_identity={"kind": "commit", "value": "b" * 40},
        transition_evidence={"review": digest},
        created_at=1_786_560_001,
    )
    record = _record(jn.MILESTONE_REVIEW_APPROVED, handoff=approval)
    assert jn.render_milestone_message(record).startswith("Independent Reviewer approved")
```

Add `from hermes_cli import jobs_handoffs as handoffs` to the message test module; reuse its existing `_record` helper so the tests exercise the real `NotificationRecord` shape.

- [ ] **Step 2: Render handoffs first and legacy milestones only as compatibility fallback**

`render_milestone_message` checks `payload["handoff"]`. When present, validate and call `jobs_handoffs.render_handoff`. When absent, render one `Legacy Jobs update — ...` line from the old milestone. It never emits both messages for one row and never turns a malformed handoff into generic success.

- [ ] **Step 3: Stop elapsed time from creating a progress claim**

Remove `enqueue_due_heartbeats(conn, now=now)` from `deliver_due_notification_once`. Keep historical heartbeat rows readable. Replace heartbeat tests with:

```python
def test_elapsed_time_alone_never_enqueues_or_delivers_progress(jobs_path, session_db, base):
    _create_job(jobs_path)
    session_db.create_session("session-1", source="telegram")
    assert _deliver(session_db, jobs_path, now=base) is not None
    assert _deliver(session_db, jobs_path, now=base + 86_400) is None
    assert len(_contents(session_db)) == 1


def test_legacy_heartbeat_row_remains_readable_once(jobs_path, session_db, base):
    job_id = _create_job(jobs_path)
    _enqueue_milestone(jobs_path, job_id, jn.MILESTONE_HEARTBEAT, revision=2, now=base)
    assert _deliver(session_db, jobs_path, now=base) is not None
    assert sum("Legacy Jobs update" in text for text in _contents(session_db)) == 1
```

- [ ] **Step 4: Expose speaker and next-owner metadata without changing destination rules**

Add `speaker_role`, `speaker_executor`, `next_owner_role`, and `outcome` to `_display_metadata` only when a validated handoff exists. Preserve `notification_id`, Job ID, revision, platform, chat, thread, profile, and exact resolved session.

- [ ] **Step 5: Prove malformed handoffs fail closed in delivery**

If a new-format payload contains an invalid handoff, render one bounded `Hermes could not explain this handoff` diagnostic with Job number/name and current safe state. Do not render the claimed outcome, evidence, or next owner. Stable notification ID makes restart delivery idempotent.

- [ ] **Step 6: Run renderer, delivery, and noise tests**

Run: `uv run --extra dev pytest -q tests/hermes_cli/test_jobs_notification_messages.py tests/gateway/test_jobs_notification_delivery.py tests/gateway/test_jobs_notification_heartbeat.py`

Expected: PASS with substantive messages, exact-origin metadata, legacy compatibility, no elapsed-time spam, and no false success fallback.

- [ ] **Step 7: Commit watcher messages**

```bash
git add hermes_cli/jobs_notifications.py gateway/jobs_notifications.py tests/hermes_cli/test_jobs_notification_messages.py tests/gateway/test_jobs_notification_delivery.py tests/gateway/test_jobs_notification_heartbeat.py
git commit -m "feat(jobs): deliver substantive exact-origin updates"
```

## Task 6: End-to-End Handoff, Correction, Decision, and Restart Proof

**Files:**
- Create: `tests/hermes_cli/test_jobs_handoff_notifications_e2e.py`
- Modify: `tests/hermes_cli/test_jobs_notification_outbox.py`
- Modify: `tests/gateway/test_jobs_notification_delivery.py`

- [ ] **Step 1: Build a deterministic full conversation fixture**

The fixture creates an exact-origin Job and a decoy session, runs these durable transitions, and drains the real watcher after each transition:

```python
EXPECTED_STATES = (
    "ASSIGNED",
    "BUILDING",
    "EVIDENCE_COLLECTING",
    "REVIEWING",
    "FAILED",
    "ASSIGNED",
    "BUILDING",
    "EVIDENCE_COLLECTING",
    "REVIEWING",
    "VERIFIED",
    "COMPLETED",
)
```

The first review rejects a 320 px overflow, the correction handoff names the changed container width, the second review passes, and completion names the exact candidate commit and activation evidence.

- [ ] **Step 2: Assert the origin receives substantive messages once and in order**

```python
contents = messages(origin_session)
assert any(text.startswith("Claude Builder → Claude Tester") for text in contents)
assert any("sent it back" in text and "320 px" in text for text in contents)
assert any("changed the container width" in text for text in contents)
assert any(text.startswith("Independent Reviewer approved") for text in contents)
assert contents[-1].startswith("Job complete")
assert messages(decoy_session) == []
assert not any(text.startswith(("Assigned —", "Building —", "Correcting —")) for text in contents)
```

- [ ] **Step 3: Prove decision requests contain the actionable question**

Run a separate auth-blocked Job and assert the visible message names the exact lane/provider, two options and consequences, one recommendation, blocked scope, safe state, and next owner. Assert “requires your decision” without those facts never appears.

- [ ] **Step 4: Prove crash-after-append and watcher restart do not duplicate**

Reuse the existing append-then-acknowledge crash seam. Deliver a handoff once, reset only its outbox acknowledgement, run a new watcher instance, and assert `platform_message_id == notification_id` appears exactly once in `SessionDB`.

- [ ] **Step 5: Prove per-Job ordering survives a temporarily unavailable origin**

Make the exact origin busy, enqueue builder→tester and tester→reviewer handoffs, run the watcher, and assert neither reaches a decoy. Release the busy check and assert both arrive in transition order.

- [ ] **Step 6: Run the full handoff integration set**

Run: `uv run --extra dev pytest -q tests/hermes_cli/test_jobs_handoff_notifications_e2e.py tests/hermes_cli/test_jobs_notification_outbox.py tests/gateway/test_jobs_notification_delivery.py`

Expected: PASS for correction, re-review, decision detail, exact-origin isolation, ordering, retry, and restart idempotency.

- [ ] **Step 7: Commit end-to-end proof**

```bash
git add tests/hermes_cli/test_jobs_handoff_notifications_e2e.py tests/hermes_cli/test_jobs_notification_outbox.py tests/gateway/test_jobs_notification_delivery.py
git commit -m "test(jobs): prove substantive handoff conversation"
```

## Task 7: Immutable Release, Broad Regression, and Live Canary

**Files:**
- Modify: `hermes_cli/jobs_release.py:51-96`
- Modify: `tests/hermes_cli/test_jobs_release.py`
- Create: `scripts/jobs-handoff-canary.py`
- Create: `tests/hermes_cli/test_jobs_handoff_canary.py`

- [ ] **Step 1: Write a failing release-manifest test**

```python
def test_release_contains_handoff_contract_and_canary():
    assert "hermes_cli/jobs_handoffs.py" in jobs_release.RELEASE_FILES
    assert "scripts/jobs-handoff-canary.py" in jobs_release.RELEASE_FILES
```

- [ ] **Step 2: Add the new runtime files to the explicit release set**

Add only `hermes_cli/jobs_handoffs.py` and `scripts/jobs-handoff-canary.py` to `RELEASE_FILES`. Existing modified schema, dispatcher, database, reliability, notification, gateway, and remote-worker files are already explicit release members.

- [ ] **Step 3: Add an installed-runtime canary with temporary stores**

`scripts/jobs-handoff-canary.py` accepts `--runtime-root` and `--output`. It imports `jobs_handoffs`, `jobs_db`, and `gateway.jobs_notifications` strictly from that immutable runtime; creates temporary Jobs and Session databases; delivers one builder→tester, reviewer rejection, correction, approval, decision, and completion sequence; verifies a decoy receives zero messages; and writes a JSON receipt containing release SHA, notification IDs, message digests, origin count, decoy count, and verdict. It never touches the live Jobs database, profile data, credentials, or provider lanes.

- [ ] **Step 4: Test canary refusal and success**

```python
def run_canary(runtime_root, *, output=None):
    receipt = output or (Path(runtime_root).parent / "canary-receipt.json")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/jobs-handoff-canary.py",
            "--runtime-root",
            str(runtime_root),
            "--output",
            str(receipt),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return completed
    return json.loads(receipt.read_text(encoding="utf-8"))


def test_canary_refuses_mutable_or_mismatched_runtime(tmp_path):
    result = run_canary(tmp_path / "missing-runtime")
    assert result.returncode == 2


def test_canary_proves_exact_origin_and_substantive_sequence(release_runtime, tmp_path):
    receipt = run_canary(release_runtime, output=tmp_path / "receipt.json")
    assert receipt["verdict"] == "PASS"
    assert receipt["decoy_message_count"] == 0
    assert receipt["duplicate_notification_ids"] == []
    assert receipt["generic_status_message_count"] == 0
```

Import `json`, `subprocess`, `sys`, and `Path`; define `REPO_ROOT = Path(__file__).resolve().parents[2]`. The `release_runtime` fixture copies exactly `jobs_release.RELEASE_FILES` under a temporary `hermes-agent-` directory whose suffix is the test Git SHA, then runs the release verifier before yielding it.

- [ ] **Step 5: Run focused and affected broad suites**

Run:

```bash
uv run --extra dev pytest -q \
  tests/hermes_cli/test_jobs_handoffs.py \
  tests/hermes_cli/test_jobs_production_reliability.py \
  tests/hermes_cli/test_jobs_graph.py \
  tests/hermes_cli/test_jobs_graph_db.py \
  tests/hermes_cli/test_jobs_notification_outbox.py \
  tests/hermes_cli/test_jobs_notification_messages.py \
  tests/hermes_cli/test_jobs_dispatch.py \
  tests/hermes_cli/test_jobs_reliability_e2e.py \
  tests/hermes_cli/test_jobs_handoff_notifications_e2e.py \
  tests/hermes_cli/test_jobs_handoff_canary.py \
  tests/gateway/test_jobs_notification_delivery.py \
  tests/gateway/test_jobs_notification_heartbeat.py \
  tests/hermes_cli/test_jobs_release.py
```

Expected: all selected tests PASS with no critical skip.

Then run:

```bash
uv run --extra dev pytest -q tests/hermes_cli/test_jobs_*.py tests/gateway/test_jobs_*.py
```

Expected: all Jobs and gateway Jobs tests PASS with no critical skip.

- [ ] **Step 6: Commit release and canary support**

```bash
git add hermes_cli/jobs_release.py tests/hermes_cli/test_jobs_release.py scripts/jobs-handoff-canary.py tests/hermes_cli/test_jobs_handoff_canary.py
git commit -m "chore(jobs): package handoff reliability release"
```

- [ ] **Step 7: Build and verify the clean immutable candidate**

```bash
handoff_candidate_sha=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
uv run python scripts/jobs-release-activate.py \
  --source-root "$(pwd)" \
  --output-root /Users/brandon/.hermes/releases
handoff_release_root="/Users/brandon/.hermes/releases/hermes-agent-${handoff_candidate_sha}"
uv run python scripts/jobs-handoff-canary.py \
  --runtime-root "$handoff_release_root" \
  --output "/Users/brandon/.hermes/activation-preflights/handoff-canary-${handoff_candidate_sha}.json"
```

Expected: source is clean, immutable release verification passes, and canary receipt verdict is `PASS` with zero decoy or duplicate messages.

- [ ] **Step 8: Write live preflight and activate the exact Mac release**

```bash
handoff_candidate_sha=$(git rev-parse HEAD)
handoff_preflight="/Users/brandon/.hermes/activation-preflights/handoff-${handoff_candidate_sha}.json"
uv run python scripts/jobs-release-activate.py \
  --source-root "$(pwd)" \
  --live-profile mac \
  --expected-release-sha "$handoff_candidate_sha" \
  --write-live-preflight "$handoff_preflight"
uv run python scripts/jobs-release-activate.py \
  --source-root "$(pwd)" \
  --live-profile mac \
  --expected-release-sha "$handoff_candidate_sha" \
  --acknowledge-live-activation "ACTIVATE-JOBS-LIVE:mac:${handoff_candidate_sha}" \
  --preflight "$handoff_preflight" \
  --apply
```

Expected: activation emits a receipt naming the exact release, protected files, backup location, and rollback inputs. It changes no Job row, profile data, credential, provider lane, or PC state.

- [ ] **Step 9: Point the gateway at the installed release and restart only that service**

```bash
handoff_candidate_sha=$(git rev-parse HEAD)
handoff_release_root="/Users/brandon/.hermes/releases/hermes-agent-${handoff_candidate_sha}"
handoff_pointer_backup="/Users/brandon/.hermes/activation-receipts/gateway-plist-${handoff_candidate_sha}.bak"
uv run python scripts/jobs-gateway-runtime-pointer.py \
  --plist /Users/brandon/Library/LaunchAgents/ai.hermes.gateway.plist \
  --runtime-root "$handoff_release_root" \
  --backup "$handoff_pointer_backup" \
  --apply
launchctl kickstart -k "gui/$(id -u)/ai.hermes.gateway"
```

Expected: `ai.hermes.gateway` returns with one running PID and its launch configuration names the immutable candidate release.

- [ ] **Step 10: Verify live health and retain rollback evidence**

Run:

```bash
launchctl list | grep '^.*ai\.hermes\.gateway$'
grep -F "jobs notifications" /Users/brandon/.hermes/logs/gateway.log | tail -20
```

Expected: gateway is running, the watcher has no import/schema/render loop error, and no repeated generic status message is emitted. Preserve the activation receipt, installed-runtime canary receipt, gateway pointer backup, and release directory. If watcher delivery or gateway health fails, restore the pointer backup and kickstart `ai.hermes.gateway`; do not rewrite Jobs history or replay business actions.

## Final Acceptance Checklist

- [ ] Every material custody transfer has one persisted, signed handoff.
- [ ] Builder substance reaches tester/reviewer; reviewer findings reach the next correction builder.
- [ ] `UNABLE_TO_VERIFY` cannot become approval.
- [ ] User-decision messages contain the exact question, options, recommendation, consequences, blocked scope, safe state, and next owner.
- [ ] New handoff messages replace, rather than accompany, generic milestone chat lines.
- [ ] Elapsed time alone cannot claim progress or create heartbeat spam.
- [ ] Exact-origin delivery remains ordered, retryable, idempotent, restart-safe, and has no fallback chat.
- [ ] Raw logs, prompts, credentials, sealed reviewer instructions, and unrestricted memory never enter handoff payloads.
- [ ] Focused suites, broad Jobs suites, immutable release verification, installed-runtime canary, live gateway health, and rollback evidence all pass.
