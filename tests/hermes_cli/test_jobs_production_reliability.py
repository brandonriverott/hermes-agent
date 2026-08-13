"""Production reliability adapters must be installed and provider-isolated."""

from __future__ import annotations

import json
import subprocess
import base64
import io
import inspect
import unicodedata
import platform
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from hermes_cli import jobs_dispatch as dispatch
from hermes_cli import jobs_execution
from hermes_cli import jobs_executors
from hermes_cli import jobs_identity
from hermes_cli import jobs_reliability
from hermes_cli import jobs_receipts
from hermes_cli import jobs_runtime
from hermes_cli import jobs_lanes
from hermes_cli import jobs_loop
from scripts import jobs_remote_worker


def _phase_schema(filename: str) -> dict:
    path = Path(jobs_reliability.__file__).resolve().parent / "data" / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_build_payload(**overrides) -> dict:
    payload = {
        "outcome": "succeeded",
        "failure_class": None,
        "reason": None,
        "tests": [
            {
                "cmd": "uv run pytest -q tests/focused.py",
                "result": "pass",
                "evidence": "1 passed",
                "classification": None,
                "note": None,
            }
        ],
        "critical_user_journey": {
            "name": "mobile card drag",
            "result": "pass",
            "evidence": "Card remained inside the 390 px viewport.",
        },
        "handoff": {
            "summary": "I completed the scoped card change.",
            "next_action": "Test the committed change independently.",
        },
    }
    payload.update(overrides)
    return payload


def _valid_review_payload(**overrides) -> dict:
    payload = {
        "verdict": "PASS",
        "finding_type": "none",
        "findings": [],
        "checks_run": ["Reviewed the candidate diff and focused evidence."],
        "handoff": {
            "summary": "I independently verified the candidate.",
            "next_action": "Hermes can continue to the remaining gate.",
        },
        "outcome_evidence": {
            "critical_user_journey": "Open the candidate and verify behavior.",
            "verdict": "PASS",
            "environment": "isolated test worktree",
            "checks_run": ["Run the critical user journey."],
            "observed_behavior": "The expected behavior was observed.",
            "artifact_digests": ["sha256:" + "7" * 64],
            "observed_at": 1786233600,
        },
        "knowledge_closure": None,
    }
    payload.update(overrides)
    return payload


def _valid_transport_payload(provider: str = "codex", *, status: str = "succeeded"):
    display = provider.capitalize()
    if status == "failed":
        return {
            "status": "failed",
            "commit": None,
            "tests": [],
            "review": {},
            "executor_exit_digest": "sha256:" + "3" * 64,
            "output_capture_digest": "sha256:" + "4" * 64,
            "build_handoff": None,
            "critical_user_journey": None,
            "review_handoff": None,
            "failure_reason_code": "PROCESS_CRASHED",
            "http_status": None,
            "safety_gate": False,
        }
    review = _valid_review_payload()
    return {
        "status": "succeeded",
        "commit": "c" * 40,
        "tests": _valid_build_payload()["tests"],
        "review": review,
        "executor_exit_digest": "sha256:" + "3" * 64,
        "output_capture_digest": "sha256:" + "4" * 64,
        "build_handoff": {
            "speaker_role": f"{display} Builder",
            "speaker_executor": provider,
            "summary": "I built the candidate.",
            "next_action": "Test it.",
            "next_owner_role": f"{display} Tester",
        },
        "critical_user_journey": {
            "name": "mobile drag",
            "result": "pass",
            "evidence": "The focused check passed.",
        },
        "review_handoff": {
            "speaker_role": "Independent Reviewer",
            "speaker_executor": provider,
            "summary": review["handoff"]["summary"],
            "next_action": review["handoff"]["next_action"],
            "next_owner_role": "Hermes",
            "verdict": "PASS",
            "issues": [],
        },
        "failure_reason_code": None,
        "http_status": None,
        "safety_gate": False,
    }


def test_claude_phase_env_inherits_only_automation_oauth_credential(
    tmp_path, monkeypatch
):
    """A Claude lane needs subscription OAuth without receiving API secrets."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-test-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-worker")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "must-not-reach-worker")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-reach-worker")
    monkeypatch.setattr(jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/claude")

    env = jobs_reliability._phase_env(
        "claude", tmp_path / "claude-mac-1", tmp_path / "handoff"
    )

    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-test-value"
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_TOKEN" not in env
    assert "UNRELATED_SECRET" not in env


def test_codex_phase_env_does_not_inherit_claude_oauth_credential(
    tmp_path, monkeypatch
):
    """The Claude credential must not cross into a Codex worker lane."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-test-value")
    monkeypatch.setattr(jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex")

    env = jobs_reliability._phase_env(
        "codex", tmp_path / "codex-mac-1", tmp_path / "handoff"
    )

    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_production_auth_preflight_runs_in_exact_claude_lane(
    tmp_path, monkeypatch
):
    """An idle lane is not authenticated until its real CLI proves it."""
    lane_root = tmp_path / "claude-mac-1"
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": "oauth_token",
                    "apiProvider": "firstParty",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(jobs_reliability.subprocess, "run", run)
    monkeypatch.setattr(jobs_reliability.shutil, "which", lambda *_a, **_k: "/bin/claude")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-test-value")

    assert jobs_reliability.production_auth_preflight("claude", lane_root) is True
    argv, kwargs = calls[0]
    assert argv == ["/bin/claude", "auth", "status"]
    assert kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(lane_root / "auth")
    assert kwargs["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-test-value"
    assert "ANTHROPIC_API_KEY" not in kwargs["env"]


def test_production_auth_preflight_rejects_logged_out_claude_lane(
    tmp_path, monkeypatch
):
    """A successful process exit cannot hide Claude's logged-out result."""
    monkeypatch.setattr(jobs_reliability.shutil, "which", lambda *_a, **_k: "/bin/claude")
    monkeypatch.setattr(
        jobs_reliability.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"loggedIn": False, "authMethod": "none"}),
            stderr="",
        ),
    )

    assert (
        jobs_reliability.production_auth_preflight(
            "claude", tmp_path / "claude-mac-1"
        )
        is False
    )


@pytest.mark.parametrize(
    "filename",
    ["jobs-build-result.v1.schema.json", "jobs-review-result.v1.schema.json"],
)
def test_codex_phase_schemas_are_strict_response_format_compatible(filename):
    """Every object property must be required for Codex strict schemas.

    Fields that are logically optional must express absence with ``null``.
    Codex rejects schemas that omit object properties from ``required`` before
    it starts the worker, so this validates the release artifact directly.
    """
    schema = json.loads(
        (
            Path(jobs_reliability.__file__).resolve().parent / "data" / filename
        ).read_text()
    )

    def assert_strict(node):
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if isinstance(properties, dict):
            assert set(node.get("required", ())) == set(properties)
            for child in properties.values():
                assert_strict(child)
        assert_strict(node.get("items"))

    assert_strict(schema)


def test_build_schema_requires_handoff_and_critical_user_journey():
    schema = _phase_schema("jobs-build-result.v1.schema.json")

    assert {"handoff", "critical_user_journey"} <= set(schema["required"])
    assert schema["properties"]["critical_user_journey"]["properties"]["result"][
        "enum"
    ] == ["pass", "fail", "not_applicable"]


def test_review_schema_requires_handoff_and_unable_to_verify_verdict():
    schema = _phase_schema("jobs-review-result.v1.schema.json")

    assert "handoff" in schema["required"]
    assert schema["properties"]["verdict"]["enum"] == [
        "PASS",
        "NEEDS_CHANGES",
        "UNABLE_TO_VERIFY",
    ]
    assert schema["properties"]["findings"]["items"]["minLength"] == 1
    assert schema["properties"]["checks_run"]["items"]["minLength"] == 1
    assert schema["properties"]["finding_type"]["enum"] == [
        "none",
        "defect",
        "missing_evidence",
    ]


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("jobs-build-result.v1.schema.json", _valid_build_payload()),
        ("jobs-review-result.v1.schema.json", _valid_review_payload()),
    ],
)
def test_phase_schemas_reject_extra_missing_and_oversized_handoff_fields(
    filename, payload
):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(_phase_schema(filename))
    validator.validate(payload)

    extra = json.loads(json.dumps(payload))
    extra["handoff"]["raw_output"] = "must not cross the boundary"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(extra)

    missing = json.loads(json.dumps(payload))
    del missing["handoff"]["next_action"]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(missing)

    oversized = json.loads(json.dumps(payload))
    oversized["handoff"]["summary"] = "x" * 701
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(oversized)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(extra="unknown"),
        lambda payload: payload["handoff"].update(extra="unknown"),
        lambda payload: payload["critical_user_journey"].update(extra="unknown"),
        lambda payload: payload["tests"][0].update(extra="unknown"),
        lambda payload: payload["tests"][0].pop("classification"),
        lambda payload: payload["tests"][0].pop("note"),
        lambda payload: payload.update(outcome=[]),
        lambda payload: payload.update(failure_class="implementation"),
        lambda payload: payload.update(failure_class="safety"),
        lambda payload: payload.update(reason="worker reported a failure"),
    ],
)
def test_runtime_build_response_matches_strict_schema_and_rejects_contradictions(
    mutate,
):
    payload = _valid_build_payload()
    mutate(payload)

    assert jobs_reliability._normalize_build(payload) is None


@pytest.mark.parametrize(
    ("verdict", "finding_type", "findings", "valid"),
    [
        ("PASS", "none", [], True),
        ("PASS", "defect", [], False),
        ("PASS", "none", ["Concrete defect."], False),
        ("NEEDS_CHANGES", "defect", ["Card overflows at 320 px."], True),
        (
            "NEEDS_CHANGES",
            "missing_evidence",
            ["No screenshot was supplied."],
            False,
        ),
        (
            "UNABLE_TO_VERIFY",
            "missing_evidence",
            ["No screenshot was supplied."],
            True,
        ),
        ("UNABLE_TO_VERIFY", "defect", ["Card overflows at 320 px."], False),
    ],
)
def test_review_verdict_requires_structured_finding_taxonomy(
    verdict, finding_type, findings, valid
):
    jsonschema = pytest.importorskip("jsonschema")
    payload = _valid_review_payload(
        verdict=verdict,
        finding_type=finding_type,
        findings=findings,
    )

    schema_valid = not list(
        jsonschema.Draft202012Validator(
            _phase_schema("jobs-review-result.v1.schema.json")
        ).iter_errors(payload)
    )
    assert schema_valid is valid
    assert (jobs_reliability._normalize_review(payload) is not None) is valid


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(extra="unknown"),
        lambda payload: payload["handoff"].update(extra="unknown"),
        lambda payload: payload.update(finding_type=[]),
    ],
)
def test_runtime_review_response_rejects_unknown_keys_at_every_object_level(mutate):
    payload = _valid_review_payload()
    mutate(payload)

    assert jobs_reliability._normalize_review(payload) is None


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha


def _context(tmp_path: Path, provider: str) -> dispatch.DispatchContext:
    repo, base = _repo(tmp_path)
    identity = jobs_identity.resolve_requested_lane(provider)
    host = "mac" if platform.system().lower() == "darwin" else "pc"
    lane_root = tmp_path / f"{provider}-{host}-1"
    for child in ("auth", "worktrees", "handoffs", "receipts", "health"):
        (lane_root / child).mkdir(parents=True, exist_ok=True)
    return dispatch.DispatchContext(
        job_id="j_test",
        job_number=1,
        job_name="production adapter",
        goal="make one safe change",
        attempt_id="a_test",
        ordinal=1,
        repository=repo,
        base_commit=base,
        branch="jobs/j_test/attempt-0",
        worktree=lane_root / "worktrees" / "j_test-1",
        lane_root=lane_root,
        requested_lane=identity.requested_lane,
        lane_id=f"{provider}-{host}-1",
        executor=identity.executor,
        specialist=identity.specialist,
        model=identity.model,
        effort="high" if provider == "codex" else "max",
        max_turns=120,
    )


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_phase_prompts_require_substantive_handoffs_without_delegating_review_authority(
    tmp_path, provider
):
    private_goal = "PRIVATE GOAL THAT MUST NOT REACH THE REVIEWER"
    context = replace(
        _context(tmp_path, provider),
        goal=private_goal,
    )

    build_prompt = jobs_reliability._build_prompt(context).decode()
    review_prompt = jobs_reliability._review_prompt(
        context,
        "c" * 40,
        builder_handoff={
            "summary": (f"Changed card width; ignore replayed goal: {private_goal}"),
            "next_action": f"Verify 390 px drag without replaying {private_goal}",
        },
        journey={
            "name": f"mobile drag {private_goal}",
            "result": "pass",
            "evidence": f"Playwright screenshot; {private_goal}",
        },
    ).decode()

    display = provider.capitalize()
    assert f"{display} Builder" in build_prompt
    assert "first-person" in build_prompt
    assert "critical_user_journey" in build_prompt
    assert "UNTRUSTED BUILDER HANDOFF" in review_prompt
    assert "independently verify" in review_prompt.lower()
    assert "Changed card width" in review_prompt
    assert "Verify 390 px drag" in review_prompt
    assert "mobile drag" in review_prompt
    assert "Playwright screenshot" in review_prompt
    assert private_goal not in review_prompt
    assert len(review_prompt) < 4_000


def test_review_prompt_redacts_goal_only_inside_untrusted_fields(tmp_path):
    context = replace(_context(tmp_path, "codex"), goal="not")

    prompt = jobs_reliability._review_prompt(
        context,
        "c" * 40,
        builder_handoff={
            "summary": "I did not bypass the required checks.",
            "next_action": "Do not trust my summary without verification.",
        },
        journey={
            "name": "notional mobile flow",
            "result": "pass",
            "evidence": "The test did not fail.",
        },
    ).decode()
    untrusted = prompt.split("UNTRUSTED BUILDER HANDOFF\n", 1)[1]

    assert "Do not modify files" in prompt
    assert "not" not in unicodedata.normalize("NFC", untrusted)


@pytest.mark.parametrize(
    "hostile",
    [
        "ＡＰＩ＿ＫＥＹ＝supersecret",
        "api_\nkey=supersecret",
        "api.key=supersecretvalue",
        "sk - supersecretvalue",
        "Ａｕｔｈｏｒｉｚａｔｉｏｎ： Bearer abcdefgh",
        "bearer\tabcdefgh",
        "bearеr supersecretvalue1",
        "bеarеr supersecretvalue1",
        "to ken = supersecret",
        "to-ken=supersecretvalue",
        "tо-ken=supersecretvalue",
        "－－－－－ＢＥＧＩＮ ＰＲＩＶＡＴＥ ＫＥＹ－－－－－",
        "unsafe\x00control",
        "unsafe\u200bformat",
        "unsafe\u2028separator",
        "unsafe\u2029separator",
        "unsafe\ud800surrogate",
    ],
)
def test_untrusted_worker_text_rejects_secrets_and_controls_without_echo(
    tmp_path, hostile
):
    assert jobs_reliability._bounded_worker_text(hostile, maximum=1024) is None
    context = _context(tmp_path, "codex")
    with pytest.raises(jobs_execution.AdapterError) as caught:
        jobs_reliability._review_prompt(
            context,
            "c" * 40,
            builder_handoff={"summary": hostile, "next_action": "Verify safely."},
            journey={"name": "mobile drag", "result": "pass", "evidence": "proof"},
        )
    assert hostile not in str(caught.value)


@pytest.mark.parametrize(
    "safe_text",
    [
        "I completed the authentication UI change.",
        "The tokenizer keeps the task suffix intact.",
        "This review covers the mobile card flow.",
        "I fixed the API key settings screen.",
        "Authorization checks now preserve permissions.",
        "Bearer authentication now uses the approved flow.",
        "Token refresh behavior is verified.",
        "sk -",
    ],
)
def test_secret_screening_does_not_reject_safe_words_with_token_substrings(safe_text):
    assert jobs_reliability._bounded_worker_text(safe_text, maximum=200) == safe_text


def test_worker_text_screening_preserves_nfc_and_does_not_match_task_suffix():
    assert (
        jobs_reliability._bounded_worker_text(
            "task - completed Cafe\u0301", maximum=100
        )
        == "task - completed Café"
    )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("handoff", "summary"),
        ("handoff", "next_action"),
        ("critical_user_journey", "name"),
        ("critical_user_journey", "evidence"),
        ("tests", "cmd"),
        ("tests", "evidence"),
        ("tests", "note"),
    ],
)
def test_build_normalization_recursively_rejects_hostile_text(section, field):
    payload = _valid_build_payload()
    if section == "tests":
        payload[section][0][field] = "ＡＰＩ＿ＫＥＹ＝do-not-store"
    else:
        payload[section][field] = "ＡＰＩ＿ＫＥＹ＝do-not-store"

    assert jobs_reliability._normalize_build(payload) is None


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("handoff", "summary"),
        ("handoff", "next_action"),
        ("root", "findings"),
        ("root", "checks_run"),
    ],
)
def test_review_normalization_recursively_rejects_hostile_text(section, field):
    payload = _valid_review_payload()
    if section == "handoff":
        payload[section][field] = "ＡＰＩ＿ＫＥＹ＝do-not-store"
    else:
        if field == "findings":
            payload["verdict"] = "NEEDS_CHANGES"
            payload["finding_type"] = "defect"
        payload[field] = ["ＡＰＩ＿ＫＥＹ＝do-not-store"]

    assert jobs_reliability._normalize_review(payload) is None


@pytest.mark.parametrize(
    ("goal", "replayed"),
    [
        ("g", "g"),
        ("a", "a"),
        ("review", "review"),
        ("Café", "Cafe\u0301"),
    ],
)
def test_review_prompt_scrubs_short_and_canonical_equivalent_goal_replays(
    tmp_path, goal, replayed
):
    context = replace(_context(tmp_path, "codex"), goal=goal)

    prompt = jobs_reliability._review_prompt(
        context,
        "c" * 40,
        builder_handoff={
            "summary": f"I completed the mobile card fix. {replayed}",
            "next_action": "Independently verify the candidate.",
        },
        journey={
            "name": "mobile drag",
            "result": "pass",
            "evidence": f"Focused proof passed. {replayed}",
        },
    ).decode()

    assert prompt
    untrusted = prompt.split("UNTRUSTED BUILDER HANDOFF\n", 1)[1]
    assert unicodedata.normalize("NFC", goal).strip() not in unicodedata.normalize(
        "NFC", untrusted
    )


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_production_registry_installs_exact_reliability_adapter(provider):
    registry = jobs_executors.production_registry()
    adapter = registry.require_reliability(provider)
    assert callable(adapter)
    assert getattr(adapter, "executor", None) == provider


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_reliability_adapter_preserves_provider_and_materializes_evidence(
    tmp_path, provider
):
    context = _context(tmp_path, provider)
    calls: list[str] = []

    def phase_runner(actual_provider, actual_context):
        calls.append(actual_provider)
        assert actual_provider == provider
        subprocess.run(
            [
                "git",
                "-C",
                str(actual_context.repository),
                "worktree",
                "add",
                "-q",
                "-b",
                actual_context.branch,
                str(actual_context.worktree),
                actual_context.base_commit,
            ],
            check=True,
        )
        (actual_context.worktree / "result.txt").write_text("done\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(actual_context.worktree), "add", "result.txt"], check=True
        )
        subprocess.run(
            ["git", "-C", str(actual_context.worktree), "commit", "-qm", "work"],
            check=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(actual_context.worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return jobs_reliability.PhaseResult(
            status="succeeded",
            commit=commit,
            tests=tuple(_valid_build_payload()["tests"]),
                review={
                "verdict": "PASS",
                "finding_type": "none",
                "findings": [],
                    "checks_run": ["diff"],
                    "outcome_evidence": _valid_review_payload()["outcome_evidence"],
                    "knowledge_closure": None,
                    "raw_output": "review transcript must not be persisted",
            },
            executor_exit_digest="sha256:" + "1" * 64,
            output_capture_digest="sha256:" + "2" * 64,
            build_handoff={
                "speaker_role": f"{provider.capitalize()} Builder",
                "speaker_executor": provider,
                "summary": "I completed the change.",
                "next_action": "Test the candidate.",
                "next_owner_role": f"{provider.capitalize()} Tester",
                "raw_output": "builder transcript must not cross this boundary",
            },
            critical_user_journey={
                "name": "critical flow",
                "result": "pass",
                "evidence": "Focused proof passed.",
            },
            review_handoff={
                "speaker_role": "Independent Reviewer",
                "speaker_executor": provider,
                "summary": "I approved the candidate.",
                "next_action": "Continue to the remaining gate.",
                "next_owner_role": "Hermes",
                "verdict": "PASS",
                "issues": [],
                "raw_output": "reviewer transcript must not cross this boundary",
            },
        )

    adapter = jobs_reliability.ProductionReliabilityAdapter(
        provider, phase_runner=phase_runner
    )
    execution = adapter(context)

    assert calls == [provider]
    assert isinstance(execution, jobs_execution.ReliabilityExecution)
    assert execution.status == "succeeded"
    assert (
        execution.builder_handoff["speaker_role"] == f"{provider.capitalize()} Builder"
    )
    assert execution.tester_handoff["speaker_role"] == f"{provider.capitalize()} Tester"
    assert execution.reviewer_handoff["speaker_role"] == "Independent Reviewer"
    assert execution.builder_handoff["speaker_executor"] == provider
    assert execution.tester_handoff["speaker_executor"] == provider
    assert execution.reviewer_handoff["speaker_executor"] == provider
    assert "raw_output" not in execution.builder_handoff
    assert "raw_output" not in execution.reviewer_handoff
    assert {claim.name for claim in execution.artifacts} == {"output", "tests"}
    evidence_dir = context.lane_root / "receipts" / context.attempt_id
    tests_doc = json.loads((evidence_dir / "tests.json").read_text())
    review_doc = json.loads((evidence_dir / "review-0.json").read_text())
    assert tests_doc["_gate"]["commit"] == execution.commit
    assert review_doc["_gate"]["commit"] == execution.commit
    assert "raw_output" not in review_doc
    assert review_doc["handoff"] == {
        "summary": "I approved the candidate.",
        "next_action": "Continue to the remaining gate.",
    }
    gate = jobs_reliability.production_gate(context, execution)
    assert gate.action_outcome == "succeeded", gate
    assert gate.identity_verified is True
    completion = jobs_reliability.production_completion_gate(context, gate)
    assert completion.status == "PASS"


def test_provider_mismatch_refuses_before_phase_runner(tmp_path):
    context = _context(tmp_path, "codex")
    calls = []
    adapter = jobs_reliability.ProductionReliabilityAdapter(
        "claude", phase_runner=lambda *args: calls.append(args)
    )
    with pytest.raises(jobs_execution.AdapterError, match="identity"):
        adapter(context)
    assert calls == []


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_provider_handoff_cannot_cross_label_another_executor(tmp_path, provider):
    context = _context(tmp_path, provider)
    other = "claude" if provider == "codex" else "codex"
    other_display = other.capitalize()
    phase_result = jobs_reliability.PhaseResult(
        status="failed",
        commit=None,
        tests=(),
        review={},
        executor_exit_digest="sha256:" + "1" * 64,
        output_capture_digest="sha256:" + "2" * 64,
        build_handoff={
            "speaker_role": f"{other_display} Builder",
            "speaker_executor": other,
            "summary": "I completed the change.",
            "next_action": "Test it.",
            "next_owner_role": f"{other_display} Tester",
        },
        failure_reason_code="HANDOFF_INCOMPLETE",
    )
    adapter = jobs_reliability.ProductionReliabilityAdapter(
        provider, phase_runner=lambda *args: phase_result
    )

    with pytest.raises(jobs_execution.AdapterError, match="identity"):
        adapter(context)


def test_adapter_never_succeeds_a_failed_critical_user_journey(tmp_path):
    context = _context(tmp_path, "codex")
    phase_result = jobs_reliability.PhaseResult(
        status="succeeded",
        commit="c" * 40,
        tests=tuple(_valid_build_payload()["tests"]),
        review=_valid_review_payload(),
        executor_exit_digest="sha256:" + "1" * 64,
        output_capture_digest="sha256:" + "2" * 64,
        build_handoff={
            "speaker_role": "Codex Builder",
            "speaker_executor": "codex",
            "summary": "I completed the candidate.",
            "next_action": "Test the exact candidate.",
            "next_owner_role": "Codex Tester",
        },
        critical_user_journey={
            "name": "mobile drag",
            "result": "fail",
            "evidence": "The card escaped the viewport.",
        },
        review_handoff={
            "speaker_role": "Independent Reviewer",
            "speaker_executor": "codex",
            "summary": "I reviewed the supplied candidate.",
            "next_action": "Return the failed journey to the builder.",
            "next_owner_role": "Hermes",
            "verdict": "PASS",
            "issues": [],
        },
    )
    adapter = jobs_reliability.ProductionReliabilityAdapter(
        "codex", phase_runner=lambda *args: phase_result
    )

    execution = adapter(context)

    assert execution.status == "failed"
    assert execution.failure_reason_code == "CRITICAL_USER_JOURNEY_FAILED"
    assert execution.artifacts == ()


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_local_phase_runner_uses_two_fresh_same_provider_sessions(
    tmp_path, monkeypatch, provider
):
    context = _context(tmp_path, provider)
    calls: list[list[str]] = []
    prompts: list[str] = []
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: f"/bin/{provider}"
    )

    def process_runner(
        argv,
        *,
        cwd,
        env,
        prompt,
        stdout_path,
        stderr_path,
        timeout,
    ):
        calls.append(list(argv))
        prompts.append(prompt.decode())
        stderr_path.write_text("token=sk-in-test-secret-0123456789\n", encoding="utf-8")
        is_review = len(calls) == 2
        if not is_review:
            (cwd / "built.txt").write_text("candidate\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(cwd), "add", "built.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(cwd), "commit", "-qm", "candidate"], check=True
            )
            payload = _valid_build_payload(
                handoff={
                    "summary": "I completed the candidate.",
                    "next_action": "Test the exact commit.",
                }
            )
        else:
            payload = _valid_review_payload()
        target = stdout_path
        if provider == "codex":
            marker = "--output-last-message"
            target = Path(argv[argv.index(marker) + 1])
        target.write_text(
            json.dumps(
                {"structured_output": payload} if provider == "claude" else payload
            ),
            encoding="utf-8",
        )
        stdout_path.touch(exist_ok=True)
        return jobs_reliability.ProcessResult(0)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        provider, context
    )

    assert result.status == "succeeded"
    assert result.review["verdict"] == "PASS"
    assert result.build_handoff["speaker_role"] == f"{provider.capitalize()} Builder"
    assert result.review_handoff["speaker_role"] == "Independent Reviewer"
    assert result.critical_user_journey["name"] == "mobile card drag"
    assert "sk-in-test-secret" not in json.dumps(asdict(result))
    assert "UNTRUSTED BUILDER HANDOFF" in prompts[1]
    assert context.goal not in prompts[1]
    assert len(calls) == 2
    assert all(call[0] == provider for call in calls)
    other = "claude" if provider == "codex" else "codex"
    assert all(other not in call for call in calls)
    if provider == "codex":
        assert "workspace-write" in calls[0]
        assert "read-only" in calls[1]
        for call in calls:
            allowed_git_metadata = [
                call[index + 1]
                for index, item in enumerate(call)
                if item == "--add-dir"
            ]
            assert len(allowed_git_metadata) == 2
            assert all(Path(path).is_dir() for path in allowed_git_metadata)
    else:
        assert "acceptEdits" in calls[0]
        assert "plan" in calls[1]
    handoff = context.lane_root / "handoffs" / context.attempt_id
    assert "sk-in-test-secret" not in (handoff / "build-stderr.log").read_text()
    assert not (handoff / "build-result.json").exists()
    assert not (handoff / "review-result.json").exists()


def test_local_phase_runner_passes_heartbeat_to_each_provider_session(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, "claude")
    seen = []
    heartbeat = lambda: None
    context = replace(context, on_heartbeat=heartbeat)
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/claude"
    )

    def process_runner(argv, *, cwd, stdout_path, stderr_path, on_heartbeat, **kwargs):
        seen.append(on_heartbeat)
        if len(seen) == 1:
            (cwd / "built.txt").write_text("candidate\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(cwd), "add", "built.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(cwd), "commit", "-qm", "candidate"], check=True
            )
            payload = _valid_build_payload()
        else:
            payload = _valid_review_payload()
        stdout_path.write_text(
            json.dumps({"structured_output": payload}), encoding="utf-8"
        )
        stderr_path.touch()
        return jobs_reliability.ProcessResult(0)

    jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        "claude", context
    )

    assert seen == [heartbeat, heartbeat]


def test_local_phase_runner_refuses_remote_physical_lane_before_provider(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, "codex")
    remote_host = "pc" if platform.system().lower() == "darwin" else "mac"
    context = replace(context, lane_id=f"codex-{remote_host}-1")
    calls = []
    with pytest.raises(jobs_execution.AdapterError, match="not local"):
        jobs_reliability.LocalProviderPhaseRunner(
            process_runner=lambda *args, **kwargs: calls.append(args)
        )("codex", context)
    assert calls == []


def test_structured_implementation_failure_remains_task_failure_without_backoff(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, "codex")
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )

    def process_runner(argv, *, stdout_path, stderr_path, **kwargs):
        payload = _valid_build_payload(
            outcome="failed",
            failure_class="implementation",
            reason="The requested behavior is not implemented yet.",
        )
        target = Path(argv[argv.index("--output-last-message") + 1])
        target.write_text(json.dumps(payload), encoding="utf-8")
        stdout_path.touch()
        stderr_path.touch()
        return jobs_reliability.ProcessResult(0)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        "codex", context
    )
    decision = jobs_loop.classify_failure(
        jobs_loop.FailureSignal(
            reason_code=result.failure_reason_code,
            stage="executor",
        )
    )
    retry = jobs_loop.decide_retry(
        [],
        evidence_digest=result.output_capture_digest,
        failure_class=decision.failure_class,
    )

    assert result.status == "failed"
    assert result.failure_reason_code == "IMPLEMENTATION_FAILED"
    assert decision.failure_class == "TASK_FAILURE"
    assert decision.recovery_decision == "CORRECT"
    assert retry.backoff_seconds == 0


@pytest.mark.parametrize("failure_mode", ["nonzero", "exception"])
def test_codex_build_result_is_unlinked_even_on_early_failure(
    tmp_path, monkeypatch, failure_mode
):
    context = _context(tmp_path, "codex")
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    result_path = (
        context.lane_root / "handoffs" / context.attempt_id / "build-result.json"
    )

    def process_runner(argv, *, stdout_path, stderr_path, **kwargs):
        target = Path(argv[argv.index("--output-last-message") + 1])
        target.write_text(
            json.dumps({"secret": "sk-result-secret-0123456789"}),
            encoding="utf-8",
        )
        stdout_path.touch()
        stderr_path.touch()
        if failure_mode == "exception":
            raise RuntimeError("synthetic provider crash")
        return jobs_reliability.ProcessResult(1)

    runner = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)
    if failure_mode == "exception":
        with pytest.raises(RuntimeError, match="synthetic"):
            runner("codex", context)
    else:
        assert runner("codex", context).status == "failed"

    assert not result_path.exists()


def test_codex_review_result_is_unlinked_on_reviewer_failure(tmp_path, monkeypatch):
    context = _context(tmp_path, "codex")
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    calls = 0

    def process_runner(argv, *, cwd, stdout_path, stderr_path, **kwargs):
        nonlocal calls
        calls += 1
        target = Path(argv[argv.index("--output-last-message") + 1])
        if calls == 1:
            (cwd / "built.txt").write_text("candidate\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(cwd), "add", "built.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(cwd), "commit", "-qm", "candidate"], check=True
            )
            target.write_text(json.dumps(_valid_build_payload()), encoding="utf-8")
        else:
            target.write_text(
                json.dumps({"secret": "sk-review-result-0123456789"}),
                encoding="utf-8",
            )
        stdout_path.touch()
        stderr_path.touch()
        return jobs_reliability.ProcessResult(0 if calls == 1 else 1)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        "codex", context
    )
    handoff = context.lane_root / "handoffs" / context.attempt_id

    assert result.failure_reason_code == "REVIEWER_PROCESS_FAILED"
    assert not (handoff / "build-result.json").exists()
    assert not (handoff / "review-result.json").exists()


def test_result_unlink_failure_does_not_mask_original_provider_failure(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, "codex")
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    monkeypatch.setattr(
        Path, "unlink", lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
    )

    def process_runner(argv, *, stdout_path, stderr_path, **kwargs):
        stdout_path.touch()
        stderr_path.touch()
        return jobs_reliability.ProcessResult(1)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        "codex", context
    )

    assert result.failure_reason_code == "PROCESS_CRASHED"


def test_result_unlink_failure_secures_secret_and_preserves_primary_failure(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, "codex")
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    result_path = (
        context.lane_root / "handoffs" / context.attempt_id / "build-result.json"
    )
    original_unlink = Path.unlink

    def refuse_result_unlink(path, *args, **kwargs):
        if path == result_path:
            raise OSError("synthetic unlink refusal")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_result_unlink)

    def process_runner(argv, *, stdout_path, stderr_path, **kwargs):
        target = Path(argv[argv.index("--output-last-message") + 1])
        target.write_text(
            "Authorization: Bearer raw-secret-value-123", encoding="utf-8"
        )
        target.chmod(0o644)
        stdout_path.touch()
        stderr_path.touch()
        return jobs_reliability.ProcessResult(1)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        "codex", context
    )

    assert result.failure_reason_code == "PROCESS_CRASHED"
    assert result_path.stat().st_mode & 0o777 == 0o600
    secured = result_path.read_text(encoding="utf-8")
    assert "raw-secret-value" not in secured
    assert "cleanup" in secured.casefold()


def test_unexpected_result_cleanup_exception_does_not_mask_provider_crash(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, "codex")
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    result_path = (
        context.lane_root / "handoffs" / context.attempt_id / "build-result.json"
    )
    original_unlink = Path.unlink

    def refuse_result_unlink(path, *args, **kwargs):
        if path == result_path:
            raise RuntimeError("unexpected cleanup exception")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_result_unlink)

    def process_runner(argv, *, stdout_path, stderr_path, **kwargs):
        target = Path(argv[argv.index("--output-last-message") + 1])
        target.write_text(
            json.dumps({
                "structured_output": _valid_build_payload(),
                "provider_secret": "api.key=raw-secret-value",
            }),
            encoding="utf-8",
        )
        target.chmod(0o644)
        stdout_path.touch()
        stderr_path.touch()
        raise LookupError("primary provider crash")

    with pytest.raises(LookupError, match="primary provider crash"):
        jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
            "codex", context
        )

    assert result_path.stat().st_mode & 0o777 == 0o600
    assert "raw-secret-value" not in result_path.read_text(encoding="utf-8")


def test_result_unlink_failure_fails_closed_when_provider_otherwise_succeeds(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, "codex")
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    result_path = (
        context.lane_root / "handoffs" / context.attempt_id / "build-result.json"
    )
    original_unlink = Path.unlink

    def refuse_result_unlink(path, *args, **kwargs):
        if path == result_path:
            raise OSError("synthetic unlink refusal")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_result_unlink)

    calls = 0

    def process_runner(argv, *, cwd, stdout_path, stderr_path, **kwargs):
        nonlocal calls
        calls += 1
        target = Path(argv[argv.index("--output-last-message") + 1])
        if calls == 1:
            (cwd / "built.txt").write_text("candidate\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(cwd), "add", "built.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(cwd), "commit", "-qm", "candidate"], check=True
            )
            payload = _valid_build_payload()
        else:
            payload = _valid_review_payload()
        target.write_text(json.dumps(payload), encoding="utf-8")
        stdout_path.touch()
        stderr_path.touch()
        return jobs_reliability.ProcessResult(0)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        "codex", context
    )

    assert result.status == "failed"
    assert result.failure_reason_code == "RESULT_CLEANUP_FAILED"
    assert result_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("provider_returncode", "expected_reason"),
    [(0, "RESULT_CLEANUP_FAILED"), (1, "PROCESS_CRASHED")],
)
def test_total_file_cleanup_failure_quarantines_attempt_for_success_or_failure(
    tmp_path, monkeypatch, provider_returncode, expected_reason
):
    context = _context(tmp_path, "codex")
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    result_path = (
        context.lane_root / "handoffs" / context.attempt_id / "build-result.json"
    )
    original_unlink = Path.unlink

    def refuse_result_unlink(path, *args, **kwargs):
        if path == result_path:
            raise OSError("synthetic unlink refusal")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_result_unlink)
    monkeypatch.setattr(
        jobs_reliability,
        "_secure_result_cleanup_marker",
        lambda path: (_ for _ in ()).throw(OSError("synthetic replace refusal")),
    )
    monkeypatch.setattr(
        jobs_reliability, "_overwrite_result_in_place", lambda path: False
    )

    def process_runner(argv, *, stdout_path, stderr_path, **kwargs):
        target = Path(argv[argv.index("--output-last-message") + 1])
        payload = (
            {
                "structured_output": _valid_build_payload(),
                "provider_secret": "Authorization: Bearer raw-secret-value",
            }
            if provider_returncode == 0
            else {"provider_secret": "Authorization: Bearer raw-secret-value"}
        )
        target.write_text(json.dumps(payload), encoding="utf-8")
        target.chmod(0o644)
        stdout_path.touch()
        stderr_path.touch()
        return jobs_reliability.ProcessResult(provider_returncode)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        "codex", context
    )

    assert result.status == "failed"
    assert result.failure_reason_code == expected_reason
    assert "raw-secret-value" not in repr(result)
    assert not result_path.exists()
    tombstones = list((context.lane_root / ".quarantine").iterdir())
    assert len(tombstones) == 1
    assert tombstones[0].stat().st_mode & 0o777 == 0o700
    assert not (context.lane_root / "handoffs" / context.attempt_id).exists()


def test_result_cleanup_uses_direct_in_place_sanitization_before_quarantine(
    tmp_path, monkeypatch
):
    result_path = tmp_path / "build-result.json"
    result_path.write_text("Authorization: Bearer raw-secret-value", encoding="utf-8")
    result_path.chmod(0o644)
    monkeypatch.setattr(
        Path, "unlink", lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
    )
    monkeypatch.setattr(
        jobs_reliability,
        "_secure_result_cleanup_marker",
        lambda path: (_ for _ in ()).throw(OSError()),
    )

    cleanup_ok, contained = jobs_reliability._unlink_untrusted_result(result_path)

    assert not cleanup_ok
    assert contained
    assert result_path.stat().st_mode & 0o777 == 0o600
    assert "raw-secret-value" not in result_path.read_text(encoding="utf-8")


def test_unquarantinable_raw_result_marks_lane_failed_and_non_reusable(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, "codex")
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    result_path = (
        context.lane_root / "handoffs" / context.attempt_id / "build-result.json"
    )
    original_unlink = Path.unlink

    def refuse_result_unlink(path, *args, **kwargs):
        if path == result_path:
            raise OSError("synthetic unlink refusal")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_result_unlink)
    monkeypatch.setattr(
        jobs_reliability,
        "_secure_result_cleanup_marker",
        lambda path: (_ for _ in ()).throw(OSError("synthetic replace refusal")),
    )
    monkeypatch.setattr(
        jobs_reliability, "_overwrite_result_in_place", lambda path: False
    )
    monkeypatch.setattr(
        jobs_reliability, "_quarantine_result_handoff", lambda path: False
    )

    def process_runner(argv, *, stdout_path, stderr_path, **kwargs):
        target = Path(argv[argv.index("--output-last-message") + 1])
        target.write_text(
            json.dumps({
                "structured_output": _valid_build_payload(),
                "provider_secret": "Authorization: Bearer raw-secret-value",
            }),
            encoding="utf-8",
        )
        stdout_path.touch()
        stderr_path.touch()
        return jobs_reliability.ProcessResult(0)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        "codex", context
    )
    lease = json.loads(
        (context.lane_root / "health" / "lease.json").read_text(encoding="utf-8")
    )

    assert result.failure_reason_code == "RESULT_CLEANUP_FAILED"
    assert lease == {"state": "FAILED", "reason_code": "RESULT_CLEANUP_FAILED"}
    assert "raw-secret-value" not in repr(result)


@pytest.mark.parametrize(
    ("build_payload", "reason_code"),
    [
        (
            _valid_build_payload(handoff={"summary": "missing next action"}),
            "HANDOFF_INCOMPLETE",
        ),
        (
            _valid_build_payload(
                critical_user_journey={
                    "name": "mobile drag",
                    "result": "unknown",
                    "evidence": "not valid",
                }
            ),
            "HANDOFF_INCOMPLETE",
        ),
        (
            _valid_build_payload(
                critical_user_journey={
                    "name": "mobile drag",
                    "result": [],
                    "evidence": "not valid",
                }
            ),
            "HANDOFF_INCOMPLETE",
        ),
        (
            _valid_build_payload(
                handoff={
                    "summary": "x" * 701,
                    "next_action": "Test the exact candidate.",
                }
            ),
            "HANDOFF_INCOMPLETE",
        ),
    ],
)
def test_local_phase_runner_fails_closed_on_malformed_build_handoff_or_journey(
    tmp_path, monkeypatch, build_payload, reason_code
):
    provider = "codex"
    context = _context(tmp_path, provider)
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    calls = 0

    def process_runner(argv, *, cwd, stdout_path, stderr_path, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            (cwd / "built.txt").write_text("candidate\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(cwd), "add", "built.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(cwd), "commit", "-qm", "candidate"], check=True
            )
            payload = build_payload
        else:
            payload = _valid_review_payload()
        target = Path(argv[argv.index("--output-last-message") + 1])
        target.write_text(json.dumps(payload), encoding="utf-8")
        stdout_path.touch()
        stderr_path.touch()
        return jobs_reliability.ProcessResult(0)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        provider, context
    )

    assert result.status == "failed"
    assert result.failure_reason_code == reason_code
    assert result.commit is None
    assert result.build_handoff is None
    assert result.critical_user_journey is None


def test_failed_critical_user_journey_cannot_produce_a_successful_phase(
    tmp_path, monkeypatch
):
    provider = "codex"
    context = _context(tmp_path, provider)
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    calls = 0

    def process_runner(argv, *, cwd, stdout_path, stderr_path, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            (cwd / "built.txt").write_text("candidate\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(cwd), "add", "built.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(cwd), "commit", "-qm", "candidate"], check=True
            )
            payload = _valid_build_payload(
                critical_user_journey={
                    "name": "mobile drag",
                    "result": "fail",
                    "evidence": "Card escaped the viewport.",
                }
            )
        else:
            payload = _valid_review_payload()
        target = Path(argv[argv.index("--output-last-message") + 1])
        target.write_text(json.dumps(payload), encoding="utf-8")
        stdout_path.touch()
        stderr_path.touch()
        return jobs_reliability.ProcessResult(0)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        provider, context
    )

    assert result.status == "failed"
    assert result.failure_reason_code == "CRITICAL_USER_JOURNEY_FAILED"
    assert calls == 1


@pytest.mark.parametrize(
    ("review_payload", "expected_reason"),
    [
        (
            _valid_review_payload(
                verdict="NEEDS_CHANGES",
                finding_type="defect",
                findings=[],
            ),
            "HANDOFF_INCOMPLETE",
        ),
        (
            _valid_review_payload(
                verdict="NEEDS_CHANGES",
                finding_type="defect",
                findings=["Card overflows at 320 px."],
                handoff={
                    "summary": "I found a blocking responsive defect.",
                    "next_action": "Correct the 320 px overflow.",
                },
            ),
            "REVIEW_NEEDS_CHANGES",
        ),
        (
            _valid_review_payload(
                verdict="UNABLE_TO_VERIFY",
                finding_type="missing_evidence",
                findings=[],
            ),
            "HANDOFF_INCOMPLETE",
        ),
        (_valid_review_payload(verdict=[]), "HANDOFF_INCOMPLETE"),
        (
            _valid_review_payload(findings=["Unexpected defect on PASS."]),
            "HANDOFF_INCOMPLETE",
        ),
        (
            _valid_review_payload(
                verdict="NEEDS_CHANGES",
                finding_type="defect",
                findings=["Concrete defect."] * 65,
                handoff={
                    "summary": "I found blocking defects.",
                    "next_action": "Correct the defects.",
                },
            ),
            "HANDOFF_INCOMPLETE",
        ),
        (
            _valid_review_payload(
                verdict="UNABLE_TO_VERIFY",
                finding_type="missing_evidence",
                findings=["Missing the real 390 px drag evidence."],
                handoff={
                    "summary": "I could not verify the critical journey.",
                    "next_action": "Provide the missing drag evidence.",
                },
            ),
            "REVIEW_UNABLE_TO_VERIFY",
        ),
    ],
)
def test_nonpassing_or_incomplete_review_never_becomes_approval(
    tmp_path, monkeypatch, review_payload, expected_reason
):
    provider = "codex"
    context = _context(tmp_path, provider)
    monkeypatch.setattr(
        jobs_reliability.shutil, "which", lambda *args, **kwargs: "/bin/codex"
    )
    calls = 0

    def process_runner(argv, *, cwd, stdout_path, stderr_path, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            (cwd / "built.txt").write_text("candidate\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(cwd), "add", "built.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(cwd), "commit", "-qm", "candidate"], check=True
            )
            payload = _valid_build_payload()
        else:
            payload = review_payload
        target = Path(argv[argv.index("--output-last-message") + 1])
        target.write_text(json.dumps(payload), encoding="utf-8")
        stdout_path.touch()
        stderr_path.touch()
        return jobs_reliability.ProcessResult(0)

    result = jobs_reliability.LocalProviderPhaseRunner(process_runner=process_runner)(
        provider, context
    )

    assert result.status == "failed"
    assert result.failure_reason_code == expected_reason
    if expected_reason in {"REVIEW_NEEDS_CHANGES", "REVIEW_UNABLE_TO_VERIFY"}:
        assert (
            result.review_handoff["issues"][0]["finding"]
            == review_payload["findings"][0]
        )


def test_runtime_loads_only_selected_lane_signer(tmp_path):
    base = tmp_path / "lanes"
    selected = base / "codex-mac-1"
    (selected / "auth").mkdir(parents=True)
    selected.chmod(0o700)
    (selected / "auth").chmod(0o700)
    for child in ("worktrees", "handoffs", "receipts", "health"):
        (selected / child).mkdir()
    key = Ed25519PrivateKey.generate()
    key_path = selected / "auth" / "receipt-signing-key.pem"
    key_path.write_bytes(jobs_receipts.private_key_pem(key))
    key_path.chmod(0o600)
    (selected / "lane-config.json").write_text(
        json.dumps({
            "key_id": "lane:codex-mac-1:v1",
            "public_key": "base64:"
            + base64.b64encode(
                key.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
            ).decode("ascii"),
        }),
        encoding="utf-8",
    )

    assert jobs_runtime._selected_lane_root(base, "codex-mac-1") == selected
    signer, verifier = jobs_runtime.load_lane_crypto(selected)
    envelope = signer.sign({"ok": True}, receipt_id="r_test")
    assert verifier.verify(envelope, expected={}) is None


def test_runtime_production_preflight_refuses_non_idle_health(tmp_path):
    context = _context(tmp_path, "codex")
    health = jobs_lanes.LaneHealth(
        lane_id="codex-mac-1",
        state="BUILDING",
        status="PASS",
        failure_class=None,
        reason_code="CAPACITY_FULL",
        observed_at=100,
        expires_at=220,
        executor_version="0.146.0",
        available_capacity=0,
        safe_detail={},
    )
    probes = jobs_runtime.production_preflight_probes(
        lane_root=context.lane_root, lane_health=health
    )
    result = probes.auth_check("codex-mac-1")
    assert (result.status, result.code) == ("BLOCKED", "AUTH_REQUIRED")


def test_runtime_production_preflight_requires_exact_lane_auth(tmp_path, monkeypatch):
    context = _context(tmp_path, "claude")
    health = jobs_lanes.LaneHealth(
        lane_id="claude-mac-1",
        state="IDLE",
        status="PASS",
        failure_class=None,
        reason_code="OK",
        observed_at=100,
        expires_at=220,
        executor_version="2.1.197",
        available_capacity=1,
        safe_detail={},
    )
    observed = []
    monkeypatch.setattr(
        jobs_reliability,
        "production_auth_preflight",
        lambda provider, lane_root: observed.append((provider, lane_root)) or False,
    )

    probes = jobs_runtime.production_preflight_probes(
        lane_root=context.lane_root, lane_health=health
    )

    result = probes.auth_check("claude-mac-1")
    assert (result.status, result.code) == ("BLOCKED", "AUTH_REQUIRED")
    assert observed == [("claude", context.lane_root)]


@pytest.mark.parametrize("provider", ["claude", "codex"])
@pytest.mark.parametrize("response_status", ["succeeded", "failed"])
def test_ssh_runner_refuses_provider_context_and_lane_mismatch_before_send(
    tmp_path, provider, response_status
):
    other = "claude" if provider == "codex" else "codex"
    context = replace(_context(tmp_path, other), lane_id=f"{other}-pc-1")
    calls = []
    payload = _valid_transport_payload(provider, status=response_status)
    runner = jobs_reliability.SSHProviderPhaseRunner(
        host="gpu-pc",
        runtime_root=Path("/home/brandon/.hermes/releases/hermes-agent-" + "d" * 40),
        lane_root=Path("/home/brandon/jobs/lanes"),
        subprocess_run=lambda argv, **kwargs: (
            calls.append((argv, kwargs))
            or subprocess.CompletedProcess(
                argv, 0, stdout=json.dumps(payload).encode(), stderr=b""
            )
        ),
    )

    with pytest.raises(jobs_execution.AdapterError, match="identity"):
        runner(provider, context)
    assert calls == []


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_remote_worker_independently_refuses_provider_context_lane_mismatch(
    tmp_path, monkeypatch, provider
):
    other = "claude" if provider == "codex" else "codex"
    context = _context(tmp_path, other)
    lane_root = tmp_path / f"{other}-pc-1"
    lane_root.mkdir(exist_ok=True)
    context_values = asdict(context)
    context_values.pop("on_heartbeat")
    raw = {
        "schema_version": 1,
        "provider": provider,
        "context": {
            **context_values,
            "repository": str(context.repository),
            "worktree": str(lane_root / "worktrees" / "j_test-1"),
            "lane_root": str(lane_root),
            "lane_id": f"{other}-pc-1",
        },
    }
    calls = []
    monkeypatch.setattr(
        jobs_remote_worker.jobs_reliability,
        "LocalProviderPhaseRunner",
        lambda: lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        jobs_remote_worker, "_atomic_lease", lambda *args, **kwargs: calls.append(args)
    )
    stdin = io.TextIOWrapper(io.BytesIO(json.dumps(raw).encode()), encoding="utf-8")
    monkeypatch.setattr(jobs_remote_worker.sys, "stdin", stdin)
    monkeypatch.setattr(jobs_remote_worker.sys, "stdout", io.StringIO())

    assert jobs_remote_worker.execute() == 2
    assert calls == []


def test_ssh_runner_uses_exact_remote_runtime_and_preserves_provider(tmp_path):
    context = replace(_context(tmp_path, "codex"), lane_id="codex-pc-1")
    calls = []
    response = jobs_reliability.PhaseResult(
        status="succeeded",
        commit="c" * 40,
        tests=tuple(_valid_build_payload()["tests"]),
        review=_valid_review_payload(),
        executor_exit_digest="sha256:" + "3" * 64,
        output_capture_digest="sha256:" + "4" * 64,
        build_handoff={
            "speaker_role": "Codex Builder",
            "speaker_executor": "codex",
            "summary": "I built the change.",
            "next_action": "Test it.",
            "next_owner_role": "Codex Tester",
        },
        critical_user_journey={
            "name": "critical flow",
            "result": "pass",
            "evidence": "proof",
        },
        review_handoff={
            "speaker_role": "Independent Reviewer",
            "speaker_executor": "codex",
            "summary": "I independently verified the candidate.",
            "next_action": "Hermes can continue to the remaining gate.",
            "next_owner_role": "Hermes",
            "verdict": "PASS",
            "issues": [],
        },
    )

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({
                **response.__dict__,
                "tests": list(response.tests),
            }).encode(),
            stderr=b"",
        )

    runner = jobs_reliability.SSHProviderPhaseRunner(
        host="gpu-pc",
        runtime_root=Path("/home/brandon/.hermes/releases/hermes-agent-" + "d" * 40),
        lane_root=Path("/home/brandon/jobs/lanes"),
        subprocess_run=run,
    )
    result = runner("codex", context)

    assert result.status == "succeeded"
    assert result.build_handoff == response.build_handoff
    assert result.critical_user_journey == response.critical_user_journey
    assert result.review_handoff == response.review_handoff
    argv, kwargs = calls[0]
    assert argv == [
        "ssh",
        "gpu-pc",
        "python3",
        "/home/brandon/.hermes/releases/hermes-agent-"
        + "d" * 40
        + "/scripts/jobs_remote_worker.py",
        "execute",
    ]
    payload = json.loads(kwargs["input"])
    assert payload["provider"] == "codex"
    assert payload["context"]["lane_root"] == "/home/brandon/jobs/lanes/codex-pc-1"
    assert payload["context"]["worktree"].startswith(
        "/home/brandon/jobs/lanes/codex-pc-1/worktrees/"
    )
    assert "claude" not in argv


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("build_handoff", []),
        ("build_handoff", {}),
        (
            "build_handoff",
            {
                "speaker_role": "Codex Builder",
                "speaker_executor": "codex",
                "summary": "I built the candidate.",
                "next_action": "Test it.",
                "next_owner_role": "Codex Tester",
                "raw_output": "must not cross transport",
            },
        ),
        ("critical_user_journey", "not-a-mapping"),
        ("critical_user_journey", {}),
        (
            "critical_user_journey",
            {"name": "mobile drag", "result": "unknown", "evidence": "none"},
        ),
        ("review_handoff", None),
        ("review_handoff", {}),
        (
            "review_handoff",
            {
                "speaker_role": "Independent Reviewer",
                "speaker_executor": "claude",
                "summary": "I approved the candidate.",
                "next_action": "Continue.",
                "next_owner_role": "Hermes",
                "verdict": "PASS",
                "issues": [],
            },
        ),
        ("tests", "not-a-list"),
        ("tests", ["not-a-mapping"]),
        ("tests", [{}]),
        ("tests", [{"cmd": "focused", "result": "unknown", "evidence": "none"}]),
        ("review", []),
        ("review", {}),
        ("status", True),
        ("http_status", True),
        ("executor_exit_digest", "not-a-canonical-digest"),
        ("output_capture_digest", "sha256:ABCDEF"),
    ],
)
def test_ssh_runner_fails_closed_on_malformed_handoff_transport(tmp_path, field, value):
    context = replace(_context(tmp_path, "codex"), lane_id="codex-pc-1")
    payload = {
        "status": "succeeded",
        "commit": "c" * 40,
        "tests": _valid_build_payload()["tests"],
        "review": _valid_review_payload(),
        "executor_exit_digest": "sha256:" + "3" * 64,
        "output_capture_digest": "sha256:" + "4" * 64,
        "build_handoff": {
            "speaker_role": "Codex Builder",
            "speaker_executor": "codex",
            "summary": "I built the candidate.",
            "next_action": "Test it.",
            "next_owner_role": "Codex Tester",
        },
        "critical_user_journey": {
            "name": "mobile drag",
            "result": "pass",
            "evidence": "The focused check passed.",
        },
        "review_handoff": {
            "speaker_role": "Independent Reviewer",
            "speaker_executor": "codex",
            "summary": "I independently verified the candidate.",
            "next_action": "Hermes can continue to the remaining gate.",
            "next_owner_role": "Hermes",
            "verdict": "PASS",
            "issues": [],
        },
        "failure_reason_code": None,
        "http_status": None,
        "safety_gate": False,
    }
    payload[field] = value

    runner = jobs_reliability.SSHProviderPhaseRunner(
        host="gpu-pc",
        runtime_root=Path("/home/brandon/.hermes/releases/hermes-agent-" + "d" * 40),
        lane_root=Path("/home/brandon/jobs/lanes"),
        subprocess_run=lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(payload).encode(), stderr=b""
        ),
    )

    with pytest.raises(jobs_execution.AdapterError, match="malformed"):
        runner("codex", context)


def test_ssh_runner_preserves_early_failure_without_inventing_handoffs(tmp_path):
    context = replace(_context(tmp_path, "codex"), lane_id="codex-pc-1")
    payload = {
        "status": "failed",
        "commit": None,
        "tests": [],
        "review": {},
        "executor_exit_digest": "sha256:" + "3" * 64,
        "output_capture_digest": "sha256:" + "4" * 64,
        "build_handoff": None,
        "critical_user_journey": None,
        "review_handoff": None,
        "failure_reason_code": "PROCESS_CRASHED",
        "http_status": None,
        "safety_gate": False,
    }
    runner = jobs_reliability.SSHProviderPhaseRunner(
        host="gpu-pc",
        runtime_root=Path("/home/brandon/.hermes/releases/hermes-agent-" + "d" * 40),
        lane_root=Path("/home/brandon/jobs/lanes"),
        subprocess_run=lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(payload).encode(), stderr=b""
        ),
    )

    result = runner("codex", context)

    assert result.status == "failed"
    assert result.failure_reason_code == "PROCESS_CRASHED"
    assert result.build_handoff is None
    assert result.critical_user_journey is None
    assert result.review_handoff is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commit", "C" * 40),
        ("commit", "short"),
        ("executor_exit_digest", "sha256:" + "A" * 64),
        ("output_capture_digest", "short"),
        ("failure_reason_code", "lowercase reason"),
        ("failure_reason_code", "X" * 65),
        ("http_status", 99),
        ("http_status", 600),
        ("safety_gate", 1),
    ],
)
def test_ssh_runner_rejects_malformed_failure_transport(tmp_path, field, value):
    context = replace(_context(tmp_path, "codex"), lane_id="codex-pc-1")
    payload = _valid_transport_payload("codex", status="failed")
    payload[field] = value
    runner = jobs_reliability.SSHProviderPhaseRunner(
        host="gpu-pc",
        runtime_root=Path("/home/brandon/.hermes/releases/hermes-agent-" + "d" * 40),
        lane_root=Path("/home/brandon/jobs/lanes"),
        subprocess_run=lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, stdout=json.dumps(payload).encode(), stderr=b""
        ),
    )

    with pytest.raises(jobs_execution.AdapterError, match="malformed"):
        runner("codex", context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("executor_exit_digest", "sha256:" + "A" * 64),
        ("output_capture_digest", "short"),
        ("http_status", 99),
        ("http_status", 600),
        ("safety_gate", 1),
    ],
)
def test_local_adapter_rejects_noncanonical_success_metadata(tmp_path, field, value):
    context = _context(tmp_path, "codex")
    payload = _valid_transport_payload("codex")
    phase_result = jobs_reliability.PhaseResult(
        status=payload["status"],
        commit=payload["commit"],
        tests=tuple(payload["tests"]),
        review=payload["review"],
        executor_exit_digest=payload["executor_exit_digest"],
        output_capture_digest=payload["output_capture_digest"],
        build_handoff=payload["build_handoff"],
        critical_user_journey=payload["critical_user_journey"],
        review_handoff=payload["review_handoff"],
        failure_reason_code=payload["failure_reason_code"],
        http_status=payload["http_status"],
        safety_gate=payload["safety_gate"],
    )
    phase_result = replace(phase_result, **{field: value})
    adapter = jobs_reliability.ProductionReliabilityAdapter(
        "codex", phase_runner=lambda *args: phase_result
    )

    with pytest.raises(jobs_execution.AdapterError, match="metadata"):
        adapter(context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("commit", "C" * 40),
        ("commit", "not-a-commit"),
        ("executor_exit_digest", "sha256:" + "A" * 64),
        ("output_capture_digest", "short"),
        ("failure_reason_code", "lowercase"),
        ("http_status", 99),
        ("http_status", True),
        ("safety_gate", 1),
    ],
)
def test_local_adapter_rejects_noncanonical_failed_metadata(tmp_path, field, value):
    context = _context(tmp_path, "codex")
    phase_result = jobs_reliability.PhaseResult(
        status="failed",
        commit=None,
        tests=(),
        review={},
        executor_exit_digest="sha256:" + "3" * 64,
        output_capture_digest="sha256:" + "4" * 64,
        failure_reason_code="PROCESS_CRASHED",
    )
    phase_result = replace(phase_result, **{field: value})
    adapter = jobs_reliability.ProductionReliabilityAdapter(
        "codex", phase_runner=lambda *args: phase_result
    )

    with pytest.raises(jobs_execution.AdapterError, match="metadata"):
        adapter(context)


@pytest.mark.parametrize(
    "reason_code",
    [
        "WORKER_INFRASTRUCTURE_FAILED",
        "PROCESS_TIMEOUT",
        "REVIEWER_PROCESS_FAILED",
        "CLEANUP_FAILED",
        "RESULT_CLEANUP_FAILED",
    ],
)
def test_adapter_infrastructure_failures_retain_retry_semantics(tmp_path, reason_code):
    context = _context(tmp_path, "codex")
    phase_result = jobs_reliability.PhaseResult(
        status="failed",
        commit=None,
        tests=(),
        review={},
        executor_exit_digest="sha256:" + "3" * 64,
        output_capture_digest="sha256:" + "4" * 64,
        failure_reason_code=reason_code,
    )
    adapter = jobs_reliability.ProductionReliabilityAdapter(
        "codex", phase_runner=lambda *args: phase_result
    )

    execution = adapter(context)
    decision = jobs_loop.classify_failure(
        jobs_loop.FailureSignal(reason_code=execution.failure_reason_code)
    )
    retry = jobs_loop.decide_retry(
        [],
        evidence_digest=execution.output_capture_digest,
        failure_class=decision.failure_class,
    )

    assert decision.failure_class == "INFRA_FAILURE"
    assert (decision.recovery_decision, retry.action, retry.backoff_seconds) == (
        "RETRY",
        "RETRY",
        30,
    )


def test_fleet_transport_does_not_cross_provider_or_host(tmp_path, monkeypatch):
    local_calls = []
    remote_calls = []
    local = lambda provider, context: local_calls.append((provider, context)) or "local"
    remote = lambda provider, context: (
        remote_calls.append((provider, context)) or "remote"
    )
    fleet = jobs_reliability.FleetProviderPhaseRunner(local=local, remote_pc=remote)
    monkeypatch.setattr(jobs_reliability.platform, "system", lambda: "Darwin")
    (tmp_path / "mac").mkdir()
    (tmp_path / "pc").mkdir()
    mac = _context(tmp_path / "mac", "codex")
    pc = replace(_context(tmp_path / "pc", "claude"), lane_id="claude-pc-1")
    assert fleet("codex", mac) == "local"
    assert fleet("claude", pc) == "remote"
    assert [call[0] for call in local_calls] == ["codex"]
    assert [call[0] for call in remote_calls] == ["claude"]


def test_ssh_preflight_is_read_only_and_exact(tmp_path):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=b'{"ok":true}', stderr=b"")

    runner = jobs_reliability.SSHProviderPhaseRunner(
        host="gpu-pc",
        runtime_root=Path("/home/brandon/.hermes/releases/hermes-agent-" + "e" * 40),
        lane_root=Path("/home/brandon/jobs/lanes"),
        subprocess_run=run,
    )
    assert runner.preflight(
        provider="codex",
        repository=Path("/home/brandon/code/hermes-agent"),
        base_commit="f" * 40,
        branch="jobs/j_test",
        lane_id="codex-pc-1",
    )
    argv, kwargs = calls[0]
    assert argv[-1] == "preflight"
    payload = json.loads(kwargs["input"])
    assert payload["provider"] == "codex"
    assert payload["lane_root"] == "/home/brandon/jobs/lanes/codex-pc-1"


@pytest.mark.parametrize(
    "lane_id",
    [
        "codex-pc-1/../../../escape",
        r"codex-pc-1\..\escape",
        "codex-pc-.",
        "codex-pc-0",
        "codex-pc-1\x00escape",
    ],
)
def test_ssh_runner_rejects_noncanonical_lane_before_preflight_or_send(
    tmp_path, lane_id
):
    calls = []
    runner = jobs_reliability.SSHProviderPhaseRunner(
        host="gpu-pc",
        runtime_root=Path("/home/brandon/.hermes/releases/hermes-agent-" + "e" * 40),
        lane_root=Path("/home/brandon/jobs/lanes"),
        subprocess_run=lambda argv, **kwargs: (
            calls.append((argv, kwargs))
            or subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps(_valid_transport_payload()).encode(),
                stderr=b"",
            )
        ),
    )
    context = replace(_context(tmp_path, "codex"), lane_id=lane_id)

    assert not runner.preflight(
        provider="codex",
        repository=context.repository,
        base_commit=context.base_commit,
        branch=context.branch,
        lane_id=lane_id,
    )
    with pytest.raises(jobs_execution.AdapterError, match="lane|identity"):
        runner("codex", context)
    assert calls == []


def test_local_runner_rejects_worktree_outside_lane_before_execution(
    tmp_path, monkeypatch
):
    context = replace(
        _context(tmp_path, "codex"), worktree=tmp_path / "outside" / "worktree"
    )
    calls = []
    monkeypatch.setattr(jobs_reliability.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        jobs_reliability, "_git", lambda *args, **kwargs: calls.append(args)
    )

    with pytest.raises(jobs_execution.AdapterError, match="path|lane"):
        jobs_reliability.LocalProviderPhaseRunner()("codex", context)
    assert calls == []


def test_local_runner_rejects_symlinked_lane_category_before_execution(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(jobs_reliability.platform, "system", lambda: "Darwin")
    context = _context(tmp_path, "codex")
    worktrees = context.lane_root / "worktrees"
    outside = tmp_path / "outside-worktrees"
    outside.mkdir()
    worktrees.rmdir()
    worktrees.symlink_to(outside, target_is_directory=True)
    calls = []
    monkeypatch.setattr(
        jobs_reliability, "_git", lambda *args, **kwargs: calls.append(args)
    )

    with pytest.raises(jobs_execution.AdapterError, match="layout|lane"):
        jobs_reliability.LocalProviderPhaseRunner()("codex", context)
    assert calls == []


def test_local_runner_rejects_handoff_traversal_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_reliability.platform, "system", lambda: "Darwin")
    context = replace(_context(tmp_path, "codex"), attempt_id="../../escape")
    calls = []
    monkeypatch.setattr(
        jobs_reliability, "_git", lambda *args, **kwargs: calls.append(args)
    )

    with pytest.raises(jobs_execution.AdapterError, match="handoff|path"):
        jobs_reliability.LocalProviderPhaseRunner()("codex", context)
    assert calls == []


def test_ssh_runner_rejects_handoff_traversal_before_send(tmp_path):
    context = replace(
        _context(tmp_path, "codex"),
        lane_id="codex-pc-1",
        attempt_id=r"..\..\escape",
    )
    calls = []
    runner = jobs_reliability.SSHProviderPhaseRunner(
        host="gpu-pc",
        runtime_root=Path("/home/brandon/.hermes/releases/hermes-agent-" + "e" * 40),
        lane_root=Path("/home/brandon/jobs/lanes"),
        subprocess_run=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(jobs_execution.AdapterError, match="handoff|component"):
        runner("codex", context)
    assert calls == []


def test_remote_repository_mapping_is_explicit_and_fail_closed(monkeypatch):
    local = Path("/Users/brandon/project")
    remote = Path("/home/brandon/project")
    monkeypatch.setenv("HERMES_JOBS_PC_REPO_MAP", json.dumps({str(local): str(remote)}))

    assert jobs_reliability._remote_repository(local) == remote

    monkeypatch.setenv(
        "HERMES_JOBS_PC_REPO_MAP", json.dumps({"/Users/brandon/other": str(remote)})
    )
    with pytest.raises(jobs_execution.AdapterError, match="mapping"):
        jobs_reliability._remote_repository(local)


def test_remote_cleanup_failure_preserves_all_handoff_fields(tmp_path, monkeypatch):
    lane_root = tmp_path / "codex-pc-1"
    for child in ("auth", "worktrees", "handoffs", "receipts", "health"):
        (lane_root / child).mkdir(parents=True)
    result = jobs_reliability.PhaseResult(
        status="succeeded",
        commit="c" * 40,
        tests=tuple(_valid_build_payload()["tests"]),
        review=_valid_review_payload(),
        executor_exit_digest="sha256:" + "3" * 64,
        output_capture_digest="sha256:" + "4" * 64,
        build_handoff={"summary": "built"},
        critical_user_journey={"name": "journey"},
        review_handoff={"summary": "reviewed"},
    )
    raw = {
        "schema_version": 1,
        "provider": "codex",
        "context": {
            "job_id": "j_test",
            "job_number": 1,
            "job_name": "test",
            "goal": "test",
            "attempt_id": "a_test",
            "ordinal": 1,
            "repository": str(tmp_path),
            "base_commit": "b" * 40,
            "branch": "jobs/j_test/attempt-0",
            "worktree": str(lane_root / "worktrees" / "j_test-1"),
            "lane_root": str(lane_root),
            "requested_lane": "codex",
            "lane_id": "codex-pc-1",
            "executor": "codex",
            "specialist": "codex-builder",
            "model": "gpt-5.6-sol",
            "effort": "high",
            "max_turns": 120,
        },
    }
    monkeypatch.setattr(
        jobs_remote_worker.jobs_reliability,
        "LocalProviderPhaseRunner",
        lambda: lambda provider, context: result,
    )
    monkeypatch.setattr(jobs_remote_worker, "_cleanup", lambda context: False)
    monkeypatch.setattr(
        jobs_remote_worker, "_atomic_lease", lambda *args, **kwargs: None
    )
    stdin = io.TextIOWrapper(io.BytesIO(json.dumps(raw).encode()), encoding="utf-8")
    stdout = io.StringIO()
    monkeypatch.setattr(jobs_remote_worker.sys, "stdin", stdin)
    monkeypatch.setattr(jobs_remote_worker.sys, "stdout", stdout)

    assert jobs_remote_worker.execute() == 0
    response = json.loads(stdout.getvalue())
    assert response["failure_reason_code"] == "CLEANUP_FAILED"
    assert response["build_handoff"] == result.build_handoff
    assert response["critical_user_journey"] == result.critical_user_journey
    assert response["review_handoff"] == result.review_handoff


def _remote_worker_request(tmp_path: Path, *, lane_id: str = "codex-pc-1") -> dict:
    lane_root = tmp_path / "codex-pc-1"
    for child in ("auth", "worktrees", "handoffs", "receipts", "health"):
        (lane_root / child).mkdir(parents=True, exist_ok=True)
    context_root = tmp_path / "context"
    context_root.mkdir()
    context = _context(context_root, "codex")
    identity = jobs_identity.resolve_requested_lane("codex")
    context_values = asdict(context)
    context_values.pop("on_heartbeat")
    return {
        "schema_version": 1,
        "provider": "codex",
        "context": {
            **context_values,
            "repository": str(context.repository),
            "worktree": str(lane_root / "worktrees" / "j_test-1"),
            "lane_root": str(lane_root),
            "lane_id": lane_id,
            "requested_lane": identity.requested_lane,
            "executor": identity.executor,
            "specialist": identity.specialist,
            "model": identity.model,
        },
    }


@pytest.mark.parametrize(
    ("lane_id", "outside_worktree"),
    [
        ("codex-pc-1/../../../escape", False),
        (r"codex-pc-1\..\escape", False),
        ("codex-pc-1", True),
    ],
)
def test_remote_worker_rejects_lane_or_path_traversal_before_lease(
    tmp_path, monkeypatch, lane_id, outside_worktree
):
    raw = _remote_worker_request(tmp_path, lane_id=lane_id)
    if outside_worktree:
        raw["context"]["worktree"] = str(tmp_path / "outside" / "j_test-1")
    calls = []
    monkeypatch.setattr(
        jobs_remote_worker.jobs_reliability,
        "LocalProviderPhaseRunner",
        lambda: lambda *args: calls.append(("runner", args)),
    )
    monkeypatch.setattr(
        jobs_remote_worker,
        "_atomic_lease",
        lambda *args, **kwargs: calls.append(("lease", args, kwargs)),
    )
    monkeypatch.setattr(
        jobs_remote_worker.sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(json.dumps(raw).encode()), encoding="utf-8"),
    )
    monkeypatch.setattr(jobs_remote_worker.sys, "stdout", io.StringIO())

    assert jobs_remote_worker.execute() == 2
    assert calls == []


def test_remote_worker_rejects_handoff_traversal_before_lease(tmp_path, monkeypatch):
    raw = _remote_worker_request(tmp_path)
    raw["context"]["attempt_id"] = "../../escape"
    calls = []
    monkeypatch.setattr(
        jobs_remote_worker.jobs_reliability,
        "LocalProviderPhaseRunner",
        lambda: lambda *args: calls.append(("runner", args)),
    )
    monkeypatch.setattr(
        jobs_remote_worker,
        "_atomic_lease",
        lambda *args, **kwargs: calls.append(("lease", args, kwargs)),
    )
    monkeypatch.setattr(
        jobs_remote_worker.sys,
        "stdin",
        io.TextIOWrapper(io.BytesIO(json.dumps(raw).encode()), encoding="utf-8"),
    )
    monkeypatch.setattr(jobs_remote_worker.sys, "stdout", io.StringIO())

    assert jobs_remote_worker.execute() == 2
    assert calls == []


def test_remote_cleanup_still_removes_handoff_when_worktree_removal_fails(
    tmp_path, monkeypatch
):
    raw = _remote_worker_request(tmp_path)
    context = type("Context", (), raw["context"])()
    worktree = Path(context.worktree)
    handoff = Path(context.lane_root) / "handoffs" / context.attempt_id
    worktree.mkdir()
    handoff.mkdir()
    (handoff / "temporary.txt").write_text("bounded", encoding="utf-8")
    monkeypatch.setattr(
        jobs_remote_worker.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "git")
        ),
    )

    assert not jobs_remote_worker._cleanup(context)
    assert not handoff.exists()


@pytest.mark.parametrize("cleanup_behavior", ["ok", "returns_false", "raises"])
def test_remote_runner_exception_cleans_and_settles_failed_without_masking_primary(
    tmp_path, monkeypatch, cleanup_behavior
):
    raw = _remote_worker_request(tmp_path)
    calls = []

    def raise_primary(provider, context):
        calls.append(("runner", provider))
        raise RuntimeError("primary runner failure")

    def cleanup(context):
        calls.append(("cleanup", cleanup_behavior))
        if cleanup_behavior == "raises":
            raise OSError("synthetic cleanup failure")
        return cleanup_behavior == "ok"

    monkeypatch.setattr(
        jobs_remote_worker.jobs_reliability,
        "LocalProviderPhaseRunner",
        lambda: raise_primary,
    )
    monkeypatch.setattr(
        jobs_remote_worker,
        "_cleanup",
        cleanup,
    )
    monkeypatch.setattr(
        jobs_remote_worker,
        "_atomic_lease",
        lambda root, **kwargs: calls.append(("lease", kwargs)),
    )
    stdin = io.TextIOWrapper(io.BytesIO(json.dumps(raw).encode()), encoding="utf-8")
    stderr = io.StringIO()
    monkeypatch.setattr(jobs_remote_worker.sys, "stdin", stdin)
    monkeypatch.setattr(jobs_remote_worker.sys, "stdout", io.StringIO())
    monkeypatch.setattr(jobs_remote_worker.sys, "stderr", stderr)

    assert jobs_remote_worker.execute() == 2
    leases = [entry[1] for entry in calls if entry[0] == "lease"]
    assert leases[0]["state"] == "BUILDING"
    assert leases[-1]["state"] == "FAILED"
    assert any(entry[0] == "cleanup" for entry in calls)
    assert "RuntimeError" in stderr.getvalue()


def test_worker_handoff_feature_is_jobs_scoped_without_global_chat_hooks():
    module_source = inspect.getsource(jobs_reliability).casefold()
    assert "gateway" not in module_source
    assert "direct_chat" not in module_source
    assert not any(
        name.startswith(("register_global", "install_global", "register_chat"))
        for name in vars(jobs_reliability)
    )

    package = Path(jobs_reliability.__file__).resolve().parent
    callers = {
        path.name
        for path in package.glob("*.py")
        if path.name != "jobs_executors.py"
        and "production_registry()" in path.read_text(encoding="utf-8")
    }
    assert callers == {"jobs_run.py", "jobs_runtime.py"}
