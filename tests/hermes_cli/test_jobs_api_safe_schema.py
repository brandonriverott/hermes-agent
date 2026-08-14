"""The model-facing schema view must satisfy OpenAI structured-output limits.

The on-disk schema files are the strict contract (the engine validates results
against them with full JSON Schema). The API refuses ``allOf`` and ``oneOf``
outright, which killed every codex build at spawn — so the spawn path hands
codex a transformed copy and never rewrites the files.
"""

import json
from types import SimpleNamespace

from hermes_cli import jobs_reliability as jr


def _banned(value, keywords=("allOf", "oneOf")):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keywords:
                yield key
            yield from _banned(item, keywords)
    elif isinstance(value, list):
        for item in value:
            yield from _banned(item, keywords)


def test_api_safe_schema_strips_banned_keywords_from_real_schemas():
    for schema_path in (jr._BUILD_SCHEMA, jr._REVIEW_SCHEMA):
        original = json.loads(schema_path.read_text(encoding="utf-8"))
        assert list(_banned(original)), (
            f"{schema_path.name} lost its strict conditional rules; "
            "if that is deliberate, this transform may be removable"
        )
        transformed = jr._api_safe_schema(original)
        assert not list(_banned(transformed))
        # The strict file itself is never rewritten.
        assert json.loads(schema_path.read_text(encoding="utf-8")) == original


def test_codex_command_uses_api_safe_schema_file(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    context = SimpleNamespace(model="gpt-5.6-sol", worktree=tmp_path, effort="max")
    result_path = tmp_path / "build-result.json"
    command = jr._provider_command(
        "codex", phase="build", context=context, result_path=result_path
    )
    schema_arg = command[command.index("--output-schema") + 1]
    assert schema_arg == str(tmp_path / "build-schema.api.json")
    written = json.loads((tmp_path / "build-schema.api.json").read_text())
    assert not list(_banned(written))


def test_claude_command_gets_api_safe_schema_text(tmp_path):
    """claude 2.1.197 silently disables structured output on a "$schema" key."""
    context = SimpleNamespace(model="claude-opus-5", worktree=tmp_path, effort="max")
    command = jr._provider_command(
        "claude",
        phase="build",
        context=context,
        result_path=tmp_path / "build-result.json",
    )
    schema_text = command[command.index("--json-schema") + 1]
    parsed = json.loads(schema_text)
    assert not list(_banned(parsed))
    assert "$schema" not in parsed


def test_default_process_runner_actually_runs_a_process(tmp_path):
    """The runner must survive a real spawn.

    Every unit test stubs the process runner, which let it ship calling
    ``time.monotonic()`` without importing ``time`` — every real spawn
    crashed the engine one line after Popen and orphaned the child as
    PROCESS_CRASHED. This is the one test that runs the real thing.
    """
    result = jr._default_process_runner(
        ["/bin/cat"],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        prompt=b"runner smoke\n",
        stdout_path=tmp_path / "out.txt",
        stderr_path=tmp_path / "err.txt",
        timeout=30,
    )
    assert result.returncode == 0
    assert not result.timed_out
    assert (tmp_path / "out.txt").read_bytes() == b"runner smoke\n"


def test_coercion_reimposes_conditionals_on_benign_decoration():
    succeeded = {"outcome": "succeeded", "failure_class": "implementation", "reason": "all good"}
    fixed = jr._coerce_api_schema_result("codex", "build", succeeded)
    assert fixed["failure_class"] is None and fixed["reason"] is None
    # A PASS review carrying findings is self-contradiction, never coerced.
    passed = {"verdict": "PASS", "finding_type": "defect", "findings": [{"x": 1}]}
    assert jr._coerce_api_schema_result("codex", "review", passed) == passed
    # Real disagreements pass through untouched and fail strict normalization.
    failed = {"outcome": "failed", "failure_class": None, "reason": None}
    assert jr._coerce_api_schema_result("codex", "build", failed) == failed
    # Claude now receives the same API-safe schema, so it coerces too.
    fixed = jr._coerce_api_schema_result("claude", "build", succeeded)
    assert fixed["failure_class"] is None and fixed["reason"] is None


def test_sanitizer_keeps_evidence_with_ansi_colors(tmp_path):
    """A pytest transcript with color codes must survive sanitization."""
    noisy = tmp_path / "build-stdout.json"
    noisy.write_bytes(
        b'{"evidence": "\x1b[32m2 passed\x1b[0m in 0.01s"}\n'
        b'{"more": "plain line"}\n'
    )
    body = jr._sanitize_phase_file(noisy)
    assert b"redacted" not in body
    assert b"2 passed" in body
    assert b"\x1b" not in body


def test_review_phase_can_execute_but_not_write(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    context = SimpleNamespace(model="claude-opus-5", worktree=tmp_path, effort="max")
    review = jr._provider_command(
        "claude", phase="review", context=context, result_path=tmp_path / "r.json"
    )
    assert review[review.index("--permission-mode") + 1] == "default"
    allow = review[review.index("--allowedTools") + 1]
    assert "Bash(pytest:*)" in allow and "Bash(python3:*)" in allow
    assert "git add" not in allow and "git commit" not in allow

    codex_review = jr._provider_command(
        "codex", phase="review", context=context, result_path=tmp_path / "r.json"
    )
    assert codex_review[codex_review.index("--sandbox") + 1] == "workspace-write"
