#!/usr/bin/env python3
"""Run a provider-free, exact-origin Jobs handoff canary.

The canary is intentionally self-contained at its trust boundary: it verifies
the immutable runtime root before importing any Hermes application module.  It
then uses temporary Jobs/SessionDB files to exercise signed graph edges,
transactional outbox delivery, reviewer correction, a blocked decision, and
completion.  It never reads a profile, lane credential, or live database.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping


MAX_RECEIPT_BYTES = 16 * 1024
SHA256 = "sha256:" + "a" * 64
COMMIT = "c" * 40
BASE_COMMIT = "b" * 40
DELIVERY_NOW = 4_000_000_000


class CanaryRefused(ValueError):
    """A safe refusal before any temporary canary store is created."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, object], field: str) -> str:
    clone = dict(value)
    clone.pop(field, None)
    payload = json.dumps(clone, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bundle_digest(manifest: Mapping[str, object]) -> str:
    payload = {
        "schema_version": manifest.get("schema_version"),
        "release_sha": manifest.get("release_sha"),
        "files": manifest.get("files"),
        "runtime_archive": manifest.get("runtime_archive"),
        "runtime_files": manifest.get("runtime_files"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_relative(root: Path, rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts:
        raise CanaryRefused("runtime inventory escapes runtime root")
    return root / path


def _verify_manifest_and_runtime(runtime_root: Path, bundle: Path) -> dict:
    """Verify immutable bytes without importing the source checkout."""
    runtime_root = runtime_root.resolve()
    bundle = bundle.resolve()
    if not runtime_root.is_dir():
        raise CanaryRefused("runtime root is missing")
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise CanaryRefused("release manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryRefused("release manifest is malformed") from exc
    release_sha = manifest.get("release_sha")
    if (
        not isinstance(release_sha, str)
        or len(release_sha) != 40
        or any(char not in "0123456789abcdef" for char in release_sha)
    ):
        raise CanaryRefused("release manifest has an invalid release SHA")
    runtime = manifest.get("runtime_archive")
    if not isinstance(runtime, dict) or runtime.get("relpath") != "runtime.tar.gz":
        raise CanaryRefused("release manifest has no runtime archive")
    archive = bundle / "runtime.tar.gz"
    if not archive.is_file() or _sha256(archive) != runtime.get("sha256"):
        raise CanaryRefused("runtime archive digest mismatch")
    if archive.stat().st_size != runtime.get("size"):
        raise CanaryRefused("runtime archive size mismatch")
    if _bundle_digest(manifest) != manifest.get("bundle_digest"):
        raise CanaryRefused("release bundle digest mismatch")
    # jobs_release's manifest digest includes every field except itself.
    if _canonical_digest(manifest, "manifest_digest") != manifest.get(
        "manifest_digest"
    ):
        raise CanaryRefused("release manifest digest mismatch")
    expected: dict[str, dict] = {}
    inventory = manifest.get("runtime_files")
    if not isinstance(inventory, list) or not inventory:
        raise CanaryRefused("release runtime inventory is missing")
    for entry in inventory:
        if not isinstance(entry, dict) or not isinstance(entry.get("relpath"), str):
            raise CanaryRefused("release runtime inventory is malformed")
        rel = entry["relpath"].replace(os.sep, "/")
        if rel in expected:
            raise CanaryRefused("release runtime inventory contains duplicates")
        expected[rel] = entry
        path = _safe_relative(runtime_root, rel)
        if path.is_symlink() or not path.is_file():
            raise CanaryRefused("runtime root is missing or symlinked")
        if _sha256(path) != entry.get("sha256") or path.stat().st_size != entry.get(
            "size"
        ):
            raise CanaryRefused("runtime root digest mismatch")
        if path.stat().st_mode & 0o777 != entry.get("mode"):
            raise CanaryRefused("runtime root mode mismatch")
    observed: set[str] = set()
    for path in runtime_root.rglob("*"):
        if path.is_symlink():
            raise CanaryRefused("runtime root contains a symlink")
        if path.is_file():
            observed.add(path.relative_to(runtime_root).as_posix())
    extra = sorted(observed - set(expected))
    if extra:
        raise CanaryRefused("runtime root contains unrecorded files")
    return manifest


def _prepare_runtime_import(runtime_root: Path):
    """Import Hermes only after root verification and bind modules to it."""
    runtime_root = runtime_root.resolve()
    source_scripts = Path(__file__).resolve().parent
    source_root = source_scripts.parent
    sys.path[:] = [
        str(runtime_root),
        *[
            item
            for item in sys.path
            if Path(item or os.curdir).resolve() not in {source_scripts, runtime_root}
            and Path(item or os.curdir).resolve() != source_root
        ],
    ]
    modules = {
        "jdb": importlib.import_module("hermes_cli.jobs_db"),
        "handoffs": importlib.import_module("hermes_cli.jobs_handoffs"),
        "graph": importlib.import_module("hermes_cli.jobs_graph"),
        "receipts": importlib.import_module("hermes_cli.jobs_receipts"),
        "notifications": importlib.import_module("hermes_cli.jobs_notifications"),
        "gateway_notifications": importlib.import_module("gateway.jobs_notifications"),
        "SessionDB": importlib.import_module("hermes_state").SessionDB,
    }
    for name, module in modules.items():
        if name == "SessionDB":
            module_file = Path(sys.modules["hermes_state"].__file__).resolve()
        else:
            module_file = Path(module.__file__).resolve()
        if not module_file.is_relative_to(runtime_root):
            raise CanaryRefused(f"{name} imported outside runtime root")
    return modules


def _origin(session_id: str = "origin") -> dict[str, str]:
    return {
        "platform": "telegram",
        "chat_id": "chat-origin",
        "session_id": session_id,
        "chat_type": "dm",
        "thread_id": "thread-origin",
        "user_id": "user-origin",
        "profile": "default",
    }


def _decision() -> dict[str, object]:
    return {
        "question": "Which safe path should the blocked Job take?",
        "options": [
            {
                "id": "retry",
                "label": "Retry safely",
                "consequence": "Resume the bounded correction.",
            },
            {
                "id": "stop",
                "label": "Stop this Job",
                "consequence": "Leave this Job blocked.",
            },
        ],
        "recommendation": "retry",
        "recommendation_reason": "Retry is bounded and leaves the workspace safe.",
        "blocked_scope": "Only the canary decision Job is blocked.",
        "safe_state": "No provider or workspace mutation occurred.",
        "next_owner_role": "operator",
    }


def _edge(
    modules: Mapping[str, object],
    conn,
    key,
    *,
    job_id: str,
    attempt_id: str,
    source: str | None,
    target: str,
    speaker_id: str,
    speaker_role: str,
    outcome: str,
    at: int,
    next_owner: str | None,
    summary: str,
    next_action: str,
    required: tuple[str, ...],
    issues=(),
    decision_request=None,
    artifact=None,
):
    jdb = modules["jdb"]
    handoffs = modules["handoffs"]
    graph = modules["graph"]
    receipts = modules["receipts"]
    evidence = {name: SHA256 for name in required}
    raw = {
        "summary": summary,
        "evidence_summary": [
            {"label": "Signed canary evidence", "result": "PASS", "digest": SHA256}
        ],
        "next_action": next_action,
        "issues": [dict(item) for item in issues],
        "decision_request": decision_request,
    }
    handoff = handoffs.normalize_handoff(
        raw,
        job_id=job_id,
        attempt_id=attempt_id,
        speaker_id=speaker_id,
        speaker_role=speaker_role,
        speaker_executor="hermes-canary",
        from_phase="ATTEMPT_CREATED" if source is None else source,
        to_phase=target,
        next_owner_role=next_owner,
        outcome=outcome,
        artifact_identity=artifact,
        transition_evidence=evidence,
        created_at=at,
    )
    revision = jdb.get_job(conn, job_id).revision
    receipt_id = f"receipt:{job_id}:{attempt_id}:{at}:{target}"
    request = graph.TransitionRequest(
        job_id=job_id,
        attempt_id=attempt_id,
        source_state=source,
        target_state=target,
        initiator_type="system",
        initiator_id=speaker_id,
        expected_job_revision=revision,
        evidence=evidence,
        failure_class="reviewer_rejection"
        if outcome == "rejected"
        else ("USER_DECISION" if outcome == "blocked" else None),
        blocker_code="CANARY_DECISION" if outcome == "blocked" else None,
        commit=COMMIT,
        receipt_id=receipt_id,
        component="jobs_graph",
        component_version="1",
        idempotency_key=f"canary:{job_id}:{attempt_id}:{at}:{target}",
        created_at=at,
        handoff=handoff,
    )
    envelope = receipts.sign_receipt(
        {
            "job_id": job_id,
            "attempt_id": attempt_id,
            "commit": COMMIT,
            "state": target,
            "evidence": evidence,
            "handoff": handoff,
            "timestamp": f"canary:{at}",
            "transitioned_by": "jobs_graph/1",
        },
        receipt_id=receipt_id,
        key_id="lane:jobs-canary:v1",
        private_key=key,
    )
    return graph.transition_attempt(
        conn, request, envelope=envelope, verifier=modules["verifier"]
    )


def _start(modules, conn, job_id: str, worker: str, at: int) -> str:
    claim = modules["jdb"].claim_job(
        conn,
        worker=worker,
        specialist="codex-builder",
        lease_seconds=600,
        job=job_id,
        now=at,
    )
    if claim is None:
        raise CanaryRefused("canary could not claim Job")
    return modules["jdb"].start_attempt(
        conn,
        job_id,
        claim_token=claim.claim_token,
        specialist="codex-builder",
        base_commit=BASE_COMMIT,
        commit=COMMIT,
        now=at + 1,
    )


def _run_canary(modules, work: Path, manifest: Mapping[str, object]) -> dict:
    jdb = modules["jdb"]
    key_type = importlib.import_module(
        "cryptography.hazmat.primitives.asymmetric.ed25519"
    ).Ed25519PrivateKey
    key = key_type.generate()
    modules["verifier"] = modules["graph"].ReceiptVerifier(
        trusted_keys={"lane:jobs-canary:v1": key.public_key()}
    )
    jobs_path = work / "jobs.db"
    state_path = work / "state.db"
    session_db = modules["SessionDB"](db_path=state_path)
    conn = jdb.connect(jobs_path)
    try:
        origin = _origin()
        job_id = jdb.create_job(
            conn,
            name="Handoff canary",
            goal="Exercise signed builder, tester, reviewer, and completion flow",
            requested_lane="codex",
            origin=origin,
        )
        decision_job_id = jdb.create_job(
            conn,
            name="Decision canary",
            goal="Exercise a complete blocked decision handoff",
            requested_lane="codex",
            origin=origin,
        )
        attempt = _start(modules, conn, job_id, "builder-1", 100)
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source=None,
            target="QUEUED",
            speaker_id="router-0",
            speaker_role="router",
            outcome="started",
            at=105,
            next_owner="router",
            summary="The builder attempt entered the queued state.",
            next_action="Router assigns the canary builder.",
            required=("attempt_created",),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="QUEUED",
            target="ASSIGNED",
            speaker_id="router-1",
            speaker_role="router",
            outcome="handed_off",
            at=110,
            next_owner="builder",
            summary="Router assigned the canary builder.",
            next_action="Builder starts the scoped change.",
            required=("preflight", "route", "lane_health"),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="ASSIGNED",
            target="BUILDING",
            speaker_id="builder-1",
            speaker_role="builder",
            outcome="started",
            at=120,
            next_owner="builder",
            summary="Builder started the canary change.",
            next_action="Builder hands evidence to the tester.",
            required=("claim", "worktree", "attempt_started"),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="BUILDING",
            target="EVIDENCE_COLLECTING",
            speaker_id="builder-1",
            speaker_role="builder",
            outcome="handed_off",
            at=130,
            next_owner="tester",
            summary="Builder handed the change to the tester.",
            next_action="Tester checks the signed evidence.",
            required=("executor_exit", "output_capture"),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="EVIDENCE_COLLECTING",
            target="REVIEWING",
            speaker_id="tester-1",
            speaker_role="tester",
            outcome="handed_off",
            at=140,
            next_owner="reviewer",
            summary="Tester handed evidence to the reviewer.",
            next_action="Reviewer records the decision.",
            required=("tests", "readback"),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=attempt,
            source="REVIEWING",
            target="FAILED",
            speaker_id="reviewer-1",
            speaker_role="reviewer",
            outcome="rejected",
            at=150,
            next_owner="builder",
            summary="Reviewer rejected the first canary change.",
            next_action="Builder applies the correction and retries.",
            required=("tests", "readback"),
            issues=(
                {
                    "requirement": "The handoff must be reviewable.",
                    "finding": "The first canary change missed the requested behavior.",
                    "required_fix": "Apply the reviewer correction before retrying.",
                },
            ),
        )
        retry = _start(modules, conn, job_id, "builder-2", 160)
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=retry,
            source=None,
            target="QUEUED",
            speaker_id="router-retry",
            speaker_role="router",
            outcome="started",
            at=170,
            next_owner="builder",
            summary="Correction retry was queued.",
            next_action="Builder applies the reviewer correction.",
            required=("attempt_created",),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=retry,
            source="QUEUED",
            target="ASSIGNED",
            speaker_id="router-2",
            speaker_role="router",
            outcome="handed_off",
            at=180,
            next_owner="builder",
            summary="Correction retry was assigned.",
            next_action="Builder starts the correction.",
            required=("preflight", "route", "lane_health"),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=retry,
            source="ASSIGNED",
            target="BUILDING",
            speaker_id="builder-2",
            speaker_role="builder",
            outcome="started",
            at=190,
            next_owner="builder",
            summary="Builder started the correction.",
            next_action="Builder hands corrected evidence to the tester.",
            required=("claim", "worktree", "attempt_started"),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=retry,
            source="BUILDING",
            target="EVIDENCE_COLLECTING",
            speaker_id="builder-2",
            speaker_role="builder",
            outcome="handed_off",
            at=200,
            next_owner="tester",
            summary="Builder handed corrected evidence to the tester.",
            next_action="Tester rechecks the correction.",
            required=("executor_exit", "output_capture"),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=retry,
            source="EVIDENCE_COLLECTING",
            target="REVIEWING",
            speaker_id="tester-1",
            speaker_role="tester",
            outcome="handed_off",
            at=210,
            next_owner="reviewer",
            summary="Tester handed corrected evidence to the reviewer.",
            next_action="Reviewer approves the correction.",
            required=("tests", "readback"),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=retry,
            source="REVIEWING",
            target="VERIFIED",
            speaker_id="reviewer-1",
            speaker_role="reviewer",
            outcome="passed",
            at=220,
            next_owner="builder",
            summary="Reviewer approved the corrected canary change.",
            next_action="Complete the Job.",
            required=("themis_review", "receipt_verification"),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=job_id,
            attempt_id=retry,
            source="VERIFIED",
            target="COMPLETED",
            speaker_id="builder-2",
            speaker_role="builder",
            outcome="completed",
            at=230,
            next_owner=None,
            summary="The corrected canary change is complete.",
            next_action="No further action is required.",
            required=("activation_gate", "completion_receipt"),
            artifact={"kind": "commit", "value": COMMIT},
        )

        decision_attempt = _start(
            modules, conn, decision_job_id, "builder-decision", 300
        )
        _edge(
            modules,
            conn,
            key,
            job_id=decision_job_id,
            attempt_id=decision_attempt,
            source=None,
            target="QUEUED",
            speaker_id="router-decision-0",
            speaker_role="router",
            outcome="started",
            at=305,
            next_owner="router",
            summary="The decision attempt entered the queued state.",
            next_action="Router assigns the decision canary.",
            required=("attempt_created",),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=decision_job_id,
            attempt_id=decision_attempt,
            source="QUEUED",
            target="ASSIGNED",
            speaker_id="router-decision",
            speaker_role="router",
            outcome="handed_off",
            at=310,
            next_owner="builder",
            summary="Decision canary was assigned.",
            next_action="Builder encounters the bounded decision.",
            required=("preflight", "route", "lane_health"),
        )
        _edge(
            modules,
            conn,
            key,
            job_id=decision_job_id,
            attempt_id=decision_attempt,
            source="ASSIGNED",
            target="BLOCKED",
            speaker_id="hermes",
            speaker_role="hermes",
            outcome="blocked",
            at=320,
            next_owner="operator",
            summary="Decision canary is blocked pending an explicit choice.",
            next_action="Operator chooses one supplied option.",
            required=("preflight", "route", "lane_health"),
            decision_request=_decision(),
        )
    finally:
        conn.close()

    session_db.create_session("origin", source="telegram")
    session_db.create_session("decoy", source="telegram")
    delivered = []
    while True:
        notification_id = modules[
            "gateway_notifications"
        ].deliver_due_notification_once(
            session_db, jobs_path=jobs_path, now=DELIVERY_NOW
        )
        if notification_id is None:
            break
        delivered.append(notification_id)
    messages = session_db.get_messages("origin")
    decoy_messages = session_db.get_messages("decoy")
    conn = jdb.connect(jobs_path)
    try:
        notifications = modules["notifications"].list_notifications(conn)
        transitions = conn.execute(
            "SELECT id FROM job_attempt_transitions ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
        session_db.close()
    ids = [row.notification_id for row in notifications]
    generic = sum(
        1
        for message in messages
        if "Legacy Jobs update" in str(message.get("content", ""))
        and message.get("display_metadata", {}).get("milestone") != "queued"
    )
    duplicate_ids = len(ids) - len(set(ids))
    substantive = [
        message
        for message in messages
        if "Legacy Jobs update" not in str(message.get("content", ""))
    ]
    if not messages or decoy_messages or duplicate_ids or generic:
        raise CanaryRefused("canary delivery invariants failed")
    required_words = (
        "tester",
        "reviewer",
        "correction",
        "approved",
        "decision",
        "complete",
    )
    if not all(
        any(word in str(item.get("content", "")).lower() for item in substantive)
        for word in required_words
    ):
        raise CanaryRefused("canary substantive conversation is incomplete")
    digest_list = sorted({SHA256})
    receipt = {
        "schema_version": 1,
        "kind": "jobs-handoff-canary",
        "release_sha": manifest["release_sha"],
        "bundle_digest": manifest["bundle_digest"],
        "job_ids": [job_id, decision_job_id],
        "attempt_ids": [attempt, retry, decision_attempt],
        "notification_ids": ids,
        "transition_count": len(transitions),
        "notification_count": len(notifications),
        "delivered_count": len(delivered),
        "message_count": len(messages),
        "decoy_message_count": len(decoy_messages),
        "substantive_message_count": len(substantive),
        "duplicate_notification_ids": duplicate_ids,
        "generic_status_messages": generic,
        "evidence_digests": digest_list,
        "verdict": "PASS",
    }
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise CanaryRefused("canary receipt exceeds bounded size")
    receipt["receipt_digest"] = hashlib.sha256(encoded).hexdigest()
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--receipt-out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = _verify_manifest_and_runtime(args.runtime_root, args.bundle)
        modules = _prepare_runtime_import(args.runtime_root)
        with tempfile.TemporaryDirectory(prefix="jobs-handoff-canary-") as directory:
            receipt = _run_canary(modules, Path(directory), manifest)
        output = args.receipt_out.resolve()
        home = Path.home().resolve()
        if (
            output == home / ".hermes"
            or output.is_relative_to(home / ".hermes")
            or output == home / ".local"
            or output.is_relative_to(home / ".local")
        ):
            raise CanaryRefused("receipt path is a live Hermes path")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (CanaryRefused, OSError, ImportError, ValueError, RuntimeError) as exc:
        print(f"canary refused: {exc}", file=sys.stderr)
        return 2
    print(f"verdict: {receipt['verdict']}")
    print(f"release_sha: {receipt['release_sha']}")
    print(f"receipt: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
