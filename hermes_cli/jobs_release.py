"""Immutable Jobs release packaging, parity, activation, and rollback.

Task 10 rollout tooling. Builds a versioned, content-addressed bundle of the
Jobs plugin/runtime/policy files plus a SHA-256 manifest; verifies bundle
integrity; validates identical install/check on simulated Mac and PC roots;
and provides dry-run activation with an activation receipt plus rollback that
restores exactly the recorded targets.

Refusals (always safe, never silent):
  * dirty source tree (uncommitted changes or untracked files);
  * missing manifest file in a bundle;
  * any SHA-256 digest mismatch (bundle file vs manifest, or installed target
    vs manifest);
  * target drift (an installed target whose current bytes differ from the
    release);
  * a live Hermes root (``~/.hermes`` / ``~/.local``) -- real activation is a
    Task 11 decision, never an accident of this module.

This module never touches a live Hermes directory, cron, gateway restart,
Tailscale connection, or real activation. Credentials stay in lane auth roots
and are never part of a release.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = 1
BUNDLE_NAME = "jobs-release-v1"
MANIFEST_NAME = "manifest.json"
FILES_SUBDIR = "files"
RECEIPT_SUBDIR = "activation-receipts"
BACKUP_DIRNAME = ".jobs-release-backup"
KIND = "jobs-release-activation"
ROLLBACK_KIND = "jobs-release-rollback"

# Explicit Jobs plugin/runtime/policy file set (repo-root relative).
# Deliberately not a glob: an immutable release must be reproducible and
# reviewable, and broad recursive targets are forbidden.
RELEASE_FILES: tuple[str, ...] = (
    "gateway/jobs_notifications.py",
    "hermes_cli/data/jobs-lanes.v1.json",
    "hermes_cli/jobs_adapter_claude.py",
    "hermes_cli/jobs_adapter_codex.py",
    "hermes_cli/jobs_contract.py",
    "hermes_cli/jobs_db.py",
    "hermes_cli/jobs_dispatch.py",
    "hermes_cli/jobs_exec.py",
    "hermes_cli/jobs_execution.py",
    "hermes_cli/jobs_executors.py",
    "hermes_cli/jobs_graph.py",
    "hermes_cli/jobs_harness.py",
    "hermes_cli/jobs_identity.py",
    "hermes_cli/jobs_lanes.py",
    "hermes_cli/jobs_loop.py",
    "hermes_cli/jobs_model_policy.py",
    "hermes_cli/jobs_notifications.py",
    "hermes_cli/jobs_receipts.py",
    "hermes_cli/jobs_release.py",
    "hermes_cli/jobs_run.py",
    "hermes_cli/jobs_runtime.py",
    "hermes_cli/jobs_skills.py",
    "hermes_cli/jobs_tool.py",
    "scripts/jobs-release-activate.py",
    "scripts/jobs-release-rollback.py",
    "scripts/jobs-runtime-parity.py",
    "scripts/jobs_evidence_gate.py",
    "scripts/jobs_lane_health.py",
    "scripts/jobs_provision_lanes.py",
)


class ReleaseError(ValueError):
    """Invalid release input or state."""


class ReleaseRefused(ReleaseError):
    """A safe refusal: dirty source, drift, mismatch, or live root."""


# ── small helpers ─────────────────────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy into place via same-directory temp file + atomic rename."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dst.name + ".", suffix=".tmp", dir=str(dst.parent))
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            shutil.copyfileobj(inp, out)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_digest(manifest: dict) -> str:
    clone = dict(manifest)
    clone.pop("bundle_digest", None)
    payload = json.dumps(clone, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── source gating ─────────────────────────────────────────────────────────────


def git_head_sha(source_root: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, capture_output=True, text=True
    )
    if out.returncode != 0:
        raise ReleaseError("cannot resolve source HEAD: " + out.stderr.strip())
    sha = out.stdout.strip()
    if len(sha) != 40:
        raise ReleaseError("unexpected HEAD length: " + sha)
    return sha


def assert_source_clean(source_root: Path) -> None:
    if not (source_root / ".git").exists():
        raise ReleaseRefused("source tree is not a git repository")
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_root, capture_output=True, text=True
    )
    if out.returncode != 0:
        raise ReleaseError("cannot read git status: " + out.stderr.strip())
    if out.stdout.strip():
        raise ReleaseRefused(
            "dirty source tree: uncommitted changes or untracked files"
        )


def resolve_release_files(source_root: Path) -> list[str]:
    missing = [rel for rel in RELEASE_FILES if not (source_root / rel).is_file()]
    if missing:
        raise ReleaseError(
            "missing release files in source tree: " + ", ".join(missing)
        )
    return list(RELEASE_FILES)


# ── bundle construction and verification ─────────────────────────────────────


def build_bundle(
    source_root: Path,
    output_root: Path,
    *,
    files: Sequence[str] | None = None,
) -> tuple[Path, dict]:
    """Build a versioned, content-addressed bundle plus SHA-256 manifest.

    Refuses a dirty source tree and any missing release file. The bundle
    lives at ``<output_root>/releases/<release_sha>/jobs-release-v1/``.
    """
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    assert_source_clean(source_root)

    rels = [rel.replace(os.sep, "/") for rel in (files or resolve_release_files(source_root))]
    rels = sorted(set(rels))
    for rel in rels:
        src = source_root / rel
        if not src.is_file():
            raise ReleaseError(f"missing release file in source tree: {rel}")

    release_sha = git_head_sha(source_root)
    bundle = output_root / "releases" / release_sha / BUNDLE_NAME
    files_dir = bundle / FILES_SUBDIR

    entries: list[dict] = []
    for rel in rels:
        src = source_root / rel
        dst = files_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        _atomic_copy(src, dst)
        entries.append(
            {"relpath": rel, "sha256": sha256_file(src), "size": src.stat().st_size}
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_sha": release_sha,
        "created_at": _utcnow_iso(),
        "created_by": "jobs_release.build_bundle",
        "source": str(source_root),
        "files": entries,
        "bundle_digest": "",
    }
    manifest["bundle_digest"] = _canonical_digest(manifest)
    _write_json(bundle / MANIFEST_NAME, manifest)
    return bundle, manifest


def load_manifest(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ReleaseRefused(f"missing manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseRefused(f"malformed manifest: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseRefused(
            f"unsupported manifest schema: {manifest.get('schema_version')}"
        )
    if not manifest.get("release_sha") or not manifest.get("files"):
        raise ReleaseRefused("manifest missing release_sha or files")
    return manifest


def verify_bundle(bundle_dir: Path) -> dict:
    """Verify a bundle against its manifest; refuse on any mismatch."""
    bundle_dir = bundle_dir.resolve()
    manifest = load_manifest(bundle_dir)
    files_dir = bundle_dir / FILES_SUBDIR

    missing: list[str] = []
    mismatched: list[str] = []
    for entry in manifest["files"]:
        rel = entry["relpath"]
        f = files_dir / rel
        if not f.is_file():
            missing.append(rel)
            continue
        if sha256_file(f) != entry.get("sha256") or f.stat().st_size != entry.get("size"):
            mismatched.append(rel)
    if missing:
        raise ReleaseRefused("missing manifest files in bundle: " + ", ".join(missing))
    if mismatched:
        raise ReleaseRefused("digest mismatch in bundle: " + ", ".join(mismatched))
    if _canonical_digest(manifest) != manifest.get("bundle_digest"):
        raise ReleaseRefused("bundle digest mismatch in manifest")
    return manifest


# ── install: check, dry-run, apply ────────────────────────────────────────────


def install_check(root: Path, bundle_dir: Path) -> dict[str, dict]:
    """Report per-file status; refuse if any installed target drifted."""
    root = root.resolve()
    manifest = verify_bundle(bundle_dir)
    statuses: dict[str, dict] = {}
    drifted: list[str] = []
    for entry in manifest["files"]:
        rel = entry["relpath"]
        target = root / rel
        if not target.is_file():
            statuses[rel] = {"status": "missing", "sha256": None}
            continue
        digest = sha256_file(target)
        if digest == entry["sha256"]:
            statuses[rel] = {"status": "match", "sha256": digest}
        else:
            statuses[rel] = {"status": "drift", "sha256": digest}
            drifted.append(rel)
    if drifted:
        raise ReleaseRefused("target drift refuses install: " + ", ".join(drifted))
    return statuses


def install_release(
    root: Path,
    bundle_dir: Path,
    *,
    dry_run: bool = True,
    backup_dir: Path | None = None,
    host_kind: str = "mac",
) -> dict:
    """Install (or dry-run) a verified bundle into a target root.

    Dry-run is the validation gate: it refuses target drift and writes
    nothing. Apply mode backs up every existing file (whatever its bytes --
    an older recorded version is expected) before replacing it with the
    release bytes, then re-verifies every installed file against the
    manifest. The returned activation receipt records exact targets, prior/
    installed digests, and backup locations so rollback can restore the
    prior recorded state.
    """
    root = root.resolve()
    bundle_dir = bundle_dir.resolve()
    manifest = verify_bundle(bundle_dir)
    if dry_run:
        install_check(root, bundle_dir)  # refuse drift in validation mode

    files_dir = bundle_dir / FILES_SUBDIR
    if not dry_run:
        if backup_dir is None:
            backup_dir = root / BACKUP_DIRNAME / time.strftime("%Y%m%d-%H%M%S")
        backup_dir = backup_dir.resolve()
        backup_dir.mkdir(parents=True, exist_ok=False)

    targets: list[dict] = []
    for entry in manifest["files"]:
        rel = entry["relpath"]
        target = root / rel
        before_sha = sha256_file(target) if target.is_file() else None
        backup = None
        if dry_run:
            after_sha = entry["sha256"]
        else:
            if target.is_file():
                backup = backup_dir / rel
                _atomic_copy(target, backup)
            _atomic_copy(files_dir / rel, target)
            after_sha = sha256_file(target)
            if after_sha != entry["sha256"]:
                raise ReleaseError(f"post-install verification failed for {rel}")
        targets.append(
            {
                "relpath": rel,
                "target": str(target),
                "before_sha": before_sha,
                "after_sha": after_sha,
                "backup": str(backup) if backup is not None else None,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "dry_run": bool(dry_run),
        "release_sha": manifest["release_sha"],
        "host_kind": host_kind,
        "root": str(root),
        "created_at": _utcnow_iso(),
        "bundle_dir": str(bundle_dir),
        "backup_dir": str(backup_dir) if not dry_run else None,
        "targets": targets,
    }


# ── parity: byte-identical install on simulated Mac and PC roots ─────────────


def check_parity(mac_root: Path, pc_root: Path, bundle_dir: Path) -> dict:
    """Refuse unless every manifest file is byte-identical on both roots."""
    mac_root = mac_root.resolve()
    pc_root = pc_root.resolve()
    manifest = verify_bundle(bundle_dir)

    failures: list[str] = []
    checked = 0
    for entry in manifest["files"]:
        rel = entry["relpath"]
        mac_f = mac_root / rel
        pc_f = pc_root / rel
        if not mac_f.is_file() or not pc_f.is_file():
            failures.append(f"{rel}: missing on {'pc' if mac_f.is_file() else 'mac'}")
            continue
        mac_sha, pc_sha = sha256_file(mac_f), sha256_file(pc_f)
        checked += 1
        if mac_sha != pc_sha or mac_sha != entry["sha256"]:
            failures.append(f"{rel}: mac={mac_sha[:12]} pc={pc_sha[:12]}")
    if failures:
        raise ReleaseRefused(
            "parity failed across mac/pc roots: " + "; ".join(failures)
        )
    return {
        "ok": True,
        "checked": checked,
        "mac_root": str(mac_root),
        "pc_root": str(pc_root),
    }


# ── live-root guard ───────────────────────────────────────────────────────────


def refuse_live_root(root: Path, home_dir: Path | None = None) -> None:
    """Refuse target roots inside real Hermes install directories.

    Real activation is a Task 11 decision; until then, no tooling may target a
    live ``~/.hermes`` / ``~/.local`` tree.
    """
    root = root.resolve()
    home = (home_dir or Path.home()).resolve()
    for live in (home / ".hermes", home / ".local"):
        if root == live or root.is_relative_to(live):
            raise ReleaseRefused(f"refusing live Hermes root: {root}")


# ── rollback: restore exactly the recorded targets ───────────────────────────


def rollback(
    root: Path,
    receipt_path: Path,
    *,
    evidence_dir: Path | None = None,
) -> dict:
    """Restore the prior recorded state from an activation receipt.

    Only targets recorded in the receipt are touched. Files that did not
    exist before activation are removed (after verifying the installed bytes
    still match the receipt); files that did exist are restored from their
    backup and re-verified. Evidence is written next to the receipt.
    """
    root = root.resolve()
    receipt_path = receipt_path.resolve()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseRefused(f"malformed receipt: {exc}") from exc

    if receipt.get("kind") != KIND:
        raise ReleaseRefused("receipt is not a jobs-release activation receipt")
    if receipt.get("dry_run"):
        raise ReleaseRefused("refusing rollback of a dry-run receipt (nothing was installed)")
    if Path(receipt["root"]).resolve() != root:
        raise ReleaseRefused(
            f"receipt root mismatch: receipt={receipt['root']} root={root}"
        )

    outcomes: list[dict] = []
    for target in receipt.get("targets", []):
        rel = target["relpath"]
        target_path = root / rel
        if not target_path.is_relative_to(root):
            raise ReleaseRefused(f"target escapes root: {target_path}")
        before_sha = target.get("before_sha")
        after_sha = target.get("after_sha")
        backup = target.get("backup")

        if before_sha is None:
            # Did not exist before activation: remove only our exact bytes.
            if target_path.is_file():
                current = sha256_file(target_path)
                if after_sha is not None and current != after_sha:
                    raise ReleaseRefused(
                        f"refusing to remove drifted file {rel} (sha {current[:12]})"
                    )
                target_path.unlink()
                outcomes.append({"relpath": rel, "action": "removed", "final_sha": None})
            else:
                outcomes.append({"relpath": rel, "action": "absent", "final_sha": None})
        else:
            if not backup or not Path(backup).is_file():
                raise ReleaseRefused(f"missing backup for {rel}: {backup}")
            _atomic_copy(Path(backup), target_path)
            final_sha = sha256_file(target_path)
            if final_sha != before_sha:
                raise ReleaseRefused(f"rollback verification failed for {rel}")
            outcomes.append({"relpath": rel, "action": "restored", "final_sha": final_sha})

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "kind": ROLLBACK_KIND,
        "created_at": _utcnow_iso(),
        "receipt": str(receipt_path),
        "root": str(root),
        "release_sha": receipt.get("release_sha"),
        "result": "ok",
        "targets": outcomes,
    }
    ev_dir = evidence_dir or receipt_path.parent
    ev_path = ev_dir / f"rollback-{time.strftime('%Y%m%d-%H%M%S')}.json"
    _write_json(ev_path, evidence)
    evidence["evidence"] = str(ev_path)
    return evidence
