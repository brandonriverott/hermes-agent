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
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

SCHEMA_VERSION = 1
BUNDLE_NAME = "jobs-release-v1"
MANIFEST_NAME = "manifest.json"
FILES_SUBDIR = "files"
RUNTIME_ARCHIVE_NAME = "runtime.tar.gz"
RECEIPT_SUBDIR = "activation-receipts"
BACKUP_DIRNAME = ".jobs-release-backup"
KIND = "jobs-release-activation"
ROLLBACK_KIND = "jobs-release-rollback"
LIVE_PREFLIGHT_KIND = "jobs-release-live-preflight"
LIVE_PROFILES = frozenset({"mac", "pc"})

# Explicit Jobs plugin/runtime/policy file set (repo-root relative).
# Deliberately not a glob: an immutable release must be reproducible and
# reviewable, and broad recursive targets are forbidden.
RELEASE_FILES: tuple[str, ...] = (
    "gateway/jobs_dispatcher.py",
    "gateway/jobs_notifications.py",
    "gateway/run.py",
    "hermes_cli/data/jobs-codex-result.v1.schema.json",
    "hermes_cli/data/jobs-build-result.v1.schema.json",
    "hermes_cli/data/jobs-review-result.v1.schema.json",
    "hermes_cli/data/jobs-lanes.v1.json",
    "hermes_cli/jobs.py",
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
    "hermes_cli/jobs_reliability.py",
    "hermes_cli/jobs_release.py",
    "hermes_cli/jobs_run.py",
    "hermes_cli/jobs_runtime.py",
    "hermes_cli/jobs_skills.py",
    "hermes_cli/jobs_tool.py",
    "hermes_cli/plugins.py",
    "plugins/jobs/__init__.py",
    "plugins/jobs/plugin.yaml",
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


@dataclass(frozen=True)
class LiveActivationAuthorization:
    profile: str
    release_sha: str
    bundle_digest: str
    root: Path
    backup_dir: Path
    receipt_dir: Path
    preflight_path: Path


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
    clone.pop("manifest_digest", None)
    payload = json.dumps(clone, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _content_digest(manifest: dict) -> str:
    payload = {
        "schema_version": manifest.get("schema_version"),
        "release_sha": manifest.get("release_sha"),
        "files": manifest.get("files"),
        "runtime_archive": manifest.get("runtime_archive"),
        "runtime_files": manifest.get("runtime_files"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


def _tracked_runtime_files(source_root: Path) -> list[dict]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=source_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ReleaseError("cannot enumerate tracked runtime files")
    entries: list[dict] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            rel = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReleaseError("tracked runtime path is not UTF-8") from exc
        parts = Path(rel).parts
        if not parts or Path(rel).is_absolute() or ".." in parts:
            raise ReleaseError("tracked runtime path escapes source root")
        source = source_root / rel
        if source.is_symlink():
            raise ReleaseError("tracked runtime symlinks are not supported")
        if not source.is_file():
            raise ReleaseError(f"tracked runtime file is missing: {rel}")
        entries.append(
            {
                "relpath": rel.replace(os.sep, "/"),
                "sha256": sha256_file(source),
                "size": source.stat().st_size,
                "mode": source.stat().st_mode & 0o777,
            }
        )
    if not entries:
        raise ReleaseError("tracked runtime is empty")
    return sorted(entries, key=lambda item: item["relpath"])


def _build_runtime_archive(source_root: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(fd)
    try:
        completed = subprocess.run(
            ["git", "archive", "--format=tar.gz", "-o", tmp_name, "HEAD"],
            cwd=source_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ReleaseError("cannot build complete runtime archive")
        os.replace(tmp_name, destination)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return {
        "relpath": RUNTIME_ARCHIVE_NAME,
        "sha256": sha256_file(destination),
        "size": destination.stat().st_size,
    }


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
    runtime_archive = _build_runtime_archive(
        source_root, bundle / RUNTIME_ARCHIVE_NAME
    )
    runtime_files = _tracked_runtime_files(source_root)

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
        "runtime_archive": runtime_archive,
        "runtime_files": runtime_files,
        "bundle_digest": "",
        "manifest_digest": "",
    }
    manifest["bundle_digest"] = _content_digest(manifest)
    manifest["manifest_digest"] = _canonical_digest(manifest)
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
    runtime = manifest.get("runtime_archive")
    if not isinstance(runtime, dict) or runtime.get("relpath") != RUNTIME_ARCHIVE_NAME:
        raise ReleaseRefused("manifest missing complete runtime archive")
    archive = bundle_dir / RUNTIME_ARCHIVE_NAME
    if not archive.is_file():
        raise ReleaseRefused("missing complete runtime archive")
    if (
        sha256_file(archive) != runtime.get("sha256")
        or archive.stat().st_size != runtime.get("size")
    ):
        raise ReleaseRefused("runtime archive digest mismatch")
    runtime_files = manifest.get("runtime_files")
    if not isinstance(runtime_files, list) or not runtime_files:
        raise ReleaseRefused("manifest missing complete runtime file inventory")
    if _content_digest(manifest) != manifest.get("bundle_digest"):
        raise ReleaseRefused("bundle digest mismatch in manifest")
    if _canonical_digest(manifest) != manifest.get("manifest_digest"):
        raise ReleaseRefused("manifest digest mismatch")
    return manifest


def verify_runtime_root(root: Path, bundle_dir: Path) -> dict:
    """Verify every tracked runtime file and refuse unrecorded files."""
    root = root.resolve()
    manifest = verify_bundle(bundle_dir)
    expected = {entry["relpath"]: entry for entry in manifest["runtime_files"]}
    observed = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    mismatched = []
    mode_mismatched = []
    for rel in sorted(set(expected) & set(observed)):
        path = observed[rel]
        entry = expected[rel]
        if path.is_symlink() or sha256_file(path) != entry["sha256"]:
            mismatched.append(rel)
        elif path.stat().st_size != entry["size"]:
            mismatched.append(rel)
        elif path.stat().st_mode & 0o777 != entry["mode"]:
            mode_mismatched.append(rel)
    if missing or extra or mismatched or mode_mismatched:
        raise ReleaseRefused(
            "runtime root mismatch"
            f" (missing={len(missing)}, extra={len(extra)}, "
            f"digest={len(mismatched)}, mode={len(mode_mismatched)})"
        )
    return {"ok": True, "checked": len(expected), "root": str(root)}


def _extract_runtime_root(root: Path, bundle_dir: Path) -> None:
    """Atomically materialize the verified full runtime into a new root."""
    if root.exists():
        raise ReleaseRefused("immutable runtime root already exists")
    manifest = verify_bundle(bundle_dir)
    expected_entries = {
        entry["relpath"]: entry for entry in manifest["runtime_files"]
    }
    expected = set(expected_entries)
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.stage-", dir=str(root.parent))
    )
    try:
        with tarfile.open(bundle_dir / RUNTIME_ARCHIVE_NAME, mode="r:gz") as archive:
            archived_files: set[str] = set()
            for member in archive.getmembers():
                rel = member.name.rstrip("/")
                parts = Path(rel).parts
                if not rel or Path(rel).is_absolute() or ".." in parts:
                    raise ReleaseRefused("runtime archive path escapes release root")
                destination = stage / rel
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ReleaseRefused("runtime archive contains unsupported entry")
                archived_files.add(rel)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ReleaseRefused("runtime archive member cannot be read")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                os.chmod(destination, expected_entries[rel]["mode"])
            if archived_files != expected:
                raise ReleaseRefused("runtime archive inventory mismatch")
        verify_runtime_root(stage, bundle_dir)
        os.replace(stage, root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


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
    home_dir: Path | None = None,
    live_authorization: LiveActivationAuthorization | None = None,
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
    if live_authorization is None:
        refuse_live_root(root, home_dir=home_dir)
    else:
        expected_root = live_target_root(
            home_dir or Path.home(), live_authorization.profile, manifest["release_sha"]
        )
        if live_authorization.root != root or root != expected_root:
            raise ReleaseRefused("live activation target root mismatch")
        if live_authorization.release_sha != manifest["release_sha"]:
            raise ReleaseRefused("live activation release SHA mismatch")
        if live_authorization.bundle_digest != manifest["bundle_digest"]:
            raise ReleaseRefused("live activation bundle digest mismatch")
        if backup_dir is None or backup_dir.resolve() != live_authorization.backup_dir:
            raise ReleaseRefused("live activation backup directory mismatch")
    if dry_run:
        install_check(root, bundle_dir)  # refuse drift in validation mode

    files_dir = bundle_dir / FILES_SUBDIR
    runtime_root_created = False
    if not dry_run:
        if backup_dir is None:
            backup_dir = root / BACKUP_DIRNAME / time.strftime("%Y%m%d-%H%M%S")
        backup_dir = backup_dir.resolve()
        backup_dir.mkdir(parents=True, exist_ok=False)
        if live_authorization is not None:
            _extract_runtime_root(root, bundle_dir)
            runtime_root_created = True

    targets: list[dict] = []
    for entry in manifest["files"]:
        rel = entry["relpath"]
        target = root / rel
        before_sha = (
            None
            if runtime_root_created
            else (sha256_file(target) if target.is_file() else None)
        )
        backup = None
        if dry_run:
            after_sha = entry["sha256"]
        else:
            if live_authorization is None:
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
        "live_profile": live_authorization.profile if live_authorization else None,
        "bundle_digest": manifest["bundle_digest"],
        "root": str(root),
        "created_at": _utcnow_iso(),
        "bundle_dir": str(bundle_dir),
        "backup_dir": str(backup_dir) if not dry_run else None,
        "runtime_archive_sha256": manifest["runtime_archive"]["sha256"],
        "runtime_file_count": len(manifest["runtime_files"]),
        "runtime_root_created": runtime_root_created,
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


def _require_live_profile(profile: str) -> None:
    if profile not in LIVE_PROFILES:
        raise ReleaseRefused("unknown live target profile")


def live_target_root(home_dir: Path, profile: str, release_sha: str) -> Path:
    """Return the sole versioned runtime root allowed for a live profile."""
    _require_live_profile(profile)
    if len(release_sha) != 40 or any(c not in "0123456789abcdef" for c in release_sha):
        raise ReleaseRefused("invalid release SHA")
    return (home_dir.resolve() / ".hermes" / "releases" / f"hermes-agent-{release_sha}").resolve()


def live_backup_dir(home_dir: Path, profile: str, release_sha: str) -> Path:
    _require_live_profile(profile)
    return (
        home_dir.resolve()
        / ".hermes"
        / "backups"
        / "jobs-releases"
        / release_sha
        / profile
    ).resolve()


def live_receipt_dir(home_dir: Path) -> Path:
    return (
        home_dir.resolve() / ".hermes" / "activation-receipts" / "jobs-releases"
    ).resolve()


def live_acknowledgement(profile: str, release_sha: str) -> str:
    _require_live_profile(profile)
    return f"ACTIVATE-JOBS-LIVE:{profile}:{release_sha}"


def _preflight_digest(preflight: dict) -> str:
    clone = dict(preflight)
    clone.pop("preflight_digest", None)
    payload = json.dumps(clone, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_live_preflight(
    path: Path,
    bundle_dir: Path,
    *,
    profile: str,
    home_dir: Path,
    backup_dir: Path,
    receipt_dir: Path,
) -> dict:
    """Write read-only evidence binding one profile to one verified bundle."""
    _require_live_profile(profile)
    manifest = verify_bundle(bundle_dir)
    release_sha = manifest["release_sha"]
    expected_backup = live_backup_dir(home_dir, profile, release_sha)
    expected_receipts = live_receipt_dir(home_dir)
    if backup_dir.resolve() != expected_backup:
        raise ReleaseRefused("live activation backup directory is not declared")
    if receipt_dir.resolve() != expected_receipts:
        raise ReleaseRefused("live activation receipt directory is not declared")
    preflight = {
        "schema_version": SCHEMA_VERSION,
        "kind": LIVE_PREFLIGHT_KIND,
        "profile": profile,
        "release_sha": release_sha,
        "bundle_digest": manifest["bundle_digest"],
        "root": str(live_target_root(home_dir, profile, release_sha)),
        "backup_dir": str(expected_backup),
        "receipt_dir": str(expected_receipts),
        "created_at": _utcnow_iso(),
        "preflight_digest": "",
    }
    preflight["preflight_digest"] = _preflight_digest(preflight)
    _write_json(path.resolve(), preflight)
    return preflight


def authorize_live_activation(
    bundle_dir: Path,
    *,
    profile: str,
    expected_release_sha: str,
    acknowledgement: str,
    preflight_path: Path,
    home_dir: Path,
    backup_dir: Path,
    receipt_dir: Path,
) -> LiveActivationAuthorization:
    """Validate the explicit, profile-bound authority required for live apply."""
    _require_live_profile(profile)
    manifest = verify_bundle(bundle_dir)
    release_sha = manifest["release_sha"]
    if expected_release_sha != release_sha:
        raise ReleaseRefused("live activation release SHA mismatch")
    if acknowledgement != live_acknowledgement(profile, release_sha):
        raise ReleaseRefused("live activation acknowledgement mismatch")
    if not preflight_path.is_file():
        raise ReleaseRefused("live activation preflight is missing")
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseRefused("live activation preflight is malformed") from exc
    expected = {
        "kind": LIVE_PREFLIGHT_KIND,
        "profile": profile,
        "release_sha": release_sha,
        "bundle_digest": manifest["bundle_digest"],
        "root": str(live_target_root(home_dir, profile, release_sha)),
        "backup_dir": str(live_backup_dir(home_dir, profile, release_sha)),
        "receipt_dir": str(live_receipt_dir(home_dir)),
    }
    if _preflight_digest(preflight) != preflight.get("preflight_digest"):
        raise ReleaseRefused("live activation preflight digest mismatch")
    for field, value in expected.items():
        if preflight.get(field) != value:
            raise ReleaseRefused(f"live activation preflight {field} mismatch")
    if backup_dir.resolve() != Path(expected["backup_dir"]):
        raise ReleaseRefused("live activation backup directory mismatch")
    if receipt_dir.resolve() != Path(expected["receipt_dir"]):
        raise ReleaseRefused("live activation receipt directory mismatch")
    return LiveActivationAuthorization(
        profile=profile,
        release_sha=release_sha,
        bundle_digest=manifest["bundle_digest"],
        root=Path(expected["root"]),
        backup_dir=Path(expected["backup_dir"]),
        receipt_dir=Path(expected["receipt_dir"]),
        preflight_path=preflight_path.resolve(),
    )


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
    runtime_root_action = None
    targets = receipt.get("targets", [])
    if receipt.get("runtime_root_created") is True:
        bundle_dir = Path(str(receipt.get("bundle_dir", "")))
        manifest = verify_bundle(bundle_dir)
        if manifest["runtime_archive"]["sha256"] != receipt.get(
            "runtime_archive_sha256"
        ):
            raise ReleaseRefused("runtime archive receipt mismatch")
        verify_runtime_root(root, bundle_dir)
        shutil.rmtree(root)
        runtime_root_action = "removed"
        outcomes.append(
            {"relpath": ".", "action": "removed", "final_sha": None}
        )
        targets = []

    for target in targets:
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
        "runtime_root_action": runtime_root_action,
        "targets": outcomes,
    }
    ev_dir = evidence_dir or receipt_path.parent
    ev_path = ev_dir / f"rollback-{time.strftime('%Y%m%d-%H%M%S')}.json"
    _write_json(ev_path, evidence)
    evidence["evidence"] = str(ev_path)
    return evidence
