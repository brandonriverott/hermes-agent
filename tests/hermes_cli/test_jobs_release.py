"""Focused tests for immutable Jobs release packaging, parity, and rollback.

Task 10 scope: versioned content-addressed bundle + SHA-256 manifest,
refusal of dirty source / missing manifest files / digest mismatch / target
drift, identical install/check on simulated Mac and PC roots, dry-run
activation with an activation receipt, and rollback that restores exactly the
recorded targets. Everything runs against temporary roots; no live Hermes
directory, cron, gateway restart, Tailscale connection, or real activation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli import jobs_release
from hermes_cli.jobs_release import ReleaseError, ReleaseRefused

FILES = [
    "hermes_cli/jobs_identity.py",
    "hermes_cli/data/jobs-lanes.v1.json",
    "scripts/jobs-release-activate.py",
]


def _write(root: Path, rel: str, text: str = "") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text or f"content of {rel}\n", encoding="utf-8")
    return p


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _make_source(tmp_path: Path, files=FILES) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    for rel in files:
        _write(root, rel)
    _write(root, "runtime_only.py", "FULL_RUNTIME_SENTINEL = True\n")
    _git("init", "-q", "-b", "main", cwd=root)
    _git("config", "user.email", "test@example.com", cwd=root)
    _git("config", "user.name", "Test", cwd=root)
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "seed", cwd=root)
    return root


def _build(tmp_path: Path):
    src = _make_source(tmp_path)
    out = tmp_path / "out"
    bundle, manifest = jobs_release.build_bundle(src, out, files=FILES)
    return src, out, bundle, manifest


def _manifest_sha(manifest: dict, rel: str) -> str:
    return next(f["sha256"] for f in manifest["files"] if f["relpath"] == rel)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── bundle construction / refusal of dirty source ────────────────────────────


def test_build_refuses_dirty_source(tmp_path):
    src = _make_source(tmp_path)
    _write(src, "hermes_cli/jobs_identity.py", "uncommitted change\n")
    with pytest.raises(ReleaseRefused, match="dirty"):
        jobs_release.build_bundle(src, tmp_path / "out", files=FILES)


def test_build_refuses_untracked_file(tmp_path):
    src = _make_source(tmp_path)
    _write(src, "notes.txt", "untracked\n")
    with pytest.raises(ReleaseRefused, match="dirty"):
        jobs_release.build_bundle(src, tmp_path / "out", files=FILES)


def test_build_refuses_missing_source_file(tmp_path):
    src = _make_source(tmp_path, files=[FILES[0]])
    with pytest.raises(ReleaseError, match="missing"):
        jobs_release.build_bundle(src, tmp_path / "out", files=FILES)


def test_build_creates_versioned_immutable_bundle(tmp_path, monkeypatch):
    src = _make_source(tmp_path)
    head = _git("rev-parse", "HEAD", cwd=src)
    out = tmp_path / "out"
    bundle, manifest = jobs_release.build_bundle(src, out, files=FILES)

    # Content-addressed location and creation metadata.
    assert bundle == out / "releases" / head / jobs_release.BUNDLE_NAME
    assert manifest["schema_version"] == 1
    assert manifest["release_sha"] == head
    assert manifest["created_at"]
    assert manifest["created_by"]
    assert manifest["source"] == str(src.resolve())

    # Exact file list with per-file SHA-256 digest and size.
    assert sorted(f["relpath"] for f in manifest["files"]) == sorted(FILES)
    for entry in manifest["files"]:
        src_file = src / entry["relpath"]
        bundle_file = bundle / jobs_release.FILES_SUBDIR / entry["relpath"]
        assert bundle_file.read_bytes() == src_file.read_bytes()
        assert entry["sha256"] == _sha_bytes(src_file.read_bytes())
        assert entry["size"] == src_file.stat().st_size
    assert len(manifest["bundle_digest"]) == 64
    runtime = manifest["runtime_archive"]
    assert runtime["relpath"] == jobs_release.RUNTIME_ARCHIVE_NAME
    assert len(runtime["sha256"]) == 64
    assert runtime["size"] > 0
    assert any(
        entry["relpath"] == "runtime_only.py"
        for entry in manifest["runtime_files"]
    )
    assert (bundle / jobs_release.RUNTIME_ARCHIVE_NAME).is_file()

    # Rebuilding from the same clean source yields the same release SHA.
    monkeypatch.setattr(jobs_release, "_utcnow_iso", lambda: "2099-01-01T00:00:00+00:00")
    out2 = tmp_path / "out2"
    bundle2, manifest2 = jobs_release.build_bundle(src, out2, files=FILES)
    assert bundle2.name == bundle.name
    assert manifest2["release_sha"] == head
    assert manifest2["created_at"] != manifest["created_at"]
    assert manifest2["bundle_digest"] == manifest["bundle_digest"]


def test_release_file_set_contains_every_rollout_runtime_seam():
    required = {
        "gateway/jobs_dispatcher.py",
        "gateway/jobs_notifications.py",
        "gateway/run.py",
        "hermes_cli/data/jobs-codex-result.v1.schema.json",
        "hermes_cli/data/jobs-build-result.v1.schema.json",
        "hermes_cli/data/jobs-review-result.v1.schema.json",
        "hermes_cli/jobs.py",
        "hermes_cli/jobs_db.py",
        "hermes_cli/jobs_dispatch.py",
        "hermes_cli/jobs_execution.py",
        "hermes_cli/jobs_graph.py",
        "hermes_cli/jobs_handoffs.py",
        "hermes_cli/jobs_loop.py",
        "hermes_cli/jobs_notifications.py",
        "hermes_cli/jobs_reliability.py",
        "hermes_cli/jobs_release.py",
        "hermes_cli/plugins.py",
        "scripts/jobs_remote_worker.py",
        "scripts/jobs-handoff-canary.py",
        "plugins/jobs/__init__.py",
        "plugins/jobs/plugin.yaml",
    }
    assert required <= set(jobs_release.RELEASE_FILES)


# ── bundle verification: missing manifest / missing files / digest mismatch ──


def test_verify_refuses_missing_manifest(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    (bundle / jobs_release.MANIFEST_NAME).unlink()
    with pytest.raises(ReleaseRefused, match="manifest"):
        jobs_release.verify_bundle(bundle)


def test_verify_refuses_missing_manifest_file(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    (bundle / jobs_release.FILES_SUBDIR / FILES[0]).unlink()
    with pytest.raises(ReleaseRefused, match="missing"):
        jobs_release.verify_bundle(bundle)


def test_verify_refuses_digest_mismatch(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    f = bundle / jobs_release.FILES_SUBDIR / FILES[0]
    f.write_bytes(f.read_bytes() + b"corrupt")
    with pytest.raises(ReleaseRefused, match="digest"):
        jobs_release.verify_bundle(bundle)


def test_verify_refuses_runtime_archive_digest_mismatch(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    archive = bundle / jobs_release.RUNTIME_ARCHIVE_NAME
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    with pytest.raises(ReleaseRefused, match="runtime archive digest"):
        jobs_release.verify_bundle(bundle)


def test_runtime_verification_refuses_mode_drift(tmp_path):
    _, _, bundle, manifest = _build(tmp_path)
    home = tmp_path / "home"
    backup = jobs_release.live_backup_dir(home, "mac", manifest["release_sha"])
    receipts = jobs_release.live_receipt_dir(home)
    preflight = tmp_path / "preflight.json"
    root = jobs_release.live_target_root(home, "mac", manifest["release_sha"])
    jobs_release.write_live_preflight(
        preflight,
        bundle,
        profile="mac",
        home_dir=home,
        backup_dir=backup,
        receipt_dir=receipts,
    )
    authorization = jobs_release.authorize_live_activation(
        bundle,
        profile="mac",
        expected_release_sha=manifest["release_sha"],
        acknowledgement=jobs_release.live_acknowledgement(
            "mac", manifest["release_sha"]
        ),
        preflight_path=preflight,
        home_dir=home,
        backup_dir=backup,
        receipt_dir=receipts,
    )
    jobs_release.install_release(
        root,
        bundle,
        dry_run=False,
        backup_dir=backup,
        host_kind="mac",
        home_dir=home,
        live_authorization=authorization,
    )
    target = root / "runtime_only.py"
    target.chmod(0o777)

    with pytest.raises(ReleaseRefused, match="runtime root mismatch"):
        jobs_release.verify_runtime_root(root, bundle)


# ── install check: target drift refusal, missing vs match ────────────────────


def test_install_check_refuses_target_drift(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    root = tmp_path / "mac"
    _write(root, FILES[0], "drifted content\n")
    with pytest.raises(ReleaseRefused, match="drift"):
        jobs_release.install_check(root, bundle)


def test_install_check_reports_missing_and_accepts_match(tmp_path):
    src, _, bundle, _ = _build(tmp_path)
    root = tmp_path / "mac"
    (root / FILES[0]).parent.mkdir(parents=True)
    shutil.copy2(src / FILES[0], root / FILES[0])
    statuses = jobs_release.install_check(root, bundle)
    assert statuses[FILES[0]]["status"] == "match"
    assert statuses[FILES[1]]["status"] == "missing"


# ── install: dry-run preserves targets + receipt; apply backs up + installs ──


def test_install_dry_run_preserves_targets_and_receipts(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    root = tmp_path / "mac"
    root.mkdir()
    receipt = jobs_release.install_release(root, bundle, dry_run=True, host_kind="mac")

    assert receipt["dry_run"] is True
    assert receipt["host_kind"] == "mac"
    assert receipt["release_sha"]
    assert receipt["root"] == str(root.resolve())
    assert list(root.rglob("*")) == []  # nothing was written to the target
    assert receipt["backup_dir"] is None

    by_rel = {t["relpath"]: t for t in receipt["targets"]}
    assert set(by_rel) == set(FILES)
    for rel in FILES:
        assert by_rel[rel]["before_sha"] is None
        assert len(by_rel[rel]["after_sha"]) == 64
        assert by_rel[rel]["backup"] is None


def test_install_apply_backs_up_and_installs(tmp_path):
    src, _, bundle, manifest = _build(tmp_path)
    root = tmp_path / "mac"
    prior = _write(root, FILES[1], "pre-existing plugin bytes\n")
    prior_sha = _sha_bytes(prior.read_bytes())
    backup_dir = tmp_path / "backup"

    receipt = jobs_release.install_release(
        root, bundle, dry_run=False, backup_dir=backup_dir, host_kind="mac"
    )

    # Installed bytes match the bundle exactly.
    for rel in FILES:
        assert (root / rel).read_bytes() == (src / rel).read_bytes()

    # The pre-existing file was backed up with its prior bytes.
    backup_file = backup_dir / FILES[1]
    assert backup_file.read_bytes() == b"pre-existing plugin bytes\n"

    # Receipt records exact targets, prior and installed state.
    by_rel = {t["relpath"]: t for t in receipt["targets"]}
    assert by_rel[FILES[0]]["before_sha"] is None
    assert by_rel[FILES[1]]["before_sha"] == prior_sha
    assert by_rel[FILES[1]]["backup"] == str(backup_file.resolve())
    assert by_rel[FILES[0]]["after_sha"] == _manifest_sha(manifest, FILES[0])
    assert by_rel[FILES[1]]["after_sha"] == _manifest_sha(manifest, FILES[1])


# ── parity: identical installs pass; any hash difference refuses ─────────────


def test_parity_identical_and_refuses_drift(tmp_path):
    src, _, bundle, _ = _build(tmp_path)
    mac, pc = tmp_path / "mac", tmp_path / "pc"
    jobs_release.install_release(
        mac, bundle, dry_run=False, backup_dir=tmp_path / "bk-mac", host_kind="mac"
    )
    jobs_release.install_release(
        pc, bundle, dry_run=False, backup_dir=tmp_path / "bk-pc", host_kind="pc"
    )
    result = jobs_release.check_parity(mac, pc, bundle)
    assert result["ok"] is True
    assert result["checked"] == len(FILES)

    (pc / FILES[0]).write_bytes(b"tampered\n")
    with pytest.raises(ReleaseRefused, match="parity"):
        jobs_release.check_parity(mac, pc, bundle)


def test_parity_refuses_missing_on_one_root(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    mac, pc = tmp_path / "mac", tmp_path / "pc"
    jobs_release.install_release(
        mac, bundle, dry_run=False, backup_dir=tmp_path / "bk-mac", host_kind="mac"
    )
    jobs_release.install_release(
        pc, bundle, dry_run=False, backup_dir=tmp_path / "bk-pc", host_kind="pc"
    )
    (pc / FILES[1]).unlink()
    with pytest.raises(ReleaseRefused, match="parity"):
        jobs_release.check_parity(mac, pc, bundle)


# ── rollback: restores prior recorded state, refuses unsafe receipts ─────────


def test_rollback_restores_prior_state(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    root = tmp_path / "mac"
    prior = _write(root, FILES[0], "original plugin\n")
    prior_sha = _sha_bytes(prior.read_bytes())

    receipt = jobs_release.install_release(
        root, bundle, dry_run=False, backup_dir=tmp_path / "backup", host_kind="mac"
    )
    receipt_path = tmp_path / "activation-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    # Something changes the installed file after activation.
    (root / FILES[0]).write_bytes(b"later drift\n")

    evidence = jobs_release.rollback(root, receipt_path)

    # Prior recorded state restored; files absent before install are removed.
    assert (root / FILES[0]).read_bytes() == b"original plugin\n"
    assert _sha_bytes((root / FILES[0]).read_bytes()) == prior_sha
    assert not (root / FILES[1]).exists()

    assert evidence["result"] == "ok"
    actions = {e["relpath"]: e["action"] for e in evidence["targets"]}
    assert actions[FILES[0]] == "restored"
    assert actions[FILES[1]] == "removed"


def test_rollback_leaves_unrecorded_files_alone(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    root = tmp_path / "mac"
    receipt = jobs_release.install_release(
        root, bundle, dry_run=False, backup_dir=tmp_path / "backup", host_kind="mac"
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    unrelated = root / "notes.txt"
    unrelated.write_text("keep me\n", encoding="utf-8")

    jobs_release.rollback(root, receipt_path)
    assert unrelated.read_text(encoding="utf-8") == "keep me\n"


def test_rollback_refuses_dry_run_receipt(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    root = tmp_path / "mac"
    root.mkdir()
    receipt = jobs_release.install_release(root, bundle, dry_run=True, host_kind="mac")
    receipt_path = tmp_path / "dry.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ReleaseRefused, match="dry.run|dry_run"):
        jobs_release.rollback(root, receipt_path)


def test_rollback_refuses_receipt_root_mismatch(tmp_path):
    _, _, bundle, _ = _build(tmp_path)
    mac, pc = tmp_path / "mac", tmp_path / "pc"
    receipt = jobs_release.install_release(
        mac, bundle, dry_run=False, backup_dir=tmp_path / "backup", host_kind="mac"
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ReleaseRefused, match="root mismatch"):
        jobs_release.rollback(pc, receipt_path)


# ── live-root guard: real Hermes directories are refused until Task 11 ───────


def test_refuse_live_root(tmp_path):
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)

    with pytest.raises(ReleaseRefused, match="live"):
        jobs_release.refuse_live_root(hermes, home_dir=home)
    with pytest.raises(ReleaseRefused, match="live"):
        jobs_release.refuse_live_root(hermes / "plugins", home_dir=home)
    with pytest.raises(ReleaseRefused, match="live"):
        jobs_release.refuse_live_root(local_bin, home_dir=home)

    scratch = tmp_path / "scratch" / "mac"
    scratch.mkdir(parents=True)
    jobs_release.refuse_live_root(scratch, home_dir=home)  # must not raise


def test_live_activation_requires_exact_profile_sha_ack_and_preflight(tmp_path):
    _, _, bundle, manifest = _build(tmp_path)
    home = tmp_path / "home"
    backup = jobs_release.live_backup_dir(home, "mac", manifest["release_sha"])
    receipts = jobs_release.live_receipt_dir(home)
    preflight = tmp_path / "preflight-mac.json"
    root = jobs_release.live_target_root(home, "mac", manifest["release_sha"])

    jobs_release.write_live_preflight(
        preflight,
        bundle,
        profile="mac",
        home_dir=home,
        backup_dir=backup,
        receipt_dir=receipts,
    )
    authorization = jobs_release.authorize_live_activation(
        bundle,
        profile="mac",
        expected_release_sha=manifest["release_sha"],
        acknowledgement=jobs_release.live_acknowledgement(
            "mac", manifest["release_sha"]
        ),
        preflight_path=preflight,
        home_dir=home,
        backup_dir=backup,
        receipt_dir=receipts,
    )

    receipt = jobs_release.install_release(
        root,
        bundle,
        dry_run=False,
        backup_dir=backup,
        host_kind="mac",
        home_dir=home,
        live_authorization=authorization,
    )
    assert receipt["live_profile"] == "mac"
    assert receipt["bundle_digest"] == manifest["bundle_digest"]
    assert Path(receipt["root"]) == root
    assert receipt["runtime_root_created"] is True
    assert (root / "runtime_only.py").read_text(encoding="utf-8") == (
        "FULL_RUNTIME_SENTINEL = True\n"
    )
    assert jobs_release.verify_runtime_root(root, bundle)["ok"] is True


def test_live_activation_never_overwrites_existing_same_sha_root(tmp_path):
    _, _, bundle, manifest = _build(tmp_path)
    home = tmp_path / "home"
    backup = jobs_release.live_backup_dir(home, "mac", manifest["release_sha"])
    receipts = jobs_release.live_receipt_dir(home)
    preflight = tmp_path / "preflight-mac.json"
    root = jobs_release.live_target_root(home, "mac", manifest["release_sha"])
    root.mkdir(parents=True)
    (root / "operator-file").write_text("must survive\n", encoding="utf-8")

    jobs_release.write_live_preflight(
        preflight,
        bundle,
        profile="mac",
        home_dir=home,
        backup_dir=backup,
        receipt_dir=receipts,
    )
    authorization = jobs_release.authorize_live_activation(
        bundle,
        profile="mac",
        expected_release_sha=manifest["release_sha"],
        acknowledgement=jobs_release.live_acknowledgement(
            "mac", manifest["release_sha"]
        ),
        preflight_path=preflight,
        home_dir=home,
        backup_dir=backup,
        receipt_dir=receipts,
    )

    with pytest.raises(ReleaseRefused, match="immutable runtime root already exists"):
        jobs_release.install_release(
            root,
            bundle,
            dry_run=False,
            backup_dir=backup,
            host_kind="mac",
            home_dir=home,
            live_authorization=authorization,
        )
    assert (root / "operator-file").read_text(encoding="utf-8") == "must survive\n"


def test_live_rollback_removes_exact_verified_runtime_root(tmp_path):
    _, _, bundle, manifest = _build(tmp_path)
    home = tmp_path / "home"
    backup = jobs_release.live_backup_dir(home, "pc", manifest["release_sha"])
    receipts = jobs_release.live_receipt_dir(home)
    preflight = tmp_path / "preflight-pc.json"
    root = jobs_release.live_target_root(home, "pc", manifest["release_sha"])
    jobs_release.write_live_preflight(
        preflight,
        bundle,
        profile="pc",
        home_dir=home,
        backup_dir=backup,
        receipt_dir=receipts,
    )
    authorization = jobs_release.authorize_live_activation(
        bundle,
        profile="pc",
        expected_release_sha=manifest["release_sha"],
        acknowledgement=jobs_release.live_acknowledgement(
            "pc", manifest["release_sha"]
        ),
        preflight_path=preflight,
        home_dir=home,
        backup_dir=backup,
        receipt_dir=receipts,
    )
    receipt = jobs_release.install_release(
        root,
        bundle,
        dry_run=False,
        backup_dir=backup,
        host_kind="pc",
        home_dir=home,
        live_authorization=authorization,
    )
    receipt_path = receipts / "pc-activation.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    evidence = jobs_release.rollback(root, receipt_path)

    assert not root.exists()
    assert evidence["runtime_root_action"] == "removed"


@pytest.mark.parametrize(
    ("profile", "sha", "ack", "preflight_name", "match"),
    [
        ("unknown", "manifest", "valid", "valid", "profile"),
        ("mac", "wrong", "valid", "valid", "release SHA"),
        ("mac", "manifest", "wrong", "valid", "acknowledgement"),
        ("mac", "manifest", "valid", "missing", "preflight"),
    ],
)
def test_live_activation_refuses_incomplete_or_mismatched_authority(
    tmp_path, profile, sha, ack, preflight_name, match
):
    _, _, bundle, manifest = _build(tmp_path)
    home = tmp_path / "home"
    backup = jobs_release.live_backup_dir(home, "mac", manifest["release_sha"])
    receipts = jobs_release.live_receipt_dir(home)
    preflight = tmp_path / "preflight.json"
    jobs_release.write_live_preflight(
        preflight,
        bundle,
        profile="mac",
        home_dir=home,
        backup_dir=backup,
        receipt_dir=receipts,
    )
    expected_sha = manifest["release_sha"] if sha == "manifest" else "0" * 40
    acknowledgement = (
        jobs_release.live_acknowledgement("mac", manifest["release_sha"])
        if ack == "valid"
        else "yes"
    )
    chosen_preflight = preflight if preflight_name == "valid" else tmp_path / "missing.json"

    with pytest.raises(ReleaseRefused, match=match):
        jobs_release.authorize_live_activation(
            bundle,
            profile=profile,
            expected_release_sha=expected_sha,
            acknowledgement=acknowledgement,
            preflight_path=chosen_preflight,
            home_dir=home,
            backup_dir=backup,
            receipt_dir=receipts,
        )


def test_live_root_install_refuses_without_authorization(tmp_path):
    _, _, bundle, manifest = _build(tmp_path)
    home = tmp_path / "home"
    root = jobs_release.live_target_root(home, "pc", manifest["release_sha"])
    with pytest.raises(ReleaseRefused, match="live"):
        jobs_release.install_release(
            root,
            bundle,
            dry_run=False,
            backup_dir=tmp_path / "backup",
            host_kind="pc",
            home_dir=home,
        )
