#!/usr/bin/env python3
"""Build, verify, and dry-run/apply an immutable Jobs release (Task 10).

Defaults to dry-run: no target root is mutated and the activation receipt
records exactly what WOULD happen. --apply performs the install on the given
simulated target roots with a backup of every replaced file and writes an
activation receipt that the rollback tool can restore from.

Live activation remains unavailable by default. It requires an exact mac/pc
profile, release-SHA acknowledgement, and a matching preflight document.
This tool never restarts services, changes cron, or opens a Tailscale connection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_cli import jobs_release  # noqa: E402


def _write_receipt(receipt: dict, receipt_dir: Path) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = (
        receipt_dir
        / f"{receipt['host_kind']}-{receipt['release_sha'][:12]}-"
        f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path,
                        help="clean git source tree to bundle")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="where releases and activation receipts are written")
    parser.add_argument("--root", type=Path,
                        help="Mac target root (simulated in tests)")
    parser.add_argument("--pc-root", type=Path,
                        help="PC target root (simulated in tests)")
    parser.add_argument("--apply", action="store_true",
                        help="mutate target roots; default is dry-run")
    parser.add_argument("--receipt-out", type=Path,
                        help="receipt directory (default <output-root>/activation-receipts)")
    parser.add_argument("--live-profile", choices=sorted(jobs_release.LIVE_PROFILES),
                        help="exact live target profile; derives target paths from HOME")
    parser.add_argument("--expected-release-sha",
                        help="exact 40-character SHA required for live apply")
    parser.add_argument("--acknowledge-live-activation",
                        help="exact ACTIVATE-JOBS-LIVE:<profile>:<sha> acknowledgement")
    parser.add_argument("--preflight", type=Path,
                        help="live preflight evidence to authorize apply")
    parser.add_argument("--write-live-preflight", type=Path,
                        help="write read-only live preflight evidence and stop")
    args = parser.parse_args(argv)

    try:
        bundle_dir, manifest = jobs_release.build_bundle(args.source_root, args.output_root)
        jobs_release.verify_bundle(bundle_dir)

        if args.live_profile:
            if args.root is not None or args.pc_root is not None:
                raise jobs_release.ReleaseRefused(
                    "live profile refuses arbitrary --root/--pc-root"
                )
            home = Path.home().resolve()
            profile = args.live_profile
            release_sha = manifest["release_sha"]
            root = jobs_release.live_target_root(home, profile, release_sha)
            backup_dir = jobs_release.live_backup_dir(home, profile, release_sha)
            receipt_dir = jobs_release.live_receipt_dir(home)
            if args.receipt_out is not None and args.receipt_out.resolve() != receipt_dir:
                raise jobs_release.ReleaseRefused(
                    "live profile refuses undeclared receipt directory"
                )
            if args.write_live_preflight is not None:
                if args.apply:
                    raise jobs_release.ReleaseRefused(
                        "preflight creation is read-only and refuses --apply"
                    )
                jobs_release.write_live_preflight(
                    args.write_live_preflight,
                    bundle_dir,
                    profile=profile,
                    home_dir=home,
                    backup_dir=backup_dir,
                    receipt_dir=receipt_dir,
                )
                print(f"release_sha: {release_sha}")
                print(f"bundle_digest: {manifest['bundle_digest']}")
                print(f"target: {root}")
                print(f"backup: {backup_dir}")
                print(f"receipt-dir: {receipt_dir}")
                print(f"live preflight: ok ({args.write_live_preflight.resolve()})")
                return 0
            if not args.apply:
                raise jobs_release.ReleaseRefused(
                    "live profile requires --write-live-preflight or gated --apply"
                )
            if not args.expected_release_sha or not args.preflight:
                raise jobs_release.ReleaseRefused(
                    "live apply requires expected release SHA and preflight"
                )
            authorization = jobs_release.authorize_live_activation(
                bundle_dir,
                profile=profile,
                expected_release_sha=args.expected_release_sha,
                acknowledgement=args.acknowledge_live_activation or "",
                preflight_path=args.preflight,
                home_dir=home,
                backup_dir=backup_dir,
                receipt_dir=receipt_dir,
            )
            live_receipt = jobs_release.install_release(
                root,
                bundle_dir,
                dry_run=False,
                backup_dir=backup_dir,
                host_kind=profile,
                home_dir=home,
                live_authorization=authorization,
            )
            receipt_path = _write_receipt(live_receipt, receipt_dir)
            print(f"release_sha: {release_sha}")
            print(f"bundle_digest: {manifest['bundle_digest']}")
            print(f"bundle: {bundle_dir}")
            print("mode: live-apply")
            print(f"receipt: {receipt_path}")
            print("status: ok")
            return 0

        if args.root is None:
            raise jobs_release.ReleaseRefused(
                "simulated activation requires --root"
            )
        jobs_release.refuse_live_root(args.root)
        if args.pc_root is not None:
            jobs_release.refuse_live_root(args.pc_root)
        receipt_dir = args.receipt_out or (args.output_root / jobs_release.RECEIPT_SUBDIR)
        mode = "apply" if args.apply else "dry-run"

        mac_receipt = jobs_release.install_release(
            args.root, bundle_dir, dry_run=not args.apply, host_kind="mac"
        )
        mac_path = _write_receipt(mac_receipt, receipt_dir)

        pc_path = None
        if args.pc_root is not None:
            pc_receipt = jobs_release.install_release(
                args.pc_root, bundle_dir, dry_run=not args.apply, host_kind="pc"
            )
            pc_path = _write_receipt(pc_receipt, receipt_dir)
            jobs_release.check_parity(args.root, args.pc_root, bundle_dir)
    except (jobs_release.ReleaseError, OSError) as exc:
        print(f"activation refused: {exc}", file=sys.stderr)
        return 2

    print(f"release_sha: {manifest['release_sha']}")
    print(f"bundle: {bundle_dir}")
    print(f"manifest: {bundle_dir / jobs_release.MANIFEST_NAME}")
    print(f"mode: {mode}")
    print(f"receipt: {mac_path}")
    if pc_path is not None:
        print(f"receipt: {pc_path}")
    print("status: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
