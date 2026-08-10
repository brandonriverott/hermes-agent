#!/usr/bin/env python3
"""Build, verify, and dry-run/apply an immutable Jobs release (Task 10).

Defaults to dry-run: no target root is mutated and the activation receipt
records exactly what WOULD happen. --apply performs the install on the given
simulated target roots with a backup of every replaced file and writes an
activation receipt that the rollback tool can restore from.

Refusals: dirty source tree, missing manifest files, digest mismatch, target
drift, and any live Hermes root (~/.hermes / ~/.local). Real activation is a
Task 11 decision; this tool never touches a live Hermes directory, cron,
gateway restart, or Tailscale connection.
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
    parser.add_argument("--root", required=True, type=Path,
                        help="Mac target root (simulated in tests)")
    parser.add_argument("--pc-root", type=Path,
                        help="PC target root (simulated in tests)")
    parser.add_argument("--apply", action="store_true",
                        help="mutate target roots; default is dry-run")
    parser.add_argument("--receipt-out", type=Path,
                        help="receipt directory (default <output-root>/activation-receipts)")
    args = parser.parse_args(argv)

    receipt_dir = args.receipt_out or (args.output_root / jobs_release.RECEIPT_SUBDIR)
    try:
        jobs_release.refuse_live_root(args.root)
        if args.pc_root is not None:
            jobs_release.refuse_live_root(args.pc_root)

        bundle_dir, manifest = jobs_release.build_bundle(args.source_root, args.output_root)
        jobs_release.verify_bundle(bundle_dir)
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
