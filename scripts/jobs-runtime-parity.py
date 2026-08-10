#!/usr/bin/env python3
"""Verify byte-identical Jobs runtime install across simulated Mac/PC roots.

Refuses (exit 2) when any manifest file is missing on either root or its
SHA-256 differs between roots or from the release manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_cli import jobs_release  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path,
                        help="path to a jobs-release-v1 bundle directory")
    parser.add_argument("--mac-root", required=True, type=Path,
                        help="Mac target root (simulated in tests)")
    parser.add_argument("--pc-root", required=True, type=Path,
                        help="PC target root (simulated in tests)")
    args = parser.parse_args(argv)

    try:
        result = jobs_release.check_parity(args.mac_root, args.pc_root, args.bundle)
    except jobs_release.ReleaseError as exc:
        print(f"parity refused: {exc}", file=sys.stderr)
        return 2

    print(f"parity: ok ({result['checked']} files identical on mac and pc roots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
