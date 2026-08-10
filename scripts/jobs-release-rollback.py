#!/usr/bin/env python3
"""Roll back a Jobs activation to its prior recorded state (Task 10).

Restores only the exact targets recorded in the activation receipt: files
that existed before activation are restored from backup, files that did not
are removed (only if their installed bytes still match the receipt). Refuses
a dry-run receipt, a root mismatch, missing backups, and drifted files.
Evidence of the rollback is written next to the receipt.
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
    parser.add_argument("--root", required=True, type=Path,
                        help="target root to roll back")
    parser.add_argument("--receipt", required=True, type=Path,
                        help="activation receipt written by jobs-release-activate.py")
    parser.add_argument("--evidence-out", type=Path,
                        help="evidence directory (default: next to the receipt)")
    args = parser.parse_args(argv)

    try:
        evidence = jobs_release.rollback(
            args.root, args.receipt, evidence_dir=args.evidence_out
        )
    except (jobs_release.ReleaseError, OSError) as exc:
        print(f"rollback refused: {exc}", file=sys.stderr)
        return 2

    counts: dict[str, int] = {}
    for entry in evidence["targets"]:
        counts[entry["action"]] = counts.get(entry["action"], 0) + 1
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "no targets"
    print(f"rollback: ok ({summary})")
    print(f"evidence: {evidence['evidence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
