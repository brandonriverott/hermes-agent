#!/usr/bin/env python3
"""Point the main Hermes gateway launchd job at one immutable runtime.

Defaults to a no-write inspection.  ``--apply`` preserves a byte-for-byte
backup before atomically replacing only the launchd plist's WorkingDirectory.
Service restart is intentionally a separate, explicit activation step.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import sys
import tempfile
from pathlib import Path


EXPECTED_LABEL = "ai.hermes.gateway"


def update_pointer(plist_path: Path, runtime_root: Path, backup: Path, *, apply: bool) -> str:
    """Validate and, when authorized, atomically replace only WorkingDirectory."""
    raw = plist_path.read_bytes()
    document = plistlib.loads(raw)
    if document.get("Label") != EXPECTED_LABEL:
        raise ValueError("refusing non-gateway launchd plist")
    if not runtime_root.is_absolute():
        raise ValueError("runtime root must be absolute")
    if not apply:
        return "dry-run"

    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise ValueError("backup already exists")
    backup.write_bytes(raw)
    document["WorkingDirectory"] = str(runtime_root)
    replacement = plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=False)
    with tempfile.NamedTemporaryFile(dir=plist_path.parent, delete=False) as handle:
        handle.write(replacement)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, plist_path)
    finally:
        temporary.unlink(missing_ok=True)
    return "applied"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plist", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    backup = args.backup or args.plist.with_suffix(args.plist.suffix + ".jobs-backup")
    try:
        status = update_pointer(args.plist, args.runtime_root, backup, apply=args.apply)
    except (OSError, ValueError, plistlib.InvalidFileException) as exc:
        print(f"pointer refused: {exc}", file=sys.stderr)
        return 2
    print(f"pointer: {status}")
    if args.apply:
        print(f"backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
