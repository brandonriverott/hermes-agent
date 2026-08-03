"""Read-only containment checkpoint for the live Jobs and Kanban databases.

Run before, between, and after every Jobs V3 test group. It opens nothing for
writing: the SHA-256, size, and nanosecond mtime are taken from the file itself,
and the SQLite integrity read goes through an immutable URI so the connection
cannot create a journal, a WAL, or a hot-journal rollback beside the database.

Usage::

    python -m tests.hermes_cli.jobs_containment_check           # print tuples
    python -m tests.hermes_cli.jobs_containment_check --expect  # compare + exit 1

``--expect`` compares against the frozen incident baselines below; any drift in
checksum, size, or mtime_ns is a containment failure, not a warning.
"""

from __future__ import annotations

import hashlib
import os
import pwd
import sqlite3
import sys
from pathlib import Path

# The operator's real home, read from the password database rather than ``HOME``.
# Every test in this suite deliberately re-points ``HOME``, and a containment
# check that followed it would cheerfully verify the temporary directory it was
# supposed to be proving nothing escaped from.
_LIVE_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir)

LIVE = {
    "jobs": _LIVE_HOME / ".hermes" / "jobs.db",
    "kanban": _LIVE_HOME / ".hermes" / "kanban.db",
}

BASELINE = {
    "jobs": (
        "b38a2e090d20a6e96c5170c327578e2e0cc2f4311b5725e173a211b79347ff4c",
        110592,
        1785650498820842263,
    ),
    "kanban": (
        "fc6790228748bbe124d6249432d20c4beaa8bfe733ec76f83c7325158696c604",
        12828672,
        1785639383998822199,
    ),
}


def observe(path: Path):
    st = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return (digest.hexdigest(), st.st_size, st.st_mtime_ns)


def integrity(path: Path) -> str:
    """``PRAGMA integrity_check`` over an immutable, read-only URI."""
    uri = f"file:{path}?immutable=1&mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()


def main(argv) -> int:
    label = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else ""
    expect = "--expect" in argv
    failed = False
    for name, path in LIVE.items():
        seen = observe(path)
        ok = seen == BASELINE[name]
        print(
            f"containment{(' ' + label) if label else ''} {name} "
            f"sha256={seen[0]} size={seen[1]} mtime_ns={seen[2]} "
            f"integrity={integrity(path)} match={ok}"
        )
        if expect and not ok:
            failed = True
            print(f"  expected {BASELINE[name]}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main(sys.argv))
