"""Publication behavior when a live process holds or runs from an existing entry."""

from pathlib import Path

import pytest

from pm import install
from pm.store import Store


class _VerifiedPackage:
    name = "tool"

    def verify(self, entry, target):
        return None if (entry / "bin/tool").is_file() else "tool is missing"


def _write_entry(path: Path, contents: bytes = b"tool") -> None:
    (path / "bin").mkdir(parents=True)
    (path / "bin/tool").write_bytes(contents)


def _hold_entry_rename(monkeypatch, entry, previous) -> None:
    real_rename = Path.rename

    def held_entry_rename(self, destination):
        if self == entry and destination == previous:
            raise PermissionError("live entry is held")
        return real_rename(self, destination)

    monkeypatch.setattr(install, "retry_held", lambda operation: operation())
    monkeypatch.setattr(Path, "rename", held_entry_rename)


def test_publish_keeps_a_byte_identical_held_entry(tmp_path, monkeypatch):
    """Regression for #139088: an unchanged live entry need not be republished."""
    store = Store(tmp_path / "store")
    entry, previous, staged = (store.entry("tool-1"), store.entry(".previous-tool-1"),
                               tmp_path / "staged")
    _write_entry(entry)
    _write_entry(staged)
    _hold_entry_rename(monkeypatch, entry, previous)

    with install._publish_entry(_VerifiedPackage(), store, staged, entry, previous, "target"):
        pass

    assert (entry / "bin/tool").read_bytes() == b"tool"
    assert not previous.exists()


def test_publish_leaves_an_unheld_identical_entry_in_place(tmp_path):
    """POSIX renames a running entry; an identical one must keep its inode so running processes
    keep a resolvable executable path (macOS TCC denies processes whose path vanished)."""
    store = Store(tmp_path / "store")
    entry, previous, staged = (store.entry("tool-1"), store.entry(".previous-tool-1"),
                               tmp_path / "staged")
    _write_entry(entry)
    _write_entry(staged)
    inode = (entry / "bin/tool").stat().st_ino

    with install._publish_entry(_VerifiedPackage(), store, staged, entry, previous, "target"):
        pass

    assert (entry / "bin/tool").stat().st_ino == inode
    assert not previous.exists()


def test_failure_after_keeping_an_identical_entry_leaves_it_untouched(tmp_path):
    store = Store(tmp_path / "store")
    entry, previous, staged = (store.entry("tool-1"), store.entry(".previous-tool-1"),
                               tmp_path / "staged")
    _write_entry(entry)
    _write_entry(staged)
    inode = (entry / "bin/tool").stat().st_ino

    with pytest.raises(RuntimeError, match="facts commit failed"):
        with install._publish_entry(_VerifiedPackage(), store, staged, entry, previous, "target"):
            raise RuntimeError("facts commit failed")

    assert (entry / "bin/tool").stat().st_ino == inode
    assert not previous.exists()


def test_publish_replaces_a_symlinked_entry_even_when_identical(tmp_path):
    store = Store(tmp_path / "store")
    entry, previous, staged = (store.entry("tool-1"), store.entry(".previous-tool-1"),
                               tmp_path / "staged")
    target_dir = tmp_path / "elsewhere"
    _write_entry(target_dir)
    store.root.mkdir(parents=True, exist_ok=True)
    entry.symlink_to(target_dir, target_is_directory=True)
    _write_entry(staged)

    with install._publish_entry(_VerifiedPackage(), store, staged, entry, previous, "target"):
        pass

    assert not entry.is_symlink()
    assert (entry / "bin/tool").read_bytes() == b"tool"
    assert (target_dir / "bin/tool").is_file()


def test_fresh_copy_republishes_an_unheld_identical_entry(tmp_path):
    """A fresh copy must own independent bytes, so identity alone does not keep the entry."""
    store = Store(tmp_path / "store")
    entry, previous, staged = (store.entry("tool-1"), store.entry(".previous-tool-1"),
                               tmp_path / "staged")
    _write_entry(entry)
    _write_entry(staged)
    inode = (entry / "bin/tool").stat().st_ino

    with install._publish_entry(_VerifiedPackage(), store, staged, entry, previous, "target",
                                keep_identical=False):
        pass

    assert (entry / "bin/tool").stat().st_ino != inode
    assert not previous.exists()


def test_publish_replaces_an_unheld_entry_that_differs(tmp_path):
    store = Store(tmp_path / "store")
    entry, previous, staged = (store.entry("tool-1"), store.entry(".previous-tool-1"),
                               tmp_path / "staged")
    _write_entry(entry)
    _write_entry(staged, b"replacement")

    with install._publish_entry(_VerifiedPackage(), store, staged, entry, previous, "target"):
        pass

    assert (entry / "bin/tool").read_bytes() == b"replacement"
    assert not previous.exists()


def test_publish_raises_when_a_held_entry_differs_from_staged_bytes(tmp_path, monkeypatch):
    """A held entry remains an error unless it is exactly the staged replacement."""
    store = Store(tmp_path / "store")
    entry, previous, staged = (store.entry("tool-1"), store.entry(".previous-tool-1"),
                               tmp_path / "staged")
    _write_entry(entry)
    _write_entry(staged, b"replacement")
    _hold_entry_rename(monkeypatch, entry, previous)

    with pytest.raises(PermissionError, match="live entry is held"):
        with install._publish_entry(_VerifiedPackage(), store, staged, entry, previous, "target"):
            pass
