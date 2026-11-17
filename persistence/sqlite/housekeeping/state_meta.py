"""``StateMetaRepository`` on SQLite: the ``state_meta`` key/value rows, read and written through the store holder's
own ``get_meta``, ``set_meta`` and ``list_meta_prefix``."""

from __future__ import annotations

from typing import Optional

from persistence.housekeeping.state_meta import StateMetaEntry

from .._backend import SQLiteStore, register_repository, translated

__all__ = ["SQLiteStateMetaRepository"]


class SQLiteStateMetaRepository:
    """Core-owned bookkeeping keys of the bound profile's ``state.db``."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def get(self, key: str) -> Optional[str]:
        if not key:
            raise ValueError("state_meta key must be non-empty")
        with translated():
            return self._store.holder().get_meta(key)

    def set(self, key: str, value: str) -> None:
        if not key:
            raise ValueError("state_meta key must be non-empty")
        with translated():
            self._store.holder().set_meta(key, value)

    def insert_if_absent(self, key: str, value: str) -> bool:
        if not key:
            raise ValueError("state_meta key must be non-empty")
        with translated():
            return self._store.holder()._write_rowcount(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO NOTHING",
                (key, value),
            ) == 1

    def replace_if_current(self, key: str, expected: str, value: str) -> bool:
        if not key:
            raise ValueError("state_meta key must be non-empty")
        with translated():
            return self._store.holder()._write_rowcount(
                "UPDATE state_meta SET value = ? WHERE key = ? AND value = ?", (value, key, expected)
            ) == 1

    def move_if_current(self, source: str, target: str, expected: str, archive: str) -> bool:
        if not source or not target or source == target:
            raise ValueError("state_meta move requires two distinct non-empty keys")

        def move(conn):
            inserted = conn.execute(
                "INSERT INTO state_meta (key, value) SELECT ?, value FROM state_meta "
                "WHERE key = ? AND value = ? ON CONFLICT(key) DO NOTHING", (target, source, expected)
            )
            if inserted.rowcount != 1:
                return False
            conn.execute("UPDATE state_meta SET value = ? WHERE key = ?", (archive, source))
            return True

        with translated():
            return self._store.holder()._execute_write(move)

    def list_prefix(self, prefix: str) -> tuple[StateMetaEntry, ...]:
        if not prefix:
            raise ValueError("state_meta prefix must be non-empty")
        with translated():
            rows = self._store.holder().list_meta_prefix(prefix)
        # The store's LIKE folds ASCII case and reads in table order.
        return tuple(StateMetaEntry(key, value) for key, value in rows)


register_repository("state_meta", SQLiteStateMetaRepository)
