"""A surviving gateway takes over dispatch without a process restart."""

import asyncio
import threading

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.kanban_watchers_common import (
    _acquire_singleton_lock,
    _release_singleton_lock,
)


@pytest.fixture
def election(tmp_path, monkeypatch):
    from hermes_cli import config, kanban_db

    monkeypatch.delenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", raising=False)
    monkeypatch.setattr(kanban_db, "kanban_home", lambda: tmp_path)
    monkeypatch.setattr(config, "load_config", lambda: {"kanban": {}})
    return tmp_path / "kanban" / ".dispatcher.lock"


@pytest.mark.platforms("posix")
def test_standby_takes_over_only_after_owner_releases(election, monkeypatch):
    import gateway.kanban_watchers as watchers

    owner, state = _acquire_singleton_lock(election)
    assert state == "held"
    calls = []

    class Standby(GatewayKanbanWatchersMixin):
        _running = True

        async def _sleep_between_ticks(self, interval):
            if not self._running:
                return
            calls.append("waiting")
            assert not self._owns_kanban_dispatcher_lock()
            other, state = _acquire_singleton_lock(election)
            assert other is None and state == "contended"
            _release_singleton_lock(owner)

    runner = Standby()

    async def dispatch_thread(fn, *args):
        if fn.__name__ == "tick_once":
            calls.append("dispatch")
            assert runner._owns_kanban_dispatcher_lock()
            other, state = _acquire_singleton_lock(election)
            assert other is None and state == "contended"
            runner._running = False
        return False if fn.__name__ == "ready_nonempty" else []

    async def initial_sleep(delay):
        return None

    monkeypatch.setattr(watchers.asyncio, "sleep", initial_sleep)
    monkeypatch.setattr(watchers, "_dispatcher_service", dispatch_thread)
    monkeypatch.setattr(watchers, "_kanban_dispatch_allowed", lambda: True)
    try:
        asyncio.run(asyncio.wait_for(runner._kanban_dispatcher_watcher(), 3))
        assert calls == ["waiting", "dispatch"]
        assert not runner._owns_kanban_dispatcher_lock()
        successor, state = _acquire_singleton_lock(election)
        try:
            assert state == "held"
        finally:
            _release_singleton_lock(successor)
    finally:
        if not owner.closed:
            _release_singleton_lock(owner)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("phase", ["standby", "initial_delay", "between_ticks", "disabled"])
def test_shutdown_releases_election_without_dispatch_when_disabled(election, monkeypatch, phase):
    import gateway.kanban_watchers as watchers
    from hermes_cli import config

    owner = None
    if phase == "standby":
        owner, state = _acquire_singleton_lock(election)
        assert state == "held"
    if phase == "disabled":
        monkeypatch.setattr(config, "load_config", lambda: {"kanban": {"dispatch_in_gateway": False}})

    async def cancelled(*args):
        raise asyncio.CancelledError

    class Runner(GatewayKanbanWatchersMixin):
        _running = True
        _sleep_between_ticks = cancelled

    async def dispatch_thread(fn, *args):
        return [] if fn.__name__ != "ready_nonempty" else False

    async def initial_sleep(delay):
        if phase == "initial_delay":
            raise asyncio.CancelledError

    monkeypatch.setattr(watchers.asyncio, "sleep", initial_sleep)
    monkeypatch.setattr(watchers, "_dispatcher_service", dispatch_thread)
    monkeypatch.setattr(watchers, "_kanban_dispatch_allowed", lambda: False)
    runner = Runner()
    try:
        if phase == "disabled":
            asyncio.run(runner._kanban_dispatcher_watcher())
        else:
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(runner._kanban_dispatcher_watcher())
        assert not runner._owns_kanban_dispatcher_lock()
    finally:
        _release_singleton_lock(owner)
    successor, state = _acquire_singleton_lock(election)
    try:
        assert state == "held"
    finally:
        _release_singleton_lock(successor)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("cancel_all", [False, True])
def test_cancelled_dispatch_holds_lock_until_worker_thread_finishes(election, monkeypatch, cancel_all):
    import gateway.kanban_watchers as watchers
    from hermes_cli import config, kanban_db_dispatch

    release = threading.Event()
    monkeypatch.setattr(config, "load_config", lambda: {"kanban": {"auto_decompose": False}})
    monkeypatch.setattr(kanban_db_dispatch, "reap_worker_zombies", list)
    monkeypatch.setattr(watchers, "_kanban_dispatch_allowed", lambda: True)

    async def initial_sleep(delay):
        return None

    monkeypatch.setattr(watchers.asyncio, "sleep", initial_sleep)

    async def scenario():
        started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def blocking_tick(self):
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5), "test failed to release the dispatcher thread"
            return []

        monkeypatch.setattr(watchers._KanbanDispatcher, "tick_once", blocking_tick)
        runner = GatewayKanbanWatchersMixin()
        runner._running = True
        task = asyncio.create_task(runner._kanban_dispatcher_watcher())
        try:
            await asyncio.wait_for(started.wait(), 3)
            for _ in range(2):
                targets = asyncio.all_tasks() - {asyncio.current_task()} if cancel_all else {task}
                for target in targets:
                    target.cancel()
                progressed = asyncio.Event()
                loop.call_soon(progressed.set)
                await progressed.wait()
                assert not task.done()
                other, state = _acquire_singleton_lock(election)
                _release_singleton_lock(other)
                assert state == "contended"
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 3)
        successor, state = _acquire_singleton_lock(election)
        try:
            assert state == "held"
        finally:
            _release_singleton_lock(successor)

    asyncio.run(scenario())
