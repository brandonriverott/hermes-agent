"""Opening a chat must not wait for a slow status-bar skill inventory."""
import asyncio
import threading
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_snapshot_returns_history_metadata_before_slow_inventory_then_publishes_it(monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway import session_local, session_local_inventory, session_policy

    release = threading.Event()
    events = []
    policy = SimpleNamespace(source='cli')
    live = SimpleNamespace(source=object(), event_stream=SimpleNamespace(
        publish=lambda sid, payload, **kwargs: events.append((sid, payload, kwargs))))
    authority = SimpleNamespace(profile_id='profile', sessions={'chat': live}, runner=object(),
                                agent=lambda ref: None)
    monkeypatch.setattr(session_local, 'local_session_info', lambda authority, ref: {'model': 'current'})
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)

    def slow_inventory(*args):
        release.wait(5)
        return {'skills': {'general': ['one']}, 'tools': {'other': ['terminal']}}

    monkeypatch.setattr(session_local_inventory, 'session_inventory', slow_inventory)
    try:
        info = await asyncio.wait_for(session_local_inventory.snapshot_info(
            authority, SessionRef('profile', 'chat'), defer_inventory=True), timeout=2)
        assert info['model'] == 'current'
        assert info.get('skills') != {'general': ['one']}
    finally:
        release.set()
    from tests.waiting import async_wait_until
    await async_wait_until(lambda: bool(events), timeout=2)
    assert events == [('chat', {'skills': {'general': ['one']}, 'tools': {'other': ['terminal']},
                                'model': 'current', 'tool_count': 1}, {'event_type': 'session.info'})]


@pytest.mark.asyncio
async def test_cli_viewer_of_gui_chat_waits_while_desktop_shares_the_inventory_scan(monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway import session_local, session_local_inventory, session_policy

    release = threading.Event()
    events = []
    policy = SimpleNamespace(source='gui')
    live = SimpleNamespace(source=object(), event_stream=SimpleNamespace(
        publish=lambda *args, **kwargs: events.append((args, kwargs))))
    authority = SimpleNamespace(profile_id='profile', sessions={'chat': live}, runner=object(),
                                agent=lambda ref: None)
    monkeypatch.setattr(session_local, 'local_session_info', lambda authority, ref: {'model': 'current'})
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)

    scans = []
    def slow_inventory(*args):
        scans.append(True)
        release.wait(5)
        return {'tools': {'other': ['terminal']}}

    monkeypatch.setattr(session_local_inventory, 'session_inventory', slow_inventory)
    task = asyncio.create_task(session_local_inventory.snapshot_info(authority, SessionRef('profile', 'chat')))
    try:
        desktop = await asyncio.wait_for(session_local_inventory.snapshot_info(
            authority, SessionRef('profile', 'chat'), defer_inventory=True), timeout=2)
        assert 'tools' not in desktop
        assert not task.done()
    finally:
        release.set()
    assert (await task)['tools'] == {'other': ['terminal']}
    assert scans == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize('adopt', [False, True])
async def test_desktop_create_and_adopt_defer_for_current_viewer_but_cli_resume_waits(tmp_path, monkeypatch, adopt):
    from gateway import run, session_local_inventory
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from tests.waiting import async_wait_until

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'model: {default: test, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
    inventory = {'tools': {'other': ['terminal']}}
    monkeypatch.setattr(session_local_inventory, 'session_inventory', lambda *args: inventory)

    async def call(method, **params):
        reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, reply
        return reply['result']

    existing = None
    if adopt:
        existing = await call('session.create', request_id='cli', source='cli', title='Shared', cwd=str(tmp_path))
    release = threading.Event()
    entered = threading.Event()
    def scan(*args):
        entered.set()
        release.wait(5)
        return inventory
    monkeypatch.setattr(session_local_inventory, 'session_inventory', scan)
    cli = None
    try:
        desktop = await asyncio.wait_for(call('session.create', request_id='desktop', source='gui',
                                              title='Shared', cwd=str(tmp_path)), timeout=2)
        assert 'tools' not in desktop['info']
        if existing:
            assert desktop['session_id'] == existing['session_id']
        cli = asyncio.create_task(call('session.resume', session_id=desktop['session_id']))
        # A second Desktop read reaches the same scan after the CLI has attached.
        again = await asyncio.wait_for(call('session.resume', session_id=desktop['session_id'], source='desktop'), timeout=2)
        await async_wait_until(entered.is_set, timeout=2)
        assert 'tools' not in again['info']
        assert not cli.done()
    finally:
        release.set()
    assert (await cli)['info']['tools'] == inventory['tools']
