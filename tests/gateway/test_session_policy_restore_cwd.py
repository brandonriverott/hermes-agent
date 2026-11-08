import json
import os

import pytest

from gateway.session_policy import restore_policy
from hermes_state_runtime import RuntimeStoreError


def _policy(cwd):
    # The local terminal backend must carry the session cwd, or restore refuses with "terminal policy mismatch".
    return {
        'source': 'tool', 'platform': 'cli', 'cwd': cwd, 'model': 'm', 'toolsets': [],
        'config_json': '{}', 'request_json': '{}', 'terminal_json': json.dumps({'TERMINAL_CWD': cwd}),
    }


@pytest.mark.parametrize('cwd', [
    '/Users/brandon/Documents/x',
    'C:/Users/casp3/AppData/Local/hermes/profiles/default/workspace/inbox-watch',
    'M:/PCStack/Hermes/pc-fleet/profiles/default/workspace/inbox-watch',
    'C:\\Users\\casp3',
])
def test_restore_accepts_absolute_and_windows_absolute_cwd(cwd):
    assert restore_policy(_policy(cwd)).cwd == cwd


@pytest.mark.parametrize('cwd', ['', 'relative/path', 'C:relative', 'C:', 'workspace'])
def test_restore_rejects_relative_or_empty_cwd(cwd):
    with pytest.raises(RuntimeStoreError):
        restore_policy(_policy(cwd))


def test_restore_accepts_root_cwd():
    assert restore_policy(_policy('/')).cwd == '/'


@pytest.mark.skipif(os.name == 'nt', reason='a drive path is a real absolute path on Windows')
def test_windows_cwd_is_not_ready_when_a_same_named_relative_folder_exists(tmp_path, monkeypatch):
    # The repo root really holds stray "G:/..." and "C:\..." folders; a restored Windows cwd must not match them.
    from gateway.session_policy import workspace_ready
    (tmp_path / 'C:' / 'Users' / 'x').mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert workspace_ready(restore_policy(_policy('C:/Users/x'))) is False
