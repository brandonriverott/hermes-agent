import json

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


@pytest.mark.parametrize('cwd', ['', 'relative/path', 'C:relative', '/', 'workspace'])
def test_restore_rejects_relative_or_empty_cwd(cwd):
    if cwd == '/':
        restore_policy(_policy(cwd))  # root is absolute; still a valid cwd
        return
    with pytest.raises(RuntimeStoreError):
        restore_policy(_policy(cwd))
