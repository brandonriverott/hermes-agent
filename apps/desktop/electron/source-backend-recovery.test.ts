import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import * as probes from './backend-probes'
import { createInstalledRuntimeGate } from './backend-resolution'
import { resolveInstalledSourceBackend, resolveSourceInstallationBackend } from './source-backend'
import * as launchers from './updater-process'

test('a present installation with a failed runtime probe remains an installed-runtime error', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'runtime-recovery-'))
  fs.mkdirSync(path.join(root, 'hermes_cli'))
  fs.writeFileSync(path.join(root, 'hermes_cli', 'main.py'), '')
  fs.mkdirSync(path.join(root, '.hermes', 'bin'), { recursive: true })
  fs.writeFileSync(path.join(root, '.hermes', 'bin', 'hermes'), '')
  const probe = vi.spyOn(probes, 'execProbe').mockRejectedValue(new Error('probe deadline exceeded'))
  try {
    await assert.rejects(resolveSourceInstallationBackend(root, [], { isWindows: false }), /installed.*probe deadline exceeded/i)
    probe.mockResolvedValue(undefined)
    assert.equal((await resolveSourceInstallationBackend(root, [], { isWindows: false }))?.root, root)
    probe.mockClear()
    const concurrent = await Promise.all([
      resolveSourceInstallationBackend(root, ['serve'], { isWindows: false }),
      resolveSourceInstallationBackend(root, ['gateway'], { isWindows: false })
    ])
    assert.equal(probe.mock.calls.length, 1)
    assert.deepEqual(concurrent.map(backend => backend?.args), [['serve'], ['gateway']])
    assert.equal(await resolveSourceInstallationBackend(path.join(root, 'missing'), []), null)
  } finally {
    probe.mockRestore()
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('a failed canonical runtime falls through to a healthy published installation', async () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'runtime-alternate-'))
  const canonical = path.join(base, 'canonical')
  const alternate = path.join(base, 'alternate')
  for (const root of [canonical, alternate]) {
    fs.mkdirSync(path.join(root, 'hermes_cli'), { recursive: true })
    fs.writeFileSync(path.join(root, 'hermes_cli', 'main.py'), '')
    fs.mkdirSync(path.join(root, '.hermes', 'bin'), { recursive: true })
    fs.writeFileSync(path.join(root, '.hermes', 'bin', 'hermes'), '')
  }
  const probe = vi.spyOn(probes, 'execProbe').mockImplementation(async (_command, _args, options) => {
    if (options?.cwd === canonical) { throw new Error('canonical runtime unavailable') }
  })
  const discover = vi.spyOn(launchers, 'userLauncherInstallRoot').mockResolvedValue({ root: alternate, launcher: '/published/hermes' })
  try {
    assert.equal((await resolveInstalledSourceBackend(canonical, ['serve'], { isWindows: false }))?.root, alternate)
    discover.mockResolvedValue(null)
    await assert.rejects(resolveInstalledSourceBackend(canonical, [], { isWindows: false }), /canonical runtime unavailable/)
    discover.mockResolvedValue({ root: canonical, launcher: '/published/hermes' })
    probe.mockClear()
    await assert.rejects(resolveInstalledSourceBackend(canonical, [], { isWindows: false }), /canonical runtime unavailable/)
    assert.equal(probe.mock.calls.length, 1)
  } finally {
    probe.mockRestore()
    discover.mockRestore()
    fs.rmSync(base, { recursive: true, force: true })
  }
})

test('ignore-existing and explicit repair skip the entire installed discovery ladder', async () => {
  const discover = vi.spyOn(launchers, 'userLauncherInstallRoot').mockRejectedValue(new Error('broken published launcher'))
  const probe = vi.spyOn(probes, 'execProbe').mockRejectedValue(new Error('must not probe'))
  try {
    for (const repair of [false, true]) {
      const gate = createInstalledRuntimeGate(repair ? {} : { HERMES_DESKTOP_IGNORE_EXISTING: '1' }, () => {})
      assert.equal(await gate.resolve('/absent', () => resolveInstalledSourceBackend('/absent', []), repair), null)
    }
    assert.equal(discover.mock.calls.length, 0)
    assert.equal(probe.mock.calls.length, 0)
  } finally {
    discover.mockRestore()
    probe.mockRestore()
  }
})
