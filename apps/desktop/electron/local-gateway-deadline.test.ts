import { EventEmitter } from 'node:events'

import { afterEach, expect, test, vi } from 'vitest'

afterEach(() => { vi.useRealTimers(); vi.doUnmock('node:child_process'); vi.resetModules() })

test('an ensure client that ignores termination is killed and its promise rejects within a second deadline', async () => {
  vi.useFakeTimers()

  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(), stderr: new EventEmitter(), kill: vi.fn(() => true)
  })

  vi.doMock('node:child_process', () => ({ spawn: () => child }))
  const { runGatewayEnsure } = await import('./local-gateway')
  const result = runGatewayEnsure({ command: 'owned-client', args: [], env: {}, shell: false }, '.', 'profile', {}, { timeoutMs: 10 })
  const rejected = expect(result).rejects.toThrow('timed out')
  await vi.advanceTimersByTimeAsync(10)
  expect(child.kill.mock.calls).toEqual([[]])
  await vi.advanceTimersByTimeAsync(1000)
  await rejected
  expect(child.kill.mock.calls).toEqual([[], ['SIGKILL']])
})

// An explicit lifecycle command has its own bounded CLI client. A timeout
// terminates that client without taking the gateway owner with it.
test('an explicit restart client timeout resolves when its CLI obeys SIGTERM', async () => {
  vi.useFakeTimers()

  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(), stderr: new EventEmitter(),
    kill: vi.fn((signal?: string) => {
      if (!signal) { queueMicrotask(() => child.emit('close', null)) }

      return true
    })
  })

  vi.doMock('node:child_process', () => ({ spawn: () => child }))
  const { runGatewayEnsure } = await import('./local-gateway')

  const result = runGatewayEnsure({ command: 'owned-client', args: [], env: {}, shell: false }, '.', 'default', {},
    { timeoutMs: 10, label: 'hermes gateway restart' })

  await vi.advanceTimersByTimeAsync(10)
  await expect(result).resolves.toEqual({ code: 7, stdout: '', stderr: 'hermes gateway restart timed out' })
  expect(child.kill.mock.calls).toEqual([[]])
})

// `hermes gateway ensure` owns a 60 s startup deadline (hermes_cli/gateway_runtime.py
// DEFAULT_ENSURE_TIMEOUT) and always answers with a protocol verdict by then. Killing the client
// earlier turned a slow-but-valid cold start into "produced no result … Update Hermes".
test('the default ensure client outlives the protocol deadline and reads a late verdict', async () => {
  vi.useFakeTimers()

  const child = Object.assign(new EventEmitter(), {
    stdout: new EventEmitter(), stderr: new EventEmitter(), kill: vi.fn(() => true)
  })

  vi.doMock('node:child_process', () => ({ spawn: () => child }))
  const { runGatewayEnsure } = await import('./local-gateway')
  const result = runGatewayEnsure({ command: 'owned-client', args: [], env: {}, shell: false }, '.', 'profile', {})
  await vi.advanceTimersByTimeAsync(60_000)
  child.stdout.emit('data', Buffer.from('{"state":"starting","reason_code":"deadline"}'))
  child.emit('close', 5)
  await expect(result).resolves.toEqual({ code: 5, stdout: '{"state":"starting","reason_code":"deadline"}', stderr: '' })
  expect(child.kill).not.toHaveBeenCalled()
})
