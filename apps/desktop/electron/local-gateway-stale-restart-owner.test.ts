import { expect, test, vi } from 'vitest'

import { createStaleGatewayAttacher, ensureLocalGateway } from './local-gateway'

test('a code mismatch attaches to the serving owner without draining active replies', async () => {
  const log = vi.fn()

  const endpoint = { profile_id: '/h', instance_id: 'serving', authority_epoch: 1, runtime_protocol: 1,
    api_origin: 'http://127.0.0.1:1234', capabilities: ['session-authority-v1'], supervisor: 'none', code_sha: 'old' }

  const ensure = vi.fn(async () => ({ code: 0, stdout: JSON.stringify({ state: 'ready', endpoint, client_code_sha: 'new' }) }))
  const attached = await ensureLocalGateway(ensure, undefined, createStaleGatewayAttacher(log))

  expect(attached.gatewayEndpoint.instance_id).toBe('serving')
  expect(ensure).toHaveBeenCalledTimes(1)
  expect(log).toHaveBeenCalled()
})
