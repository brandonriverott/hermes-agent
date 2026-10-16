import { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import type { ClientSessionState } from '@/app/types'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $currentModel, setCurrentModel, setCurrentProvider } from '@/store/session'
import type * as SessionStates from '@/store/session-states'
import { $sessionStates, publishSessionState } from '@/store/session-states'

import { deferred } from '../../../test/deferred'

import { useModelControls } from './use-model-controls'

vi.mock('@/hermes', () => ({ getGlobalModelInfo: vi.fn(), setApiRequestProfile: vi.fn() }))
vi.mock('@/store/notifications', () => ({
  dismissNotification: vi.fn(), notify: vi.fn(), notifyError: vi.fn()
}))
vi.mock('@/store/session-states', async importOriginal => {
  const actual = await importOriginal<typeof SessionStates>()

  return {
    ...actual,
    sessionTileDelegate: () => ({
      updateSession: (id: string, update: (state: ClientSessionState) => ClientSessionState) => {
        const state = update(actual.$sessionStates.get()[id])
        actual.publishSessionState(id, state)
        return state
      }
    })
  }
})

function seed(id: string, model: string) {
  publishSessionState(id, {
    ...createClientSessionState(`stored-${id}`), model, provider: 'nous', reasoningEffortWire: 'max'
  })
}

beforeEach(() => {
  $sessionStates.set({})
  $activeGatewayProfile.set('default')
  $activeSessionId.set('view-a')
  seed('view-a', 'old-a')
  setCurrentModel('old-a')
  setCurrentProvider('nous')
})

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  $sessionStates.set({})
})

it.each(['immediate', 'queued'])('preserves confirmed reasoning metadata before a %s switch response', async mode => {
  let switches = 0
  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'session.active_list') { return { sessions: [] } as never }
    if (method === 'session.activate') { return {} as never }
    switches += 1
    if (mode === 'queued' && switches === 1) { throw new Error('session_busy') }

    // The gateway can report the new route before acknowledging config.set.
    publishSessionState('view-a', {
      ...$sessionStates.get()['view-a'],
      model: 'new-a', provider: 'xai', reasoningEffort: 'ultra', reasoningEffortWire: 'max'
    })
    return {} as never
  })
  const { result } = renderHook(() => useModelControls({ queryClient: new QueryClient(), requestGateway }))

  await act(() => result.current.selectModel({ model: 'new-a', provider: 'xai' }))
  await waitFor(() => expect(switches).toBe(mode === 'queued' ? 2 : 1), { timeout: 4000 })

  expect(PRIMARY_SESSION_VIEW.$model.get()).toBe('new-a')
  expect(PRIMARY_SESSION_VIEW.$reasoningEffort.get()).toBe('ultra')
  expect(PRIMARY_SESSION_VIEW.$reasoningEffortWire.get()).toBe('max')
})

it.each([false, true])('paints a committed busy pick into its chat after navigation=%s', async navigate => {
  let working = true
  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'session.active_list') {
      return { sessions: [{ id: 'view-a', status: working ? 'working' : 'idle' }] } as never
    }
    if (method === 'session.activate') { return {} as never }
    if (working) { throw new Error('session_busy') }
    return {} as never
  })
  const { result } = renderHook(() => useModelControls({ queryClient: new QueryClient(), requestGateway }))

  await act(() => result.current.selectModel({ model: 'new-a', provider: 'xai' }))
  expect(PRIMARY_SESSION_VIEW.$model.get()).toBe('old-a')
  if (navigate) {
    seed('view-b', 'model-b')
    $activeSessionId.set('view-b')
    setCurrentModel('model-b')
  }

  working = false
  await waitFor(() => expect($sessionStates.get()['view-a'].model).toBe('new-a'), { timeout: 4000 })
  expect($sessionStates.get()['view-a']).toMatchObject({ provider: 'xai', reasoningEffortWire: '' })
  expect(PRIMARY_SESSION_VIEW.$model.get()).toBe(navigate ? 'model-b' : 'new-a')
  expect($currentModel.get()).toBe(navigate ? 'model-b' : 'new-a')
})

it('rolls a rejected pick back in its chat after navigation without changing the new foreground', async () => {
  const response = deferred<never>()
  const requestGateway = vi.fn(async () => response.promise)
  const { result } = renderHook(() => useModelControls({ queryClient: new QueryClient(), requestGateway }))

  const pending = result.current.selectModel({ model: 'new-a', provider: 'xai' })
  expect(PRIMARY_SESSION_VIEW.$model.get()).toBe('new-a')
  seed('view-b', 'model-b')
  $activeSessionId.set('view-b')
  setCurrentModel('model-b')
  response.reject(new Error('revision_conflict'))
  await expect(pending).resolves.toBe(false)

  expect($sessionStates.get()['view-a']).toMatchObject({ model: 'old-a', provider: 'nous', reasoningEffortWire: 'max' })
  expect(PRIMARY_SESSION_VIEW.$model.get()).toBe('model-b')
  expect($currentModel.get()).toBe('model-b')
})
