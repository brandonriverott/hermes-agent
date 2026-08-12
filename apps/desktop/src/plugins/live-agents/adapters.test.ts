import { host } from '@hermes/plugin-sdk'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { controlRun, loadFleetEvidence } from './adapters'
import { aggregateFleet, type FleetRun } from './model'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('live agents public-source adapters', () => {
  it('uses read-only RPCs, reports missing registered seams, and never invokes a provider', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'agents.list') {return { processes: [{ session_id: 'p1', command: 'npm test -- PROMPT_SENTINEL', status: 'running', uptime: 2, output_tail: 'RAW_OUTPUT_SENTINEL /Users/example/private.txt' }] }}

      if (method === 'delegation.status') {return { active: [{ subagent_id: 'd1', goal: 'PROMPT_SENTINEL must never render', current_tool: 'read_file', status: 'running' }] }}

      if (method === 'spawn_tree.list') {return { entries: [] }}

      if (method === 'projects.list') {return { projects: [{ name: 'Hermes Agent', board_slug: 'main', primary_path: '/Users/example/repo' }] }}

      throw new Error(`unknown method: ${method}`)
    })

    const rest = vi.fn(async () => ({
      profiles: [{ name: 'argus', description: 'Researches durable questions.', gateway_running: true }],
      runs: [{ id: 'r1', task_id: 't1', identity_key: 'kanban-worker-0123456789abcdef', title: 'Ship it', board: 'main', status: 'done', ended_at: 20 }]
    }))

    const remote = vi.fn(async () => ({
      agents: [
        { id: 'mac-builder', identity_key: 'profile:default', name: 'Mac builder', machine: 'Mac', status: 'idle', updated_at: 40, prompt: 'REMOTE_PROMPT_SENTINEL' },
        { id: 'pc-builder', name: 'PC builder', machine: 'PC', status: 'unreachable', updated_at: 30 }
      ]
    }))

    const snapshot = await loadFleetEvidence(request, rest, 'default', remote)
    expect(snapshot.evidence.map(item => item.run.source).sort()).toEqual([
      'background-process',
      'delegation',
      'kanban',
      'profile',
      'remote',
      'remote'
    ])
    expect(snapshot.evidence.find(item => item.run.source === 'profile')).toMatchObject({
      identityKey: 'profile:argus',
      name: 'argus',
      role: 'Permanent Hermes profile',
      run: { status: 'waiting' }
    })
    expect(snapshot.evidence.find(item => item.run.source === 'kanban')?.run.project).toBe('Hermes Agent')
    expect(snapshot.evidence.find(item => item.run.source === 'delegation')?.run).toMatchObject({
      assignment: 'Delegated work',
      latestActivity: 'read_file'
    })
    expect(snapshot.evidence.find(item => item.run.source === 'delegation')?.run.capabilities.steer).toMatchObject({ supported: false })
    expect(snapshot.evidence.find(item => item.run.source === 'background-process')?.run).toMatchObject({
      assignment: 'Background process (npm)',
      latestActivity: 'Tracked background process is running.',
      log: []
    })
    expect(JSON.stringify(snapshot.evidence)).not.toContain('PROMPT_SENTINEL')
    expect(JSON.stringify(snapshot.evidence)).not.toContain('RAW_OUTPUT_SENTINEL')
    expect(JSON.stringify(snapshot.evidence)).not.toContain('REMOTE_PROMPT_SENTINEL')
    expect(JSON.stringify(snapshot.evidence)).not.toContain('/Users/example')
    expect(snapshot.sources.filter(item => item.state === 'unavailable')).toEqual([])
    expect(aggregateFleet(snapshot.evidence).find(item => item.id === 'profile:default')?.runs.map(run => run.source).sort()).toEqual([
      'background-process',
      'remote'
    ])
    expect(request.mock.calls.map(([method]) => method)).not.toContain(expect.stringMatching(/model|provider|completion|chat/))
    expect(request.mock.calls.map(([method]) => method)).toEqual([
      'agents.list',
      'delegation.status',
      'spawn_tree.list',
      'projects.list'
    ])
    expect(request).toHaveBeenCalledWith('spawn_tree.list', { cross_session: true, limit: 20 })
    expect(remote).toHaveBeenCalledTimes(1)
    expect(rest).toHaveBeenCalledWith('/snapshot')
  })

  it('does not fabricate evidence from unavailable sources', async () => {
    const snapshot = await loadFleetEvidence(vi.fn(async () => { throw new Error('offline') }))
    expect(snapshot.evidence).toEqual([])
    expect(snapshot.sources.every(item => item.state === 'unavailable')).toBe(true)
  })

  it('loads each immutable delegation history snapshot only once while polling', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'agents.list') {return { processes: [] }}

      if (method === 'delegation.status') {return { active: [] }}

      if (method === 'projects.list') {return { projects: [] }}

      if (method === 'spawn_tree.list') {return { entries: [{ path: '/private/history-once.json', finished_at: 123 }] }}

      if (method === 'spawn_tree.load') {return { finished_at: 123, subagents: [{ subagent_id: 'history-once', status: 'finished', updated_at: 1 }] }}
      throw new Error(`unexpected ${method}`)
    })

    const first = await loadFleetEvidence(request, undefined, 'poll-cache-test')
    await loadFleetEvidence(request, undefined, 'poll-cache-test')

    expect(first.evidence[0]?.run.finishedAt).toBe(123_000)
    expect(request.mock.calls.filter(([method]) => method === 'spawn_tree.load')).toHaveLength(1)
  })

  it('does not enable Kanban controls without a real run id', async () => {
    const snapshot = await loadFleetEvidence(
      vi.fn(async method => method === 'projects.list' ? { projects: [] } : method === 'spawn_tree.list' ? { entries: [] } : method === 'delegation.status' ? { active: [] } : { processes: [] }),
      vi.fn(async () => ({ profiles: [], runs: [{ id: 'task:t1', task_id: 't1', identity_key: 'kanban-worker-0123456789abcdef', board: 'main', status: 'running' }] })),
      'default'
    )

    expect(snapshot.evidence[0]?.run.capabilities).toMatchObject({
      steer: { supported: false },
      stop: { supported: false }
    })
  })

  it('routes delegation steering only through its exact bound owner session', async () => {
    const request = vi.spyOn(host, 'request').mockImplementation(async method =>
      method === 'subagent.steer' ? { status: 'queued' } : { found: true }
    )

    vi.spyOn(host.state.activeSessionId, 'get').mockReturnValue('owner-session')

    const run: FleetRun = {
      id: 'subagent-1',
      source: 'delegation',
      status: 'active',
      assignment: 'Delegated work',
      machine: 'Local machine',
      updatedAt: 1,
      log: [],
      usage: { kind: 'unavailable' },
      artifacts: [],
      capabilities: { steer: { supported: true }, stop: { supported: true } },
      control: { subagentId: 'subagent-1', ownerSessionId: 'owner-session' }
    }

    await controlRun('steer', run, 'Use the focused test', undefined)
    await controlRun('stop', run, undefined, undefined)

    expect(request).toHaveBeenNthCalledWith(1, 'subagent.steer', {
      session_id: 'owner-session',
      subagent_id: 'subagent-1',
      text: 'Use the focused test'
    })
    expect(request).toHaveBeenNthCalledWith(2, 'subagent.interrupt', { subagent_id: 'subagent-1' })
  })

  it('fails closed before mutation when delegation steering has no exact owner binding', async () => {
    const request = vi.spyOn(host, 'request').mockResolvedValue({ status: 'queued' })
    vi.spyOn(host.state.activeSessionId, 'get').mockReturnValue('different-session')
    const run: FleetRun = {
      id: 'subagent-1', source: 'delegation', status: 'active', assignment: 'Delegated work', machine: 'Local machine', updatedAt: 1,
      log: [], usage: { kind: 'unavailable' }, artifacts: [], capabilities: { steer: { supported: true } }, control: { subagentId: 'subagent-1', ownerSessionId: 'owner-session' }
    }

    await expect(controlRun('steer', run, 'Do not send')).rejects.toThrow('exact owning session')
    await expect(controlRun('steer', { ...run, control: { subagentId: 'subagent-1' } }, 'Do not send')).rejects.toThrow('exact owning session')
    expect(request).not.toHaveBeenCalled()
  })

  it('does not retain raw Kanban assignees in fleet evidence', async () => {
    const snapshot = await loadFleetEvidence(
      vi.fn(async method => method === 'projects.list' ? { projects: [] } : method === 'spawn_tree.list' ? { entries: [] } : method === 'delegation.status' ? { active: [] } : { processes: [] }),
      vi.fn(async () => ({ profiles: [], runs: [{ id: '42', task_id: 't1', identity_key: 'kanban-worker-0123456789abcdef', title: 'Ship it', board: 'main', status: 'running', assignee: 'ASSIGNEE_SENTINEL' }] })),
      'default'
    )

    expect(JSON.stringify(snapshot.evidence)).not.toContain('ASSIGNEE_SENTINEL')
    expect(snapshot.evidence[0]).toMatchObject({ identityKey: 'kanban:kanban-worker-0123456789abcdef', name: 'Kanban builder' })
  })

  it('routes Kanban steering through the scoped backend and rejects stale targets before mutation', async () => {
    const rest = vi.fn().mockResolvedValue({ ok: true })

    const run: FleetRun = {
      id: '42',
      source: 'kanban',
      status: 'active',
      assignment: 'Ship it',
      machine: 'Local machine',
      updatedAt: 1,
      log: [],
      usage: { kind: 'unavailable' },
      artifacts: [],
      capabilities: { steer: { supported: true }, stop: { supported: true } },
      control: { board: 'main', taskId: 't1', runId: '42' }
    }

    await controlRun('steer', run, 'Check the handoff', rest)
    await controlRun('stop', run, undefined, rest)

    expect(rest).toHaveBeenNthCalledWith(1, '/runs/42/steer?board=main', {
      method: 'POST',
      body: { task_id: 't1', text: 'Check the handoff' }
    })
    expect(rest).toHaveBeenNthCalledWith(2, '/runs/42/terminate?board=main', {
      method: 'POST',
      body: { task_id: 't1', reason: 'Stopped from Live Agents' }
    })

    await expect(controlRun('steer', { ...run, control: { board: 'main', taskId: 't1' } }, 'x', rest)).rejects.toThrow('stale or incomplete')
    expect(rest).toHaveBeenCalledTimes(2)
  })

  it('opens only the exact reported delegation result session', async () => {
    const navigate = vi.spyOn(host, 'navigate').mockImplementation(() => undefined)

    const run: FleetRun = {
      id: 'subagent-1',
      source: 'delegation',
      status: 'finished',
      assignment: 'Delegated work',
      machine: 'Local machine',
      updatedAt: 1,
      log: [],
      usage: { kind: 'unavailable' },
      artifacts: [],
      capabilities: { openResult: { supported: true } },
      control: { sessionId: 'result/session' }
    }

    await controlRun('openResult', run)

    expect(navigate).toHaveBeenCalledWith('/session/result%2Fsession')
  })
})
