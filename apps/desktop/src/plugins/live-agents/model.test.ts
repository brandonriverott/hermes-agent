import { describe, expect, it } from 'vitest'

import { activeRosterCount, aggregateFleet, buildRosterGroups, filterFleet, type FleetEvidence, fleetStorageKey, mergeFleetHistory, normalizeStatus, parseRosterTarget, privacySafeFleetHistory, safeArtifactName, sanitizePresentation } from './model'

const run = (id: string, source: string, status: 'active' | 'blocked' | 'finished', updatedAt: number) => ({
  id, source, status, assignment: `${source} assignment`, machine: 'local', updatedAt, log: [], usage: { kind: 'unavailable' as const },
  artifacts: [], capabilities: {}
})

const evidence = (identityKey: string, id: string, source: string, status: 'active' | 'blocked' | 'finished', updatedAt: number): FleetEvidence => ({
  identityKey, name: 'Hermes worker', role: 'Builder', brief: 'Builds registered project work.', run: run(id, source, status, updatedAt)
})

describe('live agents fleet model', () => {
  it('groups permanent profiles, builder lanes, and temporary workers in the approved order', () => {
    const agents = aggregateFleet([
      evidence('profile:argus', 'profile:argus', 'profile', 'finished', 30),
      evidence('profile:builder', 'kanban-1', 'kanban', 'active', 20),
      evidence('delegation:review', 'delegation-1', 'delegation', 'active', 10)
    ])

    expect(buildRosterGroups(agents).map(group => [group.id, group.agents.map(agent => agent.id)])).toEqual([
      ['profiles', ['profile:argus']],
      ['builders', ['profile:builder']],
      ['temporary', ['delegation:review']]
    ])
  })

  it('counts only working and needs-attention agents in the status chip', () => {
    const agents = aggregateFleet([
      evidence('working', '1', 'delegation', 'active', 40),
      evidence('attention', '2', 'kanban', 'blocked', 30),
      evidence('completed', '3', 'kanban', 'finished', 20),
      { ...evidence('offline', '4', 'profile', 'finished', 10), run: { ...run('4', 'profile', 'finished', 10), status: 'unavailable' } }
    ])

    expect(activeRosterCount(agents)).toBe(2)
  })

  it('distinguishes waiting work from offline profiles', () => {
    expect(normalizeStatus('ready')).toBe('waiting')
    expect(normalizeStatus('idle')).toBe('waiting')
    expect(normalizeStatus('offline')).toBe('offline')
    expect(normalizeStatus('failed', 123)).toBe('blocked')
  })

  it('parses a roster drill-down target without accepting unrelated routes', () => {
    expect(parseRosterTarget('#/live-agents?agent=profile%3Aargus&run=42')).toEqual({ agent: 'profile:argus', run: '42' })
    expect(parseRosterTarget('#/settings?agent=profile%3Aargus')).toEqual({})
  })

  it('filters by stable agent and run identities for roster drill-down', () => {
    const agents = aggregateFleet([evidence('profile:builder', 'run-42', 'kanban', 'active', 20)])

    expect(filterFleet(agents, { search: 'profile:builder', statuses: [], role: '', project: '', machine: '' })).toHaveLength(1)
    expect(filterFleet(agents, { search: 'run-42', statuses: [], role: '', project: '', machine: '' })).toHaveLength(1)
  })

  it('deduplicates overlapping sources into one permanent identity with distinct runs', () => {
    const agents = aggregateFleet([
      evidence('profile:builder', 'delegation-1', 'delegation', 'active', 20),
      evidence('profile:builder', 'kanban-1', 'kanban', 'finished', 10),
      evidence('profile:builder', 'process-1', 'background-process', 'active', 30),
      evidence('profile:builder', 'project-1', 'project', 'finished', 5),
      evidence('profile:builder', 'remote-1', 'remote', 'blocked', 15)
    ])

    expect(agents).toHaveLength(1)
    expect(agents[0]?.runs.map(item => item.id)).toEqual(['process-1', 'delegation-1', 'remote-1', 'kanban-1', 'project-1'])
  })

  it('sorts active, attention, then finished in reverse chronology', () => {
    const agents = aggregateFleet([
      evidence('finished-old', '1', 'session', 'finished', 1),
      evidence('blocked', '2', 'session', 'blocked', 2),
      evidence('finished-new', '3', 'session', 'finished', 3),
      evidence('active', '4', 'session', 'active', 0)
    ])

    expect(agents.map(item => item.id)).toEqual(['active', 'blocked', 'finished-new', 'finished-old'])
  })

  it('filters across required dimensions without mutating history', () => {
    const agents = aggregateFleet([evidence('a', '1', 'kanban', 'active', 20), evidence('b', '2', 'delegation', 'finished', 10)])
    expect(filterFleet(agents, { search: 'kanban', statuses: ['active'], role: 'Builder', project: '', machine: 'local', since: 15 })).toHaveLength(1)
    expect(agents).toHaveLength(2)
  })

  it('redacts secrets and private paths at the presentation boundary', () => {
    expect(sanitizePresentation('token=sentinel /Users/brandon/private.txt password: nope')).toBe('[redacted] [private path] [redacted]')
    expect(safeArtifactName('C:\\secret\\report.txt')).toBe('report.txt')
  })

  it('drops legacy Kanban history that could contain an assignee name', () => {
    const legacy = evidence('profile:ASSIGNEE_SENTINEL', '1', 'kanban', 'finished', 1)
    const opaque = evidence('kanban:kanban-worker-0123456789abcdef', '2', 'kanban', 'finished', 1)

    expect(privacySafeFleetHistory([legacy, opaque])).toEqual([opaque])
  })

  it('represents remote online, unreachable, and stale states without treating them as live', () => {
    expect(normalizeStatus('online')).toBe('waiting')
    expect(normalizeStatus('unreachable')).toBe('unavailable')
    expect(normalizeStatus('stale')).toBe('unavailable')
  })

  it('retains a completion by its actual finish time when a later poll still reports it active', () => {
    const finished = {
      ...evidence('a', '1', 'delegation', 'finished', 900),
      run: { ...run('1', 'delegation', 'finished', 900), finishedAt: 200 }
    }
    const staleActive = evidence('a', '1', 'delegation', 'active', 1_000)
    const earlierCompletion = {
      ...evidence('a', '2', 'kanban', 'finished', 950),
      run: { ...run('2', 'kanban', 'finished', 950), finishedAt: 100 }
    }
    const laterCompletion = {
      ...evidence('a', '2', 'kanban', 'finished', 300),
      run: { ...run('2', 'kanban', 'finished', 300), finishedAt: 250 }
    }
    const merged = mergeFleetHistory([finished, earlierCompletion], [staleActive, laterCompletion])

    expect(merged).toHaveLength(2)
    expect(merged.find(item => item.run.id === '1')?.run.status).toBe('finished')
    expect(merged.find(item => item.run.id === '2')?.run.finishedAt).toBe(250)
  })

  it('scopes durable presentation state to the active profile', () => {
    expect(fleetStorageKey('default', 'history')).toBe('profile:default:history')
    expect(fleetStorageKey('Tea Time', 'history')).toBe('profile:tea time:history')
    expect(fleetStorageKey('', 'history')).toBe('profile:default:history')
  })
})
