export type FleetStatus = 'active' | 'waiting' | 'blocked' | 'finished' | 'offline' | 'unavailable'
export type UsageKind = 'reported' | 'estimated' | 'unavailable'

export interface FleetArtifact {
  id: string
  name: string
  kind?: string
}

export interface FleetSource {
  id: string
  label: string
  state: 'available' | 'unavailable'
  reason?: string
}

export interface FleetSnapshot {
  evidence: FleetEvidence[]
  sources: FleetSource[]
}

export interface FleetRun {
  id: string
  source: string
  status: FleetStatus
  assignment: string
  project?: string
  machine: string
  startedAt?: number
  updatedAt: number
  finishedAt?: number
  latestActivity?: string
  log: string[]
  usage: { kind: UsageKind; tokens?: number; costUsd?: number }
  artifacts: FleetArtifact[]
  capabilities: Partial<Record<'pause' | 'steer' | 'stop' | 'openResult', { supported: boolean; reason?: string }>>
  control?: {
    sessionId?: string
    ownerSessionId?: string
    processId?: string
    subagentId?: string
    board?: string
    taskId?: string
    runId?: string
  }
}

export interface FleetEvidence {
  identityKey: string
  name: string
  role: string
  brief: string
  run: FleetRun
}

export interface FleetAgent {
  id: string
  name: string
  role: string
  brief: string
  status: FleetStatus
  updatedAt: number
  runs: FleetRun[]
}

export interface FleetFilters {
  search: string
  statuses: FleetStatus[]
  role: string
  project: string
  machine: string
  since?: number
}

export type RosterGroupId = 'profiles' | 'builders' | 'temporary'

export interface RosterGroup {
  id: RosterGroupId
  label: string
  agents: FleetAgent[]
}

export interface RosterTarget {
  agent?: string
  run?: string
}

export type FleetStorageSlot = 'collapsed' | 'dismissed' | 'history'

const STATUS_RANK: Record<FleetStatus, number> = {
  active: 0,
  blocked: 1,
  waiting: 2,
  finished: 3,
  offline: 4,
  unavailable: 4
}

const PRIVATE_PATH = /(?:[A-Za-z]:\\|\/(?:Users|home|private|var|tmp)\/)[^\s"'`]+/g
const SECRET = /\b(?:api[_-]?key|token|password|secret|authorization)\s*[:=]\s*\S+/gi
const OPAQUE_KANBAN_IDENTITY = /^kanban:kanban-worker-[a-f0-9]{16}$/

export function sanitizePresentation(value: unknown): string {
  return String(value ?? '')
    .replace(SECRET, '[redacted]')
    .replace(PRIVATE_PATH, '[private path]')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 500)
}

export function safeArtifactName(value: unknown): string {
  const normalized = String(value ?? '').replaceAll('\\', '/')

  return sanitizePresentation(normalized.split('/').pop() || 'artifact')
}

/** Drops legacy Kanban records that persisted a raw assignee before redaction. */
export function privacySafeFleetHistory(history: FleetEvidence[]): FleetEvidence[] {
  return history.filter(item => item.run.source !== 'kanban' || OPAQUE_KANBAN_IDENTITY.test(item.identityKey))
}

export function fleetStorageKey(profileName: string, slot: FleetStorageSlot): string {
  const profile = sanitizePresentation(profileName).toLowerCase() || 'default'

  return `profile:${profile}:${slot}`
}

export function normalizeStatus(value: unknown, finishedAt?: number): FleetStatus {
  const status = String(value ?? '').toLowerCase()

  if (['failed', 'error'].includes(status)) {return 'blocked'}

  if (finishedAt || ['done', 'finished', 'completed', 'exited', 'stopped', 'cancelled'].includes(status)) {
    return 'finished'
  }

  if (['blocked', 'paused', 'attention', 'unreachable', 'stale', 'error'].includes(status)) {
    return status === 'unreachable' || status === 'stale' ? 'unavailable' : 'blocked'
  }

  if (['ready', 'todo', 'triage', 'queued', 'waiting', 'idle', 'available', 'online'].includes(status)) {return 'waiting'}

  if (status === 'offline') {return 'offline'}

  return ['active', 'running', 'working', 'streaming', 'in_progress'].includes(status) ? 'active' : 'unavailable'
}

export function aggregateFleet(evidence: FleetEvidence[]): FleetAgent[] {
  const agents = new Map<string, FleetAgent>()

  for (const item of evidence) {
    const id = item.identityKey.trim().toLowerCase()

    if (!id || !item.run.id) {continue}

    const existing = agents.get(id) ?? {
      id,
      name: sanitizePresentation(item.name) || 'Unknown agent',
      role: sanitizePresentation(item.role) || 'Agent',
      brief: sanitizePresentation(item.brief) || 'No permanent brief is available.',
      status: item.run.status,
      updatedAt: item.run.updatedAt,
      runs: []
    }

    if (!existing.runs.some(run => run.id === item.run.id && run.source === item.run.source)) {
      existing.runs.push({
        ...item.run,
        assignment: sanitizePresentation(item.run.assignment),
        latestActivity: sanitizePresentation(item.run.latestActivity),
        log: item.run.log.map(sanitizePresentation).filter(Boolean),
        artifacts: item.run.artifacts.map(artifact => ({ ...artifact, name: safeArtifactName(artifact.name) }))
      })
    }

    existing.runs.sort((a, b) => b.updatedAt - a.updatedAt)
    existing.updatedAt = Math.max(existing.updatedAt, item.run.updatedAt)
    existing.status = existing.runs.reduce(
      (best, run) => (STATUS_RANK[run.status] < STATUS_RANK[best] ? run.status : best),
      existing.runs[0]?.status ?? 'unavailable'
    )
    agents.set(id, existing)
  }

  return [...agents.values()].sort(
    (a, b) => STATUS_RANK[a.status] - STATUS_RANK[b.status] || b.updatedAt - a.updatedAt || a.name.localeCompare(b.name)
  )
}

export function buildRosterGroups(agents: FleetAgent[]): RosterGroup[] {
  const groups: RosterGroup[] = [
    { id: 'profiles', label: 'Permanent Hermes profiles', agents: [] },
    { id: 'builders', label: 'Builder lanes', agents: [] },
    { id: 'temporary', label: 'Temporary delegated agents', agents: [] }
  ]

  for (const agent of agents) {
    const sources = new Set(agent.runs.map(run => run.source))
    const group = sources.has('profile') ? groups[0] : sources.has('kanban') ? groups[1] : groups[2]

    group!.agents.push(agent)
  }

  return groups
}

export function activeRosterCount(agents: FleetAgent[]): number {
  return agents.filter(agent => agent.status === 'active' || agent.status === 'blocked').length
}

export function parseRosterTarget(hash: string): RosterTarget {
  const route = hash.replace(/^#/, '')
  const [path, query = ''] = route.split('?', 2)

  if (path !== '/live-agents') {return {}}

  const params = new URLSearchParams(query)
  const agent = params.get('agent')?.trim()
  const run = params.get('run')?.trim()

  return {
    ...(agent ? { agent } : {}),
    ...(run ? { run } : {})
  }
}

export function filterFleet(agents: FleetAgent[], filters: FleetFilters): FleetAgent[] {
  const needle = filters.search.trim().toLowerCase()

  return agents
    .map(agent => ({
      ...agent,
      runs: agent.runs.filter(run => {
        const haystack = [agent.id, agent.name, agent.role, agent.brief, run.id, run.assignment, run.project, run.machine, run.source]
          .join(' ')
          .toLowerCase()

        return (
          (!needle || haystack.includes(needle)) &&
          (!filters.statuses.length || filters.statuses.includes(run.status)) &&
          (!filters.role || agent.role === filters.role) &&
          (!filters.project || run.project === filters.project) &&
          (!filters.machine || run.machine === filters.machine) &&
          (!filters.since || run.updatedAt >= filters.since)
        )
      })
    }))
    .filter(agent => agent.runs.length > 0)
}

function evidenceTime(item: FleetEvidence): number {
  return item.run.status === 'finished'
    ? item.run.finishedAt ?? item.run.updatedAt
    : item.run.updatedAt
}

/** Merge a fresh observation into durable plugin history. Runs are never aged out. */
export function mergeFleetHistory(history: FleetEvidence[], fresh: FleetEvidence[]): FleetEvidence[] {
  const merged = new Map<string, FleetEvidence>()

  for (const item of [...history, ...fresh]) {
    const key = `${item.identityKey.trim().toLowerCase()}\0${item.run.source}\0${item.run.id}`
    const previous = merged.get(key)

    if (!previous) {
      merged.set(key, item)
      continue
    }

    // A run id is immutable. Once an authoritative source records its
    // completion, a later polling observation that still says "active" is
    // stale and must never resurrect the run. Competing completions compare
    // their actual finish times rather than their observation times.
    if (previous.run.status === 'finished' && item.run.status !== 'finished') {continue}
    if (item.run.status === 'finished' && previous.run.status !== 'finished') {
      merged.set(key, item)
      continue
    }

    merged.set(key, evidenceTime(item) >= evidenceTime(previous) ? item : previous)
  }

  return [...merged.values()]
}
