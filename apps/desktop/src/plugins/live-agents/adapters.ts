import { host } from '@hermes/plugin-sdk'

import type { FleetEvidence, FleetRun, FleetSnapshot, FleetSource } from './model'
import { normalizeStatus, safeArtifactName, sanitizePresentation } from './model'

type AgentProcess = {
  session_id?: string
  command?: string
  status?: string
  uptime?: number
  started_at?: string
  output_tail?: string
  exit_code?: number
}

type Request = (method: string, params?: Record<string, unknown>) => Promise<unknown>
type SnapshotRestResult = { profiles?: HermesProfile[]; runs?: KanbanRun[] }
type RestOptions = { method?: string; body?: unknown }
type Rest = (path: string, options?: RestOptions) => Promise<unknown>

const unsupported = (reason: string) => ({ supported: false, reason })

const SAFE_COMMAND_NAMES = /^(?:bash|bun|cargo|docker|git|go|make|node|npm|pnpm|python\d*|pytest|sh|uv|yarn|zsh)$/i

function backgroundCommandLabel(value: unknown): string {
  const firstToken = String(value ?? '').trim().split(/\s+/, 1)[0]?.replace(/^['"]|['"]$/g, '')
  const candidate = safeArtifactName(firstToken).replace(/\.exe$/i, '')

  return SAFE_COMMAND_NAMES.test(candidate) ? candidate : 'command'
}

function processEvidence(proc: AgentProcess, now: number, profileName: string): FleetEvidence | null {
  if (!proc.session_id) {return null}
  const status = normalizeStatus(proc.status)
  const command = backgroundCommandLabel(proc.command)
  const assignment = `Background process (${command})`
  const reportedStart = proc.started_at ? Date.parse(proc.started_at) : Number.NaN

  const startedAt = Number.isFinite(reportedStart)
    ? reportedStart
    : typeof proc.uptime === 'number'
      ? now - proc.uptime * 1000
      : undefined

  const run: FleetRun = {
    id: proc.session_id,
    source: 'background-process',
    status,
    assignment,
    machine: 'Local machine',
    startedAt,
    updatedAt: status === 'finished' ? (startedAt ?? now) : now,
    finishedAt: status === 'finished' ? (startedAt ?? now) : undefined,
    latestActivity: status === 'finished'
      ? 'Tracked background process finished.'
      : status === 'active'
        ? 'Tracked background process is running.'
        : 'Tracked background process state is unavailable.',
    // agents.list exposes a raw command/output preview. It is intentionally
    // not persisted or rendered because it may contain prompt bodies or tool
    // arguments that generic redaction cannot identify safely.
    log: [],
    usage: { kind: 'unavailable' },
    artifacts: [],
    capabilities: {
      pause: unsupported('This process runtime does not expose pause.'),
      steer: unsupported('Background terminal processes cannot be steered.'),
      stop: unsupported('The fleet listing does not expose the owning live session required for a safe targeted stop.'),
      openResult: unsupported('This source did not report a previewable result.')
    },
    control: { processId: proc.session_id }
  }

  return {
    identityKey: profileName ? `profile:${profileName.toLowerCase()}` : `process:${proc.session_id}`,
    name: profileName || assignment,
    role: profileName ? 'Hermes profile worker' : 'Background worker',
    brief: profileName
      ? 'Runs tracked work for a permanent Hermes profile.'
      : 'Runs tracked terminal work and reports authoritative process output.',
    run
  }
}

const source = (id: string, label: string, state: FleetSource['state'], reason?: string): FleetSource => ({
  id,
  label,
  state,
  reason: reason ? sanitizePresentation(reason) : undefined
})

async function optional<T>(
  request: Request,
  id: string,
  label: string,
  method: string,
  params: Record<string, unknown> = {}
): Promise<{ data?: T; source: FleetSource }> {
  try {
    return { data: await request(method, params) as T, source: source(id, label, 'available') }
  } catch (error) {
    return {
      source: source(id, label, 'unavailable', error instanceof Error ? error.message : 'The source could not be reached.')
    }
  }
}

type Delegation = {
  subagent_id?: string
  id?: string
  parent_id?: string
  owner_session_id?: string
  child_session_id?: string
  goal?: string
  status?: string
  started_at?: number
  updated_at?: number
  finished_at?: number
  duration_seconds?: number
  model?: string
  current_tool?: string
  summary?: string
  cost_usd?: number
  input_tokens?: number
  output_tokens?: number
  files_written?: string[]
  output_tail?: Array<{ preview?: string; tool?: string }>
}

function delegationEvidence(item: Delegation, now: number, snapshotFinishedAt?: number): FleetEvidence | null {
  const id = item.subagent_id || item.id

  if (!id) {return null}
  const status = normalizeStatus(item.status)
  const startedAt = item.started_at ? item.started_at * 1000 : item.duration_seconds ? now - item.duration_seconds * 1000 : undefined
  const finishedAt = status === 'finished'
    ? (snapshotFinishedAt ?? item.finished_at ?? item.updated_at ?? now / 1000) * 1000
    : undefined

  const tail = (item.output_tail ?? []).map(entry => sanitizePresentation(entry.tool)).filter(Boolean)

  const artifacts = (item.files_written ?? []).map((name, index) => ({
    id: `${id}:file:${index}`,
    name: safeArtifactName(name),
    kind: 'file'
  }))

  const sessionId = item.child_session_id
  const ownerSessionId = item.owner_session_id?.trim() || undefined

  return {
    identityKey: `delegation:${id}`,
    name: sanitizePresentation(item.model) || 'Delegated agent',
    role: 'Delegated worker',
    brief: 'Handles a bounded task delegated by a Hermes conversation.',
    run: {
      id,
      source: 'delegation',
      status,
      assignment: 'Delegated work',
      machine: 'Local machine',
      startedAt,
      updatedAt: finishedAt ?? (item.updated_at ? item.updated_at * 1000 : now),
      finishedAt,
      latestActivity: sanitizePresentation(item.current_tool || tail.at(-1) || (status === 'finished' ? 'Delegated work finished.' : 'Delegated work is running.')),
      log: tail,
      usage: {
        kind: item.cost_usd != null || item.input_tokens != null || item.output_tokens != null ? 'reported' : 'unavailable',
        tokens: (item.input_tokens ?? 0) + (item.output_tokens ?? 0) || undefined,
        costUsd: item.cost_usd
      },
      artifacts,
      capabilities: {
        pause: unsupported('The runtime only exposes a global delegation spawn pause, not per-run pause.'),
        steer: status === 'active' && ownerSessionId
          ? { supported: true }
          : unsupported(status === 'active'
            ? 'The exact owning session is not bound by this delegation observation.'
            : 'Only a running delegation can be steered.'),
        stop: status === 'active' ? { supported: true } : unsupported('Only a running delegation can be stopped.'),
        openResult: sessionId
          ? { supported: true }
          : unsupported('This delegation did not report a child session result.')
      },
      control: { sessionId, ownerSessionId, subagentId: id }
    }
  }
}

type KanbanRun = {
  id?: string
  task_id?: string
  title?: string
  identity_key?: string
  board?: string
  status?: string
  started_at?: number
  updated_at?: number
  ended_at?: number
  latest_activity?: string
  log?: string[]
  artifacts?: Array<{ id?: string; name?: string; kind?: string }>
}

type HermesProfile = {
  name?: string
  description?: string
  gateway_running?: boolean
}

type HermesProject = {
  name?: string
  board_slug?: string
}

type RemoteAgent = {
  id?: string
  identity_key?: string
  name?: string
  role?: string
  brief?: string
  assignment_summary?: string
  machine?: string
  status?: string
  updated_at?: number
  started_at?: number
  latest_activity_summary?: string
}

type RemoteLoader = () => Promise<{ agents?: RemoteAgent[] }>
const delegationHistoryCache = new Map<string, FleetEvidence[]>()

function profileEvidence(item: HermesProfile, now: number): FleetEvidence | null {
  const name = sanitizePresentation(item.name)

  if (!name) {return null}

  return {
    identityKey: `profile:${name.toLowerCase()}`,
    name,
    role: 'Permanent Hermes profile',
    brief: sanitizePresentation(item.description) || 'A permanent Hermes profile available for routed work.',
    run: {
      id: `profile:${name.toLowerCase()}`,
      source: 'profile',
      status: item.gateway_running ? 'waiting' : 'offline',
      assignment: item.gateway_running ? 'Available for routed work' : 'No active gateway work',
      machine: 'Local machine',
      updatedAt: now,
      latestActivity: item.gateway_running ? 'Profile gateway is available and idle.' : 'Profile gateway is offline.',
      log: [],
      usage: { kind: 'unavailable' },
      artifacts: [],
      capabilities: {}
    }
  }
}

const KANBAN_WORKER_IDENTITY = /^kanban-worker-[a-f0-9]{16}$/
const CROSS_SESSION_HISTORY_LIMIT = 20
const CROSS_SESSION_SUBAGENT_LIMIT = 20

function kanbanEvidence(item: KanbanRun, now: number, projectName?: string): FleetEvidence | null {
  const workerIdentity = String(item.identity_key ?? '').toLowerCase()

  if (!item.id || !item.task_id || !KANBAN_WORKER_IDENTITY.test(workerIdentity)) {return null}
  const status = normalizeStatus(item.status, item.ended_at)
  const hasLiveRunTarget = /^\d+$/.test(item.id)

  return {
    identityKey: `kanban:${workerIdentity}`,
    name: 'Kanban builder',
    role: 'Kanban builder',
    brief: 'Builds assigned work from registered Hermes Kanban boards.',
    run: {
      id: item.id,
      source: 'kanban',
      status,
      assignment: sanitizePresentation(item.title) || 'Kanban task',
      project: sanitizePresentation(projectName || item.board) || undefined,
      machine: 'Local machine',
      startedAt: item.started_at,
      updatedAt: item.updated_at ?? item.ended_at ?? item.started_at ?? now,
      finishedAt: status === 'finished' ? item.ended_at ?? item.updated_at ?? now : undefined,
      latestActivity: sanitizePresentation(item.latest_activity),
      log: (item.log ?? []).map(sanitizePresentation).filter(Boolean),
      usage: { kind: 'unavailable' },
      artifacts: (item.artifacts ?? []).filter(artifact => artifact.id && artifact.name).map(artifact => ({
        id: artifact.id!,
        name: artifact.name!,
        kind: artifact.kind
      })),
      capabilities: {
        pause: unsupported('Kanban does not expose a per-run pause operation.'),
        steer: status === 'active' && hasLiveRunTarget ? { supported: true } : unsupported(hasLiveRunTarget ? 'Only a running Kanban worker can be steered.' : 'No live Kanban run target was reported.'),
        stop: status === 'active' && hasLiveRunTarget ? { supported: true } : unsupported(hasLiveRunTarget ? 'Only a running Kanban worker can be stopped.' : 'No live Kanban run target was reported.'),
        openResult: unsupported('A safe Desktop artifact preview target was not reported.')
      },
      control: { board: item.board, taskId: item.task_id, runId: item.id }
    }
  }
}

function remoteEvidence(item: RemoteAgent, now: number): FleetEvidence | null {
  const id = sanitizePresentation(item.id)

  if (!id) {return null}
  const name = sanitizePresentation(item.name) || 'Remote agent'
  const machine = sanitizePresentation(item.machine) || 'Remote machine'
  const status = normalizeStatus(item.status)

  const reportedIdentity = sanitizePresentation(item.identity_key).toLowerCase()

  const identityKey = /^(?:profile|remote):[a-z0-9][a-z0-9 _-]{0,127}$/.test(reportedIdentity)
    ? reportedIdentity
    : `remote:${id.toLowerCase()}`

  return {
    identityKey,
    name,
    role: sanitizePresentation(item.role) || 'Remote worker',
    brief: sanitizePresentation(item.brief) || 'Reports work through a configured remote Hermes source.',
    run: {
      id: `remote:${id.toLowerCase()}`,
      source: 'remote',
      status,
      assignment: sanitizePresentation(item.assignment_summary) || (status === 'waiting' ? 'Available for routed work' : 'Remote status unavailable'),
      machine,
      startedAt: item.started_at,
      updatedAt: item.updated_at ?? now,
      latestActivity: sanitizePresentation(item.latest_activity_summary) || (status === 'unavailable' ? 'The remote agent cannot be verified.' : 'Remote source is connected.'),
      log: [],
      usage: { kind: 'unavailable' },
      artifacts: [],
      capabilities: {
        pause: unsupported('The remote connector does not advertise pause.'),
        steer: unsupported('The remote connector does not advertise steering.'),
        stop: unsupported('The remote connector does not advertise stop.'),
        openResult: unsupported('The remote connector did not report a previewable result.')
      }
    }
  }
}

export async function loadFleetEvidence(
  request: Request = host.request,
  rest?: Rest,
  profileName: string = host.state.profile.get(),
  remoteLoader?: RemoteLoader
): Promise<FleetSnapshot> {
  const now = Date.now()
  const evidence: FleetEvidence[] = []

  const [processes, delegations, historyIndex, projects, remote, kanban] = await Promise.all([
    optional<{ processes?: AgentProcess[] }>(request, 'processes', 'Background processes', 'agents.list'),
    optional<{ active?: Delegation[] }>(request, 'delegations', 'Delegations', 'delegation.status'),
    optional<{ entries?: Array<{ path?: string; finished_at?: number }> }>(
      request,
      'delegation-history',
      'Delegation history',
      'spawn_tree.list',
      { cross_session: true, limit: CROSS_SESSION_HISTORY_LIMIT }
    ),
    optional<{ projects?: HermesProject[] }>(request, 'projects', 'Profiles and projects', 'projects.list'),
    remoteLoader
      ? remoteLoader()
          .then(data => ({ data, source: source('remote', 'Configured remote machines', 'available') }))
          .catch(error => ({
            data: undefined,
            source: source('remote', 'Configured remote machines', 'unavailable', error instanceof Error ? error.message : 'The source could not be reached.')
          }))
      : Promise.resolve({
          data: undefined,
          source: source('remote', 'Configured remote machines', 'unavailable', 'No safe registered remote agent source is configured.')
        }),
    rest
      ? rest('/snapshot')
          .then(data => ({ data: data as SnapshotRestResult, source: source('kanban', 'Kanban and builders', 'available') }))
          .catch(error => ({
            data: undefined,
            source: source('kanban', 'Kanban and builders', 'unavailable', error instanceof Error ? error.message : 'The source could not be reached.')
          }))
      : Promise.resolve({ data: undefined, source: source('kanban', 'Kanban and builders', 'unavailable', 'The bundled Kanban adapter is not registered.') })
  ])

  for (const proc of processes.data?.processes ?? []) {
    const item = processEvidence(proc, now, sanitizePresentation(profileName))

    if (item) {evidence.push(item)}
  }

  for (const delegation of delegations.data?.active ?? []) {
    const item = delegationEvidence(delegation, now)

    if (item) {evidence.push(item)}
  }

  const history = await Promise.all((historyIndex.data?.entries ?? []).slice(0, CROSS_SESSION_HISTORY_LIMIT).map(async entry => {
    if (!entry.path) {return []}
    const cacheKey = `${profileName}\0${entry.path}\0${entry.finished_at ?? ''}`
    const cached = delegationHistoryCache.get(cacheKey)

    if (cached) {return cached}

    try {
      const snapshot = await request('spawn_tree.load', { path: entry.path }) as {
        finished_at?: number
        session_id?: string
        subagents?: Delegation[]
      }

      const loaded = (snapshot.subagents ?? []).slice(0, CROSS_SESSION_SUBAGENT_LIMIT).map(item => delegationEvidence({
        ...item,
        status: item.status || 'finished'
      }, now, snapshot.finished_at)).filter((item): item is FleetEvidence => item != null)

      delegationHistoryCache.set(cacheKey, loaded)

      return loaded
    } catch {
      return []
    }
  }))

  evidence.push(...history.flat())

  const projectNames = new Map(
    (projects.data?.projects ?? [])
      .filter(project => project.board_slug && project.name)
      .map(project => [project.board_slug!, sanitizePresentation(project.name)] as const)
  )

  for (const run of kanban.data?.runs ?? []) {
    const item = kanbanEvidence(run, now, run.board ? projectNames.get(run.board) : undefined)

    if (item) {evidence.push(item)}
  }

  for (const profile of kanban.data?.profiles ?? []) {
    const item = profileEvidence(profile, now)

    if (item) {evidence.push(item)}
  }

  for (const agent of remote.data?.agents ?? []) {
    const item = remoteEvidence(agent, now)

    if (item) {evidence.push(item)}
  }

  return {
    evidence,
    sources: [
      processes.source,
      delegations.source,
      historyIndex.source,
      projects.source,
      kanban.source,
      remote.source
    ]
  }
}

export async function controlRun(
  action: 'steer' | 'stop' | 'openResult',
  run: FleetRun,
  message?: string,
  rest?: Rest
): Promise<void> {
  const capability = run.capabilities[action]

  if (!capability?.supported) {throw new Error(capability?.reason || `${action} is unsupported`)}

  if (action === 'openResult') {
    if (!run.control?.sessionId) {throw new Error('The result target is stale or incomplete.')}
    host.navigate(`/session/${encodeURIComponent(run.control.sessionId)}`)

    return
  }

  if (action === 'stop' && run.source === 'delegation') {
    const subagentId = run.control?.subagentId

    if (!subagentId) {throw new Error('The delegation target is stale or incomplete.')}
    const result = await host.request<{ found?: boolean }>('subagent.interrupt', { subagent_id: subagentId })

    if (!result.found) {throw new Error('The delegation is no longer running.')}

    return
  }

  if (action === 'steer' && run.source === 'delegation') {
    const subagentId = run.control?.subagentId
    const ownerSessionId = run.control?.ownerSessionId
    const activeSessionId = host.state.activeSessionId.get()

    if (!subagentId || !ownerSessionId || ownerSessionId !== activeSessionId || !message?.trim()) {
      throw new Error('The exact owning session is not bound to this live delegation target.')
    }

    const result = await host.request<{ status?: string }>('subagent.steer', {
      session_id: ownerSessionId,
      subagent_id: subagentId,
      text: message.trim()
    })

    if (result.status !== 'queued') {throw new Error('The delegation rejected the steering message because it is stale or belongs to another session.')}

    return
  }

  if (action === 'steer' && run.source === 'kanban') {
    const { board, runId, taskId } = run.control ?? {}

    if (!rest || !board || !runId || !taskId || !message?.trim()) {throw new Error('The Kanban target is stale or incomplete.')}

    const result = await rest(`/runs/${encodeURIComponent(runId)}/steer?board=${encodeURIComponent(board)}`, {
      method: 'POST',
      body: { task_id: taskId, text: message.trim() }
    }) as { ok?: boolean }

    if (!result.ok) {throw new Error('The Kanban worker rejected the steering message.')}

    return
  }

  if (action === 'stop' && run.source === 'kanban') {
    const { board, runId, taskId } = run.control ?? {}

    if (!rest || !board || !runId || !taskId) {throw new Error('The Kanban target is stale or incomplete.')}

    const result = await rest(`/runs/${encodeURIComponent(runId)}/terminate?board=${encodeURIComponent(board)}`, {
      method: 'POST',
      body: { task_id: taskId, reason: 'Stopped from Live Agents' }
    }) as { ok?: boolean }

    if (!result.ok) {throw new Error('The Kanban worker rejected the stop request.')}

    return
  }

  if (action === 'stop') {
    if (!run.control?.processId || !run.control.sessionId) {throw new Error('The target process is stale or incomplete.')}
    await host.request('process.kill', { process_id: run.control.processId, session_id: run.control.sessionId })

    return
  }

  if (!run.control?.sessionId || !message?.trim()) {throw new Error('A live target and steering message are required.')}
  await host.request('session.steer', { session_id: run.control.sessionId, text: message.trim() })
}
