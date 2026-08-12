import {
  Badge,
  Button,
  coarseElapsed,
  Codicon,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  EmptyState,
  ErrorState,
  type HermesPlugin,
  host,
  Loader,
  PALETTE_AREA,
  PANES_AREA,
  type PluginStorage,
  ROUTES_AREA,
  SearchField,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  SIDEBAR_NAV_AREA,
  STATUSBAR_AREAS,
  StatusDot,
  Textarea,
  Tip,
  useMutation,
  useQuery,
  useQueryClient,
  useValue
} from '@hermes/plugin-sdk'
import { useEffect, useMemo, useState } from 'react'

import { controlRun, loadFleetEvidence } from './adapters'
import { activeRosterCount, aggregateFleet, buildRosterGroups, filterFleet, type FleetAgent, type FleetEvidence, type FleetFilters, type FleetRun, type FleetSource, type FleetStatus, fleetStorageKey, mergeFleetHistory, parseRosterTarget, privacySafeFleetHistory } from './model'

const QUERY_KEY = ['live-agents', 'fleet']
const ROSTER_PANE_ID = 'live-agents:roster'
const ROSTER_FOCUS_EVENT = 'hermes:live-agents-focus-roster'
const STATUS: FleetStatus[] = ['active', 'waiting', 'blocked', 'finished', 'offline', 'unavailable']

const statusTone = (status: FleetStatus) =>
  status === 'active' ? 'good' : status === 'blocked' ? 'warn' : status === 'waiting' || status === 'finished' || status === 'offline' ? 'muted' : 'bad'

function elapsed(startedAt?: number) {
  if (!startedAt) {return 'Elapsed unavailable'}
  const { unit, value } = coarseElapsed(Date.now() - startedAt)

  return `${value}${unit[0]} elapsed`
}

function Usage({ run }: { run: FleetRun }) {
  if (run.usage.kind === 'unavailable') {return <span>Cost unavailable</span>}
  const tokens = run.usage.tokens == null ? 'tokens unavailable' : `${run.usage.tokens.toLocaleString()} tokens`
  const cost = run.usage.costUsd == null ? 'cost unavailable' : `$${run.usage.costUsd.toFixed(4)}`

  return <span>{`${run.usage.kind === 'estimated' ? 'Estimated: ' : ''}${tokens} · ${cost}`}</span>
}

function CapabilityButton({
  action,
  run,
  onRun
}: {
  action: 'pause' | 'steer' | 'stop' | 'openResult'
  run: FleetRun
  onRun: (action: 'steer' | 'stop' | 'openResult', run: FleetRun) => void
}) {
  const cap = run.capabilities[action] ?? { supported: false, reason: 'The source did not advertise this capability.' }

  const button = (
    <Button
      aria-label={`${action} ${run.assignment}`}
      disabled={!cap.supported || action === 'pause'}
      onClick={() => action !== 'pause' && onRun(action, run)}
      size="sm"
      variant={action === 'stop' ? 'destructive' : 'outline'}
    >
      {action === 'openResult' ? 'Open Result' : action[0]!.toUpperCase() + action.slice(1)}
    </Button>
  )

  return cap.supported ? button : <Tip label={cap.reason ?? 'Unsupported'}>{button}</Tip>
}

function RunView({ run, onRun }: { run: FleetRun; onRun: (action: 'steer' | 'stop' | 'openResult', run: FleetRun) => void }) {
  return (
    <section aria-label={`Run ${run.assignment}`} className="rounded-md border border-(--ui-border) bg-(--ui-bg-subtle) p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="font-medium">{run.assignment}</div>
          <div className="text-xs text-(--ui-text-tertiary)">
            {run.source} · {run.machine} · {elapsed(run.startedAt)}
          </div>
        </div>
        <Badge>{run.status}</Badge>
      </div>
      <div className="mt-2 text-sm">{run.latestActivity || 'No activity reported.'}</div>
      <div className="mt-1 text-xs text-(--ui-text-tertiary)"><Usage run={run} /></div>
      {run.log.length ? (
        <pre aria-label="Streaming work log" className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-(--ui-bg) p-2 text-xs">
          {run.log.join('\n')}
        </pre>
      ) : null}
      {run.artifacts.length ? (
        <div aria-label="Produced artifacts" className="mt-2">
          <div className="text-xs font-medium">Produced artifacts</div>
          <ul className="mt-1 text-xs text-(--ui-text-secondary)">
            {run.artifacts.map(artifact => <li key={artifact.id}>{artifact.name}</li>)}
          </ul>
        </div>
      ) : null}
      <div className="mt-3 flex flex-wrap gap-2">
        {(['pause', 'steer', 'stop', 'openResult'] as const).map(action => (
          <CapabilityButton action={action} key={action} onRun={onRun} run={run} />
        ))}
      </div>
    </section>
  )
}

function AgentCard({
  agent,
  collapsed,
  onCollapse,
  onDismiss,
  onRun
}: {
  agent: FleetAgent
  collapsed: boolean
  onCollapse: () => void
  onDismiss: () => void
  onRun: (action: 'steer' | 'stop' | 'openResult', run: FleetRun) => void
}) {
  const current = agent.runs[0]!

  return (
    <article aria-label={`${agent.name}, ${agent.status}`} className="rounded-lg border border-(--ui-border) bg-(--ui-bg-elevated) p-4">
      <header className="flex items-start justify-between gap-3">
        <button aria-expanded={!collapsed} className="min-w-0 flex-1 text-left" onClick={onCollapse}>
          <div className="flex items-center gap-2">
            <StatusDot tone={statusTone(agent.status)} />
            <h2 className="truncate font-semibold">{agent.name}</h2>
            <Badge>{agent.role}</Badge>
          </div>
          <p className="mt-1 text-sm text-(--ui-text-secondary)">{agent.brief}</p>
          <p className="mt-2 text-sm"><span className="text-(--ui-text-tertiary)">Current assignment: </span>{current.assignment}</p>
          <p className="text-xs text-(--ui-text-tertiary)">Latest: {current.latestActivity || 'No activity reported'}</p>
        </button>
        <Button aria-label={`Dismiss ${agent.name}`} onClick={onDismiss} size="sm" variant="ghost"><Codicon name="close" /></Button>
      </header>
      {!collapsed ? <div className="mt-4 grid gap-3">{agent.runs.map(run => <RunView key={`${run.source}:${run.id}`} onRun={onRun} run={run} />)}</div> : null}
    </article>
  )
}

function LiveAgentsPage(props: { rest: <T>(path: string) => Promise<T>; storage: PluginStorage }) {
  const profileName = useValue(host.state.profile) || 'default'

  return <LiveAgentsProfilePage key={profileName} profileName={profileName} {...props} />
}

function LiveAgentsProfilePage({ profileName, rest, storage }: { profileName: string; rest: <T>(path: string) => Promise<T>; storage: PluginStorage }) {
  const queryClient = useQueryClient()
  const queryKey = [...QUERY_KEY, profileName]
  const target = parseRosterTarget(window.location.hash)
  const [search, setSearch] = useState(target.agent || target.run || '')
  const [status, setStatus] = useState('all')
  const [role, setRole] = useState('all')
  const [project, setProject] = useState('all')
  const [machine, setMachine] = useState('all')
  const [timeRange, setTimeRange] = useState('all')
  const [collapsed, setCollapsed] = useState<string[]>(() => storage.get<string[]>(fleetStorageKey(profileName, 'collapsed'), []).filter(id => id !== target.agent))
  const [dismissed, setDismissed] = useState<string[]>(() => storage.get(fleetStorageKey(profileName, 'dismissed'), []))
  const [history, setHistory] = useState<FleetEvidence[]>(() => privacySafeFleetHistory(storage.get(fleetStorageKey(profileName, 'history'), [])))
  const [steerTarget, setSteerTarget] = useState<FleetRun | null>(null)
  const [steerText, setSteerText] = useState('')
  const query = useQuery({ queryKey, queryFn: () => loadFleetEvidence(host.request, rest, profileName), refetchInterval: 15_000, staleTime: 5_000 })
  useEffect(() => host.onEvent('*', () => void queryClient.invalidateQueries({ queryKey: [...QUERY_KEY, profileName] })), [profileName, queryClient])
  useEffect(() => {
    if (!query.data) {return}
    const next = mergeFleetHistory(history, query.data.evidence)

    if (next === history || JSON.stringify(next) === JSON.stringify(history)) {return}
    storage.set(fleetStorageKey(profileName, 'history'), next)
    setHistory(next)
  }, [history, profileName, query.data, storage])
  const agents = useMemo(() => aggregateFleet(history), [history])

  const options = useMemo(() => ({
    machines: [...new Set(agents.flatMap(agent => agent.runs.map(run => run.machine)))].sort(),
    projects: [...new Set(agents.flatMap(agent => agent.runs.map(run => run.project).filter(Boolean) as string[]))].sort(),
    roles: [...new Set(agents.map(agent => agent.role))].sort()
  }), [agents])

  const filters: FleetFilters = {
    machine: machine === 'all' ? '' : machine,
    project: project === 'all' ? '' : project,
    role: role === 'all' ? '' : role,
    search,
    since: timeRange === 'day' ? Date.now() - 86_400_000 : timeRange === 'week' ? Date.now() - 604_800_000 : undefined,
    statuses: status === 'all' ? [] : [status as FleetStatus]
  }

  const visible = filterFleet(agents, filters).filter(agent => !dismissed.includes(agent.id))

  const mutation = useMutation({
    mutationFn: ({ action, message, run }: { action: 'steer' | 'stop' | 'openResult'; message?: string; run: FleetRun }) => controlRun(action, run, message, rest),
    onError: error => host.notifyError(error, 'Live Agents control failed'),
    onSuccess: (_result, variables) => {
      if (variables.action === 'steer') {
        setSteerTarget(null)
        setSteerText('')
      }

      void queryClient.invalidateQueries({ queryKey })
    }
  })

  const persist = (key: 'collapsed' | 'dismissed', value: string[]) => {
    storage.set(fleetStorageKey(profileName, key), value)
    key === 'collapsed' ? setCollapsed(value) : setDismissed(value)
  }

  if (query.isLoading) {return <div className="flex h-full items-center justify-center"><Loader type="lemniscate-bloom" /></div>}

  if (query.error) {return <ErrorState description={String(query.error)} title="Live Agents unavailable"><Button onClick={() => void query.refetch()}>Retry</Button></ErrorState>}

  return (
    <main aria-label="Live Agents fleet monitor" className="flex h-full min-w-0 flex-col">
      <div className="border-b border-(--ui-border) p-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div><h1 className="text-xl font-semibold">Live Agents</h1><p className="text-sm text-(--ui-text-secondary)">Durable local fleet history. Viewing this page never invokes a model.</p></div>
          <Button onClick={() => void query.refetch()} variant="outline"><Codicon name="refresh" /> Refresh</Button>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          <SearchField aria-label="Search agents" onChange={setSearch} placeholder="Search agents, roles, projects…" value={search} />
          <Select onValueChange={setStatus} value={status}><SelectTrigger aria-label="Filter by status" className="w-44"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All statuses</SelectItem>{STATUS.map(item => <SelectItem key={item} value={item}>{item}</SelectItem>)}</SelectContent></Select>
          <Select onValueChange={setRole} value={role}><SelectTrigger aria-label="Filter by role" className="w-40"><SelectValue placeholder="Role" /></SelectTrigger><SelectContent><SelectItem value="all">All roles</SelectItem>{options.roles.map(item => <SelectItem key={item} value={item}>{item}</SelectItem>)}</SelectContent></Select>
          <Select onValueChange={setProject} value={project}><SelectTrigger aria-label="Filter by project" className="w-40"><SelectValue placeholder="Project" /></SelectTrigger><SelectContent><SelectItem value="all">All projects</SelectItem>{options.projects.map(item => <SelectItem key={item} value={item}>{item}</SelectItem>)}</SelectContent></Select>
          <Select onValueChange={setMachine} value={machine}><SelectTrigger aria-label="Filter by machine" className="w-40"><SelectValue placeholder="Machine" /></SelectTrigger><SelectContent><SelectItem value="all">All machines</SelectItem>{options.machines.map(item => <SelectItem key={item} value={item}>{item}</SelectItem>)}</SelectContent></Select>
          <Select onValueChange={setTimeRange} value={timeRange}><SelectTrigger aria-label="Filter by time range" className="w-40"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All time</SelectItem><SelectItem value="day">Last 24 hours</SelectItem><SelectItem value="week">Last 7 days</SelectItem></SelectContent></Select>
          {dismissed.length ? <Button onClick={() => persist('dismissed', [])} variant="ghost">Restore dismissed ({dismissed.length})</Button> : null}
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-4">
        <SourceHealth sources={query.data?.sources ?? []} />
        {visible.length ? <div className="grid gap-3">{visible.map(agent => <AgentCard agent={agent} collapsed={collapsed.includes(agent.id)} key={agent.id} onCollapse={() => persist('collapsed', collapsed.includes(agent.id) ? collapsed.filter(id => id !== agent.id) : [...collapsed, agent.id])} onDismiss={() => persist('dismissed', [...dismissed, agent.id])} onRun={(action, run) => action === 'steer' ? setSteerTarget(run) : mutation.mutate({ action, run })} />)}</div> : <EmptyState description="Known sources have no matching agent evidence. Unreachable sources are never represented as live." title="No agents match" />}
      </div>
      <Dialog onOpenChange={open => !open && !mutation.isPending && setSteerTarget(null)} open={Boolean(steerTarget)}>
        <DialogContent className="max-w-md">
          <form
            onSubmit={event => {
              event.preventDefault()

              if (steerTarget && steerText.trim()) {mutation.mutate({ action: 'steer', message: steerText.trim(), run: steerTarget })}
            }}
          >
            <DialogHeader>
              <DialogTitle>Steer this run</DialogTitle>
              <DialogDescription>Send an instruction to the exact active run. The authoritative source decides whether it is still valid.</DialogDescription>
            </DialogHeader>
            <Textarea
              aria-label="Steering instruction"
              autoFocus
              className="mt-4 min-h-28"
              onChange={event => setSteerText(event.target.value)}
              placeholder="What should this worker change or check?"
              value={steerText}
            />
            <DialogFooter className="mt-4">
              <Button disabled={mutation.isPending} onClick={() => setSteerTarget(null)} type="button" variant="ghost">Cancel</Button>
              <Button disabled={mutation.isPending || !steerText.trim()} type="submit">{mutation.isPending ? 'Sending…' : 'Send instruction'}</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </main>
  )
}

function SourceHealth({ sources }: { sources: FleetSource[] }) {
  const unavailable = sources.filter(item => item.state === 'unavailable')

  if (!unavailable.length) {return null}

  return (
    <aside aria-label="Unavailable agent sources" className="mb-3 rounded-md border border-(--ui-border) p-3">
      <div className="text-sm font-medium">Some sources are unavailable</div>
      <ul className="mt-1 text-xs text-(--ui-text-tertiary)">
        {unavailable.map(item => <li key={item.id}>{item.label}: {item.reason || 'No public source contract is registered.'}</li>)}
      </ul>
    </aside>
  )
}

function useFleetRoster(rest: <T>(path: string) => Promise<T>) {
  const profileName = useValue(host.state.profile) || 'default'

  const query = useQuery({
    queryKey: [...QUERY_KEY, profileName],
    queryFn: () => loadFleetEvidence(host.request, rest, profileName),
    refetchInterval: 15_000,
    staleTime: 5_000
  })

  const agents = useMemo(() => aggregateFleet(query.data?.evidence ?? []), [query.data?.evidence])

  return { agents, query }
}

function openRoster() {
  window.dispatchEvent(new CustomEvent('hermes:pane-toggle-reveal', { detail: { id: ROSTER_PANE_ID, mode: 'open' } }))
  window.dispatchEvent(new CustomEvent(ROSTER_FOCUS_EVENT))
}

function AgentsChip({ rest }: { rest: <T>(path: string) => Promise<T> }) {
  const { agents } = useFleetRoster(rest)
  const count = activeRosterCount(agents)

  return (
    <button
      aria-label={`Open Agent Roster, ${count} active or needing attention`}
      className="flex items-center gap-1.5 px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) focus-visible:outline focus-visible:outline-2 focus-visible:outline-(--ui-accent)"
      onClick={openRoster}
      onKeyDown={event => {
        if (event.key !== 'Enter' && event.key !== ' ') {return}
        event.preventDefault()
        openRoster()
      }}
      type="button"
    >
      <Codicon name="organization" />
      <span>Agents</span>
      <Badge>{count}</Badge>
    </button>
  )
}

function rosterStatus(status: FleetStatus) {
  if (status === 'active') {return 'Working'}

  if (status === 'blocked') {return 'Needs Attention'}

  if (status === 'waiting') {return 'Waiting'}

  if (status === 'finished') {return 'Completed'}

  return status === 'offline' ? 'Offline' : 'Unavailable'
}

function AgentRoster({ rest }: { rest: <T>(path: string) => Promise<T> }) {
  const { agents, query } = useFleetRoster(rest)
  const groups = useMemo(() => buildRosterGroups(agents), [agents])

  useEffect(() => {
    const focus = () => document.querySelector<HTMLElement>('[data-live-agents-roster]')?.focus()
    window.addEventListener(ROSTER_FOCUS_EVENT, focus)

    return () => window.removeEventListener(ROSTER_FOCUS_EVENT, focus)
  }, [])

  if (query.isLoading) {return <div className="flex h-full items-center justify-center"><Loader type="lemniscate-bloom" /></div>}

  if (query.error) {return <ErrorState description={String(query.error)} title="Agent Roster unavailable"><Button onClick={() => void query.refetch()}>Retry</Button></ErrorState>}

  return (
    <aside aria-label="Agent Roster" className="flex h-full min-w-0 flex-col" data-live-agents-roster tabIndex={-1}>
      <header className="flex items-center justify-between border-b border-(--ui-border) p-3">
        <div>
          <h2 className="font-semibold">Agent Roster</h2>
          <p className="text-xs text-(--ui-text-tertiary)">Read-only live state</p>
        </div>
        <Button aria-label="Refresh Agent Roster" onClick={() => void query.refetch()} size="sm" variant="ghost"><Codicon name="refresh" /></Button>
      </header>
      <div className="min-h-0 flex-1 overflow-auto p-3">
        <SourceHealth sources={query.data?.sources ?? []} />
        {groups.map(group => (
          <section aria-labelledby={`roster-${group.id}`} className="mb-4" key={group.id}>
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-(--ui-text-tertiary)" id={`roster-${group.id}`}>{group.label}</h3>
            {group.agents.length ? (
              <div className="grid gap-2">
                {group.agents.map(agent => {
                  const current = agent.runs[0]!

                  return (
                    <button
                      aria-label={`Open ${agent.name} details, ${rosterStatus(agent.status)}`}
                      className="min-w-0 rounded-md border border-(--ui-border) p-3 text-left hover:bg-(--ui-bg-subtle) focus-visible:outline focus-visible:outline-2 focus-visible:outline-(--ui-accent)"
                      key={agent.id}
                      onClick={() => host.navigate(`/live-agents?agent=${encodeURIComponent(agent.id)}&run=${encodeURIComponent(current.id)}`)}
                      type="button"
                    >
                      <div className="flex items-center gap-2">
                        <StatusDot tone={statusTone(agent.status)} />
                        <span className="min-w-0 flex-1 truncate font-medium">{agent.name}</span>
                        <Badge>{rosterStatus(agent.status)}</Badge>
                      </div>
                      <div className="mt-1 truncate text-xs text-(--ui-text-secondary)">{agent.role}</div>
                      <div className="mt-1 truncate text-xs" title={current.assignment}>{current.assignment}</div>
                      <div className="text-xs text-(--ui-text-tertiary)">{elapsed(current.startedAt)}</div>
                    </button>
                  )
                })}
              </div>
            ) : <p className="text-xs text-(--ui-text-tertiary)">No entries reported.</p>}
          </section>
        ))}
      </div>
    </aside>
  )
}

const plugin: HermesPlugin = {
  id: 'live-agents',
  name: 'Live Agents',
  defaultEnabled: true,
  register(ctx) {
    ctx.registerMany([
      { id: 'page', area: ROUTES_AREA, data: { path: '/live-agents' }, render: () => <LiveAgentsPage rest={ctx.rest} storage={ctx.storage} /> },
      { id: 'nav', area: SIDEBAR_NAV_AREA, order: 35, data: { path: '/live-agents', label: 'Live Agents', codicon: 'organization' } },
      {
        id: 'roster',
        area: PANES_AREA,
        title: 'Agent Roster',
        data: { placement: 'right', collapsible: true, dock: { pane: 'workspace', pos: 'right' }, width: 'clamp(18rem, 30vw, 24rem)' },
        render: () => <AgentRoster rest={ctx.rest} />
      },
      { id: 'chip', area: STATUSBAR_AREAS.right, order: 120, render: () => <AgentsChip rest={ctx.rest} /> },
      { id: 'open-roster', area: PALETTE_AREA, data: { id: 'live-agents.open-roster', label: 'Open Agent Roster', keywords: ['agents', 'roster', 'workers'], run: openRoster } }
    ])
  }
}

export default plugin
