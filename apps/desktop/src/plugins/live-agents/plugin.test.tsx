import { host } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import plugin from './plugin'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.location.hash = ''
})

function publicSources(overrides: Record<string, unknown> = {}) {
  return vi.spyOn(host, 'request').mockImplementation(async method => {
    if (method in overrides) {return overrides[method]}

    if (method === 'agents.list') {return { processes: [] }}

    if (method === 'delegation.status') {return { active: [] }}

    if (method === 'spawn_tree.list') {return { entries: [] }}

    if (method === 'projects.list') {return { projects: [] }}

    if (method === 'agents.remote.list') {return { agents: [] }}
    throw new Error(`Unexpected request: ${method}`)
  })
}

describe('Live Agents desktop contributions', () => {
  it('registers the detailed page, Agents chip, dockable roster, and command-palette action', () => {
    const registerMany = vi.fn()

    plugin.register({
      registerMany,
      rest: vi.fn(),
      storage: { get: vi.fn((_key, fallback) => fallback), remove: vi.fn(), set: vi.fn() }
    } as never)

    const contributions = registerMany.mock.calls[0]?.[0] ?? []

    expect(contributions.map((item: { area: string; id: string }) => [item.id, item.area])).toEqual([
      ['page', 'routes'],
      ['nav', 'sidebar.nav'],
      ['roster', 'panes'],
      ['chip', 'statusBar.right'],
      ['open-roster', 'palette']
    ])
    expect(contributions.find((item: { id: string }) => item.id === 'roster')).toMatchObject({
      title: 'Agent Roster',
      data: { placement: 'right', collapsible: true }
    })
  })

  it('opens the roster from keyboard activation and reports a truthful active count', async () => {
    publicSources()

    const registerMany = vi.fn()

    const rest = vi.fn(async () => ({
      profiles: [],
      runs: [{ id: 'r1', task_id: 't1', identity_key: 'kanban-worker-0123456789abcdef', title: 'Ship it', board: 'main', status: 'running' }]
    }))

    plugin.register({
      registerMany,
      rest,
      storage: { get: vi.fn((_key, fallback) => fallback), remove: vi.fn(), set: vi.fn() }
    } as never)

    const chip = registerMany.mock.calls[0]?.[0].find((item: { id: string }) => item.id === 'chip')
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const opened = vi.fn()
    const onOpen = () => opened()
    const chipNode = chip.render()

    window.addEventListener('hermes:live-agents-focus-roster', onOpen)
    render(<QueryClientProvider client={client}>{chipNode}</QueryClientProvider>)

    const button = await screen.findByRole('button', { name: 'Open Agent Roster, 1 active or needing attention' })

    button.focus()
    fireEvent.keyDown(button, { key: 'Enter' })

    expect(opened).toHaveBeenCalledTimes(1)
    window.removeEventListener('hermes:live-agents-focus-roster', onOpen)
  })

  it('retains finished history and collapse/dismiss choices across page remounts', async () => {
    publicSources()
    const values = new Map<string, unknown>()

    const storage = {
      get: vi.fn((key: string, fallback: unknown) => values.has(key) ? values.get(key) : fallback),
      remove: vi.fn((key: string) => values.delete(key)),
      set: vi.fn((key: string, value: unknown) => values.set(key, value))
    }

    const rest = vi.fn(async () => ({
      profiles: [],
      runs: [{ id: 'r-finished', task_id: 't1', identity_key: 'kanban-worker-0123456789abcdef', title: 'Retained result', board: 'main', status: 'done', ended_at: 20 }]
    }))

    const registerMany = vi.fn()

    plugin.register({ registerMany, rest, storage } as never)
    const page = registerMany.mock.calls[0]?.[0].find((item: { id: string }) => item.id === 'page')
    const firstClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const firstPage = page.render()
    const first = render(<QueryClientProvider client={firstClient}>{firstPage}</QueryClientProvider>)

    expect((await screen.findAllByText('Retained result')).length).toBeGreaterThan(0)
    await waitFor(() => expect(values.get('profile:default:history')).toBeTruthy())

    fireEvent.click(within(screen.getByRole('article', { name: 'Kanban builder, finished' })).getByRole('button', { expanded: true }))
    expect(screen.queryByRole('region', { name: 'Run Retained result' })).toBeNull()
    first.unmount()

    const secondClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const secondPage = page.render()

    render(<QueryClientProvider client={secondClient}>{secondPage}</QueryClientProvider>)

    expect((await screen.findAllByText('Retained result')).length).toBeGreaterThan(0)
    expect(screen.queryByRole('region', { name: 'Run Retained result' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss Kanban builder' }))
    expect(screen.queryByText('Retained result')).toBeNull()
    expect(values.get('profile:default:dismissed')).toEqual(['kanban:kanban-worker-0123456789abcdef'])
  })

  it('enables only controls backed by an exact public capability', async () => {
    publicSources()
    const registerMany = vi.fn()

    const rest = vi.fn(async () => ({
      profiles: [],
      runs: [{ id: '42', task_id: 't1', identity_key: 'kanban-worker-0123456789abcdef', title: 'Ship it', board: 'main', status: 'running', started_at: 10 }]
    }))

    plugin.register({
      registerMany,
      rest,
      storage: { get: vi.fn((_key, fallback) => fallback), remove: vi.fn(), set: vi.fn() }
    } as never)
    const page = registerMany.mock.calls[0]?.[0].find((item: { id: string }) => item.id === 'page')
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const pageNode = page.render()

    render(<QueryClientProvider client={client}>{pageNode}</QueryClientProvider>)

    expect((await screen.findByRole('button', { name: 'steer Ship it' }) as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'stop Ship it' }) as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'pause Ship it' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'openResult Ship it' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('steers an exact active run through the native dialog instead of window.prompt', async () => {
    publicSources()
    const registerMany = vi.fn()

    const rest = vi.fn(async (path: string) => path === '/snapshot'
      ? {
          profiles: [],
          runs: [{ id: '42', task_id: 't1', identity_key: 'kanban-worker-0123456789abcdef', title: 'Ship it', board: 'main', status: 'running', started_at: 10 }]
        }
      : { ok: true })

    plugin.register({
      registerMany,
      rest,
      storage: { get: vi.fn((_key, fallback) => fallback), remove: vi.fn(), set: vi.fn() }
    } as never)
    const page = registerMany.mock.calls[0]?.[0].find((item: { id: string }) => item.id === 'page')
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const pageNode = page.render()

    render(<QueryClientProvider client={client}>{pageNode}</QueryClientProvider>)

    fireEvent.click(await screen.findByRole('button', { name: 'steer Ship it' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Steering instruction' }), {
      target: { value: 'Check the isolated handoff' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Send instruction' }))

    await waitFor(() => expect(rest).toHaveBeenCalledWith('/runs/42/steer?board=main', {
      method: 'POST',
      body: { task_id: 't1', text: 'Check the isolated handoff' }
    }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  })
})
