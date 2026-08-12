/**
 * End-to-end proof for the bundled Live Agents fleet monitor.
 *
 * The test uses the existing credential-stripped Desktop sandbox and mock
 * inference server. The Kanban CLI dispatches a real worker process against
 * that isolated backend; the mock controls only its deterministic model reply.
 */
import { spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import {
  buildAppEnv,
  type MockBackendFixture,
  setupMockBackend,
  waitForAppReady,
} from './fixtures'
import { createBackgroundReleaseHandle, restartMockServer } from './mock-server'

const DESKTOP_ROOT = path.resolve(import.meta.dirname, '..')
const REPO_ROOT = path.resolve(DESKTOP_ROOT, '..', '..')

const PYTHON = [
  path.join(REPO_ROOT, 'venv', 'bin', 'python'),
  path.join(REPO_ROOT, '.venv', 'bin', 'python'),
  path.join(REPO_ROOT, '.venv', 'bin', 'python3'),
].find(candidate => fs.existsSync(candidate)) ?? 'python3'

const ACTIVE_TITLE = 'E2E Live Agents active worker'
const STOP_TITLE = 'E2E Live Agents stopped worker'
const STEER_TEXT = 'Confirm the isolated Live Agents handoff.'

function runHermes(fixture: MockBackendFixture, args: string[]): string {
  const result = spawnSync(PYTHON, ['-m', 'hermes_cli.main', ...args], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    env: {
      ...buildAppEnv(fixture.sandbox),
      MOCK_API_KEY: 'e2e-mock-key',
    },
  })

  expect(result.status, result.stderr || result.stdout).toBe(0)

  return result.stdout
}

function createKanbanTask(fixture: MockBackendFixture, title = ACTIVE_TITLE): string {
  const created = JSON.parse(runHermes(fixture, [
    'kanban',
    'create',
    title,
    '--body',
    'Exercise the isolated fleet monitor with an actual worker.',
    '--assignee',
    'default',
    '--workspace',
    'scratch',
    '--json',
  ])) as { id: string }

  const dispatched = JSON.parse(runHermes(fixture, ['kanban', 'dispatch', '--max', '1', '--json'])) as {
    spawned: Array<{ task_id: string }>
  }

  expect(dispatched.spawned).toContainEqual(expect.objectContaining({ task_id: created.id }))

  return created.id
}

test.describe('Live Agents — real isolated sources', () => {
  test.describe.configure({ mode: 'serial' })

  let fixture: MockBackendFixture
  let taskId = ''

  test.beforeAll(async () => {
    restartMockServer()
    fixture = await setupMockBackend({
      mockServer: {
        holdKanbanWorker: true,
      },
    })
    await waitForAppReady(fixture, 120_000)

    taskId = createKanbanTask(fixture)
    await expect.poll(
      () => JSON.parse(runHermes(fixture, ['kanban', 'show', taskId, '--json'])) as { task: { status: string } },
      { message: 'the real Kanban worker should remain active at the isolated tool boundary', timeout: 30_000 },
    ).toMatchObject({ task: { status: 'running' } })
    await expect.poll(
      () => fixture.mock.receivedPrompts.some(prompt => prompt === `work kanban task ${taskId}`),
      { message: 'the real Kanban worker should reach the isolated inference server', timeout: 30_000 },
    ).toBe(true)
    await expect.poll(() => fixture.mock.heldCompletionCount(), {
      message: 'the real Kanban worker response should be held at the isolated inference boundary',
      timeout: 30_000,
    }).toBe(1)
  })

  test.afterAll(async () => {
    fixture?.mock.releaseHeldStream()
    await fixture?.cleanup()
  })

  test('shows active multi-source work, controls the exact worker, and retains completion', async () => {
    const page = fixture.page

    await page.getByText('Live Agents', { exact: true }).first().click()
    await expect(page.getByRole('heading', { name: 'Live Agents' })).toBeVisible()
    await page.getByRole('button', { name: 'Refresh' }).click()

    const activeRun = page.getByRole('region', { name: `Run ${ACTIVE_TITLE}` })

    await expect(activeRun).toBeVisible({ timeout: 30_000 })
    await expect(activeRun.getByText('active', { exact: true })).toBeVisible()
    await expect(page.getByRole('region', { name: 'Run No active gateway work' })).toBeVisible()
    await expect(page.getByRole('complementary', { name: 'Unavailable agent sources' })).toContainText('Configured remote machines')

    const firstCard = page.getByRole('article').first()
    await expect(firstCard).toHaveAccessibleName(/active$/)
    await expect(activeRun.getByRole('button', { name: `pause ${ACTIVE_TITLE}` })).toBeDisabled()
    await expect(activeRun.getByRole('button', { name: `steer ${ACTIVE_TITLE}` })).toBeEnabled()
    await expect(activeRun.getByRole('button', { name: `stop ${ACTIVE_TITLE}` })).toBeEnabled()
    await expect(activeRun.getByRole('button', { name: `openResult ${ACTIVE_TITLE}` })).toBeDisabled()

    await page.screenshot({ path: 'test-results/live-agents-active-multi-source.png', fullPage: true })

    await activeRun.getByRole('button', { name: `steer ${ACTIVE_TITLE}` }).click()
    await page.getByRole('textbox', { name: 'Steering instruction' }).fill(STEER_TEXT)
    await page.getByRole('button', { name: 'Send instruction' }).click()
    await expect.poll(
      () => runHermes(fixture, ['kanban', 'show', taskId, '--json']),
      { timeout: 15_000 },
    ).toContain(STEER_TEXT)

    fixture.mock.releaseHeldStream()
    await activeRun.getByRole('button', { name: `stop ${ACTIVE_TITLE}` }).click()
    await expect.poll(
      () => JSON.parse(runHermes(fixture, ['kanban', 'show', taskId, '--json'])) as {
        task: { status: string }
        runs: Array<{ ended_at: number | null }>
      },
      { message: 'the exact worker should stop before the terminal-state projection is completed', timeout: 30_000 },
    ).toMatchObject({ task: { status: 'ready' }, runs: [{ ended_at: expect.any(Number) }] })
    runHermes(fixture, [
      'kanban',
      'complete',
      taskId,
      '--summary',
      'Completed the isolated Live Agents terminal-state projection.',
    ])
    await expect.poll(
      () => JSON.parse(runHermes(fixture, ['kanban', 'show', taskId, '--json'])) as { task: { status: string } },
      { message: 'the real Kanban worker should finish', timeout: 45_000 },
    ).toMatchObject({ task: { status: 'done' } })

    await page.getByRole('button', { name: 'Refresh' }).click()
    await expect(activeRun.getByText('finished', { exact: true })).toBeVisible({ timeout: 30_000 })
    await expect(activeRun).toBeVisible()
    await page.screenshot({ path: 'test-results/live-agents-finished-retained.png', fullPage: true })
  })
})

test.describe('Live Agents — real local agent sources', () => {
  test.describe.configure({ mode: 'serial' })

  const backgroundRelease = createBackgroundReleaseHandle()
  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    restartMockServer()
    fixture = await setupMockBackend({
      mockServer: {
        backgroundReleasePath: backgroundRelease.path,
        holdFirstCompletionContaining: 'Analyze cross-session state',
      },
    })
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    fixture?.mock.releaseHeldStream()
    backgroundRelease.release()
    await fixture?.cleanup()
    backgroundRelease.cleanup()
  })

  test('shows a real background process and delegated agent without exposing their private inputs', async () => {
    const page = fixture.page
    const composer = page.locator('[contenteditable="true"]').first()

    await composer.waitFor({ state: 'visible', timeout: 10_000 })
    await composer.fill('E2E_SIDEBAR_CROSS')
    await page.keyboard.press('Enter')

    await expect.poll(() => fixture.mock.heldCompletionCount(), {
      message: 'the real delegated agent should remain active at the isolated inference boundary',
      timeout: 30_000,
    }).toBe(1)

    await page.getByText('Live Agents', { exact: true }).first().click()
    await expect(page.getByRole('heading', { name: 'Live Agents' })).toBeVisible()
    await page.getByRole('button', { name: 'Refresh' }).click()

    const backgroundRun = page.getByRole('region', { name: 'Run Background process (command)' })
    const delegationRun = page.getByRole('region', { name: 'Run Delegated work' })

    await expect(backgroundRun).toBeVisible({ timeout: 30_000 })
    await expect(backgroundRun.getByText('active', { exact: true })).toBeVisible()
    await expect(delegationRun).toBeVisible({ timeout: 30_000 })
    await expect(delegationRun.getByText('active', { exact: true })).toBeVisible()
    await expect(page.locator('body')).not.toContainText('long bg output')
    await expect(page.locator('body')).not.toContainText('Analyze cross-session state')

    await page.screenshot({ path: 'test-results/live-agents-real-local-sources.png', fullPage: true })
  })
})

test.describe('Live Agents — exact stop control', () => {
  test.describe.configure({ mode: 'serial' })

  let fixture: MockBackendFixture
  let taskId = ''

  test.beforeAll(async () => {
    restartMockServer()
    fixture = await setupMockBackend({ mockServer: { holdKanbanWorker: true } })
    await waitForAppReady(fixture, 120_000)
    taskId = createKanbanTask(fixture, STOP_TITLE)
    await expect.poll(() => fixture.mock.heldCompletionCount(), {
      message: 'the stop target should be a real worker held at inference',
      timeout: 30_000,
    }).toBe(1)
  })

  test.afterAll(async () => {
    fixture?.mock.releaseHeldStream()
    await fixture?.cleanup()
  })

  test('stops only the exact active worker and releases its board claim', async () => {
    const page = fixture.page

    await page.getByText('Live Agents', { exact: true }).first().click()
    await expect(page.getByRole('heading', { name: 'Live Agents' })).toBeVisible()
    await page.getByRole('button', { name: 'Refresh' }).click()

    const run = page.getByRole('region', { name: `Run ${STOP_TITLE}` })
    await expect(run.getByText('active', { exact: true })).toBeVisible({ timeout: 30_000 })
    await run.getByRole('button', { name: `stop ${STOP_TITLE}` }).click()

    await expect.poll(
      () => JSON.parse(runHermes(fixture, ['kanban', 'show', taskId, '--json'])) as {
        task: { status: string }
        runs: Array<{ ended_at: number | null }>
      },
      { message: 'the exact task should be requeued with its active run ended', timeout: 30_000 },
    ).toMatchObject({ task: { status: 'ready' }, runs: [{ ended_at: expect.any(Number) }] })

    await page.screenshot({ path: 'test-results/live-agents-exact-stop.png', fullPage: true })
  })
})
