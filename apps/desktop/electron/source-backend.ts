import { existsSync } from 'node:fs'
import path from 'node:path'

import { buildDesktopBackendEnv } from './backend-env'
import { execProbe, PROBE_TIMEOUT_MS } from './backend-probes'
import { resolveInstallationLauncher, userLauncherInstallRoot } from './updater-process'

export interface SourceBackend {
  kind: 'command' | 'python'
  label: string
  command: string
  args: string[]
  env: NodeJS.ProcessEnv
  root: string
  bootstrap: false
  shell: boolean
  local: 'installed'
}

interface SourceOptions {
  isWindows?: boolean
  env?: NodeJS.ProcessEnv
}

const runtimeProbes = new Map<string, Promise<void>>()

export async function isSourceRuntimeUsable(root: string, options: SourceOptions & { hermesHome?: string } = {}): Promise<boolean> {
  try {
    return (await resolveSourceInstallationBackend(root, [], options)) !== null
  } catch {
    return false
  }
}

function probeRuntime(command: string, options: Parameters<typeof execProbe>[2]): Promise<void> {
  const key = JSON.stringify([command, options])
  const pending = runtimeProbes.get(key)
  if (pending) { return pending }
  const probe = execProbe(command, ['--version'], options).finally(() => runtimeProbes.delete(key))
  runtimeProbes.set(key, probe)
  return probe
}

/** Keep the validated command. PM owns interpreter and generation selection. */
export async function resolveSourceInstallationBackend(
  root: string,
  args: string[],
  options: SourceOptions & { hermesHome?: string } = {}
): Promise<SourceBackend | null> {
  if (!existsSync(path.join(root, 'hermes_cli', 'main.py'))) {
    return null
  }

  const isWindows: boolean = options.isWindows ?? process.platform === 'win32'
  const launcher: string | null = await resolveInstallationLauncher(root, isWindows, options.hermesHome)

  if (!launcher) {
    throw new Error(`Hermes is installed at ${root}, but its installation launcher is unavailable. Repair this installation before retrying.`)
  }

  const shell: boolean = isWindows && /\.(cmd|bat)$/i.test(launcher)
  const command: string = shell ? `"${launcher}"` : launcher
  const env: NodeJS.ProcessEnv = buildDesktopBackendEnv({ currentEnv: options.env ?? process.env })

  try {
    await probeRuntime(command, {
      cwd: root,
      env: { ...process.env, ...options.env, ...env },
      shell,
      stdio: 'ignore',
      timeout: PROBE_TIMEOUT_MS,
      windowsHide: true
    })
  } catch (error) {
    throw new Error(`Hermes is installed at ${root}, but its runtime could not be reached: ${String(error)}`, { cause: error })
  }

  return {
    kind: 'command',
    label: `Hermes at ${root}`,
    command,
    args: [...args],
    env,
    root,
    bootstrap: false,
    shell,
    local: 'installed'
  }
}

/** Try every installed location before surfacing a damaged runtime. The caller
 * gates this entire discovery step so ignored installs are never executed. */
export async function resolveInstalledSourceBackend(
  root: string,
  args: string[],
  options: SourceOptions & { hermesHome?: string } = {}
): Promise<SourceBackend | null> {
  let failure: unknown
  try {
    const active = await resolveSourceInstallationBackend(root, args, options)
    if (active) { return active }
  } catch (error) {
    failure = error
  }

  try {
    const published = await userLauncherInstallRoot(options.isWindows, options.hermesHome)
    if (published && published.root !== root) {
      const alternate = await resolveSourceInstallationBackend(published.root, args, options)
      if (alternate) { return alternate }
    }
  } catch (error) {
    failure ??= error
  }

  if (failure) { throw failure }
  return null
}

/** Developer overrides retain their interpreter, even outside the checkout. */
export function createSourcePythonBackend(
  root: string,
  python: string | null,
  args: string[],
  options: SourceOptions = {}
): SourceBackend | null {
  if (!python) {
    return null
  }

  let command: string = python

  if ((options.isWindows ?? process.platform === 'win32') && /[\\/]pythonw\.exe$/i.test(python)) {
    // Use only the console interpreter beside the selected windowless one.
    const consolePython: string = python.replace(/pythonw\.exe$/i, 'python.exe')

    if (existsSync(consolePython)) {
      command = consolePython
    }
  }

  return {
    kind: 'python',
    label: `Hermes source at ${root}`,
    command,
    args: ['-m', 'hermes_cli.main', ...args],
    // The backend runs in the user's workspace cwd, and the selected
    // interpreter need not have this checkout installed: name it explicitly.
    // (The scrubbed inherited value could point at another checkout.)
    env: { ...buildDesktopBackendEnv({ currentEnv: options.env ?? process.env }), PYTHONPATH: root },
    root,
    bootstrap: false,
    shell: false,
    local: 'installed'
  }
}
