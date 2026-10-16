import type { ModelOptionsResult } from '@hermes/shared'
import { type QueryClient } from '@tanstack/react-query'
import { useCallback, useRef } from 'react'

import type { ModelSelection } from '@/app/shell/model-menu-panel'
import { getGlobalModelInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { isBusySessionModelSwitch } from '@/lib/gateway-rpc'
import { surfaceModelSwitchConfirm } from '@/lib/guarded-model-switch'
import {
  customDefaultSupersedesPick,
  moaPickRemoved,
  modelOptionsQueryKey,
  requestModelOptions
} from '@/lib/model-options'
import { dismissNotification, notify, notifyError } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import {
  $activeSessionId,
  $currentModel,
  $currentProvider,
  $currentReasoningEffortWire,
  type ComposerModelSource,
  getComposerSelectionGeneration,
  getCurrentModelSource,
  markComposerSelectionManual,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider,
  setCurrentReasoningEffortWire
} from '@/store/session'
import { $sessionStates, sessionTileDelegate } from '@/store/session-states'

interface ModelControlsOptions {
  cacheOwnerConnectionId?: string
  cacheProfile?: string
  queryClient: QueryClient
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

interface ModelSwitchResponse {
  confirm_message?: string
  confirm_required?: boolean
  deferred?: boolean
}

// One pending intent per owned conversation. A later picker choice supersedes it;
// the running turn is never interrupted to change its frozen model policy.
const pendingModelSwitches = new Map<string, symbol>()

const acknowledgedModels = new Map<string, {
  model: string; provider: string; source: ComposerModelSource; wire: string
}>()

const waitForModelTurn = () => new Promise<void>(resolve => { window.setTimeout(resolve, 1000) })

function modelSwitchBaseline(pendingKey: string, touchesPrimary: boolean, liveSessionId: null | string) {
  const state = liveSessionId ? $sessionStates.get()[liveSessionId] : undefined
  return acknowledgedModels.get(pendingKey) ?? {
    model: state?.model ?? (touchesPrimary ? $currentModel.get() : ''),
    provider: state?.provider ?? (touchesPrimary ? $currentProvider.get() : ''),
    wire: state?.reasoningEffortWire ?? (touchesPrimary ? $currentReasoningEffortWire.get() : ''),
    source: getCurrentModelSource()
  }
}

async function runQueuedModelSwitch({
  isCurrentIntent, requestGateway, sessionId, requestSwitch, handleConfirmation,
  paintSelection, cacheSelection, selection, onError, onDone
}: {
  isCurrentIntent: () => boolean
  requestGateway: ModelControlsOptions['requestGateway']
  sessionId: string
  requestSwitch: () => Promise<ModelSwitchResponse>
  handleConfirmation: (result: ModelSwitchResponse) => Promise<boolean>
  paintSelection: () => void
  cacheSelection: (provider: string, model: string) => void
  selection: ModelSelection
  onError: (error: unknown) => void
  onDone: () => void
}): Promise<void> {
  try {
    while (isCurrentIntent()) {
      await waitForModelTurn()

      if (!isCurrentIntent()) { return }

      const active = await requestGateway<{ sessions: Array<{ id: string; status: string }> }>(
        'session.active_list', { current_session_id: sessionId })

      if (!Array.isArray(active.sessions)) { throw new Error('Could not verify whether the turn ended') }
      const status = active.sessions.find(row => row.id === sessionId)?.status

      if (status === 'starting' || status === 'waiting' || status === 'working') { continue }
      await requestGateway('session.activate', { session_id: sessionId })

      if (!isCurrentIntent()) { return }
      let result: ModelSwitchResponse

      try {
        result = await requestSwitch()
      } catch (retryError) {
        if (isBusySessionModelSwitch(retryError)) { continue }
        throw retryError
      }

      if (!isCurrentIntent()) { return }

      if (result?.confirm_required) { await handleConfirmation(result) }
      else {
        paintSelection()
        cacheSelection(selection.provider, selection.model)
      }

      return
    }
  } catch (retryError) {
    if (isCurrentIntent()) { onError(retryError) }
  } finally {
    if (isCurrentIntent()) { onDone() }
  }
}

export function useModelControls({
  cacheOwnerConnectionId,
  cacheProfile,
  queryClient,
  requestGateway
}: ModelControlsOptions) {
  const { t } = useI18n()
  const copy = t.desktop
  const profileRefreshEpochRef = useRef(0)

  // All callbacks here read reactive session state from the store (.get())
  // rather than capturing it as a prop. The actions bag in wiring.tsx mutates
  // in place to keep a stable identity, so memoized surfaces capture these
  // callbacks once and never re-evaluate — a captured prop would be stale
  // forever. The store read is always current.
  const updateModelOptionsCache = useCallback(
    (
      sessionId: null | string,
      provider: string,
      model: string,
      includeGlobal: boolean,
      profile = cacheProfile || $activeGatewayProfile.get(),
      ownerConnectionId = cacheOwnerConnectionId
    ) => {
      const patch = (prev: ModelOptionsResult | undefined) => {
        // Selection state can update before the catalog query has resolved.
        // Keep that optimistic cache structurally complete; the composer
        // interprets a response without `providers` as an empty catalog.
        const providers = prev?.providers?.length
          ? prev.providers
          : provider && model
            ? [{ models: [model], name: provider, slug: provider }]
            : []

        return { ...prev, provider, model, providers }
      }

      queryClient.setQueryData<ModelOptionsResult>(modelOptionsQueryKey(profile, sessionId, ownerConnectionId), patch)

      if (includeGlobal) {
        queryClient.setQueryData<ModelOptionsResult>(modelOptionsQueryKey(profile, null, ownerConnectionId), patch)
      }
    },
    [cacheOwnerConnectionId, cacheProfile, queryClient]
  )

  // Settings → Model writes the profile default, which the backend applies to
  // new sessions only. Keep a live session's renderer state and session-scoped
  // model-options cache authoritative instead of briefly painting the saved
  // default as if the active agent had switched. Marking the composer as
  // default-derived still lets the next fresh draft reseed from profile config.
  const applySavedMainModel = useCallback(
    (provider: string, model: string) => {
      const liveSessionId = $activeSessionId.get()

      setCurrentModelSource('default')

      if (!liveSessionId) {
        setCurrentProvider(provider)
        setCurrentModel(model)
      }

      // A null session id is the profile-global model-options key. Never patch
      // the live session key here: only config.set --session may change it.
      updateModelOptionsCache(null, provider, model, false)
    },
    [updateModelOptionsCache]
  )

  // Seed the composer's model state from the profile default. `force` reseeds
  // for a profile swap (the new profile has its own default); otherwise this
  // only fills an EMPTY selection so a user's pick (plain UI state in
  // $currentModel) survives the lifecycle refreshes that fire on boot / fresh
  // draft / session events. A live session owns the footer, so skip entirely.
  // Two provably-stale manual picks are the exception and reseed from the
  // profile default: the virtual `moa` provider (#90244) and a bare provider
  // slug the default has migrated to its `custom:<key>` form (#81922).
  const refreshCurrentModel = useCallback(
    async (force = false) => {
      // A forced profile swap opens a new intent epoch; an older in-flight
      // response for a previous profile must stand down when it resolves.
      if (force) {
        profileRefreshEpochRef.current += 1
      }

      const profileRefreshEpoch = profileRefreshEpochRef.current
      const profile = $activeGatewayProfile.get()

      try {
        if ($activeSessionId.get()) {
          return
        }

        // A manual pick is sticky. It is never diffed against the catalog: rows
        // are hints, and a custom slug the row lacks is still the user's choice
        // (the gateway validates it on switch). ONE exception, narrower than a
        // catalog diff: a pick pointing at the virtual `moa` provider, whose row
        // the catalog omits entirely once no preset is enabled — that absence is
        // authoritative, and without the exception the pill reads
        // `Model · moa: default` forever (#90244).
        const manualPick = () => Boolean($currentModel.get()) && getCurrentModelSource() === 'manual'

        const pickProvider = () => ($currentProvider.get() || '').trim()

        const staleMoaPick = () => !force && manualPick() && pickProvider().toLowerCase() === 'moa'

        // A SECOND exception: a bare provider slug can be a stale spelling of a
        // `custom:<key>` profile default (#87035 aliases the two spellings for
        // one endpoint). Shipping the bare form resolves the NATIVE provider and
        // silently drops the entry's `extra_body` (#81922), so the pick must
        // reseed. Only a bare slug qualifies — a pick that already names a
        // provider class is a distinct choice and stays sticky — and the profile
        // default is a backend fact, so this needs the same `getGlobalModelInfo`
        // fetch the empty / default-sourced path already makes.
        const maybeSupersededPick = () =>
          !force && manualPick() && pickProvider() !== '' && !pickProvider().includes(':')

        if (manualPick() && !force && !staleMoaPick() && !maybeSupersededPick()) {
          return
        }

        // Snapshot the selection generation before awaiting so a picker click
        // that lands while getGlobalModelInfo is in flight wins over this older
        // default — value comparisons alone miss re-selecting the same row.
        const selectionGeneration = getComposerSelectionGeneration()

        // Judge the moa pick against the catalog: peek the picker's own cache
        // first and only fetch (deduped with the in-flight UI query) when it is
        // empty, so the pill reseeds even before the chat view mounts its query.
        // A catalog that fails to load keeps the pick — absence of data is not
        // absence of the preset.
        let reseedStaleMoa = false

        if (staleMoaPick()) {
          const catalogProfile = cacheProfile || profile
          const catalogKey = modelOptionsQueryKey(catalogProfile, null, cacheOwnerConnectionId)

          const catalog =
            queryClient.getQueryData<ModelOptionsResult>(catalogKey) ??
            (await queryClient.fetchQuery({
              queryKey: catalogKey,
              queryFn: (): Promise<ModelOptionsResult> =>
                requestModelOptions({ profile: catalogProfile, request: requestGateway })
            }))

          reseedStaleMoa = moaPickRemoved(catalog, 'moa', $currentModel.get())

          if (!reseedStaleMoa) {
            return
          }
        }

        const result = await getGlobalModelInfo(profile)

        if (
          profileRefreshEpochRef.current !== profileRefreshEpoch ||
          $activeSessionId.get() ||
          getComposerSelectionGeneration() !== selectionGeneration ||
          (manualPick() &&
            !force &&
            !reseedStaleMoa &&
            !customDefaultSupersedesPick($currentProvider.get(), result.provider ?? ''))
        ) {
          return
        }

        if (typeof result.model === 'string') {
          setCurrentModel(result.model)
        }

        if (typeof result.provider === 'string') {
          setCurrentProvider(result.provider)
        }

        if (typeof result.model === 'string' || typeof result.provider === 'string') {
          setCurrentModelSource('default')
        }
      } catch {
        // The delayed session.info event still updates this once the agent is ready.
      }
    },
    [cacheOwnerConnectionId, cacheProfile, queryClient, requestGateway]
  )

  // Drop a sticky composer pick so new chats follow Settings → Model again,
  // without making the user re-apply the default they already have (#107410).
  const followDefaultModel = useCallback(() => {
    setCurrentModelSource('default')
    void refreshCurrentModel()
  }, [refreshCurrentModel])

  // Returns whether the switch was applied so callers can await it before
  // applying follow-up changes. `true` means applied (or deferred/busy-queued
  // for the next turn). `false` means NOT applied — either pending
  // confirmation (warning with Confirm action already shown, pill rolled back)
  // or a real failure (error toast). Callers must NOT treat `false` as a
  // generic failure: for `pending` the gateway intentionally returned
  // `confirm_required` and no error should be surfaced.
  // The composer model is plain UI state: with no live session it's just
  // stored (and shipped on the next session.create); with one it's scoped to
  // that session via config.set. It NEVER writes the profile default — that
  // lives in Settings → Model — so picking a model here can't silently mutate
  // global config.
  //
  // `selection.sessionId` targets a specific surface (tile). When omitted, the
  // primary `$activeSessionId` is used (overlay / legacy callers). A tile
  // switch must not touch the primary globals — and must not be blocked by a
  // busy primary turn.
  const selectModel = useCallback(
    async (selection: ModelSelection): Promise<boolean> => {
      const primaryRuntimeId = $activeSessionId.get()
      const liveSessionId = 'sessionId' in selection ? (selection.sessionId ?? null) : primaryRuntimeId
      const touchesPrimary = !liveSessionId || liveSessionId === primaryRuntimeId

      const liveGatewayProfile = cacheProfile || $activeGatewayProfile.get()
      const pendingKey = JSON.stringify([cacheOwnerConnectionId, liveGatewayProfile, liveSessionId])

      // A superseding picker sees the earlier optimistic paint. Keep the last
      // acknowledged route instead of treating that paint as the rollback target.
      const previous = modelSwitchBaseline(pendingKey, touchesPrimary, liveSessionId)

      if (pendingModelSwitches.has(pendingKey)) { dismissNotification(`model-switch-${pendingKey}`) }
      pendingModelSwitches.delete(pendingKey)
      // Claim this choice before the first RPC: an older response can arrive
      // after a newer choice has already been applied.
      const token = Symbol('model pick')

      if (liveSessionId) {
        pendingModelSwitches.set(pendingKey, token)
        acknowledgedModels.set(pendingKey, previous)
      }

      const isCurrentIntent = () => pendingModelSwitches.get(pendingKey) === token

      const clearIntent = () => {
        if (isCurrentIntent()) {
          pendingModelSwitches.delete(pendingKey)
          acknowledgedModels.delete(pendingKey)
        }
      }

      const paintSelection = () => {
        if (liveSessionId) {
          // Every live surface, including primary, renders from its session slice.
          // Update the owning cache even if focus moved while the RPC was pending.
          sessionTileDelegate()?.updateSession(liveSessionId, state => ({
            ...state,
            model: selection.model,
            provider: selection.provider,
            // session.info can confirm this route before the mutation response.
            // Only withdraw a stamp that still belongs to a different route.
            reasoningEffortWire: state.model === selection.model && state.provider === selection.provider
              ? state.reasoningEffortWire
              : ''
          }))
        }

        if (touchesPrimary && (!liveSessionId || $activeSessionId.get() === liveSessionId)) {
          setCurrentModel(selection.model)
          setCurrentProvider(selection.provider)
          markComposerSelectionManual()
        }
      }

      const cacheSelection = (provider: string, model: string) => {
        updateModelOptionsCache(liveSessionId, provider, model, touchesPrimary && !liveSessionId, liveGatewayProfile)
      }

      const rollbackSelection = () => {
        if (liveSessionId && !isCurrentIntent()) { return }

        if (liveSessionId) {
          sessionTileDelegate()?.updateSession(liveSessionId, state => ({
            ...state,
            model: previous.model,
            provider: previous.provider,
            reasoningEffortWire: previous.wire
          }))
        }

        if (touchesPrimary && (!liveSessionId || $activeSessionId.get() === liveSessionId)) {
          setCurrentModel(previous.model)
          setCurrentProvider(previous.provider)
          // The setters withdraw the wire stamp on a change; the old route's stamp is still true.
          setCurrentReasoningEffortWire(previous.wire)
          setCurrentModelSource(previous.source)
        }

        cacheSelection(previous.provider, previous.model)
      }

      paintSelection()
      cacheSelection(selection.provider, selection.model)

      // No live session yet: the pick is pure UI state. session.create reads
      // $currentModel/$currentProvider and applies it as that session's override.
      if (!liveSessionId) {
        return true
      }

      // The PRIMARY profile's main agent lets the gateway decide persistence
      // (resolve_persist_behavior): session-only by default, persisted when
      // model.persist_switch_by_default is true or when no default has ever
      // been configured (the first-ever pick, so resolve_provider never falls
      // through to a leftover OPENAI_API_KEY env var — #86414). A plain pick
      // no longer silently rewrites config.yaml (#90235); Settings → Model
      // remains the explicit "set as default" door.
      //
      // Two things stay --session, deliberately:
      //  - a SECONDARY chat tile: picking a model there must not rewrite the
      //    profile default (the cross-session-contamination guard).
      //  - MoA (mixture-of-agents) presets: a transient orchestration choice
      //    that must never become the persisted global gateway default.
      const isSessionOnlyPreset = (selection.provider || '').toLowerCase() === 'moa'
      const scope = touchesPrimary && !isSessionOnlyPreset ? '' : ' --session'

      const requestSwitch = async (confirmExpensiveModel = false) => {
        const result = await requestGateway<ModelSwitchResponse>('config.set', {
          session_id: liveSessionId,
          key: 'model',
          value: `${selection.model} --provider ${selection.provider}${scope}`,
          ...(confirmExpensiveModel ? { confirm_expensive_model: true } : {})
        })

        // Record every committed response, including queued/confirmed requests
        // superseded while in flight. Newer intent keeps its optimistic paint,
        // but must roll back to the model the gateway actually acknowledged.
        if (!result?.confirm_required && !result?.deferred) {
          if (acknowledgedModels.get(pendingKey) === previous) {
            previous.model = selection.model
            previous.provider = selection.provider
            previous.wire = ''
            previous.source = 'manual'
          }

          void queryClient.invalidateQueries({
            queryKey: modelOptionsQueryKey(liveGatewayProfile, liveSessionId, cacheOwnerConnectionId)
          })
        }

        return result
      }

      const handleConfirmation = (result: ModelSwitchResponse) => {
        if (!isCurrentIntent()) { return Promise.resolve(false) }
        rollbackSelection()
        if (touchesPrimary && $activeSessionId.get() !== liveSessionId) { return Promise.resolve(false) }

          // ONE shared applier for guarded switches (#95293): the same
          // confirm flow the Bots editor routes through — never fork this
          // logic per surface.
          // Not awaited: `selectModel` answers "was the switch applied NOW",
          // and that answer only exists once the user answers the dialog.
        return surfaceModelSwitchConfirm({
            confirmMessage: result.confirm_message,
            failureMessage: copy.modelSwitchFailed,
            // Staleness guard — the session or model can move on while the
            // dialog is open. Answering it must not clobber the newer choice:
            // bail (with a notice) if the live state no longer matches the
            // snapshot this prompt was created for.
            isStale: () =>
              !isCurrentIntent() || (touchesPrimary
                ? $activeSessionId.get() !== liveSessionId ||
                  $currentModel.get() !== previous.model ||
                  $currentProvider.get() !== previous.provider
                : !liveSessionId ||
                  $sessionStates.get()[liveSessionId]?.model !== previous.model ||
                  $sessionStates.get()[liveSessionId]?.provider !== previous.provider),
            model: selection.model,
            repaint: () => {
              paintSelection()
              cacheSelection(selection.provider, selection.model)
            },
            requestConfirmed: () => requestSwitch(true),
            rollback: rollbackSelection
        })
      }

      try {
        const result = await requestSwitch()

        if (!isCurrentIntent()) {
          return false
        }

        if (result?.confirm_required) {
          void handleConfirmation(result).finally(() => {
            clearIntent()
          })

          return false
        }

        if (!result?.deferred) { paintSelection() }
        clearIntent()

        return true
      } catch (err) {
        if (!isCurrentIntent()) { return false }

        // The canonical owner refuses a mutation while the turn owns its frozen
        // policy. Keep the current model truthful, then retry after the owner
        // reports that turn idle. A fresh resume supplies the new CAS revision.
        if (isBusySessionModelSwitch(err)) {
          rollbackSelection()

          const notificationId = notify({
            id: `model-switch-${pendingKey}`,
            kind: 'info',
            title: 'Model switch queued',
            message: 'The current reply will finish before the model changes.',
            durationMs: 0
          })

          void runQueuedModelSwitch({
            isCurrentIntent,
            requestGateway,
            sessionId: liveSessionId,
            requestSwitch,
            handleConfirmation,
            paintSelection,
            cacheSelection,
            selection,
            onError: retryError => notifyError(retryError, copy.modelSwitchFailed),
            onDone: () => {
              clearIntent()
              dismissNotification(notificationId)
            }
          })

          return true
        }

        rollbackSelection()
        clearIntent()
        notifyError(err, copy.modelSwitchFailed)

        return false
      }
    },
    [cacheOwnerConnectionId, cacheProfile, copy.modelSwitchFailed, queryClient, requestGateway, updateModelOptionsCache]
  )

  return { applySavedMainModel, followDefaultModel, refreshCurrentModel, selectModel }
}
