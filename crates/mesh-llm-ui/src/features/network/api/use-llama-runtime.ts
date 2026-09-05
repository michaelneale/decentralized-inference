import { useEffect, useRef, useState } from 'react'
import type { LlamaRuntimePayload } from '@/lib/api/types'
import { env } from '@/lib/env'
import { fetchLlamaRuntimeWithSignal } from '@/features/network/api/runtime'
import { useRuntimeEventsV1 } from '@/features/network/api/use-runtime-events-v1'

const LLAMA_RUNTIME_REFRESH_MS = 2_500
const LLAMA_RUNTIME_RECONNECT_MS = 1_000

type LlamaRuntimeState = {
  data: LlamaRuntimePayload | null
  loading: boolean
  error: string | null
}

/**
 * `/api/runtime/llama` stays the authoritative snapshot. The v1 runtime event
 * stream is consumed as the live signal that a snapshot is worth re-reading;
 * the legacy `/api/runtime/events` EventSource opens only once v1 has
 * resolved to a documented fallback (capability absent/unsupported, 404/403,
 * a malformed mandatory frame, or reconnect exhaustion). Startup never waits
 * on v1: the first authoritative fetch and the periodic refresh run while the
 * capability probe is still in flight.
 */
export function useLlamaRuntime(enabled: boolean) {
  const [state, setState] = useState<LlamaRuntimeState>({
    data: null,
    loading: false,
    error: null
  })
  const runtimeEvents = useRuntimeEventsV1({ enabled })
  const legacyStreamEnabled = runtimeEvents.mode === 'fallback'
  const reloadRef = useRef<() => void>(() => undefined)

  useEffect(() => {
    if (!enabled) {
      reloadRef.current = () => undefined
      setState({ data: null, loading: false, error: null })
      return undefined
    }

    let cancelled = false
    let runtimeRequest: AbortController | null = null
    let runtimeEventSource: EventSource | null = null
    let reconnectTimer: number | null = null
    let refreshTimer: number | null = null

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
    }

    const clearRefreshTimer = () => {
      if (refreshTimer !== null) {
        window.clearInterval(refreshTimer)
        refreshTimer = null
      }
    }

    const closeRuntimeEvents = () => {
      if (!runtimeEventSource) return
      runtimeEventSource.onopen = null
      runtimeEventSource.onmessage = null
      runtimeEventSource.onerror = null
      runtimeEventSource.close()
      runtimeEventSource = null
    }

    const abortRuntimeRequest = () => {
      runtimeRequest?.abort()
      runtimeRequest = null
    }

    async function loadRuntime() {
      abortRuntimeRequest()
      const controller = new AbortController()
      runtimeRequest = controller
      setState((current) => ({ ...current, loading: current.data == null, error: null }))

      try {
        const payload = await fetchLlamaRuntimeWithSignal(controller.signal)
        if (!cancelled && runtimeRequest === controller) {
          runtimeRequest = null
          setState({ data: payload, loading: false, error: null })
        }
      } catch (error) {
        if (cancelled || controller.signal.aborted) return
        if (runtimeRequest === controller) {
          runtimeRequest = null
        }
        setState((current) => ({
          data: current.data,
          loading: false,
          error: error instanceof Error ? error.message : 'Runtime llama request failed'
        }))
      }
    }

    const ensurePeriodicRefresh = () => {
      if (refreshTimer !== null) return
      refreshTimer = window.setInterval(() => void loadRuntime(), LLAMA_RUNTIME_REFRESH_MS)
    }

    const scheduleReconnect = () => {
      if (cancelled || reconnectTimer !== null) return
      ensurePeriodicRefresh()
      closeRuntimeEvents()
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null
        connectLegacyRuntimeEvents()
      }, LLAMA_RUNTIME_RECONNECT_MS)
    }

    function connectLegacyRuntimeEvents() {
      if (cancelled || runtimeEventSource) return

      if (typeof EventSource === 'undefined') {
        ensurePeriodicRefresh()
        return
      }

      let source: EventSource
      try {
        source = new EventSource(`${env.managementApiUrl}/api/runtime/events`)
      } catch (error) {
        console.warn(
          'Failed to connect llama runtime stream:',
          error instanceof Error ? error.message : 'failed to create EventSource'
        )
        scheduleReconnect()
        return
      }

      runtimeEventSource = source
      source.onopen = () => {
        if (cancelled) return
        abortRuntimeRequest()
        clearRefreshTimer()
        setState((current) => ({ ...current, error: null }))
      }
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data as string) as LlamaRuntimePayload
          if (!cancelled) {
            setState({ data: payload, loading: false, error: null })
          }
        } catch {
          // ignore malformed runtime event payloads
        }
      }
      source.onerror = () => {
        if (cancelled) return
        scheduleReconnect()
      }
    }

    reloadRef.current = () => void loadRuntime()
    void loadRuntime()
    if (legacyStreamEnabled) connectLegacyRuntimeEvents()
    else ensurePeriodicRefresh()

    return () => {
      cancelled = true
      reloadRef.current = () => undefined
      abortRuntimeRequest()
      clearReconnectTimer()
      clearRefreshTimer()
      closeRuntimeEvents()
    }
  }, [enabled, legacyStreamEnabled])

  useEffect(() => {
    if (!enabled || runtimeEvents.revision === 0) return
    reloadRef.current()
  }, [enabled, runtimeEvents.revision])

  return state
}
