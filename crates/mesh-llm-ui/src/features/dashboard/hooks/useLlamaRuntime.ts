import { useEffect, useRef, useState } from 'react'

import type { LlamaRuntimePayload } from '@/features/app-shell/lib/status-types'
import { useRuntimeEventsV1 } from '@/features/network/api/use-runtime-events-v1'

const LLAMA_RUNTIME_REFRESH_MS = 2_500
const LLAMA_RUNTIME_RECONNECT_MS = 1_000

type LlamaRuntimeState = {
  data: LlamaRuntimePayload | null
  loading: boolean
  error: string | null
}

/**
 * Dashboard twin of `@/features/network/api/use-llama-runtime`: the v1 runtime
 * event stream is the live signal, `/api/runtime/llama` stays authoritative,
 * and the legacy `/api/runtime/events` EventSource opens only once v1 has
 * resolved to a documented fallback. Both hooks share the one v1 consumer, so
 * the console never opens a third runtime-event connection.
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
    let controller: AbortController | null = null
    let runtimeEventSource: EventSource | null = null
    let reconnectTimer: number | null = null
    let refreshInterval: number | null = null

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer)
        reconnectTimer = null
      }
    }

    const clearRefreshInterval = () => {
      if (refreshInterval !== null) {
        window.clearInterval(refreshInterval)
        refreshInterval = null
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
      controller?.abort()
      controller = null
    }

    async function loadRuntime() {
      abortRuntimeRequest()
      const requestController = new AbortController()
      controller = requestController
      setState((current) => ({ ...current, loading: current.data == null, error: null }))
      try {
        const response = await fetch('/api/runtime/llama', {
          signal: requestController.signal
        })
        if (!response.ok) {
          throw new Error(`Runtime llama request failed with HTTP ${response.status}`)
        }
        const payload = (await response.json()) as LlamaRuntimePayload
        if (!cancelled && controller === requestController) {
          controller = null
          setState({ data: payload, loading: false, error: null })
        }
      } catch (error) {
        if (requestController.signal.aborted || cancelled) return
        if (controller === requestController) {
          controller = null
        }
        setState((current) => ({
          data: current.data,
          loading: false,
          error: error instanceof Error ? error.message : 'Runtime llama request failed'
        }))
      }
    }

    const ensurePeriodicRefresh = () => {
      if (refreshInterval !== null) return
      refreshInterval = window.setInterval(() => void loadRuntime(), LLAMA_RUNTIME_REFRESH_MS)
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

    const connectLegacyRuntimeEvents = () => {
      if (cancelled || runtimeEventSource) return

      if (typeof EventSource === 'undefined') {
        ensurePeriodicRefresh()
        return
      }

      let source: EventSource
      try {
        source = new EventSource('/api/runtime/events')
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
        clearRefreshInterval()
        setState((current) => ({ ...current, error: null }))
      }
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as LlamaRuntimePayload
          if (!cancelled) {
            setState({ data: payload, loading: false, error: null })
          }
        } catch {
          // ignore malformed runtime event
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
      controller?.abort()
      clearReconnectTimer()
      clearRefreshInterval()
      closeRuntimeEvents()
    }
  }, [enabled, legacyStreamEnabled])

  useEffect(() => {
    if (!enabled || runtimeEvents.revision === 0) return
    reloadRef.current()
  }, [enabled, runtimeEvents.revision])

  return state
}
