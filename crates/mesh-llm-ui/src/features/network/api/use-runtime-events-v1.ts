/**
 * The console's single consumer of `GET /api/runtime/events/v1`.
 *
 * It follows `@/features/logs/api/use-logs-live-recovery`'s recovery shape —
 * connect, accept typed frames, hydrate authoritatively on a gap, degrade to
 * a documented fallback — but drives the socket with `fetch` rather than
 * `EventSource`, because the frozen contract carries the cursor as
 * `?cursor=` on connect and `EventSource` cannot set it.
 *
 * The hook is a LIVENESS SIGNAL, not a data source: it never decides
 * readiness and never projects runtime domain state. Consumers re-read their
 * own authoritative snapshot when `revision` advances. Startup is never
 * blocked on v1 — the capability probe runs beside the consumer's first
 * authoritative fetch, and every failure path resolves to `fallback` so the
 * caller can run its legacy stream instead.
 */

import { useEffect, useRef, useState } from 'react'

import { env } from '@/lib/env'
import {
  RuntimeEventsV1FrameError,
  parseRuntimeEventsV1Frame,
  parseSseBlock,
  splitSseBlocks
} from '@/features/network/api/runtime-events-v1-sse'

/** Kept below the server's frozen 10-connects-per-60s per-client limit. */
const MAX_RECONNECT_ATTEMPTS = 5
const RECONNECT_DELAY_MS = 1_000
/** A stream must stay live this long before it clears fast-failure history. */
export const STREAM_LIVENESS_MS = 60_000
const CAPABILITY_VERSION = 1
const CAPABILITY_CURSOR_PREFIX = 'rt1'

export type RuntimeEventsV1Mode = 'probing' | 'live' | 'fallback'

export type RuntimeEventsV1FallbackReason =
  | 'capability_absent'
  | 'capability_unsupported'
  | 'not_found'
  | 'forbidden'
  | 'malformed_frame'
  | 'reconnect_exhausted'
  | 'unsupported_environment'

export type RuntimeEventsResponse = {
  readonly ok: boolean
  readonly status: number
  readonly body: ReadableStream<Uint8Array> | null
  json(): Promise<unknown>
}

export type RuntimeEventsFetch = (url: string, init: { signal: AbortSignal }) => Promise<RuntimeEventsResponse>

export type RuntimeEventsV1Options = {
  readonly enabled: boolean
  readonly fetchImpl?: RuntimeEventsFetch
}

export type RuntimeEventsV1 = {
  readonly mode: RuntimeEventsV1Mode
  readonly fallbackReason: RuntimeEventsV1FallbackReason | null
  /** Advances once per accepted frame. Consumers re-hydrate when it moves. */
  readonly revision: number
  readonly cursor: string | null
  readonly processInstanceId: string | null
  readonly rebuildGeneration: number | null
}

const IDLE: RuntimeEventsV1 = {
  mode: 'probing',
  fallbackReason: null,
  revision: 0,
  cursor: null,
  processInstanceId: null,
  rebuildGeneration: null
}

type Capability = { readonly endpoint: string }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Read the exact advertised object. An absent capability and an
 * unreadable/failed status probe are both `capability_absent` — the console
 * falls back rather than treating v1 discovery as an error.
 */
function readCapability(status: unknown): Capability | RuntimeEventsV1FallbackReason {
  const runtime = isRecord(status) ? status.runtime : undefined
  const capabilities = isRecord(runtime) ? runtime.capabilities : undefined
  const advertised = isRecord(capabilities) ? capabilities.runtime_events : undefined
  if (!isRecord(advertised)) return 'capability_absent'
  if (advertised.version !== CAPABILITY_VERSION) return 'capability_unsupported'
  if (advertised.cursor !== CAPABILITY_CURSOR_PREFIX) return 'capability_unsupported'
  if (typeof advertised.endpoint !== 'string' || advertised.endpoint.length === 0) return 'capability_unsupported'
  return { endpoint: advertised.endpoint }
}

function defaultFetch(): RuntimeEventsFetch | undefined {
  if (typeof fetch === 'undefined') return undefined
  // `fetch` structurally satisfies this narrower shape; the cast keeps the
  // hook's surface free of the DOM `RequestInit` union.
  return fetch as unknown as RuntimeEventsFetch
}

function delay(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve) => {
    if (signal.aborted) {
      resolve()
      return
    }

    const timer = { id: 0 }
    const cleanup = (onAbort: () => void) => {
      window.clearTimeout(timer.id)
      signal.removeEventListener('abort', onAbort)
    }
    const onAbort = () => {
      cleanup(onAbort)
      resolve()
    }
    timer.id = window.setTimeout(() => {
      cleanup(onAbort)
      resolve()
    }, ms)
    signal.addEventListener('abort', onAbort, { once: true })
    // Abort can be requested between the initial check and listener
    // registration by an embedder's synchronous signal implementation.
    if (signal.aborted) {
      onAbort()
    }
  })
}

/** One connection attempt: `fatal` has already set a fallback reason. */
type Attempt = 'transient' | 'sustained' | 'fatal'

function monotonicNow(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now()
  }
  return Date.now()
}

export function useRuntimeEventsV1({ enabled, fetchImpl }: RuntimeEventsV1Options): RuntimeEventsV1 {
  const [state, setState] = useState<RuntimeEventsV1>(IDLE)
  const cursorRef = useRef<string | null>(null)

  useEffect(() => {
    if (!enabled) return undefined

    const request = new AbortController()
    const connect = fetchImpl ?? defaultFetch()
    let disposed = false

    const fallback = (reason: RuntimeEventsV1FallbackReason) => {
      if (disposed) return
      setState((current) => ({ ...current, mode: 'fallback', fallbackReason: reason }))
    }

    const accept = (chunk: { event: string; lastEventId: string; data: string }) => {
      const frame = parseRuntimeEventsV1Frame(chunk)
      cursorRef.current = frame.envelope.cursor.toString()
      if (disposed) return
      setState((current) => ({
        mode: 'live',
        fallbackReason: null,
        revision: current.revision + 1,
        cursor: frame.envelope.cursor.toString(),
        processInstanceId: frame.envelope.processInstanceId,
        rebuildGeneration: frame.envelope.rebuildGeneration
      }))
    }

    const readStream = async (body: ReadableStream<Uint8Array>): Promise<Attempt> => {
      const reader = body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      const connectedAt = monotonicNow()
      let receivedBodyActivity = false
      const connectionOutcome = (): Attempt =>
        receivedBodyActivity && monotonicNow() - connectedAt >= STREAM_LIVENESS_MS ? 'sustained' : 'transient'
      try {
        for (;;) {
          const { done, value } = await reader.read()
          if (done || disposed) return connectionOutcome()
          if (value && value.byteLength > 0) receivedBodyActivity = true
          buffer += decoder.decode(value, { stream: true })
          const split = splitSseBlocks(buffer)
          buffer = split.rest
          for (const block of split.blocks) {
            const chunk = parseSseBlock(block)
            if (!chunk) continue
            accept(chunk)
          }
        }
      } catch (error) {
        if (error instanceof RuntimeEventsV1FrameError) {
          fallback('malformed_frame')
          return 'fatal'
        }
        return connectionOutcome()
      } finally {
        reader.cancel().catch(() => undefined)
      }
    }

    const attempt = async (endpoint: string): Promise<Attempt> => {
      const cursor = cursorRef.current
      const query = cursor === null ? '' : `?cursor=${encodeURIComponent(cursor)}`
      let response: RuntimeEventsResponse
      try {
        response = await connect!(`${env.managementApiUrl}${endpoint}${query}`, { signal: request.signal })
      } catch {
        return 'transient'
      }
      if (response.status === 404) {
        fallback('not_found')
        return 'fatal'
      }
      if (response.status === 403) {
        fallback('forbidden')
        return 'fatal'
      }
      if (!response.ok || !response.body) return 'transient'
      return readStream(response.body)
    }

    const probe = async (): Promise<Capability | undefined> => {
      try {
        const response = await connect!(`${env.managementApiUrl}/api/status`, { signal: request.signal })
        if (!response.ok) {
          fallback('capability_absent')
          return undefined
        }
        const capability = readCapability(await response.json())
        if (typeof capability === 'string') {
          fallback(capability)
          return undefined
        }
        return capability
      } catch {
        fallback('capability_absent')
        return undefined
      }
    }

    const run = async () => {
      if (!connect) {
        fallback('unsupported_environment')
        return
      }
      const capability = await probe()
      if (!capability || disposed) return

      let failures = 0
      while (!disposed) {
        const outcome = await attempt(capability.endpoint)
        if (outcome === 'fatal' || disposed) return
        if (outcome === 'sustained') {
          failures = 0
        } else {
          failures += 1
        }
        if (failures >= MAX_RECONNECT_ATTEMPTS) {
          fallback('reconnect_exhausted')
          return
        }
        await delay(RECONNECT_DELAY_MS, request.signal)
      }
    }

    void run()

    return () => {
      disposed = true
      request.abort()
      cursorRef.current = null
      setState(IDLE)
    }
  }, [enabled, fetchImpl])

  return enabled ? state : IDLE
}
