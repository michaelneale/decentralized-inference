// @vitest-environment jsdom

import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useLlamaRuntime } from '@/features/network/api/use-llama-runtime'

const INSTANCE = '0195f000-0000-7000-8000-000000000001'
const ADVERTISED = { version: 1, endpoint: '/api/runtime/events/v1', cursor: 'rt1' }

type FakeEventSource = {
  url: string
  closed: boolean
  onopen: ((event: Event) => void) | null
  onmessage: ((event: MessageEvent<string>) => void) | null
  onerror: ((event: Event) => void) | null
  close: () => void
}

let eventSources: FakeEventSource[] = []
let streamController: ReadableStreamDefaultController<Uint8Array> | undefined

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

function installFakeEventSource() {
  eventSources = []
  vi.stubGlobal(
    'EventSource',
    class {
      url: string
      closed = false
      onopen: ((event: Event) => void) | null = null
      onmessage: ((event: MessageEvent<string>) => void) | null = null
      onerror: ((event: Event) => void) | null = null
      constructor(url: string) {
        this.url = url
        eventSources.push(this as unknown as FakeEventSource)
      }
      close() {
        this.closed = true
      }
    }
  )
}

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

function sseStreamResponse() {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      streamController = controller
    }
  })
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
}

function runtimeLlamaPayload(status: string) {
  return { metrics: { status }, slots: { status } }
}

function encodeFrame(event: string, sequence: number, body: Record<string, unknown>) {
  const payload = {
    version: 1,
    cursor: `rt1:${INSTANCE}:${sequence}`,
    process_instance_id: INSTANCE,
    sequence,
    rebuild_generation: 0,
    ...body
  }
  return `id: rt1:${INSTANCE}:${sequence}\nevent: ${event}\ndata: ${JSON.stringify(payload)}\n\n`
}

/** Routes every URL the hook touches, with a per-test v1 capability answer. */
function installFetch(capability: unknown, llamaStatuses: string[]) {
  let llamaCalls = 0
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input.toString()
    if (url.includes('/api/status')) {
      return jsonResponse({ runtime: { capabilities: { worker_capable: true, runtime_events: capability } } })
    }
    if (url.includes('/api/runtime/events/v1')) return sseStreamResponse()
    if (url.includes('/api/runtime/llama')) {
      const status = llamaStatuses[Math.min(llamaCalls, llamaStatuses.length - 1)] ?? 'ready'
      llamaCalls += 1
      return jsonResponse(runtimeLlamaPayload(status))
    }
    throw new Error(`unexpected fetch: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  return { fetchMock, llamaCallCount: () => llamaCalls }
}

async function settle() {
  await act(async () => {
    for (let index = 0; index < 6; index += 1) await Promise.resolve()
  })
}

describe('useLlamaRuntime (network)', () => {
  beforeEach(() => {
    streamController = undefined
    installFakeEventSource()
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  })

  it('consumes the advertised v1 stream instead of the legacy runtime EventSource', async () => {
    const { fetchMock } = installFetch(ADVERTISED, ['ready'])

    renderHook(() => useLlamaRuntime(true))
    await settle()

    expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/api/runtime/events/v1'))).toBe(true)
    expect(eventSources).toHaveLength(0)
  })

  it('re-hydrates the authoritative runtime snapshot when a v1 frame arrives', async () => {
    const { llamaCallCount } = installFetch(ADVERTISED, ['loading', 'ready'])

    const { result } = renderHook(() => useLlamaRuntime(true))
    await settle()
    const before = llamaCallCount()

    await act(async () => {
      streamController?.enqueue(new TextEncoder().encode(encodeFrame('runtime_health', 1, { health: {} })))
      for (let index = 0; index < 8; index += 1) await Promise.resolve()
    })

    await waitFor(() => expect(llamaCallCount()).toBeGreaterThan(before))
    await waitFor(() => expect(result.current.data?.metrics.status).toBe('ready'))
  })

  it('falls back to the legacy runtime EventSource when v1 is not advertised', async () => {
    installFetch(undefined, ['ready'])

    renderHook(() => useLlamaRuntime(true))
    await settle()

    await waitFor(() => expect(eventSources).toHaveLength(1))
    expect(eventSources[0]?.url).toContain('/api/runtime/events')
    expect(eventSources[0]?.url).not.toContain('/v1')
  })

  it('falls back to the legacy path when the v1 stream answers 404', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString()
      if (url.includes('/api/status')) {
        return jsonResponse({ runtime: { capabilities: { runtime_events: ADVERTISED } } })
      }
      if (url.includes('/api/runtime/events/v1')) return jsonResponse({}, 404)
      return jsonResponse(runtimeLlamaPayload('ready'))
    })
    vi.stubGlobal('fetch', fetchMock)

    renderHook(() => useLlamaRuntime(true))
    await settle()

    await waitFor(() => expect(eventSources).toHaveLength(1))
    expect(eventSources[0]?.url).toContain('/api/runtime/events')
  })

  it('closes the legacy stream and stops polling on teardown', async () => {
    installFetch(undefined, ['ready'])

    const { unmount } = renderHook(() => useLlamaRuntime(true))
    await settle()
    await waitFor(() => expect(eventSources).toHaveLength(1))

    unmount()
    expect(eventSources[0]?.closed).toBe(true)
  })
})
