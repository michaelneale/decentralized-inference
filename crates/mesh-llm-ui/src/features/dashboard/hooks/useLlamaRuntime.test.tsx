// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest'

import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useLlamaRuntime } from '@/features/dashboard/hooks/useLlamaRuntime'

const INSTANCE = '0195f000-0000-7000-8000-000000000001'
const ADVERTISED = { version: 1, endpoint: '/api/runtime/events/v1', cursor: 'rt1' }

type FakeEventSource = { url: string; closed: boolean }

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

describe('useLlamaRuntime', () => {
  beforeEach(() => {
    streamController = undefined
    installFakeEventSource()
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  })

  it('does not surface expected aborts from overlapping polls', async () => {
    vi.useFakeTimers()
    const runtimeRequests: DeferredFetch[] = []
    const fetchMock = vi.fn((input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
      const deferred = createDeferredFetch(init?.signal)
      if (String(input).includes('/api/runtime/llama')) runtimeRequests.push(deferred)
      return deferred.promise
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<RuntimeProbe />)
    expect(screen.getByTestId('loading')).toHaveTextContent('loading')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2_500)
    })

    expect(runtimeRequests).toHaveLength(2)
    await act(async () => {
      runtimeRequests[1].resolve(jsonResponse({ metrics: { status: 'ready' }, slots: { status: 'ready' } }))
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(screen.getByTestId('loading')).toHaveTextContent('idle')
    expect(screen.getByTestId('error')).toHaveTextContent('none')
  })

  it('consumes the advertised v1 stream instead of the legacy runtime EventSource', async () => {
    const fetchMock = installRoutedFetch(ADVERTISED)

    render(<RuntimeProbe />)
    await settle()

    expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/api/runtime/events/v1'))).toBe(true)
    expect(eventSources).toHaveLength(0)
  })

  it('re-hydrates the authoritative runtime snapshot when a v1 frame arrives', async () => {
    installRoutedFetch(ADVERTISED, ['loading', 'ready'])

    render(<RuntimeProbe />)
    await settle()
    expect(screen.getByTestId('metrics-status')).toHaveTextContent('loading')

    await act(async () => {
      streamController?.enqueue(new TextEncoder().encode(healthFrame(1)))
      for (let index = 0; index < 8; index += 1) await Promise.resolve()
    })

    await waitFor(() => expect(screen.getByTestId('metrics-status')).toHaveTextContent('ready'))
  })

  it('falls back to the legacy runtime EventSource when v1 is not advertised', async () => {
    installRoutedFetch(undefined)

    render(<RuntimeProbe />)
    await settle()

    await waitFor(() => expect(eventSources).toHaveLength(1))
    expect(eventSources[0]?.url).toBe('/api/runtime/events')
  })

  it('falls back to the legacy runtime EventSource when the v1 stream is forbidden', async () => {
    installRoutedFetch(ADVERTISED, ['ready'], 403)

    render(<RuntimeProbe />)
    await settle()

    await waitFor(() => expect(eventSources).toHaveLength(1))
    expect(eventSources[0]?.url).toBe('/api/runtime/events')
  })

  it('closes the legacy stream on teardown', async () => {
    installRoutedFetch(undefined)

    const view = render(<RuntimeProbe />)
    await settle()
    await waitFor(() => expect(eventSources).toHaveLength(1))

    view.unmount()
    expect(eventSources[0]?.closed).toBe(true)
  })
})

function RuntimeProbe() {
  const runtime = useLlamaRuntime(true)
  return (
    <div>
      <div data-testid="loading">{runtime.loading ? 'loading' : 'idle'}</div>
      <div data-testid="error">{runtime.error ?? 'none'}</div>
      <div data-testid="metrics-status">{runtime.data?.metrics.status ?? 'none'}</div>
    </div>
  )
}

type DeferredFetch = {
  promise: Promise<Response>
  resolve: (response: Response) => void
  reject: (error: Error) => void
}

function createDeferredFetch(signal: AbortSignal | null | undefined): DeferredFetch {
  let resolve: (response: Response) => void = () => undefined
  let reject: (error: Error) => void = () => undefined
  const promise = new Promise<Response>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  signal?.addEventListener(
    'abort',
    () => {
      reject(new DOMException('The operation was aborted.', 'AbortError'))
    },
    { once: true }
  )
  return { promise, resolve, reject }
}

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

function healthFrame(sequence: number) {
  const payload = {
    version: 1,
    cursor: `rt1:${INSTANCE}:${sequence}`,
    process_instance_id: INSTANCE,
    sequence,
    rebuild_generation: 0,
    health: {}
  }
  return `id: rt1:${INSTANCE}:${sequence}\nevent: runtime_health\ndata: ${JSON.stringify(payload)}\n\n`
}

function installRoutedFetch(capability: unknown, llamaStatuses: string[] = ['ready'], streamStatus = 200) {
  let llamaCalls = 0
  const fetchMock = vi.fn(async (input: Parameters<typeof fetch>[0]) => {
    const url = String(input)
    if (url.includes('/api/status')) {
      return jsonResponse({ runtime: { capabilities: { runtime_events: capability } } })
    }
    if (url.includes('/api/runtime/events/v1')) {
      if (streamStatus !== 200) return jsonResponse({}, streamStatus)
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          streamController = controller
        }
      })
      return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    }
    const status = llamaStatuses[Math.min(llamaCalls, llamaStatuses.length - 1)] ?? 'ready'
    llamaCalls += 1
    return jsonResponse({ metrics: { status }, slots: { status } })
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

async function settle() {
  await act(async () => {
    for (let index = 0; index < 8; index += 1) await Promise.resolve()
  })
}
