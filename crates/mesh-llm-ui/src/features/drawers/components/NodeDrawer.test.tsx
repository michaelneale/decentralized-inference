import { act, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { NodeDrawer } from '@/features/drawers/components/NodeDrawer'
import type { LlamaRuntimePayload } from '@/lib/api/types'
import type { MeshNode, Peer } from '@/features/app-tabs/types'
import { env } from '@/lib/env'

type MockEventSourceInstance = {
  url: string
  onopen: ((event: Event) => void) | null
  onmessage: ((event: MessageEvent<string>) => void) | null
  onerror: ((event: Event) => void) | null
  close: () => void
  emitOpen: () => void
  emitMessage: (payload: LlamaRuntimePayload) => void
  emitError: () => void
  closed: boolean
}

const SELF_NODE: MeshNode = {
  id: 'self',
  peerId: 'peer-self',
  label: 'CARRACK',
  x: 0,
  y: 0,
  status: 'online',
  role: 'self'
}

const SELF_PEER: Peer = {
  id: 'peer-self',
  hostname: 'carrack',
  region: 'tor-1',
  status: 'online',
  hostedModels: ['llama-local'],
  sharePct: 38,
  latencyMs: 1,
  loadPct: 12,
  shortId: '990232e1',
  role: 'you',
  version: '0.64.0',
  vramGB: 64,
  hardwareLabel: 'Jetson AGX Orin · 64 GB',
  ownership: 'Unsigned',
  owner: 'Unsigned'
}

const PEER_NODE: MeshNode = {
  id: 'peer',
  peerId: 'peer-2',
  label: 'LEMONY-28',
  x: 0,
  y: 0,
  status: 'online',
  role: 'peer'
}

const PEER: Peer = {
  id: 'peer-2',
  hostname: 'lemony-28',
  region: 'nyc-2',
  status: 'online',
  hostedModels: ['llama-remote'],
  sharePct: 31,
  latencyMs: 2,
  loadPct: 16,
  shortId: 'e5c42cc0',
  role: 'host',
  version: '0.64.0',
  vramGB: 48,
  hardwareLabel: 'Mac Studio M2 Ultra · 64 GB',
  ownership: 'Unsigned',
  owner: 'Unsigned'
}

const CLIENT_NODE: MeshNode = {
  id: 'client-node',
  peerId: 'peer-client',
  label: 'CLIENT-1',
  x: 0,
  y: 0,
  status: 'online',
  role: 'peer',
  renderKind: 'client',
  meshState: 'client',
  client: true
}

const CLIENT_PEER: Peer = {
  id: 'peer-client',
  hostname: 'client-1',
  region: 'remote',
  status: 'online',
  hostedModels: [],
  sharePct: 0,
  latencyMs: 0.7,
  loadPct: 0,
  shortId: '65482bce',
  role: 'client',
  nodeState: 'client',
  vramGB: 0,
  ownership: 'Unsigned',
  owner: 'Unsigned'
}

const WORKER_NODE: MeshNode = {
  id: 'worker-node',
  peerId: 'peer-worker',
  label: 'WORKER-1',
  x: 0,
  y: 0,
  status: 'online',
  role: 'peer',
  renderKind: 'worker',
  meshState: 'standby'
}

const WORKER_PEER: Peer = {
  id: 'peer-worker',
  hostname: 'worker-1',
  region: 'remote',
  status: 'online',
  hostedModels: [],
  sharePct: 0,
  latencyMs: 0.9,
  loadPct: 0,
  shortId: '9d20f6c1',
  role: 'worker',
  nodeState: 'standby',
  vramGB: 0,
  ownership: 'Unsigned',
  owner: 'Unsigned'
}

const INITIAL_RUNTIME: LlamaRuntimePayload = {
  metrics: {
    status: 'ready',
    samples: [{ name: 'llamacpp:requests_processing', value: 1 }]
  },
  slots: {
    status: 'ready',
    slots: [
      { index: 0, is_processing: true },
      { index: 1, is_processing: false }
    ]
  },
  items: {
    metrics: [{ name: 'llamacpp:requests_processing', value: 1 }],
    slots: [
      { index: 0, is_processing: true },
      { index: 1, is_processing: false }
    ],
    slots_total: 2,
    slots_busy: 1
  }
}

const UPDATED_RUNTIME: LlamaRuntimePayload = {
  metrics: {
    status: 'ready',
    samples: [{ name: 'llamacpp:requests_processing', value: 7 }]
  },
  slots: {
    status: 'ready',
    slots: [
      { index: 0, is_processing: true },
      { index: 1, is_processing: true },
      { index: 2, is_processing: false }
    ]
  },
  items: {
    metrics: [{ name: 'llamacpp:requests_processing', value: 7 }],
    slots: [
      { index: 0, is_processing: true },
      { index: 1, is_processing: true },
      { index: 2, is_processing: false }
    ],
    slots_total: 3,
    slots_busy: 2
  }
}

class MockEventSource implements MockEventSourceInstance {
  static instances: MockEventSource[] = []

  closed = false
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(public url: string) {
    MockEventSource.instances.push(this)
  }

  close() {
    this.closed = true
  }

  emitOpen() {
    this.onopen?.(new Event('open'))
  }

  emitMessage(payload: LlamaRuntimePayload) {
    this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent<string>)
  }

  emitError() {
    this.onerror?.(new Event('error'))
  }
}

// Fresh Response per call: a body is single-use and there are now two consumers.
function runtimeFetch(runtime: () => LlamaRuntimePayload) {
  return vi.fn(async (input: RequestInfo | URL) =>
    String(input).includes('/api/status')
      ? new Response('{}', { status: 200 })
      : new Response(JSON.stringify(runtime()), { status: 200 })
  )
}

describe('NodeDrawer runtime section', () => {
  beforeEach(() => {
    MockEventSource.instances = []
    vi.stubGlobal('EventSource', MockEventSource)
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('renders the initial runtime fetch in the self-node drawer', async () => {
    vi.stubGlobal(
      'fetch',
      runtimeFetch(() => INITIAL_RUNTIME)
    )

    render(<NodeDrawer open node={SELF_NODE} peer={SELF_PEER} onClose={vi.fn()} />)

    expect(screen.getByRole('heading', { name: 'Runtime' })).toBeInTheDocument()
    // Legacy stream opens only after the /api/status v1-capability probe resolves.
    await waitFor(() => expect(MockEventSource.instances[0]?.url).toBe(`${env.managementApiUrl}/api/runtime/events`))
    expect(await screen.findByText('Metrics • Live')).toBeInTheDocument()
    expect(screen.getByText('Slots • Live')).toBeInTheDocument()
    expect(screen.getByText('1/2 slots busy')).toBeInTheDocument()
    expect(screen.getByText('Metric')).toBeInTheDocument()
    expect(screen.getByText('Value')).toBeInTheDocument()
    expect(screen.getByText('requests processing')).toBeInTheDocument()
    expect(screen.getByText('1.00')).toBeInTheDocument()
    expect(screen.getByText('Slot context map')).toBeInTheDocument()
    expect(screen.getByText('Available')).toBeInTheDocument()
    expect(screen.getByText('Active')).toBeInTheDocument()
    expect(screen.getByRole('list', { name: 'Llama slot context map. 1 of 2 slots active.' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '#0 · Active · context n/a' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '#1 · Available · context n/a' })).toBeInTheDocument()
  })

  it('replaces the fetched runtime with SSE payloads', async () => {
    vi.stubGlobal(
      'fetch',
      runtimeFetch(() => INITIAL_RUNTIME)
    )

    render(<NodeDrawer open node={SELF_NODE} peer={SELF_PEER} onClose={vi.fn()} />)

    expect(await screen.findByText('1.00')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '#1 · Available · context n/a' })).toBeInTheDocument()
    await waitFor(() => expect(MockEventSource.instances).toHaveLength(1))

    await act(async () => {
      MockEventSource.instances[0]?.emitMessage(UPDATED_RUNTIME)
    })

    await waitFor(() => {
      expect(screen.getByText('2/3 slots busy')).toBeInTheDocument()
      expect(screen.getByText('7.00')).toBeInTheDocument()
      expect(screen.getByRole('list', { name: 'Llama slot context map. 2 of 3 slots active.' })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: '#1 · Active · context n/a' })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: '#2 · Available · context n/a' })).toBeInTheDocument()
    })
  })

  it('falls back to polling when the runtime stream fails', async () => {
    let servedRuntime: LlamaRuntimePayload = INITIAL_RUNTIME
    vi.stubGlobal(
      'fetch',
      runtimeFetch(() => servedRuntime)
    )

    render(<NodeDrawer open node={SELF_NODE} peer={SELF_PEER} onClose={vi.fn()} />)

    expect(await screen.findByText('1.00')).toBeInTheDocument()
    await waitFor(() => expect(MockEventSource.instances).toHaveLength(1))
    servedRuntime = UPDATED_RUNTIME

    vi.useFakeTimers()

    await act(async () => {
      MockEventSource.instances[0]?.emitError()
      await vi.advanceTimersByTimeAsync(2_500)
    })

    expect(screen.getByText('2/3 slots busy')).toBeInTheDocument()
    expect(screen.getByText('7.00')).toBeInTheDocument()
    expect(screen.getByRole('list', { name: 'Llama slot context map. 2 of 3 slots active.' })).toBeInTheDocument()
  })

  it('shows a local unavailable state when runtime fetch and polling both fail', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('runtime fetch failed')))

    render(<NodeDrawer open node={SELF_NODE} peer={SELF_PEER} onClose={vi.fn()} />)

    expect(await screen.findByText('Runtime unavailable: runtime fetch failed')).toBeInTheDocument()

    vi.useFakeTimers()

    await act(async () => {
      MockEventSource.instances[0]?.emitError()
      vi.advanceTimersByTime(2_500)
      await Promise.resolve()
    })

    expect(screen.getByText('Runtime unavailable: runtime fetch failed')).toBeInTheDocument()
    expect(screen.queryByText('requests processing')).not.toBeInTheDocument()
    expect(screen.queryByText('Slot context map')).not.toBeInTheDocument()
  })

  it('keeps runtime hidden for peer nodes', () => {
    vi.stubGlobal('fetch', vi.fn())

    render(<NodeDrawer open node={PEER_NODE} peer={PEER} onClose={vi.fn()} />)

    expect(screen.queryByRole('heading', { name: 'Runtime' })).not.toBeInTheDocument()
  })

  it('uses live mesh role and state labels for client and worker peers', () => {
    const { rerender } = render(<NodeDrawer open node={CLIENT_NODE} peer={CLIENT_PEER} onClose={vi.fn()} />)

    expect(screen.getByText('Client')).toBeInTheDocument()
    expect(screen.getByText('API-only')).toBeInTheDocument()
    expect(screen.queryByText('Host')).not.toBeInTheDocument()

    rerender(<NodeDrawer open node={WORKER_NODE} peer={WORKER_PEER} onClose={vi.fn()} />)

    expect(screen.getByText('Worker')).toBeInTheDocument()
    expect(screen.getByText('Standby')).toBeInTheDocument()
    expect(screen.queryByText('Client')).not.toBeInTheDocument()
  })
})
