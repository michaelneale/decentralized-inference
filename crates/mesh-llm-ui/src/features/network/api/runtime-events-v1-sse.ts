/**
 * Wire contract for `GET /api/runtime/events/v1`.
 *
 * Shaped after `@/features/logs/api/sse` — a pure, transport-free frame
 * parser plus a typed cursor — so the v1 hook reuses the logs parser
 * conventions instead of `EventSource`'s implicit framing. The constants
 * below mirror the Rust-generated normative corpus at
 * `crates/mesh-llm-runtime-event-contracts/fixtures/runtime_events_v1/`;
 * `use-runtime-events-v1.test.tsx` reads those fixture files directly and
 * asserts equality, so a server-side contract change fails the UI suite
 * rather than drifting silently. They are declared here rather than
 * imported because this module ships in the browser bundle.
 */

/** Frozen cursor grammar: `rt1:<lowercase-hyphenated-uuid>:<canonical-decimal>`. */
const CURSOR_PATTERN = /^rt1:([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}):(0|[1-9][0-9]*)$/
const MAX_SEQUENCE = 18_446_744_073_709_551_615n

export const RUNTIME_EVENT_NAMES = ['runtime_state', 'runtime_event', 'runtime_health', 'runtime_replay_gap'] as const

export const REQUIRED_ENVELOPE_KEYS = [
  'version',
  'cursor',
  'process_instance_id',
  'sequence',
  'rebuild_generation'
] as const

export const PER_EVENT_REQUIRED_KEYS = {
  runtime_state: ['state'],
  runtime_event: ['event'],
  runtime_health: ['health'],
  runtime_replay_gap: ['requested_cursor', 'reason']
} as const

export const STATE_TOP_LEVEL_KEYS = ['node', 'models', 'stages', 'sessions', 'requests', 'devices', 'cache'] as const

/** Deny-by-default at field level: a `runtime_event` may carry nothing else. */
export const EVENT_PROJECTED_KEY_ALLOWLIST = [
  'category',
  'kind',
  'producer',
  'severity',
  'wall_clock_unix_ns',
  'process_monotonic_ns',
  'native_monotonic_ns',
  'native_sequence',
  'summary',
  'scope',
  'state',
  'progress',
  'outcome',
  'reason_code',
  'duration_ms',
  'numeric_summaries',
  'operation'
] as const

export const REPLAY_GAP_REASONS = ['stale_instance', 'evicted'] as const

export const KEEPALIVE_FRAME = ': keepalive\n\n'

export type RuntimeEventName = (typeof RUNTIME_EVENT_NAMES)[number]
export type ReplayGapReason = (typeof REPLAY_GAP_REASONS)[number]

/** A mandatory-field violation. Callers treat this as a fallback trigger. */
export class RuntimeEventsV1FrameError extends Error {
  constructor(detail: string) {
    super(`runtime events v1 frame is invalid: ${detail}`)
    this.name = 'RuntimeEventsV1FrameError'
  }
}

/** A parsed `rt1:` cursor that has crossed the v1 stream boundary. */
export class RuntimeEventCursor {
  readonly processInstanceId: string
  readonly sequence: bigint

  private constructor(processInstanceId: string, sequence: bigint) {
    this.processInstanceId = processInstanceId
    this.sequence = sequence
  }

  static parse(value: string): RuntimeEventCursor {
    const cursor = RuntimeEventCursor.tryParse(value)
    if (!cursor) throw new RuntimeEventsV1FrameError(`noncanonical cursor ${JSON.stringify(value)}`)
    return cursor
  }

  static tryParse(value: string): RuntimeEventCursor | undefined {
    const match = CURSOR_PATTERN.exec(value)
    if (!match) return undefined
    const sequence = BigInt(match[2])
    if (sequence > MAX_SEQUENCE) return undefined
    return new RuntimeEventCursor(match[1], sequence)
  }

  toString() {
    return `rt1:${this.processInstanceId}:${this.sequence}`
  }
}

export type RuntimeEventEnvelope = {
  readonly cursor: RuntimeEventCursor
  readonly processInstanceId: string
  readonly sequence: bigint
  readonly rebuildGeneration: number
}

export type RuntimeEventsV1Frame =
  | { readonly type: 'runtime_state'; readonly envelope: RuntimeEventEnvelope; readonly state: Record<string, unknown> }
  | { readonly type: 'runtime_event'; readonly envelope: RuntimeEventEnvelope; readonly event: Record<string, unknown> }
  | {
      readonly type: 'runtime_health'
      readonly envelope: RuntimeEventEnvelope
      readonly health: Record<string, unknown>
    }
  | {
      readonly type: 'runtime_replay_gap'
      readonly envelope: RuntimeEventEnvelope
      readonly requestedCursor: RuntimeEventCursor
      readonly reason: ReplayGapReason
      readonly oldestAvailableCursor?: RuntimeEventCursor
      readonly latestCursor?: RuntimeEventCursor
    }

export type RuntimeEventsV1FrameInput = {
  readonly event: string
  readonly lastEventId: string
  readonly data: string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isRuntimeEventName(value: string): value is RuntimeEventName {
  return (RUNTIME_EVENT_NAMES as readonly string[]).includes(value)
}

function parsePayload(data: string): Record<string, unknown> {
  let payload: unknown
  try {
    payload = JSON.parse(data)
  } catch {
    throw new RuntimeEventsV1FrameError('data is not JSON')
  }
  if (!isRecord(payload)) throw new RuntimeEventsV1FrameError('data is not a JSON object')
  return payload
}

function requireRecord(payload: Record<string, unknown>, key: string): Record<string, unknown> {
  const value = payload[key]
  if (!isRecord(value)) throw new RuntimeEventsV1FrameError(`${key} is missing or not an object`)
  return value
}

/**
 * Sequences beyond `Number.MAX_SAFE_INTEGER` cannot round-trip through
 * `JSON.parse`, so the payload sequence is checked as a safe integer while
 * the authoritative value stays the cursor's `bigint`. A single process
 * would need 2^53 ingress sequences to reach that bound.
 */
function parseEnvelope(payload: Record<string, unknown>, lastEventId: string): RuntimeEventEnvelope {
  if (payload.version !== 1) throw new RuntimeEventsV1FrameError('version is missing or not 1')

  const id = RuntimeEventCursor.parse(lastEventId)
  const rawCursor = payload.cursor
  if (typeof rawCursor !== 'string') throw new RuntimeEventsV1FrameError('cursor is missing')
  const cursor = RuntimeEventCursor.parse(rawCursor)
  if (cursor.toString() !== id.toString()) {
    throw new RuntimeEventsV1FrameError('payload cursor disagrees with the SSE id')
  }

  if (payload.process_instance_id !== cursor.processInstanceId) {
    throw new RuntimeEventsV1FrameError('process_instance_id disagrees with the cursor')
  }

  const sequence = payload.sequence
  if (typeof sequence !== 'number' || !Number.isSafeInteger(sequence) || BigInt(sequence) !== cursor.sequence) {
    throw new RuntimeEventsV1FrameError('sequence is missing or disagrees with the cursor')
  }

  const rebuildGeneration = payload.rebuild_generation
  if (typeof rebuildGeneration !== 'number' || !Number.isSafeInteger(rebuildGeneration)) {
    throw new RuntimeEventsV1FrameError('rebuild_generation is missing')
  }

  return { cursor, processInstanceId: cursor.processInstanceId, sequence: cursor.sequence, rebuildGeneration }
}

function parseEventBody(payload: Record<string, unknown>): Record<string, unknown> {
  const event = requireRecord(payload, 'event')
  const allowed = new Set<string>(EVENT_PROJECTED_KEY_ALLOWLIST)
  const escaped = Object.keys(event).filter((key) => !allowed.has(key))
  if (escaped.length > 0) {
    throw new RuntimeEventsV1FrameError(`projected keys escape the allowlist: ${escaped.join(', ')}`)
  }
  return event
}

function parseStateBody(payload: Record<string, unknown>): Record<string, unknown> {
  const state = requireRecord(payload, 'state')
  const missing = STATE_TOP_LEVEL_KEYS.filter((key) => !(key in state))
  if (missing.length > 0) throw new RuntimeEventsV1FrameError(`state is missing ${missing.join(', ')}`)
  return state
}

function optionalCursor(payload: Record<string, unknown>, key: string): RuntimeEventCursor | undefined {
  const value = payload[key]
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'string') throw new RuntimeEventsV1FrameError(`${key} is not a cursor string`)
  return RuntimeEventCursor.parse(value)
}

function parseReplayGap(
  payload: Record<string, unknown>,
  envelope: RuntimeEventEnvelope
): Extract<RuntimeEventsV1Frame, { type: 'runtime_replay_gap' }> {
  const requested = payload.requested_cursor
  if (typeof requested !== 'string') throw new RuntimeEventsV1FrameError('requested_cursor is missing')
  const reason = payload.reason
  if (typeof reason !== 'string' || !(REPLAY_GAP_REASONS as readonly string[]).includes(reason)) {
    throw new RuntimeEventsV1FrameError(`reason ${JSON.stringify(reason)} is not a frozen replay-gap reason`)
  }
  return {
    type: 'runtime_replay_gap',
    envelope,
    requestedCursor: RuntimeEventCursor.parse(requested),
    reason: reason as ReplayGapReason,
    oldestAvailableCursor: optionalCursor(payload, 'oldest_available_cursor'),
    latestCursor: optionalCursor(payload, 'latest_cursor')
  }
}

/**
 * Parse one v1 data frame. Unknown additive fields are ignored everywhere;
 * every mandatory violation throws `RuntimeEventsV1FrameError`.
 */
export function parseRuntimeEventsV1Frame(input: RuntimeEventsV1FrameInput): RuntimeEventsV1Frame {
  if (!isRuntimeEventName(input.event)) {
    throw new RuntimeEventsV1FrameError(`unknown event name ${JSON.stringify(input.event)}`)
  }
  const payload = parsePayload(input.data)
  const envelope = parseEnvelope(payload, input.lastEventId)

  switch (input.event) {
    case 'runtime_state':
      return { type: 'runtime_state', envelope, state: parseStateBody(payload) }
    case 'runtime_event':
      return { type: 'runtime_event', envelope, event: parseEventBody(payload) }
    case 'runtime_health':
      return { type: 'runtime_health', envelope, health: requireRecord(payload, 'health') }
    case 'runtime_replay_gap':
      return parseReplayGap(payload, envelope)
  }
}

export type SseChunk = { readonly event: string; readonly lastEventId: string; readonly data: string }

/** Split a decode buffer into complete `\n\n`-terminated SSE blocks. */
export function splitSseBlocks(buffer: string): { readonly blocks: readonly string[]; readonly rest: string } {
  const parts = buffer.split('\n\n')
  return { blocks: parts.slice(0, -1), rest: parts.at(-1) ?? '' }
}

/**
 * Decode one SSE block into its `id`/`event`/`data` fields. Empty and comment
 * blocks (including the frozen keepalive) return `undefined`; any nonempty
 * block must contain all three fields or it is a malformed stream frame.
 */
export function parseSseBlock(block: string): SseChunk | undefined {
  let event: string | undefined
  let lastEventId: string | undefined
  const data: string[] = []
  let hasNonCommentContent = false

  for (const line of block.split('\n')) {
    if (line === '' || line.startsWith(':')) continue
    hasNonCommentContent = true
    const separator = line.indexOf(':')
    const field = separator === -1 ? line : line.slice(0, separator)
    const rawValue = separator === -1 ? '' : line.slice(separator + 1)
    const value = rawValue.startsWith(' ') ? rawValue.slice(1) : rawValue
    if (field === 'event') event = value
    else if (field === 'id') lastEventId = value
    else if (field === 'data') data.push(value)
  }

  if (!hasNonCommentContent) return undefined
  if (event === undefined || lastEventId === undefined || data.length === 0) {
    const missing = [
      event === undefined ? 'event' : undefined,
      lastEventId === undefined ? 'id' : undefined,
      data.length === 0 ? 'data' : undefined
    ].filter((field): field is string => field !== undefined)
    throw new RuntimeEventsV1FrameError(`SSE block is missing ${missing.join(', ')}`)
  }
  return { event, lastEventId, data: data.join('\n') }
}
