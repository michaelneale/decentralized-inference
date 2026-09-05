import { describe, expect, it } from 'vitest'

import { parseLogsSseFrame, serializeLogsSseSubscription } from '@/features/logs/api/sse'
import { LogReplayCursor } from '@/features/logs/api/ids'
import { LogsDtoError } from '@/features/logs/api/schemas'

// Regression guard for the pure, transport-free frame parser that
// `use-logs-live-recovery` and the runtime-events v1 consumer both build on:
// a frame is keyed by its SSE `event` name, its cursor comes from the SSE id,
// and payload/cursor coherence is a mandatory-field check rather than a
// tolerated mismatch.

const EVENT_ID = '0195f000-0000-7000-8000-00000000000a'
const REQUEST_ID = '0195f000-0000-7000-8000-00000000000b'

function replayEvent(sequence: number) {
  return {
    eventId: EVENT_ID,
    requestId: REQUEST_ID,
    occurredAt: '2026-09-02T12:00:00.000Z',
    channel: 'requests',
    sequence,
    kind: 'admitted'
  }
}

describe('parseLogsSseFrame', () => {
  it('keys the frame on the SSE event name and takes its cursor from the SSE id', () => {
    const frame = parseLogsSseFrame({
      event: 'log_event',
      lastEventId: 'v1:7.0.0',
      data: JSON.stringify(replayEvent(7))
    })

    expect(frame.type).toBe('log_event')
    expect(frame.cursor.toString()).toBe('v1:7.0.0')
    expect(frame.cursor).toBeInstanceOf(LogReplayCursor)
  })

  it('rejects a frame whose payload sequence disagrees with its cursor', () => {
    expect(() =>
      parseLogsSseFrame({ event: 'log_event', lastEventId: 'v1:7.0.0', data: JSON.stringify(replayEvent(8)) })
    ).toThrow(LogsDtoError)
  })

  it('rejects an unknown event name rather than silently dropping the frame', () => {
    expect(() => parseLogsSseFrame({ event: 'runtime_state', lastEventId: 'v1:1.0.0', data: '{}' })).toThrow(
      LogsDtoError
    )
  })

  it('rejects a non-JSON data field', () => {
    expect(() => parseLogsSseFrame({ event: 'log_event', lastEventId: 'v1:1.0.0', data: 'not-json' })).toThrow(
      LogsDtoError
    )
  })

  it('rejects a noncanonical cursor in the SSE id', () => {
    expect(() =>
      parseLogsSseFrame({ event: 'log_event', lastEventId: 'v1:oops', data: JSON.stringify(replayEvent(1)) })
    ).toThrow()
  })

  it('parses a replay gap and carries its recovery cursor', () => {
    const frame = parseLogsSseFrame({
      event: 'replay_gap',
      lastEventId: 'v1:9.0.0',
      data: JSON.stringify({
        channel: 'requests',
        fromSequence: 4,
        toSequence: 9,
        recovery: { endpoint: '/api/logs/requests', cursor: null }
      })
    })

    if (frame.type !== 'replay_gap') throw new Error('expected a replay gap frame')
    expect(frame.gap.fromSequence).toBe(4)
    expect(frame.gap.recovery.cursor).toBeUndefined()
  })

  it('serializes a cursor onto the subscription query string', () => {
    const query = serializeLogsSseSubscription({
      channels: ['requests'],
      cursor: LogReplayCursor.parse('v1:3.0.0')
    })

    expect(new URLSearchParams(query).get('cursor')).toBe('v1:3.0.0')
  })
})
