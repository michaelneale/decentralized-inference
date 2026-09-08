//! The live socket loop: subscribe, headers, connection-shape recovery
//! frames, then keepalive/health/event fan-out until the client disconnects
//! or its lag bound is exceeded.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::broadcast::error::RecvError;

use crate::api::management_lifecycle::record_response_status;
use crate::runtime_events::config::{KEEPALIVE_INTERVAL, TUI_RENDER_TICK};
use crate::runtime_events::engine::{RuntimeEventAttachment, RuntimeEventEngine};
use crate::runtime_events::health::HealthDeliveryGate;
use crate::runtime_events::replay::ReplayFrame;
use crate::runtime_events::subscribers::SubscriptionHandle;

use super::cursor::Cursor;
use super::frames::{self, KEEPALIVE_FRAME};
use super::recovery::ConnectionShape;

const WRITE_TIMEOUT: Duration = Duration::from_millis(250);
const SSE_HEADERS: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";

pub(super) async fn run(
    stream: &mut TcpStream,
    engine: &Arc<RuntimeEventEngine>,
    attachment: RuntimeEventAttachment,
    shape: ConnectionShape,
) -> anyhow::Result<()> {
    // `engine.attach()` registered the subscription and captured replay,
    // reducer state, health, and the applied publication frontier while the
    // engine's publication guard was held. No publication can fall between
    // this initial write set and the live queue.
    let RuntimeEventAttachment {
        mut subscription,
        reducer,
        health,
        ingress_p99_us,
        published_frontier,
        rebuild_generation,
        ..
    } = attachment;

    stream.write_all(SSE_HEADERS).await?;
    record_response_status(200);

    if let Some(last_delivered_sequence) = write_initial_frames(
        stream,
        engine,
        &mut subscription,
        &shape,
        InitialSnapshot {
            reducer: &reducer,
            health,
            ingress_p99_us,
            rebuild_generation,
            published_frontier,
        },
    )
    .await
    {
        // Task 8 (`.omo/plans/event-system-fixes.md`, defect D9): this
        // connection's own independent health-delivery gate, seeded from
        // the version `write_initial_frames` already sent above so the
        // live loop's first eligible check does not immediately re-deliver
        // the same snapshot (see `HealthDeliveryGate::seeded`).
        let health_gate = HealthDeliveryGate::seeded(health.version, Instant::now());
        live_loop(
            stream,
            engine,
            &mut subscription,
            health_gate,
            last_delivered_sequence,
        )
        .await;
    }
    Ok(())
}

struct InitialFrame {
    wire_bytes: Arc<[u8]>,
    recorded_at: Instant,
}

/// The immutable snapshot fields captured by `RuntimeEventEngine::attach`.
/// Keeping them together also makes it harder for an initial writer to pair
/// one captured field with a later engine read.
struct InitialSnapshot<'a> {
    reducer: &'a crate::runtime_events::reducer::ReducerSnapshot,
    health: crate::runtime_events::health::EngineHealthSnapshot,
    ingress_p99_us: Option<u64>,
    rebuild_generation: u64,
    published_frontier: u64,
}

/// Build and emit the frozen frame order for `shape`. The full initial write
/// set is reserved against this subscriber before the first socket write, so
/// replay/state/health bytes are included in exact lag accounting while the
/// connection is catching up. The returned cursor is the last runtime-event
/// sequence delivered to the live loop. A no-cursor or gap snapshot instead
/// establishes a coherent state checkpoint at the captured frontier; later
/// live health frames must retain that checkpoint rather than regress to the
/// empty-snapshot sentinel.
async fn write_initial_frames(
    stream: &mut TcpStream,
    engine: &Arc<RuntimeEventEngine>,
    subscription: &mut SubscriptionHandle,
    shape: &ConnectionShape,
    snapshot: InitialSnapshot<'_>,
) -> Option<u64> {
    let cursor = Cursor::new(engine.process_instance(), snapshot.published_frontier);
    // A reconnect that is already caught up has no replay frame to establish
    // a new delivery cursor; its requested frontier is the appropriate
    // starting point. A no-cursor or gap snapshot is itself a coherent state
    // checkpoint at the captured frontier, so live health keeps that cursor
    // instead of regressing to the empty-snapshot sentinel.
    let live_cursor = match shape {
        ConnectionShape::InWindow { frames } => frames
            .last()
            .map(|frame| frame.sequence.get())
            .unwrap_or(snapshot.published_frontier),
        ConnectionShape::NoCursor | ConnectionShape::Gap(_) => snapshot.published_frontier,
    };
    let now = Instant::now();
    let mut initial = Vec::new();
    match shape {
        ConnectionShape::NoCursor => {
            initial.push(InitialFrame {
                wire_bytes: Arc::from(
                    frames::state_frame_from_snapshot(snapshot.reducer, cursor).into_bytes(),
                ),
                recorded_at: now,
            });
            initial.push(InitialFrame {
                wire_bytes: Arc::from(
                    frames::health_frame_from_snapshot(
                        snapshot.health,
                        snapshot.ingress_p99_us,
                        cursor,
                    )
                    .into_bytes(),
                ),
                recorded_at: now,
            });
        }
        ConnectionShape::InWindow { frames: replay } => {
            for frame in replay {
                initial.push(InitialFrame {
                    wire_bytes: Arc::clone(&frame.wire_bytes),
                    // Replay retention age describes how long a frame may
                    // remain available for reconnect. Subscriber lag age
                    // starts when this connection begins its catch-up write;
                    // an old-but-still-retained frame must not disconnect a
                    // legitimate reconnect before it can be delivered.
                    recorded_at: now,
                });
            }
            initial.push(InitialFrame {
                wire_bytes: Arc::from(
                    frames::health_frame_from_snapshot(
                        snapshot.health,
                        snapshot.ingress_p99_us,
                        cursor,
                    )
                    .into_bytes(),
                ),
                recorded_at: now,
            });
        }
        ConnectionShape::Gap(gap) => {
            initial.push(InitialFrame {
                wire_bytes: Arc::from(
                    frames::replay_gap_frame_at(
                        engine.process_instance(),
                        snapshot.rebuild_generation,
                        gap,
                    )
                    .into_bytes(),
                ),
                recorded_at: now,
            });
            initial.push(InitialFrame {
                wire_bytes: Arc::from(
                    frames::state_frame_from_snapshot(snapshot.reducer, cursor).into_bytes(),
                ),
                recorded_at: now,
            });
            initial.push(InitialFrame {
                wire_bytes: Arc::from(
                    frames::health_frame_from_snapshot(
                        snapshot.health,
                        snapshot.ingress_p99_us,
                        cursor,
                    )
                    .into_bytes(),
                ),
                recorded_at: now,
            });
        }
    }

    let bytes = initial
        .iter()
        .map(|frame| frame.wire_bytes.len())
        .sum::<usize>();
    let oldest = initial
        .iter()
        .map(|frame| frame.recorded_at)
        .min()
        .unwrap_or(now);
    if !subscription.reserve_pending(initial.len(), bytes, oldest) {
        subscription.record_disconnect(engine.health());
        return None;
    }
    for frame in initial {
        let byte_len = frame.wire_bytes.len();
        let wrote = write_frame(stream, frame.wire_bytes).await;
        subscription.complete_pending(byte_len);
        if !wrote {
            return None;
        }
        // A large replay batch can spend longer than the age bound in socket
        // writes even though it was fresh when reserved. Re-check after each
        // write so the first exceeded bound closes the connection before the
        // remainder is sent.
        if subscription.lag_bound_exceeded(Instant::now()) {
            subscription.record_disconnect(engine.health());
            return None;
        }
    }
    Some(live_cursor)
}

/// Task 8 (`.omo/plans/event-system-fixes.md`, defect D9): `health_gate` is
/// checked on every health-check tick, every keepalive tick, AND after every
/// delivered event frame —
/// not gated by the removed engine-global `publish_at` cadence, which let a
/// busy OTHER subscriber's own tick consume the one shared publish window
/// and starve this connection for up to ~50 minutes under load. This
/// connection owns its gate independently of every other connection.
///
/// Task 9 (`.omo/plans/event-system-fixes.md`, defect D11): the socket is
/// split into independent read/write halves (`TcpStream::split`) so a
/// THIRD `select!` branch can read-probe for client EOF/error concurrently
/// with the existing keepalive/recv write paths, without the two sides
/// contending over one `&mut TcpStream` borrow. An SSE client never
/// legitimately sends bytes on this connection, so ANY completed read on
/// that branch -- EOF, a read error, or unexpected client bytes -- means the
/// server-side subscription must end; unexpected bytes are invalid protocol
/// input but do not by themselves prove the peer is gone. This lets a closed
/// client free its subscriber slot immediately instead of only being
/// discovered on the next write attempt.
async fn live_loop(
    stream: &mut TcpStream,
    engine: &Arc<RuntimeEventEngine>,
    subscription: &mut SubscriptionHandle,
    mut health_gate: HealthDeliveryGate,
    mut last_delivered_sequence: u64,
) {
    let mut keepalive = tokio::time::interval(KEEPALIVE_INTERVAL);
    keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    keepalive.tick().await;
    let mut health_check = tokio::time::interval(TUI_RENDER_TICK);
    health_check.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    health_check.tick().await;

    let (mut read_half, mut write_half) = stream.split();
    let mut disconnect_probe = [0u8; 64];

    loop {
        tokio::select! {
            _ = keepalive.tick() => {
                if !write_keepalive_and_health(
                    &mut write_half,
                    engine,
                    subscription,
                    &mut health_gate,
                    last_delivered_sequence,
                ).await {
                    break;
                }
            }
            _ = health_check.tick() => {
                if !write_health_tick(
                    &mut write_half,
                    engine,
                    subscription,
                    &mut health_gate,
                    last_delivered_sequence,
                ).await {
                    break;
                }
            }
            received = subscription.recv() => match received {
                Ok(frame) => {
                    if !deliver_event_frame(
                        &mut write_half,
                        engine,
                        subscription,
                        frame,
                        &mut health_gate,
                        &mut last_delivered_sequence,
                    )
                    .await {
                        break;
                    }
                }
                Err(RecvError::Lagged(_)) => {
                    subscription.record_disconnect(engine.health());
                    break;
                }
                Err(RecvError::Closed) => break,
            },
            // An SSE client never legitimately sends bytes on this read-only
            // connection. Any completed read is therefore a protocol-close
            // signal, whether it is EOF, an I/O error, or invalid client
            // input; invalid input does not by itself prove the client is gone.
            _ = read_half.read(&mut disconnect_probe) => break,
        }
    }
}

async fn write_keepalive_and_health(
    stream: &mut (impl AsyncWrite + Unpin),
    engine: &Arc<RuntimeEventEngine>,
    subscription: &SubscriptionHandle,
    health_gate: &mut HealthDeliveryGate,
    last_delivered_sequence: u64,
) -> bool {
    if subscription.lag_bound_exceeded(Instant::now()) {
        subscription.record_disconnect(engine.health());
        return false;
    }
    if !write_live_frame(
        stream,
        subscription,
        KEEPALIVE_FRAME.as_bytes(),
        Instant::now(),
    )
    .await
    {
        if subscription.lag_bound_exceeded(Instant::now()) {
            subscription.record_disconnect(engine.health());
        }
        return false;
    }
    maybe_write_health(
        stream,
        engine,
        subscription,
        health_gate,
        Cursor::new(engine.process_instance(), last_delivered_sequence),
    )
    .await
}

async fn write_health_tick(
    stream: &mut (impl AsyncWrite + Unpin),
    engine: &Arc<RuntimeEventEngine>,
    subscription: &SubscriptionHandle,
    health_gate: &mut HealthDeliveryGate,
    last_delivered_sequence: u64,
) -> bool {
    if subscription.lag_bound_exceeded(Instant::now()) {
        subscription.record_disconnect(engine.health());
        return false;
    }
    maybe_write_health(
        stream,
        engine,
        subscription,
        health_gate,
        Cursor::new(engine.process_instance(), last_delivered_sequence),
    )
    .await
}

/// Enforce per-subscriber lag bounds, write one pre-serialized event frame,
/// then check this connection's independent health gate. The ordering matches
/// the live-loop receive arm: a lagged subscriber is recorded and dropped
/// before any frame write; a successful event write is followed by the health
/// check for that event's cursor.
async fn deliver_event_frame(
    stream: &mut (impl AsyncWrite + Unpin),
    engine: &Arc<RuntimeEventEngine>,
    subscription: &SubscriptionHandle,
    frame: ReplayFrame,
    health_gate: &mut HealthDeliveryGate,
    last_delivered_sequence: &mut u64,
) -> bool {
    if subscription.lag_bound_exceeded(Instant::now()) {
        subscription.record_disconnect(engine.health());
        return false;
    }

    let sequence = frame.sequence.get();
    if !write_live_frame(stream, subscription, &frame.wire_bytes, frame.recorded_at).await {
        if subscription.lag_bound_exceeded(Instant::now()) {
            subscription.record_disconnect(engine.health());
        }
        return false;
    }
    *last_delivered_sequence = sequence;
    let cursor = Cursor::new(engine.process_instance(), *last_delivered_sequence);
    maybe_write_health(stream, engine, subscription, health_gate, cursor).await
}

/// Deliver a `runtime_health` frame only when `health_gate` says this
/// connection's last-delivered version is stale. Returns `false` only on a
/// write failure, matching every other frame writer in this loop so the
/// caller's `break` handling stays uniform; a gate-suppressed check (nothing
/// to deliver) returns `true`.
async fn maybe_write_health(
    stream: &mut (impl AsyncWrite + Unpin),
    engine: &Arc<RuntimeEventEngine>,
    subscription: &SubscriptionHandle,
    health_gate: &mut HealthDeliveryGate,
    cursor: Cursor,
) -> bool {
    let snapshot = engine.health().snapshot();
    let ingress_p99_us = engine.ingress_p99_us();
    if !health_gate.should_deliver(&snapshot, Instant::now()) {
        return true;
    }
    let frame = Arc::<[u8]>::from(
        frames::health_frame_from_snapshot(snapshot, ingress_p99_us, cursor).into_bytes(),
    );
    let wrote = write_live_frame(stream, subscription, &frame, Instant::now()).await;
    if !wrote && subscription.lag_bound_exceeded(Instant::now()) {
        subscription.record_disconnect(engine.health());
    }
    wrote
}

/// Reserve one frame's exact bytes while its socket write is in flight. The
/// queue has already removed a received event, so this short pending period
/// closes the accounting gap between receive and completion of `write_all`.
async fn write_live_frame(
    stream: &mut (impl AsyncWrite + Unpin),
    subscription: &SubscriptionHandle,
    frame: impl AsRef<[u8]>,
    recorded_at: Instant,
) -> bool {
    let bytes = frame.as_ref();
    if !subscription.reserve_pending(1, bytes.len(), recorded_at) {
        return false;
    }
    let byte_len = bytes.len();
    let wrote = write_frame(stream, bytes).await;
    subscription.complete_pending(byte_len);
    wrote && !subscription.lag_bound_exceeded(Instant::now())
}

/// Generic over the writer (`&mut TcpStream` before `live_loop` splits the
/// socket; `&mut WriteHalf<'_>` once it does) and over the payload
/// (`String` for the JSON-frame writers; `Arc<[u8]>` for a `ReplayFrame`'s
/// pre-serialized `wire_bytes`, task 9,
/// `.omo/plans/event-system-fixes.md`) so every call site in this module
/// shares one write-with-timeout implementation.
async fn write_frame(stream: &mut (impl AsyncWrite + Unpin), frame: impl AsRef<[u8]>) -> bool {
    tokio::time::timeout(WRITE_TIMEOUT, stream.write_all(frame.as_ref()))
        .await
        .is_ok_and(|result| result.is_ok())
}

// Task 8-fix E5 (`.omo/plans/event-system-fixes.md`): `live_loop`'s recv
// arm calls `maybe_write_health` after every delivered event frame
// (`Ok(frame) =>` above) so a busy v1 subscriber's health delivery is
// never starved behind the 15s keepalive tick alone -- but
// `live_loop`/`maybe_write_health` had zero test callers anywhere in the
// repo before this. `live_loop` takes a real `&mut TcpStream`, so this
// drives it over a genuine loopback TCP pair rather than a mock writer.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use mesh_llm_runtime_event_contracts::{
        EventSequence, FamilyFact, NativeRuntimeEventKind, OperationId, OperationScope,
        RuntimeEventIngress, RuntimeFact, SubmitOutcome,
    };

    use crate::runtime_events::config::{HEALTH_PUBLISH_MIN_INTERVAL, SUBSCRIBER_LAG_MAX_AGE};

    use super::*;

    /// A real loopback TCP pair: `server` is what `live_loop` writes SSE
    /// frames into, exactly like the socket `run` hands it; `client` is
    /// read from directly by the test, exactly like a real subscriber.
    async fn loopback_pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback listener");
        let addr = listener.local_addr().expect("listener local addr");
        let client = TcpStream::connect(addr)
            .await
            .expect("connect loopback client");
        let (server, _) = listener.accept().await.expect("accept loopback server");
        (server, client)
    }

    /// Replay retention is deliberately longer than the per-subscriber lag
    /// bound. An in-window reconnect may therefore receive a frame recorded
    /// more than thirty seconds ago; subscriber age starts at this
    /// connection's catch-up enqueue time rather than at the historical
    /// publication time. This keeps a valid replay catch-up from being
    /// disconnected before its first write.
    #[tokio::test]
    async fn retained_replay_catch_up_starts_lag_age_at_attachment() {
        let engine = RuntimeEventEngine::new();
        let mut subscription = engine.subscribers().subscribe().expect("subscribe");
        let (mut server, _client) = loopback_pair().await;
        let old_recorded_at = Instant::now() - SUBSCRIBER_LAG_MAX_AGE - Duration::from_secs(1);
        let frame = ReplayFrame {
            sequence: EventSequence::new(1),
            rebuild_generation: 0,
            scope: OperationScope::root_only(OperationId::new()),
            fact: Arc::new(distinct_state_transition_fact()),
            recorded_at: old_recorded_at,
            wire_bytes: Arc::from(b"event: runtime_event\n\n".as_slice()),
        };
        let shape = ConnectionShape::InWindow {
            frames: vec![frame],
        };

        let wrote = write_initial_frames(
            &mut server,
            &engine,
            &mut subscription,
            &shape,
            InitialSnapshot {
                reducer: &engine.reducer_snapshot(),
                health: engine.health().snapshot(),
                ingress_p99_us: engine.ingress_p99_us(),
                rebuild_generation: 0,
                published_frontier: 1,
            },
        )
        .await;

        assert_eq!(
            wrote,
            Some(1),
            "a retained replay frame must survive initial catch-up"
        );
        assert_eq!(subscription.outstanding_bytes(), 0);
    }

    #[tokio::test]
    async fn snapshot_only_initial_shapes_preserve_the_checkpoint_for_live_health() {
        let engine = RuntimeEventEngine::new();
        let mut subscription = engine.subscribers().subscribe().expect("subscribe");
        let (mut server, mut client) = loopback_pair().await;
        let initial_health = engine.health().snapshot();
        let delivered = write_initial_frames(
            &mut server,
            &engine,
            &mut subscription,
            &ConnectionShape::NoCursor,
            InitialSnapshot {
                reducer: &engine.reducer_snapshot(),
                health: engine.health().snapshot(),
                ingress_p99_us: engine.ingress_p99_us(),
                rebuild_generation: 0,
                published_frontier: 7,
            },
        )
        .await;
        assert_eq!(delivered, Some(7));

        engine.health().bump_reservation_exhausted();
        let mut health_gate = HealthDeliveryGate::seeded(
            initial_health.version,
            Instant::now() - HEALTH_PUBLISH_MIN_INTERVAL,
        );
        assert!(
            maybe_write_health(
                &mut server,
                &engine,
                &subscription,
                &mut health_gate,
                Cursor::new(engine.process_instance(), delivered.expect("checkpoint")),
            )
            .await
        );

        let mut received = String::new();
        let mut buffer = [0u8; 4096];
        for _ in 0..10 {
            let count = tokio::time::timeout(Duration::from_millis(100), client.read(&mut buffer))
                .await
                .expect("initial and follow-up health frames must be readable")
                .expect("read initial and follow-up health frames");
            if count == 0 {
                break;
            }
            received.push_str(&String::from_utf8_lossy(&buffer[..count]));
            if received.matches("event: runtime_health").count() >= 2 {
                break;
            }
        }
        let health_blocks: Vec<&str> = received
            .split("\n\n")
            .filter(|block| block.contains("event: runtime_health"))
            .collect();
        assert_eq!(health_blocks.len(), 2);
        let checkpoint = Cursor::new(engine.process_instance(), 7).encode();
        let checkpoint_line = format!("id: {checkpoint}");
        assert!(
            health_blocks
                .iter()
                .all(|block| block.lines().next() == Some(checkpoint_line.as_str()))
        );

        let engine = RuntimeEventEngine::new();
        let mut subscription = engine.subscribers().subscribe().expect("subscribe");
        let (mut server, _client) = loopback_pair().await;
        let gap = super::super::recovery::Gap {
            reason: super::super::recovery::GapReason::Evicted,
            requested: Cursor::new(engine.process_instance(), 0),
            oldest_available: Some(8),
            latest: Some(9),
        };
        let delivered = write_initial_frames(
            &mut server,
            &engine,
            &mut subscription,
            &ConnectionShape::Gap(gap),
            InitialSnapshot {
                reducer: &engine.reducer_snapshot(),
                health: engine.health().snapshot(),
                ingress_p99_us: engine.ingress_p99_us(),
                rebuild_generation: 0,
                published_frontier: 7,
            },
        )
        .await;
        assert_eq!(delivered, Some(7));
    }

    fn distinct_state_transition_fact() -> RuntimeFact {
        RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeInitialized))
    }

    /// Fails if `maybe_write_health`'s call in `live_loop`'s recv arm is
    /// removed (mutation M3, `.omo/evidence/event-system-fixes/task-08/mutation-proof.txt`).
    /// A fresh `HealthDeliveryGate` (mirroring `HealthDeliveryGate::new`'s
    /// own documented "never delivered yet" contract, exercised by
    /// `health_delivery_gate_delivers_on_the_first_check_from_new` in
    /// `health.rs`) delivers unconditionally on its first eligible check,
    /// so the FIRST time the recv arm's `maybe_write_health` runs -- right
    /// after this test's one submitted event frame -- it must write a
    /// `runtime_health` frame. The 15s `KEEPALIVE_INTERVAL` never fires
    /// within this test's real-time read-retry budget, so a
    /// `runtime_health` frame on the wire is attributable to the recv arm
    /// alone, never the keepalive branch.
    #[tokio::test]
    async fn live_loop_delivers_health_right_after_an_event_frame_not_only_on_keepalive() {
        let engine = RuntimeEventEngine::new();
        let mut subscription = engine.subscribers().subscribe().expect("subscribe");
        let (mut server, mut client) = loopback_pair().await;
        let health_gate = HealthDeliveryGate::new();

        let drive_engine = Arc::clone(&engine);
        let loop_task = tokio::spawn(async move {
            live_loop(
                &mut server,
                &drive_engine,
                &mut subscription,
                health_gate,
                0,
            )
            .await;
        });

        let scope = OperationScope::root_only(OperationId::new());
        let outcome = engine
            .unreserved_ingress(scope)
            .try_submit(distinct_state_transition_fact());
        assert_eq!(outcome, SubmitOutcome::Accepted);
        engine.drain();

        let mut received = String::new();
        let mut buf = [0u8; 4096];
        for _ in 0..100 {
            match tokio::time::timeout(Duration::from_millis(5), client.read(&mut buf)).await {
                Ok(Ok(n)) if n > 0 => {
                    received.push_str(&String::from_utf8_lossy(&buf[..n]));
                    if received.contains("event: runtime_event")
                        && received.contains("event: runtime_health")
                    {
                        break;
                    }
                }
                _ => {}
            }
        }

        loop_task.abort();

        assert!(
            received.contains("event: runtime_event"),
            "the submitted fact must reach the client as a runtime_event frame; got: {received:?}"
        );
        assert!(
            received.contains("event: runtime_health"),
            "the per-frame maybe_write_health call must deliver a health frame right \
             after the event frame, not only on the 15s keepalive tick (never reached \
             within this test's real-time read budget); got: {received:?}"
        );
    }

    /// Task 9 (`.omo/plans/event-system-fixes.md`, defect D11): a closed
    /// client socket must free its subscriber slot right away, not sit
    /// occupied until the next `KEEPALIVE_INTERVAL` (15s) write attempt
    /// discovers the write failing. This closes the CLIENT half of a real
    /// loopback pair and asserts the engine's active-subscriber count
    /// drops back to zero well within a budget far shorter than one
    /// keepalive interval. Nothing submits an event and the keepalive tick
    /// is 15s away, so this exercises the read-side disconnect probe.
    #[tokio::test]
    async fn closing_the_client_socket_frees_the_subscriber_slot_before_the_next_keepalive() {
        let engine = RuntimeEventEngine::new();
        let mut subscription = engine.subscribers().subscribe().expect("subscribe");
        let (mut server, client) = loopback_pair().await;
        let health_gate = HealthDeliveryGate::new();

        assert_eq!(engine.subscribers().active_count(), 1);

        let drive_engine = Arc::clone(&engine);
        let loop_task = tokio::spawn(async move {
            live_loop(
                &mut server,
                &drive_engine,
                &mut subscription,
                health_gate,
                0,
            )
            .await;
        });

        // Close the client's side of the connection -- the server's next
        // read attempt on this socket must observe EOF.
        drop(client);

        let freed = tokio::time::timeout(Duration::from_secs(1), async {
            while engine.subscribers().active_count() != 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(
            freed.is_ok(),
            "the subscriber slot must free within 1s of the client closing its \
             socket, well under the 15s keepalive interval; still occupied: {}",
            engine.subscribers().active_count()
        );

        tokio::time::timeout(Duration::from_millis(200), loop_task)
            .await
            .expect("live_loop must exit promptly once the client closes its socket")
            .expect("live_loop task must not panic");
    }

    #[tokio::test]
    async fn client_bytes_free_the_subscriber_slot() {
        let engine = RuntimeEventEngine::new();
        let mut subscription = engine.subscribers().subscribe().expect("subscribe");
        let (mut server, mut client) = loopback_pair().await;
        let health_gate = HealthDeliveryGate::new();

        assert_eq!(engine.subscribers().active_count(), 1);

        let drive_engine = Arc::clone(&engine);
        let loop_task = tokio::spawn(async move {
            live_loop(
                &mut server,
                &drive_engine,
                &mut subscription,
                health_gate,
                0,
            )
            .await;
        });

        client
            .write_all(b"unexpected client bytes")
            .await
            .expect("write client bytes");

        let freed = tokio::time::timeout(Duration::from_secs(1), async {
            while engine.subscribers().active_count() != 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(
            freed.is_ok(),
            "unexpected client bytes must free the subscriber slot; still occupied: {}",
            engine.subscribers().active_count()
        );

        tokio::time::timeout(Duration::from_millis(200), loop_task)
            .await
            .expect("live_loop must exit after client bytes")
            .expect("live_loop task must not panic");
    }

    #[tokio::test]
    async fn idle_changed_health_is_delivered_before_the_keepalive() {
        let engine = RuntimeEventEngine::new();
        let mut subscription = engine.subscribers().subscribe().expect("subscribe");
        let (mut server, mut client) = loopback_pair().await;
        let health_gate =
            HealthDeliveryGate::seeded(engine.health().snapshot().version, Instant::now());

        let drive_engine = Arc::clone(&engine);
        let loop_task = tokio::spawn(async move {
            live_loop(
                &mut server,
                &drive_engine,
                &mut subscription,
                health_gate,
                0,
            )
            .await;
        });

        engine.health().bump_reservation_exhausted();

        let received = tokio::time::timeout(Duration::from_secs(2), async {
            let mut received = String::new();
            let mut buf = [0u8; 4096];
            loop {
                let count = client.read(&mut buf).await.expect("read health frame");
                if count == 0 {
                    break;
                }
                received.push_str(&String::from_utf8_lossy(&buf[..count]));
                if received.contains("event: runtime_health") {
                    break;
                }
            }
            received
        })
        .await
        .expect("changed health must be delivered before the 15s keepalive");

        loop_task.abort();

        assert!(
            received.contains("event: runtime_health"),
            "idle changed health must reach the client on the independent health tick; got: {received:?}"
        );
    }
}
