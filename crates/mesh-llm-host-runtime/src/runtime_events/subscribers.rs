//! Bounded in-process subscriber handles.
//!
//! Each subscriber owns a small queue rather than sharing a
//! `tokio::sync::broadcast` ring. A broadcast ring can report how many
//! messages a receiver has missed, but it cannot report the bytes of the
//! messages still outstanding for that receiver. Runtime-event frames have
//! deliberately variable wire sizes, so the registry keeps exact per-handle
//! frame, age, and byte accounting. A slow handle is marked lagged at the
//! first exceeded bound; other handles continue to receive independently.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::Notify;
use tokio::sync::broadcast::error::RecvError;

use super::config::{
    MAX_CONCURRENT_SUBSCRIBERS, SUBSCRIBER_LAG_MAX_AGE, SUBSCRIBER_LAG_MAX_BYTES,
    SUBSCRIBER_LAG_MAX_FRAMES,
};
use super::health::EngineHealth;
use super::replay::ReplayFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscribeError {
    CapacityReached,
}

#[derive(Debug)]
struct QueuedFrame {
    frame: ReplayFrame,
    byte_cost: usize,
}

#[derive(Debug)]
struct QueueState {
    frames: VecDeque<QueuedFrame>,
    queued_bytes: usize,
    pending_frames: usize,
    pending_bytes: usize,
    pending_oldest: Option<Instant>,
    lagged: Option<usize>,
}

impl QueueState {
    fn new() -> Self {
        Self {
            frames: VecDeque::new(),
            queued_bytes: 0,
            pending_frames: 0,
            pending_bytes: 0,
            pending_oldest: None,
            lagged: None,
        }
    }

    fn outstanding_frames(&self) -> usize {
        self.frames.len().saturating_add(self.pending_frames)
    }

    fn outstanding_bytes(&self) -> usize {
        self.queued_bytes.saturating_add(self.pending_bytes)
    }

    fn oldest_recorded_at(&self) -> Option<Instant> {
        self.frames
            .front()
            .map(|queued| queued.frame.recorded_at)
            .into_iter()
            .chain(self.pending_oldest)
            .min()
    }

    fn exceeds(&self, frame_capacity: usize, now: Instant) -> bool {
        self.outstanding_frames() > frame_capacity
            || self.outstanding_bytes() > SUBSCRIBER_LAG_MAX_BYTES
            || self.oldest_recorded_at().is_some_and(|oldest| {
                now.saturating_duration_since(oldest) > SUBSCRIBER_LAG_MAX_AGE
            })
    }

    fn mark_lagged(&mut self) {
        if self.lagged.is_some() {
            return;
        }
        // The exact number is no longer useful once the queue is closed, but
        // preserving a positive value keeps the broadcast receiver contract.
        self.lagged = Some(self.outstanding_frames().max(1));
        self.frames.clear();
        self.queued_bytes = 0;
        self.pending_frames = 0;
        self.pending_bytes = 0;
        self.pending_oldest = None;
    }
}

struct SubscriberState {
    queue: Mutex<QueueState>,
    notify: Notify,
}

struct RegistryInner {
    subscribers: Mutex<HashMap<u64, Arc<SubscriberState>>>,
    next_id: AtomicU64,
    active: AtomicUsize,
    frame_capacity: usize,
}

/// Shared registry: `publish` fans a frame out to every live subscriber.
#[derive(Clone)]
pub struct SubscriberRegistry {
    inner: Arc<RegistryInner>,
}

/// A live subscription. Dropping it frees a slot back to the registry.
pub struct SubscriptionHandle {
    registry: Arc<RegistryInner>,
    id: u64,
    state: Arc<SubscriberState>,
}

impl SubscriberRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(SUBSCRIBER_LAG_MAX_FRAMES)
    }

    /// Same as [`Self::new`] with an explicit frame-count lag capacity;
    /// tests use this to shrink the queue so slow-consumer scenarios are
    /// reachable in a handful of publishes.
    #[must_use]
    pub fn with_capacity(lag_frames: usize) -> Self {
        Self {
            inner: Arc::new(RegistryInner {
                subscribers: Mutex::new(HashMap::new()),
                next_id: AtomicU64::new(0),
                active: AtomicUsize::new(0),
                frame_capacity: lag_frames,
            }),
        }
    }

    pub fn subscribe(&self) -> Result<SubscriptionHandle, SubscribeError> {
        let active = self.inner.active.fetch_add(1, Ordering::AcqRel);
        if active >= MAX_CONCURRENT_SUBSCRIBERS {
            self.inner.active.fetch_sub(1, Ordering::AcqRel);
            return Err(SubscribeError::CapacityReached);
        }

        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let state = Arc::new(SubscriberState {
            queue: Mutex::new(QueueState::new()),
            notify: Notify::new(),
        });
        self.inner
            .subscribers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, Arc::clone(&state));
        Ok(SubscriptionHandle {
            registry: Arc::clone(&self.inner),
            id,
            state,
        })
    }

    /// Fan `frame` out to every live subscriber without waiting on any
    /// consumer. Each queue makes its own first-limit decision using the
    /// exact bytes of this frame, so a large frame cannot falsely disconnect
    /// a fast neighbor merely because another queue is full.
    pub fn publish(&self, frame: ReplayFrame) {
        let subscribers: Vec<Arc<SubscriberState>> = self
            .inner
            .subscribers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .cloned()
            .collect();

        for state in subscribers {
            let mut queue = state
                .queue
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if queue.lagged.is_some() {
                continue;
            }
            let byte_cost = frame.wire_bytes.len();
            queue.frames.push_back(QueuedFrame {
                frame: frame.clone(),
                byte_cost,
            });
            queue.queued_bytes = queue.queued_bytes.saturating_add(byte_cost);
            if queue.exceeds(self.inner.frame_capacity, Instant::now()) {
                queue.mark_lagged();
            }
            drop(queue);
            state.notify.notify_one();
        }
    }

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.inner.active.load(Ordering::Acquire)
    }
}

impl Default for SubscriberRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SubscriptionHandle {
    /// Receive the next frame. The queue is checked for an age violation
    /// before every delivery, so a stale connection is disconnected even if
    /// no new publication arrives after its oldest frame expires.
    pub async fn recv(&mut self) -> Result<ReplayFrame, RecvError> {
        loop {
            let notified = self.state.notify.notified();
            let result = {
                let mut queue = self
                    .state
                    .queue
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                self.check_locked(&mut queue, Instant::now());
                if let Some(missed) = queue.lagged {
                    Some(Err(RecvError::Lagged(missed as u64)))
                } else if let Some(queued) = queue.frames.pop_front() {
                    queue.queued_bytes = queue.queued_bytes.saturating_sub(queued.byte_cost);
                    Some(Ok(queued.frame))
                } else {
                    None
                }
            };
            if let Some(result) = result {
                return result;
            }
            notified.await;
        }
    }

    /// Number of live event frames waiting in this handle's queue. Pending
    /// initial/replay/state/health writes are tracked separately in the exact
    /// byte/count budget and are intentionally not reported as queued events.
    #[must_use]
    pub fn backlog_len(&self) -> usize {
        self.state
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .frames
            .len()
    }

    /// Exact bytes currently outstanding for this subscriber, including
    /// queued live frames and pending initial/replay/state/health writes.
    #[must_use]
    pub fn outstanding_bytes(&self) -> usize {
        self.state
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .outstanding_bytes()
    }

    /// Reserve a batch of pending wire frames before their first write. The
    /// caller releases each frame with [`Self::complete_pending`]. This lets
    /// a replay-window catch-up count the entire captured write set while the
    /// socket is still sending it, rather than temporarily undercounting a
    /// slow connection.
    pub fn reserve_pending(&self, frame_count: usize, bytes: usize, recorded_at: Instant) -> bool {
        let mut queue = self
            .state
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if queue.lagged.is_some() {
            return false;
        }
        queue.pending_frames = queue.pending_frames.saturating_add(frame_count);
        queue.pending_bytes = queue.pending_bytes.saturating_add(bytes);
        queue.pending_oldest = Some(match queue.pending_oldest {
            Some(previous) => previous.min(recorded_at),
            None => recorded_at,
        });
        if queue.exceeds(self.registry.frame_capacity, Instant::now()) {
            queue.mark_lagged();
            drop(queue);
            self.state.notify.notify_one();
            return false;
        }
        true
    }

    /// Mark one previously reserved pending frame as written (or failed).
    pub fn complete_pending(&self, bytes: usize) {
        let mut queue = self
            .state
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        queue.pending_frames = queue.pending_frames.saturating_sub(1);
        queue.pending_bytes = queue.pending_bytes.saturating_sub(bytes);
        if queue.pending_frames == 0 {
            queue.pending_oldest = None;
        }
    }

    /// Check the exact count/age/byte budget and mark this connection lagged
    /// at the first exceeded bound. Returns true when it is closed for lag
    /// (or was already closed for lag).
    pub fn lag_bound_exceeded(&self, now: Instant) -> bool {
        let mut queue = self
            .state
            .queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.check_locked(&mut queue, now)
    }

    fn check_locked(&self, queue: &mut QueueState, now: Instant) -> bool {
        if queue.lagged.is_some() {
            return true;
        }
        if queue.exceeds(self.registry.frame_capacity, now) {
            queue.mark_lagged();
            self.state.notify.notify_one();
            return true;
        }
        false
    }

    pub fn record_disconnect(&self, health: &EngineHealth) {
        health.bump_subscriber_disconnected();
    }
}

impl Drop for SubscriptionHandle {
    fn drop(&mut self) {
        self.registry
            .subscribers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.id);
        self.registry.active.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use mesh_llm_runtime_event_contracts::{
        EventSequence, FamilyFact, NativeRuntimeEventKind, OperationId, OperationScope, RuntimeFact,
    };

    use super::*;

    fn frame(sequence: u64, bytes: usize) -> ReplayFrame {
        ReplayFrame {
            sequence: EventSequence::new(sequence),
            rebuild_generation: 0,
            scope: OperationScope::root_only(OperationId::new()),
            fact: Arc::new(RuntimeFact::NativeRuntime(FamilyFact::new(
                NativeRuntimeEventKind::RuntimeStopped,
            ))),
            recorded_at: Instant::now(),
            wire_bytes: Arc::from(vec![0u8; bytes]),
        }
    }

    #[tokio::test]
    async fn subscriber_receives_published_frames() {
        let registry = SubscriberRegistry::new();
        let mut handle = registry.subscribe().expect("subscribe");

        registry.publish(frame(1, 10));
        let received = handle.recv().await.expect("recv");
        assert_eq!(received.sequence.get(), 1);
    }

    #[test]
    fn subscription_beyond_the_cap_is_rejected() {
        let registry = SubscriberRegistry::new();
        let mut handles = Vec::new();
        for _ in 0..MAX_CONCURRENT_SUBSCRIBERS {
            handles.push(registry.subscribe().expect("subscribe under cap"));
        }

        assert!(matches!(
            registry.subscribe(),
            Err(SubscribeError::CapacityReached)
        ));
        assert_eq!(registry.active_count(), MAX_CONCURRENT_SUBSCRIBERS);
    }

    #[test]
    fn dropping_a_subscription_frees_its_slot() {
        let registry = SubscriberRegistry::new();
        let handle = registry.subscribe().expect("subscribe");
        assert_eq!(registry.active_count(), 1);
        drop(handle);
        assert_eq!(registry.active_count(), 0);
    }

    #[tokio::test]
    async fn a_subscriber_that_stops_draining_is_lagged_by_exact_frame_count() {
        let registry = SubscriberRegistry::with_capacity(4);
        let mut handle = registry.subscribe().expect("subscribe");

        for sequence in 0..5 {
            registry.publish(frame(sequence, 10));
        }

        assert!(matches!(
            handle.recv().await,
            Err(RecvError::Lagged(missed)) if missed == 5
        ));
    }

    #[test]
    fn record_disconnect_bumps_the_subscriber_disconnected_health_counter() {
        let registry = SubscriberRegistry::new();
        let handle = registry.subscribe().expect("subscribe");
        let health = EngineHealth::default();

        handle.record_disconnect(&health);

        assert_eq!(health.snapshot().subscriber_disconnected, 1);
    }

    #[test]
    fn exact_bytes_include_pending_initial_frames() {
        let registry = SubscriberRegistry::new();
        let handle = registry.subscribe().expect("subscribe");
        let now = Instant::now();
        assert!(handle.reserve_pending(1, 100, now));
        assert_eq!(handle.outstanding_bytes(), 100);
        handle.complete_pending(100);
        assert_eq!(handle.outstanding_bytes(), 0);
    }

    #[tokio::test]
    async fn one_large_frame_does_not_disconnect_a_fast_neighbor() {
        let registry = SubscriberRegistry::with_capacity(16);
        let mut fast = registry.subscribe().expect("fast subscribe");
        let mut slow = registry.subscribe().expect("slow subscribe");

        let large = SUBSCRIBER_LAG_MAX_BYTES / 4;
        for sequence in 0..4 {
            registry.publish(frame(sequence, large));
            assert_eq!(
                fast.recv().await.expect("fast frame").sequence.get(),
                sequence
            );
        }
        registry.publish(frame(4, 1));

        assert!(matches!(slow.recv().await, Err(RecvError::Lagged(_))));
        assert_eq!(fast.recv().await.expect("fast frame").sequence.get(), 4);
    }

    #[test]
    fn age_bound_is_checked_without_a_new_publication() {
        let registry = SubscriberRegistry::new();
        let handle = registry.subscribe().expect("subscribe");
        let old = Instant::now();
        assert!(handle.reserve_pending(1, 1, old));
        assert!(handle.lag_bound_exceeded(old + SUBSCRIBER_LAG_MAX_AGE + Duration::from_secs(1)));
    }
}
