//! OTLP-agnostic runtime-event telemetry bridge.
//!
//! This module owns exactly the pieces that need engine-internal access
//! (the reservation table, `EngineHealth`) but have no opinion about how
//! samples get exported: no `opentelemetry` dependency, no attribute
//! allowlist, no exporter. `crate::runtime::survey::runtime_events` is the
//! OTLP-specific consumer that turns [`RuntimeEventTelemetrySample`] values
//! into allowlisted counters/histograms; this module never imports it (the
//! dependency points one way, consumer -> bridge, matching every other
//! consumer in this plan reading engine state rather than the engine
//! reaching out to a specific consumer).
//!
//! [`RuntimeEventTelemetryQueue::push`] never waits for the queue lock.
//! Contention drops the incoming sample; capacity pressure drops the oldest
//! sample. Both are counted so the OTLP consumer can report telemetry loss.
//! The queue never calls back
//! into `RuntimeEventIngress::try_submit` -- pushing telemetry can never
//! itself generate a new runtime event, so there is no recursion path.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, TryLockError};
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{
    DeliveryClass, RuntimeEventIngress, RuntimeFact, SubmitOutcome,
};

use super::engine::RuntimeEventEngine;
use super::health::EngineHealthSnapshot;

/// A privacy-safe, ID-free structured telemetry sample. Every variant here
/// is bounded discrete data (an enum, a count, a duration) -- never an
/// operation/session/request ID, never prompt/completion content.
#[derive(Debug, Clone, Copy)]
pub enum RuntimeEventTelemetrySample {
    /// One observed `try_submit` call: the delivery class the fact carried,
    /// the outcome the engine returned, and the wall-clock time the call
    /// took. Produced either by [`ObservingIngress`] (an opt-in decorator)
    /// or by `RuntimeEventEngine::submit`'s own installed-queue hook (the
    /// live production path -- see `RuntimeEventEngine::install_telemetry_queue`).
    /// This is the source for the certification ingress-latency (p99)
    /// instrument.
    ClassOutcome {
        class: DeliveryClass,
        outcome: SubmitOutcome,
        ingress_elapsed: Duration,
    },
    /// A coalesced `EngineHealth` snapshot, sampled at the same
    /// once-per-second-or-less cadence the sampler's own `HealthDeliveryGate`
    /// enforces (task 8) -- never sampled per-event.
    EngineHealth(EngineHealthSnapshot),
    /// Point-in-time reservation-table occupancy, sampled at the same
    /// cadence as `EngineHealth`. The "queue depth" signal.
    ReservationOccupancy { occupied: usize, capacity: usize },
    /// This telemetry pipeline's own bounded-queue health: a cumulative
    /// count of samples dropped because the queue was full when pushed
    /// (i.e. the exporter/worker fell behind). Proves telemetry survives a
    /// slow or stalled exporter without blocking or growing unbounded.
    Pipeline(TelemetryPipelineSnapshot),
}

/// Cumulative counters describing this telemetry pipeline's own health.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TelemetryPipelineSnapshot {
    pub samples_dropped: u64,
}

/// Bounded, drop-oldest sample queue. The only way a sample reaches an
/// exporter: producers/samplers push here (always fast, never blocking),
/// an independent worker drains and records.
pub struct RuntimeEventTelemetryQueue {
    capacity: usize,
    samples: Mutex<VecDeque<RuntimeEventTelemetrySample>>,
    dropped: AtomicU64,
}

impl RuntimeEventTelemetryQueue {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            samples: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
            dropped: AtomicU64::new(0),
        }
    }

    /// Push without waiting for a consumer or another producer. Contention
    /// drops this sample; a full queue drops its oldest sample instead.
    pub fn push(&self, sample: RuntimeEventTelemetrySample) {
        let mut samples = match self.samples.try_lock() {
            Ok(samples) => samples,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        if samples.len() >= self.capacity {
            samples.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        samples.push_back(sample);
    }

    /// Drain every currently-queued sample for the worker to record.
    pub fn drain(&self) -> Vec<RuntimeEventTelemetrySample> {
        let mut samples = self
            .samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        samples.drain(..).collect()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.samples
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// This pipeline's own health as a sample-shaped snapshot, ready to be
    /// pushed back into itself by a periodic sampler.
    #[must_use]
    pub fn pipeline_snapshot(&self) -> TelemetryPipelineSnapshot {
        TelemetryPipelineSnapshot {
            samples_dropped: self.dropped.load(Ordering::Relaxed),
        }
    }

    /// Record one observed `try_submit` call. The single construction point
    /// for a `ClassOutcome` sample, shared by [`ObservingIngress`] (an
    /// explicit producer-side decorator) and `RuntimeEventEngine::submit`'s
    /// own installed-queue hook (the live wiring every producer already
    /// funnels through -- see `RuntimeEventEngine::install_telemetry_queue`).
    pub fn record_class_outcome(
        &self,
        class: DeliveryClass,
        outcome: SubmitOutcome,
        ingress_elapsed: Duration,
    ) {
        self.push(RuntimeEventTelemetrySample::ClassOutcome {
            class,
            outcome,
            ingress_elapsed,
        });
    }
}

/// A `RuntimeEventIngress` decorator: forwards every `try_submit` to
/// `inner` unchanged, then records a [`RuntimeEventTelemetrySample::ClassOutcome`]
/// sample via [`RuntimeEventTelemetryQueue::record_class_outcome`]. Available
/// for a producer that wants to opt a specific ingress handle in directly;
/// the live production wiring instead installs a queue on the engine itself
/// (`RuntimeEventEngine::install_telemetry_queue`), which observes every
/// producer's real traffic without touching task 9-12's owned files -- see
/// that method's doc for why the engine is the chosen seam.
pub struct ObservingIngress<I: RuntimeEventIngress> {
    inner: I,
    queue: std::sync::Arc<RuntimeEventTelemetryQueue>,
}

impl<I: RuntimeEventIngress> ObservingIngress<I> {
    pub fn new(inner: I, queue: std::sync::Arc<RuntimeEventTelemetryQueue>) -> Self {
        Self { inner, queue }
    }
}

impl<I: RuntimeEventIngress> RuntimeEventIngress for ObservingIngress<I> {
    fn try_submit(&self, fact: RuntimeFact) -> SubmitOutcome {
        let class = fact.delivery_class();
        let start = Instant::now();
        let outcome = self.inner.try_submit(fact);
        self.queue
            .record_class_outcome(class, outcome, start.elapsed());
        outcome
    }
}

/// Sample `engine`'s coalesced health and reservation occupancy into
/// `queue`. Task 8-fix E4 (`.omo/plans/event-system-fixes.md`): ungated by
/// design -- the sole caller, `spawn_runtime_event_telemetry_sampler`
/// below, already drives this from its own
/// `tokio::time::interval(HEALTH_PUBLISH_MIN_INTERVAL)`, so the 1 Hz
/// cadence is the CALLER's job, not this function's. This used to also
/// gate on its own `HealthDeliveryGate` (requiring `changed`, mirroring
/// the frame-delivery consumers in `stream.rs`/`presentation/subscriber.rs`),
/// but that pattern fits a push-stream consumer -- where "never repeat an
/// identical frame" genuinely matters -- not an already-periodic sampler.
/// Every one of `EngineHealth`'s 11 counters is a pure anomaly counter, so
/// on a healthy node the gated version silenced BOTH the health sample AND
/// `ReservationOccupancy` (a true instantaneous gauge, unrelated to
/// whether health "changed") for the rest of the process's life after the
/// first tick, turning "the 'queue depth' signal" (this module's doc
/// comment on `ReservationOccupancy` above) into a metric that stops
/// reporting on a healthy node -- most dashboards/alerts read a stalled
/// gauge as "exporter died," not "nothing to report." Pushing
/// unconditionally restores the parent commit's `EngineHealth::publish_at`
/// semantics exactly (pure interval gating, no change check). The OTLP
/// recorder (`runtime::survey::runtime_events::record_health_delta`)
/// already computes real deltas and only calls `Counter::add` when a
/// counter actually moved, so pushing an unchanged snapshot every second is
/// a harmless no-op there -- only the continuously-live
/// `reservation_occupied` Gauge benefits from the restored cadence.
pub fn sample_engine(engine: &RuntimeEventEngine, queue: &RuntimeEventTelemetryQueue) {
    let snapshot = engine.health().snapshot();
    queue.push(RuntimeEventTelemetrySample::EngineHealth(snapshot));
    let table = engine.table();
    let capacity = table.capacity();
    let occupied = (0..capacity)
        .filter(|&index| table.is_occupied(index).is_some())
        .count();
    queue.push(RuntimeEventTelemetrySample::ReservationOccupancy { occupied, capacity });
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_runtime_event_contracts::{FamilyFact, NativeRuntimeEventKind, OperationId};
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    fn terminal_fact() -> RuntimeFact {
        RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
    }

    // ─── Queue: bounded, drop-oldest, never blocks ─────────────────────

    #[test]
    fn queue_push_never_exceeds_capacity() {
        let queue = RuntimeEventTelemetryQueue::new(2);
        for _ in 0..5 {
            queue.push(RuntimeEventTelemetrySample::ReservationOccupancy {
                occupied: 0,
                capacity: 0,
            });
        }
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn queue_drops_oldest_and_counts_the_drop() {
        let queue = RuntimeEventTelemetryQueue::new(1);
        queue.push(RuntimeEventTelemetrySample::ReservationOccupancy {
            occupied: 1,
            capacity: 1,
        });
        queue.push(RuntimeEventTelemetrySample::ReservationOccupancy {
            occupied: 2,
            capacity: 2,
        });
        assert_eq!(queue.pipeline_snapshot().samples_dropped, 1);
        let drained = queue.drain();
        assert_eq!(drained.len(), 1);
        assert!(matches!(
            drained[0],
            RuntimeEventTelemetrySample::ReservationOccupancy { occupied: 2, .. }
        ));
    }

    #[test]
    fn a_consumer_holding_the_queue_cannot_block_ingress_telemetry() {
        let queue = Arc::new(RuntimeEventTelemetryQueue::new(2));
        let held = queue.samples.lock().expect("hold consumer lock");
        let (finished, completion) = std::sync::mpsc::channel();
        let producer_queue = Arc::clone(&queue);
        let producer = std::thread::spawn(move || {
            let observing = ObservingIngress::new(
                CountingIngress {
                    calls: AtomicUsize::new(0),
                    outcome: SubmitOutcome::Accepted,
                },
                producer_queue,
            );
            finished
                .send(observing.try_submit(terminal_fact()))
                .unwrap();
        });
        let result = completion.recv_timeout(Duration::from_secs(2));
        // Release before asserting, so a regression fails instead of leaving
        // a blocked producer thread behind.
        drop(held);
        producer.join().expect("producer thread");
        assert_eq!(
            result.expect("ingress must not wait"),
            SubmitOutcome::Accepted
        );
        assert_eq!(queue.pipeline_snapshot().samples_dropped, 1);
        assert!(queue.is_empty());
    }

    #[test]
    fn queue_recovers_after_a_backpressure_burst() {
        // A slow/stalled consumer manifests as nobody draining the queue: a
        // burst of pushes past capacity drops old samples but never
        // corrupts the queue or blocks -- once draining resumes, new
        // samples flow normally again (self-healing, not a permanent
        // broken state).
        let queue = RuntimeEventTelemetryQueue::new(2);
        for _ in 0..50 {
            queue.push(RuntimeEventTelemetrySample::ReservationOccupancy {
                occupied: 0,
                capacity: 0,
            });
        }
        assert_eq!(queue.pipeline_snapshot().samples_dropped, 48);
        let _ = queue.drain();
        assert!(queue.is_empty());

        queue.push(RuntimeEventTelemetrySample::ReservationOccupancy {
            occupied: 7,
            capacity: 10,
        });
        let drained = queue.drain();
        assert_eq!(drained.len(), 1);
        assert!(matches!(
            drained[0],
            RuntimeEventTelemetrySample::ReservationOccupancy { occupied: 7, .. }
        ));
    }

    // ─── ObservingIngress: forwards, times, never recurses ─────────────

    struct CountingIngress {
        calls: AtomicUsize,
        outcome: SubmitOutcome,
    }

    impl RuntimeEventIngress for CountingIngress {
        fn try_submit(&self, _fact: RuntimeFact) -> SubmitOutcome {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.outcome
        }
    }

    #[test]
    fn observing_ingress_forwards_the_outcome_and_records_a_sample() {
        let inner = CountingIngress {
            calls: AtomicUsize::new(0),
            outcome: SubmitOutcome::Accepted,
        };
        let queue = Arc::new(RuntimeEventTelemetryQueue::new(4));
        let observing = ObservingIngress::new(inner, Arc::clone(&queue));

        let outcome = observing.try_submit(terminal_fact());

        assert_eq!(outcome, SubmitOutcome::Accepted);
        assert_eq!(observing.inner.calls.load(Ordering::SeqCst), 1);
        let drained = queue.drain();
        assert_eq!(drained.len(), 1);
        match drained[0] {
            RuntimeEventTelemetrySample::ClassOutcome { class, outcome, .. } => {
                assert_eq!(class, DeliveryClass::Terminal);
                assert_eq!(outcome, SubmitOutcome::Accepted);
            }
            other => panic!("expected ClassOutcome, got {other:?}"),
        }
    }

    #[test]
    fn observing_ingress_calls_inner_exactly_once_never_recursing_into_ingress() {
        // Recursion prevention: recording telemetry about a submission must
        // never itself submit a new fact. A queue at capacity still only
        // ever touches its own Mutex<VecDeque> -- never `try_submit` again.
        let inner = CountingIngress {
            calls: AtomicUsize::new(0),
            outcome: SubmitOutcome::Accepted,
        };
        let queue = Arc::new(RuntimeEventTelemetryQueue::new(1));
        // Saturate the queue before submitting so the push inside
        // `try_submit` is forced through the drop-oldest path.
        queue.push(RuntimeEventTelemetrySample::ReservationOccupancy {
            occupied: 0,
            capacity: 0,
        });
        let observing = ObservingIngress::new(inner, Arc::clone(&queue));

        let _ = observing.try_submit(terminal_fact());

        assert_eq!(observing.inner.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn observing_ingress_never_blocks_even_when_the_queue_is_saturated() {
        let inner = CountingIngress {
            calls: AtomicUsize::new(0),
            outcome: SubmitOutcome::DroppedDiagnostic,
        };
        let queue = Arc::new(RuntimeEventTelemetryQueue::new(1));
        let observing = ObservingIngress::new(inner, Arc::clone(&queue));

        let start = Instant::now();
        for _ in 0..1_000 {
            let _ = observing.try_submit(terminal_fact());
        }
        // No exporter, no I/O anywhere in this path -- 1,000 saturated
        // pushes complete well under a second on any machine; a generous
        // bound catches an accidental blocking regression without being
        // flaky under CI load.
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    // ─── sample_engine: ungated (task 8-fix E4), real occupancy ────────

    #[test]
    fn sample_engine_pushes_both_samples_unconditionally_on_every_call() {
        // Task 8-fix E4: `sample_engine` no longer gates internally -- its
        // sole caller's own 1 Hz `tokio::time::interval` is the cadence
        // authority now (`spawn_runtime_event_telemetry_sampler`). Three
        // back-to-back calls with nothing changing between them must still
        // push 2 samples EACH (6 total), proving no internal gate remains.
        let engine = RuntimeEventEngine::with_capacity(4);
        let queue = RuntimeEventTelemetryQueue::new(8);

        sample_engine(&engine, &queue);
        sample_engine(&engine, &queue);
        sample_engine(&engine, &queue);

        assert_eq!(queue.drain().len(), 6);
    }

    #[test]
    fn sample_engine_reports_real_reservation_occupancy() {
        let engine = RuntimeEventEngine::with_capacity(4);
        let queue = RuntimeEventTelemetryQueue::new(8);
        let reservation = engine
            .reserve_root(OperationId::new(), terminal_fact)
            .expect("reserve");

        sample_engine(&engine, &queue);

        let drained = queue.drain();
        let occupancy = drained.iter().find_map(|sample| match sample {
            RuntimeEventTelemetrySample::ReservationOccupancy { occupied, .. } => Some(*occupied),
            _ => None,
        });
        assert_eq!(occupancy, Some(1));
        drop(reservation);
    }

    #[test]
    fn sample_engine_never_generates_a_new_event_no_recursion() {
        // Reserve, then sample repeatedly: occupancy must stay 1 the whole
        // time (sampling itself never reserves/submits).
        let engine = RuntimeEventEngine::with_capacity(4);
        let queue = RuntimeEventTelemetryQueue::new(64);
        let reservation = engine
            .reserve_root(OperationId::new(), terminal_fact)
            .expect("reserve");
        assert_eq!(engine.occupied_count(), 1);

        for _ in 0..5u32 {
            sample_engine(&engine, &queue);
        }

        assert_eq!(engine.occupied_count(), 1);
        drop(reservation);
    }

    // ─── No-ID inspection ───────────────────────────────────────────────

    #[test]
    fn class_outcome_sample_carries_no_identifiers() {
        let sample = RuntimeEventTelemetrySample::ClassOutcome {
            class: DeliveryClass::Progress,
            outcome: SubmitOutcome::Coalesced,
            ingress_elapsed: Duration::from_micros(42),
        };
        let debug = format!("{sample:?}");
        // The Debug rendering of a bounded enum + a duration must never
        // resemble a UUID (operation IDs are UUIDs; the sample type has no
        // field capable of holding one).
        assert!(!debug.contains('-') || debug.matches('-').count() < 4);
    }
}
