//! Fixed-capacity, allocation-free ring of recent `try_submit` ingress
//! durations -- task 13 (`.omo/plans/event-system-fixes.md`, defect D13's
//! p99 half): the certification p99 instrument, independent of whether
//! OTLP telemetry is configured. `RuntimeEventTelemetryQueue`
//! (`runtime_events::telemetry`) also records every `ClassOutcome`
//! sample's `ingress_elapsed`, but that queue is only ever installed when
//! OTLP telemetry is configured
//! (`RuntimeEventEngine::install_telemetry_queue`, called from
//! `survey::runtime_events::RuntimeEventTelemetry::start`). This reservoir
//! is a SEPARATE, always-present field on `RuntimeEventEngine`, written
//! unconditionally by every `submit` call regardless of that installation
//! state, so `ingress_p99_us` is available on the wire and in the
//! `event_system_health` log line in every serving mode, with or without a
//! telemetry exporter configured.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::config::{INGRESS_LATENCY_MIN_SAMPLES, INGRESS_LATENCY_RESERVOIR_CAPACITY};

/// A fixed ring of the frozen [`INGRESS_LATENCY_RESERVOIR_CAPACITY`] most
/// recent ingress durations, in whole microseconds. Every slot is
/// preallocated once at construction (engine startup); [`Self::record`]
/// only ever performs atomic stores into already-owned memory, so it
/// allocates nothing on the hot submit path.
pub(crate) struct IngressLatencyReservoir {
    samples: Box<[AtomicU64]>,
    total_writes: AtomicU64,
}

impl IngressLatencyReservoir {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::with_capacity(INGRESS_LATENCY_RESERVOIR_CAPACITY)
    }

    /// Explicit-capacity constructor: production always uses the frozen
    /// [`INGRESS_LATENCY_RESERVOIR_CAPACITY`] via [`Self::new`]; tests use
    /// this directly so wraparound is provable in a handful of samples
    /// instead of the frozen 4,096.
    #[must_use]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            samples: (0..capacity).map(|_| AtomicU64::new(0)).collect(),
            total_writes: AtomicU64::new(0),
        }
    }

    /// Record one observed ingress duration. Lock-free and
    /// allocation-free: one atomic fetch-add for the write cursor plus one
    /// atomic store into already-owned memory. Returns `true` exactly when
    /// this write completed a full [`INGRESS_LATENCY_MIN_SAMPLES`]-sample
    /// milestone (the 100th, 200th, ... write), so the caller can cheaply
    /// decide whether to bump `EngineHealth`'s version -- see
    /// `RuntimeEventEngine::submit` and
    /// `.omo/evidence/event-system-fixes/task-13/p99-cadence-note.txt` for
    /// why bumping on every sample (flooding health delivery) and never
    /// bumping (stranding a real p99 change forever) are both wrong.
    pub(crate) fn record(&self, duration: Duration) -> bool {
        let micros = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        let previous_total = self.total_writes.fetch_add(1, Ordering::Relaxed);
        let index = (previous_total as usize) % self.samples.len();
        self.samples[index].store(micros, Ordering::Relaxed);
        let count_after = previous_total + 1;
        count_after.is_multiple_of(INGRESS_LATENCY_MIN_SAMPLES as u64)
    }

    /// p99 (99th percentile, nearest-rank) over the currently held
    /// samples, or `None` before [`INGRESS_LATENCY_MIN_SAMPLES`] samples
    /// have ever been recorded (frozen: "nullable until 100 samples
    /// exist"). Never on the submit path -- called only when building a
    /// health snapshot for delivery (at most once per second per
    /// consumer, per `HealthDeliveryGate`), so the sort below is cheap in
    /// practice despite allocating a scratch `Vec`.
    #[must_use]
    pub(crate) fn p99_micros(&self) -> Option<u64> {
        let total = self.total_writes.load(Ordering::Relaxed);
        if total < INGRESS_LATENCY_MIN_SAMPLES as u64 {
            return None;
        }
        let filled = total.min(self.samples.len() as u64) as usize;
        let mut values: Vec<u64> = self.samples[..filled]
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .collect();
        values.sort_unstable();
        let rank = ((filled as f64) * 0.99).ceil() as usize;
        let index = rank.saturating_sub(1).min(filled - 1);
        Some(values[index])
    }

    #[cfg(test)]
    fn total_writes(&self) -> u64 {
        self.total_writes.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_p99(mut values: Vec<u64>) -> u64 {
        values.sort_unstable();
        let rank = ((values.len() as f64) * 0.99).ceil() as usize;
        let index = rank.saturating_sub(1).min(values.len() - 1);
        values[index]
    }

    #[test]
    fn p99_is_null_before_the_minimum_sample_threshold() {
        let reservoir = IngressLatencyReservoir::with_capacity(4_096);
        for i in 0..(INGRESS_LATENCY_MIN_SAMPLES - 1) {
            reservoir.record(Duration::from_micros(i as u64));
        }
        assert_eq!(reservoir.p99_micros(), None);
    }

    #[test]
    fn p99_becomes_available_at_exactly_the_minimum_sample_threshold() {
        let reservoir = IngressLatencyReservoir::with_capacity(4_096);
        for i in 0..INGRESS_LATENCY_MIN_SAMPLES {
            reservoir.record(Duration::from_micros(i as u64));
        }
        assert!(reservoir.p99_micros().is_some());
    }

    #[test]
    fn p99_matches_a_reference_computation_on_a_seeded_sample() {
        // A deterministic pseudo-random-looking but fixed sequence (a
        // linear congruential sequence, not `rand`, so this test needs no
        // new dependency and is byte-reproducible across runs).
        let mut seed: u64 = 12_345;
        let mut values = Vec::with_capacity(500);
        let reservoir = IngressLatencyReservoir::with_capacity(4_096);
        for _ in 0..500 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let micros = seed % 10_000;
            values.push(micros);
            reservoir.record(Duration::from_micros(micros));
        }

        let expected = reference_p99(values);
        assert_eq!(reservoir.p99_micros(), Some(expected));
    }

    #[test]
    fn ring_retains_only_the_most_recent_capacity_samples() {
        let capacity = 16;
        let reservoir = IngressLatencyReservoir::with_capacity(capacity);
        // Fill past capacity with an ascending sequence: the ring must
        // keep only the LAST `capacity` values (the earliest ones are
        // overwritten), so p99 must reflect that tail window, not the
        // whole submitted history.
        let total_writes = INGRESS_LATENCY_MIN_SAMPLES + capacity * 3;
        for i in 0..total_writes {
            reservoir.record(Duration::from_micros(i as u64));
        }
        assert_eq!(reservoir.total_writes(), total_writes as u64);

        let tail_start = total_writes - capacity;
        let expected_tail: Vec<u64> = (tail_start..total_writes).map(|v| v as u64).collect();
        let expected = reference_p99(expected_tail);
        assert_eq!(reservoir.p99_micros(), Some(expected));
    }

    #[test]
    fn record_reports_a_milestone_exactly_every_min_samples_writes() {
        let reservoir = IngressLatencyReservoir::with_capacity(4_096);
        let mut milestones = 0;
        for i in 0..(INGRESS_LATENCY_MIN_SAMPLES * 3) {
            if reservoir.record(Duration::from_micros(i as u64)) {
                milestones += 1;
            }
        }
        assert_eq!(
            milestones, 3,
            "a milestone must fire on exactly the 100th, 200th, and 300th write"
        );
    }

    #[test]
    fn record_never_reports_a_milestone_between_multiples() {
        let reservoir = IngressLatencyReservoir::with_capacity(4_096);
        for i in 0..(INGRESS_LATENCY_MIN_SAMPLES - 1) {
            assert!(!reservoir.record(Duration::from_micros(i as u64)));
        }
    }
}
