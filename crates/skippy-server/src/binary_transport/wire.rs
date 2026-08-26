use std::{
    cell::Cell,
    io,
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::Duration,
};

use anyhow::{Result, bail};
use skippy_protocol::binary::{StageWireMessage, write_stage_message};

const WIRE_SAMPLE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Clone, Copy, Debug)]
pub struct WireCondition {
    delay_ms: f64,
    mbps: Option<f64>,
    jitter_ms: f64,
    stall_ms: f64,
    stall_p: f64,
}

/// Upper bound for each simulated delay component. An hour-long simulated
/// stall is already far beyond any useful wire model, and bounding the inputs
/// keeps the combined delay inside `Duration::from_secs_f64`'s domain.
const MAX_SIMULATED_DELAY_MS: f64 = 3_600_000.0;

impl WireCondition {
    pub fn new(delay_ms: f64, mbps: Option<f64>) -> Result<Self> {
        Self::with_jitter(delay_ms, mbps, 0.0, 0.0, 0.0)
    }

    /// A wire condition with a stochastic component, modeling jittery links
    /// (Wi-Fi, congested WAN) instead of a constant-latency pipe:
    ///
    /// - `delay_ms`: fixed one-way propagation delay per message.
    /// - `jitter_ms`: mean of an exponentially distributed extra delay added
    ///   per message (heavy-ish tail, like contention/retransmit variance).
    /// - `stall_ms`/`stall_p`: with probability `stall_p` a message is hit by
    ///   an additional `stall_ms` burst stall (radio retry storms, channel
    ///   scans). Later messages queue behind it FIFO, which matches the
    ///   head-of-line blocking of an ordered transport.
    pub fn with_jitter(
        delay_ms: f64,
        mbps: Option<f64>,
        jitter_ms: f64,
        stall_ms: f64,
        stall_p: f64,
    ) -> Result<Self> {
        if !delay_ms.is_finite() || delay_ms < 0.0 {
            bail!("downstream wire delay must be finite and non-negative");
        }
        if mbps.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            bail!("downstream wire mbps must be finite and greater than zero");
        }
        if !jitter_ms.is_finite() || jitter_ms < 0.0 {
            bail!("downstream wire jitter must be finite and non-negative");
        }
        if !stall_ms.is_finite() || stall_ms < 0.0 {
            bail!("downstream wire stall must be finite and non-negative");
        }
        if !stall_p.is_finite() || !(0.0..=1.0).contains(&stall_p) {
            bail!("downstream wire stall probability must be within [0, 1]");
        }
        if stall_p > 0.0 && stall_ms == 0.0 {
            bail!("downstream wire stall probability requires a stall duration");
        }
        // After the finiteness checks, so an infinite input reports what is
        // actually wrong with it rather than the magnitude bound.
        for (value, name) in [
            (delay_ms, "delay"),
            (jitter_ms, "jitter"),
            (stall_ms, "stall"),
        ] {
            if value > MAX_SIMULATED_DELAY_MS {
                bail!("downstream wire {name} must not exceed {MAX_SIMULATED_DELAY_MS} ms");
            }
        }
        Ok(Self {
            delay_ms,
            mbps,
            jitter_ms,
            stall_ms,
            stall_p,
        })
    }

    /// Samples the propagation delay for one message. With no stochastic
    /// component configured this is the constant `delay_ms` and draws nothing
    /// from the sample sequence.
    pub(crate) fn propagation_delay(&self) -> Duration {
        let mut delay_ms = self.delay_ms;
        if self.jitter_ms > 0.0 {
            // Inverse-CDF exponential sample with mean `jitter_ms`.
            let uniform = next_uniform_sample();
            delay_ms += -self.jitter_ms * (1.0 - uniform).ln();
        }
        if self.stall_p > 0.0 && next_uniform_sample() < self.stall_p {
            delay_ms += self.stall_ms;
        }
        // The exponential jitter tail is unbounded, so clamp the combined
        // delay: `Duration::from_secs_f64` panics on overflow.
        Duration::from_secs_f64(delay_ms.min(MAX_SIMULATED_DELAY_MS) / 1000.0)
    }

    fn sleep_for(&self, message: &StageWireMessage) {
        thread::sleep(self.propagation_delay());
        self.sleep_for_bandwidth(message);
    }

    /// Serialization delay for `bytes` on this link. A near-zero `mbps`
    /// makes the quotient arbitrarily large, so it is clamped to the same
    /// bound as the propagation delay: `Duration::from_secs_f64` panics on an
    /// out-of-domain value.
    pub(crate) fn bandwidth_delay(&self, bytes: usize) -> Duration {
        let Some(mbps) = self.mbps else {
            return Duration::ZERO;
        };
        let seconds = bytes as f64 / (mbps * 125_000.0);
        if !seconds.is_finite() || seconds <= 0.0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64((seconds * 1000.0).min(MAX_SIMULATED_DELAY_MS) / 1000.0)
    }

    fn sleep_for_bandwidth(&self, message: &StageWireMessage) {
        let delay = self.bandwidth_delay(message.estimated_wire_bytes());
        if !delay.is_zero() {
            thread::sleep(delay);
        }
    }
}

/// Hands each conditioned thread a distinct stream ordinal. A process-global
/// *draw* counter would make every stream depend on how the scheduler
/// interleaves the others; a per-thread draw index alone would hand every
/// thread the identical stream, which models synchronized loss rather than a
/// contended link. Salting the per-thread index with a per-thread ordinal
/// gives streams that are both reproducible and independent.
static WIRE_STREAM_ORDINALS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// This thread's stream ordinal, claimed on first draw.
    static WIRE_STREAM_ORDINAL: Cell<Option<u64>> = const { Cell::new(None) };
    /// This thread's draw index within its stream.
    static WIRE_SAMPLE_INDEX: Cell<u64> = const { Cell::new(0) };
}

/// Deterministic uniform sample in [0, 1) via splitmix64 over one stream's
/// draw index. Not cryptographic; just reproducible-enough conditioning for
/// benches and tests. Pure in `(stream, index)` so both properties the model
/// depends on — reproducibility within a lane, independence across lanes —
/// are directly testable.
fn uniform_sample(stream: u64, index: u64) -> f64 {
    let mut state = index
        .wrapping_mul(0x2545_F491_4F6C_DD1D)
        .wrapping_add(stream.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ WIRE_SAMPLE_SEED;
    state ^= state >> 30;
    state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^= state >> 31;
    (state >> 11) as f64 / (1u64 << 53) as f64
}

fn next_uniform_sample() -> f64 {
    let stream = WIRE_STREAM_ORDINAL.with(|ordinal| match ordinal.get() {
        Some(stream) => stream,
        None => {
            let stream = WIRE_STREAM_ORDINALS.fetch_add(1, Ordering::Relaxed);
            ordinal.set(Some(stream));
            stream
        }
    });
    let index = WIRE_SAMPLE_INDEX.with(|counter| {
        let index = counter.get();
        counter.set(index.wrapping_add(1));
        index
    });
    uniform_sample(stream, index)
}

pub(crate) fn write_stage_message_conditioned(
    writer: impl io::Write,
    message: &StageWireMessage,
    condition: WireCondition,
) -> io::Result<()> {
    condition.sleep_for(message);
    write_stage_message(writer, message)
}

pub(crate) fn write_stage_message_after_propagation(
    writer: impl io::Write,
    message: &StageWireMessage,
    condition: WireCondition,
) -> io::Result<()> {
    condition.sleep_for_bandwidth(message);
    write_stage_message(writer, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_condition_rejects_non_finite_or_negative_delay() {
        for delay_ms in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(WireCondition::new(delay_ms, None).is_err());
        }
    }

    #[test]
    fn wire_condition_rejects_non_finite_or_non_positive_bandwidth() {
        for mbps in [-1.0, 0.0, f64::NAN, f64::INFINITY] {
            assert!(WireCondition::new(0.0, Some(mbps)).is_err());
        }
    }

    #[test]
    fn wire_condition_rejects_invalid_jitter_and_stall_shapes() {
        for jitter_ms in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(WireCondition::with_jitter(0.0, None, jitter_ms, 0.0, 0.0).is_err());
        }
        for stall_ms in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(WireCondition::with_jitter(0.0, None, 0.0, stall_ms, 0.5).is_err());
        }
        for stall_p in [-0.1, 1.1, f64::NAN] {
            assert!(WireCondition::with_jitter(0.0, None, 0.0, 10.0, stall_p).is_err());
        }
        assert!(WireCondition::with_jitter(0.0, None, 0.0, 0.0, 0.5).is_err());
    }

    #[test]
    fn propagation_delay_is_exposed_without_bandwidth_serialization() {
        let condition = WireCondition::new(25.0, Some(100.0)).unwrap();

        assert_eq!(condition.propagation_delay(), Duration::from_millis(25));
    }

    #[test]
    fn constant_condition_never_draws_samples() {
        let condition = WireCondition::new(3.0, None).unwrap();
        let before = WIRE_SAMPLE_INDEX.with(Cell::get);
        let _ = condition.propagation_delay();
        assert_eq!(WIRE_SAMPLE_INDEX.with(Cell::get), before);
    }

    #[test]
    fn a_stream_is_reproducible_from_its_ordinal_and_index() {
        // Reproducibility: a lane replaying the same draws sees the same
        // sequence, with no dependence on other threads' interleaving.
        let first = (0..4)
            .map(|index| uniform_sample(7, index))
            .collect::<Vec<_>>();
        let second = (0..4)
            .map(|index| uniform_sample(7, index))
            .collect::<Vec<_>>();

        assert_eq!(first, second);
    }

    #[test]
    fn separate_streams_are_independent_not_identical() {
        // Independence: two lanes must not take their burst stalls on the
        // same message index, which is a synchronized-loss model rather than
        // the contended link the flag documents.
        let lanes = (0..4)
            .map(|stream| {
                (0..8)
                    .map(|index| uniform_sample(stream, index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for (left_index, left) in lanes.iter().enumerate() {
            for right in lanes.iter().skip(left_index + 1) {
                assert_ne!(left, right, "distinct streams must not share a sequence");
            }
        }
    }

    #[test]
    fn each_thread_claims_its_own_stream() {
        let condition = WireCondition::with_jitter(0.0, None, 5.0, 0.0, 0.0).unwrap();
        let sample_three = move || {
            (0..3)
                .map(|_| condition.propagation_delay())
                .collect::<Vec<_>>()
        };
        let first = thread::spawn(sample_three).join().expect("first thread");
        let second = thread::spawn(sample_three).join().expect("second thread");

        assert_ne!(first, second, "per-lane writer threads must decorrelate");
    }

    #[test]
    fn a_near_zero_rate_link_yields_a_bounded_bandwidth_delay() {
        let condition = WireCondition::with_jitter(0.0, Some(f64::MIN_POSITIVE), 0.0, 0.0, 0.0)
            .expect("a positive rate is accepted");

        // Unclamped this overflows `Duration::from_secs_f64` and panics.
        let delay = condition.bandwidth_delay(64 * 1024);

        assert_eq!(
            delay,
            Duration::from_secs_f64(MAX_SIMULATED_DELAY_MS / 1000.0)
        );
        assert_eq!(condition.bandwidth_delay(0), Duration::ZERO);
        assert_eq!(
            WireCondition::new(1.0, None).unwrap().bandwidth_delay(4096),
            Duration::ZERO
        );
    }

    #[test]
    fn jittered_condition_adds_a_bounded_positive_tail() {
        let condition = WireCondition::with_jitter(2.0, None, 5.0, 0.0, 0.0).unwrap();
        let base = Duration::from_millis(2);
        let mut above_base = 0usize;
        for _ in 0..256 {
            let sampled = condition.propagation_delay();
            assert!(sampled >= base);
            // An exponential with mean 5ms virtually never exceeds 200ms;
            // treat that as the sanity bound rather than an exact quantile.
            assert!(sampled < base + Duration::from_millis(200));
            if sampled > base {
                above_base += 1;
            }
        }
        assert!(above_base > 200, "jitter should almost always add delay");
    }

    #[test]
    fn stall_probability_gates_the_burst_component() {
        let never = WireCondition::with_jitter(1.0, None, 0.0, 50.0, 0.0).unwrap();
        for _ in 0..64 {
            assert_eq!(never.propagation_delay(), Duration::from_millis(1));
        }

        let always = WireCondition::with_jitter(1.0, None, 0.0, 50.0, 1.0).unwrap();
        for _ in 0..64 {
            assert_eq!(always.propagation_delay(), Duration::from_millis(51));
        }

        let sometimes = WireCondition::with_jitter(0.0, None, 0.0, 50.0, 0.25).unwrap();
        let stalled = (0..512)
            .filter(|_| sometimes.propagation_delay() >= Duration::from_millis(50))
            .count();
        assert!(
            (32..480).contains(&stalled),
            "stall rate {stalled}/512 is not plausibly 25%"
        );
    }
}
