//! Bounded, immutable replay retention.
//!
//! Frames are appended only after the reducer has acknowledged a wake entry,
//! so replay never holds partially applied state. Retention is bounded on
//! all three frozen dimensions (frame count, age, bytes; first limit wins):
//! a push evicts from the front until every bound is satisfied.
//!
//! Task 9 (`.omo/plans/event-system-fixes.md`, defect D11): the byte
//! dimension now charges each retained frame the REAL length of its
//! pre-serialized `runtime_event` wire bytes (`ReplayFrame::wire_bytes`),
//! not a fixed per-frame memory-footprint estimate. The bytes are computed
//! exactly ONCE, at push, by `engine::drain::apply_and_publish_fact` (via
//! the same `api::routes::runtime_events::frames::event_frame` encoder the
//! wire uses) and shared -- by `Arc` clone, never re-encoded -- with every
//! subscriber that ever sees this frame, whether through a live broadcast
//! delivery or a replay-window catch-up on a fresh connection. See
//! [`ReplayBuffer::push`]'s doc comment for what charging REAL, VARIABLE
//! per-frame bytes (instead of a FIXED constant) changes about which
//! retention dimension can evict more than one frame in a single push.
//!
//! [`ReplayBuffer::frames_after`] additionally enforces the AGE bound AT
//! READ TIME, against the caller's own `now` -- not only at the moment of
//! the last push. Without this, a frame that outlives `REPLAY_MAX_AGE`
//! while the engine is otherwise idle (no new push ever runs to trigger
//! `push_at`'s own eviction loop) would sit in the buffer, still
//! technically "retained", and get served to any client that reconnects
//! asking for it -- silently violating the retention-window contract for
//! exactly the clients who waited the longest.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use mesh_llm_runtime_event_contracts::{EventSequence, OperationScope, RuntimeFact};

use super::config::{REPLAY_MAX_AGE, REPLAY_MAX_BYTES, REPLAY_MAX_FRAMES};

#[derive(Debug, Clone)]
pub struct ReplayFrame {
    pub sequence: EventSequence,
    pub rebuild_generation: u64,
    pub scope: OperationScope,
    pub fact: Arc<RuntimeFact>,
    pub recorded_at: Instant,
    /// The exact `runtime_event` SSE frame bytes for this fact, serialized
    /// ONCE at push (task 9, `.omo/plans/event-system-fixes.md`, defect
    /// D11) by `engine::drain::apply_and_publish_fact` via
    /// `api::routes::runtime_events::frames::event_frame` -- byte-for-byte
    /// identical to what that encoder has always produced (see
    /// `runtime_event_api_tests::sample_frames_fixture_is_byte_exact_for_every_frame_type`,
    /// unaffected by this change). Every consumer -- a live subscriber's
    /// `recv()` delivery and a fresh connection's replay-window catch-up
    /// alike -- writes these bytes directly instead of re-running
    /// `event_projection` + JSON encoding per delivery. Cloning a
    /// `ReplayFrame` clones this `Arc` (cheap refcount bump), never the
    /// underlying bytes.
    pub wire_bytes: Arc<[u8]>,
}

struct RetainedFrame {
    frame: ReplayFrame,
    byte_cost: usize,
}

struct Inner {
    frames: VecDeque<RetainedFrame>,
    total_bytes: usize,
    /// Highest published sequence removed from this buffer. Sequence holes
    /// caused by rejected/coalesced ingress are not replay gaps; only a
    /// cursor before an actually evicted publication is a gap.
    evicted_through: Option<u64>,
}

#[derive(Debug)]
pub struct ReplayBuffer {
    inner: Mutex<Inner>,
    capacity: usize,
    max_bytes: usize,
    max_age: Duration,
}

impl std::fmt::Debug for Inner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Inner")
            .field("frame_count", &self.frames.len())
            .field("total_bytes", &self.total_bytes)
            .finish()
    }
}

/// Read-time lookup result for [`ReplayBuffer::frames_after`]: either the
/// live frames strictly after the requested cursor, or an eviction signal
/// when the next expected frame is unavailable -- whether because a prior
/// push already evicted it (count/byte bound) or because THIS READ found
/// it older than the age bound even though push-time eviction has not run
/// since (task 9, `.omo/plans/event-system-fixes.md`, defect D11).
#[derive(Debug)]
pub enum ReplayLookup {
    /// Every retained, still-live frame strictly after the requested
    /// cursor, in sequence order. Empty when the caller is already caught
    /// up to the newest known sequence.
    InWindow(Vec<ReplayFrame>),
    /// The frame immediately after the requested cursor is unavailable.
    /// `oldest_available` is the oldest LIVE (non-stale) frame's sequence,
    /// if any survive; `latest` is the newest sequence physically
    /// retained (whether or not it is itself stale), for diagnostics.
    Evicted {
        oldest_available: Option<u64>,
        latest: Option<u64>,
    },
}

/// The result of a read-time replay lookup, including the retention
/// watermark learned from stale frames during this read. The live buffer
/// stores the returned watermark; an attachment classifier uses the same
/// value transiently because its captured replay is immutable.
#[derive(Debug)]
pub(crate) struct ReplayRead {
    pub(crate) lookup: ReplayLookup,
    pub(crate) evicted_through: Option<u64>,
}

/// Classify frames strictly after `cursor` against one immutable replay
/// snapshot. This is shared by the live [`ReplayBuffer`] and the connection
/// attachment path so an all-stale snapshot has identical caught-up and
/// behind-cursor semantics in both entry points.
pub(crate) fn classify_frames_after<'a, I>(
    replay: I,
    cursor: u64,
    evicted_through: Option<u64>,
    now: Instant,
    max_age: Duration,
) -> ReplayRead
where
    I: Iterator<Item = &'a ReplayFrame> + Clone,
{
    let latest = replay.clone().last().map(|frame| frame.sequence.get());
    let live_start = replay
        .clone()
        .position(|frame| now.saturating_duration_since(frame.recorded_at) <= max_age);
    let stale_through = match live_start {
        Some(0) => None,
        Some(start) => replay
            .clone()
            .nth(start - 1)
            .map(|frame| frame.sequence.get()),
        None => replay.clone().last().map(|frame| frame.sequence.get()),
    };
    let effective_evicted_through = match (evicted_through, stale_through) {
        (Some(evicted), Some(stale)) => Some(evicted.max(stale)),
        (Some(evicted), None) => Some(evicted),
        (None, Some(stale)) => Some(stale),
        (None, None) => None,
    };

    let lookup = if effective_evicted_through.is_some_and(|evicted| cursor < evicted) {
        ReplayLookup::Evicted {
            oldest_available: live_start
                .and_then(|start| replay.clone().nth(start).map(|frame| frame.sequence.get())),
            latest,
        }
    } else if let Some(start) = live_start {
        ReplayLookup::InWindow(
            replay
                .skip(start)
                .filter(|frame| frame.sequence.get() > cursor)
                .cloned()
                .collect(),
        )
    } else {
        ReplayLookup::InWindow(Vec::new())
    };

    ReplayRead {
        lookup,
        evicted_through: effective_evicted_through,
    }
}

impl ReplayBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(REPLAY_MAX_FRAMES)
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_bounds(capacity, REPLAY_MAX_BYTES, REPLAY_MAX_AGE)
    }

    /// Full three-dimensional constructor. `with_capacity` delegates here
    /// using the frozen `REPLAY_MAX_BYTES`/`REPLAY_MAX_AGE` values; tests
    /// use this directly to exercise the byte/age dimensions with small,
    /// deterministic bounds.
    #[must_use]
    pub fn with_bounds(capacity: usize, max_bytes: usize, max_age: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner {
                frames: VecDeque::with_capacity(capacity),
                total_bytes: 0,
                evicted_through: None,
            }),
            capacity,
            max_bytes,
            max_age,
        }
    }

    /// Append `frame`, then evict from the front while ANY of the three
    /// frozen bounds is exceeded (first limit wins — whichever bound is
    /// currently violated drives eviction, not all three at once). Returns
    /// the NUMBER of frames evicted (`0` when none were).
    ///
    /// Task 8-fix E1 (`.omo/plans/event-system-fixes.md`): this used to
    /// return `bool` ("did at least one eviction happen"), which the sole
    /// production caller (`engine::drain::apply_and_publish_fact`) turned
    /// into exactly one `EngineHealth::bump_replay_evicted()` regardless of
    /// how many frames were actually evicted. That undercounts: a single
    /// push CAN evict more than one frame. The frozen `replay_evicted`
    /// semantics are "one increment per evicted frame", so the caller must
    /// know the real count, not just "at least one".
    ///
    /// Task 9 CORRECTION (`.omo/plans/event-system-fixes.md`, defect D11):
    /// the reasoning that used to live here claimed the count/byte
    /// dimensions could never evict more than one frame per push "when
    /// frames are pushed one at a time, since each push's own eviction
    /// loop already restores the invariant before the next push can
    /// violate it again" -- true ONLY while `byte_cost` was the FIXED
    /// `APPROX_FRAME_BYTE_COST` constant task 8 used (so `total_bytes ==
    /// len * cost`, and one push could only ever add exactly `cost` bytes,
    /// meaning at most one eviction could restore the bound). That
    /// constant is gone: `byte_cost` is now the REAL, VARIABLE length of
    /// each frame's pre-serialized wire bytes (`ReplayFrame::wire_bytes`),
    /// so ONE large frame's push can add far more than any single small
    /// retained frame's worth of bytes -- satisfying the bound again can
    /// then require evicting SEVERAL smaller frames, not just the oldest.
    /// See
    /// `tests::a_single_large_frame_can_evict_multiple_smaller_frames_via_the_byte_bound`
    /// below for a same-push, byte-bound-only proof (`max_bytes` finite,
    /// `max_age` effectively infinite, so ONLY the byte dimension can be
    /// firing).
    ///
    /// The COUNT dimension is unaffected by this change and still can
    /// never evict more than one frame per push: each push adds exactly
    /// one frame to `frames.len()`, so a count-only overflow is always by
    /// exactly one. The AGE dimension's original reasoning is unaffected
    /// too -- see
    /// `tests::age_bound_evicts_every_frame_that_outlives_it_not_just_the_oldest`
    /// and
    /// `engine::drain::tests::a_single_push_that_evicts_several_stale_frames_credits_every_one_at_the_engine_level`
    /// (which pins `max_bytes: usize::MAX`, so only the age dimension can
    /// be the one firing there) -- both still pass unchanged.
    pub fn push(&self, frame: ReplayFrame) -> usize {
        self.push_at(frame, Instant::now())
    }

    /// Same as [`Self::push`] with an explicit "now", so age-bound eviction
    /// is deterministically testable without sleeping on the wall clock.
    pub(crate) fn push_at(&self, frame: ReplayFrame, now: Instant) -> usize {
        let byte_cost = frame.wire_bytes.len();
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        inner.frames.push_back(RetainedFrame { frame, byte_cost });
        inner.total_bytes += byte_cost;

        let mut evicted_count = 0usize;
        while self.exceeds_any_bound(&inner, now) {
            let Some(removed) = inner.frames.pop_front() else {
                break;
            };
            inner.total_bytes -= removed.byte_cost;
            inner.evicted_through = Some(
                inner
                    .evicted_through
                    .unwrap_or(removed.frame.sequence.get())
                    .max(removed.frame.sequence.get()),
            );
            evicted_count += 1;
        }
        evicted_count
    }

    fn exceeds_any_bound(&self, inner: &Inner, now: Instant) -> bool {
        if inner.frames.len() > self.capacity {
            return true;
        }
        if inner.total_bytes > self.max_bytes {
            return true;
        }
        inner.frames.front().is_some_and(|oldest| {
            now.saturating_duration_since(oldest.frame.recorded_at) > self.max_age
        })
    }

    /// Drop every retained frame, used by `rebuild()` to make a fresh
    /// generation's replay window coherent rather than mixing generations.
    pub fn evict_all(&self) -> usize {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let count = inner.frames.len();
        if let Some(last) = inner.frames.back() {
            inner.evicted_through = Some(
                inner
                    .evicted_through
                    .unwrap_or(last.frame.sequence.get())
                    .max(last.frame.sequence.get()),
            );
        }
        inner.frames.clear();
        inner.total_bytes = 0;
        count
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .frames
            .len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn snapshot(&self) -> Vec<ReplayFrame> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .frames
            .iter()
            .map(|retained| retained.frame.clone())
            .collect()
    }

    /// Highest sequence of a frame that has actually been removed from the
    /// replay window. This is distinct from the ingress frontier: rejected,
    /// coalesced, and dropped ingress sequences were never published and do
    /// not create a replay gap.
    #[must_use]
    pub fn evicted_through(&self) -> Option<u64> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .evicted_through
    }

    /// Look up every retained frame strictly after `cursor`, enforcing the
    /// AGE bound (`max_age`) AT READ TIME against `now` -- not only at the
    /// last push (task 9, `.omo/plans/event-system-fixes.md`, defect D11).
    /// `recorded_at` is non-decreasing front-to-back (frames are only ever
    /// appended in increasing-sequence order), so the still-live frames
    /// always form a contiguous SUFFIX of the retained deque: everything
    /// before that suffix is either already push-evicted (physically
    /// absent) or read-time-stale (physically present, but too old to
    /// serve). Read-time staleness advances the eviction watermark while
    /// retaining the physical frame for diagnostics; the returned
    /// `Vec<ReplayFrame>` is a one-shot snapshot of the matching frames --
    /// exactly like [`Self::snapshot`] already returns for the whole
    /// buffer, just filtered and age-checked in the same locked pass, not
    /// a persistent per-caller copy of the buffer itself.
    #[must_use]
    pub fn frames_after(&self, cursor: u64, now: Instant) -> ReplayLookup {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let replay = inner.frames.iter().map(|retained| &retained.frame);
        let read = classify_frames_after(replay, cursor, inner.evicted_through, now, self.max_age);
        inner.evicted_through = read.evicted_through;
        read.lookup
    }
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use mesh_llm_runtime_event_contracts::{FamilyFact, NativeRuntimeEventKind, OperationId};

    use super::*;

    fn frame(sequence: u64) -> ReplayFrame {
        frame_at(sequence, Instant::now())
    }

    fn frame_at(sequence: u64, recorded_at: Instant) -> ReplayFrame {
        ReplayFrame {
            sequence: EventSequence::new(sequence),
            rebuild_generation: 0,
            scope: OperationScope::root_only(OperationId::new()),
            fact: Arc::new(RuntimeFact::NativeRuntime(FamilyFact::new(
                NativeRuntimeEventKind::RuntimeStopped,
            ))),
            recorded_at,
            wire_bytes: Arc::from(Vec::new()),
        }
    }

    /// A frame whose `wire_bytes` is a REAL, controlled-size byte buffer
    /// (task 9, `.omo/plans/event-system-fixes.md`) -- not the removed
    /// fixed `APPROX_FRAME_BYTE_COST` estimate -- so byte-bound tests can
    /// pin exact eviction thresholds against known real sizes.
    fn frame_with_bytes(sequence: u64, byte_len: usize) -> ReplayFrame {
        let mut built = frame(sequence);
        built.wire_bytes = Arc::from(vec![0u8; byte_len]);
        built
    }

    #[test]
    fn push_beyond_capacity_evicts_the_oldest_frame() {
        let buffer = ReplayBuffer::with_capacity(2);
        assert_eq!(buffer.push(frame(0)), 0);
        assert_eq!(buffer.push(frame(1)), 0);
        assert_eq!(buffer.push(frame(2)), 1);

        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![1, 2]);
    }

    #[test]
    fn evict_all_clears_the_buffer_and_reports_the_prior_count() {
        let buffer = ReplayBuffer::with_capacity(4);
        buffer.push(frame(0));
        buffer.push(frame(1));

        assert_eq!(buffer.evict_all(), 2);
        assert!(buffer.is_empty());
    }

    #[test]
    fn push_beyond_the_byte_bound_evicts_based_on_real_frame_sizes() {
        // Each frame's wire_bytes is a REAL, controlled 100-byte buffer --
        // not the old fixed APPROX_FRAME_BYTE_COST estimate -- so a
        // 350-byte bound allows exactly 3 such frames before a 4th evicts
        // the oldest. 1,000 frames of headroom on the count dimension so
        // only the byte bound can be the one firing.
        let buffer = ReplayBuffer::with_bounds(1_000, 350, Duration::MAX);
        assert_eq!(buffer.push(frame_with_bytes(0, 100)), 0);
        assert_eq!(buffer.push(frame_with_bytes(1, 100)), 0);
        assert_eq!(buffer.push(frame_with_bytes(2, 100)), 0);
        assert_eq!(buffer.push(frame_with_bytes(3, 100)), 1);

        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![1, 2, 3]);
    }

    /// Proves the CRITICAL claim this task's brief demanded be checked:
    /// with REAL, VARIABLE per-frame byte costs, the byte dimension --
    /// like the age dimension, but UNLIKE the count dimension -- can now
    /// evict more than one frame in a single push. This was structurally
    /// impossible before task 9 (a fixed `APPROX_FRAME_BYTE_COST` meant
    /// `total_bytes == len * cost`, so one push could only ever add one
    /// frame's worth of cost, and evicting any single frame always
    /// restored the bound).
    #[test]
    fn a_single_large_frame_can_evict_multiple_smaller_frames_via_the_byte_bound() {
        // Three small (10-byte) frames fit comfortably under a 35-byte
        // bound (total 30 <= 35). A fourth, much larger (30-byte) frame
        // brings the total to 60 -- evicting only the oldest 10-byte frame
        // brings it to 50, still over; only evicting ALL THREE small
        // frames restores it to 30 (<= 35).
        let buffer = ReplayBuffer::with_bounds(1_000, 35, Duration::MAX);
        assert_eq!(buffer.push(frame_with_bytes(0, 10)), 0);
        assert_eq!(buffer.push(frame_with_bytes(1, 10)), 0);
        assert_eq!(buffer.push(frame_with_bytes(2, 10)), 0);

        let evicted = buffer.push(frame_with_bytes(3, 30));
        assert_eq!(
            evicted, 3,
            "one push must evict all three smaller frames to satisfy the \
             real byte bound now that per-frame cost is variable, not just \
             the single oldest one"
        );
        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![3]);
    }

    #[test]
    fn push_beyond_the_age_bound_evicts_stale_frames_even_under_the_frame_count_bound() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::from_secs(30));
        let start = Instant::now();
        assert_eq!(buffer.push_at(frame_at(0, start), start), 0);
        assert_eq!(
            buffer.push_at(
                frame_at(1, start + Duration::from_secs(10)),
                start + Duration::from_secs(10)
            ),
            0
        );

        // Sequence 0 is now 31s old (past the 30s age bound); sequence 1 is
        // only 21s old (still within it).
        let now = start + Duration::from_secs(31);
        let evicted = buffer.push_at(frame_at(2, now), now);
        assert_eq!(evicted, 1);
        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![1, 2]);
    }

    #[test]
    fn age_bound_evicts_every_frame_that_outlives_it_not_just_the_oldest() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::from_secs(10));
        let start = Instant::now();
        assert_eq!(buffer.push_at(frame_at(0, start), start), 0);
        assert_eq!(
            buffer.push_at(
                frame_at(1, start + Duration::from_secs(1)),
                start + Duration::from_secs(1)
            ),
            0
        );

        // Both prior frames are now well past the 10s bound; only the new
        // frame should remain.
        let now = start + Duration::from_secs(50);
        let evicted = buffer.push_at(frame_at(2, now), now);
        assert_eq!(
            evicted, 2,
            "one push must count BOTH stale frames it evicts here, not just \
             report that at least one eviction happened"
        );
        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![2]);
    }

    #[test]
    fn frame_count_bound_still_wins_when_it_is_the_tightest_dimension() {
        // Byte and age bounds are effectively unlimited here; only the
        // frame-count dimension is tight, proving first-limit-wins picks
        // whichever bound actually fires, not always the byte/age ones.
        let buffer = ReplayBuffer::with_bounds(2, usize::MAX, Duration::MAX);
        assert_eq!(buffer.push(frame(0)), 0);
        assert_eq!(buffer.push(frame(1)), 0);
        assert_eq!(buffer.push(frame(2)), 1);
        let remaining: Vec<u64> = buffer
            .snapshot()
            .iter()
            .map(|entry| entry.sequence.get())
            .collect();
        assert_eq!(remaining, vec![1, 2]);
    }

    // ─── frames_after: read-time lookup (task 9) ───────────────────────

    #[test]
    fn frames_after_returns_only_frames_strictly_after_the_cursor_while_fresh() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::from_secs(30));
        let start = Instant::now();
        assert_eq!(buffer.push_at(frame_at(5, start), start), 0);
        assert_eq!(buffer.push_at(frame_at(6, start), start), 0);

        match buffer.frames_after(5, start) {
            ReplayLookup::InWindow(frames) => {
                let sequences: Vec<u64> = frames.iter().map(|f| f.sequence.get()).collect();
                assert_eq!(sequences, vec![6]);
            }
            other => panic!("expected InWindow, got {other:?}"),
        }
    }

    #[test]
    fn frames_after_allows_published_sequence_holes_without_eviction() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::MAX);
        let start = Instant::now();
        assert_eq!(buffer.push_at(frame_at(5, start), start), 0);
        // Sequence 6 may have been rejected or coalesced before publication;
        // that ingress hole alone does not make sequence 7 unreplayable.
        assert_eq!(buffer.push_at(frame_at(7, start), start), 0);

        match buffer.frames_after(5, start) {
            ReplayLookup::InWindow(frames) => {
                assert_eq!(
                    frames
                        .iter()
                        .map(|frame| frame.sequence.get())
                        .collect::<Vec<_>>(),
                    vec![7]
                );
            }
            other => {
                panic!("expected a coalesced sequence hole to remain in-window, got {other:?}")
            }
        }
    }

    #[test]
    fn frames_after_reports_a_gap_when_the_oldest_retained_frame_is_ahead_of_the_cursor() {
        let buffer = ReplayBuffer::with_bounds(2, usize::MAX, Duration::MAX);
        assert_eq!(buffer.push(frame(0)), 0);
        assert_eq!(buffer.push(frame(1)), 0);
        assert_eq!(buffer.push(frame(2)), 1); // push-evicts sequence 0
        assert_eq!(buffer.push(frame(3)), 1); // push-evicts sequence 1

        // The client already has sequence 0 and wants sequence 1 onward,
        // but sequence 1 was itself push-evicted -- a genuine gap.
        match buffer.frames_after(0, Instant::now()) {
            ReplayLookup::Evicted {
                oldest_available,
                latest,
            } => {
                assert_eq!(oldest_available, Some(2));
                assert_eq!(latest, Some(3));
            }
            other => panic!(
                "expected Evicted (sequence 1 was push-evicted and the cursor wants it), got {other:?}"
            ),
        }
    }

    /// Acceptance bullet 2 (`.omo/plans/event-system-fixes.md`, task 9,
    /// defect D11): "a frame older than the age bound is reported evicted
    /// without a push". The frame here is NEVER push-evicted -- `len()`
    /// stays 1 throughout -- yet a read past the age bound must still
    /// report it evicted, purely from THIS read's own `now`.
    #[test]
    fn frames_after_reports_a_still_retained_but_stale_frame_as_evicted_without_any_push() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::from_secs(30));
        let start = Instant::now();
        assert_eq!(buffer.push_at(frame_at(5, start), start), 0);

        // Still fresh: in-window.
        match buffer.frames_after(4, start) {
            ReplayLookup::InWindow(frames) => {
                let sequences: Vec<u64> = frames.iter().map(|f| f.sequence.get()).collect();
                assert_eq!(sequences, vec![5]);
            }
            other => panic!("expected InWindow while fresh, got {other:?}"),
        }

        // 31s later -- past the 30s age bound -- with NO further push: the
        // frame is still physically retained (nothing push-evicted it),
        // but a read at this instant must report it evicted.
        let now = start + Duration::from_secs(31);
        match buffer.frames_after(4, now) {
            ReplayLookup::Evicted {
                oldest_available,
                latest,
            } => {
                assert_eq!(oldest_available, None, "no live frame remains to serve");
                assert_eq!(
                    latest,
                    Some(5),
                    "the physically-retained (if stale) frame's sequence is \
                     still reported as the latest known, for diagnostics"
                );
            }
            other => {
                panic!("expected Evicted once the only retained frame is stale, got {other:?}")
            }
        }
        assert_eq!(
            buffer.len(),
            1,
            "eviction here was a READ-TIME determination only -- no push \
             occurred, so the frame is still physically in the buffer"
        );
    }

    #[test]
    fn frames_after_all_stale_frames_is_in_window_when_cursor_is_caught_up() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::from_secs(30));
        let start = Instant::now();
        assert_eq!(buffer.push_at(frame_at(5, start), start), 0);

        match buffer.frames_after(5, start + Duration::from_secs(31)) {
            ReplayLookup::InWindow(frames) => assert!(frames.is_empty()),
            other => panic!("a caught-up cursor must remain in-window, got {other:?}"),
        }
    }

    #[test]
    fn frames_after_all_stale_frames_is_evicted_when_cursor_is_behind() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::from_secs(30));
        let start = Instant::now();
        assert_eq!(buffer.push_at(frame_at(5, start), start), 0);

        match buffer.frames_after(4, start + Duration::from_secs(31)) {
            ReplayLookup::Evicted {
                oldest_available,
                latest,
            } => {
                assert_eq!(oldest_available, None);
                assert_eq!(latest, Some(5));
            }
            other => {
                panic!("a behind cursor must report the stale frame as evicted, got {other:?}")
            }
        }
    }

    #[test]
    fn frames_after_on_a_pristine_empty_buffer_is_in_window() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::from_secs(30));
        match buffer.frames_after(0, Instant::now()) {
            ReplayLookup::InWindow(frames) => assert!(frames.is_empty()),
            other => panic!("expected InWindow on a pristine empty buffer, got {other:?}"),
        }
    }

    #[test]
    fn frames_after_on_an_empty_evicted_buffer_preserves_the_prior_watermark() {
        let buffer = ReplayBuffer::with_capacity(0);
        assert_eq!(buffer.push(frame(5)), 1);
        assert!(buffer.is_empty());

        match buffer.frames_after(5, Instant::now()) {
            ReplayLookup::InWindow(frames) => assert!(frames.is_empty()),
            other => panic!("a caught-up cursor must remain in-window, got {other:?}"),
        }
        match buffer.frames_after(4, Instant::now()) {
            ReplayLookup::Evicted {
                oldest_available,
                latest,
            } => {
                assert_eq!(oldest_available, None);
                assert_eq!(latest, None);
            }
            other => {
                panic!("a behind cursor must report prior empty-buffer eviction, got {other:?}")
            }
        }
    }

    #[test]
    fn frames_after_with_a_mixed_stale_and_live_buffer_only_excludes_the_stale_prefix() {
        let buffer = ReplayBuffer::with_bounds(1_000, usize::MAX, Duration::from_secs(10));
        let start = Instant::now();
        assert_eq!(buffer.push_at(frame_at(5, start), start), 0);
        assert_eq!(buffer.push_at(frame_at(6, start), start), 0);
        let fresh_at = start + Duration::from_secs(9);
        assert_eq!(buffer.push_at(frame_at(7, fresh_at), fresh_at), 0);

        // At `fresh_at + 2s` (11s after sequences 5/6 were recorded, 2s
        // after sequence 7), 5 and 6 are stale (>10s) but 7 is still live
        // (2s old). A cursor of 6 (client already has up to 6) must still
        // resolve in-window to just [7] -- no gap, since the client isn't
        // asking for the now-stale 5/6 anyway.
        let now = fresh_at + Duration::from_secs(2);
        match buffer.frames_after(6, now) {
            ReplayLookup::InWindow(frames) => {
                let sequences: Vec<u64> = frames.iter().map(|f| f.sequence.get()).collect();
                assert_eq!(sequences, vec![7]);
            }
            other => {
                panic!("expected InWindow (cursor already covers the stale prefix), got {other:?}")
            }
        }

        // But a cursor of 4 (client wants 5 onward) now hits a gap: 5/6
        // are stale/unavailable even though never push-evicted.
        match buffer.frames_after(4, now) {
            ReplayLookup::Evicted {
                oldest_available,
                latest,
            } => {
                assert_eq!(oldest_available, Some(7));
                assert_eq!(latest, Some(7));
            }
            other => panic!("expected Evicted (5/6 are stale, requested from 4), got {other:?}"),
        }
    }
}
