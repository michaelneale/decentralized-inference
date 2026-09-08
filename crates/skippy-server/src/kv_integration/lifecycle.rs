//! Optional, dependency-safe observation boundary for
//! [`super::KvStageIntegration`]'s resident-prefix cache decisions (plan
//! task 12, §8.11). Mirrors the `GenerationReceiptSink`/
//! `GenerationLifecycleIngress` shape in `frontend::generation_receipt`:
//! this crate defines the contract, a host runtime-event system implements
//! it, and `KvStageIntegration` never depends on host-runtime types.
//!
//! Implementations must return promptly: no I/O, no blocking, no panics.
//! A missing or slow observer never changes cache behavior -- every real
//! call site here calls `observe` best-effort AFTER the cache decision is
//! already made, never as a gate on it.

/// One resident-prefix KV cache decision, carrying bounded counts only.
/// Never a cache key, page id, namespace, token ID, or content of any kind.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvLifecycleEvent {
    /// `KvStageIntegration::probe_resident_prefix` found a usable resident
    /// prefix. `matched_tokens` is the shared-prefix length; `resident_entries`
    /// is the cache's current entry count.
    CacheLookupHit {
        matched_tokens: usize,
        resident_entries: usize,
    },
    /// `probe_resident_prefix` found no usable resident prefix (disabled,
    /// below the minimum shared-prefix threshold, or a genuine miss).
    CacheLookupMiss,
    /// `restore_resident_prefix` actually restored a resident prefix onto a
    /// session's KV cells.
    PrefixRestored {
        restored_tokens: usize,
        resident_entries: usize,
    },
    /// `evict_resident_prefix_for_tokens` released resident-prefix entries.
    CacheEviction {
        evicted_entries: usize,
        evicted_tokens: u64,
    },
    /// `KvStageIntegration::from_config` began attempting real
    /// initialization (past every disabled/unsupported/not-applicable
    /// early return).
    KvInitStarted,
    /// `from_config` completed initialization and returned a usable
    /// integration.
    KvInitCompleted,
    /// `from_config` failed during real initialization (a propagated
    /// `Result::Err`, never a disabled/unsupported `Ok(None)`).
    KvInitFailed,
    /// `enqueue_exact_state_record` declined a record request (worker
    /// queue full or worker stopped). There is no matching "queued"
    /// signal: admission alone is not a completed record, and this
    /// contract deliberately never claims one -- only the real failure
    /// (never-recorded) outcome is reported here.
    ExactStateRecordFailed,
    /// `run_exact_state_record_job` (the async worker thread) actually
    /// wrote the record -- the real completion this family's `Completed`
    /// name refers to, distinct from mere queue admission.
    ExactStateRecordCompleted,
    /// `admit_resident_capacity` could not admit a request within the
    /// configured free-space watermark, even after evicting every
    /// releasable candidate. `admission_deficit_tokens` is the exact
    /// existing shortfall the capacity planner already computes.
    CapacityApproachingLimit { admission_deficit_tokens: u64 },
}

/// Optional sink for [`KvLifecycleEvent`]s. `try_observe` never returns an
/// error to the caller: an observer that cannot accept an event degrades
/// silently, exactly like [`super::super::frontend::GenerationLifecycleIngress`].
pub trait KvLifecycleObserver: Send + Sync {
    fn observe(&self, event: KvLifecycleEvent);
}
