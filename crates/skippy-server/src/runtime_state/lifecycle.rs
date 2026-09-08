//! Optional, dependency-safe observation boundary for real session
//! lifecycle transitions [`super::RuntimeState`] already decides (plan
//! task 12, §8.7). Mirrors `kv_integration::lifecycle`'s shape exactly:
//! this crate defines the contract, a host runtime-event system
//! implements it, and `RuntimeState` never depends on host-runtime types.
//!
//! Implementations must return promptly: no I/O, no blocking, no panics.
//! A missing or slow observer never changes session/lane behavior.

/// One session-lifecycle transition, carrying bounded counts/durations
/// only. Never a session id, page id, or content of any kind.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SessionLifecycleEvent {
    /// `RuntimeState::drop_session_timed` reset a lane's native session
    /// and returned it to the idle pool (or discarded it because the idle
    /// pool was already full) -- the real "session reset" decision.
    SessionReset { reset_ms: f64 },
    /// `drop_session_timed`'s native `reset()` call failed, so the lane
    /// was discarded rather than reused -- the real "session
    /// abandoned/reclaimed" decision (the lane's capacity is reclaimed by
    /// the free-index pool rather than returned to service).
    SessionReclaimed,
    /// `RuntimeState::trim_session` (and its `align_session_to_token_count_if_ahead`
    /// caller) discarded native KV cells to move the session's tracked
    /// position backwards to `token_count` -- the real "session trimmed"
    /// decision.
    SessionTrimmed { token_count: u64 },
    /// `RuntimeState::export_state`/`export_full_state` returned `Ok`.
    RuntimeStateExportCompleted,
    /// `export_state`/`export_full_state` returned `Err`.
    RuntimeStateExportFailed,
    /// `RuntimeState::import_state`/`import_full_state`/
    /// `*_for_token_count` returned `Ok`.
    RuntimeStateImportCompleted,
    /// `import_state`/`import_full_state`/`*_for_token_count` returned
    /// `Err`.
    RuntimeStateImportFailed,
}

/// Optional sink for [`SessionLifecycleEvent`]s. Degrades silently on
/// failure, exactly like [`super::super::kv_integration::KvLifecycleObserver`].
pub trait SessionLifecycleObserver: Send + Sync {
    fn observe(&self, event: SessionLifecycleEvent);
}
