//! Synchronous producer boundary for dependency-leaf runtime facts.
//!
//! Implementations must return immediately without waiting for consumers.
//! Delivery class is derived from the fact, never supplied by a caller.

use crate::RuntimeFact;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitOutcome {
    Accepted,
    Coalesced,
    DroppedProgress,
    DroppedDiagnostic,
    RejectedShuttingDown,
    /// A new distinct state-transition key could not enter the bounded lane.
    /// No previously accepted state is evicted to make room.
    RejectedCapacity,
    /// A reservation-bound state-transition submission used a stale or
    /// explicitly cancelled generation and was refused without changing
    /// capacity health or reducer state.
    RejectedCancelled,
    TerminalDeliveryFailed,
}

pub trait RuntimeEventIngress: Send + Sync {
    fn try_submit(&self, fact: RuntimeFact) -> SubmitOutcome;
}
