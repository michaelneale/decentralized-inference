//! Privacy-safe presentation projections for the host runtime-event engine.
//!
//! This module is the ONLY place a `RuntimeFact` is turned into an
//! `OutputEvent` for operational log / TUI presentation. It owns exactly
//! three responsibilities, kept in separate files so each stays small and
//! independently testable:
//!
//! - [`coalescer`]: bounded, latest-value-per-operation progress coalescing,
//!   entirely BEFORE anything reaches `mesh_llm_events::emit_event` -- the
//!   TUI's own `OutputCommand` channel (`mesh-llm-tui`) is never touched and
//!   never coalesces anything itself (see `subscriber`'s module doc for why
//!   that channel stays untouched by this task).
//! - [`projection`]: the privacy-safe, deny-by-default field allowlist that
//!   turns a `RuntimeFact` (or a coalesced `EngineHealthSnapshot`) into an
//!   `OutputEvent::Info`. Domain authority stays with the reducer (task 4)
//!   -- this module only renders, it never feeds anything back into the
//!   engine or reducer.
//! - [`subscriber`]: wires the two above to the engine's real
//!   `SubscriberRegistry` (task 3), routing by delivery class so Terminal
//!   facts and the coalesced health snapshot are reserved (forwarded
//!   immediately, every single one, never dropped or coalesced) while
//!   Progress facts coalesce to at most one per operation per render tick.
//!   `spawn_presentation_subscriber` is called from `runtime/run_auto.rs`
//!   right after the engine is installed, so this is a live, always-on
//!   consumer for the life of the process on the mesh-serve/TUI path (never
//!   on `local_model_only.rs` -- see the `subscriber` module doc).
//!
//! Operational logs (JSON/pretty stdout) and the TUI dashboard become
//! downstream projections of the runtime-event stream this way: both
//! consume the identical `OutputEvent::Info` value this module produces for
//! a given fact, so their content can never diverge (see
//! `tests::parity`).

mod coalescer;
mod projection;
mod subscriber;

#[cfg(test)]
mod tests;

pub use coalescer::ProgressCoalescer;
pub use projection::{fact_projection_event, health_projection_event};
pub use subscriber::{
    EmitEventSink, PresentationSink, attach, drive_presentation_subscriber,
    spawn_presentation_subscriber,
};
