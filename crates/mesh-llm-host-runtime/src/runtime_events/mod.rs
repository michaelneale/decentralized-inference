//! Host runtime-event engine: bounded slot-owned terminal ingress, the
//! minimal acknowledgement seam a reducer drains, bounded replay, bounded
//! in-process subscribers, coalesced health, and deterministic shutdown.
//!
//! Built on the dependency-leaf contracts in `mesh-llm-runtime-event-contracts`
//! (facts, identity, delivery classification, `RuntimeEventIngress`). This
//! module owns admission, the reservation table (the only terminal channel),
//! and the frozen bounds from the plan's engine-bounds table. State
//! reduction semantics (ordering, degradation policy, rebuild policy beyond
//! the minimal ack seam) belong to a later task.

pub mod config;
pub mod driver;
pub mod engine;
pub mod health;
mod ingress_latency;
pub mod presentation;
pub mod reducer;
pub mod replay;
pub mod reservation;
mod state;
pub mod subscribers;
pub mod telemetry;
pub mod wake;

#[cfg(test)]
pub use state::clear_runtime_event_engine;
pub use state::{
    clear_runtime_event_engine_if_owned, install_runtime_event_engine, runtime_event_engine,
};
