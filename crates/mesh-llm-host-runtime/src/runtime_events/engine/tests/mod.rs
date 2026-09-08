//! Deterministic acceptance tests for the host runtime-event engine.
//! Split by acceptance class; every test is synchronous, sleep-free, and
//! drives concurrency (where used) with real threads plus a `Barrier`.

mod cancellation;
mod capacity;
mod children;
mod class_bypass;
mod classes;
mod eviction;
mod fixtures;
mod lanes;
mod ordering;
mod rebuild;
mod shutdown;
mod telemetry;
mod terminal;
