//! Wire-facing projections for the runtime-event engine that live outside
//! `crate::runtime_events` (the engine itself) and `crate::api::routes`
//! (the SSE route/frame-encoding surface). `state_projection` is task 6's
//! (`.omo/plans/event-system-fixes.md`) home for the `runtime_state`
//! category projection: relocated here from
//! `api::routes::runtime_events::state_projection` so the frozen
//! `runtime_event_api::state_projection` test path resolves against a real
//! module.

pub(crate) mod state_projection;
