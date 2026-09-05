//! `GET /api/runtime/events/v1` — restart-aware runtime-event SSE stream.
//!
//! Trusted-local, read-only (classified in `api::access`). This module owns
//! wire encoding, cursor transport, and connection-shape recovery ordering
//! only; it never rebuilds replay or reducer logic — those stay owned by
//! `crate::runtime_events` (task 3/4's host engine).
//!
//! `frames` is `pub(crate)` (task 9, `.omo/plans/event-system-fixes.md`,
//! defect D11): `runtime_events::engine::drain::apply_and_publish_fact`
//! calls `frames::event_frame` to pre-serialize each frame's wire bytes
//! exactly once, at push, instead of every subscriber re-encoding it on
//! delivery. `mod routes;`/`mod runtime_events;` one level up (`api/mod.rs`,
//! `api/routes/mod.rs`) are `pub(crate)` too, for the same reason: Rust
//! path visibility requires every segment along the way to be reachable,
//! not just the leaf item.

mod cursor;
pub(crate) mod frames;
mod reconnect;
mod recovery;
mod stream;

#[cfg(test)]
mod runtime_event_api_tests;

use tokio::net::TcpStream;

use super::super::MeshApi;
use super::super::http::respond_error;
use cursor::CursorError;

pub(super) async fn handle(
    stream: &mut TcpStream,
    _state: &MeshApi,
    path: &str,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    let Some(engine) = crate::runtime_events::runtime_event_engine() else {
        return respond_error(stream, 503, "runtime event engine is not running").await;
    };

    let peer_key = stream.peer_addr().ok().map(|addr| addr.ip());
    if !reconnect::record_attempt(peer_key) {
        return respond_error(stream, 429, "runtime event reconnect rate limit exceeded").await;
    }

    let requested_cursor = match cursor::resolve(path, raw_request) {
        Ok(cursor) => cursor,
        Err(CursorError::Malformed) => {
            return respond_error(stream, 400, "malformed runtime event cursor").await;
        }
    };

    let attachment = match engine.attach() {
        Ok(attachment) => attachment,
        Err(crate::runtime_events::subscribers::SubscribeError::CapacityReached) => {
            return respond_error(stream, 503, "maximum runtime event subscribers reached").await;
        }
    };

    let shape = match recovery::classify_attachment(
        &attachment,
        engine.process_instance(),
        requested_cursor,
    ) {
        Ok(shape) => shape,
        Err(CursorError::Malformed) => {
            return respond_error(stream, 400, "runtime event cursor is out of range").await;
        }
    };

    stream::run(stream, &engine, attachment, shape).await
}
