//! Connection-shape classification: no-cursor, in-window, or a replay gap.
//!
//! Reads only the engine's already-public attachment (`process_instance`, the
//! captured frontier, replay snapshot, reducer, and health) — no reducer or
//! replay logic is reimplemented here.
//!
//! Task 9 (`.omo/plans/event-system-fixes.md`, defect D11): the
//! "did we miss anything, and if so is it a gap" decision for a
//! same-instance cursor now delegates to the shared replay read-time helper,
//! which enforces the AGE bound AT READ TIME (a frame can go stale while
//! the engine is otherwise idle, with no push ever running to trigger
//! push-time eviction) for both the live buffer and immutable attachments.

use std::time::Instant;

use crate::runtime_events::config::REPLAY_MAX_AGE;
use crate::runtime_events::engine::RuntimeEventAttachment;
use crate::runtime_events::replay::{ReplayFrame, ReplayLookup, classify_frames_after};

use super::cursor::{Cursor, CursorError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GapReason {
    StaleInstance,
    Evicted,
}

pub(super) struct Gap {
    pub(super) reason: GapReason,
    pub(super) requested: Cursor,
    pub(super) oldest_available: Option<u64>,
    pub(super) latest: Option<u64>,
}

pub(super) enum ConnectionShape {
    NoCursor,
    InWindow { frames: Vec<ReplayFrame> },
    Gap(Gap),
}

/// Classify against the immutable attachment captured while the engine's
/// publication guard was held. The route uses this form so its replay
/// decision, reducer state, health snapshot, and live subscription all share
/// one applied publication frontier.
pub(super) fn classify_attachment(
    attachment: &RuntimeEventAttachment,
    process_instance: mesh_llm_runtime_event_contracts::ProcessInstanceId,
    requested: Option<Cursor>,
) -> Result<ConnectionShape, CursorError> {
    classify_captured(
        process_instance,
        attachment.published_frontier,
        &attachment.replay,
        attachment.replay_evicted_through,
        attachment.rebuild_invalidated_through,
        requested,
    )
}

fn classify_captured(
    process_instance: mesh_llm_runtime_event_contracts::ProcessInstanceId,
    frontier: u64,
    replay: &[ReplayFrame],
    evicted_through: Option<u64>,
    rebuild_invalidated_through: Option<u64>,
    requested: Option<Cursor>,
) -> Result<ConnectionShape, CursorError> {
    let Some(cursor) = requested else {
        return Ok(ConnectionShape::NoCursor);
    };

    if cursor.process_instance != process_instance {
        // Purely diagnostic (what does THIS instance currently hold?), so
        // the plain push-time-only snapshot is enough here -- unlike the
        // same-instance path below, there is no "gap relative to this
        // cursor" decision to make against a foreign instance.
        return Ok(ConnectionShape::Gap(Gap {
            reason: GapReason::StaleInstance,
            requested: cursor,
            oldest_available: oldest_sequence(replay),
            latest: latest_sequence(replay),
        }));
    }

    if cursor.sequence > frontier {
        return Err(CursorError::Malformed);
    }

    let rebuild_gap =
        rebuild_invalidated_through.is_some_and(|invalidated| cursor.sequence <= invalidated);
    if frontier == cursor.sequence
        && !rebuild_gap
        && evicted_through.is_none_or(|evicted| cursor.sequence >= evicted)
    {
        // Nothing has ever been minted past this cursor for the current
        // instance, so there is nothing to look up in replay at all.
        return Ok(ConnectionShape::InWindow { frames: Vec::new() });
    }

    if rebuild_gap {
        return Ok(ConnectionShape::Gap(Gap {
            reason: GapReason::Evicted,
            requested: cursor,
            oldest_available: oldest_sequence(replay),
            latest: latest_sequence(replay),
        }));
    }

    let read = classify_frames_after(
        replay.iter(),
        cursor.sequence,
        evicted_through,
        Instant::now(),
        REPLAY_MAX_AGE,
    );
    match read.lookup {
        ReplayLookup::InWindow(frames) => Ok(ConnectionShape::InWindow { frames }),
        ReplayLookup::Evicted {
            oldest_available,
            latest,
        } => Ok(ConnectionShape::Gap(Gap {
            reason: GapReason::Evicted,
            requested: cursor,
            oldest_available,
            latest,
        })),
    }
}

fn oldest_sequence(snapshot: &[ReplayFrame]) -> Option<u64> {
    snapshot.first().map(|frame| frame.sequence.get())
}

fn latest_sequence(snapshot: &[ReplayFrame]) -> Option<u64> {
    snapshot.last().map(|frame| frame.sequence.get())
}
