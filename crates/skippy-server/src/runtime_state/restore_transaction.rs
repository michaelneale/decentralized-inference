use super::*;

trait RestoreRollback {
    fn drop_dirty_session(&mut self, session_id: &str) -> Result<()>;
    fn reacquire_clean_session(&mut self, session_id: &str) -> Result<()>;
    fn restored_native_position(&mut self, session_id: &str) -> Result<u64>;
    fn restored_token_count(&self, session_id: &str) -> u64;
    fn discard_dirty_session(&mut self, session_id: &str);
}

fn rollback_restore_failure(
    runtime: &mut impl RestoreRollback,
    session_id: &str,
    restore_error: anyhow::Error,
) -> anyhow::Error {
    if let Err(cleanup_error) = runtime.drop_dirty_session(session_id) {
        runtime.discard_dirty_session(session_id);
        return anyhow::anyhow!(
            "cache restore failed ({restore_error:#}); could not clean native session {session_id}: {cleanup_error:#}"
        );
    }

    if let Err(reacquire_error) = runtime.reacquire_clean_session(session_id) {
        runtime.discard_dirty_session(session_id);
        return anyhow::anyhow!(
            "cache restore failed ({restore_error:#}); could not reacquire a clean native session {session_id}: {reacquire_error:#}"
        );
    }

    let native_position = match runtime.restored_native_position(session_id) {
        Ok(position) => position,
        Err(position_error) => {
            runtime.discard_dirty_session(session_id);
            return anyhow::anyhow!(
                "cache restore failed ({restore_error:#}); could not verify clean native session {session_id}: {position_error:#}"
            );
        }
    };
    if native_position != 0 || runtime.restored_token_count(session_id) != 0 {
        runtime.discard_dirty_session(session_id);
        return anyhow::anyhow!(
            "cache restore failed ({restore_error:#}); rollback left native session {session_id} at position {native_position}"
        );
    }

    restore_error.context(format!(
        "cache restore transaction rolled back for session {session_id}"
    ))
}

impl RestoreRollback for RuntimeState {
    fn drop_dirty_session(&mut self, session_id: &str) -> Result<()> {
        self.drop_session_timed(session_id).map(|_| ())
    }

    fn reacquire_clean_session(&mut self, session_id: &str) -> Result<()> {
        self.ensure_session_active(session_id)
    }

    fn restored_native_position(&mut self, session_id: &str) -> Result<u64> {
        self.active_session(session_id)
            .and_then(|session| session.native_position())
    }

    fn restored_token_count(&self, session_id: &str) -> u64 {
        self.session_token_count(session_id).unwrap_or_default()
    }

    fn discard_dirty_session(&mut self, session_id: &str) {
        self.force_discard_session(session_id);
    }
}

impl RuntimeState {
    /// Run a cache restore as a transaction over the native session.
    ///
    /// State imports are not guaranteed to be atomic at the C ABI boundary:
    /// an import can populate one component and then fail while validating a
    /// later component or its position.  The caller must therefore never
    /// continue a cache-off prefill on the same session after an error.  Drop
    /// the affected lane, reacquire a fresh/reset lane, and verify both the
    /// Rust bookkeeping and native position before returning the original
    /// error.  If any part of that rollback cannot be proven, return an error
    /// that explicitly tells the caller not to continue on this lane.
    pub fn restore_transaction<T>(
        &mut self,
        session_id: &str,
        restore: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let result = restore(self);
        let Err(restore_error) = result else {
            return result;
        };

        Err(rollback_restore_failure(self, session_id, restore_error))
    }

    /// Remove a session even when the normal reset path itself failed.  The
    /// StageSession destructor is the native ABI's authoritative sequence
    /// release, so dropping the lane is safer than allowing a dirty session to
    /// be reused by a cache-off fallback.
    fn force_discard_session(&mut self, session_id: &str) {
        if let Some(lane_session) = self.sessions.remove(session_id) {
            let lane_index = lane_session.index;
            drop(lane_session);
            if !self.free_lane_indices.contains(&lane_index) {
                self.free_lane_indices.push(lane_index);
            }
        }
        self.session_token_counts.remove(session_id);
        self.session_resident_prefixes.remove(session_id);
    }
}

#[cfg(test)]
#[path = "restore_transaction/tests.rs"]
mod tests;
