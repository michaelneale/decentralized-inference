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

    pub fn prewarm_idle_sessions(
        &mut self,
        target_idle_sessions: usize,
    ) -> Result<RuntimeSessionStats> {
        let target_idle_sessions =
            capped_target_idle_sessions(target_idle_sessions, self.max_idle_sessions);
        while self.idle_sessions.len() < target_idle_sessions {
            if self.sessions.len() + self.idle_sessions.len() >= self.lane_count as usize {
                break;
            }
            let lane_session = self.create_lane_session()?;
            self.idle_sessions.push(lane_session);
        }
        Ok(self.session_stats())
    }

    /// Release the session slot identified by `session_id`.
    ///
    /// This is the cleanup path called at the end of every chat
    /// completion (success, cancellation, or backend error). It must
    /// leave [`Self`] in a self-consistent state regardless of whether
    /// the underlying StageSession can be reset cleanly:
    ///
    ///  - The lane is either returned to `idle_sessions` (reset OK) or
    ///    dropped entirely (reset failed). Dropping the lane triggers
    ///    `StageSession::drop`, which calls `skippy_session_free` on
    ///    the C side — the authoritative path for releasing native KV
    ///    cells held by that sequence id.
    ///  - `session_token_counts` and `session_resident_prefixes` for
    ///    `session_id` are always removed.
    ///  - The function always returns `Ok` so per-request cleanup at
    ///    callsites never propagates a reset failure as a request
    ///    error. The outcome is reported via [`RuntimeSessionDropStats`]
    ///    fields (`lane_discarded`, `lane_discard_reason`) for
    ///    telemetry.
    ///
    /// Previously a reset error propagated `?` through this function,
    /// which left `session_token_counts` holding stale entries and dropped the lane on the floor without
    /// any record. That accumulated bookkeeping drift over time and
    /// could leave the native KV cache reporting "all slots in use"
    /// long after the owning sessions were gone, producing
    /// `failed to find a memory slot` errors on subsequent admissions.
    pub fn drop_session_timed(&mut self, session_id: &str) -> Result<RuntimeSessionDropStats> {
        let reset_started = Instant::now();
        let mut reset_session = false;
        let preserved_resident_prefix = false;
        let mut lane_discarded = false;
        let mut lane_discard_reason: Option<String> = None;

        if let Some(mut lane_session) = self.sessions.remove(session_id) {
            let lane_index = lane_session.index;
            // Always release the lane's native KV cells back to the
            // unified pool. The trim+preserve path kept the lane's cells
            // pinned to a specific (`page_id`, `token_count`) pair so a
            // future request whose content prefix hashed to the *exact*
            // same `page_id` AND same `token_count` could acquire the
            // warm lane via `acquire_resident_prefix_lane`. Real chat /
            // agent workloads vary the conversation tail every turn, so
            // both the hash and the length change request-to-request and
            // that exact-match acquisition almost never fires. Meanwhile
            // the pinned cells remain claimed in the unified pool, in
            // parallel with the cells the cache layer itself pins, and
            // the pool runs out of contiguous space — producing
            // `decode: failed to find a memory slot` under repeated
            // tool-using agent traffic (#652). Cross-request prefix
            // reuse is still done by the cache layer (by `page_id`); we
            // just stop double-claiming cells on the lane side.
            self.session_resident_prefixes.remove(session_id);
            reset_session = true;
            let idle_pool_full = self
                .max_idle_sessions
                .is_some_and(|max| self.idle_sessions.len() >= max);
            match lane_session.session.reset() {
                Ok(()) if idle_pool_full => {
                    // The idle pool is already at model_fit.cache_idle_slots
                    // capacity: drop this lane (releasing its native KV
                    // cells via StageSession::drop) instead of growing the
                    // pool past the configured bound.
                    drop(lane_session);
                    self.free_lane_indices.push(lane_index);
                }
                Ok(()) => {
                    lane_session.resident_prefix = None;
                    self.idle_sessions.push(lane_session);
                }
                Err(reset_err) => {
                    lane_discarded = true;
                    let reason = format!("reset() failed ({reset_err:#})");
                    let _ = mesh_llm_events::emit_event(mesh_llm_events::OutputEvent::Warning {
                        message: "Discarding Skippy runtime lane after reset failure".to_string(),
                        context: Some(format!(
                            "lane_index={lane_index} session_id={session_id} reason={reason}"
                        )),
                    });
                    lane_discard_reason = Some(reason);
                    drop(lane_session);
                    self.free_lane_indices.push(lane_index);
                }
            }
        }

        // Always clear per-session bookkeeping. The previous version
        // skipped these when reset returned Err, which leaked entries.
        //
        // session_resident_prefixes is also cleared here defensively:
        // it's already removed above on the active-session path, but
        // calling drop_session_timed for an id that's no longer in
        // `sessions` (idempotent cleanup, stale callers) must still
        // clear any stray resident-prefix entry under that id.
        self.session_token_counts.remove(session_id);
        self.session_resident_prefixes.remove(session_id);

        Ok(RuntimeSessionDropStats {
            reset_session,
            reset_ms: reset_started.elapsed().as_secs_f64() * 1000.0,
            preserved_resident_prefix,
            lane_discarded,
            lane_discard_reason,
            stats_after: self.session_stats(),
        })
    }

    pub fn session_stats(&self) -> RuntimeSessionStats {
        let mut max_session_tokens = 0u64;
        let mut total_session_tokens = 0u64;
        let mut lanes = (0..self.lane_count as usize)
            .map(|index| RuntimeSessionLaneStats {
                index,
                active: false,
                session_id: None,
                token_count: None,
            })
            .collect::<Vec<_>>();

        for (session_id, lane_session) in &self.sessions {
            if let Some(token_count) = self.session_token_counts.get(session_id).copied() {
                max_session_tokens = max_session_tokens.max(token_count);
                total_session_tokens = total_session_tokens.saturating_add(token_count);
            }
            if let Some(lane) = lanes.get_mut(lane_session.index) {
                lane.active = true;
                lane.session_id = Some(session_id.clone());
                lane.token_count = self.session_token_counts.get(session_id).copied();
            }
        }

        RuntimeSessionStats {
            lane_count: self.lane_count as usize,
            active_sessions: self.sessions.len(),
            idle_sessions: self.idle_sessions.len(),
            idle_resident_prefixes: self
                .idle_sessions
                .iter()
                .filter(|idle| idle.resident_prefix.is_some())
                .count(),
            tracked_token_counts: self.session_token_counts.len(),
            max_session_tokens,
            total_session_tokens,
            lanes,
        }
    }

    pub(super) fn take_idle_session(&mut self) -> Option<RuntimeLaneSession> {
        if let Some(index) = self
            .idle_sessions
            .iter()
            .position(|idle| idle.resident_prefix.is_none())
        {
            return Some(self.idle_sessions.swap_remove(index));
        }
        self.idle_sessions.pop()
    }

    pub fn retain_resident_prefix_on_drop(
        &mut self,
        session_id: &str,
        page_id: String,
        token_count: u64,
    ) -> Result<()> {
        if !self.sessions.contains_key(session_id) {
            bail!("session {session_id} does not exist");
        }
        if self
            .session_resident_prefixes
            .get(session_id)
            .is_some_and(|current| current.token_count >= token_count)
        {
            return Ok(());
        }
        self.session_resident_prefixes.insert(
            session_id.to_string(),
            ResidentLanePrefix {
                page_id,
                token_count,
            },
        );
        Ok(())
    }

    pub fn acquire_resident_prefix_lane(
        &mut self,
        session_id: &str,
        page_id: &str,
        token_count: u64,
    ) -> Result<bool> {
        if self.sessions.contains_key(session_id) {
            bail!("session {session_id} already exists");
        }
        let Some(index) = self.idle_sessions.iter().position(|idle| {
            idle.resident_prefix.as_ref().is_some_and(|prefix| {
                prefix.page_id == page_id && prefix.token_count == token_count
            })
        }) else {
            return Ok(false);
        };
        let mut idle = self.idle_sessions.swap_remove(index);
        idle.resident_prefix = None;
        self.sessions.insert(session_id.to_string(), idle);
        self.session_token_counts
            .insert(session_id.to_string(), token_count);
        self.session_resident_prefixes.insert(
            session_id.to_string(),
            ResidentLanePrefix {
                page_id: page_id.to_string(),
                token_count,
            },
        );
        Ok(true)
    }

    pub fn has_session_range(&self, session_id: &str, token_start: u64, token_count: u64) -> bool {
        let Some(token_end) = token_start.checked_add(token_count) else {
            return false;
        };
        self.session_token_counts
            .get(session_id)
            .copied()
            .is_some_and(|known_tokens| token_end <= known_tokens)
    }

    pub fn export_kv_page(
        &mut self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<RuntimeKvPage> {
        self.validate_export_range(session_id, token_start, token_count)?;
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.export_kv_page(layer_start, layer_end, token_start, token_count)
    }

    pub fn probe_kv_page(
        &mut self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<RuntimeKvPageDesc> {
        self.validate_export_range(session_id, token_start, token_count)?;
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        let page = session.export_kv_page(layer_start, layer_end, token_start, token_count)?;
        Ok(page.desc)
    }

    pub fn import_kv_page(
        &mut self,
        session_id: &str,
        desc: &RuntimeKvPageDesc,
        bytes: &[u8],
    ) -> Result<()> {
        let session = self.session(session_id)?;
        session.import_kv_page(desc, bytes)?;
        let token_end = desc
            .token_start
            .checked_add(desc.token_count)
            .ok_or_else(|| anyhow::anyhow!("KV page token range overflows"))?;
        self.session_token_counts
            .entry(session_id.to_string())
            .and_modify(|current| *current = (*current).max(token_end))
            .or_insert(token_end);
        Ok(())
    }

    pub fn export_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.export_state(layer_start, layer_end)
    }

    pub fn import_state(&mut self, session_id: &str, bytes: &[u8]) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.import_state(layer_start, layer_end, bytes)
    }

    pub fn import_state_for_token_count(
        &mut self,
        session_id: &str,
        bytes: &[u8],
        token_count: u64,
    ) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.import_state_for_token_count(layer_start, layer_end, bytes, token_count)?;
        record_restored_session_token_count(
            &mut self.session_token_counts,
            session_id,
            token_count,
        );
        Ok(())
    }

    pub fn export_full_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.export_full_state(layer_start, layer_end)
    }

    pub fn import_full_state(&mut self, session_id: &str, bytes: &[u8]) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.import_full_state(layer_start, layer_end, bytes)
    }

    pub fn import_full_state_for_token_count(
        &mut self,
        session_id: &str,
        bytes: &[u8],
        token_count: u64,
    ) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        session.import_full_state_for_token_count(layer_start, layer_end, bytes, token_count)?;
        record_restored_session_token_count(
            &mut self.session_token_counts,
            session_id,
            token_count,
        );
        Ok(())
    }

    pub fn export_recurrent_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        self.session(session_id)?.export_recurrent_state()
    }

    /// Finalize a session's position after page imports on a family with no
    /// recurrent memory. Pure-attention state carries no snapshot to import,
    /// so the position is set directly and the tracked count follows it.
    pub fn set_session_position(&mut self, session_id: &str, token_count: u64) -> Result<()> {
        self.session(session_id)?.set_position(token_count)?;
        record_restored_session_token_count(
            &mut self.session_token_counts,
            session_id,
            token_count,
        );
        Ok(())
    }

    pub fn import_recurrent_state_for_token_count(
        &mut self,
        session_id: &str,
        bytes: &[u8],
        token_count: u64,
    ) -> Result<()> {
        self.session(session_id)?
            .import_recurrent_state_for_token_count(bytes, token_count)?;
        record_restored_session_token_count(
            &mut self.session_token_counts,
            session_id,
            token_count,
        );
        Ok(())
    }

    pub fn save_resident_prefix(
        &mut self,
        session_id: &str,
        cache_seq_id: i32,
        token_count: u64,
    ) -> Result<()> {
        self.session(session_id)?
            .save_prefix(cache_seq_id, token_count)
    }

    pub fn restore_resident_prefix(
        &mut self,
        session_id: &str,
        cache_seq_id: i32,
        token_ids: &[i32],
    ) -> Result<()> {
        let session = self.session(session_id)?;
        session.restore_prefix(cache_seq_id, token_ids)?;
        self.session_token_counts
            .insert(session_id.to_string(), token_ids.len() as u64);
        Ok(())
    }

    pub fn borrow_resident_prefix_session(
        &mut self,
        session_id: &str,
        cache_seq_id: i32,
        token_ids: &[i32],
    ) -> Result<()> {
        if self.sessions.contains_key(session_id) {
            bail!("session {session_id} already exists");
        }
        let model = &self.model;
        let (index, session) = create_indexed_lane_resource(
            &mut self.next_lane_index,
            &mut self.free_lane_indices,
            self.lane_count,
            || model.create_session_from_resident_prefix(cache_seq_id, token_ids),
        )?;
        let lane_session = RuntimeLaneSession {
            index,
            session,
            resident_prefix: None,
        };
        self.sessions.insert(session_id.to_string(), lane_session);
        self.session_token_counts
            .insert(session_id.to_string(), token_ids.len() as u64);
        Ok(())
    }

    pub fn drop_resident_prefix_sequence(
        &mut self,
        session_id: &str,
        cache_seq_id: i32,
    ) -> Result<()> {
        self.active_session(session_id)?.drop_sequence(cache_seq_id)
    }

    pub fn memory_used_cells(&mut self, session_id: &str) -> Result<u64> {
        self.active_session(session_id)?.memory_used_cells()
    }

    pub(super) fn add_session_tokens(&mut self, session_id: &str, count: u64) {
        self.session_token_counts
            .entry(session_id.to_string())
            .and_modify(|current| *current = current.saturating_add(count))
            .or_insert(count);
    }

    fn validate_export_range(
        &self,
        session_id: &str,
        token_start: u64,
        token_count: u64,
    ) -> Result<()> {
        let token_end = token_start
            .checked_add(token_count)
            .ok_or_else(|| anyhow::anyhow!("KV page token range overflows"))?;
        let known_tokens = self
            .session_token_counts
            .get(session_id)
            .copied()
            .unwrap_or_default();
        if token_end > known_tokens {
            bail!(
                "cannot export KV page [{token_start}, {token_end}) from session with {known_tokens} known tokens"
            );
        }
        Ok(())
    }

    fn model_layer_start(&self) -> u32 {
        self.layer_start
    }

    fn model_layer_end(&self) -> u32 {
        self.layer_end
    }

    pub(super) fn create_lane_session(&mut self) -> Result<RuntimeLaneSession> {
        let model = &self.model;
        let (index, session) = create_indexed_lane_resource(
            &mut self.next_lane_index,
            &mut self.free_lane_indices,
            self.lane_count,
            || model.create_session(),
        )?;
        Ok(RuntimeLaneSession {
            index,
            session,
            resident_prefix: None,
        })
    }
}

/// Clamps a requested idle-pool prewarm target to `model_fit.cache_idle_slots`
/// (`max_idle_sessions`). `None` preserves today's behavior: the target is
/// bounded only by `lane_count` in [`RuntimeState::prewarm_idle_sessions`].
pub(super) fn capped_target_idle_sessions(
    target_idle_sessions: usize,
    max_idle_sessions: Option<usize>,
) -> usize {
    match max_idle_sessions {
        Some(max) => target_idle_sessions.min(max),
        None => target_idle_sessions,
    }
}

fn record_restored_session_token_count(
    session_token_counts: &mut BTreeMap<String, u64>,
    session_id: &str,
    token_count: u64,
) {
    // A prefix restore can move an existing lane backwards to a shorter
    // common prefix. The tracked position must follow the imported native
    // state exactly; retaining the previous high-water mark submits the next
    // divergent token at the wrong position and makes llama_decode fail.
    session_token_counts.insert(session_id.to_string(), token_count);
}

/// Allocate the next lane slot.
///
/// Prefers indices in `free_lane_indices` (lanes previously discarded
/// via [`RuntimeState::drop_session_timed`]) so they can be reused
/// without growing `next_lane_index` past `lane_count`. If the free
/// list is empty, falls through to bumping `next_lane_index`. If both
/// are exhausted, returns "all execution lanes are busy".
///
/// If `create()` fails after popping from the free list, the index is
/// pushed back so a retry can reuse it. The high-water counter is only
/// bumped on success, matching the prior behavior.
fn create_indexed_lane_resource<T>(
    next_lane_index: &mut usize,
    free_lane_indices: &mut Vec<usize>,
    lane_count: u32,
    create: impl FnOnce() -> Result<T>,
) -> Result<(usize, T)> {
    if let Some(index) = free_lane_indices.pop() {
        let resource = match create() {
            Ok(resource) => resource,
            Err(err) => {
                // Return the freed index so the next allocation can
                // still reuse it.
                free_lane_indices.push(index);
                return Err(err);
            }
        };
        return Ok((index, resource));
    }
    if *next_lane_index >= lane_count as usize {
        bail!("all execution lanes are busy");
    }
    let index = *next_lane_index;
    let resource = create()?;
    *next_lane_index = index + 1;
    Ok((index, resource))
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Result, bail};

    #[derive(Default)]
    struct FakeRestoreRollback {
        cleanup_error: Option<&'static str>,
        reacquire_error: Option<&'static str>,
        position_error: Option<&'static str>,
        native_position: u64,
        token_count: u64,
        discarded: bool,
    }

    impl RestoreRollback for FakeRestoreRollback {
        fn drop_dirty_session(&mut self, _session_id: &str) -> Result<()> {
            match self.cleanup_error {
                Some(message) => bail!(message),
                None => Ok(()),
            }
        }

        fn reacquire_clean_session(&mut self, _session_id: &str) -> Result<()> {
            match self.reacquire_error {
                Some(message) => bail!(message),
                None => Ok(()),
            }
        }

        fn restored_native_position(&mut self, _session_id: &str) -> Result<u64> {
            match self.position_error {
                Some(message) => bail!(message),
                None => Ok(self.native_position),
            }
        }

        fn restored_token_count(&self, _session_id: &str) -> u64 {
            self.token_count
        }

        fn discard_dirty_session(&mut self, _session_id: &str) {
            self.discarded = true;
        }
    }

    #[test]
    fn restore_rollback_returns_original_error_after_proving_a_clean_lane() {
        let mut runtime = FakeRestoreRollback::default();

        let error = rollback_restore_failure(
            &mut runtime,
            "lane-a",
            anyhow::anyhow!("injected import failure"),
        );

        assert_eq!(
            error.to_string(),
            "cache restore transaction rolled back for session lane-a"
        );
        assert!(format!("{error:#}").contains("injected import failure"));
        assert!(!runtime.discarded);
    }

    #[test]
    fn restore_rollback_discards_when_cleanup_or_reacquire_fails() {
        for mut runtime in [
            FakeRestoreRollback {
                cleanup_error: Some("cleanup failed"),
                ..FakeRestoreRollback::default()
            },
            FakeRestoreRollback {
                reacquire_error: Some("reacquire failed"),
                ..FakeRestoreRollback::default()
            },
        ] {
            let error =
                rollback_restore_failure(&mut runtime, "lane-a", anyhow::anyhow!("restore failed"));
            assert!(format!("{error:#}").contains("restore failed"));
            assert!(runtime.discarded);
        }
    }

    #[test]
    fn restore_rollback_discards_unverifiable_or_dirty_lanes() {
        for mut runtime in [
            FakeRestoreRollback {
                position_error: Some("position unavailable"),
                ..FakeRestoreRollback::default()
            },
            FakeRestoreRollback {
                native_position: 1,
                ..FakeRestoreRollback::default()
            },
            FakeRestoreRollback {
                token_count: 1,
                ..FakeRestoreRollback::default()
            },
        ] {
            let error =
                rollback_restore_failure(&mut runtime, "lane-a", anyhow::anyhow!("restore failed"));
            assert!(format!("{error:#}").contains("restore failed"));
            assert!(runtime.discarded);
        }
    }

    #[test]
    fn prefix_restore_moves_tracked_position_backwards() {
        let mut token_counts = std::collections::BTreeMap::from([("lane-a".to_string(), 3_535)]);

        record_restored_session_token_count(&mut token_counts, "lane-a", 3_530);

        assert_eq!(token_counts.get("lane-a"), Some(&3_530));
    }

    #[test]
    fn create_indexed_lane_resource_keeps_index_available_when_creation_fails() {
        let mut next_lane_index = 0;
        let mut free_lane_indices: Vec<usize> = Vec::new();

        let error = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            2,
            || -> Result<()> { bail!("transient session creation failure") },
        )
        .expect_err("failed creation should propagate the original error");

        assert_eq!(error.to_string(), "transient session creation failure");
        assert_eq!(next_lane_index, 0);
        assert!(free_lane_indices.is_empty());

        let (index, resource) =
            create_indexed_lane_resource(&mut next_lane_index, &mut free_lane_indices, 2, || {
                Ok("lane")
            })
            .expect("successful retry should reuse the unconsumed lane index");

        assert_eq!(index, 0);
        assert_eq!(resource, "lane");
        assert_eq!(next_lane_index, 1);
    }

    #[test]
    fn create_indexed_lane_resource_reuses_freed_indices_before_growing() {
        // Simulate the wedge scenario: all lanes allocated, one lane
        // freed via the discard path, next allocation must reuse the
        // freed index rather than bailing with "all execution lanes
        // are busy".
        let mut next_lane_index = 0;
        let mut free_lane_indices: Vec<usize> = Vec::new();
        let lane_count = 2;

        // Allocate both lanes.
        let (a_idx, _) = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            lane_count,
            || Ok("a"),
        )
        .expect("first allocation should succeed");
        let (b_idx, _) = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            lane_count,
            || Ok("b"),
        )
        .expect("second allocation should succeed");
        assert_eq!(a_idx, 0);
        assert_eq!(b_idx, 1);
        assert_eq!(next_lane_index, 2);

        // Pool is full at the high-water mark. A third allocation must
        // fail.
        let error = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            lane_count,
            || Ok("c"),
        )
        .expect_err("allocating past lane_count should fail when no slots are free");
        assert!(error.to_string().contains("all execution lanes are busy"));

        // Discard one lane: the caller pushes its freed index onto the
        // free list (this is what drop_session_timed does on the
        // discard branch).
        free_lane_indices.push(a_idx);

        // The next allocation MUST reuse the freed index instead of
        // bailing. This is the wedge regression: previously
        // next_lane_index stayed at lane_count and every allocation
        // failed forever.
        let (reused_idx, _) = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            lane_count,
            || Ok("c"),
        )
        .expect("allocation must reuse a freed index, not stay wedged");
        assert_eq!(reused_idx, 0);
        assert_eq!(next_lane_index, 2);
        assert!(free_lane_indices.is_empty());
    }

    #[test]
    fn create_indexed_lane_resource_returns_freed_index_on_create_failure() {
        // If create() fails while consuming a freed index, the index
        // must go back onto the free list so a retry can use it.
        let mut next_lane_index = 1;
        let mut free_lane_indices: Vec<usize> = vec![0];

        let error = create_indexed_lane_resource(
            &mut next_lane_index,
            &mut free_lane_indices,
            2,
            || -> Result<()> { bail!("create failed mid-reuse") },
        )
        .expect_err("failed creation should propagate");
        assert_eq!(error.to_string(), "create failed mid-reuse");
        assert_eq!(next_lane_index, 1);
        assert_eq!(free_lane_indices, vec![0]);

        // A retry should now succeed using the same freed index.
        let (idx, _) =
            create_indexed_lane_resource(&mut next_lane_index, &mut free_lane_indices, 2, || {
                Ok("retry")
            })
            .expect("retry should succeed");
        assert_eq!(idx, 0);
        assert_eq!(next_lane_index, 1);
        assert!(free_lane_indices.is_empty());
    }

    #[test]
    fn capped_target_idle_sessions_clamps_to_the_configured_bound() {
        assert_eq!(capped_target_idle_sessions(10, Some(2)), 2);
        assert_eq!(capped_target_idle_sessions(1, Some(2)), 1);
    }

    #[test]
    fn capped_target_idle_sessions_is_unbounded_when_unset() {
        assert_eq!(capped_target_idle_sessions(10, None), 10);
    }
}
