use std::time::Instant;

use anyhow::{Context, Result};
use skippy_cache::ExactStatePayload;

use crate::runtime_state::RuntimeState;

use super::{
    ExactStateExtra, ExactStateRecord, ExactStateRecordAdmission, ExactStateRestore,
    KvStageIntegration, PendingExactStateRecord, PrefillKvIdentity, StagePrefixCachePayload,
    records::add_reconstruct_stats,
};

fn l3_fill_claim_key(l3: &skippy_cache::L3Tier, location: &skippy_cache::L3Location) -> String {
    format!("{}:{}", l3.state_identity(), location.manifest_key)
}

impl KvStageIntegration {
    pub fn restore_exact_state(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
    ) -> Result<Option<ExactStateRestore>> {
        runtime.restore_transaction(session_id, |runtime| {
            self.restore_exact_state_inner(runtime, session_id, identities)
        })
    }

    fn restore_exact_state_inner(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
    ) -> Result<Option<ExactStateRestore>> {
        if !self.should_lookup() || !self.payload.is_exact_state() {
            return Ok(None);
        }
        for identity in identities {
            let lookup_started = Instant::now();
            let (lookup, entries) = {
                let mut radix = self
                    .radix
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let lookup = radix.acquire_recurrent(&identity.namespace, &identity.token_ids);
                let entries = radix.stats().recurrent_entries;
                (lookup, entries)
            };
            let Some(lookup) = lookup else {
                // Radix miss: the durable tier may still hold this prefix.
                // Runs inside the restore transaction, so a failed import
                // rolls the lane back exactly as a radix restore would.
                if let Some(restored) =
                    self.restore_from_l3(runtime, session_id, identity, lookup_started)?
                {
                    return Ok(Some(restored));
                }
                continue;
            };
            let lease = ExactStateLease {
                radix: std::sync::Arc::clone(&self.radix),
                namespace: identity.namespace.clone(),
                stored_tokens: lookup.stored_tokens.clone(),
            };
            let token_count = lookup.stored_tokens.len() as u64;
            let lookup_ms = lookup_started.elapsed().as_secs_f64() * 1000.0;
            let mut reconstruct_ms = 0.0;
            let mut reconstruct_bytes = 0u64;
            let mut reconstruct_blocks = 0usize;
            let mut kv_import_ms = 0.0;
            let mut recurrent_import_ms = 0.0;
            let mut deterministic_failure = false;
            let restore_result = (|| -> Result<bool> {
                match lookup.value.payload.kind().into() {
                    StagePrefixCachePayload::FullState => {
                        let (full_state, stats) = lookup
                            .value
                            .payload
                            .full_state_bytes_timed()
                            .context("reconstruct cached full-state payload")
                            .map_err(|error| {
                                mark_deterministic_failure(&mut deterministic_failure, error)
                            })?;
                        if full_state.is_empty() {
                            deterministic_failure = true;
                            return Err(anyhow::anyhow!("cached full-state payload is empty"));
                        }
                        add_reconstruct_stats(
                            &mut reconstruct_ms,
                            &mut reconstruct_bytes,
                            &mut reconstruct_blocks,
                            stats,
                        );
                        let import_started = Instant::now();
                        runtime.import_full_state_for_token_count(
                            session_id,
                            full_state.as_ref(),
                            token_count,
                        )?;
                        kv_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                    }
                    StagePrefixCachePayload::KvRecurrent => {
                        if let Some((kv, stats)) = lookup
                            .value
                            .payload
                            .kv_bytes_timed()
                            .context("reconstruct cached KV payload")
                            .map_err(|error| {
                                mark_deterministic_failure(&mut deterministic_failure, error)
                            })?
                        {
                            add_reconstruct_stats(
                                &mut reconstruct_ms,
                                &mut reconstruct_bytes,
                                &mut reconstruct_blocks,
                                stats,
                            );
                            if let Some(desc) = lookup.value.extra.kv_desc.as_ref() {
                                desc.validate_payload(kv.len()).map_err(|error| {
                                    mark_deterministic_failure(&mut deterministic_failure, error)
                                })?;
                                if desc.token_start != 0 || desc.token_count != token_count {
                                    deterministic_failure = true;
                                    return Err(anyhow::anyhow!(
                                        "cached KV page token range mismatch for exact-state checkpoint"
                                    ));
                                }
                                let import_started = Instant::now();
                                runtime.import_kv_page(session_id, desc, kv.as_ref())?;
                                kv_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                            } else if !kv.is_empty() {
                                deterministic_failure = true;
                                return Err(anyhow::anyhow!(
                                    "cached KV payload is missing its descriptor"
                                ));
                            }
                        }
                        let (recurrent, stats) = lookup
                            .value
                            .payload
                            .recurrent_state_bytes_timed()
                            .context("reconstruct cached recurrent payload")
                            .map_err(|error| {
                                mark_deterministic_failure(&mut deterministic_failure, error)
                            })?;
                        if recurrent.is_empty() && !self.dense_without_recurrent {
                            deterministic_failure = true;
                            return Err(anyhow::anyhow!("cached recurrent-state payload is empty"));
                        }
                        add_reconstruct_stats(
                            &mut reconstruct_ms,
                            &mut reconstruct_bytes,
                            &mut reconstruct_blocks,
                            stats,
                        );
                        let import_started = Instant::now();
                        if recurrent.is_empty() {
                            // Known-dense model: there is no snapshot to
                            // import, only a position to finalize.
                            runtime.set_session_position(session_id, token_count)?;
                        } else {
                            runtime.import_recurrent_state_for_token_count(
                                session_id,
                                recurrent.as_ref(),
                                token_count,
                            )?;
                        }
                        recurrent_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                    }
                    _ => return Ok(false),
                }
                Ok(true)
            })();
            let restored_payload = match restore_result {
                Ok(restored_payload) => restored_payload,
                Err(error) => {
                    drop(lease);
                    if deterministic_failure
                        && let Err(quarantine_error) = self.quarantine_exact_state_entry(
                            &identity.namespace,
                            &lookup.stored_tokens,
                            &lookup.value.page_id,
                        )
                    {
                        let _ =
                            mesh_llm_events::emit_event(mesh_llm_events::OutputEvent::Warning {
                                message: "Skippy exact-state quarantine failed".to_string(),
                                context: Some(format!(
                                    "page_id={} error={quarantine_error:#}",
                                    lookup.value.page_id
                                )),
                            });
                        return Err(error.context(format!(
                            "failed to fully quarantine corrupt exact-state entry: {quarantine_error:#}"
                        )));
                    }
                    return Err(error);
                }
            };
            if !restored_payload {
                drop(lease);
                continue;
            }
            let restored = ExactStateRestore {
                page_id: lookup.value.page_id,
                token_count: token_count as usize,
                payload_kind: lookup.value.payload.kind(),
                logical_bytes: lookup.logical_bytes,
                entries,
                reconstruct_ms,
                reconstruct_bytes,
                reconstruct_blocks,
                lookup_ms,
                kv_import_ms,
                recurrent_import_ms,
                source: "radix",
                fill_ms: 0.0,
                rewarm_enqueued: false,
            };
            drop(lease);
            return Ok(Some(restored));
        }
        Ok(None)
    }

    fn quarantine_exact_state_entry(
        &self,
        namespace: &str,
        tokens: &[i32],
        page_id: &str,
    ) -> Result<bool> {
        let removed = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove_recurrent_if(namespace, tokens, |entry| entry.page_id == page_id);
        let Some(entry) = removed else {
            return Ok(false);
        };
        entry.payload.release_from(
            &mut self
                .exact_blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )?;
        Ok(true)
    }

    pub fn record_exact_state(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
    ) -> Result<Option<ExactStateRecord>> {
        if !self.should_record() || !self.payload.is_exact_state() {
            return Ok(None);
        }
        let token_count = identity.identity.token_count;
        if token_count < self.checkpoint_policy.min_tokens {
            return Ok(None);
        }
        if !self.try_begin_record(&identity.page_id) {
            return Ok(None);
        }
        let already_recorded =
            match try_touch_exact_state(&self.radix, &identity.namespace, &identity.token_ids) {
                Ok(Some(already_recorded)) => already_recorded,
                Ok(None) => {
                    // Recording is optional. A background worker may hold this lock
                    // while hashing hundreds of MiB; never make inference wait for it.
                    self.finish_record(&identity.page_id);
                    return Ok(None);
                }
                Err(error) => {
                    self.finish_record(&identity.page_id);
                    return Err(error);
                }
            };
        if already_recorded {
            self.finish_record(&identity.page_id);
            return Ok(None);
        }
        // Avoid paying a potentially multi-hundred-MiB runtime export when the
        // bounded worker queue is already occupied. Admission remains best-effort:
        // another producer may win the race before `try_send` below.
        if !self.has_exact_state_record_capacity() {
            self.finish_record(&identity.page_id);
            return Ok(None);
        }
        let exported = match self.payload {
            StagePrefixCachePayload::FullState => {
                runtime.export_full_state(session_id).map(|state| {
                    (
                        ExactStatePayload::full_state(state),
                        ExactStateExtra::default(),
                    )
                })
            }
            StagePrefixCachePayload::KvRecurrent => (|| {
                let kv = match runtime.export_kv_page(session_id, 0, token_count) {
                    Ok(kv) => Some(kv),
                    Err(error) if is_native_kv_unavailable(&error) => None,
                    Err(error) => return Err(error),
                };
                let recurrent = match runtime.export_recurrent_state(session_id) {
                    Ok(recurrent) => recurrent,
                    // A known-dense model has no recurrent memory to export;
                    // its snapshot is legitimately empty.
                    Err(error)
                        if self.dense_without_recurrent && is_recurrent_unavailable(&error) =>
                    {
                        Vec::new()
                    }
                    Err(error) => return Err(error),
                };
                Ok((
                    ExactStatePayload::kv_recurrent(
                        kv.as_ref().map(|kv| kv.payload.clone()).unwrap_or_default(),
                        recurrent,
                    ),
                    ExactStateExtra {
                        kv_desc: kv.as_ref().map(|kv| kv.desc.clone()),
                    },
                ))
            })(),
            StagePrefixCachePayload::Disabled | StagePrefixCachePayload::ResidentKv => {
                self.finish_record(&identity.page_id);
                return Ok(None);
            }
        };
        let (payload, extra) = match exported {
            Ok(exported) => exported,
            Err(error) => {
                self.finish_record(&identity.page_id);
                return Err(error);
            }
        };
        if payload.byte_len() == 0 {
            // A dense model whose native KV export was unavailable has no
            // state component at all. Recording it would later restore as a
            // bare position advance over missing attention state.
            self.finish_record(&identity.page_id);
            return Ok(None);
        }
        let payload_kind = payload.kind();
        let logical_bytes = payload.byte_len();
        match self.enqueue_exact_state_record(PendingExactStateRecord {
            page_id: identity.page_id.clone(),
            payload,
            extra,
            namespace: identity.namespace.clone(),
            token_ids: identity.token_ids.clone(),
            l3_fill_claim: None,
        }) {
            ExactStateRecordAdmission::Queued => {
                // Recording owns the radix/blob locks while it hashes a potentially
                // multi-hundred-MiB payload. Telemetry must not turn that background
                // work back into request latency by waiting for cache stats here.
                let entries = self
                    .radix
                    .try_lock()
                    .ok()
                    .map(|radix| radix.stats().recurrent_entries)
                    .unwrap_or_default();
                let physical_bytes = self
                    .exact_blobs
                    .try_lock()
                    .ok()
                    .map(|blobs| blobs.physical_bytes())
                    .unwrap_or_default();
                Ok(Some(ExactStateRecord {
                    page_id: identity.page_id.clone(),
                    token_count: token_count as usize,
                    payload_kind,
                    stored: false,
                    logical_bytes,
                    physical_bytes,
                    entries,
                    evicted_entries: 0,
                    evicted_logical_bytes: 0,
                    dedupe: Default::default(),
                }))
            }
            ExactStateRecordAdmission::DroppedFull | ExactStateRecordAdmission::WorkerStopped => {
                Ok(None)
            }
        }
    }
}

impl KvStageIntegration {
    /// Fill a radix miss from the durable tier with the longest recorded
    /// prefix of the query, import it, and enqueue a radix re-warm so the
    /// next lookup hits RAM. `None` when there is no tier, nothing usable is
    /// stored, or another fill of the same entry is in flight: concurrent
    /// misses must not each read the entry from disk, so the loser prefills
    /// normally while the winner warms the radix for everyone.
    fn restore_from_l3(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
        lookup_started: Instant,
    ) -> Result<Option<ExactStateRestore>> {
        const MAX_PREFIX_PROBES: usize = 64;
        let Some(l3) = &self.l3 else {
            return Ok(None);
        };
        // Locate first (cheap index probes), then single-flight the expensive
        // load on the located entry's manifest key: same-length queries for
        // different prefixes never suppress each other, and different-length
        // queries resolving to one entry never load it twice.
        let location =
            match l3.locate_longest(&identity.namespace, &identity.token_ids, MAX_PREFIX_PROBES) {
                Ok(Some(location)) => location,
                // Nothing stored, or a corrupt / identity-mismatched entry.
                // Either way the miss path is the safe one; the tier has
                // recorded the reason for the status surface.
                Ok(None) | Err(_) => return Ok(None),
            };
        // Segment and manifest digests intentionally deduplicate bytes across
        // numerical states. A fill claim must not: one state's fill cannot
        // warm another state's radix namespace, even when their payload bytes
        // happen to be identical.
        let fill_claim = l3_fill_claim_key(l3, &location);
        {
            let mut inflight = self
                .inflight_fills
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !inflight.insert(fill_claim.clone()) {
                return Ok(None);
            }
        }
        let outcome =
            self.fill_and_import(runtime, session_id, identity, lookup_started, l3, &location);
        // On success the claim travels with the re-warm record and the worker
        // releases it once the entry is radix-resident. On any other outcome
        // release it here.
        let handed_to_worker = matches!(&outcome, Ok(Some(restored)) if restored.rewarm_enqueued);
        if !handed_to_worker {
            self.inflight_fills
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&fill_claim);
        }
        outcome
    }

    fn fill_and_import(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
        lookup_started: Instant,
        l3: &std::sync::Arc<skippy_cache::L3Tier>,
        location: &skippy_cache::L3Location,
    ) -> Result<Option<ExactStateRestore>> {
        let fill_started = Instant::now();
        // A load failure (corrupt segment, now quarantined) is a miss, not a
        // request failure. Import failures below do propagate: the transaction
        // rolls the lane back and the caller falls back to cold prefill.
        let Ok(fill) = l3.load(location) else {
            return Ok(None);
        };
        if fill.payload.byte_len() == 0 {
            return Ok(None);
        }
        let fill_ms = fill_started.elapsed().as_secs_f64() * 1000.0;
        let token_count = fill.token_count;
        let kv_desc: Option<skippy_runtime::RuntimeKvPageDesc> = fill
            .kv_desc_json
            .as_deref()
            .and_then(|json| serde_json::from_str(json).ok());
        let lookup_ms = lookup_started.elapsed().as_secs_f64() * 1000.0;
        let mut kv_import_ms = 0.0;
        let mut recurrent_import_ms = 0.0;
        match fill.payload.kind().into() {
            StagePrefixCachePayload::FullState => {
                let (full_state, _) = fill
                    .payload
                    .full_state_bytes_timed()
                    .context("reconstruct L3 full-state payload")?;
                if full_state.is_empty() {
                    return Ok(None);
                }
                let import_started = Instant::now();
                runtime.import_full_state_for_token_count(
                    session_id,
                    full_state.as_ref(),
                    token_count,
                )?;
                kv_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
            }
            StagePrefixCachePayload::KvRecurrent => {
                // Every check runs before the first import. Once bytes have
                // gone into the session, the only acceptable exit is `Err`,
                // which the transaction rolls back; an `Ok(None)` after a
                // partial import would hand a dirty lane to cold prefill.
                let kv = fill
                    .payload
                    .kv_bytes()
                    .context("reconstruct L3 KV payload")?;
                let recurrent = fill
                    .payload
                    .recurrent_state_bytes()
                    .context("reconstruct L3 recurrent payload")?;
                if recurrent.is_empty() && !self.dense_without_recurrent {
                    return Ok(None);
                }
                let kv_page = match (kv.as_ref(), kv_desc.as_ref()) {
                    (Some(kv), Some(desc)) => {
                        // Same fail-closed checks as a radix restore: a
                        // descriptor that does not describe these bytes, or a
                        // page that is not the whole prefix, is a miss.
                        if desc.validate_payload(kv.len()).is_err()
                            || desc.token_start != 0
                            || desc.token_count != token_count
                        {
                            return Ok(None);
                        }
                        Some((kv, desc))
                    }
                    (Some(kv), None) if !kv.is_empty() => return Ok(None),
                    _ => None,
                };

                if let Some((kv, desc)) = kv_page {
                    let import_started = Instant::now();
                    runtime.import_kv_page(session_id, desc, kv.as_ref())?;
                    kv_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
                }
                let import_started = Instant::now();
                if recurrent.is_empty() {
                    runtime.set_session_position(session_id, token_count)?;
                } else {
                    runtime.import_recurrent_state_for_token_count(
                        session_id,
                        recurrent.as_ref(),
                        token_count,
                    )?;
                }
                recurrent_import_ms = import_started.elapsed().as_secs_f64() * 1000.0;
            }
            _ => return Ok(None),
        }
        let logical_bytes = fill.payload.byte_len();
        let payload_kind = fill.payload.kind();
        // Re-warm the RAM tier off the request path. A drop is fine: the
        // disk copy stays authoritative. The fill claim rides along so the
        // worker releases it only once the entry is radix-resident.
        let admission = self.enqueue_exact_state_record(PendingExactStateRecord {
            page_id: identity.page_id.clone(),
            payload: fill.payload,
            extra: ExactStateExtra { kv_desc },
            namespace: identity.namespace.clone(),
            token_ids: identity.token_ids[..token_count as usize].to_vec(),
            l3_fill_claim: Some(l3_fill_claim_key(l3, location)),
        });
        let rewarm_enqueued = matches!(admission, ExactStateRecordAdmission::Queued);
        Ok(Some(ExactStateRestore {
            page_id: identity.page_id.clone(),
            token_count: token_count as usize,
            payload_kind,
            logical_bytes,
            entries: 0,
            reconstruct_ms: 0.0,
            reconstruct_bytes: 0,
            reconstruct_blocks: 0,
            lookup_ms,
            kv_import_ms,
            recurrent_import_ms,
            source: "l3",
            fill_ms,
            rewarm_enqueued,
        }))
    }
}

fn is_recurrent_unavailable(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string().contains("no recurrent memory"))
}

struct ExactStateLease {
    radix: std::sync::Arc<
        std::sync::Mutex<
            skippy_cache::UnifiedRadixCache<super::RadixResidentEntry, super::RadixExactEntry>,
        >,
    >,
    namespace: String,
    stored_tokens: Vec<i32>,
}

impl Drop for ExactStateLease {
    fn drop(&mut self) {
        let released = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .release_recurrent(&self.namespace, &self.stored_tokens);
        debug_assert!(released, "recurrent radix acquire/release must balance");
    }
}

fn try_touch_exact_state(
    radix: &std::sync::Mutex<
        skippy_cache::UnifiedRadixCache<super::RadixResidentEntry, super::RadixExactEntry>,
    >,
    namespace: &str,
    token_ids: &[i32],
) -> Result<Option<bool>> {
    match radix.try_lock() {
        Ok(mut radix) => Ok(Some(radix.recurrent_exact(namespace, token_ids).is_some())),
        Err(std::sync::TryLockError::WouldBlock) => Ok(None),
        Err(std::sync::TryLockError::Poisoned(poisoned)) => Ok(Some(
            poisoned
                .into_inner()
                .recurrent_exact(namespace, token_ids)
                .is_some(),
        )),
    }
}

fn is_native_kv_unavailable(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.contains("runtime memory type is not supported for native KV pages")
            || message.contains("runtime has no attention KV cache")
    })
}

fn mark_deterministic_failure(
    deterministic_failure: &mut bool,
    error: anyhow::Error,
) -> anyhow::Error {
    *deterministic_failure = true;
    error
}

impl StagePrefixCachePayload {
    pub(crate) fn is_exact_state(self) -> bool {
        matches!(self, Self::KvRecurrent | Self::FullState)
    }
}

impl From<skippy_cache::ExactStatePayloadKind> for StagePrefixCachePayload {
    fn from(kind: skippy_cache::ExactStatePayloadKind) -> Self {
        match kind {
            skippy_cache::ExactStatePayloadKind::FullState => Self::FullState,
            skippy_cache::ExactStatePayloadKind::KvRecurrent => Self::KvRecurrent,
            skippy_cache::ExactStatePayloadKind::RecurrentOnly => Self::Disabled,
        }
    }
}

impl From<StagePrefixCachePayload> for skippy_cache::ExactStatePayloadKind {
    fn from(payload: StagePrefixCachePayload) -> Self {
        match payload {
            StagePrefixCachePayload::FullState => Self::FullState,
            StagePrefixCachePayload::KvRecurrent => Self::KvRecurrent,
            StagePrefixCachePayload::Disabled | StagePrefixCachePayload::ResidentKv => {
                Self::FullState
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Mutex},
        time::{Duration, Instant},
    };

    use skippy_cache::UnifiedRadixCache;

    use super::try_touch_exact_state;

    type TestRadix = UnifiedRadixCache<
        crate::kv_integration::RadixResidentEntry,
        crate::kv_integration::RadixExactEntry,
    >;

    #[test]
    fn busy_exact_state_lock_skips_touch_without_waiting() {
        let cache = Arc::new(Mutex::new(TestRadix::new()));
        let locked = cache.clone();
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _guard = locked.lock().unwrap();
            locked_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        locked_rx.recv().unwrap();

        let started = Instant::now();
        assert_eq!(
            try_touch_exact_state(&cache, "namespace", &[1]).unwrap(),
            None
        );
        assert!(started.elapsed() < Duration::from_millis(100));

        release_tx.send(()).unwrap();
        holder.join().unwrap();
    }

    #[test]
    fn poisoned_exact_state_lock_recovers_without_panicking() {
        let cache = Arc::new(Mutex::new(TestRadix::new()));
        let poisoned = cache.clone();
        assert!(
            std::thread::spawn(move || {
                let _guard = poisoned.lock().unwrap();
                panic!("poison exact-state cache for test");
            })
            .join()
            .is_err()
        );

        assert_eq!(
            try_touch_exact_state(&cache, "namespace", &[1]).unwrap(),
            Some(false)
        );
    }
}
