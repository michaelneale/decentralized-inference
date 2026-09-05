use anyhow::{Context, Result};
use skippy_scheduler::{
    CapacityDemand, ComponentCapacitySnapshot, EvictableCacheEntry, plan_component_capacity,
    rank_eviction_candidates,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::runtime_state::RuntimeState;

use super::{
    KvLifecycleEvent, KvStageIntegration, PrefillKvIdentity, RadixResidentEntry,
    ResidentPrefixRecord, ResidentPrefixRestore, ResidentSequencePool, StagePrefixCachePayload,
    lock_resident_sequences,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ResidentPrefixEviction {
    pub target_tokens: u64,
    pub evicted_entries: usize,
    pub evicted_tokens: u64,
}

#[derive(Debug, Clone)]
struct ResidentCapacityReservationEntry {
    target_session_tokens: u64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ResidentCapacityReservations {
    entries: Arc<Mutex<BTreeMap<String, ResidentCapacityReservationEntry>>>,
}

impl ResidentCapacityReservations {
    pub(crate) fn stats(&self) -> (usize, u64) {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (
            entries.len(),
            entries
                .values()
                .map(|entry| entry.target_session_tokens)
                .fold(0, u64::saturating_add),
        )
    }
}

/// Keeps one request's projected KV demand visible until its runtime session
/// has either materialized those cells or completed cleanup.
pub(crate) struct ResidentCapacityReservation {
    reservations: ResidentCapacityReservations,
    session_id: String,
}

impl Drop for ResidentCapacityReservation {
    fn drop(&mut self) {
        self.reservations
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.session_id);
    }
}

struct ResidentRadixLease {
    radix: std::sync::Arc<
        std::sync::Mutex<
            skippy_cache::UnifiedRadixCache<super::RadixResidentEntry, super::RadixExactEntry>,
        >,
    >,
    namespace: String,
    stored_tokens: Vec<i32>,
}

impl Drop for ResidentRadixLease {
    fn drop(&mut self) {
        let released = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .release_resident(&self.namespace, &self.stored_tokens);
        debug_assert!(released, "resident radix acquire/release must balance");
    }
}

#[derive(Debug, Clone, Default)]
pub struct ResidentCapacityDecision {
    pub enabled: bool,
    pub capacity_known: bool,
    pub admitted: bool,
    pub capacity_tokens: u64,
    pub active_tokens: u64,
    pub physical_used_tokens: u64,
    pub pinned_tokens: u64,
    pub request_tokens: u64,
    pub inflight_reservations: usize,
    pub inflight_outstanding_tokens: u64,
    pub minimum_free_tokens: u64,
    pub target_free_tokens: u64,
    pub projected_free_tokens: u64,
    pub admission_deficit_tokens: u64,
    pub required_eviction_tokens: u64,
    pub evicted_entries: usize,
    pub evicted_tokens: u64,
    pub physical_evicted_tokens: u64,
    pub predicted_recompute_cost: u64,
}

impl KvStageIntegration {
    pub(crate) fn reserve_resident_capacity(
        &self,
        session_id: &str,
        target_session_tokens: u64,
    ) -> Result<Option<ResidentCapacityReservation>> {
        if self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(None);
        }
        let mut entries = self
            .resident_capacity_reservations
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if entries.contains_key(session_id) {
            anyhow::bail!("resident capacity reservation already exists for session {session_id}");
        }
        entries.insert(
            session_id.to_string(),
            ResidentCapacityReservationEntry {
                target_session_tokens,
            },
        );
        drop(entries);
        Ok(Some(ResidentCapacityReservation {
            reservations: self.resident_capacity_reservations.clone(),
            session_id: session_id.to_string(),
        }))
    }

    fn outstanding_resident_capacity_demand(&self, runtime: &RuntimeState) -> (usize, u64) {
        let entries = self
            .resident_capacity_reservations
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let outstanding_tokens = entries
            .iter()
            .map(|(session_id, entry)| {
                entry
                    .target_session_tokens
                    .saturating_sub(runtime.session_token_count(session_id).unwrap_or_default())
            })
            .fold(0, u64::saturating_add);
        (entries.len(), outstanding_tokens)
    }

    /// Admit one resident-KV operation against the native unified KV pool and
    /// release deterministic unreferenced prefixes needed to reach the healthy
    /// free-space watermark. Native sequence deletion always precedes radix
    /// mutation, and each deletion is reflected in the radix before physical
    /// occupancy is measured again.
    pub fn admit_resident_capacity(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        request_tokens: u64,
        minimum_free_tokens: u64,
        target_free_tokens: u64,
        protected_seq_id: Option<i32>,
    ) -> Result<ResidentCapacityDecision> {
        if self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(ResidentCapacityDecision {
                admitted: true,
                ..ResidentCapacityDecision::default()
            });
        }
        let capacity_tokens = u64::from(runtime.kv_pool_tokens());
        let active_tokens = runtime.session_stats().total_session_tokens;
        let (inflight_reservations, inflight_outstanding_tokens) =
            self.outstanding_resident_capacity_demand(runtime);
        let request_tokens = request_tokens.max(inflight_outstanding_tokens);
        if capacity_tokens == 0 {
            self.notify_kv_lifecycle(KvLifecycleEvent::CapacityApproachingLimit {
                admission_deficit_tokens: request_tokens,
            });
            return Ok(ResidentCapacityDecision {
                enabled: true,
                capacity_known: false,
                admitted: false,
                active_tokens,
                request_tokens,
                inflight_reservations,
                inflight_outstanding_tokens,
                minimum_free_tokens,
                target_free_tokens,
                admission_deficit_tokens: request_tokens,
                ..ResidentCapacityDecision::default()
            });
        }

        let mut radix = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let stats = radix.stats();
        let candidates = radix.resident_eviction_candidates();
        #[cfg(test)]
        let mut used_tokens = if runtime.is_modelless_for_test() {
            active_tokens.saturating_add(stats.resident_tokens)
        } else {
            runtime.ensure_session_active(session_id)?;
            runtime.memory_used_cells(session_id)?
        };
        #[cfg(not(test))]
        let mut used_tokens = {
            runtime.ensure_session_active(session_id)?;
            runtime.memory_used_cells(session_id)?
        };
        let initial_used_tokens = used_tokens;
        let occupied_after_request = used_tokens.saturating_add(request_tokens);
        let available_free = capacity_tokens.saturating_sub(occupied_after_request);
        let minimum_free = if minimum_free_tokens >= capacity_tokens {
            available_free
        } else {
            minimum_free_tokens
        };
        let target_free = if target_free_tokens >= capacity_tokens {
            available_free
        } else {
            target_free_tokens
        }
        .max(minimum_free)
        .min(capacity_tokens);
        let required_eviction_tokens = occupied_after_request
            .saturating_add(target_free)
            .saturating_sub(capacity_tokens);
        let mut ranked = candidates
            .iter()
            .filter(|candidate| Some(candidate.value.seq_id) != protected_seq_id)
            .map(|candidate| EvictableCacheEntry {
                id: candidate.value.seq_id.to_string(),
                units: resident_candidate_units(candidate),
                recompute_cost: candidate.value.recompute_cost,
                last_used: candidate.last_used,
            })
            .collect::<Vec<_>>();
        rank_eviction_candidates(&mut ranked);
        let mut decision = ResidentCapacityDecision {
            enabled: true,
            capacity_known: true,
            capacity_tokens,
            active_tokens,
            physical_used_tokens: initial_used_tokens,
            pinned_tokens: stats.resident_pinned_tokens,
            request_tokens,
            inflight_reservations,
            inflight_outstanding_tokens,
            minimum_free_tokens,
            target_free_tokens,
            required_eviction_tokens,
            ..ResidentCapacityDecision::default()
        };
        let mut sequences = lock_resident_sequences(&self.resident_sequences);
        for ranked_victim in ranked {
            if used_tokens
                .saturating_add(request_tokens)
                .saturating_add(target_free)
                <= capacity_tokens
            {
                break;
            }
            let victim_id = &ranked_victim.id;
            let Some(victim) = candidates
                .iter()
                .find(|candidate| candidate.value.seq_id.to_string() == *victim_id)
            else {
                anyhow::bail!("capacity planner selected missing resident victim {victim_id}");
            };
            #[cfg(test)]
            if !runtime.is_modelless_for_test() {
                runtime.drop_resident_prefix_sequence(session_id, victim.value.seq_id)?;
            }
            #[cfg(not(test))]
            runtime.drop_resident_prefix_sequence(session_id, victim.value.seq_id)?;
            let removed = radix
                .evict_resident_candidate(&victim.namespace, &victim.tokens)
                .with_context(|| {
                    format!(
                        "resident victim {} became unavailable after native deletion",
                        victim.value.seq_id
                    )
                })?;
            sequences.release(removed.value.seq_id)?;
            #[cfg(test)]
            let used_after = if runtime.is_modelless_for_test() {
                active_tokens.saturating_add(radix.stats().resident_tokens)
            } else {
                runtime.memory_used_cells(session_id)?
            };
            #[cfg(not(test))]
            let used_after = runtime.memory_used_cells(session_id)?;
            decision.evicted_entries = decision.evicted_entries.saturating_add(1);
            decision.evicted_tokens = decision
                .evicted_tokens
                .saturating_add(removed.value.token_count);
            decision.physical_evicted_tokens = decision
                .physical_evicted_tokens
                .saturating_add(used_tokens.saturating_sub(used_after));
            decision.predicted_recompute_cost = decision
                .predicted_recompute_cost
                .saturating_add(removed.value.recompute_cost);
            used_tokens = used_after;
        }
        decision.projected_free_tokens =
            capacity_tokens.saturating_sub(used_tokens.saturating_add(request_tokens));
        decision.admission_deficit_tokens = used_tokens
            .saturating_add(request_tokens)
            .saturating_add(minimum_free)
            .saturating_sub(capacity_tokens);
        decision.admitted = decision.admission_deficit_tokens == 0;
        if decision.evicted_entries > 0 {
            self.notify_kv_lifecycle(KvLifecycleEvent::CacheEviction {
                evicted_entries: decision.evicted_entries,
                evicted_tokens: decision.evicted_tokens,
            });
        }
        if !decision.admitted {
            self.notify_kv_lifecycle(KvLifecycleEvent::CapacityApproachingLimit {
                admission_deficit_tokens: decision.admission_deficit_tokens,
            });
        }
        self.verify_resident_ownership(radix.stats().resident_entries, sequences.stats().0)?;
        Ok(decision)
    }

    pub fn probe_resident_prefix(
        &self,
        identity: &PrefillKvIdentity,
    ) -> Option<ResidentPrefixRestore> {
        if !self.should_lookup() || self.payload != StagePrefixCachePayload::ResidentKv {
            return None;
        }
        let radix = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(radix_hit) = radix.peek_resident(&identity.namespace, &identity.token_ids) else {
            self.notify_kv_lifecycle(KvLifecycleEvent::CacheLookupMiss);
            return None;
        };
        if !self.meets_shared_prefix_min_tokens(radix_hit.matched_tokens) {
            self.notify_kv_lifecycle(KvLifecycleEvent::CacheLookupMiss);
            return None;
        }
        let entries = radix.stats().resident_entries;
        self.notify_kv_lifecycle(KvLifecycleEvent::CacheLookupHit {
            matched_tokens: radix_hit.matched_tokens,
            resident_entries: entries,
        });
        Some(ResidentPrefixRestore {
            page_id: radix_hit.value.page_id,
            token_count: radix_hit.matched_tokens,
            seq_id: radix_hit.value.seq_id,
            entries,
        })
    }

    pub fn restore_resident_prefix(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
        token_ids: &[i32],
    ) -> Result<Option<ResidentPrefixRestore>> {
        runtime.restore_transaction(session_id, |runtime| {
            self.restore_resident_prefix_inner(runtime, session_id, identities, token_ids)
        })
    }

    fn restore_resident_prefix_inner(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
        token_ids: &[i32],
    ) -> Result<Option<ResidentPrefixRestore>> {
        if !self.should_lookup() || self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(None);
        }
        for identity in identities {
            let radix_hit = {
                let mut radix = self
                    .radix
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let eligible = radix
                    .peek_resident(&identity.namespace, &identity.token_ids)
                    .is_some_and(|hit| self.meets_shared_prefix_min_tokens(hit.matched_tokens));
                if eligible {
                    radix.acquire_resident(&identity.namespace, &identity.token_ids)
                } else {
                    None
                }
            };
            let Some(radix_hit) = radix_hit else {
                continue;
            };
            let _lease = ResidentRadixLease {
                radix: std::sync::Arc::clone(&self.radix),
                namespace: identity.namespace.clone(),
                stored_tokens: radix_hit.stored_tokens.clone(),
            };
            let page_id = radix_hit.value.page_id.clone();
            let token_count = radix_hit.matched_tokens.min(token_ids.len());
            if token_count == 0 {
                continue;
            }
            let restore = runtime.restore_resident_prefix(
                session_id,
                radix_hit.value.seq_id,
                &token_ids[..token_count],
            );
            restore?;
            let resident_entries = self
                .radix
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .stats()
                .resident_entries;
            self.notify_kv_lifecycle(KvLifecycleEvent::PrefixRestored {
                restored_tokens: token_count,
                resident_entries,
            });
            return Ok(Some(ResidentPrefixRestore {
                page_id,
                token_count,
                seq_id: radix_hit.value.seq_id,
                entries: resident_entries,
            }));
        }
        Ok(None)
    }

    /// Evict enough resident-prefix entries to release `target_tokens` KV
    /// cells, or all currently releasable entries.
    pub fn evict_resident_prefix_for_tokens(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        target_tokens: u64,
    ) -> Result<ResidentPrefixEviction> {
        if self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(ResidentPrefixEviction::default());
        }
        let mut radix = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut sequences = lock_resident_sequences(&self.resident_sequences);
        let mut evicted_entries = 0usize;
        let mut evicted_tokens = 0u64;
        while evicted_tokens < target_tokens {
            let Some(removed) = evict_one_resident(&mut radix, &mut sequences, |seq_id| {
                runtime.drop_resident_prefix_sequence(session_id, seq_id)
            })?
            else {
                break;
            };
            evicted_entries = evicted_entries.saturating_add(1);
            evicted_tokens = evicted_tokens.saturating_add(removed.value.token_count);
        }
        self.verify_resident_ownership(radix.stats().resident_entries, sequences.stats().0)?;
        if evicted_entries > 0 {
            self.notify_kv_lifecycle(KvLifecycleEvent::CacheEviction {
                evicted_entries,
                evicted_tokens,
            });
        }
        Ok(ResidentPrefixEviction {
            target_tokens,
            evicted_entries,
            evicted_tokens,
        })
    }

    /// Evict only the resident-prefix cells needed to leave one native decode
    /// batch of headroom in the unified KV pool.
    ///
    /// The resident cache and active lanes share `n_ctx`. Treating `n_batch`
    /// as an unconditional eviction amount drains a healthy cache even when
    /// the pool already has ample room (and can erase every prefix when
    /// `n_batch` is larger than the resident working set). Account for both
    /// active-lane and resident-prefix occupancy first, then evict only the
    /// actual deficit. A zero pool size is the modelless/unknown-capacity
    /// fallback and conservatively reserves one complete decode batch.
    pub fn evict_resident_prefix_for_decode_batch(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
    ) -> Result<ResidentPrefixEviction> {
        let decode_batch_tokens = runtime.active_session_batch_size(session_id)? as u64;
        if runtime.kv_pool_tokens() == 0 {
            return self.evict_resident_prefix_for_tokens(runtime, session_id, decode_batch_tokens);
        }
        let decision = self.admit_resident_capacity(
            runtime,
            session_id,
            0,
            decode_batch_tokens,
            decode_batch_tokens,
            None,
        )?;
        if !decision.admitted {
            return self.evict_resident_prefix_for_tokens(runtime, session_id, decode_batch_tokens);
        }
        Ok(ResidentPrefixEviction {
            target_tokens: decision.required_eviction_tokens,
            evicted_entries: decision.evicted_entries,
            evicted_tokens: decision.evicted_tokens,
        })
    }

    pub fn record_resident_prefix(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identity: &PrefillKvIdentity,
        token_ids: &[i32],
    ) -> Result<Option<ResidentPrefixRecord>> {
        if !self.should_record() || self.payload != StagePrefixCachePayload::ResidentKv {
            return Ok(None);
        }
        let token_count = identity
            .identity
            .token_count
            .try_into()
            .unwrap_or(usize::MAX)
            .min(token_ids.len());
        if token_count == 0 || (token_count as u64) < self.checkpoint_policy.min_tokens {
            return Ok(None);
        }
        let layer_count = identity
            .identity
            .layer_end
            .saturating_sub(identity.identity.layer_start)
            .max(1);
        let estimated_bytes = resident_estimated_bytes(token_count as u64, layer_count);
        if (self.resident_config.max_bytes > 0 && estimated_bytes > self.resident_config.max_bytes)
            || (self.resident_config.max_resident_tokens > 0
                && token_count as u64 > self.resident_config.max_resident_tokens)
        {
            return Ok(None);
        }
        let mut evicted_entries = 0usize;
        let mut evicted_tokens = 0u64;
        let seq_id = {
            let mut radix = self
                .radix
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut sequences = lock_resident_sequences(&self.resident_sequences);
            if let Some(existing) =
                radix.resident_exact(&identity.namespace, &identity.token_ids[..token_count])
            {
                let stats = radix.stats();
                self.verify_resident_ownership(stats.resident_entries, sequences.stats().0)?;
                return Ok(Some(ResidentPrefixRecord {
                    page_id: existing.value.page_id,
                    token_count,
                    seq_id: existing.value.seq_id,
                    stored: false,
                    evicted_entries: 0,
                    evicted_tokens: 0,
                    entries: stats.resident_entries,
                    resident_tokens: stats.resident_tokens,
                }));
            }

            loop {
                let stats = radix.stats();
                // `resident_tokens` is a logical sum of every radix checkpoint
                // depth. Native sequence snapshots share the underlying KV
                // cells, so using that sum as physical occupancy evicts family
                // prefixes even when the unified pool still has room. Physical
                // capacity is enforced before restore/prefill by
                // `admit_resident_capacity`; recording an alias adds no cells.
                if !resident_index_over_capacity(self.resident_config, stats, estimated_bytes) {
                    break;
                }
                let Some(removed) = evict_one_resident(&mut radix, &mut sequences, |seq_id| {
                    runtime.drop_resident_prefix_sequence(session_id, seq_id)
                })?
                else {
                    return Ok(None);
                };
                evicted_entries = evicted_entries.saturating_add(1);
                evicted_tokens = evicted_tokens.saturating_add(removed.value.token_count);
            }

            sequences.allocate()?
        };
        if let Err(error) = runtime.save_resident_prefix(session_id, seq_id, token_count as u64) {
            let mut sequences = lock_resident_sequences(&self.resident_sequences);
            if let Err(release_error) = sequences.release(seq_id) {
                sequences.force_quarantine(seq_id);
                return Err(error).with_context(|| {
                    format!(
                        "quarantined resident sequence {seq_id} after native save and release failed: {release_error:#}"
                    )
                });
            }
            return Err(error);
        }
        let mut radix = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut sequences = lock_resident_sequences(&self.resident_sequences);
        let inserted = insert_saved_resident(
            &mut radix,
            &mut sequences,
            identity.namespace.clone(),
            &identity.token_ids[..token_count],
            estimated_bytes,
            RadixResidentEntry {
                page_id: identity.page_id.clone(),
                seq_id,
                token_count: token_count as u64,
                recompute_cost: (token_count as u64).saturating_mul(u64::from(layer_count)),
            },
            |seq_id| runtime.drop_resident_prefix_sequence(session_id, seq_id),
        )?;
        if !inserted {
            let existing = radix
                .resident_exact(&identity.namespace, &identity.token_ids[..token_count])
                .context("occupied resident radix entry disappeared after rejected insert")?;
            let stats = radix.stats();
            self.verify_resident_ownership(stats.resident_entries, sequences.stats().0)?;
            return Ok(Some(ResidentPrefixRecord {
                page_id: existing.value.page_id,
                token_count,
                seq_id: existing.value.seq_id,
                stored: false,
                evicted_entries,
                evicted_tokens,
                entries: stats.resident_entries,
                resident_tokens: stats.resident_tokens,
            }));
        }
        let stats = radix.stats();
        self.verify_resident_ownership(stats.resident_entries, sequences.stats().0)?;
        Ok(Some(ResidentPrefixRecord {
            page_id: identity.page_id.clone(),
            token_count,
            seq_id,
            stored: true,
            evicted_entries,
            evicted_tokens,
            entries: stats.resident_entries,
            resident_tokens: stats.resident_tokens,
        }))
    }
}

fn evict_one_resident(
    radix: &mut skippy_cache::UnifiedRadixCache<RadixResidentEntry, super::RadixExactEntry>,
    sequences: &mut ResidentSequencePool,
    mut drop_native: impl FnMut(i32) -> Result<()>,
) -> Result<Option<skippy_cache::RadixEviction<RadixResidentEntry>>> {
    let candidates = radix.resident_eviction_candidates();
    if candidates.is_empty() {
        return Ok(None);
    }
    let plan = plan_component_capacity(
        &ComponentCapacitySnapshot {
            component: "resident-kv-tokens".to_string(),
            capacity_units: candidates
                .iter()
                .map(resident_candidate_units)
                .fold(0u64, u64::saturating_add),
            active_units: 0,
            pinned_cache_units: 0,
            evictable_entries: candidates
                .iter()
                .map(|candidate| EvictableCacheEntry {
                    id: candidate.value.seq_id.to_string(),
                    units: resident_candidate_units(candidate),
                    recompute_cost: candidate.value.recompute_cost,
                    last_used: candidate.last_used,
                })
                .collect(),
        },
        CapacityDemand {
            request_units: 0,
            minimum_free_units: 0,
            target_free_units: 1,
        },
    );
    let victim_id = plan
        .victim_ids
        .first()
        .context("resident capacity planner returned no victim")?;
    let victim = candidates
        .iter()
        .find(|candidate| candidate.value.seq_id.to_string() == *victim_id)
        .context("resident capacity planner selected missing victim")?;
    drop_native(victim.value.seq_id)?;
    let removed = radix
        .evict_resident_candidate(&victim.namespace, &victim.tokens)
        .expect("selected radix resident victim should exist");
    debug_assert_eq!(removed.value.page_id, victim.value.page_id);
    sequences.release(removed.value.seq_id)?;
    Ok(Some(removed))
}

fn resident_candidate_units(
    candidate: &skippy_cache::RadixEvictionCandidate<RadixResidentEntry>,
) -> u64 {
    candidate
        .value
        .token_count
        .max(candidate.tokens.len() as u64)
}

fn insert_saved_resident(
    radix: &mut skippy_cache::UnifiedRadixCache<RadixResidentEntry, super::RadixExactEntry>,
    sequences: &mut ResidentSequencePool,
    namespace: String,
    tokens: &[i32],
    logical_bytes: u64,
    entry: RadixResidentEntry,
    mut drop_native: impl FnMut(i32) -> Result<()>,
) -> Result<bool> {
    let seq_id = entry.seq_id;
    match radix.insert_resident_if_vacant(namespace, tokens, logical_bytes, entry) {
        Ok(None) => Ok(true),
        Ok(Some(rejected)) => {
            if let Err(error) = drop_native(rejected.seq_id) {
                sequences.quarantine(rejected.seq_id)?;
                return Err(error).with_context(|| {
                    format!(
                        "quarantine native resident sequence {} after duplicate radix insert",
                        rejected.seq_id
                    )
                });
            }
            sequences.release(rejected.seq_id)?;
            Ok(false)
        }
        Err(error) => {
            if let Err(native_error) = drop_native(seq_id) {
                sequences.quarantine(seq_id)?;
                return Err(native_error).with_context(|| {
                    format!(
                        "roll back native resident sequence {seq_id} after radix insert failed: {error:#}"
                    )
                });
            }
            sequences.release(seq_id)?;
            Err(error)
        }
    }
}

fn resident_estimated_bytes(token_count: u64, layer_count: u32) -> u64 {
    token_count
        .saturating_mul(u64::from(layer_count))
        .saturating_mul(2)
}

fn resident_index_over_capacity(
    config: skippy_cache::ResidentCacheConfig,
    stats: skippy_cache::UnifiedRadixCacheStats,
    new_entry_bytes: u64,
) -> bool {
    stats.resident_entries.saturating_add(1) > config.max_entries
        || (config.max_bytes > 0
            && stats.resident_logical_bytes.saturating_add(new_entry_bytes) > config.max_bytes)
}

#[cfg(test)]
mod proactive_eviction_tests {
    use super::*;
    use crate::kv_integration::KvLifecycleObserver;
    use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload};
    use skippy_runtime::ModelStateKind;
    use std::sync::Arc;

    #[derive(Default)]
    struct RecordingObserver(Mutex<Vec<KvLifecycleEvent>>);

    impl KvLifecycleObserver for RecordingObserver {
        fn observe(&self, event: KvLifecycleEvent) {
            self.0.lock().unwrap().push(event);
        }
    }

    #[test]
    fn logical_checkpoint_depth_does_not_trigger_physical_kv_eviction() {
        let config = skippy_cache::ResidentCacheConfig {
            max_entries: 64,
            max_bytes: 0,
            min_tokens: 64,
            reserved_seq_count: 32,
            max_resident_tokens: 114_688,
        };
        let stats = skippy_cache::UnifiedRadixCacheStats {
            resident_entries: 35,
            resident_tokens: 140_000,
            ..skippy_cache::UnifiedRadixCacheStats::default()
        };

        assert!(!resident_index_over_capacity(config, stats, 8_000));
    }

    #[test]
    fn admitted_capacity_eviction_reports_the_removed_entries() {
        let config = StageConfig {
            ctx_size: 10,
            lane_count: 1,
            kv_cache: Some(StageKvCacheConfig {
                mode: StageKvCacheMode::LookupRecord,
                payload: StageKvCachePayload::ResidentKv,
                max_entries: 4,
                max_bytes: 0,
                min_tokens: 1,
                shared_prefix_stride_tokens: 1,
                shared_prefix_record_limit: 1,
            }),
            ..StageConfig::default()
        };
        let observer = Arc::new(RecordingObserver::default());
        let integration =
            KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Dense), None)
                .unwrap()
                .expect("resident cache should be enabled")
                .with_kv_lifecycle_observer(observer.clone());
        let seq_id = lock_resident_sequences(&integration.resident_sequences)
            .allocate()
            .unwrap();
        integration
            .radix
            .lock()
            .unwrap()
            .insert_resident(
                "stage",
                &[1, 2, 3, 4, 5, 6, 7, 8],
                8,
                RadixResidentEntry {
                    page_id: "page".to_string(),
                    seq_id,
                    token_count: 8,
                    recompute_cost: 8,
                },
            )
            .unwrap();

        let mut runtime =
            crate::runtime_state::RuntimeState::new_modelless_with_capacity_for_test(1, 10);
        let decision = integration
            .admit_resident_capacity(&mut runtime, "session", 5, 0, 0, None)
            .unwrap();

        assert!(decision.admitted);
        assert_eq!(decision.evicted_entries, 1);
        assert_eq!(decision.evicted_tokens, 8);
        assert_eq!(
            *observer.0.lock().unwrap(),
            vec![KvLifecycleEvent::CacheEviction {
                evicted_entries: 1,
                evicted_tokens: 8,
            }]
        );
    }

    #[test]
    fn native_drop_failure_preserves_the_radix_entry_and_sequence_id() {
        let mut radix = skippy_cache::UnifiedRadixCache::new();
        let mut sequences = ResidentSequencePool::new(4);
        let seq_id = sequences.allocate().unwrap();
        radix
            .insert_resident(
                "stage",
                &[1, 2, 3],
                3,
                RadixResidentEntry {
                    page_id: "page".to_string(),
                    seq_id,
                    token_count: 3,
                    recompute_cost: 3,
                },
            )
            .unwrap();

        let error = evict_one_resident(&mut radix, &mut sequences, |_| {
            anyhow::bail!("native drop failed")
        })
        .unwrap_err();

        assert_eq!(error.to_string(), "native drop failed");
        assert!(radix.resident_exact("stage", &[1, 2, 3]).is_some());
        assert_eq!(sequences.allocate().unwrap(), seq_id + 1);
    }

    #[test]
    fn active_resident_entry_has_no_releasable_eviction_candidate() {
        let mut radix = skippy_cache::UnifiedRadixCache::new();
        let mut sequences = ResidentSequencePool::new(1);
        let seq_id = sequences.allocate().unwrap();
        radix
            .insert_resident(
                "stage",
                &[1, 2, 3],
                3,
                RadixResidentEntry {
                    page_id: "page".to_string(),
                    seq_id,
                    token_count: 3,
                    recompute_cost: 3,
                },
            )
            .unwrap();
        radix.acquire_resident("stage", &[1, 2, 3]).unwrap();
        let mut dropped = false;

        let removed = evict_one_resident(&mut radix, &mut sequences, |_| {
            dropped = true;
            Ok(())
        })
        .unwrap();

        assert!(removed.is_none());
        assert!(!dropped);
        assert_eq!(radix.stats().resident_entries, 1);
    }

    #[test]
    fn zero_metadata_token_count_falls_back_to_radix_path_length() {
        let mut radix = skippy_cache::UnifiedRadixCache::new();
        let mut sequences = ResidentSequencePool::new(1);
        let seq_id = sequences.allocate().unwrap();
        radix
            .insert_resident(
                "stage",
                &[1, 2, 3],
                3,
                RadixResidentEntry {
                    page_id: "legacy-zero".to_string(),
                    seq_id,
                    token_count: 0,
                    recompute_cost: 0,
                },
            )
            .unwrap();
        let mut dropped = None;

        let removed = evict_one_resident(&mut radix, &mut sequences, |seq_id| {
            dropped = Some(seq_id);
            Ok(())
        })
        .unwrap()
        .expect("zero-metadata resident entry should remain evictable");

        assert_eq!(dropped, Some(seq_id));
        assert_eq!(removed.value.page_id, "legacy-zero");
        assert_eq!(radix.stats().resident_entries, 0);
    }

    #[test]
    fn resident_lease_releases_reference_during_unwind() {
        let radix =
            std::sync::Arc::new(std::sync::Mutex::new(skippy_cache::UnifiedRadixCache::new()));
        radix
            .lock()
            .unwrap()
            .insert_resident(
                "stage",
                &[1, 2, 3],
                3,
                RadixResidentEntry {
                    page_id: "page".to_string(),
                    seq_id: 1,
                    token_count: 3,
                    recompute_cost: 3,
                },
            )
            .unwrap();
        let hit = radix
            .lock()
            .unwrap()
            .acquire_resident("stage", &[1, 2, 3])
            .unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let radix = std::sync::Arc::clone(&radix);
            move || {
                let _lease = ResidentRadixLease {
                    radix,
                    namespace: "stage".to_string(),
                    stored_tokens: hit.stored_tokens,
                };
                panic!("restore panicked");
            }
        }));

        assert!(result.is_err());
        assert!(radix.lock().unwrap().lru_resident_candidate().is_some());
    }

    #[test]
    fn eviction_prefers_low_recompute_cost_over_lru_age() {
        let mut radix = skippy_cache::UnifiedRadixCache::new();
        let mut sequences = ResidentSequencePool::new(1);
        let expensive_seq_id = sequences.allocate().unwrap();
        radix
            .insert_resident(
                "stage",
                &[1, 2, 3],
                3,
                RadixResidentEntry {
                    page_id: "old-expensive".to_string(),
                    seq_id: expensive_seq_id,
                    token_count: 3,
                    recompute_cost: 300,
                },
            )
            .unwrap();
        let cheap_seq_id = sequences.allocate().unwrap();
        radix
            .insert_resident(
                "stage",
                &[4, 5, 6],
                3,
                RadixResidentEntry {
                    page_id: "new-cheap".to_string(),
                    seq_id: cheap_seq_id,
                    token_count: 3,
                    recompute_cost: 3,
                },
            )
            .unwrap();
        let mut dropped = None;

        let removed = evict_one_resident(&mut radix, &mut sequences, |seq_id| {
            dropped = Some(seq_id);
            Ok(())
        })
        .unwrap()
        .unwrap();

        assert_eq!(dropped, Some(cheap_seq_id));
        assert_eq!(removed.value.page_id, "new-cheap");
        assert!(radix.resident_exact("stage", &[1, 2, 3]).is_some());
    }

    #[test]
    fn radix_insert_failure_rolls_back_native_state_and_recycles_sequence_id() {
        let mut radix = skippy_cache::UnifiedRadixCache::new();
        let mut sequences = ResidentSequencePool::new(4);
        let seq_id = sequences.allocate().unwrap();
        let mut dropped = None;

        let error = insert_saved_resident(
            &mut radix,
            &mut sequences,
            "stage".to_string(),
            &[],
            0,
            RadixResidentEntry {
                page_id: "page".to_string(),
                seq_id,
                token_count: 0,
                recompute_cost: 0,
            },
            |seq_id| {
                dropped = Some(seq_id);
                Ok(())
            },
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "radix cache key must contain at least one token"
        );
        assert_eq!(dropped, Some(seq_id));
        assert_eq!(sequences.allocate().unwrap(), seq_id);
        assert_eq!(radix.stats().resident_entries, 0);
    }

    #[test]
    fn failed_native_rollback_quarantines_sequence_capacity() {
        let mut radix = skippy_cache::UnifiedRadixCache::new();
        let mut sequences = ResidentSequencePool::new(4);
        let seq_id = sequences.allocate().unwrap();

        let error = insert_saved_resident(
            &mut radix,
            &mut sequences,
            "stage".to_string(),
            &[],
            0,
            RadixResidentEntry {
                page_id: "page".to_string(),
                seq_id,
                token_count: 0,
                recompute_cost: 0,
            },
            |_| anyhow::bail!("native drop failed"),
        )
        .unwrap_err();

        assert!(format!("{error:#}").contains("native drop failed"));
        assert!(sequences.quarantined_seq_ids.contains(&seq_id));
        assert_eq!(sequences.allocate().unwrap(), seq_id + 1);
        assert_eq!(radix.stats().resident_entries, 0);
    }

    #[test]
    fn duplicate_sequence_release_fails_without_free_list_corruption() {
        let mut sequences = ResidentSequencePool::new(4);
        let seq_id = sequences.allocate().unwrap();

        sequences.release(seq_id).unwrap();
        let error = sequences.release(seq_id).unwrap_err();

        assert_eq!(
            error.to_string(),
            format!("resident prefix sequence id {seq_id} is not allocated")
        );
        assert_eq!(sequences.allocate().unwrap(), seq_id);
        assert_eq!(sequences.allocate().unwrap(), seq_id + 1);
    }

    #[test]
    fn duplicate_saved_resident_preserves_existing_native_sequence() {
        let mut radix = skippy_cache::UnifiedRadixCache::new();
        let mut sequences = ResidentSequencePool::new(4);
        let existing_seq_id = sequences.allocate().unwrap();
        radix
            .insert_resident(
                "stage",
                &[1, 2, 3],
                3,
                RadixResidentEntry {
                    page_id: "existing".to_string(),
                    seq_id: existing_seq_id,
                    token_count: 3,
                    recompute_cost: 3,
                },
            )
            .unwrap();
        let duplicate_seq_id = sequences.allocate().unwrap();
        let mut dropped = None;

        let inserted = insert_saved_resident(
            &mut radix,
            &mut sequences,
            "stage".to_string(),
            &[1, 2, 3],
            3,
            RadixResidentEntry {
                page_id: "duplicate".to_string(),
                seq_id: duplicate_seq_id,
                token_count: 3,
                recompute_cost: 3,
            },
            |seq_id| {
                dropped = Some(seq_id);
                Ok(())
            },
        )
        .unwrap();

        assert!(!inserted);
        assert_eq!(dropped, Some(duplicate_seq_id));
        let existing = radix.resident_exact("stage", &[1, 2, 3]).unwrap();
        assert_eq!(existing.value.page_id, "existing");
        assert_eq!(existing.value.seq_id, existing_seq_id);
        assert_eq!(sequences.allocate().unwrap(), duplicate_seq_id);
    }
}
