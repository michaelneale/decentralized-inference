use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use mesh_llm_events::OutputEvent;
use skippy_cache::{
    CacheBlobStore, ResidentActivationCache, ResidentCacheConfig, SparseCheckpointPolicy,
    UnifiedRadixCache,
};
use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload};
use skippy_runtime::ModelStateKind;

use super::{
    EXACT_STATE_RECORD_CAPACITY, ExactStateByteLimits, KvLifecycleEvent, KvLifecycleObserver,
    KvStageIntegration, PendingExactStateRecord, RadixExactEntry, ResidentSequencePool,
    StageKvMode, StagePrefixCachePayload,
    model_capability::{ModelKvCapability, loaded_model_kv_capability},
    output_tokens::OutputTokenCache,
};

// Recurrent and hybrid payloads share the native n_ctx cell pool across
// sequence lanes, so their exact-state catalog must remain deliberately small.
const RECURRENT_CACHE_MAX_ENTRIES: usize = 16;

// Exact-state snapshots are indivisible and carry architecture-specific state
// (recurrent and convolution buffers) that `estimate_stage_cache_max_bytes`
// cannot see, because that estimate is derived from attention KV metadata
// alone. A single snapshot therefore routinely exceeds the soft cap. Keep this
// many snapshots (subject to the configured entry limit) before the soft cap
// evicts anything: falling to one entry
// turns the catalog into a single-entry cache, where two concurrent sessions
// on the same stage evict each other on every request.
const EXACT_STATE_MIN_RETAINED_ENTRIES: usize = 4;

// That retention floor is not allowed to grow without bound. Once the catalog
// crosses this multiple of the soft cap it evicts again, down to the single
// entry that keeps exact prefix reuse alive for the stage. That indivisible
// entry may itself exceed the limit.
const EXACT_STATE_HARD_CAP_MULTIPLE: u64 = 8;

// Operator override for the limit above, in bytes, for workers whose memory
// headroom does not match the attention-derived estimate. The last indivisible
// snapshot may exceed it. Zero disables the limit.
const EXACT_STATE_MAX_BYTES_ENV: &str = "SKIPPY_KV_CACHE_EXACT_MAX_BYTES";

impl KvStageIntegration {
    /// Opens KV integration for an ALREADY-LOADED model runtime.
    /// `model_state_kind` is the authoritative loaded-runtime state kind
    /// (Dense/Recurrent/Hybrid/Diffusion) -- never a file-path or
    /// tensor-name heuristic, since llama.cpp has already resolved the
    /// actual architecture by the time a model is loaded (see
    /// `model_capability::loaded_model_kv_capability`).
    ///
    /// `observer` accepts the lifecycle observer at CONSTRUCTION time --
    /// before any real initialization work happens -- so
    /// `KvInitStarted`/`KvInitCompleted`/`KvInitFailed` can be observed.
    /// Every disabled/unsupported/not-applicable early `Ok(None)` return
    /// above the real-initialization point never calls the observer at
    /// all: those are deliberate non-attempts, not failures, and this
    /// function does not misreport them as either. Pass `None` when no
    /// observer is available at the call site.
    pub fn from_loaded_model(
        config: &StageConfig,
        model_state_kind: Option<ModelStateKind>,
        observer: Option<Arc<dyn KvLifecycleObserver>>,
    ) -> Result<Option<Self>> {
        let Some(mut cache_config) = effective_cache_config(config) else {
            return Ok(None);
        };
        let mode = match cache_config.mode {
            StageKvCacheMode::Disabled | StageKvCacheMode::Auto => StageKvMode::Disabled,
            StageKvCacheMode::Record => StageKvMode::Record,
            StageKvCacheMode::LookupRecord => StageKvMode::LookupRecord,
        };
        if mode == StageKvMode::Disabled {
            return Ok(None);
        }
        let model_capability = loaded_model_kv_capability(model_state_kind);
        if let ModelKvCapability::Unknown(reason) = &model_capability {
            emit_cache_disabled_warning(config, reason);
            return Ok(None);
        }
        let payload = effective_cache_payload(cache_config.payload, &model_capability);
        if payload == StagePrefixCachePayload::Disabled {
            return Ok(None);
        }
        if matches!(model_capability, ModelKvCapability::KnownRecurrent)
            && matches!(payload, StagePrefixCachePayload::ResidentKv)
        {
            emit_cache_disabled_warning(
                config,
                "resident KV was requested for a recurrent-state model",
            );
            return Ok(None);
        }
        if matches!(model_capability, ModelKvCapability::KnownDense)
            && matches!(payload, StagePrefixCachePayload::KvRecurrent)
        {
            emit_cache_disabled_warning(
                config,
                "recurrent KV state was requested for a dense model",
            );
            return Ok(None);
        }
        // FullState is architecture-neutral: the native runtime serializes the
        // complete session state for both dense and recurrent model families.
        if matches!(model_capability, ModelKvCapability::KnownRecurrent) {
            cache_config.max_entries = cache_config.max_entries.min(RECURRENT_CACHE_MAX_ENTRIES);
        }
        let mut checkpoint_policy = SparseCheckpointPolicy::from_cache(&cache_config);
        let resident_config = ResidentCacheConfig::from_stage(config, &cache_config);
        if resident_config.max_entries == 0 {
            return Ok(None);
        }
        // Past this point every remaining early return is a genuine failure,
        // never a disabled/unsupported non-attempt -- this is where real
        // initialization (radix cache, blob store, worker thread) begins.
        if let Some(observer) = observer.as_ref() {
            observer.observe(KvLifecycleEvent::KvInitStarted);
        }
        // Activation checkpoints still use their own sparse policy; serving KV
        // contributes one full token path to the radix tree.
        checkpoint_policy.max_resident_tokens_hint = resident_config.max_resident_tokens;
        let exact_max_entries = cache_config.max_entries.clamp(1, 512);
        let exact_byte_limits = exact_state_byte_limits(
            cache_config.max_bytes,
            std::env::var(EXACT_STATE_MAX_BYTES_ENV).ok().as_deref(),
        );
        let radix = Arc::new(Mutex::new(UnifiedRadixCache::new()));
        let exact_blobs = Arc::new(Mutex::new(CacheBlobStore::default()));
        let (exact_state_record_tx, exact_state_record_rx) =
            std::sync::mpsc::sync_channel::<PendingExactStateRecord>(EXACT_STATE_RECORD_CAPACITY);
        let worker_radix = radix.clone();
        let worker_exact_blobs = exact_blobs.clone();
        let inflight_records = Arc::new(Mutex::new(BTreeSet::new()));
        let worker_inflight_records = inflight_records.clone();
        let exact_state_records_queued = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let exact_state_records_dropped = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let worker_exact_state_records_dropped = exact_state_records_dropped.clone();
        let exact_state_records_pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_exact_state_records_pending = exact_state_records_pending.clone();
        let exact_state_record_worker_healthy = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let worker_exact_state_record_worker_healthy = exact_state_record_worker_healthy.clone();
        let exact_state_record_worker_panics = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let worker_exact_state_record_worker_panics = exact_state_record_worker_panics.clone();
        let worker_observer = observer.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("skippy-exact-cache-{}", config.stage_id))
            .spawn(move || {
                while let Ok(pending) = exact_state_record_rx.recv() {
                    super::run_exact_state_record_job(
                        super::ExactStateWorkerHandles {
                            inflight_records: &worker_inflight_records,
                            dropped: &worker_exact_state_records_dropped,
                            pending_count: &worker_exact_state_records_pending,
                            worker_healthy: &worker_exact_state_record_worker_healthy,
                            worker_panics: &worker_exact_state_record_worker_panics,
                        },
                        worker_observer.as_ref(),
                        pending,
                        |pending| {
                            store_exact_radix_record(
                                &worker_radix,
                                &worker_exact_blobs,
                                exact_max_entries,
                                exact_byte_limits,
                                pending,
                            )
                        },
                    );
                }
            });
        if spawned.is_err()
            && let Some(observer) = observer.as_ref()
        {
            observer.observe(KvLifecycleEvent::KvInitFailed);
        }
        spawned?;
        if let Some(observer) = observer.as_ref() {
            observer.observe(KvLifecycleEvent::KvInitCompleted);
        }
        Ok(Some(Self {
            mode,
            payload,
            correctness_mode: false,
            trust_local_writes: true,
            checkpoint_policy,
            inflight_records,
            resident_config,
            resident_capacity_reservations: Default::default(),
            resident_sequences: Arc::new(Mutex::new(ResidentSequencePool::new(
                resident_config.reserved_seq_count,
            ))),
            activations: Arc::new(Mutex::new(ResidentActivationCache::new(resident_config))),
            radix,
            exact_blobs,
            exact_max_entries,
            exact_byte_limits,
            exact_state_record_tx,
            exact_state_records_queued,
            exact_state_records_dropped,
            exact_state_records_pending,
            exact_state_record_worker_healthy,
            exact_state_record_worker_panics,
            cache_healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            output_tokens: Arc::new(Mutex::new(OutputTokenCache::new(exact_max_entries))),
            split_prefill_tokens: Arc::new(Mutex::new(BTreeMap::new())),
            kv_lifecycle_observer: observer,
        }))
    }

    #[cfg(test)]
    pub(crate) fn from_config(
        config: &StageConfig,
        model_state_kind: ModelStateKind,
    ) -> Result<Option<Self>> {
        Self::from_loaded_model(config, Some(model_state_kind), None)
    }
}

fn emit_cache_disabled_warning(config: &StageConfig, reason: &str) {
    let _ = mesh_llm_events::emit_event(OutputEvent::Warning {
        message: "Skippy KV cache disabled for this model stage".to_string(),
        context: Some(format!(
            "stage_id={} model_id={} reason={reason}",
            config.stage_id, config.model_id
        )),
    });
}

fn store_exact_radix_record(
    radix: &Mutex<UnifiedRadixCache<super::RadixResidentEntry, RadixExactEntry>>,
    blobs: &Mutex<CacheBlobStore>,
    max_entries: usize,
    limits: ExactStateByteLimits,
    pending: PendingExactStateRecord,
) -> Result<()> {
    let logical_bytes = pending.payload.byte_len();
    let (payload, _) = pending.payload.dedupe_into(
        &mut blobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    // Cloning retains the Arc-backed blocks without changing blob-store
    // accounting, leaving `payload` available to roll that accounting back if
    // the radix rejects the insert.
    let mut released = Vec::new();
    let insert_result = {
        let mut radix = radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let insert_result = radix.insert_recurrent(
            pending.namespace,
            &pending.token_ids,
            logical_bytes,
            RadixExactEntry {
                page_id: pending.page_id,
                payload: payload.clone(),
                extra: pending.extra,
            },
        );
        match insert_result {
            Err(error) => Err(error),
            Ok(replaced) => {
                if let Some(replaced) = replaced {
                    released.push(replaced.payload);
                }
                while radix.stats().recurrent_entries > max_entries {
                    let Some(evicted) = radix.evict_lru_recurrent() else {
                        break;
                    };
                    released.push(evicted.value.payload);
                }
                Ok(())
            }
        }
    };
    if let Err(error) = insert_result {
        payload.release_from(
            &mut blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )?;
        return Err(error);
    }
    if !released.is_empty() {
        let mut blobs = blobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for payload in released {
            payload.release_from(&mut blobs)?;
        }
    }
    // Exact snapshots are indivisible, and the soft cap is estimated from
    // attention KV metadata that cannot include architecture-specific
    // recurrent state, so one snapshot can legitimately exceed it. Hold a small
    // working set past the soft cap (never more than the configured entry
    // limit) so concurrent sessions on a stage stop evicting each other, then
    // apply the hard limit so that allowance stays bounded apart from one
    // indivisible snapshot. Neither pass evicts the last entry: a snapshot that
    // self-evicts disables exact prefix caching for the stage entirely, which
    // is the regression this retention policy exists to prevent.
    evict_exact_entries_over(
        radix,
        blobs,
        limits.soft_bytes,
        EXACT_STATE_MIN_RETAINED_ENTRIES.min(max_entries),
    )?;
    evict_exact_entries_over(radix, blobs, limits.hard_bytes, 1)?;
    Ok(())
}

/// Evicts least-recently-used exact entries while the catalog holds more than
/// `limit_bytes`, never dropping below `min_retained_entries`. A zero limit
/// means the catalog has no byte ceiling and nothing is evicted.
fn evict_exact_entries_over(
    radix: &Mutex<UnifiedRadixCache<super::RadixResidentEntry, RadixExactEntry>>,
    blobs: &Mutex<CacheBlobStore>,
    limit_bytes: u64,
    min_retained_entries: usize,
) -> Result<()> {
    if limit_bytes == 0 {
        return Ok(());
    }
    let floor = min_retained_entries.max(1);
    loop {
        let over_bytes = blobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .physical_bytes()
            > limit_bytes;
        let above_floor = radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stats()
            .recurrent_entries
            > floor;
        if !over_bytes || !above_floor {
            break;
        }
        let evicted = radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .evict_lru_recurrent();
        let Some(evicted) = evicted else {
            break;
        };
        evicted.value.payload.release_from(
            &mut blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )?;
    }
    Ok(())
}

/// Resolves the exact-state byte budget from the stage cache cap.
///
/// `configured_max_bytes` is the attention-derived stage budget, which is used
/// as the soft cap. The hard limit defaults to a multiple of it so the working
/// set stays bounded without inheriting an estimate that structurally
/// undercounts exact-state payloads. One indivisible snapshot may remain above
/// that limit. `override_max_bytes` is the operator escape hatch and wins
/// outright when it parses; zero means unbounded.
fn exact_state_byte_limits(
    configured_max_bytes: u64,
    override_max_bytes: Option<&str>,
) -> ExactStateByteLimits {
    let hard_bytes = override_max_bytes
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or_else(|| configured_max_bytes.saturating_mul(EXACT_STATE_HARD_CAP_MULTIPLE));
    ExactStateByteLimits {
        soft_bytes: configured_max_bytes,
        hard_bytes,
    }
}

fn effective_cache_payload(
    requested: StageKvCachePayload,
    capability: &ModelKvCapability,
) -> StagePrefixCachePayload {
    match requested {
        StageKvCachePayload::ResidentKv => StagePrefixCachePayload::ResidentKv,
        StageKvCachePayload::KvRecurrent => StagePrefixCachePayload::KvRecurrent,
        StageKvCachePayload::FullState => StagePrefixCachePayload::FullState,
        StageKvCachePayload::Auto => match capability {
            ModelKvCapability::KnownDense => StagePrefixCachePayload::ResidentKv,
            ModelKvCapability::KnownRecurrent => StagePrefixCachePayload::KvRecurrent,
            ModelKvCapability::Unknown(_) => StagePrefixCachePayload::Disabled,
        },
    }
}

/// Either kill-switch variable resolving to `off` disables the cache, so an
/// explicit `SKIPPY_KV_CACHE=on` cannot mask `SKIPPY_PREFIX_CACHE=off`.
fn cache_disabled_by_env(kv_cache: Option<&str>, prefix_cache: Option<&str>) -> bool {
    [kv_cache, prefix_cache]
        .into_iter()
        .flatten()
        .any(|value| parse_cache_mode(value) == Some(StageKvCacheMode::Disabled))
}

fn effective_cache_config(config: &StageConfig) -> Option<StageKvCacheConfig> {
    // An explicit environment kill-switch beats the planned stage config so
    // benches and incident response can turn the prefix cache off without
    // replanning the topology.
    if cache_disabled_by_env(
        std::env::var("SKIPPY_KV_CACHE").ok().as_deref(),
        std::env::var("SKIPPY_PREFIX_CACHE").ok().as_deref(),
    ) {
        return None;
    }
    if let Some(cache) = config.kv_cache.clone() {
        return Some(cache);
    }
    let mode = std::env::var("SKIPPY_KV_CACHE")
        .or_else(|_| std::env::var("SKIPPY_PREFIX_CACHE"))
        .ok()
        .and_then(|value| parse_cache_mode(&value));
    let mode = mode?;
    let max_entries = std::env::var("SKIPPY_KV_CACHE_MAX_ENTRIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let max_bytes = std::env::var("SKIPPY_KV_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let min_tokens = std::env::var("SKIPPY_KV_CACHE_MIN_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let shared_prefix_stride_tokens = std::env::var("SKIPPY_KV_CACHE_SHARED_STRIDE_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128);
    let shared_prefix_record_limit = std::env::var("SKIPPY_KV_CACHE_SHARED_RECORD_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    let payload = std::env::var("SKIPPY_KV_CACHE_PAYLOAD")
        .ok()
        .and_then(|value| parse_cache_payload(&value))
        .unwrap_or(StageKvCachePayload::Auto);
    Some(StageKvCacheConfig {
        mode,
        payload,
        max_entries,
        max_bytes,
        min_tokens,
        shared_prefix_stride_tokens,
        shared_prefix_record_limit,
    })
}

fn parse_cache_payload(value: &str) -> Option<StageKvCachePayload> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "" | "auto" => Some(StageKvCachePayload::Auto),
        "resident" | "resident-kv" | "kv" => Some(StageKvCachePayload::ResidentKv),
        "kv-recurrent" | "kvrecurrent" => Some(StageKvCachePayload::KvRecurrent),
        "full" | "full-state" | "fullstate" => Some(StageKvCachePayload::FullState),
        _ => None,
    }
}

fn parse_cache_mode(value: &str) -> Option<StageKvCacheMode> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "" | "auto" => Some(StageKvCacheMode::Auto),
        "0" | "off" | "false" | "disabled" | "disable" => Some(StageKvCacheMode::Disabled),
        "record" => Some(StageKvCacheMode::Record),
        "1" | "on" | "true" | "lookup-record" | "lookuprecord" | "exact" => {
            Some(StageKvCacheMode::LookupRecord)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{FlashAttentionType, LoadMode};

    fn limits(soft_bytes: u64, hard_bytes: u64) -> ExactStateByteLimits {
        ExactStateByteLimits {
            soft_bytes,
            hard_bytes,
        }
    }

    fn pending(page_id: &str, tokens: &[i32], bytes: &[u8]) -> PendingExactStateRecord {
        PendingExactStateRecord {
            page_id: page_id.to_string(),
            payload: skippy_cache::ExactStatePayload::full_state(bytes.to_vec()),
            extra: super::super::ExactStateExtra::default(),
            namespace: "model".to_string(),
            token_ids: tokens.to_vec(),
        }
    }

    #[test]
    fn exact_payloads_live_on_radix_nodes_and_release_deduped_blocks_on_eviction() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4));

        store_exact_radix_record(
            &radix,
            &blobs,
            1,
            limits(0, 0),
            pending("first", &[1, 2], b"aaaabbbb"),
        )
        .unwrap();
        store_exact_radix_record(
            &radix,
            &blobs,
            1,
            limits(0, 0),
            pending("second", &[1, 3], b"aaaacccc"),
        )
        .unwrap();

        let mut radix = radix.lock().unwrap();
        let blobs = blobs.lock().unwrap();
        assert_eq!(radix.stats().recurrent_entries, 1);
        assert_eq!(blobs.physical_bytes(), 8);
        assert!(radix.lookup_recurrent("model", &[1, 2]).is_none());
        assert_eq!(
            radix
                .lookup_recurrent("model", &[1, 3])
                .expect("second exact payload should remain")
                .value
                .page_id,
            "second"
        );
    }

    #[test]
    fn invalid_exact_radix_key_releases_deduped_payload() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4));

        let error = store_exact_radix_record(
            &radix,
            &blobs,
            1,
            limits(0, 0),
            pending("empty", &[], b"aaaabbbb"),
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "radix cache key must contain at least one token"
        );
        assert_eq!(blobs.lock().unwrap().physical_bytes(), 0);
        assert_eq!(radix.lock().unwrap().stats().recurrent_entries, 0);
    }

    #[test]
    fn oversized_exact_payloads_retain_a_reusable_working_set() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4));

        // Every payload here is twice the soft cap, which is the recurrent case:
        // the attention-derived estimate cannot cover a full-state snapshot.
        for (page_id, tokens, bytes) in [
            ("first", [1, 2], b"aaaabbbb"),
            ("second", [1, 3], b"ccccdddd"),
            ("third", [1, 4], b"eeeeffff"),
        ] {
            store_exact_radix_record(
                &radix,
                &blobs,
                8,
                limits(4, 1024),
                pending(page_id, &tokens, bytes),
            )
            .unwrap();
        }

        // Concurrent sessions on one stage must not evict each other just
        // because a single snapshot cannot fit under the soft cap.
        assert_eq!(blobs.lock().unwrap().physical_bytes(), 24);
        let mut radix = radix.lock().unwrap();
        assert_eq!(radix.stats().recurrent_entries, 3);
        assert!(radix.lookup_recurrent("model", &[1, 2]).is_some());
        assert!(radix.lookup_recurrent("model", &[1, 3]).is_some());
        assert!(radix.lookup_recurrent("model", &[1, 4]).is_some());
    }

    #[test]
    fn exact_soft_retention_does_not_exceed_the_configured_entry_limit() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4));

        for (page_id, tokens, bytes) in [
            ("first", [1, 2], b"aaaabbbb"),
            ("second", [1, 3], b"ccccdddd"),
            ("third", [1, 4], b"eeeeffff"),
        ] {
            store_exact_radix_record(
                &radix,
                &blobs,
                2,
                limits(4, 1_024),
                pending(page_id, &tokens, bytes),
            )
            .unwrap();
        }

        let mut radix = radix.lock().unwrap();
        assert_eq!(radix.stats().recurrent_entries, 2);
        assert!(radix.lookup_recurrent("model", &[1, 2]).is_none());
        assert!(radix.lookup_recurrent("model", &[1, 3]).is_some());
        assert!(radix.lookup_recurrent("model", &[1, 4]).is_some());
    }

    #[test]
    fn exact_working_set_is_bounded_by_the_hard_ceiling() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4));

        for (page_id, tokens, bytes) in [
            ("first", [1, 2], b"aaaabbbb"),
            ("second", [1, 3], b"ccccdddd"),
            ("third", [1, 4], b"eeeeffff"),
        ] {
            store_exact_radix_record(
                &radix,
                &blobs,
                8,
                limits(4, 8),
                pending(page_id, &tokens, bytes),
            )
            .unwrap();
        }

        // The retention floor yields to the ceiling, leaving the most recent
        // snapshot rather than an unbounded working set.
        let mut radix = radix.lock().unwrap();
        assert_eq!(radix.stats().recurrent_entries, 1);
        assert!(radix.lookup_recurrent("model", &[1, 2]).is_none());
        assert!(radix.lookup_recurrent("model", &[1, 3]).is_none());
        assert!(radix.lookup_recurrent("model", &[1, 4]).is_some());
        assert_eq!(blobs.lock().unwrap().physical_bytes(), 8);
    }

    #[test]
    fn single_exact_payload_over_the_ceiling_is_still_reusable() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4));

        store_exact_radix_record(
            &radix,
            &blobs,
            8,
            limits(2, 4),
            pending("checkpoint", &[1, 2], b"aaaabbbb"),
        )
        .unwrap();

        // Self-eviction here is the original regression: it leaves the stage
        // with no exact prefix cache at all.
        let mut radix = radix.lock().unwrap();
        assert_eq!(radix.stats().recurrent_entries, 1);
        assert!(radix.lookup_recurrent("model", &[1, 2]).is_some());
    }

    #[test]
    fn exact_state_byte_limits_scale_the_ceiling_off_the_stage_budget() {
        assert_eq!(
            exact_state_byte_limits(1_024, None),
            limits(1_024, 1_024 * EXACT_STATE_HARD_CAP_MULTIPLE)
        );
        // An unset stage budget stays unbounded, as it was before.
        assert_eq!(exact_state_byte_limits(0, None), limits(0, 0));
    }

    #[test]
    fn exact_state_byte_limits_honour_the_operator_override() {
        assert_eq!(
            exact_state_byte_limits(1_024, Some(" 4096 ")),
            limits(1_024, 4_096)
        );
        assert_eq!(exact_state_byte_limits(1_024, Some("0")), limits(1_024, 0));
        // A malformed override must not silently disable the ceiling.
        assert_eq!(
            exact_state_byte_limits(1_024, Some("not-a-number")),
            limits(1_024, 1_024 * EXACT_STATE_HARD_CAP_MULTIPLE)
        );
    }

    #[test]
    fn either_cache_kill_switch_disables_the_cache() {
        assert!(!cache_disabled_by_env(None, None));
        assert!(!cache_disabled_by_env(Some("on"), None));
        assert!(cache_disabled_by_env(Some("off"), None));
        assert!(cache_disabled_by_env(None, Some("off")));
        assert!(cache_disabled_by_env(Some("on"), Some("off")));
        assert!(cache_disabled_by_env(Some("off"), Some("on")));
    }

    #[test]
    fn explicit_cache_payload_is_checked_against_loaded_model_capability() {
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::ResidentKv,
                &ModelKvCapability::KnownDense,
            ),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::KvRecurrent,
                &ModelKvCapability::KnownRecurrent,
            ),
            StagePrefixCachePayload::KvRecurrent
        );
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::FullState,
                &ModelKvCapability::KnownRecurrent,
            ),
            StagePrefixCachePayload::FullState
        );
    }

    #[test]
    fn loaded_hybrid_state_overrides_a_misleading_dense_model_name() {
        let config = enabled_auto_config("nvidia/Nemotron-3-Super-120B-A12B-NVFP4-MTPv2");

        let kv = KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Hybrid), None)
            .unwrap()
            .expect("hybrid loaded model should enable the recurrent cache");

        assert_eq!(kv.payload, StagePrefixCachePayload::KvRecurrent);
    }

    #[test]
    fn loaded_dense_state_selects_resident_kv_independent_of_model_name() {
        let config = enabled_auto_config("future/unknown-architecture-name");

        let kv = KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Dense), None)
            .unwrap()
            .expect("dense loaded model should enable resident KV");

        assert_eq!(kv.payload, StagePrefixCachePayload::ResidentKv);
    }

    #[test]
    fn missing_loaded_descriptor_disables_cache() {
        let config = enabled_auto_config("Qwen/Qwen3-8B");
        assert!(
            KvStageIntegration::from_loaded_model(&config, None, None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn explicit_resident_kv_is_rejected_for_loaded_hybrid_model() {
        let mut config = enabled_auto_config("future/model");
        config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::ResidentKv;

        assert!(
            KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Hybrid), None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn explicit_recurrent_kv_is_rejected_for_loaded_dense_model() {
        let mut config = enabled_auto_config("future/model");
        config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::KvRecurrent;

        assert!(
            KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Dense), None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn explicit_full_state_remains_architecture_neutral() {
        let mut config = enabled_auto_config("future/model");
        config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::FullState;

        for state_kind in [ModelStateKind::Dense, ModelStateKind::Recurrent] {
            let kv = KvStageIntegration::from_loaded_model(&config, Some(state_kind), None)
                .unwrap()
                .expect("full-state caching should support every loaded model state kind");
            assert_eq!(kv.payload, StagePrefixCachePayload::FullState);
        }
    }

    #[test]
    fn recurrent_cache_cardinality_is_capped_after_load() {
        let mut config = enabled_auto_config("future/model");
        config.ctx_size = 65_536;
        let kv =
            KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Recurrent), None)
                .unwrap()
                .expect("recurrent loaded model should enable exact-state caching");

        assert_eq!(kv.payload, StagePrefixCachePayload::KvRecurrent);
        assert_eq!(kv.exact_max_entries, RECURRENT_CACHE_MAX_ENTRIES);
    }

    #[test]
    fn parses_cache_mode_and_payload_aliases() {
        assert_eq!(
            parse_cache_mode("lookup_record"),
            Some(StageKvCacheMode::LookupRecord)
        );
        assert_eq!(
            parse_cache_mode("exact"),
            Some(StageKvCacheMode::LookupRecord)
        );
        assert_eq!(parse_cache_mode("off"), Some(StageKvCacheMode::Disabled));
        assert_eq!(
            parse_cache_payload("kv_recurrent"),
            Some(StageKvCachePayload::KvRecurrent)
        );
        assert_eq!(
            parse_cache_payload("resident"),
            Some(StageKvCachePayload::ResidentKv)
        );
        assert_eq!(parse_cache_payload("nope"), None);
    }

    fn test_config(model_id: &str) -> StageConfig {
        StageConfig {
            run_id: "test-run".to_string(),
            topology_id: "test-topology".to_string(),
            model_id: model_id.to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 256,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: false,
            resident_tensor_names: Vec::new(),
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
            ..StageConfig::default()
        }
    }

    fn enabled_auto_config(model_id: &str) -> StageConfig {
        let mut config = test_config(model_id);
        config.kv_cache = Some(StageKvCacheConfig {
            mode: StageKvCacheMode::LookupRecord,
            payload: StageKvCachePayload::Auto,
            max_entries: 512,
            max_bytes: 0,
            min_tokens: 1,
            shared_prefix_stride_tokens: 1,
            shared_prefix_record_limit: 1,
        });
        config
    }
}
