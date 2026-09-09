use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use mesh_llm_events::OutputEvent;
use skippy_cache::{
    CacheBlobStore, GeometryBlock, GeometryKind, L3CacheManager, L3Tier, PayloadGeometry,
    ResidentActivationCache, ResidentCacheConfig, SparseCheckpointPolicy, StoreLimits,
    UnifiedRadixCache, exact_state_identity_for_stage, numerical_model_identity_for_stage,
};
use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload};
use skippy_runtime::{ModelStateKind, RuntimeKvPageDesc};

use super::{
    EXACT_STATE_RECORD_CAPACITY, ExactStateByteLimits, KvStageIntegration, PendingExactStateRecord,
    RadixExactEntry, ResidentSequencePool, StageKvMode, StagePrefixCachePayload,
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
    pub fn from_loaded_model(
        config: &StageConfig,
        model_state_kind: Option<ModelStateKind>,
    ) -> Result<Option<Self>> {
        Self::from_loaded_model_with_l3(config, model_state_kind, || l3_manager_from_env(config))
    }

    /// Construct a stage with an explicitly injected node cache manager.
    /// Embedders that own several stages can use this path without consulting
    /// process environment or opening the root again.
    pub fn from_loaded_model_with_l3_manager(
        config: &StageConfig,
        model_state_kind: Option<ModelStateKind>,
        manager: Option<L3CacheManager>,
    ) -> Result<Option<Self>> {
        Self::from_loaded_model_with_l3(config, model_state_kind, || Ok(manager))
    }

    fn from_loaded_model_with_l3(
        config: &StageConfig,
        model_state_kind: Option<ModelStateKind>,
        manager: impl FnOnce() -> Result<Option<L3CacheManager>>,
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
        let dense_without_recurrent = matches!(model_capability, ModelKvCapability::KnownDense);
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
        let l3_manager = manager()?;
        let durable_payload = l3_manager.as_ref().map(|_| {
            if payload == StagePrefixCachePayload::ResidentKv && dense_without_recurrent {
                // Resident KV stays the in-process fast path. Dense families
                // export KV pages with an empty recurrent snapshot for L3;
                // the known-dense capability makes that empty component valid.
                StagePrefixCachePayload::KvRecurrent
            } else {
                payload
            }
        });
        let l3 = l3_manager
            .map(|manager| {
                l3_tier_for_manager(
                    config,
                    durable_payload.expect("enabled cache has a durable payload"),
                    manager,
                )
            })
            .transpose()?;
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
        let worker_l3 = l3.clone();
        let inflight_records: Arc<Mutex<BTreeSet<String>>> = l3.as_ref().map_or_else(
            || Arc::new(Mutex::new(BTreeSet::new())),
            |tier| tier.manager().record_claims(tier.state_identity()),
        );
        let worker_inflight_records = inflight_records.clone();
        let inflight_fills: Arc<Mutex<BTreeSet<String>>> = l3.as_ref().map_or_else(
            || Arc::new(Mutex::new(BTreeSet::new())),
            |tier| tier.manager().fill_claims(),
        );
        let worker_inflight_fills = inflight_fills.clone();
        let exact_state_record_queue_bytes = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let worker_exact_state_record_queue_bytes = exact_state_record_queue_bytes.clone();
        let exact_state_records_queued = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let exact_state_records_dropped = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let worker_exact_state_records_dropped = exact_state_records_dropped.clone();
        let exact_state_records_pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_exact_state_records_pending = exact_state_records_pending.clone();
        let exact_state_record_worker_healthy = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let worker_exact_state_record_worker_healthy = exact_state_record_worker_healthy.clone();
        let exact_state_record_worker_panics = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let worker_exact_state_record_worker_panics = exact_state_record_worker_panics.clone();
        std::thread::Builder::new()
            .name(format!("skippy-exact-cache-{}", config.stage_id))
            .spawn(move || {
                while let Ok(pending) = exact_state_record_rx.recv() {
                    let fill_claim = pending.l3_fill_claim.clone();
                    super::run_exact_state_record_job(
                        &worker_inflight_records,
                        &worker_exact_state_records_dropped,
                        &worker_exact_state_records_pending,
                        &worker_exact_state_record_queue_bytes,
                        &worker_exact_state_record_worker_healthy,
                        &worker_exact_state_record_worker_panics,
                        pending,
                        |pending| {
                            store_exact_radix_record(
                                &worker_radix,
                                &worker_exact_blobs,
                                exact_max_entries,
                                exact_byte_limits,
                                worker_l3.as_deref(),
                                pending,
                            )
                        },
                    );
                    if let Some(fill_claim) = fill_claim {
                        // The filled entry is radix-resident now (or the
                        // insert failed, and a re-fill is the right call
                        // either way): release the claim.
                        worker_inflight_fills
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .remove(&fill_claim);
                    }
                }
            })?;
        Ok(Some(Self {
            mode,
            payload,
            durable_payload,
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
            exact_state_record_queue_bytes,
            l3,
            inflight_fills,
            dense_without_recurrent,
        }))
    }

    #[cfg(test)]
    pub(crate) fn from_config(
        config: &StageConfig,
        model_state_kind: ModelStateKind,
    ) -> Result<Option<Self>> {
        Self::from_loaded_model(config, Some(model_state_kind))
    }
}

/// Open the durable L3 tier when `SKIPPY_L3_DIR` is set.
///
/// Experimental, environment-only plumbing: the public `[runtime.kv_cache.disk]`
/// configuration replaces it and this reader becomes a one-release
/// compatibility shim with a deprecation warning. The tier identity is the
/// radix cache's own numerical namespace, so restarts reuse it and a change
/// in weights, cache dtypes, layout or platform refuses stale state.
const DEFAULT_L3_BUDGET_BYTES: u64 = 32 * 1024 * 1024 * 1024;
const DEFAULT_L3_MINIMUM_FREE_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const L3_SEGMENT_BYTES: usize = 8 * 1024 * 1024;
/// Cap on rows per segment.
///
/// Windows are the dedupe granularity: a turn leaves a partial window in every
/// run, and those rows are rewritten next turn, so amplification is roughly
/// `1 + (tokens_so_far mod window) / new_tokens`. Measured on an M4 mini with
/// 2000-token base and 300-token turns: 512 rows gives 1.92x over the soak
/// (worst turn 2.55x) and misses §13.4's 1.2x gate; 128 rows gives 1.18x with
/// no margin; 64 rows gives 1.07x (worst turn 1.17x). The cost of the smaller
/// window is segment count, which is why it is capped rather than shrunk
/// further.
const L3_MAX_WINDOW_ROWS: u64 = 64;

fn l3_manager_from_env(config: &StageConfig) -> Result<Option<L3CacheManager>> {
    let Ok(root) = std::env::var("SKIPPY_L3_DIR") else {
        return Ok(None);
    };
    if root.trim().is_empty() {
        return Ok(None);
    }
    let budget_bytes = match std::env::var("SKIPPY_L3_BUDGET_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
    {
        // There is no unbounded mode. Zero used to mean "no cap"; it is now
        // rejected rather than silently reinterpreted.
        Some(0) => {
            let _ = mesh_llm_events::emit_event(OutputEvent::Warning {
                message: "SKIPPY_L3_BUDGET_BYTES=0 is not unbounded; using the default budget"
                    .to_string(),
                context: Some(format!("budget_bytes={DEFAULT_L3_BUDGET_BYTES}")),
            });
            DEFAULT_L3_BUDGET_BYTES
        }
        Some(bytes) => bytes,
        None => DEFAULT_L3_BUDGET_BYTES,
    };
    let manager = match L3CacheManager::acquire(
        &root,
        StoreLimits::new(budget_bytes, DEFAULT_L3_MINIMUM_FREE_BYTES),
    ) {
        Ok(manager) => manager,
        Err(error) => {
            // A tier that cannot open is a visible decline, not a crash:
            // serving continues cache-off for this stage.
            let _ = mesh_llm_events::emit_event(OutputEvent::Warning {
                message: "Skippy L3 disk cache disabled for this model stage".to_string(),
                context: Some(format!(
                    "stage_id={} directory={root} reason={error:#}",
                    config.stage_id
                )),
            });
            return Ok(None);
        }
    };
    Ok(Some(manager))
}

fn l3_tier_for_manager(
    config: &StageConfig,
    payload: StagePrefixCachePayload,
    manager: L3CacheManager,
) -> Result<Arc<L3Tier>> {
    let identity = exact_state_identity_for_stage(config, l3_payload_kind(payload));
    let model_identity = numerical_model_identity_for_stage(config);
    let root = manager.root().display().to_string();
    let tier = manager.tier_for_model(model_identity, identity, L3_SEGMENT_BYTES);
    // Warm state must never be invisible: say what the tier can restore the
    // moment the stage comes up.
    match tier.status() {
        Ok(status) => {
            let _ = mesh_llm_events::emit_event(OutputEvent::Info {
                message: "Skippy L3 disk cache open".to_string(),
                context: Some(format!(
                    "stage_id={} directory={root} restorable_manifests={} restorable_tokens={} used_bytes={} budget_bytes={}",
                    config.stage_id,
                    status.restorable_manifests,
                    status.restorable_tokens,
                    status.usage.used_bytes,
                    status.usage.budget_bytes
                )),
            });
        }
        Err(error) => {
            let _ = mesh_llm_events::emit_event(OutputEvent::Warning {
                message: "Skippy L3 disk cache open; status unavailable".to_string(),
                context: Some(format!("stage_id={} reason={error:#}", config.stage_id)),
            });
        }
    }
    Ok(Arc::new(tier))
}

fn l3_payload_kind(payload: StagePrefixCachePayload) -> &'static str {
    match payload {
        StagePrefixCachePayload::KvRecurrent => "kv-recurrent",
        StagePrefixCachePayload::FullState => "full-state",
        StagePrefixCachePayload::ResidentKv | StagePrefixCachePayload::Disabled => "unsupported",
    }
}

/// Describe an exported KV page so the store can cut segments where a growing
/// prefix keeps its bytes still.
///
/// The runtime writes every selected layer's K rows, then every layer's V rows,
/// then the indexer rows, each run holding one row per token in token order
/// (`llama_kv_cache::stage_export_kv_page`). Cutting that on fixed byte offsets
/// re-writes the whole payload every turn, because adding tokens shifts every
/// run after the first — measured at 8x on an M4 mini. Cutting per run into
/// fixed token windows writes only the new rows.
///
/// Returns `None` when the layout is not one this mapping can state exactly;
/// the store then falls back to fixed-size cutting, which is correct but not
/// cheap.
fn kv_page_geometry(desc: &RuntimeKvPageDesc, payload_bytes: u64) -> Option<PayloadGeometry> {
    // A composite (ISWA) page is two independently-shaped components; its
    // geometry is not this single-run description.
    if desc.component_count != 0 || desc.token_count == 0 || desc.layer_count == 0 {
        return None;
    }
    let rows = desc.token_count;
    let layers = desc.layer_count;
    let k_stride = u64::from(desc.k_row_bytes);
    if k_stride == 0 {
        return None;
    }
    let mut blocks = Vec::new();
    for layer in 0..layers {
        blocks.push(GeometryBlock {
            stride: k_stride,
            kind: GeometryKind::Key,
            layer,
            column: 0,
        });
    }
    let transposed = desc.flags & skippy_runtime::KV_PAGE_FLAG_V_TRANSPOSED != 0;
    if transposed {
        // Transposed V is stored column-major, but each column is still one
        // contiguous run of one element per token, so it windows the same way.
        // The column count is not in the descriptor; derive it from the bytes
        // the K and indexer runs do not claim.
        let element_bytes = u64::from(desc.v_element_bytes);
        let k_idx_stride = u64::from(desc.k_idx_row_bytes);
        let claimed = u64::from(layers)
            .checked_mul(rows)?
            .checked_mul(k_stride.checked_add(k_idx_stride)?)?;
        let v_bytes = payload_bytes.checked_sub(claimed)?;
        let per_layer = v_bytes.checked_div(u64::from(layers))?;
        let column_bytes = rows.checked_mul(element_bytes)?;
        if element_bytes == 0 || column_bytes == 0 || per_layer % column_bytes != 0 {
            return None;
        }
        let columns = u32::try_from(per_layer / column_bytes).ok()?;
        for layer in 0..layers {
            for column in 0..columns {
                blocks.push(GeometryBlock {
                    stride: element_bytes,
                    kind: GeometryKind::Value,
                    layer,
                    column,
                });
            }
        }
    } else if desc.v_row_bytes > 0 {
        for layer in 0..layers {
            blocks.push(GeometryBlock {
                stride: u64::from(desc.v_row_bytes),
                kind: GeometryKind::Value,
                layer,
                column: 0,
            });
        }
    }
    if desc.k_idx_row_bytes > 0 {
        for layer in 0..layers {
            blocks.push(GeometryBlock {
                stride: u64::from(desc.k_idx_row_bytes),
                kind: GeometryKind::KeyIndex,
                layer,
                column: 0,
            });
        }
    }
    // The window must depend only on the model's shape, never on this entry's
    // token count, or the boundaries move between turns and nothing dedupes.
    let widest = blocks.iter().map(|block| block.stride).max()?;
    let window_rows = (L3_SEGMENT_BYTES as u64 / widest.max(1))
        .clamp(1, L3_MAX_WINDOW_ROWS)
        .next_power_of_two()
        .min(L3_MAX_WINDOW_ROWS);
    let geometry = PayloadGeometry {
        blocks,
        rows,
        window_rows,
        // A recurrent snapshot rides after the KV page and has no row
        // structure; it is whatever the payload has left.
        tail_bytes: 0,
    };
    let described = geometry.total_bytes();
    let tail = payload_bytes.checked_sub(described)?;
    Some(PayloadGeometry {
        tail_bytes: tail,
        ..geometry
    })
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

fn emit_l3_state_transitions(l3: &L3Tier) {
    for transition in l3.manager().take_state_transitions() {
        let context = serde_json::to_string(&transition).ok();
        let _ = mesh_llm_events::emit_event(OutputEvent::Info {
            message: "Skippy L3 disk cache state changed".to_string(),
            context,
        });
    }
}

fn store_exact_radix_record(
    radix: &Mutex<UnifiedRadixCache<super::RadixResidentEntry, RadixExactEntry>>,
    blobs: &Mutex<CacheBlobStore>,
    max_entries: usize,
    limits: ExactStateByteLimits,
    l3: Option<&L3Tier>,
    pending: PendingExactStateRecord,
) -> Result<()> {
    // Write through to the durable tier before the payload is deduplicated
    // into blocks, while its bytes are still contiguous. Best-effort: a full
    // or failing disk must not fail the in-memory record. The refusal reason
    // lands in the tier's status; one warning per process keeps a full disk
    // from flooding the log.
    if let Some(l3) = l3 {
        let kv_desc_json = pending
            .extra
            .kv_desc
            .as_ref()
            .and_then(|desc| serde_json::to_string(desc).ok());
        let geometry = pending
            .extra
            .kv_desc
            .as_ref()
            .and_then(|desc| kv_page_geometry(desc, pending.payload.byte_len()));
        let spill = l3.spill(
            &pending.namespace,
            &pending.token_ids,
            &pending.payload,
            kv_desc_json,
            geometry.as_ref(),
        );
        emit_l3_state_transitions(l3);
        if let Err(error) = spill {
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::AcqRel) {
                let _ = mesh_llm_events::emit_event(OutputEvent::Warning {
                    message: "Skippy L3 disk cache write refused; see kv-cache status".to_string(),
                    context: Some(format!("page_id={} reason={error:#}", pending.page_id)),
                });
            }
        }
    }
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
            l3_fill_claim: None,
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
            None,
            pending("first", &[1, 2], b"aaaabbbb"),
        )
        .unwrap();
        store_exact_radix_record(
            &radix,
            &blobs,
            1,
            limits(0, 0),
            None,
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
            None,
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
    fn exact_records_write_through_to_l3_and_survive_radix_eviction() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4));
        let root = std::env::temp_dir()
            .join("skippy-server-l3-tests")
            .join(format!("write-through-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let tier = L3Tier::open(&root, 0, "blake3:test-tier".to_string(), 4096).unwrap();

        // Two records with max_entries = 1: the first is evicted from RAM.
        store_exact_radix_record(
            &radix,
            &blobs,
            1,
            limits(0, 0),
            Some(&tier),
            pending("first", &[1, 2], b"first-exact-state"),
        )
        .unwrap();
        store_exact_radix_record(
            &radix,
            &blobs,
            1,
            limits(0, 0),
            Some(&tier),
            pending("second", &[1, 3], b"second-exact-state"),
        )
        .unwrap();
        assert!(
            radix
                .lock()
                .unwrap()
                .lookup_recurrent("model", &[1, 2])
                .is_none(),
            "first record must be evicted from RAM"
        );

        // The evicted prefix still fills from the durable tier, including
        // for a longer query that extends the recorded path.
        let fill = tier
            .fill_longest("model", &[1, 2, 7, 8], 64)
            .unwrap()
            .expect("evicted record must remain in L3");
        assert_eq!(fill.token_count, 2);
        assert_eq!(
            fill.payload
                .full_state_bytes_timed()
                .unwrap()
                .0
                .into_owned(),
            b"first-exact-state".to_vec()
        );
        let status = tier.status().unwrap();
        assert_eq!(status.activity.writes, 2);
        assert_eq!(status.activity.fills, 1);
    }

    fn kv_desc(layers: u32, tokens: u64, k_row: u32, v_row: u32) -> RuntimeKvPageDesc {
        let mut desc = RuntimeKvPageDesc {
            version: 1,
            layer_start: 0,
            layer_end: layers as i32,
            token_start: 0,
            token_count: tokens,
            layer_count: layers,
            k_type: 1,
            v_type: 1,
            k_row_bytes: k_row,
            v_row_bytes: v_row,
            v_element_bytes: 2,
            k_idx_row_bytes: 0,
            payload_bytes: 0,
            flags: 0,
            codec: 0,
            component_count: 0,
            components: Default::default(),
        };
        desc.payload_bytes = u64::from(layers) * tokens * (u64::from(k_row) + u64::from(v_row));
        desc
    }

    #[test]
    fn kv_page_geometry_describes_the_runtime_export_layout() {
        // Every layer's K rows, then every layer's V rows: the order
        // `stage_export_kv_page` writes them in.
        let desc = kv_desc(4, 2048, 1024, 1024);
        let geometry =
            kv_page_geometry(&desc, desc.payload_bytes).expect("dense page must be describable");

        assert_eq!(geometry.blocks.len(), 8);
        assert_eq!(geometry.rows, 2048);
        assert!(
            geometry.blocks[..4]
                .iter()
                .all(|block| block.kind == GeometryKind::Key)
        );
        assert!(
            geometry.blocks[4..]
                .iter()
                .all(|block| block.kind == GeometryKind::Value)
        );
        assert_eq!(geometry.total_bytes(), desc.payload_bytes);
        assert!(geometry.matches(desc.payload_bytes));
    }

    #[test]
    fn kv_page_geometry_windows_are_stable_as_the_prefix_grows() {
        // The property the whole fix rests on: the same model must produce the
        // same window size at every length, or turn N+1's cuts land elsewhere
        // and nothing is reused.
        let short = kv_desc(4, 2048, 1024, 1024);
        let long = kv_desc(4, 4096, 1024, 1024);
        let short_geometry = kv_page_geometry(&short, short.payload_bytes).expect("short");
        let long_geometry = kv_page_geometry(&long, long.payload_bytes).expect("long");
        assert_eq!(short_geometry.window_rows, long_geometry.window_rows);
        assert_eq!(short_geometry.blocks, long_geometry.blocks);
    }

    #[test]
    fn kv_page_geometry_accounts_for_a_recurrent_tail() {
        let desc = kv_desc(2, 512, 512, 512);
        let tail = 4096;
        let geometry = kv_page_geometry(&desc, desc.payload_bytes + tail).expect("hybrid page");
        assert_eq!(geometry.tail_bytes, tail);
        assert!(geometry.matches(desc.payload_bytes + tail));
    }

    #[test]
    fn kv_page_geometry_declines_what_it_cannot_state_exactly() {
        // A composite ISWA page is two differently-shaped components.
        let mut composite = kv_desc(4, 512, 1024, 1024);
        composite.component_count = 2;
        assert!(kv_page_geometry(&composite, composite.payload_bytes).is_none());

        // Bytes that the described runs cannot account for.
        let desc = kv_desc(4, 512, 1024, 1024);
        assert!(kv_page_geometry(&desc, desc.payload_bytes - 1).is_none());
    }

    #[test]
    fn kv_page_geometry_windows_transposed_value_columns() {
        // Transposed V is column-major, but each column is one contiguous run
        // of one element per token, so it windows like any other run.
        let mut desc = kv_desc(2, 256, 1024, 0);
        desc.flags = skippy_runtime::KV_PAGE_FLAG_V_TRANSPOSED;
        desc.v_element_bytes = 2;
        let columns = 512u64;
        let payload = desc.payload_bytes + u64::from(desc.layer_count) * columns * 256 * 2;
        let geometry = kv_page_geometry(&desc, payload).expect("transposed page");

        let value_blocks = geometry
            .blocks
            .iter()
            .filter(|block| block.kind == GeometryKind::Value)
            .count();
        assert_eq!(value_blocks as u64, u64::from(desc.layer_count) * columns);
        assert!(geometry.matches(payload));
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
                None,
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
                None,
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
                None,
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
            None,
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

        let kv = KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Hybrid))
            .unwrap()
            .expect("hybrid loaded model should enable the recurrent cache");

        assert_eq!(kv.payload, StagePrefixCachePayload::KvRecurrent);
    }

    #[test]
    fn loaded_dense_state_selects_resident_kv_independent_of_model_name() {
        let config = enabled_auto_config("future/unknown-architecture-name");

        let kv = KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Dense))
            .unwrap()
            .expect("dense loaded model should enable resident KV");

        assert_eq!(kv.payload, StagePrefixCachePayload::ResidentKv);
    }

    #[test]
    fn missing_loaded_descriptor_disables_cache() {
        let config = enabled_auto_config("Qwen/Qwen3-8B");
        assert!(
            KvStageIntegration::from_loaded_model(&config, None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn explicit_resident_kv_is_rejected_for_loaded_hybrid_model() {
        let mut config = enabled_auto_config("future/model");
        config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::ResidentKv;

        assert!(
            KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Hybrid))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn explicit_recurrent_kv_is_rejected_for_loaded_dense_model() {
        let mut config = enabled_auto_config("future/model");
        config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::KvRecurrent;

        assert!(
            KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Dense))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn explicit_full_state_remains_architecture_neutral() {
        let mut config = enabled_auto_config("future/model");
        config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::FullState;

        for state_kind in [ModelStateKind::Dense, ModelStateKind::Recurrent] {
            let kv = KvStageIntegration::from_loaded_model(&config, Some(state_kind))
                .unwrap()
                .expect("full-state caching should support every loaded model state kind");
            assert_eq!(kv.payload, StagePrefixCachePayload::FullState);
        }
    }

    #[test]
    fn recurrent_cache_cardinality_is_capped_after_load() {
        let mut config = enabled_auto_config("future/model");
        config.ctx_size = 65_536;
        let kv = KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Recurrent))
            .unwrap()
            .expect("recurrent loaded model should enable exact-state caching");

        assert_eq!(kv.payload, StagePrefixCachePayload::KvRecurrent);
        assert_eq!(kv.exact_max_entries, RECURRENT_CACHE_MAX_ENTRIES);
    }

    #[test]
    fn injected_manager_is_shared_across_placement_equivalent_stages() {
        let root = std::env::temp_dir()
            .join("skippy-server-l3-manager-tests")
            .join(format!("shared-stages-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
        let mut first = enabled_auto_config("future/model");
        first.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::FullState;
        let second = StageConfig {
            stage_id: "replica-stage".to_string(),
            stage_index: 7,
            topology_id: "other-topology".to_string(),
            run_id: "other-run".to_string(),
            ..first.clone()
        };

        let first = KvStageIntegration::from_loaded_model_with_l3_manager(
            &first,
            Some(ModelStateKind::Dense),
            Some(manager.clone()),
        )
        .unwrap()
        .unwrap();
        let second = KvStageIntegration::from_loaded_model_with_l3_manager(
            &second,
            Some(ModelStateKind::Dense),
            Some(manager),
        )
        .unwrap()
        .unwrap();
        let first_l3 = first.l3.as_ref().unwrap();
        let second_l3 = second.l3.as_ref().unwrap();

        assert!(first_l3.manager().shares_root_with(second_l3.manager()));
        assert_eq!(first_l3.state_identity(), second_l3.state_identity());
        assert_eq!(first_l3.model_identity(), second_l3.model_identity());
        assert!(Arc::ptr_eq(&first.inflight_fills, &second.inflight_fills));
        assert!(Arc::ptr_eq(
            &first.inflight_records,
            &second.inflight_records
        ));
        assert!(first.try_begin_record("shared-page"));
        assert!(
            !second.try_begin_record("shared-page"),
            "placement replicas performed duplicate record work"
        );
        first.finish_record("shared-page");
        assert!(second.try_begin_record("shared-page"));
        second.finish_record("shared-page");
    }

    #[test]
    fn dense_disk_cache_preserves_resident_fast_path_and_exports_exact_state() {
        let root = std::env::temp_dir()
            .join("skippy-server-l3-manager-tests")
            .join(format!("dense-export-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
        let config = enabled_auto_config("future/model");

        let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
            &config,
            Some(ModelStateKind::Dense),
            Some(manager),
        )
        .unwrap()
        .expect("dense disk cache should remain enabled");

        assert_eq!(kv.payload, StagePrefixCachePayload::ResidentKv);
        assert_eq!(
            kv.durable_payload,
            Some(StagePrefixCachePayload::KvRecurrent)
        );
        assert_eq!(
            kv.exact_state_payload(),
            Some(StagePrefixCachePayload::KvRecurrent)
        );
        assert!(kv.l3.is_some());
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
