use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize},
        mpsc::{SyncSender, TrySendError},
    },
};

use anyhow::{Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use skippy_cache::{
    CacheBlobStore, ExactStatePayload, ResidentActivationCache, ResidentCacheConfig,
    SparseCheckpointPolicy, UnifiedRadixCache,
};
use skippy_metrics::attr as attr_key;
use skippy_runtime::{ActivationFrame, RuntimeKvPageDesc};

use crate::kv_proto::{
    Checksum, ChecksumAlgorithm, KvPageManifest, MANIFEST_SCHEMA_VERSION, PageIdentity, PageState,
};

mod activation;
mod cache_affinity;
mod config;
mod exact_state;
mod identity;
pub mod lifecycle;
mod model_capability;
mod output_tokens;
mod records;
mod resident_prefix;

pub use lifecycle::{KvLifecycleEvent, KvLifecycleObserver};
pub use records::{
    AttachedPage, ExactStateRecord, ExactStateRestore, LookupBatchOutcome, PrefillKvIdentity,
    RecordPageOutcome, ResidentActivationRecord, ResidentActivationRestore, ResidentPrefixRecord,
    ResidentPrefixRestore,
};
pub use resident_prefix::{ResidentCapacityDecision, ResidentPrefixEviction};

/// Return a bounded, stable telemetry class without exporting error text.
///
/// Detailed errors remain available to callers for local diagnostics, while metrics
/// only receive one of this fixed set of labels.
pub(crate) fn telemetry_error_class(error: &anyhow::Error) -> &'static str {
    telemetry_error_class_from_message(&error.to_string())
}

pub(crate) fn telemetry_error_class_from_message(message: &str) -> &'static str {
    let message = message.to_ascii_lowercase();
    if message.contains("checksum") || message.contains("digest") {
        "integrity"
    } else if message.contains("not found") || message.contains("missing") {
        "not_found"
    } else if message.contains("unsupported") || message.contains("disabled") {
        "unsupported"
    } else if message.contains("invalid") || message.contains("mismatch") {
        "invalid_data"
    } else if message.contains("timeout") || message.contains("unavailable") {
        "unavailable"
    } else if message.contains("permission") || message.contains("denied") {
        "permission"
    } else if message.contains("io error")
        || message.contains("i/o error")
        || message.contains("failed to read")
        || message.contains("failed to write")
    {
        "io"
    } else if message.contains("native") || message.contains("runtime") {
        "runtime"
    } else {
        "internal"
    }
}

pub(crate) fn proactive_eviction_error_kind(error: &anyhow::Error) -> &'static str {
    let message = error.to_string();
    if message.contains("is not active") {
        "inactive_session"
    } else if message.contains("batch size") {
        "invalid_batch_size"
    } else {
        "native_drop_failed"
    }
}

pub(crate) fn proactive_eviction_attrs(
    status: &str,
    error_kind: Option<&str>,
    target_tokens: u64,
    evicted_entries: usize,
    evicted_tokens: u64,
) -> BTreeMap<String, Value> {
    let mut attrs = BTreeMap::from([
        (
            "skippy.kv.decision".to_string(),
            json!("proactive_eviction"),
        ),
        (
            attr_key::KV_PROACTIVE_EVICTION_STATUS.to_string(),
            json!(status),
        ),
        (
            attr_key::KV_PROACTIVE_EVICTION_TARGET_TOKENS.to_string(),
            json!(target_tokens),
        ),
        (
            attr_key::KV_PROACTIVE_EVICTED_ENTRIES.to_string(),
            json!(evicted_entries),
        ),
        (
            attr_key::KV_PROACTIVE_EVICTED_TOKENS.to_string(),
            json!(evicted_tokens),
        ),
    ]);
    if let Some(error_kind) = error_kind {
        attrs.insert(
            attr_key::KV_PROACTIVE_EVICTION_ERROR_KIND.to_string(),
            json!(error_kind),
        );
    }
    attrs
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKvMode {
    Disabled,
    Record,
    LookupRecord,
    Correctness,
}

/// Byte budget for the exact-state catalog.
///
/// `soft_bytes` is the stage cache budget, which is derived from attention KV
/// metadata and therefore cannot account for the recurrent and convolution
/// buffers an exact-state snapshot also carries. A single snapshot can
/// legitimately exceed it, so the soft budget only bounds the catalog once it
/// already holds a working set. `hard_bytes` bounds that allowance except that
/// the last indivisible snapshot is retained even when it exceeds the limit.
/// Zero means unbounded for either field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExactStateByteLimits {
    pub(crate) soft_bytes: u64,
    pub(crate) hard_bytes: u64,
}

#[derive(Clone)]
pub struct KvStageIntegration {
    pub(crate) mode: StageKvMode,
    pub(crate) payload: StagePrefixCachePayload,
    pub(crate) correctness_mode: bool,
    pub(crate) trust_local_writes: bool,
    pub(crate) checkpoint_policy: SparseCheckpointPolicy,
    pub(crate) inflight_records: Arc<Mutex<BTreeSet<String>>>,
    pub(crate) resident_config: ResidentCacheConfig,
    pub(crate) resident_capacity_reservations: resident_prefix::ResidentCapacityReservations,
    pub(crate) resident_sequences: Arc<Mutex<ResidentSequencePool>>,
    pub(crate) activations: Arc<Mutex<ResidentActivationCache<ActivationFrame>>>,
    pub(crate) radix: Arc<Mutex<UnifiedRadixCache<RadixResidentEntry, RadixExactEntry>>>,
    pub(crate) exact_blobs: Arc<Mutex<CacheBlobStore>>,
    pub(crate) exact_max_entries: usize,
    pub(crate) exact_byte_limits: ExactStateByteLimits,
    pub(crate) exact_state_record_tx: SyncSender<PendingExactStateRecord>,
    pub(crate) exact_state_records_queued: Arc<AtomicU64>,
    pub(crate) exact_state_records_dropped: Arc<AtomicU64>,
    pub(crate) exact_state_records_pending: Arc<AtomicUsize>,
    pub(crate) exact_state_record_worker_healthy: Arc<AtomicBool>,
    pub(crate) exact_state_record_worker_panics: Arc<AtomicU64>,
    pub(crate) cache_healthy: Arc<AtomicBool>,
    pub(crate) output_tokens: Arc<Mutex<output_tokens::OutputTokenCache>>,
    pub(crate) split_prefill_tokens: Arc<Mutex<BTreeMap<String, Vec<i32>>>>,
    pub(crate) kv_lifecycle_observer: Option<Arc<dyn KvLifecycleObserver>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StagePrefixCachePayload {
    Disabled,
    ResidentKv,
    KvRecurrent,
    FullState,
}

pub(crate) const EXACT_STATE_RECORD_CAPACITY: usize = 1;

#[derive(Debug)]
pub(crate) struct PendingExactStateRecord {
    pub(crate) page_id: String,
    pub(crate) payload: ExactStatePayload,
    pub(crate) extra: ExactStateExtra,
    pub(crate) namespace: String,
    pub(crate) token_ids: Vec<i32>,
}

#[derive(Debug, Clone)]
pub(crate) struct RadixResidentEntry {
    pub(crate) page_id: String,
    pub(crate) seq_id: i32,
    pub(crate) token_count: u64,
    /// Deterministic first-order estimate of work needed to recreate this
    /// entry: cached tokens multiplied by stage-local layer count.
    pub(crate) recompute_cost: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct RadixExactEntry {
    pub(crate) page_id: String,
    pub(crate) payload: ExactStatePayload,
    pub(crate) extra: ExactStateExtra,
}

#[derive(Debug)]
pub(crate) struct ResidentSequencePool {
    reserved_seq_count: i32,
    next_seq_id: i32,
    free_seq_ids: Vec<i32>,
    allocated_seq_ids: BTreeSet<i32>,
    quarantined_seq_ids: BTreeSet<i32>,
}

impl ResidentSequencePool {
    fn new(reserved_seq_count: i32) -> Self {
        Self {
            reserved_seq_count,
            next_seq_id: reserved_seq_count,
            free_seq_ids: Vec::new(),
            allocated_seq_ids: BTreeSet::new(),
            quarantined_seq_ids: BTreeSet::new(),
        }
    }

    pub(crate) fn allocate(&mut self) -> Result<i32> {
        if let Some(seq_id) = self.free_seq_ids.pop() {
            if !self.allocated_seq_ids.insert(seq_id) {
                bail!("resident prefix sequence id {seq_id} is already allocated");
            }
            return Ok(seq_id);
        }
        let seq_id = self.next_seq_id;
        if seq_id < self.reserved_seq_count || seq_id >= skippy_cache::LLAMA_MAX_SEQ {
            bail!("resident prefix sequence id capacity exhausted");
        }
        self.next_seq_id = self
            .next_seq_id
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("resident prefix sequence id overflow"))?;
        if !self.allocated_seq_ids.insert(seq_id) {
            bail!("resident prefix sequence id {seq_id} is already allocated");
        }
        Ok(seq_id)
    }

    fn release(&mut self, seq_id: i32) -> Result<()> {
        self.validate_allocated(seq_id)?;
        self.allocated_seq_ids.remove(&seq_id);
        self.free_seq_ids.push(seq_id);
        Ok(())
    }

    fn quarantine(&mut self, seq_id: i32) -> Result<()> {
        self.validate_allocated(seq_id)?;
        self.allocated_seq_ids.remove(&seq_id);
        self.quarantined_seq_ids.insert(seq_id);
        Ok(())
    }

    fn force_quarantine(&mut self, seq_id: i32) {
        self.allocated_seq_ids.remove(&seq_id);
        self.free_seq_ids.retain(|candidate| *candidate != seq_id);
        self.quarantined_seq_ids.insert(seq_id);
    }

    fn validate_allocated(&self, seq_id: i32) -> Result<()> {
        if seq_id < self.reserved_seq_count || seq_id >= skippy_cache::LLAMA_MAX_SEQ {
            bail!("resident prefix sequence id {seq_id} is out of range");
        }
        if !self.allocated_seq_ids.contains(&seq_id) {
            bail!("resident prefix sequence id {seq_id} is not allocated");
        }
        Ok(())
    }

    fn stats(&self) -> (usize, usize, usize) {
        (
            self.allocated_seq_ids.len(),
            self.free_seq_ids.len(),
            self.quarantined_seq_ids.len(),
        )
    }
}

fn lock_resident_sequences(
    sequences: &Mutex<ResidentSequencePool>,
) -> std::sync::MutexGuard<'_, ResidentSequencePool> {
    sequences
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExactStateRecordAdmission {
    Queued,
    DroppedFull,
    WorkerStopped,
}

fn has_exact_state_record_capacity(pending_count: &AtomicUsize) -> bool {
    pending_count.load(std::sync::atomic::Ordering::Acquire) < EXACT_STATE_RECORD_CAPACITY
}

fn finish_exact_state_record(
    inflight_records: &Mutex<BTreeSet<String>>,
    pending_count: &AtomicUsize,
    page_id: &str,
) {
    inflight_records
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(page_id);
    pending_count.fetch_sub(1, std::sync::atomic::Ordering::Release);
}

/// Shared bookkeeping handles for the exact-state-record background worker,
/// grouped so `run_exact_state_record_job` stays within clippy's
/// too-many-arguments limit. Every field is itself an already-shared
/// (`Arc`-backed at the call site) handle; this struct only borrows them for
/// the duration of one job.
struct ExactStateWorkerHandles<'a> {
    inflight_records: &'a Mutex<BTreeSet<String>>,
    dropped: &'a AtomicU64,
    pending_count: &'a AtomicUsize,
    worker_healthy: &'a AtomicBool,
    worker_panics: &'a AtomicU64,
}

fn run_exact_state_record_job(
    handles: ExactStateWorkerHandles<'_>,
    observer: Option<&Arc<dyn KvLifecycleObserver>>,
    pending: PendingExactStateRecord,
    work: impl FnOnce(PendingExactStateRecord) -> Result<()>,
) {
    let notify = |event: KvLifecycleEvent| {
        if let Some(observer) = observer {
            observer.observe(event);
        }
    };
    let page_id = pending.page_id.clone();
    if !handles
        .worker_healthy
        .load(std::sync::atomic::Ordering::Acquire)
    {
        handles
            .dropped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        notify(KvLifecycleEvent::ExactStateRecordFailed);
        finish_exact_state_record(handles.inflight_records, handles.pending_count, &page_id);
        return;
    }

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(pending))) {
        Ok(Ok(())) => notify(KvLifecycleEvent::ExactStateRecordCompleted),
        Ok(Err(_)) => {
            handles
                .dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            notify(KvLifecycleEvent::ExactStateRecordFailed);
        }
        Err(_) => {
            handles
                .dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            handles
                .worker_panics
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            handles
                .worker_healthy
                .store(false, std::sync::atomic::Ordering::Release);
            notify(KvLifecycleEvent::ExactStateRecordFailed);
        }
    }
    finish_exact_state_record(handles.inflight_records, handles.pending_count, &page_id);
}

fn enqueue_exact_state_record(
    sender: &SyncSender<PendingExactStateRecord>,
    inflight_records: &Mutex<BTreeSet<String>>,
    queued: &AtomicU64,
    dropped: &AtomicU64,
    pending_count: &AtomicUsize,
    worker_healthy: &AtomicBool,
    pending: PendingExactStateRecord,
) -> ExactStateRecordAdmission {
    if !worker_healthy.load(std::sync::atomic::Ordering::Acquire) {
        inflight_records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&pending.page_id);
        dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return ExactStateRecordAdmission::WorkerStopped;
    }
    pending_count.fetch_add(1, std::sync::atomic::Ordering::Release);
    match sender.try_send(pending) {
        Ok(()) => {
            queued.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ExactStateRecordAdmission::Queued
        }
        Err(TrySendError::Full(pending)) => {
            pending_count.fetch_sub(1, std::sync::atomic::Ordering::Release);
            inflight_records
                .lock()
                .expect("kv inflight record lock poisoned")
                .remove(&pending.page_id);
            dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ExactStateRecordAdmission::DroppedFull
        }
        Err(TrySendError::Disconnected(pending)) => {
            pending_count.fetch_sub(1, std::sync::atomic::Ordering::Release);
            inflight_records
                .lock()
                .expect("kv inflight record lock poisoned")
                .remove(&pending.page_id);
            dropped.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            ExactStateRecordAdmission::WorkerStopped
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct ExactStateExtra {
    pub(crate) kv_desc: Option<RuntimeKvPageDesc>,
}

fn verify_resident_ownership(
    cache_healthy: &AtomicBool,
    resident_entries: usize,
    allocated_sequences: usize,
) -> Result<()> {
    if resident_entries == allocated_sequences {
        return Ok(());
    }
    if cache_healthy.swap(false, std::sync::atomic::Ordering::AcqRel) {
        let _ = mesh_llm_events::emit_event(mesh_llm_events::OutputEvent::Warning {
            message: "Skippy KV cache disabled after resident ownership mismatch".to_string(),
            context: Some(format!(
                "radix_entries={resident_entries} allocated_sequences={allocated_sequences}"
            )),
        });
    }
    bail!(
        "resident cache ownership mismatch: radix_entries={resident_entries} allocated_sequences={allocated_sequences}"
    )
}

impl KvStageIntegration {
    /// Attaches an optional resident-prefix cache lifecycle observer.
    /// Never required; a stage with no observer behaves identically.
    #[must_use]
    pub fn with_kv_lifecycle_observer(mut self, observer: Arc<dyn KvLifecycleObserver>) -> Self {
        self.kv_lifecycle_observer = Some(observer);
        self
    }

    pub(crate) fn notify_kv_lifecycle(&self, event: KvLifecycleEvent) {
        if let Some(observer) = self.kv_lifecycle_observer.as_ref() {
            observer.observe(event);
        }
    }

    pub fn mode(&self) -> StageKvMode {
        self.mode
    }

    pub(crate) fn payload_is_exact_state(&self) -> bool {
        self.payload.is_exact_state()
    }

    pub fn should_lookup(&self) -> bool {
        self.cache_healthy
            .load(std::sync::atomic::Ordering::Acquire)
            && matches!(
                self.mode,
                StageKvMode::LookupRecord | StageKvMode::Correctness
            )
    }

    pub fn should_record(&self) -> bool {
        self.cache_healthy
            .load(std::sync::atomic::Ordering::Acquire)
            && matches!(
                self.mode,
                StageKvMode::Record | StageKvMode::LookupRecord | StageKvMode::Correctness
            )
    }

    pub(crate) fn verify_resident_ownership(
        &self,
        resident_entries: usize,
        allocated_sequences: usize,
    ) -> Result<()> {
        verify_resident_ownership(&self.cache_healthy, resident_entries, allocated_sequences)
    }

    pub(crate) fn meets_shared_prefix_min_tokens(&self, matched_tokens: usize) -> bool {
        u64::try_from(matched_tokens).unwrap_or(u64::MAX) >= self.checkpoint_policy.min_tokens
    }

    pub fn try_begin_record(&self, page_id: &str) -> bool {
        if !self
            .exact_state_record_worker_healthy
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return false;
        }
        self.inflight_records
            .lock()
            .expect("kv inflight record lock poisoned")
            .insert(page_id.to_string())
    }

    pub fn finish_record(&self, page_id: &str) {
        self.inflight_records
            .lock()
            .expect("kv inflight record lock poisoned")
            .remove(page_id);
    }

    pub(crate) fn has_exact_state_record_capacity(&self) -> bool {
        self.exact_state_record_worker_healthy
            .load(std::sync::atomic::Ordering::Acquire)
            && has_exact_state_record_capacity(&self.exact_state_records_pending)
    }

    pub(crate) fn enqueue_exact_state_record(
        &self,
        pending: PendingExactStateRecord,
    ) -> ExactStateRecordAdmission {
        let admission = enqueue_exact_state_record(
            &self.exact_state_record_tx,
            &self.inflight_records,
            &self.exact_state_records_queued,
            &self.exact_state_records_dropped,
            &self.exact_state_records_pending,
            &self.exact_state_record_worker_healthy,
            pending,
        );
        if matches!(
            admission,
            ExactStateRecordAdmission::DroppedFull | ExactStateRecordAdmission::WorkerStopped
        ) {
            self.notify_kv_lifecycle(KvLifecycleEvent::ExactStateRecordFailed);
        }
        admission
    }

    pub async fn hello(&self) -> Result<()> {
        Ok(())
    }

    pub async fn lookup_prefixes(
        &self,
        _identities: Vec<PageIdentity>,
    ) -> Result<LookupBatchOutcome> {
        Ok(LookupBatchOutcome {
            pages: Vec::new(),
            errors: Vec::new(),
        })
    }

    pub async fn record_page(
        &self,
        page_id: String,
        identity: PageIdentity,
        bytes: &[u8],
        annotations: BTreeMap<String, String>,
    ) -> Result<KvPageManifest> {
        Ok(self
            .record_page_into(page_id, identity, bytes.len(), annotations, |output| {
                output.copy_from_slice(bytes);
                Ok(())
            })
            .await?
            .manifest)
    }

    pub async fn record_page_into(
        &self,
        page_id: String,
        identity: PageIdentity,
        byte_size: usize,
        mut annotations: BTreeMap<String, String>,
        write_page: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<RecordPageOutcome> {
        let mut bytes = vec![0; byte_size];
        write_page(&mut bytes)?;
        let checksum = local_trust_checksum(&page_id, byte_size as u64);
        annotations.insert(
            "mesh.skippy.prefix-cache-disabled".to_string(),
            "true".to_string(),
        );
        Ok(RecordPageOutcome {
            manifest: KvPageManifest {
                schema_version: MANIFEST_SCHEMA_VERSION,
                page_id,
                identity: Some(identity),
                state: PageState::Empty as i32,
                byte_size: byte_size as u64,
                shm_offset: 0,
                shm_len: byte_size as u64,
                checksum: Some(checksum),
                lease: None,
                annotations,
            },
            write_ms: 0.0,
            checksum_ms: 0.0,
        })
    }

    pub async fn attach_page(&self, _page_id: &str) -> Result<AttachedPage> {
        bail!("prefix cache integration is not included in mesh skippy-server")
    }

    pub async fn drop_session(&self, _session_id: &str) -> Result<u64> {
        Ok(0)
    }

    pub fn attrs(&self) -> Vec<(&'static str, Value)> {
        let radix_stats = match self.radix.try_lock() {
            Ok(radix) => Some(radix.stats()),
            Err(std::sync::TryLockError::Poisoned(error)) => Some(error.into_inner().stats()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        };
        let radix = radix_stats.unwrap_or_default();
        let activations = self
            .activations
            .lock()
            .expect("resident activation cache lock poisoned");
        let activations = activations.stats();
        let exact_blob_stats = self.exact_blobs.try_lock().ok().map(|blobs| {
            (
                blobs.physical_bytes(),
                blobs.block_count(),
                blobs.logical_ref_count(),
            )
        });
        let exact_state_stats_busy = radix_stats.is_none() || exact_blob_stats.is_none();
        let (exact_physical_bytes, exact_blocks, exact_block_refs) =
            exact_blob_stats.unwrap_or_default();
        let (resident_allocated_sequences, resident_free_sequences, resident_quarantined_sequences) =
            lock_resident_sequences(&self.resident_sequences).stats();
        let resident_sequence_drift = radix
            .resident_entries
            .abs_diff(resident_allocated_sequences);
        let (resident_capacity_reservations, resident_capacity_reserved_tokens) =
            self.resident_capacity_reservations.stats();
        let output_token_entries = self
            .output_tokens
            .try_lock()
            .ok()
            .map(|tokens| tokens.len())
            .unwrap_or_default();
        let (split_prefill_sessions, split_prefill_tokens) = self
            .split_prefill_tokens
            .try_lock()
            .ok()
            .map(|sessions| {
                (
                    sessions.len(),
                    sessions
                        .values()
                        .map(Vec::len)
                        .fold(0, usize::saturating_add),
                )
            })
            .unwrap_or_default();
        vec![
            ("skippy.kv.mode", json!(format!("{:?}", self.mode))),
            ("skippy.kv.payload", json!(format!("{:?}", self.payload))),
            (
                "skippy.kv.page_size_tokens",
                json!(self.checkpoint_policy.page_size_tokens),
            ),
            ("skippy.kv.resident_entries", json!(radix.resident_entries)),
            ("skippy.kv.resident_tokens", json!(radix.resident_tokens)),
            ("skippy.kv.radix.namespaces", json!(radix.namespaces)),
            ("skippy.kv.radix.nodes", json!(radix.nodes)),
            ("skippy.kv.radix.token_edges", json!(radix.token_edges)),
            ("skippy.kv.radix.splits", json!(radix.splits)),
            (
                "skippy.kv.radix.resident_entries",
                json!(radix.resident_entries),
            ),
            (
                "skippy.kv.radix.resident_active_refs",
                json!(radix.resident_active_refs),
            ),
            (
                "skippy.kv.radix.recurrent_entries",
                json!(radix.recurrent_entries),
            ),
            (
                "skippy.kv.radix.recurrent_active_refs",
                json!(radix.recurrent_active_refs),
            ),
            (
                "skippy.kv.radix.resident_evictions",
                json!(radix.resident_evictions),
            ),
            (
                "skippy.kv.radix.recurrent_evictions",
                json!(radix.recurrent_evictions),
            ),
            (
                "skippy.kv.resident_estimated_bytes",
                json!(radix.resident_logical_bytes),
            ),
            (
                "skippy.kv.max_entries",
                json!(self.resident_config.max_entries),
            ),
            ("skippy.kv.max_bytes", json!(self.resident_config.max_bytes)),
            (
                "skippy.activation_cache.entries",
                json!(activations.entries),
            ),
            (
                "skippy.activation_cache.resident_bytes",
                json!(activations.resident_bytes),
            ),
            ("skippy.exact_cache.entries", json!(radix.recurrent_entries)),
            (
                "skippy.exact_cache.logical_bytes",
                json!(radix.recurrent_logical_bytes),
            ),
            (
                "skippy.exact_cache.physical_bytes",
                json!(exact_physical_bytes),
            ),
            ("skippy.exact_cache.blocks", json!(exact_blocks)),
            ("skippy.exact_cache.block_refs", json!(exact_block_refs)),
            (
                "skippy.exact_cache.stats_busy",
                json!(exact_state_stats_busy),
            ),
            (
                "skippy.exact_cache.records_queued",
                json!(
                    self.exact_state_records_queued
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
            ),
            (
                "skippy.exact_cache.records_dropped",
                json!(
                    self.exact_state_records_dropped
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
            ),
            (
                "skippy.exact_cache.records_pending",
                json!(
                    self.exact_state_records_pending
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
            ),
            (
                "skippy.exact_cache.worker_healthy",
                json!(
                    self.exact_state_record_worker_healthy
                        .load(std::sync::atomic::Ordering::Acquire)
                ),
            ),
            (
                "skippy.exact_cache.worker_panics",
                json!(
                    self.exact_state_record_worker_panics
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
            ),
            (
                "skippy.exact_cache.max_bytes",
                json!(self.exact_byte_limits.soft_bytes),
            ),
            (
                "skippy.exact_cache.hard_max_bytes",
                json!(self.exact_byte_limits.hard_bytes),
            ),
            (
                "skippy.exact_cache.max_entries",
                json!(self.exact_max_entries),
            ),
            (
                "skippy.kv.output_token_entries",
                json!(output_token_entries),
            ),
            (
                "skippy.kv.split_prefill_sessions",
                json!(split_prefill_sessions),
            ),
            (
                "skippy.kv.split_prefill_tokens",
                json!(split_prefill_tokens),
            ),
            (
                "skippy.kv.split_prefill_bytes",
                json!(split_prefill_tokens.saturating_mul(std::mem::size_of::<i32>())),
            ),
            (
                "skippy.kv.resident_allocated_sequences",
                json!(resident_allocated_sequences),
            ),
            (
                "skippy.kv.resident_free_sequences",
                json!(resident_free_sequences),
            ),
            (
                "skippy.kv.resident_quarantined_sequences",
                json!(resident_quarantined_sequences),
            ),
            (
                "skippy.kv.resident_sequence_drift",
                json!(resident_sequence_drift),
            ),
            (
                "skippy.kv.capacity_reservations",
                json!(resident_capacity_reservations),
            ),
            (
                "skippy.kv.capacity_reserved_tokens",
                json!(resident_capacity_reserved_tokens),
            ),
            ("skippy.kv.correctness_mode", json!(self.correctness_mode)),
            (
                "skippy.kv.cache_healthy",
                json!(
                    self.cache_healthy
                        .load(std::sync::atomic::Ordering::Acquire)
                ),
            ),
            (
                "skippy.kv.trust_local_writes",
                json!(self.trust_local_writes),
            ),
            (
                "skippy.kv.shared_prefix_min_tokens",
                json!(self.checkpoint_policy.min_tokens),
            ),
            (
                "skippy.kv.shared_prefix_stride_tokens",
                json!(self.checkpoint_policy.stride_tokens),
            ),
            (
                "skippy.kv.shared_prefix_record_limit",
                json!(self.checkpoint_policy.record_limit),
            ),
        ]
        .into_iter()
        .collect()
    }

    /// Test-only compatibility helper for exercising the side-cache capacity
    /// accounting without constructing a frontend sampling fingerprint.
    #[cfg(test)]
    pub fn record_cached_first_token(&self, identity: &PrefillKvIdentity, predicted: i32) -> bool {
        self.record_cached_first_token_with_key(&identity.page_id, identity, predicted)
    }

    /// Records a first token under a key that includes the request's sampling
    /// semantics. Sampled-output paths must use this method.
    pub(crate) fn record_cached_first_token_with_key(
        &self,
        cache_key: &str,
        identity: &PrefillKvIdentity,
        predicted: i32,
    ) -> bool {
        if !self.should_record()
            || identity.identity.token_count < self.checkpoint_policy.min_tokens
        {
            return false;
        }
        self.output_tokens
            .lock()
            .expect("output-token cache lock poisoned")
            .record_first(cache_key, predicted)
    }

    /// Test-only compatibility helper for the identity-only cache probe.
    #[cfg(test)]
    pub fn lookup_cached_first_token(&self, identity: &PrefillKvIdentity) -> Option<i32> {
        self.lookup_cached_first_token_with_key(&identity.page_id)
    }

    /// Looks up a first token using the sampling-aware cache key generated by
    /// the frontend. Callers should gate this operation on replay-safe
    /// sampling before attempting the lookup.
    pub(crate) fn lookup_cached_first_token_with_key(&self, cache_key: &str) -> Option<i32> {
        if !self.should_lookup() {
            return None;
        }
        self.output_tokens
            .lock()
            .expect("output-token cache lock poisoned")
            .lookup_first(cache_key)
    }

    pub fn record_cached_replay_tokens(
        &self,
        cache_key: &str,
        identity: &PrefillKvIdentity,
        previous: &[i32],
        predicted: i32,
        max_replay_tokens: usize,
    ) -> Option<usize> {
        if !self.should_record()
            || max_replay_tokens == 0
            || previous.len() >= max_replay_tokens
            || identity.identity.token_count < self.checkpoint_policy.min_tokens
        {
            return None;
        }
        self.output_tokens
            .lock()
            .expect("output-token cache lock poisoned")
            .record_replay(cache_key, previous, predicted, max_replay_tokens)
    }

    pub fn lookup_cached_replay_tokens(&self, cache_key: &str, max_tokens: usize) -> Vec<i32> {
        if !self.should_lookup() || max_tokens == 0 {
            return Vec::new();
        }
        self.output_tokens
            .lock()
            .expect("output-token cache lock poisoned")
            .lookup_replay(cache_key, max_tokens)
    }
}

fn local_trust_checksum(page_id: &str, byte_size: u64) -> Checksum {
    let mut digest = Sha256::new();
    digest.update(b"skippy-local-trust-v1");
    digest.update(page_id.as_bytes());
    digest.update(byte_size.to_le_bytes());
    Checksum {
        algorithm: ChecksumAlgorithm::Sha256 as i32,
        digest: digest.finalize().to_vec(),
    }
}

#[cfg(test)]
mod exact_state_record_queue_tests {
    use skippy_cache::ExactStatePayload;
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::sync_channel,
    };

    use super::{
        BTreeSet, EXACT_STATE_RECORD_CAPACITY, ExactStateExtra, ExactStateRecordAdmission,
        ExactStateWorkerHandles, KvLifecycleEvent, KvLifecycleObserver, PendingExactStateRecord,
        enqueue_exact_state_record, has_exact_state_record_capacity, run_exact_state_record_job,
    };

    struct RecordingObserver(Arc<Mutex<Vec<KvLifecycleEvent>>>);

    impl KvLifecycleObserver for RecordingObserver {
        fn observe(&self, event: KvLifecycleEvent) {
            self.0.lock().unwrap().push(event);
        }
    }

    fn pending(page_id: &str) -> PendingExactStateRecord {
        PendingExactStateRecord {
            page_id: page_id.to_string(),
            payload: ExactStatePayload::full_state(vec![1]),
            extra: ExactStateExtra::default(),
            namespace: "test".to_string(),
            token_ids: vec![1],
        }
    }

    #[test]
    fn pending_capacity_signal_rejects_work_before_export() {
        let pending_count = AtomicUsize::new(0);
        assert!(has_exact_state_record_capacity(&pending_count));

        pending_count.store(EXACT_STATE_RECORD_CAPACITY, Ordering::Release);
        assert!(!has_exact_state_record_capacity(&pending_count));
    }

    #[test]
    fn full_queue_drops_optional_record_and_releases_inflight_page() {
        let (sender, _receiver) = sync_channel(1);
        sender.send(pending("queued")).unwrap();
        let inflight = Mutex::new(BTreeSet::from(["dropped".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(0);
        let worker_healthy = AtomicBool::new(true);

        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &worker_healthy,
                pending("dropped"),
            ),
            ExactStateRecordAdmission::DroppedFull
        );
        assert!(!inflight.lock().unwrap().contains("dropped"));
        assert_eq!(queued.load(Ordering::Relaxed), 0);
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(pending_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn disconnected_worker_releases_inflight_page() {
        let (sender, receiver) = sync_channel(1);
        drop(receiver);
        let inflight = Mutex::new(BTreeSet::from(["orphaned".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(0);
        let worker_healthy = AtomicBool::new(true);

        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &worker_healthy,
                pending("orphaned"),
            ),
            ExactStateRecordAdmission::WorkerStopped
        );
        assert!(!inflight.lock().unwrap().contains("orphaned"));
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(pending_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn worker_keeps_page_inflight_until_record_finishes() {
        let (sender, receiver) = sync_channel(1);
        let inflight = Arc::new(Mutex::new(BTreeSet::from(["page".to_string()])));
        let worker_inflight = inflight.clone();
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = Arc::new(AtomicUsize::new(0));
        let worker_pending_count = pending_count.clone();
        let worker_healthy = AtomicBool::new(true);

        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &worker_healthy,
                pending("page"),
            ),
            ExactStateRecordAdmission::Queued
        );
        assert!(inflight.lock().unwrap().contains("page"));

        let worker = std::thread::spawn(move || {
            let pending = receiver.recv().unwrap();
            assert!(worker_inflight.lock().unwrap().contains(&pending.page_id));
            worker_inflight.lock().unwrap().remove(&pending.page_id);
            worker_pending_count.fetch_sub(1, Ordering::Relaxed);
        });
        worker.join().unwrap();

        assert!(!inflight.lock().unwrap().contains("page"));
        assert_eq!(queued.load(Ordering::Relaxed), 1);
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
        assert_eq!(pending_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn worker_success_notifies_completed_not_mere_admission() {
        let inflight = Mutex::new(BTreeSet::from(["written".to_string()]));
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(1);
        let worker_healthy = AtomicBool::new(true);
        let worker_panics = AtomicU64::new(0);
        let events: Arc<Mutex<Vec<KvLifecycleEvent>>> = Arc::default();
        let observer: Arc<dyn KvLifecycleObserver> = Arc::new(RecordingObserver(events.clone()));

        run_exact_state_record_job(
            ExactStateWorkerHandles {
                inflight_records: &inflight,
                dropped: &dropped,
                pending_count: &pending_count,
                worker_healthy: &worker_healthy,
                worker_panics: &worker_panics,
            },
            Some(&observer),
            pending("written"),
            |_| Ok(()),
        );

        assert_eq!(
            *events.lock().unwrap(),
            vec![KvLifecycleEvent::ExactStateRecordCompleted]
        );
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn worker_write_error_notifies_failed_not_completed() {
        let inflight = Mutex::new(BTreeSet::from(["broken".to_string()]));
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(1);
        let worker_healthy = AtomicBool::new(true);
        let worker_panics = AtomicU64::new(0);
        let events: Arc<Mutex<Vec<KvLifecycleEvent>>> = Arc::default();
        let observer: Arc<dyn KvLifecycleObserver> = Arc::new(RecordingObserver(events.clone()));

        run_exact_state_record_job(
            ExactStateWorkerHandles {
                inflight_records: &inflight,
                dropped: &dropped,
                pending_count: &pending_count,
                worker_healthy: &worker_healthy,
                worker_panics: &worker_panics,
            },
            Some(&observer),
            pending("broken"),
            |_| anyhow::bail!("simulated write failure"),
        );

        assert_eq!(
            *events.lock().unwrap(),
            vec![KvLifecycleEvent::ExactStateRecordFailed]
        );
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn worker_panic_fails_closed_and_releases_all_record_bookkeeping() {
        let (sender, _receiver) = sync_channel(1);
        let inflight = Mutex::new(BTreeSet::from(["panicked".to_string()]));
        let queued = AtomicU64::new(0);
        let dropped = AtomicU64::new(0);
        let pending_count = AtomicUsize::new(1);
        let worker_healthy = AtomicBool::new(true);
        let worker_panics = AtomicU64::new(0);

        run_exact_state_record_job(
            ExactStateWorkerHandles {
                inflight_records: &inflight,
                dropped: &dropped,
                pending_count: &pending_count,
                worker_healthy: &worker_healthy,
                worker_panics: &worker_panics,
            },
            None,
            pending("panicked"),
            |_| panic!("injected exact-record worker failure"),
        );

        assert!(!worker_healthy.load(Ordering::Acquire));
        assert_eq!(worker_panics.load(Ordering::Relaxed), 1);
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(pending_count.load(Ordering::Acquire), 0);
        assert!(inflight.lock().unwrap().is_empty());

        inflight.lock().unwrap().insert("later".to_string());
        assert_eq!(
            enqueue_exact_state_record(
                &sender,
                &inflight,
                &queued,
                &dropped,
                &pending_count,
                &worker_healthy,
                pending("later"),
            ),
            ExactStateRecordAdmission::WorkerStopped
        );
        assert!(inflight.lock().unwrap().is_empty());
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
        assert_eq!(pending_count.load(Ordering::Acquire), 0);
    }
}

#[cfg(test)]
mod telemetry_error_class_tests {
    use super::telemetry_error_class_from_message;

    #[test]
    fn maps_detailed_errors_to_bounded_classes() {
        assert_eq!(
            telemetry_error_class_from_message("checksum mismatch for page abc"),
            "integrity"
        );
        assert_eq!(
            telemetry_error_class_from_message("permission denied: /secret/path"),
            "permission"
        );
        assert_eq!(
            telemetry_error_class_from_message("arbitrary secret detail 123"),
            "internal"
        );
    }
}

#[cfg(test)]
mod resident_ownership_reconciliation_tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };

    use super::{ResidentSequencePool, lock_resident_sequences, verify_resident_ownership};

    #[test]
    fn ownership_mismatch_permanently_disables_cache_operations() {
        let healthy = AtomicBool::new(true);

        let error = verify_resident_ownership(&healthy, 2, 1).unwrap_err();

        assert_eq!(
            error.to_string(),
            "resident cache ownership mismatch: radix_entries=2 allocated_sequences=1"
        );
        assert!(!healthy.load(Ordering::Acquire));
        verify_resident_ownership(&healthy, 1, 1).unwrap();
        assert!(!healthy.load(Ordering::Acquire));
    }

    #[test]
    fn poisoned_resident_sequence_lock_recovers_without_reusing_state() {
        let sequences = Arc::new(Mutex::new(ResidentSequencePool::new(4)));
        let poisoned = sequences.clone();
        assert!(
            std::thread::spawn(move || {
                let mut guard = poisoned.lock().unwrap();
                guard.allocate().unwrap();
                panic!("poison resident sequence pool for test");
            })
            .join()
            .is_err()
        );

        let mut guard = lock_resident_sequences(&sequences);
        assert_eq!(guard.stats(), (1, 0, 0));
        assert_eq!(guard.allocate().unwrap(), 5);
    }

    #[test]
    fn forced_quarantine_removes_a_sequence_from_every_reusable_set() {
        let mut sequences = ResidentSequencePool::new(4);
        let seq_id = sequences.allocate().unwrap();
        sequences.release(seq_id).unwrap();

        sequences.force_quarantine(seq_id);

        assert_eq!(sequences.stats(), (0, 0, 1));
        assert_eq!(sequences.allocate().unwrap(), seq_id + 1);
    }
}
