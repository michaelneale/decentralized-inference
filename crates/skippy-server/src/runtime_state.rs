use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use skippy_protocol::{FlashAttentionType, LoadMode, SplitMode, StageConfig};
use skippy_runtime::{
    ActivationBoundaryDesc, ActivationFrame, DecodeBatchRequest, DecodeFrameBatchOutput,
    DecodeFrameBatchRequest, FlashAttentionType as RuntimeFlashAttentionType,
    GenerationSignalWindow, GlmDsaPolicy as RuntimeGlmDsaPolicy, IterationBatchOutput,
    IterationBatchPhase, IterationBatchRequest, MediaInput, MediaPrefill, MediaPrefillFrame,
    ModelStateKind, MtpSource, NativeMtpDraft, RuntimeConfig, RuntimeKvPage, RuntimeKvPageDesc,
    RuntimeLoadMode, SamplingConfig, SplitMode as RuntimeSplitMode, StageModel, StageSession,
    TokenSignal, parse_cache_type,
};

mod frame_operations;
mod lane_lifecycle;
pub mod lifecycle;
mod restore_transaction;
mod state_transfer;

pub use lifecycle::{SessionLifecycleEvent, SessionLifecycleObserver};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeLaunchOverrides {
    pub n_threads: Option<usize>,
    pub n_threads_batch: Option<usize>,
    pub mtp_source: MtpSource,
}

pub struct RuntimeState {
    pub model: StageModel,
    layer_start: u32,
    layer_end: u32,
    lane_count: u32,
    /// Size of the context's KV cell pool, in tokens (llama.cpp `n_ctx`). In
    /// unified-KV mode every lane draws decode/prefill cells from this single
    /// shared pool, so it is the real ceiling for scheduler admission — see
    /// [`Self::kv_pool_tokens`].
    ctx_size: u32,
    /// High-water mark of lane indices ever handed out. Combined with
    /// [`Self::free_lane_indices`], the count of live lanes equals
    /// `next_lane_index - free_lane_indices.len()`.
    next_lane_index: usize,
    /// Lane indices that were previously handed out but are now free
    /// to reuse. An index lands here only when the lane's underlying
    /// StageSession has been dropped (which calls skippy_session_free
    /// on the C side, clearing that seq_id's KV cells).
    ///
    /// Without this list, a discarded lane (see
    /// [`Self::drop_session_timed`]) would permanently consume one of
    /// the slots represented by [`Self::next_lane_index`], leading to
    /// "all execution lanes are busy" errors long before the runtime
    /// has actually run out of capacity.
    free_lane_indices: Vec<usize>,
    sessions: BTreeMap<String, RuntimeLaneSession>,
    idle_sessions: Vec<RuntimeLaneSession>,
    /// Upper bound on `idle_sessions.len()`, from `model_fit.cache_idle_slots`.
    /// `None` preserves today's unbounded idle-pool behavior (bounded only by
    /// `lane_count` through `prewarm_idle_sessions`'s admission check).
    max_idle_sessions: Option<usize>,
    session_token_counts: BTreeMap<String, u64>,
    session_resident_prefixes: BTreeMap<String, ResidentLanePrefix>,
    session_lifecycle_observer: Option<Arc<dyn SessionLifecycleObserver>>,
    #[cfg(test)]
    modelless_for_test: bool,
}

struct RuntimeLaneSession {
    index: usize,
    session: StageSession,
    resident_prefix: Option<ResidentLanePrefix>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeSessionLaneStats {
    pub index: usize,
    pub active: bool,
    pub session_id: Option<String>,
    pub token_count: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeSessionStats {
    pub lane_count: usize,
    pub active_sessions: usize,
    pub idle_sessions: usize,
    pub idle_resident_prefixes: usize,
    pub tracked_token_counts: usize,
    pub max_session_tokens: u64,
    pub total_session_tokens: u64,
    pub lanes: Vec<RuntimeSessionLaneStats>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeSessionDropStats {
    pub reset_session: bool,
    pub reset_ms: f64,
    pub preserved_resident_prefix: bool,
    /// True when the lane could not be returned to the idle pool because
    /// the underlying StageSession failed to reset cleanly. The lane is
    /// dropped (which invokes the C-side skippy_session_free) and the
    /// pool capacity is restored on the next prewarm/admission cycle.
    pub lane_discarded: bool,
    /// Reset-error detail, when [`Self::lane_discarded`] is true.
    pub lane_discard_reason: Option<String>,
    pub stats_after: RuntimeSessionStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeSessionAlignStats {
    pub before_token_count: u64,
    pub after_token_count: u64,
}

pub struct RuntimeDecodeBatchRequest<'a> {
    pub session_id: &'a str,
    pub token_id: i32,
    pub sampling: Option<&'a SamplingConfig>,
}

pub struct RuntimeDecodeFrameBatchRequest<'a> {
    pub session_id: &'a str,
    pub token_id: i32,
    pub sampling: Option<&'a SamplingConfig>,
    pub input: Option<&'a ActivationFrame>,
}

pub struct RuntimeIterationBatchRequest<'a> {
    pub session_id: &'a str,
    pub token_ids: &'a [i32],
    pub positions: &'a [i32],
    pub sampling: Option<&'a SamplingConfig>,
    pub input: Option<&'a ActivationFrame>,
    pub sample_last: bool,
    pub phase: IterationBatchPhase,
}

#[derive(Debug, Clone)]
struct ResidentLanePrefix {
    page_id: String,
    token_count: u64,
}

impl RuntimeState {
    pub fn input_activation_boundary(&self) -> Option<ActivationBoundaryDesc> {
        self.model.input_activation_boundary()
    }

    pub fn output_activation_boundary(&self) -> Option<ActivationBoundaryDesc> {
        self.model.output_activation_boundary()
    }
    /// A runtime with no model behind it, for tests that exercise code paths
    /// which never touch the model.
    ///
    /// [`Self::session_stats`] is pure Rust over the lane bookkeeping below, so
    /// status/observability behaviour can be tested without loading a GGUF.
    /// Any call that reaches [`Self::model`] will dereference a null handle, so
    /// this must not be used to drive inference.
    #[cfg(test)]
    pub(crate) fn new_modelless_for_test(lane_count: u32) -> Self {
        Self::new_modelless_with_capacity_for_test(lane_count, 0)
    }

    #[cfg(test)]
    pub(crate) fn new_modelless_with_capacity_for_test(lane_count: u32, ctx_size: u32) -> Self {
        Self {
            model: StageModel::new_dummy(),
            layer_start: 0,
            layer_end: 1,
            lane_count,
            ctx_size,
            next_lane_index: 0,
            free_lane_indices: Vec::new(),
            sessions: BTreeMap::new(),
            idle_sessions: Vec::new(),
            max_idle_sessions: None,
            session_token_counts: BTreeMap::new(),
            session_resident_prefixes: BTreeMap::new(),
            session_lifecycle_observer: None,
            modelless_for_test: true,
        }
    }

    /// Attaches an optional session-lifecycle observer. Never required;
    /// a runtime with no observer behaves identically.
    #[must_use]
    pub fn with_session_lifecycle_observer(
        mut self,
        observer: Arc<dyn SessionLifecycleObserver>,
    ) -> Self {
        self.session_lifecycle_observer = Some(observer);
        self
    }

    pub(crate) fn notify_session_lifecycle(&self, event: SessionLifecycleEvent) {
        if let Some(observer) = self.session_lifecycle_observer.as_ref() {
            observer.observe(event);
        }
    }

    #[cfg(test)]
    pub(crate) fn track_session_tokens_for_test(&mut self, session_id: &str, token_count: u64) {
        self.session_token_counts
            .insert(session_id.to_string(), token_count);
    }

    pub fn lane_count(&self) -> u32 {
        self.lane_count
    }

    pub(crate) fn active_session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Total KV cell pool available to this context, in tokens (`n_ctx`).
    ///
    /// In unified-KV mode all lanes share this single pool, so it is the real
    /// token budget the iteration scheduler must admit against. Returns 0 for
    /// the modelless test runtime, in which case callers should fall back to a
    /// configured default.
    pub fn kv_pool_tokens(&self) -> u32 {
        self.ctx_size
    }

    #[cfg(test)]
    pub(crate) fn is_modelless_for_test(&self) -> bool {
        self.modelless_for_test
    }
}

impl Drop for RuntimeState {
    fn drop(&mut self) {
        self.sessions.clear();
        self.idle_sessions.clear();
    }
}

pub fn load_runtime(config: &StageConfig) -> Result<Option<Arc<Mutex<RuntimeState>>>> {
    load_runtime_with_overrides(config, &RuntimeLaunchOverrides::default(), None)
}

/// Return the state semantics captured from the model that was actually
/// opened by llama.cpp. Callers use this after load so cache selection never
/// depends on a repository name or pre-load family guess.
pub fn loaded_model_state_kind(
    runtime: Option<&Arc<Mutex<RuntimeState>>>,
) -> Option<ModelStateKind> {
    runtime.and_then(|runtime| {
        runtime
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .model
            .capability()
            .map(|capability| capability.state_kind)
    })
}

pub fn load_runtime_with_overrides(
    config: &StageConfig,
    overrides: &RuntimeLaunchOverrides,
    session_lifecycle_observer: Option<Arc<dyn SessionLifecycleObserver>>,
) -> Result<Option<Arc<Mutex<RuntimeState>>>> {
    reject_legacy_serving_package(config)?;
    let runtime_config = runtime_config_from_stage_config(config, overrides)?;

    let admitted_model_parts = config
        .model_part_paths
        .iter()
        .map(std::path::PathBuf::from)
        .collect::<Vec<_>>();
    let model = match config.load_mode {
        _ if std::env::var("MESH_LLM_BYPASS_SKIPPY_MODEL_LOAD").is_ok() => {
            skippy_runtime::StageModel::new_dummy()
        }
        _ if !admitted_model_parts.is_empty() => {
            open_stage_model_from_parts(&admitted_model_parts, &runtime_config)?
        }
        _ => {
            let Some(model_path) = config.model_path.as_ref().map(std::path::Path::new) else {
                return Ok(None);
            };
            open_stage_model(model_path, &runtime_config)?
        }
    };

    Ok(Some(Arc::new(Mutex::new(RuntimeState {
        model,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        lane_count: config.lane_count,
        ctx_size: config.ctx_size,
        next_lane_index: 0,
        free_lane_indices: Vec::new(),
        sessions: BTreeMap::new(),
        idle_sessions: Vec::new(),
        max_idle_sessions: max_idle_sessions_from_stage_config(config),
        session_token_counts: BTreeMap::new(),
        session_resident_prefixes: BTreeMap::new(),
        session_lifecycle_observer,
        #[cfg(test)]
        modelless_for_test: false,
    }))))
}

pub fn load_runtime_with_overrides_and_open_events(
    config: &StageConfig,
    overrides: &RuntimeLaunchOverrides,
    operation_id: skippy_runtime::OperationId,
    model_open_event_reporter: Option<&mut (dyn FnMut(skippy_runtime::RuntimeEvent) + Send)>,
    session_lifecycle_observer: Option<Arc<dyn SessionLifecycleObserver>>,
) -> Result<Option<Arc<Mutex<RuntimeState>>>> {
    reject_legacy_serving_package(config)?;
    let runtime_config = runtime_config_from_stage_config(config, overrides)?;

    let admitted_model_parts = config
        .model_part_paths
        .iter()
        .map(std::path::PathBuf::from)
        .collect::<Vec<_>>();
    let model = match config.load_mode {
        _ if std::env::var("MESH_LLM_BYPASS_SKIPPY_MODEL_LOAD").is_ok() => {
            skippy_runtime::StageModel::new_dummy()
        }
        _ if !admitted_model_parts.is_empty() => open_stage_model_from_parts_with_events(
            &admitted_model_parts,
            &runtime_config,
            operation_id,
            model_open_event_reporter,
        )?,
        _ => {
            let Some(model_path) = config.model_path.as_ref().map(std::path::Path::new) else {
                return Ok(None);
            };
            open_stage_model_with_events(
                model_path,
                &runtime_config,
                operation_id,
                model_open_event_reporter,
            )?
        }
    };

    Ok(Some(Arc::new(Mutex::new(RuntimeState {
        model,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        lane_count: config.lane_count,
        ctx_size: config.ctx_size,
        next_lane_index: 0,
        free_lane_indices: Vec::new(),
        sessions: BTreeMap::new(),
        idle_sessions: Vec::new(),
        max_idle_sessions: max_idle_sessions_from_stage_config(config),
        session_token_counts: BTreeMap::new(),
        session_resident_prefixes: BTreeMap::new(),
        session_lifecycle_observer,
        #[cfg(test)]
        modelless_for_test: false,
    }))))
}

/// Translates `model_fit.cache_idle_slots` into the idle-session-pool bound.
/// `None`/unset preserves today's unbounded behavior.
fn max_idle_sessions_from_stage_config(config: &StageConfig) -> Option<usize> {
    config.cache_idle_slots.map(|slots| slots as usize)
}

fn reject_legacy_serving_package(config: &StageConfig) -> Result<()> {
    anyhow::ensure!(
        config.load_mode != LoadMode::LayerPackage,
        "layer-package schema v1 is offline-only; split serving requires package-v2 graph admission"
    );
    Ok(())
}

fn runtime_config_from_stage_config(
    config: &StageConfig,
    overrides: &RuntimeLaunchOverrides,
) -> Result<RuntimeConfig> {
    let cache_type_k = parse_cache_type(&config.cache_type_k)
        .with_context(|| format!("parse cache_type_k for {}", config.stage_id))?;
    let cache_type_v = parse_cache_type(&config.cache_type_v)
        .with_context(|| format!("parse cache_type_v for {}", config.stage_id))?;
    let n_threads = overrides
        .n_threads
        .map(u32::try_from)
        .transpose()
        .with_context(|| format!("n_threads exceeds u32 for {}", config.stage_id))?;
    let n_threads_batch = overrides
        .n_threads_batch
        .map(u32::try_from)
        .transpose()
        .with_context(|| format!("n_threads_batch exceeds u32 for {}", config.stage_id))?;
    Ok(RuntimeConfig {
        stage_index: config.stage_index,
        layer_start: config.layer_start,
        layer_end: config.layer_end,
        ctx_size: config.ctx_size,
        lane_count: config.lane_count,
        n_batch: config.n_batch,
        n_ubatch: config.n_ubatch,
        n_threads,
        n_threads_batch,
        n_gpu_layers: config.n_gpu_layers,
        mmap: config.mmap,
        mlock: config.mlock,
        repack: config.repack,
        op_offload: config.op_offload,
        no_host_buffer: config.no_host_buffer,
        check_tensors: config.check_tensors,
        direct_io: config.direct_io,
        main_gpu: config.main_gpu,
        split_mode: match config.split_mode {
            SplitMode::Auto => RuntimeSplitMode::Auto,
            SplitMode::None => RuntimeSplitMode::None,
            SplitMode::Layer => RuntimeSplitMode::Layer,
            SplitMode::Row => RuntimeSplitMode::Row,
            SplitMode::Tensor => RuntimeSplitMode::Tensor,
        },
        selected_backend_device: config
            .selected_device
            .as_ref()
            .map(|device| device.backend_device.clone()),
        cache_type_k,
        cache_type_v,
        flash_attn_type: match config.flash_attn_type {
            FlashAttentionType::Auto => RuntimeFlashAttentionType::Auto,
            FlashAttentionType::Disabled => RuntimeFlashAttentionType::Disabled,
            FlashAttentionType::Enabled => RuntimeFlashAttentionType::Enabled,
        },
        load_mode: match config.load_mode {
            LoadMode::RuntimeSlice => RuntimeLoadMode::RuntimeSlice,
            LoadMode::LayerPackage => RuntimeLoadMode::LayerPackage,
            LoadMode::ArtifactSlice => RuntimeLoadMode::ArtifactSlice,
        },
        kv_offload: config.kv_offload,
        kv_unified: config.kv_unified,
        swa_full: config.swa_full,
        projector_path: config.projector_path.clone(),
        projector_use_gpu: config.projector_use_gpu,
        media_marker: config.media_marker.clone(),
        image_min_tokens: config.image_min_tokens,
        image_max_tokens: config.image_max_tokens,
        batch_max_tokens: config.batch_max_tokens,
        glm_dsa_policy: match config.glm_dsa_policy {
            skippy_protocol::GlmDsaPolicy::Auto => RuntimeGlmDsaPolicy::Auto,
            skippy_protocol::GlmDsaPolicy::V1 => RuntimeGlmDsaPolicy::V1,
        },
        include_embeddings: config.layer_start == 0,
        include_output: config.downstream.is_none(),
        mtp_source: overrides.mtp_source,
        filter_tensors_on_load: config.filter_tensors_on_load,
        resident_tensor_names: config.resident_tensor_names.clone(),
        checkpoint_quantization: config
            .checkpoint_quantization
            .as_deref()
            .unwrap_or("preserve")
            .parse()
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("parse checkpoint_quantization for {}", config.stage_id))?,
        checkpoint_imatrix: config.checkpoint_imatrix.as_deref().map(Into::into),
        checkpoint_imatrix_sha256: config.checkpoint_imatrix_sha256.clone(),
    })
}

fn open_stage_model(path: &std::path::Path, runtime_config: &RuntimeConfig) -> Result<StageModel> {
    StageModel::open(path, runtime_config)
}

fn open_stage_model_with_events(
    path: &std::path::Path,
    runtime_config: &RuntimeConfig,
    operation_id: skippy_runtime::OperationId,
    model_open_event_reporter: Option<&mut (dyn FnMut(skippy_runtime::RuntimeEvent) + Send)>,
) -> Result<StageModel> {
    match model_open_event_reporter {
        Some(event_reporter) => {
            StageModel::open_with_events(path, runtime_config, operation_id, event_reporter)
        }
        None => StageModel::open(path, runtime_config),
    }
}

fn open_stage_model_from_parts(
    paths: &[std::path::PathBuf],
    runtime_config: &RuntimeConfig,
) -> Result<StageModel> {
    StageModel::open_from_parts(paths, runtime_config)
}

fn open_stage_model_from_parts_with_events(
    paths: &[std::path::PathBuf],
    runtime_config: &RuntimeConfig,
    operation_id: skippy_runtime::OperationId,
    model_open_event_reporter: Option<&mut (dyn FnMut(skippy_runtime::RuntimeEvent) + Send)>,
) -> Result<StageModel> {
    match model_open_event_reporter {
        Some(event_reporter) => StageModel::open_from_parts_with_events(
            paths,
            runtime_config,
            operation_id,
            event_reporter,
        ),
        None => StageModel::open_from_parts(paths, runtime_config),
    }
}

#[cfg(test)]
mod tests {
    use skippy_protocol::{
        FlashAttentionType, LoadMode, PeerConfig, SplitMode, StageConfig, StageDevice,
    };
    use skippy_runtime::{
        ActivationDesc, ActivationFrame, CheckpointQuantization,
        FlashAttentionType as RuntimeFlashAttentionType, MtpSource, RuntimeActivationDType,
        RuntimeActivationLayout, RuntimeConfig, SamplingConfig,
    };

    use super::{
        RuntimeLaunchOverrides, RuntimeState, load_runtime_with_overrides,
        max_idle_sessions_from_stage_config, reject_legacy_serving_package,
        runtime_config_from_stage_config,
    };

    #[test]
    fn modelless_runtime_reports_zero_kv_pool_so_scheduler_uses_fallback() {
        // The scheduler derives its admission budget from `kv_pool_tokens()` and
        // keeps its configured default when the runtime reports 0. The modelless
        // test runtime carries no context, so it must report 0 (not panic or
        // report a stale non-zero pool).
        let rt = RuntimeState::new_modelless_for_test(4);
        assert_eq!(rt.kv_pool_tokens(), 0);
        assert_eq!(rt.lane_count(), 4);
    }

    #[test]
    fn runtime_config_preserves_selected_backend_device_and_thread_overrides() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: Some("/tmp/mmproj.gguf".to_string()),
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 24,
            ctx_size: 512,
            lane_count: 2,
            n_batch: Some(1024),
            n_ubatch: Some(256),
            n_gpu_layers: -1,
            mmap: Some(false),
            mlock: true,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Enabled,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: true,
            resident_tensor_names: Vec::new(),
            selected_device: Some(StageDevice {
                backend_device: "Vulkan1".into(),
                stable_id: Some("pci:0000:65:00.0".into()),
                index: Some(1),
                vram_bytes: Some(16_000_000_000),
            }),
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
            ..StageConfig::default()
        };

        let overrides = RuntimeLaunchOverrides {
            n_threads: Some(8),
            n_threads_batch: Some(4),
            mtp_source: MtpSource::External,
        };

        let runtime_config = runtime_config_from_stage_config(&config, &overrides).unwrap();

        assert_eq!(
            runtime_config.selected_backend_device.as_deref(),
            Some("Vulkan1")
        );
        assert_eq!(runtime_config.lane_count, 2);
        assert_eq!(runtime_config.n_batch, Some(1024));
        assert_eq!(runtime_config.n_ubatch, Some(256));
        assert_eq!(runtime_config.n_threads, Some(8));
        assert_eq!(runtime_config.n_threads_batch, Some(4));
        assert_eq!(runtime_config.mmap, Some(false));
        assert!(runtime_config.mlock);
        assert_eq!(
            runtime_config.flash_attn_type,
            RuntimeFlashAttentionType::Enabled
        );
        assert_eq!(runtime_config.mtp_source, MtpSource::External);
    }

    #[test]
    fn runtime_config_parses_checkpoint_quantization() {
        let config = StageConfig {
            stage_id: "stage-0".to_string(),
            layer_end: 1,
            checkpoint_quantization: Some("Q4_K_M".to_string()),
            ..StageConfig::default()
        };

        let runtime_config =
            runtime_config_from_stage_config(&config, &RuntimeLaunchOverrides::default()).unwrap();
        assert_eq!(
            runtime_config.checkpoint_quantization,
            CheckpointQuantization::Q4KM
        );
    }

    #[test]
    fn runtime_config_accepts_importance_aware_checkpoint_quantization() {
        let config = StageConfig {
            stage_id: "stage-0".to_string(),
            layer_end: 1,
            checkpoint_quantization: Some("IQ2_XXS".to_string()),
            checkpoint_imatrix: Some("/models/imatrix.gguf".to_string()),
            checkpoint_imatrix_sha256: Some("a".repeat(64)),
            ..StageConfig::default()
        };

        let runtime =
            runtime_config_from_stage_config(&config, &RuntimeLaunchOverrides::default()).unwrap();
        assert_eq!(
            runtime.checkpoint_quantization,
            CheckpointQuantization::IQ2XXS
        );
        assert_eq!(
            runtime.checkpoint_imatrix.as_deref(),
            Some(std::path::Path::new("/models/imatrix.gguf"))
        );
        assert_eq!(runtime.checkpoint_imatrix_sha256, Some("a".repeat(64)));
    }

    fn fake_stage_config_with_cache_idle_slots(cache_idle_slots: Option<u32>) -> StageConfig {
        StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 24,
            ctx_size: 512,
            lane_count: 4,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots,
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

    #[test]
    fn cache_idle_slots_reaches_the_idle_session_pool_bound() {
        let unset = fake_stage_config_with_cache_idle_slots(None);
        let two = fake_stage_config_with_cache_idle_slots(Some(2));
        let five = fake_stage_config_with_cache_idle_slots(Some(5));

        assert_eq!(max_idle_sessions_from_stage_config(&unset), None);
        assert_eq!(max_idle_sessions_from_stage_config(&two), Some(2));
        assert_eq!(max_idle_sessions_from_stage_config(&five), Some(5));
        assert_ne!(
            max_idle_sessions_from_stage_config(&two),
            max_idle_sessions_from_stage_config(&five),
            "cache_idle_slots=2 and cache_idle_slots=5 must produce different idle-pool bounds"
        );
    }

    #[test]
    fn legacy_layer_packages_are_rejected_before_model_open() {
        let mut config = fake_stage_config_with_cache_idle_slots(None);
        config.load_mode = LoadMode::LayerPackage;

        let error = reject_legacy_serving_package(&config)
            .expect_err("schema-v1 layer packages must be offline-only");

        assert!(error.to_string().contains("package-v2 graph admission"));
    }

    #[test]
    fn runtime_config_does_not_infer_embedding_ownership_from_stage_role() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: Some("/tmp/package".to_string()),
            manifest_sha256: Some("manifest".to_string()),
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/package".to_string()),
            projector_path: None,
            stage_id: "stage-2".to_string(),
            stage_index: 2,
            layer_start: 20,
            layer_end: 30,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: true,
            resident_tensor_names: Vec::new(),
            selected_device: Some(StageDevice {
                backend_device: "CPU".into(),
                stable_id: None,
                index: None,
                vram_bytes: None,
            }),
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::LayerPackage,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: Some(PeerConfig {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                endpoint: "tcp://127.0.0.1:19001".to_string(),
            }),
            downstream: None,
            ..StageConfig::default()
        };

        let runtime_config =
            runtime_config_from_stage_config(&config, &RuntimeLaunchOverrides::default()).unwrap();

        assert!(!runtime_config.include_embeddings);
        assert!(runtime_config.include_output);
        assert_eq!(runtime_config.mtp_source, MtpSource::Disabled);
    }

    fn glm52_mtp_fixture() -> anyhow::Result<Option<(std::path::PathBuf, StageConfig)>> {
        let Some(package_path) =
            std::env::var_os("SKIPPY_GLM52_MTP_PACKAGE").map(std::path::PathBuf::from)
        else {
            return Ok(None);
        };
        if !package_path.join("model-package.json").is_file() {
            eprintln!(
                "skipping: {} does not look like a package-v2 directory",
                package_path.display()
            );
            return Ok(None);
        }
        let manifest_path = package_path.join("model-package.json");
        let manifest: skippy_package_format::PackageManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        let descriptor = skippy_package_format::stage_admission::StageAdmissionDescriptor {
            package_id: manifest.package_id.clone(),
            resident_tensor_ids: manifest
                .tensor_catalog
                .entries
                .iter()
                .filter(|tensor| {
                    matches!(
                        tensor.storage,
                        skippy_package_format::TensorStorage::Owned { .. }
                    )
                })
                .map(|tensor| tensor.id.clone())
                .collect(),
            sidecars: Vec::new(),
        };
        let admission = manifest.resolve_stage_admission(&descriptor)?;
        let mut resident_tensor_names = admission
            .tensor_bindings
            .iter()
            .map(|tensor| tensor.native_name.to_string())
            .collect::<Vec<_>>();
        resident_tensor_names.sort();
        resident_tensor_names.dedup();
        let mut artifacts = admission.required_artifacts;
        artifacts.sort_by(|left, right| {
            let left_primary = left.id == manifest.source_model.metadata_artifact_id;
            let right_primary = right.id == manifest.source_model.metadata_artifact_id;
            right_primary
                .cmp(&left_primary)
                .then_with(|| left.id.cmp(&right.id))
        });
        let model_part_paths = artifacts
            .into_iter()
            .map(|artifact| {
                package_path
                    .join(&artifact.path)
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        let model_path = model_part_paths.first().cloned();
        let config = StageConfig {
            run_id: "glm52-mtp-smoke".to_string(),
            topology_id: "glm52-mtp-smoke-topology".to_string(),
            model_id: "meshllm/GLM-5.2-Q2_K-MTP-Q8-layers".to_string(),
            package_ref: Some(package_path.to_string_lossy().to_string()),
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path,
            model_part_paths,
            projector_path: None,
            stage_id: "stage-final".to_string(),
            stage_index: 1,
            // GLM-DSA stages must begin on a full-indexer layer. Layer 78 is
            // the auxiliary next-token head rather than a base transformer
            // layer, so the smallest valid final-stage fixture is 74..78;
            // native MTP loading retains layer 78's nextn tensors alongside it.
            layer_start: 74,
            layer_end: 78,
            ctx_size: 128,
            lane_count: 1,
            n_batch: Some(1),
            n_ubatch: Some(1),
            n_gpu_layers: 0,
            mmap: Some(true),
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Disabled,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: true,
            resident_tensor_names,
            selected_device: Some(StageDevice {
                backend_device: "CPU".into(),
                stable_id: None,
                index: None,
                vram_bytes: None,
            }),
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: Some(PeerConfig {
                stage_id: "stage-prev".to_string(),
                stage_index: 0,
                endpoint: "tcp://127.0.0.1:19000".to_string(),
            }),
            downstream: None,
            ..StageConfig::default()
        };
        Ok(Some((package_path, config)))
    }

    fn glm52_mtp_input(token_count: u32) -> ActivationFrame {
        let hidden_bytes = 6144 * token_count as usize * std::mem::size_of::<f32>();
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 74,
                token_count,
                sequence_count: 1,
                payload_bytes: hidden_bytes as u64,
                flags: 0,
            },
            payload: vec![0; hidden_bytes],
        }
    }

    #[test]
    fn glm52_final_stage_package_executes_native_mtp_when_fixture_is_set() -> anyhow::Result<()> {
        let Some((_package_path, config)) = glm52_mtp_fixture()? else {
            eprintln!("skipping: SKIPPY_GLM52_MTP_PACKAGE is not set");
            return Ok(());
        };

        let runtime = load_runtime_with_overrides(
            &config,
            &RuntimeLaunchOverrides {
                mtp_source: MtpSource::Integrated,
                ..RuntimeLaunchOverrides::default()
            },
            None,
        )?
        .expect("GLM final stage should load from the package");
        let mut runtime = runtime.lock().expect("runtime mutex poisoned");
        let input = glm52_mtp_input(1);
        let sampling = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let (predicted, draft, _output) =
            runtime.decode_frame_sampled_mtp("smoke", 1, Some(&sampling), Some(&input), 0, 1)?;
        let draft = draft.expect("GLM final stage should return a native MTP draft");

        assert!(predicted >= 0);
        assert_eq!(draft.token_ids.len(), 1);
        assert!(draft.token_ids[0] >= 0);
        let verify_inputs = [predicted, draft.token_ids[0]];
        let (verified, _next_draft, _output) = runtime.verify_frame_sampled(
            "smoke",
            &verify_inputs,
            Some(&sampling),
            Some(&glm52_mtp_input(2)),
            0,
            1,
        )?;
        assert!(!verified.is_empty());
        runtime.retire_verify_checkpoint("smoke", 1, 2)?;
        Ok(())
    }

    #[test]
    fn glm52_final_stage_does_not_create_integrated_mtp_when_disabled() -> anyhow::Result<()> {
        let Some((_package_path, config)) = glm52_mtp_fixture()? else {
            eprintln!("skipping: SKIPPY_GLM52_MTP_PACKAGE is not set");
            return Ok(());
        };
        let runtime = load_runtime_with_overrides(
            &config,
            &RuntimeLaunchOverrides {
                mtp_source: MtpSource::Disabled,
                ..RuntimeLaunchOverrides::default()
            },
            None,
        )?
        .expect("GLM final stage should load from the package");
        let mut runtime = runtime.lock().expect("runtime mutex poisoned");
        let sampling = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let (predicted, draft, _output) = runtime.decode_frame_sampled_mtp(
            "disabled-mtp",
            1,
            Some(&sampling),
            Some(&glm52_mtp_input(1)),
            0,
            1,
        )?;

        assert!(predicted >= 0);
        assert!(
            draft.is_none(),
            "disabled MTP must not create a draft context"
        );
        Ok(())
    }

    #[test]
    fn glm52_external_sidecar_attaches_when_target_has_integrated_mtp_tensors() -> anyhow::Result<()>
    {
        let Some((_package_path, config)) = glm52_mtp_fixture()? else {
            eprintln!("skipping: SKIPPY_GLM52_MTP_PACKAGE is not set");
            return Ok(());
        };
        let Some(sidecar_path) = std::env::var_os("SKIPPY_GLM52_MTP_SIDECAR") else {
            eprintln!("skipping: SKIPPY_GLM52_MTP_SIDECAR is not set");
            return Ok(());
        };
        let sidecar_path = std::path::PathBuf::from(sidecar_path);
        if !sidecar_path.is_file() {
            eprintln!(
                "skipping: {} is not an MTP sidecar GGUF",
                sidecar_path.display()
            );
            return Ok(());
        }

        let runtime = load_runtime_with_overrides(
            &config,
            &RuntimeLaunchOverrides {
                mtp_source: MtpSource::External,
                ..RuntimeLaunchOverrides::default()
            },
            None,
        )?
        .expect("GLM final stage should load from the package");
        let mut runtime = runtime.lock().expect("runtime mutex poisoned");
        runtime.model.attach_mtp_draft_model(
            &sidecar_path,
            &RuntimeConfig {
                ctx_size: config.ctx_size,
                lane_count: config.lane_count,
                n_batch: config.n_batch,
                n_ubatch: config.n_ubatch,
                n_gpu_layers: config.n_gpu_layers,
                mmap: config.mmap,
                mlock: config.mlock,
                selected_backend_device: Some("CPU".to_string()),
                mtp_source: MtpSource::External,
                ..RuntimeConfig::default()
            },
        )?;
        let sampling = SamplingConfig {
            temperature: 0.0,
            ..SamplingConfig::default()
        };
        let (predicted, draft, _output) = runtime.decode_frame_sampled_mtp(
            "external-mtp",
            1,
            Some(&sampling),
            Some(&glm52_mtp_input(1)),
            0,
            1,
        )?;
        let draft = draft.expect("external MTP sidecar should attach to the target");

        assert!(predicted >= 0);
        assert_eq!(draft.token_ids.len(), 1);
        assert!(draft.token_ids[0] >= 0);
        Ok(())
    }

    #[test]
    fn runtime_config_preserves_default_runtime_threads_when_omitted() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 24,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: SplitMode::Auto,
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
        };

        let runtime_config =
            runtime_config_from_stage_config(&config, &RuntimeLaunchOverrides::default()).unwrap();

        assert_eq!(runtime_config.n_threads, None);
        assert_eq!(runtime_config.n_threads_batch, None);
        assert_eq!(runtime_config.n_batch, None);
        assert_eq!(runtime_config.n_ubatch, None);
    }

    #[test]
    fn runtime_config_preserves_multimodal_and_glm_dsa_native_controls() {
        let config: StageConfig = serde_json::from_value(serde_json::json!({
            "run_id": "run-a",
            "topology_id": "topology-a",
            "model_id": "model-a",
            "model_path": "/tmp/model.gguf",
            "projector_path": "/tmp/mmproj.gguf",
            "projector_use_gpu": false,
            "media_marker": "<media>",
            "image_min_tokens": 32,
            "image_max_tokens": 1536,
            "batch_max_tokens": 384,
            "glm_dsa_policy": "v1",
            "generation_signal_window": 20,
            "stage_id": "stage-0",
            "stage_index": 0,
            "layer_start": 0,
            "layer_end": 24,
            "ctx_size": 512,
            "lane_count": 1,
            "n_gpu_layers": -1,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "native_mtp_enabled": true,
            "load_mode": "runtime-slice",
            "bind_addr": "127.0.0.1:0"
        }))
        .expect("stage config should deserialize");

        let runtime_config =
            runtime_config_from_stage_config(&config, &RuntimeLaunchOverrides::default())
                .expect("runtime config should build");
        let debug = format!("{runtime_config:?}");

        assert!(debug.contains("projector_use_gpu: Some(false)"));
        assert!(debug.contains("media_marker: Some(\"<media>\")"));
        assert!(debug.contains("image_min_tokens: Some(32)"));
        assert!(debug.contains("image_max_tokens: Some(1536)"));
        assert!(debug.contains("batch_max_tokens: Some(384)"));
        assert!(debug.contains("glm_dsa_policy: V1"));
    }

    #[test]
    fn runtime_config_rejects_unsupported_cache_type_before_launch() {
        let config = StageConfig {
            run_id: "run-a".to_string(),
            topology_id: "topology-a".to_string(),
            model_id: "model-a".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some("/tmp/model.gguf".to_string()),
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 24,
            ctx_size: 512,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: -1,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: SplitMode::Auto,
            cache_type_k: "auto".to_string(),
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
        };

        let error = runtime_config_from_stage_config(&config, &RuntimeLaunchOverrides::default())
            .expect_err("unsupported cache types should fail during runtime config construction");

        assert!(
            error.to_string().contains("parse cache_type_k for stage-0"),
            "unexpected error: {error:#}"
        );
    }
}
