use crate::SpeculativeDecodeConfig;
use crate::runtime_state::RuntimeState;
use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use skippy_protocol::StageConfig;
use skippy_runtime::FlashAttentionType as RuntimeFlashAttentionType;
use skippy_runtime::RuntimeConfig;
use skippy_runtime::RuntimeLoadMode;
use skippy_runtime::StageModel;
use skippy_runtime::StageSession;
use skippy_runtime::{ModelInfo, MtpSource};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

fn map_split_mode(mode: skippy_protocol::SplitMode) -> skippy_runtime::SplitMode {
    match mode {
        skippy_protocol::SplitMode::Auto => skippy_runtime::SplitMode::Auto,
        skippy_protocol::SplitMode::None => skippy_runtime::SplitMode::None,
        skippy_protocol::SplitMode::Layer => skippy_runtime::SplitMode::Layer,
        skippy_protocol::SplitMode::Row => skippy_runtime::SplitMode::Row,
        skippy_protocol::SplitMode::Tensor => skippy_runtime::SplitMode::Tensor,
    }
}

pub(in crate::frontend) struct DraftRunner {
    pub(in crate::frontend) path: PathBuf,
    pub(in crate::frontend) window: usize,
    pub(in crate::frontend) _model: StageModel,
    pub(in crate::frontend) session: StageSession,
}

impl DraftRunner {
    pub(in crate::frontend) fn open(
        path: &Path,
        config: &StageConfig,
        n_gpu_layers: Option<i32>,
        window: usize,
        speculative: &SpeculativeDecodeConfig,
    ) -> Result<Self> {
        if !path.is_file() {
            bail!("draft model does not exist: {}", path.display());
        }
        let layer_count = model_layer_count(path)?;
        let model = StageModel::open(
            path,
            &draft_runtime_config(
                config,
                n_gpu_layers,
                speculative,
                MtpSource::Disabled,
                layer_count,
            )?,
        )
        .with_context(|| format!("open draft model {}", path.display()))?;
        let session = model.create_session().context("create draft session")?;
        Ok(Self {
            path: path.to_path_buf(),
            window,
            _model: model,
            session,
        })
    }

    pub(in crate::frontend) fn reset_to_context(&mut self, context_tokens: &[i32]) -> Result<()> {
        self.session.reset().context("reset draft session")?;
        if context_tokens.len() > 1 {
            self.session
                .prefill_chunk(&context_tokens[..context_tokens.len() - 1])
                .context("prefill draft context")?;
        }
        Ok(())
    }

    pub(in crate::frontend) fn propose(
        &mut self,
        mut current: i32,
        max_tokens: usize,
    ) -> Result<Vec<i32>> {
        let mut tokens = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            current = self
                .session
                .decode_step(current)
                .context("draft decode step")?;
            tokens.push(current);
        }
        Ok(tokens)
    }
}

pub(in crate::frontend) fn open_draft_runner(
    path: Option<&Path>,
    config: &StageConfig,
    n_gpu_layers: Option<i32>,
    window: usize,
    speculative: &SpeculativeDecodeConfig,
) -> Result<Option<Arc<Mutex<DraftRunner>>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    Ok(Some(Arc::new(Mutex::new(DraftRunner::open(
        path,
        config,
        n_gpu_layers,
        window,
        speculative,
    )?))))
}

pub(in crate::frontend) fn attach_native_mtp_draft_model(
    path: Option<&Path>,
    runtime: &Arc<Mutex<RuntimeState>>,
    config: &StageConfig,
    n_gpu_layers: Option<i32>,
    speculative: &SpeculativeDecodeConfig,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    if !path.is_file() {
        bail!("MTP draft model does not exist: {}", path.display());
    }
    let layer_count = model_layer_count(path)?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("runtime lock poisoned"))?;
    runtime
        .model
        .attach_mtp_draft_model(
            path,
            &draft_runtime_config(
                config,
                n_gpu_layers,
                speculative,
                MtpSource::External,
                layer_count,
            )?,
        )
        .with_context(|| format!("attach MTP draft model {}", path.display()))
}

pub(in crate::frontend) fn draft_runtime_config(
    config: &StageConfig,
    n_gpu_layers: Option<i32>,
    speculative: &SpeculativeDecodeConfig,
    mtp_source: MtpSource,
    layer_count: u32,
) -> Result<RuntimeConfig> {
    let draft_threads = speculative
        .draft_threads
        .map(u32::try_from)
        .transpose()
        .context("draft thread count exceeds u32")?;
    Ok(RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: layer_count,
        ctx_size: config.ctx_size,
        lane_count: match mtp_source {
            MtpSource::Disabled => 1,
            MtpSource::Integrated | MtpSource::External => config.lane_count,
        },
        n_batch: None,
        n_ubatch: None,
        n_threads: draft_threads,
        n_threads_batch: draft_threads,
        n_gpu_layers: n_gpu_layers.unwrap_or(config.n_gpu_layers),
        mmap: config.mmap,
        mlock: config.mlock,
        repack: config.repack,
        op_offload: config.op_offload,
        no_host_buffer: config.no_host_buffer,
        check_tensors: config.check_tensors,
        direct_io: config.direct_io,
        main_gpu: config.main_gpu,
        split_mode: map_split_mode(config.split_mode),
        selected_backend_device: speculative.draft_device.clone().or_else(|| {
            config
                .selected_device
                .as_ref()
                .map(|device| device.backend_device.clone())
        }),
        cache_type_k: skippy_runtime::parse_cache_type(&speculative.draft_cache_type_k)?,
        cache_type_v: skippy_runtime::parse_cache_type(&speculative.draft_cache_type_v)?,
        flash_attn_type: RuntimeFlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_runtime::GlmDsaPolicy::Auto,
        include_embeddings: true,
        include_output: true,
        mtp_source,
        filter_tensors_on_load: false,
        resident_tensor_names: Vec::new(),
        checkpoint_quantization: config
            .checkpoint_quantization
            .as_deref()
            .unwrap_or("preserve")
            .parse()
            .map_err(anyhow::Error::msg)
            .context("parse checkpoint_quantization for draft model")?,
        checkpoint_imatrix: config.checkpoint_imatrix.as_deref().map(Into::into),
        checkpoint_imatrix_sha256: config.checkpoint_imatrix_sha256.clone(),
        kv_offload: config.kv_offload,
        kv_unified: config.kv_unified,
        swa_full: config.swa_full,
    })
}

pub(in crate::frontend) fn model_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    let layer_count = info
        .tensors()?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .ok_or_else(|| anyhow!("could not infer layer count for {}", path.display()))?;
    Ok(layer_count)
}
