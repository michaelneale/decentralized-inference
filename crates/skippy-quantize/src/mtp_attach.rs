use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Serialize;
use skippy_runtime::{
    FlashAttentionType, GGML_TYPE_F16, ModelInfo, MtpSource, RuntimeConfig, RuntimeLoadMode,
    StageModel,
};

use crate::output::{print_json_pretty, print_success};

#[derive(Debug, Parser)]
pub(crate) struct ValidateMtpAttachArgs {
    #[arg(long = "model", required = true)]
    model_parts: Vec<PathBuf>,
    #[arg(long)]
    mtp_draft: PathBuf,
    #[arg(long)]
    layer_count: u32,
    #[arg(long)]
    mtp_layer_count: Option<u32>,
    #[arg(long)]
    projector: Option<PathBuf>,
    #[arg(long, default_value_t = 64)]
    ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    n_gpu_layers: i32,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Serialize)]
struct MtpAttachReport {
    model_parts: Vec<PathBuf>,
    mtp_draft: PathBuf,
    projector: Option<PathBuf>,
    layer_count: u32,
    mtp_layer_count: u32,
    ctx_size: u32,
    native_mtp_multimodal_feature: bool,
    session_created: bool,
}

pub(crate) fn run_validate_mtp_attach(args: ValidateMtpAttachArgs) -> Result<()> {
    validate_paths(&args)?;
    ensure!(
        skippy_ffi::native_runtime_loaded(),
        "validate-mtp-attach requires a statically linked standalone build or a loaded native runtime"
    );
    let abi_features = skippy_ffi::try_abi_features()
        .context("loaded native runtime does not expose Skippy ABI feature probing")?;
    let native_mtp_multimodal_feature = abi_features & skippy_ffi::FEATURE_INKLING_MTP_MM != 0;
    ensure!(
        native_mtp_multimodal_feature,
        "native runtime does not advertise Inkling multimodal MTP support"
    );
    let target_config = runtime_config(
        args.layer_count,
        args.ctx_size,
        args.n_gpu_layers,
        args.projector.as_deref(),
        MtpSource::Disabled,
    );
    let mut target = if args.model_parts.len() == 1 {
        StageModel::open(&args.model_parts[0], &target_config)
    } else {
        StageModel::open_from_parts(&args.model_parts, &target_config)
    }
    .context("open target model for MTP attach validation")?;

    let mtp_layer_count = match args.mtp_layer_count {
        Some(layer_count) => layer_count,
        None => infer_layer_count(&args.mtp_draft)?,
    };
    ensure!(
        mtp_layer_count > 0,
        "--mtp-layer-count must be greater than zero"
    );
    let draft_config = runtime_config(
        mtp_layer_count,
        args.ctx_size,
        args.n_gpu_layers,
        None,
        MtpSource::External,
    );
    target
        .attach_mtp_draft_model(&args.mtp_draft, &draft_config)
        .context("attach MTP draft model")?;
    let _session = target
        .create_session()
        .context("create target session after MTP attach")?;

    let report = MtpAttachReport {
        model_parts: args.model_parts,
        mtp_draft: args.mtp_draft,
        projector: args.projector,
        layer_count: args.layer_count,
        mtp_layer_count,
        ctx_size: args.ctx_size,
        native_mtp_multimodal_feature,
        session_created: true,
    };
    if args.json {
        print_json_pretty(&report)?;
    } else {
        print_success(format!(
            "MTP attach valid: parts={} layers={} mtp_layers={} projector={} session_created=true",
            report.model_parts.len(),
            report.layer_count,
            report.mtp_layer_count,
            report
                .projector
                .as_deref()
                .map_or_else(|| "none".to_string(), |path| path.display().to_string())
        ));
    }
    Ok(())
}

fn validate_paths(args: &ValidateMtpAttachArgs) -> Result<()> {
    ensure!(
        args.layer_count > 0,
        "--layer-count must be greater than zero"
    );
    ensure!(args.ctx_size > 0, "--ctx-size must be greater than zero");
    for path in &args.model_parts {
        ensure!(
            path.is_file(),
            "model part does not exist: {}",
            path.display()
        );
    }
    ensure!(
        args.mtp_draft.is_file(),
        "MTP draft does not exist: {}",
        args.mtp_draft.display()
    );
    if let Some(projector) = &args.projector {
        ensure!(
            projector.is_file(),
            "projector does not exist: {}",
            projector.display()
        );
    }
    Ok(())
}

fn infer_layer_count(path: &Path) -> Result<u32> {
    ModelInfo::open(path)
        .with_context(|| format!("open MTP model info {}", path.display()))?
        .tensors()?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .and_then(|index| index.checked_add(1))
        .context("MTP draft contains no layer-indexed tensors")
}

fn runtime_config(
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    projector: Option<&Path>,
    mtp_source: MtpSource,
) -> RuntimeConfig {
    RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end,
        ctx_size,
        lane_count: 1,
        n_batch: Some(ctx_size),
        n_ubatch: Some(ctx_size),
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers,
        mmap: Some(true),
        mlock: false,
        repack: false,
        op_offload: None,
        no_host_buffer: false,
        check_tensors: false,
        direct_io: false,
        main_gpu: None,
        split_mode: skippy_runtime::SplitMode::Auto,
        selected_backend_device: None,
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: projector.map(|path| path.display().to_string()),
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
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
        checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attach_probe_uses_small_mmap_runtime_config() {
        let config = runtime_config(
            66,
            64,
            0,
            Some(Path::new("/models/mmproj.gguf")),
            MtpSource::Disabled,
        );

        assert_eq!(config.layer_start, 0);
        assert_eq!(config.layer_end, 66);
        assert_eq!(config.ctx_size, 64);
        assert_eq!(config.n_batch, Some(64));
        assert_eq!(config.n_ubatch, Some(64));
        assert_eq!(config.mmap, Some(true));
        assert_eq!(
            config.projector_path.as_deref(),
            Some("/models/mmproj.gguf")
        );
        assert!(config.include_embeddings);
        assert!(config.include_output);
    }
}
