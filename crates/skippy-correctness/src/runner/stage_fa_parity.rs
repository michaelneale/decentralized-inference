use anyhow::{Context, Result, bail};
use skippy_runtime::{
    FlashAttentionType, GGML_TYPE_F16, MtpSource, RuntimeConfig, RuntimeLoadMode, StageModel,
    package::{PackageStageRequest, select_layer_package_parts},
};

use crate::cli::StageFaParityArgs;

pub fn stage_fa_parity(args: StageFaParityArgs) -> Result<()> {
    if args.layer_start >= args.layer_end {
        bail!(
            "layer_start ({}) must be less than layer_end ({})",
            args.layer_start,
            args.layer_end
        );
    }
    let enabled = decode_boundary(&args, FlashAttentionType::Enabled)?;
    let disabled = decode_boundary(&args, FlashAttentionType::Disabled)?;
    if enabled.desc != disabled.desc {
        bail!(
            "activation descriptors differ: enabled={:?} disabled={:?}",
            enabled.desc,
            disabled.desc
        );
    }
    if let Some(path) = args.enabled_output.as_deref() {
        std::fs::write(path, &enabled.payload)
            .with_context(|| format!("write enabled activation {}", path.display()))?;
    }
    if let Some(path) = args.disabled_output.as_deref() {
        std::fs::write(path, &disabled.payload)
            .with_context(|| format!("write disabled activation {}", path.display()))?;
    }
    let enabled_values = payload_f32(&enabled.payload)?;
    let disabled_values = payload_f32(&disabled.payload)?;
    if enabled_values.is_empty() || enabled_values.len() != disabled_values.len() {
        bail!(
            "activation payload length mismatch or empty: enabled={} disabled={}",
            enabled_values.len(),
            disabled_values.len()
        );
    }
    let mut max_abs = 0.0_f32;
    let mut sum_sq = 0.0_f64;
    for (lhs, rhs) in enabled_values.iter().zip(&disabled_values) {
        let delta = (lhs - rhs).abs();
        max_abs = max_abs.max(delta);
        sum_sq += f64::from(delta) * f64::from(delta);
    }
    let rms = (sum_sq / enabled_values.len() as f64).sqrt();
    println!(
        "stage_fa_parity elements={} max_abs={max_abs:.8e} rms={rms:.8e}",
        enabled_values.len()
    );
    if max_abs > args.max_abs {
        bail!(
            "stage FA parity max_abs {max_abs:.8e} exceeds tolerance {:.8e}",
            args.max_abs
        );
    }
    Ok(())
}

fn decode_boundary(
    args: &StageFaParityArgs,
    flash_attn_type: FlashAttentionType,
) -> Result<skippy_runtime::ActivationFrame> {
    let config = RuntimeConfig {
        stage_index: 0,
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        mmap: None,
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
        flash_attn_type,
        load_mode: RuntimeLoadMode::LayerPackage,
        projector_path: None,
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_runtime::GlmDsaPolicy::Auto,
        include_embeddings: true,
        include_output: false,
        mtp_source: MtpSource::Disabled,
        filter_tensors_on_load: true,
        resident_tensor_names: Vec::new(),
        checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };
    let selection = select_layer_package_parts(&PackageStageRequest {
        model_id: args.model_id.clone(),
        topology_id: "stage-fa-parity".to_string(),
        package_ref: args.model.display().to_string(),
        stage_id: "stage-0".to_string(),
        layer_start: args.layer_start,
        layer_end: args.layer_end,
        include_embeddings: true,
        include_output: false,
    })
    .context("select package parts")?;
    let model = StageModel::open_from_parts(&selection.absolute_paths, &config)
        .context("open stage model")?;
    let tokens = model
        .tokenize(&args.prompt, true)
        .context("tokenize prompt")?;
    if tokens.is_empty() {
        bail!("prompt produced no token");
    }
    let mut session = model.create_session().context("create stage session")?;
    if tokens.len() == 1 {
        let (_, frame) = session
            .decode_step_frame(tokens[0], None, 0)
            .context("decode stage boundary")?;
        Ok(frame)
    } else {
        session
            .prefill_chunk_frame(&tokens, None, 0)
            .context("prefill stage boundary")
    }
}

fn payload_f32(payload: &[u8]) -> Result<Vec<f32>> {
    if !payload.len().is_multiple_of(4) {
        bail!(
            "activation payload is not f32-aligned: {} bytes",
            payload.len()
        );
    }
    Ok(payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect())
}
