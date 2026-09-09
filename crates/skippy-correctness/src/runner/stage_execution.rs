use std::{
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use model_artifact::ModelIdentity;
use model_hf::HfModelRepository;
use model_ref::ModelRef;
use serde::Deserialize;
use serde_json::json;
use skippy_protocol::binary::{
    StageReply, StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind,
    activation_state_flags_from_frame_flags, recv_reply, write_stage_message,
};
use skippy_runtime::{
    ActivationFrame, GGML_TYPE_F16, MtpSource, RuntimeConfig, RuntimeLoadMode,
    package::{MaterializedPackage, PackageStageRequest, materialize_layer_package_details},
};

use crate::{
    cli::{FlashAttentionArg, RuntimeArgs, StageLoadMode, StatePayloadKind},
    report::{
        BaselineReport, BoundaryReport, NativeMtpSidebandReport, NativeMtpVerificationReport,
        PackagePartReport, PackageStageReport, SplitReport, StageModelReport,
    },
};

pub(in crate::runner) struct FullModelResult {
    pub(in crate::runner) token_id: i32,
    pub(in crate::runner) predicted_token: i32,
    pub(in crate::runner) second_predicted_token: Option<i32>,
}

pub(in crate::runner) struct BinarySplitConfig {
    pub(in crate::runner) stage_server_bin: PathBuf,
    pub(in crate::runner) model: PathBuf,
    pub(in crate::runner) stage_model: Option<PathBuf>,
    pub(in crate::runner) stage_load_mode: StageLoadMode,
    pub(in crate::runner) split_layer: u32,
    pub(in crate::runner) layer_end: u32,
    pub(in crate::runner) ctx_size: u32,
    pub(in crate::runner) n_batch: Option<u32>,
    pub(in crate::runner) n_ubatch: Option<u32>,
    pub(in crate::runner) n_gpu_layers: i32,
    pub(in crate::runner) flash_attn: FlashAttentionArg,
    pub(in crate::runner) prompt: String,
    pub(in crate::runner) stage1_bind_addr: SocketAddr,
    pub(in crate::runner) child_logs: bool,
    pub(in crate::runner) startup_timeout_secs: u64,
    pub(in crate::runner) max_inflight: usize,
    pub(in crate::runner) model_identity: ModelIdentity,
    pub(in crate::runner) native_mtp_verification: bool,
}

pub(in crate::runner) struct BinarySplitResult {
    pub(in crate::runner) token_id: i32,
    pub(in crate::runner) predicted_token: i32,
    pub(in crate::runner) second_predicted_token: Option<i32>,
    pub(in crate::runner) native_mtp: NativeMtpSidebandReport,
    pub(in crate::runner) native_mtp_verification_compute_us: Option<i64>,
    pub(in crate::runner) activation_width: i32,
    pub(in crate::runner) boundary_producer_stage_index: i32,
    pub(in crate::runner) boundary_layer_start: i32,
    pub(in crate::runner) boundary_layer_end: i32,
    pub(in crate::runner) boundary_token_count: u32,
    pub(in crate::runner) boundary_payload_bytes: u64,
    pub(in crate::runner) boundary_wire_payload_bytes: usize,
    pub(in crate::runner) stage_models: Vec<StageModelReport>,
}

pub(in crate::runner) struct BinaryStateHandoffConfig {
    pub(in crate::runner) stage_server_bin: PathBuf,
    pub(in crate::runner) model: PathBuf,
    pub(in crate::runner) stage_model: Option<PathBuf>,
    pub(in crate::runner) stage_load_mode: StageLoadMode,
    pub(in crate::runner) state_layer_start: u32,
    pub(in crate::runner) state_layer_end: u32,
    pub(in crate::runner) state_stage_index: u32,
    pub(in crate::runner) layer_end: u32,
    pub(in crate::runner) ctx_size: u32,
    pub(in crate::runner) n_batch: Option<u32>,
    pub(in crate::runner) n_ubatch: Option<u32>,
    pub(in crate::runner) n_gpu_layers: i32,
    pub(in crate::runner) flash_attn: FlashAttentionArg,
    pub(in crate::runner) prompt: String,
    pub(in crate::runner) source_bind_addr: SocketAddr,
    pub(in crate::runner) restore_bind_addr: SocketAddr,
    pub(in crate::runner) activation_width: i32,
    pub(in crate::runner) state_payload_kind: StatePayloadKind,
    pub(in crate::runner) prefix_token_count: Option<usize>,
    pub(in crate::runner) cache_hit_repeats: usize,
    pub(in crate::runner) runtime_lane_count: Option<u32>,
    pub(in crate::runner) borrow_resident_hits: bool,
    pub(in crate::runner) cache_decoded_result_hits: bool,
    pub(in crate::runner) skip_suffix_prefill_check: bool,
    pub(in crate::runner) synthetic_input_activation: bool,
    pub(in crate::runner) binary_control: bool,
    pub(in crate::runner) child_logs: bool,
    pub(in crate::runner) startup_timeout_secs: u64,
    pub(in crate::runner) max_inflight: usize,
    pub(in crate::runner) model_identity: ModelIdentity,
}

pub(in crate::runner) struct BinaryDecodeMessageArgs<'a> {
    pub(in crate::runner) token_id: i32,
    pub(in crate::runner) decode_step: i32,
    pub(in crate::runner) source_stage_index: i32,
    pub(in crate::runner) boundary: &'a ActivationFrame,
    pub(in crate::runner) activation_width: i32,
    pub(in crate::runner) request_id: u64,
    pub(in crate::runner) session_id: u64,
}

pub(in crate::runner) fn binary_decode_message(
    args: BinaryDecodeMessageArgs<'_>,
) -> Result<StageWireMessage> {
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
    state.prompt_token_count = 0;
    state.decode_step = args.decode_step;
    state.current_token = args.token_id;
    state.source_stage_index = args.source_stage_index;
    state.flags |= activation_state_flags(args.boundary);
    let activation = skippy_protocol::binary::encode_activation_payload_with_state_flags(
        state.activation_codec,
        1,
        args.activation_width,
        &args.boundary.payload,
        activation_state_flags(args.boundary),
    )
    .context("failed to encode boundary activation for wire")?;
    Ok(StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: args.decode_step,
        token_count: 1,
        state,
        request_id: args.request_id,
        session_id: args.session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![args.token_id],
        positions: vec![args.decode_step],
        activation,
        raw_bytes: Vec::new(),
    })
}

pub(in crate::runner) fn ensure_reply_kind(
    reply: &StageReply,
    expected: WireReplyKind,
) -> Result<()> {
    if reply.kind != expected {
        bail!("expected {expected:?} reply, got {:?}", reply.kind);
    }
    Ok(())
}
pub(in crate::runner) struct CorrectnessTopologyStage<'a> {
    pub(in crate::runner) stage_id: &'a str,
    pub(in crate::runner) stage_index: u32,
    pub(in crate::runner) endpoint: String,
    pub(in crate::runner) layer_start: u32,
    pub(in crate::runner) layer_end: u32,
    pub(in crate::runner) load_mode: &'static str,
}

pub(in crate::runner) fn correctness_topology(
    topology_id: &str,
    model_id: &str,
    stages: &[CorrectnessTopologyStage<'_>],
) -> serde_json::Value {
    json!({
        "topology_id": topology_id,
        "model_id": model_id,
        "stages": stages.iter().map(|stage| {
            json!({
                "stage_id": stage.stage_id,
                "stage_index": stage.stage_index,
                "host": "localhost",
                "endpoint": stage.endpoint,
                "layer_start": stage.layer_start,
                "layer_end": stage.layer_end,
                "load_mode": stage.load_mode,
            })
        }).collect::<Vec<_>>(),
    })
}

pub(in crate::runner) fn send_generation_config(
    stream: &mut std::net::TcpStream,
    request_id: u64,
    session_id: u64,
    prompt_token_count: usize,
) -> Result<()> {
    let message = StageWireMessage::configure_generation(
        request_id,
        session_id,
        i32::try_from(prompt_token_count).context("prompt token count exceeds i32")?,
        None,
        None,
    );
    write_stage_message(&mut *stream, &message).context("send configure-generation")?;
    let reply = recv_reply(&mut *stream).context("receive configure-generation ACK")?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected configure-generation ACK, got {:?}", reply.kind);
    }
    Ok(())
}
pub(in crate::runner) fn stage_id_for_index(stage_index: u32) -> &'static str {
    match stage_index {
        0 => "stage-0",
        1 => "stage-1",
        2 => "stage-2",
        _ => "stage-n",
    }
}

pub(in crate::runner) fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

pub(in crate::runner) fn elapsed_us(started: Instant) -> i64 {
    (started.elapsed().as_secs_f64() * 1_000_000.0).round() as i64
}

pub(in crate::runner) fn mean_pair_sum(left: &[f64], right: &[f64]) -> f64 {
    let count = left.len().min(right.len());
    if count == 0 {
        return 0.0;
    }
    left.iter()
        .zip(right.iter())
        .take(count)
        .map(|(left, right)| left + right)
        .sum::<f64>()
        / count as f64
}

pub(in crate::runner) fn speedup(recompute_ms: f64, cache_ms: f64) -> f64 {
    if cache_ms <= f64::EPSILON {
        return 0.0;
    }
    recompute_ms / cache_ms
}
pub(in crate::runner) fn activation_state_flags(frame: &ActivationFrame) -> i32 {
    activation_state_flags_from_frame_flags(frame.desc.flags)
}

pub(in crate::runner) fn baseline_report(result: FullModelResult) -> BaselineReport {
    BaselineReport {
        token_id: result.token_id,
        predicted_token: result.predicted_token,
        second_predicted_token: result.second_predicted_token,
    }
}

pub(in crate::runner) fn split_report(
    result: BinarySplitResult,
    native_mtp_verification: Option<NativeMtpVerificationReport>,
) -> SplitReport {
    SplitReport {
        token_id: result.token_id,
        predicted_token: result.predicted_token,
        second_predicted_token: result.second_predicted_token,
        native_mtp: result.native_mtp,
        native_mtp_verification,
        activation_width: result.activation_width,
        boundary: BoundaryReport {
            producer_stage_index: result.boundary_producer_stage_index,
            layer_start: result.boundary_layer_start,
            layer_end: result.boundary_layer_end,
            token_count: result.boundary_token_count,
            payload_bytes: result.boundary_payload_bytes,
            wire_payload_bytes: result.boundary_wire_payload_bytes,
        },
    }
}

#[derive(Clone, Copy)]
pub(in crate::runner) struct PackageStageSpec {
    pub(in crate::runner) topology_id: &'static str,
    pub(in crate::runner) stage_id: &'static str,
    pub(in crate::runner) stage_index: u32,
    pub(in crate::runner) layer_start: u32,
    pub(in crate::runner) layer_end: u32,
    pub(in crate::runner) include_embeddings: bool,
    pub(in crate::runner) include_output: bool,
}

pub(in crate::runner) struct StageModelResolution {
    pub(in crate::runner) path: PathBuf,
    pub(in crate::runner) report: StageModelReport,
}

#[derive(Debug, Deserialize)]
struct SliceManifest {
    pub(in crate::runner) stages: Vec<SliceManifestStage>,
}

#[derive(Debug, Deserialize)]
struct SliceManifestStage {
    pub(in crate::runner) stage_index: usize,
    pub(in crate::runner) path: String,
}

pub(in crate::runner) fn stage_model_resolution(
    baseline_model: &Path,
    stage_model: Option<&PathBuf>,
    stage_load_mode: StageLoadMode,
    model_identity: &ModelIdentity,
    spec: PackageStageSpec,
) -> Result<StageModelResolution> {
    let (path, package) = match stage_load_mode {
        StageLoadMode::RuntimeSlice => (baseline_model.to_path_buf(), None),
        StageLoadMode::ArtifactSlice => (artifact_stage_path(stage_model, spec.stage_index)?, None),
        StageLoadMode::LayerPackage => {
            let package_ref = layer_package_ref(baseline_model, stage_model);
            let package_ref = package_ref.to_string_lossy().into_owned();
            let materialized = materialize_layer_package_details(&PackageStageRequest {
                model_id: model_identity.model_id.clone(),
                topology_id: spec.topology_id.to_string(),
                package_ref: package_ref.clone(),
                stage_id: spec.stage_id.to_string(),
                layer_start: spec.layer_start,
                layer_end: spec.layer_end,
                include_embeddings: spec.include_embeddings,
                include_output: spec.include_output,
            })?;
            let path = materialized.output_path.clone();
            (path, Some(package_stage_report(package_ref, materialized)))
        }
    };
    Ok(StageModelResolution {
        report: StageModelReport {
            stage_index: spec.stage_index,
            layer_start: spec.layer_start,
            layer_end: spec.layer_end,
            load_mode: protocol_load_mode(stage_load_mode),
            model_path: path.to_string_lossy().into_owned(),
            package,
        },
        path,
    })
}

pub(in crate::runner) fn package_stage_report(
    package_ref: String,
    materialized: MaterializedPackage,
) -> PackageStageReport {
    PackageStageReport {
        package_ref,
        materialized_path: materialized.output_path.to_string_lossy().into_owned(),
        manifest_sha256: materialized.manifest_sha256,
        selected_parts: materialized
            .selected_parts
            .into_iter()
            .map(|part| PackagePartReport {
                role: part.role,
                layer_index: part.layer_index,
                path: part.path.to_string_lossy().into_owned(),
                sha256: part.sha256,
                artifact_bytes: part.artifact_bytes,
            })
            .collect(),
    }
}

pub(in crate::runner) fn stage_server_model_path(
    baseline_model: &Path,
    stage_model: Option<&PathBuf>,
    stage_load_mode: StageLoadMode,
    spec: PackageStageSpec,
) -> Result<String> {
    match stage_load_mode {
        StageLoadMode::RuntimeSlice => Ok(baseline_model.to_string_lossy().into_owned()),
        StageLoadMode::ArtifactSlice => Ok(artifact_stage_path(stage_model, spec.stage_index)?
            .to_string_lossy()
            .into_owned()),
        StageLoadMode::LayerPackage => Ok(layer_package_ref(baseline_model, stage_model)
            .to_string_lossy()
            .into_owned()),
    }
}

pub(in crate::runner) fn tokenizer_model_for_state_handoff(
    args: &BinaryStateHandoffConfig,
) -> Result<(PathBuf, RuntimeConfig)> {
    let (path, layer_end, load_mode, filter_tensors_on_load) = match args.stage_load_mode {
        StageLoadMode::LayerPackage => {
            let package_ref = layer_package_ref(&args.model, args.stage_model.as_ref());
            let package_ref_string = package_ref.to_string_lossy().into_owned();
            let materialized = materialize_layer_package_details(&PackageStageRequest {
                model_id: args.model_identity.model_id.clone(),
                topology_id: "correctness-tokenizer".to_string(),
                package_ref: package_ref_string,
                stage_id: "tokenizer".to_string(),
                layer_start: 0,
                layer_end: 1,
                include_embeddings: true,
                include_output: false,
            })?;
            (
                materialized.output_path,
                1,
                RuntimeLoadMode::LayerPackage,
                true,
            )
        }
        StageLoadMode::ArtifactSlice => {
            let path = artifact_stage_path(args.stage_model.as_ref(), 0)?;
            (path, args.layer_end, RuntimeLoadMode::ArtifactSlice, true)
        }
        StageLoadMode::RuntimeSlice => (
            args.model.clone(),
            args.layer_end,
            RuntimeLoadMode::RuntimeSlice,
            false,
        ),
    };

    Ok((
        path,
        RuntimeConfig {
            stage_index: 0,
            layer_start: 0,
            layer_end,
            ctx_size: args.ctx_size,
            lane_count: 1,
            n_batch: args.n_batch,
            n_ubatch: args.n_ubatch,
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
            load_mode,
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
            filter_tensors_on_load,
            resident_tensor_names: Vec::new(),
            checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
            checkpoint_imatrix: None,
            checkpoint_imatrix_sha256: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: runtime_flash_attn(args.flash_attn),
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
        },
    ))
}

pub(in crate::runner) fn layer_package_ref<'a>(
    baseline_model: &'a Path,
    stage_model: Option<&'a PathBuf>,
) -> &'a Path {
    stage_model.map(PathBuf::as_path).unwrap_or(baseline_model)
}

pub(in crate::runner) fn runtime_load_mode(stage_load_mode: StageLoadMode) -> RuntimeLoadMode {
    match stage_load_mode {
        StageLoadMode::RuntimeSlice => RuntimeLoadMode::RuntimeSlice,
        StageLoadMode::ArtifactSlice => RuntimeLoadMode::ArtifactSlice,
        StageLoadMode::LayerPackage => RuntimeLoadMode::LayerPackage,
    }
}

pub(in crate::runner) fn runtime_flash_attn(
    value: FlashAttentionArg,
) -> skippy_runtime::FlashAttentionType {
    match value {
        FlashAttentionArg::Auto => skippy_runtime::FlashAttentionType::Auto,
        FlashAttentionArg::Disabled => skippy_runtime::FlashAttentionType::Disabled,
        FlashAttentionArg::Enabled => skippy_runtime::FlashAttentionType::Enabled,
    }
}

pub(in crate::runner) fn protocol_flash_attn(value: FlashAttentionArg) -> &'static str {
    match value {
        FlashAttentionArg::Auto => "auto",
        FlashAttentionArg::Disabled => "disabled",
        FlashAttentionArg::Enabled => "enabled",
    }
}

pub(in crate::runner) fn protocol_load_mode(stage_load_mode: StageLoadMode) -> &'static str {
    match stage_load_mode {
        StageLoadMode::RuntimeSlice => "runtime-slice",
        StageLoadMode::ArtifactSlice => "artifact-slice",
        StageLoadMode::LayerPackage => "layer-package",
    }
}

fn artifact_stage_path(stage_model: Option<&PathBuf>, stage_index: u32) -> Result<PathBuf> {
    let stage_model =
        stage_model.context("--stage-model is required when --stage-load-mode artifact-slice")?;
    if stage_model.is_dir() {
        let manifest_path = stage_model.join("slice-manifest.json");
        if manifest_path.is_file() {
            return artifact_stage_path_from_manifest(&manifest_path, stage_index);
        }
        return Ok(stage_model.join(format!("stage-{stage_index:03}.gguf")));
    }
    if stage_model
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "slice-manifest.json")
    {
        return artifact_stage_path_from_manifest(stage_model, stage_index);
    }
    if stage_index == 0 {
        return Ok(stage_model.to_path_buf());
    }
    bail!(
        "artifact-slice --stage-model must be a slice directory or slice-manifest.json for multi-stage correctness"
    )
}

fn artifact_stage_path_from_manifest(manifest_path: &Path, stage_index: u32) -> Result<PathBuf> {
    let manifest: SliceManifest = serde_json::from_str(
        &fs::read_to_string(manifest_path)
            .with_context(|| format!("read slice manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse slice manifest {}", manifest_path.display()))?;
    let stage_index = stage_index as usize;
    let stage = manifest
        .stages
        .iter()
        .find(|stage| stage.stage_index == stage_index)
        .with_context(|| format!("slice manifest is missing stage {stage_index}"))?;
    let path = PathBuf::from(&stage.path);
    if path.is_absolute() {
        Ok(path)
    } else {
        Ok(manifest_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join(path))
    }
}

pub(in crate::runner) fn runtime_model_identity(args: &RuntimeArgs) -> Result<ModelIdentity> {
    if let Some(model_id) = args.model_id.as_ref() {
        let model_ref = ModelRef::parse(model_id)
            .with_context(|| format!("--model-id must be a model coordinate, got {model_id:?}"))?;
        return Ok(ModelIdentity::from_model_id(model_ref.display_id()));
    }

    if let Some(identity) = HfModelRepository::from_env()
        .ok()
        .and_then(|repository| repository.identity_for_path(&args.model))
    {
        return Ok(identity.to_model_identity());
    }

    bail!(
        "--model-id is required for local model paths that are not in the Hugging Face cache; pass a coordinate like org/repo:Q4_K_M"
    )
}

pub(in crate::runner) fn parse_chain_splits(spec: &str) -> Result<(u32, u32)> {
    let splits = parse_csv(spec)?
        .into_iter()
        .map(|value| {
            value
                .parse::<u32>()
                .with_context(|| format!("invalid split {value}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if splits.len() != 2 {
        bail!("--splits for chain must contain exactly two comma-separated layer indexes");
    }
    Ok((splits[0], splits[1]))
}

pub(in crate::runner) fn parse_split_list(spec: &str) -> Result<Vec<u32>> {
    if let Some((start, end)) = spec.split_once("..") {
        let start = start.parse::<u32>().context("invalid split range start")?;
        let end = end.parse::<u32>().context("invalid split range end")?;
        if start >= end {
            bail!("split range start must be less than end");
        }
        return Ok((start..end).collect());
    }
    parse_csv(spec)?
        .into_iter()
        .map(|value| {
            value
                .parse::<u32>()
                .with_context(|| format!("invalid split {value}"))
        })
        .collect()
}

pub(in crate::runner) fn parse_csv(spec: &str) -> Result<Vec<String>> {
    let values = spec
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if values.is_empty() {
        bail!("list must not be empty");
    }
    Ok(values)
}

pub(in crate::runner) fn configure_child_logs(command: &mut Command, child_logs: bool) {
    if child_logs {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
}

pub(in crate::runner) fn ensure_matches(matches: bool, allow_mismatch: bool) -> Result<()> {
    if !matches && !allow_mismatch {
        bail!("staged execution did not match full-model baseline");
    }
    Ok(())
}

pub(in crate::runner) fn status(matches: bool) -> &'static str {
    if matches { "pass" } else { "fail" }
}
