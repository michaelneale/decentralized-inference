use std::{fs, net::SocketAddr, process::Command, time::Instant};

use anyhow::{Context, Result, bail};
use model_artifact::ModelIdentity;
use serde_json::json;
use skippy_protocol::binary::{StageWireMessage, WireReplyKind, recv_reply, write_stage_message};
use skippy_runtime::{GGML_TYPE_F16, MtpSource, RuntimeConfig, RuntimeLoadMode, StageModel};

use crate::{
    cli::{RuntimeArgs, ServerArgs, SingleStepArgs},
    report::SingleStepReport,
    support::{
        ChildGuard, activation_width, connect_ready_child, generate_run_id, temp_config_path_for,
    },
};

use super::{
    native_mtp::{
        NativeMtpRequirement, emit_report, ensure_native_mtp_artifact_if_required,
        native_mtp_requirement, native_mtp_satisfies_requirement, native_mtp_sideband_report,
        native_mtp_verification_report, native_mtp_verification_satisfies_requirement,
    },
    stage_execution::{
        BinaryDecodeMessageArgs, BinarySplitConfig, BinarySplitResult, CorrectnessTopologyStage,
        FullModelResult, PackageStageSpec, baseline_report, binary_decode_message,
        configure_child_logs, correctness_topology, elapsed_us, ensure_matches, ensure_reply_kind,
        protocol_flash_attn, protocol_load_mode, runtime_flash_attn, runtime_load_mode,
        runtime_model_identity, send_generation_config, split_report, stage_model_resolution,
        stage_server_model_path, status,
    },
};

pub fn single_step(args: SingleStepArgs) -> Result<()> {
    let native_mtp = native_mtp_requirement(args.native_mtp);
    ensure_native_mtp_artifact_if_required(&args.runtime, native_mtp)?;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let baseline = run_full_model_decode(&args.runtime)?;
    let report = run_single_step_with_baseline(
        &args.runtime,
        &args.server,
        &model_identity,
        baseline,
        SingleStepCase {
            split_layer: args.split_layer,
            stage1_bind_addr: args.stage1_bind_addr,
            native_mtp,
        },
    )?;
    emit_report(&report, args.output.report_out.as_deref())?;
    ensure_matches(report.matches, args.allow_mismatch)?;
    Ok(())
}
pub(in crate::runner) struct SingleStepCase {
    pub(in crate::runner) split_layer: u32,
    pub(in crate::runner) stage1_bind_addr: SocketAddr,
    pub(in crate::runner) native_mtp: NativeMtpRequirement,
}

pub(in crate::runner) fn run_single_step_with_baseline(
    runtime: &RuntimeArgs,
    server: &ServerArgs,
    model_identity: &ModelIdentity,
    baseline: FullModelResult,
    case: SingleStepCase,
) -> Result<SingleStepReport> {
    let split = run_binary_split(BinarySplitConfig {
        stage_server_bin: server.stage_server_bin.clone(),
        model: runtime.model.clone(),
        stage_model: runtime.stage_model.clone(),
        stage_load_mode: runtime.stage_load_mode,
        split_layer: case.split_layer,
        layer_end: runtime.layer_end,
        ctx_size: runtime.ctx_size,
        n_batch: runtime.n_batch,
        n_ubatch: runtime.n_ubatch,
        n_gpu_layers: runtime.n_gpu_layers,
        flash_attn: runtime.flash_attn,
        prompt: runtime.prompt.clone(),
        stage1_bind_addr: case.stage1_bind_addr,
        child_logs: server.child_logs,
        startup_timeout_secs: server.startup_timeout_secs,
        max_inflight: server.max_inflight,
        model_identity: model_identity.clone(),
        native_mtp_verification: case.native_mtp.require_draft,
    })?;
    let native_mtp_verification = native_mtp_verification_report(
        case.native_mtp.require_draft,
        &split.native_mtp,
        split.second_predicted_token,
        baseline.second_predicted_token,
        split.native_mtp_verification_compute_us,
    );
    let matches = baseline.predicted_token == split.predicted_token
        && native_mtp_satisfies_requirement(&split.native_mtp, case.native_mtp)
        && native_mtp_verification_satisfies_requirement(&native_mtp_verification, case.native_mtp);
    let stage_models = split.stage_models.clone();
    Ok(SingleStepReport {
        mode: "single-step",
        status: status(matches),
        model_identity: model_identity.clone(),
        matches,
        native_mtp_draft_required: case.native_mtp.require_draft,
        baseline: baseline_report(baseline),
        split: split_report(split, native_mtp_verification),
        stage_models,
    })
}

pub(in crate::runner) fn run_full_model_decode(args: &RuntimeArgs) -> Result<FullModelResult> {
    let config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.layer_end,
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
        mtp_source: MtpSource::Disabled,
        filter_tensors_on_load: false,
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
    };
    let model = StageModel::open(&args.model, &config).context("failed to open full model")?;
    let tokens = model
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt with full model")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session = model
        .create_session()
        .context("failed to create full-model session")?;
    let predicted_token = session
        .decode_step_frame(token_id, None, 0)
        .context("full model failed to decode")?
        .0;
    let second_predicted_token = session
        .decode_step_frame(predicted_token, None, 0)
        .context("full model failed to decode second token")?
        .0;
    Ok(FullModelResult {
        token_id,
        predicted_token,
        second_predicted_token: Some(second_predicted_token),
    })
}
pub(in crate::runner) fn run_binary_split(args: BinarySplitConfig) -> Result<BinarySplitResult> {
    if args.split_layer == 0 || args.split_layer >= args.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    let stage0_spec = PackageStageSpec {
        topology_id: "correctness-single-step",
        stage_id: "stage-0",
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
        include_embeddings: true,
        include_output: false,
    };
    let stage1_spec = PackageStageSpec {
        topology_id: "correctness-single-step",
        stage_id: "stage-1",
        stage_index: 1,
        layer_start: args.split_layer,
        layer_end: args.layer_end,
        include_embeddings: false,
        include_output: true,
    };
    let stage0_resolution = stage_model_resolution(
        &args.model,
        args.stage_model.as_ref(),
        args.stage_load_mode,
        &args.model_identity,
        stage0_spec,
    )?;
    let stage1_resolution = stage_model_resolution(
        &args.model,
        args.stage_model.as_ref(),
        args.stage_load_mode,
        &args.model_identity,
        stage1_spec,
    )?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
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
        load_mode: runtime_load_mode(args.stage_load_mode),
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
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: runtime_flash_attn(args.flash_attn),
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };
    let stage0 = StageModel::open(&stage0_resolution.path, &stage0_config)
        .context("failed to open stage 0")?;
    let tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let (_boundary_prediction, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty activation frame");
    }
    let activation_width = activation_width(&boundary)?;

    let run_id = generate_run_id();
    let model_id = args.model_identity.model_id.clone();
    let config_path = temp_config_path_for(&run_id, "stage-1");
    let topology_path = temp_config_path_for(&run_id, "topology");
    let config = json!({
        "run_id": run_id,
        "topology_id": "correctness-single-step",
        "model_id": model_id,
        "model_path": stage_server_model_path(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            stage1_spec,
        )?,
        "stage_id": "stage-1",
        "stage_index": 1,
        "layer_start": args.split_layer,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_batch": args.n_batch,
        "n_ubatch": args.n_ubatch,
        "n_gpu_layers": args.n_gpu_layers,
        "flash_attn_type": protocol_flash_attn(args.flash_attn),
        "filter_tensors_on_load": true,
        "load_mode": protocol_load_mode(args.stage_load_mode),
        "bind_addr": args.stage1_bind_addr,
        "upstream": {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": null
    });
    let topology = correctness_topology(
        "correctness-single-step",
        &model_id,
        &[
            CorrectnessTopologyStage {
                stage_id: "stage-0",
                stage_index: 0,
                endpoint: "driver".to_string(),
                layer_start: 0,
                layer_end: args.split_layer,
                load_mode: protocol_load_mode(args.stage_load_mode),
            },
            CorrectnessTopologyStage {
                stage_id: "stage-1",
                stage_index: 1,
                endpoint: format!("tcp://{}", args.stage1_bind_addr),
                layer_start: args.split_layer,
                layer_end: args.layer_end,
                load_mode: protocol_load_mode(args.stage_load_mode),
            },
        ],
    );
    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)
        .with_context(|| format!("failed to write {}", config_path.display()))?;
    fs::write(&topology_path, serde_json::to_vec_pretty(&topology)?)
        .with_context(|| format!("failed to write {}", topology_path.display()))?;

    let mut stage_command = Command::new(&args.stage_server_bin);
    stage_command.args([
        "serve-binary",
        "--config",
        config_path
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--topology",
        topology_path
            .to_str()
            .context("topology path is not valid UTF-8")?,
        "--max-inflight",
        &args.max_inflight.to_string(),
    ]);
    configure_child_logs(&mut stage_command, args.child_logs);
    let mut stage1 = ChildGuard::spawn(stage_command)?;

    let mut stream = connect_ready_child(
        args.stage1_bind_addr,
        args.startup_timeout_secs,
        &mut stage1,
    )
    .context("stage 1 binary server did not become ready")?;
    let request_id = 1;
    let session_id = 1;
    send_generation_config(&mut stream, request_id, session_id, 1)
        .context("send binary generation config")?;
    let message = binary_decode_message(BinaryDecodeMessageArgs {
        token_id,
        decode_step: 0,
        source_stage_index: 0,
        boundary: &boundary,
        activation_width,
        request_id,
        session_id,
    })?;
    write_stage_message(&mut stream, &message).context("send binary decode")?;
    let reply = recv_reply(&mut stream).context("receive binary prediction reply")?;
    ensure_reply_kind(&reply, WireReplyKind::PredictedToken)?;
    let native_mtp = native_mtp_sideband_report(&reply);
    let (second_predicted_token, native_mtp_verification_compute_us) = if args
        .native_mtp_verification
    {
        let verification_timer = Instant::now();
        let (_boundary_prediction, second_boundary) = session0
            .decode_step_frame(reply.predicted, None, 0)
            .context("stage 0 failed to produce second activation frame")?;
        let second_message = binary_decode_message(BinaryDecodeMessageArgs {
            token_id: reply.predicted,
            decode_step: 1,
            source_stage_index: 0,
            boundary: &second_boundary,
            activation_width,
            request_id,
            session_id,
        })?;
        write_stage_message(&mut stream, &second_message).context("send second binary decode")?;
        let second_reply =
            recv_reply(&mut stream).context("receive second binary prediction reply")?;
        ensure_reply_kind(&second_reply, WireReplyKind::PredictedToken)?;
        (
            Some(second_reply.predicted),
            Some(elapsed_us(verification_timer)),
        )
    } else {
        (None, None)
    };
    write_stage_message(&mut stream, &StageWireMessage::stop()).context("send binary stop")?;

    Ok(BinarySplitResult {
        token_id,
        predicted_token: reply.predicted,
        second_predicted_token,
        native_mtp,
        native_mtp_verification_compute_us,
        activation_width,
        boundary_producer_stage_index: boundary.desc.producer_stage_index,
        boundary_layer_start: boundary.desc.layer_start,
        boundary_layer_end: boundary.desc.layer_end,
        boundary_token_count: boundary.desc.token_count,
        boundary_payload_bytes: boundary.desc.payload_bytes,
        boundary_wire_payload_bytes: message.activation.len(),
        stage_models: vec![stage0_resolution.report, stage1_resolution.report],
    })
}
