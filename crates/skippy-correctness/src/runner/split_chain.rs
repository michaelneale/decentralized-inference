use std::{
    fs,
    net::SocketAddr,
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use model_artifact::ModelIdentity;
use serde_json::json;
use skippy_protocol::binary::{StageWireMessage, WireReplyKind, write_stage_message};
use skippy_runtime::{GGML_TYPE_F16, MtpSource, RuntimeConfig, StageModel};

use crate::{
    cli::{ChainArgs, FlashAttentionArg, SplitScanArgs, StageLoadMode},
    report::{
        ChainReport, ChainStageReport, NativeMtpSidebandReport, SplitScanReport, StageModelReport,
    },
    support::{
        ChildGuard, activation_width, connect_ready_child, generate_run_id, temp_config_path_for,
    },
};

use super::{
    native_mtp::{
        emit_report, ensure_native_mtp_artifact_if_required, native_mtp_requirement,
        native_mtp_satisfies_requirement, native_mtp_sideband_report,
        native_mtp_verification_report, native_mtp_verification_satisfies_requirement,
    },
    prediction_return::PredictionReturnListener,
    single_step::{SingleStepCase, run_full_model_decode, run_single_step_with_baseline},
    stage_execution::{
        BinaryDecodeMessageArgs, CorrectnessTopologyStage, FullModelResult, PackageStageSpec,
        baseline_report, binary_decode_message, configure_child_logs, correctness_topology,
        elapsed_us, ensure_matches, ensure_reply_kind, parse_chain_splits, parse_split_list,
        protocol_flash_attn, protocol_load_mode, runtime_flash_attn, runtime_load_mode,
        runtime_model_identity, send_generation_config, stage_model_resolution,
        stage_server_model_path, status,
    },
};

struct BinaryChainConfig {
    pub(in crate::runner) stage_server_bin: PathBuf,
    pub(in crate::runner) model: PathBuf,
    pub(in crate::runner) stage_model: Option<PathBuf>,
    pub(in crate::runner) stage_load_mode: StageLoadMode,
    pub(in crate::runner) split_layer_1: u32,
    pub(in crate::runner) split_layer_2: u32,
    pub(in crate::runner) layer_end: u32,
    pub(in crate::runner) ctx_size: u32,
    pub(in crate::runner) n_batch: Option<u32>,
    pub(in crate::runner) n_ubatch: Option<u32>,
    pub(in crate::runner) n_gpu_layers: i32,
    pub(in crate::runner) flash_attn: FlashAttentionArg,
    pub(in crate::runner) prompt: String,
    pub(in crate::runner) stage1_bind_addr: SocketAddr,
    pub(in crate::runner) stage2_bind_addr: SocketAddr,
    pub(in crate::runner) child_logs: bool,
    pub(in crate::runner) startup_timeout_secs: u64,
    pub(in crate::runner) max_inflight: usize,
    pub(in crate::runner) model_identity: ModelIdentity,
    pub(in crate::runner) native_mtp_verification: bool,
}

struct BinaryChainResult {
    pub(in crate::runner) token_id: i32,
    pub(in crate::runner) predicted_token: i32,
    pub(in crate::runner) second_predicted_token: Option<i32>,
    pub(in crate::runner) native_mtp: NativeMtpSidebandReport,
    pub(in crate::runner) native_mtp_verification_compute_us: Option<i64>,
    pub(in crate::runner) activation_width: i32,
    pub(in crate::runner) stage0_wire_payload_bytes: usize,
    pub(in crate::runner) stage0_payload_bytes: u64,
    pub(in crate::runner) split_layer_1: u32,
    pub(in crate::runner) split_layer_2: u32,
    pub(in crate::runner) layer_end: u32,
    pub(in crate::runner) stage_models: Vec<StageModelReport>,
}
pub fn chain(args: ChainArgs) -> Result<()> {
    let native_mtp_requirement = native_mtp_requirement(args.native_mtp);
    ensure_native_mtp_artifact_if_required(&args.runtime, native_mtp_requirement)?;
    let splits = parse_chain_splits(&args.splits)?;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let baseline = run_full_model_decode(&args.runtime)?;
    let chain = run_binary_chain(BinaryChainConfig {
        stage_server_bin: args.server.stage_server_bin,
        model: args.runtime.model,
        stage_model: args.runtime.stage_model,
        stage_load_mode: args.runtime.stage_load_mode,
        split_layer_1: splits.0,
        split_layer_2: splits.1,
        layer_end: args.runtime.layer_end,
        ctx_size: args.runtime.ctx_size,
        n_batch: args.runtime.n_batch,
        n_ubatch: args.runtime.n_ubatch,
        n_gpu_layers: args.runtime.n_gpu_layers,
        flash_attn: args.runtime.flash_attn,
        prompt: args.runtime.prompt,
        stage1_bind_addr: args.stage1_bind_addr,
        stage2_bind_addr: args.stage2_bind_addr,
        child_logs: args.server.child_logs,
        startup_timeout_secs: args.server.startup_timeout_secs,
        max_inflight: args.server.max_inflight,
        model_identity: model_identity.clone(),
        native_mtp_verification: native_mtp_requirement.require_draft,
    })?;
    let native_mtp = chain.native_mtp.clone();
    let native_mtp_verification = native_mtp_verification_report(
        native_mtp_requirement.require_draft,
        &native_mtp,
        chain.second_predicted_token,
        baseline.second_predicted_token,
        chain.native_mtp_verification_compute_us,
    );
    let matches = baseline.predicted_token == chain.predicted_token
        && native_mtp_satisfies_requirement(&native_mtp, native_mtp_requirement)
        && native_mtp_verification_satisfies_requirement(
            &native_mtp_verification,
            native_mtp_requirement,
        );
    let report = ChainReport {
        mode: "chain",
        status: status(matches),
        model_identity,
        matches,
        native_mtp_draft_required: native_mtp_requirement.require_draft,
        baseline: baseline_report(baseline),
        token_id: chain.token_id,
        predicted_token: chain.predicted_token,
        second_predicted_token: chain.second_predicted_token,
        native_mtp,
        native_mtp_verification,
        activation_width: chain.activation_width,
        stages: vec![
            ChainStageReport {
                stage_index: 0,
                layer_start: 0,
                layer_end: chain.split_layer_1,
                payload_bytes: Some(chain.stage0_payload_bytes),
                wire_payload_bytes: Some(chain.stage0_wire_payload_bytes),
                forwarded_over_binary: false,
                returned_predicted_token: false,
            },
            ChainStageReport {
                stage_index: 1,
                layer_start: chain.split_layer_1,
                layer_end: chain.split_layer_2,
                payload_bytes: None,
                wire_payload_bytes: None,
                forwarded_over_binary: true,
                returned_predicted_token: false,
            },
            ChainStageReport {
                stage_index: 2,
                layer_start: chain.split_layer_2,
                layer_end: chain.layer_end,
                payload_bytes: None,
                wire_payload_bytes: None,
                forwarded_over_binary: false,
                returned_predicted_token: true,
            },
        ],
        stage_models: chain.stage_models,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    ensure_matches(report.matches, args.allow_mismatch)?;
    Ok(())
}

pub fn split_scan(args: SplitScanArgs) -> Result<()> {
    let native_mtp = native_mtp_requirement(args.native_mtp);
    ensure_native_mtp_artifact_if_required(&args.runtime, native_mtp)?;
    let splits = parse_split_list(&args.splits)?;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let baseline = run_full_model_decode(&args.runtime)?;
    let mut results = Vec::with_capacity(splits.len());
    for split_layer in splits {
        if split_layer == 0 || split_layer >= args.runtime.layer_end {
            bail!(
                "split layer {split_layer} must be greater than zero and less than layer_end {}",
                args.runtime.layer_end
            );
        }
        results.push(run_single_step_with_baseline(
            &args.runtime,
            &args.server,
            &model_identity,
            FullModelResult {
                token_id: baseline.token_id,
                predicted_token: baseline.predicted_token,
                second_predicted_token: baseline.second_predicted_token,
            },
            SingleStepCase {
                split_layer,
                stage1_bind_addr: args.stage1_bind_addr,
                native_mtp,
            },
        )?);
    }
    let mismatch_count = results.iter().filter(|result| !result.matches).count();
    let report = SplitScanReport {
        mode: "split-scan",
        status: status(mismatch_count == 0),
        model_identity,
        baseline: baseline_report(baseline),
        split_count: results.len(),
        mismatch_count,
        results,
    };
    emit_report(&report, args.output.report_out.as_deref())?;
    ensure_matches(mismatch_count == 0, args.allow_mismatch)?;
    Ok(())
}

fn run_binary_chain(args: BinaryChainConfig) -> Result<BinaryChainResult> {
    if args.split_layer_1 == 0
        || args.split_layer_1 >= args.split_layer_2
        || args.split_layer_2 >= args.layer_end
    {
        bail!("splits must partition 0..layer_end in ascending order");
    }
    let stage0_spec = PackageStageSpec {
        topology_id: "correctness-chain",
        stage_id: "stage-0",
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer_1,
        include_embeddings: true,
        include_output: false,
    };
    let stage1_spec = PackageStageSpec {
        topology_id: "correctness-chain",
        stage_id: "stage-1",
        stage_index: 1,
        layer_start: args.split_layer_1,
        layer_end: args.split_layer_2,
        include_embeddings: false,
        include_output: false,
    };
    let stage2_spec = PackageStageSpec {
        topology_id: "correctness-chain",
        stage_id: "stage-2",
        stage_index: 2,
        layer_start: args.split_layer_2,
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
    let stage2_resolution = stage_model_resolution(
        &args.model,
        args.stage_model.as_ref(),
        args.stage_load_mode,
        &args.model_identity,
        stage2_spec,
    )?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer_1,
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
    let prediction_return = PredictionReturnListener::start()?;
    let stage0_endpoint = prediction_return.endpoint();

    let run_id = generate_run_id();
    let model_id = args.model_identity.model_id.clone();
    let stage1_config_path = temp_config_path_for(&run_id, "stage-1");
    let stage2_config_path = temp_config_path_for(&run_id, "stage-2");
    let topology_path = temp_config_path_for(&run_id, "topology");
    let stage2_config = json!({
        "run_id": run_id,
        "topology_id": "correctness-chain",
        "model_id": model_id,
        "model_path": stage_server_model_path(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            stage2_spec,
        )?,
        "stage_id": "stage-2",
        "stage_index": 2,
        "layer_start": args.split_layer_2,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_batch": args.n_batch,
        "n_ubatch": args.n_ubatch,
        "n_gpu_layers": args.n_gpu_layers,
        "flash_attn_type": protocol_flash_attn(args.flash_attn),
        "filter_tensors_on_load": true,
        "load_mode": protocol_load_mode(args.stage_load_mode),
        "bind_addr": args.stage2_bind_addr,
        "upstream": {
            "stage_id": "stage-1",
            "stage_index": 1,
            "endpoint": format!("tcp://{}", args.stage1_bind_addr)
        },
        "downstream": null
    });
    let stage1_config = json!({
        "run_id": run_id,
        "topology_id": "correctness-chain",
        "model_id": model_id,
        "model_path": stage_server_model_path(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            stage1_spec,
        )?,
        "stage_id": "stage-1",
        "stage_index": 1,
        "layer_start": args.split_layer_1,
        "layer_end": args.split_layer_2,
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
            "endpoint": stage0_endpoint
        },
        "downstream": {
            "stage_id": "stage-2",
            "stage_index": 2,
            "endpoint": format!("tcp://{}", args.stage2_bind_addr)
        }
    });
    let topology = correctness_topology(
        "correctness-chain",
        &model_id,
        &[
            CorrectnessTopologyStage {
                stage_id: "stage-0",
                stage_index: 0,
                endpoint: stage0_endpoint,
                layer_start: 0,
                layer_end: args.split_layer_1,
                load_mode: protocol_load_mode(args.stage_load_mode),
            },
            CorrectnessTopologyStage {
                stage_id: "stage-1",
                stage_index: 1,
                endpoint: format!("tcp://{}", args.stage1_bind_addr),
                layer_start: args.split_layer_1,
                layer_end: args.split_layer_2,
                load_mode: protocol_load_mode(args.stage_load_mode),
            },
            CorrectnessTopologyStage {
                stage_id: "stage-2",
                stage_index: 2,
                endpoint: format!("tcp://{}", args.stage2_bind_addr),
                layer_start: args.split_layer_2,
                layer_end: args.layer_end,
                load_mode: protocol_load_mode(args.stage_load_mode),
            },
        ],
    );
    fs::write(
        &stage2_config_path,
        serde_json::to_vec_pretty(&stage2_config)?,
    )
    .with_context(|| format!("failed to write {}", stage2_config_path.display()))?;
    fs::write(
        &stage1_config_path,
        serde_json::to_vec_pretty(&stage1_config)?,
    )
    .with_context(|| format!("failed to write {}", stage1_config_path.display()))?;
    fs::write(&topology_path, serde_json::to_vec_pretty(&topology)?)
        .with_context(|| format!("failed to write {}", topology_path.display()))?;

    let mut stage2_command = Command::new(&args.stage_server_bin);
    stage2_command.args([
        "serve-binary",
        "--config",
        stage2_config_path
            .to_str()
            .context("stage 2 config path is not valid UTF-8")?,
        "--topology",
        topology_path
            .to_str()
            .context("topology path is not valid UTF-8")?,
        "--max-inflight",
        &args.max_inflight.to_string(),
    ]);
    configure_child_logs(&mut stage2_command, args.child_logs);
    let mut stage2 = ChildGuard::spawn(stage2_command)?;
    drop(
        connect_ready_child(
            args.stage2_bind_addr,
            args.startup_timeout_secs,
            &mut stage2,
        )
        .context("stage 2 binary server did not become ready")?,
    );

    let mut stage1_command = Command::new(&args.stage_server_bin);
    stage1_command.args([
        "serve-binary",
        "--config",
        stage1_config_path
            .to_str()
            .context("stage 1 config path is not valid UTF-8")?,
        "--topology",
        topology_path
            .to_str()
            .context("topology path is not valid UTF-8")?,
        "--max-inflight",
        &args.max_inflight.to_string(),
    ]);
    configure_child_logs(&mut stage1_command, args.child_logs);
    let mut stage1 = ChildGuard::spawn(stage1_command)?;

    let mut stream = connect_ready_child(
        args.stage1_bind_addr,
        args.startup_timeout_secs,
        &mut stage1,
    )
    .context("stage 1 binary server did not become ready")?;
    let request_id = 2;
    let session_id = 2;
    send_generation_config(&mut stream, request_id, session_id, 1)
        .context("send binary chain generation config")?;
    let message = binary_decode_message(BinaryDecodeMessageArgs {
        token_id,
        decode_step: 0,
        source_stage_index: 0,
        boundary: &boundary,
        activation_width,
        request_id,
        session_id,
    })?;
    write_stage_message(&mut stream, &message).context("send binary chain decode")?;
    let reply = prediction_return
        .receive(Duration::from_secs(args.startup_timeout_secs))
        .context("receive binary chain direct prediction reply")?;
    ensure_reply_kind(&reply, WireReplyKind::PredictedToken)?;
    let native_mtp = native_mtp_sideband_report(&reply);
    let (second_predicted_token, native_mtp_verification_compute_us) =
        if args.native_mtp_verification {
            let verification_timer = Instant::now();
            let (_boundary_prediction, second_boundary) = session0
                .decode_step_frame(reply.predicted, None, 0)
                .context("stage 0 failed to produce second chain activation frame")?;
            let second_message = binary_decode_message(BinaryDecodeMessageArgs {
                token_id: reply.predicted,
                decode_step: 1,
                source_stage_index: 0,
                boundary: &second_boundary,
                activation_width,
                request_id,
                session_id,
            })?;
            write_stage_message(&mut stream, &second_message)
                .context("send second binary chain decode")?;
            let second_reply = prediction_return
                .receive(Duration::from_secs(args.startup_timeout_secs))
                .context("receive second binary chain direct prediction reply")?;
            ensure_reply_kind(&second_reply, WireReplyKind::PredictedToken)?;
            (
                Some(second_reply.predicted),
                Some(elapsed_us(verification_timer)),
            )
        } else {
            (None, None)
        };
    write_stage_message(&mut stream, &StageWireMessage::stop())
        .context("send binary chain stop")?;

    Ok(BinaryChainResult {
        token_id,
        predicted_token: reply.predicted,
        second_predicted_token,
        native_mtp,
        native_mtp_verification_compute_us,
        activation_width,
        stage0_wire_payload_bytes: message.activation.len(),
        stage0_payload_bytes: boundary.desc.payload_bytes,
        split_layer_1: args.split_layer_1,
        split_layer_2: args.split_layer_2,
        layer_end: args.layer_end,
        stage_models: vec![
            stage0_resolution.report,
            stage1_resolution.report,
            stage2_resolution.report,
        ],
    })
}
