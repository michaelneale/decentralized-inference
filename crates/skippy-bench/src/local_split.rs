use std::{
    fs,
    process::{Command, Stdio},
};

use anyhow::{Context, Result, bail};
use model_artifact::ModelIdentity;
use serde_json::json;
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind, recv_reply,
    write_stage_message,
};
use skippy_runtime::{MtpSource, RuntimeConfig, RuntimeLoadMode, StageModel};
use skippy_topology::{
    BoundaryDecision, NodeSpec, PlannerPolicy, TopologyPlanRequest, dense_attention_layers,
    infer_family_capability, plan_contiguous_with_splits,
};

use crate::{
    cli::{
        LocalSplitBinaryArgs, LocalSplitChainBinaryArgs, LocalSplitCompareArgs,
        LocalSplitInprocessArgs,
    },
    model_identity::model_identity_for_path,
    support::{
        ChildGuard, activation_width, connect_ready, ensure_release_skippy_server_bin,
        generate_run_id, temp_config_path_for,
    },
};

struct BinarySplitResult {
    model_identity: ModelIdentity,
    token_id: i32,
    predicted_token: i32,
    activation_width: i32,
    boundary_producer_stage_index: i32,
    boundary_layer_start: i32,
    boundary_layer_end: i32,
    boundary_token_count: u32,
    boundary_payload_bytes: u64,
    boundary_wire_payload_bytes: usize,
}

struct BinaryChainResult {
    model_identity: ModelIdentity,
    token_id: i32,
    predicted_token: i32,
    activation_width: i32,
    stage0_wire_payload_bytes: usize,
    stage0_payload_bytes: u64,
    split_layer_1: u32,
    split_layer_2: u32,
    layer_end: u32,
}

fn ensure_reply_kind(
    reply: &skippy_protocol::binary::StageReply,
    expected: WireReplyKind,
) -> Result<()> {
    if reply.kind != expected {
        bail!("expected {expected:?} reply, got {:?}", reply.kind);
    }
    Ok(())
}

pub fn local_split_binary(args: LocalSplitBinaryArgs) -> Result<()> {
    let result = run_binary_split(BinarySplitConfig {
        stage_server_bin: args.stage_server_bin,
        model_path: args.model_path,
        model_id: args.model_id,
        split_layer: args.split_layer,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        prompt: args.prompt,
        stage1_bind_addr: args.stage1_bind_addr,
        child_logs: args.child_logs,
        startup_timeout_secs: args.startup_timeout_secs,
    })?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "local-split-binary",
            "model_identity": result.model_identity,
            "token_id": result.token_id,
            "predicted_token": result.predicted_token,
            "activation_width": result.activation_width,
            "boundary": {
                "producer_stage_index": result.boundary_producer_stage_index,
                "layer_start": result.boundary_layer_start,
                "layer_end": result.boundary_layer_end,
                "token_count": result.boundary_token_count,
                "payload_bytes": result.boundary_payload_bytes,
                "wire_payload_bytes": result.boundary_wire_payload_bytes,
            }
        }))?
    );

    Ok(())
}

pub fn local_split_compare(args: LocalSplitCompareArgs) -> Result<()> {
    let baseline = run_full_model_decode(
        &args.model_path,
        args.layer_end,
        args.ctx_size,
        args.n_gpu_layers,
        &args.prompt,
    )?;
    let split = run_binary_split(BinarySplitConfig {
        stage_server_bin: args.stage_server_bin,
        model_path: args.model_path,
        model_id: args.model_id,
        split_layer: args.split_layer,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        prompt: args.prompt,
        stage1_bind_addr: args.stage1_bind_addr,
        child_logs: args.child_logs,
        startup_timeout_secs: args.startup_timeout_secs,
    })?;

    let matches = baseline.predicted_token == split.predicted_token;
    let output = json!({
        "mode": "local-split-compare",
        "model_identity": split.model_identity,
        "matches": matches,
        "baseline": {
            "token_id": baseline.token_id,
            "predicted_token": baseline.predicted_token,
        },
        "split": {
            "token_id": split.token_id,
            "predicted_token": split.predicted_token,
            "activation_width": split.activation_width,
            "boundary": {
                "producer_stage_index": split.boundary_producer_stage_index,
                "layer_start": split.boundary_layer_start,
                "layer_end": split.boundary_layer_end,
                "token_count": split.boundary_token_count,
                "payload_bytes": split.boundary_payload_bytes,
                "wire_payload_bytes": split.boundary_wire_payload_bytes,
            }
        }
    });
    println!("{}", serde_json::to_string_pretty(&output)?);

    if !matches && !args.allow_mismatch {
        bail!(
            "split predicted token {} did not match full-model predicted token {}",
            split.predicted_token,
            baseline.predicted_token
        );
    }

    Ok(())
}

pub fn local_split_chain_binary(args: LocalSplitChainBinaryArgs) -> Result<()> {
    let result = run_binary_chain(args)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "local-split-chain-binary",
            "model_identity": result.model_identity,
            "token_id": result.token_id,
            "predicted_token": result.predicted_token,
            "activation_width": result.activation_width,
            "stages": [
                {
                    "stage_index": 0,
                    "layer_start": 0,
                    "layer_end": result.split_layer_1,
                    "payload_bytes": result.stage0_payload_bytes,
                    "wire_payload_bytes": result.stage0_wire_payload_bytes,
                },
                {
                    "stage_index": 1,
                    "layer_start": result.split_layer_1,
                    "layer_end": result.split_layer_2,
                    "forwarded_over_binary": true,
                },
                {
                    "stage_index": 2,
                    "layer_start": result.split_layer_2,
                    "layer_end": result.layer_end,
                    "returned_predicted_token": true,
                }
            ]
        }))?
    );
    Ok(())
}

struct FullModelResult {
    token_id: i32,
    predicted_token: i32,
}

fn run_full_model_decode(
    model_path: &std::path::Path,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    prompt: &str,
) -> Result<FullModelResult> {
    let config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end,
        ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers,
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
        cache_type_k: skippy_runtime::GGML_TYPE_F16,
        cache_type_v: skippy_runtime::GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
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
        checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };
    let model = StageModel::open(model_path, &config).context("failed to open full model")?;
    let tokens = model
        .tokenize(prompt, true)
        .context("failed to tokenize prompt with full model")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session = model
        .create_session()
        .context("failed to create full-model session")?;
    let predicted_token = session
        .decode_step_frame(token_id, None, 0)
        .context("full model failed to decode")?
        .0;
    Ok(FullModelResult {
        token_id,
        predicted_token,
    })
}

struct BinarySplitConfig {
    stage_server_bin: std::path::PathBuf,
    model_path: std::path::PathBuf,
    model_id: String,
    split_layer: u32,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    prompt: String,
    stage1_bind_addr: std::net::SocketAddr,
    child_logs: bool,
    startup_timeout_secs: u64,
}

fn run_binary_split(args: BinarySplitConfig) -> Result<BinarySplitResult> {
    if args.split_layer == 0 || args.split_layer >= args.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    ensure_release_skippy_server_bin(&args.stage_server_bin)?;
    validate_local_topology_plan(&args.model_path, args.layer_end, &[args.split_layer], 2)?;
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
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
        cache_type_k: skippy_runtime::GGML_TYPE_F16,
        cache_type_v: skippy_runtime::GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
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
        checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };
    let stage0 =
        StageModel::open(&args.model_path, &stage0_config).context("failed to open stage 0")?;
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
    let config_path = temp_config_path_for(&run_id, "stage-1");
    let topology_path = temp_config_path_for(&run_id, "topology");
    let config = json!({
        "run_id": run_id,
        "topology_id": "local-split-binary",
        "model_id": model_identity.model_id,
        "model_path": args.model_path,
        "stage_id": "stage-1",
        "stage_index": 1,
        "layer_start": args.split_layer,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": args.stage1_bind_addr,
        "upstream": {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": null
    });
    let topology = local_split_topology(
        "local-split-binary",
        &model_identity.model_id,
        &[
            LocalSplitTopologyStage {
                stage_id: "stage-0",
                stage_index: 0,
                endpoint: "driver".to_string(),
                layer_start: 0,
                layer_end: args.split_layer,
            },
            LocalSplitTopologyStage {
                stage_id: "stage-1",
                stage_index: 1,
                endpoint: format!("tcp://{}", args.stage1_bind_addr),
                layer_start: args.split_layer,
                layer_end: args.layer_end,
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
            .context("stage topology path is not valid UTF-8")?,
    ]);
    if args.child_logs {
        stage_command
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
    } else {
        stage_command.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let _stage1 = ChildGuard::spawn(stage_command)?;

    let mut stream = connect_ready(args.stage1_bind_addr, args.startup_timeout_secs)
        .context("stage 1 binary server did not become ready")?;
    let request_id = 1;
    let session_id = 1;
    send_generation_config(&mut stream, request_id, session_id, 1)
        .context("send binary split generation config")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
    state.prompt_token_count = 0;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = 0;
    state.flags |=
        skippy_protocol::binary::activation_state_flags_from_frame_flags(boundary.desc.flags);
    let activation = skippy_protocol::binary::encode_activation_payload_with_state_flags(
        state.activation_codec,
        1,
        activation_width,
        &boundary.payload,
        state.flags,
    )
    .context("failed to encode boundary activation for wire")?;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 0,
        token_count: 1,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        positions: vec![0],
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut stream, &message).context("send binary decode")?;
    let reply = recv_reply(&mut stream).context("receive binary split prediction reply")?;
    ensure_reply_kind(&reply, WireReplyKind::PredictedToken)?;
    write_stage_message(&mut stream, &StageWireMessage::stop()).context("send binary stop")?;

    Ok(BinarySplitResult {
        model_identity,
        token_id,
        predicted_token: reply.predicted,
        activation_width,
        boundary_producer_stage_index: boundary.desc.producer_stage_index,
        boundary_layer_start: boundary.desc.layer_start,
        boundary_layer_end: boundary.desc.layer_end,
        boundary_token_count: boundary.desc.token_count,
        boundary_payload_bytes: boundary.desc.payload_bytes,
        boundary_wire_payload_bytes: message.activation.len(),
    })
}

fn run_binary_chain(args: LocalSplitChainBinaryArgs) -> Result<BinaryChainResult> {
    ensure_release_skippy_server_bin(&args.stage_server_bin)?;
    if args.split_layer_1 == 0
        || args.split_layer_1 >= args.split_layer_2
        || args.split_layer_2 >= args.layer_end
    {
        bail!("split_layer_1 and split_layer_2 must partition 0..layer_end in ascending order");
    }
    validate_local_topology_plan(
        &args.model_path,
        args.layer_end,
        &[args.split_layer_1, args.split_layer_2],
        3,
    )?;
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer_1,
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
        cache_type_k: skippy_runtime::GGML_TYPE_F16,
        cache_type_v: skippy_runtime::GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
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
        checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };
    let stage0 =
        StageModel::open(&args.model_path, &stage0_config).context("failed to open stage 0")?;
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
    let stage1_config_path = temp_config_path_for(&run_id, "stage-1");
    let stage2_config_path = temp_config_path_for(&run_id, "stage-2");
    let topology_path = temp_config_path_for(&run_id, "topology");
    let stage2_config = json!({
        "run_id": run_id,
        "topology_id": "local-split-chain-binary",
        "model_id": model_identity.model_id,
        "model_path": args.model_path,
        "stage_id": "stage-2",
        "stage_index": 2,
        "layer_start": args.split_layer_2,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
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
        "topology_id": "local-split-chain-binary",
        "model_id": model_identity.model_id,
        "model_path": args.model_path,
        "stage_id": "stage-1",
        "stage_index": 1,
        "layer_start": args.split_layer_1,
        "layer_end": args.split_layer_2,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": args.stage1_bind_addr,
        "upstream": {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": {
            "stage_id": "stage-2",
            "stage_index": 2,
            "endpoint": format!("tcp://{}", args.stage2_bind_addr)
        }
    });
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
    let topology = local_split_topology(
        "local-split-chain-binary",
        &model_identity.model_id,
        &[
            LocalSplitTopologyStage {
                stage_id: "stage-0",
                stage_index: 0,
                endpoint: "driver".to_string(),
                layer_start: 0,
                layer_end: args.split_layer_1,
            },
            LocalSplitTopologyStage {
                stage_id: "stage-1",
                stage_index: 1,
                endpoint: format!("tcp://{}", args.stage1_bind_addr),
                layer_start: args.split_layer_1,
                layer_end: args.split_layer_2,
            },
            LocalSplitTopologyStage {
                stage_id: "stage-2",
                stage_index: 2,
                endpoint: format!("tcp://{}", args.stage2_bind_addr),
                layer_start: args.split_layer_2,
                layer_end: args.layer_end,
            },
        ],
    );
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
            .context("stage topology path is not valid UTF-8")?,
    ]);
    configure_child_logs(&mut stage2_command, args.child_logs);
    let _stage2 = ChildGuard::spawn(stage2_command)?;

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
            .context("stage topology path is not valid UTF-8")?,
    ]);
    configure_child_logs(&mut stage1_command, args.child_logs);
    let _stage1 = ChildGuard::spawn(stage1_command)?;

    let mut stream = connect_ready(args.stage1_bind_addr, args.startup_timeout_secs)
        .context("stage 1 binary server did not become ready")?;
    let request_id = 2;
    let session_id = 2;
    send_generation_config(&mut stream, request_id, session_id, 1)
        .context("send binary chain generation config")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
    state.prompt_token_count = 0;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = 0;
    state.flags |=
        skippy_protocol::binary::activation_state_flags_from_frame_flags(boundary.desc.flags);
    let activation = skippy_protocol::binary::encode_activation_payload_with_state_flags(
        state.activation_codec,
        1,
        activation_width,
        &boundary.payload,
        state.flags,
    )
    .context("failed to encode boundary activation for wire")?;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 0,
        token_count: 1,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        positions: vec![0],
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut stream, &message).context("send binary chain decode")?;
    let reply = recv_reply(&mut stream).context("receive binary chain prediction reply")?;
    ensure_reply_kind(&reply, WireReplyKind::PredictedToken)?;
    write_stage_message(&mut stream, &StageWireMessage::stop())
        .context("send binary chain stop")?;

    Ok(BinaryChainResult {
        model_identity,
        token_id,
        predicted_token: reply.predicted,
        activation_width,
        stage0_wire_payload_bytes: message.activation.len(),
        stage0_payload_bytes: boundary.desc.payload_bytes,
        split_layer_1: args.split_layer_1,
        split_layer_2: args.split_layer_2,
        layer_end: args.layer_end,
    })
}

struct LocalSplitTopologyStage<'a> {
    stage_id: &'a str,
    stage_index: u32,
    endpoint: String,
    layer_start: u32,
    layer_end: u32,
}

fn local_split_topology(
    topology_id: &str,
    model_id: &str,
    stages: &[LocalSplitTopologyStage<'_>],
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
                "load_mode": "runtime-slice",
            })
        }).collect::<Vec<_>>(),
    })
}

fn send_generation_config(
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

fn validate_local_topology_plan(
    model_path: &std::path::Path,
    layer_end: u32,
    splits: &[u32],
    stage_count: usize,
) -> Result<()> {
    let identity = model_path.display().to_string();
    let family = infer_family_capability(&identity, layer_end, 0);
    let request = TopologyPlanRequest {
        topology_id: "local-split-binary".to_string(),
        model_id: identity,
        layers: dense_attention_layers(layer_end, 0),
        nodes: (0..stage_count)
            .map(|index| NodeSpec {
                node_id: format!("local-stage-{index}"),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect(),
        family: family.clone(),
        policy: PlannerPolicy::default(),
    };
    let plan = plan_contiguous_with_splits(&request, splits).context("topology planner failed")?;

    let rejected = plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.decision == BoundaryDecision::Rejected)
        .collect::<Vec<_>>();
    if !rejected.is_empty() {
        let reasons = rejected
            .iter()
            .map(|boundary| {
                format!(
                    "layer {}: {:?}: {}",
                    boundary.layer_boundary,
                    boundary.reason_codes,
                    boundary.messages.join("; ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        bail!("topology planner rejected split plan:\n{reasons}");
    }

    Ok(())
}

fn configure_child_logs(command: &mut Command, child_logs: bool) {
    if child_logs {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
}

pub fn local_split_inprocess(args: LocalSplitInprocessArgs) -> Result<()> {
    if args.split_layer == 0 || args.split_layer >= args.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }

    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
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
        cache_type_k: skippy_runtime::GGML_TYPE_F16,
        cache_type_v: skippy_runtime::GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
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
        checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };
    let stage1_config = RuntimeConfig {
        stage_index: 1,
        layer_start: args.split_layer,
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
        cache_type_k: skippy_runtime::GGML_TYPE_F16,
        cache_type_v: skippy_runtime::GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_runtime::GlmDsaPolicy::Auto,
        include_embeddings: false,
        include_output: true,
        mtp_source: MtpSource::Disabled,
        filter_tensors_on_load: true,
        checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };

    let stage0 =
        StageModel::open(&args.model_path, &stage0_config).context("failed to open stage 0")?;
    let stage1 =
        StageModel::open(&args.model_path, &stage1_config).context("failed to open stage 1")?;
    let tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;

    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let mut session1 = stage1
        .create_session()
        .context("failed to create stage 1 session")?;

    let (_boundary_prediction, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty activation frame");
    }

    let (predicted_token, final_frame) = session1
        .decode_step_frame(token_id, Some(&boundary), 0)
        .context("stage 1 failed to consume activation frame")?;
    if !final_frame.payload.is_empty() {
        bail!("final stage unexpectedly produced an activation payload");
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "local-split-inprocess",
            "token_id": token_id,
            "predicted_token": predicted_token,
            "boundary": {
                "producer_stage_index": boundary.desc.producer_stage_index,
                "layer_start": boundary.desc.layer_start,
                "layer_end": boundary.desc.layer_end,
                "token_count": boundary.desc.token_count,
                "sequence_count": boundary.desc.sequence_count,
                "payload_bytes": boundary.desc.payload_bytes,
                "actual_payload_bytes": boundary.payload.len(),
            },
            "final": {
                "producer_stage_index": final_frame.desc.producer_stage_index,
                "layer_start": final_frame.desc.layer_start,
                "layer_end": final_frame.desc.layer_end,
                "payload_bytes": final_frame.desc.payload_bytes,
            }
        }))?
    );

    Ok(())
}
