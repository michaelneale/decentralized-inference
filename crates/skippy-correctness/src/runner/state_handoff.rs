use std::{collections::HashSet, fs, process::Command, time::Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;
use sha2::{Digest, Sha256};
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind,
    activation_state_flags_from_frame_flags, read_stage_message, recv_reply, state_flags,
    write_stage_message,
};
use skippy_runtime::{
    ActivationFrame, GGML_TYPE_F16, MtpSource, RuntimeConfig, RuntimeKvPageDesc, StageModel,
    StageSession,
};

use crate::{
    cli::{StageLoadMode, StateHandoffArgs, StatePayloadKind},
    report::{
        StageModelReport, StateHandoffReport, StatePayloadBlockDigestReport,
        StatePayloadDigestReport,
    },
    support::{
        ChildGuard, activation_width, connect_ready_child, generate_run_id, temp_config_path_for,
    },
};

use super::native_mtp::emit_report;
use super::stage_execution::{
    BinaryStateHandoffConfig, PackageStageSpec, StageModelResolution, configure_child_logs,
    elapsed_ms, ensure_matches, mean_pair_sum, protocol_flash_attn, protocol_load_mode,
    runtime_flash_attn, runtime_load_mode, runtime_model_identity, speedup, stage_id_for_index,
    stage_model_resolution, stage_server_model_path, status, tokenizer_model_for_state_handoff,
};

struct BinaryStateHandoffResult {
    pub(in crate::runner) prompt_token_count: usize,
    pub(in crate::runner) benchmark_prompt_token_count: usize,
    pub(in crate::runner) benchmark_prompt_text: String,
    pub(in crate::runner) requested_prefix_token_count: Option<usize>,
    pub(in crate::runner) stage_index: u32,
    pub(in crate::runner) layer_start: u32,
    pub(in crate::runner) layer_end: u32,
    pub(in crate::runner) include_embeddings: bool,
    pub(in crate::runner) include_output: bool,
    pub(in crate::runner) handoff_transport: &'static str,
    pub(in crate::runner) state_payload_kind: StatePayloadKind,
    pub(in crate::runner) borrowed_resident_hits: bool,
    pub(in crate::runner) cached_decoded_result_hits: bool,
    pub(in crate::runner) activation_width: i32,
    pub(in crate::runner) source_predicted_token: i32,
    pub(in crate::runner) restored_predicted_token: i32,
    pub(in crate::runner) state_bytes: usize,
    pub(in crate::runner) cache_storage_bytes: Option<usize>,
    pub(in crate::runner) resident_state_bytes: Option<usize>,
    pub(in crate::runner) roundtrip_state_bytes: usize,
    pub(in crate::runner) payload_digest: StatePayloadDigestReport,
    pub(in crate::runner) tokenize_ms: f64,
    pub(in crate::runner) source_prefill_ms: f64,
    pub(in crate::runner) source_export_ms: f64,
    pub(in crate::runner) source_decode_ms: f64,
    pub(in crate::runner) restore_import_ms: f64,
    pub(in crate::runner) restore_export_ms: f64,
    pub(in crate::runner) restore_decode_ms: f64,
    pub(in crate::runner) cache_hit_import_ms: Vec<f64>,
    pub(in crate::runner) cache_hit_decode_ms: Vec<f64>,
    pub(in crate::runner) matches: bool,
    pub(in crate::runner) predicted_token_matches: bool,
    pub(in crate::runner) roundtrip_state_matches: bool,
    pub(in crate::runner) restored_output_matches: Option<bool>,
    pub(in crate::runner) suffix_prefill_matches: Option<bool>,
    pub(in crate::runner) cache_hit_matches: bool,
    pub(in crate::runner) stage_models: Vec<StageModelReport>,
}

#[derive(Clone)]
enum LocalStatePayload {
    ResidentKv {
        cache_seq_id: i32,
        token_count: u64,
    },
    FullState(Vec<u8>),
    RecurrentOnly(Vec<u8>),
    KvRecurrent {
        kv_desc: Option<RuntimeKvPageDesc>,
        kv: Vec<u8>,
        recurrent: Vec<u8>,
    },
}

impl LocalStatePayload {
    fn byte_len(&self) -> usize {
        match self {
            Self::ResidentKv { .. } => 0,
            Self::FullState(bytes) | Self::RecurrentOnly(bytes) => bytes.len(),
            Self::KvRecurrent { kv, recurrent, .. } => kv.len().saturating_add(recurrent.len()),
        }
    }

    fn same_payload(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::ResidentKv {
                    token_count: a_count,
                    ..
                },
                Self::ResidentKv {
                    token_count: b_count,
                    ..
                },
            ) => a_count == b_count,
            (Self::FullState(a), Self::FullState(b))
            | (Self::RecurrentOnly(a), Self::RecurrentOnly(b)) => a == b,
            (
                Self::KvRecurrent {
                    kv_desc: a_desc,
                    kv: a_kv,
                    recurrent: a_recurrent,
                },
                Self::KvRecurrent {
                    kv_desc: b_desc,
                    kv: b_kv,
                    recurrent: b_recurrent,
                },
            ) => a_desc == b_desc && a_kv == b_kv && a_recurrent == b_recurrent,
            _ => false,
        }
    }

    fn digest_report(&self) -> StatePayloadDigestReport {
        match self {
            Self::ResidentKv { .. } => payload_digest_report(
                state_payload_kind_name(StatePayloadKind::ResidentKv),
                &[],
                None,
                None,
            ),
            Self::FullState(bytes) => payload_digest_report(
                state_payload_kind_name(StatePayloadKind::FullState),
                bytes,
                None,
                None,
            ),
            Self::RecurrentOnly(recurrent) => payload_digest_report(
                state_payload_kind_name(StatePayloadKind::RecurrentOnly),
                recurrent,
                None,
                Some(recurrent.as_slice()),
            ),
            Self::KvRecurrent { kv, recurrent, .. } => {
                let mut hasher = Sha256::new();
                hasher.update(b"kv-recurrent:kv:");
                hasher.update((kv.len() as u64).to_le_bytes());
                hasher.update(kv);
                hasher.update(b":recurrent:");
                hasher.update((recurrent.len() as u64).to_le_bytes());
                hasher.update(recurrent);
                let mut report = payload_digest_report(
                    state_payload_kind_name(StatePayloadKind::KvRecurrent),
                    &[],
                    Some(kv.as_slice()),
                    Some(recurrent.as_slice()),
                );
                report.payload_sha256 = hex_sha256_finish(hasher);
                report.total_bytes = kv.len().saturating_add(recurrent.len());
                report
            }
        }
    }
}
pub fn state_handoff(args: StateHandoffArgs) -> Result<()> {
    let report_out = args.output.report_out;
    let model_identity = runtime_model_identity(&args.runtime)?;
    let state_layer_end = args.state_layer_end.unwrap_or(args.runtime.layer_end);
    let state_stage_index = args.state_stage_index.unwrap_or({
        if args.state_layer_start == 0 {
            0
        } else if state_layer_end == args.runtime.layer_end {
            2
        } else {
            1
        }
    });
    let handoff = run_binary_state_handoff(BinaryStateHandoffConfig {
        stage_server_bin: args.server.stage_server_bin,
        model: args.runtime.model,
        stage_model: args.runtime.stage_model,
        stage_load_mode: args.runtime.stage_load_mode,
        state_layer_start: args.state_layer_start,
        state_layer_end,
        state_stage_index,
        layer_end: args.runtime.layer_end,
        ctx_size: args.runtime.ctx_size,
        n_batch: args.runtime.n_batch,
        n_ubatch: args.runtime.n_ubatch,
        n_gpu_layers: args.runtime.n_gpu_layers,
        flash_attn: args.runtime.flash_attn,
        prompt: args.runtime.prompt,
        source_bind_addr: args.source_bind_addr,
        restore_bind_addr: args.restore_bind_addr,
        activation_width: args.activation_width,
        state_payload_kind: args.state_payload_kind,
        prefix_token_count: args.prefix_token_count,
        cache_hit_repeats: args.cache_hit_repeats,
        runtime_lane_count: args.runtime_lane_count,
        borrow_resident_hits: args.borrow_resident_hits,
        cache_decoded_result_hits: args.cache_decoded_result_hits,
        skip_suffix_prefill_check: args.skip_suffix_prefill_check,
        synthetic_input_activation: args.synthetic_input_activation,
        binary_control: args.binary_control,
        child_logs: args.server.child_logs,
        startup_timeout_secs: args.server.startup_timeout_secs,
        max_inflight: args.server.max_inflight,
        model_identity: model_identity.clone(),
    })?;
    let report = StateHandoffReport {
        mode: "state-handoff",
        status: status(handoff.matches),
        model_identity,
        matches: handoff.matches,
        predicted_token_matches: handoff.predicted_token_matches,
        roundtrip_state_matches: handoff.roundtrip_state_matches,
        restored_output_matches: handoff.restored_output_matches,
        suffix_prefill_matches: handoff.suffix_prefill_matches,
        cache_hit_matches: handoff.cache_hit_matches,
        stage_index: handoff.stage_index,
        layer_start: handoff.layer_start,
        layer_end: handoff.layer_end,
        include_embeddings: handoff.include_embeddings,
        include_output: handoff.include_output,
        handoff_transport: handoff.handoff_transport,
        state_payload_kind: state_payload_kind_name(handoff.state_payload_kind),
        borrowed_resident_hits: handoff.borrowed_resident_hits,
        cached_decoded_result_hits: handoff.cached_decoded_result_hits,
        source_predicted_token: handoff.source_predicted_token,
        restored_predicted_token: handoff.restored_predicted_token,
        prompt_token_count: handoff.prompt_token_count,
        benchmark_prompt_token_count: handoff.benchmark_prompt_token_count,
        benchmark_prompt_text: handoff.benchmark_prompt_text,
        requested_prefix_token_count: handoff.requested_prefix_token_count,
        activation_width: handoff.activation_width,
        state_bytes: handoff.state_bytes,
        state_bytes_per_prompt_token: handoff.state_bytes as f64
            / handoff.prompt_token_count as f64,
        cache_storage_bytes: handoff.cache_storage_bytes,
        cache_storage_bytes_per_prompt_token: handoff
            .cache_storage_bytes
            .map(|bytes| bytes as f64 / handoff.prompt_token_count as f64),
        resident_state_bytes: handoff.resident_state_bytes,
        roundtrip_state_bytes: handoff.roundtrip_state_bytes,
        payload_digest: handoff.payload_digest,
        tokenize_ms: handoff.tokenize_ms,
        source_prefill_ms: handoff.source_prefill_ms,
        source_export_ms: handoff.source_export_ms,
        source_decode_ms: handoff.source_decode_ms,
        restore_import_ms: handoff.restore_import_ms,
        restore_export_ms: handoff.restore_export_ms,
        restore_decode_ms: handoff.restore_decode_ms,
        cache_hit_repeats: handoff.cache_hit_import_ms.len(),
        recompute_total_ms: handoff.source_prefill_ms + handoff.source_decode_ms,
        cache_hit_total_ms: mean_pair_sum(
            &handoff.cache_hit_import_ms,
            &handoff.cache_hit_decode_ms,
        ),
        cache_hit_speedup: speedup(
            handoff.source_prefill_ms + handoff.source_decode_ms,
            mean_pair_sum(&handoff.cache_hit_import_ms, &handoff.cache_hit_decode_ms),
        ),
        cache_hit_import_ms: handoff.cache_hit_import_ms,
        cache_hit_decode_ms: handoff.cache_hit_decode_ms,
        stage_models: handoff.stage_models,
    };
    emit_report(&report, report_out.as_deref())?;
    ensure_matches(report.matches, args.allow_mismatch)?;
    Ok(())
}
fn run_binary_state_handoff(args: BinaryStateHandoffConfig) -> Result<BinaryStateHandoffResult> {
    let tokenize_started = Instant::now();
    if args.state_layer_start >= args.state_layer_end || args.state_layer_end > args.layer_end {
        bail!(
            "state handoff range {}..{} must be non-empty and within 0..{}",
            args.state_layer_start,
            args.state_layer_end,
            args.layer_end
        );
    }
    if args.cache_hit_repeats == 0 {
        bail!("cache_hit_repeats must be greater than zero");
    }
    let include_embeddings = args.state_layer_start == 0;
    let include_output = args.state_layer_end == args.layer_end;
    let stage_spec = PackageStageSpec {
        topology_id: "correctness-state-handoff",
        stage_id: stage_id_for_index(args.state_stage_index),
        stage_index: args.state_stage_index,
        layer_start: args.state_layer_start,
        layer_end: args.state_layer_end,
        include_embeddings,
        include_output,
    };
    let stage_resolution = stage_model_resolution(
        &args.model,
        args.stage_model.as_ref(),
        args.stage_load_mode,
        &args.model_identity,
        stage_spec,
    )?;
    let (tokenizer_path, tokenizer_config) = tokenizer_model_for_state_handoff(&args)?;
    let tokenizer = StageModel::open(&tokenizer_path, &tokenizer_config).with_context(|| {
        format!(
            "failed to open tokenizer model {}",
            tokenizer_path.display()
        )
    })?;
    let tokens = state_handoff_tokens(&tokenizer, &args.prompt, args.prefix_token_count)
        .context("failed to tokenize state handoff prompt")?;
    let split = args.prefix_token_count.unwrap_or(tokens.len() - 1);
    let prefix = tokens[..split].to_vec();
    let continuation = tokens[split];
    let benchmark_prompt_text = tokenizer
        .detokenize(&tokens[..=split])
        .context("failed to detokenize state handoff benchmark prompt")?;
    drop(tokenizer);
    let tokenize_ms = elapsed_ms(tokenize_started);
    let input_started = Instant::now();
    let input_resolution = if args.state_layer_start == 0 || args.synthetic_input_activation {
        None
    } else {
        Some(stage_model_resolution(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            &args.model_identity,
            PackageStageSpec {
                topology_id: "correctness-state-handoff",
                stage_id: stage_id_for_index(args.state_stage_index.saturating_sub(1)),
                stage_index: args.state_stage_index.saturating_sub(1),
                layer_start: 0,
                layer_end: args.state_layer_start,
                include_embeddings: true,
                include_output: false,
            },
        )?)
    };
    let (prefill_input, decode_input, stage_activation_width) =
        build_state_handoff_inputs(&args, input_resolution.as_ref(), &prefix, continuation)
            .context("build state handoff input activations")?;
    let input_build_ms = elapsed_ms(input_started);
    let use_binary_control = args.binary_control
        && include_output
        && args.state_payload_kind == StatePayloadKind::FullState;
    if !use_binary_control {
        return run_local_state_handoff(
            &args,
            stage_resolution,
            input_resolution,
            prefix,
            continuation,
            benchmark_prompt_text,
            prefill_input,
            decode_input,
            tokenize_ms + input_build_ms,
            stage_activation_width,
            include_embeddings,
            include_output,
        );
    }

    let run_id = generate_run_id();
    let model_id = args.model_identity.model_id.clone();
    let source_config_path = temp_config_path_for(&run_id, "state-source");
    let restore_config_path = temp_config_path_for(&run_id, "state-restore");
    let source_config = json!({
        "run_id": run_id,
        "topology_id": "correctness-state-handoff",
        "model_id": model_id,
        "model_path": stage_server_model_path(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            stage_spec,
        )?,
        "stage_id": "state-source",
        "stage_index": args.state_stage_index,
        "layer_start": args.state_layer_start,
        "layer_end": args.state_layer_end,
        "ctx_size": args.ctx_size,
        "n_batch": args.n_batch,
        "n_ubatch": args.n_ubatch,
        "n_gpu_layers": args.n_gpu_layers,
        "flash_attn_type": protocol_flash_attn(args.flash_attn),
        "filter_tensors_on_load": should_filter_state_handoff_tensors(&args),
        "load_mode": protocol_load_mode(args.stage_load_mode),
        "bind_addr": args.source_bind_addr,
        "upstream": {
            "stage_id": "driver",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": null
    });
    let restore_config = json!({
        "run_id": run_id,
        "topology_id": "correctness-state-handoff",
        "model_id": model_id,
        "model_path": stage_server_model_path(
            &args.model,
            args.stage_model.as_ref(),
            args.stage_load_mode,
            stage_spec,
        )?,
        "stage_id": "state-restore",
        "stage_index": args.state_stage_index,
        "layer_start": args.state_layer_start,
        "layer_end": args.state_layer_end,
        "ctx_size": args.ctx_size,
        "n_batch": args.n_batch,
        "n_ubatch": args.n_ubatch,
        "n_gpu_layers": args.n_gpu_layers,
        "flash_attn_type": protocol_flash_attn(args.flash_attn),
        "filter_tensors_on_load": should_filter_state_handoff_tensors(&args),
        "load_mode": protocol_load_mode(args.stage_load_mode),
        "bind_addr": args.restore_bind_addr,
        "upstream": {
            "stage_id": "driver",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": null
    });
    fs::write(
        &source_config_path,
        serde_json::to_vec_pretty(&source_config)?,
    )
    .with_context(|| format!("failed to write {}", source_config_path.display()))?;
    fs::write(
        &restore_config_path,
        serde_json::to_vec_pretty(&restore_config)?,
    )
    .with_context(|| format!("failed to write {}", restore_config_path.display()))?;

    let mut source_command = Command::new(&args.stage_server_bin);
    source_command.args([
        "serve-binary",
        "--config",
        source_config_path
            .to_str()
            .context("source config path is not valid UTF-8")?,
        "--max-inflight",
        &args.max_inflight.to_string(),
    ]);
    configure_child_logs(&mut source_command, args.child_logs);
    let mut source = ChildGuard::spawn(source_command)?;

    let mut source_stream = connect_ready_child(
        args.source_bind_addr,
        args.startup_timeout_secs,
        &mut source,
    )
    .context("source binary server did not become ready")?;
    let source_prefill_started = Instant::now();
    send_prefill_for_state_handoff(
        &mut source_stream,
        &prefix,
        prefill_input.as_ref(),
        stage_activation_width,
    )
    .context("send source prefill")?;
    let source_prefill_ms = elapsed_ms(source_prefill_started);
    let source_export_started = Instant::now();
    let state_bytes = export_state_over_binary(&mut source_stream, args.activation_width, true)
        .context("export source state")?;
    let source_export_ms = elapsed_ms(source_export_started);
    let source_decode_started = Instant::now();
    let source_predicted_token = decode_for_state_handoff(
        &mut source_stream,
        continuation,
        prefix.len(),
        decode_input.as_ref(),
        stage_activation_width,
    )
    .context("decode source continuation")?;
    let source_decode_ms = elapsed_ms(source_decode_started);
    write_stage_message(&mut source_stream, &StageWireMessage::stop())
        .context("send source stop")?;
    drop(source_stream);
    drop(source);

    let mut restore_command = Command::new(&args.stage_server_bin);
    restore_command.args([
        "serve-binary",
        "--config",
        restore_config_path
            .to_str()
            .context("restore config path is not valid UTF-8")?,
        "--max-inflight",
        &args.max_inflight.to_string(),
    ]);
    configure_child_logs(&mut restore_command, args.child_logs);
    let mut restore = ChildGuard::spawn(restore_command)?;

    let mut restore_stream = connect_ready_child(
        args.restore_bind_addr,
        args.startup_timeout_secs,
        &mut restore,
    )
    .context("restore binary server did not become ready")?;
    let restore_import_started = Instant::now();
    import_state_over_binary(&mut restore_stream, &state_bytes, true)
        .context("import state into restore server")?;
    let restore_import_ms = elapsed_ms(restore_import_started);
    let restore_export_started = Instant::now();
    let roundtrip_state_bytes =
        export_state_over_binary(&mut restore_stream, args.activation_width, true)
            .context("export restored state")?;
    let restore_export_ms = elapsed_ms(restore_export_started);
    let restore_decode_started = Instant::now();
    let restored_predicted_token = decode_for_state_handoff(
        &mut restore_stream,
        continuation,
        prefix.len(),
        decode_input.as_ref(),
        stage_activation_width,
    )
    .context("decode restored continuation")?;
    let restore_decode_ms = elapsed_ms(restore_decode_started);
    let mut cache_hit_import_ms = vec![restore_import_ms];
    let mut cache_hit_decode_ms = vec![restore_decode_ms];
    let predicted_token_matches = source_predicted_token == restored_predicted_token;
    let mut cache_hit_matches = predicted_token_matches;
    for _ in 1..args.cache_hit_repeats {
        let import_started = Instant::now();
        import_state_over_binary(&mut restore_stream, &state_bytes, true)
            .context("repeat import state into restore server")?;
        cache_hit_import_ms.push(elapsed_ms(import_started));
        let decode_started = Instant::now();
        let predicted = decode_for_state_handoff(
            &mut restore_stream,
            continuation,
            prefix.len(),
            decode_input.as_ref(),
            stage_activation_width,
        )
        .context("repeat decode restored continuation")?;
        cache_hit_decode_ms.push(elapsed_ms(decode_started));
        cache_hit_matches &= predicted == source_predicted_token;
    }
    write_stage_message(&mut restore_stream, &StageWireMessage::stop())
        .context("send restore stop")?;

    let mut stage_models = Vec::new();
    if let Some(input_resolution) = input_resolution {
        stage_models.push(input_resolution.report);
    }
    stage_models.push(stage_resolution.report);

    let roundtrip_state_matches = state_bytes == roundtrip_state_bytes;
    Ok(BinaryStateHandoffResult {
        prompt_token_count: prefix.len(),
        benchmark_prompt_token_count: prefix.len().saturating_add(1),
        benchmark_prompt_text: benchmark_prompt_text.clone(),
        requested_prefix_token_count: args.prefix_token_count,
        stage_index: args.state_stage_index,
        layer_start: args.state_layer_start,
        layer_end: args.state_layer_end,
        include_embeddings,
        include_output,
        handoff_transport: "binary-control",
        state_payload_kind: args.state_payload_kind,
        borrowed_resident_hits: false,
        cached_decoded_result_hits: false,
        activation_width: stage_activation_width,
        source_predicted_token,
        restored_predicted_token,
        state_bytes: state_bytes.len(),
        cache_storage_bytes: Some(state_bytes.len()),
        resident_state_bytes: None,
        roundtrip_state_bytes: roundtrip_state_bytes.len(),
        payload_digest: payload_digest_report(
            state_payload_kind_name(args.state_payload_kind),
            &state_bytes,
            None,
            None,
        ),
        tokenize_ms: tokenize_ms + input_build_ms,
        source_prefill_ms,
        source_export_ms,
        source_decode_ms,
        restore_import_ms,
        restore_export_ms,
        restore_decode_ms,
        cache_hit_import_ms,
        cache_hit_decode_ms,
        matches: predicted_token_matches && cache_hit_matches,
        predicted_token_matches,
        roundtrip_state_matches,
        restored_output_matches: None,
        suffix_prefill_matches: None,
        cache_hit_matches,
        stage_models,
    })
}
fn state_handoff_tokens(
    tokenizer: &StageModel,
    prompt: &str,
    prefix_token_count: Option<usize>,
) -> Result<Vec<i32>> {
    let mut text = prompt.to_string();
    let needed = prefix_token_count.unwrap_or(1).saturating_add(1);
    let mut tokens = tokenizer
        .tokenize(&text, true)
        .context("failed to tokenize prompt")?;
    if tokens.len() >= needed {
        tokens.truncate(needed);
        return Ok(tokens);
    }

    let filler = " Deterministic full-state cache economics filler sentence.";
    for _ in 0..10_000 {
        text.push_str(filler);
        tokens = tokenizer
            .tokenize(&text, true)
            .context("failed to tokenize expanded prompt")?;
        if tokens.len() >= needed {
            tokens.truncate(needed);
            return Ok(tokens);
        }
    }

    bail!(
        "could not expand prompt to {needed} tokens for state handoff prefix sweep; reached {} tokens",
        tokens.len()
    )
}

#[allow(clippy::too_many_arguments)]
fn run_local_state_handoff(
    args: &BinaryStateHandoffConfig,
    stage_resolution: StageModelResolution,
    input_resolution: Option<StageModelResolution>,
    prefix: Vec<i32>,
    continuation: i32,
    benchmark_prompt_text: String,
    prefill_input: Option<ActivationFrame>,
    decode_input: Option<ActivationFrame>,
    tokenize_ms: f64,
    activation_width: i32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<BinaryStateHandoffResult> {
    let lane_count = effective_state_handoff_lane_count(args);
    let runtime_config = RuntimeConfig {
        stage_index: args.state_stage_index,
        layer_start: args.state_layer_start,
        layer_end: args.state_layer_end,
        ctx_size: args.ctx_size,
        lane_count,
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
        include_embeddings,
        include_output,
        mtp_source: MtpSource::Disabled,
        filter_tensors_on_load: should_filter_state_handoff_tensors(args),
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
    let model = StageModel::open(&stage_resolution.path, &runtime_config)
        .context("failed to open local state handoff stage")?;

    if args.borrow_resident_hits && args.state_payload_kind == StatePayloadKind::ResidentKv {
        return run_local_resident_slot_handoff(
            model,
            args,
            stage_resolution,
            input_resolution,
            prefix,
            continuation,
            benchmark_prompt_text,
            prefill_input,
            decode_input,
            tokenize_ms,
            activation_width,
            include_embeddings,
            include_output,
        );
    }

    let mut source = model
        .create_session()
        .context("failed to create local state handoff source session")?;
    let source_prefill_started = Instant::now();
    if prefill_input.is_some() {
        source
            .prefill_chunk_frame(&prefix, prefill_input.as_ref(), 0)
            .context("local state handoff source prefill failed")?;
    } else {
        source
            .prefill_chunked(&prefix)
            .context("local state handoff source prefill failed")?;
    }
    let source_prefill_ms = elapsed_ms(source_prefill_started);

    let source_export_started = Instant::now();
    let state_payload = export_local_state_payload(&mut source, args, prefix.len() as u64)
        .context("local state handoff source export failed")?;
    let source_export_ms = elapsed_ms(source_export_started);

    let source_decode_started = Instant::now();
    let (source_predicted_token, source_output) = source
        .decode_step_frame(continuation, decode_input.as_ref(), 0)
        .context("local state handoff source decode failed")?;
    let source_decode_ms = elapsed_ms(source_decode_started);

    let resident_state_bytes = measure_resident_state_bytes(&mut source, args, prefix.len() as u64)
        .context("local state handoff resident KV size measurement failed")?;
    let source_guard = (args.state_payload_kind == StatePayloadKind::ResidentKv).then_some(source);

    let (
        roundtrip_state_payload,
        restored_predicted_token,
        restored_output,
        restore_import_ms,
        restore_export_ms,
        restore_decode_ms,
    ) = {
        let restore_import_started = Instant::now();
        let mut restore = create_local_cache_hit_session(
            &model,
            args,
            &state_payload,
            prefix.len() as u64,
            &prefix,
        )
        .context("local state handoff restore import failed")?;
        let restore_import_ms = elapsed_ms(restore_import_started);

        let restore_export_started = Instant::now();
        let roundtrip_state_payload = if args.borrow_resident_hits
            && args.state_payload_kind == StatePayloadKind::ResidentKv
        {
            state_payload.clone()
        } else {
            export_local_state_payload(&mut restore, args, prefix.len() as u64)
                .context("local state handoff restore export failed")?
        };
        let restore_export_ms = elapsed_ms(restore_export_started);

        let restore_decode_started = Instant::now();
        let (restored_predicted_token, restored_output) = restore
            .decode_step_frame(continuation, decode_input.as_ref(), 0)
            .context("local state handoff restore decode failed")?;
        let restore_decode_ms = elapsed_ms(restore_decode_started);
        (
            roundtrip_state_payload,
            restored_predicted_token,
            restored_output,
            restore_import_ms,
            restore_export_ms,
            restore_decode_ms,
        )
    };
    let mut cache_hit_import_ms = vec![restore_import_ms];
    let mut cache_hit_decode_ms = vec![restore_decode_ms];
    let predicted_token_matches = source_predicted_token == restored_predicted_token;
    let restored_output_matches = source_output.payload == restored_output.payload;
    let mut cache_hit_matches = predicted_token_matches && restored_output_matches;
    for _ in 1..args.cache_hit_repeats {
        let import_started = Instant::now();
        let mut hit = create_local_cache_hit_session(
            &model,
            args,
            &state_payload,
            prefix.len() as u64,
            &prefix,
        )
        .context("local state handoff repeat import failed")?;
        cache_hit_import_ms.push(elapsed_ms(import_started));
        let decode_started = Instant::now();
        let (predicted, output) = hit
            .decode_step_frame(continuation, decode_input.as_ref(), 0)
            .context("local state handoff repeat decode failed")?;
        cache_hit_decode_ms.push(elapsed_ms(decode_started));
        cache_hit_matches &=
            predicted == source_predicted_token && output.payload == source_output.payload;
    }
    let suffix_prefill_matches = if args.skip_suffix_prefill_check {
        drop(source_guard);
        None
    } else {
        let mut suffix_restored = create_local_cache_hit_session(
            &model,
            args,
            &state_payload,
            prefix.len() as u64,
            &prefix,
        )
        .context("local state handoff suffix-prefill restore import failed")?;
        drop(source_guard);
        Some(
            run_local_suffix_prefill_remap_check(
                &model,
                &mut suffix_restored,
                &prefix,
                continuation,
                prefill_input.as_ref(),
                decode_input.as_ref(),
                include_output,
            )
            .context("local state handoff suffix-prefill remap check failed")?,
        )
    };

    let mut stage_models = Vec::new();
    if let Some(input_resolution) = input_resolution {
        stage_models.push(input_resolution.report);
    }
    stage_models.push(stage_resolution.report);

    let roundtrip_state_matches = state_payload.same_payload(&roundtrip_state_payload);
    Ok(BinaryStateHandoffResult {
        prompt_token_count: prefix.len(),
        benchmark_prompt_token_count: prefix.len().saturating_add(1),
        benchmark_prompt_text,
        requested_prefix_token_count: args.prefix_token_count,
        stage_index: args.state_stage_index,
        layer_start: args.state_layer_start,
        layer_end: args.state_layer_end,
        include_embeddings,
        include_output,
        handoff_transport: "local-runtime",
        state_payload_kind: args.state_payload_kind,
        borrowed_resident_hits: args.borrow_resident_hits
            && args.state_payload_kind == StatePayloadKind::ResidentKv,
        cached_decoded_result_hits: false,
        activation_width,
        source_predicted_token,
        restored_predicted_token,
        state_bytes: state_payload.byte_len(),
        cache_storage_bytes: match args.state_payload_kind {
            StatePayloadKind::ResidentKv => resident_state_bytes,
            _ => Some(state_payload.byte_len()),
        },
        resident_state_bytes,
        roundtrip_state_bytes: roundtrip_state_payload.byte_len(),
        payload_digest: state_payload.digest_report(),
        tokenize_ms,
        source_prefill_ms,
        source_export_ms,
        source_decode_ms,
        restore_import_ms,
        restore_export_ms,
        restore_decode_ms,
        cache_hit_import_ms,
        cache_hit_decode_ms,
        matches: predicted_token_matches
            && restored_output_matches
            && suffix_prefill_matches.unwrap_or(true)
            && cache_hit_matches,
        predicted_token_matches,
        roundtrip_state_matches,
        restored_output_matches: Some(restored_output_matches),
        suffix_prefill_matches,
        cache_hit_matches,
        stage_models,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_local_resident_slot_handoff(
    model: StageModel,
    args: &BinaryStateHandoffConfig,
    stage_resolution: StageModelResolution,
    input_resolution: Option<StageModelResolution>,
    prefix: Vec<i32>,
    continuation: i32,
    benchmark_prompt_text: String,
    prefill_input: Option<ActivationFrame>,
    decode_input: Option<ActivationFrame>,
    tokenize_ms: f64,
    activation_width: i32,
    include_embeddings: bool,
    include_output: bool,
) -> Result<BinaryStateHandoffResult> {
    let mut slot = model
        .create_session()
        .context("failed to create local resident slot session")?;
    let source_prefill_started = Instant::now();
    if prefill_input.is_some() {
        slot.prefill_chunk_frame(&prefix, prefill_input.as_ref(), 0)
            .context("local resident slot source prefill failed")?;
    } else {
        slot.prefill_chunked(&prefix)
            .context("local resident slot source prefill failed")?;
    }
    let source_prefill_ms = elapsed_ms(source_prefill_started);

    let resident_state_bytes = measure_resident_state_bytes(&mut slot, args, prefix.len() as u64)
        .context("local resident slot KV size measurement failed")?;
    let cache_seq_id = resident_cache_seq_id(args);
    slot.save_prefix(cache_seq_id, prefix.len() as u64)
        .context("local resident slot save prefix failed")?;
    let state_payload = LocalStatePayload::ResidentKv {
        cache_seq_id,
        token_count: prefix.len() as u64,
    };

    let source_export_ms = 0.0;
    let source_decode_started = Instant::now();
    let (source_predicted_token, source_output) = slot
        .decode_step_frame(continuation, decode_input.as_ref(), 0)
        .context("local resident slot source decode failed")?;
    let source_decode_ms = elapsed_ms(source_decode_started);

    let restore_import_started = Instant::now();
    let mut restore = model
        .create_session_from_resident_prefix(cache_seq_id, &prefix)
        .context("local resident slot restore borrow failed")?;
    let restore_import_ms = elapsed_ms(restore_import_started);
    let restore_export_ms = 0.0;
    let restore_decode_started = Instant::now();
    let (restored_predicted_token, restored_output) =
        if args.cache_decoded_result_hits && decode_input.is_none() {
            (
                restore
                    .sample_current(None)
                    .context("local resident slot restore sample failed")?,
                ActivationFrame {
                    desc: skippy_runtime::ActivationDesc {
                        version: 0,
                        dtype: skippy_runtime::RuntimeActivationDType::Unknown,
                        layout: skippy_runtime::RuntimeActivationLayout::Opaque,
                        producer_stage_index: args.state_stage_index as i32,
                        layer_start: args.state_layer_start as i32,
                        layer_end: args.state_layer_end as i32,
                        token_count: 0,
                        sequence_count: 0,
                        payload_bytes: 0,
                        flags: 0,
                    },
                    payload: Vec::new(),
                },
            )
        } else {
            restore
                .decode_step_frame(continuation, decode_input.as_ref(), 0)
                .context("local resident slot restore decode failed")?
        };
    let restore_decode_ms = elapsed_ms(restore_decode_started);
    drop(restore);

    let mut cache_hit_import_ms = vec![restore_import_ms];
    let mut cache_hit_decode_ms = vec![restore_decode_ms];
    let predicted_token_matches = source_predicted_token == restored_predicted_token;
    let restored_output_matches = source_output.payload == restored_output.payload;
    let mut cache_hit_matches = predicted_token_matches && restored_output_matches;
    for _ in 1..args.cache_hit_repeats {
        let import_started = Instant::now();
        let mut hit = model
            .create_session_from_resident_prefix(cache_seq_id, &prefix)
            .context("local resident slot repeat borrow failed")?;
        cache_hit_import_ms.push(elapsed_ms(import_started));
        let decode_started = Instant::now();
        let (predicted, output) = if args.cache_decoded_result_hits && decode_input.is_none() {
            (
                hit.sample_current(None)
                    .context("local resident slot repeat sample failed")?,
                ActivationFrame {
                    desc: skippy_runtime::ActivationDesc {
                        version: 0,
                        dtype: skippy_runtime::RuntimeActivationDType::Unknown,
                        layout: skippy_runtime::RuntimeActivationLayout::Opaque,
                        producer_stage_index: args.state_stage_index as i32,
                        layer_start: args.state_layer_start as i32,
                        layer_end: args.state_layer_end as i32,
                        token_count: 0,
                        sequence_count: 0,
                        payload_bytes: 0,
                        flags: 0,
                    },
                    payload: Vec::new(),
                },
            )
        } else {
            hit.decode_step_frame(continuation, decode_input.as_ref(), 0)
                .context("local resident slot repeat decode failed")?
        };
        cache_hit_decode_ms.push(elapsed_ms(decode_started));
        cache_hit_matches &=
            predicted == source_predicted_token && output.payload == source_output.payload;
    }
    let mut suffix_restored =
        create_local_cache_hit_session(&model, args, &state_payload, prefix.len() as u64, &prefix)
            .context("local resident slot suffix-prefill restore import failed")?;
    drop(slot);
    let suffix_prefill_matches = run_local_suffix_prefill_remap_check(
        &model,
        &mut suffix_restored,
        &prefix,
        continuation,
        prefill_input.as_ref(),
        decode_input.as_ref(),
        include_output,
    )
    .context("local resident slot suffix-prefill remap check failed")?;

    let mut stage_models = Vec::new();
    if let Some(input_resolution) = input_resolution {
        stage_models.push(input_resolution.report);
    }
    stage_models.push(stage_resolution.report);

    Ok(BinaryStateHandoffResult {
        prompt_token_count: prefix.len(),
        benchmark_prompt_token_count: prefix.len().saturating_add(1),
        benchmark_prompt_text,
        requested_prefix_token_count: args.prefix_token_count,
        stage_index: args.state_stage_index,
        layer_start: args.state_layer_start,
        layer_end: args.state_layer_end,
        include_embeddings,
        include_output,
        handoff_transport: "local-runtime",
        state_payload_kind: args.state_payload_kind,
        borrowed_resident_hits: true,
        cached_decoded_result_hits: args.cache_decoded_result_hits,
        activation_width,
        source_predicted_token,
        restored_predicted_token,
        state_bytes: state_payload.byte_len(),
        cache_storage_bytes: resident_state_bytes,
        resident_state_bytes,
        roundtrip_state_bytes: state_payload.byte_len(),
        payload_digest: state_payload.digest_report(),
        tokenize_ms,
        source_prefill_ms,
        source_export_ms,
        source_decode_ms,
        restore_import_ms,
        restore_export_ms,
        restore_decode_ms,
        cache_hit_import_ms,
        cache_hit_decode_ms,
        matches: predicted_token_matches
            && restored_output_matches
            && suffix_prefill_matches
            && cache_hit_matches,
        predicted_token_matches,
        roundtrip_state_matches: true,
        restored_output_matches: Some(restored_output_matches),
        suffix_prefill_matches: Some(suffix_prefill_matches),
        cache_hit_matches,
        stage_models,
    })
}

fn run_local_suffix_prefill_remap_check(
    model: &StageModel,
    restored: &mut StageSession,
    prefix: &[i32],
    suffix_token: i32,
    prefill_input: Option<&ActivationFrame>,
    decode_input: Option<&ActivationFrame>,
    include_output: bool,
) -> Result<bool> {
    let mut source = model
        .create_session()
        .context("failed to create suffix-prefill source session")?;
    if prefill_input.is_some() {
        source
            .prefill_chunk_frame(prefix, prefill_input, 0)
            .context("suffix-prefill source prefix prefill failed")?;
    } else {
        source
            .prefill_chunked(prefix)
            .context("suffix-prefill source prefix prefill failed")?;
    }
    let (source_predicted, source_frame) = if include_output {
        let (predicted, _, frame) = source
            .verify_tokens_frame(&[suffix_token], decode_input, 0)
            .context("suffix-prefill source suffix verify failed")?;
        (Some(predicted), frame)
    } else {
        (
            None,
            source
                .prefill_chunk_frame(&[suffix_token], decode_input, 0)
                .context("suffix-prefill source suffix prefill failed")?,
        )
    };

    let (restored_predicted, restored_frame) = if include_output {
        let (predicted, _, frame) = restored
            .verify_tokens_frame(&[suffix_token], decode_input, 0)
            .context("suffix-prefill restored suffix verify failed")?;
        (Some(predicted), frame)
    } else {
        (
            None,
            restored
                .prefill_chunk_frame(&[suffix_token], decode_input, 0)
                .context("suffix-prefill restored suffix prefill failed")?,
        )
    };

    Ok(source_frame.payload == restored_frame.payload && source_predicted == restored_predicted)
}

fn create_local_cache_hit_session(
    model: &StageModel,
    args: &BinaryStateHandoffConfig,
    payload: &LocalStatePayload,
    token_count: u64,
    token_ids: &[i32],
) -> Result<StageSession> {
    if args.borrow_resident_hits && args.state_payload_kind == StatePayloadKind::ResidentKv {
        let LocalStatePayload::ResidentKv {
            cache_seq_id,
            token_count: payload_token_count,
        } = payload
        else {
            bail!("borrowed resident hit requested with non-resident payload");
        };
        if *payload_token_count != token_count {
            bail!(
                "resident KV payload token count {} does not match requested import token count {token_count}",
                payload_token_count
            );
        }
        return model
            .create_session_from_resident_prefix(*cache_seq_id, token_ids)
            .context("failed to borrow resident prefix session");
    }
    let mut session = model
        .create_session()
        .context("failed to create local state handoff cache-hit session")?;
    import_local_state_payload(&mut session, args, payload, token_count, token_ids)?;
    Ok(session)
}

fn export_local_state_payload(
    session: &mut StageSession,
    args: &BinaryStateHandoffConfig,
    token_count: u64,
) -> Result<LocalStatePayload> {
    match args.state_payload_kind {
        StatePayloadKind::ResidentKv => {
            let cache_seq_id = resident_cache_seq_id(args);
            session.save_prefix(cache_seq_id, token_count)?;
            Ok(LocalStatePayload::ResidentKv {
                cache_seq_id,
                token_count,
            })
        }
        StatePayloadKind::FullState => Ok(LocalStatePayload::FullState(
            session
                .export_full_state(args.state_layer_start as i32, args.state_layer_end as i32)?,
        )),
        StatePayloadKind::RecurrentOnly => Ok(LocalStatePayload::RecurrentOnly(
            session.export_recurrent_state()?,
        )),
        StatePayloadKind::KvRecurrent => {
            let page = export_optional_kv_page(
                session,
                args.state_layer_start as i32,
                args.state_layer_end as i32,
                0,
                token_count,
            )?;
            let recurrent = session.export_recurrent_state()?;
            Ok(LocalStatePayload::KvRecurrent {
                kv_desc: page.as_ref().map(|page| page.desc.clone()),
                kv: page.map(|page| page.payload).unwrap_or_default(),
                recurrent,
            })
        }
    }
}

fn measure_resident_state_bytes(
    session: &mut StageSession,
    args: &BinaryStateHandoffConfig,
    token_count: u64,
) -> Result<Option<usize>> {
    if args.state_payload_kind != StatePayloadKind::ResidentKv {
        return Ok(None);
    }
    let page = match session.export_kv_page(
        args.state_layer_start as i32,
        args.state_layer_end as i32,
        0,
        token_count,
    ) {
        Ok(page) => page,
        Err(_) => return Ok(None),
    };
    Ok(Some(
        usize::try_from(page.desc.payload_bytes).unwrap_or(page.payload.len()),
    ))
}

fn import_local_state_payload(
    session: &mut StageSession,
    args: &BinaryStateHandoffConfig,
    payload: &LocalStatePayload,
    token_count: u64,
    token_ids: &[i32],
) -> Result<()> {
    match payload {
        LocalStatePayload::ResidentKv {
            cache_seq_id,
            token_count: payload_token_count,
        } => {
            if *payload_token_count != token_count {
                bail!(
                    "resident KV payload token count {} does not match requested import token count {token_count}",
                    payload_token_count
                );
            }
            session.restore_prefix(*cache_seq_id, token_ids)
        }
        LocalStatePayload::FullState(bytes) => session.import_full_state_for_token_count(
            args.state_layer_start as i32,
            args.state_layer_end as i32,
            bytes,
            token_count,
        ),
        LocalStatePayload::RecurrentOnly(bytes) => {
            session.import_recurrent_state_for_token_count(bytes, token_count)
        }
        LocalStatePayload::KvRecurrent {
            kv_desc,
            kv,
            recurrent,
        } => {
            if let Some(kv_desc) = kv_desc {
                session.import_kv_page(kv_desc, kv)?;
            } else if !kv.is_empty() {
                bail!("KV-recurrent payload has KV bytes but no KV descriptor");
            }
            session.import_recurrent_state_for_token_count(recurrent, token_count)
        }
    }
}

fn export_optional_kv_page(
    session: &mut StageSession,
    layer_start: i32,
    layer_end: i32,
    token_start: u64,
    token_count: u64,
) -> Result<Option<skippy_runtime::RuntimeKvPage>> {
    match session.export_kv_page(layer_start, layer_end, token_start, token_count) {
        Ok(page) => Ok(Some(page)),
        Err(error) if is_native_kv_unavailable(&error) => Ok(None),
        Err(error) => Err(error),
    }
}

fn is_native_kv_unavailable(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.contains("runtime memory type is not supported for native KV pages")
            || message.contains("runtime has no attention KV cache")
    })
}

fn state_payload_kind_name(kind: StatePayloadKind) -> &'static str {
    match kind {
        StatePayloadKind::ResidentKv => "resident-kv",
        StatePayloadKind::FullState => "full-state",
        StatePayloadKind::RecurrentOnly => "recurrent-only",
        StatePayloadKind::KvRecurrent => "kv-recurrent",
    }
}

fn resident_cache_seq_id(args: &BinaryStateHandoffConfig) -> i32 {
    let lane_count = effective_state_handoff_lane_count(args) as i32;
    lane_count.saturating_mul(2).saturating_add(1)
}

fn effective_state_handoff_lane_count(args: &BinaryStateHandoffConfig) -> u32 {
    args.runtime_lane_count
        .unwrap_or_else(|| args.cache_hit_repeats.saturating_add(2).max(2) as u32)
        .max(1)
}

fn payload_digest_report(
    payload_kind: &'static str,
    full_payload: &[u8],
    kv: Option<&[u8]>,
    recurrent: Option<&[u8]>,
) -> StatePayloadDigestReport {
    const BLOCK_SIZE_BYTES: usize = 1024 * 1024;
    let mut blocks = Vec::new();
    if full_payload.is_empty() {
        if let Some(kv) = kv {
            blocks.extend(block_digests("kv", kv, BLOCK_SIZE_BYTES));
        }
        if let Some(recurrent) = recurrent {
            blocks.extend(block_digests("recurrent", recurrent, BLOCK_SIZE_BYTES));
        }
    } else {
        blocks.extend(block_digests("payload", full_payload, BLOCK_SIZE_BYTES));
    }
    let unique_block_count = blocks
        .iter()
        .map(|block| block.sha256.as_str())
        .collect::<HashSet<_>>()
        .len();
    let block_count = blocks.len();
    StatePayloadDigestReport {
        payload_kind,
        payload_sha256: sha256_hex(full_payload),
        total_bytes: full_payload.len(),
        kv_bytes: kv.map_or(0, <[u8]>::len),
        kv_sha256: kv.map(sha256_hex),
        recurrent_bytes: recurrent.map_or(0, <[u8]>::len),
        recurrent_sha256: recurrent.map(sha256_hex),
        block_size_bytes: BLOCK_SIZE_BYTES,
        block_count,
        unique_block_count,
        duplicate_block_count: block_count.saturating_sub(unique_block_count),
        blocks,
    }
}

fn block_digests(
    component: &'static str,
    bytes: &[u8],
    block_size: usize,
) -> Vec<StatePayloadBlockDigestReport> {
    bytes
        .chunks(block_size)
        .enumerate()
        .map(|(index, chunk)| StatePayloadBlockDigestReport {
            component,
            index,
            offset: index.saturating_mul(block_size),
            bytes: chunk.len(),
            sha256: sha256_hex(chunk),
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_sha256_finish(hasher)
}

fn hex_sha256_finish(hasher: Sha256) -> String {
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn should_filter_state_handoff_tensors(args: &BinaryStateHandoffConfig) -> bool {
    args.stage_load_mode != StageLoadMode::RuntimeSlice
        || args.state_layer_start != 0
        || args.state_layer_end != args.layer_end
}

fn build_state_handoff_inputs(
    args: &BinaryStateHandoffConfig,
    input_resolution: Option<&StageModelResolution>,
    prefix: &[i32],
    continuation: i32,
) -> Result<(Option<ActivationFrame>, Option<ActivationFrame>, i32)> {
    if args.synthetic_input_activation {
        if args.state_layer_start == 0 {
            bail!("--synthetic-input-activation requires --state-layer-start greater than zero");
        }
        let prefill_input = synthetic_activation_frame(args, prefix.len() as u32, 0);
        let decode_input = synthetic_activation_frame(args, 1, continuation);
        return Ok((
            Some(prefill_input),
            Some(decode_input),
            args.activation_width,
        ));
    }
    let Some(input_resolution) = input_resolution else {
        return Ok((None, None, args.activation_width));
    };
    let input_config = RuntimeConfig {
        stage_index: args.state_stage_index.saturating_sub(1),
        layer_start: 0,
        layer_end: args.state_layer_start,
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
    let input_model = StageModel::open(&input_resolution.path, &input_config)
        .context("failed to open state handoff input producer")?;
    let mut input_session = input_model
        .create_session()
        .context("failed to create state handoff input producer session")?;
    let prefill_input = input_session
        .prefill_chunk_frame(prefix, None, 0)
        .context("state handoff input producer failed to prefill prefix")?;
    let (_, decode_input) = input_session
        .decode_step_frame(continuation, None, 0)
        .context("state handoff input producer failed to decode continuation")?;
    let prefill_width = activation_width(&prefill_input)?;
    let decode_width = activation_width(&decode_input)?;
    if prefill_width != decode_width {
        bail!(
            "state handoff input width changed between prefill ({prefill_width}) and decode ({decode_width})"
        );
    }
    Ok((Some(prefill_input), Some(decode_input), prefill_width))
}

fn synthetic_activation_frame(
    args: &BinaryStateHandoffConfig,
    token_count: u32,
    token_seed: i32,
) -> ActivationFrame {
    let width = args.activation_width.max(0) as usize;
    let token_count_usize = token_count as usize;
    let mut payload = Vec::with_capacity(token_count_usize * width * std::mem::size_of::<f32>());
    for token_index in 0..token_count_usize {
        for column in 0..width {
            let raw = (token_index as i32 * 31
                + column as i32 * 17
                + token_seed
                + args.state_layer_start as i32 * 13)
                .rem_euclid(2048);
            let value = (raw as f32 / 2048.0) - 0.5;
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    ActivationFrame {
        desc: skippy_runtime::ActivationDesc {
            version: 1,
            dtype: skippy_runtime::RuntimeActivationDType::F32,
            layout: skippy_runtime::RuntimeActivationLayout::TokenMajor,
            producer_stage_index: args.state_stage_index.saturating_sub(1) as i32,
            layer_start: 0,
            layer_end: args.state_layer_start as i32,
            token_count,
            sequence_count: if token_count > 0 { 1 } else { 0 },
            payload_bytes: payload.len() as u64,
            flags: 0,
        },
        payload,
    }
}

fn send_prefill_for_state_handoff(
    stream: &mut std::net::TcpStream,
    tokens: &[i32],
    input: Option<&ActivationFrame>,
    activation_width: i32,
) -> Result<()> {
    let token_count = i32::try_from(tokens.len()).context("too many prompt tokens")?;
    let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd);
    state.prompt_token_count = token_count;
    state.current_token = tokens
        .last()
        .copied()
        .unwrap_or(skippy_protocol::binary::LLAMA_TOKEN_NULL);
    state.source_stage_index = input
        .map(|frame| frame.desc.producer_stage_index)
        .unwrap_or(-1);
    state.flags |= activation_state_flags_optional(input);
    let activation =
        encode_handoff_activation(state.activation_codec, input, token_count, activation_width)?;
    let message = StageWireMessage {
        kind: WireMessageKind::PrefillEmbd,
        pos_start: 0,
        token_count,
        state,
        request_id: 1,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: tokens.to_vec(),
        positions: Vec::new(),
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message)?;
    let reply = recv_reply(&mut *stream).context("receive prefill ACK")?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected prefill ACK, got {:?}", reply.kind);
    }
    Ok(())
}

fn export_state_over_binary(
    stream: &mut std::net::TcpStream,
    activation_width: i32,
    full_state: bool,
) -> Result<Vec<u8>> {
    let mut state = StageStateHeader::new(WireMessageKind::StateExport);
    if full_state {
        state.flags |= state_flags::FULL_STATE;
    }
    let message = StageWireMessage {
        kind: WireMessageKind::StateExport,
        pos_start: 0,
        token_count: 0,
        state,
        request_id: 2,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message)?;
    let reply =
        read_stage_message(&mut *stream, activation_width).context("receive state export")?;
    if reply.kind != WireMessageKind::StateImport {
        bail!("expected state import payload, got {:?}", reply.kind);
    }
    if reply.raw_bytes.is_empty() {
        bail!("state export returned an empty payload");
    }
    Ok(reply.raw_bytes)
}

fn import_state_over_binary(
    stream: &mut std::net::TcpStream,
    state_bytes: &[u8],
    full_state: bool,
) -> Result<()> {
    let token_count = i32::try_from(state_bytes.len()).context("state payload is too large")?;
    let mut state = StageStateHeader::new(WireMessageKind::StateImport);
    if full_state {
        state.flags |= state_flags::FULL_STATE;
    }
    let message = StageWireMessage {
        kind: WireMessageKind::StateImport,
        pos_start: 0,
        token_count,
        state,
        request_id: 3,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: state_bytes.to_vec(),
    };
    write_stage_message(&mut *stream, &message)?;
    let reply = recv_reply(&mut *stream).context("receive state import ACK")?;
    if reply.kind != WireReplyKind::Ack {
        bail!("expected state import ACK, got {:?}", reply.kind);
    }
    Ok(())
}

fn decode_for_state_handoff(
    stream: &mut std::net::TcpStream,
    token_id: i32,
    pos_start: usize,
    input: Option<&ActivationFrame>,
    activation_width: i32,
) -> Result<i32> {
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd);
    state.prompt_token_count = i32::try_from(pos_start).context("prompt position exceeds i32")?;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = input
        .map(|frame| frame.desc.producer_stage_index)
        .unwrap_or(-1);
    state.flags |= activation_state_flags_optional(input);
    let activation = encode_handoff_activation(state.activation_codec, input, 1, activation_width)?;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: i32::try_from(pos_start).context("decode position exceeds i32")?,
        token_count: 1,
        state,
        request_id: 4,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        positions: Vec::new(),
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut *stream, &message)?;
    let reply = recv_reply(&mut *stream).context("receive decode reply")?;
    if reply.kind != WireReplyKind::PredictedToken {
        bail!("expected decode predicted token, got {:?}", reply.kind);
    }
    Ok(reply.predicted)
}

fn encode_handoff_activation(
    codec: skippy_protocol::StageActivationCodec,
    input: Option<&ActivationFrame>,
    token_count: i32,
    activation_width: i32,
) -> Result<Vec<u8>> {
    let Some(input) = input else {
        return Ok(Vec::new());
    };
    skippy_protocol::binary::encode_activation_payload_with_state_flags(
        codec,
        token_count,
        activation_width,
        &input.payload,
        activation_state_flags(input),
    )
    .context("failed to encode state handoff input activation")
}

fn activation_state_flags(frame: &ActivationFrame) -> i32 {
    activation_state_flags_from_frame_flags(frame.desc.flags)
}

fn activation_state_flags_optional(frame: Option<&ActivationFrame>) -> i32 {
    frame.map(activation_state_flags).unwrap_or(0)
}
