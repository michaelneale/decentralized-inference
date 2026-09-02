use super::ConnectionWorkerControl;
use super::async_forwarder::AsyncForwarder;
use super::control_messages::{
    handle_generation_control, handle_prefix_cache_control, handle_session_control, handle_stop,
    handle_verify_retirement,
};
use super::message_receive::{next_connection_session_id, spawn_message_reader};
use super::reply::reply_window_for_message;
use super::reply::send_stage_reply;
use super::session_lifecycle::{align_session_to_target, record_session_auto_align};
use super::session_tracker::{
    ConnectionSessionOwnership, ConnectionSessionTracker, combine_connection_and_cleanup_results,
    release_tracked_connection_sessions,
};
use super::stale_discard::StaleDiscardRegistry;
use super::summary::BinaryMessageObservation;
use super::summary::BinaryRequestSummary;
use super::telemetry::UpstreamReplyWriteSpan;
use super::telemetry::{
    BinaryMessageTiming, emit_binary_message_received, emit_binary_message_timing,
    emit_upstream_reply_write_span, insert_runtime_session_stats, record_prefill_edge_transport,
    record_prefill_stage_compute, record_verify_window_timing,
};
use crate::binary_transport::BinaryStageExecutionOptions;
use crate::binary_transport::WireCondition;
use crate::binary_transport::binary_kv::BinaryKvLookupResult;
use crate::binary_transport::binary_kv::accumulate_shared_prefill_tokens;
use crate::binary_transport::binary_kv::add_binary_record_stats;
use crate::binary_transport::binary_kv::emit_binary_proactive_eviction;
use crate::binary_transport::binary_kv::maybe_lookup_binary_prefill;
use crate::binary_transport::binary_kv::maybe_record_binary_full_prefill;
use crate::binary_transport::direct_return;
use crate::binary_transport::direct_return::PredictionReturnSinks;
use crate::binary_transport::forwarded_stage_message_timed;
use crate::binary_transport::kv_eviction::binary_proactive_eviction_plan;
use crate::binary_transport::kv_eviction::evict_binary_resident_prefix_for_decode;
use crate::binary_transport::prefill_execution::{
    executable_prefill_start, suffix_activation_frame,
};
use crate::binary_transport::run_binary_stage_message;
use crate::binary_transport::stage_execution::binary_message_attrs;
use crate::binary_transport::stage_execution::binary_message_session_id;
use crate::binary_transport::stage_execution::decode_record_tokens_sideband;
use crate::binary_transport::stage_execution::elapsed_ms;
use crate::binary_transport::stage_execution::empty_activation_frame;
use crate::binary_transport::stage_execution::input_activation_frame;
use crate::binary_transport::stage_execution::is_decode_frame_batch_candidate;
use crate::binary_transport::stage_execution::nanos_delta_ms;
use crate::binary_transport::stage_execution::runtime_sampling_config;
use crate::binary_transport::stage_execution::split_native_mtp_reply;
use crate::binary_transport::stage_execution::stage_mask;
use crate::binary_transport::stage_execution::token_sideband_or_fill;
use crate::binary_transport::stage_output_activation_capacity;
use crate::binary_transport::write_stage_message_conditioned;
use crate::frontend::iteration_scheduler::IterationScheduler;
use crate::kv_integration::KvStageIntegration;
use crate::telemetry::Telemetry;
use crate::telemetry::now_unix_nanos;
use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_protocol::binary::StageReply;
use skippy_protocol::binary::StageReplyStats;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireMessageKind;
use skippy_protocol::binary::WireReplyKind;
use skippy_protocol::binary::recv_reply;
use skippy_protocol::binary::send_reply_ack;
use skippy_protocol::binary::send_reply_ack_with_stats;
use skippy_protocol::{StageConfig, StageTopology};
use std::collections::BTreeMap;
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
pub(super) fn handle_binary_connection(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    iteration_scheduler: &IterationScheduler,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    upstream: &mut TcpStream,
    downstream: Option<TcpStream>,
    input_activation_width: i32,
    output_activation_width: i32,
    max_inflight: usize,
    reply_credit_limit: Option<usize>,
    async_prefill_forward: bool,
    downstream_wire_condition: WireCondition,
    downstream_connect_timeout_secs: u64,
    native_mtp_enabled: bool,
    prediction_return_sinks: &PredictionReturnSinks,
    session_ownership: Arc<ConnectionSessionOwnership>,
    worker_control: Arc<ConnectionWorkerControl>,
    first_message: StageWireMessage,
) -> Result<()> {
    let mut session_tracker =
        ConnectionSessionTracker::new(next_connection_session_id(), session_ownership);
    let result = handle_binary_connection_messages(
        config,
        topology,
        iteration_scheduler,
        kv,
        telemetry,
        upstream,
        downstream,
        input_activation_width,
        output_activation_width,
        max_inflight,
        reply_credit_limit,
        async_prefill_forward,
        downstream_wire_condition,
        downstream_connect_timeout_secs,
        native_mtp_enabled,
        prediction_return_sinks,
        worker_control,
        first_message,
        &mut session_tracker,
    );
    let cleanup_result = release_tracked_connection_sessions(
        config,
        iteration_scheduler,
        kv,
        telemetry,
        &mut session_tracker,
    );
    combine_connection_and_cleanup_results(result, cleanup_result)
}

#[allow(clippy::too_many_arguments)]
fn handle_binary_connection_messages(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    iteration_scheduler: &IterationScheduler,
    kv: Option<&Arc<KvStageIntegration>>,
    telemetry: &Telemetry,
    upstream: &mut TcpStream,
    mut downstream: Option<TcpStream>,
    input_activation_width: i32,
    output_activation_width: i32,
    max_inflight: usize,
    reply_credit_limit: Option<usize>,
    async_prefill_forward: bool,
    downstream_wire_condition: WireCondition,
    downstream_connect_timeout_secs: u64,
    native_mtp_enabled: bool,
    prediction_return_sinks: &PredictionReturnSinks,
    worker_control: Arc<ConnectionWorkerControl>,
    first_message: StageWireMessage,
    session_tracker: &mut ConnectionSessionTracker,
) -> Result<()> {
    let connection_session_id = session_tracker.connection_id;
    let max_deferred_prefill_replies =
        reply_credit_limit.unwrap_or_else(|| max_inflight.saturating_sub(1));
    let mut pending_prefill_replies = 0usize;
    let mut pending_reply_stats = StageReplyStats::default();
    let mut request_summary = BinaryRequestSummary::default();
    let mut prediction_return_streams: BTreeMap<(u64, u64), TcpStream> = BTreeMap::new();
    let mut next_message = Some(first_message);
    let discard_registry = Arc::new(StaleDiscardRegistry::default());
    let inbound_reader = spawn_message_reader(
        upstream,
        input_activation_width,
        max_inflight.max(1),
        discard_registry.clone(),
        worker_control,
    )?;
    let mut async_forwarder = if async_prefill_forward || max_inflight > 1 {
        downstream
            .as_ref()
            .map(|downstream| {
                AsyncForwarder::new(downstream, telemetry.clone(), max_inflight.max(1))
            })
            .transpose()
            .context("create async activation forwarder")?
    } else {
        None
    };

    loop {
        let recv_start_unix_nanos = now_unix_nanos() as u64;
        let recv_started = Instant::now();
        let Some(mut message) = inbound_reader.next(
            next_message.take(),
            pending_prefill_replies,
            request_summary.message_count,
        )?
        else {
            return Ok(());
        };
        let recv_end_unix_nanos = now_unix_nanos() as u64;
        let recv_read_ms = elapsed_ms(recv_started);
        let message_start_unix_nanos = now_unix_nanos() as u64;
        let message_started = Instant::now();
        let session_id = binary_message_session_id(connection_session_id, &message);
        let session_key = session_id.to_string();
        session_tracker.touch(&session_key);
        emit_binary_message_received(
            telemetry,
            config,
            session_id,
            &message,
            recv_start_unix_nanos,
            recv_end_unix_nanos,
            recv_read_ms,
        );

        if message.kind == WireMessageKind::Stop {
            handle_stop(
                config,
                iteration_scheduler,
                kv,
                telemetry,
                upstream,
                downstream.as_mut(),
                downstream_wire_condition,
                &message,
                &session_key,
                session_id,
                pending_prefill_replies,
                &mut pending_reply_stats,
                &mut request_summary,
                async_forwarder.as_mut(),
                session_tracker,
                &mut prediction_return_streams,
                prediction_return_sinks,
            )?;
            continue;
        }

        if message.kind.is_stale_window_discard() {
            // The reader thread already recorded the range; middle stages
            // forward it so downstream stages can skip their buffered stale
            // windows too. No reply is expected.
            if let Some(downstream) = downstream.as_mut() {
                if let Some(forwarder) = async_forwarder.as_mut() {
                    forwarder
                        .send(message, downstream_wire_condition, BTreeMap::new())
                        .context("forward stale window discard downstream")?;
                } else {
                    write_stage_message_conditioned(
                        &mut *downstream,
                        &message,
                        downstream_wire_condition,
                    )
                    .context("forward stale window discard downstream")?;
                }
            }
            continue;
        }

        if message.kind.is_verify_retirement() {
            handle_verify_retirement(
                iteration_scheduler,
                downstream.as_mut(),
                downstream_wire_condition,
                &message,
                &session_key,
                async_forwarder.as_mut(),
            )?;
            continue;
        }

        if message.kind.is_session_control() {
            handle_session_control(
                iteration_scheduler,
                upstream,
                downstream.as_mut(),
                downstream_wire_condition,
                &message,
                &session_key,
                &mut pending_prefill_replies,
                &mut pending_reply_stats,
                async_forwarder.as_mut(),
            )?;
            continue;
        }

        if message.kind.is_generation_control() {
            handle_generation_control(
                config,
                topology,
                iteration_scheduler,
                upstream,
                downstream.as_mut(),
                downstream_wire_condition,
                downstream_connect_timeout_secs,
                &message,
                &session_key,
                &mut pending_prefill_replies,
                &mut pending_reply_stats,
                async_forwarder.as_mut(),
                prediction_return_sinks,
                &mut prediction_return_streams,
            )?;
            continue;
        }

        if message.kind.is_prefix_cache_control() {
            handle_prefix_cache_control(
                config,
                topology,
                iteration_scheduler,
                kv,
                telemetry,
                upstream,
                downstream.as_mut(),
                downstream_wire_condition,
                downstream_connect_timeout_secs,
                output_activation_width,
                native_mtp_enabled,
                message,
                &session_key,
                session_id,
                &mut pending_prefill_replies,
                &mut pending_reply_stats,
                async_forwarder.as_mut(),
                prediction_return_sinks,
                &mut prediction_return_streams,
            )?;
            continue;
        }

        if message.kind == WireMessageKind::StateImport {
            bail!("binary state import is no longer supported by the skippy runtime ABI");
        }

        if message.kind == WireMessageKind::StateExport {
            bail!("binary state export is no longer supported by the skippy runtime ABI");
        }

        if !message.state.matches_kind(message.kind) {
            bail!("binary stage state does not match message kind");
        }

        if message.kind == WireMessageKind::VerifyWindow
            && downstream.is_none()
            && discard_registry.is_discarded(
                message.request_id,
                message.session_id,
                message.state.seq_id,
            )
        {
            // A discard raced ahead of this buffered stale window: answer with
            // an empty prediction set instead of executing it. The driver's
            // stale drain only uses the window id for FIFO bookkeeping.
            let reply = StageReply {
                kind: WireReplyKind::PredictedTokens,
                predicted: message.state.current_token,
                predicted_tokens: Vec::new(),
                native_mtp_draft: None,
                window: reply_window_for_message(&message),
                stats: StageReplyStats::default(),
            };
            if let Some(return_stream) =
                prediction_return_streams.get_mut(&(message.request_id, message.session_id))
            {
                direct_return::send_direct_prediction_return(return_stream, reply)
                    .context("send discarded verify window reply")?;
            } else {
                send_stage_reply(&mut *upstream, reply)
                    .context("send discarded verify window reply")?;
            }
            request_summary.message_count += 1;
            continue;
        }

        let requires_predicted = message.kind.requires_predicted_reply();
        let early_prefill_ack = message.kind.is_prefill() && !requires_predicted;
        let mut upstream_reply_start_unix_nanos = None;
        let mut upstream_reply_end_unix_nanos = None;
        let mut early_reply_ms = 0.0;
        if early_prefill_ack {
            let reply_start_unix_nanos = now_unix_nanos() as u64;
            upstream_reply_start_unix_nanos = Some(reply_start_unix_nanos);
            let reply_started = Instant::now();
            send_reply_ack(&mut *upstream).context("early prefill ack")?;
            upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
            early_reply_ms = elapsed_ms(reply_started);
        }

        let token_ids = token_sideband_or_fill(&message)?;
        // Session alignment must run on the scheduler before any lookup or
        // compute touches the runtime, but it does not deserve its own
        // scheduler round-trip: it rides inside the first runtime closure
        // that executes for this message (lookup when the prefix cache will
        // run one, otherwise the compute dispatch, and the batched decode
        // path aligns internally from its target token count).
        let align_target = message.authoritative_session_position();
        let mut session_auto_align_count = 0usize;
        let mut session_auto_align_ms = 0.0;
        let mut session_auto_align_trimmed_tokens = 0u64;
        if message.kind.is_prefill()
            && let Some(cache) = kv
        {
            accumulate_shared_prefill_tokens(
                &cache.split_prefill_tokens,
                &session_key,
                message.pos_start.max(0) as usize,
                &token_ids,
            );
        }
        let mut message_reply_stats = StageReplyStats::default();
        let lookup_needed = kv.is_some_and(|cache| cache.should_lookup())
            && message.kind.is_prefill()
            && !message.kind.requires_predicted_reply()
            && !token_ids.is_empty();
        let lookup_result = if lookup_needed {
            let lookup_config = config.clone();
            let lookup_kv = kv.cloned();
            let lookup_telemetry = telemetry.clone();
            let lookup_session_key = session_key.clone();
            let lookup_message = message.clone();
            let lookup_token_ids = token_ids.clone();
            let (auto_align, lookup_result) = iteration_scheduler
                .execute_runtime("binary-prefix-lookup", move |runtime| {
                    let auto_align = align_session_to_target(
                        runtime,
                        &lookup_telemetry,
                        &lookup_session_key,
                        align_target,
                    )
                    .map_err(|error| openai_frontend::OpenAiError::backend(format!("{error:#}")))?;
                    Ok((
                        auto_align,
                        maybe_lookup_binary_prefill(
                            &lookup_config,
                            runtime,
                            lookup_kv.as_ref(),
                            &lookup_telemetry,
                            &lookup_session_key,
                            &lookup_message,
                            &lookup_token_ids,
                            output_activation_width,
                        ),
                    ))
                })
                .map_err(|error| anyhow::anyhow!(format!("{error:#}")))?;
            session_auto_align_count = auto_align.count;
            session_auto_align_ms = auto_align.elapsed_ms;
            session_auto_align_trimmed_tokens = auto_align.trimmed_tokens;
            lookup_result
        } else {
            BinaryKvLookupResult::default()
        };
        message_reply_stats.merge(lookup_result.stats);
        let restored_prefill =
            lookup_result.restored_tokens >= token_ids.len() && !token_ids.is_empty();
        let executable_prefill_start = executable_prefill_start(
            message.kind,
            lookup_result.restored_tokens,
            token_ids.len(),
            config.layer_start,
            config.downstream.is_some(),
        );
        let executable_token_ids = &token_ids[executable_prefill_start..];
        if executable_prefill_start > 0 && message.positions.len() == token_ids.len() {
            message.positions.drain(..executable_prefill_start);
        }
        let compute_start_unix_nanos: u64;
        let compute_end_unix_nanos: u64;
        let mut input_activation_decode_ms = 0.0;
        let mut runtime_lock_wait_ms = 0.0;
        let mut runtime_lock_hold_ms = 0.0;
        let mut runtime_lock_acquires = 0usize;
        let mut runtime_sessions_before = None;
        let mut runtime_sessions_after = None;
        let mut decode_batch_size = 1usize;
        let mut decode_batch_wait_ms = 0.0;
        let input_activation_bytes = message.activation.len();
        let mut proactive_eviction = None;
        let (predicted_token, mut predicted_tokens, mut output, native_mtp_draft, compute_ms) =
            if restored_prefill {
                let now = now_unix_nanos() as u64;
                compute_start_unix_nanos = now;
                compute_end_unix_nanos = now;
                (
                    message.state.current_token,
                    Vec::new(),
                    lookup_result
                        .activation
                        .clone()
                        .unwrap_or_else(|| empty_activation_frame(config, &message)),
                    None,
                    0.0,
                )
            } else {
                let input_decode_started = Instant::now();
                let input = suffix_activation_frame(
                    input_activation_frame(config, topology, &mut message)?,
                    executable_prefill_start,
                )?;
                input_activation_decode_ms = if input_activation_bytes == 0 {
                    0.0
                } else {
                    elapsed_ms(input_decode_started)
                };
                compute_start_unix_nanos = now_unix_nanos() as u64;
                let compute_started = Instant::now();
                let use_decode_frame_batch =
                    is_decode_frame_batch_candidate(config, &message, executable_token_ids);
                let result = if use_decode_frame_batch {
                    let token_id = executable_token_ids
                        .first()
                        .copied()
                        .unwrap_or(message.state.current_token);
                    let sampling = runtime_sampling_config(message.sampling.as_ref());
                    let target_token_count =
                        message.authoritative_session_position().ok_or_else(|| {
                            anyhow::anyhow!("batched decode frame has no authoritative position")
                        })?;
                    let outcome = iteration_scheduler
                        .execute_frame_iteration(
                            &session_key,
                            target_token_count,
                            &[token_id],
                            &[],
                            sampling.as_ref(),
                            input,
                            true,
                        )
                        .map_err(|error| anyhow::anyhow!(format!("{error:#}")))
                        .context("execute batched binary decode frame")?;
                    runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
                    runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
                    runtime_lock_acquires = 1;
                    decode_batch_size = outcome.batch_size;
                    decode_batch_wait_ms = outcome.batch_wait_ms;
                    // The batched frame aligns the session internally from its
                    // target token count, so surface that trim on the same
                    // counter and event as the explicit alignment paths. It is
                    // fused into the decode, so it reports no standalone elapsed.
                    if let Some(align) = outcome.session_alignment.as_ref() {
                        let observation =
                            record_session_auto_align(telemetry, &session_key, align, 0.0);
                        session_auto_align_count = observation.count;
                        session_auto_align_ms = observation.elapsed_ms;
                        session_auto_align_trimmed_tokens = observation.trimmed_tokens;
                    }
                    (outcome.predicted, Vec::new(), outcome.output, None)
                } else {
                    let eviction_plan = binary_proactive_eviction_plan(
                        message.kind,
                        restored_prefill,
                        executable_token_ids.len(),
                        (message.state.prompt_token_count.max(0) as usize)
                            .saturating_sub(message.pos_start.max(0) as usize),
                    );
                    let output_capacity = stage_output_activation_capacity(
                        config,
                        message.token_count,
                        output_activation_width,
                    )?;
                    let sample_prefill_final =
                        message.kind == WireMessageKind::PrefillFinalEmbd && downstream.is_none();
                    let scheduler_session_key = session_key.clone();
                    let scheduler_message = message.clone();
                    let scheduler_token_ids = executable_token_ids.to_vec();
                    let scheduler_kv = kv.cloned();
                    let scheduler_telemetry = telemetry.clone();
                    let align_in_compute = !lookup_needed;
                    let collect_session_stats = telemetry.is_debug_enabled();
                    let outcome = iteration_scheduler
                        .execute_runtime_timed("binary-stage-execute", move |runtime| {
                            let auto_align = if align_in_compute {
                                align_session_to_target(
                                    runtime,
                                    &scheduler_telemetry,
                                    &scheduler_session_key,
                                    align_target,
                                )
                                .map_err(|error| {
                                    openai_frontend::OpenAiError::backend(format!("{error:#}"))
                                })?
                            } else {
                                Default::default()
                            };
                            let sessions_before =
                                collect_session_stats.then(|| runtime.session_stats());
                            let eviction = if eviction_plan.required {
                                Some(
                                    evict_binary_resident_prefix_for_decode(
                                        runtime,
                                        scheduler_kv.as_ref(),
                                        &scheduler_session_key,
                                        eviction_plan,
                                    )
                                    .map_err(|error| {
                                        openai_frontend::OpenAiError::backend(format!("{error:#}"))
                                    })?,
                                )
                            } else {
                                None
                            };
                            let result = run_binary_stage_message(
                                runtime,
                                &scheduler_session_key,
                                &scheduler_message,
                                &scheduler_token_ids,
                                input.as_ref(),
                                BinaryStageExecutionOptions::new(
                                    sample_prefill_final,
                                    output_capacity,
                                    native_mtp_enabled,
                                ),
                            )
                            .map_err(|error| {
                                openai_frontend::OpenAiError::backend(format!("{error:#}"))
                            })?;
                            let sessions_after =
                                collect_session_stats.then(|| runtime.session_stats());
                            Ok((
                                auto_align,
                                sessions_before,
                                sessions_after,
                                eviction,
                                result,
                            ))
                        })
                        .map_err(|error| anyhow::anyhow!(format!("{error:#}")))
                        .with_context(|| {
                            format!(
                                "execute scheduler-owned binary stage message \
                                 kind={:?} pos_start={} token_count={} tokens={} \
                                 executable_tokens={} activation_bytes={}",
                                message.kind,
                                message.pos_start,
                                message.token_count,
                                message.tokens.len(),
                                executable_token_ids.len(),
                                input_activation_bytes,
                            )
                        })?;
                    runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
                    runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
                    runtime_lock_acquires = 1;
                    let (auto_align, sessions_before, sessions_after, eviction, result) =
                        outcome.value;
                    if align_in_compute {
                        session_auto_align_count = auto_align.count;
                        session_auto_align_ms = auto_align.elapsed_ms;
                        session_auto_align_trimmed_tokens = auto_align.trimmed_tokens;
                    }
                    runtime_sessions_before = sessions_before;
                    runtime_sessions_after = sessions_after;
                    proactive_eviction = eviction;
                    result
                };
                let compute_ms = elapsed_ms(compute_started);
                compute_end_unix_nanos = now_unix_nanos() as u64;
                (result.0, result.1, result.2, result.3, compute_ms)
            };
        record_prefill_stage_compute(&mut message_reply_stats, config, &message, compute_ms);
        if telemetry.is_debug_enabled() {
            let mut decode_attrs = binary_message_attrs(config, session_id, &message);
            decode_attrs.insert(
                "skippy.output_activation_bytes".to_string(),
                json!(output.payload.len()),
            );
            decode_attrs.insert("skippy.compute_ms".to_string(), json!(compute_ms));
            decode_attrs.insert(
                "llama_stage.input_activation_decode_ms".to_string(),
                json!(input_activation_decode_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(runtime_lock_acquires),
            );
            decode_attrs.insert(
                "llama_stage.decode_batch_size".to_string(),
                json!(decode_batch_size),
            );
            decode_attrs.insert(
                "llama_stage.decode_batch_wait_ms".to_string(),
                json!(decode_batch_wait_ms),
            );
            if let Some(stats) = runtime_sessions_before.as_ref() {
                insert_runtime_session_stats(
                    &mut decode_attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = runtime_sessions_after.as_ref() {
                insert_runtime_session_stats(
                    &mut decode_attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            if let Some(eviction) = proactive_eviction.as_ref() {
                eviction.insert_attrs(&mut decode_attrs);
            }
            decode_attrs.insert(
                "skippy.kv.restored_prefill".to_string(),
                json!(restored_prefill),
            );
            decode_attrs.insert(
                "llama_stage.compute_start_unix_nanos".to_string(),
                json!(compute_start_unix_nanos),
            );
            decode_attrs.insert(
                "llama_stage.compute_end_unix_nanos".to_string(),
                json!(compute_end_unix_nanos),
            );
            telemetry.emit_debug_span(
                "stage.binary_llama_decode",
                decode_attrs,
                compute_start_unix_nanos,
                compute_end_unix_nanos,
            );
        }
        if let Some(eviction) = proactive_eviction {
            emit_binary_proactive_eviction(telemetry, &eviction);
        }

        let accumulated_prefill_tokens = kv.and_then(|cache| {
            cache
                .split_prefill_tokens
                .lock()
                .expect("split prefill token lock poisoned")
                .get(&session_key)
                .cloned()
        });
        if message.kind.is_prefill() && !restored_prefill {
            let record_config = config.clone();
            let record_kv = kv.cloned();
            let record_telemetry = telemetry.clone();
            let record_session_key = session_key.clone();
            let record_message = message.clone();
            let record_accumulated_prefill_tokens = accumulated_prefill_tokens.clone();
            let record_token_ids = token_ids.clone();
            let restored_tokens = lookup_result.restored_tokens as u64;
            let (record, returned_output) = iteration_scheduler
                .execute_runtime("binary-prefill-record", move |runtime| {
                    let record = super::prefill_recording::record_completed_prefill(
                        &record_config,
                        runtime,
                        record_kv.as_ref(),
                        &record_telemetry,
                        &record_session_key,
                        &record_message,
                        record_accumulated_prefill_tokens.as_deref(),
                        &record_token_ids,
                        restored_tokens,
                        output_activation_width,
                        &output,
                    );
                    Ok((record, output))
                })
                .map_err(|error| anyhow::anyhow!(format!("{error:#}")))?;
            output = returned_output;
            if record.recorded_pages > 0 {
                message_reply_stats.kv_recorded_pages += record.recorded_pages as i64;
                message_reply_stats.kv_record_stage_mask |= stage_mask(config.stage_index);
            }
            if record.recorded_activations > 0 {
                message_reply_stats.kv_recorded_bytes = message_reply_stats
                    .kv_recorded_bytes
                    .saturating_add(record.recorded_activation_bytes as i64);
            }
        }

        let completed_prompt_tokens = decode_record_tokens_sideband(&message).or_else(|| {
            message
                .kind
                .requires_predicted_reply()
                .then_some(accumulated_prefill_tokens.as_deref())
                .flatten()
        });
        // The record helper's own no-record gates are runtime-free; checking
        // them here keeps the recorder's scheduler round-trip (and the
        // message/token clones feeding it) off the per-token path whenever
        // the prefix cache would not record anyway.
        let full_prefill_record_needed = kv.is_some_and(|cache| cache.should_record())
            && completed_prompt_tokens.is_some_and(|tokens| !tokens.is_empty());
        if full_prefill_record_needed && let Some(full_prompt_tokens) = completed_prompt_tokens {
            let record_config = config.clone();
            let record_kv = kv.cloned();
            let record_telemetry = telemetry.clone();
            let record_session_key = session_key.clone();
            let record_message = message.clone();
            let record_tokens = full_prompt_tokens.to_vec();
            let record = iteration_scheduler
                .execute_runtime("binary-full-prefill-record", move |runtime| {
                    Ok(maybe_record_binary_full_prefill(
                        &record_config,
                        runtime,
                        record_kv.as_ref(),
                        &record_telemetry,
                        &record_session_key,
                        &record_message,
                        &record_tokens,
                    ))
                })
                .map_err(|error| anyhow::anyhow!(format!("{error:#}")))?;
            add_binary_record_stats(&mut message_reply_stats, config, &record);
        }

        let mut forward_write_ms = 0.0;
        let mut forward_activation_encode_ms = 0.0;
        let mut forward_activation_bytes = 0usize;
        let mut downstream_wait_ms = 0.0;
        let mut upstream_reply_ms = early_reply_ms;
        let mut forward_write_start_unix_nanos = None;
        let mut forward_write_end_unix_nanos = None;
        let mut downstream_wait_start_unix_nanos = None;
        let mut downstream_wait_end_unix_nanos = None;
        let mut forward_mode = "none";
        let pending_prefill_replies_before = pending_prefill_replies;
        let mut credit_wait_count = 0usize;
        let mut deferred_prefill_replies_drained = 0usize;

        if let Some(downstream) = downstream.as_mut() {
            if output.payload.is_empty() {
                bail!("stage has downstream but produced an empty activation payload");
            }
            let forwarded =
                forwarded_stage_message_timed(config, &message, &output, output_activation_width)?;
            forward_activation_encode_ms += forwarded.activation_encode_ms;
            forward_activation_bytes = forwarded.message.activation.len();
            let mut downstream_write_attrs = BTreeMap::new();
            if telemetry.is_debug_enabled() {
                downstream_write_attrs = binary_message_attrs(config, session_id, &message);
                downstream_write_attrs.insert(
                    "llama_stage.forward_activation_bytes".to_string(),
                    json!(forward_activation_bytes),
                );
                downstream_write_attrs.insert(
                    "llama_stage.activation_encode_ms".to_string(),
                    json!(forwarded.activation_encode_ms),
                );
                downstream_write_attrs.insert(
                    "llama_stage.output_activation_bytes".to_string(),
                    json!(output.payload.len()),
                );
            }
            let forward_start_unix_nanos = now_unix_nanos() as u64;
            forward_write_start_unix_nanos = Some(forward_start_unix_nanos);
            let forward_started = Instant::now();
            let async_verify_forward =
                message.kind == WireMessageKind::VerifyWindow && max_inflight > 1;
            if (async_prefill_forward && early_prefill_ack && max_deferred_prefill_replies > 0)
                || async_verify_forward
            {
                forward_mode = "async_enqueue";
                if telemetry.is_debug_enabled() {
                    downstream_write_attrs.insert(
                        "llama_stage.forward_mode".to_string(),
                        json!("async_writer"),
                    );
                }
                let forwarder = async_forwarder
                    .as_mut()
                    .context("missing async activation forwarder")?;
                forwarder
                    .send(
                        forwarded.message,
                        downstream_wire_condition,
                        downstream_write_attrs,
                    )
                    .context("queue async activation frame downstream")?;
            } else {
                forward_mode = "sync_write";
                if telemetry.is_debug_enabled() {
                    downstream_write_attrs
                        .insert("llama_stage.forward_mode".to_string(), json!("sync_write"));
                }
                if let Some(forwarder) = async_forwarder.as_mut() {
                    forwarder.flush().context("flush async activation frames")?;
                }
                let downstream_write_start_unix_nanos = now_unix_nanos() as u64;
                let downstream_write_started = Instant::now();
                write_stage_message_conditioned(
                    &mut *downstream,
                    &forwarded.message,
                    downstream_wire_condition,
                )
                .context("forward activation frame downstream")?;
                let downstream_write_end_unix_nanos = now_unix_nanos() as u64;
                if telemetry.is_debug_enabled() {
                    downstream_write_attrs.insert(
                        "llama_stage.forward_write_ms".to_string(),
                        json!(elapsed_ms(downstream_write_started)),
                    );
                    telemetry.emit_debug_span(
                        "stage.binary_downstream_write",
                        downstream_write_attrs,
                        downstream_write_start_unix_nanos,
                        downstream_write_end_unix_nanos,
                    );
                }
            }
            forward_write_end_unix_nanos = Some(now_unix_nanos() as u64);
            forward_write_ms += elapsed_ms(forward_started);

            if requires_predicted {
                while pending_prefill_replies > 0 {
                    let wait_start_unix_nanos = now_unix_nanos() as u64;
                    downstream_wait_start_unix_nanos.get_or_insert(wait_start_unix_nanos);
                    let wait_started = Instant::now();
                    let reply = recv_reply(&mut *downstream)
                        .context("drain deferred downstream prefill reply")?;
                    downstream_wait_ms += elapsed_ms(wait_started);
                    if reply.kind != WireReplyKind::Ack {
                        bail!("expected deferred downstream ACK");
                    }
                    pending_reply_stats.merge(reply.stats);
                    pending_prefill_replies -= 1;
                    deferred_prefill_replies_drained += 1;
                }
            } else if max_deferred_prefill_replies == 0 {
                let wait_start_unix_nanos = now_unix_nanos() as u64;
                downstream_wait_start_unix_nanos.get_or_insert(wait_start_unix_nanos);
                let wait_started = Instant::now();
                let reply = recv_reply(&mut *downstream).context("downstream ACK")?;
                downstream_wait_end_unix_nanos = Some(now_unix_nanos() as u64);
                downstream_wait_ms += elapsed_ms(wait_started);
                if reply.kind != WireReplyKind::Ack {
                    bail!("expected downstream ACK");
                }
                message_reply_stats.merge(reply.stats);
                if !early_prefill_ack {
                    let reply_start_unix_nanos = now_unix_nanos() as u64;
                    upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
                    let reply_started = Instant::now();
                    send_reply_ack_with_stats(&mut *upstream, message_reply_stats)
                        .context("relay ACK")?;
                    upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
                    let reply_write_ms = elapsed_ms(reply_started);
                    upstream_reply_ms += reply_write_ms;
                    emit_upstream_reply_write_span(
                        telemetry,
                        config,
                        session_id,
                        &message,
                        UpstreamReplyWriteSpan {
                            reply_kind: WireReplyKind::Ack,
                            predicted_token_count: 0,
                            start_unix_nanos: reply_start_unix_nanos,
                            end_unix_nanos: upstream_reply_end_unix_nanos
                                .unwrap_or(reply_start_unix_nanos),
                            write_ms: reply_write_ms,
                        },
                    );
                } else {
                    pending_reply_stats.merge(message_reply_stats);
                }
            } else {
                while pending_prefill_replies >= max_deferred_prefill_replies {
                    credit_wait_count += 1;
                    let wait_start_unix_nanos = now_unix_nanos() as u64;
                    downstream_wait_start_unix_nanos.get_or_insert(wait_start_unix_nanos);
                    let wait_started = Instant::now();
                    let reply =
                        recv_reply(&mut *downstream).context("bounded-credit downstream ACK")?;
                    downstream_wait_end_unix_nanos = Some(now_unix_nanos() as u64);
                    downstream_wait_ms += elapsed_ms(wait_started);
                    if reply.kind != WireReplyKind::Ack {
                        bail!("expected downstream ACK while enforcing credit");
                    }
                    pending_reply_stats.merge(reply.stats);
                    pending_prefill_replies -= 1;
                    deferred_prefill_replies_drained += 1;
                }
                pending_prefill_replies += 1;
                if !early_prefill_ack {
                    let reply_start_unix_nanos = now_unix_nanos() as u64;
                    upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
                    let reply_started = Instant::now();
                    send_reply_ack_with_stats(&mut *upstream, message_reply_stats)
                        .context("deferred relay ACK")?;
                    upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
                    let reply_write_ms = elapsed_ms(reply_started);
                    upstream_reply_ms += reply_write_ms;
                    emit_upstream_reply_write_span(
                        telemetry,
                        config,
                        session_id,
                        &message,
                        UpstreamReplyWriteSpan {
                            reply_kind: WireReplyKind::Ack,
                            predicted_token_count: 0,
                            start_unix_nanos: reply_start_unix_nanos,
                            end_unix_nanos: upstream_reply_end_unix_nanos
                                .unwrap_or(reply_start_unix_nanos),
                            write_ms: reply_write_ms,
                        },
                    );
                } else {
                    pending_reply_stats.merge(message_reply_stats);
                }
            }
        } else if requires_predicted {
            record_prefill_edge_transport(
                &mut message_reply_stats,
                config,
                &message,
                forward_write_ms,
                downstream_wait_ms,
                forward_activation_bytes,
            );
            message_reply_stats.merge(pending_reply_stats);
            pending_reply_stats = StageReplyStats::default();
            record_verify_window_timing(
                &mut message_reply_stats,
                &message,
                compute_ms,
                forward_write_ms,
                downstream_wait_ms,
            );
            let reply_kind = if message.kind == WireMessageKind::VerifyWindow {
                WireReplyKind::PredictedTokens
            } else {
                WireReplyKind::PredictedToken
            };
            let native_mtp_draft = match native_mtp_draft {
                Some(draft) => Some(draft),
                None => split_native_mtp_reply(&message, &mut predicted_tokens)?,
            };
            let predicted_token_count = if message.kind == WireMessageKind::VerifyWindow {
                predicted_tokens.len()
            } else {
                predicted_tokens.len().max(1)
            };
            let reply_start_unix_nanos = now_unix_nanos() as u64;
            upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
            let reply_started = Instant::now();
            let reply_window = reply_window_for_message(&message);
            let reply = StageReply {
                kind: reply_kind,
                predicted: predicted_token,
                predicted_tokens,
                native_mtp_draft,
                window: reply_window,
                stats: message_reply_stats,
            };
            if let Some(return_stream) =
                prediction_return_streams.get_mut(&(message.request_id, message.session_id))
            {
                direct_return::send_direct_prediction_return(return_stream, reply)
                    .context("send direct predicted reply")?;
            } else {
                send_stage_reply(&mut *upstream, reply)
                    .context("send fallback upstream predicted reply")?;
            }
            upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
            let reply_write_ms = elapsed_ms(reply_started);
            upstream_reply_ms += reply_write_ms;
            emit_upstream_reply_write_span(
                telemetry,
                config,
                session_id,
                &message,
                UpstreamReplyWriteSpan {
                    reply_kind,
                    predicted_token_count,
                    start_unix_nanos: reply_start_unix_nanos,
                    end_unix_nanos: upstream_reply_end_unix_nanos.unwrap_or(reply_start_unix_nanos),
                    write_ms: reply_write_ms,
                },
            );
        } else if !early_prefill_ack {
            record_prefill_edge_transport(
                &mut message_reply_stats,
                config,
                &message,
                forward_write_ms,
                downstream_wait_ms,
                forward_activation_bytes,
            );
            let reply_start_unix_nanos = now_unix_nanos() as u64;
            upstream_reply_start_unix_nanos.get_or_insert(reply_start_unix_nanos);
            let reply_started = Instant::now();
            send_reply_ack_with_stats(&mut *upstream, message_reply_stats).context("send ACK")?;
            upstream_reply_end_unix_nanos = Some(now_unix_nanos() as u64);
            let reply_write_ms = elapsed_ms(reply_started);
            upstream_reply_ms += reply_write_ms;
            emit_upstream_reply_write_span(
                telemetry,
                config,
                session_id,
                &message,
                UpstreamReplyWriteSpan {
                    reply_kind: WireReplyKind::Ack,
                    predicted_token_count: 0,
                    start_unix_nanos: reply_start_unix_nanos,
                    end_unix_nanos: upstream_reply_end_unix_nanos.unwrap_or(reply_start_unix_nanos),
                    write_ms: reply_write_ms,
                },
            );
        } else {
            record_prefill_edge_transport(
                &mut message_reply_stats,
                config,
                &message,
                forward_write_ms,
                downstream_wait_ms,
                forward_activation_bytes,
            );
            pending_reply_stats.merge(message_reply_stats);
        }

        let message_end_unix_nanos = now_unix_nanos() as u64;
        let message_elapsed_ms = elapsed_ms(message_started);
        let verify_window_pre_compute_ms = if message.kind == WireMessageKind::VerifyWindow {
            nanos_delta_ms(message_start_unix_nanos, compute_start_unix_nanos)
        } else {
            0.0
        };
        let verify_window_post_compute_ms = if message.kind == WireMessageKind::VerifyWindow {
            nanos_delta_ms(compute_end_unix_nanos, message_end_unix_nanos)
        } else {
            0.0
        };
        let verify_window_pre_reply_ms = if message.kind == WireMessageKind::VerifyWindow {
            upstream_reply_start_unix_nanos
                .map(|reply_start| nanos_delta_ms(compute_end_unix_nanos, reply_start))
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let verify_window_after_reply_ms = if message.kind == WireMessageKind::VerifyWindow {
            upstream_reply_end_unix_nanos
                .map(|reply_end| nanos_delta_ms(reply_end, message_end_unix_nanos))
                .unwrap_or(0.0)
        } else {
            0.0
        };
        request_summary.observe(BinaryMessageObservation {
            config,
            message: &message,
            reply_stats: message_reply_stats,
            compute_ms,
            forward_write_ms,
            downstream_wait_ms,
            upstream_reply_ms,
            message_elapsed_ms,
            input_activation_bytes,
            output_activation_bytes: output.payload.len(),
            input_activation_decode_ms,
            forward_activation_encode_ms,
            runtime_lock_hold_ms,
            prefill_credit_limit: max_deferred_prefill_replies,
            pending_prefill_replies_before,
            pending_prefill_replies_after: pending_prefill_replies,
            credit_wait_count,
            deferred_prefill_replies_drained,
            session_auto_align_count,
            session_auto_align_ms,
            session_auto_align_trimmed_tokens,
            verify_window_pre_compute_ms,
            verify_window_post_compute_ms,
            verify_window_pre_reply_ms,
            verify_window_after_reply_ms,
            upstream_message_wait_ms: recv_read_ms,
        });

        emit_binary_message_timing(
            telemetry,
            config,
            session_id,
            &message,
            BinaryMessageTiming {
                message_start_unix_nanos,
                message_end_unix_nanos,
                compute_start_unix_nanos,
                compute_end_unix_nanos,
                forward_write_start_unix_nanos,
                forward_write_end_unix_nanos,
                downstream_wait_start_unix_nanos,
                downstream_wait_end_unix_nanos,
                upstream_reply_start_unix_nanos,
                upstream_reply_end_unix_nanos,
                compute_ms,
                recv_read_ms,
                input_activation_decode_ms,
                runtime_lock_wait_ms,
                runtime_lock_hold_ms,
                runtime_lock_acquires,
                runtime_sessions_before: runtime_sessions_before.as_ref(),
                runtime_sessions_after: runtime_sessions_after.as_ref(),
                forward_write_ms,
                forward_activation_encode_ms,
                downstream_wait_ms,
                upstream_reply_ms,
                forward_mode,
                message_elapsed_ms,
                input_activation_bytes,
                output_activation_bytes: output.payload.len(),
                max_deferred_prefill_replies,
                pending_prefill_replies_before,
                pending_prefill_replies_after: pending_prefill_replies,
                credit_wait_count,
                deferred_prefill_replies_drained,
            },
        );
    }
}
