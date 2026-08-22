use std::collections::VecDeque;

mod fused_decode;
mod lifecycle;
mod prefix_restore;
mod speculative_policy;

use super::*;
use crate::binary_transport::{
    AsyncForwarder, BinaryStageExecutionOptions, forwarded_stage_message,
    forwarded_stage_message_timed, run_binary_stage_message, write_stage_message_conditioned,
};
use crate::frontend::embedded_execution::{StaleWindowDiscard, VerifyRetirement};
use crate::frontend::request::wire_sampling_config;
use crate::frontend::speculative::{
    OpenAiSpeculativeStats, classify_verify_window_with_threshold, propose_configured_ngram_tokens,
    verify_checkpoint_no_longer_needed, verify_inputs_for_proposals,
};
use crate::frontend::util::{ms_to_us, openai_backend_error, openai_io_error, saturating_u32};
use crate::frontend::wire_messages::{
    OpenAiPrefillChunk, ReusableDecodeMessage, ReusableDecodeMessageArgs, VerifyWindowMessageArgs,
    embedded_prefill_message, embedded_verify_window_message, generation_config_message,
};
use crate::frontend::{
    NativeMtpDecodeCounters, NativeMtpDecodeOptions, NativeMtpDraft, NativeMtpDraftOrigin,
    NativeMtpVerifier,
    generation::{
        EmbeddedStageZeroGeneration, GenerationCacheStats, PhaseTimer, StageOpenAiBackend,
        TokenControl, attrs_insert_prefill_chunk_policy, decode_token_phase,
    },
    prefill::{
        PrefillChunkObservation, drain_embedded_prefill_replies, drain_one_embedded_prefill_reply,
        representative_prefill_compute_sample,
    },
};
use crate::telemetry::now_unix_nanos;
use lifecycle::{
    DirectPredictionReturnPath, EmbeddedDecodeSummary, PipelinedCompositeWindow, can_seed_pipeline,
    compose_target_predictions, decode_uses_context_sideband, direct_prediction_return_path,
    mark_epoch_stale, open_upstream_prediction_return, pipelined_window_layout,
    queued_active_tokens, refill_pipeline_ngram_candidates, speculation_after_prefix_restore,
    stale_window_id_range,
};
use openai_frontend::{OpenAiError, OpenAiResult};
use prefix_restore::EmbeddedPrefixRestore;
use serde_json::json;
use skippy_protocol::binary::{StageReplyStats, WireReplyKind, recv_reply};

fn prefill_chunk_end(
    pos_start: usize,
    chunk_size: usize,
    prefill_token_count: usize,
    exact_checkpoint_boundary: Option<usize>,
) -> usize {
    let mut end = pos_start
        .saturating_add(chunk_size)
        .min(prefill_token_count);
    if let Some(boundary) = exact_checkpoint_boundary
        && pos_start < boundary
    {
        end = end.min(boundary);
    }
    end
}

impl StageOpenAiBackend {
    pub(super) fn generate_embedded_stage_zero_tokens(
        &self,
        request: EmbeddedStageZeroGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        if request.config.downstream.is_none() {
            return self.generate_embedded_request_locally(request, on_token);
        }

        let wire_sampling = wire_sampling_config(request.sampling);
        let session_id = request.ids.session_id;
        let request_id = request.ids.request_id;
        let session_key = session_id.to_string();
        let lane_pool = request
            .lane_pool
            .as_ref()
            .ok_or_else(|| OpenAiError::backend("embedded stage 0 has no downstream lane pool"))?;
        let mut lane = lane_pool.checkout(request.ids)?;
        let direct_prediction_return_opened = open_upstream_prediction_return(&request);
        let mut cache_stats = GenerationCacheStats::default();

        let result = (|| {
            let downstream = &mut lane.stream;
            let prefill_token_count = request.prompt_token_ids.len().saturating_sub(1);
            let prefill_timer = PhaseTimer::start();
            let mut prefill_chunks = 0usize;
            let mut prefill_min_chunk_size = usize::MAX;
            let mut prefill_max_chunk_size = 0usize;
            let mut prefill_stage0_compute_ms = 0.0;
            let mut prefill_stage0_compute_max_ms = 0.0_f64;
            let mut prefill_stage0_compute_max_tokens = 0usize;
            let mut prefill_runtime_lock_wait_ms = 0.0;
            let mut prefill_runtime_lock_wait_max_ms = 0.0_f64;
            let mut prefill_runtime_lock_hold_ms = 0.0;
            let mut prefill_runtime_lock_hold_max_ms = 0.0_f64;
            let mut prefill_runtime_lock_acquires = 0usize;
            let mut prefill_runtime_sessions_before = None;
            let mut prefill_runtime_sessions_after = None;
            let mut prefill_forward_write_ms = 0.0;
            let mut prefill_forward_write_max_ms = 0.0_f64;
            let mut prefill_output_activation_bytes = 0usize;
            let mut prefill_forward_activation_bytes = 0usize;
            let mut prefill_downstream_wait_ms = 0.0;
            let mut prefill_downstream_wait_max_ms = 0.0_f64;
            let mut pending_prefill_replies = 0usize;
            let mut prefill_credit_wait_count = 0usize;
            let mut prefill_deferred_replies_drained = 0usize;
            let mut prefill_pending_replies_max = 0usize;
            let mut prefill_stage0_cache_hits = 0usize;
            let mut prefill_stage0_cache_misses = 0usize;
            let mut prefill_stage0_cache_errors = 0usize;
            let mut prefill_chain_cache_restored = false;
            let mut prefill_chain_restored_tokens = 0usize;
            let mut prefill_chain_cache_stats = StageReplyStats::default();
            let mut prefill_stage0_full_recorded = false;
            let mut fused_first_decode = None;
            let mut prefill_planner = request.prefill_chunk_policy.planner();
            if let Some(estimate) = lane_pool.prefill_transport_estimate() {
                prefill_planner.calibrate_slowest_stage_rate(estimate.slowest_compute_ms_per_token);
            }
            if let Some(seed) = lane_pool.prefill_transport_seed() {
                prefill_planner.observe(seed);
            }
            if prefill_token_count > 0 {
                let prefill_tokens = &request.prompt_token_ids[..prefill_token_count];
                if self.kv.is_some() {
                    cache_stats.status = if request.native_mtp_enabled {
                        "bypass_native_mtp"
                    } else {
                        "miss"
                    };
                }
                let EmbeddedPrefixRestore {
                    allowed: prefix_restore_allowed,
                    chain_cache_restored,
                    chain_restored_tokens,
                    chain_cache_stats,
                    fused_first_decode: restored_first_decode,
                } = self.restore_embedded_prefix(
                    &request,
                    &session_key,
                    downstream,
                    prefill_tokens,
                    wire_sampling.clone(),
                    &mut cache_stats,
                )?;
                prefill_chain_cache_restored = chain_cache_restored;
                prefill_chain_restored_tokens = chain_restored_tokens;
                prefill_chain_cache_stats = chain_cache_stats;
                fused_first_decode = restored_first_decode;
                let mut pos_start = prefill_chain_restored_tokens.min(prefill_tokens.len());
                let exact_checkpoint_boundary =
                    self.embedded_exact_checkpoint_boundary(&request, prefill_tokens, pos_start);
                let mut chunk_index = 0usize;
                while pos_start < prefill_tokens.len() {
                    if request
                        .cancellation
                        .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                    {
                        drain_embedded_prefill_replies(
                            downstream,
                            &mut pending_prefill_replies,
                            &mut prefill_chain_cache_stats,
                        )?;
                        return Ok(());
                    }
                    let chunk_size =
                        prefill_planner.chunk_size_for(chunk_index, prefill_token_count);
                    let end = prefill_chunk_end(
                        pos_start,
                        chunk_size,
                        prefill_tokens.len(),
                        exact_checkpoint_boundary,
                    );
                    let chunk = &prefill_tokens[pos_start..end];
                    prefill_min_chunk_size = prefill_min_chunk_size.min(chunk.len());
                    prefill_max_chunk_size = prefill_max_chunk_size.max(chunk.len());
                    let message = embedded_prefill_message(OpenAiPrefillChunk {
                        seq_id: chunk_index,
                        pos_start,
                        prefill_token_count,
                        tokens: chunk,
                        request_id,
                        session_id,
                    })?;
                    let stage0_timer = PhaseTimer::start();
                    let pending_prefill_replies_before = pending_prefill_replies;
                    let mut output = if prefix_restore_allowed {
                        self.restore_embedded_stage0_prefill(
                            &session_key,
                            request.ids,
                            pos_start as u64,
                            chunk,
                            request.activation_width,
                        )?
                    } else {
                        None
                    };
                    if output.is_some() {
                        prefill_stage0_cache_hits += 1;
                    } else {
                        prefill_stage0_cache_misses += usize::from(pos_start == 0);
                    }
                    let output = match output.take() {
                        Some(output) => output,
                        None => {
                            self.evict_embedded_stage0_resident_prefix(
                                &session_key,
                                request.ids,
                                Some(
                                    u64::try_from(prefill_tokens.len().saturating_sub(pos_start))
                                        .unwrap_or(u64::MAX),
                                ),
                            )?;
                            let scheduler_session_key = session_key.clone();
                            let scheduler_message = message.clone();
                            let scheduler_chunk = chunk.to_vec();
                            let native_mtp_enabled = request.native_mtp_enabled;
                            let native_mtp_max_tokens =
                                request.speculative.native_mtp.max_draft_tokens;
                            let outcome = self.iteration_scheduler.execute_runtime_timed(
                                "embedded-prefill",
                                move |runtime| {
                                    let sessions_before = runtime.session_stats();
                                    let output = run_binary_stage_message(
                                        runtime,
                                        &scheduler_session_key,
                                        &scheduler_message,
                                        &scheduler_chunk,
                                        None,
                                        BinaryStageExecutionOptions::new(
                                            false,
                                            0,
                                            native_mtp_enabled,
                                        )
                                        .with_native_mtp_max_tokens(native_mtp_max_tokens),
                                    )
                                    .map_err(openai_backend_error)?
                                    .2;
                                    let sessions_after = runtime.session_stats();
                                    Ok((sessions_before, sessions_after, output))
                                },
                            )?;
                            let (sessions_before, sessions_after, output) = outcome.value;
                            prefill_runtime_lock_wait_ms += outcome.runtime_lock_wait_ms;
                            prefill_runtime_lock_wait_max_ms =
                                prefill_runtime_lock_wait_max_ms.max(outcome.runtime_lock_wait_ms);
                            prefill_runtime_lock_acquires += 1;
                            prefill_runtime_sessions_before.get_or_insert(sessions_before);
                            prefill_runtime_sessions_after = Some(sessions_after);
                            prefill_runtime_lock_hold_ms += outcome.runtime_lock_hold_ms;
                            prefill_runtime_lock_hold_max_ms =
                                prefill_runtime_lock_hold_max_ms.max(outcome.runtime_lock_hold_ms);
                            output
                        }
                    };
                    if let Err(error) = self.record_embedded_stage0_prefill(
                        &session_key,
                        request.ids,
                        pos_start as u64,
                        chunk,
                        request.activation_width,
                        &output,
                    ) {
                        prefill_stage0_cache_errors += 1;
                        let mut attrs = self.openai_attrs(request.ids);
                        attrs.insert(
                            "skippy.kv.decision".to_string(),
                            json!("stage0_record_error"),
                        );
                        attrs.insert(
                            "skippy.kv.error_class".to_string(),
                            json!(crate::kv_integration::telemetry_error_class_from_message(
                                &error.to_string()
                            )),
                        );
                        self.telemetry
                            .emit("stage.openai_kv_record_decision", attrs);
                    }
                    if exact_checkpoint_boundary == Some(end) {
                        prefill_stage0_full_recorded |= self
                            .record_embedded_stage0_exact_checkpoint(
                                &session_key,
                                request.ids,
                                &prefill_tokens[..end],
                            )?;
                    }
                    let chunk_stage0_compute_ms = stage0_timer.elapsed_ms();
                    prefill_stage0_compute_ms += chunk_stage0_compute_ms;
                    (
                        prefill_stage0_compute_max_ms,
                        prefill_stage0_compute_max_tokens,
                    ) = representative_prefill_compute_sample(
                        prefill_stage0_compute_max_ms,
                        prefill_stage0_compute_max_tokens,
                        chunk_stage0_compute_ms,
                        chunk.len(),
                    );
                    let forwarded = forwarded_stage_message(
                        request.config,
                        &message,
                        &output,
                        request.activation_width,
                    )
                    .map_err(openai_backend_error)?;
                    prefill_output_activation_bytes =
                        prefill_output_activation_bytes.saturating_add(output.payload.len());
                    prefill_forward_activation_bytes =
                        prefill_forward_activation_bytes.saturating_add(forwarded.activation.len());
                    let write_timer = PhaseTimer::start();
                    write_stage_message_conditioned(
                        &mut *downstream,
                        &forwarded,
                        request.downstream_wire_condition,
                    )
                    .map_err(openai_io_error)?;
                    let chunk_forward_write_ms = write_timer.elapsed_ms();
                    prefill_forward_write_ms += chunk_forward_write_ms;
                    prefill_forward_write_max_ms =
                        prefill_forward_write_max_ms.max(chunk_forward_write_ms);
                    let mut chunk_downstream_wait_ms = 0.0;
                    let mut chunk_deferred_replies_drained = 0usize;
                    let mut chunk_credit_wait_count = 0usize;
                    if request.prefill_reply_credit_limit == 0 {
                        let wait_timer = PhaseTimer::start();
                        let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
                        chunk_downstream_wait_ms = wait_timer.elapsed_ms();
                        if reply.kind != WireReplyKind::Ack {
                            return Err(OpenAiError::backend(format!(
                                "expected prefill ACK from downstream, got {:?}",
                                reply.kind
                            )));
                        }
                        prefill_chain_cache_stats.merge(reply.stats);
                    } else {
                        while pending_prefill_replies >= request.prefill_reply_credit_limit {
                            prefill_credit_wait_count = prefill_credit_wait_count.saturating_add(1);
                            chunk_credit_wait_count = chunk_credit_wait_count.saturating_add(1);
                            let drained = drain_one_embedded_prefill_reply(
                                downstream,
                                &mut pending_prefill_replies,
                                &mut prefill_chain_cache_stats,
                            )?;
                            prefill_deferred_replies_drained = prefill_deferred_replies_drained
                                .saturating_add(drained.drained_replies);
                            chunk_deferred_replies_drained = chunk_deferred_replies_drained
                                .saturating_add(drained.drained_replies);
                            chunk_downstream_wait_ms += drained.downstream_wait_ms;
                        }
                        pending_prefill_replies = pending_prefill_replies.saturating_add(1);
                        prefill_pending_replies_max =
                            prefill_pending_replies_max.max(pending_prefill_replies);
                    }
                    prefill_downstream_wait_ms += chunk_downstream_wait_ms;
                    prefill_downstream_wait_max_ms =
                        prefill_downstream_wait_max_ms.max(chunk_downstream_wait_ms);
                    prefill_planner.observe(PrefillChunkObservation {
                        compute_ms: chunk_stage0_compute_ms,
                        forward_write_ms: chunk_forward_write_ms,
                        downstream_wait_ms: chunk_downstream_wait_ms,
                    });
                    let mut chunk_attrs = self.openai_attrs(request.ids);
                    chunk_attrs
                        .insert("llama_stage.message_kind".to_string(), json!("PrefillEmbd"));
                    chunk_attrs.insert("llama_stage.seq_id".to_string(), json!(chunk_index));
                    chunk_attrs.insert("llama_stage.pos_start".to_string(), json!(pos_start));
                    chunk_attrs.insert("llama_stage.token_count".to_string(), json!(chunk.len()));
                    chunk_attrs.insert(
                        "llama_stage.stage0_compute_ms".to_string(),
                        json!(chunk_stage0_compute_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.forward_write_ms".to_string(),
                        json!(chunk_forward_write_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.downstream_wait_ms".to_string(),
                        json!(chunk_downstream_wait_ms),
                    );
                    chunk_attrs.insert(
                        "llama_stage.output_activation_bytes".to_string(),
                        json!(output.payload.len()),
                    );
                    chunk_attrs.insert(
                        "llama_stage.forward_activation_bytes".to_string(),
                        json!(forwarded.activation.len()),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_credit_limit".to_string(),
                        json!(request.prefill_reply_credit_limit),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_pending_replies_before".to_string(),
                        json!(pending_prefill_replies_before),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_pending_replies_after".to_string(),
                        json!(pending_prefill_replies),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_credit_wait_count".to_string(),
                        json!(chunk_credit_wait_count),
                    );
                    chunk_attrs.insert(
                        "skippy.prefill_deferred_replies_drained".to_string(),
                        json!(chunk_deferred_replies_drained),
                    );
                    self.telemetry.emit_debug_span(
                        "stage.openai_prefill_chunk",
                        chunk_attrs,
                        stage0_timer.start_unix_nanos,
                        now_unix_nanos() as u64,
                    );
                    prefill_chunks += 1;
                    pos_start = end;
                    chunk_index += 1;
                }
                let drained = drain_embedded_prefill_replies(
                    downstream,
                    &mut pending_prefill_replies,
                    &mut prefill_chain_cache_stats,
                )?;
                prefill_deferred_replies_drained =
                    prefill_deferred_replies_drained.saturating_add(drained.drained_replies);
                prefill_downstream_wait_ms += drained.downstream_wait_ms;
                prefill_downstream_wait_max_ms =
                    prefill_downstream_wait_max_ms.max(drained.downstream_wait_max_ms);
                if !prefill_chain_cache_restored && !prefill_stage0_full_recorded {
                    prefill_stage0_full_recorded = self.record_embedded_stage0_full_prefill(
                        &session_key,
                        request.ids,
                        prefill_tokens,
                    )?;
                }
            }
            let mut prefill_attrs = self.openai_attrs(request.ids);
            prefill_attrs.insert(
                "llama_stage.prefill_token_count".to_string(),
                json!(prefill_token_count),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_chunk_count".to_string(),
                json!(prefill_chunks),
            );
            attrs_insert_prefill_chunk_policy(
                &mut prefill_attrs,
                request.prefill_chunk_policy,
                prefill_min_chunk_size,
                prefill_max_chunk_size,
            );
            prefill_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(prefill_stage0_compute_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(prefill_runtime_lock_wait_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(prefill_runtime_lock_wait_max_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(prefill_runtime_lock_hold_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(prefill_runtime_lock_hold_max_ms),
            );
            prefill_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(prefill_runtime_lock_acquires),
            );
            if let Some(stats) = prefill_runtime_sessions_before.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut prefill_attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = prefill_runtime_sessions_after.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut prefill_attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            prefill_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(prefill_forward_write_ms),
            );
            prefill_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(prefill_output_activation_bytes),
            );
            prefill_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(prefill_forward_activation_bytes),
            );
            prefill_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(prefill_downstream_wait_ms),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_write_us_max".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_write_us_max),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_wait_us_max".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_wait_us_max),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_total_us_max".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_total_us_max),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_stage_index".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_stage_index),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_edge_observation_count".to_string(),
                json!(prefill_chain_cache_stats.prefill_edge_observation_count),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_compute_us_at_slowest_rate".to_string(),
                json!(prefill_chain_cache_stats.prefill_compute_us_at_slowest_rate),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_compute_stage_index".to_string(),
                json!(prefill_chain_cache_stats.prefill_compute_stage_index),
            );
            prefill_attrs.insert(
                "llama_stage.prefill_compute_observation_count".to_string(),
                json!(prefill_chain_cache_stats.prefill_compute_observation_count),
            );
            if let Some(estimate) = lane_pool.prefill_transport_estimate() {
                prefill_attrs.insert(
                    "llama_stage.prefill_bottleneck_compute_ms".to_string(),
                    json!(estimate.slowest_compute_ms),
                );
                prefill_attrs.insert(
                    "llama_stage.prefill_bottleneck_compute_ms_per_token".to_string(),
                    json!(estimate.slowest_compute_ms_per_token),
                );
                prefill_attrs.insert(
                    "llama_stage.prefill_bottleneck_stage_index".to_string(),
                    json!(estimate.bottleneck_stage_index),
                );
                prefill_attrs.insert(
                    "llama_stage.prefill_bottleneck_token_count".to_string(),
                    json!(estimate.bottleneck_token_count),
                );
            }
            prefill_attrs.insert(
                "skippy.prefill_credit_limit".to_string(),
                json!(request.prefill_reply_credit_limit),
            );
            prefill_attrs.insert(
                "skippy.prefill_pending_replies_max".to_string(),
                json!(prefill_pending_replies_max),
            );
            prefill_attrs.insert(
                "skippy.prefill_pending_replies_after".to_string(),
                json!(pending_prefill_replies),
            );
            prefill_attrs.insert(
                "skippy.prefill_credit_wait_count".to_string(),
                json!(prefill_credit_wait_count),
            );
            prefill_attrs.insert(
                "skippy.prefill_deferred_replies_drained".to_string(),
                json!(prefill_deferred_replies_drained),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_hits".to_string(),
                json!(prefill_stage0_cache_hits),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_misses".to_string(),
                json!(prefill_stage0_cache_misses),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_cache_errors".to_string(),
                json!(prefill_stage0_cache_errors),
            );
            prefill_attrs.insert(
                "skippy.kv.stage0_full_recorded".to_string(),
                json!(prefill_stage0_full_recorded),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_restored".to_string(),
                json!(prefill_chain_cache_restored),
            );
            prefill_attrs.insert(
                "skippy.kv.matched_prefix_tokens".to_string(),
                json!(prefill_chain_restored_tokens),
            );
            prefill_attrs.insert(
                "skippy.kv.suffix_prefill_tokens".to_string(),
                json!(prefill_token_count.saturating_sub(prefill_chain_restored_tokens)),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_hits".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_hits),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_misses".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_misses),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_errors".to_string(),
                json!(prefill_chain_cache_stats.kv_lookup_errors),
            );
            prefill_attrs.insert(
                "skippy.kv.chain_cache_hit_stage_mask".to_string(),
                json!(prefill_chain_cache_stats.kv_hit_stage_mask),
            );
            super::prefix_cache::insert_chain_prefix_cache_savings_attrs(
                &mut prefill_attrs,
                super::prefix_cache::chain_prefix_cache_savings(
                    &prefill_chain_cache_stats,
                    prefill_chain_restored_tokens,
                    request.activation_width,
                ),
            );
            cache_stats.prompt_ms = prefill_timer.elapsed_ms();
            self.emit_openai_phase("stage.openai_prefill", prefill_timer, prefill_attrs);

            let message = generation_config_message(
                request_id,
                session_id,
                request.prompt_token_ids.len(),
                wire_sampling.clone(),
                request.chat_sampling_metadata,
            )?;
            write_stage_message_conditioned(
                &mut *downstream,
                &message,
                request.downstream_wire_condition,
            )
            .map_err(openai_io_error)?;
            let reply = recv_reply(&mut *downstream).map_err(openai_io_error)?;
            if reply.kind != WireReplyKind::Ack {
                return Err(OpenAiError::backend(format!(
                    "expected generation config ACK from downstream, got {:?}",
                    reply.kind
                )));
            }
            prefill_chain_cache_stats.merge(reply.stats);
            lane_pool.observe_prefill_transport(
                &prefill_chain_cache_stats,
                prefill_stage0_compute_max_ms,
                prefill_stage0_compute_max_tokens,
                prefill_forward_write_max_ms,
                prefill_downstream_wait_max_ms,
            );
            if let Some(estimate) = lane_pool.prefill_transport_estimate() {
                let mut calibration_attrs = self.openai_attrs(request.ids);
                calibration_attrs.insert(
                    "llama_stage.prefill_bottleneck_compute_ms".to_string(),
                    json!(estimate.slowest_compute_ms),
                );
                calibration_attrs.insert(
                    "llama_stage.prefill_bottleneck_compute_ms_per_token".to_string(),
                    json!(estimate.slowest_compute_ms_per_token),
                );
                calibration_attrs.insert(
                    "llama_stage.prefill_bottleneck_stage_index".to_string(),
                    json!(estimate.bottleneck_stage_index),
                );
                calibration_attrs.insert(
                    "llama_stage.prefill_bottleneck_token_count".to_string(),
                    json!(estimate.bottleneck_token_count),
                );
                calibration_attrs.insert(
                    "llama_stage.prefill_transport_write_to_compute".to_string(),
                    json!(estimate.write_to_compute),
                );
                calibration_attrs.insert(
                    "llama_stage.prefill_transport_wait_to_compute".to_string(),
                    json!(estimate.wait_to_compute),
                );
                calibration_attrs.insert(
                    "llama_stage.prefill_calibration_observations".to_string(),
                    json!(estimate.observations),
                );
                self.telemetry
                    .emit("stage.openai_prefill_calibration", calibration_attrs);
            }
            if prefill_chain_cache_restored {
                self.evict_embedded_stage0_resident_prefix(&session_key, request.ids, None)?;
            }

            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut decode_stage0_compute_ms = 0.0;
            let mut decode_runtime_lock_wait_ms = 0.0;
            let mut decode_runtime_lock_wait_max_ms = 0.0_f64;
            let mut decode_runtime_lock_hold_ms = 0.0;
            let mut decode_runtime_lock_hold_max_ms = 0.0_f64;
            let mut decode_runtime_lock_acquires = 0usize;
            let mut decode_batch_size_max = 1usize;
            let mut decode_batch_wait_ms = 0.0;
            let mut decode_forward_write_ms = 0.0;
            let mut decode_forward_activation_encode_ms = 0.0;
            let mut decode_output_activation_bytes = 0usize;
            let mut decode_forward_activation_bytes = 0usize;
            let mut decode_downstream_wait_ms = 0.0;
            let mut current = *request
                .prompt_token_ids
                .last()
                .expect("checked non-empty prompt");
            let mut context_tokens = request.prompt_token_ids.to_vec();
            let mut exact_replay_tokens = Vec::new();
            let mut decode_message = ReusableDecodeMessage::new(ReusableDecodeMessageArgs {
                request_id,
                session_id,
                prompt_token_count: request.prompt_token_ids.len(),
                base_pos_start: prefill_token_count,
                sampling: wire_sampling.clone(),
                sideband_capacity: skippy_protocol::binary::MAX_STAGE_SIDEBAND_VALUES,
            })?;
            let mut native_mtp = NativeMtpVerifier::default();
            let effective_speculative =
                speculation_after_prefix_restore(request.speculative, prefill_chain_cache_restored);
            if request.speculative.ngram.is_some() && effective_speculative.ngram.is_none() {
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.spec.bypass_reason".to_string(),
                    json!("distributed_prefix_restored"),
                );
                self.telemetry
                    .emit("stage.openai_speculation_bypass", attrs);
            }
            let effective_speculative = effective_speculative.as_ref();
            let native_mtp_options = NativeMtpDecodeOptions::from_config(effective_speculative);
            let mut native_mtp_counters = NativeMtpDecodeCounters::default();
            let mut native_mtp_reject_cooldown_remaining = 0usize;
            let mut native_mtp_suppress_cooldown_drafts_remaining = 0usize;
            let mut ngram_sidecar_controller =
                NgramSidecarController::new(native_mtp_options.ngram_max_proposal_tokens);
            let fused_reached_stop =
                fused_decode::apply_fused_first_decode(fused_decode::FusedFirstDecodeContext {
                    backend: self,
                    request: &request,
                    fused_first_decode: &mut fused_first_decode,
                    native_mtp: &mut native_mtp,
                    current: &mut current,
                    decoded_tokens: &mut decoded_tokens,
                    exact_replay_tokens: &mut exact_replay_tokens,
                    context_tokens: &mut context_tokens,
                    decode_stage0_compute_ms: &mut decode_stage0_compute_ms,
                    decode_runtime_lock_wait_ms: &mut decode_runtime_lock_wait_ms,
                    decode_runtime_lock_wait_max_ms: &mut decode_runtime_lock_wait_max_ms,
                    decode_runtime_lock_hold_ms: &mut decode_runtime_lock_hold_ms,
                    decode_runtime_lock_hold_max_ms: &mut decode_runtime_lock_hold_max_ms,
                    decode_runtime_lock_acquires: &mut decode_runtime_lock_acquires,
                    decode_forward_activation_encode_ms: &mut decode_forward_activation_encode_ms,
                    decode_output_activation_bytes: &mut decode_output_activation_bytes,
                    decode_forward_activation_bytes: &mut decode_forward_activation_bytes,
                    decode_forward_write_ms: &mut decode_forward_write_ms,
                    decode_downstream_wait_ms: &mut decode_downstream_wait_ms,
                    on_token: &mut on_token,
                })?;
            let mut cached_ngram_proposer =
                HistoryNgramProposer::from_config(effective_speculative)?;
            let max_speculative_window = request.speculative_window.max(1);
            let mut adaptive_window = if request.adaptive_speculative_window {
                max_speculative_window.min(4)
            } else {
                max_speculative_window
            };
            let mut speculative_stats = OpenAiSpeculativeStats {
                adaptive_window_start: adaptive_window,
                adaptive_window_final: adaptive_window,
                adaptive_window_max: max_speculative_window,
                adaptive_window_min: if request.draft.is_some()
                    || effective_speculative.ngram.is_some()
                {
                    adaptive_window
                } else {
                    0
                },
                adaptive_window_max_seen: adaptive_window,
                adaptive_window_enabled: request.adaptive_speculative_window,
                ..OpenAiSpeculativeStats::default()
            };
            let mut draft_guard = match request.draft.as_ref() {
                Some(draft) if request.speculative_window > 0 => {
                    let draft_reset_timer = PhaseTimer::start();
                    let mut draft = draft
                        .lock()
                        .map_err(|_| OpenAiError::backend("draft model lock poisoned"))?;
                    draft
                        .reset_to_context(&context_tokens)
                        .map_err(openai_backend_error)?;
                    speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                    let mut attrs = self.openai_attrs(request.ids);
                    attrs.insert(
                        "llama_stage.draft_model_path".to_string(),
                        json!(draft.path.display().to_string()),
                    );
                    attrs.insert(
                        "llama_stage.speculative_window".to_string(),
                        json!(draft.window),
                    );
                    attrs.insert(
                        "llama_stage.adaptive_speculative_window".to_string(),
                        json!(request.adaptive_speculative_window),
                    );
                    self.emit_openai_phase("stage.openai_draft_reset", draft_reset_timer, attrs);
                    Some(draft)
                }
                _ => None,
            };
            let mut verify_window_scheduler = VerifyWindowScheduler::new(
                if effective_speculative.verify_window.runahead_max_tokens > 0 {
                    VerifyWindowPipelineConfig::with_runahead(
                        effective_speculative.verify_window.runahead_max_tokens,
                    )
                } else {
                    VerifyWindowPipelineConfig::new(
                        effective_speculative.verify_window.pipeline_depth,
                    )
                },
            );
            let composite_sidecar_enabled =
                native_mtp_options.ngram_hybrid && draft_guard.is_none();
            // A standalone N-gram plan (no native MTP, no draft model) can drive
            // the same verify-window pipeline; the single-window native-MTP path
            // stays composite-only, so standalone drafting still falls back to the
            // serial block at depth 1.
            let standalone_ngram_pipelining = !request.native_mtp_enabled
                && effective_speculative.ngram.is_some()
                && draft_guard.is_none();
            let native_mtp_verify_windows_enabled =
                (request.native_mtp_enabled || composite_sidecar_enabled) && draft_guard.is_none();
            let pipelined_decode_enabled = (composite_sidecar_enabled
                || standalone_ngram_pipelining)
                && verify_window_scheduler.depth() > 1;
            let mut verify_window_forwarder = None;
            if let Some(direct_return_path) = direct_prediction_return_path(
                native_mtp_verify_windows_enabled || pipelined_decode_enabled,
                request.prediction_return.is_some(),
                direct_prediction_return_opened,
            )? {
                // The final stage first consumes the upstream-opened sink, then
                // falls back to opening the v11 direct-return stream back to the
                // registered stage-0 receiver. A transient failure opening the
                // preferred sink must not fail an otherwise healthy request.
                verify_window_scheduler.mark_direct_prediction_return(matches!(
                    direct_return_path,
                    DirectPredictionReturnPath::UpstreamOpened
                ));
            }
            let mut pipelined_windows = VecDeque::new();
            let mut pipelined = None;
            let mut pipelined_boundary_prediction = None;
            let mut pipeline_epoch = 0u64;
            let mut composite_proposal_buffer = None;
            let mut adaptive_verify_window = AdaptiveVerifyWindow::new(native_mtp_options);
            while decoded_tokens < request.max_tokens as usize {
                let decode_step = saturating_u32(decoded_tokens);
                if fused_reached_stop {
                    break;
                }
                if decoded_tokens >= request.max_tokens as usize {
                    break;
                }
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                let token_timer = PhaseTimer::start();
                let effective_native_mtp_options = native_mtp_options;
                let native_mtp_remaining =
                    (request.max_tokens as usize).saturating_sub(decoded_tokens);
                let mut pipeline_seed = None;
                if pipelined_decode_enabled
                    && pipelined.is_none()
                    && can_seed_pipeline(&pipelined_windows)
                {
                    let pending = (request.native_mtp_enabled
                        && native_mtp_reject_cooldown_remaining == 0)
                        .then(|| native_mtp.take_pending_draft())
                        .flatten();
                    let native_mtp_origin = pending.as_ref().map(|draft| draft.origin);
                    let native_mtp_tokens = pending
                        .as_ref()
                        .map(|draft| {
                            draft
                                .tokens
                                .iter()
                                .copied()
                                .take(native_mtp_options.max_draft_tokens)
                                .take(native_mtp_remaining.saturating_sub(1))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let native_mtp_tokens =
                        if native_mtp_tokens.len() >= native_mtp_options.min_draft_tokens {
                            native_mtp_tokens.as_slice()
                        } else {
                            &[]
                        };
                    let proposal = CompositeProposalProvider::from_options(native_mtp_options)
                        .propose_with_ngram_extension(
                            native_mtp_tokens,
                            &context_tokens,
                            native_mtp_remaining,
                            ngram_sidecar_controller.extension_limit(
                                native_mtp_tokens,
                                native_mtp_remaining.saturating_sub(native_mtp_tokens.len()),
                            ),
                            cached_ngram_proposer.as_mut(),
                        )?;
                    if proposal.supports_positional_pipeline(verify_window_scheduler.depth())
                        && ngram_sidecar_controller.permit_pipeline_start()
                        && verify_window_scheduler.supports_pipelining(
                            proposal
                                .tokens()
                                .len()
                                .min(native_mtp_options.verify_window_max_tokens.max(1)),
                        )
                    {
                        pipeline_seed = Some((
                            proposal,
                            if native_mtp_tokens.is_empty() {
                                None
                            } else {
                                native_mtp_origin
                            },
                        ));
                    } else if let Some(pending) = pending {
                        native_mtp.restore_pending_draft(pending);
                    }
                }
                if native_mtp_verify_windows_enabled
                    && (!pipelined_decode_enabled
                        || (pipelined.is_none()
                            && pipeline_seed.is_none()
                            && pipelined_windows.is_empty()))
                    && native_mtp_reject_cooldown_remaining == 0
                    && native_mtp_remaining >= 2
                {
                    let pending_native_mtp_draft = (request.native_mtp_enabled
                        && composite_proposal_buffer.is_none())
                    .then(|| native_mtp.take_pending_draft())
                    .flatten();
                    match self.execute_native_mtp_verify_window(
                        &request,
                        downstream,
                        &session_key,
                        request_id,
                        session_id,
                        prefill_token_count,
                        &wire_sampling,
                        &effective_native_mtp_options,
                        &mut verify_window_scheduler,
                        pending_native_mtp_draft,
                        &mut composite_proposal_buffer,
                        &mut cached_ngram_proposer,
                        &mut adaptive_verify_window,
                        &mut current,
                        decode_step,
                        &mut decoded_tokens,
                        &mut context_tokens,
                        &mut exact_replay_tokens,
                        &mut native_mtp,
                        &mut native_mtp_counters,
                        &mut native_mtp_reject_cooldown_remaining,
                        &mut native_mtp_suppress_cooldown_drafts_remaining,
                        &mut ngram_sidecar_controller,
                        &mut decode_stage0_compute_ms,
                        &mut decode_runtime_lock_wait_ms,
                        &mut decode_runtime_lock_wait_max_ms,
                        &mut decode_runtime_lock_hold_ms,
                        &mut decode_runtime_lock_hold_max_ms,
                        &mut decode_runtime_lock_acquires,
                        &mut decode_forward_activation_encode_ms,
                        &mut decode_output_activation_bytes,
                        &mut decode_forward_activation_bytes,
                        &mut decode_forward_write_ms,
                        &mut decode_downstream_wait_ms,
                        &mut on_token,
                    )? {
                        NativeMtpVerifyWindowControl::ReachedStop => break,
                        NativeMtpVerifyWindowControl::Continue => continue,
                        NativeMtpVerifyWindowControl::NoProposal => {}
                    }
                }
                if pipelined_decode_enabled {
                    if let Some((proposal, origin)) = pipeline_seed {
                        if verify_window_forwarder.is_none() {
                            verify_window_forwarder = Some(
                                AsyncForwarder::new(
                                    &*downstream,
                                    self.telemetry.clone(),
                                    verify_window_scheduler.depth(),
                                )
                                .map_err(openai_backend_error)?,
                            );
                        }
                        pipeline_epoch = pipeline_epoch.checked_add(1).ok_or_else(|| {
                            OpenAiError::backend("verify window pipeline epoch overflow")
                        })?;
                        pipelined_boundary_prediction = None;
                        pipelined = Some(CompositeProposalPipeline::new(proposal, origin));
                    }
                    if let Some(pipeline) = pipelined.as_mut() {
                        let pipeline_in_flight_limit = verify_window_scheduler.depth();
                        let chunk_width = speculative_policy::pipeline_chunk_width(
                            native_mtp_options.verify_window_max_tokens,
                            effective_speculative.draft_split_probability,
                            decoded_tokens.saturating_add(
                                usize::try_from(pipeline_epoch).unwrap_or(usize::MAX),
                            ),
                        );
                        while verify_window_scheduler.has_capacity()
                            && verify_window_scheduler.in_flight_len() < pipeline_in_flight_limit
                            && decoded_tokens + queued_active_tokens(&pipelined_windows)
                                < request.max_tokens as usize
                        {
                            let refill_threshold = chunk_width;
                            if pipeline.candidate_len() < refill_threshold {
                                let available_refill_tokens =
                                    native_mtp_options.ngram_max_proposal_tokens.min(
                                        native_mtp_remaining
                                            .saturating_sub(pipeline.optimistic_suffix().len()),
                                    );
                                let refill_budget =
                                    ngram_sidecar_controller.refill_limit(available_refill_tokens);
                                let appended = refill_pipeline_ngram_candidates(
                                    pipeline,
                                    &context_tokens,
                                    &mut cached_ngram_proposer,
                                    refill_budget,
                                )?;
                                verify_window_scheduler.record_horizon_refill(appended);
                            }
                            if !pipeline.has_remaining_candidates() {
                                break;
                            }
                            let Some(planned) = pipeline.next_chunk(chunk_width) else {
                                break;
                            };
                            let proposal_tokens = planned.proposal_tokens().to_vec();
                            let native_mtp_token_count = planned.native_mtp_token_count();
                            let starts_epoch = planned.starts_epoch();
                            let offset = queued_active_tokens(&pipelined_windows);
                            let layout = pipelined_window_layout(
                                prefill_token_count,
                                decoded_tokens,
                                offset,
                                starts_epoch,
                                current,
                                &proposal_tokens,
                            );
                            let window = verify_window_scheduler.open(
                                layout.pos_start,
                                layout.decode_step,
                                layout.input_tokens.len(),
                            )?;
                            let input_tokens = layout.input_tokens;
                            let message =
                                embedded_verify_window_message(VerifyWindowMessageArgs {
                                    window_id: window.id,
                                    request_id,
                                    session_id,
                                    prompt_token_count: request.prompt_token_ids.len(),
                                    pos_start: window.base_position,
                                    decode_step: window.decode_step,
                                    tokens: &input_tokens,
                                    sampling: wire_sampling.clone(),
                                })?;
                            let dispatched = self.dispatch_embedded_stage_message(
                                &request,
                                downstream,
                                &session_key,
                                &message,
                                &input_tokens,
                                verify_window_forwarder.as_mut(),
                            )?;
                            pipelined_windows.push_back(PipelinedCompositeWindow {
                                epoch: pipeline_epoch,
                                stale: false,
                                starts_epoch,
                                window,
                                input_tokens,
                                proposal_tokens,
                                native_mtp_token_count,
                                planned_advance_tokens: planned.advance_tokens(),
                                dispatched,
                            });
                        }
                    }
                    if let Some(window) = pipelined_windows.pop_front() {
                        let starts_epoch = window.starts_epoch;
                        let proposal_count = window.proposal_tokens.len();
                        let completion_timer = PhaseTimer::start();
                        let verify = self.complete_dispatched_stage_message_direct(
                            &request,
                            downstream,
                            window.dispatched,
                            WireReplyKind::PredictedTokens,
                        )?;
                        let completed =
                            verify_window_scheduler.complete_next(verify.reply.window.window_id)?;
                        if completed != window.window {
                            return Err(OpenAiError::backend(
                                "verify window scheduler lost FIFO state",
                            ));
                        }
                        if window.stale {
                            verify_window_scheduler.record_stale_execution(
                                completion_timer.elapsed_ms(),
                                verify.stats.stage0_compute_ms,
                                verify.stats.forward_write_ms,
                                verify.stats.downstream_wait_ms,
                                verify.elapsed_ms,
                            );
                            decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
                            decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
                            decode_runtime_lock_wait_max_ms = decode_runtime_lock_wait_max_ms
                                .max(verify.stats.runtime_lock_wait_ms);
                            decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
                            decode_runtime_lock_hold_max_ms = decode_runtime_lock_hold_max_ms
                                .max(verify.stats.runtime_lock_hold_ms);
                            decode_runtime_lock_acquires += 1;
                            decode_forward_activation_encode_ms +=
                                verify.stats.activation_encode_ms;
                            decode_output_activation_bytes = decode_output_activation_bytes
                                .saturating_add(verify.stats.output_activation_bytes);
                            decode_forward_activation_bytes = decode_forward_activation_bytes
                                .saturating_add(verify.stats.forward_activation_bytes);
                            decode_forward_write_ms += verify.stats.forward_write_ms;
                            decode_downstream_wait_ms += verify.stats.downstream_wait_ms;
                            continue;
                        }
                        if window.epoch != pipeline_epoch || pipelined.is_none() {
                            return Err(OpenAiError::backend(
                                "active verify window has no matching proposal epoch",
                            ));
                        }
                        let target_predictions = compose_target_predictions(
                            starts_epoch,
                            proposal_count,
                            pipelined_boundary_prediction,
                            &verify.reply.predicted_tokens,
                        )?;
                        let native_mtp_verify_decision = classify_native_mtp_verify_window(
                            &window.proposal_tokens,
                            &target_predictions,
                            decoded_tokens,
                            request.max_tokens as usize,
                            |token| {
                                self.iteration_scheduler.execute_runtime(
                                    "embedded-token-eog",
                                    move |runtime| {
                                        runtime
                                            .model
                                            .token_is_eog(token)
                                            .map_err(openai_backend_error)
                                    },
                                )
                            },
                        )?;
                        let acceptance = speculative_policy::pipeline_acceptance_policy(
                            native_mtp_verify_decision.accepted_proposal_tokens,
                            window.proposal_tokens.len(),
                            native_mtp_verify_decision.rejected,
                            effective_speculative.draft_acceptance_threshold,
                        );
                        let fully_accepted_window = acceptance.fully_accepted;
                        let pipeline_continues = fully_accepted_window;
                        let accepted_candidate_tokens = acceptance.accepted_candidate_tokens;
                        if window.native_mtp_token_count > 0 {
                            let pipeline = pipelined.as_ref().expect("pipeline retained");
                            let span = native_mtp.observe_taken_draft_span(
                                &window.proposal_tokens[..window.native_mtp_token_count],
                                &target_predictions,
                                ms_to_us(verify.elapsed_ms),
                            );
                            for index in 0..span.accepted_count + usize::from(span.rejected) {
                                native_mtp_counters.observe_verify_window_verification(
                                    pipeline.origin().expect("native MTP candidate has origin"),
                                    index < span.accepted_count,
                                );
                            }
                        }
                        speculative_stats.windows += 1;
                        speculative_stats.draft_tokens += window.proposal_tokens.len();
                        speculative_stats.primary_verify_requests += 1;
                        speculative_stats.primary_verify_tokens += window.input_tokens.len();
                        speculative_stats.primary_verify_elapsed_ms += verify.elapsed_ms;
                        speculative_stats.primary_verify_stage0_compute_ms +=
                            verify.stats.stage0_compute_ms;
                        speculative_stats.primary_verify_runtime_lock_wait_ms +=
                            verify.stats.runtime_lock_wait_ms;
                        speculative_stats.primary_verify_runtime_lock_hold_ms +=
                            verify.stats.runtime_lock_hold_ms;
                        speculative_stats.primary_verify_activation_encode_ms +=
                            verify.stats.activation_encode_ms;
                        speculative_stats.primary_verify_forward_write_ms +=
                            verify.stats.forward_write_ms;
                        speculative_stats.primary_verify_downstream_wait_ms +=
                            verify.stats.downstream_wait_ms;
                        speculative_stats.primary_verify_output_activation_bytes =
                            speculative_stats
                                .primary_verify_output_activation_bytes
                                .saturating_add(verify.stats.output_activation_bytes);
                        speculative_stats.primary_verify_forward_activation_bytes =
                            speculative_stats
                                .primary_verify_forward_activation_bytes
                                .saturating_add(verify.stats.forward_activation_bytes);
                        decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
                        decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
                        decode_runtime_lock_wait_max_ms =
                            decode_runtime_lock_wait_max_ms.max(verify.stats.runtime_lock_wait_ms);
                        decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
                        decode_runtime_lock_hold_max_ms =
                            decode_runtime_lock_hold_max_ms.max(verify.stats.runtime_lock_hold_ms);
                        decode_runtime_lock_acquires += 1;
                        decode_forward_activation_encode_ms += verify.stats.activation_encode_ms;
                        decode_output_activation_bytes = decode_output_activation_bytes
                            .saturating_add(verify.stats.output_activation_bytes);
                        decode_forward_activation_bytes = decode_forward_activation_bytes
                            .saturating_add(verify.stats.forward_activation_bytes);
                        decode_forward_write_ms += verify.stats.forward_write_ms;
                        decode_downstream_wait_ms += verify.stats.downstream_wait_ms;
                        if fully_accepted_window {
                            speculative_stats.accepted_tokens += accepted_candidate_tokens;
                            speculative_stats.full_accept_windows += 1;
                            let pipeline = pipelined.as_mut().expect("pipeline retained");
                            pipeline.set_next_draft(
                                request.native_mtp_enabled,
                                verify
                                    .reply
                                    .native_mtp_draft
                                    .clone()
                                    .map(NativeMtpDraft::from_stage_draft),
                            );
                        } else {
                            speculative_stats.rejected_tokens += window
                                .proposal_tokens
                                .len()
                                .saturating_sub(accepted_candidate_tokens)
                                .max(1);
                            speculative_stats.rejected_windows += 1;
                            if accepted_candidate_tokens + 1 < window.proposal_tokens.len() {
                                speculative_stats.early_reject_windows += 1;
                            }
                            speculative_stats.first_reject_position_sum +=
                                accepted_candidate_tokens + 1;
                        }
                        pipelined
                            .as_mut()
                            .expect("pipeline retained")
                            .observe_accepted(accepted_candidate_tokens);
                        let mut reached_stop = false;
                        let later_active_window = pipelined_windows
                            .iter()
                            .any(|queued| queued.epoch == pipeline_epoch && !queued.stale);
                        let undispatched_candidates = pipelined
                            .as_ref()
                            .is_some_and(CompositeProposalPipeline::has_remaining_candidates);
                        let commit_count = speculative_policy::pipeline_commit_count(
                            window.planned_advance_tokens,
                            native_mtp_verify_decision.commit_count,
                            acceptance.threshold_met,
                            fully_accepted_window,
                            later_active_window || undispatched_candidates,
                        );
                        if verify_checkpoint_no_longer_needed(
                            commit_count,
                            window.input_tokens.len(),
                        ) {
                            self.retire_verify_window(
                                &request,
                                downstream,
                                verify_window_forwarder.as_mut(),
                                &session_key,
                                VerifyRetirement {
                                    request_id,
                                    session_id,
                                    token_start: window.window.base_position,
                                    token_count: window.input_tokens.len(),
                                },
                            )?;
                        }
                        for token in target_predictions.iter().copied().take(commit_count) {
                            current = token;
                            decoded_tokens += 1;
                            exact_replay_tokens.push(current);
                            context_tokens.push(current);
                            if on_token(current)? == TokenControl::Stop
                                || decoded_tokens >= request.max_tokens as usize
                            {
                                reached_stop = true;
                                break;
                            }
                        }
                        if !pipeline_continues || reached_stop {
                            pipelined_boundary_prediction = None;
                            let stale_count =
                                mark_epoch_stale(&mut pipelined_windows, pipeline_epoch);
                            verify_window_scheduler.mark_recovery_epoch(stale_count);
                            if verify_window_scheduler.is_runahead()
                                && let Some((min_id, max_id)) =
                                    stale_window_id_range(&pipelined_windows, pipeline_epoch)
                            {
                                self.discard_stale_windows(
                                    &request,
                                    downstream,
                                    verify_window_forwarder.as_mut(),
                                    StaleWindowDiscard {
                                        request_id,
                                        session_id,
                                        min_window_id: min_id,
                                        max_window_id: max_id,
                                    },
                                )?;
                            }
                            let pipeline = pipelined.take().expect("pipeline retained");
                            if ngram_sidecar_controller.observe_tail_outcome(
                                pipeline.proposal(),
                                pipeline.accepted_tokens(),
                            ) {
                                native_mtp_counters.observe_ngram_tail_rejection();
                            }
                            native_mtp_counters.observe_hybrid_proposal(
                                pipeline.proposal(),
                                pipeline.accepted_tokens(),
                            );
                            native_mtp.clear_pending_draft();
                            if native_mtp_verify_decision.rejected
                                && pipeline
                                    .proposal()
                                    .native_mtp_prefix_rejected(pipeline.accepted_tokens())
                                && native_mtp_options.reject_cooldown_tokens > 0
                            {
                                native_mtp_reject_cooldown_remaining =
                                    native_mtp_options.reject_cooldown_tokens;
                                native_mtp_suppress_cooldown_drafts_remaining =
                                    native_mtp_options.suppress_cooldown_draft_limit;
                            }
                            if reached_stop {
                                break;
                            }
                        } else {
                            pipelined_boundary_prediction = target_predictions
                                .get(window.proposal_tokens.len())
                                .copied();
                        }
                        if pipeline_continues
                            && can_seed_pipeline(&pipelined_windows)
                            && pipelined
                                .as_ref()
                                .is_some_and(|pipeline| !pipeline.has_remaining_candidates())
                        {
                            let mut pipeline = pipelined.take().expect("pipeline retained");
                            let next_draft_available = pipeline.next_draft().is_some();
                            ngram_sidecar_controller.observe_tail_outcome(
                                pipeline.proposal(),
                                pipeline.accepted_tokens(),
                            );
                            native_mtp_counters.observe_hybrid_proposal(
                                pipeline.proposal(),
                                pipeline.accepted_tokens(),
                            );
                            native_mtp_counters.observe_verify_next_draft(
                                next_draft_available,
                                next_draft_available,
                            );
                            if let Some(next_draft) = pipeline.take_next_draft() {
                                native_mtp.observe_next_draft(
                                    Some(next_draft),
                                    NativeMtpDraftOrigin::VerifyNext,
                                );
                            }
                        }
                        if self.telemetry.is_debug_enabled() {
                            let mut attrs = self.openai_attrs(request.ids);
                            attrs.insert(
                                "llama_stage.message_kind".to_string(),
                                json!("VerifyWindow"),
                            );
                            attrs.insert(
                                "llama_stage.spec.proposal_source".to_string(),
                                json!("composite_mtp_ngram"),
                            );
                            attrs.insert(
                                "llama_stage.verify_window_id".to_string(),
                                json!(window.window.id),
                            );
                            attrs.insert(
                                "llama_stage.verify_window.accepted".to_string(),
                                json!(pipeline_continues),
                            );
                            attrs.insert(
                                "llama_stage.verify_window.in_flight_after".to_string(),
                                json!(verify_window_scheduler.in_flight_len()),
                            );
                            attrs.insert(
                                "llama_stage.verify_window.stale_discarded".to_string(),
                                json!(verify_window_scheduler.stale_discard_count()),
                            );
                            self.emit_openai_phase(
                                "stage.openai_decode_verify_window",
                                token_timer,
                                attrs,
                            );
                        }
                        continue;
                    }
                }
                if draft_guard.is_some()
                    || (effective_speculative.ngram.is_some() && !pipelined_decode_enabled)
                {
                    let remaining = (request.max_tokens as usize).saturating_sub(decoded_tokens);
                    if remaining == 0 {
                        break;
                    }
                    let mut proposal_source = "none";
                    let proposal_limit = remaining.min(adaptive_window);
                    let propose_timer = PhaseTimer::start();
                    let mut draft_tokens = Vec::new();
                    if let (true, Some(draft)) =
                        (draft_tokens.is_empty(), draft_guard.as_deref_mut())
                    {
                        let proposal_limit = proposal_limit.min(draft.window);
                        draft_tokens = draft
                            .propose(current, proposal_limit)
                            .map_err(openai_backend_error)?;
                        if !draft_tokens.is_empty() {
                            proposal_source = "draft-model";
                        }
                    }
                    if draft_tokens.is_empty() && effective_speculative.ngram.is_some() {
                        let proposal = propose_configured_ngram_tokens(
                            effective_speculative,
                            &mut cached_ngram_proposer,
                            &context_tokens,
                            proposal_limit.min(request.ngram_max),
                        )?;
                        draft_tokens = proposal.tokens;
                        if !draft_tokens.is_empty() {
                            proposal_source = proposal.source;
                        }
                    }
                    let draft_propose_ms = propose_timer.elapsed_ms();
                    speculative_stats.draft_propose_ms += draft_propose_ms;
                    if !draft_tokens.is_empty() {
                        let split_len = speculative_policy::draft_split_len(
                            draft_tokens.len(),
                            effective_speculative.draft_split_probability,
                            decoded_tokens,
                        );
                        draft_tokens.truncate(split_len);
                        let verify_inputs = verify_inputs_for_proposals(current, &draft_tokens);
                        let message = embedded_verify_window_message(VerifyWindowMessageArgs {
                            window_id: i32::try_from(decoded_tokens)
                                .map_err(|_| OpenAiError::backend("decode step exceeds i32"))?,
                            request_id,
                            session_id,
                            prompt_token_count: request.prompt_token_ids.len(),
                            pos_start: prefill_token_count + decoded_tokens,
                            decode_step: decoded_tokens,
                            tokens: &verify_inputs,
                            sampling: wire_sampling.clone(),
                        })?;
                        let verify = self.execute_embedded_stage_message(
                            &request,
                            downstream,
                            &session_key,
                            &message,
                            &verify_inputs,
                            WireReplyKind::PredictedTokens,
                        )?;
                        speculative_stats.windows += 1;
                        speculative_stats.draft_tokens += draft_tokens.len();
                        speculative_stats.primary_verify_requests += 1;
                        speculative_stats.primary_verify_tokens += verify_inputs.len();
                        speculative_stats.primary_verify_elapsed_ms += verify.elapsed_ms;
                        speculative_stats.primary_verify_stage0_compute_ms +=
                            verify.stats.stage0_compute_ms;
                        speculative_stats.primary_verify_runtime_lock_wait_ms +=
                            verify.stats.runtime_lock_wait_ms;
                        speculative_stats.primary_verify_runtime_lock_hold_ms +=
                            verify.stats.runtime_lock_hold_ms;
                        speculative_stats.primary_verify_activation_encode_ms +=
                            verify.stats.activation_encode_ms;
                        speculative_stats.primary_verify_forward_write_ms +=
                            verify.stats.forward_write_ms;
                        speculative_stats.primary_verify_downstream_wait_ms +=
                            verify.stats.downstream_wait_ms;
                        speculative_stats.primary_verify_output_activation_bytes =
                            speculative_stats
                                .primary_verify_output_activation_bytes
                                .saturating_add(verify.stats.output_activation_bytes);
                        speculative_stats.primary_verify_forward_activation_bytes =
                            speculative_stats
                                .primary_verify_forward_activation_bytes
                                .saturating_add(verify.stats.forward_activation_bytes);
                        decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
                        decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
                        decode_runtime_lock_wait_max_ms =
                            decode_runtime_lock_wait_max_ms.max(verify.stats.runtime_lock_wait_ms);
                        decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
                        decode_runtime_lock_hold_max_ms =
                            decode_runtime_lock_hold_max_ms.max(verify.stats.runtime_lock_hold_ms);
                        decode_runtime_lock_acquires += 1;
                        decode_forward_activation_encode_ms += verify.stats.activation_encode_ms;
                        decode_output_activation_bytes = decode_output_activation_bytes
                            .saturating_add(verify.stats.output_activation_bytes);
                        decode_forward_activation_bytes = decode_forward_activation_bytes
                            .saturating_add(verify.stats.forward_activation_bytes);
                        decode_forward_write_ms += verify.stats.forward_write_ms;
                        decode_downstream_wait_ms += verify.stats.downstream_wait_ms;
                        let decision = classify_verify_window_with_threshold(
                            &draft_tokens,
                            &verify.reply.predicted_tokens,
                            decoded_tokens,
                            request.max_tokens as usize,
                            effective_speculative.draft_acceptance_threshold,
                            |token| {
                                self.iteration_scheduler.execute_runtime(
                                    "embedded-token-eog",
                                    move |runtime| {
                                        runtime
                                            .model
                                            .token_is_eog(token)
                                            .map_err(openai_backend_error)
                                    },
                                )
                            },
                        )?;
                        let checkpoint_no_longer_needed = verify_checkpoint_no_longer_needed(
                            decision.commit_count,
                            verify_inputs.len(),
                        );
                        if checkpoint_no_longer_needed {
                            self.retire_verify_window(
                                &request,
                                downstream,
                                None,
                                &session_key,
                                VerifyRetirement {
                                    request_id,
                                    session_id,
                                    token_start: prefill_token_count + decoded_tokens,
                                    token_count: verify_inputs.len(),
                                },
                            )?;
                        }
                        speculative_stats.observe_verify_decision(
                            decision,
                            &mut adaptive_window,
                            request.adaptive_speculative_window,
                            max_speculative_window,
                        );
                        let commit_tokens =
                            verify.reply.predicted_tokens[..decision.commit_count].to_vec();
                        let mut reached_stop = false;
                        for token in commit_tokens {
                            current = token;
                            decoded_tokens += 1;
                            context_tokens.push(current);
                            if on_token(current)? == TokenControl::Stop {
                                reached_stop = true;
                            }
                            if reached_stop || decoded_tokens >= request.max_tokens as usize {
                                break;
                            }
                        }
                        speculative_stats.adaptive_window_final = adaptive_window;
                        if proposal_source == "draft-model" && (decision.rejected() || reached_stop)
                        {
                            let draft_reset_timer = PhaseTimer::start();
                            if let Some(draft) = draft_guard.as_deref_mut() {
                                draft
                                    .reset_to_context(&context_tokens)
                                    .map_err(openai_backend_error)?;
                                speculative_stats.draft_reset_ms += draft_reset_timer.elapsed_ms();
                            }
                        }
                        let mut token_attrs = self.openai_attrs(request.ids);
                        token_attrs
                            .insert("llama_stage.decode_step".to_string(), json!(decode_step));
                        token_attrs.insert(
                            "llama_stage.message_kind".to_string(),
                            json!("VerifyWindow"),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.windows".to_string(),
                            json!(speculative_stats.windows),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposed".to_string(),
                            json!(draft_tokens.len()),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.accepted".to_string(),
                            json!(decision.accepted_before_reject),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.rejected".to_string(),
                            json!(decision.rejected()),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.draft_propose_ms".to_string(),
                            json!(draft_propose_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposal_source".to_string(),
                            json!(proposal_source),
                        );
                        token_attrs.insert(
                            "llama_stage.spec.proposal_limit".to_string(),
                            json!(proposal_limit),
                        );
                        token_attrs.insert(
                            "llama_stage.stage0_compute_ms".to_string(),
                            json!(verify.stats.stage0_compute_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_wait_ms".to_string(),
                            json!(verify.stats.runtime_lock_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.runtime_lock_hold_ms".to_string(),
                            json!(verify.stats.runtime_lock_hold_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.activation_encode_ms".to_string(),
                            json!(verify.stats.activation_encode_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_write_ms".to_string(),
                            json!(verify.stats.forward_write_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.downstream_wait_ms".to_string(),
                            json!(verify.stats.downstream_wait_ms),
                        );
                        token_attrs.insert(
                            "llama_stage.output_activation_bytes".to_string(),
                            json!(verify.stats.output_activation_bytes),
                        );
                        token_attrs.insert(
                            "llama_stage.forward_activation_bytes".to_string(),
                            json!(verify.stats.forward_activation_bytes),
                        );
                        self.emit_openai_phase(
                            "stage.openai_decode_verify_window",
                            token_timer,
                            token_attrs,
                        );
                        if reached_stop {
                            break;
                        }
                        continue;
                    }
                }
                let uses_context_sideband = decode_uses_context_sideband(
                    &context_tokens,
                    current,
                    skippy_protocol::binary::MAX_STAGE_SIDEBAND_VALUES,
                );
                let records_replay_checkpoint = uses_context_sideband && context_tokens.len() > 1;
                let records_full_prompt_checkpoint = decode_step == 0
                    && uses_context_sideband
                    && context_tokens.len() == request.prompt_token_ids.len();
                let decode_step_index = usize::try_from(decode_step)
                    .map_err(|_| OpenAiError::backend("decode step exceeds usize"))?;
                let message = if uses_context_sideband {
                    decode_message.update_with_tokens(
                        decode_step_index,
                        current,
                        &context_tokens,
                    )?
                } else {
                    decode_message.update(decode_step_index, current)?
                };
                let stage0_timer = PhaseTimer::start();
                let batch_outcome = self.iteration_scheduler.execute_frame_iteration(
                    &session_key,
                    u64::try_from(message.pos_start).map_err(|_| {
                        OpenAiError::backend("negative authoritative decode position")
                    })?,
                    &[current],
                    &[],
                    request.sampling.enabled.then_some(request.sampling),
                    None,
                    true,
                )?;
                if let Some(align) = batch_outcome.session_alignment {
                    let mut attrs = self.openai_attrs(request.ids);
                    attrs.insert(
                        "llama_stage.session_auto_align_before_tokens".to_string(),
                        json!(align.before_token_count),
                    );
                    attrs.insert(
                        "llama_stage.session_auto_align_after_tokens".to_string(),
                        json!(align.after_token_count),
                    );
                    self.telemetry
                        .emit_debug("stage.openai_session_auto_align", attrs);
                }
                let token_runtime_lock_wait_ms = batch_outcome.runtime_lock_wait_ms;
                let token_runtime_lock_hold_ms = batch_outcome.runtime_lock_hold_ms;
                decode_runtime_lock_wait_ms += token_runtime_lock_wait_ms;
                decode_runtime_lock_wait_max_ms =
                    decode_runtime_lock_wait_max_ms.max(token_runtime_lock_wait_ms);
                decode_runtime_lock_hold_ms += token_runtime_lock_hold_ms;
                decode_runtime_lock_hold_max_ms =
                    decode_runtime_lock_hold_max_ms.max(token_runtime_lock_hold_ms);
                decode_runtime_lock_acquires += 1;
                decode_batch_size_max = decode_batch_size_max.max(batch_outcome.batch_size);
                decode_batch_wait_ms += batch_outcome.batch_wait_ms;
                let output = batch_outcome.output;
                let stage0_compute_ms = stage0_timer.elapsed_ms();
                decode_stage0_compute_ms += stage0_compute_ms;
                let forwarded = forwarded_stage_message_timed(
                    request.config,
                    message,
                    &output,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?;
                decode_forward_activation_encode_ms += forwarded.activation_encode_ms;
                decode_output_activation_bytes =
                    decode_output_activation_bytes.saturating_add(output.payload.len());
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(forwarded.message.activation.len());
                let write_timer = PhaseTimer::start();
                write_stage_message_conditioned(
                    &mut *downstream,
                    &forwarded.message,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let forward_write_ms = write_timer.elapsed_ms();
                decode_forward_write_ms += forward_write_ms;
                let wait_timer = PhaseTimer::start();
                let reply = super::embedded_execution::receive_embedded_stage_reply(
                    downstream,
                    request.prediction_return.as_ref(),
                    WireReplyKind::PredictedToken,
                )?;
                let downstream_wait_ms = wait_timer.elapsed_ms();
                decode_downstream_wait_ms += downstream_wait_ms;
                if records_replay_checkpoint
                    && super::prefix_cache::request_allows_exact_replay(&request)
                {
                    self.record_embedded_stage0_replay_checkpoint(
                        super::prefix_cache::EmbeddedReplayCheckpointRecord {
                            session_id: &session_key,
                            ids: request.ids,
                            prompt_token_ids: request.prompt_token_ids,
                            checkpoint_token_ids: &context_tokens,
                            predicted_tokens: &exact_replay_tokens,
                            predicted: reply.predicted,
                            sampling: request.sampling,
                            chat_sampling_metadata: request.chat_sampling_metadata,
                        },
                    )?;
                } else if records_full_prompt_checkpoint {
                    self.record_embedded_stage0_full_prompt_first_token(
                        &session_key,
                        request.ids,
                        request.prompt_token_ids,
                        reply.predicted,
                        request.sampling,
                        request.chat_sampling_metadata,
                    )?;
                }
                current = reply.predicted;
                let suppress_cooldown_draft_broad = native_mtp_options.suppress_cooldown_drafts
                    && native_mtp_reject_cooldown_remaining > 0;
                let suppress_cooldown_draft_limited = native_mtp_reject_cooldown_remaining > 0
                    && native_mtp_suppress_cooldown_drafts_remaining > 0;
                let suppress_cooldown_draft =
                    suppress_cooldown_draft_broad || suppress_cooldown_draft_limited;
                let native_mtp_draft = if suppress_cooldown_draft {
                    None
                } else {
                    reply
                        .native_mtp_draft
                        .clone()
                        .map(NativeMtpDraft::from_stage_draft)
                };
                if suppress_cooldown_draft {
                    native_mtp.clear_pending_draft();
                    native_mtp_counters.observe_suppressed_cooldown_draft();
                    native_mtp_suppress_cooldown_drafts_remaining =
                        native_mtp_suppress_cooldown_drafts_remaining.saturating_sub(1);
                }
                let native_mtp_decision = native_mtp.observe_target_token(
                    current,
                    ms_to_us(downstream_wait_ms),
                    native_mtp_draft,
                    if native_mtp_counters.verify_window_verification_count() == 0 {
                        NativeMtpDraftOrigin::InitialSerial
                    } else {
                        NativeMtpDraftOrigin::SerialAfterGap
                    },
                );
                native_mtp_reject_cooldown_remaining =
                    native_mtp_reject_cooldown_remaining.saturating_sub(1);
                decoded_tokens += 1;
                exact_replay_tokens.push(current);
                context_tokens.push(current);
                if self.telemetry.is_debug_enabled() {
                    let mut token_attrs = self.openai_attrs(request.ids);
                    token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                    token_attrs.insert(
                        "llama_stage.decode_token_phase".to_string(),
                        json!(decode_token_phase(decode_step)),
                    );
                    token_attrs.insert(
                        "llama_stage.stage0_compute_ms".to_string(),
                        json!(stage0_compute_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.runtime_lock_wait_ms".to_string(),
                        json!(token_runtime_lock_wait_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.runtime_lock_hold_ms".to_string(),
                        json!(token_runtime_lock_hold_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_batch_size".to_string(),
                        json!(batch_outcome.batch_size),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_batch_wait_ms".to_string(),
                        json!(batch_outcome.batch_wait_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.output_activation_bytes".to_string(),
                        json!(output.payload.len()),
                    );
                    token_attrs.insert(
                        "llama_stage.forward_activation_bytes".to_string(),
                        json!(forwarded.message.activation.len()),
                    );
                    token_attrs.insert(
                        "llama_stage.activation_encode_ms".to_string(),
                        json!(forwarded.activation_encode_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.forward_write_ms".to_string(),
                        json!(forward_write_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.downstream_wait_ms".to_string(),
                        json!(downstream_wait_ms),
                    );
                    token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                    token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeEmbd"));
                    token_attrs.insert(
                        "llama_stage.native_mtp.verification".to_string(),
                        json!(native_mtp_decision.label()),
                    );
                    token_attrs.insert(
                        "llama_stage.native_mtp.suppress_cooldown_drafts".to_string(),
                        json!(native_mtp_options.suppress_cooldown_drafts),
                    );
                    token_attrs.insert(
                        "llama_stage.native_mtp.suppress_cooldown_draft_limit".to_string(),
                        json!(native_mtp_options.suppress_cooldown_draft_limit),
                    );
                    token_attrs.insert(
                        "llama_stage.native_mtp.cooldown_draft_suppressed".to_string(),
                        json!(suppress_cooldown_draft),
                    );
                    self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
                }
                if on_token(current)? == TokenControl::Stop {
                    break;
                }
            }
            if !pipelined_windows.is_empty() {
                let stale_count = mark_epoch_stale(&mut pipelined_windows, pipeline_epoch);
                verify_window_scheduler.mark_stale(stale_count);
                if verify_window_scheduler.is_runahead()
                    && let Some((min_id, max_id)) =
                        stale_window_id_range(&pipelined_windows, pipeline_epoch)
                {
                    self.discard_stale_windows(
                        &request,
                        downstream,
                        verify_window_forwarder.as_mut(),
                        StaleWindowDiscard {
                            request_id,
                            session_id,
                            min_window_id: min_id,
                            max_window_id: max_id,
                        },
                    )?;
                }
                while let Some(stale) = pipelined_windows.pop_front() {
                    let stale_drain_timer = PhaseTimer::start();
                    let stale_reply = self.complete_dispatched_stage_message_direct(
                        &request,
                        downstream,
                        stale.dispatched,
                        WireReplyKind::PredictedTokens,
                    )?;
                    verify_window_scheduler.complete_next(stale_reply.reply.window.window_id)?;
                    verify_window_scheduler.record_stale_execution(
                        stale_drain_timer.elapsed_ms(),
                        stale_reply.stats.stage0_compute_ms,
                        stale_reply.stats.forward_write_ms,
                        stale_reply.stats.downstream_wait_ms,
                        stale_reply.elapsed_ms,
                    );
                    decode_stage0_compute_ms += stale_reply.stats.stage0_compute_ms;
                    decode_runtime_lock_wait_ms += stale_reply.stats.runtime_lock_wait_ms;
                    decode_runtime_lock_wait_max_ms =
                        decode_runtime_lock_wait_max_ms.max(stale_reply.stats.runtime_lock_wait_ms);
                    decode_runtime_lock_hold_ms += stale_reply.stats.runtime_lock_hold_ms;
                    decode_runtime_lock_hold_max_ms =
                        decode_runtime_lock_hold_max_ms.max(stale_reply.stats.runtime_lock_hold_ms);
                    decode_runtime_lock_acquires += 1;
                    decode_forward_activation_encode_ms += stale_reply.stats.activation_encode_ms;
                    decode_output_activation_bytes = decode_output_activation_bytes
                        .saturating_add(stale_reply.stats.output_activation_bytes);
                    decode_forward_activation_bytes = decode_forward_activation_bytes
                        .saturating_add(stale_reply.stats.forward_activation_bytes);
                    decode_forward_write_ms += stale_reply.stats.forward_write_ms;
                    decode_downstream_wait_ms += stale_reply.stats.downstream_wait_ms;
                }
            }
            if let Some(pipeline) = pipelined.take() {
                native_mtp_counters
                    .observe_hybrid_proposal(pipeline.proposal(), pipeline.accepted_tokens());
                native_mtp.clear_pending_draft();
            }
            if let Some(proposer) = cached_ngram_proposer.as_ref() {
                native_mtp_counters.observe_history_ngram_proposer(proposer.stats());
            }
            let native_mtp_stats = native_mtp.stats();
            self.record_embedded_decode_summary(
                &request,
                &mut cache_stats,
                decode_timer,
                EmbeddedDecodeSummary {
                    decoded_tokens,
                    stage0_compute_ms: decode_stage0_compute_ms,
                    runtime_lock_wait_ms: decode_runtime_lock_wait_ms,
                    runtime_lock_wait_max_ms: decode_runtime_lock_wait_max_ms,
                    runtime_lock_hold_ms: decode_runtime_lock_hold_ms,
                    runtime_lock_hold_max_ms: decode_runtime_lock_hold_max_ms,
                    runtime_lock_acquires: decode_runtime_lock_acquires,
                    decode_batch_size_max,
                    decode_batch_wait_ms,
                    forward_write_ms: decode_forward_write_ms,
                    activation_encode_ms: decode_forward_activation_encode_ms,
                    output_activation_bytes: decode_output_activation_bytes,
                    forward_activation_bytes: decode_forward_activation_bytes,
                    downstream_wait_ms: decode_downstream_wait_ms,
                    speculative_stats: &speculative_stats,
                    native_mtp_stats,
                    native_mtp_counters,
                    native_mtp_options,
                    verify_window_scheduler: &verify_window_scheduler,
                },
            );
            Ok(())
        })();

        self.finish_embedded_generation_session(&request, lane_pool, lane, &result, &session_key)?;
        result?;
        Ok(cache_stats)
    }
}

#[cfg(test)]
mod tests {
    use super::prefill_chunk_end;

    #[test]
    fn exact_checkpoint_splits_prefill_at_the_native_state_boundary() {
        assert_eq!(prefill_chunk_end(0, 1024, 1400, Some(768)), 768);
        assert_eq!(prefill_chunk_end(768, 1024, 1400, Some(768)), 1400);
    }

    #[test]
    fn exact_checkpoint_preserves_earlier_adaptive_chunks() {
        assert_eq!(prefill_chunk_end(0, 256, 1400, Some(768)), 256);
        assert_eq!(prefill_chunk_end(256, 256, 1400, Some(768)), 512);
        assert_eq!(prefill_chunk_end(512, 512, 1400, Some(768)), 768);
    }
}
