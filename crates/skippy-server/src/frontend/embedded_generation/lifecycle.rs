use std::{borrow::Cow, collections::VecDeque};

use openai_frontend::{OpenAiError, OpenAiResult};
use serde_json::json;
use skippy_protocol::binary::{StageWireMessage, WireReplyKind, recv_reply, write_stage_message};

use crate::frontend::{
    CompositeProposalPipeline, HistoryNgramProposer, NativeMtpDecodeCounters,
    NativeMtpDecodeOptions, NativeMtpDecodeTelemetry, NativeMtpStats,
    decode_scheduler::{VerifyWindow, VerifyWindowScheduler},
    embedded_execution::DispatchedEmbeddedStage,
    generation::{
        EmbeddedStageZeroGeneration, GenerationCacheStats, LocalGeneration, PersistentStageLane,
        PersistentStageLanePool, PhaseTimer, StageOpenAiBackend, TokenControl,
    },
    speculative::{OpenAiSpeculativeStats, SpeculativeDecodeConfig},
    util::{openai_backend_error, openai_io_error},
};

/// Keeps the configured speculative plan unless a distributed prefix restore
/// has already populated every stage's session. Pure N-gram verification after
/// that restore currently races the restored session lifecycle, so only the
/// history-based N-gram pieces are suppressed for this request.
pub(super) fn speculation_after_prefix_restore(
    config: &SpeculativeDecodeConfig,
    prefix_restored: bool,
) -> Cow<'_, SpeculativeDecodeConfig> {
    if !prefix_restored || config.ngram.is_none() {
        return Cow::Borrowed(config);
    }

    let mut safe = config.clone();
    safe.ngram = None;
    safe.extension = None;
    Cow::Owned(safe)
}

pub(super) struct PipelinedCompositeWindow {
    pub(super) epoch: u64,
    pub(super) stale: bool,
    pub(super) starts_epoch: bool,
    pub(super) window: VerifyWindow,
    pub(super) input_tokens: Vec<i32>,
    pub(super) proposal_tokens: Vec<i32>,
    pub(super) native_mtp_token_count: usize,
    pub(super) planned_advance_tokens: usize,
    pub(super) dispatched: DispatchedEmbeddedStage,
}

pub(super) struct PipelinedWindowLayout {
    pub(super) pos_start: usize,
    pub(super) decode_step: usize,
    pub(super) input_tokens: Vec<i32>,
}

/// Places an epoch-start chunk and its non-overlapping continuations on one
/// monotonically increasing absolute token axis.
pub(super) fn pipelined_window_layout(
    prefill_token_count: usize,
    decoded_tokens: usize,
    queued_advance_tokens: usize,
    starts_epoch: bool,
    current: i32,
    proposal_tokens: &[i32],
) -> PipelinedWindowLayout {
    let continuation_offset = usize::from(!starts_epoch);
    let decode_step = decoded_tokens + queued_advance_tokens + continuation_offset;
    let mut input_tokens = Vec::with_capacity(proposal_tokens.len() + usize::from(starts_epoch));
    if starts_epoch {
        input_tokens.push(current);
    }
    input_tokens.extend_from_slice(proposal_tokens);
    PipelinedWindowLayout {
        pos_start: prefill_token_count + decode_step,
        decode_step,
        input_tokens,
    }
}

/// Reconstructs the target predictions for one proposal span.
///
/// The epoch-start traversal predicts every proposal plus one free token. A
/// continuation predicts proposal 2..K plus the free token; proposal 1 is the
/// preceding traversal's boundary prediction.
pub(super) fn compose_target_predictions(
    starts_epoch: bool,
    proposal_count: usize,
    prior_boundary_prediction: Option<i32>,
    traversal_predictions: &[i32],
) -> OpenAiResult<Vec<i32>> {
    let required = proposal_count.saturating_add(1);
    // Stateful chat sampling stops verifying at the first mismatched row and
    // returns only the rows it sampled, so a shorter prediction array is a
    // legitimate early rejection — pass through what arrived and let window
    // classification reject at the divergence. An empty reply is still a
    // protocol error.
    if starts_epoch {
        if traversal_predictions.is_empty() {
            return Err(OpenAiError::backend(
                "epoch-start verify window returned no predictions",
            ));
        }
        let available = required.min(traversal_predictions.len());
        return Ok(traversal_predictions[..available].to_vec());
    }

    let boundary = prior_boundary_prediction.ok_or_else(|| {
        OpenAiError::backend("continuation verify window has no prior boundary prediction")
    })?;
    if traversal_predictions.is_empty() {
        return Err(OpenAiError::backend(
            "continuation verify window returned no predictions",
        ));
    }
    let available = proposal_count.min(traversal_predictions.len());
    let mut predictions = Vec::with_capacity(available.saturating_add(1));
    predictions.push(boundary);
    predictions.extend_from_slice(&traversal_predictions[..available]);
    Ok(predictions)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DirectPredictionReturnPath {
    UpstreamOpened,
    ReverseFallback,
}

pub(super) fn should_open_upstream_prediction_return(
    native_mtp_enabled: bool,
    standalone_ngram_pipelining: bool,
) -> bool {
    native_mtp_enabled || standalone_ngram_pipelining
}

pub(super) fn open_upstream_prediction_return(request: &EmbeddedStageZeroGeneration<'_>) -> bool {
    let standalone_ngram_pipelining = !request.native_mtp_enabled
        && request.speculative.ngram.is_some()
        && request.draft.is_none()
        && request.speculative.verify_window.pipeline_depth > 1;
    if !should_open_upstream_prediction_return(
        request.native_mtp_enabled,
        standalone_ngram_pipelining,
    ) {
        return false;
    }
    let Some(prediction_return) = request.prediction_return.as_ref() else {
        return false;
    };
    match crate::binary_transport::direct_return::open_downstream_prediction_return_stream(
        request.config,
        request.ids.request_id,
        request.ids.session_id,
    ) {
        Ok(stream) => {
            prediction_return.attach_opened_stream(stream);
            true
        }
        Err(error) => {
            eprintln!("direct prediction return upstream-opened sink unavailable: {error:#}");
            false
        }
    }
}

pub(super) fn direct_prediction_return_path(
    verify_windows_enabled: bool,
    receiver_registered: bool,
    upstream_opened: bool,
) -> OpenAiResult<Option<DirectPredictionReturnPath>> {
    if !verify_windows_enabled {
        return Ok(None);
    }
    if !receiver_registered {
        return Err(OpenAiError::backend(
            "native MTP verify windows require direct prediction return",
        ));
    }
    Ok(Some(if upstream_opened {
        DirectPredictionReturnPath::UpstreamOpened
    } else {
        DirectPredictionReturnPath::ReverseFallback
    }))
}

pub(super) fn queued_active_tokens(windows: &VecDeque<PipelinedCompositeWindow>) -> usize {
    windows
        .iter()
        .filter(|window| !window.stale)
        .map(|window| window.planned_advance_tokens)
        .sum()
}

pub(super) fn can_seed_pipeline(windows: &VecDeque<PipelinedCompositeWindow>) -> bool {
    windows.iter().all(|window| window.stale)
}

/// Inclusive window-id range of the stale windows of `epoch`, if any.
pub(super) fn stale_window_id_range(
    windows: &VecDeque<PipelinedCompositeWindow>,
    epoch: u64,
) -> Option<(i32, i32)> {
    let mut range: Option<(i32, i32)> = None;
    for window in windows {
        if window.epoch == epoch && window.stale {
            let id = window.window.id;
            range = Some(match range {
                Some((min, max)) => (min.min(id), max.max(id)),
                None => (id, id),
            });
        }
    }
    range
}

pub(super) fn mark_epoch_stale(
    windows: &mut VecDeque<PipelinedCompositeWindow>,
    epoch: u64,
) -> usize {
    let mut marked = 0usize;
    for window in windows {
        if window.epoch == epoch && !window.stale {
            window.stale = true;
            marked = marked.saturating_add(1);
        }
    }
    marked
}

#[cfg(test)]
mod prediction_tests {
    use super::*;

    #[test]
    fn epoch_start_uses_its_own_boundary_prediction() {
        let predictions = compose_target_predictions(true, 2, None, &[11, 12, 13]).unwrap();

        assert_eq!(predictions, [11, 12, 13]);
    }

    #[test]
    fn continuation_prepends_the_prior_boundary_prediction() {
        let predictions = compose_target_predictions(false, 2, Some(11), &[12, 13]).unwrap();

        assert_eq!(predictions, [11, 12, 13]);
    }

    #[test]
    fn early_stopped_replies_pass_through_short_prediction_arrays() {
        let predictions = compose_target_predictions(false, 3, Some(11), &[12]).unwrap();

        assert_eq!(predictions, [11, 12]);
    }

    #[test]
    fn empty_replies_are_protocol_errors_on_both_paths() {
        assert!(compose_target_predictions(true, 2, None, &[]).is_err());
        assert!(compose_target_predictions(false, 2, Some(11), &[]).is_err());
    }

    #[test]
    fn continuation_starts_after_the_epoch_start_traversal_without_overlap() {
        let first = pipelined_window_layout(100, 4, 0, true, 10, &[11, 12]);
        let second = pipelined_window_layout(100, 4, 2, false, 12, &[13, 14]);

        assert_eq!(first.pos_start, 104);
        assert_eq!(first.input_tokens, [10, 11, 12]);
        assert_eq!(second.pos_start, 107);
        assert_eq!(second.input_tokens, [13, 14]);
        assert_eq!(first.pos_start + first.input_tokens.len(), second.pos_start);
    }
}

/// Extends the optimistic branch without adding speculative tokens to the
/// committed N-gram index. This removes the wait-for-empty bubble between
/// bounded proposal spans while preserving target-owned commit order.
pub(super) fn refill_pipeline_ngram_candidates(
    pipeline: &mut CompositeProposalPipeline,
    committed_tokens: &[i32],
    cached_ngram_proposer: &mut Option<HistoryNgramProposer>,
    max_tokens: usize,
) -> OpenAiResult<usize> {
    if max_tokens == 0 {
        return Ok(0);
    }
    let Some(cache) = cached_ngram_proposer.as_mut() else {
        return Ok(0);
    };
    let tokens = cache.propose(committed_tokens, pipeline.optimistic_suffix(), max_tokens)?;
    Ok(pipeline.append_ngram_candidates(&tokens))
}

pub(super) struct EmbeddedDecodeSummary<'a> {
    pub(super) decoded_tokens: usize,
    pub(super) stage0_compute_ms: f64,
    pub(super) runtime_lock_wait_ms: f64,
    pub(super) runtime_lock_wait_max_ms: f64,
    pub(super) runtime_lock_hold_ms: f64,
    pub(super) runtime_lock_hold_max_ms: f64,
    pub(super) runtime_lock_acquires: usize,
    pub(super) decode_batch_size_max: usize,
    pub(super) decode_batch_wait_ms: f64,
    pub(super) forward_write_ms: f64,
    pub(super) activation_encode_ms: f64,
    pub(super) output_activation_bytes: usize,
    pub(super) forward_activation_bytes: usize,
    pub(super) downstream_wait_ms: f64,
    pub(super) speculative_stats: &'a OpenAiSpeculativeStats,
    pub(super) native_mtp_stats: NativeMtpStats,
    pub(super) native_mtp_counters: NativeMtpDecodeCounters,
    pub(super) native_mtp_options: NativeMtpDecodeOptions,
    pub(super) verify_window_scheduler: &'a VerifyWindowScheduler,
}

impl StageOpenAiBackend {
    pub(super) fn generate_embedded_request_locally(
        &self,
        request: EmbeddedStageZeroGeneration<'_>,
        on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        self.generate_local_tokens(
            LocalGeneration {
                prompt_token_ids: request.prompt_token_ids,
                recurrent_cache_prefix_token_ids: None,
                max_tokens: request.max_tokens,
                sampling: request.sampling,
                chat_sampling_metadata: request.chat_sampling_metadata,
                speculative: request.speculative,
                native_mtp_enabled: request.native_mtp_enabled,
                hook_request: request.hook_request,
                hook_runtime: request.hook_runtime,
                cancellation: request.cancellation,
                ids: request.ids,
            },
            on_token,
        )
    }

    pub(super) fn record_embedded_decode_summary(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        cache_stats: &mut GenerationCacheStats,
        decode_timer: PhaseTimer,
        summary: EmbeddedDecodeSummary<'_>,
    ) {
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "llama_stage.decode_token_count".to_string(),
            json!(summary.decoded_tokens),
        );
        attrs.insert(
            "llama_stage.stage0_compute_ms".to_string(),
            json!(summary.stage0_compute_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(summary.runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_wait_max_ms".to_string(),
            json!(summary.runtime_lock_wait_max_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(summary.runtime_lock_hold_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_max_ms".to_string(),
            json!(summary.runtime_lock_hold_max_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_acquires".to_string(),
            json!(summary.runtime_lock_acquires),
        );
        attrs.insert(
            "llama_stage.decode_batch_size_max".to_string(),
            json!(summary.decode_batch_size_max),
        );
        attrs.insert(
            "llama_stage.decode_batch_wait_ms".to_string(),
            json!(summary.decode_batch_wait_ms),
        );
        attrs.insert(
            "llama_stage.forward_write_ms".to_string(),
            json!(summary.forward_write_ms),
        );
        attrs.insert(
            "llama_stage.activation_encode_ms".to_string(),
            json!(summary.activation_encode_ms),
        );
        attrs.insert(
            "llama_stage.output_activation_bytes".to_string(),
            json!(summary.output_activation_bytes),
        );
        attrs.insert(
            "llama_stage.forward_activation_bytes".to_string(),
            json!(summary.forward_activation_bytes),
        );
        attrs.insert(
            "llama_stage.downstream_wait_ms".to_string(),
            json!(summary.downstream_wait_ms),
        );
        request.speculative.insert_telemetry_attrs(&mut attrs);
        summary.speculative_stats.insert_attrs(&mut attrs);
        summary.native_mtp_stats.insert_attrs(&mut attrs);
        summary
            .native_mtp_counters
            .insert_summary_attrs(&mut attrs, summary.native_mtp_options);
        summary
            .verify_window_scheduler
            .insert_pipeline_telemetry_attrs(&mut attrs);

        cache_stats.native_mtp_stats = summary.native_mtp_stats;
        cache_stats.native_mtp_decode_telemetry = Some(NativeMtpDecodeTelemetry::new(
            summary.native_mtp_options,
            summary.native_mtp_counters,
        ));
        cache_stats.verify_window_pipeline_stats = Some(summary.verify_window_scheduler.stats());
        cache_stats.speculative_stats = Some(summary.speculative_stats.clone());
        cache_stats.predicted_ms = decode_timer.elapsed_ms();
        self.emit_openai_summary("stage.openai_decode", decode_timer, attrs);
    }

    pub(super) fn finish_embedded_generation_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        lane_pool: &PersistentStageLanePool,
        mut lane: PersistentStageLane,
        result: &OpenAiResult<()>,
        session_key: &str,
    ) -> OpenAiResult<()> {
        if result.is_err() {
            // The generation error may be the downstream peer disappearing.
            // A graceful Stop/ACK exchange would then turn the bounded decode
            // failure into an unbounded teardown wait. Retire the suspect lane
            // immediately; replacement uses its own bounded handshake. Drop
            // the old stream before opening its replacement so the remote
            // connection handler can reclaim the request's execution session
            // before the replacement admits another request on a one-lane
            // stage.
            self.drop_embedded_runtime_session(request, session_key);
            let lane_id = lane.id;
            drop(lane);
            lane_pool.replace_lane(lane_id);
            return Ok(());
        }

        let stop_result = write_stage_message(
            &mut lane.stream,
            &StageWireMessage::stop_with_identity(request.ids.request_id, request.ids.session_id),
        )
        .and_then(|_| recv_reply(&mut lane.stream).map(|reply| reply.kind))
        .and_then(|kind| {
            if kind == WireReplyKind::Ack {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("expected stop ACK, got {kind:?}"),
                ))
            }
        });
        self.drop_embedded_runtime_session(request, session_key);

        let lane_id = lane.id;
        let stop_result = stop_result.map_err(openai_io_error);
        match &stop_result {
            Ok(_) => lane_pool.return_lane(lane),
            Err(_) => {
                drop(lane);
                lane_pool.replace_lane(lane_id);
            }
        }
        stop_result?;
        Ok(())
    }

    fn drop_embedded_runtime_session(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session_key: &str,
    ) {
        let scheduler_session_key = session_key.to_string();
        let Ok(outcome) = self.iteration_scheduler.execute_runtime_timed(
            "embedded-session-drop",
            move |runtime| {
                runtime
                    .drop_session_timed(&scheduler_session_key)
                    .map_err(openai_backend_error)
            },
        ) else {
            return;
        };
        let drop_stats = outcome.value;
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(outcome.runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.session_reset_ms".to_string(),
            json!(drop_stats.reset_ms),
        );
        attrs.insert(
            "llama_stage.session_reset".to_string(),
            json!(drop_stats.reset_session),
        );
        attrs.insert(
            "llama_stage.lane_discarded".to_string(),
            json!(drop_stats.lane_discarded),
        );
        if let Some(reason) = drop_stats.lane_discard_reason.as_deref() {
            attrs.insert("llama_stage.lane_discard_reason".to_string(), json!(reason));
        }
        Self::insert_runtime_session_stats(
            &mut attrs,
            "llama_stage.runtime_sessions_after",
            &drop_stats.stats_after,
        );
        self.telemetry
            .emit_debug("stage.openai_session_stop", attrs);
    }
}

pub(super) fn decode_uses_context_sideband(
    context_token_ids: &[i32],
    current: i32,
    sideband_capacity: usize,
) -> bool {
    context_token_ids.len() <= sideband_capacity
        && context_token_ids.last().copied() == Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::{
        NativeMtpHybridProposal,
        speculative::{NgramProposalConfig, NgramProposerKind},
    };

    fn ngram_config() -> SpeculativeDecodeConfig {
        SpeculativeDecodeConfig {
            requested_strategy: "ngram".to_string(),
            effective_strategy: "ngram-suffix".to_string(),
            ngram: Some(NgramProposalConfig {
                kind: NgramProposerKind::Suffix,
                min_ngram: 2,
                max_ngram: 16,
                max_proposal_tokens: 4,
            }),
            ..SpeculativeDecodeConfig::default()
        }
    }

    #[test]
    fn restored_prefix_bypasses_history_ngram_for_that_request() {
        let config = ngram_config();

        let effective = speculation_after_prefix_restore(&config, true);

        assert!(matches!(effective, Cow::Owned(_)));
        assert!(effective.ngram.is_none());
        assert!(config.ngram.is_some());
    }

    #[test]
    fn cache_miss_preserves_configured_ngram() {
        let config = ngram_config();

        let effective = speculation_after_prefix_restore(&config, false);

        assert!(matches!(effective, Cow::Borrowed(_)));
        assert!(effective.ngram.is_some());
    }

    #[test]
    fn direct_return_falls_back_only_with_a_registered_receiver() {
        assert_eq!(
            direct_prediction_return_path(true, true, false).unwrap(),
            Some(DirectPredictionReturnPath::ReverseFallback)
        );
        assert!(direct_prediction_return_path(true, false, false).is_err());
        assert_eq!(
            direct_prediction_return_path(false, false, false).unwrap(),
            None
        );
    }

    #[test]
    fn direct_return_opens_for_native_mtp_or_pipelined_ngram() {
        assert!(!should_open_upstream_prediction_return(false, false));
        assert!(should_open_upstream_prediction_return(true, false));
        assert!(should_open_upstream_prediction_return(false, true));
    }

    #[test]
    fn refills_from_an_optimistic_suffix_without_indexing_it() {
        let committed = vec![1, 2, 3, 1, 2, 3, 1, 2];
        let mut cache = Some(HistoryNgramProposer::new_cache(2, 2).unwrap());
        let initial = cache.as_mut().unwrap().propose(&committed, &[], 2).unwrap();
        assert_eq!(initial, vec![3, 1]);
        let proposal = NativeMtpHybridProposal::from_parts(initial, 0, true);
        let mut pipeline = CompositeProposalPipeline::new(proposal, None);

        let appended =
            refill_pipeline_ngram_candidates(&mut pipeline, &committed, &mut cache, 2).unwrap();

        assert_eq!(appended, 2);
        assert_eq!(pipeline.proposal().tokens(), &[3, 1, 2, 3]);
        assert_eq!(pipeline.candidate_len(), 4);
        assert_eq!(
            cache.as_mut().unwrap().propose(&committed, &[], 2).unwrap(),
            vec![3, 1]
        );
    }
}
