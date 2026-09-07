use std::net::TcpStream;

use openai_frontend::{OpenAiError, OpenAiResult};
use skippy_protocol::binary::{StageNativeMtpDraft, WireReplyKind};

use crate::frontend::embedded_execution::VerifyRetirement;

use super::super::{
    AdaptiveVerifyWindow, BufferedCompositeProposal, CompositeProposalProvider,
    EmbeddedStageZeroGeneration, HistoryNgramProposer, NativeMtpDecodeCounters,
    NativeMtpDecodeOptions, NativeMtpDraft, NativeMtpDraftOrigin, NativeMtpVerifier,
    NgramSidecarController, PendingNativeMtpDraft, PhaseTimer, StageOpenAiBackend, TokenControl,
    VerifyWindowMessageArgs, VerifyWindowScheduler, WireSamplingConfig,
    classify_native_mtp_verify_window, embedded_verify_window_message, ms_to_us,
    token_is_eog_with_runtime, verify_checkpoint_no_longer_needed,
};

/// Control signal returned after processing a batched native MTP verify step.
pub(in crate::frontend) enum NativeMtpVerifyWindowControl {
    /// The on_token callback returned Stop — outer loop should break.
    ReachedStop,
    /// Continue the outer decode loop normally.
    Continue,
    /// No native-MTP or N-gram candidate was available for this position.
    NoProposal,
}

impl StageOpenAiBackend {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::frontend) fn execute_native_mtp_verify_window(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        request_id: u64,
        session_id: u64,
        prefill_token_count: usize,
        wire_sampling: &Option<WireSamplingConfig>,
        native_mtp_options: &NativeMtpDecodeOptions,
        verify_window_scheduler: &mut VerifyWindowScheduler,
        pending_native_mtp_draft: Option<PendingNativeMtpDraft>,
        proposal_buffer: &mut Option<BufferedCompositeProposal>,
        cached_ngram_proposer: &mut Option<HistoryNgramProposer>,
        adaptive_verify_window: &mut AdaptiveVerifyWindow,
        current: &mut i32,
        decode_step: u32,
        // Mutable decode loop state
        decoded_tokens: &mut usize,
        context_tokens: &mut Vec<i32>,
        exact_replay_tokens: &mut Vec<i32>,
        native_mtp: &mut NativeMtpVerifier,
        native_mtp_counters: &mut NativeMtpDecodeCounters,
        native_mtp_reject_cooldown_remaining: &mut usize,
        native_mtp_suppress_cooldown_drafts_remaining: &mut usize,
        ngram_sidecar_controller: &mut NgramSidecarController,
        // Mutable decode accumulators
        decode_stage0_compute_ms: &mut f64,
        decode_runtime_lock_wait_ms: &mut f64,
        decode_runtime_lock_wait_max_ms: &mut f64,
        decode_runtime_lock_hold_ms: &mut f64,
        decode_runtime_lock_hold_max_ms: &mut f64,
        decode_runtime_lock_acquires: &mut usize,
        decode_forward_activation_encode_ms: &mut f64,
        decode_output_activation_bytes: &mut usize,
        decode_forward_activation_bytes: &mut usize,
        decode_forward_write_ms: &mut f64,
        decode_downstream_wait_ms: &mut f64,
        // Token emission callback
        on_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<NativeMtpVerifyWindowControl> {
        let verify_window_timer = self.telemetry.is_debug_enabled().then(PhaseTimer::start);
        let native_mtp_remaining = (request.max_tokens as usize).saturating_sub(*decoded_tokens);
        let native_mtp_draft_origin = pending_native_mtp_draft.as_ref().map(|draft| draft.origin);
        let native_mtp_draft_tokens = pending_native_mtp_draft
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
        if proposal_buffer.is_none() {
            let native_mtp_tokens =
                if native_mtp_draft_tokens.len() >= native_mtp_options.min_draft_tokens {
                    native_mtp_draft_tokens.as_slice()
                } else {
                    &[]
                };
            let proposal = CompositeProposalProvider::from_options(*native_mtp_options)
                .propose_with_ngram_extension(
                    native_mtp_tokens,
                    context_tokens,
                    native_mtp_remaining.saturating_sub(1),
                    ngram_sidecar_controller.extension_limit(
                        native_mtp_tokens,
                        native_mtp_remaining.saturating_sub(native_mtp_tokens.len() + 1),
                    ),
                    cached_ngram_proposer.as_mut(),
                )?;
            if proposal.tokens().is_empty() {
                return Ok(NativeMtpVerifyWindowControl::NoProposal);
            }
            *proposal_buffer = Some(BufferedCompositeProposal::new(proposal));
        }
        let requested_verify_width = proposal_buffer
            .as_ref()
            .map(|buffer| adaptive_verify_window.width(buffer.remaining_len()))
            .unwrap_or_default();
        let remaining_native_mtp_tokens = proposal_buffer
            .as_ref()
            .map(BufferedCompositeProposal::remaining_native_mtp_tokens)
            .unwrap_or_default();
        let verify_width = ngram_sidecar_controller.verify_width(requested_verify_width);
        let proposal_tokens = {
            let buffer = proposal_buffer
                .as_ref()
                .expect("proposal buffer initialized");
            buffer.verify_tokens(verify_width)
        };
        if proposal_tokens.is_empty() {
            return Ok(NativeMtpVerifyWindowControl::NoProposal);
        }
        let verify_inputs = native_mtp_verify_window_inputs(*current, &proposal_tokens);
        let window = verify_window_scheduler.open(
            prefill_token_count + *decoded_tokens,
            *decoded_tokens,
            verify_inputs.len(),
        )?;
        let message = embedded_verify_window_message(VerifyWindowMessageArgs {
            window_id: window.id,
            request_id,
            session_id,
            prompt_token_count: request.prompt_token_ids.len(),
            pos_start: prefill_token_count + *decoded_tokens,
            decode_step: *decoded_tokens,
            tokens: &verify_inputs,
            sampling: wire_sampling.clone(),
        })?;
        let verify = self.execute_embedded_stage_message(
            request,
            downstream,
            session_key,
            &message,
            &verify_inputs,
            WireReplyKind::PredictedTokens,
        )?;
        let completed = verify_window_scheduler.complete_next(verify.reply.window.window_id)?;
        if completed != window {
            return Err(OpenAiError::backend(
                "verify window scheduler lost FIFO state",
            ));
        }
        let native_mtp_verify_decision = classify_native_mtp_verify_window(
            &proposal_tokens,
            &verify.reply.predicted_tokens,
            *decoded_tokens,
            request.max_tokens as usize,
            |token| token_is_eog_with_runtime(&self.runtime, token),
        )?;
        let target_token = verify.reply.predicted_tokens[0];
        let verify_next_mtp_draft = next_native_mtp_draft(
            request.native_mtp_enabled,
            verify.reply.native_mtp_draft.clone(),
        );
        let native_mtp_decision = (!native_mtp_draft_tokens.is_empty()).then(|| {
            let span = native_mtp.observe_taken_draft_span(
                &native_mtp_draft_tokens,
                &verify.reply.predicted_tokens,
                ms_to_us(verify.elapsed_ms),
            );
            let verified_draft_count = span.accepted_count + usize::from(span.rejected);
            for index in 0..verified_draft_count {
                native_mtp_counters.observe_verify_window_verification(
                    native_mtp_draft_origin.expect("native MTP draft has origin"),
                    index < span.accepted_count,
                );
            }
            span.first_decision
        });
        let commit_token_count = native_mtp_verify_decision.commit_count;
        let consumed_positions = verify_inputs.len();
        let mut committed_positions = 0usize;
        let mut reached_stop = false;
        for token in verify
            .reply
            .predicted_tokens
            .iter()
            .copied()
            .take(commit_token_count)
        {
            *current = token;
            *decoded_tokens += 1;
            committed_positions += 1;
            exact_replay_tokens.push(*current);
            context_tokens.push(*current);
            if on_token(*current)? == TokenControl::Stop {
                reached_stop = true;
                break;
            }
            if *decoded_tokens >= request.max_tokens as usize {
                break;
            }
        }
        let fully_accepted_window = !native_mtp_verify_decision.rejected
            && native_mtp_verify_decision.accepted_proposal_tokens == proposal_tokens.len()
            && committed_positions == consumed_positions
            && !reached_stop;
        let checkpoint_no_longer_needed =
            verify_checkpoint_no_longer_needed(committed_positions, consumed_positions);
        if checkpoint_no_longer_needed {
            self.retire_verify_window(
                request,
                downstream,
                None,
                session_key,
                VerifyRetirement {
                    request_id,
                    session_id,
                    token_start: window.base_position,
                    token_count: verify_inputs.len(),
                },
            )?;
        }
        let decision_rejected_native_mtp_prefix = proposal_buffer.as_ref().is_some_and(|buffer| {
            buffer.native_mtp_prefix_rejected_after(
                native_mtp_verify_decision.accepted_proposal_tokens,
            )
        });
        let dependent_target_is_native = remaining_native_mtp_tokens > proposal_tokens.len();
        let (buffer_exhausted, accepted_proposal_tokens, dependent_target_rejected) = {
            let buffer = proposal_buffer.as_mut().expect("proposal buffer retained");
            let dependent_target_rejected = if fully_accepted_window {
                buffer.accept_window(
                    &proposal_tokens,
                    verify
                        .reply
                        .predicted_tokens
                        .get(proposal_tokens.len())
                        .copied(),
                )
            } else {
                buffer.reject_window(native_mtp_verify_decision.accepted_proposal_tokens);
                false
            };
            let accepted_proposal_tokens = buffer.accepted_tokens();
            let buffer_exhausted = buffer.is_empty();
            (
                buffer_exhausted,
                accepted_proposal_tokens,
                dependent_target_rejected,
            )
        };
        let native_mtp_prefix_rejected = decision_rejected_native_mtp_prefix
            || (dependent_target_rejected && dependent_target_is_native);
        let previous_verify_width = adaptive_verify_window.current_tokens();
        let window_adjusted = adaptive_verify_window.observe(fully_accepted_window);
        native_mtp_counters.observe_adaptive_verify_window(
            proposal_tokens.len(),
            previous_verify_width,
            adaptive_verify_window.current_tokens(),
        );
        if native_mtp_prefix_rejected && native_mtp_options.reject_cooldown_tokens > 0 {
            *native_mtp_reject_cooldown_remaining = native_mtp_options.reject_cooldown_tokens;
            *native_mtp_suppress_cooldown_drafts_remaining =
                native_mtp_options.suppress_cooldown_draft_limit;
            native_mtp.clear_pending_draft();
        }
        let verify_next_mtp_draft_available = verify_next_mtp_draft.is_some();
        let verify_next_mtp_draft_adopted = buffer_exhausted
            && fully_accepted_window
            && !native_mtp_prefix_rejected
            && *decoded_tokens < request.max_tokens as usize
            && verify_next_mtp_draft.is_some();
        native_mtp_counters.observe_verify_next_draft(
            verify_next_mtp_draft_available,
            verify_next_mtp_draft_adopted,
        );
        if verify_next_mtp_draft_adopted {
            native_mtp.observe_next_draft(
                verify_next_mtp_draft.clone(),
                NativeMtpDraftOrigin::VerifyNext,
            );
        }
        if buffer_exhausted {
            let buffer = proposal_buffer
                .take()
                .expect("empty proposal buffer retained");
            if ngram_sidecar_controller
                .observe_tail_outcome(buffer.proposal(), accepted_proposal_tokens)
            {
                native_mtp_counters.observe_ngram_tail_rejection();
            }
            native_mtp_counters
                .observe_hybrid_proposal(buffer.proposal(), buffer.accepted_tokens());
        }
        *decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
        *decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
        *decode_runtime_lock_wait_max_ms =
            decode_runtime_lock_wait_max_ms.max(verify.stats.runtime_lock_wait_ms);
        *decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
        *decode_runtime_lock_hold_max_ms =
            decode_runtime_lock_hold_max_ms.max(verify.stats.runtime_lock_hold_ms);
        *decode_runtime_lock_acquires += 1;
        *decode_forward_activation_encode_ms += verify.stats.activation_encode_ms;
        *decode_output_activation_bytes =
            decode_output_activation_bytes.saturating_add(verify.stats.output_activation_bytes);
        *decode_forward_activation_bytes =
            decode_forward_activation_bytes.saturating_add(verify.stats.forward_activation_bytes);
        *decode_forward_write_ms += verify.stats.forward_write_ms;
        *decode_downstream_wait_ms += verify.stats.downstream_wait_ms;

        if let Some(verify_window_timer) = verify_window_timer {
            let mut token_attrs = self.openai_attrs(request.ids);
            token_attrs.insert(
                "llama_stage.decode_step".to_string(),
                serde_json::json!(decode_step),
            );
            token_attrs.insert(
                "llama_stage.message_kind".to_string(),
                serde_json::json!("VerifyWindow"),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_window_batch".to_string(),
                serde_json::json!(true),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verification".to_string(),
                serde_json::json!(native_mtp_decision.map_or("ngram", |decision| decision.label())),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_elapsed_ms".to_string(),
                serde_json::json!(verify.elapsed_ms),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.draft_tokens".to_string(),
                serde_json::json!(native_mtp_draft_tokens),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.pending_origin".to_string(),
                serde_json::json!(
                    native_mtp_draft_origin.map_or("ngram", NativeMtpDraftOrigin::label)
                ),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.target_token".to_string(),
                serde_json::json!(target_token),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.accepted_count".to_string(),
                serde_json::json!(native_mtp_verify_decision.accepted_proposal_tokens),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.hybrid_proposal_len".to_string(),
                serde_json::json!(proposal_tokens.len()),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_window_width".to_string(),
                serde_json::json!(proposal_tokens.len()),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_window_next_width".to_string(),
                serde_json::json!(adaptive_verify_window.current_tokens()),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_window_adjusted".to_string(),
                serde_json::json!(window_adjusted),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_next_draft_available".to_string(),
                serde_json::json!(verify_next_mtp_draft_available),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_next_draft_adopted".to_string(),
                serde_json::json!(verify_next_mtp_draft_adopted),
            );
            if let Some(next_draft) = verify_next_mtp_draft.as_ref() {
                token_attrs.insert(
                    "llama_stage.native_mtp.verify_next_draft_tokens".to_string(),
                    serde_json::json!(next_draft.tokens),
                );
                token_attrs.insert(
                    "llama_stage.native_mtp.verify_next_draft_compute_us".to_string(),
                    serde_json::json!(next_draft.proposal_compute_us),
                );
            }
            token_attrs.insert(
                "llama_stage.native_mtp.consumed_positions".to_string(),
                serde_json::json!(consumed_positions),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.committed_positions".to_string(),
                serde_json::json!(committed_positions),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.dependent_target_rejected".to_string(),
                serde_json::json!(dependent_target_rejected),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.reject_cooldown_tokens".to_string(),
                serde_json::json!(native_mtp_options.reject_cooldown_tokens),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.reject_cooldown_remaining".to_string(),
                serde_json::json!(*native_mtp_reject_cooldown_remaining),
            );
            token_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                serde_json::json!(verify.stats.stage0_compute_ms),
            );
            token_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                serde_json::json!(verify.stats.runtime_lock_wait_ms),
            );
            token_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                serde_json::json!(verify.stats.runtime_lock_hold_ms),
            );
            token_attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                serde_json::json!(verify.stats.activation_encode_ms),
            );
            token_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                serde_json::json!(verify.stats.forward_write_ms),
            );
            token_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                serde_json::json!(verify.stats.downstream_wait_ms),
            );
            token_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                serde_json::json!(verify.stats.output_activation_bytes),
            );
            token_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                serde_json::json!(verify.stats.forward_activation_bytes),
            );
            self.emit_openai_phase(
                "stage.openai_native_mtp_verify",
                verify_window_timer,
                token_attrs,
            );
        }

        if reached_stop {
            return Ok(NativeMtpVerifyWindowControl::ReachedStop);
        }
        Ok(NativeMtpVerifyWindowControl::Continue)
    }
}

fn native_mtp_verify_window_inputs(current: i32, proposals: &[i32]) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len().saturating_add(1));
    tokens.push(current);
    tokens.extend_from_slice(proposals);
    tokens
}

fn next_native_mtp_draft(
    native_mtp_enabled: bool,
    stage_draft: Option<StageNativeMtpDraft>,
) -> Option<NativeMtpDraft> {
    native_mtp_enabled
        .then(|| stage_draft.map(NativeMtpDraft::from_stage_draft))
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::{native_mtp_verify_window_inputs, next_native_mtp_draft};
    use skippy_protocol::binary::StageNativeMtpDraft;

    #[test]
    fn verify_window_inputs_include_every_native_mtp_proposal() {
        assert_eq!(native_mtp_verify_window_inputs(10, &[11, 12]), [10, 11, 12]);
    }

    #[test]
    fn pure_ngram_verify_does_not_capture_a_native_mtp_draft() {
        assert_eq!(
            next_native_mtp_draft(
                false,
                Some(StageNativeMtpDraft {
                    token_ids: vec![11],
                    proposal_compute_us: 12,
                }),
            ),
            None
        );
    }

    #[test]
    fn native_mtp_verify_captures_the_next_native_draft() {
        let draft = next_native_mtp_draft(
            true,
            Some(StageNativeMtpDraft {
                token_ids: vec![11],
                proposal_compute_us: 12,
            }),
        )
        .expect("native MTP draft should be retained");

        assert_eq!(draft.tokens, vec![11]);
        assert_eq!(draft.proposal_compute_us, 12);
    }
}
