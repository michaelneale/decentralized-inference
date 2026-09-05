use std::collections::BTreeMap;
use std::time::Duration;

use openai_frontend::{OpenAiError, OpenAiResult};
use serde_json::json;

use crate::frontend::generation::{LocalGeneration, StageOpenAiBackend, TokenControl};
use crate::frontend::linear_proposal::{
    LinearProposalDiscardReason, LinearProposalExecutionParams, LinearProposalQueryOutcome,
    LinearProposalQueryParams, execute_linear_proposal_with_terminal_discard,
    query_linear_proposal, report_linear_proposal_receipt,
};
use crate::frontend::{LinearProposalDisposition, LinearProposalSourceTelemetry};

use super::token_generation::{DecodeState, LinearProposalProgress};

impl StageOpenAiBackend {
    pub(super) fn try_execute_linear_proposal(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &mut DecodeState,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<LinearProposalProgress> {
        self.try_execute_linear_proposal_with_executor(
            request,
            session_id,
            state,
            emit_token,
            |backend, params, queried, cancellation, emit_token| {
                backend.execute_local_linear_proposal(params, queried, cancellation, emit_token)
            },
        )
    }

    fn try_execute_linear_proposal_with_executor<F>(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &mut DecodeState,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
        execute: F,
    ) -> OpenAiResult<LinearProposalProgress>
    where
        F: for<'a> FnOnce(
            &StageOpenAiBackend,
            LinearProposalExecutionParams<'a>,
            crate::frontend::linear_proposal::QueriedLinearProposal,
            Option<&'a openai_frontend::CancellationToken>,
            &'a mut dyn FnMut(i32) -> OpenAiResult<TokenControl>,
        ) -> OpenAiResult<
            Option<crate::frontend::linear_proposal::LinearProposalReceipt>,
        >,
    {
        let Some(config) = self.linear_proposal_ingress.as_ref() else {
            return Ok(LinearProposalProgress::NotUsed);
        };
        let Some(committed_token_ids) = state.linear_context_tokens.as_mut() else {
            return Ok(LinearProposalProgress::NotUsed);
        };
        if request
            .cancellation
            .is_some_and(openai_frontend::CancellationToken::is_cancelled)
        {
            return Err(OpenAiError::backend("request cancelled"));
        }
        let remaining_new_tokens =
            (request.max_tokens as usize).saturating_sub(state.decoded_tokens);
        // Prefill leaves the final prompt token undecoded. When whole-prompt
        // prefill also samples the first target token, decoded_tokens is one;
        // otherwise it is zero. Those two modes therefore share this position.
        let base_position = u64::try_from(
            request
                .prompt_token_ids
                .len()
                .saturating_sub(1)
                .checked_add(state.decoded_tokens)
                .ok_or_else(|| {
                    OpenAiError::backend("linear proposal base position exceeds usize")
                })?,
        )
        .map_err(|_| OpenAiError::backend("linear proposal base position exceeds u64"))?;
        let queried = match query_linear_proposal(
            config,
            LinearProposalQueryParams {
                request_id: request.ids.request_id,
                session_id: request.ids.session_id,
                prompt_token_count: request.prompt_token_ids.len(),
                decode_step: state.decoded_tokens,
                committed_token_count: request
                    .prompt_token_ids
                    .len()
                    .saturating_add(state.decoded_tokens),
                remaining_new_tokens,
                runtime_max_proposal_tokens: state.linear_proposal_max_tokens,
                pending_token_ids: state
                    .pending_linear_proposal_tokens
                    .clone()
                    .into_boxed_slice(),
            },
        ) {
            Ok(LinearProposalQueryOutcome::Skipped) => {
                return Ok(LinearProposalProgress::NotUsed);
            }
            Ok(outcome) => {
                // The source has synchronously received this delta; do not
                // resend it on the next proposal boundary.
                state.pending_linear_proposal_tokens.clear();
                match outcome {
                    LinearProposalQueryOutcome::Skipped => unreachable!("handled above"),
                    LinearProposalQueryOutcome::NoProposal { source_telemetry } => {
                        self.emit_linear_proposal_source_telemetry(
                            source_telemetry,
                            state.emit_token_debug,
                        );
                        None
                    }
                    LinearProposalQueryOutcome::DeadlineExceeded {
                        proposal_elapsed_us,
                        source_telemetry,
                    } => {
                        self.emit_linear_proposal_source_telemetry(
                            source_telemetry,
                            state.emit_token_debug,
                        );
                        let mut attrs = BTreeMap::new();
                        attrs.insert(
                            "llama_stage.linear_proposal.discard_reason".to_string(),
                            json!("deadline_exceeded"),
                        );
                        attrs.insert(
                            "llama_stage.linear_proposal.proposal_us".to_string(),
                            json!(proposal_elapsed_us),
                        );
                        self.telemetry
                            .emit("stage.openai_linear_proposal_late", attrs);
                        None
                    }
                    LinearProposalQueryOutcome::Ready(queried) => {
                        self.emit_linear_proposal_source_telemetry(
                            queried.source_telemetry,
                            state.emit_token_debug,
                        );
                        Some(queried)
                    }
                }
            }
            Err(error) => {
                state.linear_context_tokens = None;
                state.linear_proposal_max_tokens = 0;
                self.telemetry.emit(
                    "stage.openai_linear_proposal_disabled",
                    BTreeMap::from([(
                        "llama_stage.linear_proposal.error".to_string(),
                        json!(error.to_string()),
                    )]),
                );
                return Ok(LinearProposalProgress::NotUsed);
            }
        };
        if request
            .cancellation
            .is_some_and(openai_frontend::CancellationToken::is_cancelled)
        {
            return Err(OpenAiError::backend("request cancelled"));
        }
        let Some(queried) = queried else {
            return Ok(LinearProposalProgress::NotUsed);
        };
        let decision_id = queried.proposal.decision_id.clone();
        let receipt = execute_linear_proposal_with_terminal_discard(config, &decision_id, || {
            execute(
                self,
                LinearProposalExecutionParams {
                    request_id: request.ids.request_id,
                    request_session_id: request.ids.session_id,
                    session_id,
                    current: state.current,
                    base_position,
                    generated_len: state.decoded_tokens,
                    max_new_tokens: request.max_tokens as usize,
                    sampling: request.sampling,
                    chat_sampling_metadata: request.chat_sampling_metadata,
                    prompt_token_count: request.prompt_token_ids.len(),
                },
                queried,
                request.cancellation,
                emit_token,
            )
        })?;
        if receipt.is_none() {
            let discard_failed = config
                .source()
                .discard(&decision_id, LinearProposalDiscardReason::PositionMismatch)
                .is_err();
            if discard_failed {
                self.telemetry.emit(
                    "stage.openai_linear_proposal_discard_failed",
                    BTreeMap::from([(
                        "llama_stage.linear_proposal.discard_reason".to_string(),
                        json!("position_mismatch"),
                    )]),
                );
            }
        }
        let Some(receipt) = receipt else {
            return Ok(LinearProposalProgress::NotUsed);
        };
        if report_linear_proposal_receipt(config, &receipt).is_some() {
            let mut attrs = BTreeMap::new();
            receipt.insert_telemetry_attrs(&mut attrs);
            attrs.insert(
                "llama_stage.linear_proposal.report_outcome".to_string(),
                json!("failed"),
            );
            self.telemetry
                .emit("stage.openai_linear_proposal_report_failed", attrs);
        }
        let proposal_runtime_lock_wait_ms =
            Duration::from_micros(receipt.runtime_lock_wait_us).as_secs_f64() * 1_000.0;
        let proposal_runtime_lock_hold_ms =
            Duration::from_micros(receipt.runtime_lock_hold_us).as_secs_f64() * 1_000.0;
        state.runtime_lock_wait_ms += proposal_runtime_lock_wait_ms;
        state.runtime_lock_wait_max_ms = state
            .runtime_lock_wait_max_ms
            .max(proposal_runtime_lock_wait_ms);
        state.runtime_lock_hold_ms += proposal_runtime_lock_hold_ms;
        state.runtime_lock_hold_max_ms = state
            .runtime_lock_hold_max_ms
            .max(proposal_runtime_lock_hold_ms);
        state.runtime_lock_acquires = state
            .runtime_lock_acquires
            .saturating_add(receipt.runtime_lock_acquires);
        state.decoded_tokens = state
            .decoded_tokens
            .checked_add(receipt.committed_tokens.len())
            .ok_or_else(|| OpenAiError::backend("linear proposal decode count overflow"))?;
        state.current = *receipt
            .committed_tokens
            .last()
            .ok_or_else(|| OpenAiError::backend("linear proposal receipt committed no tokens"))?;
        append_pending_linear_proposal_tokens(
            &mut state.pending_linear_proposal_tokens,
            &receipt.committed_tokens,
        );
        committed_token_ids.extend_from_slice(&receipt.committed_tokens);
        let stopped_by_proposal = receipt.disposition == LinearProposalDisposition::Stopped;
        if state.emit_token_debug {
            let mut proposal_attrs = BTreeMap::new();
            receipt.insert_telemetry_attrs(&mut proposal_attrs);
            self.telemetry
                .emit_debug("stage.openai_linear_proposal", proposal_attrs);
        }
        if stopped_by_proposal || state.decoded_tokens >= request.max_tokens as usize {
            Ok(LinearProposalProgress::Stop)
        } else {
            Ok(LinearProposalProgress::Continue)
        }
    }

    fn emit_linear_proposal_source_telemetry(
        &self,
        source_telemetry: Option<LinearProposalSourceTelemetry>,
        emit_token_debug: bool,
    ) {
        let Some(source_telemetry) = source_telemetry else {
            return;
        };
        let mut attrs = BTreeMap::new();
        source_telemetry.insert_telemetry_attrs(&mut attrs);
        match source_telemetry.outcome {
            crate::frontend::LinearProposalSourceOutcome::Ready
            | crate::frontend::LinearProposalSourceOutcome::Abstained
                if emit_token_debug =>
            {
                self.telemetry
                    .emit_debug("stage.openai_linear_proposal_source", attrs);
            }
            crate::frontend::LinearProposalSourceOutcome::Ready
            | crate::frontend::LinearProposalSourceOutcome::Abstained => {}
            _ => self
                .telemetry
                .emit("stage.openai_linear_proposal_source", attrs),
        }
    }
}

fn append_pending_linear_proposal_tokens(pending: &mut Vec<i32>, committed_tokens: &[i32]) {
    if !committed_tokens.is_empty() && !pending.ends_with(committed_tokens) {
        pending.extend_from_slice(committed_tokens);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, atomic::AtomicUsize};

    use super::*;
    use crate::frontend::admission::GenerationTokenBudget;
    use crate::frontend::generation::{
        GenerationConcurrencyController, OpenAiBackendMode, OpenAiCacheHints, OpenAiGenerationIds,
    };
    use crate::frontend::iteration_scheduler::IterationScheduler;
    use crate::frontend::linear_proposal::{
        LinearProposal, LinearProposalDiscardReason, LinearProposalIngress,
        LinearProposalIngressConfig, LinearProposalQuery, LinearProposalReceipt,
        LinearProposalSourceResponse, OpaqueProposalDecisionId,
    };
    use crate::frontend::native_mtp::{NativeMtpDecodeOptions, NativeMtpVerifier};
    use crate::frontend::{EmbeddedOpenAiRequestDefaults, SpeculativeDecodeConfig};
    use crate::runtime_state::RuntimeState;
    use crate::telemetry::{Telemetry, TelemetryLevel};
    use anyhow::Result;
    use skippy_protocol::{LoadMode, StageConfig};
    use skippy_runtime::SamplingConfig;

    #[derive(Default)]
    struct PendingTokenIngress {
        pending: Mutex<Vec<Box<[i32]>>>,
        reports: Mutex<Vec<Box<[i32]>>>,
        proposals: AtomicUsize,
    }

    impl LinearProposalIngress for PendingTokenIngress {
        fn propose(&self, query: LinearProposalQuery) -> Result<LinearProposalSourceResponse> {
            self.pending.lock().unwrap().push(query.pending_token_ids);
            if self
                .proposals
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                == 0
            {
                Ok(LinearProposalSourceResponse::new(Some(
                    LinearProposal::new(OpaqueProposalDecisionId::new("test-decision")?, [41, 42]),
                )))
            } else {
                Ok(LinearProposalSourceResponse::new(None))
            }
        }

        fn report(&self, receipt: &LinearProposalReceipt) -> Result<()> {
            self.reports
                .lock()
                .unwrap()
                .push(receipt.committed_tokens.clone());
            Ok(())
        }

        fn discard(
            &self,
            _decision_id: &crate::frontend::linear_proposal::OpaqueProposalDecisionId,
            _reason: LinearProposalDiscardReason,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn accepted_proposal_tokens_are_pending_on_the_next_query() {
        let source = Arc::new(PendingTokenIngress::default());
        let config =
            LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();
        let stage_config = StageConfig {
            run_id: "linear-proposal-test".to_string(),
            topology_id: "linear-proposal-test".to_string(),
            model_id: "linear-proposal-test".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 128,
            lane_count: 1,
            n_batch: Some(4),
            n_ubatch: Some(4),
            n_gpu_layers: 0,
            mmap: Some(true),
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: Default::default(),
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: false,
            resident_tensor_names: Vec::new(),
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: false,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
            ..StageConfig::default()
        };
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
        let speculative = SpeculativeDecodeConfig::default();
        let telemetry = Telemetry::new(None, 1, stage_config.clone(), TelemetryLevel::Off);
        let iteration_scheduler =
            IterationScheduler::new(runtime.clone(), &stage_config, 1, true, telemetry.clone())
                .unwrap();
        let backend = StageOpenAiBackend {
            runtime: runtime.clone(),
            config: stage_config.clone(),
            telemetry,
            model_id: "linear-proposal-test".to_string(),
            default_max_tokens: 4,
            request_defaults: EmbeddedOpenAiRequestDefaults::default(),
            ctx_size: 128,
            mode: OpenAiBackendMode::LocalRuntime,
            draft: None,
            speculative_window: 0,
            adaptive_speculative_window: false,
            ngram_max: 0,
            speculative: speculative.clone(),
            generation_limit: Arc::new(GenerationConcurrencyController::fixed(1)),
            generation_queue_depth: Arc::new(AtomicUsize::new(0)),
            generation_queue_limit: 1,
            generation_admission_timeout: std::time::Duration::from_secs(10),
            generation_service_estimator: Arc::new(
                crate::frontend::GenerationServiceEstimator::new(1),
            ),
            generation_session_locks: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
            generation_token_budget: Arc::new(GenerationTokenBudget::new(128)),
            hook_policy: None,
            generation_receipt: None,
            generation_lifecycle: None,
            linear_proposal_ingress: Some(config),
            kv: None,
            iteration_scheduler,
        };
        let sampling = SamplingConfig::default();
        let ids =
            OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), None, false, None);
        let prompt_token_ids = [1, 2];
        let request = LocalGeneration {
            prompt_token_ids: &prompt_token_ids,
            recurrent_cache_prefix_token_ids: None,
            max_tokens: 5,
            sampling: &sampling,
            chat_sampling_metadata: None,
            speculative: &speculative,
            native_mtp_enabled: false,
            hook_request: None,
            hook_runtime: None,
            cancellation: None,
            ids: &ids,
        };
        let mut state = DecodeState {
            decoded_tokens: 0,
            current: 2,
            generated_token_ids: Vec::new(),
            stopped: false,
            runtime_lock_wait_ms: 0.0,
            runtime_lock_wait_max_ms: 0.0,
            runtime_lock_hold_ms: 0.0,
            runtime_lock_hold_max_ms: 0.0,
            runtime_lock_acquires: 0,
            runtime_sessions_before: None,
            runtime_sessions_after: None,
            hook_request: None,
            hook_runtime: None,
            generation_hooks_active: false,
            linear_proposal_max_tokens: 4,
            linear_context_tokens: Some(prompt_token_ids.to_vec()),
            pending_linear_proposal_tokens: Vec::new(),
            emit_token_debug: false,
            native_mtp_options: NativeMtpDecodeOptions::from_config(&speculative),
            native_mtp: NativeMtpVerifier::default(),
            native_mtp_span_admitted: false,
            post_prefill_hook_checked: false,
            last_mid_generation_hook_at: None,
            direct_iteration_channel: None,
        };
        let mut emitted = Vec::new();

        assert!(matches!(
            backend
                .try_execute_linear_proposal_with_executor(
                    &request,
                    "linear-proposal-session",
                    &mut state,
                    &mut |token| {
                        emitted.push(token);
                        Ok(TokenControl::Continue)
                    },
                    |_backend, _params, queried, _cancellation, _emit_token| {
                        Ok(Some(LinearProposalReceipt {
                            request_id: 7,
                            session_id: 8,
                            decision_id: queried.proposal.decision_id,
                            disposition: LinearProposalDisposition::FullAccept,
                            proposal_token_count: 2,
                            verification_rows: 3,
                            accepted_proposal_tokens: 2,
                            committed_tokens: [41, 42].into(),
                            verification_row_predictions: [41, 42, 43].into(),
                            canonical_prediction_count: 2,
                            generated_token_count: 2,
                            correction_or_boundary_token: Some(43),
                            base_position: 1,
                            position_after_verification: 4,
                            canonical_position: 3,
                            trimmed_rows: 1,
                            proposal_elapsed_us: 0,
                            verification_elapsed_us: 0,
                            repair_elapsed_us: 0,
                            total_elapsed_us: 0,
                            runtime_lock_wait_us: 0,
                            runtime_lock_hold_us: 0,
                            runtime_lock_acquires: 0,
                        }))
                    },
                )
                .unwrap(),
            LinearProposalProgress::Continue
        ));

        assert_eq!(state.pending_linear_proposal_tokens, vec![41, 42]);
        assert_eq!(
            state.linear_context_tokens.as_deref(),
            Some(&[1, 2, 41, 42][..])
        );

        assert!(matches!(
            backend
                .try_execute_linear_proposal_with_executor(
                    &request,
                    "linear-proposal-session",
                    &mut state,
                    &mut |_| Ok(TokenControl::Continue),
                    |_backend, _params, _queried, _cancellation, _emit_token| Ok(None),
                )
                .unwrap(),
            LinearProposalProgress::NotUsed
        ));
        assert_eq!(
            source.pending.lock().unwrap().as_slice(),
            &[
                Vec::<i32>::new().into_boxed_slice(),
                vec![41, 42].into_boxed_slice(),
            ]
        );
        assert_eq!(
            source.reports.lock().unwrap().as_slice(),
            &[vec![41, 42].into_boxed_slice()]
        );
        assert!(emitted.is_empty());
    }
}
