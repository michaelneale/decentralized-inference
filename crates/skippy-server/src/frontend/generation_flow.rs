mod text_generation;

use crate::binary_transport::forwarded_stage_message_timed;
use crate::binary_transport::write_stage_message_conditioned;
use crate::frontend::generation::GeneratedText;
use crate::frontend::generation::GenerationCacheStats;
use crate::frontend::generation::GenerationTokenLimit;
use crate::frontend::generation::OpenAiBackendMode;
use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::PreparedGenerationPrompt;
use crate::frontend::generation::SplitMultimodalGeneration;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::TextGenerationCollector;
use crate::frontend::generation::TokenControl;
use crate::frontend::generation::emulation_generation_active;
use crate::frontend::generation_receipt::GenerationLifecycleState;
use crate::frontend::generation_receipt::complete_generation_before_cleanup;
use crate::frontend::local_generation::LocalGenerationReceiptFinalization;
use crate::frontend::request::wire_sampling_config;
use crate::frontend::util::generation_stop_values;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::openai_io_error;
use crate::frontend::wire_messages::MultimodalPrefillArgs;
use crate::frontend::wire_messages::ReusableDecodeMessage;
use crate::frontend::wire_messages::ReusableDecodeMessageArgs;
use crate::frontend::wire_messages::generation_config_message;
use crate::frontend::wire_messages::multimodal_prefill_message;
use crate::frontend::{GenerationCommit, GenerationStart};
use crate::kv_integration::proactive_eviction_attrs;
use anyhow::anyhow;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_protocol::binary::StageWireMessage;
use skippy_protocol::binary::WireReplyKind;
use skippy_protocol::binary::recv_reply;
use skippy_protocol::binary::write_stage_message;
use skippy_runtime::SamplingConfig;
use std::sync::Arc;

struct LocalSessionCleanupGuard<F: FnOnce()> {
    cleanup: Option<F>,
}

impl<F: FnOnce()> LocalSessionCleanupGuard<F> {
    fn new(cleanup: F) -> Self {
        Self {
            cleanup: Some(cleanup),
        }
    }

    fn cleanup(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }
    }

    fn run<T, E>(cleanup: F, operation: impl FnOnce() -> Result<T, E>) -> (Result<T, E>, Self) {
        let guard = Self::new(cleanup);
        let result = operation();
        (result, guard)
    }
}

impl<F: FnOnce()> Drop for LocalSessionCleanupGuard<F> {
    fn drop(&mut self) {
        self.cleanup();
    }
}

impl StageOpenAiBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn generate_multimodal_text(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<&openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<ChatCompletionRequest>,
        hook_runtime: Option<tokio::runtime::Handle>,
        cancellation: Option<&openai_frontend::CancellationToken>,
        ids: OpenAiGenerationIds,
        on_text_chunk: impl FnMut(&str) -> OpenAiResult<()>,
    ) -> OpenAiResult<GeneratedText> {
        match self.mode.clone() {
            OpenAiBackendMode::EmbeddedStageZero {
                config,
                activation_width,
                downstream_wire_condition,
                lane_pool,
                prediction_returns,
                ..
            } if config.downstream.is_some() => {
                let lane_pool = lane_pool.ok_or_else(|| {
                    OpenAiError::backend("embedded stage 0 has no downstream lane pool")
                })?;
                let prediction_return = prediction_returns
                    .as_ref()
                    .map(|hub| hub.register(ids.request_id, ids.session_id))
                    .transpose()
                    .map_err(openai_backend_error)?;
                let emulation_active = emulation_generation_active(hook_request.as_ref(), &prompt);
                return self.generate_split_multimodal_text(
                    SplitMultimodalGeneration {
                        prompt,
                        max_tokens,
                        stop,
                        sampling,
                        cancellation,
                        ids,
                        config,
                        activation_width,
                        downstream_wire_condition,
                        lane_pool,
                        prediction_return,
                        emulation_active,
                    },
                    on_text_chunk,
                );
            }
            _ => {}
        }

        match &self.mode {
            OpenAiBackendMode::LocalRuntime => {}
            OpenAiBackendMode::EmbeddedStageZero { config, .. } if config.downstream.is_none() => {}
            OpenAiBackendMode::EmbeddedStageZero { .. } => {
                return Err(OpenAiError::unsupported(
                    "multimodal requests require an embedded stage-0 runtime",
                ));
            }
        }

        // Media embeddings do not have token IDs. The receipt binds the exact
        // target-tokenized rendered prompt text that selects their positions.
        let receipt_prompt_token_ids = self
            .generation_receipt
            .as_ref()
            .map(|_| self.tokenize(&prompt.text))
            .transpose()?
            .map(Arc::<[i32]>::from);
        let lifecycle_prompt_token_ids = self
            .generation_lifecycle
            .as_ref()
            .map(|_| self.tokenize(&prompt.text))
            .transpose()?
            .unwrap_or_default();
        let mut lifecycle = GenerationLifecycleState::new(
            self.generation_lifecycle.as_ref(),
            ids.request_id,
            ids.session_id,
            ids.agent_session_id.clone(),
            ids.frontend_request_id,
            &lifecycle_prompt_token_ids,
        );
        let stop_value_storage =
            generation_stop_values(stop, prompt.chat_parse_metadata.as_deref());
        let stop_values = stop_value_storage
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let emulation_active = emulation_generation_active(hook_request.as_ref(), &prompt);
        let session_id = ids.session_label.clone();
        let signal_window_tokens = self.generation_signal_window_tokens();
        let mut prefill_token_count = 0usize;
        let mut receipt_observation = None;
        let mut receipt_started = false;
        let mut receipt_cancelled = false;
        let mut receipt_model_generation_elapsed = None;
        let mut collector =
            TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk)?
                .with_emulation_stop(emulation_active)
                .with_ignore_eos(sampling.ignore_eos);
        let (result, mut session_cleanup) = LocalSessionCleanupGuard::run(
            || self.cleanup_local_generation_session(&session_id, &ids),
            || {
                let prefill_timer = PhaseTimer::start();
                let (prefill, mut token_signal, mut signal_window) = {
                    let scheduler_session_id = session_id.clone();
                    let scheduler_prompt = prompt.clone();
                    let scheduler_sampling = sampling.clone();
                    let outcome = self.iteration_scheduler.execute_runtime_timed(
                        "openai-media-prefill",
                        move |runtime| {
                            if !runtime.has_media_projector() {
                                return Err(OpenAiError::invalid_request(
                                    "multimodal request requires a configured projector",
                                ));
                            }
                            let runtime_sessions_before = runtime.session_stats();
                            let prefill = runtime
                                .prefill_media(
                                    &scheduler_session_id,
                                    &scheduler_prompt.text,
                                    &scheduler_prompt.media,
                                    scheduler_sampling.enabled.then_some(&scheduler_sampling),
                                )
                                .map_err(openai_backend_error)?;
                            let token_signal =
                                runtime.last_token_signal(&scheduler_session_id).ok();
                            let signal_window = runtime
                                .signal_window(&scheduler_session_id, signal_window_tokens)
                                .ok();
                            let runtime_sessions_after = runtime.session_stats();
                            Ok((
                                prefill,
                                token_signal,
                                signal_window,
                                runtime_sessions_before,
                                runtime_sessions_after,
                            ))
                        },
                    )?;
                    let lock_wait_ms = outcome.runtime_lock_wait_ms;
                    let runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
                    let (
                        prefill,
                        token_signal,
                        signal_window,
                        runtime_sessions_before,
                        runtime_sessions_after,
                    ) = outcome.value;
                    let mut attrs = self.openai_attrs(&ids);
                    attrs.insert(
                        "llama_stage.prefill_token_count".to_string(),
                        json!(prefill.token_count),
                    );
                    attrs.insert(
                        "llama_stage.prefill_position".to_string(),
                        json!(prefill.position),
                    );
                    attrs.insert(
                        "llama_stage.media_item_count".to_string(),
                        json!(prompt.media.len()),
                    );
                    attrs.insert(
                        "llama_stage.runtime_lock_wait_ms".to_string(),
                        json!(lock_wait_ms),
                    );
                    attrs.insert(
                        "llama_stage.runtime_lock_hold_ms".to_string(),
                        json!(runtime_lock_hold_ms),
                    );
                    attrs.insert("llama_stage.runtime_lock_acquires".to_string(), json!(1));
                    Self::insert_runtime_session_stats(
                        &mut attrs,
                        "llama_stage.runtime_sessions_before",
                        &runtime_sessions_before,
                    );
                    Self::insert_runtime_session_stats(
                        &mut attrs,
                        "llama_stage.runtime_sessions_after",
                        &runtime_sessions_after,
                    );
                    self.emit_openai_phase("stage.openai_media_prefill", prefill_timer, attrs);
                    (prefill, token_signal, signal_window)
                };
                prefill_token_count = prefill.token_count;
                let max_tokens = max_tokens.resolve(prefill.position as usize, self.ctx_size)?;
                receipt_observation = self.generation_receipt.as_ref().map(|config| {
                    config.observation(
                        usize::try_from(max_tokens)
                            .expect("supported targets represent u32 token budgets as usize"),
                    )
                });

                // Proactive eviction: free one native decode batch worth of resident
                // prefix KV cells for grammar-triggered retries during the coming
                // decode loop.
                let mut proactive_eviction_status = "disabled";
                let mut proactive_eviction_error_kind_attr = None;
                let mut proactive_eviction_target_tokens = 0_u64;
                let mut proactive_evicted_entries = 0_usize;
                let mut proactive_evicted_tokens = 0_u64;
                let mut proactive_eviction_error = None;
                if let Some(kv) = self.kv.as_ref() {
                    let scheduler_kv = Arc::clone(kv);
                    let scheduler_session_id = session_id.clone();
                    match self.iteration_scheduler.execute_runtime(
                        "openai-proactive-eviction",
                        move |runtime| {
                            scheduler_kv
                                .evict_resident_prefix_for_decode_batch(
                                    runtime,
                                    &scheduler_session_id,
                                )
                                .map_err(|error| {
                                    openai_frontend::OpenAiError::backend(error.to_string())
                                })
                        },
                    ) {
                        Ok(eviction) => {
                            proactive_eviction_status = if eviction.evicted_entries > 0 {
                                "evicted"
                            } else {
                                "noop"
                            };
                            proactive_eviction_target_tokens = eviction.target_tokens;
                            proactive_evicted_entries = eviction.evicted_entries;
                            proactive_evicted_tokens = eviction.evicted_tokens;
                        }
                        Err(error) => {
                            proactive_eviction_status = "error";
                            proactive_eviction_error_kind_attr = Some("scheduler_runtime_error");
                            proactive_eviction_error = Some(anyhow!(error.to_string()).context(
                                "evict resident-prefix KV before multimodal OpenAI decode",
                            ));
                        }
                    }
                }
                self.telemetry.emit(
                    "stage.openai_kv_record_decision",
                    proactive_eviction_attrs(
                        proactive_eviction_status,
                        proactive_eviction_error_kind_attr,
                        proactive_eviction_target_tokens,
                        proactive_evicted_entries,
                        proactive_evicted_tokens,
                    ),
                );
                if let Some(error) = proactive_eviction_error {
                    return Err(openai_backend_error(error));
                }

                if let (Some(config), Some(prompt_token_ids)) = (
                    self.generation_receipt.as_ref(),
                    receipt_prompt_token_ids.as_ref(),
                ) {
                    config.begin(GenerationStart {
                        request_id: ids.request_id,
                        session_id: ids.session_id,
                        agent_session_id: ids.agent_session_id.clone(),
                        prompt_token_ids: Arc::clone(prompt_token_ids),
                        frontend_request_id: ids.frontend_request_id,
                    });
                    receipt_started = true;
                }

                let decode_timer = PhaseTimer::start();
                let mut decoded_tokens = 0usize;
                let mut current = prefill.first_token;
                let mut runtime_lock_wait_ms = 0.0;
                let mut runtime_lock_wait_max_ms = 0.0_f64;
                let mut runtime_lock_hold_ms = 0.0;
                let mut runtime_lock_hold_max_ms = 0.0_f64;
                let mut runtime_lock_acquires = 0usize;
                let mut runtime_sessions_before = None;
                let mut runtime_sessions_after = None;
                let mut hook_request = hook_request;
                let hook_runtime = hook_runtime;
                let mut post_prefill_hook_checked = false;
                let mut last_mid_generation_hook_at = None;

                while decoded_tokens < max_tokens as usize {
                    if cancellation.is_some_and(openai_frontend::CancellationToken::is_cancelled) {
                        receipt_cancelled = true;
                        break;
                    }
                    if let Some(injected_current) = self.maybe_run_generation_hooks(
                        &session_id,
                        &mut hook_request,
                        hook_runtime.as_ref(),
                        decoded_tokens,
                        &mut post_prefill_hook_checked,
                        &mut last_mid_generation_hook_at,
                        token_signal.take(),
                        signal_window.take(),
                    )? {
                        current = injected_current;
                        continue;
                    }
                    if let Some(observation) = receipt_observation.as_mut() {
                        observation.record_token(current, ids.request_started_at.elapsed());
                    }
                    if let Some(config) = self.generation_receipt.as_ref() {
                        config.committed(GenerationCommit {
                            request_id: ids.request_id,
                            session_id: ids.session_id,
                            generated_token_count: decoded_tokens.saturating_add(1),
                            token_ids: vec![current].into_boxed_slice(),
                        });
                    }
                    lifecycle.commit(current, ids.request_started_at.elapsed());
                    if collector.push_token(current)? == TokenControl::Stop {
                        if let Some(observation) = receipt_observation.as_mut() {
                            observation.mark_callback_stop();
                        }
                        lifecycle.mark_callback_stop();
                        decoded_tokens += 1;
                        break;
                    }
                    decoded_tokens += 1;
                    if decoded_tokens >= max_tokens as usize {
                        break;
                    }

                    let token_timer = PhaseTimer::start();
                    let token_runtime_lock_wait_ms;
                    let token_runtime_lock_hold_ms;
                    let token_signal_next;
                    let signal_window_next;
                    let decode_step = decoded_tokens;
                    current = {
                        let scheduler_session_id = session_id.clone();
                        let scheduler_sampling = sampling.clone();
                        let outcome = self.iteration_scheduler.execute_runtime_timed(
                            "openai-media-decode",
                            move |runtime| {
                                let sessions_before = runtime.session_stats();
                                let predicted = runtime
                                    .decode_sampled(
                                        &scheduler_session_id,
                                        current,
                                        scheduler_sampling.enabled.then_some(&scheduler_sampling),
                                    )
                                    .map_err(openai_backend_error)?;
                                let token_signal =
                                    runtime.last_token_signal(&scheduler_session_id).ok();
                                let signal_window = runtime
                                    .signal_window(&scheduler_session_id, signal_window_tokens)
                                    .ok();
                                let sessions_after = runtime.session_stats();
                                Ok((
                                    predicted,
                                    token_signal,
                                    signal_window,
                                    sessions_before,
                                    sessions_after,
                                ))
                            },
                        )?;
                        token_runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
                        runtime_lock_wait_ms += token_runtime_lock_wait_ms;
                        runtime_lock_wait_max_ms =
                            runtime_lock_wait_max_ms.max(token_runtime_lock_wait_ms);
                        runtime_lock_acquires += 1;
                        token_runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
                        runtime_lock_hold_ms += token_runtime_lock_hold_ms;
                        runtime_lock_hold_max_ms =
                            runtime_lock_hold_max_ms.max(token_runtime_lock_hold_ms);
                        let (predicted, next_signal, next_window, sessions_before, sessions_after) =
                            outcome.value;
                        runtime_sessions_before.get_or_insert(sessions_before);
                        runtime_sessions_after = Some(sessions_after);
                        token_signal_next = next_signal;
                        signal_window_next = next_window;
                        predicted
                    };
                    token_signal = token_signal_next;
                    signal_window = signal_window_next;
                    let mut token_attrs = self.openai_attrs(&ids);
                    token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                    token_attrs.insert(
                        "llama_stage.stage0_compute_ms".to_string(),
                        json!(token_timer.elapsed_ms()),
                    );
                    token_attrs.insert(
                        "llama_stage.runtime_lock_wait_ms".to_string(),
                        json!(token_runtime_lock_wait_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.runtime_lock_hold_ms".to_string(),
                        json!(token_runtime_lock_hold_ms),
                    );
                    token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                    token_attrs
                        .insert("llama_stage.message_kind".to_string(), json!("DecodeToken"));
                    self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
                }
                let mut attrs = self.openai_attrs(&ids);
                attrs.insert(
                    "llama_stage.decode_token_count".to_string(),
                    json!(decoded_tokens),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_wait_max_ms".to_string(),
                    json!(runtime_lock_wait_max_ms),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(runtime_lock_hold_ms),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_hold_max_ms".to_string(),
                    json!(runtime_lock_hold_max_ms),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_acquires".to_string(),
                    json!(runtime_lock_acquires),
                );
                if let Some(stats) = runtime_sessions_before.as_ref() {
                    Self::insert_runtime_session_stats(
                        &mut attrs,
                        "llama_stage.runtime_sessions_before",
                        stats,
                    );
                }
                if let Some(stats) = runtime_sessions_after.as_ref() {
                    Self::insert_runtime_session_stats(
                        &mut attrs,
                        "llama_stage.runtime_sessions_after",
                        stats,
                    );
                }
                receipt_model_generation_elapsed = Some(decode_timer.start_instant.elapsed());
                self.emit_openai_summary("stage.openai_decode", decode_timer, attrs);
                Ok(())
            },
        );
        let generation_succeeded = result.is_ok();
        if receipt_cancelled {
            lifecycle.mark_cancelled();
        }
        complete_generation_before_cleanup(
            result,
            || {
                let receipt_result = if receipt_started {
                    self.finalize_generation_receipt(
                        LocalGenerationReceiptFinalization {
                            session_label: &session_id,
                            request_id: ids.request_id,
                            session_id: ids.session_id,
                            agent_session_id: ids.agent_session_id.as_deref(),
                            prompt_token_ids: receipt_prompt_token_ids.unwrap_or_default(),
                            observation: receipt_observation,
                            cancelled: receipt_cancelled,
                            model_generation_elapsed: receipt_model_generation_elapsed,
                            frontend_request_id: ids.frontend_request_id,
                        },
                        generation_succeeded,
                    )
                } else {
                    Ok(())
                };
                lifecycle.finish(generation_succeeded);
                receipt_result
            },
            || session_cleanup.cleanup(),
        )?;
        collector.finish(prefill_token_count, GenerationCacheStats::default())
    }

    pub(super) fn generate_split_multimodal_text(
        &self,
        request: SplitMultimodalGeneration<'_>,
        on_text_chunk: impl FnMut(&str) -> OpenAiResult<()>,
    ) -> OpenAiResult<GeneratedText> {
        let stop_value_storage =
            generation_stop_values(request.stop, request.prompt.chat_parse_metadata.as_deref());
        let stop_values = stop_value_storage
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let mut collector =
            TextGenerationCollector::new(self.runtime.clone(), stop_values, on_text_chunk)?
                .with_emulation_stop(request.emulation_active)
                .with_ignore_eos(request.sampling.ignore_eos);
        let wire_sampling = wire_sampling_config(&request.sampling);
        let session_id = request.ids.session_id;
        let request_id = request.ids.request_id;
        let session_key = session_id.to_string();
        let receipt_prompt_token_ids = self
            .generation_receipt
            .as_ref()
            .map(|_| self.tokenize(&request.prompt.text))
            .transpose()?
            .map(Arc::<[i32]>::from);
        let lifecycle_prompt_token_ids = self
            .generation_lifecycle
            .as_ref()
            .map(|_| self.tokenize(&request.prompt.text))
            .transpose()?
            .unwrap_or_default();
        let mut lane = request.lane_pool.checkout(&request.ids)?;
        let mut lifecycle = GenerationLifecycleState::new(
            self.generation_lifecycle.as_ref(),
            request.ids.request_id,
            request.ids.session_id,
            request.ids.agent_session_id.clone(),
            request.ids.frontend_request_id,
            &lifecycle_prompt_token_ids,
        );
        if let (Some(config), Some(prompt_token_ids)) = (
            self.generation_receipt.as_ref(),
            receipt_prompt_token_ids.as_ref(),
        ) {
            config.begin(GenerationStart {
                request_id,
                session_id,
                agent_session_id: request.ids.agent_session_id.clone(),
                prompt_token_ids: Arc::clone(prompt_token_ids),
                frontend_request_id: request.ids.frontend_request_id,
            });
        }

        let mut prompt_tokens = 0usize;
        let mut receipt_observation = None;
        let mut receipt_cancelled = false;
        let mut receipt_model_generation_elapsed = None;
        let result = (|| {
            let prefill_timer = PhaseTimer::start();
            let prefill = {
                let scheduler_session_key = session_key.clone();
                let scheduler_prompt = request.prompt.clone();
                let outcome = self.iteration_scheduler.execute_runtime_timed(
                    "embedded-media-prefill",
                    move |runtime| {
                        if !runtime.has_media_projector() {
                            return Err(OpenAiError::invalid_request(
                                "multimodal request requires a configured projector",
                            ));
                        }
                        let runtime_sessions_before = runtime.session_stats();
                        let prefill = runtime
                            .prefill_media_frame(
                                &scheduler_session_key,
                                &scheduler_prompt.text,
                                &scheduler_prompt.media,
                            )
                            .map_err(openai_backend_error)?;
                        let runtime_sessions_after = runtime.session_stats();
                        Ok((prefill, runtime_sessions_before, runtime_sessions_after))
                    },
                )?;
                let lock_wait_ms = outcome.runtime_lock_wait_ms;
                let runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
                let (prefill, runtime_sessions_before, runtime_sessions_after) = outcome.value;
                let mut attrs = self.openai_attrs(&request.ids);
                attrs.insert(
                    "llama_stage.prefill_token_count".to_string(),
                    json!(prefill.token_count),
                );
                attrs.insert(
                    "llama_stage.prefill_position".to_string(),
                    json!(prefill.position),
                );
                attrs.insert(
                    "llama_stage.media_item_count".to_string(),
                    json!(request.prompt.media.len()),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(runtime_lock_hold_ms),
                );
                attrs.insert("llama_stage.runtime_lock_acquires".to_string(), json!(1));
                attrs.insert(
                    "llama_stage.output_activation_bytes".to_string(),
                    json!(prefill.output.payload.len()),
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_before",
                    &runtime_sessions_before,
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    &runtime_sessions_after,
                );
                self.emit_openai_phase("stage.openai_media_prefill", prefill_timer, attrs);
                prefill
            };
            prompt_tokens = prefill.token_count;
            let max_tokens = request
                .max_tokens
                .resolve(prefill.position as usize, self.ctx_size)?;
            receipt_observation = self.generation_receipt.as_ref().map(|config| {
                config.observation(
                    usize::try_from(max_tokens)
                        .expect("supported targets represent u32 token budgets as usize"),
                )
            });

            let message = generation_config_message(
                request_id,
                session_id,
                prefill.token_count,
                wire_sampling.clone(),
                request.prompt.chat_parse_metadata.as_deref(),
            )?;
            write_stage_message_conditioned(
                &mut lane.stream,
                &message,
                request.downstream_wire_condition,
            )
            .map_err(openai_io_error)?;
            let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
            if reply.kind != WireReplyKind::Ack {
                return Err(OpenAiError::backend(format!(
                    "expected multimodal generation config ACK from downstream, got {:?}",
                    reply.kind
                )));
            }

            let media_chunks = if prefill.chunks.is_empty() {
                return Err(OpenAiError::backend(
                    "multimodal prefill produced no activation chunks",
                ));
            } else {
                &prefill.chunks
            };
            let prefill_forward_timer = PhaseTimer::start();
            let mut final_reply = None;
            let mut prefill_pos_start = 0usize;
            let mut forward_activation_bytes = 0usize;
            let mut activation_encode_ms = 0.0;
            let mut forward_write_ms = 0.0;
            let mut downstream_wait_ms = 0.0;
            for (chunk_index, chunk) in media_chunks.iter().enumerate() {
                let is_final_chunk = chunk_index + 1 == media_chunks.len();
                let message = multimodal_prefill_message(MultimodalPrefillArgs {
                    request_id,
                    session_id,
                    prompt_token_count: prefill.token_count,
                    pos_start: prefill_pos_start,
                    token_count: chunk.token_count,
                    tokens: chunk.tokens.clone(),
                    positions: chunk.positions.clone(),
                    sampling: is_final_chunk.then_some(wire_sampling.clone()).flatten(),
                    final_chunk: is_final_chunk,
                })?;
                let forwarded = forwarded_stage_message_timed(
                    &request.config,
                    &message,
                    &chunk.output,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?;
                let write_timer = PhaseTimer::start();
                write_stage_message_conditioned(
                    &mut lane.stream,
                    &forwarded.message,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                forward_write_ms += write_timer.elapsed_ms();
                let wait_timer = PhaseTimer::start();
                let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
                downstream_wait_ms += wait_timer.elapsed_ms();
                let expected = if is_final_chunk {
                    WireReplyKind::PredictedToken
                } else {
                    WireReplyKind::Ack
                };
                if reply.kind != expected {
                    return Err(OpenAiError::backend(format!(
                        "expected multimodal prefill {expected:?} reply from downstream chunk {chunk_index}, got {:?}",
                        reply.kind
                    )));
                }
                forward_activation_bytes += forwarded.message.activation.len();
                activation_encode_ms += forwarded.activation_encode_ms;
                if is_final_chunk {
                    final_reply = Some(reply);
                }
                prefill_pos_start = prefill_pos_start
                    .checked_add(chunk.token_count)
                    .ok_or_else(|| {
                        OpenAiError::backend("multimodal prefill token offset overflow")
                    })?;
            }
            let reply = final_reply.ok_or_else(|| {
                OpenAiError::backend("multimodal prefill produced no predicted token")
            })?;
            let mut attrs = self.openai_attrs(&request.ids);
            attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(forward_activation_bytes),
            );
            attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                json!(activation_encode_ms),
            );
            attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(forward_write_ms),
            );
            attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(downstream_wait_ms),
            );
            self.emit_openai_phase(
                "stage.openai_media_prefill_forward",
                prefill_forward_timer,
                attrs,
            );

            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut current = reply.predicted;
            let mut decode_stage0_compute_ms = 0.0;
            let mut decode_runtime_lock_wait_ms = 0.0;
            let mut decode_runtime_lock_hold_ms = 0.0;
            let mut decode_runtime_lock_acquires = 0usize;
            let mut decode_batch_size_max = 1usize;
            let mut decode_batch_wait_ms = 0.0;
            let mut decode_forward_write_ms = 0.0;
            let mut decode_downstream_wait_ms = 0.0;
            let mut decode_output_activation_bytes = 0usize;
            let mut decode_forward_activation_bytes = 0usize;
            let mut decode_message = ReusableDecodeMessage::new(ReusableDecodeMessageArgs {
                request_id,
                session_id,
                prompt_token_count: prefill.token_count,
                base_pos_start: prefill.token_count,
                sampling: wire_sampling.clone(),
                sideband_capacity: 1,
            })?;

            while decoded_tokens < max_tokens as usize {
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    receipt_cancelled = true;
                    break;
                }
                if let Some(observation) = receipt_observation.as_mut() {
                    observation.record_token(current, request.ids.request_started_at.elapsed());
                }
                if let Some(config) = self.generation_receipt.as_ref() {
                    config.committed(GenerationCommit {
                        request_id,
                        session_id,
                        generated_token_count: decoded_tokens.saturating_add(1),
                        token_ids: vec![current].into_boxed_slice(),
                    });
                }
                lifecycle.commit(current, request.ids.request_started_at.elapsed());
                if collector.push_token(current)? == TokenControl::Stop {
                    if let Some(observation) = receipt_observation.as_mut() {
                        observation.mark_callback_stop();
                    }
                    lifecycle.mark_callback_stop();
                    decoded_tokens += 1;
                    break;
                }
                decoded_tokens += 1;
                if decoded_tokens >= max_tokens as usize {
                    break;
                }

                let decode_input_index = decoded_tokens - 1;
                let message = decode_message.update(decode_input_index, current)?;
                let token_timer = PhaseTimer::start();
                let stage0_timer = PhaseTimer::start();
                let batch_outcome = self.iteration_scheduler.execute_frame_iteration(
                    &session_key,
                    u64::try_from(message.pos_start).map_err(|_| {
                        OpenAiError::backend("negative authoritative decode position")
                    })?,
                    &[current],
                    &[],
                    request.sampling.enabled.then_some(&request.sampling),
                    None,
                    true,
                )?;
                if let Some(alignment) = batch_outcome.session_alignment {
                    let mut attrs = self.openai_attrs(&request.ids);
                    attrs.insert(
                        "llama_stage.session_auto_align_before_tokens".to_string(),
                        json!(alignment.before_token_count),
                    );
                    attrs.insert(
                        "llama_stage.session_auto_align_after_tokens".to_string(),
                        json!(alignment.after_token_count),
                    );
                    self.telemetry
                        .emit_debug("stage.openai_session_auto_align", attrs);
                }
                decode_runtime_lock_wait_ms += batch_outcome.runtime_lock_wait_ms;
                decode_runtime_lock_hold_ms += batch_outcome.runtime_lock_hold_ms;
                decode_runtime_lock_acquires += 1;
                decode_batch_size_max = decode_batch_size_max.max(batch_outcome.batch_size);
                decode_batch_wait_ms += batch_outcome.batch_wait_ms;
                let output = batch_outcome.output;
                let stage0_compute_ms = stage0_timer.elapsed_ms();
                decode_stage0_compute_ms += stage0_compute_ms;
                let forwarded = forwarded_stage_message_timed(
                    &request.config,
                    message,
                    &output,
                    request.activation_width,
                )
                .map_err(openai_backend_error)?;
                decode_output_activation_bytes =
                    decode_output_activation_bytes.saturating_add(output.payload.len());
                decode_forward_activation_bytes = decode_forward_activation_bytes
                    .saturating_add(forwarded.message.activation.len());
                let write_timer = PhaseTimer::start();
                write_stage_message_conditioned(
                    &mut lane.stream,
                    &forwarded.message,
                    request.downstream_wire_condition,
                )
                .map_err(openai_io_error)?;
                let forward_write_ms = write_timer.elapsed_ms();
                decode_forward_write_ms += forward_write_ms;
                let wait_timer = PhaseTimer::start();
                let reply = super::embedded_execution::receive_embedded_stage_reply(
                    &mut lane.stream,
                    request.prediction_return.as_ref(),
                    WireReplyKind::PredictedToken,
                )?;
                let downstream_wait_ms = wait_timer.elapsed_ms();
                decode_downstream_wait_ms += downstream_wait_ms;
                current = reply.predicted;
                if self.telemetry.is_debug_enabled() {
                    let mut token_attrs = self.openai_attrs(&request.ids);
                    token_attrs.insert(
                        "llama_stage.decode_step".to_string(),
                        json!(decode_input_index),
                    );
                    token_attrs.insert(
                        "llama_stage.stage0_compute_ms".to_string(),
                        json!(stage0_compute_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.forward_write_ms".to_string(),
                        json!(forward_write_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.downstream_wait_ms".to_string(),
                        json!(downstream_wait_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_batch_size".to_string(),
                        json!(batch_outcome.batch_size),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_batch_wait_ms".to_string(),
                        json!(batch_outcome.batch_wait_ms),
                    );
                    token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                    token_attrs.insert("llama_stage.message_kind".to_string(), json!("DecodeEmbd"));
                    self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
                }
            }

            let mut decode_attrs = self.openai_attrs(&request.ids);
            decode_attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            decode_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                json!(decode_stage0_compute_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(decode_runtime_lock_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(decode_runtime_lock_hold_ms),
            );
            decode_attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(decode_runtime_lock_acquires),
            );
            decode_attrs.insert(
                "llama_stage.decode_batch_size_max".to_string(),
                json!(decode_batch_size_max),
            );
            decode_attrs.insert(
                "llama_stage.decode_batch_wait_ms".to_string(),
                json!(decode_batch_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                json!(decode_forward_write_ms),
            );
            decode_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                json!(decode_downstream_wait_ms),
            );
            decode_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                json!(decode_output_activation_bytes),
            );
            decode_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                json!(decode_forward_activation_bytes),
            );
            receipt_model_generation_elapsed = Some(decode_timer.start_instant.elapsed());
            self.emit_openai_summary("stage.openai_decode", decode_timer, decode_attrs);
            Ok(())
        })();

        let generation_succeeded = result.is_ok();
        if receipt_cancelled {
            lifecycle.mark_cancelled();
        }
        lifecycle.finish(generation_succeeded);
        let receipt_result = self.finalize_generation_receipt(
            LocalGenerationReceiptFinalization {
                session_label: &session_key,
                request_id,
                session_id,
                agent_session_id: request.ids.agent_session_id.as_deref(),
                prompt_token_ids: receipt_prompt_token_ids.unwrap_or_default(),
                observation: receipt_observation,
                cancelled: receipt_cancelled,
                model_generation_elapsed: receipt_model_generation_elapsed,
                frontend_request_id: request.ids.frontend_request_id,
            },
            generation_succeeded,
        );

        let stop_result = write_stage_message(
            &mut lane.stream,
            &StageWireMessage::stop_with_identity(request_id, session_id),
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
        let scheduler_session_key = session_key.clone();
        if let Ok(outcome) = self.iteration_scheduler.execute_runtime_timed(
            "openai-embedded-session-stop",
            move |runtime| {
                runtime
                    .drop_session_timed(&scheduler_session_key)
                    .map_err(|error| openai_frontend::OpenAiError::backend(error.to_string()))
            },
        ) {
            let runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
            let drop_stats = outcome.value;
            let mut attrs = self.openai_attrs(&request.ids);
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
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
        let lane_id = lane.id;
        let stop_result = stop_result.map_err(openai_io_error);
        match (&result, &stop_result) {
            (Ok(_), Ok(_)) => request.lane_pool.return_lane(lane),
            _ => request.lane_pool.replace_lane(lane_id),
        }
        if result.is_ok() {
            receipt_result?;
            stop_result?;
        }
        result?;
        collector.finish(prompt_tokens, GenerationCacheStats::default())
    }
}

#[cfg(test)]
mod tests {
    use super::{LocalSessionCleanupGuard, text_generation::resident_capacity_target_tokens};
    use std::cell::Cell;

    #[test]
    fn resident_capacity_target_excludes_outer_decode_headroom() {
        let prompt_tokens = 31_101;
        let outer_reserved_tokens = prompt_tokens + 512;

        assert_eq!(resident_capacity_target_tokens(prompt_tokens), 31_101);
        assert_ne!(
            resident_capacity_target_tokens(prompt_tokens),
            u64::try_from(outer_reserved_tokens).unwrap()
        );
    }

    #[test]
    fn post_admission_multimodal_failure_releases_session() {
        let session_active = Cell::new(false);

        let (generation, cleanup) = LocalSessionCleanupGuard::run(
            || session_active.set(false),
            || {
                session_active.set(true);
                Err::<(), _>("injected media prefill failure")
            },
        );
        drop(cleanup);

        assert_eq!(generation, Err("injected media prefill failure"));
        assert!(!session_active.get(), "failed prefill leaked its session");
    }
}
