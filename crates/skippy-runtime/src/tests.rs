#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::Value;

    use super::{
        ChatReasoningFormat, ChatTemplateJsonOptions, ChatTemplateMessage, DecodeFrameBatchRequest,
        CheckpointQuantization, FlashAttentionType, GGML_TYPE_F16, GlmDsaPolicy,
        IterationBatchPhase, IterationBatchRequest, ModelInfo, MtpSource, NativeMtpDraft,
        RuntimeConfig, RuntimeLoadMode, SamplingConfig, SplitMode, StageModel, StageSession, Status,
        TensorRole, format_skippy_error,
    };
    use std::{
        env,
        path::PathBuf,
        time::{Duration, Instant},
    };

    const TOOL_CALLS_JSON: &str = r#"[{"type":"function","function":{"name":"execute_bash","description":"Run a command.","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]"#;

    fn correctness_model() -> Option<PathBuf> {
        env::var_os("SKIPPY_CORRECTNESS_MODEL").map(PathBuf::from)
    }

    fn infer_layer_end(path: &PathBuf) -> anyhow::Result<u32> {
        let info = ModelInfo::open(path)?;
        let layer_end = info
            .tensors()?
            .into_iter()
            .filter(|tensor| tensor.role == TensorRole::Layer)
            .filter_map(|tensor| tensor.layer_index)
            .max()
            .map(|layer| layer + 1)
            .unwrap_or(1);
        Ok(layer_end)
    }

    #[test]
    fn invalid_selected_backend_device_fails_before_model_open() {
        let _native_log_guard = crate::logging::native_log_test_guard();
        let config = RuntimeConfig {
            selected_backend_device: Some("definitely-not-a-device".to_string()),
            ..RuntimeConfig::default()
        };

        let error = match StageModel::open("/definitely/missing/model.gguf", &config) {
            Ok(_) => panic!("invalid device should fail before model load"),
            Err(error) => error.to_string(),
        };

        assert!(
            error.contains("unknown selected backend device: definitely-not-a-device"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn model_reader_prevents_late_model_mutation() {
        let mut model = StageModel::new_dummy();
        let reader = model.reader();
        let error = model
            .attach_mtp_draft_model("/definitely/missing/model.gguf", &RuntimeConfig::default())
            .unwrap_err();
        assert!(error.to_string().contains("model readers are active"));

        drop(reader);
        let error = model
            .attach_mtp_draft_model("/definitely/missing/model.gguf", &RuntimeConfig::default())
            .unwrap_err();
        assert!(error.to_string().contains("null model"));
    }

    fn open_correctness_model(model_path: &PathBuf) -> anyhow::Result<StageModel> {
        open_correctness_model_with_context(model_path, 256)
    }

    fn open_correctness_model_with_context(
        model_path: &PathBuf,
        ctx_size: u32,
    ) -> anyhow::Result<StageModel> {
        open_correctness_model_with_context_and_lanes(model_path, ctx_size, 1)
    }

    fn open_correctness_model_with_context_and_lanes(
        model_path: &PathBuf,
        ctx_size: u32,
        lane_count: u32,
    ) -> anyhow::Result<StageModel> {
        let layer_end = infer_layer_end(model_path)?;
        let config = RuntimeConfig {
            stage_index: 0,
            layer_start: 0,
            layer_end,
            ctx_size,
            lane_count,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            repack: false,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: FlashAttentionType::Auto,
            load_mode: RuntimeLoadMode::RuntimeSlice,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: GlmDsaPolicy::Auto,
            include_embeddings: true,
            include_output: true,
            mtp_source: MtpSource::Disabled,
            filter_tensors_on_load: false,
            resident_tensor_names: Vec::new(),
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: SplitMode::Auto,
            checkpoint_quantization: CheckpointQuantization::Preserve,
            checkpoint_imatrix: None,
            checkpoint_imatrix_sha256: None,
        };
        StageModel::open(model_path, &config)
    }

    #[test]
    fn mixed_prefill_decode_matches_serial_when_model_is_configured() -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        // The mixed and serial sides each keep three sessions live at once.
        let model = open_correctness_model_with_context_and_lanes(&model_path, 256, 6)?;
        let tokens = model.tokenize(
            "Mixed scheduling keeps decode latency bounded while prompts continue arriving.",
            true,
        )?;
        assert!(tokens.len() >= 3, "correctness prompt must produce three tokens");
        let prefix = &tokens[..2];
        let long_prefill = &tokens[..tokens.len().min(6)];
        let short_prefill = &tokens[..3];
        let mut mixed = [
            model.create_session()?,
            model.create_session()?,
            model.create_session()?,
        ];
        let mut serial = [
            model.create_session()?,
            model.create_session()?,
            model.create_session()?,
        ];
        let (mixed_decode_token, _) =
            mixed[0].prefill_chunk_frame_sampled(prefix, None, None, 0)?;
        let (serial_decode_token, _) =
            serial[0].prefill_chunk_frame_sampled(prefix, None, None, 0)?;
        assert_eq!(mixed_decode_token, serial_decode_token);
        let decode_tokens = [mixed_decode_token];
        let output = {
            let [decode, long, short] = &mut mixed;
            let mut requests = [
                IterationBatchRequest {
                    session: decode,
                    token_ids: &decode_tokens,
                    positions: &[],
                    sampling: None,
                    input: None,
                    sample_last: true,
                    phase: IterationBatchPhase::Decode,
                },
                IterationBatchRequest {
                    session: long,
                    token_ids: long_prefill,
                    positions: &[],
                    sampling: None,
                    input: None,
                    sample_last: false,
                    phase: IterationBatchPhase::Prefill,
                },
                IterationBatchRequest {
                    session: short,
                    token_ids: short_prefill,
                    positions: &[],
                    sampling: None,
                    input: None,
                    sample_last: true,
                    phase: IterationBatchPhase::Prefill,
                },
            ];
            StageSession::iteration_batch_sampled(&mut requests)?
        };

        let (serial_decode_prediction, _) =
            serial[0].decode_step_frame_sampled(serial_decode_token, None, None, 0)?;
        serial[1].prefill_chunk_frame(long_prefill, None, 0)?;
        let (serial_short_prediction, _) =
            serial[2].prefill_chunk_frame_sampled(short_prefill, None, None, 0)?;
        assert_eq!(mixed[0].last_token_signal()?, serial[0].last_token_signal()?);
        assert_eq!(mixed[2].last_token_signal()?, serial[2].last_token_signal()?);
        assert_eq!(
            output
                .samples
                .iter()
                .map(|sample| (sample.request_index, sample.predicted_token))
                .collect::<Vec<_>>(),
            [
                (0, serial_decode_prediction),
                (2, serial_short_prediction)
            ]
        );
        assert_eq!(output.request_outputs.len(), 3);
        for index in 0..3 {
            assert_eq!(mixed[index].token_count(), serial[index].token_count());
            assert_eq!(mixed[index].native_position()?, serial[index].native_position()?);
        }
        Ok(())
    }

    #[test]
    fn legacy_batched_decode_preserves_lazy_token_signals_when_model_is_configured()
    -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model_with_context_and_lanes(&model_path, 256, 4)?;
        let tokens = model.tokenize("Legacy batch sampling keeps token signals lazy.", true)?;
        assert!(tokens.len() >= 2, "correctness prompt must produce two tokens");
        let prefix = &tokens[..2];
        let mut batched = [model.create_session()?, model.create_session()?];
        let mut serial = [model.create_session()?, model.create_session()?];
        let mut batched_inputs = [0; 2];
        let mut serial_inputs = [0; 2];
        for index in 0..2 {
            batched_inputs[index] = batched[index]
                .prefill_chunk_frame_sampled(prefix, None, None, 0)?
                .0;
            serial_inputs[index] = serial[index]
                .prefill_chunk_frame_sampled(prefix, None, None, 0)?
                .0;
            assert_eq!(batched_inputs[index], serial_inputs[index]);
        }

        let outputs = {
            let [first, second] = &mut batched;
            let mut requests = [
                DecodeFrameBatchRequest {
                    session: first,
                    token_id: batched_inputs[0],
                    sampling: None,
                    input: None,
                },
                DecodeFrameBatchRequest {
                    session: second,
                    token_id: batched_inputs[1],
                    sampling: None,
                    input: None,
                },
            ];
            StageSession::decode_step_frame_batch_sampled(&mut requests)?
        };
        let serial_outputs = [
            serial[0]
                .decode_step_frame_sampled(serial_inputs[0], None, None, 0)?
                .0,
            serial[1]
                .decode_step_frame_sampled(serial_inputs[1], None, None, 0)?
                .0,
        ];

        assert_eq!(
            outputs
                .iter()
                .map(|output| output.predicted_token)
                .collect::<Vec<_>>(),
            serial_outputs,
        );
        for index in 0..2 {
            assert_eq!(batched[index].last_token_signal()?, serial[index].last_token_signal()?);
        }
        Ok(())
    }

    fn tool_call_template_options() -> ChatTemplateJsonOptions {
        ChatTemplateJsonOptions {
            tools_json: Some(TOOL_CALLS_JSON.to_string()),
            ..ChatTemplateJsonOptions::default()
        }
    }

    #[test]
    fn chat_template_applies_when_model_is_configured() -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping chat template smoke: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model(&model_path)?;
        let prompt = model.apply_chat_template(
            &[
                ChatTemplateMessage::new("system", "You are concise."),
                ChatTemplateMessage::new("user", "Template smoke prompt."),
            ],
            true,
        )?;
        assert!(prompt.contains("Template smoke prompt."));
        assert!(prompt.len() >= "Template smoke prompt.".len());
        Ok(())
    }

    // Requires SKIPPY_CORRECTNESS_MODEL to point at a reasoning-capable model
    // family whose chat parser extracts <think> blocks (e.g. Qwen3).
    #[test]
    fn chat_reasoning_markers_are_stripped_and_extracted_when_model_is_configured()
    -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping chat reasoning smoke: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model(&model_path)?;
        let rendered = model.apply_chat_template_json(
            r#"[{"role":"user","content":"Say hi."}]"#,
            ChatTemplateJsonOptions {
                reasoning_format: Some(ChatReasoningFormat::Hidden),
                ..ChatTemplateJsonOptions::default()
            },
        )?;
        let metadata: Value = serde_json::from_str(&rendered.metadata_json)?;
        assert_eq!(
            metadata.get("reasoning_format").and_then(Value::as_str),
            Some("auto"),
        );

        // The generation prompt may already open the thought block, in which
        // case the model output continues inside it without the opening tag.
        let generation_prompt = metadata
            .get("generation_prompt")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let generated = if generation_prompt.contains("<think>") {
            "Consider the greeting.</think>Hi there!"
        } else {
            "<think>Consider the greeting.</think>Hi there!"
        };
        let parsed = model.parse_chat_response_json(generated, &rendered.metadata_json, false)?;
        let message: Value = serde_json::from_str(&parsed)?;
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(
            !content.contains("<think>") && !content.contains("</think>"),
            "reasoning markers must be stripped from content: {content:?}"
        );
        assert!(
            content.contains("Hi there!"),
            "visible content must survive reasoning extraction: {content:?}"
        );
        assert_eq!(
            message.get("reasoning_content").and_then(Value::as_str),
            Some("Consider the greeting."),
            "reasoning content must be extracted from the thought block"
        );
        Ok(())
    }

    #[test]
    fn chat_template_kwargs_are_accepted_by_native_renderer_when_model_is_configured()
    -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping chat kwargs smoke: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model(&model_path)?;
        let rendered = model.apply_chat_template_json(
            r#"[{"role":"user","content":"Say hi."}]"#,
            ChatTemplateJsonOptions {
                chat_template_kwargs: Some(
                    r#"{"reasoning_effort":"max","mesh_test_mode":7}"#.to_string(),
                ),
                ..ChatTemplateJsonOptions::default()
            },
        )?;

        assert!(!rendered.prompt.is_empty());
        assert!(!rendered.metadata_json.is_empty());
        Ok(())
    }

    #[test]
    fn format_skippy_error_omits_abi_envelope() {
        let err = format_skippy_error(Status::RuntimeError, "something broke");
        assert!(
            !err.contains("skippy ABI call failed"),
            "error format must not contain the old ABI envelope prefix: {err}"
        );
        assert!(
            err.contains("RuntimeError"),
            "error must contain the status variant"
        );
        assert!(
            err.contains("something broke"),
            "error must contain the message"
        );
    }

    #[test]
    fn format_skippy_error_works_without_message() {
        let err = format_skippy_error(Status::Unsupported, "");
        assert!(!err.contains("skippy ABI call failed"));
        assert!(err.contains("Unsupported"));
    }

    #[test]
    fn format_skippy_error_covers_all_status_variants() {
        for status in [
            Status::Error,
            Status::InvalidArgument,
            Status::Unsupported,
            Status::BufferTooSmall,
            Status::IoError,
            Status::ModelError,
            Status::RuntimeError,
        ] {
            let err = format_skippy_error(status, "test");
            assert!(
                !err.contains("skippy ABI call failed"),
                "error must not contain ABI envelope for {status:?}: {err}"
            );
            assert!(err.contains("test"));
        }
    }

    #[test]
    fn configure_chat_sampling_survives_bad_metadata_json() -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model(&model_path)?;
        let mut session = model.create_session()?;
        let sampling = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        // Send deliberately malformed JSON — the C++ catch blocks
        // should clear chat sampling and return success instead of
        // surfacing the parse error as a fatal status.
        let result = session.configure_chat_sampling("this is not valid json", 0, Some(&sampling));
        assert!(
            result.is_ok(),
            "configure_chat_sampling should return Ok even with bad metadata: {result:?}"
        );
        Ok(())
    }

    #[test]
    fn batched_sampled_verification_matches_serial_across_lazy_grammar_trigger()
    -> anyhow::Result<()> {
        let Some(model_path) = correctness_model() else {
            eprintln!("skipping: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model(&model_path)?;
        let rendered = model.apply_chat_template_json(
            r#"[{"role":"user","content":"Call execute_bash."}]"#,
            tool_call_template_options(),
        )?;
        let metadata: Value = serde_json::from_str(&rendered.metadata_json)?;
        assert!(
            metadata
                .get("grammar")
                .and_then(Value::as_str)
                .is_some_and(|grammar| !grammar.is_empty()),
            "tool-capable template must produce a grammar"
        );
        assert_eq!(
            metadata.get("grammar_lazy").and_then(Value::as_bool),
            Some(true),
            "tool grammar must wait for its trigger"
        );

        let prompt_tokens = model.tokenize(&rendered.prompt, true)?;
        assert!(prompt_tokens.len() > 1);
        let mut verify_inputs = vec![*prompt_tokens.last().expect("checked nonempty prompt")];
        verify_inputs.extend(model.tokenize("<tool_call>", false)?);
        verify_inputs.extend(model.tokenize(
            "execute_bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>",
            false,
        )?);

        let sampling = SamplingConfig {
            enabled: true,
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            ..SamplingConfig::default()
        };
        let prompt_prefix = &prompt_tokens[..prompt_tokens.len() - 1];
        let prompt_token_count = u64::try_from(prompt_tokens.len())?;

        let mut serial = model.create_session()?;
        serial.prefill_chunked(prompt_prefix)?;
        serial.configure_chat_sampling(
            &rendered.metadata_json,
            prompt_token_count,
            Some(&sampling),
        )?;
        let mut serial_predictions = Vec::with_capacity(verify_inputs.len());
        for (index, token) in verify_inputs.iter().copied().enumerate() {
            let predicted = serial.decode_step_sampled(token, Some(&sampling))?;
            serial_predictions.push(predicted);
            if index + 1 < verify_inputs.len() && predicted != verify_inputs[index + 1] {
                break;
            }
        }
        let serial_token_count = serial.token_count();
        let serial_native_position = serial.native_position()?;
        drop(serial);

        let mut batched = model.create_session()?;
        batched.prefill_chunked(prompt_prefix)?;
        batched.configure_chat_sampling(
            &rendered.metadata_json,
            prompt_token_count,
            Some(&sampling),
        )?;
        let batched_predictions = batched.verify_tokens_sampled(&verify_inputs, Some(&sampling))?;
        assert_eq!(
            batched_predictions, serial_predictions,
            "batched verification must stop at the first target mismatch"
        );
        batched.trim_session(serial_token_count)?;
        assert_eq!(batched.token_count(), serial_token_count);
        assert_eq!(batched.native_position()?, serial_native_position);
        Ok(())
    }

    #[test]
    fn long_resident_tool_context_preserves_grammar_and_native_mtp_acceptance() -> anyhow::Result<()>
    {
        const MIN_RESIDENT_TOKENS: usize = 8_192;
        const CONTEXT_SIZE: u32 = 10_240;
        // Resident prefixes reserve IDs immediately after the active lane IDs.
        // A single-lane runtime uses `3`, matching the state-handoff harness.
        const RESIDENT_PREFIX_ID: i32 = 3;

        let Some(model_path) = correctness_model() else {
            eprintln!("skipping: SKIPPY_CORRECTNESS_MODEL is not set");
            return Ok(());
        };
        let model = open_correctness_model_with_context(&model_path, CONTEXT_SIZE)?;

        let resident_sentence =
            "The resident tool context records a completed command result cwd workspace. ";
        let sentence_tokens = model.tokenize(resident_sentence, false)?;
        assert!(
            !sentence_tokens.is_empty(),
            "resident context sentence must tokenize"
        );
        let mut resident_sentence_count = MIN_RESIDENT_TOKENS / sentence_tokens.len() + 1;
        let (rendered, prompt_tokens) = loop {
            let resident_context = resident_sentence.repeat(resident_sentence_count);
            let rendered = model.apply_chat_template_json(
                &format!(
                    r#"[{{"role":"user","content":"Call execute_bash after this resident context: {resident_context}"}}]"#
                ),
                tool_call_template_options(),
            )?;
            let prompt_tokens = model.tokenize(&rendered.prompt, true)?;
            if prompt_tokens.len() >= MIN_RESIDENT_TOKENS {
                break (rendered, prompt_tokens);
            }
            let shortfall = MIN_RESIDENT_TOKENS - prompt_tokens.len();
            resident_sentence_count +=
                shortfall * resident_sentence_count / prompt_tokens.len() + 1;
        };
        let metadata: Value = serde_json::from_str(&rendered.metadata_json)?;
        assert_eq!(
            metadata.get("grammar_lazy").and_then(Value::as_bool),
            Some(true),
            "tool grammar must wait for its trigger"
        );

        assert!(
            prompt_tokens.len() >= MIN_RESIDENT_TOKENS,
            "expected at least {MIN_RESIDENT_TOKENS} resident tokens, got {}",
            prompt_tokens.len()
        );
        assert!(
            prompt_tokens.len() < CONTEXT_SIZE as usize,
            "resident prompt must leave room for tool sampling"
        );
        let prompt_prefix = &prompt_tokens[..prompt_tokens.len() - 1];
        let prompt_token_count = u64::try_from(prompt_tokens.len())?;
        let last_prompt_token = *prompt_tokens.last().expect("checked nonempty prompt");
        let sampling = SamplingConfig {
            enabled: true,
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            ..SamplingConfig::default()
        };

        let mut prefix_owner = model.create_session()?;
        prefix_owner.prefill_chunked(prompt_prefix)?;
        prefix_owner.save_prefix(RESIDENT_PREFIX_ID, prompt_prefix.len() as u64)?;
        drop(prefix_owner);

        let mut verify_inputs = vec![last_prompt_token];
        verify_inputs.extend(model.tokenize("<tool_call>", false)?);
        verify_inputs.extend(model.tokenize(
            "execute_bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>",
            false,
        )?);

        let mut serial =
            model.create_session_from_resident_prefix(RESIDENT_PREFIX_ID, prompt_prefix)?;
        serial.configure_chat_sampling(
            &rendered.metadata_json,
            prompt_token_count,
            Some(&sampling),
        )?;
        let mut serial_predictions = Vec::with_capacity(verify_inputs.len());
        for (index, token) in verify_inputs.iter().copied().enumerate() {
            let predicted = serial.decode_step_sampled(token, Some(&sampling))?;
            serial_predictions.push(predicted);
            if index + 1 < verify_inputs.len() && predicted != verify_inputs[index + 1] {
                break;
            }
        }
        let serial_token_count = serial.token_count();
        let serial_native_position = serial.native_position()?;
        drop(serial);

        let mut batched =
            model.create_session_from_resident_prefix(RESIDENT_PREFIX_ID, prompt_prefix)?;
        batched.configure_chat_sampling(
            &rendered.metadata_json,
            prompt_token_count,
            Some(&sampling),
        )?;
        let batched_predictions = batched.verify_tokens_sampled(&verify_inputs, Some(&sampling))?;
        assert_eq!(
            batched_predictions, serial_predictions,
            "resident-KV verification must stop at the first tool-grammar mismatch"
        );
        batched.trim_session(serial_token_count)?;
        assert_eq!(batched.token_count(), serial_token_count);
        assert_eq!(batched.native_position()?, serial_native_position);
        drop(batched);

        let mut native_mtp =
            model.create_session_from_resident_prefix(RESIDENT_PREFIX_ID, prompt_prefix)?;
        native_mtp.configure_chat_sampling(
            &rendered.metadata_json,
            prompt_token_count,
            Some(&sampling),
        )?;
        let decode_started = Instant::now();
        let (predicted, draft) =
            native_mtp.decode_step_sampled_mtp(last_prompt_token, Some(&sampling), 4)?;
        let resident_decode_elapsed = decode_started.elapsed();
        assert!(
            resident_decode_elapsed < Duration::from_secs(30),
            "sampling after an 8k resident KV prefix took {resident_decode_elapsed:?}"
        );
        drop(native_mtp);

        if let Some(draft) = draft {
            let mut target =
                model.create_session_from_resident_prefix(RESIDENT_PREFIX_ID, prompt_prefix)?;
            target.configure_chat_sampling(
                &rendered.metadata_json,
                prompt_token_count,
                Some(&sampling),
            )?;
            let mut target_inputs = vec![last_prompt_token, predicted];
            target_inputs.extend(&draft.token_ids);
            let target_predictions =
                target.verify_tokens_sampled(&target_inputs, Some(&sampling))?;
            assert_eq!(
                target_predictions.first(),
                Some(&predicted),
                "resident target decode must agree with the native-MTP source token"
            );
            let accepted_draft_tokens = target_predictions
                .iter()
                .skip(1)
                .zip(&draft.token_ids)
                .take_while(|(target, draft)| target == draft)
                .count();
            assert!(
                accepted_draft_tokens > 0,
                "native MTP must accept a draft token after the resident tool context; draft={:?}, target={target_predictions:?}",
                draft.token_ids
            );
        }

        Ok(())
    }

    #[test]
    fn ignore_eos_sets_the_native_sampling_flag() {
        let sampling = SamplingConfig {
            enabled: true,
            ignore_eos: true,
            ..SamplingConfig::default()
        };
        assert_eq!(sampling.as_raw().unwrap().flags & 0b11, 0b11);
    }

    #[test]
    fn stage_session_exposes_non_frame_native_mtp_decode_api() {
        type DecodeStepSampledMtp = fn(
            &mut StageSession,
            i32,
            Option<&SamplingConfig>,
            usize,
        ) -> Result<(i32, Option<NativeMtpDraft>)>;

        let _decode: DecodeStepSampledMtp = StageSession::decode_step_sampled_mtp;
    }
}

#[cfg(test)]
#[test]
fn model_open_events_success() {
    runtime_events::tests::assert_model_open_events_success();
}

#[cfg(test)]
#[test]
fn model_open_events_handled_failure() {
    runtime_events::tests::assert_model_open_events_handled_failure();
}

#[cfg(test)]
#[test]
fn model_open_events_missing_terminal_callback_uses_return() {
    runtime_events::tests::assert_model_open_events_missing_terminal_callback_uses_return();
}

#[cfg(test)]
#[test]
fn model_open_events_forwarded_before_open_returns() {
    runtime_events::tests::assert_model_open_events_forwarded_before_open_returns();
}

#[cfg(test)]
#[test]
fn model_open_events_feature_missing_falls_back() {
    runtime_events::tests::assert_model_open_events_feature_missing_falls_back();
}
