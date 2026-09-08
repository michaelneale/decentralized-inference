pub fn binary_repl(args: BinaryReplArgs) -> Result<()> {
    if args.native_logs {
        restore_native_logs();
    } else {
        suppress_native_logs();
    }
    let requested_tokenizer_path = args
        .tokenizer_model_path
        .as_deref()
        .unwrap_or(args.model_path.as_path());
    let materialized_tokenizer =
        materialize_tokenizer_package_if_needed(&args, requested_tokenizer_path)?;
    let tokenizer_path = materialized_tokenizer
        .as_deref()
        .unwrap_or(requested_tokenizer_path);
    let tokenizer = StageModel::open(
        tokenizer_path,
        &RuntimeConfig {
            stage_index: 0,
            layer_start: tokenizer_layer_start(&args, materialized_tokenizer.is_some()),
            layer_end: tokenizer_layer_end(&args, materialized_tokenizer.is_some()),
            ctx_size: args.ctx_size,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: args.tokenizer_n_gpu_layers,
            mmap: args.mmap,
            mlock: args.mlock,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_runtime::SplitMode::Auto,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
            load_mode: tokenizer_load_mode(&args, materialized_tokenizer.is_some()),
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
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
        },
    )
    .with_context(|| format!("open tokenizer model {}", tokenizer_path.display()))?;
    eprintln!(
        "tokenizer model: {} load_mode={} n_gpu_layers={}",
        tokenizer_path.display(),
        tokenizer_load_mode_label(&args, materialized_tokenizer.is_some()),
        args.tokenizer_n_gpu_layers
    );
    let thinking_override = prompt_thinking_override(&args)?;
    let hf_repl = args
        .model_path
        .to_str()
        .is_some_and(|s| s.starts_with("hf://"));
    let chat_template_model = if !args.raw_prompt
        && !hf_repl
        && materialized_tokenizer.is_none()
        && args
            .tokenizer_model_path
            .as_deref()
            .is_some_and(|path| path != args.model_path.as_path())
    {
        let model = StageModel::open(
            &args.model_path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: 0,
                layer_end: 1,
                ctx_size: args.ctx_size,
                lane_count: 1,
                n_batch: None,
                n_ubatch: None,
                n_threads: None,
                n_threads_batch: None,
                n_gpu_layers: 0,
                mmap: args.mmap,
                mlock: args.mlock,
                repack: false,
                op_offload: None,
                no_host_buffer: false,
                check_tensors: false,
                direct_io: false,
                main_gpu: None,
                split_mode: skippy_runtime::SplitMode::Auto,
                selected_backend_device: None,
                cache_type_k: GGML_TYPE_F16,
                cache_type_v: GGML_TYPE_F16,
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
                resident_tensor_names: Vec::new(),
                checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
                checkpoint_imatrix: None,
                checkpoint_imatrix_sha256: None,
                kv_offload: None,
                kv_unified: None,
                swa_full: None,
            },
        )
        .with_context(|| format!("open chat template model {}", args.model_path.display()))?;
        eprintln!(
            "chat template model: {} load_mode={:?} n_gpu_layers=0",
            args.model_path.display(),
            ReplLoadMode::RuntimeSlice
        );
        Some(model)
    } else {
        None
    };
    let mut draft = match args.draft_model_path.as_ref() {
        Some(path) => Some(DraftRunner::open(
            path,
            args.ctx_size,
            args.n_gpu_layers,
            args.speculative_window,
        )?),
        None => None,
    };

    eprintln!(
        "binary REPL connected to first stage at {}; max_new_tokens={} prefill_chunk_size={}",
        args.first_stage_addr,
        format_prompt_max_new_tokens(args.max_new_tokens),
        args.prefill_chunk_size
    );
    if let Some(draft) = draft.as_ref() {
        eprintln!(
            "draft model enabled: {} speculative_window={}",
            draft.path.display(),
            draft.window
        );
    }
    if args.adaptive_speculative_window {
        eprintln!(
            "adaptive speculative window enabled: start={} max={}",
            args.speculative_window.clamp(1, 4),
            args.speculative_window.max(1)
        );
    }
    if args.raw_prompt {
        eprintln!("prompt mode: raw completion");
    } else {
        eprintln!("prompt mode: chat template");
        match thinking_override {
            Some(false) => eprintln!("thinking: disabled at chat-template render"),
            Some(true) => eprintln!(
                "thinking: enabled at chat-template render; requested budget={}",
                args.thinking_token_budget
                    .map(|budget| budget.to_string())
                    .unwrap_or_else(|| "default".to_string())
            ),
            None => eprintln!("thinking: model template default"),
        }
    }
    eprintln!("thinking output: raw returned tokens");
    let default_session_id = args.session_id.clone().unwrap_or_else(default_session_id);
    let default_wire_session_id = stable_wire_id(&[default_session_id.as_bytes()]);
    eprintln!("session_id={default_session_id} wire_session_id={default_wire_session_id}");
    let interrupt = install_prompt_interrupt_handler()?;
    let mut history = PromptHistory::load(args.history_path.as_deref())?;
    let mut prompt_input = prompt_input(&history)?;
    if args.log_context.is_some() {
        eprintln!(
            "Type a prompt, use Up/Down for history, Ctrl-C to interrupt generation, :history, :logs [name] [lines], :rerun N, :noappend, :append, or :quit."
        );
    } else {
        eprintln!(
            "Type a prompt, use Up/Down for history, Ctrl-C to interrupt generation, :history, :rerun N, :noappend, :append, or :quit."
        );
    }

    let mut prompt_index = 0usize;
    let mut live_session = PromptLiveSession::default();
    let mut append_transcript = true;
    loop {
        let Some(input) = read_history_prompt(&mut prompt_input, "> ")? else {
            break;
        };
        let raw_input = input.trim_end_matches(['\r', '\n']);
        if raw_input.trim().is_empty() {
            continue;
        }
        if let Some((prompt_session_id, prompt)) = parse_prompt_json_command(raw_input)? {
            let prompt_session_id = if prompt_session_id.is_empty() {
                default_session_id.clone()
            } else {
                prompt_session_id
            };
            let wire_session_id = stable_wire_id(&[prompt_session_id.as_bytes()]);
            run_prompt(PromptRun {
                args: &args,
                tokenizer: &tokenizer,
                chat_template_model: chat_template_model.as_ref(),
                draft: draft.as_mut(),
                interrupt: &interrupt,
                session_id: &prompt_session_id,
                wire_session_id,
                prompt_index,
                prompt: &prompt,
                live_session: None,
            })
            .or_else(|error| handle_prompt_error(error, &interrupt, prompt_index))?;
            prompt_index += 1;
            continue;
        }
        let input = raw_input.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, ":q" | ":quit" | ":exit") {
            break;
        }
        if input == ":history" {
            history.print();
            continue;
        }
        if input == ":noappend" {
            if append_transcript {
                stop_live_prompt_session(
                    &mut live_session,
                    &args,
                    prompt_index,
                    default_wire_session_id,
                )?;
            }
            append_transcript = false;
            eprintln!("prompt mode: noappend; each prompt is sent as a fresh request");
            continue;
        }
        if input == ":append" {
            append_transcript = true;
            eprintln!("prompt mode: append; prompts continue the live chat transcript");
            continue;
        }
        if input == ":logs" || input.starts_with(":logs ") {
            show_prompt_logs(
                args.log_context.as_ref(),
                input.trim_start_matches(":logs").trim(),
            )?;
            continue;
        }
        if let Some(index) = input.strip_prefix(":rerun ") {
            let index = index
                .trim()
                .parse::<usize>()
                .context("parse :rerun index")?;
            let Some(prompt) = history.get(index).map(str::to_string) else {
                eprintln!("history entry {index} does not exist");
                continue;
            };
            eprintln!("rerun {index}: {prompt}");
            run_prompt(PromptRun {
                args: &args,
                tokenizer: &tokenizer,
                chat_template_model: chat_template_model.as_ref(),
                draft: draft.as_mut(),
                interrupt: &interrupt,
                session_id: &default_session_id,
                wire_session_id: default_wire_session_id,
                prompt_index,
                prompt: &prompt,
                live_session: None,
            })
            .or_else(|error| handle_prompt_error(error, &interrupt, prompt_index))?;
            prompt_index += 1;
            continue;
        }
        history.push(input)?;
        prompt_input.add_history_entry(input);
        let prompt_result = run_prompt(PromptRun {
            args: &args,
            tokenizer: &tokenizer,
            chat_template_model: chat_template_model.as_ref(),
            draft: draft.as_mut(),
            interrupt: &interrupt,
            session_id: &default_session_id,
            wire_session_id: default_wire_session_id,
            prompt_index,
            prompt: input,
            live_session: append_transcript.then_some(&mut live_session),
        });
        if prompt_result.is_err() {
            live_session.mark_dirty();
        }
        prompt_result.or_else(|error| handle_prompt_error(error, &interrupt, prompt_index))?;
        prompt_index += 1;
    }

    stop_live_prompt_session(
        &mut live_session,
        &args,
        prompt_index,
        default_wire_session_id,
    )?;

    Ok(())
}

fn materialize_tokenizer_package_if_needed(
    args: &BinaryReplArgs,
    tokenizer_path: &Path,
) -> Result<Option<PathBuf>> {
    if args.tokenizer_load_mode != ReplLoadMode::LayerPackage
        || !tokenizer_path.join("model-package.json").is_file()
    {
        return Ok(None);
    }

    let package_ref = tokenizer_path.to_string_lossy();
    let package = inspect_layer_package(&package_ref)
        .with_context(|| format!("inspect tokenizer package {}", tokenizer_path.display()))?;
    eprintln!(
        "materializing tokenizer model from package: {} layers=0..1",
        tokenizer_path.display()
    );
    let tokenizer_gguf = materialize_layer_package(&PackageStageRequest {
        model_id: package.model_id,
        topology_id: "binary-repl-tokenizer".to_string(),
        package_ref: package_ref.into_owned(),
        stage_id: "tokenizer".to_string(),
        layer_start: 0,
        layer_end: 1,
        include_embeddings: true,
        include_output: true,
    })?;
    Ok(Some(tokenizer_gguf))
}

fn tokenizer_load_mode(args: &BinaryReplArgs, materialized_package: bool) -> RuntimeLoadMode {
    if materialized_package {
        RuntimeLoadMode::ArtifactSlice
    } else {
        args.tokenizer_load_mode.into()
    }
}

fn tokenizer_load_mode_label(args: &BinaryReplArgs, materialized_package: bool) -> String {
    if materialized_package {
        "artifact-slice(materialized layer package)".to_string()
    } else {
        format!("{:?}", args.tokenizer_load_mode)
    }
}

fn tokenizer_layer_start(args: &BinaryReplArgs, materialized_package: bool) -> u32 {
    if materialized_package {
        0
    } else {
        args.tokenizer_layer_start
    }
}

fn tokenizer_layer_end(args: &BinaryReplArgs, materialized_package: bool) -> u32 {
    if materialized_package {
        1
    } else {
        args.tokenizer_layer_end
    }
}
