fn write_local_configs(
    args: &PromptArgs,
    run_id: &str,
    _run_dir: &Path,
    kv_mode: StageKvCacheMode,
    stages: &[LocalStage],
    _metrics_otlp_url: &str,
    hf_package_ref: bool,
) -> Result<()> {
    for stage in stages {
        let upstream = if stage.stage_index == 0 {
            Some(PeerConfig {
                stage_id: "driver".to_string(),
                stage_index: 0,
                endpoint: "driver".to_string(),
            })
        } else {
            let previous = &stages[stage.stage_index - 1];
            Some(PeerConfig {
                stage_id: previous.stage_id.clone(),
                stage_index: previous.stage_index as u32,
                endpoint: format!("tcp://{}", previous.endpoint_addr),
            })
        };
        let downstream = stages.get(stage.stage_index + 1).map(|next| PeerConfig {
            stage_id: next.stage_id.clone(),
            stage_index: next.stage_index as u32,
            endpoint: format!("tcp://{}", next.endpoint_addr),
        });
        let stage_model_path = if hf_package_ref {
            args.model_path.to_string_lossy().to_string()
        } else {
            stage
                .remote
                .as_ref()
                .map(|remote| remote.model_path.clone())
                .unwrap_or_else(|| stage.model_path.to_string_lossy().to_string())
        };
        let load_mode = if hf_package_ref || stage.remote.is_some() {
            LoadMode::LayerPackage
        } else {
            LoadMode::ArtifactSlice
        };
        let kv_cache = prompt_stage_kv_cache_config(args, stage, kv_mode.clone(), hf_package_ref)?;
        let stage_config = StageConfig {
            run_id: run_id.to_string(),
            topology_id: "local-binary-kv-repl".to_string(),
            model_id: args.model_id.clone(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: prompt_source_model_path(args, hf_package_ref)?,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: Some(stage_model_path),
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            activation_codec: skippy_protocol::StageActivationCodec::RawF32V1,
            activation_codec_policy: skippy_protocol::StageActivationCodecPolicy::default(),
            stage_id: stage.stage_id.clone(),
            stage_index: stage.stage_index as u32,
            layer_start: stage.layer_start,
            layer_end: stage.layer_end,
            ctx_size: args.ctx_size,
            lane_count: args.stage_max_inflight.max(1) as u32,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: args.n_gpu_layers,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: args.cache_type_k.clone(),
            cache_type_v: args.cache_type_v.clone(),
            flash_attn_type: StageFlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: true,
            checkpoint_quantization: None,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
            selected_device: None,
            kv_cache,
            native_mtp_enabled: true,
            load_mode,
            bind_addr: stage.bind_addr.clone(),
            upstream,
            downstream,
        };
        write_json(&stage.config_path, &serde_json::to_value(stage_config)?)?;
    }
    Ok(())
}

fn parse_stage_kv_mode(value: &str) -> Result<StageKvCacheMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "disabled" | "disable" | "off" | "false" => Ok(StageKvCacheMode::Disabled),
        "record" => Ok(StageKvCacheMode::Record),
        "lookup-record" | "lookup_record" | "lookuprecord" | "lookup" | "on" | "true" => {
            Ok(StageKvCacheMode::LookupRecord)
        }
        "correctness" => Ok(StageKvCacheMode::LookupRecord),
        other => bail!("unsupported kv mode {other}"),
    }
}

fn prompt_stage_kv_cache_config(
    args: &PromptArgs,
    stage: &LocalStage,
    mode: StageKvCacheMode,
    hf_package_ref: bool,
) -> Result<Option<StageKvCacheConfig>> {
    if mode == StageKvCacheMode::Disabled {
        return Ok(Some(StageKvCacheConfig {
            mode,
            payload: StageKvCachePayload::Auto,
            max_entries: 1,
            max_bytes: 0,
            min_tokens: args.kv_page_size_tokens.max(1),
            shared_prefix_stride_tokens: 128,
            shared_prefix_record_limit: 2,
        }));
    }

    let max_bytes = prompt_stage_cache_max_bytes(args, stage, hf_package_ref)?;
    let payload = prompt_stage_cache_payload(args, stage);
    Ok(Some(StageKvCacheConfig {
        mode,
        payload,
        max_entries: 128,
        max_bytes,
        min_tokens: args.kv_page_size_tokens.max(1),
        shared_prefix_stride_tokens: 128,
        shared_prefix_record_limit: 2,
    }))
}

fn prompt_stage_cache_payload(_args: &PromptArgs, _stage: &LocalStage) -> StageKvCachePayload {
    // The concrete payload is selected only after llama.cpp loads the model
    // and reports its actual dense/recurrent/hybrid state semantics.
    StageKvCachePayload::Auto
}

fn prompt_stage_cache_max_bytes(
    args: &PromptArgs,
    stage: &LocalStage,
    hf_package_ref: bool,
) -> Result<u64> {
    if let Some(meta) = prompt_cache_meta(args, stage, hf_package_ref)?
        && let Some(bytes) = estimate_prompt_stage_cache_max_bytes(
            stage.layer_start,
            stage.layer_end,
            args.ctx_size,
            args.stage_max_inflight.max(1) as u32,
            &args.cache_type_k,
            &args.cache_type_v,
            &meta,
        )
    {
        return Ok(bytes);
    }

    estimate_prompt_stage_cache_max_bytes_from_width(
        stage.layer_start,
        stage.layer_end,
        args.ctx_size,
        args.stage_max_inflight.max(1) as u32,
        &args.cache_type_k,
        &args.cache_type_v,
        u32::try_from(args.activation_width).context("activation_width must be non-negative")?,
    )
    .context("derive prompt stage cache max bytes")
}

fn prompt_cache_meta(
    args: &PromptArgs,
    stage: &LocalStage,
    hf_package_ref: bool,
) -> Result<Option<GgufCompactMeta>> {
    if args.model_path.is_file() {
        return Ok(scan_gguf_compact_meta(&args.model_path));
    }
    if args.model_path.join("model-package.json").is_file() {
        let info = inspect_layer_package(path_str(&args.model_path)?)
            .with_context(|| format!("inspect package {}", args.model_path.display()))?;
        return Ok(scan_gguf_compact_meta(Path::new(&info.source_model_path)));
    }
    if !hf_package_ref && stage.remote.is_none() {
        return Ok(scan_gguf_compact_meta(&stage.model_path));
    }
    Ok(None)
}

fn prompt_source_model_path(args: &PromptArgs, _hf_package_ref: bool) -> Result<Option<String>> {
    if args.model_path.is_file() {
        return Ok(Some(args.model_path.to_string_lossy().to_string()));
    }
    if args.model_path.join("model-package.json").is_file() {
        let info = inspect_layer_package(path_str(&args.model_path)?)
            .with_context(|| format!("inspect package {}", args.model_path.display()))?;
        return Ok(Some(info.source_model_path));
    }
    Ok(None)
}

fn estimate_prompt_stage_cache_max_bytes(
    layer_start: u32,
    layer_end: u32,
    ctx_size: u32,
    lane_count: u32,
    cache_type_k: &str,
    cache_type_v: &str,
    meta: &GgufCompactMeta,
) -> Option<u64> {
    let kv_heads = if meta.kv_head_count > 0 {
        meta.kv_head_count
    } else {
        meta.head_count
    };
    let key_width = if meta.key_length > 0 {
        meta.key_length
    } else if meta.embedding_size > 0 && kv_heads > 0 {
        meta.embedding_size.checked_div(kv_heads)?
    } else {
        return None;
    };
    let value_width = if meta.value_length > 0 {
        meta.value_length
    } else if meta.embedding_size > 0 && kv_heads > 0 {
        meta.embedding_size.checked_div(kv_heads)?
    } else {
        return None;
    };
    let key_elems = u64::from(key_width).checked_mul(u64::from(kv_heads))?;
    let value_elems = u64::from(value_width).checked_mul(u64::from(kv_heads))?;
    estimate_prompt_stage_cache_max_bytes_for_elements(
        PromptStageCacheShape {
            layer_start,
            layer_end,
            ctx_size,
            lane_count,
            cache_type_k,
            cache_type_v,
        },
        key_elems,
        value_elems,
    )
}

fn estimate_prompt_stage_cache_max_bytes_from_width(
    layer_start: u32,
    layer_end: u32,
    ctx_size: u32,
    lane_count: u32,
    cache_type_k: &str,
    cache_type_v: &str,
    activation_width: u32,
) -> Option<u64> {
    let elements = u64::from(activation_width);
    estimate_prompt_stage_cache_max_bytes_for_elements(
        PromptStageCacheShape {
            layer_start,
            layer_end,
            ctx_size,
            lane_count,
            cache_type_k,
            cache_type_v,
        },
        elements,
        elements,
    )
}

struct PromptStageCacheShape<'a> {
    layer_start: u32,
    layer_end: u32,
    ctx_size: u32,
    lane_count: u32,
    cache_type_k: &'a str,
    cache_type_v: &'a str,
}

fn estimate_prompt_stage_cache_max_bytes_for_elements(
    shape: PromptStageCacheShape<'_>,
    key_elems_per_token: u64,
    value_elems_per_token: u64,
) -> Option<u64> {
    let stage_layers = shape.layer_end.checked_sub(shape.layer_start)?;
    if stage_layers == 0 {
        return None;
    }
    let key_bytes = prompt_dtype_bytes(key_elems_per_token, shape.cache_type_k)?;
    let value_bytes = prompt_dtype_bytes(value_elems_per_token, shape.cache_type_v)?;
    key_bytes
        .checked_add(value_bytes)?
        .checked_mul(u64::from(stage_layers))?
        .checked_mul(u64::from(shape.ctx_size.max(1)))?
        .checked_mul(u64::from(shape.lane_count.max(1)))
        .filter(|bytes| *bytes > 0)
}

fn prompt_dtype_bytes(elements: u64, dtype: &str) -> Option<u64> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "f32" => elements.checked_mul(4),
        "f16" | "bf16" => elements.checked_mul(2),
        "q8" | "q8_0" => prompt_ggml_block_bytes(elements, 32, 34),
        "q8_1" => prompt_ggml_block_bytes(elements, 32, 36),
        "q4" | "q4_0" | "iq4_nl" => prompt_ggml_block_bytes(elements, 32, 18),
        "q4_1" => prompt_ggml_block_bytes(elements, 32, 20),
        _ => None,
    }
}

fn prompt_ggml_block_bytes(elements: u64, block_size: u64, type_size: u64) -> Option<u64> {
    elements.div_ceil(block_size).checked_mul(type_size)
}

fn materialize_stage_artifacts(args: &PromptArgs, stages: &[LocalStage]) -> Result<()> {
    let Some(first_stage) = stages.first() else {
        bail!("no stages to materialize");
    };
    let model_cache_dir = first_stage
        .model_path
        .parent()
        .context("stage model path has no parent")?;
    fs::create_dir_all(model_cache_dir)
        .with_context(|| format!("create model cache dir {}", model_cache_dir.display()))?;

    let total = stages.len();
    for (done, stage) in stages.iter().enumerate() {
        if stage_artifact_available(&stage.model_path)? {
            print_progress(
                "materializing GGUF shards",
                done + 1,
                total,
                &format!(
                    "cached {} layers {}..{}",
                    stage.stage_id, stage.layer_start, stage.layer_end
                ),
                Duration::ZERO,
            );
            eprintln!();
            continue;
        }
        let mut command = Command::new(&args.model_slice_bin);
        command.args([
            "write",
            path_str(&args.model_path)?,
            "--layers",
            &format!("{}..{}", stage.layer_start, stage.layer_end),
            "--out",
            path_str(&stage.model_path)?,
            "--stage-index",
            &stage.stage_index.to_string(),
        ]);
        if stage.layer_start == 0 {
            command.arg("--include-embeddings");
        }
        if stage.stage_index + 1 == total {
            command.arg("--include-output");
        }
        command.stdout(Stdio::null()).stderr(Stdio::null());
        let label = format!(
            "{} layers {}..{}",
            stage.stage_id, stage.layer_start, stage.layer_end
        );
        let status = run_with_progress(command, "materializing GGUF shards", done, total, &label)
            .with_context(|| format!("run skippy-model-package for {}", stage.stage_id))?;
        if !status.success() {
            bail!(
                "skippy-model-package failed for {} with status {status}",
                stage.stage_id
            );
        }
    }

    Ok(())
}

fn materialize_model_package(args: &PromptArgs, package_dir: &Path) -> Result<()> {
    if package_artifact_available(package_dir)? {
        eprintln!(
            "materializing GGUF package: cached {}",
            package_dir.display()
        );
        return Ok(());
    }

    if package_dir.exists() {
        fs::remove_dir_all(package_dir)
            .with_context(|| format!("remove incomplete package {}", package_dir.display()))?;
    }
    let parent = package_dir
        .parent()
        .context("model package cache path has no parent")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create model package cache dir {}", parent.display()))?;

    let mut command = Command::new(&args.model_slice_bin);
    command.args([
        "write-package",
        path_str(&args.model_path)?,
        "--out-dir",
        path_str(package_dir)?,
        "--model-id",
        &args.model_id,
    ]);
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let status = run_with_progress(
        command,
        "materializing GGUF package",
        0,
        1,
        &format!("{} -> {}", args.model_path.display(), package_dir.display()),
    )
    .with_context(|| "run skippy-model-package write-package")?;
    if !status.success() {
        bail!("skippy-model-package write-package failed with status {status}");
    }

    fs::write(
        package_dir.join(".complete"),
        format!(
            "model_id={}\nsource={}\n",
            args.model_id,
            args.model_path.display()
        ),
    )
    .with_context(|| format!("write package completion marker {}", package_dir.display()))?;
    Ok(())
}

fn run_with_progress(
    mut command: Command,
    title: &str,
    completed: usize,
    total: usize,
    label: &str,
) -> Result<std::process::ExitStatus> {
    let started = Instant::now();
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn {:?}", command))?;
    loop {
        print_progress(title, completed, total, label, started.elapsed());
        if let Some(status) = child.try_wait()? {
            print_progress(title, completed + 1, total, label, started.elapsed());
            eprintln!();
            return Ok(status);
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn print_progress(title: &str, completed: usize, total: usize, label: &str, elapsed: Duration) {
    let width = 24usize;
    let filled = completed
        .saturating_mul(width)
        .checked_div(total)
        .unwrap_or(0);
    let empty = width.saturating_sub(filled);
    eprint!(
        "\r\x1b[2K{} [{}{}] {}/{} {:>5.1}s {}",
        title,
        "#".repeat(filled),
        "-".repeat(empty),
        completed,
        total,
        elapsed.as_secs_f64(),
        truncate_label(label, 96)
    );
    io::stderr().flush().ok();
}

fn truncate_label(label: &str, max_chars: usize) -> String {
    let char_count = label.chars().count();
    if char_count <= max_chars {
        return label.to_string();
    }
    let keep = max_chars.saturating_sub(3);
    format!("{}...", label.chars().take(keep).collect::<String>())
}

fn stage_artifact_available(path: &Path) -> Result<bool> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.is_file() && metadata.len() > 0),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("stat {}", path.display())),
    }
}

fn package_artifact_available(path: &Path) -> Result<bool> {
    match fs::metadata(path.join("model-package.json")) {
        Ok(metadata) if metadata.is_file() && metadata.len() > 0 => {}
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "stat package manifest {}",
                    path.join("model-package.json").display()
                )
            });
        }
    }
    match fs::metadata(path.join(".complete")) {
        Ok(metadata) => Ok(metadata.is_file() && metadata.len() > 0),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("stat package marker {}", path.display())),
    }
}
