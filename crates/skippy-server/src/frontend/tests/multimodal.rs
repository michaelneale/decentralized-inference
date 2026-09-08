use super::*;

const MM_MODEL_ENV: &str = "SKIPPY_MM_MODEL";
const MM_PROJECTOR_ENV: &str = "SKIPPY_MM_PROJECTOR";
const MM_IMAGE_ENV: &str = "SKIPPY_MM_IMAGE";
const MM_ACTIVATION_WIDTH_ENV: &str = "SKIPPY_MM_ACTIVATION_WIDTH";
const MM_SPLIT_LAYER_ENV: &str = "SKIPPY_MM_SPLIT_LAYER";
const MM_CTX_SIZE_ENV: &str = "SKIPPY_MM_CTX_SIZE";
const MM_MAX_TOKENS_ENV: &str = "SKIPPY_MM_MAX_TOKENS";
const MM_N_GPU_LAYERS_ENV: &str = "SKIPPY_MM_N_GPU_LAYERS";

struct MultimodalSmokeFixture {
    model_path: PathBuf,
    projector_path: PathBuf,
    image_path: PathBuf,
    layer_end: u32,
    activation_width: i32,
    ctx_size: u32,
    max_tokens: u32,
    n_gpu_layers: i32,
}

fn multimodal_smoke_fixture() -> Result<Option<MultimodalSmokeFixture>> {
    let model_path = match env::var_os(MM_MODEL_ENV) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!(
                "skipping real multimodal smoke: set {MM_MODEL_ENV}, {MM_PROJECTOR_ENV}, and {MM_IMAGE_ENV}"
            );
            return Ok(None);
        }
    };
    let projector_path = match env::var_os(MM_PROJECTOR_ENV) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("skipping real multimodal smoke: set {MM_PROJECTOR_ENV}");
            return Ok(None);
        }
    };
    let image_path = match env::var_os(MM_IMAGE_ENV) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("skipping real multimodal smoke: set {MM_IMAGE_ENV}");
            return Ok(None);
        }
    };
    if !model_path.is_file() {
        bail!(
            "{MM_MODEL_ENV} does not point at a file: {}",
            model_path.display()
        );
    }
    if !projector_path.is_file() {
        bail!(
            "{MM_PROJECTOR_ENV} does not point at a file: {}",
            projector_path.display()
        );
    }
    if !image_path.is_file() {
        bail!(
            "{MM_IMAGE_ENV} does not point at a file: {}",
            image_path.display()
        );
    }
    let layer_end = model_layer_count(&model_path)?;
    let activation_width = env_i32(MM_ACTIVATION_WIDTH_ENV)?
        .map(Ok)
        .unwrap_or_else(|| infer_activation_width(&model_path))?;
    let ctx_size = env_u32(MM_CTX_SIZE_ENV)?.unwrap_or(2048);
    let max_tokens = env_u32(MM_MAX_TOKENS_ENV)?.unwrap_or(16);
    let n_gpu_layers = env_i32(MM_N_GPU_LAYERS_ENV)?.unwrap_or(0);
    Ok(Some(MultimodalSmokeFixture {
        model_path,
        projector_path,
        image_path,
        layer_end,
        activation_width,
        ctx_size,
        max_tokens,
        n_gpu_layers,
    }))
}

fn env_i32(name: &str) -> Result<Option<i32>> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<i32>()
                .with_context(|| format!("parse {name}={value:?} as i32"))
        })
        .transpose()
}

fn env_u32(name: &str) -> Result<Option<u32>> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<u32>()
                .with_context(|| format!("parse {name}={value:?} as u32"))
        })
        .transpose()
}

fn infer_activation_width(path: &Path) -> Result<i32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    let candidates = [
        "attn_norm.weight",
        "attention_norm.weight",
        "input_layernorm.weight",
        "ln_1.weight",
    ];
    let width = info
        .tensors()?
        .into_iter()
        .filter(|tensor| tensor.layer_index == Some(0))
        .find(|tensor| {
            candidates
                .iter()
                .any(|suffix| tensor.name.ends_with(suffix))
        })
        .map(|tensor| tensor.element_count)
        .ok_or_else(|| {
            anyhow!(
                "could not infer activation width for {}; set {MM_ACTIVATION_WIDTH_ENV}",
                path.display()
            )
        })?;
    i32::try_from(width).context("activation width exceeds i32")
}

fn multimodal_stage_config(
    fixture: &MultimodalSmokeFixture,
    stage_id: &str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    bind_addr: SocketAddr,
) -> StageConfig {
    StageConfig {
        run_id: "mm-smoke-run".to_string(),
        topology_id: "mm-smoke-topology".to_string(),
        model_id: "mm-smoke".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(fixture.model_path.to_string_lossy().to_string()),
        projector_path: (stage_index == 0)
            .then(|| fixture.projector_path.to_string_lossy().to_string()),
        stage_id: stage_id.to_string(),
        stage_index,
        layer_start,
        layer_end,
        ctx_size: fixture.ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_gpu_layers: fixture.n_gpu_layers,
        mmap: None,
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
        flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
        cache_idle_slots: None,
        filter_tensors_on_load: layer_start != 0 || layer_end != fixture.layer_end,
        resident_tensor_names: Vec::new(),
        selected_device: None,
        kv_cache: None,
        native_mtp_enabled: true,
        load_mode: skippy_protocol::LoadMode::RuntimeSlice,
        bind_addr: bind_addr.to_string(),
        upstream: None,
        downstream: None,
        ..StageConfig::default()
    }
}

fn local_openai_backend(config: StageConfig) -> Result<StageOpenAiBackend> {
    let runtime = load_runtime(&config)?.context("load smoke runtime")?;
    let ctx_size = usize::try_from(config.ctx_size).unwrap_or(usize::MAX);
    let telemetry = Telemetry::new(
        None,
        1,
        config.clone(),
        crate::telemetry::TelemetryLevel::Off,
    );
    let iteration_scheduler =
        IterationScheduler::new(runtime.clone(), &config, 1, true, telemetry.clone())?;
    Ok(StageOpenAiBackend {
        runtime,
        telemetry,
        config,
        model_id: "mm-smoke".to_string(),
        default_max_tokens: 16,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size,
        mode: OpenAiBackendMode::LocalRuntime,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_max: 0,
        speculative: SpeculativeDecodeConfig::default(),
        generation_limit: Arc::new(GenerationConcurrencyController::fixed(1)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: 1,
        generation_admission_timeout: std::time::Duration::from_secs(10),
        generation_service_estimator: Arc::new(crate::frontend::GenerationServiceEstimator::new(1)),
        generation_session_locks: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(ctx_size)),
        hook_policy: None,
        generation_receipt: None,
        linear_proposal_ingress: None,
        kv: None,
        iteration_scheduler,
    })
}

fn multimodal_chat_request(fixture: &MultimodalSmokeFixture) -> Result<ChatCompletionRequest> {
    multimodal_chat_request_with_max_tokens(fixture, fixture.max_tokens)
}

fn multimodal_chat_request_with_max_tokens(
    fixture: &MultimodalSmokeFixture,
    max_tokens: u32,
) -> Result<ChatCompletionRequest> {
    let image = fs::read(&fixture.image_path)
        .with_context(|| format!("read smoke image {}", fixture.image_path.display()))?;
    let mime_type = match fixture
        .image_path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    };
    let encoded = base64::engine::general_purpose::STANDARD.encode(image);
    serde_json::from_value(json!({
            "model": "mm-smoke",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {"type": "image_url", "image_url": {"url": format!("data:{mime_type};base64,{encoded}")}}
                ]
            }],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }))
        .context("build multimodal smoke request")
}

fn malformed_multimodal_chat_request() -> ChatCompletionRequest {
    serde_json::from_value(json!({
        "model": "mm-smoke",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image briefly."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,aGVsbG8="
                }}
            ]
        }],
        "max_tokens": 1,
        "temperature": 0.0
    }))
    .expect("malformed multimodal request should deserialize")
}

fn assert_nonempty_chat_response(response: &ChatCompletionResponse) {
    let content = response
        .choices
        .first()
        .and_then(|choice| choice.message.content.as_deref())
        .unwrap_or_default()
        .trim();
    assert!(
        !content.is_empty(),
        "expected non-empty multimodal response"
    );
}

fn available_loopback_addr() -> Result<SocketAddr> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

fn split_layer_for_fixture(fixture: &MultimodalSmokeFixture) -> Result<Option<u32>> {
    if fixture.layer_end < 2 {
        eprintln!("skipping split multimodal smoke: model has fewer than two layers");
        return Ok(None);
    }
    let split = env_u32(MM_SPLIT_LAYER_ENV)?.unwrap_or(fixture.layer_end / 2);
    if split == 0 || split >= fixture.layer_end {
        bail!(
            "{MM_SPLIT_LAYER_ENV} must be in 1..{} for this model",
            fixture.layer_end
        );
    }
    Ok(Some(split))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_multimodal_local_smoke_when_fixture_is_set() -> Result<()> {
    let Some(fixture) = multimodal_smoke_fixture()? else {
        return Ok(());
    };
    let config = multimodal_stage_config(
        &fixture,
        "stage-0",
        0,
        0,
        fixture.layer_end,
        available_loopback_addr()?,
    );
    let backend = local_openai_backend(config)?;
    let response = backend
        .chat_completion(multimodal_chat_request(&fixture)?)
        .await?;

    assert_nonempty_chat_response(&response);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_multimodal_local_prefill_failure_releases_lane_when_fixture_is_set() -> Result<()> {
    let Some(fixture) = multimodal_smoke_fixture()? else {
        return Ok(());
    };
    let config = multimodal_stage_config(
        &fixture,
        "stage-0",
        0,
        0,
        fixture.layer_end,
        available_loopback_addr()?,
    );
    let backend = local_openai_backend(config)?;

    for attempt in 0..2 {
        let result = backend
            .chat_completion(malformed_multimodal_chat_request())
            .await;
        assert!(
            result.is_err(),
            "malformed media prefill attempt {attempt} unexpectedly succeeded"
        );

        let stats = backend
            .runtime
            .lock()
            .expect("runtime mutex poisoned")
            .session_stats();
        assert_eq!(
            stats.active_sessions, 0,
            "failed multimodal prefill attempt {attempt} leaked an active lane: {stats:?}"
        );
        assert_eq!(
            stats.tracked_token_counts, 0,
            "failed multimodal prefill attempt {attempt} leaked session bookkeeping: {stats:?}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_multimodal_length_limit_then_next_image_uses_clean_lane_when_fixture_is_set()
-> Result<()> {
    let Some(fixture) = multimodal_smoke_fixture()? else {
        return Ok(());
    };
    let mut config = multimodal_stage_config(
        &fixture,
        "stage-0",
        0,
        0,
        fixture.layer_end,
        available_loopback_addr()?,
    );
    config.lane_count = 2;
    let backend = local_openai_backend(config)?;
    backend
        .runtime
        .lock()
        .expect("runtime mutex poisoned")
        .prewarm_idle_sessions(2)?;

    let limited = backend
        .chat_completion(multimodal_chat_request_with_max_tokens(&fixture, 1)?)
        .await?;
    assert_eq!(
        limited.choices[0].finish_reason,
        Some(FinishReason::Length),
        "one-token multimodal generation should finish by length"
    );

    let next = backend
        .chat_completion(multimodal_chat_request(&fixture)?)
        .await?;
    assert_nonempty_chat_response(&next);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_multimodal_split_smoke_when_fixture_is_set() -> Result<()> {
    let Some(fixture) = multimodal_smoke_fixture()? else {
        return Ok(());
    };
    let Some(split_layer) = split_layer_for_fixture(&fixture)? else {
        return Ok(());
    };
    let stage1_addr = available_loopback_addr()?;
    let stage0_addr = available_loopback_addr()?;
    let mut stage1_config = multimodal_stage_config(
        &fixture,
        "stage-1",
        1,
        split_layer,
        fixture.layer_end,
        stage1_addr,
    );
    stage1_config.upstream = Some(skippy_protocol::PeerConfig {
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        endpoint: stage0_addr.to_string(),
    });
    let mut stage0_config =
        multimodal_stage_config(&fixture, "stage-0", 0, 0, split_layer, stage0_addr);
    stage0_config.downstream = Some(skippy_protocol::PeerConfig {
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        endpoint: stage1_addr.to_string(),
    });

    let stage1_handle =
        crate::embedded::start_binary_stage(crate::binary_transport::BinaryStageOptions {
            config: stage1_config,
            topology: None,
            bind_addr: stage1_addr,
            metrics_otlp_grpc: None,
            telemetry_queue_capacity: 1,
            telemetry_level: crate::telemetry::TelemetryLevel::Off,
            max_inflight: 4,
            reply_credit_limit: None,
            async_prefill_forward: false,
            downstream_wire_condition: WireCondition::new(0.0, None)?,
            downstream_connect_timeout_secs: 5,
            native_mtp_enabled: true,
            continuous_batching: true,
            openai: None,
        });
    let ready = connect_endpoint_ready(&stage1_addr.to_string(), 120);
    if let Err(error) = ready {
        let status = stage1_handle.status();
        stage1_handle.abort();
        return Err(error.context(format!(
            "wait for stage-1 binary server; status={:?} last_error={:?}",
            status.state, status.last_error
        )));
    }

    let telemetry = Telemetry::new(
        None,
        1,
        stage0_config.clone(),
        crate::telemetry::TelemetryLevel::Off,
    );
    let lane_pool = PersistentStageLanePool::new(&stage0_config, 1, 5, telemetry.clone())?
        .context("create split smoke lane pool")?;
    let runtime = load_runtime(&stage0_config)?.context("load stage-0 smoke runtime")?;
    let ctx_size = usize::try_from(stage0_config.ctx_size).unwrap_or(usize::MAX);
    let iteration_scheduler =
        IterationScheduler::new(runtime.clone(), &stage0_config, 1, true, telemetry.clone())?;
    let backend = StageOpenAiBackend {
        runtime,
        telemetry,
        config: stage0_config.clone(),
        model_id: "mm-smoke".to_string(),
        default_max_tokens: 16,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size,
        mode: OpenAiBackendMode::EmbeddedStageZero {
            config: stage0_config,
            prefill_chunk_policy: PrefillChunkPolicy::Fixed { chunk_size: 64 },
            activation_width: fixture.activation_width,
            downstream_wire_condition: WireCondition::new(0.0, None)?,
            prefill_reply_credit_limit: 0,
            lane_pool: Some(lane_pool),
            prediction_returns: None,
        },
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_max: 0,
        speculative: SpeculativeDecodeConfig {
            native_mtp: NativeMtpProposalConfig {
                enabled: true,
                max_draft_tokens: 3,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 0,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 0,
            },
            effective_strategy: "native-mtp".to_string(),
            ..SpeculativeDecodeConfig::default()
        },
        generation_limit: Arc::new(GenerationConcurrencyController::fixed(1)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: 1,
        generation_admission_timeout: std::time::Duration::from_secs(10),
        generation_service_estimator: Arc::new(crate::frontend::GenerationServiceEstimator::new(1)),
        generation_session_locks: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(ctx_size)),
        hook_policy: None,
        generation_receipt: None,
        linear_proposal_ingress: None,
        kv: None,
        iteration_scheduler,
    };
    let response = backend
        .chat_completion(multimodal_chat_request(&fixture)?)
        .await;
    stage1_handle.shutdown().await?;

    let response = response?;
    assert_nonempty_chat_response(&response);
    Ok(())
}

#[test]
fn message_content_to_generation_text_inserts_media_markers() {
    let content: MessageContent = serde_json::from_value(json!([
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
    ]))
    .unwrap();
    let mut media = Vec::new();

    let text = message_content_to_generation_text(&content, "<__media__>", &mut media)
        .expect("media text");

    assert_eq!(text, "what is this?\n<__media__>");
    assert_eq!(media.len(), 1);
    assert_eq!(media[0].bytes, b"hello");
}

#[test]
fn message_content_to_generation_text_rejects_remote_media_urls() {
    let content: MessageContent = serde_json::from_value(json!([
        {"type": "input_image", "image_url": "https://example.com/image.png"}
    ]))
    .unwrap();
    let mut media = Vec::new();

    let error =
        message_content_to_generation_text(&content, "<__media__>", &mut media).unwrap_err();

    assert_eq!(
        error.body().error.code.as_deref(),
        Some("unsupported_model_feature")
    );
}

#[test]
fn rescued_audio_media_becomes_text_only_before_prompt_media_extraction() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "please transcribe this"},
                {"type": "input_audio", "input_audio": {
                    "data": "YWJj",
                    "format": "wav"
                }}
            ]
        }],
        "mesh_hooks": true
    }))
    .unwrap();
    let media = openai_frontend::first_chat_media(&request.messages).expect("media");

    apply_chat_hook_outcome(
        &mut request,
        &ChatHookOutcome::injected_with_consumed_media("[Audio context: hello]\n\n", media),
    );

    let content = request.messages[0].content.as_ref().expect("content");
    let mut media = Vec::new();
    let text = message_content_to_generation_text(content, "<__media__>", &mut media)
        .expect("generation text");

    assert!(media.is_empty());
    assert!(!text.contains("<__media__>"));
    assert!(text.contains("[Audio context: hello]"));
    assert!(text.contains("please transcribe this"));
}

#[test]
fn rescued_media_leaves_unhandled_second_media_in_prompt_media() {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "auto",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "compare these"},
                {"type": "input_audio", "input_audio": {
                    "data": "YXVkaW8=",
                    "format": "wav"
                }},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,aW1hZ2U="
                }}
            ]
        }],
        "mesh_hooks": true
    }))
    .unwrap();
    let media = openai_frontend::first_chat_media(&request.messages).expect("media");

    apply_chat_hook_outcome(
        &mut request,
        &ChatHookOutcome::injected_with_consumed_media("[Audio context: hello]\n\n", media),
    );

    let content = request.messages[0].content.as_ref().expect("content");
    let mut media = Vec::new();
    let text = message_content_to_generation_text(content, "<__media__>", &mut media)
        .expect("generation text");

    assert_eq!(media.len(), 1);
    assert_eq!(media[0].bytes, b"image");
    assert_eq!(
        text,
        "[Audio context: hello]\n\n\ncompare these\n<__media__>"
    );
}
