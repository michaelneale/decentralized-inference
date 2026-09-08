use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::{thread, time::Duration};

use crate::frontend::SpeculativeDecodeConfig;
use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::generation::{
    GenerationConcurrencyController, LocalGeneration, OpenAiBackendMode, OpenAiCacheHints,
    OpenAiGenerationIds, StageOpenAiBackend, TokenControl,
};
use crate::frontend::iteration_scheduler::IterationScheduler;
use crate::frontend::local_generation::{
    linear_proposal_allowed, native_mtp_dispatch_counts_for_test, post_decode_checkpoint_tokens,
    prompt_fits_single_prefill_sample,
};
use crate::frontend::{
    EmbeddedOpenAiRequestDefaults, GenerationAbort, GenerationCommit, GenerationReceipt,
    GenerationReceiptConfig, GenerationReceiptSink, GenerationStart, GenerationTermination,
};
use crate::kv_integration::KvStageIntegration;
use crate::runtime_state::{RuntimeState, load_runtime};
use crate::telemetry::{Telemetry, TelemetryLevel};
use anyhow::{Result, bail};
use openai_frontend::{ChatCompletionRequest, OpenAiBackend};
use skippy_protocol::{
    LoadMode, StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
};
use skippy_runtime::SamplingConfig;

// The real-model tests below are deliberately ignored by default. The
// fixture contract is explicit: run them with both model-path variables set;
// optionally set SKIPPY_RECURRENT_CACHE_TEST_MODEL_ID to override the model id.
// For example:
//
// SKIPPY_RECURRENT_CACHE_TEST_MODEL=/path/model.gguf \
// SKIPPY_RECURRENT_CACHE_TEST_MODEL_LAYERS=40 \
// just with-lld cargo test -p skippy-server recurrent_ --lib -- --ignored --nocapture

#[derive(Default)]
struct RecordingReceiptSink {
    receipts: Mutex<Vec<GenerationReceipt>>,
    commits: Mutex<Vec<GenerationCommit>>,
    fail: AtomicBool,
}

impl GenerationReceiptSink for RecordingReceiptSink {
    fn begin(&self, _start: &GenerationStart) -> Result<()> {
        Ok(())
    }

    fn committed(&self, commit: &GenerationCommit) -> Result<()> {
        self.commits.lock().unwrap().push(commit.clone());
        Ok(())
    }

    fn abort(&self, _abort: &GenerationAbort) -> Result<()> {
        Ok(())
    }

    fn record(&self, receipt: &GenerationReceipt) -> Result<()> {
        self.receipts.lock().unwrap().push(receipt.clone());
        if self.fail.load(Ordering::Relaxed) {
            bail!("synthetic generation receipt sink failure");
        }
        Ok(())
    }
}

fn wait_for_receipts(sink: &RecordingReceiptSink, expected: usize) {
    for _ in 0..100 {
        if sink.receipts.lock().unwrap().len() >= expected {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }
    panic!("timed out waiting for {expected} generation receipts");
}

fn recurrent_test_backend(
    run_id: &str,
    backend_model_id: &str,
    ctx_size: usize,
    batch_size: u32,
    default_max_tokens: u32,
    token_budget: usize,
) -> Result<(StageOpenAiBackend, SpeculativeDecodeConfig)> {
    let model_path = std::env::var_os("SKIPPY_RECURRENT_CACHE_TEST_MODEL").ok_or_else(|| {
        anyhow::anyhow!("SKIPPY_RECURRENT_CACHE_TEST_MODEL is required for this ignored test")
    })?;
    let layer_count =
        std::env::var_os("SKIPPY_RECURRENT_CACHE_TEST_MODEL_LAYERS").ok_or_else(|| {
            anyhow::anyhow!(
                "SKIPPY_RECURRENT_CACHE_TEST_MODEL_LAYERS is required for this ignored test"
            )
        })?;
    let layer_count = layer_count
        .to_string_lossy()
        .parse::<u32>()
        .map_err(|error| anyhow::anyhow!("invalid recurrent cache test layer count: {error}"))?;
    let config = StageConfig {
        run_id: run_id.to_string(),
        topology_id: run_id.to_string(),
        model_id: std::env::var("SKIPPY_RECURRENT_CACHE_TEST_MODEL_ID")
            .unwrap_or_else(|_| "unsloth/Qwen3.5-0.8B-GGUF:Q6_K".to_string()),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(model_path.to_string_lossy().into_owned()),
        projector_path: None,
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: layer_count,
        ctx_size: u32::try_from(ctx_size)
            .map_err(|error| anyhow::anyhow!("invalid recurrent test context size: {error}"))?,
        lane_count: 1,
        n_batch: Some(batch_size),
        n_ubatch: Some(batch_size),
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
        kv_cache: Some(StageKvCacheConfig {
            mode: StageKvCacheMode::LookupRecord,
            payload: StageKvCachePayload::KvRecurrent,
            max_entries: 8,
            max_bytes: 0,
            min_tokens: 1,
            shared_prefix_stride_tokens: 1,
            shared_prefix_record_limit: 0,
        }),
        native_mtp_enabled: false,
        load_mode: LoadMode::RuntimeSlice,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: None,
        ..StageConfig::default()
    };
    let runtime = load_runtime(&config)?
        .ok_or_else(|| anyhow::anyhow!("recurrent cache test runtime was not loaded"))?;
    let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Recurrent)?
        .map(Arc::new)
        .ok_or_else(|| anyhow::anyhow!("recurrent cache test did not enable KV integration"))?;
    let telemetry = Telemetry::new(None, 1, config.clone(), TelemetryLevel::Off);
    let speculative = SpeculativeDecodeConfig::default();
    let iteration_scheduler =
        IterationScheduler::new(runtime.clone(), &config, 1, true, telemetry.clone())?;
    let backend = StageOpenAiBackend {
        runtime: runtime.clone(),
        config,
        telemetry,
        model_id: backend_model_id.to_string(),
        default_max_tokens,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size,
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
        generation_service_estimator: Arc::new(crate::frontend::GenerationServiceEstimator::new(1)),
        generation_session_locks: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(token_budget)),
        hook_policy: None,
        generation_receipt: None,
        linear_proposal_ingress: None,
        kv: Some(kv),
        iteration_scheduler,
    };
    Ok((backend, speculative))
}

#[test]
fn single_prefill_sample_requires_prompt_to_fit_session_batch() {
    assert!(!prompt_fits_single_prefill_sample(0, 2048));
    assert!(!prompt_fits_single_prefill_sample(1, 2048));
    assert!(prompt_fits_single_prefill_sample(2048, 2048));
    assert!(!prompt_fits_single_prefill_sample(2049, 2048));
}

#[test]
fn local_generation_signal_window_uses_configured_value() {
    let config = StageConfig {
        run_id: "signal-window-test".to_string(),
        topology_id: "signal-window-test".to_string(),
        model_id: "signal-window-test".to_string(),
        generation_signal_window: Some(37),
        ..StageConfig::default()
    };
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let telemetry = Telemetry::new(None, 1, config.clone(), TelemetryLevel::Off);
    let iteration_scheduler =
        IterationScheduler::new(runtime.clone(), &config, 1, true, telemetry.clone()).unwrap();
    let backend = StageOpenAiBackend {
        runtime,
        config,
        telemetry,
        model_id: "signal-window-test".to_string(),
        default_max_tokens: 1,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size: 4096,
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
        generation_token_budget: Arc::new(GenerationTokenBudget::new(4096)),
        hook_policy: None,
        generation_receipt: None,
        linear_proposal_ingress: None,
        kv: None,
        iteration_scheduler,
    };

    assert_eq!(backend.generation_signal_window_tokens(), 37);
}

#[test]
fn local_native_mtp_decode_uses_non_frame_runtime_api() {
    let (sampled_calls, frame_calls) = native_mtp_dispatch_counts_for_test();
    assert_eq!(sampled_calls, 1);
    assert_eq!(frame_calls, 0);
}

#[test]
fn post_decode_checkpoint_names_only_tokens_consumed_by_runtime() {
    let prompt = [10, 11, 12];

    // One decode consumes the final prompt token and returns token 20. The
    // recurrent state is therefore exactly at the full-prompt boundary; token
    // 20 has been emitted but has not yet been fed back into the model.
    assert_eq!(
        post_decode_checkpoint_tokens(&prompt, &[20]),
        Some(vec![10, 11, 12])
    );

    // Later decodes consume every generated token except the newest one.
    assert_eq!(
        post_decode_checkpoint_tokens(&prompt, &[20, 21, 22]),
        Some(vec![10, 11, 12, 20, 21])
    );
    assert_eq!(post_decode_checkpoint_tokens(&prompt, &[]), None);
}

#[test]
fn recurrent_cache_gates_initial_linear_proposal_until_checkpoint() {
    // An initial proposal can commit multiple tokens, so its post-proposal
    // boundary cannot name the original full prompt. The first proposal must
    // wait for the serial checkpoint instead.
    assert!(!linear_proposal_allowed(true, false));
    assert!(linear_proposal_allowed(true, true));

    // Non-recurrent paths keep their existing proposal scheduling.
    assert!(linear_proposal_allowed(false, false));
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn recurrent_post_decode_checkpoint_reuses_a_growing_prompt() -> Result<()> {
    let (backend, speculative) = recurrent_test_backend(
        "recurrent-cache-test",
        "recurrent-cache-test",
        128,
        32,
        2,
        128,
    )?;
    let sampling = SamplingConfig::default();
    let first_prompt = [1, 2, 3];
    let first_ids = OpenAiGenerationIds::new_with_trust(
        OpenAiCacheHints::default(),
        Some("recurrent-cache-test"),
        true,
    );
    let mut first_output = Vec::new();
    let first_stats = backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: &first_prompt,
            recurrent_cache_prefix_token_ids: None,
            max_tokens: 2,
            sampling: &sampling,
            chat_sampling_metadata: None,
            speculative: &speculative,
            native_mtp_enabled: false,
            hook_request: None,
            hook_runtime: None,
            cancellation: None,
            ids: &first_ids,
        },
        |token_id| {
            first_output.push(token_id);
            Ok(TokenControl::Continue)
        },
    )?;
    assert_eq!(first_stats.cached_prompt_tokens, 0);
    assert_eq!(first_output.len(), 2);

    // The next prompt extends the exact state captured after the first
    // generated token was consumed. The final token is deliberately new so
    // the second request is a growing prompt rather than an exact replay.
    let second_prompt = [
        first_prompt[0],
        first_prompt[1],
        first_prompt[2],
        first_output[0],
        4,
    ];
    let second_ids = OpenAiGenerationIds::new_with_trust(
        OpenAiCacheHints::default(),
        Some("recurrent-cache-test"),
        true,
    );
    let second_stats = backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: &second_prompt,
            recurrent_cache_prefix_token_ids: None,
            max_tokens: 1,
            sampling: &sampling,
            chat_sampling_metadata: None,
            speculative: &speculative,
            native_mtp_enabled: false,
            hook_request: None,
            hook_runtime: None,
            cancellation: None,
            ids: &second_ids,
        },
        |_| Ok(TokenControl::Continue),
    )?;
    assert_eq!(second_stats.status, "hit");
    assert_eq!(second_stats.cached_prompt_tokens, 4);
    assert_eq!(second_stats.matched_prefix_tokens, 4);
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn recurrent_chat_checkpoint_preserves_cached_output_parity() -> Result<()> {
    let (backend, _speculative) = recurrent_test_backend(
        "recurrent-chat-cache-test",
        "recurrent-chat-cache-test",
        512,
        64,
        8,
        512,
    )?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let first_request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "recurrent-chat-cache-test",
            "messages": [
                {"role": "system", "content": "You are a deterministic cache test."},
                {"role": "user", "content": "Reply with a short plain answer: CACHE_SEED"}
            ],
            "max_tokens": 8,
            "temperature": 0,
            "reasoning_effort": "none",
            "prompt_cache_key": "recurrent-chat-cache"
        }))?;
        let first_prompt = backend.prepare_chat_prompt(
            &first_request,
            crate::frontend::request::chat_template_options(
                &first_request,
                &backend.request_defaults,
            )?,
        )?;
        let first_prompt_tokens = backend.tokenize(&first_prompt.text)?;
        let first_boundary_tokens = backend.tokenize(
            first_prompt
                .recurrent_cache_prefix_text
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("first chat prompt had no cache boundary"))?,
        )?;
        assert!(!first_boundary_tokens.is_empty());
        assert!(first_boundary_tokens.len() < first_prompt_tokens.len());
        assert_eq!(
            &first_prompt_tokens[..first_boundary_tokens.len()],
            first_boundary_tokens.as_slice(),
            "first rendered prompt boundary was not an exact token prefix"
        );

        let first_response = backend.chat_completion(first_request).await?;
        let first_content = first_response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("first chat response had no assistant content"))?;
        if first_content.is_empty() {
            return Err(anyhow::anyhow!(
                "first chat response had empty assistant content"
            ));
        }

        let second_messages = serde_json::json!([
            {"role": "system", "content": "You are a deterministic cache test."},
            {"role": "user", "content": "Reply with a short plain answer: CACHE_SEED"},
            {"role": "assistant", "content": first_content},
            {"role": "user", "content": "Now reply with a short plain answer: CACHE_TAIL"}
        ]);
        let cached_request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "recurrent-chat-cache-test",
            "messages": second_messages.clone(),
            "max_tokens": 4,
            "temperature": 0,
            "reasoning_effort": "none",
            "prompt_cache_key": "recurrent-chat-cache"
        }))?;
        let second_prompt = backend.prepare_chat_prompt(
            &cached_request,
            crate::frontend::request::chat_template_options(
                &cached_request,
                &backend.request_defaults,
            )?,
        )?;
        let second_prompt_tokens = backend.tokenize(&second_prompt.text)?;
        assert!(second_prompt_tokens.len() > first_boundary_tokens.len());
        assert_eq!(
            &second_prompt_tokens[..first_boundary_tokens.len()],
            first_boundary_tokens.as_slice(),
            "first message-history boundary was not a prefix of growing chat prompt"
        );
        let cached_response = backend.chat_completion(cached_request).await?;
        let cached_content = cached_response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("cached chat response had no assistant content"))?;
        let cached_tokens = cached_response
            .usage
            .prompt_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens)
            .unwrap_or(0);

        let uncached_request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "recurrent-chat-cache-test",
            "messages": second_messages,
            "max_tokens": 4,
            "temperature": 0,
            "reasoning_effort": "none",
            "prompt_cache_key": "recurrent-chat-cache-control"
        }))?;
        let uncached_response = backend.chat_completion(uncached_request).await?;
        let uncached_content = uncached_response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .ok_or_else(|| anyhow::anyhow!("uncached chat response had no assistant content"))?;
        let uncached_tokens = uncached_response
            .usage
            .prompt_tokens_details
            .as_ref()
            .map(|details| details.cached_tokens)
            .unwrap_or(0);

        assert!(
            cached_tokens > 0,
            "growing chat request did not hit KV cache"
        );
        assert_eq!(uncached_tokens, 0);
        assert_eq!(cached_content, uncached_content);
        Ok::<(), anyhow::Error>(())
    })
}

#[test]
#[ignore = "requires SKIPPY_GENERATION_RECEIPT_MODEL and _LAYERS; run explicitly with --ignored"]
fn local_generation_eventually_delivers_receipts_and_cleanup_survives_sink_errors() -> Result<()> {
    let model_path = std::env::var_os("SKIPPY_GENERATION_RECEIPT_MODEL").ok_or_else(|| {
        anyhow::anyhow!("SKIPPY_GENERATION_RECEIPT_MODEL is required for this ignored test")
    })?;
    let layer_count =
        std::env::var_os("SKIPPY_GENERATION_RECEIPT_MODEL_LAYERS").ok_or_else(|| {
            anyhow::anyhow!(
                "SKIPPY_GENERATION_RECEIPT_MODEL_LAYERS is required for this ignored test"
            )
        })?;
    let layer_count = layer_count
        .to_string_lossy()
        .parse::<u32>()
        .map_err(|error| anyhow::anyhow!("invalid receipt test layer count: {error}"))?;
    let config = StageConfig {
        run_id: "generation-receipt-test".to_string(),
        topology_id: "generation-receipt-test".to_string(),
        model_id: "generation-receipt-test".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(model_path.to_string_lossy().into_owned()),
        projector_path: None,
        stage_id: "stage-0".to_string(),
        stage_index: 0,
        layer_start: 0,
        layer_end: layer_count,
        ctx_size: 128,
        lane_count: 1,
        n_batch: Some(32),
        n_ubatch: Some(32),
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
    let runtime = load_runtime(&config)?
        .ok_or_else(|| anyhow::anyhow!("receipt test runtime was not loaded"))?;
    let sink = Arc::new(RecordingReceiptSink::default());
    let telemetry = Telemetry::new(None, 1, config.clone(), TelemetryLevel::Off);
    let speculative = SpeculativeDecodeConfig::default();
    let iteration_scheduler =
        IterationScheduler::new(runtime.clone(), &config, 1, true, telemetry.clone())?;
    let backend = StageOpenAiBackend {
        runtime: runtime.clone(),
        config,
        telemetry,
        model_id: "generation-receipt-test".to_string(),
        default_max_tokens: 1,
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
        generation_service_estimator: Arc::new(crate::frontend::GenerationServiceEstimator::new(1)),
        generation_session_locks: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(128)),
        hook_policy: None,
        generation_receipt: Some(GenerationReceiptConfig::new(sink.clone())),
        linear_proposal_ingress: None,
        kv: None,
        iteration_scheduler,
    };
    let sampling = SamplingConfig::default();
    // A multi-token prompt takes the whole-prompt prefill path. Keep this
    // above one token so the test exercises a fresh runtime session before
    // its batch size is queried.
    let prompt_token_ids = [1, 2];
    let ids = OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), None, false);
    let mut emitted = Vec::new();
    backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: &prompt_token_ids,
            recurrent_cache_prefix_token_ids: None,
            max_tokens: 1,
            sampling: &sampling,
            chat_sampling_metadata: None,
            speculative: &speculative,
            native_mtp_enabled: false,
            hook_request: None,
            hook_runtime: None,
            cancellation: None,
            ids: &ids,
        },
        |token_id| {
            emitted.push(token_id);
            Ok(TokenControl::Continue)
        },
    )?;

    wait_for_receipts(&sink, 1);
    let commits = sink.commits.lock().unwrap();
    assert_eq!(commits.len(), emitted.len());
    let mut committed_tokens = Vec::new();
    for (index, commit) in commits.iter().enumerate() {
        assert_eq!(commit.request_id, ids.request_id);
        assert_eq!(commit.session_id, ids.session_id);
        assert_eq!(commit.generated_token_count, index + 1);
        assert_eq!(commit.token_ids.len(), 1);
        committed_tokens.extend_from_slice(&commit.token_ids);
    }
    assert_eq!(committed_tokens, emitted);
    drop(commits);
    let receipts = sink.receipts.lock().unwrap();
    assert_eq!(receipts.len(), 1);
    assert_eq!(receipts[0].request_id, ids.request_id);
    assert_eq!(receipts[0].session_id, ids.session_id);
    assert_eq!(receipts[0].prompt_token_count, prompt_token_ids.len());
    assert_eq!(receipts[0].generated_token_ids.as_ref(), emitted.as_slice());
    assert_eq!(receipts[0].termination, GenerationTermination::MaxTokens);
    assert!(receipts[0].final_session_position >= prompt_token_ids.len() as u64);
    drop(receipts);
    assert!(
        runtime
            .lock()
            .unwrap()
            .session_stats()
            .lanes
            .iter()
            .all(|lane| lane.session_id.as_deref() != Some(&ids.session_label))
    );

    sink.fail.store(true, Ordering::Relaxed);
    let failing_ids = OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), None, false);
    backend.generate_local_tokens(
        LocalGeneration {
            prompt_token_ids: &prompt_token_ids,
            recurrent_cache_prefix_token_ids: None,
            max_tokens: 1,
            sampling: &sampling,
            chat_sampling_metadata: None,
            speculative: &speculative,
            native_mtp_enabled: false,
            hook_request: None,
            hook_runtime: None,
            cancellation: None,
            ids: &failing_ids,
        },
        |_| Ok(TokenControl::Continue),
    )?;
    wait_for_receipts(&sink, 2);
    assert!(
        runtime
            .lock()
            .unwrap()
            .session_stats()
            .lanes
            .iter()
            .all(|lane| lane.session_id.as_deref() != Some(&failing_ids.session_label))
    );

    Ok(())
}
