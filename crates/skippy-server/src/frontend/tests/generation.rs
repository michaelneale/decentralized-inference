use super::*;
use crate::frontend::backend::run_blocking_generation_worker;
use std::sync::atomic::AtomicBool;

fn assert_generation_rate_limit(error: OpenAiError, message_fragment: &str) {
    assert_eq!(error.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = error.body();
    assert_eq!(body.error.code.as_deref(), Some("rate_limit_exceeded"));
    assert!(
        body.error.message.contains(message_fragment),
        "expected {:?} to contain {:?}",
        body.error.message,
        message_fragment
    );

    let response = error.into_response();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok()),
        Some("1")
    );
}

#[tokio::test]
async fn generation_admission_uses_open_lane_without_queueing() {
    let generation_limit = Arc::new(Semaphore::new(1));
    let generation_queue_depth = Arc::new(AtomicUsize::new(0));

    let permit = acquire_generation_permit_with_queue(
        generation_limit,
        generation_queue_depth.clone(),
        1,
        Duration::from_millis(10),
    )
    .await
    .unwrap();

    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 0);
    drop(permit);
}

#[tokio::test]
async fn generation_admission_rejects_when_queue_full() {
    let generation_limit = Arc::new(Semaphore::new(0));
    let generation_queue_depth = Arc::new(AtomicUsize::new(1));

    let error = acquire_generation_permit_with_queue(
        generation_limit,
        generation_queue_depth.clone(),
        1,
        Duration::from_millis(10),
    )
    .await
    .unwrap_err();

    assert_generation_rate_limit(error, "queue is full");
    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn generation_admission_times_out_and_releases_queue_slot() {
    let generation_limit = Arc::new(Semaphore::new(0));
    let generation_queue_depth = Arc::new(AtomicUsize::new(0));

    let error = acquire_generation_permit_with_queue(
        generation_limit,
        generation_queue_depth.clone(),
        1,
        Duration::from_millis(5),
    )
    .await
    .unwrap_err();

    assert_generation_rate_limit(error, "timed out waiting");
    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn generation_admission_honors_existing_absolute_deadline() {
    let generation_limit = Arc::new(Semaphore::new(0));
    let generation_queue_depth = Arc::new(AtomicUsize::new(0));
    let reservation = reserve_generation_queue(generation_queue_depth.clone(), 1)
        .expect("generation queue reservation");
    let cancellation = openai_frontend::CancellationToken::new();
    let started = std::time::Instant::now();

    let error = acquire_generation_permit_with_queue_reservation(
        generation_limit,
        reservation,
        Duration::from_secs(1),
        std::time::Instant::now() + Duration::from_millis(5),
        &cancellation,
    )
    .await
    .unwrap_err();

    assert_generation_rate_limit(error, "timed out waiting");
    assert!(
        started.elapsed() < Duration::from_millis(250),
        "admission must use the supplied absolute deadline"
    );
    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn cancelled_queued_admission_releases_queue_slot() {
    let generation_limit = Arc::new(Semaphore::new(0));
    let generation_queue_depth = Arc::new(AtomicUsize::new(0));
    let reservation = reserve_generation_queue(generation_queue_depth.clone(), 1)
        .expect("generation queue reservation");
    let cancellation = openai_frontend::CancellationToken::new();
    let waiter_cancellation = cancellation.clone();
    let waiter = tokio::spawn(async move {
        acquire_generation_permit_with_queue_reservation(
            generation_limit,
            reservation,
            Duration::from_secs(1),
            std::time::Instant::now() + Duration::from_secs(1),
            &waiter_cancellation,
        )
        .await
    });

    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 1);
    cancellation.cancel();
    let result = tokio::time::timeout(Duration::from_millis(100), waiter)
        .await
        .expect("cancelled queue waiter returned")
        .expect("queue waiter task completed");
    let error = match result {
        Ok(_) => panic!("cancelled queue waiter unexpectedly acquired a permit"),
        Err(error) => error,
    };
    assert!(error.body().error.message.contains("request cancelled"));
    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn generation_admission_waits_for_released_lane() {
    let generation_limit = Arc::new(Semaphore::new(0));
    let generation_queue_depth = Arc::new(AtomicUsize::new(0));
    let release_limit = generation_limit.clone();
    let release_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        release_limit.add_permits(1);
    });

    let permit = acquire_generation_permit_with_queue(
        generation_limit,
        generation_queue_depth.clone(),
        1,
        Duration::from_secs(1),
    )
    .await
    .unwrap();

    release_task.await.unwrap();
    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 0);
    drop(permit);
}

#[test]
fn queued_arrival_cannot_be_overtaken_when_a_lane_opens() {
    let queue = Arc::new(GenerationAdmissionQueue::new());
    let generation_limit = Arc::new(Semaphore::new(0));
    let generation_queue_depth = Arc::new(AtomicUsize::new(0));
    let token_budget = Arc::new(GenerationTokenBudget::new(128));

    let first = queue
        .claim_or_enqueue(
            generation_limit.clone(),
            token_budget.clone(),
            GenerationTokenBudgetRequest::new(1, 1),
            GenerationAdmissionScheduling::default(),
            generation_queue_depth.clone(),
            2,
        )
        .unwrap();
    assert!(matches!(&first, GenerationAdmissionClaim::Queued(_)));

    generation_limit.add_permits(1);
    let second = queue
        .claim_or_enqueue(
            generation_limit.clone(),
            token_budget,
            GenerationTokenBudgetRequest::new(1, 1),
            GenerationAdmissionScheduling::default(),
            generation_queue_depth.clone(),
            2,
        )
        .unwrap();

    assert!(matches!(&second, GenerationAdmissionClaim::Queued(_)));
    assert_eq!(generation_limit.available_permits(), 1);
    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 2);
    drop((first, second));
    assert_eq!(generation_queue_depth.load(Ordering::Acquire), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_worker_stops_before_the_next_request_acquires_the_only_lane() {
    let generation_limit = Arc::new(Semaphore::new(1));
    let queue_depth = Arc::new(AtomicUsize::new(0));
    let first_permit = acquire_generation_permit_with_queue(
        generation_limit.clone(),
        queue_depth.clone(),
        1,
        Duration::from_secs(1),
    )
    .await
    .unwrap();
    let first_context = OpenAiRequestContext::new();
    let worker_started = Arc::new(AtomicBool::new(false));
    let worker_stopped = Arc::new(AtomicBool::new(false));
    let started = worker_started.clone();
    let stopped = worker_stopped.clone();
    let first_worker = tokio::spawn(run_blocking_generation_worker(
        first_permit,
        first_context.clone(),
        move |cancellation| {
            started.store(true, Ordering::Release);
            while !cancellation.is_cancelled() {
                std::thread::sleep(Duration::from_millis(1));
            }
            stopped.store(true, Ordering::Release);
        },
    ));

    tokio::time::timeout(Duration::from_millis(100), async {
        while !worker_started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("first generation worker started");

    let second_limit = generation_limit.clone();
    let second_queue_depth = queue_depth.clone();
    let second = tokio::spawn(async move {
        let permit = acquire_generation_permit_with_queue(
            second_limit,
            second_queue_depth,
            1,
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        assert!(
            worker_stopped.load(Ordering::Acquire),
            "the first worker must stop before its permit is released"
        );
        drop(permit);
    });

    first_context.cancel();
    tokio::time::timeout(Duration::from_millis(200), second)
        .await
        .expect("second request acquired the lane promptly")
        .unwrap();
    first_worker.await.unwrap().unwrap();
    assert_eq!(generation_limit.available_permits(), 1);
}

#[test]
fn trims_at_first_stop_sequence() {
    assert_eq!(trim_at_stop("hello END world", &["END"]), "hello ");
    assert_eq!(trim_at_stop("abc xyz def", &["def", "xyz"]), "abc ");
    assert_eq!(trim_at_stop("abc", &[""]), "abc");
}

#[test]
fn generation_stop_values_include_chat_template_stops() {
    let request_stop = openai_frontend::StopSequence::One("</stop>".to_string());
    let metadata = json!({
        "additional_stops": ["<|user|>", "<|observation|>", ""],
    })
    .to_string();

    let stops = generation_stop_values(Some(&request_stop), Some(&metadata));

    assert_eq!(stops, vec!["</stop>", "<|user|>", "<|observation|>"]);
}

#[test]
fn valid_utf8_prefix_skips_incomplete_suffix() {
    assert_eq!(valid_utf8_prefix_len("hello".as_bytes()), 5);
    assert_eq!(valid_utf8_prefix_len(&[b'h', b'i', 0xE2, 0x82]), 2);
    assert_eq!(valid_utf8_prefix_len(&[0xF0, 0x9F, 0x98]), 0);
}

#[test]
fn hook_injected_text_concatenates_injection_actions() {
    let outcome = ChatHookOutcome {
        actions: vec![
            ChatHookAction::InjectText {
                text: "[first]\n".to_string(),
            },
            ChatHookAction::None,
            ChatHookAction::InjectText {
                text: "[second]\n".to_string(),
            },
        ],
    };

    assert_eq!(
        hook_injected_text(&outcome),
        Some("[first]\n[second]\n".to_string())
    );
}

#[test]
fn mid_generation_window_requires_minimum_tokens_and_cooldown() {
    let window = GenerationSignalWindow {
        token_count: 16,
        mean_entropy: 4.5,
        max_entropy: 5.0,
        mean_margin: 0.02,
        min_margin: 0.01,
        high_entropy_count: 12,
        repetition_count: 0,
    };

    assert!(!mid_generation_window_should_fire(11, &None, &window));
    assert!(!mid_generation_window_should_fire(20, &Some(0), &window));
    assert!(mid_generation_window_should_fire(33, &Some(0), &window));
}

#[test]
fn mid_generation_window_fires_on_repetition_even_with_low_entropy() {
    let window = GenerationSignalWindow {
        token_count: 16,
        mean_entropy: 0.3,
        max_entropy: 0.7,
        mean_margin: 0.7,
        min_margin: 0.4,
        high_entropy_count: 0,
        repetition_count: 3,
    };

    assert!(mid_generation_window_should_fire(16, &None, &window));
}

#[test]
fn maps_generation_exhaustion_to_length_finish_reason() {
    assert_eq!(finish_reason_for_generation(true), FinishReason::Length);
    assert_eq!(finish_reason_for_generation(false), FinishReason::Stop);
}

#[test]
fn generation_ids_are_unique_under_fast_creation() {
    let ids = (0..1024)
        .map(|_| {
            OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), None, false, None)
        })
        .collect::<Vec<_>>();
    let mut sessions = std::collections::BTreeSet::new();
    let mut requests = std::collections::BTreeSet::new();
    for id in ids {
        assert!(sessions.insert(id.session_id));
        assert!(requests.insert(id.request_id));
    }
}

#[test]
fn model_matching_is_exact_for_mesh_style_ids() {
    ensure_requested_model(
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
    )
    .unwrap();

    let error = ensure_requested_model(
        "jc-builds/SmolLM2-135M-Instruct-Q4_K_M-GGUF:Q4_K_M",
        "org/repo:Q5_K_M",
    )
    .unwrap_err();
    assert_eq!(error.body().error.code.as_deref(), Some("model_not_found"));
}

#[test]
fn model_matching_normalizes_default_revision() {
    // Advertised with @main, requested without (public display form)
    ensure_requested_model(
        "unsloth/Qwen3-32B-GGUF@main:UD-Q4_K_XL",
        "unsloth/Qwen3-32B-GGUF:UD-Q4_K_XL",
    )
    .unwrap();

    // Advertised without, requested with @main
    ensure_requested_model(
        "unsloth/Qwen3-32B-GGUF:UD-Q4_K_XL",
        "unsloth/Qwen3-32B-GGUF@main:UD-Q4_K_XL",
    )
    .unwrap();

    // Both with @main — exact match still works
    ensure_requested_model(
        "unsloth/Qwen3-32B-GGUF@main:UD-Q4_K_XL",
        "unsloth/Qwen3-32B-GGUF@main:UD-Q4_K_XL",
    )
    .unwrap();

    // Bare repo@main without selector
    ensure_requested_model("org/repo@main", "org/repo").unwrap();

    // Different quants still rejected
    let error = ensure_requested_model(
        "unsloth/Qwen3-32B-GGUF@main:UD-Q4_K_XL",
        "unsloth/Qwen3-32B-GGUF:Q5_K_M",
    )
    .unwrap_err();
    assert_eq!(error.body().error.code.as_deref(), Some("model_not_found"));
}

#[test]
fn rejects_requests_whose_prompt_alone_exceeds_context_window() {
    assert_eq!(context_budget_completion_tokens(4, 8).unwrap(), 4);

    let error = context_budget_completion_tokens(9, 8).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("context_length_exceeded")
    );
}

#[test]
fn omitted_max_tokens_can_use_remaining_context_budget() {
    let limit = GenerationTokenLimit::from_request(None, CONTEXT_BUDGET_MAX_TOKENS);
    assert_eq!(limit.resolve(5, 8).unwrap(), 3);
}

#[test]
fn omitted_max_tokens_with_embedded_default_is_bounded() {
    // Server picked DEFAULT_EMBEDDED_MAX_TOKENS as the cap because the
    // client omitted max_tokens. With a large ctx window the cap is
    // the binding limit.
    let limit = GenerationTokenLimit::from_request(None, DEFAULT_EMBEDDED_MAX_TOKENS);
    let ctx_size = 32_000;
    let resolved = limit.resolve(128, ctx_size).unwrap();
    assert_eq!(resolved, DEFAULT_EMBEDDED_MAX_TOKENS);
    assert!((resolved as usize) < ctx_size);
}

#[test]
fn omitted_max_tokens_clamps_to_remaining_budget_in_small_ctx() {
    // When the configured ctx_size is smaller than the server-picked
    // default, the omitted-max_tokens path must clamp to remaining
    // budget rather than reject the request. The client didn't ask
    // for the specific number; the server picked it.
    let limit = GenerationTokenLimit::from_request(None, DEFAULT_EMBEDDED_MAX_TOKENS);
    let ctx_size = 1024;
    let prompt_tokens = 128;
    let resolved = limit.resolve(prompt_tokens, ctx_size).unwrap();
    assert_eq!(resolved, (ctx_size - prompt_tokens) as u32);
    assert!(resolved < DEFAULT_EMBEDDED_MAX_TOKENS);
}

#[test]
fn omitted_max_tokens_errors_only_when_prompt_already_exceeds_ctx() {
    // Even on the silently-clamping default path, a prompt that
    // already overflows the context window is an error the client
    // needs to see.
    let limit = GenerationTokenLimit::from_request(None, DEFAULT_EMBEDDED_MAX_TOKENS);
    let error = limit.resolve(2048, 1024).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("context_length_exceeded")
    );
}

#[test]
fn explicit_max_tokens_clamps_to_remaining_context() {
    // `max_tokens` is a ceiling, not a reservation. A client asking for more
    // than fits gets the remaining context and stops at `length`, matching the
    // server-picked default path. Only a prompt that overflows on its own is
    // an error. See issue #1350.
    let limit = GenerationTokenLimit::from_request(Some(4), 999);
    assert_eq!(limit.resolve(4, 8).unwrap(), 4);
    assert_eq!(limit.resolve(5, 8).unwrap(), 3);

    let limit = GenerationTokenLimit::from_request(Some(65_536), 999);
    assert_eq!(limit.resolve(8_911, 65_536).unwrap(), 65_536 - 8_911);

    let error = limit.resolve(65_537, 65_536).unwrap_err();
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("context_length_exceeded")
    );
}

#[test]
fn strip_default_revision_removes_at_main_before_quant() {
    assert_eq!(
        super::strip_default_revision("org/repo@main:Q4"),
        "org/repo:Q4"
    );
}

#[test]
fn strip_default_revision_removes_at_main_at_end() {
    assert_eq!(super::strip_default_revision("org/repo@main"), "org/repo");
}

#[test]
fn strip_default_revision_preserves_mainland() {
    assert_eq!(
        super::strip_default_revision("org/repo@mainland:Q4"),
        "org/repo@mainland:Q4"
    );
}

#[test]
fn strip_default_revision_preserves_no_revision() {
    assert_eq!(super::strip_default_revision("org/repo:Q4"), "org/repo:Q4");
}
