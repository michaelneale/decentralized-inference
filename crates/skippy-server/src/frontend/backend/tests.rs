use super::*;
use crate::frontend::EmbeddedOpenAiRequestDefaults;
use crate::frontend::SpeculativeDecodeConfig;
use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::generation::ADMISSION_STARVATION_BOUND_TURNS;
use crate::frontend::generation::OpenAiBackendMode;
use crate::frontend::iteration_scheduler::IterationScheduler;
use crate::runtime_state::RuntimeState;
use futures_util::StreamExt;
use openai_frontend::ChatCompletionChunk;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::ChatCompletionResponse;
use openai_frontend::ChatHookOutcome;
use openai_frontend::FinishReason;
use openai_frontend::OpenAiHookPolicy;
use openai_frontend::Usage;
use openai_frontend::set_chat_mesh_hooks_enabled;
use serde_json::json;
use tokio::runtime::Runtime;

/// A disabled telemetry sink for `StreamEventSender` construction in tests.
///
/// `TelemetryLevel::Off` makes `emit` a no-op, so these tests exercise the
/// stall/drop control flow without needing a collector; the sink only has to
/// be a valid handle.
fn test_telemetry() -> crate::telemetry::Telemetry {
    let config: skippy_protocol::StageConfig = serde_json::from_value(json!({
        "run_id": "run",
        "topology_id": "topology",
        "model_id": "org/model:Q4_K_M",
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 4,
        "load_mode": "runtime-slice",
        "bind_addr": "127.0.0.1:0",
    }))
    .expect("minimal stage config for telemetry");
    crate::telemetry::Telemetry::new(None, 1, config, crate::telemetry::TelemetryLevel::Off)
}

fn trusted_ids(session_id: &str) -> OpenAiGenerationIds {
    OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), Some(session_id), true, None)
}

fn trusted_session_key(session_id: &str) -> String {
    trusted_generation_session_key(&trusted_ids(session_id)).expect("trusted session key")
}

fn admission_controller(
    generation_concurrency: usize,
    generation_queue_limit: usize,
) -> GenerationAdmissionController {
    admission_controller_with_budget(generation_concurrency, generation_queue_limit, 4_096)
}

fn admission_controller_with_budget(
    generation_concurrency: usize,
    generation_queue_limit: usize,
    token_capacity: usize,
) -> GenerationAdmissionController {
    GenerationAdmissionController {
        generation_limit: Arc::new(GenerationConcurrencyController::fixed(
            generation_concurrency,
        )),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit,
        generation_service_estimator: Arc::new(GenerationServiceEstimator::new(
            generation_concurrency,
        )),
        generation_session_locks: Arc::new(Mutex::new(BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(token_capacity)),
    }
}

fn result_error<T>(result: OpenAiResult<T>) -> OpenAiError {
    match result {
        Ok(_) => panic!("expected generation admission to fail"),
        Err(error) => error,
    }
}

#[tokio::test]
async fn queued_admission_balances_shared_prefix_families_across_lane_wave() {
    let controller = admission_controller(1, 4);
    let work = GenerationAdmissionWork::new(4, 1);
    let active = controller
        .acquire_work(
            &trusted_ids("active"),
            &openai_frontend::CancellationToken::new(),
            Duration::from_secs(2),
            work,
        )
        .await
        .expect("active request admission");
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    for (label, prompt) in [
        ("family-a-1", vec![1, 1, 3, 4]),
        ("family-a-2", vec![1, 1, 3, 5]),
        ("family-b-1", vec![2, 2, 3, 4]),
        ("family-b-2", vec![2, 2, 3, 5]),
    ] {
        let controller = controller.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            let cancellation = openai_frontend::CancellationToken::new();
            let admitted = controller
                .acquire_scheduled_work(
                    &trusted_ids(label),
                    &cancellation,
                    Duration::from_secs(2),
                    work,
                    GenerationAdmissionScheduling::new(
                        Arc::from(prompt),
                        Arc::new(skippy_scheduler::CacheAffinity::default),
                    ),
                )
                .await
                .expect("queued request admission");
            tx.send((label, admitted)).await.unwrap();
        });
    }
    drop(tx);
    tokio::time::timeout(Duration::from_secs(1), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 4 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all prompts become scheduler-visible");

    drop(active);
    let (first_label, first) = tokio::time::timeout(Duration::from_millis(250), rx.recv())
        .await
        .expect("first queued admission was promoted")
        .expect("first queued admission");
    drop(first);
    let (second_label, second) = rx.recv().await.expect("second queued admission");
    assert_ne!(
        first_label.split('-').nth(1),
        second_label.split('-').nth(1),
        "one family must not drain the whole lane wave"
    );
    drop(second);
    let (_, third) = rx.recv().await.expect("third queued admission");
    drop(third);
    let (_, fourth) = rx.recv().await.expect("fourth queued admission");
    drop(fourth);
}

#[tokio::test]
async fn capacity_waiter_holds_neither_a_lane_nor_kv_until_atomic_promotion() {
    let controller = admission_controller_with_budget(2, 2, 10);
    let active = controller
        .acquire_work(
            &trusted_ids("active"),
            &openai_frontend::CancellationToken::new(),
            Duration::ZERO,
            GenerationAdmissionWork::new(7, 0),
        )
        .await
        .expect("first capacity reservation");
    let waiting_controller = controller.clone();
    let waiter = tokio::spawn(async move {
        waiting_controller
            .acquire_work(
                &trusted_ids("waiting"),
                &openai_frontend::CancellationToken::new(),
                Duration::ZERO,
                GenerationAdmissionWork::new(7, 0),
            )
            .await
    });

    tokio::time::timeout(Duration::from_millis(100), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("capacity waiter entered the queue");
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_token_budget.active_tokens(), 7);

    drop(active);
    let promoted = tokio::time::timeout(Duration::from_millis(100), waiter)
        .await
        .expect("capacity waiter promoted")
        .expect("capacity waiter task completed")
        .expect("capacity waiter admission");
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_token_budget.active_tokens(), 7);
    drop(promoted);
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
}

#[tokio::test]
async fn capacity_waiters_drain_serially_after_each_kv_release() {
    let controller = admission_controller_with_budget(2, 4, 10);
    let active = controller
        .acquire_work(
            &trusted_ids("active"),
            &openai_frontend::CancellationToken::new(),
            Duration::ZERO,
            GenerationAdmissionWork::new(10, 0),
        )
        .await
        .expect("active capacity reservation");
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    for index in 0..4 {
        let controller = controller.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            let cancellation = openai_frontend::CancellationToken::new();
            let admitted = controller
                .acquire_work(
                    &trusted_ids(&format!("waiting-{index}")),
                    &cancellation,
                    Duration::ZERO,
                    GenerationAdmissionWork::new(10, 0),
                )
                .await
                .expect("queued capacity admission");
            tx.send(admitted).await.unwrap();
        });
    }
    drop(tx);
    tokio::time::timeout(Duration::from_secs(1), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 4 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("all capacity waiters entered the queue");

    drop(active);
    for _ in 0..4 {
        let admitted = tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("next capacity waiter promoted")
            .expect("queued capacity admission");
        assert_eq!(controller.generation_token_budget.active_tokens(), 10);
        drop(admitted);
    }
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
}

#[tokio::test]
async fn full_pool_waiter_is_admitted_after_bounded_half_pool_bypasses() {
    let controller = admission_controller_with_budget(2, 4, 10);
    let half_pool_work = GenerationAdmissionWork::new(5, 0);
    let full_pool_work = GenerationAdmissionWork::new(10, 0);
    let active = controller
        .acquire_work(
            &trusted_ids("active-half-pool"),
            &openai_frontend::CancellationToken::new(),
            Duration::ZERO,
            half_pool_work,
        )
        .await
        .expect("initial half-pool admission");

    let full_pool_controller = controller.clone();
    let full_pool_waiter = tokio::spawn(async move {
        full_pool_controller
            .acquire_work(
                &trusted_ids("full-pool-waiter"),
                &openai_frontend::CancellationToken::new(),
                Duration::ZERO,
                full_pool_work,
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("full-pool request entered the queue");

    // Keep one half-pool reservation active and repeatedly fill/release the
    // other half. The full-pool request cannot fit during these admissions.
    for wave in 0..ADMISSION_STARVATION_BOUND_TURNS {
        let bypass = tokio::time::timeout(
            Duration::from_secs(1),
            controller.acquire_work(
                &trusted_ids(&format!("half-pool-bypass-{wave}")),
                &openai_frontend::CancellationToken::new(),
                Duration::ZERO,
                half_pool_work,
            ),
        )
        .await
        .expect("fitting half-pool request completed before the timeout")
        .expect("fitting half-pool request admitted before the bound");
        assert_eq!(controller.generation_token_budget.active_tokens(), 10);
        drop(bypass);
    }

    // The next fitting arrival reaches the bound and must remain queued while
    // the controller drains capacity toward the older full-pool request.
    let bypass_controller = controller.clone();
    let blocked_bypass = tokio::spawn(async move {
        bypass_controller
            .acquire_work(
                &trusted_ids("half-pool-after-bound"),
                &openai_frontend::CancellationToken::new(),
                Duration::ZERO,
                half_pool_work,
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("post-bound half-pool request remained queued");
    assert!(!full_pool_waiter.is_finished());
    assert!(!blocked_bypass.is_finished());

    drop(active);
    let admitted_full_pool = tokio::time::timeout(Duration::from_secs(1), full_pool_waiter)
        .await
        .expect("full-pool waiter admitted after capacity drained")
        .expect("full-pool waiter task completed")
        .expect("full-pool admission succeeded");
    assert_eq!(controller.generation_token_budget.active_tokens(), 10);
    assert!(!blocked_bypass.is_finished());

    drop(admitted_full_pool);
    let admitted_bypass = tokio::time::timeout(Duration::from_secs(1), blocked_bypass)
        .await
        .expect("younger half-pool waiter admitted after the senior completed")
        .expect("half-pool waiter task completed")
        .expect("half-pool admission succeeded");
    drop(admitted_bypass);
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
}

#[tokio::test]
async fn request_larger_than_the_kv_pool_fails_without_queueing_or_taking_a_lane() {
    let controller = admission_controller_with_budget(2, 2, 128);
    let error = result_error(
        controller
            .acquire_work(
                &trusted_ids("too-large"),
                &openai_frontend::CancellationToken::new(),
                Duration::ZERO,
                GenerationAdmissionWork::new(129, 0),
            )
            .await,
    );

    assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
    assert_eq!(
        error.body().error.code.as_deref(),
        Some("context_length_exceeded")
    );
    assert!(
        error
            .body()
            .error
            .message
            .contains("runtime pool holds 128")
    );
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
}

#[tokio::test]
async fn cancelling_a_capacity_waiter_leaks_neither_lane_nor_kv_reservation() {
    let controller = admission_controller_with_budget(2, 2, 128);
    let active = controller
        .acquire_work(
            &trusted_ids("active"),
            &openai_frontend::CancellationToken::new(),
            Duration::ZERO,
            GenerationAdmissionWork::new(100, 0),
        )
        .await
        .expect("first capacity reservation");
    let cancellation = openai_frontend::CancellationToken::new();
    let waiter_cancellation = cancellation.clone();
    let waiting_controller = controller.clone();
    let waiter = tokio::spawn(async move {
        waiting_controller
            .acquire_work(
                &trusted_ids("waiting"),
                &waiter_cancellation,
                Duration::ZERO,
                GenerationAdmissionWork::new(64, 0),
            )
            .await
    });
    tokio::time::timeout(Duration::from_millis(100), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("capacity waiter entered the queue");

    cancellation.cancel();
    let error = result_error(
        tokio::time::timeout(Duration::from_millis(100), waiter)
            .await
            .expect("cancelled capacity waiter returned")
            .expect("capacity waiter task completed"),
    );
    assert!(error.body().error.message.contains("request cancelled"));
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_token_budget.active_tokens(), 100);

    drop(active);
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert_eq!(controller.generation_token_budget.active_tokens(), 0);
}

#[tokio::test]
async fn predicted_wait_rejection_preserves_queue_capacity() {
    let controller = admission_controller(1, 2);
    let work = GenerationAdmissionWork::new(100, 100);
    controller
        .generation_service_estimator
        .observe_completed(work, 100.0, 100.0);
    let active = controller
        .acquire_work(
            &trusted_ids("agent-1"),
            &openai_frontend::CancellationToken::new(),
            Duration::from_secs(1),
            work,
        )
        .await
        .expect("active request admission");

    let error = result_error(
        controller
            .acquire_work(
                &trusted_ids("agent-2"),
                &openai_frontend::CancellationToken::new(),
                Duration::from_millis(199),
                work,
            )
            .await,
    );

    assert!(
        error
            .body()
            .error
            .message
            .contains("predicted generation wait")
    );
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    drop(active);
    assert_eq!(controller.generation_limit.available_permits(), 1);
}

#[test]
fn session_registry_counts_live_leases_and_cleans_replaced_entries() {
    let registry = Arc::new(Mutex::new(BTreeMap::new()));
    let first = GenerationSessionPermit::new(registry.clone(), "agent-1".to_owned())
        .expect("first session lease");
    let second = GenerationSessionPermit::new(registry.clone(), "agent-1".to_owned())
        .expect("second session lease");

    {
        let locks = registry.lock().expect("session registry lock");
        let entry = locks.get("agent-1").expect("shared session entry");
        assert_eq!(entry.users.load(Ordering::Acquire), 2);
    }

    drop(first);
    {
        let locks = registry.lock().expect("session registry lock");
        let entry = locks.get("agent-1").expect("live session entry");
        assert_eq!(entry.users.load(Ordering::Acquire), 1);
    }

    drop(second);
    assert!(registry.lock().expect("session registry lock").is_empty());

    let replacement = GenerationSessionPermit::new(registry.clone(), "agent-1".to_owned())
        .expect("replacement session lease");
    assert_eq!(registry.lock().expect("session registry lock").len(), 1);
    drop(replacement);
    assert!(registry.lock().expect("session registry lock").is_empty());
}

#[tokio::test]
async fn same_trusted_session_serializes_without_consuming_global_queue_capacity() {
    let controller = admission_controller(1, 1);
    let session_key = trusted_session_key("agent-1");
    let first_cancellation = openai_frontend::CancellationToken::new();
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let second_controller = controller.clone();
    let second_cancellation = openai_frontend::CancellationToken::new();
    let waiter_cancellation = second_cancellation.clone();
    let waiter = tokio::spawn(async move {
        second_controller
            .acquire(
                &trusted_ids("agent-1"),
                &waiter_cancellation,
                Duration::from_secs(1),
            )
            .await
    });

    tokio::time::timeout(Duration::from_millis(100), async {
        loop {
            let users = controller
                .generation_session_locks
                .lock()
                .expect("session registry lock")
                .get(&session_key)
                .map_or(0, |entry| entry.users.load(Ordering::Acquire));
            if users == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("second turn registered its session wait");

    assert!(!waiter.is_finished());
    assert_eq!(
        controller.generation_queue_depth.load(Ordering::Acquire),
        0,
        "session contention must not consume global queue capacity"
    );

    second_cancellation.cancel();
    let error = result_error(
        tokio::time::timeout(Duration::from_millis(100), waiter)
            .await
            .expect("cancelled session waiter returned")
            .expect("session waiter task completed"),
    );
    assert!(error.body().error.message.contains("request cancelled"));
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    drop(first);
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .is_empty()
    );
}

#[tokio::test]
async fn same_trusted_session_acquires_only_after_the_first_turn_releases() {
    let controller = admission_controller(1, 1);
    let first_cancellation = openai_frontend::CancellationToken::new();
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let second_controller = controller.clone();
    let second = tokio::spawn(async move {
        second_controller
            .acquire(
                &trusted_ids("agent-1"),
                &openai_frontend::CancellationToken::new(),
                Duration::from_secs(1),
            )
            .await
    });

    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(!second.is_finished());
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    drop(first);
    let second = tokio::time::timeout(Duration::from_millis(100), second)
        .await
        .expect("second turn acquired after first released")
        .expect("second turn task completed")
        .expect("second session admission");
    assert_eq!(controller.generation_limit.available_permits(), 0);
    drop(second);
    assert_eq!(controller.generation_limit.available_permits(), 1);
}

#[tokio::test]
async fn session_and_global_admission_share_one_absolute_deadline() {
    let controller = admission_controller(1, 1);
    let first_cancellation = openai_frontend::CancellationToken::new();
    let (global_permit, session_permit) = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");
    let session_permit = session_permit.expect("trusted session permit");
    let release_session = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(140)).await;
        drop(session_permit);
    });
    let started = Instant::now();

    let error = result_error(
        controller
            .acquire(
                &trusted_ids("agent-1"),
                &openai_frontend::CancellationToken::new(),
                Duration::from_millis(200),
            )
            .await,
    );

    assert!(error.body().error.message.contains("timed out waiting"));
    assert!(
        started.elapsed() < Duration::from_millis(300),
        "global-lane waiting must not restart the request admission timeout"
    );
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    release_session
        .await
        .expect("session release task completed");
    drop(global_permit);
}

#[tokio::test]
async fn unrelated_session_is_not_starved_by_a_same_session_waiter() {
    let controller = admission_controller(2, 2);
    let session_key = trusted_session_key("agent-1");
    let first_cancellation = openai_frontend::CancellationToken::new();
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let duplicate_controller = controller.clone();
    let duplicate_cancellation = openai_frontend::CancellationToken::new();
    let waiter_cancellation = duplicate_cancellation.clone();
    let duplicate = tokio::spawn(async move {
        duplicate_controller
            .acquire(
                &trusted_ids("agent-1"),
                &waiter_cancellation,
                Duration::from_secs(1),
            )
            .await
    });

    tokio::time::timeout(Duration::from_millis(100), async {
        loop {
            let users = controller
                .generation_session_locks
                .lock()
                .expect("session registry lock")
                .get(&session_key)
                .map_or(0, |entry| entry.users.load(Ordering::Acquire));
            if users == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("duplicate turn registered its session wait");

    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    let unrelated = controller
        .acquire(
            &trusted_ids("agent-2"),
            &openai_frontend::CancellationToken::new(),
            Duration::from_millis(100),
        )
        .await
        .expect("unrelated session used the free global lane");
    assert_eq!(controller.generation_limit.available_permits(), 0);
    assert!(!duplicate.is_finished());

    duplicate_cancellation.cancel();
    let duplicate_error = result_error(
        tokio::time::timeout(Duration::from_millis(100), duplicate)
            .await
            .expect("duplicate waiter cancelled")
            .expect("duplicate waiter task completed"),
    );
    assert!(
        duplicate_error
            .body()
            .error
            .message
            .contains("request cancelled")
    );
    drop((first, unrelated));
    assert_eq!(controller.generation_limit.available_permits(), 2);
}

#[tokio::test]
async fn same_session_waiter_does_not_reserve_the_only_global_queue_slot() {
    let controller = admission_controller(1, 1);
    let session_key = trusted_session_key("agent-1");
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &openai_frontend::CancellationToken::new(),
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let duplicate_controller = controller.clone();
    let duplicate_cancellation = openai_frontend::CancellationToken::new();
    let waiter_cancellation = duplicate_cancellation.clone();
    let duplicate = tokio::spawn(async move {
        duplicate_controller
            .acquire(
                &trusted_ids("agent-1"),
                &waiter_cancellation,
                Duration::from_secs(1),
            )
            .await
    });
    tokio::time::timeout(Duration::from_millis(100), async {
        loop {
            let users = controller
                .generation_session_locks
                .lock()
                .expect("session registry lock")
                .get(&session_key)
                .map_or(0, |entry| entry.users.load(Ordering::Acquire));
            if users == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("duplicate turn registered its session wait");
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    let unrelated_controller = controller.clone();
    let unrelated = tokio::spawn(async move {
        unrelated_controller
            .acquire(
                &trusted_ids("agent-2"),
                &openai_frontend::CancellationToken::new(),
                Duration::from_secs(1),
            )
            .await
    });
    tokio::time::timeout(Duration::from_millis(100), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("unrelated turn reserved the only global queue slot");
    assert!(!unrelated.is_finished());

    duplicate_cancellation.cancel();
    let duplicate_error = result_error(
        tokio::time::timeout(Duration::from_millis(100), duplicate)
            .await
            .expect("duplicate waiter cancelled")
            .expect("duplicate waiter task completed"),
    );
    assert_eq!(duplicate_error.status().as_u16(), 499);

    drop(first);
    let unrelated = tokio::time::timeout(Duration::from_millis(100), unrelated)
        .await
        .expect("unrelated turn acquired the released lane")
        .expect("unrelated waiter task completed")
        .expect("unrelated session admission");
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    drop(unrelated);
}

#[tokio::test]
async fn different_trusted_sessions_can_hold_generation_lanes_concurrently() {
    let controller = admission_controller(2, 2);
    let first_cancellation = openai_frontend::CancellationToken::new();
    let second_cancellation = openai_frontend::CancellationToken::new();
    let first_ids = trusted_ids("agent-1");
    let second_ids = trusted_ids("agent-2");

    let (first, second) = tokio::join!(
        controller.acquire(&first_ids, &first_cancellation, Duration::from_secs(1)),
        controller.acquire(&second_ids, &second_cancellation, Duration::from_secs(1)),
    );
    let first = first.expect("first session admission");
    let second = second.expect("second session admission");

    assert_eq!(controller.generation_limit.available_permits(), 0);
    assert_eq!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .len(),
        2
    );

    drop((first, second));
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .is_empty()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn blocking_worker_holds_global_and_session_permits_until_work_finishes() {
    let controller = admission_controller(1, 1);
    let session_key = trusted_session_key("agent-1");
    let cancellation = openai_frontend::CancellationToken::new();
    let (global_permit, session_permit) = controller
        .acquire(
            &trusted_ids("agent-1"),
            &cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("worker admission");
    let worker_context = OpenAiRequestContext::new();
    let worker_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let release_worker = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let started = worker_started.clone();
    let release = release_worker.clone();

    let worker = tokio::spawn(run_blocking_generation_worker(
        global_permit,
        worker_context,
        move |_| {
            let _session_permit = session_permit;
            started.store(true, Ordering::Release);
            while !release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        },
    ));

    tokio::time::timeout(Duration::from_millis(100), async {
        while !worker_started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("blocking generation worker started");

    assert_eq!(controller.generation_limit.available_permits(), 0);
    let entry = controller
        .generation_session_locks
        .lock()
        .expect("session registry lock")
        .get(&session_key)
        .expect("worker retains session entry")
        .semaphore
        .clone();
    assert_eq!(entry.available_permits(), 0);

    release_worker.store(true, Ordering::Release);
    tokio::time::timeout(Duration::from_millis(100), worker)
        .await
        .expect("blocking generation worker completed")
        .expect("worker task completed")
        .expect("blocking worker joined");
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .is_empty()
    );
}

#[test]
fn untrusted_conversation_affinity_bypasses_session_registry() {
    let registry = Arc::new(Mutex::new(BTreeMap::new()));
    let untrusted = OpenAiGenerationIds::new_with_trust(
        OpenAiCacheHints::default(),
        Some("conversation-7"),
        false,
        None,
    );
    assert!(trusted_generation_session_key(&untrusted).is_none());
    assert!(registry.lock().expect("session registry lock").is_empty());

    let trusted = trusted_ids("agent-7");
    let key = trusted_generation_session_key(&trusted).expect("trusted session key");
    assert_eq!(key, trusted.session_id_string());
    let _permit =
        GenerationSessionPermit::new(registry.clone(), key).expect("trusted session lease");
    assert_eq!(registry.lock().expect("session registry lock").len(), 1);
}

#[test]
fn direct_backend_calls_ignore_spoofed_request_trust_metadata() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "capture-model",
        "messages": [{"role": "user", "content": "hello"}],
        "mesh_internal_agent_session_id": "spoofed-session",
        "mesh_internal_agent_session_source": "x-litellm-session-id",
        "mesh_internal_agent_session_trusted": true
    }))
    .expect("request with spoofed metadata");
    let context = OpenAiRequestContext::new();
    let ids = generation_ids(
        OpenAiCacheHints::from_chat_request(&request),
        request.agent_session(),
        &context,
    );

    assert_eq!(ids.agent_session_id.as_deref(), Some("spoofed-session"));
    assert!(!ids.agent_session_trusted);
    assert!(trusted_generation_session_key(&ids).is_none());
}

#[test]
fn internal_stream_usage_observation_preserves_client_wire_preference() {
    let direct = OpenAiRequestContext::new();
    assert!(!should_emit_stream_usage(false, &direct));
    assert!(should_emit_stream_usage(true, &direct));

    let observed = OpenAiRequestContext::new().with_stream_usage_observation();
    assert!(should_emit_stream_usage(false, &observed));
}

/// Reproduces the orphaned-generation report: a client can vanish (dropped
/// connection, or one that hasn't been noticed yet -- e.g. behind a proxy
/// that doesn't propagate the close) leaving the SSE receiver alive but
/// permanently undrained. `StreamEventSender::send` must not let that pin
/// the generation worker, and the execution lane it holds, forever: once the
/// request is cancelled it must give up promptly even though the channel
/// stays full and the receiver is never dropped.
///
/// This runs the send on its own thread and waits for a result over a
/// bounded `recv_timeout` rather than joining directly, so a regression back
/// to an unconditional blocking send fails this test instead of hanging the
/// suite. It uses the real `STREAM_SEND_STALL_TIMEOUT`, so cancellation --
/// not the stall timeout -- must be what ends the wait.
#[test]
fn stalled_receiver_does_not_pin_the_generation_worker_forever() {
    let (tx, rx) = mpsc::channel(1);
    tx.try_send(Ok(GenerationStreamEvent::Delta("first".to_owned())))
        .expect("channel has room for the first event");
    let context = OpenAiRequestContext::new();
    let rt = Runtime::new().expect("tokio runtime for stall test");
    let sender = StreamEventSender::new(
        tx,
        rt.handle().clone(),
        STREAM_SEND_STALL_TIMEOUT,
        "test-request".to_owned(),
        test_telemetry(),
    );

    let sender_context = context.clone();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = sender.send(
            Ok(GenerationStreamEvent::Delta("second".to_owned())),
            &sender_context,
        );
        // Keep `rx` alive without draining it until after the send settles,
        // so a fix that works only because the channel closed doesn't pass.
        drop(rx);
        let _ = done_tx.send(result.is_err());
    });

    // Give the sender thread a chance to observe the full channel before
    // cancelling -- simulating cancellation arriving (e.g. from a
    // connection-drop observer) after the worker is already stuck sending.
    std::thread::sleep(Duration::from_millis(50));
    context.cancel();

    let cancelled = done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("a stalled send must be interrupted by cancellation, not block forever");
    assert!(cancelled, "cancelled send must return an error");
}

/// Covers the case the report actually flagged as unproven: nothing ever
/// calls `cancel()` -- the connection-drop observer (`CancelOnDropSseStream`)
/// simply never fires, e.g. because the client vanished behind a proxy that
/// kept the socket to mesh-llm open. A stalled, never-dropped, never-drained
/// receiver must still cause the send to give up and self-cancel once it has
/// been full for the (here, injected and short) stall timeout, so the lane
/// isn't held indefinitely.
#[test]
fn stalled_receiver_self_cancels_after_the_stall_timeout_with_no_external_cancel() {
    let (tx, rx) = mpsc::channel(1);
    tx.try_send(Ok(GenerationStreamEvent::Delta("first".to_owned())))
        .expect("channel has room for the first event");
    let context = OpenAiRequestContext::new();
    let rt = Runtime::new().expect("tokio runtime for stall test");
    let sender = StreamEventSender::new(
        tx,
        rt.handle().clone(),
        Duration::from_millis(50),
        "test-request".to_owned(),
        test_telemetry(),
    );

    let result = sender.send(
        Ok(GenerationStreamEvent::Delta("second".to_owned())),
        &context,
    );

    assert!(
        result.is_err(),
        "a send stalled past the timeout must fail rather than hang"
    );
    assert!(
        context.is_cancelled(),
        "a self-detected stall must cancel the request so the lane is freed"
    );
    drop(rx);
}

/// Red->green for the swallowed-terminal-frame defect: on the pre-fix code,
/// the `run_generation_stream` cancellation branch checked
/// `context.is_cancelled()` before sending, so an already-cancelled request
/// caused the cancellation error frame -- and, by the same shape, the
/// `parser.finish` error frame and the outer generation error frame -- to be
/// silently dropped instead of enqueued. That flips
/// `stream_lifecycle`'s terminal classification: without the `Err` frame,
/// `drop_outcome()` falls through to `StreamDropOutcome::Cancelled` instead
/// of the `BackendError`/`StreamTerminal` path `lifecycle.failed(error)`
/// drives. `send_terminal` must deliver the frame to a receiver that is
/// merely cancelled but still alive and draining, while `send` (used only
/// for in-flight events) must still refuse to send once cancelled.
#[test]
fn terminal_frames_are_delivered_after_the_request_is_cancelled() {
    let (tx, mut rx) = mpsc::channel(4);
    let context = OpenAiRequestContext::new();
    context.cancel();
    let rt = Runtime::new().expect("tokio runtime for terminal-delivery test");
    let sender = StreamEventSender::new(
        tx,
        rt.handle().clone(),
        STREAM_SEND_STALL_TIMEOUT,
        "test-request".to_owned(),
        test_telemetry(),
    );

    sender
        .send_terminal(Ok(GenerationStreamEvent::Done(FinishReason::Stop)))
        .expect("terminal frames must still reach a live, cancelled-but-draining receiver");

    let received = rx
        .try_recv()
        .expect("the terminal frame must be enqueued, not silently swallowed");
    assert!(matches!(
        received,
        Ok(GenerationStreamEvent::Done(FinishReason::Stop))
    ));

    let send_result = sender.send(
        Ok(GenerationStreamEvent::Delta("late".to_owned())),
        &context,
    );
    assert!(
        send_result.is_err(),
        "the cancellation check is bypassed only for terminal frames, not in-flight ones"
    );
}

/// Once streaming has committed HTTP 200, a native generation failure cannot
/// be converted into a new HTTP status. It must remain an `Err` item on the
/// backend stream so `openai-frontend` can frame an explicit SSE error event
/// before `[DONE]` instead of making the response look like a zero-token
/// success.
#[test]
fn backend_errors_are_delivered_as_terminal_stream_frames() {
    let (tx, mut rx) = mpsc::channel(4);
    let rt = Runtime::new().expect("tokio runtime for terminal-error test");
    let sender = StreamEventSender::new(
        tx,
        rt.handle().clone(),
        STREAM_SEND_STALL_TIMEOUT,
        "test-request".to_owned(),
        test_telemetry(),
    );

    sender
        .send_terminal(Err(OpenAiError::backend(
            "native decode failed to find a memory slot",
        )))
        .expect("backend failure must reach a live stream receiver");

    let error = match rx.try_recv().expect("backend failure must be enqueued") {
        Ok(_) => panic!("terminal item must remain an error"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("native decode failed to find a memory slot")
    );
}

/// Bounds the double-wait hazard: once an in-flight send has already proven
/// the receiver unreachable (stalled past the timeout, here injected short),
/// a subsequent terminal send must not wait out the same stall timeout a
/// second time -- that would double the execution lane's hold to
/// `2 * stall_timeout` and defeat the point of freeing it promptly.
#[test]
fn terminal_frames_are_dropped_once_the_receiver_is_proven_unreachable() {
    let (tx, rx) = mpsc::channel(1);
    tx.try_send(Ok(GenerationStreamEvent::Delta("first".to_owned())))
        .expect("channel has room for the first event");
    let context = OpenAiRequestContext::new();
    let rt = Runtime::new().expect("tokio runtime for double-wait test");
    // Inject a generous stall timeout so the short-circuit assertion has a wide
    // margin on a loaded CI runner: a terminal send that (wrongly) waited out
    // the stall again would take at least `stall_timeout`, while the correct
    // short-circuit is one atomic load. Deriving the bound from the timeout
    // instead of a fixed wall-clock number keeps the two coupled.
    let stall_timeout = Duration::from_millis(500);
    let sender = StreamEventSender::new(
        tx,
        rt.handle().clone(),
        stall_timeout,
        "test-request".to_owned(),
        test_telemetry(),
    );

    let stalled = sender.send(
        Ok(GenerationStreamEvent::Delta("second".to_owned())),
        &context,
    );
    assert!(
        stalled.is_err(),
        "the in-flight send must self-cancel once the receiver proves unreachable"
    );

    let started = Instant::now();
    let terminal = sender.send_terminal(Ok(GenerationStreamEvent::Done(FinishReason::Stop)));
    let elapsed = started.elapsed();

    assert!(
        terminal.is_err(),
        "a proven-unreachable receiver must not be handed a terminal frame either"
    );
    // The short-circuit must complete in a small fraction of the injected
    // stall timeout; a second wait would consume at least the whole timeout.
    assert!(
        elapsed < stall_timeout / 5,
        "terminal send must short-circuit instead of waiting out the stall timeout again, took {elapsed:?} (timeout {stall_timeout:?})"
    );
    drop(rx);
}

// --- Terminal-hook lifecycle wiring (mesh1437 production wiring) ---
//
// `chat_completion_with_hooks`/`chat_completion_stream_with_hooks` are unit
// tested directly with a fake `dispatch` closure rather than through the
// full `chat_completion_with_context`/`chat_completion_stream` trait methods:
// real generation needs a loaded GGUF (see `recurrent_test_backend` in
// `local_generation/tests.rs`, gated on `SKIPPY_RECURRENT_CACHE_TEST_MODEL`),
// but the hook lifecycle itself never touches `self.runtime` — it only reads
// `self.hook_policy` — so it's fully exercisable on a modelless backend.

fn hooks_test_backend(hook_policy: Option<Arc<dyn OpenAiHookPolicy>>) -> StageOpenAiBackend {
    let config: skippy_protocol::StageConfig = serde_json::from_value(json!({
        "run_id": "hooks-test",
        "topology_id": "hooks-test",
        "model_id": "hooks-test-model",
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 1,
        "load_mode": "runtime-slice",
        "bind_addr": "127.0.0.1:0",
    }))
    .expect("minimal stage config for hook lifecycle tests");
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let telemetry = crate::telemetry::Telemetry::new(
        None,
        1,
        config.clone(),
        crate::telemetry::TelemetryLevel::Off,
    );
    let iteration_scheduler =
        IterationScheduler::new(runtime.clone(), &config, 1, true, telemetry.clone())
            .expect("iteration scheduler for hook lifecycle tests");
    StageOpenAiBackend {
        runtime: runtime.clone(),
        config: config.clone(),
        telemetry,
        model_id: "hooks-test-model".to_string(),
        default_max_tokens: 16,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size: 128,
        mode: OpenAiBackendMode::LocalRuntime,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_max: 0,
        speculative: SpeculativeDecodeConfig::default(),
        generation_limit: Arc::new(GenerationConcurrencyController::fixed(1)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: 1,
        generation_admission_timeout: Duration::from_secs(10),
        generation_service_estimator: Arc::new(GenerationServiceEstimator::new(1)),
        generation_session_locks: Arc::new(Mutex::new(BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(128)),
        hook_policy,
        generation_receipt: None,
        generation_lifecycle: None,
        linear_proposal_ingress: None,
        kv: None,
        iteration_scheduler,
    }
}

fn mesh_hooks_request(model: &str) -> ChatCompletionRequest {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("minimal chat completion request");
    set_chat_mesh_hooks_enabled(&mut request, true);
    request
}

#[derive(Debug, Clone, PartialEq)]
enum HookTerminalRecord {
    Success { model: String },
    Error { status: u16, message: String },
    Denied { status: u16, reason: String },
    Cancelled,
    StreamCompleted,
}

#[derive(Default)]
struct RecordingHookPolicy {
    deny: bool,
    hang_before_dispatch: bool,
    terminals: Mutex<Vec<HookTerminalRecord>>,
}

#[async_trait]
impl OpenAiHookPolicy for RecordingHookPolicy {
    async fn before_chat_completion(
        &self,
        _request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<ChatHookOutcome> {
        if self.hang_before_dispatch {
            std::future::pending::<()>().await;
        }
        if self.deny {
            return Err(OpenAiError::invalid_request("denied by policy"));
        }
        Ok(ChatHookOutcome::none())
    }

    async fn on_chat_completion_terminal(
        &self,
        _request: &ChatCompletionRequest,
        _exchange_id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        let record = match outcome {
            ChatCompletionOutcome::Success { response } => HookTerminalRecord::Success {
                model: response.model.clone(),
            },
            ChatCompletionOutcome::Error { status, message } => HookTerminalRecord::Error {
                status: *status,
                message: (*message).to_string(),
            },
            ChatCompletionOutcome::Denied { status, reason } => HookTerminalRecord::Denied {
                status: *status,
                reason: (*reason).to_string(),
            },
            ChatCompletionOutcome::Cancelled => HookTerminalRecord::Cancelled,
            ChatCompletionOutcome::StreamCompleted => HookTerminalRecord::StreamCompleted,
            other => {
                unreachable!("unhandled ChatCompletionOutcome variant in test fixture: {other:?}")
            }
        };
        self.terminals.lock().unwrap().push(record);
    }
}

/// Terminal delivery for a dropped/streamed exchange fires from a detached
/// spawned task (see `TerminalGuard::drop`/`fire_detached`), so it lands
/// sometime after the driving future is aborted or the stream stops
/// yielding items, not synchronously at that instant.
async fn wait_for_hook_terminal(policy: &RecordingHookPolicy) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
    loop {
        if !policy.terminals.lock().unwrap().is_empty() {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "terminal event never fired"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn stage_backend_success_fires_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let response = backend
        .chat_completion_with_hooks(request, |request| async move {
            Ok(ChatCompletionResponse::new(
                request.model,
                "ok",
                Usage::new(1, 1),
            ))
        })
        .await
        .expect("fake dispatch succeeds");
    assert_eq!(response.model, "hooks-test-model");

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(
        terminals.as_slice(),
        [HookTerminalRecord::Success {
            model: "hooks-test-model".to_string()
        }]
    );
}

#[tokio::test]
async fn stage_backend_dispatch_error_fires_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let error = backend
        .chat_completion_with_hooks(request, |_request| async move {
            Err(OpenAiError::backend("upstream exploded"))
        })
        .await
        .expect_err("fake dispatch fails");
    assert_eq!(error.status().as_u16(), 502);

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        HookTerminalRecord::Error { status: 502, message }
            if message.contains("upstream exploded")
    ));
}

#[tokio::test]
async fn stage_backend_denied_request_never_dispatches_and_fires_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy {
        deny: true,
        ..RecordingHookPolicy::default()
    });
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");
    let dispatched = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let dispatched_flag = dispatched.clone();

    let error = backend
        .chat_completion_with_hooks(request, move |request| {
            dispatched_flag.store(true, Ordering::Release);
            async move {
                Ok(ChatCompletionResponse::new(
                    request.model,
                    "ok",
                    Usage::new(0, 0),
                ))
            }
        })
        .await
        .expect_err("policy denies the request");
    assert_eq!(error.status().as_u16(), 400);
    assert!(
        !dispatched.load(Ordering::Acquire),
        "a denied request must never reach dispatch"
    );

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        HookTerminalRecord::Denied { status: 400, reason }
            if reason.contains("denied by policy")
    ));
}

#[tokio::test]
async fn stage_backend_dropped_during_admission_hook_fires_exactly_one_cancelled_terminal() {
    let policy = Arc::new(RecordingHookPolicy {
        hang_before_dispatch: true,
        ..RecordingHookPolicy::default()
    });
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let handle = tokio::spawn(async move {
        backend
            .chat_completion_with_hooks(request, |request| async move {
                Ok(ChatCompletionResponse::new(
                    request.model,
                    "ok",
                    Usage::new(0, 0),
                ))
            })
            .await
    });

    // Let the task run until it's parked in `before_chat_completion`, then
    // cancel it the way an outer timeout or client disconnect would.
    tokio::task::yield_now().await;
    handle.abort();
    let _ = handle.await;
    wait_for_hook_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [HookTerminalRecord::Cancelled]);
}

#[tokio::test]
async fn stage_backend_stream_that_ends_normally_fires_stream_completed_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let mut stream = backend
        .chat_completion_stream_with_hooks(request, |request| async move {
            Ok(Box::pin(futures_util::stream::iter(vec![
                Ok(ChatCompletionChunk::delta(request.model.clone(), "hi")),
                Ok(ChatCompletionChunk::done(request.model)),
            ])) as ChatCompletionStream)
        })
        .await
        .expect("stream created");
    while stream
        .next()
        .await
        .transpose()
        .expect("no chunk errors")
        .is_some()
    {}
    wait_for_hook_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [HookTerminalRecord::StreamCompleted]);
}

/// The explicit case this wiring exists for: a client disconnects (or an
/// outer timeout fires) after a stream has already delivered a chunk but
/// before it ends on its own. Without `TerminalGuardedChatStream` wired into
/// `StageOpenAiBackend`, this exchange got zero terminal events.
#[tokio::test]
async fn stage_backend_stream_dropped_mid_stream_fires_exactly_one_cancelled_terminal() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let mut stream = backend
        .chat_completion_stream_with_hooks(request, |request| async move {
            let first = ChatCompletionChunk::delta(request.model, "partial");
            Ok(Box::pin(
                futures_util::stream::once(async move { Ok(first) })
                    .chain(futures_util::stream::pending()),
            ) as ChatCompletionStream)
        })
        .await
        .expect("stream created");
    let first = stream.next().await;
    assert!(matches!(first, Some(Ok(_))), "first chunk should flow");
    drop(stream);
    wait_for_hook_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [HookTerminalRecord::Cancelled]);
}

#[tokio::test]
async fn stage_backend_stream_error_chunk_fires_error_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let mut stream = backend
        .chat_completion_stream_with_hooks(request, |request| async move {
            Ok(Box::pin(futures_util::stream::iter(vec![
                Ok(ChatCompletionChunk::delta(request.model, "hi")),
                Err(OpenAiError::backend("upstream exploded")),
            ])) as ChatCompletionStream)
        })
        .await
        .expect("stream created");
    while let Some(item) = stream.next().await {
        let _ = item;
    }
    wait_for_hook_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        HookTerminalRecord::Error { status: 502, message }
            if message.contains("upstream exploded")
    ));
}

#[tokio::test]
async fn stage_backend_stream_denied_never_dispatches_and_fires_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy {
        deny: true,
        ..RecordingHookPolicy::default()
    });
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let error = match backend
        .chat_completion_stream_with_hooks(request, |_request| async move {
            panic!("a denied request must never reach dispatch")
        })
        .await
    {
        Ok(_) => panic!("policy denies the request"),
        Err(error) => error,
    };
    assert_eq!(error.status().as_u16(), 400);

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        HookTerminalRecord::Denied { status: 400, reason }
            if reason.contains("denied by policy")
    ));
}

#[test]
fn generation_ids_carries_the_frontend_request_id_byte_equal_to_the_context() {
    let request_id = openai_frontend::parse_request_id("c0a801ef-2a39-4f52-99f5-bdc849127cde")
        .expect("test UUID should parse");
    let context = OpenAiRequestContext::with_request_id(request_id);
    let ids = generation_ids(OpenAiCacheHints::default(), None, &context);
    assert_eq!(
        ids.frontend_request_id,
        Some(request_id.as_uuid().into_bytes())
    );
}

#[test]
fn generation_ids_leaves_frontend_request_id_absent_for_a_non_frontend_context() {
    let context = OpenAiRequestContext::new();
    let ids = generation_ids(OpenAiCacheHints::default(), None, &context);
    assert_eq!(ids.frontend_request_id, None);
}
