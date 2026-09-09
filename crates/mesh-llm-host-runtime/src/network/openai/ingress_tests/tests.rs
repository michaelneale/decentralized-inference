use std::sync::{Arc, Mutex};

use crate::logging::{
    LoggingService, OpenAiLifecycleAttachment, RawMeshLifecycleOwners, RawMeshRequestLifecycle,
    TerminalOutcome,
};
use crate::plugin::openai_exchange::{OpenAiExchangeChannel, OpenAiExchangeEnvelope};
use async_trait::async_trait;
use mesh_llm_events::logging::{events::LifecycleEvent, identifiers::RequestId};

use super::*;

/// Recording double for `OpenAiExchangeChannel` — accumulates every published
/// envelope so tests can assert on count, dispatch path, exchange id, and nonce.
#[derive(Default)]
struct RecordingChannel {
    events: Mutex<Vec<OpenAiExchangeEnvelope>>,
}

#[async_trait]
impl OpenAiExchangeChannel for RecordingChannel {
    async fn publish(&self, event: &OpenAiExchangeEnvelope) {
        self.events.lock().unwrap().push(event.clone());
    }
}

fn large_tokenize_request(model: &str) -> proxy::BufferedHttpRequest {
    proxy::BufferedHttpRequest {
        raw: b"unchanged tokenizer wire".to_vec(),
        method: "POST".to_owned(),
        path: "/v1/tokenize".to_owned(),
        client_path: "/v1/tokenize".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: 140_000,
        completion_tokens: None,
        stream: None,
        model_name: Some(model.to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::None,
        correlation_id: None,
    }
}

fn recorded_lifecycle_events(service: &LoggingService) -> Vec<LifecycleEvent> {
    service
        .bus_ref()
        .replay_window()
        .records
        .into_iter()
        .filter_map(|record| {
            let envelope = serde_json::from_str::<serde_json::Value>(&record.entry.payload).ok()?;
            let payload = envelope.get("payload")?.as_str()?;
            serde_json::from_str(payload).ok()
        })
        .collect()
}

fn plugin_lifecycle() -> (Arc<LoggingService>, OpenAiLifecycleAttachment) {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let parent = RawMeshRequestLifecycle::register(
        Arc::clone(&service),
        Arc::new(RawMeshLifecycleOwners::default()),
        RequestId::new(),
    )
    .expect("plugin test should claim one parent");
    (service, OpenAiLifecycleAttachment::new(Some(parent)))
}

fn record_plugin_attempt(
    observer: OpenAiRouteObserver<'_>,
    model: &str,
    provider: &str,
    engine: &str,
    result: super::super::response::RouteAttemptResult,
) -> super::super::response::RouteAttemptResult {
    observer.route_selected_with_metadata(Some(model), Some(provider), Some(engine));
    let attempt_id = observer.start_attempt();
    match result {
        super::super::response::RouteAttemptResult::Delivered { status_code, .. } => {
            observer.complete_attempt(attempt_id, status_code);
        }
        _ => observer.fail_attempt(
            attempt_id,
            super::super::response::route_attempt_result_label(&result),
        ),
    }
    result
}

fn assert_payload_free(events: &[LifecycleEvent]) {
    let serialized = serde_json::to_string(events).expect("events should serialize");
    for forbidden in [
        "body",
        "headers",
        "prompt",
        "authorization",
        "secret",
        "completion",
    ] {
        assert!(!serialized.to_ascii_lowercase().contains(forbidden));
    }
}

/// A model nobody serves must not stay in the auto pool. Before this,
/// the readiness check fell through to `true`, so `auto` could select a
/// phantom (stale gossip, or a peer that unloaded) and 404 the caller on
/// a model they never named.
#[tokio::test]
async fn phantom_model_is_not_auto_route_eligible() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    let targets = election::ModelTargets::default();
    let affinity = affinity::AffinityRouter::new();

    let eligible = auto_route_model_has_ready_ingress_target(
        &node,
        &targets,
        "phantom/model:Q4_K_M",
        None,
        &affinity,
    )
    .await;

    assert!(
        !eligible,
        "a model with no local target and no remote host must not be auto-route eligible"
    );
}

/// A freshly started serve node records its model before the target table
/// and gossip catch up. Failing closed must not exclude this node's own
/// model during that window.
#[tokio::test]
async fn freshly_served_local_model_is_auto_route_eligible() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_hosted_models(vec!["local/fresh-model:Q4_K_M".to_string()])
        .await;
    let targets = election::ModelTargets::default();
    let affinity = affinity::AffinityRouter::new();

    let eligible = auto_route_model_has_ready_ingress_target(
        &node,
        &targets,
        "local/fresh-model:Q4_K_M",
        None,
        &affinity,
    )
    .await;

    assert!(
        eligible,
        "a locally served model must stay eligible before targets populate"
    );
}

#[test]
fn parse_model_with_profile_with_named_profile() {
    let (model_ref, profile) = parse_model_with_profile("Qwen3-8B#low-ctx");
    assert_eq!(model_ref, "Qwen3-8B");
    assert_eq!(profile, "low-ctx");
}

#[test]
fn parse_model_with_profile_without_profile() {
    let (model_ref, profile) = parse_model_with_profile("Qwen3-8B");
    assert_eq!(model_ref, "Qwen3-8B");
    assert_eq!(profile, "");
}

#[test]
fn parse_model_with_profile_empty_profile_after_hash() {
    let (model_ref, profile) = parse_model_with_profile("Qwen3-8B#");
    assert_eq!(model_ref, "Qwen3-8B");
    assert_eq!(profile, "");
}

#[test]
fn parse_model_with_profile_huggingface_ref_with_quant() {
    let (model_ref, profile) = parse_model_with_profile("org/repo:Q4_K_M#profile");
    assert_eq!(model_ref, "org/repo:Q4_K_M");
    assert_eq!(profile, "profile");
}

#[test]
fn parse_model_with_profile_multiple_hashes_uses_last() {
    let (model_ref, profile) = parse_model_with_profile("model#with#hash#profile");
    assert_eq!(model_ref, "model#with#hash");
    assert_eq!(profile, "profile");
}

/// Regression: `model=mesh` stays in the Mesh gateway with one admitted worker.
///
/// Prompt heuristics may activate deterministic rescue inside the gateway, but
/// they must never decide whether the virtual Mesh model enters the gateway.
#[tokio::test]
async fn moa_single_worker_stays_in_gateway() {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_hosted_models(vec!["local/only-model:Q4_K_M".to_string()])
        .await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "local/only-model:Q4_K_M".to_string(),
        vec![election::InferenceTarget::Local(1)],
    );
    let affinity = affinity::AffinityRouter::new();

    // The helper owns the connected stream while the gateway runs. With the
    // fake worker endpoint unreachable, the turn may fail, but it must be
    // handled by the gateway rather than rewritten into direct routing.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let client = tokio::net::TcpStream::connect(addr);
    let server = async { listener.accept().await.map(|(stream, _)| stream) };
    let (_client_side, server_side) = tokio::join!(client, server);
    let tcp_stream = server_side.expect("accept");

    let body = br#"{"model":"mesh","messages":[{"role":"user","content":"hi"}]}"#;
    let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes()
        .into_iter()
        .chain(body.iter().copied())
        .collect::<Vec<u8>>();
    let mut request = proxy::BufferedHttpRequest {
        raw,
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: body.len(),
        completion_tokens: None,
        stream: None,
        model_name: Some("mesh".to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::OpenAiChatCompletionsJson,
        correlation_id: None,
    };
    let decision = AutoRouteDecision {
        effective_model: Some("mesh".to_owned()),
        classification: None,
        required_tokens: None,
    };
    let ctx = ProxyConnectionContext {
        route: IngressRouteContext {
            node: &node,
            targets: &targets,
            affinity: &affinity,
            plugin_manager: None,
            exchange_channel: None,
        },
    };
    let lifecycle = OpenAiLifecycleAttachment::unowned();

    let result = try_handle_moa_intercept(
        tcp_stream.into(),
        &mut request,
        &ctx,
        &decision,
        lifecycle.route_observer(),
    )
    .await;

    match result {
        MoaInterceptResult::Handled(_) => {
            assert_eq!(
                request.model_name.as_deref(),
                Some("mesh"),
                "the gateway must preserve the virtual routing model"
            );
        }
        MoaInterceptResult::NotMoa(_) => {
            panic!("model=mesh fell through without entering the Mesh gateway");
        }
        MoaInterceptResult::Degraded { model, .. } => {
            panic!("one-worker model=mesh degraded to direct routing as {model:?}");
        }
    }
}

#[test]
fn moa_degraded_model_is_consumed_by_pipeline_dispatch() {
    use crate::network::router::{Category, Classification, Complexity};

    let request = proxy::BufferedHttpRequest {
        raw: Vec::new(),
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: 0,
        completion_tokens: None,
        stream: None,
        model_name: Some("local/only-model:Q4_K_M".to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::None,
        correlation_id: None,
    };
    let decision = AutoRouteDecision {
        effective_model: Some("mesh".to_owned()),
        classification: Some(Classification {
            category: Category::Code,
            complexity: Complexity::Deep,
            needs_tools: true,
            has_media_inputs: false,
        }),
        required_tokens: None,
    };

    assert_eq!(
        pipeline_route_model(&request, &decision, request.model_name.as_deref(),),
        Some("local/only-model:Q4_K_M"),
        "pipeline dispatch must consume the post-degradation model, not stale 'mesh'"
    );
}

// --- Routing behavior tests for model-independent daemon support ---

#[test]
fn has_local_unavailable_returns_false_for_empty_targets() {
    let targets = election::ModelTargets::default();
    assert!(!has_local_unavailable_candidates(&targets, "nonexistent"));
}

#[test]
fn has_local_unavailable_returns_true_when_all_none() {
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "loading-model".to_string(),
        vec![
            election::InferenceTarget::None,
            election::InferenceTarget::None,
        ],
    );
    assert!(has_local_unavailable_candidates(&targets, "loading-model"));
}

#[test]
fn has_local_unavailable_returns_false_when_any_available() {
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "partial-model".to_string(),
        vec![
            election::InferenceTarget::None,
            election::InferenceTarget::Local(9337),
        ],
    );
    assert!(!has_local_unavailable_candidates(&targets, "partial-model"));
}

#[test]
fn callable_models_excludes_all_none_targets() {
    let mut targets = election::ModelTargets::default();
    // Available model - included in callable list
    targets.targets.insert(
        "available".to_string(),
        vec![election::InferenceTarget::Local(9337)],
    );
    // Unavailable model (loading/draining) - excluded from callable list
    targets.targets.insert(
        "unavailable".to_string(),
        vec![
            election::InferenceTarget::None,
            election::InferenceTarget::None,
        ],
    );

    let models = callable_models(&targets);
    assert!(models.contains(&"available".to_string()));
    assert!(!models.contains(&"unavailable".to_string()));
}

#[test]
fn callable_models_returns_empty_when_no_targets() {
    let targets = election::ModelTargets::default();
    let models = callable_models(&targets);
    assert!(models.is_empty());
}

// --- Daemon state derivation tests for plugin-only and remote-only daemons ---

#[test]
fn daemon_ready_proxying_when_only_plugins_available() {
    use crate::api::status::{DaemonState, derive_daemon_state};

    assert_eq!(
        derive_daemon_state(
            false, // shutdown_requested
            false, // has_terminal_failure
            false, // priority_degraded
            false, // local_serving - no local models
            true,  // proxying - plugin endpoints available for routing
            true,  // listeners_ready
        ),
        DaemonState::ReadyProxying,
    );
}

#[test]
fn daemon_ready_proxying_when_only_remote_mesh_available() {
    use crate::api::status::{DaemonState, derive_daemon_state};

    assert_eq!(
        derive_daemon_state(
            false, // shutdown_requested
            false, // has_terminal_failure
            false, // priority_degraded
            false, // local_serving - no local models
            true,  // proxying - remote mesh targets available for routing
            true,  // listeners_ready
        ),
        DaemonState::ReadyProxying,
    );
}

#[test]
fn daemon_degraded_on_terminal_failure_not_killed() {
    use crate::api::status::{DaemonState, derive_daemon_state};

    assert_eq!(
        derive_daemon_state(
            false, // shutdown_requested - NOT stopping
            true,  // has_terminal_failure - model failed
            false, // priority_degraded
            false, // local_serving
            true,  // proxying still works for other capabilities
            true,  // listeners_ready
        ),
        DaemonState::Degraded,
    );
}

#[test]
fn daemon_stopping_only_when_shutdown_requested() {
    use crate::api::status::{DaemonState, derive_daemon_state};

    assert_eq!(
        derive_daemon_state(
            true,  // shutdown_requested - explicitly stopping
            false, // has_terminal_failure
            false, // priority_degraded
            true,  // local_serving (irrelevant when stopping)
            true,  // proxying (irrelevant when stopping)
            true,  // listeners_ready
        ),
        DaemonState::Stopping,
    );
}

#[test]
fn daemon_ready_idle_when_no_models_but_listeners_up() {
    use crate::api::status::{DaemonState, derive_daemon_state};

    assert_eq!(
        derive_daemon_state(
            false, // shutdown_requested
            false, // has_terminal_failure
            false, // priority_degraded
            false, // local_serving - no models loaded yet (on-demand mode)
            false, // proxying - not yet routing to mesh or plugins
            true,  // listeners_ready - HTTP listeners are up and accepting connections
        ),
        DaemonState::ReadyIdle,
    );
}

#[tokio::test]
async fn api_proxy_tokenizer_route_ignores_generation_context_budget() {
    let model = "acme/code-model:Q4_K_M";
    let mut request = large_tokenize_request(model);
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node should start");
    node.set_model_runtime_context_length(model, Some(32_768))
        .await;
    let target = election::InferenceTarget::Local(19_337);
    let mut targets = election::ModelTargets::default();
    targets
        .targets
        .insert(model.to_owned(), vec![target.clone()]);
    let affinity = affinity::AffinityRouter::new();
    let ctx = IngressRouteContext {
        node: &node,
        targets: &targets,
        affinity: &affinity,
        plugin_manager: None,
        exchange_channel: None,
    };
    let raw_before_decision = request.raw.clone();

    let generation_budget =
        proxy::request_budget_tokens_from_parts(request.body_len_bytes, request.completion_tokens);
    assert!(generation_budget.is_some_and(|tokens| tokens > 32_768));
    assert!(
        crate::network::openai::routing_rank::order_targets_by_context(
            &node,
            model,
            generation_budget,
            std::slice::from_ref(&target),
        )
        .await
        .is_empty(),
        "a generation budget would incorrectly reject the tokenizer target"
    );

    let decision = prepare_auto_route_decision(&mut request, &ctx, &[])
        .await
        .expect("tokenizer route should not enter media auto-routing");
    assert_eq!(decision.effective_model.as_deref(), Some(model));
    assert_eq!(decision.required_tokens, None);
    assert_eq!(request.raw, raw_before_decision);
    assert!(request.body_json.is_none());
    assert!(!request.body_json_attempted);
    assert_eq!(proxy::request_context_budget(&request), None);
    assert_eq!(
        crate::network::openai::routing_rank::order_targets_by_context(
            &node,
            model,
            proxy::request_context_budget(&request),
            std::slice::from_ref(&target),
        )
        .await,
        vec![target]
    );
}

// --- #1331 M2: path 2 (raw-proxy ingress) openai-exchange status mapping ---
//
// `try_route_plugin_model` itself needs a live TCP stream and a real
// `mesh::Node` (see the `plugin_lifecycle`/`record_plugin_attempt` doubles
// above, which exist because the whole function is not economical to invoke
// directly in a unit test); `plugin_route_status` is the pure piece of that
// call site's openai-exchange wiring, so it's tested directly instead.
#[test]
fn plugin_route_status_maps_responded_and_responded_with_usage() {
    assert_eq!(
        plugin_route_status(&proxy::RouteDispatchOutcome::Responded(200)),
        Some(200)
    );
    assert_eq!(
        plugin_route_status(&proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: 200,
            usage: mesh_llm_events::logging::events::TokenUsage::from_counts(
                Some(1),
                Some(1),
                Some(2)
            )
            .unwrap(),
        }),
        Some(200)
    );
}

#[test]
fn plugin_route_status_maps_failed_with_status_and_omits_statusless_outcomes() {
    assert_eq!(
        plugin_route_status(&proxy::RouteDispatchOutcome::FailedWithStatus {
            status_code: 503,
            reason: "plugin_endpoint_failed",
        }),
        Some(503)
    );
    assert_eq!(
        plugin_route_status(&proxy::RouteDispatchOutcome::Failed(
            "plugin_endpoint_failed"
        )),
        None
    );
    assert_eq!(
        plugin_route_status(&proxy::RouteDispatchOutcome::Dropped(
            "response_write_failed"
        )),
        None
    );
}

#[test]
fn plugin_route_success_records_one_attempt_and_one_terminal_outcome() {
    let (service, mut attachment) = plugin_lifecycle();
    let observer = attachment.route_observer();
    let result = record_plugin_attempt(
        observer,
        "plugin-model",
        "acme/plugin",
        "endpoint-prod",
        super::super::response::RouteAttemptResult::Delivered {
            status_code: 200,
            usage: None,
            cache_cost: None,
        },
    );
    assert!(matches!(
        result,
        super::super::response::RouteAttemptResult::Delivered {
            status_code: 200,
            ..
        }
    ));

    attachment.terminal(terminal_outcome_for_dispatch(
        proxy::RouteDispatchOutcome::Responded(200),
    ));
    attachment.terminal(TerminalOutcome::Failed("late_plugin_failure".into()));

    let events = recorded_lifecycle_events(&service);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::RouteSelected { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::AttemptStarted { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::AttemptCompleted { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Completed { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Failed { .. }))
            .count(),
        0
    );
    match events
        .iter()
        .find(|event| matches!(event, LifecycleEvent::RouteSelected { .. }))
    {
        Some(LifecycleEvent::RouteSelected {
            model,
            provider,
            engine,
        }) => {
            assert_eq!(model.as_deref(), Some("plugin-model"));
            assert_eq!(provider.as_deref(), Some("acme/plugin"));
            assert_eq!(engine.as_deref(), Some("endpoint-prod"));
        }
        other => panic!("expected one plugin route selection, got {other:?}"),
    }
    assert_payload_free(&events);
}

#[test]
fn plugin_route_failure_records_failed_attempt_and_terminal_outcome() {
    let (service, mut attachment) = plugin_lifecycle();
    let observer = attachment.route_observer();
    let result = record_plugin_attempt(
        observer,
        "plugin-model",
        "plugin.example",
        "sk_test",
        super::super::response::RouteAttemptResult::RetryableUnavailable,
    );
    assert_eq!(
        result,
        super::super::response::RouteAttemptResult::RetryableUnavailable
    );

    attachment.terminal(terminal_outcome_for_dispatch(
        proxy::RouteDispatchOutcome::Failed("plugin_endpoint_failed"),
    ));
    attachment.terminal(TerminalOutcome::Completed);

    let events = recorded_lifecycle_events(&service);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::AttemptStarted { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::AttemptFailed { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Failed { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Completed { .. }))
            .count(),
        0
    );
    match events
        .iter()
        .find(|event| matches!(event, LifecycleEvent::AttemptFailed { .. }))
    {
        Some(LifecycleEvent::AttemptFailed { error, .. }) => {
            assert_eq!(error.as_deref(), Some("retryable_unavailable"));
        }
        other => panic!("expected one plugin attempt failure, got {other:?}"),
    }
    assert_payload_free(&events);
}

#[test]
fn plugin_route_without_endpoint_records_decision_without_attempt_or_payload() {
    let (service, mut attachment) = plugin_lifecycle();
    let observer = attachment.route_observer();
    observer.route_selected_with_metadata(
        Some("plugin-model"),
        Some("plugin"),
        Some("inference_endpoint"),
    );
    attachment.terminal(terminal_outcome_for_dispatch(
        proxy::RouteDispatchOutcome::Responded(404),
    ));
    attachment.terminal(TerminalOutcome::Failed("late_plugin_failure".into()));

    let events = recorded_lifecycle_events(&service);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::RouteSelected { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::AttemptStarted { .. }))
            .count(),
        0
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Rejected { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, LifecycleEvent::Failed { .. }))
            .count(),
        0
    );
    assert_payload_free(&events);
}

#[test]
fn load_and_unload_error_statuses_never_complete_lifecycle() {
    for status in [400, 404, 409, 500, 503] {
        assert!(!matches!(
            terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(status)),
            TerminalOutcome::Completed
        ));
    }
}

#[test]
fn unknown_and_unavailable_models_map_to_rejected_and_failed() {
    assert!(matches!(
        terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(404)),
        TerminalOutcome::RejectedWithStatus {
            status_code: 404,
            ..
        }
    ));
    assert!(matches!(
        terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(503)),
        TerminalOutcome::FailedWithStatus {
            status_code: 503,
            ..
        }
    ));
}

#[test]
fn invalid_and_failed_moa_responses_map_from_http_status() {
    assert!(matches!(
        terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(400)),
        TerminalOutcome::RejectedWithStatus {
            status_code: 400,
            ..
        }
    ));
    assert!(matches!(
        terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(502)),
        TerminalOutcome::FailedWithStatus {
            status_code: 502,
            ..
        }
    ));
}

#[test]
fn usage_never_turns_moa_or_pipeline_error_statuses_into_success() {
    let usage = mesh_llm_events::logging::events::TokenUsage {
        prompt_tokens: Some(8),
        cached_prompt_tokens: None,
        completion_tokens: Some(5),
        total_tokens: Some(13),
    };
    assert!(matches!(
        terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: 400,
            usage,
        }),
        TerminalOutcome::RejectedWithStatus {
            status_code: 400,
            ..
        }
    ));
    assert!(matches!(
        terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: 502,
            usage,
        }),
        TerminalOutcome::FailedWithStatus {
            status_code: 502,
            ..
        }
    ));
}

#[test]
fn streamed_moa_chat_and_responses_record_compatible_usage_lifecycle() {
    let usage = mesh_llm_events::logging::events::TokenUsage {
        prompt_tokens: Some(8),
        cached_prompt_tokens: None,
        completion_tokens: Some(5),
        total_tokens: Some(13),
    };
    for adapter in [
        proxy::ResponseAdapter::OpenAiChatCompletionsStream,
        proxy::ResponseAdapter::OpenAiResponsesStream,
    ] {
        let (service, mut attachment) = plugin_lifecycle();
        let outcome = proxy::RouteDispatchOutcome::RespondedWithUsage {
            status_code: 200,
            usage,
        };
        proxy::record_moa_stream_lifecycle(attachment.route_observer(), adapter, outcome);
        attachment.terminal(terminal_outcome_for_dispatch(outcome));

        let events = recorded_lifecycle_events(&service);
        assert!(events.iter().any(|event| matches!(
            event,
            LifecycleEvent::StreamStarted { model }
                if model.as_deref() == Some(moa::VIRTUAL_MODEL_NAME)
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            LifecycleEvent::StreamCompleted {
                tokens: Some(5),
                usage: Some(recorded),
            } if *recorded == usage
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            LifecycleEvent::Completed {
                status_code: Some(200),
                usage: Some(recorded),
                ..
            } if *recorded == usage
        )));
    }
}

#[test]
fn pipeline_server_error_is_failed_not_completed() {
    assert!(matches!(
        terminal_outcome_for_dispatch(proxy::RouteDispatchOutcome::Responded(500)),
        TerminalOutcome::FailedWithStatus {
            status_code: 500,
            ..
        }
    ));
}

#[test]
fn disconnect_is_dropped_and_cannot_audit_model_access_as_success() {
    let outcome = proxy::RouteDispatchOutcome::Dropped("client_disconnected");
    assert!(matches!(
        terminal_outcome_for_dispatch(outcome),
        TerminalOutcome::Dropped(_)
    ));
    assert!(!model_access_succeeded(outcome));
    assert!(!model_access_succeeded(
        proxy::RouteDispatchOutcome::Responded(502)
    ));
    assert!(model_access_succeeded(
        proxy::RouteDispatchOutcome::Responded(200)
    ));
}

// --- #1668 round-2: call-site test for route_missing_local_model ---

/// Seed a `mesh::Node` with one admitted `Host` peer serving `model` at a
/// fake address. `hosts_for_model` returns the peer's `EndpointId` without
/// any gossip round trip, so `remote_mesh_targets` returns `Some` and
/// `route_missing_local_model` takes the remote-mesh branch.
fn test_remote_peer(seed: u32, model: &str) -> mesh::PeerInfo {
    // Deterministic fake id derived from seed so callers can build multiple
    // non-colliding peers.
    let secret = {
        let mut bytes = [0u8; 32];
        let seed_bytes = seed.to_le_bytes();
        bytes[..4].copy_from_slice(&seed_bytes);
        bytes[4] = 0xde;
        bytes[5] = 0xad;
        iroh::SecretKey::try_from(bytes).expect("fixed-length test key")
    };
    let peer_id = iroh::EndpointId::from(secret.public());
    mesh::PeerInfo {
        id: peer_id,
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        // Host role is required for `accepts_http_inference()` and
        // therefore for `routes_http_model()` and `hosts_for_model()`.
        role: mesh::NodeRole::Host { http_port: 9337 },
        first_joined_mesh_ts: None,
        models: vec![model.to_string()],
        vram_bytes: 16 * 1024 * 1024 * 1024,
        rtt_ms: None,
        model_source: None,
        // admitted: true required for `is_admitted()`.
        admitted: true,
        serving_models: vec![model.to_string()],
        hosted_models: vec![model.to_string()],
        hosted_models_known: true,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        last_seen: std::time::Instant::now(),
        last_mentioned: std::time::Instant::now(),
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: std::collections::HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        advertised_model_throughput: vec![],
        cache_affinity: None,
        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
        owner_summary: crate::crypto::OwnershipSummary::default(),
        inference_admission_state: None,
    }
}

/// Verifies that `route_missing_local_model` enters the remote-mesh branch
/// when at least one admitted peer serves the requested model, AND that the
/// function publishes BOTH the effective-request and terminal envelopes on
/// that branch via `IngressRouteContext::exchange_channel`.
///
/// This test was previously defective (erlich review): it passed
/// `plugin_manager: None`, so the two publish calls — both guarded by
/// `if let Some(plugin_manager) = ctx.plugin_manager` — were never executed,
/// and the test could not observe the publish pair it claimed to prove. The
/// fix adds a `#[cfg(test)]` `exchange_channel` field to `IngressRouteContext`
/// that accepts a recording double injected here without needing a live
/// `PluginManager`.
///
/// erlich's four required assertions:
///   (1) TWO messages published (effective + terminal)
///   (2) `dispatch_path: RemoteMesh` on both envelopes
///   (3) matching `exchange_id` across the pair
///   (4) the SAME nonce on both (matches the nonce in the forwarded request)
///
/// Failure proof: if either publish call in `route_missing_local_model` is
/// deleted, `events.len()` drops to 1 and assertion (1) fails. Run:
///   `cargo test -p mesh-llm-host-runtime route_missing_local_model 2>&1`
/// with one publish removed to confirm the test catches the defect.
#[tokio::test]
async fn route_missing_local_model_enters_remote_mesh_branch_when_peer_serves_model() {
    use crate::plugin::openai_exchange::{OpenAiExchangeDispatchPath, OpenAiExchangePhase};

    let model = "acme/remote-model:Q4_K_M";
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.insert_test_peer(test_remote_peer(1, model)).await;

    let targets = election::ModelTargets::default();
    let affinity = affinity::AffinityRouter::new();

    // Recording double — accumulates every envelope the publish calls emit.
    let recording = RecordingChannel::default();

    // Loopback TCP pair — the server side becomes the ClientStream.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("local addr");
    let client_connect = tokio::net::TcpStream::connect(addr);
    let server_accept = async { listener.accept().await.map(|(s, _)| s) };
    let (_client_side, server_side) = tokio::join!(client_connect, server_accept);
    let tcp_stream = server_side.expect("accept server side");

    // Build a minimal chat-completion request stamped with a client nonce so
    // `capsule_nonce_headers()` returns `Some`. The nonce must survive into
    // both published envelopes unchanged (assertion 4).
    let body =
        br#"{"model":"acme/remote-model:Q4_K_M","messages":[{"role":"user","content":"hi"}]}"#;
    let nonce = "a1b2c3d4-e5f6-7890-abcd-ef1234567890";
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nx-capsule-client-nonce: {nonce}\r\n\r\n",
        len = body.len(),
        nonce = nonce,
    )
    .into_bytes()
    .into_iter()
    .chain(body.iter().copied())
    .collect::<Vec<u8>>();
    let request = proxy::BufferedHttpRequest {
        raw,
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: body.len(),
        completion_tokens: None,
        stream: None,
        model_name: Some(model.to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::OpenAiChatCompletionsJson,
        correlation_id: None,
    };

    // Confirm the nonce header round-trips through the raw bytes before the
    // function reads it — a prerequisite for assertion (4).
    let (parsed_nonce, _origin) = request.capsule_nonce_headers();
    assert_eq!(
        parsed_nonce.as_deref(),
        Some(nonce),
        "nonce header must be readable from the raw request bytes"
    );

    let ctx = IngressRouteContext {
        node: &node,
        targets: &targets,
        affinity: &affinity,
        plugin_manager: None,
        // Inject the recording double so both publish calls are observable
        // even though plugin_manager is None.
        exchange_channel: Some(&recording),
    };
    let lifecycle = OpenAiLifecycleAttachment::unowned();

    let outcome = route_missing_local_model(
        tcp_stream.into(),
        &request,
        &ctx,
        model,
        None,
        lifecycle.route_observer(),
    )
    .await;

    // A 404 would mean remote_mesh_targets returned None (peer not seen).
    // Any other outcome — Failed, Dropped, or a non-404 status — proves the
    // remote-mesh branch was entered, which is what this test pins.
    assert!(
        !matches!(outcome, proxy::RouteDispatchOutcome::Responded(404)),
        "expected remote-mesh branch (not 404), got {outcome:?} — \
         the test peer may not be visible to hosts_for_model()"
    );

    let events = recording.events.lock().unwrap();

    // (1) TWO messages published — effective + terminal.
    // If either publish call is deleted this assertion is the first to fail.
    assert_eq!(
        events.len(),
        2,
        "route_missing_local_model must publish both the effective-request \
         and terminal envelopes on the remote-mesh branch; got {} event(s)",
        events.len()
    );

    // (2) Both envelopes carry `RemoteMesh` as the dispatch path.
    assert_eq!(
        events[0].dispatch_path,
        OpenAiExchangeDispatchPath::RemoteMesh,
        "effective envelope must carry RemoteMesh dispatch path"
    );
    assert_eq!(
        events[1].dispatch_path,
        OpenAiExchangeDispatchPath::RemoteMesh,
        "terminal envelope must carry RemoteMesh dispatch path"
    );

    // Sanity: correct phases.
    assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
    assert_eq!(events[1].phase, OpenAiExchangePhase::Terminal);

    // (3) Matching exchange_id across the pair.
    assert!(
        !events[0].exchange_id.is_empty(),
        "exchange_id must be non-empty"
    );
    assert_eq!(
        events[0].exchange_id, events[1].exchange_id,
        "effective and terminal envelopes must share the same exchange_id"
    );

    // (4) The SAME nonce on both envelopes, matching the request's nonce header.
    assert_eq!(
        events[0].nonce.as_deref(),
        Some(nonce),
        "effective envelope must carry the forwarded client nonce"
    );
    assert_eq!(
        events[0].nonce, events[1].nonce,
        "effective and terminal envelopes must carry the same nonce"
    );
}

/// Verifies that `route_missing_local_model` sets `nonce_source =
/// Some(SidecarGeneratedFallback)` on both published envelopes when the
/// request carries BOTH `x-capsule-client-nonce` AND `x-capsule-nonce-origin`.
///
/// The `remote_mesh_nonce_source` helper (ingress.rs lines 35-46) maps:
/// - nonce=Some, nonce_origin=None  → `ClientSupplied`   (covered by the
///   sibling test above)
/// - nonce=Some, nonce_origin=Some  → `SidecarGeneratedFallback`  ← this test
///
/// This test exercises the second branch by stamping both headers on the raw
/// request, then asserting that every published envelope reports
/// `nonce_source == Some(SidecarGeneratedFallback)`.
#[tokio::test]
async fn route_missing_local_model_sidecar_generated_nonce_origin_sets_sidecar_fallback_source() {
    use crate::plugin::openai_exchange::{
        ClientNonceSource, OpenAiExchangeDispatchPath, OpenAiExchangePhase,
    };

    let model = "acme/remote-model:Q4_K_M";
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.insert_test_peer(test_remote_peer(1, model)).await;

    let targets = election::ModelTargets::default();
    let affinity = affinity::AffinityRouter::new();

    let recording = RecordingChannel::default();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("local addr");
    let client_connect = tokio::net::TcpStream::connect(addr);
    let server_accept = async { listener.accept().await.map(|(s, _)| s) };
    let (_client_side, server_side) = tokio::join!(client_connect, server_accept);
    let tcp_stream = server_side.expect("accept server side");

    // Build a request stamped with BOTH x-capsule-client-nonce AND
    // x-capsule-nonce-origin. The presence of x-capsule-nonce-origin signals
    // that the frontend generated the nonce as a fallback (SidecarGeneratedFallback).
    let body =
        br#"{"model":"acme/remote-model:Q4_K_M","messages":[{"role":"user","content":"hi"}]}"#;
    let nonce = "b2c3d4e5-f6a7-8901-bcde-f12345678901";
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nx-capsule-client-nonce: {nonce}\r\nx-capsule-nonce-origin: frontend\r\n\r\n",
        len = body.len(),
        nonce = nonce,
    )
    .into_bytes()
    .into_iter()
    .chain(body.iter().copied())
    .collect::<Vec<u8>>();
    let request = proxy::BufferedHttpRequest {
        raw,
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: body.len(),
        completion_tokens: None,
        stream: None,
        model_name: Some(model.to_owned()),
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::OpenAiChatCompletionsJson,
        correlation_id: None,
    };

    // Confirm both headers are readable before the routing function runs.
    let (parsed_nonce, parsed_origin) = request.capsule_nonce_headers();
    assert_eq!(
        parsed_nonce.as_deref(),
        Some(nonce),
        "nonce header must be readable from the raw request bytes"
    );
    assert!(
        parsed_origin.is_some(),
        "nonce-origin header must be readable from the raw request bytes"
    );

    let ctx = IngressRouteContext {
        node: &node,
        targets: &targets,
        affinity: &affinity,
        plugin_manager: None,
        exchange_channel: Some(&recording),
    };
    let lifecycle = OpenAiLifecycleAttachment::unowned();

    let outcome = route_missing_local_model(
        tcp_stream.into(),
        &request,
        &ctx,
        model,
        None,
        lifecycle.route_observer(),
    )
    .await;

    assert!(
        !matches!(outcome, proxy::RouteDispatchOutcome::Responded(404)),
        "expected remote-mesh branch (not 404), got {outcome:?}"
    );

    let events = recording.events.lock().unwrap();

    assert_eq!(
        events.len(),
        2,
        "route_missing_local_model must publish both the effective-request \
         and terminal envelopes on the remote-mesh branch; got {} event(s)",
        events.len()
    );

    assert_eq!(
        events[0].dispatch_path,
        OpenAiExchangeDispatchPath::RemoteMesh,
        "effective envelope must carry RemoteMesh dispatch path"
    );
    assert_eq!(
        events[1].dispatch_path,
        OpenAiExchangeDispatchPath::RemoteMesh,
        "terminal envelope must carry RemoteMesh dispatch path"
    );

    assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
    assert_eq!(events[1].phase, OpenAiExchangePhase::Terminal);

    // Both envelopes must report SidecarGeneratedFallback because
    // x-capsule-nonce-origin was present on the request.
    assert_eq!(
        events[0].nonce_source,
        Some(ClientNonceSource::SidecarGeneratedFallback),
        "effective envelope must carry SidecarGeneratedFallback nonce_source \
         when x-capsule-nonce-origin is present"
    );
    assert_eq!(
        events[1].nonce_source,
        Some(ClientNonceSource::SidecarGeneratedFallback),
        "terminal envelope must carry SidecarGeneratedFallback nonce_source \
         when x-capsule-nonce-origin is present"
    );
}
