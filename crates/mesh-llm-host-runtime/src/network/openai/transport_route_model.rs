use super::*;
use crate::network::openai::routing_rank::{RankedCandidates, rank_targets_by_context};
use crate::network::reservations::RoutingReservation;

pub(crate) struct RouteModelRequestContext<'a> {
    pub(crate) required_tokens: Option<u32>,
    pub(crate) affinity: &'a AffinityRouter,
    pub(crate) route_observer: OpenAiRouteObserver<'a>,
}

pub async fn route_model_request(
    node: mesh::Node,
    tcp_stream: ClientStream,
    targets: &election::ModelTargets,
    model: &str,
    request: &BufferedHttpRequest,
    context: RouteModelRequestContext<'_>,
) -> RouteDispatchOutcome {
    let args = RouteModelRequestArgs {
        node,
        tcp_stream,
        targets,
        model,
        request,
        required_tokens: context.required_tokens,
        affinity: context.affinity,
        route_observer: context.route_observer,
    };
    route_model_request_inner(args).await
}

struct RouteModelRequestArgs<'a> {
    node: mesh::Node,
    tcp_stream: ClientStream,
    targets: &'a election::ModelTargets,
    model: &'a str,
    request: &'a BufferedHttpRequest,
    required_tokens: Option<u32>,
    affinity: &'a AffinityRouter,
    route_observer: OpenAiRouteObserver<'a>,
}

struct RouteModelState {
    route_started: Instant,
    attempts: usize,
    refreshed: bool,
}

enum RouteModelDisposition {
    Continue,
    Return(RouteDispatchOutcome),
}

fn no_context_eligible_target_reason(model: &str, required_tokens: Option<u32>) -> String {
    match required_tokens {
        Some(tokens) => format!(
            "no context-compatible target for model '{model}' can fit approximately {tokens} tokens"
        ),
        None => format!("no eligible target for model '{model}'"),
    }
}

async fn cache_target_for_request(
    node: &mesh::Node,
    affinity: &AffinityRouter,
    model: &str,
    prefix_hash: Option<u64>,
    candidates: &[election::InferenceTarget],
) -> Option<election::InferenceTarget> {
    let prefix_hash = prefix_hash?;
    if let Some(target) = affinity.lookup_cache_lease(model, prefix_hash, candidates) {
        return Some(target);
    }

    let lease_epoch = affinity.cache_lease_epoch();
    let selected = node
        .select_cache_target(model, prefix_hash, candidates)
        .await;
    if let Some(target) = selected.as_ref() {
        affinity.remember_cache_lease_if_epoch(model, prefix_hash, target, lease_epoch);
    }
    selected
}

async fn route_model_request_inner(args: RouteModelRequestArgs<'_>) -> RouteDispatchOutcome {
    let RouteModelRequestArgs {
        node,
        tcp_stream,
        targets,
        model,
        request,
        required_tokens,
        affinity,
        route_observer,
    } = args;
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    let ranked =
        rank_targets_by_context(&node, model, required_tokens, &targets.candidates(model)).await;
    let ordered_candidates = affinity.route_eligible_candidates(model, &ranked.ordered);
    if ordered_candidates.is_empty() {
        record_route_model_unavailable(&node, model, 0);
        let reason = no_context_eligible_target_reason(model, required_tokens);
        return response_outcome(
            503,
            send_503_observed(tcp_stream, &reason, route_observer).await,
        );
    }
    route_observer.route_selected(Some(model));

    let prefix_hash = crate::network::affinity::cache_prefix_hash(request.body_json.as_ref());
    let cache_target =
        cache_target_for_request(&node, affinity, model, prefix_hash, &ordered_candidates).await;
    let Some(ReservedModelRoute {
        selection,
        ordered,
        mut reservation,
    }) = select_and_reserve_model_route(
        targets,
        &ranked,
        model,
        request.body_json.as_ref(),
        affinity,
        cache_target,
    )
    else {
        return send_route_model_none_target(&node, tcp_stream, model, route_observer).await;
    };
    let total_targets = ordered.len();
    let mut state = RouteModelState {
        route_started,
        attempts: 0,
        refreshed: false,
    };
    // `request.raw` was already stabilized at ingress (finalize_forwarded_request
    // resolves the capsule client nonce once, before target selection), so every
    // attempt here — including a timeout retry to a different target — forwards
    // the identical nonce instead of letting each target's frontend mint its own.
    let forwarding_raw = request.raw.as_slice();
    for (idx, target) in ordered.into_iter().enumerate() {
        reservation.transfer_to(&target);
        state.attempts += 1;
        let attempt_started = Instant::now();
        let retry_policy = ResponseRetryPolicy::next_target_available(idx + 1 < total_targets);
        let attempt_result = route_attempt_for_target(
            &node,
            &mut tcp_stream,
            &target,
            forwarding_raw,
            retry_policy,
            RouteAttemptLoggingContext {
                request_id: request.request_id,
                retry_policy,
                response_adapter: request.response_adapter,
                route_observer,
            },
        )
        .await;
        let queue_wait = attempt_started.duration_since(route_started);
        let attempt_time = attempt_started.elapsed();
        record_route_model_attempt(
            &node,
            model,
            &target,
            queue_wait,
            attempt_time,
            &attempt_result,
        );
        affinity.record_target_outcome(
            Some(model),
            &target,
            target_health_outcome_for_attempt(&attempt_result),
        );
        tracing::info!(
            model = model,
            target = ?target,
            attempt = state.attempts,
            total_targets = total_targets,
            outcome = route_attempt_result_label(&attempt_result),
            attempt_ms = attempt_started.elapsed().as_millis(),
            total_route_ms = route_started.elapsed().as_millis(),
            "openai route_model_request attempt"
        );
        match handle_route_model_attempt_result(
            &node,
            model,
            &target,
            &selection,
            affinity,
            attempt_result,
            &mut state,
        ) {
            RouteModelDisposition::Continue => continue,
            RouteModelDisposition::Return(result) => {
                return finalize_route_model_result(
                    &node,
                    model,
                    request,
                    route_started,
                    state.attempts,
                    result,
                    &target,
                );
            }
        }
    }

    finish_exhausted_route_model_request(
        &node,
        tcp_stream,
        model,
        total_targets,
        &state,
        route_observer,
    )
    .await
}

struct ReservedModelRoute {
    selection: TargetSelection,
    ordered: Vec<election::InferenceTarget>,
    reservation: RoutingReservation,
}

fn select_and_reserve_model_route(
    targets: &election::ModelTargets,
    ranked: &RankedCandidates<election::InferenceTarget>,
    model: &str,
    parsed_body: Option<&serde_json::Value>,
    affinity: &AffinityRouter,
    cache_target: Option<election::InferenceTarget>,
) -> Option<ReservedModelRoute> {
    // Cache lookup can await while another request cools a target. Refresh
    // health once here, then use exactly this snapshot for selection,
    // reservation spreading, and retries. A second filter inside selection
    // would let the reservation undo a health decision it never saw.
    let mut ordered = affinity.route_eligible_candidates(model, &ranked.ordered);
    let mut selection = crate::network::affinity::select_model_target_from_eligible_candidates(
        targets,
        &ordered,
        parsed_body,
        affinity,
        cache_target,
    );
    if matches!(selection.target, election::InferenceTarget::None) {
        return None;
    }
    // Health policy can remove or reorder candidates, so the original
    // equivalent run can shrink or disappear. Never let its old length admit
    // a lower-ranked fallback.
    let spread_limit = ordered
        .iter()
        .take_while(|candidate| ranked.ordered[..ranked.equivalent_prefix].contains(candidate))
        .count();
    let (target, reservation) = affinity.reserve_route(
        model,
        &ordered,
        spread_limit,
        &selection.target,
        selection.affinity_applied,
    )?;
    selection.target = target;
    move_target_first(&mut ordered, &selection.target);
    Some(ReservedModelRoute {
        selection,
        ordered,
        reservation,
    })
}

fn record_route_model_unavailable(node: &mesh::Node, model: &str, attempts: usize) {
    node.record_routed_request(
        Some(model),
        attempts,
        crate::network::metrics::RequestOutcome::Unavailable,
    );
}

async fn send_route_model_none_target(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    model: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteDispatchOutcome {
    record_route_model_unavailable(node, model, 0);
    let result = send_503_observed(
        tcp_stream,
        &format!("target for model '{model}' resolved to None (election in progress or host down)"),
        route_observer,
    )
    .await;
    response_outcome(503, result)
}

async fn finish_exhausted_route_model_request(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    model: &str,
    total_targets: usize,
    state: &RouteModelState,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteDispatchOutcome {
    let result = send_503_observed(
        tcp_stream,
        &format!("all {} target(s) for model '{model}' failed", total_targets),
        route_observer,
    )
    .await;
    record_route_model_unavailable(node, model, state.attempts);
    tracing::warn!(
        model = model,
        attempts = state.attempts,
        route_ms = state.route_started.elapsed().as_millis(),
        "openai route_model_request exhausted targets"
    );
    response_outcome(503, result)
}

fn handle_route_model_attempt_result(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    selection: &TargetSelection,
    affinity: &AffinityRouter,
    attempt_result: RouteAttemptResult,
    state: &mut RouteModelState,
) -> RouteModelDisposition {
    match attempt_result {
        RouteAttemptResult::Delivered {
            status_code,
            usage,
            cache_cost,
        } => handle_delivered_route_model_attempt(
            DeliveredRouteModelContext {
                node,
                model,
                target,
                selection,
                affinity,
                state,
            },
            status_code,
            usage,
            cache_cost,
        ),
        RouteAttemptResult::RetryableContextOverflow => {
            handle_retryable_route_model_context(target)
        }
        RouteAttemptResult::RetryableResponseQuality(failure) => {
            handle_retryable_route_model_response_quality(target, failure)
        }
        RouteAttemptResult::RetryableTimeout => {
            handle_retryable_route_model_timeout(node, target, state)
        }
        RouteAttemptResult::RetryableUnavailable => {
            handle_retryable_route_model_unavailable(node, target, state)
        }
        RouteAttemptResult::CommittedStreamFailure { status_code } => {
            RouteModelDisposition::Return(RouteDispatchOutcome::FailedWithStatus {
                status_code,
                reason: "upstream_stream_incomplete",
            })
        }
        RouteAttemptResult::ClientDisconnected => {
            tracing::info!(
                model = model,
                attempts = state.attempts,
                route_ms = state.route_started.elapsed().as_millis(),
                "openai route_model_request downstream disconnected"
            );
            RouteModelDisposition::Return(RouteDispatchOutcome::Dropped("client_disconnected"))
        }
    }
}

struct DeliveredRouteModelContext<'a> {
    node: &'a mesh::Node,
    model: &'a str,
    target: &'a election::InferenceTarget,
    selection: &'a TargetSelection,
    affinity: &'a AffinityRouter,
    state: &'a RouteModelState,
}

fn handle_delivered_route_model_attempt(
    context: DeliveredRouteModelContext<'_>,
    status_code: u16,
    usage: Option<TokenUsage>,
    cache_cost: Option<CacheCostObservation>,
) -> RouteModelDisposition {
    update_local_cache_evidence(&context, status_code, usage.as_ref(), cache_cost);
    context.node.record_routed_request(
        Some(context.model),
        context.state.attempts,
        request_outcome_for_status(status_code, request_service_for_target(context.target)),
    );
    tracing::info!(
        model = context.model,
        attempts = context.state.attempts,
        status_code = status_code,
        route_ms = context.state.route_started.elapsed().as_millis(),
        "openai route_model_request delivered"
    );
    RouteModelDisposition::Return(
        usage.map_or(RouteDispatchOutcome::Responded(status_code), |usage| {
            RouteDispatchOutcome::RespondedWithUsage { status_code, usage }
        }),
    )
}

fn update_local_cache_evidence(
    context: &DeliveredRouteModelContext<'_>,
    status_code: u16,
    usage: Option<&TokenUsage>,
    cache_cost: Option<CacheCostObservation>,
) {
    if !(200..400).contains(&status_code) {
        return;
    }
    let is_local = matches!(context.target, election::InferenceTarget::Local(_));
    if is_local {
        context.node.observe_local_prefill_cost(
            context.model,
            cache_cost.and_then(|cost| cost.prefill_micros_per_token),
        );
    }

    let Some(prefix_hash) = context.selection.prefix_hash else {
        return;
    };
    let Some(cached_tokens) = usage.and_then(|usage| usage.cached_prompt_tokens) else {
        return;
    };
    if cached_tokens > 0 {
        if is_local {
            let suffix = usage
                .and_then(|usage| usage.prompt_tokens)
                .unwrap_or(cached_tokens)
                .saturating_sub(cached_tokens);
            context.node.record_local_cache_hit_with_cost(
                context.model,
                prefix_hash,
                u32::try_from(cached_tokens).unwrap_or(u32::MAX),
                u32::try_from(suffix).unwrap_or(u32::MAX),
                crate::network::affinity::LocalCacheCost {
                    queue_delay_micros: cache_cost.map_or(0, |cost| cost.queue_delay_micros),
                    restore_micros: cache_cost.map_or(0, |cost| cost.restore_micros),
                    // The model-level observation was recorded above so each
                    // response contributes exactly once to its EWMA.
                    prefill_micros_per_token: None,
                },
            );
        }
        return;
    }

    let inventory_invalidated = is_local
        && context
            .node
            .invalidate_local_cache_evidence(context.model, prefix_hash);
    let lease_invalidated = context
        .affinity
        .forget_cache_lease(context.model, prefix_hash);
    if inventory_invalidated || lease_invalidated {
        tracing::debug!(
            model = context.model,
            prefix_hash,
            inventory_invalidated,
            lease_invalidated,
            "invalidated cache affinity after authoritative miss"
        );
    }
}

fn handle_retryable_route_model_context(
    target: &election::InferenceTarget,
) -> RouteModelDisposition {
    tracing::warn!(
        "Target {target:?} rejected request with context overflow-style 400, trying next"
    );
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_response_quality(
    target: &election::InferenceTarget,
    failure: ResponseQualityFailure,
) -> RouteModelDisposition {
    tracing::warn!(
        reason = failure.label(),
        "Target {target:?} returned low-quality success response, trying next"
    );
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_timeout(
    node: &mesh::Node,
    target: &election::InferenceTarget,
    state: &mut RouteModelState,
) -> RouteModelDisposition {
    spawn_mesh_refresh_once(node, &mut state.refreshed);
    tracing::warn!("Target {target:?} timed out, trying next");
    RouteModelDisposition::Continue
}

fn handle_retryable_route_model_unavailable(
    node: &mesh::Node,
    target: &election::InferenceTarget,
    state: &mut RouteModelState,
) -> RouteModelDisposition {
    spawn_mesh_refresh_once(node, &mut state.refreshed);
    tracing::warn!("Target {target:?} unavailable, trying next");
    RouteModelDisposition::Continue
}

pub(crate) fn finalize_route_model_result(
    node: &mesh::Node,
    model: &str,
    _request: &BufferedHttpRequest,
    _route_started: Instant,
    _attempts: usize,
    result: RouteDispatchOutcome,
    target: &election::InferenceTarget,
) -> RouteDispatchOutcome {
    if let RouteDispatchOutcome::RespondedWithUsage { status_code, usage } = result {
        node.record_prompt_shape(
            Some(model),
            usage.prompt_tokens,
            usage.completion_tokens,
            request_outcome_for_status(status_code, request_service_for_target(target)),
        );
    }
    result
}

fn record_route_model_attempt(
    node: &mesh::Node,
    model: &str,
    target: &election::InferenceTarget,
    queue_wait: Duration,
    attempt_time: Duration,
    attempt_result: &RouteAttemptResult,
) {
    if matches!(attempt_result, RouteAttemptResult::ClientDisconnected) {
        return;
    }
    node.record_inference_attempt(
        Some(model),
        target,
        queue_wait,
        attempt_time,
        attempt_outcome_for_result(attempt_result),
        completion_tokens_for_result(attempt_result),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::target_health::TargetHealthOutcome;
    use iroh::SecretKey;

    #[test]
    fn reservation_does_not_reintroduce_target_cooled_during_cache_lookup() {
        let affinity = AffinityRouter::with_config(true, true);
        let first = election::InferenceTarget::Local(9001);
        let second = election::InferenceTarget::Local(9002);
        let ranked = RankedCandidates {
            ordered: vec![first.clone(), second.clone()],
            equivalent_prefix: 2,
        };
        let (_, pressure) = affinity
            .reserve_route("qwen", &ranked.ordered, 2, &second, true)
            .unwrap();
        // This is the snapshot passed to the asynchronous cache lookup.
        let before_lookup = affinity.route_eligible_candidates("qwen", &ranked.ordered);
        assert_eq!(before_lookup, ranked.ordered);
        // Another request times out while that lookup is suspended.
        affinity.record_target_outcome(Some("qwen"), &first, TargetHealthOutcome::Timeout);
        for cache_target in [None, Some(first.clone())] {
            let route = select_and_reserve_model_route(
                &election::ModelTargets::default(),
                &ranked,
                "qwen",
                None,
                &affinity,
                cache_target,
            )
            .unwrap();
            assert_eq!(route.selection.target, second);
            assert_eq!(route.ordered, vec![second.clone()]);
            assert!(!route.selection.affinity_applied);
            assert_eq!(affinity.stats_snapshot().reservation_active, 2);
        }
        drop(pressure);
        assert_eq!(affinity.stats_snapshot().reservation_active, 0);
    }

    #[test]
    fn refreshed_spread_window_does_not_admit_lower_ranked_fallback() {
        let first = election::InferenceTarget::Local(9001);
        let second = election::InferenceTarget::Local(9002);
        let fallback = election::InferenceTarget::Local(9003);
        // Cover both a shortened equivalent run and one removed entirely.
        for equivalent_prefix in [2, 1] {
            let affinity = AffinityRouter::new();
            let ranked = RankedCandidates {
                ordered: vec![first.clone(), second.clone(), fallback.clone()],
                equivalent_prefix,
            };
            let (_, _pressure) = affinity
                .reserve_route("qwen", &ranked.ordered, 3, &second, true)
                .unwrap();
            affinity.record_target_outcome(Some("qwen"), &first, TargetHealthOutcome::Timeout);
            let route = select_and_reserve_model_route(
                &election::ModelTargets::default(),
                &ranked,
                "qwen",
                None,
                &affinity,
                None,
            )
            .unwrap();
            assert_eq!(route.selection.target, second);
            assert_eq!(route.ordered, vec![second.clone(), fallback.clone()]);
        }
    }

    #[test]
    fn refreshed_route_preserves_healthy_cache_and_session_affinity_under_pressure() {
        let affinity = AffinityRouter::with_config(true, true);
        let ranked = RankedCandidates {
            ordered: vec![
                election::InferenceTarget::Local(9001),
                election::InferenceTarget::Local(9002),
            ],
            equivalent_prefix: 2,
        };
        let body = serde_json::json!({"user": "same-session"});
        for (parsed_body, cache_target) in
            [(None, Some(ranked.ordered[0].clone())), (Some(&body), None)]
        {
            let select = || {
                select_and_reserve_model_route(
                    &election::ModelTargets::default(),
                    &ranked,
                    "qwen",
                    parsed_body,
                    &affinity,
                    cache_target.clone(),
                )
                .unwrap()
            };
            let held = select();
            let next = select();
            assert!(held.selection.affinity_applied);
            assert!(next.selection.affinity_applied);
            assert_eq!(next.selection.target, held.selection.target);
        }
        assert_eq!(affinity.stats_snapshot().reservation_active, 0);
    }

    #[test]
    fn refreshed_route_keeps_all_cooling_availability_fallback() {
        let affinity = AffinityRouter::new();
        let ranked = RankedCandidates {
            ordered: vec![
                election::InferenceTarget::Local(9001),
                election::InferenceTarget::Local(9002),
            ],
            equivalent_prefix: 2,
        };
        for target in &ranked.ordered {
            affinity.record_target_outcome(Some("qwen"), target, TargetHealthOutcome::Timeout);
        }
        let route = select_and_reserve_model_route(
            &election::ModelTargets::default(),
            &ranked,
            "qwen",
            None,
            &affinity,
            None,
        )
        .unwrap();
        assert!(ranked.ordered.contains(&route.selection.target));
        assert_eq!(route.ordered.len(), 2);
    }

    #[test]
    fn empty_refreshed_route_does_not_reserve() {
        let affinity = AffinityRouter::new();
        assert!(
            select_and_reserve_model_route(
                &election::ModelTargets::default(),
                &RankedCandidates {
                    ordered: vec![],
                    equivalent_prefix: 0,
                },
                "qwen",
                None,
                &affinity,
                Some(election::InferenceTarget::Local(9001)),
            )
            .is_none()
        );
        assert_eq!(affinity.stats_snapshot().reservation_active, 0);
    }

    async fn cache_context(
        prefix_hash: u64,
        target: election::InferenceTarget,
    ) -> (
        mesh::Node,
        AffinityRouter,
        election::InferenceTarget,
        TargetSelection,
        RouteModelState,
    ) {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Host { http_port: 9337 })
            .await
            .expect("test node");
        let affinity = AffinityRouter::with_config(true, true);
        let selection = TargetSelection {
            target: target.clone(),
            prefix_hash: Some(prefix_hash),
            cache_target: Some(target.clone()),
            affinity_applied: true,
        };
        let state = RouteModelState {
            route_started: Instant::now(),
            attempts: 1,
            refreshed: false,
        };
        (node, affinity, target, selection, state)
    }

    #[tokio::test]
    async fn authoritative_local_miss_invalidates_inventory_and_lease() {
        let prefix_hash = 0xfeed_beef;
        let (node, affinity, target, selection, state) =
            cache_context(prefix_hash, election::InferenceTarget::Local(9337)).await;
        node.record_local_cache_hit("qwen", prefix_hash, 512, 24, 0);
        affinity.remember_cache_lease("qwen", prefix_hash, &target);
        let context = DeliveredRouteModelContext {
            node: &node,
            model: "qwen",
            target: &target,
            selection: &selection,
            affinity: &affinity,
            state: &state,
        };

        update_local_cache_evidence(
            &context,
            200,
            Some(&TokenUsage {
                prompt_tokens: Some(536),
                cached_prompt_tokens: Some(0),
                ..TokenUsage::default()
            }),
            None,
        );

        assert_eq!(
            node.select_cache_target("qwen", prefix_hash, std::slice::from_ref(&target))
                .await,
            None
        );
        assert_eq!(
            affinity.lookup_cache_lease("qwen", prefix_hash, std::slice::from_ref(&target)),
            None
        );
    }

    #[tokio::test]
    async fn authoritative_remote_miss_invalidates_lease_but_not_local_inventory() {
        let prefix_hash = 0xfeed_beef;
        let mut bytes = [0u8; 32];
        bytes[0] = 1;
        let remote = election::InferenceTarget::Remote(SecretKey::from_bytes(&bytes).public());
        let (node, affinity, target, selection, state) = cache_context(prefix_hash, remote).await;
        let local = election::InferenceTarget::Local(9337);
        node.record_local_cache_hit("qwen", prefix_hash, 512, 24, 0);
        affinity.remember_cache_lease("qwen", prefix_hash, &target);
        let context = DeliveredRouteModelContext {
            node: &node,
            model: "qwen",
            target: &target,
            selection: &selection,
            affinity: &affinity,
            state: &state,
        };

        update_local_cache_evidence(
            &context,
            200,
            Some(&TokenUsage {
                prompt_tokens: Some(536),
                cached_prompt_tokens: Some(0),
                ..TokenUsage::default()
            }),
            None,
        );

        assert_eq!(
            affinity.lookup_cache_lease("qwen", prefix_hash, std::slice::from_ref(&target)),
            None
        );
        assert_eq!(
            node.select_cache_target("qwen", prefix_hash, std::slice::from_ref(&local))
                .await,
            Some(local)
        );
    }

    #[tokio::test]
    async fn missing_cache_usage_does_not_refute_positive_evidence() {
        let prefix_hash = 0xfeed_beef;
        let (node, affinity, target, selection, state) =
            cache_context(prefix_hash, election::InferenceTarget::Local(9337)).await;
        node.record_local_cache_hit("qwen", prefix_hash, 512, 24, 0);
        affinity.remember_cache_lease("qwen", prefix_hash, &target);
        let context = DeliveredRouteModelContext {
            node: &node,
            model: "qwen",
            target: &target,
            selection: &selection,
            affinity: &affinity,
            state: &state,
        };

        update_local_cache_evidence(
            &context,
            200,
            Some(&TokenUsage {
                prompt_tokens: Some(536),
                cached_prompt_tokens: None,
                ..TokenUsage::default()
            }),
            None,
        );

        assert_eq!(
            node.select_cache_target("qwen", prefix_hash, std::slice::from_ref(&target))
                .await,
            Some(target.clone())
        );
        assert_eq!(
            affinity.lookup_cache_lease("qwen", prefix_hash, std::slice::from_ref(&target)),
            Some(target)
        );
    }

    #[tokio::test]
    async fn timing_without_cache_usage_still_calibrates_local_prefill_cost() {
        let prefix_hash = 0xfeed_beef;
        let calibration_probe = prefix_hash + 1;
        let (node, affinity, target, selection, state) =
            cache_context(prefix_hash, election::InferenceTarget::Local(9337)).await;
        let context = DeliveredRouteModelContext {
            node: &node,
            model: "qwen",
            target: &target,
            selection: &selection,
            affinity: &affinity,
            state: &state,
        };

        update_local_cache_evidence(
            &context,
            200,
            Some(&TokenUsage {
                prompt_tokens: Some(536),
                cached_prompt_tokens: None,
                ..TokenUsage::default()
            }),
            Some(CacheCostObservation {
                queue_delay_micros: 4_000,
                restore_micros: 8_000,
                prefill_micros_per_token: Some(250),
            }),
        );

        node.record_local_cache_hit("qwen", calibration_probe, 512, 24, 0);
        let entry = node
            .cache_affinity_inventory
            .lock()
            .unwrap()
            .probe_local("qwen", calibration_probe)
            .expect("calibrated local evidence");
        assert_eq!(entry.prefill_micros_per_token, 250);
    }

    #[tokio::test]
    async fn local_hit_records_measured_queue_restore_and_prefill_costs() {
        let prefix_hash = 0xfeed_beef;
        let (node, affinity, target, selection, state) =
            cache_context(prefix_hash, election::InferenceTarget::Local(9337)).await;
        let context = DeliveredRouteModelContext {
            node: &node,
            model: "qwen",
            target: &target,
            selection: &selection,
            affinity: &affinity,
            state: &state,
        };

        update_local_cache_evidence(
            &context,
            200,
            Some(&TokenUsage {
                prompt_tokens: Some(536),
                cached_prompt_tokens: Some(512),
                ..TokenUsage::default()
            }),
            Some(CacheCostObservation {
                queue_delay_micros: 4_000,
                restore_micros: 8_000,
                prefill_micros_per_token: Some(250),
            }),
        );

        let entry = node
            .cache_affinity_inventory
            .lock()
            .unwrap()
            .probe_local("qwen", prefix_hash)
            .expect("measured local evidence");
        assert_eq!(entry.queue_delay_micros, 4_000);
        assert_eq!(entry.restore_micros, 8_000);
        assert_eq!(entry.prefill_micros_per_token, 250);
    }
}
