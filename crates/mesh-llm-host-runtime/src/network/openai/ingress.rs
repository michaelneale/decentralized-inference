use crate::inference::{election, pipeline};
use crate::logging::{CallerPathType, OpenAiLifecycleAttachment, OpenAiRouteObserver};
use crate::mesh;
use crate::network::affinity;
use crate::network::openai::auto_route;
use crate::network::openai::automatic;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::transport as proxy;
use crate::network::router;
use crate::plugin::openai_exchange::{
    OpenAiExchangeChannel, OpenAiExchangeDispatchPath, OpenAiExchangeEnvelope,
};
use mesh_llm_events::audit::{audit_events, emit_audit};
use mesh_llm_events::{OutputEvent, emit_event};
use mesh_mixture_of_agents as moa;

/// The status code an out-of-process plugin sees for path 2's terminal
/// event, best-effort from [`proxy::RouteDispatchOutcome`] — `None` when the
/// outcome carries no HTTP status at all (a dropped/failed connection).
fn plugin_route_status(outcome: &proxy::RouteDispatchOutcome) -> Option<u16> {
    match *outcome {
        proxy::RouteDispatchOutcome::Responded(status) => Some(status),
        proxy::RouteDispatchOutcome::RespondedWithUsage { status_code, .. } => Some(status_code),
        proxy::RouteDispatchOutcome::FailedWithStatus { status_code, .. } => Some(status_code),
        proxy::RouteDispatchOutcome::Failed(_) | proxy::RouteDispatchOutcome::Dropped(_) => None,
    }
}

enum AutoRouteResolution {
    Continue {
        effective_model: Option<String>,
        classification: Option<router::Classification>,
    },
    MediaUnsupported,
}

struct IngressRouteContext<'a> {
    node: &'a mesh::Node,
    targets: &'a election::ModelTargets,
    affinity: &'a affinity::AffinityRouter,
    plugin_manager: Option<&'a crate::plugin::PluginManager>,
}

struct ProxyConnectionContext<'a> {
    route: IngressRouteContext<'a>,
}

struct AutoRouteDecision {
    effective_model: Option<String>,
    classification: Option<router::Classification>,
    required_tokens: Option<u32>,
}

fn terminal_outcome_for_dispatch(
    outcome: proxy::RouteDispatchOutcome,
) -> crate::logging::TerminalOutcome {
    outcome.terminal_outcome()
}

fn model_access_succeeded(outcome: proxy::RouteDispatchOutcome) -> bool {
    matches!(
        outcome,
        proxy::RouteDispatchOutcome::Responded(200..=299)
            | proxy::RouteDispatchOutcome::RespondedWithUsage {
                status_code: 200..=299,
                ..
            }
    )
}

fn response_outcome(status_code: u16, result: std::io::Result<()>) -> proxy::RouteDispatchOutcome {
    match result {
        Ok(()) => proxy::RouteDispatchOutcome::Responded(status_code),
        Err(_) => proxy::RouteDispatchOutcome::Dropped("response_write_failed"),
    }
}

/// Check activity policy admission and reject with 503 if paused.
async fn check_activity_admission(
    tcp_stream: ClientStream,
    guard: &crate::runtime::ActivityPolicyGuard,
    ingress_type: crate::runtime::IngressType,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<ClientStream, proxy::RouteDispatchOutcome> {
    match guard.check_admission(ingress_type) {
        crate::runtime::AdmissionResult::Allowed => Ok(tcp_stream),
        crate::runtime::AdmissionResult::Paused { reason, .. } => {
            tracing::debug!(reason, "Ingress rejected by activity policy");
            Err(response_outcome(
                503,
                proxy::send_503_observed(
                    tcp_stream,
                    &format!("inference paused: {reason}"),
                    route_observer,
                )
                .await,
            ))
        }
    }
}

/// Parse a model identifier that may include a profile suffix.
///
/// Returns `(model_ref, profile)` where:
/// - `model_ref` is the base model identifier (without `#profile`)
/// - `profile` is `Some(profile_name)` if `#profile` was present, `None` otherwise
///
/// Examples:
/// - `"Qwen/Qwen3-8B:Q4_K_M"` → `("Qwen/Qwen3-8B:Q4_K_M", None)`
/// - `"Qwen/Qwen3-8B:Q4_K_M#low-ctx"` → `("Qwen/Qwen3-8B:Q4_K_M", Some("low-ctx"))`
/// - `"model#"` → `("model", None)` (empty profile treated as None)
pub(super) fn parse_model_with_profile(model: &str) -> (&str, &str) {
    if let Some(hash_pos) = model.rfind('#') {
        let model_ref = &model[..hash_pos];
        let profile = &model[hash_pos + 1..];
        if profile.is_empty() {
            (model_ref, "")
        } else {
            (model_ref, profile)
        }
    } else {
        (model, "")
    }
}

async fn bind_api_proxy_listener(
    port: u16,
    existing_listener: Option<tokio::net::TcpListener>,
    listen_all: bool,
) -> Option<tokio::net::TcpListener> {
    match existing_listener {
        Some(listener) => Some(listener),
        None => {
            let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
            match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
                Ok(listener) => Some(listener),
                Err(error) => {
                    tracing::error!("Failed to bind API proxy to port {port}: {error}");
                    None
                }
            }
        }
    }
}

async fn handle_models_list_request(
    tcp_stream: ClientStream,
    node: &mesh::Node,
    targets: &election::ModelTargets,
    plugin_manager: Option<&crate::plugin::PluginManager>,
) -> proxy::RouteDispatchOutcome {
    let mut models = callable_models(targets);
    models.extend(node.models_being_served().await);
    if let Some(plugin_manager) = plugin_manager
        && let Ok(mut external_models) = plugin_manager.inference_models().await
    {
        models.append(&mut external_models);
    }
    models.sort();
    models.dedup();
    let descriptors = node.all_served_model_descriptors().await;
    let runtimes = node.all_model_runtime_descriptors().await;
    response_outcome(
        200,
        proxy::send_models_list_with_descriptors(tcp_stream, &models, &descriptors, &runtimes)
            .await,
    )
}

async fn collect_available_models_for_auto_route(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    plugin_manager: Option<&crate::plugin::PluginManager>,
) -> Vec<String> {
    let mut available_models = callable_models(targets);
    for name in node.models_being_served().await {
        if !available_models.iter().any(|existing| existing == &name) {
            available_models.push(name);
        }
    }
    if let Some(plugin_manager) = plugin_manager
        && let Ok(external_models) = plugin_manager.inference_models().await
    {
        for name in external_models {
            if !available_models.iter().any(|existing| existing == &name) {
                available_models.push(name);
            }
        }
    }
    available_models
}

async fn resolve_auto_routed_model(
    node: &mesh::Node,
    request: &mut proxy::BufferedHttpRequest,
    targets: &election::ModelTargets,
    plugin_manager: Option<&crate::plugin::PluginManager>,
    descriptors: &[crate::mesh::ServedModelDescriptor],
    required_tokens: Option<u32>,
    affinity: &affinity::AffinityRouter,
) -> AutoRouteResolution {
    // An explicitly named model routes to itself. The automatic directive (in
    // either spelling) resolves here, so `mesh` reaches the same
    // media-capability filter and readiness/affinity selection as `auto` —
    // without it, a `mesh` request with an image skipped the filter entirely
    // and had its image silently dropped downstream.
    if let Some(model) = request.model_name.as_deref()
        && !automatic::is_directive(model)
    {
        return AutoRouteResolution::Continue {
            effective_model: request.model_name.clone(),
            classification: None,
        };
    }

    request.ensure_body_json();
    let Some(body_json) = request.body_json.as_ref() else {
        return AutoRouteResolution::Continue {
            effective_model: None,
            classification: None,
        };
    };

    automatic::warn_if_deprecated_alias(request.model_name.as_deref());

    let mode = automatic::serving_mode(automatic::AutomaticRequest {
        model: request.model_name.as_deref(),
        // The forwarded path: `/v1/responses` has already been normalised onto
        // chat completions by this point, so it stays committee-eligible.
        path: &request.path,
        body: body_json,
    });
    match mode {
        // Committee mode keeps the directive as the effective model so the MoA
        // gateway picks the request up.
        automatic::ServingMode::Committee => {
            return AutoRouteResolution::Continue {
                effective_model: Some(automatic::DIRECTIVE.to_string()),
                classification: None,
            };
        }
        // Single-model mode falls through to the capability-, readiness- and
        // affinity-aware selection below.
        automatic::ServingMode::SingleModel(reason) => tracing::debug!(
            reason = reason.as_str(),
            "automatic routing: serving from a single model"
        ),
    }

    let classification = router::classify(body_json);
    let media = router::media_requirements(body_json);
    let available_models =
        collect_available_models_for_auto_route(node, targets, plugin_manager).await;
    let metrics = node.routing_metrics();
    let available: Vec<router::RoutingCandidate<'_>> = available_models
        .iter()
        .map(|name| {
            let caps = proxy::capabilities_for_model(name, descriptors);
            let (tps_hint, throughput_samples) = metrics
                .tps_for_model(name)
                .map(|(t, s)| (Some(t), s))
                .unwrap_or((None, 0));
            router::RoutingCandidate {
                name: name.as_str(),
                caps,
                parameter_count_b: proxy::descriptor_metadata_for_model(name, descriptors)
                    .and_then(|metadata| metadata.parameter_count_b),
                tps_hint,
                throughput_samples,
            }
        })
        .collect();
    let Some(available) = router::filter_media_compatible_candidates(&available, &media) else {
        proxy::release_request_objects(node, &request.request_object_request_ids).await;
        return AutoRouteResolution::MediaUnsupported;
    };
    let available =
        auto_route_pool_for_ready_models(node, targets, required_tokens, &available, affinity)
            .await;

    let effective_model = router::pick_model_classified(&classification, &available).map(|name| {
        tracing::info!(
            "router: {:?}/{:?} tools={} → {name}",
            classification.category,
            classification.complexity,
            classification.needs_tools
        );
        name.to_string()
    });

    AutoRouteResolution::Continue {
        effective_model,
        classification: Some(classification),
    }
}

async fn auto_route_pool_for_ready_models<'a>(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    required_tokens: Option<u32>,
    available: &[router::RoutingCandidate<'a>],
    affinity: &affinity::AffinityRouter,
) -> Vec<router::RoutingCandidate<'a>> {
    let mut ready_models = Vec::new();
    for candidate in available {
        if auto_route_model_has_ready_ingress_target(
            node,
            targets,
            candidate.name,
            required_tokens,
            affinity,
        )
        .await
        {
            ready_models.push(candidate.name);
        }
    }
    auto_route::pool_for_ready_models(available, &ready_models)
}

async fn auto_route_model_has_ready_ingress_target(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    model: &str,
    required_tokens: Option<u32>,
    affinity: &affinity::AffinityRouter,
) -> bool {
    let local_candidates = targets.candidates(model);
    if contains_routable_candidate(&local_candidates) {
        return auto_route::model_has_eligible_target(
            node,
            model,
            required_tokens,
            &local_candidates,
            affinity,
        )
        .await;
    }

    let remote_candidates = node
        .hosts_for_model(model)
        .await
        .into_iter()
        .map(election::InferenceTarget::Remote)
        .collect::<Vec<_>>();
    if !remote_candidates.is_empty() {
        return auto_route::model_has_eligible_target(
            node,
            model,
            required_tokens,
            &remote_candidates,
            affinity,
        )
        .await;
    }

    // No routable local target and no peer advertises this model. Fail closed:
    // such a model is a phantom (stale gossip, or a peer that unloaded it), and
    // letting it stay in the pool means `auto` can pick a model that then 404s
    // on a model the user never named. Explicit requests still 404 honestly.
    //
    // The one exception is a freshly started serve node: its own model is
    // loaded and in `serving_models` before the target table and gossip catch
    // up, so keep it eligible rather than excluding this node's own model.
    auto_route::model_is_locally_served(node, model).await
}

/// Commit an automatic routing decision to the request that will be forwarded.
///
/// The backend reads `model` out of `request.raw` and rejects anything that is
/// not its own advertised identity (`ensure_requested_model`,
/// `skippy-server/src/frontend/generation/parsing.rs:21-32`), so a selected
/// model that is not written back 404s. This gate previously matched only
/// `None` and `auto`, which left a single-model `mesh` request forwarding
/// `"model":"mesh"` while routing to a concrete target.
///
/// The tokenize guard is load-bearing, not a precaution. Tokenize forces
/// `model_name` to `expected_identity.model_id` (`request_parse.rs:319-330`),
/// and that string could itself be `mesh` or `auto` — which the widened
/// predicate below would match and then rewrite, diverging the wire identity
/// from the authoritative expected identity. `transport.rs` carries the same
/// guard for the same reason.
fn maybe_enable_auto_route_hooks(
    request: &mut proxy::BufferedHttpRequest,
    effective_model: Option<&str>,
) {
    if request.is_tokenize_request() {
        return;
    }
    let is_automatic = request
        .model_name
        .as_deref()
        .is_none_or(automatic::is_directive);
    if !is_automatic {
        return;
    }
    proxy::inject_mesh_hooks_flag(&mut request.raw, true);
    // The directive itself is not a model. Committee mode keeps it in the body
    // for the MoA gateway; only a resolved concrete model gets written back.
    if let Some(model) = effective_model.filter(|name| !automatic::is_directive(name)) {
        proxy::rewrite_model_field(request, model);
    }
}

async fn try_pipeline_proxy(
    node: &mesh::Node,
    tcp_stream: &mut ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    targets: &election::ModelTargets,
    strong_name: &str,
) -> Option<proxy::RouteDispatchOutcome> {
    let (planner_name, planner_port, strong_port) = pipeline_local_ports(targets, strong_name)?;

    request.ensure_body_json();
    let Some(body_json) = request.body_json.clone() else {
        warn_pipeline_fallback(strong_name);
        return None;
    };

    // The pipeline proxy rebuilds the outbound strong-model request from the
    // parsed JSON body rather than forwarding `request.raw` byte-for-byte, so
    // the capsule nonce stabilized at ingress would otherwise be dropped. Read
    // it back off `raw` and thread it onto the strong-model call so the
    // downstream response echoes the same value the client expects.
    let (client_nonce, nonce_origin) = request.capsule_nonce_headers();
    let capsule_nonce = proxy::PipelineCapsuleNonce {
        client_nonce,
        nonce_origin,
    };

    tracing::info!("pipeline: {planner_name} (plan) → {strong_name} (execute)");
    let result = proxy::pipeline_proxy_local(
        tcp_stream,
        &request.path,
        body_json,
        planner_port,
        &planner_name,
        strong_port,
        node,
        &capsule_nonce,
    )
    .await;
    match result {
        proxy::PipelineProxyResult::Responded(status) => {
            Some(proxy::RouteDispatchOutcome::Responded(status))
        }
        proxy::PipelineProxyResult::RespondedWithUsage { status_code, usage } => {
            Some(proxy::RouteDispatchOutcome::RespondedWithUsage { status_code, usage })
        }
        proxy::PipelineProxyResult::Dropped => Some(proxy::RouteDispatchOutcome::Dropped(
            "pipeline_response_write_failed",
        )),
        proxy::PipelineProxyResult::FallbackToDirect => {
            warn_pipeline_fallback(strong_name);
            None
        }
    }
}

fn pipeline_local_ports(
    targets: &election::ModelTargets,
    strong_name: &str,
) -> Option<(String, u16, u16)> {
    let (planner_name, planner_port) = targets
        .targets
        .iter()
        .find(|(name, target_vec)| {
            *name != strong_name
                && target_vec
                    .iter()
                    .any(|target| matches!(target, election::InferenceTarget::Local(_)))
        })
        .and_then(|(name, target_vec)| {
            target_vec.iter().find_map(|target| match target {
                election::InferenceTarget::Local(port) => Some((name.clone(), *port)),
                _ => None,
            })
        })?;
    let strong_port = targets.targets.get(strong_name).and_then(|target_vec| {
        target_vec.iter().find_map(|target| match target {
            election::InferenceTarget::Local(port) => Some(*port),
            _ => None,
        })
    })?;
    Some((planner_name, planner_port, strong_port))
}

fn warn_pipeline_fallback(strong_name: &str) {
    tracing::warn!("pipeline: falling back to direct proxy for {strong_name}");
}

async fn route_missing_local_model(
    tcp_stream: ClientStream,
    request: &proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
    required_tokens: Option<u32>,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    // Try remote mesh first.
    if let Some(mesh_targets) = remote_mesh_targets(ctx, model_name).await {
        return proxy::route_model_request(
            ctx.node.clone(),
            tcp_stream,
            &mesh_targets,
            model_name,
            request,
            proxy::RouteModelRequestContext {
                required_tokens,
                affinity: ctx.affinity,
                route_observer,
            },
        )
        .await;
    }

    // Check if the model is known locally but unavailable
    // (e.g., loading/draining/failed with all-None candidates). Return 503 for these cases so
    // clients can retry; return 404 only when the model truly doesn't exist anywhere.
    if has_local_unavailable_candidates(ctx.targets, model_name) {
        return response_outcome(
            503,
            proxy::send_503_observed(
                tcp_stream,
                &format!("model '{model_name}' is unavailable locally (loading or draining)"),
                route_observer,
            )
            .await,
        );
    }

    // Try plugin dispatch (admission-checked inside).
    if ctx.plugin_manager.is_some() {
        return try_route_plugin_model(ctx, tcp_stream, request, model_name, route_observer).await;
    }

    // Model not found anywhere — return 404. This replaces the old fallback behavior that would
    // route to an arbitrary target, which was confusing when no host served this model.
    response_outcome(
        404,
        proxy::send_error_observed(
            tcp_stream,
            404,
            &format!("model '{model_name}' not found (no local or remote host serving this model)"),
            route_observer,
        )
        .await,
    )
}

/// Check whether the model is known locally but currently unavailable — all local candidates are None.
fn has_local_unavailable_candidates(targets: &election::ModelTargets, model_name: &str) -> bool {
    let cands = targets.candidates(model_name);
    !cands.is_empty()
        && cands
            .iter()
            .all(|t| matches!(t, election::InferenceTarget::None))
}

async fn remote_mesh_targets(
    ctx: &IngressRouteContext<'_>,
    model_name: &str,
) -> Option<election::ModelTargets> {
    let remote_hosts = ctx.node.hosts_for_model(model_name).await;
    if remote_hosts.is_empty() {
        return None;
    }
    let mut mesh_targets = ctx.targets.clone();
    mesh_targets.targets.insert(
        model_name.to_string(),
        remote_hosts
            .into_iter()
            .map(election::InferenceTarget::Remote)
            .collect(),
    );
    Some(mesh_targets)
}

async fn try_route_plugin_model(
    ctx: &IngressRouteContext<'_>,
    mut tcp_stream: ClientStream,
    request: &proxy::BufferedHttpRequest,
    model_name: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    // Admission check for plugin dispatch ingress.
    match check_activity_admission(
        tcp_stream,
        &ctx.node.activity_policy_guard,
        crate::runtime::IngressType::PluginDispatch,
        route_observer,
    )
    .await
    {
        Ok(stream) => tcp_stream = stream,
        Err(outcome) => {
            route_observer.route_selected_with_metadata(
                Some(model_name),
                Some("plugin"),
                Some("admission"),
            );
            return outcome;
        }
    }

    let plugin_manager = ctx
        .plugin_manager
        .expect("plugin route called without plugin manager");
    match plugin_manager
        .inference_endpoint_for_model(model_name)
        .await
    {
        Ok(Some(endpoint)) => {
            // Path 2's own "effective request" moment: the plugin/endpoint
            // is resolved and dispatch is about to happen. There is no typed
            // `ChatCompletionRequest` on this path (see the #1331 design
            // note), so the envelope carries only the model — the same
            // narrow route fact path 1's `ChatExchangeRoute` carries. Mint
            // the exchange id here, at admission, so it can pair this
            // effective event with its terminal event below even when
            // concurrent raw-proxy requests share the same model.
            let exchange_id = uuid::Uuid::new_v4().to_string();
            plugin_manager
                .publish(&OpenAiExchangeEnvelope::effective(
                    exchange_id.clone(),
                    OpenAiExchangeDispatchPath::RawProxy,
                    model_name,
                ))
                .await;
            let outcome = proxy::route_http_endpoint_request(
                ctx.node,
                Some(model_name),
                proxy::RouteSelectionMetadata {
                    provider: Some(&endpoint.plugin_name),
                    engine: Some(&endpoint.endpoint_id),
                },
                &mut tcp_stream,
                &endpoint.address,
                request,
                route_observer,
            )
            .await;
            let final_outcome = if !outcome.response_written()
                && !matches!(outcome, proxy::RouteDispatchOutcome::Dropped(_))
            {
                response_outcome(
                    503,
                    proxy::send_503_observed(
                        tcp_stream,
                        &format!("plugin endpoint for model '{model_name}' failed"),
                        route_observer,
                    )
                    .await,
                )
            } else {
                outcome
            };
            plugin_manager
                .publish(&OpenAiExchangeEnvelope::terminal(
                    exchange_id,
                    OpenAiExchangeDispatchPath::RawProxy,
                    model_name,
                    plugin_route_status(&final_outcome),
                    // No X-Capsule-Id marker on this path: it never runs
                    // through `openai-frontend`'s `OpenAiHookPolicy`, the
                    // only place a marker is minted (see the design note).
                    None,
                    // No marker means no nonce, so no nonce_source either.
                    None,
                ))
                .await;
            final_outcome
        }
        Ok(None) => {
            route_observer.route_selected_with_metadata(
                Some(model_name),
                Some("plugin"),
                Some("inference_endpoint"),
            );
            // Plugin manager exists but doesn't serve this model — send 404.
            response_outcome(
                404,
                proxy::send_error_observed(
                    tcp_stream,
                    404,
                    &format!(
                        "model '{model_name}' not found (no local or remote host serving this model)"
                    ),
                    route_observer,
                )
                .await,
            )
        }
        Err(error) => {
            tracing::warn!(
                "API proxy: failed to resolve external endpoint for model '{}': {}",
                model_name,
                error
            );
            route_observer.route_selected_with_metadata(
                Some(model_name),
                Some("plugin"),
                Some("inference_endpoint"),
            );
            // Plugin resolution failure — degrade gracefully with 503. The daemon stays alive; only the affected capability is degraded.
            response_outcome(
                503,
                proxy::send_503_observed(
                    tcp_stream,
                    &format!("plugin endpoint for model '{model_name}' unavailable"),
                    route_observer,
                )
                .await,
            )
        }
    }
}

async fn route_request(
    tcp_stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    effective_model: Option<&str>,
    required_tokens: Option<u32>,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    prepare_cache_routing_body(request, effective_model);
    if let Some(model_name) = effective_model {
        // Model explicitly requested. Check local candidates first.
        if !has_available_candidates(ctx.targets, model_name) {
            return route_missing_local_model(
                tcp_stream,
                request,
                ctx,
                model_name,
                required_tokens,
                route_observer,
            )
            .await;
        }

        // Local candidates available — route normally.
        proxy::route_model_request(
            ctx.node.clone(),
            tcp_stream,
            ctx.targets,
            model_name,
            request,
            proxy::RouteModelRequestContext {
                required_tokens,
                affinity: ctx.affinity,
                route_observer,
            },
        )
        .await
    } else {
        // No model specified — generic fallback routing to first available target.

        proxy::route_to_target(
            ctx.node.clone(),
            tcp_stream,
            None,
            first_available_target(ctx.targets),
            &request.raw,
            proxy::RouteTargetContext {
                request_id: request.request_id,
                response_adapter: request.response_adapter,
                route_observer,
            },
        )
        .await
    }
}

fn prepare_cache_routing_body(
    request: &mut proxy::BufferedHttpRequest,
    effective_model: Option<&str>,
) {
    // Cache routing and provider-confirmed local receipts need the same
    // prefix key even when this node currently has only one eligible target.
    // The body is already bounded and buffered at ingress; parsing here does
    // not change the forwarded bytes.
    if effective_model.is_some() && !request.is_tokenize_request() {
        request.ensure_body_json();
    }
}

async fn prepare_auto_route_decision(
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    descriptors: &[crate::mesh::ServedModelDescriptor],
) -> Result<AutoRouteDecision, ()> {
    let required_tokens = proxy::request_context_budget(request);
    match resolve_auto_routed_model(
        ctx.node,
        request,
        ctx.targets,
        ctx.plugin_manager,
        descriptors,
        required_tokens,
        ctx.affinity,
    )
    .await
    {
        AutoRouteResolution::Continue {
            effective_model,
            classification,
        } => {
            maybe_enable_auto_route_hooks(request, effective_model.as_deref());
            if let Some(name) = effective_model.as_ref() {
                ctx.node.record_request(name);
            }
            Ok(AutoRouteDecision {
                effective_model,
                classification,
                required_tokens,
            })
        }
        AutoRouteResolution::MediaUnsupported => Err(()),
    }
}

async fn send_media_unsupported(
    tcp_stream: ClientStream,
    route_observer: OpenAiRouteObserver<'_>,
) -> proxy::RouteDispatchOutcome {
    response_outcome(
        422,
        proxy::send_error_observed(
            tcp_stream,
            422,
            "no served model can satisfy the requested media inputs",
            route_observer,
        )
        .await,
    )
}

fn callable_models_with_local_served(
    targets: &election::ModelTargets,
    local_models: Vec<String>,
) -> Vec<String> {
    let mut callable = callable_models(targets);
    for name in local_models {
        if !callable.iter().any(|existing| existing == &name) {
            callable.push(name);
        }
    }
    callable.sort();
    callable
}

async fn maybe_handle_control_request(
    tcp_stream: ClientStream,
    request: &proxy::BufferedHttpRequest,
    ctx: &ProxyConnectionContext<'_>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<proxy::RouteDispatchOutcome, ClientStream> {
    if proxy::is_legacy_lifecycle_path(&request.path) {
        return Ok(proxy::reject_legacy_lifecycle_request(tcp_stream, route_observer).await);
    }

    if proxy::is_models_list_request(&request.method, &request.path) {
        let outcome = handle_models_list_request(
            tcp_stream,
            ctx.route.node,
            ctx.route.targets,
            ctx.route.plugin_manager,
        )
        .await;
        return Ok(outcome);
    }

    Err(tcp_stream)
}

fn pipeline_route_model<'a>(
    request: &proxy::BufferedHttpRequest,
    decision: &AutoRouteDecision,
    routing_model: Option<&'a str>,
) -> Option<&'a str> {
    let use_pipeline = decision
        .classification
        .as_ref()
        .map(pipeline::should_pipeline)
        .unwrap_or(false)
        && request.response_adapter == proxy::ResponseAdapter::None;
    use_pipeline.then_some(routing_model).flatten()
}

async fn try_pipeline_route(
    tcp_stream: &mut ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &IngressRouteContext<'_>,
    decision: &AutoRouteDecision,
    routing_model: Option<&str>,
) -> Option<proxy::RouteDispatchOutcome> {
    let strong_name = pipeline_route_model(request, decision, routing_model)?;
    try_pipeline_proxy(ctx.node, tcp_stream, request, ctx.targets, strong_name).await
}

enum MoaInterceptResult {
    /// MoA handled the request; the response has been written and the stream
    /// is consumed.
    Handled(proxy::RouteDispatchOutcome),
    /// Not an MoA request — caller should continue with normal routing,
    /// reusing the returned stream.
    NotMoa(ClientStream),
    /// MoA could not form a committee but degraded `model=mesh` to a real
    /// single model (already rewritten on the request). Caller routes it
    /// normally, but must use this model rather than the stale
    /// `decision.effective_model` (still "mesh").
    Degraded {
        stream: ClientStream,
        model: Option<String>,
    },
}

/// Dispatch to the MoA gateway when `model == "mesh"`. Self-gates on the
/// effective model so the call site is unconditional.
async fn try_handle_moa_intercept(
    tcp_stream: ClientStream,
    request: &mut proxy::BufferedHttpRequest,
    ctx: &ProxyConnectionContext<'_>,
    decision: &AutoRouteDecision,
    route_observer: OpenAiRouteObserver<'_>,
) -> MoaInterceptResult {
    if decision.effective_model.as_deref() != Some(moa::VIRTUAL_MODEL_NAME) {
        return MoaInterceptResult::NotMoa(tcp_stream);
    }
    // `try_handle_moa` self-gates on the model name and consumes the
    // stream when it accepts. The outer gate above guarantees the gate
    // matches, so the inner call always returns `None` here — the stream
    // is gone, either with the MoA response, a 503, or a 400. Discard
    // the return value explicitly. The previous shape kept an
    // `if let Some(_) = … { tracing::error!(...) }` branch that could
    // never fire and made the control flow confusing to read.
    let result = crate::network::openai::moa_gateway::try_handle_moa(
        ctx.route.node,
        tcp_stream,
        request,
        decision.effective_model.as_deref(),
        super::moa_gateway::MoaRoutingContext {
            targets: Some(ctx.route.targets),
            required_tokens: decision.required_tokens,
            affinity: ctx.route.affinity,
        },
        route_observer,
    )
    .await;
    match result {
        crate::network::openai::moa_gateway::MoaDispatchResult::Passthrough(stream) => {
            // The gateway hands the stream back in two cases: the request was
            // never MoA-shaped, or MoA degraded `model=mesh` to a real single
            // model by rewriting the request in place. The outer gate above
            // guarantees we got here with `effective_model == "mesh"`, so this
            // is the degrade case: routing must use the rewritten model, not
            // the stale decision.
            MoaInterceptResult::Degraded {
                stream,
                model: request.model_name.clone(),
            }
        }
        crate::network::openai::moa_gateway::MoaDispatchResult::Responded(status) => {
            proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids)
                .await;
            MoaInterceptResult::Handled(proxy::RouteDispatchOutcome::Responded(status))
        }
        crate::network::openai::moa_gateway::MoaDispatchResult::RespondedWithUsage {
            status_code,
            usage,
        } => {
            proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids)
                .await;
            MoaInterceptResult::Handled(proxy::RouteDispatchOutcome::RespondedWithUsage {
                status_code,
                usage,
            })
        }
        crate::network::openai::moa_gateway::MoaDispatchResult::FailedWithStatus {
            status_code,
            reason,
        } => {
            proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids)
                .await;
            MoaInterceptResult::Handled(proxy::RouteDispatchOutcome::FailedWithStatus {
                status_code,
                reason,
            })
        }
        crate::network::openai::moa_gateway::MoaDispatchResult::Dropped(reason) => {
            proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids)
                .await;
            MoaInterceptResult::Handled(proxy::RouteDispatchOutcome::Dropped(reason))
        }
    }
}

async fn handle_buffered_api_request(
    tcp_stream: ClientStream,
    mut request: proxy::BufferedHttpRequest,
    ctx: ProxyConnectionContext<'_>,
    source_addr: Option<std::net::SocketAddr>,
    ingress_type: crate::runtime::IngressType,
) {
    // Claim the parent at host OpenAI ingress. All downstream dispatch sees
    // only a metadata observer; this scope remains the sole terminal owner.
    let caller_addr = source_addr.map(|addr| addr.to_string());
    let request_metadata =
        crate::logging::RequestSummaryMetadata::from_openai_ingress_path(&request.client_path)
            .with_source(Some(
                if ingress_type == crate::runtime::IngressType::RemoteQuicHttp {
                    "remote_quic_http"
                } else {
                    "direct_http"
                },
            ))
            .with_method(Some(&request.method))
            .with_caller_identity(
                None,
                caller_addr.as_deref(),
                caller_addr.as_ref().map(|_| CallerPathType::LocalHttp),
            );
    let mut lifecycle = crate::logging_runtime_state()
        .map(|state| state.openai_ingress_attachment(request.request_id, request_metadata))
        .unwrap_or_else(OpenAiLifecycleAttachment::unowned);
    if lifecycle.owns_parent() {
        request.mark_raw_lifecycle_owned();
        if let Some(body) = request.body_bytes.as_deref() {
            lifecycle.capture_request_body(body, request.artifact_request_media_kind());
        }
    }

    let tcp_stream =
        match maybe_handle_control_request(tcp_stream, &request, &ctx, lifecycle.route_observer())
            .await
        {
            Ok(outcome) => {
                lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
                return;
            }
            Err(tcp_stream) => tcp_stream,
        };

    let local_models = ctx.route.node.models_being_served().await;
    let callable = callable_models_with_local_served(ctx.route.targets, local_models);
    let descriptors = ctx.route.node.all_served_model_descriptors().await;
    proxy::rewrite_public_model_alias(&mut request, &callable, &descriptors);

    // Admission applies to inference work after control-path rejection.
    let tcp_stream = match check_activity_admission(
        tcp_stream,
        &ctx.route.node.activity_policy_guard,
        ingress_type,
        lifecycle.route_observer(),
    )
    .await
    {
        Ok(stream) => stream,
        Err(outcome) => {
            lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
            return;
        }
    };

    let decision = match prepare_auto_route_decision(&mut request, &ctx.route, &descriptors).await {
        Ok(decision) => decision,
        Err(()) => {
            let outcome = send_media_unsupported(tcp_stream, lifecycle.route_observer()).await;
            lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
            return;
        }
    };

    let mut routing_model = decision.effective_model.clone();
    let tcp_stream = match try_handle_moa_intercept(
        tcp_stream,
        &mut request,
        &ctx,
        &decision,
        lifecycle.route_observer(),
    )
    .await
    {
        MoaInterceptResult::Handled(outcome) => {
            proxy::record_moa_stream_lifecycle(
                lifecycle.route_observer(),
                request.response_adapter,
                outcome,
            );
            lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
            return;
        }
        MoaInterceptResult::NotMoa(stream) => stream,
        MoaInterceptResult::Degraded { stream, model } => {
            routing_model = model;
            stream
        }
    };

    let mut tcp_stream = tcp_stream;
    if let Some(outcome) = try_pipeline_route(
        &mut tcp_stream,
        &mut request,
        &ctx.route,
        &decision,
        routing_model.as_deref(),
    )
    .await
    {
        proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids).await;
        lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
        return;
    }

    let outcome = {
        let route_observer = lifecycle.route_observer();
        route_request(
            tcp_stream,
            &mut request,
            &ctx.route,
            routing_model.as_deref(),
            decision.required_tokens,
            route_observer,
        )
        .await
    };
    if let Some(model) = routing_model.as_deref()
        && !request.is_tokenize_request()
    {
        let mut event =
            audit_events::model_access(None, model, "route", model_access_succeeded(outcome));
        if let Some(cid) = request.correlation_id.as_deref() {
            event = event.with_metadata("request_id", serde_json::Value::String(cid.to_string()));
        }
        let _ = emit_audit(event);
    }
    proxy::release_request_objects(ctx.route.node, &request.request_object_request_ids).await;
    lifecycle.terminal(terminal_outcome_for_dispatch(outcome));
}

async fn handle_api_proxy_connection(
    node: mesh::Node,
    mut tcp_stream: ClientStream,
    targets: election::ModelTargets,
    affinity: affinity::AffinityRouter,
    ingress_type: crate::runtime::IngressType,
) {
    let source_addr = tcp_stream.peer_addr().ok();
    let plugin_manager = node.plugin_manager().await;
    match proxy::read_http_request_with_plugin_manager_with_context(
        &mut tcp_stream,
        plugin_manager.as_ref(),
    )
    .await
    {
        Ok(request) => {
            let route = IngressRouteContext {
                node: &node,
                targets: &targets,
                affinity: &affinity,
                plugin_manager: plugin_manager.as_ref(),
            };
            handle_buffered_api_request(
                tcp_stream,
                request,
                ProxyConnectionContext { route },
                source_addr,
                ingress_type,
            )
            .await;
        }
        Err(error) => {
            let _ = super::parse_failure::send_read_failure(tcp_stream, &error).await;
        }
    }
}

pub(crate) async fn handle_remote_http_stream(
    node: mesh::Node,
    stream: ClientStream,
    targets: election::ModelTargets,
    affinity: affinity::AffinityRouter,
) {
    handle_api_proxy_connection(
        node,
        stream,
        targets,
        affinity,
        crate::runtime::IngressType::RemoteQuicHttp,
    )
    .await;
}

/// Model-aware API proxy. Parses the "model" field from POST request bodies
/// and routes to the correct host. Falls back to the first available target
/// if model is not specified or not found.
pub(crate) async fn api_proxy(
    node: mesh::Node,
    port: u16,
    target_rx: tokio::sync::watch::Receiver<election::ModelTargets>,
    existing_listener: Option<tokio::net::TcpListener>,
    listen_all: bool,
    affinity: affinity::AffinityRouter,
) {
    let Some(listener) = bind_api_proxy_listener(port, existing_listener, listen_all).await else {
        return;
    };

    loop {
        let (tcp_stream, _addr) = match listener.accept().await {
            Ok(r) => r,
            Err(_) => break,
        };
        let _ = tcp_stream.set_nodelay(true);

        let targets = target_rx.borrow().clone();
        let node = node.clone();
        let affinity = affinity.clone();
        tokio::spawn(async move {
            handle_api_proxy_connection(
                node,
                tcp_stream.into(),
                targets,
                affinity,
                crate::runtime::IngressType::LocalOpenAi,
            )
            .await;
        });
    }
}

/// Bootstrap proxy: runs during GPU startup, tunnels all requests to mesh hosts.
/// Returns the TcpListener when signaled to stop (so api_proxy can take it over).
pub(crate) async fn bootstrap_proxy(
    node: mesh::Node,
    port: u16,
    mut stop_rx: tokio::sync::mpsc::Receiver<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>,
    listen_all: bool,
    affinity: affinity::AffinityRouter,
) {
    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Bootstrap proxy: failed to bind to port {port}: {e}");
            return;
        }
    };
    let _ = emit_event(OutputEvent::Info {
        message: format!("API ready (bootstrap): http://localhost:{port}"),
        context: Some("bootstrap_proxy".to_string()),
    });
    let _ = emit_event(OutputEvent::Info {
        message: "Requests tunneled to mesh while GPU loads...".to_string(),
        context: Some("bootstrap_proxy".to_string()),
    });

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (tcp_stream, _addr) = match accept {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let _ = tcp_stream.set_nodelay(true);
                let node = node.clone();
                let affinity = affinity.clone();
                tokio::spawn(Box::pin(proxy::handle_mesh_request(node, tcp_stream.into(), true, affinity)));
            }
            resp_tx = stop_rx.recv() => {
                if let Some(tx) = resp_tx {
                    let _ = emit_event(OutputEvent::Info {
                        message: "Bootstrap proxy handing off to full API proxy".to_string(),
                        context: Some("bootstrap_proxy".to_string()),
                    });
                    let _ = tx.send(listener);
                }
                return;
            }
        }
    }
}

fn first_available_target(targets: &election::ModelTargets) -> election::InferenceTarget {
    for hosts in targets.targets.values() {
        for target in hosts {
            if !matches!(target, election::InferenceTarget::None) {
                return target.clone();
            }
        }
    }
    election::InferenceTarget::None
}

fn has_available_candidates(targets: &election::ModelTargets, model: &str) -> bool {
    contains_routable_candidate(&targets.candidates(model))
}

fn contains_routable_candidate(candidates: &[election::InferenceTarget]) -> bool {
    candidates
        .iter()
        .any(|target| !matches!(target, election::InferenceTarget::None))
}

pub(crate) fn callable_models(targets: &election::ModelTargets) -> Vec<String> {
    let mut models: Vec<String> = targets
        .targets
        .iter()
        .filter(|(_, hosts)| {
            hosts
                .iter()
                .any(|target| !matches!(target, election::InferenceTarget::None))
        })
        .map(|(name, _)| name.clone())
        .collect();
    models.sort();
    models
}

#[cfg(test)]
#[path = "ingress_tests/durable_artifacts.rs"]
mod durable_artifacts;

#[cfg(test)]
#[path = "ingress_tests/automatic_routing.rs"]
mod automatic_routing;

#[cfg(test)]
#[path = "ingress_tests/tests.rs"]
mod tests;
