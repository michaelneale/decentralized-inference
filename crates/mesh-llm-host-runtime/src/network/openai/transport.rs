//! HTTP proxy plumbing — request parsing, model routing, response helpers.
//!
//! Used by the API proxy (port 9337), bootstrap proxy, and passive mode.
//! All inference traffic flows through these functions.

use crate::inference::election;
use crate::logging::{
    CallerPathType, OpenAiLifecycleAttachment, OpenAiRouteAttempt, OpenAiRouteObserver,
    ProxyAttemptFinish,
};
use crate::mesh;
use crate::network::affinity::{
    AffinityRouter, PreparedTargets, TargetSelection, prepare_remote_targets_for_request,
};
use crate::network::openai::auto_route;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::response_quality::ResponseQualityFailure;
use crate::network::router;
use std::time::{Duration, Instant};

pub use super::request_normalize::{ResponseAdapter, release_request_objects};
pub(crate) use super::request_parse::read_http_request_with_plugin_manager_with_context;
pub use super::request_parse::{
    BufferedHttpRequest, inject_mesh_hooks_flag, is_legacy_lifecycle_path, is_models_list_request,
    read_http_request, rewrite_model_field, rewrite_public_model_alias,
};
pub(crate) use super::response::{
    PipelineCapsuleNonce, PipelineProxyResult, append_safe_header, pipeline_proxy_local,
    send_400_observed, send_503_observed, send_error_observed, send_json_ok_with_headers,
    send_json_with_status_and_headers_observed, send_models_list_with_descriptors,
};
pub(crate) use super::routing_rank::{
    capabilities_for_model, descriptor_metadata_for_model, request_budget_tokens_from_parts,
};

use super::response::{
    CacheCostObservation, ResponseRetryPolicy, RouteAttemptLoggingContext, RouteAttemptResult,
    attempt_outcome_for_result, completion_tokens_for_result, request_outcome_for_status,
    request_service_for_target, route_attempt_result_label, route_http_endpoint_attempt,
    route_local_attempt, route_remote_attempt, target_health_outcome_for_attempt,
};
use super::routing_rank::{
    cached_auto_model_satisfies_media_requirements, move_target_first,
    order_remote_hosts_by_context, order_targets_by_context,
};
use mesh_llm_events::logging::events::TokenUsage;
use mesh_llm_events::logging::identifiers::RequestId;

const REMOTE_UNCOMMITTED_RETRIES: usize = 1;

#[path = "transport_route_model.rs"]
mod route_model;
pub(crate) use route_model::RouteModelRequestContext;
#[cfg(test)]
pub(crate) use route_model::finalize_route_model_result;
pub use route_model::route_model_request;

/// Response result returned to the ingress boundary. Unlike the historical
/// boolean, this preserves the downstream HTTP status and distinguishes a
/// transport failure from a client disconnect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RouteDispatchOutcome {
    Responded(u16),
    RespondedWithUsage {
        status_code: u16,
        usage: TokenUsage,
    },
    Failed(&'static str),
    FailedWithStatus {
        status_code: u16,
        reason: &'static str,
    },
    Dropped(&'static str),
}

pub(super) fn record_moa_stream_lifecycle(
    observer: OpenAiRouteObserver<'_>,
    adapter: ResponseAdapter,
    outcome: RouteDispatchOutcome,
) {
    if !matches!(
        adapter,
        ResponseAdapter::OpenAiChatCompletionsStream | ResponseAdapter::OpenAiResponsesStream
    ) {
        return;
    }
    observer.stream_started(Some(mesh_mixture_of_agents::VIRTUAL_MODEL_NAME));
    match outcome {
        RouteDispatchOutcome::RespondedWithUsage {
            status_code: 200..=299,
            usage,
        } => observer.stream_completed(Some(usage)),
        RouteDispatchOutcome::Responded(200..=299) => observer.stream_completed(None),
        RouteDispatchOutcome::Failed(_)
        | RouteDispatchOutcome::FailedWithStatus { .. }
        | RouteDispatchOutcome::Responded(_)
        | RouteDispatchOutcome::RespondedWithUsage { .. }
        | RouteDispatchOutcome::Dropped(_) => observer.stream_error("moa_stream_failed"),
    }
}

impl RouteDispatchOutcome {
    pub(crate) const fn response_written(self) -> bool {
        matches!(self, Self::Responded(_) | Self::RespondedWithUsage { .. })
    }

    pub(crate) fn terminal_outcome(self) -> crate::logging::TerminalOutcome {
        match self {
            Self::Responded(status @ 200..=299) => {
                crate::logging::TerminalOutcome::CompletedWithStatus(status)
            }
            Self::RespondedWithUsage { status_code, usage } => match status_code {
                200..=299 => {
                    crate::logging::TerminalOutcome::CompletedWithUsage { status_code, usage }
                }
                400..=499 => crate::logging::TerminalOutcome::RejectedWithStatus {
                    reason: Some(format!("http_status_{status_code}")),
                    status_code,
                },
                _ => crate::logging::TerminalOutcome::FailedWithStatus {
                    error: format!("http_status_{status_code}"),
                    status_code,
                },
            },
            Self::Responded(status @ 400..=499) => {
                crate::logging::TerminalOutcome::RejectedWithStatus {
                    reason: Some(format!("http_status_{status}")),
                    status_code: status,
                }
            }
            Self::Responded(status) => crate::logging::TerminalOutcome::FailedWithStatus {
                error: format!("http_status_{status}"),
                status_code: status,
            },
            Self::Failed(reason) => crate::logging::TerminalOutcome::Failed(reason.into()),
            Self::FailedWithStatus {
                status_code,
                reason,
            } => crate::logging::TerminalOutcome::FailedWithStatus {
                error: reason.into(),
                status_code,
            },
            Self::Dropped(reason) => crate::logging::TerminalOutcome::Dropped(Some(reason.into())),
        }
    }
}

fn response_outcome(status_code: u16, result: std::io::Result<()>) -> RouteDispatchOutcome {
    match result {
        Ok(()) => RouteDispatchOutcome::Responded(status_code),
        Err(_) => RouteDispatchOutcome::Dropped("response_write_failed"),
    }
}

const LEGACY_LIFECYCLE_ROUTE_MESSAGE: &str =
    "model lifecycle routes moved to the trusted local management API at :3131/api/runtime/models";

pub(crate) async fn reject_legacy_lifecycle_request(
    tcp_stream: ClientStream,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteDispatchOutcome {
    response_outcome(
        410,
        send_error_observed(
            tcp_stream,
            410,
            LEGACY_LIFECYCLE_ROUTE_MESSAGE,
            route_observer,
        )
        .await,
    )
}

/// Generation context is a property of decode requests, not capability RPCs.
/// A tokenizer request may carry a megabyte of source text while using no
/// target KV context at all.
pub(crate) fn request_context_budget(request: &BufferedHttpRequest) -> Option<u32> {
    if request.is_tokenize_request() {
        None
    } else {
        request_budget_tokens_from_parts(request.body_len_bytes, request.completion_tokens)
    }
}

enum AutoModelResolution {
    Model(Option<String>),
    UnsupportedMedia,
}

enum MeshTargetResolution {
    Hosts(Vec<iroh::EndpointId>),
    ModelUnavailable(String),
    NoHostsAvailable,
}

struct MeshRequestPlan {
    effective_model: Option<String>,
    auto_session_key: Option<u64>,
    target_hosts: Vec<iroh::EndpointId>,
    affinity_applied: bool,
}

enum MeshRequestFailure {
    UnsupportedMedia,
    ModelUnavailable(String),
    NoHostsAvailable,
}

struct MeshAttemptState {
    route_started: Instant,
    attempts: usize,
    last_retryable: bool,
    refreshed: bool,
}

enum MeshAttemptDisposition {
    Continue,
    Return(RouteAttemptResult),
}

enum MeshRouteResult {
    Exhausted(ClientStream),
    Finished(RouteAttemptResult),
}

#[derive(Clone, Copy, Default)]
pub(crate) struct RouteSelectionMetadata<'a> {
    pub(crate) provider: Option<&'a str>,
    pub(crate) engine: Option<&'a str>,
}

fn capture_path_for_request(request: &BufferedHttpRequest) -> &str {
    &request.client_path
}

fn attach_request_logging(
    request: &mut BufferedHttpRequest,
    source_addr: Option<std::net::SocketAddr>,
) -> OpenAiLifecycleAttachment {
    let caller_addr = source_addr.map(|addr| addr.to_string());
    let metadata =
        crate::logging::RequestSummaryMetadata::from_openai_ingress_path(&request.client_path)
            .with_source(Some("mesh_forwarded"))
            .with_method(Some(&request.method))
            .with_caller_identity(
                None,
                caller_addr.as_deref(),
                caller_addr.as_ref().map(|_| CallerPathType::LocalHttp),
            );
    let lifecycle = crate::logging_runtime_state()
        .map(|state| state.openai_ingress_attachment(request.request_id, metadata))
        .unwrap_or_else(OpenAiLifecycleAttachment::unowned);
    if lifecycle.owns_parent() {
        request.mark_raw_lifecycle_owned();
        if let Some(body) = request.body_bytes.as_deref() {
            lifecycle.capture_request_body(body, request.artifact_request_media_kind());
        }
    }
    lifecycle
}

async fn handle_mesh_control_request(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    request: &BufferedHttpRequest,
    lifecycle: &mut OpenAiLifecycleAttachment,
) -> Option<ClientStream> {
    if is_legacy_lifecycle_path(&request.path) {
        let outcome = reject_legacy_lifecycle_request(tcp_stream, lifecycle.route_observer()).await;
        lifecycle.terminal(outcome.terminal_outcome());
        release_request_objects(node, &request.request_object_request_ids).await;
        return None;
    }

    if is_models_list_request(&request.method, &request.path) {
        let served = node.models_being_served().await;
        let descriptors = node.all_served_model_descriptors().await;
        let runtimes = node.all_model_runtime_descriptors().await;
        let outcome = response_outcome(
            200,
            send_models_list_with_descriptors(tcp_stream, &served, &descriptors, &runtimes).await,
        );
        lifecycle.terminal(outcome.terminal_outcome());
        return None;
    }

    Some(tcp_stream)
}

// ── Model-aware tunnel routing ──

/// The common request-handling path used by idle proxy, passive proxy, and bootstrap proxy.
///
/// Peeks at the HTTP request, handles `/v1/models`, resolves the target host
/// by model name (or falls back to any host), and tunnels the request via QUIC.
///
/// Set `track_demand` to record requests for demand-based rebalancing.
pub async fn handle_mesh_request(
    node: mesh::Node,
    tcp_stream: ClientStream,
    track_demand: bool,
    affinity: AffinityRouter,
) {
    let mut tcp_stream = tcp_stream;
    let source_addr = tcp_stream.peer_addr().ok();
    let plugin_manager = node.plugin_manager().await;
    let mut request = match read_http_request_with_plugin_manager_with_context(
        &mut tcp_stream,
        plugin_manager.as_ref(),
    )
    .await
    {
        Ok(v) => v,
        Err(err) => {
            let _ = super::parse_failure::send_read_failure(tcp_stream, &err).await;
            return;
        }
    };
    // The parsed host ingress owns the parent. Downstream route code receives
    // only the attachment's metadata observer and cannot terminalize it.
    let mut lifecycle = attach_request_logging(&mut request, source_addr);
    if node.swarm_capture_enabled() {
        node.capture_http_request(crate::mesh::HttpCaptureEvent {
            event: "openai_ingress_http_request",
            source_addr,
            method: &request.method,
            path: capture_path_for_request(&request),
            body_len_bytes: request.body_len_bytes,
            model_name: request.model_name.as_deref(),
            completion_tokens: request.completion_tokens,
            stream: request.stream,
        });
    }

    let Some(tcp_stream) =
        handle_mesh_control_request(&node, tcp_stream, &request, &mut lifecycle).await
    else {
        return;
    };

    // MoA routing directive: `model: "mesh"` triggers mixture-of-agents
    // fan-out. Orchestration happens here, regardless of whether this node
    // is serving models locally — the worker pool is built from gossip.
    // On a pure --client node every backend is remote (QUIC tunnels to
    // peers serving each model); on a host node the locally-served model
    // is wired directly to its skippy port via the targets table.
    //
    // Tokenization is a direct capability RPC, never an MoA generation. Other
    // requests retain the normal self-gating MoA path.
    let tcp_stream = match route_mesh_moa_or_passthrough(
        &node,
        tcp_stream,
        &mut request,
        lifecycle.route_observer(),
    )
    .await
    {
        Ok(stream) => stream,
        Err(outcome) => {
            // MoA handled the request and consumed the stream.
            lifecycle.terminal(outcome.terminal_outcome());
            release_request_objects(&node, &request.request_object_request_ids).await;
            return;
        }
    };

    let plan = match build_mesh_request_plan(&node, &mut request, track_demand, &affinity).await {
        Ok(plan) => plan,
        Err(failure) => {
            let outcome = terminal_outcome_for_mesh_request_failure(&failure);
            handle_mesh_request_failure(
                &node,
                tcp_stream,
                &request,
                failure,
                lifecycle.route_observer(),
            )
            .await;
            lifecycle.terminal(outcome);
            return;
        }
    };
    let outcome = {
        let route_observer = lifecycle.route_observer();
        route_observer.route_selected(plan.effective_model.as_deref());
        match route_mesh_request_attempts(
            &node,
            tcp_stream,
            &request,
            &plan,
            &affinity,
            route_observer,
        )
        .await
        {
            MeshRouteResult::Exhausted(tcp_stream) => {
                finish_exhausted_mesh_request(
                    &node,
                    tcp_stream,
                    plan.effective_model.as_deref(),
                    plan.target_hosts.len(),
                    &affinity,
                    route_observer,
                )
                .await;
                crate::logging::TerminalOutcome::Failed("mesh_route_exhausted".into())
            }
            MeshRouteResult::Finished(result) => terminal_outcome_for_mesh_route_result(result),
        }
    };
    lifecycle.terminal(outcome);
    release_request_objects(&node, &request.request_object_request_ids).await;
}

async fn route_mesh_moa_or_passthrough(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    request: &mut BufferedHttpRequest,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<ClientStream, RouteDispatchOutcome> {
    if request.is_tokenize_request() {
        return Ok(tcp_stream);
    }
    let moa_model_name = request.model_name.clone();
    let moa_required_tokens = request_context_budget(request);
    let adapter = request.response_adapter;
    let result = match crate::network::openai::moa_gateway::try_handle_moa(
        node,
        tcp_stream,
        request,
        moa_model_name.as_deref(),
        None, // passive path has no local targets table
        moa_required_tokens,
        route_observer,
    )
    .await
    {
        crate::network::openai::moa_gateway::MoaDispatchResult::Passthrough(stream) => Ok(stream),
        crate::network::openai::moa_gateway::MoaDispatchResult::Responded(status) => {
            Err(RouteDispatchOutcome::Responded(status))
        }
        crate::network::openai::moa_gateway::MoaDispatchResult::RespondedWithUsage {
            status_code,
            usage,
        } => Err(RouteDispatchOutcome::RespondedWithUsage { status_code, usage }),
        crate::network::openai::moa_gateway::MoaDispatchResult::FailedWithStatus {
            status_code,
            reason,
        } => Err(RouteDispatchOutcome::FailedWithStatus {
            status_code,
            reason,
        }),
        crate::network::openai::moa_gateway::MoaDispatchResult::Dropped(reason) => {
            Err(RouteDispatchOutcome::Dropped(reason))
        }
    };
    if let Err(outcome) = &result {
        record_moa_stream_lifecycle(route_observer, adapter, *outcome);
    }
    result
}

async fn build_mesh_request_plan(
    node: &mesh::Node,
    request: &mut BufferedHttpRequest,
    track_demand: bool,
    affinity: &AffinityRouter,
) -> std::result::Result<MeshRequestPlan, MeshRequestFailure> {
    let served = node.models_being_served().await;
    let descriptors = node.all_served_model_descriptors().await;
    rewrite_public_model_alias(request, &served, &descriptors);

    let tokenize_request = request.is_tokenize_request();
    // The automatic directive (either spelling) and a model-less request both
    // resolve through the auto selector, so they get the media-capability
    // filter, readiness, affinity and context-budget fit. A `mesh` request
    // reaches here only in single-model mode — the MoA gateway has already
    // taken any request it is serving as a committee.
    let is_auto_request = !tokenize_request
        && request
            .model_name
            .as_deref()
            .is_none_or(crate::network::openai::automatic::is_directive);
    let auto_session_key = auto_session_key_for_request(request, is_auto_request);
    let required_tokens = request_context_budget(request);
    let effective_model = match resolve_auto_model_request(AutoModelRequestArgs {
        node,
        request,
        served: &served,
        descriptors: &descriptors,
        is_auto_request,
        auto_session_key,
        required_tokens,
        affinity,
    })
    .await
    {
        AutoModelResolution::Model(model) => model.or(request.model_name.clone()),
        AutoModelResolution::UnsupportedMedia => {
            return Err(MeshRequestFailure::UnsupportedMedia);
        }
    };
    rewrite_effective_model(request, effective_model.as_deref());
    if is_auto_request {
        inject_mesh_hooks_flag(&mut request.raw, true);
    }
    if track_demand && let Some(name) = effective_model.as_deref() {
        node.record_request(name);
    }

    let resolved_hosts = match resolve_mesh_target_hosts(node, effective_model.as_deref()).await {
        MeshTargetResolution::Hosts(hosts) => hosts,
        MeshTargetResolution::ModelUnavailable(model) => {
            return Err(MeshRequestFailure::ModelUnavailable(model));
        }
        MeshTargetResolution::NoHostsAvailable => return Err(MeshRequestFailure::NoHostsAvailable),
    };

    let mut prepared = prepare_mesh_targets(
        request,
        effective_model.as_deref(),
        &resolved_hosts,
        affinity,
    );
    let target_hosts = order_mesh_target_hosts(
        node,
        effective_model.as_deref(),
        required_tokens,
        &mut prepared,
        affinity,
    )
    .await;
    Ok(MeshRequestPlan {
        effective_model,
        auto_session_key,
        target_hosts,
        affinity_applied: prepared.affinity_applied,
    })
}

fn rewrite_effective_model(request: &mut BufferedHttpRequest, effective_model: Option<&str>) {
    if request.is_tokenize_request() {
        // The tokenizer's expected identity is authoritative. Never let a
        // later routing decision diverge the routing key from the unchanged
        // wire identity.
        return;
    }
    if let Some(name) = effective_model
        && request.model_name.as_deref() != Some(name)
    {
        rewrite_model_field(request, name);
    }
}

fn prepare_mesh_targets(
    request: &mut BufferedHttpRequest,
    effective_model: Option<&str>,
    target_hosts: &[iroh::EndpointId],
    affinity: &AffinityRouter,
) -> PreparedTargets {
    if !request.is_tokenize_request() && effective_model.is_some() && !target_hosts.is_empty() {
        request.ensure_body_json();
    }
    let body_json = request.body_json.as_ref();
    effective_model
        .map(|name| prepare_remote_targets_for_request(name, target_hosts, body_json, affinity))
        .unwrap_or(PreparedTargets {
            ordered: target_hosts
                .iter()
                .copied()
                .map(election::InferenceTarget::Remote)
                .collect(),
            prefix_hash: None,
            cache_target: None,
            affinity_applied: false,
        })
}

async fn order_mesh_target_hosts(
    node: &mesh::Node,
    effective_model: Option<&str>,
    required_tokens: Option<u32>,
    prepared: &mut PreparedTargets,
    affinity: &AffinityRouter,
) -> Vec<iroh::EndpointId> {
    let target_hosts: Vec<iroh::EndpointId> = prepared
        .ordered
        .iter()
        .filter_map(|target| match target {
            election::InferenceTarget::Remote(host_id) => Some(*host_id),
            _ => None,
        })
        .collect();
    let Some(name) = effective_model else {
        return target_hosts;
    };
    let mut ordered =
        order_remote_hosts_by_context(node, name, required_tokens, &target_hosts).await;
    if affinity.prefix_enabled()
        && let Some(prefix_hash) = prepared.prefix_hash
    {
        let candidates: Vec<_> = ordered
            .iter()
            .copied()
            .map(election::InferenceTarget::Remote)
            .collect();
        prepared.cache_target = match affinity.lookup_cache_lease(name, prefix_hash, &candidates) {
            Some(target) => Some(target),
            None => {
                let lease_epoch = affinity.cache_lease_epoch();
                let selected = node
                    .select_cache_target(name, prefix_hash, &candidates)
                    .await;
                if let Some(target) = selected.as_ref() {
                    affinity.remember_cache_lease_if_epoch(name, prefix_hash, target, lease_epoch);
                }
                selected
            }
        };
        prepared.affinity_applied |= prepared.cache_target.is_some();
        affinity.record_cache_probe(prepared.cache_target.is_some());
        if let Some(election::InferenceTarget::Remote(cache_host)) = prepared.cache_target.as_ref()
        {
            move_target_first(&mut ordered, cache_host);
        }
    }
    ordered
}

async fn handle_mesh_request_failure(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    request: &BufferedHttpRequest,
    failure: MeshRequestFailure,
    route_observer: OpenAiRouteObserver<'_>,
) {
    let mut tcp_stream = Some(tcp_stream);
    match failure {
        MeshRequestFailure::UnsupportedMedia => {
            let _ = send_error_observed(
                tcp_stream.take().unwrap(),
                422,
                "no served model can satisfy the requested media inputs",
                route_observer,
            )
            .await;
        }
        MeshRequestFailure::ModelUnavailable(model) => {
            node.record_routed_request(
                Some(&model),
                0,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            tracing::warn!(
                "API proxy: model {:?} not available, no hosts serving it",
                model
            );
            let _ = send_error_observed(
                tcp_stream.take().unwrap(),
                429,
                &format!("model {:?} not currently available — retry later", model),
                route_observer,
            )
            .await;
        }
        MeshRequestFailure::NoHostsAvailable => {
            node.record_routed_request(
                None,
                0,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            let _ = send_503_observed(
                tcp_stream.take().unwrap(),
                "no peers serving any model (mesh empty or gossip stale)",
                route_observer,
            )
            .await;
        }
    }
    release_request_objects(node, &request.request_object_request_ids).await;
}

async fn route_mesh_request_attempts(
    node: &mesh::Node,
    mut tcp_stream: ClientStream,
    request: &BufferedHttpRequest,
    plan: &MeshRequestPlan,
    affinity: &AffinityRouter,
    route_observer: OpenAiRouteObserver<'_>,
) -> MeshRouteResult {
    let effective_model = plan.effective_model.as_deref();
    let auto_session_key = plan.auto_session_key;
    let (target_hosts, mut reservation) = reserve_mesh_request_target(plan, affinity);
    let total_targets = target_hosts.len();
    let mut state = MeshAttemptState {
        route_started: Instant::now(),
        attempts: 0,
        last_retryable: false,
        refreshed: false,
    };
    for (idx, target_host) in target_hosts.iter().enumerate() {
        transfer_mesh_reservation(reservation.as_mut(), *target_host);
        state.attempts += 1;
        let attempt_started = Instant::now();
        let attempt_result = route_remote_attempt_with_retry(
            node,
            &mut tcp_stream,
            *target_host,
            &request.raw,
            ResponseRetryPolicy::next_target_available(idx + 1 < total_targets),
            RouteAttemptLoggingContext {
                request_id: request.request_id,
                retry_policy: ResponseRetryPolicy::next_target_available(idx + 1 < total_targets),
                response_adapter: request.response_adapter,
                route_observer,
            },
        )
        .await;
        let attempt_target = election::InferenceTarget::Remote(*target_host);
        record_mesh_request_attempt(
            node,
            effective_model,
            &attempt_target,
            attempt_started.duration_since(state.route_started),
            attempt_started.elapsed(),
            &attempt_result,
        );
        affinity.record_target_outcome(
            effective_model,
            &attempt_target,
            target_health_outcome_for_attempt(&attempt_result),
        );
        let mut context = MeshAttemptResultContext {
            node,
            effective_model,
            auto_session_key,
            target_host: *target_host,
            state: &mut state,
            affinity,
        };
        match handle_mesh_attempt_result(&mut context, attempt_result) {
            MeshAttemptDisposition::Continue => continue,
            MeshAttemptDisposition::Return(result) => return MeshRouteResult::Finished(result),
        }
    }
    if state.last_retryable {
        tracing::warn!("All hosts failed for model {:?}", effective_model);
        if let Some(key) = auto_session_key {
            tracing::debug!(
                "auto: all hosts failed for cached model, forgetting session {key:016x}"
            );
            affinity.forget_auto_model(key);
        }
    }
    node.record_routed_request(
        effective_model,
        state.attempts,
        crate::network::metrics::RequestOutcome::Unavailable,
    );
    MeshRouteResult::Exhausted(tcp_stream)
}

fn reserve_mesh_request_target(
    plan: &MeshRequestPlan,
    affinity: &AffinityRouter,
) -> (
    Vec<iroh::EndpointId>,
    Option<crate::network::reservations::RoutingReservation>,
) {
    let mut target_hosts = plan.target_hosts.clone();
    let reservation = plan.effective_model.as_deref().and_then(|model| {
        let candidates = target_hosts
            .iter()
            .copied()
            .map(election::InferenceTarget::Remote)
            .collect::<Vec<_>>();
        let preferred = candidates.first()?;
        let (selected, reservation) =
            affinity.reserve_route(model, &candidates, preferred, plan.affinity_applied)?;
        if let election::InferenceTarget::Remote(selected_host) = selected {
            move_target_first(&mut target_hosts, &selected_host);
        }
        Some(reservation)
    });
    (target_hosts, reservation)
}

fn transfer_mesh_reservation(
    reservation: Option<&mut crate::network::reservations::RoutingReservation>,
    target_host: iroh::EndpointId,
) {
    if let Some(reservation) = reservation {
        reservation.transfer_to(&election::InferenceTarget::Remote(target_host));
    }
}

fn finish_route_attempt(
    route_observer: OpenAiRouteObserver<'_>,
    attempt: Option<OpenAiRouteAttempt>,
    target: &'static str,
    response_adapter: ResponseAdapter,
    result: &RouteAttemptResult,
) {
    let (status_code, error) = proxy_result_metadata(result);
    route_observer.finish_proxy_attempt(
        attempt,
        ProxyAttemptFinish {
            target,
            provider: proxy_provider_for_target(target),
            engine: proxy_engine_for_response_adapter(response_adapter),
            status_code,
            lifecycle_error: status_code
                .is_none()
                .then(|| route_attempt_result_label(result)),
            error,
        },
    );
}

fn proxy_provider_for_target(target: &'static str) -> Option<&'static str> {
    match target {
        "local" | "remote" | "external" => Some(target),
        "none" => None,
        _ => None,
    }
}

fn proxy_engine_for_response_adapter(adapter: ResponseAdapter) -> Option<&'static str> {
    match adapter {
        ResponseAdapter::None => None,
        ResponseAdapter::OpenAiChatCompletionsJson => Some("chat_completion"),
        ResponseAdapter::OpenAiChatCompletionsStream => Some("chat_completion_stream"),
        ResponseAdapter::OpenAiResponsesJson => Some("responses"),
        ResponseAdapter::OpenAiResponsesStream => Some("responses_stream"),
    }
}

fn proxy_result_metadata(result: &RouteAttemptResult) -> (Option<u16>, Option<&'static str>) {
    match result {
        RouteAttemptResult::Delivered { status_code, .. } => (
            Some(*status_code),
            (!(200..400).contains(status_code)).then_some("upstream_status"),
        ),
        RouteAttemptResult::RetryableTimeout => (None, Some("timeout")),
        RouteAttemptResult::RetryableUnavailable => (None, Some("unavailable")),
        RouteAttemptResult::RetryableContextOverflow
        | RouteAttemptResult::RetryableResponseQuality(_) => (None, Some("rejected")),
        RouteAttemptResult::CommittedStreamFailure { status_code } => {
            (Some(*status_code), Some("upstream_stream_incomplete"))
        }
        RouteAttemptResult::ClientDisconnected => (None, Some("client_disconnected")),
    }
}

fn record_mesh_request_attempt(
    node: &mesh::Node,
    effective_model: Option<&str>,
    attempt_target: &election::InferenceTarget,
    queue_wait: Duration,
    attempt_time: Duration,
    attempt_result: &RouteAttemptResult,
) {
    if matches!(attempt_result, RouteAttemptResult::ClientDisconnected) {
        return;
    }
    node.record_inference_attempt(
        effective_model,
        attempt_target,
        queue_wait,
        attempt_time,
        attempt_outcome_for_result(attempt_result),
        completion_tokens_for_result(attempt_result),
    );
}

struct MeshAttemptResultContext<'a> {
    node: &'a mesh::Node,
    effective_model: Option<&'a str>,
    auto_session_key: Option<u64>,
    target_host: iroh::EndpointId,
    state: &'a mut MeshAttemptState,
    affinity: &'a AffinityRouter,
}

fn handle_mesh_attempt_result(
    context: &mut MeshAttemptResultContext<'_>,
    attempt_result: RouteAttemptResult,
) -> MeshAttemptDisposition {
    match attempt_result {
        RouteAttemptResult::Delivered {
            status_code,
            usage,
            cache_cost,
        } => {
            let outcome = request_outcome_for_status(
                status_code,
                crate::network::metrics::RequestService::Remote,
            );
            if let Some(usage) = usage.as_ref() {
                context.node.record_prompt_shape(
                    context.effective_model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    outcome,
                );
            }
            handle_delivered_mesh_attempt(context, status_code);
            MeshAttemptDisposition::Return(RouteAttemptResult::Delivered {
                status_code,
                usage,
                cache_cost,
            })
        }
        RouteAttemptResult::RetryableContextOverflow => handle_retryable_context_overflow(context),
        RouteAttemptResult::RetryableResponseQuality(failure) => {
            handle_retryable_mesh_response_quality(context, failure)
        }
        RouteAttemptResult::RetryableTimeout => handle_retryable_mesh_timeout(context),
        RouteAttemptResult::RetryableUnavailable => handle_retryable_mesh_unavailable(context),
        RouteAttemptResult::CommittedStreamFailure { .. } => {
            MeshAttemptDisposition::Return(attempt_result)
        }
        RouteAttemptResult::ClientDisconnected => {
            tracing::info!(
                "Downstream client disconnected while routing to host {}",
                context.target_host.fmt_short()
            );
            MeshAttemptDisposition::Return(RouteAttemptResult::ClientDisconnected)
        }
    }
}

fn handle_delivered_mesh_attempt(context: &MeshAttemptResultContext<'_>, status_code: u16) {
    if let Some(key) = context
        .auto_session_key
        .filter(|_| (500..600).contains(&status_code))
    {
        tracing::debug!(
            "auto: upstream returned {status_code}, forgetting cached model for session {key:016x}"
        );
        context.affinity.forget_auto_model(key);
    }
    context.node.record_routed_request(
        context.effective_model,
        context.state.attempts,
        request_outcome_for_status(status_code, crate::network::metrics::RequestService::Remote),
    );
}

fn terminal_outcome_for_mesh_route_result(
    result: RouteAttemptResult,
) -> crate::logging::TerminalOutcome {
    match result {
        RouteAttemptResult::Delivered {
            status_code,
            usage: Some(usage),
            ..
        } if (200..400).contains(&status_code) => {
            crate::logging::TerminalOutcome::CompletedWithUsage { status_code, usage }
        }
        RouteAttemptResult::Delivered { status_code, .. } if (200..400).contains(&status_code) => {
            crate::logging::TerminalOutcome::CompletedWithStatus(status_code)
        }
        RouteAttemptResult::Delivered { status_code, .. } if (400..500).contains(&status_code) => {
            crate::logging::TerminalOutcome::RejectedWithStatus {
                reason: Some(format!("upstream_status_{status_code}")),
                status_code,
            }
        }
        RouteAttemptResult::Delivered { status_code, .. } => {
            crate::logging::TerminalOutcome::FailedWithStatus {
                error: format!("upstream_status_{status_code}"),
                status_code,
            }
        }
        RouteAttemptResult::CommittedStreamFailure { status_code } => {
            crate::logging::TerminalOutcome::FailedWithStatus {
                error: "upstream_stream_incomplete".into(),
                status_code,
            }
        }
        RouteAttemptResult::ClientDisconnected => {
            crate::logging::TerminalOutcome::Cancelled(Some("client_disconnected".into()))
        }
        RouteAttemptResult::RetryableContextOverflow
        | RouteAttemptResult::RetryableResponseQuality(_)
        | RouteAttemptResult::RetryableTimeout
        | RouteAttemptResult::RetryableUnavailable => {
            crate::logging::TerminalOutcome::Failed("mesh_route_unavailable".into())
        }
    }
}

fn terminal_outcome_for_mesh_request_failure(
    failure: &MeshRequestFailure,
) -> crate::logging::TerminalOutcome {
    match failure {
        MeshRequestFailure::UnsupportedMedia => {
            crate::logging::TerminalOutcome::Rejected(Some("unsupported_media".into()))
        }
        MeshRequestFailure::ModelUnavailable(_) => {
            crate::logging::TerminalOutcome::Failed("model_unavailable".into())
        }
        MeshRequestFailure::NoHostsAvailable => {
            crate::logging::TerminalOutcome::Failed("no_hosts_available".into())
        }
    }
}

fn handle_retryable_context_overflow(
    context: &mut MeshAttemptResultContext<'_>,
) -> MeshAttemptDisposition {
    tracing::warn!(
        "Host {} rejected request with context overflow-style 400, trying next",
        context.target_host.fmt_short()
    );
    forget_failed_target_cache_leases(context);
    context.state.last_retryable = true;
    MeshAttemptDisposition::Continue
}

fn handle_retryable_mesh_response_quality(
    context: &mut MeshAttemptResultContext<'_>,
    failure: ResponseQualityFailure,
) -> MeshAttemptDisposition {
    tracing::warn!(
        reason = failure.label(),
        "Host {} returned low-quality success response, trying next",
        context.target_host.fmt_short()
    );
    forget_failed_target_cache_leases(context);
    context.state.last_retryable = true;
    MeshAttemptDisposition::Continue
}

fn handle_retryable_mesh_timeout(
    context: &mut MeshAttemptResultContext<'_>,
) -> MeshAttemptDisposition {
    tracing::warn!(
        "Host {} timed out, trying next",
        context.target_host.fmt_short()
    );
    forget_failed_target_cache_leases(context);
    context.state.last_retryable = true;
    spawn_mesh_refresh_once(context.node, &mut context.state.refreshed);
    MeshAttemptDisposition::Continue
}

fn handle_retryable_mesh_unavailable(
    context: &mut MeshAttemptResultContext<'_>,
) -> MeshAttemptDisposition {
    tracing::warn!(
        "Failed to tunnel to host {}, trying next",
        context.target_host.fmt_short()
    );
    forget_failed_target_cache_leases(context);
    context.state.last_retryable = true;
    spawn_mesh_refresh_once(context.node, &mut context.state.refreshed);
    MeshAttemptDisposition::Continue
}

fn forget_failed_target_cache_leases(context: &MeshAttemptResultContext<'_>) {
    let target = election::InferenceTarget::Remote(context.target_host);
    let invalidated = context.affinity.forget_cache_leases_for_target(&target);
    if invalidated > 0 {
        tracing::debug!(
            target = %context.target_host.fmt_short(),
            invalidated,
            "invalidated cache leases after retryable target failure"
        );
    }
}

fn spawn_mesh_refresh_once(node: &mesh::Node, refreshed: &mut bool) {
    if *refreshed {
        return;
    }
    let refresh_node = node.clone();
    tokio::spawn(async move {
        refresh_node.gossip_one_peer().await;
    });
    *refreshed = true;
}

async fn finish_exhausted_mesh_request(
    node: &mesh::Node,
    tcp_stream: ClientStream,
    effective_model: Option<&str>,
    total_targets: usize,
    affinity: &AffinityRouter,
    route_observer: OpenAiRouteObserver<'_>,
) {
    let reason = format!(
        "all {} tunnel(s) to hosts for {:?} failed (mesh request)",
        total_targets, effective_model,
    );
    let _ = affinity;
    let _ = node;
    let _ = send_503_observed(tcp_stream, &reason, route_observer).await;
}

fn auto_session_key_for_request(
    request: &mut BufferedHttpRequest,
    is_auto_request: bool,
) -> Option<u64> {
    if !is_auto_request {
        return None;
    }
    request.ensure_body_json();
    request
        .body_json
        .as_ref()
        .and_then(|body| crate::network::affinity::auto_model_session_key(Some(body)))
}

struct AutoModelRequestArgs<'a> {
    node: &'a mesh::Node,
    request: &'a mut BufferedHttpRequest,
    served: &'a [String],
    descriptors: &'a [mesh::ServedModelDescriptor],
    is_auto_request: bool,
    auto_session_key: Option<u64>,
    required_tokens: Option<u32>,
    affinity: &'a AffinityRouter,
}

async fn resolve_auto_model_request(args: AutoModelRequestArgs<'_>) -> AutoModelResolution {
    let AutoModelRequestArgs {
        node,
        request,
        served,
        descriptors,
        is_auto_request,
        auto_session_key,
        required_tokens,
        affinity,
    } = args;
    if !is_auto_request {
        return AutoModelResolution::Model(None);
    }
    request.ensure_body_json();
    let Some(body_json) = request.body_json.as_ref() else {
        return AutoModelResolution::Model(None);
    };
    let media = router::media_requirements(body_json);
    // Build candidates with observed throughput so pick_model_classified
    // can weight by locally-measured tok/s where samples exist.
    let routing_metrics = node.routing_metrics();
    let with_caps: Vec<router::RoutingCandidate<'_>> = served
        .iter()
        .map(|name| {
            let caps = capabilities_for_model(name, descriptors);
            let (tps_hint, throughput_samples) = routing_metrics
                .tps_for_model(name)
                .map(|(tps, samples)| (Some(tps), samples))
                .unwrap_or((None, 0));
            router::RoutingCandidate {
                name: name.as_str(),
                caps,
                parameter_count_b: descriptor_metadata_for_model(name, descriptors)
                    .and_then(|metadata| metadata.parameter_count_b),
                tps_hint,
                throughput_samples,
            }
        })
        .collect();
    let available = router::filter_media_compatible_candidates(&with_caps, &media);
    let ready_models = if let Some(available) = available.as_ref() {
        auto_route::ready_remote_models(node, required_tokens, available, affinity).await
    } else {
        Vec::new()
    };
    if let Some(model) = lookup_cached_auto_model(
        node,
        descriptors,
        affinity,
        auto_session_key,
        &media,
        &ready_models,
    )
    .await
    {
        return AutoModelResolution::Model(Some(model));
    }

    let Some(available) = available else {
        return AutoModelResolution::UnsupportedMedia;
    };
    let available = auto_route::pool_for_ready_models(&available, &ready_models);
    let cl = router::classify(body_json);
    let picked = router::pick_model_classified(&cl, &available).map(str::to_string);
    if let Some(name) = picked.as_deref() {
        tracing::info!(
            "router: {:?}/{:?} tools={} media={} → {name}",
            cl.category,
            cl.complexity,
            cl.needs_tools,
            cl.has_media_inputs
        );
        if let Some(key) = auto_session_key {
            affinity.remember_auto_model(key, name);
        }
    }
    AutoModelResolution::Model(picked)
}

async fn lookup_cached_auto_model(
    node: &mesh::Node,
    descriptors: &[mesh::ServedModelDescriptor],
    affinity: &AffinityRouter,
    auto_session_key: Option<u64>,
    media: &router::MediaRequirements,
    ready_models: &[&str],
) -> Option<String> {
    let key = auto_session_key?;
    let model = affinity.lookup_auto_model(key)?;
    if let Some(reason) =
        cached_auto_model_reclassify_reason(node, &model, media, descriptors, ready_models).await
    {
        tracing::debug!("auto: cached model {model} {reason}, reclassifying");
        affinity.forget_auto_model(key);
        return None;
    }
    tracing::debug!("auto: reusing cached model {model} for session {key:016x}");
    Some(model)
}

async fn cached_auto_model_reclassify_reason(
    node: &mesh::Node,
    model: &str,
    media: &router::MediaRequirements,
    descriptors: &[mesh::ServedModelDescriptor],
    ready_models: &[&str],
) -> Option<&'static str> {
    if cached_auto_model_missing(node, model).await {
        return Some("no longer served");
    }
    if cached_auto_model_needs_reclassify(model, media, descriptors) {
        return Some("cannot satisfy media requirements");
    }
    if !ready_models.is_empty() && !ready_models.contains(&model) {
        return Some("has no eligible target for this request");
    }
    None
}

async fn cached_auto_model_missing(node: &mesh::Node, model: &str) -> bool {
    node.hosts_for_model(model).await.is_empty()
}

fn cached_auto_model_needs_reclassify(
    model: &str,
    media: &router::MediaRequirements,
    descriptors: &[mesh::ServedModelDescriptor],
) -> bool {
    !cached_auto_model_satisfies_media_requirements(model, media, descriptors)
}

async fn resolve_mesh_target_hosts(
    node: &mesh::Node,
    effective_model: Option<&str>,
) -> MeshTargetResolution {
    let target_hosts = if let Some(name) = effective_model {
        node.hosts_for_model(name).await
    } else {
        Vec::new()
    };
    if !target_hosts.is_empty() {
        return MeshTargetResolution::Hosts(target_hosts);
    }
    if let Some(model) = effective_model {
        return MeshTargetResolution::ModelUnavailable(model.to_string());
    }
    match node.any_host().await {
        Some(peer) => MeshTargetResolution::Hosts(vec![peer.id]),
        None => MeshTargetResolution::NoHostsAvailable,
    }
}

async fn route_attempt_for_target(
    node: &mesh::Node,
    tcp_stream: &mut ClientStream,
    target: &election::InferenceTarget,
    prefetched: &[u8],
    retry_policy: ResponseRetryPolicy,
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let logging = RouteAttemptLoggingContext {
        retry_policy,
        ..logging
    };
    let route_observer = logging.route_observer;
    match target {
        election::InferenceTarget::Local(port) => {
            route_local_transport_attempt(
                node,
                tcp_stream,
                *port,
                prefetched,
                retry_policy,
                logging,
            )
            .await
        }
        election::InferenceTarget::Remote(host_id) => {
            route_remote_attempt_with_retry(
                node,
                tcp_stream,
                *host_id,
                prefetched,
                retry_policy,
                logging,
            )
            .await
        }
        election::InferenceTarget::None => {
            let lifecycle_attempt = route_observer.start_proxy_attempt();
            let result = RouteAttemptResult::RetryableUnavailable;
            finish_route_attempt(
                route_observer,
                lifecycle_attempt,
                "none",
                logging.response_adapter,
                &result,
            );
            result
        }
    }
}

async fn route_local_transport_attempt(
    node: &mesh::Node,
    tcp_stream: &mut ClientStream,
    port: u16,
    prefetched: &[u8],
    retry_policy: ResponseRetryPolicy,
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let logging = RouteAttemptLoggingContext {
        retry_policy,
        ..logging
    };
    let route_observer = logging.route_observer;
    let lifecycle_attempt = route_observer.start_proxy_attempt();
    let result = route_local_attempt(node, tcp_stream, port, prefetched, logging).await;
    finish_route_attempt(
        route_observer,
        lifecycle_attempt,
        "local",
        logging.response_adapter,
        &result,
    );
    result
}

async fn route_remote_attempt_with_retry(
    node: &mesh::Node,
    tcp_stream: &mut ClientStream,
    host_id: iroh::EndpointId,
    prefetched: &[u8],
    retry_policy: ResponseRetryPolicy,
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let mut result = route_remote_transport_attempt(
        node,
        tcp_stream,
        host_id,
        prefetched,
        retry_policy,
        logging,
    )
    .await;
    for retry in 1..=REMOTE_UNCOMMITTED_RETRIES {
        if !should_retry_uncommitted_remote_attempt(result) {
            return result;
        }
        tracing::warn!(
            host = %host_id.fmt_short(),
            retry,
            outcome = route_attempt_result_label(&result),
            "API proxy: retrying remote target on fresh tunnel before committing response"
        );
        result = route_remote_transport_attempt(
            node,
            tcp_stream,
            host_id,
            prefetched,
            retry_policy,
            logging,
        )
        .await;
    }
    result
}

async fn route_remote_transport_attempt(
    node: &mesh::Node,
    tcp_stream: &mut ClientStream,
    host_id: iroh::EndpointId,
    prefetched: &[u8],
    retry_policy: ResponseRetryPolicy,
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let logging = RouteAttemptLoggingContext {
        retry_policy,
        ..logging
    };
    let route_observer = logging.route_observer;
    let lifecycle_attempt = route_observer.start_proxy_attempt();
    let result = route_remote_attempt(node, tcp_stream, host_id, prefetched, logging).await;
    finish_route_attempt(
        route_observer,
        lifecycle_attempt,
        "remote",
        logging.response_adapter,
        &result,
    );
    result
}

#[cfg(test)]
fn record_local_inference_attempt(
    route_observer: OpenAiRouteObserver<'_>,
    result: RouteAttemptResult,
) -> RouteAttemptResult {
    let lifecycle_attempt = route_observer.start_proxy_attempt();
    finish_route_attempt(
        route_observer,
        lifecycle_attempt,
        "local",
        ResponseAdapter::None,
        &result,
    );
    result
}

#[cfg(test)]
fn record_remote_transport_attempt(
    route_observer: OpenAiRouteObserver<'_>,
    result: RouteAttemptResult,
) -> RouteAttemptResult {
    let lifecycle_attempt = route_observer.start_proxy_attempt();
    finish_route_attempt(
        route_observer,
        lifecycle_attempt,
        "remote",
        ResponseAdapter::None,
        &result,
    );
    result
}

fn should_retry_uncommitted_remote_attempt(result: RouteAttemptResult) -> bool {
    matches!(
        result,
        RouteAttemptResult::RetryableTimeout | RouteAttemptResult::RetryableUnavailable
    )
}

/// Route a request to a known inference target (local OpenAI surface or remote host).
///
/// Used by the API proxy after election has determined the target.
pub(crate) struct RouteTargetContext<'a> {
    pub(crate) request_id: RequestId,
    pub(crate) response_adapter: ResponseAdapter,
    pub(crate) route_observer: OpenAiRouteObserver<'a>,
}

pub async fn route_to_target(
    node: mesh::Node,
    tcp_stream: ClientStream,
    model: Option<&str>,
    target: election::InferenceTarget,
    prefetched: &[u8],
    context: RouteTargetContext<'_>,
) -> RouteDispatchOutcome {
    let RouteTargetContext {
        request_id,
        response_adapter,
        route_observer,
    } = context;
    let route_started = Instant::now();
    let mut tcp_stream = tcp_stream;
    route_observer.route_selected(model);
    tracing::info!("API proxy: routing to target {target:?}");
    let retry_policy = ResponseRetryPolicy::next_target_available(false);
    let result = route_attempt_for_target(
        &node,
        &mut tcp_stream,
        &target,
        prefetched,
        retry_policy,
        RouteAttemptLoggingContext {
            request_id,
            retry_policy,
            response_adapter,
            route_observer,
        },
    )
    .await;
    node.record_inference_attempt(
        model,
        &target,
        Duration::ZERO,
        route_started.elapsed(),
        attempt_outcome_for_result(&result),
        completion_tokens_for_result(&result),
    );
    tracing::info!(
        target = ?target,
        outcome = route_attempt_result_label(&result),
        route_ms = route_started.elapsed().as_millis(),
        "openai route_to_target result"
    );
    match result {
        RouteAttemptResult::Delivered {
            status_code, usage, ..
        } => {
            let service = request_service_for_target(&target);
            let outcome = request_outcome_for_status(status_code, service);
            if let Some(usage) = usage.as_ref() {
                node.record_prompt_shape(
                    model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    outcome,
                );
            }
            node.record_routed_request(model, 1, outcome);
            usage.map_or(RouteDispatchOutcome::Responded(status_code), |usage| {
                RouteDispatchOutcome::RespondedWithUsage { status_code, usage }
            })
        }
        RouteAttemptResult::RetryableTimeout
        | RouteAttemptResult::RetryableContextOverflow
        | RouteAttemptResult::RetryableResponseQuality(_)
        | RouteAttemptResult::RetryableUnavailable => {
            node.record_routed_request(
                model,
                1,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            let result = send_503_observed(
                tcp_stream,
                &format!("single target {target:?} unavailable (route_to_target)"),
                route_observer,
            )
            .await;
            response_outcome(503, result)
        }
        RouteAttemptResult::CommittedStreamFailure { status_code } => {
            RouteDispatchOutcome::FailedWithStatus {
                status_code,
                reason: "upstream_stream_incomplete",
            }
        }
        RouteAttemptResult::ClientDisconnected => {
            RouteDispatchOutcome::Dropped("client_disconnected")
        }
    }
}

pub async fn route_http_endpoint_request(
    node: &mesh::Node,
    model: Option<&str>,
    route_metadata: RouteSelectionMetadata<'_>,
    tcp_stream: &mut ClientStream,
    base_url: &str,
    request: &BufferedHttpRequest,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteDispatchOutcome {
    let started = Instant::now();
    route_observer.route_selected_with_metadata(
        model,
        route_metadata.provider,
        route_metadata.engine,
    );
    let lifecycle_attempt = route_observer.start_proxy_attempt();
    let result = route_http_endpoint_attempt(
        tcp_stream,
        base_url,
        &request.raw,
        &request.path,
        RouteAttemptLoggingContext {
            request_id: request.request_id,
            retry_policy: ResponseRetryPolicy::next_target_available(false),
            response_adapter: request.response_adapter,
            route_observer,
        },
    )
    .await;
    finish_route_attempt(
        route_observer,
        lifecycle_attempt,
        "external",
        request.response_adapter,
        &result,
    );
    node.record_endpoint_attempt(
        model,
        base_url,
        Duration::ZERO,
        started.elapsed(),
        attempt_outcome_for_result(&result),
        completion_tokens_for_result(&result),
    );
    tracing::info!(
        endpoint = base_url,
        path = request.path,
        outcome = route_attempt_result_label(&result),
        route_ms = started.elapsed().as_millis(),
        "openai route_http_endpoint_request result"
    );
    match result {
        RouteAttemptResult::Delivered {
            status_code, usage, ..
        } => {
            let outcome = request_outcome_for_status(
                status_code,
                crate::network::metrics::RequestService::Endpoint,
            );
            if let Some(usage) = usage.as_ref() {
                node.record_prompt_shape(
                    model,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    outcome,
                );
            }
            node.record_routed_request(model, 1, outcome);
            usage.map_or(RouteDispatchOutcome::Responded(status_code), |usage| {
                RouteDispatchOutcome::RespondedWithUsage { status_code, usage }
            })
        }
        RouteAttemptResult::RetryableTimeout
        | RouteAttemptResult::RetryableContextOverflow
        | RouteAttemptResult::RetryableResponseQuality(_)
        | RouteAttemptResult::RetryableUnavailable => {
            node.record_routed_request(
                model,
                1,
                crate::network::metrics::RequestOutcome::Unavailable,
            );
            RouteDispatchOutcome::Failed("endpoint_transport_failed")
        }
        RouteAttemptResult::CommittedStreamFailure { status_code } => {
            RouteDispatchOutcome::FailedWithStatus {
                status_code,
                reason: "upstream_stream_incomplete",
            }
        }
        RouteAttemptResult::ClientDisconnected => {
            RouteDispatchOutcome::Dropped("client_disconnected")
        }
    }
}
#[cfg(test)]
#[path = "transport_tests.rs"]
mod tests;
