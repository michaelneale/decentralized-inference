mod chat;
mod control_apply_diagnostics;
mod diagnostics;
mod discover;
mod health;
pub(crate) mod logs;
mod mcp;
mod mesh_hook;
mod model_interests;
mod model_targets;
mod objects;
mod path_picker;
mod plugins;
pub(crate) mod runtime;
mod runtime_activity;
pub(crate) mod runtime_control_state;
mod runtime_control_state_sources;
pub(crate) mod runtime_events;
mod search;

use super::MeshApi;
use std::future::Future;
use std::pin::Pin;
use tokio::net::TcpStream;

type DispatchRequestFn =
    for<'a> fn(
        &'a mut TcpStream,
        &'a MeshApi,
        &'a str,
        &'a str,
        &'a str,
        &'a str,
        &'a str,
        &'a [u8],
    ) -> Pin<Box<dyn Future<Output = anyhow::Result<bool>> + Send + 'a>>;

pub(super) const DISPATCH_REQUEST: DispatchRequestFn =
    |stream, state, method, path, path_only, body, req, raw_request| {
        Box::pin(async move {
            match (method, path_only) {
                (method, route_path) if logs::is_route(route_path) => {
                    logs::handle(stream, method, path, body, raw_request).await?;
                    Ok(true)
                }
                ("GET", "/api/discover") => {
                    discover::handle(stream, state).await?;
                    Ok(true)
                }
                ("GET", "/health") => {
                    health::handle(stream, state).await?;
                    Ok(true)
                }
                ("POST", p) if p == crate::network::discovery::LAN_DETAILS_PATH => {
                    discover::handle_lan_details(stream, state, body).await?;
                    Ok(true)
                }
                ("GET", "/api/diagnostics/split-readiness") => {
                    diagnostics::handle(stream, state, path).await?;
                    Ok(true)
                }
                ("GET", "/api/diagnostics") => {
                    diagnostics::handle(stream, state, path).await?;
                    Ok(true)
                }
                ("GET" | "POST" | "DELETE", "/mcp") => {
                    mcp::handle(stream, state, raw_request).await?;
                    Ok(true)
                }
                ("POST", "/api/runtime/pick-directory") => {
                    path_picker::handle(stream).await?;
                    Ok(true)
                }
                ("GET", "/api/runtime/events/v1") => {
                    runtime_events::handle(stream, state, path, raw_request).await?;
                    Ok(true)
                }
                ("GET", "/api/status")
                | ("GET", "/api/models")
                | ("GET", "/api/runtime")
                | ("GET", "/api/runtime/llama")
                | ("GET", "/api/runtime/events")
                | ("GET", "/api/runtime/endpoints")
                | ("GET", "/api/runtime/processes")
                | ("GET", "/api/runtime/stages")
                | ("GET", "/api/runtime/config-schema")
                | ("GET", "/api/runtime/config-control-state")
                | ("GET", "/api/runtime/control-bootstrap")
                | ("GET", "/api/runtime/intents")
                | ("GET", "/api/runtime/activity")
                | ("POST", "/api/runtime/control/get-config")
                | ("POST", "/api/runtime/control/scan-refresh")
                | ("POST", "/api/runtime/control/refresh-inventory")
                | ("POST", "/api/runtime/control/apply-config")
                | ("POST", "/api/runtime/control/load-model")
                | ("POST", "/api/runtime/control/unload-model")
                | ("POST", "/api/runtime/control/ensure-model")
                | ("POST", "/api/runtime/control/drain-model")
                | ("POST", "/api/runtime/config/validate")
                | ("POST", "/api/runtime/mesh-guardrails")
                | ("POST", "/api/runtime/models")
                | ("PUT", "/api/runtime/activity/override")
                | ("DELETE", "/api/runtime/activity/override")
                | ("GET", "/api/events") => {
                    runtime::handle(stream, state, method, path_only, body).await?;
                    Ok(true)
                }
                ("DELETE", p) if p.starts_with("/api/runtime/instances/") => {
                    runtime::handle(stream, state, method, path_only, body).await?;
                    Ok(true)
                }
                ("DELETE", p) if p.starts_with("/api/runtime/models/") => {
                    runtime::handle(stream, state, method, path_only, body).await?;
                    Ok(true)
                }
                ("GET", "/api/search") => {
                    search::handle(stream, path).await?;
                    Ok(true)
                }
                ("GET", "/api/model-interests") | ("POST", "/api/model-interests") => {
                    model_interests::handle(stream, state, method, path_only, body).await?;
                    Ok(true)
                }
                ("GET", "/api/model-targets") => {
                    model_targets::handle(stream, state).await?;
                    Ok(true)
                }
                ("DELETE", p) if p.starts_with("/api/model-interests/") => {
                    model_interests::handle(stream, state, method, path_only, body).await?;
                    Ok(true)
                }
                ("GET", "/api/plugins") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", "/api/plugins/endpoints") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", "/api/plugins/providers") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", p) if p.starts_with("/api/plugins/providers/") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/manifest") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("GET", p) if p.starts_with("/api/plugins/") && p.ends_with("/tools") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                ("POST", p) if p.starts_with("/api/plugins/") && p.contains("/tools/") => {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                (m, p)
                    if p.starts_with("/api/plugins/")
                        && matches!(m, "GET" | "POST" | "PUT" | "PATCH" | "DELETE") =>
                {
                    plugins::handle(stream, state, method, path, path_only, body, raw_request)
                        .await?;
                    Ok(true)
                }
                // Mesh hook callbacks from the serving runtime
                ("POST", "/mesh/hook") => {
                    mesh_hook::handle(stream, state, method, path_only, body).await?;
                    Ok(true)
                }
                ("POST", "/api/objects")
                | ("POST", "/api/objects/complete")
                | ("POST", "/api/objects/abort") => {
                    objects::handle(stream, state, method, path_only, body).await?;
                    Ok(true)
                }
                (m, p)
                    if matches!(m, "GET" | "POST" | "OPTIONS")
                        && (p.starts_with("/v1/") || p == "/models") =>
                {
                    chat::handle(stream, state, method, path_only, req).await?;
                    Ok(true)
                }
                (m, p)
                    if m != "POST"
                        && (p.starts_with("/api/chat") || p.starts_with("/api/responses")) =>
                {
                    chat::handle(stream, state, method, path_only, req).await?;
                    Ok(true)
                }
                ("POST", p) if p.starts_with("/api/chat") || p.starts_with("/api/responses") => {
                    chat::handle(stream, state, method, path_only, req).await?;
                    Ok(true)
                }
                _ => Ok(false),
            }
        })
    };

pub(super) use DISPATCH_REQUEST as dispatch_request;
