use super::super::{
    MeshApi, RuntimeControlRequest,
    http::{respond_error, respond_json, respond_runtime_error},
    status::decode_runtime_model_path,
};
use super::control_apply_diagnostics::{
    LocalControlApplyDiagnosticPayload, local_control_apply_diagnostic_payload,
    local_control_apply_diagnostic_payload_from_local,
};
use super::runtime_control_state::collect_runtime_config_control_state_payload;
use crate::config_schema::{EngineConfigSchemaDescriptor, export_runtime_config_schema_reference};
use crate::crypto::{
    OwnerKeychainLoadError, keystore_metadata, load_keystore, load_owner_keypair_from_keychain,
};
use crate::plugin::validate_config_diagnostics_with_installed_plugin_schemas;
use futures_util::{FutureExt, StreamExt, stream};
use mesh_client::{
    ClientBuilder, ControlPlaneBootstrapOptions, ControlPlaneClientError, ControlPlaneConnection,
    InviteToken, OwnerControlRemoteError, client::control_plane::OwnerControlScanRefreshResult,
};
use mesh_llm_config::{
    ConfigConditionValue, ConfigControlAvailabilitySource, ConfigDisabledWritePolicy,
    ConfigOptionsSource,
};
use mesh_llm_config::{ConfigDiagnosticSeverity, legacy_validation_error_text};
use mesh_llm_node::serving::{UnloadOptions, UnloadTarget};
use openai_frontend::GuardrailMode;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::net::SocketAddr;

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use zeroize::Zeroizing;

const CONTROL_KV_CACHE_MAX_ENDPOINTS: usize = 256;
const CONTROL_KV_CACHE_CONCURRENCY: usize = 8;
const CONTROL_KV_CACHE_BATCH_TIMEOUT_SECS: u64 = 45;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    match method {
        "GET" => handle_get(stream, state, path_only).await,
        "POST" => handle_post(stream, state, path_only, body).await,
        "PUT" => handle_put(stream, state, path_only, body).await,
        "DELETE" => handle_delete(stream, state, path_only, body).await,
        _ => Ok(()),
    }
}

async fn handle_get(
    stream: &mut TcpStream,
    state: &MeshApi,
    path_only: &str,
) -> anyhow::Result<()> {
    match path_only {
        "/api/status" => handle_status(stream, state).await,
        "/api/models" => handle_models(stream, state).await,
        "/api/runtime" => handle_runtime_status(stream, state).await,
        "/api/runtime/llama" => handle_runtime_llama(stream, state).await,
        "/api/runtime/events" => handle_runtime_events(stream, state).await,
        "/api/runtime/endpoints" => handle_runtime_endpoints(stream, state).await,
        "/api/runtime/processes" => handle_runtime_processes(stream, state).await,
        "/api/runtime/stages" => handle_runtime_stages(stream, state).await,
        "/api/runtime/config-schema" => handle_runtime_config_schema(stream).await,
        "/api/runtime/config-control-state" => {
            handle_runtime_config_control_state(stream, state).await
        }
        "/api/runtime/kv-cache" => super::kv_cache::handle_status(stream).await,
        "/api/runtime/control-bootstrap" => handle_control_bootstrap(stream, state).await,
        "/api/runtime/intents" => handle_get_intents(stream, state).await,
        "/api/runtime/activity" => super::runtime_activity::handle_get(stream, state).await,
        "/api/events" => handle_events(stream, state).await,
        _ => Ok(()),
    }
}

async fn handle_post(
    stream: &mut TcpStream,
    state: &MeshApi,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    match path_only {
        "/api/runtime/control/get-config" => handle_control_get_config(stream, state, body).await,
        "/api/runtime/control/scan-refresh" => {
            handle_control_scan_refresh(stream, state, body).await
        }
        "/api/runtime/control/refresh-inventory" => {
            handle_control_refresh_inventory(stream, state, body).await
        }
        "/api/runtime/control/apply-config" => {
            handle_control_apply_config(stream, state, body).await
        }
        "/api/runtime/control/kv-cache" => handle_control_kv_cache(stream, state, body).await,
        "/api/runtime/control/load-model" => handle_control_load_model(stream, state, body).await,
        "/api/runtime/control/unload-model" => {
            handle_control_unload_model(stream, state, body).await
        }
        "/api/runtime/control/ensure-model" => {
            handle_control_ensure_model(stream, state, body).await
        }
        "/api/runtime/control/drain-model" => handle_control_drain_model(stream, state, body).await,
        "/api/runtime/config/validate" => handle_runtime_config_validate(stream, body).await,
        "/api/runtime/mesh-guardrails" => handle_set_mesh_guardrails(stream, state, body).await,
        "/api/runtime/models" => handle_load_model(stream, state, body).await,
        "/api/runtime/kv-cache/prune" => super::kv_cache::handle_prune(stream, body).await,
        _ => Ok(()),
    }
}

async fn handle_put(
    stream: &mut TcpStream,
    state: &MeshApi,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    match path_only {
        "/api/runtime/activity/override" => {
            super::runtime_activity::handle_put(stream, state, body).await
        }
        _ => Ok(()),
    }
}

async fn handle_delete(
    stream: &mut TcpStream,
    state: &MeshApi,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    match path_only {
        "/api/runtime/activity/override" => {
            super::runtime_activity::handle_delete(stream, state).await
        }
        "/api/runtime/kv-cache" => super::kv_cache::handle_clear(stream, body).await,
        p if p.starts_with("/api/runtime/instances/") => {
            handle_unload_instance(stream, state, p).await
        }
        p if p.starts_with("/api/runtime/models/") => handle_unload_model(stream, state, p).await,
        _ => Ok(()),
    }
}

async fn handle_control_bootstrap(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    respond_json(stream, 200, &state.control_bootstrap().await).await
}

async fn handle_runtime_config_schema(stream: &mut TcpStream) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }

    match export_runtime_config_schema_reference(std::iter::empty::<EngineConfigSchemaDescriptor>())
    {
        Ok(schema) => respond_json(stream, 200, &schema).await,
        Err(error) => respond_error(stream, 500, &error.to_string()).await,
    }
}

async fn handle_runtime_config_control_state(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }

    respond_json(
        stream,
        200,
        &collect_runtime_config_control_state_payload(state).await,
    )
    .await
}

#[derive(Debug, Deserialize)]
struct ControlEndpointRequest {
    endpoint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApplyConfigRequest {
    endpoint: Option<String>,
    expected_revision: u64,
    config: crate::plugin::MeshConfig,
}

#[derive(Debug, Deserialize)]
struct RawApplyConfigRequest {
    endpoint: Option<String>,
    expected_revision: u64,
    config: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ControlKvCacheRequest {
    endpoints: Vec<String>,
    operation: String,
    target_bytes: Option<u64>,
    model_identity: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ValidateConfigRequest {
    toml: String,
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiGuardrailsModeRequest {
    mode: String,
}

#[derive(Debug, Deserialize)]
struct ControlLifecycleModelRequest {
    endpoint: Option<String>,
    model: Option<String>,
    instance_id: Option<String>,
    profile: Option<String>,
}

#[derive(Clone, Copy)]
enum ControlLifecycleOperation {
    Load,
    Unload,
    Ensure,
    Drain,
}

struct ControlLifecycleTarget {
    model: String,
    instance_id: Option<String>,
    profile: Option<String>,
}

#[derive(Debug, Serialize)]
struct LocalControlSnapshotPayload {
    node_id: String,
    revision: u64,
    config_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hostname: Option<String>,
    config: crate::plugin::MeshConfig,
}

#[derive(Debug, Serialize)]
struct LocalControlScanRefreshPayload {
    target_node_id: String,
    disposition: Option<String>,
    inventory: Option<Vec<LocalControlInventoryEntryPayload>>,
}

#[derive(Debug, Serialize)]
struct LocalControlInventoryEntryPayload {
    canonical_model_ref: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,
    total_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<LocalControlModelMetadataPayload>,
}

#[derive(Debug, Serialize)]
struct LocalControlModelMetadataPayload {
    model_key: String,
    context_length: u32,
    vocab_size: u32,
    embedding_size: u32,
    head_count: u32,
    layer_count: u32,
    feed_forward_length: u32,
    key_length: u32,
    value_length: u32,
    architecture: String,
    tokenizer_model_name: String,
    special_tokens: Vec<LocalControlSpecialTokenPayload>,
    rope_scale: f32,
    rope_freq_base: f32,
    is_moe: bool,
    expert_count: u32,
    used_expert_count: u32,
    quantization_type: String,
    kv_head_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameter_size: Option<String>,
}

#[derive(Debug, Serialize)]
struct LocalControlSpecialTokenPayload {
    name: String,
    token_id: u32,
}

#[derive(Debug, Serialize)]
struct LocalControlApplyPayload {
    success: bool,
    current_revision: u64,
    config_hash: String,
    apply_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    diagnostics: Vec<LocalControlApplyDiagnosticPayload>,
}

#[derive(Debug, Serialize)]
struct LocalConfigValidatePayload {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    diagnostics: Vec<LocalControlApplyDiagnosticPayload>,
}

#[derive(Debug, Serialize)]
struct LocalControlErrorPayload {
    code: String,
    message: String,
    legacy_retry_allowed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    current_revision: Option<u64>,
}

#[derive(Debug, Serialize)]
struct LocalControlKvCachePayload {
    results: Vec<LocalControlKvCacheResult>,
}

#[derive(Debug, Serialize)]
struct LocalControlKvCacheResult {
    endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_node_id: Option<String>,
    operation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    freed_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<LocalControlErrorPayload>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ConfigControlStatePayload {
    #[serde(default)]
    pub(crate) settings: BTreeMap<String, ConfigControlStateEntry>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ConfigControlStateEntry {
    pub(crate) enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) note: Option<String>,
    pub(crate) source: ConfigControlAvailabilitySource,
    pub(crate) write_policy: ConfigDisabledWritePolicy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) options: Option<Vec<ConfigControlOption>>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ConfigControlOption {
    pub(crate) value: ConfigConditionValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) note: Option<String>,
    pub(crate) disabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<String>,
    pub(crate) source: ConfigOptionsSource,
}

async fn handle_control_get_config(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: ControlEndpointRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let endpoint = match required_control_endpoint(request.endpoint) {
        Ok(endpoint) => endpoint,
        Err(error) => return respond_control_error(stream, error).await,
    };
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            let result = client.get_config().await;
            client.close().await;
            match result {
                Ok(snapshot) => respond_json(
                    stream,
                    200,
                    &serde_json::json!({ "snapshot": local_control_snapshot_payload(snapshot) }),
                )
                .await,
                Err(error) => respond_control_error(stream, control_error_from_client(error)).await,
            }
        }
        Err(error) => respond_control_error(stream, error).await,
    }
}

async fn handle_control_refresh_inventory(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: ControlEndpointRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let endpoint = match required_control_endpoint(request.endpoint) {
        Ok(endpoint) => endpoint,
        Err(error) => return respond_control_error(stream, error).await,
    };
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            let result = client.refresh_inventory().await;
            client.close().await;
            match result {
                Ok(snapshot) => respond_json(
                    stream,
                    200,
                    &serde_json::json!({ "snapshot": local_control_snapshot_payload(snapshot) }),
                )
                .await,
                Err(error) => respond_control_error(stream, control_error_from_client(error)).await,
            }
        }
        Err(error) => respond_control_error(stream, error).await,
    }
}

async fn handle_control_scan_refresh(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: ControlEndpointRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let endpoint = match required_control_endpoint(request.endpoint) {
        Ok(endpoint) => endpoint,
        Err(error) => return respond_control_error(stream, error).await,
    };
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            let result = client.scan_refresh().await;
            client.close().await;
            match result {
                Ok(response) => {
                    respond_json(stream, 200, &local_control_scan_refresh_payload(response)).await
                }
                Err(error) => respond_control_error(stream, control_error_from_client(error)).await,
            }
        }
        Err(error) => respond_control_error(stream, error).await,
    }
}

async fn handle_control_apply_config(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let raw_request: RawApplyConfigRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let raw_config_toml = toml::to_string(&raw_request.config).ok();
    let request = ApplyConfigRequest {
        endpoint: raw_request.endpoint,
        expected_revision: raw_request.expected_revision,
        config: match serde_json::from_value(raw_request.config) {
            Ok(config) => config,
            Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
        },
    };
    let endpoint = match required_control_endpoint(request.endpoint) {
        Ok(endpoint) => endpoint,
        Err(error) => return respond_control_error(stream, error).await,
    };
    let diagnostics = validate_config_diagnostics_with_installed_plugin_schemas(
        &request.config,
        raw_config_toml.as_deref(),
    );
    if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
    {
        return respond_json(
            stream,
            200,
            &LocalControlApplyPayload {
                success: false,
                current_revision: request.expected_revision,
                config_hash: hex::encode(crate::protocol::convert::canonical_config_hash(
                    &crate::protocol::convert::mesh_config_to_proto(&request.config),
                )),
                apply_mode: "unspecified".to_string(),
                error: Some(legacy_validation_error_text(&diagnostics)),
                diagnostics: diagnostics
                    .iter()
                    .map(local_control_apply_diagnostic_payload_from_local)
                    .collect(),
            },
        )
        .await;
    }
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            let result = client
                .apply_config(
                    request.expected_revision,
                    crate::protocol::convert::mesh_config_to_proto(&request.config),
                )
                .await;
            client.close().await;
            match result {
                Ok(response) => {
                    respond_json(
                        stream,
                        200,
                        &LocalControlApplyPayload {
                            success: response.success,
                            current_revision: response.current_revision,
                            config_hash: hex::encode(response.config_hash),
                            apply_mode: control_apply_mode_label(response.apply_mode),
                            error: response.error,
                            diagnostics: response
                                .diagnostics
                                .iter()
                                .map(local_control_apply_diagnostic_payload)
                                .collect(),
                        },
                    )
                    .await
                }
                Err(error) => respond_control_error(stream, control_error_from_client(error)).await,
            }
        }
        Err(error) => respond_control_error(stream, error).await,
    }
}

async fn handle_control_kv_cache(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: ControlKvCacheRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    if request.endpoints.is_empty() {
        return respond_error(
            stream,
            400,
            "at least one owner-control endpoint is required",
        )
        .await;
    }
    if request.endpoints.len() > CONTROL_KV_CACHE_MAX_ENDPOINTS {
        return respond_error(
            stream,
            400,
            "at most 256 owner-control endpoints are allowed",
        )
        .await;
    }
    let operation = match request.operation.as_str() {
        "status" => mesh_client::proto::node::OwnerControlKvCacheOperation::Status,
        "prune" => mesh_client::proto::node::OwnerControlKvCacheOperation::Prune,
        "clear" => mesh_client::proto::node::OwnerControlKvCacheOperation::Clear,
        _ => return respond_error(stream, 400, "unknown kv-cache operation").await,
    };
    let endpoints = request.endpoints;
    let timeout_endpoints = endpoints.clone();
    let operation_label = request.operation.clone();
    let mut operations = stream::iter(endpoints.into_iter().enumerate())
        .map(|(index, endpoint)| {
            execute_control_kv_cache_target(
                state,
                endpoint,
                request.operation.clone(),
                operation,
                request.target_bytes,
                request.model_identity.clone(),
            )
            .map(move |result| (index, result))
        })
        .buffer_unordered(CONTROL_KV_CACHE_CONCURRENCY);
    let deadline = tokio::time::Instant::now()
        + std::time::Duration::from_secs(CONTROL_KV_CACHE_BATCH_TIMEOUT_SECS);
    let mut indexed_results = Vec::with_capacity(timeout_endpoints.len());
    loop {
        match tokio::time::timeout_at(deadline, operations.next()).await {
            Ok(Some(result)) => indexed_results.push(result),
            Ok(None) => break,
            Err(_) => break,
        }
    }
    drop(operations);
    if indexed_results.len() < timeout_endpoints.len() {
        let completed = indexed_results
            .iter()
            .map(|(index, _)| *index)
            .collect::<std::collections::HashSet<_>>();
        indexed_results.extend(
            timeout_endpoints
                .into_iter()
                .enumerate()
                .filter(|(index, _)| !completed.contains(index))
                .map(|(index, endpoint)| {
                    (
                        index,
                        LocalControlKvCacheResult {
                            endpoint,
                            target_node_id: None,
                            operation: operation_label.clone(),
                            freed_bytes: None,
                            status: None,
                            error: Some(LocalControlErrorPayload {
                                code: "control_timeout".to_string(),
                                message: format!(
                                    "kv-cache batch exceeded its {CONTROL_KV_CACHE_BATCH_TIMEOUT_SECS}-second deadline before a receipt was returned"
                                ),
                                legacy_retry_allowed: false,
                                current_revision: None,
                            }),
                        },
                    )
                }),
        );
    }
    indexed_results.sort_by_key(|(index, _)| *index);
    let results = indexed_results
        .into_iter()
        .map(|(_, result)| result)
        .collect();
    respond_json(stream, 200, &LocalControlKvCachePayload { results }).await
}

async fn execute_control_kv_cache_target(
    state: &MeshApi,
    endpoint: String,
    operation_label: String,
    operation: mesh_client::proto::node::OwnerControlKvCacheOperation,
    target_bytes: Option<u64>,
    model_identity: Option<String>,
) -> LocalControlKvCacheResult {
    let mut result = LocalControlKvCacheResult {
        endpoint: endpoint.clone(),
        target_node_id: None,
        operation: operation_label,
        freed_bytes: None,
        status: None,
        error: None,
    };
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            result.target_node_id = Some(hex::encode(client.target_node_id()));
            let response = client
                .kv_cache(operation, target_bytes, model_identity)
                .await;
            client.close().await;
            match response {
                Ok(response) => {
                    result.freed_bytes = response.freed_bytes;
                    match serde_json::from_slice(&response.status_json) {
                        Ok(status) => result.status = Some(status),
                        Err(error) => {
                            result.error = Some(control_error_from_anyhow(anyhow::anyhow!(
                                "invalid kv-cache status from target: {error}"
                            )));
                        }
                    }
                }
                Err(error) => result.error = Some(control_error_from_client(error)),
            }
        }
        Err(error) => result.error = Some(error),
    }
    result
}

async fn handle_control_load_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    handle_control_lifecycle_model(stream, state, body, ControlLifecycleOperation::Load).await
}

async fn handle_control_unload_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    handle_control_lifecycle_model(stream, state, body, ControlLifecycleOperation::Unload).await
}

async fn handle_control_ensure_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    handle_control_lifecycle_model(stream, state, body, ControlLifecycleOperation::Ensure).await
}

async fn handle_control_drain_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    handle_control_lifecycle_model(stream, state, body, ControlLifecycleOperation::Drain).await
}

async fn handle_control_lifecycle_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
    operation: ControlLifecycleOperation,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: ControlLifecycleModelRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let endpoint = request.endpoint.clone();
    let target = match validate_control_lifecycle_target(operation, request) {
        Ok(target) => target,
        Err(message) => return respond_error(stream, 400, message).await,
    };
    let endpoint = match required_control_endpoint(endpoint) {
        Ok(endpoint) => endpoint,
        Err(error) => return respond_control_error(stream, error).await,
    };
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            let result = invoke_control_lifecycle(&client, operation, target).await;
            client.close().await;
            match result {
                Ok(response) => respond_json(stream, 200, &response).await,
                Err(error) => respond_control_error(stream, control_error_from_client(error)).await,
            }
        }
        Err(error) => respond_control_error(stream, error).await,
    }
}

fn validate_control_lifecycle_target(
    operation: ControlLifecycleOperation,
    request: ControlLifecycleModelRequest,
) -> Result<ControlLifecycleTarget, &'static str> {
    let ControlLifecycleModelRequest {
        model,
        instance_id,
        profile,
        ..
    } = request;
    match operation {
        ControlLifecycleOperation::Load | ControlLifecycleOperation::Ensure => {
            match (model, instance_id) {
                (Some(model), None) if !model.trim().is_empty() => Ok(ControlLifecycleTarget {
                    model,
                    instance_id: None,
                    profile,
                }),
                _ if matches!(operation, ControlLifecycleOperation::Load) => {
                    Err("load-model requires a model reference and does not accept an instance id")
                }
                _ => Err(
                    "ensure-model requires a model reference and does not accept an instance id",
                ),
            }
        }
        ControlLifecycleOperation::Unload | ControlLifecycleOperation::Drain => {
            let target = match (model, instance_id, profile) {
                (Some(model), None, None) if !model.trim().is_empty() => {
                    Some(ControlLifecycleTarget {
                        model,
                        instance_id: None,
                        profile: None,
                    })
                }
                (None, Some(instance_id), None) if !instance_id.trim().is_empty() => {
                    Some(ControlLifecycleTarget {
                        model: String::new(),
                        instance_id: Some(instance_id),
                        profile: None,
                    })
                }
                _ => None,
            };
            target.ok_or({
                if matches!(operation, ControlLifecycleOperation::Unload) {
                    "unload-model requires exactly one model reference or instance id"
                } else {
                    "drain-model requires exactly one model reference or instance id"
                }
            })
        }
    }
}

async fn invoke_control_lifecycle(
    client: &mesh_client::OwnerControlClient,
    operation: ControlLifecycleOperation,
    target: ControlLifecycleTarget,
) -> Result<serde_json::Value, ControlPlaneClientError> {
    let (intent_id, accepted_state, response_target) = match operation {
        ControlLifecycleOperation::Load => {
            let response = client.load_model(target.model, target.profile).await?;
            (response.intent_id, response.accepted_state, response.target)
        }
        ControlLifecycleOperation::Unload => {
            let response = client
                .unload_model(target.model, target.instance_id)
                .await?;
            (response.intent_id, response.accepted_state, response.target)
        }
        ControlLifecycleOperation::Ensure => {
            let response = client.ensure_model(target.model, target.profile).await?;
            (response.intent_id, response.accepted_state, response.target)
        }
        ControlLifecycleOperation::Drain => {
            let response = client.drain_model(target.model, target.instance_id).await?;
            (response.intent_id, response.accepted_state, response.target)
        }
    };
    Ok(serde_json::json!({
        "accepted": true,
        "intent_id": intent_id,
        "accepted_state": accepted_state,
        "model": response_target.as_ref().map(|target| &target.canonical_model_ref),
        "instance_id": response_target.as_ref().and_then(|target| target.instance_id.as_deref()),
    }))
}

async fn handle_runtime_config_validate(stream: &mut TcpStream, body: &str) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }

    let request: ValidateConfigRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };

    let config: crate::plugin::MeshConfig = match toml::from_str(&request.toml) {
        Ok(config) => config,
        Err(error) => {
            return respond_json(
                stream,
                200,
                &LocalConfigValidatePayload {
                    ok: false,
                    path: request.path,
                    error: Some(format!("Invalid config TOML: {error}")),
                    diagnostics: Vec::new(),
                },
            )
            .await;
        }
    };

    let diagnostics =
        validate_config_diagnostics_with_installed_plugin_schemas(&config, Some(&request.toml));
    let ok = diagnostics
        .iter()
        .all(|diagnostic| diagnostic.severity != ConfigDiagnosticSeverity::Error);
    respond_json(
        stream,
        200,
        &LocalConfigValidatePayload {
            ok,
            path: request.path,
            error: None,
            diagnostics: diagnostics
                .iter()
                .map(local_control_apply_diagnostic_payload_from_local)
                .collect(),
        },
    )
    .await
}

async fn connect_owner_control_client(
    state: &MeshApi,
    endpoint: &str,
) -> Result<mesh_client::OwnerControlClient, LocalControlErrorPayload> {
    let owner_key_path = state.owner_key_path().await;
    let owner_keypair =
        load_local_owner_keypair(owner_key_path.as_deref()).map_err(control_error_from_anyhow)?;
    let client = ClientBuilder::new(owner_keypair, InviteToken("local-control".to_string()))
        .build()
        .map_err(|error| control_error_from_anyhow(anyhow::anyhow!(error.to_string())))?;
    let connection = client
        .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(endpoint))
        .await
        .map_err(control_error_from_client)?;
    match connection {
        ControlPlaneConnection::OwnerControl(client) => Ok(*client),
    }
}

fn required_control_endpoint(endpoint: Option<String>) -> Result<String, LocalControlErrorPayload> {
    match endpoint.map(|value| value.trim().to_string()) {
        Some(endpoint) if !endpoint.is_empty() => Ok(endpoint),
        _ => Err(LocalControlErrorPayload {
            code: "control_endpoint_required".to_string(),
            message:
                "owner-control endpoint must be supplied explicitly; no gossip or peer inference is used"
                    .to_string(),
            legacy_retry_allowed: false,
            current_revision: None,
        }),
    }
}

pub(super) async fn ensure_loopback_control_caller(stream: &mut TcpStream) -> anyhow::Result<bool> {
    ensure_loopback_control_caller_for_peer_addr(stream, stream.peer_addr()).await
}

pub(crate) async fn ensure_loopback_control_caller_for_peer_addr(
    stream: &mut TcpStream,
    peer_addr: std::io::Result<std::net::SocketAddr>,
) -> anyhow::Result<bool> {
    match peer_addr {
        Ok(addr) if is_loopback_control_caller(addr) => Ok(true),
        Ok(addr) => {
            tracing::warn!("runtime control: rejected non-loopback caller {addr}");
            respond_json(
                stream,
                403,
                &serde_json::json!({"error": "runtime control endpoints only accept localhost connections"}),
            )
            .await?;
            Ok(false)
        }
        Err(error) => {
            tracing::warn!("runtime control: could not determine caller address: {error}");
            respond_json(
                stream,
                403,
                &serde_json::json!({"error": "runtime control endpoints require a localhost caller"}),
            )
            .await?;
            Ok(false)
        }
    }
}

fn is_loopback_control_caller(addr: SocketAddr) -> bool {
    addr.ip().is_loopback()
}

fn local_control_snapshot_payload(
    snapshot: mesh_client::proto::node::OwnerControlConfigSnapshot,
) -> LocalControlSnapshotPayload {
    let config = snapshot
        .config
        .as_ref()
        .map(crate::protocol::convert::proto_config_to_mesh)
        .unwrap_or_default();
    LocalControlSnapshotPayload {
        node_id: hex::encode(snapshot.node_id),
        revision: snapshot.revision,
        config_hash: hex::encode(snapshot.config_hash),
        hostname: snapshot.hostname,
        config,
    }
}

fn local_control_scan_refresh_payload(
    response: OwnerControlScanRefreshResult,
) -> LocalControlScanRefreshPayload {
    let target_node_id = hex::encode(&response.snapshot.node_id);
    let (disposition, inventory) = match response.inventory {
        Some(inventory) => (
            Some(control_scan_disposition_label(inventory.disposition)),
            Some(
                inventory
                    .entries
                    .into_iter()
                    .map(local_control_inventory_entry_payload)
                    .collect(),
            ),
        ),
        None => (None, None),
    };
    LocalControlScanRefreshPayload {
        target_node_id,
        disposition,
        inventory,
    }
}

fn control_scan_disposition_label(value: i32) -> String {
    match mesh_client::proto::node::OwnerControlRefreshInventoryDisposition::try_from(value) {
        Ok(mesh_client::proto::node::OwnerControlRefreshInventoryDisposition::Executed) => {
            "executed"
        }
        Ok(mesh_client::proto::node::OwnerControlRefreshInventoryDisposition::Coalesced) => {
            "coalesced"
        }
        _ => "unspecified",
    }
    .to_string()
}

fn local_control_inventory_entry_payload(
    entry: mesh_client::proto::node::OwnerControlInventoryEntry,
) -> LocalControlInventoryEntryPayload {
    LocalControlInventoryEntryPayload {
        canonical_model_ref: entry.canonical_model_ref,
        display_name: entry.display_name,
        total_size_bytes: entry.total_size_bytes,
        metadata: entry.metadata.map(local_control_model_metadata_payload),
    }
}

fn local_control_model_metadata_payload(
    metadata: mesh_client::proto::node::CompactModelMetadata,
) -> LocalControlModelMetadataPayload {
    LocalControlModelMetadataPayload {
        model_key: metadata.model_key,
        context_length: metadata.context_length,
        vocab_size: metadata.vocab_size,
        embedding_size: metadata.embedding_size,
        head_count: metadata.head_count,
        layer_count: metadata.layer_count,
        feed_forward_length: metadata.feed_forward_length,
        key_length: metadata.key_length,
        value_length: metadata.value_length,
        architecture: metadata.architecture,
        tokenizer_model_name: metadata.tokenizer_model_name,
        special_tokens: metadata
            .special_tokens
            .into_iter()
            .map(|token| LocalControlSpecialTokenPayload {
                name: token.name,
                token_id: token.token_id,
            })
            .collect(),
        rope_scale: metadata.rope_scale,
        rope_freq_base: metadata.rope_freq_base,
        is_moe: metadata.is_moe,
        expert_count: metadata.expert_count,
        used_expert_count: metadata.used_expert_count,
        quantization_type: metadata.quantization_type,
        kv_head_count: metadata.kv_head_count,
        parameter_size: metadata.parameter_size,
    }
}

fn control_apply_mode_label(value: i32) -> String {
    match mesh_client::proto::node::ConfigApplyMode::try_from(value) {
        Ok(mesh_client::proto::node::ConfigApplyMode::Staged) => "staged".to_string(),
        Ok(mesh_client::proto::node::ConfigApplyMode::Live) => "live".to_string(),
        Ok(mesh_client::proto::node::ConfigApplyMode::Noop) => "noop".to_string(),
        Ok(mesh_client::proto::node::ConfigApplyMode::Unspecified) => "unspecified".to_string(),
        _ => "unspecified".to_string(),
    }
}

fn load_local_owner_keypair(
    path: Option<&std::path::Path>,
) -> anyhow::Result<crate::crypto::OwnerKeypair> {
    let path =
        path.ok_or_else(|| anyhow::anyhow!("local owner keystore unavailable for this runtime"))?;
    let info = keystore_metadata(path)?;
    if info.encrypted && std::env::var("MESH_LLM_OWNER_PASSPHRASE").is_err() {
        match load_owner_keypair_from_keychain(path) {
            Ok(keypair) => return Ok(keypair),
            Err(OwnerKeychainLoadError::NoEntry)
            | Err(OwnerKeychainLoadError::Crypto(crate::crypto::CryptoError::DecryptionFailed))
            | Err(OwnerKeychainLoadError::Crypto(
                crate::crypto::CryptoError::KeychainUnavailable { .. },
            ))
            | Err(OwnerKeychainLoadError::Crypto(
                crate::crypto::CryptoError::KeychainAccessDenied { .. },
            )) => {}
            Err(OwnerKeychainLoadError::Crypto(err)) => {
                let error: anyhow::Error = err.into();
                return Err(
                    error.context(format!("Failed to load owner keystore {}", path.display()))
                );
            }
        }
    }
    let passphrase = resolve_owner_passphrase(path)?;
    load_keystore(path, passphrase.as_deref().map(|value| value.as_str()))
        .map_err(Into::into)
        .map_err(|error: anyhow::Error| {
            error.context(format!("Failed to load owner keystore {}", path.display()))
        })
}

fn resolve_owner_passphrase(path: &std::path::Path) -> anyhow::Result<Option<Zeroizing<String>>> {
    if let Ok(passphrase) = std::env::var("MESH_LLM_OWNER_PASSPHRASE") {
        return Ok(Some(Zeroizing::new(passphrase)));
    }
    let info = keystore_metadata(path)?;
    if !info.encrypted {
        return Ok(None);
    }
    Err(crate::crypto::CryptoError::MissingPassphrase.into())
}

fn control_error_from_client(error: ControlPlaneClientError) -> LocalControlErrorPayload {
    match error {
        ControlPlaneClientError::Negotiation(error) => LocalControlErrorPayload {
            code: owner_control_error_code_label(error.code),
            message: error.message,
            legacy_retry_allowed: error.legacy_retry_allowed,
            current_revision: None,
        },
        ControlPlaneClientError::Remote(error) => control_error_from_remote(error),
        ControlPlaneClientError::Transport(message) => LocalControlErrorPayload {
            code: "control_unavailable".to_string(),
            message,
            legacy_retry_allowed: false,
            current_revision: None,
        },
        ControlPlaneClientError::Protocol(message) => LocalControlErrorPayload {
            code: "control_protocol_error".to_string(),
            message,
            legacy_retry_allowed: false,
            current_revision: None,
        },
    }
}

fn control_error_from_remote(error: OwnerControlRemoteError) -> LocalControlErrorPayload {
    LocalControlErrorPayload {
        code: owner_control_error_code_label(error.code),
        message: error.message,
        legacy_retry_allowed: false,
        current_revision: error.current_revision,
    }
}

fn control_error_from_anyhow(error: anyhow::Error) -> LocalControlErrorPayload {
    LocalControlErrorPayload {
        code: "control_unavailable".to_string(),
        message: error.to_string(),
        legacy_retry_allowed: false,
        current_revision: None,
    }
}

fn owner_control_error_code_label(code: mesh_client::proto::node::OwnerControlErrorCode) -> String {
    match code {
        mesh_client::proto::node::OwnerControlErrorCode::Unspecified => "unspecified",
        mesh_client::proto::node::OwnerControlErrorCode::ControlEndpointRequired => {
            "control_endpoint_required"
        }
        mesh_client::proto::node::OwnerControlErrorCode::ControlUnavailable => {
            "control_unavailable"
        }
        mesh_client::proto::node::OwnerControlErrorCode::ControlUnsupported => {
            "control_unsupported"
        }
        mesh_client::proto::node::OwnerControlErrorCode::Unauthorized => "unauthorized",
        mesh_client::proto::node::OwnerControlErrorCode::TargetNodeMismatch => {
            "target_node_mismatch"
        }
        mesh_client::proto::node::OwnerControlErrorCode::RevisionConflict => "revision_conflict",
        mesh_client::proto::node::OwnerControlErrorCode::InvalidHandshake => "invalid_handshake",
        mesh_client::proto::node::OwnerControlErrorCode::LegacyJsonUnsupported => {
            "legacy_json_unsupported"
        }
        mesh_client::proto::node::OwnerControlErrorCode::UnknownCommand => "unknown_command",
        mesh_client::proto::node::OwnerControlErrorCode::BadRequest => "bad_request",
    }
    .to_string()
}

fn control_error_status_code(error: &LocalControlErrorPayload) -> u16 {
    match error.code.as_str() {
        "control_endpoint_required" | "bad_request" | "target_node_mismatch" => 400,
        "unauthorized" => 403,
        "revision_conflict" => 409,
        "control_unavailable" | "control_unsupported" => 503,
        _ => 502,
    }
}

async fn respond_control_error(
    stream: &mut TcpStream,
    error: LocalControlErrorPayload,
) -> anyhow::Result<()> {
    respond_json(
        stream,
        control_error_status_code(&error),
        &serde_json::json!({ "error": error }),
    )
    .await
}

async fn handle_runtime_stages(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_stages()).await {
        Ok(runtime_stages) => respond_json(stream, 200, &runtime_stages).await,
        Err(_) => respond_error(stream, 503, "Runtime stage status temporarily unavailable").await,
    }
}

async fn handle_status(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.status()).await {
        Ok(status) => respond_json(stream, 200, &status).await,
        Err(_) => respond_error(stream, 503, "Status temporarily unavailable").await,
    }
}

async fn handle_models(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let mesh_models = state.mesh_models().await;
    respond_json(
        stream,
        200,
        &serde_json::json!({ "mesh_models": mesh_models }),
    )
    .await
}

async fn handle_runtime_status(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_status()).await {
        Ok(runtime_status) => respond_json(stream, 200, &runtime_status).await,
        Err(_) => respond_error(stream, 503, "Runtime status temporarily unavailable").await,
    }
}

async fn handle_runtime_processes(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_processes()).await {
        Ok(runtime_processes) => respond_json(stream, 200, &runtime_processes).await,
        Err(_) => {
            respond_error(
                stream,
                503,
                "Runtime process status temporarily unavailable",
            )
            .await
        }
    }
}

async fn handle_runtime_llama(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_llama()).await {
        Ok(runtime_llama) => respond_json(stream, 200, &runtime_llama).await,
        Err(_) => {
            respond_error(
                stream,
                503,
                "Runtime llama snapshot temporarily unavailable",
            )
            .await
        }
    }
}

async fn handle_runtime_events(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let request_id = super::super::management_lifecycle::response_request_id_header()
        .map(|request_id| format!("x-request-id: {request_id}\r\n"))
        .unwrap_or_default();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n{request_id}\r\n"
    );
    stream.write_all(header.as_bytes()).await?;
    super::super::management_lifecycle::record_response_status(200);

    let mut subscription = {
        state
            .inner
            .lock()
            .await
            .runtime_data_collector
            .clone()
            .subscribe()
    };
    let mut last_sent_json = None;

    let runtime_llama = state.runtime_llama().await;
    if let Ok(json) = serde_json::to_string(&runtime_llama) {
        stream
            .write_all(format!("data: {json}\n\n").as_bytes())
            .await?;
        last_sent_json = Some(json);
    }

    loop {
        tokio::select! {
            changed = subscription.changed() => {
                match changed {
                    Ok(()) => {
                        let subscription_state = *subscription.borrow_and_update();
                        if !subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::RUNTIME) {
                            continue;
                        }
                        let runtime_llama = state.runtime_llama().await;
                        let Ok(json) = serde_json::to_string(&runtime_llama) else {
                            continue;
                        };
                        if last_sent_json.as_deref() == Some(json.as_str()) {
                            continue;
                        }
                        if stream.write_all(format!("data: {json}\n\n").as_bytes()).await.is_err() {
                            break;
                        }
                        last_sent_json = Some(json);
                    }
                    Err(_) => break,
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => {
                if stream.write_all(b": keepalive\n\n").await.is_err() {
                    break;
                }
            }
        }
    }

    Ok(())
}

async fn handle_runtime_endpoints(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match state.runtime_endpoints().await {
        Ok(endpoints) => {
            let endpoints = endpoints
                .into_iter()
                .map(|endpoint| endpoint.redacted_for_network())
                .collect::<Vec<_>>();
            respond_json(stream, 200, &serde_json::json!({ "endpoints": endpoints })).await
        }
        Err(err) => respond_error(stream, 500, &err.to_string()).await,
    }
}

async fn handle_set_mesh_guardrails(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }

    let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
        return respond_error(stream, 503, "Runtime control unavailable").await;
    };
    let request: OpenAiGuardrailsModeRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let Some(mode) = parse_guardrail_mode(&request.mode) else {
        return respond_error(stream, 400, "Invalid guardrail mode").await;
    };

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let _ = control_tx.send(RuntimeControlRequest::SetOpenAiGuardrailMode {
        mode,
        resp: resp_tx,
    });
    match resp_rx.await {
        Ok(Ok(updated)) => respond_json(stream, 200, &updated).await?,
        Ok(Err(e)) => respond_runtime_error(stream, &e.to_string()).await?,
        Err(_) => respond_error(stream, 503, "Runtime control unavailable").await?,
    }
    Ok(())
}

fn parse_guardrail_mode(mode: &str) -> Option<GuardrailMode> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "disabled" | "disable" | "off" => Some(GuardrailMode::Disabled),
        "metrics" | "metrics_only" | "metrics-only" => Some(GuardrailMode::MetricsOnly),
        "enforce" | "enforced" => Some(GuardrailMode::Enforce),
        _ => None,
    }
}

async fn handle_load_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
        return respond_error(stream, 503, "Runtime control unavailable").await;
    };

    let (spec, profile) = match parse_runtime_load_request(body) {
        Ok((spec, profile)) => (spec, profile),
        Err(RuntimeLoadRequestParseError::InvalidJson) => {
            respond_error(stream, 400, "Invalid JSON body").await?;
            return Ok(());
        }
        Err(RuntimeLoadRequestParseError::MissingModel) => {
            respond_error(stream, 400, "Missing 'model' field").await?;
            return Ok(());
        }
    };

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let _ = control_tx.send(RuntimeControlRequest::Load {
        spec,
        config_model_id: None,
        profile,
        resp: resp_tx,
    });
    match resp_rx.await {
        Ok(Ok(loaded)) => {
            respond_json(
                stream,
                201,
                &serde_json::json!({
                    "loaded": loaded.model,
                    "instance_id": loaded.instance_id,
                }),
            )
            .await?;
        }
        Ok(Err(e)) => {
            respond_runtime_error(stream, &e.to_string()).await?;
        }
        Err(_) => {
            respond_error(stream, 503, "Runtime control unavailable").await?;
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum RuntimeLoadRequestParseError {
    InvalidJson,
    MissingModel,
}

fn parse_runtime_load_request(
    body: &str,
) -> Result<(String, String), RuntimeLoadRequestParseError> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|_| RuntimeLoadRequestParseError::InvalidJson)?;
    let model = value
        .get("model")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .ok_or(RuntimeLoadRequestParseError::MissingModel)?;
    let (model_ref, profile) = parse_model_with_profile(model);
    Ok((model_ref.to_string(), profile.to_string()))
}

fn parse_model_with_profile(model: &str) -> (&str, &str) {
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

async fn handle_unload_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
        return respond_error(stream, 503, "Runtime control unavailable").await;
    };
    let Some(model_name) = decode_runtime_model_path(path, "/api/runtime/models/") else {
        return respond_error(stream, 400, "Missing model path").await;
    };

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let _ = control_tx.send(RuntimeControlRequest::Unload {
        target: UnloadTarget::Model(model_name.clone()),
        options: UnloadOptions::default(),
        resp: resp_tx,
    });
    match resp_rx.await {
        Ok(Ok(dropped)) => {
            respond_json(
                stream,
                200,
                &serde_json::json!({
                    "dropped": dropped.model,
                    "instance_id": dropped.instance_id,
                }),
            )
            .await?;
        }
        Ok(Err(e)) => {
            respond_runtime_error(stream, &e.to_string()).await?;
        }
        Err(_) => {
            respond_error(stream, 503, "Runtime control unavailable").await?;
        }
    }
    Ok(())
}

async fn handle_unload_instance(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
        return respond_error(stream, 503, "Runtime control unavailable").await;
    };
    let Some(instance_id) = decode_runtime_model_path(path, "/api/runtime/instances/") else {
        return respond_error(stream, 400, "Missing runtime instance path").await;
    };

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let _ = control_tx.send(RuntimeControlRequest::Unload {
        target: UnloadTarget::Instance(instance_id.clone()),
        options: UnloadOptions::default(),
        resp: resp_tx,
    });
    match resp_rx.await {
        Ok(Ok(dropped)) => {
            respond_json(
                stream,
                200,
                &serde_json::json!({
                    "dropped": dropped.model,
                    "instance_id": dropped.instance_id,
                }),
            )
            .await?;
        }
        Ok(Err(e)) => {
            respond_runtime_error(stream, &e.to_string()).await?;
        }
        Err(_) => {
            respond_error(stream, 503, "Runtime control unavailable").await?;
        }
    }
    Ok(())
}

async fn handle_events(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let request_id = super::super::management_lifecycle::response_request_id_header()
        .map(|request_id| format!("x-request-id: {request_id}\r\n"))
        .unwrap_or_default();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n{request_id}\r\n"
    );
    stream.write_all(header.as_bytes()).await?;
    super::super::management_lifecycle::record_response_status(200);

    let status = state.status().await;
    let mut last_sent_json = None;
    if let Ok(json) = serde_json::to_string(&status) {
        stream
            .write_all(format!("data: {json}\n\n").as_bytes())
            .await?;
        last_sent_json = Some(json);
    }

    let mut subscription = {
        state
            .inner
            .lock()
            .await
            .runtime_data_collector
            .clone()
            .subscribe()
    };

    loop {
        tokio::select! {
            changed = subscription.changed() => {
                match changed {
                    Ok(()) => {
                        let subscription_state = *subscription.borrow_and_update();
                        let interesting = subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::STATUS)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::MODELS)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::ROUTING)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::PROCESSES)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::INVENTORY)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::PLUGINS);
                        if !interesting {
                            continue;
                        }
                        let status = state.status().await;
                        let Ok(json) = serde_json::to_string(&status) else {
                            continue;
                        };
                        if last_sent_json.as_deref() == Some(json.as_str()) {
                            continue;
                        }
                        if stream.write_all(format!("data: {json}\n\n").as_bytes()).await.is_err() {
                            break;
                        }
                        last_sent_json = Some(json);
                    }
                    Err(_) => break,
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => {
                if stream.write_all(b": keepalive\n\n").await.is_err() {
                    break;
                }
            }
        }
    }

    Ok(())
}

// ─── Intent and Activity API handlers ─────────────────────────────────────────

use crate::api::status::{IntentEntry, IntentListPayload};

/// Maximum number of intent entries returned in a single response.
const INTENT_CAP: usize = 256;

async fn handle_get_intents(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }

    let intent_history = {
        let inner = state.inner.lock().await;
        inner
            .node
            .runtime_intents
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    };

    let total_count = intent_history.len();
    let truncated = total_count > INTENT_CAP;
    let entries = intent_history
        .into_iter()
        .rev()
        .take(INTENT_CAP)
        .map(|intent| IntentEntry {
            intent_id: intent.intent_id,
            model_ref: intent.canonical_model_ref,
            profile: intent.profile,
            source: intent.source.into(),
            desired_state: intent.desired_state.as_str().to_string(),
            instance_target: intent.instance_target,
            persistence: match intent.persistence {
                crate::runtime::IntentPersistence::Process => "process",
                crate::runtime::IntentPersistence::Session => "session",
                crate::runtime::IntentPersistence::Ephemeral => "ephemeral",
            }
            .to_string(),
            created_at_secs: intent.created_at_secs,
            updated_at_secs: intent.updated_at_secs,
            last_error: intent.last_error,
        })
        .collect();

    respond_json(
        stream,
        200,
        &IntentListPayload {
            intents: entries,
            total_count,
            truncated,
        },
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::{
        GuardrailMode, RuntimeLoadRequestParseError, is_loopback_control_caller,
        local_control_scan_refresh_payload, parse_guardrail_mode, parse_model_with_profile,
        parse_runtime_load_request,
    };
    use mesh_client::client::control_plane::OwnerControlScanRefreshResult;

    #[test]
    fn loopback_control_caller_accepts_localhost_only() {
        assert!(is_loopback_control_caller(
            "127.0.0.1:3131".parse().unwrap()
        ));
        assert!(is_loopback_control_caller("[::1]:3131".parse().unwrap()));
        assert!(!is_loopback_control_caller(
            "192.0.2.10:3131".parse().unwrap()
        ));
        assert!(!is_loopback_control_caller(
            "[2001:db8::1]:3131".parse().unwrap()
        ));
    }

    #[test]
    fn parse_guardrail_mode_accepts_operator_labels() {
        assert_eq!(
            parse_guardrail_mode("disabled"),
            Some(GuardrailMode::Disabled)
        );
        assert_eq!(parse_guardrail_mode("off"), Some(GuardrailMode::Disabled));
        assert_eq!(
            parse_guardrail_mode("metrics-only"),
            Some(GuardrailMode::MetricsOnly)
        );
        assert_eq!(
            parse_guardrail_mode("enforce"),
            Some(GuardrailMode::Enforce)
        );
    }

    #[test]
    fn parse_guardrail_mode_rejects_unknown_labels() {
        assert_eq!(parse_guardrail_mode(""), None);
        assert_eq!(parse_guardrail_mode("strict"), None);
    }

    #[test]
    fn parse_runtime_load_request_with_profile() {
        assert_eq!(
            parse_runtime_load_request(r#"{"model": "Qwen3-8B#low-ctx"}"#).unwrap(),
            ("Qwen3-8B".to_string(), "low-ctx".to_string()),
        );
        assert_eq!(
            parse_runtime_load_request(r#"{"model": "Qwen3-8B"}"#).unwrap(),
            ("Qwen3-8B".to_string(), String::new()),
        );
    }

    #[test]
    fn parse_runtime_load_request_missing_model() {
        assert!(matches!(
            parse_runtime_load_request(r#"{"foo": "bar"}"#),
            Err(RuntimeLoadRequestParseError::MissingModel)
        ));
    }

    #[test]
    fn parse_runtime_load_request_rejects_invalid_json() {
        assert!(matches!(
            parse_runtime_load_request("invalid"),
            Err(RuntimeLoadRequestParseError::InvalidJson)
        ));
    }

    #[test]
    fn parse_model_with_profile_from_runtime_route() {
        let (model_ref, profile) = parse_model_with_profile("Qwen3-8B#low-ctx");
        assert_eq!(model_ref, "Qwen3-8B");
        assert_eq!(profile, "low-ctx");
    }

    #[test]
    fn scan_refresh_payload_preserves_snapshot_only_compatibility() {
        let payload = local_control_scan_refresh_payload(OwnerControlScanRefreshResult {
            snapshot: mesh_client::proto::node::OwnerControlConfigSnapshot {
                node_id: vec![0xab, 0xcd],
                ..Default::default()
            },
            inventory: None,
        });

        assert_eq!(
            serde_json::to_value(payload).unwrap(),
            serde_json::json!({
                "target_node_id": "abcd",
                "disposition": null,
                "inventory": null,
            })
        );
    }

    #[test]
    fn scan_refresh_payload_maps_sorted_inventory_and_metadata_explicitly() {
        let payload = local_control_scan_refresh_payload(OwnerControlScanRefreshResult {
            snapshot: mesh_client::proto::node::OwnerControlConfigSnapshot {
                node_id: vec![1],
                ..Default::default()
            },
            inventory: Some(mesh_client::proto::node::OwnerControlRefreshInventory {
                entries: vec![mesh_client::proto::node::OwnerControlInventoryEntry {
                    canonical_model_ref: "a/model".to_string(),
                    display_name: Some("Model A".to_string()),
                    total_size_bytes: 42,
                    metadata: Some(mesh_client::proto::node::CompactModelMetadata {
                        model_key: "a/model".to_string(),
                        architecture: "llama".to_string(),
                        special_tokens: vec![mesh_client::proto::node::SpecialToken {
                            name: "bos".to_string(),
                            token_id: 1,
                        }],
                        ..Default::default()
                    }),
                }],
                disposition:
                    mesh_client::proto::node::OwnerControlRefreshInventoryDisposition::Executed
                        as i32,
            }),
        });

        let value = serde_json::to_value(payload).unwrap();
        assert_eq!(value["target_node_id"], "01");
        assert_eq!(value["disposition"], "executed");
        assert_eq!(value["inventory"][0]["canonical_model_ref"], "a/model");
        assert_eq!(value["inventory"][0]["metadata"]["architecture"], "llama");
        assert_eq!(
            value["inventory"][0]["metadata"]["special_tokens"][0]["name"],
            "bos"
        );
    }
}
