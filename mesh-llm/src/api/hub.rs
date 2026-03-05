use super::*;
use base64::Engine as _;
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tokio::net::TcpStream;

const HUB_MEMBERSHIP_SYNC_INTERVAL_SECS: u64 = 60;
const HUB_MEMBERSHIP_TTL_SECS: u64 = 15 * 60;
const HUB_OUTAGE_GRACE_SECS: u64 = 10 * 60;
const HUB_RUNTIME_SESSION_TTL_SECS: u64 = 15 * 60;
const HUB_RUNTIME_SESSION_REFRESH_BEFORE_SECS: u64 = 5 * 60;
const HUB_TELEMETRY_SYNC_INTERVAL_SECS: u64 = 30;

#[derive(Clone, Default)]
pub(super) struct HubState {
    pub(super) base_url: String,
    pub(super) access_token: Option<String>,
    pub(super) auth_pending: bool,
    pub(super) profile: Option<HubProfile>,
    pub(super) default_startup_model: Option<String>,
    pub(super) default_mesh_selector: Option<String>,
    pub(super) default_invite_token: Option<String>,
    pub(super) node_id: Option<String>,
    pub(super) linked_mesh_id: Option<String>,
    pub(super) linked_mesh_slug: Option<String>,
    pub(super) linked_mesh_name: Option<String>,
    pub(super) linked_mesh_visibility: Option<String>,
    pub(super) link_state: String,
    pub(super) membership_enforcement: String,
    pub(super) last_membership_sync_ok_unix: Option<u64>,
    pub(super) runtime_access_token: Option<String>,
    pub(super) runtime_access_token_expires_unix: Option<u64>,
    pub(super) telemetry_seq: u64,
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub(super) struct HubProfile {
    pub(super) name: Option<String>,
    pub(super) handle: Option<String>,
    pub(super) avatar_url: Option<String>,
}

#[derive(Serialize, Deserialize, Default)]
struct HubSessionFile {
    access_token: Option<String>,
    profile: Option<HubProfile>,
    default_startup_model: Option<String>,
    default_mesh_selector: Option<String>,
    default_invite_token: Option<String>,
    node_id: Option<String>,
    linked_mesh_id: Option<String>,
    linked_mesh_slug: Option<String>,
    linked_mesh_name: Option<String>,
    linked_mesh_visibility: Option<String>,
    link_state: Option<String>,
    membership_enforcement: Option<String>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct HubIdentityFile {
    signing_key_b64: String,
    public_key_b64: String,
}

struct HubTelemetrySnapshot {
    node_id: String,
    hub_mesh_id: String,
    active_nodes: u64,
    total_vram_gb: f64,
    node_total_vram_gb: f64,
    node_available_vram_gb: f64,
    node_gpu_count: u64,
    node_inflight_requests: u64,
    node_requests_total: u64,
    node_requests_failed_total: u64,
    node_input_tokens_total: u64,
    node_output_tokens_total: u64,
    mesh_avg_ttft_ms: u64,
    node_warm_models: Vec<String>,
    mesh_warm_models: Vec<String>,
}
pub(super) fn load_state() -> HubState {
    let hub_base_url = std::env::var("MESH_LLM_HUB_BASE_URL")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "https://www.inferencehub.cc".to_string())
        .trim_end_matches('/')
        .to_string();
    let session = load_hub_session().unwrap_or_default();
    HubState {
        base_url: hub_base_url,
        access_token: session.access_token,
        auth_pending: false,
        profile: session.profile,
        default_startup_model: session.default_startup_model,
        default_mesh_selector: session.default_mesh_selector,
        default_invite_token: session.default_invite_token,
        node_id: session.node_id,
        linked_mesh_id: session.linked_mesh_id,
        linked_mesh_slug: session.linked_mesh_slug,
        linked_mesh_name: session.linked_mesh_name,
        linked_mesh_visibility: session.linked_mesh_visibility,
        link_state: session.link_state.unwrap_or_else(|| "unlinked".to_string()),
        membership_enforcement: session
            .membership_enforcement
            .unwrap_or_else(|| "local".to_string()),
        last_membership_sync_ok_unix: None,
        runtime_access_token: None,
        runtime_access_token_expires_unix: None,
        telemetry_seq: 0,
    }
}

fn hub_session_path() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    Some(home.join(".mesh-llm").join("hub-session.json"))
}

fn hub_identity_path() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    Some(home.join(".mesh-llm").join("hub-identity.json"))
}

fn load_hub_session() -> Option<HubSessionFile> {
    let path = hub_session_path()?;
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<HubSessionFile>(&data).ok()
}

fn save_hub_session(session: &HubSessionFile) -> anyhow::Result<()> {
    let Some(path) = hub_session_path() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string_pretty(session)?;
    std::fs::write(path, payload)?;
    Ok(())
}

fn load_hub_identity() -> Option<HubIdentityFile> {
    let path = hub_identity_path()?;
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<HubIdentityFile>(&data).ok()
}

fn save_hub_identity(identity: &HubIdentityFile) -> anyhow::Result<()> {
    let Some(path) = hub_identity_path() else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string_pretty(identity)?;
    std::fs::write(path, payload)?;
    Ok(())
}

fn load_or_create_hub_identity() -> anyhow::Result<HubIdentityFile> {
    if let Some(existing) = load_hub_identity() {
        if !existing.signing_key_b64.trim().is_empty() && !existing.public_key_b64.trim().is_empty()
        {
            return Ok(existing);
        }
    }
    let sk: [u8; 32] = rand::random();
    let signing_key = SigningKey::from_bytes(&sk);
    let verifying_key = signing_key.verifying_key();
    let identity = HubIdentityFile {
        signing_key_b64: base64::engine::general_purpose::STANDARD.encode(signing_key.to_bytes()),
        public_key_b64: base64::engine::general_purpose::STANDARD.encode(verifying_key.to_bytes()),
    };
    save_hub_identity(&identity)?;
    Ok(identity)
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn default_node_display_name() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "mesh-node".to_string())
}

fn sign_hub_message(identity: &HubIdentityFile, message: &str) -> anyhow::Result<String> {
    let secret_bytes = base64::engine::general_purpose::STANDARD
        .decode(identity.signing_key_b64.as_bytes())
        .map_err(|e| anyhow::anyhow!("invalid stored signing key: {e}"))?;
    let signing_key_bytes: [u8; 32] = secret_bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("stored signing key has invalid length"))?;
    let signing_key = SigningKey::from_bytes(&signing_key_bytes);
    let signature = signing_key.sign(message.as_bytes());
    Ok(base64::engine::general_purpose::STANDARD.encode(signature.to_bytes()))
}

pub(super) async fn initialize(state: &MeshApi) {
    // Apply persisted hub link identity to gossip announcements on startup.
    state.apply_hub_identity_to_node().await;
    if state.hub_access_token().await.is_some() {
        if let Ok(profile) = state.hub_fetch_profile().await {
            state.set_hub_profile(profile).await;
        }
        state.refresh_linked_mesh_metadata().await;
        if let Err(err) = state.auto_link_default_target_on_startup().await {
            tracing::warn!("Hub auto-link on startup failed: {err}");
        }
    }
}

pub(super) fn spawn_background_tasks(state: &MeshApi) {
    // Runtime membership enforcement for linked meshes.
    // In hub_enforced mode, periodically re-validate peers against hub membership.
    let state6 = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(
            HUB_MEMBERSHIP_SYNC_INTERVAL_SECS,
        ));
        interval.tick().await;
        loop {
            interval.tick().await;
            if !state6.is_hub_enforced_mode().await {
                continue;
            }
            let (dropped, status_changed) = state6.enforce_hub_membership_once().await;
            if dropped > 0 || status_changed {
                state6.push_status().await;
            }
        }
    });

    // Runtime telemetry publishing for hub-linked meshes.
    let state7 = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(
            HUB_TELEMETRY_SYNC_INTERVAL_SECS,
        ));
        interval.tick().await;
        loop {
            interval.tick().await;
            if let Err(err) = state7.publish_hub_telemetry_once().await {
                tracing::warn!("Hub telemetry publish failed: {err}");
            }
        }
    });
}

pub(super) async fn is_hub_enforced_mode(state: &MeshApi) -> bool {
    state.is_hub_enforced_mode().await
}

pub(super) async fn handle_route(
    state: &MeshApi,
    method: &str,
    path_only: &str,
    req: &str,
    stream: &mut TcpStream,
) -> anyhow::Result<bool> {
    match (method, path_only) {
        ("POST", "/api/hub/login-device/start") => {
            let body = parse_json_body(req).unwrap_or_default();
            let client_name = body
                .get("client_name")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .unwrap_or("mesh-llm");
            let base_url = state.hub_base_url().await;
            let client = reqwest::Client::new();
            match client
                .post(format!("{base_url}/api/v0/device-auth/start"))
                .json(&serde_json::json!({ "client_name": client_name }))
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let text = resp.text().await.unwrap_or_else(|_| {
                        "{\"error\":\"failed to read hub response\"}".to_string()
                    });
                    if (200..300).contains(&status) {
                        state.set_hub_auth_pending(true).await;
                        state.push_status().await;
                    }
                    respond_json_raw(stream, status, &text).await?;
                }
                Err(e) => {
                    respond_error(stream, 502, &format!("Hub auth start failed: {e}")).await?;
                }
            }
        }

        ("POST", "/api/hub/login-device/poll") => {
            let body = parse_json_body(req).unwrap_or_default();
            let Some(device_code) = body
                .get("device_code")
                .and_then(|v| v.as_str())
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            else {
                respond_error(stream, 400, "device_code is required").await?;
                return Ok(true);
            };
            let base_url = state.hub_base_url().await;
            let client = reqwest::Client::new();
            match client
                .post(format!("{base_url}/api/v0/device-auth/poll"))
                .json(&serde_json::json!({ "device_code": device_code }))
                .send()
                .await
            {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let text = resp.text().await.unwrap_or_else(|_| {
                        "{\"error\":\"failed to read hub response\"}".to_string()
                    });
                    if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&text) {
                        match payload.get("status").and_then(|v| v.as_str()) {
                            Some("authorized") => {
                                if let Some(token) = payload
                                    .get("access_token")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                                {
                                    state.set_hub_access_token(Some(token)).await;
                                    state.set_hub_auth_pending(false).await;
                                    if let Ok(profile) = state.hub_fetch_profile().await {
                                        state.set_hub_profile(profile).await;
                                    }
                                }
                            }
                            Some("pending") => {
                                state.set_hub_auth_pending(true).await;
                            }
                            _ => {
                                state.set_hub_auth_pending(false).await;
                            }
                        }
                        state.push_status().await;
                    }
                    respond_json_raw(stream, status, &text).await?;
                }
                Err(e) => {
                    respond_error(stream, 502, &format!("Hub auth poll failed: {e}")).await?;
                }
            }
        }

        ("GET", "/api/hub/profile") => {
            if state.hub_access_token().await.is_none() {
                let body = serde_json::json!({ "authenticated": false, "profile": serde_json::Value::Null });
                respond_json_value(stream, 200, &body).await?;
                return Ok(true);
            }
            match state.hub_fetch_profile().await {
                Ok(profile) => {
                    state.set_hub_profile(profile.clone()).await;
                    state.push_status().await;
                    let body = if let Some(profile) = profile {
                        serde_json::json!({ "authenticated": true, "profile": profile })
                    } else {
                        serde_json::json!({ "authenticated": false, "profile": serde_json::Value::Null })
                    };
                    respond_json_value(stream, 200, &body).await?;
                }
                Err(e) => {
                    respond_error(stream, 502, &format!("Hub profile fetch failed: {e}")).await?;
                }
            }
        }

        ("POST", "/api/hub/logout") => {
            state.set_hub_auth_pending(false).await;
            state.set_hub_access_token(None).await;
            state.set_hub_profile(None).await;
            state.set_hub_node_id(None).await;
            state
                .set_hub_link_mode("unlinked", "local", None, None, None, None)
                .await;
            state.push_status().await;
            respond_json_value(stream, 200, &serde_json::json!({ "ok": true })).await?;
        }

        ("POST", "/api/hub/leave-local") => {
            let dropped = state.leave_local_mesh_for_hub_join().await;
            state.push_status().await;
            respond_json_value(
                stream,
                200,
                &serde_json::json!({
                    "ok": true,
                    "dropped_peer_count": dropped,
                    "mode": "local",
                }),
            )
            .await?;
        }

        ("POST", "/api/hub/link-preflight") => {
            let body = parse_json_body(req).unwrap_or_default();
            let requested_hub_mesh_id = body
                .get("hub_mesh_id")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            if state.hub_access_token().await.is_none() {
                respond_error(stream, 401, "Login with InferenceHub required").await?;
                return Ok(true);
            }
            let hub_mesh_id = match state
                .hub_resolve_target_mesh_id(requested_hub_mesh_id)
                .await
            {
                Ok(id) => id,
                Err(e) => {
                    respond_error(stream, 400, &e.to_string()).await?;
                    return Ok(true);
                }
            };
            let node_id = match state.hub_ensure_registered_node().await {
                Ok(id) => id,
                Err(e) => {
                    respond_error(stream, 502, &format!("Node registration failed: {e}")).await?;
                    return Ok(true);
                }
            };
            match state.compute_link_preflight(&hub_mesh_id).await {
                Ok(mut payload) => {
                    if let Some(obj) = payload.as_object_mut() {
                        obj.insert("node_id".to_string(), serde_json::Value::String(node_id));
                    }
                    respond_json_value(stream, 200, &payload).await?;
                }
                Err(e) => {
                    respond_error(stream, 502, &format!("Preflight failed: {e}")).await?;
                }
            }
        }

        ("POST", "/api/hub/link-commit") => {
            let body = parse_json_body(req).unwrap_or_default();
            let requested_hub_mesh_id = body
                .get("hub_mesh_id")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty());
            let confirm_phrase = body
                .get("confirm_phrase")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            if confirm_phrase != "LINK MESH NOW" {
                respond_error(stream, 400, "confirm_phrase must be LINK MESH NOW").await?;
                return Ok(true);
            }
            if state.hub_access_token().await.is_none() {
                respond_error(stream, 401, "Login with InferenceHub required").await?;
                return Ok(true);
            }
            let hub_mesh_id = match state
                .hub_resolve_target_mesh_id(requested_hub_mesh_id)
                .await
            {
                Ok(id) => id,
                Err(e) => {
                    respond_error(stream, 400, &e.to_string()).await?;
                    return Ok(true);
                }
            };

            let preflight = match state.compute_link_preflight(&hub_mesh_id).await {
                Ok(v) => v,
                Err(e) => {
                    respond_error(stream, 502, &format!("Preflight failed: {e}")).await?;
                    return Ok(true);
                }
            };
            if preflight
                .get("would_block_reason")
                .and_then(|v| v.as_str())
                .is_some()
            {
                respond_json_value(stream, 409, &preflight).await?;
                return Ok(true);
            }

            let node_id = match state.hub_ensure_registered_node().await {
                Ok(id) => id,
                Err(e) => {
                    respond_error(stream, 502, &format!("Node registration failed: {e}")).await?;
                    return Ok(true);
                }
            };
            if let Err(e) = state.attach_node_to_hub_mesh(&hub_mesh_id, &node_id).await {
                respond_error(stream, 502, &format!("Attach failed: {e}")).await?;
                return Ok(true);
            }

            let (linked_mesh_slug, linked_mesh_name, linked_mesh_visibility) =
                match state.hub_fetch_mesh_metadata(&hub_mesh_id).await {
                    Ok(metadata) => metadata,
                    Err(err) => {
                        tracing::warn!("Failed to resolve linked mesh metadata: {err}");
                        (None, None, None)
                    }
                };
            state
                .set_hub_link_mode(
                    "linked",
                    "hub_enforced",
                    Some(hub_mesh_id.clone()),
                    linked_mesh_slug,
                    linked_mesh_name,
                    linked_mesh_visibility,
                )
                .await;
            state.apply_hub_identity_to_node().await;

            let hub_member_nodes = match state.hub_fetch_mesh_node_ids(&hub_mesh_id).await {
                Ok(ids) => ids,
                Err(e) => {
                    respond_error(stream, 502, &format!("Membership fetch failed: {e}")).await?;
                    return Ok(true);
                }
            };
            let node = state.inner.lock().await.node.clone();
            let peers = node.peers().await;
            let mut drop_ids = Vec::new();
            for peer in peers {
                let in_mesh = peer.hub_mesh_id.as_deref() == Some(hub_mesh_id.as_str());
                let member_ok = peer
                    .hub_node_id
                    .as_ref()
                    .map(|id| hub_member_nodes.contains(id))
                    .unwrap_or(false);
                if !(in_mesh && member_ok) {
                    drop_ids.push(peer.id);
                }
            }
            let dropped_count = drop_ids.len();
            if !drop_ids.is_empty() {
                node.drop_peers(&drop_ids).await;
            }
            state.push_status().await;
            respond_json_value(
                stream,
                200,
                &serde_json::json!({
                    "mode": "hub_enforced",
                    "hub_mesh_id": hub_mesh_id,
                    "node_id": node_id,
                    "dropped_peer_count": dropped_count,
                }),
            )
            .await?;
        }

        ("POST", "/api/hub/invites/create") => {
            if state.hub_access_token().await.is_none() {
                respond_error(stream, 401, "Login with InferenceHub required").await?;
                return Ok(true);
            }
            let hub_mesh_id = {
                let inner = state.inner.lock().await;
                inner.hub.linked_mesh_id.clone()
            };
            let Some(hub_mesh_id) = hub_mesh_id else {
                respond_error(stream, 409, "No linked InferenceHub mesh").await?;
                return Ok(true);
            };

            match state.hub_create_invite_for_linked_mesh(&hub_mesh_id).await {
                Ok((status, payload)) => {
                    respond_json_value(stream, status, &payload).await?;
                }
                Err(err) => {
                    respond_error(stream, 502, &format!("Invite creation failed: {err}")).await?;
                }
            }
        }
        _ => return Ok(false),
    }
    Ok(true)
}

impl MeshApi {
    pub async fn set_hub_first_time_onboarding(&self, enabled: bool) {
        self.inner.lock().await.hub_first_time_onboarding = enabled;
    }

    fn persist_hub_state(inner: &ApiInner) {
        let _ = save_hub_session(&HubSessionFile {
            access_token: inner.hub.access_token.clone(),
            profile: inner.hub.profile.clone(),
            default_startup_model: inner.hub.default_startup_model.clone(),
            default_mesh_selector: inner.hub.default_mesh_selector.clone(),
            default_invite_token: inner.hub.default_invite_token.clone(),
            node_id: inner.hub.node_id.clone(),
            linked_mesh_id: inner.hub.linked_mesh_id.clone(),
            linked_mesh_slug: inner.hub.linked_mesh_slug.clone(),
            linked_mesh_name: inner.hub.linked_mesh_name.clone(),
            linked_mesh_visibility: inner.hub.linked_mesh_visibility.clone(),
            link_state: Some(inner.hub.link_state.clone()),
            membership_enforcement: Some(inner.hub.membership_enforcement.clone()),
        });
    }
    async fn hub_base_url(&self) -> String {
        self.inner.lock().await.hub.base_url.clone()
    }

    async fn hub_access_token(&self) -> Option<String> {
        self.inner.lock().await.hub.access_token.clone()
    }

    async fn set_hub_auth_pending(&self, pending: bool) {
        self.inner.lock().await.hub.auth_pending = pending;
    }

    async fn set_hub_access_token(&self, token: Option<String>) {
        let mut inner = self.inner.lock().await;
        inner.hub.access_token = token;
        if inner.hub.access_token.is_none() {
            inner.hub.profile = None;
            inner.hub.last_membership_sync_ok_unix = None;
            inner.hub.runtime_access_token = None;
            inner.hub.runtime_access_token_expires_unix = None;
        }
        Self::persist_hub_state(&inner);
    }

    async fn set_hub_profile(&self, profile: Option<HubProfile>) {
        let mut inner = self.inner.lock().await;
        inner.hub.profile = profile;
        Self::persist_hub_state(&inner);
    }

    async fn set_hub_node_id(&self, node_id: Option<String>) {
        let (node, mesh_id) = {
            let mut inner = self.inner.lock().await;
            inner.hub.node_id = node_id.clone();
            if inner.hub.node_id.is_none() {
                inner.hub.runtime_access_token = None;
                inner.hub.runtime_access_token_expires_unix = None;
            }
            let mesh_id = inner.hub.linked_mesh_id.clone();
            Self::persist_hub_state(&inner);
            (inner.node.clone(), mesh_id)
        };
        node.set_hub_identity(node_id, mesh_id).await;
    }

    async fn set_hub_link_mode(
        &self,
        link_state: &str,
        membership_enforcement: &str,
        linked_mesh_id: Option<String>,
        linked_mesh_slug: Option<String>,
        linked_mesh_name: Option<String>,
        linked_mesh_visibility: Option<String>,
    ) {
        let (node, hub_node_id, hub_mesh_id) = {
            let mut inner = self.inner.lock().await;
            inner.hub.link_state = link_state.to_string();
            inner.hub.membership_enforcement = membership_enforcement.to_string();
            inner.hub.linked_mesh_id = linked_mesh_id;
            inner.hub.linked_mesh_slug = linked_mesh_slug;
            inner.hub.linked_mesh_name = linked_mesh_name;
            inner.hub.linked_mesh_visibility = linked_mesh_visibility;
            if inner.hub.membership_enforcement != "hub_enforced"
                || inner.hub.linked_mesh_id.is_none()
            {
                inner.hub.runtime_access_token = None;
                inner.hub.runtime_access_token_expires_unix = None;
            }
            inner.hub.last_membership_sync_ok_unix = if membership_enforcement == "hub_enforced" {
                Some(now_unix_secs())
            } else {
                None
            };
            Self::persist_hub_state(&inner);
            (
                inner.node.clone(),
                inner.hub.node_id.clone(),
                inner.hub.linked_mesh_id.clone(),
            )
        };
        node.set_hub_identity(hub_node_id, hub_mesh_id).await;
    }

    async fn set_hub_linked_mesh_metadata(
        &self,
        slug: Option<String>,
        name: Option<String>,
        visibility: Option<String>,
    ) {
        let mut inner = self.inner.lock().await;
        inner.hub.linked_mesh_slug = slug;
        inner.hub.linked_mesh_name = name;
        inner.hub.linked_mesh_visibility = visibility;
        Self::persist_hub_state(&inner);
    }

    async fn auto_link_default_target_on_startup(&self) -> anyhow::Result<()> {
        let (has_target, already_linked, onboarding_mode) = {
            let inner = self.inner.lock().await;
            let has_target = inner.hub.default_mesh_selector.is_some()
                || inner.hub.default_invite_token.is_some()
                || inner.hub.linked_mesh_id.is_some();
            let already_linked = inner.hub.membership_enforcement == "hub_enforced"
                && inner.hub.linked_mesh_id.is_some();
            (has_target, already_linked, inner.hub_first_time_onboarding)
        };
        if !onboarding_mode || !has_target || already_linked {
            return Ok(());
        }

        let hub_mesh_id = self.hub_resolve_target_mesh_id(None).await?;
        let preflight = self.compute_link_preflight(&hub_mesh_id).await?;
        if let Some(reason) = preflight.get("would_block_reason").and_then(|v| v.as_str()) {
            tracing::warn!(
                "Hub auto-link preflight blocked for mesh {}: {}",
                hub_mesh_id,
                reason
            );
            return Ok(());
        }

        let node_id = self.hub_ensure_registered_node().await?;
        self.attach_node_to_hub_mesh(&hub_mesh_id, &node_id).await?;

        let (linked_mesh_slug, linked_mesh_name, linked_mesh_visibility) = match self
            .hub_fetch_mesh_metadata(&hub_mesh_id)
            .await
        {
            Ok(metadata) => metadata,
            Err(err) => {
                tracing::warn!("Failed to resolve linked mesh metadata during auto-link: {err}");
                (None, None, None)
            }
        };
        self.set_hub_link_mode(
            "linked",
            "hub_enforced",
            Some(hub_mesh_id.clone()),
            linked_mesh_slug,
            linked_mesh_name,
            linked_mesh_visibility,
        )
        .await;
        self.apply_hub_identity_to_node().await;

        let hub_member_nodes = self.hub_fetch_mesh_node_ids(&hub_mesh_id).await?;
        let node = self.inner.lock().await.node.clone();
        let peers = node.peers().await;
        let mut drop_ids = Vec::new();
        for peer in peers {
            let in_mesh = peer.hub_mesh_id.as_deref() == Some(hub_mesh_id.as_str());
            let member_ok = peer
                .hub_node_id
                .as_ref()
                .map(|id| hub_member_nodes.contains(id))
                .unwrap_or(false);
            if !(in_mesh && member_ok) {
                drop_ids.push(peer.id);
            }
        }
        if !drop_ids.is_empty() {
            node.drop_peers(&drop_ids).await;
        }

        self.set_hub_first_time_onboarding(false).await;
        self.push_status().await;
        tracing::info!("Auto-linked to InferenceHub mesh {}", hub_mesh_id);
        Ok(())
    }

    async fn set_hub_default_target(
        &self,
        default_mesh_selector: Option<String>,
        default_invite_token: Option<String>,
    ) {
        let mut inner = self.inner.lock().await;
        inner.hub.default_mesh_selector = default_mesh_selector;
        inner.hub.default_invite_token = default_invite_token;
        Self::persist_hub_state(&inner);
    }

    async fn set_runtime_access_token(&self, token: Option<String>, expires_unix: Option<u64>) {
        let mut inner = self.inner.lock().await;
        inner.hub.runtime_access_token = token;
        inner.hub.runtime_access_token_expires_unix = expires_unix;
    }

    async fn next_hub_telemetry_seq(&self) -> u64 {
        let mut inner = self.inner.lock().await;
        inner.hub.telemetry_seq = inner.hub.telemetry_seq.wrapping_add(1);
        inner.hub.telemetry_seq
    }

    async fn is_hub_enforced_mode(&self) -> bool {
        let inner = self.inner.lock().await;
        inner.hub.membership_enforcement == "hub_enforced" && inner.hub.linked_mesh_id.is_some()
    }

    async fn hub_fetch_profile(&self) -> anyhow::Result<Option<HubProfile>> {
        let base_url = self.hub_base_url().await;
        let Some(token) = self.hub_access_token().await else {
            return Ok(None);
        };
        let client = reqwest::Client::new();
        let response = client
            .get(format!("{base_url}/api/profile"))
            .bearer_auth(token)
            .send()
            .await?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            self.set_hub_access_token(None).await;
            self.set_hub_auth_pending(false).await;
            self.push_status().await;
            return Ok(None);
        }
        if !response.status().is_success() {
            anyhow::bail!("hub profile request failed with {}", response.status());
        }
        let payload = response.json::<serde_json::Value>().await?;
        let name = payload
            .get("name")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let email = payload
            .get("email")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let handle = email
            .as_deref()
            .map(|e| e.split('@').next().unwrap_or_default().trim().to_string())
            .filter(|s| !s.is_empty());
        let avatar_url = payload
            .get("image")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        Ok(Some(HubProfile {
            name,
            handle,
            avatar_url,
        }))
    }

    async fn refresh_linked_mesh_metadata(&self) {
        let (linked_mesh_id, linked_mesh_slug, linked_mesh_name, linked_mesh_visibility) = {
            let inner = self.inner.lock().await;
            (
                inner.hub.linked_mesh_id.clone(),
                inner.hub.linked_mesh_slug.clone(),
                inner.hub.linked_mesh_name.clone(),
                inner.hub.linked_mesh_visibility.clone(),
            )
        };
        if linked_mesh_id.is_none()
            || (linked_mesh_slug.is_some()
                && linked_mesh_name.is_some()
                && linked_mesh_visibility.is_some())
        {
            return;
        }
        let Some(hub_mesh_id) = linked_mesh_id else {
            return;
        };
        match self.hub_fetch_mesh_metadata(&hub_mesh_id).await {
            Ok((slug, name, visibility)) => {
                self.set_hub_linked_mesh_metadata(slug, name, visibility)
                    .await;
            }
            Err(err) => {
                tracing::warn!("Failed to refresh linked mesh metadata: {err}");
            }
        }
    }

    async fn hub_fetch_mesh_metadata(
        &self,
        hub_mesh_id: &str,
    ) -> anyhow::Result<(Option<String>, Option<String>, Option<String>)> {
        let Some(token) = self.hub_access_token().await else {
            return Ok((None, None, None));
        };
        let base_url = self.hub_base_url().await;
        let client = reqwest::Client::new();
        let response = client
            .get(format!("{base_url}/api/v0/meshes/{hub_mesh_id}"))
            .bearer_auth(token)
            .send()
            .await?;
        if !response.status().is_success() {
            return Ok((None, None, None));
        }
        let payload = response.json::<serde_json::Value>().await?;
        let slug = payload
            .get("slug")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_string);
        let name = payload
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_string)
            .or_else(|| {
                payload
                    .get("ownerPrivateLabel")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .map(str::to_string)
            });
        let visibility = payload
            .get("visibility")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(str::to_string);
        Ok((slug, name, visibility))
    }

    async fn hub_registered_node_id(&self) -> Option<String> {
        self.inner.lock().await.hub.node_id.clone()
    }

    async fn hub_ensure_registered_node(&self) -> anyhow::Result<String> {
        if let Some(existing) = self.hub_registered_node_id().await {
            return Ok(existing);
        }
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let identity = load_or_create_hub_identity()?;
        let client = reqwest::Client::new();

        let challenge_resp = client
            .post(format!("{base_url}/api/v0/nodes/register"))
            .bearer_auth(&token)
            .json(&serde_json::json!({ "public_key": identity.public_key_b64 }))
            .send()
            .await?;
        if !challenge_resp.status().is_success() {
            anyhow::bail!(
                "node register challenge failed with {}",
                challenge_resp.status()
            );
        }
        let challenge_payload = challenge_resp.json::<serde_json::Value>().await?;
        let challenge_id = challenge_payload
            .get("challenge_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let nonce = challenge_payload
            .get("challenge_nonce")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if challenge_id.is_empty() || nonce.is_empty() {
            anyhow::bail!("node register challenge payload missing required fields");
        }

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let signed_message = format!(
            "inferencehub-v1\npurpose=node_register\nchallenge_id={challenge_id}\nnonce={nonce}\nnode_id=\nts={ts}"
        );
        let signature_b64 = sign_hub_message(&identity, &signed_message)?;
        let hostname = default_node_display_name();
        let complete_resp = client
            .post(format!("{base_url}/api/v0/nodes/register/complete"))
            .bearer_auth(&token)
            .json(&serde_json::json!({
                "challenge_id": challenge_id,
                "public_key": identity.public_key_b64,
                "signature": signature_b64,
                "signed_message": signed_message,
                "display_name": hostname,
                "client": {
                    "version": crate::VERSION,
                    "platform": std::env::consts::OS,
                }
            }))
            .send()
            .await?;
        if !complete_resp.status().is_success() {
            anyhow::bail!(
                "node register complete failed with {}",
                complete_resp.status()
            );
        }
        let complete_payload = complete_resp.json::<serde_json::Value>().await?;
        let node_id = complete_payload
            .get("node_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if node_id.is_empty() {
            anyhow::bail!("node register response missing node_id");
        }
        self.set_hub_node_id(Some(node_id.clone())).await;
        Ok(node_id)
    }

    async fn hub_fetch_mesh_node_ids(&self, hub_mesh_id: &str) -> anyhow::Result<HashSet<String>> {
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let client = reqwest::Client::new();
        let response = client
            .get(format!("{base_url}/api/v0/meshes/{hub_mesh_id}/nodes"))
            .bearer_auth(token)
            .send()
            .await?;
        if !response.status().is_success() {
            anyhow::bail!("mesh nodes fetch failed with {}", response.status());
        }
        let payload = response.json::<serde_json::Value>().await?;
        let ids = payload
            .get("items")
            .and_then(|v| v.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        item.get("node_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .collect::<HashSet<String>>()
            })
            .unwrap_or_default();
        Ok(ids)
    }

    async fn hub_resolve_mesh_id_by_selector(&self, selector: &str) -> anyhow::Result<String> {
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let selector = selector.trim();
        if selector.is_empty() {
            anyhow::bail!("empty mesh selector");
        }
        let client = reqwest::Client::new();

        // First try as mesh ID.
        let by_id_resp = client
            .get(format!("{base_url}/api/v0/meshes/{selector}"))
            .bearer_auth(&token)
            .send()
            .await?;
        if by_id_resp.status().is_success() {
            let payload = by_id_resp.json::<serde_json::Value>().await?;
            let id = payload
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .unwrap_or_default()
                .to_string();
            if !id.is_empty() {
                return Ok(id);
            }
        }

        let list_resp = client
            .get(format!("{base_url}/api/v0/meshes"))
            .bearer_auth(token)
            .send()
            .await?;
        if !list_resp.status().is_success() {
            anyhow::bail!("mesh list request failed with {}", list_resp.status());
        }
        let payload = list_resp.json::<serde_json::Value>().await?;
        let Some(items) = payload.as_array() else {
            anyhow::bail!("mesh list response format was invalid");
        };

        let selector_lower = selector.to_lowercase();
        let mut matches: Vec<String> = items
            .iter()
            .filter_map(|item| {
                let id = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|v| !v.is_empty())?;
                let slug = item
                    .get("slug")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                let name = item
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());

                if id == selector
                    || slug
                        .as_deref()
                        .map(|s| s.eq_ignore_ascii_case(selector))
                        .unwrap_or(false)
                    || name
                        .as_deref()
                        .map(|s| s.to_lowercase() == selector_lower)
                        .unwrap_or(false)
                {
                    Some(id.to_string())
                } else {
                    None
                }
            })
            .collect();

        matches.sort();
        matches.dedup();
        match matches.as_slice() {
            [single] => Ok(single.clone()),
            [] => anyhow::bail!("no InferenceHub mesh matched '{selector}'"),
            _ => anyhow::bail!("multiple meshes matched '{selector}'. Use an exact mesh ID."),
        }
    }

    async fn hub_redeem_invite_for_mesh_id(&self, invite_token: &str) -> anyhow::Result<String> {
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{base_url}/api/v0/invites/redeem"))
            .bearer_auth(token)
            .json(&serde_json::json!({
                "invite_token": invite_token,
            }))
            .send()
            .await?;
        if !response.status().is_success() {
            let status = response.status();
            let payload = response.text().await.unwrap_or_default();
            anyhow::bail!("invite redeem failed with {status}: {payload}");
        }
        let payload = response.json::<serde_json::Value>().await?;
        let hub_mesh_id = payload
            .get("hub_mesh_id")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or_default()
            .to_string();
        if hub_mesh_id.is_empty() {
            anyhow::bail!("invite redeem response missing hub_mesh_id");
        }
        Ok(hub_mesh_id)
    }

    async fn hub_create_invite_for_linked_mesh(
        &self,
        hub_mesh_id: &str,
    ) -> anyhow::Result<(u16, serde_json::Value)> {
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let client = reqwest::Client::new();
        let response = client
            .post(format!("{base_url}/api/v0/meshes/{hub_mesh_id}/invites"))
            .bearer_auth(token)
            .json(&serde_json::json!({}))
            .send()
            .await?;
        let status = response.status().as_u16();
        let payload = response.json::<serde_json::Value>().await.unwrap_or_else(
            |_| serde_json::json!({ "error": "failed to parse hub invite response" }),
        );
        Ok((status, payload))
    }

    async fn hub_resolve_target_mesh_id(
        &self,
        requested_hub_mesh_id: Option<String>,
    ) -> anyhow::Result<String> {
        if let Some(id) = requested_hub_mesh_id
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
        {
            return Ok(id);
        }

        let (default_mesh_selector, default_invite_token, linked_mesh_id) = {
            let inner = self.inner.lock().await;
            (
                inner.hub.default_mesh_selector.clone(),
                inner.hub.default_invite_token.clone(),
                inner.hub.linked_mesh_id.clone(),
            )
        };

        if let Some(invite) = default_invite_token.as_deref() {
            let hub_mesh_id = self.hub_redeem_invite_for_mesh_id(invite).await?;
            self.set_hub_default_target(Some(hub_mesh_id.clone()), None)
                .await;
            return Ok(hub_mesh_id);
        }

        if let Some(selector) = default_mesh_selector.as_deref() {
            return self.hub_resolve_mesh_id_by_selector(selector).await;
        }

        if let Some(linked) = linked_mesh_id {
            return Ok(linked);
        }

        anyhow::bail!(
            "hub_mesh_id is required (or pass --hub-mesh / --hub-invite when starting mesh-llm)"
        )
    }

    async fn hub_telemetry_snapshot(
        &self,
        hub_mesh_id: &str,
        node_id: &str,
    ) -> anyhow::Result<HubTelemetrySnapshot> {
        let node = self.inner.lock().await.node.clone();
        let peers = node.peers().await;
        let linked_peers: Vec<_> = peers
            .into_iter()
            .filter(|peer| peer.hub_mesh_id.as_deref() == Some(hub_mesh_id))
            .collect();

        let node_total_vram_gb = node.vram_bytes() as f64 / 1e9;
        let peer_total_vram_gb = linked_peers
            .iter()
            .map(|peer| peer.vram_bytes as f64 / 1e9)
            .sum::<f64>();
        let node_inflight_requests = node.inflight_requests();
        let node_requests_total = node.total_inference_requests();
        let node_requests_failed_total = node.failed_inference_requests();
        let node_output_tokens_total = node.total_output_tokens();
        let node_input_tokens_total = node.total_input_tokens();
        let mesh_avg_ttft_ms = node.average_inference_latency_ms();
        let node_warm_models = node
            .serving()
            .await
            .map(|name| vec![name])
            .unwrap_or_default();

        Ok(HubTelemetrySnapshot {
            node_id: node_id.to_string(),
            hub_mesh_id: hub_mesh_id.to_string(),
            active_nodes: 1 + linked_peers.len() as u64,
            total_vram_gb: node_total_vram_gb + peer_total_vram_gb,
            node_total_vram_gb,
            node_available_vram_gb: node_total_vram_gb,
            node_gpu_count: if node_total_vram_gb > 0.0 { 1 } else { 0 },
            node_inflight_requests,
            node_requests_total,
            node_requests_failed_total,
            node_input_tokens_total,
            node_output_tokens_total,
            mesh_avg_ttft_ms,
            node_warm_models,
            mesh_warm_models: node.models_being_served().await,
        })
    }

    async fn hub_ensure_runtime_access_token(&self, node_id: &str) -> anyhow::Result<String> {
        let now = now_unix_secs();
        {
            let inner = self.inner.lock().await;
            if let (Some(token), Some(expires_unix)) = (
                inner.hub.runtime_access_token.clone(),
                inner.hub.runtime_access_token_expires_unix,
            ) {
                if expires_unix > now + HUB_RUNTIME_SESSION_REFRESH_BEFORE_SECS {
                    return Ok(token);
                }
            }
        }

        let base_url = self.hub_base_url().await;
        let identity = load_or_create_hub_identity()?;
        let client = reqwest::Client::new();
        let challenge_resp = client
            .post(format!(
                "{base_url}/api/v0/nodes/{node_id}/session/challenge"
            ))
            .send()
            .await?;
        if !challenge_resp.status().is_success() {
            let status = challenge_resp.status();
            let body = challenge_resp.text().await.unwrap_or_default();
            anyhow::bail!("runtime challenge failed with {status}: {body}");
        }
        let challenge_payload = challenge_resp.json::<serde_json::Value>().await?;
        let challenge_id = challenge_payload
            .get("challenge_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let nonce = challenge_payload
            .get("challenge_nonce")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if challenge_id.is_empty() || nonce.is_empty() {
            anyhow::bail!("runtime challenge payload missing required fields");
        }

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let signed_message = format!(
            "inferencehub-v1\npurpose=node_session\nchallenge_id={challenge_id}\nnonce={nonce}\nnode_id={node_id}\nts={ts}"
        );
        let signature_b64 = sign_hub_message(&identity, &signed_message)?;

        let session_resp = client
            .post(format!("{base_url}/api/v0/nodes/{node_id}/session"))
            .json(&serde_json::json!({
                "challenge_id": challenge_id,
                "signature": signature_b64,
                "signed_message": signed_message,
            }))
            .send()
            .await?;
        if !session_resp.status().is_success() {
            let status = session_resp.status();
            let body = session_resp.text().await.unwrap_or_default();
            anyhow::bail!("runtime session exchange failed with {status}: {body}");
        }
        let session_payload = session_resp.json::<serde_json::Value>().await?;
        let access_token = session_payload
            .get("access_token")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if access_token.is_empty() {
            anyhow::bail!("runtime session response missing access_token");
        }
        self.set_runtime_access_token(
            Some(access_token.clone()),
            Some(now + HUB_RUNTIME_SESSION_TTL_SECS),
        )
        .await;
        Ok(access_token)
    }

    async fn hub_post_runtime_json(
        &self,
        node_id: &str,
        path: &str,
        body: &serde_json::Value,
    ) -> anyhow::Result<()> {
        let base_url = self.hub_base_url().await;
        let url = format!("{base_url}{path}");
        let client = reqwest::Client::new();

        for attempt in 0..2u8 {
            let token = self.hub_ensure_runtime_access_token(node_id).await?;
            let response = client
                .post(&url)
                .bearer_auth(&token)
                .json(body)
                .send()
                .await?;
            if response.status() == reqwest::StatusCode::UNAUTHORIZED {
                self.set_runtime_access_token(None, None).await;
                if attempt == 0 {
                    continue;
                }
            }
            if !response.status().is_success() {
                let status = response.status();
                let payload = response.text().await.unwrap_or_default();
                anyhow::bail!("hub telemetry post {path} failed with {status}: {payload}");
            }
            return Ok(());
        }

        anyhow::bail!("hub telemetry auth failed for {path}")
    }

    async fn publish_hub_telemetry_once(&self) -> anyhow::Result<()> {
        let (node_id, hub_mesh_id, should_publish) = {
            let inner = self.inner.lock().await;
            (
                inner.hub.node_id.clone(),
                inner.hub.linked_mesh_id.clone(),
                inner.hub.membership_enforcement == "hub_enforced",
            )
        };
        if !should_publish {
            return Ok(());
        }
        let Some(node_id) = node_id else {
            return Ok(());
        };
        let Some(hub_mesh_id) = hub_mesh_id else {
            return Ok(());
        };

        let telemetry = self.hub_telemetry_snapshot(&hub_mesh_id, &node_id).await?;
        let hostname = default_node_display_name();
        let report_ts_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let telemetry_seq = self.next_hub_telemetry_seq().await;
        let interval_ms = HUB_TELEMETRY_SYNC_INTERVAL_SECS * 1000;

        self.hub_post_runtime_json(
            &node_id,
            &format!("/api/v0/nodes/{node_id}/heartbeat"),
            &serde_json::json!({}),
        )
        .await?;
        self.hub_post_runtime_json(
            &node_id,
            &format!("/api/v0/nodes/{node_id}/capabilities"),
            &serde_json::json!({
                "capabilities": {
                    "total_vram_gb": telemetry.node_total_vram_gb,
                    "available_vram_gb": telemetry.node_available_vram_gb,
                    "gpu_count": telemetry.node_gpu_count,
                    "hostname": hostname,
                    "platform": std::env::consts::OS,
                    "client_version": crate::VERSION,
                }
            }),
        )
        .await?;
        self.hub_post_runtime_json(
            &node_id,
            &format!("/api/v0/nodes/{node_id}/metrics"),
            &serde_json::json!({
                "hub_mesh_id": telemetry.hub_mesh_id,
                "metrics": {
                    "node_total_vram_gb": telemetry.node_total_vram_gb,
                    "node_warm_model_count": telemetry.node_warm_models.len(),
                    "active_nodes": telemetry.active_nodes,
                    "total_vram_gb": telemetry.total_vram_gb,
                    "warm_model_count": telemetry.mesh_warm_models.len(),
                    "avg_ttft_ms": telemetry.mesh_avg_ttft_ms,
                }
            }),
        )
        .await?;
        self.hub_post_runtime_json(
            &node_id,
            &format!("/api/v0/nodes/{node_id}/models"),
            &serde_json::json!({
                "hub_mesh_id": telemetry.hub_mesh_id,
                "warm_models": telemetry.node_warm_models,
            }),
        )
        .await?;

        let otlp_payload = serde_json::json!({
            "resourceMetrics": [{
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "mesh-llm"}},
                        {"key": "service.version", "value": {"stringValue": MESH_LLM_VERSION}},
                        {"key": "host.name", "value": {"stringValue": hostname}},
                        {"key": "inferencehub.node_id", "value": {"stringValue": telemetry.node_id}},
                        {"key": "inferencehub.mesh_id", "value": {"stringValue": telemetry.hub_mesh_id}},
                        {"key": "inferencehub.report_ts_unix_ms", "value": {"intValue": report_ts_unix_ms.to_string()}},
                        {"key": "inferencehub.seq", "value": {"intValue": telemetry_seq.to_string()}},
                        {"key": "inferencehub.interval_ms", "value": {"intValue": interval_ms.to_string()}},
                    ]
                },
                "scopeMetrics": [{
                    "metrics": [
                        {
                            "name": "inferencehub.node.total_vram_gb",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.node_total_vram_gb }] }
                        },
                        {
                            "name": "inferencehub.node.available_vram_gb",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.node_available_vram_gb }] }
                        },
                        {
                            "name": "inferencehub.node.gpu_count",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.node_gpu_count }] }
                        },
                        {
                            "name": "inferencehub.node.warm_model_count",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.node_warm_models.len() }] }
                        },
                        {
                            "name": "inferencehub.node.inflight_requests",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.node_inflight_requests }] }
                        },
                        {
                            "name": "inferencehub.node.requests_total",
                            "sum": { "dataPoints": [{ "asDouble": telemetry.node_requests_total }] }
                        },
                        {
                            "name": "inferencehub.node.requests_failed_total",
                            "sum": { "dataPoints": [{ "asDouble": telemetry.node_requests_failed_total }] }
                        },
                        {
                            "name": "inferencehub.node.input_tokens_total",
                            "sum": { "dataPoints": [{ "asDouble": telemetry.node_input_tokens_total }] }
                        },
                        {
                            "name": "inferencehub.node.output_tokens_total",
                            "sum": { "dataPoints": [{ "asDouble": telemetry.node_output_tokens_total }] }
                        },
                        {
                            "name": "inferencehub.node.heartbeat",
                            "gauge": { "dataPoints": [{ "asDouble": 1 }] }
                        },
                        {
                            "name": "inferencehub.mesh.active_nodes",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.active_nodes }] }
                        },
                        {
                            "name": "inferencehub.mesh.total_vram_gb",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.total_vram_gb }] }
                        },
                        {
                            "name": "inferencehub.mesh.warm_model_count",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.mesh_warm_models.len() }] }
                        },
                        {
                            "name": "inferencehub.mesh.avg_ttft_ms",
                            "gauge": { "dataPoints": [{ "asDouble": telemetry.mesh_avg_ttft_ms }] }
                        }
                    ]
                }]
            }]
        });
        self.hub_post_runtime_json(&node_id, "/api/v0/oltp/v1/metrics", &otlp_payload)
            .await?;

        Ok(())
    }

    async fn attach_node_to_hub_mesh(
        &self,
        hub_mesh_id: &str,
        node_id: &str,
    ) -> anyhow::Result<()> {
        let Some(token) = self.hub_access_token().await else {
            anyhow::bail!("not authenticated with InferenceHub");
        };
        let base_url = self.hub_base_url().await;
        let node = self.inner.lock().await.node.clone();
        let local_mesh_id = node.mesh_id().await;
        let client = reqwest::Client::new();
        let response = client
            .post(format!(
                "{base_url}/api/v0/meshes/{hub_mesh_id}/nodes/attach"
            ))
            .bearer_auth(token)
            .json(&serde_json::json!({
                "node_id": node_id,
                "local_mesh_id": local_mesh_id,
                "local_mesh_label": std::env::var("HOSTNAME").ok(),
            }))
            .send()
            .await?;
        if !response.status().is_success() {
            anyhow::bail!("node attach failed with {}", response.status());
        }
        Ok(())
    }

    async fn compute_link_preflight(&self, hub_mesh_id: &str) -> anyhow::Result<serde_json::Value> {
        let node = self.inner.lock().await.node.clone();
        let hub_member_nodes = self.hub_fetch_mesh_node_ids(hub_mesh_id).await?;
        let peers = node.peers().await;
        let mut peer_verified = 0usize;
        let mut peer_unverified = 0usize;
        let mut verified_host_capable_count = 0usize;
        let mut drop_ids = Vec::new();

        for peer in &peers {
            let in_mesh = peer.hub_mesh_id.as_deref() == Some(hub_mesh_id);
            let member_ok = peer
                .hub_node_id
                .as_ref()
                .map(|id| hub_member_nodes.contains(id))
                .unwrap_or(false);
            let verified = in_mesh && member_ok;
            if verified {
                peer_verified += 1;
                if !matches!(peer.role, mesh::NodeRole::Client) {
                    verified_host_capable_count += 1;
                }
            } else {
                peer_unverified += 1;
                drop_ids.push(peer.id);
            }
        }

        let self_host_capable = {
            let inner = self.inner.lock().await;
            !inner.is_client && inner.node.vram_bytes() > 0
        };
        let would_block_reason = if verified_host_capable_count == 0 && !self_host_capable {
            Some("No verified host-capable peers and this node cannot host".to_string())
        } else {
            None
        };

        Ok(serde_json::json!({
            "hub_mesh_id": hub_mesh_id,
            "peer_total": peers.len(),
            "peer_verified": peer_verified,
            "peer_unverified": peer_unverified,
            "verified_host_capable_count": verified_host_capable_count,
            "self_host_capable": self_host_capable,
            "would_block_reason": would_block_reason,
            "drop_peer_ids": drop_ids.iter().map(|id| id.fmt_short().to_string()).collect::<Vec<_>>(),
        }))
    }

    async fn leave_local_mesh_for_hub_join(&self) -> usize {
        let node = self.inner.lock().await.node.clone();
        let peers = node.peers().await;
        let drop_ids: Vec<_> = peers.into_iter().map(|p| p.id).collect();
        let dropped = drop_ids.len();
        if !drop_ids.is_empty() {
            node.drop_peers(&drop_ids).await;
        }
        self.set_hub_link_mode("unlinked", "local", None, None, None, None)
            .await;
        dropped
    }

    async fn enforce_hub_membership_once(&self) -> (usize, bool) {
        let (hub_mesh_id, should_enforce) = {
            let inner = self.inner.lock().await;
            (
                inner.hub.linked_mesh_id.clone(),
                inner.hub.membership_enforcement == "hub_enforced",
            )
        };
        if !should_enforce {
            return (0, false);
        }
        let Some(hub_mesh_id) = hub_mesh_id else {
            return (0, false);
        };

        let now = now_unix_secs();
        let hub_member_nodes = match self.hub_fetch_mesh_node_ids(&hub_mesh_id).await {
            Ok(ids) => {
                let mut status_changed = false;
                {
                    let mut inner = self.inner.lock().await;
                    inner.hub.last_membership_sync_ok_unix = Some(now);
                    if inner.hub.link_state != "linked" {
                        inner.hub.link_state = "linked".to_string();
                        Self::persist_hub_state(&inner);
                        status_changed = true;
                    }
                }
                if status_changed {
                    self.push_status().await;
                }
                ids
            }
            Err(err) => {
                let (node, should_block, status_changed, age_secs) = {
                    let mut inner = self.inner.lock().await;
                    if inner.hub.last_membership_sync_ok_unix.is_none() {
                        inner.hub.last_membership_sync_ok_unix = Some(now);
                    }
                    let last_ok = inner.hub.last_membership_sync_ok_unix.unwrap_or(now);
                    let age = now.saturating_sub(last_ok);
                    let block = age > (HUB_MEMBERSHIP_TTL_SECS + HUB_OUTAGE_GRACE_SECS);
                    let mut changed = false;
                    if inner.hub.link_state != "degraded" {
                        inner.hub.link_state = "degraded".to_string();
                        Self::persist_hub_state(&inner);
                        changed = true;
                    }
                    (inner.node.clone(), block, changed, age)
                };
                tracing::warn!(
                    "Hub membership sync failed (age={}s, ttl={}s, grace={}s): {}",
                    age_secs,
                    HUB_MEMBERSHIP_TTL_SECS,
                    HUB_OUTAGE_GRACE_SECS,
                    err
                );
                if should_block {
                    let peers = node.peers().await;
                    let drop_ids: Vec<_> = peers.into_iter().map(|p| p.id).collect();
                    let dropped = drop_ids.len();
                    if !drop_ids.is_empty() {
                        node.drop_peers(&drop_ids).await;
                    }
                    return (dropped, true);
                }
                return (0, status_changed);
            }
        };

        let node = self.inner.lock().await.node.clone();
        let peers = node.peers().await;
        let mut drop_ids = Vec::new();
        for peer in peers {
            let in_mesh = peer.hub_mesh_id.as_deref() == Some(hub_mesh_id.as_str());
            let member_ok = peer
                .hub_node_id
                .as_ref()
                .map(|id| hub_member_nodes.contains(id))
                .unwrap_or(false);
            if !(in_mesh && member_ok) {
                drop_ids.push(peer.id);
            }
        }
        let dropped = drop_ids.len();
        if !drop_ids.is_empty() {
            node.drop_peers(&drop_ids).await;
        }
        (dropped, dropped > 0)
    }

    async fn apply_hub_identity_to_node(&self) {
        let (node, hub_node_id, hub_mesh_id) = {
            let inner = self.inner.lock().await;
            (
                inner.node.clone(),
                inner.hub.node_id.clone(),
                inner.hub.linked_mesh_id.clone(),
            )
        };
        node.set_hub_identity(hub_node_id, hub_mesh_id).await;
    }
}
