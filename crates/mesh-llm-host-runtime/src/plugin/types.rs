use super::proto;
use super::startup::PluginStartupSummary;
use super::{PluginWebUiManifestOverview, PluginWebUiState};
use crate::logging::policy::is_http_url;
use serde::Serialize;
use std::future::Future;
use std::pin::Pin;
use tokio::sync::oneshot;

#[derive(Debug)]
pub enum PluginMeshEvent {
    Channel {
        plugin_id: String,
        message: proto::ChannelMessage,
    },
    BulkTransfer {
        plugin_id: String,
        message: proto::BulkTransferMessage,
    },
    OpenStream {
        plugin_id: String,
        request: proto::OpenMeshStreamRequest,
        response_tx: oneshot::Sender<Result<proto::OpenMeshStreamResponse, proto::ErrorResponse>>,
    },
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ToolSummary {
    pub name: String,
    pub description: String,
    pub input_schema_json: String,
}

#[derive(Clone, Debug)]
pub struct ToolCallResult {
    pub content_json: String,
    pub is_error: bool,
}

#[derive(Clone, Debug)]
pub struct RpcResult {
    pub result_json: String,
}

pub(crate) type BridgeFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

pub trait PluginRpcBridge: Send + Sync {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>>;

    fn handle_notification(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<()>;
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginSummary {
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest: Option<PluginManifestOverview>,
    #[serde(default)]
    pub web_ui: PluginWebUiState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub startup: Option<PluginStartupSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginManifestOverview {
    pub operations: usize,
    pub resources: usize,
    pub resource_templates: usize,
    pub prompts: usize,
    pub completions: usize,
    pub http_bindings: usize,
    pub endpoints: usize,
    pub mesh_channels: usize,
    pub mesh_event_subscriptions: usize,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_ui: Option<PluginWebUiManifestOverview>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginEndpointSummary {
    pub plugin_name: String,
    pub plugin_status: String,
    pub endpoint_id: String,
    pub state: String,
    pub available: bool,
    pub kind: String,
    pub transport_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protocol: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    pub supports_streaming: bool,
    pub managed_by_plugin: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub models: Vec<String>,
}

impl PluginEndpointSummary {
    /// Return a copy safe to serialize across a network boundary.
    ///
    /// A plugin endpoint `address` can embed credentials directly in an HTTP
    /// URL (`https://user:pass@host/...` userinfo, or `?api_key=...` query),
    /// and the health `detail` string echoes the probed URL. These summaries
    /// are served by the management API, which can be bound to a non-loopback
    /// interface, so any credential in the address would be an unauthenticated
    /// network read. Redact userinfo and sensitive query params from HTTP(S)
    /// addresses and from any URL embedded in the detail before the summary
    /// leaves the box. Non-HTTP addresses (stdio commands, unix sockets, named
    /// pipes) are left untouched — they carry no URL credentials and must stay
    /// verbatim for local consumers that reconstruct the transport.
    pub(crate) fn redacted_for_network(mut self) -> Self {
        if let Some(address) = self.address.as_deref()
            && is_http_url(address)
        {
            self.address = Some(crate::logging::policy::redact_url_query(address));
        }
        if let Some(detail) = self.detail.as_deref() {
            self.detail = Some(crate::logging::policy::redact_urls_in_text(detail));
        }
        self
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginCapabilityProvider {
    pub capability: String,
    pub plugin_name: String,
    pub plugin_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_id: Option<String>,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl PluginCapabilityProvider {
    /// Return a copy safe to serialize across a network boundary.
    ///
    /// Provider health details can echo the HTTP endpoint URL that was
    /// probed, including URL userinfo or sensitive query parameters. The
    /// provider projection has no address of its own, so only the embedded
    /// URLs in `detail` need redaction.
    pub(crate) fn redacted_for_network(mut self) -> Self {
        if let Some(detail) = self.detail.as_deref() {
            self.detail = Some(crate::logging::policy::redact_urls_in_text(detail));
        }
        self
    }
}

#[derive(Clone, Debug)]
pub struct InferenceEndpointRoute {
    pub strip_caller_credentials: bool,
    pub plugin_name: String,
    pub endpoint_id: String,
    pub address: String,
    pub models: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn summary_with(address: Option<&str>, detail: Option<&str>) -> PluginEndpointSummary {
        PluginEndpointSummary {
            plugin_name: "demo".into(),
            plugin_status: "running".into(),
            endpoint_id: "embed".into(),
            state: "healthy".into(),
            available: true,
            kind: "openai".into(),
            transport_kind: "http".into(),
            protocol: None,
            address: address.map(str::to_string),
            args: Vec::new(),
            namespace: None,
            supports_streaming: false,
            managed_by_plugin: false,
            detail: detail.map(str::to_string),
            models: Vec::new(),
        }
    }

    #[test]
    fn http_address_userinfo_and_query_are_redacted_for_network() {
        let redacted = summary_with(
            Some("https://alice:s3cret@host:8000/v1?api_key=abc123"),
            None,
        )
        .redacted_for_network();
        let address = redacted.address.expect("address present");
        assert!(
            !address.contains("alice:s3cret"),
            "userinfo leaked: {address}"
        );
        assert!(!address.contains("abc123"), "api_key leaked: {address}");
        assert!(address.contains("host:8000"), "host lost: {address}");
    }

    #[test]
    fn health_detail_url_credentials_are_redacted_for_network() {
        let redacted = summary_with(None, Some("GET https://bob:pw@host/v1/models -> 200 OK"))
            .redacted_for_network();
        let detail = redacted.detail.expect("detail present");
        assert!(
            !detail.contains("bob:pw"),
            "detail userinfo leaked: {detail}"
        );
    }

    #[test]
    fn uppercase_scheme_address_is_still_redacted_for_network() {
        // URL schemes are case-insensitive (RFC 3986); an uppercase scheme
        // must not bypass credential redaction.
        let redacted = summary_with(
            Some("HTTPS://alice:s3cret@host:8000/v1?api_key=abc123"),
            None,
        )
        .redacted_for_network();
        let address = redacted.address.expect("address present");
        assert!(
            !address.contains("alice:s3cret"),
            "userinfo leaked for uppercase scheme: {address}"
        );
        assert!(
            !address.contains("abc123"),
            "api_key leaked for uppercase scheme: {address}"
        );
    }

    #[test]
    fn non_http_address_is_left_verbatim_for_local_transport() {
        // stdio command addresses must survive intact so MCP transports can be
        // reconstructed locally; there is no URL credential to redact.
        let address = "/usr/local/bin/my-mcp-server --flag value";
        let redacted = summary_with(Some(address), None).redacted_for_network();
        assert_eq!(redacted.address.as_deref(), Some(address));
    }

    #[test]
    fn provider_health_detail_url_credentials_are_redacted_for_network() {
        let provider = PluginCapabilityProvider {
            capability: "chat".into(),
            plugin_name: "demo".into(),
            plugin_status: "running".into(),
            endpoint_id: Some("chat".into()),
            available: true,
            detail: Some("GET https://alice:s3cret@host/v1/models?api_key=abc123 -> 200 OK".into()),
        }
        .redacted_for_network();
        let detail = provider.detail.expect("detail present");
        assert!(
            !detail.contains("alice:s3cret"),
            "userinfo leaked: {detail}"
        );
        assert!(!detail.contains("abc123"), "api_key leaked: {detail}");
        assert!(detail.contains("host"), "host lost: {detail}");
    }

    #[test]
    fn punctuation_delimited_detail_url_credentials_are_redacted_for_network() {
        let provider = PluginCapabilityProvider {
            capability: "chat".into(),
            plugin_name: "demo".into(),
            plugin_status: "running".into(),
            endpoint_id: Some("chat".into()),
            available: true,
            detail: Some("GET (https://alice:s3cret@host/v1) -> 200 OK".into()),
        }
        .redacted_for_network();
        let detail = provider.detail.expect("detail present");
        assert!(
            !detail.contains("alice:s3cret"),
            "userinfo leaked: {detail}"
        );
        assert!(
            detail.contains("(https://[REDACTED]@host/v1)"),
            "URL punctuation lost: {detail}"
        );
    }
}
