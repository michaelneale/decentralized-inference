use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

pub use mesh_llm_embedded_runtime::{
    EmbeddedMeshAdmissionConfig, EmbeddedMeshDiscoveryMode, EmbeddedMeshHttpConfig,
    EmbeddedMeshLogFormat, EmbeddedMeshNetworkConfig, EmbeddedMeshNodeConfig, EmbeddedMeshNodeMode,
    EmbeddedMeshRequirementsConfig, EmbeddedMeshServingConfig, EmbeddedMeshStorageConfig,
    EmbeddedTrustPolicy, SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION,
};

pub type MeshNodeStatus = mesh_llm_embedded_runtime::EmbeddedMeshNodeStatus;

pub struct MeshNode {
    handle: mesh_llm_embedded_runtime::EmbeddedMeshNodeHandle,
}

impl MeshNode {
    pub fn builder() -> MeshNodeBuilder {
        MeshNodeBuilder::default()
    }

    pub fn api_base_url(&self) -> &str {
        self.handle.api_base_url()
    }

    pub fn console_url(&self) -> &str {
        self.handle.console_url()
    }

    pub fn invite_token(&self) -> Option<&str> {
        self.handle.invite_token()
    }

    pub fn openai_client(&self) -> OpenAiClient {
        OpenAiClient::new(self.api_base_url())
    }

    pub async fn status(&self) -> anyhow::Result<MeshNodeStatus> {
        self.handle.status().await
    }

    pub async fn shutdown(self) -> anyhow::Result<()> {
        self.handle.stop().await
    }

    pub async fn stop(self) -> anyhow::Result<()> {
        self.shutdown().await
    }

    pub fn into_inner(self) -> mesh_llm_embedded_runtime::EmbeddedMeshNodeHandle {
        self.handle
    }
}

#[derive(Clone, Debug, Default)]
pub struct MeshNodeBuilder {
    inner: mesh_llm_embedded_runtime::EmbeddedMeshNodeBuilder,
}

impl MeshNodeBuilder {
    pub fn mode(mut self, mode: EmbeddedMeshNodeMode) -> Self {
        self.inner = self.inner.mode(mode);
        self
    }

    pub fn serve(mut self) -> Self {
        self.inner = self.inner.serve();
        self
    }

    /// Serve one already-running OpenAI-compatible HTTP server instead of a
    /// local model. This node forwards to `address` and never loads a native
    /// inference runtime.
    ///
    /// Fixed for the node's lifetime: stop the node to stop sharing. The
    /// upstream server is never started or stopped by Mesh.
    pub fn share_endpoint(mut self, address: impl Into<String>) -> Self {
        self.inner = self.inner.share_endpoint(address);
        self
    }

    pub fn client(mut self) -> Self {
        self.inner = self.inner.client();
        self
    }

    pub fn model(mut self, model_ref: impl Into<String>) -> Self {
        self.inner = self.inner.model(model_ref);
        self
    }

    pub fn models<I, S>(mut self, model_refs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.models(model_refs);
        self
    }

    pub fn max_vram_gb(mut self, max_vram_gb: f64) -> Self {
        self.inner = self.inner.max_vram_gb(max_vram_gb);
        self
    }

    pub fn api_port(mut self, port: u16) -> Self {
        self.inner = self.inner.api_port(port);
        self
    }

    pub fn console_port(mut self, port: u16) -> Self {
        self.inner = self.inner.console_port(port);
        self
    }

    pub fn console_ui(mut self, enabled: bool) -> Self {
        self.inner = self.inner.console_ui(enabled);
        self
    }

    pub fn join_token(mut self, token: impl Into<String>) -> Self {
        self.inner = self.inner.join_token(token);
        self
    }

    pub fn join_tokens<I, S>(mut self, tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.join_tokens(tokens);
        self
    }

    pub fn auto_join(mut self, enabled: bool) -> Self {
        self.inner = self.inner.auto_join(enabled);
        self
    }

    pub fn auto_join_public_mesh(mut self) -> Self {
        self.inner = self
            .inner
            .auto_join(true)
            .discovery_mode(EmbeddedMeshDiscoveryMode::Nostr);
        self
    }

    pub fn discovery_mode(mut self, mode: EmbeddedMeshDiscoveryMode) -> Self {
        self.inner = self.inner.discovery_mode(mode);
        self
    }

    pub fn publish(mut self, enabled: bool) -> Self {
        self.inner = self.inner.publish(enabled);
        self
    }

    pub fn mesh_name(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.mesh_name(name);
        self
    }

    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.inner = self.inner.region(region);
        self
    }

    pub fn node_name(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.node_name(name);
        self
    }

    pub fn iroh_relay(mut self, url: impl Into<String>) -> Self {
        self.inner = self.inner.iroh_relay(url);
        self
    }

    pub fn iroh_relays<I, S>(mut self, urls: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.iroh_relays(urls);
        self
    }

    pub fn iroh_relay_auth(
        mut self,
        relay_url: impl Into<String>,
        bearer_token: impl Into<String>,
    ) -> Self {
        self.inner = self.inner.iroh_relay_auth(relay_url, bearer_token);
        self
    }

    pub fn nostr_relay(mut self, url: impl Into<String>) -> Self {
        self.inner = self.inner.nostr_relay(url);
        self
    }

    pub fn nostr_relays<I, S>(mut self, urls: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.nostr_relays(urls);
        self
    }

    pub fn bind_ip(mut self, ip: IpAddr) -> Self {
        self.inner = self.inner.bind_ip(ip);
        self
    }

    pub fn bind_port(mut self, port: u16) -> Self {
        self.inner = self.inner.bind_port(port);
        self
    }

    pub fn listen_all(mut self, enabled: bool) -> Self {
        self.inner = self.inner.listen_all(enabled);
        self
    }

    pub fn enumerate_host(mut self, enabled: bool) -> Self {
        self.inner = self.inner.enumerate_host(enabled);
        self
    }

    pub fn owner_key(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner = self.inner.owner_key(path);
        self
    }

    pub fn owner_required(mut self, required: bool) -> Self {
        self.inner = self.inner.owner_required(required);
        self
    }

    pub fn node_label(mut self, label: impl Into<String>) -> Self {
        self.inner = self.inner.node_label(label);
        self
    }

    pub fn trust_policy(mut self, policy: EmbeddedTrustPolicy) -> Self {
        self.inner = self.inner.trust_policy(policy);
        self
    }

    pub fn trust_owner(mut self, owner_id: impl Into<String>) -> Self {
        self.inner = self.inner.trust_owner(owner_id);
        self
    }

    pub fn trust_owners<I, S>(mut self, owner_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.trust_owners(owner_ids);
        self
    }

    pub fn min_node_version(mut self, version: impl Into<String>) -> Self {
        self.inner = self.inner.min_node_version(version);
        self
    }

    pub fn max_node_version(mut self, version: impl Into<String>) -> Self {
        self.inner = self.inner.max_node_version(version);
        self
    }

    pub fn min_protocol_version(mut self, version: u32) -> Self {
        self.inner = self.inner.min_protocol_version(version);
        self
    }

    pub fn signed_join_tokens(mut self, enabled: bool) -> Self {
        self.inner = self.inner.signed_join_tokens(enabled);
        self
    }

    pub fn max_protocol_version(mut self, version: u32) -> Self {
        self.inner = self.inner.max_protocol_version(version);
        self
    }

    pub fn require_release_attestation(mut self, required: bool) -> Self {
        self.inner = self.inner.require_release_attestation(required);
        self
    }

    pub fn release_signer_key(mut self, key: impl Into<String>) -> Self {
        self.inner = self.inner.release_signer_key(key);
        self
    }

    pub fn release_signer_keys<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner = self.inner.release_signer_keys(keys);
        self
    }

    pub fn config_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.inner = self.inner.config_path(path);
        self
    }

    pub fn isolated_config(mut self, enabled: bool) -> Self {
        self.inner = self.inner.isolated_config(enabled);
        self
    }

    pub fn log_format(mut self, format: EmbeddedMeshLogFormat) -> Self {
        self.inner = self.inner.log_format(format);
        self
    }

    pub fn startup_timeout(mut self, timeout: Duration) -> Self {
        self.inner = self.inner.startup_timeout(timeout);
        self
    }

    pub fn build(self) -> EmbeddedMeshNodeConfig {
        self.inner.build()
    }

    pub async fn start(self) -> anyhow::Result<MeshNode> {
        let handle = mesh_llm_embedded_runtime::start_embedded_node(self.build()).await?;
        Ok(MeshNode { handle })
    }
}

#[derive(Clone)]
pub struct OpenAiClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key: "mesh".to_string(),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.http
    }

    pub async fn models(&self) -> anyhow::Result<serde_json::Value> {
        self.get_json("models").await
    }

    pub async fn chat_completions(
        &self,
        body: impl serde::Serialize,
    ) -> anyhow::Result<serde_json::Value> {
        self.post_json("chat/completions", body).await
    }

    pub async fn responses(
        &self,
        body: impl serde::Serialize,
    ) -> anyhow::Result<serde_json::Value> {
        self.post_json("responses", body).await
    }

    async fn get_json(&self, path: &str) -> anyhow::Result<serde_json::Value> {
        let response = self
            .http
            .get(self.url(path))
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;
        Ok(response.json().await?)
    }

    async fn post_json(
        &self,
        path: &str,
        body: impl serde::Serialize,
    ) -> anyhow::Result<serde_json::Value> {
        let response = self
            .http
            .post(self.url(path))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(response.json().await?)
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn share_endpoint_builder_starts_without_selecting_a_model() {
        let config = MeshNode::builder()
            .share_endpoint("http://localhost:11434")
            .build();
        assert_eq!(
            config.mode,
            EmbeddedMeshNodeMode::SharedEndpoint {
                address: "http://localhost:11434".to_string()
            }
        );
        assert!(config.serving.models.is_empty());
    }

    #[test]
    fn builder_shapes_public_auto_join_serve_node() {
        let config = MeshNode::builder()
            .serve()
            .model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
            .auto_join_public_mesh()
            .api_port(19447)
            .console_port(13141)
            .build();

        assert_eq!(config.mode, EmbeddedMeshNodeMode::Serve);
        assert_eq!(
            config.serving.models,
            vec!["unsloth/Qwen3-0.6B-GGUF:Q4_K_M"]
        );
        assert!(config.network.auto_join);
        assert_eq!(
            config.network.discovery_mode,
            EmbeddedMeshDiscoveryMode::Nostr
        );
        assert_eq!(config.http.api_port, 19447);
        assert_eq!(config.http.console_port, 13141);
    }

    #[test]
    fn openai_client_builds_v1_urls() {
        let client = OpenAiClient::new("http://127.0.0.1:9337/v1/");
        assert_eq!(client.url("models"), "http://127.0.0.1:9337/v1/models");
        assert_eq!(
            client.url("chat/completions"),
            "http://127.0.0.1:9337/v1/chat/completions"
        );
    }
}
