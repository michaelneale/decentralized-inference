#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::net::IpAddr;
use std::path::PathBuf;
#[cfg(feature = "serving")]
use std::time::Duration;

#[cfg(feature = "serving")]
use anyhow::Result;

#[cfg(feature = "client")]
pub use mesh_llm_api_client::*;

/// Detect OpenAI-compatible LLM servers already running on the local machine
/// (Ollama, LM Studio, LiteLLM, vLLM, TGI, Lemonade Server).
///
/// Probes go to loopback addresses only, never traverse a configured HTTP
/// proxy, and use a sub-second timeout. Nothing here reads or writes
/// configuration — it reports what is listening so the caller can decide.
///
/// ```no_run
/// # async fn example() {
/// for endpoint in mesh_llm_sdk::endpoint_discovery::discover_local_endpoints().await {
///     println!("{} serves {:?}", endpoint.describe(), endpoint.models);
/// }
/// # }
/// ```
#[cfg(feature = "endpoint-discovery")]
pub mod endpoint_discovery {
    pub use endpoint_discovery::{DiscoveredEndpoint, discover_local_endpoints};
}

#[cfg(feature = "console")]
pub mod console {
    pub use mesh_llm_console_server::{
        ConsoleServerHandle, ConsoleServerOptions, start_file_console,
    };
}

#[cfg(feature = "node")]
pub mod node {
    pub use mesh_llm_api_server::*;
}

#[cfg(feature = "serving")]
pub mod embedded_node;

#[cfg(feature = "serving")]
pub use embedded_node::{MeshNode, MeshNodeBuilder, MeshNodeStatus, OpenAiClient};

#[cfg(feature = "serving")]
pub use mesh_llm_embedded_runtime::initialize_host_runtime;

#[cfg(feature = "serving")]
pub mod embedded_runtime {
    pub use mesh_llm_embedded_runtime::{
        EmbeddedChatMessage, EmbeddedMeshAdmissionConfig, EmbeddedMeshDiscoveryMode,
        EmbeddedMeshHttpConfig, EmbeddedMeshLogFormat, EmbeddedMeshNetworkConfig,
        EmbeddedMeshNodeBuilder, EmbeddedMeshNodeConfig, EmbeddedMeshNodeHandle,
        EmbeddedMeshNodeMode, EmbeddedMeshNodeStatus, EmbeddedMeshRequirementsConfig,
        EmbeddedMeshServingConfig, EmbeddedMeshStorageConfig, EmbeddedServeConfig,
        EmbeddedServeHandle, EmbeddedServeMode, EmbeddedServeStatus, EmbeddedServingController,
        EmbeddedTrustPolicy, SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION, initialize_host_runtime,
        start_embedded_node, start_embedded_serve,
    };
}

#[cfg(feature = "serving")]
pub mod native_runtime {
    pub use mesh_llm_runtime_install::*;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MeshDiscoveryMode {
    #[default]
    Nostr,
    Mdns,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LogFormat {
    Pretty,
    #[default]
    Json,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TrustPolicy {
    #[default]
    Off,
    PreferOwned,
    RequireOwned,
    Allowlist,
}

#[derive(Clone, Debug)]
pub struct HttpConfig {
    pub api_port: u16,
    pub console_port: u16,
    pub console_ui: bool,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            api_port: 9337,
            console_port: 3131,
            console_ui: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NetworkConfig {
    pub join_tokens: Vec<String>,
    pub auto_join: bool,
    pub discovery_mode: MeshDiscoveryMode,
    pub publish: bool,
    pub mesh_name: Option<String>,
    pub region: Option<String>,
    pub node_name: Option<String>,
    pub iroh_relays: Vec<String>,
    pub iroh_relay_auth: BTreeMap<String, String>,
    pub disable_iroh_relays: bool,
    pub nostr_relays: Vec<String>,
    pub bind_ip: Option<IpAddr>,
    pub bind_port: Option<u16>,
    pub listen_all: bool,
    pub enumerate_host: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            join_tokens: Vec::new(),
            auto_join: false,
            discovery_mode: MeshDiscoveryMode::Nostr,
            publish: false,
            mesh_name: None,
            region: None,
            node_name: None,
            iroh_relays: Vec::new(),
            iroh_relay_auth: BTreeMap::new(),
            disable_iroh_relays: false,
            nostr_relays: Vec::new(),
            bind_ip: None,
            bind_port: None,
            listen_all: false,
            enumerate_host: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StorageConfig {
    pub config_path: Option<PathBuf>,
    pub isolated_config: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            config_path: None,
            isolated_config: true,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MeshRequirementsConfig {
    pub min_node_version: Option<String>,
    pub max_node_version: Option<String>,
    pub min_protocol_version: Option<u32>,
    pub max_protocol_version: Option<u32>,
    pub require_release_attestation: bool,
    pub release_signer_keys: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AdmissionConfig {
    pub owner_key: Option<PathBuf>,
    pub owner_required: bool,
    pub node_label: Option<String>,
    pub trust_policy: Option<TrustPolicy>,
    pub trusted_owners: Vec<String>,
    pub mesh_requirements: MeshRequirementsConfig,
}

#[derive(Clone, Debug, Default)]
pub struct ServingConfig {
    pub models: Vec<String>,
    pub max_vram_gb: Option<f64>,
}

#[cfg(feature = "serving")]
macro_rules! impl_common_builder_methods {
    () => {
        pub fn api_port(mut self, port: u16) -> Self {
            self.config.http.api_port = port;
            self
        }

        pub fn console_port(mut self, port: u16) -> Self {
            self.config.http.console_port = port;
            self
        }

        pub fn console_ui(mut self, enabled: bool) -> Self {
            self.config.http.console_ui = enabled;
            self
        }

        pub fn join_token(mut self, token: impl Into<String>) -> Self {
            self.config.network.join_tokens.push(token.into());
            self
        }

        pub fn join_tokens<I, S>(mut self, tokens: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.config.network.join_tokens = tokens.into_iter().map(Into::into).collect();
            self
        }

        pub fn auto_join(mut self, enabled: bool) -> Self {
            self.config.network.auto_join = enabled;
            self
        }

        pub fn discovery_mode(mut self, mode: MeshDiscoveryMode) -> Self {
            self.config.network.discovery_mode = mode;
            self
        }

        pub fn publish(mut self, enabled: bool) -> Self {
            self.config.network.publish = enabled;
            self
        }

        pub fn mesh_name(mut self, name: impl Into<String>) -> Self {
            self.config.network.mesh_name = Some(name.into());
            self
        }

        pub fn region(mut self, region: impl Into<String>) -> Self {
            self.config.network.region = Some(region.into());
            self
        }

        pub fn node_name(mut self, name: impl Into<String>) -> Self {
            self.config.network.node_name = Some(name.into());
            self
        }

        pub fn iroh_relay(mut self, url: impl Into<String>) -> Self {
            self.config.network.iroh_relays.push(url.into());
            self
        }

        pub fn iroh_relays<I, S>(mut self, urls: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.config.network.iroh_relays = urls.into_iter().map(Into::into).collect();
            self
        }

        pub fn iroh_relay_auth(
            mut self,
            relay_url: impl Into<String>,
            bearer_token: impl Into<String>,
        ) -> Self {
            self.config
                .network
                .iroh_relay_auth
                .insert(relay_url.into(), bearer_token.into());
            self
        }

        pub fn disable_iroh_relays(mut self, disabled: bool) -> Self {
            self.config.network.disable_iroh_relays = disabled;
            self
        }

        pub fn nostr_relay(mut self, url: impl Into<String>) -> Self {
            self.config.network.nostr_relays.push(url.into());
            self
        }

        pub fn nostr_relays<I, S>(mut self, urls: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.config.network.nostr_relays = urls.into_iter().map(Into::into).collect();
            self
        }

        pub fn bind_ip(mut self, ip: IpAddr) -> Self {
            self.config.network.bind_ip = Some(ip);
            self
        }

        pub fn bind_port(mut self, port: u16) -> Self {
            self.config.network.bind_port = Some(port);
            self
        }

        pub fn listen_all(mut self, enabled: bool) -> Self {
            self.config.network.listen_all = enabled;
            self
        }

        pub fn enumerate_host(mut self, enabled: bool) -> Self {
            self.config.network.enumerate_host = enabled;
            self
        }

        pub fn owner_key(mut self, path: impl Into<PathBuf>) -> Self {
            self.config.admission.owner_key = Some(path.into());
            self
        }

        pub fn owner_required(mut self, required: bool) -> Self {
            self.config.admission.owner_required = required;
            self
        }

        pub fn node_label(mut self, label: impl Into<String>) -> Self {
            self.config.admission.node_label = Some(label.into());
            self
        }

        pub fn trust_policy(mut self, policy: TrustPolicy) -> Self {
            self.config.admission.trust_policy = Some(policy);
            self
        }

        pub fn trust_owner(mut self, owner_id: impl Into<String>) -> Self {
            self.config.admission.trusted_owners.push(owner_id.into());
            self
        }

        pub fn trust_owners<I, S>(mut self, owner_ids: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.config.admission.trusted_owners = owner_ids.into_iter().map(Into::into).collect();
            self
        }

        pub fn min_node_version(mut self, version: impl Into<String>) -> Self {
            self.config.admission.mesh_requirements.min_node_version = Some(version.into());
            self
        }

        pub fn max_node_version(mut self, version: impl Into<String>) -> Self {
            self.config.admission.mesh_requirements.max_node_version = Some(version.into());
            self
        }

        pub fn min_protocol_version(mut self, version: u32) -> Self {
            self.config.admission.mesh_requirements.min_protocol_version = Some(version);
            self
        }

        /// Make mesh originators emit signed bootstrap tokens instead of legacy endpoint tokens.
        pub fn signed_join_tokens(mut self, enabled: bool) -> Self {
            if enabled {
                self.config.admission.mesh_requirements.min_protocol_version = Some(
                    self.config
                        .admission
                        .mesh_requirements
                        .min_protocol_version
                        .unwrap_or(crate::embedded_runtime::SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION)
                        .max(crate::embedded_runtime::SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION),
                );
            } else if self.config.admission.mesh_requirements.min_protocol_version
                == Some(crate::embedded_runtime::SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION)
            {
                self.config.admission.mesh_requirements.min_protocol_version = None;
            }
            self
        }

        pub fn max_protocol_version(mut self, version: u32) -> Self {
            self.config.admission.mesh_requirements.max_protocol_version = Some(version);
            self
        }

        pub fn require_release_attestation(mut self, required: bool) -> Self {
            self.config
                .admission
                .mesh_requirements
                .require_release_attestation = required;
            self
        }

        pub fn release_signer_key(mut self, key: impl Into<String>) -> Self {
            self.config
                .admission
                .mesh_requirements
                .release_signer_keys
                .push(key.into());
            self
        }

        pub fn release_signer_keys<I, S>(mut self, keys: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.config.admission.mesh_requirements.release_signer_keys =
                keys.into_iter().map(Into::into).collect();
            self
        }

        pub fn config_path(mut self, path: impl Into<PathBuf>) -> Self {
            self.config.storage.config_path = Some(path.into());
            self
        }

        pub fn isolated_config(mut self, enabled: bool) -> Self {
            self.config.storage.isolated_config = enabled;
            self
        }

        pub fn log_format(mut self, format: LogFormat) -> Self {
            self.config.log_format = format;
            self
        }

        pub fn startup_timeout(mut self, timeout: Duration) -> Self {
            self.config.startup_timeout = timeout;
            self
        }
    };
}

#[cfg(feature = "serving")]
pub struct EmbeddedNodeHandle {
    inner: mesh_llm_embedded_runtime::EmbeddedMeshNodeHandle,
}

#[cfg(feature = "serving")]
impl EmbeddedNodeHandle {
    pub fn api_base_url(&self) -> &str {
        self.inner.api_base_url()
    }

    pub fn console_url(&self) -> &str {
        self.inner.console_url()
    }

    pub fn invite_token(&self) -> Option<&str> {
        self.inner.invite_token()
    }

    pub async fn status(&self) -> Result<EmbeddedNodeStatus> {
        self.inner.status().await.map(EmbeddedNodeStatus::from)
    }

    pub async fn join_token(&self, token: impl Into<String>) -> Result<()> {
        self.inner.join_token(token).await
    }

    pub async fn stop(self) -> Result<()> {
        self.inner.stop().await
    }
}

#[cfg(feature = "serving")]
#[derive(Clone, Debug)]
pub struct EmbeddedNodeStatus {
    pub api_base_url: String,
    pub console_url: String,
    pub invite_token: Option<String>,
    pub payload: serde_json::Value,
}

#[cfg(feature = "serving")]
impl From<mesh_llm_embedded_runtime::EmbeddedMeshNodeStatus> for EmbeddedNodeStatus {
    fn from(status: mesh_llm_embedded_runtime::EmbeddedMeshNodeStatus) -> Self {
        Self {
            api_base_url: status.api_base_url,
            console_url: status.console_url,
            invite_token: status.invite_token,
            payload: status.payload,
        }
    }
}

#[cfg(feature = "serving")]
pub mod client {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct EmbeddedClientConfig {
        pub http: HttpConfig,
        pub network: NetworkConfig,
        pub admission: AdmissionConfig,
        pub storage: StorageConfig,
        pub log_format: LogFormat,
        pub startup_timeout: Duration,
    }

    impl Default for EmbeddedClientConfig {
        fn default() -> Self {
            Self {
                http: HttpConfig::default(),
                network: NetworkConfig::default(),
                admission: AdmissionConfig::default(),
                storage: StorageConfig::default(),
                log_format: LogFormat::default(),
                startup_timeout: Duration::from_secs(30),
            }
        }
    }

    impl EmbeddedClientConfig {
        pub fn builder() -> EmbeddedClientConfigBuilder {
            EmbeddedClientConfigBuilder::default()
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct EmbeddedClientConfigBuilder {
        config: EmbeddedClientConfig,
    }

    impl EmbeddedClientConfigBuilder {
        impl_common_builder_methods!();

        pub fn build(self) -> EmbeddedClientConfig {
            self.config
        }
    }

    pub async fn start(config: EmbeddedClientConfig) -> Result<EmbeddedNodeHandle> {
        start_embedded_node(EmbeddedNodeParts {
            mode: EmbeddedMode::Client,
            http: config.http,
            network: config.network,
            admission: config.admission,
            storage: config.storage,
            serving: ServingConfig::default(),
            log_format: config.log_format,
            startup_timeout: config.startup_timeout,
        })
        .await
    }
}

#[cfg(feature = "serving")]
pub mod serve {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct EmbeddedServeConfig {
        pub http: HttpConfig,
        pub network: NetworkConfig,
        pub admission: AdmissionConfig,
        pub storage: StorageConfig,
        pub serving: ServingConfig,
        pub log_format: LogFormat,
        pub startup_timeout: Duration,
    }

    impl Default for EmbeddedServeConfig {
        fn default() -> Self {
            Self {
                http: HttpConfig::default(),
                network: NetworkConfig::default(),
                admission: AdmissionConfig::default(),
                storage: StorageConfig::default(),
                serving: ServingConfig::default(),
                log_format: LogFormat::default(),
                startup_timeout: Duration::from_secs(30),
            }
        }
    }

    impl EmbeddedServeConfig {
        pub fn builder() -> EmbeddedServeConfigBuilder {
            EmbeddedServeConfigBuilder::default()
        }
    }

    #[derive(Clone, Debug, Default)]
    pub struct EmbeddedServeConfigBuilder {
        config: EmbeddedServeConfig,
    }

    impl EmbeddedServeConfigBuilder {
        impl_common_builder_methods!();

        pub fn model(mut self, model_ref: impl Into<String>) -> Self {
            self.config.serving.models.push(model_ref.into());
            self
        }

        pub fn models<I, S>(mut self, model_refs: I) -> Self
        where
            I: IntoIterator<Item = S>,
            S: Into<String>,
        {
            self.config.serving.models = model_refs.into_iter().map(Into::into).collect();
            self
        }

        pub fn max_vram_gb(mut self, max_vram_gb: f64) -> Self {
            self.config.serving.max_vram_gb = Some(max_vram_gb);
            self
        }

        pub fn build(self) -> EmbeddedServeConfig {
            self.config
        }
    }

    pub async fn start(config: EmbeddedServeConfig) -> Result<EmbeddedNodeHandle> {
        start_embedded_node(EmbeddedNodeParts {
            mode: EmbeddedMode::Serve,
            http: config.http,
            network: config.network,
            admission: config.admission,
            storage: config.storage,
            serving: config.serving,
            log_format: config.log_format,
            startup_timeout: config.startup_timeout,
        })
        .await
    }
}

#[cfg(feature = "serving")]
#[derive(Clone, Copy, Debug)]
enum EmbeddedMode {
    #[cfg(feature = "serving")]
    Serve,
    Client,
}

#[cfg(feature = "serving")]
struct EmbeddedNodeParts {
    mode: EmbeddedMode,
    http: HttpConfig,
    network: NetworkConfig,
    admission: AdmissionConfig,
    storage: StorageConfig,
    serving: ServingConfig,
    log_format: LogFormat,
    startup_timeout: Duration,
}

#[cfg(feature = "serving")]
async fn start_embedded_node(parts: EmbeddedNodeParts) -> Result<EmbeddedNodeHandle> {
    let handle = mesh_llm_embedded_runtime::start_embedded_node(host_config(parts)).await?;
    Ok(EmbeddedNodeHandle { inner: handle })
}

#[cfg(feature = "serving")]
fn host_config(parts: EmbeddedNodeParts) -> mesh_llm_embedded_runtime::EmbeddedMeshNodeConfig {
    mesh_llm_embedded_runtime::EmbeddedMeshNodeConfig {
        mode: match parts.mode {
            EmbeddedMode::Serve => mesh_llm_embedded_runtime::EmbeddedMeshNodeMode::Serve,
            EmbeddedMode::Client => mesh_llm_embedded_runtime::EmbeddedMeshNodeMode::Client,
        },
        http: mesh_llm_embedded_runtime::EmbeddedMeshHttpConfig {
            api_port: parts.http.api_port,
            console_port: parts.http.console_port,
            console_ui: parts.http.console_ui,
        },
        serving: mesh_llm_embedded_runtime::EmbeddedMeshServingConfig {
            models: parts.serving.models,
            max_vram_gb: parts.serving.max_vram_gb,
        },
        network: mesh_llm_embedded_runtime::EmbeddedMeshNetworkConfig {
            join_tokens: parts.network.join_tokens,
            auto_join: parts.network.auto_join,
            discovery_mode: match parts.network.discovery_mode {
                MeshDiscoveryMode::Nostr => {
                    mesh_llm_embedded_runtime::EmbeddedMeshDiscoveryMode::Nostr
                }
                MeshDiscoveryMode::Mdns => {
                    mesh_llm_embedded_runtime::EmbeddedMeshDiscoveryMode::Mdns
                }
            },
            publish: parts.network.publish,
            mesh_name: parts.network.mesh_name,
            region: parts.network.region,
            node_name: parts.network.node_name,
            iroh_relays: parts.network.iroh_relays,
            iroh_relay_auth: parts.network.iroh_relay_auth,
            disable_iroh_relays: parts.network.disable_iroh_relays,
            nostr_relays: parts.network.nostr_relays,
            bind_ip: parts.network.bind_ip,
            bind_port: parts.network.bind_port,
            listen_all: parts.network.listen_all,
            enumerate_host: parts.network.enumerate_host,
        },
        admission: mesh_llm_embedded_runtime::EmbeddedMeshAdmissionConfig {
            owner_key: parts.admission.owner_key,
            owner_required: parts.admission.owner_required,
            node_label: parts.admission.node_label,
            trust_policy: parts.admission.trust_policy.map(|policy| match policy {
                TrustPolicy::Off => mesh_llm_embedded_runtime::EmbeddedTrustPolicy::Off,
                TrustPolicy::PreferOwned => {
                    mesh_llm_embedded_runtime::EmbeddedTrustPolicy::PreferOwned
                }
                TrustPolicy::RequireOwned => {
                    mesh_llm_embedded_runtime::EmbeddedTrustPolicy::RequireOwned
                }
                TrustPolicy::Allowlist => mesh_llm_embedded_runtime::EmbeddedTrustPolicy::Allowlist,
            }),
            trusted_owners: parts.admission.trusted_owners,
            mesh_requirements: mesh_llm_embedded_runtime::EmbeddedMeshRequirementsConfig {
                min_node_version: parts.admission.mesh_requirements.min_node_version,
                max_node_version: parts.admission.mesh_requirements.max_node_version,
                min_protocol_version: parts.admission.mesh_requirements.min_protocol_version,
                max_protocol_version: parts.admission.mesh_requirements.max_protocol_version,
                require_release_attestation: parts
                    .admission
                    .mesh_requirements
                    .require_release_attestation,
                release_signer_keys: parts.admission.mesh_requirements.release_signer_keys,
            },
        },
        storage: mesh_llm_embedded_runtime::EmbeddedMeshStorageConfig {
            config_path: parts.storage.config_path,
            isolated_config: parts.storage.isolated_config,
        },
        log_format: match parts.log_format {
            LogFormat::Pretty => mesh_llm_embedded_runtime::EmbeddedMeshLogFormat::Pretty,
            LogFormat::Json => mesh_llm_embedded_runtime::EmbeddedMeshLogFormat::Json,
        },
        startup_timeout: parts.startup_timeout,
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "serving")]
    use super::*;

    #[test]
    #[cfg(feature = "serving")]
    fn client_builder_sets_network_fields() {
        let config = client::EmbeddedClientConfig::builder()
            .auto_join(true)
            .join_token("mesh-token")
            .api_port(19337)
            .console_port(13131)
            .discovery_mode(MeshDiscoveryMode::Mdns)
            .disable_iroh_relays(true)
            .log_format(LogFormat::Pretty)
            .build();

        assert!(config.network.auto_join);
        assert_eq!(config.network.join_tokens, vec!["mesh-token"]);
        assert_eq!(config.http.api_port, 19337);
        assert_eq!(config.http.console_port, 13131);
        assert_eq!(config.network.discovery_mode, MeshDiscoveryMode::Mdns);
        assert!(config.network.disable_iroh_relays);
        assert_eq!(config.log_format, LogFormat::Pretty);
    }

    #[test]
    #[cfg(feature = "serving")]
    fn serve_builder_sets_model_fields() {
        let config = serve::EmbeddedServeConfig::builder()
            .model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
            .max_vram_gb(6.0)
            .mesh_name("sprout")
            .build();

        assert_eq!(
            config.serving.models,
            vec!["unsloth/Qwen3-0.6B-GGUF:Q4_K_M"]
        );
        assert_eq!(config.serving.max_vram_gb, Some(6.0));
        assert_eq!(config.network.mesh_name.as_deref(), Some("sprout"));
    }

    #[test]
    #[cfg(feature = "serving")]
    fn serve_builder_sets_admission_fields() {
        let config = serve::EmbeddedServeConfig::builder()
            .owner_key("/tmp/sprout-owner.json")
            .owner_required(true)
            .node_label("sprout-desktop")
            .trust_policy(TrustPolicy::RequireOwned)
            .trust_owner("owner-a")
            .trust_owner("owner-b")
            .min_node_version("0.65.0")
            .max_node_version("0.66.0")
            .signed_join_tokens(true)
            .max_protocol_version(2)
            .require_release_attestation(false)
            .release_signer_keys(["ed25519:abc"])
            .build();

        assert_eq!(
            config.admission.owner_key.as_deref(),
            Some(std::path::Path::new("/tmp/sprout-owner.json"))
        );
        assert!(config.admission.owner_required);
        assert_eq!(
            config.admission.node_label.as_deref(),
            Some("sprout-desktop")
        );
        assert_eq!(
            config.admission.trust_policy,
            Some(TrustPolicy::RequireOwned)
        );
        assert_eq!(config.admission.trusted_owners, vec!["owner-a", "owner-b"]);
        assert_eq!(
            config
                .admission
                .mesh_requirements
                .min_node_version
                .as_deref(),
            Some("0.65.0")
        );
        assert_eq!(
            config
                .admission
                .mesh_requirements
                .max_node_version
                .as_deref(),
            Some("0.66.0")
        );
        assert_eq!(
            config.admission.mesh_requirements.min_protocol_version,
            Some(1)
        );
        assert_eq!(
            config.admission.mesh_requirements.max_protocol_version,
            Some(2)
        );
        assert!(
            !config
                .admission
                .mesh_requirements
                .require_release_attestation
        );
        assert_eq!(
            config.admission.mesh_requirements.release_signer_keys,
            vec!["ed25519:abc"]
        );
    }

    #[test]
    #[cfg(feature = "serving")]
    fn signed_join_tokens_sets_genesis_requirement_without_lowering_existing_bound() {
        let config = serve::EmbeddedServeConfig::builder()
            .signed_join_tokens(true)
            .build();
        assert_eq!(
            config.admission.mesh_requirements.min_protocol_version,
            Some(crate::embedded_runtime::SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION)
        );

        let config = serve::EmbeddedServeConfig::builder()
            .min_protocol_version(2)
            .signed_join_tokens(true)
            .build();
        assert_eq!(
            config.admission.mesh_requirements.min_protocol_version,
            Some(2)
        );
    }
}
