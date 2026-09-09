use std::collections::BTreeMap;
use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Smallest mesh protocol generation that makes an originator emit signed bootstrap tokens.
pub const SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EmbeddedMeshNodeMode {
    /// Full mesh participation serving one already-running OpenAI-compatible
    /// HTTP server, with local inference disabled.
    ///
    /// The address is fixed for the node's lifetime — stop the node to stop
    /// sharing. Mesh never starts or stops the upstream server.
    SharedEndpoint {
        address: String,
    },
    Serve,
    Client,
}

pub type EmbeddedServeMode = EmbeddedMeshNodeMode;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EmbeddedMeshDiscoveryMode {
    #[default]
    Nostr,
    Mdns,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EmbeddedMeshLogFormat {
    Pretty,
    #[default]
    Json,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EmbeddedTrustPolicy {
    #[default]
    Off,
    PreferOwned,
    RequireOwned,
    Allowlist,
}

impl From<EmbeddedMeshLogFormat> for mesh_llm_events::LogFormat {
    fn from(format: EmbeddedMeshLogFormat) -> Self {
        match format {
            EmbeddedMeshLogFormat::Pretty => Self::Pretty,
            EmbeddedMeshLogFormat::Json => Self::Json,
        }
    }
}

impl From<EmbeddedTrustPolicy> for crate::crypto::TrustPolicy {
    fn from(policy: EmbeddedTrustPolicy) -> Self {
        match policy {
            EmbeddedTrustPolicy::Off => Self::Off,
            EmbeddedTrustPolicy::PreferOwned => Self::PreferOwned,
            EmbeddedTrustPolicy::RequireOwned => Self::RequireOwned,
            EmbeddedTrustPolicy::Allowlist => Self::Allowlist,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmbeddedMeshHttpConfig {
    pub api_port: u16,
    pub console_port: u16,
    pub console_ui: bool,
}

impl Default for EmbeddedMeshHttpConfig {
    fn default() -> Self {
        Self {
            api_port: 9337,
            console_port: 3131,
            console_ui: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmbeddedMeshServingConfig {
    pub models: Vec<String>,
    pub max_vram_gb: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct EmbeddedMeshNetworkConfig {
    pub join_tokens: Vec<String>,
    pub auto_join: bool,
    pub discovery_mode: EmbeddedMeshDiscoveryMode,
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

impl Default for EmbeddedMeshNetworkConfig {
    fn default() -> Self {
        Self {
            join_tokens: Vec::new(),
            auto_join: false,
            discovery_mode: EmbeddedMeshDiscoveryMode::Nostr,
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EmbeddedMeshRequirementsConfig {
    pub min_node_version: Option<String>,
    pub max_node_version: Option<String>,
    pub min_protocol_version: Option<u32>,
    pub max_protocol_version: Option<u32>,
    pub require_release_attestation: bool,
    pub release_signer_keys: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EmbeddedMeshAdmissionConfig {
    pub owner_key: Option<PathBuf>,
    pub owner_required: bool,
    pub node_label: Option<String>,
    pub trust_policy: Option<EmbeddedTrustPolicy>,
    pub trusted_owners: Vec<String>,
    pub mesh_requirements: EmbeddedMeshRequirementsConfig,
}

#[derive(Clone, Debug)]
pub struct EmbeddedMeshStorageConfig {
    pub config_path: Option<PathBuf>,
    pub isolated_config: bool,
}

impl Default for EmbeddedMeshStorageConfig {
    fn default() -> Self {
        Self {
            config_path: None,
            isolated_config: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmbeddedMeshNodeConfig {
    pub mode: EmbeddedMeshNodeMode,
    pub http: EmbeddedMeshHttpConfig,
    pub serving: EmbeddedMeshServingConfig,
    pub network: EmbeddedMeshNetworkConfig,
    pub admission: EmbeddedMeshAdmissionConfig,
    pub storage: EmbeddedMeshStorageConfig,
    pub log_format: EmbeddedMeshLogFormat,
    pub startup_timeout: Duration,
}

impl Default for EmbeddedMeshNodeConfig {
    fn default() -> Self {
        Self {
            mode: EmbeddedMeshNodeMode::Serve,
            http: EmbeddedMeshHttpConfig::default(),
            serving: EmbeddedMeshServingConfig::default(),
            network: EmbeddedMeshNetworkConfig::default(),
            admission: EmbeddedMeshAdmissionConfig::default(),
            storage: EmbeddedMeshStorageConfig::default(),
            log_format: EmbeddedMeshLogFormat::default(),
            startup_timeout: Duration::from_secs(30),
        }
    }
}

impl EmbeddedMeshNodeConfig {
    pub fn builder() -> EmbeddedMeshNodeBuilder {
        EmbeddedMeshNodeBuilder::default()
    }
}

#[derive(Clone, Debug, Default)]
pub struct EmbeddedMeshNodeBuilder {
    config: EmbeddedMeshNodeConfig,
}

impl EmbeddedMeshNodeBuilder {
    pub fn mode(mut self, mode: EmbeddedMeshNodeMode) -> Self {
        self.config.mode = mode;
        self
    }

    pub fn serve(mut self) -> Self {
        self.config.mode = EmbeddedMeshNodeMode::Serve;
        self
    }

    /// Serve one already-running OpenAI-compatible HTTP server, without
    /// loading a native inference runtime or a local model.
    pub fn share_endpoint(mut self, address: impl Into<String>) -> Self {
        self.config.mode = EmbeddedMeshNodeMode::SharedEndpoint {
            address: address.into(),
        };
        self
    }

    pub fn client(mut self) -> Self {
        self.config.mode = EmbeddedMeshNodeMode::Client;
        self
    }

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

    pub fn discovery_mode(mut self, mode: EmbeddedMeshDiscoveryMode) -> Self {
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

    pub fn trust_policy(mut self, policy: EmbeddedTrustPolicy) -> Self {
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
                    .unwrap_or(SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION)
                    .max(SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION),
            );
        } else if self.config.admission.mesh_requirements.min_protocol_version
            == Some(SIGNED_JOIN_TOKEN_MIN_PROTOCOL_VERSION)
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

    pub fn log_format(mut self, format: EmbeddedMeshLogFormat) -> Self {
        self.config.log_format = format;
        self
    }

    pub fn startup_timeout(mut self, timeout: Duration) -> Self {
        self.config.startup_timeout = timeout;
        self
    }

    pub fn build(self) -> EmbeddedMeshNodeConfig {
        self.config
    }
}

#[derive(Clone, Debug)]
pub struct EmbeddedServeConfig {
    pub mode: EmbeddedMeshNodeMode,
    pub models: Vec<String>,
    pub join: Vec<String>,
    pub auto: bool,
    pub api_port: u16,
    pub console_port: u16,
    pub mesh_name: Option<String>,
    pub max_vram_gb: Option<f64>,
    pub publish: bool,
    pub discovery_mode: EmbeddedMeshDiscoveryMode,
    pub relay: Vec<String>,
    pub relay_auth: BTreeMap<String, String>,
    pub disable_iroh_relays: bool,
    pub nostr_relay: Vec<String>,
    pub region: Option<String>,
    pub node_name: Option<String>,
    pub bind_ip: Option<IpAddr>,
    pub bind_port: Option<u16>,
    pub listen_all: bool,
    pub enumerate_host: bool,
    pub console_ui: bool,
    pub admission: EmbeddedMeshAdmissionConfig,
    pub config_path: Option<PathBuf>,
    pub isolated_config: bool,
    pub log_format: EmbeddedMeshLogFormat,
    pub startup_timeout: Duration,
}

impl Default for EmbeddedServeConfig {
    fn default() -> Self {
        Self {
            mode: EmbeddedMeshNodeMode::Serve,
            models: Vec::new(),
            join: Vec::new(),
            auto: false,
            api_port: 9337,
            console_port: 3131,
            mesh_name: None,
            max_vram_gb: None,
            publish: false,
            discovery_mode: EmbeddedMeshDiscoveryMode::Nostr,
            relay: Vec::new(),
            relay_auth: BTreeMap::new(),
            disable_iroh_relays: false,
            nostr_relay: Vec::new(),
            region: None,
            node_name: None,
            bind_ip: None,
            bind_port: None,
            listen_all: false,
            enumerate_host: true,
            console_ui: false,
            admission: EmbeddedMeshAdmissionConfig::default(),
            config_path: None,
            isolated_config: true,
            log_format: EmbeddedMeshLogFormat::default(),
            startup_timeout: Duration::from_secs(30),
        }
    }
}

impl From<EmbeddedServeConfig> for EmbeddedMeshNodeConfig {
    fn from(config: EmbeddedServeConfig) -> Self {
        Self {
            mode: config.mode,
            http: EmbeddedMeshHttpConfig {
                api_port: config.api_port,
                console_port: config.console_port,
                console_ui: config.console_ui,
            },
            serving: EmbeddedMeshServingConfig {
                models: config.models,
                max_vram_gb: config.max_vram_gb,
            },
            network: EmbeddedMeshNetworkConfig {
                join_tokens: config.join,
                auto_join: config.auto,
                discovery_mode: config.discovery_mode,
                publish: config.publish,
                mesh_name: config.mesh_name,
                region: config.region,
                node_name: config.node_name,
                iroh_relays: config.relay,
                iroh_relay_auth: config.relay_auth,
                disable_iroh_relays: config.disable_iroh_relays,
                nostr_relays: config.nostr_relay,
                bind_ip: config.bind_ip,
                bind_port: config.bind_port,
                listen_all: config.listen_all,
                enumerate_host: config.enumerate_host,
            },
            admission: config.admission,
            storage: EmbeddedMeshStorageConfig {
                config_path: config.config_path,
                isolated_config: config.isolated_config,
            },
            log_format: config.log_format,
            startup_timeout: config.startup_timeout,
        }
    }
}

impl From<EmbeddedMeshNodeConfig> for EmbeddedServeConfig {
    fn from(config: EmbeddedMeshNodeConfig) -> Self {
        Self {
            mode: config.mode,
            models: config.serving.models,
            join: config.network.join_tokens,
            auto: config.network.auto_join,
            api_port: config.http.api_port,
            console_port: config.http.console_port,
            mesh_name: config.network.mesh_name,
            max_vram_gb: config.serving.max_vram_gb,
            publish: config.network.publish,
            discovery_mode: config.network.discovery_mode,
            relay: config.network.iroh_relays,
            relay_auth: config.network.iroh_relay_auth,
            disable_iroh_relays: config.network.disable_iroh_relays,
            nostr_relay: config.network.nostr_relays,
            region: config.network.region,
            node_name: config.network.node_name,
            bind_ip: config.network.bind_ip,
            bind_port: config.network.bind_port,
            listen_all: config.network.listen_all,
            enumerate_host: config.network.enumerate_host,
            console_ui: config.http.console_ui,
            admission: config.admission,
            config_path: config.storage.config_path,
            isolated_config: config.storage.isolated_config,
            log_format: config.log_format,
            startup_timeout: config.startup_timeout,
        }
    }
}
