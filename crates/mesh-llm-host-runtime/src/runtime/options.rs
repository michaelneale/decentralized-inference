use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

use mesh_llm_events::{
    LogFormat,
    audit::{AuditLevel, AuditLogFormat},
};

use crate::crypto::TrustPolicy;
use crate::discovery::MeshDiscoveryMode;
use crate::plugin::SpeculativeConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeSurface {
    Serve,
    Client,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MeshGuardrailMode {
    #[default]
    Disabled,
    Metrics,
    Enforce,
}

#[derive(Clone, Debug)]
pub struct RuntimeOptions {
    pub log_format: LogFormat,
    pub debug: bool,
    pub skippy_metrics_otlp_grpc: Option<String>,
    pub mesh_guardrails: MeshGuardrailMode,
    pub help_text: Option<String>,
    pub join: Vec<String>,
    pub discover: Option<String>,
    pub auto: bool,
    pub mesh_discovery_mode: MeshDiscoveryMode,
    pub model: Vec<PathBuf>,
    pub gguf: Vec<PathBuf>,
    pub mmproj: Option<PathBuf>,
    pub checkpoint_quantization: Option<String>,
    pub checkpoint_imatrix: Option<PathBuf>,
    pub port: u16,
    pub local_model_only: bool,
    pub native_serving_plugin: Option<PathBuf>,
    pub native_serving_plugin_config: Option<PathBuf>,
    pub native_serving_plugin_state: Option<PathBuf>,
    pub native_serving_plugin_deadline_ms: Option<u64>,
    pub client: bool,
    pub console: u16,
    pub headless: bool,
    pub swarm_capture: Option<PathBuf>,
    pub publish: bool,
    pub mesh_name: Option<String>,
    pub region: Option<String>,
    pub min_node_version: Option<String>,
    pub max_node_version: Option<String>,
    pub min_protocol_version: Option<u32>,
    pub max_protocol_version: Option<u32>,
    pub require_release_attestation: bool,
    pub release_signer_key: Vec<String>,
    pub name: Option<String>,
    pub plugin: Option<String>,
    pub auto_update: bool,
    pub command_is_update: bool,
    pub command_uses_machine_output: bool,
    pub draft: Option<PathBuf>,
    pub draft_max: u16,
    pub no_draft: bool,
    pub speculative_overrides: Option<SpeculativeConfig>,
    pub split: bool,
    pub split_topology_lock: Option<PathBuf>,
    pub ctx_size: Option<u32>,
    pub max_vram: Option<f64>,
    pub no_enumerate_host: bool,
    pub bin_dir: Option<PathBuf>,
    pub llama_flavor: Option<mesh_llm_system::backend::BinaryFlavor>,
    pub device: Option<String>,
    pub tensor_split: Option<String>,
    pub relay: Vec<String>,
    pub relay_auth: Vec<(String, String)>,
    pub disable_iroh_relays: bool,
    pub bind_port: Option<u16>,
    pub bind_ip: Option<IpAddr>,
    pub listen_all: bool,
    pub max_clients: Option<usize>,
    pub nostr_relay: Vec<String>,
    pub no_console: bool,
    pub config: Option<PathBuf>,
    pub kv_cache_disk: Option<String>,
    pub kv_cache_disk_dir: Option<PathBuf>,
    pub kv_cache_min_free: Option<String>,
    pub owner_key: Option<PathBuf>,
    pub control_bind: Option<SocketAddr>,
    pub control_advertise_addr: Option<SocketAddr>,
    pub owner_required: bool,
    pub node_label: Option<String>,
    pub trust_policy: Option<TrustPolicy>,
    pub trust_owner: Vec<String>,
    pub nostr_discovery: bool,
    pub audit_log_path: Option<PathBuf>,
    pub audit_log_format: AuditLogFormat,
    pub audit_log_level: AuditLevel,
    pub audit_max_file_size: u64,
    pub audit_max_files: usize,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            log_format: LogFormat::Pretty,
            debug: false,
            skippy_metrics_otlp_grpc: None,
            mesh_guardrails: MeshGuardrailMode::Disabled,
            help_text: None,
            join: Vec::new(),
            discover: None,
            auto: false,
            mesh_discovery_mode: MeshDiscoveryMode::Nostr,
            model: Vec::new(),
            gguf: Vec::new(),
            mmproj: None,
            checkpoint_quantization: None,
            checkpoint_imatrix: None,
            port: 9337,
            local_model_only: false,
            native_serving_plugin: None,
            native_serving_plugin_config: None,
            native_serving_plugin_state: None,
            native_serving_plugin_deadline_ms: None,
            client: false,
            console: 3131,
            headless: false,
            swarm_capture: None,
            publish: false,
            mesh_name: None,
            region: None,
            min_node_version: None,
            max_node_version: None,
            min_protocol_version: None,
            max_protocol_version: None,
            require_release_attestation: false,
            release_signer_key: Vec::new(),
            name: None,
            plugin: None,
            auto_update: false,
            command_is_update: false,
            command_uses_machine_output: false,
            draft: None,
            draft_max: 8,
            no_draft: false,
            speculative_overrides: None,
            split: false,
            split_topology_lock: None,
            ctx_size: None,
            max_vram: None,
            no_enumerate_host: false,
            bin_dir: None,
            llama_flavor: None,
            device: None,
            tensor_split: None,
            relay: Vec::new(),
            relay_auth: Vec::new(),
            disable_iroh_relays: false,
            bind_port: None,
            bind_ip: None,
            listen_all: false,
            max_clients: None,
            nostr_relay: Vec::new(),
            no_console: false,
            config: None,
            kv_cache_disk: None,
            kv_cache_disk_dir: None,
            kv_cache_min_free: None,
            owner_key: None,
            control_bind: None,
            control_advertise_addr: None,
            owner_required: false,
            node_label: None,
            trust_policy: None,
            trust_owner: Vec::new(),
            nostr_discovery: false,
            audit_log_path: None,
            audit_log_format: AuditLogFormat::JsonLines,
            audit_log_level: AuditLevel::Info,
            audit_max_file_size: 100 * 1024 * 1024,
            audit_max_files: 10,
        }
    }
}

impl RuntimeOptions {
    pub fn validate_discovery_mode_args(&self) -> anyhow::Result<()> {
        if self.mesh_discovery_mode != MeshDiscoveryMode::Mdns {
            return Ok(());
        }

        if !self.nostr_relay.is_empty() {
            anyhow::bail!("--nostr-relay is only valid with --mesh-discovery-mode nostr");
        }
        if !self.relay.is_empty() {
            anyhow::bail!("--relay is only valid with --mesh-discovery-mode nostr");
        }
        if !self.relay_auth.is_empty() {
            anyhow::bail!("--relay-auth is only valid with --mesh-discovery-mode nostr");
        }

        Ok(())
    }
}
