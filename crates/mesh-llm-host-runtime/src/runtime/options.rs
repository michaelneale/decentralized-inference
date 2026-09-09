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
    Share,
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
    /// Serve one already-running external OpenAI-compatible HTTP endpoint and
    /// nothing else.
    ///
    /// `Some` is the whole of sharing mode: there is no separate flag and no
    /// state where sharing is enabled without an upstream. The node stays a
    /// full mesh participant — it gossips, claims the host role, and accepts
    /// inbound requests — but never loads a native inference runtime or a
    /// local model. It is deliberately NOT `client`: a client node never
    /// claims the host role, so it could not serve the upstream to peers.
    ///
    /// Set per invocation only; never read from or written to config.
    pub shared_endpoint: Option<String>,
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
            shared_endpoint: None,
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
    /// Whether this invocation may load models on local hardware.
    ///
    /// Two distinct startups answer `false` for different reasons: a client
    /// never serves at all, and a sharing node serves exclusively by
    /// forwarding to an already-running external server. Callers that gate
    /// native initialization, local model scanning, GPU survey, or local load
    /// intents should ask this rather than testing `client` alone.
    #[must_use]
    pub const fn allows_local_inference(&self) -> bool {
        !self.client && self.shared_endpoint.is_none()
    }

    /// Validate the upstream URL and reject flags that ask a sharing node to
    /// do local work.
    ///
    /// Sharing forwards to an external server and nothing else. Silently
    /// ignoring a `--model` or `--split` would leave the user with a node that
    /// looks configured for local serving and never does it.
    pub fn validate_shared_endpoint_args(&self) -> anyhow::Result<()> {
        if self.shared_endpoint.is_none() {
            return Ok(());
        }
        // Fail on a malformed or unsupported URL before anything starts.
        crate::plugin::shared_endpoint::normalize_shared_endpoint_url(
            self.shared_endpoint.as_deref().unwrap_or_default(),
        )?;
        anyhow::ensure!(
            !self.client,
            "sharing an endpoint cannot be combined with client mode: a client node never \
             claims the host role, so it cannot serve a shared endpoint to peers"
        );
        anyhow::ensure!(
            self.model.is_empty() && self.gguf.is_empty(),
            "sharing an endpoint cannot be combined with --model or --gguf: this node forwards \
             to an already-running server and never loads a model itself"
        );
        anyhow::ensure!(
            self.mmproj.is_none(),
            "--mmproj is not valid when sharing an endpoint: it applies to a locally loaded model"
        );
        anyhow::ensure!(
            !self.split,
            "--split is not valid when sharing an endpoint: there is no local model to split"
        );
        anyhow::ensure!(
            self.split_topology_lock.is_none(),
            "--split-topology-lock is not valid when sharing an endpoint: there is no local \
             model to split"
        );
        anyhow::ensure!(
            !self.local_model_only,
            "--local-model-only is not valid when sharing an endpoint: it selects a local \
             serving topology with no mesh participation"
        );
        anyhow::ensure!(
            self.plugin.is_none()
                && self.native_serving_plugin.is_none()
                && self.native_serving_plugin_config.is_none()
                && self.native_serving_plugin_state.is_none()
                && self.native_serving_plugin_deadline_ms.is_none(),
            "--plugin runs this process as a plugin host and is not valid when sharing an \
             endpoint"
        );
        anyhow::ensure!(
            self.checkpoint_quantization.is_none() && self.checkpoint_imatrix.is_none(),
            "--checkpoint-quantization and --checkpoint-imatrix require a local model and are \
             not valid when sharing an endpoint"
        );
        Ok(())
    }

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

#[cfg(test)]
mod shared_endpoint_tests {
    use super::*;
    use std::path::PathBuf;

    fn sharing() -> RuntimeOptions {
        RuntimeOptions {
            shared_endpoint: Some("http://localhost:11434".to_string()),
            ..RuntimeOptions::default()
        }
    }

    #[test]
    fn default_serve_allows_local_inference() {
        assert!(RuntimeOptions::default().allows_local_inference());
    }

    #[test]
    fn client_does_not_allow_local_inference() {
        let options = RuntimeOptions {
            client: true,
            ..RuntimeOptions::default()
        };
        assert!(!options.allows_local_inference());
    }

    #[test]
    fn sharing_does_not_allow_local_inference() {
        assert!(!sharing().allows_local_inference());
    }

    #[test]
    fn a_bare_sharing_invocation_is_valid_and_is_not_client_mode() {
        let options = sharing();
        options.validate_shared_endpoint_args().unwrap();
        assert!(
            !options.client,
            "sharing must stay a full mesh participant so it can claim the host role"
        );
    }

    #[test]
    fn sharing_rejects_an_unsupported_upstream_url() {
        let options = RuntimeOptions {
            shared_endpoint: Some("https://api.example.com/v1".to_string()),
            ..RuntimeOptions::default()
        };
        let error = options
            .validate_shared_endpoint_args()
            .expect_err("HTTPS upstreams are not supported yet");
        assert!(error.to_string().contains("HTTPS"));
    }

    #[test]
    fn sharing_rejects_explicit_local_model() {
        let mut options = sharing();
        options.model.push(PathBuf::from("Qwen3-8B-Q4_K_M"));
        let error = options
            .validate_shared_endpoint_args()
            .expect_err("an explicit local model must be rejected");
        assert!(error.to_string().contains("--model"));
    }

    #[test]
    fn sharing_rejects_explicit_gguf() {
        let mut options = sharing();
        options.gguf.push(PathBuf::from("/models/model.gguf"));
        assert!(options.validate_shared_endpoint_args().is_err());
    }

    #[test]
    fn sharing_rejects_split() {
        let options = RuntimeOptions {
            split: true,
            ..sharing()
        };
        assert!(options.validate_shared_endpoint_args().is_err());
    }

    #[test]
    fn sharing_rejects_plugin_process_mode() {
        let options = RuntimeOptions {
            plugin: Some("openai-endpoint".to_string()),
            ..sharing()
        };
        assert!(options.validate_shared_endpoint_args().is_err());
    }

    #[test]
    fn sharing_rejects_client_combination() {
        let options = RuntimeOptions {
            client: true,
            ..sharing()
        };
        let error = options
            .validate_shared_endpoint_args()
            .expect_err("client and sharing are contradictory");
        assert!(error.to_string().contains("host role"));
    }

    #[test]
    fn sharing_rejects_local_model_only_topology() {
        let options = RuntimeOptions {
            local_model_only: true,
            ..sharing()
        };
        assert!(options.validate_shared_endpoint_args().is_err());
    }

    #[test]
    fn sharing_rejects_checkpoint_quantization() {
        let options = RuntimeOptions {
            checkpoint_quantization: Some("Q4_K_M".to_string()),
            ..sharing()
        };
        assert!(options.validate_shared_endpoint_args().is_err());
    }

    /// Normal serve and client startups must be completely unaffected.
    #[test]
    fn non_sharing_startups_skip_the_guard_entirely() {
        let mut serve = RuntimeOptions::default();
        serve.model.push(PathBuf::from("Qwen3-8B-Q4_K_M"));
        serve.split = true;
        assert!(serve.validate_shared_endpoint_args().is_ok());

        let client = RuntimeOptions {
            client: true,
            ..RuntimeOptions::default()
        };
        assert!(client.validate_shared_endpoint_args().is_ok());
    }
}
