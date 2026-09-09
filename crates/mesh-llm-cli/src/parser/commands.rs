use clap::{Parser, Subcommand, ValueEnum};
use std::ffi::OsString;
use std::net::IpAddr;
use std::path::PathBuf;

use crate::benchmark::{BenchmarkCommand, GpuBenchmarkBackend};
use crate::models;
use crate::runtime::RuntimeCommand;
use mesh_llm_events::LogFormat;
use mesh_llm_events::audit::{AuditLevel, AuditLogFormat};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum BinaryFlavor {
    #[default]
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum TrustPolicy {
    #[default]
    Off,
    PreferOwned,
    RequireOwned,
    Allowlist,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum MeshDiscoveryMode {
    #[default]
    Nostr,
    Mdns,
}

impl MeshDiscoveryMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nostr => "nostr",
            Self::Mdns => "mdns",
        }
    }

    pub const fn source(self) -> &'static str {
        match self {
            Self::Nostr => "nostr-relay",
            Self::Mdns => "mdns-sd",
        }
    }

    pub const fn scope(self) -> DiscoveryScope {
        match self {
            Self::Nostr => DiscoveryScope::Public,
            Self::Mdns => DiscoveryScope::Lan,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryScope {
    Public,
    Lan,
}

impl DiscoveryScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Lan => "lan",
        }
    }
}

/// Parse a `URL=TOKEN` pair for `--relay-auth`. Splits on the first `=` only,
/// so tokens may contain `=` (base64 padding, JWTs).
///
/// Error messages must never include the token portion of the input —
/// `--relay-auth` carries bearer credentials, and a parse failure could
/// otherwise leak them into terminal output, logs, and bug reports. The URL
/// is safe to echo back (it's the public identity of the relay).
fn parse_relay_auth_pair(s: &str) -> Result<(String, String), String> {
    let Some((url, token)) = s.split_once('=') else {
        return Err("expected URL=TOKEN, no '=' separator found (token redacted)".to_string());
    };
    if url.is_empty() {
        return Err("expected URL=TOKEN, got empty URL (token redacted)".to_string());
    }
    if token.is_empty() {
        return Err(format!(
            "expected URL=TOKEN, got empty token for URL {url:?}"
        ));
    }
    Ok((url.to_string(), token.to_string()))
}

#[cfg(test)]
mod relay_auth_parser_tests {
    use super::parse_relay_auth_pair;

    #[test]
    fn parses_simple_pair() {
        let (url, token) = parse_relay_auth_pair("https://r.example/=abc123").unwrap();
        assert_eq!(url, "https://r.example/");
        assert_eq!(token, "abc123");
    }

    #[test]
    fn preserves_equals_in_token() {
        // Base64-padded tokens and NIP-98-style payloads often contain `=`.
        let (_, token) = parse_relay_auth_pair("https://r/=eyJhbGciOiJFZERTQSJ9.payload==")
            .expect("token with '=' must parse");
        assert_eq!(token, "eyJhbGciOiJFZERTQSJ9.payload==");
    }

    #[test]
    fn rejects_missing_separator() {
        assert!(parse_relay_auth_pair("no-separator").is_err());
    }

    #[test]
    fn rejects_empty_url() {
        assert!(parse_relay_auth_pair("=token").is_err());
    }

    #[test]
    fn rejects_empty_token() {
        assert!(parse_relay_auth_pair("https://r/=").is_err());
    }

    #[test]
    fn parser_errors_never_leak_token_portion() {
        // --relay-auth carries bearer credentials; if parsing fails, the
        // token portion of the input must never appear in the error
        // message (which lands in terminal output, logs, and bug reports).
        // The URL is safe to echo back — it's the public identity of the
        // relay — but everything after the first `=` is secret.
        let secret_token = "super-secret-bearer-token-xyz-12345";

        // Case 1: no `=` separator. Whole input is treated as a malformed
        // URL-or-token blob; we cannot tell which it is, so redact both.
        let err = parse_relay_auth_pair(secret_token).expect_err("should fail");
        assert!(
            !err.contains(secret_token),
            "missing-separator error must not echo the input: {err}"
        );

        // Case 2: empty URL (`=token`). URL is empty, the token portion is
        // the secret — must not appear.
        let err = parse_relay_auth_pair(&format!("={secret_token}")).expect_err("should fail");
        assert!(
            !err.contains(secret_token),
            "empty-URL error must not echo the token: {err}"
        );

        // Case 3: empty token (`URL=`). Token is empty, no secret to leak;
        // the URL is fine to include and helps the user diagnose.
        let err = parse_relay_auth_pair("https://r.example/=").expect_err("should fail");
        assert!(
            err.contains("https://r.example/"),
            "empty-token error should name the URL: {err}"
        );
    }
}

#[derive(Subcommand, Debug)]
pub enum TrustCommand {
    /// Add an owner to the local trust store allowlist.
    Add {
        /// Owner ID to trust.
        owner_id: String,
        /// Optional human label for this owner.
        #[arg(long)]
        label: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Remove an owner from the local trust store allowlist.
    Remove {
        /// Owner ID to remove.
        owner_id: String,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Show the current trust store contents.
    List {
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
pub enum AuthCommand {
    /// Generate a new owner keypair and save to keystore.
    Init {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Overwrite an existing keystore.
        #[arg(long)]
        force: bool,
        /// Skip passphrase prompt (store keys unencrypted).
        #[arg(long, conflicts_with = "keychain")]
        no_passphrase: bool,
        /// Store a random unlock passphrase in the OS keychain (macOS Keychain,
        /// Windows Credential Manager, Linux Secret Service). New keystores
        /// already default to this when a backend is available; use this flag
        /// to force it when overwriting an existing keystore.
        #[arg(long)]
        keychain: bool,
    },
    /// Show current owner identity status.
    Status {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Path to the node ownership certificate.
        #[arg(long)]
        node_ownership: Option<PathBuf>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Sign the current node identity with the existing owner keystore.
    SignNode {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Output path for the signed node certificate.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Optional hostname hint attached to the certificate.
        #[arg(long)]
        hostname_hint: Option<String>,
        /// Optional human label attached to this node certificate.
        #[arg(long)]
        node_label: Option<String>,
        /// Certificate lifetime in hours.
        #[arg(long, default_value = "168")]
        expires_in_hours: u64,
    },
    /// Renew the local node ownership certificate in place.
    RenewNode {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Output path for the signed node certificate.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Optional hostname hint attached to the certificate.
        #[arg(long)]
        hostname_hint: Option<String>,
        /// Optional human label attached to this node certificate.
        #[arg(long)]
        node_label: Option<String>,
        /// Certificate lifetime in hours.
        #[arg(long, default_value = "168")]
        expires_in_hours: u64,
    },
    /// Verify a node ownership certificate.
    VerifyNode {
        /// Path to the signed node certificate.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Override the node ID to verify against.
        #[arg(long)]
        node_id: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
        /// Override trust policy used for verification.
        #[arg(long = "verify-trust-policy", value_enum)]
        trust_policy: Option<TrustPolicy>,
    },
    /// Rotate the local node identity key.
    RotateNode {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Output path for the signed node certificate.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Optional hostname hint attached to the certificate.
        #[arg(long)]
        hostname_hint: Option<String>,
        /// Optional human label attached to this node certificate.
        #[arg(long)]
        node_label: Option<String>,
        /// Certificate lifetime in hours.
        #[arg(long, default_value = "168")]
        expires_in_hours: u64,
        /// Revoke the current certificate and node ID in the local trust store first.
        #[arg(long)]
        revoke_current: bool,
        /// Optional revocation reason stored in the trust store.
        #[arg(long)]
        reason: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Revoke an owner in the local trust store.
    RevokeOwner {
        /// Owner ID to revoke.
        owner_id: String,
        /// Optional reason stored in the trust store.
        #[arg(long)]
        reason: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Revoke a node certificate or node ID in the local trust store.
    RevokeNode {
        /// Certificate ID to revoke.
        #[arg(long)]
        cert_id: Option<String>,
        /// Node endpoint ID to revoke.
        #[arg(long)]
        node_id: Option<String>,
        /// Optional reason stored in the trust store.
        #[arg(long)]
        reason: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Rotate the existing owner keystore identity.
    RotateOwner {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Skip passphrase prompt (store keys unencrypted).
        #[arg(long)]
        no_passphrase: bool,
        /// Overwrite an existing backup file if present.
        #[arg(long)]
        force: bool,
    },
    /// Manage the local trust store.
    Trust {
        #[command(subcommand)]
        command: TrustCommand,
    },
}

#[derive(Subcommand, Debug)]
pub enum GpuCommand {
    /// Detect and benchmark local GPUs, rewriting the cached fingerprint.
    Detect {
        /// Print machine-readable JSON output.
        #[arg(long)]
        json: bool,
    },
    /// Run one backend benchmark probe and print raw JSON output.
    #[command(name = "run-benchmark", hide = true)]
    RunBenchmark {
        #[arg(long, value_enum)]
        backend: GpuBenchmarkBackend,
    },
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum MeshGuardrailCliMode {
    #[default]
    Disabled,
    Metrics,
    Enforce,
}

impl MeshGuardrailCliMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Metrics => "metrics",
            Self::Enforce => "enforce",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum SpeculativeNgramProposerCli {
    Cache,
    Suffix,
}

impl SpeculativeNgramProposerCli {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cache => "cache",
            Self::Suffix => "suffix",
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "mesh-llm",
    version = mesh_llm_build_info::BUILD_VERSION,
    about = "Pool GPUs over the internet for LLM inference",
    after_help = "Preferred runtime entrypoints:\n  mesh-llm serve\n  mesh-llm serve --model Qwen3-8B-Q4_K_M\n  mesh-llm client --auto\n  mesh-llm gpus\n\n`mesh-llm serve` loads startup models from ~/.mesh-llm/config.toml.\nRun with --help-advanced for all options.\n\nExternal backends (vLLM, TGI, Ollama):\n  Install the plugin:\n    mesh-llm plugins install openai-endpoint\n  Add to ~/.mesh-llm/config.toml:\n    [[plugin]]\n    name = \"openai-endpoint\"\n    url = \"http://gpu-box:8000/v1\"\n  Then: mesh-llm serve     (or: mesh-llm client  for client-only mode)\n\nFlash-MoE SSD backend:\n  Install the plugin:\n    mesh-llm plugins install flash-moe\n  Add [[plugin]] name = \"flash-moe\" with url or plugin-owned args.\n  Then: mesh-llm serve     (or: mesh-llm client  for client-only mode)"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Terminal output format for app-owned runtime events.
    #[arg(long, value_enum, default_value_t = LogFormat::Pretty)]
    pub log_format: LogFormat,

    /// Audit log file path (enables audit logging when set).
    #[arg(long, hide = true)]
    pub audit_log_path: Option<PathBuf>,

    /// Audit log format.
    #[arg(long, value_enum, default_value_t = AuditLogFormat::JsonLines, hide = true)]
    pub audit_log_format: AuditLogFormat,

    /// Minimum audit log level.
    #[arg(long, value_enum, default_value_t = AuditLevel::Info, hide = true)]
    pub audit_log_level: AuditLevel,

    /// Enable mesh runtime debug output; set MESH_LLM_DEBUG_NATIVE_VERBOSE=1 for verbose llama.cpp native logs.
    #[arg(long)]
    pub debug: bool,

    /// OTLP/gRPC endpoint for embedded Skippy debug telemetry, for example http://127.0.0.1:14317.
    #[arg(long, hide = true)]
    pub skippy_metrics_otlp_grpc: Option<String>,

    /// Server-side mesh guardrail mode for hosted Skippy backends.
    #[arg(long = "mesh-guardrails", value_enum, default_value_t = MeshGuardrailCliMode::Disabled)]
    pub mesh_guardrails: MeshGuardrailCliMode,

    /// Show all options (including advanced/niche ones).
    #[arg(long, hide = true)]
    pub help_advanced: bool,

    /// Join a mesh via invite token (can repeat).
    #[arg(long, short)]
    pub join: Vec<String>,

    /// Discover a mesh and join it.
    #[arg(long, default_missing_value = "", num_args = 0..=1)]
    pub discover: Option<String>,

    /// Auto-join the best mesh found via discovery.
    #[arg(long)]
    pub auto: bool,

    /// Discovery provider for --auto, --discover, --publish, and the discover command.
    #[arg(long, value_enum, default_value_t = MeshDiscoveryMode::Nostr, global = true)]
    pub mesh_discovery_mode: MeshDiscoveryMode,

    /// Model to serve (path, remote catalog name, or Hugging Face ref).
    #[arg(long)]
    pub model: Vec<PathBuf>,

    /// Raw local GGUF file to serve directly (repeatable).
    #[arg(long)]
    pub gguf: Vec<PathBuf>,

    /// Explicit mmproj sidecar for the primary served model.
    #[arg(long, hide = true)]
    pub mmproj: Option<PathBuf>,

    /// Quantize SafeTensors weights while loading.
    #[arg(
        long = "quant",
        visible_alias = "checkpoint-quantization",
        long_help = "Quantize SafeTensors weights while loading. The default is preserve.\n\nValid recipes: preserve, F32, F16, BF16, Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ1_S, IQ1_M, TQ1_0, TQ2_0, Q2_K, Q2_K_S, IQ3_XS, IQ3_XXS, IQ3_S, IQ3_M, Q3_K_S, Q3_K_M, Q3_K_L, IQ4_NL, IQ4_XS, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0, MXFP4_MOE. Some low-bit recipes require --checkpoint-imatrix."
    )]
    pub checkpoint_quantization: Option<String>,

    /// Importance matrix for load-time checkpoint quantization.
    #[arg(long, hide = true)]
    pub checkpoint_imatrix: Option<PathBuf>,

    /// API port (default: 9337).
    #[arg(long, default_value = "9337")]
    pub port: u16,

    /// Serve one model directly through the local OpenAI API without starting a mesh node.
    #[arg(long)]
    pub local_model_only: bool,

    /// Load a native serving plugin into the dedicated local-model path.
    #[arg(
        long,
        hide = true,
        requires = "local_model_only",
        requires_all = [
            "native_serving_plugin_config",
            "native_serving_plugin_state",
            "native_serving_plugin_deadline_ms"
        ]
    )]
    pub native_serving_plugin: Option<PathBuf>,

    /// Absolute path to the native serving plugin configuration.
    #[arg(long, hide = true, requires = "native_serving_plugin")]
    pub native_serving_plugin_config: Option<PathBuf>,

    /// Absolute path to the native serving plugin state directory.
    #[arg(long, hide = true, requires = "native_serving_plugin")]
    pub native_serving_plugin_state: Option<PathBuf>,

    /// Mesh-enforced hard proposal deadline in milliseconds.
    #[arg(
        long,
        hide = true,
        requires = "native_serving_plugin",
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    pub native_serving_plugin_deadline_ms: Option<u64>,

    /// Run as a client — no GPU, no model needed.
    #[arg(long)]
    pub client: bool,

    /// Normalization target for the `share <url>` positional.
    ///
    /// `share` is rewritten to this flag before clap parsing so ordinary
    /// runtime flags work on either side of the URL, the same way `client` is
    /// rewritten to `--client`. Not part of the documented surface.
    #[arg(long, hide = true)]
    pub shared_endpoint: Option<String>,

    /// Web console port (default: 3131).
    #[arg(long, default_value = "3131")]
    pub console: u16,

    /// Disable the embedded web UI but keep the management API on the --console port.
    #[arg(long)]
    pub headless: bool,

    /// Write passive swarm debug capture JSONL to this local directory (opt-in, no telemetry egress).
    #[arg(long)]
    pub swarm_capture: Option<PathBuf>,

    /// Publish this mesh for discovery by other nodes.
    /// Without this flag, your mesh is private and only joinable via invite token.
    #[arg(long)]
    pub publish: bool,

    /// Human-readable name for this mesh (shown in discovery when combined with --publish).
    /// Naming a mesh does NOT make it publicly discoverable — use --publish for that.
    #[arg(long)]
    pub mesh_name: Option<String>,

    /// Region tag, e.g. "US", "EU", "AU" (shown in discovery).
    #[arg(long)]
    pub region: Option<String>,

    /// Minimum mesh-llm node version required when creating a new mesh.
    #[arg(long)]
    pub min_node_version: Option<String>,

    /// Maximum mesh-llm node version allowed when creating a new mesh.
    #[arg(long)]
    pub max_node_version: Option<String>,

    /// Minimum protocol generation required when creating a new mesh.
    #[arg(long)]
    pub min_protocol_version: Option<u32>,

    /// Maximum protocol generation allowed when creating a new mesh.
    #[arg(long)]
    pub max_protocol_version: Option<u32>,

    /// Require release attestation when creating a new mesh.
    #[arg(long)]
    pub require_release_attestation: bool,

    /// Allowed release signer key for mesh creation-time attestation policy (repeatable).
    #[arg(long = "release-signer-key")]
    pub release_signer_key: Vec<String>,

    /// Display name for this node.
    #[arg(long)]
    pub name: Option<String>,

    /// Internal plugin service mode.
    #[arg(long, hide = true)]
    pub plugin: Option<String>,

    /// Update mesh-llm before continuing for release-bundle installs if a newer bundled release is available.
    #[arg(long, global = true)]
    pub auto_update: bool,

    // ── Advanced options (hidden from default --help) ─────────────
    /// Override speculative decoding (`mtp`, `ngram-cache`, `ngram-suffix`, or a package strategy id).
    #[arg(long, hide = true)]
    pub speculative_strategy: Option<String>,

    /// Minimum matching N-gram length for the request-local MTP cache extension.
    #[arg(long, hide = true)]
    pub speculative_ngram_min: Option<u32>,

    /// Maximum matching N-gram length for the request-local MTP cache extension.
    #[arg(long, hide = true)]
    pub speculative_ngram_max: Option<u32>,

    /// Cap N-gram tokens proposed in one verify window.
    #[arg(long, hide = true)]
    pub speculative_ngram_max_proposal_tokens: Option<u32>,

    /// Standalone N-gram proposer kind (`cache` or `suffix`).
    #[arg(long, hide = true, value_enum)]
    pub speculative_ngram_proposer: Option<SpeculativeNgramProposerCli>,

    /// Maximum N-gram extension length for a composite MTP strategy.
    #[arg(long, hide = true)]
    pub speculative_extension_max_tokens: Option<u32>,

    /// Native MTP rejection cooldown in generated tokens.
    #[arg(long, hide = true)]
    pub speculative_native_mtp_reject_cooldown_tokens: Option<u32>,

    /// Suppress native MTP drafts while its rejection cooldown is active.
    #[arg(long, hide = true)]
    pub speculative_native_mtp_suppress_cooldown_drafts: bool,

    /// Keep native MTP drafts during its rejection cooldown.
    #[arg(
        long,
        hide = true,
        conflicts_with = "speculative_native_mtp_suppress_cooldown_drafts"
    )]
    pub speculative_native_mtp_allow_cooldown_drafts: bool,

    /// Maximum native MTP drafts suppressed by a cooldown.
    #[arg(long, hide = true)]
    pub speculative_native_mtp_suppress_cooldown_draft_limit: Option<u32>,

    /// Minimum tokens to include in a pipelined verify window.
    #[arg(long, hide = true)]
    pub speculative_verify_window_min_tokens: Option<u32>,

    /// Maximum tokens to include in a pipelined verify window.
    #[arg(long, hide = true)]
    pub speculative_verify_window_max_tokens: Option<u32>,

    /// Number of in-flight pipelined verify windows.
    #[arg(long, hide = true)]
    pub speculative_verify_window_pipeline_depth: Option<u32>,

    /// Draft model for speculative decoding.
    #[arg(long, hide = true)]
    pub draft: Option<PathBuf>,

    /// Max draft tokens (default: 8).
    #[arg(long, default_value = "8", hide = true)]
    pub draft_max: u16,

    /// Disable automatic draft model detection.
    #[arg(long, hide = true)]
    pub no_draft: bool,

    /// Force tensor split even if the model fits on one node.
    #[arg(long, hide = true)]
    pub split: bool,

    /// Pin split-serving node order and layer ranges from a JSON topology lock.
    #[arg(long, value_name = "PATH", requires = "split", hide = true)]
    pub split_topology_lock: Option<PathBuf>,

    /// Override context size (tokens). Default: auto-scaled to available VRAM.
    #[arg(long, hide = true)]
    pub ctx_size: Option<u32>,

    /// Cap VRAM used for planning, local-fit decisions, and mesh advertisement (GB).
    #[arg(long)]
    pub max_vram: Option<f64>,

    /// Disable broadcasting GPU name, hostname, VRAM, and reserved bytes to peers. By default all nodes announce this hardware info.
    #[arg(long = "no-enumerate-host", hide = true)]
    pub no_enumerate_host: bool,

    /// Path to bundled mesh support binaries.
    #[arg(long, hide = true)]
    pub bin_dir: Option<PathBuf>,

    /// Override which bundled llama.cpp flavor to use.
    #[arg(long, value_enum)]
    pub llama_flavor: Option<BinaryFlavor>,

    /// Device override for local backend selection.
    #[arg(long, hide = true)]
    pub device: Option<String>,

    /// Deprecated tensor split override retained for CLI compatibility.
    #[arg(long, hide = true)]
    pub tensor_split: Option<String>,

    /// Override iroh relay URLs.
    #[arg(long, hide = true)]
    pub relay: Vec<String>,

    /// Per-relay bearer token for gated iroh relays, formatted as
    /// `URL=TOKEN`. Repeatable. The token is sent as
    /// `Authorization: Bearer <TOKEN>` on the WebSocket upgrade to the
    /// matching `--relay` URL. Relays not listed here register without
    /// authentication (the correct behavior for public relays).
    ///
    /// Splits on the first `=` only, so tokens may contain `=` (base64
    /// padding, JWTs, etc.).
    #[arg(long = "relay-auth", value_parser = parse_relay_auth_pair, hide = true)]
    pub relay_auth: Vec<(String, String)>,

    /// Disable iroh relays even when public mesh discovery would normally use them.
    #[arg(long = "disable-iroh-relays", hide = true)]
    pub disable_iroh_relays: bool,

    /// Bind QUIC to a fixed UDP port (for NAT port forwarding).
    #[arg(long, hide = true)]
    pub bind_port: Option<u16>,

    /// Bind mesh QUIC to a specific local IP address.
    #[arg(long, hide = true)]
    pub bind_ip: Option<IpAddr>,

    /// Bind to 0.0.0.0 (for containers/Fly.io).
    #[arg(long, hide = true)]
    pub listen_all: bool,

    /// Stop advertising when N clients connected.
    #[arg(long, hide = true)]
    pub max_clients: Option<usize>,

    /// Custom Nostr relay URLs.
    #[arg(long, hide = true)]
    pub nostr_relay: Vec<String>,

    /// Ignored (backward compat).
    #[arg(long, hide = true)]
    pub no_console: bool,

    /// Optional path to the mesh-llm config file.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Path to the owner keystore used to attest this node.
    #[arg(long)]
    pub owner_key: Option<PathBuf>,

    /// Bind address for the owner-control listener. Defaults to 127.0.0.1:0 when owner identity is configured.
    #[arg(long, hide = true)]
    pub control_bind: Option<std::net::SocketAddr>,

    /// Advertised owner-control address encoded into the local-only bootstrap token.
    #[arg(long, hide = true)]
    pub control_advertise_addr: Option<std::net::SocketAddr>,

    /// Fail startup if owner attestation cannot be loaded or signed.
    #[arg(long)]
    pub owner_required: bool,

    /// Optional human label attached to this node certificate.
    #[arg(long)]
    pub node_label: Option<String>,

    /// Override peer ownership trust policy.
    #[arg(long, value_enum)]
    pub trust_policy: Option<TrustPolicy>,

    /// Add trusted owner IDs on top of the local trust store.
    #[arg(long)]
    pub trust_owner: Vec<String>,

    /// Internal: set when this node joined via Nostr discovery (not --join).
    #[arg(skip)]
    pub nostr_discovery: bool,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Share an already-running OpenAI-compatible HTTP server. No local inference runtime.
    Share {
        /// Existing API base URL, for example http://localhost:11434.
        url: String,
    },
    /// Serve local models and join or publish a mesh.
    Serve,
    /// Run as a client-only mesh node with no local model required.
    Client,
    /// Manage model storage, migration, and update checks.
    Models {
        #[command(subcommand)]
        command: models::ModelsCommand,
    },
    /// Download a model from the remote catalog or Hugging Face
    Download {
        /// Model name (e.g. "Qwen2.5-32B-Instruct-Q4_K_M" or just "32b")
        name: Option<String>,
        /// Also download the recommended draft model for speculative decoding
        #[arg(long)]
        draft: bool,
    },
    /// Update mesh-llm to a bundled release and exit.
    Update {
        /// Install this specific release tag or version (e.g. v0.60.0 or 0.60.0-rc.1).
        #[arg(long)]
        version: Option<String>,
        /// Install this release bundle flavor instead of the default installed flavor.
        #[arg(long, value_enum, conflicts_with = "detect_flavor")]
        flavor: Option<BinaryFlavor>,
        /// Re-detect the best host backend flavor before selecting the release bundle.
        #[arg(long, conflicts_with = "flavor")]
        detect_flavor: bool,
    },
    /// Inspect local GPUs, stable IDs, and cached bandwidth.
    #[command(alias = "gpu")]
    Gpus {
        /// Print machine-readable JSON output.
        #[arg(long)]
        json: bool,
        #[command(subcommand)]
        command: Option<GpuCommand>,
    },
    /// Inspect and manage native runtimes.
    Runtime {
        #[command(subcommand)]
        command: Option<RuntimeCommand>,
    },
    /// Inspect and validate mesh-llm configuration files.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Diagnose local mesh, runtime, and split-readiness problems.
    Doctor {
        /// Print machine-readable JSON for the default doctor report.
        #[arg(long)]
        json: bool,
        #[command(subcommand)]
        command: Option<DoctorCommand>,
    },
    /// Bootstrap a new installation.
    Setup {
        /// Automatically answer yes to prompts.
        #[arg(long)]
        yes: bool,
        /// Run without prompting for interactive input.
        #[arg(long = "no-interactive")]
        no_interactive: bool,
        /// Install and enable the mesh-llm service.
        #[arg(long, conflicts_with = "no_service")]
        service: bool,
        /// Skip installing and enabling the mesh-llm service.
        #[arg(long = "no-service", conflicts_with = "service")]
        no_service: bool,
        /// Skip downloading or configuring the native runtime.
        #[arg(long = "skip-runtime")]
        skip_runtime: bool,
        /// Print detailed setup paths, commands, and follow-up guidance.
        #[arg(long)]
        verbose: bool,
    },
    /// Remove mesh-llm binaries, service files, and optional caches.
    Uninstall {
        /// Print what would be removed without changing the machine.
        #[arg(long)]
        dry_run: bool,
        /// Do not prompt before removing files and services.
        #[arg(long)]
        yes: bool,
        /// Preserve native runtime caches.
        #[arg(long)]
        keep_cache: bool,
        /// Preserve setup-owned service helper files.
        #[arg(long)]
        keep_service_files: bool,
        /// Also remove ~/.mesh-llm configuration and identity data.
        #[arg(long, conflicts_with = "keep_config")]
        purge_config: bool,
        /// Explicitly preserve ~/.mesh-llm configuration and identity data.
        #[arg(long, conflicts_with = "purge_config")]
        keep_config: bool,
        /// Override the installed binary path to remove.
        #[arg(long)]
        binary_path: Option<std::path::PathBuf>,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
        /// Print detailed cleanup steps and removed paths.
        #[arg(long)]
        verbose: bool,
    },
    /// Load a local model into a running mesh-llm instance.
    Load {
        /// Model name/path/url to load
        name: String,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Unload a local model from a running mesh-llm instance.
    #[command(alias = "drop")]
    Unload {
        /// Model name to unload
        name: String,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Show local model status on a running mesh-llm instance.
    Status {
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Discover meshes and optionally auto-join one.
    Discover {
        /// Filter by mesh name (case-insensitive exact match)
        #[arg(long)]
        name: Option<String>,
        /// Filter by model name (substring match)
        #[arg(long)]
        model: Option<String>,
        /// Filter by minimum VRAM (GB)
        #[arg(long)]
        min_vram: Option<f64>,
        /// Filter by region
        #[arg(long)]
        region: Option<String>,
        /// Print the invite token of the best match (for piping to --join)
        #[arg(long)]
        auto: bool,
        /// Nostr relay URLs (default: see DEFAULT_RELAYS)
        #[arg(long)]
        relay: Vec<String>,
    },
    /// Rotate all identity keys (node + Nostr).
    #[command(hide = true)]
    RotateKey,
    /// Launch Goose with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on --port, this auto-joins the mesh as a client.
    #[command(name = "goose")]
    Goose {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// API port for mesh-llm (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Launch Claude Code with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on --port, this auto-joins the mesh as a client.
    #[command(name = "claude")]
    Claude {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// API port for mesh-llm (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Launch pi with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on a loopback/localhost target, this auto-joins the mesh as a client.
    /// Writes a mesh provider into ~/.pi/agent/models.json and launches pi unless --write is set.
    #[command(name = "pi")]
    Pi {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// mesh-llm host or URL for Pi (default: 127.0.0.1:9337)
        #[arg(long, default_value = "127.0.0.1:9337")]
        host: String,
        /// Write the mesh provider config to Pi's models.json instead of launching.
        #[arg(long)]
        write: bool,
    },
    /// Launch OpenCode with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on a loopback/localhost target, this auto-joins the mesh as a client.
    #[command(name = "opencode")]
    Opencode {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// mesh-llm host or URL for OpenCode (default: 127.0.0.1:9337)
        #[arg(long, default_value = "127.0.0.1:9337")]
        host: String,
        /// Write the mesh provider config to opencode's config file instead of launching.
        #[arg(long)]
        write: bool,
    },
    /// Stop running mesh-llm processes.
    Stop,
    /// Plugin management.
    #[command(name = "plugins", alias = "plugin")]
    Plugin {
        #[command(subcommand)]
        command: PluginCommand,
    },
    /// Install agent skills exposed by installed plugins.
    Skills {
        #[command(subcommand)]
        command: SkillCommand,
    },
    /// Benchmark and compare model/runtime strategies.
    Benchmark {
        #[command(subcommand)]
        command: BenchmarkCommand,
    },
    /// Prepare a model for distributed inference by splitting it into
    /// per-layer files on HF compute.
    ///
    /// Submits an HF Job that builds skippy-model-package from source,
    /// splits the model, publishes the layer package, and updates the
    /// meshllm/catalog.
    #[command(name = "model-prepare", hide = true, alias = "model-package")]
    ModelPrepare {
        /// Source HuggingFace model ref (e.g. unsloth/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL).
        source_repo: Option<String>,

        /// Quantization variant (deprecated; prefer source refs like repo:Q4_K_M).
        #[arg(long)]
        quant: Option<String>,

        /// Target repo for the layer package (auto-derived if omitted).
        #[arg(long)]
        target: Option<String>,

        /// Override model ID in the manifest.
        #[arg(long)]
        model_id: Option<String>,

        /// HF Job hardware flavor. Use auto for the default CPU splitter baseline.
        #[arg(long, default_value = "auto")]
        flavor: String,

        /// Requested job timeout; raised automatically by model-size minimums.
        #[arg(long, default_value = "1h")]
        timeout: String,

        /// Branch or tag of mesh-llm to build in the job [default: main].
        #[arg(long, default_value = "main")]
        mesh_llm_ref: String,

        /// Explicitly keep this as a dry run. This is the default unless --confirm is set.
        #[arg(long)]
        dry_run: bool,

        /// Actually submit the HF Job. Without this, the command only prints plan, spec, and max cost.
        #[arg(long)]
        confirm: bool,

        /// Stream job logs after submission until completion.
        #[arg(long)]
        follow: bool,

        /// Emit JSON output.
        #[arg(long)]
        json: bool,

        /// Check status of a previously submitted job.
        #[arg(long)]
        status: Option<String>,

        /// Fetch logs for a previously submitted job.
        #[arg(long)]
        logs: Option<String>,

        /// Cancel a running job.
        #[arg(long)]
        cancel: Option<String>,

        /// List recent model-package jobs.
        #[arg(long)]
        list: bool,

        /// Upload the latest job script to the meshllm bucket (requires org access).
        #[arg(long)]
        update_script: bool,
    },
    /// Manage owner identity and keystore.
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
    /// Run a CLI command contributed by a configured plugin.
    #[command(external_subcommand)]
    ExternalPlugin(Vec<OsString>),
}

#[derive(Subcommand, Debug)]
pub enum ConfigCommand {
    /// Validate a config TOML file without starting a node.
    Validate {
        /// Config TOML path to validate. Defaults to --config, MESH_LLM_CONFIG, or ~/.mesh-llm/config.toml.
        #[arg(long = "config-path")]
        config_path: Option<PathBuf>,
        /// Print machine-readable JSON output.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum PluginCommand {
    /// Install a native plugin from the catalog, GitHub, or a local release archive.
    Install {
        /// Plugin catalog name, GitHub owner/repo, or GitHub URL.
        #[arg(required_unless_present = "archive", conflicts_with = "archive")]
        reference: Option<String>,
        /// Install a local .tar.gz or .zip release archive. Requires --name.
        #[arg(long, value_name = "PATH", requires = "name")]
        archive: Option<PathBuf>,
        /// Plugin name used to validate a local archive. Required with --archive.
        #[arg(long, requires = "archive")]
        name: Option<String>,
        /// Version recorded for a local archive install. Defaults to dev.
        #[arg(long, requires = "archive")]
        version: Option<String>,
    },
    /// Update an installed native plugin.
    Update {
        /// Plugin name.
        name: String,
    },
    /// Enable an installed native plugin.
    Enable {
        /// Plugin name.
        name: String,
    },
    /// Disable an installed native plugin.
    Disable {
        /// Plugin name.
        name: String,
    },
    /// Delete an installed native plugin.
    Delete {
        /// Plugin name.
        name: String,
    },
    /// Show installed plugin details.
    Info {
        /// Plugin name.
        name: String,
    },
    /// Search the plugin catalog.
    Search {
        /// Optional search query.
        query: Option<String>,
    },
    /// List installed, auto-registered, and configured plugins.
    List,
}

#[derive(Subcommand, Debug)]
pub enum SkillCommand {
    /// Install skills exposed by installed plugins into supported agent skill folders.
    Install {
        /// Agent to install for. Repeat to install to several agents.
        #[arg(long, value_enum, conflicts_with = "all")]
        agent: Vec<SkillAgentArg>,
        /// Install to all supported agent locations, even if the agent is not detected.
        #[arg(long)]
        all: bool,
        /// Show what would be installed without writing files.
        #[arg(long)]
        dry_run: bool,
        /// Replace an existing non-mesh-managed skill with the same directory name.
        #[arg(long)]
        force: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum SkillAgentArg {
    Global,
    Goose,
    Pi,
    Codex,
    Opencode,
    Claude,
}

#[derive(Subcommand, Debug)]
pub enum DoctorCommand {
    /// Diagnose split-readiness for a model on a running local mesh node.
    Split {
        /// Model ref/name to diagnose.
        #[arg(long, visible_alias = "model")]
        model_ref: String,
        /// Console/API port of the running mesh-llm instance.
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
        /// Write a split and Skippy diagnostic bundle to this directory.
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ModelSearchSort, ModelsCommand};
    use clap::{CommandFactory, Parser, error::ErrorKind};
    use mesh_llm_events::LogFormat;

    #[test]
    fn native_serving_plugin_deadline_rejects_zero() {
        let normalized = crate::parser::normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--local-model-only",
            "--native-serving-plugin",
            "/tmp/plugin.dylib",
            "--native-serving-plugin-config",
            "/tmp/plugin.json",
            "--native-serving-plugin-state",
            "/tmp/plugin-state",
            "--native-serving-plugin-deadline-ms",
            "0",
        ]);
        let error = Cli::try_parse_from(normalized.normalized).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn serve_parses_speculative_decode_overrides() {
        let normalized = crate::parser::normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--speculative-strategy",
            "mtp-cache",
            "--speculative-ngram-min",
            "2",
            "--speculative-ngram-max",
            "6",
            "--speculative-extension-max-tokens",
            "8",
            "--speculative-native-mtp-allow-cooldown-drafts",
            "--speculative-verify-window-pipeline-depth",
            "3",
        ]);
        let cli = Cli::try_parse_from(normalized.normalized).expect("clap parse");
        assert_eq!(cli.speculative_strategy.as_deref(), Some("mtp-cache"));
        assert_eq!(cli.speculative_ngram_min, Some(2));
        assert_eq!(cli.speculative_ngram_max, Some(6));
        assert_eq!(cli.speculative_extension_max_tokens, Some(8));
        assert!(cli.speculative_native_mtp_allow_cooldown_drafts);
        assert_eq!(cli.speculative_verify_window_pipeline_depth, Some(3));
    }

    #[test]
    fn serve_parses_checkpoint_quantization_overrides() {
        let normalized = crate::parser::normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--quant",
            "IQ2_XS",
            "--checkpoint-imatrix",
            "/tmp/qwen.imatrix",
        ]);
        let cli = Cli::try_parse_from(normalized.normalized).expect("clap parse");

        assert_eq!(cli.checkpoint_quantization.as_deref(), Some("IQ2_XS"));
        assert_eq!(
            cli.checkpoint_imatrix,
            Some(PathBuf::from("/tmp/qwen.imatrix"))
        );
    }

    #[test]
    fn serve_accepts_explicit_checkpoint_quantization_alias() {
        let normalized = crate::parser::normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--checkpoint-quantization",
            "Q4_K_M",
        ]);
        let cli = Cli::try_parse_from(normalized.normalized).expect("clap parse");

        assert_eq!(cli.checkpoint_quantization.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn serve_parses_standalone_suffix_strategy() {
        let normalized = crate::parser::normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--speculative-strategy",
            "ngram-suffix",
            "--speculative-ngram-proposer",
            "suffix",
            "--speculative-ngram-min",
            "5",
            "--speculative-ngram-max",
            "32",
        ]);
        let cli = Cli::try_parse_from(normalized.normalized).expect("clap parse");
        assert_eq!(cli.speculative_strategy.as_deref(), Some("ngram-suffix"));
        assert_eq!(
            cli.speculative_ngram_proposer,
            Some(SpeculativeNgramProposerCli::Suffix)
        );
        assert_eq!(cli.speculative_ngram_min, Some(5));
        assert_eq!(cli.speculative_ngram_max, Some(32));
    }

    #[test]
    fn auth_status_accepts_owner_key_locally() {
        let cli = Cli::parse_from(["mesh-llm", "auth", "status", "--owner-key", "owner.json"]);

        match cli.command.expect("auth command expected") {
            Command::Auth {
                command: AuthCommand::Status { owner_key, .. },
            } => {
                assert_eq!(owner_key, Some(PathBuf::from("owner.json")));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn auth_status_rejects_runtime_only_owner_required_flag() {
        let err = Cli::try_parse_from(["mesh-llm", "auth", "status", "--owner-required"])
            .expect_err("runtime-only flag should be rejected for auth status");

        let rendered = err.to_string();
        assert!(rendered.contains("--owner-required"));
    }

    #[test]
    fn gpu_and_gpus_spellings_are_synonymous() {
        let cases = [
            (&["gpus"][..], false, None),
            (&["gpu"][..], false, None),
            (&["gpus", "--json"][..], true, None),
            (&["gpu", "--json"][..], true, None),
            (&["gpus", "detect"][..], false, Some(false)),
            (&["gpu", "detect"][..], false, Some(false)),
            (&["gpus", "detect", "--json"][..], false, Some(true)),
            (&["gpu", "detect", "--json"][..], false, Some(true)),
        ];

        for (args, expected_command_json, expected_detect_json) in cases {
            assert_gpu_command_parse(args, expected_command_json, expected_detect_json);
        }
    }

    #[test]
    fn gpu_tune_is_not_a_gpu_subcommand() {
        for spelling in ["gpu", "gpus"] {
            let err = Cli::try_parse_from(["mesh-llm", spelling, "tune"])
                .expect_err("tune should live under benchmark, not gpu/gpus");

            let rendered = err.to_string();
            assert!(rendered.contains("tune"), "unexpected error: {rendered}");
        }
    }

    #[test]
    fn benchmark_tune_parses_model_trial_options() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "benchmark",
            "tune",
            "--model",
            "qwen.gguf",
            "--ctx-sizes",
            "4096,8192",
            "--batch-sizes",
            "1024,2048",
            "--ubatch-sizes",
            "256,512",
            "--mmap-values",
            "auto,true,false",
            "--mlock-values",
            "true,false",
            "--speculative-types",
            "mtp,draft,mtp-ngram,disabled",
            "--spec-draft-models",
            "/models/qwen-draft.gguf",
            "--spec-draft-max-tokens",
            "4,8",
            "--spec-draft-min-tokens",
            "1,2",
            "--spec-ngram-min",
            "2,3",
            "--spec-ngram-max",
            "3,4",
            "--throughput-tolerance-pct",
            "2.5",
            "--max-tokens",
            "64",
            "--startup-timeout-secs",
            "30",
            "--request-timeout-secs",
            "45",
            "--debug-telemetry",
            "--apply",
            "--replace-existing",
            "--launch-args",
            "--prompt",
            "hello",
            "--json",
        ]);

        let Some(Command::Benchmark {
            command: BenchmarkCommand::Tune(tune),
        }) = cli.command
        else {
            panic!("expected benchmark tune command");
        };
        assert_benchmark_tune_core_options(&tune);
        assert_benchmark_tune_speculative_options(&tune);
    }

    fn assert_benchmark_tune_core_options(tune: &crate::benchmark::BenchmarkTuneCommand) {
        assert_eq!(tune.model.as_deref(), Some("qwen.gguf"));
        assert!(tune.models.is_empty());
        assert!(tune.json);
        assert_eq!(tune.ctx_sizes, vec![4096, 8192]);
        assert_eq!(tune.batch_sizes, vec![1024, 2048]);
        assert_eq!(tune.ubatch_sizes, vec![256, 512]);
        assert!(tune.apply);
        assert!(tune.replace_existing);
        assert!(tune.launch_args);
        assert_eq!(
            tune.mmap_values,
            vec![
                crate::benchmark::BenchmarkBoolOrAuto::Auto,
                crate::benchmark::BenchmarkBoolOrAuto::Enabled,
                crate::benchmark::BenchmarkBoolOrAuto::Disabled,
            ]
        );
        assert_eq!(
            tune.mlock_values,
            vec![
                crate::benchmark::BenchmarkBool::Enabled,
                crate::benchmark::BenchmarkBool::Disabled,
            ]
        );
        assert_eq!(tune.throughput_tolerance_pct, 2.5);
        assert_eq!(tune.max_tokens, 64);
        assert_eq!(tune.startup_timeout_secs, 30);
        assert_eq!(tune.request_timeout_secs, 45);
        assert!(tune.debug_telemetry);
        assert_eq!(tune.prompt, "hello");
    }

    fn assert_benchmark_tune_speculative_options(tune: &crate::benchmark::BenchmarkTuneCommand) {
        assert_eq!(
            tune.speculative_types,
            vec![
                crate::benchmark::BenchmarkSpeculativeType::Mtp,
                crate::benchmark::BenchmarkSpeculativeType::Draft,
                crate::benchmark::BenchmarkSpeculativeType::MtpNgram,
                crate::benchmark::BenchmarkSpeculativeType::Disabled,
            ]
        );
        assert!(!tune.no_speculative_tune);
        assert_eq!(
            tune.spec_draft_models,
            vec![std::path::PathBuf::from("/models/qwen-draft.gguf")]
        );
        assert_eq!(tune.spec_draft_max_tokens, vec![4, 8]);
        assert_eq!(tune.spec_draft_min_tokens, vec![1, 2]);
        assert_eq!(tune.spec_ngram_min, vec![2, 3]);
        assert_eq!(tune.spec_ngram_max, vec![3, 4]);
    }

    #[test]
    fn benchmark_tune_rejects_conflicting_model_selectors() {
        let err = Cli::try_parse_from([
            "mesh-llm",
            "benchmark",
            "tune",
            "--model",
            "one.gguf",
            "--models",
            "two.gguf,three.gguf",
        ])
        .expect_err("conflicting benchmark tune model selectors should be rejected");

        let rendered = err.to_string();
        assert!(rendered.contains("--model"));
        assert!(rendered.contains("--models"));
    }

    #[test]
    fn benchmark_tune_no_speculative_tune_conflicts_with_explicit_speculative_types() {
        for (flag, value) in [
            ("--speculative-types", "draft"),
            ("--spec-draft-models", "/models/draft.gguf"),
            ("--spec-draft-max-tokens", "8"),
            ("--spec-draft-min-tokens", "2"),
            ("--spec-draft-acceptance-threshold", "0.8"),
            ("--spec-draft-split-probability", "0.5"),
            ("--spec-ngram-min", "2"),
            ("--spec-ngram-max", "4"),
        ] {
            let err = Cli::try_parse_from([
                "mesh-llm",
                "benchmark",
                "tune",
                "--model",
                "qwen.gguf",
                "--no-speculative-tune",
                flag,
                value,
            ])
            .expect_err("conflicting speculative tune controls should be rejected");

            let rendered = err.to_string();
            assert!(rendered.contains("--no-speculative-tune"));
            assert!(rendered.contains(flag));
        }
    }

    #[test]
    fn benchmark_tune_defaults_to_broad_throughput_tolerance() {
        let cli = Cli::parse_from(["mesh-llm", "benchmark", "tune", "--model", "qwen.gguf"]);

        let Some(Command::Benchmark {
            command: BenchmarkCommand::Tune(tune),
        }) = cli.command
        else {
            panic!("expected benchmark tune command");
        };
        let throughput_tolerance_pct = tune.throughput_tolerance_pct;
        assert!(!tune.apply, "apply should be off by default");
        assert!(
            !tune.replace_existing,
            "replace-existing should be off by default"
        );
        assert!(!tune.launch_args, "launch-args should be off by default");

        assert_eq!(throughput_tolerance_pct, 10.0);
    }

    #[test]
    fn benchmark_tune_replace_existing_requires_apply() {
        let err = Cli::try_parse_from([
            "mesh-llm",
            "benchmark",
            "tune",
            "--model",
            "qwen.gguf",
            "--replace-existing",
        ])
        .expect_err("replace-existing should require apply");

        let rendered = err.to_string();
        assert!(rendered.contains("--apply"), "unexpected error: {rendered}");
    }

    #[test]
    fn hidden_gpu_run_benchmark_parses_backend() {
        let cli = Cli::parse_from(["mesh-llm", "gpus", "run-benchmark", "--backend", "cuda"]);

        let Some(Command::Gpus {
            command: Some(GpuCommand::RunBenchmark { backend }),
            ..
        }) = cli.command
        else {
            panic!("expected hidden gpu run-benchmark command");
        };

        assert_eq!(backend, GpuBenchmarkBackend::Cuda);
    }

    fn assert_gpu_command_parse(
        args: &[&str],
        expected_command_json: bool,
        expected_detect_json: Option<bool>,
    ) {
        let cli = Cli::parse_from(std::iter::once("mesh-llm").chain(args.iter().copied()));

        match cli.command.expect("gpu command expected") {
            Command::Gpus { json, command } => {
                assert_eq!(json, expected_command_json, "command json for {args:?}");
                match (command, expected_detect_json) {
                    (None, None) => {}
                    (Some(GpuCommand::Detect { json }), Some(expected_json)) => {
                        assert_eq!(json, expected_json, "detect json for {args:?}");
                    }
                    (actual, expected) => {
                        panic!(
                            "unexpected detect command for {args:?}: {actual:?}, expected {expected:?}"
                        );
                    }
                }
            }
            other => panic!("unexpected command for {args:?}: {other:?}"),
        }
    }

    #[test]
    fn config_validate_command_parses_config_path_and_json() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "config",
            "validate",
            "--config-path",
            "mesh.toml",
            "--json",
        ]);

        let Some(Command::Config {
            command: ConfigCommand::Validate { config_path, json },
        }) = cli.command
        else {
            panic!("expected config validate command");
        };
        assert_eq!(config_path, Some(PathBuf::from("mesh.toml")));
        assert!(json);
    }

    #[test]
    fn help_text_mentions_headless_keeps_management_api() {
        let help = Cli::command().render_help().to_string();
        assert!(
            help.contains("headless") || help.contains("management API"),
            "help text should mention headless or management API"
        );
    }

    #[test]
    fn opencode_command_accepts_host_flag() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "opencode",
            "--host",
            "https://mesh.example.com:9443",
        ]);

        match cli.command.expect("opencode command expected") {
            Command::Opencode { model, host, write } => {
                assert_eq!(model, None);
                assert_eq!(host, "https://mesh.example.com:9443");
                assert!(!write);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn opencode_command_rejects_port_flag() {
        let err = Cli::try_parse_from(["mesh-llm", "opencode", "--port", "9337"])
            .expect_err("opencode should reject --port");

        let rendered = err.to_string();
        assert!(rendered.contains("--port"));
    }

    #[test]
    fn skills_install_accepts_global_agent_target() {
        let cli = Cli::parse_from(["mesh-llm", "skills", "install", "--agent", "global"]);

        match cli.command.expect("skills command expected") {
            Command::Skills {
                command:
                    SkillCommand::Install {
                        agent, all: false, ..
                    },
            } => {
                assert_eq!(agent, vec![SkillAgentArg::Global]);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn plugins_install_accepts_local_archive_options() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "plugins",
            "install",
            "--archive",
            "/tmp/demo.tar.gz",
            "--name",
            "demo",
            "--version",
            "0.1.0",
        ]);

        match cli.command.expect("plugins command") {
            Command::Plugin {
                command:
                    PluginCommand::Install {
                        reference: None,
                        archive: Some(archive),
                        name: Some(name),
                        version: Some(version),
                    },
            } => {
                assert_eq!(archive, PathBuf::from("/tmp/demo.tar.gz"));
                assert_eq!(name, "demo");
                assert_eq!(version, "0.1.0");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn plugins_install_rejects_reference_with_local_archive() {
        let error = Cli::try_parse_from([
            "mesh-llm",
            "plugins",
            "install",
            "demo",
            "--archive",
            "/tmp/demo.tar.gz",
            "--name",
            "demo",
        ])
        .expect_err("reference and local archive must conflict");

        assert!(error.to_string().contains("cannot be used with"));
    }

    #[test]
    fn cli_rejects_invalid_log_format_values() {
        let err = Cli::try_parse_from(["mesh-llm", "--log-format", "invalid"])
            .expect_err("invalid log format should be rejected");

        assert_eq!(err.kind(), ErrorKind::InvalidValue);
        let rendered = err.to_string();
        assert!(rendered.contains("--log-format <LOG_FORMAT>"));
        assert!(rendered.contains("pretty"));
        assert!(rendered.contains("json"));
    }

    #[test]
    fn cli_help_documents_log_format_flag() {
        let mut command = Cli::command();
        let help = command.render_long_help().to_string();

        assert!(help.contains("--log-format <LOG_FORMAT>"));
        assert!(help.contains("Terminal output format for app-owned runtime events"));
        assert!(help.contains("[default: pretty]"));
        assert!(help.contains("[possible values: pretty, json]"));
    }

    #[test]
    fn cli_log_format_selection_is_independent_across_runs() {
        let pretty = Cli::parse_from(["mesh-llm", "--log-format", "pretty"]);
        assert_eq!(pretty.log_format, LogFormat::Pretty);

        let json = Cli::parse_from(["mesh-llm", "--log-format", "json"]);
        assert_eq!(json.log_format, LogFormat::Json);

        let pretty_again = Cli::parse_from(["mesh-llm", "--log-format", "pretty"]);
        assert_eq!(pretty_again.log_format, LogFormat::Pretty);

        let json_again = Cli::parse_from(["mesh-llm", "--log-format", "json"]);
        assert_eq!(json_again.log_format, LogFormat::Json);
    }

    #[test]
    fn models_search_accepts_canonical_parameter_sort_names() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "models",
            "search",
            "qwen",
            "--sort",
            "parameters-desc",
        ]);

        match cli.command.expect("models command expected") {
            Command::Models {
                command:
                    ModelsCommand::Search {
                        sort: ModelSearchSort::ParametersDesc,
                        ..
                    },
            } => {}
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn models_search_keeps_legacy_parameter_sort_aliases_parsing() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "models",
            "search",
            "qwen",
            "--sort",
            "most-parameters",
        ]);

        match cli.command.expect("models command expected") {
            Command::Models {
                command:
                    ModelsCommand::Search {
                        sort: ModelSearchSort::ParametersDesc,
                        ..
                    },
            } => {}
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn models_certify_parses_package_gate_options() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "models",
            "certify",
            "hf://meshllm/demo-layers@abc123",
            "--package-only",
            "--report-out",
            "/tmp/cert.json",
            "--json",
            "--prompt",
            "Say ok.",
            "--max-tokens",
            "2",
        ]);

        match cli.command.expect("models command expected") {
            Command::Models {
                command:
                    ModelsCommand::Certify {
                        model,
                        package_only: true,
                        json: true,
                        report_out: Some(report_out),
                        prompt,
                        max_tokens: 2,
                        ..
                    },
            } => {
                assert_eq!(model, "hf://meshllm/demo-layers@abc123");
                assert_eq!(report_out, std::path::PathBuf::from("/tmp/cert.json"));
                assert_eq!(prompt, "Say ok.");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn cli_help_documents_serve_and_client_commands() {
        let mut help_bytes = Vec::new();
        Cli::command()
            .write_help(&mut help_bytes)
            .expect("render help");
        let help_text = String::from_utf8(help_bytes).expect("utf8 help");

        assert!(help_text.contains("serve"));
        assert!(help_text.contains("Serve local models and join or publish a mesh"));
        assert!(help_text.contains("client"));
        assert!(help_text.contains("Run as a client-only mesh node with no local model required"));
    }
}
