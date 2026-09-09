use super::*;
use crate::mesh::identity_persistence::load_or_create_key;

pub fn detect_vram_bytes_capped(max_vram_gb: Option<f64>) -> u64 {
    let mut detected = crate::system::hardware::survey().vram_bytes;
    if let Some(cap) = max_vram_gb {
        let cap_bytes = (cap * 1e9) as u64;
        if cap_bytes < detected {
            detected = cap_bytes;
        }
    }
    detected
}

pub(super) async fn startup_secret_key(role: &NodeRole) -> Result<SecretKey> {
    if matches!(role, NodeRole::Client) || std::env::var("MESH_LLM_EPHEMERAL_KEY").is_ok() {
        let key = SecretKey::generate();
        tracing::info!("Using ephemeral key (unique identity)");
        Ok(key)
    } else {
        load_or_create_key().await
    }
}

pub(crate) fn startup_transport_config() -> iroh::endpoint::QuicTransportConfig {
    // We only raise the concurrent bidi-stream ceiling; everything else uses
    // iroh's tuned defaults.
    //
    // History: this function used to override keep-alive (10s) and idle
    // timeouts (300s connection + 300s per path). iroh 1.0 clamps per-path idle
    // to 15s and already sends keep-alive PINGs every 5s, so those overrides do
    // not provide the intended behavior and needlessly diverge from iroh's path
    // management defaults.
    // Mesh multiplexes many concurrent streams (gossip + heartbeat + inference
    // tunnels) over one connection per peer, so we keep a generous bidi ceiling.
    iroh::endpoint::QuicTransportConfig::builder()
        .max_concurrent_bidi_streams(1024u32.into())
        .build()
}

fn relay_mode_for_startup(relay: RelayConfig<'_>) -> Result<iroh::endpoint::RelayMode> {
    let urls = effective_relay_urls(relay.policy, relay.urls);
    if relay.policy.uses_relay() {
        tracing::info!("Relay: {:?}", urls);
        Ok(iroh::endpoint::RelayMode::Custom(relay_map_from_urls(
            &urls,
            relay.auths,
        )?))
    } else {
        let reason = match relay.policy {
            RelayPolicy::ExplicitlyDisabled => "disabled by embedded config",
            RelayPolicy::Disabled => "disabled by LAN-only discovery mode",
            RelayPolicy::DefaultPublic => unreachable!("default public uses relays"),
        };
        tracing::info!("Relay: {reason}");
        Ok(iroh::endpoint::RelayMode::Disabled)
    }
}

pub(super) async fn bind_mesh_endpoint(
    secret_key: SecretKey,
    relay: RelayConfig<'_>,
    quic_bind: QuicBindSelection,
) -> Result<Endpoint> {
    let mut builder = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(secret_key)
        .alpns(vec![
            ALPN_V1.to_vec(),
            skippy_protocol::STAGE_ALPN_V2.to_vec(),
        ])
        .transport_config(startup_transport_config())
        .relay_mode(relay_mode_for_startup(relay)?);

    if let Some(addr) = quic_bind_addr(quic_bind) {
        tracing::info!("Binding QUIC to {addr}");
        if !relay.policy.uses_relay() && addr.is_ipv4() {
            // LAN-only (relay-disabled) mode with a specific IPv4 bind: clear the
            // pre-configured default sockets first. `bind_addr` only replaces the
            // default for the *same* address family, so binding a specific IPv4
            // would otherwise leave the default IPv6 `[::]` socket in place. That
            // extra local IPv6 path becomes a second candidate, and with no relay
            // iroh's multipath negotiation across the IPv4+IPv6 locals fails with
            // `MultipathNotNegotiated`, stalling the connection with no fallback.
            // Pinning a single IPv4 socket keeps one local path family so the LAN
            // direct path establishes cleanly. In relay (public) mode we keep the
            // defaults so relay/IPv6 reachability is unaffected.
            builder = builder.clear_ip_transports();
        }
        builder = builder.bind_addr(addr)?;
    }

    builder.bind().await.map_err(Into::into)
}

pub(super) async fn wait_for_endpoint_online(
    endpoint: &Endpoint,
    connected_log: &str,
    timeout_log: &str,
) {
    match tokio::time::timeout(std::time::Duration::from_secs(5), endpoint.online()).await {
        Ok(()) => tracing::info!("{connected_log}"),
        Err(_) => tracing::warn!("{timeout_log}"),
    }
}

/// A zero-capacity participant has no reason to initialize accelerator discovery.
/// Keep hostname identity, but do not probe native runtimes or publish GPU inventory.
pub(super) fn hardware_survey_for_start(
    max_vram_gb: Option<f64>,
    enumerate_host: bool,
) -> crate::system::hardware::HardwareSurvey {
    if max_vram_gb == Some(0.0) && !enumerate_host {
        crate::system::hardware::query(&[crate::system::hardware::Metric::Hostname])
    } else {
        crate::system::hardware::survey()
    }
}

pub(crate) fn hardware_snapshot_for_start(
    hw: crate::system::hardware::HardwareSurvey,
    role: &NodeRole,
    max_vram_gb: Option<f64>,
) -> NodeHardwareSnapshot {
    let local_runtime_capacity_bytes =
        super::super::capacity::capped_capacity_bytes(hw.vram_bytes, max_vram_gb);
    let mut vram_bytes = super::super::capacity::advertised_capacity_bytes(&hw, max_vram_gb);
    let gpu_name = if matches!(role, NodeRole::Client) {
        None
    } else {
        hw.gpu_name
    };
    let hostname = hw.hostname;
    let is_soc = Some(hw.is_soc);
    let gpu_vram = (!hw.gpu_vram.is_empty()).then(|| {
        hw.gpu_vram
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(",")
    });
    let gpu_reserved_bytes = if hw.gpu_reserved.iter().all(Option::is_none) {
        None
    } else {
        Some(
            hw.gpu_reserved
                .iter()
                .map(|value| value.map(|v| v.to_string()).unwrap_or_default())
                .collect::<Vec<_>>()
                .join(","),
        )
    };

    log_detected_vram(&mut vram_bytes, max_vram_gb);

    NodeHardwareSnapshot {
        vram_bytes,
        local_runtime_capacity_bytes,
        gpu_name,
        hostname,
        is_soc,
        gpu_vram,
        gpu_reserved_bytes,
    }
}

fn detected_vram_log(vram_bytes: u64, max_vram_gb: Option<f64>) -> DetectedVramLog {
    let detected_gb = vram_bytes as f64 / 1e9;
    let capped_bytes = max_vram_gb
        .map(|max_gb| ((max_gb * 1e9) as u64, max_gb))
        .and_then(|(max_bytes, _)| (max_bytes < vram_bytes).then_some(max_bytes));
    DetectedVramLog {
        detected_gb,
        max_gb: max_vram_gb,
        capped_bytes,
    }
}

fn log_detected_vram(vram_bytes: &mut u64, max_vram_gb: Option<f64>) {
    let log = detected_vram_log(*vram_bytes, max_vram_gb);
    if let Some(max_gb) = log.max_gb {
        log_detected_vram_with_cap(vram_bytes, log.detected_gb, max_gb, log.capped_bytes);
    } else {
        tracing::info!("Detected VRAM: {:.1} GB", log.detected_gb);
    }
}

fn log_detected_vram_with_cap(
    vram_bytes: &mut u64,
    detected_gb: f64,
    max_gb: f64,
    capped_bytes: Option<u64>,
) {
    if let Some(capped_bytes) = capped_bytes {
        tracing::info!(
            "Detected VRAM: {:.1} GB, capped to {:.1} GB (--max-vram)",
            detected_gb,
            max_gb
        );
        *vram_bytes = capped_bytes;
    } else {
        tracing::info!(
            "Detected VRAM: {:.1} GB (--max-vram {:.1} has no effect)",
            detected_gb,
            max_gb
        );
    }
}

pub(super) fn init_owner_runtime(
    owner_config: Option<&OwnerRuntimeConfig>,
    endpoint_id: EndpointId,
    hostname: Option<String>,
) -> Result<OwnerRuntimeInit> {
    let trust_store = owner_config
        .map(|config| config.trust_store.clone())
        .unwrap_or_default();
    let trust_policy = owner_config
        .map(|config| config.trust_policy)
        .unwrap_or_default();
    let owner_attestation = match owner_config.and_then(|config| config.keypair.as_ref()) {
        Some(keypair) => Some(load_or_refresh_owner_attestation(
            keypair,
            endpoint_id,
            owner_config.and_then(|config| config.node_label.clone()),
            hostname,
        )?),
        None => None,
    };

    Ok(OwnerRuntimeInit {
        trust_store,
        trust_policy,
        owner_attestation,
    })
}

pub(crate) fn default_plugin_event_source(endpoint_id: EndpointId, source_peer_id: &mut String) {
    if source_peer_id.is_empty() {
        *source_peer_id = endpoint_id_hex(endpoint_id);
    }
}

#[cfg(test)]
mod zero_capacity_tests {
    use super::*;

    #[test]
    fn zero_capacity_start_has_no_accelerator_inventory() {
        let hw = hardware_survey_for_start(Some(0.0), false);
        assert_eq!(hw.vram_bytes, 0);
        assert!(hw.gpu_name.is_none());
        assert!(hw.gpus.is_empty());
        let snapshot = hardware_snapshot_for_start(hw, &NodeRole::Worker, Some(0.0));
        assert_eq!(snapshot.vram_bytes, 0);
        assert_eq!(snapshot.local_runtime_capacity_bytes, 0);
        assert!(snapshot.gpu_name.is_none());
        assert!(snapshot.gpu_vram.is_none());
        assert!(snapshot.gpu_reserved_bytes.is_none());
    }
}
