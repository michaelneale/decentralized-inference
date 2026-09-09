use super::*;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

mod inbound;
mod tunnel;

pub(crate) struct NodeHardwareSnapshot {
    /// Accelerator-resident capacity advertised for mesh stage placement.
    pub(crate) vram_bytes: u64,
    /// Broader local fit budget, which may include CPU offload memory.
    pub(crate) local_runtime_capacity_bytes: u64,
    pub(crate) gpu_name: Option<String>,
    pub(crate) hostname: Option<String>,
    pub(crate) is_soc: Option<bool>,
    pub(crate) gpu_vram: Option<String>,
    pub(crate) gpu_reserved_bytes: Option<String>,
    /// Itemized view of `vram_bytes`, announced alongside the GPU inventory.
    pub(crate) memory: AdvertisedMemory,
}

pub(crate) struct OwnerRuntimeInit {
    pub(crate) trust_store: TrustStore,
    pub(crate) trust_policy: TrustPolicy,
    pub(crate) owner_attestation: Option<SignedNodeOwnership>,
}

pub(crate) struct DetectedVramLog {
    pub(crate) detected_gb: f64,
    pub(crate) max_gb: Option<f64>,
    pub(crate) capped_bytes: Option<u64>,
}

pub(crate) struct AcceptedMeshStream {
    pub(crate) remote: EndpointId,
    pub(crate) send: iroh::endpoint::SendStream,
    pub(crate) recv: iroh::endpoint::RecvStream,
    pub(crate) stream_type: u8,
}

pub(crate) const MAX_CONTROL_STREAM_WORK_PER_CONNECTION: usize = 32;
const MESH_STREAM_TYPE_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Grace period for a replaced peer connection to finish in-flight streams
/// before it is explicitly closed.
const REPLACED_CONNECTION_DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

pub(crate) fn control_stream_semaphore() -> Arc<tokio::sync::Semaphore> {
    Arc::new(tokio::sync::Semaphore::new(
        MAX_CONTROL_STREAM_WORK_PER_CONNECTION,
    ))
}

pub(crate) enum ClosedConnectionRecovery {
    Reconnect(EndpointAddr),
    RemovePeer,
    AlreadyReplaced,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QuicBindSelection {
    pub ip: Option<IpAddr>,
    pub port: Option<u16>,
}

/// Relay map plus per-relay bearer tokens for gated iroh-relays.
///
/// `urls` is the relay map; `auths` is a sparse map of relay URL -> bearer
/// token used when registering with relays running `AccessConfig::Restricted`.
/// Public relays in the same map continue to register without auth.
#[derive(Clone, Copy, Debug)]
pub struct RelayConfig<'a> {
    pub urls: &'a [String],
    pub auths: &'a std::collections::HashMap<String, String>,
    pub policy: RelayPolicy,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RelayPolicy {
    #[default]
    DefaultPublic,
    ExplicitlyDisabled,
    Disabled,
}

impl RelayPolicy {
    pub(crate) fn uses_relay(self) -> bool {
        matches!(self, Self::DefaultPublic)
    }

    pub(crate) fn uses_raw_stun(self) -> bool {
        matches!(self, Self::DefaultPublic | Self::ExplicitlyDisabled)
    }
}

pub(crate) fn quic_bind_addr(bind: QuicBindSelection) -> Option<SocketAddr> {
    if let Some(ip) = bind.ip {
        return Some(SocketAddr::new(
            ip,
            bind.port.unwrap_or(EPHEMERAL_QUIC_PORT),
        ));
    }

    if let Some(port) = bind.port {
        return Some(SocketAddr::from(([0, 0, 0, 0], port)));
    }

    #[cfg(target_os = "windows")]
    {
        Some(std::net::SocketAddr::from((
            [127, 0, 0, 1],
            EPHEMERAL_QUIC_PORT,
        )))
    }

    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub(crate) fn default_control_bind_addr() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([127, 0, 0, 1], 0))
}

pub(crate) fn is_public_ipv4_candidate(socket: &SocketAddr) -> bool {
    match socket.ip() {
        IpAddr::V4(ip) => is_global_ipv4_candidate(ip),
        IpAddr::V6(_) => false,
    }
}

pub(crate) fn is_global_ipv4_candidate(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_multicast()
        || ip.is_broadcast()
        || ip.is_unspecified()
        || (a == 100 && (64..=127).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 240)
}

pub(crate) fn endpoint_addr_has_public_ipv4(addr: &EndpointAddr) -> bool {
    addr.addrs.iter().any(|candidate| match candidate {
        TransportAddr::Ip(socket) => is_public_ipv4_candidate(socket),
        _ => false,
    })
}

// Host-network Docker and CNI bridges commonly reuse the same 172.* addresses
// on every host. When a node selects a bind IP, only advertise that direct IP.
// Public discovery keeps public candidates for non-LAN reachability; LAN-only
// discovery strips them so peers try the selected lab interface directly.
pub(crate) fn filter_endpoint_addr_for_bind_ip(
    mut addr: EndpointAddr,
    bind_ip: Option<IpAddr>,
    preserve_public_ipv4_candidates: bool,
) -> EndpointAddr {
    let Some(bind_ip) = bind_ip else {
        return addr;
    };
    addr.addrs.retain(|candidate| match candidate {
        TransportAddr::Ip(socket) => {
            socket.ip() == bind_ip
                || (preserve_public_ipv4_candidates && is_public_ipv4_candidate(socket))
        }
        _ => true,
    });
    addr
}

pub(crate) fn effective_relay_urls(policy: RelayPolicy, relay_urls: &[String]) -> Vec<String> {
    match policy {
        RelayPolicy::Disabled | RelayPolicy::ExplicitlyDisabled => Vec::new(),
        RelayPolicy::DefaultPublic if relay_urls.is_empty() => vec![
            "https://usw1-2.relay.michaelneale.mesh-llm.iroh.link./".into(),
            "https://aps1-1.relay.michaelneale.mesh-llm.iroh.link./".into(),
            "https://euc1-1.relay.michaelneale.mesh-llm.iroh.link./".into(),
            "https://use1-1.relay.michaelneale.mesh-llm.iroh.link./".into(),
        ],
        RelayPolicy::DefaultPublic => relay_urls.to_vec(),
    }
}

#[cfg(test)]
mod relay_policy_tests {
    use super::{RelayPolicy, effective_relay_urls};

    #[test]
    pub(crate) fn default_policy_uses_managed_relays_when_no_urls_are_given() {
        let urls = effective_relay_urls(RelayPolicy::DefaultPublic, &[]);

        assert_eq!(
            urls,
            vec![
                "https://usw1-2.relay.michaelneale.mesh-llm.iroh.link./",
                "https://aps1-1.relay.michaelneale.mesh-llm.iroh.link./",
                "https://euc1-1.relay.michaelneale.mesh-llm.iroh.link./",
                "https://use1-1.relay.michaelneale.mesh-llm.iroh.link./",
            ]
        );
    }

    #[test]
    pub(crate) fn default_policy_uses_custom_relay_urls_when_supplied() {
        let custom = vec!["https://relay.example/".to_string()];

        assert_eq!(
            effective_relay_urls(RelayPolicy::DefaultPublic, &custom),
            custom
        );
    }

    #[test]
    pub(crate) fn disabled_policy_uses_no_relays_but_explicit_disable_keeps_raw_stun() {
        let custom = vec!["https://relay.example/".to_string()];

        assert!(effective_relay_urls(RelayPolicy::Disabled, &custom).is_empty());
        assert!(effective_relay_urls(RelayPolicy::ExplicitlyDisabled, &custom).is_empty());
        assert!(!RelayPolicy::Disabled.uses_relay());
        assert!(!RelayPolicy::ExplicitlyDisabled.uses_relay());
        assert!(!RelayPolicy::Disabled.uses_raw_stun());
        assert!(RelayPolicy::ExplicitlyDisabled.uses_raw_stun());
    }
}

/// Build an [`iroh::RelayMap`] from URLs, attaching per-relay auth tokens
/// where configured.
///
/// `auths` maps relay URLs (as they appear in `urls`) to bearer tokens. Tokens
/// are passed to `iroh::RelayConfig::with_auth_token` which sends them as
/// `Authorization: Bearer <token>` on the WebSocket upgrade. Relays not present
/// in the map register unauthenticated, which is the correct behavior for
/// public (`AccessConfig::Everyone`) relays.
///
/// This is the wire-up that lets a gated iroh-relay (e.g. one running
/// `AccessConfig::Restricted` with NIP-98 admission) admit this node while
/// public relays in the same map continue to work normally.
pub(crate) fn relay_map_from_urls(
    urls: &[String],
    auths: &std::collections::HashMap<String, String>,
) -> Result<iroh::RelayMap> {
    let configs = urls.iter().map(|url| {
        let parsed: iroh::RelayUrl = url
            .parse()
            .with_context(|| format!("invalid relay URL `{url}`"))?;
        // Preserve iroh's default QUIC Address Discovery (QAD, UDP 7842).
        // `iroh::RelayConfig::new(parsed, None)` explicitly DISABLES QAD, so
        // net-report never learns a reflexive public UDP candidate and direct
        // paths across NAT cannot form (nodes fall back to relay). Constructing
        // via `RelayUrl::into()` keeps QAD enabled — iroh's default. QAD is
        // client-driven and gated behind `has_ip_transports`, so relay-only and
        // LAN-only endpoints are unaffected. See issue #1065.
        let cfg: iroh::RelayConfig = parsed.into();
        Ok(match auths.get(url) {
            Some(token) => cfg.with_auth_token(token.clone()),
            None => cfg,
        })
    });
    configs
        .collect::<Result<Vec<_>>>()
        .map(iroh::RelayMap::from_iter)
}

#[cfg(test)]
mod relay_map_tests {
    use super::relay_map_from_urls;
    use std::collections::HashMap;
    use std::sync::Arc;

    pub(crate) fn configs(map: &iroh::RelayMap) -> Vec<Arc<iroh::RelayConfig>> {
        map.relays::<Vec<_>>()
    }

    #[test]
    pub(crate) fn builds_map_with_default_qad_without_auth() {
        let urls = vec![
            "https://r1.example/".to_string(),
            "https://r2.example/".to_string(),
        ];
        let map = relay_map_from_urls(&urls, &HashMap::new()).expect("relay map should build");
        let cfgs = configs(&map);
        assert_eq!(cfgs.len(), 2);
        assert!(cfgs.iter().all(|config| config.auth_token.is_none()));
        // QAD must be enabled (issue #1065): the relay config must carry a QUIC
        // address-discovery config on the default port, else nodes never learn a
        // reflexive candidate and direct-path upgrades across NAT cannot form.
        assert!(
            cfgs.iter()
                .all(|config| { config.quic.as_ref().is_some_and(|quic| quic.port == 7842) })
        );
        // QAD must be enabled (issue #1065): the relay config must carry a QUIC
        // address-discovery config on the default port, else nodes never learn a
        // reflexive candidate and direct-path upgrades across NAT cannot form.
        let quic = cfgs[0]
            .quic
            .as_ref()
            .expect("default QUIC address discovery (QAD) should be enabled");
        assert_eq!(
            quic.port, 7842,
            "QAD should use iroh's default relay QUIC port"
        );
    }

    #[test]
    pub(crate) fn attaches_auth_token_for_matching_url() {
        let urls = vec!["https://gated.example/".to_string()];
        let mut auths = HashMap::new();
        auths.insert("https://gated.example/".to_string(), "nip98-bearer".into());
        let map = relay_map_from_urls(&urls, &auths).expect("relay map should build");
        let cfgs = configs(&map);
        assert_eq!(cfgs.len(), 1);
        assert_eq!(cfgs[0].auth_token.as_deref(), Some("nip98-bearer"));
    }

    #[test]
    pub(crate) fn leaves_other_relays_unauthenticated_in_mixed_map() {
        // The whole point: gated relay gets a token, public relays don't.
        let urls = vec![
            "https://gated.example/".to_string(),
            "https://public.iroh/".to_string(),
        ];
        let mut auths = HashMap::new();
        auths.insert("https://gated.example/".to_string(), "bearer-xyz".into());

        let map = relay_map_from_urls(&urls, &auths).expect("relay map should build");
        let by_url: HashMap<String, Option<String>> = configs(&map)
            .into_iter()
            .map(|cfg| (cfg.url.to_string(), cfg.auth_token.clone()))
            .collect();

        // Find the entries by matching on host substring, since iroh-relay may
        // canonicalise the URL form (e.g. trailing dot on the host).
        let gated = by_url
            .iter()
            .find(|(u, _)| u.contains("gated.example"))
            .expect("gated relay should be in the map");
        let public = by_url
            .iter()
            .find(|(u, _)| u.contains("public.iroh"))
            .expect("public relay should be in the map");

        assert_eq!(
            gated.1.as_deref(),
            Some("bearer-xyz"),
            "gated relay must carry its token"
        );
        assert!(
            public.1.is_none(),
            "public relay must register without a token, got {:?}",
            public.1
        );
    }

    #[test]
    pub(crate) fn invalid_relay_url_returns_contextual_error() {
        let urls = vec!["not a relay url".to_string()];

        let error = relay_map_from_urls(&urls, &HashMap::new()).expect_err("invalid URL must fail");

        assert!(
            format!("{error:#}").contains("invalid relay URL `not a relay url`"),
            "error should identify the invalid relay URL, got {error:#}"
        );
    }
}

/// End-to-end regression tests for `--relay-auth` against a real in-process
/// iroh-relay running a custom [`iroh_relay::server::AccessControl`].
///
/// These tests do not go through the full `Node::start` path — they exercise
/// `relay_map_from_urls` (the new wiring) plus the iroh `Endpoint` builder
/// the same way `bind_mesh_endpoint` does, with `ca_tls_config` overridden
/// for the relay's self-signed test cert. The contract being defended is:
///
///  1. A token configured for a gated relay URL reaches iroh as
///     `RelayConfig::with_auth_token`, gets sent as `Authorization: Bearer`
///     on the WebSocket upgrade, and the relay admits the endpoint.
///  2. The wrong token (or no token) is rejected with `not authorized` and
///     the endpoint never reaches `online()`.
///  3. Mixed maps work: a gated relay with the right token coexists with a
///     public relay (no token) in the same `RelayMap`.
#[cfg(test)]
mod gated_relay_e2e_tests {
    use super::relay_map_from_urls;
    use futures_util::StreamExt;
    use iroh::SecretKey;
    use iroh::Watcher;
    use iroh::endpoint::{Endpoint, RelayMode, presets};
    use iroh::test_utils::run_relay_server_with_access;
    use iroh_relay::server::{Access, AccessControl, AllowAll, ClientRequest};
    use iroh_relay::tls::CaTlsConfig;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    #[derive(Debug)]
    struct TokenAccess(&'static str);

    impl AccessControl for TokenAccess {
        async fn on_connect(&self, request: &ClientRequest) -> Access {
            if request.auth_token().as_deref() == Some(self.0) {
                Access::Allow
            } else {
                Access::Deny { reason: None }
            }
        }
    }

    /// Spawn an in-process iroh-relay that only admits `expected_token`.
    /// Returns (relay_url_string, drop-guard server).
    pub(crate) async fn spawn_gated_relay(
        expected_token: &'static str,
    ) -> (String, iroh_relay::server::Server) {
        let access = Arc::new(TokenAccess(expected_token));
        let (_relay_map, relay_url, server) = run_relay_server_with_access(false, access)
            .await
            .expect("spawn gated relay");
        (relay_url.to_string(), server)
    }

    /// Build an `Endpoint` configured the same way `bind_mesh_endpoint` does,
    /// but using `relay_map_from_urls` for the relay map and accepting the
    /// relay's self-signed test cert via `insecure_skip_verify`.
    pub(crate) async fn build_endpoint(
        relay_urls: &[String],
        relay_auths: &HashMap<String, String>,
    ) -> Endpoint {
        let relay_map =
            relay_map_from_urls(relay_urls, relay_auths).expect("relay map should build");
        assert!(
            relay_map
                .relays::<Vec<_>>()
                .iter()
                .all(|config| { config.quic.as_ref().is_some_and(|quic| quic.port == 7842) }),
            "end-to-end relay configs should keep default QAD enabled"
        );
        Endpoint::builder(presets::Minimal)
            .secret_key(SecretKey::generate())
            .relay_mode(RelayMode::Custom(relay_map))
            .ca_tls_config(CaTlsConfig::insecure_skip_verify())
            .bind()
            .await
            .expect("endpoint bind")
    }

    #[tokio::test]
    pub(crate) async fn qad_enabled_map_admits_endpoint_to_gated_relay() {
        const TOKEN: &str = "secret-token";
        let (relay_url, _server) = spawn_gated_relay(TOKEN).await;

        let urls = vec![relay_url.clone()];
        let mut auths = HashMap::new();
        auths.insert(relay_url, TOKEN.to_string());

        let ep = build_endpoint(&urls, &auths).await;
        tokio::time::timeout(Duration::from_secs(5), ep.online())
            .await
            .expect("endpoint with matching token should come online");
    }

    #[tokio::test]
    pub(crate) async fn wrong_token_is_rejected_by_gated_relay() {
        const TOKEN: &str = "secret-token";
        let (relay_url, _server) = spawn_gated_relay(TOKEN).await;

        let urls = vec![relay_url.clone()];
        let mut auths = HashMap::new();
        auths.insert(relay_url, "wrong-token".to_string());

        let ep = build_endpoint(&urls, &auths).await;

        // Observe the relay-side denial via home_relay_status before falling
        // back to the timeout. We must see `not authorized` to prove the
        // token actually reached the relay (rather than e.g. silently being
        // dropped before the WebSocket upgrade).
        let mut stream = ep.home_relay_status().stream();
        let auth_err = tokio::time::timeout(Duration::from_secs(5), async {
            while let Some(status) = stream.next().await {
                if let Some(err) = status.iter().filter_map(|s| s.last_error()).next() {
                    return Some(format!("{err:#}"));
                }
            }
            None
        })
        .await
        .expect("home relay status should report an error within 5s")
        .expect("home relay status should yield an error");
        assert!(
            auth_err.contains("not authorized"),
            "expected 'not authorized' in error, got: {auth_err}"
        );

        // And the endpoint must NOT come online.
        let online = tokio::time::timeout(Duration::from_millis(500), ep.online()).await;
        assert!(
            online.is_err(),
            "endpoint with wrong token must not reach online() within 500ms"
        );
    }

    #[tokio::test]
    pub(crate) async fn missing_token_for_gated_relay_is_rejected() {
        const TOKEN: &str = "secret-token";
        let (relay_url, _server) = spawn_gated_relay(TOKEN).await;

        // No auth in the map at all → relay must deny.
        let urls = vec![relay_url];
        let auths = HashMap::new();
        let ep = build_endpoint(&urls, &auths).await;

        let online = tokio::time::timeout(Duration::from_millis(500), ep.online()).await;
        assert!(
            online.is_err(),
            "endpoint without a token must not be admitted by a gated relay"
        );
    }

    #[tokio::test]
    pub(crate) async fn mixed_map_authenticates_only_the_gated_relay() {
        const TOKEN: &str = "secret-token";
        let (gated_url, _gated) = spawn_gated_relay(TOKEN).await;

        // Spin up a second, fully-open relay to stand in for a public iroh
        // relay sharing the same map.
        let (_public_map, public_url, _public) =
            run_relay_server_with_access(false, Arc::new(AllowAll))
                .await
                .expect("spawn public relay");
        let public_url = public_url.to_string();

        let urls = vec![gated_url.clone(), public_url.clone()];
        let mut auths = HashMap::new();
        auths.insert(gated_url, TOKEN.to_string());
        // Public relay intentionally absent from `auths`.

        let ep = build_endpoint(&urls, &auths).await;
        tokio::time::timeout(Duration::from_secs(5), ep.online())
            .await
            .expect("endpoint should come online via the mixed relay map");
    }
}

pub(crate) fn encode_endpoint_addr_token(addr: &EndpointAddr) -> String {
    let json = serde_json::to_vec(addr).expect("endpoint addr should serialize");
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)
}

#[derive(Clone, Debug)]
pub(crate) enum InviteTokenMaterial {
    Legacy(EndpointAddr),
    Signed(Box<crate::SignedBootstrapToken>),
}

#[derive(Clone, Debug)]
pub(crate) struct ActiveMeshPolicyState {
    pub(crate) mesh_id: String,
    pub(crate) policy_hash: String,
    pub(crate) policy: crate::MeshGenesisPolicy,
}

pub(crate) fn decode_invite_token_payload(invite_token: &str) -> Result<Vec<u8>> {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(invite_token)
        .context("invalid invite token encoding")
}

pub(crate) fn parse_invite_token(
    invite_token: &str,
) -> std::result::Result<InviteTokenMaterial, MeshRequirementRejectReason> {
    let payload = decode_invite_token_payload(invite_token)
        .map_err(|_| MeshRequirementRejectReason::BootstrapTokenInvalid)?;
    if let Ok(addr) = serde_json::from_slice::<EndpointAddr>(&payload) {
        return Ok(InviteTokenMaterial::Legacy(addr));
    }
    let token = serde_json::from_slice::<crate::SignedBootstrapToken>(&payload)
        .map_err(|_| MeshRequirementRejectReason::BootstrapTokenInvalid)?;
    Ok(InviteTokenMaterial::Signed(Box::new(token)))
}

pub(crate) fn decode_signed_bootstrap_addrs(
    token: &crate::SignedBootstrapToken,
) -> Result<Vec<EndpointAddr>> {
    anyhow::ensure!(
        !token.serialized_addrs.is_empty(),
        "bootstrap token does not contain any endpoint addresses"
    );
    token
        .serialized_addrs
        .iter()
        .map(|bytes| {
            serde_json::from_slice(bytes)
                .context("bootstrap token contains an invalid serialized endpoint address")
        })
        .collect()
}

pub(crate) fn control_endpoint_addr(
    endpoint: &Endpoint,
    advertise_addr: Option<std::net::SocketAddr>,
) -> EndpointAddr {
    let mut addr = endpoint.addr();
    if let Some(advertise_addr) = advertise_addr {
        addr.addrs
            .retain(|addr| matches!(addr, TransportAddr::Relay(_)));
        addr.addrs.insert(TransportAddr::Ip(advertise_addr));
    }
    addr
}

impl Node {
    /// Open an HTTP tunnel bi-stream to a peer (tagged STREAM_TUNNEL_HTTP).
    /// If no connection exists, tries to connect on-demand (for passive nodes
    /// that learned about hosts from routing table but aren't directly connected).
    pub async fn open_http_tunnel(
        &self,
        peer_id: EndpointId,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        let conn = self.connection_to_peer(peer_id).await?;
        let result = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let (mut send, recv) = conn.open_bi().await?;
            send.write_all(&[STREAM_TUNNEL_HTTP]).await?;
            Ok::<_, anyhow::Error>((send, recv))
        })
        .await
        .map_err(|_| anyhow::anyhow!("Timeout opening tunnel to {}", peer_id.fmt_short()))?;

        if result.is_err() {
            // Connection failed — peer is likely dead, broadcast it
            tracing::info!(
                "Tunnel to {} failed, broadcasting death",
                peer_id.fmt_short()
            );
            self.handle_peer_death_with_reason(peer_id, MeshPeerRemovalReason::TunnelOpenFailed)
                .await;
        }

        result
    }

    // --- Connection handling ---

    pub(crate) async fn accept_loop(&self) {
        // Wait until start_accepting() is called before processing any connections.
        // Check flag first to handle the case where start_accepting() was called before we got here.
        if !self.accepting.1.load(std::sync::atomic::Ordering::Acquire) {
            self.accepting.0.notified().await;
        }
        tracing::info!("Accept loop: now accepting inbound connections");

        loop {
            let incoming = match self.endpoint.accept().await {
                Some(i) => i,
                None => break,
            };
            let node = self.clone();
            tokio::spawn(async move {
                if let Err(e) = node.handle_incoming(incoming).await {
                    tracing::warn!("Incoming connection error: {e}");
                }
            });
        }
    }

    pub(crate) async fn control_accept_loop(
        &self,
        endpoint: Endpoint,
        shutdown_requested: Arc<std::sync::atomic::AtomicBool>,
        shutdown: Arc<tokio::sync::Notify>,
    ) {
        loop {
            if shutdown_requested.load(std::sync::atomic::Ordering::Acquire) {
                break;
            }
            tokio::select! {
                _ = shutdown.notified() => break,
                incoming = endpoint.accept() => {
                    let Some(incoming) = incoming else {
                        break;
                    };
                    let node = self.clone();
                    tokio::spawn(Box::pin(async move {
                        if let Err(error) = node.handle_control_incoming(incoming).await {
                            tracing::debug!("Control-plane incoming connection error: {error}");
                        }
                    }));
                }
            }
        }
    }

    pub(crate) async fn remember_incoming_connection(
        &self,
        remote: EndpointId,
        conn: &Connection,
    ) -> (bool, bool) {
        let mut state = self.state.lock().await;
        let was_dead = state.dead_peers.remove(&remote).is_some();
        let admitted = state.peers.contains_key(&remote);
        if was_dead {
            emit_mesh_info(format!(
                "🔄 Previously dead peer {} reconnected",
                remote.fmt_short()
            ));
        }
        let replaced = state.connections.insert(remote, conn.clone());
        drop(state);
        if let Some(replaced) = replaced
            && replaced.stable_id() != conn.stable_id()
        {
            Self::spawn_replaced_connection_drain(remote, replaced);
        }
        (was_dead, admitted)
    }

    /// Retire a connection that a newer one just replaced.
    ///
    /// Existing streams get a bounded grace period to finish, then the
    /// connection is explicitly closed. Without the terminal close a replaced
    /// connection is preserved indefinitely; without the grace period in-flight
    /// requests are cut off mid-stream.
    pub(super) fn spawn_replaced_connection_drain(remote: EndpointId, replaced: Connection) {
        tokio::spawn(async move {
            tracing::debug!(
                peer = %remote.fmt_short(),
                drain_ms = REPLACED_CONNECTION_DRAIN_GRACE.as_millis(),
                "draining replaced peer connection"
            );
            tokio::select! {
                _ = replaced.closed() => {}
                _ = tokio::time::sleep(REPLACED_CONNECTION_DRAIN_GRACE) => {
                    replaced.close(0u32.into(), b"connection-replaced");
                }
            }
        });
    }

    pub(crate) fn spawn_reconnect_gossip(&self, conn: Connection, remote: EndpointId) {
        let node = self.clone();
        tokio::spawn(async move {
            if let Err(e) = node.initiate_gossip_inner(conn, remote, false).await {
                tracing::debug!("Reconnect gossip with {} failed: {e}", remote.fmt_short());
            }
        });
    }
}
impl Node {
    pub(crate) async fn recover_closed_connection(
        &self,
        remote: EndpointId,
        closing_stable_id: usize,
    ) {
        let recovery = self
            .remove_closed_connection(remote, closing_stable_id)
            .await;
        self.release_direct_rescue_endpoint(remote, closing_stable_id)
            .await;
        match recovery {
            ClosedConnectionRecovery::Reconnect(addr) => {
                self.reconnect_closed_connection_or_remove(remote, addr)
                    .await;
            }
            ClosedConnectionRecovery::RemovePeer => {
                self.remove_peer(remote, MeshPeerRemovalReason::ClosedConnectionNoAddress)
                    .await;
            }
            ClosedConnectionRecovery::AlreadyReplaced => {}
        }
    }

    pub(crate) async fn reconnect_closed_connection_or_remove(
        &self,
        remote: EndpointId,
        addr: EndpointAddr,
    ) {
        tracing::info!("Attempting reconnect to {}...", remote.fmt_short());
        match self.reconnect_closed_peer(remote, addr).await {
            Some(new_conn) => {
                self.complete_recovered_connection(remote, new_conn).await;
            }
            _ => {
                tracing::info!("Reconnect to {} failed — removing peer", remote.fmt_short());
                self.remove_peer(remote, MeshPeerRemovalReason::ReconnectFailed)
                    .await;
            }
        }
    }

    pub(crate) async fn remove_closed_connection(
        &self,
        remote: EndpointId,
        closing_stable_id: usize,
    ) -> ClosedConnectionRecovery {
        let mut state = self.state.lock().await;
        if !heartbeat::should_remove_connection(
            state.connections.get(&remote).map(|conn| conn.stable_id()),
            closing_stable_id,
        ) {
            tracing::debug!(
                "Connection dispatcher for {} closed after the tracked connection was replaced",
                remote.fmt_short()
            );
            return ClosedConnectionRecovery::AlreadyReplaced;
        }
        state.connections.remove(&remote);
        match state.peers.get(&remote).map(|peer| peer.addr.clone()) {
            Some(addr) => ClosedConnectionRecovery::Reconnect(addr),
            None => ClosedConnectionRecovery::RemovePeer,
        }
    }

    pub(crate) async fn reconnect_closed_peer(
        &self,
        remote: EndpointId,
        addr: EndpointAddr,
    ) -> Option<Connection> {
        match tokio::time::timeout(
            std::time::Duration::from_secs(10),
            connect_mesh(&self.endpoint, addr),
        )
        .await
        {
            Ok(Ok(new_conn)) => {
                tracing::info!("Reconnected to {}", remote.fmt_short());
                Some(new_conn)
            }
            _ => None,
        }
    }

    pub(crate) async fn complete_recovered_connection(
        &self,
        remote: EndpointId,
        new_conn: Connection,
    ) {
        let owner = match self.reserve_pending_connection(remote).await {
            PendingConnectionReservation::Owner(owner) => owner,
            PendingConnectionReservation::Waiter(waiter) => {
                new_conn.close(0u32.into(), b"recovered-connection-raced");
                self.complete_recovered_connection_waiter(waiter).await;
                return;
            }
        };
        if self
            .recovered_connection_gossip_ok(remote, new_conn.clone())
            .await
        {
            {
                let mut state = self.state.lock().await;
                state.connections.insert(remote, new_conn.clone());
            }
            let node = self.clone();
            tokio::spawn(async move {
                node.dispatch_streams(new_conn, remote).await;
            });
            self.finish_pending_connection(owner, PendingConnectionOutcome::Admitted)
                .await;
        } else {
            new_conn.close(0u32.into(), b"recovered-gossip-failed");
            self.finish_pending_connection(
                owner,
                PendingConnectionOutcome::Failed("recovered connection gossip failed".to_string()),
            )
            .await;
            self.remove_peer_after_recovered_gossip_failure(remote)
                .await;
        }
    }

    pub(crate) async fn complete_recovered_connection_waiter(
        &self,
        waiter: PendingConnectionWaiter,
    ) {
        let remote = waiter.peer_id;
        if self.await_pending_connection(waiter).await.is_err() {
            self.remove_peer_after_recovered_gossip_failure(remote)
                .await;
        }
    }

    pub(crate) async fn remove_peer_after_recovered_gossip_failure(&self, remote: EndpointId) {
        tracing::info!(
            "Reconnect gossip to {} failed — peer is dead, removing",
            remote.fmt_short()
        );
        self.remove_peer(remote, MeshPeerRemovalReason::RecoveredGossipFailed)
            .await;
    }

    pub(crate) async fn recovered_connection_gossip_ok(
        &self,
        remote: EndpointId,
        new_conn: Connection,
    ) -> bool {
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            self.initiate_gossip(new_conn, remote),
        )
        .await
        .map(|result| result.is_ok())
        .unwrap_or(false)
    }

    pub(crate) fn spawn_gossip_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            if let Err(error) = node
                .handle_gossip_stream(remote, protocol, send, recv)
                .await
            {
                tracing::warn!("Gossip stream error from {}: {error}", remote.fmt_short());
            }
        });
    }

    pub(crate) fn spawn_tunnel_map_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            if let Err(error) = node.handle_tunnel_map_stream(remote, protocol, recv).await {
                tracing::warn!(
                    "Tunnel map stream error from {}: {error}",
                    remote.fmt_short()
                );
            }
        });
    }

    pub(crate) fn spawn_route_request_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            if protocol == ControlProtocol::ProtoV1 {
                let proto_buf = match read_len_prefixed(&mut recv).await {
                    Ok(buf) => buf,
                    Err(error) => {
                        tracing::warn!(
                            "Route request: failed to read proto body — rejecting: {error}"
                        );
                        node.capture_route_request(remote, protocol, "read_error");
                        return;
                    }
                };
                let req = match crate::proto::node::RouteTableRequest::decode(proto_buf.as_slice())
                {
                    Ok(request) => request,
                    Err(error) => {
                        tracing::warn!("Route request: invalid protobuf — rejecting: {error}");
                        node.capture_route_request(remote, protocol, "decode_error");
                        return;
                    }
                };
                if let Err(error) = req.validate_frame() {
                    tracing::warn!("Route request: frame validation failed — rejecting: {error}");
                    node.capture_route_request(remote, protocol, "validation_error");
                    return;
                }
            }
            if node
                .state
                .lock()
                .await
                .requirement_rejected_peers
                .contains(&remote)
            {
                tracing::warn!(
                    "Route request: refusing topology disclosure to requirement-rejected peer {}",
                    remote.fmt_short()
                );
                return;
            }
            let is_admitted = node
                .state
                .lock()
                .await
                .peers
                .get(&remote)
                .is_some_and(PeerInfo::is_admitted);
            if !is_admitted {
                tracing::warn!(
                    "Route request: refusing topology disclosure to unadmitted peer {}",
                    remote.fmt_short()
                );
                return;
            }
            use prost::Message as _;
            let mut send = send;
            let table = node.routing_table().await;
            let proto_table = routing_table_to_proto(&table);
            if write_len_prefixed(&mut send, &proto_table.encode_to_vec())
                .await
                .is_err()
            {
                node.capture_route_request(remote, protocol, "write_error");
                return;
            }
            node.capture_route_request(remote, protocol, "served");
            let _ = send.finish();
        });
    }

    pub(crate) fn spawn_plugin_channel_stream(
        &self,
        remote: EndpointId,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            if let Err(error) = node.handle_plugin_channel_stream(remote, send, recv).await {
                tracing::debug!(
                    "Plugin channel stream error from {}: {error}",
                    remote.fmt_short()
                );
            }
        });
    }

    pub(crate) fn spawn_plugin_bulk_stream(
        &self,
        remote: EndpointId,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            if let Err(error) = node.handle_plugin_bulk_stream(remote, send, recv).await {
                tracing::debug!(
                    "Plugin bulk stream error from {}: {error}",
                    remote.fmt_short()
                );
            }
        });
    }

    pub(crate) fn spawn_plugin_mesh_stream(
        &self,
        remote: EndpointId,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            if let Err(error) = node.handle_plugin_mesh_stream(remote, send, recv).await {
                tracing::debug!(
                    "Plugin mesh stream error from {}: {error}",
                    remote.fmt_short()
                );
            }
        });
    }

    pub(crate) fn spawn_subprotocol_stream(
        &self,
        remote: EndpointId,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            if let Err(error) = node
                .handle_mesh_subprotocol_stream(remote, send, recv)
                .await
            {
                tracing::debug!(
                    "subprotocol stream error from {}: {error}",
                    remote.fmt_short()
                );
            }
        });
    }

    pub(crate) fn spawn_peer_down_stream(
        &self,
        remote: EndpointId,
        recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            node.handle_peer_down_stream(remote, recv).await;
        });
    }
}
impl Node {
    pub(crate) fn spawn_non_tunnel_mesh_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        stream_type: u8,
        send: iroh::endpoint::SendStream,
        recv: iroh::endpoint::RecvStream,
    ) {
        match stream_type {
            STREAM_GOSSIP => self.spawn_gossip_stream(remote, protocol, send, recv),
            STREAM_TUNNEL_MAP => self.spawn_tunnel_map_stream(remote, protocol, recv),
            STREAM_ROUTE_REQUEST => self.spawn_route_request_stream(remote, protocol, send, recv),
            STREAM_PEER_DOWN => self.spawn_peer_down_stream(remote, recv),
            STREAM_PEER_LEAVING => self.spawn_peer_leaving_stream(remote, recv),
            STREAM_DIRECT_PATH_REQUEST => self.spawn_direct_path_request_stream(remote, recv),
            STREAM_PLUGIN_CHANNEL => self.spawn_plugin_channel_stream(remote, send, recv),
            STREAM_PLUGIN_BULK_TRANSFER => self.spawn_plugin_bulk_stream(remote, send, recv),
            STREAM_PLUGIN_MESH_STREAM => self.spawn_plugin_mesh_stream(remote, send, recv),
            STREAM_SUBPROTOCOL => self.spawn_subprotocol_stream(remote, send, recv),
            other => tracing::warn!("Unknown stream type {other} from {}", remote.fmt_short()),
        }
    }

    pub(crate) fn spawn_peer_leaving_stream(
        &self,
        remote: EndpointId,
        recv: iroh::endpoint::RecvStream,
    ) {
        let node = self.clone();
        tokio::spawn(async move {
            node.handle_peer_leaving_stream(remote, recv).await;
        });
    }

    pub(crate) async fn handle_peer_leaving_stream(
        &self,
        remote: EndpointId,
        mut recv: iroh::endpoint::RecvStream,
    ) {
        let Some(leaving_id) = self.decode_peer_leaving(remote, &mut recv).await else {
            return;
        };
        emit_mesh_info(format!(
            "👋 Peer {} announced clean shutdown",
            leaving_id.fmt_short()
        ));
        self.cleanup_peer_leaving(leaving_id).await;
    }

    pub(crate) async fn cleanup_peer_leaving(&self, leaving_id: EndpointId) {
        let mut state = self.state.lock().await;
        state
            .dead_peers
            .insert(leaving_id, std::time::Instant::now());
        state.connections.remove(&leaving_id);
        drop(state);
        self.remove_peer(leaving_id, MeshPeerRemovalReason::CleanShutdown)
            .await;
    }

    pub(crate) async fn decode_peer_leaving(
        &self,
        remote: EndpointId,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> Option<EndpointId> {
        let frame = self.read_peer_leaving_frame(recv).await?;
        self.resolve_peer_leaving_frame(remote, &frame)
    }

    pub(crate) async fn read_peer_leaving_frame(
        &self,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> Option<crate::proto::node::PeerLeaving> {
        let proto_buf = match read_len_prefixed(recv).await {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!("PeerLeaving: failed to read proto body — rejecting: {e}");
                return None;
            }
        };
        self.decode_peer_leaving_proto(&proto_buf)
    }

    pub(crate) fn decode_peer_leaving_proto(
        &self,
        proto_buf: &[u8],
    ) -> Option<crate::proto::node::PeerLeaving> {
        let frame = match crate::proto::node::PeerLeaving::decode(proto_buf) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("PeerLeaving: invalid protobuf — rejecting: {e}");
                return None;
            }
        };
        if let Err(e) = frame.validate_frame() {
            tracing::warn!("PeerLeaving: frame validation failed — rejecting: {e}");
            return None;
        }
        Some(frame)
    }

    pub(crate) fn resolve_peer_leaving_frame(
        &self,
        remote: EndpointId,
        frame: &crate::proto::node::PeerLeaving,
    ) -> Option<EndpointId> {
        match resolve_peer_leaving(remote, frame) {
            Ok(id) => Some(id),
            Err(e) => {
                tracing::warn!("PeerLeaving from {}: rejected ({})", remote.fmt_short(), e);
                None
            }
        }
    }
}
impl Node {
    pub(crate) async fn reserve_pending_connection(
        &self,
        peer_id: EndpointId,
    ) -> PendingConnectionReservation {
        let mut state = self.state.lock().await;
        if let Some(pending) = state
            .pending_connection_is_active(peer_id)
            .then(|| state.pending_connections.get(&peer_id))
            .flatten()
        {
            return PendingConnectionReservation::Waiter(pending.waiter(peer_id));
        }

        let attempt_id = PendingConnectionAttemptId(state.next_pending_connection_attempt);
        state.next_pending_connection_attempt =
            state.next_pending_connection_attempt.wrapping_add(1);
        let (outcome_tx, outcome_rx) = watch::channel(None);
        state.pending_connections.insert(
            peer_id,
            PendingConnectionHandshake {
                attempt_id,
                outcome_rx,
            },
        );
        PendingConnectionReservation::Owner(PendingConnectionAttemptOwner {
            peer_id,
            attempt_id,
            outcome_tx,
        })
    }

    pub(crate) async fn finish_pending_connection(
        &self,
        owner: PendingConnectionAttemptOwner,
        outcome: PendingConnectionOutcome,
    ) {
        owner.outcome_tx.send_replace(Some(outcome));
        let mut state = self.state.lock().await;
        if state
            .pending_connections
            .get(&owner.peer_id)
            .is_some_and(|pending| pending.attempt_id == owner.attempt_id)
        {
            state.pending_connections.remove(&owner.peer_id);
        }
    }

    pub(crate) async fn remove_pending_connection_if_attempt(
        &self,
        peer_id: EndpointId,
        attempt_id: PendingConnectionAttemptId,
    ) {
        let mut state = self.state.lock().await;
        if state
            .pending_connections
            .get(&peer_id)
            .is_some_and(|pending| pending.attempt_id == attempt_id)
        {
            state.pending_connections.remove(&peer_id);
        }
    }

    pub(crate) async fn await_pending_connection(
        &self,
        mut waiter: PendingConnectionWaiter,
    ) -> Result<()> {
        loop {
            let outcome = waiter.outcome_rx.borrow().clone();
            if let Some(outcome) = outcome {
                return outcome.into_result(waiter.peer_id);
            }
            if waiter.outcome_rx.changed().await.is_err() {
                self.remove_pending_connection_if_attempt(waiter.peer_id, waiter.attempt_id)
                    .await;
                anyhow::bail!(
                    "connection attempt to {} ended without a terminal result",
                    waiter.peer_id.fmt_short()
                )
            }
        }
    }

    async fn close_connection_after_failed_gossip(
        &self,
        peer_id: EndpointId,
        conn: &Connection,
        reason: &'static [u8],
    ) {
        let removed = self.remove_connection_if_stable_id(peer_id, conn).await;
        if let Some(removed_conn) = removed {
            removed_conn.close(0u32.into(), reason);
        } else {
            conn.close(0u32.into(), reason);
        }
    }

    pub(crate) async fn complete_cached_connection_gossip(
        &self,
        peer_id: EndpointId,
        conn: Connection,
        discover_peers: bool,
    ) -> Result<()> {
        let result = if discover_peers {
            self.initiate_gossip(conn.clone(), peer_id).await
        } else {
            self.initiate_gossip_inner(conn.clone(), peer_id, false)
                .await
        };
        if let Err(error) = result {
            self.close_connection_after_failed_gossip(peer_id, &conn, b"cached-gossip-failed")
                .await;
            return Err(error);
        }
        Ok(())
    }

    pub(crate) async fn complete_cached_connection_gossip_single_flight(
        &self,
        peer_id: EndpointId,
        conn: Connection,
        discover_peers: bool,
    ) -> Result<()> {
        match self.reserve_pending_connection(peer_id).await {
            PendingConnectionReservation::Owner(owner) => {
                let result = self
                    .complete_cached_connection_gossip(peer_id, conn, discover_peers)
                    .await;
                let outcome = match &result {
                    Ok(()) => PendingConnectionOutcome::Admitted,
                    Err(error) => PendingConnectionOutcome::Failed(error.to_string()),
                };
                self.finish_pending_connection(owner, outcome).await;
                result
            }
            PendingConnectionReservation::Waiter(waiter) => {
                self.await_pending_connection(waiter).await
            }
        }
    }

    async fn use_cached_connection_if_ready(&self, peer_id: EndpointId) -> Result<bool> {
        let cached_conn = {
            let state = self.state.lock().await;
            match state.connections.get(&peer_id).cloned() {
                Some(_) if state.peers.get(&peer_id).is_some_and(PeerInfo::is_admitted) => {
                    return Ok(true);
                }
                Some(conn) => Some(conn),
                None => None,
            }
        };

        let Some(conn) = cached_conn else {
            return Ok(false);
        };

        self.complete_cached_connection_gossip_single_flight(peer_id, conn, true)
            .await?;
        self.schedule_selected_path_recheck(peer_id);
        Ok(true)
    }

    async fn outbound_connection_preflight_complete(&self, peer_id: EndpointId) -> Result<bool> {
        if peer_id == self.endpoint.id() {
            return Ok(true);
        }

        if self.use_cached_connection_if_ready(peer_id).await? {
            return Ok(true);
        }

        {
            let state = self.state.lock().await;
            if state
                .dead_peers
                .get(&peer_id)
                .is_some_and(|t| t.elapsed() < DEAD_PEER_TTL)
            {
                tracing::debug!("Skipping connection to dead peer {}", peer_id.fmt_short());
                return Ok(true);
            }
        }

        Ok(false)
    }

    async fn reserve_outbound_pending_connection(
        &self,
        peer_id: EndpointId,
    ) -> Result<Option<PendingConnectionAttemptOwner>> {
        match self.reserve_pending_connection(peer_id).await {
            PendingConnectionReservation::Owner(owner) => Ok(Some(owner)),
            PendingConnectionReservation::Waiter(waiter) => {
                self.await_pending_connection(waiter).await?;
                self.schedule_selected_path_recheck(peer_id);
                Ok(None)
            }
        }
    }

    async fn open_outbound_pending_connection(
        &self,
        addr: EndpointAddr,
        owner: PendingConnectionAttemptOwner,
    ) -> Result<(Connection, PendingConnectionAttemptOwner)> {
        let peer_id = addr.id;
        tracing::info!("Connecting to peer {}...", peer_id.fmt_short());
        let conn = match tokio::time::timeout(
            PEER_CONNECT_AND_GOSSIP_TIMEOUT,
            connect_mesh(&self.endpoint, addr),
        )
        .await
        {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => {
                let error = anyhow::anyhow!("Failed to connect to {}: {e}", peer_id.fmt_short());
                self.finish_pending_connection(
                    owner,
                    PendingConnectionOutcome::Failed(error.to_string()),
                )
                .await;
                return Err(error);
            }
            Err(_) => {
                let error = anyhow::anyhow!(
                    "Timeout connecting to {} ({}s)",
                    peer_id.fmt_short(),
                    PEER_CONNECT_AND_GOSSIP_TIMEOUT.as_secs()
                );
                self.finish_pending_connection(
                    owner,
                    PendingConnectionOutcome::Failed(error.to_string()),
                )
                .await;
                return Err(error);
            }
        };

        Ok((conn, owner))
    }

    async fn complete_outbound_pending_gossip(
        &self,
        peer_id: EndpointId,
        conn: &Connection,
        owner: PendingConnectionAttemptOwner,
    ) -> Result<PendingConnectionAttemptOwner> {
        if let Err(error) = self.initiate_gossip(conn.clone(), peer_id).await {
            conn.close(0u32.into(), b"connect-peer-gossip-failed");
            self.finish_pending_connection(
                owner,
                PendingConnectionOutcome::Failed(error.to_string()),
            )
            .await;
            return Err(error);
        }

        Ok(owner)
    }

    async fn publish_admitted_outbound_connection(
        &self,
        peer_id: EndpointId,
        conn: Connection,
        owner: PendingConnectionAttemptOwner,
    ) {
        {
            let mut state = self.state.lock().await;
            state.dead_peers.remove(&peer_id);
            state.connections.insert(peer_id, conn.clone());
        }

        let node_for_dispatch = self.clone();
        let conn_for_dispatch = conn.clone();
        tokio::spawn(async move {
            node_for_dispatch
                .dispatch_streams(conn_for_dispatch, peer_id)
                .await;
        });

        // Schedule a delayed selected-path refresh: the first gossip often goes
        // via relay because direct holepunch has not completed yet. Iroh may
        // upgrade the connection after admission, and diagnostics should report
        // the selected path it ultimately owns.
        self.schedule_selected_path_recheck(peer_id);
        self.finish_pending_connection(owner, PendingConnectionOutcome::Admitted)
            .await;
    }

    pub(crate) async fn connect_to_peer(&self, addr: EndpointAddr) -> Result<()> {
        let peer_id = addr.id;
        if self.outbound_connection_preflight_complete(peer_id).await? {
            return Ok(());
        }

        let Some(owner) = self.reserve_outbound_pending_connection(peer_id).await? else {
            return Ok(());
        };
        let (conn, owner) = self.open_outbound_pending_connection(addr, owner).await?;
        let owner = self
            .complete_outbound_pending_gossip(peer_id, &conn, owner)
            .await?;
        self.publish_admitted_outbound_connection(peer_id, conn, owner)
            .await;

        Ok(())
    }

    /// Spawn a delayed task that re-reads the currently-selected QUIC path for
    /// `peer_id` after the relay→direct transition typically completes, and
    /// updates the tracked selected-path/RTT observation. The first gossip
    /// round-trip often runs over the relay (inflated RTT) before holepunch
    /// finishes; this refresh records the real direct RTT and can trigger a
    /// re-election for split mode.
    pub(super) fn schedule_selected_path_recheck(&self, peer_id: EndpointId) {
        let node_for_recheck = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let conn = node_for_recheck
                .state
                .lock()
                .await
                .connections
                .get(&peer_id)
                .cloned();
            let Some(conn) = conn else {
                return;
            };
            let path_list = conn.paths();
            for path_info in &path_list {
                if !path_info.is_selected() {
                    continue;
                }
                let rtt_ms = path_info.rtt().as_millis() as u32;
                let rtt_ms = (rtt_ms != 0).then_some(rtt_ms);
                let path_type = if path_info.is_ip() { "direct" } else { "relay" };
                if let Some(rtt_ms) = rtt_ms {
                    emit_mesh_info(format!(
                        "📡 Peer {} RTT recheck: {}ms ({})",
                        peer_id.fmt_short(),
                        rtt_ms,
                        path_type
                    ));
                }
                node_for_recheck
                    .update_peer_selected_path(
                        peer_id,
                        SelectedPathObservation {
                            path_type,
                            rtt_ms,
                            observed_direct_remote_addr: match path_info.remote_addr() {
                                TransportAddr::Ip(addr) => Some(*addr),
                                _ => None,
                            },
                        },
                    )
                    .await;
                break;
            }
        });
    }
}
