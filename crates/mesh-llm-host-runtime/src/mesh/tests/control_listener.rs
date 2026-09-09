use super::*;

async fn build_mesh_api_for_control_tests(node: Node) -> api::MeshApi {
    let resolved_plugins = plugin::ResolvedPlugins {
        shared_endpoint: None,
        externals: vec![],
        inactive: vec![],
    };
    let (mesh_tx, _mesh_rx) = tokio::sync::mpsc::channel(1);
    let plugin_manager = plugin::PluginManager::start(
        &resolved_plugins,
        plugin::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
    )
    .await
    .unwrap();
    let runtime_data_collector = node.runtime_data_collector();
    let runtime_data_producer =
        runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
    api::MeshApi::new(api::MeshApiConfig {
        node,
        model_name: "test-model".to_string(),
        api_port: 3131,
        model_size_bytes: 0,
        owner_key_path: None,
        plugin_manager,
        affinity_router: affinity::AffinityRouter::default(),
        runtime_data_collector,
        runtime_data_producer,
    })
}

#[tokio::test]
async fn control_plane_listener_starts_with_owner() -> anyhow::Result<()> {
    let (node, secret_key) =
        Node::new_for_tests_with_secret(super::super::NodeRole::Worker).await?;
    *node.owner_summary.lock().await = verified_owner_summary("owner-a");

    node.maybe_start_control_listener(secret_key, None, None)
        .await?;

    let endpoint = node
        .control_endpoint()
        .await
        .expect("verified owner should start a control listener");
    let decoded = Node::decode_invite_token(&endpoint)?;
    assert_eq!(decoded.id, node.endpoint.id());
    assert_ne!(decoded, node.endpoint.addr());
    assert!(decoded.addrs.iter().any(|addr| match addr {
        iroh::TransportAddr::Ip(sock) => sock.ip().is_loopback(),
        _ => false,
    }));

    node.shutdown_control_listener().await;
    Ok(())
}

/// Regression test for the owner-control / main-mesh relay endpoint-id
/// collision.
///
/// The owner-control listener shares the node's secret key (and therefore its
/// iroh endpoint id) with the main mesh endpoint, because the control protocol
/// validates the dialed `target_node_id` against the main endpoint id. An iroh
/// relay keeps only one active connection per endpoint id, so if the control
/// endpoint also registered with the relay it would evict the main mesh
/// endpoint's relay registration ("Another endpoint connected with the same
/// endpoint id. No more messages will be received."), silently killing all
/// relay-delivered mesh traffic — gossip, joins, inference routing — for peers
/// that cannot reach this node directly. That defeats relay fallback and is the
/// root cause of consume-side "connects but never joins / catalog never syncs".
///
/// The fix keeps the control endpoint relay-disabled. This test pins the
/// observable invariant: the control endpoint token must advertise NO relay
/// URLs, so it can never contend for the shared id's relay slot. If someone
/// re-enables relay on the control endpoint, this fails.
#[tokio::test]
async fn control_plane_listener_token_carries_no_relay_urls() -> anyhow::Result<()> {
    let (node, secret_key) =
        Node::new_for_tests_with_secret(super::super::NodeRole::Worker).await?;
    *node.owner_summary.lock().await = verified_owner_summary("owner-a");

    node.maybe_start_control_listener(secret_key, None, None)
        .await?;

    let endpoint = node
        .control_endpoint()
        .await
        .expect("verified owner should start a control listener");
    let decoded = Node::decode_invite_token(&endpoint)?;

    // Same id as the main mesh endpoint (required by target-node validation)...
    assert_eq!(decoded.id, node.endpoint.id());
    // ...but ZERO relay URLs, so it cannot evict the main endpoint's relay slot.
    let relay_addrs = decoded
        .addrs
        .iter()
        .filter(|addr| matches!(addr, iroh::TransportAddr::Relay(_)))
        .count();
    assert_eq!(
        relay_addrs, 0,
        "owner-control token must not advertise relay URLs (shares the main \
         endpoint id; a relay registration would evict the main mesh endpoint)"
    );

    // Invariant (defence-in-depth): the control endpoint is loopback-only by
    // default — it must NOT advertise any non-loopback IP transport. This keeps
    // the control plane non-routable off-box unless an explicit advertise addr
    // is configured (see the dedicated test), preserving the separation intent
    // of the split-control design (#539) while fixing the relay-slot eviction.
    for addr in &decoded.addrs {
        if let iroh::TransportAddr::Ip(sock) = addr {
            assert!(
                sock.ip().is_loopback(),
                "owner-control token must only advertise loopback IPs by default, \
                 found non-loopback {sock}"
            );
        }
    }

    node.shutdown_control_listener().await;
    Ok(())
}

#[tokio::test]
async fn control_plane_listener_uses_explicit_advertised_address() -> anyhow::Result<()> {
    let (node, secret_key) =
        Node::new_for_tests_with_secret(super::super::NodeRole::Worker).await?;
    *node.owner_summary.lock().await = verified_owner_summary("owner-a");
    let advertised_addr = std::net::SocketAddr::from(([203, 0, 113, 10], 18443));

    node.maybe_start_control_listener(secret_key, None, Some(advertised_addr))
        .await?;

    let endpoint = node
        .control_endpoint()
        .await
        .expect("verified owner should start a control listener");
    let decoded = Node::decode_invite_token(&endpoint)?;
    assert_eq!(decoded.id, node.endpoint.id());
    assert_eq!(decoded.addrs.len(), 1);
    assert!(
        decoded
            .addrs
            .contains(&iroh::TransportAddr::Ip(advertised_addr))
    );

    node.shutdown_control_listener().await;
    Ok(())
}

#[tokio::test]
async fn control_plane_listener_disabled_without_owner() -> anyhow::Result<()> {
    let (node, secret_key) =
        Node::new_for_tests_with_secret(super::super::NodeRole::Worker).await?;

    node.maybe_start_control_listener(secret_key, Some("127.0.0.1:7447".parse().unwrap()), None)
        .await?;

    assert!(node.control_endpoint().await.is_none());
    Ok(())
}

#[tokio::test]
async fn control_plane_listener_accepts_only_control_alpn() -> anyhow::Result<()> {
    let (node, secret_key) =
        Node::new_for_tests_with_secret(super::super::NodeRole::Worker).await?;
    *node.owner_summary.lock().await = verified_owner_summary("owner-a");
    node.maybe_start_control_listener(secret_key, None, None)
        .await?;
    let endpoint = Node::decode_invite_token(
        &node
            .control_endpoint()
            .await
            .expect("verified owner should expose control endpoint"),
    )?;
    let client = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec(), ALPN_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;

    client
        .connect(endpoint.clone(), ALPN_CONTROL_V1)
        .await
        .expect("control endpoint should accept mesh-llm-control/1");
    assert!(client.connect(endpoint, ALPN_V1).await.is_err());

    node.shutdown_control_listener().await;
    Ok(())
}

#[tokio::test]
async fn stalled_owner_control_handshake_expires_deterministically() -> anyhow::Result<()> {
    let owner_keypair = test_owner_keypair(0x8d, 0x8e);
    let tmp = std::env::temp_dir().join(format!(
        "mesh-llm-control-stalled-handshake-{}",
        rand::random::<u64>()
    ));
    let (server, _secret_key, _config_path) =
        start_owner_control_test_server(&owner_keypair, &tmp).await?;
    let control_addr = Node::decode_invite_token(
        &server
            .control_endpoint()
            .await
            .expect("owner-control endpoint should be available"),
    )?;
    let client = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let connection = client.connect(control_addr, ALPN_CONTROL_V1).await?;
    let (mut send, mut recv) = connection.open_bi().await?;
    send.write_all(&[0, 0]).await?;

    let envelope = tokio::time::timeout(
        std::time::Duration::from_secs(4),
        read_owner_control_envelope(&mut recv),
    )
    .await
    .expect("server handshake deadline should expire")?;
    let error = envelope
        .error
        .expect("stalled handshake should return a structured error");
    assert_eq!(
        crate::proto::node::OwnerControlErrorCode::try_from(error.code),
        Ok(crate::proto::node::OwnerControlErrorCode::InvalidHandshake)
    );
    assert_eq!(error.message, "owner-control handshake timed out after 2s");

    server.shutdown_control_listener().await;
    std::fs::remove_dir_all(&tmp).ok();
    Ok(())
}

#[tokio::test]
async fn control_plane_endpoint_not_in_gossip_or_status() -> anyhow::Result<()> {
    let (node, secret_key) =
        Node::new_for_tests_with_secret(super::super::NodeRole::Worker).await?;
    *node.owner_summary.lock().await = verified_owner_summary("owner-a");
    node.maybe_start_control_listener(secret_key, None, None)
        .await?;
    let control_endpoint = node
        .control_endpoint()
        .await
        .expect("verified owner should expose control endpoint");

    let announcements = node.collect_announcements().await;
    assert!(
        announcements
            .iter()
            .all(|announcement| encode_endpoint_addr_token(&announcement.addr) != control_endpoint)
    );

    let api = build_mesh_api_for_control_tests(node.clone()).await;
    api.set_control_bootstrap(api::ControlBootstrapPayload {
        enabled: true,
        local_only: true,
        requires_explicit_remote_endpoint: true,
        endpoint: Some(control_endpoint.clone()),
        disabled_reason: None,
        message: None,
        suggested_commands: None,
    })
    .await;
    let status_snapshot = api.status_snapshot_string().await;
    assert!(!status_snapshot.contains(&control_endpoint));

    node.shutdown_control_listener().await;
    Ok(())
}

#[tokio::test]
async fn control_plane_listener_shutdown_stops_listener_task() -> anyhow::Result<()> {
    let (node, secret_key) =
        Node::new_for_tests_with_secret(super::super::NodeRole::Worker).await?;
    *node.owner_summary.lock().await = verified_owner_summary("owner-a");
    node.maybe_start_control_listener(secret_key, None, None)
        .await?;
    let endpoint = Node::decode_invite_token(
        &node
            .control_endpoint()
            .await
            .expect("verified owner should expose control endpoint"),
    )?;

    node.shutdown_control_listener().await;
    assert!(node.control_endpoint().await.is_none());

    let client = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(SecretKey::generate())
        .alpns(vec![ALPN_CONTROL_V1.to_vec()])
        .relay_mode(iroh::endpoint::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let reconnect = tokio::time::timeout(
        std::time::Duration::from_millis(500),
        client.connect(endpoint, ALPN_CONTROL_V1),
    )
    .await;
    assert!(
        !matches!(reconnect, Ok(Ok(_))),
        "closed control endpoint unexpectedly accepted a connection"
    );
    Ok(())
}
