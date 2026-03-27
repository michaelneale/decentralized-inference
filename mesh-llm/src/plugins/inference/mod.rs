use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::{
    api, blackboard_display_name, build_serving_list, check_unserved_model, download, election,
    ensure_draft, launch, load_resolved_plugins, mesh, mesh::NodeRole, nostr, nostr_rediscovery,
    nostr_relays, pick_model_assignment, pipeline, plugin, plugin_host_mode, proxy, router, tunnel,
    update_pi_models_json, Cli, VERSION,
};

/// Built-in inference orchestration.
///
/// This is the first extraction step toward an explicit `inference` plugin
/// boundary: the mesh core stays in the crate root while the serving, console,
/// and request-routing runtime moves behind `plugins::inference`.
pub(crate) async fn run_auto(
    mut cli: Cli,
    resolved_models: Vec<PathBuf>,
    requested_model_names: Vec<String>,
    bin_dir: PathBuf,
) -> Result<()> {
    let api_port = cli.port;
    let console_port = if cli.no_console {
        None
    } else {
        Some(cli.console)
    };
    let local_models = mesh::scan_local_models();
    let is_client = cli.client;

    let resolved_plugins = load_resolved_plugins(&cli)?;

    let role = if is_client {
        NodeRole::Client
    } else {
        NodeRole::Worker
    };
    let (node, channels) = mesh::Node::start(
        role,
        &cli.relay,
        cli.bind_port,
        cli.max_vram,
        cli.enumerate_host,
    )
    .await?;
    node.set_available_models(local_models.clone()).await;
    node.set_requested_models(requested_model_names).await;
    node.set_blackboard_name(blackboard_display_name(&cli, &node))
        .await;

    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(&cli), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);

    if !cli.join.is_empty() {
        for token in &cli.join {
            if let Err(e) = node.join(token).await {
                eprintln!("⚠ Failed to join mesh: {e}");
            }
        }
        node.start_accepting();
        if !cli.client {
            node.set_role(NodeRole::Worker).await;
        }

        if cli.auto {
            let rediscover_node = node.clone();
            let rediscover_relays = nostr_relays(&cli.nostr_relay);
            let rediscover_relay_urls = cli.relay.clone();
            let rediscover_mesh_name = cli.mesh_name.clone();
            tokio::spawn(async move {
                nostr_rediscovery(
                    rediscover_node,
                    rediscover_relays,
                    rediscover_relay_urls,
                    rediscover_mesh_name,
                )
                .await;
            });
        }
    }

    let mut bootstrap_listener_tx = if !cli.join.is_empty() {
        let (stop_tx, stop_rx) =
            tokio::sync::mpsc::channel::<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>(1);
        let boot_node = node.clone();
        let boot_port = api_port;
        let listen_all = cli.listen_all;
        tokio::spawn(async move {
            bootstrap_proxy(boot_node, boot_port, stop_rx, listen_all).await;
        });
        Some(stop_tx)
    } else {
        None
    };

    let model = if !resolved_models.is_empty() {
        resolved_models[0].clone()
    } else {
        eprintln!("No --model specified, checking local models against mesh...");
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;

        let assignment = pick_model_assignment(&node, &local_models).await;
        let assignment = if assignment.is_none() && cli.auto && !is_client {
            let pack = nostr::auto_model_pack(node.vram_bytes() as f64 / 1e9);
            if !pack.is_empty() {
                eprintln!(
                    "📋 No unserved demand — serving {} for {:.0}GB VRAM",
                    pack[0],
                    node.vram_bytes() as f64 / 1e9
                );
                Some(pack[0].clone())
            } else {
                assignment
            }
        } else {
            assignment
        };
        if let Some(model_name) = assignment {
            eprintln!("Mesh assigned model: {model_name}");
            let model_path = mesh::find_model_path(&model_name);
            if model_path.exists() {
                model_path
            } else if let Some(cat) = download::find_model(&model_name) {
                eprintln!("📥 Downloading {} for mesh...", model_name);
                let dest = download::models_dir().join(cat.file);
                download::download_model(cat).await?;
                dest
            } else {
                let alt = download::models_dir().join(&model_name);
                if alt.exists() {
                    alt
                } else {
                    model_path
                }
            }
        } else {
            drop(bootstrap_listener_tx.take());
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            if is_client {
                eprintln!("📡 Running as client — proxying requests to mesh");
            } else {
                eprintln!("💤 No matching model on disk — running as standby GPU node");
                eprintln!(
                    "   VRAM: {:.1}GB, models on disk: {:?}",
                    node.vram_bytes() as f64 / 1e9,
                    local_models
                );
                eprintln!("   Proxying requests to other nodes. Will activate when needed.");
            }
            match run_passive(&cli, node.clone(), is_client, plugin_manager.clone()).await? {
                Some(model_name) => {
                    let model_path = mesh::find_model_path(&model_name);
                    if model_path.exists() {
                        model_path
                    } else {
                        let alt = download::models_dir().join(&model_name);
                        if alt.exists() {
                            alt
                        } else {
                            model_path
                        }
                    }
                }
                None => return Ok(()),
            }
        }
    };

    let model_name = {
        let stem = model
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        router::strip_split_suffix_owned(&stem)
    };

    let model_source = if !cli.model.is_empty() {
        cli.model[0].to_string_lossy().to_string()
    } else {
        model_name.clone()
    };
    node.set_model_source(model_source).await;
    let all_serving = build_serving_list(&resolved_models, &model_name);
    node.set_serving_models(all_serving.clone()).await;
    node.set_models(all_serving).await;
    node.regossip().await;

    if cli.draft.is_none() && !cli.no_draft {
        if let Some(draft_path) = ensure_draft(&model).await {
            eprintln!("Auto-detected draft model: {}", draft_path.display());
            cli.draft = Some(draft_path);
        }
    }

    launch::kill_orphan_rpc_servers().await;

    let rpc_port = launch::start_rpc_server(&bin_dir, cli.device.as_deref(), Some(&model)).await?;
    tracing::info!("rpc-server on 127.0.0.1:{rpc_port} serving {model_name}");

    let tunnel_mgr =
        tunnel::Manager::start(node.clone(), rpc_port, channels.rpc, channels.http).await?;

    let (target_tx, target_rx) = tokio::sync::watch::channel(election::ModelTargets::default());
    let target_tx = std::sync::Arc::new(target_tx);
    let (drop_tx, mut drop_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    let existing_listener = if let Some(tx) = bootstrap_listener_tx {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let _ = tx.send(resp_tx).await;
        resp_rx.await.ok()
    } else {
        None
    };

    let proxy_node = node.clone();
    let proxy_rx = target_rx.clone();
    let listen_all = cli.listen_all;
    tokio::spawn(async move {
        api_proxy(
            proxy_node,
            api_port,
            proxy_rx,
            drop_tx,
            existing_listener,
            listen_all,
        )
        .await;
    });

    let model_name_for_console = model_name.clone();
    let console_state = if let Some(cport) = console_port {
        let model_size_bytes = election::total_model_bytes(&model);
        let cs = api::MeshApi::new(
            node.clone(),
            model_name_for_console.clone(),
            api_port,
            model_size_bytes,
            plugin_manager.clone(),
        );
        cs.set_nostr_relays(nostr_relays(&cli.nostr_relay)).await;
        cs.set_nostr_discovery(cli.nostr_discovery).await;
        if let Some(draft) = &cli.draft {
            let dn = draft
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            cs.set_draft_name(dn).await;
        }
        if let Some(ref name) = cli.mesh_name {
            cs.set_mesh_name(name.clone()).await;
        }
        let cs2 = cs.clone();
        let console_rx = target_rx.clone();
        let mn = model_name_for_console.clone();
        let listen_all = cli.listen_all;
        tokio::spawn(async move {
            let (adapted_tx, adapted_rx) =
                tokio::sync::watch::channel(election::InferenceTarget::None);
            tokio::spawn(async move {
                let mut rx = console_rx;
                loop {
                    let targets = rx.borrow().clone();
                    let target = targets.get(&mn);
                    adapted_tx.send_replace(target);
                    if rx.changed().await.is_err() {
                        break;
                    }
                }
            });
            api::start(cport, cs2, adapted_rx, listen_all).await;
        });
        Some(cs)
    } else {
        None
    };

    tracing::info!("Entering auto-election for model: {model_name}");
    let node2 = node.clone();
    let tunnel_mgr2 = tunnel_mgr.clone();
    let bin_dir2 = bin_dir.clone();
    let model2 = model.clone();
    let draft2 = cli.draft.clone();
    let draft_max = cli.draft_max;
    let force_split = cli.split;
    let cb_console_port = console_port;
    let model_name_for_cb = model_name.clone();
    let model_name_for_election = model_name.clone();
    let node_for_cb = node.clone();
    let primary_target_tx = target_tx.clone();
    tokio::spawn(async move {
        election::election_loop(
            node2,
            tunnel_mgr2,
            rpc_port,
            bin_dir2,
            model2,
            model_name_for_election,
            draft2,
            draft_max,
            force_split,
            cli.ctx_size,
            primary_target_tx,
            move |is_host, llama_ready| {
                if llama_ready {
                    let n = node_for_cb.clone();
                    tokio::spawn(async move {
                        n.set_llama_ready(true).await;
                    });
                }
                if is_host && llama_ready {
                    let url = format!("http://localhost:{api_port}");
                    eprintln!("  API:     {url}");
                    if let Some(cp) = cb_console_port {
                        eprintln!("  Console: http://localhost:{cp}");
                    }
                    update_pi_models_json(&model_name_for_cb, api_port);
                    eprintln!();
                    eprintln!("  pi:    pi --provider mesh --model {model_name_for_cb}");
                    eprintln!(
                        "  goose: GOOSE_PROVIDER=openai OPENAI_HOST={url} OPENAI_API_KEY=mesh GOOSE_MODEL={model_name_for_cb} goose session"
                    );
                } else if is_host {
                    eprintln!("⏳ Starting llama-server...");
                } else {
                    eprintln!("  API: http://localhost:{api_port} (proxied to host)");
                }
                if let Some(ref cs) = console_state {
                    let cs = cs.clone();
                    tokio::spawn(async move {
                        cs.update(is_host, llama_ready).await;
                    });
                }
            },
        )
        .await;
    });

    if resolved_models.len() > 1 {
        eprintln!(
            "🔀 Multi-model mode: {} additional model(s)",
            resolved_models.len() - 1
        );
        let all_names: Vec<String> = resolved_models
            .iter()
            .map(|m| {
                m.file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            })
            .collect();
        node.set_models(all_names).await;
        node.regossip().await;

        for extra_model in resolved_models.iter().skip(1) {
            let extra_name = {
                let stem = extra_model
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                router::strip_split_suffix_owned(&stem)
            };
            let extra_node = node.clone();
            let extra_tunnel = tunnel_mgr.clone();
            let extra_bin = bin_dir.clone();
            let extra_path = extra_model.clone();
            let extra_target_tx = target_tx.clone();
            let extra_model_name = extra_name.clone();
            let api_port_extra = api_port;
            let ctx_size = cli.ctx_size;
            eprintln!("  + {extra_name}");
            tokio::spawn(async move {
                election::election_loop(
                    extra_node,
                    extra_tunnel,
                    0,
                    extra_bin,
                    extra_path,
                    extra_model_name.clone(),
                    None,
                    8,
                    false,
                    ctx_size,
                    extra_target_tx,
                    move |is_host, llama_ready| {
                        if is_host && llama_ready {
                            eprintln!("✅ [{extra_model_name}] ready (multi-model)");
                            eprintln!(
                                "  API: http://localhost:{api_port_extra} (model={extra_model_name})"
                            );
                        }
                    },
                )
                .await;
            });
        }
    }

    let nostr_publisher = if cli.publish {
        let nostr_keys = nostr::load_or_create_keys()?;
        let relays = nostr_relays(&cli.nostr_relay);
        let pub_node = node.clone();
        let pub_name = cli.mesh_name.clone();
        let pub_region = cli.region.clone();
        let pub_max_clients = cli.max_clients;
        Some(tokio::spawn(async move {
            nostr::publish_loop(
                pub_node,
                nostr_keys,
                relays,
                pub_name,
                pub_region,
                pub_max_clients,
                60,
            )
            .await;
        }))
    } else if cli.auto {
        let relays = nostr_relays(&cli.nostr_relay);
        let wd_node = node.clone();
        let wd_name = cli.mesh_name.clone();
        let wd_region = cli.region.clone();
        Some(tokio::spawn(async move {
            nostr::publish_watchdog(wd_node, relays, wd_name, wd_region, 120).await;
        }))
    } else {
        None
    };

    let drop_model_name = model_name.clone();
    let drop_node = node.clone();
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            eprintln!("\nShutting down...");
        }
        dropped = async {
            while let Some(name) = drop_rx.recv().await {
                if name == drop_model_name {
                    return name;
                }
                eprintln!("⚠ Drop request for '{}' — not our model ({}), ignoring", name, drop_model_name);
            }
            drop_model_name.clone()
        } => {
            eprintln!("\n🗑 Model '{}' dropped from mesh — shutting down", dropped);
            drop_node.set_serving(None).await;
        }
    }

    node.broadcast_leaving().await;

    if cli.publish {
        if let Ok(keys) = nostr::load_or_create_keys() {
            let relays = nostr_relays(&cli.nostr_relay);
            if let Ok(publisher) = nostr::Publisher::new(keys, &relays).await {
                let _ = publisher.unpublish().await;
                eprintln!("Removed Nostr listing");
            }
        }
    }
    if let Some(handle) = nostr_publisher {
        handle.abort();
    }

    launch::kill_llama_server().await;
    launch::kill_orphan_rpc_servers().await;
    Ok(())
}

pub(crate) async fn run_idle(cli: Cli, _bin_dir: PathBuf) -> Result<()> {
    let resolved_plugins = load_resolved_plugins(&cli)?;
    let my_vram_gb = mesh::detect_vram_bytes_capped(cli.max_vram) as f64 / 1e9;
    let local_models = mesh::scan_local_models();
    eprintln!(
        "mesh-llm v{VERSION} — {:.0}GB VRAM, {} models on disk",
        my_vram_gb,
        local_models.len()
    );
    eprintln!();
    eprintln!("  Console: http://localhost:{}", cli.console);
    eprintln!();
    eprintln!("  Start a mesh:");
    eprintln!("    mesh-llm --model Qwen2.5-32B                 serve a model");
    eprintln!("    mesh-llm --auto --model GLM-4.7-Flash-Q4_K_M --mesh-name \"my-mesh\"");
    eprintln!();
    eprintln!("  Join a mesh:");
    eprintln!("    mesh-llm --auto              discover and join automatically");
    eprintln!("    mesh-llm --join <token>      join by invite token");
    eprintln!("    mesh-llm --client --auto     join as API-only client");
    eprintln!();

    let (node, _channels) = mesh::Node::start(
        NodeRole::Worker,
        &cli.relay,
        cli.bind_port,
        cli.max_vram,
        cli.enumerate_host,
    )
    .await?;
    node.set_available_models(local_models).await;
    node.set_blackboard_name(blackboard_display_name(&cli, &node))
        .await;
    let (plugin_mesh_tx, plugin_mesh_rx) = tokio::sync::mpsc::channel(256);
    let plugin_manager =
        plugin::PluginManager::start(&resolved_plugins, plugin_host_mode(&cli), plugin_mesh_tx)
            .await?;
    node.set_plugin_manager(plugin_manager.clone()).await;
    node.start_plugin_channel_forwarder(plugin_mesh_rx);

    let cs = api::MeshApi::new(node.clone(), "(idle)".into(), cli.port, 0, plugin_manager);
    cs.set_nostr_relays(nostr_relays(&cli.nostr_relay)).await;
    cs.update(false, false).await;
    let cs2 = cs.clone();
    let (_tx, rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
    let listen_all = cli.listen_all;
    let console_port = cli.console;
    tokio::spawn(async move {
        api::start(console_port, cs2, rx, listen_all).await;
    });

    tokio::signal::ctrl_c().await?;
    eprintln!("\nShutting down...");
    Ok(())
}

async fn run_passive(
    cli: &Cli,
    node: mesh::Node,
    is_client: bool,
    plugin_manager: plugin::PluginManager,
) -> Result<Option<String>> {
    let local_port = cli.port;
    node.set_blackboard_name(blackboard_display_name(cli, &node))
        .await;

    if cli.publish && !is_client {
        let pub_node = node.clone();
        let nostr_keys = nostr::load_or_create_keys()?;
        let relays = nostr_relays(&cli.nostr_relay);
        let pub_name = cli.mesh_name.clone();
        let pub_region = cli.region.clone();
        let pub_max_clients = cli.max_clients;
        tokio::spawn(async move {
            nostr::publish_loop(
                pub_node,
                nostr_keys,
                relays,
                pub_name,
                pub_region,
                pub_max_clients,
                60,
            )
            .await;
        });
    } else if cli.auto && !is_client {
        let relays = nostr_relays(&cli.nostr_relay);
        let wd_node = node.clone();
        let wd_name = cli.mesh_name.clone();
        let wd_region = cli.region.clone();
        tokio::spawn(async move {
            nostr::publish_watchdog(wd_node, relays, wd_name, wd_region, 120).await;
        });
    }

    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let served = node.models_being_served().await;
    if !served.is_empty() {
        eprintln!("Models available in mesh: {:?}", served);
    }

    let addr = if cli.listen_all {
        "0.0.0.0"
    } else {
        "127.0.0.1"
    };
    let listener = tokio::net::TcpListener::bind(format!("{addr}:{local_port}"))
        .await
        .with_context(|| format!("Failed to bind to port {local_port}"))?;
    let mode = if is_client { "client" } else { "standby" };
    eprintln!("Passive {mode} ready:");
    eprintln!("  API:     http://localhost:{local_port}");
    eprintln!("  Console: http://localhost:{}", cli.console);

    {
        let cport = cli.console;
        let label = if is_client {
            "(client)".to_string()
        } else {
            "(standby)".to_string()
        };
        let cs = api::MeshApi::new(node.clone(), label, local_port, 0, plugin_manager);
        cs.set_nostr_relays(nostr_relays(&cli.nostr_relay)).await;
        cs.set_nostr_discovery(cli.nostr_discovery).await;
        if is_client {
            cs.set_client(true).await;
        }
        cs.update(false, true).await;
        let (_tx, rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
        let la = cli.listen_all;
        tokio::spawn(async move {
            api::start(cport, cs, rx, la).await;
        });
    }

    let (promote_tx, mut promote_rx) = tokio::sync::mpsc::channel::<String>(1);
    if !is_client {
        let watch_node = node.clone();
        let mut peer_rx = node.peer_change_rx.clone();
        let local_models = mesh::scan_local_models();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            let mut demand_interval = tokio::time::interval(std::time::Duration::from_secs(60));
            demand_interval.tick().await;
            loop {
                tokio::select! {
                    res = peer_rx.changed() => {
                        if res.is_err() { break; }
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                        while peer_rx.has_changed().unwrap_or(false) {
                            let _ = peer_rx.borrow_and_update();
                        }
                    }
                    _ = demand_interval.tick() => {}
                }
                if let Some(model_name) = check_unserved_model(&watch_node, &local_models).await {
                    eprintln!("🚀 Promoting from standby — serving {model_name}");
                    let _ = promote_tx.send(model_name).await;
                    break;
                }
            }
        });
    }

    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                let (tcp_stream, addr) = accept_result?;
                tcp_stream.set_nodelay(true)?;
                tracing::info!("Connection from {addr}");
                let node = node.clone();
                tokio::spawn(proxy::handle_mesh_request(node, tcp_stream, true));
            }
            Some(model_name) = promote_rx.recv() => {
                eprintln!("⬆️  Standby promoting to serve: {model_name}");
                return Ok(Some(model_name));
            }
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\nShutting down...");
                node.broadcast_leaving().await;
                return Ok(None);
            }
        }
    }
}

async fn api_proxy(
    node: mesh::Node,
    port: u16,
    target_rx: tokio::sync::watch::Receiver<election::ModelTargets>,
    drop_tx: tokio::sync::mpsc::UnboundedSender<String>,
    existing_listener: Option<tokio::net::TcpListener>,
    listen_all: bool,
) {
    let listener = match existing_listener {
        Some(l) => l,
        None => {
            let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
            match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!("Failed to bind API proxy to port {port}: {e}");
                    return;
                }
            }
        }
    };

    loop {
        let (tcp_stream, addr) = match listener.accept().await {
            Ok(r) => r,
            Err(_) => break,
        };
        let _ = tcp_stream.set_nodelay(true);

        let targets = target_rx.borrow().clone();
        let node = node.clone();
        let drop_tx = drop_tx.clone();
        tokio::spawn(async move {
            let mut buf = vec![0u8; 32768];
            match proxy::peek_request(&tcp_stream, &mut buf).await {
                Ok((n, model_name)) => {
                    if proxy::is_models_list_request(&buf[..n]) {
                        let models: Vec<String> = targets.targets.keys().cloned().collect();
                        let _ = proxy::send_models_list(tcp_stream, &models).await;
                        return;
                    }

                    if proxy::is_drop_request(&buf[..n]) {
                        if let Some(ref name) = model_name {
                            let _ = drop_tx.send(name.clone());
                            let _ = proxy::send_json_ok(
                                tcp_stream,
                                &serde_json::json!({"dropped": name}),
                            )
                            .await;
                        } else {
                            let _ = proxy::send_400(tcp_stream, "missing 'model' field").await;
                        }
                        return;
                    }

                    let (effective_model, classification) =
                        if model_name.is_none() || model_name.as_deref() == Some("auto") {
                            if let Some(body_json) = proxy::extract_body_json(&buf[..n]) {
                                let cl = router::classify(&body_json);
                                let available: Vec<(&str, f64)> = targets
                                    .targets
                                    .keys()
                                    .map(|name| (name.as_str(), 0.0))
                                    .collect();
                                let picked = router::pick_model_classified(&cl, &available);
                                if let Some(name) = picked {
                                    tracing::info!(
                                        "router: {:?}/{:?} tools={} → {name}",
                                        cl.category,
                                        cl.complexity,
                                        cl.needs_tools
                                    );
                                    (Some(name.to_string()), Some(cl))
                                } else {
                                    (None, Some(cl))
                                }
                            } else {
                                (None, None)
                            }
                        } else {
                            (model_name.clone(), None)
                        };

                    if let Some(ref name) = effective_model {
                        node.record_request(name);
                    }

                    let use_pipeline = classification
                        .as_ref()
                        .map(pipeline::should_pipeline)
                        .unwrap_or(false);

                    if use_pipeline {
                        if let Some(ref strong_name) = effective_model {
                            let planner = targets
                                .targets
                                .iter()
                                .find(|(name, target_vec)| {
                                    *name != strong_name
                                        && target_vec.iter().any(|t| {
                                            matches!(t, election::InferenceTarget::Local(_))
                                        })
                                })
                                .and_then(|(name, target_vec)| {
                                    target_vec.iter().find_map(|t| match t {
                                        election::InferenceTarget::Local(p) => {
                                            Some((name.clone(), *p))
                                        }
                                        _ => None,
                                    })
                                });

                            let strong_local_port =
                                targets.targets.get(strong_name.as_str()).and_then(|tv| {
                                    tv.iter().find_map(|t| match t {
                                        election::InferenceTarget::Local(p) => Some(*p),
                                        _ => None,
                                    })
                                });

                            if let (Some((planner_name, planner_port)), Some(strong_port)) =
                                (planner, strong_local_port)
                            {
                                tracing::info!(
                                    "pipeline: {planner_name} (plan) → {strong_name} (execute)"
                                );
                                proxy::pipeline_proxy_local(
                                    tcp_stream,
                                    &buf,
                                    n,
                                    planner_port,
                                    &planner_name,
                                    strong_port,
                                    &node,
                                )
                                .await;
                                return;
                            }
                        }
                    }

                    let target = if targets.moe.is_some() {
                        let session_hint = proxy::extract_session_hint(&buf[..n])
                            .unwrap_or_else(|| format!("{addr}"));
                        targets
                            .get_moe_target(&session_hint)
                            .unwrap_or(first_available_target(&targets))
                    } else if let Some(ref name) = effective_model {
                        let t = targets.get(name);
                        if matches!(t, election::InferenceTarget::None) {
                            tracing::debug!("Model '{}' not found, trying first available", name);
                            first_available_target(&targets)
                        } else {
                            t
                        }
                    } else {
                        first_available_target(&targets)
                    };

                    proxy::route_to_target(node, tcp_stream, target).await;
                }
                Err(_) => return,
            };
        });
    }
}

async fn bootstrap_proxy(
    node: mesh::Node,
    port: u16,
    mut stop_rx: tokio::sync::mpsc::Receiver<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>,
    listen_all: bool,
) {
    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    let listener = match tokio::net::TcpListener::bind(format!("{addr}:{port}")).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("Bootstrap proxy: failed to bind to port {port}: {e}");
            return;
        }
    };
    eprintln!("⚡ API ready (bootstrap): http://localhost:{port}");
    eprintln!("  Requests tunneled to mesh while GPU loads...");

    loop {
        tokio::select! {
            accept = listener.accept() => {
                let (tcp_stream, _addr) = match accept {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let _ = tcp_stream.set_nodelay(true);
                let node = node.clone();
                tokio::spawn(proxy::handle_mesh_request(node, tcp_stream, true));
            }
            resp_tx = stop_rx.recv() => {
                if let Some(tx) = resp_tx {
                    eprintln!("⚡ Bootstrap proxy handing off to full API proxy");
                    let _ = tx.send(listener);
                }
                return;
            }
        }
    }
}

fn first_available_target(targets: &election::ModelTargets) -> election::InferenceTarget {
    for hosts in targets.targets.values() {
        for target in hosts {
            if !matches!(target, election::InferenceTarget::None) {
                return target.clone();
            }
        }
    }
    election::InferenceTarget::None
}

use mesh_llm_plugin::{
    json_schema_tool, PluginMetadata, PluginRuntime as PluginProcessRuntime, SimplePlugin,
    ToolRouter,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct InferencePluginConfig {
    #[serde(default)]
    api_port: Option<u16>,
    #[serde(default)]
    console_port: Option<u16>,
    #[serde(default)]
    bind_address: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct InferencePluginHostContext {
    #[serde(default)]
    argv: Vec<String>,
    #[serde(default)]
    cwd: String,
    #[serde(default)]
    api_port: u16,
    #[serde(default)]
    console_port: u16,
    #[serde(default)]
    listen_all: bool,
    #[serde(default)]
    client: bool,
    #[serde(default)]
    auto: bool,
    #[serde(default)]
    publish: bool,
    #[serde(default)]
    mesh_name: Option<String>,
    #[serde(default)]
    region: Option<String>,
    #[serde(default)]
    display_name: Option<String>,
    #[serde(default)]
    bind_port: Option<u16>,
    #[serde(default)]
    raw_models: Vec<String>,
    #[serde(default)]
    join_tokens: Vec<String>,
    #[serde(default)]
    bin_dir_hint: Option<String>,
    #[serde(default)]
    config_path: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
struct InferencePluginStatusRequest {}

#[derive(Clone, Debug, Default, Serialize)]
struct InferencePluginState {
    host_context: Option<InferencePluginHostContext>,
    plugin_config: InferencePluginConfig,
    host_protocol_version: Option<u32>,
    host_version: Option<String>,
    mesh_visibility: Option<String>,
    api_addr: Option<String>,
    console_addr: Option<String>,
    local_peer_id: String,
    mesh_id: String,
    peers: std::collections::BTreeSet<String>,
    mesh_event_count: usize,
}

pub(crate) async fn run_plugin(name: String) -> Result<()> {
    PluginProcessRuntime::run(build_inference_plugin(name)).await
}

fn build_inference_plugin(name: String) -> SimplePlugin {
    let state = std::sync::Arc::new(tokio::sync::Mutex::new(InferencePluginState::default()));
    let init_state = state.clone();
    let event_state = state.clone();
    let health_state = state.clone();

    SimplePlugin::new(
        PluginMetadata::new(
            name.clone(),
            crate::VERSION,
            mesh_llm_plugin::plugin_server_info(
                "mesh-inference",
                crate::VERSION,
                "Inference Plugin",
                "Proof-of-concept inference plugin that owns its own HTTP listeners and consumes startup context from the host.",
                Some("Open the plugin-owned HTTP endpoints to inspect the startup context and live mesh event state."),
            ),
        )
        .with_capabilities(vec!["http-owned".into(), "startup-context".into()]),
    )
    .with_tool_router(inference_tool_router(state))
    .on_initialize(move |request, _context| {
        let state = init_state.clone();
        Box::pin(async move { initialize_inference_plugin(request, state).await })
    })
    .on_mesh_event(move |event, _context| {
        let state = event_state.clone();
        Box::pin(async move {
            record_inference_mesh_event(state, event).await;
            Ok(())
        })
    })
    .with_health(move |_context| {
        let state = health_state.clone();
        Box::pin(async move {
            let snapshot = snapshot_inference_plugin_state(&state).await;
            Ok(format!(
                "api={} console={} peers={}",
                snapshot.api_addr.unwrap_or_else(|| "unbound".into()),
                snapshot.console_addr.unwrap_or_else(|| "unbound".into()),
                snapshot.peers.len()
            ))
        })
    })
}

fn inference_tool_router(
    state: std::sync::Arc<tokio::sync::Mutex<InferencePluginState>>,
) -> ToolRouter {
    let mut router = ToolRouter::new();
    router.add_json_default::<InferencePluginStatusRequest, InferencePluginState, _>(
        json_schema_tool::<InferencePluginStatusRequest>(
            "status",
            "Inspect the inference plugin startup context and current HTTP bindings.",
        ),
        move |_request, _context| {
            let state = state.clone();
            Box::pin(async move { Ok(snapshot_inference_plugin_state(&state).await) })
        },
    );
    router
}

async fn initialize_inference_plugin(
    request: mesh_llm_plugin::PluginInitializeRequest,
    state: std::sync::Arc<tokio::sync::Mutex<InferencePluginState>>,
) -> mesh_llm_plugin::PluginResult<()> {
    let host_context: InferencePluginHostContext = serde_json::from_str(&request.host_context_json)
        .map_err(|err| mesh_llm_plugin::PluginError::invalid_params(err.to_string()))?;
    let plugin_config: InferencePluginConfig = serde_json::from_str(&request.plugin_config_json)
        .map_err(|err| mesh_llm_plugin::PluginError::invalid_params(err.to_string()))?;

    let bind_address = plugin_config.bind_address.clone().unwrap_or_else(|| {
        if host_context.listen_all {
            "0.0.0.0".into()
        } else {
            "127.0.0.1".into()
        }
    });
    let api_port = plugin_config.api_port.unwrap_or(host_context.api_port);
    let console_port = plugin_config
        .console_port
        .unwrap_or(host_context.console_port);

    let api_addr = start_inference_listener(&bind_address, api_port, "api", state.clone())
        .await
        .map_err(mesh_llm_plugin::PluginError::from)?;
    let console_addr = if console_port == api_port {
        api_addr.clone()
    } else {
        start_inference_listener(&bind_address, console_port, "console", state.clone())
            .await
            .map_err(mesh_llm_plugin::PluginError::from)?
    };

    {
        let mut guard = state.lock().await;
        guard.host_context = Some(host_context);
        guard.plugin_config = plugin_config;
        guard.host_protocol_version = Some(request.host_protocol_version);
        guard.host_version = Some(request.host_version);
        guard.mesh_visibility = Some(format!("{:?}", request.mesh_visibility).to_lowercase());
        guard.api_addr = Some(api_addr.clone());
        guard.console_addr = Some(console_addr.clone());
    }

    eprintln!("Inference plugin demo API: http://{api_addr}/startup-context");
    eprintln!("Inference plugin demo UI:  http://{console_addr}/");
    Ok(())
}

async fn record_inference_mesh_event(
    state: std::sync::Arc<tokio::sync::Mutex<InferencePluginState>>,
    event: mesh_llm_plugin::proto::MeshEvent,
) {
    let mut guard = state.lock().await;
    guard.mesh_event_count += 1;
    if !event.local_peer_id.is_empty() {
        guard.local_peer_id = event.local_peer_id;
    }
    if !event.mesh_id.is_empty() {
        guard.mesh_id = event.mesh_id;
    }
    if let Some(peer) = event.peer {
        let peer_id = peer.peer_id;
        match mesh_llm_plugin::proto::mesh_event::Kind::try_from(event.kind).ok() {
            Some(mesh_llm_plugin::proto::mesh_event::Kind::PeerDown) => {
                guard.peers.remove(&peer_id);
            }
            _ => {
                guard.peers.insert(peer_id);
            }
        }
    }
}

async fn snapshot_inference_plugin_state(
    state: &std::sync::Arc<tokio::sync::Mutex<InferencePluginState>>,
) -> InferencePluginState {
    state.lock().await.clone()
}

async fn start_inference_listener(
    bind_address: &str,
    port: u16,
    listener_kind: &'static str,
    state: std::sync::Arc<tokio::sync::Mutex<InferencePluginState>>,
) -> Result<String> {
    let listener = tokio::net::TcpListener::bind(format!("{bind_address}:{port}")).await?;
    let local_addr = listener.local_addr()?.to_string();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let state = state.clone();
            tokio::spawn(async move {
                let _ = handle_inference_http_request(&mut stream, listener_kind, state).await;
            });
        }
    });
    Ok(local_addr)
}

async fn handle_inference_http_request(
    stream: &mut tokio::net::TcpStream,
    listener_kind: &str,
    state: std::sync::Arc<tokio::sync::Mutex<InferencePluginState>>,
) -> Result<()> {
    let mut buf = vec![0u8; 8192];
    let n = stream.read(&mut buf).await?;
    if n == 0 {
        return Ok(());
    }
    let req = String::from_utf8_lossy(&buf[..n]);
    let method = req.split_whitespace().next().unwrap_or("GET");
    let path = req.split_whitespace().nth(1).unwrap_or("/");
    let snapshot = snapshot_inference_plugin_state(&state).await;

    match (method, path) {
        ("GET", "/") if listener_kind == "console" || listener_kind == "api" => {
            let html = format!(
                "<!doctype html><html><head><meta charset=\"utf-8\"><title>Inference Plugin</title></head><body><h1>Inference Plugin</h1><p>This page is served by the plugin process, not mesh-llm core.</p><p>API: {}</p><p>Console: {}</p><p>Mesh visibility: {}</p><p>Peers: {}</p><p><a href=\"/startup-context\">startup-context</a> · <a href=\"/state\">state</a> · <a href=\"/health\">health</a></p><pre>{}</pre></body></html>",
                snapshot.api_addr.clone().unwrap_or_default(),
                snapshot.console_addr.clone().unwrap_or_default(),
                snapshot.mesh_visibility.clone().unwrap_or_default(),
                snapshot.peers.len(),
                serde_json::to_string_pretty(&snapshot.host_context).unwrap_or_else(|_| "{}".into())
            );
            respond_with_body(
                stream,
                200,
                "OK",
                "text/html; charset=utf-8",
                html.as_bytes(),
            )
            .await?;
        }
        ("GET", "/startup-context") | ("GET", "/state") => {
            let body = serde_json::to_vec_pretty(&snapshot)?;
            respond_with_body(stream, 200, "OK", "application/json", &body).await?;
        }
        ("GET", "/health") => {
            let body = serde_json::to_vec(&serde_json::json!({
                "ok": true,
                "plugin": plugin::INFERENCE_PLUGIN_ID,
                "api_addr": snapshot.api_addr,
                "console_addr": snapshot.console_addr,
                "peer_count": snapshot.peers.len(),
            }))?;
            respond_with_body(stream, 200, "OK", "application/json", &body).await?;
        }
        _ => {
            respond_with_body(
                stream,
                404,
                "Not Found",
                "text/plain; charset=utf-8",
                b"not found",
            )
            .await?;
        }
    }
    Ok(())
}

async fn respond_with_body(
    stream: &mut tokio::net::TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    let header = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(body).await?;
    Ok(())
}
