use super::config::{MeshConfig, PluginConfigEntry};
use super::*;

fn private_host_mode() -> PluginHostMode {
    PluginHostMode {
        mesh_visibility: MeshVisibility::Private,
    }
}

#[test]
fn external_plugin_startup_policy_is_resolved() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "metrics".into(),
            enabled: Some(true),
            web_ui_enabled: None,
            command: Some("mesh-llm-plugin-metrics".into()),
            args: Vec::new(),
            url: None,
            settings: Default::default(),
            startup: PluginStartupConfig {
                connect_timeout_secs: Some(75),
                init_timeout_secs: Some(90),
                optional: true,
                lazy_start: true,
            },
        }],
        defaults: None,
        ..MeshConfig::default()
    };

    let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
    let spec = resolved
        .externals
        .iter()
        .find(|spec| spec.name == "metrics")
        .expect("configured plugin should resolve");

    assert_eq!(spec.startup.connect_timeout().as_secs(), 75);
    assert_eq!(spec.startup.init_timeout().as_secs(), 90);
    assert!(spec.startup.optional);
    assert!(spec.startup.lazy_start);
}

#[test]
fn optional_missing_installed_plugin_becomes_inactive_summary() {
    let config = MeshConfig {
        plugins: vec![PluginConfigEntry {
            name: "missing-optional".into(),
            enabled: Some(true),
            web_ui_enabled: None,
            command: None,
            args: Vec::new(),
            url: None,
            settings: Default::default(),
            startup: PluginStartupConfig {
                optional: true,
                ..PluginStartupConfig::default()
            },
        }],
        defaults: None,
        ..MeshConfig::default()
    };

    let resolved = resolve_plugins(&config, private_host_mode()).unwrap();

    assert_eq!(
        resolved
            .inactive
            .iter()
            .filter(|summary| summary.name == "missing-optional")
            .count(),
        1
    );
    let summary = resolved
        .inactive
        .iter()
        .find(|summary| summary.name == "missing-optional")
        .unwrap();
    assert_eq!(summary.status, "missing");
    assert_eq!(
        summary.startup.as_ref().map(|startup| startup.optional),
        Some(true)
    );
    assert!(
        summary
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("optional")
    );
}

#[tokio::test]
async fn required_plugin_load_failure_stops_manager_startup() {
    let specs = ResolvedPlugins {
        shared_endpoint: None,
        externals: vec![ExternalPluginSpec {
            name: "broken".into(),
            command: "mesh-llm-definitely-missing-plugin-binary".into(),
            args: vec!["--stdio".into()],
            url: None,
            env: BTreeMap::new(),
            startup: PluginStartupOptions::default(),
            web_ui_enabled: None,
            installed_metadata: None,
        }],
        inactive: Vec::new(),
    };
    let (mesh_tx, _mesh_rx) = mpsc::channel(1);

    let error = match PluginManager::start(&specs, private_host_mode(), mesh_tx).await {
        Ok(manager) => {
            manager.shutdown().await;
            panic!("required plugin failure must stop manager startup");
        }
        Err(error) => error,
    };

    let error_text = format!("{error:#}");
    assert!(error_text.contains("broken"), "{error_text}");
}

#[tokio::test]
async fn required_plugin_failure_rolls_back_plugins_loaded_earlier() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("plugin listener");
    let address = listener.local_addr().expect("listener address");
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("plugin connection");
        let mut stream = transport::LocalStream::Tcp(stream);
        let request = transport::read_envelope(&mut stream)
            .await
            .expect("initialize request");
        transport::write_envelope(
            &mut stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "first".into(),
                request_id: request.request_id,
                payload: Some(proto::envelope::Payload::InitializeResponse(
                    proto::InitializeResponse {
                        plugin_id: "first".into(),
                        plugin_protocol_version: PROTOCOL_VERSION,
                        plugin_version: "v1.0.0".into(),
                        server_info_json: serde_json::to_string(&ServerInfo::default())
                            .expect("server info"),
                        capabilities: Vec::new(),
                        manifest: None,
                    },
                )),
            },
        )
        .await
        .expect("initialize response");
        let mut byte = [0_u8; 1];
        let read = tokio::time::timeout(std::time::Duration::from_secs(2), stream.read(&mut byte))
            .await
            .expect("rollback should close the control connection")
            .unwrap_or(0);
        assert_eq!(read, 0, "rollback must disconnect the loaded plugin");
    });
    let specs = ResolvedPlugins {
        shared_endpoint: None,
        externals: vec![
            ExternalPluginSpec {
                name: "first".into(),
                command: String::new(),
                args: Vec::new(),
                url: Some(format!("test+tcp://{address}")),
                env: BTreeMap::new(),
                startup: PluginStartupOptions::default(),
                web_ui_enabled: None,
                installed_metadata: None,
            },
            ExternalPluginSpec {
                name: "broken".into(),
                command: "mesh-llm-definitely-missing-plugin-binary".into(),
                args: Vec::new(),
                url: None,
                env: BTreeMap::new(),
                startup: PluginStartupOptions::default(),
                web_ui_enabled: None,
                installed_metadata: None,
            },
        ],
        inactive: Vec::new(),
    };
    let (mesh_tx, _mesh_rx) = mpsc::channel(1);

    let error = match PluginManager::start(&specs, private_host_mode(), mesh_tx).await {
        Ok(manager) => {
            manager.shutdown().await;
            panic!("required plugin failure must stop manager startup");
        }
        Err(error) => error,
    };

    let error_text = format!("{error:#}");
    assert!(error_text.contains("broken"), "{error_text}");
    server.await.expect("plugin server task");
}

#[tokio::test]
async fn optional_plugin_load_failure_becomes_inactive_summary() {
    let specs = ResolvedPlugins {
        shared_endpoint: None,
        externals: vec![ExternalPluginSpec {
            name: "optional-broken".into(),
            command: "mesh-llm-definitely-missing-plugin-binary".into(),
            args: Vec::new(),
            url: None,
            env: BTreeMap::new(),
            startup: PluginStartupOptions {
                optional: true,
                ..PluginStartupOptions::default()
            },
            web_ui_enabled: None,
            installed_metadata: None,
        }],
        inactive: Vec::new(),
    };
    let (mesh_tx, _mesh_rx) = mpsc::channel(1);

    let manager = PluginManager::start(&specs, private_host_mode(), mesh_tx)
        .await
        .expect("optional plugin failure must not stop manager startup");
    let summaries = manager.list().await;
    manager.shutdown().await;

    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].name, "optional-broken");
    assert_eq!(summaries[0].status, "error");
    assert_eq!(
        summaries[0]
            .startup
            .as_ref()
            .map(|startup| startup.optional),
        Some(true)
    );
    assert!(!summaries[0].error.as_deref().unwrap_or_default().is_empty());
}

#[tokio::test]
async fn remote_connect_failures_honor_required_and_optional_policy() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve address");
    let address = listener.local_addr().expect("listener address");
    drop(listener);
    let remote_spec = |name: &str, optional| ExternalPluginSpec {
        name: name.into(),
        command: String::new(),
        args: Vec::new(),
        url: Some(format!("tcp://{address}")),
        env: BTreeMap::new(),
        startup: PluginStartupOptions {
            optional,
            ..PluginStartupOptions::default()
        },
        web_ui_enabled: None,
        installed_metadata: None,
    };

    let (required_tx, _required_rx) = mpsc::channel(1);
    let required = ResolvedPlugins {
        shared_endpoint: None,
        externals: vec![remote_spec("required-remote", false)],
        inactive: Vec::new(),
    };
    let required_error =
        match PluginManager::start(&required, private_host_mode(), required_tx).await {
            Ok(manager) => {
                manager.shutdown().await;
                panic!("required connection failure must stop startup");
            }
            Err(error) => error,
        };
    assert!(required_error.to_string().contains("required-remote"));

    let (optional_tx, _optional_rx) = mpsc::channel(1);
    let optional = ResolvedPlugins {
        shared_endpoint: None,
        externals: vec![remote_spec("optional-remote", true)],
        inactive: Vec::new(),
    };
    let manager = PluginManager::start(&optional, private_host_mode(), optional_tx)
        .await
        .expect("optional connection failure should not stop startup");
    let summaries = manager.list().await;
    manager.shutdown().await;
    assert_eq!(summaries[0].name, "optional-remote");
    assert_eq!(summaries[0].status, "error");
}

#[tokio::test]
async fn lazy_start_plugin_does_not_block_manager_startup() {
    let specs = ResolvedPlugins {
        shared_endpoint: None,
        externals: vec![ExternalPluginSpec {
            name: "lazy".into(),
            command: "mesh-llm-definitely-missing-plugin-binary".into(),
            args: Vec::new(),
            url: None,
            env: BTreeMap::new(),
            startup: PluginStartupOptions {
                optional: true,
                lazy_start: true,
                ..PluginStartupOptions::default()
            },
            web_ui_enabled: None,
            installed_metadata: None,
        }],
        inactive: Vec::new(),
    };
    let (mesh_tx, _mesh_rx) = mpsc::channel(1);

    let manager = PluginManager::start(&specs, private_host_mode(), mesh_tx)
        .await
        .expect("lazy plugin should not start during manager startup");
    let summaries = manager.list().await;
    manager.shutdown().await;

    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].name, "lazy");
    assert_eq!(summaries[0].status, "deferred");
    assert_eq!(
        summaries[0]
            .startup
            .as_ref()
            .map(|startup| startup.lazy_start),
        Some(true)
    );
    assert!(summaries[0].pid.is_none());
    assert!(
        summaries[0]
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("lazy")
    );
}
