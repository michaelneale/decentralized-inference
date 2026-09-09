use super::{
    BootstrapProxyStopTx, ConsoleSessionMode, DASHBOARD_FIRST_PAINT_TIMEOUT, DashboardContextUsage,
    InitialPromptMode, ManagedModelController, OpenAiGuardrailPolicyHandle,
    PassivePublicationSetup, RunAutoModelSelection, RunAutoModelSelectionContext,
    RuntimeCapacityLedger, RuntimeDashboardSnapshotProvider, RuntimeEvent, RuntimeInstanceRegistry,
    RuntimeOptions, StartupLocalModelTask, StartupModelPlan, StartupReadyReporter, api_proxy,
    bootstrap_proxy, bridge_publication_state, maybe_spawn_passive_promotion_task,
    next_runtime_instance_id, node_display_name, nostr_relays, resolve_runtime_owner_key_path,
    resolved_model_name, runtime_resource_planning_profile, select_run_auto_model_path,
    setup_passive_publication, shutdown_passive_runtime, sort_dashboard_endpoint_rows,
    spawn_embedded_runtime_control_forwarder, startup_local_model_loop, wait_shutdown_signal,
};
use crate::api;
use crate::inference::{election, skippy};
use crate::mesh;
use crate::network::{affinity, discovery as mesh_discovery, nostr, tunnel};
use crate::plugin;
use crate::runtime::interactive;
use crate::runtime::survey;
use crate::runtime::{InstanceLifecycleRecord, InstanceLifecycleState};
#[cfg(test)]
use crate::runtime::{StartupPinnedGpuTarget, dashboard_lanes_for_process};
use crate::system::backend;
use anyhow::{Context, Result};
use mesh_llm_events::{
    DashboardEndpointRow, DashboardLaunchPlan, DashboardModelRow, DashboardProcessRow, OutputEvent,
    RuntimeStatus, emit_event, output_sink,
};
#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[cfg(test)]
use skippy_protocol::FlashAttentionType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]

pub(super) struct InteractiveSpawnRequest {
    pub(super) prompt_mode: InitialPromptMode,
}

pub(super) fn serve_path_interactive_spawn_request(
    input_handler_enabled: bool,
    interactive_started: &AtomicBool,
    stdin_is_tty: bool,
) -> Option<InteractiveSpawnRequest> {
    if !input_handler_enabled || !stdin_is_tty {
        return None;
    }
    if interactive_started.swap(true, Ordering::AcqRel) {
        return None;
    }
    Some(InteractiveSpawnRequest {
        prompt_mode: InitialPromptMode::Deferred,
    })
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "used by the retained passive runtime compatibility lane and its focused tests"
    )
)]
pub(super) fn passive_path_interactive_spawn_request(
    console_session_mode: Option<ConsoleSessionMode>,
    stdin_is_tty: bool,
) -> Option<InteractiveSpawnRequest> {
    if console_session_mode.is_some() && stdin_is_tty {
        Some(InteractiveSpawnRequest {
            prompt_mode: InitialPromptMode::Immediate,
        })
    } else {
        None
    }
}

pub(super) fn startup_launch_plan(
    startup_models: &[StartupModelPlan],
    primary_model_name: &str,
    api_port: u16,
    console_port: Option<u16>,
    headless: bool,
    default_parallel: Option<usize>,
    default_backend_device: Option<String>,
) -> DashboardLaunchPlan {
    let mut llama_process_rows = Vec::new();

    let mut model_rows: Vec<_> = startup_models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            let model_name = startup_model_display_name(model);
            llama_process_rows.push(DashboardProcessRow {
                name: format!("llama-server {model_name}"),
                backend: String::new(),
                status: RuntimeStatus::Loading,
                port: 0,
                pid: 0,
            });

            DashboardModelRow {
                name: model_name,
                role: Some(if index == 0 { "primary" } else { "model" }.to_string()),
                status: RuntimeStatus::Loading,
                port: None,
                device: model
                    .pinned_gpu
                    .as_ref()
                    .map(|gpu| gpu.backend_device.clone())
                    .or_else(|| model.gpu_id.clone())
                    .or_else(|| default_backend_device.clone()),
                slots: Some(super::startup_models::resolve_model_parallel_slots(
                    model.parallel,
                    &plugin::GpuConfig {
                        assignment: plugin::GpuAssignment::Auto,
                        parallel: default_parallel,
                    },
                    4,
                )),
                quantization: None,
                ctx_size: model.ctx_size,
                ctx_used_tokens: None,
                lanes: None,
                file_size_gb: None,
            }
        })
        .collect();

    let mut webserver_rows = vec![DashboardEndpointRow {
        label: "API".to_string(),
        status: RuntimeStatus::NotReady,
        url: format!("http://localhost:{api_port}"),
        port: api_port,
        pid: None,
    }];
    if !headless && let Some(console_port) = console_port {
        webserver_rows.push(DashboardEndpointRow {
            label: "Console".to_string(),
            status: RuntimeStatus::NotReady,
            url: format!("http://localhost:{console_port}"),
            port: console_port,
            pid: None,
        });
    }
    sort_dashboard_endpoint_rows(&mut webserver_rows);

    if startup_models.is_empty() {
        llama_process_rows.push(DashboardProcessRow {
            name: format!("llama-server {primary_model_name}"),
            backend: String::new(),
            status: RuntimeStatus::Loading,
            port: 0,
            pid: 0,
        });
        model_rows.push(DashboardModelRow {
            name: primary_model_name.to_string(),
            role: Some("primary".to_string()),
            status: RuntimeStatus::Loading,
            port: None,
            device: default_backend_device,
            slots: default_parallel,
            quantization: None,
            ctx_size: None,
            ctx_used_tokens: None,
            lanes: None,
            file_size_gb: None,
        });
    }

    DashboardLaunchPlan {
        llama_process_rows,
        webserver_rows,
        loaded_model_rows: model_rows,
    }
}

pub(super) fn serve_path_builtin_endpoint_ready_events(
    api_url: String,
    console_url: Option<String>,
    headless: bool,
) -> Vec<OutputEvent> {
    let mut events = vec![OutputEvent::ApiReady { url: api_url }];

    if !headless && let Some(console_url) = console_url {
        events.push(OutputEvent::WebserverReady { url: console_url });
    }

    events
}

pub(super) fn socket_addr_http_url(addr: std::net::SocketAddr) -> String {
    format!("http://{addr}")
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "used by the retained passive runtime compatibility lane and listener tests"
    )
)]
pub(super) fn listener_http_url(
    listener: &tokio::net::TcpListener,
    fallback_port: u16,
    label: &str,
) -> String {
    listener_http_endpoint(listener, fallback_port, label).0
}

pub(super) fn listener_http_endpoint(
    listener: &tokio::net::TcpListener,
    fallback_port: u16,
    label: &str,
) -> (String, u16) {
    listener
        .local_addr()
        .map(|addr| (socket_addr_http_url(addr), addr.port()))
        .unwrap_or_else(|err| {
            tracing::warn!("{label}: failed to read listener address: {err}");
            (format!("http://localhost:{fallback_port}"), fallback_port)
        })
}

pub(super) async fn bind_runtime_tcp_listener(
    port: u16,
    listen_all: bool,
    label: &str,
) -> Result<tokio::net::TcpListener> {
    let addr = if listen_all { "0.0.0.0" } else { "127.0.0.1" };
    tokio::net::TcpListener::bind(format!("{addr}:{port}"))
        .await
        .with_context(|| format!("Failed to bind {label} to port {port}"))
}

pub(super) fn startup_default_backend_device(
    binary_flavor: Option<backend::BinaryFlavor>,
) -> Option<String> {
    let flavor = binary_flavor.or_else(platform_default_backend_flavor);
    if flavor == Some(backend::BinaryFlavor::Metal) {
        backend::backend_device_for_flavor(0, backend::BinaryFlavor::Metal)
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
pub(super) fn platform_default_backend_flavor() -> Option<backend::BinaryFlavor> {
    Some(backend::BinaryFlavor::Metal)
}

#[cfg(not(target_os = "macos"))]
pub(super) fn platform_default_backend_flavor() -> Option<backend::BinaryFlavor> {
    None
}

pub(super) fn startup_model_display_name(model: &StartupModelPlan) -> String {
    let declared_ref = model.declared_ref.trim();
    if declared_ref.is_empty() {
        resolved_model_name(&model.resolved_path)
    } else {
        declared_ref.to_string()
    }
}

pub(super) async fn wait_for_dashboard_first_paint(
    first_paint_rx: tokio::sync::oneshot::Receiver<std::io::Result<()>>,
) {
    if let Some(message) = dashboard_first_paint_warning(
        tokio::time::timeout(DASHBOARD_FIRST_PAINT_TIMEOUT, first_paint_rx).await,
    ) {
        tracing::warn!("{message}");
    }
}

pub(super) fn dashboard_first_paint_warning(
    result: std::result::Result<
        std::result::Result<std::io::Result<()>, tokio::sync::oneshot::error::RecvError>,
        tokio::time::error::Elapsed,
    >,
) -> Option<String> {
    match result {
        Ok(Ok(Ok(()))) => None,
        Ok(Ok(Err(err))) => Some(format!("interactive dashboard first paint failed: {err}")),
        Ok(Err(_)) => Some(
            "interactive dashboard first paint channel closed before acknowledgement".to_string(),
        ),
        Err(_) => Some(
            "interactive dashboard first paint did not acknowledge before startup continued"
                .to_string(),
        ),
    }
}

#[cfg(test)]
pub(crate) fn assert_active_serve_path_spawn_gate_behavior() {
    let interactive_started = AtomicBool::new(false);

    let request = serve_path_interactive_spawn_request(true, &interactive_started, true)
        .expect("active serve path should request interactive startup before llama_ready");
    assert_eq!(request.prompt_mode, InitialPromptMode::Deferred);
    interactive::assert_deferred_initial_prompt_waits_for_runtime_ready();
    assert_eq!(
        interactive::interactive_entry_kind(Some(ConsoleSessionMode::InteractiveDashboard)),
        interactive::InteractiveEntryKind::Tui
    );
    assert_eq!(
        serve_path_interactive_spawn_request(true, &interactive_started, true),
        None,
        "the active serve path should only request interactive startup once"
    );
}

#[cfg(test)]
pub(crate) fn assert_interactive_handler_spawns_once_across_startup_callbacks() {
    let interactive_started = AtomicBool::new(false);

    let request = serve_path_interactive_spawn_request(true, &interactive_started, true)
        .expect("console bootstrap should claim the one-shot interactive spawn gate");
    assert_eq!(request.prompt_mode, InitialPromptMode::Deferred);

    assert_eq!(
        serve_path_interactive_spawn_request(true, &interactive_started, true),
        None,
        "later startup or election callbacks must not spawn a second interactive handler"
    );
    assert_eq!(
        serve_path_interactive_spawn_request(false, &interactive_started, true),
        None,
        "disabling the input handler later must not reopen the one-shot spawn gate"
    );
    assert!(
        interactive_started.load(Ordering::Acquire),
        "the console-bootstrap spawn should consume the one-shot gate permanently"
    );
}

#[cfg(test)]
pub(crate) fn assert_passive_path_immediate_spawn_behavior() {
    let request = passive_path_interactive_spawn_request(
        Some(ConsoleSessionMode::InteractiveDashboard),
        true,
    )
    .expect("passive/client pretty sessions should request interactive startup immediately");

    assert_eq!(request.prompt_mode, InitialPromptMode::Immediate);
    assert_eq!(
        interactive::interactive_entry_kind(Some(ConsoleSessionMode::InteractiveDashboard)),
        interactive::InteractiveEntryKind::Tui
    );
    assert_eq!(
        passive_path_interactive_spawn_request(
            Some(ConsoleSessionMode::InteractiveDashboard),
            false
        ),
        None,
        "stdin must still be a TTY before passive/client startup requests interactive input"
    );
}

#[cfg(test)]
pub(crate) fn assert_quitting_during_startup_cancels_without_late_ready_render() {
    let reporter = StartupReadyReporter::new(
        &["Qwen3-8B-Q4_K_M".to_string()],
        "http://127.0.0.1:9337".to_string(),
        Some("http://127.0.0.1:3131".to_string()),
        9337,
        Some(3131),
    );
    reporter.mark_shutdown_requested();
    assert!(
        reporter
            .mark_ready_and_build_event(0, "Qwen3-8B-Q4_K_M")
            .is_none(),
        "startup shutdown should cancel any late RuntimeReady emission"
    );
}

#[cfg(test)]
pub(crate) fn assert_startup_ready_reporter_waits_for_rust_owned_model_ready_edges() {
    let models = vec!["model-a".to_string(), "model-b".to_string()];
    let reporter = StartupReadyReporter::new(
        &models,
        "http://127.0.0.1:9337".to_string(),
        Some("http://127.0.0.1:3131".to_string()),
        9337,
        Some(3131),
    );

    assert!(
        reporter.mark_ready_and_build_event(0, "model-a").is_none(),
        "one model-ready edge must not replace the remaining Rust-owned readiness edges"
    );
    assert!(
        reporter.mark_ready_and_build_event(0, "model-a").is_none(),
        "a repeated edge for one startup slot must not mark a different slot ready"
    );
    assert!(
        matches!(
            reporter.mark_ready_and_build_event(1, "model-b"),
            Some(OutputEvent::RuntimeReady { .. })
        ),
        "RuntimeReady should appear only after every startup model hits the Rust-owned ready path"
    );
}

#[cfg(test)]
pub(crate) fn assert_startup_launch_plan_describes_planned_runtime_before_process_start() {
    let startup_models = startup_model_plan_fixture();

    let plan = startup_launch_plan(
        &startup_models,
        "Fallback-Model",
        9337,
        Some(3131),
        false,
        Some(4),
        None,
    );

    assert_llama_process_row(&plan, "llama-server unsloth/Model-A-GGUF:Q4_K_M");
    assert_llama_process_row(&plan, "llama-server Model-B");
    assert_eq!(plan.llama_process_rows.len(), 2);
    assert_webserver_plan_row(&plan, "API", 9337);
    assert_webserver_plan_row(&plan, "Console", 3131);

    let headless_plan = startup_launch_plan(
        &startup_models,
        "Fallback-Model",
        9337,
        Some(3131),
        true,
        Some(4),
        None,
    );
    assert_headless_launch_plan(&headless_plan);
    assert_loaded_model_plan_row(
        &plan,
        "unsloth/Model-A-GGUF:Q4_K_M",
        "primary",
        Some("GPU0"),
        2,
    );
    assert_loaded_model_plan_row(&plan, "Model-B", "model", Some("CUDA1"), 4);

    let fallback_plan =
        startup_launch_plan(&[], "Auto-Assigned-Model", 9337, None, false, Some(8), None);
    assert_llama_process_row(&fallback_plan, "llama-server Auto-Assigned-Model");
    assert_loaded_model_plan_row(&fallback_plan, "Auto-Assigned-Model", "primary", None, 8);
}

#[cfg(test)]
pub(super) fn startup_model_plan_fixture() -> Vec<StartupModelPlan> {
    vec![
        StartupModelPlan {
            declared_ref: "unsloth/Model-A-GGUF:Q4_K_M".to_string(),
            config_model_id: None,
            resolved_path: PathBuf::from("/tmp/Model-A-Q4_K_M.gguf"),
            preindexed_split_package: None,
            mmproj_path: None,
            ctx_size: Some(8192),
            gpu_id: Some("GPU0".to_string()),
            pinned_gpu: None,
            parallel: Some(2),
            cache_type_k: None,
            cache_type_v: None,
            n_batch: None,
            n_ubatch: None,
            flash_attention: FlashAttentionType::Auto,
            local_source_required: false,
            profile: String::new(),
        },
        StartupModelPlan {
            declared_ref: "Model-B".to_string(),
            config_model_id: None,
            resolved_path: PathBuf::from("/tmp/Model-B.gguf"),
            preindexed_split_package: None,
            mmproj_path: None,
            ctx_size: Some(4096),
            gpu_id: None,
            pinned_gpu: Some(StartupPinnedGpuTarget {
                index: 1,
                stable_id: "gpu-b".to_string(),
                backend_device: "CUDA1".to_string(),
                vram_bytes: 24 * 1024 * 1024 * 1024,
                reserved_bytes: None,
            }),
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            n_batch: None,
            n_ubatch: None,
            flash_attention: FlashAttentionType::Auto,
            local_source_required: false,
            profile: String::new(),
        },
    ]
}

#[cfg(test)]
pub(super) fn assert_llama_process_row(plan: &DashboardLaunchPlan, name: &str) {
    assert!(
        plan.llama_process_rows.iter().any(|row| {
            row.name == name && row.status == RuntimeStatus::Loading && row.port == 0
        })
    );
}

#[cfg(test)]
pub(super) fn assert_webserver_plan_row(plan: &DashboardLaunchPlan, label: &str, port: u16) {
    let row = plan
        .webserver_rows
        .iter()
        .find(|row| row.label == label)
        .unwrap_or_else(|| panic!("launch plan should include planned {label} row"));
    assert_eq!(row.status, RuntimeStatus::NotReady);
    assert_eq!(row.port, port);
}

#[cfg(test)]
pub(super) fn assert_headless_launch_plan(plan: &DashboardLaunchPlan) {
    assert!(
        plan.webserver_rows.iter().any(|row| row.label == "API"),
        "headless launch plan should keep the API row"
    );
    assert!(
        plan.webserver_rows.iter().all(|row| row.label != "Console"),
        "headless launch plan should not seed a stale Console row"
    );
}

#[cfg(test)]
pub(super) fn assert_loaded_model_plan_row(
    plan: &DashboardLaunchPlan,
    name: &str,
    role: &str,
    device: Option<&str>,
    slots: usize,
) {
    let row = plan
        .loaded_model_rows
        .iter()
        .find(|row| row.name == name)
        .unwrap_or_else(|| panic!("launch plan should include loaded-model row for {name}"));
    assert_eq!(row.role.as_deref(), Some(role));
    assert_eq!(row.status, RuntimeStatus::Loading);
    assert_eq!(row.device.as_deref(), device);
    assert_eq!(row.slots, Some(slots));
    assert_eq!(row.file_size_gb, None);
}

#[test]
pub(super) fn startup_launch_plan_uses_metal_device_fallback_for_unpinned_model() {
    let startup_models = vec![StartupModelPlan {
        declared_ref: "Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_k_m".to_string(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(4096),
        gpu_id: None,
        pinned_gpu: None,
        parallel: Some(4),
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];

    let plan = startup_launch_plan(
        &startup_models,
        "Fallback-Model",
        9337,
        None,
        false,
        Some(4),
        startup_default_backend_device(Some(backend::BinaryFlavor::Metal)),
    );
    let model = plan
        .loaded_model_rows
        .iter()
        .find(|row| row.name == startup_models[0].declared_ref)
        .expect("launch plan should include unpinned local model row");

    assert_eq!(model.device.as_deref(), Some("MTL0"));
}

#[test]
pub(super) fn serve_path_builtin_endpoint_ready_events_cover_api_and_console() {
    let events = serve_path_builtin_endpoint_ready_events(
        "http://127.0.0.1:9337".to_string(),
        Some("http://127.0.0.1:3131".to_string()),
        false,
    );
    assert_eq!(events.len(), 2);
    assert!(matches!(
        &events[0],
        OutputEvent::ApiReady { url } if url == "http://127.0.0.1:9337"
    ));
    assert!(matches!(
        &events[1],
        OutputEvent::WebserverReady { url } if url == "http://127.0.0.1:3131"
    ));

    let headless_events = serve_path_builtin_endpoint_ready_events(
        "http://127.0.0.1:9444".to_string(),
        Some("http://127.0.0.1:3222".to_string()),
        true,
    );
    assert_eq!(headless_events.len(), 1);
    assert!(matches!(
        &headless_events[0],
        OutputEvent::ApiReady { url } if url == "http://127.0.0.1:9444"
    ));
}

#[cfg(test)]
#[tokio::test]
pub(super) async fn listener_http_url_uses_bound_ephemeral_addr() {
    let listener = bind_runtime_tcp_listener(0, false, "test listener")
        .await
        .expect("ephemeral listener should bind");
    let addr = listener
        .local_addr()
        .expect("bound listener should expose local address");

    let url = listener_http_url(&listener, 0, "test listener");

    assert_eq!(url, socket_addr_http_url(addr));
    assert_ne!(url, "http://localhost:0");
    assert!(!url.ends_with(":0"));
}

#[cfg(test)]
#[tokio::test]
pub(super) async fn startup_ready_reporter_uses_bound_urls_for_runtime_ready() {
    let api_listener = bind_runtime_tcp_listener(0, false, "test API listener")
        .await
        .expect("ephemeral API listener should bind");
    let console_listener = bind_runtime_tcp_listener(0, false, "test console listener")
        .await
        .expect("ephemeral console listener should bind");
    let (api_url, api_port) = listener_http_endpoint(&api_listener, 0, "test API listener");
    let (console_url, console_port) =
        listener_http_endpoint(&console_listener, 0, "test console listener");
    let models = vec!["model-a".to_string()];
    let reporter = StartupReadyReporter::new(
        &models,
        api_url.clone(),
        Some(console_url.clone()),
        api_port,
        Some(console_port),
    );

    let Some(OutputEvent::RuntimeReady {
        api_url: reported_api_url,
        console_url: reported_console_url,
        api_port: reported_api_port,
        console_port: reported_console_port,
        ..
    }) = reporter.mark_ready_and_build_event(0, "model-a")
    else {
        panic!("reporter should emit RuntimeReady when the model is ready");
    };

    assert_eq!(reported_api_url, api_url);
    assert_eq!(reported_console_url.as_deref(), Some(console_url.as_str()));
    assert_eq!(reported_api_port, api_port);
    assert_eq!(reported_console_port, Some(console_port));
    assert_ne!(reported_api_url, "http://localhost:0");
    assert_ne!(reported_console_url.as_deref(), Some("http://localhost:0"));
}

#[test]
pub(super) fn startup_ready_reporter_waits_for_rust_owned_model_ready_edges() {
    assert_startup_ready_reporter_waits_for_rust_owned_model_ready_edges();
}

#[test]
fn startup_ready_commands_use_the_primary_served_model_in_either_load_order() {
    let primary_id = "local-gguf/sha256-f90897d3a4d7e185";
    for primary_first in [true, false] {
        let models = vec![
            "/models/primary.gguf".to_string(),
            "other-model".to_string(),
        ];
        let reporter = StartupReadyReporter::new(
            &models,
            "http://127.0.0.1:9337".to_string(),
            None,
            9337,
            None,
        );
        let (first, last) = if primary_first { (0, 1) } else { (1, 0) };
        let served_models = [primary_id, "other-served-model"];
        assert!(
            reporter
                .mark_ready_and_build_event(first, served_models[first])
                .is_none()
        );
        let Some(OutputEvent::RuntimeReady {
            pi_command,
            goose_command,
            ..
        }) = reporter
            .clone()
            .mark_ready_and_build_event(last, served_models[last])
        else {
            panic!("both loaded models should make the runtime ready");
        };
        assert_eq!(
            pi_command.unwrap(),
            format!("mesh-llm pi --host 127.0.0.1:9337 --model '{primary_id}'")
        );
        assert_eq!(
            goose_command.unwrap(),
            format!(
                "GOOSE_PROVIDER=openai OPENAI_HOST=http://127.0.0.1:9337 OPENAI_API_KEY=mesh GOOSE_MODEL={primary_id} goose session"
            )
        );
        assert!(
            reporter
                .mark_ready_and_build_event(last, served_models[last])
                .is_none()
        );
    }
}

#[cfg(test)]
#[test]
pub(super) fn dashboard_lanes_prefer_sparse_slot_ids() {
    let snapshots_by_instance = BTreeMap::new();
    let mut snapshots_by_model = BTreeMap::new();
    let mut snapshot = crate::runtime_data::RuntimeLlamaRuntimeSnapshot::default();
    snapshot.items.slots = vec![
        crate::runtime_data::RuntimeLlamaSlotItem {
            index: 0,
            id: Some(20),
            id_task: None,
            n_ctx: None,
            is_processing: false,
        },
        crate::runtime_data::RuntimeLlamaSlotItem {
            index: 1,
            id: Some(10),
            id_task: None,
            n_ctx: None,
            is_processing: true,
        },
    ];
    snapshots_by_model.insert("model-a".to_string(), snapshot);
    let process = api::RuntimeProcessPayload {
        name: "model-a".to_string(),
        instance_id: None,
        profile: String::new(),
        backend: "skippy".to_string(),
        status: "ready".to_string(),
        port: 4001,
        pid: 1234,
        slots: 2,
        context_length: Some(8192),
    };

    let lanes = dashboard_lanes_for_process(&snapshots_by_instance, &snapshots_by_model, &process)
        .expect("snapshot with slots should produce dashboard lanes");

    assert_eq!(lanes.len(), 2);
    assert_eq!(lanes[0].index, 10);
    assert!(lanes[0].active);
    assert_eq!(lanes[1].index, 20);
    assert!(!lanes[1].active);
}

#[cfg(test)]
#[test]
pub(super) fn dashboard_lanes_fall_back_to_slot_index_when_id_is_missing() {
    let snapshots_by_instance = BTreeMap::new();
    let mut snapshots_by_model = BTreeMap::new();
    let mut snapshot = crate::runtime_data::RuntimeLlamaRuntimeSnapshot::default();
    snapshot.items.slots = vec![crate::runtime_data::RuntimeLlamaSlotItem {
        index: 7,
        id: None,
        id_task: None,
        n_ctx: None,
        is_processing: true,
    }];
    snapshots_by_model.insert("model-a".to_string(), snapshot);
    let process = api::RuntimeProcessPayload {
        name: "model-a".to_string(),
        instance_id: None,
        profile: String::new(),
        backend: "skippy".to_string(),
        status: "ready".to_string(),
        port: 4001,
        pid: 1234,
        slots: 1,
        context_length: Some(8192),
    };

    let lanes = dashboard_lanes_for_process(&snapshots_by_instance, &snapshots_by_model, &process)
        .expect("snapshot with slots should produce dashboard lanes");

    assert_eq!(lanes.len(), 1);
    assert_eq!(lanes[0].index, 7);
    assert!(lanes[0].active);
}

#[cfg(test)]
#[test]
pub(super) fn dashboard_lanes_prefer_instance_snapshot_for_duplicate_models() {
    let mut snapshots_by_instance = BTreeMap::new();
    let snapshots_by_model = BTreeMap::new();
    let mut first_snapshot = crate::runtime_data::RuntimeLlamaRuntimeSnapshot::default();
    first_snapshot.items.slots = vec![crate::runtime_data::RuntimeLlamaSlotItem {
        index: 0,
        id: Some(1),
        id_task: None,
        n_ctx: None,
        is_processing: false,
    }];
    let mut second_snapshot = crate::runtime_data::RuntimeLlamaRuntimeSnapshot::default();
    second_snapshot.items.slots = vec![crate::runtime_data::RuntimeLlamaSlotItem {
        index: 0,
        id: Some(2),
        id_task: None,
        n_ctx: None,
        is_processing: true,
    }];
    snapshots_by_instance.insert("runtime-1".to_string(), first_snapshot);
    snapshots_by_instance.insert("runtime-2".to_string(), second_snapshot);

    let process = api::RuntimeProcessPayload {
        name: "model-a".to_string(),
        instance_id: Some("runtime-2".to_string()),
        profile: String::new(),
        backend: "skippy".to_string(),
        status: "ready".to_string(),
        port: 4002,
        pid: 1235,
        slots: 1,
        context_length: Some(8192),
    };

    let lanes = dashboard_lanes_for_process(&snapshots_by_instance, &snapshots_by_model, &process)
        .expect("instance snapshot should produce dashboard lanes");

    assert_eq!(lanes.len(), 1);
    assert_eq!(lanes[0].index, 2);
    assert!(lanes[0].active);
}

pub(super) fn should_start_bootstrap_proxy(
    options: &RuntimeOptions,
    auto_join_candidates: &[(String, Option<String>)],
) -> bool {
    !options.join.is_empty() || !auto_join_candidates.is_empty()
}

pub(super) fn start_run_auto_bootstrap_proxy(
    options: &RuntimeOptions,
    node: &mesh::Node,
    api_port: u16,
    affinity_router: &affinity::AffinityRouter,
    auto_join_candidates: &[(String, Option<String>)],
) -> Option<BootstrapProxyStopTx> {
    if !should_start_bootstrap_proxy(options, auto_join_candidates) {
        return None;
    }

    let (stop_tx, stop_rx) =
        tokio::sync::mpsc::channel::<tokio::sync::oneshot::Sender<tokio::net::TcpListener>>(1);
    let boot_node = node.clone();
    let boot_port = api_port;
    let boot_affinity = affinity_router.clone();
    let listen_all = options.listen_all;
    tokio::spawn(async move {
        bootstrap_proxy(boot_node, boot_port, stop_rx, listen_all, boot_affinity).await;
    });
    Some(stop_tx)
}

#[expect(
    dead_code,
    reason = "owned by the retained passive runtime compatibility lane"
)]
pub(super) struct PassiveConsoleRuntime {
    pub(super) control_rx: tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>,
    pub(super) console_server_handle: Option<tokio::task::JoinHandle<()>>,
}

#[expect(
    dead_code,
    reason = "owned by the retained passive runtime compatibility lane"
)]
pub(super) struct PassiveConsoleSetupContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) node: &'a mesh::Node,
    pub(super) is_client: bool,
    pub(super) plugin_manager: &'a plugin::PluginManager,
    pub(super) affinity_router: &'a affinity::AffinityRouter,
    pub(super) local_port: u16,
    pub(super) cport: u16,
    pub(super) embedded_control_rx:
        Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
}

pub(super) struct RunAutoConsoleStateContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) node: &'a mesh::Node,
    pub(super) console_enabled: bool,
    pub(super) model_name: &'a str,
    pub(super) model_path: &'a Path,
    pub(super) api_port: u16,
    pub(super) plugin_manager: &'a plugin::PluginManager,
    pub(super) affinity_router: &'a affinity::AffinityRouter,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) owner_key_path: &'a Option<PathBuf>,
}

pub(super) struct RunAutoAdditionalModelsContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) config: &'a plugin::MeshConfig,
    pub(super) node: &'a mesh::Node,
    pub(super) tunnel_mgr: &'a tunnel::Manager,
    pub(super) startup_models: &'a [StartupModelPlan],
    pub(super) primary_model_name: &'a str,
    pub(super) target_tx: &'a Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    pub(super) managed_models: &'a mut HashMap<String, ManagedModelController>,
    pub(super) next_runtime_instance_sequence: &'a mut u64,
    pub(super) dashboard_processes: &'a Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) dashboard_context_usage: &'a DashboardContextUsage,
    pub(super) runtime_instance_registry: &'a RuntimeInstanceRegistry,
    pub(super) runtime_capacity_ledger: &'a RuntimeCapacityLedger,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) startup_ready_reporter: &'a StartupReadyReporter,
    pub(super) startup_load_gate: &'a Arc<tokio::sync::Mutex<()>>,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) runtime_event_tx: &'a tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    pub(super) survey_telemetry: &'a survey::SurveyTelemetry,
    pub(super) skippy_telemetry: &'a skippy::SkippyTelemetryOptions,
    pub(super) openai_guardrail_policy: &'a OpenAiGuardrailPolicyHandle,
}

pub(super) struct RunAutoServingSurfaceContext<'a> {
    pub(super) options: &'a RuntimeOptions,
    pub(super) node: &'a mesh::Node,
    pub(super) api_port: u16,
    pub(super) console_port: Option<u16>,
    pub(super) is_client: bool,
    pub(super) target_rx: &'a tokio::sync::watch::Receiver<election::ModelTargets>,
    pub(super) control_tx: &'a tokio::sync::mpsc::UnboundedSender<api::RuntimeControlRequest>,
    pub(super) affinity_router: &'a affinity::AffinityRouter,
    pub(super) bootstrap_listener_tx: Option<BootstrapProxyStopTx>,
    pub(super) input_handler_enabled: bool,
    pub(super) interactive_started: &'a Arc<AtomicBool>,
    pub(super) console_state: Option<&'a api::MeshApi>,
    pub(super) model_name_for_console: &'a str,
}

pub(super) struct RunAutoServingSurface {
    pub(super) api_proxy_handle: tokio::task::JoinHandle<()>,
    pub(super) console_server_handle: Option<tokio::task::JoinHandle<()>>,
    pub(super) api_ready_url: String,
    pub(super) ready_console_url: Option<String>,
    pub(super) ready_api_port: u16,
    pub(super) ready_console_port: Option<u16>,
}

pub(super) async fn setup_run_auto_console_state(
    ctx: RunAutoConsoleStateContext<'_>,
) -> Result<Option<api::MeshApi>> {
    if !ctx.console_enabled {
        return Ok(None);
    }
    let model_size_bytes = election::total_model_bytes(ctx.model_path);
    let runtime_data_collector = ctx.node.runtime_data_collector();
    let runtime_data_producer =
        runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
    let console_state = api::MeshApi::new(api::MeshApiConfig {
        node: ctx.node.clone(),
        model_name: ctx.model_name.to_string(),
        api_port: ctx.api_port,
        model_size_bytes,
        owner_key_path: ctx.owner_key_path.clone(),
        plugin_manager: ctx.plugin_manager.clone(),
        affinity_router: ctx.affinity_router.clone(),
        runtime_data_collector,
        runtime_data_producer,
    });
    console_state.set_primary_backend("skippy".into()).await;
    console_state
        .set_runtime_control(ctx.control_tx.clone())
        .await;
    console_state
        .set_control_bootstrap(api::ControlBootstrapPayload::from_control_endpoint(
            ctx.node.control_endpoint().await,
        ))
        .await;
    console_state
        .set_nostr_relays(nostr_relays(&ctx.options.nostr_relay))
        .await;
    console_state
        .set_mesh_discovery_mode(ctx.options.mesh_discovery_mode)
        .await;
    console_state
        .set_nostr_discovery(ctx.options.nostr_discovery)
        .await;
    if let Some(draft) = &ctx.options.draft {
        let dn = draft
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        console_state.set_draft_name(dn).await;
    }
    console_state
        .set_mesh_publication_metadata(
            ctx.options.mesh_name.clone(),
            ctx.options.region.clone(),
            ctx.options.max_clients,
        )
        .await;
    Ok(Some(console_state))
}

#[expect(
    dead_code,
    reason = "bridges the retained advertised-model and passive runtime compatibility lanes"
)]
pub(super) async fn run_auto_model_path_or_shutdown(
    ctx: &mut RunAutoModelSelectionContext<'_>,
) -> Result<Option<PathBuf>> {
    match select_run_auto_model_path(ctx).await? {
        RunAutoModelSelection::Model(model) => Ok(Some(model)),
        RunAutoModelSelection::Shutdown => Ok(None),
    }
}

pub(super) async fn spawn_run_auto_discovery_publisher(
    options: &RuntimeOptions,
    node: &mesh::Node,
    console_state: Option<&api::MeshApi>,
) -> Option<tokio::task::JoinHandle<()>> {
    if options.publish {
        return match options.mesh_discovery_mode {
            mesh_discovery::MeshDiscoveryMode::Nostr => {
                spawn_run_auto_nostr_publisher(options, node, console_state).await
            }
            mesh_discovery::MeshDiscoveryMode::Mdns => {
                spawn_run_auto_mdns_publisher(options, node, console_state)
            }
        };
    }
    if options.mesh_discovery_mode == mesh_discovery::MeshDiscoveryMode::Nostr
        && (options.auto || options.discover.is_some())
    {
        return Some(spawn_run_auto_nostr_watchdog(options, node, console_state));
    }
    None
}

pub(super) async fn spawn_run_auto_nostr_publisher(
    options: &RuntimeOptions,
    node: &mesh::Node,
    console_state: Option<&api::MeshApi>,
) -> Option<tokio::task::JoinHandle<()>> {
    match nostr::load_or_create_keys() {
        Ok(nostr_keys) => {
            let relays = nostr_relays(&options.nostr_relay);
            let pub_node = node.clone();
            let pub_name = options.mesh_name.clone();
            let pub_region = options.region.clone();
            let pub_max_clients = options.max_clients;
            let (status_tx, status_rx) = tokio::sync::watch::channel(None);
            if let Some(cs) = console_state {
                bridge_publication_state(cs.clone(), status_rx);
            }
            Some(tokio::spawn(Box::pin(nostr::publish_loop(
                pub_node,
                nostr_keys,
                nostr::PublishLoopConfig {
                    relays,
                    name: pub_name,
                    region: pub_region,
                    max_clients: pub_max_clients,
                    interval_secs: 60,
                    status_tx: Some(status_tx),
                },
            ))))
        }
        Err(e) => {
            let _ = emit_event(OutputEvent::Warning {
                message: format!(
                    "Publishing to Nostr failed: {e}. Mesh is running privately — add --publish after fixing the issue to make discoverable."
                ),
                context: options
                    .mesh_name
                    .as_ref()
                    .map(|mesh_name| format!("mesh={mesh_name}")),
            });
            tracing::warn!("Nostr publish failed: {e}");
            if let Some(cs) = console_state {
                cs.set_publication_state(api::PublicationState::PublishFailed)
                    .await;
            }
            None
        }
    }
}

pub(super) fn spawn_run_auto_mdns_publisher(
    options: &RuntimeOptions,
    node: &mesh::Node,
    console_state: Option<&api::MeshApi>,
) -> Option<tokio::task::JoinHandle<()>> {
    let pub_node = node.clone();
    let pub_name = options.mesh_name.clone();
    let pub_region = options.region.clone();
    let pub_max_clients = options.max_clients;
    let pub_api_port = options.console;
    let pub_details_reachable = options.listen_all;
    let (status_tx, status_rx) = tokio::sync::watch::channel(None);
    if let Some(cs) = console_state {
        bridge_publication_state(cs.clone(), status_rx);
    }
    Some(tokio::spawn(Box::pin(mesh_discovery::publish_lan_loop(
        pub_node,
        mesh_discovery::LanPublishConfig {
            name: pub_name,
            region: pub_region,
            max_clients: pub_max_clients,
            api_port: pub_api_port,
            details_reachable: pub_details_reachable,
            interval_secs: 60,
            status_tx: Some(status_tx),
        },
    ))))
}

pub(super) fn spawn_run_auto_nostr_watchdog(
    options: &RuntimeOptions,
    node: &mesh::Node,
    console_state: Option<&api::MeshApi>,
) -> tokio::task::JoinHandle<()> {
    let relays = nostr_relays(&options.nostr_relay);
    let wd_node = node.clone();
    let wd_name = options.mesh_name.clone();
    let wd_region = options.region.clone();
    let watchdog_status_rx = console_state.map(|cs| {
        let (status_tx, status_rx) = tokio::sync::watch::channel(None);
        bridge_publication_state(cs.clone(), status_rx);
        status_tx
    });
    tokio::spawn(async move {
        nostr::publish_watchdog(wd_node, relays, wd_name, wd_region, 120, watchdog_status_rx).await;
    })
}

pub(super) async fn spawn_run_auto_additional_model_tasks(ctx: RunAutoAdditionalModelsContext<'_>) {
    if ctx.startup_models.len() <= 1 {
        return;
    }

    let all_names: Vec<String> = ctx
        .startup_models
        .iter()
        .map(|model| model.declared_ref.clone())
        .collect();
    let _ = emit_event(OutputEvent::MultiModelMode {
        count: all_names.len(),
        models: all_names.clone(),
    });
    ctx.node.set_models(all_names).await;
    ctx.node.regossip().await;

    for (readiness_index, extra_model) in ctx.startup_models.iter().enumerate().skip(1) {
        let extra_name = extra_model.declared_ref.clone();
        let (extra_stop_tx, extra_stop_rx) = tokio::sync::watch::channel(false);
        let extra_instance_id = next_runtime_instance_id(ctx.next_runtime_instance_sequence);
        let extra_lifecycle = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
            InstanceLifecycleState::Planned,
            32,
        )));
        let extra_lifecycle_port = Arc::new(std::sync::atomic::AtomicU16::new(0));
        let extra_task = tokio::spawn(Box::pin(startup_local_model_loop(StartupLocalModelTask {
            node: ctx.node.clone(),
            config: ctx.config.clone(),
            tunnel_mgr: ctx.tunnel_mgr.clone(),
            target_tx: ctx.target_tx.clone(),
            model_path: extra_model.resolved_path.clone(),
            model_ref: extra_model.declared_ref.clone(),
            preindexed_split_package: extra_model.preindexed_split_package.clone(),
            config_model_id: extra_model.config_model_id.clone(),
            readiness_index,
            profile: extra_model.profile.clone(),
            model_name: extra_name.clone(),
            instance_id: extra_instance_id.clone(),
            primary_model_name: ctx.primary_model_name.to_string(),
            mmproj_path: extra_model.mmproj_path.clone(),
            ctx_size: extra_model.ctx_size,
            pinned_gpu: extra_model.pinned_gpu.clone(),
            device_override: super::startup_models::startup_device_override(
                extra_model.gpu_id.as_deref(),
            ),
            runtime_capacity_ledger: ctx.runtime_capacity_ledger.clone(),
            cache_type_k: extra_model.cache_type_k.clone(),
            cache_type_v: extra_model.cache_type_v.clone(),
            n_batch: extra_model.n_batch,
            n_ubatch: extra_model.n_ubatch,
            flash_attention: extra_model.flash_attention,
            parallel_override: super::startup_models::resolve_model_parallel_override(
                extra_model.parallel,
                &ctx.config.gpu,
            ),
            local_source_required: extra_model.local_source_required,
            split_topology_lock: ctx.options.split_topology_lock.clone(),
            resource_planning_profile: runtime_resource_planning_profile(ctx.options),
            openai_guardrail_policy: ctx.openai_guardrail_policy.clone(),
            split: ctx.options.split,
            skippy_telemetry: ctx.skippy_telemetry.clone(),
            survey_telemetry: ctx.survey_telemetry.clone(),
            survey_launch_kind: survey::SurveyLaunchKind::MultiModel,
            stop_rx: extra_stop_rx,
            dashboard_processes: ctx.dashboard_processes.clone(),
            dashboard_context_usage: ctx.dashboard_context_usage.clone(),
            runtime_instance_registry: ctx.runtime_instance_registry.clone(),
            console_state: ctx.console_state.cloned(),
            api_port: ctx.options.port,
            startup_ready_reporter: ctx.startup_ready_reporter.clone(),
            runtime_event_tx: ctx.runtime_event_tx.clone(),
            startup_load_gate: ctx.startup_load_gate.clone(),
            input_handler_enabled: false,
            interactive_started: Arc::new(AtomicBool::new(true)),
            interactive_control_tx: ctx.control_tx.clone(),
            interactive_console_state: None,
            lifecycle: extra_lifecycle.clone(),
            lifecycle_port: extra_lifecycle_port.clone(),
        })));
        ctx.managed_models.insert(
            extra_instance_id,
            ManagedModelController {
                model_name: extra_name,
                profile: extra_model.profile.clone(),
                stop_tx: extra_stop_tx,
                task: extra_task,
                lifecycle: extra_lifecycle,
                port: extra_lifecycle_port,
            },
        );
    }
}

pub(super) async fn setup_run_auto_serving_surface(
    ctx: RunAutoServingSurfaceContext<'_>,
) -> Result<RunAutoServingSurface> {
    wait_for_run_auto_first_paint(&ctx).await;
    let api_listener =
        run_auto_api_listener(ctx.options, ctx.api_port, ctx.bootstrap_listener_tx).await?;
    let console_listener =
        run_auto_console_listener(ctx.options, ctx.console_port, ctx.console_state).await?;
    let (api_ready_url, ready_api_port) =
        listener_http_endpoint(&api_listener, ctx.api_port, "OpenAI-compatible API");
    let (ready_console_url, ready_console_port) =
        run_auto_ready_console_endpoint(&console_listener);
    emit_run_auto_builtin_endpoint_ready(ctx.options, &api_ready_url, ready_console_url.as_ref());
    let api_proxy_handle = spawn_run_auto_api_proxy(
        ctx.options,
        ctx.node,
        ctx.api_port,
        api_listener,
        ctx.target_rx,
        ctx.affinity_router,
    );
    let console_server_handle = spawn_run_auto_console_server(
        ctx.options,
        ctx.target_rx,
        console_listener,
        ctx.console_state,
        ctx.model_name_for_console,
    );
    spawn_run_auto_local_instance_scanner(ctx.is_client, ctx.console_state).await;
    Ok(RunAutoServingSurface {
        api_proxy_handle,
        console_server_handle,
        api_ready_url,
        ready_console_url,
        ready_api_port,
        ready_console_port,
    })
}

pub(super) async fn wait_for_run_auto_first_paint(ctx: &RunAutoServingSurfaceContext<'_>) {
    let Some(request) = serve_path_interactive_spawn_request(
        ctx.input_handler_enabled,
        ctx.interactive_started.as_ref(),
        std::io::stdin().is_terminal(),
    ) else {
        return;
    };
    let Some(cs) = ctx.console_state.cloned() else {
        return;
    };
    let (first_paint_tx, first_paint_rx) = tokio::sync::oneshot::channel();
    let Some(sink) = output_sink() else {
        return;
    };
    interactive::spawn_handler_with_first_paint_ack(
        ctx.control_tx.clone(),
        cs,
        sink,
        request.prompt_mode,
        Some(first_paint_tx),
    );
    wait_for_dashboard_first_paint(first_paint_rx).await;
}

pub(super) async fn run_auto_api_listener(
    options: &RuntimeOptions,
    api_port: u16,
    bootstrap_listener_tx: Option<BootstrapProxyStopTx>,
) -> Result<tokio::net::TcpListener> {
    if let Some(tx) = bootstrap_listener_tx {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let _ = tx.send(resp_tx).await;
        return resp_rx
            .await
            .context("bootstrap API listener handoff was cancelled");
    }
    bind_runtime_tcp_listener(api_port, options.listen_all, "OpenAI-compatible API").await
}

pub(super) async fn run_auto_console_listener(
    options: &RuntimeOptions,
    console_port: Option<u16>,
    console_state: Option<&api::MeshApi>,
) -> Result<Option<(u16, tokio::net::TcpListener)>> {
    match (console_port, console_state) {
        (Some(cport), Some(_)) => Ok(Some((
            cport,
            bind_runtime_tcp_listener(cport, options.listen_all, "Web console").await?,
        ))),
        _ => Ok(None),
    }
}

pub(super) fn run_auto_ready_console_endpoint(
    console_listener: &Option<(u16, tokio::net::TcpListener)>,
) -> (Option<String>, Option<u16>) {
    let ready_console_endpoint = console_listener
        .as_ref()
        .map(|(port, listener)| listener_http_endpoint(listener, *port, "Web console"));
    (
        ready_console_endpoint.as_ref().map(|(url, _)| url.clone()),
        ready_console_endpoint.map(|(_, port)| port),
    )
}

pub(super) fn emit_run_auto_builtin_endpoint_ready(
    options: &RuntimeOptions,
    api_ready_url: &str,
    ready_console_url: Option<&String>,
) {
    for event in serve_path_builtin_endpoint_ready_events(
        api_ready_url.to_string(),
        ready_console_url.cloned(),
        options.headless,
    ) {
        let _ = emit_event(event);
    }
}

pub(super) fn spawn_run_auto_api_proxy(
    options: &RuntimeOptions,
    node: &mesh::Node,
    api_port: u16,
    api_listener: tokio::net::TcpListener,
    target_rx: &tokio::sync::watch::Receiver<election::ModelTargets>,
    affinity_router: &affinity::AffinityRouter,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(Box::pin(api_proxy(
        node.clone(),
        api_port,
        target_rx.clone(),
        Some(api_listener),
        options.listen_all,
        affinity_router.clone(),
    )))
}

pub(super) fn spawn_run_auto_console_server(
    options: &RuntimeOptions,
    target_rx: &tokio::sync::watch::Receiver<election::ModelTargets>,
    console_listener: Option<(u16, tokio::net::TcpListener)>,
    console_state: Option<&api::MeshApi>,
    model_name_for_console: &str,
) -> Option<tokio::task::JoinHandle<()>> {
    let ((cport, listener), cs) = (console_listener?, console_state.cloned()?);
    let cs2 = cs.clone();
    let console_rx = target_rx.clone();
    let mn = model_name_for_console.to_string();
    let listen_all = options.listen_all;
    let headless = options.headless;
    Some(tokio::spawn(async move {
        let (adapted_tx, adapted_rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
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
        api::start_with_listener(cport, cs2, adapted_rx, listen_all, headless, Some(listener))
            .await;
    }))
}

pub(super) async fn spawn_run_auto_local_instance_scanner(
    is_client: bool,
    console_state: Option<&api::MeshApi>,
) {
    if is_client {
        return;
    }
    let Some(cs) = console_state else {
        return;
    };
    let Ok(root) = crate::runtime::instance::runtime_root() else {
        return;
    };
    let runtime_data_producer = cs.runtime_data_producer().await;
    if let Ok(initial) =
        crate::runtime::instance::scan_local_instances(&root, std::process::id()).await
    {
        crate::runtime::instance::publish_local_instance_scan_results(
            &runtime_data_producer,
            initial,
        );
    }
    crate::runtime::instance::spawn_local_instance_scanner(
        root,
        std::process::id(),
        runtime_data_producer,
    );
}

#[expect(
    dead_code,
    reason = "owned by the retained passive runtime compatibility lane"
)]
pub(super) async fn setup_passive_console_runtime(
    ctx: PassiveConsoleSetupContext<'_>,
    console_listener: tokio::net::TcpListener,
) -> Result<PassiveConsoleRuntime> {
    let PassiveConsoleSetupContext {
        options,
        node,
        is_client,
        plugin_manager,
        affinity_router,
        local_port,
        cport,
        embedded_control_rx,
    } = ctx;
    let (control_tx, control_rx) =
        tokio::sync::mpsc::unbounded_channel::<api::RuntimeControlRequest>();
    spawn_embedded_runtime_control_forwarder(embedded_control_rx, control_tx.clone());
    let dashboard_processes = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let label = if is_client {
        "(client)".to_string()
    } else {
        "(standby)".to_string()
    };
    let runtime_data_collector = node.runtime_data_collector();
    let runtime_data_producer =
        runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
    let console_state = api::MeshApi::new(api::MeshApiConfig {
        node: node.clone(),
        model_name: label,
        api_port: local_port,
        model_size_bytes: 0,
        owner_key_path: resolve_runtime_owner_key_path(options)?,
        plugin_manager: plugin_manager.clone(),
        affinity_router: affinity_router.clone(),
        runtime_data_collector,
        runtime_data_producer,
    });
    console_state.set_runtime_control(control_tx.clone()).await;
    console_state
        .set_control_bootstrap(api::ControlBootstrapPayload::from_control_endpoint(
            node.control_endpoint().await,
        ))
        .await;
    console_state
        .set_nostr_relays(nostr_relays(&options.nostr_relay))
        .await;
    console_state
        .set_mesh_discovery_mode(options.mesh_discovery_mode)
        .await;
    console_state
        .set_nostr_discovery(options.nostr_discovery)
        .await;
    console_state
        .set_mesh_publication_metadata(
            options.mesh_name.clone(),
            options.region.clone(),
            options.max_clients,
        )
        .await;
    if is_client {
        console_state.set_client(true).await;
        if options.nostr_discovery {
            console_state
                .set_publication_state(api::PublicationState::Public)
                .await;
        }
    }
    console_state.update(false, true).await;
    let PassivePublicationSetup {
        state: passive_publication_state,
        status_rx: passive_publication_rx,
    } = setup_passive_publication(options, node, is_client).await;
    if let Some(state) = passive_publication_state {
        console_state.set_publication_state(state).await;
    }
    if let Some(status_rx) = passive_publication_rx {
        bridge_publication_state(console_state.clone(), status_rx);
    }
    let (_tx, rx) = tokio::sync::watch::channel(election::InferenceTarget::None);
    let la = options.listen_all;
    let headless = options.headless;
    let console_state_for_server = console_state.clone();
    let console_server_handle = Some(tokio::spawn(async move {
        api::start_with_listener(
            cport,
            console_state_for_server,
            rx,
            la,
            headless,
            Some(console_listener),
        )
        .await;
    }));
    if let Some(sink) = output_sink() {
        sink.register_dashboard_snapshot_provider(Arc::new(RuntimeDashboardSnapshotProvider::new(
            node.clone(),
            dashboard_processes,
            Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            Some(plugin_manager.clone()),
            local_port,
            Some(cport),
            headless,
        )));
    }
    if let Some(request) = passive_path_interactive_spawn_request(
        output_sink().and_then(|sink| sink.console_session_mode()),
        std::io::stdin().is_terminal(),
    ) && let Some(sink) = output_sink()
    {
        interactive::spawn_handler(control_tx.clone(), console_state, sink, request.prompt_mode);
    }
    Ok(PassiveConsoleRuntime {
        control_rx,
        console_server_handle,
    })
}

#[expect(
    dead_code,
    reason = "owned by the retained passive runtime compatibility lane"
)]
pub(super) async fn run_passive_listener_loop(
    listener: tokio::net::TcpListener,
    node: mesh::Node,
    affinity_router: affinity::AffinityRouter,
    plugin_manager: plugin::PluginManager,
    mut control_rx: tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>,
    mut console_server_handle: Option<tokio::task::JoinHandle<()>>,
    is_client: bool,
) -> Result<Option<String>> {
    let (promote_tx, mut promote_rx) = tokio::sync::mpsc::channel::<String>(1);
    maybe_spawn_passive_promotion_task(is_client, &node, promote_tx);

    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                let (tcp_stream, addr) = accept_result?;
                tcp_stream.set_nodelay(true)?;
                tracing::info!("Connection from {addr}");
                let node = node.clone();
                let affinity = affinity_router.clone();
                tokio::spawn(Box::pin(crate::network::proxy::handle_mesh_request(
                    node, tcp_stream.into(), true, affinity,
                )));
            }
            Some(model_name) = promote_rx.recv() => {
                return Ok(Some(model_name));
            }
            Some(cmd) = control_rx.recv() => {
                match cmd {
                    api::RuntimeControlRequest::Shutdown { source } => {
                        shutdown_passive_runtime(
                            &node,
                            &plugin_manager,
                            &mut console_server_handle,
                            source,
                        )
                        .await;
                        return Ok(None);
                    }
                    api::RuntimeControlRequest::Join { invite_token, resp } => {
                        let result = node.join_with_retry(&invite_token).await;
                        let _ = resp.send(result);
                    }
                    _ => {}
                }
            }
            signal = wait_shutdown_signal() => {
                shutdown_passive_runtime(&node, &plugin_manager, &mut console_server_handle, signal)
                    .await;
                return Ok(None);
            }
        }
    }
}

#[expect(
    dead_code,
    reason = "retained for compatibility with passive client and standby startup"
)]
pub(super) async fn run_passive(
    options: &RuntimeOptions,
    node: mesh::Node,
    is_client: bool,
    plugin_manager: plugin::PluginManager,
    api_listener: Option<tokio::net::TcpListener>,
    embedded_control_rx: Option<tokio::sync::mpsc::UnboundedReceiver<api::RuntimeControlRequest>>,
) -> Result<Option<String>> {
    let local_port = options.port;
    let affinity_router = affinity::AffinityRouter::new();
    node.set_display_name(node_display_name(options, &node))
        .await;

    // Wait briefly for gossip to propagate
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let served = node.models_being_served().await;
    if !served.is_empty() {
        let _ = emit_event(OutputEvent::Info {
            message: format!("Models available in mesh: {:?}", served),
            context: None,
        });
    }

    let listener = if let Some(listener) = api_listener {
        listener
    } else {
        bind_runtime_tcp_listener(local_port, options.listen_all, "OpenAI-compatible API")
            .await
            .with_context(|| format!("Failed to bind to port {local_port}"))?
    };
    let api_ready_url = listener_http_url(&listener, local_port, "OpenAI-compatible API");
    let cport = options.console;
    let console_listener =
        bind_runtime_tcp_listener(cport, options.listen_all, "Web console").await?;
    let console_ready_url = listener_http_url(&console_listener, cport, "Web console");
    emit_passive_ready_events(options, &node, is_client, api_ready_url, console_ready_url).await;

    let PassiveConsoleRuntime {
        control_rx,
        console_server_handle,
    } = setup_passive_console_runtime(
        PassiveConsoleSetupContext {
            options,
            node: &node,
            is_client,
            plugin_manager: &plugin_manager,
            affinity_router: &affinity_router,
            local_port,
            cport,
            embedded_control_rx,
        },
        console_listener,
    )
    .await?;

    run_passive_listener_loop(
        listener,
        node,
        affinity_router,
        plugin_manager,
        control_rx,
        console_server_handle,
        is_client,
    )
    .await
}

#[expect(
    dead_code,
    reason = "owned by the retained passive runtime compatibility lane and its ready-event tests"
)]
pub(super) async fn emit_passive_ready_events(
    options: &RuntimeOptions,
    node: &mesh::Node,
    is_client: bool,
    api_ready_url: String,
    console_ready_url: String,
) {
    let passive_mode_event = if is_client {
        OutputEvent::PassiveMode {
            role: "client".to_string(),
            status: RuntimeStatus::Ready,
            capacity_gb: None,
            models_on_disk: None,
            detail: Some("Client ready".to_string()),
        }
    } else {
        OutputEvent::PassiveMode {
            role: "standby".to_string(),
            status: RuntimeStatus::Ready,
            capacity_gb: Some(node.vram_bytes() as f64 / 1e9),
            models_on_disk: None,
            detail: Some("Standby ready".to_string()),
        }
    };
    let _ = emit_event(passive_mode_event);
    let _ = emit_event(OutputEvent::ApiReady { url: api_ready_url });
    if options.headless {
        let _ = emit_event(OutputEvent::Info {
            message: format!("Management API: {console_ready_url}"),
            context: None,
        });
    } else {
        let _ = emit_event(OutputEvent::WebserverReady {
            url: console_ready_url,
        });
    }
}

pub(super) fn detect_bin_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let dir = exe.parent().context("Binary has no parent directory")?;
    Ok(dir.to_path_buf())
}

/// Update ~/.pi/agent/models.json to include a "mesh" provider.
pub(super) fn update_pi_models_json(model_id: &str, port: u16) {
    let Some(home) = dirs::home_dir() else { return };
    let models_path = home.join(".pi/agent/models.json");

    let mut root: serde_json::Value = if models_path.exists() {
        match std::fs::read_to_string(&models_path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|_| serde_json::json!({})),
            Err(_) => serde_json::json!({}),
        }
    } else {
        serde_json::json!({})
    };

    let providers = root.as_object_mut().and_then(|r| {
        r.entry("providers")
            .or_insert_with(|| serde_json::json!({}));
        r.get_mut("providers")?.as_object_mut()
    });
    let Some(providers) = providers else { return };

    let mesh = serde_json::json!({
        "baseUrl": format!("http://localhost:{port}/v1"),
        "api": "openai-completions",
        "apiKey": "mesh",
        "models": [{
            "id": model_id,
            "name": model_id,
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 32768,
            "maxTokens": 8192,
            "compat": {
                "supportsUsageInStreaming": false,
                "maxTokensField": "max_tokens",
                "supportsDeveloperRole": false
            }
        }]
    });

    providers.insert("mesh".to_string(), mesh);

    if let Some(parent) = models_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&root)
        && let Err(e) = std::fs::write(&models_path, json)
    {
        tracing::warn!("Failed to update {}: {e}", models_path.display());
    }
}

/// Resolve Nostr relay URLs from CLI or defaults.
/// Build the list of model refs this node is assigned to serve for gossip announcement.
/// The primary model ref must always appear first in the result.
pub(super) fn build_serving_list(
    startup_models: &[StartupModelPlan],
    model_ref: &str,
) -> Vec<String> {
    let mut all: Vec<String> = startup_models
        .iter()
        .map(|model| model.declared_ref.clone())
        .collect();
    if !all.iter().any(|model| model == model_ref) {
        all.insert(0, model_ref.to_string());
    }
    all.sort();
    if let Some(pos) = all.iter().position(|model| model == model_ref) {
        let primary = all.remove(pos);
        all.insert(0, primary);
    }
    all.dedup();
    all
}

#[cfg(test)]
pub(super) fn format_console_ready_line(headless: bool, console_url: &str) -> String {
    if headless {
        format!("  Management API: {console_url}")
    } else {
        format!("  Console: {console_url}")
    }
}
