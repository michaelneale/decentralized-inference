use super::*;
use crate::api;
use crate::mesh;
use crate::models;
use crate::network::affinity;
use crate::network::{discovery as mesh_discovery, nostr};
use crate::plugin;
use crate::plugin::{GpuAssignment, GpuConfig, ModelConfigEntry};
use crate::runtime::status::{current_time_unix_ms, publication_state_from_update};
use crate::system::hardware::GpuFacts;
use crate::system::{backend, benchmark, hardware};
use hf_hub::RepoTypeModel;
use mesh_llm_events::{DashboardEndpointRow, DashboardSnapshotProvider, RuntimeStatus};
use model_hf::{huggingface_repo_folder_name, huggingface_snapshot_path};
use skippy_protocol::FlashAttentionType;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

fn restore_env(key: &str, value: Option<std::ffi::OsString>) {
    match value {
        Some(value) => unsafe { std::env::set_var(key, value) },
        None => unsafe { std::env::remove_var(key) },
    }
}

async fn wait_for_condition<F, Fut>(timeout: Duration, mut check: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if check().await {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "condition timed out"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

async fn build_test_mesh_api() -> api::MeshApi {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .expect("test node");
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
    .expect("plugin manager");
    let runtime_data_collector = crate::runtime_data::RuntimeDataCollector::new();
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

mod auto_join;
mod dashboard;
mod local_model_only;
mod logging;
mod model_lifecycle;
mod publication;
mod serving_surface;
mod startup_models;
mod tracing_writer;
