use super::*;

use crate::plugin::{GpuAssignment, GpuConfig, MeshConfig};
use mesh_llm_config::{
    ConfigDiagnosticCode, ConfigDiagnosticSeverity, validate_config_diagnostics,
};
use mesh_llm_plugin_manager::{
    InstalledPluginConfigSchema, InstalledPluginManifestMetadata, InstalledPluginMetadata,
    PluginStore, SUPPORTED_PLUGIN_SCHEMA_VERSION,
};
use std::collections::BTreeSet;

#[path = "config_state_tests/support.rs"]
mod support;
use support::*;
#[path = "config_state_tests/diagnostics.rs"]
mod diagnostics;
#[path = "config_state_tests/kv_disk.rs"]
mod kv_disk;
#[path = "config_state_tests/logging.rs"]
mod logging;
#[path = "config_state_tests/persistence.rs"]
mod persistence;
#[path = "config_state_tests/sync.rs"]
mod sync;
