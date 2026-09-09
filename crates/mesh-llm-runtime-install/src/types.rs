//! Native runtime install option and outcome types.

use mesh_llm_native_runtime::{InstalledNativeRuntime, RuntimeSelection};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
pub const CURRENT_MESH_VERSION: &str = mesh_llm_build_info::RELEASE_VERSION;
pub const NATIVE_RUNTIME_CACHE_DIR_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR";
pub const NATIVE_RUNTIME_MANIFEST_URL_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL";

pub type NativeRuntimeDownloadProgressCallback =
    Arc<dyn Fn(NativeRuntimeDownloadProgress) + Send + Sync + 'static>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimeVerificationPolicy {
    #[default]
    RequireChecksum,
    RequireChecksumAndSignature,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimeBundleInstallPolicy {
    #[default]
    UseInPlace,
    InstallExplicitBundlesIntoCache,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeDownloadProgress {
    pub native_runtime_id: String,
    pub url: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub finished: bool,
}

#[derive(Clone)]
pub struct NativeRuntimeManifestOptions {
    pub mesh_version: String,
    pub manifest_path: Option<PathBuf>,
    pub manifest_url: Option<String>,
    pub bundle_dirs: Vec<PathBuf>,
    pub allow_default_manifest_url: bool,
}

#[derive(Clone)]
pub struct NativeRuntimeInstallOptions {
    pub mesh_version: String,
    pub skippy_abi_version: Option<String>,
    pub selection: RuntimeSelection,
    pub manifest_path: Option<PathBuf>,
    pub manifest_url: Option<String>,
    pub bundle_dirs: Vec<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub verification_policy: NativeRuntimeVerificationPolicy,
    pub bundle_install_policy: NativeRuntimeBundleInstallPolicy,
    pub progress: Option<NativeRuntimeDownloadProgressCallback>,
    pub allow_download: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimeInstallStatus {
    AlreadyInstalled,
    Installed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeInstallOutcome {
    pub status: NativeRuntimeInstallStatus,
    pub runtime: InstalledNativeRuntime,
    pub resolution: mesh_llm_native_runtime::NativeRuntimeResolution,
    /// Which catalogs were consulted to reach this resolution.
    #[serde(default)]
    pub sources: crate::manifest::NativeRuntimeCatalogSources,
}

impl Default for NativeRuntimeManifestOptions {
    fn default() -> Self {
        Self {
            mesh_version: CURRENT_MESH_VERSION.to_string(),
            manifest_path: None,
            manifest_url: None,
            bundle_dirs: Vec::new(),
            allow_default_manifest_url: true,
        }
    }
}

impl Default for NativeRuntimeInstallOptions {
    fn default() -> Self {
        Self {
            mesh_version: CURRENT_MESH_VERSION.to_string(),
            skippy_abi_version: None,
            selection: RuntimeSelection::Recommended,
            manifest_path: None,
            manifest_url: None,
            bundle_dirs: Vec::new(),
            cache_dir: None,
            verification_policy: NativeRuntimeVerificationPolicy::RequireChecksum,
            bundle_install_policy: NativeRuntimeBundleInstallPolicy::UseInPlace,
            progress: None,
            allow_download: true,
        }
    }
}
