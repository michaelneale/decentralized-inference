#![recursion_limit = "256"]
#![allow(
    unfulfilled_lint_expectations,
    reason = "workspace feature unification can make compatibility helpers live"
)]

mod api;
mod capture;
pub mod command_support;
pub mod config_schema;
pub mod crypto;
pub mod discovery;
pub mod inference;
mod logging;
mod mesh;
pub mod models;
mod network;
pub mod plugin;
mod plugins;
mod protocol;
mod runtime;
mod runtime_data;
mod system;

pub mod sdk;

pub mod proto {
    pub use mesh_llm_protocol::proto::*;
}

use anyhow::Result;
pub use crypto::{
    ReleaseAttestationClaims, ReleaseAttestationStatus, ReleaseAttestationSummary,
    ReleaseBuildAttestation, ReleaseSignerTrustStore, TrustedReleaseSigner,
    default_release_signer_trust_store_path, load_release_signer_trust_store,
    parse_release_signer_public_key, release_signer_key_id, save_release_signer_trust_store,
    verify_release_attestation,
};
pub use logging::{
    LoggingRuntimeState, OperationalAuditContext, OperationalAuditPathType, OperationalAuditRecord,
    OperationalAuditSeverity, OperationalAuditSubjectKind,
};
pub use mesh::requirements::{
    BootstrapStatus, DIRECT_NODE_ADMISSION_PROOF_MAX_CLOCK_SKEW_MS, DirectNodeAdmissionProof,
    DirectPeerProofStatus, MeshGenesisPolicy, MeshRequirementDecision,
    MeshRequirementEvaluationInput, MeshRequirementRejectReason, MeshRequirements,
    NodeVersionBounds, PeerReleaseAttestationStatus, ProtocolGenerationBounds,
    ReleaseAttestationRequirement, SignedBootstrapToken, SignedMeshGenesisPolicy,
};
use std::path::Path;
use std::sync::{Arc, LazyLock, RwLock};
use tokio::sync::Mutex as AsyncMutex;

use logging::foundation::LoggingFoundation;
use logging::{LoggingDynamicLimits, LoggingRuntimeApplyError};

/// The current process-local logging foundation.
///
/// Startup can be invoked more than once by embedded hosts and tests.  This
/// must therefore be replaceable instead of using a `OnceLock`: a one-shot
/// holder would silently keep a foundation resolved from an earlier config.
static LOGGING_FOUNDATION: LazyLock<RwLock<Option<Arc<LoggingFoundation>>>> =
    LazyLock::new(|| RwLock::new(None));

/// Durable local logging resources derived from the current foundation.
///
/// This stays separate from the foundation so a store or artifact failure can
/// be represented without changing root-resolution semantics.
static LOGGING_RUNTIME_STATE: LazyLock<RwLock<Option<Arc<LoggingRuntimeState>>>> =
    LazyLock::new(|| RwLock::new(None));

/// Serialize the full async replacement lifecycle. Global holders are only
/// locked long enough to take or publish an `Arc`; the old state's cleanup and
/// bounded persistence drain run with no ordinary global lock held.
static LOGGING_REPLACEMENT: LazyLock<AsyncMutex<()>> = LazyLock::new(|| AsyncMutex::new(()));

pub fn logging_foundation() -> Option<Arc<LoggingFoundation>> {
    LOGGING_FOUNDATION
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Return the process-local metadata/artifact capability state for internal
/// runtime consumers. No filesystem paths or storage errors cross this API.
pub fn logging_runtime_state() -> Option<Arc<LoggingRuntimeState>> {
    LOGGING_RUNTIME_STATE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Apply the logging values whose exported schema explicitly allows live
/// mutation. The result deliberately contains no filesystem details.
pub(crate) fn apply_live_logging_limits(
    config: &mesh_llm_config::LoggingConfig,
) -> Result<(), LoggingRuntimeApplyError> {
    let state = LOGGING_RUNTIME_STATE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .ok_or(LoggingRuntimeApplyError::Unavailable)?;
    state.apply_dynamic_limits(LoggingDynamicLimits::from_config(config))
}

pub fn logging_health_summary() -> String {
    match LOGGING_FOUNDATION
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_deref()
    {
        Some(foundation) => foundation.health_summary(),
        None => "logging not initialized".to_string(),
    }
}

pub const BUILD_VERSION: &str = mesh_llm_build_info::BUILD_VERSION;
pub const RELEASE_VERSION: &str = mesh_llm_build_info::RELEASE_VERSION;
pub const VERSION: &str = RELEASE_VERSION;

/// Configure the Hugging Face TLS provider before any one-shot or runtime
/// client is constructed.
pub fn configure_hf_tls_provider() {
    let _ = model_hf::configure_hf_tls_provider();
}

pub use runtime::{
    MeshGuardrailMode, RuntimeOptions, RuntimeSurface, console_session_mode_for_runtime_surface,
};

/// Configure the ggml Metal pipeline cache directory.
///
/// Point the patched ggml Metal backend at a process-wide on-disk pipeline
/// cache before any native runtime library is loaded. Embedded hosts and
/// binaries must call this exactly once from synchronous bootstrap, before
/// the Tokio runtime is constructed; see
/// `inference::skippy::metal_pipeline_cache` for the full contract. An
/// explicit `GGML_METAL_PIPELINE_CACHE_DIR` set by the user wins.
///
/// # Safety
///
/// The caller must ensure no other thread can read or write the process
/// environment for the duration of this call.
pub unsafe fn configure_metal_pipeline_cache() {
    unsafe { inference::skippy::metal_pipeline_cache::configure_metal_pipeline_cache() }
}

pub async fn run() -> Result<()> {
    initialize_host_runtime().await?;
    runtime::run().await
}

pub async fn run_runtime(
    options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    legacy_warning: Option<String>,
) -> Result<()> {
    initialize_host_runtime_for_options(&options).await?;
    run_runtime_initialized(options, explicit_surface, legacy_warning).await
}

pub async fn run_runtime_initialized(
    options: RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    legacy_warning: Option<String>,
) -> Result<()> {
    runtime::run_cli(options, explicit_surface, legacy_warning).await
}

pub async fn initialize_host_runtime() -> Result<()> {
    initialize_host_runtime_with_config(None).await
}

/// Initialize only the durable logging foundation for a parsed CLI boundary.
///
/// One-shot commands must be auditable, but they must not load a native
/// runtime, require a model, or start serving infrastructure just to emit the
/// static command lifecycle record.
pub async fn initialize_logging_for_cli(config_path: Option<&Path>) -> Result<()> {
    let config = plugin::load_config(config_path)?;
    initialize_logging_foundation(&config.logging).await;
    Ok(())
}

/// Drain the logging-only one-shot CLI boundary before process exit.
pub async fn shutdown_logging_for_one_shot_cli() -> bool {
    let Some(state) = logging_runtime_state() else {
        return false;
    };
    state.shutdown_for_one_shot_cli().await
}

pub async fn initialize_host_runtime_for_options(options: &RuntimeOptions) -> Result<()> {
    options.validate_shared_endpoint_args()?;
    configure_hf_tls_provider();
    if options.plugin.is_some() {
        return Ok(());
    }
    if !runtime_options_require_native_runtime(options) {
        return initialize_logging_for_cli(options.config.as_deref()).await;
    }
    initialize_host_runtime_with_config(options.config.as_deref()).await
}

/// Whether this startup must resolve a native inference runtime.
///
/// A client never serves, and a sharing node serves only by forwarding to an
/// already-running external server, so neither needs the native runtime
/// resolved or loaded. For sharing this is a requirement rather than a
/// tolerated failure: skipping initialization entirely is what distinguishes
/// "deliberately sharing an external endpoint" from "the runtime install is
/// broken", which a downgraded warning cannot express.
fn runtime_options_require_native_runtime(options: &RuntimeOptions) -> bool {
    options.allows_local_inference() && options.plugin.is_none()
}

pub async fn initialize_host_runtime_with_config(config_path: Option<&Path>) -> Result<()> {
    let config = plugin::load_config(config_path)?;

    // Logging config is validated as part of config loading and must be resolved
    // before native runtime setup. Foundation failures remain fail-open so they
    // cannot prevent serving from starting.
    initialize_logging_foundation(&config.logging).await;

    #[cfg(feature = "dynamic-native-runtime")]
    {
        let native_runtime = config.runtime.native_runtime;
        let startup_selection = match native_runtime.mesh_version {
            Some(mesh_version) => {
                let runtime_selection = mesh_llm_native_runtime::RuntimeSelection::parse(
                    native_runtime.selection.as_deref(),
                )?;
                system::native_runtime::NativeRuntimeStartupSelection::explicit(
                    mesh_version,
                    native_runtime.skippy_abi,
                    runtime_selection,
                )
            }
            None => system::native_runtime::NativeRuntimeStartupSelection::current(),
        };
        if let Some(runtime) =
            system::native_runtime::try_load_installed_native_runtime(startup_selection).await?
        {
            tracing::info!(
                native_runtime_id = %runtime.native_runtime_id,
                libraries = ?runtime.libraries,
                "Loaded MeshLLM native runtime"
            );
        }
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        let _ = config;
    }

    Ok(())
}

/// Install the process-local logging resources for one validated runtime config.
///
/// This is shared by the CLI/native startup path and the embedded SDK entrypoint.
/// The holders are intentionally replaceable: an embedded host may start a new
/// runtime with a different config in the same process.
pub(crate) async fn initialize_logging_foundation(config: &mesh_llm_config::LoggingConfig) {
    install_logging_runtime_state(config, |foundation| {
        LoggingRuntimeState::initialize(foundation, config)
    })
    .await;
}

/// Replace the process-local logging state without holding the global holder
/// locks across asynchronous worker retirement. Both the normal host path and
/// the embedded entrypoint use this one boundary.
async fn install_logging_runtime_state<F>(config: &mesh_llm_config::LoggingConfig, build_state: F)
where
    F: FnOnce(&LoggingFoundation) -> LoggingRuntimeState,
{
    let _replacement = LOGGING_REPLACEMENT.lock().await;
    if !retire_previous_logging_runtime_state().await {
        return;
    }

    let foundation = logging_foundation_from_config(config);
    if !foundation.is_healthy() {
        tracing::warn!(
            summary = %foundation.health_summary(),
            "Logging foundation initialized unhealthy (fail-open)"
        );
    }
    let runtime_state = build_state(&foundation);
    if !runtime_state.health().metadata_available {
        tracing::warn!("Logging metadata storage unavailable (fail-open)");
    }
    *LOGGING_RUNTIME_STATE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(runtime_state));
    *LOGGING_FOUNDATION
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Arc::new(foundation));
}

/// Withdraw and retire the old process-local resources before a replacement
/// can be built. A timeout restores the retired state for truthful health and
/// prevents a second cleanup owner from being installed.
async fn retire_previous_logging_runtime_state() -> bool {
    let previous_runtime_state = LOGGING_RUNTIME_STATE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    let previous_foundation = LOGGING_FOUNDATION
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    if let Some(previous_runtime_state) = previous_runtime_state
        && !previous_runtime_state.retire_and_shutdown().await
    {
        restore_timed_out_logging_runtime_state(previous_runtime_state, previous_foundation);
        tracing::warn!("logging cleanup shutdown timed out; replacement deferred");
        return false;
    }
    true
}

/// Preserve a retired runtime after a bounded cleanup timeout. It remains
/// visible as `timed_out` and cannot start a replacement scheduler until a
/// later initialization retry joins the old task safely.
fn restore_timed_out_logging_runtime_state(
    runtime_state: Arc<LoggingRuntimeState>,
    foundation: Option<Arc<LoggingFoundation>>,
) {
    *LOGGING_RUNTIME_STATE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(runtime_state);
    *LOGGING_FOUNDATION
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = foundation;
}

/// Test-only variant of the normal installation boundary with a deterministic
/// store clock. Keeping it here makes process/runtime tests prove the same
/// replacement and `run_auto` startup path as production, without sleeping or
/// depending on wall time.
#[cfg(test)]
pub(crate) async fn initialize_logging_foundation_with_store_clock_for_test(
    config: &mesh_llm_config::LoggingConfig,
    clock: Arc<dyn mesh_llm_log_store::Clock>,
) {
    install_logging_runtime_state(config, |foundation| {
        LoggingRuntimeState::initialize_with_store_clock_for_test(foundation, config, clock)
    })
    .await;
}

fn logging_foundation_from_config(config: &mesh_llm_config::LoggingConfig) -> LoggingFoundation {
    LoggingFoundation::init(config.enabled, config.application_state_root.as_ref())
}

#[cfg(test)]
mod native_runtime_gate_tests {
    use super::*;

    fn sharing() -> RuntimeOptions {
        RuntimeOptions {
            shared_endpoint: Some("http://localhost:11434".to_string()),
            ..RuntimeOptions::default()
        }
    }

    #[test]
    fn sharing_startup_skips_native_runtime_resolution() {
        assert!(!runtime_options_require_native_runtime(&sharing()));
    }

    #[test]
    fn plain_serve_still_requires_a_native_runtime() {
        assert!(runtime_options_require_native_runtime(
            &RuntimeOptions::default()
        ));
    }

    #[test]
    fn client_startup_still_skips_native_runtime_resolution() {
        let options = RuntimeOptions {
            client: true,
            ..RuntimeOptions::default()
        };
        assert!(!runtime_options_require_native_runtime(&options));
    }
}

#[cfg(test)]
#[test]
fn logging_foundation_uses_the_configured_application_state_root() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let configured_root = temporary_directory.path().join("configured-root");
    let config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(configured_root.clone()),
        ..Default::default()
    };

    let foundation = logging_foundation_from_config(&config);

    assert!(foundation.is_healthy());
    assert_eq!(foundation.app_state_root(), configured_root);
}

#[cfg(test)]
#[test]
fn disabled_logging_config_creates_no_application_state_layout() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let configured_root = temporary_directory.path().join("disabled-root");
    let config = mesh_llm_config::LoggingConfig {
        enabled: false,
        application_state_root: Some(configured_root.clone()),
        ..Default::default()
    };

    let foundation = logging_foundation_from_config(&config);

    assert!(!foundation.is_healthy());
    assert!(
        !configured_root.exists(),
        "disabled logging must not create the configured application-state root"
    );
}

#[cfg(test)]
#[tokio::test]
#[serial_test::serial]
async fn logging_foundation_install_replaces_a_previous_in_process_config() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let first_root = temporary_directory.path().join("first-root");
    let second_root = temporary_directory.path().join("second-root");
    let first_config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(first_root),
        ..Default::default()
    };
    let second_config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(second_root.clone()),
        ..Default::default()
    };

    initialize_logging_foundation(&first_config).await;
    initialize_logging_foundation(&second_config).await;

    let installed = logging_foundation().expect("logging foundation should be installed");
    assert_eq!(installed.app_state_root(), second_root);
}

#[cfg(test)]
#[tokio::test]
#[serial_test::serial]
async fn logging_only_cli_initialization_is_inert_when_logging_is_disabled() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let config_path = temporary_directory.path().join("config.toml");
    let disabled_root = temporary_directory.path().join("disabled-state");
    std::fs::write(
        &config_path,
        format!(
            "[logging]\nenabled = false\napplication_state_root = {:?}\n",
            disabled_root
        ),
    )
    .expect("write disabled logging config");

    initialize_logging_for_cli(Some(&config_path))
        .await
        .expect("logging-only CLI initialization remains fail-open");

    assert!(
        !disabled_root.exists(),
        "disabled one-shot logging must not create durable state"
    );
    assert!(
        !logging_runtime_state()
            .expect("inert logging state is installed")
            .health()
            .metadata_available,
        "no store or serving runtime is initialized for disabled logging"
    );
    assert!(!shutdown_logging_for_one_shot_cli().await);
}

#[cfg(test)]
#[tokio::test]
#[serial_test::serial]
async fn normal_runtime_replacement_retires_captured_workers_before_installing_new_state() {
    fn write_config(path: &Path, root: &Path, enabled: bool) {
        let root = root
            .to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"");
        std::fs::write(
            path,
            format!("[logging]\nenabled = {enabled}\napplication_state_root = \"{root}\"\n"),
        )
        .expect("write logging config");
    }

    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let first_config_path = temporary_directory.path().join("first.toml");
    let second_config_path = temporary_directory.path().join("second.toml");
    let first_root = temporary_directory.path().join("first-root");
    let second_root = temporary_directory.path().join("second-root");
    write_config(&first_config_path, &first_root, true);
    write_config(&second_config_path, &second_root, true);

    let first_options = RuntimeOptions {
        client: true,
        config: Some(first_config_path),
        ..Default::default()
    };
    let second_options = RuntimeOptions {
        client: true,
        config: Some(second_config_path),
        ..Default::default()
    };

    initialize_host_runtime_for_options(&first_options)
        .await
        .expect("initialize first normal runtime");
    let retired_state = logging_runtime_state().expect("first logging runtime state");
    let retired_service = retired_state
        .start_persistence_worker()
        .await
        .expect("start first persistence worker");
    assert!(retired_service.is_spawned());

    initialize_host_runtime_for_options(&second_options)
        .await
        .expect("replace normal runtime logging");

    assert!(retired_state.is_retired());
    assert!(retired_state.start_persistence_worker().await.is_none());
    assert!(!retired_service.is_startable());
    assert!(!retired_service.spawn());
    assert!(!retired_service.is_spawned());

    let installed = logging_foundation().expect("replacement foundation");
    assert_eq!(installed.app_state_root(), second_root);
    let current_state = logging_runtime_state().expect("replacement logging state");
    let current_service = current_state
        .start_persistence_worker()
        .await
        .expect("replacement state starts its own worker");
    assert!(current_service.is_spawned());
    initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
    assert!(!current_service.is_spawned());
}

#[cfg(test)]
#[tokio::test]
#[serial_test::serial]
async fn concurrent_replacements_leave_no_worker_owned_by_the_displaced_state() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let initial_config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("initial")),
        ..Default::default()
    };
    initialize_logging_foundation(&initial_config).await;
    let displaced_state = logging_runtime_state().expect("initial logging state");
    let displaced_service = displaced_state
        .start_persistence_worker()
        .await
        .expect("start initial worker");

    let first_config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("first")),
        ..Default::default()
    };
    let second_config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("second")),
        ..Default::default()
    };
    let first = tokio::spawn(async move { initialize_logging_foundation(&first_config).await });
    let second = tokio::spawn(async move { initialize_logging_foundation(&second_config).await });
    first.await.expect("first replacement task");
    second.await.expect("second replacement task");

    assert!(displaced_state.is_retired());
    assert!(displaced_state.start_persistence_worker().await.is_none());
    assert!(!displaced_service.is_startable());
    assert!(!displaced_service.is_spawned());

    initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}

#[cfg(test)]
#[tokio::test]
#[serial_test::serial]
async fn live_logging_limit_application_updates_the_installed_service() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let mut config = mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("live-limits")),
        retention_ttl_secs: 3_600,
        replay_capacity: 4,
        ..Default::default()
    };
    initialize_logging_foundation(&config).await;

    config.retention_ttl_secs = 7_200;
    config.replay_capacity = 2;
    apply_live_logging_limits(&config).expect("healthy installed logging runtime");

    assert_eq!(
        logging_runtime_state()
            .expect("installed runtime state")
            .dynamic_limits(),
        Some(LoggingDynamicLimits {
            retention_ttl_secs: 7_200,
            replay_capacity: 2,
        })
    );

    let config_path = temporary_directory.path().join("owner-control-config.toml");
    let mut config_state = runtime::config_state::ConfigState::load(&config_path)
        .expect("load owner-control config state");
    let mut persisted = plugin::MeshConfig {
        logging: mesh_llm_config::LoggingConfig {
            application_state_root: config.application_state_root.clone(),
            retention_ttl_secs: 7_200,
            replay_capacity: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let revision = match config_state.apply(persisted.clone(), 0) {
        runtime::config_state::ApplyResult::Applied { revision, .. }
        | runtime::config_state::ApplyResult::AppliedWithRestartRequired { revision, .. } => {
            revision
        }
        result => panic!("baseline config application failed: {result:?}"),
    };

    persisted.logging.retention_ttl_secs = 10_800;
    persisted.logging.replay_capacity = 3;
    match config_state.apply_with_live_logging(persisted, revision) {
        runtime::config_state::ApplyResult::Applied {
            apply_mode: runtime::config_state::ConfigApplyMode::Live,
            ..
        } => {}
        result => panic!("expected live dynamic logging apply, got {result:?}"),
    }
}

#[cfg(test)]
include!("exact_test_wrappers.rs");
