use super::{
    PreparedRuntimeStartup, RuntimeOptions, RuntimeSurface, detect_bin_dir,
    model_fits_runtime_capacity, runtime_model_required_bytes,
};
use crate::MeshRequirements;
use crate::crypto::{
    OwnerKeychainLoadError, default_keystore_path, default_trust_store_path, keystore_exists,
    keystore_metadata, load_keystore, load_owner_keypair_from_keychain, load_trust_store,
};
use crate::inference::{election, skippy};
use crate::mesh;
use crate::models;
use crate::plugin;
use crate::system::{backend, hardware};
use anyhow::{Context, Result};
use mesh_llm_events::{OutputEvent, emit_event};
use skippy_protocol::FlashAttentionType;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

mod local_required;

use local_required::resolve_local_required_startup_model;
pub(super) use local_required::validate_local_required_source;

#[derive(Clone, Debug, PartialEq, Eq)]

pub(super) struct StartupMeshCreationState {
    pub(super) requirements: MeshRequirements,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StartupModelSpec {
    pub(super) model_ref: PathBuf,
    pub(super) declared_ref: Option<String>,
    pub(super) config_model_id: Option<String>,
    pub(super) mmproj_ref: Option<PathBuf>,
    pub(super) ctx_size: Option<u32>,
    pub(super) gpu_id: Option<String>,
    /// Whether `gpu_id` came from an explicit non-auto CLI `--device`.
    pub(super) cli_device_override: bool,
    /// Whether pinned startup preflight must resolve this model's GPU selector.
    ///
    /// Config-owned models opt in. An explicit CLI `--model` also opts in
    /// when its ref exactly and uniquely matches a configured model. Ad-hoc
    /// model refs and GGUF paths remain outside per-model config ownership.
    pub(super) resolve_pinned_gpu: bool,
    pub(super) parallel: Option<usize>,
    pub(super) cache_type_k: Option<String>,
    pub(super) cache_type_v: Option<String>,
    pub(super) n_batch: Option<u32>,
    pub(super) n_ubatch: Option<u32>,
    pub(super) flash_attention: FlashAttentionType,
    pub(super) local_source_required: bool,
    pub(super) profile: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StartupPinnedGpuTarget {
    pub(crate) index: usize,
    pub(crate) stable_id: String,
    pub(crate) backend_device: String,
    pub(crate) vram_bytes: u64,
    pub(crate) reserved_bytes: Option<u64>,
}

impl StartupPinnedGpuTarget {
    pub(crate) fn allocatable_vram_bytes(&self) -> u64 {
        mesh_llm_system::vram::allocatable_bytes(self.vram_bytes, self.reserved_bytes)
    }
}

#[derive(Clone, Debug)]
pub(super) struct StartupModelPlan {
    pub(super) declared_ref: String,
    pub(super) config_model_id: Option<String>,
    pub(super) resolved_path: PathBuf,
    pub(super) preindexed_split_package: Option<skippy::SkippyPackageIdentity>,
    pub(super) mmproj_path: Option<PathBuf>,
    pub(super) ctx_size: Option<u32>,
    pub(super) gpu_id: Option<String>,
    pub(super) pinned_gpu: Option<StartupPinnedGpuTarget>,
    pub(super) parallel: Option<usize>,
    pub(super) cache_type_k: Option<String>,
    pub(super) cache_type_v: Option<String>,
    pub(super) n_batch: Option<u32>,
    pub(super) n_ubatch: Option<u32>,
    pub(super) flash_attention: FlashAttentionType,
    pub(super) local_source_required: bool,
    pub(super) profile: String,
}

pub(super) fn resolve_runtime_owner_key_path(options: &RuntimeOptions) -> Result<Option<PathBuf>> {
    if let Some(path) = options.owner_key.clone() {
        return Ok(Some(path));
    }

    let default_path = default_keystore_path()?;
    if keystore_exists(&default_path) {
        Ok(Some(default_path))
    } else {
        Ok(None)
    }
}

pub(super) fn resolve_owner_passphrase(path: &Path) -> Result<Option<Zeroizing<String>>> {
    let info = keystore_metadata(path)?;
    if !info.encrypted {
        return Ok(None);
    }

    if let Ok(passphrase) = std::env::var("MESH_LLM_OWNER_PASSPHRASE") {
        return Ok(Some(Zeroizing::new(passphrase)));
    }

    if std::io::stdin().is_terminal() && std::io::stderr().is_terminal() {
        let prompt = format!("Enter owner keystore passphrase for {}: ", path.display());
        let passphrase = rpassword::prompt_password(&prompt)?;
        return Ok(Some(Zeroizing::new(passphrase)));
    }

    Err(crate::crypto::CryptoError::MissingPassphrase.into())
}

pub(super) fn load_owner_keypair_for_runtime(path: &Path) -> Result<crate::crypto::OwnerKeypair> {
    let info = keystore_metadata(path)?;
    if info.encrypted && std::env::var("MESH_LLM_OWNER_PASSPHRASE").is_err() {
        match load_owner_keypair_from_keychain(path) {
            Ok(keypair) => return Ok(keypair),
            Err(OwnerKeychainLoadError::NoEntry)
            | Err(OwnerKeychainLoadError::Crypto(crate::crypto::CryptoError::DecryptionFailed))
            | Err(OwnerKeychainLoadError::Crypto(
                crate::crypto::CryptoError::KeychainUnavailable { .. },
            ))
            | Err(OwnerKeychainLoadError::Crypto(
                crate::crypto::CryptoError::KeychainAccessDenied { .. },
            )) => {}
            Err(OwnerKeychainLoadError::Crypto(err)) => {
                return Err(err)
                    .with_context(|| format!("Failed to load owner keystore {}", path.display()));
            }
        }
    }

    let passphrase = resolve_owner_passphrase(path)?;
    load_keystore(path, passphrase.as_deref().map(|value| value.as_str()))
        .with_context(|| format!("Failed to load owner keystore {}", path.display()))
}

pub(super) fn owner_runtime_config(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
) -> Result<mesh::OwnerRuntimeConfig> {
    let trust_store_path = default_trust_store_path()?;
    let trust_store = load_trust_store(&trust_store_path)
        .with_context(|| format!("Failed to load trust store {}", trust_store_path.display()))?
        .merged_with_trusted_owners(&options.trust_owner);
    let trust_policy = options.trust_policy.unwrap_or(trust_store.policy);

    let keypair = match resolve_runtime_owner_key_path(options)? {
        Some(path) => match load_owner_keypair_for_runtime(&path) {
            Ok(keypair) => Some(keypair),
            Err(err) if !options.owner_required => {
                let _ = emit_event(OutputEvent::Warning {
                    message: format!(
                        "Owner identity unavailable: {err}. Starting without owner attestation."
                    ),
                    context: Some(path.display().to_string()),
                });
                None
            }
            Err(err) => return Err(err),
        },
        None if options.owner_required => {
            anyhow::bail!(
                "Owner identity is required but no keystore was found. To enable owner control, run `mesh-llm auth init --no-passphrase`, then restart with `mesh-llm serve --owner-required`."
            );
        }
        None => None,
    };

    Ok(mesh::OwnerRuntimeConfig {
        keypair,
        control_bind: options.control_bind.or(config.owner_control.bind),
        control_advertise_addr: options
            .control_advertise_addr
            .or(config.owner_control.advertise_addr),
        node_label: options.node_label.clone(),
        trust_store,
        trust_policy,
        activity: config.runtime.activity.clone(),
        drain_timeout_secs: config.runtime.drain_timeout_secs,
        drain_timeout_max_secs: config.runtime.drain_timeout_max_secs,
        public_mesh: options.publish || options.nostr_discovery,
    })
}

pub(super) fn emit_configuration_ui_read_only_hint() {
    let _ = emit_event(OutputEvent::Warning {
        message: "Configuration UI is read-only: no owner identity found. To enable saving config from the UI:\n  mesh-llm auth init --no-passphrase\n  mesh-llm serve --owner-required".to_string(),
        context: None,
    });
}

pub(super) fn resolve_startup_mesh_creation_state(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
) -> Result<StartupMeshCreationState> {
    let merged = plugin::MeshRequirementsConfig {
        min_node_version: options
            .min_node_version
            .clone()
            .or_else(|| config.mesh_requirements.min_node_version.clone()),
        max_node_version: options
            .max_node_version
            .clone()
            .or_else(|| config.mesh_requirements.max_node_version.clone()),
        min_protocol_version: options
            .min_protocol_version
            .or(config.mesh_requirements.min_protocol_version),
        max_protocol_version: options
            .max_protocol_version
            .or(config.mesh_requirements.max_protocol_version),
        require_release_attestation: options.require_release_attestation
            || config.mesh_requirements.require_release_attestation,
        release_signer_keys: if options.release_signer_key.is_empty() {
            config.mesh_requirements.release_signer_keys.clone()
        } else {
            options.release_signer_key.clone()
        },
    };
    let requirements = plugin::mesh_requirements_config_to_runtime(&merged);
    requirements
        .validate()
        .map_err(|reason| anyhow::anyhow!(plugin::mesh_requirements_validation_error(reason)))?;
    requirements
        .release_attestation
        .validate_signer_key_shapes()
        .map_err(|reason| anyhow::anyhow!(plugin::mesh_requirements_validation_error(reason)))?;
    Ok(StartupMeshCreationState { requirements })
}

#[cfg(test)]
pub(super) fn ensure_existing_mesh_requirements_match(
    startup_state: &StartupMeshCreationState,
    existing_policy: &crate::MeshGenesisPolicy,
) -> Result<()> {
    if existing_policy.requirements == startup_state.requirements {
        return Ok(());
    }
    anyhow::bail!(
        "Local mesh requirements conflict with the joined mesh genesis policy. Changing mesh requirements creates a new mesh; remove the local creation-time overrides or start a new mesh instead."
    );
}

#[cfg(test)]
pub(crate) fn assert_mesh_requirements_cli_accepts_each_bound_independently() {
    let min_only = runtime_options_for_test(&["mesh-llm", "--min-node-version", "0.65.0"]);
    assert_eq!(min_only.min_node_version.as_deref(), Some("0.65.0"));
    assert_eq!(min_only.max_node_version, None);

    let max_only = runtime_options_for_test(&["mesh-llm", "--max-node-version", "0.65.9"]);
    assert_eq!(max_only.min_node_version, None);
    assert_eq!(max_only.max_node_version.as_deref(), Some("0.65.9"));

    let min_protocol = runtime_options_for_test(&["mesh-llm", "--min-protocol-version", "1"]);
    assert_eq!(min_protocol.min_protocol_version, Some(1));
    assert_eq!(min_protocol.max_protocol_version, None);

    let max_protocol = runtime_options_for_test(&["mesh-llm", "--max-protocol-version", "3"]);
    assert_eq!(max_protocol.min_protocol_version, None);
    assert_eq!(max_protocol.max_protocol_version, Some(3));

    let attestation = runtime_options_for_test(&[
        "mesh-llm",
        "--require-release-attestation",
        "--release-signer-key",
        "signer-a",
        "--release-signer-key",
        "signer-b",
    ]);
    assert!(attestation.require_release_attestation);
    assert_eq!(
        attestation.release_signer_key,
        vec!["signer-a".to_string(), "signer-b".to_string()]
    );
}

#[cfg(test)]
pub(crate) fn assert_mesh_requirements_cli_overrides_config_per_field_before_genesis() {
    let options = runtime_options_for_test(&[
        "mesh-llm",
        "--min-node-version",
        "0.65.3",
        "--max-protocol-version",
        "5",
        "--release-signer-key",
        "ed25519:3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
    ]);
    let config = plugin::MeshConfig {
        mesh_requirements: plugin::MeshRequirementsConfig {
            min_node_version: Some("0.65.0".into()),
            max_node_version: Some("0.65.9".into()),
            min_protocol_version: Some(1),
            max_protocol_version: Some(2),
            require_release_attestation: true,
            release_signer_keys: vec![
                "ed25519:d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a".into(),
            ],
        },
        ..plugin::MeshConfig::default()
    };

    let startup_state = resolve_startup_mesh_creation_state(&options, &config)
        .expect("merged requirements should validate");
    let policy = crate::MeshGenesisPolicy::new(
        "owner-123",
        1_717_171_717_000,
        startup_state.requirements.clone(),
    )
    .expect("genesis policy should validate after merge");

    assert_eq!(
        startup_state.requirements.node_version.min.as_deref(),
        Some("0.65.3")
    );
    assert_eq!(
        startup_state.requirements.node_version.max.as_deref(),
        Some("0.65.9")
    );
    assert_eq!(startup_state.requirements.protocol_generation.min, Some(1));
    assert_eq!(startup_state.requirements.protocol_generation.max, Some(5));
    assert!(startup_state.requirements.release_attestation.required);
    assert_eq!(
        startup_state
            .requirements
            .release_attestation
            .allowed_signer_keys,
        vec![
            "ed25519:3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c".to_string()
        ]
    );
    assert_eq!(policy.requirements, startup_state.requirements);
    assert_eq!(
        runtime_startup_requirements(&startup_state),
        &startup_state.requirements,
        "merged mesh requirements must remain available after entering runtime startup state"
    );
}

#[cfg(test)]
pub(crate) fn assert_mesh_requirements_config_rejects_min_greater_than_max_after_merge() {
    let options = runtime_options_for_test(&["mesh-llm", "--min-node-version", "0.65.5"]);
    let config = plugin::MeshConfig {
        mesh_requirements: plugin::MeshRequirementsConfig {
            max_node_version: Some("0.65.4".into()),
            ..plugin::MeshRequirementsConfig::default()
        },
        ..plugin::MeshConfig::default()
    };

    let err = resolve_startup_mesh_creation_state(&options, &config)
        .expect_err("merged bounds should be rejected");
    assert!(err.to_string().contains(
        "mesh_requirements.min_node_version must be less than or equal to mesh_requirements.max_node_version"
    ));
}

#[cfg(test)]
pub(crate) fn assert_mesh_requirements_rejects_local_policy_mutation_on_existing_mesh() {
    let options = runtime_options_for_test(&["mesh-llm", "--max-node-version", "0.65.9"]);
    let config = plugin::MeshConfig {
        mesh_requirements: plugin::MeshRequirementsConfig {
            require_release_attestation: true,
            release_signer_keys: vec![
                "ed25519:d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a".into(),
            ],
            ..plugin::MeshRequirementsConfig::default()
        },
        ..plugin::MeshConfig::default()
    };
    let startup_state = resolve_startup_mesh_creation_state(&options, &config)
        .expect("local requirements should validate");
    let existing_policy = crate::MeshGenesisPolicy::new(
        "owner-123",
        1_717_171_717_000,
        MeshRequirements::unrestricted(),
    )
    .expect("existing policy should validate");

    let err = ensure_existing_mesh_requirements_match(&startup_state, &existing_policy)
        .expect_err("policy mutation should be rejected");
    assert_eq!(
        err.to_string(),
        "Local mesh requirements conflict with the joined mesh genesis policy. Changing mesh requirements creates a new mesh; remove the local creation-time overrides or start a new mesh instead."
    );
}

pub(super) fn runtime_startup_requirements(state: &StartupMeshCreationState) -> &MeshRequirements {
    &state.requirements
}

pub(super) fn validate_runtime_cli_model_options(options: &RuntimeOptions) -> Result<()> {
    if options.client && (!options.model.is_empty() || !options.gguf.is_empty()) {
        anyhow::bail!("--client and --model are mutually exclusive");
    }
    if let Some(mmproj) = &options.mmproj {
        anyhow::ensure!(!options.client, "--mmproj cannot be used with --client");
        anyhow::ensure!(
            !options.model.is_empty() || !options.gguf.is_empty(),
            "--mmproj requires an explicit primary model via --model or --gguf"
        );
        anyhow::ensure!(
            mmproj.is_file(),
            "mmproj path is not a file: {}",
            mmproj.display()
        );
    }
    if let Some(path) = &options.split_topology_lock {
        anyhow::ensure!(options.split, "--split-topology-lock requires --split");
        anyhow::ensure!(
            path.is_file(),
            "split topology lock path is not a file: {}",
            path.display()
        );
    }
    Ok(())
}

pub(super) async fn prepare_runtime_startup(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
    explicit_surface: Option<RuntimeSurface>,
    effective_mode: mesh_llm_config::RuntimeMode,
) -> Result<Option<PreparedRuntimeStartup>> {
    validate_runtime_cli_model_options(options)?;
    let startup_specs = if effective_mode == mesh_llm_config::RuntimeMode::OnDemand
        && !cli_has_explicit_models(options)
    {
        Vec::new()
    } else {
        build_startup_model_specs(options, config)?
    };
    if options.split_topology_lock.is_some() {
        anyhow::ensure!(
            startup_specs.len() == 1,
            "--split-topology-lock requires exactly one startup model"
        );
    }
    let _ = explicit_surface;

    let bin_dir = match &options.bin_dir {
        Some(dir) => dir.clone(),
        None => detect_bin_dir()?,
    };
    // Model resolution is deliberately deferred until run_auto has bound the
    // management, OpenAI, plugin, and control surfaces.
    let requested_model_names = startup_specs
        .iter()
        .map(|spec| {
            spec.declared_ref
                .clone()
                .unwrap_or_else(|| spec.model_ref.to_string_lossy().into_owned())
        })
        .collect();
    Ok(Some(PreparedRuntimeStartup {
        startup_specs,
        requested_model_names,
        bin_dir,
    }))
}

pub(super) async fn resolve_eager_startup_models(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
    startup_specs: &[StartupModelSpec],
) -> Result<Vec<StartupModelPlan>> {
    let mut startup_models = resolve_startup_models(startup_specs, options.split).await?;
    preflight_pinned_startup_models(
        config,
        startup_specs,
        &mut startup_models,
        options.llama_flavor,
        None,
    )?;
    let resolved_models: Vec<PathBuf> = startup_models
        .iter()
        .map(|model| model.resolved_path.clone())
        .collect();
    spawn_advisory_startup_task(move || {
        models::warn_about_updates_for_paths(&resolved_models);
    });
    Ok(startup_models)
}

// Snapshot update checks are advisory. Serving must not wait on Hub reachability.
pub(in crate::runtime) fn spawn_advisory_startup_task(task: impl FnOnce() + Send + 'static) {
    std::mem::drop(tokio::task::spawn_blocking(task));
}

#[cfg(test)]
pub(super) fn runtime_options_for_test(args: &[&str]) -> RuntimeOptions {
    let mut options = RuntimeOptions::default();
    let mut iter = args.iter().copied();
    while let Some(arg) = iter.next() {
        match arg {
            "mesh-llm" | "serve" => {}
            "client" | "--client" => options.client = true,
            "--auto" => options.auto = true,
            "--publish" => options.publish = true,
            "--local-model-only" => options.local_model_only = true,
            "--discover" => options.discover = Some(next_test_arg(&mut iter, arg).to_string()),
            "--split" => options.split = true,
            "--require-release-attestation" => options.require_release_attestation = true,
            "--join" => options.join.push(next_test_arg(&mut iter, arg).to_string()),
            "--model" => options.model.push(next_test_arg(&mut iter, arg).into()),
            "--gguf" => options.gguf.push(next_test_arg(&mut iter, arg).into()),
            "--mmproj" => options.mmproj = Some(next_test_arg(&mut iter, arg).into()),
            "--device" => options.device = Some(next_test_arg(&mut iter, arg).to_string()),
            "--ctx-size" => {
                options.ctx_size = Some(
                    next_test_arg(&mut iter, arg)
                        .parse()
                        .expect("valid --ctx-size test value"),
                );
            }
            "--mesh-name" => options.mesh_name = Some(next_test_arg(&mut iter, arg).to_string()),
            "--swarm-capture" => options.swarm_capture = Some(next_test_arg(&mut iter, arg).into()),
            "--min-node-version" => {
                options.min_node_version = Some(next_test_arg(&mut iter, arg).to_string());
            }
            "--max-node-version" => {
                options.max_node_version = Some(next_test_arg(&mut iter, arg).to_string());
            }
            "--min-protocol-version" => {
                options.min_protocol_version = Some(
                    next_test_arg(&mut iter, arg)
                        .parse()
                        .expect("valid --min-protocol-version test value"),
                );
            }
            "--max-protocol-version" => {
                options.max_protocol_version = Some(
                    next_test_arg(&mut iter, arg)
                        .parse()
                        .expect("valid --max-protocol-version test value"),
                );
            }
            "--release-signer-key" => {
                options
                    .release_signer_key
                    .push(next_test_arg(&mut iter, arg).to_string());
            }
            "--config" => options.config = Some(next_test_arg(&mut iter, arg).into()),
            "--max-vram" => {
                options.max_vram = Some(
                    next_test_arg(&mut iter, arg)
                        .parse()
                        .expect("valid --max-vram test value"),
                );
            }
            "--port" => {
                options.port = next_test_arg(&mut iter, arg)
                    .parse()
                    .expect("valid --port test value");
            }
            "--console" => {
                options.console = next_test_arg(&mut iter, arg)
                    .parse()
                    .expect("valid --console test value");
            }
            other => panic!("unsupported runtime_options_for_test arg: {other}"),
        }
    }
    options
}

#[cfg(test)]
pub(super) fn next_test_arg<'a>(iter: &mut impl Iterator<Item = &'a str>, flag: &str) -> &'a str {
    iter.next()
        .unwrap_or_else(|| panic!("missing value for {flag}"))
}

/// Resolve a model path: local file, catalog name, or HuggingFace URL.
pub(super) async fn resolve_model(input: &std::path::Path) -> Result<PathBuf> {
    models::resolve_model_spec(input).await
}

pub(super) fn cli_has_explicit_models(options: &RuntimeOptions) -> bool {
    !options.model.is_empty() || !options.gguf.is_empty()
}

pub(super) fn resolve_model_parallel_override(
    model_parallel: Option<usize>,
    gpu_config: &plugin::GpuConfig,
) -> Option<usize> {
    model_parallel.or(gpu_config.parallel)
}

pub(super) fn resolve_model_parallel_slots(
    model_parallel: Option<usize>,
    gpu_config: &plugin::GpuConfig,
    default_slots: usize,
) -> usize {
    resolve_model_parallel_override(model_parallel, gpu_config).unwrap_or(default_slots)
}

/// Detect the `--gguf <path> --model <alias>` naming form.
///
/// Returns the alias when the user supplied exactly one local GGUF and exactly
/// one `--model` value that is a plain name rather than another model spec.
/// Anything that could itself resolve — an existing path, a Hugging Face ref,
/// or a URL — keeps the previous behaviour of being served as its own model.
fn gguf_alias_from_cli(options: &RuntimeOptions) -> Option<String> {
    if options.gguf.len() != 1 || options.model.len() != 1 {
        return None;
    }
    let candidate = options.model[0].to_str()?;
    if candidate.is_empty() || !is_plain_model_alias(candidate) {
        return None;
    }
    if options.model[0].exists() {
        return None;
    }
    Some(candidate.to_string())
}

/// A plain alias carries no path, ref, or URL syntax.
fn is_plain_model_alias(candidate: &str) -> bool {
    !candidate.contains('/')
        && !candidate.contains('\\')
        && !candidate.contains(':')
        && !candidate.contains('@')
}

fn effective_startup_model_config(
    model_ref: &str,
    model: Option<&plugin::ModelConfigEntry>,
    defaults: Option<&plugin::ModelConfigDefaults>,
) -> plugin::ModelConfigEntry {
    let profile_entry = model
        .cloned()
        .unwrap_or_else(|| plugin::ModelConfigEntry {
            model: model_ref.to_string(),
            ..plugin::ModelConfigEntry::default()
        })
        .with_profile_defaults(defaults);
    let default_fit = defaults.and_then(|value| value.model_fit.as_ref());
    let default_hardware = defaults.and_then(|value| value.hardware.as_ref());
    let default_throughput = defaults.and_then(|value| value.throughput.as_ref());
    let default_multimodal = defaults.and_then(|value| value.multimodal.as_ref());
    plugin::ModelConfigEntry {
        model: model_ref.to_string(),
        mmproj: model
            .and_then(|value| value.mmproj.clone())
            .or_else(|| default_multimodal.and_then(|value| value.mmproj.clone()))
            .or_else(|| default_hardware.and_then(|value| value.mmproj.clone())),
        ctx_size: model
            .and_then(|value| value.ctx_size)
            .or_else(|| default_fit.and_then(|value| value.ctx_size)),
        gpu_id: model
            .and_then(|value| value.gpu_id.clone())
            .or_else(|| default_hardware.and_then(|value| value.device.clone())),
        parallel: model
            .and_then(|value| value.parallel)
            .or_else(|| default_throughput.and_then(|value| value.parallel)),
        cache_type_k: model
            .and_then(|value| value.cache_type_k.clone())
            .or_else(|| default_fit.and_then(|value| value.cache_type_k.clone())),
        cache_type_v: model
            .and_then(|value| value.cache_type_v.clone())
            .or_else(|| default_fit.and_then(|value| value.cache_type_v.clone())),
        batch: model
            .and_then(|value| value.batch)
            .or_else(|| default_fit.and_then(|value| value.batch)),
        ubatch: model
            .and_then(|value| value.ubatch)
            .or_else(|| default_fit.and_then(|value| value.ubatch)),
        flash_attention: model
            .and_then(|value| value.flash_attention)
            .or_else(|| default_fit.and_then(|value| value.flash_attention)),
        model_fit: profile_entry.model_fit,
        hardware: profile_entry.hardware,
        throughput: profile_entry.throughput,
        skippy: profile_entry.skippy,
        ..plugin::ModelConfigEntry::default()
    }
}

fn configured_server_alias(
    model: &plugin::ModelConfigEntry,
    defaults: Option<&plugin::ModelConfigDefaults>,
) -> Option<String> {
    model
        .advanced
        .as_ref()
        .and_then(|advanced| advanced.server.as_ref())
        .and_then(|server| server.alias.as_deref())
        .or_else(|| {
            defaults
                .and_then(|value| value.advanced.as_ref())
                .and_then(|advanced| advanced.server.as_ref())
                .and_then(|server| server.alias.as_deref())
        })
        .map(str::trim)
        .filter(|alias| !alias.is_empty())
        .map(str::to_string)
}

/// Return the effective persisted device selector for one configured model.
fn configured_model_gpu_id(
    config: &plugin::MeshConfig,
    model: &plugin::ModelConfigEntry,
) -> Option<String> {
    model
        .hardware
        .as_ref()
        .and_then(|hardware| hardware.device.clone())
        .or_else(|| model.gpu_id.clone())
        .or_else(|| configured_default_gpu_id(config))
}

/// Return the global device selector inherited by ad-hoc startup models.
fn configured_default_gpu_id(config: &plugin::MeshConfig) -> Option<String> {
    config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.hardware.as_ref())
        .and_then(|hardware| hardware.device.clone())
}

/// Match only an exact CLI `--model` ref.
///
/// The CLI has no profile selector, so duplicate refs are ambiguous. GGUF
/// paths and aliases do not participate in per-model config matching.
fn matching_config_model<'a>(
    config: &'a plugin::MeshConfig,
    model_ref: &Path,
) -> Result<Option<&'a plugin::ModelConfigEntry>> {
    let Some(model_ref) = model_ref.to_str() else {
        return Ok(None);
    };
    let mut matches = config
        .models
        .iter()
        .filter(|model| model.model == model_ref);
    let Some(first) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() {
        anyhow::bail!(
            "CLI --model '{}' matches multiple configured model entries; select a unique configured model ref",
            model_ref
        );
    }
    Ok(Some(first))
}

/// Apply CLI device precedence without treating `--device auto` as a pin.
fn effective_startup_gpu_id(options: &RuntimeOptions, persisted: Option<&str>) -> Option<String> {
    options
        .device
        .as_deref()
        .map(str::trim)
        .filter(|device| !device.is_empty() && !device.eq_ignore_ascii_case("auto"))
        .map(str::to_string)
        .or_else(|| persisted.map(str::to_string))
}

fn has_explicit_startup_device(options: &RuntimeOptions) -> bool {
    options.device.as_deref().is_some_and(|device| {
        let device = device.trim();
        !device.is_empty() && !device.eq_ignore_ascii_case("auto")
    })
}

pub(super) fn skippy_local_source_required(
    model_skippy: Option<&plugin::SkippyConfig>,
    default_skippy: Option<&plugin::SkippyConfig>,
) -> bool {
    model_skippy
        .and_then(|skippy| skippy.source_policy.as_deref())
        .or_else(|| default_skippy.and_then(|skippy| skippy.source_policy.as_deref()))
        == Some("local-required")
}

pub(super) fn build_startup_model_specs(
    options: &RuntimeOptions,
    config: &plugin::MeshConfig,
) -> Result<Vec<StartupModelSpec>> {
    if options.client {
        return Ok(Vec::new());
    }

    let mut specs = Vec::new();
    let default_skippy = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.skippy.as_ref());
    if cli_has_explicit_models(options) {
        let defaults = config.defaults.as_ref();
        // `--gguf <path> --model <alias>` names the local file rather than
        // requesting a second model: bind the alias to the GGUF so we never
        // try to resolve it against the Hugging Face hub or the catalog.
        if let Some(alias) = gguf_alias_from_cli(options) {
            let path = &options.gguf[0];
            if !path.exists() {
                anyhow::bail!("GGUF file not found: {}", path.display());
            }
            let effective = effective_startup_model_config(&alias, None, defaults);
            let local_source_required =
                skippy_local_source_required(effective.skippy.as_ref(), None);
            if local_source_required {
                validate_local_required_source(path, &alias)?;
            }
            specs.push(StartupModelSpec {
                model_ref: path.clone(),
                declared_ref: Some(alias),
                config_model_id: None,
                mmproj_ref: options
                    .mmproj
                    .clone()
                    .or_else(|| effective.mmproj.as_ref().map(PathBuf::from)),
                ctx_size: options.ctx_size.or(effective.ctx_size),
                gpu_id: effective_startup_gpu_id(options, effective.gpu_id.as_deref()),
                cli_device_override: has_explicit_startup_device(options),
                resolve_pinned_gpu: false,
                parallel: effective.parallel,
                cache_type_k: effective.cache_type_k.clone(),
                cache_type_v: effective.cache_type_v.clone(),
                n_batch: effective.batch,
                n_ubatch: effective.ubatch,
                flash_attention: effective
                    .flash_attention
                    .unwrap_or(FlashAttentionType::Auto),
                local_source_required,
                profile: effective.derived_profile(),
            });
            return Ok(specs);
        }

        for path in &options.gguf {
            if !path.exists() {
                anyhow::bail!("GGUF file not found: {}", path.display());
            }
            let effective =
                effective_startup_model_config(&path.display().to_string(), None, defaults);
            let local_source_required =
                skippy_local_source_required(effective.skippy.as_ref(), None);
            if local_source_required {
                validate_local_required_source(path, path.to_str().unwrap_or("local model"))?;
            }
            specs.push(StartupModelSpec {
                model_ref: path.clone(),
                declared_ref: None,
                config_model_id: None,
                mmproj_ref: effective.mmproj.as_ref().map(PathBuf::from),
                ctx_size: options.ctx_size.or(effective.ctx_size),
                gpu_id: effective_startup_gpu_id(options, effective.gpu_id.as_deref()),
                cli_device_override: has_explicit_startup_device(options),
                resolve_pinned_gpu: false,
                parallel: effective.parallel,
                cache_type_k: effective.cache_type_k.clone(),
                cache_type_v: effective.cache_type_v.clone(),
                n_batch: effective.batch,
                n_ubatch: effective.ubatch,
                flash_attention: effective
                    .flash_attention
                    .unwrap_or(FlashAttentionType::Auto),
                local_source_required,
                profile: effective.derived_profile(),
            });
        }
        for model in &options.model {
            let matching_config = matching_config_model(config, model)?;
            let effective =
                effective_startup_model_config(&model.display().to_string(), None, defaults);
            let local_source_required = skippy_local_source_required(
                matching_config.and_then(|model| model.skippy.as_ref()),
                default_skippy,
            );
            if local_source_required {
                validate_local_required_source(model, model.to_str().unwrap_or("local model"))?;
            }
            let persisted_gpu_id = matching_config
                .and_then(|model| configured_model_gpu_id(config, model))
                .or_else(|| configured_default_gpu_id(config));
            specs.push(StartupModelSpec {
                model_ref: model.clone(),
                declared_ref: None,
                config_model_id: None,
                mmproj_ref: effective.mmproj.as_ref().map(PathBuf::from),
                ctx_size: options.ctx_size.or(effective.ctx_size),
                gpu_id: effective_startup_gpu_id(options, persisted_gpu_id.as_deref()),
                cli_device_override: has_explicit_startup_device(options),
                resolve_pinned_gpu: matching_config.is_some(),
                parallel: effective.parallel,
                cache_type_k: effective.cache_type_k.clone(),
                cache_type_v: effective.cache_type_v.clone(),
                n_batch: effective.batch,
                n_ubatch: effective.ubatch,
                flash_attention: effective
                    .flash_attention
                    .unwrap_or(FlashAttentionType::Auto),
                local_source_required,
                profile: effective.derived_profile(),
            });
        }
        if let Some(mmproj) = &options.mmproj
            && let Some(primary) = specs.first_mut()
        {
            primary.mmproj_ref = Some(mmproj.clone());
        }
        return Ok(specs);
    }

    let default_model_path = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.hardware.as_ref())
        .and_then(|hardware| hardware.model_path.as_ref());
    if config.models.len() > 1 && default_model_path.is_some() {
        anyhow::bail!(
            "defaults.hardware.model_path cannot be used when multiple logical models are configured; set hardware.model_path on each model"
        );
    }

    for model in &config.models {
        let effective =
            effective_startup_model_config(&model.model, Some(model), config.defaults.as_ref());
        let local_source_required =
            skippy_local_source_required(model.skippy.as_ref(), default_skippy);
        let configured_model_path = model
            .hardware
            .as_ref()
            .and_then(|hardware| hardware.model_path.as_ref())
            .or(default_model_path);
        let alias = configured_server_alias(model, config.defaults.as_ref());
        let (model_ref, fallback_declared_ref) =
            if let Some(configured_path) = configured_model_path {
                let path = PathBuf::from(configured_path);
                if !path.is_absolute() {
                    anyhow::bail!(
                        "configured hardware.model_path for {} must be absolute: {}",
                        model.model,
                        path.display()
                    );
                }
                let metadata = std::fs::symlink_metadata(&path).with_context(|| {
                    format!(
                        "configured hardware.model_path for {} is unavailable: {}",
                        model.model,
                        path.display()
                    )
                })?;
                if metadata.file_type().is_symlink() || !metadata.is_file() {
                    anyhow::bail!(
                        "configured hardware.model_path for {} must be a non-symlink file: {}",
                        model.model,
                        path.display()
                    );
                }
                (path, Some(model.model.clone()))
            } else {
                (PathBuf::from(model.model.clone()), None)
            };
        if local_source_required {
            validate_local_required_source(&model_ref, &model.model)?;
        }
        specs.push(StartupModelSpec {
            model_ref,
            declared_ref: alias.or(fallback_declared_ref),
            config_model_id: Some(model.model.clone()),
            mmproj_ref: effective.mmproj.as_ref().map(PathBuf::from),
            ctx_size: options.ctx_size.or(effective.ctx_size),
            gpu_id: effective_startup_gpu_id(options, effective.gpu_id.as_deref()),
            cli_device_override: has_explicit_startup_device(options),
            resolve_pinned_gpu: true,
            parallel: effective.parallel,
            cache_type_k: effective.cache_type_k.clone(),
            cache_type_v: effective.cache_type_v.clone(),
            n_batch: effective.batch,
            n_ubatch: effective.ubatch,
            flash_attention: effective
                .flash_attention
                .unwrap_or(FlashAttentionType::Auto),
            local_source_required,
            profile: effective.derived_profile(),
        });
    }
    Ok(specs)
}

pub(super) async fn resolve_startup_models(
    specs: &[StartupModelSpec],
    _split: bool,
) -> Result<Vec<StartupModelPlan>> {
    resolve_startup_models_with_package_discovery(specs).await
}

/// Resolve startup models for the direct local serving topology.
///
/// The direct topology never falls back to a distributed layer package. It
/// resolves the requested model itself and later rejects the launch if that
/// complete model does not fit on the local machine.
pub(super) async fn resolve_local_model_only_startup_models(
    specs: &[StartupModelSpec],
) -> Result<Vec<StartupModelPlan>> {
    let mut plans = Vec::with_capacity(specs.len());
    for spec in specs {
        let resolved_path = resolve_pinned_local_file(&spec.model_ref, "model")?;
        let mmproj_path = spec
            .mmproj_ref
            .as_ref()
            .map(|path| resolve_pinned_local_file(path, "multimodal projector"))
            .transpose()?;
        let declared_ref = spec
            .declared_ref
            .clone()
            .unwrap_or_else(|| models::model_ref_for_path(&resolved_path));
        plans.push(StartupModelPlan {
            declared_ref,
            config_model_id: spec.config_model_id.clone(),
            resolved_path,
            preindexed_split_package: None,
            mmproj_path,
            ctx_size: spec.ctx_size,
            gpu_id: spec.gpu_id.clone(),
            pinned_gpu: None,
            parallel: spec.parallel,
            cache_type_k: spec.cache_type_k.clone(),
            cache_type_v: spec.cache_type_v.clone(),
            n_batch: spec.n_batch,
            n_ubatch: spec.n_ubatch,
            flash_attention: spec.flash_attention,
            local_source_required: spec.local_source_required,
            profile: spec.profile.clone(),
        });
    }
    Ok(plans)
}

fn resolve_pinned_local_file(path: &Path, label: &str) -> Result<PathBuf> {
    anyhow::ensure!(
        path.is_absolute(),
        "--local-model-only requires an absolute local {label} path: {}",
        path.display()
    );
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("pinned local {label} is unavailable: {}", path.display()))?;
    anyhow::ensure!(
        !metadata.file_type().is_symlink() && metadata.is_file(),
        "--local-model-only requires a non-symlink {label} file: {}",
        path.display()
    );
    path.canonicalize()
        .with_context(|| format!("canonicalize pinned local {label}: {}", path.display()))
}

async fn resolve_startup_models_with_package_discovery(
    specs: &[StartupModelSpec],
) -> Result<Vec<StartupModelPlan>> {
    let mut plans = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.local_source_required {
            let mut plan = resolve_local_required_startup_model(spec)?;
            let model_id = spec
                .config_model_id
                .as_deref()
                .or(spec.declared_ref.as_deref())
                .unwrap_or(plan.declared_ref.as_str())
                .to_string();
            let model_path = plan.resolved_path.clone();
            let identity = tokio::task::spawn_blocking(move || {
                skippy::synthetic_direct_gguf_package(&model_id, &model_path)
            })
            .await
            .context("join direct GGUF indexing task")??;
            if spec.config_model_id.is_none() && spec.declared_ref.is_none() {
                plan.declared_ref = direct_gguf_model_id(&identity);
            }
            plan.preindexed_split_package = Some(identity);
            plans.push(plan);
            continue;
        }
        let direct_local_gguf = is_direct_local_gguf_source(&spec.model_ref);
        let requested_ref = spec
            .config_model_id
            .clone()
            .or_else(|| spec.declared_ref.clone())
            .unwrap_or_else(|| spec.model_ref.to_string_lossy().into_owned());

        // Check the remote catalog for a pre-split layer package before
        // downloading a remote monolithic GGUF. Auto-split can decide to split
        // later, so layer-package discovery must not depend on `--split`.
        let requested_ref_for_catalog = requested_ref.clone();
        let model_ref_for_catalog = spec.model_ref.clone();
        let resolved_path = if let Some(package_ref) = tokio::task::spawn_blocking(move || {
            resolve_split_layer_package(&requested_ref_for_catalog, &model_ref_for_catalog)
        })
        .await
        .context("join resolve layer package task")?
        {
            PathBuf::from(package_ref)
        } else {
            resolve_model(&spec.model_ref).await?
        };

        let mmproj_path = match spec.mmproj_ref.as_ref() {
            Some(mmproj) => Some(resolve_model(mmproj).await?),
            None => None,
        };
        let preindexed_split_package = if direct_local_gguf {
            let model_id = spec
                .config_model_id
                .as_deref()
                .or(spec.declared_ref.as_deref())
                .unwrap_or(requested_ref.as_str())
                .to_string();
            let model_path = resolved_path.clone();
            Some(
                tokio::task::spawn_blocking(move || {
                    skippy::synthetic_direct_gguf_package(&model_id, &model_path)
                })
                .await
                .context("join direct GGUF indexing task")??,
            )
        } else {
            None
        };
        let declared_ref = if let Some(declared_ref) = &spec.declared_ref {
            declared_ref.clone()
        } else if spec.config_model_id.is_none()
            && let Some(identity) = preindexed_split_package.as_ref()
        {
            direct_gguf_model_id(identity)
        } else {
            find_remote_catalog_model_exact_blocking(requested_ref.clone())
                .await
                .map(|model| models::remote_catalog_model_ref(&model))
                .unwrap_or_else(|| {
                    // For hf:// layer package refs, use the requested ref as the model ref
                    // rather than trying to parse the hf:// URL as a filesystem path.
                    let path_str = resolved_path.to_string_lossy();
                    if path_str.starts_with("hf://") {
                        requested_ref.clone()
                    } else if resolved_path.join("model-package.json").is_file() {
                        // Layer package directory: read the canonical model_id from the manifest
                        // so that all nodes agree on the model name regardless of local path.
                        read_layer_package_model_id(&resolved_path)
                            .unwrap_or_else(|| models::model_ref_for_path(&resolved_path))
                    } else {
                        models::model_ref_for_path(&resolved_path)
                    }
                })
        };
        plans.push(StartupModelPlan {
            declared_ref,
            config_model_id: spec.config_model_id.clone(),
            resolved_path,
            preindexed_split_package,
            mmproj_path,
            ctx_size: spec.ctx_size,
            gpu_id: spec.gpu_id.clone(),
            pinned_gpu: None,
            parallel: spec.parallel,
            cache_type_k: spec.cache_type_k.clone(),
            cache_type_v: spec.cache_type_v.clone(),
            n_batch: spec.n_batch,
            n_ubatch: spec.n_ubatch,
            flash_attention: spec.flash_attention,
            local_source_required: false,
            profile: spec.profile.clone(),
        });
    }
    Ok(plans)
}

pub(super) fn is_direct_local_gguf_source(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

fn direct_gguf_model_id(identity: &skippy::SkippyPackageIdentity) -> String {
    format!("local-gguf/sha256-{}", identity.source_model_sha256)
}

/// Read the `model_id` field from a layer package's `model-package.json`.
pub(super) fn read_layer_package_model_id(package_dir: &Path) -> Option<String> {
    let manifest_path = package_dir.join("model-package.json");
    let contents = std::fs::read(&manifest_path).ok()?;
    let manifest: serde_json::Value = serde_json::from_slice(&contents).ok()?;
    manifest
        .get("model_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Check the remote catalog for a layer package matching the model.
/// Returns `Some("hf://meshllm/...")` or a local package dir if found, None otherwise.
pub(super) fn resolve_split_layer_package(model_query: &str, model_path: &Path) -> Option<String> {
    // Already an hf:// ref — use as-is
    let path_str = model_path.to_string_lossy();
    if path_str.starts_with("hf://") {
        return Some(path_str.to_string());
    }

    // Local directory with model-package.json — already a layer package on disk
    if model_path.join("model-package.json").is_file() {
        return Some(path_str.to_string());
    }

    // Existing local GGUFs should stay local. Layer-package lookup is only meant
    // to avoid remote monolithic downloads, not replace an explicit local file.
    if model_path.exists() {
        return None;
    }

    // Try remote catalog first for curated source-model metadata, then probe
    // Hugging Face directly for uncataloged package repos.
    match models::remote_catalog::ensure_catalog() {
        Ok(()) => {
            if let Some(package_ref) = models::remote_catalog::find_layer_package(model_query) {
                return Some(package_ref);
            }
        }
        Err(err) => tracing::debug!("remote catalog unavailable: {err:#}"),
    }
    models::remote_catalog::find_huggingface_layer_package(model_query)
}

pub(super) fn preflight_pinned_startup_models(
    config: &plugin::MeshConfig,
    specs: &[StartupModelSpec],
    plans: &mut [StartupModelPlan],
    binary_flavor: Option<backend::BinaryFlavor>,
    backend_probe: Option<&backend::BinaryBackendDeviceProbe>,
) -> Result<()> {
    if config.gpu.assignment != plugin::GpuAssignment::Pinned
        && plans.iter().all(|plan| plan.gpu_id.is_none())
    {
        return Ok(());
    }

    let binary_flavor = backend_probe
        .and_then(|probe| probe.flavor)
        .or(binary_flavor);
    let mut survey = hardware::query(pinned_startup_preflight_metrics());
    apply_backend_devices_for_flavor(&mut survey.gpus, binary_flavor);
    preflight_pinned_startup_models_with_gpus(config, specs, plans, &survey.gpus, backend_probe)
}

pub(super) fn apply_backend_devices_for_flavor(
    gpus: &mut [hardware::GpuFacts],
    binary_flavor: Option<backend::BinaryFlavor>,
) {
    let Some(binary_flavor) = binary_flavor else {
        return;
    };

    for gpu in gpus {
        gpu.backend_device = backend::backend_device_for_flavor(gpu.index, binary_flavor);
    }
}

pub(super) fn swarm_capture_observer_requested(options: &RuntimeOptions) -> bool {
    options.client
        && (options.swarm_capture.is_some()
            || std::env::var_os(crate::capture::SWARM_CAPTURE_ENV)
                .is_some_and(|value| !value.is_empty()))
}

pub(super) fn pinned_startup_preflight_metrics() -> &'static [hardware::Metric] {
    &[
        hardware::Metric::GpuName,
        hardware::Metric::GpuFacts,
        hardware::Metric::VramBytes,
        hardware::Metric::IsSoc,
    ]
}

pub(super) fn preflight_pinned_startup_models_with_gpus(
    config: &plugin::MeshConfig,
    specs: &[StartupModelSpec],
    plans: &mut [StartupModelPlan],
    gpus: &[hardware::GpuFacts],
    backend_probe: Option<&backend::BinaryBackendDeviceProbe>,
) -> Result<()> {
    anyhow::ensure!(
        specs.len() == plans.len(),
        "startup model preflight received mismatched specs/plans"
    );

    for (spec, plan) in specs.iter().zip(plans.iter_mut()) {
        let must_resolve_device =
            spec.resolve_pinned_gpu && config.gpu.assignment == plugin::GpuAssignment::Pinned;
        if !must_resolve_device && plan.gpu_id.is_none() {
            continue;
        }
        if is_cpu_startup_device(plan.gpu_id.as_deref()) {
            continue;
        }

        let resolved_gpu = resolve_requested_startup_device(
            plan.gpu_id.as_deref(),
            gpus,
            spec.cli_device_override,
        )
        .with_context(|| {
            format!(
                "startup model '{}' failed pinned GPU preflight",
                plan.declared_ref
            )
        })?;

        let stable_id = resolved_gpu.stable_id.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "startup model '{}' resolved pinned GPU at index {} without a stable_id",
                plan.declared_ref,
                resolved_gpu.index
            )
        })?;

        let backend_device = resolved_gpu
            .backend_device
            .clone()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "startup model '{}' resolved pinned GPU '{}' at index {} without a backend_device",
                    plan.declared_ref,
                    stable_id,
                    resolved_gpu.index
                )
            })
            .with_context(|| {
                format!(
                    "startup model '{}' failed pinned GPU preflight",
                    plan.declared_ref
                )
            })?;
        let backend_device = if let Some(probe) = backend_probe {
            backend::resolve_requested_device_from_available(
                &probe.available_devices,
                &probe.path,
                &backend_device,
            )
            .with_context(|| {
                format!(
                    "startup model '{}' failed pinned GPU preflight",
                    plan.declared_ref
                )
            })?
        } else {
            backend_device
        };

        plan.pinned_gpu = Some(StartupPinnedGpuTarget {
            index: resolved_gpu.index,
            stable_id,
            backend_device,
            vram_bytes: resolved_gpu.vram_bytes,
            reserved_bytes: resolved_gpu.reserved_bytes,
        });
    }

    Ok(())
}

/// Resolve a stable GPU ID or a concrete backend device name.
fn resolve_requested_startup_device<'a>(
    requested_device: Option<&str>,
    gpus: &'a [hardware::GpuFacts],
    allow_backend_device_name: bool,
) -> Result<&'a hardware::GpuFacts> {
    match hardware::resolve_pinned_gpu_strict(requested_device, gpus) {
        Ok(gpu) => Ok(gpu),
        Err(hardware::PinnedGpuResolverError::NonPinnableConfiguredId {
            configured_id, ..
        }) if allow_backend_device_name => {
            resolve_startup_backend_device_by_name(&configured_id, gpus)
        }
        Err(err) => Err(anyhow::Error::new(err)),
    }
}

fn is_cpu_startup_device(requested_device: Option<&str>) -> bool {
    requested_device.is_some_and(|device| device.trim().eq_ignore_ascii_case("CPU"))
}

/// Preserve CPU for the runtime resolver while bypassing GPU-only preflight.
pub(super) fn startup_device_override(requested_device: Option<&str>) -> Option<String> {
    requested_device
        .filter(|device| is_cpu_startup_device(Some(device)))
        .map(|_| "CPU".to_string())
}

fn resolve_startup_backend_device_by_name<'a>(
    requested_device: &str,
    gpus: &'a [hardware::GpuFacts],
) -> Result<&'a hardware::GpuFacts> {
    let matches: Vec<&hardware::GpuFacts> = gpus
        .iter()
        .filter(|gpu| {
            gpu.backend_device.as_deref().is_some_and(|backend_device| {
                backend::backend_device_names_match(backend_device, requested_device)
            })
        })
        .collect();

    match matches.as_slice() {
        [gpu] => Ok(gpu),
        [] => anyhow::bail!(
            "requested device '{requested_device}' did not match any detected GPU backend device. Available devices: {}",
            display_available_backend_devices(gpus)
        ),
        _ => anyhow::bail!(
            "requested device '{requested_device}' matched multiple detected GPU backend devices. Available devices: {}",
            display_available_backend_devices(gpus)
        ),
    }
}

fn display_available_backend_devices(gpus: &[hardware::GpuFacts]) -> String {
    let names: Vec<&str> = gpus
        .iter()
        .filter_map(|gpu| gpu.backend_device.as_deref())
        .collect();
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "configuration guidance policy remains covered by startup matrix tests"
    )
)]
pub(super) fn should_show_serve_config_help(
    _explicit_surface: Option<RuntimeSurface>,
    _options: &RuntimeOptions,
    _startup_specs: &[StartupModelSpec],
) -> bool {
    false
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "legacy size parsing remains covered by capacity compatibility tests"
    )
)]
pub(super) fn parse_size_str(s: &str) -> Option<u64> {
    let s = s.trim();
    if let Some(gb) = s.strip_suffix("GB") {
        gb.parse::<f64>().ok().map(|gb| (gb * 1e9) as u64)
    } else if let Some(mb) = s.strip_suffix("MB") {
        mb.parse::<f64>().ok().map(|mb| (mb * 1e6) as u64)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "capacity projection remains covered by startup planning tests"
    )
)]
pub(super) struct RuntimeModelCapacity {
    pub(super) required_bytes: u64,
    pub(super) fits: bool,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "path capacity projection remains covered by startup planning tests"
    )
)]
pub(super) fn runtime_model_capacity_for_path(
    model_path: &Path,
    vram_bytes: u64,
) -> RuntimeModelCapacity {
    let model_bytes = election::total_model_bytes(model_path);
    let required_bytes = runtime_model_required_bytes(model_bytes);
    RuntimeModelCapacity {
        required_bytes,
        fits: model_bytes == 0 || model_fits_runtime_capacity(model_bytes, vram_bytes),
    }
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "reference capacity projection remains covered by startup planning tests"
    )
)]
pub(super) fn runtime_model_capacity_for_ref(model: &str, vram_bytes: u64) -> RuntimeModelCapacity {
    let model_path = models::find_model_path(model);
    runtime_model_capacity_for_path(&model_path, vram_bytes)
}

pub(super) async fn find_remote_catalog_model_exact_blocking(
    query: String,
) -> Option<models::remote_catalog::RemoteCatalogModel> {
    tokio::task::spawn_blocking(move || models::find_remote_catalog_model_exact(&query))
        .await
        .ok()
        .flatten()
}
