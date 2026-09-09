mod commands;
mod logging_help;
mod normalization;
mod runtime_surface_help;
mod validation;

pub use commands::{
    AuthCommand, BinaryFlavor, Cli, Command, ConfigCommand, DiscoveryScope, DoctorCommand,
    GpuCommand, KvCacheCommand, MeshDiscoveryMode, MeshGuardrailCliMode, PluginCommand,
    SkillAgentArg, SkillCommand, TrustCommand, TrustPolicy,
};
pub use logging_help::logging_help;
pub use normalization::{
    NormalizedRuntimeArgs, RuntimeSurface, legacy_runtime_surface_warning,
    normalize_runtime_surface_args,
};
pub use runtime_surface_help::runtime_surface_help;
pub use validation::validate_discovery_mode_args;

#[cfg(test)]
mod setup_tests;

#[cfg(test)]
mod uninstall_tests;

#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
use clap::Parser;

#[cfg(test)]
use crate::runtime::RuntimeCommand;

#[cfg(test)]
pub fn assert_mesh_requirements_docs_examples_parse() {
    let unrestricted_args =
        normalize_runtime_surface_args(["mesh-llm", "serve", "--model", "Qwen3-8B-Q4_K_M"]);
    let unrestricted = Cli::parse_from(unrestricted_args.normalized.clone());
    assert!(unrestricted.command.is_none());
    assert_eq!(unrestricted.model, vec![PathBuf::from("Qwen3-8B-Q4_K_M")]);
    assert!(!unrestricted.publish);

    let signed_public_args = normalize_runtime_surface_args([
        "mesh-llm",
        "serve",
        "--model",
        "Qwen3-8B-Q4_K_M",
        "--publish",
        "--require-release-attestation",
        "--release-signer-key",
        "ed25519:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "--owner-key",
        "~/.mesh-llm/owner-keystore.json",
        "--owner-required",
        "--trust-policy",
        "require-owned",
        "--node-label",
        "lab-a",
    ]);
    let signed_public = Cli::parse_from(signed_public_args.normalized.clone());
    assert!(signed_public.command.is_none());
    assert_eq!(signed_public.model, vec![PathBuf::from("Qwen3-8B-Q4_K_M")]);
    assert!(signed_public.publish);
    assert!(signed_public.require_release_attestation);
    assert_eq!(
        signed_public.release_signer_key,
        vec![
            "ed25519:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
        ]
    );
    assert_eq!(
        signed_public.owner_key,
        Some(PathBuf::from("~/.mesh-llm/owner-keystore.json"))
    );
    assert!(signed_public.owner_required);
    assert_eq!(signed_public.trust_policy, Some(TrustPolicy::RequireOwned));
    assert_eq!(signed_public.node_label, Some("lab-a".to_string()));

    let signed_bootstrap_args =
        normalize_runtime_surface_args(["mesh-llm", "serve", "--join", "signed-bootstrap-token"]);
    let signed_bootstrap = Cli::parse_from(signed_bootstrap_args.normalized.clone());
    assert!(signed_bootstrap.command.is_none());
    assert_eq!(
        signed_bootstrap.join,
        vec!["signed-bootstrap-token".to_string()]
    );

    let runtime_bootstrap = Cli::parse_from(["mesh-llm", "runtime", "bootstrap", "--port", "3131"]);
    match runtime_bootstrap.command.expect("runtime command expected") {
        Command::Runtime {
            command: Some(RuntimeCommand::Bootstrap { port, json }),
        } => {
            assert_eq!(port, 3131);
            assert!(!json);
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{Cli, Command, KvCacheCommand};
    use crate::models::ModelsCommand;
    use clap::Parser;

    #[test]
    fn mesh_requirements_docs_examples_parse() {
        super::assert_mesh_requirements_docs_examples_parse();
    }

    #[test]
    fn split_topology_lock_requires_split_mode() {
        let error = Cli::try_parse_from([
            "mesh-llm",
            "--model",
            "model.gguf",
            "--split-topology-lock",
            "topology.json",
        ])
        .expect_err("topology lock without --split should fail");

        assert!(error.to_string().contains("--split"));
    }

    #[test]
    fn split_topology_lock_parses_with_split_mode() {
        let cli = Cli::try_parse_from([
            "mesh-llm",
            "--model",
            "model.gguf",
            "--split",
            "--split-topology-lock",
            "topology.json",
        ])
        .expect("locked split CLI should parse");

        assert!(cli.split);
        assert_eq!(
            cli.split_topology_lock,
            Some(std::path::PathBuf::from("topology.json"))
        );
    }

    #[test]
    fn disk_cache_runtime_flags_parse_without_rewriting_config() {
        let normalized = super::normalize_runtime_surface_args(
            [
                "mesh-llm",
                "serve",
                "--kv-cache-disk",
                "32GiB",
                "--kv-cache-disk-dir",
                "/fast/mesh-kv",
                "--kv-cache-min-free",
                "16GiB",
            ]
            .into_iter()
            .map(std::ffi::OsString::from)
            .collect::<Vec<_>>(),
        );
        let cli = Cli::try_parse_from(normalized.normalized)
            .expect("disk cache runtime flags should parse");

        assert_eq!(cli.kv_cache_disk.as_deref(), Some("32GiB"));
        assert_eq!(
            cli.kv_cache_disk_dir,
            Some(std::path::PathBuf::from("/fast/mesh-kv"))
        );
        assert_eq!(cli.kv_cache_min_free.as_deref(), Some("16GiB"));
    }

    #[test]
    fn kv_cache_lifecycle_commands_parse_exact_model_identity_and_confirmation() {
        let cli = Cli::try_parse_from([
            "mesh-llm",
            "kv-cache",
            "prune",
            "--target",
            "16GiB",
            "--model-identity",
            "blake3:model",
            "--yes",
            "--endpoint",
            "node-a",
            "--endpoint",
            "node-b",
            "--json",
        ])
        .expect("kv-cache prune should parse");
        match cli.command {
            Some(Command::KvCache {
                command:
                    KvCacheCommand::Prune {
                        target,
                        model_identity,
                        yes,
                        endpoints,
                        json,
                        ..
                    },
            }) => {
                assert_eq!(target.as_deref(), Some("16GiB"));
                assert_eq!(model_identity.as_deref(), Some("blake3:model"));
                assert!(yes);
                assert_eq!(endpoints, vec!["node-a", "node-b"]);
                assert!(json);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn models_package_parses_experimental_publication() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "models",
            "package",
            "unsloth/inkling-GGUF:UD-Q2_K_XL",
            "--experimental",
            "--dry-run",
        ]);

        match cli.command.expect("models command expected") {
            Command::Models {
                command:
                    ModelsCommand::Package {
                        source_repo: Some(source_repo),
                        experimental: true,
                        dry_run: true,
                        ..
                    },
            } => assert_eq!(source_repo, "unsloth/inkling-GGUF:UD-Q2_K_XL"),
            other => panic!("unexpected command: {other:?}"),
        }
    }
}
