use super::{SummaryAssembly, command_summary};
use clap::Parser;
use mesh_llm_events::CliCommandSummary;

fn parsed_summary(args: &[&str]) -> String {
    let cli = mesh_llm_cli::Cli::parse_from(args);
    let Some(command) = cli.command.as_ref() else {
        return String::new();
    };
    match command_summary(command) {
        Some(summary) => summary.as_str().to_owned(),
        None => String::new(),
    }
}

#[test]
fn representative_command_summaries_survive_strict_sanitization() {
    let commands: &[&[&str]] = &[
        &["mesh-llm", "setup"],
        &["mesh-llm", "uninstall", "--yes"],
        &["mesh-llm", "gpus", "detect"],
        &["mesh-llm", "discover", "--auto"],
        &["mesh-llm", "goose"],
        &["mesh-llm", "plugins", "list"],
        &["mesh-llm", "config", "validate"],
        &[
            "mesh-llm",
            "doctor",
            "split",
            "--model-ref",
            "private/model",
        ],
        &["mesh-llm", "models", "recommended"],
        &["mesh-llm", "runtime", "bootstrap"],
        &[
            "mesh-llm",
            "benchmark",
            "import-prompts",
            "--source",
            "mt-bench",
            "--output",
            "private.jsonl",
        ],
        &["mesh-llm", "model-prepare", "--list"],
        &["mesh-llm", "auth", "status"],
    ];

    for args in commands {
        let summary = parsed_summary(args);
        let sanitized = CliCommandSummary::sanitize(&summary)
            .unwrap_or_else(|| panic!("producer emitted unsanitized summary: {summary}"));
        assert_eq!(sanitized.as_str(), summary);
    }
}

#[test]
fn over_limit_producer_summary_is_omitted_without_fallback() {
    let mut assembly = SummaryAssembly::new("mesh-llm");
    for _ in 0..32 {
        assembly.redact("name", true);
    }

    assert!(assembly.finish().is_none());
}

#[test]
fn uninstall_boolean_flags_survive_producer_sanitization() {
    let summaries = [
        parsed_summary(&[
            "mesh-llm",
            "uninstall",
            "--dry-run",
            "--yes",
            "--keep-cache",
            "--keep-service-files",
            "--keep-config",
            "--json",
            "--verbose",
        ]),
        parsed_summary(&["mesh-llm", "uninstall", "--purge-config"]),
    ];

    for summary in summaries {
        assert!(!summary.is_empty());
        let sanitized = CliCommandSummary::sanitize(&summary)
            .unwrap_or_else(|| panic!("producer omitted valid summary: {summary}"));
        assert_eq!(sanitized.as_str(), summary);
    }
}

#[test]
fn command_summary_retains_static_prefix_nested_names_and_redacts_plugin_values() {
    let summary = parsed_summary(&[
        "mesh-llm",
        "plugins",
        "install",
        "https://user:secret@example.test/plugin?token=query-secret",
    ]);
    assert_eq!(summary, "mesh-llm plugins install reference [REDACTED]");
    assert!(!summary.contains("secret"));
    assert!(!summary.contains("query-secret"));
}

#[test]
fn command_summary_covers_plugin_config_and_doctor_values_without_defaults() {
    assert_eq!(
        parsed_summary(&["mesh-llm", "plugins", "update", "private-plugin"]),
        "mesh-llm plugins update name [REDACTED]"
    );
    assert_eq!(
        parsed_summary(&[
            "mesh-llm",
            "config",
            "validate",
            "--config-path",
            "/private/config.toml"
        ]),
        "mesh-llm config validate --config-path [REDACTED]"
    );
    assert_eq!(
        parsed_summary(&[
            "mesh-llm",
            "doctor",
            "split",
            "--model-ref",
            "private/model",
            "--output-dir",
            "/private/report"
        ]),
        "mesh-llm doctor split --model-ref [REDACTED] --output-dir [REDACTED]"
    );
}

#[test]
fn command_summary_records_kv_cache_subcommands_and_redacts_values() {
    assert_eq!(
        parsed_summary(&[
            "mesh-llm",
            "kv-cache",
            "status",
            "--endpoint",
            "secret-endpoint",
            "--port",
            "4444",
            "--json",
        ]),
        "mesh-llm kv-cache status --json --port 4444 --endpoint [REDACTED]"
    );
    assert_eq!(
        parsed_summary(&[
            "mesh-llm",
            "kv-cache",
            "prune",
            "--target",
            "4GiB",
            "--model-identity",
            "private-model",
            "--yes",
        ]),
        "mesh-llm kv-cache prune --yes --target [REDACTED] --model-identity [REDACTED]"
    );
    assert_eq!(
        parsed_summary(&["mesh-llm", "kv-cache", "clear", "--yes"]),
        "mesh-llm kv-cache clear --yes"
    );
}

#[test]
fn command_summary_covers_auth_trust_and_nested_command_families() {
    let auth = parsed_summary(&[
        "mesh-llm",
        "auth",
        "rotate-node",
        "--owner-key",
        "/private/owner.json",
        "--node-key",
        "/private/node.key",
        "--reason",
        "private reason",
        "--trust-store",
        "/private/trust.json",
    ]);
    for marker in [
        "mesh-llm auth rotate-node",
        "--owner-key [REDACTED]",
        "--node-key [REDACTED]",
        "--reason [REDACTED]",
        "--trust-store [REDACTED]",
    ] {
        assert!(auth.contains(marker));
    }
    assert_eq!(
        parsed_summary(&[
            "mesh-llm",
            "auth",
            "trust",
            "add",
            "private-owner-id",
            "--trust-store",
            "/private/trust.json"
        ]),
        "mesh-llm auth trust add owner_id [REDACTED] --trust-store [REDACTED]"
    );
    assert!(
        parsed_summary(&["mesh-llm", "models", "show", "private/model"])
            .contains("mesh-llm models show model [REDACTED]")
    );
    assert!(
        parsed_summary(&["mesh-llm", "runtime", "load", "private/model"])
            .contains("mesh-llm runtime load name [REDACTED]")
    );
    assert!(
        parsed_summary(&["mesh-llm", "benchmark", "tune", "--model", "private/model"])
            .contains("mesh-llm benchmark tune --model [REDACTED]")
    );
}

#[test]
fn command_summary_retains_explicit_runtime_install_without_runtime_value() {
    let summary = parsed_summary(&["mesh-llm", "runtime", "install", "private-runtime-value"]);

    assert_eq!(summary, "mesh-llm runtime install runtime_ref [REDACTED]");
    assert!(!summary.contains("private-runtime-value"));
    let sanitized = CliCommandSummary::sanitize(&summary)
        .unwrap_or_else(|| panic!("producer omitted valid summary: {summary}"));
    assert_eq!(sanitized.as_str(), summary);
}

#[test]
fn command_summary_omits_default_ports_and_retains_explicit_safe_port() {
    assert_eq!(
        parsed_summary(&["mesh-llm", "load", "private/model"]),
        "mesh-llm load name [REDACTED]"
    );
    assert_eq!(
        parsed_summary(&["mesh-llm", "load", "private/model", "--port", "41731"]),
        "mesh-llm load --port 41731 name [REDACTED]"
    );
}

#[test]
fn command_summary_uses_guardrail_mode_contract_tokens() {
    for mode in ["disabled", "metrics", "enforce"] {
        assert_eq!(
            parsed_summary(&["mesh-llm", "runtime", "guardrails", "--mode", mode]),
            format!("mesh-llm runtime guardrails --mode {mode}")
        );
    }
}

#[test]
fn command_summary_deduplicates_parent_and_child_json_flags() {
    assert_eq!(
        parsed_summary(&[
            "mesh-llm",
            "doctor",
            "--json",
            "split",
            "--model-ref",
            "private/model",
            "--json",
        ]),
        "mesh-llm doctor split --json --model-ref [REDACTED]"
    );
    assert_eq!(
        parsed_summary(&["mesh-llm", "gpus", "--json", "detect", "--json"]),
        "mesh-llm gpus detect --json"
    );
    assert_eq!(
        parsed_summary(&[
            "mesh-llm",
            "gpus",
            "--json",
            "run-benchmark",
            "--backend",
            "cuda",
        ]),
        "mesh-llm gpus run-benchmark --backend cuda --json"
    );
}

#[test]
fn summary_assembly_deduplicates_repeated_flags() {
    let mut assembly = SummaryAssembly::new("mesh-llm models list");
    assembly.flag("json", true);
    assembly.flag("json", true);

    assert_eq!(
        assembly.finish().map(|summary| summary.as_str().to_owned()),
        Some("mesh-llm models list --json".to_owned())
    );
}

#[test]
fn command_summary_redacts_non_default_model_search_sort() {
    let summary = parsed_summary(&[
        "mesh-llm",
        "models",
        "search",
        "private-query",
        "--sort",
        "downloads",
    ]);
    assert_eq!(
        summary,
        "mesh-llm models search query [REDACTED] --sort [REDACTED]"
    );
    let sanitized = CliCommandSummary::sanitize(&summary)
        .unwrap_or_else(|| panic!("producer omitted valid summary: {summary}"));
    assert_eq!(sanitized.as_str(), summary);
}

#[test]
fn command_summary_redacts_spec_draft_tune_values() {
    assert_eq!(
        parsed_summary(&[
            "mesh-llm",
            "benchmark",
            "tune",
            "--spec-draft-acceptance-threshold",
            "0.8125,0.9375",
        ]),
        "mesh-llm benchmark tune --spec-draft-acceptance-threshold [REDACTED]"
    );
    assert_eq!(
        parsed_summary(&[
            "mesh-llm",
            "benchmark",
            "tune",
            "--spec-draft-split-probability",
            "0.125,0.375",
        ]),
        "mesh-llm benchmark tune --spec-draft-split-probability [REDACTED]"
    );
    assert_eq!(
        parsed_summary(&["mesh-llm", "benchmark", "tune"]),
        "mesh-llm benchmark tune"
    );
}

#[test]
fn command_summary_redacts_external_argv_without_global_cli_context() {
    assert_eq!(
        parsed_summary(&["mesh-llm", "private-plugin", "--api-key", "private-key"]),
        "mesh-llm external-plugin argv [REDACTED]"
    );
    let cli = mesh_llm_cli::Cli::parse_from([
        "mesh-llm",
        "--join",
        "private-invite-token",
        "--relay-auth",
        "https://relay.example/?token=private-token=padding",
        "load",
        "private/model",
    ]);
    let Some(command) = cli.command.as_ref() else {
        return;
    };
    let summary = command_summary(command)
        .map(|value| value.as_str().to_owned())
        .unwrap_or_default();
    assert_eq!(summary, "mesh-llm load name [REDACTED]");
}
