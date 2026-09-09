#![recursion_limit = "256"]

use std::sync::Arc;
use std::time::Duration;
use std::{ffi::OsString, fmt, path::PathBuf};

use clap::{CommandFactory, Parser};

mod commands;

pub use mesh_llm_host_runtime::*;

pub async fn run_main() -> i32 {
    match run_cli_entrypoint().await {
        Ok(()) => 0,
        Err(err) => {
            if let Some(exit) = err.downcast_ref::<CliParseExit>() {
                return exit.0;
            }
            let _ = mesh_llm_tui::emit_fatal_error(&err);
            tokio::time::sleep(Duration::from_millis(50)).await;
            1
        }
    }
}

async fn run_cli_entrypoint() -> anyhow::Result<()> {
    let raw_args: Vec<_> = std::env::args_os().collect();
    if maybe_print_binary_help(&raw_args) {
        emit_early_cli_process_event(
            &raw_args,
            mesh_llm_events::CliCommandFamily::Process,
            mesh_llm_events::CliCommandOutcome::Completed,
        )
        .await;
        return Ok(());
    }

    let normalized_args = mesh_llm_cli::normalize_runtime_surface_args(raw_args);
    let cli = match mesh_llm_cli::Cli::try_parse_from(normalized_args.normalized.clone()) {
        Ok(cli) => cli,
        Err(error) => {
            let outcome = parse_exit_outcome(&error);
            let family = if error.kind() == clap::error::ErrorKind::DisplayVersion {
                mesh_llm_events::CliCommandFamily::Process
            } else {
                parse_failure_family(
                    &normalized_args.normalized,
                    normalized_args.explicit_surface,
                )
            };
            emit_early_cli_process_event(&normalized_args.normalized, family, outcome).await;
            let exit_code = error.exit_code();
            let _ = error.print();
            return Err(anyhow::Error::new(CliParseExit(exit_code)));
        }
    };
    let warning = mesh_llm_cli::legacy_runtime_surface_warning(
        &cli,
        &normalized_args.original,
        normalized_args.explicit_surface,
    );
    let explicit_surface = normalized_args
        .explicit_surface
        .or(match cli.command.as_ref() {
            Some(mesh_llm_cli::Command::Serve) => Some(mesh_llm_cli::RuntimeSurface::Serve),
            Some(mesh_llm_cli::Command::Client) => Some(mesh_llm_cli::RuntimeSurface::Client),
            _ => None,
        })
        .map(map_runtime_surface);

    // Command lifecycle events are terminal noise for one-shot commands unless
    // the user asked for verbose output with --debug. The durable audit bridge
    // installed below is unaffected by this toggle.
    mesh_llm_events::set_cli_command_event_verbose(cli.debug);

    if cli.command.as_ref().is_some_and(|c| {
        !matches!(
            c,
            mesh_llm_cli::Command::Serve | mesh_llm_cli::Command::Client
        )
    }) {
        // Install the durable audit bridge before command dispatch. This
        // performs only config-backed logging setup; one-shot commands do not
        // need a native runtime, a model, or serving infrastructure.
        let logging_initialized =
            mesh_llm_host_runtime::initialize_logging_for_cli(cli.config.as_deref())
                .await
                .is_ok();
        if logging_initialized {
            install_cli_operational_audit_bridge();
        } else {
            // Logging is fail-open for one-shot commands. A missing or invalid
            // logging config must not turn an otherwise independent command
            // such as `gpus` into a startup failure.
            mesh_llm_commands::operational_logging::clear_cli_operational_audit_bridge();
        }

        // Drain whether the one-shot command succeeds or fails: the
        // dispatcher has already emitted its static terminal audit outcome in
        // either case.
        let command_dispatch = commands::dispatch(&cli).await;
        if logging_initialized {
            let _ = mesh_llm_host_runtime::shutdown_logging_for_one_shot_cli().await;
        }
        mesh_llm_commands::operational_logging::clear_cli_operational_audit_bridge();
        if command_dispatch? {
            return Ok(());
        }
    }

    let options = runtime_options_from_cli(cli);
    mesh_llm_host_runtime::initialize_host_runtime_for_options(&options).await?;
    install_cli_operational_audit_bridge();
    mesh_llm_tui::output::OutputManager::init_global(
        options.log_format,
        mesh_llm_host_runtime::console_session_mode_for_runtime_surface(explicit_surface),
    );
    mesh_llm_tui::install_terminal_panic_hook();

    mesh_llm_host_runtime::run_runtime_initialized(options, explicit_surface, warning).await
}

async fn emit_early_cli_process_event(
    args: &[OsString],
    family: mesh_llm_events::CliCommandFamily,
    outcome: mesh_llm_events::CliCommandOutcome,
) {
    // Parse has not succeeded yet, so read --debug straight from argv to keep
    // early process events consistent with the --debug presentation toggle.
    mesh_llm_events::set_cli_command_event_verbose(args.iter().any(|arg| arg == "--debug"));
    let config_path = config_path_from_args(args);
    let logging_initialized =
        mesh_llm_host_runtime::initialize_logging_for_cli(config_path.as_deref())
            .await
            .is_ok();
    if logging_initialized {
        install_cli_operational_audit_bridge();
    }
    mesh_llm_commands::operational_logging::emit_cli_process_event(family, outcome);
    if logging_initialized {
        let _ = mesh_llm_host_runtime::shutdown_logging_for_one_shot_cli().await;
    }
    mesh_llm_commands::operational_logging::clear_cli_operational_audit_bridge();
}

fn parse_exit_outcome(error: &clap::Error) -> mesh_llm_events::CliCommandOutcome {
    match error.kind() {
        clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => {
            mesh_llm_events::CliCommandOutcome::Completed
        }
        _ => mesh_llm_events::CliCommandOutcome::ParseFailed,
    }
}

/// Bridge command boundary emissions to the installed logging runtime so the
/// static command outcome codes become durable audit records. Callers install
/// it after either logging-only one-shot initialization or normal host/client
/// initialization, before any command/runtime surface can emit boundaries.
fn install_cli_operational_audit_bridge() {
    use mesh_llm_host_runtime::{
        OperationalAuditContext, OperationalAuditRecord, OperationalAuditSeverity,
        OperationalAuditSubjectKind,
    };

    let bridge: mesh_llm_commands::operational_logging::CliOperationalAuditBridge = Arc::new(
        |family: mesh_llm_events::CliCommandFamily,
         outcome: mesh_llm_events::CliCommandOutcome,
         summary: Option<mesh_llm_events::CliCommandSummary>| {
            let Some(state) = mesh_llm_host_runtime::logging_runtime_state() else {
                return;
            };
            let severity = match outcome {
                mesh_llm_events::CliCommandOutcome::Started
                | mesh_llm_events::CliCommandOutcome::Completed => OperationalAuditSeverity::Info,
                mesh_llm_events::CliCommandOutcome::Failed
                | mesh_llm_events::CliCommandOutcome::Rejected
                | mesh_llm_events::CliCommandOutcome::ParseFailed => {
                    OperationalAuditSeverity::Warning
                }
            };
            let record = OperationalAuditRecord::builder("cli", outcome.code())
                .severity(severity)
                .build()
                .with_context(
                    OperationalAuditContext::new()
                        .subject(OperationalAuditSubjectKind::CliCommand, family.as_str())
                        .outcome(outcome.as_str())
                        .command_summary(summary.as_ref().map_or("", |summary| summary.as_str())),
                );
            let _ = state.write_operational_audit(record);
        },
    );
    mesh_llm_commands::operational_logging::install_cli_operational_audit_bridge(bridge);
}

#[derive(Debug)]
struct CliParseExit(i32);

impl fmt::Display for CliParseExit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("command-line parsing failed")
    }
}

impl std::error::Error for CliParseExit {}

fn config_path_from_args(args: &[OsString]) -> Option<PathBuf> {
    for (index, argument) in args.iter().enumerate().skip(1) {
        if argument == "--config" {
            return args.get(index + 1).cloned().map(PathBuf::from);
        }
        if let Some(value) = argument
            .to_str()
            .and_then(|value| value.strip_prefix("--config="))
            && !value.is_empty()
        {
            return Some(PathBuf::from(value));
        }
    }
    None
}

fn parse_failure_family(
    args: &[OsString],
    explicit_surface: Option<mesh_llm_cli::RuntimeSurface>,
) -> mesh_llm_events::CliCommandFamily {
    use mesh_llm_events::CliCommandFamily;

    if explicit_surface.is_some() {
        return CliCommandFamily::Runtime;
    }
    let value_taking_options: Vec<String> = mesh_llm_cli::Cli::command()
        .get_arguments()
        .filter(|argument| {
            matches!(
                argument.get_action(),
                clap::ArgAction::Set | clap::ArgAction::Append
            )
        })
        .flat_map(|argument| {
            argument
                .get_long()
                .map(|long| format!("--{long}"))
                .into_iter()
                .chain(argument.get_short().map(|short| format!("-{short}")))
        })
        .collect();
    let is_value_taking_option =
        |argument: &str| value_taking_options.iter().any(|option| option == argument);
    let mut skip_next = false;
    for argument in args.iter().skip(1).filter_map(|argument| argument.to_str()) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if let Some((option, _value)) = argument.split_once('=')
            && is_value_taking_option(option)
        {
            continue;
        }
        if is_value_taking_option(argument) {
            skip_next = true;
            continue;
        }
        if argument.starts_with('-') {
            continue;
        }
        let family = match argument {
            "serve" | "client" => Some(CliCommandFamily::Runtime),
            "models" | "download" | "model-prepare" | "model-package" => {
                Some(CliCommandFamily::Models)
            }
            "update" | "setup" | "uninstall" => Some(CliCommandFamily::Installation),
            "gpus" | "gpu" => Some(CliCommandFamily::Hardware),
            "runtime" | "load" | "unload" | "drop" | "status" | "stop" => {
                Some(CliCommandFamily::Runtime)
            }
            "config" => Some(CliCommandFamily::Configuration),
            "doctor" => Some(CliCommandFamily::Diagnostics),
            "discover" => Some(CliCommandFamily::Discovery),
            "rotate-key" | "auth" => Some(CliCommandFamily::Identity),
            "goose" | "claude" | "pi" | "opencode" => Some(CliCommandFamily::Agent),
            "plugins" | "plugin" => Some(CliCommandFamily::Plugin),
            "skills" => Some(CliCommandFamily::Skills),
            "benchmark" => Some(CliCommandFamily::Benchmark),
            _ => None,
        };
        if let Some(family) = family {
            return family;
        }
    }
    CliCommandFamily::Unknown
}

fn maybe_print_binary_help(args: &[OsString]) -> bool {
    if binary_help_request(args.iter().cloned()) {
        mesh_llm_cli::Cli::command().print_help().ok();
        return true;
    }
    if let Some(surface) = runtime_surface_help_request(args.iter().cloned()) {
        print!("{}", mesh_llm_cli::parser::runtime_surface_help(surface));
        return true;
    }
    if args.iter().any(|arg| arg == "--help-advanced") {
        print_advanced_help();
        return true;
    }
    false
}

fn binary_help_request<I>(args: I) -> bool
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let args: Vec<_> = args.into_iter().collect();
    match args.as_slice() {
        [_program] => true,
        [_program, arg] => arg == "--help" || arg == "-h",
        [_program, help, arg] => help == "help" && (arg == "--help" || arg == "-h"),
        _ => false,
    }
}

fn runtime_surface_help_request<I>(args: I) -> Option<mesh_llm_cli::RuntimeSurface>
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let args: Vec<_> = args.into_iter().collect();
    let has_help_flag = args.iter().any(|arg| arg == "--help" || arg == "-h");

    let value_taking_options: Vec<String> = mesh_llm_cli::Cli::command()
        .get_arguments()
        .filter(|argument| {
            matches!(
                argument.get_action(),
                clap::ArgAction::Set | clap::ArgAction::Append
            )
        })
        .flat_map(|argument| {
            argument
                .get_long()
                .map(|long| format!("--{long}"))
                .into_iter()
                .chain(argument.get_short().map(|short| format!("-{short}")))
        })
        .collect();
    let is_value_taking_option =
        |argument: &str| value_taking_options.iter().any(|option| option == argument);

    let mut positional: Vec<&str> = Vec::new();
    let mut skip_next = false;
    for arg in args.iter().skip(1).filter_map(|arg| arg.to_str()) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if let Some((option, _value)) = arg.split_once('=')
            && is_value_taking_option(option)
        {
            continue;
        }
        if is_value_taking_option(arg) {
            skip_next = true;
            continue;
        }
        if arg.starts_with('-') {
            continue;
        }
        positional.push(arg);
    }

    match positional.as_slice() {
        ["help", "serve", ..] => Some(mesh_llm_cli::RuntimeSurface::Serve),
        ["help", "client", ..] => Some(mesh_llm_cli::RuntimeSurface::Client),
        ["serve", ..] if has_help_flag => Some(mesh_llm_cli::RuntimeSurface::Serve),
        ["client", ..] if has_help_flag => Some(mesh_llm_cli::RuntimeSurface::Client),
        [] if has_help_flag => mesh_llm_cli::normalize_runtime_surface_args(args).explicit_surface,
        _ => None,
    }
}

fn print_advanced_help() {
    print!("{}", advanced_help_text());
    print!("{}", mesh_llm_cli::parser::logging_help());
    eprintln!();
}

fn advanced_help_text() -> String {
    let mut command = mesh_llm_cli::Cli::command();
    let args: Vec<clap::Id> = command
        .get_arguments()
        .map(|arg| arg.get_id().clone())
        .collect();
    for id in args {
        command = command.mut_arg(id, |arg| arg.hide(false));
    }
    let subcommands: Vec<String> = command
        .get_subcommands()
        .map(|subcommand| subcommand.get_name().to_string())
        .collect();
    for name in subcommands {
        command = command.mut_subcommand(name, |subcommand| subcommand.hide(false));
    }
    let mut bytes = Vec::new();
    command.write_long_help(&mut bytes).ok();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn runtime_help_text() -> Option<String> {
    let mut bytes = Vec::new();
    mesh_llm_cli::Cli::command()
        .write_help(&mut bytes)
        .ok()
        .and_then(|()| String::from_utf8(bytes).ok())
}

fn runtime_options_from_cli(cli: mesh_llm_cli::Cli) -> mesh_llm_host_runtime::RuntimeOptions {
    let speculative_overrides = speculative_overrides_from_cli(&cli);
    mesh_llm_host_runtime::RuntimeOptions {
        log_format: cli.log_format,
        debug: cli.debug,
        skippy_metrics_otlp_grpc: cli.skippy_metrics_otlp_grpc,
        mesh_guardrails: map_mesh_guardrail_mode(cli.mesh_guardrails),
        help_text: runtime_help_text(),
        join: cli.join,
        discover: cli.discover,
        auto: cli.auto,
        mesh_discovery_mode: map_mesh_discovery_mode(cli.mesh_discovery_mode),
        model: cli.model,
        gguf: cli.gguf,
        mmproj: cli.mmproj,
        checkpoint_quantization: cli.checkpoint_quantization,
        checkpoint_imatrix: cli.checkpoint_imatrix,
        port: cli.port,
        local_model_only: cli.local_model_only,
        native_serving_plugin: cli.native_serving_plugin,
        native_serving_plugin_config: cli.native_serving_plugin_config,
        native_serving_plugin_state: cli.native_serving_plugin_state,
        native_serving_plugin_deadline_ms: cli.native_serving_plugin_deadline_ms,
        client: cli.client,
        console: cli.console,
        headless: cli.headless,
        swarm_capture: cli.swarm_capture,
        publish: cli.publish,
        mesh_name: cli.mesh_name,
        region: cli.region,
        min_node_version: cli.min_node_version,
        max_node_version: cli.max_node_version,
        min_protocol_version: cli.min_protocol_version,
        max_protocol_version: cli.max_protocol_version,
        require_release_attestation: cli.require_release_attestation,
        release_signer_key: cli.release_signer_key,
        name: cli.name,
        plugin: cli.plugin,
        auto_update: cli.auto_update,
        command_is_update: matches!(cli.command, Some(mesh_llm_cli::Command::Update { .. })),
        command_uses_machine_output: command_uses_machine_output(cli.command.as_ref()),
        draft: cli.draft,
        draft_max: cli.draft_max,
        no_draft: cli.no_draft,
        speculative_overrides,
        split: cli.split,
        split_topology_lock: cli.split_topology_lock,
        ctx_size: cli.ctx_size,
        max_vram: cli.max_vram,
        no_enumerate_host: cli.no_enumerate_host,
        bin_dir: cli.bin_dir,
        llama_flavor: cli.llama_flavor.map(map_binary_flavor),
        device: cli.device,
        tensor_split: cli.tensor_split,
        relay: cli.relay,
        relay_auth: cli.relay_auth,
        disable_iroh_relays: cli.disable_iroh_relays,
        bind_port: cli.bind_port,
        bind_ip: cli.bind_ip,
        listen_all: cli.listen_all,
        max_clients: cli.max_clients,
        nostr_relay: cli.nostr_relay,
        no_console: cli.no_console,
        config: cli.config,
        kv_cache_disk: cli.kv_cache_disk,
        kv_cache_disk_dir: cli.kv_cache_disk_dir,
        kv_cache_min_free: cli.kv_cache_min_free,
        owner_key: cli.owner_key,
        control_bind: cli.control_bind,
        control_advertise_addr: cli.control_advertise_addr,
        owner_required: cli.owner_required,
        node_label: cli.node_label,
        trust_policy: cli.trust_policy.map(map_trust_policy),
        trust_owner: cli.trust_owner,
        nostr_discovery: cli.nostr_discovery,
        audit_log_path: cli.audit_log_path,
        audit_log_format: cli.audit_log_format,
        audit_log_level: cli.audit_log_level,
        audit_max_file_size: 100 * 1024 * 1024,
        audit_max_files: 10,
    }
}

fn speculative_overrides_from_cli(
    cli: &mesh_llm_cli::Cli,
) -> Option<mesh_llm_host_runtime::plugin::SpeculativeConfig> {
    let suppress_cooldown_drafts = if cli.speculative_native_mtp_allow_cooldown_drafts {
        Some(false)
    } else {
        cli.speculative_native_mtp_suppress_cooldown_drafts
            .then_some(true)
    };
    let mut overrides = mesh_llm_host_runtime::plugin::SpeculativeConfig::default();
    overrides.strategy = cli.speculative_strategy.clone();
    overrides.ngram_min = cli.speculative_ngram_min;
    overrides.ngram_max = cli.speculative_ngram_max;
    overrides.ngram_max_proposal_tokens = cli.speculative_ngram_max_proposal_tokens;
    overrides.ngram_proposer = cli
        .speculative_ngram_proposer
        .map(|proposer| proposer.as_str().to_string());
    overrides.extension_max_tokens = cli.speculative_extension_max_tokens;
    overrides.native_mtp_reject_cooldown_tokens = cli.speculative_native_mtp_reject_cooldown_tokens;
    overrides.native_mtp_suppress_cooldown_drafts = suppress_cooldown_drafts;
    overrides.native_mtp_suppress_cooldown_draft_limit =
        cli.speculative_native_mtp_suppress_cooldown_draft_limit;
    overrides.verify_window_min_tokens = cli.speculative_verify_window_min_tokens;
    overrides.verify_window_max_tokens = cli.speculative_verify_window_max_tokens;
    overrides.verify_window_pipeline_depth = cli.speculative_verify_window_pipeline_depth;
    (overrides != Default::default()).then_some(overrides)
}

fn command_uses_machine_output(command: Option<&mesh_llm_cli::Command>) -> bool {
    matches!(
        command,
        Some(mesh_llm_cli::Command::Doctor {
            json: true,
            command: None,
        }) | Some(mesh_llm_cli::Command::Runtime {
            command: Some(
                mesh_llm_cli::runtime::RuntimeCommand::List { json: true, .. }
                    | mesh_llm_cli::runtime::RuntimeCommand::Install { json: true, .. }
                    | mesh_llm_cli::runtime::RuntimeCommand::Remove { json: true, .. }
                    | mesh_llm_cli::runtime::RuntimeCommand::Prune { json: true, .. },
            ),
        })
    )
}

fn map_runtime_surface(
    surface: mesh_llm_cli::RuntimeSurface,
) -> mesh_llm_host_runtime::RuntimeSurface {
    match surface {
        mesh_llm_cli::RuntimeSurface::Serve => mesh_llm_host_runtime::RuntimeSurface::Serve,
        mesh_llm_cli::RuntimeSurface::Client => mesh_llm_host_runtime::RuntimeSurface::Client,
    }
}

fn map_mesh_discovery_mode(
    mode: mesh_llm_cli::MeshDiscoveryMode,
) -> mesh_llm_host_runtime::discovery::MeshDiscoveryMode {
    match mode {
        mesh_llm_cli::MeshDiscoveryMode::Nostr => {
            mesh_llm_host_runtime::discovery::MeshDiscoveryMode::Nostr
        }
        mesh_llm_cli::MeshDiscoveryMode::Mdns => {
            mesh_llm_host_runtime::discovery::MeshDiscoveryMode::Mdns
        }
    }
}

fn map_mesh_guardrail_mode(
    mode: mesh_llm_cli::MeshGuardrailCliMode,
) -> mesh_llm_host_runtime::MeshGuardrailMode {
    match mode {
        mesh_llm_cli::MeshGuardrailCliMode::Disabled => {
            mesh_llm_host_runtime::MeshGuardrailMode::Disabled
        }
        mesh_llm_cli::MeshGuardrailCliMode::Metrics => {
            mesh_llm_host_runtime::MeshGuardrailMode::Metrics
        }
        mesh_llm_cli::MeshGuardrailCliMode::Enforce => {
            mesh_llm_host_runtime::MeshGuardrailMode::Enforce
        }
    }
}

fn map_binary_flavor(flavor: mesh_llm_cli::BinaryFlavor) -> mesh_llm_system::backend::BinaryFlavor {
    match flavor {
        mesh_llm_cli::BinaryFlavor::Cpu => mesh_llm_system::backend::BinaryFlavor::Cpu,
        mesh_llm_cli::BinaryFlavor::Cuda => mesh_llm_system::backend::BinaryFlavor::Cuda,
        mesh_llm_cli::BinaryFlavor::Rocm => mesh_llm_system::backend::BinaryFlavor::Rocm,
        mesh_llm_cli::BinaryFlavor::Vulkan => mesh_llm_system::backend::BinaryFlavor::Vulkan,
        mesh_llm_cli::BinaryFlavor::Metal => mesh_llm_system::backend::BinaryFlavor::Metal,
    }
}

fn map_trust_policy(
    policy: mesh_llm_cli::TrustPolicy,
) -> mesh_llm_host_runtime::crypto::TrustPolicy {
    match policy {
        mesh_llm_cli::TrustPolicy::Off => mesh_llm_host_runtime::crypto::TrustPolicy::Off,
        mesh_llm_cli::TrustPolicy::PreferOwned => {
            mesh_llm_host_runtime::crypto::TrustPolicy::PreferOwned
        }
        mesh_llm_cli::TrustPolicy::RequireOwned => {
            mesh_llm_host_runtime::crypto::TrustPolicy::RequireOwned
        }
        mesh_llm_cli::TrustPolicy::Allowlist => {
            mesh_llm_host_runtime::crypto::TrustPolicy::Allowlist
        }
    }
}

#[cfg(test)]
mod cli_entrypoint_tests {
    use clap::Parser;
    use std::ffi::OsString;

    #[test]
    fn runtime_surface_help_request_handles_serve_and_client_help() {
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("serve"),
                OsString::from("--help"),
            ]),
            Some(mesh_llm_cli::RuntimeSurface::Serve)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("client"),
                OsString::from("-h"),
            ]),
            Some(mesh_llm_cli::RuntimeSurface::Client)
        );
    }

    #[test]
    fn runtime_surface_help_request_skips_leading_global_flags() {
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("--log-format"),
                OsString::from("json"),
                OsString::from("serve"),
                OsString::from("--help"),
            ]),
            Some(mesh_llm_cli::RuntimeSurface::Serve)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("--relay-auth=https://relay.example=token"),
                OsString::from("client"),
                OsString::from("-h"),
            ]),
            Some(mesh_llm_cli::RuntimeSurface::Client)
        );
    }

    #[test]
    fn binary_help_request_handles_help_help() {
        assert!(super::binary_help_request([
            OsString::from("mesh-llm"),
            OsString::from("help"),
            OsString::from("--help"),
        ]));
    }

    #[test]
    fn advanced_help_lists_the_complete_checkpoint_quantization_surface() {
        let help = super::advanced_help_text();

        assert!(help.contains("--quant <CHECKPOINT_QUANTIZATION>"));
        assert!(help.contains("Valid recipes: preserve, F32, F16, BF16, Q1_0, Q2_0"));
        assert!(help.contains("IQ2_XXS"));
        assert!(help.contains("IQ1_M"));
        assert!(help.contains("MXFP4_MOE"));
        assert!(help.contains("--checkpoint-imatrix"));
    }

    #[test]
    fn parse_failure_family_uses_runtime_surface_and_unknown_fallback() {
        use mesh_llm_events::CliCommandFamily;

        assert_eq!(
            super::parse_failure_family(
                &[
                    OsString::from("mesh-llm"),
                    OsString::from("serve"),
                    OsString::from("--bad-flag")
                ],
                None,
            ),
            CliCommandFamily::Runtime
        );
        assert_eq!(
            super::parse_failure_family(
                &[
                    OsString::from("mesh-llm"),
                    OsString::from("client"),
                    OsString::from("--bad-flag")
                ],
                None,
            ),
            CliCommandFamily::Runtime
        );
        assert_eq!(
            super::parse_failure_family(
                &[
                    OsString::from("mesh-llm"),
                    OsString::from("--port"),
                    OsString::from("bad")
                ],
                Some(mesh_llm_cli::RuntimeSurface::Serve),
            ),
            CliCommandFamily::Runtime
        );
        assert_eq!(
            super::parse_failure_family(
                &[
                    OsString::from("mesh-llm"),
                    OsString::from("definitely-not-a-command")
                ],
                None,
            ),
            CliCommandFamily::Unknown
        );
        assert_eq!(
            super::parse_failure_family(
                &[
                    OsString::from("mesh-llm"),
                    OsString::from("--config"),
                    OsString::from("runtime"),
                    OsString::from("--bad-flag"),
                ],
                None,
            ),
            CliCommandFamily::Unknown
        );
    }

    #[test]
    fn clap_display_exits_are_completed_while_invalid_values_are_parse_failures() {
        use mesh_llm_events::CliCommandOutcome;

        let version = mesh_llm_cli::Cli::try_parse_from(["mesh-llm", "--version"])
            .expect_err("version is represented as an early clap exit");
        assert_eq!(
            super::parse_exit_outcome(&version),
            CliCommandOutcome::Completed
        );

        let invalid = mesh_llm_cli::Cli::try_parse_from(["mesh-llm", "--port", "invalid"])
            .expect_err("invalid port must fail parsing");
        assert_eq!(
            super::parse_exit_outcome(&invalid),
            CliCommandOutcome::ParseFailed
        );
    }

    #[test]
    fn parse_failure_config_path_is_used_without_retaining_other_arguments() {
        assert_eq!(
            super::config_path_from_args(&[
                OsString::from("mesh-llm"),
                OsString::from("--config"),
                OsString::from("/tmp/logging-test.toml"),
                OsString::from("--port"),
                OsString::from("private-invalid-value"),
            ]),
            Some(std::path::PathBuf::from("/tmp/logging-test.toml"))
        );
    }

    #[test]
    fn cli_ngram_override_reaches_runtime_config() {
        let normalized = mesh_llm_cli::normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--speculative-strategy",
            "mtp",
            "--speculative-ngram-min",
            "2",
            "--speculative-ngram-max",
            "6",
            "--speculative-ngram-max-proposal-tokens",
            "5",
        ]);
        let cli = mesh_llm_cli::Cli::try_parse_from(normalized.normalized).expect("CLI parses");

        let config = super::speculative_overrides_from_cli(&cli)
            .expect("speculative flags produce an override");

        assert_eq!(config.strategy.as_deref(), Some("mtp"));
        assert_eq!(config.ngram_min, Some(2));
        assert_eq!(config.ngram_max, Some(6));
        assert_eq!(config.ngram_max_proposal_tokens, Some(5));
    }

    #[test]
    fn runtime_surface_help_request_handles_flags_and_help_subcommands() {
        use mesh_llm_cli::RuntimeSurface;

        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("serve"),
                OsString::from("--help"),
            ]),
            Some(RuntimeSurface::Serve)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("help"),
                OsString::from("serve"),
            ]),
            Some(RuntimeSurface::Serve)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("help"),
                OsString::from("serve"),
                OsString::from("--help"),
            ]),
            Some(RuntimeSurface::Serve)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("--config"),
                OsString::from("mesh.toml"),
                OsString::from("help"),
                OsString::from("serve"),
            ]),
            Some(RuntimeSurface::Serve)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("--log-format"),
                OsString::from("json"),
                OsString::from("help"),
                OsString::from("client"),
            ]),
            Some(RuntimeSurface::Client)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("client"),
                OsString::from("-h"),
            ]),
            Some(RuntimeSurface::Client)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("help"),
                OsString::from("client"),
            ]),
            Some(RuntimeSurface::Client)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("help"),
                OsString::from("client"),
                OsString::from("-h"),
            ]),
            Some(RuntimeSurface::Client)
        );
        assert_eq!(
            super::runtime_surface_help_request([
                OsString::from("mesh-llm"),
                OsString::from("--help"),
            ]),
            None
        );
    }
}
