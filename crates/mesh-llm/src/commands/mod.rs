mod config;
mod discover;
mod doctor;
mod download;
mod models;
mod plugin_cli;
mod runtime;
mod setup;

use anyhow::Result;
use mesh_llm_cli::{Cli, Command};

use self::config::dispatch_config_command;
use self::discover::{DiscoverOptions, run_discover, run_stop};
use self::doctor::dispatch_doctor_command;
use self::download::dispatch_download_command;
use self::models::dispatch_models_command;
use self::plugin_cli::run_external_plugin_command;
use self::runtime::{dispatch_runtime_command, run_drop, run_load, run_status};
use self::setup::dispatch_setup_command;

pub async fn dispatch(cli: &Cli) -> Result<bool> {
    let Some(cmd) = cli.command.as_ref() else {
        return Ok(false);
    };
    if matches!(cmd, Command::Serve | Command::Client) {
        return Ok(false);
    }
    let family = match cmd {
        Command::ExternalPlugin(args) => plugin_cli::external_cli_command_family(cli, args),
        _ => mesh_llm_commands::operational_logging::command_family(cmd),
    };
    let command_boundary =
        mesh_llm_commands::operational_logging::CommandDispatchBoundary::start_with_cli(
            cli, family,
        );
    let result = dispatch_command(cli, cmd).await;
    command_boundary.finish(&result);
    result?;
    Ok(true)
}

async fn dispatch_command(cli: &Cli, cmd: &Command) -> Result<()> {
    match cmd {
        Command::Auth { command } => mesh_llm_commands::auth::run_auth_command(command),
        Command::KvCache { command } => {
            mesh_llm_commands::kv_cache::dispatch_kv_cache_command(command).await
        }
        Command::Runtime { command } => {
            dispatch_runtime_command(command.as_ref(), cli.config.as_deref()).await
        }
        Command::ModelPrepare { .. } => dispatch_model_prepare(cmd).await,
        _ => dispatch_general_command(cli, cmd).await,
    }
}

async fn dispatch_general_command(cli: &Cli, cmd: &Command) -> Result<()> {
    match cmd {
        Command::Serve | Command::Client => Ok(()),
        Command::Models { command } => {
            dispatch_models_command(command).await?;
            Ok(())
        }
        Command::Download { name, draft } => {
            dispatch_download_command(name.as_deref(), *draft).await
        }
        Command::Update { .. } => mesh_llm_commands::update::run_update(cli).await,
        Command::Gpus { json, command } => {
            mesh_llm_commands::gpus::dispatch_gpu_command(*json, command.as_ref())?;
            Ok(())
        }
        Command::Runtime { .. } | Command::KvCache { .. } => {
            unreachable!("runtime and kv-cache commands are dispatched before general commands")
        }
        Command::Setup { .. } => dispatch_setup_command(cmd, cli.config.as_deref()).await,
        Command::Uninstall {
            dry_run,
            yes,
            keep_cache,
            keep_service_files,
            purge_config,
            keep_config,
            binary_path,
            json,
            verbose,
        } => mesh_llm_commands::uninstall::run_uninstall_command(
            mesh_llm_commands::uninstall::UninstallOptions {
                dry_run: *dry_run,
                yes: *yes,
                keep_cache: *keep_cache,
                keep_service_files: *keep_service_files,
                purge_config: *purge_config,
                keep_config: *keep_config,
                binary_path: binary_path.clone(),
                json: *json,
                verbose: *verbose,
            },
            run_stop,
        ),
        Command::Config { command } => dispatch_config_command(cli, command),
        Command::Doctor { command, json } => {
            dispatch_doctor_command(command.as_ref(), cli.config.as_deref(), *json).await
        }
        Command::Load { name, port } => run_load(name, *port).await,
        Command::Unload { name, port } => run_drop(name, *port).await,
        Command::Status { port } => run_status(*port).await,
        Command::Stop => run_stop(),
        Command::Discover {
            name,
            model,
            min_vram,
            region,
            auto,
            relay,
        } => {
            run_discover(DiscoverOptions {
                name: name.clone(),
                model: model.clone(),
                min_vram_gb: *min_vram,
                region: region.clone(),
                auto_join: *auto,
                relays: relay.clone(),
                discovery_mode: cli.mesh_discovery_mode,
                supplied_join_tokens: cli.join.clone(),
            })
            .await
        }
        Command::RotateKey => {
            mesh_llm_host_runtime::command_support::discovery::nostr::rotate_keys()
        }
        Command::Goose { model, port } => {
            mesh_llm_commands::agent_cli::run_goose(model.clone(), *port).await
        }
        Command::Claude { model, port } => {
            mesh_llm_commands::agent_cli::run_claude(model.clone(), *port).await
        }
        Command::Pi { model, host, write } => {
            mesh_llm_commands::agent_cli::run_pi(model.clone(), host, *write).await
        }
        Command::Opencode { model, host, write } => {
            mesh_llm_commands::agent_cli::run_opencode(model.clone(), host, *write).await
        }
        Command::Skills { command } => mesh_llm_commands::skills::run_skills_command(command),
        Command::Plugin { command } => run_plugin_command(command, cli).await,
        Command::Benchmark { command } => {
            mesh_llm_commands::benchmark::dispatch_benchmark_command(cli.config.as_deref(), command)
                .await
        }
        Command::ModelPrepare { .. } => dispatch_model_prepare(cmd).await,
        Command::Auth { command } => mesh_llm_commands::auth::run_auth_command(command),
        Command::ExternalPlugin(args) => run_external_plugin_command(cli, args).await,
    }
}

async fn dispatch_model_prepare(cmd: &Command) -> Result<()> {
    let Command::ModelPrepare {
        source_repo,
        quant,
        target,
        model_id,
        flavor,
        timeout,
        mesh_llm_ref,
        dry_run,
        confirm,
        follow,
        json,
        status,
        logs,
        cancel,
        list,
        update_script,
    } = cmd
    else {
        unreachable!("dispatch_model_prepare called for non-model-prepare command");
    };

    mesh_llm_commands::model_package::dispatch_model_package(
        mesh_llm_commands::model_package::ModelPrepareArgs {
            source_repo: source_repo.as_deref(),
            quant: quant.as_deref(),
            target: target.as_deref(),
            model_id: model_id.as_deref(),
            flavor,
            timeout,
            mesh_llm_ref,
            experimental: false,
            dry_run: *dry_run,
            confirm: *confirm,
            follow: *follow,
            json: *json,
            status: status.as_deref(),
            logs: logs.as_deref(),
            cancel: cancel.as_deref(),
            list: *list,
            update_script: *update_script,
        },
    )
    .await
}

async fn run_plugin_command(command: &mesh_llm_cli::PluginCommand, cli: &Cli) -> Result<()> {
    match command {
        mesh_llm_cli::PluginCommand::List => {
            let rows = resolved_plugin_list_rows(cli)?;
            mesh_llm_commands::plugin::run_plugin_command(command, Some(&rows)).await?;
        }
        mesh_llm_cli::PluginCommand::Info { .. } => {
            if !mesh_llm_commands::plugin::run_plugin_command(command, None).await? {
                let rows = resolved_plugin_list_rows(cli)?;
                mesh_llm_commands::plugin::run_plugin_command(command, Some(&rows)).await?;
            }
        }
        _ => {
            mesh_llm_commands::plugin::run_plugin_command(command, None).await?;
        }
    }
    Ok(())
}

fn resolved_plugin_list_rows(cli: &Cli) -> Result<mesh_llm_commands::plugin::PluginListRows> {
    let options = plugin_runtime_options_from_cli(cli);
    let resolved = mesh_llm_host_runtime::command_support::plugin::load_resolved_plugins(&options)?;
    Ok(mesh_llm_commands::plugin::PluginListRows {
        externals: resolved
            .externals
            .into_iter()
            .map(|spec| mesh_llm_commands::plugin::RuntimePluginRow {
                name: spec.name,
                command: spec.command,
                args: spec.args,
            })
            .collect(),
        inactive: resolved
            .inactive
            .into_iter()
            .map(|summary| mesh_llm_commands::plugin::InactivePluginRow {
                name: summary.name,
                kind: summary.kind,
                status: summary.status,
                error: summary.error,
            })
            .collect(),
    })
}

fn plugin_runtime_options_from_cli(cli: &Cli) -> mesh_llm_host_runtime::RuntimeOptions {
    mesh_llm_host_runtime::RuntimeOptions {
        config: cli.config.clone(),
        publish: cli.publish,
        nostr_discovery: cli.nostr_discovery,
        ..mesh_llm_host_runtime::RuntimeOptions::default()
    }
}
