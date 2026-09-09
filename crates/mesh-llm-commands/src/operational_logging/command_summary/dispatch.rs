use mesh_llm_cli::Command;

use super::{
    DEFAULT_AGENT_PORT, DEFAULT_LOCAL_PORT, ModelPrepareSummary, SummaryAssembly, administration,
    auth, benchmark, models, runtime,
};

pub(super) fn format_command(command: &Command, assembly: &mut SummaryAssembly) {
    match command {
        Command::Share { .. } => {
            assembly.command.push_str(" share");
            assembly.redact("url", true);
        }
        Command::Serve => assembly.command.push_str(" serve"),
        Command::Client => assembly.command.push_str(" client"),
        Command::Models { command } => models::format_models(command, assembly),
        Command::Runtime { command } => runtime::format_runtime(command.as_ref(), assembly),
        Command::Plugin { command } => administration::format_plugin(command, assembly),
        Command::Auth { command } => auth::format_auth(command, assembly),
        Command::Benchmark { command } => benchmark::format_benchmark(command, assembly),
        Command::Config { command } => administration::format_config(command, assembly),
        Command::Doctor { command, json } => {
            administration::format_doctor(command.as_ref(), *json, assembly);
        }
        Command::Skills { command } => administration::format_skills(command, assembly),
        Command::Download { name, draft } => {
            assembly.command.push_str(" download");
            assembly.redact("name", name.is_some());
            assembly.flag("draft", *draft);
        }
        Command::Update {
            version,
            flavor,
            detect_flavor,
        } => {
            assembly.command.push_str(" update");
            assembly.redact("--version", version.is_some());
            assembly.redact("--flavor", flavor.is_some());
            assembly.flag("detect-flavor", *detect_flavor);
        }
        Command::Gpus { json, command } => {
            administration::format_gpus(*json, command.as_ref(), assembly);
        }
        Command::Setup {
            yes,
            no_interactive,
            service,
            no_service,
            skip_runtime,
            verbose,
        } => {
            assembly.command.push_str(" setup");
            assembly.flag("yes", *yes);
            assembly.flag("no-interactive", *no_interactive);
            assembly.flag("service", *service);
            assembly.flag("no-service", *no_service);
            assembly.flag("skip-runtime", *skip_runtime);
            assembly.flag("verbose", *verbose);
        }
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
        } => {
            assembly.command.push_str(" uninstall");
            assembly.flag("dry-run", *dry_run);
            assembly.flag("yes", *yes);
            assembly.flag("keep-cache", *keep_cache);
            assembly.flag("keep-service-files", *keep_service_files);
            assembly.flag("purge-config", *purge_config);
            assembly.flag("keep-config", *keep_config);
            assembly.redact("--binary-path", binary_path.is_some());
            assembly.flag("json", *json);
            assembly.flag("verbose", *verbose);
        }
        Command::Load { port, .. } => runtime::format_local_model("load", *port, assembly),
        Command::Unload { port, .. } => runtime::format_local_model("unload", *port, assembly),
        Command::Status { port } => {
            assembly.command.push_str(" status");
            assembly.port(*port, DEFAULT_LOCAL_PORT);
        }
        Command::Discover {
            name,
            model,
            min_vram,
            region,
            auto,
            relay,
        } => {
            assembly.command.push_str(" discover");
            assembly.redact("--name", name.is_some());
            assembly.redact("--model", model.is_some());
            assembly.redact("--min-vram", min_vram.is_some());
            assembly.redact("--region", region.is_some());
            assembly.flag("auto", *auto);
            assembly.redact("--relay", !relay.is_empty());
        }
        Command::RotateKey => assembly.command.push_str(" rotate-key"),
        Command::Goose { model, port } => format_agent("goose", model, *port, assembly),
        Command::Claude { model, port } => format_agent("claude", model, *port, assembly),
        Command::Pi { model, host, write } => format_client("pi", model, host, *write, assembly),
        Command::Opencode { model, host, write } => {
            format_client("opencode", model, host, *write, assembly);
        }
        Command::Stop => assembly.command.push_str(" stop"),
        Command::ModelPrepare {
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
        } => models::format_model_prepare(
            ModelPrepareSummary {
                source_repo,
                quant,
                target,
                model_id,
                flavor,
                timeout,
                mesh_llm_ref,
                dry_run: *dry_run,
                confirm: *confirm,
                follow: *follow,
                json: *json,
                status,
                logs,
                cancel,
                list: *list,
                update_script: *update_script,
            },
            assembly,
        ),
        Command::ExternalPlugin(args) => {
            assembly.command.push_str(" external-plugin");
            assembly.redact("argv", !args.is_empty());
        }
    }
}

fn format_agent(name: &str, model: &Option<String>, port: u16, assembly: &mut SummaryAssembly) {
    assembly.command.push(' ');
    assembly.command.push_str(name);
    assembly.redact("--model", model.is_some());
    assembly.port(port, DEFAULT_AGENT_PORT);
}

fn format_client(
    name: &str,
    model: &Option<String>,
    host: &str,
    write: bool,
    assembly: &mut SummaryAssembly,
) {
    assembly.command.push(' ');
    assembly.command.push_str(name);
    assembly.redact("--model", model.is_some());
    assembly.redact("--host", host != "127.0.0.1:9337");
    assembly.flag("write", write);
}
