use super::{DEFAULT_LOCAL_PORT, SummaryAssembly};

pub(super) fn format_gpus(
    json: bool,
    command: Option<&mesh_llm_cli::GpuCommand>,
    assembly: &mut SummaryAssembly,
) {
    assembly.command.push_str(" gpus");
    let child_json = match command {
        None => false,
        Some(mesh_llm_cli::GpuCommand::Detect { json }) => {
            assembly.command.push_str(" detect");
            *json
        }
        Some(mesh_llm_cli::GpuCommand::RunBenchmark { backend }) => {
            assembly.command.push_str(" run-benchmark --backend ");
            assembly
                .command
                .push_str(&format!("{backend:?}").to_lowercase());
            false
        }
    };
    assembly.flag("json", json || child_json);
}

pub(super) fn format_config(command: &mesh_llm_cli::ConfigCommand, assembly: &mut SummaryAssembly) {
    let mesh_llm_cli::ConfigCommand::Validate { config_path, json } = command;
    assembly.command.push_str(" config validate");
    assembly.redact("--config-path", config_path.is_some());
    assembly.flag("json", *json);
}

pub(super) fn format_doctor(
    command: Option<&mesh_llm_cli::DoctorCommand>,
    json: bool,
    assembly: &mut SummaryAssembly,
) {
    assembly.command.push_str(" doctor");
    let mut child_json = false;
    if let Some(mesh_llm_cli::DoctorCommand::Split {
        model_ref,
        port,
        json: split_json,
        output_dir,
    }) = command
    {
        assembly.command.push_str(" split");
        assembly.redact("--model-ref", !model_ref.is_empty());
        assembly.port(*port, DEFAULT_LOCAL_PORT);
        child_json = *split_json;
        assembly.redact("--output-dir", output_dir.is_some());
    }
    assembly.flag("json", json || child_json);
}

pub(super) fn format_skills(command: &mesh_llm_cli::SkillCommand, assembly: &mut SummaryAssembly) {
    let mesh_llm_cli::SkillCommand::Install {
        agent,
        all,
        dry_run,
        force,
    } = command;
    assembly.command.push_str(" skills install");
    assembly.redact("--agent", !agent.is_empty());
    assembly.flag("all", *all);
    assembly.flag("dry-run", *dry_run);
    assembly.flag("force", *force);
}

pub(super) fn format_plugin(command: &mesh_llm_cli::PluginCommand, assembly: &mut SummaryAssembly) {
    use mesh_llm_cli::PluginCommand;
    match command {
        PluginCommand::Install {
            reference,
            archive,
            name,
            version,
        } => {
            assembly.command.push_str(" plugins install");
            assembly.redact("reference", reference.is_some());
            assembly.redact("--archive", archive.is_some());
            assembly.redact("--name", name.is_some());
            assembly.redact("--version", version.is_some());
        }
        PluginCommand::Update { .. } => format_plugin_name("update", assembly),
        PluginCommand::Enable { .. } => format_plugin_name("enable", assembly),
        PluginCommand::Disable { .. } => format_plugin_name("disable", assembly),
        PluginCommand::Delete { .. } => format_plugin_name("delete", assembly),
        PluginCommand::Info { .. } => format_plugin_name("info", assembly),
        PluginCommand::Search { query } => {
            assembly.command.push_str(" plugins search");
            assembly.redact("query", query.is_some());
        }
        PluginCommand::List => assembly.command.push_str(" plugins list"),
        PluginCommand::Discover { apply } => {
            assembly.command.push_str(" plugins discover");
            assembly.flag("apply", *apply);
        }
    }
}

fn format_plugin_name(name: &str, assembly: &mut SummaryAssembly) {
    assembly.command.push_str(" plugins ");
    assembly.command.push_str(name);
    assembly.redact("name", true);
}
