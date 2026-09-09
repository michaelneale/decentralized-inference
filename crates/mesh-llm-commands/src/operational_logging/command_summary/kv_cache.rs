use mesh_llm_cli::KvCacheCommand;

use super::{DEFAULT_LOCAL_PORT, SummaryAssembly};

pub(super) fn format_kv_cache(command: &KvCacheCommand, assembly: &mut SummaryAssembly) {
    assembly.command.push_str(" kv-cache");
    match command {
        KvCacheCommand::Status {
            endpoints,
            port,
            json,
        } => {
            assembly.command.push_str(" status");
            assembly.redact("--endpoint", !endpoints.is_empty());
            assembly.port(*port, DEFAULT_LOCAL_PORT);
            assembly.flag("json", *json);
        }
        KvCacheCommand::Prune {
            target,
            model_identity,
            yes,
            endpoints,
            port,
            json,
        } => {
            assembly.command.push_str(" prune");
            assembly.redact("--target", target.is_some());
            assembly.redact("--model-identity", model_identity.is_some());
            assembly.redact("--endpoint", !endpoints.is_empty());
            assembly.port(*port, DEFAULT_LOCAL_PORT);
            assembly.flag("yes", *yes);
            assembly.flag("json", *json);
        }
        KvCacheCommand::Clear {
            model_identity,
            yes,
            endpoints,
            port,
            json,
        } => {
            assembly.command.push_str(" clear");
            assembly.redact("--model-identity", model_identity.is_some());
            assembly.redact("--endpoint", !endpoints.is_empty());
            assembly.port(*port, DEFAULT_LOCAL_PORT);
            assembly.flag("yes", *yes);
            assembly.flag("json", *json);
        }
    }
}
