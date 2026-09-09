mod administration;
mod auth;
mod benchmark;
mod dispatch;
mod kv_cache;
mod models;
mod runtime;

use mesh_llm_cli::Command;
use mesh_llm_events::CliCommandSummary;

const DEFAULT_LOCAL_PORT: u16 = 3131;
const DEFAULT_AGENT_PORT: u16 = 9337;

#[derive(Default)]
struct SummaryAssembly {
    command: String,
    flags: Vec<String>,
    values: Vec<String>,
    redacted: Vec<&'static str>,
}

struct ModelPrepareSummary<'a> {
    source_repo: &'a Option<String>,
    quant: &'a Option<String>,
    target: &'a Option<String>,
    model_id: &'a Option<String>,
    flavor: &'a str,
    timeout: &'a str,
    mesh_llm_ref: &'a str,
    dry_run: bool,
    confirm: bool,
    follow: bool,
    json: bool,
    status: &'a Option<String>,
    logs: &'a Option<String>,
    cancel: &'a Option<String>,
    list: bool,
    update_script: bool,
}

impl SummaryAssembly {
    fn new(command: &str) -> Self {
        Self {
            command: command.to_owned(),
            ..Self::default()
        }
    }

    fn flag(&mut self, name: &'static str, present: bool) {
        if present {
            let marker = format!("--{name}");
            if !self.flags.contains(&marker) {
                self.flags.push(marker);
            }
        }
    }

    fn redact(&mut self, name: &'static str, present: bool) {
        if present {
            self.redacted.push(name);
        }
    }

    fn port(&mut self, port: u16, default: u16) {
        if port != default {
            self.values.push(format!("--port {port}"));
        }
    }

    fn finish(self) -> Option<CliCommandSummary> {
        let mut value = self.command;
        for part in self.flags.into_iter().chain(self.values) {
            value.push(' ');
            value.push_str(&part);
        }
        for name in self.redacted {
            value.push(' ');
            value.push_str(name);
            value.push_str(" [REDACTED]");
        }
        CliCommandSummary::sanitize(&value)
    }
}

pub(super) fn command_summary(command: &Command) -> Option<CliCommandSummary> {
    let mut assembly = SummaryAssembly::new("mesh-llm");
    dispatch::format_command(command, &mut assembly);
    assembly.finish()
}

#[cfg(test)]
#[path = "command_summary_tests.rs"]
mod command_summary_tests;

#[cfg(test)]
#[path = "command_summary_context_tests.rs"]
mod command_summary_context_tests;
