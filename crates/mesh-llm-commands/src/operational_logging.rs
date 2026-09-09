//! Bounded CLI command dispatch events.

mod command_summary;

use anyhow::Result;
use command_summary::command_summary;
use mesh_llm_cli::Command;
#[cfg(test)]
use mesh_llm_events::OutputEvent;
use mesh_llm_events::{
    CliCommandFamily, CliCommandOutcome, CliCommandSummary, emit_cli_command_event,
};
use std::fmt;
use std::sync::{Arc, LazyLock, RwLock};

/// Marker a handler can retain in its error chain when it explicitly rejects a
/// parsed command.
#[derive(Debug)]
pub struct CommandDispatchRejected;

impl fmt::Display for CommandDispatchRejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("command dispatch rejected")
    }
}

impl std::error::Error for CommandDispatchRejected {}

/// Fail-open durable-audit bridge invoked after each command boundary emission.
pub type CliOperationalAuditBridge =
    Arc<dyn Fn(CliCommandFamily, CliCommandOutcome, Option<CliCommandSummary>) + Send + Sync>;

static CLI_OPERATIONAL_AUDIT_BRIDGE: LazyLock<RwLock<Option<CliOperationalAuditBridge>>> =
    LazyLock::new(|| RwLock::new(None));

pub fn install_cli_operational_audit_bridge(bridge: CliOperationalAuditBridge) {
    *CLI_OPERATIONAL_AUDIT_BRIDGE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(bridge);
}

pub fn clear_cli_operational_audit_bridge() {
    *CLI_OPERATIONAL_AUDIT_BRIDGE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

pub fn cli_operational_audit_bridge() -> Option<CliOperationalAuditBridge> {
    CLI_OPERATIONAL_AUDIT_BRIDGE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandDispatchBoundary {
    family: CliCommandFamily,
    summary: Option<CliCommandSummary>,
}

impl CommandDispatchBoundary {
    pub fn start(command: &Command) -> Self {
        Self::start_with_family(command_family(command), command)
    }

    pub fn start_with_family(family: CliCommandFamily, command: &Command) -> Self {
        Self::start_with_summary(family, command_summary(command))
    }

    pub fn start_with_cli(cli: &mesh_llm_cli::Cli, family: CliCommandFamily) -> Self {
        let Some(command) = cli.command.as_ref() else {
            return Self::start_family(family);
        };
        let mut summary = command_summary(command).map(|value| value.as_str().to_owned());
        if let Some(summary_value) = summary.as_mut() {
            for (present, marker) in [
                (!cli.join.is_empty(), " --join [REDACTED]"),
                (!cli.relay.is_empty(), " --root-relay [REDACTED]"),
                (!cli.relay_auth.is_empty(), " --relay-auth [REDACTED]"),
            ] {
                if present && !summary_value.contains(marker) {
                    summary_value.push_str(marker);
                }
            }
        }
        Self::start_with_summary(
            family,
            summary.and_then(|value| CliCommandSummary::sanitize(&value)),
        )
    }

    fn start_with_summary(family: CliCommandFamily, summary: Option<CliCommandSummary>) -> Self {
        let boundary = Self { family, summary };
        boundary.emit(CliCommandOutcome::Started);
        boundary
    }

    pub fn start_family(family: CliCommandFamily) -> Self {
        let boundary = Self {
            family,
            summary: None,
        };
        boundary.emit(CliCommandOutcome::Started);
        boundary
    }

    pub fn finish(self, result: &Result<()>) {
        self.emit(command_outcome(result));
    }

    fn emit(&self, outcome: CliCommandOutcome) {
        let _ = emit_cli_command_event(self.family, outcome);
        if let Some(bridge) = cli_operational_audit_bridge() {
            bridge(self.family, outcome, self.summary.clone());
        }
    }

    #[cfg(test)]
    fn event(&self, outcome: CliCommandOutcome) -> OutputEvent {
        OutputEvent::CliCommandLifecycle {
            family: self.family,
            outcome,
        }
    }
}

pub fn emit_cli_process_event(family: CliCommandFamily, outcome: CliCommandOutcome) {
    let boundary = CommandDispatchBoundary {
        family,
        summary: None,
    };
    boundary.emit(outcome);
}

pub fn command_family(command: &Command) -> CliCommandFamily {
    match command {
        Command::Models { .. } | Command::Download { .. } | Command::ModelPrepare { .. } => {
            CliCommandFamily::Models
        }
        Command::Update { .. } | Command::Setup { .. } | Command::Uninstall { .. } => {
            CliCommandFamily::Installation
        }
        Command::Gpus { .. } => CliCommandFamily::Hardware,
        Command::Serve
        | Command::Client
        | Command::Runtime { .. }
        | Command::KvCache { .. }
        | Command::Load { .. }
        | Command::Unload { .. }
        | Command::Status { .. }
        | Command::Stop => CliCommandFamily::Runtime,
        Command::Config { .. } => CliCommandFamily::Configuration,
        Command::Doctor { .. } => CliCommandFamily::Diagnostics,
        Command::Discover { .. } => CliCommandFamily::Discovery,
        Command::RotateKey | Command::Auth { .. } => CliCommandFamily::Identity,
        Command::Goose { .. }
        | Command::Claude { .. }
        | Command::Pi { .. }
        | Command::Opencode { .. } => CliCommandFamily::Agent,
        Command::Plugin { .. } | Command::ExternalPlugin(_) => CliCommandFamily::Plugin,
        Command::Skills { .. } => CliCommandFamily::Skills,
        Command::Benchmark { .. } => CliCommandFamily::Benchmark,
    }
}

fn command_outcome(result: &Result<()>) -> CliCommandOutcome {
    match result {
        Ok(()) => CliCommandOutcome::Completed,
        Err(error) if error.downcast_ref::<CommandDispatchRejected>().is_some() => {
            CliCommandOutcome::Rejected
        }
        Err(_) => CliCommandOutcome::Failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Error, anyhow};
    use clap::Parser;
    use mesh_llm_events::{clear_output_sink, set_cli_command_event_verbose, set_output_sink};
    use std::io;

    #[derive(Default)]
    struct RecordingBridge {
        calls: std::sync::Mutex<Vec<(CliCommandFamily, CliCommandOutcome)>>,
    }

    impl RecordingBridge {
        fn take_calls(&self) -> Vec<(CliCommandFamily, CliCommandOutcome)> {
            std::mem::take(&mut *self.calls.lock().expect("recording bridge mutex poisoned"))
        }
    }

    struct BridgeResetGuard;

    impl Drop for BridgeResetGuard {
        fn drop(&mut self) {
            clear_cli_operational_audit_bridge();
        }
    }

    #[derive(Default)]
    struct RecordingOutputSink {
        events: std::sync::Mutex<Vec<OutputEvent>>,
    }

    impl RecordingOutputSink {
        fn take_events(&self) -> Vec<OutputEvent> {
            std::mem::take(&mut *self.events.lock().expect("recording sink mutex poisoned"))
        }
    }

    impl mesh_llm_events::OutputSink for RecordingOutputSink {
        fn emit_event(&self, event: OutputEvent) -> io::Result<()> {
            self.events
                .lock()
                .expect("recording sink mutex poisoned")
                .push(event);
            Ok(())
        }
    }

    struct OutputSinkResetGuard;

    impl Drop for OutputSinkResetGuard {
        fn drop(&mut self) {
            clear_output_sink();
        }
    }

    struct CommandEventVerboseResetGuard;

    impl Drop for CommandEventVerboseResetGuard {
        fn drop(&mut self) {
            set_cli_command_event_verbose(false);
        }
    }

    fn lifecycle_events(command: &Command, result: &Result<()>) -> [OutputEvent; 2] {
        let boundary = CommandDispatchBoundary::start(command);
        [
            boundary.event(CliCommandOutcome::Started),
            boundary.event(command_outcome(result)),
        ]
    }

    #[test]
    #[serial_test::serial]
    fn command_dispatch_orders_started_before_completed_without_command_arguments() {
        let command = Command::Load {
            name: "private-model.gguf?token=private-token".to_string(),
            port: 41731,
        };
        let events = lifecycle_events(&command, &Ok(()));
        assert_eq!(
            events[0].clone(),
            OutputEvent::CliCommandLifecycle {
                family: CliCommandFamily::Runtime,
                outcome: CliCommandOutcome::Started
            }
        );
        assert_eq!(
            events[1].clone(),
            OutputEvent::CliCommandLifecycle {
                family: CliCommandFamily::Runtime,
                outcome: CliCommandOutcome::Completed
            }
        );
        let serialized = format!("{events:?}");
        assert!(!serialized.contains("private-token"));
    }

    #[test]
    #[serial_test::serial]
    fn explicit_handler_rejection_maps_to_rejected_without_error_detail() {
        let command = Command::Discover {
            name: Some("private-mesh".to_string()),
            model: None,
            min_vram: None,
            region: Some("private-region".to_string()),
            auto: false,
            relay: vec!["wss://relay.private.example/?token=private-token".to_string()],
        };
        let result: Result<()> = Err(Error::new(CommandDispatchRejected)
            .context("private command rejection detail with private-token"));
        let events = lifecycle_events(&command, &result);
        assert_eq!(
            events[1].clone(),
            OutputEvent::CliCommandLifecycle {
                family: CliCommandFamily::Discovery,
                outcome: CliCommandOutcome::Rejected
            }
        );
        assert!(!format!("{events:?}").contains("private-token"));
    }

    #[test]
    #[serial_test::serial]
    fn unmarked_dispatch_failure_maps_to_failed_without_error_detail() {
        let command = Command::Download {
            name: Some("private/model?token=private-token".to_string()),
            draft: true,
        };
        let result: Result<()> = Err(anyhow!(
            "download failed for https://private.example/model?token=private-token"
        ));
        let events = lifecycle_events(&command, &result);
        assert_eq!(
            events[1].clone(),
            OutputEvent::CliCommandLifecycle {
                family: CliCommandFamily::Models,
                outcome: CliCommandOutcome::Failed
            }
        );
        assert!(!format!("{events:?}").contains("private-token"));
    }

    #[test]
    #[serial_test::serial]
    fn command_dispatch_keeps_output_events_unchanged_when_bridge_installed() {
        let sink = Arc::new(RecordingOutputSink::default());
        let _sink_guard = OutputSinkResetGuard;
        let _verbose_guard = CommandEventVerboseResetGuard;
        set_cli_command_event_verbose(true);
        set_output_sink(sink.clone());
        let recording = Arc::new(RecordingBridge::default());
        let _bridge_guard = BridgeResetGuard;
        let bridge_recording = recording.clone();
        install_cli_operational_audit_bridge(Arc::new(move |family, outcome, _summary| {
            bridge_recording
                .calls
                .lock()
                .expect("recording bridge mutex poisoned")
                .push((family, outcome));
        }));
        let boundary = CommandDispatchBoundary::start(&Command::Load {
            name: "model.gguf".to_string(),
            port: 9337,
        });
        boundary.finish(&Ok(()));
        assert_eq!(sink.take_events().len(), 2);
        assert_eq!(
            recording.take_calls(),
            vec![
                (CliCommandFamily::Runtime, CliCommandOutcome::Started),
                (CliCommandFamily::Runtime, CliCommandOutcome::Completed)
            ]
        );
    }

    #[test]
    #[serial_test::serial]
    fn command_dispatch_without_bridge_stays_fail_open() {
        clear_cli_operational_audit_bridge();
        let boundary = CommandDispatchBoundary::start(&Command::Load {
            name: "private-model.gguf".to_string(),
            port: 41731,
        });
        boundary.finish(&Err(anyhow!("private dispatch failure")));
        assert!(cli_operational_audit_bridge().is_none());
    }

    #[test]
    #[serial_test::serial]
    fn bridge_receives_only_static_family_and_outcome_without_command_arguments() {
        let recording = Arc::new(RecordingBridge::default());
        let _bridge_guard = BridgeResetGuard;
        let bridge_recording = recording.clone();
        install_cli_operational_audit_bridge(Arc::new(move |family, outcome, _summary| {
            bridge_recording
                .calls
                .lock()
                .expect("recording bridge mutex poisoned")
                .push((family, outcome));
        }));
        let boundary = CommandDispatchBoundary::start(&Command::Discover {
            name: Some("private-mesh".to_string()),
            model: None,
            min_vram: None,
            region: Some("private-region".to_string()),
            auto: false,
            relay: vec!["wss://relay.private.example/?token=private-token".to_string()],
        });
        boundary.finish(&Err(Error::new(CommandDispatchRejected)
            .context("private command rejection detail with private-token")));
        assert_eq!(
            recording.take_calls(),
            vec![
                (CliCommandFamily::Discovery, CliCommandOutcome::Started),
                (CliCommandFamily::Discovery, CliCommandOutcome::Rejected)
            ]
        );
    }

    #[test]
    #[serial_test::serial]
    fn dispatch_summary_is_shared_by_started_and_terminal_bridge_records() {
        let recording = Arc::new(std::sync::Mutex::new(Vec::new()));
        let expected = CliCommandSummary::sanitize("mesh-llm load --port 9337 name [REDACTED]")
            .expect("bounded summary");
        let _bridge_guard = BridgeResetGuard;
        let calls = recording.clone();
        install_cli_operational_audit_bridge(Arc::new(move |_, _, summary| {
            calls.lock().expect("bridge mutex").push(summary)
        }));
        let boundary = CommandDispatchBoundary::start(&Command::Load {
            name: "model.gguf".to_owned(),
            port: 9337,
        });
        boundary.finish(&Ok(()));
        assert_eq!(
            recording.lock().expect("bridge mutex").as_slice(),
            &[Some(expected.clone()), Some(expected)]
        );
    }

    #[test]
    #[serial_test::serial]
    fn start_with_cli_redacts_global_credentials_in_both_bridge_records() {
        let cli = mesh_llm_cli::Cli::parse_from([
            "mesh-llm",
            "--join",
            "private-invite-token",
            "--relay",
            "wss://relay.example/?token=private-relay-token",
            "--relay-auth",
            "https://relay.example/?token=private-token=padding",
            "load",
            "private/model",
        ]);
        let recording = Arc::new(std::sync::Mutex::new(Vec::new()));
        let _bridge_guard = BridgeResetGuard;
        let calls = recording.clone();
        install_cli_operational_audit_bridge(Arc::new(move |family, outcome, summary| {
            calls
                .lock()
                .expect("bridge mutex")
                .push((family, outcome, summary));
        }));

        let boundary = CommandDispatchBoundary::start_with_cli(&cli, CliCommandFamily::Runtime);
        boundary.finish(&Ok(()));

        let calls = recording.lock().expect("bridge mutex");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, CliCommandFamily::Runtime);
        assert_eq!(calls[1].0, CliCommandFamily::Runtime);
        assert_eq!(calls[0].1, CliCommandOutcome::Started);
        assert_eq!(calls[1].1, CliCommandOutcome::Completed);
        assert_eq!(calls[0].2, calls[1].2);
        let summary = calls[0].2.as_ref().expect("CLI summary").as_str();
        assert!(summary.contains("--join [REDACTED]"));
        assert!(summary.contains("--root-relay [REDACTED]"));
        assert!(summary.contains("--relay-auth [REDACTED]"));
        assert!(CliCommandSummary::sanitize(summary).is_some());
        assert!(!summary.contains("private-invite-token"));
        assert!(!summary.contains("private-relay-token"));
        assert!(!summary.contains("private-token"));
    }

    #[test]
    #[serial_test::serial]
    fn start_with_cli_does_not_fabricate_global_markers_without_base_summary() {
        let cli = mesh_llm_cli::Cli::parse_from([
            "mesh-llm",
            "--join",
            "private-invite-token",
            "--relay-auth",
            "https://relay.example/?token=private-token=padding",
            "benchmark",
            "tune",
            "--model",
            "private-model",
            "--json",
            "--ctx-sizes",
            "1",
            "--batch-sizes",
            "1",
            "--ubatch-sizes",
            "1",
            "--mmap-values",
            "auto",
            "--mlock-values",
            "enabled",
            "--flash-attention",
            "on",
            "--speculative-types",
            "auto",
            "--spec-draft-models",
            "private-draft.gguf",
            "--spec-draft-max-tokens",
            "1",
            "--spec-draft-min-tokens",
            "1",
            "--spec-ngram-min",
            "1",
            "--spec-ngram-max",
            "1",
            "--apply",
            "--launch-args",
            "--debug-telemetry",
        ]);
        let recording = Arc::new(std::sync::Mutex::new(Vec::new()));
        let _bridge_guard = BridgeResetGuard;
        let calls = recording.clone();
        install_cli_operational_audit_bridge(Arc::new(move |family, outcome, summary| {
            calls
                .lock()
                .expect("bridge mutex")
                .push((family, outcome, summary));
        }));

        let boundary = CommandDispatchBoundary::start_with_cli(&cli, CliCommandFamily::Benchmark);
        boundary.finish(&Ok(()));

        let calls = recording.lock().expect("bridge mutex");
        assert_eq!(calls.len(), 2);
        assert!(calls[0].2.is_none());
        assert!(calls[1].2.is_none());
    }
}
