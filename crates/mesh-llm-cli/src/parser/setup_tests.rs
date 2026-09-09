use super::{Cli, Command};
use clap::{Parser, error::ErrorKind};

#[test]
fn setup_command_parses_without_plugin_fallback() {
    let cli = Cli::parse_from([
        "mesh-llm",
        "setup",
        "--yes",
        "--no-interactive",
        "--skip-runtime",
        "--verbose",
    ]);

    match cli.command.expect("setup command expected") {
        Command::Setup {
            yes,
            no_interactive,
            service,
            no_service,
            skip_runtime,
            verbose,
        } => {
            assert!(yes);
            assert!(no_interactive);
            assert!(!service);
            assert!(!no_service);
            assert!(skip_runtime);
            assert!(verbose);
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[test]
fn setup_command_rejects_conflicting_service_flags() {
    let err = Cli::try_parse_from(["mesh-llm", "setup", "--service", "--no-service"])
        .expect_err("setup should reject conflicting service flags");

    assert_eq!(err.kind(), ErrorKind::ArgumentConflict);
    assert!(err.to_string().contains("--service"));
    assert!(err.to_string().contains("--no-service"));
}

#[test]
fn setup_command_rejects_skip_doctor_flag() {
    let err = Cli::try_parse_from(["mesh-llm", "setup", "--skip-doctor"])
        .expect_err("setup should reject unknown skip-doctor flag");

    assert_eq!(err.kind(), ErrorKind::UnknownArgument);
    assert!(err.to_string().contains("--skip-doctor"));
}
