use super::runtime::{native_runtime_command_selection, native_runtime_config_selector};
use anyhow::{Result, bail};
use mesh_llm_cli::Command;
use mesh_llm_commands::setup::{SetupCommandArgs, SetupEnvironment, SetupOptions, SetupPlatform};
use std::io::IsTerminal;
use std::path::Path;

pub(crate) async fn dispatch_setup_command(
    cmd: &Command,
    config_path: Option<&Path>,
) -> Result<()> {
    let selector = native_runtime_config_selector(config_path)?;
    let args = setup_command_args(cmd, native_runtime_command_selection(selector.as_ref()))?;
    mesh_llm_commands::setup::run_setup_command(args).await
}

fn setup_command_args<'a>(
    cmd: &Command,
    configured: mesh_llm_commands::runtime_native::NativeRuntimeConfigSelection<'a>,
) -> Result<SetupCommandArgs<'a>> {
    let Command::Setup {
        yes,
        no_interactive,
        service,
        no_service,
        skip_runtime,
        verbose,
    } = cmd
    else {
        bail!("dispatch_setup_command called for non-setup command");
    };

    Ok(SetupCommandArgs {
        options: SetupOptions {
            yes: *yes,
            no_interactive: *no_interactive,
            service: *service,
            no_service: *no_service,
            skip_runtime: *skip_runtime,
            verbose: *verbose,
        },
        environment: SetupEnvironment {
            platform: current_setup_platform()?,
            interactive: std::io::stdin().is_terminal() && std::io::stderr().is_terminal(),
        },
        configured,
    })
}

fn current_setup_platform() -> Result<SetupPlatform> {
    setup_platform_from_os(std::env::consts::OS)
}

fn setup_platform_from_os(os: &str) -> Result<SetupPlatform> {
    match os {
        "linux" => Ok(SetupPlatform::Linux),
        "macos" => Ok(SetupPlatform::MacOs),
        "windows" => Ok(SetupPlatform::Windows),
        _ => bail!("mesh-llm setup is not supported on {os}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_dispatch_maps_cli_booleans_directly_into_setup_options() {
        let args = setup_command_args(
            &Command::Setup {
                yes: true,
                no_interactive: false,
                service: true,
                no_service: false,
                skip_runtime: true,
                verbose: true,
            },
            mesh_llm_commands::runtime_native::NativeRuntimeConfigSelection::default(),
        )
        .expect("setup args should build");

        assert_eq!(
            args.options,
            SetupOptions {
                yes: true,
                no_interactive: false,
                service: true,
                no_service: false,
                skip_runtime: true,
                verbose: true,
            }
        );
    }

    #[test]
    fn setup_dispatch_rejects_doctor_commands_instead_of_falling_through() {
        let error = setup_command_args(
            &Command::Doctor {
                json: false,
                command: None,
            },
            mesh_llm_commands::runtime_native::NativeRuntimeConfigSelection::default(),
        )
        .expect_err("doctor command must not route through setup dispatch");

        assert!(
            error
                .to_string()
                .contains("dispatch_setup_command called for non-setup command")
        );
    }

    #[test]
    fn setup_platform_rejects_unknown_operating_systems() {
        let error = setup_platform_from_os("freebsd").expect_err("unsupported OS should fail");

        assert!(
            error
                .to_string()
                .contains("mesh-llm setup is not supported on freebsd")
        );
    }
}
