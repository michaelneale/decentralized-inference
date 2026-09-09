use std::ffi::OsString;

use super::commands::Cli;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeSurface {
    Share,
    Serve,
    Client,
}

#[derive(Clone, Debug)]
pub struct NormalizedRuntimeArgs {
    pub original: Vec<OsString>,
    pub normalized: Vec<OsString>,
    pub explicit_surface: Option<RuntimeSurface>,
}

pub fn normalize_runtime_surface_args<I, S>(args: I) -> NormalizedRuntimeArgs
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let original: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let mut normalized = original.clone();
    let mut explicit_surface = None;

    // Skip leading global flags to find the pseudo-subcommand position.
    // Recognized value-taking flags: --log-format, --mesh-discovery-mode, --max-vram,
    // --llama-flavor, --device, --tensor-split, --bind-port, --bind-ip, --max-clients,
    // --port, --console, --swarm-capture, --draft-max, --ctx-size.
    // Boolean flags: --help-advanced, --auto, --client, --local-model-only, --headless, --publish,
    // --plugin, --auto-update, --no-draft, --split, --no-enumerate-host, --listen-all,
    // --no-console, --owner-required.
    let value_taking_flags = [
        "--log-format",
        "--mesh-discovery-mode",
        "--max-vram",
        "--llama-flavor",
        "--device",
        "--tensor-split",
        "--bind-port",
        "--bind-ip",
        "--max-clients",
        "--port",
        "--console",
        "--swarm-capture",
        "--draft-max",
        "--ctx-size",
        "--model",
        "--gguf",
        "--mmproj",
        "--checkpoint-quantization",
        "--quant",
        "--checkpoint-imatrix",
        "--join",
        "--discover",
        "--mesh-name",
        "--region",
        "--name",
        "--plugin",
        "--draft",
        "--bin-dir",
        "--relay",
        "--relay-auth",
        "--nostr-relay",
        "--config",
        "--owner-key",
        "--control-bind",
        "--control-advertise-addr",
        "--node-label",
        "--trust-policy",
        "--trust-owner",
    ];

    let mut pos = 1;
    while pos < original.len() {
        let arg_str = original.get(pos).and_then(|arg| arg.to_str()).unwrap_or("");

        // Check for --flag=value form
        if let Some(eq_idx) = arg_str.find('=') {
            let flag_part = &arg_str[..eq_idx];
            if value_taking_flags.contains(&flag_part) {
                pos += 1;
                continue;
            }
        }

        // Check for --flag value form
        if value_taking_flags.contains(&arg_str) {
            // Advance by 2 if next token exists and doesn't start with '-'
            if let Some(next) = original.get(pos + 1).and_then(|arg| arg.to_str())
                && !next.starts_with('-')
            {
                pos += 2;
                continue;
            }
            // If next doesn't exist or starts with '-', advance by 1 (let Clap handle the error)
            pos += 1;
            continue;
        }

        // If it starts with '-' but isn't a recognized flag, it's likely a parse error or unknown flag
        if arg_str.starts_with('-') {
            pos += 1;
            continue;
        }

        // Found the first positional argument (serve/client/other subcommand)
        break;
    }

    // Now apply the serve/client normalization logic at the discovered position
    match original.get(pos).and_then(|arg| arg.to_str()) {
        Some("serve") => match original.get(pos + 1).and_then(|arg| arg.to_str()) {
            Some(arg) if arg.starts_with('-') => {
                normalized.remove(pos);
                explicit_surface = Some(RuntimeSurface::Serve);
            }
            None => {
                normalized.remove(pos);
                explicit_surface = Some(RuntimeSurface::Serve);
            }
            _ => {}
        },
        // `share <url>` becomes `--shared-endpoint <url>` so runtime flags
        // parse on either side of the URL. When no URL follows, the `share`
        // subcommand is left in place so clap reports the missing argument
        // and renders help.
        Some("share") => {
            explicit_surface = Some(RuntimeSurface::Share);
            if original
                .get(pos + 1)
                .and_then(|arg| arg.to_str())
                .is_some_and(|arg| !arg.starts_with('-'))
            {
                normalized.remove(pos);
                normalized.insert(pos, OsString::from("--shared-endpoint"));
            }
        }
        Some("client") => {
            normalized.remove(pos);
            normalized.insert(pos, OsString::from("--client"));
            explicit_surface = Some(RuntimeSurface::Client);
        }
        _ => {}
    }

    NormalizedRuntimeArgs {
        original,
        normalized,
        explicit_surface,
    }
}

pub fn legacy_runtime_surface_warning(
    cli: &Cli,
    original_args: &[OsString],
    explicit_surface: Option<RuntimeSurface>,
) -> Option<String> {
    if explicit_surface.is_some() || cli.command.is_some() {
        return None;
    }

    if cli.client {
        return Some(format!(
            "⚠️ top-level `--client` now maps to `mesh-llm client`.\n  Please use: {}",
            suggested_client_command(original_args)
        ));
    }

    if !cli.model.is_empty()
        || !cli.gguf.is_empty()
        || cli.mmproj.is_some()
        || cli.checkpoint_quantization.is_some()
        || cli.checkpoint_imatrix.is_some()
    {
        return Some(format!(
            "⚠️ top-level serving flags now map to `mesh-llm serve`.\n  Please use: {}",
            suggested_serve_command(original_args)
        ));
    }

    None
}

fn suggested_serve_command(original_args: &[OsString]) -> String {
    let mut args = Vec::with_capacity(original_args.len() + 1);
    if let Some(program) = original_args.first() {
        args.push(program.clone());
    } else {
        args.push(OsString::from("mesh-llm"));
    }
    args.push(OsString::from("serve"));
    args.extend(original_args.iter().skip(1).cloned());
    shell_join(&args)
}

fn suggested_client_command(original_args: &[OsString]) -> String {
    let mut args = Vec::with_capacity(original_args.len());
    if let Some(program) = original_args.first() {
        args.push(program.clone());
    } else {
        args.push(OsString::from("mesh-llm"));
    }
    args.push(OsString::from("client"));
    let mut skipped_client = false;
    for arg in original_args.iter().skip(1) {
        if !skipped_client && arg.to_string_lossy() == "--client" {
            skipped_client = true;
            continue;
        }
        args.push(arg.clone());
    }
    shell_join(&args)
}

fn shell_join(args: &[OsString]) -> String {
    args.iter().map(shell_display).collect::<Vec<_>>().join(" ")
}

fn shell_display(arg: &OsString) -> String {
    let text = arg.to_string_lossy();
    if text.is_empty() {
        "\"\"".into()
    } else if text
        .chars()
        .any(|ch| ch.is_whitespace() || matches!(ch, '"' | '\'' | '\\'))
    {
        format!("{text:?}")
    } else {
        text.into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{Cli, Command, MeshDiscoveryMode};
    use clap::Parser;
    use mesh_llm_events::LogFormat;
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn share_accepts_runtime_flags_before_and_after_url() {
        for args in [
            vec![
                "mesh-llm",
                "share",
                "http://localhost:11434",
                "--port",
                "9447",
            ],
            vec![
                "mesh-llm",
                "--port",
                "9447",
                "share",
                "http://localhost:11434",
            ],
        ] {
            let normalized = normalize_runtime_surface_args(args);
            assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Share));
            let cli = Cli::try_parse_from(normalized.normalized).unwrap();
            assert_eq!(
                cli.shared_endpoint.as_deref(),
                Some("http://localhost:11434")
            );
            assert_eq!(cli.port, 9447);
            assert!(!cli.client);
        }
    }

    #[test]
    fn share_requires_url_and_has_help() {
        for args in [
            vec!["mesh-llm", "share"],
            vec!["mesh-llm", "share", "--port", "9447"],
        ] {
            let normalized = normalize_runtime_surface_args(args);
            assert!(Cli::try_parse_from(normalized.normalized).is_err());
        }
        let normalized = normalize_runtime_surface_args(["mesh-llm", "share", "--help"]);
        let error = Cli::try_parse_from(normalized.normalized).unwrap_err();
        assert_eq!(error.kind(), clap::error::ErrorKind::DisplayHelp);
    }

    #[test]
    fn normalize_runtime_surface_args_rewrites_serve_invocation() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--auto",
            "--model",
            "Qwen3-8B-Q4_K_M",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "--auto", "--model", "Qwen3-8B-Q4_K_M"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_skips_quant_value_before_serve() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--quant",
            "Q4_K_M",
            "serve",
            "--model",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec![
                "mesh-llm",
                "--quant",
                "Q4_K_M",
                "--model",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
            ]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_bare_serve_loads_default_config() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "serve"]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_rewrites_client_invocation() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "client", "--auto", "--port", "9337"]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "--client", "--auto", "--port", "9337"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_treats_relay_auth_as_value_taking_before_serve() {
        // Regression: --relay-auth carries a `URL=TOKEN` value, so the
        // pseudo-subcommand scanner must skip the value and still discover
        // `serve` (or `client`) as the runtime surface. If --relay-auth is not
        // in the value-taking list the scanner stops at the token and Clap
        // sees a malformed command.
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--relay-auth",
            "https://gated.example/=token",
            "serve",
            "--relay",
            "https://gated.example/",
            "--auto",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec![
                "mesh-llm",
                "--relay-auth",
                "https://gated.example/=token",
                "--relay",
                "https://gated.example/",
                "--auto",
            ]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>()
        );

        // And the resulting argv must actually parse cleanly through Clap so
        // the relay-auth value reaches `Cli::relay_auth`.
        let cli = Cli::try_parse_from(&normalized.normalized).expect("clap parse");
        assert_eq!(
            cli.relay_auth,
            vec![("https://gated.example/".to_string(), "token".to_string())],
        );
    }

    #[test]
    fn normalize_runtime_surface_args_relay_auth_before_client_invocation() {
        // Same regression but for the `client` surface, including a token
        // containing `=` (NIP-98-style base64 padding).
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--relay-auth",
            "https://gated.example/=eyJhbGciOiJFZERTQSJ9.payload==",
            "client",
            "--auto",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
        let cli = Cli::try_parse_from(&normalized.normalized).expect("clap parse");
        assert!(cli.client, "client surface flag should be set");
        assert_eq!(
            cli.relay_auth,
            vec![(
                "https://gated.example/".to_string(),
                "eyJhbGciOiJFZERTQSJ9.payload==".to_string()
            )],
        );
    }

    #[test]
    fn normalize_runtime_surface_args_keeps_non_runtime_subcommands() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "download", "foo"]);

        assert_eq!(normalized.explicit_surface, None);
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "download", "foo"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn legacy_runtime_surface_warning_for_top_level_serve_flags() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "--auto", "--model", "Qwen3-8B-Q4_K_M"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        let warning =
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .expect("warning should be present");

        assert!(warning.contains("mesh-llm serve --auto --model Qwen3-8B-Q4_K_M"));
    }

    #[test]
    fn legacy_runtime_surface_warning_for_top_level_client_flag() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "--auto", "--client"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        let warning =
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .expect("warning should be present");

        assert!(warning.contains("mesh-llm client --auto"));
    }

    #[test]
    fn explicit_runtime_surface_suppresses_legacy_warning() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "client", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        assert!(
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .is_none()
        );
    }

    #[test]
    fn cli_accepts_headless_flag_for_serve_surface() {
        let args = vec!["mesh-llm", "serve", "--headless", "--auto"];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();
        assert!(cli.headless);
    }

    #[test]
    fn cli_accepts_headless_flag_for_client_surface() {
        let args = vec!["mesh-llm", "client", "--headless", "--auto"];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();
        assert!(cli.headless);
    }

    #[test]
    fn cli_accepts_swarm_capture_flag_for_client_surface() {
        let args = vec![
            "mesh-llm",
            "client",
            "--swarm-capture",
            "/tmp/mesh-capture",
            "--auto",
        ];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();

        assert!(cli.client);
        assert_eq!(cli.swarm_capture, Some(PathBuf::from("/tmp/mesh-capture")));
    }

    #[test]
    fn cli_accepts_global_swarm_capture_before_client() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--swarm-capture",
            "/tmp/mesh-capture",
            "client",
            "--auto",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert!(cli.client);
        assert_eq!(cli.swarm_capture, Some(PathBuf::from("/tmp/mesh-capture")));
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
    }

    #[test]
    fn legacy_no_console_remains_ignored_in_headless_tests() {
        let args = vec!["mesh-llm", "serve", "--no-console"];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();
        assert!(
            !cli.headless,
            "--no-console must not activate headless mode"
        );
    }

    #[test]
    fn local_model_only_is_an_explicit_serve_topology() {
        let args = vec![
            "mesh-llm",
            "serve",
            "--local-model-only",
            "--model",
            "/models/model.gguf",
        ];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert!(cli.local_model_only);
        assert!(!cli.client);
    }

    #[test]
    fn unknown_top_level_command_is_captured_for_plugin_dispatch() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "goose-next",
            "--model",
            "auto",
            "--",
            "prompt.txt",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        match cli.command.expect("external plugin command expected") {
            Command::ExternalPlugin(args) => {
                assert_eq!(
                    args,
                    vec![
                        OsString::from("goose-next"),
                        OsString::from("--model"),
                        OsString::from("auto"),
                        OsString::from("--"),
                        OsString::from("prompt.txt"),
                    ]
                );
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn cli_defaults_log_format_to_pretty() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Pretty);
    }

    #[test]
    fn cli_accepts_json_log_format() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "serve", "--log-format", "json", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
    }

    #[test]
    fn cli_accepts_global_log_format_before_serve() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "--log-format", "json", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_log_format_before_serve_with_model() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--log-format",
            "json",
            "serve",
            "--model",
            "Qwen3-8B-Q4_K_M",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(cli.model, vec![std::path::PathBuf::from("Qwen3-8B-Q4_K_M")]);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_log_format_equals_before_serve() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "--log-format=json", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_log_format_before_client() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--log-format",
            "json",
            "client",
            "--auto",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
    }

    #[test]
    fn cli_accepts_global_bind_ip_before_serve() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--bind-ip",
            "10.1.2.3",
            "serve",
            "--bind-port",
            "47916",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.bind_ip, Some("10.1.2.3".parse().unwrap()));
        assert_eq!(cli.bind_port, Some(47916));
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_mesh_discovery_mode_before_serve() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--mesh-discovery-mode",
            "mdns",
            "serve",
            "--auto",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.mesh_discovery_mode, MeshDiscoveryMode::Mdns);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_defaults_mesh_discovery_mode_to_nostr() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.mesh_discovery_mode, MeshDiscoveryMode::Nostr);
    }

    #[test]
    fn cli_accepts_mdns_discovery_mode_for_runtime_surfaces() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "client", "--mesh-discovery-mode", "mdns"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert!(cli.client);
        assert_eq!(cli.mesh_discovery_mode, MeshDiscoveryMode::Mdns);
    }
}
