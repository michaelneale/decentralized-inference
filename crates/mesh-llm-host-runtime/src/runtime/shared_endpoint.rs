//! Startup registration for `mesh-llm share <url>`.
//!
//! Sharing is fixed for the process lifetime: the upstream is validated and
//! probed once here, and the plugin manager's ordinary endpoint health
//! machinery owns it from then on. There is no attach/detach control surface —
//! stopping this process stops sharing, and never touches the upstream server.

use super::RuntimeOptions;
use crate::plugin;
use anyhow::{Context, Result, ensure};

/// Probe the configured upstream and seed its health record.
///
/// Returns the model names the upstream reported, for the startup log. An
/// unreachable upstream is an error: `mesh-llm share` against a server that is
/// not running must say so rather than joining the mesh advertising nothing.
pub(super) async fn start_from_options(
    options: &RuntimeOptions,
    plugin_manager: &plugin::PluginManager,
) -> Result<Option<Vec<String>>> {
    let Some(address) = &options.shared_endpoint else {
        return Ok(None);
    };
    reject_local_proxy(options, address).await?;
    let models = plugin_manager.start_shared_endpoint().await?;
    tracing::info!(
        models = ?models,
        "Sharing an external inference endpoint; local inference is disabled"
    );
    Ok(Some(models))
}

/// Refuse to forward to this node's own proxy or console.
///
/// Sharing our own listener would make every request loop back into itself.
/// The check is by resolved address rather than by string so `localhost`,
/// `127.0.0.1`, `[::1]`, and a wildcard bind are all caught.
async fn reject_local_proxy(options: &RuntimeOptions, address: &str) -> Result<()> {
    let url = plugin::shared_endpoint::normalize_shared_endpoint_url(address)?;
    let port = url.port_or_known_default().unwrap_or(80);
    if port != options.port && (options.no_console || port != options.console) {
        return Ok(());
    }
    let host = url
        .host_str()
        .context("endpoint requires a host")?
        .trim_matches(['[', ']']);
    let addresses = tokio::time::timeout(
        std::time::Duration::from_secs(3),
        tokio::net::lookup_host((host, port)),
    )
    .await
    .context("endpoint DNS lookup timed out")?
    .context("endpoint DNS lookup failed")?;
    ensure!(
        !addresses.into_iter().any(|address| {
            address.ip().is_loopback()
                || address.ip().is_unspecified()
                || Some(address.ip()) == options.bind_ip
                // A wildcard listener also answers on local interface addresses.
                // Binding an ephemeral UDP socket tests local ownership without
                // enumerating hardware or sending traffic to the upstream.
                || std::net::UdpSocket::bind((address.ip(), 0)).is_ok()
        }),
        "cannot share this node's own proxy or console; choose the upstream server port"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_own_api_and_console_but_not_upstream_ports() {
        let options = RuntimeOptions::default();
        for host in ["localhost", "127.0.0.1", "[::1]", "0.0.0.0"] {
            for port in [options.port, options.console] {
                assert!(
                    reject_local_proxy(&options, &format!("http://{host}:{port}"))
                        .await
                        .is_err(),
                    "{host}:{port} is this node's own listener"
                );
            }
        }
        assert!(
            reject_local_proxy(&options, "http://localhost:11434")
                .await
                .is_ok()
        );
        let no_console = RuntimeOptions {
            no_console: true,
            ..options
        };
        assert!(
            reject_local_proxy(
                &no_console,
                &format!("http://localhost:{}", no_console.console)
            )
            .await
            .is_ok()
        );
    }

    #[tokio::test]
    async fn an_unsupported_url_is_rejected_before_any_lookup() {
        let options = RuntimeOptions::default();
        assert!(
            reject_local_proxy(&options, "https://api.example.com/v1")
                .await
                .is_err()
        );
    }
}
