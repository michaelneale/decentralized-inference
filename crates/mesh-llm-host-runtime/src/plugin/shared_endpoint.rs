//! One operator-selected upstream, registered at startup and owned by the node
//! rather than by a plugin process.
//!
//! There is deliberately no lifecycle here. The address is fixed for the
//! lifetime of the `PluginManager`, and its health, model inventory, and route
//! entry come from the same machinery that serves configured plugin endpoints
//! (`super::health`).

use anyhow::{Result, ensure};
use url::Url;

/// Route identity for the shared endpoint. It is not a plugin and never
/// appears in the plugin list; these names only key its health record and
/// label its route.
pub(in crate::plugin) const SHARED_ENDPOINT_NAME: &str = "mesh-share";
pub(in crate::plugin) const SHARED_ENDPOINT_ID: &str = "openai";

/// Validate an operator-supplied upstream URL and normalize it to an API base.
///
/// Rejected input is never echoed back: it may contain credentials.
pub(crate) fn normalize_shared_endpoint_url(address: &str) -> Result<Url> {
    let mut url =
        Url::parse(address).map_err(|_| anyhow::anyhow!("expected an HTTP endpoint URL"))?;
    ensure!(
        url.scheme() == "http",
        "sharing currently supports HTTP upstreams only; HTTPS is not supported yet"
    );
    ensure!(url.host().is_some(), "endpoint URL requires a host");
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "credentials in endpoint URLs are not supported"
    );
    ensure!(
        url.query().is_none() && url.fragment().is_none(),
        "endpoint URL must not contain a query or fragment"
    );
    let path = url.path().trim_end_matches('/');
    ensure!(
        !path.ends_with("/models") && !path.ends_with("/chat/completions"),
        "provide the API base URL, not a models or chat route"
    );
    let path = if path.is_empty() {
        "/v1".to_string()
    } else if path.ends_with("/v1") {
        path.to_string()
    } else {
        format!("{path}/v1")
    };
    url.set_path(&path);
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_only_supported_base_urls() {
        for (input, expected) in [
            ("http://localhost:11434", "http://localhost:11434/v1"),
            ("http://localhost:11434/v1/", "http://localhost:11434/v1"),
            ("http://[::1]:8000/api/v1", "http://[::1]:8000/api/v1"),
        ] {
            assert_eq!(
                normalize_shared_endpoint_url(input).unwrap().as_str(),
                expected
            );
        }
    }

    #[test]
    fn rejects_unsupported_urls_without_echoing_them() {
        for input in [
            "https://example.com/v1",
            "file:///tmp/model",
            "http://u:secret@localhost",
            "http://localhost?key=secret",
            "http://localhost/#secret",
            "http://localhost/v1/models",
            "not a url",
        ] {
            let error = normalize_shared_endpoint_url(input)
                .unwrap_err()
                .to_string();
            assert!(!error.contains("secret"), "leaked input: {error}");
        }
    }
}
