use super::common::{
    ResponseRetryPolicy, RouteAttemptLoggingContext, RouteAttemptResult,
    retryable_route_result_from_error,
};
use super::dispatch::{RelayAttemptContext, relay_attempted_response};
use super::probe::probe_http_response;
use crate::logging::OpenAiRouteObserver;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::request_normalize::ResponseAdapter;
use anyhow::{Context, Result};
use mesh_llm_events::logging::identifiers::RequestId;
use tokio::io::AsyncWriteExt;
use url::Url;

struct ExternalEndpointTarget {
    host: String,
    port: u16,
    forwarded: Vec<u8>,
}

pub(in crate::network::openai) async fn route_http_endpoint_attempt(
    tcp_stream: &mut ClientStream,
    base_url: &str,
    prefetched: &[u8],
    request_path: &str,
    strip_caller_credentials: bool,
    logging: RouteAttemptLoggingContext<'_>,
) -> RouteAttemptResult {
    let RouteAttemptLoggingContext {
        request_id,
        retry_policy,
        response_adapter,
        route_observer,
    } = logging;
    let target = match build_external_endpoint_target(
        base_url,
        request_path,
        prefetched,
        strip_caller_credentials,
    ) {
        Ok(target) => target,
        Err(()) => return RouteAttemptResult::RetryableUnavailable,
    };
    let mut upstream = match connect_external_endpoint(base_url, &target).await {
        Ok(upstream) => upstream,
        Err(result) => return result,
    };
    if let Err(result) = forward_external_endpoint_request(&mut upstream, base_url, &target).await {
        return result;
    }
    route_http_endpoint_attempt_after_forward(
        tcp_stream,
        &mut upstream,
        base_url,
        request_id,
        retry_policy,
        response_adapter,
        route_observer,
    )
    .await
}

async fn connect_external_endpoint(
    base_url: &str,
    target: &ExternalEndpointTarget,
) -> std::result::Result<ClientStream, RouteAttemptResult> {
    match ClientStream::connect(format!("{}:{}", target.host, target.port)).await {
        Ok(upstream) => Ok(upstream),
        Err(err) => {
            tracing::warn!(
                "API proxy: can't reach external inference endpoint {}: {}",
                base_url,
                err
            );
            Err(if err.kind() == std::io::ErrorKind::TimedOut {
                RouteAttemptResult::RetryableTimeout
            } else {
                RouteAttemptResult::RetryableUnavailable
            })
        }
    }
}

async fn forward_external_endpoint_request(
    upstream: &mut ClientStream,
    base_url: &str,
    target: &ExternalEndpointTarget,
) -> std::result::Result<(), RouteAttemptResult> {
    let _ = upstream.set_nodelay(true);
    match upstream.write_all(&target.forwarded).await {
        Ok(()) => Ok(()),
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to forward buffered request to external endpoint {}: {}",
                base_url,
                err
            );
            Err(RouteAttemptResult::RetryableUnavailable)
        }
    }
}

async fn route_http_endpoint_attempt_after_forward(
    tcp_stream: &mut ClientStream,
    upstream: &mut ClientStream,
    base_url: &str,
    request_id: RequestId,
    retry_policy: ResponseRetryPolicy,
    response_adapter: ResponseAdapter,
    route_observer: OpenAiRouteObserver<'_>,
) -> RouteAttemptResult {
    match probe_http_response(upstream).await {
        Ok(probe) => {
            let result = relay_attempted_response(
                tcp_stream,
                upstream,
                probe,
                RelayAttemptContext {
                    request_id,
                    disconnect_message: "API proxy (external endpoint): downstream client disconnected during relay",
                    commit_message: "API proxy (external endpoint) ended after commit",
                    route_observer,
                },
                retry_policy,
                response_adapter,
            )
            .await;
            if matches!(result, RouteAttemptResult::ClientDisconnected) {
                let _ = upstream.shutdown().await;
            }
            result
        }
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to read response from external endpoint {}: {}",
                base_url,
                err
            );
            retryable_route_result_from_error(&err)
        }
    }
}

fn build_external_endpoint_target(
    base_url: &str,
    request_path: &str,
    prefetched: &[u8],
    strip_caller_credentials: bool,
) -> std::result::Result<ExternalEndpointTarget, ()> {
    let (url, host) = parse_external_endpoint_url(base_url)?;
    let port = url.port_or_known_default().unwrap_or(80);
    let forward_path = endpoint_forward_path(&url, request_path);
    let forwarded = rewrite_external_endpoint_request(
        base_url,
        prefetched,
        &forward_path,
        &host,
        port,
        strip_caller_credentials,
    )?;
    Ok(ExternalEndpointTarget {
        host,
        port,
        forwarded,
    })
}

fn parse_external_endpoint_url(base_url: &str) -> std::result::Result<(Url, String), ()> {
    let url = parse_external_endpoint_base_url(base_url)?;
    validate_external_endpoint_scheme(base_url, &url)?;
    let host = parse_external_endpoint_host(base_url, &url)?;
    Ok((url, host))
}

fn parse_external_endpoint_base_url(base_url: &str) -> std::result::Result<Url, ()> {
    Url::parse(base_url).map_err(|err| {
        tracing::warn!("API proxy: invalid external inference endpoint '{base_url}': {err}");
    })
}

fn validate_external_endpoint_scheme(base_url: &str, url: &Url) -> std::result::Result<(), ()> {
    if url.scheme() == "http" {
        return Ok(());
    }
    tracing::warn!(
        "API proxy: unsupported external inference endpoint scheme '{}' for {}",
        url.scheme(),
        base_url
    );
    Err(())
}

fn parse_external_endpoint_host(base_url: &str, url: &Url) -> std::result::Result<String, ()> {
    url.host_str().map(str::to_string).ok_or_else(|| {
        tracing::warn!("API proxy: missing host in external inference endpoint {base_url}");
    })
}

fn rewrite_external_endpoint_request(
    base_url: &str,
    prefetched: &[u8],
    forward_path: &str,
    host: &str,
    port: u16,
    strip_caller_credentials: bool,
) -> std::result::Result<Vec<u8>, ()> {
    match rewrite_http_request_target(
        prefetched,
        forward_path,
        host,
        port,
        strip_caller_credentials,
    ) {
        Ok(forwarded) => Ok(forwarded),
        Err(err) => {
            tracing::warn!(
                "API proxy: failed to rewrite buffered request for external endpoint {}: {}",
                base_url,
                err
            );
            Err(())
        }
    }
}

fn endpoint_forward_path(base_url: &Url, request_path: &str) -> String {
    let (path_only, query) = request_path
        .split_once('?')
        .map(|(path, query)| (path, Some(query)))
        .unwrap_or((request_path, None));
    let base_path = base_url.path().trim_end_matches('/');
    let mapped_path = if base_path.is_empty() || base_path == "/" {
        path_only.to_string()
    } else if let Some(suffix) = path_only.strip_prefix("/v1") {
        if base_path.ends_with("/v1") {
            format!("{base_path}{suffix}")
        } else {
            format!("{base_path}/v1{suffix}")
        }
    } else if let Some(suffix) = path_only.strip_prefix("/models") {
        format!("{base_path}{suffix}")
    } else {
        format!("{base_path}{path_only}")
    };
    match query {
        Some(query) if !query.is_empty() => format!("{mapped_path}?{query}"),
        _ => mapped_path,
    }
}

fn rewrite_http_request_target(
    raw: &[u8],
    new_path: &str,
    host: &str,
    port: u16,
    strip_caller_credentials: bool,
) -> Result<Vec<u8>> {
    use crate::network::openai::forwarded_request::finalize_forwarded_request;
    let omitted = if strip_caller_credentials {
        &[
            "host",
            "authorization",
            "proxy-authorization",
            "x-api-key",
            "api-key",
        ][..]
    } else {
        &["host"][..]
    };
    let mut forwarded = finalize_forwarded_request(raw, false, Some(new_path), None, omitted)?;
    let line_end = forwarded
        .windows(2)
        .position(|bytes| bytes == b"\r\n")
        .context("missing HTTP request line")?
        + 2;
    let host_header = format!("Host: {host}:{port}\r\n");
    forwarded.splice(line_end..line_end, host_header.bytes());
    Ok(forwarded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_endpoint_keeps_its_existing_authentication_headers() {
        let raw = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer token\r\nX-Api-Key: key\r\nContent-Length: 2\r\n\r\n{}";
        let forwarded =
            rewrite_http_request_target(raw, "/api/v1/chat/completions", "localhost", 8000, false)
                .unwrap();
        let headers = String::from_utf8(forwarded).unwrap();
        assert!(headers.contains("Authorization: Bearer token\r\n"));
        assert!(headers.contains("X-Api-Key: key\r\n"));
    }

    #[test]
    fn shared_endpoint_strips_credentials_and_preserves_header_bytes() {
        let raw = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nX-Api-Key: secret\r\nProxy-Authorization: secret\r\nApi-Key: secret\r\nX-Trace: \xff\r\nContent-Length: 2\r\n\r\n{}";
        let rewritten =
            rewrite_http_request_target(raw, "/v1/chat/completions", "[::1]", 11434, true).unwrap();
        assert!(!rewritten.windows(6).any(|bytes| bytes == b"secret"));
        assert!(
            rewritten
                .windows(12)
                .any(|bytes| bytes == b"X-Trace: \xff\r\n")
        );
        assert!(rewritten.ends_with(b"\r\n\r\n{}"));
    }

    #[test]
    fn test_endpoint_forward_path_maps_v1_requests_onto_api_v1_base() {
        let url = Url::parse("http://localhost:8000/api/v1").unwrap();
        let forwarded = endpoint_forward_path(&url, "/v1/chat/completions?stream=true");
        assert_eq!(forwarded, "/api/v1/chat/completions?stream=true");
    }

    #[test]
    fn test_rewrite_http_request_target_updates_request_line_and_host() {
        let raw = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:9337\r\nx-request-id: 4c3ca94d-bc1f-4759-912d-f4f6d77d5515\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}";
        let rewritten =
            rewrite_http_request_target(raw, "/api/v1/chat/completions", "localhost", 8000, false)
                .unwrap();
        let rewritten = String::from_utf8(rewritten).unwrap();
        assert!(rewritten.starts_with("POST /api/v1/chat/completions HTTP/1.1\r\n"));
        assert!(rewritten.contains("\r\nHost: localhost:8000\r\n"));
        assert!(rewritten.contains("\r\nx-request-id: 4c3ca94d-bc1f-4759-912d-f4f6d77d5515\r\n"));
        assert!(rewritten.ends_with("\r\n\r\n{}"));
    }
}
