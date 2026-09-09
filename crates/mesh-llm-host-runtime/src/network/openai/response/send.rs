use crate::logging::OpenAiRouteObserver;
use crate::network::openai::client_stream::ClientStream;
use tokio::io::AsyncWriteExt;

/// RFC 7230 tchar set for header field names: ASCII alphanumeric plus
/// `!#$%&'*+-.^_`|~`. We additionally forbid `:` because it terminates
/// the field-name in the wire grammar. Used to reject caller-provided
/// header names that could carry CR/LF or other injection bytes.
pub(crate) fn is_valid_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(
                    b,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

/// Append a single `name: value` header line if `name` is a valid HTTP
/// header field name. CR/LF in `value` is stripped defensively. Used by
/// the `*_with_headers` writers below so a malformed header from a
/// future caller can't inject extra headers / smuggle a response.
pub(crate) fn append_safe_header(headers: &mut String, name: &str, value: &str) {
    if !is_valid_header_name(name) {
        tracing::warn!(
            "openai transport: dropping header with invalid name `{name}` (RFC 7230 tchar required)"
        );
        return;
    }
    let safe_value: String = value.chars().filter(|c| *c != '\r' && *c != '\n').collect();
    headers.push_str(name);
    headers.push_str(": ");
    headers.push_str(&safe_value);
    headers.push_str("\r\n");
}

/// Like `send_json_ok` but allows the caller to append arbitrary response
/// headers (e.g. `x-moa-*` observability headers).
///
/// Header names must satisfy the RFC 7230 tchar grammar (ASCII
/// alphanumeric + a small symbol set); invalid names are dropped with a
/// warning rather than written verbatim. Values are stripped of CR/LF.
pub async fn send_json_ok_with_headers(
    mut stream: ClientStream,
    data: &serde_json::Value,
    extra_headers: &[(&str, String)],
) -> std::io::Result<()> {
    let body = data.to_string();
    let mut headers = String::from("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n");
    for (name, value) in extra_headers {
        append_safe_header(&mut headers, name, value);
    }
    headers.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.shutdown().await?;
    Ok(())
}

/// Send a bounded JSON error and record it only after the exact client-visible
/// body has been written successfully.
pub(crate) async fn send_json_with_status_and_headers_observed(
    stream: ClientStream,
    code: u16,
    data: &serde_json::Value,
    extra_headers: &[(&str, String)],
    route_observer: OpenAiRouteObserver<'_>,
) -> std::io::Result<()> {
    send_json_with_status_and_headers_inner(stream, code, data, extra_headers, Some(route_observer))
        .await
}

async fn send_json_with_status_and_headers_inner(
    mut stream: ClientStream,
    code: u16,
    data: &serde_json::Value,
    extra_headers: &[(&str, String)],
    route_observer: Option<OpenAiRouteObserver<'_>>,
) -> std::io::Result<()> {
    let status = match code {
        400 => "Bad Request",
        404 => "Not Found",
        410 => "Gone",
        409 => "Conflict",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Error",
    };
    let body = data.to_string();
    let mut headers = format!("HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\n");
    for (name, value) in extra_headers {
        append_safe_header(&mut headers, name, value);
    }
    headers.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    stream.shutdown().await?;
    if let Some(route_observer) = route_observer {
        route_observer.capture_response_body(body.as_bytes(), Some("application/json"));
    }
    Ok(())
}

pub async fn send_400(stream: ClientStream, msg: &str) -> std::io::Result<()> {
    send_openai_error(stream, 400, msg, None).await
}

#[cfg(test)]
pub async fn send_error(stream: ClientStream, code: u16, msg: &str) -> std::io::Result<()> {
    send_openai_error(stream, code, msg, None).await
}

/// Write a locally generated bounded OpenAI JSON error and record the exact
/// body only after the client write succeeds. The observer remains the narrow
/// logging boundary: it receives no request headers or arbitrary media type.
pub(crate) async fn send_error_observed(
    stream: ClientStream,
    code: u16,
    msg: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> std::io::Result<()> {
    send_openai_error(stream, code, msg, Some(route_observer)).await
}

pub(crate) async fn send_400_observed(
    stream: ClientStream,
    msg: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> std::io::Result<()> {
    send_error_observed(stream, 400, msg, route_observer).await
}

pub(crate) async fn send_409_observed(
    stream: ClientStream,
    msg: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> std::io::Result<()> {
    send_error_observed(stream, 409, msg, route_observer).await
}

pub(crate) async fn send_503_observed(
    stream: ClientStream,
    reason: &str,
    route_observer: OpenAiRouteObserver<'_>,
) -> std::io::Result<()> {
    tracing::warn!("503 → client: {reason}");
    send_error_observed(stream, 503, reason, route_observer).await
}

async fn send_openai_error(
    mut stream: ClientStream,
    code: u16,
    msg: &str,
    route_observer: Option<OpenAiRouteObserver<'_>>,
) -> std::io::Result<()> {
    let status = match code {
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        410 => "Gone",
        409 => "Conflict",
        413 => "Payload Too Large",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Bad Request",
    };
    let body = openai_error_body(code, msg);
    let retry_after = if code == 429 {
        "Retry-After: 5\r\n"
    } else {
        ""
    };
    let resp = format!(
        "HTTP/1.1 {code} {status}\r\nContent-Type: application/json\r\n{retry_after}Content-Length: {}\r\n\r\n{}",
        body.len(),
        String::from_utf8_lossy(&body)
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.shutdown().await?;
    if let Some(route_observer) = route_observer {
        route_observer.capture_response_body(&body, Some("application/json"));
    }
    Ok(())
}

pub async fn send_503(stream: ClientStream, reason: &str) -> std::io::Result<()> {
    tracing::warn!("503 → client: {reason}");
    send_openai_error(stream, 503, reason, None).await
}

fn openai_error_body(status_code: u16, message: &str) -> Vec<u8> {
    let status =
        http::StatusCode::from_u16(status_code).unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR);
    let kind = openai_error_kind_for_status(status_code);
    let error = openai_frontend::OpenAiError::from_kind(status, kind, message)
        .with_code(openai_error_code_for_status(status_code));
    serde_json::to_vec(&error.body()).expect("serializing JSON error response should not fail")
}

const fn openai_error_kind_for_status(status_code: u16) -> openai_frontend::OpenAiErrorKind {
    match status_code {
        401 => openai_frontend::OpenAiErrorKind::Authentication,
        403 => openai_frontend::OpenAiErrorKind::Permission,
        404 | 410 => openai_frontend::OpenAiErrorKind::NotFound,
        413 => openai_frontend::OpenAiErrorKind::PayloadTooLarge,
        429 => openai_frontend::OpenAiErrorKind::RateLimit,
        500 => openai_frontend::OpenAiErrorKind::Internal,
        502 => openai_frontend::OpenAiErrorKind::ServiceUnavailable,
        503 => openai_frontend::OpenAiErrorKind::ServiceUnavailable,
        504 => openai_frontend::OpenAiErrorKind::Timeout,
        _ => openai_frontend::OpenAiErrorKind::InvalidRequest,
    }
}

const fn openai_error_code_for_status(status_code: u16) -> &'static str {
    match status_code {
        400 => "bad_request",
        401 => "invalid_api_key",
        403 => "permission_denied",
        404 => "model_not_found",
        410 => "legacy_route_gone",
        409 => "conflict",
        413 => "payload_too_large",
        422 => "unprocessable_content",
        429 => "rate_limit_exceeded",
        500 => "internal_server_error",
        502 => "service_unavailable",
        503 => "service_unavailable",
        504 => "timeout",
        _ => "invalid_request",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logging::{OpenAiArtifactCapture, OpenAiRouteObserver};
    use mesh_llm_events::logging::identifiers::RequestId;
    use std::future::Future;
    use std::sync::{Arc, Mutex};
    use tokio::net::TcpListener;

    #[test]
    fn is_valid_header_name_accepts_normal_observability_headers() {
        assert!(is_valid_header_name("x-moa-elapsed-ms"));
        assert!(is_valid_header_name("X-MoA-Workers"));
        assert!(is_valid_header_name("Content-Type"));
        assert!(is_valid_header_name("x-request-id"));
    }

    #[test]
    fn is_valid_header_name_rejects_injection_attempts() {
        // Regression for PR #566 review item #5c: header NAMES were not
        // sanitized, only values. A name carrying CR/LF or a colon would
        // smuggle extra headers / split the response.
        assert!(!is_valid_header_name("x-evil\r\nSet-Cookie"));
        assert!(!is_valid_header_name("x-evil\nSet-Cookie"));
        assert!(!is_valid_header_name("x-evil: hijacked"));
        assert!(!is_valid_header_name("x evil")); // space inside name
        assert!(!is_valid_header_name(""));
    }

    #[test]
    fn append_safe_header_drops_invalid_name() {
        let mut buf = String::new();
        append_safe_header(&mut buf, "x-evil\r\nSet-Cookie", "bad");
        assert!(buf.is_empty(), "invalid name must be dropped, got {buf:?}");
    }

    #[test]
    fn append_safe_header_strips_crlf_from_value() {
        let mut buf = String::new();
        append_safe_header(&mut buf, "x-ok", "ok\r\nSet-Cookie: hijack");
        assert!(
            buf.starts_with("x-ok: okSet-Cookie: hijack\r\n"),
            "value CRLF must be stripped; got {buf:?}"
        );
        assert_eq!(buf.matches("\r\n").count(), 1);
    }

    #[tokio::test]
    async fn test_send_error_429_includes_retry_after() {
        let response = capture_proxy_error_response(|stream| async move {
            super::send_error(stream, 429, "model not available").await
        })
        .await;
        let body = response_json_body(&response);

        assert!(response.starts_with("HTTP/1.1 429 Too Many Requests\r\n"));
        assert!(response.contains("Retry-After: 5\r\n"));
        assert_eq!(body["error"]["message"], "model not available");
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(body["error"]["code"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn test_send_503_uses_openai_error_shape() {
        let response = capture_proxy_error_response(|stream| async move {
            super::send_503(stream, "skippy ABI call failed: Unsupported").await
        })
        .await;
        let body = response_json_body(&response);

        assert!(response.starts_with("HTTP/1.1 503 Service Unavailable\r\n"));
        assert_eq!(
            body["error"]["message"],
            "skippy ABI call failed: Unsupported"
        );
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "service_unavailable");
    }

    type CapturedArtifact = (String, Vec<u8>, Option<String>);

    #[derive(Default)]
    struct Captures(Mutex<Vec<CapturedArtifact>>);

    impl OpenAiArtifactCapture for Captures {
        fn capture_body(
            &self,
            _request_id: RequestId,
            kind: &'static str,
            content: &[u8],
            media_kind: Option<&str>,
        ) {
            self.0.lock().unwrap().push((
                kind.to_owned(),
                content.to_vec(),
                media_kind.map(str::to_owned),
            ));
        }

        fn capture_unavailable(
            &self,
            _request_id: RequestId,
            _kind: &'static str,
            _reason: crate::logging::ArtifactUnavailableReason,
        ) {
        }
    }

    #[tokio::test]
    async fn observed_error_captures_exact_json_only_after_a_successful_write() {
        let captures = Arc::new(Captures::default());
        let capture: Arc<dyn OpenAiArtifactCapture> = captures.clone();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let capture = capture.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let stream: ClientStream = stream.into();
            super::send_error_observed(
                stream,
                404,
                "model missing",
                OpenAiRouteObserver::capture_test_observer(RequestId::new(), &capture),
            )
            .await
            .unwrap();
        });

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        tokio::io::AsyncReadExt::read_to_end(&mut client, &mut output)
            .await
            .unwrap();
        server.await.unwrap();
        let body = response_json_body(std::str::from_utf8(&output).unwrap());
        let body_bytes = &output[output
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .unwrap()
            + 4..];
        assert_eq!(body["error"]["message"], "model missing");
        let recorded = captures.0.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0, "response");
        assert_eq!(recorded[0].1, body_bytes);
        assert_eq!(recorded[0].2.as_deref(), Some("application/json"));
    }
    async fn capture_proxy_error_response<F, Fut>(send: F) -> String
    where
        F: FnOnce(ClientStream) -> Fut + Send + 'static,
        Fut: Future<Output = std::io::Result<()>> + Send + 'static,
    {
        use tokio::io::AsyncReadExt;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let stream: ClientStream = stream.into();
            send(stream).await.unwrap();
        });

        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        client.read_to_end(&mut output).await.unwrap();
        server.await.unwrap();
        String::from_utf8(output).unwrap()
    }

    fn response_json_body(response: &str) -> serde_json::Value {
        let body_start = response
            .find("\r\n\r\n")
            .map(|index| index + 4)
            .expect("response contains header terminator");
        serde_json::from_str(&response[body_start..]).unwrap()
    }
}
