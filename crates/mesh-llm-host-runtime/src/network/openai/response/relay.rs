use super::cache_cost::parse_cache_cost_from_json_body;
use super::common::{
    ResponseRetryPolicy, RouteAttemptResult, parse_token_usage_from_json_body,
    retryable_quality_result,
};
use super::probe::MESH_SERVED_BY_HEADER;
use super::probe::{
    ParsedResponseHeaders, ResponseProbe, append_capsule_nonce_headers, insert_header_before_body,
    read_response_chunk, try_parse_response_headers,
};
use crate::logging::{ArtifactUnavailableReason, OpenAiRouteObserver};
use crate::network::openai::client_stream::ClientStream;
use anyhow::Result;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

const MAX_ERROR_RESPONSE_BYTES: usize = 256 * 1024;

fn http_body(response: &[u8]) -> &[u8] {
    response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map_or(&[][..], |header_end| &response[header_end + 4..])
}

fn reason_phrase(status_code: u16) -> &'static str {
    match status_code {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

pub(in crate::network::openai::response) fn remap_error_http_response(
    status_code: u16,
    header_end: usize,
    full_response: &[u8],
) -> Option<Vec<u8>> {
    if status_code < 400 || header_end > full_response.len() {
        return None;
    }
    let mapped_body =
        openai_frontend::map_upstream_error_body(status_code, &full_response[header_end..])?;
    // The upstream frontend echoes the capsule nonce on its error responses too;
    // rebuilding the header from scratch would drop it, so re-append whatever the
    // upstream sent. This keeps the nonce contract intact even when a llama.cpp
    // error body is remapped into an OpenAI-shaped one.
    let upstream = try_parse_response_headers(full_response).ok().flatten();
    let mut header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        status_code,
        reason_phrase(status_code),
        mapped_body.len()
    );
    if let Some(parsed) = &upstream {
        append_capsule_nonce_headers(
            &mut header,
            parsed.client_nonce.as_deref(),
            parsed.nonce_origin.as_deref(),
        );
    }
    header.push_str("\r\n");
    let mut response = header.into_bytes();
    response.extend_from_slice(&mapped_body);
    Some(response)
}

fn oversized_error_http_response(status_code: u16) -> Vec<u8> {
    let body = serde_json::json!({
        "error": {
            "message": "upstream error response exceeded proxy limit",
            "type": "server_error",
            "param": serde_json::Value::Null,
            "code": "upstream_error_too_large",
        }
    })
    .to_string();
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status_code,
        reason_phrase(status_code),
        body.len(),
        body
    )
    .into_bytes()
}

/// Byte offset just past the terminating `\r\n\r\n` of a freshly-built
/// response, for splicing an extra header into a buffer whose header block
/// wasn't tracked through the branch that produced it (oversized / remapped /
/// passthrough error bodies each build `outgoing` differently).
///
/// `None` when no `\r\n\r\n` terminator is found (e.g. an upstream that ends
/// its header block with a bare LF) -- returning `response.len()` here used
/// to look like a valid offset to `insert_header_before_body`, which would
/// then splice two bytes before the end of an already-complete response,
/// corrupting it silently instead of skipping the insert.
fn response_header_end(response: &[u8]) -> Option<usize> {
    response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|pos| pos + 4)
}

/// Splice `x-mesh-served-by` into an already-built error response, when set.
fn append_served_by_to_error_response(outgoing: &mut Vec<u8>, served_by: Option<&str>) {
    let Some(served_by) = served_by else { return };
    let Some(header_end) = response_header_end(outgoing) else {
        tracing::debug!(
            "no header terminator found while echoing x-mesh-served-by on an error response; skipping insert"
        );
        return;
    };
    insert_header_before_body(outgoing, header_end, MESH_SERVED_BY_HEADER, served_by);
}

/// Relay a non-2xx upstream response, echoing `x-mesh-served-by` when set.
pub(in crate::network::openai::response) async fn relay_error_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<RouteAttemptResult> {
    let status_code = probe.status_code;
    let header_end = probe.header_end;
    let mut buffered = probe.buffered;
    let mut limited = reader.take((MAX_ERROR_RESPONSE_BYTES + 1) as u64);
    if let Err(err) = limited.read_to_end(&mut buffered).await {
        tracing::debug!("error response relay read ended before EOF: {err}");
    }
    let mut outgoing = if buffered.len().saturating_sub(header_end) > MAX_ERROR_RESPONSE_BYTES {
        tracing::warn!(
            "upstream error body exceeded {} bytes for status {}",
            MAX_ERROR_RESPONSE_BYTES,
            status_code
        );
        oversized_error_http_response(status_code)
    } else {
        remap_error_http_response(status_code, header_end, &buffered).unwrap_or(buffered)
    };
    append_served_by_to_error_response(&mut outgoing, served_by);
    tcp_stream.write_all(&outgoing).await?;
    let media_kind = try_parse_response_headers(&outgoing)
        .ok()
        .flatten()
        .and_then(|headers| headers.content_type);
    route_observer.capture_response_body(http_body(&outgoing), media_kind.as_deref());
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code,
        usage: None,
        cache_cost: None,
    })
}

pub(in crate::network::openai::response) async fn relay_success_response<R: AsyncRead + Unpin>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    parsed: ParsedResponseHeaders,
    retry_policy: ResponseRetryPolicy,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<RouteAttemptResult> {
    if let Some(content_length) = parsed.content_length {
        const MAX_SUCCESS_METRICS_BODY_BYTES: usize = 1024 * 1024;
        if content_length <= MAX_SUCCESS_METRICS_BODY_BYTES {
            let mut buffered = probe.buffered;
            let body_end = parsed
                .header_end
                .checked_add(content_length)
                .ok_or_else(|| anyhow::anyhow!("upstream Content-Length overflow"))?;
            while buffered.len() < body_end {
                read_response_chunk(reader, &mut buffered).await?;
            }
            let body = &buffered[parsed.header_end..body_end];
            if let Some(result) = retryable_quality_result(body, retry_policy) {
                return Ok(result);
            }
            let usage = parse_token_usage_from_json_body(body);
            let cache_cost = parse_cache_cost_from_json_body(body);
            let body_len = body.len();
            let mut outgoing_end = body_end;
            if let Some(served_by) = served_by {
                let delta = insert_header_before_body(
                    &mut buffered,
                    parsed.header_end,
                    MESH_SERVED_BY_HEADER,
                    served_by,
                );
                outgoing_end = outgoing_end.saturating_add_signed(delta);
            }
            // Reads may include bytes beyond the declared HTTP body. Only the
            // declared response is client-visible and capturable.
            tcp_stream.write_all(&buffered[..outgoing_end]).await?;
            route_observer.capture_response_body(
                &buffered[outgoing_end - body_len..outgoing_end],
                parsed.content_type.as_deref(),
            );
            let _ = tcp_stream.shutdown().await;
            return Ok(RouteAttemptResult::Delivered {
                status_code: probe.status_code,
                usage,
                cache_cost,
            });
        }
    }

    let mut buffered = probe.buffered;
    if let Some(served_by) = served_by {
        insert_header_before_body(
            &mut buffered,
            parsed.header_end,
            MESH_SERVED_BY_HEADER,
            served_by,
        );
    }
    tcp_stream.write_all(&buffered).await?;
    route_observer.capture_response_unavailable(ArtifactUnavailableReason::ResponseBodyNotBounded);
    if let Err(err) = tokio::io::copy(reader, &mut *tcp_stream).await {
        tracing::debug!("response relay ended after headers were committed: {err}");
    }
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: probe.status_code,
        usage: None,
        cache_cost: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_header_end_finds_the_terminator() {
        let response = b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(response_header_end(response), Some(response.len()));
    }

    /// Regression (erlich, PR #1671 round 2): a response whose header block
    /// ends in a bare LF (no `\r\n\r\n`) must report `None`, not
    /// `response.len()` -- the old fallback looked like a valid offset to
    /// `insert_header_before_body` and caused it to splice two bytes before
    /// the end of an already-complete response.
    #[test]
    fn response_header_end_is_none_without_a_terminator() {
        let response = b"HTTP/1.1 500 Internal Server Error\nContent-Length: 0\n\n{}";
        assert_eq!(response_header_end(response), None);
    }

    /// Regression (erlich, PR #1671 round 2): with no terminator found,
    /// `append_served_by_to_error_response` must be a no-op -- never splice
    /// at `buf.len() - 2`, which would corrupt whatever bytes are there
    /// (here, the tail of the body) instead of skipping the insert.
    #[test]
    fn append_served_by_to_error_response_skips_insert_without_a_terminator() {
        let original = b"HTTP/1.1 500 Internal Server Error\nContent-Length: 2\n\n{}".to_vec();
        let mut outgoing = original.clone();
        append_served_by_to_error_response(&mut outgoing, Some("ab12cd34"));
        assert_eq!(
            outgoing, original,
            "no header terminator means the served-by insert must be skipped entirely"
        );
    }

    use crate::logging::{OpenAiArtifactCapture, OpenAiRouteObserver};
    use mesh_llm_events::logging::identifiers::RequestId;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    type CaptureRecord = (String, Vec<u8>, Option<String>);

    #[derive(Default)]
    struct Captures(Mutex<Vec<CaptureRecord>>);

    impl OpenAiArtifactCapture for Captures {
        fn capture_body(
            &self,
            _request_id: RequestId,
            kind: &'static str,
            content: &[u8],
            media_kind: Option<&str>,
        ) {
            self.0.lock().unwrap().push((
                kind.to_string(),
                content.to_vec(),
                media_kind.map(str::to_owned),
            ));
        }

        fn capture_unavailable(
            &self,
            _request_id: RequestId,
            _kind: &'static str,
            _reason: ArtifactUnavailableReason,
        ) {
        }
    }

    #[test]
    fn test_remap_error_http_response_rewrites_llama_error_body() {
        let upstream = b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 52\r\n\r\n{\"type\":\"not_found_error\",\"message\":\"model missing\"}";
        let header_end = upstream
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|idx| idx + 4)
            .unwrap();
        let remapped = remap_error_http_response(404, header_end, upstream)
            .expect("llama error should be remapped");
        let remapped_text = String::from_utf8(remapped).unwrap();

        assert!(remapped_text.starts_with("HTTP/1.1 404 Not Found\r\n"));
        assert!(remapped_text.contains("\r\nContent-Type: application/json\r\n"));
        assert!(remapped_text.contains("\"type\":\"invalid_request_error\""));
        assert!(remapped_text.contains("\"code\":\"model_not_found\""));
        assert!(remapped_text.contains("\"message\":\"model missing\""));
    }

    #[test]
    fn test_remap_error_http_response_preserves_upstream_nonce_headers() {
        let nonce_header = openai_frontend::lifecycle::CLIENT_NONCE_HEADER.as_str();
        let origin_header = openai_frontend::lifecycle::CLIENT_NONCE_ORIGIN_HEADER.as_str();
        let upstream = format!(
            "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n{nonce_header}: 11111111-1111-4111-8111-111111111111\r\n{origin_header}: frontend\r\nContent-Length: 52\r\n\r\n{{\"type\":\"not_found_error\",\"message\":\"model missing\"}}"
        );
        let upstream = upstream.into_bytes();
        let header_end = upstream
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|idx| idx + 4)
            .unwrap();
        let remapped = remap_error_http_response(404, header_end, &upstream)
            .expect("llama error should be remapped");
        let remapped_text = String::from_utf8(remapped).unwrap();

        assert!(
            remapped_text.contains(&format!(
                "{nonce_header}: 11111111-1111-4111-8111-111111111111\r\n"
            )),
            "the upstream-echoed nonce must survive an error remap: {remapped_text}"
        );
        assert!(
            remapped_text.contains(&format!("{origin_header}: frontend\r\n")),
            "the upstream-echoed origin marker must survive an error remap: {remapped_text}"
        );
        // The body is still remapped to the OpenAI shape.
        assert!(remapped_text.contains("\"code\":\"model_not_found\""));
    }

    #[test]
    fn test_remap_error_http_response_keeps_openai_error_passthrough() {
        let upstream = b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 110\r\n\r\n{\"error\":{\"message\":\"bad request\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":\"invalid_value\"}}";
        let header_end = upstream
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|idx| idx + 4)
            .unwrap();
        assert!(remap_error_http_response(400, header_end, upstream).is_none());
    }

    #[tokio::test]
    async fn relay_success_captures_client_visible_non_stream_body() {
        let body = br#"{"id":"chatcmpl-safe","usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}"#;
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let captured = Arc::new(Captures::default());
        let captures: Arc<dyn OpenAiArtifactCapture> = captured.clone();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            let observer = OpenAiRouteObserver::capture_test_observer(RequestId::new(), &captures);
            relay_success_response(
                &mut client,
                &mut upstream_reader,
                ResponseProbe {
                    buffered: header.clone().into_bytes(),
                    header_end: header.len(),
                    status_code: 200,
                    retryable_context_overflow: false,
                },
                ParsedResponseHeaders {
                    header_end: header.len(),
                    status_code: 200,
                    content_length: Some(body.len()),
                    content_type: Some("application/json".to_owned()),
                    client_nonce: None,
                    nonce_origin: None,
                },
                ResponseRetryPolicy::next_target_available(false),
                None,
                observer,
            )
            .await
            .unwrap();
        });
        upstream_writer.write_all(body).await.unwrap();
        drop(upstream_writer);
        let mut socket = ClientStream::connect(address).await.unwrap();
        let mut client_response = Vec::new();
        socket.read_to_end(&mut client_response).await.unwrap();
        task.await.unwrap();

        let captures = captured.0.lock().unwrap();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].0, "response");
        assert_eq!(captures[0].1, body);
        assert_eq!(captures[0].2.as_deref(), Some("application/json"));
        assert!(client_response.ends_with(body));
    }

    #[tokio::test]
    async fn relay_success_echoes_served_by_header_only_when_set() {
        let body = br#"{"id":"chatcmpl-safe"}"#;
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let parsed = ParsedResponseHeaders {
            header_end: header.len(),
            status_code: 200,
            content_length: Some(body.len()),
            content_type: Some("application/json".to_owned()),
            client_nonce: None,
            nonce_origin: None,
        };
        let probe_for = |header: &str| ResponseProbe {
            buffered: header.as_bytes().to_vec(),
            header_end: header.len(),
            status_code: 200,
            retryable_context_overflow: false,
        };

        // `x-mesh-target` was used: the resolved peer must be echoed back.
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task_header = header.clone();
        let task_parsed = ParsedResponseHeaders {
            header_end: parsed.header_end,
            status_code: parsed.status_code,
            content_length: parsed.content_length,
            content_type: parsed.content_type.clone(),
            client_nonce: None,
            nonce_origin: None,
        };
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            relay_success_response(
                &mut client,
                &mut upstream_reader,
                probe_for(&task_header),
                task_parsed,
                ResponseRetryPolicy::next_target_available(false),
                Some("ab12cd34"),
                OpenAiRouteObserver::default(),
            )
            .await
            .unwrap();
        });
        upstream_writer.write_all(body).await.unwrap();
        drop(upstream_writer);
        let mut socket = ClientStream::connect(address).await.unwrap();
        let mut with_target_response = Vec::new();
        socket.read_to_end(&mut with_target_response).await.unwrap();
        task.await.unwrap();
        let with_target_text = String::from_utf8_lossy(&with_target_response);
        assert!(
            with_target_text.contains("x-mesh-served-by: ab12cd34\r\n"),
            "x-mesh-target dispatch must echo the resolved peer: {with_target_text}"
        );
        assert!(with_target_response.ends_with(body));

        // Absent `x-mesh-target`: today's response, byte-for-byte -- no
        // `x-mesh-served-by` line added.
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task_header = header.clone();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            relay_success_response(
                &mut client,
                &mut upstream_reader,
                probe_for(&task_header),
                parsed,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
            )
            .await
            .unwrap();
        });
        upstream_writer.write_all(body).await.unwrap();
        drop(upstream_writer);
        let mut socket = ClientStream::connect(address).await.unwrap();
        let mut without_target_response = Vec::new();
        socket
            .read_to_end(&mut without_target_response)
            .await
            .unwrap();
        task.await.unwrap();
        assert!(!String::from_utf8_lossy(&without_target_response).contains("x-mesh-served-by"));
        let mut expected = header.into_bytes();
        expected.extend_from_slice(body);
        assert_eq!(
            without_target_response, expected,
            "absent x-mesh-target must relay today's response byte-for-byte"
        );
    }

    #[tokio::test]
    async fn relay_error_echoes_served_by_header_only_when_set() {
        let body = br#"{"error":{"message":"boom","type":"server_error","param":null,"code":"upstream_failed"}}"#;
        let header = format!(
            "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let probe_for = |header: &str| ResponseProbe {
            buffered: header.as_bytes().to_vec(),
            header_end: header.len(),
            status_code: 500,
            retryable_context_overflow: false,
        };

        // A non-2xx response from a resolved `x-mesh-target` peer must still
        // echo `x-mesh-served-by` -- the client asked to reach that peer
        // specifically and needs to know the error came from there.
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task_header = header.clone();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            relay_error_response(
                &mut client,
                &mut upstream_reader,
                probe_for(&task_header),
                Some("ab12cd34"),
                OpenAiRouteObserver::default(),
            )
            .await
            .unwrap();
        });
        upstream_writer.write_all(body).await.unwrap();
        drop(upstream_writer);
        let mut socket = ClientStream::connect(address).await.unwrap();
        let mut with_served_by = Vec::new();
        socket.read_to_end(&mut with_served_by).await.unwrap();
        task.await.unwrap();
        let with_served_by_text = String::from_utf8_lossy(&with_served_by);
        assert!(
            with_served_by_text.contains("x-mesh-served-by: ab12cd34\r\n"),
            "a non-2xx response from a resolved peer must echo x-mesh-served-by: {with_served_by_text}"
        );
        assert!(with_served_by.ends_with(body));

        // Absent `served_by`: no header line added.
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task_header = header.clone();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            relay_error_response(
                &mut client,
                &mut upstream_reader,
                probe_for(&task_header),
                None,
                OpenAiRouteObserver::default(),
            )
            .await
            .unwrap();
        });
        upstream_writer.write_all(body).await.unwrap();
        drop(upstream_writer);
        let mut socket = ClientStream::connect(address).await.unwrap();
        let mut without_served_by = Vec::new();
        socket.read_to_end(&mut without_served_by).await.unwrap();
        task.await.unwrap();
        assert!(!String::from_utf8_lossy(&without_served_by).contains("x-mesh-served-by"));
    }

    #[tokio::test]
    async fn relay_success_excludes_bytes_read_past_declared_content_length() {
        let body = br#"{"id":"chatcmpl-safe"}"#;
        let overread = b"NEXT-RESPONSE-MUST-NOT-BE-CAPTURED";
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let mut buffered = header.clone().into_bytes();
        buffered.extend_from_slice(body);
        buffered.extend_from_slice(overread);
        let captured = Arc::new(Captures::default());
        let captures: Arc<dyn OpenAiArtifactCapture> = captured.clone();
        let task_header = header.clone();
        let task = tokio::spawn(async move {
            let (client, _) = listener.accept().await.unwrap();
            let mut client: ClientStream = client.into();
            let observer = OpenAiRouteObserver::capture_test_observer(RequestId::new(), &captures);
            let mut empty_reader = tokio::io::empty();
            relay_success_response(
                &mut client,
                &mut empty_reader,
                ResponseProbe {
                    buffered,
                    header_end: task_header.len(),
                    status_code: 200,
                    retryable_context_overflow: false,
                },
                ParsedResponseHeaders {
                    header_end: task_header.len(),
                    status_code: 200,
                    content_length: Some(body.len()),
                    content_type: Some("application/json; charset=utf-8".to_owned()),
                    client_nonce: None,
                    nonce_origin: None,
                },
                ResponseRetryPolicy::next_target_available(false),
                None,
                observer,
            )
            .await
            .unwrap();
        });
        let mut socket = ClientStream::connect(address).await.unwrap();
        let mut client_response = Vec::new();
        socket.read_to_end(&mut client_response).await.unwrap();
        task.await.unwrap();

        assert_eq!(client_response, [header.as_bytes(), body].concat());
        let captures = captured.0.lock().unwrap();
        assert_eq!(captures[0].1, body);
        assert!(
            !captures[0]
                .1
                .windows(overread.len())
                .any(|part| part == overread)
        );
    }
}
