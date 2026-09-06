use super::cache_cost::parse_cache_cost_from_json_body;
use super::common::{
    ResponseRetryPolicy, RouteAttemptResult, parse_token_usage_from_json_body,
    retryable_quality_result,
};
use super::probe::{
    ResponseBodyReadLimits, ResponseProbe, append_capsule_nonce_headers,
    append_mesh_served_by_header, read_transformed_response_body, try_parse_response_headers,
};
use super::relay::relay_error_response;
use crate::logging::OpenAiRouteObserver;
use crate::network::openai::client_stream::ClientStream;
use crate::network::openai::response_adapter;
use crate::network::openai::tool_call_ids::normalize_chat_completion_json_body;
use anyhow::{Result, anyhow};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWriteExt};

const MAX_TRANSFORMED_RESPONSE_BODY_BYTES: usize = 8 * 1024 * 1024;
const TRANSFORMED_RESPONSE_BODY_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
const TRANSFORMED_RESPONSE_READ_LIMITS: ResponseBodyReadLimits = ResponseBodyReadLimits {
    max_body_bytes: MAX_TRANSFORMED_RESPONSE_BODY_BYTES,
    idle_timeout: TRANSFORMED_RESPONSE_BODY_IDLE_TIMEOUT,
};

/// Relay a chat-completions upstream response translated into Responses-API JSON.
pub(in crate::network::openai::response) async fn relay_translated_responses_json<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<RouteAttemptResult> {
    if retry_policy.context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe, served_by, route_observer).await;
    }
    let mut buffered = probe.buffered;
    let parsed = try_parse_response_headers(&buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    let body_end = read_transformed_response_body(
        reader,
        &mut buffered,
        parsed.header_end,
        parsed.content_length,
        TRANSFORMED_RESPONSE_READ_LIMITS,
    )
    .await?;
    let body = &buffered[parsed.header_end..body_end];
    if let Some(result) = retryable_quality_result(body, retry_policy) {
        return Ok(result);
    }
    let translated_body = response_adapter::translate_chat_completion_to_responses(body)?;
    let usage = parse_token_usage_from_json_body(&translated_body);
    let cache_cost = parse_cache_cost_from_json_body(body);
    let mut header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
        translated_body.len()
    );
    append_capsule_nonce_headers(
        &mut header,
        parsed.client_nonce.as_deref(),
        parsed.nonce_origin.as_deref(),
    );
    append_mesh_served_by_header(&mut header, served_by);
    header.push_str("Connection: close\r\n\r\n");
    tcp_stream.write_all(header.as_bytes()).await?;
    tcp_stream.write_all(&translated_body).await?;
    route_observer.capture_response_body(&translated_body, Some("application/json"));
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: 200,
        usage,
        cache_cost,
    })
}

/// Relay a chat-completions upstream response through JSON body normalization.
pub(in crate::network::openai::response) async fn relay_normalized_chat_completion_json<
    R: AsyncRead + Unpin,
>(
    tcp_stream: &mut ClientStream,
    reader: &mut R,
    probe: ResponseProbe,
    retry_policy: ResponseRetryPolicy,
    served_by: Option<&str>,
    route_observer: OpenAiRouteObserver<'_>,
) -> Result<RouteAttemptResult> {
    if retry_policy.context_overflow && probe.retryable_context_overflow {
        return Ok(RouteAttemptResult::RetryableContextOverflow);
    }

    if !(200..300).contains(&probe.status_code) {
        return relay_error_response(tcp_stream, reader, probe, served_by, route_observer).await;
    }
    let mut buffered = probe.buffered;
    let parsed = try_parse_response_headers(&buffered)?
        .ok_or_else(|| anyhow!("incomplete HTTP response"))?;
    let body_end = read_transformed_response_body(
        reader,
        &mut buffered,
        parsed.header_end,
        parsed.content_length,
        TRANSFORMED_RESPONSE_READ_LIMITS,
    )
    .await?;
    let body = &buffered[parsed.header_end..body_end];
    let normalized_body =
        normalize_chat_completion_json_body(body).unwrap_or_else(|| body.to_vec());
    if let Some(result) = retryable_quality_result(&normalized_body, retry_policy) {
        return Ok(result);
    }
    let usage = parse_token_usage_from_json_body(&normalized_body);
    let cache_cost = parse_cache_cost_from_json_body(&normalized_body);
    let mut header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
        normalized_body.len()
    );
    append_capsule_nonce_headers(
        &mut header,
        parsed.client_nonce.as_deref(),
        parsed.nonce_origin.as_deref(),
    );
    append_mesh_served_by_header(&mut header, served_by);
    header.push_str("Connection: close\r\n\r\n");
    tcp_stream.write_all(header.as_bytes()).await?;
    tcp_stream.write_all(&normalized_body).await?;
    route_observer.capture_response_body(&normalized_body, Some("application/json"));
    let _ = tcp_stream.shutdown().await;
    Ok(RouteAttemptResult::Delivered {
        status_code: 200,
        usage,
        cache_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[test]
    fn test_translate_chat_completion_to_responses_json() {
        let translated = response_adapter::translate_chat_completion_to_responses(
            serde_json::json!({
                "id": "chatcmpl_123",
                "object": "chat.completion",
                "created": 1234,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello from mesh"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8
                }
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();
        let response: serde_json::Value = serde_json::from_slice(&translated).unwrap();

        assert_eq!(response["object"], "response");
        assert_eq!(response["model"], "test-model");
        assert_eq!(response["output_text"], "hello from mesh");
        assert_eq!(response["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(response["usage"]["input_tokens"], 5);
        assert_eq!(response["usage"]["output_tokens"], 3);
        assert_eq!(response["usage"]["total_tokens"], 8);
    }

    #[tokio::test]
    async fn relay_normalized_chat_completion_json_adds_missing_tool_call_id() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let body = br#"{"id":"chatcmpl-a","object":"chat.completion","created":1,"model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"lookup_fixture_fact","arguments":"{\"key\":\"codeword\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":2,"completion_tokens":4,"total_tokens":6}}"#;
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header = format!(
            "HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let header_end = header.len();
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: header.into_bytes(),
                header_end,
                status_code: 201,
                retryable_context_overflow: false,
            };
            relay_normalized_chat_completion_json(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
            )
            .await
            .expect("relay")
        });

        upstream_writer.write_all(body).await.unwrap();
        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), client.read_to_end(&mut output))
            .await
            .expect("relay should not wait for upstream keep-alive close")
            .unwrap();
        drop(upstream_writer);
        let route_result = server_task.await.expect("server task");
        let body_start = output
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&output[body_start..]).unwrap();

        assert!(output.starts_with(b"HTTP/1.1 200 OK\r\n"));

        assert_eq!(
            route_result,
            RouteAttemptResult::Delivered {
                status_code: 200,
                usage: Some(mesh_llm_events::logging::events::TokenUsage {
                    prompt_tokens: Some(2),
                    cached_prompt_tokens: None,
                    completion_tokens: Some(4),
                    total_tokens: Some(6),
                }),
                cache_cost: None,
            }
        );
        assert_eq!(
            parsed["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_mesh_chatcmpl_a_0_0"
        );
        assert_eq!(
            parsed["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "lookup_fixture_fact"
        );
    }

    #[tokio::test]
    async fn relay_normalized_chat_completion_json_echoes_capsule_nonce_headers() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let body = br#"{"id":"chatcmpl-a","object":"chat.completion","created":1,"model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nx-capsule-client-nonce: nonce-under-test\r\nx-capsule-nonce-origin: frontend\r\n\r\n",
            body.len()
        );
        let header_end = header.len();
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: header.into_bytes(),
                header_end,
                status_code: 200,
                retryable_context_overflow: false,
            };
            relay_normalized_chat_completion_json(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
            )
            .await
            .expect("relay")
        });

        upstream_writer.write_all(body).await.unwrap();
        let mut client = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), client.read_to_end(&mut output))
            .await
            .expect("relay should not wait for upstream keep-alive close")
            .unwrap();
        drop(upstream_writer);
        server_task.await.expect("server task");

        let output_text = String::from_utf8_lossy(&output);
        assert!(
            output_text.contains("x-capsule-client-nonce: nonce-under-test\r\n"),
            "public-proxy JSON response must echo the client nonce header: {output_text}"
        );
        assert!(
            output_text.contains("x-capsule-nonce-origin: frontend\r\n"),
            "public-proxy JSON response must echo the nonce origin marker: {output_text}"
        );
    }

    #[tokio::test]
    async fn translated_responses_json_reports_client_visible_status_and_usage() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let body = br#"{"id":"chatcmpl-a","object":"chat.completion","created":1,"model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":4,"total_tokens":6}}"#;
        let (mut upstream_writer, mut upstream_reader) = tokio::io::duplex(64 * 1024);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let header = format!(
            "HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let header_end = header.len();
        let server_task = tokio::spawn(async move {
            let (client_socket, _) = listener.accept().await.unwrap();
            let mut client_socket: ClientStream = client_socket.into();
            let probe = ResponseProbe {
                buffered: header.into_bytes(),
                header_end,
                status_code: 201,
                retryable_context_overflow: false,
            };
            relay_translated_responses_json(
                &mut client_socket,
                &mut upstream_reader,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
            )
            .await
            .expect("relay")
        });

        upstream_writer.write_all(body).await.unwrap();
        let mut client = ClientStream::connect(addr).await.unwrap();
        let mut output = Vec::new();
        tokio::time::timeout(Duration::from_secs(1), client.read_to_end(&mut output))
            .await
            .expect("relay should not wait for upstream keep-alive close")
            .unwrap();
        drop(upstream_writer);
        let route_result = server_task.await.expect("server task");

        assert!(output.starts_with(b"HTTP/1.1 200 OK\r\n"));
        assert_eq!(
            route_result,
            RouteAttemptResult::Delivered {
                status_code: 200,
                usage: Some(mesh_llm_events::logging::events::TokenUsage {
                    prompt_tokens: Some(2),
                    cached_prompt_tokens: None,
                    completion_tokens: Some(4),
                    total_tokens: Some(6),
                }),
                cache_cost: None,
            }
        );
    }
}
