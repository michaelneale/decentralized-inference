use super::*;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::TcpStream;

#[test]
fn canonical_request_id_from_header_prefix_requires_one_valid_uuid() {
    let request_id = RequestId::new();
    let prefix = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nx-request-id: {}\r\n\r\nsecret-body",
        request_id.as_uuid()
    );
    assert_eq!(
        canonical_request_id_from_header_prefix(prefix.as_bytes()),
        Some(request_id)
    );
    assert_eq!(
        canonical_request_id_from_header_prefix(b"GET / HTTP/1.1\r\n\r\n"),
        None
    );
    assert_eq!(
        canonical_request_id_from_header_prefix(
            format!(
                "GET / HTTP/1.1\r\nx-request-id: {}\r\n",
                request_id.as_uuid()
            )
            .as_bytes(),
        ),
        None
    );
    assert_eq!(
        canonical_request_id_from_header_prefix(
            b"GET / HTTP/1.1\r\nx-request-id: not-a-uuid\r\n\r\n",
        ),
        None
    );
    assert_eq!(
        canonical_request_id_from_header_prefix(
            format!(
                "GET / HTTP/1.1\r\nx-request-id: {0}\r\nx-request-id: {0}\r\n\r\n",
                request_id.as_uuid()
            )
            .as_bytes(),
        ),
        None
    );
}

#[test]
fn raw_lifecycle_marker_requires_one_matching_canonical_request_id() {
    let request_id = RequestId::new();
    let valid = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nx-request-id: {}\r\n{}: {}\r\n\r\n",
        request_id.as_uuid(),
        RAW_LIFECYCLE_OWNER_HEADER,
        request_id.as_uuid()
    );
    assert_eq!(
        raw_lifecycle_owner_from_header_prefix(valid.as_bytes()),
        Some(request_id)
    );

    let mismatched = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nx-request-id: {}\r\n{}: {}\r\n\r\n",
        request_id.as_uuid(),
        RAW_LIFECYCLE_OWNER_HEADER,
        RequestId::new().as_uuid()
    );
    assert_eq!(
        raw_lifecycle_owner_from_header_prefix(mismatched.as_bytes()),
        None
    );

    let duplicate = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nx-request-id: {}\r\n{}: {}\r\n{}: {}\r\n\r\n",
        request_id.as_uuid(),
        RAW_LIFECYCLE_OWNER_HEADER,
        request_id.as_uuid(),
        RAW_LIFECYCLE_OWNER_HEADER,
        request_id.as_uuid()
    );
    assert_eq!(
        raw_lifecycle_owner_from_header_prefix(duplicate.as_bytes()),
        None
    );
}

#[tokio::test]
async fn parse_failures_expose_lifecycle_context_only_after_complete_headers() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let request_id = RequestId::new();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        read_http_request_with_plugin_manager_with_context(&mut stream, None)
            .await
            .unwrap_err()
    });
    let mut client = TcpStream::connect(address).await.unwrap();
    let request = format!(
        "POST /v1/tokenize HTTP/1.1\r\nHost: localhost\r\nx-request-id: {request_id}\r\nContent-Length: 1\r\n\r\n{{",
        request_id = request_id.as_uuid(),
    );
    client.write_all(request.as_bytes()).await.unwrap();
    let error = server.await.unwrap();
    let context = error.context().expect("complete headers establish context");
    assert_eq!(context.request_id, request_id);
    assert_eq!(context.client_path, "/v1/tokenize");

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        read_http_request_with_plugin_manager_with_context(&mut stream, None)
            .await
            .unwrap_err()
    });
    let mut client = TcpStream::connect(address).await.unwrap();
    client
        .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nBad")
        .await
        .unwrap();
    client.shutdown().await.unwrap();
    assert!(server.await.unwrap().context().is_none());
}

fn catalog_model_ref_descriptor(model_name: &str) -> mesh::ServedModelDescriptor {
    mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model_name.to_string(),
            source_kind: mesh::ModelSourceKind::Catalog,
            canonical_ref: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}
async fn read_request_from_parts_with_limits(
    parts: Vec<Vec<u8>>,
    limits: HttpReadLimits,
) -> BufferedHttpRequest {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        read_http_request_with_limits(&mut stream, limits, None)
            .await
            .unwrap()
    });

    let client = tokio::spawn(async move {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        for part in parts {
            stream.write_all(&part).await.unwrap();
        }
    });

    client.await.unwrap();
    server.await.unwrap()
}

async fn read_request_from_parts(parts: Vec<Vec<u8>>) -> BufferedHttpRequest {
    read_request_from_parts_with_limits(parts, HTTP_READ_LIMITS).await
}

fn tokenize_http_request(model_id: &str, text_bytes: usize) -> (Vec<u8>, Vec<u8>) {
    let body = serde_json::to_vec(&serde_json::json!({
        "expected_identity": {
            "model_id": model_id,
            "source_model_sha256": "a".repeat(64),
            "tokenizer_id": format!("gguf-source-sha256:{}", "a".repeat(64)),
        },
        "text": "x".repeat(text_bytes),
        "add_special": false,
        "include_token_pieces": false,
    }))
    .expect("tokenizer request should serialize");
    let mut raw = format!(
            "POST /v1/tokenize HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
    raw.extend_from_slice(&body);
    (raw, body)
}

fn forwarded_request_id_headers(raw: &[u8]) -> Vec<&str> {
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("forwarded request should contain headers");
    std::str::from_utf8(&raw[..header_end])
        .expect("forwarded headers should remain UTF-8")
        .split("\r\n")
        .filter(|line| {
            line.split_once(':')
                .is_some_and(|(name, _)| name.eq_ignore_ascii_case("x-request-id"))
        })
        .collect()
}

#[tokio::test]
async fn preserves_a_single_valid_request_id_in_forwarded_headers() {
    const REQUEST_ID: &str = "4c3ca94d-bc1f-4759-912d-f4f6d77d5515";
    let request = read_request_from_parts(vec![
        format!("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nX-Request-Id: {REQUEST_ID}\r\n\r\n")
            .into_bytes(),
    ])
    .await;

    assert_eq!(request.request_id.as_uuid().to_string(), REQUEST_ID);
    let headers = forwarded_request_id_headers(&request.raw);
    assert_eq!(headers.len(), 1);
    assert_eq!(headers[0], format!("x-request-id: {REQUEST_ID}"));
}

#[tokio::test]
async fn strips_client_supplied_raw_lifecycle_marker_before_forwarding() {
    let request = read_request_from_parts(vec![
        format!(
            "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nx-request-id: {}\r\n{}: {}\r\n\r\n",
            RequestId::new().as_uuid(),
            RAW_LIFECYCLE_OWNER_HEADER,
            RequestId::new().as_uuid()
        )
        .into_bytes(),
    ])
    .await;
    let forwarded = std::str::from_utf8(&request.raw).unwrap();
    assert!(
        !forwarded
            .to_ascii_lowercase()
            .contains(RAW_LIFECYCLE_OWNER_HEADER)
    );
}

#[tokio::test]
async fn raw_lifecycle_marker_is_added_only_after_explicit_owner_claim() {
    let mut request = read_request_from_parts(vec![
        b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec(),
    ])
    .await;
    assert!(raw_lifecycle_owner_from_header_prefix(&request.raw).is_none());

    request.mark_raw_lifecycle_owned();
    assert_eq!(
        raw_lifecycle_owner_from_header_prefix(&request.raw),
        Some(request.request_id)
    );
}

#[tokio::test]
async fn generates_a_request_id_when_the_header_is_missing() {
    let request = read_request_from_parts(vec![
        b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec(),
    ])
    .await;

    let headers = forwarded_request_id_headers(&request.raw);
    assert_eq!(headers.len(), 1);
    assert_eq!(
        headers[0],
        format!("x-request-id: {}", request.request_id.as_uuid())
    );
}

#[tokio::test]
async fn replaces_an_invalid_request_id_header_with_one_canonical_uuid() {
    let request = read_request_from_parts(vec![
        b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nX-Request-Id: not-a-uuid\r\n\r\n".to_vec(),
    ])
    .await;

    let headers = forwarded_request_id_headers(&request.raw);
    assert_eq!(headers.len(), 1);
    assert_eq!(
        headers[0],
        format!("x-request-id: {}", request.request_id.as_uuid())
    );
    assert!(
        !std::str::from_utf8(&request.raw)
            .expect("forwarded request should remain UTF-8")
            .contains("not-a-uuid")
    );
}

#[tokio::test]
async fn replaces_duplicate_request_id_headers_with_one_canonical_uuid() {
    const FIRST_REQUEST_ID: &str = "4c3ca94d-bc1f-4759-912d-f4f6d77d5515";
    const SECOND_REQUEST_ID: &str = "35e3c909-bd0b-420a-8d7a-e7aeb5e34a32";
    let request = read_request_from_parts(vec![format!(
            "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nX-Request-Id: {FIRST_REQUEST_ID}\r\nx-request-id: {SECOND_REQUEST_ID}\r\n\r\n"
        )
        .into_bytes()])
        .await;

    let headers = forwarded_request_id_headers(&request.raw);
    assert_eq!(headers.len(), 1);
    assert_eq!(
        headers[0],
        format!("x-request-id: {}", request.request_id.as_uuid())
    );
    assert_ne!(request.request_id.as_uuid().to_string(), FIRST_REQUEST_ID);
    assert_ne!(request.request_id.as_uuid().to_string(), SECOND_REQUEST_ID);
}

fn build_chunked_request(body: &[u8], chunks: &[usize]) -> Vec<u8> {
    let mut out = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
    let mut pos = 0usize;
    for &chunk_len in chunks {
        let end = pos + chunk_len;
        out.extend_from_slice(format!("{chunk_len:x}\r\n").as_bytes());
        out.extend_from_slice(&body[pos..end]);
        out.extend_from_slice(b"\r\n");
        pos = end;
    }
    out.extend_from_slice(b"0\r\n\r\n");
    out
}

fn build_chunked_request_one_byte_chunks(body: &[u8], extension_len: usize) -> Vec<u8> {
    let mut out = b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
    let extension = "x".repeat(extension_len);
    for byte in body {
        out.extend_from_slice(b"1");
        if !extension.is_empty() {
            out.extend_from_slice(b";");
            out.extend_from_slice(extension.as_bytes());
        }
        out.extend_from_slice(b"\r\n");
        out.push(*byte);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"0\r\n\r\n");
    out
}
#[test]
fn public_model_alias_rewrites_request_to_internal_model_name() {
    let models = vec!["Falcon-H1-1.5B-Instruct-Q4_K_M".to_string()];
    let descriptors = vec![catalog_model_ref_descriptor(&models[0])];
    let body = serde_json::json!({
        "model": "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let body_bytes = serde_json::to_vec(&body).unwrap();
    let mut raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
        body_bytes.len()
    )
    .into_bytes();
    raw.extend_from_slice(&body_bytes);
    let mut request = BufferedHttpRequest {
        raw,
        method: "POST".to_string(),
        path: "/v1/chat/completions".to_string(),
        client_path: "/v1/chat/completions".to_string(),
        request_id: RequestId::default(),
        body_json: Some(body),
        body_json_attempted: true,
        body_bytes: Some(body_bytes),
        body_len_bytes: 0,
        completion_tokens: None,
        model_name: Some("tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M".to_string()),
        stream: None,
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
    };

    rewrite_public_model_alias(&mut request, &models, &descriptors);

    assert_eq!(request.model_name.as_deref(), Some(models[0].as_str()));
    assert_eq!(request.body_json.as_ref().unwrap()["model"], models[0]);
}

#[test]
fn tokenize_route_is_strict() {
    assert!(is_tokenize_request("POST", "/v1/tokenize"));
    assert!(!is_tokenize_request("GET", "/v1/tokenize"));
    assert!(!is_tokenize_request("POST", "/v1/tokenize?mode=fast"));
    assert!(!is_tokenize_request("POST", "/v1/tokenize/"));
}

#[tokio::test]
async fn large_tokenize_request_routes_by_expected_identity_without_parsing_chat_body() {
    let model_id = "acme/code-model:Q4_K_M";
    let (raw, body) = tokenize_http_request(model_id, 140_000);
    let request = read_request_from_parts(vec![raw]).await;

    assert!(request.is_tokenize_request());
    assert_eq!(request.model_name.as_deref(), Some(model_id));
    assert!(request.body_json.is_none());
    assert!(!request.body_json_attempted);
    assert!(request.body_len_bytes / 4 > 32_768);
    assert_eq!(request.body_bytes.as_deref(), Some(body.as_slice()));
    let forwarded_body_start = request
        .raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("forwarded headers should terminate")
        + 4;
    assert_eq!(&request.raw[forwarded_body_start..], body.as_slice());
}

#[tokio::test]
async fn tokenizer_identity_is_not_alias_rewritten() {
    let internal = "CodeModel-Q4_K_M".to_owned();
    let descriptors = vec![catalog_model_ref_descriptor(&internal)];
    let public = "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M";
    let (raw, _) = tokenize_http_request(public, 140_000);
    let mut request = read_request_from_parts(vec![raw]).await;
    let forwarded_before_alias_resolution = request.raw.clone();

    rewrite_public_model_alias(&mut request, std::slice::from_ref(&internal), &descriptors);

    assert_eq!(request.raw, forwarded_before_alias_resolution);
    assert_eq!(request.model_name.as_deref(), Some(public));
    assert!(request.body_json.is_none());
    assert!(!request.body_json_attempted);
}
#[test]
fn test_pipeline_request_supported_chat_completions() {
    let body = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
    assert!(pipeline_request_supported(
        "/v1/chat/completions?stream=1",
        &body
    ));
}

#[test]
fn test_pipeline_request_supported_rejects_other_endpoint() {
    let body = serde_json::json!({"messages":[{"role":"user","content":"hi"}]});
    assert!(!pipeline_request_supported("/v1/responses", &body));
}
#[test]
fn test_pipeline_request_supported_rejects_missing_messages() {
    let body = serde_json::json!({"input":"hi"});
    assert!(!pipeline_request_supported("/v1/chat/completions", &body));
}

#[test]
fn legacy_lifecycle_paths_are_rejected_even_with_query_strings() {
    for path in [
        "/mesh/load",
        "/mesh/load?model=ignored",
        "/mesh/drop",
        "/mesh/drop?model=ignored",
    ] {
        assert!(
            is_legacy_lifecycle_path(path),
            "expected legacy path: {path}"
        );
    }
    assert!(!is_legacy_lifecycle_path("/v1/chat/completions"));
    assert!(!is_legacy_lifecycle_path("/mesh/load/"));
}

#[tokio::test]
async fn test_read_http_request_fragmented_post_body() {
    let body = br#"{"model":"qwen","user":"alice","messages":[{"role":"user","content":"hi"}]}"#;
    let headers = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );

    let request = read_request_from_parts(vec![
        headers.as_bytes()[..40].to_vec(),
        headers.as_bytes()[40..].to_vec(),
        body[..12].to_vec(),
        body[12..].to_vec(),
    ])
    .await;

    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(request.model_name.as_deref(), Some("qwen"));
    assert_eq!(
        request.response_adapter,
        ResponseAdapter::OpenAiChatCompletionsJson
    );

    assert!(request.body_json.is_none());
}

#[tokio::test]
async fn chat_reasoning_effort_none_is_canonicalized_before_forwarding() {
    let body = serde_json::json!({
        "model": "qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none"
    })
    .to_string();
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let request = read_request_from_parts(vec![raw.into_bytes()]).await;
    let forwarded = parse_json_body_from_http_request(&request.raw).unwrap();

    assert_eq!(
        forwarded["chat_template_kwargs"]["enable_thinking"],
        serde_json::json!(false)
    );
    assert_eq!(request.body_json, Some(forwarded));
}

#[tokio::test]
async fn chat_existing_template_kwargs_survive_forwarding_rewrite() {
    let body = serde_json::json!({
        "model": "qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 32,
        "reasoning_effort": "low",
        "chat_template_kwargs": {
            "enable_thinking": false,
            "custom": "kept"
        }
    })
    .to_string();
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let request = read_request_from_parts(vec![raw.into_bytes()]).await;
    let forwarded = parse_json_body_from_http_request(&request.raw).unwrap();

    assert_eq!(forwarded["max_tokens"], serde_json::json!(32));
    assert!(forwarded.get("max_completion_tokens").is_none());
    assert_eq!(
        forwarded["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": false, "custom": "kept"})
    );
}

#[tokio::test]
async fn chat_reasoning_enabled_false_wins_over_nested_effort_before_forwarding() {
    let body = serde_json::json!({
        "model": "qwen",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning": {"enabled": false, "effort": "low"}
    })
    .to_string();
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let request = read_request_from_parts(vec![raw.into_bytes()]).await;
    let forwarded = parse_json_body_from_http_request(&request.raw).unwrap();

    assert_eq!(
        forwarded["chat_template_kwargs"]["enable_thinking"],
        serde_json::json!(false)
    );
    assert_eq!(request.body_json, Some(forwarded));
}

#[tokio::test]
async fn test_read_http_request_preserves_client_path_for_responses_capture() {
    let body = br#"{"model":"qwen","stream":true,"input":"hello"}"#;
    let request = format!(
        "POST /v1/responses?foo=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        std::str::from_utf8(body).unwrap()
    );

    let request = read_request_from_parts(vec![request.into_bytes()]).await;

    assert_eq!(request.path, "/v1/chat/completions?foo=1");
    assert_eq!(request.client_path, "/v1/responses?foo=1");
}
#[tokio::test]
async fn test_read_http_request_large_body_over_32k() {
    let large = "x".repeat(40_000);
    let body = serde_json::json!({
        "model": "qwen",
        "messages": [{"role": "user", "content": large}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut request = read_request_from_parts(vec![request.into_bytes()]).await;

    assert_eq!(request.model_name.as_deref(), Some("qwen"));
    request.ensure_body_json();
    let body_json = request.body_json.unwrap();
    let content = body_json["messages"][0]["content"].as_str().unwrap();
    assert_eq!(content.len(), 40_000);
}

#[tokio::test]
async fn test_read_http_request_chunked_body() {
    let body = br#"{"model":"auto","session_id":"sess-42","messages":[{"role":"user","content":"hello"}]}"#;
    let request = build_chunked_request(body, &[18, 17, body.len() - 35]);

    let request = read_request_from_parts(vec![request]).await;

    assert_eq!(request.model_name.as_deref(), Some("auto"));

    assert!(request.body_json.is_none());
}

#[tokio::test]
async fn test_read_http_request_chunked_body_allows_wire_overhead() {
    let limits = HttpReadLimits {
        max_header_bytes: MAX_HEADER_BYTES,
        max_body_bytes: 256,
        max_chunked_wire_bytes: 4 * 1024,
    };
    let large = "x".repeat(48);
    let body = serde_json::json!({
        "model": "qwen",
        "messages": [{"role": "user", "content": large}],
    })
    .to_string();
    let request = build_chunked_request_one_byte_chunks(body.as_bytes(), 16);

    let mut request = read_request_from_parts_with_limits(vec![request], limits).await;

    assert_eq!(request.model_name.as_deref(), Some("qwen"));
    assert!(request.raw.len() > limits.max_body_bytes);
    request.ensure_body_json();
    let body_json = request.body_json.unwrap();
    let content = body_json["messages"][0]["content"].as_str().unwrap();
    assert_eq!(content.len(), 48);
}

#[tokio::test]
async fn test_read_http_request_allows_large_object_upload_body() {
    let body = vec![b'x'; MAX_BODY_BYTES + 1];
    let headers = format!(
            "POST /api/objects HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();

    let request = read_request_from_parts(vec![headers, body.clone()]).await;

    assert_eq!(request.path, "/api/objects");
    assert!(request.raw.ends_with(&body));
    assert!(request.body_json.is_none());
    assert!(request.request_object_request_ids.is_empty());
}

#[tokio::test]
async fn test_read_http_request_expect_100_continue() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let body = br#"{"model":"qwen","user":"bob","messages":[]}"#.to_vec();
    let headers = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nExpect: 100-continue\r\n\r\n",
        body.len()
    );

    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        read_http_request(&mut stream).await.unwrap()
    });

    let client = tokio::spawn(async move {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(headers.as_bytes()).await.unwrap();

        let mut interim = [0u8; 64];
        let n = stream.read(&mut interim).await.unwrap();
        assert_eq!(
            std::str::from_utf8(&interim[..n]).unwrap(),
            "HTTP/1.1 100 Continue\r\n\r\n"
        );

        stream.write_all(&body).await.unwrap();
    });

    client.await.unwrap();
    let request = server.await.unwrap();
    assert_eq!(request.model_name.as_deref(), Some("qwen"));

    let raw = String::from_utf8(request.raw).unwrap();
    assert!(!raw.contains("Expect: 100-continue"));
    assert!(raw.contains("Connection: close"));
}
#[tokio::test]
async fn test_read_http_request_truncates_pipelined_follow_up_bytes() {
    let request = read_request_from_parts(vec![
            b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\nGET /mesh/drop HTTP/1.1\r\nHost: localhost\r\n\r\n"
                .to_vec(),
        ])
        .await;

    let raw = String::from_utf8(request.raw).unwrap();
    assert!(raw.starts_with("GET /v1/models HTTP/1.1\r\n"));
    assert!(!raw.contains("/mesh/drop"));
    assert!(raw.contains("Connection: close\r\n\r\n"));
}

/// `probe_http_response_local` uses a much longer timeout (10 min)
/// than `probe_http_response` (5 min), because local prefill can
/// legitimately take minutes under load.
///
/// This test sends a response after a 2s delay and verifies that
/// `probe_http_response_local` waits for it (well within its 10-min
#[test]
fn test_inject_mesh_hooks_enabled() {
    let mut raw = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 25\r\n\r\n{\"model\":\"auto\",\"n\":1}".to_vec();
    inject_mesh_hooks_flag(&mut raw, true);
    let body_start = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let body = std::str::from_utf8(&raw[body_start..]).unwrap();
    assert!(body.starts_with("{\"mesh_hooks\":true,"), "body: {body}");
    // Content-Length must match actual body length
    let cl_line = std::str::from_utf8(&raw[..body_start])
        .unwrap()
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .unwrap();
    let declared: usize = cl_line.split(':').nth(1).unwrap().trim().parse().unwrap();
    assert_eq!(declared, raw.len() - body_start);
}

#[test]
fn test_inject_mesh_hooks_disabled() {
    let mut raw = b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 25\r\n\r\n{\"model\":\"auto\",\"n\":1}".to_vec();
    inject_mesh_hooks_flag(&mut raw, false);
    let body_start = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let body = std::str::from_utf8(&raw[body_start..]).unwrap();
    assert!(body.starts_with("{\"mesh_hooks\":false,"), "body: {body}");
}

#[test]
fn test_inject_mesh_hooks_no_body() {
    let mut raw = b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec();
    let before = raw.clone();
    inject_mesh_hooks_flag(&mut raw, true);
    assert_eq!(raw, before, "GET with no body should be unchanged");
}

#[test]
fn test_rewrite_model_field_updates_body_and_content_length() {
    let mut request = BufferedHttpRequest {
            raw: b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 45\r\n\r\n{\"model\":\"auto\",\"messages\":[],\"mesh_hooks\":true}".to_vec(),
            method: "POST".to_string(),
            path: "/v1/chat/completions".to_string(),
            client_path: "/v1/chat/completions".to_string(),
            request_id: RequestId::default(),
            body_json: None,
            body_json_attempted: false,
            body_bytes: None,
            body_len_bytes: 45,
            completion_tokens: None,
            model_name: Some("auto".to_string()),
            stream: None,
            request_object_request_ids: Vec::new(),
            response_adapter: ResponseAdapter::None,
            correlation_id: None,
        };

    rewrite_model_field(&mut request, "SmolLM2-135M-Instruct-Q8_0");

    let body_start = request
        .raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap()
        + 4;
    let body: serde_json::Value = serde_json::from_slice(&request.raw[body_start..]).unwrap();
    assert_eq!(body["model"], "SmolLM2-135M-Instruct-Q8_0");
    assert_eq!(body["mesh_hooks"], true);
    assert_eq!(
        request.model_name.as_deref(),
        Some("SmolLM2-135M-Instruct-Q8_0")
    );

    let cl_line = std::str::from_utf8(&request.raw[..body_start])
        .unwrap()
        .lines()
        .find(|line| line.to_ascii_lowercase().starts_with("content-length:"))
        .unwrap();
    let declared: usize = cl_line.split(':').nth(1).unwrap().trim().parse().unwrap();
    assert_eq!(declared, request.raw.len() - body_start);
    assert_eq!(declared, request.body_len_bytes);
}

#[test]
fn artifact_media_kind_is_closed_to_parsed_openai_json_routes() {
    let request = |path: &str, body: Option<&[u8]>| BufferedHttpRequest {
        raw: Vec::new(),
        method: "POST".into(),
        path: path.into(),
        client_path: path.into(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: true,
        body_bytes: body.map(ToOwned::to_owned),
        body_len_bytes: body.map_or(0, <[u8]>::len),
        completion_tokens: None,
        model_name: None,
        stream: None,
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
    };

    assert_eq!(
        request("/v1/chat/completions?trace=ignored", Some(br#"{}"#)).artifact_request_media_kind(),
        Some("application/json")
    );
    assert_eq!(
        request("/v1/responses", Some(b"not json")).artifact_request_media_kind(),
        None
    );
    assert_eq!(
        request("/mesh/load", Some(br#"{}"#)).artifact_request_media_kind(),
        None
    );
}

#[test]
fn public_model_id_with_named_profile() {
    let result = public_model_id("Qwen3-8B", None, "low-ctx");
    assert_eq!(result, "Qwen3-8B#low-ctx");
}

#[test]
fn public_model_id_without_profile() {
    let result = public_model_id("Qwen3-8B", None, "");
    assert_eq!(result, "Qwen3-8B");
}

#[test]
fn public_model_id_with_empty_profile() {
    let result = public_model_id("Qwen3-8B", None, "");
    assert_eq!(result, "Qwen3-8B");
}

#[test]
fn public_model_id_with_huggingface_ref_and_profile() {
    let result = public_model_id("org/repo:Q4_K_M", None, "high-ctx");
    assert_eq!(result, "org/repo:Q4_K_M#high-ctx");
}

// ── Capsule client-nonce ingress normalization (PR #1397) ──

fn client_nonce_header_values(raw: &[u8]) -> Vec<String> {
    let nonce_header = openai_frontend::lifecycle::CLIENT_NONCE_HEADER.as_str();
    std::str::from_utf8(raw)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case(nonce_header)
                .then(|| value.trim().to_string())
        })
        .collect()
}

fn nonce_origin_header_values(raw: &[u8]) -> Vec<String> {
    let origin_header = openai_frontend::lifecycle::CLIENT_NONCE_ORIGIN_HEADER.as_str();
    std::str::from_utf8(raw)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case(origin_header)
                .then(|| value.trim().to_string())
        })
        .collect()
}

#[tokio::test]
async fn ingress_mints_exactly_one_stable_nonce_when_absent() {
    // The nonce is resolved once at ingress, so `request.raw` — the bytes
    // every retry/hop path forwards — carries one stable UUIDv4 nonce and a
    // trusted `frontend` origin marker.
    let request = read_request_from_parts(vec![
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}"
            .to_vec(),
    ])
    .await;

    let nonces = client_nonce_header_values(&request.raw);
    assert_eq!(
        nonces.len(),
        1,
        "exactly one nonce header must be forwarded"
    );
    assert_eq!(
        uuid::Uuid::parse_str(&nonces[0]).unwrap().get_version_num(),
        4
    );
    assert_eq!(
        nonce_origin_header_values(&request.raw),
        vec![openai_frontend::lifecycle::CLIENT_NONCE_ORIGIN_FRONTEND.to_string()],
        "a minted nonce carries exactly one trusted origin marker"
    );
}

#[tokio::test]
async fn ingress_preserves_a_client_supplied_valid_uuidv4_without_marker() {
    let supplied_nonce = "11111111-1111-4111-8111-111111111111";
    let request = read_request_from_parts(vec![
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n{}: {supplied_nonce}\r\nContent-Length: 2\r\n\r\n{{}}",
            openai_frontend::lifecycle::CLIENT_NONCE_HEADER.as_str(),
        )
        .into_bytes(),
    ])
    .await;

    assert_eq!(
        client_nonce_header_values(&request.raw),
        vec![supplied_nonce.to_string()]
    );
    assert!(
        nonce_origin_header_values(&request.raw).is_empty(),
        "forwarding a client-supplied nonce must not stamp a frontend marker"
    );
}

#[tokio::test]
async fn ingress_strips_forged_origin_marker_on_forwarded_nonce() {
    // RED-before-fix: prior to normalizing at ingress, the raw proxy paths
    // forwarded a client-supplied `x-capsule-nonce-origin: frontend` verbatim,
    // letting a caller assert this frontend minted a nonce it chose itself.
    let supplied_nonce = "11111111-1111-4111-8111-111111111111";
    let request = read_request_from_parts(vec![
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n{}: {supplied_nonce}\r\n{}: frontend\r\nContent-Length: 2\r\n\r\n{{}}",
            openai_frontend::lifecycle::CLIENT_NONCE_HEADER.as_str(),
            openai_frontend::lifecycle::CLIENT_NONCE_ORIGIN_HEADER.as_str(),
        )
        .into_bytes(),
    ])
    .await;

    assert_eq!(
        client_nonce_header_values(&request.raw),
        vec![supplied_nonce.to_string()],
        "the client-supplied nonce value is still forwarded"
    );
    assert!(
        nonce_origin_header_values(&request.raw).is_empty(),
        "a caller-forged frontend origin marker must be stripped on every ingress path"
    );
}

#[tokio::test]
async fn ingress_rejects_non_uuid_nonce_and_mints_a_marked_one() {
    let request = read_request_from_parts(vec![
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n{}: attacker-chosen-value\r\nContent-Length: 2\r\n\r\n{{}}",
            openai_frontend::lifecycle::CLIENT_NONCE_HEADER.as_str(),
        )
        .into_bytes(),
    ])
    .await;

    let nonces = client_nonce_header_values(&request.raw);
    assert_eq!(nonces.len(), 1);
    assert_ne!(nonces[0], "attacker-chosen-value");
    assert_eq!(
        uuid::Uuid::parse_str(&nonces[0]).unwrap().get_version_num(),
        4
    );
    assert_eq!(
        nonce_origin_header_values(&request.raw),
        vec![openai_frontend::lifecycle::CLIENT_NONCE_ORIGIN_FRONTEND.to_string()]
    );
}

#[tokio::test]
async fn ingress_rejects_duplicate_nonce_headers_and_mints_a_single_one() {
    let supplied_nonce = "11111111-1111-4111-8111-111111111111";
    let nonce_header = openai_frontend::lifecycle::CLIENT_NONCE_HEADER.as_str();
    let request = read_request_from_parts(vec![
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n{nonce_header}: {supplied_nonce}\r\n{nonce_header}: {supplied_nonce}\r\nContent-Length: 2\r\n\r\n{{}}",
        )
        .into_bytes(),
    ])
    .await;

    let nonces = client_nonce_header_values(&request.raw);
    assert_eq!(nonces.len(), 1, "duplicates must collapse to one nonce");
    assert_ne!(
        nonces[0], supplied_nonce,
        "a duplicated nonce is untrusted and must be replaced with a minted one"
    );
    assert_eq!(
        nonce_origin_header_values(&request.raw),
        vec![openai_frontend::lifecycle::CLIENT_NONCE_ORIGIN_FRONTEND.to_string()]
    );
}

#[test]
fn capsule_nonce_headers_from_raw_reads_back_stabilized_headers() {
    // The request-rebuilding proxy paths (pipeline / MoA) reconstruct the
    // outbound request from a parsed body and would otherwise drop the nonce
    // stabilized at ingress. They read it back off `raw` via this helper, so it
    // must surface both the minted nonce and the trusted origin marker.
    let raw = concat!(
        "POST /v1/chat/completions HTTP/1.1\r\n",
        "host: 127.0.0.1\r\n",
        "x-capsule-client-nonce: 11111111-2222-4333-8444-555555555555\r\n",
        "x-capsule-nonce-origin: frontend\r\n",
        "\r\n",
        "{}",
    )
    .as_bytes();
    let (nonce, origin) = capsule_nonce_headers_from_raw(raw);
    assert_eq!(
        nonce.as_deref(),
        Some("11111111-2222-4333-8444-555555555555")
    );
    assert_eq!(origin.as_deref(), Some("frontend"));
}

#[test]
fn capsule_nonce_headers_from_raw_returns_none_without_headers() {
    let raw = concat!(
        "POST /v1/chat/completions HTTP/1.1\r\n",
        "host: 127.0.0.1\r\n",
        "\r\n",
        "{}",
    )
    .as_bytes();
    let (nonce, origin) = capsule_nonce_headers_from_raw(raw);
    assert_eq!(nonce, None);
    assert_eq!(origin, None);
}

fn request_with_raw(raw: &[u8]) -> BufferedHttpRequest {
    BufferedHttpRequest {
        raw: raw.to_vec(),
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: 0,
        completion_tokens: None,
        stream: None,
        model_name: None,
        request_object_request_ids: Vec::new(),
        response_adapter: ResponseAdapter::None,
        correlation_id: None,
    }
}

#[test]
fn mesh_routing_header_values_absent_is_empty() {
    let request = request_with_raw(
        concat!(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            "host: 127.0.0.1\r\n",
            "\r\n",
            "{}",
        )
        .as_bytes(),
    );
    let (target, exclude) = request.mesh_routing_header_values().unwrap();
    assert!(target.is_empty());
    assert!(exclude.is_empty());
}

#[test]
fn mesh_routing_header_values_reads_both_headers_verbatim() {
    let request = request_with_raw(
        concat!(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            "host: 127.0.0.1\r\n",
            "x-mesh-target: aabbcc\r\n",
            "x-mesh-exclude: 112233,445566\r\n",
            "\r\n",
            "{}",
        )
        .as_bytes(),
    );
    let (target, exclude) = request.mesh_routing_header_values().unwrap();
    assert_eq!(target, vec!["aabbcc".to_string()]);
    assert_eq!(exclude, vec!["112233,445566".to_string()]);
}

#[test]
fn mesh_routing_header_values_surfaces_every_duplicate_x_mesh_target() {
    // Ambiguity (more than one value) is the router's call, not this
    // parser's -- it must see every occurrence, not just the first.
    let request = request_with_raw(
        concat!(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            "host: 127.0.0.1\r\n",
            "x-mesh-target: aabbcc\r\n",
            "x-mesh-target: ddeeff\r\n",
            "\r\n",
            "{}",
        )
        .as_bytes(),
    );
    let (target, exclude) = request.mesh_routing_header_values().unwrap();
    assert_eq!(target, vec!["aabbcc".to_string(), "ddeeff".to_string()]);
    assert!(exclude.is_empty());
}

#[test]
fn mesh_routing_header_values_rejects_non_utf8_x_mesh_target() {
    let mut raw =
        b"POST /v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\nx-mesh-target: ".to_vec();
    raw.extend_from_slice(&[0xff, 0xfe]);
    raw.extend_from_slice(b"\r\n\r\n{}");
    let request = request_with_raw(&raw);
    assert!(request.mesh_routing_header_values().is_err());
}

#[test]
fn mesh_routing_header_values_rejects_non_utf8_x_mesh_exclude() {
    let mut raw =
        b"POST /v1/chat/completions HTTP/1.1\r\nhost: 127.0.0.1\r\nx-mesh-exclude: ".to_vec();
    raw.extend_from_slice(&[0xff, 0xfe]);
    raw.extend_from_slice(b"\r\n\r\n{}");
    let request = request_with_raw(&raw);
    assert!(request.mesh_routing_header_values().is_err());
}
