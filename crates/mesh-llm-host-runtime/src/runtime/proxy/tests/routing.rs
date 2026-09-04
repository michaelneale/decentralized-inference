#[tokio::test]
async fn test_api_proxy_retries_context_overflow_bad_request_to_next_target() {
    let overflow_body =
        r#"{"error":{"message":"prompt tokens exceed context window (n_ctx=4096)"}}"#;
    let (small_port, small_rx, small_handle) =
        spawn_status_upstream("400 Bad Request", overflow_body).await;
    let (large_port, large_rx, large_handle) = spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle, affinity) = spawn_api_proxy_test_harness_with_affinity(
        single_model_targets("test", &[small_port, large_port]),
    )
    .await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "overflow then retry"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let first_raw = String::from_utf8(small_rx.await.unwrap()).unwrap();
    let second_raw = String::from_utf8(large_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(response.contains(r#"{"ok":true}"#));
    assert!(first_raw.contains("overflow then retry"));
    assert!(second_raw.contains("overflow then retry"));
    let stats = affinity.stats_snapshot();
    assert_eq!(stats.reservation_active, 0);
    assert_eq!(stats.reservation_created, 1);
    assert_eq!(stats.reservation_transferred, 1);
    assert_eq!(stats.reservation_released, 1);

    proxy_handle.abort();
    let _ = small_handle.await;
    let _ = large_handle.await;
}

#[tokio::test]
async fn real_proxy_avoids_the_round_robin_target_that_is_still_in_flight() {
    let (first_port, first_count, first_release, first_handle) =
        spawn_held_upstream(r#"{"ok":"first"}"#).await;
    let (second_port, second_count, second_release, second_handle) =
        spawn_held_upstream(r#"{"ok":"second"}"#).await;
    let (proxy_addr, proxy_handle, affinity) = spawn_api_proxy_test_harness_with_affinity(
        single_model_targets("test", &[first_port, second_port]),
    )
    .await;

    let request = |index| {
        let body = json!({
            "model": "test",
            "messages": [{"role": "user", "content": format!("burst {index}")}],
        })
        .to_string();
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        )
    };

    let first_request = tokio::spawn(send_request_and_read_response(
        proxy_addr,
        vec![request(0).into_bytes()],
    ));
    tokio::time::timeout(Duration::from_secs(2), async {
        while first_count.load(std::sync::atomic::Ordering::SeqCst) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("first request reached the first round-robin target");

    let second_request = tokio::spawn(send_request_and_read_response(
        proxy_addr,
        vec![request(1).into_bytes()],
    ));
    tokio::time::timeout(Duration::from_secs(2), async {
        while second_count.load(std::sync::atomic::Ordering::SeqCst) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("second request reached the second round-robin target");

    second_release.add_permits(1);
    let response = second_request.await.expect("second request task");
    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(affinity.stats_snapshot().reservation_active, 1);

    let third_request = tokio::spawn(send_request_and_read_response(
        proxy_addr,
        vec![request(2).into_bytes()],
    ));
    tokio::time::timeout(Duration::from_secs(2), async {
        while second_count.load(std::sync::atomic::Ordering::SeqCst) != 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("reservation pressure redirected the third request");

    assert_eq!(first_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(second_count.load(std::sync::atomic::Ordering::SeqCst), 2);
    let active = affinity.stats_snapshot();
    assert_eq!(active.reservation_active, 2);
    assert_eq!(active.reservation_created, 3);
    assert_eq!(active.reservation_spread_selections, 1);

    first_release.add_permits(1);
    second_release.add_permits(1);
    for request in [first_request, third_request] {
        let response = request.await.expect("request task");
        assert!(response.starts_with("HTTP/1.1 200 OK"));
    }
    let released = affinity.stats_snapshot();
    assert_eq!(released.reservation_active, 0);
    assert_eq!(released.reservation_released, 3);

    proxy_handle.abort();
    first_handle.abort();
    second_handle.abort();
}

#[tokio::test]
async fn test_api_proxy_preserves_context_overflow_bad_request_for_single_target() {
    let overflow_body =
        r#"{"error":{"message":"prompt tokens exceed context window (n_ctx=4096)"}}"#;
    let (port, upstream_rx, upstream_handle) =
        spawn_status_upstream("400 Bad Request", overflow_body).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", port)])).await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "single target overflow should stay 400"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    assert!(response.contains("context window"));
    assert!(raw.contains("single target overflow should stay 400"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_returns_last_context_overflow_bad_request_when_all_targets_overflow() {
    let first_body = r#"{"error":{"message":"prompt tokens exceed context window (n_ctx=2048)"}}"#;
    let second_body = r#"{"error":{"message":"prompt tokens exceed context window (n_ctx=4096)"}}"#;
    let (first_port, first_rx, first_handle) =
        spawn_status_upstream("400 Bad Request", first_body).await;
    let (second_port, second_rx, second_handle) =
        spawn_status_upstream("400 Bad Request", second_body).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(single_model_targets("test", &[first_port, second_port]))
            .await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "all targets overflow"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let first_raw = String::from_utf8(first_rx.await.unwrap()).unwrap();
    let second_raw = String::from_utf8(second_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    assert!(response.contains("n_ctx=4096"));
    assert!(first_raw.contains("all targets overflow"));
    assert!(second_raw.contains("all targets overflow"));

    proxy_handle.abort();
    let _ = first_handle.await;
    let _ = second_handle.await;
}

#[tokio::test]
async fn test_api_proxy_rejects_request_when_all_known_contexts_too_small() {
    let (first_port, first_rx, first_handle) = spawn_capturing_upstream(r#"{"ok":"first"}"#).await;
    let (second_port, second_rx, second_handle) =
        spawn_capturing_upstream(r#"{"ok":"second"}"#).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_contexts(
        single_model_targets("test", &[first_port, second_port]),
        &[("test", 4096)],
    )
    .await;

    let body = json!({
        "model": "test",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "x".repeat(20_000)}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let first_seen = tokio::time::timeout(Duration::from_millis(100), first_rx).await;
    let second_seen = tokio::time::timeout(Duration::from_millis(100), second_rx).await;

    proxy_handle.abort();
    first_handle.abort();
    second_handle.abort();

    assert!(response.starts_with("HTTP/1.1 503 Service Unavailable"));
    assert!(
        response.contains("context") || response.contains("target"),
        "response should explain why no target was eligible: {response}"
    );
    assert!(
        first_seen.is_err(),
        "proxy should not contact a known-too-small target"
    );
    assert!(
        second_seen.is_err(),
        "proxy should not contact any known-too-small fallback target"
    );
}

#[tokio::test]
async fn test_api_proxy_retries_empty_success_response_to_next_target() {
    let empty_body = json!({
        "id": "chatcmpl-empty",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let healthy_body = json!({
        "id": "chatcmpl-healthy",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "recovered answer"},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let (empty_port, empty_rx, empty_handle) = spawn_capturing_upstream(&empty_body).await;
    let (healthy_port, healthy_rx, healthy_handle) = spawn_capturing_upstream(&healthy_body).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(single_model_targets("test", &[empty_port, healthy_port]))
            .await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "empty then retry"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let empty_raw = String::from_utf8(empty_rx.await.unwrap()).unwrap();
    let healthy_raw = String::from_utf8(
        tokio::time::timeout(Duration::from_secs(2), healthy_rx)
            .await
            .expect("proxy did not retry empty success response to the healthy target")
            .unwrap(),
    )
    .unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(response.contains("recovered answer"));
    assert!(!response.contains("chatcmpl-empty"));
    assert!(empty_raw.contains("empty then retry"));
    assert!(healthy_raw.contains("empty then retry"));

    proxy_handle.abort();
    let _ = empty_handle.await;
    let _ = healthy_handle.await;
}

#[tokio::test]
async fn test_api_proxy_retries_length_finish_success_response_to_next_target() {
    let truncated_body = json!({
        "id": "chatcmpl-length",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "partial"},
            "finish_reason": "length"
        }]
    })
    .to_string();
    let healthy_body = json!({
        "id": "chatcmpl-healthy",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "complete answer"},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let (truncated_port, truncated_rx, truncated_handle) =
        spawn_capturing_upstream(&truncated_body).await;
    let (healthy_port, healthy_rx, healthy_handle) = spawn_capturing_upstream(&healthy_body).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness(single_model_targets(
        "test",
        &[truncated_port, healthy_port],
    ))
    .await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "length then retry"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let truncated_raw = String::from_utf8(truncated_rx.await.unwrap()).unwrap();
    let healthy_raw = String::from_utf8(
        tokio::time::timeout(Duration::from_secs(2), healthy_rx)
            .await
            .expect("proxy did not retry length-truncated success response to the healthy target")
            .unwrap(),
    )
    .unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(response.contains("complete answer"));
    assert!(!response.contains("chatcmpl-length"));
    assert!(truncated_raw.contains("length then retry"));
    assert!(healthy_raw.contains("length then retry"));

    proxy_handle.abort();
    let _ = truncated_handle.await;
    let _ = healthy_handle.await;
}

#[tokio::test]
async fn test_api_proxy_does_not_retry_generic_bad_request() {
    let bad_request_body = r#"{"error":{"message":"missing required field: messages"}}"#;
    let (bad_port, bad_rx, bad_handle) =
        spawn_status_upstream("400 Bad Request", bad_request_body).await;
    let (unused_port, unused_rx, unused_handle) = spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(single_model_targets("test", &[bad_port, unused_port])).await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "bad request should stop"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let first_raw = String::from_utf8(bad_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    assert!(response.contains("missing required field"));
    assert!(first_raw.contains("bad request should stop"));
    assert!(
        tokio::time::timeout(Duration::from_millis(250), unused_rx)
            .await
            .is_err()
    );

    proxy_handle.abort();
    let _ = bad_handle.await;
    unused_handle.abort();
}

#[tokio::test]
async fn test_api_proxy_normalizes_max_completion_tokens_for_upstream() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "max_completion_tokens": 32,
        "messages": [{"role": "user", "content": "normalize token alias"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("\"max_tokens\":32"));
    assert!(!raw.contains("max_completion_tokens"));
    assert!(raw.contains("normalize token alias"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_does_not_retry_after_successful_stream_starts() {
    let (stream_port, stream_rx, stream_handle) = spawn_streaming_upstream(
        "text/event-stream",
        vec![
            (Duration::ZERO, br#"data: {"delta":"first"}\n\n"#.to_vec()),
            (
                Duration::from_millis(50),
                br#"data: {"delta":"second"}\n\n"#.to_vec(),
            ),
        ],
    )
    .await;
    let (unused_port, unused_rx, unused_handle) = spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(single_model_targets("test", &[stream_port, unused_port]))
            .await;

    let body = json!({
        "model": "test",
        "stream": true,
        "messages": [{"role": "user", "content": "stream wins immediately"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();

    let first = read_until_contains(
        &mut stream,
        br#"data: {"delta":"first"}\n\n"#,
        Duration::from_secs(2),
    )
    .await;
    let first_text = String::from_utf8_lossy(&first);
    let raw = String::from_utf8(stream_rx.await.unwrap()).unwrap();

    assert!(first_text.contains("HTTP/1.1 200 OK"));
    assert!(first_text.contains(r#"data: {"delta":"first"}\n\n"#));
    assert!(raw.contains("stream wins immediately"));
    assert!(
        tokio::time::timeout(Duration::from_millis(250), unused_rx)
            .await
            .is_err()
    );

    drop(stream);
    proxy_handle.abort();
    tokio::time::timeout(Duration::from_secs(1), stream_handle)
        .await
        .expect("streaming upstream hung")
        .unwrap();
    unused_handle.abort();
}

#[tokio::test]
async fn test_api_proxy_passes_through_native_base64_image() {
    // A client that already has a base64-encoded image (data URI) and sends it
    // directly to /v1/chat/completions should have it forwarded unchanged.
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgAB"}}
            ]
        }],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains(r#""type":"image_url""#));
    assert!(raw.contains("data:image/jpeg;base64,/9j/4AAQSkZJRgAB"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_passes_through_native_base64_audio() {
    // A client that already has base64-encoded audio and sends it in the
    // input_audio format directly to /v1/chat/completions should have it
    // forwarded unchanged.
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe this"},
                {"type": "input_audio", "input_audio": {
                    "data": "UklGRg==",
                    "format": "wav"
                }}
            ]
        }],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains(r#""type":"input_audio""#));
    assert!(raw.contains(r#""data":"UklGRg==""#));
    assert!(raw.contains(r#""format":"wav""#));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}
