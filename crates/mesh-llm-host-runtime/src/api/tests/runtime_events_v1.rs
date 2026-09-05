// Black-box `GET /api/runtime/events/v1` coverage: real TCP connections
// against a real `MeshApi` + a real `RuntimeEventEngine`, asserting exact
// SSE bytes and connection-shape ordering per the plan's frozen contract.
// Named `runtime_event_api` (not `runtime_events_v1`) so it is selected by
// the plan's frozen Task 13 focused command, `cargo test -p
// mesh-llm-host-runtime runtime_event_api --lib`.
mod runtime_event_api {
    use super::*;

    fn runtime_events_terminal_success() -> mesh_llm_runtime_event_contracts::RuntimeFact {
        use mesh_llm_runtime_event_contracts::{FactData, FamilyFact, Outcome, RequestEventKind};
        mesh_llm_runtime_event_contracts::RuntimeFact::Request(FamilyFact::with_data(
            RequestEventKind::RequestCompleted,
            FactData {
                outcome: Some(Outcome::Success),
                ..FactData::default()
            },
        ))
    }

    fn runtime_events_synthetic_unknown() -> mesh_llm_runtime_event_contracts::RuntimeFact {
        use mesh_llm_runtime_event_contracts::{
            FactData, FamilyFact, Outcome, ReasonCode, RequestEventKind,
        };
        mesh_llm_runtime_event_contracts::RuntimeFact::Request(FamilyFact::with_data(
            RequestEventKind::RequestFailed,
            FactData {
                outcome: Some(Outcome::Unknown),
                reason: Some(ReasonCode::TerminalNotDelivered),
                ..FactData::default()
            },
        ))
    }

    async fn connect_runtime_events(
        addr: std::net::SocketAddr,
        extra_headers: &str,
        query: &str,
    ) -> TcpStream {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let request = format!(
            "GET /api/runtime/events/v1{query} HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n{extra_headers}\r\n"
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        stream
    }

    #[tokio::test]
    #[serial(runtime_event_engine_state)]
    async fn runtime_events_capability_is_advertised_on_both_status_routes() {
        crate::runtime_events::clear_runtime_event_engine();
        let expected = json!({"version": 1, "endpoint": "/api/runtime/events/v1", "cursor": "rt1"});

        let state = build_test_mesh_api().await;
        let status_body = request_management_json(state, "/api/status").await;
        assert_eq!(
            status_body["runtime"]["capabilities"]["runtime_events"],
            expected
        );

        let state = build_test_mesh_api().await;
        let runtime_body = request_management_json(state, "/api/runtime").await;
        assert_eq!(runtime_body["capabilities"]["runtime_events"], expected);
    }

    #[tokio::test]
    #[serial(runtime_event_engine_state)]
    async fn absent_engine_yields_service_unavailable() {
        crate::runtime_events::clear_runtime_event_engine();
        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let response = send_management_request(
        addr,
        "GET /api/runtime/events/v1 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n"
            .to_string(),
    )
    .await;

        assert!(
            response.starts_with("HTTP/1.1 503"),
            "unexpected response: {response}"
        );
        handle.abort();
    }

    #[tokio::test]
    #[serial(runtime_event_engine_state)]
    async fn no_cursor_connection_emits_state_then_health_with_no_store_headers() {
        crate::runtime_events::clear_runtime_event_engine();
        let engine = crate::runtime_events::engine::RuntimeEventEngine::new();
        crate::runtime_events::install_runtime_event_engine(engine.clone());

        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let mut stream = connect_runtime_events(addr, "", "").await;
        let bytes = read_until_contains(
            &mut stream,
            b"event: runtime_health",
            Duration::from_secs(2),
        )
        .await;
        let text = String::from_utf8(bytes).unwrap();

        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
        assert!(text.contains("Content-Type: text/event-stream"), "{text}");
        assert!(text.contains("Cache-Control: no-store"), "{text}");
        assert!(text.contains("X-Accel-Buffering: no"), "{text}");

        let state_at = text
            .find("event: runtime_state")
            .expect("runtime_state frame");
        let health_at = text
            .find("event: runtime_health")
            .expect("runtime_health frame");
        assert!(
            state_at < health_at,
            "no-cursor order must be state before health: {text}"
        );
        assert!(text.contains("\nid: rt1:"), "{text}");

        handle.abort();
        crate::runtime_events::clear_runtime_event_engine();
    }

    #[tokio::test]
    #[serial(runtime_event_engine_state)]
    async fn malformed_header_cursor_is_rejected_before_headers() {
        crate::runtime_events::clear_runtime_event_engine();
        let engine = crate::runtime_events::engine::RuntimeEventEngine::new();
        crate::runtime_events::install_runtime_event_engine(engine);

        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let response = send_management_request(
        addr,
        "GET /api/runtime/events/v1 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\nLast-Event-ID: not-a-cursor\r\n\r\n".to_string(),
    )
    .await;

        assert!(
            response.starts_with("HTTP/1.1 400"),
            "unexpected response: {response}"
        );
        handle.abort();
        crate::runtime_events::clear_runtime_event_engine();
    }

    #[tokio::test]
    #[serial(runtime_event_engine_state)]
    async fn malformed_query_cursor_is_rejected_before_headers() {
        crate::runtime_events::clear_runtime_event_engine();
        let engine = crate::runtime_events::engine::RuntimeEventEngine::new();
        crate::runtime_events::install_runtime_event_engine(engine);

        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let response = send_management_request(
        addr,
        "GET /api/runtime/events/v1?cursor=garbage HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n".to_string(),
    )
    .await;

        assert!(
            response.starts_with("HTTP/1.1 400"),
            "unexpected response: {response}"
        );
        handle.abort();
        crate::runtime_events::clear_runtime_event_engine();
    }

    #[tokio::test]
    #[serial(runtime_event_engine_state)]
    async fn future_sequence_for_the_current_instance_is_rejected() {
        crate::runtime_events::clear_runtime_event_engine();
        let engine = crate::runtime_events::engine::RuntimeEventEngine::new();
        crate::runtime_events::install_runtime_event_engine(engine.clone());

        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let cursor = format!("rt1:{}:5", engine.process_instance().as_uuid());
        let response = send_management_request(
        addr,
        format!(
            "GET /api/runtime/events/v1 HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\nLast-Event-ID: {cursor}\r\n\r\n"
        ),
    )
    .await;

        assert!(
            response.starts_with("HTTP/1.1 400"),
            "unexpected response: {response}"
        );
        handle.abort();
        crate::runtime_events::clear_runtime_event_engine();
    }

    #[tokio::test]
    #[serial(runtime_event_engine_state)]
    async fn header_cursor_wins_over_query_cursor_and_replays_only_the_in_window_tail() {
        crate::runtime_events::clear_runtime_event_engine();
        let engine = crate::runtime_events::engine::RuntimeEventEngine::new();
        crate::runtime_events::install_runtime_event_engine(engine.clone());

        // Submit three terminal facts (sequences 1, 2, and 3).
        for _ in 0..3 {
            let reservation = engine
                .reserve_root(
                    mesh_llm_runtime_event_contracts::OperationId::new(),
                    runtime_events_synthetic_unknown,
                )
                .expect("reserve");
            use mesh_llm_runtime_event_contracts::RuntimeEventIngress;
            assert_eq!(
                reservation
                    .ingress()
                    .try_submit(runtime_events_terminal_success()),
                mesh_llm_runtime_event_contracts::SubmitOutcome::Accepted
            );
        }
        let report = engine.drain();
        assert_eq!(report.applied, 3);

        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        // The header says "I've seen sequence 1" while the query says 0.
        // Only the header cursor should win, so replay starts at sequence 2.
        let header_cursor = format!("rt1:{}:1", engine.process_instance().as_uuid());
        let query = format!("?cursor=rt1%3A{}%3A0", engine.process_instance().as_uuid());
        let extra = format!("Last-Event-ID: {header_cursor}\r\n");
        let mut stream = connect_runtime_events(addr, &extra, &query).await;
        let bytes = read_until_contains(
            &mut stream,
            b"event: runtime_health",
            Duration::from_secs(2),
        )
        .await;
        let text = String::from_utf8(bytes).unwrap();

        // Only sequences 2 and 3 should replay (the header cursor won over 0).
        assert_eq!(text.matches("event: runtime_event").count(), 2, "{text}");
        assert!(text.contains(&format!(
            "id: rt1:{}:2",
            engine.process_instance().as_uuid()
        )));
        assert!(text.contains(&format!(
            "id: rt1:{}:3",
            engine.process_instance().as_uuid()
        )));
        let event_at = text.find("event: runtime_event").unwrap();
        let health_at = text.find("event: runtime_health").unwrap();
        assert!(
            event_at < health_at,
            "in-window order must be events then health: {text}"
        );

        handle.abort();
        crate::runtime_events::clear_runtime_event_engine();
    }

    #[tokio::test]
    #[serial(runtime_event_engine_state)]
    async fn stale_process_instance_yields_gap_state_health_order() {
        crate::runtime_events::clear_runtime_event_engine();
        let engine = crate::runtime_events::engine::RuntimeEventEngine::new();
        crate::runtime_events::install_runtime_event_engine(engine.clone());

        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let foreign_instance = uuid::Uuid::new_v4();
        let cursor = format!("rt1:{foreign_instance}:0");
        let extra = format!("Last-Event-ID: {cursor}\r\n");
        let mut stream = connect_runtime_events(addr, &extra, "").await;
        let bytes = read_until_contains(
            &mut stream,
            b"event: runtime_health",
            Duration::from_secs(2),
        )
        .await;
        let text = String::from_utf8(bytes).unwrap();

        let gap_at = text.find("event: runtime_replay_gap").expect("gap frame");
        let state_at = text.find("event: runtime_state").expect("state frame");
        let health_at = text.find("event: runtime_health").expect("health frame");
        assert!(
            gap_at < state_at && state_at < health_at,
            "stale-instance order must be gap, state, health: {text}"
        );
        assert!(text.contains("\"reason\":\"stale_instance\""), "{text}");
        assert!(
            text.contains(&format!("\"requested_cursor\":\"{cursor}\"")),
            "{text}"
        );

        handle.abort();
        crate::runtime_events::clear_runtime_event_engine();
    }

    #[tokio::test]
    #[serial(runtime_event_engine_state)]
    async fn evicted_cursor_after_rebuild_yields_gap_state_health_order() {
        crate::runtime_events::clear_runtime_event_engine();
        let engine = crate::runtime_events::engine::RuntimeEventEngine::new();
        crate::runtime_events::install_runtime_event_engine(engine.clone());

        for _ in 0..2 {
            let reservation = engine
                .reserve_root(
                    mesh_llm_runtime_event_contracts::OperationId::new(),
                    runtime_events_synthetic_unknown,
                )
                .expect("reserve");
            use mesh_llm_runtime_event_contracts::RuntimeEventIngress;
            reservation
                .ingress()
                .try_submit(runtime_events_terminal_success());
        }
        engine.drain();
        engine.rebuild();

        let state = build_test_mesh_api().await;
        let (addr, handle) = spawn_management_test_server(state).await;

        let cursor = format!("rt1:{}:0", engine.process_instance().as_uuid());
        let extra = format!("Last-Event-ID: {cursor}\r\n");
        let mut stream = connect_runtime_events(addr, &extra, "").await;
        let bytes = read_until_contains(
            &mut stream,
            b"event: runtime_health",
            Duration::from_secs(2),
        )
        .await;
        let text = String::from_utf8(bytes).unwrap();

        let gap_at = text.find("event: runtime_replay_gap").expect("gap frame");
        let state_at = text.find("event: runtime_state").expect("state frame");
        let health_at = text.find("event: runtime_health").expect("health frame");
        assert!(
            gap_at < state_at && state_at < health_at,
            "evicted order must be gap, state, health: {text}"
        );
        assert!(text.contains("\"reason\":\"evicted\""), "{text}");

        handle.abort();
        crate::runtime_events::clear_runtime_event_engine();
    }
} // mod runtime_event_api
