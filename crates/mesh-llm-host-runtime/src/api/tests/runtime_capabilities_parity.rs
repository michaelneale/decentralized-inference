// Black-box coverage for review defect D4: `GET /api/runtime` must advertise
// the same derived role-capability booleans as `GET /api/status` for one
// node state. Pre-fix, `runtime_status()` hardcoded
// `RuntimeCapabilityFlags::default()` (every role boolean false) with only
// `runtime_events` populated, while `status()` derived real values
// (`worker_capable`, `local_serving`, `proxying`, `plugin_ingress`,
// `accepting_local`, `accepting_remote`) from live node state through
// `derive_capability_flags`. See `.omo/plans/event-system-fixes.md` Task 2.
#[tokio::test]
async fn runtime_capabilities_parity() {
    // `build_test_mesh_api()` builds a Worker-role node with activity policy
    // disabled (the default `RuntimeActivityConfig`), which is an
    // "accepting" state: `check_admission` returns `Allowed` for every
    // ingress type regardless of ingress kind.
    let state = build_test_mesh_api().await;

    let status_body = request_management_json(state.clone(), "/api/status").await;
    let runtime_body = request_management_json(state, "/api/runtime").await;

    let status_capabilities = &status_body["runtime"]["capabilities"];
    let runtime_capabilities = &runtime_body["capabilities"];

    assert_eq!(
        runtime_capabilities, status_capabilities,
        "GET /api/runtime capabilities must equal GET /api/status runtime.capabilities \
         for the same node state (review defect D4); status={status_body}, runtime={runtime_body}"
    );

    // Pin the specific accepting-state booleans so a regression back to
    // `RuntimeCapabilityFlags::default()` (all-false) on either route is
    // caught even if both routes drifted to agree on a wrong value.
    assert_eq!(runtime_capabilities["worker_capable"], json!(true));
    assert_eq!(runtime_capabilities["accepting_local"], json!(true));
    assert_eq!(runtime_capabilities["accepting_remote"], json!(true));
    assert_eq!(
        runtime_capabilities["runtime_events"]["endpoint"],
        json!("/api/runtime/events/v1")
    );
}
