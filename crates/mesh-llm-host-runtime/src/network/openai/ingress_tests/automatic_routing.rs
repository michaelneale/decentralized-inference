//! Mode selection for the automatic routing directive on the host path.
//!
//! These cover the decision `resolve_auto_routed_model` makes *before* any
//! model is contacted: whether a request that asked for automatic routing
//! convenes a committee or is served by one capability-selected model.

use crate::inference::election;
use crate::mesh;
use crate::network::affinity;
use crate::network::openai::automatic;
use crate::network::openai::transport as proxy;
use mesh_llm_events::logging::identifiers::RequestId;

use super::super::ingress::{
    AutoRouteResolution, prepare_cache_routing_body, resolve_auto_routed_model,
};

/// A served model with the given capabilities, ready for the media filter.
fn descriptor(model: &str, vision: bool, audio: bool) -> mesh::ServedModelDescriptor {
    use crate::models::CapabilityLevel;
    mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model.to_string(),
            ..Default::default()
        },
        // Runtime-verified, so `supports_*_runtime()` accepts these.
        capabilities_known: true,
        capabilities: crate::models::ModelCapabilities {
            multimodal: vision || audio,
            vision: if vision {
                CapabilityLevel::Supported
            } else {
                CapabilityLevel::None
            },
            audio: if audio {
                CapabilityLevel::Supported
            } else {
                CapabilityLevel::None
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn request_with_body(model: Option<&str>, body: &serde_json::Value) -> proxy::BufferedHttpRequest {
    let body = serde_json::to_vec(body).expect("serialize body");
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes()
    .into_iter()
    .chain(body.iter().copied())
    .collect::<Vec<u8>>();
    proxy::BufferedHttpRequest {
        raw,
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: body.len(),
        completion_tokens: None,
        stream: None,
        model_name: model.map(str::to_owned),
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::OpenAiChatCompletionsJson,
        correlation_id: None,
    }
}

fn text_body(model: Option<&str>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "messages": [{ "role": "user", "content": "hello" }],
    });
    if let Some(model) = model {
        body["model"] = serde_json::json!(model);
    }
    body
}

#[test]
fn generation_body_is_parsed_for_cache_evidence_with_one_target() {
    let model = "local-model";
    let mut request = request_with_body(Some(model), &text_body(Some(model)));
    assert!(request.body_json.is_none());

    prepare_cache_routing_body(&mut request, Some(model));

    assert!(request.body_json.is_some());
    assert!(crate::network::affinity::cache_prefix_hash(request.body_json.as_ref()).is_some());
}

fn image_body(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": "what is in this image?" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
            ],
        }],
    })
}

/// Node serving `models`, each locally callable.
async fn node_serving(models: &[&str]) -> (mesh::Node, election::ModelTargets) {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_hosted_models(models.iter().map(|m| (*m).to_string()).collect())
        .await;
    let mut targets = election::ModelTargets::default();
    for (index, model) in models.iter().enumerate() {
        targets.targets.insert(
            (*model).to_string(),
            vec![election::InferenceTarget::Local(9000 + index as u16)],
        );
    }
    (node, targets)
}

async fn resolve(
    model: Option<&str>,
    body: &serde_json::Value,
    node: &mesh::Node,
    targets: &election::ModelTargets,
    descriptors: &[mesh::ServedModelDescriptor],
) -> AutoRouteResolution {
    let mut request = request_with_body(model, body);
    let affinity = affinity::AffinityRouter::new();
    resolve_auto_routed_model(
        node,
        &mut request,
        targets,
        None,
        descriptors,
        None,
        &affinity,
    )
    .await
}

#[tokio::test]
async fn plain_text_directive_stays_on_the_committee() {
    let (node, targets) = node_serving(&["vision-model", "text-model"]).await;
    let descriptors = vec![
        descriptor("vision-model", true, false),
        descriptor("text-model", false, false),
    ];

    let resolution = resolve(
        Some(automatic::DIRECTIVE),
        &text_body(Some(automatic::DIRECTIVE)),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    // The directive must survive resolution so the MoA gateway picks it up.
    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => assert_eq!(effective_model.as_deref(), Some(automatic::DIRECTIVE)),
        AutoRouteResolution::MediaUnsupported => panic!("text request is not a media failure"),
    }
}

#[tokio::test]
async fn image_request_resolves_to_a_vision_capable_model() {
    // The defect this pins: `model=mesh` with an image used to skip the media
    // filter entirely and reach MoA, whose text extraction drops the image and
    // answers the text half as if no image were sent.
    let (node, targets) = node_serving(&["text-model", "vision-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("vision-model", true, false),
    ];

    let resolution = resolve(
        Some(automatic::DIRECTIVE),
        &image_body(automatic::DIRECTIVE),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => assert_eq!(
            effective_model.as_deref(),
            Some("vision-model"),
            "an image request must resolve to the vision-capable model, not the directive"
        ),
        AutoRouteResolution::MediaUnsupported => {
            panic!("a vision-capable model is served, so this must not fail")
        }
    }
}

#[tokio::test]
async fn image_request_with_no_capable_model_is_reported_unsupported() {
    // Honest failure beats a confident answer to the text half.
    let (node, targets) = node_serving(&["text-model", "other-text-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("other-text-model", false, false),
    ];

    let resolution = resolve(
        Some(automatic::DIRECTIVE),
        &image_body(automatic::DIRECTIVE),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    assert!(
        matches!(resolution, AutoRouteResolution::MediaUnsupported),
        "no served model can satisfy the image, so the request must be refused"
    );
}

#[tokio::test]
async fn deprecated_alias_behaves_exactly_like_the_directive() {
    let (node, targets) = node_serving(&["text-model", "vision-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("vision-model", true, false),
    ];

    let via_alias = resolve(
        Some(automatic::DEPRECATED_ALIAS),
        &text_body(Some(automatic::DEPRECATED_ALIAS)),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    // `auto` is the same directive, so it must also reach the committee rather
    // than resolving to a single model as it did historically.
    match via_alias {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => assert_eq!(effective_model.as_deref(), Some(automatic::DIRECTIVE)),
        AutoRouteResolution::MediaUnsupported => panic!("text request is not a media failure"),
    }
}

#[tokio::test]
async fn streaming_directive_resolves_to_a_single_model() {
    // A committee cannot stream: workers are called non-streaming and the SSE
    // is synthesised afterwards. A client asking to stream gets one model.
    let (node, targets) = node_serving(&["text-model", "other-text-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("other-text-model", false, false),
    ];
    let mut body = text_body(Some(automatic::DIRECTIVE));
    body["stream"] = serde_json::json!(true);

    let resolution = resolve(
        Some(automatic::DIRECTIVE),
        &body,
        &node,
        &targets,
        &descriptors,
    )
    .await;

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => {
            let model = effective_model.expect("a streaming request must resolve to a model");
            assert_ne!(
                model,
                automatic::DIRECTIVE,
                "a streaming request must not stay on the committee"
            );
            assert!(
                model == "text-model" || model == "other-text-model",
                "must resolve to a served model, got {model}"
            );
        }
        AutoRouteResolution::MediaUnsupported => panic!("no media in this request"),
    }
}

#[tokio::test]
async fn model_less_request_resolves_to_a_single_model() {
    // A client that named nothing never opted into committee cost.
    let (node, targets) = node_serving(&["text-model", "other-text-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("other-text-model", false, false),
    ];

    let resolution = resolve(None, &text_body(None), &node, &targets, &descriptors).await;

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => {
            let model = effective_model.expect("must resolve to a concrete model");
            assert_ne!(
                model,
                automatic::DIRECTIVE,
                "a model-less request must not silently convene a committee"
            );
        }
        AutoRouteResolution::MediaUnsupported => panic!("no media in this request"),
    }
}

/// A peer advertising one model, with the capabilities in `descriptor`.
fn peer_serving(peer_id: iroh::EndpointId, model: &str, vision: bool) -> mesh::PeerInfo {
    mesh::PeerInfo {
        id: peer_id,
        addr: iroh::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        },
        mesh_id: None,
        mesh_policy_hash: None,
        genesis_policy: None,
        role: mesh::NodeRole::Host { http_port: 9337 },
        first_joined_mesh_ts: None,
        models: vec![model.to_string()],
        vram_bytes: 16 * 1024 * 1024 * 1024,
        rtt_ms: None,
        model_source: None,
        admitted: true,
        serving_models: vec![model.to_string()],
        hosted_models: vec![model.to_string()],
        hosted_models_known: true,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        last_seen: std::time::Instant::now(),
        last_mentioned: std::time::Instant::now(),
        version: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        memory: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: std::collections::HashMap::new(),
        served_model_descriptors: vec![descriptor(model, vision, false)],
        served_model_runtime: vec![],
        owner_attestation: None,
        release_attestation_summary: crate::ReleaseAttestationSummary::default(),
        artifact_transfer_supported: false,
        stage_protocol_generation_supported: false,
        stage_status_list_supported: false,
        local_gguf_content_id_supported: false,
        advertised_model_throughput: vec![],
        cache_affinity: None,
        display_rtt: None,
        selected_path: None,
        propagated_latency: None,
        owner_summary: crate::crypto::OwnershipSummary::default(),
        inference_admission_state: None,
    }
}

/// Two nodes: this one serves `local_model`, a peer serves `remote_model`.
///
/// The peer's model is reachable only as `InferenceTarget::Remote`, so a test
/// that expects it to be selected is exercising cross-node selection.
async fn two_node_mesh(
    local_model: &str,
    remote_model: &str,
    remote_is_vision: bool,
) -> (mesh::Node, election::ModelTargets) {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_hosted_models(vec![local_model.to_string()]).await;

    let peer_id = iroh::EndpointId::from(iroh::SecretKey::generate().public());
    node.insert_test_peer(peer_serving(peer_id, remote_model, remote_is_vision))
        .await;

    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        local_model.to_string(),
        vec![election::InferenceTarget::Local(9000)],
    );
    targets.targets.insert(
        remote_model.to_string(),
        vec![election::InferenceTarget::Remote(peer_id)],
    );
    (node, targets)
}

#[tokio::test]
async fn image_request_selects_a_vision_model_served_only_by_a_peer() {
    // The vision model is not served locally: it reaches the candidate set only
    // because a peer advertises it (via gossip and a `Remote` target). A
    // single-node fixture cannot catch a regression that drops peer-served
    // models from the media filter.
    //
    // Scope: this pins candidate *selection* across nodes. It does not prove the
    // QUIC dispatch to that peer — no request is sent here, and either source
    // (gossip or the target table) is sufficient on its own. Proving the
    // delivery path needs a live two-node run.
    let (node, targets) = two_node_mesh("local-text-model", "remote-vision-model", true).await;
    let descriptors = vec![
        descriptor("local-text-model", false, false),
        descriptor("remote-vision-model", true, false),
    ];

    let resolution = resolve(
        Some(automatic::DIRECTIVE),
        &image_body(automatic::DIRECTIVE),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => assert_eq!(
            effective_model.as_deref(),
            Some("remote-vision-model"),
            "the only vision-capable model is on the peer and must still be chosen"
        ),
        AutoRouteResolution::MediaUnsupported => {
            panic!("a peer serves a vision model, so this must not be refused")
        }
    }
}

#[tokio::test]
async fn text_request_still_convenes_a_committee_across_two_nodes() {
    // Two models, one local and one remote: enough for a committee, and the
    // directive must survive resolution so the MoA gateway forms one.
    let (node, targets) = two_node_mesh("local-text-model", "remote-text-model", false).await;
    let descriptors = vec![
        descriptor("local-text-model", false, false),
        descriptor("remote-text-model", false, false),
    ];

    let resolution = resolve(
        Some(automatic::DIRECTIVE),
        &text_body(Some(automatic::DIRECTIVE)),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => assert_eq!(effective_model.as_deref(), Some(automatic::DIRECTIVE)),
        AutoRouteResolution::MediaUnsupported => panic!("text request is not a media failure"),
    }
}

#[tokio::test]
async fn models_listing_advertises_the_directive_with_the_mesh_capability_union() {
    // The directive must report what the *mesh* can accept. With a text-only
    // local model and a vision model on a peer, `mesh` has to advertise vision
    // — a media request will be routed to that peer.
    use crate::network::openai::moa_gateway::context_selection::virtual_mesh_capabilities;

    let models = vec![
        "local-text-model".to_string(),
        "remote-vision-model".to_string(),
    ];
    let descriptors = vec![
        descriptor("local-text-model", false, false),
        descriptor("remote-vision-model", true, false),
    ];

    let union = virtual_mesh_capabilities(&models, &descriptors);
    assert!(
        union.supports_vision_runtime(),
        "one peer serves a vision model, so the directive must advertise vision"
    );
    assert!(
        crate::network::openai::moa_gateway::context_selection::should_advertise_virtual_mesh(
            &models
        ),
        "the directive must be listed whenever the mesh serves anything"
    );
}

#[tokio::test]
async fn an_explicitly_named_model_is_never_reinterpreted() {
    let (node, targets) = node_serving(&["text-model", "vision-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("vision-model", true, false),
    ];

    let resolution = resolve(
        Some("text-model"),
        &text_body(Some("text-model")),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => assert_eq!(effective_model.as_deref(), Some("text-model")),
        AutoRouteResolution::MediaUnsupported => panic!("explicit routing is untouched"),
    }
}

/// Forwarded body `model` for an automatic request, after the decision is
/// committed to the request that will actually be sent.
async fn forwarded_model_field(
    model: Option<&str>,
    body: &serde_json::Value,
    node: &mesh::Node,
    targets: &election::ModelTargets,
    descriptors: &[mesh::ServedModelDescriptor],
) -> (Option<String>, Option<String>) {
    let mut request = request_with_body(model, body);
    let affinity = affinity::AffinityRouter::new();
    let resolution = resolve_auto_routed_model(
        node,
        &mut request,
        targets,
        None,
        descriptors,
        None,
        &affinity,
    )
    .await;
    let AutoRouteResolution::Continue {
        effective_model, ..
    } = resolution
    else {
        panic!("expected an automatic resolution, got MediaUnsupported");
    };
    super::super::ingress::maybe_enable_auto_route_hooks(&mut request, effective_model.as_deref());
    let forwarded = request
        .raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
        .and_then(|start| serde_json::from_slice::<serde_json::Value>(&request.raw[start..]).ok())
        .and_then(|body| {
            body.get("model")
                .and_then(serde_json::Value::as_str)
                .map(ToString::to_string)
        });
    (effective_model, forwarded)
}

#[tokio::test]
async fn a_single_model_decision_is_written_into_the_forwarded_body() {
    // The backend reads `model` from the forwarded bytes and 404s anything that
    // is not its own advertised identity, so selecting `vision-model` while
    // still sending `"model":"mesh"` never reaches a model at all.
    let (node, targets) = node_serving(&["text-model", "vision-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("vision-model", true, false),
    ];

    let (effective, forwarded) = forwarded_model_field(
        Some(automatic::DIRECTIVE),
        &image_body(automatic::DIRECTIVE),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    assert_eq!(effective.as_deref(), Some("vision-model"));
    assert_eq!(
        forwarded.as_deref(),
        Some("vision-model"),
        "the forwarded body must carry the model the request was routed to"
    );
}

#[tokio::test]
async fn committee_mode_keeps_the_directive_in_the_forwarded_body() {
    // The MoA gateway self-gates on the body's model, so committee mode must
    // not have the directive rewritten away.
    let (node, targets) = node_serving(&["text-model", "other-text-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("other-text-model", false, false),
    ];

    let (effective, forwarded) = forwarded_model_field(
        Some(automatic::DIRECTIVE),
        &text_body(Some(automatic::DIRECTIVE)),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    assert_eq!(effective.as_deref(), Some(automatic::DIRECTIVE));
    assert_eq!(forwarded.as_deref(), Some(automatic::DIRECTIVE));
}

#[tokio::test]
async fn a_non_chat_endpoint_resolves_to_a_single_model() {
    // `model=auto` on `/v1/completions` selected one concrete model before this
    // change. Entering committee mode would reject it for having no `messages`
    // array, breaking a request shape that already worked.
    let (node, targets) = node_serving(&["text-model", "other-text-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("other-text-model", false, false),
    ];
    let body = serde_json::json!({
        "model": automatic::DEPRECATED_ALIAS,
        "prompt": "once upon a time",
    });
    let mut request = request_with_body(Some(automatic::DEPRECATED_ALIAS), &body);
    request.path = "/v1/completions".to_owned();
    request.client_path = "/v1/completions".to_owned();
    let affinity = affinity::AffinityRouter::new();

    let resolution = resolve_auto_routed_model(
        &node,
        &mut request,
        &targets,
        None,
        &descriptors,
        None,
        &affinity,
    )
    .await;

    // Pin the reason, not just "left the committee": asserting only
    // `!= DIRECTIVE` would keep passing if the endpoint gate were deleted and
    // some unrelated condition happened to divert the request anyway.
    assert_eq!(
        automatic::envelope_mode(automatic::AutomaticRequest {
            model: Some(automatic::DEPRECATED_ALIAS),
            path: "/v1/completions",
            body: &body,
        }),
        automatic::ServingMode::SingleModel(automatic::SingleModelReason::NonChatRequest)
    );

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => {
            let model = effective_model.expect("a completions request must resolve to a model");
            assert_ne!(
                model,
                automatic::DIRECTIVE,
                "a non-chat request must not convene a committee it cannot fan out"
            );
        }
        AutoRouteResolution::MediaUnsupported => panic!("no media in this request"),
    }
}
