use crate::mesh::{Node, owner_control_error_envelope};
use crate::proto::node::{
    OwnerControlDrainModelRequest, OwnerControlDrainModelResponse, OwnerControlEnsureModelRequest,
    OwnerControlEnsureModelResponse, OwnerControlEnvelope, OwnerControlErrorCode,
    OwnerControlLoadModelRequest, OwnerControlLoadModelResponse, OwnerControlResponse,
    OwnerControlUnloadModelRequest, OwnerControlUnloadModelResponse,
};
use crate::protocol::NODE_PROTOCOL_GENERATION;
use crate::runtime::{IntentSource, ModelIntent, UnloadOptions, UnloadTarget};
use mesh_llm_protocol::protocol::{
    validate_owner_control_model_for_load_or_ensure,
    validate_owner_control_model_for_unload_or_drain,
};

pub(crate) async fn execute_load(
    node: &Node,
    request_id: u64,
    request: OwnerControlLoadModelRequest,
) -> OwnerControlEnvelope {
    let intent_id = owner_intent_id(&request.requester_node_id, request_id);
    let profile = request.profile.unwrap_or_default();
    let model_ref = match extract_present_model_ref(request.model, request_id) {
        Ok(ref_) => ref_,
        Err(envelope) => return *envelope,
    };

    send_model_intent(
        node,
        request_id,
        ModelIntent::Load {
            intent_id: Some(intent_id.clone()),
            spec: model_ref.canonical_model_ref.clone(),
            config_model_id: Some(model_ref.canonical_model_ref.clone()),
            profile,
            source: IntentSource::OwnerLoad,
            completion: None,
        },
        LifecycleOperation::Load,
        intent_id,
        model_ref,
    )
    .await
}

pub(crate) async fn execute_unload(
    node: &Node,
    request_id: u64,
    request: OwnerControlUnloadModelRequest,
) -> OwnerControlEnvelope {
    let intent_id = owner_intent_id(&request.requester_node_id, request_id);
    let model_ref = match extract_absent_model_ref(request.model, request_id) {
        Ok(ref_) => ref_,
        Err(envelope) => return *envelope,
    };

    let target = build_unload_target(&model_ref);
    send_model_intent(
        node,
        request_id,
        ModelIntent::Unload {
            intent_id: Some(intent_id.clone()),
            canonical_model_ref: non_empty_model_ref(&model_ref),
            target,
            options: UnloadOptions::default(),
            source: IntentSource::OwnerUnload,
            completion: None,
        },
        LifecycleOperation::Unload,
        intent_id,
        model_ref,
    )
    .await
}

pub(crate) async fn execute_ensure(
    node: &Node,
    request_id: u64,
    request: OwnerControlEnsureModelRequest,
) -> OwnerControlEnvelope {
    let intent_id = owner_intent_id(&request.requester_node_id, request_id);
    let profile = request.profile.unwrap_or_default();
    let model_ref = match extract_present_model_ref(request.model, request_id) {
        Ok(ref_) => ref_,
        Err(envelope) => return *envelope,
    };

    send_model_intent(
        node,
        request_id,
        ModelIntent::Load {
            intent_id: Some(intent_id.clone()),
            spec: model_ref.canonical_model_ref.clone(),
            config_model_id: Some(model_ref.canonical_model_ref.clone()),
            profile,
            source: IntentSource::OwnerEnsure,
            completion: None,
        },
        LifecycleOperation::Ensure,
        intent_id,
        model_ref,
    )
    .await
}

pub(crate) async fn execute_drain(
    node: &Node,
    request_id: u64,
    request: OwnerControlDrainModelRequest,
) -> OwnerControlEnvelope {
    let intent_id = owner_intent_id(&request.requester_node_id, request_id);
    let drain_timeout_secs = request
        .drain_timeout_secs
        .unwrap_or(node.drain_timeout_secs);
    if let Err(envelope) =
        validate_drain_timeout(drain_timeout_secs, node.drain_timeout_max_secs, request_id)
    {
        return *envelope;
    }
    let model_ref = match extract_absent_model_ref(request.model, request_id) {
        Ok(ref_) => ref_,
        Err(envelope) => return *envelope,
    };

    let target = build_unload_target(&model_ref);
    send_model_intent(
        node,
        request_id,
        ModelIntent::Unload {
            intent_id: Some(intent_id.clone()),
            canonical_model_ref: non_empty_model_ref(&model_ref),
            target,
            options: UnloadOptions {
                drain_timeout: std::time::Duration::from_secs(drain_timeout_secs),
                force: false,
            },
            source: IntentSource::OwnerDrain,
            completion: None,
        },
        LifecycleOperation::Drain,
        intent_id,
        model_ref,
    )
    .await
}

fn validate_drain_timeout(
    drain_timeout_secs: u64,
    drain_timeout_max_secs: u64,
    request_id: u64,
) -> Result<(), Box<OwnerControlEnvelope>> {
    if drain_timeout_secs == 0 || drain_timeout_secs > drain_timeout_max_secs {
        Err(Box::new(owner_control_error_envelope(
            OwnerControlErrorCode::BadRequest,
            Some(request_id),
            None,
            "drain_timeout_secs must be positive and not exceed configured maximum",
        )))
    } else {
        Ok(())
    }
}

fn extract_present_model_ref(
    model: Option<crate::proto::node::OwnerControlModelRef>,
    request_id: u64,
) -> Result<crate::proto::node::OwnerControlModelRef, Box<OwnerControlEnvelope>> {
    match model {
        Some(ref_) if validate_owner_control_model_for_load_or_ensure(&ref_).is_ok() => Ok(ref_),
        Some(_) => Err(Box::new(owner_control_error_envelope(
            OwnerControlErrorCode::BadRequest,
            Some(request_id),
            None,
            "load and ensure require a canonical model reference only",
        ))),
        None => Err(Box::new(owner_control_error_envelope(
            OwnerControlErrorCode::BadRequest,
            Some(request_id),
            None,
            "model field is required",
        ))),
    }
}

fn extract_absent_model_ref(
    model: Option<crate::proto::node::OwnerControlModelRef>,
    request_id: u64,
) -> Result<crate::proto::node::OwnerControlModelRef, Box<OwnerControlEnvelope>> {
    match model {
        Some(ref_) if validate_owner_control_model_for_unload_or_drain(&ref_).is_ok() => Ok(ref_),
        Some(_) => Err(Box::new(owner_control_error_envelope(
            OwnerControlErrorCode::BadRequest,
            Some(request_id),
            None,
            "unload and drain require exactly one model reference or instance id",
        ))),
        None => Err(Box::new(owner_control_error_envelope(
            OwnerControlErrorCode::BadRequest,
            Some(request_id),
            None,
            "model field is required",
        ))),
    }
}

fn non_empty_model_ref(model_ref: &crate::proto::node::OwnerControlModelRef) -> Option<String> {
    (!model_ref.canonical_model_ref.trim().is_empty())
        .then(|| model_ref.canonical_model_ref.clone())
}

fn build_unload_target(model_ref: &crate::proto::node::OwnerControlModelRef) -> UnloadTarget {
    match &model_ref.instance_id {
        Some(id) if !id.trim().is_empty() => UnloadTarget::Instance(id.clone()),
        _ => UnloadTarget::Model(model_ref.canonical_model_ref.clone()),
    }
}

async fn send_model_intent(
    node: &Node,
    request_id: u64,
    intent: ModelIntent,
    operation: LifecycleOperation,
    intent_id: String,
    target: crate::proto::node::OwnerControlModelRef,
) -> OwnerControlEnvelope {
    let sender = {
        let guard = node.model_intent_tx.lock().await;
        guard.as_ref().cloned()
    };

    let Some(tx) = sender else {
        return owner_control_error_envelope(
            OwnerControlErrorCode::ControlUnavailable,
            Some(request_id),
            None,
            "model intent channel unavailable",
        );
    };

    match tx.send(intent).await {
        Ok(()) => success_lifecycle_envelope(request_id, operation, intent_id, target),
        Err(_) => owner_control_error_envelope(
            OwnerControlErrorCode::ControlUnavailable,
            Some(request_id),
            None,
            "failed to enqueue model intent",
        ),
    }
}

#[derive(Clone, Copy)]
enum LifecycleOperation {
    Load,
    Unload,
    Ensure,
    Drain,
}

fn owner_intent_id(requester_node_id: &[u8], request_id: u64) -> String {
    let requester = hex::encode(requester_node_id.get(..8).unwrap_or(requester_node_id));
    format!("owner-{requester}-{request_id}")
}

fn success_lifecycle_envelope(
    request_id: u64,
    operation: LifecycleOperation,
    intent_id: String,
    target: crate::proto::node::OwnerControlModelRef,
) -> OwnerControlEnvelope {
    let mut response = OwnerControlResponse {
        request_id,
        get_config: None,
        watch_config: None,
        apply_config: None,
        refresh_inventory: None,
        load_model: None,
        unload_model: None,
        ensure_model: None,
        drain_model: None,
        kv_cache: None,
    };
    match operation {
        LifecycleOperation::Load => {
            response.load_model = Some(OwnerControlLoadModelResponse {
                intent_id,
                accepted_state: "present".to_string(),
                target: Some(target),
            });
        }
        LifecycleOperation::Unload => {
            response.unload_model = Some(OwnerControlUnloadModelResponse {
                intent_id,
                accepted_state: "absent".to_string(),
                target: Some(target),
            });
        }
        LifecycleOperation::Ensure => {
            response.ensure_model = Some(OwnerControlEnsureModelResponse {
                intent_id,
                accepted_state: "present".to_string(),
                target: Some(target),
            });
        }
        LifecycleOperation::Drain => {
            response.drain_model = Some(OwnerControlDrainModelResponse {
                intent_id,
                accepted_state: "draining".to_string(),
                target: Some(target),
            });
        }
    }

    OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: None,
        response: Some(response),
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unload_and_drain_accept_exactly_one_model_or_instance_target() {
        let instance = extract_absent_model_ref(
            Some(crate::proto::node::OwnerControlModelRef {
                canonical_model_ref: String::new(),
                instance_id: Some("runtime-2".to_string()),
            }),
            42,
        )
        .expect("instance-only target should be valid");
        assert_eq!(
            build_unload_target(&instance),
            UnloadTarget::Instance("runtime-2".into())
        );
        assert_eq!(non_empty_model_ref(&instance), None);

        let model = extract_absent_model_ref(
            Some(crate::proto::node::OwnerControlModelRef {
                canonical_model_ref: "model/test".to_string(),
                instance_id: None,
            }),
            42,
        )
        .expect("model-only target should be valid");
        assert_eq!(
            build_unload_target(&model),
            UnloadTarget::Model("model/test".into())
        );
        assert_eq!(non_empty_model_ref(&model).as_deref(), Some("model/test"));

        let model_with_blank_instance = extract_absent_model_ref(
            Some(crate::proto::node::OwnerControlModelRef {
                canonical_model_ref: "model/test".to_string(),
                instance_id: Some("   ".to_string()),
            }),
            42,
        )
        .expect("blank instance id should be treated as absent");
        assert_eq!(
            build_unload_target(&model_with_blank_instance),
            UnloadTarget::Model("model/test".into())
        );

        let invalid = extract_absent_model_ref(
            Some(crate::proto::node::OwnerControlModelRef {
                canonical_model_ref: "model/test".to_string(),
                instance_id: Some("runtime-2".to_string()),
            }),
            42,
        )
        .expect_err("ambiguous target should be rejected");
        assert_eq!(
            invalid.error.as_ref().and_then(|error| error.request_id),
            Some(42)
        );

        let missing = extract_absent_model_ref(None, 43).expect_err("missing target should fail");
        assert_eq!(
            missing.error.as_ref().and_then(|error| error.request_id),
            Some(43)
        );
    }

    #[test]
    fn drain_timeout_must_be_positive_and_within_the_configured_maximum() {
        for accepted in [1, 60] {
            assert!(validate_drain_timeout(accepted, 60, 41).is_ok());
        }

        for rejected in [0, 61] {
            let envelope =
                validate_drain_timeout(rejected, 60, 42).expect_err("timeout should fail");
            let error = envelope.error.expect("error envelope");
            assert_eq!(error.code, OwnerControlErrorCode::BadRequest as i32);
            assert_eq!(error.request_id, Some(42));
        }
    }

    #[test]
    fn lifecycle_success_envelope_contains_only_requested_response() {
        let operations = [
            LifecycleOperation::Load,
            LifecycleOperation::Unload,
            LifecycleOperation::Ensure,
            LifecycleOperation::Drain,
        ];

        for (index, operation) in operations.into_iter().enumerate() {
            let target = crate::proto::node::OwnerControlModelRef {
                canonical_model_ref: "model/test".to_string(),
                instance_id: None,
            };
            let response = success_lifecycle_envelope(
                index as u64,
                operation,
                format!("test-intent-{index}"),
                target.clone(),
            )
            .response
            .expect("response envelope");
            let present = [
                response.load_model.is_some(),
                response.unload_model.is_some(),
                response.ensure_model.is_some(),
                response.drain_model.is_some(),
            ];
            assert_eq!(present.iter().filter(|value| **value).count(), 1);
            assert!(present[index]);
            let (intent_id, accepted_state, response_target) = match operation {
                LifecycleOperation::Load => {
                    let payload = response.load_model.expect("load response");
                    (payload.intent_id, payload.accepted_state, payload.target)
                }
                LifecycleOperation::Unload => {
                    let payload = response.unload_model.expect("unload response");
                    (payload.intent_id, payload.accepted_state, payload.target)
                }
                LifecycleOperation::Ensure => {
                    let payload = response.ensure_model.expect("ensure response");
                    (payload.intent_id, payload.accepted_state, payload.target)
                }
                LifecycleOperation::Drain => {
                    let payload = response.drain_model.expect("drain response");
                    (payload.intent_id, payload.accepted_state, payload.target)
                }
            };
            assert_eq!(intent_id, format!("test-intent-{index}"));
            assert_eq!(response_target, Some(target));
            assert_eq!(
                accepted_state,
                match operation {
                    LifecycleOperation::Load | LifecycleOperation::Ensure => "present",
                    LifecycleOperation::Unload => "absent",
                    LifecycleOperation::Drain => "draining",
                }
            );
        }
    }
}
