pub(crate) mod kv_cache;
pub(crate) mod model_lifecycle;
pub(crate) mod scan_refresh;

const OWNER_CONTROL_KV_CACHE_DEADLINE_SECS: u64 = 30;

use crate::proto::node::{
    OwnerControlApplyConfigRequest, OwnerControlDrainModelRequest, OwnerControlEnsureModelRequest,
    OwnerControlGetConfigRequest, OwnerControlKvCacheRequest, OwnerControlLoadModelRequest,
    OwnerControlRefreshInventoryRequest, OwnerControlRequest, OwnerControlUnloadModelRequest,
    OwnerControlWatchConfigRequest,
};
use std::future::Future;
use std::time::Duration;

pub(crate) const OWNER_CONTROL_SCAN_DEADLINE_SECS: u64 = 30;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OwnedNodeCommandExecutionShape {
    Unary,
    Watch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OwnedNodeCommandDeadline {
    Unary(Duration),
    Scan(Duration),
    Watch,
}

impl OwnedNodeCommandDeadline {
    pub(crate) fn timeout_message(self) -> String {
        match self {
            Self::Unary(duration) => format!(
                "owner-control unary command timed out after {}s",
                duration.as_secs()
            ),
            Self::Scan(duration) => format!(
                "owner-control inventory scan timed out after {}s",
                duration.as_secs()
            ),
            Self::Watch => "owner-control watch commands do not have a unary deadline".to_string(),
        }
    }
}

pub(crate) async fn await_command_deadline<T, F>(
    deadline: OwnedNodeCommandDeadline,
    future: F,
) -> Result<T, OwnedNodeCommandDeadline>
where
    F: Future<Output = T>,
{
    let duration = match deadline {
        OwnedNodeCommandDeadline::Unary(duration) | OwnedNodeCommandDeadline::Scan(duration) => {
            duration
        }
        OwnedNodeCommandDeadline::Watch => return Ok(future.await),
    };
    tokio::time::timeout(duration, future)
        .await
        .map_err(|_| deadline)
}

#[derive(Debug)]
pub(crate) enum OwnedNodeCommand {
    GetConfig {
        request_id: u64,
        request: OwnerControlGetConfigRequest,
    },
    WatchConfig {
        request_id: u64,
        request: OwnerControlWatchConfigRequest,
    },
    ApplyConfig {
        request_id: u64,
        request: OwnerControlApplyConfigRequest,
    },
    ScanRefresh {
        request_id: u64,
        request: OwnerControlRefreshInventoryRequest,
    },
    LoadModel {
        request_id: u64,
        request: OwnerControlLoadModelRequest,
    },
    UnloadModel {
        request_id: u64,
        request: OwnerControlUnloadModelRequest,
    },
    EnsureModel {
        request_id: u64,
        request: OwnerControlEnsureModelRequest,
    },
    DrainModel {
        request_id: u64,
        request: OwnerControlDrainModelRequest,
    },
    KvCache {
        request_id: u64,
        request: OwnerControlKvCacheRequest,
    },
}

impl OwnedNodeCommand {
    pub(crate) fn decode(request: OwnerControlRequest) -> Option<Self> {
        let request_id = request.request_id;
        if let Some(request) = request.get_config {
            return Some(Self::GetConfig {
                request_id,
                request,
            });
        }
        if let Some(request) = request.watch_config {
            return Some(Self::WatchConfig {
                request_id,
                request,
            });
        }
        if let Some(request) = request.apply_config {
            return Some(Self::ApplyConfig {
                request_id,
                request,
            });
        }
        if let Some(request) = request.refresh_inventory {
            return Some(Self::ScanRefresh {
                request_id,
                request,
            });
        }
        if let Some(request) = request.load_model {
            return Some(Self::LoadModel {
                request_id,
                request,
            });
        }
        if let Some(request) = request.unload_model {
            return Some(Self::UnloadModel {
                request_id,
                request,
            });
        }
        if let Some(request) = request.ensure_model {
            return Some(Self::EnsureModel {
                request_id,
                request,
            });
        }
        if let Some(request) = request.drain_model {
            return Some(Self::DrainModel {
                request_id,
                request,
            });
        }
        request.kv_cache.map(|request| Self::KvCache {
            request_id,
            request,
        })
    }

    pub(crate) fn request_id(&self) -> u64 {
        match self {
            Self::GetConfig { request_id, .. }
            | Self::WatchConfig { request_id, .. }
            | Self::ApplyConfig { request_id, .. }
            | Self::ScanRefresh { request_id, .. }
            | Self::LoadModel { request_id, .. }
            | Self::UnloadModel { request_id, .. }
            | Self::EnsureModel { request_id, .. }
            | Self::DrainModel { request_id, .. }
            | Self::KvCache { request_id, .. } => *request_id,
        }
    }

    pub(crate) fn requester_node_id(&self) -> &[u8] {
        match self {
            Self::GetConfig { request, .. } => &request.requester_node_id,
            Self::WatchConfig { request, .. } => &request.requester_node_id,
            Self::ApplyConfig { request, .. } => &request.requester_node_id,
            Self::ScanRefresh { request, .. } => &request.requester_node_id,
            Self::LoadModel { request, .. } => &request.requester_node_id,
            Self::UnloadModel { request, .. } => &request.requester_node_id,
            Self::EnsureModel { request, .. } => &request.requester_node_id,
            Self::DrainModel { request, .. } => &request.requester_node_id,
            Self::KvCache { request, .. } => &request.requester_node_id,
        }
    }

    pub(crate) fn target_node_id(&self) -> &[u8] {
        match self {
            Self::GetConfig { request, .. } => &request.target_node_id,
            Self::WatchConfig { request, .. } => &request.target_node_id,
            Self::ApplyConfig { request, .. } => &request.target_node_id,
            Self::ScanRefresh { request, .. } => &request.target_node_id,
            Self::LoadModel { request, .. } => &request.target_node_id,
            Self::UnloadModel { request, .. } => &request.target_node_id,
            Self::EnsureModel { request, .. } => &request.target_node_id,
            Self::DrainModel { request, .. } => &request.target_node_id,
            Self::KvCache { request, .. } => &request.target_node_id,
        }
    }

    pub(crate) fn execution_shape(&self) -> OwnedNodeCommandExecutionShape {
        match self {
            Self::WatchConfig { .. } => OwnedNodeCommandExecutionShape::Watch,
            Self::GetConfig { .. }
            | Self::ApplyConfig { .. }
            | Self::ScanRefresh { .. }
            | Self::LoadModel { .. }
            | Self::UnloadModel { .. }
            | Self::EnsureModel { .. }
            | Self::DrainModel { .. }
            | Self::KvCache { .. } => OwnedNodeCommandExecutionShape::Unary,
        }
    }

    pub(crate) fn deadline(&self) -> OwnedNodeCommandDeadline {
        match self {
            Self::GetConfig { .. }
            | Self::ApplyConfig { .. }
            | Self::LoadModel { .. }
            | Self::UnloadModel { .. }
            | Self::EnsureModel { .. }
            | Self::DrainModel { .. } => OwnedNodeCommandDeadline::Unary(Duration::from_secs(5)),
            Self::KvCache { .. } => OwnedNodeCommandDeadline::Unary(Duration::from_secs(
                OWNER_CONTROL_KV_CACHE_DEADLINE_SECS,
            )),
            Self::ScanRefresh { .. } => OwnedNodeCommandDeadline::Scan(Duration::from_secs(
                OWNER_CONTROL_SCAN_DEADLINE_SECS,
            )),
            Self::WatchConfig { .. } => OwnedNodeCommandDeadline::Watch,
        }
    }

    pub(crate) fn is_model_lifecycle(&self) -> bool {
        matches!(
            self,
            Self::LoadModel { .. }
                | Self::UnloadModel { .. }
                | Self::EnsureModel { .. }
                | Self::DrainModel { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ValidateControlFrame;
    use prost::Message;

    #[test]
    fn command_metadata_is_exhaustive_and_shared() {
        let command = OwnedNodeCommand::decode(OwnerControlRequest {
            request_id: 41,
            get_config: None,
            watch_config: None,
            apply_config: None,
            refresh_inventory: Some(crate::proto::node::OwnerControlRefreshInventoryRequest {
                requester_node_id: vec![1],
                target_node_id: vec![2],
            }),
            load_model: None,
            unload_model: None,
            ensure_model: None,
            drain_model: None,
            kv_cache: None,
        })
        .expect("typed command");

        assert_eq!(command.request_id(), 41);
        assert_eq!(command.requester_node_id(), [1]);
        assert_eq!(command.target_node_id(), [2]);
        assert_eq!(
            command.execution_shape(),
            OwnedNodeCommandExecutionShape::Unary
        );
        assert_eq!(
            command.deadline(),
            OwnedNodeCommandDeadline::Scan(Duration::from_secs(30))
        );
    }

    #[test]
    fn kv_cache_command_is_authenticated_unary_work() {
        let command = OwnedNodeCommand::decode(OwnerControlRequest {
            request_id: 42,
            kv_cache: Some(crate::proto::node::OwnerControlKvCacheRequest {
                requester_node_id: vec![1; 32],
                target_node_id: vec![2; 32],
                operation: crate::proto::node::OwnerControlKvCacheOperation::Status as i32,
                target_bytes: None,
                model_identity: None,
            }),
            ..Default::default()
        })
        .expect("typed command");

        assert_eq!(command.request_id(), 42);
        assert_eq!(command.requester_node_id(), [1; 32]);
        assert_eq!(command.target_node_id(), [2; 32]);
        assert_eq!(
            command.execution_shape(),
            OwnedNodeCommandExecutionShape::Unary
        );
        assert_eq!(
            command.deadline(),
            OwnedNodeCommandDeadline::Unary(Duration::from_secs(
                OWNER_CONTROL_KV_CACHE_DEADLINE_SECS
            ))
        );
        assert!(!command.is_model_lifecycle());
    }

    #[tokio::test(start_paused = true)]
    async fn slow_scan_within_deadline_completes() {
        let deadline = OwnedNodeCommandDeadline::Scan(Duration::from_secs(30));
        let result = await_command_deadline(deadline, async {
            tokio::time::sleep(Duration::from_secs(29)).await;
            "complete"
        })
        .await;

        assert_eq!(result, Ok("complete"));
    }

    #[tokio::test(start_paused = true)]
    async fn scan_exceeding_deadline_is_cancelled_deterministically() {
        let deadline = OwnedNodeCommandDeadline::Scan(Duration::from_secs(30));
        let result = await_command_deadline(deadline, async {
            tokio::time::sleep(Duration::from_secs(31)).await;
        })
        .await;

        assert_eq!(result, Err(deadline));
        assert_eq!(
            deadline.timeout_message(),
            "owner-control inventory scan timed out after 30s"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn accepted_watch_does_not_inherit_unary_deadline() {
        let result = await_command_deadline(OwnedNodeCommandDeadline::Watch, async {
            tokio::time::sleep(Duration::from_secs(31)).await;
            "still-open"
        })
        .await;

        assert_eq!(result, Ok("still-open"));
    }

    #[test]
    fn oversized_command_result_maps_to_control_unavailable() {
        let envelope = crate::proto::node::OwnerControlEnvelope {
            r#gen: crate::protocol::NODE_PROTOCOL_GENERATION,
            handshake: None,
            request: None,
            response: Some(crate::proto::node::OwnerControlResponse {
                request_id: 73,
                get_config: None,
                watch_config: None,
                apply_config: None,
                refresh_inventory: Some(crate::proto::node::OwnerControlRefreshInventoryResponse {
                    snapshot: None,
                    inventory: Some(crate::proto::node::OwnerControlRefreshInventory {
                        entries: vec![crate::proto::node::OwnerControlInventoryEntry {
                            canonical_model_ref: "x"
                                .repeat(crate::protocol::MAX_CONTROL_FRAME_BYTES),
                            display_name: None,
                            total_size_bytes: 0,
                            metadata: None,
                        }],
                        disposition:
                            crate::proto::node::OwnerControlRefreshInventoryDisposition::Executed
                                as i32,
                    }),
                }),
                load_model: None,
                unload_model: None,
                ensure_model: None,
                drain_model: None,
                kv_cache: None,
            }),
            error: None,
        };
        assert!(
            envelope.encode_to_vec().len() > crate::protocol::MAX_CONTROL_FRAME_BYTES,
            "fixture must exceed the bound"
        );

        let bounded = super::super::bound_owner_control_envelope(envelope);
        let error = bounded
            .error
            .as_ref()
            .expect("oversized result becomes an error");
        assert_eq!(error.request_id, Some(73));
        assert_eq!(
            crate::proto::node::OwnerControlErrorCode::try_from(error.code),
            Ok(crate::proto::node::OwnerControlErrorCode::ControlUnavailable)
        );
        assert!(bounded.encode_to_vec().len() < crate::protocol::MAX_CONTROL_FRAME_BYTES);
    }

    fn lifecycle_model_request(model: &str) -> crate::proto::node::OwnerControlModelRef {
        crate::proto::node::OwnerControlModelRef {
            canonical_model_ref: model.to_string(),
            instance_id: Some("instance-default".to_string()),
        }
    }

    fn make_lifecycle_base(request_id: u64) -> crate::proto::node::OwnerControlRequest {
        crate::proto::node::OwnerControlRequest {
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
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn owner_lifecycle_messages_round_trip() {
        let requester = vec![1u8; 32];
        let target = vec![2u8; 32];
        let model_ref = lifecycle_model_request("org/model:file.gguf");

        let load_request = crate::proto::node::OwnerControlRequest {
            load_model: Some(crate::proto::node::OwnerControlLoadModelRequest {
                requester_node_id: requester.clone(),
                target_node_id: target.clone(),
                model: Some(model_ref.clone()),
                profile: Some("low-ctx".into()),
            }),
            ..make_lifecycle_base(101)
        };
        let encoded = load_request.encode_to_vec();
        let parsed = crate::proto::node::OwnerControlRequest::decode(encoded.as_slice())
            .expect("load-model request should decode");
        let command =
            OwnedNodeCommand::decode(parsed).expect("load-model must decode to a command");
        match command {
            OwnedNodeCommand::LoadModel {
                request_id,
                request,
            } => {
                assert_eq!(request_id, 101);
                assert_eq!(request.requester_node_id, requester);
                assert_eq!(request.target_node_id, target);
                assert_eq!(request.model, Some(model_ref.clone()));
                assert_eq!(request.profile.as_deref(), Some("low-ctx"));
            }
            _ => panic!("expected owner-control load-model command"),
        }

        let unload_request = crate::proto::node::OwnerControlRequest {
            unload_model: Some(crate::proto::node::OwnerControlUnloadModelRequest {
                requester_node_id: requester.clone(),
                target_node_id: target.clone(),
                model: Some(model_ref.clone()),
            }),
            ..make_lifecycle_base(102)
        };
        let encoded = unload_request.encode_to_vec();
        let parsed = crate::proto::node::OwnerControlRequest::decode(encoded.as_slice())
            .expect("unload-model request should decode");
        let command =
            OwnedNodeCommand::decode(parsed).expect("unload-model must decode to a command");
        match command {
            OwnedNodeCommand::UnloadModel {
                request_id,
                request,
            } => {
                assert_eq!(request_id, 102);
                assert_eq!(request.requester_node_id, requester);
                assert_eq!(request.target_node_id, target);
                assert_eq!(request.model, Some(model_ref.clone()));
            }
            _ => panic!("expected owner-control unload-model command"),
        }

        let ensure_request = crate::proto::node::OwnerControlRequest {
            ensure_model: Some(crate::proto::node::OwnerControlEnsureModelRequest {
                requester_node_id: requester.clone(),
                target_node_id: target.clone(),
                model: Some(model_ref.clone()),
                profile: Some("throughput".into()),
            }),
            ..make_lifecycle_base(103)
        };
        let encoded = ensure_request.encode_to_vec();
        let parsed = crate::proto::node::OwnerControlRequest::decode(encoded.as_slice())
            .expect("ensure-model request should decode");
        let command =
            OwnedNodeCommand::decode(parsed).expect("ensure-model must decode to a command");
        match command {
            OwnedNodeCommand::EnsureModel {
                request_id,
                request,
            } => {
                assert_eq!(request_id, 103);
                assert_eq!(request.requester_node_id, requester);
                assert_eq!(request.target_node_id, target);
                assert_eq!(request.model, Some(model_ref.clone()));
                assert_eq!(request.profile.as_deref(), Some("throughput"));
            }
            _ => panic!("expected owner-control ensure-model command"),
        }

        let drain_request = crate::proto::node::OwnerControlRequest {
            drain_model: Some(crate::proto::node::OwnerControlDrainModelRequest {
                requester_node_id: requester,
                target_node_id: target,
                model: Some(model_ref),
                drain_timeout_secs: None,
            }),
            ..make_lifecycle_base(104)
        };
        let encoded = drain_request.encode_to_vec();
        let parsed = crate::proto::node::OwnerControlRequest::decode(encoded.as_slice())
            .expect("drain-model request should decode");
        let command =
            OwnedNodeCommand::decode(parsed).expect("drain-model must decode to a command");
        match command {
            OwnedNodeCommand::DrainModel {
                request_id,
                request,
            } => {
                assert_eq!(request_id, 104);
                assert_eq!(request.target_node_id.len(), 32);
                assert_eq!(
                    request.model.map(|m| m.canonical_model_ref),
                    Some("org/model:file.gguf".to_string())
                );
            }
            _ => panic!("expected owner-control drain-model command"),
        }
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn owner_lifecycle_typed_decode() {
        let requester = vec![1u8; 32];
        let target = vec![2u8; 32];
        let load_model = crate::proto::node::OwnerControlModelRef {
            canonical_model_ref: "org/model:file.gguf".into(),
            instance_id: Some("instance-load".into()),
        };

        let load_request = crate::proto::node::OwnerControlRequest {
            request_id: 201,
            load_model: Some(crate::proto::node::OwnerControlLoadModelRequest {
                requester_node_id: requester.clone(),
                target_node_id: target.clone(),
                model: Some(load_model.clone()),
                profile: None,
            }),
            ..make_lifecycle_base(201)
        };
        let load_decoded = OwnedNodeCommand::decode(
            crate::proto::node::OwnerControlRequest::decode(
                load_request.encode_to_vec().as_slice(),
            )
            .expect("load command should decode"),
        )
        .expect("load command should map");
        match load_decoded {
            OwnedNodeCommand::LoadModel { request, .. } => {
                assert_eq!(request.requester_node_id, requester);
                assert_eq!(request.target_node_id, target);
                assert_eq!(
                    request.model,
                    Some(crate::proto::node::OwnerControlModelRef {
                        canonical_model_ref: "org/model:file.gguf".into(),
                        instance_id: Some("instance-load".into()),
                    })
                );
            }
            _ => panic!("expected load-model typed variant"),
        }

        let unload_request = crate::proto::node::OwnerControlRequest {
            request_id: 202,
            unload_model: Some(crate::proto::node::OwnerControlUnloadModelRequest {
                requester_node_id: vec![1u8; 32],
                target_node_id: vec![2u8; 32],
                model: Some(crate::proto::node::OwnerControlModelRef {
                    canonical_model_ref: "org/model:file.gguf".into(),
                    instance_id: Some("instance-unload".into()),
                }),
            }),
            ..make_lifecycle_base(202)
        };
        let unload_decoded = OwnedNodeCommand::decode(
            crate::proto::node::OwnerControlRequest::decode(
                unload_request.encode_to_vec().as_slice(),
            )
            .expect("unload command should decode"),
        )
        .expect("unload command should map");
        match unload_decoded {
            OwnedNodeCommand::UnloadModel { request, .. } => {
                assert_eq!(request.requester_node_id, vec![1u8; 32]);
                assert_eq!(request.target_node_id, vec![2u8; 32]);
                assert_eq!(
                    request.model,
                    Some(crate::proto::node::OwnerControlModelRef {
                        canonical_model_ref: "org/model:file.gguf".into(),
                        instance_id: Some("instance-unload".into()),
                    })
                );
            }
            _ => panic!("expected unload-model typed variant"),
        }

        let ensure_request = crate::proto::node::OwnerControlRequest {
            request_id: 203,
            ensure_model: Some(crate::proto::node::OwnerControlEnsureModelRequest {
                requester_node_id: vec![1u8; 32],
                target_node_id: vec![2u8; 32],
                model: Some(crate::proto::node::OwnerControlModelRef {
                    canonical_model_ref: "org/model:file.gguf".into(),
                    instance_id: Some("instance-ensure".into()),
                }),
                profile: None,
            }),
            ..make_lifecycle_base(203)
        };
        let ensure_decoded = OwnedNodeCommand::decode(
            crate::proto::node::OwnerControlRequest::decode(
                ensure_request.encode_to_vec().as_slice(),
            )
            .expect("ensure command should decode"),
        )
        .expect("ensure command should map");
        match ensure_decoded {
            OwnedNodeCommand::EnsureModel { request, .. } => {
                assert_eq!(request.requester_node_id, vec![1u8; 32]);
                assert_eq!(request.target_node_id, vec![2u8; 32]);
                assert_eq!(
                    request.model,
                    Some(crate::proto::node::OwnerControlModelRef {
                        canonical_model_ref: "org/model:file.gguf".into(),
                        instance_id: Some("instance-ensure".into()),
                    })
                );
            }
            _ => panic!("expected ensure-model typed variant"),
        }

        let drain_request = crate::proto::node::OwnerControlRequest {
            request_id: 204,
            drain_model: Some(crate::proto::node::OwnerControlDrainModelRequest {
                requester_node_id: vec![1u8; 32],
                target_node_id: vec![2u8; 32],
                model: Some(crate::proto::node::OwnerControlModelRef {
                    canonical_model_ref: "org/model:file.gguf".into(),
                    instance_id: Some("instance-drain".into()),
                }),
                drain_timeout_secs: None,
            }),
            ..make_lifecycle_base(204)
        };
        let drain_decoded = OwnedNodeCommand::decode(
            crate::proto::node::OwnerControlRequest::decode(
                drain_request.encode_to_vec().as_slice(),
            )
            .expect("drain command should decode"),
        )
        .expect("drain command should map");
        match drain_decoded {
            OwnedNodeCommand::DrainModel { request, .. } => {
                assert_eq!(request.requester_node_id, vec![1u8; 32]);
                assert_eq!(request.target_node_id, vec![2u8; 32]);
                assert_eq!(
                    request.model,
                    Some(crate::proto::node::OwnerControlModelRef {
                        canonical_model_ref: "org/model:file.gguf".into(),
                        instance_id: Some("instance-drain".into()),
                    })
                );
            }
            _ => panic!("expected drain-model typed variant"),
        }

        assert_eq!(load_request.request_id, 201);
        assert_eq!(unload_request.request_id, 202);
        assert_eq!(ensure_request.request_id, 203);
        assert_eq!(drain_request.request_id, 204);
    }

    #[test]
    fn owner_lifecycle_rejects_ambiguous_and_legacy_requests() {
        let empty = make_lifecycle_base(300);
        assert!(OwnedNodeCommand::decode(empty).is_none());

        let ambiguous = crate::proto::node::OwnerControlRequest {
            load_model: Some(crate::proto::node::OwnerControlLoadModelRequest {
                requester_node_id: vec![1u8; 32],
                target_node_id: vec![2u8; 32],
                model: Some(crate::proto::node::OwnerControlModelRef {
                    canonical_model_ref: "org/model:file.gguf".into(),
                    instance_id: Some("instance-ambiguous".into()),
                }),
                profile: None,
            }),
            unload_model: Some(crate::proto::node::OwnerControlUnloadModelRequest {
                requester_node_id: vec![1u8; 32],
                target_node_id: vec![2u8; 32],
                model: Some(crate::proto::node::OwnerControlModelRef {
                    canonical_model_ref: "org/model:file.gguf".into(),
                    instance_id: Some("instance-ambiguous-2".into()),
                }),
            }),
            request_id: 301,
            ..make_lifecycle_base(301)
        };
        assert!(
            ambiguous.validate_frame().is_err(),
            "multi-command envelopes must fail validation before command execution"
        );

        let malformed = vec![0x08, 0xff];
        let malformed_decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::proto::node::OwnerControlRequest::decode(malformed.as_slice())
        }));
        assert!(malformed_decoded.is_ok());
        assert!(malformed_decoded.expect("decode should not panic").is_err());
    }
}
