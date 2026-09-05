/// Compatibility namespace for tokenizer contracts.
///
/// New consumers should depend on `skippy-tokenizer` directly. Keeping this
/// re-export avoids breaking older protocol users while the contract moves
/// out of the wire-protocol crate.
pub mod tokenizer {
    pub use skippy_tokenizer::*;
}

pub mod binary;
pub mod proto {
    pub mod stage {
        include!(concat!(env!("OUT_DIR"), "/skippy.stage.v1.rs"));
    }
}

mod admission;
mod config;
mod messages;
mod validation;

pub use admission::{
    STAGE_ADMISSION_DESCRIPTOR_VERSION, StageAdmissionDescriptor, StageAdmissionProfile,
    StageAdmissionSidecar, StageAdmissionSidecarKind,
};
pub use config::{
    ActivationDType, ActivationDescriptor, ActivationLayout, FlashAttentionType, GlmDsaPolicy,
    LoadMode, PeerConfig, SplitMode, StageActivationCodec, StageConfig, StageDevice, StageIdentity,
    StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload, StageTopology, StageTopologyEntry,
};
pub use messages::{
    AckMessage, DecodeTokenMessage, ErrorMessage, FinalPrefillChunkMessage, MessageBase,
    MessageKind, PrefillChunkMessage, ReadyMessage, StageMessage, StateExportMessage,
    StateImportMessage, StopMessage, TokenReplyMessage,
};
pub use validation::{
    MAX_STAGE_FRAME_BYTES, MAX_VERIFY_WINDOW_PIPELINE_DEPTH, SCHEMA_VERSION, STAGE_ALPN_V2,
    STAGE_PROTOCOL_GENERATION, STAGE_STREAM_ARTIFACT_TRANSFER, STAGE_STREAM_CONTROL,
    STAGE_STREAM_TRANSPORT, STAGE_SUBPROTOCOL_FEATURE_ARTIFACT_TRANSFER,
    STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1, STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL,
    STAGE_SUBPROTOCOL_FEATURE_STAGE_GENERATION,
    STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V8, STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST,
    STAGE_SUBPROTOCOL_MAJOR, STAGE_SUBPROTOCOL_NAME, StageFrameError,
    validate_stage_admission_descriptor, validate_stage_artifact_transfer_request,
    validate_stage_artifact_transfer_response, validate_stage_control_request,
    validate_stage_control_response, validate_stage_transport_open,
};

#[cfg(test)]
mod tests {
    use prost::Message as _;

    #[derive(Clone, PartialEq, prost::Message)]
    struct LegacyStageControlRequest {
        #[prost(uint32, tag = "1")]
        r#gen: u32,
        #[prost(bytes = "vec", tag = "2")]
        requester_id: Vec<u8>,
        #[prost(oneof = "LegacyStageCommand", tags = "3")]
        command: Option<LegacyStageCommand>,
    }

    #[derive(Clone, PartialEq, prost::Oneof)]
    enum LegacyStageCommand {
        #[prost(message, tag = "3")]
        LoadStage(super::proto::stage::LoadStage),
    }

    use super::proto::stage::{
        CancelPrepareStage, GetLayerInventory, GetStageStatus, LayerInventory, LayerRange,
        LoadStage, PrepareStage, PrepareStageAccepted, SourceModelKind, SourceResolutionPolicy,
        StageActivationCodec, StageAdmissionDescriptor, StageAdmissionProfile,
        StageAdmissionSidecar, StageAdmissionSidecarKind, StageArtifactTransferRequest,
        StageArtifactTransferResponse, StageControlRequest, StageControlResponse, StageLoadMode,
        StagePreparationState, StagePreparationStatus, StageReady, StageRuntimeState, StageStatus,
        StageStatusAck, StageStatusList, StageStatusUpdate, StageTopologyStage, StageTransportOpen,
        StopStage, stage_control_request, stage_control_response,
    };

    fn admission(layer_start: u32, layer_end: u32) -> StageAdmissionDescriptor {
        StageAdmissionDescriptor {
            version: super::STAGE_ADMISSION_DESCRIPTOR_VERSION,
            package_id: format!("sha256:{}", "c7".repeat(32)),
            plan_id: format!("skippy-plan:v1:{}", "d8".repeat(32)),
            layer_start,
            layer_end,
            resident_tensor_ids: vec!["tensor-a".to_string(), "tensor-b".to_string()],
            sidecars: vec![StageAdmissionSidecar {
                kind: StageAdmissionSidecarKind::Mmproj as i32,
                artifact_id: "mmproj-a".to_string(),
                name: None,
            }],
            profiles: vec![StageAdmissionProfile {
                profile_id: "decode".to_string(),
                graph_identity: "graph-a".to_string(),
                profile_identity: "profile-a".to_string(),
                slice_identity: "slice-a".to_string(),
                source_snapshot_identity: "snapshot-a".to_string(),
                graph_configuration_id: "graph-config-a".to_string(),
                backend_id: "cpu".to_string(),
            }],
        }
    }

    #[test]
    fn activation_codec_default_is_bit_exact() {
        assert_eq!(
            super::StageActivationCodec::default(),
            super::StageActivationCodec::RawF32V1
        );
    }
    use super::{
        STAGE_PROTOCOL_GENERATION, STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V8,
        StageFrameError, validate_stage_admission_descriptor,
        validate_stage_artifact_transfer_request, validate_stage_artifact_transfer_response,
        validate_stage_control_request, validate_stage_control_response,
        validate_stage_transport_open,
    };

    #[test]
    fn stage_protocol_generation_feature_names_current_generation() {
        assert_eq!(
            STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V8,
            format!("stage-generation-{STAGE_PROTOCOL_GENERATION}")
        );
    }

    #[test]
    fn stage_admission_descriptor_is_fail_closed_and_canonical() {
        let descriptor = admission(8, 16);
        validate_stage_admission_descriptor(&descriptor).unwrap();

        let mut invalid = descriptor.clone();
        invalid.version = 0;
        assert!(validate_stage_admission_descriptor(&invalid).is_err());

        let mut invalid = descriptor.clone();
        invalid.package_id = format!("sha256:{}", "C7".repeat(32));
        assert!(validate_stage_admission_descriptor(&invalid).is_err());

        let mut invalid = descriptor.clone();
        invalid.plan_id = format!("sha256:{}", "d8".repeat(32));
        assert!(validate_stage_admission_descriptor(&invalid).is_err());

        let mut invalid = descriptor.clone();
        invalid.resident_tensor_ids.reverse();
        assert!(validate_stage_admission_descriptor(&invalid).is_err());

        let mut invalid = descriptor.clone();
        invalid.sidecars.push(invalid.sidecars[0].clone());
        assert!(validate_stage_admission_descriptor(&invalid).is_err());

        let mut invalid = descriptor.clone();
        invalid.profiles[0].graph_identity.clear();
        assert!(validate_stage_admission_descriptor(&invalid).is_err());

        let mut invalid = descriptor;
        let mut earlier = invalid.profiles[0].clone();
        earlier.profile_id = "batch".to_string();
        invalid.profiles.push(earlier);
        assert!(validate_stage_admission_descriptor(&invalid).is_err());
    }

    #[test]
    fn stage_control_request_validates_generation_sender_and_command() {
        let frame = StageControlRequest {
            r#gen: STAGE_PROTOCOL_GENERATION,
            requester_id: vec![9u8; 32],
            command: Some(stage_control_request::Command::GetStageStatus(
                GetStageStatus {
                    topology_id: Some("topology-a".to_string()),
                    run_id: Some("run-a".to_string()),
                    stage_id: Some("stage-0".to_string()),
                },
            )),
        };
        validate_stage_control_request(&frame).unwrap();

        let load = StageControlRequest {
            command: Some(stage_control_request::Command::LoadStage(LoadStage {
                topology_id: "topology-a".to_string(),
                run_id: "run-a".to_string(),
                model_id: "qwen".to_string(),
                backend: "skippy".to_string(),
                package_ref: "hf://repo/model".to_string(),
                manifest_sha256: "a5".repeat(32),
                stage_id: "stage-0".to_string(),
                layer_end: 16,
                admission: Some(admission(0, 16)),
                participant_set_hash: "participants".to_string(),
                topology_hash: "topology".to_string(),
                activation_codec: StageActivationCodec::F16RneV1 as i32,
                projector_path: Some("/models/mmproj.gguf".to_string()),
                source_model_sha256: Some("b6".repeat(32)),
                source_resolution_policy: SourceResolutionPolicy::Fallback as i32,
                runtime_profile: Some("strict-profile".to_string()),
                bind_addr: "127.0.0.1:9000".to_string(),
                topology_stages: vec![StageTopologyStage {
                    stage_id: "stage-0".to_string(),
                    stage_index: 0,
                    node_id: vec![7u8; 32],
                    layer_start: 0,
                    layer_end: 16,
                    bind_addr: "127.0.0.1:9000".to_string(),
                }],
                ..Default::default()
            })),
            ..frame.clone()
        };
        let decoded = StageControlRequest::decode(load.encode_to_vec().as_slice()).unwrap();
        match decoded.command {
            Some(stage_control_request::Command::LoadStage(load)) => {
                assert_eq!(load.projector_path.as_deref(), Some("/models/mmproj.gguf"));
                assert_eq!(
                    load.source_model_sha256.as_deref(),
                    Some("b6".repeat(32).as_str())
                );
                assert_eq!(
                    load.source_resolution_policy,
                    SourceResolutionPolicy::Fallback as i32
                );
                assert_eq!(load.runtime_profile.as_deref(), Some("strict-profile"));
            }
            other => panic!("expected LoadStage, got {other:?}"),
        }

        let mut missing_claim_hash = load.clone();
        let Some(stage_control_request::Command::LoadStage(load_stage)) =
            missing_claim_hash.command.as_mut()
        else {
            unreachable!("load fixture must contain LoadStage");
        };
        load_stage.topology_hash.clear();
        assert!(matches!(
            validate_stage_control_request(&missing_claim_hash),
            Err(StageFrameError::MissingLoadClaimHashes)
        ));

        let stop = StageControlRequest {
            command: Some(stage_control_request::Command::StopStage(StopStage {
                topology_id: "topology-a".to_string(),
                run_id: "run-a".to_string(),
                stage_id: "stage-0".to_string(),
                shutdown_generation: 7,
                coordinator_term: 7,
            })),
            ..frame.clone()
        };
        validate_stage_control_request(&stop).unwrap();

        let inventory = StageControlRequest {
            command: Some(stage_control_request::Command::GetLayerInventory(
                GetLayerInventory {
                    model_id: "qwen".to_string(),
                    package_ref: "hf://repo/model".to_string(),
                    manifest_sha256: "a5".repeat(32),
                    expected_source_model_sha256: Some("b6".repeat(32)),
                    source_resolution_policy: SourceResolutionPolicy::LocalRequired as i32,
                    runtime_profile: Some("strict-profile".to_string()),
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_request(&inventory).unwrap();

        let mut unknown_source_policy = load.clone();
        let Some(stage_control_request::Command::LoadStage(load_stage)) =
            unknown_source_policy.command.as_mut()
        else {
            unreachable!("load fixture must contain LoadStage");
        };
        load_stage.source_resolution_policy = 99;
        assert!(matches!(
            validate_stage_control_request(&unknown_source_policy),
            Err(StageFrameError::InvalidSourceResolutionPolicy { got: 99 })
        ));

        let mut strict_load_stage = match load.command.clone() {
            Some(stage_control_request::Command::LoadStage(load)) => load,
            _ => unreachable!("load fixture must contain LoadStage"),
        };
        strict_load_stage.package_ref = format!("local-gguf://sha256/{}", "b6".repeat(32));
        strict_load_stage.load_mode = StageLoadMode::RuntimeSlice as i32;
        let legacy_content_ref_load = StageControlRequest {
            command: Some(stage_control_request::Command::LoadStage(
                strict_load_stage.clone(),
            )),
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_request(&legacy_content_ref_load),
            Err(StageFrameError::LocalSourceCommandRequired)
        ));
        strict_load_stage.source_resolution_policy = SourceResolutionPolicy::LocalRequired as i32;
        let legacy_strict_load = StageControlRequest {
            command: Some(stage_control_request::Command::LoadStage(
                strict_load_stage.clone(),
            )),
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_request(&legacy_strict_load),
            Err(StageFrameError::LocalSourceCommandRequired)
        ));
        let strict_load = StageControlRequest {
            command: Some(stage_control_request::Command::LoadLocalStage(
                strict_load_stage.clone(),
            )),
            ..frame.clone()
        };
        validate_stage_control_request(&strict_load).unwrap();
        let mut malformed_reference = strict_load.clone();
        let Some(stage_control_request::Command::LoadLocalStage(load)) =
            malformed_reference.command.as_mut()
        else {
            unreachable!("strict fixture must contain LoadLocalStage")
        };
        load.package_ref = "local-gguf://sha256/not-a-digest".to_string();
        assert!(matches!(
            validate_stage_control_request(&malformed_reference),
            Err(StageFrameError::InvalidLocalSourceReference)
        ));
        let mut mismatched_reference = strict_load.clone();
        let Some(stage_control_request::Command::LoadLocalStage(load)) =
            mismatched_reference.command.as_mut()
        else {
            unreachable!("strict fixture must contain LoadLocalStage")
        };
        load.package_ref = format!("local-gguf://sha256/{}", "c7".repeat(32));
        assert!(matches!(
            validate_stage_control_request(&mismatched_reference),
            Err(StageFrameError::InvalidLocalSourceReference)
        ));
        for invalid_mode in [
            StageLoadMode::Unspecified as i32,
            StageLoadMode::LayerPackage as i32,
            StageLoadMode::ArtifactSlice as i32,
            99,
        ] {
            let mut invalid = strict_load.clone();
            let Some(stage_control_request::Command::LoadLocalStage(load)) =
                invalid.command.as_mut()
            else {
                unreachable!("strict fixture must contain LoadLocalStage")
            };
            load.load_mode = invalid_mode;
            assert!(matches!(
                validate_stage_control_request(&invalid),
                Err(StageFrameError::InvalidLocalSourceLoadMode { got }) if got == invalid_mode
            ));
        }

        let legacy_decoded =
            LegacyStageControlRequest::decode(strict_load.encode_to_vec().as_slice()).unwrap();
        assert!(legacy_decoded.command.is_none());

        let mut fallback_strict_load = strict_load_stage;
        fallback_strict_load.source_resolution_policy = SourceResolutionPolicy::Fallback as i32;
        let fallback_strict = StageControlRequest {
            command: Some(stage_control_request::Command::LoadLocalStage(
                fallback_strict_load,
            )),
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_request(&fallback_strict),
            Err(StageFrameError::LocalSourcePolicyRequired)
        ));

        let prepare = StageControlRequest {
            command: Some(stage_control_request::Command::PrepareStage(PrepareStage {
                load_stage: Some(LoadStage {
                    topology_id: "topology-a".to_string(),
                    run_id: "run-a".to_string(),
                    model_id: "qwen".to_string(),
                    backend: "skippy".to_string(),
                    package_ref: "gguf:///model.gguf".to_string(),
                    manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
                    stage_id: "stage-1".to_string(),
                    stage_index: 1,
                    layer_start: 8,
                    layer_end: 16,
                    admission: Some(admission(8, 16)),
                    participant_set_hash: "participants".to_string(),
                    topology_hash: "topology".to_string(),
                    activation_codec: StageActivationCodec::F16RneV1 as i32,
                    bind_addr: "127.0.0.1:9001".to_string(),
                    topology_stages: vec![
                        StageTopologyStage {
                            stage_id: "stage-0".to_string(),
                            stage_index: 0,
                            node_id: vec![7u8; 32],
                            layer_start: 0,
                            layer_end: 8,
                            bind_addr: "127.0.0.1:9000".to_string(),
                        },
                        StageTopologyStage {
                            stage_id: "stage-1".to_string(),
                            stage_index: 1,
                            node_id: vec![8u8; 32],
                            layer_start: 8,
                            layer_end: 16,
                            bind_addr: "127.0.0.1:9001".to_string(),
                        },
                    ],
                    ..Default::default()
                }),
                coordinator_id: Some(vec![8u8; 32]),
            })),
            ..frame.clone()
        };
        validate_stage_control_request(&prepare).unwrap();

        let mut legacy_content_ref_prepare = prepare.clone();
        let Some(stage_control_request::Command::PrepareStage(legacy_prepare_payload)) =
            legacy_content_ref_prepare.command.as_mut()
        else {
            unreachable!("prepare fixture must contain PrepareStage")
        };
        legacy_prepare_payload
            .load_stage
            .as_mut()
            .expect("prepare load")
            .package_ref = format!("local-gguf://sha256/{}", "b6".repeat(32));
        assert!(matches!(
            validate_stage_control_request(&legacy_content_ref_prepare),
            Err(StageFrameError::LocalSourceCommandRequired)
        ));

        let mut strict_prepare = prepare;
        let Some(stage_control_request::Command::PrepareStage(prepare)) =
            strict_prepare.command.as_mut()
        else {
            unreachable!("prepare fixture must contain PrepareStage")
        };
        prepare
            .load_stage
            .as_mut()
            .expect("prepare load")
            .source_resolution_policy = SourceResolutionPolicy::LocalRequired as i32;
        assert!(matches!(
            validate_stage_control_request(&strict_prepare),
            Err(StageFrameError::LocalSourceCommandRequired)
        ));

        let status_update = StageControlRequest {
            command: Some(stage_control_request::Command::StageStatusUpdate(
                StageStatusUpdate {
                    status: Some(StagePreparationStatus {
                        topology_id: "topology-a".to_string(),
                        run_id: "run-a".to_string(),
                        model_id: "qwen".to_string(),
                        backend: "skippy".to_string(),
                        package_ref: "gguf:///model.gguf".to_string(),
                        manifest_sha256: "direct-gguf:1:model.gguf".to_string(),
                        stage_id: "stage-1".to_string(),
                        stage_index: 1,
                        layer_start: 8,
                        layer_end: 16,
                        admission: Some(admission(8, 16)),
                        activation_codec: StageActivationCodec::F16RneV1 as i32,
                        state: StagePreparationState::Loading as i32,
                        bytes_done: Some(10),
                        bytes_total: Some(20),
                        shutdown_generation: 7,
                        ..Default::default()
                    }),
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_request(&status_update).unwrap();

        let missing_status_admission = StageControlRequest {
            command: Some(stage_control_request::Command::StageStatusUpdate(
                StageStatusUpdate {
                    status: Some(StagePreparationStatus {
                        topology_id: "topology-a".to_string(),
                        run_id: "run-a".to_string(),
                        model_id: "qwen".to_string(),
                        layer_start: 8,
                        layer_end: 16,
                        ..Default::default()
                    }),
                },
            )),
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_request(&missing_status_admission),
            Err(StageFrameError::MissingStageAdmissionDescriptor)
        ));

        let missing_prepare_load = StageControlRequest {
            command: Some(stage_control_request::Command::PrepareStage(PrepareStage {
                load_stage: None,
                coordinator_id: None,
            })),
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_request(&missing_prepare_load),
            Err(StageFrameError::MissingStageAdmissionDescriptor)
        ));

        let cancel = StageControlRequest {
            command: Some(stage_control_request::Command::CancelPrepareStage(
                CancelPrepareStage {
                    topology_id: "topology-a".to_string(),
                    run_id: "run-a".to_string(),
                    stage_id: "stage-1".to_string(),
                    shutdown_generation: 8,
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_request(&cancel).unwrap();

        let missing_command = StageControlRequest {
            command: None,
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_request(&missing_command),
            Err(StageFrameError::MissingStageControlCommand)
        ));

        let wrong_gen = StageControlRequest {
            r#gen: STAGE_PROTOCOL_GENERATION - 1,
            ..frame
        };
        assert!(matches!(
            validate_stage_control_request(&wrong_gen),
            Err(StageFrameError::BadGeneration { got }) if got == STAGE_PROTOCOL_GENERATION - 1
        ));
    }

    #[test]
    fn stage_control_response_validates_generation_and_response() {
        let frame = StageControlResponse {
            r#gen: STAGE_PROTOCOL_GENERATION,
            response: Some(stage_control_response::Response::StageReady(StageReady {
                accepted: true,
                status: Some(StageStatus {
                    topology_id: "topology-a".to_string(),
                    run_id: "run-a".to_string(),
                    model_id: "qwen".to_string(),
                    backend: "skippy".to_string(),
                    stage_id: "stage-0".to_string(),
                    stage_index: 0,
                    layer_start: 0,
                    layer_end: 16,
                    admission: Some(admission(0, 16)),
                    activation_codec: StageActivationCodec::F16RneV1 as i32,
                    state: StageRuntimeState::Ready as i32,
                    bind_addr: "127.0.0.1:0".to_string(),
                    shutdown_generation: 7,
                    ctx_size: 8192,
                    lane_count: 2,
                    projector_path: Some("/models/mmproj.gguf".to_string()),
                    ..Default::default()
                }),
                error: None,
            })),
        };
        let decoded = StageControlResponse::decode(frame.encode_to_vec().as_slice()).unwrap();
        validate_stage_control_response(&decoded).unwrap();
        match decoded.response {
            Some(stage_control_response::Response::StageReady(ready)) => {
                let status = ready.status.expect("stage-ready status");
                assert_eq!(
                    status.projector_path.as_deref(),
                    Some("/models/mmproj.gguf")
                );
                assert_eq!(status.lane_count, 2);
            }
            other => panic!("expected StageReady, got {other:?}"),
        }

        let inventory_response = StageControlResponse {
            response: Some(stage_control_response::Response::LayerInventory(
                LayerInventory {
                    model_id: "qwen".to_string(),
                    package_ref: "hf://repo/model".to_string(),
                    manifest_sha256: "a5".repeat(32),
                    layer_count: 16,
                    source_model_path: Some("/model.gguf".to_string()),
                    source_model_bytes: Some(1024),
                    source_model_kind: SourceModelKind::PlainGguf as i32,
                    source_model_sha256: Some("b6".repeat(32)),
                    content_addressed_local_source: Some(true),
                    ready_ranges: vec![LayerRange {
                        layer_start: 0,
                        layer_end: 8,
                    }],
                    ..Default::default()
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_response(&inventory_response).unwrap();

        let prepare_response = StageControlResponse {
            response: Some(stage_control_response::Response::PrepareStageAccepted(
                PrepareStageAccepted {
                    accepted: true,
                    status: Some(StagePreparationStatus {
                        topology_id: "topology-a".to_string(),
                        run_id: "run-a".to_string(),
                        model_id: "qwen".to_string(),
                        backend: "skippy".to_string(),
                        package_ref: "hf://repo/model".to_string(),
                        manifest_sha256: "a5".repeat(32),
                        stage_id: "stage-1".to_string(),
                        stage_index: 1,
                        layer_start: 8,
                        layer_end: 16,
                        admission: Some(admission(8, 16)),
                        activation_codec: StageActivationCodec::F16RneV1 as i32,
                        state: StagePreparationState::Assigned as i32,
                        shutdown_generation: 7,
                        ..Default::default()
                    }),
                    error: None,
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_response(&prepare_response).unwrap();

        let ack_response = StageControlResponse {
            response: Some(stage_control_response::Response::StageStatusAck(
                StageStatusAck {
                    accepted: true,
                    error: None,
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_response(&ack_response).unwrap();

        let status_list_response = StageControlResponse {
            response: Some(stage_control_response::Response::StageStatuses(
                StageStatusList {
                    statuses: vec![StageStatus {
                        topology_id: "topology-a".to_string(),
                        run_id: "run-a".to_string(),
                        model_id: "qwen".to_string(),
                        backend: "skippy".to_string(),
                        stage_id: "stage-0".to_string(),
                        stage_index: 0,
                        layer_start: 0,
                        layer_end: 16,
                        admission: Some(admission(0, 16)),
                        activation_codec: StageActivationCodec::F16RneV1 as i32,
                        state: StageRuntimeState::Ready as i32,
                        bind_addr: "127.0.0.1:51234".to_string(),
                        shutdown_generation: 7,
                        ctx_size: 8192,
                        lane_count: 2,
                        ..Default::default()
                    }],
                },
            )),
            ..frame.clone()
        };
        validate_stage_control_response(&status_list_response).unwrap();

        let missing_response = StageControlResponse {
            response: None,
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_control_response(&missing_response),
            Err(StageFrameError::MissingStageControlResponse)
        ));

        let wrong_gen = StageControlResponse { r#gen: 1, ..frame };
        assert!(matches!(
            validate_stage_control_response(&wrong_gen),
            Err(StageFrameError::BadGeneration { got: 1 })
        ));
    }

    #[test]
    fn stage_transport_open_validates_generation_sender_and_target() {
        let frame = StageTransportOpen {
            r#gen: STAGE_PROTOCOL_GENERATION,
            requester_id: vec![7u8; 32],
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            stage_id: "stage-1".to_string(),
        };
        validate_stage_transport_open(&frame).unwrap();

        let missing_target = StageTransportOpen {
            stage_id: String::new(),
            ..frame.clone()
        };
        assert!(matches!(
            validate_stage_transport_open(&missing_target),
            Err(StageFrameError::MissingStageTransportTarget)
        ));

        let wrong_gen = StageTransportOpen { r#gen: 1, ..frame };
        assert!(matches!(
            validate_stage_transport_open(&wrong_gen),
            Err(StageFrameError::BadGeneration { got: 1 })
        ));
    }

    #[test]
    fn stage_artifact_transfer_frames_validate_skippy_owned_contract() {
        let request = StageArtifactTransferRequest {
            r#gen: STAGE_PROTOCOL_GENERATION,
            requester_id: vec![7u8; 32],
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            stage_id: "stage-0".to_string(),
            package_ref: "hf://meshllm/demo-layers@abc123".to_string(),
            manifest_sha256: "a".repeat(64),
            relative_path: "layers/layer-000.gguf".to_string(),
            offset: 0,
            expected_size: Some(8),
            expected_sha256: Some("b".repeat(64)),
        };
        let decoded =
            StageArtifactTransferRequest::decode(request.encode_to_vec().as_slice()).unwrap();
        validate_stage_artifact_transfer_request(&decoded).unwrap();
        assert_eq!(decoded.stage_id, "stage-0");

        let mut unsafe_path = request.clone();
        unsafe_path.relative_path = "../layer.gguf".to_string();
        assert!(matches!(
            validate_stage_artifact_transfer_request(&unsafe_path),
            Err(StageFrameError::InvalidArtifactPath)
        ));

        let mut bad_offset = request.clone();
        bad_offset.offset = 9;
        assert!(matches!(
            validate_stage_artifact_transfer_request(&bad_offset),
            Err(StageFrameError::InvalidArtifactOffset)
        ));

        let mut missing_target = request.clone();
        missing_target.topology_id.clear();
        assert!(matches!(
            validate_stage_artifact_transfer_request(&missing_target),
            Err(StageFrameError::MissingStageArtifactTarget)
        ));

        let response = StageArtifactTransferResponse {
            r#gen: STAGE_PROTOCOL_GENERATION,
            accepted: true,
            total_size: 8,
            sha256: Some("b".repeat(64)),
            error: None,
        };
        let decoded =
            StageArtifactTransferResponse::decode(response.encode_to_vec().as_slice()).unwrap();
        validate_stage_artifact_transfer_response(&decoded).unwrap();

        let bad_response_sha = StageArtifactTransferResponse {
            sha256: Some("not-a-sha".to_string()),
            ..response
        };
        assert!(matches!(
            validate_stage_artifact_transfer_response(&bad_response_sha),
            Err(StageFrameError::InvalidArtifactDigestLength { .. })
        ));
    }
}
