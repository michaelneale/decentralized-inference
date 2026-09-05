use std::collections::HashMap;

use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig, StageConfig, StageDevice};

use super::family_policy::FamilyPolicy;
use super::materialization::StagePackageInfo;
use super::topology::MeshStagePlan;
use super::{
    KvCachePolicy, StageLoadRequest, StageLoadRuntimeSettings, StagePeerDescriptor,
    StageStatusSnapshot, StageStopRequest,
};
use crate::mesh;

pub(crate) struct StageDeploymentContext<'a> {
    pub(crate) topology_id: &'a str,
    pub(crate) run_id: &'a str,
    pub(crate) model_id: &'a str,
    pub(crate) package: &'a StagePackageInfo,
    pub(crate) admission: skippy_protocol::StageAdmissionDescriptor,
    pub(crate) participant_set_hash: &'a str,
    pub(crate) topology_hash: &'a str,
    pub(crate) family_policy: &'a FamilyPolicy,
    pub(crate) ctx_size: u32,
    pub(crate) lane_count: u32,
    pub(crate) continuous_batching: bool,
    pub(crate) n_batch: Option<u32>,
    pub(crate) n_ubatch: Option<u32>,
    pub(crate) kv_cache: KvCachePolicy,
    pub(crate) flash_attn_type: FlashAttentionType,
    pub(crate) kv_offload: Option<bool>,
    pub(crate) kv_unified: Option<bool>,
    pub(crate) swa_full: Option<bool>,
    pub(crate) mmap: Option<bool>,
    pub(crate) mlock: bool,
    pub(crate) projector_path: Option<String>,
    pub(crate) native_mtp_enabled: bool,
    pub(crate) runtime_settings: StageLoadRuntimeSettings,
}

pub(crate) fn remote_stage_load_request(
    context: &StageDeploymentContext<'_>,
    stage: &MeshStagePlan,
    downstream: Option<StagePeerDescriptor>,
) -> StageLoadRequest {
    StageLoadRequest {
        topology_id: context.topology_id.to_string(),
        run_id: context.run_id.to_string(),
        model_id: context.model_id.to_string(),
        runtime_profile: Some(String::new()),
        backend: "skippy".to_string(),
        package_ref: context.package.package_ref.clone(),
        manifest_sha256: context.package.manifest_sha256.clone(),
        stage_id: stage.stage_id.clone(),
        stage_index: stage.stage_index,
        layer_start: stage.layer_start,
        layer_end: stage.layer_end,
        admission: context.admission.clone(),
        participant_set_hash: context.participant_set_hash.to_string(),
        topology_hash: context.topology_hash.to_string(),
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        model_path: Some(context.package.package_ref.clone()),
        source_model_bytes: context.package.source_model_bytes,
        source_model_sha256: Some(context.package.source_model_sha256.clone()),
        local_source_required: false,
        projector_path: None,
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
        generation_signal_window: None,
        selected_device: None,
        bind_addr: "127.0.0.1:0".to_string(),
        ctx_size: context.ctx_size,
        lane_count: context.lane_count,
        continuous_batching: context.continuous_batching,
        n_batch: context.n_batch,
        n_ubatch: context.n_ubatch,
        n_gpu_layers: -1,
        mmap: context.mmap,
        mlock: context.mlock,
        cache_type_k: context.kv_cache.cache_type_k().to_string(),
        cache_type_v: context.kv_cache.cache_type_v().to_string(),
        flash_attn_type: context.flash_attn_type,
        runtime_settings: context.runtime_settings,
        native_mtp_enabled: context.native_mtp_enabled,
        shutdown_generation: 1,
        coordinator_term: 0,
        coordinator_id: None,
        lease_until_unix_ms: 0,
        load_mode: LoadMode::LayerPackage,
        upstream: None,
        downstream,
    }
}

pub(crate) fn stage0_config(
    context: &StageDeploymentContext<'_>,
    stage0: &MeshStagePlan,
    downstream_stage: &MeshStagePlan,
    downstream_endpoint: String,
    selected_device: Option<StageDevice>,
) -> StageConfig {
    let mut config = StageConfig {
        run_id: context.run_id.to_string(),
        topology_id: context.topology_id.to_string(),
        model_id: context.model_id.to_string(),
        package_ref: Some(context.package.package_ref.clone()),
        manifest_sha256: Some(context.package.manifest_sha256.clone()),
        source_model_path: Some(context.package.source_model_path.clone()),
        source_model_sha256: Some(context.package.source_model_sha256.clone()),
        source_model_bytes: context.package.source_model_bytes,
        materialized_path: None,
        materialized_pinned: false,
        model_path: Some(context.package.package_ref.clone()),
        projector_path: context
            .projector_path
            .clone()
            .or_else(|| context.package.projector_path.clone()),
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
        generation_signal_window: None,
        activation_codec: skippy_protocol::StageActivationCodec::default(),
        stage_id: stage0.stage_id.clone(),
        stage_index: stage0.stage_index,
        layer_start: stage0.layer_start,
        layer_end: stage0.layer_end,
        ctx_size: context.ctx_size,
        lane_count: context.lane_count,
        n_batch: context.n_batch,
        n_ubatch: context.n_ubatch,
        n_gpu_layers: -1,
        mmap: context.mmap,
        mlock: context.mlock,
        repack: context.runtime_settings.repack,
        op_offload: context.runtime_settings.op_offload,
        no_host_buffer: context.runtime_settings.no_host_buffer,
        check_tensors: context.runtime_settings.check_tensors,
        direct_io: context.runtime_settings.direct_io,
        main_gpu: context.runtime_settings.main_gpu,
        split_mode: context.runtime_settings.split_mode,
        cache_type_k: context.kv_cache.cache_type_k().to_string(),
        cache_type_v: context.kv_cache.cache_type_v().to_string(),
        flash_attn_type: context.flash_attn_type,
        kv_offload: context.runtime_settings.kv_offload,
        kv_unified: context.runtime_settings.kv_unified,
        swa_full: context.runtime_settings.swa_full,
        cache_idle_slots: context.runtime_settings.cache_idle_slots,
        filter_tensors_on_load: true,
        checkpoint_quantization: None,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        selected_device,
        kv_cache: None,
        native_mtp_enabled: context.native_mtp_enabled,
        load_mode: LoadMode::LayerPackage,
        bind_addr: "127.0.0.1:0".to_string(),
        upstream: None,
        downstream: Some(PeerConfig {
            stage_id: downstream_stage.stage_id.clone(),
            stage_index: downstream_stage.stage_index,
            endpoint: downstream_endpoint,
        }),
    };
    config.kv_cache = context
        .family_policy
        .stage_kv_cache_config_for_package(&config, &context.package.package_dir);
    config
}

pub(crate) fn stage_stop_request(
    context: &StageDeploymentContext<'_>,
    stage: &MeshStagePlan,
    shutdown_generation: u64,
) -> StageStopRequest {
    StageStopRequest {
        topology_id: context.topology_id.to_string(),
        run_id: context.run_id.to_string(),
        stage_id: stage.stage_id.clone(),
        shutdown_generation,
        coordinator_term: 0,
    }
}

pub(crate) fn stage_topology_instance(
    context: &StageDeploymentContext<'_>,
    stages: &[MeshStagePlan],
    ready_statuses: &HashMap<String, StageStatusSnapshot>,
    stage0_bind_addr: String,
) -> mesh::StageTopologyInstance {
    mesh::StageTopologyInstance {
        topology_id: context.topology_id.to_string(),
        run_id: context.run_id.to_string(),
        model_id: context.model_id.to_string(),
        package_ref: context.package.package_ref.clone(),
        manifest_sha256: context.package.manifest_sha256.clone(),
        stages: stages
            .iter()
            .map(|stage| mesh::StageAssignment {
                stage_id: stage.stage_id.clone(),
                stage_index: stage.stage_index,
                node_id: stage.node_id,
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                endpoint: mesh::StageEndpoint {
                    bind_addr: ready_statuses
                        .get(&stage.stage_id)
                        .map(|status| status.bind_addr.clone())
                        .unwrap_or_else(|| stage0_bind_addr.clone()),
                },
            })
            .collect(),
    }
}

pub(crate) fn pinned_stage_device(
    pinned_gpu: Option<&crate::runtime::StartupPinnedGpuTarget>,
) -> Option<StageDevice> {
    pinned_gpu.map(|gpu| StageDevice {
        backend_device: gpu.backend_device.clone(),
        stable_id: Some(gpu.stable_id.clone()),
        index: Some(gpu.index),
        vram_bytes: Some(gpu.vram_bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::skippy::materialization::StagePackageLayerInfo;
    use iroh::SecretKey;
    use std::path::PathBuf;

    fn make_id(seed: u8) -> iroh::EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SecretKey::from_bytes(&bytes).public()
    }

    fn package() -> StagePackageInfo {
        StagePackageInfo {
            package_ref: "hf://Mesh-LLM/demo-package".to_string(),
            package_dir: PathBuf::from("/tmp/package"),
            manifest_sha256: "manifest".to_string(),
            model_id: "model".to_string(),
            source_model_path: "model.gguf".to_string(),
            source_model_sha256: "source".to_string(),
            source_model_bytes: Some(100),
            layer_count: 4,
            activation_width: 0,
            generation: None,
            projector_path: Some("/tmp/package/projectors/mmproj.gguf".to_string()),
            layers: vec![StagePackageLayerInfo {
                layer_index: 0,
                tensor_count: 1,
                tensor_bytes: 10,
                artifact_bytes: 12,
            }],
        }
    }

    #[test]
    fn remote_load_request_uses_package_identity_and_layer_mode() {
        let package = package();
        let context = StageDeploymentContext {
            topology_id: "topology-a",
            run_id: "run-a",
            model_id: "model-a",
            package: &package,
            admission: crate::inference::skippy::test_stage_admission(4, 8),
            participant_set_hash: "participants",
            topology_hash: "topology-hash",
            family_policy: &crate::inference::skippy::family_policy::family_policy_for_model_path(
                "model.gguf",
            ),
            ctx_size: 8192,
            lane_count: 2,
            continuous_batching: true,
            n_batch: None,
            n_ubatch: None,
            kv_cache: KvCachePolicy::for_model_size(0),
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            mmap: Some(false),
            mlock: true,
            projector_path: Some("/models/mmproj.gguf".to_string()),
            native_mtp_enabled: false,
            runtime_settings: StageLoadRuntimeSettings {
                repack: true,
                op_offload: Some(false),
                no_host_buffer: true,
                check_tensors: true,
                direct_io: true,
                main_gpu: Some(2),
                split_mode: skippy_protocol::SplitMode::Row,
                kv_offload: Some(false),
                kv_unified: Some(true),
                swa_full: Some(false),
                cache_idle_slots: Some(3),
            },
        };
        let request = remote_stage_load_request(
            &context,
            &MeshStagePlan {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: make_id(1),
                layer_start: 4,
                layer_end: 8,
                parameter_bytes: 50,
            },
            None,
        );

        assert_eq!(request.package_ref, "hf://Mesh-LLM/demo-package");
        assert_eq!(request.manifest_sha256, "manifest");
        assert_eq!(request.load_mode, LoadMode::LayerPackage);
        assert_eq!(request.source_model_bytes, Some(100));
        assert_eq!(
            request.model_path.as_deref(),
            Some("hf://Mesh-LLM/demo-package")
        );
        assert_eq!((request.layer_start, request.layer_end), (4, 8));
        assert!(request.projector_path.is_none());
        assert!(!request.native_mtp_enabled);
        assert_eq!(request.runtime_settings, context.runtime_settings);

        let stage0 = stage0_config(
            &context,
            &MeshStagePlan {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: make_id(0),
                layer_start: 0,
                layer_end: 4,
                parameter_bytes: 50,
            },
            &MeshStagePlan {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: make_id(1),
                layer_start: 4,
                layer_end: 8,
                parameter_bytes: 50,
            },
            "127.0.0.1:9001".to_string(),
            None,
        );
        assert_eq!(
            stage0.projector_path.as_deref(),
            Some("/models/mmproj.gguf")
        );
        assert!(!stage0.native_mtp_enabled);
        assert_eq!(stage0.kv_offload, context.runtime_settings.kv_offload);
        assert_eq!(stage0.kv_unified, context.runtime_settings.kv_unified);
        assert_eq!(stage0.swa_full, context.runtime_settings.swa_full);
    }
}
