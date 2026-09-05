#[cfg(test)]
use crate::mesh::RouteEntry;
use crate::mesh::{ModelDemand, NodeRole, PeerAnnouncement, RoutingTable};
use crate::protocol::NODE_PROTOCOL_GENERATION;
pub(crate) use crate::protocol::config_diagnostic::{
    config_diagnostic_to_proto, proto_config_diagnostic_to_local,
};
use anyhow::{Context, Result};
use iroh::{EndpointAddr, EndpointId};
use std::collections::{HashMap, HashSet};

fn skippy_stage_subprotocols(
    artifact_transfer_supported: bool,
    stage_protocol_generation_supported: bool,
    status_list_supported: bool,
    local_gguf_content_id_supported: bool,
) -> Vec<crate::proto::node::MeshSubprotocol> {
    let mut features = vec![skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL.to_string()];
    if local_gguf_content_id_supported {
        features
            .push(skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1.to_string());
    }
    if stage_protocol_generation_supported {
        features.push(
            skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V8.to_string(),
        );
    }
    if artifact_transfer_supported {
        features.push(skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_ARTIFACT_TRANSFER.to_string());
    }
    if status_list_supported {
        features.push(skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST.to_string());
    }
    vec![crate::proto::node::MeshSubprotocol {
        name: skippy_protocol::STAGE_SUBPROTOCOL_NAME.to_string(),
        major: skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
        features,
    }]
}

fn supports_skippy_artifact_transfer(subprotocols: &[crate::proto::node::MeshSubprotocol]) -> bool {
    supports_skippy_stage_feature(
        subprotocols,
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_ARTIFACT_TRANSFER,
    )
}

fn supports_skippy_status_list(subprotocols: &[crate::proto::node::MeshSubprotocol]) -> bool {
    supports_skippy_stage_feature(
        subprotocols,
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST,
    )
}

fn supports_local_gguf_content_id(subprotocols: &[crate::proto::node::MeshSubprotocol]) -> bool {
    supports_skippy_stage_feature(
        subprotocols,
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1,
    )
}

fn supports_skippy_stage_generation(subprotocols: &[crate::proto::node::MeshSubprotocol]) -> bool {
    let required_features = [
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_PROTOCOL_GENERATION_V8,
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STAGE_CONTROL,
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_STATUS_LIST,
        skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_LOCAL_GGUF_CONTENT_ID_V1,
    ];
    subprotocols.iter().any(|subprotocol| {
        subprotocol.name == skippy_protocol::STAGE_SUBPROTOCOL_NAME
            && subprotocol.major == skippy_protocol::STAGE_SUBPROTOCOL_MAJOR
            && required_features.iter().all(|required| {
                subprotocol
                    .features
                    .iter()
                    .any(|feature| feature == required)
            })
    })
}

fn supports_skippy_stage_feature(
    subprotocols: &[crate::proto::node::MeshSubprotocol],
    expected_feature: &str,
) -> bool {
    subprotocols.iter().any(|subprotocol| {
        subprotocol.name == skippy_protocol::STAGE_SUBPROTOCOL_NAME
            && subprotocol.major == skippy_protocol::STAGE_SUBPROTOCOL_MAJOR
            && subprotocol
                .features
                .iter()
                .any(|feature| feature == expected_feature)
    })
}

fn split_optional_csv(values: Option<&str>) -> Vec<Option<String>> {
    values
        .map(|values| {
            values
                .split(',')
                .map(|value| {
                    let value = value.trim();
                    (!value.is_empty()).then(|| value.to_string())
                })
                .collect()
        })
        .unwrap_or_default()
}

fn join_optional_csv(values: &[Option<String>]) -> Option<String> {
    if values.is_empty() {
        return None;
    }

    let has_present_value = values.iter().any(|value| {
        value
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
    });

    if !has_present_value {
        return None;
    }

    Some(
        values
            .iter()
            .map(|value| value.clone().unwrap_or_default())
            .collect::<Vec<_>>()
            .join(","),
    )
}

fn local_owner_attestation_to_proto(
    attestation: &crate::crypto::SignedNodeOwnership,
) -> Option<crate::proto::node::SignedNodeOwnership> {
    let owner_sign_public_key = decode_local_owner_attestation_hex(
        "owner_sign_public_key",
        &attestation.claim.owner_sign_public_key,
    )?;
    let node_endpoint_id = decode_local_owner_attestation_hex(
        "node_endpoint_id",
        &attestation.claim.node_endpoint_id,
    )?;
    let signature = decode_local_owner_attestation_hex("signature", &attestation.signature)?;
    Some(crate::proto::node::SignedNodeOwnership {
        version: attestation.claim.version,
        cert_id: attestation.claim.cert_id.clone(),
        owner_id: attestation.claim.owner_id.clone(),
        owner_sign_public_key,
        node_endpoint_id,
        issued_at_unix_ms: attestation.claim.issued_at_unix_ms,
        expires_at_unix_ms: attestation.claim.expires_at_unix_ms,
        node_label: attestation.claim.node_label.clone(),
        hostname_hint: attestation.claim.hostname_hint.clone(),
        signature,
    })
}

fn decode_local_owner_attestation_hex(field_name: &str, value: &str) -> Option<Vec<u8>> {
    match hex::decode(value) {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            tracing::warn!(
                "dropping local owner attestation from gossip: invalid {field_name} hex: {err}"
            );
            None
        }
    }
}

fn proto_owner_attestation_to_local(
    attestation: &crate::proto::node::SignedNodeOwnership,
) -> crate::crypto::SignedNodeOwnership {
    crate::crypto::SignedNodeOwnership {
        claim: crate::crypto::NodeOwnershipClaim {
            version: attestation.version,
            cert_id: attestation.cert_id.clone(),
            owner_id: attestation.owner_id.clone(),
            owner_sign_public_key: hex::encode(&attestation.owner_sign_public_key),
            node_endpoint_id: hex::encode(&attestation.node_endpoint_id),
            issued_at_unix_ms: attestation.issued_at_unix_ms,
            expires_at_unix_ms: attestation.expires_at_unix_ms,
            node_label: attestation.node_label.clone(),
            hostname_hint: attestation.hostname_hint.clone(),
        },
        signature: hex::encode(&attestation.signature),
    }
}

fn proto_release_attestation_to_local(
    attestation: &crate::proto::node::ReleaseBuildAttestation,
) -> crate::ReleaseBuildAttestation {
    crate::ReleaseBuildAttestation {
        version: attestation.version,
        node_version: attestation.node_version.clone(),
        build_id: attestation.build_id.clone(),
        commit: attestation.commit.clone(),
        target_triple: attestation.target_triple.clone(),
        supported_protocol_generation_min: attestation.supported_protocol_generation_min,
        supported_protocol_generation_max: attestation.supported_protocol_generation_max,
        artifact_digest: attestation.artifact_digest.clone(),
        signer_key_id: attestation.signer_key_id.clone(),
        signature_algorithm: attestation.signature_algorithm.clone(),
        signature: attestation.signature.clone(),
    }
}

fn local_source_kind_to_proto(kind: crate::mesh::ModelSourceKind) -> i32 {
    match kind {
        crate::mesh::ModelSourceKind::Catalog => {
            crate::proto::node::ModelSourceKind::Catalog as i32
        }
        crate::mesh::ModelSourceKind::HuggingFace => {
            crate::proto::node::ModelSourceKind::HuggingFace as i32
        }
        crate::mesh::ModelSourceKind::LocalGguf => {
            crate::proto::node::ModelSourceKind::LocalGguf as i32
        }
        crate::mesh::ModelSourceKind::DirectUrl => {
            crate::proto::node::ModelSourceKind::DirectUrl as i32
        }
        crate::mesh::ModelSourceKind::Unknown => {
            crate::proto::node::ModelSourceKind::Unknown as i32
        }
    }
}

fn proto_source_kind_to_local(kind: i32) -> crate::mesh::ModelSourceKind {
    match crate::proto::node::ModelSourceKind::try_from(kind)
        .unwrap_or(crate::proto::node::ModelSourceKind::Unknown)
    {
        crate::proto::node::ModelSourceKind::Catalog => crate::mesh::ModelSourceKind::Catalog,
        crate::proto::node::ModelSourceKind::HuggingFace => {
            crate::mesh::ModelSourceKind::HuggingFace
        }
        crate::proto::node::ModelSourceKind::LocalGguf => crate::mesh::ModelSourceKind::LocalGguf,
        crate::proto::node::ModelSourceKind::DirectUrl => crate::mesh::ModelSourceKind::DirectUrl,
        crate::proto::node::ModelSourceKind::Unknown
        | crate::proto::node::ModelSourceKind::Unspecified => crate::mesh::ModelSourceKind::Unknown,
    }
}

fn local_capability_level_to_proto(level: crate::models::CapabilityLevel) -> i32 {
    match level {
        crate::models::CapabilityLevel::None => crate::proto::node::CapabilityLevel::None as i32,
        crate::models::CapabilityLevel::Likely => {
            crate::proto::node::CapabilityLevel::Likely as i32
        }
        crate::models::CapabilityLevel::Supported => {
            crate::proto::node::CapabilityLevel::Supported as i32
        }
    }
}

fn proto_capability_level_to_local(level: i32) -> crate::models::CapabilityLevel {
    match crate::proto::node::CapabilityLevel::try_from(level)
        .unwrap_or(crate::proto::node::CapabilityLevel::None)
    {
        crate::proto::node::CapabilityLevel::Likely => crate::models::CapabilityLevel::Likely,
        crate::proto::node::CapabilityLevel::Supported => crate::models::CapabilityLevel::Supported,
        crate::proto::node::CapabilityLevel::None
        | crate::proto::node::CapabilityLevel::Unspecified => crate::models::CapabilityLevel::None,
    }
}

fn descriptor_identity_to_proto(
    identity: &crate::mesh::ServedModelIdentity,
) -> crate::proto::node::ServedModelIdentity {
    crate::proto::node::ServedModelIdentity {
        model_name: identity.model_name.clone(),
        is_primary: identity.is_primary,
        source_kind: local_source_kind_to_proto(identity.source_kind),
        canonical_ref: identity.canonical_ref.clone(),
        repository: identity.repository.clone(),
        revision: identity.revision.clone(),
        artifact: identity.artifact.clone(),
        local_file_name: identity.local_file_name.clone(),
        identity_hash: identity.identity_hash.clone(),
        // `weights_digest` deliberately does NOT cross the gossip wire: it is
        // a hash of file bytes only THIS node can read, so a peer receiving
        // it over gossip could never verify it against anything -- it would
        // be a claim, not a locally-checkable fact. It rides the
        // `openai.exchange.v1` serving-provenance event instead, straight to
        // whoever actually served the exchange.
    }
}

fn proto_identity_to_local(
    identity: &crate::proto::node::ServedModelIdentity,
) -> crate::mesh::ServedModelIdentity {
    crate::mesh::ServedModelIdentity {
        model_name: identity.model_name.clone(),
        is_primary: identity.is_primary,
        source_kind: proto_source_kind_to_local(identity.source_kind),
        canonical_ref: identity.canonical_ref.clone(),
        repository: identity.repository.clone(),
        revision: identity.revision.clone(),
        artifact: identity.artifact.clone(),
        local_file_name: identity.local_file_name.clone(),
        identity_hash: identity.identity_hash.clone(),
        // Never carried on the gossip wire (see `descriptor_identity_to_proto`)
        // -- a descriptor rebuilt from a peer's proto message has no
        // load-time weights digest to report, honestly `None`.
        weights_digest: None,
    }
}

fn legacy_descriptor_from_identity(
    identity: &crate::proto::node::ServedModelIdentity,
) -> crate::mesh::ServedModelDescriptor {
    crate::mesh::ServedModelDescriptor {
        identity: proto_identity_to_local(identity),
        capabilities_known: false,
        capabilities: crate::models::ModelCapabilities::default(),
        topology: None,
        metadata: None,
    }
}

fn local_model_metadata_to_proto(
    metadata: &crate::mesh::ServedModelMetadata,
) -> crate::proto::node::ServedModelMetadata {
    crate::proto::node::ServedModelMetadata {
        architecture: metadata.architecture.clone(),
        parameter_size: metadata.parameter_size.clone(),
        parameter_count_b: metadata.parameter_count_b,
        quant: metadata.quant.clone(),
        native_context_length: metadata.native_context_length,
        tokenizer: metadata.tokenizer.clone(),
        layer_count: metadata.layer_count,
        embedding_size: metadata.embedding_size,
        head_count: metadata.head_count,
        kv_head_count: metadata.kv_head_count,
        expert_count: metadata.expert_count,
        active_expert_count: metadata.active_expert_count,
    }
}

fn proto_model_metadata_to_local(
    metadata: &crate::proto::node::ServedModelMetadata,
) -> crate::mesh::ServedModelMetadata {
    crate::mesh::ServedModelMetadata {
        architecture: metadata.architecture.clone(),
        parameter_size: metadata.parameter_size.clone(),
        parameter_count_b: metadata.parameter_count_b,
        quant: metadata.quant.clone(),
        native_context_length: metadata.native_context_length,
        tokenizer: metadata.tokenizer.clone(),
        layer_count: metadata.layer_count,
        embedding_size: metadata.embedding_size,
        head_count: metadata.head_count,
        kv_head_count: metadata.kv_head_count,
        expert_count: metadata.expert_count,
        active_expert_count: metadata.active_expert_count,
    }
}

fn runtime_descriptor_to_proto(
    descriptor: &crate::mesh::ModelRuntimeDescriptor,
) -> crate::proto::node::ModelRuntimeDescriptor {
    crate::proto::node::ModelRuntimeDescriptor {
        model_name: descriptor.model_name.clone(),
        identity_hash: descriptor.identity_hash.clone(),
        context_length: descriptor.context_length,
        ready: descriptor.ready,
    }
}

fn proto_runtime_descriptor_to_local(
    descriptor: &crate::proto::node::ModelRuntimeDescriptor,
) -> crate::mesh::ModelRuntimeDescriptor {
    crate::mesh::ModelRuntimeDescriptor {
        model_name: descriptor.model_name.clone(),
        identity_hash: descriptor.identity_hash.clone(),
        context_length: descriptor.context_length,
        ready: descriptor.ready,
    }
}

fn local_gpu_info_to_proto(ann: &PeerAnnouncement) -> Vec<crate::proto::node::GpuInfo> {
    let legacy_field_count = [
        split_optional_csv(ann.gpu_vram.as_deref()).len(),
        split_optional_csv(ann.gpu_reserved_bytes.as_deref()).len(),
        split_optional_csv(ann.gpu_mem_bandwidth_gbps.as_deref()).len(),
        split_optional_csv(ann.gpu_compute_tflops_fp32.as_deref()).len(),
        split_optional_csv(ann.gpu_compute_tflops_fp16.as_deref()).len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let names =
        crate::system::hardware::expand_gpu_names(ann.gpu_name.as_deref(), legacy_field_count);
    let vram = split_optional_csv(ann.gpu_vram.as_deref());
    let reserved = split_optional_csv(ann.gpu_reserved_bytes.as_deref());
    let mem_bandwidth = split_optional_csv(ann.gpu_mem_bandwidth_gbps.as_deref());
    let fp32 = split_optional_csv(ann.gpu_compute_tflops_fp32.as_deref());
    let fp16 = split_optional_csv(ann.gpu_compute_tflops_fp16.as_deref());
    let count = [
        legacy_field_count,
        names.len(),
        vram.len(),
        reserved.len(),
        mem_bandwidth.len(),
        fp32.len(),
        fp16.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);

    (0..count)
        .map(|index| crate::proto::node::GpuInfo {
            name: names.get(index).cloned(),
            vram_bytes: vram.get(index).cloned().flatten(),
            reserved_bytes: reserved.get(index).cloned().flatten(),
            mem_bandwidth_gbps: mem_bandwidth.get(index).cloned().flatten(),
            compute_tflops_fp32: fp32.get(index).cloned().flatten(),
            compute_tflops_fp16: fp16.get(index).cloned().flatten(),
        })
        .collect()
}

fn local_hardware_info_to_proto(
    ann: &PeerAnnouncement,
) -> Option<crate::proto::node::HardwareInfo> {
    let gpus = local_gpu_info_to_proto(ann);
    if ann.hostname.is_none() && ann.is_soc.is_none() && gpus.is_empty() {
        None
    } else {
        Some(crate::proto::node::HardwareInfo {
            is_soc: ann.is_soc,
            hostname: ann.hostname.clone(),
            gpus,
        })
    }
}

struct LegacyGpuFields {
    gpu_name: Option<String>,
    gpu_vram: Option<String>,
    gpu_reserved_bytes: Option<String>,
    gpu_mem_bandwidth_gbps: Option<String>,
    gpu_compute_tflops_fp32: Option<String>,
    gpu_compute_tflops_fp16: Option<String>,
}

fn proto_gpu_info_to_legacy_fields(gpus: &[crate::proto::node::GpuInfo]) -> LegacyGpuFields {
    let names: Vec<String> = gpus.iter().filter_map(|gpu| gpu.name.clone()).collect();
    let gpu_name = crate::system::hardware::summarize_gpu_name(&names);
    let gpu_vram = join_optional_csv(
        &gpus
            .iter()
            .map(|gpu| gpu.vram_bytes.clone())
            .collect::<Vec<_>>(),
    );
    let gpu_reserved_bytes = join_optional_csv(
        &gpus
            .iter()
            .map(|gpu| gpu.reserved_bytes.clone())
            .collect::<Vec<_>>(),
    );
    let gpu_mem_bandwidth_gbps = join_optional_csv(
        &gpus
            .iter()
            .map(|gpu| gpu.mem_bandwidth_gbps.clone())
            .collect::<Vec<_>>(),
    );
    let gpu_compute_tflops_fp32 = join_optional_csv(
        &gpus
            .iter()
            .map(|gpu| gpu.compute_tflops_fp32.clone())
            .collect::<Vec<_>>(),
    );
    let gpu_compute_tflops_fp16 = join_optional_csv(
        &gpus
            .iter()
            .map(|gpu| gpu.compute_tflops_fp16.clone())
            .collect::<Vec<_>>(),
    );

    LegacyGpuFields {
        gpu_name,
        gpu_vram,
        gpu_reserved_bytes,
        gpu_mem_bandwidth_gbps,
        gpu_compute_tflops_fp32,
        gpu_compute_tflops_fp16,
    }
}

/// Returns `true` when a proto descriptor carries a non-empty model name.
/// Descriptors without a valid identity are discarded so a partial list
/// cannot suppress the legacy-identity backfill fallback.
fn proto_descriptor_has_valid_identity(
    descriptor: &crate::proto::node::ServedModelDescriptor,
) -> bool {
    descriptor
        .identity
        .as_ref()
        .map(|id| !id.model_name.is_empty())
        .unwrap_or(false)
}

pub(crate) fn sanitize_gossip_announcement_for_wire(ann: &PeerAnnouncement) -> PeerAnnouncement {
    let mut sanitized = ann.clone();
    sanitized.available_models.clear();
    sanitized.available_model_metadata.clear();
    sanitized.available_model_sizes.clear();
    sanitized.advertised_model_throughput = sanitize_model_throughput_hints_for_ann(&sanitized);
    sanitized.cache_affinity = sanitize_cache_affinity_for_ann(&sanitized);
    sanitized
}

fn sanitize_cache_affinity_for_ann(
    ann: &PeerAnnouncement,
) -> Option<mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement> {
    let routable = routable_model_names(ann);
    let mut advertisement = ann.cache_affinity.clone()?;
    if advertisement.ttl_ms == 0
        || advertisement.ttl_ms
            > u32::try_from(mesh_llm_routing::cache_inventory::CACHE_AFFINITY_TTL.as_millis())
                .unwrap_or(u32::MAX)
    {
        return None;
    }
    advertisement.entries.retain(|entry| {
        entry.matched_tokens > 0
            && entry.tier == mesh_llm_routing::cache_inventory::CacheTier::L1
            && routable.contains(&entry.model)
    });
    advertisement
        .entries
        .truncate(mesh_llm_routing::cache_inventory::CACHE_AFFINITY_MAX_ENTRIES);
    Some(advertisement)
}

fn routable_model_names(ann: &PeerAnnouncement) -> HashSet<String> {
    let mut names = HashSet::new();
    for model in ann
        .hosted_models
        .iter()
        .flatten()
        .chain(&ann.serving_models)
    {
        let model = model.trim();
        if !model.is_empty() {
            names.insert(model.to_string());
        }
    }
    for descriptor in &ann.served_model_descriptors {
        let model = descriptor.identity.model_name.trim();
        if !model.is_empty() {
            names.insert(model.to_string());
        }
    }
    names
}

fn sanitize_model_throughput_hints_for_ann(
    ann: &PeerAnnouncement,
) -> Vec<crate::network::metrics::ModelThroughputHint> {
    let routable = routable_model_names(ann);
    if routable.is_empty() {
        return Vec::new();
    }
    crate::network::metrics::sanitize_model_throughput_hints(
        ann.advertised_model_throughput.clone(),
    )
    .into_iter()
    .filter(|hint| routable.contains(&hint.model_name))
    .collect()
}

pub(crate) fn local_role_to_proto(role: &NodeRole) -> (i32, Option<u32>) {
    match role {
        NodeRole::Worker => (crate::proto::node::NodeRole::Worker as i32, None),
        NodeRole::Host { http_port } => (
            crate::proto::node::NodeRole::Host as i32,
            Some(*http_port as u32),
        ),
        NodeRole::Client => (crate::proto::node::NodeRole::Client as i32, None),
    }
}

pub(crate) fn proto_role_to_local(role_int: i32, http_port: Option<u32>) -> NodeRole {
    match crate::proto::node::NodeRole::try_from(role_int).unwrap_or_default() {
        crate::proto::node::NodeRole::Host => NodeRole::Host {
            http_port: http_port.unwrap_or(0) as u16,
        },
        crate::proto::node::NodeRole::Client => NodeRole::Client,
        _ => NodeRole::Worker,
    }
}

fn local_throughput_hint_to_proto(
    hint: &crate::network::metrics::ModelThroughputHint,
) -> crate::proto::node::AdvertisedModelThroughput {
    crate::proto::node::AdvertisedModelThroughput {
        model_name: hint.model_name.clone(),
        avg_tokens_per_second_milli: hint.avg_tokens_per_second_milli,
        throughput_samples: hint.throughput_samples,
    }
}

fn proto_throughput_hint_to_local(
    hint: &crate::proto::node::AdvertisedModelThroughput,
) -> crate::network::metrics::ModelThroughputHint {
    crate::network::metrics::ModelThroughputHint {
        model_name: hint.model_name.clone(),
        avg_tokens_per_second_milli: hint.avg_tokens_per_second_milli,
        throughput_samples: hint.throughput_samples,
    }
}

fn local_cache_affinity_to_proto(
    advertisement: &mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement,
) -> crate::proto::node::CacheAffinityAdvertisement {
    crate::proto::node::CacheAffinityAdvertisement {
        salt: advertisement.salt.to_vec(),
        epoch: advertisement.epoch,
        generated_at_unix_ms: advertisement.generated_at_unix_ms,
        ttl_ms: advertisement.ttl_ms,
        entries: advertisement
            .entries
            .iter()
            .map(|entry| crate::proto::node::CacheAffinityEntry {
                model_name: entry.model.clone(),
                prefix_digest: entry.prefix_digest.to_vec(),
                matched_tokens: entry.matched_tokens,
                suffix_prefill_tokens: entry.suffix_prefill_tokens,
                tier: match entry.tier {
                    mesh_llm_routing::cache_inventory::CacheTier::L1 => {
                        crate::proto::node::CacheTier::L1 as i32
                    }
                    mesh_llm_routing::cache_inventory::CacheTier::L3 => {
                        crate::proto::node::CacheTier::L3 as i32
                    }
                },
                restore_micros: entry.restore_micros,
                queue_delay_micros: entry.queue_delay_micros,
                prefill_micros_per_token: entry.prefill_micros_per_token,
            })
            .collect(),
    }
}

fn proto_cache_affinity_to_local(
    advertisement: &crate::proto::node::CacheAffinityAdvertisement,
) -> Option<mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement> {
    use mesh_llm_routing::cache_inventory::{
        CACHE_AFFINITY_DIGEST_BYTES, CACHE_AFFINITY_MAX_ENTRIES, CACHE_AFFINITY_SALT_BYTES,
        CACHE_AFFINITY_TTL, CacheAffinityAdvertisement, CacheAffinityEntry, CacheTier,
    };

    let salt: [u8; CACHE_AFFINITY_SALT_BYTES] = advertisement.salt.as_slice().try_into().ok()?;
    let max_ttl_ms = u32::try_from(CACHE_AFFINITY_TTL.as_millis()).unwrap_or(u32::MAX);
    if advertisement.ttl_ms == 0
        || advertisement.ttl_ms > max_ttl_ms
        || advertisement.entries.len() > CACHE_AFFINITY_MAX_ENTRIES
    {
        return None;
    }
    let entries = advertisement
        .entries
        .iter()
        .filter_map(|entry| {
            if entry.model_name.is_empty()
                || entry.model_name.len() > MAX_REMOTE_MODEL_NAME_BYTES
                || entry.matched_tokens == 0
                || crate::proto::node::CacheTier::try_from(entry.tier).ok()?
                    != crate::proto::node::CacheTier::L1
            {
                return None;
            }
            let prefix_digest: [u8; CACHE_AFFINITY_DIGEST_BYTES] =
                entry.prefix_digest.as_slice().try_into().ok()?;
            Some(CacheAffinityEntry {
                model: entry.model_name.clone(),
                prefix_digest,
                matched_tokens: entry.matched_tokens,
                suffix_prefill_tokens: entry.suffix_prefill_tokens,
                tier: CacheTier::L1,
                restore_micros: entry.restore_micros,
                queue_delay_micros: entry.queue_delay_micros,
                prefill_micros_per_token: entry.prefill_micros_per_token,
            })
        })
        .collect();
    let advertisement = CacheAffinityAdvertisement {
        salt,
        epoch: advertisement.epoch,
        generated_at_unix_ms: advertisement.generated_at_unix_ms,
        ttl_ms: advertisement.ttl_ms,
        entries,
    };
    advertisement
        .is_fresh_at(crate::mesh::current_time_unix_ms())
        .then_some(advertisement)
}

pub(crate) fn local_ann_to_proto_ann(
    ann: &PeerAnnouncement,
) -> crate::proto::node::PeerAnnouncement {
    let ann = sanitize_gossip_announcement_for_wire(ann);
    let (role_int, http_port) = local_role_to_proto(&ann.role);
    let serialized_addr = serde_json::to_vec(&ann.addr).unwrap_or_default();
    let demand: Vec<crate::proto::node::ModelDemandEntry> = ann
        .model_demand
        .iter()
        .map(
            |(name, d): (&String, &ModelDemand)| crate::proto::node::ModelDemandEntry {
                model_name: name.clone(),
                last_active: d.last_active,
                request_count: d.request_count,
            },
        )
        .collect();
    let served_model_identities = ann
        .served_model_descriptors
        .iter()
        .map(|descriptor| descriptor_identity_to_proto(&descriptor.identity))
        .collect();
    let served_model_descriptors = ann
        .served_model_descriptors
        .iter()
        .map(|descriptor| crate::proto::node::ServedModelDescriptor {
            identity: Some(descriptor_identity_to_proto(&descriptor.identity)),
            capabilities_known: Some(descriptor.capabilities_known),
            capabilities: Some(crate::proto::node::ModelCapabilities {
                vision: local_capability_level_to_proto(descriptor.capabilities.vision),
                reasoning: local_capability_level_to_proto(descriptor.capabilities.reasoning),
                tool_use: local_capability_level_to_proto(descriptor.capabilities.tool_use),
                moe: descriptor.capabilities.moe,
                multimodal: descriptor.capabilities.multimodal,
                audio: local_capability_level_to_proto(descriptor.capabilities.audio),
            }),
            topology: descriptor.topology.as_ref().map(|topology| {
                crate::proto::node::ModelTopology {
                    moe: topology
                        .moe
                        .as_ref()
                        .map(|moe| crate::proto::node::ModelMoeInfo {
                            expert_count: moe.expert_count,
                            used_expert_count: moe.used_expert_count,
                            min_experts_per_node: moe.min_experts_per_node,
                            source: moe.source.clone(),
                            ranking_source: moe.ranking_source.clone(),
                            ranking_origin: moe.ranking_origin.clone(),
                            ranking: moe.ranking.clone(),
                            ranking_prompt_count: moe.ranking_prompt_count,
                            ranking_tokens: moe.ranking_tokens,
                            ranking_layer_scope: moe.ranking_layer_scope.clone(),
                        }),
                }
            }),
            metadata: descriptor
                .metadata
                .as_ref()
                .map(local_model_metadata_to_proto),
        })
        .collect();
    let served_model_runtime = ann
        .served_model_runtime
        .iter()
        .map(runtime_descriptor_to_proto)
        .collect();
    let hardware = local_hardware_info_to_proto(&ann);
    crate::proto::node::PeerAnnouncement {
        endpoint_id: ann.addr.id.as_bytes().to_vec(),
        role: role_int,
        http_port,
        version: ann.version.clone(),
        gpu_name: ann.gpu_name.clone(),
        hostname: ann.hostname.clone(),
        is_soc: ann.is_soc,
        gpu_vram: ann.gpu_vram.clone(),
        available_models: ann.available_models.clone(),
        serving_models: ann.serving_models.clone(),
        requested_models: ann.requested_models.clone(),
        explicit_model_interests: ann.explicit_model_interests.clone(),
        available_model_metadata: ann.available_model_metadata.clone(),
        experts_summary: ann.experts_summary.clone(),
        rtt_ms: None,
        catalog_models: ann.models.clone(),
        vram_bytes: ann.vram_bytes,
        model_source: ann.model_source.clone(),
        primary_serving: ann.serving_models.first().cloned(),
        mesh_id: ann.mesh_id.clone(),
        mesh_policy_hash: ann.mesh_policy_hash.clone(),
        demand,
        available_model_sizes: ann.available_model_sizes.clone(),
        serialized_addr,
        hosted_models: ann.hosted_models.clone().unwrap_or_default(),
        hosted_models_known: Some(ann.hosted_models.is_some()),
        served_model_identities,
        served_model_descriptors,
        served_model_runtime,
        owner_attestation: ann
            .owner_attestation
            .as_ref()
            .and_then(local_owner_attestation_to_proto),
        genesis_policy: ann
            .genesis_policy
            .as_ref()
            .map(crate::SignedMeshGenesisPolicy::to_proto),
        release_attestation: ann
            .release_attestation
            .as_ref()
            .map(crate::ReleaseBuildAttestation::to_proto),
        direct_admission_proof: ann
            .direct_admission_proof
            .as_ref()
            .map(crate::DirectNodeAdmissionProof::to_proto),
        // Legacy GPU metric fields (29-32) are populated alongside `hardware` so that
        // pre-v0.60.0 peers that do not decode the new `hardware` block can still read
        // bandwidth/tflops/reserved data from the flat fields they already know.
        gpu_mem_bandwidth_gbps: ann.gpu_mem_bandwidth_gbps.clone(),
        gpu_compute_tflops_fp32: ann.gpu_compute_tflops_fp32.clone(),
        gpu_compute_tflops_fp16: ann.gpu_compute_tflops_fp16.clone(),
        gpu_reserved_bytes: ann.gpu_reserved_bytes.clone(),
        hardware,
        first_joined_mesh_ts: ann.first_joined_mesh_ts,
        latency_ms: ann.latency_ms,
        latency_source: match ann.latency_source {
            Some(s) => s as i32,
            None => 0i32,
        },
        latency_age_ms: ann.latency_age_ms.map(|v| v as u32),
        latency_observer_id: ann
            .latency_observer_id
            .as_ref()
            .map(|id| id.as_bytes().to_vec()),
        advertised_model_throughput: sanitize_model_throughput_hints_for_ann(&ann)
            .iter()
            .map(local_throughput_hint_to_proto)
            .collect(),
        subprotocols: skippy_stage_subprotocols(
            ann.artifact_transfer_supported,
            ann.stage_protocol_generation_supported,
            ann.stage_status_list_supported,
            ann.local_gguf_content_id_supported,
        ),
        inference_admission_state: ann.inference_admission_state.map(|state| state as i32),
        cache_affinity: ann
            .cache_affinity
            .as_ref()
            .map(local_cache_affinity_to_proto),
    }
}

pub(crate) fn build_gossip_frame(
    anns: &[PeerAnnouncement],
    sender_id: EndpointId,
) -> crate::proto::node::GossipFrame {
    let peers: Vec<crate::proto::node::PeerAnnouncement> =
        anns.iter().map(local_ann_to_proto_ann).collect();
    crate::proto::node::GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: sender_id.as_bytes().to_vec(),
        peers,
    }
}

/// Upper bound on the number of model-name entries accepted from a single
/// remote peer announcement.
///
/// Peer gossip carries `serving_models` / `hosted_models` (and the other
/// model-name lists) straight through `/api/status` into the console, where
/// they are sorted and rendered one DOM element per entry. The wire frame is
/// only bounded by the generic 8 MiB control-frame ceiling, which still allows
/// hundreds of thousands of short names. Capping the list length at the trust
/// boundary stops a hostile peer from freezing another operator's dashboard
/// with an oversized (or rapidly re-advertised) model list (#860). A real node
/// serves at most a handful of models, so this ceiling is far above any
/// legitimate value.
const MAX_REMOTE_MODEL_LIST_LEN: usize = 256;

/// Upper bound on the byte length of a single remotely-advertised model name.
/// Legitimate model references (including HuggingFace repo + revision) are well
/// under this; anything larger is dropped rather than rendered.
const MAX_REMOTE_MODEL_NAME_BYTES: usize = 512;

/// Sanitize a remotely-supplied list of model names: drop entries that exceed
/// the per-name byte cap and keep at most `MAX_REMOTE_MODEL_LIST_LEN` of them.
fn cap_remote_model_names(names: &[String]) -> Vec<String> {
    names
        .iter()
        .filter(|name| name.len() <= MAX_REMOTE_MODEL_NAME_BYTES)
        .take(MAX_REMOTE_MODEL_LIST_LEN)
        .cloned()
        .collect()
}

pub(crate) fn proto_ann_to_local(
    pa: &crate::proto::node::PeerAnnouncement,
) -> Option<(EndpointAddr, PeerAnnouncement)> {
    let id_arr: [u8; 32] = pa.endpoint_id.as_slice().try_into().ok()?;
    let pk = iroh::PublicKey::from_bytes(&id_arr).ok()?;
    let peer_id = EndpointId::from(pk);
    let addr: EndpointAddr = if !pa.serialized_addr.is_empty() {
        serde_json::from_slice(&pa.serialized_addr).unwrap_or(EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        })
    } else {
        EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        }
    };
    let role = proto_role_to_local(pa.role, pa.http_port);
    let model_demand: HashMap<String, ModelDemand> = pa
        .demand
        .iter()
        .map(|e| {
            (
                e.model_name.clone(),
                ModelDemand {
                    last_active: e.last_active,
                    request_count: e.request_count,
                },
            )
        })
        .collect();
    let hosted_models = pa
        .hosted_models_known
        .unwrap_or(!pa.hosted_models.is_empty())
        .then(|| cap_remote_model_names(&pa.hosted_models));
    let hardware = pa.hardware.as_ref();
    let legacy_gpu_fields = proto_gpu_info_to_legacy_fields(
        hardware
            .map(|hardware| hardware.gpus.as_slice())
            .unwrap_or(&[]),
    );
    let mut ann = PeerAnnouncement {
        addr: addr.clone(),
        role,
        first_joined_mesh_ts: pa.first_joined_mesh_ts,
        models: cap_remote_model_names(&pa.catalog_models),
        vram_bytes: pa.vram_bytes,
        model_source: pa.model_source.clone(),
        serving_models: cap_remote_model_names(&pa.serving_models),
        hosted_models,
        available_models: Vec::new(),
        requested_models: cap_remote_model_names(&pa.requested_models),
        explicit_model_interests: cap_remote_model_names(&pa.explicit_model_interests),
        version: pa.version.clone(),
        model_demand,
        mesh_id: pa.mesh_id.clone(),
        mesh_policy_hash: pa.mesh_policy_hash.clone(),
        gpu_name: legacy_gpu_fields.gpu_name.or_else(|| pa.gpu_name.clone()),
        hostname: hardware
            .and_then(|hardware| hardware.hostname.clone())
            .or_else(|| pa.hostname.clone()),
        is_soc: hardware.and_then(|hardware| hardware.is_soc).or(pa.is_soc),
        gpu_vram: legacy_gpu_fields.gpu_vram.or_else(|| pa.gpu_vram.clone()),
        gpu_reserved_bytes: legacy_gpu_fields
            .gpu_reserved_bytes
            .or_else(|| pa.gpu_reserved_bytes.clone()),
        gpu_mem_bandwidth_gbps: legacy_gpu_fields
            .gpu_mem_bandwidth_gbps
            .or_else(|| pa.gpu_mem_bandwidth_gbps.clone()),
        gpu_compute_tflops_fp32: legacy_gpu_fields
            .gpu_compute_tflops_fp32
            .or_else(|| pa.gpu_compute_tflops_fp32.clone()),
        gpu_compute_tflops_fp16: legacy_gpu_fields
            .gpu_compute_tflops_fp16
            .or_else(|| pa.gpu_compute_tflops_fp16.clone()),
        available_model_metadata: Vec::new(),
        experts_summary: pa.experts_summary.clone(),
        available_model_sizes: HashMap::new(),
        served_model_runtime: pa
            .served_model_runtime
            .iter()
            .map(proto_runtime_descriptor_to_local)
            .collect(),
        served_model_descriptors: if !pa.served_model_descriptors.is_empty() {
            let descriptors: Vec<_> =
                pa.served_model_descriptors
                    .iter()
                    .filter(|descriptor| proto_descriptor_has_valid_identity(descriptor))
                    .map(|descriptor| {
                        let capabilities = descriptor
                            .capabilities
                            .as_ref()
                            .map(|caps| crate::models::ModelCapabilities {
                                multimodal: caps.multimodal,
                                vision: proto_capability_level_to_local(caps.vision),
                                audio: proto_capability_level_to_local(caps.audio),
                                reasoning: proto_capability_level_to_local(caps.reasoning),
                                tool_use: proto_capability_level_to_local(caps.tool_use),
                                moe: caps.moe,
                            })
                            .unwrap_or_default();
                        crate::mesh::ServedModelDescriptor {
                            identity: descriptor
                                .identity
                                .as_ref()
                                .map(proto_identity_to_local)
                                .unwrap_or_default(),
                            capabilities_known: descriptor.capabilities_known.unwrap_or(
                                capabilities != crate::models::ModelCapabilities::default(),
                            ),
                            capabilities,
                            topology: descriptor.topology.as_ref().map(|topology| {
                                crate::models::ModelTopology {
                                    moe: topology.moe.as_ref().map(|moe| {
                                        crate::models::ModelMoeInfo {
                                            expert_count: moe.expert_count,
                                            used_expert_count: moe.used_expert_count,
                                            min_experts_per_node: moe.min_experts_per_node,
                                            source: moe.source.clone(),
                                            ranking_source: moe.ranking_source.clone(),
                                            ranking_origin: moe.ranking_origin.clone(),
                                            ranking: moe.ranking.clone(),
                                            ranking_prompt_count: moe.ranking_prompt_count,
                                            ranking_tokens: moe.ranking_tokens,
                                            ranking_layer_scope: moe.ranking_layer_scope.clone(),
                                        }
                                    }),
                                }
                            }),
                            metadata: descriptor
                                .metadata
                                .as_ref()
                                .map(proto_model_metadata_to_local),
                        }
                    })
                    .collect();
            if descriptors.is_empty() {
                // All descriptors were invalid — fall back to legacy identity list.
                pa.served_model_identities
                    .iter()
                    .map(legacy_descriptor_from_identity)
                    .collect()
            } else {
                descriptors
            }
        } else {
            pa.served_model_identities
                .iter()
                .map(legacy_descriptor_from_identity)
                .collect()
        },
        owner_attestation: pa
            .owner_attestation
            .as_ref()
            .map(proto_owner_attestation_to_local),
        genesis_policy: pa
            .genesis_policy
            .as_ref()
            .and_then(|policy| crate::SignedMeshGenesisPolicy::from_proto(policy).ok()),
        release_attestation: pa
            .release_attestation
            .as_ref()
            .map(proto_release_attestation_to_local),
        direct_admission_proof: pa
            .direct_admission_proof
            .as_ref()
            .and_then(|proof| crate::DirectNodeAdmissionProof::from_proto(proof).ok()),
        artifact_transfer_supported: supports_skippy_artifact_transfer(&pa.subprotocols),
        stage_protocol_generation_supported: supports_skippy_stage_generation(&pa.subprotocols),
        stage_status_list_supported: supports_skippy_status_list(&pa.subprotocols),
        local_gguf_content_id_supported: supports_local_gguf_content_id(&pa.subprotocols),
        advertised_model_throughput: pa
            .advertised_model_throughput
            .iter()
            .map(proto_throughput_hint_to_local)
            .collect(),
        latency_ms: pa.latency_ms,
        latency_source: crate::proto::node::LatencySource::try_from(pa.latency_source).ok(),
        latency_age_ms: pa.latency_age_ms.map(|v| v as u64),
        latency_observer_id: pa.latency_observer_id.as_ref().and_then(|bytes| {
            let arr: [u8; 32] = bytes.as_slice().try_into().ok()?;
            iroh::PublicKey::from_bytes(&arr).ok()
        }),
        inference_admission_state: pa
            .inference_admission_state
            .and_then(|v| crate::proto::node::InferenceAdmissionState::try_from(v).ok()),
        cache_affinity: pa
            .cache_affinity
            .as_ref()
            .and_then(proto_cache_affinity_to_local),
    };
    crate::mesh::backfill_legacy_descriptors(&mut ann);
    ann.advertised_model_throughput = sanitize_model_throughput_hints_for_ann(&ann);
    ann.cache_affinity = sanitize_cache_affinity_for_ann(&ann);
    Some((addr, ann))
}

pub(crate) fn routing_table_to_proto(table: &RoutingTable) -> crate::proto::node::RouteTable {
    let entries = table
        .hosts
        .iter()
        .map(|e| crate::proto::node::RouteEntry {
            endpoint_id: e.endpoint_id.as_bytes().to_vec(),
            model: e.model.clone(),
        })
        .collect();
    crate::proto::node::RouteTable {
        entries,
        mesh_id: table.mesh_id.clone(),
        r#gen: NODE_PROTOCOL_GENERATION,
    }
}

pub(crate) fn mesh_config_to_proto(
    config: &crate::plugin::MeshConfig,
) -> crate::proto::node::NodeConfigSnapshot {
    use crate::plugin::GpuAssignment;
    fn configured_model_ref(declared_ref: &str) -> crate::proto::node::ConfiguredModelRef {
        crate::proto::node::ConfiguredModelRef {
            declared_ref: declared_ref.to_string(),
            source_kind: None,
            revision: None,
        }
    }

    let assignment = match config.gpu.assignment {
        GpuAssignment::Auto => crate::proto::node::GpuAssignment::Auto as i32,
        GpuAssignment::Pinned => crate::proto::node::GpuAssignment::Pinned as i32,
    };
    let models = config
        .models
        .iter()
        .map(|m| crate::proto::node::NodeModelEntry {
            model: m.model.clone(),
            mmproj: m.mmproj.clone(),
            ctx_size: m.ctx_size,
            gpu_id: m.gpu_id.clone(),
            model_ref: Some(configured_model_ref(&m.model)),
            mmproj_ref: m.mmproj.as_deref().map(configured_model_ref),
        })
        .collect();
    let plugins = config
        .plugins
        .iter()
        .map(|p| crate::proto::node::NodePluginEntry {
            name: p.name.clone(),
            enabled: p.enabled,
            command: p.command.clone(),
            args: p.args.clone(),
        })
        .collect();
    let mesh_requirements = {
        let runtime_requirements =
            crate::plugin::mesh_requirements_config_to_runtime(&config.mesh_requirements);
        if runtime_requirements == crate::MeshRequirements::unrestricted() {
            None
        } else {
            Some(runtime_requirements.to_proto())
        }
    };
    crate::proto::node::NodeConfigSnapshot {
        version: config.version.unwrap_or(1),
        gpu: Some(crate::proto::node::NodeGpuConfig { assignment }),
        models,
        plugins,
        config_toml: crate::plugin::config_to_toml(config).ok(),
        mesh_requirements,
    }
}

pub(crate) fn proto_config_to_mesh(
    snapshot: &crate::proto::node::NodeConfigSnapshot,
) -> crate::plugin::MeshConfig {
    if let Ok(Some(parsed)) = full_config_toml_to_mesh(snapshot) {
        return parsed;
    }

    legacy_proto_config_to_mesh(snapshot)
}

pub(crate) fn proto_config_to_mesh_strict(
    snapshot: &crate::proto::node::NodeConfigSnapshot,
) -> Result<crate::plugin::MeshConfig> {
    if let Some(parsed) = full_config_toml_to_mesh(snapshot)? {
        return Ok(parsed);
    }

    Ok(legacy_proto_config_to_mesh(snapshot))
}

fn full_config_toml_to_mesh(
    snapshot: &crate::proto::node::NodeConfigSnapshot,
) -> Result<Option<crate::plugin::MeshConfig>> {
    let Some(config_toml) = snapshot.config_toml.as_deref() else {
        return Ok(None);
    };

    let mut parsed = crate::plugin::parse_config_toml(config_toml)
        .context("invalid full config_toml payload")?;
    if parsed.version.is_none() {
        parsed.version = Some(snapshot.version);
    }
    Ok(Some(parsed))
}

fn legacy_proto_config_to_mesh(
    snapshot: &crate::proto::node::NodeConfigSnapshot,
) -> crate::plugin::MeshConfig {
    use crate::plugin::{
        GpuAssignment, GpuConfig, MeshConfig, ModelConfigEntry, PluginConfigEntry,
    };

    fn declared_ref_or_none(
        configured: Option<&crate::proto::node::ConfiguredModelRef>,
    ) -> Option<String> {
        configured.and_then(|configured| {
            let declared_ref = configured.declared_ref.trim();
            if declared_ref.is_empty() {
                None
            } else {
                Some(declared_ref.to_string())
            }
        })
    }

    let assignment = match snapshot.gpu.as_ref().map(|g| g.assignment) {
        Some(v) if v == crate::proto::node::GpuAssignment::Pinned as i32 => GpuAssignment::Pinned,
        _ => GpuAssignment::Auto,
    };
    let models = snapshot
        .models
        .iter()
        .map(|m| ModelConfigEntry {
            model: declared_ref_or_none(m.model_ref.as_ref()).unwrap_or_else(|| m.model.clone()),
            mmproj: declared_ref_or_none(m.mmproj_ref.as_ref()).or_else(|| m.mmproj.clone()),
            ctx_size: m.ctx_size,
            gpu_id: m.gpu_id.clone(),
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        })
        .collect();
    let plugins = snapshot
        .plugins
        .iter()
        .map(|p| PluginConfigEntry {
            name: p.name.clone(),
            enabled: p.enabled,
            web_ui_enabled: None,
            command: p.command.clone(),
            args: p.args.clone(),
            url: None,
            settings: Default::default(),
            startup: Default::default(),
        })
        .collect();
    let mesh_requirements = snapshot
        .mesh_requirements
        .as_ref()
        .and_then(|proto| crate::MeshRequirements::from_proto(proto).ok())
        .map(|requirements| crate::plugin::mesh_requirements_config_from_runtime(&requirements))
        .unwrap_or_default();
    MeshConfig {
        version: Some(snapshot.version),
        gpu: GpuConfig {
            assignment,
            parallel: None,
        },
        mesh_requirements,
        owner_control: Default::default(),
        telemetry: Default::default(),
        defaults: None,
        runtime: Default::default(),
        models,
        plugins,
        logging: Default::default(),
        extra: Default::default(),
    }
}

pub(crate) fn canonical_config_hash(snapshot: &crate::proto::node::NodeConfigSnapshot) -> [u8; 32] {
    use prost::Message as _;
    use sha2::{Digest, Sha256};
    let bytes = snapshot.encode_to_vec();
    let hash = Sha256::digest(&bytes);
    hash.into()
}

#[cfg(test)]
pub(crate) fn proto_route_table_to_local(table: &crate::proto::node::RouteTable) -> RoutingTable {
    let hosts = table
        .entries
        .iter()
        .filter_map(|e| {
            let arr: [u8; 32] = e.endpoint_id.as_slice().try_into().ok()?;
            let pk = iroh::PublicKey::from_bytes(&arr).ok()?;
            let endpoint_id = EndpointId::from(pk);
            Some(RouteEntry {
                model: e.model.clone(),
                node_id: endpoint_id.fmt_short().to_string(),
                endpoint_id,
                vram_gb: 0.0,
            })
        })
        .collect();
    RoutingTable {
        hosts,
        mesh_id: table.mesh_id.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::requirements::peer_release_attestation_status;

    #[test]
    fn proto_ann_to_local_preserves_malformed_release_attestation_for_later_rejection() {
        let proto = crate::proto::node::PeerAnnouncement {
            endpoint_id: vec![1; 32],
            role: crate::proto::node::NodeRole::Worker as i32,
            release_attestation: Some(crate::proto::node::ReleaseBuildAttestation {
                version: 0,
                node_version: String::new(),
                build_id: String::new(),
                commit: String::new(),
                target_triple: String::new(),
                supported_protocol_generation_min: None,
                supported_protocol_generation_max: None,
                artifact_digest: None,
                signer_key_id: String::new(),
                signature_algorithm: String::new(),
                signature: vec![],
            }),
            ..Default::default()
        };

        let (_addr, ann) = proto_ann_to_local(&proto).expect("announcement should decode");
        let attestation = ann
            .release_attestation
            .as_ref()
            .expect("malformed attestation should still be preserved");

        assert_eq!(
            peer_release_attestation_status(Some(attestation)),
            crate::PeerReleaseAttestationStatus::Invalid
        );
    }

    #[test]
    fn proto_ann_to_local_preserves_missing_release_attestation_as_none() {
        let proto = crate::proto::node::PeerAnnouncement {
            endpoint_id: vec![1; 32],
            role: crate::proto::node::NodeRole::Worker as i32,
            ..Default::default()
        };

        let (_addr, ann) = proto_ann_to_local(&proto).expect("announcement should decode");
        assert!(ann.release_attestation.is_none());
        assert_eq!(
            peer_release_attestation_status(ann.release_attestation.as_ref()),
            crate::PeerReleaseAttestationStatus::Unsigned
        );
    }

    #[test]
    fn proto_ann_to_local_caps_oversized_remote_model_lists() {
        let flood: Vec<String> = (0..10_000).map(|i| format!("model-{i}")).collect();
        let proto = crate::proto::node::PeerAnnouncement {
            endpoint_id: vec![1; 32],
            role: crate::proto::node::NodeRole::Worker as i32,
            serving_models: flood.clone(),
            catalog_models: flood.clone(),
            requested_models: flood.clone(),
            explicit_model_interests: flood.clone(),
            hosted_models: flood.clone(),
            hosted_models_known: Some(true),
            ..Default::default()
        };

        let (_addr, ann) = proto_ann_to_local(&proto).expect("announcement should decode");

        assert_eq!(ann.serving_models.len(), MAX_REMOTE_MODEL_LIST_LEN);
        assert_eq!(ann.models.len(), MAX_REMOTE_MODEL_LIST_LEN);
        assert_eq!(ann.requested_models.len(), MAX_REMOTE_MODEL_LIST_LEN);
        assert_eq!(
            ann.explicit_model_interests.len(),
            MAX_REMOTE_MODEL_LIST_LEN
        );
        assert_eq!(
            ann.hosted_models.as_ref().map(Vec::len),
            Some(MAX_REMOTE_MODEL_LIST_LEN)
        );
    }

    #[test]
    fn proto_ann_to_local_drops_oversized_remote_model_names() {
        let huge_name = "x".repeat(MAX_REMOTE_MODEL_NAME_BYTES + 1);
        let proto = crate::proto::node::PeerAnnouncement {
            endpoint_id: vec![1; 32],
            role: crate::proto::node::NodeRole::Worker as i32,
            serving_models: vec!["ok-model".to_string(), huge_name],
            ..Default::default()
        };

        let (_addr, ann) = proto_ann_to_local(&proto).expect("announcement should decode");

        assert_eq!(ann.serving_models, vec!["ok-model".to_string()]);
    }
}
