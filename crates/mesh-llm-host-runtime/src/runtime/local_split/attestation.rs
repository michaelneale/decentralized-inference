use crate::inference::skippy;
use skippy_protocol::LoadMode;

pub(super) fn split_stage_source_is_ready(
    inventory: &skippy::StageLayerInventory,
    load: &skippy::StageLoadRequest,
) -> bool {
    if load.local_source_required
        && (inventory.model_id != load.model_id
            || inventory.package_ref != load.package_ref
            || inventory.manifest_sha256 != load.manifest_sha256
            || inventory.content_addressed_local_source != Some(true)
            || inventory.source_model_sha256 != load.source_model_sha256)
    {
        return false;
    }
    let ready_running_stage = inventory
        .ready_ranges
        .iter()
        .any(|range| split_layer_range_covers(range, load));
    if ready_running_stage {
        return true;
    }
    if load.load_mode != LoadMode::LayerPackage {
        return inventory
            .available_ranges
            .iter()
            .any(|range| split_layer_range_covers(range, load));
    }
    inventory.preparing_ranges.iter().any(|status| {
        status.topology_id == load.topology_id
            && status.run_id == load.run_id
            && status.stage_id == load.stage_id
            && status.model_id == load.model_id
            && status.package_ref == load.package_ref
            && status.manifest_sha256 == load.manifest_sha256
            && status.admission.as_ref() == Some(&load.admission)
            && status.activation_codec == load.activation_codec
            && status.activation_codec_policy == load.activation_codec_policy
            && status.layer_start <= load.layer_start
            && status.layer_end >= load.layer_end
            && matches!(
                status.state,
                skippy::StagePreparationState::Available | skippy::StagePreparationState::Ready
            )
    })
}

pub(super) fn strict_ready_status_matches(
    status: &skippy::StageStatusSnapshot,
    load: &skippy::StageLoadRequest,
) -> bool {
    let bind_addr_is_ready = status
        .bind_addr
        .parse::<std::net::SocketAddr>()
        .is_ok_and(|addr| addr.port() != 0);
    status.state == skippy::StageRuntimeState::Ready
        && status.topology_id == load.topology_id
        && status.run_id == load.run_id
        && status.model_id == load.model_id
        && status.backend == load.backend
        && status.package_ref.as_deref() == Some(load.package_ref.as_str())
        && status.manifest_sha256.as_deref() == Some(load.manifest_sha256.as_str())
        && status.source_model_sha256 == load.source_model_sha256
        && status.source_model_bytes == load.source_model_bytes
        && status.source_model_path.is_none()
        && status.stage_id == load.stage_id
        && status.stage_index == load.stage_index
        && status.layer_start == load.layer_start
        && status.layer_end == load.layer_end
        && status.admission.as_ref() == Some(&load.admission)
        && status.activation_codec == load.activation_codec
        && status.activation_codec_policy == load.activation_codec_policy
        && status.shutdown_generation == load.shutdown_generation
        && status.coordinator_term == load.coordinator_term
        && status.coordinator_id == load.coordinator_id
        && bind_addr_is_ready
}

fn split_layer_range_covers(range: &skippy::LayerRange, load: &skippy::StageLoadRequest) -> bool {
    range.layer_start <= load.layer_start && range.layer_end >= load.layer_end
}
