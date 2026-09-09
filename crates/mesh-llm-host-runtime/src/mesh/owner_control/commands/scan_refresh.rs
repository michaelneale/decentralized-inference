use crate::mesh::{Node, owner_control_error_envelope};
use crate::proto::node::{
    OwnerControlEnvelope, OwnerControlErrorCode, OwnerControlRefreshInventory,
    OwnerControlRefreshInventoryDisposition, OwnerControlRefreshInventoryResponse,
    OwnerControlResponse,
};
use crate::protocol::NODE_PROTOCOL_GENERATION;
use crate::runtime_data::{
    InventoryScanDisposition, InventoryScanOutcome, sorted_inventory_entries,
};

pub(crate) async fn execute(node: &Node, request_id: u64) -> OwnerControlEnvelope {
    match node.refresh_local_inventory_snapshot().await {
        Ok(outcome) => success_envelope(node, request_id, outcome).await,
        Err(error) => owner_control_error_envelope(
            OwnerControlErrorCode::ControlUnavailable,
            Some(request_id),
            None,
            error.to_string(),
        ),
    }
}

async fn success_envelope(
    node: &Node,
    request_id: u64,
    outcome: InventoryScanOutcome,
) -> OwnerControlEnvelope {
    let inventory = OwnerControlRefreshInventory {
        entries: sorted_inventory_entries(&outcome.snapshot),
        disposition: match outcome.disposition {
            InventoryScanDisposition::Executed => {
                OwnerControlRefreshInventoryDisposition::Executed as i32
            }
            InventoryScanDisposition::Coalesced => {
                OwnerControlRefreshInventoryDisposition::Coalesced as i32
            }
        },
    };
    OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: None,
        response: Some(OwnerControlResponse {
            request_id,
            get_config: None,
            watch_config: None,
            apply_config: None,
            refresh_inventory: Some(OwnerControlRefreshInventoryResponse {
                snapshot: Some(node.current_owner_control_snapshot().await),
                inventory: Some(inventory),
            }),
            load_model: None,
            unload_model: None,
            ensure_model: None,
            drain_model: None,
            kv_cache: None,
        }),
        error: None,
    }
}
