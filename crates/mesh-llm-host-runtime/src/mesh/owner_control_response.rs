use mesh_llm_protocol::proto::node;
use mesh_llm_protocol::protocol::NODE_PROTOCOL_GENERATION;

use crate::runtime::config_state::ConfigApplyMode;

pub(super) fn apply_response_envelope(
    request_id: u64,
    apply_config: node::OwnerControlApplyConfigResponse,
) -> node::OwnerControlEnvelope {
    node::OwnerControlEnvelope {
        r#gen: NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: None,
        response: Some(node::OwnerControlResponse {
            request_id,
            get_config: None,
            watch_config: None,
            apply_config: Some(apply_config),
            refresh_inventory: None,
            load_model: None,
            unload_model: None,
            ensure_model: None,
            drain_model: None,
            kv_cache: None,
        }),
        error: None,
    }
}

pub(super) fn config_diagnostics_to_proto(
    diagnostics: &[mesh_llm_config::ConfigDiagnostic],
) -> Vec<node::ConfigDiagnostic> {
    diagnostics
        .iter()
        .map(crate::protocol::convert::config_diagnostic_to_proto)
        .collect()
}

pub(super) fn proto_apply_mode(apply_mode: ConfigApplyMode) -> i32 {
    match apply_mode {
        ConfigApplyMode::Staged => node::ConfigApplyMode::Staged as i32,
        ConfigApplyMode::Live => node::ConfigApplyMode::Live as i32,
        ConfigApplyMode::Noop => node::ConfigApplyMode::Noop as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staged_config_changes_never_advertise_live_application() {
        assert_eq!(
            proto_apply_mode(ConfigApplyMode::Staged),
            node::ConfigApplyMode::Staged as i32
        );
        assert_ne!(
            proto_apply_mode(ConfigApplyMode::Staged),
            node::ConfigApplyMode::Live as i32
        );
    }

    #[test]
    fn live_config_changes_advertise_live_application() {
        assert_eq!(
            proto_apply_mode(ConfigApplyMode::Live),
            node::ConfigApplyMode::Live as i32
        );
    }

    #[test]
    fn config_apply_modes_map_to_explicit_protocol_modes() {
        for apply_mode in [
            ConfigApplyMode::Staged,
            ConfigApplyMode::Live,
            ConfigApplyMode::Noop,
        ] {
            assert_ne!(
                proto_apply_mode(apply_mode),
                node::ConfigApplyMode::Unspecified as i32,
                "{apply_mode:?} must not be advertised as unspecified"
            );
        }

        assert_eq!(
            proto_apply_mode(ConfigApplyMode::Noop),
            node::ConfigApplyMode::Noop as i32
        );
    }
}
