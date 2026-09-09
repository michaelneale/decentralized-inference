//! Prefix affinity and sticky routing helpers for inference target selection.

use crate::inference::election;
use iroh::EndpointId;
use mesh_llm_routing::affinity as shared_affinity;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Debug, Default, Serialize)]
pub struct AffinityStatsSnapshot {
    pub prefix_enabled: bool,
    pub sticky_enabled: bool,
    pub prefix_entries: usize,
    pub prefix_lookups: u64,
    pub prefix_hits: u64,
    pub prefix_misses: u64,
    pub prefix_stale: u64,
    pub prefix_routes: u64,
    pub sticky_routes: u64,
    pub session_routes: u64,
    pub learned: u64,
    pub evicted: u64,
}

mesh_llm_routing::impl_affinity_stats_snapshot!(AffinityStatsSnapshot);

#[derive(Clone)]
pub struct AffinityRouter {
    inner: Arc<shared_affinity::AffinityRouter>,
}

impl AffinityRouter {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(shared_affinity::AffinityRouter::new()),
        }
    }

    #[cfg(test)]
    fn with_config(prefix_enabled: bool, sticky_enabled: bool) -> Self {
        Self {
            inner: Arc::new(shared_affinity::AffinityRouter::with_config(
                prefix_enabled,
                sticky_enabled,
            )),
        }
    }

    pub fn stats_snapshot(&self) -> AffinityStatsSnapshot {
        AffinityStatsSnapshot::from_affinity_stats(
            self.inner.stats_snapshot(),
            self.inner.prefix_enabled(),
            self.inner.sticky_enabled(),
        )
    }

    pub fn sticky_enabled(&self) -> bool {
        self.inner.sticky_enabled()
    }

    pub fn record_sticky_route(&self) {
        self.inner.record_sticky_route();
    }

    pub fn record_session_route(&self) {
        self.inner.record_session_route();
    }
}

impl Default for AffinityRouter {
    fn default() -> Self {
        Self::new()
    }
}

type RoutingKeys = shared_affinity::RoutingKeys;
pub use shared_affinity::{PreparedTargets, TargetSelection};

#[cfg(test)]
pub(crate) fn extract_session_hint_from_body(body: &Value) -> Option<String> {
    shared_affinity::extract_session_hint_from_body(body, &["user", "session_id"])
}

fn routing_keys(parsed_body: Option<&Value>) -> RoutingKeys {
    shared_affinity::routing_keys(
        parsed_body,
        &["prompt_cache_key"],
        &["user", "session_id"],
        false,
    )
}

#[cfg(test)]
fn scaffold_prefix_hash_from_body(body: &Value) -> Option<u64> {
    shared_affinity::scaffold_prefix_hash_from_body(body, false)
}

/// Select an inference target for a model request from a caller-supplied candidate
/// list instead of pulling it from `targets`. This avoids cloning the entire
/// `ModelTargets` when the caller has already reordered the candidates (e.g. by
/// context capacity).
pub fn select_model_target_from_candidates(
    targets: &election::ModelTargets,
    candidates: &[election::InferenceTarget],
    _model: &str,
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
) -> TargetSelection {
    let routing = routing_keys(parsed_body);
    shared_affinity::select_model_target_from_keys(
        targets,
        candidates,
        &routing,
        &affinity.inner,
        None,
    )
}

pub fn prepare_remote_targets_for_request(
    model: &str,
    hosts: &[EndpointId],
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
) -> PreparedTargets {
    let routing = routing_keys(parsed_body);
    let _ = model;
    shared_affinity::prepare_remote_targets_from_keys(hosts, &routing, &affinity.inner, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;

    fn make_id(seed: u8) -> EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SecretKey::from_bytes(&bytes).public()
    }

    fn remote(seed: u8) -> election::InferenceTarget {
        election::InferenceTarget::Remote(make_id(seed))
    }

    fn parse_body(body: &str) -> Value {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn legacy_result_paths_reexport_shared_structs() {
        let target = remote(1);
        let selection: TargetSelection = TargetSelection {
            target: target.clone(),
            prefix_hash: None,
            cache_target: None,
            affinity_applied: false,
        };
        let prepared: PreparedTargets = PreparedTargets {
            ordered: vec![target],
            prefix_hash: selection.prefix_hash,
            cache_target: selection.cache_target,
            affinity_applied: selection.affinity_applied,
        };
        assert_eq!(prepared.ordered.len(), 1);
    }

    #[test]
    fn test_extract_session_hint_from_body_user_preferred() {
        let body =
            parse_body(r#"{"prompt_cache_key":"ignored","user":"bob","session_id":"sess-1"}"#);
        assert_eq!(
            extract_session_hint_from_body(&body),
            Some("bob".to_string())
        );
    }

    #[test]
    fn user_only_chat_has_no_scaffold_prefix_hash() {
        let body = parse_body(r#"{"messages":[{"role":"user","content":"hello"}]}"#);

        assert_eq!(scaffold_prefix_hash_from_body(&body), None);
    }

    #[test]
    fn test_routing_keys_prefix_is_shared_across_first_users() {
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug B"}]}"#,
        );

        let keys_a = routing_keys(Some(&req_a));
        let keys_b = routing_keys(Some(&req_b));

        assert_eq!(keys_a.prefix_hash, keys_b.prefix_hash);
        assert_eq!(keys_a.sticky_hash, None);
        assert_eq!(keys_b.sticky_hash, None);
    }

    #[test]
    fn test_routing_keys_prefix_ignores_object_key_order() {
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run","description":"Run a command","parameters":{"type":"object","properties":{"path":{"type":"string"},"mode":{"type":"string"}},"required":["path","mode"]}}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"function":{"parameters":{"required":["path","mode"],"properties":{"mode":{"type":"string"},"path":{"type":"string"}},"type":"object"},"description":"Run a command","name":"run"},"type":"function"}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );

        let keys_a = routing_keys(Some(&req_a));
        let keys_b = routing_keys(Some(&req_b));

        assert_eq!(keys_a.prefix_hash, keys_b.prefix_hash);
        assert_eq!(keys_a.sticky_hash, keys_b.sticky_hash);
    }

    #[test]
    fn client_does_not_invent_cache_evidence() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let mut targets = election::ModelTargets::default();
        targets.targets.insert(
            "qwen".to_string(),
            vec![
                election::InferenceTarget::Remote(id_a),
                election::InferenceTarget::Remote(id_b),
            ],
        );

        let affinity = AffinityRouter::with_config(true, true);
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task A"}]}"#,
        );
        let candidates = targets.candidates("qwen");
        let selection = select_model_target_from_candidates(
            &targets,
            &candidates,
            "qwen",
            Some(&req_a),
            &affinity,
        );
        assert!(selection.prefix_hash.is_some());
        assert_eq!(selection.cache_target, None);
    }

    #[test]
    fn remote_preparation_preserves_no_cache_evidence() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let hosts = vec![id_a, id_b];
        let affinity = AffinityRouter::with_config(true, true);
        let req = parse_body(
            r#"{"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task A"}]}"#,
        );

        let prepared = prepare_remote_targets_for_request("qwen", &hosts, Some(&req), &affinity);
        assert_eq!(prepared.ordered.len(), 2);
        assert_eq!(prepared.cache_target, None);
    }
}
