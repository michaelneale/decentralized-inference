//! Prefix affinity and sticky routing helpers for inference target selection.

use crate::election;
use iroh::EndpointId;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const AFFINITY_TTL: Duration = Duration::from_secs(20 * 60);
const AFFINITY_MAX_ENTRIES: usize = 4096;

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

fn prefix_only_enabled() -> bool {
    std::env::var("MESH_LLM_PREFIX_ONLY")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug)]
struct AffinityConfig {
    prefix_enabled: bool,
    sticky_enabled: bool,
}

impl AffinityConfig {
    fn from_env() -> Self {
        Self {
            prefix_enabled: std::env::var_os("MESH_LLM_DISABLE_PREFIX_AFFINITY").is_none(),
            sticky_enabled: std::env::var_os("MESH_LLM_DISABLE_STICKY_ROUTING").is_none(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct AffinityKey {
    model: String,
    prefix_hash: u64,
}

#[derive(Clone, Debug)]
struct AffinityEntry {
    target: election::InferenceTarget,
    last_used: Instant,
}

#[derive(Default)]
struct AffinityState {
    entries: HashMap<AffinityKey, AffinityEntry>,
    lru: VecDeque<AffinityKey>,
    stats: AffinityStatsSnapshot,
}

#[derive(Clone)]
pub struct AffinityRouter {
    inner: Arc<Mutex<AffinityState>>,
    config: Arc<AffinityConfig>,
}

impl AffinityRouter {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AffinityState::default())),
            config: Arc::new(AffinityConfig::from_env()),
        }
    }

    #[cfg(test)]
    fn with_config(prefix_enabled: bool, sticky_enabled: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AffinityState::default())),
            config: Arc::new(AffinityConfig {
                prefix_enabled,
                sticky_enabled,
            }),
        }
    }

    pub fn stats_snapshot(&self) -> AffinityStatsSnapshot {
        let mut state = self.inner.lock().unwrap();
        state.prune_expired();
        let mut stats = state.stats.clone();
        stats.prefix_entries = state.entries.len();
        stats.prefix_enabled = self.config.prefix_enabled;
        stats.sticky_enabled = self.config.sticky_enabled;
        stats
    }

    pub fn sticky_enabled(&self) -> bool {
        self.config.sticky_enabled
    }

    pub fn record_sticky_route(&self) {
        let mut state = self.inner.lock().unwrap();
        state.stats.sticky_routes += 1;
    }

    pub fn record_session_route(&self) {
        let mut state = self.inner.lock().unwrap();
        state.stats.session_routes += 1;
    }

    pub fn lookup_target(
        &self,
        model: &str,
        prefix_hash: u64,
        candidates: &[election::InferenceTarget],
    ) -> Option<election::InferenceTarget> {
        if !self.config.prefix_enabled {
            return None;
        }
        let key = AffinityKey {
            model: model.to_string(),
            prefix_hash,
        };
        let mut state = self.inner.lock().unwrap();
        state.prune_expired();
        state.stats.prefix_lookups += 1;
        let entry = match state.entries.get(&key).cloned() {
            Some(entry) => entry,
            None => {
                state.stats.prefix_misses += 1;
                return None;
            }
        };
        if !candidates.contains(&entry.target) {
            state.remove_key(&key);
            state.stats.prefix_stale += 1;
            state.stats.prefix_misses += 1;
            return None;
        }
        state.touch_key(&key);
        if let Some(existing) = state.entries.get_mut(&key) {
            existing.last_used = Instant::now();
        }
        state.stats.prefix_hits += 1;
        state.stats.prefix_routes += 1;
        Some(entry.target)
    }

    pub fn learn_target(&self, model: &str, prefix_hash: u64, target: &election::InferenceTarget) {
        if !self.config.prefix_enabled
            || matches!(
                target,
                election::InferenceTarget::None
                    | election::InferenceTarget::MoeLocal(_)
                    | election::InferenceTarget::MoeRemote(_)
            )
        {
            return;
        }

        let key = AffinityKey {
            model: model.to_string(),
            prefix_hash,
        };
        let now = Instant::now();
        let mut state = self.inner.lock().unwrap();
        state.prune_expired();
        state.entries.insert(
            key.clone(),
            AffinityEntry {
                target: target.clone(),
                last_used: now,
            },
        );
        state.touch_key(&key);
        state.stats.learned += 1;
        while state.entries.len() > AFFINITY_MAX_ENTRIES {
            if let Some(oldest) = state.lru.pop_front() {
                if state.entries.remove(&oldest).is_some() {
                    state.stats.evicted += 1;
                }
            } else {
                break;
            }
        }
    }

    pub fn forget_target(&self, model: &str, prefix_hash: u64, target: &election::InferenceTarget) {
        if !self.config.prefix_enabled {
            return;
        }
        let key = AffinityKey {
            model: model.to_string(),
            prefix_hash,
        };
        let mut state = self.inner.lock().unwrap();
        if state
            .entries
            .get(&key)
            .map(|entry| &entry.target == target)
            .unwrap_or(false)
        {
            state.remove_key(&key);
            state.stats.prefix_stale += 1;
        }
    }
}

impl Default for AffinityRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl AffinityState {
    fn prune_expired(&mut self) {
        let now = Instant::now();

        loop {
            let front_key = match self.lru.front() {
                Some(key) => key.clone(),
                None => break,
            };

            match self.entries.get(&front_key) {
                Some(entry) => {
                    if now.duration_since(entry.last_used) > AFFINITY_TTL {
                        // Oldest entry is expired: evict it.
                        self.lru.pop_front();
                        if self.entries.remove(&front_key).is_some() {
                            self.stats.prefix_stale += 1;
                        }
                        // Continue to check next-oldest entry.
                    } else {
                        // Oldest entry is not expired; newer ones cannot be expired yet.
                        break;
                    }
                }
                None => {
                    // Key is in LRU but missing from entries; drop it from LRU and continue.
                    self.lru.pop_front();
                }
            }
        }
    }

    fn touch_key(&mut self, key: &AffinityKey) {
        if let Some(pos) = self.lru.iter().position(|existing| existing == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key.clone());
    }

    fn remove_key(&mut self, key: &AffinityKey) {
        self.entries.remove(key);
        if let Some(pos) = self.lru.iter().position(|existing| existing == key) {
            self.lru.remove(pos);
        }
    }
}

#[derive(Clone, Debug, Default)]
struct RoutingKeys {
    session_hash: Option<u64>,
    prefix_hash: Option<u64>,
    sticky_hash: Option<u64>,
}

pub struct TargetSelection {
    pub target: election::InferenceTarget,
    pub learn_prefix_hash: Option<u64>,
    pub cached_target: Option<election::InferenceTarget>,
}

pub struct PreparedTargets {
    pub ordered: Vec<election::InferenceTarget>,
    pub learn_prefix_hash: Option<u64>,
    pub cached_target: Option<election::InferenceTarget>,
}

pub(crate) fn extract_session_hint_from_body(body: &Value) -> Option<String> {
    top_level_string(body, "user").or_else(|| top_level_string(body, "session_id"))
}

fn top_level_string(body: &Value, key: &str) -> Option<String> {
    body.get(key)
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn message_text(msg: &Value) -> Option<String> {
    if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
        return Some(s.to_string());
    }
    if let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) {
        let mut out = String::new();
        for block in blocks {
            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                out.push_str(text);
                out.push('\n');
            }
        }
        if !out.is_empty() {
            return Some(out);
        }
    }
    None
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |acc, &b| {
        (acc ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn hash_combine(a: u64, b: u64) -> u64 {
    a.wrapping_mul(31).wrapping_add(b)
}

fn hash_tagged_text(mut acc: u64, tag: &str, text: &str) -> u64 {
    acc = hash_combine(acc, hash_bytes(tag.as_bytes()));
    hash_combine(acc, hash_bytes(text.as_bytes()))
}

fn hash_json_value(mut acc: u64, value: &Value) -> u64 {
    match value {
        Value::Null => hash_combine(acc, hash_bytes(b"null")),
        Value::Bool(boolean) => {
            acc = hash_combine(acc, hash_bytes(b"bool"));
            hash_combine(acc, hash_bytes(boolean.to_string().as_bytes()))
        }
        Value::Number(number) => {
            acc = hash_combine(acc, hash_bytes(b"number"));
            hash_combine(acc, hash_bytes(number.to_string().as_bytes()))
        }
        Value::String(text) => {
            acc = hash_combine(acc, hash_bytes(b"string"));
            hash_combine(acc, hash_bytes(text.as_bytes()))
        }
        Value::Array(items) => {
            acc = hash_combine(acc, hash_bytes(b"array"));
            acc = hash_combine(acc, items.len() as u64);
            for item in items {
                acc = hash_json_value(acc, item);
            }
            acc
        }
        Value::Object(map) => {
            acc = hash_combine(acc, hash_bytes(b"object"));
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort_unstable();
            for key in keys {
                acc = hash_combine(acc, hash_bytes(key.as_bytes()));
                acc = hash_json_value(acc, &map[key]);
            }
            acc
        }
    }
}

fn hash_tagged_json(mut acc: u64, tag: &str, value: &Value) -> u64 {
    acc = hash_combine(acc, hash_bytes(tag.as_bytes()));
    hash_json_value(acc, value)
}

fn scaffold_prefix_hash_from_body(body: &Value) -> Option<u64> {
    let mut hash = 0u64;
    let mut found = false;

    for key in [
        "tools",
        "functions",
        "response_format",
        "tool_choice",
        "parallel_tool_calls",
    ] {
        if let Some(value) = body.get(key) {
            hash = hash_tagged_json(hash, key, value);
            found = true;
        }
    }

    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            match role {
                "system" | "developer" => {
                    if let Some(text) = message_text(msg) {
                        hash = hash_tagged_text(hash, role, &text);
                        found = true;
                    }
                }
                "user" => break,
                _ => {}
            }
        }
    }

    found.then_some(hash)
}

fn first_user_hash_from_body(body: &Value) -> Option<u64> {
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                return message_text(msg).map(|text| hash_tagged_text(0, "user", &text));
            }
        }
    }
    body.get("prompt")
        .and_then(|value| value.as_str())
        .map(|prompt| hash_tagged_text(0, "prompt", prompt))
}

fn routing_keys(parsed_body: Option<&Value>) -> RoutingKeys {
    let Some(body) = parsed_body else {
        return RoutingKeys::default();
    };

    let session_hash = extract_session_hint_from_body(body).map(|hint| hash_bytes(hint.as_bytes()));
    let prefix_hash = scaffold_prefix_hash_from_body(body);
    let sticky_hash = session_hash.or_else(|| {
        let mut hash = 0u64;
        let mut found = false;
        if let Some(prefix_hash) = prefix_hash {
            hash = hash_combine(hash, prefix_hash);
            found = true;
        }
        if let Some(user_hash) = first_user_hash_from_body(body) {
            hash = hash_combine(hash, user_hash);
            found = true;
        }
        found.then_some(hash)
    });

    RoutingKeys {
        session_hash,
        prefix_hash,
        sticky_hash,
    }
}

fn rotate_targets_by_hash(targets: &mut [election::InferenceTarget], key: u64) {
    if !targets.is_empty() {
        let idx = key as usize % targets.len();
        targets.rotate_left(idx);
    }
}

fn move_target_first(
    targets: &mut [election::InferenceTarget],
    target: &election::InferenceTarget,
) -> bool {
    if let Some(pos) = targets.iter().position(|candidate| candidate == target) {
        targets[..=pos].rotate_right(1);
        true
    } else {
        false
    }
}

pub fn select_model_target_for_request(
    targets: &election::ModelTargets,
    model: &str,
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
) -> TargetSelection {
    let routing = routing_keys(parsed_body);
    let candidates = targets.candidates(model);

    if let Some(session_hash) = routing.session_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_session_route();
        return TargetSelection {
            target: targets.get_sticky(model, session_hash),
            learn_prefix_hash: None,
            cached_target: None,
        };
    }

    if let Some(prefix_hash) = routing.prefix_hash {
        if let Some(target) = affinity.lookup_target(model, prefix_hash, &candidates) {
            return TargetSelection {
                target: target.clone(),
                learn_prefix_hash: Some(prefix_hash),
                cached_target: Some(target),
            };
        }

        if prefix_only_enabled() {
            return TargetSelection {
                target: targets.get_sticky(model, prefix_hash),
                learn_prefix_hash: Some(prefix_hash),
                cached_target: None,
            };
        }

        if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled()) {
            affinity.record_sticky_route();
            return TargetSelection {
                target: targets.get_sticky(model, sticky_hash),
                learn_prefix_hash: Some(prefix_hash),
                cached_target: None,
            };
        }

        return TargetSelection {
            target: targets.get(model),
            learn_prefix_hash: Some(prefix_hash),
            cached_target: None,
        };
    }

    if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_sticky_route();
        return TargetSelection {
            target: targets.get_sticky(model, sticky_hash),
            learn_prefix_hash: None,
            cached_target: None,
        };
    }

    TargetSelection {
        target: targets.get(model),
        learn_prefix_hash: None,
        cached_target: None,
    }
}

pub fn prepare_remote_targets_for_request(
    model: &str,
    hosts: &[EndpointId],
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
) -> PreparedTargets {
    let routing = routing_keys(parsed_body);
    let mut ordered: Vec<election::InferenceTarget> = hosts
        .iter()
        .copied()
        .map(election::InferenceTarget::Remote)
        .collect();
    let mut cached_target = None;
    let mut learn_prefix_hash = None;

    if let Some(session_hash) = routing.session_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_session_route();
        rotate_targets_by_hash(&mut ordered, session_hash);
        return PreparedTargets {
            ordered,
            learn_prefix_hash: None,
            cached_target: None,
        };
    }

    if let Some(prefix_hash) = routing.prefix_hash {
        learn_prefix_hash = Some(prefix_hash);
        if let Some(target) = affinity.lookup_target(model, prefix_hash, &ordered) {
            move_target_first(&mut ordered, &target);
            cached_target = Some(target);
        } else if prefix_only_enabled() {
            rotate_targets_by_hash(&mut ordered, prefix_hash);
        } else if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled())
        {
            affinity.record_sticky_route();
            rotate_targets_by_hash(&mut ordered, sticky_hash);
        }
    } else if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_sticky_route();
        rotate_targets_by_hash(&mut ordered, sticky_hash);
    }

    PreparedTargets {
        ordered,
        learn_prefix_hash,
        cached_target,
    }
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

    fn parse_body(body: &str) -> Value {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn test_extract_session_hint_from_body_user_preferred() {
        let body = parse_body(r#"{"user":"bob","session_id":"sess-1"}"#);
        assert_eq!(
            extract_session_hint_from_body(&body),
            Some("bob".to_string())
        );
    }

    #[test]
    fn test_routing_keys_prefix_shared_across_first_user_changes() {
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug B"}]}"#,
        );

        let keys_a = routing_keys(Some(&req_a));
        let keys_b = routing_keys(Some(&req_b));

        assert_eq!(keys_a.prefix_hash, keys_b.prefix_hash);
        assert_ne!(keys_a.sticky_hash, keys_b.sticky_hash);
    }

    #[test]
    fn test_routing_keys_prefix_ignores_object_key_order() {
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run","description":"Run a command","parameters":{"type":"object","properties":{"path":{"type":"string"},"mode":{"type":"string"}},"required":["path","mode"]}}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"function":{"parameters":{"required":["path","mode"],"properties":{"mode":{"type":"string"},"path":{"type":"string"}},"type":"object"},"description":"Run a command","name":"run"},"type":"function"}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug B"}]}"#,
        );

        let keys_a = routing_keys(Some(&req_a));
        let keys_b = routing_keys(Some(&req_b));

        assert_eq!(keys_a.prefix_hash, keys_b.prefix_hash);
        assert_ne!(keys_a.sticky_hash, keys_b.sticky_hash);
    }

    #[test]
    fn test_select_model_target_uses_cached_prefix_target() {
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
        let req_b = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task B"}]}"#,
        );

        let first = select_model_target_for_request(&targets, "qwen", Some(&req_a), &affinity);
        let prefix_hash = first.learn_prefix_hash.unwrap();
        affinity.learn_target("qwen", prefix_hash, &first.target);

        let second = select_model_target_for_request(&targets, "qwen", Some(&req_b), &affinity);
        assert_eq!(Some(second.target.clone()), second.cached_target);
        assert_eq!(first.target, second.target);
    }

    #[test]
    fn test_prepare_remote_targets_prefers_cached_host() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let hosts = vec![id_a, id_b];
        let affinity = AffinityRouter::with_config(true, true);
        let req = parse_body(
            r#"{"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task A"}]}"#,
        );

        let prefix_hash = routing_keys(Some(&req)).prefix_hash.unwrap();
        affinity.learn_target(
            "qwen",
            prefix_hash,
            &election::InferenceTarget::Remote(id_b),
        );

        let prepared = prepare_remote_targets_for_request("qwen", &hosts, Some(&req), &affinity);
        assert_eq!(
            prepared.ordered.first(),
            Some(&election::InferenceTarget::Remote(id_b))
        );
        assert_eq!(
            prepared.cached_target,
            Some(election::InferenceTarget::Remote(id_b))
        );
    }
}
