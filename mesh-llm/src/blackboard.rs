//! Blackboard — shared ephemeral text messages across the mesh.
//!
//! Every node holds the same in-memory list (eventually consistent via flood-fill).
//! Items expire after 48 hours and the list is capped at 500 items.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Max items to keep in memory.
const MAX_ITEMS: usize = 500;
/// Items older than this are pruned.
const TTL_SECS: u64 = 48 * 3600; // 48 hours
/// Max posts per peer per minute.
const RATE_LIMIT_PER_MIN: usize = 10;
/// Max text length per message (bytes).
const MAX_TEXT_LEN: usize = 4096;

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// A single blackboard item — just text from someone at a time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlackboardItem {
    /// Unique ID (timestamp nanos + random bits to avoid collision).
    pub id: u64,
    /// Display name of the author.
    pub from: String,
    /// Peer endpoint ID (hex-encoded, for dedup across name collisions).
    pub peer_id: String,
    /// Unix timestamp (seconds).
    pub timestamp: u64,
    /// The message.
    pub text: String,

}

impl BlackboardItem {
    pub fn new(from: String, peer_id: String, text: String) -> Self {
        let ts = now_secs();
        // ID = timestamp nanos for uniqueness, with random low bits
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let rand_bits: u16 = (nanos & 0xFFFF) as u16 ^ (std::process::id() as u16);
        let id = (nanos & !0xFFFF) | rand_bits as u64;
        Self {
            id,
            from,
            peer_id,
            timestamp: ts,
            text,
        }
    }
}

/// Wire message for blackboard protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BlackboardMessage {
    /// Broadcast a new item to peers.
    Post(BlackboardItem),
    /// Request: send me all your item IDs.
    SyncRequest,
    /// Response: here are my item IDs.
    SyncDigest(Vec<u64>),
    /// Request: send me these items (by ID).
    FetchRequest(Vec<u64>),
    /// Response: here are the items you asked for.
    FetchResponse(Vec<BlackboardItem>),
}

/// In-memory blackboard store. Shared across the node.
#[derive(Clone)]
pub struct BlackboardStore {
    items: Arc<Mutex<Vec<BlackboardItem>>>,
    enabled: Arc<std::sync::atomic::AtomicBool>,
    /// Rate limit tracking: peer_id → list of post timestamps (unix secs).
    rate_log: Arc<Mutex<std::collections::HashMap<String, Vec<u64>>>>,
}

impl BlackboardStore {
    pub fn new(enabled: bool) -> Self {
        Self {
            items: Arc::new(Mutex::new(Vec::new())),
            enabled: Arc::new(std::sync::atomic::AtomicBool::new(enabled)),
            rate_log: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_enabled(&self, v: bool) {
        self.enabled.store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Insert an item if not already present. Returns true if new.
    /// Used for items received from the network (no rate limiting).
    pub async fn insert(&self, item: BlackboardItem) -> bool {
        let mut items = self.items.lock().await;
        if items.iter().any(|i| i.id == item.id) {
            return false;
        }
        items.push(item);
        self.prune_locked(&mut items);
        true
    }

    /// Post a new item locally — enforces rate limit and text length.
    /// Returns Ok(item) on success, Err(reason) if rejected.
    pub async fn post(&self, item: BlackboardItem) -> Result<BlackboardItem, String> {
        // Text length check
        if item.text.len() > MAX_TEXT_LEN {
            return Err(format!("Message too long ({} bytes, max {})", item.text.len(), MAX_TEXT_LEN));
        }

        // Rate limit check
        let now = now_secs();
        let mut log = self.rate_log.lock().await;
        let timestamps = log.entry(item.peer_id.clone()).or_default();
        // Prune old entries
        timestamps.retain(|&t| now - t < 60);
        if timestamps.len() >= RATE_LIMIT_PER_MIN {
            return Err(format!("Rate limited ({} posts/min max)", RATE_LIMIT_PER_MIN));
        }
        timestamps.push(now);
        drop(log);

        // Insert
        self.insert(item.clone()).await;
        Ok(item)
    }

    /// Get all items (newest first).
    pub async fn all(&self) -> Vec<BlackboardItem> {
        let mut items = self.items.lock().await;
        self.prune_locked(&mut items);
        let mut result = items.clone();
        result.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        result
    }

    /// Get all item IDs.
    pub async fn ids(&self) -> Vec<u64> {
        let items = self.items.lock().await;
        items.iter().map(|i| i.id).collect()
    }

    /// Get items by IDs.
    pub async fn get_by_ids(&self, ids: &[u64]) -> Vec<BlackboardItem> {
        let items = self.items.lock().await;
        items.iter().filter(|i| ids.contains(&i.id)).cloned().collect()
    }

    /// Search items by multi-term OR matching (like megamind).
    /// Query is split into terms — any term matching is a hit.
    /// Results ranked by number of matching terms (most relevant first).
    /// `since` filters to items newer than this unix timestamp (0 = no filter).
    pub async fn search(&self, query: &str, since: u64) -> Vec<BlackboardItem> {
        let terms: Vec<String> = query.to_lowercase()
            .split_whitespace()
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect();
        if terms.is_empty() {
            return self.all().await;
        }
        let mut items = self.items.lock().await;
        self.prune_locked(&mut items);
        let mut scored: Vec<(usize, BlackboardItem)> = items.iter()
            .filter(|i| since == 0 || i.timestamp > since)
            .filter_map(|i| {
                let text_lower = i.text.to_lowercase();
                let from_lower = i.from.to_lowercase();
                let hits = terms.iter()
                    .filter(|t| text_lower.contains(t.as_str()) || from_lower.contains(t.as_str()))
                    .count();
                if hits > 0 { Some((hits, i.clone())) } else { None }
            })
            .collect();
        // Sort by hits descending, then by timestamp descending
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.timestamp.cmp(&a.1.timestamp)));
        scored.into_iter().map(|(_, item)| item).collect()
    }


    /// Feed: items newer than a timestamp, optionally filtered by peer.
    #[allow(dead_code)]
    pub async fn feed(&self, since: u64, from: Option<&str>, limit: usize) -> Vec<BlackboardItem> {
        let mut items = self.items.lock().await;
        self.prune_locked(&mut items);
        let mut result: Vec<_> = items.iter()
            .filter(|i| i.timestamp > since)
            .filter(|i| from.map(|f| i.from.to_lowercase().contains(&f.to_lowercase())).unwrap_or(true))
            .cloned()
            .collect();
        result.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        result.truncate(limit);
        result
    }

    /// Prune old and excess items.
    fn prune_locked(&self, items: &mut Vec<BlackboardItem>) {
        let cutoff = now_secs().saturating_sub(TTL_SECS);
        items.retain(|i| i.timestamp > cutoff);
        if items.len() > MAX_ITEMS {
            items.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
            items.truncate(MAX_ITEMS);
        }
    }
}

// ── PII filter ──

/// Check text for obvious PII/secrets. Returns list of issues found.
/// If empty, text is clean.
pub fn pii_check(text: &str) -> Vec<String> {
    let mut issues = Vec::new();

    // Email addresses
    for word in text.split_whitespace() {
        let w = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '.' && c != '-' && c != '_');
        if w.contains('@') && w.contains('.') && w.len() > 5 {
            let parts: Vec<&str> = w.split('@').collect();
            if parts.len() == 2 && parts[1].contains('.') && parts[0].len() > 0 {
                issues.push(format!("Possible email: {}", w));
            }
        }
    }

    // API keys / tokens (common prefixes)
    let key_prefixes = ["sk-", "pk-", "ghp_", "ghu_", "ghs_", "AKIA", "xoxb-", "xoxp-", "Bearer "];
    for prefix in &key_prefixes {
        if text.contains(prefix) {
            issues.push(format!("Possible API key/token ({}...)", prefix));
        }
    }

    // High-entropy strings (likely secrets)
    for word in text.split_whitespace() {
        let w = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '=' && c != '+' && c != '/');
        if w.len() >= 20 {
            let entropy = shannon_entropy(w);
            if entropy > 4.5 {
                issues.push(format!("High-entropy string (possible secret): {}...", &w[..20.min(w.len())]));
            }
        }
    }

    // Private file paths
    if text.contains("/Users/") || text.contains("/home/") || text.contains("C:\\Users\\") {
        issues.push("Contains private file path".into());
    }

    // IP addresses (v4)
    let ip_re_simple = regex_lite::Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap();
    for m in ip_re_simple.find_iter(text) {
        let ip = m.as_str();
        // Skip common non-private IPs like version numbers
        if !ip.starts_with("0.") && !ip.starts_with("127.") {
            issues.push(format!("Possible IP address: {}", ip));
        }
    }

    // SSH keys / PEM blocks
    if text.contains("-----BEGIN") || text.contains("ssh-rsa") || text.contains("ssh-ed25519") {
        issues.push("Contains SSH key or PEM block".into());
    }

    // Password patterns
    let pw_patterns = ["password=", "passwd=", "secret=", "token=", "api_key="];
    let lower = text.to_lowercase();
    for pat in &pw_patterns {
        if lower.contains(pat) {
            issues.push(format!("Possible credential pattern: {}", pat));
        }
    }

    issues
}

/// Scrub known PII patterns from text, returning cleaned version.
pub fn pii_scrub(text: &str) -> String {
    let mut result = text.to_string();

    // Scrub private paths
    let path_re = regex_lite::Regex::new(r"/Users/[a-zA-Z0-9_.-]+/").unwrap();
    result = path_re.replace_all(&result, "~/").to_string();
    let path_re2 = regex_lite::Regex::new(r"/home/[a-zA-Z0-9_.-]+/").unwrap();
    result = path_re2.replace_all(&result, "~/").to_string();

    result
}

/// Shannon entropy in bits per character.
fn shannon_entropy(s: &str) -> f64 {
    let len = s.len() as f64;
    if len == 0.0 { return 0.0; }
    let mut freq = [0u32; 256];
    for b in s.bytes() {
        freq[b as usize] += 1;
    }
    let mut entropy = 0.0;
    for &count in &freq {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }
    entropy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pii_check_email() {
        let issues = pii_check("contact me at bob@example.com for details");
        assert!(issues.iter().any(|i| i.contains("email")), "Should detect email");
    }

    #[test]
    fn test_pii_check_api_key() {
        let issues = pii_check("use this key: sk-abc123xyz");
        assert!(issues.iter().any(|i| i.contains("API key")), "Should detect API key prefix");
    }

    #[test]
    fn test_pii_check_clean() {
        let issues = pii_check("Set ctx-size to 2048 to fix CUDA OOM on 8GB cards");
        assert!(issues.is_empty(), "Clean text should have no issues: {:?}", issues);
    }

    #[test]
    fn test_pii_check_path() {
        let issues = pii_check("the file is at /Users/michael/secret/data.txt");
        assert!(issues.iter().any(|i| i.contains("file path")), "Should detect private path");
    }

    #[test]
    fn test_pii_scrub_path() {
        let scrubbed = pii_scrub("look at /Users/michael/code/main.rs");
        assert_eq!(scrubbed, "look at ~/code/main.rs");
    }

    #[test]
    fn test_blackboard_item_unique_ids() {
        let a = BlackboardItem::new("alice".into(), "abc".into(), "hello".into());
        std::thread::sleep(std::time::Duration::from_millis(1));
        let b = BlackboardItem::new("bob".into(), "def".into(), "world".into());
        assert_ne!(a.id, b.id);
    }

    #[tokio::test]
    async fn test_store_insert_dedup() {
        let store = BlackboardStore::new(true);
        let item = BlackboardItem::new("alice".into(), "abc".into(), "hello".into());
        assert!(store.insert(item.clone()).await);
        assert!(!store.insert(item).await); // duplicate
        assert_eq!(store.all().await.len(), 1);
    }

    #[tokio::test]
    async fn test_store_search_single_term() {
        let store = BlackboardStore::new(true);
        store.insert(BlackboardItem::new("alice".into(), "a".into(), "CUDA OOM fix".into())).await;
        store.insert(BlackboardItem::new("bob".into(), "b".into(), "networking stuff".into())).await;
        let results = store.search("cuda", 0).await;
        assert_eq!(results.len(), 1);
        assert!(results[0].text.contains("CUDA"));
    }

    #[tokio::test]
    async fn test_store_search_multi_term_or() {
        let store = BlackboardStore::new(true);
        store.insert(BlackboardItem::new("alice".into(), "a".into(), "CUDA OOM fix".into())).await;
        store.insert(BlackboardItem::new("bob".into(), "b".into(), "networking refactor".into())).await;
        store.insert(BlackboardItem::new("carol".into(), "c".into(), "unrelated stuff".into())).await;
        // "CUDA networking" should match both alice and bob (OR)
        let results = store.search("CUDA networking", 0).await;
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_store_search_ranking() {
        let store = BlackboardStore::new(true);
        store.insert(BlackboardItem::new("alice".into(), "a".into(), "CUDA OOM on GPU".into())).await;
        std::thread::sleep(std::time::Duration::from_millis(1));
        store.insert(BlackboardItem::new("bob".into(), "b".into(), "CUDA fix for GPU OOM issue".into())).await;
        // "CUDA OOM GPU" — bob matches 3 terms, alice matches 3 terms, bob is newer
        let results = store.search("CUDA OOM GPU", 0).await;
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_post_rate_limit() {
        let store = BlackboardStore::new(true);
        for i in 0..10 {
            let item = BlackboardItem::new("alice".into(), "a".into(), format!("msg {i}"));
            assert!(store.post(item).await.is_ok());
        }
        // 11th should be rate limited
        let item = BlackboardItem::new("alice".into(), "a".into(), "one too many".into());
        assert!(store.post(item).await.is_err());
    }

    #[tokio::test]
    async fn test_post_text_too_long() {
        let store = BlackboardStore::new(true);
        let long_text = "x".repeat(5000);
        let item = BlackboardItem::new("alice".into(), "a".into(), long_text);
        let result = store.post(item).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too long"));
    }

    #[test]
    fn test_shannon_entropy() {
        // Low entropy (all same char)
        assert!(shannon_entropy("aaaaaaaaa") < 1.0);
        // High entropy (random-looking)
        assert!(shannon_entropy("aB3xK9mQ2pL7wR4y") > 3.5);
    }
}
