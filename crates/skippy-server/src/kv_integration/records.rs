use skippy_runtime::ActivationFrame;

use crate::kv_proto::{KvPageManifest, PageIdentity};
use skippy_cache::{CacheBytesReconstructStats, CacheDedupeStats, ExactStatePayloadKind};

#[derive(Debug, Clone)]
pub struct PrefillKvIdentity {
    pub identity: PageIdentity,
    pub page_id: String,
    pub namespace: String,
    pub token_ids: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct LookupBatchOutcome {
    pub pages: Vec<KvPageManifest>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RecordPageOutcome {
    pub manifest: KvPageManifest,
    pub write_ms: f64,
    pub checksum_ms: f64,
}

#[derive(Debug)]
pub struct AttachedPage {
    pub manifest: KvPageManifest,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixRestore {
    pub page_id: String,
    pub token_count: usize,
    pub seq_id: i32,
    pub entries: usize,
}

#[derive(Debug, Clone)]
pub struct ResidentPrefixRecord {
    pub page_id: String,
    pub token_count: usize,
    pub seq_id: i32,
    pub stored: bool,
    pub evicted_entries: usize,
    pub evicted_tokens: u64,
    pub entries: usize,
    pub resident_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct ResidentActivationRestore {
    pub identity: PrefillKvIdentity,
    pub page_id: String,
    pub token_count: usize,
    pub payload_bytes: usize,
    pub entries: usize,
    pub frame: ActivationFrame,
}

#[derive(Debug, Clone)]
pub struct ResidentActivationRecord {
    pub page_id: String,
    pub token_count: usize,
    pub payload_bytes: usize,
    pub evicted_entries: usize,
    pub evicted_bytes: u64,
    pub entries: usize,
    pub resident_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ExactStateRestore {
    pub page_id: String,
    pub token_count: usize,
    pub payload_kind: ExactStatePayloadKind,
    pub logical_bytes: u64,
    pub entries: usize,
    pub reconstruct_ms: f64,
    pub reconstruct_bytes: u64,
    pub reconstruct_blocks: usize,
    pub lookup_ms: f64,
    pub kv_import_ms: f64,
    pub recurrent_import_ms: f64,
    /// Where the state came from: `"radix"` (RAM) or `"l3"` (disk).
    pub source: &'static str,
    /// Store-side cost of an L3 fill (read, digest verification, assembly),
    /// kept apart from the runtime import above so restore thresholds come
    /// from real numbers. Zero for a radix hit.
    pub fill_ms: f64,
    /// Whether an L3 fill's radix re-warm record was accepted by the worker.
    pub rewarm_enqueued: bool,
}

#[derive(Debug, Clone)]
pub struct ExactStateRecord {
    pub page_id: String,
    pub token_count: usize,
    pub payload_kind: ExactStatePayloadKind,
    pub stored: bool,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub entries: usize,
    pub evicted_entries: usize,
    pub evicted_logical_bytes: u64,
    pub dedupe: CacheDedupeStats,
}

impl AttachedPage {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

pub(crate) fn add_reconstruct_stats(
    total_ms: &mut f64,
    total_bytes: &mut u64,
    total_blocks: &mut usize,
    stats: CacheBytesReconstructStats,
) {
    *total_ms += stats.reconstruct_ms;
    *total_bytes = total_bytes.saturating_add(stats.reconstruct_bytes);
    *total_blocks = total_blocks.saturating_add(stats.reconstruct_blocks);
}
