pub mod config;
pub mod fsinfo;
pub mod identity;
pub mod l3;
pub mod manager;
pub mod payload;
pub mod radix;
pub mod resident;
pub mod source;
pub mod tier;

pub use config::{ResidentCacheConfig, SparseCheckpointPolicy};
pub use identity::{
    ExactStateIdentityParams, NATIVE_KV_DTYPE, NATIVE_KV_RUNTIME_ABI_VERSION, PrefixIdentity,
    activation_page_id, exact_state_identity, exact_state_identity_for_stage,
    numerical_model_identity_for_stage, prefix_hash, prefix_hash_with_namespace, prefix_identity,
    prefix_identity_with_namespace, prefix_namespace_hash,
};
pub use l3::{
    GeometryBlock, GeometryKind, HandoffManifest, HandoffSegmentRef, HandoffSegmentStore,
    MANIFEST_VERSION, ManifestPin, PayloadGeometry, Reservation, SegmentHold, SegmentPut,
    StoreLimits, StoreReconciliation, StoreUsage, StoredSegment, WriteRefusal, segment_digest,
};
pub use manager::{
    L3ActivitySnapshot, L3CacheManager, L3EffectiveState, L3EffectiveStatus, L3InventoryEntry,
    L3StateReason, L3StateTransition,
};
pub use payload::{
    CacheBlobStore, CacheBytes, CacheBytesReconstructStats, CacheDedupeStats, ExactStatePayload,
    ExactStatePayloadKind,
};
pub use radix::{
    RadixEviction, RadixEvictionCandidate, RadixMatch, UnifiedRadixCache, UnifiedRadixCacheStats,
};
pub use resident::{
    ResidentActivationCache, ResidentActivationLookup, ResidentActivationRecordOutcome,
    ResidentActivationStats,
};
pub use source::{ManifestSource, SegmentSource};

pub use tier::{L3Fill, L3Location, L3Status, L3Tier, l3_namespace_key, l3_prefix_key};

/// llama.cpp's hard sequence-id capacity for one context.
pub const LLAMA_MAX_SEQ: i32 = 256;

#[cfg(test)]
mod legacy_prefix_index_absence_tests {
    use std::path::Path;

    #[test]
    fn removed_flat_prefix_indexes_cannot_reappear() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for removed in ["exact_state.rs", "resident/prefix.rs"] {
            assert!(
                !source.join(removed).exists(),
                "removed flat prefix index reappeared: {removed}"
            );
        }
    }
}
