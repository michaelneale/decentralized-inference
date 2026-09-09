//! Read-only seam for future cache sources.
//!
//! Disk remains the writable L3 tier and owns admission. A network transport
//! can later implement these traits, verify the same manifest/segment format,
//! and commit fetched objects through the local manager before runtime fill.

use anyhow::Result;

use crate::l3::{HandoffManifest, HandoffSegmentStore};

pub trait ManifestSource: Send + Sync {
    fn recorded_prefix_lengths(&self, namespace_key: &str) -> Result<Vec<u64>>;

    fn manifest_for_prefix(
        &self,
        namespace_key: &str,
        token_len: u64,
        prefix_key: &str,
    ) -> Result<Option<HandoffManifest>>;

    fn load_manifest(&self, payload_digest: &str) -> Result<Option<HandoffManifest>>;
}

pub trait SegmentSource: Send + Sync {
    fn read_segment(&self, digest: &str) -> Result<Option<Vec<u8>>>;
}

impl ManifestSource for HandoffSegmentStore {
    fn recorded_prefix_lengths(&self, namespace_key: &str) -> Result<Vec<u64>> {
        HandoffSegmentStore::recorded_prefix_lengths(self, namespace_key)
    }

    fn manifest_for_prefix(
        &self,
        namespace_key: &str,
        token_len: u64,
        prefix_key: &str,
    ) -> Result<Option<HandoffManifest>> {
        HandoffSegmentStore::manifest_for_prefix(self, namespace_key, token_len, prefix_key)
    }

    fn load_manifest(&self, payload_digest: &str) -> Result<Option<HandoffManifest>> {
        match HandoffSegmentStore::load_manifest(self, payload_digest) {
            Ok(manifest) => Ok(Some(manifest)),
            Err(error)
                if error
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
}

impl SegmentSource for HandoffSegmentStore {
    fn read_segment(&self, digest: &str) -> Result<Option<Vec<u8>>> {
        match HandoffSegmentStore::read_segment(self, digest) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error)
                if error
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_store_satisfies_future_read_source_contracts() {
        fn assert_sources<T: ManifestSource + SegmentSource>() {}
        assert_sources::<HandoffSegmentStore>();
    }
}
