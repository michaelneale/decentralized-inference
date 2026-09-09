//! The L3 tier under the radix cache.
//!
//! `UnifiedRadixCache` holds `ExactStatePayload` entries in RAM (L1/L2);
//! this tier gives them a durable floor: spill a payload under its prefix
//! identity when the radix cache evicts it, and fill it back from local disk
//! on a radix miss. A later peer source may supply the same manifest/segment
//! format, but it must commit through the local manager before this tier can
//! fill it. State never crosses a numerical-identity boundary: spills stamp
//! both model and exact-state identities, and fills require both.

use std::sync::atomic::Ordering;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::l3::{
    HandoffManifest, HandoffSegmentRef, HandoffSegmentStore, MANIFEST_VERSION, PayloadGeometry,
    StoreLimits, StoreUsage, segment_digest,
};
use crate::manager::{L3ActivitySnapshot, L3CacheManager, L3EffectiveStatus};
use crate::payload::{ExactStatePayload, ExactStatePayloadKind};

/// Everything the status contract needs from the tier, in one read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct L3Status {
    pub model_identity: String,
    pub state_identity: String,
    pub effective: L3EffectiveStatus,
    pub format_version: u32,
    pub usage: StoreUsage,
    /// Manifests stamped with this tier's identity: what a restart can reuse.
    pub restorable_manifests: u64,
    pub restorable_tokens: u64,
    pub namespaces: u64,
    pub activity: L3ActivitySnapshot,
}

/// Key an L3 entry by the radix coordinates that identify it in RAM: the
/// namespace (which already binds the numerical stage identity) and the
/// exact token path.
pub fn l3_prefix_key(namespace: &str, token_ids: &[i32]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"l3-prefix-key-v1");
    hasher.update(namespace.as_bytes());
    for token_id in token_ids {
        hasher.update(&token_id.to_le_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

/// The namespace's own index key.
pub fn l3_namespace_key(namespace: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"l3-namespace-key-v1");
    hasher.update(namespace.as_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

/// A located entry: the cheap index-probe result, addressing exactly one
/// recorded manifest. Splitting locate from load lets callers single-flight
/// the expensive load on the entry itself (namespace + recorded length +
/// manifest key) rather than on the query's shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L3Location {
    pub namespace_key: String,
    pub prefix_key: String,
    pub token_count: u64,
    /// Manifest key (payload digest) — the identity of the physical entry;
    /// the correct single-flight claim key.
    pub manifest_key: String,
}

/// A successful fill from the tier.
pub struct L3Fill {
    pub payload: ExactStatePayload,
    /// How many of the query's leading tokens the filled state covers — the
    /// length the entry was recorded at, which may be shorter than the
    /// query (longest-recorded-prefix semantics, mirroring the radix).
    pub token_count: u64,
    pub kv_desc_json: Option<String>,
    pub payload_bytes: u64,
}

pub struct L3Tier {
    manager: L3CacheManager,
    model_identity: String,
    state_identity: String,
    segment_bytes: usize,
}

impl L3Tier {
    /// Open a tier capped by `budget_bytes` with no free-space reserve.
    /// Prefer [`Self::open_with_limits`].
    pub fn open(
        root: impl Into<std::path::PathBuf>,
        budget_bytes: u64,
        state_identity: String,
        segment_bytes: usize,
    ) -> Result<Self> {
        Self::open_with_limits(
            root,
            StoreLimits::new(budget_bytes, 0),
            state_identity,
            segment_bytes,
        )
    }

    pub fn open_with_limits(
        root: impl Into<std::path::PathBuf>,
        limits: StoreLimits,
        state_identity: String,
        segment_bytes: usize,
    ) -> Result<Self> {
        let manager = L3CacheManager::acquire(root.into(), limits)?;
        Ok(manager.tier(state_identity, segment_bytes))
    }

    pub fn open_with_identities(
        root: impl Into<std::path::PathBuf>,
        limits: StoreLimits,
        model_identity: String,
        state_identity: String,
        segment_bytes: usize,
    ) -> Result<Self> {
        let manager = L3CacheManager::acquire(root.into(), limits)?;
        Ok(manager.tier_for_model(model_identity, state_identity, segment_bytes))
    }

    pub(crate) fn from_manager(
        manager: L3CacheManager,
        model_identity: String,
        state_identity: String,
        segment_bytes: usize,
    ) -> Self {
        Self {
            manager,
            model_identity,
            state_identity,
            segment_bytes: segment_bytes.max(1),
        }
    }

    pub fn store(&self) -> &HandoffSegmentStore {
        self.manager.store()
    }

    pub fn manager(&self) -> &L3CacheManager {
        &self.manager
    }

    pub fn state_identity(&self) -> &str {
        &self.state_identity
    }

    pub fn model_identity(&self) -> &str {
        &self.model_identity
    }

    /// Point-in-time activity counters.
    pub fn activity(&self) -> L3ActivitySnapshot {
        self.manager.activity_snapshot()
    }

    /// The full status snapshot. One pass over the manifests; safe to call
    /// while serving, since it takes no lock a request path holds.
    pub fn status(&self) -> Result<L3Status> {
        let (restorable_manifests, restorable_tokens, _) = self.restorable_summary()?;
        Ok(L3Status {
            model_identity: self.model_identity.clone(),
            state_identity: self.state_identity.clone(),
            effective: self.manager.effective_status(),
            format_version: MANIFEST_VERSION,
            usage: self.store().usage()?,
            restorable_manifests: restorable_manifests as u64,
            restorable_tokens,
            namespaces: self.store().namespace_count()?,
            activity: self.activity(),
        })
    }

    /// Spill a radix payload under its (namespace, token-path) coordinates.
    /// Returns the manifest key. Entries at many lengths coexist — each is a
    /// complete state for its own length, which is what longest-prefix fill
    /// leans on.
    ///
    /// Zero-byte payloads are refused: a dense family whose native KV export
    /// was unavailable would otherwise spill nothing and restore as a bare
    /// position advance over missing state.
    pub fn spill(
        &self,
        namespace: &str,
        token_ids: &[i32],
        payload: &ExactStatePayload,
        kv_desc_json: Option<String>,
        geometry: Option<&PayloadGeometry>,
    ) -> Result<String> {
        let _operation = self.manager.operation_guard();
        let result = self.spill_inner(namespace, token_ids, payload, kv_desc_json, geometry);
        if let Err(error) = &result {
            self.manager.activity_counters().record_error(error);
        }
        result
    }

    fn spill_inner(
        &self,
        namespace: &str,
        token_ids: &[i32],
        payload: &ExactStatePayload,
        kv_desc_json: Option<String>,
        geometry: Option<&PayloadGeometry>,
    ) -> Result<String> {
        if payload.byte_len() == 0 {
            bail!(
                "refusing to spill an empty exact-state payload: no state component was exported"
            );
        }
        let token_count = token_ids.len() as u64;
        let (kv, recurrent): (Vec<u8>, Vec<u8>) = match payload.kind() {
            ExactStatePayloadKind::FullState => (
                payload
                    .full_state_bytes_timed()
                    .context("failed to reconstruct full state for spill")?
                    .0
                    .into_owned(),
                Vec::new(),
            ),
            ExactStatePayloadKind::RecurrentOnly => (
                Vec::new(),
                payload
                    .recurrent_state_bytes()
                    .context("failed to reconstruct recurrent state for spill")?
                    .into_owned(),
            ),
            ExactStatePayloadKind::KvRecurrent => (
                payload
                    .kv_bytes()
                    .context("failed to reconstruct KV bytes for spill")?
                    .map(|bytes| bytes.into_owned())
                    .unwrap_or_default(),
                payload
                    .recurrent_state_bytes()
                    .context("failed to reconstruct recurrent state for spill")?
                    .into_owned(),
            ),
        };

        // For full-state and recurrent-only exactly one component is populated,
        // and KV states reach gigabytes: concatenating would peak at twice the
        // payload for no benefit. Only a genuine composite needs the copy.
        let (kv_bytes, recurrent_bytes) = (kv.len() as u64, recurrent.len() as u64);
        let wire = if recurrent.is_empty() {
            kv
        } else if kv.is_empty() {
            recurrent
        } else {
            let mut wire = Vec::with_capacity(kv.len() + recurrent.len());
            wire.extend_from_slice(&kv);
            wire.extend_from_slice(&recurrent);
            wire
        };
        let payload_digest = segment_digest(&wire);

        let mut manifest = HandoffManifest::new_for_model(
            self.model_identity.clone(),
            self.state_identity.clone(),
            payload.kind().as_str().to_string(),
        );
        manifest.total_bytes = wire.len() as u64;
        manifest.payload_digest = payload_digest.clone();
        manifest.kv_bytes = kv_bytes;
        manifest.recurrent_bytes = recurrent_bytes;
        manifest.kv_desc_json = kv_desc_json;
        manifest.token_count = token_count;
        // Cut on the payload's own geometry when the caller knows it, so a
        // longer prefix reuses the segments of the shorter one it extends. A
        // geometry that does not describe these exact bytes is ignored rather
        // than trusted: mis-cutting would still reassemble, but silently write
        // the whole payload again every turn.
        let geometry = geometry.filter(|geometry| {
            let matches = geometry.matches(wire.len() as u64);
            if !matches {
                self.manager
                    .activity_counters()
                    .geometry_rejected
                    .fetch_add(1, Ordering::Relaxed);
            }
            matches
        });
        let cuts = match geometry {
            Some(geometry) => geometry.plan(self.segment_bytes as u64),
            None => {
                let mut cuts = Vec::new();
                let mut offset = 0u64;
                while (offset as usize) < wire.len() {
                    let len = self.segment_bytes.min(wire.len() - offset as usize) as u64;
                    cuts.push((offset, len, String::new()));
                    offset += len;
                }
                cuts
            }
        };
        let mut new_bytes = 0u64;
        // Held until after the commit below: until the manifest names them
        // these segments are unreferenced, and an eviction triggered by
        // another writer would collect them mid-build.
        let mut held = Vec::with_capacity(manifest.segments.capacity());
        for (index, (offset, len, label)) in cuts.into_iter().enumerate() {
            let start = usize::try_from(offset).context("segment offset exceeds usize")?;
            let end = start
                .checked_add(usize::try_from(len).context("segment length exceeds usize")?)
                .context("segment range overflows")?;
            let stored = match self.store().try_put_segment(&wire[start..end]) {
                Ok(Ok(stored)) => stored,
                Ok(Err(refusal)) => {
                    self.manager.record_write_refusal(refusal);
                    bail!("cannot store segment: {}", refusal.reason());
                }
                Err(error) => {
                    self.manager.record_storage_error();
                    return Err(error);
                }
            };
            if stored.put.new {
                new_bytes = new_bytes.saturating_add(stored.put.bytes);
            }
            manifest.segments.push(HandoffSegmentRef {
                index: index as u32,
                offset,
                bytes: len,
                digest: stored.digest.clone(),
                meta_json: (!label.is_empty()).then_some(label),
            });
            held.push(stored);
        }
        // Pin before publishing the manifest so another stage cannot evict
        // it in the gap between commit and prefix-link publication.
        let _manifest_pin = self.store().pin(&payload_digest);
        match self.store().try_commit(&manifest) {
            Ok(Ok(())) => {}
            Ok(Err(refusal)) => {
                self.manager.record_write_refusal(refusal);
                bail!(
                    "cannot commit manifest {}: {}",
                    manifest.payload_digest,
                    refusal.reason()
                );
            }
            Err(error) => {
                self.manager.record_storage_error();
                return Err(error);
            }
        }
        if let Err(error) = self.store().link_prefix(
            &l3_namespace_key(namespace),
            token_count,
            &l3_prefix_key(namespace, token_ids),
            &payload_digest,
        ) {
            self.manager.record_storage_error();
            return Err(error);
        }
        drop(held);
        self.manager
            .activity_counters()
            .writes
            .fetch_add(1, Ordering::Relaxed);
        self.manager
            .activity_counters()
            .bytes_written
            .fetch_add(new_bytes, Ordering::Relaxed);
        self.manager.record_successful_write();
        Ok(payload_digest)
    }

    /// Locate the longest recorded prefix of the query, mirroring the radix
    /// cache's longest-component-prefix semantics: probe recorded lengths
    /// for this namespace from longest to shortest (capped at `max_probes`),
    /// hashing the query's own leading tokens at each length — so a
    /// recorded entry only matches when the query genuinely starts with the
    /// tokens it was recorded for. Index probes only; the expensive load is
    /// `load`, so callers can single-flight on the returned entry.
    pub fn locate_longest(
        &self,
        namespace: &str,
        token_ids: &[i32],
        max_probes: usize,
    ) -> Result<Option<L3Location>> {
        let result = self.locate_longest_inner(namespace, token_ids, max_probes);
        match &result {
            Ok(Some(_)) => {
                self.manager
                    .activity_counters()
                    .hits
                    .fetch_add(1, Ordering::Relaxed);
            }
            Ok(None) => {
                self.manager
                    .activity_counters()
                    .misses
                    .fetch_add(1, Ordering::Relaxed);
            }
            Err(error) => self.manager.activity_counters().record_error(error),
        }
        result
    }

    fn locate_longest_inner(
        &self,
        namespace: &str,
        token_ids: &[i32],
        max_probes: usize,
    ) -> Result<Option<L3Location>> {
        let namespace_key = l3_namespace_key(namespace);
        let query_len = token_ids.len() as u64;
        let lengths = self.store().recorded_prefix_lengths(&namespace_key)?;
        for length in lengths
            .into_iter()
            .filter(|length| *length > 0 && *length <= query_len)
            .take(max_probes.max(1))
        {
            let prefix_key = l3_prefix_key(namespace, &token_ids[..length as usize]);
            let Some(manifest) =
                self.store()
                    .manifest_for_prefix(&namespace_key, length, &prefix_key)?
            else {
                continue;
            };
            if manifest.model_identity != self.model_identity
                || manifest.state_identity != self.state_identity
            {
                bail!(
                    "L3 entry for this prefix was spilled under state identity {} but the tier serves {}",
                    manifest.state_identity,
                    self.state_identity
                );
            }
            if manifest.token_count != length {
                bail!(
                    "L3 index length {length} disagrees with manifest token count {}",
                    manifest.token_count
                );
            }
            return Ok(Some(L3Location {
                namespace_key,
                prefix_key,
                token_count: length,
                manifest_key: manifest.payload_digest,
            }));
        }
        Ok(None)
    }

    /// Load a located entry. Memory bound: `assemble` materializes the
    /// payload once (`total_bytes`); the kv/recurrent split below reuses
    /// that allocation via `split_off`, so peak extra memory is the payload
    /// itself. Full-state fills are whole-blob by nature; kv-recurrent
    /// fills could stream per segment later.
    pub fn load(&self, location: &L3Location) -> Result<L3Fill> {
        let Some(_operation) = self.manager.try_operation_guard() else {
            bail!("disk cache lifecycle operation in progress");
        };
        // Pinned for the whole load: eviction under a concurrent spill must
        // not remove the segments this fill is assembling.
        let _pin = self.store().pin(&location.manifest_key);
        let result = self.load_inner(location);
        match &result {
            Ok(fill) => {
                self.manager
                    .activity_counters()
                    .fills
                    .fetch_add(1, Ordering::Relaxed);
                self.manager
                    .activity_counters()
                    .bytes_read
                    .fetch_add(fill.payload_bytes, Ordering::Relaxed);
            }
            Err(error) => self.manager.activity_counters().record_error(error),
        }
        result
    }

    fn load_inner(&self, location: &L3Location) -> Result<L3Fill> {
        let manifest = self.store().load_manifest(&location.manifest_key)?;
        if manifest.model_identity != self.model_identity
            || manifest.state_identity != self.state_identity
        {
            bail!(
                "L3 manifest {} was spilled under state identity {} but the tier serves {}",
                location.manifest_key,
                manifest.state_identity,
                self.state_identity
            );
        }
        let mut wire = self.store().assemble(&manifest)?;
        let payload_bytes = wire.len() as u64;
        let kv_bytes = usize::try_from(manifest.kv_bytes).context("kv bytes exceed usize")?;
        let payload = match manifest.payload_kind.as_str() {
            "full-state" => ExactStatePayload::full_state(wire),
            "recurrent-only" => ExactStatePayload::recurrent_only(wire),
            "kv-recurrent" => {
                // kv_bytes is an independent manifest field: a corrupt or
                // truncated one must be a recorded miss, not a panic on a
                // serving thread.
                if kv_bytes > wire.len() {
                    bail!(
                        "L3 manifest claims {kv_bytes} kv bytes but the payload holds {}",
                        wire.len()
                    );
                }
                let recurrent = wire.split_off(kv_bytes);
                ExactStatePayload::kv_recurrent(wire, recurrent)
            }
            other => bail!("L3 manifest holds unknown payload kind {other}"),
        };
        Ok(L3Fill {
            payload,
            token_count: manifest.token_count,
            kv_desc_json: manifest.kv_desc_json,
            payload_bytes,
        })
    }

    /// Locate and load in one step.
    pub fn fill_longest(
        &self,
        namespace: &str,
        token_ids: &[i32],
        max_probes: usize,
    ) -> Result<Option<L3Fill>> {
        match self.locate_longest(namespace, token_ids, max_probes)? {
            Some(location) => Ok(Some(self.load(&location)?)),
            None => Ok(None),
        }
    }

    /// What the tier can restore right now: (manifest count, restorable
    /// token total, segment footprint bytes). Startup visibility so warm
    /// state is never invisible.
    pub fn restorable_summary(&self) -> Result<(usize, u64, u64)> {
        let keys = self.store().list_manifests()?;
        let mut tokens = 0u64;
        let mut count = 0usize;
        for key in &keys {
            if let Ok(manifest) = self.store().load_manifest(key)
                && manifest.model_identity == self.model_identity
                && manifest.state_identity == self.state_identity
            {
                tokens = tokens.saturating_add(manifest.token_count);
                count += 1;
            }
        }
        Ok((count, tokens, self.store().segment_footprint_bytes()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l3::{GeometryBlock, GeometryKind};

    fn temp_root(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir()
            .join("skippy-l3-tier-tests")
            .join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    fn tier(name: &str, identity: &str) -> L3Tier {
        L3Tier::open(temp_root(name), 0, identity.to_string(), 4096).expect("open tier")
    }

    fn tokens(len: usize) -> Vec<i32> {
        (0..len as i32).collect()
    }

    /// Mirrors the runtime's export layout: every layer's K rows, then every
    /// layer's V rows, each run holding one row per token in token order.
    fn kv_geometry(
        layers: u32,
        rows: u64,
        k_stride: u64,
        v_stride: u64,
        window: u64,
    ) -> PayloadGeometry {
        let mut blocks = Vec::new();
        for layer in 0..layers {
            blocks.push(GeometryBlock {
                stride: k_stride,
                kind: GeometryKind::Key,
                layer,
                column: 0,
            });
        }
        for layer in 0..layers {
            blocks.push(GeometryBlock {
                stride: v_stride,
                kind: GeometryKind::Value,
                layer,
                column: 0,
            });
        }
        PayloadGeometry {
            blocks,
            rows,
            window_rows: window,
            tail_bytes: 0,
        }
    }

    /// A growing prefix keeps earlier tokens' rows byte-identical, so the
    /// export for turn N+1 is turn N's runs each extended in place.
    fn kv_payload(layers: u32, rows: u64, k_stride: u64, v_stride: u64) -> ExactStatePayload {
        let mut wire = Vec::new();
        for (kind, stride) in [(0u8, k_stride), (1u8, v_stride)] {
            for layer in 0..layers {
                for row in 0..rows {
                    // Stamp kind, layer and row into every row so no two runs
                    // can hash alike: content addressing would legitimately
                    // collapse them, and the test would be measuring that
                    // instead of the windowing.
                    for byte in 0..stride {
                        wire.push(match byte {
                            0 => kind,
                            1 => layer as u8,
                            2 => row as u8,
                            3 => (row >> 8) as u8,
                            _ => (byte as u8).wrapping_mul(7) ^ (row as u8).wrapping_mul(31),
                        });
                    }
                }
            }
        }
        ExactStatePayload::kv_recurrent(wire, Vec::new())
    }

    #[test]
    fn geometry_cut_segments_survive_prefix_growth() {
        // The measured failure: the export is layer-major, so a fixed byte cut
        // lands in a different place every turn and nothing is reused.
        let (layers, k_stride, v_stride, window) = (4u32, 64u64, 64u64, 8u64);
        let tier = tier("geometry-growth", "blake3:geometry");

        let first = kv_payload(layers, 32, k_stride, v_stride);
        tier.spill(
            "ns",
            &tokens(32),
            &first,
            None,
            Some(&kv_geometry(layers, 32, k_stride, v_stride, window)),
        )
        .expect("spill 32");
        let after_first = tier.status().expect("status").activity.bytes_written;
        assert_eq!(
            after_first,
            first.byte_len(),
            "first spill writes everything"
        );

        // Eight more tokens: only the new rows are new bytes.
        let second = kv_payload(layers, 40, k_stride, v_stride);
        tier.spill(
            "ns",
            &tokens(40),
            &second,
            None,
            Some(&kv_geometry(layers, 40, k_stride, v_stride, window)),
        )
        .expect("spill 40");
        let status = tier.status().expect("status");
        let physical = status.activity.bytes_written - after_first;
        let ideal = second.byte_len() - first.byte_len();
        assert_eq!(status.activity.geometry_rejected, 0);
        assert_eq!(
            physical, ideal,
            "growth wrote {physical} bytes for {ideal} bytes of new state"
        );

        // And the entry still reassembles to exactly what was spilled.
        let fill = tier
            .fill_longest("ns", &tokens(40), 64)
            .expect("fill")
            .expect("entry must be loadable");
        assert_eq!(
            fill.payload
                .kv_bytes()
                .expect("kv")
                .expect("some")
                .into_owned(),
            second.kv_bytes().expect("kv").expect("some").into_owned()
        );
    }

    #[test]
    fn fixed_offset_cutting_is_what_fails_the_gate() {
        // The same growth without geometry: this is the 8x the probe measured,
        // kept as a test so the regression is visible rather than remembered.
        let (layers, k_stride, v_stride) = (4u32, 64u64, 64u64);
        let tier = tier("geometry-fixed", "blake3:geometry");
        let first = kv_payload(layers, 32, k_stride, v_stride);
        tier.spill("ns", &tokens(32), &first, None, None)
            .expect("spill 32");
        let after_first = tier.status().expect("status").activity.bytes_written;
        let second = kv_payload(layers, 40, k_stride, v_stride);
        tier.spill("ns", &tokens(40), &second, None, None)
            .expect("spill 40");
        let physical = tier.status().expect("status").activity.bytes_written - after_first;
        let ideal = second.byte_len() - first.byte_len();
        assert!(
            physical > ideal * 2,
            "fixed cutting unexpectedly deduped: {physical} vs {ideal}"
        );
    }

    #[test]
    fn a_geometry_that_does_not_describe_the_payload_is_ignored() {
        let tier = tier("geometry-mismatch", "blake3:geometry");
        let payload = kv_payload(2, 16, 32, 32);
        // Wrong row count: cutting to it would silently mis-window every turn.
        let wrong = kv_geometry(2, 15, 32, 32, 4);
        tier.spill("ns", &tokens(16), &payload, None, Some(&wrong))
            .expect("spill still succeeds");
        let status = tier.status().expect("status");
        assert_eq!(status.activity.geometry_rejected, 1);
        let fill = tier
            .fill_longest("ns", &tokens(16), 8)
            .expect("fill")
            .expect("entry must be loadable");
        assert_eq!(fill.payload.byte_len(), payload.byte_len());
    }

    #[test]
    fn status_counters_reconcile_with_what_the_tier_did() {
        let root = temp_root("status");
        let tier = L3Tier::open(&root, 0, "blake3:status".to_string(), 4096).unwrap();
        // Every 4096-byte segment distinct, so within-spill dedup does not
        // hide what the duplicate spill below is meant to show.
        let bytes: Vec<u8> = (0..10_000u32)
            .map(|value| (value / 4096) as u8 ^ value as u8)
            .collect();
        let payload = ExactStatePayload::full_state(bytes);

        tier.spill("ns", &[1, 2, 3], &payload, None, None).unwrap();
        // Identical bytes again: a write, but no new segment bytes.
        tier.spill("ns", &[1, 2, 3], &payload, None, None).unwrap();
        assert!(
            tier.locate_longest("ns", &[1, 2, 3, 4], 8)
                .unwrap()
                .is_some()
        );
        assert!(tier.locate_longest("other", &[9], 8).unwrap().is_none());
        let location = tier.locate_longest("ns", &[1, 2, 3], 8).unwrap().unwrap();
        let fill = tier.load(&location).unwrap();

        let status = tier.status().unwrap();
        assert_eq!(status.format_version, MANIFEST_VERSION);
        assert_eq!(status.restorable_manifests, 1);
        assert_eq!(status.restorable_tokens, 3);
        assert_eq!(status.namespaces, 1);
        assert_eq!(status.activity.writes, 2);
        assert_eq!(
            status.activity.bytes_written, 10_000,
            "duplicate spill re-wrote bytes"
        );
        assert_eq!(status.activity.hits, 2);
        assert_eq!(status.activity.misses, 1);
        assert_eq!(status.activity.fills, 1);
        assert_eq!(status.activity.bytes_read, fill.payload_bytes);
        assert_eq!(status.activity.evictions, 0);
        assert_eq!(status.activity.corrupt_entries, 0);
        assert_eq!(status.activity.last_error, None);
    }

    #[test]
    fn spill_and_fill_roundtrip_all_payload_kinds() {
        let tier = tier("roundtrip", "blake3:identity-a");
        let cases = vec![
            (
                "namespace-full",
                ExactStatePayload::full_state((0..50_000u32).map(|v| v as u8).collect()),
            ),
            (
                "namespace-recurrent",
                ExactStatePayload::recurrent_only(vec![9u8; 10_000]),
            ),
            (
                "namespace-kv",
                ExactStatePayload::kv_recurrent(vec![1u8; 20_000], vec![2u8; 5_000]),
            ),
        ];
        for (namespace, payload) in cases {
            tier.spill(
                namespace,
                &tokens(512),
                &payload,
                Some("{\"desc\":1}".to_string()),
                None,
            )
            .expect("spill");
            let fill = tier
                .fill_longest(namespace, &tokens(512), 64)
                .expect("fill")
                .expect("tier must hold the prefix");
            assert_eq!(fill.token_count, 512);
            assert_eq!(fill.kv_desc_json.as_deref(), Some("{\"desc\":1}"));
            assert_eq!(fill.payload.kind(), payload.kind());
            match payload.kind() {
                ExactStatePayloadKind::KvRecurrent => {
                    assert_eq!(
                        fill.payload.kv_bytes().unwrap().unwrap().into_owned(),
                        payload.kv_bytes().unwrap().unwrap().into_owned()
                    );
                    assert_eq!(
                        fill.payload.recurrent_state_bytes().unwrap().into_owned(),
                        payload.recurrent_state_bytes().unwrap().into_owned()
                    );
                }
                ExactStatePayloadKind::RecurrentOnly => assert_eq!(
                    fill.payload.recurrent_state_bytes().unwrap().into_owned(),
                    payload.recurrent_state_bytes().unwrap().into_owned()
                ),
                ExactStatePayloadKind::FullState => assert_eq!(
                    fill.payload
                        .full_state_bytes_timed()
                        .unwrap()
                        .0
                        .into_owned(),
                    payload.full_state_bytes_timed().unwrap().0.into_owned()
                ),
            }
        }
    }

    /// The sacrament case: a later, longer prompt (multi-turn growth) must
    /// find the longest recorded shorter prefix, not just an exact match.
    #[test]
    fn longer_query_fills_from_longest_recorded_prefix() {
        let tier = tier("longest", "blake3:identity-a");
        tier.spill(
            "ns",
            &tokens(800),
            &ExactStatePayload::full_state(vec![1u8; 2048]),
            None,
            None,
        )
        .expect("spill 800");
        tier.spill(
            "ns",
            &tokens(1200),
            &ExactStatePayload::full_state(vec![2u8; 2048]),
            None,
            None,
        )
        .expect("spill 1200");

        // Query extends the 1200-token path: the longest entry wins.
        let fill = tier
            .fill_longest("ns", &tokens(1900), 64)
            .expect("fill")
            .expect("hit");
        assert_eq!(fill.token_count, 1200);
        assert_eq!(
            fill.payload
                .full_state_bytes_timed()
                .unwrap()
                .0
                .into_owned(),
            vec![2u8; 2048]
        );

        // Query between the two recorded lengths: the shorter entry wins.
        let fill = tier
            .fill_longest("ns", &tokens(1000), 64)
            .expect("fill")
            .expect("hit");
        assert_eq!(fill.token_count, 800);
    }

    /// A recorded length only matches when the query genuinely starts with
    /// the recorded tokens — a divergent prompt of the same length must
    /// miss, not corrupt.
    #[test]
    fn divergent_tokens_at_a_recorded_length_miss() {
        let tier = tier("divergent", "blake3:identity-a");
        tier.spill(
            "ns",
            &tokens(600),
            &ExactStatePayload::full_state(vec![3u8; 1024]),
            None,
            None,
        )
        .expect("spill");
        let mut divergent = tokens(600);
        divergent[100] = 999_999;
        assert!(
            tier.fill_longest("ns", &divergent, 64)
                .expect("fill")
                .is_none()
        );
    }

    #[test]
    fn unknown_namespace_fills_none() {
        let tier = tier("miss", "blake3:identity-a");
        assert!(
            tier.fill_longest("never-spilled", &tokens(64), 64)
                .expect("fill")
                .is_none()
        );
    }

    #[test]
    fn identity_mismatch_is_refused_not_served() {
        let root = temp_root("identity");
        let writer = L3Tier::open(&root, 0, "blake3:identity-a".to_string(), 4096).unwrap();
        writer
            .spill(
                "ns",
                &tokens(128),
                &ExactStatePayload::full_state(vec![5u8; 1024]),
                None,
                None,
            )
            .expect("spill");
        let reader = L3Tier::open(&root, 0, "blake3:identity-b".to_string(), 4096).unwrap();
        assert!(reader.fill_longest("ns", &tokens(128), 64).is_err());
    }

    #[test]
    fn model_identity_mismatch_is_refused_even_when_state_identity_matches() {
        let root = temp_root("model-identity-mismatch");
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
        let writer = manager.tier_for_model(
            "blake3:model-a".to_string(),
            "blake3:state".to_string(),
            4096,
        );
        writer
            .spill(
                "ns",
                &tokens(128),
                &ExactStatePayload::full_state(vec![1u8; 2048]),
                None,
                None,
            )
            .expect("spill");
        let reader = manager.tier_for_model(
            "blake3:model-b".to_string(),
            "blake3:state".to_string(),
            4096,
        );

        assert!(reader.fill_longest("ns", &tokens(128), 64).is_err());
    }

    #[test]
    fn respilling_a_length_supersedes_the_older_entry() {
        let tier = tier("supersede", "blake3:identity-a");
        tier.spill(
            "ns",
            &tokens(100),
            &ExactStatePayload::full_state(vec![1u8; 2048]),
            None,
            None,
        )
        .expect("first spill");
        tier.spill(
            "ns",
            &tokens(100),
            &ExactStatePayload::full_state(vec![2u8; 2048]),
            None,
            None,
        )
        .expect("second spill");
        let fill = tier
            .fill_longest("ns", &tokens(100), 64)
            .expect("fill")
            .expect("present");
        assert_eq!(fill.token_count, 100);
        assert_eq!(
            fill.payload
                .full_state_bytes_timed()
                .unwrap()
                .0
                .into_owned(),
            vec![2u8; 2048]
        );
    }

    /// A dense family whose KV export was unavailable must not persist an
    /// empty entry that would later restore as position-without-state.
    #[test]
    fn empty_payloads_are_refused_at_spill() {
        let tier = tier("empty", "blake3:identity-a");
        let empty = ExactStatePayload::kv_recurrent(Vec::new(), Vec::new());
        assert!(tier.spill("ns", &tokens(128), &empty, None, None).is_err());
        assert!(
            tier.fill_longest("ns", &tokens(128), 64)
                .expect("fill")
                .is_none()
        );
    }

    /// The locate/load split: locate is a cheap index probe addressing one
    /// physical entry (the single-flight claim key), and load returns the
    /// same payload fill_longest would.
    #[test]
    fn locate_addresses_one_entry_and_load_fetches_it() {
        let tier = tier("locate", "blake3:identity-a");
        tier.spill(
            "ns",
            &tokens(400),
            &ExactStatePayload::full_state(vec![7u8; 4096]),
            None,
            None,
        )
        .expect("spill");
        let location = tier
            .locate_longest("ns", &tokens(500), 64)
            .expect("locate")
            .expect("hit");
        assert_eq!(location.token_count, 400);
        // Two queries of different lengths that resolve to the same recorded
        // prefix share one claim key.
        let other = tier
            .locate_longest("ns", &tokens(450), 64)
            .expect("locate")
            .expect("hit");
        assert_eq!(location.manifest_key, other.manifest_key);
        let fill = tier.load(&location).expect("load");
        assert_eq!(fill.token_count, 400);
        assert_eq!(
            fill.payload
                .full_state_bytes_timed()
                .unwrap()
                .0
                .into_owned(),
            vec![7u8; 4096]
        );
    }

    #[test]
    fn restorable_summary_counts_matching_identity_only() {
        let root = temp_root("summary");
        let tier_a = L3Tier::open(&root, 0, "blake3:identity-a".to_string(), 4096).unwrap();
        let tier_b = L3Tier::open(&root, 0, "blake3:identity-b".to_string(), 4096).unwrap();
        tier_a
            .spill(
                "ns",
                &tokens(300),
                &ExactStatePayload::full_state(vec![1u8; 512]),
                None,
                None,
            )
            .unwrap();
        tier_b
            .spill(
                "ns",
                &tokens(700),
                &ExactStatePayload::full_state(vec![2u8; 512]),
                None,
                None,
            )
            .unwrap();
        let (count, restorable_tokens, footprint) = tier_a.restorable_summary().unwrap();
        assert_eq!(count, 1);
        assert_eq!(restorable_tokens, 300);
        assert!(footprint >= 1024);
    }
}
