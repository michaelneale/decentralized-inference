use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Instant,
};

use anyhow::{Result, bail};

use super::CacheBytes;
use super::bytes::{CacheBlockBytes, CacheBlockRef, CacheBytesRepr};

const DEFAULT_BLOCK_SIZE_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub struct CacheBlobStore {
    block_size: usize,
    physical_bytes: u64,
    blocks: HashMap<String, CacheBlob>,
    storages: HashMap<usize, CacheStorage>,
}

impl Default for CacheBlobStore {
    fn default() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE_BYTES)
    }
}

#[derive(Debug)]
struct CacheBlob {
    bytes: Arc<RwLock<CacheBlockBytes>>,
    ref_count: u64,
}

#[derive(Debug)]
struct CacheStorage {
    /// Count the complete backing allocation, rather than its block slices,
    /// so the configured physical-byte bound remains honest.
    bytes: u64,
    block_count: usize,
    contiguous_owners: u64,
}

impl CacheBlobStore {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            physical_bytes: 0,
            blocks: HashMap::new(),
            storages: HashMap::new(),
        }
    }

    pub fn store_bytes(&mut self, bytes: CacheBytes) -> (CacheBytes, CacheDedupeStats) {
        let len = bytes.len;
        let bytes = match bytes.repr {
            CacheBytesRepr::Inline(bytes) => bytes,
            CacheBytesRepr::Blocks {
                blocks,
                contiguous: _,
            } => {
                let mut stats = CacheDedupeStats::default();
                let mut canonical = Vec::with_capacity(blocks.len());
                for block in blocks.iter() {
                    stats.block_count = stats.block_count.saturating_add(1);
                    if !self.blocks.contains_key(&block.hash) {
                        self.register_storage_block(&block.bytes);
                        stats.new_block_count = stats.new_block_count.saturating_add(1);
                        self.blocks.insert(
                            block.hash.clone(),
                            CacheBlob {
                                bytes: block.bytes.clone(),
                                ref_count: 0,
                            },
                        );
                    }
                    let entry = self
                        .blocks
                        .get_mut(&block.hash)
                        .expect("cache block inserted or already present");
                    if entry.ref_count > 0 {
                        stats.reused_block_count = stats.reused_block_count.saturating_add(1);
                    }
                    entry.ref_count = entry.ref_count.saturating_add(1);
                    canonical.push(CacheBlockRef::new(block.hash.clone(), entry.bytes.clone()));
                }
                let contiguous = contiguous_storage(&canonical, len);
                self.register_contiguous_owner(contiguous.as_ref());
                return (
                    CacheBytes {
                        len,
                        repr: CacheBytesRepr::Blocks {
                            blocks: canonical.into(),
                            contiguous,
                        },
                    },
                    stats,
                );
            }
        };
        let mut blocks = Vec::new();
        let started = Instant::now();
        let mut stats = CacheDedupeStats {
            hash_bytes: bytes.len() as u64,
            ..CacheDedupeStats::default()
        };
        let chunks = (0..bytes.len())
            .step_by(self.block_size)
            .map(|start| {
                let end = start.saturating_add(self.block_size).min(bytes.len());
                let hash = blake3::hash(&bytes[start..end]).to_hex().to_string();
                (hash, start..end)
            })
            .collect::<Vec<_>>();
        let unique_hashes = chunks
            .iter()
            .map(|(hash, _)| hash.as_str())
            .collect::<std::collections::HashSet<_>>();
        let preserve_contiguous = unique_hashes.len() == chunks.len()
            && chunks
                .iter()
                .all(|(hash, _)| !self.blocks.contains_key(hash));
        for (hash, range) in chunks {
            stats.block_count = stats.block_count.saturating_add(1);
            if !self.blocks.contains_key(&hash) {
                let block_bytes = if preserve_contiguous {
                    CacheBlockBytes::new(Arc::clone(&bytes), range.clone())
                } else {
                    let standalone = Arc::new(bytes[range.clone()].to_vec());
                    let len = standalone.len();
                    CacheBlockBytes::new(standalone, 0..len)
                };
                let block_bytes = Arc::new(RwLock::new(block_bytes));
                self.register_storage_block(&block_bytes);
                stats.new_block_count = stats.new_block_count.saturating_add(1);
                self.blocks.insert(
                    hash.clone(),
                    CacheBlob {
                        bytes: block_bytes,
                        ref_count: 0,
                    },
                );
            }
            let entry = self
                .blocks
                .get_mut(&hash)
                .expect("cache block inserted or already present");
            if entry.ref_count > 0 {
                stats.reused_block_count = stats.reused_block_count.saturating_add(1);
            }
            entry.ref_count = entry.ref_count.saturating_add(1);
            blocks.push(CacheBlockRef::new(hash, entry.bytes.clone()));
        }
        stats.hash_ms = started.elapsed().as_secs_f64() * 1000.0;
        let contiguous = contiguous_storage(&blocks, bytes.len() as u64);
        self.register_contiguous_owner(contiguous.as_ref());
        (
            CacheBytes::blocks(bytes.len() as u64, blocks, contiguous),
            stats,
        )
    }

    fn register_storage_block(&mut self, bytes: &Arc<RwLock<CacheBlockBytes>>) {
        let bytes = bytes
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let storage = self.storages.entry(bytes.storage_key()).or_insert_with(|| {
            let storage_bytes = bytes.storage_len() as u64;
            self.physical_bytes = self.physical_bytes.saturating_add(storage_bytes);
            CacheStorage {
                bytes: storage_bytes,
                block_count: 0,
                contiguous_owners: 0,
            }
        });
        storage.block_count = storage.block_count.saturating_add(1);
    }

    fn register_contiguous_owner(&mut self, contiguous: Option<&Arc<Vec<u8>>>) {
        let Some(contiguous) = contiguous else {
            return;
        };
        let storage_key = Arc::as_ptr(contiguous) as usize;
        let storage = self
            .storages
            .get_mut(&storage_key)
            .expect("contiguous payload storage must own at least one cache block");
        storage.contiguous_owners = storage.contiguous_owners.saturating_add(1);
    }

    pub fn release_bytes(&mut self, bytes: &CacheBytes) -> Result<()> {
        self.release_bytes_batch(&[bytes])
    }

    pub(crate) fn release_bytes_batch(&mut self, payloads: &[&CacheBytes]) -> Result<()> {
        let mut releases = HashMap::<String, u64>::new();
        let mut contiguous_releases = HashMap::<usize, u64>::new();
        for bytes in payloads {
            for hash in bytes.block_hashes() {
                let count = releases.entry(hash.to_string()).or_default();
                *count = count
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("cache blob release count overflow"))?;
            }
            if let CacheBytesRepr::Blocks {
                contiguous: Some(contiguous),
                ..
            } = &bytes.repr
            {
                let count = contiguous_releases
                    .entry(Arc::as_ptr(contiguous) as usize)
                    .or_default();
                *count = count.checked_add(1).ok_or_else(|| {
                    anyhow::anyhow!("cache contiguous owner release count overflow")
                })?;
            }
        }
        for (hash, count) in &releases {
            let Some(entry) = self.blocks.get(hash) else {
                bail!("cache blob release references missing block {hash}");
            };
            if entry.ref_count < *count {
                bail!(
                    "cache blob release underflow for block {hash}: refs={} releases={count}",
                    entry.ref_count
                );
            }
        }
        let accounted_physical_bytes = self
            .storages
            .values()
            .map(|storage| storage.bytes)
            .try_fold(0u64, u64::checked_add)
            .ok_or_else(|| anyhow::anyhow!("cache blob physical byte count overflow"))?;
        if accounted_physical_bytes != self.physical_bytes {
            bail!(
                "cache blob physical byte accounting underflow: bytes={} storages={accounted_physical_bytes}",
                self.physical_bytes
            );
        }
        for (storage_key, owner_releases) in &contiguous_releases {
            let Some(storage) = self.storages.get(storage_key) else {
                bail!("cache contiguous release references missing storage {storage_key}");
            };
            if storage.contiguous_owners < *owner_releases {
                bail!(
                    "cache contiguous owner release underflow: owners={} releases={owner_releases}",
                    storage.contiguous_owners
                );
            }
        }

        let exhausted_contiguous_storages = contiguous_releases
            .iter()
            .filter_map(|(storage_key, releases)| {
                (self.storages[storage_key].contiguous_owners == *releases).then_some(*storage_key)
            })
            .collect::<std::collections::HashSet<_>>();
        for (storage_key, owner_releases) in contiguous_releases {
            self.storages
                .get_mut(&storage_key)
                .expect("contiguous release prevalidation guarantees storage presence")
                .contiguous_owners -= owner_releases;
        }
        let mut materializations = Vec::new();
        for (hash, count) in &releases {
            let entry = &self.blocks[hash];
            if entry.ref_count <= *count {
                continue;
            }
            let bytes = entry
                .bytes
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if exhausted_contiguous_storages.contains(&bytes.storage_key()) {
                materializations.push((entry.bytes.clone(), bytes.storage_key()));
            }
        }
        for (block, old_storage_key) in materializations {
            {
                let mut bytes = block
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let standalone = Arc::new(bytes.as_slice().to_vec());
                let len = standalone.len();
                *bytes = CacheBlockBytes::new(standalone, 0..len);
            }
            self.unregister_storage_block(old_storage_key)?;
            self.register_storage_block(&block);
        }

        for (hash, count) in releases {
            let removed = {
                let entry = self
                    .blocks
                    .get_mut(&hash)
                    .expect("release prevalidation guarantees block presence");
                entry.ref_count -= count;
                (entry.ref_count == 0).then(|| entry.bytes.clone())
            };
            if let Some(block) = removed {
                self.blocks.remove(&hash);
                let storage_key = block
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .storage_key();
                self.unregister_storage_block(storage_key)?;
            }
        }
        Ok(())
    }

    fn unregister_storage_block(&mut self, storage_key: usize) -> Result<()> {
        let storage = self
            .storages
            .get_mut(&storage_key)
            .ok_or_else(|| anyhow::anyhow!("cache block release references missing storage"))?;
        storage.block_count = storage
            .block_count
            .checked_sub(1)
            .ok_or_else(|| anyhow::anyhow!("cache storage block release underflow"))?;
        if storage.block_count == 0 {
            if storage.contiguous_owners != 0 {
                bail!("cache storage lost its blocks while contiguous owners remain");
            }
            let bytes = storage.bytes;
            self.storages.remove(&storage_key);
            self.physical_bytes = self
                .physical_bytes
                .checked_sub(bytes)
                .ok_or_else(|| anyhow::anyhow!("cache blob physical byte accounting underflow"))?;
        }
        Ok(())
    }

    pub fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn logical_ref_count(&self) -> u64 {
        self.blocks
            .values()
            .map(|block| block.ref_count)
            .fold(0, u64::saturating_add)
    }
}

fn contiguous_storage(blocks: &[CacheBlockRef], expected_len: u64) -> Option<Arc<Vec<u8>>> {
    let first = blocks.first()?;
    let first = first
        .bytes
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let storage = Arc::clone(&first.storage);
    let start = first.range.start;
    let mut end = start;
    drop(first);
    for block in blocks {
        let bytes = block
            .bytes
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !Arc::ptr_eq(&storage, &bytes.storage) || bytes.range.start != end {
            return None;
        }
        end = bytes.range.end;
    }
    let len = u64::try_from(end.checked_sub(start)?).ok()?;
    (start == 0 && len == expected_len && end == storage.len()).then_some(storage)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CacheDedupeStats {
    pub hash_ms: f64,
    pub hash_bytes: u64,
    pub block_count: usize,
    pub new_block_count: usize,
    pub reused_block_count: usize,
}

impl CacheDedupeStats {
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            hash_ms: self.hash_ms + other.hash_ms,
            hash_bytes: self.hash_bytes.saturating_add(other.hash_bytes),
            block_count: self.block_count.saturating_add(other.block_count),
            new_block_count: self.new_block_count.saturating_add(other.new_block_count),
            reused_block_count: self
                .reused_block_count
                .saturating_add(other.reused_block_count),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        borrow::Cow,
        collections::HashMap,
        sync::{Arc, Barrier},
        thread,
    };

    use crate::payload::{CacheBlobStore, ExactStatePayload, ExactStatePayloadKind};

    struct DeterministicRng(u64);

    impl DeterministicRng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }

        fn below(&mut self, ceiling: usize) -> usize {
            (self.next() as usize) % ceiling
        }
    }

    #[derive(Clone)]
    enum ExpectedPayload {
        Full(Vec<u8>),
        Recurrent(Vec<u8>),
        Composite { kv: Vec<u8>, recurrent: Vec<u8> },
    }

    fn state_machine_budget() -> (usize, usize) {
        let seeds = std::env::var("SKIPPY_CACHE_STATE_MACHINE_SEEDS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8)
            .clamp(1, 4_096);
        let steps = std::env::var("SKIPPY_CACHE_STATE_MACHINE_STEPS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(2_000)
            .clamp(1, 100_000);
        (seeds, steps)
    }

    fn random_bytes(rng: &mut DeterministicRng) -> Vec<u8> {
        let len = rng.below(16) + 1;
        (0..len)
            .map(|index| ((rng.below(6) + index % 3) as u8) * 17)
            .collect()
    }

    fn make_payload(rng: &mut DeterministicRng) -> (ExactStatePayload, ExpectedPayload) {
        match rng.below(3) {
            0 => {
                let bytes = random_bytes(rng);
                (
                    ExactStatePayload::full_state(bytes.clone()),
                    ExpectedPayload::Full(bytes),
                )
            }
            1 => {
                let bytes = random_bytes(rng);
                (
                    ExactStatePayload::recurrent_only(bytes.clone()),
                    ExpectedPayload::Recurrent(bytes),
                )
            }
            2 => {
                let kv = random_bytes(rng);
                let recurrent = random_bytes(rng);
                (
                    ExactStatePayload::kv_recurrent(kv.clone(), recurrent.clone()),
                    ExpectedPayload::Composite { kv, recurrent },
                )
            }
            _ => unreachable!(),
        }
    }

    fn components(expected: &ExpectedPayload) -> Vec<&[u8]> {
        match expected {
            ExpectedPayload::Full(bytes) | ExpectedPayload::Recurrent(bytes) => {
                vec![bytes.as_slice()]
            }
            ExpectedPayload::Composite { kv, recurrent } => {
                vec![kv.as_slice(), recurrent.as_slice()]
            }
        }
    }

    fn assert_payload(
        payload: &ExactStatePayload,
        expected: &ExpectedPayload,
        seed: u64,
        step: usize,
    ) {
        match expected {
            ExpectedPayload::Full(bytes) => {
                assert_eq!(payload.kind(), ExactStatePayloadKind::FullState);
                assert_eq!(
                    payload.full_state_bytes_timed().unwrap().0.as_ref(),
                    bytes,
                    "seed={seed:#x} step={step}"
                );
            }
            ExpectedPayload::Recurrent(bytes) => {
                assert_eq!(payload.kind(), ExactStatePayloadKind::RecurrentOnly);
                assert_eq!(
                    payload.recurrent_state_bytes().unwrap().as_ref(),
                    bytes,
                    "seed={seed:#x} step={step}"
                );
            }
            ExpectedPayload::Composite { kv, recurrent } => {
                assert_eq!(payload.kind(), ExactStatePayloadKind::KvRecurrent);
                assert_eq!(
                    payload.kv_bytes().unwrap().unwrap().as_ref(),
                    kv,
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    payload.recurrent_state_bytes().unwrap().as_ref(),
                    recurrent,
                    "seed={seed:#x} step={step}"
                );
            }
        }
    }

    fn payload_bytes(payload: &ExactStatePayload) -> Vec<&super::super::CacheBytes> {
        match payload {
            ExactStatePayload::FullState { bytes } => vec![bytes],
            ExactStatePayload::RecurrentOnly { recurrent } => vec![recurrent],
            ExactStatePayload::KvRecurrent { kv, recurrent } => vec![kv, recurrent],
        }
    }

    fn assert_storage_accounting(
        blobs: &CacheBlobStore,
        owners: &[Option<(ExactStatePayload, ExpectedPayload)>],
        seed: u64,
        step: usize,
    ) {
        let mut expected = HashMap::<usize, (u64, usize)>::new();
        for block in blobs.blocks.values() {
            let bytes = block
                .bytes
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let entry = expected
                .entry(bytes.storage_key())
                .or_insert((bytes.storage_len() as u64, 0));
            assert_eq!(entry.0, bytes.storage_len() as u64);
            entry.1 += 1;
        }
        assert_eq!(
            blobs.storages.len(),
            expected.len(),
            "seed={seed:#x} step={step}"
        );
        for (key, (bytes, block_count)) in &expected {
            let storage = &blobs.storages[key];
            assert_eq!(storage.bytes, *bytes, "seed={seed:#x} step={step}");
            assert_eq!(
                storage.block_count, *block_count,
                "seed={seed:#x} step={step}"
            );
        }
        let mut expected_contiguous_owners = HashMap::<usize, u64>::new();
        for (payload, _) in owners.iter().flatten() {
            for bytes in payload_bytes(payload) {
                if let super::CacheBytesRepr::Blocks {
                    contiguous: Some(contiguous),
                    ..
                } = &bytes.repr
                {
                    *expected_contiguous_owners
                        .entry(Arc::as_ptr(contiguous) as usize)
                        .or_default() += 1;
                }
            }
        }
        for (key, storage) in &blobs.storages {
            assert_eq!(
                storage.contiguous_owners,
                expected_contiguous_owners
                    .get(key)
                    .copied()
                    .unwrap_or_default(),
                "seed={seed:#x} step={step}"
            );
        }
        assert_eq!(
            blobs.physical_bytes(),
            expected.values().map(|(bytes, _)| bytes).sum::<u64>(),
            "seed={seed:#x} step={step}"
        );
    }

    #[test]
    fn block_store_dedupes_repeated_payload_blocks() {
        let mut blobs = CacheBlobStore::new(4);
        let first = ExactStatePayload::full_state(b"aaaabbbb".to_vec());
        let second = ExactStatePayload::full_state(b"aaaacccc".to_vec());

        let (first, first_stats) = first.dedupe_into(&mut blobs);
        let (second, second_stats) = second.dedupe_into(&mut blobs);

        assert_eq!(first_stats.new_block_count, 2);
        assert_eq!(second_stats.new_block_count, 1);
        assert_eq!(second_stats.reused_block_count, 1);
        assert_eq!(blobs.physical_bytes(), 12);
        assert_eq!(second.byte_len(), 8);

        let (first_bytes, first_reconstruct) = first.full_state_bytes_timed().unwrap();
        assert!(matches!(first_bytes, Cow::Borrowed(_)));
        assert_eq!(first_reconstruct.reconstruct_bytes, 0);
        assert_eq!(first_reconstruct.reconstruct_blocks, 0);

        let (second_bytes, second_reconstruct) = second.full_state_bytes_timed().unwrap();
        assert!(matches!(second_bytes, Cow::Owned(_)));
        assert_eq!(second_bytes.as_ref(), b"aaaacccc");
        assert_eq!(second_reconstruct.reconstruct_bytes, 8);
        assert_eq!(second_reconstruct.reconstruct_blocks, 2);

        first.release_from(&mut blobs).unwrap();
        // Releasing the last contiguous owner materializes only the shared
        // block, allowing the original backing allocation to be reclaimed.
        assert_eq!(blobs.physical_bytes(), 8);
        assert_eq!(
            second.full_state_bytes_timed().unwrap().0.as_ref(),
            b"aaaacccc"
        );
        second.release_from(&mut blobs).unwrap();
        assert_eq!(blobs.physical_bytes(), 0);
    }

    #[test]
    fn rededuping_block_payload_retains_a_logical_owner() {
        let mut blobs = CacheBlobStore::new(4);
        let original = ExactStatePayload::full_state(b"aaaabbbb".to_vec());
        let (first_owner, _) = original.dedupe_into(&mut blobs);
        let (second_owner, stats) = first_owner.clone().dedupe_into(&mut blobs);

        assert_eq!(stats.reused_block_count, 2);
        assert_eq!(blobs.physical_bytes(), 8);
        first_owner.release_from(&mut blobs).unwrap();
        assert_eq!(blobs.physical_bytes(), 8);
        let (bytes, reconstruct) = second_owner.full_state_bytes_timed().unwrap();
        assert_eq!(bytes.as_ref(), b"aaaabbbb");
        assert!(matches!(bytes, Cow::Borrowed(_)));
        assert_eq!(reconstruct.reconstruct_bytes, 0);
        second_owner.release_from(&mut blobs).unwrap();
        assert_eq!(blobs.physical_bytes(), 0);
    }

    #[test]
    fn partial_dedupe_restore_remains_valid_while_contiguous_owner_is_released() {
        let mut blobs = CacheBlobStore::new(4);
        let (first, _) =
            ExactStatePayload::full_state(b"aaaabbbb".to_vec()).dedupe_into(&mut blobs);
        let (second, _) =
            ExactStatePayload::full_state(b"aaaacccc".to_vec()).dedupe_into(&mut blobs);
        let reader_payload = second.clone();
        let barrier = Arc::new(Barrier::new(2));
        let reader_barrier = Arc::clone(&barrier);
        let reader = thread::spawn(move || {
            reader_barrier.wait();
            for _ in 0..1_000 {
                assert_eq!(
                    reader_payload.full_state_bytes_timed().unwrap().0.as_ref(),
                    b"aaaacccc"
                );
            }
        });

        barrier.wait();
        first.release_from(&mut blobs).unwrap();
        reader.join().unwrap();

        assert_eq!(
            second.full_state_bytes_timed().unwrap().0.as_ref(),
            b"aaaacccc"
        );
        assert_eq!(blobs.physical_bytes(), 8);
        second.release_from(&mut blobs).unwrap();
    }

    #[test]
    fn duplicate_release_is_reported_without_accounting_drift() {
        let mut blobs = CacheBlobStore::new(4);
        let (payload, _) =
            ExactStatePayload::full_state(b"aaaabbbb".to_vec()).dedupe_into(&mut blobs);

        payload.release_from(&mut blobs).unwrap();
        let error = payload.release_from(&mut blobs).unwrap_err();

        assert!(error.to_string().contains("missing block"));
        assert_eq!(blobs.physical_bytes(), 0);
        assert_eq!(blobs.block_count(), 0);
    }

    #[test]
    fn composite_release_prevalidates_every_component() {
        let mut blobs = CacheBlobStore::new(4);
        let (payload, _) =
            ExactStatePayload::kv_recurrent(b"aaaabbbb".to_vec(), b"ccccdddd".to_vec())
                .dedupe_into(&mut blobs);
        let ExactStatePayload::KvRecurrent { recurrent, .. } = &payload else {
            panic!("expected composite payload");
        };

        blobs.release_bytes(recurrent).unwrap();
        let physical_before = blobs.physical_bytes();
        let refs_before = blobs.logical_ref_count();

        let error = payload.release_from(&mut blobs).unwrap_err();

        assert!(error.to_string().contains("missing block"));
        assert_eq!(blobs.physical_bytes(), physical_before);
        assert_eq!(blobs.logical_ref_count(), refs_before);
        assert_eq!(blobs.block_count(), 2);
    }

    #[test]
    fn physical_accounting_underflow_does_not_mutate_references() {
        let mut blobs = CacheBlobStore::new(4);
        let (payload, _) =
            ExactStatePayload::full_state(b"aaaabbbb".to_vec()).dedupe_into(&mut blobs);
        let refs_before = blobs.logical_ref_count();
        blobs.physical_bytes = 0;

        let error = payload.release_from(&mut blobs).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("physical byte accounting underflow")
        );
        assert_eq!(blobs.logical_ref_count(), refs_before);
        assert_eq!(blobs.block_count(), 2);
    }

    #[test]
    fn randomized_payload_ownership_reconciles_after_every_operation() {
        let (seed_count, steps) = state_machine_budget();
        for seed_index in 0..seed_count {
            let seed = 0x83d2_74a9_5e10_b6c3_u64
                .wrapping_add((seed_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let mut rng = DeterministicRng(seed);
            let mut blobs = CacheBlobStore::new(4);
            let mut owners = Vec::<Option<(ExactStatePayload, ExpectedPayload)>>::new();

            for step in 0..steps {
                let live = owners
                    .iter()
                    .enumerate()
                    .filter_map(|(index, owner)| owner.as_ref().map(|_| index))
                    .collect::<Vec<_>>();
                let operation = if live.len() >= 64 { 3 } else { rng.below(6) };
                match operation {
                    0 | 1 => {
                        let (payload, expected) = make_payload(&mut rng);
                        let (payload, _) = payload.dedupe_into(&mut blobs);
                        owners.push(Some((payload, expected)));
                    }
                    2 if !live.is_empty() => {
                        let source = live[rng.below(live.len())];
                        let (payload, expected) =
                            owners[source].as_ref().expect("live owner index").clone();
                        let (payload, _) = payload.dedupe_into(&mut blobs);
                        owners.push(Some((payload, expected)));
                    }
                    3..=5 if !live.is_empty() => {
                        let victim = live[rng.below(live.len())];
                        let (payload, _) = owners[victim].take().expect("live owner index");
                        payload.release_from(&mut blobs).unwrap();
                    }
                    _ => {}
                }

                let mut expected_blocks = HashMap::<Vec<u8>, u64>::new();
                for (payload, expected) in owners.iter().flatten() {
                    assert_payload(payload, expected, seed, step);
                    for component in components(expected) {
                        for chunk in component.chunks(4) {
                            *expected_blocks.entry(chunk.to_vec()).or_default() += 1;
                        }
                    }
                }
                assert_eq!(
                    blobs.block_count(),
                    expected_blocks.len(),
                    "seed={seed:#x} step={step}"
                );
                assert_storage_accounting(&blobs, &owners, seed, step);
                assert!(
                    blobs.physical_bytes()
                        >= expected_blocks
                            .keys()
                            .map(|block| block.len() as u64)
                            .sum::<u64>(),
                    "seed={seed:#x} step={step}"
                );
                assert_eq!(
                    blobs.logical_ref_count(),
                    expected_blocks.values().sum::<u64>(),
                    "seed={seed:#x} step={step}"
                );
            }

            for owner in owners.into_iter().flatten() {
                owner.0.release_from(&mut blobs).unwrap();
            }
            assert_eq!(blobs.block_count(), 0, "seed={seed:#x}");
            assert_eq!(blobs.physical_bytes(), 0, "seed={seed:#x}");
            assert_eq!(blobs.logical_ref_count(), 0, "seed={seed:#x}");
        }
    }
}
