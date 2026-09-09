//! Node-scoped ownership for the durable L3 cache root.

use std::{
    collections::{BTreeMap, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, LazyLock, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::{
    l3::{HandoffSegmentStore, StoreLimits, StoreReconciliation, StoreUsage, WriteRefusal},
    tier::L3Tier,
};

static ROOT_MANAGERS: LazyLock<Mutex<BTreeMap<PathBuf, Weak<L3ManagerInner>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));
const MAX_PENDING_STATE_TRANSITIONS: usize = 64;

/// What every stage attached to the node's L3 root has done since open.
#[derive(Debug, Default)]
pub(crate) struct L3Activity {
    pub(crate) fills: AtomicU64,
    pub(crate) hits: AtomicU64,
    pub(crate) misses: AtomicU64,
    pub(crate) writes: AtomicU64,
    pub(crate) geometry_rejected: AtomicU64,
    pub(crate) bytes_read: AtomicU64,
    pub(crate) bytes_written: AtomicU64,
    last_error: Mutex<Option<String>>,
}

impl L3Activity {
    pub(crate) fn record_error(&self, error: &anyhow::Error) {
        *self
            .last_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(format!("{error:#}"));
    }

    fn snapshot(&self, usage: Option<&StoreUsage>) -> L3ActivitySnapshot {
        L3ActivitySnapshot {
            fills: self.fills.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            writes: self.writes.load(Ordering::Relaxed),
            evictions: usage.map_or(0, |usage| usage.evicted_manifests),
            corrupt_entries: usage.map_or(0, |usage| usage.quarantined_objects),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            geometry_rejected: self.geometry_rejected.load(Ordering::Relaxed),
            last_error: self
                .last_error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
        }
    }
}

/// Point-in-time activity across every stage attached to one root manager.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct L3ActivitySnapshot {
    pub fills: u64,
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
    pub evictions: u64,
    pub corrupt_entries: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub geometry_rejected: u64,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct L3InventoryEntry {
    pub model_identity: String,
    pub state_identity: String,
    pub payload_kind: String,
    pub token_count: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum L3EffectiveState {
    Active,
    ReadOnlyLowSpace,
    Degraded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum L3StateReason {
    ReadOnlyLowSpace,
    InsufficientSpace,
    StorageError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct L3EffectiveStatus {
    pub state: L3EffectiveState,
    pub reason: Option<L3StateReason>,
}

impl Default for L3EffectiveStatus {
    fn default() -> Self {
        Self {
            state: L3EffectiveState::Active,
            reason: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct L3StateTransition {
    pub previous: L3EffectiveStatus,
    pub current: L3EffectiveStatus,
}

#[derive(Debug)]
struct L3ManagerInner {
    store: Arc<HandoffSegmentStore>,
    activity: Arc<L3Activity>,
    fill_claims: Arc<Mutex<std::collections::BTreeSet<String>>>,
    record_claims: Mutex<BTreeMap<String, Arc<Mutex<std::collections::BTreeSet<String>>>>>,
    effective: Mutex<L3EffectiveStatus>,
    transitions: Mutex<VecDeque<L3StateTransition>>,
    operations: RwLock<()>,
    reconciliation: StoreReconciliation,
}

/// The single physical owner of a node-local L3 root.
///
/// Clones are cheap stage handles into the same reservation, pin, lifecycle,
/// activity, and filesystem-lock domain.
#[derive(Clone, Debug)]
pub struct L3CacheManager {
    inner: Arc<L3ManagerInner>,
}

impl L3CacheManager {
    /// Acquire the manager for `root`, reusing the live node owner when one
    /// exists. A second process is rejected by the store's root lock.
    pub fn acquire(root: impl AsRef<Path>, limits: StoreLimits) -> Result<Self> {
        let root = canonical_cache_root(root.as_ref())?;
        let mut managers = ROOT_MANAGERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // An entry whose strong count has reached zero is not necessarily gone:
        // the owning thread decrements the count before dropping the inner
        // value, and the store's root `flock` is only released by that drop.
        let expiring_owner = managers
            .get(&root)
            .is_some_and(|manager| manager.strong_count() == 0);
        managers.retain(|_, manager| manager.strong_count() > 0);
        if let Some(inner) = managers.get(&root).and_then(Weak::upgrade) {
            if inner.store.limits() != limits {
                bail!(
                    "cache root {} is already open with different limits",
                    root.display()
                );
            }
            return Ok(Self { inner });
        }

        // Taking the root lock while the previous in-process owner is still
        // unwinding would report the root as owned by another manager, which
        // is a false answer: no other process holds it. Give that drop a
        // bounded moment to finish rather than failing the caller.
        let store = Arc::new(open_store_for_acquire(&root, limits, expiring_owner)?);
        let reconciliation = store.reconcile_startup()?;
        let inner = Arc::new(L3ManagerInner {
            store,
            activity: Arc::new(L3Activity::default()),
            fill_claims: Arc::new(Mutex::new(std::collections::BTreeSet::new())),
            record_claims: Mutex::new(BTreeMap::new()),
            effective: Mutex::new(L3EffectiveStatus::default()),
            transitions: Mutex::new(VecDeque::new()),
            operations: RwLock::new(()),
            reconciliation,
        });
        managers.insert(root, Arc::downgrade(&inner));
        Ok(Self { inner })
    }

    pub fn tier(&self, state_identity: String, segment_bytes: usize) -> L3Tier {
        self.tier_for_model(state_identity.clone(), state_identity, segment_bytes)
    }

    pub fn tier_for_model(
        &self,
        model_identity: String,
        state_identity: String,
        segment_bytes: usize,
    ) -> L3Tier {
        L3Tier::from_manager(self.clone(), model_identity, state_identity, segment_bytes)
    }

    pub fn root(&self) -> &Path {
        self.inner.store.root()
    }

    pub fn limits(&self) -> StoreLimits {
        self.inner.store.limits()
    }

    pub fn update_limits(&self, limits: StoreLimits) -> Result<StoreLimits> {
        let _lifecycle = self.lifecycle_guard();
        self.inner.store.update_limits(limits)
    }

    pub fn reconciliation(&self) -> StoreReconciliation {
        self.inner.reconciliation
    }

    pub fn usage(&self) -> Result<StoreUsage> {
        self.inner.store.usage()
    }

    pub fn activity(&self) -> Result<L3ActivitySnapshot> {
        let usage = self.usage()?;
        Ok(self.inner.activity.snapshot(Some(&usage)))
    }

    pub fn inventory(&self) -> Result<Vec<L3InventoryEntry>> {
        let mut inventory = self
            .inner
            .store
            .list_manifests()?
            .into_iter()
            .filter_map(|key| self.inner.store.load_manifest(&key).ok())
            .map(|manifest| L3InventoryEntry {
                model_identity: manifest.model_identity,
                state_identity: manifest.state_identity,
                payload_kind: manifest.payload_kind,
                token_count: manifest.token_count,
                total_bytes: manifest.total_bytes,
            })
            .collect::<Vec<_>>();
        inventory.sort_by(|left, right| {
            left.model_identity
                .cmp(&right.model_identity)
                .then_with(|| left.state_identity.cmp(&right.state_identity))
                .then_with(|| right.token_count.cmp(&left.token_count))
        });
        Ok(inventory)
    }

    pub fn effective_status(&self) -> L3EffectiveStatus {
        *self
            .inner
            .effective
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn take_state_transitions(&self) -> Vec<L3StateTransition> {
        self.inner
            .transitions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
            .collect()
    }

    pub fn prune_to(&self, target_bytes: u64) -> Result<u64> {
        let _lifecycle = self.lifecycle_guard();
        self.inner.store.prune_to(target_bytes)
    }

    pub fn prune_model_to(&self, model_identity: &str, target_bytes: u64) -> Result<u64> {
        let _lifecycle = self.lifecycle_guard();
        self.inner
            .store
            .prune_model_to(model_identity, target_bytes)
    }

    pub fn clear(&self) -> Result<u64> {
        let _lifecycle = self.lifecycle_guard();
        self.inner.store.clear()
    }

    pub fn clear_model(&self, model_identity: &str) -> Result<u64> {
        let _lifecycle = self.lifecycle_guard();
        self.inner.store.clear_model(model_identity)
    }

    pub fn shares_root_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Manifest keys currently being filled from this root. Keeping claims
    /// here makes single-flight node-wide instead of duplicating physical
    /// reads when placement-equivalent stages miss at the same time.
    pub fn fill_claims(&self) -> Arc<Mutex<std::collections::BTreeSet<String>>> {
        self.inner.fill_claims.clone()
    }

    /// Record claims are shared only by stages with the same full numerical
    /// state identity. This prevents duplicate exports and temporary writes
    /// across placement replicas without suppressing a different payload or
    /// stage layout that happens to use the same radix page id.
    pub fn record_claims(
        &self,
        state_identity: &str,
    ) -> Arc<Mutex<std::collections::BTreeSet<String>>> {
        self.inner
            .record_claims
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(state_identity.to_string())
            .or_default()
            .clone()
    }

    pub(crate) fn store(&self) -> &HandoffSegmentStore {
        &self.inner.store
    }

    pub(crate) fn activity_counters(&self) -> &L3Activity {
        &self.inner.activity
    }

    pub(crate) fn activity_snapshot(&self) -> L3ActivitySnapshot {
        let usage = self.usage().ok();
        self.inner.activity.snapshot(usage.as_ref())
    }

    pub(crate) fn record_write_refusal(&self, refusal: WriteRefusal) {
        let next = match refusal {
            WriteRefusal::SkippedOversize => return,
            WriteRefusal::ReadOnlyLowSpace => L3EffectiveStatus {
                state: L3EffectiveState::ReadOnlyLowSpace,
                reason: Some(L3StateReason::ReadOnlyLowSpace),
            },
            WriteRefusal::InsufficientSpace => L3EffectiveStatus {
                state: L3EffectiveState::Degraded,
                reason: Some(L3StateReason::InsufficientSpace),
            },
        };
        self.transition_to(next);
    }

    pub(crate) fn record_successful_write(&self) {
        let current = self.effective_status();
        if matches!(
            current.reason,
            Some(L3StateReason::ReadOnlyLowSpace | L3StateReason::InsufficientSpace)
        ) {
            self.transition_to(L3EffectiveStatus::default());
        }
    }

    pub(crate) fn record_storage_error(&self) {
        self.transition_to(L3EffectiveStatus {
            state: L3EffectiveState::Degraded,
            reason: Some(L3StateReason::StorageError),
        });
    }

    fn transition_to(&self, next: L3EffectiveStatus) {
        let mut current = self
            .inner
            .effective
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *current == next {
            return;
        }
        let previous = *current;
        *current = next;
        drop(current);
        let mut transitions = self
            .inner
            .transitions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if transitions.len() == MAX_PENDING_STATE_TRANSITIONS {
            transitions.pop_front();
        }
        transitions.push_back(L3StateTransition {
            previous,
            current: next,
        });
    }

    pub(crate) fn operation_guard(&self) -> RwLockReadGuard<'_, ()> {
        self.inner
            .operations
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn try_operation_guard(&self) -> Option<RwLockReadGuard<'_, ()>> {
        match self.inner.operations.try_read() {
            Ok(guard) => Some(guard),
            Err(std::sync::TryLockError::Poisoned(error)) => Some(error.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        }
    }

    fn lifecycle_guard(&self) -> RwLockWriteGuard<'_, ()> {
        self.inner
            .operations
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Open the store, retrying briefly when the previous in-process owner for
/// this root is still being dropped.
///
/// Only the handoff window is retried. A root genuinely held by another
/// process fails with the same error it always did, one short delay later.
fn open_store_for_acquire(
    root: &Path,
    limits: StoreLimits,
    expiring_owner: bool,
) -> Result<HandoffSegmentStore> {
    const HANDOFF_ATTEMPTS: u32 = 20;
    const HANDOFF_BACKOFF: std::time::Duration = std::time::Duration::from_millis(5);

    let attempts = if expiring_owner { HANDOFF_ATTEMPTS } else { 1 };
    let mut last = None;
    for attempt in 0..attempts {
        match HandoffSegmentStore::open_with_limits(root, limits) {
            Ok(store) => return Ok(store),
            Err(error) => {
                last = Some(error);
                if attempt + 1 < attempts {
                    std::thread::sleep(HANDOFF_BACKOFF);
                }
            }
        }
    }
    Err(last.expect("at least one attempt was made"))
}

fn canonical_cache_root(root: &Path) -> Result<PathBuf> {
    if !root.is_absolute() {
        bail!("cache root must be absolute: {}", root.display());
    }
    crate::fsinfo::refuse_symlink(root)?;
    fs::create_dir_all(root)
        .with_context(|| format!("failed to create cache root {}", root.display()))?;
    fs::canonicalize(root)
        .with_context(|| format!("failed to resolve cache root {}", root.display()))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;
    use crate::ExactStatePayload;

    fn temp_root(name: &str) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir()
            .join("skippy-l3-manager-tests")
            .join(format!(
                "{name}-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        let _ = fs::remove_dir_all(&root);
        root
    }

    #[test]
    fn one_live_manager_owns_each_root() {
        let root = temp_root("shared-owner");
        let limits = StoreLimits::new(1_000_000, 0);
        let first = L3CacheManager::acquire(&root, limits).expect("first manager");
        let second = L3CacheManager::acquire(&root, limits).expect("shared manager");

        assert!(first.shares_root_with(&second));
        assert_eq!(first.root(), second.root());
        assert!(
            L3CacheManager::acquire(&root, StoreLimits::new(2_000_000, 0)).is_err(),
            "one root accepted contradictory budgets"
        );
    }

    #[test]
    fn concurrent_stages_cannot_double_reserve_the_budget() {
        let root = temp_root("atomic-reservation");
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(10_000, 0)).unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let mut tasks = Vec::new();
        for _ in 0..2 {
            let manager = manager.clone();
            let barrier = barrier.clone();
            tasks.push(std::thread::spawn(move || {
                let reservation = manager.store().reserve(8_000).unwrap();
                let admitted = reservation.is_ok();
                barrier.wait();
                admitted
            }));
        }
        barrier.wait();
        let admitted = tasks
            .into_iter()
            .map(|task| task.join().unwrap())
            .filter(|admitted| *admitted)
            .count();

        assert_eq!(admitted, 1, "two stages reserved the same bytes");
        assert_eq!(manager.usage().unwrap().reserved_inflight_bytes, 0);
    }

    #[test]
    fn activity_and_store_accounting_are_node_wide() {
        let root = temp_root("node-status");
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
        let stage_a = manager.tier("state-a".to_string(), 4);
        let stage_b = manager.tier("state-b".to_string(), 4);

        stage_a
            .spill(
                "namespace-a",
                &[1, 2, 3],
                &ExactStatePayload::full_state(b"stage-a-state".to_vec()),
                None,
                None,
            )
            .unwrap();

        assert_eq!(stage_b.activity().writes, 1);
        let stage_a_usage = stage_a.status().unwrap().usage;
        let stage_b_usage = stage_b.status().unwrap().usage;
        // Free filesystem capacity is sampled per status call and may change
        // between these reads; compare only the manager-owned accounting.
        assert_eq!(stage_a_usage.budget_bytes, stage_b_usage.budget_bytes);
        assert_eq!(stage_a_usage.used_bytes, stage_b_usage.used_bytes);
        assert_eq!(
            stage_a_usage.reserved_inflight_bytes,
            stage_b_usage.reserved_inflight_bytes
        );
        assert_eq!(
            stage_a_usage.minimum_free_bytes,
            stage_b_usage.minimum_free_bytes
        );
        assert_eq!(stage_a_usage.manifests, stage_b_usage.manifests);
        assert_eq!(stage_a_usage.unique_segments, stage_b_usage.unique_segments);
        assert_eq!(
            stage_a_usage.evicted_manifests,
            stage_b_usage.evicted_manifests
        );
        assert_eq!(
            stage_a_usage.quarantined_objects,
            stage_b_usage.quarantined_objects
        );
        assert_eq!(stage_a.status().unwrap().restorable_manifests, 1);
        assert_eq!(stage_b.status().unwrap().restorable_manifests, 0);
    }

    #[test]
    fn one_stage_cannot_evict_another_stages_pinned_manifest() {
        let root = temp_root("cross-stage-pin");
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
        let stage_a = manager.tier("state-a".to_string(), 4);
        let stage_b = manager.tier("state-b".to_string(), 4);
        let key = stage_a
            .spill(
                "namespace-a",
                &[1, 2, 3],
                &ExactStatePayload::full_state(b"stage-a-state".to_vec()),
                None,
                None,
            )
            .unwrap();

        let pin = stage_a.store().pin(&key);
        assert_eq!(stage_b.manager().prune_to(0).unwrap(), 0);
        assert!(stage_a.store().load_manifest(&key).is_ok());

        drop(pin);
        assert!(stage_b.manager().prune_to(0).unwrap() > 0);
        assert!(stage_a.store().load_manifest(&key).is_err());
    }

    #[test]
    fn root_lock_rejects_an_independent_store_owner() {
        let root = temp_root("root-lock");
        let limits = StoreLimits::new(1_000_000, 0);
        let _manager = L3CacheManager::acquire(&root, limits).unwrap();

        let error = HandoffSegmentStore::open_with_limits(&root, limits)
            .expect_err("a second physical root owner acquired the lock");
        assert!(error.to_string().contains("already owned"));
    }

    #[test]
    fn startup_reconciles_temps_corrupt_manifests_links_and_orphans() {
        let root = temp_root("reconcile");
        let limits = StoreLimits::new(1_000_000, 0);
        let manager = L3CacheManager::acquire(&root, limits).unwrap();
        fs::write(root.join("segments/.tmp-dead"), b"partial").unwrap();
        fs::write(root.join("segments/orphan.seg"), b"orphan").unwrap();
        fs::write(root.join("manifests/corrupt.json"), b"{").unwrap();
        let mut incompatible =
            crate::HandoffManifest::new("state".to_string(), "full-state".to_string());
        incompatible.version = crate::MANIFEST_VERSION + 1;
        incompatible.payload_digest = "incompatible".to_string();
        fs::write(
            root.join("manifests/incompatible.json"),
            serde_json::to_vec(&incompatible).unwrap(),
        )
        .unwrap();
        let mut incomplete =
            crate::HandoffManifest::new("state".to_string(), "full-state".to_string());
        incomplete.payload_digest = "incomplete".to_string();
        incomplete.total_bytes = 4;
        incomplete.segments.push(crate::HandoffSegmentRef {
            index: 0,
            offset: 0,
            bytes: 4,
            digest: "missing-segment".to_string(),
            meta_json: None,
        });
        fs::write(
            root.join("manifests/incomplete.json"),
            serde_json::to_vec(&incomplete).unwrap(),
        )
        .unwrap();
        let namespace = root.join("prefixes/namespace");
        fs::create_dir_all(&namespace).unwrap();
        fs::write(namespace.join("000000000001-prefix.key"), b"missing").unwrap();
        drop(manager);

        let reopened = L3CacheManager::acquire(&root, limits).unwrap();
        let report = reopened.reconciliation();
        assert_eq!(report.removed_temporary_files, 1);
        assert_eq!(report.quarantined_manifests, 3);
        assert_eq!(report.removed_prefix_links, 1);
        assert_eq!(report.removed_orphan_bytes, 6);
        assert!(!root.join("segments/orphan.seg").exists());
        assert!(root.join("quarantine/corrupt.json").exists());
        assert!(root.join("quarantine/incompatible.json").exists());
        assert!(root.join("quarantine/incomplete.json").exists());
    }

    #[test]
    fn low_space_state_transitions_once_and_recovers_after_a_write() {
        let root = temp_root("state-transitions");
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();

        manager.record_write_refusal(WriteRefusal::ReadOnlyLowSpace);
        manager.record_write_refusal(WriteRefusal::ReadOnlyLowSpace);
        assert_eq!(
            manager.effective_status(),
            L3EffectiveStatus {
                state: L3EffectiveState::ReadOnlyLowSpace,
                reason: Some(L3StateReason::ReadOnlyLowSpace),
            }
        );
        assert_eq!(manager.take_state_transitions().len(), 1);

        manager.record_successful_write();
        assert_eq!(manager.effective_status(), L3EffectiveStatus::default());
        let recovery = manager.take_state_transitions();
        assert_eq!(recovery.len(), 1);
        assert_eq!(recovery[0].current, L3EffectiveStatus::default());
    }

    #[test]
    fn request_path_can_skip_cache_while_a_lifecycle_operation_drains() {
        let root = temp_root("lifecycle-fallback");
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();

        let lifecycle = manager.lifecycle_guard();
        assert!(manager.try_operation_guard().is_none());
        drop(lifecycle);
        assert!(manager.try_operation_guard().is_some());
    }

    #[test]
    fn model_clear_matches_internal_identity_exactly() {
        let root = temp_root("model-clear");
        let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
        let model_a = manager.tier_for_model("model-a".to_string(), "state-a".to_string(), 4);
        let model_b = manager.tier_for_model("model-b".to_string(), "state-b".to_string(), 4);
        let key_a = model_a
            .spill(
                "namespace-a",
                &[1, 2, 3],
                &ExactStatePayload::full_state(b"stage-a-state".to_vec()),
                None,
                None,
            )
            .unwrap();
        let key_b = model_b
            .spill(
                "namespace-b",
                &[1, 2, 3],
                &ExactStatePayload::full_state(b"stage-b-state".to_vec()),
                None,
                None,
            )
            .unwrap();

        assert!(manager.clear_model("model-a").unwrap() > 0);
        assert!(manager.store().load_manifest(&key_a).is_err());
        assert!(manager.store().load_manifest(&key_b).is_ok());
        assert_eq!(model_a.status().unwrap().restorable_manifests, 0);
        assert_eq!(model_b.status().unwrap().restorable_manifests, 1);
    }
}
