//! Tests for the L3 segment store.
//!
//! Split out of `l3.rs` to keep that file under the 2,000-line limit the
//! coding guidelines set.

use super::*;

fn store(root: &Path, budget: u64) -> HandoffSegmentStore {
    HandoffSegmentStore::open(root, budget).expect("open store")
}

/// Builds a manifest and returns the write-side holds alongside it: a
/// caller that puts segments and commits later must keep them alive, or an
/// eviction in between collects the segments it is about to reference.
/// This is the contract `L3Tier::spill` follows in production.
fn manifest_for<'store>(
    store: &'store HandoffSegmentStore,
    payload: &[u8],
    segment_bytes: usize,
) -> (HandoffManifest, Vec<StoredSegment<'store>>) {
    let mut manifest = HandoffManifest::new("blake3:test".to_string(), "full-state".into());
    let mut held = Vec::new();
    for (index, chunk) in payload.chunks(segment_bytes).enumerate() {
        let stored = store.put_segment(chunk).expect("put segment");
        manifest.segments.push(HandoffSegmentRef {
            index: index as u32,
            offset: (index * segment_bytes) as u64,
            bytes: chunk.len() as u64,
            digest: stored.digest.clone(),
            meta_json: None,
        });
        held.push(stored);
    }
    manifest.total_bytes = payload.len() as u64;
    manifest.payload_digest = segment_digest(payload);
    (manifest, held)
}

/// Put a payload's segments and commit the manifest that binds them,
/// releasing the write-side holds afterwards. The shape production uses:
/// hold across the commit, then let eviction have them.
fn commit_payload(
    store: &HandoffSegmentStore,
    payload: &[u8],
    segment_bytes: usize,
) -> HandoffManifest {
    let (manifest, held) = manifest_for(store, payload, segment_bytes);
    store.commit(&manifest).expect("commit");
    drop(held);
    manifest
}

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir()
        .join("skippy-l3-tests")
        .join(format!("{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    root
}

#[test]
fn roundtrip_assembles_identical_payload() {
    let root = temp_root("roundtrip");
    let store = store(&root, 0);
    let payload: Vec<u8> = (0..100_000u32).map(|value| value as u8).collect();
    let manifest = commit_payload(&store, &payload, 4096);
    let loaded = store
        .load_manifest(&manifest.payload_digest)
        .expect("load manifest");
    assert_eq!(store.assemble(&loaded).expect("assemble"), payload);
}

#[test]
fn cached_usage_never_diverges_from_a_full_scan() {
    // The incremental total exists to keep `reserve` off an O(files) scan
    // per segment put. It is only safe while it agrees with the disk, so
    // check it after every kind of mutation the store performs.
    let root = temp_root("usage-drift");
    let store = store(&root, 0);
    let reserved = || store.reserved_inflight.load(Ordering::Acquire);
    let scanned =
        |store: &HandoffSegmentStore| store.rescan_usage_bytes().expect("rescan") + reserved();

    let assert_agrees = |store: &HandoffSegmentStore, stage: &str| {
        let cached = store.managed_usage_bytes().expect("cached usage");
        let truth = scanned(store);
        assert_eq!(cached, truth, "cached usage diverged after {stage}");
    };

    assert_agrees(&store, "open");

    // Segment puts: the one path that adjusts the total incrementally.
    let payload: Vec<u8> = (0..50_000u32).map(|value| value as u8).collect();
    let manifest = commit_payload(&store, &payload, 4096);
    assert_agrees(&store, "put and commit");

    // A prefix link is a new file under the index tree.
    store
        .link_prefix("namespace", 2, "prefix-key", &manifest.payload_digest)
        .expect("link prefix");
    assert_agrees(&store, "link_prefix");

    // A hit touches metadata only.
    let _ = store.manifest_for_prefix("namespace", 2, "prefix-key");
    assert_agrees(&store, "prefix hit");

    // Bulk removal.
    store.clear().expect("clear");
    assert_agrees(&store, "clear");

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn puts_are_idempotent_and_deduplicated() {
    let root = temp_root("idempotent");
    let store = store(&root, 0);
    let first = store.put_segment(b"same bytes").expect("first put");
    let first_digest = first.digest.clone();
    let second = store.put_segment(b"same bytes").expect("second put");
    let second_digest = second.digest.clone();
    assert_eq!(first_digest, second_digest);
    assert!(first.put.new);
    assert!(!second.put.new);
    assert_eq!(store.segment_footprint_bytes().expect("footprint"), 10);
    drop(first);
    assert_eq!(
        store.collect_unreferenced_segments().unwrap(),
        0,
        "one writer released a segment still held by another writer"
    );
    drop(second);
    assert_eq!(store.collect_unreferenced_segments().unwrap(), 10);
}

#[test]
fn commit_rejects_missing_segments_and_bad_tiling() {
    let root = temp_root("completeness");
    let store = store(&root, 0);
    let payload = vec![7u8; 10_000];
    let (mut manifest, _held) = manifest_for(&store, &payload, 4096);

    let mut missing = manifest.clone();
    missing.segments[1].digest = segment_digest(b"never stored");
    assert!(store.commit(&missing).is_err());

    manifest.segments[2].offset += 1;
    assert!(store.commit(&manifest).is_err());
}

#[test]
fn corrupted_segment_fails_verification_on_read() {
    let root = temp_root("corruption");
    let store = store(&root, 0);
    let payload = vec![42u8; 8192];
    let manifest = commit_payload(&store, &payload, 4096);

    let victim = store.segment_path(&manifest.segments[0].digest);
    let mut bytes = fs::read(&victim).expect("read segment file");
    bytes[0] ^= 0xFF;
    fs::write(&victim, bytes).expect("corrupt segment file");

    assert!(store.assemble(&manifest).is_err());
}

#[test]
fn budget_evicts_the_least_recently_used_manifest() {
    let root = temp_root("budget");
    // Budget fits one payload but not two.
    let store = store(&root, 12_000);
    let old_payload = vec![1u8; 8_000];
    let new_payload = vec![2u8; 8_000];
    let old_manifest = commit_payload(&store, &old_payload, 4096);
    // Ensure a later mtime for the second manifest.
    std::thread::sleep(std::time::Duration::from_millis(20));
    let new_manifest = commit_payload(&store, &new_payload, 4096);

    let manifests = store.list_manifests().expect("list");
    assert!(
        !manifests.contains(&old_manifest.payload_digest),
        "the older entry survived eviction: {manifests:?}"
    );
    assert_eq!(manifests, vec![new_manifest.payload_digest.clone()]);
    assert!(store.assemble(&new_manifest).is_ok());
    assert!(store.segment_footprint_bytes().expect("footprint") <= 12_000);
}

#[test]
fn eviction_follows_last_use_not_last_write() {
    let root = temp_root("lru-by-use");
    // Fits two payloads plus bookkeeping, not three.
    let store = store(&root, 20_000);
    let first = commit_payload(&store, &vec![1u8; 8_000], 4096);
    std::thread::sleep(std::time::Duration::from_millis(20));
    let second = commit_payload(&store, &vec![2u8; 8_000], 4096);

    // The older entry is the one being read, so it is the one that should
    // survive. Under least-recently-written it would be evicted first.
    std::thread::sleep(std::time::Duration::from_millis(20));
    store.touch_manifest(&first.payload_digest);

    std::thread::sleep(std::time::Duration::from_millis(20));
    // A third entry the budget cannot hold: something must go.
    commit_payload(&store, &vec![3u8; 8_000], 4096);

    let manifests = store.list_manifests().expect("list");
    assert!(
        manifests.contains(&first.payload_digest),
        "the recently used entry was evicted: {manifests:?}"
    );
    assert!(
        !manifests.contains(&second.payload_digest),
        "the least recently used entry survived: {manifests:?}"
    );
}

#[test]
fn a_segment_larger_than_the_budget_is_refused() {
    let root = temp_root("oversize-segment");
    let store = store(&root, 4_000);
    let refusal = store
        .try_put_segment(&vec![7u8; 16_000])
        .expect("put")
        .expect_err("a segment larger than the whole budget was stored");
    assert_eq!(refusal, WriteRefusal::SkippedOversize);
    assert_eq!(refusal.reason(), "skipped_oversize");
    assert_eq!(store.segment_footprint_bytes().expect("footprint"), 0);
}

#[test]
fn an_entry_larger_than_a_shrunken_budget_is_refused_at_commit() {
    // A budget shrunk between runs is the realistic way an entry ends up
    // bigger than the cap: it was admissible when its segments were
    // written and is not any more.
    let root = temp_root("oversize-commit");
    let uncapped = store(&root, 0);
    let manifest = {
        let (manifest, held) = manifest_for(&uncapped, &vec![7u8; 16_000], 4_000);
        drop(held);
        manifest
    };
    drop(uncapped);

    let capped = store(&root, 8_000);
    let error = capped
        .commit(&manifest)
        .expect_err("an entry larger than the budget was committed");
    assert!(
        format!("{error:#}").contains("skipped_oversize"),
        "refusal did not carry the reason code: {error:#}"
    );
    assert!(
        capped.list_manifests().expect("list").is_empty(),
        "the refused entry was left loadable"
    );
}

#[test]
fn managed_usage_counts_more_than_segments() {
    let root = temp_root("usage");
    let store = store(&root, 0);
    let manifest = commit_payload(&store, &vec![5u8; 4096], 4096);
    store.commit(&manifest).expect("commit");
    store
        .link_prefix("namespace", 128, "prefix", &manifest.payload_digest)
        .expect("link prefix");

    let segments = store.segment_footprint_bytes().expect("footprint");
    let managed = store.managed_usage_bytes().expect("usage");
    assert!(
        managed > segments,
        "managed usage {managed} ignored manifests and index files (segments {segments})"
    );
}

#[test]
fn prefix_links_obey_the_hard_budget_before_creating_the_index_tree() {
    let root = temp_root("prefix-budget");
    let store = store(&root, 0);
    let manifest = commit_payload(&store, &vec![5u8; 4096], 4096);
    let pin = store.pin(&manifest.payload_digest);
    let used = store.managed_usage_bytes().expect("usage before link");
    store
        .update_limits(StoreLimits::new(used, 0))
        .expect("set exact hard cap");

    let error = store
        .link_prefix("namespace", 128, "prefix", &manifest.payload_digest)
        .expect_err("prefix link exceeded the hard budget");
    assert!(
        format!("{error:#}").contains("insufficient_space"),
        "unexpected refusal: {error:#}"
    );
    assert!(
        !store.namespace_dir("namespace").exists(),
        "a refused prefix link created index directories"
    );
    assert_eq!(store.managed_usage_bytes().expect("usage after link"), used);
    drop(pin);
}

#[test]
fn atomic_publish_removes_temporary_file_after_rename_failure() {
    let root = temp_root("atomic-cleanup");
    fs::create_dir_all(&root).expect("create root");
    let destination = root.join("destination");
    fs::create_dir(&destination).expect("create blocking directory");

    write_atomically(&destination, b"partial bytes")
        .expect_err("publishing a file over a directory succeeded");

    let leftovers: Vec<_> = fs::read_dir(&root)
        .expect("read root")
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(".tmp-"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temporary files survived: {leftovers:?}"
    );
}

#[test]
fn a_pinned_manifest_outranks_a_new_write() {
    // Under pressure the store refuses the incoming write rather than
    // pulling state out from under an operation still using it. The new
    // entry is a miss; the pinned one stays loadable.
    let root = temp_root("pinned");
    let store = store(&root, 12_000);
    let pinned = commit_payload(&store, &vec![1u8; 8_000], 4096);
    store.commit(&pinned).expect("commit pinned");
    let guard = store.pin(&pinned.payload_digest);

    std::thread::sleep(std::time::Duration::from_millis(20));
    let refusal = store
        .try_put_segment(&vec![2u8; 8_000])
        .expect("put")
        .expect_err("the pinned entry was evicted to admit a new write");
    assert_eq!(refusal, WriteRefusal::InsufficientSpace);

    let manifests = store.list_manifests().expect("list");
    assert!(
        manifests.contains(&pinned.payload_digest),
        "a pinned manifest was evicted: {manifests:?}"
    );

    // Once nothing is using it, the same write is admitted.
    drop(guard);
    store
        .try_put_segment(&vec![2u8; 8_000])
        .expect("put")
        .expect("the write stayed refused after the pin was released");
}

#[test]
fn the_free_space_reserve_refuses_writes() {
    let root = temp_root("reserve");
    // A reserve no filesystem can satisfy, rather than one derived from
    // live free space: another test freeing a few MiB mid-run must not
    // decide whether this one passes.
    let store = HandoffSegmentStore::open_with_limits(&root, StoreLimits::new(0, u64::MAX))
        .expect("open store");
    let refusal = store
        .try_put_segment(b"bytes that do not fit the reserve")
        .expect("put")
        .expect_err("write was admitted below the reserve");
    assert_eq!(refusal, WriteRefusal::ReadOnlyLowSpace);
    assert_eq!(refusal.reason(), "read_only_low_space");
}

#[test]
fn reservations_are_released_after_the_write() {
    let root = temp_root("reservation");
    let store = store(&root, 1_000_000);
    store.put_segment(b"some bytes").expect("put");
    assert_eq!(
        store.usage().expect("usage").reserved_inflight_bytes,
        0,
        "a completed write left capacity reserved"
    );
}

#[test]
fn clear_removes_every_unpinned_entry() {
    let root = temp_root("clear");
    let store = store(&root, 0);
    let linked = commit_payload(&store, &vec![1u8; 4096], 4096);
    store
        .link_prefix("namespace", 128, "prefix", &linked.payload_digest)
        .unwrap();
    commit_payload(&store, &vec![2u8; 4096], 4096);

    let freed = store.clear().expect("clear");
    assert!(freed > 0, "clear freed nothing");
    assert!(store.list_manifests().expect("list").is_empty());
    assert_eq!(store.segment_footprint_bytes().expect("footprint"), 0);
    assert!(
        store
            .recorded_prefix_lengths("namespace")
            .unwrap()
            .is_empty(),
        "clear left a dangling prefix link"
    );
}

#[test]
fn prune_frees_down_to_the_target() {
    let root = temp_root("prune");
    let store = store(&root, 0);
    for fill in 1u8..=3 {
        let manifest = commit_payload(&store, &vec![fill; 8_000], 4096);
        store.commit(&manifest).expect("commit");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let before = store.managed_usage_bytes().expect("usage");
    store.prune_to(before / 2).expect("prune");
    let after = store.managed_usage_bytes().expect("usage");
    assert!(after < before, "prune freed nothing ({before} -> {after})");
}

#[test]
fn live_limit_update_changes_the_pair_and_prunes_inactive_entries() {
    let root = temp_root("live-limits");
    let store = store(&root, 1_000_000);
    for fill in 1u8..=3 {
        let manifest = commit_payload(&store, &vec![fill; 8_000], 4096);
        store.commit(&manifest).expect("commit");
    }
    let before = store.managed_usage_bytes().expect("usage");
    let next = StoreLimits::new(before / 2, 4096);
    let previous = store.update_limits(next).expect("update limits");
    assert_eq!(previous, StoreLimits::new(1_000_000, 0));
    assert_eq!(store.limits(), next);
    assert!(
        store.managed_usage_bytes().expect("usage after") <= next.budget_bytes,
        "live shrink did not prune to the new cap"
    );
}

#[test]
fn a_corrupt_segment_is_quarantined_not_left_in_place() {
    let root = temp_root("quarantine");
    let store = store(&root, 0);
    let digest = store.put_segment(b"segment bytes").expect("put").digest;
    fs::write(
        root.join("segments").join(format!("{digest}.seg")),
        b"tampered",
    )
    .expect("tamper with the segment");

    let error = store
        .read_segment(&digest)
        .expect_err("a tampered segment was served");
    assert!(
        format!("{error:#}").contains("quarantined"),
        "corrupt segment was not quarantined: {error:#}"
    );
    assert!(
        !store.has_segment(&digest),
        "the corrupt segment is still in the managed tree"
    );
    assert!(
        root.join("quarantine").exists(),
        "nothing was moved to quarantine"
    );
}

/// Not a pass/fail assertion: a stopwatch on the cost that smaller windows
/// buy. Eviction parses every manifest to build its reference map, and a
/// 64-row window turns a 19K-token entry into ~9.5k segment refs. Run with
/// `cargo test -p skippy-cache --lib eviction_cost -- --ignored --nocapture`.
#[test]
#[ignore = "measurement, not a check; takes tens of seconds"]
fn eviction_cost_at_realistic_segment_counts() {
    const SEGMENTS_PER_MANIFEST: usize = 9_504; // 16 layers x 2 x ceil(19000/64)
    const MANIFESTS: usize = 20;
    let root = temp_root("eviction-cost");
    let store = store(&root, 0);

    // One physical segment shared by every ref: this measures manifest
    // parsing and reference mapping, not filesystem write throughput.
    let bytes = vec![7u8; 65_536];
    let digest = store.put_segment(&bytes).expect("put").digest;
    let build = std::time::Instant::now();
    for manifest_index in 0..MANIFESTS {
        let mut manifest = HandoffManifest::new("blake3:cost".to_string(), "full-state".into());
        for index in 0..SEGMENTS_PER_MANIFEST {
            manifest.segments.push(HandoffSegmentRef {
                index: index as u32,
                offset: (index * bytes.len()) as u64,
                bytes: bytes.len() as u64,
                digest: digest.clone(),
                meta_json: Some(format!("k:{}:0:{}", index % 32, index / 32)),
            });
        }
        manifest.total_bytes = (SEGMENTS_PER_MANIFEST * bytes.len()) as u64;
        manifest.payload_digest = format!("blake3:manifest-{manifest_index}");
        store.commit(&manifest).expect("commit");
    }
    let build_ms = build.elapsed().as_millis();

    let manifest_bytes = directory_bytes(&root.join(MANIFEST_DIR)).expect("manifest bytes");
    let usage = store.managed_usage_bytes().expect("usage");
    let evict = std::time::Instant::now();
    store.enforce_budget_to(usage / 2).expect("enforce");
    let evict_ms = evict.elapsed().as_millis();

    println!(
        "eviction cost: {MANIFESTS} manifests x {SEGMENTS_PER_MANIFEST} refs, \
         manifest bytes {manifest_bytes} ({} KiB each), build {build_ms} ms, \
         enforce_budget {evict_ms} ms",
        manifest_bytes / MANIFESTS as u64 / 1024
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn eviction_leaves_headroom_so_a_full_cache_is_not_repriced_per_commit() {
    let root = temp_root("low-water");
    let store = store(&root, 40_000);
    let mut eviction_triggered = false;
    for fill in 1u8..=16 {
        let manifest = commit_payload(&store, &vec![fill; 8_000], 4096);
        store.commit(&manifest).expect("commit");
        if store.usage().unwrap().evicted_manifests > 0 {
            eviction_triggered = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(eviction_triggered, "fixture never crossed the budget");
    let usage = store.managed_usage_bytes().expect("usage");
    assert!(
        usage < 40_000,
        "eviction left no headroom below the cap: {usage}"
    );
    assert_eq!(
        store.enforce_budget().expect("second pass"),
        0,
        "a store already under the cap must not pay for another pass"
    );
}

#[test]
fn unreferenced_segments_are_collected() {
    let root = temp_root("gc");
    let store = store(&root, 0);
    store.put_segment(b"orphan bytes").expect("orphan put");
    let payload = vec![9u8; 4096];
    let manifest = commit_payload(&store, &payload, 4096);

    let freed = store.collect_unreferenced_segments().expect("collect");
    assert_eq!(freed, 12);
    assert!(store.assemble(&manifest).is_ok());
}
