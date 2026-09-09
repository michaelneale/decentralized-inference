use super::{WiringBehavior, WiringEntry, WiringStatus};

pub(super) const KV_CACHE_DISK_MODE: WiringEntry = wired_kv_cache("runtime.kv_cache.disk.mode");
pub(super) const KV_CACHE_DISK_DIRECTORY: WiringEntry =
    wired_kv_cache("runtime.kv_cache.disk.directory");
pub(super) const KV_CACHE_DISK_BUDGET_MIB: WiringEntry =
    wired_kv_cache("runtime.kv_cache.disk.budget_mib");
pub(super) const KV_CACHE_DISK_MINIMUM_FREE_MIB: WiringEntry =
    wired_kv_cache("runtime.kv_cache.disk.minimum_free_mib");

const fn wired_kv_cache(path: &'static str) -> WiringEntry {
    WiringEntry {
        path,
        status: WiringStatus::Wired,
        owner: "#1576",
        reason: "",
        behavior: WiringBehavior::None,
    }
}
