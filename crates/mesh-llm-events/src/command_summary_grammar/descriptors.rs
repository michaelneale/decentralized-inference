#[path = "descriptors/auth.rs"]
mod auth;
#[path = "descriptors/kv_cache.rs"]
mod kv_cache;
#[path = "descriptors/models.rs"]
mod models;
#[path = "descriptors/plugins_benchmark.rs"]
mod plugins_benchmark;
#[path = "descriptors/runtime.rs"]
mod runtime;
#[path = "descriptors/top_level.rs"]
mod top_level;

#[derive(Clone, Copy)]
pub(super) enum RawKind {
    Backend,
    Mode,
    None,
}

#[derive(Clone, Copy)]
pub(super) struct Descriptor {
    pub(super) path: &'static [&'static str],
    pub(super) booleans: &'static [&'static str],
    pub(super) redacted: &'static [&'static str],
    pub(super) conflicts: &'static [&'static [&'static str]],
    pub(super) has_port: bool,
    pub(super) raw: RawKind,
}

const NONE: &[&str] = &[];
const JSON: &[&str] = &["--json"];
const YES_JSON: &[&str] = &["--yes", "--json"];
const REDACTED_NAME: &[&str] = &["name"];
const NO_CONFLICTS: &[&[&str]] = &[];

pub(super) const GLOBAL_REDACTED: &[&str] = &["--join", "--root-relay", "--relay-auth"];

const fn descriptor(
    path: &'static [&'static str],
    booleans: &'static [&'static str],
    redacted: &'static [&'static str],
    has_port: bool,
    raw: RawKind,
) -> Descriptor {
    descriptor_with_conflicts(path, booleans, redacted, NO_CONFLICTS, has_port, raw)
}

const fn descriptor_with_conflicts(
    path: &'static [&'static str],
    booleans: &'static [&'static str],
    redacted: &'static [&'static str],
    conflicts: &'static [&'static [&'static str]],
    has_port: bool,
    raw: RawKind,
) -> Descriptor {
    Descriptor {
        path,
        booleans,
        redacted,
        conflicts,
        has_port,
        raw,
    }
}

pub(super) const DESCRIPTOR_GROUPS: &[&[Descriptor]] = &[
    top_level::DESCRIPTORS,
    plugins_benchmark::PLUGIN_DESCRIPTORS,
    models::DESCRIPTORS,
    plugins_benchmark::BENCHMARK_DESCRIPTORS,
    runtime::DESCRIPTORS,
    kv_cache::DESCRIPTORS,
    auth::DESCRIPTORS,
];
