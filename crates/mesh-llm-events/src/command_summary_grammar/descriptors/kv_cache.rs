use super::{Descriptor, JSON, RawKind, YES_JSON, descriptor};

pub(super) const DESCRIPTORS: &[Descriptor] = &[
    descriptor(
        &["mesh-llm", "kv-cache", "status"],
        JSON,
        &["--endpoint"],
        true,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "kv-cache", "prune"],
        YES_JSON,
        &["--target", "--model-identity", "--endpoint"],
        true,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "kv-cache", "clear"],
        YES_JSON,
        &["--model-identity", "--endpoint"],
        true,
        RawKind::None,
    ),
];
