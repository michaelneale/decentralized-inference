use skippy_ffi::{
    FEATURE_DEVICE_EVENTS, FEATURE_DIAGNOSTIC_EVENTS, FEATURE_KV_EVENTS,
    FEATURE_MODEL_LOAD_EVENTS_V2, FEATURE_RUNTIME_EVENT_REPORTER, FEATURE_RUNTIME_EVENTS,
    FEATURE_UNLOAD_EVENTS,
};

use crate::logging::write_native_log_note;
use crate::runtime_events::abi_features_bitmask;

/// Highest feature bit this build's probe understands. A queried bitmask
/// setting any bit above this is reserved to a future build, not tied to a
/// specific family, and is reported once rather than disabling anything.
const MAX_KNOWN_FEATURE_BIT: u32 = 36;

struct FamilySpec {
    bit: u64,
    name: &'static str,
    required_symbols: &'static [&'static [u8]],
}

const FAMILIES: &[FamilySpec] = &[
    FamilySpec {
        bit: FEATURE_RUNTIME_EVENTS,
        name: "runtime_events",
        required_symbols: &[
            b"skippy_model_open_with_events\0",
            b"skippy_model_open_from_parts_with_events\0",
        ],
    },
    FamilySpec {
        bit: FEATURE_RUNTIME_EVENT_REPORTER,
        name: "runtime_event_reporter",
        required_symbols: &[
            b"skippy_set_runtime_event_reporter\0",
            b"skippy_clear_runtime_event_reporter\0",
        ],
    },
    FamilySpec {
        bit: FEATURE_MODEL_LOAD_EVENTS_V2,
        name: "model_load_events_v2",
        required_symbols: &[b"skippy_emit_model_load_event_v2\0"],
    },
    FamilySpec {
        bit: FEATURE_KV_EVENTS,
        name: "kv_events",
        required_symbols: &[b"skippy_emit_kv_event\0"],
    },
    FamilySpec {
        bit: FEATURE_DEVICE_EVENTS,
        name: "device_events",
        required_symbols: &[b"skippy_emit_device_event\0"],
    },
    FamilySpec {
        bit: FEATURE_DIAGNOSTIC_EVENTS,
        name: "diagnostic_events",
        required_symbols: &[b"skippy_emit_diagnostic_event\0"],
    },
    FamilySpec {
        bit: FEATURE_UNLOAD_EVENTS,
        name: "unload_events",
        required_symbols: &[b"skippy_emit_unload_event\0"],
    },
];

/// Confirmed-family bitmask plus the bounded set of health messages this
/// probe run produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CapabilityReport {
    pub confirmed: u64,
    pub health_messages: Vec<String>,
}

impl CapabilityReport {
    pub fn family_confirmed(&self, bit: u64) -> bool {
        self.confirmed & bit != 0
    }
}

/// Pure decision function. A family whose bit is unset falls back silently
/// (no message). A family whose bit is set but a required symbol is missing
/// is disabled and produces exactly one message naming only that family.
/// Reserved bits beyond `MAX_KNOWN_FEATURE_BIT` produce exactly one
/// additional message and never disable a family.
pub(crate) fn build_report(
    features: u64,
    symbol_exists: impl Fn(&[u8]) -> bool,
) -> CapabilityReport {
    let mut confirmed = 0u64;
    let mut health_messages = Vec::new();

    let reserved_mask = !((1u64 << (MAX_KNOWN_FEATURE_BIT + 1)) - 1);
    let reserved_bits = features & reserved_mask;
    if reserved_bits != 0 {
        health_messages.push(format!(
            "skippy capability probe: reserved/unknown feature bits set: {reserved_bits:#x}"
        ));
    }

    for family in FAMILIES {
        if features & family.bit == 0 {
            continue;
        }
        let all_present = family
            .required_symbols
            .iter()
            .all(|symbol| symbol_exists(symbol));
        if all_present {
            confirmed |= family.bit;
        } else {
            health_messages.push(format!(
                "skippy capability probe: family '{}' advertised feature bit {:#x} but a required symbol is missing; disabling this family only",
                family.name, family.bit
            ));
        }
    }

    CapabilityReport {
        confirmed,
        health_messages,
    }
}

pub(crate) fn symbol_available(name: &[u8]) -> bool {
    #[cfg(feature = "dynamic-native-runtime")]
    {
        skippy_ffi::symbol_present(name)
    }
    #[cfg(not(feature = "dynamic-native-runtime"))]
    {
        #[cfg(unix)]
        {
            let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr().cast()) };
            !symbol.is_null()
        }
        #[cfg(not(unix))]
        {
            let _ = name;
            false
        }
    }
}

/// Probes the loaded native runtime's optional family bit+symbol groups
/// (bits 24 and 31-36), logging one bounded health record per malformed
/// family plus at most one for reserved bits. Callers must have already
/// confirmed exact ABI compatibility; this probe never runs that check
/// itself.
pub fn probe_capabilities() -> CapabilityReport {
    if !skippy_ffi::native_runtime_loaded() {
        return CapabilityReport::default();
    }
    let Some(features) = abi_features_bitmask() else {
        return CapabilityReport::default();
    };
    let report = build_report(features, symbol_available);
    for message in &report.health_messages {
        write_native_log_note(message);
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_family_falls_back_without_a_health_message() {
        let report = build_report(0, |_| false);
        assert_eq!(report.confirmed, 0);
        assert!(report.health_messages.is_empty());
    }

    #[test]
    fn complete_family_is_confirmed_without_a_health_message() {
        let report = build_report(FEATURE_KV_EVENTS, |_| true);
        assert!(report.family_confirmed(FEATURE_KV_EVENTS));
        assert!(report.health_messages.is_empty());
    }

    #[test]
    fn incomplete_symbols_disable_only_that_family() {
        let report = build_report(FEATURE_KV_EVENTS | FEATURE_DEVICE_EVENTS, |symbol| {
            symbol != b"skippy_emit_kv_event\0"
        });
        assert!(!report.family_confirmed(FEATURE_KV_EVENTS));
        assert!(report.family_confirmed(FEATURE_DEVICE_EVENTS));
        assert_eq!(report.health_messages.len(), 1);
        assert!(report.health_messages[0].contains("kv_events"));
    }

    #[test]
    fn independent_families_probe_independently() {
        let report = build_report(
            FEATURE_RUNTIME_EVENTS | FEATURE_RUNTIME_EVENT_REPORTER | FEATURE_UNLOAD_EVENTS,
            |symbol| symbol != b"skippy_set_runtime_event_reporter\0",
        );
        assert!(report.family_confirmed(FEATURE_RUNTIME_EVENTS));
        assert!(!report.family_confirmed(FEATURE_RUNTIME_EVENT_REPORTER));
        assert!(report.family_confirmed(FEATURE_UNLOAD_EVENTS));
        assert_eq!(report.health_messages.len(), 1);
    }

    #[test]
    fn malformed_reserved_bits_emit_one_message_without_disabling_a_family() {
        let reserved_bit = 1u64 << 40;
        let report = build_report(FEATURE_KV_EVENTS | reserved_bit, |_| true);
        assert!(report.family_confirmed(FEATURE_KV_EVENTS));
        assert_eq!(report.health_messages.len(), 1);
        assert!(report.health_messages[0].contains("reserved"));
    }

    #[test]
    fn larger_append_only_features_value_does_not_confuse_known_bits() {
        // A future build reporting extra high bits alongside a fully known
        // family must still confirm that family and flag only the unknown
        // bits, exactly once.
        let future_bits = 0b111u64 << 60;
        let report = build_report(FEATURE_DIAGNOSTIC_EVENTS | future_bits, |_| true);
        assert!(report.family_confirmed(FEATURE_DIAGNOSTIC_EVENTS));
        assert_eq!(report.health_messages.len(), 1);
    }
}
