//! Host-side [`skippy_server::kv_integration::KvLifecycleObserver`]
//! implementation, translating real resident-prefix cache decisions
//! (plan task 12, §8.11) into `RuntimeFact::KvRuntimeState` facts.
//!
//! Every kind here is StateTransition-class (`delivery/execution.rs`), so
//! submission uses `unreserved_ingress` with a fresh root per event -- KV
//! cache decisions have no natural per-request/per-generation correlation
//! at this layer (resident-prefix lookups happen before a generation's own
//! `OperationId` exists), matching `runtime/node_lifecycle_events.rs`'s
//! established pattern for facts with no bounded operation to attach to.
//! `KvLifecycleEvent` itself already carries bounded counts only, so no
//! additional redaction is needed here.

use mesh_llm_runtime_event_contracts::{
    BoundedNumericSummaries, FactData, KvRuntimeStateEventKind, KvRuntimeStateFact, NumericSummary,
    NumericSummaryKey, NumericValue, OperationId, OperationScope, Outcome, RuntimeEventIngress,
    RuntimeFact,
};
use skippy_server::kv_integration::{KvLifecycleEvent, KvLifecycleObserver};

use crate::runtime_events::runtime_event_engine;

fn counts(pairs: &[(&str, u64)]) -> BoundedNumericSummaries {
    let summaries = pairs
        .iter()
        .filter_map(|(key, value)| {
            let key = NumericSummaryKey::new(key).ok()?;
            Some(NumericSummary::new(key, NumericValue::Unsigned(*value)))
        })
        .collect();
    BoundedNumericSummaries::new(summaries).unwrap_or_default()
}

fn kv_fact(
    kind: KvRuntimeStateEventKind,
    outcome: Outcome,
    numeric_summaries: BoundedNumericSummaries,
) -> RuntimeFact {
    RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
        kind,
        FactData {
            outcome: Some(outcome),
            numeric_summaries,
            ..FactData::default()
        },
    ))
}

fn kv_fact_ok(
    kind: KvRuntimeStateEventKind,
    numeric_summaries: BoundedNumericSummaries,
) -> RuntimeFact {
    kv_fact(kind, Outcome::Success, numeric_summaries)
}

/// `None` for a future `#[non_exhaustive]` variant this adapter does not
/// yet model -- it must be silently dropped by the caller, never
/// mislabeled as any existing kind (in particular, never `CacheLookupMiss`).
fn fact_for(event: KvLifecycleEvent) -> Option<RuntimeFact> {
    Some(match event {
        KvLifecycleEvent::CacheLookupHit {
            matched_tokens,
            resident_entries,
        } => kv_fact_ok(
            KvRuntimeStateEventKind::CacheLookupHit,
            counts(&[
                (
                    "matched_tokens",
                    u64::try_from(matched_tokens).unwrap_or(u64::MAX),
                ),
                (
                    "resident_entries",
                    u64::try_from(resident_entries).unwrap_or(u64::MAX),
                ),
            ]),
        ),
        KvLifecycleEvent::CacheLookupMiss => {
            kv_fact_ok(KvRuntimeStateEventKind::CacheLookupMiss, counts(&[]))
        }
        KvLifecycleEvent::PrefixRestored {
            restored_tokens,
            resident_entries,
        } => kv_fact_ok(
            KvRuntimeStateEventKind::PrefixRestored,
            counts(&[
                (
                    "restored_tokens",
                    u64::try_from(restored_tokens).unwrap_or(u64::MAX),
                ),
                (
                    "resident_entries",
                    u64::try_from(resident_entries).unwrap_or(u64::MAX),
                ),
            ]),
        ),
        KvLifecycleEvent::CacheEviction {
            evicted_entries,
            evicted_tokens,
        } => kv_fact_ok(
            KvRuntimeStateEventKind::CacheEviction,
            counts(&[
                (
                    "evicted_entries",
                    u64::try_from(evicted_entries).unwrap_or(u64::MAX),
                ),
                ("evicted_tokens", evicted_tokens),
            ]),
        ),
        KvLifecycleEvent::KvInitStarted => kv_fact_ok(
            KvRuntimeStateEventKind::KvCacheInitializationStarted,
            counts(&[]),
        ),
        KvLifecycleEvent::KvInitCompleted => kv_fact_ok(
            KvRuntimeStateEventKind::KvCacheInitializationCompleted,
            counts(&[]),
        ),
        // Failure kinds MUST carry `Outcome::Failure`, never the
        // `Outcome::Success` every other kind above uses via `kv_fact_ok` --
        // a failed init or a dropped/worker-stopped record is a genuine
        // failure disposition, not a successful lifecycle step.
        KvLifecycleEvent::KvInitFailed => kv_fact(
            KvRuntimeStateEventKind::KvCacheInitializationFailed,
            Outcome::Failure,
            counts(&[]),
        ),
        KvLifecycleEvent::ExactStateRecordFailed => kv_fact(
            KvRuntimeStateEventKind::CacheRecordFailed,
            Outcome::Failure,
            counts(&[]),
        ),
        KvLifecycleEvent::ExactStateRecordCompleted => {
            kv_fact_ok(KvRuntimeStateEventKind::CacheRecordCompleted, counts(&[]))
        }
        KvLifecycleEvent::CapacityApproachingLimit {
            admission_deficit_tokens,
        } => kv_fact_ok(
            KvRuntimeStateEventKind::ContextCapacityApproachingLimit,
            counts(&[("admission_deficit_tokens", admission_deficit_tokens)]),
        ),
        // `#[non_exhaustive]`: a future variant this adapter does not yet
        // model must never be submitted -- returning `None` here (see
        // `observe` below) is the enforcement, not a fallback fact.
        _ => return None,
    })
}

pub(crate) struct SkippyKvRuntimeEventObserver;

impl SkippyKvRuntimeEventObserver {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl KvLifecycleObserver for SkippyKvRuntimeEventObserver {
    fn observe(&self, event: KvLifecycleEvent) {
        let Some(fact) = fact_for(event) else {
            return;
        };
        let Some(engine) = runtime_event_engine() else {
            return;
        };
        let ingress = engine.unreserved_ingress(OperationScope::root_only(OperationId::new()));
        let _ = ingress.try_submit(fact);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};

    fn install_test_engine() -> std::sync::Arc<RuntimeEventEngine> {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        engine
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn cache_hit_reaches_the_state_lane() {
        let engine = install_test_engine();
        let observer = SkippyKvRuntimeEventObserver::new();
        observer.observe(KvLifecycleEvent::CacheLookupHit {
            matched_tokens: 512,
            resident_entries: 3,
        });
        assert!(engine.state_lane_kinds().contains(&"cache_lookup_hit"));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn cache_miss_reaches_the_state_lane() {
        let engine = install_test_engine();
        let observer = SkippyKvRuntimeEventObserver::new();
        observer.observe(KvLifecycleEvent::CacheLookupMiss);
        assert!(engine.state_lane_kinds().contains(&"cache_lookup_miss"));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn prefix_restored_and_eviction_reach_the_state_lane() {
        let engine = install_test_engine();
        let observer = SkippyKvRuntimeEventObserver::new();
        observer.observe(KvLifecycleEvent::PrefixRestored {
            restored_tokens: 256,
            resident_entries: 2,
        });
        observer.observe(KvLifecycleEvent::CacheEviction {
            evicted_entries: 1,
            evicted_tokens: 128,
        });
        let kinds = engine.state_lane_kinds();
        assert!(kinds.contains(&"prefix_restored"));
        assert!(kinds.contains(&"cache_eviction"));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn init_lifecycle_and_new_families_reach_the_state_lane() {
        let engine = install_test_engine();
        let observer = SkippyKvRuntimeEventObserver::new();
        observer.observe(KvLifecycleEvent::KvInitStarted);
        observer.observe(KvLifecycleEvent::KvInitCompleted);
        observer.observe(KvLifecycleEvent::ExactStateRecordFailed);
        observer.observe(KvLifecycleEvent::CapacityApproachingLimit {
            admission_deficit_tokens: 42,
        });
        let kinds = engine.state_lane_kinds();
        assert!(kinds.contains(&"kv_cache_initialization_started"));
        assert!(kinds.contains(&"kv_cache_initialization_completed"));
        assert!(kinds.contains(&"cache_record_failed"));
        assert!(kinds.contains(&"context_capacity_approaching_limit"));
        clear_runtime_event_engine();
    }

    #[test]
    fn failure_kinds_carry_a_failure_outcome_never_success() {
        let init_failed = fact_for(KvLifecycleEvent::KvInitFailed).expect("known variant");
        let RuntimeFact::KvRuntimeState(init_failed) = init_failed else {
            panic!("expected a KvRuntimeState fact");
        };
        assert_eq!(init_failed.data().outcome, Some(Outcome::Failure));

        let record_failed =
            fact_for(KvLifecycleEvent::ExactStateRecordFailed).expect("known variant");
        let RuntimeFact::KvRuntimeState(record_failed) = record_failed else {
            panic!("expected a KvRuntimeState fact");
        };
        assert_eq!(record_failed.data().outcome, Some(Outcome::Failure));

        // Discriminating control: a genuine success kind must NOT also
        // report Failure -- proves the outcome mapping is kind-dependent,
        // not a constant that happens to satisfy the two assertions above.
        let hit = fact_for(KvLifecycleEvent::CacheLookupMiss).expect("known variant");
        let RuntimeFact::KvRuntimeState(hit) = hit else {
            panic!("expected a KvRuntimeState fact");
        };
        assert_eq!(hit.data().outcome, Some(Outcome::Success));
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn absent_engine_degrades_to_no_op() {
        clear_runtime_event_engine();
        let observer = SkippyKvRuntimeEventObserver::new();
        observer.observe(KvLifecycleEvent::CacheLookupMiss);
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn no_prohibited_fields_reach_the_kv_fact_shape() {
        let fact = fact_for(KvLifecycleEvent::PrefixRestored {
            restored_tokens: 10,
            resident_entries: 1,
        })
        .expect("known variant must produce a fact");
        let RuntimeFact::KvRuntimeState(fact) = fact else {
            panic!("expected a KvRuntimeState fact");
        };
        assert!(fact.data().summary.is_none());
        assert_eq!(
            fact.data().scope,
            mesh_llm_runtime_event_contracts::ScopeIdentities::default()
        );
    }
}
