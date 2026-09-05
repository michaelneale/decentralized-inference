//! Runtime-event producer wiring for the native runtime resolution/load
//! boundary (`.omo/specs/event-system.md` §8.1, plan task 9 line 271's
//! `system/native_runtime.rs`).
//!
//! One reservation per resolution attempt: `RuntimeResolutionStarted` ->
//! (optionally) `NativeLibraryLoaded` -> exactly one terminal
//! (`RuntimeResolutionCompleted` / `RuntimeResolutionFailed`). Best-effort
//! and never blocking, matching `runtime::model_lifecycle::events`'s
//! degrade-on-absent-engine / degrade-on-exhaustion contract.

use mesh_llm_runtime_event_contracts::{
    FactData, NativeRuntimeEventKind, NativeRuntimeFact, OperationId, Outcome, ReasonCode,
    RuntimeEventIngress, RuntimeFact,
};

#[cfg(feature = "dynamic-native-runtime")]
use mesh_llm_runtime_event_contracts::{
    BoundedNumericSummaries, ChildOperationId, DeliveryClass, DiagnosticEventKind, DiagnosticFact,
    FactMetadata, KvRuntimeStateEventKind, KvRuntimeStateFact, LogicalModelId,
    ModelLoadingEventKind, ModelLoadingFact, ModelUnloadingEventKind, ModelUnloadingFact,
    NativeDetail, NativeEmitter, NativeEventCategory, NativeEventKind, NativeFailureCode,
    NativeProgressUnit, NativeSequenceDomain, NativeSequenceEvidence, NativeSequenceObservation,
    NativeSourceEnvelope, NativeStatus, NumericSummary, NumericSummaryKey, NumericValue,
    OperationScope, ResourceHealthEventKind, ResourceHealthFact, ScopeIdentities, SessionId,
    Severity, SubmitOutcome,
};

use crate::runtime_events::engine::OperationReservation;
#[cfg(feature = "dynamic-native-runtime")]
use crate::runtime_events::engine::RuntimeEventEngine;
#[cfg(feature = "dynamic-native-runtime")]
use crate::runtime_events::engine::SyntheticTerminal;
use crate::runtime_events::runtime_event_engine;
#[cfg(feature = "dynamic-native-runtime")]
use skippy_runtime::{RuntimeEvent, RuntimeEventKind};
#[cfg(feature = "dynamic-native-runtime")]
use std::sync::Arc;

fn submit(reservation: &OperationReservation, kind: NativeRuntimeEventKind, data: FactData) {
    let _ =
        reservation
            .ingress()
            .try_submit(RuntimeFact::NativeRuntime(NativeRuntimeFact::with_data(
                kind, data,
            )));
}

fn synthetic_terminal() -> RuntimeFact {
    RuntimeFact::NativeRuntime(NativeRuntimeFact::with_data(
        NativeRuntimeEventKind::RuntimeResolutionFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..FactData::default()
        },
    ))
}

/// One native-runtime resolution attempt. Reserved before any discovery,
/// install, or load work begins.
pub(crate) struct NativeRuntimeResolution {
    root: Option<OperationReservation>,
}

impl NativeRuntimeResolution {
    pub(crate) fn begin() -> Self {
        let Some(engine) = runtime_event_engine() else {
            return Self { root: None };
        };
        let root = engine.reserve_root(OperationId::new(), synthetic_terminal);
        if let Some(root) = &root {
            submit(
                root,
                NativeRuntimeEventKind::RuntimeResolutionStarted,
                FactData::default(),
            );
        }
        Self { root }
    }

    pub(crate) fn library_loaded(&self) {
        if let Some(root) = &self.root {
            submit(
                root,
                NativeRuntimeEventKind::NativeLibraryLoaded,
                FactData::default(),
            );
        }
    }

    /// §8.1 `runtime initialized` -- call once the loaded library is fully
    /// set up and ready to serve (after the runtime-scoped event reporter
    /// install attempt, win or lose). StateTransition class.
    pub(crate) fn initialized(&self) {
        if let Some(root) = &self.root {
            submit(
                root,
                NativeRuntimeEventKind::RuntimeInitialized,
                FactData::default(),
            );
        }
    }

    /// No compatible native library could be found at all (as opposed to
    /// [`Self::not_needed`], where one was already loaded and no
    /// resolution work happened). Resolves with a real
    /// `RuntimeResolutionFailed` terminal, not a no-op release.
    pub(crate) fn unavailable(mut self, reason: ReasonCode) {
        if let Some(root) = &self.root {
            submit(
                root,
                NativeRuntimeEventKind::NativeLibraryUnavailable,
                FactData::default(),
            );
        }
        if let Some(root) = self.root.take() {
            submit(
                &root,
                NativeRuntimeEventKind::RuntimeResolutionFailed,
                FactData {
                    outcome: Some(Outcome::Failure),
                    reason: Some(reason),
                    ..FactData::default()
                },
            );
        }
    }

    /// A compatible runtime is already loaded, so no resolution work
    /// actually happened: release without a terminal rather than reporting
    /// a resolution outcome for work that never ran.
    pub(crate) fn not_needed(mut self) {
        if let Some(root) = self.root.take() {
            root.cancel();
        }
    }

    pub(crate) fn completed(mut self) {
        if let Some(root) = self.root.take() {
            submit(
                &root,
                NativeRuntimeEventKind::RuntimeResolutionCompleted,
                FactData {
                    outcome: Some(Outcome::Success),
                    ..FactData::default()
                },
            );
        }
    }

    pub(crate) fn failed(mut self, reason: ReasonCode) {
        if let Some(root) = self.root.take() {
            submit(
                &root,
                NativeRuntimeEventKind::RuntimeResolutionFailed,
                FactData {
                    outcome: Some(Outcome::Failure),
                    reason: Some(reason),
                    ..FactData::default()
                },
            );
        }
    }
}

#[cfg(feature = "dynamic-native-runtime")]
mod native_family_mapping {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock, PoisonError};
    use uuid::Uuid;

    const MAX_NATIVE_ID_MAPPINGS: usize = 256;

    /// Maps native callback identifiers to opaque, process-scoped contract
    /// identities. Native model and session values are often pointers or
    /// allocator-owned integers; exposing their hex representation in a
    /// typed scope would turn an internal correlation token into a wire
    /// identifier. The map is deliberately bounded and has explicit release
    /// hooks below for model teardown.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum NativeFactFamily {
        ModelLoading,
        Kv,
        Resource,
        Diagnostic,
        Unloading,
        Other,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum NativeScopeKey {
        Model(u64),
        ModelSession { model: u64, session: u64 },
        Emitter(NativeEmitter),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct NativeFamilyKey {
        scope: NativeScopeKey,
        family: NativeFactFamily,
    }

    #[derive(Default)]
    struct NativeIdentityRegistry {
        model_ids: HashMap<u64, LogicalModelId>,
        session_ids: HashMap<(u64, u64), SessionId>,
        model_operations: HashMap<u64, OperationScope>,
        family_operations: HashMap<NativeFamilyKey, OperationScope>,
    }

    static NATIVE_IDENTITIES: OnceLock<Mutex<NativeIdentityRegistry>> = OnceLock::new();

    fn native_identities() -> &'static Mutex<NativeIdentityRegistry> {
        NATIVE_IDENTITIES.get_or_init(|| Mutex::new(NativeIdentityRegistry::default()))
    }

    fn opaque_model_id(registry: &mut NativeIdentityRegistry, raw: u64) -> Option<LogicalModelId> {
        if raw == 0 {
            return None;
        }
        if let Some(id) = registry.model_ids.get(&raw) {
            return Some(id.clone());
        }
        if registry.model_ids.len() >= MAX_NATIVE_ID_MAPPINGS {
            return None;
        }
        let id = LogicalModelId::new(&format!("native-model-{}", Uuid::new_v4())).ok()?;
        registry.model_ids.insert(raw, id.clone());
        Some(id)
    }

    fn opaque_session_id(
        registry: &mut NativeIdentityRegistry,
        model_raw: u64,
        raw: u64,
    ) -> Option<SessionId> {
        if raw == 0 {
            return None;
        }
        let key = (model_raw, raw);
        if let Some(id) = registry.session_ids.get(&key) {
            return Some(id.clone());
        }
        if registry.session_ids.len() >= MAX_NATIVE_ID_MAPPINGS {
            return None;
        }
        let id = SessionId::new(&format!("native-session-{}", Uuid::new_v4())).ok()?;
        registry.session_ids.insert(key, id.clone());
        Some(id)
    }

    /// One numeric correlation field carried from the native envelope. The
    /// key allocation becomes part of the `NumericSummary` stored on the
    /// returned fact -- not a transient/logging-only allocation.
    fn native_numeric_summary(key: &str, value: u64) -> Option<NumericSummary> {
        NumericSummaryKey::new(key)
            .ok()
            .map(|key| NumericSummary::new(key, NumericValue::Unsigned(value)))
    }

    fn native_scope(event: &RuntimeEvent) -> ScopeIdentities {
        let mut registry = native_identities()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let model_id = opaque_model_id(&mut registry, event.model_id);
        let session_id = opaque_session_id(&mut registry, event.model_id, event.session_id);
        ScopeIdentities {
            model_id,
            session_id,
            ..ScopeIdentities::default()
        }
    }

    /// `FactData` keeps the native identifiers in their typed scope slots;
    /// the full source envelope is attached below as private producer
    /// metadata. Numeric summaries remain bounded and are reserved for the
    /// native extension values, rather than duplicating opaque identifiers
    /// into the public projection path.
    fn native_fact_data(event: &RuntimeEvent) -> FactData {
        let summaries = [
            event
                .numeric_summary_0
                .and_then(|value| native_numeric_summary("native_numeric_summary_0", value)),
            event
                .numeric_summary_1
                .and_then(|value| native_numeric_summary("native_numeric_summary_1", value)),
            event
                .numeric_summary_2
                .and_then(|value| native_numeric_summary("native_numeric_summary_2", value)),
            event
                .numeric_summary_3
                .and_then(|value| native_numeric_summary("native_numeric_summary_3", value)),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        FactData {
            scope: native_scope(event),
            numeric_summaries: BoundedNumericSummaries::new(summaries).unwrap_or_default(),
            ..FactData::default()
        }
    }

    fn native_severity(event: &RuntimeEvent) -> Severity {
        match event.kind {
            RuntimeEventKind::DiagnosticFatalFailure
            | RuntimeEventKind::DiagnosticInvariantViolation => Severity::Fatal,
            RuntimeEventKind::DiagnosticRecoverableFailure
            | RuntimeEventKind::DeviceDegraded
            | RuntimeEventKind::DeviceUnavailable
            | RuntimeEventKind::DeviceLost
            | RuntimeEventKind::DeviceOutOfMemory
            | RuntimeEventKind::KvContextCapacityExhausted
            | RuntimeEventKind::ModelOpenFailedHandled
            | RuntimeEventKind::UnloadFailed => Severity::Error,
            RuntimeEventKind::DiagnosticWarningRaised
            | RuntimeEventKind::DiagnosticWarningCleared
            | RuntimeEventKind::DeviceFallbackActivated
            | RuntimeEventKind::UnloadForced => Severity::Warning,
            _ if event.status != skippy_ffi::Status::Ok => Severity::Error,
            _ => Severity::Info,
        }
    }

    fn native_source(event: &RuntimeEvent) -> NativeSourceEnvelope {
        NativeSourceEnvelope {
            abi_version: event.abi_version,
            struct_size: event.struct_size,
            category: NativeEventCategory::new(event.category.raw()),
            kind: NativeEventKind::new(event.kind.raw()),
            emitter: NativeEmitter::new(event.emitter.raw()),
            reserved0: event.reserved0,
            sequence: event.sequence,
            timestamp_mono_ns: event.timestamp_mono_ns,
            model_id: event.model_id,
            stage_id: event.stage_id,
            session_id: event.session_id,
            progress_current: event.progress_current,
            progress_total: event.progress_total,
            progress_unit: NativeProgressUnit::new(event.progress_unit.raw()),
            failure_code: NativeFailureCode::new(event.failure_code.raw()),
            status: NativeStatus::new(event.status as i32),
            reserved1: event.reserved1,
            detail: NativeDetail::bytes(&event.detail_bytes),
            numeric_summary_0: event.numeric_summary_0,
            numeric_summary_1: event.numeric_summary_1,
            numeric_summary_2: event.numeric_summary_2,
            numeric_summary_3: event.numeric_summary_3,
        }
    }

    fn native_metadata_with_domain(
        event: &RuntimeEvent,
        domain: NativeSequenceDomain,
        observation: Option<NativeSequenceObservation>,
    ) -> FactMetadata {
        FactMetadata::native_source(
            native_severity(event),
            native_source(event),
            observation.or_else(|| {
                Some(NativeSequenceObservation::new(
                    domain,
                    NativeEmitter::new(event.emitter.raw()),
                    event.sequence,
                    // `events.cpp` allocates the sequence before invoking
                    // the callback and releases its dispatch mutex before
                    // delivery. A later sequence can therefore arrive
                    // before an earlier one; a jump in this callback cannot
                    // prove a source gap.
                    NativeSequenceEvidence::Unchecked,
                ))
            }),
        )
    }

    pub(crate) fn native_model_open_metadata(event: &RuntimeEvent) -> FactMetadata {
        native_metadata_with_domain(
            event,
            NativeSequenceDomain::Operation,
            Some(NativeSequenceObservation::new(
                NativeSequenceDomain::Operation,
                NativeEmitter::new(event.emitter.raw()),
                event.sequence,
                NativeSequenceEvidence::Unchecked,
            )),
        )
    }

    fn native_metadata(event: &RuntimeEvent) -> FactMetadata {
        native_metadata_with_domain(event, NativeSequenceDomain::Process, None)
    }

    fn native_terminal_failure_data(event: &RuntimeEvent, reason: ReasonCode) -> FactData {
        FactData {
            outcome: Some(Outcome::Failure),
            reason: Some(reason),
            ..native_fact_data(event)
        }
    }

    fn native_success_data(event: &RuntimeEvent) -> FactData {
        FactData {
            outcome: Some(Outcome::Success),
            ..native_fact_data(event)
        }
    }

    /// The `native_family_mappings` table (`inventory/runtime_events.toml`)
    /// realized as code: exactly the mapping the inventory contract test
    /// (`native_family_mappings.rs`) cross-checks against the native patch
    /// queue's kind literals. It performs only bounded in-process identity
    /// lookup; there is no I/O or logging on the native callback thread.
    pub(crate) fn native_family_fact(event: &RuntimeEvent) -> Option<RuntimeFact> {
        let fact = match event.kind {
            RuntimeEventKind::ModelLoadPhaseChanged
            | RuntimeEventKind::ModelLoadTensorsOffloaded
            | RuntimeEventKind::ModelLoadTokenizerReady
            | RuntimeEventKind::ModelLoadAuxComponentReady => {
                RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
                    ModelLoadingEventKind::ModelLoadPhaseChanged,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::ModelLoadMemoryAllocated => {
                RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
                    ModelLoadingEventKind::ModelMemoryAllocationSummary,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::KvInitialized => {
                RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
                    KvRuntimeStateEventKind::KvCacheInitializationCompleted,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::KvPressureCrossed => {
                RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
                    KvRuntimeStateEventKind::CachePressureCrossed,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::KvPressureCleared => {
                RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
                    KvRuntimeStateEventKind::CachePressureCleared,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::KvContextApproachingCapacity => {
                RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
                    KvRuntimeStateEventKind::ContextCapacityApproachingLimit,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::KvContextCapacityExhausted => {
                RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
                    KvRuntimeStateEventKind::ContextExhausted,
                    native_terminal_failure_data(event, ReasonCode::ContextExhausted),
                ))
            }
            RuntimeEventKind::DeviceBackendInitialized => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::BackendInitializationCompleted,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceReady => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::DeviceReady,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceDegraded => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::DeviceDegraded,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceUnavailable => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::DeviceUnavailable,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceRecovered => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::DeviceRecovered,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceLost => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::DeviceLost,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceResourceAllocated => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::ResourceAllocationCompleted,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DeviceOutOfMemory => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::OutOfMemoryCondition,
                    native_terminal_failure_data(event, ReasonCode::OutOfMemory),
                ))
            }
            RuntimeEventKind::DeviceFallbackActivated => {
                RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
                    ResourceHealthEventKind::BackendFallbackActivated,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DiagnosticWarningRaised => {
                RuntimeFact::Diagnostic(DiagnosticFact::with_data(
                    DiagnosticEventKind::WarningRaised,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DiagnosticWarningCleared => {
                RuntimeFact::Diagnostic(DiagnosticFact::with_data(
                    DiagnosticEventKind::WarningCleared,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DiagnosticRecoverableFailure => {
                RuntimeFact::Diagnostic(DiagnosticFact::with_data(
                    DiagnosticEventKind::RecoverableNativeFailure,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::DiagnosticFatalFailure => {
                RuntimeFact::Diagnostic(DiagnosticFact::with_data(
                    DiagnosticEventKind::FatalNativeFailure,
                    native_terminal_failure_data(event, ReasonCode::InternalRuntimeFailure),
                ))
            }
            RuntimeEventKind::DiagnosticInvariantViolation => {
                RuntimeFact::Diagnostic(DiagnosticFact::with_data(
                    DiagnosticEventKind::InvariantProtocolViolation,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::UnloadStarted => {
                RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::UnloadStarted,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::UnloadCompleted => {
                RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::UnloadCompleted,
                    native_success_data(event),
                ))
            }
            RuntimeEventKind::UnloadFailed => {
                RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::UnloadFailed,
                    native_terminal_failure_data(event, ReasonCode::InternalRuntimeFailure),
                ))
            }
            RuntimeEventKind::UnloadForced => {
                RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::ForcedUnload,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::UnloadSessionDraining => {
                RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::SessionDrainingStarted,
                    native_fact_data(event),
                ))
            }
            RuntimeEventKind::ModelOpenStarted
            | RuntimeEventKind::ModelOpenProgress
            | RuntimeEventKind::BackendDeviceSelected
            | RuntimeEventKind::ModelOpenFinished
            | RuntimeEventKind::ModelOpenFailedHandled
            | RuntimeEventKind::Unknown(_) => return None,
        };
        Some(fact.with_metadata(native_metadata(event)))
    }

    fn native_family_terminal_not_delivered() -> FactData {
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..FactData::default()
        }
    }

    fn synthetic_kv_runtime_state_terminal() -> RuntimeFact {
        RuntimeFact::KvRuntimeState(KvRuntimeStateFact::with_data(
            KvRuntimeStateEventKind::ContextExhausted,
            native_family_terminal_not_delivered(),
        ))
    }

    fn synthetic_resource_health_terminal() -> RuntimeFact {
        RuntimeFact::ResourceHealth(ResourceHealthFact::with_data(
            ResourceHealthEventKind::OutOfMemoryCondition,
            native_family_terminal_not_delivered(),
        ))
    }

    fn synthetic_diagnostic_terminal() -> RuntimeFact {
        RuntimeFact::Diagnostic(DiagnosticFact::with_data(
            DiagnosticEventKind::FatalNativeFailure,
            native_family_terminal_not_delivered(),
        ))
    }

    fn synthetic_native_unloading_terminal() -> RuntimeFact {
        RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
            ModelUnloadingEventKind::UnloadFailed,
            native_family_terminal_not_delivered(),
        ))
    }

    fn synthetic_terminal_for(fact: &RuntimeFact) -> SyntheticTerminal {
        match fact {
            RuntimeFact::KvRuntimeState(_) => synthetic_kv_runtime_state_terminal,
            RuntimeFact::ResourceHealth(_) => synthetic_resource_health_terminal,
            RuntimeFact::ModelUnloading(_) => synthetic_native_unloading_terminal,
            _ => synthetic_diagnostic_terminal,
        }
    }

    fn native_fact_family(fact: &RuntimeFact) -> NativeFactFamily {
        match fact {
            RuntimeFact::ModelLoading(_) => NativeFactFamily::ModelLoading,
            RuntimeFact::KvRuntimeState(_) => NativeFactFamily::Kv,
            RuntimeFact::ResourceHealth(_) => NativeFactFamily::Resource,
            RuntimeFact::Diagnostic(_) => NativeFactFamily::Diagnostic,
            RuntimeFact::ModelUnloading(_) => NativeFactFamily::Unloading,
            _ => NativeFactFamily::Other,
        }
    }

    fn family_scope(
        registry: &mut NativeIdentityRegistry,
        key: NativeFamilyKey,
        root_hint: Option<OperationId>,
    ) -> OperationScope {
        if let Some(scope) = registry.family_operations.get(&key).copied() {
            return scope;
        }
        let scope = match root_hint {
            Some(root) => OperationScope::with_child(root, ChildOperationId::new()),
            None => OperationScope::root_only(OperationId::new()),
        };
        if registry.family_operations.len() < MAX_NATIVE_ID_MAPPINGS {
            registry.family_operations.insert(key, scope);
        }
        scope
    }

    /// Register the real Rust lifecycle scope that owns a native model id.
    /// The model-load adapter calls this after it has the `LoadOperation`
    /// root/child identity; subsequent process-global native callbacks reuse
    /// that scope instead of minting one root per fact.
    pub(crate) fn register_native_model_operation(
        native_model_id: u64,
        logical_model_id: Option<LogicalModelId>,
        scope: OperationScope,
    ) {
        if native_model_id == 0 {
            return;
        }
        let mut registry = native_identities()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if registry.model_operations.len() < MAX_NATIVE_ID_MAPPINGS
            || registry.model_operations.contains_key(&native_model_id)
        {
            registry.model_operations.insert(native_model_id, scope);
        }
        if let Some(logical_model_id) = logical_model_id
            && (registry.model_ids.len() < MAX_NATIVE_ID_MAPPINGS
                || registry.model_ids.contains_key(&native_model_id))
        {
            registry.model_ids.insert(native_model_id, logical_model_id);
        }
        // Native resource, diagnostic, KV, and unload facts may be observed
        // while the Rust load reservation is live. Give each family its own
        // child under the established root for correlation, but never let a
        // family reserve or settle the Rust load slot. The entry is retained
        // for the lifetime of this model association and is never overwritten
        // by a second callback for the same native model.
        let root = scope.root();
        for family in [
            NativeFactFamily::Kv,
            NativeFactFamily::Resource,
            NativeFactFamily::Diagnostic,
            NativeFactFamily::Unloading,
        ] {
            let key = NativeFamilyKey {
                scope: NativeScopeKey::Model(native_model_id),
                family,
            };
            if registry.family_operations.len() < MAX_NATIVE_ID_MAPPINGS {
                registry
                    .family_operations
                    .entry(key)
                    .or_insert_with(|| OperationScope::with_child(root, ChildOperationId::new()));
            }
        }
    }

    /// Release the native model association once the Rust unload operation
    /// has settled. This bounds the correlation map and permits allocator
    /// address reuse without accidentally inheriting an older operation.
    pub(crate) fn release_native_model_operation(native_model_id: u64) {
        if native_model_id == 0 {
            return;
        }
        let mut registry = native_identities()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        registry.model_operations.remove(&native_model_id);
        registry.model_ids.remove(&native_model_id);
        registry.family_operations.retain(|key, _| {
            !matches!(
                key.scope,
                NativeScopeKey::Model(model_id)
                    | NativeScopeKey::ModelSession { model: model_id, .. }
                    if model_id == native_model_id
            )
        });
        registry
            .session_ids
            .retain(|(model_id, _), _| *model_id != native_model_id);
    }

    fn native_family_scope_key(fact: &RuntimeFact) -> Option<NativeFamilyKey> {
        let metadata = fact.metadata()?;
        let source = metadata.native_source.as_ref()?;
        let family = native_fact_family(fact);
        let scope = if source.model_id != 0 {
            if family == NativeFactFamily::Kv && source.session_id != 0 {
                NativeScopeKey::ModelSession {
                    model: source.model_id,
                    session: source.session_id,
                }
            } else {
                NativeScopeKey::Model(source.model_id)
            }
        } else {
            NativeScopeKey::Emitter(metadata.native_sequence?.emitter)
        };
        Some(NativeFamilyKey { scope, family })
    }

    fn native_operation_scope(fact: &RuntimeFact, terminal: bool) -> OperationScope {
        let Some(metadata) = fact.metadata() else {
            return OperationScope::root_only(OperationId::new());
        };
        let Some(source) = metadata.native_source.as_ref() else {
            return OperationScope::root_only(OperationId::new());
        };
        let mut registry = native_identities()
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let family = native_fact_family(fact);
        let registered_scope = (source.model_id != 0)
            .then(|| registry.model_operations.get(&source.model_id).copied())
            .flatten();

        // A registered Rust load scope owns only model-loading observations.
        // Every other native family gets a correlated family root and a new
        // child, so repeated diagnostics and unload/KV/resource terminals
        // never reuse a settled reservation. The family child is stable
        // across state and its first terminal observation; terminal claims
        // retire it atomically while this lock is held.
        if !terminal
            && family == NativeFactFamily::ModelLoading
            && let Some(scope) = registered_scope
        {
            return scope;
        }

        let Some(key) = native_family_scope_key(fact) else {
            return OperationScope::root_only(OperationId::new());
        };
        let scope = family_scope(
            &mut registry,
            key,
            registered_scope.map(OperationScope::root),
        );
        if terminal {
            // Claim the family scope while holding the identity lock. A
            // concurrent terminal therefore gets a newly allocated child
            // instead of racing into the same one-shot reservation. If the
            // later reservation is rejected for capacity, the lost mapping
            // is intentional: ingress health records that rejection and the
            // next observation starts a fresh bounded scope.
            registry.family_operations.remove(&key);
        }
        scope
    }

    /// Submits one native-derived fact. State-transition observations use an
    /// unreserved family scope. The first terminal observation reserves that
    /// same family child; an accepted terminal retires the child so a later
    /// terminal gets a new bounded identity while preserving model/family
    /// correlation for the active lifecycle.
    pub(crate) fn submit_native_family_fact(engine: &Arc<RuntimeEventEngine>, fact: RuntimeFact) {
        let terminal = fact.delivery_class() == DeliveryClass::Terminal;
        let scope = native_operation_scope(&fact, terminal);
        let release_model_id = matches!(
            &fact,
            RuntimeFact::ModelUnloading(fact)
                if matches!(
                    fact.kind(),
                    ModelUnloadingEventKind::UnloadCompleted
                        | ModelUnloadingEventKind::UnloadFailed
                )
        )
        .then(|| {
            fact.metadata()
                .and_then(|metadata| metadata.native_source.as_ref())
                .map(|source| source.model_id)
                .filter(|model_id| *model_id != 0)
        })
        .flatten();
        if terminal {
            let reservation = match scope {
                OperationScope::Root(root) => {
                    engine.reserve_root(root, synthetic_terminal_for(&fact))
                }
                OperationScope::Child { root, child } => {
                    engine.reserve_child(root, child, synthetic_terminal_for(&fact))
                }
            };
            if let Some(reservation) = reservation {
                let accepted = reservation.ingress().try_submit(fact) == SubmitOutcome::Accepted;
                if accepted && let Some(native_model_id) = release_model_id {
                    release_native_model_operation(native_model_id);
                }
            }
        } else {
            let ingress = engine.unreserved_ingress(scope);
            let _ = ingress.try_submit(fact);
        }
    }
}

#[cfg(feature = "dynamic-native-runtime")]
pub(crate) use native_family_mapping::{
    native_family_fact, native_model_open_metadata, register_native_model_operation,
    submit_native_family_fact,
};

#[cfg(all(test, feature = "dynamic-native-runtime"))]
mod native_runtime_events_tests {
    use super::*;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};
    use mesh_llm_runtime_event_contracts::ReasonCode;
    use skippy_runtime::{
        RuntimeEventCategory, RuntimeEventEmitterKind, RuntimeEventFailureCode,
        RuntimeEventProgressUnit, Status,
    };
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn install_test_engine() -> Arc<RuntimeEventEngine> {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        engine
    }

    fn rust_model_load_terminal() -> RuntimeFact {
        RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
            ModelLoadingEventKind::NativeModelLoadCompleted,
            FactData {
                outcome: Some(Outcome::Success),
                ..FactData::default()
            },
        ))
    }

    /// A well-formed decoded event for one native kind. `RuntimeEvent`
    /// is the SAFE, already-decoded type this sink actually receives
    /// (its `pub` fields are the crate's own contract; the raw FFI
    /// struct's `from_raw_ptr` decoder is `pub(crate)` to
    /// `skippy-runtime` and this crate never needs to duplicate it).
    fn event(kind: RuntimeEventKind, sequence: u64) -> RuntimeEvent {
        RuntimeEvent {
            abi_version: 1,
            struct_size: std::mem::size_of::<skippy_ffi::SkippyRuntimeEventV1>() as u32,
            category: RuntimeEventCategory::Unknown(0),
            kind,
            emitter: RuntimeEventEmitterKind::WorkerThread,
            reserved0: 0,
            sequence,
            timestamp_mono_ns: sequence,
            model_id: 7,
            stage_id: 0,
            session_id: 3,
            progress_current: 0,
            progress_total: 0,
            progress_unit: RuntimeEventProgressUnit::None,
            failure_code: RuntimeEventFailureCode::None,
            status: Status::Ok,
            reserved1: 0,
            detail_bytes: Vec::new(),
            numeric_summary_0: Some(sequence),
            numeric_summary_1: None,
            numeric_summary_2: None,
            numeric_summary_3: None,
        }
    }

    /// One row per `native_family_mappings` entry, the same 29 pairs
    /// the inventory contract test cross-checks, plus whether the
    /// target inventory id is Terminal-class.
    fn mapping_cases() -> Vec<(RuntimeEventKind, &'static str, bool)> {
        vec![
            (
                RuntimeEventKind::ModelLoadPhaseChanged,
                "model_load_phase_changed",
                false,
            ),
            (
                RuntimeEventKind::ModelLoadMemoryAllocated,
                "model_memory_allocation_summary",
                false,
            ),
            (
                RuntimeEventKind::ModelLoadTensorsOffloaded,
                "model_load_phase_changed",
                false,
            ),
            (
                RuntimeEventKind::ModelLoadTokenizerReady,
                "model_load_phase_changed",
                false,
            ),
            (
                RuntimeEventKind::ModelLoadAuxComponentReady,
                "model_load_phase_changed",
                false,
            ),
            (
                RuntimeEventKind::KvInitialized,
                "kv_cache_initialization_completed",
                false,
            ),
            (
                RuntimeEventKind::KvPressureCrossed,
                "cache_pressure_crossed",
                false,
            ),
            (
                RuntimeEventKind::KvPressureCleared,
                "cache_pressure_cleared",
                false,
            ),
            (
                RuntimeEventKind::KvContextApproachingCapacity,
                "context_capacity_approaching_limit",
                false,
            ),
            (
                RuntimeEventKind::KvContextCapacityExhausted,
                "context_exhausted",
                true,
            ),
            (
                RuntimeEventKind::DeviceBackendInitialized,
                "backend_initialization_completed",
                false,
            ),
            (RuntimeEventKind::DeviceReady, "device_ready", false),
            (RuntimeEventKind::DeviceDegraded, "device_degraded", false),
            (
                RuntimeEventKind::DeviceUnavailable,
                "device_unavailable",
                false,
            ),
            (RuntimeEventKind::DeviceRecovered, "device_recovered", false),
            (RuntimeEventKind::DeviceLost, "device_lost", false),
            (
                RuntimeEventKind::DeviceResourceAllocated,
                "resource_allocation_completed",
                false,
            ),
            (
                RuntimeEventKind::DeviceOutOfMemory,
                "out_of_memory_condition",
                true,
            ),
            (
                RuntimeEventKind::DeviceFallbackActivated,
                "backend_fallback_activated",
                false,
            ),
            (
                RuntimeEventKind::DiagnosticWarningRaised,
                "warning_raised",
                false,
            ),
            (
                RuntimeEventKind::DiagnosticWarningCleared,
                "warning_cleared",
                false,
            ),
            (
                RuntimeEventKind::DiagnosticRecoverableFailure,
                "recoverable_native_failure",
                false,
            ),
            (
                RuntimeEventKind::DiagnosticFatalFailure,
                "fatal_native_failure",
                true,
            ),
            (
                RuntimeEventKind::DiagnosticInvariantViolation,
                "invariant_protocol_violation",
                false,
            ),
            (RuntimeEventKind::UnloadStarted, "unload_started", false),
            (RuntimeEventKind::UnloadCompleted, "unload_completed", true),
            (RuntimeEventKind::UnloadFailed, "unload_failed", true),
            (RuntimeEventKind::UnloadForced, "forced_unload", false),
            (
                RuntimeEventKind::UnloadSessionDraining,
                "session_draining_started",
                false,
            ),
        ]
    }

    #[test]
    fn every_mapped_kind_produces_the_expected_event_id_and_delivery_class() {
        for (kind, expected_id, expected_terminal) in mapping_cases() {
            let fact = native_family_fact(&event(kind, 1))
                .unwrap_or_else(|| panic!("{kind:?} should map to a fact"));
            assert_eq!(fact.kind_id(), expected_id, "kind {kind:?}");
            assert_eq!(
                fact.delivery_class() == DeliveryClass::Terminal,
                expected_terminal,
                "kind {kind:?} delivery class"
            );
        }
    }

    #[test]
    fn model_open_kinds_and_unknown_kinds_are_not_this_sinks_family() {
        for kind in [
            RuntimeEventKind::ModelOpenStarted,
            RuntimeEventKind::ModelOpenProgress,
            RuntimeEventKind::BackendDeviceSelected,
            RuntimeEventKind::ModelOpenFinished,
            RuntimeEventKind::ModelOpenFailedHandled,
            RuntimeEventKind::Unknown(9999),
        ] {
            assert!(
                native_family_fact(&event(kind, 1)).is_none(),
                "kind {kind:?}"
            );
        }
    }

    #[test]
    fn native_correlation_fields_land_on_typed_scope_and_metadata() {
        let fact = native_family_fact(&event(RuntimeEventKind::DeviceReady, 42))
            .expect("device_ready maps");
        let second =
            native_family_fact(&event(RuntimeEventKind::DeviceLost, 43)).expect("device_lost maps");
        let model_id = fact.data().scope.model_id.as_ref().expect("model id");
        let session_id = fact.data().scope.session_id.as_ref().expect("session id");
        assert_eq!(
            Some(model_id.as_str()),
            second.data().scope.model_id.as_ref().map(|id| id.as_str())
        );
        assert_eq!(
            Some(session_id.as_str()),
            second
                .data()
                .scope
                .session_id
                .as_ref()
                .map(|id| id.as_str())
        );
        assert!(!model_id.as_str().contains("0000000000000007"));
        assert!(!session_id.as_str().contains("0000000000000003"));
        assert!(fact.data().scope.stage.is_none());
        let metadata = fact.metadata().expect("native metadata");
        assert_eq!(
            metadata.producer,
            mesh_llm_runtime_event_contracts::ProducerSource::Native
        );
        let source = metadata.native_source.as_ref().expect("native envelope");
        assert_eq!(source.sequence, 42);
        assert_eq!(source.model_id, 7);
        assert_eq!(source.session_id, 3);
        let observation = metadata.native_sequence.expect("sequence observation");
        assert_eq!(observation.domain, NativeSequenceDomain::Process);
        assert_eq!(
            observation.emitter.raw(),
            RuntimeEventEmitterKind::WorkerThread.raw()
        );
    }

    #[test]
    fn native_detail_is_not_copied_into_fact_summary() {
        let mut native_event = event(RuntimeEventKind::DeviceReady, 1);
        native_event.detail_bytes =
            b"/Users/alice/model.gguf https://example.test/token token=secret".to_vec();
        let fact = native_family_fact(&native_event).expect("device_ready maps");
        assert!(fact.data().summary.is_none());
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn non_terminal_facts_submit_unreserved_and_reach_the_state_lane() {
        let engine = install_test_engine();
        let fact = native_family_fact(&event(RuntimeEventKind::DeviceReady, 1))
            .expect("device_ready maps");
        submit_native_family_fact(&engine, fact);
        assert_eq!(
            engine.occupied_count(),
            0,
            "a StateTransition-class native fact must never consume a reservation slot"
        );
        assert!(engine.state_lane_kinds().contains(&"device_ready"));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn terminal_facts_reserve_submit_and_settle_in_one_call() {
        let engine = install_test_engine();
        let fact = native_family_fact(&event(RuntimeEventKind::UnloadCompleted, 1))
            .expect("unload_completed maps");
        submit_native_family_fact(&engine, fact);
        engine.drain();
        assert_eq!(
            engine.occupied_count(),
            0,
            "the one-shot reservation must settle, not linger"
        );
        let delivered = engine.replay().snapshot().into_iter().any(|frame| {
            matches!(
                frame.fact.as_ref(),
                RuntimeFact::ModelUnloading(fact)
                    if *fact.kind() == ModelUnloadingEventKind::UnloadCompleted
                        && fact.data().outcome == Some(Outcome::Success)
            )
        });
        assert!(delivered, "unload_completed must actually reach replay");
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn native_terminal_families_do_not_settle_live_rust_model_load() {
        let engine = install_test_engine();
        let rust_load = engine
            .reserve_root(OperationId::new(), rust_model_load_terminal)
            .expect("Rust model-load reservation");
        register_native_model_operation(7, None, rust_load.scope());

        submit_native_family_fact(
            &engine,
            native_family_fact(&event(RuntimeEventKind::DiagnosticFatalFailure, 1))
                .expect("diagnostic maps"),
        );
        submit_native_family_fact(
            &engine,
            native_family_fact(&event(RuntimeEventKind::DeviceOutOfMemory, 2))
                .expect("device terminal maps"),
        );
        submit_native_family_fact(
            &engine,
            native_family_fact(&event(RuntimeEventKind::UnloadStarted, 3))
                .expect("unload start maps"),
        );
        submit_native_family_fact(
            &engine,
            native_family_fact(&event(RuntimeEventKind::UnloadCompleted, 4))
                .expect("unload terminal maps"),
        );
        engine.drain();

        let first_frames = engine.replay().snapshot();
        let diagnostic_scope = first_frames
            .iter()
            .find_map(|frame| {
                matches!(
                    frame.fact.as_ref(),
                    RuntimeFact::Diagnostic(fact)
                        if *fact.kind() == DiagnosticEventKind::FatalNativeFailure
                )
                .then_some(frame.scope)
            })
            .expect("diagnostic scope");
        let resource_scope = first_frames
            .iter()
            .find_map(|frame| {
                matches!(
                    frame.fact.as_ref(),
                    RuntimeFact::ResourceHealth(fact)
                        if *fact.kind() == ResourceHealthEventKind::OutOfMemoryCondition
                )
                .then_some(frame.scope)
            })
            .expect("resource scope");
        let unload_started_scope = first_frames
            .iter()
            .find_map(|frame| {
                matches!(
                    frame.fact.as_ref(),
                    RuntimeFact::ModelUnloading(fact)
                        if *fact.kind() == ModelUnloadingEventKind::UnloadStarted
                )
                .then_some(frame.scope)
            })
            .expect("unload started scope");
        let unload_completed_scope = first_frames
            .iter()
            .find_map(|frame| {
                matches!(
                    frame.fact.as_ref(),
                    RuntimeFact::ModelUnloading(fact)
                        if *fact.kind() == ModelUnloadingEventKind::UnloadCompleted
                )
                .then_some(frame.scope)
            })
            .expect("unload completed scope");
        assert_eq!(unload_started_scope, unload_completed_scope);
        assert_ne!(diagnostic_scope, resource_scope);
        assert_ne!(diagnostic_scope, rust_load.scope());
        assert_ne!(resource_scope, rust_load.scope());
        assert_ne!(unload_completed_scope, rust_load.scope());

        submit_native_family_fact(
            &engine,
            native_family_fact(&event(RuntimeEventKind::DiagnosticFatalFailure, 5))
                .expect("repeated diagnostic maps"),
        );
        engine.drain();
        let diagnostic_scopes = engine
            .replay()
            .snapshot()
            .into_iter()
            .filter_map(|frame| {
                matches!(
                    frame.fact.as_ref(),
                    RuntimeFact::Diagnostic(fact)
                        if *fact.kind() == DiagnosticEventKind::FatalNativeFailure
                )
                .then_some(frame.scope)
            })
            .collect::<Vec<_>>();
        assert_eq!(diagnostic_scopes.len(), 2);
        assert_ne!(diagnostic_scopes[0], diagnostic_scopes[1]);

        let rust_terminal = rust_model_load_terminal();
        assert_eq!(
            rust_load.ingress().try_submit(rust_terminal),
            SubmitOutcome::Accepted,
            "native diagnostic/resource terminals must use separate child reservations"
        );
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn concurrent_terminal_claims_get_distinct_native_family_scopes() {
        let engine = install_test_engine();
        register_native_model_operation(7, None, OperationScope::root_only(OperationId::new()));
        let barrier = Arc::new(Barrier::new(3));
        let mut threads = Vec::new();
        for sequence in [10, 11] {
            let engine = Arc::clone(&engine);
            let barrier = Arc::clone(&barrier);
            threads.push(thread::spawn(move || {
                barrier.wait();
                submit_native_family_fact(
                    &engine,
                    native_family_fact(&event(RuntimeEventKind::DiagnosticFatalFailure, sequence))
                        .expect("diagnostic maps"),
                );
            }));
        }
        barrier.wait();
        for thread in threads {
            thread.join().expect("terminal callback thread");
        }
        engine.drain();

        let diagnostic_scopes = engine
            .replay()
            .snapshot()
            .into_iter()
            .filter_map(|frame| {
                matches!(
                    frame.fact.as_ref(),
                    RuntimeFact::Diagnostic(fact)
                        if *fact.kind() == DiagnosticEventKind::FatalNativeFailure
                )
                .then_some(frame.scope)
            })
            .collect::<Vec<_>>();
        assert_eq!(diagnostic_scopes.len(), 2);
        assert_ne!(diagnostic_scopes[0], diagnostic_scopes[1]);

        // Release the model-scoped identity map after proving both terminal
        // claims were admitted; this also exercises the unload cleanup path.
        submit_native_family_fact(
            &engine,
            native_family_fact(&event(RuntimeEventKind::UnloadCompleted, 12))
                .expect("unload terminal maps"),
        );
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn a_kv_context_exhausted_terminal_carries_the_context_exhausted_reason() {
        let engine = install_test_engine();
        let fact = native_family_fact(&event(RuntimeEventKind::KvContextCapacityExhausted, 1))
            .expect("context_exhausted maps");
        submit_native_family_fact(&engine, fact);
        engine.drain();
        let reason = engine.replay().snapshot().into_iter().find_map(|frame| {
            let RuntimeFact::KvRuntimeState(fact) = frame.fact.as_ref() else {
                return None;
            };
            (*fact.kind() == KvRuntimeStateEventKind::ContextExhausted)
                .then(|| fact.data().reason.clone())
        });
        assert_eq!(reason, Some(Some(ReasonCode::ContextExhausted)));
        clear_runtime_event_engine();
    }

    /// Acceptance: concurrent callbacks from two threads submit in
    /// order. Each thread submits a run of Terminal-class native facts
    /// with strictly increasing `native_sequence` values in its own
    /// disjoint range; the replay stream (itself in ingress-sequence,
    /// i.e. real submission, order) must show each thread's own
    /// sequence values still strictly increasing -- proof that this
    /// sink never reorders or drops within one thread's callback
    /// stream under concurrent native worker-thread traffic.
    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn concurrent_two_thread_callbacks_submit_every_fact_in_order() {
        let engine = install_test_engine();
        const PER_THREAD: u64 = 200;
        let engine_a = engine.clone();
        let engine_b = engine.clone();
        let thread_a = thread::spawn(move || {
            for sequence in 0..PER_THREAD {
                let fact =
                    native_family_fact(&event(RuntimeEventKind::DiagnosticFatalFailure, sequence))
                        .expect("fatal_native_failure maps");
                submit_native_family_fact(&engine_a, fact);
            }
        });
        let thread_b = thread::spawn(move || {
            for sequence in 0..PER_THREAD {
                let fact = native_family_fact(&event(
                    RuntimeEventKind::DiagnosticFatalFailure,
                    sequence + 1_000_000,
                ))
                .expect("fatal_native_failure maps");
                submit_native_family_fact(&engine_b, fact);
            }
        });
        thread_a.join().expect("thread a");
        thread_b.join().expect("thread b");
        engine.drain();

        let sequences_in_replay_order = engine
            .replay()
            .snapshot()
            .into_iter()
            .filter_map(|frame| {
                let RuntimeFact::Diagnostic(fact) = frame.fact.as_ref() else {
                    return None;
                };
                fact.metadata()
                    .and_then(|metadata| metadata.native_sequence)
                    .map(|observation| observation.sequence)
            })
            .collect::<Vec<_>>();

        let thread_a_sequences = sequences_in_replay_order
            .iter()
            .copied()
            .filter(|&value| value < 1_000_000)
            .collect::<Vec<_>>();
        let thread_b_sequences = sequences_in_replay_order
            .iter()
            .copied()
            .filter(|&value| value >= 1_000_000)
            .collect::<Vec<_>>();
        assert_eq!(thread_a_sequences.len(), PER_THREAD as usize);
        assert_eq!(thread_b_sequences.len(), PER_THREAD as usize);
        assert!(
            thread_a_sequences.windows(2).all(|pair| pair[0] < pair[1]),
            "thread A's own submissions must stay in order: {thread_a_sequences:?}"
        );
        assert!(
            thread_b_sequences.windows(2).all(|pair| pair[0] < pair[1]),
            "thread B's own submissions must stay in order: {thread_b_sequences:?}"
        );
        clear_runtime_event_engine();
    }
}
#[cfg(test)]
mod tests {
    use super::NativeRuntimeResolution;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};
    use mesh_llm_runtime_event_contracts::ReasonCode;

    fn install_test_engine() -> std::sync::Arc<RuntimeEventEngine> {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        engine
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn resolution_reserves_before_completing_with_one_terminal() {
        let engine = install_test_engine();
        let resolution = NativeRuntimeResolution::begin();
        assert_eq!(engine.occupied_count(), 1);
        resolution.library_loaded();
        resolution.initialized();
        resolution.completed();
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn unavailable_reports_a_real_terminal_not_a_no_op() {
        let engine = install_test_engine();
        let resolution = NativeRuntimeResolution::begin();
        assert_eq!(engine.occupied_count(), 1);
        resolution.unavailable(ReasonCode::MissingArtifact);
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        let reported = engine.replay().snapshot().into_iter().any(|frame| {
            matches!(
                frame.fact.as_ref(),
                mesh_llm_runtime_event_contracts::RuntimeFact::NativeRuntime(fact)
                    if *fact.kind()
                        == mesh_llm_runtime_event_contracts::NativeRuntimeEventKind::RuntimeResolutionFailed
            )
        });
        assert!(
            reported,
            "unavailable() must submit a real terminal, not release silently like not_needed()"
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn resolution_failure_resolves_with_one_terminal() {
        let engine = install_test_engine();
        let resolution = NativeRuntimeResolution::begin();
        resolution.failed(ReasonCode::MissingArtifact);
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn not_needed_releases_without_a_terminal() {
        let engine = install_test_engine();
        let resolution = NativeRuntimeResolution::begin();
        assert_eq!(engine.occupied_count(), 1);
        resolution.not_needed();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn dropping_without_a_terminal_synthesizes_one() {
        let engine = install_test_engine();
        {
            let resolution = NativeRuntimeResolution::begin();
            assert_eq!(engine.occupied_count(), 1);
            drop(resolution);
        }
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn absent_engine_never_panics() {
        clear_runtime_event_engine();
        let resolution = NativeRuntimeResolution::begin();
        resolution.library_loaded();
        resolution.completed();
    }
}
