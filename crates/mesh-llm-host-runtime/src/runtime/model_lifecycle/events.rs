//! Runtime-event producer wiring for the model load/unload lifecycle
//! (plan task 9, `.omo/plans/event-system.md` line 270).
//!
//! Bridges the existing load/unload call sites in `load.rs`/`unload.rs` to
//! the host runtime-event engine (`crate::runtime_events`) via reservations
//! acquired BEFORE any load/unload work begins. Every emission here is
//! best-effort: an absent engine, an exhausted reservation table, or a
//! rejected submit never blocks or fails primary model serving work -- the
//! `let _ =` submits and the `Option<OperationReservation>` degrade path are
//! the enforcement mechanism, not a convention.
//!
//! One load operation uses a root reservation (owns the `ModelAvailability`
//! terminal) plus a child reservation (owns the `ModelLoading` terminal).
//! Native model-load completion resolves only the child; `model_available`
//! is a separate, later call from Rust code after serving surfaces are
//! usable, per the plan's "native completion MUST NOT imply availability"
//! invariant.

use mesh_llm_runtime_event_contracts::{
    ChildOperationId, FactData, HumanSummary, LogicalModelId, ModelAvailabilityEventKind,
    ModelAvailabilityFact, ModelLoadingEventKind, ModelLoadingFact, ModelPreparationEventKind,
    ModelPreparationFact, ModelUnloadingEventKind, ModelUnloadingFact, NodeAvailabilityEventKind,
    NodeAvailabilityFact, OperationId, OperationScope, Outcome, Progress, ReasonCode,
    RuntimeEventIngress, RuntimeFact, ScopeIdentities,
};

use crate::runtime_events::engine::{OperationReservation, ScopedIngress};
use crate::runtime_events::runtime_event_engine;
#[cfg(any(test, not(feature = "dynamic-native-runtime")))]
use mesh_llm_runtime_event_contracts::ProgressUnit;

/// Task 10 addition (plan task 10, line 279's §8.14 `available_model_set_changed`
/// row): a purely additive, StateTransition-class, unreserved co-emission
/// alongside Task 9's own model-availability/unload terminals. Never touches
/// a reservation slot, so it is structurally invisible to every existing
/// Task 9 test in this file (StateTransition-class facts never reach
/// `replay()`/`occupied_count()` -- see runtime_events/engine/mod.rs's
/// `state_lane_kinds()` doc comment). Local-only: never gossiped.
fn emit_available_model_set_changed(model: &str) {
    let Some(engine) = runtime_event_engine() else {
        return;
    };
    let ingress = engine.unreserved_ingress(OperationScope::root_only(OperationId::new()));
    let _ = ingress.try_submit(RuntimeFact::NodeAvailability(
        NodeAvailabilityFact::with_data(
            NodeAvailabilityEventKind::AvailableModelSetChanged,
            model_scope(model),
        ),
    ));
}

/// A `FactData` carrying only the model's logical id. `model` is
/// defensively redacted (`redact_local_path`) before it ever reaches
/// `LogicalModelId` -- event-system-fixes F2 live-sampling finding: a raw
/// local filesystem path (a `mesh-llm load`/`--model`/`--gguf` argument,
/// including one re-used verbatim on a startup load's failure path that
/// never got the chance to resolve) must never reach `scope.model_id` on
/// the wire. A non-path model reference (an HF `org/repo[:sel]` ref, a
/// resolved `local-gguf/sha256-...` id, a plain catalog name) is not
/// path-shaped and passes through unchanged. An over-length model name
/// (bounded by `LogicalModelId`) still degrades to no scope rather than
/// failing the emission.
fn model_scope(model: &str) -> FactData {
    let mut data = FactData::default();
    if let Ok(id) = LogicalModelId::new(redact_local_path(model).as_ref()) {
        data.scope = ScopeIdentities {
            model_id: Some(id),
            ..ScopeIdentities::default()
        };
    }
    data
}

/// Build a native observation with the same privacy-safe model identity as
/// the Rust lifecycle facts. An observation never supplies the authoritative
/// load outcome; the ABI result settles that reservation separately.
#[cfg(feature = "dynamic-native-runtime")]
pub(crate) fn native_load_observation(
    model: &str,
    kind: ModelLoadingEventKind,
    metadata: mesh_llm_runtime_event_contracts::FactMetadata,
    progress: Option<Progress>,
) -> RuntimeFact {
    use mesh_llm_runtime_event_contracts::{
        BoundedNumericSummaries, NumericSummary, NumericSummaryKey, NumericValue,
    };
    let summaries = metadata
        .native_source
        .as_ref()
        .map_or_else(Vec::new, |source| {
            [
                ("native_numeric_summary_0", source.numeric_summary_0),
                ("native_numeric_summary_1", source.numeric_summary_1),
                ("native_numeric_summary_2", source.numeric_summary_2),
                ("native_numeric_summary_3", source.numeric_summary_3),
            ]
            .into_iter()
            .filter_map(|(key, value)| {
                Some(NumericSummary::new(
                    NumericSummaryKey::new(key).ok()?,
                    NumericValue::Unsigned(value?),
                ))
            })
            .collect()
        });
    RuntimeFact::ModelLoading(ModelLoadingFact::with_metadata(
        kind,
        FactData {
            progress,
            numeric_summaries: BoundedNumericSummaries::new(summaries).unwrap_or_default(),
            ..model_scope(model)
        },
        metadata,
    ))
}

/// Collapse `model` to its bare filename when it looks like a local
/// filesystem path (absolute Unix, `~`/`./`/`../`-relative, or a Windows
/// drive-letter path); returns `model` unchanged otherwise. Manual
/// separator handling -- not `std::path::Path` -- so the check behaves
/// identically regardless of which OS this binary happens to be built
/// for: a `C:\...` path is redacted the same way whether observed on a
/// Windows host or exercised in a portable unit test.
fn redact_local_path(model: &str) -> std::borrow::Cow<'_, str> {
    let bytes = model.as_bytes();
    let is_windows_drive_path = bytes.len() > 2
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'/' | b'\\');
    let looks_like_path = model.starts_with('/')
        || model.starts_with('\\')
        || model.starts_with("./")
        || model.starts_with("../")
        || model.starts_with('~')
        || is_windows_drive_path;
    if !looks_like_path {
        return std::borrow::Cow::Borrowed(model);
    }
    let trimmed = model.trim_end_matches(['/', '\\']);
    match trimmed
        .rsplit(['/', '\\'])
        .next()
        .filter(|segment| !segment.is_empty())
    {
        Some(basename) => std::borrow::Cow::Owned(basename.to_string()),
        None => std::borrow::Cow::Borrowed("local-model"),
    }
}

/// A stable, non-path identity for the four facts a load operation emits
/// BEFORE its model's source resolves (`ModelQueued`,
/// `ModelResolutionStarted`, `ModelLoadRequested`, `ModelLoadStarted`) --
/// the event-system-fixes F2 finding this whole file's redaction exists
/// for. `model` (the raw `mesh-llm load`/`--model`/`--gguf` argument) may
/// itself be a raw local filesystem path; `redact_local_path` strips any
/// directory component before it is folded in here, purely for
/// observability (so a human watching `runtime_state.models[]` mid-load
/// still sees a recognizable filename). The `root` component is what
/// actually matters for correctness: it makes every load operation's
/// provisional row globally unique, so two concurrent loads can never
/// collide on the SAME provisional `models[]` row even when their
/// (redacted) basenames coincide. The reducer's
/// `reconcile_model_root_identity` (`runtime_events::reducer::domain`)
/// supersedes this row the instant the SAME root reports its resolved
/// (post-resolution) identity, instead of leaving it an orphaned phantom.
fn provisional_model_identity(root: OperationId, model: &str) -> String {
    format!("pending-resolution/{root}/{}", redact_local_path(model))
}

fn terminal_not_delivered() -> FactData {
    FactData {
        outcome: Some(Outcome::Unknown),
        reason: Some(ReasonCode::TerminalNotDelivered),
        ..FactData::default()
    }
}

fn synthetic_load_terminal() -> RuntimeFact {
    RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
        ModelLoadingEventKind::ModelLoadFailed,
        terminal_not_delivered(),
    ))
}

fn synthetic_prep_terminal() -> RuntimeFact {
    RuntimeFact::ModelPreparation(ModelPreparationFact::with_data(
        ModelPreparationEventKind::ModelResolutionFailed,
        terminal_not_delivered(),
    ))
}

fn synthetic_ready_terminal() -> RuntimeFact {
    RuntimeFact::ModelPreparation(ModelPreparationFact::with_data(
        ModelPreparationEventKind::ModelPreparationFailed,
        terminal_not_delivered(),
    ))
}

fn synthetic_availability_terminal() -> RuntimeFact {
    RuntimeFact::ModelAvailability(ModelAvailabilityFact::with_data(
        ModelAvailabilityEventKind::ModelUnavailable,
        terminal_not_delivered(),
    ))
}

fn synthetic_unload_terminal() -> RuntimeFact {
    RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
        ModelUnloadingEventKind::UnloadFailed,
        terminal_not_delivered(),
    ))
}

fn submit_loading(reservation: &OperationReservation, kind: ModelLoadingEventKind, model: &str) {
    let _ =
        reservation
            .ingress()
            .try_submit(RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
                kind,
                model_scope(model),
            )));
}

fn submit_unloading(
    reservation: &OperationReservation,
    kind: ModelUnloadingEventKind,
    model: &str,
) {
    let _ = reservation
        .ingress()
        .try_submit(RuntimeFact::ModelUnloading(ModelUnloadingFact::with_data(
            kind,
            model_scope(model),
        )));
}

/// The producer for `ModelLoadingEventKind::ModelLoadProgress` (event-
/// system-fixes deferral D2: previously unproduced anywhere in the
/// workspace). `ingress` must come from [`LoadOperation::progress_ingress`]
/// so this correlates under the same operation root as the load's other
/// facts. Progress-class: coalesced per-slot and flushed on the engine's
/// frozen 100ms `PROGRESS_EXPORT_INTERVAL`
/// (`runtime_events::engine::lanes::submit_progress`) -- callers submit
/// freely and the engine throttles; this function must never add a second,
/// producer-side throttle.
#[cfg(any(test, not(feature = "dynamic-native-runtime")))]
pub(crate) fn submit_load_progress(
    ingress: &ScopedIngress,
    model: &str,
    current: u64,
    total: Option<u64>,
    unit: ProgressUnit,
) {
    let data = FactData {
        progress: Some(Progress::new(current, total, unit)),
        ..model_scope(model)
    };
    let _ = ingress.try_submit(RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
        ModelLoadingEventKind::ModelLoadProgress,
        data,
    )));
}

/// One model-load operation. Acquired before any load work begins; the root
/// reservation always outlives its three children (`prep` for §8.2 model
/// resolution, `ready` for §8.2 model preparation completion, `load` for
/// §8.3 model loading) so an availability decision can be made after
/// native loading resolves.
pub(crate) struct LoadOperation {
    root: Option<OperationReservation>,
    prep: Option<OperationReservation>,
    ready: Option<OperationReservation>,
    load: Option<OperationReservation>,
}

impl LoadOperation {
    /// Reserve root + three children before doing any load work. On
    /// reservation exhaustion this degrades to all-`None`: every subsequent
    /// call is a silent no-op and load work proceeds unaffected (engine
    /// health already counts the exhaustion internally).
    pub(crate) fn begin(model: &str) -> Self {
        let Some(engine) = runtime_event_engine() else {
            return Self {
                root: None,
                prep: None,
                ready: None,
                load: None,
            };
        };
        let root_id = OperationId::new();
        let root = engine.reserve_root(root_id, synthetic_availability_terminal);
        let prep = if root.is_some() {
            engine.reserve_child(root_id, ChildOperationId::new(), synthetic_prep_terminal)
        } else {
            None
        };
        let ready = if root.is_some() {
            engine.reserve_child(root_id, ChildOperationId::new(), synthetic_ready_terminal)
        } else {
            None
        };
        let load = if root.is_some() {
            engine.reserve_child(root_id, ChildOperationId::new(), synthetic_load_terminal)
        } else {
            None
        };
        // F2 fix: these four facts fire before the model's source is known,
        // so `model` (the raw, possibly-a-local-path argument) is never
        // used directly here -- see `provisional_model_identity`.
        let provisional = provisional_model_identity(root_id, model);
        if let Some(prep) = &prep {
            let _ = prep.ingress().try_submit(RuntimeFact::ModelPreparation(
                ModelPreparationFact::with_data(
                    ModelPreparationEventKind::ModelQueued,
                    model_scope(&provisional),
                ),
            ));
            let _ = prep.ingress().try_submit(RuntimeFact::ModelPreparation(
                ModelPreparationFact::with_data(
                    ModelPreparationEventKind::ModelResolutionStarted,
                    model_scope(&provisional),
                ),
            ));
        }
        if let Some(load) = &load {
            submit_loading(
                load,
                ModelLoadingEventKind::ModelLoadRequested,
                &provisional,
            );
            submit_loading(load, ModelLoadingEventKind::ModelLoadStarted, &provisional);
        }
        Self {
            root,
            prep,
            ready,
            load,
        }
    }

    /// §8.2 `model resolution completed` -- called once the model's source
    /// (local path / remote catalog ref) is resolved, before any native
    /// load work begins. Resolves only the `prep` child's terminal.
    ///
    /// Also resolves §8.2's DISTINCT `model preparation completed` bullet
    /// on the separate `ready` child, right here: in every call site this
    /// task wires (local-required / remote-catalog resolution with no
    /// separate download-or-materialize step owned by this task), the
    /// model is FULLY prepared the instant its source resolves -- there is
    /// no live intermediate download/materialize phase in this codepath to
    /// report progress for (that lives in `models/catalog.rs`, outside
    /// this task's file list). Folding "prepared" into this same call,
    /// on its OWN reservation/terminal rather than reusing `prep`'s slot,
    /// keeps both required events real and independently provable without
    /// inventing a fake intermediate phase.
    pub(crate) fn resolution_completed(&mut self, model: &str) {
        if let Some(prep) = self.prep.take() {
            let _ = prep.ingress().try_submit(RuntimeFact::ModelPreparation(
                ModelPreparationFact::with_data(
                    ModelPreparationEventKind::ModelResolutionCompleted,
                    FactData {
                        outcome: Some(Outcome::Success),
                        ..model_scope(model)
                    },
                ),
            ));
        }
        if let Some(ready) = self.ready.take() {
            let _ = ready.ingress().try_submit(RuntimeFact::ModelPreparation(
                ModelPreparationFact::with_data(
                    ModelPreparationEventKind::ModelPreparationCompleted,
                    FactData {
                        outcome: Some(Outcome::Success),
                        ..model_scope(model)
                    },
                ),
            ));
        }
    }

    /// §8.3 `model load progress` (event-system-fixes deferral D2):
    /// exposes a narrow, Progress-class-only capability bound to this
    /// operation's `load` child, so a caller can report native model-open
    /// progress ticks (e.g. `ModelOpenProgress`'s tensor-loading percentage)
    /// through [`submit_load_progress`] while they correlate under the SAME
    /// operation root as `model_load_requested`/`model_load_started`/
    /// `native_model_load_completed` -- never a fresh, uncorrelated root.
    /// `None` once reservation exhaustion degraded `begin()` to a no-op, or
    /// once the `load` child was already consumed by `native_load_completed`
    /// / `load_failed`.
    pub(crate) fn progress_ingress(&self) -> Option<ScopedIngress> {
        self.load.as_ref().map(OperationReservation::ingress)
    }

    /// §8.4 `model capacity changed` -- call once local capacity has been
    /// reserved for this load. StateTransition class on the root, so this
    /// never competes with the root's own `ModelAvailability` terminal.
    pub(crate) fn capacity_changed(&self, model: &str) {
        if let Some(root) = &self.root {
            let _ = root.ingress().try_submit(RuntimeFact::ModelAvailability(
                ModelAvailabilityFact::with_data(
                    ModelAvailabilityEventKind::ModelCapacityChanged,
                    model_scope(model),
                ),
            ));
        }
    }

    /// §8.3 `backend or device selected` -- called once the loaded model's
    /// backend/device is known (after native load succeeds). Uses the
    /// bounded `summary` field to carry the backend name; `LoadingFactData`
    /// has no dedicated backend field. StateTransition class, so this never
    /// competes with the child's single terminal write.
    pub(crate) fn backend_device_selected(&self, model: &str, backend: &str) {
        if let Some(load) = &self.load {
            let mut data = model_scope(model);
            data.summary = HumanSummary::new(&format!("backend={backend}")).ok();
            let _ =
                load.ingress()
                    .try_submit(RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
                        ModelLoadingEventKind::BackendDeviceSelected,
                        data,
                    )));
        }
    }

    /// Native model load completed. Resolves only the `load` child's
    /// terminal; the root (public availability) is intentionally untouched.
    /// Also emits `ModelAvailability::NativeModelLoaded` (§8.4, a distinct
    /// required event from this family, StateTransition class) on the root
    /// -- native completion informs, but never decides, availability.
    pub(crate) fn native_load_completed(mut self, model: &str) -> AvailabilityOperation {
        if let Some(load) = self.load.take() {
            let _ =
                load.ingress()
                    .try_submit(RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
                        ModelLoadingEventKind::NativeModelLoadCompleted,
                        FactData {
                            outcome: Some(Outcome::Success),
                            ..model_scope(model)
                        },
                    )));
        }
        if let Some(root) = &self.root {
            let _ = root.ingress().try_submit(RuntimeFact::ModelAvailability(
                ModelAvailabilityFact::with_data(
                    ModelAvailabilityEventKind::NativeModelLoaded,
                    model_scope(model),
                ),
            ));
        }
        AvailabilityOperation {
            root: self.root.take(),
        }
    }

    /// Load failed (native or Rust-side, before availability). Resolves
    /// the `prep` child (if not already resolved), the `load` child, and
    /// the root (never available) reservations with their terminals.
    pub(crate) fn load_failed(self, model: &str) {
        self.load_failed_with_reason(model, ReasonCode::ModelFormatOrLoadFailure);
    }

    /// Same as [`Self::load_failed`] but with an explicit reason -- used by
    /// process-supervision reconciliation when the native process crashed
    /// mid-load rather than an ordinary load error.
    pub(crate) fn load_failed_with_reason(mut self, model: &str, reason: ReasonCode) {
        if let Some(prep) = self.prep.take() {
            let _ = prep.ingress().try_submit(RuntimeFact::ModelPreparation(
                ModelPreparationFact::with_data(
                    ModelPreparationEventKind::ModelResolutionFailed,
                    FactData {
                        outcome: Some(Outcome::Failure),
                        reason: Some(reason.clone()),
                        ..model_scope(model)
                    },
                ),
            ));
        }
        if let Some(ready) = self.ready.take() {
            let _ = ready.ingress().try_submit(RuntimeFact::ModelPreparation(
                ModelPreparationFact::with_data(
                    ModelPreparationEventKind::ModelPreparationFailed,
                    FactData {
                        outcome: Some(Outcome::Failure),
                        reason: Some(reason.clone()),
                        ..model_scope(model)
                    },
                ),
            ));
        }
        if let Some(load) = self.load.take() {
            let _ =
                load.ingress()
                    .try_submit(RuntimeFact::ModelLoading(ModelLoadingFact::with_data(
                        ModelLoadingEventKind::ModelLoadFailed,
                        FactData {
                            outcome: Some(Outcome::Failure),
                            reason: Some(reason.clone()),
                            ..model_scope(model)
                        },
                    )));
        }
        if let Some(root) = self.root.take() {
            let _ = root.ingress().try_submit(RuntimeFact::ModelAvailability(
                ModelAvailabilityFact::with_data(
                    ModelAvailabilityEventKind::ModelUnavailable,
                    FactData {
                        outcome: Some(Outcome::Failure),
                        reason: Some(reason),
                        ..model_scope(model)
                    },
                ),
            ));
        }
    }
}

/// Post-native-load handle. The only holder allowed to submit
/// `ModelAvailable`, and only after Rust serving surfaces are usable.
pub(crate) struct AvailabilityOperation {
    root: Option<OperationReservation>,
}

impl AvailabilityOperation {
    /// §8.4 `Rust backend initialization started` -- call once, right
    /// before Rust begins registering serving surfaces (routing, instance
    /// registry, dashboard) and before `model_available`. StateTransition
    /// class.
    pub(crate) fn rust_backend_initialization_started(&self, model: &str) {
        if let Some(root) = &self.root {
            let _ = root.ingress().try_submit(RuntimeFact::ModelAvailability(
                ModelAvailabilityFact::with_data(
                    ModelAvailabilityEventKind::RustBackendInitializationStarted,
                    model_scope(model),
                ),
            ));
        }
    }

    pub(crate) fn model_available(mut self, model: &str) {
        if let Some(root) = self.root.take() {
            let _ = root.ingress().try_submit(RuntimeFact::ModelAvailability(
                ModelAvailabilityFact::with_data(
                    ModelAvailabilityEventKind::ModelAvailable,
                    FactData {
                        outcome: Some(Outcome::Success),
                        ..model_scope(model)
                    },
                ),
            ));
        }
        emit_available_model_set_changed(model);
    }
}

/// One model-unload operation. Acquired before any unload work begins.
pub(crate) struct UnloadOperation {
    root: Option<OperationReservation>,
}

impl UnloadOperation {
    pub(crate) fn begin(model: &str) -> Self {
        let Some(engine) = runtime_event_engine() else {
            return Self { root: None };
        };
        let root = engine.reserve_root(OperationId::new(), synthetic_unload_terminal);
        if let Some(root) = &root {
            submit_unloading(root, ModelUnloadingEventKind::UnloadRequested, model);
            submit_unloading(root, ModelUnloadingEventKind::UnloadStarted, model);
        }
        Self { root }
    }

    pub(crate) fn session_draining_started(&self, model: &str) {
        if let Some(root) = &self.root {
            submit_unloading(root, ModelUnloadingEventKind::SessionDrainingStarted, model);
        }
    }

    pub(crate) fn session_draining_completed(&self, model: &str) {
        if let Some(root) = &self.root {
            submit_unloading(
                root,
                ModelUnloadingEventKind::SessionDrainingCompleted,
                model,
            );
        }
    }

    /// Emitted when a drain deadline expired and the unload proceeded
    /// without waiting for sessions to finish (§8.5 `forced unload`).
    pub(crate) fn forced(&self, model: &str) {
        if let Some(root) = &self.root {
            submit_unloading(root, ModelUnloadingEventKind::ForcedUnload, model);
        }
    }

    pub(crate) fn completed(mut self, model: &str) {
        if let Some(root) = self.root.take() {
            let _ = root.ingress().try_submit(RuntimeFact::ModelUnloading(
                ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::UnloadCompleted,
                    FactData {
                        outcome: Some(Outcome::Success),
                        ..model_scope(model)
                    },
                ),
            ));
        }
        emit_available_model_set_changed(model);
    }

    pub(crate) fn failed(mut self, model: &str) {
        self.failed_with_reason(model, ReasonCode::InternalRuntimeFailure);
    }

    fn failed_with_reason(&mut self, model: &str, reason: ReasonCode) {
        if let Some(root) = self.root.take() {
            let _ = root.ingress().try_submit(RuntimeFact::ModelUnloading(
                ModelUnloadingFact::with_data(
                    ModelUnloadingEventKind::UnloadFailed,
                    FactData {
                        outcome: Some(Outcome::Failure),
                        reason: Some(reason),
                        ..model_scope(model)
                    },
                ),
            ));
        }
        emit_available_model_set_changed(model);
    }

    /// Reconcile the orderly unload return against process-supervision
    /// observation (the instance's own lifecycle record). An unload that
    /// returns normally while the lifecycle record shows the process had
    /// ALREADY crashed (a race between the exit watcher and this unload)
    /// resolves through `UnloadFailed`/`ProcessCrash`, never `UnloadCompleted`
    /// -- native/process-supervision observation wins over an otherwise
    /// silent "success" return.
    pub(crate) fn reconcile(mut self, model: &str, process_already_crashed: bool) {
        if process_already_crashed {
            self.failed_with_reason(model, ReasonCode::ProcessCrash);
        } else {
            self.completed(model);
        }
    }
}

/// Crash/unexpected-exit reconciliation: the process died without an
/// orderly unload sequence, so there is no live `UnloadOperation` guard to
/// resolve. Reserve-and-immediately-terminate synthesizes the exact same
/// `terminal_not_delivered`-shaped record a dropped guard would produce,
/// through the frozen `ProcessCrash` reason instead.
pub(crate) fn reconcile_process_crash(model: &str) {
    let Some(engine) = runtime_event_engine() else {
        return;
    };
    let Some(root) = engine.reserve_root(OperationId::new(), synthetic_unload_terminal) else {
        return;
    };
    let _ = root.ingress().try_submit(RuntimeFact::ModelAvailability(
        ModelAvailabilityFact::with_data(
            ModelAvailabilityEventKind::ModelUnavailable,
            FactData {
                outcome: Some(Outcome::Failure),
                reason: Some(ReasonCode::ProcessCrash),
                ..model_scope(model)
            },
        ),
    ));
    emit_available_model_set_changed(model);
}

#[cfg(test)]
mod tests {
    use super::{
        LoadOperation, UnloadOperation, reconcile_process_crash, redact_local_path,
        submit_load_progress,
    };
    use crate::runtime_events::config::PROGRESS_EXPORT_INTERVAL;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};
    use mesh_llm_runtime_event_contracts::{
        ModelLoadingEventKind, ModelUnloadingEventKind, Outcome, ProgressUnit, ReasonCode,
        RuntimeFact,
    };
    use std::time::Instant;

    fn last_unloading_reason(engine: &RuntimeEventEngine) -> Option<ReasonCode> {
        engine
            .replay()
            .snapshot()
            .into_iter()
            .rev()
            .find_map(|frame| {
                let RuntimeFact::ModelUnloading(fact) = frame.fact.as_ref() else {
                    return None;
                };
                (*fact.kind() == ModelUnloadingEventKind::UnloadFailed
                    || *fact.kind() == ModelUnloadingEventKind::UnloadCompleted)
                    .then(|| fact.data().reason.clone())
                    .flatten()
            })
    }

    fn last_unloading_outcome(engine: &RuntimeEventEngine) -> Option<Outcome> {
        engine
            .replay()
            .snapshot()
            .into_iter()
            .rev()
            .find_map(|frame| {
                let RuntimeFact::ModelUnloading(fact) = frame.fact.as_ref() else {
                    return None;
                };
                (*fact.kind() == ModelUnloadingEventKind::UnloadFailed
                    || *fact.kind() == ModelUnloadingEventKind::UnloadCompleted)
                    .then_some(fact.data().outcome)
                    .flatten()
            })
    }

    fn install_test_engine() -> std::sync::Arc<RuntimeEventEngine> {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        engine
    }

    /// Every `scope.model_id` a captured frame carries, across every
    /// family this file emits.
    fn all_model_ids(engine: &RuntimeEventEngine) -> Vec<String> {
        engine
            .replay()
            .snapshot()
            .into_iter()
            .filter_map(|frame| {
                let id = match frame.fact.as_ref() {
                    RuntimeFact::ModelPreparation(f) => f.data().scope.model_id.as_ref(),
                    RuntimeFact::ModelLoading(f) => f.data().scope.model_id.as_ref(),
                    RuntimeFact::ModelAvailability(f) => f.data().scope.model_id.as_ref(),
                    _ => None,
                };
                id.map(|id| id.as_str().to_string())
            })
            .collect()
    }

    /// F2 fix regression test (event-system-fixes, live-sampling finding):
    /// a `mesh-llm load <raw filesystem path>` argument must never appear,
    /// in whole or in the raw-path form itself, as any frame's
    /// `scope.model_id` -- the exact leak `.omo/evidence/event-system-fixes/
    /// final/f2/canary.txt` captured live (a directory-canary path leaking
    /// into `model_queued`/`model_resolution_started`/
    /// `model_load_requested`/`model_load_started`).
    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn load_with_a_raw_filesystem_path_never_leaks_it_into_scope_model_id() {
        let engine = install_test_engine();
        let raw_path = "/var/folders/T/opencode/f2/f2dircanary_9f1c2a/model.gguf";
        let op = LoadOperation::begin(raw_path);
        engine.drain();

        let ids = all_model_ids(&engine);
        assert!(
            !ids.is_empty(),
            "begin() must still emit its four pre-resolution facts"
        );
        for id in &ids {
            assert!(
                !id.contains(raw_path),
                "raw path leaked verbatim into scope.model_id: {id}"
            );
            assert!(
                !id.contains("f2dircanary"),
                "directory canary leaked into scope.model_id: {id}"
            );
            assert!(
                !id.contains("/var/folders"),
                "absolute path prefix leaked into scope.model_id: {id}"
            );
            assert!(
                !id.starts_with('/'),
                "scope.model_id must never itself be an absolute path: {id}"
            );
        }

        op.load_failed(raw_path);
        engine.drain();
        let ids_after_failure = all_model_ids(&engine);
        assert!(
            ids_after_failure
                .iter()
                .all(|id| !id.contains(raw_path) && !id.contains("f2dircanary")),
            "a load-failure path (which re-uses the raw, unresolved argument) \
             must also never leak the raw path: {ids_after_failure:?}"
        );
        clear_runtime_event_engine();
    }

    /// `redact_local_path` input -> output table (event-system-fixes
    /// deferral f2-relative-paths, `.omo/evidence/event-system-fixes/
    /// deferrals/f2-relative-paths/analysis.md`). Pins every prefix-anchored
    /// shape the function already collapses to a basename, every
    /// non-path-shaped model reference it must leave untouched, AND the two
    /// literal bare-relative-path-with-directory-components inputs from the
    /// residual report -- which this function DELIBERATELY still leaves
    /// unredacted. That is not an oversight: `models/secretdir/c.gguf` is
    /// structurally identical to a real, canonical Hugging Face exact-file
    /// model ref this codebase produces on the wire today (see
    /// `RemoteCatalogModel::exact_ref()` -> `model_resolver::
    /// format_huggingface_display_ref` -> `format_huggingface_exact_ref`,
    /// and the shipped
    /// `resolve::tests::canonicalize_interest_model_ref_normalizes_huggingface_file_refs`
    /// assertion that `"example-org/example-model-GGUF/example-model-custom.gguf"`
    /// is the CORRECT, unchanged, canonical ref) -- no segment-count,
    /// file-extension, or combined heuristic can redact one without also
    /// corrupting the other. This test exists so a future change to this
    /// function is a conscious, evidenced decision in either direction, not
    /// a silent regression.
    #[test]
    fn redact_local_path_table_covers_every_known_shape_including_the_relative_path_residual() {
        let cases: &[(&str, &str)] = &[
            // Non-path-shaped model references: pass through unchanged.
            ("org/repo", "org/repo"),
            ("org/repo:tag", "org/repo:tag"),
            ("sha256-abc123", "sha256-abc123"),
            ("model.gguf", "model.gguf"),
            // Bare relative paths WITH directory components: the residual
            // this task evaluated and declined to touch. Left unredacted on
            // purpose -- see the doc comment above.
            ("models/secretdir/c.gguf", "models/secretdir/c.gguf"),
            (
                "Users/alice/secretdir/c.gguf",
                "Users/alice/secretdir/c.gguf",
            ),
            // Prefix-anchored local filesystem paths: already redacted to
            // the bare basename by the F2 fix (commit 8354cb835).
            ("/Users/alice/secretdir/c.gguf", "c.gguf"),
            (
                "/var/folders/T/opencode/f2/f2dircanary_9f1c2a/model.gguf",
                "model.gguf",
            ),
            ("C:\\a\\b\\c.gguf", "c.gguf"),
            ("D:/a/b/c.gguf", "c.gguf"),
            ("\\\\srv\\share\\models\\c.gguf", "c.gguf"),
            ("./relative/c.gguf", "c.gguf"),
            ("../relative/c.gguf", "c.gguf"),
            ("~/home/c.gguf", "c.gguf"),
        ];

        for &(input, expected) in cases {
            let actual = redact_local_path(input);
            assert_eq!(
                actual.as_ref(),
                expected,
                "redact_local_path({input:?}) = {actual:?}, expected {expected:?}"
            );
        }
    }

    /// F2 fix: two concurrent loads whose (redacted) basenames coincide
    /// must never collide on the SAME provisional `scope.model_id` --
    /// each provisional id embeds its own operation root, so this proves
    /// uniqueness at the producer boundary (the reducer-side supersede
    /// behavior itself is proven in
    /// `runtime_events::reducer::tests::domain`).
    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn concurrent_loads_with_the_same_basename_get_distinct_provisional_ids() {
        let engine = install_test_engine();
        let _first = LoadOperation::begin("/models/a/model.gguf");
        let _second = LoadOperation::begin("/models/b/model.gguf");
        engine.drain();

        let ids = all_model_ids(&engine);
        let distinct: std::collections::HashSet<&str> =
            ids.iter().map(std::string::String::as_str).collect();
        assert_eq!(
            distinct.len(),
            2,
            "two concurrent loads sharing a basename must get distinct \
             provisional ids, got: {ids:?}"
        );
        clear_runtime_event_engine();
    }

    /// D2 fix (event-system-fixes deferral): `ModelLoadingEventKind::
    /// ModelLoadProgress` had NO production call site anywhere in the
    /// workspace before this change (F2 live-sampling: zero progress-class
    /// ModelLoading frames from any model load). Proves a model load
    /// submits at least one `ModelLoadProgress` fact through
    /// `LoadOperation::progress_ingress`, and that it correlates under the
    /// SAME operation scope as its sibling `model_load_started` fact --
    /// never a fresh, uncorrelated root.
    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn load_progress_reports_under_the_same_operation_scope_as_its_sibling_load_facts() {
        let engine = install_test_engine();
        let op = LoadOperation::begin("org/model");
        engine.drain();

        let sibling_scope = engine
            .replay()
            .snapshot()
            .into_iter()
            .find_map(|frame| {
                let RuntimeFact::ModelLoading(fact) = frame.fact.as_ref() else {
                    return None;
                };
                (*fact.kind() == ModelLoadingEventKind::ModelLoadStarted).then_some(frame.scope)
            })
            .expect("model_load_started must have published a frame");

        let ingress = op
            .progress_ingress()
            .expect("load reservation must still be live before native completion");
        let t0 = Instant::now();
        submit_load_progress(&ingress, "org/model", 250, Some(1000), ProgressUnit::Steps);

        // Progress-class facts are coalesced per-slot and flushed only once
        // the engine's frozen 100ms PROGRESS_EXPORT_INTERVAL has elapsed
        // (`runtime_events::engine::lanes::submit_progress` +
        // `engine::drain::maybe_flush_progress`) -- mirrors
        // `engine::tests::classes::progress_flushes_at_most_once_per_hundred_milliseconds_with_the_latest_value`'s
        // own pure-`Instant`-arithmetic pattern, never a real sleep.
        let flushed = engine.drain_up_to_at(None, t0 + PROGRESS_EXPORT_INTERVAL);
        assert_eq!(
            flushed.applied, 1,
            "the coalesced ModelLoadProgress fact must flush once the interval elapses"
        );

        let progress_scope = engine
            .replay()
            .snapshot()
            .into_iter()
            .rev()
            .find_map(|frame| {
                let RuntimeFact::ModelLoading(fact) = frame.fact.as_ref() else {
                    return None;
                };
                (*fact.kind() == ModelLoadingEventKind::ModelLoadProgress).then_some(frame.scope)
            })
            .expect("ModelLoadProgress must have published a frame");

        assert_eq!(
            progress_scope, sibling_scope,
            "ModelLoadProgress must correlate under the SAME operation root as model_load_started"
        );

        op.load_failed("org/model");
        engine.drain();
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn load_reserves_root_prep_ready_and_load_children_before_native_completion() {
        let engine = install_test_engine();
        let mut op = LoadOperation::begin("org/model");
        // root + prep (§8.2 resolution) + ready (§8.2 preparation) + load
        // (§8.3) all reserved up front.
        assert_eq!(engine.occupied_count(), 4);
        op.resolution_completed("org/model");
        engine.drain();
        // resolution_completed resolves BOTH prep and ready in one call.
        assert_eq!(engine.occupied_count(), 2);
        op.capacity_changed("org/model");
        op.backend_device_selected("org/model", "metal");
        let availability = op.native_load_completed("org/model");
        engine.drain();
        assert_eq!(engine.occupied_count(), 1);
        availability.model_available("org/model");
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn load_failure_resolves_all_four_reservations() {
        let engine = install_test_engine();
        let op = LoadOperation::begin("org/model");
        assert_eq!(engine.occupied_count(), 4);
        op.load_failed("org/model");
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn load_failure_after_resolution_completed_still_resolves_all_four() {
        let engine = install_test_engine();
        let mut op = LoadOperation::begin("org/model");
        op.resolution_completed("org/model");
        engine.drain();
        assert_eq!(engine.occupied_count(), 2);
        op.load_failed("org/model");
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn unload_forced_does_not_prevent_normal_completion() {
        let engine = install_test_engine();
        let op = UnloadOperation::begin("org/model");
        op.forced("org/model");
        op.completed("org/model");
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        assert_eq!(last_unloading_outcome(&engine), Some(Outcome::Success));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn dropping_load_operation_without_a_terminal_synthesizes_one() {
        let engine = install_test_engine();
        {
            let op = LoadOperation::begin("org/model");
            assert_eq!(engine.occupied_count(), 4);
            drop(op);
        }
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn unload_completes_with_one_terminal() {
        let engine = install_test_engine();
        let op = UnloadOperation::begin("org/model");
        assert_eq!(engine.occupied_count(), 1);
        op.session_draining_started("org/model");
        op.session_draining_completed("org/model");
        op.completed("org/model");
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn unload_failure_resolves_the_reservation() {
        let engine = install_test_engine();
        let op = UnloadOperation::begin("org/model");
        op.failed("org/model");
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn reconcile_reports_completed_when_process_did_not_crash() {
        let engine = install_test_engine();
        let op = UnloadOperation::begin("org/model");
        op.reconcile("org/model", false);
        engine.drain();
        assert_eq!(
            last_unloading_reason(&engine),
            None,
            "a clean unload carries no failure reason"
        );
        assert_eq!(last_unloading_outcome(&engine), Some(Outcome::Success));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn reconcile_reports_process_crash_when_lifecycle_already_failed() {
        let engine = install_test_engine();
        let op = UnloadOperation::begin("org/model");
        // The orderly unload path returned normally, but process
        // supervision (the lifecycle record) already observed a crash --
        // reconciliation must report the crash, not a false "completed".
        op.reconcile("org/model", true);
        engine.drain();
        assert_eq!(
            last_unloading_reason(&engine),
            Some(ReasonCode::ProcessCrash)
        );
        assert_eq!(last_unloading_outcome(&engine), Some(Outcome::Failure));
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn process_crash_reconciliation_synthesizes_a_terminal_without_a_live_guard() {
        let engine = install_test_engine();
        reconcile_process_crash("org/model");
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn absent_engine_degrades_to_no_op_and_never_panics() {
        clear_runtime_event_engine();
        let op = LoadOperation::begin("org/model");
        let availability = op.native_load_completed("org/model");
        availability.model_available("org/model");
        let unload = UnloadOperation::begin("org/model");
        unload.completed("org/model");
        reconcile_process_crash("org/model");
        // No assertions beyond "did not panic": there is no engine to
        // inspect, which is exactly the degraded-but-not-failing contract.
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn reservation_exhaustion_degrades_load_to_a_no_op_without_failing() {
        let engine = crate::runtime_events::engine::RuntimeEventEngine::with_capacity(0);
        clear_runtime_event_engine();
        install_runtime_event_engine(engine.clone());
        // Capacity 0: every reserve call is exhausted immediately.
        let op = LoadOperation::begin("org/model");
        let availability = op.native_load_completed("org/model");
        availability.model_available("org/model");
        assert!(engine.health().snapshot().reservation_exhausted > 0);
        clear_runtime_event_engine();
    }

    /// Plan line 274's sharpest invariant: forced reservation exhaustion
    /// must proceed with degraded observability, never refuse primary
    /// work. Every method on `LoadOperation`/`AvailabilityOperation`/
    /// `UnloadOperation` returns `()`, not `Result` -- there is no error
    /// value for a caller (`load.rs`/`unload.rs`) to propagate even if it
    /// wanted to, so "serving proceeds under exhaustion" is a structural
    /// property of this API's shape, not just this test's behavior. This
    /// test proves the full load+unload call sequence a real model
    /// load/unload performs runs to completion, unwinds no panic, and the
    /// ONLY observable effect is the counted `reservation_exhausted` health
    /// metric -- exactly "observability degrades; serving does not".
    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn reservation_exhaustion_proceeds_through_a_full_load_and_unload_sequence() {
        let engine = crate::runtime_events::engine::RuntimeEventEngine::with_capacity(0);
        clear_runtime_event_engine();
        install_runtime_event_engine(engine.clone());

        let load = LoadOperation::begin("org/model");
        let availability = load.native_load_completed("org/model");
        availability.model_available("org/model");

        let unload = UnloadOperation::begin("org/model");
        unload.session_draining_started("org/model");
        unload.session_draining_completed("org/model");
        unload.reconcile("org/model", false);

        reconcile_process_crash("org/model");

        // Every reserve above was exhausted (capacity 0), yet the entire
        // sequence above ran to completion with no panic and no engine
        // state to even release (occupied_count stays 0 throughout,
        // because nothing was ever actually reserved).
        assert_eq!(engine.occupied_count(), 0);
        // LoadOperation::begin (root only, since it fails before the child
        // attempt), UnloadOperation::begin, and reconcile_process_crash
        // each attempt exactly one reservation at capacity 0.
        assert!(engine.health().snapshot().reservation_exhausted >= 3);
        clear_runtime_event_engine();
    }

    /// Task 10 addition: `available_model_set_changed` (§8.14) co-fires on
    /// every point this file already marks the served model set as
    /// changed -- added, not altered; every assertion above this line is
    /// untouched and still passes unmodified.
    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn model_available_co_emits_available_model_set_changed() {
        let engine = install_test_engine();
        let op = LoadOperation::begin("org/model");
        let availability = op.native_load_completed("org/model");
        engine.drain();
        availability.model_available("org/model");
        assert!(
            engine
                .state_lane_kinds()
                .contains(&"available_model_set_changed")
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn unload_completed_and_failed_both_co_emit_available_model_set_changed() {
        let engine = install_test_engine();
        let op = UnloadOperation::begin("org/model");
        op.completed("org/model");
        assert!(
            engine
                .state_lane_kinds()
                .contains(&"available_model_set_changed"),
            "unload completed must report the model left the available set"
        );
        clear_runtime_event_engine();

        let engine = install_test_engine();
        let op = UnloadOperation::begin("org/model");
        op.failed("org/model");
        assert!(
            engine
                .state_lane_kinds()
                .contains(&"available_model_set_changed"),
            "a failed unload still leaves the model no longer normally served"
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn process_crash_reconciliation_co_emits_available_model_set_changed() {
        let engine = install_test_engine();
        reconcile_process_crash("org/model");
        assert!(
            engine
                .state_lane_kinds()
                .contains(&"available_model_set_changed")
        );
        clear_runtime_event_engine();
    }
}
