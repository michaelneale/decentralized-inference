use crate::runtime::operational_logging::{
    NativeSkippyOperationalEvent, record_native_skippy_operational_event,
};
use crate::runtime_events::engine::ScopedIngress;
use mesh_llm_events::{OutputEvent, emit_event};
use mesh_llm_runtime_event_contracts::ProgressUnit;
use skippy_runtime::{
    RuntimeEvent as SkippyNativeRuntimeEvent, RuntimeEventKind as SkippyNativeRuntimeEventKind,
    RuntimeEventProgressUnit as SkippyNativeRuntimeProgressUnit,
};

/// Maps the native model-open progress unit onto the runtime-event
/// contracts' unit -- a direct 1:1 correspondence except `Unknown`, which
/// degrades to `None` rather than guessing a unit the native side did not
/// actually report.
fn translate_progress_unit(unit: SkippyNativeRuntimeProgressUnit) -> ProgressUnit {
    match unit {
        SkippyNativeRuntimeProgressUnit::None => ProgressUnit::None,
        SkippyNativeRuntimeProgressUnit::Bytes => ProgressUnit::Bytes,
        SkippyNativeRuntimeProgressUnit::Items => ProgressUnit::Items,
        SkippyNativeRuntimeProgressUnit::Tensors => ProgressUnit::Tensors,
        SkippyNativeRuntimeProgressUnit::Steps => ProgressUnit::Steps,
        SkippyNativeRuntimeProgressUnit::Unknown(_) => ProgressUnit::None,
    }
}

fn skippy_native_runtime_event_context(
    sequence: u64,
    status: &str,
    emitter: &str,
) -> Option<String> {
    Some(
        [
            format!("sequence={sequence}"),
            format!("status={status}"),
            format!("emitter={emitter}"),
        ]
        .join(" "),
    )
}

struct SkippyNativeRuntimeEventSnapshot<'a> {
    kind: SkippyNativeRuntimeEventKind,
    sequence: u64,
    status: &'a str,
    emitter: &'a str,
    progress_current: u64,
    progress_total: u64,
    progress_unit: SkippyNativeRuntimeProgressUnit,
}

fn translate_skippy_native_runtime_event_snapshot(
    snapshot: SkippyNativeRuntimeEventSnapshot<'_>,
) -> Option<OutputEvent> {
    let context =
        skippy_native_runtime_event_context(snapshot.sequence, snapshot.status, snapshot.emitter);
    match snapshot.kind {
        SkippyNativeRuntimeEventKind::ModelOpenStarted => Some(OutputEvent::Info {
            message: "Native runtime started opening model".to_string(),
            context,
        }),
        SkippyNativeRuntimeEventKind::ModelOpenProgress => {
            let progress = match (
                snapshot.progress_current,
                snapshot.progress_total,
                snapshot.progress_unit,
            ) {
                (current, total, SkippyNativeRuntimeProgressUnit::Steps) if total > 0 => {
                    format!("{}%", current.saturating_mul(100) / total)
                }
                (current, total, unit) if total > 0 => {
                    format!("{current}/{total} {unit:?}")
                }
                (current, _, unit) => format!("{current} {unit:?}"),
            };
            Some(OutputEvent::Info {
                message: format!("Opening native model {progress}"),
                context,
            })
        }
        SkippyNativeRuntimeEventKind::BackendDeviceSelected => Some(OutputEvent::Info {
            message: "Native runtime selected a backend device".to_string(),
            context,
        }),
        SkippyNativeRuntimeEventKind::ModelOpenFinished => Some(OutputEvent::Info {
            message: "Native runtime finished opening model; waiting for Rust runtime readiness"
                .to_string(),
            context,
        }),
        SkippyNativeRuntimeEventKind::ModelOpenFailedHandled => Some(OutputEvent::Warning {
            message: "Native runtime reported a handled model-open failure".to_string(),
            context,
        }),
        // Every other kind, including the runtime-scoped reporter's own
        // families (task 10, `.omo/plans/event-system-fixes.md`), never
        // reaches this per-call model-open bridge.
        _ => None,
    }
}

fn translate_skippy_native_runtime_event(event: &SkippyNativeRuntimeEvent) -> Option<OutputEvent> {
    let status = format!("{:?}", event.status);
    let emitter = format!("{:?}", event.emitter);
    translate_skippy_native_runtime_event_snapshot(SkippyNativeRuntimeEventSnapshot {
        kind: event.kind,
        sequence: event.sequence,
        status: &status,
        emitter: &emitter,
        progress_current: event.progress_current,
        progress_total: event.progress_total,
        progress_unit: event.progress_unit,
    })
}

fn native_skippy_operational_event(
    kind: SkippyNativeRuntimeEventKind,
) -> Option<NativeSkippyOperationalEvent> {
    match kind {
        SkippyNativeRuntimeEventKind::ModelOpenStarted => {
            Some(NativeSkippyOperationalEvent::ModelOpenStarted)
        }
        SkippyNativeRuntimeEventKind::ModelOpenFinished => {
            Some(NativeSkippyOperationalEvent::ModelOpenFinished)
        }
        SkippyNativeRuntimeEventKind::ModelOpenFailedHandled => {
            Some(NativeSkippyOperationalEvent::ModelOpenFailed)
        }
        SkippyNativeRuntimeEventKind::ModelOpenProgress
        | SkippyNativeRuntimeEventKind::BackendDeviceSelected => None,
        // Every other kind, including the runtime-scoped reporter's own
        // families, never reaches this per-call model-open bridge.
        _ => None,
    }
}

fn emit_skippy_native_runtime_event(event: SkippyNativeRuntimeEvent) {
    if let Some(operational_event) = native_skippy_operational_event(event.kind) {
        record_native_skippy_operational_event(operational_event);
    }
    let Some(output_event) = translate_skippy_native_runtime_event(&event) else {
        return;
    };
    let _ = emit_event(output_event);
}

/// `progress_ingress` is `Some` only on the single-node runtime-load path
/// (`LoadOperation::progress_ingress`, event-system-fixes deferral D2) --
/// every other caller passes `None` and this reporter simply skips the
/// `ModelLoadProgress` co-emission below, unaffected otherwise.
pub(super) fn skippy_native_model_open_event_reporter(
    model_name: String,
    progress_ingress: Option<ScopedIngress>,
) -> crate::inference::skippy::NativeModelOpenEventReporter {
    Box::new(move |event: SkippyNativeRuntimeEvent| {
        #[cfg(feature = "dynamic-native-runtime")]
        if let Some(ingress) = progress_ingress.as_ref() {
            submit_scoped_native_observation(ingress, &model_name, &event);
        }
        #[cfg(not(feature = "dynamic-native-runtime"))]
        if event.kind == SkippyNativeRuntimeEventKind::ModelOpenProgress
            && let Some(ingress) = progress_ingress.as_ref()
        {
            let total = (event.progress_total > 0).then_some(event.progress_total);
            crate::runtime::model_lifecycle::submit_load_progress(
                ingress,
                &model_name,
                event.progress_current,
                total,
                translate_progress_unit(event.progress_unit),
            );
        }
        emit_skippy_native_runtime_event(event);
    })
}

#[cfg(feature = "dynamic-native-runtime")]
fn submit_scoped_native_observation(
    ingress: &ScopedIngress,
    model: &str,
    event: &SkippyNativeRuntimeEvent,
) {
    use crate::system::native_runtime_events::{
        native_model_open_metadata, register_native_model_operation,
    };
    use mesh_llm_runtime_event_contracts::{ModelLoadingEventKind, Progress, RuntimeEventIngress};

    let kind = match event.kind {
        SkippyNativeRuntimeEventKind::ModelOpenProgress => ModelLoadingEventKind::ModelLoadProgress,
        SkippyNativeRuntimeEventKind::BackendDeviceSelected => {
            ModelLoadingEventKind::BackendDeviceSelected
        }
        SkippyNativeRuntimeEventKind::ModelLoadMemoryAllocated => {
            ModelLoadingEventKind::ModelMemoryAllocationSummary
        }
        SkippyNativeRuntimeEventKind::ModelOpenStarted
        | SkippyNativeRuntimeEventKind::ModelOpenFinished
        | SkippyNativeRuntimeEventKind::ModelOpenFailedHandled
        | SkippyNativeRuntimeEventKind::ModelLoadPhaseChanged
        | SkippyNativeRuntimeEventKind::ModelLoadTensorsOffloaded
        | SkippyNativeRuntimeEventKind::ModelLoadTokenizerReady
        | SkippyNativeRuntimeEventKind::ModelLoadAuxComponentReady => {
            ModelLoadingEventKind::ModelLoadPhaseChanged
        }
        _ => return,
    };
    let progress = (event.kind == SkippyNativeRuntimeEventKind::ModelOpenProgress).then(|| {
        Progress::new(
            event.progress_current,
            (event.progress_total > 0).then_some(event.progress_total),
            translate_progress_unit(event.progress_unit),
        )
    });
    let fact = crate::runtime::model_lifecycle::native_load_observation(
        model,
        kind,
        native_model_open_metadata(event),
        progress,
    );
    register_native_model_operation(
        event.model_id,
        fact.data().scope.model_id.clone(),
        ingress.scope(),
    );
    let _ = ingress.try_submit(fact);
}

#[cfg(test)]
mod tests;
