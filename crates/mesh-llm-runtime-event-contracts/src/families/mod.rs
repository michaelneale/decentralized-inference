mod execution;
mod lifecycle;

pub use execution::*;
pub use lifecycle::*;

#[must_use]
pub fn all_event_ids() -> Vec<&'static str> {
    let mut ids = Vec::with_capacity(184);
    macro_rules! append {
        ($family:ty) => {
            ids.extend(<$family>::ALL.iter().copied().map(<$family>::as_str));
        };
    }
    append!(NativeRuntimeEventKind);
    append!(ModelPreparationEventKind);
    append!(ModelLoadingEventKind);
    append!(ModelAvailabilityEventKind);
    append!(ModelUnloadingEventKind);
    append!(StageTopologyEventKind);
    append!(SessionEventKind);
    append!(RequestEventKind);
    append!(PrefillEventKind);
    append!(GenerationEventKind);
    append!(KvRuntimeStateEventKind);
    append!(ResourceHealthEventKind);
    append!(DiagnosticEventKind);
    append!(NodeAvailabilityEventKind);
    append!(EventSystemHealthEventKind);
    ids
}
