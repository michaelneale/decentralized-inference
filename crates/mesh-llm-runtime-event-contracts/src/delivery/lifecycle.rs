use crate::{
    DeliveryClass, ModelAvailabilityEventKind, ModelLoadingEventKind, ModelPreparationEventKind,
    ModelUnloadingEventKind, NativeRuntimeEventKind, SessionEventKind, StageTopologyEventKind,
};

pub(super) const fn native_runtime(kind: NativeRuntimeEventKind) -> DeliveryClass {
    match kind {
        NativeRuntimeEventKind::RuntimeResolutionCompleted
        | NativeRuntimeEventKind::RuntimeResolutionFailed
        | NativeRuntimeEventKind::RuntimeStopped
        | NativeRuntimeEventKind::RuntimeCrashed => DeliveryClass::Terminal,
        NativeRuntimeEventKind::RuntimeResolutionStarted
        | NativeRuntimeEventKind::NativeLibraryLoaded
        | NativeRuntimeEventKind::NativeLibraryRejected
        | NativeRuntimeEventKind::NativeLibraryUnavailable
        | NativeRuntimeEventKind::AbiFeatureCompatibilityEstablished
        | NativeRuntimeEventKind::AbiFeatureCompatibilityFailed
        | NativeRuntimeEventKind::RuntimeInitialized
        | NativeRuntimeEventKind::RuntimeStopping => DeliveryClass::StateTransition,
    }
}

pub(super) const fn model_preparation(kind: ModelPreparationEventKind) -> DeliveryClass {
    match kind {
        ModelPreparationEventKind::ModelDownloadProgress
        | ModelPreparationEventKind::ModelPreparationProgress => DeliveryClass::Progress,
        ModelPreparationEventKind::ModelResolutionCompleted
        | ModelPreparationEventKind::ModelResolutionFailed
        | ModelPreparationEventKind::ModelDownloadCompleted
        | ModelPreparationEventKind::ModelDownloadFailed
        | ModelPreparationEventKind::ModelDownloadCancelled
        | ModelPreparationEventKind::ModelPreparationCompleted
        | ModelPreparationEventKind::ModelPreparationFailed
        | ModelPreparationEventKind::ModelPreparationCancelled => DeliveryClass::Terminal,
        ModelPreparationEventKind::ModelQueued
        | ModelPreparationEventKind::ModelResolutionStarted
        | ModelPreparationEventKind::ModelDownloadStarted
        | ModelPreparationEventKind::ModelPreparationStarted => DeliveryClass::StateTransition,
    }
}

pub(super) const fn model_loading(kind: ModelLoadingEventKind) -> DeliveryClass {
    match kind {
        ModelLoadingEventKind::ModelLoadProgress => DeliveryClass::Progress,
        ModelLoadingEventKind::NativeModelLoadCompleted
        | ModelLoadingEventKind::ModelLoadFailed
        | ModelLoadingEventKind::ModelLoadCancelled => DeliveryClass::Terminal,
        ModelLoadingEventKind::ModelLoadRequested
        | ModelLoadingEventKind::ModelLoadStarted
        | ModelLoadingEventKind::ModelLoadPhaseChanged
        | ModelLoadingEventKind::BackendDeviceSelected
        | ModelLoadingEventKind::ModelMemoryAllocationSummary
        | ModelLoadingEventKind::ModelMemoryPressure => DeliveryClass::StateTransition,
    }
}

pub(super) const fn model_availability(kind: ModelAvailabilityEventKind) -> DeliveryClass {
    match kind {
        ModelAvailabilityEventKind::ModelAvailable
        | ModelAvailabilityEventKind::ModelUnavailable
        | ModelAvailabilityEventKind::ModelRecoveryCompleted
        | ModelAvailabilityEventKind::ModelRecoveryFailed => DeliveryClass::Terminal,
        ModelAvailabilityEventKind::NativeModelLoaded
        | ModelAvailabilityEventKind::RustBackendInitializationStarted
        | ModelAvailabilityEventKind::ModelDegraded
        | ModelAvailabilityEventKind::ModelRecoveryStarted
        | ModelAvailabilityEventKind::ModelCapacityChanged => DeliveryClass::StateTransition,
    }
}

pub(super) const fn model_unloading(kind: ModelUnloadingEventKind) -> DeliveryClass {
    match kind {
        ModelUnloadingEventKind::UnloadCompleted | ModelUnloadingEventKind::UnloadFailed => {
            DeliveryClass::Terminal
        }
        ModelUnloadingEventKind::UnloadRequested
        | ModelUnloadingEventKind::UnloadStarted
        | ModelUnloadingEventKind::SessionDrainingStarted
        | ModelUnloadingEventKind::SessionDrainingCompleted
        | ModelUnloadingEventKind::ForcedUnload => DeliveryClass::StateTransition,
    }
}

pub(super) const fn stage_topology(kind: StageTopologyEventKind) -> DeliveryClass {
    match kind {
        StageTopologyEventKind::StageReady
        | StageTopologyEventKind::StageStopped
        | StageTopologyEventKind::StageFailed
        | StageTopologyEventKind::TopologyReady
        | StageTopologyEventKind::TopologyUnavailable => DeliveryClass::Terminal,
        StageTopologyEventKind::StageStarting
        | StageTopologyEventKind::StageLoading
        | StageTopologyEventKind::StageDegraded
        | StageTopologyEventKind::StageUnavailable
        | StageTopologyEventKind::StageStopping
        | StageTopologyEventKind::TopologyAssembling
        | StageTopologyEventKind::TopologyDegraded
        | StageTopologyEventKind::StageConnectionEstablished
        | StageTopologyEventKind::StageConnectionLost
        | StageTopologyEventKind::StageConnectionRecovered => DeliveryClass::StateTransition,
    }
}

pub(super) const fn session(kind: SessionEventKind) -> DeliveryClass {
    match kind {
        SessionEventKind::SessionClosed | SessionEventKind::SessionFailed => {
            DeliveryClass::Terminal
        }
        SessionEventKind::SessionRequested
        | SessionEventKind::SessionCreated
        | SessionEventKind::SessionActive
        | SessionEventKind::SessionIdle
        | SessionEventKind::SessionReusable
        | SessionEventKind::SessionReset
        | SessionEventKind::SessionTrimmed
        | SessionEventKind::SessionRestoredFromPrefixCache
        | SessionEventKind::SessionRestoredFromCheckpoint
        | SessionEventKind::SessionDraining
        | SessionEventKind::SessionAbandoned
        | SessionEventKind::SessionReclaimed => DeliveryClass::StateTransition,
    }
}
