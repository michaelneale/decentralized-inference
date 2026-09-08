use crate::{
    DeliveryClass, DiagnosticEventKind, EventSystemHealthEventKind, GenerationEventKind,
    KvRuntimeStateEventKind, NodeAvailabilityEventKind, PrefillEventKind, RequestEventKind,
    ResourceHealthEventKind,
};

pub(super) const fn request(kind: RequestEventKind) -> DeliveryClass {
    match kind {
        RequestEventKind::RequestRejected
        | RequestEventKind::RequestCompleted
        | RequestEventKind::RequestCancelled
        | RequestEventKind::RequestTimedOut
        | RequestEventKind::RequestFailed => DeliveryClass::Terminal,
        RequestEventKind::RequestReceived
        | RequestEventKind::RequestQueued
        | RequestEventKind::RequestAdmitted
        | RequestEventKind::RequestExecutionStarted => DeliveryClass::StateTransition,
    }
}

pub(super) const fn prefill(kind: PrefillEventKind) -> DeliveryClass {
    match kind {
        PrefillEventKind::PrefillProgress => DeliveryClass::Progress,
        PrefillEventKind::TokenizationCompleted
        | PrefillEventKind::TokenizationFailed
        | PrefillEventKind::PrefillCompleted
        | PrefillEventKind::PrefillCancelled
        | PrefillEventKind::PrefillFailed
        | PrefillEventKind::MediaPrefillCompleted
        | PrefillEventKind::MediaPrefillFailed => DeliveryClass::Terminal,
        PrefillEventKind::PromptProcessingStarted
        | PrefillEventKind::PrefillStarted
        | PrefillEventKind::MediaPrefillStarted
        | PrefillEventKind::PromptCacheRestoreHit
        | PrefillEventKind::PromptCacheRestoreMiss
        | PrefillEventKind::PromptCacheRestorePartial
        | PrefillEventKind::PromptCacheRestoreError => DeliveryClass::StateTransition,
    }
}

pub(super) const fn generation(kind: GenerationEventKind) -> DeliveryClass {
    match kind {
        GenerationEventKind::GenerationProgress => DeliveryClass::Progress,
        GenerationEventKind::GenerationCompleted
        | GenerationEventKind::GenerationCancelled
        | GenerationEventKind::GenerationTimedOut
        | GenerationEventKind::GenerationFailed => DeliveryClass::Terminal,
        GenerationEventKind::GenerationStarted
        | GenerationEventKind::FirstTokenProduced
        | GenerationEventKind::StopConditionReached => DeliveryClass::StateTransition,
    }
}

pub(super) const fn kv_runtime_state(kind: KvRuntimeStateEventKind) -> DeliveryClass {
    match kind {
        KvRuntimeStateEventKind::ContextExhausted => DeliveryClass::Terminal,
        KvRuntimeStateEventKind::KvCacheInitializationStarted
        | KvRuntimeStateEventKind::KvCacheInitializationCompleted
        | KvRuntimeStateEventKind::KvCacheInitializationFailed
        | KvRuntimeStateEventKind::CacheLookupHit
        | KvRuntimeStateEventKind::CacheLookupMiss
        | KvRuntimeStateEventKind::CacheLookupPartial
        | KvRuntimeStateEventKind::CacheLookupError
        | KvRuntimeStateEventKind::PrefixRestored
        | KvRuntimeStateEventKind::CheckpointRestored
        | KvRuntimeStateEventKind::CacheRecordCompleted
        | KvRuntimeStateEventKind::CacheRecordFailed
        | KvRuntimeStateEventKind::CacheTrim
        | KvRuntimeStateEventKind::CacheEviction
        | KvRuntimeStateEventKind::CacheReset
        | KvRuntimeStateEventKind::CachePressureCrossed
        | KvRuntimeStateEventKind::CachePressureCleared
        | KvRuntimeStateEventKind::ContextCapacityApproachingLimit
        | KvRuntimeStateEventKind::RuntimeStateImportCompleted
        | KvRuntimeStateEventKind::RuntimeStateImportFailed
        | KvRuntimeStateEventKind::RuntimeStateExportCompleted
        | KvRuntimeStateEventKind::RuntimeStateExportFailed => DeliveryClass::StateTransition,
    }
}

pub(super) const fn resource_health(kind: ResourceHealthEventKind) -> DeliveryClass {
    match kind {
        ResourceHealthEventKind::BackendInitializationFailed
        | ResourceHealthEventKind::ResourceAllocationFailed
        | ResourceHealthEventKind::OutOfMemoryCondition => DeliveryClass::Terminal,
        ResourceHealthEventKind::BackendInitializationStarted
        | ResourceHealthEventKind::BackendInitializationCompleted
        | ResourceHealthEventKind::DeviceSelected
        | ResourceHealthEventKind::DeviceReady
        | ResourceHealthEventKind::DeviceDegraded
        | ResourceHealthEventKind::DeviceUnavailable
        | ResourceHealthEventKind::DeviceRecovered
        | ResourceHealthEventKind::ResourceAllocationCompleted
        | ResourceHealthEventKind::MemoryPressureCrossed
        | ResourceHealthEventKind::MemoryPressureCleared
        | ResourceHealthEventKind::BackendFallbackActivated
        | ResourceHealthEventKind::CpuFallbackActivated
        | ResourceHealthEventKind::ComputeFailure
        | ResourceHealthEventKind::DeviceLost
        | ResourceHealthEventKind::DeviceReset => DeliveryClass::StateTransition,
    }
}

pub(super) const fn diagnostic(kind: DiagnosticEventKind) -> DeliveryClass {
    match kind {
        DiagnosticEventKind::FatalNativeFailure => DeliveryClass::Terminal,
        DiagnosticEventKind::WarningRaised
        | DiagnosticEventKind::WarningCleared
        | DiagnosticEventKind::RecoverableNativeFailure
        | DiagnosticEventKind::FallbackApplied
        | DiagnosticEventKind::DegradedOperationEntered
        | DiagnosticEventKind::DegradedOperationExited
        | DiagnosticEventKind::InvariantProtocolViolation => DeliveryClass::Diagnostic,
    }
}

pub(super) const fn node_availability(kind: NodeAvailabilityEventKind) -> DeliveryClass {
    match kind {
        NodeAvailabilityEventKind::NodeStopped => DeliveryClass::Terminal,
        NodeAvailabilityEventKind::NodeStarting
        | NodeAvailabilityEventKind::NodeAcceptingRequests
        | NodeAvailabilityEventKind::NodeDegraded
        | NodeAvailabilityEventKind::NodeUnavailable
        | NodeAvailabilityEventKind::NodeDraining
        | NodeAvailabilityEventKind::AvailableModelSetChanged
        | NodeAvailabilityEventKind::AvailableStageSetChanged
        | NodeAvailabilityEventKind::RequestCapacityChanged
        | NodeAvailabilityEventKind::LaneCapacityChanged
        | NodeAvailabilityEventKind::SessionCapacityChanged
        | NodeAvailabilityEventKind::ResourcePressureChanged => DeliveryClass::StateTransition,
    }
}

pub(super) const fn event_system_health(kind: EventSystemHealthEventKind) -> DeliveryClass {
    match kind {
        EventSystemHealthEventKind::IngressQueuePressure
        | EventSystemHealthEventKind::EventsCoalesced
        | EventSystemHealthEventKind::EventsSampled
        | EventSystemHealthEventKind::EventsDroppedByClass
        | EventSystemHealthEventKind::SubscriberLagging
        | EventSystemHealthEventKind::SubscriberDisconnected
        | EventSystemHealthEventKind::ReducerError
        | EventSystemHealthEventKind::TelemetryExporterDegraded
        | EventSystemHealthEventKind::TelemetryExporterRecovered
        | EventSystemHealthEventKind::EventSchemaIncompatibility
        | EventSystemHealthEventKind::UnknownNativeEventReceived => DeliveryClass::StateTransition,
    }
}
