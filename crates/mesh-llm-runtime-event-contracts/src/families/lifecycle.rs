event_family!(NativeRuntimeEventKind {
    RuntimeResolutionStarted => "runtime_resolution_started",
    RuntimeResolutionCompleted => "runtime_resolution_completed",
    RuntimeResolutionFailed => "runtime_resolution_failed",
    NativeLibraryLoaded => "native_library_loaded",
    NativeLibraryRejected => "native_library_rejected",
    NativeLibraryUnavailable => "native_library_unavailable",
    AbiFeatureCompatibilityEstablished => "abi_feature_compatibility_established",
    AbiFeatureCompatibilityFailed => "abi_feature_compatibility_failed",
    RuntimeInitialized => "runtime_initialized",
    RuntimeStopping => "runtime_stopping",
    RuntimeStopped => "runtime_stopped",
    RuntimeCrashed => "runtime_crashed",
});

event_family!(ModelPreparationEventKind {
    ModelQueued => "model_queued",
    ModelResolutionStarted => "model_resolution_started",
    ModelResolutionCompleted => "model_resolution_completed",
    ModelResolutionFailed => "model_resolution_failed",
    ModelDownloadStarted => "model_download_started",
    ModelDownloadProgress => "model_download_progress",
    ModelDownloadCompleted => "model_download_completed",
    ModelDownloadFailed => "model_download_failed",
    ModelDownloadCancelled => "model_download_cancelled",
    ModelPreparationStarted => "model_preparation_started",
    ModelPreparationProgress => "model_preparation_progress",
    ModelPreparationCompleted => "model_preparation_completed",
    ModelPreparationFailed => "model_preparation_failed",
    ModelPreparationCancelled => "model_preparation_cancelled",
});

event_family!(ModelLoadingEventKind {
    ModelLoadRequested => "model_load_requested",
    ModelLoadStarted => "model_load_started",
    ModelLoadPhaseChanged => "model_load_phase_changed",
    ModelLoadProgress => "model_load_progress",
    BackendDeviceSelected => "backend_device_selected",
    ModelMemoryAllocationSummary => "model_memory_allocation_summary",
    ModelMemoryPressure => "model_memory_pressure",
    NativeModelLoadCompleted => "native_model_load_completed",
    ModelLoadFailed => "model_load_failed",
    ModelLoadCancelled => "model_load_cancelled",
});

event_family!(ModelAvailabilityEventKind {
    NativeModelLoaded => "native_model_loaded",
    RustBackendInitializationStarted => "rust_backend_initialization_started",
    ModelAvailable => "model_available",
    ModelDegraded => "model_degraded",
    ModelUnavailable => "model_unavailable",
    ModelRecoveryStarted => "model_recovery_started",
    ModelRecoveryCompleted => "model_recovery_completed",
    ModelRecoveryFailed => "model_recovery_failed",
    ModelCapacityChanged => "model_capacity_changed",
});

event_family!(ModelUnloadingEventKind {
    UnloadRequested => "unload_requested",
    UnloadStarted => "unload_started",
    SessionDrainingStarted => "session_draining_started",
    SessionDrainingCompleted => "session_draining_completed",
    UnloadCompleted => "unload_completed",
    UnloadFailed => "unload_failed",
    ForcedUnload => "forced_unload",
});

event_family!(StageTopologyEventKind {
    StageStarting => "stage_starting",
    StageLoading => "stage_loading",
    StageReady => "stage_ready",
    StageDegraded => "stage_degraded",
    StageUnavailable => "stage_unavailable",
    StageStopping => "stage_stopping",
    StageStopped => "stage_stopped",
    StageFailed => "stage_failed",
    TopologyAssembling => "topology_assembling",
    TopologyReady => "topology_ready",
    TopologyDegraded => "topology_degraded",
    TopologyUnavailable => "topology_unavailable",
    StageConnectionEstablished => "stage_connection_established",
    StageConnectionLost => "stage_connection_lost",
    StageConnectionRecovered => "stage_connection_recovered",
});

event_family!(SessionEventKind {
    SessionRequested => "session_requested",
    SessionCreated => "session_created",
    SessionActive => "session_active",
    SessionIdle => "session_idle",
    SessionReusable => "session_reusable",
    SessionReset => "session_reset",
    SessionTrimmed => "session_trimmed",
    SessionRestoredFromPrefixCache => "session_restored_from_prefix_cache",
    SessionRestoredFromCheckpoint => "session_restored_from_checkpoint",
    SessionDraining => "session_draining",
    SessionClosed => "session_closed",
    SessionFailed => "session_failed",
    SessionAbandoned => "session_abandoned",
    SessionReclaimed => "session_reclaimed",
});
