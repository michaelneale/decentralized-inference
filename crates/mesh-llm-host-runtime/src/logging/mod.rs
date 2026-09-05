//! Operator logging privacy policy and redaction infrastructure.
//! Centralizes all credential/token/header/query/body/stack sanitization before any log event is serialized or persisted.
//!
//! Note: `#[allow(dead_code)]` and `#![allow(unused_imports)]` are intentional — this module defines the full surface
//! that will be wired into the runtime logging pipeline in subsequent tasks (Todo 6+).

#![allow(dead_code, unused_imports)]

mod bus;
mod cleanup;
pub mod foundation;
mod lifecycle;
mod limits;
mod management_lifecycle;
mod metrics;
mod openai_lifecycle;
mod operator_audit;
mod output_projection;
mod persistence;
pub mod policy;
mod raw_mesh_lifecycle;
mod registry;
mod request_metadata;
mod runtime_state;
mod sequences;
mod service;
mod service_errors;
#[cfg(test)]
mod service_tests;
mod webhook_delivery;
mod webhook_scheduler;
mod writer;

pub(crate) use bus::ReplayUpdate;
pub use bus::{
    AuditReplayRecord, AuditReplayWindow, BusEntry, PushOutcome, ReplayBus, ReplayCursor,
    ReplayRecord, ReplayWindow,
};
pub use foundation::LoggingFoundation;
pub use lifecycle::{DuplicateTerminalError, LifecycleGuard, TerminalOutcome};
pub use limits::{DynamicLoggingLimits, LoggingDynamicLimits};
pub(crate) use management_lifecycle::ManagementRequestLifecycle;
pub(crate) use metrics::{
    LoggingArtifactCaptureStatus, LoggingCleanupOutcome, LoggingMetric, LoggingMetrics,
    LoggingMetricsSink, LoggingTerminalOutcome, LoggingWebhookAttemptState,
    LoggingWebhookDeliveryOutcome,
};
pub(crate) use openai_lifecycle::{
    MAX_TRACKED_REQUESTS, OpenAiArtifactCapture, OpenAiLifecycleAttachment, OpenAiRouteAttempt,
    OpenAiRouteObserver, OpenAiStreamArtifactCapture,
};
pub use persistence::LogStoreSink;
pub(crate) use raw_mesh_lifecycle::{
    ProxyAttemptFinish, RawMeshLifecycleOwners, RawMeshProxyAttempt, RawMeshRemoteAttributionLease,
    RawMeshRemoteSuppressionLease, RawMeshRequestLifecycle,
};
pub use registry::{ActiveRequestSnapshot, RegistryConfig, RequestRegistry, RequestSummaryEntry};
pub(crate) use registry::{RequestSummaryEventSnapshots, RequestSummarySnapshot};
pub(crate) use request_metadata::{CallerPathType, RequestSummaryMetadata};
#[cfg(test)]
pub use runtime_state::ArtifactCaptureRequest;
pub(crate) use runtime_state::LoggingQueryFacade;
pub(crate) use runtime_state::{LoggingMetadataState, LoggingRuntimeStatus};
pub use runtime_state::{LoggingRuntimeApplyError, LoggingRuntimeHealth, LoggingRuntimeState};
pub use sequences::SequenceGenerators;
pub(crate) use service::ArtifactUnavailableReason;
pub use service::{
    ArtifactCaptureEntry, ArtifactPersistenceStatus, Clock, LoggingService,
    OperationalAuditContext, OperationalAuditPathType, OperationalAuditRecord,
    OperationalAuditSeverity, OperationalAuditSubjectKind, PersistSink, ServiceConfig, SystemClock,
};
pub(crate) use webhook_delivery::{
    RandomWebhookJitter, ReqwestWebhookTransport, SystemWebhookWorkerClock, WebhookDeliveryWorker,
    WebhookJitter, WebhookTransport, WebhookTransportError, WebhookWorkerClock,
    WebhookWorkerConfigError, WebhookWorkerError, WebhookWorkerOutcome,
};
pub(crate) use webhook_scheduler::WebhookDeliveryScheduler;
pub use writer::{FailOpenWriter, RecursionGuard};
