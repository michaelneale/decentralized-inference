use std::time::Duration;

use crate::{
    EventId, NativeSequenceObservation, NativeSourceEnvelope, OperationScope, RuntimeFact,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeEventSchemaVersion(pub u16);

impl RuntimeEventSchemaVersion {
    pub const CURRENT: Self = Self(1);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProducerSource {
    Native,
    Rust,
    Reconciled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Fatal,
}

/// Producer metadata carried with a family fact before the fact is reduced.
///
/// This is deliberately separate from [`RuntimeEventEnvelope`]. The family
/// fact is the value that crosses the bounded ingress lanes, while an
/// envelope is assembled later by the stream projection. Keeping the source
/// metadata here means the live path can preserve it without changing the
/// existing `RuntimeFact` variants or `try_submit` API.
#[derive(Debug, Clone, PartialEq)]
pub struct FactMetadata {
    pub producer: ProducerSource,
    pub severity: Severity,
    pub wall_clock_unix_ns: Option<u64>,
    pub process_monotonic_time: Option<Duration>,
    pub native_source: Option<NativeSourceEnvelope>,
    /// Optional evidence computed by the native adapter before any host
    /// coalescing or queue drop. A sequence jump inferred from reduced facts
    /// is not reliable because a latest-value lane may have discarded the
    /// intervening observation.
    pub native_sequence: Option<NativeSequenceObservation>,
}

impl FactMetadata {
    /// Metadata for a fact produced by Rust when no clock or native source
    /// information is available at the producer boundary.
    #[must_use]
    pub const fn rust_defaults() -> Self {
        Self {
            producer: ProducerSource::Rust,
            severity: Severity::Info,
            wall_clock_unix_ns: None,
            process_monotonic_time: None,
            native_source: None,
            native_sequence: None,
        }
    }

    #[must_use]
    pub fn native_source(
        severity: Severity,
        source: NativeSourceEnvelope,
        sequence: Option<NativeSequenceObservation>,
    ) -> Self {
        Self {
            producer: ProducerSource::Native,
            severity,
            wall_clock_unix_ns: None,
            process_monotonic_time: None,
            native_source: Some(source),
            native_sequence: sequence,
        }
    }

    #[must_use]
    pub fn with_process_times(
        mut self,
        wall_clock_unix_ns: Option<u64>,
        process_monotonic_time: Option<Duration>,
    ) -> Self {
        self.wall_clock_unix_ns = wall_clock_unix_ns;
        self.process_monotonic_time = process_monotonic_time;
        self
    }
}

impl Default for FactMetadata {
    fn default() -> Self {
        Self::rust_defaults()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeEventEnvelope {
    pub schema_version: RuntimeEventSchemaVersion,
    pub event_id: EventId,
    pub operation: OperationScope,
    pub producer: ProducerSource,
    pub severity: Severity,
    pub wall_clock_unix_ns: u64,
    pub process_monotonic_time: Duration,
    pub native_source: Option<NativeSourceEnvelope>,
    pub fact: RuntimeFact,
}

impl RuntimeEventEnvelope {
    #[must_use]
    pub fn into_parts(self) -> (RuntimeFact, Option<NativeSourceEnvelope>) {
        (self.fact, self.native_source)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarrierLocation {
    RuntimeEventEnvelope,
    NativeSourceEnvelope,
    FamilyFact,
}

impl CarrierLocation {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RuntimeEventEnvelope => "RuntimeEventEnvelope",
            Self::NativeSourceEnvelope => "NativeSourceEnvelope",
            Self::FamilyFact => "FamilyFact",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarrierKind {
    SchemaVersion,
    CategoryKind,
    Producer,
    Severity,
    WallClockTime,
    ProcessMonotonicTime,
    NativeMonotonicTime,
    NativeSequence,
    ProcessInstanceId,
    EventSequence,
    RootOperationId,
    ChildOperationId,
    EmitterIdentity,
    ScopeIdentities,
    PreviousCurrentState,
    Progress,
    OutcomeReason,
    Duration,
    BoundedSummaries,
}

impl CarrierKind {
    pub const ALL: &'static [Self] = &[
        Self::SchemaVersion,
        Self::CategoryKind,
        Self::Producer,
        Self::Severity,
        Self::WallClockTime,
        Self::ProcessMonotonicTime,
        Self::NativeMonotonicTime,
        Self::NativeSequence,
        Self::ProcessInstanceId,
        Self::EventSequence,
        Self::RootOperationId,
        Self::ChildOperationId,
        Self::EmitterIdentity,
        Self::ScopeIdentities,
        Self::PreviousCurrentState,
        Self::Progress,
        Self::OutcomeReason,
        Self::Duration,
        Self::BoundedSummaries,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SchemaVersion => "schema_version",
            Self::CategoryKind => "category_kind",
            Self::Producer => "producer",
            Self::Severity => "severity",
            Self::WallClockTime => "wall_clock_time",
            Self::ProcessMonotonicTime => "process_monotonic_time",
            Self::NativeMonotonicTime => "native_monotonic_time",
            Self::NativeSequence => "native_sequence",
            Self::ProcessInstanceId => "process_instance_id",
            Self::EventSequence => "event_sequence",
            Self::RootOperationId => "root_operation_id",
            Self::ChildOperationId => "child_operation_id",
            Self::EmitterIdentity => "emitter_identity",
            Self::ScopeIdentities => "scope_identities",
            Self::PreviousCurrentState => "previous_current_state",
            Self::Progress => "progress",
            Self::OutcomeReason => "outcome_reason",
            Self::Duration => "duration",
            Self::BoundedSummaries => "bounded_summaries",
        }
    }

    #[must_use]
    pub const fn location(self) -> CarrierLocation {
        match self {
            Self::SchemaVersion
            | Self::CategoryKind
            | Self::Producer
            | Self::Severity
            | Self::WallClockTime
            | Self::ProcessMonotonicTime
            | Self::ProcessInstanceId
            | Self::EventSequence
            | Self::RootOperationId
            | Self::ChildOperationId => CarrierLocation::RuntimeEventEnvelope,
            Self::NativeMonotonicTime | Self::NativeSequence | Self::EmitterIdentity => {
                CarrierLocation::NativeSourceEnvelope
            }
            Self::ScopeIdentities
            | Self::PreviousCurrentState
            | Self::Progress
            | Self::OutcomeReason
            | Self::Duration
            | Self::BoundedSummaries => CarrierLocation::FamilyFact,
        }
    }
}
