use std::time::Duration;

const MAX_ID_BYTES: usize = 256;
const MAX_STATE_BYTES: usize = 64;
const MAX_REASON_BYTES: usize = 128;
const MAX_SUMMARY_KEY_BYTES: usize = 64;
const MAX_HUMAN_SUMMARY_BYTES: usize = 512;
const MAX_NUMERIC_SUMMARIES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundedTextError {
    Empty,
    TooLong,
}

fn bounded_text(value: &str, max_bytes: usize) -> Result<String, BoundedTextError> {
    if value.is_empty() {
        return Err(BoundedTextError::Empty);
    }
    if value.len() > max_bytes {
        return Err(BoundedTextError::TooLong);
    }
    Ok(value.to_owned())
}

macro_rules! string_value {
    ($name:ident, $max:expr) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: &str) -> Result<Self, BoundedTextError> {
                bounded_text(value, $max).map(Self)
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

string_value!(LogicalModelId, MAX_ID_BYTES);
string_value!(TopologyId, MAX_ID_BYTES);
string_value!(StageId, MAX_ID_BYTES);
string_value!(SessionId, MAX_ID_BYTES);
string_value!(RequestId, MAX_ID_BYTES);
string_value!(DeviceId, MAX_ID_BYTES);
string_value!(StateName, MAX_STATE_BYTES);
string_value!(UnknownReasonCode, MAX_REASON_BYTES);
string_value!(NumericSummaryKey, MAX_SUMMARY_KEY_BYTES);
string_value!(HumanSummary, MAX_HUMAN_SUMMARY_BYTES);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageIdentity {
    pub id: StageId,
    pub index: u32,
}

impl StageIdentity {
    #[must_use]
    pub const fn new(id: StageId, index: u32) -> Self {
        Self { id, index }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ScopeIdentities {
    pub model_id: Option<LogicalModelId>,
    pub topology_id: Option<TopologyId>,
    pub stage: Option<StageIdentity>,
    pub session_id: Option<SessionId>,
    pub request_id: Option<RequestId>,
    pub device_id: Option<DeviceId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateTransition {
    pub previous: Option<StateName>,
    pub current: StateName,
}

impl StateTransition {
    #[must_use]
    pub const fn new(previous: Option<StateName>, current: StateName) -> Self {
        Self { previous, current }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressUnit {
    None,
    Bytes,
    Items,
    Tensors,
    Steps,
    Tokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub current: u64,
    pub total: Option<u64>,
    pub unit: ProgressUnit,
}

impl Progress {
    #[must_use]
    pub const fn new(current: u64, total: Option<u64>, unit: ProgressUnit) -> Self {
        Self {
            current,
            total,
            unit,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Failure,
    Rejected,
    Cancelled,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasonCode {
    InvalidConfiguration,
    UnsupportedCapability,
    MissingArtifact,
    ArtifactIoFailure,
    ModelFormatOrLoadFailure,
    BackendInitializationFailure,
    DeviceUnavailable,
    ResourceAllocationFailure,
    OutOfMemory,
    ContextExhausted,
    StageUnavailable,
    Timeout,
    Cancellation,
    ProcessCrash,
    IncompatibleAbiOrFeatureSet,
    InternalRuntimeFailure,
    UnknownFailure,
    TerminalNotDelivered,
    ReservationExhausted,
    Unknown(UnknownReasonCode),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericValue {
    Unsigned(u64),
    Signed(i64),
    Floating(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NumericSummary {
    pub key: NumericSummaryKey,
    pub value: NumericValue,
}

impl NumericSummary {
    #[must_use]
    pub const fn new(key: NumericSummaryKey, value: NumericValue) -> Self {
        Self { key, value }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TooManyNumericSummaries;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BoundedNumericSummaries(Vec<NumericSummary>);

impl BoundedNumericSummaries {
    pub fn new(values: Vec<NumericSummary>) -> Result<Self, TooManyNumericSummaries> {
        if values.len() > MAX_NUMERIC_SUMMARIES {
            return Err(TooManyNumericSummaries);
        }
        Ok(Self(values))
    }

    #[must_use]
    pub fn as_slice(&self) -> &[NumericSummary] {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FactData {
    pub scope: ScopeIdentities,
    pub state: Option<StateTransition>,
    pub progress: Option<Progress>,
    pub outcome: Option<Outcome>,
    pub reason: Option<ReasonCode>,
    pub duration: Option<Duration>,
    pub numeric_summaries: BoundedNumericSummaries,
    pub summary: Option<HumanSummary>,
}
