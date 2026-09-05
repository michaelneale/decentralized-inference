#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeEventCategory(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeEventKind(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NativeEmitter(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeProgressUnit(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeFailureCode(u32);

macro_rules! native_u32_value {
    ($name:ident) => {
        impl $name {
            #[must_use]
            pub const fn new(raw: u32) -> Self {
                Self(raw)
            }

            #[must_use]
            pub const fn raw(self) -> u32 {
                self.0
            }
        }
    };
}

native_u32_value!(NativeEventCategory);
native_u32_value!(NativeEventKind);
native_u32_value!(NativeEmitter);
native_u32_value!(NativeProgressUnit);
native_u32_value!(NativeFailureCode);

/// Which native callback sequence a fact belongs to. Process reporters use a
/// sequence shared by callbacks that can be assigned to different operation
/// scopes; model-open reporters use a sequence local to one operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativeSequenceDomain {
    Operation,
    Process,
}

/// Evidence computed at the native adapter boundary, before host lanes can
/// coalesce or discard an observation. The reducer records only `Gap`; it
/// never infers a gap from two surviving reduced facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeSequenceEvidence {
    Unchecked,
    First,
    Contiguous,
    Gap,
    DuplicateOrOutOfOrder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeSequenceObservation {
    pub domain: NativeSequenceDomain,
    pub emitter: NativeEmitter,
    pub sequence: u64,
    pub evidence: NativeSequenceEvidence,
}

impl NativeSequenceObservation {
    #[must_use]
    pub const fn new(
        domain: NativeSequenceDomain,
        emitter: NativeEmitter,
        sequence: u64,
        evidence: NativeSequenceEvidence,
    ) -> Self {
        Self {
            domain,
            emitter,
            sequence,
            evidence,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeStatus(i32);

impl NativeStatus {
    #[must_use]
    pub const fn new(raw: i32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> i32 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeDetail(Vec<u8>);

impl NativeDetail {
    #[must_use]
    pub fn bytes(value: &[u8]) -> Self {
        Self(value.to_vec())
    }

    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeSourceEnvelope {
    pub abi_version: u32,
    pub struct_size: u32,
    pub category: NativeEventCategory,
    pub kind: NativeEventKind,
    pub emitter: NativeEmitter,
    pub reserved0: u32,
    pub sequence: u64,
    pub timestamp_mono_ns: u64,
    pub model_id: u64,
    pub stage_id: u64,
    pub session_id: u64,
    pub progress_current: u64,
    pub progress_total: u64,
    pub progress_unit: NativeProgressUnit,
    pub failure_code: NativeFailureCode,
    pub status: NativeStatus,
    pub reserved1: u32,
    pub detail: NativeDetail,
    // Present only when the native `struct_size` covered that field's
    // offset in the raw ABI struct; `None` means the emitting native
    // runtime predates the append-only extension (or built without it),
    // not that the value is zero.
    pub numeric_summary_0: Option<u64>,
    pub numeric_summary_1: Option<u64>,
    pub numeric_summary_2: Option<u64>,
    pub numeric_summary_3: Option<u64>,
}
