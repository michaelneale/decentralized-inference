use crate::{
    NativeDetail, NativeEmitter, NativeEventCategory, NativeEventKind, NativeFailureCode,
    NativeProgressUnit, NativeSourceEnvelope, NativeStatus,
};

// `RawNativeEventV1Fixture`'s fields (declaration order) are a deliberate
// byte-for-byte mirror of `skippy_ffi::abi::SkippyRuntimeEventV1`'s fields.
// That cross-crate drift check does NOT live here as a test, because a
// published `mesh-llm-runtime-event-contracts` tarball contains only this
// crate's own package root — an `include_str!` reaching into the sibling
// `skippy-ffi` crate's source would fail to compile for any downstream
// consumer of the published crate (enforced by
// `tools/xtask/src/publish_consistency.rs`'s literal-include package-root
// check). The equivalent assertion instead runs from the repo's dev-only
// `xtask` binary, which is free to read across crate boundaries:
// `tools/xtask/src/repo_consistency.rs::check_runtime_event_abi_mirror`,
// invoked by `cargo run -p xtask -- repo-consistency ci-crate-lists` (`just
// ci-crate-lists`). Keep this struct's field names and order exactly in
// sync with `SkippyRuntimeEventV1` — the xtask check parses both structs as
// text and fails loudly if they diverge.
struct RawNativeEventV1Fixture {
    abi_version: u32,
    struct_size: u32,
    category: u32,
    kind: u32,
    emitter: u32,
    reserved0: u32,
    sequence: u64,
    timestamp_mono_ns: u64,
    model_id: u64,
    stage_id: u64,
    session_id: u64,
    progress_current: u64,
    progress_total: u64,
    progress_unit: u32,
    failure_code: u32,
    status: i32,
    reserved1: u32,
    detail_ptr: Vec<u8>,
    detail_len: u64,
    numeric_summary_0: Option<u64>,
    numeric_summary_1: Option<u64>,
    numeric_summary_2: Option<u64>,
    numeric_summary_3: Option<u64>,
}

const NATIVE_EVENT_V1_PREFIX_SIZE: u32 = 112;
const NATIVE_EVENT_V1_FULL_SIZE: u32 = 144;

impl RawNativeEventV1Fixture {
    fn to_owned_envelope(&self) -> NativeSourceEnvelope {
        assert_eq!(
            self.detail_len,
            u64::try_from(self.detail_ptr.len()).expect("fixture detail length should fit u64")
        );
        NativeSourceEnvelope {
            abi_version: self.abi_version,
            struct_size: self.struct_size,
            category: NativeEventCategory::new(self.category),
            kind: NativeEventKind::new(self.kind),
            emitter: NativeEmitter::new(self.emitter),
            reserved0: self.reserved0,
            sequence: self.sequence,
            timestamp_mono_ns: self.timestamp_mono_ns,
            model_id: self.model_id,
            stage_id: self.stage_id,
            session_id: self.session_id,
            progress_current: self.progress_current,
            progress_total: self.progress_total,
            progress_unit: NativeProgressUnit::new(self.progress_unit),
            failure_code: NativeFailureCode::new(self.failure_code),
            status: NativeStatus::new(self.status),
            reserved1: self.reserved1,
            detail: NativeDetail::bytes(&self.detail_ptr),
            numeric_summary_0: self.numeric_summary_0,
            numeric_summary_1: self.numeric_summary_1,
            numeric_summary_2: self.numeric_summary_2,
            numeric_summary_3: self.numeric_summary_3,
        }
    }

    fn from_owned(source: &NativeSourceEnvelope) -> Self {
        Self {
            abi_version: source.abi_version,
            struct_size: source.struct_size,
            category: source.category.raw(),
            kind: source.kind.raw(),
            emitter: source.emitter.raw(),
            reserved0: source.reserved0,
            sequence: source.sequence,
            timestamp_mono_ns: source.timestamp_mono_ns,
            model_id: source.model_id,
            stage_id: source.stage_id,
            session_id: source.session_id,
            progress_current: source.progress_current,
            progress_total: source.progress_total,
            progress_unit: source.progress_unit.raw(),
            failure_code: source.failure_code.raw(),
            status: source.status.raw(),
            reserved1: source.reserved1,
            detail_ptr: source.detail.as_bytes().to_vec(),
            detail_len: u64::try_from(source.detail.as_bytes().len())
                .expect("owned detail length should fit u64"),
            numeric_summary_0: source.numeric_summary_0,
            numeric_summary_1: source.numeric_summary_1,
            numeric_summary_2: source.numeric_summary_2,
            numeric_summary_3: source.numeric_summary_3,
        }
    }
}

pub fn unknown_native_source() -> NativeSourceEnvelope {
    raw_native_event_fixture(Some((50, 51, 52, 53))).to_owned_envelope()
}

/// `extension` is `None` to model a short-prefix native event (struct_size
/// covers only the original 19 fields, so the appended fields are absent —
/// `Option::None`, not zero) or `Some((..))` to model a full 23-field event.
fn raw_native_event_fixture(extension: Option<(u64, u64, u64, u64)>) -> RawNativeEventV1Fixture {
    RawNativeEventV1Fixture {
        abi_version: 17,
        struct_size: if extension.is_some() {
            NATIVE_EVENT_V1_FULL_SIZE
        } else {
            NATIVE_EVENT_V1_PREFIX_SIZE
        },
        category: u32::MAX,
        kind: u32::MAX - 1,
        emitter: u32::MAX - 2,
        reserved0: 41,
        sequence: 42,
        timestamp_mono_ns: 43,
        model_id: 44,
        stage_id: 45,
        session_id: 46,
        progress_current: 47,
        progress_total: 48,
        progress_unit: u32::MAX - 3,
        failure_code: u32::MAX - 4,
        status: i32::MIN,
        reserved1: 49,
        detail_ptr: b"\0future\xffdetail".to_vec(),
        detail_len: 14,
        numeric_summary_0: extension.map(|(a, ..)| a),
        numeric_summary_1: extension.map(|(_, b, ..)| b),
        numeric_summary_2: extension.map(|(_, _, c, _)| c),
        numeric_summary_3: extension.map(|(.., d)| d),
    }
}

/// Every field except the appended numeric summaries: the original 19-field
/// prefix shared by every native ABI version. Split out of the round-trip
/// test itself purely to keep that test's own cognitive complexity under the
/// configured Clippy limit; each assertion's exact message is unchanged.
fn assert_prefix_fields_preserved(recovered: &RawNativeEventV1Fixture, expected_struct_size: u32) {
    assert_eq!(recovered.abi_version, 17, "abi_version was not preserved");
    assert_eq!(
        recovered.struct_size, expected_struct_size,
        "struct_size was not preserved"
    );
    assert_eq!(recovered.category, u32::MAX, "category was not preserved");
    assert_eq!(recovered.kind, u32::MAX - 1, "kind was not preserved");
    assert_eq!(recovered.emitter, u32::MAX - 2, "emitter was not preserved");
    assert_eq!(recovered.reserved0, 41, "reserved0 was not preserved");
    assert_eq!(recovered.sequence, 42, "sequence was not preserved");
    assert_eq!(
        recovered.timestamp_mono_ns, 43,
        "timestamp_mono_ns was not preserved"
    );
    assert_eq!(recovered.model_id, 44, "model_id was not preserved");
    assert_eq!(recovered.stage_id, 45, "stage_id was not preserved");
    assert_eq!(recovered.session_id, 46, "session_id was not preserved");
    assert_eq!(
        recovered.progress_current, 47,
        "progress_current was not preserved"
    );
    assert_eq!(
        recovered.progress_total, 48,
        "progress_total was not preserved"
    );
    assert_eq!(
        recovered.progress_unit,
        u32::MAX - 3,
        "progress_unit was not preserved"
    );
    assert_eq!(
        recovered.failure_code,
        u32::MAX - 4,
        "failure_code was not preserved"
    );
    assert_eq!(recovered.status, i32::MIN, "status was not preserved");
    assert_eq!(recovered.reserved1, 49, "reserved1 was not preserved");
    assert_eq!(
        recovered.detail_ptr, b"\0future\xffdetail",
        "owned detail bytes were not preserved"
    );
    assert_eq!(recovered.detail_len, 14, "detail_len was not reconstructed");
}

/// The four append-only numeric summary fields, checked separately from
/// `assert_prefix_fields_preserved` for the same cognitive-complexity reason.
fn assert_numeric_summary_fields_preserved(recovered: &RawNativeEventV1Fixture) {
    assert_eq!(
        recovered.numeric_summary_0,
        Some(50),
        "numeric_summary_0 was not preserved"
    );
    assert_eq!(
        recovered.numeric_summary_1,
        Some(51),
        "numeric_summary_1 was not preserved"
    );
    assert_eq!(
        recovered.numeric_summary_2,
        Some(52),
        "numeric_summary_2 was not preserved"
    );
    assert_eq!(
        recovered.numeric_summary_3,
        Some(53),
        "numeric_summary_3 was not preserved"
    );
}

#[test]
fn unknown_native_reason_and_carriers_round_trip_exactly() {
    // Given
    let mut raw = raw_native_event_fixture(Some((50, 51, 52, 53)));

    // When
    let source = raw.to_owned_envelope();
    raw.detail_ptr.fill(b'x');
    let recovered = RawNativeEventV1Fixture::from_owned(&source);

    // Then
    assert_prefix_fields_preserved(&recovered, NATIVE_EVENT_V1_FULL_SIZE);
    assert_numeric_summary_fields_preserved(&recovered);
}

#[test]
fn short_prefix_fixture_has_no_appended_numeric_summary_fields() {
    // Given: a fixture modeling a native event whose struct_size covered
    // only the original 19 fields (extension = None), not zero-valued
    // extension fields.
    let raw = raw_native_event_fixture(None);

    // When
    let source = raw.to_owned_envelope();

    // Then
    assert_eq!(source.struct_size, NATIVE_EVENT_V1_PREFIX_SIZE);
    assert_eq!(source.numeric_summary_0, None);
    assert_eq!(source.numeric_summary_1, None);
    assert_eq!(source.numeric_summary_2, None);
    assert_eq!(source.numeric_summary_3, None);
}
