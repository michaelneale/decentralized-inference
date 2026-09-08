//! Pure `RuntimeFact` construction for the generation-lifecycle adapter.
//!
//! Every function here is a pure projection: given inputs already flowing
//! through [`super::SkippyGenerationRuntimeEventAdapter`], build the fact the
//! observation backs. No reservation, engine, or I/O concerns live here --
//! see `mod.rs` for those. Kept deliberately privacy-first: nothing in this
//! module reads `generated_token_ids`, `prompt_token_ids`, or
//! `prompt_token_digest` into a fact.

use mesh_llm_runtime_event_contracts::{
    BoundedNumericSummaries, FactData, GenerationEventKind, GenerationFact, NumericSummary,
    NumericSummaryKey, NumericValue, Outcome, PrefillEventKind, PrefillFact, ReasonCode,
    RuntimeFact, SessionEventKind, SessionFact,
};
use skippy_server::frontend::{GenerationReceipt, GenerationTermination};

pub(super) fn empty_data() -> FactData {
    FactData::default()
}

pub(super) fn generation_fact(kind: GenerationEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::Generation(GenerationFact::with_data(kind, data))
}

pub(super) fn prefill_fact(kind: PrefillEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::Prefill(PrefillFact::with_data(kind, data))
}

pub(super) fn session_fact(kind: SessionEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::Session(SessionFact::with_data(kind, data))
}

pub(super) fn synthetic_generation_terminal() -> RuntimeFact {
    generation_fact(
        GenerationEventKind::GenerationFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..empty_data()
        },
    )
}

pub(super) fn synthetic_prefill_terminal() -> RuntimeFact {
    prefill_fact(
        PrefillEventKind::PrefillFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..empty_data()
        },
    )
}

fn token_count_summaries(pairs: &[(&str, usize)]) -> BoundedNumericSummaries {
    let summaries = pairs
        .iter()
        .filter_map(|(key, count)| {
            let key = NumericSummaryKey::new(key).ok()?;
            let value = NumericValue::Unsigned(u64::try_from(*count).unwrap_or(u64::MAX));
            Some(NumericSummary::new(key, value))
        })
        .collect();
    BoundedNumericSummaries::new(summaries).unwrap_or_default()
}

fn micros_summary(key: &str, micros: u64) -> BoundedNumericSummaries {
    NumericSummaryKey::new(key)
        .ok()
        .and_then(|key| {
            BoundedNumericSummaries::new(vec![NumericSummary::new(
                key,
                NumericValue::Unsigned(micros),
            )])
            .ok()
        })
        .unwrap_or_default()
}

/// Redacted, bounded projection of a [`GenerationReceipt`]: counts and
/// timing only. Never carries token IDs, digests, or prompt content.
struct RedactedReceiptSummary {
    generated_token_count: usize,
    prompt_token_count: usize,
}

fn redact_receipt(receipt: &GenerationReceipt) -> RedactedReceiptSummary {
    RedactedReceiptSummary {
        generated_token_count: receipt.generated_token_ids.len(),
        prompt_token_count: receipt.prompt_token_count,
    }
}

/// Maps a receipt or lifecycle-only completion's termination onto the
/// family's terminal kinds. `GenerationReceiptSink::record` is only reachable
/// on a successful local generation (`GenerationReceipt` is target-authoritative
/// success evidence); `Aborted` covers the failure/cancellation path separately,
/// while a lifecycle-only completion can carry cancellation directly. In both
/// cases `outcome` reflects the ACTUAL termination, not a blanket `Success`:
/// `GenerationReceiptSink::record` fires whenever local generation stops
/// producing a target-authoritative receipt, including a mid-stream
/// cancellation that still committed a partial suffix -- see
/// `GenerationTermination::Cancelled`'s doc ("The local generation loop
/// observed request cancellation"). A `GenerationCancelled` kind reporting
/// `Outcome::Success` would misclassify a cancelled request as successful
/// in every derived readiness/health projection; `Outcome::Cancelled`
/// matches the exact same disposition `abort_terminal_fact` already reports
/// for the `GenerationAbort` path, so the two cancellation paths agree.
pub(super) fn generation_completion_terminal_fact(
    termination: GenerationTermination,
    prompt_token_count: usize,
    generated_token_count: usize,
) -> RuntimeFact {
    let (kind, outcome) = match termination {
        GenerationTermination::Cancelled => {
            (GenerationEventKind::GenerationCancelled, Outcome::Cancelled)
        }
        _ => (GenerationEventKind::GenerationCompleted, Outcome::Success),
    };
    let data = FactData {
        outcome: Some(outcome),
        numeric_summaries: token_count_summaries(&[
            ("generated_token_count", generated_token_count),
            ("prompt_token_count", prompt_token_count),
        ]),
        ..empty_data()
    };
    generation_fact(kind, data)
}

pub(super) fn receipt_terminal_fact(receipt: &GenerationReceipt) -> RuntimeFact {
    let redacted = redact_receipt(receipt);
    generation_completion_terminal_fact(
        receipt.termination,
        redacted.prompt_token_count,
        redacted.generated_token_count,
    )
}

/// §8.10's `first token produced`, backed by the real existing signal this
/// adapter already observes: the FIRST [`skippy_server::frontend::
/// GenerationCommit`] for a generation IS the moment its first token
/// became available. StateTransition-class, `unreserved_ingress`-safe.
pub(super) fn first_token_produced_fact() -> RuntimeFact {
    generation_fact(GenerationEventKind::FirstTokenProduced, empty_data())
}

/// §8.10's `stop condition reached`, backed by the real existing
/// [`GenerationTermination::CallbackStop`] variant: "the token callback
/// requested a stop, including for an end-of-generation token" IS the
/// existing signal that a stop CONDITION (as opposed to the max-token
/// budget) ended generation. Co-emitted alongside `GenerationCompleted`,
/// never in place of it. StateTransition-class, `unreserved_ingress`-safe.
pub(super) fn stop_condition_reached_fact() -> RuntimeFact {
    generation_fact(GenerationEventKind::StopConditionReached, empty_data())
}

pub(super) fn abort_terminal_fact() -> RuntimeFact {
    generation_fact(
        GenerationEventKind::GenerationCancelled,
        FactData {
            outcome: Some(Outcome::Cancelled),
            reason: Some(ReasonCode::Cancellation),
            ..empty_data()
        },
    )
}

pub(super) fn progress_fact(generated_token_count: usize) -> RuntimeFact {
    generation_fact(
        GenerationEventKind::GenerationProgress,
        FactData {
            numeric_summaries: token_count_summaries(&[(
                "generated_token_count",
                generated_token_count,
            )]),
            ..empty_data()
        },
    )
}

/// §8's prefill family, backed by the real existing
/// `GenerationReceipt::request_to_first_token_us` field or its
/// lifecycle-only equivalent: the target-authoritative
/// backend-request-start-to-first-token latency IS the existing signal for
/// "prefill completed" (decode begins at first token). A generation that never
/// emitted a token (`None`) reports the same successful terminal without the
/// timing summary rather than inventing a failure this layer cannot actually
/// distinguish from "zero tokens requested". Explicit cancellation before a
/// first token remains a cancelled prefill terminal; once first-token timing
/// exists, prefill has already completed even if decode is later cancelled.
pub(super) fn prefill_terminal_fact(receipt: &GenerationReceipt) -> RuntimeFact {
    prefill_completion_terminal_fact(receipt.termination, receipt.request_to_first_token_us)
}

pub(super) fn prefill_completion_terminal_fact(
    termination: GenerationTermination,
    request_to_first_token_us: Option<u64>,
) -> RuntimeFact {
    if matches!(termination, GenerationTermination::Cancelled)
        && request_to_first_token_us.is_none()
    {
        return prefill_cancelled_fact();
    }
    let data = match request_to_first_token_us {
        Some(micros) => FactData {
            outcome: Some(Outcome::Success),
            numeric_summaries: micros_summary("request_to_first_token_us", micros),
            ..empty_data()
        },
        None => FactData {
            outcome: Some(Outcome::Success),
            ..empty_data()
        },
    };
    prefill_fact(PrefillEventKind::PrefillCompleted, data)
}

pub(super) fn prefill_cancelled_fact() -> RuntimeFact {
    prefill_fact(
        PrefillEventKind::PrefillCancelled,
        FactData {
            outcome: Some(Outcome::Cancelled),
            reason: Some(ReasonCode::Cancellation),
            ..empty_data()
        },
    )
}

/// §9's session family. `SessionActive`/`SessionIdle` are StateTransition-
/// class (never Terminal), so they are safe to submit through
/// `unreserved_ingress` -- no reservation lifecycle applies. Backed by the
/// same generation-lifecycle observation stream this adapter already
/// receives: a session becomes active exactly when a generation starts on
/// it and idle exactly when that generation's terminal resolves. This
/// adapter never emits `SessionClosed`/`SessionFailed` (Terminal-class):
/// Skippy's generation lifecycle has no signal for "this KV session will
/// never be reused again", only "no generation is active on it right now".
pub(super) fn session_active_fact() -> RuntimeFact {
    session_fact(SessionEventKind::SessionActive, empty_data())
}

pub(super) fn session_idle_fact() -> RuntimeFact {
    session_fact(SessionEventKind::SessionIdle, empty_data())
}
