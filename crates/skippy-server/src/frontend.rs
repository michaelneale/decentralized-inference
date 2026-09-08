use skippy_protocol::binary::StageSamplingConfig as WireSamplingConfig;

mod admission;
mod backend;
mod decode_scheduler;
mod embedded_execution;
mod embedded_generation;
mod generation;
mod generation_flow;
mod generation_receipt;
mod guardrails;
pub(crate) mod iteration_scheduler;
mod linear_proposal;
mod local_generation;
mod native_mtp;
mod prefill;
mod prefix_cache;
mod prompting;
mod request;
mod sampling_cache_key;
mod speculative;
mod tool_emulation;
mod util;

#[cfg(test)]
use prompting::parse_emulated_chat_output;
mod wire_messages;

use self::{
    decode_scheduler::*, native_mtp::*, request::*, speculative::*, util::*, wire_messages::*,
};

pub use self::admission::DECODE_BATCH_HEADROOM_TOKENS;
use self::generation::*;
pub use self::generation::{
    CONTEXT_BUDGET_MAX_TOKENS, DEFAULT_EMBEDDED_MAX_TOKENS,
    DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS, EmbeddedOpenAiArgs, EmbeddedOpenAiBackend,
    EmbeddedOpenAiRequestDefaults, EmbeddedOpenAiRouter, EmbeddedReasoningBudget,
    EmbeddedReasoningEnabled, EmbeddedReasoningFormat, embedded_openai_backend,
    embedded_openai_router, serve_embedded_openai, serve_embedded_openai_with_shutdown,
    serve_openai,
};
pub(crate) use self::generation::{
    default_generation_queue_capacity, resolve_adaptive_generation_min_concurrency,
    serve_embedded_openai_with_scheduler,
};
pub use self::generation_receipt::{
    CompositeGenerationLifecycleIngress, GenerationAbort, GenerationCommit, GenerationCompletion,
    GenerationLifecycleConfig, GenerationLifecycleIngress, GenerationLifecycleObservation,
    GenerationReceipt, GenerationReceiptConfig, GenerationReceiptSink, GenerationStart,
    GenerationStateDigest, GenerationTermination, generation_token_id_digest,
};
pub use self::guardrails::{
    OpenAiGuardrailsConfig, OpenAiGuardrailsStatus, OpenAiGuardrailsTarget,
};
pub use self::linear_proposal::{
    LinearProposal, LinearProposalDiscardReason, LinearProposalDisposition, LinearProposalIngress,
    LinearProposalIngressConfig, LinearProposalQuery, LinearProposalReceipt,
    LinearProposalSourceOutcome, LinearProposalSourceResponse, LinearProposalSourceTelemetry,
    OpaqueProposalDecisionId,
};
pub use self::speculative::{
    NativeMtpProposalConfig, NgramExtensionConfig, NgramProposalConfig, NgramProposerKind,
    SpeculativeDecodeConfig, VerifyWindowConfig,
};

#[cfg(test)]
mod tests;
