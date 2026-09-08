//! Generation-receipt begin/commit bookkeeping for one local token-generation
//! call.
//!
//! Finalization (`finalize_generation_receipt`) lives in the parent
//! `local_generation` module; this module owns the two lifecycle hooks that
//! fire during the decode loop itself: [`begin_generation_receipt`] marks a
//! generation as started once, before decode begins, and
//! [`commit_local_generation_token`] records each canonical token as it is
//! emitted.

use std::sync::Arc;

use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation_receipt::{GenerationCommit, GenerationStart};

/// Marks a generation as started with the receipt sink, when a receipt
/// config is configured. Returns the prompt token ids captured for the
/// receipt (`None` when no receipt config is present); the caller carries
/// this through to `finalize_generation_receipt`.
pub(super) fn begin_generation_receipt(
    config: Option<&crate::frontend::GenerationReceiptConfig>,
    ids: &OpenAiGenerationIds,
    prompt_token_ids: &[i32],
) -> Option<Arc<[i32]>> {
    let receipt_prompt_token_ids = config.map(|_| Arc::<[i32]>::from(prompt_token_ids));
    if let Some(config) = config {
        config.begin(GenerationStart {
            request_id: ids.request_id,
            session_id: ids.session_id,
            agent_session_id: ids.agent_session_id.clone(),
            prompt_token_ids: Arc::clone(
                receipt_prompt_token_ids
                    .as_ref()
                    .expect("receipt prompt exists when receipt config exists"),
            ),
            frontend_request_id: ids.frontend_request_id,
        });
    }
    receipt_prompt_token_ids
}

/// Records one canonical generated token against the receipt sink, when a
/// receipt config is configured.
pub(super) fn commit_local_generation_token(
    config: Option<&crate::frontend::GenerationReceiptConfig>,
    request_id: u64,
    session_id: u64,
    generated_token_count: &mut usize,
    token_id: i32,
) {
    let Some(config) = config else {
        return;
    };
    *generated_token_count = generated_token_count.saturating_add(1);
    config.committed(GenerationCommit {
        request_id,
        session_id,
        generated_token_count: *generated_token_count,
        token_ids: vec![token_id].into_boxed_slice(),
    });
}
