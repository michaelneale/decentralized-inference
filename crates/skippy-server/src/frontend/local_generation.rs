mod cache_deadline;
mod decode_step;
mod linear_decode;
mod native_mtp_decode;
mod receipt_lifecycle;
mod resident_prefill;
#[cfg(test)]
mod tests;
mod token_generation;
#[cfg(test)]
pub(in crate::frontend) use token_generation::resident_capacity_admission_error;

#[cfg(test)]
pub(super) use token_generation::{
    linear_proposal_allowed, native_mtp_dispatch_counts_for_test, post_decode_checkpoint_tokens,
};

use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation_receipt::{
    GenerationReceiptObservation, LocalGenerationReceiptDelivery,
};
use crate::frontend::util::openai_backend_error;
use openai_frontend::OpenAiResult;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

pub(super) struct LocalGenerationReceiptFinalization<'a> {
    pub(super) session_label: &'a str,
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) agent_session_id: Option<&'a str>,
    pub(super) prompt_token_ids: Arc<[i32]>,
    pub(super) observation: Option<GenerationReceiptObservation>,
    pub(super) cancelled: bool,
    pub(super) model_generation_elapsed: Option<Duration>,
    pub(super) frontend_request_id: Option<[u8; 16]>,
}

impl StageOpenAiBackend {
    pub(super) fn finalize_generation_receipt(
        &self,
        mut finalization: LocalGenerationReceiptFinalization<'_>,
        generation_succeeded: bool,
    ) -> OpenAiResult<()> {
        if let Some(observation) = finalization.observation.as_mut() {
            if finalization.cancelled {
                observation.mark_cancelled();
            }
            if let Some(elapsed) = finalization.model_generation_elapsed {
                observation.set_model_generation_elapsed(elapsed);
            }
        }
        let Some(config) = self.generation_receipt.as_ref() else {
            return Ok(());
        };
        if !generation_succeeded {
            config.abort(crate::frontend::GenerationAbort {
                request_id: finalization.request_id,
                session_id: finalization.session_id,
            });
            return Ok(());
        }
        if finalization
            .observation
            .as_ref()
            .is_some_and(|observation| !observation.is_recording_enabled())
        {
            config.abort(crate::frontend::GenerationAbort {
                request_id: finalization.request_id,
                session_id: finalization.session_id,
            });
            return Ok(());
        }
        match finalization.observation {
            Some(observation) => {
                self.deliver_local_generation_receipt(LocalGenerationReceiptDelivery {
                    config,
                    session_label: finalization.session_label,
                    request_id: finalization.request_id,
                    session_id: finalization.session_id,
                    agent_session_id: finalization.agent_session_id,
                    prompt_token_ids: finalization.prompt_token_ids,
                    observation,
                    frontend_request_id: finalization.frontend_request_id,
                })
            }
            None => Ok(()),
        }
    }

    pub(super) fn cleanup_local_generation_session(
        &self,
        session_id: &str,
        ids: &crate::frontend::generation::OpenAiGenerationIds,
    ) {
        let scheduler_session_id = session_id.to_string();
        if let Ok(outcome) =
            self.iteration_scheduler
                .execute_runtime_timed("local-session-drop", move |runtime| {
                    runtime
                        .drop_session_timed(&scheduler_session_id)
                        .map_err(openai_backend_error)
                })
        {
            let drop_stats = outcome.value;
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(outcome.runtime_lock_wait_ms),
            );
            attrs.insert(
                "llama_stage.session_reset_ms".to_string(),
                json!(drop_stats.reset_ms),
            );
            attrs.insert(
                "llama_stage.session_reset".to_string(),
                json!(drop_stats.reset_session),
            );
            attrs.insert(
                "llama_stage.lane_discarded".to_string(),
                json!(drop_stats.lane_discarded),
            );
            if let Some(reason) = drop_stats.lane_discard_reason.as_deref() {
                attrs.insert("llama_stage.lane_discard_reason".to_string(), json!(reason));
            }
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_after",
                &drop_stats.stats_after,
            );
            self.telemetry
                .emit_debug("stage.openai_session_stop", attrs);
        }
    }
}

pub(super) fn prompt_fits_single_prefill_sample(
    prompt_token_count: usize,
    session_batch_size: usize,
) -> bool {
    prompt_token_count > 1 && prompt_token_count <= session_batch_size
}
