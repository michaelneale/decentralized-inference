use skippy_cache::activation_page_id;
use skippy_protocol::{MessageBase, StageConfig};
use skippy_runtime::{ActivationFrame, RuntimeActivationLayout};

use super::{KvStageIntegration, ResidentActivationRecord, ResidentActivationRestore};

impl KvStageIntegration {
    pub fn restore_resident_activation(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
        activation_width: i32,
    ) -> Option<ResidentActivationRestore> {
        if !self.should_lookup() || token_ids.is_empty() {
            return None;
        }
        // Walk lookup candidates longest-prefix-first; the first validated
        // cache hit is the longest activation frame reachable from this prompt.
        for identity in self.activation_lookup_identities(config, base, token_start, token_ids) {
            let page_id = activation_page_id(&identity.page_id, activation_width);
            let Some(lookup) = self
                .activations
                .lock()
                .expect("resident activation cache lock poisoned")
                .lookup(&page_id)
            else {
                continue;
            };
            let identity_token_count = identity.identity.token_count;
            if lookup.token_count != identity_token_count
                || u64::from(lookup.frame.desc.token_count) != identity_token_count
                || lookup.frame.desc.payload_bytes != lookup.byte_size
                || lookup.frame.payload.len() as u64 != lookup.byte_size
            {
                continue;
            }
            return Some(ResidentActivationRestore {
                identity,
                page_id,
                token_count: lookup.token_count as usize,
                payload_bytes: lookup.byte_size as usize,
                entries: lookup.entries,
                frame: lookup.frame,
            });
        }
        None
    }

    pub fn record_resident_activation(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
        activation_width: i32,
        frame: &ActivationFrame,
    ) -> Vec<ResidentActivationRecord> {
        if !self.should_record() || token_ids.is_empty() {
            return Vec::new();
        }
        let token_count = token_ids.len() as u64;
        if token_count < self.checkpoint_policy.min_tokens || frame.payload.is_empty() {
            return Vec::new();
        }
        if u64::from(frame.desc.token_count) != token_count
            || frame.desc.payload_bytes != frame.payload.len() as u64
        {
            return Vec::new();
        }
        let identities = self.activation_record_identities(config, base, token_start, token_ids);
        let mut cache = self
            .activations
            .lock()
            .expect("resident activation cache lock poisoned");
        identities
            .into_iter()
            .filter_map(|identity| {
                let candidate_token_count = usize::try_from(identity.identity.token_count).ok()?;
                let candidate_frame = activation_prefix_frame(frame, candidate_token_count)?;
                let page_id = activation_page_id(&identity.page_id, activation_width);
                let payload_bytes = candidate_frame.payload.len();
                let stored = cache.record(
                    page_id.clone(),
                    identity.identity.token_count,
                    payload_bytes as u64,
                    candidate_frame,
                );
                let stats = cache.stats();
                Some(ResidentActivationRecord {
                    page_id,
                    token_count: candidate_token_count,
                    payload_bytes,
                    evicted_entries: stored.evicted_entries,
                    evicted_bytes: stored.evicted_bytes,
                    entries: stats.entries,
                    resident_bytes: stats.resident_bytes,
                })
            })
            .collect()
    }
}

fn activation_prefix_frame(
    frame: &ActivationFrame,
    candidate_token_count: usize,
) -> Option<ActivationFrame> {
    let frame_token_count = usize::try_from(frame.desc.token_count).ok()?;
    if candidate_token_count == frame_token_count {
        return Some(frame.clone());
    }
    if candidate_token_count == 0
        || candidate_token_count > frame_token_count
        || frame.desc.layout != RuntimeActivationLayout::TokenMajor
        || frame.desc.flags != 0
        || !frame.payload.len().is_multiple_of(frame_token_count)
    {
        return None;
    }
    let payload_bytes = frame
        .payload
        .len()
        .checked_div(frame_token_count)?
        .checked_mul(candidate_token_count)?;
    let mut prefix = frame.clone();
    prefix.desc.token_count = u32::try_from(candidate_token_count).ok()?;
    prefix.desc.payload_bytes = payload_bytes as u64;
    prefix.payload.truncate(payload_bytes);
    Some(prefix)
}

#[cfg(test)]
mod tests {
    use skippy_protocol::{
        LoadMode, SCHEMA_VERSION, StageConfig, StageKvCacheConfig, StageKvCacheMode,
        StageKvCachePayload,
    };
    use skippy_runtime::{
        ActivationDesc, ActivationFrame, RuntimeActivationDType, RuntimeActivationLayout,
    };

    use super::*;

    fn test_config() -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "hugging-quants/Llama-3.2-1B-Instruct-GGUF:Q4_K_M".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 4,
            ctx_size: 8192,
            lane_count: 2,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: Default::default(),
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: false,
            resident_tensor_names: Vec::new(),
            selected_device: None,
            kv_cache: Some(StageKvCacheConfig {
                mode: StageKvCacheMode::LookupRecord,
                payload: StageKvCachePayload::ResidentKv,
                max_entries: 8,
                max_bytes: 0,
                min_tokens: 256,
                shared_prefix_stride_tokens: 128,
                shared_prefix_record_limit: 2,
            }),
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
            ..StageConfig::default()
        }
    }

    fn test_base() -> MessageBase {
        MessageBase {
            schema_version: SCHEMA_VERSION,
            run_id: "run".to_string(),
            request_id: "request".to_string(),
            session_id: "session".to_string(),
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            topology_id: "topology".to_string(),
            model_id: Some("hugging-quants/Llama-3.2-1B-Instruct-GGUF:Q4_K_M".to_string()),
            tokenizer_id: None,
            chat_template_id: Some("template".to_string()),
            seq: Some(1),
        }
    }

    fn activation_frame(token_count: u32, payload_bytes: usize) -> ActivationFrame {
        ActivationFrame {
            desc: ActivationDesc {
                version: 1,
                dtype: RuntimeActivationDType::F32,
                layout: RuntimeActivationLayout::TokenMajor,
                producer_stage_index: 0,
                layer_start: 0,
                layer_end: 4,
                token_count,
                sequence_count: 1,
                payload_bytes: payload_bytes as u64,
                flags: 0,
            },
            payload: vec![7; payload_bytes],
        }
    }

    #[test]
    fn resident_activation_records_and_restores_exact_frame() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
            .unwrap()
            .expect("resident cache enabled");
        let tokens = (0..300).collect::<Vec<_>>();
        let frame = activation_frame(tokens.len() as u32, tokens.len() * 4);

        let record = kv.record_resident_activation(&config, &test_base(), 0, &tokens, 4096, &frame);

        assert_eq!(record.len(), 2);
        let restored = kv
            .restore_resident_activation(&config, &test_base(), 0, &tokens, 4096)
            .expect("activation frame should restore by exact identity");
        assert_eq!(restored.frame, frame);
    }

    #[test]
    fn resident_activation_rejects_mismatched_frame_token_count() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
            .unwrap()
            .expect("resident cache enabled");
        let tokens = (0..300).collect::<Vec<_>>();
        let frame = activation_frame(128, 64);

        let record = kv.record_resident_activation(&config, &test_base(), 0, &tokens, 4096, &frame);

        assert!(record.is_empty());
        assert!(
            kv.restore_resident_activation(&config, &test_base(), 0, &tokens, 4096)
                .is_none()
        );
    }

    #[test]
    fn resident_activation_records_identity_matched_shared_prefix_frame() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
            .unwrap()
            .expect("resident cache enabled");
        let recorded_tokens = (0..2214).collect::<Vec<_>>();
        let mut lookup_tokens = recorded_tokens.clone();
        lookup_tokens.extend(100_000..100_017);
        let frame = activation_frame(recorded_tokens.len() as u32, recorded_tokens.len() * 4);

        let record =
            kv.record_resident_activation(&config, &test_base(), 0, &recorded_tokens, 4096, &frame);

        assert_eq!(
            record
                .iter()
                .map(|record| record.token_count)
                .collect::<Vec<_>>(),
            vec![2214, 2176]
        );
        let restored = kv
            .restore_resident_activation(&config, &test_base(), 0, &lookup_tokens, 4096)
            .expect("grid-floor activation should restore via a lookup candidate");
        assert_eq!(restored.identity.identity.token_count, 2176);
        assert_eq!(restored.token_count, 2176);
        assert_eq!(restored.frame.desc.token_count, 2176);
        assert_eq!(restored.frame.payload.len(), 2176 * 4);
    }

    #[test]
    fn restore_hits_shared_prefix_activation_for_extended_prompt() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
            .unwrap()
            .expect("resident cache enabled");
        let recorded_tokens = (0..2214).collect::<Vec<i32>>();
        let mut extended_lookup_tokens = recorded_tokens.clone();
        extended_lookup_tokens.extend(100_000..100_017);
        let frame = activation_frame(recorded_tokens.len() as u32, recorded_tokens.len() * 4);

        let records =
            kv.record_resident_activation(&config, &test_base(), 0, &recorded_tokens, 4096, &frame);
        assert_eq!(records.len(), 2);

        let restored = kv
            .restore_resident_activation(&config, &test_base(), 0, &extended_lookup_tokens, 4096)
            .expect("extended prompt should restore the shared grid-floor activation");
        assert_eq!(restored.identity.identity.token_count, 2176);
        assert_eq!(restored.token_count, 2176);
        assert_eq!(restored.frame.desc.token_count, 2176);
        assert_eq!(restored.frame.payload.len(), 2176 * 4);
    }

    #[test]
    fn resident_activation_keys_include_activation_width() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
            .unwrap()
            .expect("resident cache enabled");
        let tokens = (0..300).collect::<Vec<_>>();
        let frame = activation_frame(tokens.len() as u32, tokens.len() * 4);

        let record = kv.record_resident_activation(&config, &test_base(), 0, &tokens, 4096, &frame);

        assert_eq!(record.len(), 2);
        assert!(
            kv.restore_resident_activation(&config, &test_base(), 0, &tokens, 8192)
                .is_none()
        );
    }

    #[test]
    fn resident_activation_rejects_ambiguous_cached_token_count() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
            .unwrap()
            .expect("resident cache enabled");
        let tokens = (0..300).collect::<Vec<_>>();
        let identity = kv.prefill_identity(&config, &test_base(), 0, &tokens);
        let page_id = activation_page_id(&identity.page_id, 4096);
        kv.activations
            .lock()
            .expect("resident activation cache lock poisoned")
            .record(page_id, 256, 64, activation_frame(256, 64));

        assert!(
            kv.restore_resident_activation(&config, &test_base(), 0, &tokens, 4096)
                .is_none()
        );
    }

    #[test]
    fn resident_activation_does_not_alias_flagged_frame_to_shorter_identity() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
            .unwrap()
            .expect("resident cache enabled");
        let tokens = (0..300).collect::<Vec<_>>();
        let mut frame = activation_frame(tokens.len() as u32, 600);
        frame.desc.flags = 1;

        let records =
            kv.record_resident_activation(&config, &test_base(), 0, &tokens, 4096, &frame);

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].token_count, tokens.len());
    }
}
