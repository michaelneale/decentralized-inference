use skippy_cache::{
    NATIVE_KV_DTYPE, NATIVE_KV_RUNTIME_ABI_VERSION, prefix_identity_with_namespace,
    prefix_namespace_hash,
};
use skippy_protocol::{MessageBase, StageConfig};

use crate::kv_proto::{KvCodec, PageIdentity, PageLayout};

use super::{KvStageIntegration, PrefillKvIdentity};

impl KvStageIntegration {
    pub fn prefill_identity(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> PrefillKvIdentity {
        let prefix = prefix_identity_with_namespace(
            config,
            token_start,
            token_ids,
            base.chat_template_id.as_deref(),
        );
        let identity = PageIdentity {
            model_id: config.model_id.clone(),
            model_revision: "unknown".to_string(),
            runtime_abi_version: NATIVE_KV_RUNTIME_ABI_VERSION.to_string(),
            topology_id: config.topology_id.clone(),
            stage_id: config.stage_id.clone(),
            stage_index: config.stage_index,
            layer_start: config.layer_start,
            layer_end: config.layer_end,
            prefix_hash: prefix.prefix_hash.clone(),
            session_id: base.session_id.clone(),
            token_start,
            token_count: prefix.token_count,
            generation: 1,
            layout: PageLayout::LayerContiguous as i32,
            codec: KvCodec::Fp16 as i32,
            tokenizer_id: base
                .tokenizer_id
                .clone()
                .unwrap_or_else(|| config.model_id.clone()),
            chat_template_id: base.chat_template_id.clone().unwrap_or_default(),
            position_config_hash: format!("ctx:{}", config.ctx_size),
            kv_dtype: NATIVE_KV_DTYPE.to_string(),
        };
        PrefillKvIdentity {
            identity,
            page_id: prefix.page_id,
            namespace: prefix_namespace_hash(config, token_start, base.chat_template_id.as_deref()),
            token_ids: token_ids.to_vec(),
        }
    }

    pub fn lookup_identities(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Vec<PrefillKvIdentity> {
        if token_ids.is_empty() {
            Vec::new()
        } else {
            vec![self.prefill_identity(config, base, token_start, token_ids)]
        }
    }

    pub fn record_identities(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Vec<PrefillKvIdentity> {
        if token_ids.is_empty() {
            Vec::new()
        } else {
            vec![self.prefill_identity(config, base, token_start, token_ids)]
        }
    }

    pub(crate) fn activation_lookup_identities(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Vec<PrefillKvIdentity> {
        self.checkpoint_policy
            .candidate_token_counts(token_ids.len() as u64)
            .into_iter()
            .map(|token_count| {
                self.prefill_identity(
                    config,
                    base,
                    token_start,
                    &token_ids[..token_count as usize],
                )
            })
            .collect()
    }

    pub(crate) fn activation_record_identities(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Vec<PrefillKvIdentity> {
        self.checkpoint_policy
            .record_candidate_token_counts(token_ids.len() as u64)
            .into_iter()
            .map(|token_count| {
                self.prefill_identity(
                    config,
                    base,
                    token_start,
                    &token_ids[..token_count as usize],
                )
            })
            .collect()
    }

    /// Pick a recurrent/full-state checkpoint below the near-tail page.
    ///
    /// The near-tail grid page can still include a short request-specific tail.
    /// One additional stride below it leaves room for that tail while keeping
    /// suffix prefill bounded for a changed-tail request.
    pub(crate) fn exact_shared_checkpoint_identity(
        &self,
        config: &StageConfig,
        base: &MessageBase,
        token_start: u64,
        token_ids: &[i32],
    ) -> Option<PrefillKvIdentity> {
        let token_count = token_ids.len() as u64;
        self.exact_shared_checkpoint_token_count(token_count)
            .map(|checkpoint| {
                self.prefill_identity(config, base, token_start, &token_ids[..checkpoint as usize])
            })
    }

    pub(crate) fn exact_shared_checkpoint_token_count(&self, token_count: u64) -> Option<u64> {
        if token_count <= self.checkpoint_policy.min_tokens {
            return None;
        }
        let stride = self.checkpoint_policy.stride_tokens.max(1);
        let near_tail = token_count.saturating_sub(token_count % stride);
        let checkpoint = near_tail
            .saturating_sub(stride)
            .max(self.checkpoint_policy.min_tokens);
        (checkpoint > 0 && checkpoint < token_count).then_some(checkpoint)
    }

    pub(crate) fn exact_state_record_token_count_allowed(
        &self,
        prompt_token_count: u64,
        candidate_token_count: u64,
    ) -> bool {
        match self.exact_shared_checkpoint_token_count(prompt_token_count) {
            Some(checkpoint) => candidate_token_count == checkpoint,
            None => candidate_token_count == prompt_token_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::kv_integration::ExactStateExtra;
    use skippy_cache::ExactStatePayload;
    use skippy_protocol::{
        LoadMode, SCHEMA_VERSION, StageConfig, StageKvCacheConfig, StageKvCacheMode,
        StageKvCachePayload,
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
            ctx_size: 512,
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
                min_tokens: 64,
                shared_prefix_stride_tokens: 32,
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

    #[test]
    fn kv_identities_use_one_full_radix_path() {
        let config = test_config();
        let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
            .unwrap()
            .expect("cache enabled");
        let tokens = (0..160).collect::<Vec<_>>();

        let lookup_counts = kv
            .lookup_identities(&config, &test_base(), 0, &tokens)
            .into_iter()
            .map(|identity| identity.identity.token_count)
            .collect::<Vec<_>>();
        let record_counts = kv
            .record_identities(&config, &test_base(), 0, &tokens)
            .into_iter()
            .map(|identity| identity.identity.token_count)
            .collect::<Vec<_>>();

        assert_eq!(lookup_counts, vec![160]);
        assert_eq!(record_counts, vec![160]);
    }

    #[test]
    fn same_prefix_different_tail_identities_share_a_radix_namespace() {
        let config = StageConfig {
            ctx_size: 8192,
            kv_cache: Some(StageKvCacheConfig {
                min_tokens: 256,
                shared_prefix_stride_tokens: 128,
                ..test_config().kv_cache.expect("test cache config")
            }),
            ..test_config()
        };
        let kv = KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
            .unwrap()
            .expect("cache enabled");
        let base = test_base();
        let recorded_tokens = (0..2214).collect::<Vec<_>>();
        let mut lookup_tokens = recorded_tokens.clone();
        lookup_tokens.extend(100_000..100_017);

        let record_identities = kv.record_identities(&config, &base, 0, &recorded_tokens);
        let lookup_identities = kv.lookup_identities(&config, &base, 0, &lookup_tokens);

        let record_counts = record_identities
            .iter()
            .map(|identity| identity.identity.token_count)
            .collect::<Vec<_>>();
        assert_eq!(record_counts, vec![2214]);
        assert_eq!(lookup_identities[0].identity.token_count, 2231);
        assert_eq!(
            record_identities[0].namespace,
            lookup_identities[0].namespace
        );
        assert_ne!(record_identities[0].page_id, lookup_identities[0].page_id);

        let checkpoint = kv
            .exact_shared_checkpoint_identity(&config, &base, 0, &recorded_tokens)
            .expect("exact-state shared checkpoint");
        assert_eq!(checkpoint.identity.token_count, 2048);
        assert_eq!(checkpoint.namespace, record_identities[0].namespace);
        assert_eq!(checkpoint.token_ids, recorded_tokens[..2048]);
    }

    #[test]
    fn exact_state_radix_finds_cached_non_grid_prefix_length() {
        let config = StageConfig {
            ctx_size: 8192,
            model_id: "tiiuae/Falcon-H1-0.5B-Instruct-GGUF:Q4_K_M".to_string(),
            kv_cache: Some(StageKvCacheConfig {
                payload: StageKvCachePayload::KvRecurrent,
                min_tokens: 256,
                shared_prefix_stride_tokens: 128,
                max_entries: 1,
                ..test_config().kv_cache.expect("test cache config")
            }),
            ..test_config()
        };
        let kv =
            KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Recurrent)
                .unwrap()
                .expect("cache enabled");
        let base = test_base();
        let recorded_tokens = (0..2214).collect::<Vec<_>>();
        let recorded = kv.prefill_identity(&config, &base, 0, &recorded_tokens);
        kv.radix
            .lock()
            .expect("radix cache lock poisoned")
            .insert_recurrent(
                recorded.namespace.clone(),
                &recorded.token_ids,
                1,
                super::super::RadixExactEntry {
                    page_id: recorded.page_id.clone(),
                    payload: ExactStatePayload::kv_recurrent(Vec::new(), vec![1]),
                    extra: ExactStateExtra::default(),
                },
            )
            .unwrap();
        let mut lookup_tokens = recorded_tokens.clone();
        lookup_tokens.extend(100_000..100_017);

        let lookup = kv.lookup_identities(&config, &base, 0, &lookup_tokens);
        assert_eq!(lookup.len(), 1);
        let hit = kv
            .radix
            .lock()
            .expect("radix cache lock poisoned")
            .lookup_recurrent(&lookup[0].namespace, &lookup[0].token_ids)
            .expect("radix should find the retained exact-state prefix");

        assert_eq!(hit.matched_tokens, 2214);
        assert_eq!(hit.value.page_id, recorded.page_id);
    }

    #[test]
    fn exact_shared_checkpoint_rejects_empty_zero_minimum_checkpoint() {
        let config = StageConfig {
            model_id: "tiiuae/Falcon-H1-0.5B-Instruct-GGUF:Q4_K_M".to_string(),
            kv_cache: Some(StageKvCacheConfig {
                payload: StageKvCachePayload::KvRecurrent,
                min_tokens: 0,
                shared_prefix_stride_tokens: 128,
                ..test_config().kv_cache.expect("test cache config")
            }),
            ..test_config()
        };
        let kv =
            KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Recurrent)
                .unwrap()
                .expect("cache enabled");

        assert!(
            kv.exact_shared_checkpoint_identity(&config, &test_base(), 0, &[1])
                .is_none()
        );
    }

    #[test]
    fn exact_state_record_prefers_shared_checkpoint_over_request_tail() {
        let mut config = test_config();
        config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::KvRecurrent;
        config.kv_cache.as_mut().unwrap().min_tokens = 256;
        config
            .kv_cache
            .as_mut()
            .unwrap()
            .shared_prefix_stride_tokens = 128;
        let kv =
            KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Recurrent)
                .unwrap()
                .expect("cache enabled");

        assert!(kv.exact_state_record_token_count_allowed(970, 768));
        assert!(!kv.exact_state_record_token_count_allowed(970, 969));
        assert!(!kv.exact_state_record_token_count_allowed(970, 970));
        assert!(kv.exact_state_record_token_count_allowed(200, 200));
    }
}
