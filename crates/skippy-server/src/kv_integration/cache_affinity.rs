use skippy_protocol::StageConfig;
use skippy_scheduler::{CacheAffinity, StageCacheAffinity};

use super::{KvStageIntegration, PrefillKvIdentity, StagePrefixCachePayload};

impl KvStageIntegration {
    /// Inspect cache value for scheduling without mutating LRU recency.
    pub fn peek_cache_affinity(
        &self,
        config: &StageConfig,
        identities: &[PrefillKvIdentity],
    ) -> CacheAffinity {
        if !self.should_lookup() {
            return CacheAffinity::default();
        }
        let radix = match self.radix.try_lock() {
            Ok(radix) => radix,
            Err(std::sync::TryLockError::WouldBlock) => return CacheAffinity::default(),
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };
        let cache_epoch = radix.epoch();
        let matched_tokens = identities
            .iter()
            .filter_map(|identity| match self.payload {
                StagePrefixCachePayload::ResidentKv => radix
                    .peek_resident(&identity.namespace, &identity.token_ids)
                    .map(|hit| hit.matched_tokens),
                StagePrefixCachePayload::KvRecurrent | StagePrefixCachePayload::FullState => radix
                    .peek_recurrent(&identity.namespace, &identity.token_ids)
                    .map(|hit| hit.matched_tokens),
                StagePrefixCachePayload::Disabled => None,
            })
            .filter(|matched_tokens| self.meets_shared_prefix_min_tokens(*matched_tokens))
            .max()
            .unwrap_or(0);
        if matched_tokens == 0 {
            return CacheAffinity::default();
        }
        // Layer count is a deterministic first-order proxy for work saved on
        // heterogeneous stages. The policy type supports measured weights once
        // stage timing calibration is available.
        let prefill_cost_per_token =
            u64::from(config.layer_end.saturating_sub(config.layer_start).max(1));
        CacheAffinity::from_stage(StageCacheAffinity {
            stage_index: config.stage_index,
            matched_tokens,
            prefill_cost_per_token,
            restore_cost: 0,
            cache_epoch,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, mpsc},
        time::{Duration, Instant},
    };

    use skippy_protocol::{LoadMode, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload};

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

    #[test]
    fn busy_radix_returns_cold_affinity_without_waiting() {
        let config = test_config();
        let integration =
            KvStageIntegration::from_config(&config, skippy_runtime::ModelStateKind::Dense)
                .unwrap()
                .unwrap();
        let radix = Arc::clone(&integration.radix);
        let (locked_tx, locked_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _guard = radix.lock().unwrap();
            locked_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        locked_rx.recv().unwrap();

        let started = Instant::now();
        assert_eq!(
            integration.peek_cache_affinity(&config, &[]),
            CacheAffinity::default()
        );
        assert!(started.elapsed() < Duration::from_millis(100));

        release_tx.send(()).unwrap();
        holder.join().unwrap();
    }
}
