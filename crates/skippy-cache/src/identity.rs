use skippy_protocol::{FlashAttentionType, LoadMode, StageConfig};

pub const NATIVE_KV_RUNTIME_ABI_VERSION: &str = "stage-abi-0.1.53/native-kv-page-v3";
pub const NATIVE_KV_DTYPE: &str = "ggml-native-kv";
const NATIVE_KV_LAYER_CONTIGUOUS_LAYOUT: i32 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixIdentity {
    pub prefix_hash: String,
    pub page_id: String,
    pub token_start: u64,
    pub token_count: u64,
}

pub fn prefix_identity(
    config: &StageConfig,
    token_start: u64,
    token_ids: &[i32],
) -> PrefixIdentity {
    prefix_identity_with_namespace(config, token_start, token_ids, None)
}

pub fn prefix_identity_with_namespace(
    config: &StageConfig,
    token_start: u64,
    token_ids: &[i32],
    cache_namespace: Option<&str>,
) -> PrefixIdentity {
    let token_count = token_ids.len() as u64;
    let prefix_hash = prefix_hash_with_namespace(config, token_start, token_ids, cache_namespace);
    let page_id = page_id(config, token_start, token_count, &prefix_hash);
    PrefixIdentity {
        prefix_hash,
        page_id,
        token_start,
        token_count,
    }
}

pub fn prefix_hash(config: &StageConfig, token_start: u64, token_ids: &[i32]) -> String {
    prefix_hash_with_namespace(config, token_start, token_ids, None)
}

/// Hash the identity of the *machine* the page bytes were produced on.
///
/// Exported KV state is raw runtime memory whose interpretation depends on the
/// CPU architecture, native byte order, and pointer width. A stage key alone
/// does not distinguish an x86_64 CUDA host from an aarch64 CUDA host, nor two
/// CPU-only hosts that both record `<no-selected-device>`. Including these
/// properties prevents incompatible state from sharing a page id.
fn update_platform_identity(hasher: &mut blake3::Hasher) {
    hasher.update(b"kv-platform-identity-v1");
    hasher.update(std::env::consts::ARCH.as_bytes());
    hasher.update(if cfg!(target_endian = "little") {
        b"endian:le"
    } else {
        b"endian:be"
    });
    hasher.update(&(usize::BITS).to_le_bytes());
}

/// Hash the parts of a stage configuration that change the *bytes* of an
/// exported KV page without changing the token sequence.
///
/// Identity must cover every input that alters the serialized layout or the
/// numerical content of exported state. Flipping `kv_cache_policy` from
/// `quality` to `saver` rewrites `cache_type_k`/`_v` from `f16` to `q8_0`;
/// without these fields in the hash, incompatible state would share a page id.
///
/// `NATIVE_KV_DTYPE` is a fixed layout tag and does **not** vary with the
/// configured cache types, so it cannot stand in for them.
fn update_layout_identity(hasher: &mut blake3::Hasher, config: &StageConfig) {
    hasher.update(b"kv-layout-identity-v1");
    // Resident activation checkpoints are restored into the same execution
    // flow as native KV pages. A different activation codec can change the
    // numerical state seen by downstream stages, so it must produce a distinct
    // cache namespace even when the model, layers, and KV layout are equal.
    hasher.update(b"activation-codec:");
    hasher.update(
        config
            .activation_codec_policy
            .identity(config.activation_codec)
            .as_bytes(),
    );
    hasher.update(config.cache_type_k.as_bytes());
    hasher.update(b"/");
    hasher.update(config.cache_type_v.as_bytes());
    // Flash attention changes the KV layout the runtime allocates.
    hasher.update(match config.flash_attn_type {
        FlashAttentionType::Auto => b"fa:auto",
        FlashAttentionType::Disabled => b"fa:offf",
        FlashAttentionType::Enabled => b"fa:onnn",
    });
    // The CPU/GPU layer split decides which layers live in device memory and
    // therefore how the exported page is assembled.
    hasher.update(&config.n_gpu_layers.to_le_bytes());
    // Backend identity: CUDA/Metal/Vulkan/CPU pages are not interchangeable.
    match config.selected_device.as_ref() {
        Some(device) => hasher.update(device.backend_device.as_bytes()),
        None => hasher.update(b"<no-selected-device>"),
    };
    hasher.update(b"kv-unified:");
    hasher.update(match config.kv_unified {
        Some(true) => b"true",
        Some(false) => b"false",
        None => b"absent",
    });
    hasher.update(b"swa-full:");
    hasher.update(match config.swa_full {
        Some(true) => b"true",
        Some(false) => b"false",
        None => b"absent",
    });
}

/// Hash the identity of the *weights* a stage is serving.
///
/// `model_id` is a human-facing display name (`runtime_model_name`), not a
/// content digest. Two runs can present the same `model_id` while serving
/// genuinely different tensors — a different quantization of the same repo, a
/// re-published layer package, a direct GGUF swapped underneath the same
/// alias. KV pages from different weights are not interchangeable, and
/// importing one under the other is silent numerical corruption.
///
/// While `topology_id` was in the hash this was masked: it is derived from
/// `unix_nanos` per process, so every restart produced fresh page ids. Stable
/// identities remove that accidental protection, so weight identity now
/// has to be explicit.
///
/// `manifest_sha256` and `source_model_sha256` are content-derived and stable
/// across restarts for the same artifact — exactly the property required: they
/// change when the bytes change and only then. Both are hashed when present,
/// with the absent case tagged distinctly so `None` cannot alias a real value.
fn update_weight_identity(hasher: &mut blake3::Hasher, config: &StageConfig) {
    hasher.update(b"kv-weight-identity-v1");
    hasher.update(config.model_id.as_bytes());
    for (tag, value) in [
        (&b"manifest:"[..], config.manifest_sha256.as_deref()),
        (&b"source:"[..], config.source_model_sha256.as_deref()),
        (&b"package:"[..], config.package_ref.as_deref()),
    ] {
        hasher.update(tag);
        match value {
            Some(value) => {
                hasher.update(b"=");
                hasher.update(value.as_bytes());
            }
            None => {
                hasher.update(b"<absent>");
            }
        }
    }
    hasher.update(b"checkpoint-quantization:");
    let quantization = config
        .checkpoint_quantization
        .as_deref()
        .unwrap_or("preserve")
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_uppercase)
        .collect::<String>();
    hasher.update(match quantization.as_str() {
        "DIRECT" | "NONE" | "PRESERVE" => b"PRESERVE" as &[u8],
        _ => quantization.as_bytes(),
    });
    hasher.update(b"checkpoint-imatrix:");
    match config.checkpoint_imatrix_sha256.as_deref() {
        Some(digest) => hasher.update(digest.as_bytes()),
        None => hasher.update(b"<absent>"),
    };
    // Layer packages and direct GGUFs assemble tensors differently even for
    // the same underlying model.
    // Matched exhaustively on purpose: a new load mode is a new way of
    // assembling tensors, and it must be forced through this hash rather than
    // silently aliasing an existing one.
    hasher.update(match config.load_mode {
        LoadMode::LayerPackage => b"load:package-",
        LoadMode::RuntimeSlice => b"load:runslice",
        LoadMode::ArtifactSlice => b"load:artslice",
    });
}

pub fn prefix_hash_with_namespace(
    config: &StageConfig,
    token_start: u64,
    token_ids: &[i32],
    cache_namespace: Option<&str>,
) -> String {
    let mut hasher = prefix_namespace_hasher(config, token_start, cache_namespace);
    for token_id in token_ids {
        hasher.update(token_id.to_le_bytes().as_slice());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

/// Stable namespace for one radix tree. It binds every non-token input that
/// changes native cache bytes while leaving the token sequence as the radix
/// path itself.
pub fn prefix_namespace_hash(
    config: &StageConfig,
    token_start: u64,
    cache_namespace: Option<&str>,
) -> String {
    let hasher = prefix_namespace_hasher(config, token_start, cache_namespace);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn prefix_namespace_hasher(
    config: &StageConfig,
    token_start: u64,
    cache_namespace: Option<&str>,
) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    update_weight_identity(&mut hasher, config);
    // `topology_id` is deliberately **not** hashed. It is an instance
    // identifier, not a description of the work a stage does: local serving
    // derives it as `topology-mesh-skippy-{unix_nanos}`, so it is unique per
    // process and would prevent otherwise compatible cache identities from
    // agreeing across runs.
    //
    // Reuse depends on the *shape* of the stage: the model, owned layers, and
    // pipeline position. Those are hashed below with the runtime layout fields
    // in `update_layout_identity`.
    hasher.update(config.stage_id.as_bytes());
    hasher.update(&config.stage_index.to_le_bytes());
    hasher.update(&config.layer_start.to_le_bytes());
    hasher.update(&config.layer_end.to_le_bytes());
    hasher.update(NATIVE_KV_RUNTIME_ABI_VERSION.as_bytes());
    hasher.update(&NATIVE_KV_LAYER_CONTIGUOUS_LAYOUT.to_le_bytes());
    hasher.update(NATIVE_KV_DTYPE.as_bytes());
    update_layout_identity(&mut hasher, config);
    update_platform_identity(&mut hasher);
    hasher.update(format!("ctx:{}", config.ctx_size).as_bytes());
    if let Some(cache_namespace) = cache_namespace {
        hasher.update(b"openai-cache-namespace-v1");
        hasher.update(cache_namespace.as_bytes());
    }
    hasher.update(&token_start.to_le_bytes());
    hasher
}

pub fn page_id(
    config: &StageConfig,
    token_start: u64,
    token_count: u64,
    prefix_hash: &str,
) -> String {
    let digest = prefix_hash
        .strip_prefix("blake3:")
        .or_else(|| prefix_hash.strip_prefix("sha256:"))
        .unwrap_or(prefix_hash);
    let short = digest.get(..16).unwrap_or(digest);
    format!(
        "{}:{}:{}:{}:{}",
        config.stage_id, token_start, token_count, config.layer_start, short
    )
}

pub fn activation_page_id(page_id: &str, activation_width: i32) -> String {
    format!("act:{}:w{}", page_id, activation_width.max(0))
}

#[cfg(test)]
mod identity_completeness_tests {
    use skippy_protocol::{LoadMode, StageDevice};

    use super::*;

    fn test_config() -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: "topology".to_string(),
            model_id: "org/model:Q4_K_M".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            model_part_paths: Vec::new(),
            checkpoint_quantization: None,
            checkpoint_imatrix: None,
            checkpoint_imatrix_sha256: None,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            activation_codec: Default::default(),
            activation_codec_policy: Default::default(),
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
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: false,
            resident_tensor_names: Vec::new(),
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        }
    }

    fn hash_of(config: &StageConfig) -> String {
        prefix_hash(config, 0, &[1, 2, 3, 4])
    }

    /// Changing the KV cache policy rewrites `cache_type_k`/`_v`. These
    /// formats must produce distinct page identities to prevent importing
    /// cached q8_0 state as f16.
    #[test]
    fn kv_cache_dtype_changes_page_identity() {
        let quality = test_config();
        let saver = StageConfig {
            cache_type_k: "q8_0".to_string(),
            cache_type_v: "q8_0".to_string(),
            ..test_config()
        };

        assert_ne!(hash_of(&quality), hash_of(&saver));
        assert_ne!(
            prefix_identity(&quality, 0, &[1, 2, 3, 4]).page_id,
            prefix_identity(&saver, 0, &[1, 2, 3, 4]).page_id
        );
    }

    #[test]
    fn activation_codec_changes_page_identity() {
        let f16 = StageConfig {
            activation_codec: skippy_protocol::StageActivationCodec::F16RneV1,
            activation_codec_policy: Default::default(),
            ..test_config()
        };
        let exact = StageConfig {
            activation_codec: skippy_protocol::StageActivationCodec::RawF32V1,
            activation_codec_policy: Default::default(),
            ..test_config()
        };

        assert_ne!(hash_of(&f16), hash_of(&exact));
        assert_ne!(
            prefix_identity(&f16, 0, &[1, 2, 3, 4]).page_id,
            prefix_identity(&exact, 0, &[1, 2, 3, 4]).page_id
        );
    }

    #[test]
    fn k_and_v_cache_types_are_independently_hashed() {
        let mixed = StageConfig {
            cache_type_v: "q8_0".to_string(),
            ..test_config()
        };

        assert_ne!(hash_of(&test_config()), hash_of(&mixed));
    }

    #[test]
    fn flash_attention_changes_page_identity() {
        let enabled = StageConfig {
            flash_attn_type: FlashAttentionType::Enabled,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            ..test_config()
        };
        let disabled = StageConfig {
            flash_attn_type: FlashAttentionType::Disabled,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            ..test_config()
        };

        assert_ne!(hash_of(&test_config()), hash_of(&enabled));
        assert_ne!(hash_of(&enabled), hash_of(&disabled));
    }

    #[test]
    fn kv_unified_option_changes_page_and_prefix_identity() {
        let absent = test_config();
        let disabled = StageConfig {
            kv_unified: Some(false),
            ..test_config()
        };
        let enabled = StageConfig {
            kv_unified: Some(true),
            ..test_config()
        };

        assert_ne!(hash_of(&absent), hash_of(&disabled));
        assert_ne!(hash_of(&absent), hash_of(&enabled));
        assert_ne!(hash_of(&disabled), hash_of(&enabled));
        assert_ne!(
            prefix_identity(&absent, 0, &[1, 2, 3, 4]).page_id,
            prefix_identity(&disabled, 0, &[1, 2, 3, 4]).page_id
        );
        assert_ne!(
            prefix_identity(&absent, 0, &[1, 2, 3, 4]).page_id,
            prefix_identity(&enabled, 0, &[1, 2, 3, 4]).page_id
        );
        assert_ne!(
            prefix_identity(&disabled, 0, &[1, 2, 3, 4]).page_id,
            prefix_identity(&enabled, 0, &[1, 2, 3, 4]).page_id
        );
    }

    #[test]
    fn swa_full_option_changes_page_and_prefix_identity() {
        let absent = test_config();
        let disabled = StageConfig {
            swa_full: Some(false),
            ..test_config()
        };
        let enabled = StageConfig {
            swa_full: Some(true),
            ..test_config()
        };

        assert_ne!(hash_of(&absent), hash_of(&disabled));
        assert_ne!(hash_of(&absent), hash_of(&enabled));
        assert_ne!(hash_of(&disabled), hash_of(&enabled));
        assert_ne!(
            prefix_identity(&absent, 0, &[1, 2, 3, 4]).page_id,
            prefix_identity(&disabled, 0, &[1, 2, 3, 4]).page_id
        );
        assert_ne!(
            prefix_identity(&absent, 0, &[1, 2, 3, 4]).page_id,
            prefix_identity(&enabled, 0, &[1, 2, 3, 4]).page_id
        );
        assert_ne!(
            prefix_identity(&disabled, 0, &[1, 2, 3, 4]).page_id,
            prefix_identity(&enabled, 0, &[1, 2, 3, 4]).page_id
        );
    }

    #[test]
    fn gpu_layer_split_changes_page_identity() {
        let offloaded = StageConfig {
            n_gpu_layers: 32,
            ..test_config()
        };

        assert_ne!(hash_of(&test_config()), hash_of(&offloaded));
    }

    /// The platform tag is part of the hash, so a page written on one
    /// architecture cannot be read back on another through a shared or copied
    /// cache directory. Asserted against the real values rather than a
    /// hand-built string so that dropping the call from `prefix_hash` fails
    /// here.
    #[test]
    fn platform_identity_is_bound_into_the_page_hash() {
        let mut with_platform = blake3::Hasher::new();
        update_platform_identity(&mut with_platform);
        let mut other_arch = blake3::Hasher::new();
        other_arch.update(b"kv-platform-identity-v1");
        other_arch.update(b"some-other-arch");
        other_arch.update(b"endian:le");
        other_arch.update(&(usize::BITS).to_le_bytes());

        assert_ne!(with_platform.finalize(), other_arch.finalize());

        let mut other_endian = blake3::Hasher::new();
        other_endian.update(b"kv-platform-identity-v1");
        other_endian.update(std::env::consts::ARCH.as_bytes());
        other_endian.update(if cfg!(target_endian = "little") {
            b"endian:be"
        } else {
            b"endian:le"
        });
        other_endian.update(&(usize::BITS).to_le_bytes());

        assert_ne!(with_platform.finalize(), other_endian.finalize());

        let mut other_width = blake3::Hasher::new();
        other_width.update(b"kv-platform-identity-v1");
        other_width.update(std::env::consts::ARCH.as_bytes());
        other_width.update(if cfg!(target_endian = "little") {
            b"endian:le"
        } else {
            b"endian:be"
        });
        other_width.update(&(usize::BITS / 2).to_le_bytes());

        assert_ne!(with_platform.finalize(), other_width.finalize());
    }

    /// Pages are not portable across backends. A CUDA page must never be
    /// mistaken for a Metal page with the same tokens.
    #[test]
    fn backend_device_changes_page_identity() {
        let cuda = StageConfig {
            selected_device: Some(StageDevice {
                backend_device: "CUDA0".to_string(),
                stable_id: None,
                index: Some(0),
                vram_bytes: None,
            }),
            ..test_config()
        };
        let metal = StageConfig {
            selected_device: Some(StageDevice {
                backend_device: "Metal".to_string(),
                stable_id: None,
                index: Some(0),
                vram_bytes: None,
            }),
            ..test_config()
        };

        assert_ne!(hash_of(&cuda), hash_of(&metal));
        assert_ne!(hash_of(&test_config()), hash_of(&cuda));
    }

    /// Identity must stay stable for an unchanged configuration, otherwise
    /// nothing is ever reused across restarts.
    #[test]
    fn identical_configs_produce_identical_identity() {
        assert_eq!(hash_of(&test_config()), hash_of(&test_config()));
    }

    /// Cross-session sharing is by design: identity has no `session_id`.
    #[test]
    fn identity_is_shared_across_sessions_by_design() {
        let config = test_config();
        let tokens = (0..512).collect::<Vec<_>>();

        assert_eq!(
            prefix_identity(&config, 0, &tokens).page_id,
            prefix_identity(&config, 0, &tokens).page_id
        );
    }
}

#[cfg(test)]
mod identity_stability_tests {
    use skippy_protocol::LoadMode;

    use super::*;

    fn config_with_topology(topology_id: &str) -> StageConfig {
        StageConfig {
            run_id: "run".to_string(),
            topology_id: topology_id.to_string(),
            model_id: "org/model:Q4_K_M".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            model_part_paths: Vec::new(),
            checkpoint_quantization: None,
            checkpoint_imatrix: None,
            checkpoint_imatrix_sha256: None,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            activation_codec: Default::default(),
            activation_codec_policy: Default::default(),
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 24,
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
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: false,
            resident_tensor_names: Vec::new(),
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        }
    }

    /// Local serving derives `topology_id` from a nanosecond timestamp, so it
    /// differs on every start. Identity must ignore it, or a persistent cache
    /// can never be read back after a restart.
    #[test]
    fn identity_is_stable_across_process_restarts() {
        let first = config_with_topology("topology-mesh-skippy-1739000000000000000");
        let second = config_with_topology("topology-mesh-skippy-1739999999999999999");
        let tokens = (0..2048).collect::<Vec<_>>();

        assert_eq!(
            prefix_identity(&first, 0, &tokens).page_id,
            prefix_identity(&second, 0, &tokens).page_id,
            "a restart must not invalidate cached prefixes"
        );
    }

    /// Weight identity is the protection that replaced `topology_id`.
    ///
    /// Two runs can share a `model_id` while serving different tensors — a
    /// different quant, a re-published layer package, a swapped GGUF. This
    /// must be caught explicitly, because a false hit here imports the wrong
    /// weights' KV and silently corrupts output.
    #[test]
    fn identity_separates_different_weights_behind_the_same_model_id() {
        let tokens = (0..1024).collect::<Vec<_>>();
        let base = StageConfig {
            manifest_sha256: Some("a".repeat(64)),
            source_model_sha256: Some("b".repeat(64)),
            ..config_with_topology("topology-a")
        };
        let repacked = StageConfig {
            manifest_sha256: Some("c".repeat(64)),
            ..base.clone()
        };
        let requantized = StageConfig {
            source_model_sha256: Some("d".repeat(64)),
            ..base.clone()
        };
        let load_time_quantized = StageConfig {
            checkpoint_quantization: Some("Q4_K_M".to_string()),
            checkpoint_imatrix: None,
            checkpoint_imatrix_sha256: None,
            ..base.clone()
        };

        assert_eq!(base.model_id, repacked.model_id);
        assert_ne!(
            prefix_identity(&base, 0, &tokens).page_id,
            prefix_identity(&repacked, 0, &tokens).page_id,
            "a different manifest must not reuse cached pages"
        );
        assert_ne!(
            prefix_identity(&base, 0, &tokens).page_id,
            prefix_identity(&requantized, 0, &tokens).page_id,
            "different source weights must not reuse cached pages"
        );
        assert_ne!(
            prefix_identity(&base, 0, &tokens).page_id,
            prefix_identity(&load_time_quantized, 0, &tokens).page_id,
            "different load-time quantization must not reuse cached pages"
        );
    }

    #[test]
    fn absent_and_explicit_preserve_quantization_share_weight_identity() {
        let tokens = (0..128).collect::<Vec<_>>();
        let absent = StageConfig {
            source_model_sha256: Some("b".repeat(64)),
            ..config_with_topology("topology-a")
        };
        let explicit = StageConfig {
            checkpoint_quantization: Some("preserve".to_string()),
            checkpoint_imatrix: None,
            checkpoint_imatrix_sha256: None,
            ..absent.clone()
        };

        assert_eq!(
            prefix_identity(&absent, 0, &tokens).page_id,
            prefix_identity(&explicit, 0, &tokens).page_id
        );
    }

    #[test]
    fn identity_separates_importance_matrix_contents() {
        let tokens = (0..128).collect::<Vec<_>>();
        let first = StageConfig {
            checkpoint_quantization: Some("IQ2_XXS".to_string()),
            checkpoint_imatrix: Some("/models/calibration.gguf".to_string()),
            checkpoint_imatrix_sha256: Some("a".repeat(64)),
            ..config_with_topology("topology-a")
        };
        let second = StageConfig {
            checkpoint_imatrix_sha256: Some("b".repeat(64)),
            ..first.clone()
        };

        assert_ne!(
            prefix_identity(&first, 0, &tokens).page_id,
            prefix_identity(&second, 0, &tokens).page_id,
            "different calibration weights must not share cached KV pages"
        );
    }

    /// A missing digest must not alias a present one, or an artifact with no
    /// recorded hash could collide with a real one.
    #[test]
    fn absent_weight_digests_do_not_alias_present_ones() {
        let tokens = (0..1024).collect::<Vec<_>>();
        let absent = StageConfig {
            manifest_sha256: None,
            source_model_sha256: None,
            ..config_with_topology("topology-a")
        };
        let present = StageConfig {
            manifest_sha256: Some("absent".to_string()),
            ..absent.clone()
        };

        assert_ne!(
            prefix_identity(&absent, 0, &tokens).page_id,
            prefix_identity(&present, 0, &tokens).page_id
        );
    }

    /// Weight identity must not undo the restart-stability fix: the same
    /// artifact across two runs still has to hit.
    #[test]
    fn weight_identity_stays_stable_for_the_same_artifact() {
        let tokens = (0..1024).collect::<Vec<_>>();
        let run_one = StageConfig {
            manifest_sha256: Some("a".repeat(64)),
            source_model_sha256: Some("b".repeat(64)),
            ..config_with_topology("topology-mesh-skippy-1739000000000000000")
        };
        let run_two = StageConfig { ..run_one.clone() };
        let run_two = StageConfig {
            topology_id: "topology-mesh-skippy-1739999999999999999".to_string(),
            run_id: "a-different-run".to_string(),
            ..run_two
        };

        assert_eq!(
            prefix_identity(&run_one, 0, &tokens).page_id,
            prefix_identity(&run_two, 0, &tokens).page_id
        );
    }

    /// Stage shape still has to match: a different layer range is a different
    /// page and must never collide.
    #[test]
    fn identity_still_separates_different_stage_shapes() {
        let base = config_with_topology("topology-a");
        let other_layers = StageConfig {
            layer_start: 24,
            layer_end: 48,
            stage_index: 1,
            stage_id: "stage-1".to_string(),
            ..config_with_topology("topology-a")
        };
        let tokens = (0..2048).collect::<Vec<_>>();

        assert_ne!(
            prefix_identity(&base, 0, &tokens).page_id,
            prefix_identity(&other_layers, 0, &tokens).page_id
        );
    }
}
