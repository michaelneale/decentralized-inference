use std::ffi::CString;
use std::ptr;

use anyhow::{Context, Result, anyhow};
use skippy_ffi::{
    LoadMode, MtpSource as RawMtpSource, RuntimeConfig as RawRuntimeConfig, TRISTATE_AUTO,
    TRISTATE_FALSE, TRISTATE_TRUE,
};

use crate::checkpoint::CheckpointQuantization;

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const LLAMA_SERVER_DEFAULT_N_BATCH: u32 = 2048;
pub const LLAMA_SERVER_DEFAULT_N_UBATCH: u32 = 512;
/// Unified-KV prefill batch default. Keep llama-server's 2048-token batch so
/// the always-unified serving cutover does not regress the N=1 baseline.
pub const SKIPPY_UNIFIED_KV_DEFAULT_N_BATCH: u32 = LLAMA_SERVER_DEFAULT_N_BATCH;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum FlashAttentionType {
    #[default]
    Auto = -1,
    Disabled = 0,
    Enabled = 1,
}

/// Selects whether this target keeps no MTP tensors, its integrated MTP heads,
/// or waits for an external MTP sidecar to attach.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MtpSource {
    #[default]
    Disabled,
    Integrated,
    External,
}

impl MtpSource {
    const fn as_raw(self) -> RawMtpSource {
        match self {
            Self::Disabled => RawMtpSource::Disabled,
            Self::Integrated => RawMtpSource::Integrated,
            Self::External => RawMtpSource::External,
        }
    }
}

/// How to split the model across multiple GPUs, mirroring upstream
/// `llama_split_mode`. `Auto` preserves llama.cpp's derived default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SplitMode {
    #[default]
    Auto,
    None,
    Layer,
    Row,
    Tensor,
}

impl SplitMode {
    const fn as_raw(self) -> i32 {
        match self {
            Self::Auto => TRISTATE_AUTO,
            Self::None => 0,
            Self::Layer => 1,
            Self::Row => 2,
            Self::Tensor => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub ctx_size: u32,
    pub lane_count: u32,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_threads: Option<u32>,
    pub n_threads_batch: Option<u32>,
    pub n_gpu_layers: i32,
    pub mmap: Option<bool>,
    pub mlock: bool,
    pub repack: bool,
    pub selected_backend_device: Option<String>,
    pub cache_type_k: u32,
    pub cache_type_v: u32,
    pub flash_attn_type: FlashAttentionType,
    pub load_mode: LoadMode,
    pub projector_path: Option<String>,
    pub projector_use_gpu: Option<bool>,
    pub media_marker: Option<String>,
    pub image_min_tokens: Option<u32>,
    pub image_max_tokens: Option<u32>,
    pub batch_max_tokens: Option<u32>,
    pub glm_dsa_policy: GlmDsaPolicy,
    pub include_embeddings: bool,
    pub include_output: bool,
    pub mtp_source: MtpSource,
    pub filter_tensors_on_load: bool,
    /// Exact native tensor names admitted for this stage. Empty preserves the
    /// legacy range-based loader filter.
    pub resident_tensor_names: Vec<String>,
    /// Tensor type policy used when `StageModel::open` receives a SafeTensors checkpoint.
    pub checkpoint_quantization: CheckpointQuantization,
    /// Importance matrix used by quantization recipes that require calibration data.
    pub checkpoint_imatrix: Option<std::path::PathBuf>,
    /// Expected SHA-256 for `checkpoint_imatrix`. When set, the runtime verifies
    /// the exact bytes that it passes to the importance-matrix parser.
    pub checkpoint_imatrix_sha256: Option<String>,
    /// K/V cache backend offload. `None` preserves llama.cpp's derived
    /// default (offloaded); `Some` forces the value.
    pub kv_offload: Option<bool>,
    /// Whether the KV cache is unified across sequences/lanes. `None`
    /// preserves the lane-count/recurrent-architecture derived default;
    /// `Some` forces the value. Recurrent/hybrid architectures still force
    /// this true natively regardless of the requested value.
    pub kv_unified: Option<bool>,
    /// Sliding-window-attention full (unshifted) cache window. `None`
    /// preserves llama.cpp's built-in default (full).
    pub swa_full: Option<bool>,
    /// Whether the backend offloads host tensor operations to device. `None`
    /// preserves llama.cpp's derived default (enabled).
    pub op_offload: Option<bool>,
    /// Whether model loading bypasses the host buffer, allowing extra
    /// buffers to be used.
    pub no_host_buffer: bool,
    /// Whether to validate model tensor data while loading.
    pub check_tensors: bool,
    /// Forces direct I/O for model loading, taking precedence over `mmap`
    /// and `mlock` when set.
    pub direct_io: bool,
    /// Explicit GPU index used for the entire model when `split_mode`
    /// resolves to `none`. `None` preserves llama.cpp's derived default.
    pub main_gpu: Option<u32>,
    /// How to split the model across multiple GPUs.
    pub split_mode: SplitMode,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GlmDsaPolicy {
    #[default]
    Auto,
    V1,
}

fn tristate(value: Option<bool>) -> i32 {
    match value {
        None => TRISTATE_AUTO,
        Some(false) => TRISTATE_FALSE,
        Some(true) => TRISTATE_TRUE,
    }
}

impl RuntimeConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.layer_start >= self.layer_end {
            return Err("layer_start must be less than layer_end");
        }
        if self
            .selected_backend_device
            .as_deref()
            .is_some_and(str::is_empty)
        {
            return Err("selected_backend_device must not be empty");
        }
        if self.projector_path.as_deref().is_some_and(str::is_empty) {
            return Err("projector_path must not be empty");
        }
        if self.n_batch == Some(0) {
            return Err("n_batch must be greater than zero when provided");
        }
        if self.n_ubatch == Some(0) {
            return Err("n_ubatch must be greater than zero when provided");
        }
        if self.n_threads == Some(0) {
            return Err("n_threads must be greater than zero when provided");
        }
        if self.n_threads_batch == Some(0) {
            return Err("n_threads_batch must be greater than zero when provided");
        }
        if self.filter_tensors_on_load && self.resident_tensor_names.is_empty() {
            return Err("filtered stage loading requires an explicit admitted resident tensor set");
        }
        Ok(())
    }

    pub(crate) fn as_raw(&self) -> Result<RawRuntimeConfigParts> {
        self.validate().map_err(anyhow::Error::msg)?;
        let mmap = self.mmap.or_else(|| {
            (self.filter_tensors_on_load && self.load_mode == LoadMode::RuntimeSlice)
                .then_some(false)
        });
        let n_batch = self
            .n_batch
            .unwrap_or_else(|| default_n_batch_for_lane_count(self.lane_count));
        let n_ubatch = self.n_ubatch.unwrap_or(LLAMA_SERVER_DEFAULT_N_UBATCH);
        let selected_backend_device = self
            .selected_backend_device
            .as_ref()
            .map(|device| {
                CString::new(device.as_bytes())
                    .context("selected_backend_device contains an interior NUL byte")
            })
            .transpose()?;
        let selected_backend_device_ptr = selected_backend_device
            .as_ref()
            .map(|device| device.as_ptr())
            .unwrap_or(ptr::null());
        let resident_tensor_names = self
            .resident_tensor_names
            .iter()
            .map(|name| {
                anyhow::ensure!(!name.is_empty(), "resident tensor name must not be empty");
                CString::new(name.as_bytes())
                    .context("resident tensor name contains an interior NUL byte")
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(
            resident_tensor_names
                .windows(2)
                .all(|window| window[0].as_bytes() < window[1].as_bytes()),
            "resident tensor names must be strictly sorted and unique"
        );
        let resident_tensor_name_ptrs = resident_tensor_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();
        let resident_tensor_names_ptr = if resident_tensor_name_ptrs.is_empty() {
            ptr::null()
        } else {
            resident_tensor_name_ptrs.as_ptr()
        };
        Ok(RawRuntimeConfigParts {
            raw: RawRuntimeConfig {
                stage_index: i32::try_from(self.stage_index).context("stage_index exceeds i32")?,
                layer_start: i32::try_from(self.layer_start).context("layer_start exceeds i32")?,
                layer_end: i32::try_from(self.layer_end).context("layer_end exceeds i32")?,
                ctx_size: i32::try_from(self.ctx_size).context("ctx_size exceeds i32")?,
                lane_count: i32::try_from(self.lane_count).context("lane_count exceeds i32")?,
                n_batch: i32::try_from(n_batch).context("n_batch exceeds i32")?,
                n_ubatch: i32::try_from(n_ubatch).context("n_ubatch exceeds i32")?,
                n_threads: self
                    .n_threads
                    .map(i32::try_from)
                    .transpose()
                    .context("n_threads exceeds i32")?
                    .unwrap_or(0),
                n_threads_batch: self
                    .n_threads_batch
                    .or(self.n_threads)
                    .map(i32::try_from)
                    .transpose()
                    .context("n_threads_batch exceeds i32")?
                    .unwrap_or(0),
                n_gpu_layers: self.n_gpu_layers,
                has_mmap_override: mmap.is_some(),
                use_mmap: mmap.unwrap_or(false),
                use_mlock: self.mlock,
                cache_type_k: i32::try_from(self.cache_type_k)
                    .context("cache_type_k exceeds i32")?,
                cache_type_v: i32::try_from(self.cache_type_v)
                    .context("cache_type_v exceeds i32")?,
                flash_attn_type: self.flash_attn_type as i32,
                load_mode: self.load_mode,
                disable_repack: !self.repack,
                use_mmap_prefetch: false,
                use_mmap_buffer: false,
                filter_tensors_on_load: self.filter_tensors_on_load,
                resident_tensor_names: resident_tensor_names_ptr,
                resident_tensor_name_count: resident_tensor_name_ptrs.len(),
                include_embeddings: self.include_embeddings,
                include_output: self.include_output,
                mtp_source: self.mtp_source.as_raw(),
                selected_backend_device: selected_backend_device_ptr,
                glm_dsa_policy_profile: match self.glm_dsa_policy {
                    GlmDsaPolicy::Auto => 0,
                    GlmDsaPolicy::V1 => 1,
                },
                glm_dsa_policy_flags: 0,
                glm_dsa_short_prefill_max_tokens: match self.glm_dsa_policy {
                    GlmDsaPolicy::Auto => 0,
                    GlmDsaPolicy::V1 => 2048,
                },
                glm_dsa_direct_sparse_decode_max_top_k: 0,
                glm_dsa_dense_sparse_mask_max_bytes: match self.glm_dsa_policy {
                    GlmDsaPolicy::Auto => 0,
                    GlmDsaPolicy::V1 => 512 * 1024 * 1024,
                },
                glm_dsa_compact_flash_min_kv: match self.glm_dsa_policy {
                    GlmDsaPolicy::Auto => 0,
                    GlmDsaPolicy::V1 => 1,
                },
                kv_offload: tristate(self.kv_offload),
                kv_unified: tristate(self.kv_unified),
                swa_full: tristate(self.swa_full),
                op_offload: tristate(self.op_offload),
                no_host_buffer: self.no_host_buffer,
                check_tensors: self.check_tensors,
                use_direct_io: self.direct_io,
                has_main_gpu_override: self.main_gpu.is_some(),
                main_gpu: self
                    .main_gpu
                    .map(i32::try_from)
                    .transpose()
                    .context("main_gpu exceeds i32")?
                    .unwrap_or(0),
                split_mode: self.split_mode.as_raw(),
            },
            _selected_backend_device: selected_backend_device,
            _resident_tensor_names: resident_tensor_names,
            _resident_tensor_name_ptrs: resident_tensor_name_ptrs,
        })
    }

    pub(crate) fn native_log_summary(&self) -> String {
        let n_batch = self
            .n_batch
            .unwrap_or_else(|| default_n_batch_for_lane_count(self.lane_count));
        let n_ubatch = self.n_ubatch.unwrap_or(LLAMA_SERVER_DEFAULT_N_UBATCH);
        format!(
            "stage_index={} layers={}..{} ctx={} lanes={} n_batch={} n_ubatch={} n_gpu_layers={} mmap={} mlock={} repack={} backend={} cache_k={} cache_v={} flash_attn={:?} load_mode={:?} include_embeddings={} include_output={} mtp_source={:?} filter_tensors_on_load={} resident_tensor_count={} checkpoint_quantization={:?}",
            self.stage_index,
            self.layer_start,
            self.layer_end,
            self.ctx_size,
            self.lane_count,
            n_batch,
            n_ubatch,
            self.n_gpu_layers,
            self.mmap
                .map(|value| value.to_string())
                .unwrap_or_else(|| "auto".to_string()),
            self.mlock,
            self.repack,
            self.selected_backend_device.as_deref().unwrap_or("auto"),
            self.cache_type_k,
            self.cache_type_v,
            self.flash_attn_type,
            self.load_mode,
            self.include_embeddings,
            self.include_output,
            self.mtp_source,
            self.filter_tensors_on_load,
            self.resident_tensor_names.len(),
            self.checkpoint_quantization,
        )
    }
}

/// Unified KV is always enabled; the per-lane batch distinction is removed.
pub(crate) fn default_n_batch_for_lane_count(_lane_count: u32) -> u32 {
    SKIPPY_UNIFIED_KV_DEFAULT_N_BATCH
}

#[derive(Debug)]
pub(crate) struct RawRuntimeConfigParts {
    pub(crate) raw: RawRuntimeConfig,
    _selected_backend_device: Option<CString>,
    _resident_tensor_names: Vec<CString>,
    _resident_tensor_name_ptrs: Vec<*const std::ffi::c_char>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 512,
            lane_count: 1,
            n_batch: Some(SKIPPY_UNIFIED_KV_DEFAULT_N_BATCH),
            n_ubatch: Some(LLAMA_SERVER_DEFAULT_N_UBATCH),
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            repack: false,
            selected_backend_device: None,
            cache_type_k: GGML_TYPE_F16,
            cache_type_v: GGML_TYPE_F16,
            flash_attn_type: FlashAttentionType::Auto,
            load_mode: LoadMode::RuntimeSlice,
            projector_path: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: GlmDsaPolicy::Auto,
            include_embeddings: true,
            include_output: true,
            mtp_source: MtpSource::Disabled,
            filter_tensors_on_load: false,
            resident_tensor_names: Vec::new(),
            checkpoint_quantization: CheckpointQuantization::Preserve,
            checkpoint_imatrix: None,
            checkpoint_imatrix_sha256: None,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: SplitMode::Auto,
        }
    }
}

pub fn parse_cache_type(value: &str) -> Result<u32> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "" | "f16" => Ok(GGML_TYPE_F16),
        "q4" | "q4_0" => Ok(GGML_TYPE_Q4_0),
        "q8" | "q8_0" => Ok(GGML_TYPE_Q8_0),
        _ => Err(anyhow!("unsupported KV cache type {value:?}")),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::*;

    #[test]
    fn runtime_config_rejects_empty_selected_backend_device() {
        let config = RuntimeConfig {
            selected_backend_device: Some(String::new()),
            ..RuntimeConfig::default()
        };

        assert_eq!(
            config.validate(),
            Err("selected_backend_device must not be empty")
        );
    }

    #[test]
    fn runtime_config_rejects_zero_thread_counts() {
        let thread_config = RuntimeConfig {
            n_threads: Some(0),
            ..RuntimeConfig::default()
        };
        let batch_thread_config = RuntimeConfig {
            n_threads_batch: Some(0),
            ..RuntimeConfig::default()
        };

        assert_eq!(
            thread_config.validate(),
            Err("n_threads must be greater than zero when provided")
        );
        assert_eq!(
            batch_thread_config.validate(),
            Err("n_threads_batch must be greater than zero when provided")
        );
    }

    #[test]
    fn runtime_config_raw_mmap_override_and_mlock_are_distinct() -> anyhow::Result<()> {
        let forced_config = RuntimeConfig {
            mmap: Some(false),
            mlock: true,
            ..RuntimeConfig::default()
        };
        let forced_raw = forced_config.as_raw()?;

        assert!(forced_raw.raw.has_mmap_override);
        assert!(!forced_raw.raw.use_mmap);
        assert!(forced_raw.raw.use_mlock);

        let auto_config = RuntimeConfig {
            mmap: None,
            mlock: false,
            ..RuntimeConfig::default()
        };
        let auto_raw = auto_config.as_raw()?;

        assert!(!auto_raw.raw.has_mmap_override);
        assert!(!auto_raw.raw.use_mmap);
        assert!(!auto_raw.raw.use_mlock);

        Ok(())
    }

    #[test]
    fn filtered_runtime_slice_disables_mmap_by_default() -> anyhow::Result<()> {
        let filtered = RuntimeConfig {
            mmap: None,
            filter_tensors_on_load: true,
            resident_tensor_names: vec!["output.weight".into()],
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let explicitly_enabled = RuntimeConfig {
            mmap: Some(true),
            filter_tensors_on_load: true,
            resident_tensor_names: vec!["output.weight".into()],
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert!(filtered.has_mmap_override);
        assert!(!filtered.use_mmap);
        assert!(explicitly_enabled.has_mmap_override);
        assert!(explicitly_enabled.use_mmap);

        let materialized = RuntimeConfig {
            mmap: None,
            filter_tensors_on_load: false,
            load_mode: LoadMode::LayerPackage,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        assert!(!materialized.has_mmap_override);

        Ok(())
    }

    #[test]
    fn runtime_config_raw_preserves_exact_resident_tensor_names() -> anyhow::Result<()> {
        let parts = RuntimeConfig {
            resident_tensor_names: vec!["blk.0.attn.weight".into(), "output.weight".into()],
            ..RuntimeConfig::default()
        }
        .as_raw()?;

        assert_eq!(parts.raw.resident_tensor_name_count, 2);
        assert!(!parts.raw.resident_tensor_names.is_null());
        let names = unsafe {
            std::slice::from_raw_parts(
                parts.raw.resident_tensor_names,
                parts.raw.resident_tensor_name_count,
            )
        };
        assert_eq!(
            unsafe { CStr::from_ptr(names[0]) }.to_str()?,
            "blk.0.attn.weight"
        );
        assert_eq!(
            unsafe { CStr::from_ptr(names[1]) }.to_str()?,
            "output.weight"
        );

        let empty = RuntimeConfig::default().as_raw()?;
        assert_eq!(empty.raw.resident_tensor_name_count, 0);
        assert!(empty.raw.resident_tensor_names.is_null());
        Ok(())
    }

    #[test]
    fn runtime_config_rejects_noncanonical_resident_tensor_names() {
        for names in [
            vec!["output.weight".into(), "blk.0.attn.weight".into()],
            vec!["output.weight".into(), "output.weight".into()],
            vec![String::new()],
        ] {
            let error = RuntimeConfig {
                resident_tensor_names: names,
                ..RuntimeConfig::default()
            }
            .as_raw()
            .expect_err("noncanonical resident tensor names must fail");
            assert!(error.to_string().contains("resident tensor name"));
        }
    }

    #[test]
    fn runtime_config_rejects_filtered_load_without_admission() {
        let error = RuntimeConfig {
            filter_tensors_on_load: true,
            resident_tensor_names: Vec::new(),
            ..RuntimeConfig::default()
        }
        .as_raw()
        .expect_err("filtered loads must carry an exact resident tensor set");
        assert!(
            error
                .to_string()
                .contains("explicit admitted resident tensor set")
        );
    }

    #[test]
    fn runtime_config_raw_inverts_repack_into_disable_repack() -> anyhow::Result<()> {
        let repack_enabled = RuntimeConfig {
            repack: true,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let repack_disabled = RuntimeConfig {
            repack: false,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert!(!repack_enabled.disable_repack);
        assert!(repack_disabled.disable_repack);
        assert_ne!(
            repack_enabled.disable_repack,
            repack_disabled.disable_repack
        );

        Ok(())
    }

    #[test]
    fn runtime_config_raw_defaults_op_offload_to_auto() -> anyhow::Result<()> {
        let raw = RuntimeConfig::default().as_raw()?.raw;
        assert_eq!(raw.op_offload, skippy_ffi::TRISTATE_AUTO);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_forces_op_offload_when_configured() -> anyhow::Result<()> {
        let enabled = RuntimeConfig {
            op_offload: Some(true),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let disabled = RuntimeConfig {
            op_offload: Some(false),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert_eq!(enabled.op_offload, skippy_ffi::TRISTATE_TRUE);
        assert_eq!(disabled.op_offload, skippy_ffi::TRISTATE_FALSE);
        assert_ne!(enabled.op_offload, disabled.op_offload);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_no_host_buffer_true_and_false_are_distinct() -> anyhow::Result<()> {
        let enabled = RuntimeConfig {
            no_host_buffer: true,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let disabled = RuntimeConfig {
            no_host_buffer: false,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert!(enabled.no_host_buffer);
        assert!(!disabled.no_host_buffer);
        assert_ne!(enabled.no_host_buffer, disabled.no_host_buffer);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_check_tensors_true_and_false_are_distinct() -> anyhow::Result<()> {
        let enabled = RuntimeConfig {
            check_tensors: true,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let disabled = RuntimeConfig {
            check_tensors: false,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert!(enabled.check_tensors);
        assert!(!disabled.check_tensors);
        assert_ne!(enabled.check_tensors, disabled.check_tensors);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_direct_io_true_and_false_are_distinct() -> anyhow::Result<()> {
        let enabled = RuntimeConfig {
            direct_io: true,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let disabled = RuntimeConfig {
            direct_io: false,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert!(enabled.use_direct_io);
        assert!(!disabled.use_direct_io);
        assert_ne!(enabled.use_direct_io, disabled.use_direct_io);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_main_gpu_override_and_unset_are_distinct() -> anyhow::Result<()> {
        let forced = RuntimeConfig {
            main_gpu: Some(2),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let unset = RuntimeConfig {
            main_gpu: None,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert!(forced.has_main_gpu_override);
        assert_eq!(forced.main_gpu, 2);
        assert!(!unset.has_main_gpu_override);
        assert_ne!(forced.has_main_gpu_override, unset.has_main_gpu_override);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_split_mode_variants_are_distinct() -> anyhow::Result<()> {
        let auto = RuntimeConfig {
            split_mode: SplitMode::Auto,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let none = RuntimeConfig {
            split_mode: SplitMode::None,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let layer = RuntimeConfig {
            split_mode: SplitMode::Layer,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let row = RuntimeConfig {
            split_mode: SplitMode::Row,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let tensor = RuntimeConfig {
            split_mode: SplitMode::Tensor,
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert_eq!(auto.split_mode, skippy_ffi::TRISTATE_AUTO);
        assert_eq!(none.split_mode, 0);
        assert_eq!(layer.split_mode, 1);
        assert_eq!(row.split_mode, 2);
        assert_eq!(tensor.split_mode, 3);
        assert_ne!(none.split_mode, layer.split_mode);
        assert_ne!(layer.split_mode, row.split_mode);
        assert_ne!(row.split_mode, tensor.split_mode);
        Ok(())
    }

    #[test]
    fn parse_cache_type_accepts_legacy_mesh_kv_defaults() -> anyhow::Result<()> {
        assert_eq!(parse_cache_type("f16")?, GGML_TYPE_F16);
        assert_eq!(parse_cache_type("q8_0")?, GGML_TYPE_Q8_0);
        assert_eq!(parse_cache_type("q4_0")?, GGML_TYPE_Q4_0);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_preserves_selected_backend_device() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            selected_backend_device: Some("MTL0".to_string()),
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;
        let device =
            unsafe { std::ffi::CStr::from_ptr(raw.raw.selected_backend_device).to_string_lossy() };

        assert_eq!(device, "MTL0");
        Ok(())
    }

    #[test]
    fn runtime_config_raw_defaults_glm_dsa_controls_to_disabled() -> anyhow::Result<()> {
        let raw = RuntimeConfig::default().as_raw()?.raw;

        assert!(!raw.use_mmap_prefetch);
        assert!(!raw.use_mmap_buffer);
        assert_eq!(raw.glm_dsa_policy_profile, 0);
        assert_eq!(raw.glm_dsa_policy_flags, 0);
        assert_eq!(raw.glm_dsa_short_prefill_max_tokens, 0);
        assert_eq!(raw.glm_dsa_direct_sparse_decode_max_top_k, 0);
        assert_eq!(raw.glm_dsa_dense_sparse_mask_max_bytes, 0);
        assert_eq!(raw.glm_dsa_compact_flash_min_kv, 0);
        assert_eq!(raw.mtp_source, RawMtpSource::Disabled);
        assert!(raw.selected_backend_device.is_null());
        Ok(())
    }

    #[test]
    fn runtime_config_raw_maps_glm_dsa_v1_policy() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            glm_dsa_policy: GlmDsaPolicy::V1,
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;

        assert_eq!(raw.raw.glm_dsa_policy_profile, 1);
        assert_eq!(raw.raw.glm_dsa_policy_flags, 0);
        assert_eq!(raw.raw.glm_dsa_short_prefill_max_tokens, 2048);
        assert_eq!(raw.raw.glm_dsa_direct_sparse_decode_max_top_k, 0);
        assert_eq!(
            raw.raw.glm_dsa_dense_sparse_mask_max_bytes,
            512 * 1024 * 1024
        );
        assert_eq!(raw.raw.glm_dsa_compact_flash_min_kv, 1);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_preserves_explicit_mtp_source() -> anyhow::Result<()> {
        for (source, expected) in [
            (MtpSource::Disabled, RawMtpSource::Disabled),
            (MtpSource::Integrated, RawMtpSource::Integrated),
            (MtpSource::External, RawMtpSource::External),
        ] {
            let raw = RuntimeConfig {
                mtp_source: source,
                ..RuntimeConfig::default()
            }
            .as_raw()?
            .raw;

            assert_eq!(raw.mtp_source, expected);
        }
        Ok(())
    }

    #[test]
    fn runtime_config_raw_uses_unified_kv_batch_default_for_all_lanes() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;

        assert_eq!(raw.raw.n_batch, SKIPPY_UNIFIED_KV_DEFAULT_N_BATCH as i32);
        assert_eq!(raw.raw.n_ubatch, LLAMA_SERVER_DEFAULT_N_UBATCH as i32);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_uses_unified_kv_batch_default_for_multi_lane() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            lane_count: 4,
            n_batch: None,
            n_ubatch: None,
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;

        assert_eq!(raw.raw.n_batch, SKIPPY_UNIFIED_KV_DEFAULT_N_BATCH as i32);
        assert_eq!(raw.raw.n_ubatch, LLAMA_SERVER_DEFAULT_N_UBATCH as i32);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_preserves_explicit_unified_kv_batch() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            lane_count: 4,
            n_batch: Some(2048),
            n_ubatch: Some(256),
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;

        assert_eq!(raw.raw.n_batch, 2048);
        assert_eq!(raw.raw.n_ubatch, 256);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_preserves_thread_counts_and_batch_defaults() -> anyhow::Result<()> {
        let config = RuntimeConfig {
            n_batch: None,
            n_ubatch: None,
            n_threads: Some(12),
            n_threads_batch: Some(6),
            ..RuntimeConfig::default()
        };

        let raw = config.as_raw()?;

        assert_eq!(raw.raw.n_batch, SKIPPY_UNIFIED_KV_DEFAULT_N_BATCH as i32);
        assert_eq!(raw.raw.n_ubatch, LLAMA_SERVER_DEFAULT_N_UBATCH as i32);
        assert_eq!(raw.raw.n_threads, 12);
        assert_eq!(raw.raw.n_threads_batch, 6);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_defaults_kv_session_controls_to_auto() -> anyhow::Result<()> {
        let raw = RuntimeConfig::default().as_raw()?.raw;

        assert_eq!(raw.kv_offload, skippy_ffi::TRISTATE_AUTO);
        assert_eq!(raw.kv_unified, skippy_ffi::TRISTATE_AUTO);
        assert_eq!(raw.swa_full, skippy_ffi::TRISTATE_AUTO);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_forces_kv_offload_when_configured() -> anyhow::Result<()> {
        let enabled = RuntimeConfig {
            kv_offload: Some(true),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let disabled = RuntimeConfig {
            kv_offload: Some(false),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert_eq!(enabled.kv_offload, skippy_ffi::TRISTATE_TRUE);
        assert_eq!(disabled.kv_offload, skippy_ffi::TRISTATE_FALSE);
        assert_ne!(enabled.kv_offload, disabled.kv_offload);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_forces_kv_unified_when_configured() -> anyhow::Result<()> {
        let enabled = RuntimeConfig {
            kv_unified: Some(true),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let disabled = RuntimeConfig {
            kv_unified: Some(false),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert_eq!(enabled.kv_unified, skippy_ffi::TRISTATE_TRUE);
        assert_eq!(disabled.kv_unified, skippy_ffi::TRISTATE_FALSE);
        assert_ne!(enabled.kv_unified, disabled.kv_unified);
        Ok(())
    }

    #[test]
    fn runtime_config_raw_forces_swa_full_when_configured() -> anyhow::Result<()> {
        let enabled = RuntimeConfig {
            swa_full: Some(true),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;
        let disabled = RuntimeConfig {
            swa_full: Some(false),
            ..RuntimeConfig::default()
        }
        .as_raw()?
        .raw;

        assert_eq!(enabled.swa_full, skippy_ffi::TRISTATE_TRUE);
        assert_eq!(disabled.swa_full, skippy_ffi::TRISTATE_FALSE);
        assert_ne!(enabled.swa_full, disabled.swa_full);
        Ok(())
    }
}
