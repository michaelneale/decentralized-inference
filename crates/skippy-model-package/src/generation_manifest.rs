use serde::Deserialize;

pub(crate) const GLM_DSA_POLICY_PROFILE: &str = "glm-dsa-v1";
pub(crate) const GLM_DSA_POLICY_DECODE: &str = "compact-flash";
pub(crate) const GLM_DSA_POLICY_SHORT_PREFILL: &str = "dense";
pub(crate) const GLM_DSA_POLICY_LONG_PREFILL: &str = "sparse-chunked";
pub(crate) const GLM_DSA_POLICY_VERIFY: &str = "auto";
pub(crate) const GLM_DSA_POLICY_INDEXSHARE: &str = "required";
pub(crate) const GLM_DSA_POLICY_SELECTED_ROW_FLASH: &str = "evidence-gated";
pub(crate) const GLM_DSA_SHORT_PREFILL_MAX_TOKENS: u32 = 2048;
pub(crate) const GLM_DSA_DIRECT_SPARSE_DECODE_MAX_TOP_K: u32 = 256;
pub(crate) const GLM_DSA_COMPACT_FLASH_MIN_KV: u32 = 1;
pub(crate) const GLM_DSA_DENSE_MASK_MAX_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub(crate) struct PackageGeneration<S = serde_json::Value> {
    #[serde(default)]
    pub(crate) policy: Option<PackageGenerationPolicy>,
    #[serde(default)]
    pub(crate) thresholds: Option<PackageGenerationThresholds>,
    #[allow(dead_code)]
    pub(crate) speculative_decoding: Option<S>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PackageGenerationPolicy {
    pub(crate) profile: String,
    pub(crate) decode: String,
    pub(crate) short_prefill: String,
    pub(crate) long_prefill: String,
    pub(crate) verify: String,
    #[serde(default)]
    pub(crate) indexshare: Option<String>,
    #[serde(default)]
    pub(crate) experimental: Option<PackageGenerationExperimentalPolicy>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PackageGenerationExperimentalPolicy {
    #[serde(default)]
    pub(crate) selected_row_flash: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PackageGenerationThresholds {
    #[serde(default)]
    pub(crate) short_prefill_max_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) direct_sparse_decode_max_top_k: Option<u32>,
    #[serde(default)]
    pub(crate) compact_flash_min_kv: Option<u32>,
    #[serde(default)]
    pub(crate) dense_mask_max_bytes: Option<u64>,
}
