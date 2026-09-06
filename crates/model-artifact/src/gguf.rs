use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

mod kv_cache;
mod split_bundle;

pub use kv_cache::{GgufKvCacheQuant, GgufKvCacheType};
pub use split_bundle::scan_gguf_bundle_total_parameters;

const MAX_GGUF_STRING_BYTES: u64 = 1_000_000;
const MAX_GGUF_ARRAY_ELEMENTS: u64 = 1_000_000;
const MAX_GGUF_ARRAY_DEPTH: u32 = 64;
const MAX_GGUF_TENSOR_DIMS: u32 = 8;
const MAX_GGUF_HEADER_KV_COUNT: usize = 1_000_000;
const MAX_GGUF_TENSOR_COUNT: usize = 1_000_000;

/// GGUF value types (matching gguf.h enum).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }

    fn fixed_size(self) -> Option<usize> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

fn read_u32(f: &mut std::fs::File) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(f: &mut std::fs::File) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32(f: &mut std::fs::File) -> std::io::Result<i32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64(f: &mut std::fs::File) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_gguf_header_count(
    f: &mut std::fs::File,
    max: usize,
    label: &str,
) -> std::io::Result<usize> {
    let value = read_i64(f)?;
    let count = usize::try_from(value).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("negative {label}"))
    })?;
    if count > max {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} too large"),
        ));
    }
    Ok(count)
}

fn read_bounded_len(f: &mut std::fs::File, max: u64, label: &str) -> std::io::Result<usize> {
    let len = read_u64(f)?;
    if len > max {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} too long"),
        ));
    }
    usize::try_from(len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{label} too large"),
        )
    })
}

fn read_gguf_string(f: &mut std::fs::File) -> std::io::Result<String> {
    let buf = read_gguf_bytes(f, "string")?;
    String::from_utf8(buf).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid UTF-8 in GGUF string",
        )
    })
}

fn read_gguf_bytes(f: &mut std::fs::File, label: &str) -> std::io::Result<Vec<u8>> {
    let len = read_bounded_len(f, MAX_GGUF_STRING_BYTES, label)?;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    Ok(buf)
}

fn skip_gguf_value(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<()> {
    skip_gguf_value_with_depth(f, typ, 0)
}

fn skip_gguf_value_with_depth(
    f: &mut std::fs::File,
    typ: GgufType,
    depth: u32,
) -> std::io::Result<()> {
    match typ {
        GgufType::String => {
            let _ = read_gguf_string(f)?;
        }
        GgufType::Array => {
            if depth >= MAX_GGUF_ARRAY_DEPTH {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "GGUF nesting too deep",
                ));
            }
            let elem_type = GgufType::from_u32(read_u32(f)?).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "bad array type")
            })?;
            let count = read_bounded_len(f, MAX_GGUF_ARRAY_ELEMENTS, "array")?;
            for _ in 0..count {
                skip_gguf_value_with_depth(f, elem_type, depth + 1)?;
            }
        }
        other => {
            let size = other.fixed_size().unwrap_or(0);
            f.seek(SeekFrom::Current(size as i64))?;
        }
    }
    Ok(())
}

fn read_gguf_value_as_u32(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<u32>> {
    match typ {
        GgufType::Uint32 => Ok(Some(read_u32(f)?)),
        GgufType::Int32 => {
            let value = read_i32(f)?;
            let value = u32::try_from(value).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "negative Int32 where unsigned GGUF value was expected",
                )
            })?;
            Ok(Some(value))
        }
        GgufType::Uint16 => {
            let mut buf = [0u8; 2];
            f.read_exact(&mut buf)?;
            Ok(Some(u16::from_le_bytes(buf) as u32))
        }
        GgufType::Uint8 => {
            let mut buf = [0u8; 1];
            f.read_exact(&mut buf)?;
            Ok(Some(buf[0] as u32))
        }
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

fn read_gguf_value_as_u32_list(
    f: &mut std::fs::File,
    typ: GgufType,
) -> std::io::Result<Option<Vec<u32>>> {
    if typ != GgufType::Array {
        return Ok(read_gguf_value_as_u32(f, typ)?.map(|value| vec![value]));
    }

    let elem_type = GgufType::from_u32(read_u32(f)?)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad array type"))?;
    let count = read_bounded_len(f, MAX_GGUF_ARRAY_ELEMENTS, "array")?;
    let mut values = Vec::with_capacity(count);
    let mut supported = true;
    for _ in 0..count {
        match read_gguf_value_as_u32(f, elem_type)? {
            Some(value) if supported => values.push(value),
            Some(_) => {}
            None => supported = false,
        }
    }
    Ok(supported.then_some(values))
}

fn read_gguf_value_as_f32(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<f32>> {
    match typ {
        GgufType::Float32 => {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            Ok(Some(f32::from_le_bytes(buf)))
        }
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

fn read_gguf_value_as_bool(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Option<bool>> {
    if typ == GgufType::Bool {
        let mut value = [0u8; 1];
        f.read_exact(&mut value)?;
        return Ok(Some(value[0] != 0));
    }
    Ok(read_gguf_value_as_u32(f, typ)?.map(|value| value != 0))
}

fn read_gguf_value_as_bool_list(
    f: &mut std::fs::File,
    typ: GgufType,
) -> std::io::Result<Option<Vec<bool>>> {
    if typ != GgufType::Array {
        return Ok(read_gguf_value_as_bool(f, typ)?.map(|value| vec![value]));
    }

    let elem_type = GgufType::from_u32(read_u32(f)?)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad array type"))?;
    let count = read_bounded_len(f, MAX_GGUF_ARRAY_ELEMENTS, "array")?;
    let mut values = Vec::with_capacity(count);
    let mut supported = true;
    for _ in 0..count {
        match read_gguf_value_as_bool(f, elem_type)? {
            Some(value) if supported => values.push(value),
            Some(_) => {}
            None => supported = false,
        }
    }
    Ok(supported.then_some(values))
}

fn read_gguf_value_as_string_opt(
    f: &mut std::fs::File,
    typ: GgufType,
) -> std::io::Result<Option<String>> {
    match typ {
        GgufType::String => Ok(Some(read_gguf_string(f)?)),
        _ => {
            skip_gguf_value(f, typ)?;
            Ok(None)
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GgufCompactMeta {
    pub architecture: String,
    pub parameter_size: Option<String>,
    pub context_length: u32,
    pub vocab_size: u32,
    pub embedding_size: u32,
    pub head_count: u32,
    pub kv_head_count: u32,
    pub kv_head_counts: Vec<u32>,
    pub layer_count: u32,
    pub feed_forward_length: u32,
    pub key_length: u32,
    pub value_length: u32,
    /// Per-head key width used by sliding-window-attention layers
    /// (`<arch>.attention.key_length_swa`), when the GGUF carries one.
    /// llama.cpp's `n_embd_head_k(il)` returns this width for SWA layers and
    /// the full width otherwise, and the quantised-KV block-alignment check
    /// runs per layer — so both widths must be aligned for a quantised
    /// cache to load. `0` means the key was absent (llama.cpp then uses the
    /// full width for SWA layers too).
    pub key_length_swa: u32,
    /// Per-head value width used by SWA layers
    /// (`<arch>.attention.value_length_swa`); see `key_length_swa`.
    pub value_length_swa: u32,
    pub kv_lora_rank: u32,
    pub tokenizer_model_name: String,
    pub rope_scale: f32,
    pub rope_freq_base: f32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub nextn_predict_layers: u32,
    pub ssm_conv_kernel: u32,
    pub ssm_inner_size: u32,
    pub ssm_state_size: u32,
    pub ssm_group_count: u32,
    pub recurrent_layers: Vec<bool>,
    pub full_attention_interval: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GgufProjectorMeta {
    pub has_vision_encoder: Option<bool>,
    pub has_audio_encoder: Option<bool>,
}

impl GgufCompactMeta {
    pub fn effective_kv_head_count(&self) -> Option<u32> {
        if self.kv_head_count > 0 {
            Some(self.kv_head_count)
        } else if let Some(kv_head_count) = self.kv_head_counts.iter().copied().max()
            && kv_head_count > 0
        {
            Some(kv_head_count)
        } else if self.head_count > 0 {
            Some(self.head_count)
        } else {
            None
        }
    }

    /// Whether metadata proves that every layer is recurrent and allocates no
    /// attention KV cache.
    ///
    /// Zero head counts alone are not enough: malformed dense GGUF metadata
    /// can omit the same fields. Require positive recurrent state allocation,
    /// a complete all-recurrent layer mask, and no residual attention widths
    /// or MLA rank before topology planning may price KV bytes as zero.
    pub fn has_only_recurrent_layers_without_kv_cache(&self) -> bool {
        let layer_count = self.layer_count as usize;
        if layer_count == 0
            || self.ssm_inner_size == 0
            || (self.ssm_state_size == 0 && self.ssm_conv_kernel <= 1)
            || self.head_count != 0
            || self.kv_head_count != 0
            || self.kv_head_counts.iter().any(|count| *count != 0)
            || self.key_length != 0
            || self.value_length != 0
            || self.kv_lora_rank != 0
        {
            return false;
        }

        let recurrent_layers = self.recurrent_layer_mask();
        let recurrent_bytes = self.recurrent_bytes_per_configured_lane_by_layer();
        recurrent_layers.len() == layer_count
            && recurrent_layers.into_iter().all(|recurrent| recurrent)
            && recurrent_bytes.len() == layer_count
            && recurrent_bytes.into_iter().all(|bytes| bytes > 0)
    }

    /// F32 recurrent-state bytes allocated for one native sequence, per layer.
    ///
    /// Skippy enables two rollback snapshots for recurrent/hybrid runtimes, so
    /// each live native sequence owns three state planes (current + two
    /// snapshots). Attention-only layers report zero.
    pub fn recurrent_bytes_per_native_sequence_by_layer(&self) -> Vec<u64> {
        const F32_BYTES: u64 = 4;
        const RECURRENT_STATE_PLANES: u64 = 3;

        let conv_width = u64::from(self.ssm_conv_kernel.saturating_sub(1));
        let inner = u64::from(self.ssm_inner_size);
        let state = u64::from(self.ssm_state_size);
        let groups = u64::from(self.ssm_group_count);
        let rolling = conv_width
            .saturating_mul(inner.saturating_add(groups.saturating_mul(state).saturating_mul(2)));
        let state = state.saturating_mul(inner);
        let bytes = rolling
            .saturating_add(state)
            .saturating_mul(F32_BYTES)
            .saturating_mul(RECURRENT_STATE_PLANES);

        self.recurrent_layer_mask()
            .into_iter()
            .map(|recurrent| if recurrent { bytes } else { 0 })
            .collect()
    }

    /// Recurrent bytes charged by topology planning for one configured lane.
    /// Native recurrent runtimes reserve two sequence slots per lane.
    pub fn recurrent_bytes_per_configured_lane_by_layer(&self) -> Vec<u64> {
        const NATIVE_SEQUENCES_PER_LANE: u64 = 2;
        self.recurrent_bytes_per_native_sequence_by_layer()
            .into_iter()
            .map(|bytes| bytes.saturating_mul(NATIVE_SEQUENCES_PER_LANE))
            .collect()
    }

    fn recurrent_layer_mask(&self) -> Vec<bool> {
        let layer_count = self.layer_count as usize;
        if self.recurrent_layers.len() == layer_count {
            return self.recurrent_layers.clone();
        }

        let has_recurrent_state = self.ssm_conv_kernel > 0
            || self.ssm_inner_size > 0
            || self.ssm_state_size > 0
            || self.ssm_group_count > 0;
        if !has_recurrent_state {
            return vec![false; layer_count];
        }

        // Hybrid GGUFs can expose a per-layer KV-head count. Recurrent layers
        // have no attention KV heads, so this is an architecture-independent
        // state mask.
        if self.kv_head_counts.len() == layer_count {
            let mask = self
                .kv_head_counts
                .iter()
                .map(|count| *count == 0)
                .collect::<Vec<_>>();
            if mask.iter().any(|recurrent| *recurrent) {
                return mask;
            }
        }

        // Some models expose a generic full-attention cadence rather than an
        // explicit mask. Mirror llama.cpp's `%s.full_attention_interval`
        // contract: layers are one-based at the cadence boundary, so every
        // `interval`th layer is full attention and the intervening layers are
        // recurrent. This is metadata-driven and does not name model families.
        if self.full_attention_interval > 0 {
            let interval = self.full_attention_interval as usize;
            return (0..layer_count)
                .map(|layer| (layer + 1) % interval != 0)
                .collect();
        }

        // With recurrent state dimensions but no layer-level discriminator,
        // charge every layer. Overestimating resource use is safer than
        // silently omitting native recurrent state.
        vec![true; layer_count]
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GgufTensorByteProfile {
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub full_model_bytes: u64,
    pub base_resident_bytes: u64,
    pub expert_tensor_bytes: u64,
    pub file_overhead_bytes: u64,
}

/// The tokenizer vocabulary exposed by a source GGUF header. Non-control
/// entries intentionally retain their raw bytes: many GGUF vocabularies are
/// not valid UTF-8 and must never be round-tripped through text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufTokenizerInventory {
    pub tokens: Vec<GgufTokenizerToken>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufTokenizerToken {
    pub raw: Vec<u8>,
    pub is_control: bool,
}

#[derive(Clone, Debug)]
struct GgufTensorInfo {
    name: String,
    offset: u64,
}

/// Scan a GGUF file header and return compact structural metadata.
/// Reads only the KV section, never tensor data. Returns None on any parse failure.
struct GgufHeader {
    file: std::fs::File,
    n_tensors: usize,
    n_kv: usize,
}

/// Opens `path`, validates the GGUF magic/version, and reads the tensor/KV
/// header counts shared by every GGUF scan entry point below.
fn open_gguf_header(path: &Path) -> Option<GgufHeader> {
    let mut f = std::fs::File::open(path).ok()?;

    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = read_u32(&mut f).ok()?;
    if version < 2 {
        return None;
    }

    let n_tensors = read_gguf_header_count(&mut f, MAX_GGUF_TENSOR_COUNT, "tensor count").ok()?;
    let n_kv = read_gguf_header_count(&mut f, MAX_GGUF_HEADER_KV_COUNT, "KV count").ok()?;

    Some(GgufHeader {
        file: f,
        n_tensors,
        n_kv,
    })
}

/// Skips every KV pair without inspecting keys/values, for scans that only
/// need the tensor-info table.
fn skip_all_kv_pairs(f: &mut std::fs::File, n_kv: usize) -> Option<()> {
    for _ in 0..n_kv {
        let _key = read_gguf_string(f).ok()?;
        let vtype = GgufType::from_u32(read_u32(f).ok()?)?;
        skip_gguf_value(f, vtype).ok()?;
    }
    Some(())
}

pub fn scan_gguf_compact_meta(path: &Path) -> Option<GgufCompactMeta> {
    let GgufHeader {
        file: mut f, n_kv, ..
    } = open_gguf_header(path)?;

    let mut meta = GgufCompactMeta::default();
    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let vtype = GgufType::from_u32(read_u32(&mut f).ok()?)?;

        if key == "general.architecture" {
            meta.architecture = read_gguf_value_as_string_opt(&mut f, vtype).ok()??;
        } else if key == "general.size_label" {
            meta.parameter_size = read_gguf_value_as_string_opt(&mut f, vtype).ok()?;
        } else if key == "tokenizer.ggml.model" {
            meta.tokenizer_model_name = read_gguf_value_as_string_opt(&mut f, vtype).ok()??;
        } else if key.ends_with(".context_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.context_length = v;
            }
        } else if key.ends_with(".embedding_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.embedding_size = v;
            }
        } else if key.ends_with(".head_count") && !key.ends_with("_kv") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.head_count = v;
            }
        } else if key.ends_with(".attention.head_count_kv") {
            if let Ok(Some(values)) = read_gguf_value_as_u32_list(&mut f, vtype) {
                if values.len() == 1 {
                    meta.kv_head_count = values[0];
                } else {
                    meta.kv_head_counts = values;
                }
            }
        } else if key.ends_with(".block_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.layer_count = v;
            }
        } else if key.ends_with(".feed_forward_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.feed_forward_length = v;
            }
        } else if key.ends_with(".attention.key_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.key_length = v;
            }
        } else if key.ends_with(".attention.key_length_swa") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.key_length_swa = v;
            }
        } else if key.ends_with(".attention.value_length") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.value_length = v;
            }
        } else if key.ends_with(".attention.value_length_swa") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.value_length_swa = v;
            }
        } else if key.ends_with(".attention.kv_lora_rank") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.kv_lora_rank = v;
            }
        } else if key.ends_with(".rope.scale") {
            if let Ok(Some(v)) = read_gguf_value_as_f32(&mut f, vtype) {
                meta.rope_scale = v;
            }
        } else if key.ends_with(".rope.freq_base") {
            if let Ok(Some(v)) = read_gguf_value_as_f32(&mut f, vtype) {
                meta.rope_freq_base = v;
            }
        } else if key.ends_with(".vocab_size") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.vocab_size = v;
            }
        } else if key.ends_with(".expert_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.expert_count = v;
            }
        } else if key.ends_with(".expert_used_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.expert_used_count = v;
            }
        } else if key.ends_with(".nextn_predict_layers") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.nextn_predict_layers = v;
            }
        } else if key.ends_with(".ssm.conv_kernel") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.ssm_conv_kernel = v;
            }
        } else if key.ends_with(".ssm.inner_size") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.ssm_inner_size = v;
            }
        } else if key.ends_with(".ssm.state_size") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.ssm_state_size = v;
            }
        } else if key.ends_with(".ssm.group_count") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.ssm_group_count = v;
            }
        } else if key.ends_with(".attention.recurrent_layers") {
            if let Ok(Some(values)) = read_gguf_value_as_bool_list(&mut f, vtype) {
                meta.recurrent_layers = values;
            }
        } else if key.ends_with(".full_attention_interval") {
            if let Ok(Some(v)) = read_gguf_value_as_u32(&mut f, vtype) {
                meta.full_attention_interval = v;
            }
        } else {
            skip_gguf_value(&mut f, vtype).ok()?;
        }
    }

    let derived_key_length = (meta.key_length == 0 && meta.head_count > 0)
        .then(|| meta.embedding_size.checked_div(meta.head_count))
        .flatten();
    if let Some(key_length) = derived_key_length {
        meta.key_length = key_length;
    }
    let derived_value_length = (meta.value_length == 0)
        .then(|| {
            meta.effective_kv_head_count()
                .and_then(|effective_kv| meta.embedding_size.checked_div(effective_kv))
        })
        .flatten();
    if let Some(value_length) = derived_value_length {
        meta.value_length = value_length;
    }

    Some(meta)
}

/// Scan the modality flags stored in a multimodal projector GGUF.
pub fn scan_gguf_projector_meta(path: &Path) -> Option<GgufProjectorMeta> {
    let GgufHeader {
        file: mut f, n_kv, ..
    } = open_gguf_header(path)?;
    let mut meta = GgufProjectorMeta::default();
    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let value_type = GgufType::from_u32(read_u32(&mut f).ok()?)?;
        match key.as_str() {
            "clip.has_vision_encoder" => {
                meta.has_vision_encoder = read_gguf_value_as_bool(&mut f, value_type).ok()?;
            }
            "clip.has_audio_encoder" => {
                meta.has_audio_encoder = read_gguf_value_as_bool(&mut f, value_type).ok()?;
            }
            _ => skip_gguf_value(&mut f, value_type).ok()?,
        }
    }
    Some(meta)
}

fn read_string_array(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Vec<Vec<u8>>> {
    if typ != GgufType::Array {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "tokenizer tokens must be a GGUF array",
        ));
    }
    if GgufType::from_u32(read_u32(f)?) != Some(GgufType::String) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "tokenizer tokens must be an array of strings",
        ));
    }
    let count = read_bounded_len(f, MAX_GGUF_ARRAY_ELEMENTS, "tokenizer token array")?;
    let mut tokens = Vec::new();
    tokens.try_reserve(count).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "tokenizer token array requires too much memory",
        )
    })?;
    for _ in 0..count {
        tokens.push(read_gguf_bytes(f, "tokenizer token")?);
    }
    Ok(tokens)
}

fn read_token_type_array(f: &mut std::fs::File, typ: GgufType) -> std::io::Result<Vec<u32>> {
    if typ != GgufType::Array {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "tokenizer token types must be a GGUF array",
        ));
    }
    let element_type = GgufType::from_u32(read_u32(f)?).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "bad tokenizer token type")
    })?;
    let count = read_bounded_len(f, MAX_GGUF_ARRAY_ELEMENTS, "tokenizer token type array")?;
    let mut types = Vec::new();
    types.try_reserve(count).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "tokenizer token type array requires too much memory",
        )
    })?;
    for _ in 0..count {
        let value = read_gguf_value_as_u32(f, element_type)?.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported tokenizer token type element",
            )
        })?;
        types.push(value);
    }
    Ok(types)
}

/// Reads a source GGUF's tokenizer vocabulary and its control-token markers.
/// This scans metadata only and rejects malformed or incomplete inventories.
pub fn scan_gguf_tokenizer_inventory(path: &Path) -> Option<GgufTokenizerInventory> {
    let GgufHeader {
        file: mut f, n_kv, ..
    } = open_gguf_header(path)?;
    let mut raw_tokens = None;
    let mut token_types = None;

    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let typ = GgufType::from_u32(read_u32(&mut f).ok()?)?;
        match key.as_str() {
            "tokenizer.ggml.tokens" => raw_tokens = Some(read_string_array(&mut f, typ).ok()?),
            "tokenizer.ggml.token_type" => {
                token_types = Some(read_token_type_array(&mut f, typ).ok()?);
            }
            _ => skip_gguf_value(&mut f, typ).ok()?,
        }
    }

    let raw_tokens = raw_tokens?;
    let token_types = token_types.unwrap_or_else(|| vec![0; raw_tokens.len()]);
    if raw_tokens.is_empty() || raw_tokens.len() != token_types.len() {
        return None;
    }
    Some(GgufTokenizerInventory {
        tokens: raw_tokens
            .into_iter()
            .zip(token_types)
            // GGML's stable token type value for an explicit control boundary.
            .map(|(raw, token_type)| GgufTokenizerToken {
                raw,
                is_control: token_type == 3,
            })
            .collect(),
    })
}

fn align_offset(value: u64, alignment: u32) -> u64 {
    let alignment = u64::from(alignment.max(1));
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

fn read_tensor_infos(
    f: &mut std::fs::File,
    n_tensors: usize,
) -> std::io::Result<Vec<GgufTensorInfo>> {
    let mut tensors = Vec::new();
    tensors.try_reserve(n_tensors).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "GGUF tensor count requires too much memory",
        )
    })?;
    for _ in 0..n_tensors {
        let name = read_gguf_string(f)?;
        let n_dims = read_u32(f)?;
        if n_dims > MAX_GGUF_TENSOR_DIMS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "too many GGUF tensor dimensions",
            ));
        }
        for _ in 0..n_dims {
            let _ = read_u64(f)?;
        }
        let _ = read_u32(f)?;
        let offset = read_u64(f)?;
        tensors.push(GgufTensorInfo { name, offset });
    }
    Ok(tensors)
}

/// Sum the element counts of every tensor in a GGUF file → total stored
/// parameter count. Reads only the header and tensor-info table (dimensions),
/// never tensor data.
///
/// This is the authoritative model size: the exact number of stored weights,
/// independent of the file name. Name parsing (`NNb` in the model id) is a
/// brittle fallback — aliases and fine-tunes need not encode a size, names can
/// carry unrelated digits, and an unparseable name must read as *unknown*, not
/// as a guessed tier. Returns `None` on any parse failure (→ unknown).
pub fn scan_gguf_total_parameters(path: &Path) -> Option<u64> {
    let GgufHeader {
        file: mut f,
        n_tensors,
        n_kv,
    } = open_gguf_header(path)?;

    skip_all_kv_pairs(&mut f, n_kv)?;

    let mut total: u64 = 0;
    for _ in 0..n_tensors {
        let _name = read_gguf_string(&mut f).ok()?;
        let n_dims = read_u32(&mut f).ok()?;
        if n_dims > MAX_GGUF_TENSOR_DIMS {
            return None;
        }
        let mut elements: u64 = 1;
        for _ in 0..n_dims {
            let dim = read_u64(&mut f).ok()?;
            elements = elements.checked_mul(dim)?;
        }
        let _ggml_type = read_u32(&mut f).ok()?;
        let _offset = read_u64(&mut f).ok()?;
        total = total.checked_add(elements)?;
    }
    Some(total)
}

/// Scan GGUF tensor names and return whether any tensor matches the predicate.
/// Reads only the header and tensor-info table, never tensor data.
pub fn scan_gguf_tensor_names_any(
    path: &Path,
    mut matches: impl FnMut(&str) -> bool,
) -> Option<bool> {
    let GgufHeader {
        file: mut f,
        n_tensors,
        n_kv,
    } = open_gguf_header(path)?;

    skip_all_kv_pairs(&mut f, n_kv)?;

    for _ in 0..n_tensors {
        let name = read_gguf_string(&mut f).ok()?;
        if matches(&name) {
            return Some(true);
        }
        let n_dims = read_u32(&mut f).ok()?;
        if n_dims > MAX_GGUF_TENSOR_DIMS {
            return None;
        }
        for _ in 0..n_dims {
            let _ = read_u64(&mut f).ok()?;
        }
        let _ggml_type = read_u32(&mut f).ok()?;
        let _offset = read_u64(&mut f).ok()?;
    }

    Some(false)
}

fn is_expert_partitioned_tensor(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if lower.contains("shared_expert") || lower.contains("sharedexpert") || lower.contains("shexp")
    {
        return false;
    }

    lower.contains("ffn_gate_exps")
        || lower.contains("ffn_up_exps")
        || lower.contains("ffn_down_exps")
        || lower.contains("exp_probs")
        || lower.contains(".expert")
        || lower.contains("_expert")
}

/// Scan GGUF tensor metadata and estimate which bytes are always resident versus
/// expert-partitioned. Reads only the header and tensor-info tables.
pub fn scan_gguf_tensor_byte_profile(path: &Path) -> Option<GgufTensorByteProfile> {
    let GgufHeader {
        file: mut f,
        n_tensors,
        n_kv,
    } = open_gguf_header(path)?;
    let file_len = f.metadata().ok()?.len();

    let mut expert_count = 0u32;
    let mut expert_used_count = 0u32;
    let mut alignment = 32u32;

    for _ in 0..n_kv {
        let key = read_gguf_string(&mut f).ok()?;
        let vtype = GgufType::from_u32(read_u32(&mut f).ok()?)?;

        if key == "general.alignment" {
            if let Ok(Some(value)) = read_gguf_value_as_u32(&mut f, vtype) {
                alignment = value.max(1);
            }
        } else if key.ends_with(".expert_count") {
            if let Ok(Some(value)) = read_gguf_value_as_u32(&mut f, vtype) {
                expert_count = value;
            }
        } else if key.ends_with(".expert_used_count") {
            if let Ok(Some(value)) = read_gguf_value_as_u32(&mut f, vtype) {
                expert_used_count = value;
            }
        } else {
            skip_gguf_value(&mut f, vtype).ok()?;
        }
    }

    let mut tensors = read_tensor_infos(&mut f, n_tensors).ok()?;
    if tensors.is_empty() {
        return Some(GgufTensorByteProfile {
            expert_count,
            expert_used_count,
            full_model_bytes: file_len,
            base_resident_bytes: 0,
            expert_tensor_bytes: 0,
            file_overhead_bytes: file_len,
        });
    }

    let tensor_info_end = f.stream_position().ok()?;
    let data_start = align_offset(tensor_info_end, alignment);
    if data_start > file_len {
        return None;
    }
    let data_len = file_len - data_start;

    tensors.sort_by_key(|tensor| tensor.offset);
    if tensors.first()?.offset > data_len {
        return None;
    }

    let mut base_resident_bytes = 0u64;
    let mut expert_tensor_bytes = 0u64;
    for (index, tensor) in tensors.iter().enumerate() {
        let next_offset = tensors
            .get(index + 1)
            .map(|next| next.offset)
            .unwrap_or(data_len);
        if next_offset < tensor.offset || next_offset > data_len {
            return None;
        }
        let tensor_bytes = next_offset - tensor.offset;
        if is_expert_partitioned_tensor(&tensor.name) {
            expert_tensor_bytes = expert_tensor_bytes.saturating_add(tensor_bytes);
        } else {
            base_resident_bytes = base_resident_bytes.saturating_add(tensor_bytes);
        }
    }

    let file_overhead_bytes = file_len.saturating_sub(base_resident_bytes + expert_tensor_bytes);
    Some(GgufTensorByteProfile {
        expert_count,
        expert_used_count,
        full_model_bytes: file_len,
        base_resident_bytes,
        expert_tensor_bytes,
        file_overhead_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_file_path(prefix: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{unique}.gguf"))
    }

    fn write_bytes(prefix: &str, bytes: &[u8]) -> PathBuf {
        let path = temp_file_path(prefix);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        file.flush().unwrap();
        path
    }

    fn push_array_header(bytes: &mut Vec<u8>, elem_type: GgufType, count: u64) {
        bytes.extend_from_slice(&(elem_type as u32).to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
    }

    fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_gguf_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value);
    }

    fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&(GgufType::Uint32 as u32).to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_bool_kv(bytes: &mut Vec<u8>, key: &str, value: bool) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&(GgufType::Bool as u32).to_le_bytes());
        bytes.push(u8::from(value));
    }

    fn push_bool_array_kv(bytes: &mut Vec<u8>, key: &str, values: &[bool]) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&(GgufType::Array as u32).to_le_bytes());
        push_array_header(bytes, GgufType::Bool, values.len() as u64);
        bytes.extend(values.iter().map(|value| u8::from(*value)));
    }

    fn push_u32_array_kv(bytes: &mut Vec<u8>, key: &str, values: &[u32]) {
        push_gguf_string(bytes, key);
        bytes.extend_from_slice(&(GgufType::Array as u32).to_le_bytes());
        push_array_header(bytes, GgufType::Uint32, values.len() as u64);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    #[test]
    fn compact_meta_prices_recurrent_state_per_layer_and_native_lane_multiplier() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&7i64.to_le_bytes());
        push_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&(GgufType::String as u32).to_le_bytes());
        push_gguf_string(&mut bytes, "qwen3next");
        push_u32_kv(&mut bytes, "qwen3next.block_count", 4);
        push_u32_kv(&mut bytes, "qwen3next.ssm.conv_kernel", 4);
        push_u32_kv(&mut bytes, "qwen3next.ssm.inner_size", 2048);
        push_u32_kv(&mut bytes, "qwen3next.ssm.state_size", 128);
        push_u32_kv(&mut bytes, "qwen3next.ssm.group_count", 16);
        push_bool_array_kv(
            &mut bytes,
            "qwen3next.attention.recurrent_layers",
            &[true, true, true, false],
        );

        let path = write_bytes("model-artifact-gguf-recurrent", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse recurrent metadata");
        let native = meta.recurrent_bytes_per_native_sequence_by_layer();
        let base_elements = 3u64 * (2048 + 2 * 16 * 128) + 128 * 2048;
        let expected = base_elements * 4 * 3;
        assert_eq!(native, vec![expected, expected, expected, 0]);
        assert_eq!(
            meta.recurrent_bytes_per_configured_lane_by_layer(),
            vec![expected * 2, expected * 2, expected * 2, 0]
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn recurrent_mask_uses_per_layer_metadata_for_unknown_architecture() {
        let meta = GgufCompactMeta {
            architecture: "future_hybrid_arch".to_string(),
            layer_count: 4,
            kv_head_counts: vec![0, 8, 0, 8],
            ssm_inner_size: 128,
            ..Default::default()
        };

        assert_eq!(meta.recurrent_layer_mask(), vec![true, false, true, false]);
    }

    #[test]
    fn recurrent_mask_does_not_treat_scalar_kv_heads_as_a_layer_discriminator() {
        let composite = GgufCompactMeta {
            architecture: "future_composite_arch".to_string(),
            layer_count: 4,
            kv_head_count: 8,
            kv_head_counts: vec![8; 4],
            ssm_state_size: 64,
            ..Default::default()
        };

        assert_eq!(composite.recurrent_layer_mask(), vec![true; 4]);
    }

    #[test]
    fn recurrent_mask_fails_safe_without_family_knowledge() {
        let recurrent = GgufCompactMeta {
            architecture: "future_recurrent_arch".to_string(),
            layer_count: 3,
            ssm_state_size: 64,
            ..Default::default()
        };
        let dense = GgufCompactMeta {
            architecture: "future_dense_arch".to_string(),
            layer_count: 3,
            ..Default::default()
        };

        assert_eq!(recurrent.recurrent_layer_mask(), vec![true; 3]);
        assert_eq!(dense.recurrent_layer_mask(), vec![false; 3]);
    }

    #[test]
    fn pure_recurrent_metadata_proves_zero_kv_without_family_knowledge() {
        let pure_recurrent = GgufCompactMeta {
            architecture: "future_recurrent_arch".to_string(),
            layer_count: 3,
            ssm_conv_kernel: 4,
            ssm_inner_size: 1536,
            ssm_state_size: 16,
            ..Default::default()
        };
        let dense_missing_heads = GgufCompactMeta {
            architecture: "future_dense_arch".to_string(),
            layer_count: 3,
            ..Default::default()
        };
        let hybrid = GgufCompactMeta {
            architecture: "future_hybrid_arch".to_string(),
            layer_count: 3,
            head_count: 8,
            kv_head_count: 2,
            key_length: 64,
            value_length: 64,
            ssm_conv_kernel: 4,
            ssm_inner_size: 1536,
            ssm_state_size: 16,
            recurrent_layers: vec![true, false, true],
            ..Default::default()
        };

        assert!(pure_recurrent.has_only_recurrent_layers_without_kv_cache());
        assert!(!dense_missing_heads.has_only_recurrent_layers_without_kv_cache());
        assert!(!hybrid.has_only_recurrent_layers_without_kv_cache());
    }

    #[test]
    fn zero_kv_proof_rejects_residual_attention_metadata() {
        let mut meta = GgufCompactMeta {
            architecture: "future_recurrent_arch".to_string(),
            layer_count: 3,
            ssm_conv_kernel: 4,
            ssm_inner_size: 1536,
            ssm_state_size: 16,
            ..Default::default()
        };

        meta.key_length = 64;
        assert!(!meta.has_only_recurrent_layers_without_kv_cache());
        meta.key_length = 0;
        meta.kv_head_counts = vec![0, 1, 0];
        assert!(!meta.has_only_recurrent_layers_without_kv_cache());
    }

    #[test]
    fn recurrent_mask_uses_llama_one_based_full_attention_cadence() {
        let meta = GgufCompactMeta {
            architecture: "future_hybrid_arch".to_string(),
            layer_count: 6,
            full_attention_interval: 3,
            ssm_state_size: 64,
            ..Default::default()
        };

        assert_eq!(
            meta.recurrent_layer_mask(),
            vec![true, true, false, true, true, false]
        );
    }

    fn push_tokenizer_inventory_kvs(bytes: &mut Vec<u8>) {
        push_gguf_string(bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&(GgufType::Array as u32).to_le_bytes());
        push_array_header(bytes, GgufType::String, 2);
        push_gguf_bytes(bytes, b"hello");
        push_gguf_bytes(bytes, b"<eos>");

        push_gguf_string(bytes, "tokenizer.ggml.token_type");
        bytes.extend_from_slice(&(GgufType::Array as u32).to_le_bytes());
        push_array_header(bytes, GgufType::Uint32, 2);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
    }

    #[test]
    fn scan_gguf_tokenizer_inventory_preserves_bytes_and_controls() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        push_tokenizer_inventory_kvs(&mut bytes);

        let path = write_bytes("model-artifact-gguf-tokenizer", &bytes);
        let inventory = scan_gguf_tokenizer_inventory(&path).expect("should parse tokenizer");
        assert_eq!(inventory.tokens.len(), 2);
        assert_eq!(inventory.tokens[0].raw, b"hello");
        assert!(!inventory.tokens[0].is_control);
        assert_eq!(inventory.tokens[1].raw, b"<eos>");
        assert!(inventory.tokens[1].is_control);
        let _ = std::fs::remove_file(path);
    }

    fn push_tensor_info(bytes: &mut Vec<u8>, name: &str, offset: u64) {
        push_gguf_string(bytes, name);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&16u64.to_le_bytes());
        bytes.extend_from_slice(&(GgufType::Uint8 as u32).to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    #[test]
    fn skip_gguf_value_rejects_excessive_array_depth() {
        let mut bytes = Vec::new();
        for _ in 0..=MAX_GGUF_ARRAY_DEPTH {
            push_array_header(&mut bytes, GgufType::Array, 1);
        }
        push_array_header(&mut bytes, GgufType::Uint8, 1);
        bytes.push(0);

        let path = write_bytes("model-artifact-gguf-depth", &bytes);
        let mut file = std::fs::File::open(&path).unwrap();
        let err = skip_gguf_value(&mut file, GgufType::Array).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("nesting too deep"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn skip_gguf_value_rejects_excessive_array_count() {
        let mut bytes = Vec::new();
        push_array_header(&mut bytes, GgufType::Uint8, MAX_GGUF_ARRAY_ELEMENTS + 1);

        let path = write_bytes("model-artifact-gguf-count", &bytes);
        let mut file = std::fs::File::open(&path).unwrap();
        let err = skip_gguf_value(&mut file, GgufType::Array).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("array too long"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_returns_none_on_malicious_nested_array() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&1i64.to_le_bytes());
        push_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&(GgufType::Array as u32).to_le_bytes());
        for _ in 0..=MAX_GGUF_ARRAY_DEPTH {
            push_array_header(&mut bytes, GgufType::Array, 1);
        }
        push_array_header(&mut bytes, GgufType::Uint8, 1);
        bytes.push(0);

        let path = write_bytes("model-artifact-gguf-malicious", &bytes);
        assert!(scan_gguf_compact_meta(&path).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_derives_value_length_from_kv_heads_without_head_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
        push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);

        let path = write_bytes("model-artifact-gguf-kv-heads", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.head_count, 0);
        assert_eq!(meta.kv_head_count, 8);
        assert_eq!(meta.key_length, 0);
        assert_eq!(meta.value_length, 512);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_preserves_kv_head_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&6i64.to_le_bytes());
        push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
        push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
        push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
        push_u32_kv(&mut bytes, "llama.block_count", 24);
        push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
        push_u32_kv(&mut bytes, "llama.attention.value_length", 128);

        let path = write_bytes("model-artifact-gguf-kv-head-count", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.head_count, 32);
        assert_eq!(meta.kv_head_count, 8);
        assert_eq!(meta.effective_kv_head_count(), Some(8));
        assert_eq!(meta.k_cache_bytes_per_token_f16(), Some(49_152));
        assert_eq!(meta.v_cache_bytes_per_token_f16(), Some(49_152));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_records_swa_head_widths() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&8i64.to_le_bytes());
        push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
        push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
        push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
        push_u32_kv(&mut bytes, "llama.block_count", 24);
        push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
        push_u32_kv(&mut bytes, "llama.attention.value_length", 128);
        push_u32_kv(&mut bytes, "llama.attention.key_length_swa", 96);
        push_u32_kv(&mut bytes, "llama.attention.value_length_swa", 96);

        let path = write_bytes("model-artifact-gguf-swa-widths", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.key_length, 128);
        assert_eq!(meta.value_length, 128);
        assert_eq!(meta.key_length_swa, 96);
        assert_eq!(meta.value_length_swa, 96);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_swa_widths_absent_by_default() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&6i64.to_le_bytes());
        push_u32_kv(&mut bytes, "llama.embedding_length", 4096);
        push_u32_kv(&mut bytes, "llama.attention.head_count", 32);
        push_u32_kv(&mut bytes, "llama.attention.head_count_kv", 8);
        push_u32_kv(&mut bytes, "llama.block_count", 24);
        push_u32_kv(&mut bytes, "llama.attention.key_length", 128);
        push_u32_kv(&mut bytes, "llama.attention.value_length", 128);

        let path = write_bytes("model-artifact-gguf-no-swa", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.key_length_swa, 0);
        assert_eq!(meta.value_length_swa, 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_preserves_kv_lora_rank() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&1i64.to_le_bytes());
        push_u32_kv(&mut bytes, "glm-dsa.attention.kv_lora_rank", 512);

        let path = write_bytes("model-artifact-gguf-kv-lora-rank", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.kv_lora_rank, 512);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_prices_per_layer_kv_head_counts() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&7i64.to_le_bytes());
        push_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&(GgufType::String as u32).to_le_bytes());
        push_gguf_string(&mut bytes, "inkling");
        push_u32_kv(&mut bytes, "inkling.embedding_length", 6144);
        push_u32_kv(&mut bytes, "inkling.attention.head_count", 64);
        let kv_head_counts = (0..66)
            .map(|layer| if layer % 6 == 5 { 8 } else { 16 })
            .collect::<Vec<_>>();
        push_u32_array_kv(
            &mut bytes,
            "inkling.attention.head_count_kv",
            &kv_head_counts,
        );
        push_u32_kv(&mut bytes, "inkling.block_count", 66);
        push_u32_kv(&mut bytes, "inkling.attention.key_length", 128);
        push_u32_kv(&mut bytes, "inkling.attention.value_length", 128);

        let path = write_bytes("model-artifact-gguf-inkling-kv-head-counts", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.kv_head_count, 0);
        assert_eq!(meta.kv_head_counts, kv_head_counts);
        assert_eq!(meta.effective_kv_head_count(), Some(16));
        assert_eq!(meta.k_cache_bytes_per_token_f16(), Some(247_808));
        assert_eq!(meta.v_cache_bytes_per_token_f16(), Some(247_808));
        assert_eq!(meta.kv_cache_bytes_per_token_f16(), Some(495_616));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_projector_meta_preserves_vision_and_audio_flags() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&3i64.to_le_bytes());
        push_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&(GgufType::String as u32).to_le_bytes());
        push_gguf_string(&mut bytes, "clip");
        push_bool_kv(&mut bytes, "clip.has_vision_encoder", true);
        push_bool_kv(&mut bytes, "clip.has_audio_encoder", true);

        let path = write_bytes("model-artifact-gguf-inkling-projector", &bytes);
        let meta = scan_gguf_projector_meta(&path).expect("should parse projector GGUF");
        assert_eq!(meta.has_vision_encoder, Some(true));
        assert_eq!(meta.has_audio_encoder, Some(true));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_preserves_nextn_predict_layers() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        push_gguf_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&(GgufType::String as u32).to_le_bytes());
        push_gguf_string(&mut bytes, "deepseek2");
        push_u32_kv(&mut bytes, "deepseek2.nextn_predict_layers", 1);

        let path = write_bytes("model-artifact-gguf-nextn", &bytes);
        let meta = scan_gguf_compact_meta(&path).expect("should parse GGUF");
        assert_eq!(meta.architecture, "deepseek2");
        assert_eq!(meta.nextn_predict_layers, 1);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_tensor_names_any_detects_nextn_tensor_without_reading_data() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        push_tensor_info(&mut bytes, "blk.0.attn_q.weight", 0);
        push_tensor_info(&mut bytes, "blk.23.nextn.eh_proj.weight", 64);

        let path = write_bytes("model-artifact-gguf-nextn-tensor", &bytes);
        let has_nextn = scan_gguf_tensor_names_any(&path, |name| name.contains(".nextn."))
            .expect("tensor scan should parse");
        assert!(has_nextn);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_compact_meta_rejects_negative_kv_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());
        bytes.extend_from_slice(&(-1i64).to_le_bytes());

        let path = write_bytes("model-artifact-gguf-negative-kv", &bytes);
        assert!(scan_gguf_compact_meta(&path).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_tensor_byte_profile_rejects_excessive_tensor_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&((MAX_GGUF_TENSOR_COUNT as i64) + 1).to_le_bytes());
        bytes.extend_from_slice(&0i64.to_le_bytes());

        let path = write_bytes("model-artifact-gguf-too-many-tensors", &bytes);
        assert!(scan_gguf_tensor_byte_profile(&path).is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn read_gguf_value_as_u32_rejects_negative_int32() {
        let path = write_bytes("model-artifact-gguf-negative-int32", &(-1i32).to_le_bytes());
        let mut file = std::fs::File::open(&path).unwrap();
        let err = read_gguf_value_as_u32(&mut file, GgufType::Int32).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string()
                .contains("negative Int32 where unsigned GGUF value was expected")
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn scan_gguf_tensor_byte_profile_splits_base_and_expert_bytes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        bytes.extend_from_slice(&3i64.to_le_bytes());

        push_u32_kv(&mut bytes, "general.alignment", 32);
        push_u32_kv(&mut bytes, "llama.expert_count", 8);
        push_u32_kv(&mut bytes, "llama.expert_used_count", 2);

        push_tensor_info(&mut bytes, "blk.0.ffn_up_exps.weight", 0);
        push_tensor_info(&mut bytes, "blk.0.attn_q.weight", 64);

        let data_start = align_offset(bytes.len() as u64, 32) as usize;
        bytes.resize(data_start, 0);
        bytes.resize(data_start + 96, 0);

        let path = write_bytes("model-artifact-gguf-tensors", &bytes);
        let profile = scan_gguf_tensor_byte_profile(&path).unwrap();
        assert_eq!(profile.expert_count, 8);
        assert_eq!(profile.expert_used_count, 2);
        assert_eq!(profile.expert_tensor_bytes, 64);
        assert_eq!(profile.base_resident_bytes, 32);
        assert_eq!(profile.full_model_bytes, bytes.len() as u64);
        assert_eq!(
            profile.full_model_bytes,
            profile.base_resident_bytes + profile.expert_tensor_bytes + profile.file_overhead_bytes
        );
        let _ = std::fs::remove_file(path);
    }
}
