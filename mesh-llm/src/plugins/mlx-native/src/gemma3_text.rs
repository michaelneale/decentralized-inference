use std::{collections::HashMap, path::Path};

use mlx_lm::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::Error,
    utils::rope::{initialize_rope, FloatOrString, RopeVariant},
};
use mlx_rs::{
    arange, argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{
        clip, concatenate_axis, expand_dims,
        indexing::{IndexOp, NewAxis},
        softmax_axis,
    },
    quantization::MaybeQuantized,
    Array, Dtype,
};
use serde::Deserialize;

use crate::quantized::{
    checkpoint_quantization, checkpoint_weight_files, load_safetensors_with_quantized_key_compat,
    maybe_quantized_embedding, maybe_quantized_linear, CheckpointQuantization,
};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    #[allow(dead_code)]
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_rope_local_base_freq")]
    pub rope_local_base_freq: f32,
    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: f32,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_rope_local_base_freq() -> f32 {
    10_000.0
}

fn default_query_pre_attn_scalar() -> f32 {
    256.0
}

fn default_sliding_window() -> i32 {
    512
}

fn default_max_position_embeddings() -> i32 {
    32_768
}

impl ModelArgs {
    fn layer_type(&self, layer_idx: i32) -> &str {
        self.layer_types
            .as_ref()
            .and_then(|types| types.get(layer_idx as usize))
            .map(String::as_str)
            .unwrap_or("full_attention")
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct RmsNorm {
    #[param]
    pub weight: Param<Array>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(dims: i32, eps: f32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(mlx_rs::ops::ones::<f32>(&[dims])?),
            eps,
        })
    }
}

fn float_scalar(value: f64, dtype: Dtype) -> Result<Array, Exception> {
    match dtype {
        Dtype::Float16 | Dtype::Float32 | Dtype::Bfloat16 => {
            Array::from_f32(value as f32).as_dtype(dtype)
        }
        Dtype::Complex64 => Array::from_f32(value as f32).as_dtype(dtype),
        Dtype::Float64 => Ok(Array::from_f64(value)),
        _ => Err(Exception::custom(format!(
            "unsupported floating dtype for scalar conversion: {dtype:?}"
        ))),
    }
}

impl Module<&Array> for RmsNorm {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let weight = self.weight.value.add(&array!(1.0))?;
        mlx_rs::fast::rms_norm(x, &weight, self.eps)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[derive(Debug, Clone, Default)]
pub struct RotatingKeyValueCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
    max_size: i32,
}

impl RotatingKeyValueCache {
    pub fn new(max_size: i32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            max_size,
        }
    }
}

impl KeyValueCache for RotatingKeyValueCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        Some(self.max_size)
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let new_len = keys.shape()[keys.shape().len() - 2];

        let (mut keys, mut values) = match (self.keys.take(), self.values.take()) {
            (Some(existing_keys), Some(existing_values)) => (
                concatenate_axis(&[existing_keys, keys], -2)?,
                concatenate_axis(&[existing_values, values], -2)?,
            ),
            _ => (keys, values),
        };

        let total_cached = keys.shape()[keys.shape().len() - 2];
        if total_cached > self.max_size {
            let start = total_cached - self.max_size;
            keys = keys.index((.., .., start.., ..));
            values = values.index((.., .., start.., ..));
        }

        self.offset += new_len;
        self.keys = Some(keys);
        self.values = Some(values);

        Ok((
            self.keys.clone().expect("keys should be present"),
            self.values.clone().expect("values should be present"),
        ))
    }
}

#[derive(Debug, Clone)]
pub enum Gemma3Cache {
    Global(ConcatKeyValueCache),
    Sliding(RotatingKeyValueCache),
}

impl Gemma3Cache {
    pub fn global() -> Self {
        Self::Global(ConcatKeyValueCache::new())
    }

    pub fn sliding(max_size: i32) -> Self {
        Self::Sliding(RotatingKeyValueCache::new(max_size))
    }
}

impl KeyValueCache for Gemma3Cache {
    fn offset(&self) -> i32 {
        match self {
            Gemma3Cache::Global(cache) => cache.offset(),
            Gemma3Cache::Sliding(cache) => cache.offset(),
        }
    }

    fn max_size(&self) -> Option<i32> {
        match self {
            Gemma3Cache::Global(cache) => cache.max_size(),
            Gemma3Cache::Sliding(cache) => cache.max_size(),
        }
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        match self {
            Gemma3Cache::Global(cache) => cache.update_and_fetch(keys, values),
            Gemma3Cache::Sliding(cache) => cache.update_and_fetch(keys, values),
        }
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub repeats: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub is_sliding: bool,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: RmsNorm,
    #[param]
    pub k_norm: RmsNorm,
    #[param]
    pub rope: RopeVariant,
}

impl Attention {
    pub fn new(
        args: &ModelArgs,
        layer_idx: i32,
        prefix: &str,
        quantization: &CheckpointQuantization,
    ) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let is_sliding = args.layer_type(layer_idx) == "sliding_attention";

        let rope = if is_sliding {
            initialize_rope(
                head_dim,
                args.rope_local_base_freq,
                false,
                &None,
                args.max_position_embeddings,
            )?
        } else {
            initialize_rope(
                head_dim,
                args.rope_theta,
                false,
                &args.rope_scaling,
                args.max_position_embeddings,
            )?
        };

        Ok(Self {
            n_heads,
            n_kv_heads,
            repeats: n_heads / n_kv_heads,
            head_dim,
            scale: args.query_pre_attn_scalar.powf(-0.5),
            is_sliding,
            q_proj: maybe_quantized_linear(
                dim,
                n_heads * head_dim,
                false,
                quantization.is_quantized(&format!("{prefix}.q_proj")),
            )?,
            k_proj: maybe_quantized_linear(
                dim,
                n_kv_heads * head_dim,
                false,
                quantization.is_quantized(&format!("{prefix}.k_proj")),
            )?,
            v_proj: maybe_quantized_linear(
                dim,
                n_kv_heads * head_dim,
                false,
                quantization.is_quantized(&format!("{prefix}.v_proj")),
            )?,
            o_proj: maybe_quantized_linear(
                n_heads * head_dim,
                dim,
                false,
                quantization.is_quantized(&format!("{prefix}.o_proj")),
            )?,
            q_norm: RmsNorm::new(head_dim, args.rms_norm_eps)?,
            k_norm: RmsNorm::new(head_dim, args.rms_norm_eps)?,
            rope,
        })
    }

    fn annotate<T>(result: Result<T, Exception>, op: &str) -> Result<T, Exception> {
        result.map_err(|err| Exception::custom(format!("gemma3_text attention {op}: {err}")))
    }
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        let queries = Self::annotate(self.q_proj.forward(x), "q_proj")?;
        let keys = Self::annotate(self.k_proj.forward(x), "k_proj")?;
        let values = Self::annotate(self.v_proj.forward(x), "v_proj")?;

        let mut queries = queries
            .reshape(&[b, l, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = keys
            .reshape(&[b, l, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values
            .reshape(&[b, l, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        queries = self.q_norm.forward(&queries)?;
        keys = self.k_norm.forward(&keys)?;

        if let Some(cache) = cache.as_mut() {
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(cache.offset())
                .build()?;
            queries = self.rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys)
                .offset(cache.offset())
                .build()?;
            keys = self.rope.forward(k_input)?;
            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        queries = queries.multiply(&array!(self.scale))?;

        if self.repeats > 1 {
            queries = queries.reshape(&[b, self.n_kv_heads, self.repeats, l, self.head_dim])?;
            keys = expand_dims(&keys, 2)?;
            values = expand_dims(&values, 2)?;
        }

        let mut scores = queries.matmul(keys.swap_axes(-1, -2)?)?;
        if let Some(mask) = mask {
            let finfo_min = scores.dtype().finfo_min()?;
            let masked_value = float_scalar(finfo_min, scores.dtype())?;
            scores = mlx_rs::ops::r#where(mask, scores, masked_value)?;
        }
        scores = softmax_axis(&scores, -1, true)?;

        let mut output = scores.matmul(&values)?;
        if self.repeats > 1 {
            output = output.reshape(&[b, self.n_heads, l, self.head_dim])?;
        }
        let output = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, -1])?;

        Self::annotate(self.o_proj.forward(&output), "o_proj")
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(
        dim: i32,
        hidden_dim: i32,
        prefix: &str,
        quantization: &CheckpointQuantization,
    ) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: maybe_quantized_linear(
                dim,
                hidden_dim,
                false,
                quantization.is_quantized(&format!("{prefix}.gate_proj")),
            )?,
            down_proj: maybe_quantized_linear(
                hidden_dim,
                dim,
                false,
                quantization.is_quantized(&format!("{prefix}.down_proj")),
            )?,
            up_proj: maybe_quantized_linear(
                dim,
                hidden_dim,
                false,
                quantization.is_quantized(&format!("{prefix}.up_proj")),
            )?,
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let gate = self.gate_proj.forward(input)?;
        let up = self.up_proj.forward(input)?;
        let hidden = nn::gelu_approximate(&gate)?.multiply(up)?;
        self.down_proj.forward(&hidden)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

fn clip_residual(x: &Array, y: &Array) -> Result<Array, Exception> {
    if x.dtype() != Dtype::Float16 {
        return x.add(y);
    }

    let bound = x.dtype().finfo_max()?;
    let sum = x
        .as_dtype(Dtype::Float32)?
        .add(y.as_dtype(Dtype::Float32)?)?;
    clip(
        &sum,
        (
            float_scalar(-bound, Dtype::Float32)?,
            float_scalar(bound, Dtype::Float32)?,
        ),
    )?
    .as_dtype(Dtype::Float16)
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    pub is_sliding: bool,

    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    #[param]
    pub input_layernorm: RmsNorm,
    #[param]
    pub post_attention_layernorm: RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: RmsNorm,
    #[param]
    pub post_feedforward_layernorm: RmsNorm,
}

impl TransformerBlock {
    pub fn new(
        args: &ModelArgs,
        layer_idx: i32,
        quantization: &CheckpointQuantization,
    ) -> Result<Self, Exception> {
        let layer_prefix = format!("model.layers.{layer_idx}");
        Ok(Self {
            is_sliding: args.layer_type(layer_idx) == "sliding_attention",
            self_attn: Attention::new(
                args,
                layer_idx,
                &format!("{layer_prefix}.self_attn"),
                quantization,
            )?,
            mlp: Mlp::new(
                args.hidden_size,
                args.intermediate_size,
                &format!("{layer_prefix}.mlp"),
                quantization,
            )?,
            input_layernorm: RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
            post_attention_layernorm: RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
            pre_feedforward_layernorm: RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
            post_feedforward_layernorm: RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;
        let r = self
            .self_attn
            .forward(AttentionInput {
                x: &self.input_layernorm.forward(x)?,
                mask,
                cache,
            })
            .map_err(|err| Exception::custom(format!("gemma3_text self_attn: {err}")))?;
        let h = clip_residual(x, &self.post_attention_layernorm.forward(&r)?)?;
        let r = self
            .mlp
            .forward(&self.pre_feedforward_layernorm.forward(&h)?)?;
        clip_residual(&h, &self.post_feedforward_layernorm.forward(&r)?)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma3TextModel {
    pub hidden_size_scale: f32,
    pub sliding_window: i32,
    pub layer_is_sliding: Vec<bool>,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: RmsNorm,
}

impl Gemma3TextModel {
    pub fn new(args: &ModelArgs, quantization: &CheckpointQuantization) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|idx| TransformerBlock::new(args, idx, quantization))
            .collect::<Result<Vec<_>, _>>()?;
        let layer_is_sliding = layers.iter().map(|layer| layer.is_sliding).collect();

        Ok(Self {
            hidden_size_scale: (args.hidden_size as f32).sqrt(),
            sliding_window: args.sliding_window,
            layer_is_sliding,
            embed_tokens: maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                quantization.is_quantized("model.embed_tokens"),
            )?,
            layers,
            norm: RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
        })
    }

    fn make_cache(&self) -> Vec<Option<Gemma3Cache>> {
        self.layer_is_sliding
            .iter()
            .map(|is_sliding| {
                Some(if *is_sliding {
                    Gemma3Cache::sliding(self.sliding_window)
                } else {
                    Gemma3Cache::global()
                })
            })
            .collect()
    }
}

pub struct ModelInput<'a> {
    pub inputs: &'a Array,
    pub cache: &'a mut Vec<Option<Gemma3Cache>>,
}

impl Module<ModelInput<'_>> for Gemma3TextModel {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, cache } = input;

        let mut h = self.embed_tokens.forward(inputs)?;
        h = h.multiply(&array!(self.hidden_size_scale))?;

        if cache.is_empty() {
            *cache = self.make_cache();
        }

        for (idx, (layer, cache_slot)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let mask = create_attention_mask(
                &h,
                cache_slot.as_ref(),
                if layer.is_sliding {
                    Some(self.sliding_window)
                } else {
                    None
                },
            )?;
            h = layer
                .forward(AttentionInput {
                    x: &h,
                    mask: mask.as_ref(),
                    cache: cache_slot.as_mut(),
                })
                .map_err(|err| Exception::custom(format!("gemma3_text layer {idx}: {err}")))?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <TransformerBlock as Module<AttentionInput<'_, Gemma3Cache>>>::training_mode(
                layer, mode,
            );
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub tie_word_embeddings: bool,

    #[quantizable]
    #[param]
    pub model: Gemma3TextModel,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs, quantization: &CheckpointQuantization) -> Result<Self, Exception> {
        Ok(Self {
            tie_word_embeddings: args.tie_word_embeddings,
            model: Gemma3TextModel::new(&args, quantization)?,
            lm_head: if args.tie_word_embeddings {
                None
            } else {
                Some(maybe_quantized_linear(
                    args.hidden_size,
                    args.vocab_size,
                    false,
                    quantization.is_quantized("lm_head"),
                )?)
            },
        })
    }
}

impl Module<ModelInput<'_>> for Model {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(&out),
                MaybeQuantized::Quantized(q_embed_tokens) => q_embed_tokens.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma3TextModel as Module<ModelInput<'_>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

fn create_attention_mask(
    h: &Array,
    cache: Option<&Gemma3Cache>,
    window_size: Option<i32>,
) -> Result<Option<Array>, Exception> {
    let token_count = h.shape()[1];
    if token_count <= 1 {
        return Ok(None);
    }

    let offset = cache.map(KeyValueCache::offset).unwrap_or(0);
    create_causal_mask(token_count, offset, window_size).map(Some)
}

fn create_causal_mask(
    token_count: i32,
    offset: i32,
    window_size: Option<i32>,
) -> Result<Array, Exception> {
    let rinds = arange!(stop = offset + token_count)?;
    let linds = arange!(start = offset, stop = offset + token_count)?;
    let linds = linds.index((.., NewAxis));
    let rinds = rinds.index(NewAxis);

    let mut mask = linds.ge(&rinds)?;
    if let Some(window_size) = window_size {
        mask = mask.logical_and(&linds.le(&(rinds + window_size))?)?;
    }

    Ok(mask)
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;
    Ok(model_args)
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_model_args(model_dir)?;
    let quantization = checkpoint_quantization(model_dir)?;
    let mut model = Model::new(model_args, &quantization)?;

    for weight_file in checkpoint_weight_files(model_dir)? {
        load_safetensors_with_quantized_key_compat(&mut model, model_dir.join(weight_file))?;
    }

    Ok(model)
}

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits)
        }
    }
}

pub struct Generate<'a> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<Gemma3Cache>>,
    temp: f32,
    state: GenerateState<'a>,
}

impl<'a> Generate<'a> {
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<Gemma3Cache>>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            temp,
            state: GenerateState::Prefill { prompt_token },
        }
    }
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode { y: Array },
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a> Iterator for Generate<'a> {
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let logits = tri!(self.model.forward(ModelInput {
                    inputs: prompt_token,
                    cache: self.cache,
                }));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let logits = tri!(self.model.forward(ModelInput {
                    inputs: &inputs,
                    cache: self.cache,
                }));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}
