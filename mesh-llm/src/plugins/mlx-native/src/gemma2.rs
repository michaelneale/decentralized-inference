use std::path::Path;

use mlx_lm::{cache::KeyValueCache, error::Error};
use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    fast,
    macros::{ModuleParameters, Quantizable},
    module::{Module, Param},
    nn,
    ops::{
        expand_dims,
        indexing::{IndexOp, NewAxis},
        softmax_axis, tanh,
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
    #[serde(default)]
    pub rope_traditional: bool,
    #[serde(default = "default_attn_logit_softcapping")]
    pub attn_logit_softcapping: f32,
    #[serde(default = "default_final_logit_softcapping")]
    pub final_logit_softcapping: f32,
    #[serde(default = "default_query_pre_attn_scalar")]
    pub query_pre_attn_scalar: f32,
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_attn_logit_softcapping() -> f32 {
    50.0
}

fn default_final_logit_softcapping() -> f32 {
    30.0
}

fn default_query_pre_attn_scalar() -> f32 {
    144.0
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

impl Module<&Array> for RmsNorm {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let weight = self.weight.value.add(&array!(1.0))?;
        fast::rms_norm(x, &weight, self.eps)
    }

    fn training_mode(&mut self, _mode: bool) {}
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

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub repeats: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub attn_logit_softcapping: f32,

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
    pub rope: nn::Rope,
}

impl Attention {
    pub fn new(
        args: &ModelArgs,
        prefix: &str,
        quantization: &CheckpointQuantization,
    ) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;

        Ok(Self {
            n_heads,
            n_kv_heads,
            repeats: n_heads / n_kv_heads,
            head_dim,
            scale: 1.0 / args.query_pre_attn_scalar.sqrt(),
            attn_logit_softcapping: args.attn_logit_softcapping,
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
            rope: nn::RopeBuilder::new(head_dim)
                .traditional(args.rope_traditional)
                .base(args.rope_theta)
                .build()?,
        })
    }

    fn annotate<T>(result: Result<T, Exception>, op: &str) -> Result<T, Exception> {
        result.map_err(|err| Exception::custom(format!("gemma2 attention {op}: {err}")))
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

        if let Some(cache) = cache.as_mut() {
            queries = self.rope.forward((&queries, cache.offset()))?;
            keys = self.rope.forward((&keys, cache.offset()))?;
            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(&queries)?;
            keys = self.rope.forward(&keys)?;
        }

        queries = queries.multiply(&array!(self.scale))?;

        if self.repeats > 1 {
            queries = queries.reshape(&[b, self.n_kv_heads, self.repeats, l, self.head_dim])?;
            keys = expand_dims(&keys, 2)?;
            values = expand_dims(&values, 2)?;
        }

        let mut scores = queries.matmul(keys.swap_axes(-1, -2)?)?;
        scores = tanh(scores.divide(&array!(self.attn_logit_softcapping))?)?
            .multiply(&array!(self.attn_logit_softcapping))?;

        if let Some(mask) = mask {
            if mask.dtype() == mlx_rs::Dtype::Bool {
                let finfo_min = scores.dtype().finfo_min()?;
                let masked_value = float_scalar(finfo_min, scores.dtype())?;
                scores = mlx_rs::ops::r#where(mask, scores, masked_value)?;
            } else {
                scores = scores.add(mask)?;
            }
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

    fn annotate<T>(result: Result<T, Exception>, op: &str) -> Result<T, Exception> {
        result.map_err(|err| Exception::custom(format!("gemma2 mlp {op}: {err}")))
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let gate = Self::annotate(self.gate_proj.forward(input), "gate_proj")?;
        let up = Self::annotate(self.up_proj.forward(input), "up_proj")?;
        let hidden = nn::gelu_approximate(&gate)?.multiply(up)?;
        Self::annotate(self.down_proj.forward(&hidden), "down_proj")
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    #[param]
    pub input_layernorm: RmsNorm,
    #[param]
    pub pre_feedforward_layernorm: RmsNorm,
    #[param]
    pub post_feedforward_layernorm: RmsNorm,
    #[param]
    pub post_attention_layernorm: RmsNorm,
}

impl TransformerBlock {
    pub fn new(
        args: &ModelArgs,
        layer_idx: i32,
        quantization: &CheckpointQuantization,
    ) -> Result<Self, Exception> {
        let layer_prefix = format!("model.layers.{layer_idx}");
        Ok(Self {
            self_attn: Attention::new(args, &format!("{layer_prefix}.self_attn"), quantization)?,
            mlp: Mlp::new(
                args.hidden_size,
                args.intermediate_size,
                &format!("{layer_prefix}.mlp"),
                quantization,
            )?,
            input_layernorm: RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
            pre_feedforward_layernorm: RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
            post_feedforward_layernorm: RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
            post_attention_layernorm: RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
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
        let norm_x = self
            .input_layernorm
            .forward(x)
            .map_err(|err| Exception::custom(format!("gemma2 block input_layernorm: {err}")))?;
        let r = self
            .self_attn
            .forward(AttentionInput {
                x: &norm_x,
                mask,
                cache,
            })
            .map_err(|err| Exception::custom(format!("gemma2 block self_attn: {err}")))?;
        let post_attention = self.post_attention_layernorm.forward(&r).map_err(|err| {
            Exception::custom(format!("gemma2 block post_attention_layernorm: {err}"))
        })?;
        let h = x.add(post_attention)?;
        let pre_feedforward = self.pre_feedforward_layernorm.forward(&h).map_err(|err| {
            Exception::custom(format!("gemma2 block pre_feedforward_layernorm: {err}"))
        })?;
        let r = self
            .mlp
            .forward(&pre_feedforward)
            .map_err(|err| Exception::custom(format!("gemma2 block mlp: {err}")))?;
        let post_feedforward = self.post_feedforward_layernorm.forward(&r).map_err(|err| {
            Exception::custom(format!("gemma2 block post_feedforward_layernorm: {err}"))
        })?;
        h.add(post_feedforward)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct GemmaModel {
    pub hidden_size_scale: f32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: RmsNorm,
}

impl GemmaModel {
    pub fn new(args: &ModelArgs, quantization: &CheckpointQuantization) -> Result<Self, Exception> {
        Ok(Self {
            hidden_size_scale: (args.hidden_size as f32).sqrt(),
            embed_tokens: maybe_quantized_embedding(
                args.vocab_size,
                args.hidden_size,
                quantization.is_quantized("model.embed_tokens"),
            )?,
            layers: (0..args.num_hidden_layers)
                .map(|idx| TransformerBlock::new(args, idx, quantization))
                .collect::<Result<Vec<_>, _>>()?,
            norm: RmsNorm::new(args.hidden_size, args.rms_norm_eps)?,
        })
    }
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for GemmaModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs)?;
        h = h.multiply(&array!(self.hidden_size_scale))?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => {
                if h.shape()[1] > 1 {
                    let m =
                        nn::MultiHeadAttention::create_additive_causal_mask::<f32>(h.shape()[1])?;
                    Some(m.as_dtype(h.dtype())?)
                } else {
                    None
                }
            }
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (idx, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            h = layer
                .forward(AttentionInput {
                    x: &h,
                    mask: mask.as_ref(),
                    cache: c.as_mut(),
                })
                .map_err(|err| Exception::custom(format!("gemma2 layer {idx}: {err}")))?;
        }

        self.norm
            .forward(&h)
            .map_err(|err| Exception::custom(format!("gemma2 final norm: {err}")))
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub final_logit_softcapping: f32,
    #[quantizable]
    #[param]
    pub model: GemmaModel,
}

impl Model {
    pub fn new(args: ModelArgs, quantization: &CheckpointQuantization) -> Result<Self, Exception> {
        Ok(Self {
            final_logit_softcapping: args.final_logit_softcapping,
            model: GemmaModel::new(&args, quantization)?,
        })
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self
            .model
            .forward(input)
            .map_err(|err| Exception::custom(format!("gemma2 model body: {err}")))?;
        let logits = match &mut self.model.embed_tokens {
            MaybeQuantized::Original(embed_tokens) => embed_tokens
                .as_linear(&out)
                .map_err(|err| Exception::custom(format!("gemma2 tied embed_tokens: {err}")))?,
            MaybeQuantized::Quantized(q_embed_tokens) => {
                q_embed_tokens.as_linear(&out).map_err(|err| {
                    Exception::custom(format!("gemma2 quantized tied embed_tokens: {err}"))
                })?
            }
        };
        tanh(logits.divide(&array!(self.final_logit_softcapping))?)?
            .multiply(&array!(self.final_logit_softcapping))
    }

    fn training_mode(&mut self, mode: bool) {
        <GemmaModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
    }
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

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    temp: f32,
    state: GenerateState<'a>,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
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

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let logits = tri!(self.model.forward(ModelInput {
                    inputs: prompt_token,
                    mask: None,
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
                    mask: None,
                    cache: self.cache,
                }));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}
