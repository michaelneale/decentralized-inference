use std::path::Path;

use mlx_lm::{
    cache::KeyValueCache,
    utils::rope::{initialize_rope, FloatOrString, RopeVariant},
};
use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask},
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::indexing::{IndexOp, NewAxis},
    quantization::MaybeQuantized,
    Array,
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
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    #[serde(default)]
    pub head_dim: Option<i32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<std::collections::HashMap<String, FloatOrString>>,
}

impl ModelArgs {
    fn resolved_head_dim(&self) -> Result<i32, Exception> {
        if let Some(head_dim) = self.head_dim {
            return Ok(head_dim);
        }
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 {
            return Err(Exception::custom(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            )));
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,

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
    pub rope: RopeVariant,
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
        let head_dim = args.resolved_head_dim()?;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = maybe_quantized_linear(
            dim,
            n_heads * head_dim,
            args.attention_bias,
            quantization.is_quantized(&format!("{prefix}.q_proj")),
        )?;
        let k_proj = maybe_quantized_linear(
            dim,
            n_kv_heads * head_dim,
            args.attention_bias,
            quantization.is_quantized(&format!("{prefix}.k_proj")),
        )?;
        let v_proj = maybe_quantized_linear(
            dim,
            n_kv_heads * head_dim,
            args.attention_bias,
            quantization.is_quantized(&format!("{prefix}.v_proj")),
        )?;
        let o_proj = maybe_quantized_linear(
            n_heads * head_dim,
            dim,
            args.attention_bias,
            quantization.is_quantized(&format!("{prefix}.o_proj")),
        )?;

        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
        })
    }

    fn annotate<T>(result: Result<T, Exception>, op: &str) -> Result<T, Exception> {
        result.map_err(|err| Exception::custom(format!("mistral attention {op}: {err}")))
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

        let mask = mask.map(ScaledDotProductAttentionMask::Array);
        let output = scaled_dot_product_attention(queries, &keys, &values, self.scale, mask, None)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, -1])?;

        Self::annotate(self.o_proj.forward(&output), "o_proj")
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
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
        mlp_bias: bool,
        prefix: &str,
        quantization: &CheckpointQuantization,
    ) -> Result<Self, Exception> {
        let gate_proj = maybe_quantized_linear(
            dim,
            hidden_dim,
            mlp_bias,
            quantization.is_quantized(&format!("{prefix}.gate_proj")),
        )?;
        let down_proj = maybe_quantized_linear(
            hidden_dim,
            dim,
            mlp_bias,
            quantization.is_quantized(&format!("{prefix}.down_proj")),
        )?;
        let up_proj = maybe_quantized_linear(
            dim,
            hidden_dim,
            mlp_bias,
            quantization.is_quantized(&format!("{prefix}.up_proj")),
        )?;

        Ok(Self {
            gate_proj,
            down_proj,
            up_proj,
        })
    }

    fn annotate<T>(result: Result<T, Exception>, op: &str) -> Result<T, Exception> {
        result.map_err(|err| Exception::custom(format!("mistral mlp {op}: {err}")))
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let gate = Self::annotate(self.gate_proj.forward(input), "gate_proj")?;
        let up = Self::annotate(self.up_proj.forward(input), "up_proj")?;
        let down_proj_input = nn::silu(gate)?.multiply(up)?;
        Self::annotate(self.down_proj.forward(&down_proj_input), "down_proj")
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
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(
        args: &ModelArgs,
        layer_idx: i32,
        quantization: &CheckpointQuantization,
    ) -> Result<Self, Exception> {
        let layer_prefix = format!("model.layers.{layer_idx}");
        let self_attn = Attention::new(args, &format!("{layer_prefix}.self_attn"), quantization)?;
        let mlp = Mlp::new(
            args.hidden_size,
            args.intermediate_size,
            args.mlp_bias,
            &format!("{layer_prefix}.mlp"),
            quantization,
        )?;
        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
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
            .map_err(|err| Exception::custom(format!("mistral block input_layernorm: {err}")))?;
        let self_attn_input = AttentionInput {
            x: &norm_x,
            mask,
            cache,
        };
        let r = self
            .self_attn
            .forward(self_attn_input)
            .map_err(|err| Exception::custom(format!("mistral block self_attn: {err}")))?;
        let h = x.add(r)?;
        let post_attention = self.post_attention_layernorm.forward(&h).map_err(|err| {
            Exception::custom(format!("mistral block post_attention_layernorm: {err}"))
        })?;
        let r = self
            .mlp
            .forward(&post_attention)
            .map_err(|err| Exception::custom(format!("mistral block mlp: {err}")))?;
        h.add(r)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct MistralModel {
    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl MistralModel {
    pub fn new(args: &ModelArgs, quantization: &CheckpointQuantization) -> Result<Self, Exception> {
        let embed_tokens = maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            quantization.is_quantized("model.embed_tokens"),
        )?;
        let layers = (0..args.num_hidden_layers)
            .map(|idx| TransformerBlock::new(args, idx, quantization))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for MistralModel
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
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer
                .forward(layer_input)
                .map_err(|err| Exception::custom(format!("mistral layer {idx}: {err}")))?;
        }

        self.norm
            .forward(&h)
            .map_err(|err| Exception::custom(format!("mistral final norm: {err}")))
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
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub model: MistralModel,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs, quantization: &CheckpointQuantization) -> Result<Self, Exception> {
        let model = MistralModel::new(&args, quantization)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(maybe_quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                quantization.is_quantized("lm_head"),
            )?)
        } else {
            None
        };

        Ok(Self {
            args,
            model,
            lm_head,
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
            .map_err(|err| Exception::custom(format!("mistral model body: {err}")))?;
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head
                .forward(&out)
                .map_err(|err| Exception::custom(format!("mistral lm_head: {err}"))),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed_tokens) => embed_tokens
                    .as_linear(&out)
                    .map_err(|err| Exception::custom(format!("mistral tied embed_tokens: {err}"))),
                MaybeQuantized::Quantized(q_embed_tokens) => {
                    q_embed_tokens.as_linear(&out).map_err(|err| {
                        Exception::custom(format!("mistral quantized tied embed_tokens: {err}"))
                    })
                }
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <MistralModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn get_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, mlx_lm::error::Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;
    Ok(model_args)
}

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, mlx_lm::error::Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_model_args(model_dir)?;
    let quantization = checkpoint_quantization(model_dir)?;
    let mut model = Model::new(model_args, &quantization)?;

    for weight_file in checkpoint_weight_files(model_dir)? {
        let weights_filename = model_dir.join(weight_file);
        load_safetensors_with_quantized_key_compat(&mut model, weights_filename)?;
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
                let input = ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let input = ModelInput {
                    inputs: &inputs,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let y = tri!(sample(&logits.index((.., -1, ..)), self.temp));
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}
