use std::{collections::HashSet, path::Path, rc::Rc};

use mlx_lm::error::Error;
use mlx_rs::{
    builder::Builder,
    error::Exception,
    module::{ModuleParameters, ModuleParametersExt},
    nn,
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub weight_map: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Default)]
pub struct CheckpointQuantization {
    quantized_prefixes: HashSet<String>,
}

impl CheckpointQuantization {
    pub fn is_quantized(&self, prefix: &str) -> bool {
        self.quantized_prefixes.contains(prefix)
    }
}

pub fn maybe_quantized_linear(
    input_dims: i32,
    output_dims: i32,
    bias: bool,
    quantized: bool,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    if quantized {
        Ok(MaybeQuantized::Quantized(
            nn::QuantizedLinearBuilder::new(input_dims, output_dims)
                .bias(bias)
                .build()?,
        ))
    } else {
        Ok(MaybeQuantized::Original(
            nn::LinearBuilder::new(input_dims, output_dims)
                .bias(bias)
                .build()?,
        ))
    }
}

pub fn maybe_quantized_embedding(
    embedding_count: i32,
    dimensions: i32,
    quantized: bool,
) -> Result<MaybeQuantized<nn::Embedding>, Exception> {
    if quantized {
        Ok(MaybeQuantized::Quantized(
            nn::QuantizedEmbeddingBuilder::new(embedding_count, dimensions).build()?,
        ))
    } else {
        Ok(MaybeQuantized::Original(nn::Embedding::new(
            embedding_count,
            dimensions,
        )?))
    }
}

pub fn checkpoint_quantization(
    model_dir: impl AsRef<Path>,
) -> Result<CheckpointQuantization, Error> {
    let weights_index = model_dir.as_ref().join("model.safetensors.index.json");
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        return Ok(CheckpointQuantization {
            quantized_prefixes: quantized_prefixes(
                weight_map.weight_map.keys().map(String::as_str),
            ),
        });
    }

    let weights_filename = model_dir.as_ref().join("model.safetensors");
    if !weights_filename.exists() {
        return Ok(CheckpointQuantization::default());
    }

    let loaded = mlx_rs::Array::load_safetensors(weights_filename)?;
    Ok(CheckpointQuantization {
        quantized_prefixes: quantized_prefixes(loaded.keys().map(String::as_str)),
    })
}

pub fn checkpoint_weight_files(model_dir: impl AsRef<Path>) -> Result<Vec<String>, Error> {
    let weights_index = model_dir.as_ref().join("model.safetensors.index.json");
    if !weights_index.exists() {
        return Ok(vec!["model.safetensors".to_string()]);
    }

    let json = std::fs::read_to_string(weights_index)?;
    let weight_map: WeightMap = serde_json::from_str(&json)?;
    let mut files = weight_map
        .weight_map
        .values()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

pub fn load_safetensors_with_quantized_key_compat<M>(
    module: &mut M,
    path: impl AsRef<Path>,
) -> Result<(), Error>
where
    M: ModuleParameters + ModuleParametersExt,
{
    let loaded = Array::load_safetensors(path)?;
    let param_keys = module.parameters().flatten();
    let rewritten = loaded
        .into_iter()
        .map(|(key, value)| (rewrite_quantized_weight_key(&key, &param_keys), value))
        .collect();

    module.update_flattened(rewritten);
    module.eval()?;
    Ok(())
}

fn quantized_prefixes<'a>(names: impl IntoIterator<Item = &'a str>) -> HashSet<String> {
    let mut prefixes = HashSet::new();
    for name in names {
        if let Some(prefix) = name
            .strip_suffix(".scales")
            .or_else(|| name.strip_suffix(".biases"))
        {
            prefixes.insert(prefix.to_string());
        }
    }
    prefixes
}

fn rewrite_quantized_weight_key(
    key: &str,
    param_keys: &std::collections::HashMap<Rc<str>, &Array>,
) -> Rc<str> {
    if param_keys.contains_key(key) {
        return key.into();
    }

    if let Some(prefix) = key.strip_suffix(".weight") {
        let compat_key: Rc<str> = format!("{prefix}.inner.weight").into();
        if param_keys.contains_key(compat_key.as_ref()) {
            return compat_key;
        }
    }

    key.into()
}
