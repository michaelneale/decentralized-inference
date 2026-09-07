use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::Value;
use skippy_runtime::{CheckpointQuantization, RuntimeConfig, StageModel};

use crate::output::print_success;

#[derive(Debug, Parser)]
pub(crate) struct ValidateSafetensorsLoadArgs {
    /// Local Hugging Face checkpoint directory.
    source: PathBuf,
    /// Direct-load quantization recipe (for example BF16 or Q4_K_M).
    #[arg(long, default_value = "preserve")]
    quantization: CheckpointQuantization,
    /// Importance matrix for IQ and other calibration-dependent recipes.
    #[arg(long)]
    imatrix: Option<PathBuf>,
    /// Load the complete model instead of a one-layer stage smoke test.
    #[arg(long)]
    full_model: bool,
}

pub(crate) fn run_validate_safetensors_load(args: ValidateSafetensorsLoadArgs) -> Result<()> {
    let config_path = args.source.join("config.json");
    let config: Value = serde_json::from_slice(
        &fs::read(&config_path).with_context(|| format!("read {}", config_path.display()))?,
    )
    .with_context(|| format!("parse {}", config_path.display()))?;
    let layer_count = config
        .get("num_hidden_layers")
        .and_then(Value::as_u64)
        .context("config.json missing integer num_hidden_layers")?;
    let layer_count = u32::try_from(layer_count).context("num_hidden_layers exceeds u32")?;
    ensure!(
        layer_count > 0,
        "num_hidden_layers must be greater than zero"
    );

    let full_model = args.full_model;
    let runtime_config = RuntimeConfig {
        layer_end: if full_model { layer_count } else { 1 },
        filter_tensors_on_load: !full_model,
        resident_tensor_names: Vec::new(),
        include_output: full_model,
        ctx_size: 128,
        n_batch: Some(128),
        n_ubatch: Some(128),
        checkpoint_quantization: args.quantization,
        checkpoint_imatrix: args.imatrix,
        checkpoint_imatrix_sha256: None,
        ..RuntimeConfig::default()
    };
    let model = StageModel::open(&args.source, &runtime_config)?;
    print_success(format!(
        "SafeTensors direct-load smoke passed: source={} quantization={:?} layers=0..{}",
        args.source.display(),
        args.quantization,
        runtime_config.layer_end
    ));
    drop(model);
    Ok(())
}
