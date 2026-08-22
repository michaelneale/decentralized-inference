use std::path::{Path, PathBuf};

use crate::models::find_model_path;
use anyhow::{Result, bail};
use mesh_llm_system::util::validate_draft_min_max;
use model_artifact::gguf::{scan_gguf_compact_meta, scan_gguf_tensor_names_any};
use skippy_runtime::package::{
    PackageExtensionPolicyInfo, PackageGenerationInfo, PackageSpeculativeDecodingInfo,
    PackageSpeculativeProposerInfo, PackageSpeculativeStrategyInfo, PackageWindowPolicyInfo,
};
use skippy_server::{
    NativeMtpProposalConfig, NgramExtensionConfig, NgramProposalConfig, NgramProposerKind,
    SpeculativeDecodeConfig, VerifyWindowConfig,
};

use super::support::{pick_owned, pick_string, pick_string_owned};
use super::types::ResolvedSpeculativeConfig;
use crate::plugin::{BoolOrAuto, SpeculativeConfig};

pub(super) fn resolve_speculative_config(
    model_config: Option<&SpeculativeConfig>,
    global_config: Option<&SpeculativeConfig>,
    _model_id: &str,
    model_path: &Path,
    package_generation: Option<&PackageGenerationInfo>,
) -> Result<ResolvedSpeculativeConfig> {
    let spec_default = pick_owned(
        model_config.and_then(|config| config.spec_default.as_ref()),
        global_config.and_then(|config| config.spec_default.as_ref()),
    );
    let has_explicit_strategy = model_config
        .and_then(|config| config.strategy.as_ref())
        .is_some()
        || global_config
            .and_then(|config| config.strategy.as_ref())
            .is_some();
    let auto_defaults_enabled =
        !matches!(spec_default, Some(BoolOrAuto::Bool(false))) || has_explicit_strategy;
    let mut draft_model_path = model_config
        .and_then(configured_draft_source)
        .or_else(|| global_config.and_then(configured_draft_source))
        .map(resolve_draft_model_path)
        .map(PathBuf::from);
    let draft_selection_policy = pick_string(
        model_config.and_then(|config| config.draft_selection_policy.as_deref()),
        global_config.and_then(|config| config.draft_selection_policy.as_deref()),
        Some("auto"),
    );
    if draft_model_path.is_none()
        && draft_selection_policy == "auto"
        && !matches!(spec_default, Some(BoolOrAuto::Bool(false)))
    {
        draft_model_path = discover_sibling_draft_model(model_path);
    }
    let supports_native_mtp = package_generation_supports_native_mtp(package_generation)
        || direct_gguf_supports_native_mtp(model_path)
        || draft_model_path
            .as_ref()
            .is_some_and(|path| path.is_file() && direct_gguf_supports_native_mtp(path));
    let strategy = pick_string_owned(
        model_config.and_then(|config| config.strategy.as_deref()),
        global_config.and_then(|config| config.strategy.as_deref()),
        Some("auto"),
    );
    let (strategy, native_mtp_enabled) = resolve_native_mtp_strategy(
        strategy,
        auto_defaults_enabled,
        supports_native_mtp,
        package_generation,
        model_path,
    )?;
    let mode = pick_string_owned(
        model_config.and_then(|config| config.mode.as_deref()),
        global_config.and_then(|config| config.mode.as_deref()),
        Some("auto"),
    );
    let mut mode = mode;
    let mut draft_max_tokens = super::support::pick_value(
        model_config.and_then(|config| config.draft_max_tokens),
        global_config.and_then(|config| config.draft_max_tokens),
        0,
    );
    if (mode == "draft" || (mode == "auto" && draft_model_path.is_some())) && draft_max_tokens == 0
    {
        draft_max_tokens = 3;
    }
    let draft_min_tokens = super::support::pick_value(
        model_config.and_then(|config| config.draft_min_tokens),
        global_config.and_then(|config| config.draft_min_tokens),
        0,
    );
    let draft_n_gpu_layers = pick_owned(
        model_config.and_then(|config| config.draft_gpu_layers),
        global_config.and_then(|config| config.draft_gpu_layers),
    );
    let ngram_min = super::support::pick_value(
        model_config.and_then(|config| config.ngram_min),
        global_config.and_then(|config| config.ngram_min),
        0,
    );
    let ngram_max = super::support::pick_value(
        model_config.and_then(|config| config.ngram_max),
        global_config.and_then(|config| config.ngram_max),
        0,
    );
    let pairing_fault = normalize_pairing_fault(pick_string(
        model_config.and_then(|config| config.pairing_fault.as_deref()),
        global_config.and_then(|config| config.pairing_fault.as_deref()),
        Some("warn_disable"),
    ));
    let explicit = mode != "auto"
        || draft_model_path.is_some()
        || draft_max_tokens > 0
        || draft_min_tokens > 0
        || draft_n_gpu_layers.is_some()
        || ngram_min > 0
        || ngram_max > 0;
    if mode == "disabled" && draft_model_path.is_some() {
        bail!("skippy speculative draft source cannot be set when speculative.mode = \"disabled\"");
    }
    let effective_draft_max_tokens =
        resolved_draft_max_tokens(native_mtp_enabled, draft_max_tokens);
    validate_draft_min_max(draft_min_tokens, effective_draft_max_tokens)
        .map_err(anyhow::Error::msg)?;
    if native_mtp_enabled && draft_model_path.is_some() {
        mode = "disabled".to_string();
    } else if mode == "draft" || (mode == "auto" && draft_model_path.is_some()) {
        resolve_draft_speculative_mode(
            &mut mode,
            &mut draft_model_path,
            draft_max_tokens,
            pairing_fault.as_str(),
            model_path,
        )?;
    } else {
        mode = "disabled".to_string();
        draft_model_path = None;
    }
    let decode = resolve_decode_config(DecodeResolutionInput {
        requested_strategy: &strategy,
        native_mtp_enabled,
        draft_max_tokens: effective_draft_max_tokens,
        draft_min_tokens,
        model_config,
        global_config,
        package_generation,
    })?;
    // A standalone N-gram plan (no native MTP, no draft model) runs in the
    // legacy `ngram` mode so the embedded frontend derives its window from the
    // proposer's proposal limit.
    if !native_mtp_enabled && mode != "draft" && decode.ngram.is_some() {
        mode = "ngram".to_string();
    }
    Ok(ResolvedSpeculativeConfig {
        strategy: strategy.clone(),
        native_mtp_enabled,
        mode,
        draft_model_path,
        pairing_fault,
        draft_max_tokens: effective_draft_max_tokens,
        draft_min_tokens,
        explicit,
        draft_n_gpu_layers,
        decode,
    })
}

fn configured_draft_source(config: &SpeculativeConfig) -> Option<String> {
    config.draft_model.clone().or_else(|| {
        config
            .draft_hf_repo
            .as_ref()
            .zip(config.draft_hf_file.as_ref())
            .map(|(repo, file)| format!("{repo}:{file}"))
    })
}

fn resolve_native_mtp_strategy(
    strategy: String,
    auto_defaults_enabled: bool,
    supports_native_mtp: bool,
    package_generation: Option<&PackageGenerationInfo>,
    model_path: &Path,
) -> Result<(String, bool)> {
    let native_mtp_enabled = match strategy.as_str() {
        "auto" => {
            auto_defaults_enabled
                && package_generation_or_direct_default_supports_native_mtp(
                    package_generation,
                    model_path,
                )
        }
        "mtp" => {
            if !supports_native_mtp {
                bail!("skippy speculative.strategy = \"mtp\" requires proven native MTP support");
            }
            true
        }
        "ngram-cache" | "ngram-suffix" => false,
        "disabled" => false,
        package_strategy if package_strategy_exists(package_generation, package_strategy) => {
            let speculative = package_generation
                .and_then(|generation| generation.speculative_decoding.as_ref())
                .expect("checked package strategy exists");
            strategy_uses_native_mtp(speculative, package_strategy)
        }
        _ => bail!(
            "skippy speculative.strategy must be auto, disabled, mtp, ngram-cache, ngram-suffix, or a strategy declared by model-package.json"
        ),
    };
    Ok((strategy, native_mtp_enabled))
}

struct DecodeResolutionInput<'a> {
    requested_strategy: &'a str,
    native_mtp_enabled: bool,
    draft_max_tokens: u32,
    draft_min_tokens: u32,
    model_config: Option<&'a SpeculativeConfig>,
    global_config: Option<&'a SpeculativeConfig>,
    package_generation: Option<&'a PackageGenerationInfo>,
}

#[allow(clippy::too_many_lines)]
fn resolve_decode_config(input: DecodeResolutionInput<'_>) -> Result<SpeculativeDecodeConfig> {
    let mut config = package_decode_config(input.requested_strategy, input.package_generation)?
        .unwrap_or_else(SpeculativeDecodeConfig::default);
    config.requested_strategy = input.requested_strategy.to_string();
    config.draft_acceptance_threshold = pick_owned(
        input
            .model_config
            .and_then(|config| config.draft_acceptance_threshold),
        input
            .global_config
            .and_then(|config| config.draft_acceptance_threshold),
    )
    .unwrap_or_default();
    config.draft_split_probability = pick_owned(
        input
            .model_config
            .and_then(|config| config.draft_split_probability),
        input
            .global_config
            .and_then(|config| config.draft_split_probability),
    )
    .unwrap_or_default();
    config.draft_device = pick_owned(
        input
            .model_config
            .and_then(|config| config.draft_device.clone()),
        input
            .global_config
            .and_then(|config| config.draft_device.clone()),
    );
    config.draft_threads = pick_owned(
        input.model_config.and_then(|config| config.draft_threads),
        input.global_config.and_then(|config| config.draft_threads),
    );
    config.draft_cache_type_k = pick_string_owned(
        input
            .model_config
            .and_then(|config| config.draft_cache_type_k.as_deref()),
        input
            .global_config
            .and_then(|config| config.draft_cache_type_k.as_deref()),
        Some("f16"),
    );
    config.draft_cache_type_v = pick_string_owned(
        input
            .model_config
            .and_then(|config| config.draft_cache_type_v.as_deref()),
        input
            .global_config
            .and_then(|config| config.draft_cache_type_v.as_deref()),
        Some("f16"),
    );

    if input.requested_strategy == "disabled" {
        // A model-level disable must ignore proposer, extension, and native-MTP
        // state inherited from [defaults] or package metadata; otherwise the mode
        // override below would re-enable speculation from those settings.
        config.ngram = None;
        config.extension = None;
        config.native_mtp.enabled = false;
        config.effective_strategy = "disabled".to_string();
        return Ok(config);
    }

    if input.native_mtp_enabled {
        config.native_mtp.enabled = true;
        config.native_mtp.max_draft_tokens = input.draft_max_tokens.max(1) as usize;
        config.native_mtp.min_draft_tokens = input.draft_min_tokens as usize;
    }
    if config.native_mtp.enabled && config.effective_strategy == "disabled" {
        config.effective_strategy = "native-mtp".to_string();
    }

    let ngram_min = pick_optional_u32(
        input.model_config.and_then(|config| config.ngram_min),
        input.global_config.and_then(|config| config.ngram_min),
    )
    .unwrap_or_default();
    let ngram_max = pick_optional_u32(
        input.model_config.and_then(|config| config.ngram_max),
        input.global_config.and_then(|config| config.ngram_max),
    )
    .unwrap_or_default();
    let ngram_max_proposal_tokens = pick_optional_u32(
        input
            .model_config
            .and_then(|config| config.ngram_max_proposal_tokens),
        input
            .global_config
            .and_then(|config| config.ngram_max_proposal_tokens),
    );
    let ngram_proposer = input
        .model_config
        .and_then(|config| config.ngram_proposer.clone())
        .or_else(|| {
            input
                .global_config
                .and_then(|config| config.ngram_proposer.clone())
        });
    // An explicitly requested standalone N-gram strategy must build a proposer
    // even when both bounds are omitted, so resolution fails loudly instead of
    // succeeding with no proposer.
    let explicit_standalone_ngram =
        matches!(input.requested_strategy, "ngram-cache" | "ngram-suffix");
    if explicit_standalone_ngram || config.ngram.is_some() || ngram_min > 0 || ngram_max > 0 {
        let existing = config.ngram.as_ref();
        let min_ngram = nonzero_or(
            ngram_min,
            existing.map_or(0, |ngram| ngram.min_ngram as u32),
        );
        let max_ngram = nonzero_or(
            ngram_max,
            existing.map_or(0, |ngram| ngram.max_ngram as u32),
        );
        if min_ngram == 0 || max_ngram == 0 || min_ngram > max_ngram {
            bail!(
                "skippy speculative N-gram proposer requires both ngram_min and ngram_max with 0 < ngram_min <= ngram_max"
            );
        }
        let kind = match ngram_proposer.as_deref() {
            Some("cache") => NgramProposerKind::Cache,
            Some("suffix") => NgramProposerKind::Suffix,
            None => existing.map_or_else(
                || match input.requested_strategy {
                    "ngram-suffix" => NgramProposerKind::Suffix,
                    _ => NgramProposerKind::Cache,
                },
                |ngram| ngram.kind,
            ),
            Some(_) => unreachable!("validated by mesh configuration"),
        };
        let max_proposal_tokens = ngram_max_proposal_tokens
            .map(|value| value as usize)
            .unwrap_or_else(|| {
                existing.map_or(max_ngram as usize, |ngram| ngram.max_proposal_tokens)
            });
        config.ngram = Some(NgramProposalConfig {
            kind,
            min_ngram: min_ngram as usize,
            max_ngram: max_ngram as usize,
            max_proposal_tokens,
        });
    }

    if config.native_mtp.enabled
        && let Some(ngram) = config.ngram.as_ref()
    {
        config.effective_strategy = match ngram.kind {
            NgramProposerKind::Cache => "native-mtp+ngram-cache",
            NgramProposerKind::Suffix => "native-mtp+ngram-suffix",
        }
        .to_string();
        if config.extension.is_none() {
            config.extension = Some(NgramExtensionConfig {
                max_tokens: ngram.max_proposal_tokens,
            });
        }
    } else if let Some(ngram) = config.ngram.as_ref() {
        config.effective_strategy = ngram_effective_strategy(ngram.kind).to_string();
    }

    let extension_max = pick_optional_u32(
        input
            .model_config
            .and_then(|config| config.extension_max_tokens),
        input
            .global_config
            .and_then(|config| config.extension_max_tokens),
    );
    if extension_max.is_some() {
        let Some(extension) = config.extension.as_mut() else {
            bail!(
                "skippy speculative extension controls require native MTP and an N-gram proposer"
            );
        };
        if let Some(value) = extension_max {
            extension.max_tokens = value as usize;
        }
    }
    if config.extension.is_some() && (!config.native_mtp.enabled || config.ngram.is_none()) {
        bail!("skippy speculative extension requires both native MTP and an N-gram proposer");
    }

    config.native_mtp.reject_cooldown_tokens = pick_optional_u32(
        input
            .model_config
            .and_then(|config| config.native_mtp_reject_cooldown_tokens),
        input
            .global_config
            .and_then(|config| config.native_mtp_reject_cooldown_tokens),
    )
    .map_or(config.native_mtp.reject_cooldown_tokens, |value| {
        value as usize
    });
    config.native_mtp.suppress_cooldown_drafts = pick_owned(
        input
            .model_config
            .and_then(|config| config.native_mtp_suppress_cooldown_drafts),
        input
            .global_config
            .and_then(|config| config.native_mtp_suppress_cooldown_drafts),
    )
    .unwrap_or(config.native_mtp.suppress_cooldown_drafts);
    config.native_mtp.suppress_cooldown_draft_limit = pick_optional_u32(
        input
            .model_config
            .and_then(|config| config.native_mtp_suppress_cooldown_draft_limit),
        input
            .global_config
            .and_then(|config| config.native_mtp_suppress_cooldown_draft_limit),
    )
    .map_or(config.native_mtp.suppress_cooldown_draft_limit, |value| {
        value as usize
    });
    config.verify_window.min_tokens = pick_optional_u32(
        input
            .model_config
            .and_then(|config| config.verify_window_min_tokens),
        input
            .global_config
            .and_then(|config| config.verify_window_min_tokens),
    )
    .map_or(config.verify_window.min_tokens, |value| value as usize);
    config.verify_window.max_tokens = pick_optional_u32(
        input
            .model_config
            .and_then(|config| config.verify_window_max_tokens),
        input
            .global_config
            .and_then(|config| config.verify_window_max_tokens),
    )
    .map_or(config.verify_window.max_tokens, |value| value as usize);
    config.verify_window.pipeline_depth = pick_optional_u32(
        input
            .model_config
            .and_then(|config| config.verify_window_pipeline_depth),
        input
            .global_config
            .and_then(|config| config.verify_window_pipeline_depth),
    )
    .map_or(config.verify_window.pipeline_depth, |value| value as usize);
    config.verify_window.runahead_max_tokens = pick_optional_u32(
        input
            .model_config
            .and_then(|config| config.verify_window_runahead_tokens),
        input
            .global_config
            .and_then(|config| config.verify_window_runahead_tokens),
    )
    .map_or(config.verify_window.runahead_max_tokens, |value| {
        value as usize
    });
    if config.verify_window.min_tokens > config.verify_window.max_tokens {
        bail!("skippy speculative verify window requires min_tokens <= max_tokens");
    }
    config.validate()?;
    Ok(config)
}

fn package_decode_config(
    requested_strategy: &str,
    package_generation: Option<&PackageGenerationInfo>,
) -> Result<Option<SpeculativeDecodeConfig>> {
    let Some(speculative) =
        package_generation.and_then(|generation| generation.speculative_decoding.as_ref())
    else {
        return Ok(None);
    };
    let strategy_name = if requested_strategy == "auto" {
        speculative.default.as_str()
    } else {
        requested_strategy
    };
    let Some(strategy) = speculative.strategies.get(strategy_name) else {
        return Ok(None);
    };
    let (native_mtp, ngram) = match strategy.strategy_type.as_str() {
        "native-mtp" => (
            native_proposer_config(
                strategy
                    .proposer
                    .as_deref()
                    .and_then(|name| speculative.proposers.get(name)),
                strategy,
            )?,
            None,
        ),
        "ngram-cache" | "ngram-suffix" => (
            NativeMtpProposalConfig {
                enabled: false,
                max_draft_tokens: 0,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 0,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 0,
            },
            Some(ngram_proposer_config(
                strategy
                    .proposer
                    .as_deref()
                    .and_then(|name| speculative.proposers.get(name)),
            )?),
        ),
        "composite" => {
            let primary = strategy
                .primary
                .as_deref()
                .and_then(|name| speculative.proposers.get(name))
                .ok_or_else(|| anyhow::anyhow!("package speculative strategy {strategy_name} has no native MTP primary proposer"))?;
            let extender = strategy
                .extender
                .as_deref()
                .and_then(|name| speculative.proposers.get(name))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "package speculative strategy {strategy_name} has no N-gram extender"
                    )
                })?;
            (
                native_proposer_config(Some(primary), strategy)?,
                Some(ngram_proposer_config(Some(extender))?),
            )
        }
        other => bail!("package speculative strategy {strategy_name} has unsupported type {other}"),
    };
    let extension = strategy.extension_policy.as_ref().map(extension_config);
    let verify_window = strategy
        .window_policy
        .as_ref()
        .map(verify_window_config)
        .unwrap_or(VerifyWindowConfig {
            min_tokens: 1,
            max_tokens: 4,
            pipeline_depth: 1,
            runahead_max_tokens: 0,
        });
    let effective_strategy = match (native_mtp.enabled, ngram.as_ref().map(|value| value.kind)) {
        (true, Some(NgramProposerKind::Cache)) => "native-mtp+ngram-cache",
        (true, Some(NgramProposerKind::Suffix)) => "native-mtp+ngram-suffix",
        (true, None) => "native-mtp",
        (false, Some(kind)) => ngram_effective_strategy(kind),
        (false, None) => "disabled",
    };
    Ok(Some(SpeculativeDecodeConfig {
        requested_strategy: requested_strategy.to_string(),
        effective_strategy: effective_strategy.to_string(),
        native_mtp,
        ngram,
        extension,
        verify_window,
        ..SpeculativeDecodeConfig::default()
    }))
}

fn native_proposer_config(
    proposer: Option<&PackageSpeculativeProposerInfo>,
    legacy_strategy: &PackageSpeculativeStrategyInfo,
) -> Result<NativeMtpProposalConfig> {
    let (proposer_type, prediction_depth, layer_indices) = proposer.map_or(
        (
            legacy_strategy.strategy_type.as_str(),
            legacy_strategy.prediction_depth,
            legacy_strategy.layer_indices.as_slice(),
        ),
        |proposer| {
            (
                proposer.proposer_type.as_str(),
                proposer.prediction_depth,
                proposer.layer_indices.as_slice(),
            )
        },
    );
    if proposer_type != "native-mtp" || prediction_depth != Some(1) || layer_indices.is_empty() {
        bail!("package native MTP proposer is not valid for the embedded runtime");
    }
    Ok(NativeMtpProposalConfig {
        enabled: true,
        max_draft_tokens: 1,
        min_draft_tokens: 0,
        reject_cooldown_tokens: 0,
        suppress_cooldown_drafts: false,
        suppress_cooldown_draft_limit: 0,
    })
}

fn ngram_proposer_config(
    proposer: Option<&PackageSpeculativeProposerInfo>,
) -> Result<NgramProposalConfig> {
    let proposer = proposer
        .ok_or_else(|| anyhow::anyhow!("package N-gram strategy must reference a proposer"))?;
    let kind = match proposer.proposer_type.as_str() {
        "ngram-cache" => NgramProposerKind::Cache,
        "ngram-suffix" => NgramProposerKind::Suffix,
        other => bail!("package N-gram proposer has unsupported type {other}"),
    };
    let min_ngram = proposer
        .ngram_min
        .ok_or_else(|| anyhow::anyhow!("package N-gram proposer has no ngram_min"))?;
    let max_ngram = proposer
        .ngram_max
        .ok_or_else(|| anyhow::anyhow!("package N-gram proposer has no ngram_max"))?;
    let max_proposal_tokens = proposer.max_proposal_tokens.unwrap_or(max_ngram);
    Ok(NgramProposalConfig {
        kind,
        min_ngram: min_ngram as usize,
        max_ngram: max_ngram as usize,
        max_proposal_tokens: max_proposal_tokens as usize,
    })
}

fn extension_config(policy: &PackageExtensionPolicyInfo) -> NgramExtensionConfig {
    NgramExtensionConfig {
        max_tokens: policy.max_tokens as usize,
    }
}

fn verify_window_config(policy: &PackageWindowPolicyInfo) -> VerifyWindowConfig {
    VerifyWindowConfig {
        min_tokens: policy.min_window as usize,
        max_tokens: policy.max_window as usize,
        pipeline_depth: policy.pipeline_depth.unwrap_or(1) as usize,
        runahead_max_tokens: 0,
    }
}

fn ngram_effective_strategy(kind: NgramProposerKind) -> &'static str {
    match kind {
        NgramProposerKind::Cache => "ngram-cache",
        NgramProposerKind::Suffix => "ngram-suffix",
    }
}

fn nonzero_or(value: u32, default: u32) -> u32 {
    if value == 0 { default } else { value }
}

fn pick_optional_u32(model: Option<u32>, global: Option<u32>) -> Option<u32> {
    pick_owned(model, global)
}

fn package_strategy_exists(generation: Option<&PackageGenerationInfo>, strategy: &str) -> bool {
    generation
        .and_then(|generation| generation.speculative_decoding.as_ref())
        .is_some_and(|speculative| speculative.strategies.contains_key(strategy))
}

fn strategy_uses_native_mtp(
    speculative: &PackageSpeculativeDecodingInfo,
    strategy_name: &str,
) -> bool {
    let Some(strategy) = speculative.strategies.get(strategy_name) else {
        return false;
    };
    match strategy.strategy_type.as_str() {
        "native-mtp" => strategy
            .proposer
            .as_deref()
            .and_then(|name| speculative.proposers.get(name))
            .map_or(
                strategy.prediction_depth == Some(1) && !strategy.layer_indices.is_empty(),
                |proposer| proposer.proposer_type == "native-mtp",
            ),
        "composite" => strategy
            .primary
            .as_deref()
            .and_then(|name| speculative.proposers.get(name))
            .is_some_and(|proposer| proposer.proposer_type == "native-mtp"),
        _ => false,
    }
}

/// Default native-MTP proposal window when the operator sets no bound.
///
/// Depth 1 is the only depth measured to be a win. On Qwen3.8-27B-UD-Q4_K_XL
/// (RTX 5090, CUDA 13, 8192 ctx) depth 1 reached 42.751 tok/s against a
/// 37.503 tok/s speculation-disabled control (+14.0%), while depth 3 reached
/// 37.938 tok/s (+1.16% over control, 11.3% *slower* than depth 1): the extra
/// proposal compute is paid without a correspondingly longer verified window.
/// Raise this only when the verify-window sizing (`n_rs_seq`) makes deeper
/// drafts reachable and a re-measurement shows depth > 1 beating depth 1.
const NATIVE_MTP_DEFAULT_DRAFT_MAX_TOKENS: u32 = 1;

fn resolved_draft_max_tokens(native_mtp_enabled: bool, draft_max_tokens: u32) -> u32 {
    if native_mtp_enabled && draft_max_tokens == 0 {
        return NATIVE_MTP_DEFAULT_DRAFT_MAX_TOKENS;
    }
    draft_max_tokens
}

fn resolve_draft_model_path(raw: String) -> String {
    let raw_path = PathBuf::from(&raw);
    if raw_path.is_file() {
        return raw;
    }
    if !raw.contains(':') {
        return raw;
    }
    let candidate = find_model_path(&raw);
    if candidate.exists() {
        return candidate.to_string_lossy().into_owned();
    }
    raw
}

fn discover_sibling_draft_model(model_path: &Path) -> Option<PathBuf> {
    let mut candidates = std::fs::read_dir(model_path.parent()?)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path != model_path)
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    let name = name.to_ascii_lowercase();
                    name.contains("draft") || name.contains("eagle")
                })
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.into_iter().next()
}

fn resolve_draft_speculative_mode(
    mode: &mut String,
    draft_model_path: &mut Option<PathBuf>,
    draft_max_tokens: u32,
    pairing_fault: &str,
    model_path: &Path,
) -> Result<()> {
    if draft_model_path.is_none() {
        bail!("skippy speculative draft mode requires an explicit draft_model_path");
    }
    if draft_max_tokens == 0 {
        bail!("skippy speculative draft mode requires draft_max_tokens > 0");
    }
    *mode = "draft".to_string();
    let draft_path = draft_model_path.as_ref().expect("checked above");
    if let Some(reason) = incompatible_draft_pair_reason(model_path, draft_path) {
        match pairing_fault {
            "warn_disable" => {
                *mode = "disabled".to_string();
                *draft_model_path = None;
            }
            "fail_open" => {}
            "fail_closed" => bail!("skippy incompatible speculative draft pairing: {reason}"),
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn package_generation_or_direct_default_supports_native_mtp(
    generation: Option<&PackageGenerationInfo>,
    model_path: &Path,
) -> bool {
    package_generation_supports_default_native_mtp(generation)
        || direct_gguf_supports_native_mtp(model_path)
}

fn package_generation_supports_default_native_mtp(
    generation: Option<&PackageGenerationInfo>,
) -> bool {
    generation
        .and_then(|generation| generation.speculative_decoding.as_ref())
        .is_some_and(|speculative| strategy_uses_native_mtp(speculative, &speculative.default))
}

fn package_generation_supports_native_mtp(generation: Option<&PackageGenerationInfo>) -> bool {
    generation
        .and_then(|generation| generation.speculative_decoding.as_ref())
        .is_some_and(speculative_supports_native_mtp)
}

fn speculative_supports_native_mtp(speculative: &PackageSpeculativeDecodingInfo) -> bool {
    strategy_uses_native_mtp(speculative, "mtp")
}

fn direct_gguf_supports_native_mtp(model_path: &Path) -> bool {
    scan_gguf_compact_meta(model_path).is_some_and(|meta| meta.nextn_predict_layers > 0)
        || scan_gguf_tensor_names_any(model_path, |name| name.contains(".nextn.")).unwrap_or(false)
}

fn normalize_pairing_fault(value: &str) -> String {
    value.replace('-', "_")
}

fn incompatible_draft_pair_reason(model_path: &Path, draft_model_path: &Path) -> Option<String> {
    let target_architecture = model_architecture_from_path(model_path);
    let draft_architecture = model_architecture_from_path(draft_model_path);
    match (target_architecture, draft_architecture) {
        (None, None) => {
            Some("target and draft model architecture metadata is unavailable".to_string())
        }
        (None, Some(_)) => Some("target model architecture metadata is unavailable".to_string()),
        (Some(_), None) => Some("draft model architecture metadata is unavailable".to_string()),
        (Some(target), Some(draft)) if target != draft => Some(format!(
            "target architecture {target} does not match draft architecture {draft}"
        )),
        _ => None,
    }
}

fn model_architecture_from_path(path: &Path) -> Option<String> {
    scan_gguf_compact_meta(path)
        .or_else(|| scan_gguf_compact_meta(&path.join("shared/metadata.gguf")))
        .map(|meta| {
            meta.architecture
                .trim()
                .to_ascii_lowercase()
                .replace('-', "_")
        })
        .filter(|architecture| !architecture.is_empty())
}
