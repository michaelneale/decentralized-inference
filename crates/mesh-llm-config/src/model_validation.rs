use crate::diagnostic::{ConfigDiagnostic, DiagnosticResult, alias_diagnostic};
use crate::hardware_validation::{
    validate_gpu_assignment_constraints, validate_hardware, validate_throughput,
};
use crate::model::{
    AdvancedConfig, BoolOrAuto, ConfigPath, ConfigPathSegment, GpuAssignment, HardwareConfig,
    IntegerOrString, MeshConfig, ModelConfigDefaults, ModelConfigEntry, ModelFitConfig,
    MultimodalConfig, PrefixCacheConfig, ReasoningBudget, ReasoningEnabled, RequestDefaultsConfig,
    SkippyConfig, SpeculativeConfig, StringOrStringList, merge_hardware, merge_model_fit,
    merge_multimodal, merge_throughput,
};
use skippy_protocol::{MAX_VERIFY_WINDOW_PIPELINE_DEPTH, MAX_VERIFY_WINDOW_RUNAHEAD_TOKENS};

use crate::validation_support::{
    looks_like_model_identifier, validate_allowed, validate_bool_or_auto, validate_hf_pair,
    validate_model_identifier, validate_non_empty, validate_non_negative_f64,
    validate_optional_enum, validate_optional_http_url, validate_optional_kv_cache_type,
    validate_optional_path, validate_optional_positive_u64, validate_optional_positive_usize,
    validate_optional_u32_range, validate_positive_f64, validate_probability, validate_string_list,
    validation_diagnostic,
};

mod topology;
pub(crate) use topology::model_topology_diagnostics;

pub(crate) fn validate_duplicate_model_entries(
    models: &[ModelConfigEntry],
    defaults: Option<&ModelConfigDefaults>,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    for i in 0..models.len() {
        for j in (i + 1)..models.len() {
            let public_identity_i = models[i].model.trim();
            let public_identity_j = models[j].model.trim();
            let first_profile = effective_model_profile(&models[i], defaults);
            let second_profile = effective_model_profile(&models[j], defaults);
            if models[i].model == models[j].model && first_profile == second_profile {
                let profile_i = first_profile.clone();
                let profile_clause = if profile_i.is_empty() {
                    " and default profile".to_string()
                } else {
                    format!(" and profile=\"{profile_i}\"")
                };
                diagnostics.push(validation_diagnostic(
                    "models",
                    format!(
                        "duplicate model entry: models[{i}] and models[{j}] both have model=\"{}\"{profile_clause}",
                        models[i].model,
                    ),
                ));
            } else if !public_identity_i.is_empty() && public_identity_i == public_identity_j {
                diagnostics.push(validation_diagnostic(
                    "models",
                    format!(
                        "duplicate public served identity: models[{i}] and models[{j}] both publish \"{public_identity_i}\""
                    ),
                ));
            }

            let first_name = effective_served_model_name(&models[i], defaults);
            let second_name = effective_served_model_name(&models[j], defaults);
            if first_name == second_name
                && !(models[i].model == models[j].model && first_profile == second_profile)
            {
                diagnostics.push(validation_diagnostic(
                    "models",
                    format!(
                        "duplicate served model identity: models[{i}] and models[{j}] both resolve to {first_name:?}"
                    ),
                ));
            }
        }
    }
}

fn effective_served_model_name(
    model: &ModelConfigEntry,
    defaults: Option<&ModelConfigDefaults>,
) -> String {
    let base_name = model
        .advanced
        .as_ref()
        .and_then(|advanced| advanced.server.as_ref())
        .and_then(|server| server.alias.as_deref())
        .or_else(|| {
            defaults
                .and_then(|value| value.advanced.as_ref())
                .and_then(|advanced| advanced.server.as_ref())
                .and_then(|server| server.alias.as_deref())
        })
        .unwrap_or(&model.model)
        .trim();
    let profile = effective_model_profile(model, defaults);
    if profile.is_empty() {
        base_name.to_string()
    } else {
        format!("{base_name}#{profile}")
    }
}

fn effective_model_profile(
    model: &ModelConfigEntry,
    defaults: Option<&ModelConfigDefaults>,
) -> String {
    model.with_profile_defaults(defaults).derived_profile()
}

pub(crate) fn collect_legacy_draft_model_path_warnings(
    config: &MeshConfig,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if let Some(speculative) = config
        .defaults
        .as_ref()
        .and_then(|d| d.speculative.as_ref())
        .filter(|s| s.legacy_draft_model_path_used)
    {
        // Only warn when the value looks like a model identifier (a ':' that
        // sits after the last '/', as in `Org/Name:Q4_K_M`). Bare local paths
        // including Windows-style absolutes like `C:/models/draft.gguf` put
        // their ':' before the slash and are not identifiers, so they cannot
        // be migrated to draft_model without failing identifier validation.
        if speculative
            .draft_model
            .as_deref()
            .is_some_and(looks_like_model_identifier)
        {
            diagnostics.push(alias_diagnostic(
                ConfigPath::from_fields(["defaults", "speculative", "draft_model_path"]),
                ConfigPath::from_fields(["defaults", "speculative", "draft_model"]),
                "draft_model_path is deprecated; rename to draft_model",
            ));
        }
    }
    for (index, model) in config.models.iter().enumerate() {
        if let Some(speculative) = model
            .speculative
            .as_ref()
            .filter(|s| s.legacy_draft_model_path_used)
        {
            // Only warn when the value looks like a model identifier (a ':'
            // that sits after the last '/', as in `Org/Name:Q4_K_M`). Bare
            // local paths including Windows-style absolutes like
            // `C:/models/draft.gguf` cannot be migrated to draft_model
            // without failing identifier validation.
            if speculative
                .draft_model
                .as_deref()
                .is_some_and(looks_like_model_identifier)
            {
                let mut used_path =
                    ConfigPath::from_fields(["models", "speculative", "draft_model_path"]);
                used_path
                    .segments
                    .insert(1, ConfigPathSegment::Index { index });
                let mut canonical_path =
                    ConfigPath::from_fields(["models", "speculative", "draft_model"]);
                canonical_path
                    .segments
                    .insert(1, ConfigPathSegment::Index { index });
                diagnostics.push(alias_diagnostic(
                    used_path,
                    canonical_path,
                    "draft_model_path is deprecated; rename to draft_model",
                ));
            }
        }
    }
}

pub(crate) fn validate_model_defaults(
    defaults: &ModelConfigDefaults,
    base_path: &str,
    gpu_assignment: GpuAssignment,
) -> DiagnosticResult {
    if let Some(model_fit) = &defaults.model_fit {
        validate_model_fit(model_fit, &format!("{base_path}.model_fit"))?;
    }
    if let Some(hardware) = &defaults.hardware {
        validate_hardware(hardware, &format!("{base_path}.hardware"), gpu_assignment)?;
        validate_gpu_assignment_constraints(
            Some(hardware),
            None,
            None,
            &format!("{base_path}.hardware.device"),
            gpu_assignment,
            false,
        )?;
    }
    if let Some(throughput) = &defaults.throughput {
        validate_throughput(throughput, &format!("{base_path}.throughput"))?;
    }
    if let Some(skippy) = &defaults.skippy {
        validate_skippy(skippy, &format!("{base_path}.skippy"))?;
    }
    if let Some(speculative) = &defaults.speculative {
        validate_speculative(speculative, &format!("{base_path}.speculative"))?;
    }
    if let Some(request_defaults) = &defaults.request_defaults {
        validate_request_defaults(request_defaults, &format!("{base_path}.request_defaults"))?;
    }
    validate_multimodal_pair(
        defaults.hardware.as_ref(),
        defaults.multimodal.as_ref(),
        &format!("{base_path}.hardware"),
        &format!("{base_path}.multimodal"),
    )?;
    if let Some(multimodal) = &defaults.multimodal {
        validate_multimodal(multimodal, &format!("{base_path}.multimodal"))?;
    }
    if let Some(advanced) = &defaults.advanced {
        validate_advanced(advanced, &format!("{base_path}.advanced"))?;
    }
    Ok(())
}

pub(crate) fn validate_model_entry(
    model: &ModelConfigEntry,
    base_path: &str,
    gpu_assignment: GpuAssignment,
    defaults_hardware: Option<&HardwareConfig>,
) -> DiagnosticResult {
    let model_fit = merge_model_fit(
        model.model_fit.clone(),
        model.ctx_size,
        model.cache_type_k.clone(),
        model.cache_type_v.clone(),
        model.batch,
        model.ubatch,
        model.flash_attention,
    );
    let multimodal = merge_multimodal(model.multimodal.clone(), model.mmproj.clone());
    let hardware = merge_hardware(
        model.hardware.clone(),
        model.gpu_id.clone(),
        multimodal.as_ref().and_then(|config| config.mmproj.clone()),
        multimodal
            .as_ref()
            .and_then(|config| config.mmproj_offload.clone()),
    );
    let throughput = merge_throughput(model.throughput.clone(), model.parallel);

    if let Some(mmproj) = &model.mmproj {
        validate_non_empty(mmproj, &format!("{base_path}.multimodal.mmproj"))?;
    }
    if let Some(model_fit) = &model_fit {
        validate_model_fit(model_fit, &format!("{base_path}.model_fit"))?;
    }
    if let Some(hardware) = hardware.as_ref() {
        validate_hardware(hardware, &format!("{base_path}.hardware"), gpu_assignment)?;
    }
    if let Some(throughput) = &throughput {
        validate_throughput(throughput, &format!("{base_path}.throughput"))?;
    }
    if let Some(skippy) = &model.skippy {
        validate_skippy(skippy, &format!("{base_path}.skippy"))?;
    }
    if let Some(speculative) = &model.speculative {
        validate_speculative(speculative, &format!("{base_path}.speculative"))?;
    }
    if let Some(request_defaults) = &model.request_defaults {
        validate_request_defaults(request_defaults, &format!("{base_path}.request_defaults"))?;
    }
    validate_multimodal_pair(
        hardware.as_ref(),
        multimodal.as_ref(),
        &format!("{base_path}.hardware"),
        &format!("{base_path}.multimodal"),
    )?;
    if let Some(multimodal) = &multimodal {
        validate_multimodal(multimodal, &format!("{base_path}.multimodal"))?;
    }
    if let Some(advanced) = &model.advanced {
        validate_advanced(advanced, &format!("{base_path}.advanced"))?;
    }
    validate_gpu_assignment_constraints(
        hardware.as_ref(),
        defaults_hardware.and_then(|hardware| hardware.device.as_deref()),
        model
            .gpu_id_from_legacy_shim
            .then_some(model.gpu_id.as_deref())
            .flatten(),
        &format!("{base_path}.hardware.device"),
        gpu_assignment,
        true,
    )?;
    Ok(())
}

fn validate_model_fit(config: &ModelFitConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_u32_range(
        config.ctx_size,
        &format!("{base_path}.ctx_size"),
        1,
        1_000_000,
    )?;
    validate_optional_u32_range(config.batch, &format!("{base_path}.batch"), 1, 10_000_000)?;
    validate_optional_u32_range(config.ubatch, &format!("{base_path}.ubatch"), 1, 10_000_000)?;
    if let (Some(batch), Some(ubatch)) = (config.batch, config.ubatch)
        && ubatch > batch
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.ubatch"),
            format!("{base_path}.ubatch must be less than or equal to {base_path}.batch"),
        ));
    }
    validate_optional_kv_cache_type(
        config.cache_type_k.as_deref(),
        &format!("{base_path}.cache_type_k"),
    )?;
    validate_optional_kv_cache_type(
        config.cache_type_v.as_deref(),
        &format!("{base_path}.cache_type_v"),
    )?;
    validate_optional_enum(
        config.kv_cache_policy.as_deref(),
        &["auto", "quality", "balanced", "saver"],
        &format!("{base_path}.kv_cache_policy"),
    )?;
    validate_bool_or_auto(
        config.kv_offload.as_ref(),
        &format!("{base_path}.kv_offload"),
    )?;
    validate_bool_or_auto(
        config.kv_unified.as_ref(),
        &format!("{base_path}.kv_unified"),
    )?;
    validate_bool_or_auto(
        config.prompt_cache.as_ref(),
        &format!("{base_path}.prompt_cache"),
    )?;
    validate_bool_or_auto(
        config.context_shift.as_ref(),
        &format!("{base_path}.context_shift"),
    )?;
    if let Some(cache_idle_slots) = config.cache_idle_slots
        && cache_idle_slots > 0
        && matches!(config.prompt_cache, Some(BoolOrAuto::Bool(false)))
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.cache_idle_slots"),
            format!("{base_path}.cache_idle_slots requires {base_path}.prompt_cache = true"),
        ));
    }
    if let Some(prefix_cache) = &config.prefix_cache {
        validate_prefix_cache(prefix_cache, &format!("{base_path}.prefix_cache"))?;
    }
    if let (Some(keep_tokens), Some(ctx_size)) = (config.keep_tokens, config.ctx_size)
        && keep_tokens > ctx_size
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.keep_tokens"),
            format!("{base_path}.keep_tokens must be less than or equal to {base_path}.ctx_size"),
        ));
    }
    validate_optional_u32_range(
        config.keep_tokens,
        &format!("{base_path}.keep_tokens"),
        1,
        1_000_000,
    )?;
    validate_optional_u32_range(
        config.checkpoint_interval,
        &format!("{base_path}.checkpoint_interval"),
        1,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.checkpoint_count,
        &format!("{base_path}.checkpoint_count"),
        1,
        10_000_000,
    )?;
    validate_optional_path(
        config.lookup_cache_static.as_deref(),
        &format!("{base_path}.lookup_cache_static"),
    )?;
    validate_optional_path(
        config.lookup_cache_dynamic.as_deref(),
        &format!("{base_path}.lookup_cache_dynamic"),
    )?;
    Ok(())
}

fn validate_prefix_cache(config: &PrefixCacheConfig, base_path: &str) -> DiagnosticResult {
    if config.enabled == Some(false) {
        return Ok(());
    }
    if config.enabled == Some(true) {
        validate_optional_u32_range(
            config.max_entries,
            &format!("{base_path}.max_entries"),
            1,
            10_000_000,
        )?;
        validate_optional_u32_range(
            config.min_tokens,
            &format!("{base_path}.min_tokens"),
            1,
            10_000_000,
        )?;
        validate_optional_u32_range(
            config.shared_stride_tokens,
            &format!("{base_path}.shared_stride_tokens"),
            1,
            10_000_000,
        )?;
        validate_optional_u32_range(
            config.shared_record_limit,
            &format!("{base_path}.shared_record_limit"),
            1,
            10_000_000,
        )?;
    }
    validate_optional_enum(
        config.payload_mode.as_deref(),
        &["resident-kv", "kv-recurrent", "full-state", "auto"],
        &format!("{base_path}.payload_mode"),
    )?;
    Ok(())
}

fn validate_skippy(config: &SkippyConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_path(
        config.stage_model_path.as_deref(),
        &format!("{base_path}.stage_model_path"),
    )?;
    validate_optional_enum(
        config.source_policy.as_deref(),
        &["fallback", "local-required"],
        &format!("{base_path}.source_policy"),
    )?;
    if config.openai_frontend_mode.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.openai_frontend_mode"),
            format!("{base_path}.openai_frontend_mode is documented-rejected and must not be set"),
        ));
    }
    validate_optional_positive_u64(
        config.lifecycle_startup_timeout_ms,
        &format!("{base_path}.lifecycle_startup_timeout_ms"),
    )?;
    validate_optional_positive_u64(
        config.lifecycle_readiness_interval_ms,
        &format!("{base_path}.lifecycle_readiness_interval_ms"),
    )?;
    validate_optional_positive_u64(
        config.lifecycle_health_interval_ms,
        &format!("{base_path}.lifecycle_health_interval_ms"),
    )?;
    validate_optional_enum(
        config.prefill_chunking.as_deref(),
        &["auto", "fixed", "schedule", "adaptive-ramp"],
        &format!("{base_path}.prefill_chunking"),
    )?;
    if let Some(schedule) = &config.prefill_chunk_schedule {
        validate_non_empty(schedule, &format!("{base_path}.prefill_chunk_schedule"))?;
        for item in schedule.split(',') {
            let trimmed = item.trim();
            if trimmed.is_empty()
                || trimmed
                    .parse::<u32>()
                    .ok()
                    .filter(|value| *value > 0)
                    .is_none()
            {
                return Err(validation_diagnostic(
                    &format!("{base_path}.prefill_chunk_schedule"),
                    format!(
                        "{base_path}.prefill_chunk_schedule must contain only comma-separated positive integers"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn validate_speculative(config: &SpeculativeConfig, base_path: &str) -> DiagnosticResult {
    if let Some(strategy) = config.strategy.as_deref() {
        validate_non_empty(strategy, &format!("{base_path}.strategy"))?;
    }
    validate_optional_enum(
        config.mode.as_deref(),
        &["auto", "disabled", "draft"],
        &format!("{base_path}.mode"),
    )?;
    validate_model_identifier(
        config.draft_model.as_deref(),
        &format!("{base_path}.draft_model"),
        config.legacy_draft_model_path_used,
    )?;
    validate_hf_pair(
        config.draft_hf_repo.as_deref(),
        config.draft_hf_file.as_deref(),
        &format!("{base_path}.draft_hf_repo"),
        &format!("{base_path}.draft_hf_file"),
    )?;
    validate_optional_enum(
        config.draft_selection_policy.as_deref(),
        &["manual", "auto"],
        &format!("{base_path}.draft_selection_policy"),
    )?;
    validate_optional_enum(
        config.pairing_fault.as_deref(),
        &[
            "warn_disable",
            "fail-open",
            "fail-closed",
            "fail_open",
            "fail_closed",
        ],
        &format!("{base_path}.pairing_fault"),
    )?;
    validate_optional_u32_range(
        config.draft_min_tokens,
        &format!("{base_path}.draft_min_tokens"),
        0,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.draft_max_tokens,
        &format!("{base_path}.draft_max_tokens"),
        1,
        10_000_000,
    )?;
    if let (Some(min), Some(max)) = (config.draft_min_tokens, config.draft_max_tokens)
        && min > max
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.draft_min_tokens"),
            format!(
                "{base_path}.draft_min_tokens must be less than or equal to {base_path}.draft_max_tokens"
            ),
        ));
    }
    validate_probability(
        config.draft_acceptance_threshold,
        &format!("{base_path}.draft_acceptance_threshold"),
    )?;
    validate_probability(
        config.draft_split_probability,
        &format!("{base_path}.draft_split_probability"),
    )?;
    if let Some(gpu_layers) = config.draft_gpu_layers
        && gpu_layers < -1
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.draft_gpu_layers"),
            format!("{base_path}.draft_gpu_layers must be at least -1"),
        ));
    }
    validate_optional_positive_usize(config.draft_threads, &format!("{base_path}.draft_threads"))?;
    validate_optional_kv_cache_type(
        config.draft_cache_type_k.as_deref(),
        &format!("{base_path}.draft_cache_type_k"),
    )?;
    validate_optional_kv_cache_type(
        config.draft_cache_type_v.as_deref(),
        &format!("{base_path}.draft_cache_type_v"),
    )?;
    validate_optional_u32_range(
        config.ngram_min,
        &format!("{base_path}.ngram_min"),
        1,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.ngram_max,
        &format!("{base_path}.ngram_max"),
        1,
        10_000_000,
    )?;
    if let (Some(min), Some(max)) = (config.ngram_min, config.ngram_max)
        && max < min
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.ngram_max"),
            format!("{base_path}.ngram_max must be greater than or equal to {base_path}.ngram_min"),
        ));
    }
    validate_bool_or_auto(
        config.spec_default.as_ref(),
        &format!("{base_path}.spec_default"),
    )?;
    if config.mode.as_deref() == Some("draft")
        && config.draft_model.is_none()
        && config.draft_hf_repo.is_none()
        && config.draft_selection_policy.is_none()
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.draft_selection_policy"),
            format!(
                "{base_path}.draft_selection_policy must be set when {base_path}.mode = \"draft\" and no explicit draft model source is configured"
            ),
        ));
    }
    validate_speculative_proposer_controls(config, base_path)
}

fn validate_speculative_proposer_controls(
    config: &SpeculativeConfig,
    base_path: &str,
) -> DiagnosticResult {
    validate_optional_enum(
        config.ngram_proposer.as_deref(),
        &["cache", "suffix"],
        &format!("{base_path}.ngram_proposer"),
    )?;
    let suffix_selected = config.ngram_proposer.as_deref() == Some("suffix")
        || config.strategy.as_deref() == Some("ngram-suffix");
    if suffix_selected {
        if config.ngram_min.is_some_and(|min| min < 3) {
            return Err(validation_diagnostic(
                &format!("{base_path}.ngram_min"),
                format!("{base_path}.ngram_min must be at least 3 for the suffix proposer"),
            ));
        }
        if config.ngram_max.is_some_and(|max| max > 64) {
            return Err(validation_diagnostic(
                &format!("{base_path}.ngram_max"),
                format!("{base_path}.ngram_max must not exceed 64 for the suffix proposer"),
            ));
        }
    }
    validate_optional_u32_range(
        config.ngram_max_proposal_tokens,
        &format!("{base_path}.ngram_max_proposal_tokens"),
        1,
        10_000_000,
    )?;
    validate_extension_controls(config, base_path)?;
    validate_native_mtp_controls(config, base_path)?;
    validate_verify_window_controls(config, base_path)
}

fn validate_extension_controls(config: &SpeculativeConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_u32_range(
        config.extension_max_tokens,
        &format!("{base_path}.extension_max_tokens"),
        1,
        10_000_000,
    )
}

fn validate_native_mtp_controls(config: &SpeculativeConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_u32_range(
        config.native_mtp_reject_cooldown_tokens,
        &format!("{base_path}.native_mtp_reject_cooldown_tokens"),
        0,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.native_mtp_suppress_cooldown_draft_limit,
        &format!("{base_path}.native_mtp_suppress_cooldown_draft_limit"),
        0,
        10_000_000,
    )
}

fn validate_verify_window_controls(
    config: &SpeculativeConfig,
    base_path: &str,
) -> DiagnosticResult {
    validate_optional_u32_range(
        config.verify_window_min_tokens,
        &format!("{base_path}.verify_window_min_tokens"),
        1,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.verify_window_max_tokens,
        &format!("{base_path}.verify_window_max_tokens"),
        1,
        10_000_000,
    )?;
    if let (Some(min), Some(max)) = (
        config.verify_window_min_tokens,
        config.verify_window_max_tokens,
    ) && min > max
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.verify_window_min_tokens"),
            format!(
                "{base_path}.verify_window_min_tokens must be less than or equal to {base_path}.verify_window_max_tokens"
            ),
        ));
    }
    validate_optional_u32_range(
        config.verify_window_pipeline_depth,
        &format!("{base_path}.verify_window_pipeline_depth"),
        1,
        u32::try_from(MAX_VERIFY_WINDOW_PIPELINE_DEPTH).expect("verify depth limit fits u32"),
    )?;
    validate_optional_u32_range(
        config.verify_window_runahead_tokens,
        &format!("{base_path}.verify_window_runahead_tokens"),
        // Zero is the documented fixed-depth sentinel, so a model-level block
        // can switch run-ahead back off when the global defaults enable it.
        0,
        u32::try_from(MAX_VERIFY_WINDOW_RUNAHEAD_TOKENS).expect("runahead limit fits u32"),
    )
}

fn validate_request_defaults(config: &RequestDefaultsConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_u32_range(
        config.max_tokens,
        &format!("{base_path}.max_tokens"),
        1,
        10_000_000,
    )?;
    if let Some(stop) = &config.stop {
        match stop {
            StringOrStringList::String(value) => {
                validate_non_empty(value, &format!("{base_path}.stop"))?
            }
            StringOrStringList::List(values) => {
                validate_string_list(values, &format!("{base_path}.stop"))?
            }
        }
    }
    validate_request_sampling_defaults(config, base_path)?;
    validate_request_chat_defaults(config, base_path)
}

fn validate_request_sampling_defaults(
    config: &RequestDefaultsConfig,
    base_path: &str,
) -> DiagnosticResult {
    validate_non_negative_f64(config.temperature, &format!("{base_path}.temperature"))?;
    validate_probability(config.top_p, &format!("{base_path}.top_p"))?;
    if let Some(top_k) = config.top_k
        && top_k < 0
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.top_k"),
            format!("{base_path}.top_k must be greater than or equal to 0"),
        ));
    }
    validate_probability(config.min_p, &format!("{base_path}.min_p"))?;
    validate_probability(config.typical_p, &format!("{base_path}.typical_p"))?;
    if let Some(value) = config.top_nsigma
        && (!value.is_finite() || value < -1.0)
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.top_nsigma"),
            format!("{base_path}.top_nsigma must be finite and greater than or equal to -1"),
        ));
    }
    validate_non_negative_f64(
        config.dynatemp_range,
        &format!("{base_path}.dynatemp_range"),
    )?;
    validate_non_negative_f64(
        config.dynatemp_exponent,
        &format!("{base_path}.dynatemp_exponent"),
    )?;
    validate_non_negative_f64(
        config.repeat_penalty,
        &format!("{base_path}.repeat_penalty"),
    )?;
    if let Some(repeat_last_n) = config.repeat_last_n
        && repeat_last_n < -1
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.repeat_last_n"),
            format!("{base_path}.repeat_last_n must be greater than or equal to -1"),
        ));
    }
    validate_non_negative_f64(
        config.presence_penalty,
        &format!("{base_path}.presence_penalty"),
    )?;
    validate_non_negative_f64(
        config.frequency_penalty,
        &format!("{base_path}.frequency_penalty"),
    )?;
    if let Some(dry) = &config.dry {
        let multiplier_path = format!("{base_path}.dry.multiplier");
        if dry.multiplier.is_some_and(|value| !value.is_finite()) {
            return Err(validation_diagnostic(
                &multiplier_path,
                format!("{multiplier_path} must be finite"),
            ));
        }
        validate_non_negative_f64(dry.multiplier, &multiplier_path)?;

        let base_value_path = format!("{base_path}.dry.base");
        if dry.base.is_some_and(|value| !value.is_finite()) {
            return Err(validation_diagnostic(
                &base_value_path,
                format!("{base_value_path} must be finite"),
            ));
        }
        validate_positive_f64(dry.base, &base_value_path)?;
        if dry.allowed_length.is_some_and(|value| value < 0) {
            return Err(validation_diagnostic(
                &format!("{base_path}.dry.allowed_length"),
                format!("{base_path}.dry.allowed_length must be greater than or equal to 0"),
            ));
        }
        if dry.penalty_last_n.is_some_and(|value| value < -1) {
            return Err(validation_diagnostic(
                &format!("{base_path}.dry.penalty_last_n"),
                format!("{base_path}.dry.penalty_last_n must be greater than or equal to -1"),
            ));
        }
        if let Some(breakers) = &dry.sequence_breakers {
            if breakers.iter().any(String::is_empty) {
                return Err(validation_diagnostic(
                    &format!("{base_path}.dry.sequence_breakers"),
                    format!("{base_path}.dry.sequence_breakers must not contain empty values"),
                ));
            }
            if breakers.len() > 8 || breakers.iter().any(|value| value.len() >= 16) {
                return Err(validation_diagnostic(
                    &format!("{base_path}.dry.sequence_breakers"),
                    format!(
                        "{base_path}.dry.sequence_breakers supports at most 8 values shorter than 16 bytes"
                    ),
                ));
            }
        }
    }
    if let Some(xtc) = &config.xtc {
        validate_probability(xtc.probability, &format!("{base_path}.xtc.probability"))?;
        validate_probability(xtc.threshold, &format!("{base_path}.xtc.threshold"))?;
    }
    if let Some(mode) = &config.mirostat_mode {
        match mode {
            IntegerOrString::Integer(value) if *value == 1 || *value == 2 => {}
            IntegerOrString::String(value) => validate_allowed(
                value,
                &["disabled", "1", "2"],
                &format!("{base_path}.mirostat_mode"),
            )?,
            _ => {
                return Err(validation_diagnostic(
                    &format!("{base_path}.mirostat_mode"),
                    format!("{base_path}.mirostat_mode must be one of: disabled, 1, 2"),
                ));
            }
        }
    }
    validate_positive_f64(
        config.mirostat_entropy,
        &format!("{base_path}.mirostat_entropy"),
    )?;
    validate_positive_f64(
        config.mirostat_learning_rate,
        &format!("{base_path}.mirostat_learning_rate"),
    )?;
    if let Some(samplers) = &config.samplers {
        validate_string_list(samplers, &format!("{base_path}.samplers"))?;
        if samplers.len() > 16 {
            return Err(validation_diagnostic(
                &format!("{base_path}.samplers"),
                format!("{base_path}.samplers supports at most 16 entries"),
            ));
        }
        for sampler in samplers {
            validate_allowed(
                sampler,
                &[
                    "penalties",
                    "dry",
                    "top_n_sigma",
                    "top_k",
                    "typical_p",
                    "typ_p",
                    "top_p",
                    "min_p",
                    "xtc",
                    "temperature",
                    "temp",
                ],
                &format!("{base_path}.samplers"),
            )?;
        }
    }
    if let Some(sequence) = &config.sampler_sequence
        && let Some(invalid) = sequence.chars().find(|value| {
            !value.is_whitespace()
                && !matches!(value, 'e' | 'd' | 's' | 'k' | 'y' | 'p' | 'm' | 'x' | 't')
        })
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.sampler_sequence"),
            format!("{base_path}.sampler_sequence contains unsupported sampler code {invalid:?}"),
        ));
    }
    if config.backend_sampling.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.backend_sampling"),
            format!("{base_path}.backend_sampling is documented-rejected and must not be set"),
        ));
    }
    Ok(())
}

fn validate_request_chat_defaults(
    config: &RequestDefaultsConfig,
    base_path: &str,
) -> DiagnosticResult {
    validate_optional_enum(
        config.reasoning_format.as_deref(),
        &["auto", "none", "deepseek", "deepseek-legacy", "hidden"],
        &format!("{base_path}.reasoning_format"),
    )?;
    if let Some(reasoning_enabled) = &config.reasoning_enabled {
        match reasoning_enabled {
            ReasoningEnabled::Bool(_) => {}
            ReasoningEnabled::String(value) => validate_allowed(
                value,
                &["auto", "off", "on"],
                &format!("{base_path}.reasoning_enabled"),
            )?,
        }
    }
    if let Some(reasoning_budget) = &config.reasoning_budget {
        match reasoning_budget {
            ReasoningBudget::Integer(_) => {}
            ReasoningBudget::String(value) => validate_allowed(
                value,
                &["auto", "low", "medium", "high"],
                &format!("{base_path}.reasoning_budget"),
            )?,
        }
    }
    validate_optional_path(
        config.chat_template_file.as_deref(),
        &format!("{base_path}.chat_template_file"),
    )?;
    if config.chat_template.is_some() && config.chat_template_file.is_some() {
        return Err(validation_diagnostic(
            base_path,
            format!(
                "{base_path}.chat_template and {base_path}.chat_template_file cannot both be set"
            ),
        ));
    }
    if config
        .chat_template_kwargs
        .as_ref()
        .is_some_and(|value| !value.is_table())
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.chat_template_kwargs"),
            format!("{base_path}.chat_template_kwargs must be a table"),
        ));
    }
    if config
        .prefill_assistant
        .as_ref()
        .is_some_and(|value| !value.is_str() && !value.is_table())
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.prefill_assistant"),
            format!("{base_path}.prefill_assistant must be a string or table"),
        ));
    }
    if config.grammar.as_ref().is_some_and(|value| !value.is_str()) {
        return Err(validation_diagnostic(
            &format!("{base_path}.grammar"),
            format!("{base_path}.grammar must be a string"),
        ));
    }
    if config
        .json_schema
        .as_ref()
        .is_some_and(|value| !value.is_table())
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.json_schema"),
            format!("{base_path}.json_schema must be a table"),
        ));
    }
    if config.grammar.is_some() && config.json_schema.is_some() {
        return Err(validation_diagnostic(
            base_path,
            format!("{base_path}.grammar and {base_path}.json_schema cannot both be set"),
        ));
    }
    if config.logprobs.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.logprobs"),
            format!("{base_path}.logprobs is documented-rejected and must not be set"),
        ));
    }
    Ok(())
}

fn validate_multimodal_pair(
    hardware: Option<&HardwareConfig>,
    multimodal: Option<&MultimodalConfig>,
    hardware_path: &str,
    multimodal_path: &str,
) -> DiagnosticResult {
    if let (Some(hardware), Some(multimodal)) = (hardware, multimodal) {
        if let (Some(hardware_mmproj), Some(multimodal_mmproj)) =
            (hardware.mmproj.as_deref(), multimodal.mmproj.as_deref())
            && hardware_mmproj != multimodal_mmproj
        {
            return Err(validation_diagnostic(
                &format!("{multimodal_path}.mmproj"),
                format!(
                    "{multimodal_path}.mmproj must match {hardware_path}.mmproj when both are set"
                ),
            ));
        }
        if let (Some(hardware_offload), Some(multimodal_offload)) = (
            hardware.mmproj_offload.as_ref(),
            multimodal.mmproj_offload.as_ref(),
        ) && hardware_offload != multimodal_offload
        {
            return Err(validation_diagnostic(
                &format!("{multimodal_path}.mmproj_offload"),
                format!(
                    "{multimodal_path}.mmproj_offload must match {hardware_path}.mmproj_offload when both are set"
                ),
            ));
        }
    }
    Ok(())
}

fn validate_multimodal(config: &MultimodalConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_path(config.mmproj.as_deref(), &format!("{base_path}.mmproj"))?;
    validate_optional_http_url(
        config.mmproj_url.as_deref(),
        &format!("{base_path}.mmproj_url"),
    )?;
    validate_bool_or_auto(
        config.mmproj_offload.as_ref(),
        &format!("{base_path}.mmproj_offload"),
    )?;
    validate_optional_u32_range(
        config.image_min_tokens,
        &format!("{base_path}.image_min_tokens"),
        1,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.image_max_tokens,
        &format!("{base_path}.image_max_tokens"),
        1,
        10_000_000,
    )?;
    if let (Some(min), Some(max)) = (config.image_min_tokens, config.image_max_tokens)
        && min > max
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.image_min_tokens"),
            format!(
                "{base_path}.image_min_tokens must be less than or equal to {base_path}.image_max_tokens"
            ),
        ));
    }
    if config.image_marker.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.image_marker"),
            format!(
                "{base_path}.image_marker is not supported because mtmd removed custom image markers; use {base_path}.media_marker"
            ),
        ));
    }
    if config.media_marker.as_deref().is_some_and(str::is_empty) {
        return Err(validation_diagnostic(
            &format!("{base_path}.media_marker"),
            format!("{base_path}.media_marker must not be empty"),
        ));
    }
    validate_optional_u32_range(
        config.batch_max_tokens,
        &format!("{base_path}.batch_max_tokens"),
        1,
        i32::MAX as u32,
    )?;
    if let Some(policy) = config.glm_dsa_policy.as_deref()
        && !matches!(policy, "auto" | "v1")
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.glm_dsa_policy"),
            format!("{base_path}.glm_dsa_policy must be \"auto\" or \"v1\""),
        ));
    }
    validate_optional_u32_range(
        config.generation_signal_window,
        &format!("{base_path}.generation_signal_window"),
        1,
        4096,
    )?;
    if config.embeddings.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.embeddings"),
            format!("{base_path}.embeddings is documented-rejected and must not be set"),
        ));
    }
    if config.reranking.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.reranking"),
            format!("{base_path}.reranking is documented-rejected and must not be set"),
        ));
    }
    if config.pooling.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.pooling"),
            format!("{base_path}.pooling is documented-rejected and must not be set"),
        ));
    }
    if config.vocoder.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.vocoder"),
            format!("{base_path}.vocoder is documented-rejected and must not be set"),
        ));
    }
    Ok(())
}

fn validate_advanced(config: &AdvancedConfig, base_path: &str) -> DiagnosticResult {
    if let Some(server) = &config.server {
        if server.host.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.host"),
                format!("{base_path}.server.host is documented-rejected and must not be set"),
            ));
        }
        if server.port.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.port"),
                format!("{base_path}.server.port is documented-rejected and must not be set"),
            ));
        }
        if server.reuse_port.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.reuse_port"),
                format!("{base_path}.server.reuse_port is documented-rejected and must not be set"),
            ));
        }
        if server.timeout.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.timeout"),
                format!("{base_path}.server.timeout is documented-rejected and must not be set"),
            ));
        }
        if server.metrics.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.metrics"),
                format!("{base_path}.server.metrics is documented-rejected and must not be set"),
            ));
        }
        if server.slots.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.slots"),
                format!("{base_path}.server.slots is documented-rejected and must not be set"),
            ));
        }
        if server.props.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.props"),
                format!("{base_path}.server.props is documented-rejected and must not be set"),
            ));
        }
        if server.api_prefix.is_some() {
            return Err(validation_diagnostic(
                &format!("{base_path}.server.api_prefix"),
                format!("{base_path}.server.api_prefix is documented-rejected and must not be set"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostic::legacy_validation_error_text;
    use crate::{DrySamplingConfig, MeshConfig, validate_config, validate_config_diagnostics};

    #[test]
    fn speculative_strategy_allows_package_declared_names() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.speculative]
strategy = "mystery-oracle"
"#,
        )
        .expect("config should parse before validation");

        validate_config(&config)
            .expect("package strategy names are validated after package resolution");
        assert!(validate_config_diagnostics(&config).is_empty());
    }

    #[test]
    fn speculative_strategy_raw_name_is_deferred_to_package_resolution() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("package-strategy".to_string()),
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        validate_config(&config)
            .expect("package strategy names are validated after package resolution");
        assert!(validate_config_diagnostics(&config).is_empty());
    }

    #[test]
    fn verify_window_pipeline_depth_matches_native_retention_bound() {
        let accepted: MeshConfig = toml::from_str(
            r#"
[defaults.speculative]
verify_window_pipeline_depth = 64
"#,
        )
        .expect("bounded depth should parse");
        validate_config(&accepted).expect("native retention boundary should validate");

        let rejected: MeshConfig = toml::from_str(
            r#"
[defaults.speculative]
verify_window_pipeline_depth = 65
"#,
        )
        .expect("out-of-range depth should parse before validation");
        let diagnostics = validate_config_diagnostics(&rejected);
        let text = legacy_validation_error_text(&diagnostics);

        assert!(text.contains("verify_window_pipeline_depth"));
        assert!(
            text.contains("between 1 and 64"),
            "unexpected diagnostic: {text}"
        );
    }

    #[test]
    fn duplicate_model_with_same_profile_is_rejected() {
        let config: MeshConfig = toml::from_str(
            r#"
defaults.runtime = "metal"

[[models]]
model = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
profile = "gaming"

[[models]]
model = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
profile = "gaming"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            text.contains("duplicate model entry"),
            "expected duplicate model error, got: {text}"
        );
        assert!(
            text.contains("models[0]"),
            "expected reference to models[0], got: {text}"
        );
        assert!(
            text.contains("models[1]"),
            "expected reference to models[1], got: {text}"
        );
    }

    #[test]
    fn duplicate_model_without_profile_is_rejected() {
        let config: MeshConfig = toml::from_str(
            r#"
defaults.runtime = "metal"

[[models]]
model = "my-model"

[[models]]
model = "my-model"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            text.contains("duplicate model entry"),
            "expected duplicate model error, got: {text}"
        );
        assert!(
            text.contains("and default profile"),
            "expected 'and default profile' in error, got: {text}"
        );
    }

    #[test]
    fn duplicate_public_served_identity_is_rejected() {
        let public_identity = "Qwen/Qwen3-8B-GGUF:Q4_K_M";
        let config: MeshConfig = toml::from_str(&format!(
            r#"
[[models]]
model = "{public_identity}"
ctx_size = 4096

[[models]]
model = "{public_identity}"
ctx_size = 8192
"#,
        ))
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);

        assert!(
            !text.contains("duplicate model entry"),
            "fixture must not trigger the existing model/profile duplicate check: {text}"
        );
        assert!(
            text.contains("duplicate public served identity"),
            "expected public identity collision, got: {text}"
        );
        assert!(text.contains("models[0]"), "missing first row: {text}");
        assert!(text.contains("models[1]"), "missing second row: {text}");
        assert!(
            text.contains(public_identity),
            "missing colliding public identity: {text}"
        );
    }

    #[test]
    fn load_behavior_changes_produce_distinct_effective_profiles() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.hardware]
repack = true

[[models]]
model = "same-model"

[[models]]
model = "same-model"
[models.hardware]
repack = false
"#,
        )
        .expect("config parses");

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);

        assert!(
            !text.contains("duplicate model entry"),
            "different effective load behavior must create distinct profiles: {text}"
        );
    }

    #[test]
    fn duplicate_explicit_served_aliases_are_rejected() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "canonical/first"
[models.advanced.server]
alias = "public-model"

[[models]]
model = "canonical/second"
[models.advanced.server]
alias = "public-model"
"#,
        )
        .expect("config should parse before validation");

        let text = legacy_validation_error_text(&validate_config_diagnostics(&config));
        assert!(text.contains("duplicate served model identity"), "{text}");
        assert!(text.contains("public-model"), "{text}");
    }

    #[test]
    fn duplicate_inherited_served_aliases_are_rejected() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.advanced.server]
alias = "default-public-model"

[[models]]
model = "canonical/first"

[[models]]
model = "canonical/second"
"#,
        )
        .expect("config should parse before validation");

        let text = legacy_validation_error_text(&validate_config_diagnostics(&config));
        assert!(text.contains("duplicate served model identity"), "{text}");
        assert!(text.contains("default-public-model"), "{text}");
    }

    #[test]
    fn served_alias_collisions_compare_trimmed_names() {
        let config: MeshConfig = toml::from_str(
            r#"
[[models]]
model = "canonical/first"
[models.advanced.server]
alias = " public-model "

[[models]]
model = "canonical/second"
[models.advanced.server]
alias = "public-model"
"#,
        )
        .expect("config should parse before validation");

        let text = legacy_validation_error_text(&validate_config_diagnostics(&config));
        assert!(text.contains("duplicate served model identity"), "{text}");
    }

    #[test]
    fn served_alias_collisions_include_defaults_merged_profiles() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.model_fit]
ctx_size = 8192

[[models]]
model = "canonical/first"
[models.model_fit]
ctx_size = 8192
[models.advanced.server]
alias = "public-model"

[[models]]
model = "canonical/second"
[models.advanced.server]
alias = "public-model"
"#,
        )
        .expect("config should parse before validation");

        let text = legacy_validation_error_text(&validate_config_diagnostics(&config));
        assert!(text.contains("duplicate served model identity"), "{text}");
    }

    #[test]
    fn model_alias_override_avoids_inherited_alias_collision() {
        let config: MeshConfig = toml::from_str(
            r#"
[defaults.advanced.server]
alias = "default-public-model"

[[models]]
model = "canonical/first"

[[models]]
model = "canonical/second"
[models.advanced.server]
alias = "second-public-model"
"#,
        )
        .expect("config should parse before validation");

        validate_config(&config).expect("effective served identities are distinct");
    }

    #[test]
    fn draft_model_rejects_bare_path_without_colon() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("mtp".to_string()),
                    draft_model: Some("/models/draft.gguf".to_string()),
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            text.contains("must be a model identifier"),
            "expected identifier validation error, got: {text}"
        );
    }

    #[test]
    fn legacy_draft_model_path_skips_identifier_validation() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("mtp".to_string()),
                    draft_model: Some("/models/draft.gguf".to_string()),
                    legacy_draft_model_path_used: true,
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            !text.contains("must be a model identifier"),
            "expected no identifier error when legacy path used, got: {text}"
        );
    }

    #[test]
    fn draft_model_accepts_identifier_with_colon() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("mtp".to_string()),
                    draft_model: Some("Qwen/Qwen3-0.6B:Q4_K_M".to_string()),
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            !text.contains("must be a model identifier"),
            "expected no identifier error for valid identifier, got: {text}"
        );
    }

    #[test]
    fn speculative_hf_sources_reject_parent_directory_components() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    draft_hf_repo: Some(" ../outside/draft".to_string()),
                    draft_hf_file: Some(" ../draft.gguf".to_string()),
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(text.contains("must not contain absolute or parent-directory path components"));
    }

    #[test]
    fn speculative_hf_sources_trim_whitespace_before_rejecting_absolute_paths() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    draft_hf_repo: Some(" /absolute/draft".to_string()),
                    draft_hf_file: Some(" /draft.gguf".to_string()),
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(text.contains("must not contain absolute or parent-directory path components"));
    }

    #[test]
    fn draft_model_identifier_rejects_parent_directory_components() {
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    draft_model: Some("../outside/draft:Q4_K_M".to_string()),
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(text.contains("must not contain absolute or parent-directory path components"));
    }

    #[test]
    fn legacy_draft_model_path_emits_migration_warning() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.speculative]
strategy = "mtp"
draft_model_path = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let alias_diag = diagnostics.iter().find(|d| {
            d.code == crate::diagnostic::ConfigDiagnosticCode::AliasApplied
                && d.message.contains("draft_model_path")
        });
        assert!(
            alias_diag.is_some(),
            "expected legacy alias warning for draft_model_path, got diagnostics: {:?}",
            diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn legacy_draft_model_path_bare_path_suppresses_migration_warning() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.speculative]
strategy = "mtp"
draft_model_path = "/models/draft.gguf"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let alias_diag = diagnostics.iter().find(|d| {
            d.code == crate::diagnostic::ConfigDiagnosticCode::AliasApplied
                && d.message.contains("draft_model_path")
        });
        assert!(
            alias_diag.is_none(),
            "bare path draft_model_path should not emit migration warning, got: {:?}",
            diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn legacy_draft_model_path_windows_style_absolute_suppresses_migration_warning() {
        // The previous `contains(':')` heuristic falsely fired for
        // Windows-style absolute paths like `C:/models/draft.gguf` because
        // they contain a `:` after the drive letter. The fix requires the
        // colon quantization marker to follow the last `/`, so the path-like
        // value is no longer mistaken for an identifier.
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.speculative]
strategy = "mtp"
draft_model_path = "C:/models/draft.gguf"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let alias_diag = diagnostics.iter().find(|d| {
            d.code == crate::diagnostic::ConfigDiagnosticCode::AliasApplied
                && d.message.contains("draft_model_path")
        });
        assert!(
            alias_diag.is_none(),
            "Windows-style absolute path draft_model_path should not emit migration warning, got: {:?}",
            diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn legacy_draft_model_path_rejects_nul_bytes() {
        // Legacy-path values should not bypass `validate_path_chars`. A NUL
        // byte inside a `draft_model_path` value must be rejected even when
        // `legacy_draft_model_path_used` is true.
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("mtp".to_string()),
                    draft_model: Some("bad\0path".to_string()),
                    legacy_draft_model_path_used: true,
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            text.contains("must not contain NUL bytes"),
            "expected NUL-byte rejection on legacy path, got: {text}"
        );
    }

    #[test]
    fn legacy_draft_model_path_rejects_control_characters() {
        // Control characters must also be rejected on legacy-path values.
        let config = MeshConfig {
            defaults: Some(ModelConfigDefaults {
                speculative: Some(SpeculativeConfig {
                    strategy: Some("mtp".to_string()),
                    draft_model: Some("bad\u{0001}path".to_string()),
                    legacy_draft_model_path_used: true,
                    ..SpeculativeConfig::default()
                }),
                ..ModelConfigDefaults::default()
            }),
            ..MeshConfig::default()
        };

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            text.contains("must not contain control characters"),
            "expected control-character rejection on legacy path, got: {text}"
        );
    }

    #[test]
    fn different_models_with_different_profiles_are_allowed() {
        let config: MeshConfig = toml::from_str(
            r#"
defaults.runtime = "metal"

[[models]]
model = "Qwen/Qwen3-8B-GGUF:Q4_K_M"
ctx_size = 4096

[[models]]
model = "Qwen/Qwen3-14B-GGUF:Q4_K_M"
ctx_size = 8192
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let text = legacy_validation_error_text(&diagnostics);
        assert!(
            !text.contains("duplicate model entry"),
            "expected distinct model/profile rows to avoid model duplicate errors, got: {text}"
        );
        assert!(
            !text.contains("duplicate public served identity"),
            "expected distinct model/profile rows to avoid identity collisions, got: {text}"
        );
    }

    #[test]
    fn suffix_proposer_rejects_matches_shorter_than_its_seed() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.speculative]
strategy = "mtp"
ngram_proposer = "suffix"
ngram_min = 2
ngram_max = 32
ngram_max_proposal_tokens = 16
"#,
        )
        .expect("config should parse before validation");

        let text = legacy_validation_error_text(&validate_config_diagnostics(&config));
        assert!(text.contains("ngram_min must be at least 3"), "{text}");
    }

    #[test]
    fn suffix_proposer_rejects_windows_above_runtime_limit() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.speculative]
strategy = "mtp"
ngram_proposer = "suffix"
ngram_min = 5
ngram_max = 65
ngram_max_proposal_tokens = 16
"#,
        )
        .expect("config should parse before validation");

        let text = legacy_validation_error_text(&validate_config_diagnostics(&config));
        assert!(text.contains("ngram_max must not exceed 64"), "{text}");
    }

    #[test]
    fn standalone_suffix_strategy_uses_suffix_validation_without_a_redundant_proposer_key() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.speculative]
strategy = "ngram-suffix"
ngram_min = 2
ngram_max = 65
ngram_max_proposal_tokens = 16
"#,
        )
        .expect("config should parse before validation");

        let text = legacy_validation_error_text(&validate_config_diagnostics(&config));
        assert!(text.contains("ngram_min must be at least 3"), "{text}");
    }

    #[test]
    fn dry_sequence_breakers_accept_whitespace_delimiters_but_reject_empty_values() {
        let accepted: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.request_defaults.dry]
sequence_breakers = ["\n", " ", ":"]
"#,
        )
        .expect("DRY sequence breakers should parse");
        validate_config(&accepted).expect("whitespace delimiters should remain executable");

        let rejected: MeshConfig = toml::from_str(
            r#"
version = 1

[defaults.request_defaults.dry]
sequence_breakers = [""]
"#,
        )
        .expect("empty DRY sequence breaker should parse before validation");
        let text = legacy_validation_error_text(&validate_config_diagnostics(&rejected));
        assert!(text.contains("must not contain empty values"), "{text}");
    }

    #[test]
    fn dry_multiplier_and_base_reject_non_finite_values() {
        for (field, dry) in [
            (
                "multiplier",
                DrySamplingConfig {
                    multiplier: Some(f64::NAN),
                    ..DrySamplingConfig::default()
                },
            ),
            (
                "base",
                DrySamplingConfig {
                    base: Some(f64::INFINITY),
                    ..DrySamplingConfig::default()
                },
            ),
        ] {
            let config = MeshConfig {
                defaults: Some(ModelConfigDefaults {
                    request_defaults: Some(RequestDefaultsConfig {
                        dry: Some(dry),
                        ..RequestDefaultsConfig::default()
                    }),
                    ..ModelConfigDefaults::default()
                }),
                ..MeshConfig::default()
            };
            let text = legacy_validation_error_text(&validate_config_diagnostics(&config));
            assert!(
                text.contains(&format!("defaults.request_defaults.dry.{field}")),
                "missing DRY {field} path: {text}"
            );
            assert!(
                text.contains("must be finite"),
                "missing finite diagnostic: {text}"
            );
        }
    }
}
