use mesh_llm_config::{
    FlashAttentionType, HardwareConfig, IntegerOrString, MeshConfig, ModelConfigDefaults,
    ModelConfigEntry, ModelFitConfig,
};

use super::*;

#[test]
fn gpu_tune_recommends_stable_defaults() {
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &MeshConfig::default(),
        target: &recommendation_target(false),
        metadata: &sample_metadata(8 * gib(), 32, 131_072, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    assert_applied_kv(&plan, TuneField::CacheTypeK, TuneKvCacheType::Q8_0);
    assert_applied_kv(&plan, TuneField::CacheTypeV, TuneKvCacheType::Q8_0);
    assert_applied_flash_attention(&plan, TuneFlashAttentionValue::Enabled);
    assert_applied_context(&plan, 131_072);
    assert_applied_batch(&plan, 512);
    assert_applied_ubatch(&plan, 512);
    assert_applied_gpu_layers(&plan, TuneGpuLayersValue::All);
    assert_applied_fit_target(&plan, 22 * 1024);
}

#[test]
fn gpu_tune_uses_q4_policy_for_large_models() {
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &MeshConfig::default(),
        target: &recommendation_target(false),
        metadata: &sample_metadata(60 * gib(), 80, 65_536, 0),
        hardware: &gpu_hardware(96 * gib()),
        survey: &survey_with_gpu(96 * gib(), 128 * gib()),
    });

    assert_applied_kv(&plan, TuneField::CacheTypeK, TuneKvCacheType::Q4_0);
    assert_applied_kv(&plan, TuneField::CacheTypeV, TuneKvCacheType::Q4_0);
}

#[test]
fn gpu_tune_preserves_explicit_per_model_values() {
    let config = MeshConfig {
        models: vec![ModelConfigEntry {
            model: "hf://mesh/example.gguf".to_string(),
            model_fit: Some(ModelFitConfig {
                ctx_size: Some(8192),
                batch: Some(256),
                cache_type_k: Some("f16".to_string()),
                cache_type_v: Some("f16".to_string()),
                flash_attention: Some(FlashAttentionType::Disabled),
                ..ModelFitConfig::default()
            }),
            hardware: Some(HardwareConfig {
                gpu_layers: Some(IntegerOrString::Integer(12)),
                fit_target_mib: Some(10_240),
                ..HardwareConfig::default()
            }),
            ..ModelConfigEntry::default()
        }],
        ..MeshConfig::default()
    };
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &config,
        target: &recommendation_target(true),
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    assert_preserved(
        &plan,
        TuneField::CacheTypeK,
        "models[].model_fit.cache_type_k",
    );
    assert_preserved(
        &plan,
        TuneField::CacheTypeV,
        "models[].model_fit.cache_type_v",
    );
    assert_preserved(
        &plan,
        TuneField::FlashAttention,
        "models[].model_fit.flash_attention",
    );
    assert_preserved(&plan, TuneField::CtxSize, "models[].model_fit.ctx_size");
    assert_preserved(&plan, TuneField::Batch, "models[].model_fit.batch");
    assert_preserved(&plan, TuneField::GpuLayers, "models[].hardware.gpu_layers");
    assert_preserved(
        &plan,
        TuneField::FitTargetMib,
        "models[].hardware.fit_target_mib",
    );
}

#[test]
fn gpu_tune_preserves_effective_defaults_values() {
    let config = MeshConfig {
        defaults: Some(ModelConfigDefaults {
            model_fit: Some(ModelFitConfig {
                ctx_size: Some(16_384),
                batch: Some(384),
                cache_type_k: Some("q8_0".to_string()),
                cache_type_v: Some("q8_0".to_string()),
                ..ModelFitConfig::default()
            }),
            hardware: Some(HardwareConfig {
                gpu_layers: Some(IntegerOrString::String("auto".to_string())),
                fit_target_mib: Some(12_288),
                ..HardwareConfig::default()
            }),
            ..ModelConfigDefaults::default()
        }),
        models: vec![ModelConfigEntry {
            model: "hf://mesh/example.gguf".to_string(),
            ..ModelConfigEntry::default()
        }],
        ..MeshConfig::default()
    };
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &config,
        target: &recommendation_target(true),
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    assert_preserved(
        &plan,
        TuneField::CacheTypeK,
        "defaults.model_fit.cache_type_k",
    );
    assert_preserved(
        &plan,
        TuneField::CacheTypeV,
        "defaults.model_fit.cache_type_v",
    );
    assert_preserved(&plan, TuneField::CtxSize, "defaults.model_fit.ctx_size");
    assert_preserved(&plan, TuneField::Batch, "defaults.model_fit.batch");
    assert_preserved(&plan, TuneField::GpuLayers, "defaults.hardware.gpu_layers");
    assert_preserved(
        &plan,
        TuneField::FitTargetMib,
        "defaults.hardware.fit_target_mib",
    );
}

#[test]
fn gpu_tune_replace_existing_allows_shadowing_defaults() {
    let config = MeshConfig {
        defaults: Some(ModelConfigDefaults {
            model_fit: Some(ModelFitConfig {
                ctx_size: Some(8192),
                cache_type_k: Some("f16".to_string()),
                cache_type_v: Some("f16".to_string()),
                ..ModelFitConfig::default()
            }),
            ..ModelConfigDefaults::default()
        }),
        models: vec![ModelConfigEntry {
            model: "hf://mesh/example.gguf".to_string(),
            ..ModelConfigEntry::default()
        }],
        ..MeshConfig::default()
    };
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ReplaceExisting,
        config: &config,
        target: &recommendation_target(true),
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    assert_applied_kv(&plan, TuneField::CacheTypeK, TuneKvCacheType::Q8_0);
    assert_applied_kv(&plan, TuneField::CacheTypeV, TuneKvCacheType::Q8_0);
    assert_applied_context(&plan, 65_536);
}
