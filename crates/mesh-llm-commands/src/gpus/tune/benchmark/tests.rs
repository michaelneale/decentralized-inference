// Benchmark tests — extracted from the original benchmark.rs test module.

use super::*;
use crate::gpus::tune::{
    TuneApplyMode, TuneBoolOrAutoValue, TuneConfigEdit, TuneField, TuneFieldStatus,
    TuneGpuLayersValue, TuneKvCacheType, TunePlan, TuneRecommendation, TuneRecommendedValue,
    TuneTarget,
};
use crate::gpus::tune_apply::PreparedTunePlan;
use crate::gpus::tune_resolver::{
    ConfigModelMatch, LocalTargetSource, ResolvedTuneTarget, TuneTargetSelection,
};

#[test]
fn trial_config_renders_string_paths_and_hardware_edits() {
    let prepared = prepared_plan_fixture(
        "/tmp/model with spaces.gguf",
        Vec::new(),
        vec![
            TuneFieldStatus::Applied {
                recommendation: TuneRecommendation {
                    field: TuneField::GpuLayers,
                    value: TuneRecommendedValue::GpuLayers(TuneGpuLayersValue::All),
                    rationale: "test".to_string(),
                },
                edit: TuneConfigEdit::SetHardwareGpuLayers(TuneGpuLayersValue::All),
            },
            TuneFieldStatus::Applied {
                recommendation: TuneRecommendation {
                    field: TuneField::FitTargetMib,
                    value: TuneRecommendedValue::FitTargetMib(60_000),
                    rationale: "test".to_string(),
                },
                edit: TuneConfigEdit::SetHardwareFitTargetMib(60_000),
            },
        ],
    );
    let candidate = TuneBenchmarkCandidate {
        ctx_size: 4096,
        batch: 2048,
        ubatch: 1024,
        cache_type_k: TuneKvCacheType::Q8_0,
        cache_type_v: TuneKvCacheType::Q8_0,
        mmap: TuneBoolOrAutoValue::Disabled,
        mlock: true,
        speculative: TuneBenchmarkSpeculativeCandidate::Mtp {
            draft_model: None,
            draft_max_tokens: 3,
            draft_min_tokens: 0,
            draft_acceptance_threshold: None,
            draft_split_probability: None,
        },
        flash_attention: None,
    };

    let rendered = trial_config(
        &mesh_llm_config::MeshConfig::default(),
        &prepared,
        &candidate,
    )
    .expect("trial config renders");
    let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
    let model = parsed.models.first().expect("model row exists");

    assert_eq!(model.model, "/tmp/model with spaces.gguf");
    assert_eq!(
        model
            .model_fit
            .as_ref()
            .and_then(|model_fit| model_fit.ctx_size),
        Some(4096)
    );
    assert!(matches!(
        model
            .hardware
            .as_ref()
            .and_then(|hardware| hardware.gpu_layers.as_ref()),
        Some(mesh_llm_config::IntegerOrString::Integer(-1))
    ));
    assert_eq!(
        model
            .hardware
            .as_ref()
            .and_then(|hardware| hardware.fit_target_mib),
        Some(60_000)
    );
    assert_eq!(
        model
            .hardware
            .as_ref()
            .and_then(|hardware| hardware.model_path.as_deref()),
        Some("/tmp/model with spaces.gguf")
    );
    assert_eq!(
        model
            .hardware
            .as_ref()
            .and_then(|hardware| hardware.mmap.as_ref()),
        Some(&mesh_llm_config::BoolOrAuto::Bool(false))
    );
    assert_eq!(
        model.hardware.as_ref().and_then(|hardware| hardware.mlock),
        Some(true)
    );
    assert_eq!(
        model
            .speculative
            .as_ref()
            .and_then(|speculative| speculative.strategy.as_deref()),
        Some("mtp")
    );
    let speculative = model.speculative.as_ref().expect("speculative config");
    assert_eq!(speculative.draft_max_tokens, Some(3));
    assert_eq!(speculative.draft_min_tokens, Some(0));
    assert_eq!(
        model
            .speculative
            .as_ref()
            .and_then(|speculative| speculative.mode.as_deref()),
        Some("auto")
    );
}

#[test]
fn trial_config_includes_runtime_native_runtime() {
    let prepared = prepared_plan_fixture("/tmp/model.gguf", Vec::new(), Vec::new());
    let candidate = TuneBenchmarkCandidate {
        ctx_size: 4096,
        batch: 2048,
        ubatch: 1024,
        cache_type_k: TuneKvCacheType::Q8_0,
        cache_type_v: TuneKvCacheType::Q8_0,
        mmap: TuneBoolOrAutoValue::Disabled,
        mlock: false,
        speculative: TuneBenchmarkSpeculativeCandidate::Disabled,
        flash_attention: None,
    };

    let mut config = mesh_llm_config::MeshConfig::default();
    config.runtime.native_runtime.mesh_version = Some("0.68.0".to_string());
    config.runtime.native_runtime.skippy_abi = Some("0.1.25".to_string());
    config.runtime.native_runtime.selection =
        Some("exact:meshllm-native-runtime-linux-x86_64-cuda12".to_string());

    let rendered = trial_config(&config, &prepared, &candidate).expect("trial config renders");
    let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");

    assert_eq!(
        parsed.runtime.native_runtime.mesh_version.as_deref(),
        Some("0.68.0")
    );
    assert_eq!(
        parsed.runtime.native_runtime.skippy_abi.as_deref(),
        Some("0.1.25")
    );
    assert_eq!(
        parsed.runtime.native_runtime.selection.as_deref(),
        Some("exact:meshllm-native-runtime-linux-x86_64-cuda12")
    );
}

#[test]
fn trial_config_renders_draft_speculative_candidate() {
    let prepared = prepared_plan_fixture("/tmp/model.gguf", Vec::new(), Vec::new());
    let candidate = TuneBenchmarkCandidate {
        ctx_size: 4096,
        batch: 2048,
        ubatch: 1024,
        cache_type_k: TuneKvCacheType::Q8_0,
        cache_type_v: TuneKvCacheType::Q8_0,
        mmap: TuneBoolOrAutoValue::Disabled,
        mlock: false,
        speculative: TuneBenchmarkSpeculativeCandidate::Draft {
            draft_model: "/tmp/model-draft.gguf".to_string(),
            draft_max_tokens: 8,
            draft_min_tokens: Some(2),
            draft_acceptance_threshold: Some(0.75),
            draft_split_probability: Some(0.25),
        },
        flash_attention: None,
    };

    let rendered = trial_config(
        &mesh_llm_config::MeshConfig::default(),
        &prepared,
        &candidate,
    )
    .expect("trial config renders");
    let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
    let speculative = parsed
        .models
        .first()
        .and_then(|model| model.speculative.as_ref())
        .expect("speculative config exists");

    assert_eq!(speculative.strategy.as_deref(), Some("disabled"));
    assert_eq!(speculative.mode.as_deref(), Some("draft"));
    assert_eq!(
        speculative.draft_model.as_deref(),
        Some("/tmp/model-draft.gguf")
    );
    assert_eq!(speculative.pairing_fault.as_deref(), Some("fail_closed"));
    assert_eq!(speculative.draft_max_tokens, Some(8));
    assert_eq!(speculative.draft_min_tokens, Some(2));
    assert_eq!(speculative.draft_acceptance_threshold, Some(0.75));
    assert_eq!(speculative.draft_split_probability, Some(0.25));
}

#[test]
fn trial_config_renders_mtp_speculative_sidecar_candidate() {
    let prepared = prepared_plan_fixture("/tmp/model.gguf", Vec::new(), Vec::new());
    let candidate = TuneBenchmarkCandidate {
        ctx_size: 4096,
        batch: 2048,
        ubatch: 1024,
        cache_type_k: TuneKvCacheType::Q8_0,
        cache_type_v: TuneKvCacheType::Q8_0,
        mmap: TuneBoolOrAutoValue::Enabled,
        mlock: false,
        speculative: TuneBenchmarkSpeculativeCandidate::Mtp {
            draft_model: Some("/tmp/mtp-gemma.gguf".to_string()),
            draft_max_tokens: 3,
            draft_min_tokens: 0,
            draft_acceptance_threshold: None,
            draft_split_probability: None,
        },
        flash_attention: None,
    };

    let rendered = trial_config(
        &mesh_llm_config::MeshConfig::default(),
        &prepared,
        &candidate,
    )
    .expect("trial config renders");
    let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
    let speculative = parsed
        .models
        .first()
        .and_then(|model| model.speculative.as_ref())
        .expect("speculative config exists");

    assert_eq!(speculative.strategy.as_deref(), Some("mtp"));
    assert_eq!(speculative.mode.as_deref(), Some("auto"));
    assert_eq!(
        speculative.draft_model.as_deref(),
        Some("/tmp/mtp-gemma.gguf")
    );
    assert_eq!(speculative.pairing_fault.as_deref(), Some("fail_closed"));
    assert_eq!(speculative.draft_max_tokens, Some(3));
    assert_eq!(speculative.draft_min_tokens, Some(0));
}

#[test]
fn trial_config_pins_resolved_model_path_for_huggingface_cache_targets() {
    let prepared = PreparedTunePlan::new(
        ResolvedTuneTarget {
            requested_input: "/cache/snapshot/model.gguf".to_string(),
            canonical_model_ref: "unsloth/example-GGUF:Q4_K_M".to_string(),
            resolved_path: std::path::PathBuf::from("/cache/blobs/model"),
            local_source: LocalTargetSource::HuggingFaceCache {
                canonical_ref: "unsloth/example-GGUF@sha/model.gguf".to_string(),
            },
            config_matches: Vec::new(),
            selection: TuneTargetSelection::Explicit { configured: false },
        },
        TunePlan {
            target: TuneTarget {
                requested: "/cache/snapshot/model.gguf".to_string(),
                resolved: Some("/cache/blobs/model".to_string()),
                config_model_ref: None,
                derived_profile: None,
            },
            apply_mode: TuneApplyMode::Review,
            field_statuses: Vec::new(),
            diagnostics: Vec::new(),
        },
    );
    let candidate = TuneBenchmarkCandidate {
        ctx_size: 4096,
        batch: 2048,
        ubatch: 1024,
        cache_type_k: TuneKvCacheType::Q8_0,
        cache_type_v: TuneKvCacheType::Q8_0,
        mmap: TuneBoolOrAutoValue::Enabled,
        mlock: false,
        speculative: TuneBenchmarkSpeculativeCandidate::Disabled,
        flash_attention: None,
    };

    let rendered = trial_config(
        &mesh_llm_config::MeshConfig::default(),
        &prepared,
        &candidate,
    )
    .expect("trial config renders");
    let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
    let model = parsed.models.first().expect("model row exists");

    assert_eq!(model.model, "unsloth/example-GGUF@sha/model.gguf");
    assert_eq!(
        model
            .hardware
            .as_ref()
            .and_then(|hardware| hardware.model_path.as_deref()),
        Some("/cache/blobs/model")
    );
}

#[test]
fn trial_config_renders_mtp_ngram_speculative_candidate() {
    let prepared = prepared_plan_fixture("/tmp/model.gguf", Vec::new(), Vec::new());
    let candidate = TuneBenchmarkCandidate {
        ctx_size: 4096,
        batch: 2048,
        ubatch: 1024,
        cache_type_k: TuneKvCacheType::Q8_0,
        cache_type_v: TuneKvCacheType::Q8_0,
        mmap: TuneBoolOrAutoValue::Disabled,
        mlock: false,
        speculative: TuneBenchmarkSpeculativeCandidate::MtpNgram {
            ngram_min: 2,
            ngram_max: 4,
        },
        flash_attention: None,
    };

    let rendered = trial_config(
        &mesh_llm_config::MeshConfig::default(),
        &prepared,
        &candidate,
    )
    .expect("trial config renders");
    let parsed = mesh_llm_config::parse_config_toml(&rendered).expect("trial config parses");
    let speculative = parsed
        .models
        .first()
        .and_then(|model| model.speculative.as_ref())
        .expect("speculative config exists");

    assert_eq!(speculative.strategy.as_deref(), Some("mtp"));
    assert_eq!(speculative.mode.as_deref(), Some("auto"));
    assert_eq!(speculative.ngram_min, Some(2));
    assert_eq!(speculative.ngram_max, Some(4));
}

#[test]
fn benchmark_candidates_sweep_mmap_and_available_mlock_independently() {
    let prepared = prepared_plan_fixture(
        "/tmp/model.gguf",
        Vec::new(),
        vec![TuneFieldStatus::Applied {
            recommendation: TuneRecommendation {
                field: TuneField::Mlock,
                value: TuneRecommendedValue::Bool(true),
                rationale: "test".to_string(),
            },
            edit: TuneConfigEdit::SetHardwareMlock(true),
        }],
    );
    let prepared = [prepared];
    let config = mesh_llm_config::MeshConfig::default();
    let request = TuneBenchmarkRunRequest {
        ctx_sizes: &[4096],
        batch_sizes: &[1024],
        ubatch_sizes: &[256],
        no_speculative_tune: true,
        ..benchmark_request_fixture(&config, &prepared)
    };

    let candidates = benchmark_candidates(&request, &prepared[0]);

    assert_eq!(candidates.len(), 6);
    assert!(
        candidates
            .iter()
            .any(|candidate| { candidate.mmap == TuneBoolOrAutoValue::Auto && !candidate.mlock })
    );
    assert!(
        candidates
            .iter()
            .any(|candidate| { candidate.mmap == TuneBoolOrAutoValue::Enabled && candidate.mlock })
    );
    assert!(
        candidates.iter().any(|candidate| {
            candidate.mmap == TuneBoolOrAutoValue::Disabled && candidate.mlock
        })
    );
}

#[test]
fn benchmark_candidates_default_to_preserved_config_model_fit() {
    let config = mesh_llm_config::MeshConfig {
        models: vec![mesh_llm_config::ModelConfigEntry {
            model: "model".to_string(),
            model_fit: Some(mesh_llm_config::ModelFitConfig {
                ctx_size: Some(131_072),
                batch: Some(2048),
                ubatch: Some(1024),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    };
    let prepared = prepared_plan_fixture(
        "/tmp/model.gguf",
        vec![ConfigModelMatch {
            row_index: 0,
            configured_model: "model".to_string(),
        }],
        Vec::new(),
    );
    let prepared = [prepared];
    let request = TuneBenchmarkRunRequest {
        mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
        mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
        throughput_tolerance_pct: 10.0,
        no_speculative_tune: true,
        ..benchmark_request_fixture(&config, &prepared)
    };

    let candidates = benchmark_candidates(&request, &prepared[0]);

    assert_eq!(candidates.len(), 6);
    assert!(
        candidates.iter().all(|candidate| candidate.batch == 2048),
        "configured batch should be used when --batch-sizes is omitted"
    );
    assert!(
        candidates.iter().all(|candidate| candidate.ubatch == 1024),
        "configured ubatch should be used when --ubatch-sizes is omitted"
    );
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.ctx_size == 131_072),
        "configured context should anchor the default context ladder"
    );
}

#[test]
fn benchmark_candidates_auto_prioritizes_native_mtp_for_mtp_targets() {
    let prepared = prepared_plan_fixture("/tmp/Qwen3.6-27B-MTP-GGUF.gguf", Vec::new(), Vec::new());
    let prepared = [prepared];
    let config = mesh_llm_config::MeshConfig::default();
    let request = TuneBenchmarkRunRequest {
        ctx_sizes: &[4096],
        batch_sizes: &[1024],
        ubatch_sizes: &[256],
        mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
        mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
        ..benchmark_request_fixture(&config, &prepared)
    };

    let candidates = benchmark_candidates(&request, &prepared[0]);
    let speculation = candidates
        .iter()
        .map(|candidate| candidate.speculative.clone())
        .collect::<Vec<_>>();

    assert_eq!(
        speculation,
        vec![
            TuneBenchmarkSpeculativeCandidate::Mtp {
                draft_model: None,
                draft_max_tokens: 2,
                draft_min_tokens: 0,
                draft_acceptance_threshold: None,
                draft_split_probability: None,
            },
            TuneBenchmarkSpeculativeCandidate::Mtp {
                draft_model: None,
                draft_max_tokens: 3,
                draft_min_tokens: 0,
                draft_acceptance_threshold: None,
                draft_split_probability: None,
            },
            TuneBenchmarkSpeculativeCandidate::Mtp {
                draft_model: None,
                draft_max_tokens: 4,
                draft_min_tokens: 0,
                draft_acceptance_threshold: None,
                draft_split_probability: None,
            },
            TuneBenchmarkSpeculativeCandidate::MtpNgram {
                ngram_min: 2,
                ngram_max: 4,
            },
            TuneBenchmarkSpeculativeCandidate::Disabled,
        ]
    );
}

#[test]
fn benchmark_candidates_no_speculative_tune_uses_disabled_baseline_only() {
    let prepared = prepared_plan_fixture("/tmp/Qwen3.6-27B-MTP-GGUF.gguf", Vec::new(), Vec::new());
    let prepared = [prepared];
    let config = mesh_llm_config::MeshConfig::default();
    let request = TuneBenchmarkRunRequest {
        ctx_sizes: &[4096],
        batch_sizes: &[1024],
        ubatch_sizes: &[256],
        mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
        mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
        no_speculative_tune: true,
        ..benchmark_request_fixture(&config, &prepared)
    };

    let candidates = benchmark_candidates(&request, &prepared[0]);
    let speculation = candidates
        .iter()
        .map(|candidate| candidate.speculative.clone())
        .collect::<Vec<_>>();

    assert_eq!(
        speculation,
        vec![TuneBenchmarkSpeculativeCandidate::Disabled]
    );
}

#[test]
fn benchmark_candidates_auto_skips_mtp_ngram_for_plain_targets() {
    let target_dir = tempfile::tempdir().expect("target tempdir");
    let target_path = target_dir.path().join("qwen-target.gguf");
    let prepared =
        prepared_plan_fixture(&target_path.display().to_string(), Vec::new(), Vec::new());
    let prepared = [prepared];
    let config = mesh_llm_config::MeshConfig::default();
    let request = TuneBenchmarkRunRequest {
        ctx_sizes: &[4096],
        batch_sizes: &[1024],
        ubatch_sizes: &[256],
        mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
        mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
        spec_ngram_min: &[2],
        spec_ngram_max: &[4],
        ..benchmark_request_fixture(&config, &prepared)
    };

    let candidates = benchmark_candidates(&request, &prepared[0]);
    let speculation = candidates
        .iter()
        .map(|candidate| candidate.speculative.clone())
        .collect::<Vec<_>>();

    assert_eq!(
        speculation,
        vec![TuneBenchmarkSpeculativeCandidate::Disabled]
    );
}

#[test]
fn benchmark_candidates_auto_skips_mtp_ngram_for_plain_draft_target() {
    let target_dir = tempfile::tempdir().expect("target tempdir");
    let target_path = target_dir.path().join("qwen-target.gguf");
    let prepared =
        prepared_plan_fixture(&target_path.display().to_string(), Vec::new(), Vec::new());
    let prepared = [prepared];
    let config = mesh_llm_config::MeshConfig::default();
    let draft_model = target_dir.path().join("qwen-draft.gguf");
    let request = TuneBenchmarkRunRequest {
        ctx_sizes: &[4096],
        batch_sizes: &[1024],
        ubatch_sizes: &[256],
        mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
        mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
        spec_draft_models: std::slice::from_ref(&draft_model),
        spec_draft_max_tokens: &[4],
        spec_ngram_min: &[2],
        spec_ngram_max: &[4],
        ..benchmark_request_fixture(&config, &prepared)
    };

    let candidates = benchmark_candidates(&request, &prepared[0]);
    let speculation = candidates
        .iter()
        .map(|candidate| candidate.speculative.clone())
        .collect::<Vec<_>>();

    assert_eq!(
        speculation,
        vec![
            TuneBenchmarkSpeculativeCandidate::Draft {
                draft_model: draft_model.display().to_string(),
                draft_max_tokens: 4,
                draft_min_tokens: None,
                draft_acceptance_threshold: None,
                draft_split_probability: None,
            },
            TuneBenchmarkSpeculativeCandidate::Disabled,
        ]
    );
}

#[test]
fn benchmark_candidates_explicit_speculative_sweeps_draft_and_ngram_settings() {
    let target_dir = tempfile::tempdir().expect("target tempdir");
    let target_path = target_dir.path().join("qwen-mtp-target.gguf");
    let prepared =
        prepared_plan_fixture(&target_path.display().to_string(), Vec::new(), Vec::new());
    let prepared = [prepared];
    let config = mesh_llm_config::MeshConfig::default();
    let draft_model = target_dir.path().join("qwen-draft.gguf");
    let request = TuneBenchmarkRunRequest {
        ctx_sizes: &[4096],
        batch_sizes: &[1024],
        ubatch_sizes: &[256],
        mmap_values: &[mesh_llm_cli::benchmark::BenchmarkBoolOrAuto::Disabled],
        mlock_values: &[mesh_llm_cli::benchmark::BenchmarkBool::Disabled],
        speculative_types: &[
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::Draft,
            mesh_llm_cli::benchmark::BenchmarkSpeculativeType::MtpNgram,
        ],
        spec_draft_models: std::slice::from_ref(&draft_model),
        spec_draft_max_tokens: &[4],
        spec_draft_min_tokens: &[2],
        spec_ngram_min: &[2],
        spec_ngram_max: &[4],
        ..benchmark_request_fixture(&config, &prepared)
    };

    let candidates = benchmark_candidates(&request, &prepared[0]);
    let speculation = candidates
        .iter()
        .map(|candidate| candidate.speculative.clone())
        .collect::<Vec<_>>();

    assert_eq!(
        speculation,
        vec![
            TuneBenchmarkSpeculativeCandidate::Draft {
                draft_model: draft_model.display().to_string(),
                draft_max_tokens: 4,
                draft_min_tokens: Some(2),
                draft_acceptance_threshold: None,
                draft_split_probability: None,
            },
            TuneBenchmarkSpeculativeCandidate::MtpNgram {
                ngram_min: 2,
                ngram_max: 4,
            },
        ]
    );
}

#[test]
fn selection_prefers_larger_context_within_throughput_tolerance() {
    let trials = vec![
        succeeded_trial(8192, 18.65, 2000.0),
        succeeded_trial(262_144, 18.23, 2100.0),
        succeeded_trial(65_536, 16.0, 2200.0),
    ];

    let selection = select_benchmark_trials(&trials, 3.0);

    assert_eq!(
        selection
            .raw_best
            .as_ref()
            .expect("raw best")
            .candidate
            .ctx_size,
        8192
    );
    assert_eq!(
        selection
            .recommended
            .as_ref()
            .expect("recommended")
            .candidate
            .ctx_size,
        262_144
    );
    assert!(
        selection
            .reason
            .as_deref()
            .expect("selection reason")
            .contains("within 3.00%")
    );
}

#[test]
fn selection_keeps_pareto_frontier_tradeoffs() {
    let trials = vec![
        succeeded_trial(4096, 20.0, 2000.0),
        succeeded_trial(8192, 19.0, 2000.0),
        succeeded_trial(4096, 18.0, 1900.0),
        succeeded_trial(16_384, 16.0, 2000.0),
    ];

    let selection = select_benchmark_trials(&trials, 1.0);
    let frontier_contexts = selection
        .pareto_frontier
        .iter()
        .map(|trial| trial.candidate.ctx_size)
        .collect::<Vec<_>>();

    assert_eq!(frontier_contexts, vec![16_384, 8192, 4096]);
    assert!(
        !selection
            .pareto_frontier
            .iter()
            .any(|trial| trial.decode_tok_s == Some(18.0)),
        "dominated lower-throughput 4096 ctx trial should be excluded"
    );
}

#[test]
fn selection_tie_breaks_toward_unlocked_auto_mmap() {
    let trials = vec![
        succeeded_trial_with_memory(8192, 20.0, 2000.0, TuneBoolOrAutoValue::Enabled, true),
        succeeded_trial_with_memory(8192, 20.0, 2000.0, TuneBoolOrAutoValue::Disabled, false),
        succeeded_trial_with_memory(8192, 20.0, 2000.0, TuneBoolOrAutoValue::Auto, false),
    ];

    let selection = select_benchmark_trials(&trials, 0.0);
    let recommended = selection.recommended.expect("recommended trial");

    assert_eq!(recommended.candidate.mmap, TuneBoolOrAutoValue::Auto);
    assert!(!recommended.candidate.mlock);
}

fn prepared_plan_fixture(
    resolved_path: &str,
    config_matches: Vec<ConfigModelMatch>,
    field_statuses: Vec<TuneFieldStatus>,
) -> PreparedTunePlan {
    let config_model_ref = config_matches
        .first()
        .map(|config_match| config_match.configured_model.clone());
    let selection = if config_matches.is_empty() {
        TuneTargetSelection::Explicit { configured: false }
    } else {
        TuneTargetSelection::Configured
    };
    PreparedTunePlan::new(
        ResolvedTuneTarget {
            requested_input: "model".to_string(),
            canonical_model_ref: "model".to_string(),
            resolved_path: std::path::PathBuf::from(resolved_path),
            local_source: LocalTargetSource::FilesystemPath {
                synthetic_model_ref: "model".to_string(),
            },
            config_matches,
            selection,
        },
        TunePlan {
            target: TuneTarget {
                requested: "model".to_string(),
                resolved: Some(resolved_path.to_string()),
                config_model_ref,
                derived_profile: None,
            },
            apply_mode: TuneApplyMode::Review,
            field_statuses,
            diagnostics: Vec::new(),
        },
    )
}

fn benchmark_request_fixture<'a>(
    config: &'a mesh_llm_config::MeshConfig,
    prepared: &'a [PreparedTunePlan],
) -> TuneBenchmarkRunRequest<'a> {
    TuneBenchmarkRunRequest {
        config,
        prepared,
        ctx_sizes: &[],
        batch_sizes: &[],
        ubatch_sizes: &[],
        mmap_values: &[],
        mlock_values: &[],
        flash_attention_values: &[],
        speculative_types: &[],
        no_speculative_tune: false,
        spec_draft_models: &[],
        spec_draft_max_tokens: &[],
        spec_draft_min_tokens: &[],
        spec_draft_acceptance_threshold: &[],
        spec_draft_split_probability: &[],
        spec_ngram_min: &[],
        spec_ngram_max: &[],
        throughput_tolerance_pct: 3.0,
        max_tokens: 32,
        startup_timeout_secs: 5,
        request_timeout_secs: 5,
        debug_telemetry: false,
        prompt: "hello",
    }
}

#[test]
fn debug_telemetry_enables_child_debug_and_stderr_spans() {
    let command = build_trial_child_command(
        std::path::Path::new("/bin/mesh-llm"),
        std::path::Path::new("/tmp/config.toml"),
        9337,
        3131,
        true,
    );
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();

    assert!(args.contains(&"--debug".to_string()));
    assert_eq!(args.last().map(String::as_str), Some("serve"));
    assert_eq!(
        command
            .get_envs()
            .find(|(key, _)| *key == "SKIPPY_TELEMETRY_STDERR")
            .and_then(|(_, value)| value)
            .map(|value| value.to_string_lossy()),
        Some(std::borrow::Cow::Borrowed("1"))
    );
}

#[test]
fn child_debug_telemetry_is_opt_in() {
    let command = build_trial_child_command(
        std::path::Path::new("/bin/mesh-llm"),
        std::path::Path::new("/tmp/config.toml"),
        9337,
        3131,
        false,
    );
    let args = command
        .get_args()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect::<Vec<_>>();

    assert!(!args.contains(&"--debug".to_string()));
    assert!(
        command
            .get_envs()
            .all(|(key, _)| key != "SKIPPY_TELEMETRY_STDERR")
    );
}

fn succeeded_trial(ctx_size: u32, decode_tok_s: f64, request_ms: f64) -> TuneBenchmarkTrial {
    succeeded_trial_with_memory(
        ctx_size,
        decode_tok_s,
        request_ms,
        TuneBoolOrAutoValue::Disabled,
        false,
    )
}

fn succeeded_trial_with_memory(
    ctx_size: u32,
    decode_tok_s: f64,
    request_ms: f64,
    mmap: TuneBoolOrAutoValue,
    mlock: bool,
) -> TuneBenchmarkTrial {
    TuneBenchmarkTrial {
        candidate: TuneBenchmarkCandidate {
            ctx_size,
            batch: 2048,
            ubatch: 1024,
            cache_type_k: TuneKvCacheType::Q8_0,
            cache_type_v: TuneKvCacheType::Q8_0,
            mmap,
            mlock,
            speculative: TuneBenchmarkSpeculativeCandidate::Disabled,
            flash_attention: None,
        },
        status: TuneBenchmarkTrialStatus::Succeeded,
        completion_tokens: Some(128),
        elapsed_ms: Some(request_ms),
        decode_tok_s: Some(decode_tok_s),
        ttft_ms: None,
        decode_only_tok_s: None,
        timings: Some(TuneBenchmarkTimingStats {
            total_ms: request_ms + 1000.0,
            setup_ms: 10.0,
            readiness_ms: 900.0,
            request_ms: Some(request_ms),
            shutdown_ms: Some(90.0),
            readiness_attempts: 3,
        }),
        log_path: None,
        error: None,
    }
}

#[test]
fn trial_startup_failure_scans_json_serve_logs() {
    let log = tempfile::NamedTempFile::new().expect("temp log");
    std::fs::write(
        log.path(),
        r#"{"level":"INFO","message":"API ready"}
{"level":"ERROR","message":"Failed to start model unsloth/Qwen3.6-MTP-GGUF: skippy speculative.strategy = \"mtp\" requires proven native MTP support"}
"#,
    )
    .expect("write log");

    let error = trial_startup_failure_from_log(log.path()).expect("startup error");
    assert!(error.contains("requires proven native MTP support"));
}

#[test]
fn trial_startup_failure_scans_plain_serve_logs() {
    let line = "2026-07-02 Failed to start model qwen: bad draft pair";

    let error = trial_startup_failure_from_log_line(line).expect("startup error");
    assert_eq!(error, line);
}

#[test]
fn metrics_schema_default_is_non_streaming_historical() {
    assert_eq!(
        TuneBenchmarkMetricsSchema::default(),
        TuneBenchmarkMetricsSchema::NonStreamingHistorical
    );
}

#[test]
fn benchmark_trial_unit_definition_describes_trial_and_pair() {
    let definition = benchmark_trial_unit_definition();
    assert!(
        definition
            .trial
            .contains("warmup request excluded from metrics")
    );
    assert!(definition.pair.contains("randomized per pair"));
}

#[test]
fn null_streaming_metrics_are_omitted_from_json_not_serialized_as_null_or_zero() {
    let trial = succeeded_trial(4096, 18.65, 2000.0);
    assert_eq!(trial.ttft_ms, None);
    assert_eq!(trial.decode_only_tok_s, None);

    let value = serde_json::to_value(&trial).expect("trial serializes");
    let object = value.as_object().expect("trial serializes as an object");
    assert!(
        !object.contains_key("ttft_ms"),
        "a null ttft_ms must be omitted, not written as `null`"
    );
    assert!(
        !object.contains_key("decode_only_tok_s"),
        "a null decode_only_tok_s must be omitted, not written as `null` or `0`"
    );
}

#[test]
fn old_schema_json_without_streaming_fields_deserializes_as_non_streaming_historical() {
    let old_style = serde_json::json!({
        "requested": "model",
        "throughput_tolerance_pct": 10.0,
        "best": {
            "candidate": {
                "ctx_size": 4096,
                "batch": 2048,
                "ubatch": 1024,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "mmap": "disabled",
                "mlock": false,
                "speculative": {"type": "disabled"}
            },
            "status": "succeeded",
            "completion_tokens": 128,
            "elapsed_ms": 2000.0,
            "decode_tok_s": 18.65,
            "timings": {
                "total_ms": 2100.0,
                "setup_ms": 10.0,
                "readiness_ms": 90.0,
                "request_ms": 2000.0,
                "shutdown_ms": 10.0,
                "readiness_attempts": 3
            },
            "log_path": "/tmp/log"
        },
        "trials": []
    });

    let report: TuneBenchmarkTargetReport = serde_json::from_value(old_style)
        .expect("a pre-streaming report (missing every task-18 field) must still deserialize");

    assert_eq!(
        report.metrics_schema,
        TuneBenchmarkMetricsSchema::NonStreamingHistorical,
        "a report with no metrics_schema key must be marked non-comparable, not treated as streaming"
    );
    assert!(report.trial_unit.is_none());
    let best = report.best.expect("best trial present");
    assert_eq!(
        best.decode_tok_s,
        Some(18.65),
        "historical decode_tok_s must round-trip unchanged"
    );
    assert_eq!(best.ttft_ms, None);
    assert_eq!(best.decode_only_tok_s, None);
}

#[test]
fn new_streaming_report_marks_streaming_v1_and_records_trial_unit() {
    let report = TuneBenchmarkTargetReport {
        requested: "model".to_string(),
        throughput_tolerance_pct: 10.0,
        best: None,
        raw_best: None,
        pareto_frontier: Vec::new(),
        selection_reason: None,
        trials: Vec::new(),
        metrics_schema: TuneBenchmarkMetricsSchema::StreamingV1,
        trial_unit: Some(benchmark_trial_unit_definition()),
    };

    let value = serde_json::to_value(&report).expect("report serializes");
    assert_eq!(value["metrics_schema"], "streaming_v1");
    assert!(
        value["trial_unit"]["trial"]
            .as_str()
            .unwrap()
            .contains("warmup")
    );
}
