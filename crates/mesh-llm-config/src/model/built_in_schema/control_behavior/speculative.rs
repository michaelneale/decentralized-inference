use super::shared::{
    equals_string_condition, not_in_condition, push_dependency_disable, push_non_empty_constraint,
    push_non_empty_constraint as non_empty, push_range_constraint, push_requires_constraint,
    set_numeric, set_runtime_gpu_options, set_static_options, set_text_format,
};
use super::*;

pub(super) fn apply_speculative_behavior(
    setting: &mut ConfigSettingSchema,
    prefix: &str,
    suffix: &str,
) {
    match suffix {
        "strategy" | "mode" | "draft_selection_policy" | "pairing_fault" | "spec_default" => {
            set_static_options(setting);
        }
        "draft_model" => {
            set_text_format(setting, ConfigTextFormat::Path);
            non_empty(setting);
            push_mode_dependency(setting, prefix, "draft", suffix);
        }
        "draft_hf_repo" => {
            non_empty(setting);
            push_requires_constraint(setting, &format!("{prefix}.draft_hf_file"));
            push_mode_dependency(setting, prefix, "draft", suffix);
        }
        "draft_hf_file" => {
            non_empty(setting);
            push_requires_constraint(setting, &format!("{prefix}.draft_hf_repo"));
            push_mode_dependency(setting, prefix, "draft", suffix);
        }
        "draft_max_tokens" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), Some("tokens"));
            push_mode_dependency(setting, prefix, "draft", suffix);
        }
        "draft_min_tokens" => {
            set_numeric(setting, Some(0.0), None, Some(1.0), Some("tokens"));
            push_range_constraint(setting, None, Some(format!("{prefix}.draft_max_tokens")));
            push_mode_dependency(setting, prefix, "draft", suffix);
        }
        "draft_acceptance_threshold" | "draft_split_probability" => {
            set_numeric(setting, Some(0.0), Some(1.0), Some(0.01), None);
            push_mode_dependency(setting, prefix, "draft", suffix);
        }
        "draft_gpu_layers" => {
            set_numeric(setting, Some(-1.0), None, Some(1.0), Some("layers"));
            push_mode_dependency(setting, prefix, "draft", suffix);
        }
        "draft_device" => {
            set_runtime_gpu_options(setting);
            push_non_empty_constraint(setting);
            push_mode_dependency(setting, prefix, "draft", suffix);
        }
        "draft_threads" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), Some("threads"));
            push_mode_dependency(setting, prefix, "draft", suffix);
        }
        "draft_cache_type_k" | "draft_cache_type_v" => {
            push_non_empty_constraint(setting);
            push_mode_dependency(setting, prefix, "draft", suffix);
        }
        "ngram_min" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), None);
        }
        "ngram_max" => {
            set_numeric(setting, Some(1.0), None, Some(1.0), None);
            push_range_constraint(setting, Some(format!("{prefix}.ngram_min")), None);
        }
        "ngram_max_proposal_tokens"
        | "extension_max_tokens"
        | "native_mtp_reject_cooldown_tokens"
        | "native_mtp_suppress_cooldown_drafts"
        | "native_mtp_suppress_cooldown_draft_limit"
        | "verify_window_min_tokens"
        | "verify_window_max_tokens"
        | "verify_window_pipeline_depth"
        | "verify_window_runahead_tokens" => {}
        _ => {}
    }
}

fn push_mode_dependency(
    setting: &mut ConfigSettingSchema,
    prefix: &str,
    expected_mode: &str,
    leaf: &str,
) {
    let mode_path = format!("{prefix}.mode");
    let current = format!("{prefix}.{leaf}");

    push_enable_when(setting, equals_string_condition(&mode_path, expected_mode));
    push_dependency_disable(
        setting,
        not_in_condition(
            &mode_path,
            vec![ConfigConditionValue::String(expected_mode.to_string())],
        ),
        format!("{current} requires {mode_path} = \"{expected_mode}\""),
    );
}

fn push_enable_when(setting: &mut ConfigSettingSchema, condition: ConfigControlCondition) {
    control_behavior_mut(setting).enable_when.push(condition);
}

fn control_behavior_mut(setting: &mut ConfigSettingSchema) -> &mut ConfigControlBehavior {
    setting
        .control_behavior
        .get_or_insert_with(ConfigControlBehavior::default)
}
