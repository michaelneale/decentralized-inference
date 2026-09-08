fn build_built_in_config_schema() -> ConfigSchema {
    let mut settings = vec![
        top_level_setting("version", ConfigValueSchema::Integer),
        top_level_setting("gpu.assignment", string_enum(["auto", "pinned"])),
        top_level_setting("gpu.parallel", ConfigValueSchema::Integer),
        top_level_setting(
            "mesh_requirements.min_node_version",
            string_enum_from_slice(known_mesh_llm_versions()),
        ),
        top_level_setting(
            "mesh_requirements.max_node_version",
            string_enum_from_slice(known_mesh_llm_versions()),
        ),
        top_level_setting(
            "mesh_requirements.min_protocol_version",
            ConfigValueSchema::Integer,
        ),
        top_level_setting(
            "mesh_requirements.max_protocol_version",
            ConfigValueSchema::Integer,
        ),
        top_level_setting(
            "mesh_requirements.require_release_attestation",
            ConfigValueSchema::Boolean,
        ),
        top_level_setting(
            "mesh_requirements.release_signer_keys",
            ConfigValueSchema::Array {
                items: Box::new(ConfigValueSchema::String),
            },
        ),
        owner_control_setting("owner_control.bind", ConfigValueSchema::SocketAddr),
        owner_control_setting(
            "owner_control.advertise_addr",
            ConfigValueSchema::SocketAddr,
        ),
        telemetry_setting("telemetry.enabled", ConfigValueSchema::Boolean),
        telemetry_setting("telemetry.service_name", ConfigValueSchema::String),
        telemetry_setting("telemetry.endpoint", ConfigValueSchema::Url),
        telemetry_setting("telemetry.headers", ConfigValueSchema::object()),
        telemetry_setting("telemetry.export_interval_secs", ConfigValueSchema::Integer),
        telemetry_setting("telemetry.queue_size", ConfigValueSchema::Integer),
        telemetry_setting("telemetry.prompt_shape_metrics", ConfigValueSchema::Boolean),
        telemetry_setting("telemetry.metrics.endpoint", ConfigValueSchema::Url),
        logging_audit_setting("logging.audit.enabled", ConfigValueSchema::Boolean),
        logging_audit_setting("logging.audit.log_path", ConfigValueSchema::Path),
        logging_audit_setting("logging.audit.log_format", string_enum(["json_lines"])),
        logging_audit_setting(
            "logging.audit.log_level",
            string_enum(["info", "warn", "error", "critical"]),
        ),
        logging_audit_setting("logging.audit.max_file_size_mb", ConfigValueSchema::Integer),
        logging_audit_setting("logging.audit.max_files", ConfigValueSchema::Integer),
        startup_runtime_setting("runtime.debug", ConfigValueSchema::Boolean),
        startup_runtime_setting("runtime.listen_all", ConfigValueSchema::Boolean),
        startup_runtime_setting(
            "runtime.mode",
            string_enum(["client", "serve", "on_demand"]),
        ),
        startup_runtime_setting(
            "runtime.startup_failure_policy",
            string_enum(["best_effort", "fail_fast"]),
        ),
        startup_runtime_setting(
            "runtime.lifecycle_log_parser",
            string_enum(["auto", "enabled", "disabled"]),
        ),
        runtime_setting("runtime.drain_timeout_secs", ConfigValueSchema::Integer),
        runtime_setting("runtime.drain_timeout_max_secs", ConfigValueSchema::Integer),
        activity_runtime_setting("runtime.activity.enabled", ConfigValueSchema::Boolean),
        activity_runtime_setting(
            "runtime.activity.idle_after_secs",
            ConfigValueSchema::Integer,
        ),
        activity_runtime_setting(
            "runtime.activity.poll_interval_secs",
            ConfigValueSchema::Integer,
        ),
        activity_runtime_setting(
            "runtime.activity.resume_debounce_secs",
            ConfigValueSchema::Integer,
        ),
        activity_runtime_setting(
            "runtime.activity.response",
            string_enum(["pause_remote", "pause_all", "reduce_priority"]),
        ),
        activity_runtime_setting(
            "runtime.activity.advertisement",
            string_enum([
                "none",
                "availability_only",
                "coarse_state",
                "private_coarse_state",
            ]),
        ),
        runtime_setting(
            "runtime.reconcile_model_targets",
            ConfigValueSchema::Boolean,
        ),
        runtime_setting(
            "runtime.reconcile_model_target_demand_upgrades",
            ConfigValueSchema::Boolean,
        ),
        native_runtime_setting(
            "runtime.native_runtime.mesh_version",
            ConfigValueSchema::String,
        ),
        native_runtime_setting(
            "runtime.native_runtime.skippy_abi",
            ConfigValueSchema::String,
        ),
        native_runtime_setting(
            "runtime.native_runtime.selection",
            ConfigValueSchema::String,
        ),
        runtime_setting(
            "runtime.model_target_demand_upgrade_min_requests",
            ConfigValueSchema::Integer,
        ),
        runtime_setting(
            "runtime.model_target_demand_upgrade_max_age_secs",
            ConfigValueSchema::Integer,
        ),
    ];

    settings.extend(logging_settings());

    settings.extend(model_defaults_settings());
    settings.extend(model_entry_settings());
    settings.extend(plugin_entry_settings());
    settings
        .iter_mut()
        .for_each(apply_built_in_control_behavior);
    settings
        .iter_mut()
        .for_each(apply_built_in_presentation_metadata);

    ConfigSchema { settings }
}

fn model_defaults_settings() -> Vec<ConfigSettingSchema> {
    let mut settings = Vec::new();
    settings.extend(model_fit_settings(
        "defaults.model_fit",
        &[
            flat_alias("defaults.ctx_size"),
            flat_alias("defaults.batch"),
            flat_alias("defaults.ubatch"),
            flat_alias("defaults.cache_type_k"),
            flat_alias("defaults.cache_type_v"),
            flat_alias("defaults.flash_attention"),
        ],
    ));
    settings.extend(hardware_settings(
        "defaults.hardware",
        &[flat_alias("defaults.gpu_id")],
    ));
    settings.extend(throughput_settings(
        "defaults.throughput",
        &[flat_alias("defaults.parallel")],
    ));
    settings.extend(topology_settings("defaults.topology"));
    settings.extend(skippy_settings("defaults.skippy"));
    settings.extend(speculative_settings("defaults.speculative"));
    settings.extend(request_defaults_settings("defaults.request_defaults"));
    settings.extend(multimodal_settings(
        "defaults.multimodal",
        &[flat_alias("defaults.mmproj")],
    ));
    settings.extend(advanced_settings("defaults.advanced"));
    settings
}

fn model_entry_settings() -> Vec<ConfigSettingSchema> {
    let model_prefix = format!("models.{CANONICAL_MODEL_REF_SEGMENT}");
    let mut settings = vec![basic_setting(
        &format!("{model_prefix}.model"),
        ConfigValueSchema::String,
    )];
    settings.extend(model_fit_settings(
        &format!("{model_prefix}.model_fit"),
        &[
            flat_alias(&format!("{model_prefix}.ctx_size")),
            flat_alias(&format!("{model_prefix}.batch")),
            flat_alias(&format!("{model_prefix}.ubatch")),
            flat_alias(&format!("{model_prefix}.cache_type_k")),
            flat_alias(&format!("{model_prefix}.cache_type_v")),
            flat_alias(&format!("{model_prefix}.flash_attention")),
        ],
    ));
    settings.extend(hardware_settings(
        &format!("{model_prefix}.hardware"),
        &[flat_alias(&format!("{model_prefix}.gpu_id"))],
    ));
    settings.extend(throughput_settings(
        &format!("{model_prefix}.throughput"),
        &[flat_alias(&format!("{model_prefix}.parallel"))],
    ));
    settings.extend(topology_settings(&format!("{model_prefix}.topology")));
    settings.extend(skippy_settings(&format!("{model_prefix}.skippy")));
    settings.extend(speculative_settings(&format!("{model_prefix}.speculative")));
    settings.extend(request_defaults_settings(&format!(
        "{model_prefix}.request_defaults"
    )));
    settings.extend(multimodal_settings(
        &format!("{model_prefix}.multimodal"),
        &[flat_alias(&format!("{model_prefix}.mmproj"))],
    ));
    settings.extend(advanced_settings(&format!("{model_prefix}.advanced")));
    settings
}

fn topology_settings(prefix: &str) -> Vec<ConfigSettingSchema> {
    let node_selector = ConfigValueSchema::Object {
        properties: vec![
            ConfigObjectPropertySchema {
                name: "endpoint_id".to_string(),
                label: "Endpoint ID".to_string(),
                required: false,
                value_schema: ConfigValueSchema::String,
            },
            ConfigObjectPropertySchema {
                name: "hostname".to_string(),
                label: "Hostname".to_string(),
                required: false,
                value_schema: ConfigValueSchema::String,
            },
        ],
    };
    let stage = ConfigValueSchema::Object {
        properties: vec![
            ConfigObjectPropertySchema {
                name: "node".to_string(),
                label: "Node".to_string(),
                required: true,
                value_schema: node_selector,
            },
            ConfigObjectPropertySchema {
                name: "layer_start".to_string(),
                label: "Layer start".to_string(),
                required: true,
                value_schema: ConfigValueSchema::Integer,
            },
            ConfigObjectPropertySchema {
                name: "layer_end".to_string(),
                label: "Layer end".to_string(),
                required: true,
                value_schema: ConfigValueSchema::Integer,
            },
        ],
    };
    vec![
        basic_setting(&format!("{prefix}.mode"), string_enum(["locked"])),
        basic_setting(
            &format!("{prefix}.manifest_sha256"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.stages"),
            ConfigValueSchema::Array {
                items: Box::new(stage),
            },
        ),
    ]
}

fn plugin_entry_settings() -> Vec<ConfigSettingSchema> {
    let plugin_prefix = format!("plugin.{CANONICAL_PLUGIN_NAME_SEGMENT}");
    vec![
        plugin_setting(&format!("{plugin_prefix}.name"), ConfigValueSchema::String),
        plugin_setting(
            &format!("{plugin_prefix}.enabled"),
            ConfigValueSchema::Boolean,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.web_ui_enabled"),
            ConfigValueSchema::Boolean,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.command"),
            ConfigValueSchema::String,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.args"),
            ConfigValueSchema::Array {
                items: Box::new(ConfigValueSchema::String),
            },
        ),
        plugin_setting(&format!("{plugin_prefix}.url"), ConfigValueSchema::Url),
        plugin_setting(
            &format!("{plugin_prefix}.startup.connect_timeout_secs"),
            ConfigValueSchema::Integer,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.startup.init_timeout_secs"),
            ConfigValueSchema::Integer,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.startup.optional"),
            ConfigValueSchema::Boolean,
        ),
        plugin_setting(
            &format!("{plugin_prefix}.startup.lazy_start"),
            ConfigValueSchema::Boolean,
        ),
    ]
}

fn model_fit_settings(
    prefix: &str,
    legacy_aliases: &[ConfigPathAlias],
) -> Vec<ConfigSettingSchema> {
    let mut settings = vec![
        basic_setting(&format!("{prefix}.ctx_size"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.batch"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.ubatch"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.cache_type_k"), kv_cache_type_schema()),
        basic_setting(&format!("{prefix}.cache_type_v"), kv_cache_type_schema()),
        basic_setting(
            &format!("{prefix}.kv_cache_policy"),
            string_enum(["auto", "quality", "balanced", "saver"]),
        ),
        basic_setting(&format!("{prefix}.kv_offload"), bool_or_auto_schema()),
        basic_setting(&format!("{prefix}.kv_unified"), bool_or_auto_schema()),
        unwired_setting(
            &format!("{prefix}.cache_ram_mib"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.cache_idle_slots"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(&format!("{prefix}.prompt_cache"), bool_or_auto_schema()),
        basic_setting(
            &format!("{prefix}.prefix_cache.enabled"),
            ConfigValueSchema::Boolean,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.max_entries"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.max_bytes"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.min_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.shared_stride_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.shared_record_limit"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefix_cache.payload_mode"),
            string_enum(["resident-kv", "kv-recurrent", "full-state", "auto"]),
        ),
        unwired_setting(&format!("{prefix}.keep_tokens"), ConfigValueSchema::Integer),
        unwired_setting(&format!("{prefix}.context_shift"), bool_or_auto_schema()),
        basic_setting(&format!("{prefix}.swa_full"), ConfigValueSchema::Boolean),
        unwired_setting(
            &format!("{prefix}.checkpoint_interval"),
            ConfigValueSchema::Integer,
        ),
        unwired_setting(
            &format!("{prefix}.checkpoint_count"),
            ConfigValueSchema::Integer,
        ),
        unwired_setting(
            &format!("{prefix}.lookup_cache_static"),
            ConfigValueSchema::String,
        ),
        unwired_setting(
            &format!("{prefix}.lookup_cache_dynamic"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.flash_attention"),
            string_enum(["auto", "disabled", "enabled"]),
        ),
    ];

    if !legacy_aliases.is_empty() {
        apply_aliases(
            &mut settings,
            &format!("{prefix}.ctx_size"),
            &legacy_aliases[0..1],
        );
        apply_aliases(
            &mut settings,
            &format!("{prefix}.batch"),
            &legacy_aliases[1..2],
        );
        apply_aliases(
            &mut settings,
            &format!("{prefix}.ubatch"),
            &legacy_aliases[2..3],
        );
        apply_aliases(
            &mut settings,
            &format!("{prefix}.cache_type_k"),
            &legacy_aliases[3..4],
        );
        apply_aliases(
            &mut settings,
            &format!("{prefix}.cache_type_v"),
            &legacy_aliases[4..5],
        );
        apply_aliases(
            &mut settings,
            &format!("{prefix}.flash_attention"),
            &legacy_aliases[5..6],
        );
    }

    settings
}

fn hardware_settings(
    prefix: &str,
    legacy_device_aliases: &[ConfigPathAlias],
) -> Vec<ConfigSettingSchema> {
    let mut settings = vec![
        hidden_unsupported_setting(
            &format!("{prefix}.model_runtime"),
            string_enum(["auto", "cpu", "cuda", "rocm", "metal", "vulkan"]),
            "Model runtime is selected by the installed native runtime and hardware resolver, not by the web configuration UI.",
        ),
        basic_setting(&format!("{prefix}.device"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.gpu_layers"), integer_or_auto_schema()),
        basic_setting(
            &format!("{prefix}.stage_layer_start"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.stage_layer_end"),
            ConfigValueSchema::Integer,
        ),
        unwired_setting(
            &format!("{prefix}.placement"),
            string_enum(["auto", "pooled", "separated"]),
        ),
        unwired_setting(&format!("{prefix}.tensor_split"), tensor_split_schema()),
        experimental_setting(
            &format!("{prefix}.split_mode"),
            string_enum(["auto", "none", "layer", "row", "tensor"]),
        ),
        experimental_setting(&format!("{prefix}.main_gpu"), ConfigValueSchema::Integer),
        unwired_setting(&format!("{prefix}.cpu_moe"), bool_or_auto_schema()),
        unwired_setting(&format!("{prefix}.n_cpu_moe"), ConfigValueSchema::Integer),
        rejected_setting(
            &format!("{prefix}.rpc_backend"),
            ConfigValueSchema::object(),
            "The legacy rpc_backend escape hatch is explicitly unsupported by the embedded runtime.",
        ),
        basic_setting(
            &format!("{prefix}.fit_target_mib"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.safety_margin_gb"),
            ConfigValueSchema::Float,
        ),
        unsupported_setting(&format!("{prefix}.fit_context"), bool_or_auto_schema()),
        basic_setting(&format!("{prefix}.model_path"), ConfigValueSchema::Path),
        basic_setting(&format!("{prefix}.hf_repo"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.hf_file"), ConfigValueSchema::String),
        basic_setting(&format!("{prefix}.mmproj"), ConfigValueSchema::Path),
        basic_setting(&format!("{prefix}.mmproj_offload"), bool_or_auto_schema()),
        unsupported_setting(
            &format!("{prefix}.lora_adapters"),
            ConfigValueSchema::Array {
                items: Box::new(ConfigValueSchema::String),
            },
        ),
        unsupported_setting(
            &format!("{prefix}.control_vectors"),
            ConfigValueSchema::Array {
                items: Box::new(ConfigValueSchema::String),
            },
        ),
        experimental_setting(
            &format!("{prefix}.check_tensors"),
            ConfigValueSchema::Boolean,
        ),
        experimental_setting(
            &format!("{prefix}.checkpoint_quantization"),
            string_enum([
                "preserve", "F32", "F16", "BF16", "Q1_0", "Q2_0", "Q4_0", "Q4_1", "Q5_0",
                "Q5_1", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ1_S", "IQ1_M", "TQ1_0",
                "TQ2_0", "Q2_K", "Q2_K_S", "IQ3_XS", "IQ3_XXS", "IQ3_S", "IQ3_M", "Q3_K_S",
                "Q3_K_M", "Q3_K_L", "IQ4_NL", "IQ4_XS", "Q4_K_S", "Q4_K_M", "Q5_K_S",
                "Q5_K_M", "Q6_K", "Q8_0", "MXFP4_MOE",
            ]),
        ),
        experimental_setting(
            &format!("{prefix}.checkpoint_imatrix"),
            ConfigValueSchema::String,
        ),
        basic_setting(&format!("{prefix}.mmap"), bool_or_auto_schema()),
        unsupported_setting(
            &format!("{prefix}.use_mmap_prefetch"),
            ConfigValueSchema::Boolean,
        ),
        unsupported_setting(
            &format!("{prefix}.use_mmap_buffer"),
            ConfigValueSchema::Boolean,
        ),
        basic_setting(&format!("{prefix}.mlock"), ConfigValueSchema::Boolean),
        experimental_setting(&format!("{prefix}.direct_io"), ConfigValueSchema::Boolean),
        experimental_setting(&format!("{prefix}.repack"), ConfigValueSchema::Boolean),
        experimental_setting(&format!("{prefix}.op_offload"), ConfigValueSchema::Boolean),
        experimental_setting(
            &format!("{prefix}.no_host_buffer"),
            ConfigValueSchema::Boolean,
        ),
        unsupported_setting(&format!("{prefix}.warmup"), bool_or_auto_schema()),
    ];

    if !legacy_device_aliases.is_empty() {
        apply_aliases(
            &mut settings,
            &format!("{prefix}.device"),
            legacy_device_aliases,
        );
    }

    settings
}

fn throughput_settings(
    prefix: &str,
    legacy_parallel_aliases: &[ConfigPathAlias],
) -> Vec<ConfigSettingSchema> {
    let mut settings = vec![
        basic_setting(&format!("{prefix}.parallel"), ConfigValueSchema::Integer),
        basic_setting(
            &format!("{prefix}.continuous_batching"),
            bool_or_auto_schema(),
        ),
        basic_setting(&format!("{prefix}.threads"), ConfigValueSchema::Integer),
        basic_setting(
            &format!("{prefix}.threads_batch"),
            ConfigValueSchema::Integer,
        ),
        rejected_setting(
            &format!("{prefix}.threads_http"),
            ConfigValueSchema::Integer,
            "Dedicated HTTP worker tuning is rejected on the current embedded runtime path.",
        ),
        rejected_setting(
            &format!("{prefix}.priority"),
            integer_or_string_schema(),
            "Model-scoped scheduling priority has no runtime or OS-priority consumer.",
        ),
        rejected_setting(
            &format!("{prefix}.poll"),
            bool_or_string_enum(["auto", "busy", "sleep"]),
            "Polling policy is not exposed by the embedded native runtime.",
        ),
        rejected_setting(
            &format!("{prefix}.cpu_affinity"),
            string_or_list_schema(),
            "Inference-thread CPU affinity has no supported runtime consumer.",
        ),
        rejected_setting(
            &format!("{prefix}.numa"),
            ConfigValueSchema::String,
            "NUMA policy is not exposed by the embedded native runtime.",
        ),
        rejected_setting(
            &format!("{prefix}.slot_prompt_similarity"),
            ConfigValueSchema::Float,
            "Skippy slot and prefix-cache reuse has no configurable similarity threshold.",
        ),
        rejected_setting(
            &format!("{prefix}.sleep_idle_seconds"),
            ConfigValueSchema::Integer,
            "The sleep-idle tuning knob is documented as rejected and must never become a live exported identifier.",
        ),
        basic_setting(
            &format!("{prefix}.tuning_profile"),
            string_enum(["throughput", "balanced", "saver"]),
        ),
    ];

    if !legacy_parallel_aliases.is_empty() {
        apply_aliases(
            &mut settings,
            &format!("{prefix}.parallel"),
            legacy_parallel_aliases,
        );
    }

    settings
}

fn skippy_settings(prefix: &str) -> Vec<ConfigSettingSchema> {
    vec![
        rejected_setting(
            &format!("{prefix}.stage_model_path"),
            ConfigValueSchema::Path,
            "Stage artifacts come from the canonical model source and package planner.",
        ),
        basic_setting(
            &format!("{prefix}.source_policy"),
            string_enum(["fallback", "local-required"]),
        ),
        rejected_setting(
            &format!("{prefix}.stage_role"),
            ConfigValueSchema::String,
            "The typed topology planner derives stage roles from ordered layer assignments.",
        ),
        rejected_setting(
            &format!("{prefix}.stage_topology"),
            ConfigValueSchema::String,
            "Untyped topology strings are unsupported; use typed per-model topology configuration.",
        ),
        rejected_setting(
            &format!("{prefix}.binary_stage_transport"),
            ConfigValueSchema::String,
            "Binary stage transport is selected automatically.",
        ),
        rejected_setting(
            &format!("{prefix}.openai_frontend_mode"),
            ConfigValueSchema::object(),
            "OpenAI frontend override wiring is intentionally rejected on the built-in schema surface.",
        ),
        basic_setting(
            &format!("{prefix}.lifecycle_startup_timeout_ms"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.lifecycle_readiness_interval_ms"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.lifecycle_health_interval_ms"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefill_chunking"),
            string_enum(["auto", "fixed", "schedule", "adaptive-ramp"]),
        ),
        basic_setting(
            &format!("{prefix}.prefill_chunk_size"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.prefill_chunk_schedule"),
            ConfigValueSchema::String,
        ),
    ]
}

fn speculative_settings(prefix: &str) -> Vec<ConfigSettingSchema> {
    vec![
        basic_setting(&format!("{prefix}.strategy"), ConfigValueSchema::String),
        basic_setting(
            &format!("{prefix}.mode"),
            string_enum(["auto", "disabled", "draft"]),
        ),
        basic_setting(&format!("{prefix}.draft_model"), ConfigValueSchema::Path),
        basic_setting(
            &format!("{prefix}.draft_hf_repo"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.draft_hf_file"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.draft_selection_policy"),
            string_enum(["manual", "auto"]),
        ),
        basic_setting(
            &format!("{prefix}.pairing_fault"),
            string_enum([
                "warn_disable",
                "fail-open",
                "fail-closed",
                "fail_open",
                "fail_closed",
            ]),
        ),
        basic_setting(
            &format!("{prefix}.draft_max_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.draft_min_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.draft_acceptance_threshold"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.draft_split_probability"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.draft_gpu_layers"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(&format!("{prefix}.draft_device"), ConfigValueSchema::String),
        basic_setting(
            &format!("{prefix}.draft_threads"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.draft_cache_type_k"),
            kv_cache_type_schema(),
        ),
        basic_setting(
            &format!("{prefix}.draft_cache_type_v"),
            kv_cache_type_schema(),
        ),
        basic_setting(&format!("{prefix}.ngram_min"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.ngram_max"), ConfigValueSchema::Integer),
        basic_setting(
            &format!("{prefix}.ngram_proposer"),
            string_enum(["cache", "suffix"]),
        ),
        basic_setting(
            &format!("{prefix}.ngram_max_proposal_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.extension_max_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.native_mtp_reject_cooldown_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.native_mtp_suppress_cooldown_drafts"),
            ConfigValueSchema::Boolean,
        ),
        basic_setting(
            &format!("{prefix}.native_mtp_suppress_cooldown_draft_limit"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.verify_window_min_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.verify_window_max_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.verify_window_pipeline_depth"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(&format!("{prefix}.spec_default"), bool_or_auto_schema()),
    ]
}

fn request_defaults_settings(prefix: &str) -> Vec<ConfigSettingSchema> {
    let mut settings = vec![
        basic_setting(&format!("{prefix}.max_tokens"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.stop"), string_or_list_schema()),
        basic_setting(&format!("{prefix}.temperature"), ConfigValueSchema::Float),
        basic_setting(&format!("{prefix}.top_p"), ConfigValueSchema::Float),
        basic_setting(&format!("{prefix}.top_k"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.min_p"), ConfigValueSchema::Float),
        basic_setting(&format!("{prefix}.typical_p"), ConfigValueSchema::Float),
        basic_setting(&format!("{prefix}.top_nsigma"), ConfigValueSchema::Float),
        basic_setting(
            &format!("{prefix}.dynatemp_range"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.dynatemp_exponent"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.repeat_penalty"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.repeat_last_n"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.presence_penalty"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.frequency_penalty"),
            ConfigValueSchema::Float,
        ),
        basic_setting(&format!("{prefix}.dry"), ConfigValueSchema::object()),
        basic_setting(&format!("{prefix}.xtc"), ConfigValueSchema::object()),
        rejected_setting(
            &format!("{prefix}.adaptive"),
            ConfigValueSchema::object(),
            "Adaptive sampling is reserved and rejected until it is implemented end to end.",
        ),
        basic_setting(
            &format!("{prefix}.mirostat_mode"),
            integer_or_string_enum(["disabled", "1", "2"]),
        ),
        basic_setting(
            &format!("{prefix}.mirostat_entropy"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.mirostat_learning_rate"),
            ConfigValueSchema::Float,
        ),
        basic_setting(
            &format!("{prefix}.samplers"),
            ConfigValueSchema::Array {
                items: Box::new(ConfigValueSchema::String),
            },
        ),
        basic_setting(
            &format!("{prefix}.sampler_sequence"),
            ConfigValueSchema::String,
        ),
        basic_setting(&format!("{prefix}.seed"), ConfigValueSchema::Integer),
        basic_setting(&format!("{prefix}.logit_bias"), ConfigValueSchema::object()),
        basic_setting(&format!("{prefix}.ignore_eos"), ConfigValueSchema::Boolean),
        rejected_setting(
            &format!("{prefix}.backend_sampling"),
            ConfigValueSchema::object(),
            "Backend-owned sampler blocks are explicitly rejected from the built-in control surface.",
        ),
        basic_setting(
            &format!("{prefix}.reasoning_format"),
            string_enum(["auto", "none", "deepseek", "deepseek-legacy", "hidden"]),
        ),
        basic_setting(
            &format!("{prefix}.reasoning_enabled"),
            bool_or_string_enum(["auto", "off", "on"]),
        ),
        basic_setting(
            &format!("{prefix}.reasoning_budget"),
            integer_or_string_enum(["auto", "low", "medium", "high"]),
        ),
        basic_setting(
            &format!("{prefix}.chat_template"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.chat_template_file"),
            ConfigValueSchema::Path,
        ),
        basic_setting(&format!("{prefix}.jinja"), ConfigValueSchema::Boolean),
        basic_setting(
            &format!("{prefix}.chat_template_kwargs"),
            ConfigValueSchema::object(),
        ),
        basic_setting(
            &format!("{prefix}.skip_chat_parsing"),
            ConfigValueSchema::Boolean,
        ),
        basic_setting(
            &format!("{prefix}.prefill_assistant"),
            ConfigValueSchema::OneOf {
                variants: vec![ConfigValueSchema::String, ConfigValueSchema::object()],
            },
        ),
        basic_setting(
            &format!("{prefix}.system_prompt"),
            ConfigValueSchema::String,
        ),
        basic_setting(&format!("{prefix}.grammar"), ConfigValueSchema::String),
        basic_setting(
            &format!("{prefix}.json_schema"),
            ConfigValueSchema::object(),
        ),
        rejected_setting(
            &format!("{prefix}.logprobs"),
            ConfigValueSchema::object(),
            "Logprobs request defaults are explicitly rejected from persisted config.",
        ),
    ];
    for setting in &mut settings {
        if setting.support == ConfigSupportState::Supported {
            setting.apply_mode = ConfigApplyMode::DynamicApply;
            setting.restart_scope = ConfigRestartScope::None;
        }
    }
    settings
}

fn multimodal_settings(
    prefix: &str,
    legacy_mmproj_aliases: &[ConfigPathAlias],
) -> Vec<ConfigSettingSchema> {
    let mut settings = vec![
        basic_setting(&format!("{prefix}.mmproj"), ConfigValueSchema::Path),
        basic_setting(&format!("{prefix}.mmproj_url"), ConfigValueSchema::Url),
        basic_setting(&format!("{prefix}.mmproj_offload"), bool_or_auto_schema()),
        basic_setting(
            &format!("{prefix}.image_min_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.image_max_tokens"),
            ConfigValueSchema::Integer,
        ),
        rejected_setting(
            &format!("{prefix}.image_marker"),
            ConfigValueSchema::String,
            "mtmd removed custom image markers; use media_marker.",
        ),
        basic_setting(&format!("{prefix}.media_marker"), ConfigValueSchema::String),
        basic_setting(
            &format!("{prefix}.batch_max_tokens"),
            ConfigValueSchema::Integer,
        ),
        basic_setting(
            &format!("{prefix}.glm_dsa_policy"),
            ConfigValueSchema::String,
        ),
        basic_setting(
            &format!("{prefix}.generation_signal_window"),
            ConfigValueSchema::Integer,
        ),
        rejected_setting(
            &format!("{prefix}.embeddings"),
            ConfigValueSchema::object(),
            "Built-in multimodal embeddings controls are explicitly rejected from persisted config.",
        ),
        rejected_setting(
            &format!("{prefix}.reranking"),
            ConfigValueSchema::object(),
            "Built-in reranking controls are explicitly rejected from persisted config.",
        ),
        rejected_setting(
            &format!("{prefix}.pooling"),
            ConfigValueSchema::object(),
            "Built-in pooling controls are explicitly rejected from persisted config.",
        ),
        rejected_setting(
            &format!("{prefix}.vocoder"),
            ConfigValueSchema::object(),
            "Built-in vocoder controls are explicitly rejected from persisted config.",
        ),
    ];

    if !legacy_mmproj_aliases.is_empty() {
        apply_aliases(
            &mut settings,
            &format!("{prefix}.mmproj"),
            legacy_mmproj_aliases,
        );
    }

    settings
}

fn advanced_settings(prefix: &str) -> Vec<ConfigSettingSchema> {
    vec![
        rejected_setting(
            &format!("{prefix}.server.host"),
            ConfigValueSchema::String,
            "Server host overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.port"),
            ConfigValueSchema::Integer,
            "Server port overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.reuse_port"),
            ConfigValueSchema::Boolean,
            "reuse_port overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.timeout"),
            ConfigValueSchema::Integer,
            "Server timeout overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.metrics"),
            ConfigValueSchema::Boolean,
            "Server metrics overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.slots"),
            ConfigValueSchema::Boolean,
            "Server slot overrides are explicitly rejected from persisted model config.",
        ),
        rejected_setting(
            &format!("{prefix}.server.props"),
            ConfigValueSchema::Boolean,
            "Server props overrides are explicitly rejected from persisted model config.",
        ),
        basic_setting(&format!("{prefix}.server.alias"), ConfigValueSchema::String),
        rejected_setting(
            &format!("{prefix}.server.api_prefix"),
            ConfigValueSchema::String,
            "API prefix overrides are explicitly rejected from persisted model config.",
        ),
    ]
}
