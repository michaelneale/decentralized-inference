# PR8 configuration wiring closeout audit

This audit covers every `WIRING_MANIFEST` row whose owner is `PR8`. It records
the source-to-consumer path and the final boundary test for the repository tree
that contains this file. Re-run the listed tests after any change to a cited
path. Evidence from another commit does not satisfy this audit.

Each listed origin is parsed at the TOML boundary and validated against the
exported schema and wiring manifest before resolution. A rejected row records
the hardware limitation or ownership boundary that prevents a final consumer.

## Forward audit

| Config path | Config source and normalization | Runtime consumer | Final consumer boundary | Test evidence |
| --- | --- | --- | --- | --- |
| `telemetry.prompt_shape_metrics` | `mesh-llm-config` loads the telemetry switch; `runtime/survey.rs` creates `SurveyTelemetry` only when enabled | Named and automatic OpenAI routes record token usage plus the actual local, remote, or endpoint service and HTTP outcome through `Node::record_prompt_shape` | OTLP `mesh_llm_prompt_tokens` and `mesh_llm_completion_tokens` histograms carry only finite service and outcome enums; model, node, path, endpoint, prompt, and completion values are omitted | `network::openai::transport::tests::routing::named_model_route_records_prompt_shape_from_usage`; `runtime::survey::tests::prompt_shape_histograms_reach_exporter_with_reviewed_attributes` |
| `advanced.server.alias` | `runtime/startup_models.rs::configured_server_alias` applies per-model-over-default precedence while `StartupModelSpec::config_model_id` retains the canonical config key | startup resolution uses the canonical ID for model settings and the alias for `StartupModelPlan::declared_ref`; the Skippy resolver preserves both identities | `/v1/models` advertises the alias and routed chat completion sends that alias to the selected backend | `runtime::tests::startup_models::configured_server_alias_becomes_the_served_model_identity`; `runtime::tests::startup_models::configured_alias_keeps_canonical_model_id_through_resolution`; `runtime::local_split::test_support::runtime_resolver_uses_config_model_id_but_preserves_served_model_id`; `network::openai::response::models::tests::models_http_boundary_advertises_configured_served_alias`; `network::openai::transport::tests::routing::routed_completion_http_boundary_preserves_configured_served_alias` |
| `hardware.checkpoint_quantization` | `mesh-llm-config` parses the optional recipe and profile normalization applies per-model-over-default precedence; `runtime/run_auto.rs` applies an explicit CLI override to every selected model | the Skippy resolver carries the recipe into `StageConfig`; `skippy-server::runtime_config_from_stage_config` parses it into `CheckpointQuantization` | `skippy-runtime::open_safetensors` passes the selected llama file type to `skippy_model_open_from_source`, where native llama.cpp chooses and writes each destination tensor type | `inference::skippy::resolver::tests::hardware_checkpoint_quantization_reaches_stage_config`; `runtime_state::tests::runtime_config_parses_checkpoint_quantization`; `checkpoint::tests::canonical_quantization_catalog_round_trips`; ignored functional boundary `inference::skippy::resolver::tests::safetensors_checkpoint_reaches_mesh_host_runtime` |
| `hardware.checkpoint_imatrix` | `mesh-llm-config` parses the optional path and profile normalization applies per-model-over-default precedence; `inference/skippy/checkpoint.rs::prepare` resolves it relative to the checkpoint, canonicalizes it, and records its SHA-256 | the resolver carries the canonical path and digest into `StageConfig`; `skippy-server` carries both into `RuntimeConfig`; `skippy-runtime::NativeImatrix::load` verifies the exact bytes it then parses | importance-aware native per-tensor quantization receives only entries parsed from bytes matching the model/cache identity digest | `runtime_state::tests::runtime_config_accepts_importance_aware_checkpoint_quantization`; `checkpoint::tests::native_imatrix_verifies_the_bytes_it_parses`; `checkpoint::tests::native_imatrix_rejects_bytes_that_do_not_match_identity`; ignored functional boundary `inference::skippy::resolver::tests::safetensors_checkpoint_reaches_mesh_host_runtime` |
| `hardware.use_mmap_prefetch` | `mesh-llm-config` parses the field so validation can identify the exact path | no native model-loader consumer exists | static validation returns an error before resolution; Rust stage/FFI propagation is not counted as wiring | `exact_head_blockers::every_silent_no_op_and_false_mmap_control_fails_static_validation`; `hardware_validation::tests::mmap_native_controls_are_rejected_at_validation_boundary` |
| `hardware.use_mmap_buffer` | `mesh-llm-config` parses the field so validation can identify the exact path | no native model-loader consumer exists | static validation returns an error before resolution; Rust stage/FFI propagation is not counted as wiring | `exact_head_blockers::every_silent_no_op_and_false_mmap_control_fails_static_validation`; `hardware_validation::tests::mmap_native_controls_are_rejected_at_validation_boundary` |
| `plugin.<name>.url` | static validation accepts absolute HTTP(S) adapter URLs and rejects every `tcp://` spelling with a fixed diagnostic | configured plugin resolution and runtime startup independently reject TCP control without opening a socket | loopback, mixed-case, credentialed, query, and fragment forms fail without echoing userinfo or endpoint content | `exact_head_blockers::tcp_plugin_control_is_rejected_case_insensitively_without_echoing_secrets`; `plugin::transport::tests::remote_control_rejects_loopback_ipv4_without_connecting` |
| `plugin.<name>.startup.optional` | plugin resolution carries `PluginStartupOptions::optional` into each `ExternalPluginSpec` | `PluginManager::start` converts optional resolution, spawn, connect, and initialize failures to inactive error summaries; a required failure stops startup and shuts down plugins already loaded | required startup is transactional; optional startup returns a manager with an error summary instead of aborting host startup | `plugin::tests::optional_missing_installed_plugin_becomes_inactive_summary`; `plugin::tests::optional_plugin_load_failure_becomes_inactive_summary`; `plugin::tests::remote_connect_failures_honor_required_and_optional_policy`; `plugin::tests::required_plugin_load_failure_stops_manager_startup`; `plugin::tests::required_plugin_failure_rolls_back_plugins_loaded_earlier`; `plugin::runtime::tests::malformed_initialize_metadata_rolls_back_remote_runtime` |

## Reverse audit

The reverse audit covers every canonical manifest origin. The tables below
call out changed sinks in detail; this inventory prevents an undocumented
schema row or stale manifest row from passing review:

`version`, `gpu.assignment`, `gpu.parallel`,
`mesh_requirements.min_node_version`, `mesh_requirements.max_node_version`,
`mesh_requirements.min_protocol_version`, `mesh_requirements.max_protocol_version`,
`mesh_requirements.require_release_attestation`, `mesh_requirements.release_signer_keys`,
`owner_control.bind`, `owner_control.advertise_addr`, `telemetry.enabled`,
`telemetry.service_name`, `telemetry.endpoint`, `telemetry.headers`,
`telemetry.export_interval_secs`, `telemetry.queue_size`,
`telemetry.prompt_shape_metrics`, `telemetry.metrics.endpoint`,
`logging.audit.enabled`, `logging.audit.log_path`, `logging.audit.log_format`,
`logging.audit.log_level`, `logging.audit.max_file_size_mb`,
`logging.audit.max_files`, `logging.enabled`, `logging.application_state_root`,
`logging.summary_line_limit`, `logging.event_buffer_size`,
`logging.retention_ttl_secs`, `logging.retention_max_rows`,
`logging.replay_capacity`, `logging.queue_capacity`, `logging.export_limit_bytes`,
`logging.cleanup_cadence_secs`, `logging.artifact.capture_mode`,
`logging.artifact.byte_limit_bytes`, `logging.artifact.aggregate_limit_bytes`,
`logging.webhook.enabled`, `logging.webhook.url`, `logging.webhook.max_attempts`,
`logging.webhook.timeout_secs`, `logging.webhook.dead_letter_retention_secs`,
`runtime.debug`, `runtime.listen_all`, `runtime.mode`,
`runtime.startup_failure_policy`, `runtime.drain_timeout_secs`,
`runtime.drain_timeout_max_secs`, `runtime.activity.enabled`,
`runtime.activity.idle_after_secs`, `runtime.activity.poll_interval_secs`,
`runtime.activity.resume_debounce_secs`, `runtime.activity.response`,
`runtime.activity.advertisement`, `runtime.reconcile_model_targets`,
`runtime.reconcile_model_target_demand_upgrades`,
`runtime.native_runtime.mesh_version`, `runtime.native_runtime.skippy_abi`,
`runtime.native_runtime.selection`, `runtime.kv_cache.disk.mode`,
`runtime.kv_cache.disk.directory`, `runtime.kv_cache.disk.budget_mib`,
`runtime.kv_cache.disk.minimum_free_mib`, `runtime.model_target_demand_upgrade_min_requests`,
`runtime.model_target_demand_upgrade_max_age_secs`, `advanced.server.alias`,
`model`, `hardware.model_path`, `hardware.hf_repo`, `hardware.hf_file`,
`model_fit.ctx_size`, `model_fit.batch`, `model_fit.ubatch`,
`model_fit.cache_type_k`, `model_fit.cache_type_v`, `model_fit.kv_cache_policy`,
`model_fit.kv_offload`, `model_fit.kv_unified`, `model_fit.cache_ram_mib`,
`model_fit.cache_idle_slots`, `model_fit.prompt_cache`,
`model_fit.prefix_cache.enabled`, `model_fit.prefix_cache.max_entries`,
`model_fit.prefix_cache.max_bytes`, `model_fit.prefix_cache.min_tokens`,
`model_fit.prefix_cache.shared_stride_tokens`,
`model_fit.prefix_cache.shared_record_limit`, `model_fit.prefix_cache.payload_mode`,
`model_fit.keep_tokens`, `model_fit.context_shift`, `model_fit.swa_full`,
`model_fit.checkpoint_interval`, `model_fit.checkpoint_count`,
`model_fit.lookup_cache_static`, `model_fit.lookup_cache_dynamic`,
`model_fit.flash_attention`, `hardware.model_runtime`, `hardware.device`,
`hardware.gpu_layers`, `hardware.stage_layer_start`, `hardware.stage_layer_end`,
`hardware.placement`, `hardware.tensor_split`, `hardware.split_mode`,
`hardware.main_gpu`, `hardware.cpu_moe`, `hardware.n_cpu_moe`,
`hardware.fit_target_mib`, `hardware.safety_margin_gb`, `hardware.fit_context`,
`hardware.lora_adapters`, `hardware.control_vectors`, `hardware.check_tensors`,
`hardware.checkpoint_quantization`, `hardware.checkpoint_imatrix`,
`hardware.mmap`, `hardware.use_mmap_prefetch`, `hardware.use_mmap_buffer`,
`hardware.mlock`, `hardware.direct_io`, `hardware.repack`,
`hardware.op_offload`, `hardware.no_host_buffer`, `hardware.warmup`,
`throughput.parallel`, `throughput.continuous_batching`, `throughput.threads`,
`throughput.threads_batch`, `throughput.priority`, `throughput.poll`,
`throughput.cpu_affinity`, `throughput.numa`, `throughput.slot_prompt_similarity`,
`throughput.tuning_profile`, `topology.mode`, `topology.manifest_sha256`,
`topology.stages`, `skippy.stage_model_path`, `skippy.source_policy`, `skippy.stage_role`,
`skippy.stage_topology`, `skippy.activation_wire_dtype`,
`skippy.binary_stage_transport`, `skippy.lifecycle_startup_timeout_ms`,
`skippy.lifecycle_readiness_interval_ms`, `skippy.lifecycle_health_interval_ms`,
`skippy.prefill_chunking`, `skippy.prefill_chunk_size`,
`skippy.prefill_chunk_schedule`, `speculative.strategy`, `speculative.mode`,
`speculative.draft_model`, `speculative.draft_hf_repo`,
`speculative.draft_hf_file`, `speculative.draft_selection_policy`,
`speculative.pairing_fault`, `speculative.draft_max_tokens`,
`speculative.draft_min_tokens`, `speculative.draft_acceptance_threshold`,
`speculative.draft_split_probability`, `speculative.draft_gpu_layers`,
`speculative.draft_device`, `speculative.draft_threads`,
`speculative.draft_cache_type_k`, `speculative.draft_cache_type_v`,
`speculative.ngram_min`, `speculative.ngram_max`, `speculative.ngram_proposer`,
`speculative.ngram_max_proposal_tokens`, `speculative.extension_max_tokens`,
`speculative.native_mtp_reject_cooldown_tokens`,
`speculative.native_mtp_suppress_cooldown_drafts`,
`speculative.native_mtp_suppress_cooldown_draft_limit`,
`speculative.verify_window_min_tokens`, `speculative.verify_window_max_tokens`,
`speculative.verify_window_pipeline_depth`, `speculative.spec_default`,
`request_defaults.max_tokens`, `request_defaults.stop`,
`request_defaults.temperature`, `request_defaults.top_p`, `request_defaults.top_k`,
`request_defaults.min_p`, `request_defaults.typical_p`,
`request_defaults.top_nsigma`, `request_defaults.dynatemp_range`,
`request_defaults.dynatemp_exponent`, `request_defaults.repeat_penalty`,
`request_defaults.repeat_last_n`, `request_defaults.presence_penalty`,
`request_defaults.frequency_penalty`, `request_defaults.dry`,
`request_defaults.xtc`, `request_defaults.mirostat_mode`,
`request_defaults.mirostat_entropy`, `request_defaults.mirostat_learning_rate`,
`request_defaults.samplers`, `request_defaults.sampler_sequence`,
`request_defaults.seed`, `request_defaults.logit_bias`, `request_defaults.ignore_eos`,
`request_defaults.reasoning_format`, `request_defaults.reasoning_enabled`,
`request_defaults.reasoning_budget`, `request_defaults.chat_template`,
`request_defaults.chat_template_file`, `request_defaults.jinja`,
`request_defaults.chat_template_kwargs`, `request_defaults.skip_chat_parsing`,
`request_defaults.prefill_assistant`, `request_defaults.system_prompt`,
`multimodal.mmproj`, `hardware.mmproj`, `multimodal.mmproj_url`,
`multimodal.mmproj_offload`, `hardware.mmproj_offload`,
`multimodal.image_min_tokens`, `multimodal.image_max_tokens`,
`multimodal.image_marker`, `multimodal.media_marker`,
`multimodal.batch_max_tokens`, `multimodal.glm_dsa_policy`,
`multimodal.generation_signal_window`, `plugin.<name>.name`,
`plugin.<name>.enabled`, `plugin.<name>.web_ui_enabled`,
`plugin.<name>.command`, `plugin.<name>.args`, `plugin.<name>.url`,
`plugin.<name>.startup.connect_timeout_secs`,
`plugin.<name>.startup.init_timeout_secs`, `plugin.<name>.startup.optional`,
`plugin.<name>.startup.lazy_start`, `plugin.<name>.settings.*`,
`hardware.rpc_backend`, `throughput.threads_http`,
`throughput.sleep_idle_seconds`, `skippy.openai_frontend_mode`,
`request_defaults.backend_sampling`, `request_defaults.adaptive`,
`advanced.server.host`, `advanced.server.port`, `advanced.server.reuse_port`,
`advanced.server.timeout`, `advanced.server.metrics`, `advanced.server.slots`,
`advanced.server.props`, `advanced.server.api_prefix`, `multimodal.embeddings`,
`multimodal.reranking`, `multimodal.pooling`, `multimodal.vocoder`,
`request_defaults.grammar`, `request_defaults.json_schema`,
`request_defaults.logprobs`.

| Final consumer or changed sink | Originating PR8 config path | Reverse-path result |
| --- | --- | --- |
| Prompt and completion token histograms in the OTLP exporter | `telemetry.prompt_shape_metrics` | One originating switch. Request service and outcome come from the routed request result, not a second config field. Histogram attributes are restricted to those finite enums; model and node identities are omitted. |
| Public model IDs returned by `/v1/models` | `advanced.server.alias` | One originating identity override. Canonical model ID remains private to config lookup and runtime resolution. |
| Model value sent through routed chat completion | `advanced.server.alias` | The public alias is the served routing identity. Duplicate effective aliases are rejected during config validation, including aliases inherited from defaults. |
| Adapter endpoint environment | `plugin.<name>.url` | HTTP(S) URLs are passed to locally launched adapters. TCP control has no authenticated peer identity and is rejected statically and at runtime. |
| Continue-or-abort host startup decision after plugin failure | `plugin.<name>.startup.optional` | One originating policy bit. The decision covers resolution, process spawn, TCP connect, initialize parsing, and rollback of prior required plugins. |

## Closeout defects

| Defect | Resolution | Boundary evidence |
| --- | --- | --- |
| Served aliases compared before normalization | Alias comparison trims surrounding whitespace before constructing the public identity. | `model_validation::tests::served_alias_collisions_compare_trimmed_names` |
| Duplicate profiles compared before defaults merge | Duplicate canonical and served identities use the same effective fit, hardware, and throughput profile that startup derives from model-over-default precedence. | `model_validation::tests::served_alias_collisions_include_defaults_merged_profiles` |
| Unauthenticated TCP plugin control could impersonate a plugin | Static validation and runtime resolution reject all case variants of `tcp://` with a fixed capability-handshake reason; diagnostics and summaries do not echo userinfo, query, fragment, or raw plugin errors. | `exact_head_blockers::tcp_plugin_control_is_rejected_case_insensitively_without_echoing_secrets`; `plugin::transport::tests::remote_control_diagnostics_do_not_echo_endpoint_secrets` |
| Named-model routes discarded response usage | The named route finalizer records prompt and completion counts with the selected target's actual service and HTTP outcome. | `network::openai::transport::tests::routing::named_model_route_records_prompt_shape_from_usage` |
| Plugin-owned disabled startup was overwritten as failure | The startup-disabled protocol response remains a successful disabled lifecycle state instead of entering generic failure cleanup. | `plugin::runtime::tests::startup_disabled_response_remains_a_successful_disabled_plugin` |
| Duplicate canonical models resolved the first profile | Runtime startup passes the defaults-expanded derived profile identity into Skippy resolution, which matches that same effective profile before the canonical model fallback. | `inference::skippy::resolver::exact_head_tests::effective_derived_profile_selects_the_matching_duplicate_model_at_runtime_boundary` |
| Direct-local startup forced survey telemetry off | Direct-local launch creates `SurveyTelemetry` from the operator config and passes it into the local serving lifecycle. | `runtime::survey::tests::prompt_shape_histograms_reach_exporter_with_reviewed_attributes` |
| Wiring evidence could drift in only one direction | The contract normalizes defaults, model, plugin-name, and dynamic plugin-settings paths and rejects both schema rows missing from the manifest and stale manifest rows missing from the schema. | `wiring_manifest_covers_every_builtin_schema_path_in_both_directions` |
| Unsupported fields were warning-only no-ops | Every `SilentNoOp` manifest row is now an early validation error, including model runtime, fit controls, LoRA adapters, control vectors, warmup, and both false mmap controls. | `exact_head_blockers::every_silent_no_op_and_false_mmap_control_fails_static_validation` |
| Request model content reached OTLP labels | Lifecycle, request, route-attempt, and prompt-shape attributes omit model values entirely; route service and outcome remain finite enums. | `runtime::survey::survey_exact_head_tests::external_model_values_never_become_telemetry_attributes`; `runtime::survey::tests::prompt_shape_histograms_reach_exporter_with_reviewed_attributes` |

## Hardware and live-run limitation

No multi-node or hardware-backed live model run was available for this closeout.
The mmap decision is therefore rejection at the static boundary, not a claim of
native behavior based on Rust-only propagation. Resolver, exporter, CLI, and
build evidence is local and executable; no remote hardware evidence is claimed.

## Protocol compatibility

The stage-load runtime settings use additive optional protobuf field 43. Older
peers ignore the field, and newer peers treat its absence as default settings.
`mesh::stage_proto::tests::stage_load_wire_round_trip_preserves_runtime_controls`
proves that the optional settings survive the production encode/decode path.

## Validation commands

Run these commands serially from the repository root at the commit being
reviewed:

```bash
cargo test -p mesh-llm-config --lib
cargo test -p mesh-llm-config --test exact_head_blockers
cargo test -p mesh-llm-config --test wiring_status_contract
cargo test -p mesh-llm-host-runtime --lib
cargo test -p mesh-llm-host-runtime runtime::survey::tests --lib
cargo test -p mesh-llm-host-runtime telemetry_config --lib
cargo test -p mesh-llm-host-runtime routing_telemetry_sink_receives_request_pressure_and_attempt_events --lib
cargo test -p mesh-llm-host-runtime stage_load_wire_round_trip_preserves_runtime_controls --lib
cargo check -p mesh-llm
cargo clippy -p mesh-llm-config --all-targets -- -D warnings
cargo clippy -p mesh-llm-host-runtime --all-targets -- -D warnings
cargo clippy -p mesh-llm --all-targets -- -D warnings
cargo fmt --all --check
just no-console-print
just ci-validate
```
