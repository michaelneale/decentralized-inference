use super::*;
mod logging;

use self::logging::{is_advanced_logging_category, logging_presentation};

#[derive(Clone, Copy)]
struct CategoryPresentation {
    id: &'static str,
    label: &'static str,
    summary: &'static str,
    order: u32,
}

const RUNTIME_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "runtime",
    label: "Runtime",
    summary: "Load-time runtime behavior and concurrency defaults",
    order: 10,
};
const MESHLLM_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "meshllm",
    label: "General",
    summary: "Local node startup and observability settings",
    order: 10,
};
const NETWORK_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "network",
    label: "Network",
    summary: "Owner-control listener and advertised control endpoint settings",
    order: 20,
};
const ATTESTATION_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "attestation",
    label: "Attestation",
    summary: "Creation-time certified-build admission requirements",
    order: 30,
};
const TELEMETRY_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "telemetry",
    label: "Telemetry",
    summary: "Opt-in metrics export and local telemetry queue settings",
    order: 40,
};
const RUNTIME_POLICY_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "runtime-policy",
    label: "Runtime Policy",
    summary: "Runtime reconciliation behavior applied by the local process",
    order: 10,
};
const MEMORY_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "memory",
    label: "Memory",
    summary: "VRAM accounting and KV cache policy",
    order: 20,
};
const SPECULATIVE_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "speculative-decoding",
    label: "Speculative Decoding",
    summary: "Speculative draft policy defaults",
    order: 30,
};
const REQUEST_DEFAULTS_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "request-defaults",
    label: "Request Defaults",
    summary: "Request-time sampling and reasoning defaults",
    order: 40,
};
const SKIPPY_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "skippy-transport",
    label: "Skippy Transport",
    summary: "Stage transport, chunking, and lifecycle defaults",
    order: 50,
};
const MULTIMODAL_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "multimodal",
    label: "Multimodal",
    summary: "Vision projector and image token defaults",
    order: 60,
};
const ADVANCED_SERVER_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "advanced-server",
    label: "Advanced Server",
    summary: "Advanced server defaults and identity overrides",
    order: 70,
};
const PLUGIN_HOST_CATEGORY: CategoryPresentation = CategoryPresentation {
    id: "plugin-host",
    label: "Plugin Host",
    summary: "Host-owned plugin process and startup settings",
    order: 10,
};
const MAX_ENUM_VALUES_FOR_SEGMENTED_CONTROL: usize = 4;

struct SettingPresentation {
    label: &'static str,
    help: &'static str,
    category: CategoryPresentation,
    order: u32,
    unit: Option<&'static str>,
    placeholder: Option<&'static str>,
    control_hint: Option<&'static str>,
    renderer_id: Option<&'static str>,
    choices: &'static [(&'static str, &'static str, &'static str)],
    display_units: &'static [(&'static str, &'static str, u64)],
}

fn setting_presentation_for_path(rendered: &str) -> Option<SettingPresentation> {
    logging_presentation(rendered)
        .or_else(|| process_setting_presentation(rendered))
        .or_else(|| runtime_defaults_presentation(rendered))
        .or_else(|| generation_defaults_presentation(rendered))
        .or_else(|| skippy_multimodal_presentation(rendered))
        .or_else(|| model_and_plugin_presentation(rendered))
}

fn process_setting_presentation(rendered: &str) -> Option<SettingPresentation> {
    match rendered {
        "gpu.assignment" => Some(sp(
            "GPU assignment",
            "Choose automatic GPU placement, or require configured model entries to name a concrete GPU device.",
            RUNTIME_CATEGORY,
            10,
        )
        .hint("segmented")),
        "gpu.parallel" => Some(sp(
            "GPU parallelism",
            "Limit the local GPU startup parallelism used when configured models are launched.",
            RUNTIME_CATEGORY,
            20,
        )
        .unit("models")
        .hint("number")),
        "telemetry.enabled" => Some(sp(
            "Telemetry export",
            "Enable opt-in metrics export. Ambient OTel environment variables do not enable export by themselves.",
            TELEMETRY_CATEGORY,
            10,
        )
        .hint("toggle")),
        "telemetry.service_name" => Some(sp(
            "Service name",
            "Service name attached to exported metrics when telemetry is enabled.",
            TELEMETRY_CATEGORY,
            20,
        )
        .placeholder("mesh-llm")
        .hint("text")),
        "telemetry.endpoint" => Some(sp(
            "OTLP endpoint",
            "Default OTLP endpoint used by telemetry exporters when telemetry is enabled.",
            TELEMETRY_CATEGORY,
            30,
        )
        .placeholder("http://127.0.0.1:4317")
        .hint("text")),
        "telemetry.metrics.endpoint" => Some(sp(
            "Metrics endpoint",
            "Metrics-specific OTLP endpoint. Leave empty to inherit the default telemetry endpoint.",
            TELEMETRY_CATEGORY,
            40,
        )
        .placeholder("http://127.0.0.1:4317")
        .hint("text")),
        "telemetry.headers" => Some(sp(
            "Telemetry headers",
            "Optional JSON object of headers attached to OTLP export requests.",
            TELEMETRY_CATEGORY,
            50,
        )
        .placeholder("{\"authorization\":\"Bearer ...\"}")
        .hint("textarea")),
        "telemetry.export_interval_secs" => Some(sp(
            "Export interval",
            "Seconds between telemetry export attempts when telemetry is enabled.",
            TELEMETRY_CATEGORY,
            60,
        )
        .unit("sec")
        .hint("number")),
        "telemetry.queue_size" => Some(sp(
            "Telemetry queue size",
            "Maximum queued telemetry events before nonblocking exporters drop new events.",
            TELEMETRY_CATEGORY,
            70,
        )
        .unit("events")
        .hint("number")),
        "runtime.reconcile_model_targets" => Some(sp(
            "Reconcile model targets",
            "Allow the runtime loop to reconcile configured model targets against current mesh demand.",
            RUNTIME_POLICY_CATEGORY,
            10,
        )
        .hint("toggle")),
        "runtime.reconcile_model_target_demand_upgrades" => Some(sp(
            "Demand upgrades",
            "Allow model-target reconciliation to upgrade targets when repeated demand is observed.",
            RUNTIME_POLICY_CATEGORY,
            20,
        )
        .hint("toggle")),
        "runtime.model_target_demand_upgrade_min_requests" => Some(sp(
            "Demand upgrade request floor",
            "Minimum request count before model-target reconciliation considers a demand upgrade.",
            RUNTIME_POLICY_CATEGORY,
            30,
        )
        .unit("requests")
        .hint("number")),
        "runtime.model_target_demand_upgrade_max_age_secs" => Some(sp(
            "Demand upgrade max age",
            "Maximum age in seconds for requests that count toward demand upgrades.",
            RUNTIME_POLICY_CATEGORY,
            40,
        )
        .unit("sec")
        .hint("number")),
        "runtime.debug" => Some(
            sp(
                "Debug output",
                "Enable mesh runtime debug output on startup. Set MESH_LLM_DEBUG_NATIVE_VERBOSE=1 separately for verbose llama.cpp native logs.",
                MESHLLM_CATEGORY,
                30,
            )
            .hint("toggle"),
        ),
        "runtime.listen_all" => Some(
            sp(
                "Listen on all interfaces",
                "Bind the OpenAI-compatible API and web console listeners to 0.0.0.0 instead of 127.0.0.1. This matches --listen-all and is useful for containers or exposed LAN hosts.",
                NETWORK_CATEGORY,
                30,
            )
            .hint("toggle"),
        ),
        "owner_control.bind" => Some(sp(
            "Owner-control bind",
            "Local address used by the owner-control listener. Set this to the same port as the advertised control address when overriding owner-control discovery.",
            NETWORK_CATEGORY,
            10,
        )
        .placeholder("127.0.0.1:0")
        .hint("text")),
        "owner_control.advertise_addr" => Some(sp(
            "Advertised control address",
            "Concrete address encoded into local owner-control bootstrap payloads. Requires owner-control bind to listen on the same port.",
            NETWORK_CATEGORY,
            20,
        )
        .placeholder("127.0.0.1:7447")
        .hint("text")),
        "mesh_requirements.min_node_version" => Some(sp(
            "Minimum node version",
            "Lowest mesh-llm node version allowed when this requirement-aware mesh is created or joined.",
            ATTESTATION_CATEGORY,
            10,
        )
        .placeholder("0.69.0")
        .hint("text")),
        "mesh_requirements.max_node_version" => Some(sp(
            "Maximum node version",
            "Highest mesh-llm node version allowed when this requirement-aware mesh is created or joined.",
            ATTESTATION_CATEGORY,
            20,
        )
        .placeholder("0.76.0-rc9")
        .hint("text")),
        "mesh_requirements.min_protocol_version" => Some(sp(
            "Minimum protocol generation",
            "Lowest protocol generation allowed by this mesh admission policy.",
            ATTESTATION_CATEGORY,
            30,
        )
        .hint("number")),
        "mesh_requirements.max_protocol_version" => Some(sp(
            "Maximum protocol generation",
            "Highest protocol generation allowed by this mesh admission policy.",
            ATTESTATION_CATEGORY,
            40,
        )
        .hint("number")),
        "mesh_requirements.require_release_attestation" => Some(sp(
            "Require certified release",
            "Require peers to advertise a trusted release-build attestation at admission time. This is build provenance, not remote runtime integrity proof.",
            ATTESTATION_CATEGORY,
            50,
        )
        .hint("toggle")),
        "mesh_requirements.release_signer_keys" => Some(sp(
            "Trusted release signer keys",
            "Release signer public keys accepted by this mesh, formatted as ed25519:<64 hex characters>.",
            ATTESTATION_CATEGORY,
            60,
        )
        .placeholder("ed25519:<64 hex characters>")
        .hint("text")),
        _ => None,
    }
}

fn runtime_defaults_presentation(rendered: &str) -> Option<SettingPresentation> {
    match rendered {
        "defaults.throughput.threads" => Some(sp(
            "CPU threads",
            "Sets the default CPU thread count. Use 0 for auto; 256 is a safe UI ceiling for general-purpose systems.",
            RUNTIME_CATEGORY,
            10,
        )
        .unit("threads")
        .hint("range")),
        "defaults.throughput.threads_batch" => Some(sp(
            "Batch threads",
            "Sets the thread count used for batching. Use 0 for auto; 256 is a safe UI ceiling for general-purpose systems.",
            RUNTIME_CATEGORY,
            20,
        )
        .unit("threads")
        .hint("range")),
        "defaults.throughput.continuous_batching" => Some(sp(
            "Continuous batching",
            "Choose whether the runtime should keep batching continuously when supported.",
            RUNTIME_CATEGORY,
            30,
        )
        .hint("segmented")),
        "defaults.hardware.gpu_layers" => Some(sp(
            "GPU layers",
            "Set the GPU layer count, or use auto. The backend also accepts -1 to mean all layers.",
            RUNTIME_CATEGORY,
            40,
        )
        .placeholder("auto or integer layer count")
        .hint("text")),
        "defaults.throughput.parallel" => Some(sp(
            "Default slots / parallel requests",
            "Sets the default parallel slots for placements without their own value. More slots increase KV memory use.",
            RUNTIME_CATEGORY,
            50,
        )
        .unit("slots")
        .hint("range")
        .renderer("slot-meter")),
        "defaults.throughput.tuning_profile" => Some(sp(
            "Default tuning profile",
            "Choose the starting balance between throughput, batch size, and memory use.",
            RUNTIME_CATEGORY,
            60,
        )
        .hint("segmented")),
        "defaults.model_fit.flash_attention" => Some(sp(
            "Flash attention policy",
            "Choose the default attention kernel policy for compatible runtimes.",
            RUNTIME_CATEGORY,
            70,
        )
        .hint("segmented")),
        "defaults.hardware.device" => Some(sp(
            "Default GPU device",
            "Optional fallback device for pinned GPU assignment when a model does not set its own device.",
            RUNTIME_CATEGORY,
            90,
        )
        .placeholder("cuda:0 or CUDA0")
        .hint("text")),
        "defaults.model_fit.kv_cache_policy" => Some(sp(
            "KV cache policy",
            "Select how aggressively KV cache precision is reduced to fit larger contexts.",
            MEMORY_CATEGORY,
            10,
        )
        .hint("segmented")
        .renderer("kv-cache-policy")),
        "defaults.hardware.safety_margin_gb" => Some(sp(
            "Memory / safety margin",
            "Keep this much GPU memory free before placement fit checks pass.",
            MEMORY_CATEGORY,
            20,
        )
        .unit("GB")
        .hint("range")),
        "defaults.model_fit.ctx_size" => Some(sp(
            "Context window size",
            "Set the default context window size in tokens.",
            MEMORY_CATEGORY,
            30,
        )
        .unit("tokens")
        .hint("range")),
        "defaults.model_fit.batch" => Some(sp(
            "Batch size",
            "Set the default prefill batch size.",
            MEMORY_CATEGORY,
            40,
        )
        .unit("tokens")
        .hint("range")),
        "defaults.model_fit.ubatch" => Some(sp(
            "Micro-batch size",
            "Set the default micro-batch (physical prefill chunk) size. Values at or below 128 keep the CUDA SSM sequential-scan fallback; larger values enable the SSD chunked kernel for recurrent models.",
            MEMORY_CATEGORY,
            50,
        )
        .unit("tokens")
        .hint("range")),
        "defaults.model_fit.cache_type_k" => Some(sp(
            "KV cache type (K)",
            "Choose the KV cache dtype used for keys.",
            MEMORY_CATEGORY,
            60,
        )
        .hint("segmented")),
        "defaults.model_fit.cache_type_v" => Some(sp(
            "KV cache type (V)",
            "Choose the KV cache dtype used for values.",
            MEMORY_CATEGORY,
            70,
        )
        .hint("segmented")),
        _ => None,
    }
}

fn generation_defaults_presentation(rendered: &str) -> Option<SettingPresentation> {
    match rendered {
        "defaults.speculative.mode" => Some(
            sp(
                "Default speculation mode",
                "Choose the default speculation method, or leave the runtime in auto mode.",
                SPECULATIVE_CATEGORY,
                10,
            )
            .hint("segmented"),
        ),
        "defaults.speculative.draft_selection_policy" => Some(
            sp(
                "Default draft selection policy",
                "Choose how draft models are selected when draft-model speculation is active.",
                SPECULATIVE_CATEGORY,
                20,
            )
            .hint("toggle"),
        ),
        "defaults.speculative.pairing_fault" => Some(
            sp(
                "Incompatible pairing behavior",
                "Choose what happens when the draft and target models cannot pair.",
                SPECULATIVE_CATEGORY,
                30,
            )
            .hint("toggle"),
        ),
        "defaults.speculative.draft_max_tokens" => Some(
            sp(
                "Default draft max tokens",
                "Limit how many draft tokens can be proposed before verification.",
                SPECULATIVE_CATEGORY,
                40,
            )
            .unit("tokens")
            .hint("range"),
        ),
        "defaults.speculative.draft_min_tokens" => Some(
            sp(
                "Default draft minimum tokens",
                "Set the smallest draft batch attempted before verification.",
                SPECULATIVE_CATEGORY,
                50,
            )
            .unit("tokens")
            .hint("range"),
        ),
        "defaults.request_defaults.temperature" => Some(
            sp(
                "Temperature",
                "Fallback sampling temperature for requests that do not provide one.",
                REQUEST_DEFAULTS_CATEGORY,
                10,
            )
            .hint("range"),
        ),
        "defaults.request_defaults.top_p" => Some(
            sp(
                "Top-p",
                "Fallback nucleus sampling threshold for requests that omit one.",
                REQUEST_DEFAULTS_CATEGORY,
                20,
            )
            .hint("range"),
        ),
        "defaults.request_defaults.reasoning_format" => Some(
            sp(
                "Reasoning format",
                "Choose how thinking tokens appear in the response stream.",
                REQUEST_DEFAULTS_CATEGORY,
                30,
            )
            .hint("segmented"),
        ),
        "defaults.request_defaults.reasoning_budget" => Some(
            sp(
                "Reasoning budget",
                "Cap the reasoning tokens reserved before the final answer.",
                REQUEST_DEFAULTS_CATEGORY,
                40,
            )
            .unit("tok")
            .hint("range"),
        ),
        "defaults.request_defaults.repeat_penalty" => Some(
            sp(
                "Repeat penalty",
                "Adjust how strongly repeated tokens are discouraged.",
                REQUEST_DEFAULTS_CATEGORY,
                50,
            )
            .hint("range"),
        ),
        "defaults.request_defaults.repeat_last_n" => Some(
            sp(
                "Repeat last-n window",
                "Set how much recent token history the repeat penalty checks.",
                REQUEST_DEFAULTS_CATEGORY,
                60,
            )
            .unit("tok")
            .hint("range"),
        ),
        "defaults.request_defaults.top_k" => Some(
            sp(
                "Top-k",
                "Limit sampling to the top-k tokens.",
                REQUEST_DEFAULTS_CATEGORY,
                70,
            )
            .hint("range"),
        ),
        "defaults.request_defaults.min_p" => Some(
            sp(
                "Min-p",
                "Filter tokens below a dynamic probability floor.",
                REQUEST_DEFAULTS_CATEGORY,
                80,
            )
            .hint("range"),
        ),
        "defaults.request_defaults.presence_penalty" => Some(
            sp(
                "Presence penalty",
                "Increase or reduce the penalty for introducing new tokens.",
                REQUEST_DEFAULTS_CATEGORY,
                90,
            )
            .hint("range"),
        ),
        "defaults.request_defaults.frequency_penalty" => Some(
            sp(
                "Frequency penalty",
                "Increase or reduce the penalty for repeated tokens.",
                REQUEST_DEFAULTS_CATEGORY,
                100,
            )
            .hint("range"),
        ),
        "defaults.request_defaults.max_tokens" => Some(
            sp(
                "Max tokens",
                "Cap the number of generated tokens for a request.",
                REQUEST_DEFAULTS_CATEGORY,
                110,
            )
            .unit("tokens")
            .hint("range"),
        ),
        _ => None,
    }
}

fn skippy_multimodal_presentation(rendered: &str) -> Option<SettingPresentation> {
    match rendered {
        "defaults.skippy.stage_model_path" => Some(sp(
            "Stage model path",
            "Set the model or package path used for this skippy stage.",
            SKIPPY_CATEGORY,
            20,
        )
        .placeholder("hf://... or /path/to/stage.gguf")
        .hint("text")),
        "defaults.skippy.source_policy" => Some(sp(
            "Split source policy",
            "Require split workers to use an already-present, content-verified local GGUF, or preserve resolver fallback behavior.",
            SKIPPY_CATEGORY,
            25,
        )
        .hint("segmented")),
        "defaults.skippy.stage_role" => Some(sp(
            "Stage role",
            "Choose the stage-chain role when topology is not inferred automatically.",
            SKIPPY_CATEGORY,
            30,
        )
        .hint("select")),
        "defaults.skippy.stage_topology" => Some(sp(
            "Stage topology",
            "Describe the stage chain topology when it is supplied as a text override.",
            SKIPPY_CATEGORY,
            40,
        )
        .placeholder("topology name or path")
        .hint("text")),
        "defaults.skippy.prefill_chunking" => Some(sp(
            "Prefill chunking",
            "Choose how prefill chunks are scheduled across a skippy stage chain.",
            SKIPPY_CATEGORY,
            50,
        )
        .hint("select")),
        "defaults.skippy.prefill_chunk_size" => Some(sp(
            "Prefill chunk size",
            "Set the fixed prefill chunk size. Use 0 to keep the backend auto sentinel.",
            SKIPPY_CATEGORY,
            60,
        )
        .unit("tokens")
        .hint("range")),
        "defaults.skippy.prefill_chunk_schedule" => Some(sp(
            "Prefill chunk schedule",
            "Provide a comma-separated schedule for scheduled prefill chunking.",
            SKIPPY_CATEGORY,
            70,
        )
        .placeholder("e.g. 512,1024,2048")
        .hint("text")),
        "defaults.skippy.binary_stage_transport" => Some(sp(
            "Binary stage transport",
            "Choose whether the binary stage transport is enabled or left to auto selection.",
            SKIPPY_CATEGORY,
            80,
        )
        .hint("segmented")),
        "defaults.multimodal.mmproj_offload" => Some(sp(
            "MMProj offload",
            "Choose whether the multimodal projector stays auto-managed or explicitly on or off.",
            MULTIMODAL_CATEGORY,
            10,
        )
        .hint("segmented")),
        "defaults.multimodal.image_min_tokens" => Some(sp(
            "Image minimum tokens",
            "Set the minimum token budget reserved for each image input.",
            MULTIMODAL_CATEGORY,
            20,
        )
        .unit("tokens")
        .hint("range")),
        "defaults.multimodal.image_max_tokens" => Some(sp(
            "Image maximum tokens",
            "Set the maximum token budget allowed for each image input.",
            MULTIMODAL_CATEGORY,
            30,
        )
        .unit("tokens")
        .hint("range")),
        "defaults.multimodal.mmproj" => Some(sp(
            "MMProj path",
            "Set an explicit local path to the multimodal projector file.",
            MULTIMODAL_CATEGORY,
            40,
        )
        .placeholder("e.g. /path/to/mmproj.gguf")
        .hint("text")),
        "defaults.multimodal.mmproj_url" => Some(sp(
            "MMProj URL",
            "Set a URL used to download or reference the multimodal projector file.",
            MULTIMODAL_CATEGORY,
            50,
        )
        .placeholder("e.g. https://example.com/mmproj.gguf")
        .hint("text")),
        "defaults.advanced.server.alias" => Some(sp(
            "Server alias",
            "Set a human-friendly alias for the server in advanced deployments.",
            ADVANCED_SERVER_CATEGORY,
            10,
        )
        .placeholder("model alias")
        .hint("text")),
        _ => None,
    }
}

fn model_and_plugin_presentation(rendered: &str) -> Option<SettingPresentation> {
    match rendered {
        "models.<model-ref>.model" => Some(
            sp(
                "Model",
                "Model reference for this local placement.",
                RUNTIME_CATEGORY,
                10,
            )
            .renderer("model-placement-model"),
        ),
        "models.<model-ref>.model_fit.ctx_size" => Some(
            sp(
                "Context window size",
                "Context window size for this local placement.",
                MEMORY_CATEGORY,
                20,
            )
            .unit("tokens")
            .renderer("model-placement-context"),
        ),
        "models.<model-ref>.hardware.device" => Some(
            sp(
                "GPU device",
                "Device assignment for this local placement.",
                RUNTIME_CATEGORY,
                30,
            )
            .placeholder("cuda:0")
            .renderer("model-placement-device"),
        ),
        "models.<model-ref>.hardware.gpu_layers" => Some(
            sp(
                "GPU layers",
                "GPU layer count for this local placement.",
                RUNTIME_CATEGORY,
                40,
            )
            .placeholder("-1")
            .renderer("model-placement-gpu-layers"),
        ),
        "plugin.<plugin-name>.enabled" => Some(
            sp(
                "Enabled",
                "Enable or disable the plugin entry.",
                PLUGIN_HOST_CATEGORY,
                10,
            )
            .hint("toggle"),
        ),
        "plugin.<plugin-name>.web_ui_enabled" => Some(
            sp(
                "Web UI projection",
                "Show or hide the plugin's declared web UI without disabling the plugin process.",
                PLUGIN_HOST_CATEGORY,
                15,
            )
            .hint("toggle"),
        ),
        "plugin.<plugin-name>.url" => Some(
            sp(
                "Base URL",
                "URL used by endpoint-style plugins.",
                PLUGIN_HOST_CATEGORY,
                20,
            )
            .placeholder("http://localhost:8000/v1")
            .hint("text"),
        ),
        "plugin.<plugin-name>.command" => Some(
            sp(
                "Plugin command",
                "Optional path to the plugin binary when it is not on PATH.",
                PLUGIN_HOST_CATEGORY,
                30,
            )
            .placeholder("use bundled plugin binary")
            .hint("text"),
        ),
        "plugin.<plugin-name>.args" => Some(
            sp(
                "Args",
                "Additional CLI args passed to the plugin process.",
                PLUGIN_HOST_CATEGORY,
                40,
            )
            .placeholder("comma-separated CLI arguments")
            .hint("text"),
        ),
        "plugin.<plugin-name>.startup.connect_timeout_secs" => Some(
            sp(
                "Connect timeout",
                "Seconds to wait for the plugin transport connection.",
                PLUGIN_HOST_CATEGORY,
                50,
            )
            .unit("sec")
            .hint("number"),
        ),
        "plugin.<plugin-name>.startup.init_timeout_secs" => Some(
            sp(
                "Init timeout",
                "Seconds to wait for plugin initialization.",
                PLUGIN_HOST_CATEGORY,
                60,
            )
            .unit("sec")
            .hint("number"),
        ),
        "plugin.<plugin-name>.startup.optional" => Some(
            sp(
                "Optional startup",
                "Allow the host to continue when the plugin cannot start.",
                PLUGIN_HOST_CATEGORY,
                70,
            )
            .hint("toggle"),
        ),
        "plugin.<plugin-name>.startup.lazy_start" => Some(
            sp(
                "Lazy start",
                "Delay plugin startup until the plugin is first needed.",
                PLUGIN_HOST_CATEGORY,
                80,
            )
            .hint("toggle"),
        ),
        _ => None,
    }
}

fn sp(
    label: &'static str,
    help: &'static str,
    category: CategoryPresentation,
    order: u32,
) -> SettingPresentation {
    SettingPresentation {
        label,
        help,
        category,
        order,
        unit: None,
        placeholder: None,
        control_hint: None,
        renderer_id: None,
        choices: &[],
        display_units: &[],
    }
}

impl SettingPresentation {
    fn unit(mut self, unit: &'static str) -> Self {
        self.unit = Some(unit);
        self
    }

    fn placeholder(mut self, placeholder: &'static str) -> Self {
        self.placeholder = Some(placeholder);
        self
    }

    fn hint(mut self, control_hint: &'static str) -> Self {
        self.control_hint = Some(control_hint);
        self
    }

    fn renderer(mut self, renderer_id: &'static str) -> Self {
        self.renderer_id = Some(renderer_id);
        self
    }

    fn choices(mut self, choices: &'static [(&'static str, &'static str, &'static str)]) -> Self {
        self.choices = choices;
        self
    }

    fn display_units(mut self, units: &'static [(&'static str, &'static str, u64)]) -> Self {
        self.display_units = units;
        self
    }
}

pub(super) fn apply_built_in_presentation_metadata(setting: &mut ConfigSettingSchema) {
    let rendered = setting.path.render();
    let Some(presentation) = setting_presentation_for_path(&rendered) else {
        apply_fallback_presentation_metadata(setting, &rendered);
        return;
    };

    setting.description = Some(presentation.help.to_string());
    if presentation.category.id != ADVANCED_SERVER_CATEGORY.id
        && !is_advanced_logging_category(presentation.category.id)
    {
        setting.visibility = ConfigVisibility::User;
    }
    setting.presentation = Some(ConfigPresentationMetadata {
        label: Some(presentation.label.to_string()),
        help: Some(presentation.help.to_string()),
        category_id: Some(presentation.category.id.to_string()),
        category_label: Some(presentation.category.label.to_string()),
        category_summary: Some(presentation.category.summary.to_string()),
        category_order: Some(presentation.category.order),
        setting_order: Some(presentation.order),
        unit: presentation.unit.map(str::to_string),
        placeholder: presentation.placeholder.map(str::to_string),
        control_hint: presentation.control_hint.map(str::to_string),
        renderer_id: presentation.renderer_id.map(str::to_string),
        choices: presentation
            .choices
            .iter()
            .map(|(value, label, description)| ConfigPresentationChoice {
                value: (*value).to_string(),
                label: (*label).to_string(),
                description: Some((*description).to_string()),
            })
            .collect(),
        display_units: presentation
            .display_units
            .iter()
            .map(|(value, label, multiplier)| ConfigDisplayUnit {
                value: (*value).to_string(),
                label: (*label).to_string(),
                multiplier: *multiplier,
            })
            .collect(),
    });
}

fn apply_fallback_presentation_metadata(setting: &mut ConfigSettingSchema, rendered: &str) {
    let Some(category) = fallback_category_for_path(rendered) else {
        return;
    };
    let key = rendered.rsplit('.').next().unwrap_or(rendered);
    let order = fallback_setting_order(rendered);
    let label = title_case_config_key(key);

    setting.presentation = Some(ConfigPresentationMetadata {
        label: Some(label),
        help: setting.description.clone(),
        category_id: Some(category.id.to_string()),
        category_label: Some(category.label.to_string()),
        category_summary: Some(category.summary.to_string()),
        category_order: Some(category.order),
        setting_order: Some(order),
        unit: unit_for_path(rendered).map(str::to_string),
        placeholder: placeholder_for_path(rendered).map(str::to_string),
        control_hint: control_hint_for_schema(&setting.value_schema).map(str::to_string),
        renderer_id: None,
        choices: vec![],
        display_units: vec![],
    });
}

fn fallback_category_for_path(rendered: &str) -> Option<CategoryPresentation> {
    if rendered.starts_with("gpu.") {
        return Some(MESHLLM_CATEGORY);
    }
    if rendered.starts_with("telemetry.") {
        return Some(TELEMETRY_CATEGORY);
    }
    if rendered.starts_with("runtime.") {
        return Some(RUNTIME_POLICY_CATEGORY);
    }
    if rendered.starts_with("owner_control.") {
        return Some(NETWORK_CATEGORY);
    }
    if rendered.starts_with("mesh_requirements.") {
        return Some(ATTESTATION_CATEGORY);
    }
    if rendered.starts_with("plugin.<plugin-name>.") {
        return Some(PLUGIN_HOST_CATEGORY);
    }
    if rendered.starts_with("models.<model-ref>.model_fit.") {
        return Some(MEMORY_CATEGORY);
    }
    if rendered.starts_with("models.<model-ref>.hardware.") {
        return Some(RUNTIME_CATEGORY);
    }
    if rendered.starts_with("models.<model-ref>.") {
        return Some(RUNTIME_CATEGORY);
    }
    if rendered.starts_with("defaults.speculative.") {
        return Some(SPECULATIVE_CATEGORY);
    }
    if rendered.starts_with("defaults.request_defaults.") {
        return Some(REQUEST_DEFAULTS_CATEGORY);
    }
    if rendered.starts_with("defaults.skippy.") {
        return Some(SKIPPY_CATEGORY);
    }
    if rendered.starts_with("defaults.multimodal.") {
        return Some(MULTIMODAL_CATEGORY);
    }
    if rendered.starts_with("defaults.advanced.server.") {
        return Some(ADVANCED_SERVER_CATEGORY);
    }
    if rendered.starts_with("defaults.model_fit.") {
        return Some(MEMORY_CATEGORY);
    }
    if rendered.starts_with("defaults.hardware.safety_margin_gb") {
        return Some(MEMORY_CATEGORY);
    }
    if rendered.starts_with("defaults.hardware.") || rendered.starts_with("defaults.throughput.") {
        return Some(RUNTIME_CATEGORY);
    }
    None
}

fn title_case_config_key(key: &str) -> String {
    key.replace('_', " ")
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn fallback_setting_order(rendered: &str) -> u32 {
    rendered.bytes().fold(0u32, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u32)
    })
}

fn unit_for_path(rendered: &str) -> Option<&'static str> {
    let leaf = rendered.rsplit('.').next()?;
    match leaf {
        "ctx_size" | "batch" | "ubatch" | "max_tokens" | "draft_max_tokens"
        | "draft_min_tokens" | "image_min_tokens" | "image_max_tokens" | "prefill_chunk_size" => {
            Some("tokens")
        }
        "threads" | "threads_batch" | "draft_threads" => Some("threads"),
        "parallel" | "cache_idle_slots" => Some("slots"),
        "safety_margin_gb" => Some("GB"),
        "cache_ram_mib" | "fit_target_mib" => Some("MiB"),
        "lifecycle_startup_timeout_ms"
        | "lifecycle_readiness_interval_ms"
        | "lifecycle_health_interval_ms" => Some("ms"),
        _ => None,
    }
}

fn placeholder_for_path(rendered: &str) -> Option<&'static str> {
    let leaf = rendered.rsplit('.').next()?;
    match leaf {
        "device" => Some("cuda:0 or CUDA0"),
        "gpu_layers" => Some("auto or integer layer count"),
        "tensor_split" => Some("e.g. 0.5,0.5"),
        "cpu_affinity" => Some("e.g. 0-3,8-11"),
        "priority" => Some("e.g. 0 or normal"),
        "stage_model_path" => Some("hf://... or /path/to/stage.gguf"),
        "stage_topology" => Some("topology name or path"),
        "prefill_chunk_schedule" => Some("e.g. 512,1024,2048"),
        "mmproj" => Some("/path/to/mmproj.gguf"),
        "mmproj_url" => Some("https://example.com/mmproj.gguf"),
        "server.alias" | "alias" => Some("model alias"),
        "command" => Some("use bundled plugin binary"),
        "args" => Some("comma-separated CLI arguments"),
        "url" => Some("http://localhost:8000/v1"),
        _ => None,
    }
}

fn control_hint_for_schema(schema: &ConfigValueSchema) -> Option<&'static str> {
    match schema {
        ConfigValueSchema::Boolean => Some("toggle"),
        ConfigValueSchema::Integer | ConfigValueSchema::Float => Some("number"),
        ConfigValueSchema::Enum { values }
            if values.len() <= MAX_ENUM_VALUES_FOR_SEGMENTED_CONTROL =>
        {
            Some("segmented")
        }
        ConfigValueSchema::Enum { .. } => Some("select"),
        ConfigValueSchema::OneOf { variants } if variants.iter().any(is_boolean_schema) => {
            Some("segmented")
        }
        ConfigValueSchema::Array { .. } => Some("text"),
        ConfigValueSchema::Object { properties } if properties.is_empty() => Some("textarea"),
        ConfigValueSchema::Object { .. } => None,
        _ => None,
    }
}

fn is_boolean_schema(schema: &ConfigValueSchema) -> bool {
    matches!(schema, ConfigValueSchema::Boolean)
}
