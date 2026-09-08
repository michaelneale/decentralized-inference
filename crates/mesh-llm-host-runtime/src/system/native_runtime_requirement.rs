//! Decides whether startup must have a loadable native runtime.
//!
//! A node that only forwards inference to an external OpenAI-compatible
//! server (Ollama, LM Studio, LiteLLM, vLLM) through a configured plugin
//! never calls into the Skippy FFI. Refusing to start such a node because no
//! native runtime exists for its hardware makes the plugin path unusable on
//! exactly the machines it is for.

use mesh_llm_config::MeshConfig;

/// Whether a missing native runtime is fatal for this startup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeRuntimeRequirement {
    /// Local serving is possible, so a missing runtime must fail startup.
    Required,
    /// Every model this node can serve comes from an external inference
    /// plugin, so a missing runtime is a warning and startup continues.
    OptionalExternalInference,
}

/// Inputs that decide the requirement, kept separate from config parsing so
/// the decision itself is directly testable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeRuntimeRequirementInput {
    /// A local model was requested on the command line or in config.
    pub(crate) local_serving_requested: bool,
    /// At least one enabled plugin entry supplies an external inference URL.
    pub(crate) external_inference_configured: bool,
}

pub(crate) const fn native_runtime_requirement(
    input: NativeRuntimeRequirementInput,
) -> NativeRuntimeRequirement {
    if input.external_inference_configured && !input.local_serving_requested {
        NativeRuntimeRequirement::OptionalExternalInference
    } else {
        NativeRuntimeRequirement::Required
    }
}

/// `true` when config declares an enabled plugin entry carrying a URL.
///
/// A URL is what makes a plugin an inference *endpoint*; plugins configured
/// without one (for example the in-tree blobstore) do not serve models.
pub(crate) fn config_declares_external_inference(config: &MeshConfig) -> bool {
    config.plugins.iter().any(|plugin| {
        plugin.enabled.unwrap_or(true)
            && plugin
                .url
                .as_deref()
                .is_some_and(|url| !url.trim().is_empty())
    })
}

/// Warning emitted when startup continues without a native runtime.
pub(crate) fn external_inference_only_startup_warning(error: &str) -> String {
    format!(
        "starting without a MeshLLM native runtime: {error}. This node can only serve models \
         supplied by its configured external inference plugin, and cannot load local GGUF or \
         layer-package models. Install a runtime with `mesh-llm runtime install` to serve models \
         locally."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_config::PluginConfigEntry;

    fn plugin(name: &str, url: Option<&str>, enabled: Option<bool>) -> PluginConfigEntry {
        PluginConfigEntry {
            name: name.to_string(),
            enabled,
            web_ui_enabled: None,
            command: None,
            args: Vec::new(),
            url: url.map(str::to_string),
            settings: Default::default(),
            startup: Default::default(),
        }
    }

    fn config_with(plugins: Vec<PluginConfigEntry>) -> MeshConfig {
        MeshConfig {
            plugins,
            ..MeshConfig::default()
        }
    }

    #[test]
    fn external_endpoint_without_local_models_makes_runtime_optional() {
        assert_eq!(
            native_runtime_requirement(NativeRuntimeRequirementInput {
                local_serving_requested: false,
                external_inference_configured: true,
            }),
            NativeRuntimeRequirement::OptionalExternalInference
        );
    }

    #[test]
    fn requested_local_model_still_requires_a_runtime() {
        assert_eq!(
            native_runtime_requirement(NativeRuntimeRequirementInput {
                local_serving_requested: true,
                external_inference_configured: true,
            }),
            NativeRuntimeRequirement::Required
        );
    }

    #[test]
    fn no_external_endpoint_requires_a_runtime() {
        assert_eq!(
            native_runtime_requirement(NativeRuntimeRequirementInput {
                local_serving_requested: false,
                external_inference_configured: false,
            }),
            NativeRuntimeRequirement::Required
        );
    }

    #[test]
    fn detects_enabled_plugin_url() {
        assert!(config_declares_external_inference(&config_with(vec![
            plugin("openai-endpoint", Some("http://127.0.0.1:11434/v1"), None),
        ])));
    }

    #[test]
    fn disabled_plugin_does_not_count() {
        assert!(!config_declares_external_inference(&config_with(vec![
            plugin(
                "openai-endpoint",
                Some("http://127.0.0.1:11434/v1"),
                Some(false)
            ),
        ])));
    }

    #[test]
    fn plugin_without_url_does_not_count() {
        assert!(!config_declares_external_inference(&config_with(vec![
            plugin("blobstore", None, None),
        ])));
    }

    #[test]
    fn blank_url_does_not_count() {
        assert!(!config_declares_external_inference(&config_with(vec![
            plugin("openai-endpoint", Some("   "), None),
        ])));
    }

    #[test]
    fn warning_names_the_cause_and_the_limitation() {
        let warning = external_inference_only_startup_warning("no compatible native runtime found");
        assert!(warning.contains("no compatible native runtime found"));
        assert!(warning.contains("external inference plugin"));
        assert!(warning.contains("mesh-llm runtime install"));
    }
}
