pub(crate) use crate::diagnostic::DiagnosticResult;
pub use crate::diagnostic::{
    ConfigDiagnostic, ConfigDiagnosticCode, ConfigDiagnosticSchemaSource, ConfigDiagnosticSeverity,
    ConfigDiagnosticSource, alias_diagnostic, invalid_value_diagnostic,
    legacy_validation_error_text, rejected_field_diagnostic, unsupported_field_diagnostic,
};
use crate::model_validation::{
    collect_legacy_draft_model_path_warnings, model_topology_diagnostics,
    validate_duplicate_model_entries, validate_model_defaults, validate_model_entry,
};
use crate::plugin_validation::{
    PluginSchemaAvailability, validate_plugin_entries, validate_plugin_entries_strict,
};
pub(crate) use crate::validation_support::validation_diagnostic;
pub use crate::validation_support::{
    built_in_support_diagnostic, canonical_builtin_diagnostic_path,
};
use crate::validation_support::{
    parse_node_version, validate_optional_http_url, validate_release_signer_key_shape,
    version_precedence_cmp,
};
use crate::*;
use anyhow::Result;

pub fn validate_config_diagnostics(config: &MeshConfig) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();

    if let Some(version) = config.version
        && version != 1
    {
        diagnostics.push(validation_diagnostic(
            "version",
            format!("unsupported config version {version}; expected version = 1"),
        ));
    }
    if let Some(bind) = config.owner_control.bind
        && bind.port() == 0
        && !bind.ip().is_loopback()
    {
        diagnostics.push(validation_diagnostic(
            "owner_control.bind",
            "owner_control.bind must use a concrete port when binding a non-loopback address",
        ));
    }
    if let Some(advertise_addr) = config.owner_control.advertise_addr {
        match config.owner_control.bind {
            Some(bind) if bind.port() == 0 => {
                diagnostics.push(validation_diagnostic(
                    "owner_control.bind",
                    "owner_control.bind must use a concrete port when owner_control.advertise_addr is set",
                ));
            }
            Some(bind) if bind.port() != advertise_addr.port() => {
                diagnostics.push(validation_diagnostic(
                    "owner_control.advertise_addr",
                    "owner_control.advertise_addr must use the same port as owner_control.bind",
                ));
            }
            Some(_) => {}
            None => {
                diagnostics.push(validation_diagnostic(
                    "owner_control.advertise_addr",
                    "owner_control.advertise_addr requires owner_control.bind so the advertised port is actually listening",
                ));
            }
        }
        if advertise_addr.port() == 0 {
            diagnostics.push(validation_diagnostic(
                "owner_control.advertise_addr",
                "owner_control.advertise_addr must use a concrete port",
            ));
        }
        if advertise_addr.ip().is_unspecified() {
            diagnostics.push(validation_diagnostic(
                "owner_control.advertise_addr",
                "owner_control.advertise_addr must not use an unspecified IP address",
            ));
        }
    }
    if let Some(parallel) = config.gpu.parallel
        && parallel < 1
    {
        diagnostics.push(validation_diagnostic(
            "gpu.parallel",
            format!("gpu.parallel must be at least 1, got {parallel}"),
        ));
    }
    if let Err(diagnostic) = validate_mesh_requirements_config(&config.mesh_requirements) {
        diagnostics.push(diagnostic);
    }
    if let Err(diagnostic) = validate_telemetry_config(&config.telemetry) {
        diagnostics.push(diagnostic);
    }
    diagnostics.extend(validate_logging_config(&config.logging));
    diagnostics.extend(validate_audit_config(&config.logging.audit));

    diagnostics.extend(validate_runtime_config(&config.runtime));
    if let Err(diagnostic) = validate_plugin_entries(&config.plugins) {
        diagnostics.push(diagnostic);
    }
    let defaults_hardware = config
        .defaults
        .as_ref()
        .and_then(|defaults| defaults.hardware.as_ref());
    if let Some(defaults) = &config.defaults
        && let Err(diagnostic) =
            validate_model_defaults(defaults, "defaults", config.gpu.assignment)
    {
        diagnostics.push(diagnostic);
    }
    let mut inherited_topology_diagnostic_keys = std::collections::BTreeSet::new();
    for (index, model) in config.models.iter().enumerate() {
        if model.model.trim().is_empty() {
            diagnostics.push(validation_diagnostic(
                &format!("models[{index}].model"),
                format!("models[{index}].model must not be empty"),
            ));
        }
        if let Err(diagnostic) = validate_model_entry(
            model,
            &format!("models[{index}]"),
            config.gpu.assignment,
            defaults_hardware,
        ) {
            diagnostics.push(diagnostic);
        }
        diagnostics.extend(model_topology_diagnostics(
            config.defaults.as_ref(),
            model,
            &format!("models[{index}]"),
            &mut inherited_topology_diagnostic_keys,
        ));
    }

    collect_legacy_draft_model_path_warnings(config, &mut diagnostics);

    validate_duplicate_model_entries(&config.models, config.defaults.as_ref(), &mut diagnostics);

    let existing_paths: std::collections::BTreeSet<String> = diagnostics
        .iter()
        .filter_map(|diagnostic| diagnostic.path.as_ref().map(ConfigPath::render))
        .collect();
    diagnostics.extend(
        crate::wiring_validation::wiring_manifest_diagnostics(config)
            .into_iter()
            .filter(|diagnostic| {
                diagnostic
                    .path
                    .as_ref()
                    .is_none_or(|path| !existing_paths.contains(&path.render()))
            }),
    );

    diagnostics
}

fn validate_audit_config(config: &AuditConfig) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();
    if config.enabled == Some(true)
        && config.log_path.as_ref().is_none_or(|path| {
            path.as_os_str().is_empty() || path.to_string_lossy().trim().is_empty()
        })
    {
        diagnostics.push(validation_diagnostic(
            "logging.audit.log_path",
            "logging.audit.log_path must be set when logging.audit.enabled is true",
        ));
    }
    if config
        .log_format
        .as_deref()
        .is_some_and(|format| format != "json_lines")
    {
        diagnostics.push(validation_diagnostic(
            "logging.audit.log_format",
            "logging.audit.log_format must be json_lines",
        ));
    }
    if config.max_file_size_mb.is_some_and(|value| value == 0) {
        diagnostics.push(validation_diagnostic(
            "logging.audit.max_file_size_mb",
            "logging.audit.max_file_size_mb must be at least 1",
        ));
    }
    if config.max_files.is_some_and(|value| value == 0) {
        diagnostics.push(validation_diagnostic(
            "logging.audit.max_files",
            "logging.audit.max_files must be at least 1",
        ));
    }
    diagnostics
}

fn validate_runtime_config(config: &RuntimeConfig) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();
    let mesh_version = config.native_runtime.mesh_version.as_deref();
    let skippy_abi = config.native_runtime.skippy_abi.as_deref();
    let selection = config.native_runtime.selection.as_deref();
    if mesh_version.is_none() && (skippy_abi.is_some() || selection.is_some()) {
        diagnostics.push(validation_diagnostic(
            "runtime.native_runtime",
            "runtime.native_runtime override must set mesh_version when skippy_abi or selection is set",
        ));
    }
    if matches!(mesh_version, Some(value) if value.trim().is_empty()) {
        diagnostics.push(validation_diagnostic(
            "runtime.native_runtime.mesh_version",
            "runtime.native_runtime.mesh_version must not be empty",
        ));
    }
    if matches!(skippy_abi, Some(value) if value.trim().is_empty()) {
        diagnostics.push(validation_diagnostic(
            "runtime.native_runtime.skippy_abi",
            "runtime.native_runtime.skippy_abi must not be empty",
        ));
    }
    if matches!(selection, Some(value) if value.trim().is_empty()) {
        diagnostics.push(validation_diagnostic(
            "runtime.native_runtime.selection",
            "runtime.native_runtime.selection must not be empty",
        ));
    }
    for (path, value, min, max) in [
        (
            "runtime.activity.idle_after_secs",
            config.activity.idle_after_secs,
            30,
            86_400,
        ),
        (
            "runtime.activity.poll_interval_secs",
            config.activity.poll_interval_secs,
            1,
            60,
        ),
        (
            "runtime.activity.resume_debounce_secs",
            config.activity.resume_debounce_secs,
            0,
            300,
        ),
    ] {
        if !(min..=max).contains(&value) {
            diagnostics.push(validation_diagnostic(
                path,
                format!("{path} must be between {min} and {max}, got {value}"),
            ));
        }
    }
    if config.drain_timeout_secs == 0 {
        diagnostics.push(validation_diagnostic(
            "runtime.drain_timeout_secs",
            "runtime.drain_timeout_secs must be at least 1",
        ));
    }
    if config.drain_timeout_max_secs == 0 {
        diagnostics.push(validation_diagnostic(
            "runtime.drain_timeout_max_secs",
            "runtime.drain_timeout_max_secs must be at least 1",
        ));
    } else if config.drain_timeout_secs > config.drain_timeout_max_secs {
        diagnostics.push(validation_diagnostic(
            "runtime.drain_timeout_secs",
            "runtime.drain_timeout_secs must not exceed runtime.drain_timeout_max_secs",
        ));
    }
    diagnostics.extend(validate_kv_disk_config(&config.kv_cache.disk));
    diagnostics
}

fn validate_kv_disk_config(config: &KvDiskTierConfig) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();
    match config.mode {
        Some(KvDiskTierMode::Fixed) if config.budget_mib.is_none() => {
            diagnostics.push(validation_diagnostic(
                "runtime.kv_cache.disk.budget_mib",
                "runtime.kv_cache.disk.budget_mib is required when mode = \"fixed\"",
            ));
        }
        Some(KvDiskTierMode::Off | KvDiskTierMode::Auto) | None if config.budget_mib.is_some() => {
            diagnostics.push(validation_diagnostic(
                "runtime.kv_cache.disk.budget_mib",
                "runtime.kv_cache.disk.budget_mib is only valid when mode = \"fixed\"",
            ));
        }
        _ => {}
    }
    if config.budget_mib == Some(0) {
        diagnostics.push(validation_diagnostic(
            "runtime.kv_cache.disk.budget_mib",
            "runtime.kv_cache.disk.budget_mib must be greater than zero",
        ));
    }
    if config
        .minimum_free_mib
        .is_some_and(|minimum| minimum < MIN_KV_DISK_MINIMUM_FREE_MIB)
    {
        diagnostics.push(validation_diagnostic(
            "runtime.kv_cache.disk.minimum_free_mib",
            format!(
                "runtime.kv_cache.disk.minimum_free_mib must be at least {MIN_KV_DISK_MINIMUM_FREE_MIB}"
            ),
        ));
    }
    for (path, value) in [
        ("runtime.kv_cache.disk.budget_mib", config.budget_mib),
        (
            "runtime.kv_cache.disk.minimum_free_mib",
            config.minimum_free_mib,
        ),
    ] {
        if value.is_some_and(|mib| mib.checked_mul(1024 * 1024).is_none()) {
            diagnostics.push(validation_diagnostic(
                path,
                format!("{path} is too large to represent as bytes"),
            ));
        }
    }
    if let Some(directory) = config.directory.as_deref() {
        let rendered = directory.to_string_lossy();
        // A drive letter is absolute only where the host understands one. On
        // Unix `C:\cache` is a relative path, and accepting it would put a
        // durable cache under the process working directory.
        let windows_absolute = cfg!(windows)
            && rendered.as_bytes().get(1) == Some(&b':')
            && rendered
                .as_bytes()
                .get(2)
                .is_some_and(|byte| matches!(byte, b'/' | b'\\'));
        if rendered.trim().is_empty() {
            diagnostics.push(validation_diagnostic(
                "runtime.kv_cache.disk.directory",
                "runtime.kv_cache.disk.directory must not be empty",
            ));
        } else if !directory.is_absolute() && !windows_absolute {
            diagnostics.push(validation_diagnostic(
                "runtime.kv_cache.disk.directory",
                "runtime.kv_cache.disk.directory must be an absolute path",
            ));
        }
    }
    diagnostics
}

pub fn validate_config_diagnostics_with_plugin_schemas<F>(
    config: &MeshConfig,
    raw_toml: Option<&str>,
    schema_for_plugin: F,
) -> Vec<ConfigDiagnostic>
where
    F: FnMut(&str) -> PluginSchemaAvailability,
{
    let mut diagnostics = validate_config_diagnostics(config);
    diagnostics.extend(validate_plugin_entries_strict(
        &config.plugins,
        raw_toml,
        schema_for_plugin,
    ));
    diagnostics
}

pub fn validate_config(config: &MeshConfig) -> Result<()> {
    let diagnostics = validate_config_diagnostics(config);
    let has_errors = diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error);
    if has_errors {
        Err(anyhow::anyhow!(legacy_validation_error_text(&diagnostics)))
    } else {
        Ok(())
    }
}

pub fn validate_config_with_plugin_schemas<F>(
    config: &MeshConfig,
    raw_toml: Option<&str>,
    schema_for_plugin: F,
) -> Result<()>
where
    F: FnMut(&str) -> PluginSchemaAvailability,
{
    let diagnostics =
        validate_config_diagnostics_with_plugin_schemas(config, raw_toml, schema_for_plugin);
    let has_errors = diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error);
    if has_errors {
        Err(anyhow::anyhow!(legacy_validation_error_text(&diagnostics)))
    } else {
        Ok(())
    }
}

fn validate_mesh_requirements_config(config: &MeshRequirementsConfig) -> DiagnosticResult {
    let min_node_version = config
        .min_node_version
        .as_deref()
        .map(|value| parse_node_version(value, "mesh_requirements.min_node_version"))
        .transpose()?;
    let max_node_version = config
        .max_node_version
        .as_deref()
        .map(|value| parse_node_version(value, "mesh_requirements.max_node_version"))
        .transpose()?;
    if let (Some(min), Some(max)) = (&min_node_version, &max_node_version)
        && version_precedence_cmp(min, max).is_gt()
    {
        return Err(validation_diagnostic(
            "mesh_requirements.min_node_version",
            "mesh_requirements.min_node_version must be less than or equal to mesh_requirements.max_node_version",
        ));
    }

    if let (Some(min), Some(max)) = (config.min_protocol_version, config.max_protocol_version)
        && min > max
    {
        return Err(validation_diagnostic(
            "mesh_requirements.min_protocol_version",
            "mesh_requirements.min_protocol_version must be less than or equal to mesh_requirements.max_protocol_version",
        ));
    }

    for signer_key in &config.release_signer_keys {
        validate_release_signer_key_shape(signer_key, "mesh_requirements.release_signer_keys")?;
    }
    if config.require_release_attestation && config.release_signer_keys.is_empty() {
        return Err(validation_diagnostic(
            "mesh_requirements.require_release_attestation",
            "mesh_requirements.require_release_attestation is true but mesh_requirements.release_signer_keys is empty; certified-build admission is not remote runtime attestation, so trust must be anchored in at least one release signer key",
        ));
    }

    Ok(())
}

fn validate_telemetry_config(config: &TelemetryConfig) -> DiagnosticResult {
    if let Some(service_name) = &config.service_name {
        let trimmed = service_name.trim();
        if !trimmed.is_empty() {
            // Validate service name: alphanumeric, dash, underscore only
            if !trimmed
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Err(validation_diagnostic(
                    "telemetry.service_name",
                    "telemetry.service_name must contain only alphanumeric characters, dashes, and underscores",
                ));
            }
        }
    }
    validate_optional_http_url(config.endpoint.as_deref(), "telemetry.endpoint")?;
    validate_optional_http_url(
        config.metrics.endpoint.as_deref(),
        "telemetry.metrics.endpoint",
    )?;
    for key in config.headers.keys() {
        if key.trim().is_empty() {
            return Err(validation_diagnostic(
                "telemetry.headers",
                "telemetry.headers keys must not be empty",
            ));
        }
    }
    if let Some(export_interval_secs) = config.export_interval_secs
        && export_interval_secs < 1
    {
        return Err(validation_diagnostic(
            "telemetry.export_interval_secs",
            "telemetry.export_interval_secs must be at least 1",
        ));
    }
    if let Some(queue_size) = config.queue_size
        && queue_size < 1
    {
        return Err(validation_diagnostic(
            "telemetry.queue_size",
            "telemetry.queue_size must be at least 1",
        ));
    }
    Ok(())
}

fn validate_logging_config(config: &crate::LoggingConfig) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();

    // Application state root validation.
    if let Some(root) = &config.application_state_root {
        validate_application_state_root(root, &mut diagnostics);
    }

    // Numeric bounds validation.
    for (path, value, min, max) in [
        (
            "logging.summary_line_limit",
            config.summary_line_limit,
            1_u64,
            65_536,
        ),
        (
            "logging.event_buffer_size",
            config.event_buffer_size,
            50,
            100_000,
        ),
        (
            "logging.retention_ttl_secs",
            config.retention_ttl_secs,
            3600,      // minimum 1 hour
            7_776_000, // maximum 90 days
        ),
        (
            "logging.retention_max_rows",
            config.retention_max_rows,
            1,
            1_000_000,
        ),
    ] {
        if !(min..=max).contains(&value) {
            diagnostics.push(validation_diagnostic(
                path,
                format!("{path} must be between {min} and {max}, got {value}"),
            ));
        }
    }

    for (path, value, min, max) in [
        (
            "logging.replay_capacity",
            config.replay_capacity as u64,
            1_u64,
            10_000,
        ),
        (
            "logging.queue_capacity",
            config.queue_capacity as u64,
            64,
            131_072,
        ),
    ] {
        if !(min..=max).contains(&value) {
            diagnostics.push(validation_diagnostic(
                path,
                format!("{path} must be between {} and {}, got {}", min, max, value),
            ));
        }
    }

    if config.replay_capacity as u64 > config.event_buffer_size {
        diagnostics.push(validation_diagnostic(
            "logging.replay_capacity",
            "logging.replay_capacity must not exceed logging.event_buffer_size",
        ));
    }

    // Artifact byte limits.
    for (path, value, min, max) in [
        (
            "logging.artifact.byte_limit_bytes",
            config.artifact.byte_limit_bytes,
            1024_u64,
            16 * 1024 * 1024,
        ),
        (
            "logging.artifact.aggregate_limit_bytes",
            config.artifact.aggregate_limit_bytes,
            512 * 1024,        // minimum 512 KiB
            500 * 1024 * 1024, // maximum 500 MiB
        ),
    ] {
        if !(min..=max).contains(&value) {
            diagnostics.push(validation_diagnostic(
                path,
                format!("{path} must be between {} and {}, got {}", min, max, value),
            ));
        }
    }

    // Export limit.
    let export_min = 64 * 1024_u64; // 64 KiB
    let export_max = 100 * 1024 * 1024_u64; // 100 MiB
    if !(export_min..=export_max).contains(&config.export_limit_bytes) {
        diagnostics.push(validation_diagnostic(
            "logging.export_limit_bytes",
            format!(
                "logging.export_limit_bytes must be between {} and {}, got {}",
                export_min, export_max, config.export_limit_bytes
            ),
        ));
    }

    // Cleanup cadence.
    if !(300..=86_400).contains(&config.cleanup_cadence_secs) {
        diagnostics.push(validation_diagnostic(
            "logging.cleanup_cadence_secs",
            format!(
                "logging.cleanup_cadence_secs must be between 300 and 86400, got {}",
                config.cleanup_cadence_secs
            ),
        ));
    }

    // Webhook validation.
    validate_webhook_config(&config.webhook, &mut diagnostics);

    diagnostics
}

fn validate_application_state_root(
    root: &std::path::PathBuf,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if root.as_os_str().is_empty() {
        diagnostics.push(validation_diagnostic(
            "logging.application_state_root",
            "logging.application_state_root must not be empty",
        ));
        return;
    }

    let contains_parent_dir = root
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
        // A configuration can be authored on another OS, so reject Windows
        // separators even when validation runs on Unix (and vice versa).
        || root
            .to_string_lossy()
            .split(['/', '\\'])
            .any(|segment| segment == "..");
    if contains_parent_dir {
        diagnostics.push(validation_diagnostic(
            "logging.application_state_root",
            "logging.application_state_root must not contain directory traversal sequences",
        ));
    }

    // Reject absolute system paths that should never contain application state.
    let forbidden_roots = ["/etc", "/dev", "/proc", "/sys"];
    if let Some(path_str) = root.to_str() {
        if path_str == "/" {
            diagnostics.push(validation_diagnostic(
                "logging.application_state_root",
                "logging.application_state_root must not be the filesystem root \"/\"",
            ));
        } else {
            for forbidden_root in forbidden_roots {
                let is_descendant = path_str
                    .strip_prefix(forbidden_root)
                    .is_some_and(|suffix| suffix.starts_with('/'));
                if path_str == forbidden_root || is_descendant {
                    diagnostics.push(validation_diagnostic(
                    "logging.application_state_root",
                    format!(
                            "logging.application_state_root must not target system directories; rejecting path at or below \"{forbidden_root}\""
                    ),
                ));
                    break;
                }
            }
        }
    }

    // On Unix, check for world-writable if path exists.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(metadata) = std::fs::metadata(root) {
            let mode = metadata.mode() & 0o777;
            if mode & 0o002 != 0 {
                diagnostics.push(validation_diagnostic(
                    "logging.application_state_root",
                    format!(
                        "logging.application_state_root must not be world-writable; current mode is {:04o}",
                        metadata.mode() & 0o7777
                    ),
                ));
            }
        }
    }
}

fn validate_webhook_config(
    config: &crate::LoggingWebhookConfig,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let configured_url = config
        .url
        .as_deref()
        .map(str::trim)
        .filter(|url| !url.is_empty());
    if config.enabled && configured_url.is_none() {
        diagnostics.push(validation_diagnostic(
            "logging.webhook.url",
            "logging.webhook.url is required when logging.webhook.enabled is true",
        ));
    }
    if let Some(url) = configured_url {
        if let Err(diag) = validate_optional_http_url(Some(url), "logging.webhook.url") {
            diagnostics.push(diag);
        } else if let Ok(parsed) = url::Url::parse(url)
            && (!parsed.username().is_empty()
                || parsed.password().is_some()
                || parsed.query().is_some()
                || parsed.fragment().is_some())
        {
            diagnostics.push(validation_diagnostic(
                "logging.webhook.url",
                "logging.webhook.url must not include credentials, a query string, or a fragment",
            ));
        }
    }

    for (path, value, min, max) in [
        (
            "logging.webhook.max_attempts",
            config.max_attempts as u64,
            1_u64,
            20,
        ),
        (
            "logging.webhook.timeout_secs",
            config.timeout_secs,
            1_u64,
            60,
        ),
    ] {
        if !(min..=max).contains(&value) {
            diagnostics.push(validation_diagnostic(
                path,
                format!("{path} must be between {} and {}, got {}", min, max, value),
            ));
        }
    }

    // Dead-letter retention.
    let dlr = config.dead_letter_retention_secs;
    if !(3600..=1_555_200).contains(&dlr) {
        diagnostics.push(validation_diagnostic(
            "logging.webhook.dead_letter_retention_secs",
            format!(
                "logging.webhook.dead_letter_retention_secs must be between 3600 and 1555200, got {}",
                dlr
            ),
        ));
    }
}

#[cfg(test)]
mod schema_tests {
    use super::*;

    #[test]
    fn audit_rotation_limits_must_be_nonzero() {
        let config: MeshConfig = toml::from_str(
            r#"
[logging.audit]
max_file_size_mb = 0
max_files = 0
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let paths = diagnostics
            .iter()
            .filter_map(|diagnostic| diagnostic.path.as_ref().map(ConfigPath::render))
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec![
                "logging.audit.max_file_size_mb".to_string(),
                "logging.audit.max_files".to_string()
            ]
        );
    }

    #[test]
    fn enabled_audit_requires_a_path_and_json_lines_format() {
        let config: MeshConfig = toml::from_str(
            r#"
[logging.audit]
enabled = true
log_format = "json"
"#,
        )
        .expect("config should parse before validation");

        let paths = validate_config_diagnostics(&config)
            .iter()
            .filter_map(|diagnostic| diagnostic.path.as_ref().map(ConfigPath::render))
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec!["logging.audit.log_path", "logging.audit.log_format"]
        );
    }

    #[test]
    fn audit_json_lines_format_is_the_only_valid_exported_format() {
        let config: MeshConfig = toml::from_str(
            r#"
[logging.audit]
enabled = true
log_path = "audit.jsonl"
log_format = "json_lines"
"#,
        )
        .expect("config should parse before validation");

        assert!(
            validate_config_diagnostics(&config)
                .iter()
                .all(
                    |diagnostic| diagnostic.path.as_ref().map(ConfigPath::render)
                        != Some("logging.audit.log_format".to_string())
                )
        );
    }

    #[test]
    fn logging_summary_and_event_buffer_limits_accept_boundaries_and_reject_underflow() {
        let mut config = MeshConfig::default();
        config.logging.summary_line_limit = 1;
        config.logging.event_buffer_size = 50;
        config.logging.replay_capacity = 50;
        assert!(validate_config_diagnostics(&config).is_empty());

        config.logging.summary_line_limit = 0;
        config.logging.event_buffer_size = 49;
        let paths = validate_config_diagnostics(&config)
            .iter()
            .filter_map(|diagnostic| diagnostic.path.as_ref().map(ConfigPath::render))
            .collect::<Vec<_>>();
        assert_eq!(
            paths,
            vec![
                "logging.summary_line_limit".to_string(),
                "logging.event_buffer_size".to_string(),
                "logging.replay_capacity".to_string(),
            ]
        );
    }

    #[test]
    fn application_state_root_rejects_parent_components_on_all_platforms() {
        for root in ["../outside", "state/../outside", r"..\outside"] {
            let mut config = MeshConfig::default();
            config.logging.application_state_root = Some(root.into());
            assert!(
                validate_config_diagnostics(&config)
                    .iter()
                    .any(|diagnostic| {
                        diagnostic.path.as_ref().map(ConfigPath::render)
                            == Some("logging.application_state_root".to_string())
                    })
            );
        }
    }

    #[test]
    fn application_state_root_rejects_exact_system_directories() {
        for root in ["/etc", "/dev", "/proc", "/sys"] {
            let mut config = MeshConfig::default();
            config.logging.application_state_root = Some(root.into());
            assert!(
                validate_config_diagnostics(&config)
                    .iter()
                    .any(|diagnostic| {
                        diagnostic.path.as_ref().map(ConfigPath::render)
                            == Some("logging.application_state_root".to_string())
                    }),
                "{root} must be rejected"
            );
        }
    }

    #[test]
    fn enabled_audit_rejects_an_empty_log_path() {
        let mut config = MeshConfig::default();
        config.logging.audit.enabled = Some(true);
        config.logging.audit.log_path = Some(std::path::PathBuf::new());

        assert!(
            validate_config_diagnostics(&config)
                .iter()
                .any(
                    |diagnostic| diagnostic.path.as_ref().map(ConfigPath::render)
                        == Some("logging.audit.log_path".to_string())
                )
        );
    }

    #[test]
    fn drain_default_cannot_exceed_maximum() {
        let mut config = MeshConfig::default();
        config.runtime.drain_timeout_secs = 301;
        config.runtime.drain_timeout_max_secs = 300;
        let diagnostics = validate_config_diagnostics(&config);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].path.as_ref().map(ConfigPath::render),
            Some("runtime.drain_timeout_secs".to_string())
        );
    }

    #[test]
    fn owner_control_advertise_addr_requires_matching_bind_port() {
        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert!(
            legacy_validation_error_text(&diagnostics).contains(
                "owner_control.advertise_addr requires owner_control.bind so the advertised port is actually listening"
            )
        );

        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "127.0.0.1:17002"
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert!(
            legacy_validation_error_text(&diagnostics).contains(
                "owner_control.advertise_addr must use the same port as owner_control.bind"
            )
        );

        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "127.0.0.1:0"
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert!(legacy_validation_error_text(&diagnostics).contains(
            "owner_control.bind must use a concrete port when owner_control.advertise_addr is set"
        ));

        let config: MeshConfig = toml::from_str(
            r#"
[owner_control]
bind = "127.0.0.1:17001"
advertise_addr = "127.0.0.1:17001"
"#,
        )
        .expect("config should parse before validation");

        validate_config(&config).expect("matching bind and advertise ports should validate");
    }

    #[test]
    fn structured_diagnostics_report_canonical_path_for_alias_backed_invalid_input() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[gpu]
assignment = "auto"

[[models]]
model = "Qwen3-4B-Q4_K_M"
gpu_id = "metal:0"
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        let diagnostic = diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.canonical_path.as_ref().map(ConfigPath::render)
                    == Some("models.<model-ref>.hardware.device".to_string())
            })
            .expect("legacy gpu_id path should yield a canonical device diagnostic");

        assert_eq!(diagnostic.code, ConfigDiagnosticCode::InvalidValue);
        assert_eq!(diagnostic.severity, ConfigDiagnosticSeverity::Error);
        assert_eq!(
            diagnostic.schema_source,
            Some(ConfigDiagnosticSchemaSource::BuiltIn)
        );
        assert_eq!(
            diagnostic.path.as_ref().map(ConfigPath::render),
            Some("models[0].hardware.device".to_string())
        );
        assert_eq!(
            diagnostic.canonical_path.as_ref().map(ConfigPath::render),
            Some("models.<model-ref>.hardware.device".to_string())
        );
        assert_eq!(
            diagnostic.message,
            "models[0].hardware.device must not be set when gpu.assignment = \"auto\""
        );
    }

    #[test]
    fn legacy_validation_errors_derive_compatible_string_messages() {
        let config: MeshConfig = toml::from_str(
            r#"
version = 1

[[plugin]]
name = "metrics"
command = "mesh-llm-plugin-metrics"

[plugin.startup]
connect_timeout_secs = 0
"#,
        )
        .expect("config should parse before validation");

        let diagnostics = validate_config_diagnostics(&config);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            legacy_validation_error_text(&diagnostics),
            "plugin[0].startup.connect_timeout_secs must be at least 1 when set"
        );

        let err =
            validate_config(&config).expect_err("legacy validation surface should still fail");
        assert_eq!(
            err.to_string(),
            "plugin[0].startup.connect_timeout_secs must be at least 1 when set"
        );
    }

    #[test]
    fn runtime_activity_bounds_reject_out_of_range_values() {
        for (toml, expected_path, expected_value) in [
            (
                r#"
[runtime.activity]
idle_after_secs = 29
"#,
                "runtime.activity.idle_after_secs",
                29_u64,
            ),
            (
                r#"
[runtime.activity]
idle_after_secs = 86401
"#,
                "runtime.activity.idle_after_secs",
                86_401_u64,
            ),
            (
                r#"
[runtime.activity]
poll_interval_secs = 0
"#,
                "runtime.activity.poll_interval_secs",
                0_u64,
            ),
            (
                r#"
[runtime.activity]
poll_interval_secs = 61
"#,
                "runtime.activity.poll_interval_secs",
                61_u64,
            ),
            (
                r#"
[runtime.activity]
resume_debounce_secs = 301
"#,
                "runtime.activity.resume_debounce_secs",
                301_u64,
            ),
        ] {
            let config: MeshConfig =
                toml::from_str(toml).expect("config should parse before validation");

            let diagnostics = validate_config_diagnostics(&config);
            assert_eq!(diagnostics.len(), 1);

            let diagnostic = &diagnostics[0];
            assert_eq!(diagnostic.code, ConfigDiagnosticCode::InvalidValue);
            assert_eq!(diagnostic.severity, ConfigDiagnosticSeverity::Error);
            assert_eq!(
                diagnostic.schema_source,
                Some(ConfigDiagnosticSchemaSource::BuiltIn)
            );
            assert_eq!(
                diagnostic.path.as_ref().map(ConfigPath::render),
                Some(expected_path.to_string())
            );
            assert_eq!(
                diagnostic.canonical_path.as_ref().map(ConfigPath::render),
                Some(expected_path.to_string())
            );
            assert!(diagnostic.message.contains(expected_path));
            assert!(diagnostic.message.contains(&expected_value.to_string()));
        }
    }

    #[test]
    fn runtime_activity_bounds_accept_in_range_edges() {
        for toml in [
            r#"
[runtime.activity]
idle_after_secs = 30
poll_interval_secs = 60
resume_debounce_secs = 0
"#,
            r#"
[runtime.activity]
idle_after_secs = 86400
poll_interval_secs = 1
resume_debounce_secs = 300
"#,
        ] {
            let config: MeshConfig =
                toml::from_str(toml).expect("config should parse before validation");

            let diagnostics = validate_config_diagnostics(&config);
            assert!(diagnostics.is_empty());
        }
    }

    #[test]
    fn config_v1_without_logging_section_applies_bounded_defaults() {
        // Regression: existing v1 configs without [logging] must parse cleanly,
        // produce zero diagnostics, and apply bounded metadata-only defaults.
        let toml = r#"
[runtime]
listen_all = true

[owner_control]
bind = "0.0.0.0:9337"
"#;

        let config: MeshConfig =
            toml::from_str(toml).expect("v1 config without [logging] should parse");

        // Zero validation warnings for missing logging section.
        let diagnostics = validate_config_diagnostics(&config);
        assert!(
            diagnostics.is_empty(),
            "Expected zero diagnostics; got {:?}",
            diagnostics
        );

        // Logging subsystem enabled by default with bounded defaults.
        assert!(
            config.logging.enabled,
            "logging.enabled should default to true"
        );
        assert_eq!(config.logging.summary_line_limit, 2048);
        assert_eq!(config.logging.event_buffer_size, 10_000);
        assert_eq!(config.logging.retention_ttl_secs, 36 * 3600); // 36 hours
        assert_eq!(config.logging.retention_max_rows, 100_000);
        assert_eq!(config.logging.replay_capacity, 128);
        assert_eq!(config.logging.queue_capacity, 4096);
        assert_eq!(config.logging.export_limit_bytes, 5 * 1024 * 1024); // 5 MB
        assert_eq!(config.logging.cleanup_cadence_secs, 3600);

        // Artifact capture defaults to safe metadata-only (no raw content).
        use crate::model::CaptureMode;
        assert_eq!(
            config.logging.artifact.capture_mode,
            CaptureMode::MetadataOnly
        );

        // Webhook disabled by default.
        assert!(!config.logging.webhook.enabled);
    }

    #[test]
    fn logging_retention_max_rows_reports_a_stable_validation_path() {
        let mut config = crate::MeshConfig::default();
        config.logging.retention_max_rows = 0;

        let diagnostics = validate_config_diagnostics(&config);
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic
                .path
                .as_ref()
                .is_some_and(|path| path.render() == "logging.retention_max_rows")
                && diagnostic.message.contains("must be between 1 and 1000000")
        }));
    }

    #[test]
    fn enabled_webhook_requires_a_plain_http_url_without_secret_components() {
        let mut config = crate::MeshConfig::default();
        config.logging.webhook.enabled = true;

        let diagnostics = validate_config_diagnostics(&config);
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic
                .path
                .as_ref()
                .is_some_and(|path| path.render() == "logging.webhook.url")
                && diagnostic.message.contains("required")
        }));

        config.logging.webhook.url =
            Some("https://operator:secret@example.invalid/hook?token=secret#fragment".into());
        let diagnostics = validate_config_diagnostics(&config);
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic
                .path
                .as_ref()
                .is_some_and(|path| path.render() == "logging.webhook.url")
                && diagnostic.message.contains("must not include credentials")
        }));
        assert!(diagnostics.iter().all(|diagnostic| {
            !diagnostic.message.contains("operator")
                && !diagnostic.message.contains("secret")
                && !diagnostic.message.contains("example.invalid")
        }));

        config.logging.webhook.url = Some("http://127.0.0.1:4567/hook".into());
        assert!(validate_config_diagnostics(&config).is_empty());
    }
}
