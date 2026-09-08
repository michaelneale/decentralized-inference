use std::ffi::OsStr;

use anyhow::{Result, bail};

use crate::{LifecycleLogParserMode, MeshConfig};

pub const MESH_LLM_LIFECYCLE_LOG_PARSER_ENV: &str = "MESH_LLM_LIFECYCLE_LOG_PARSER";
pub const MESH_LLM_CONFIG_ENV: &str = "MESH_LLM_CONFIG";
/// Hidden, undocumented, TEST-ONLY gate -- see [`benchmark_tune_trial_enabled`].
pub const MESH_LLM_BENCHMARK_TUNE_TRIAL_ENV: &str = "MESH_LLM_BENCHMARK_TUNE_TRIAL";
/// Hidden, undocumented, TEST-ONLY selector -- see [`event_system_trial_mode`].
pub const MESH_LLM_EVENT_SYSTEM_TRIAL_MODE_ENV: &str = "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE";

/// Every `MESH_LLM_*` environment variable this crate owns: both true
/// `MeshConfig` overrides applied by [`apply_env_overrides`] and hidden
/// test-only gates such as [`benchmark_tune_trial_enabled`]. Every name
/// listed here must be read ONLY through this crate -- see
/// `crates/mesh-llm-host-runtime/src/plugin/config.rs`'s
/// `config_override_env_names_owner_totality_is_unchanged` tripwire, which
/// pins this list so a new override can never silently bypass the
/// plugin-aware production wrapper the way `MESH_LLM_LIFECYCLE_LOG_PARSER`
/// once did.
pub const CONFIG_OVERRIDE_ENV_NAMES: &[&str] = &[
    MESH_LLM_CONFIG_ENV,
    MESH_LLM_LIFECYCLE_LOG_PARSER_ENV,
    MESH_LLM_BENCHMARK_TUNE_TRIAL_ENV,
    MESH_LLM_EVENT_SYSTEM_TRIAL_MODE_ENV,
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConfigValueSource {
    Env,
    Config,
    #[default]
    Default,
}

impl ConfigValueSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Env => "env",
            Self::Config => "config",
            Self::Default => "default",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LifecycleLogParserSelection {
    pub mode: LifecycleLogParserMode,
    pub source: ConfigValueSource,
}

pub fn resolve_lifecycle_log_parser_override(
    configured: LifecycleLogParserMode,
    configured_source: ConfigValueSource,
    environment: Option<&OsStr>,
) -> Result<LifecycleLogParserSelection> {
    let Some(environment) = environment else {
        return Ok(LifecycleLogParserSelection {
            mode: configured,
            source: configured_source,
        });
    };
    let Some(environment) = environment.to_str() else {
        bail!("invalid {MESH_LLM_LIFECYCLE_LOG_PARSER_ENV}; expected auto, enabled, or disabled");
    };
    let mode = environment.parse().map_err(|_| {
        anyhow::anyhow!(
            "invalid {MESH_LLM_LIFECYCLE_LOG_PARSER_ENV}; expected auto, enabled, or disabled"
        )
    })?;
    Ok(LifecycleLogParserSelection {
        mode,
        source: ConfigValueSource::Env,
    })
}

pub(crate) fn config_path_override() -> Option<std::path::PathBuf> {
    std::env::var_os(MESH_LLM_CONFIG_ENV).map(std::path::PathBuf::from)
}

/// Applies every declared `MESH_LLM_*` config environment override to
/// `config` in place. This is the SINGLE typed entry point every config
/// loader — including plugin-aware wrappers outside this crate — must call
/// so environment overrides take effect consistently. An invalid override
/// value is a hard `Err`, never a silent fallback to the configured/default
/// value.
pub fn apply_env_overrides(config: &mut MeshConfig) -> Result<()> {
    let environment = std::env::var_os(MESH_LLM_LIFECYCLE_LOG_PARSER_ENV);
    if environment.is_none() {
        return Ok(());
    }
    let selection = resolve_lifecycle_log_parser_override(
        config.runtime.lifecycle_log_parser,
        config.runtime.lifecycle_log_parser_source,
        environment.as_deref(),
    )?;
    config.runtime.lifecycle_log_parser = selection.mode;
    config.runtime.lifecycle_log_parser_source = selection.source;
    Ok(())
}

/// Hidden, undocumented, TEST-ONLY gate. This is NOT a user-facing config
/// setting: do not document it, expose it as a CLI flag, or read
/// `MESH_LLM_BENCHMARK_TUNE_TRIAL` via an ad hoc `std::env::var` anywhere
/// outside this function. It exists so the benchmark-tune trial harness
/// (`mesh-llm benchmark tune`'s spawned trial children, and the
/// event-system A/B certification tooling layered on top of it) can prove a
/// process is genuinely running inside a controlled trial before accepting
/// further trial-only selectors gated on it (see the planned
/// `MESH_LLM_EVENT_SYSTEM_TRIAL_MODE` selector, which is accepted only when
/// this returns `true`).
pub fn benchmark_tune_trial_enabled() -> Result<bool> {
    resolve_benchmark_tune_trial_gate(
        std::env::var_os(MESH_LLM_BENCHMARK_TUNE_TRIAL_ENV).as_deref(),
    )
}

/// RAII guard for the test-only process-environment override helper below.
struct EnvOverrideGuard {
    name: String,
    previous: Option<std::ffi::OsString>,
}

impl EnvOverrideGuard {
    fn set(name: &str, value: &str) -> Self {
        let previous = std::env::var_os(name);
        // SAFETY: the enclosing test contract is `#[serial_test::serial]`, so no
        // other test's env read/write races this mutation.
        unsafe { std::env::set_var(name, value) };
        Self {
            name: name.to_owned(),
            previous,
        }
    }
}

impl Drop for EnvOverrideGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            // SAFETY: the enclosing test contract is `#[serial_test::serial]`, so no
            // other test's env read/write races this mutation.
            Some(previous) => unsafe { std::env::set_var(&self.name, previous) },
            // SAFETY: the enclosing test contract is `#[serial_test::serial]`, so no
            // other test's env read/write races this mutation.
            None => unsafe { std::env::remove_var(&self.name) },
        }
    }
}

/// Test-only helper: sets `name` to `value` for the duration of `f`, then
/// restores whatever `name` held before (or clears it if it was unset). The
/// unsafe environment mutation stays contained here, in the single typed
/// owner of every `MESH_LLM_*` override name, so a downstream crate with
/// `#![forbid(unsafe_code)]` (e.g. `mesh-llm-commands`) can still exercise
/// [`apply_env_overrides`] end to end in its own tests without writing
/// `unsafe` itself. Callers MUST annotate the enclosing test with
/// `#[serial_test::serial]` -- mutating the process environment races any
/// other test in the same binary otherwise.
pub fn with_env_override_for_test<T>(name: &str, value: &str, f: impl FnOnce() -> T) -> T {
    let _guard = EnvOverrideGuard::set(name, value);
    f()
}

/// Pure resolver behind [`benchmark_tune_trial_enabled`], mirroring
/// [`resolve_lifecycle_log_parser_override`]'s shape so it is testable
/// without mutating real process environment. Accepts exactly `"1"`
/// (enabled) or `"0"` (disabled, same as unset). Any other value is a hard
/// `Err` -- an invalid override value never silently falls back to a
/// default, matching this crate's existing precedent.
pub fn resolve_benchmark_tune_trial_gate(environment: Option<&OsStr>) -> Result<bool> {
    let Some(environment) = environment else {
        return Ok(false);
    };
    let Some(environment) = environment.to_str() else {
        bail!("invalid {MESH_LLM_BENCHMARK_TUNE_TRIAL_ENV}; expected 1 or 0");
    };
    match environment {
        "1" => Ok(true),
        "0" => Ok(false),
        _ => bail!("invalid {MESH_LLM_BENCHMARK_TUNE_TRIAL_ENV}; expected 1 or 0"),
    }
}

/// The event-system A/B certification trial selector. Hidden, undocumented,
/// TEST-ONLY -- see [`event_system_trial_mode`]. `production` runs the
/// complete event pipeline; `event-disabled` bypasses ONLY Progress and
/// Diagnostic class submissions at the host engine's single contract
/// boundary (`RuntimeEventEngine::submit`) and detaches every downstream
/// consumer for those two classes, while terminals, state transitions,
/// reservations, and the reducer remain fully active. These are the exact
/// two `--mode` values `scripts/run-event-benchmark-matrix.py` accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventSystemTrialMode {
    Production,
    EventDisabled,
}

/// Pure resolver behind [`event_system_trial_mode`]. `trial_gate_enabled`
/// is the caller's already-resolved [`resolve_benchmark_tune_trial_gate`]
/// value. Accepts exactly `"production"` or `"event-disabled"` when the
/// gate is on; setting this selector while the gate is OFF is a hard
/// `Err`, never a silent fallback to `Production` -- the selector only
/// exists inside a controlled trial, matching this crate's existing
/// precedent of never silently falling back on an invalid override.
pub fn resolve_event_system_trial_mode(
    trial_gate_enabled: bool,
    environment: Option<&OsStr>,
) -> Result<Option<EventSystemTrialMode>> {
    let Some(environment) = environment else {
        return Ok(None);
    };
    if !trial_gate_enabled {
        bail!(
            "{MESH_LLM_EVENT_SYSTEM_TRIAL_MODE_ENV} requires {MESH_LLM_BENCHMARK_TUNE_TRIAL_ENV}=1"
        );
    }
    let Some(environment) = environment.to_str() else {
        bail!(
            "invalid {MESH_LLM_EVENT_SYSTEM_TRIAL_MODE_ENV}; expected production or event-disabled"
        );
    };
    let mode = match environment {
        "production" => EventSystemTrialMode::Production,
        "event-disabled" => EventSystemTrialMode::EventDisabled,
        _ => bail!(
            "invalid {MESH_LLM_EVENT_SYSTEM_TRIAL_MODE_ENV}; expected production or event-disabled"
        ),
    };
    Ok(Some(mode))
}

/// Hidden, undocumented, TEST-ONLY selector. This is NOT a user-facing
/// config setting: do not document it, expose it as a CLI flag, or read
/// `MESH_LLM_EVENT_SYSTEM_TRIAL_MODE` via an ad hoc `std::env::var`
/// anywhere outside this function. Composes [`benchmark_tune_trial_enabled`]
/// with [`resolve_event_system_trial_mode`], so a caller never has to
/// re-derive the gate/selector relationship itself.
pub fn event_system_trial_mode() -> Result<Option<EventSystemTrialMode>> {
    let gate = benchmark_tune_trial_enabled()?;
    resolve_event_system_trial_mode(
        gate,
        std::env::var_os(MESH_LLM_EVENT_SYSTEM_TRIAL_MODE_ENV).as_deref(),
    )
}

/// Convenience boolean for the host engine's single class-bypass contract
/// boundary: `true` only when the trial gate is on AND the trial mode
/// selector is explicitly `event-disabled`. Every other state (gate off,
/// selector unset, selector `production`) resolves to `false`.
pub fn event_system_progress_diagnostic_bypass_enabled() -> Result<bool> {
    Ok(matches!(
        event_system_trial_mode()?,
        Some(EventSystemTrialMode::EventDisabled)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_overrides_owner_lists_every_config_override_name() {
        assert_eq!(
            CONFIG_OVERRIDE_ENV_NAMES,
            &[
                "MESH_LLM_CONFIG",
                "MESH_LLM_LIFECYCLE_LOG_PARSER",
                "MESH_LLM_BENCHMARK_TUNE_TRIAL",
                "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE"
            ]
        );
    }

    #[test]
    #[serial_test::serial]
    fn env_override_is_restored_when_the_test_body_panics() {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        const NAME: &str = "MESH_LLM_TEST_ENV_OVERRIDE_PANIC";
        let previous = std::env::var_os(NAME);
        let result = catch_unwind(AssertUnwindSafe(|| {
            with_env_override_for_test(NAME, "temporary", || panic!("test panic"));
        }));

        assert!(result.is_err());
        assert_eq!(std::env::var_os(NAME), previous);
    }
}

#[cfg(test)]
mod event_system_trial_mode_tests {
    use super::*;

    #[test]
    fn none_when_unset_regardless_of_gate() {
        assert_eq!(resolve_event_system_trial_mode(false, None).unwrap(), None);
        assert_eq!(resolve_event_system_trial_mode(true, None).unwrap(), None);
    }

    #[test]
    fn errors_hard_when_selector_set_but_gate_is_off() {
        let error = resolve_event_system_trial_mode(false, Some(OsStr::new("production")))
            .expect_err(
                "selector without the trial gate must be a hard error, never a silent fallback",
            );
        assert!(
            error
                .to_string()
                .contains(MESH_LLM_EVENT_SYSTEM_TRIAL_MODE_ENV)
        );
        assert!(
            error
                .to_string()
                .contains(MESH_LLM_BENCHMARK_TUNE_TRIAL_ENV)
        );
    }

    #[test]
    fn accepts_production_when_gate_is_on() {
        assert_eq!(
            resolve_event_system_trial_mode(true, Some(OsStr::new("production"))).unwrap(),
            Some(EventSystemTrialMode::Production)
        );
    }

    #[test]
    fn accepts_event_disabled_when_gate_is_on() {
        assert_eq!(
            resolve_event_system_trial_mode(true, Some(OsStr::new("event-disabled"))).unwrap(),
            Some(EventSystemTrialMode::EventDisabled)
        );
    }

    #[test]
    fn rejects_invalid_value_as_hard_error_even_with_gate_on() {
        let error = resolve_event_system_trial_mode(true, Some(OsStr::new("bogus")))
            .expect_err("non production/event-disabled value must be a hard error");
        assert!(
            error
                .to_string()
                .contains(MESH_LLM_EVENT_SYSTEM_TRIAL_MODE_ENV)
        );
    }

    #[test]
    fn bypass_enabled_helper_is_true_only_for_event_disabled() {
        assert!(matches!(
            resolve_event_system_trial_mode(true, Some(OsStr::new("event-disabled"))).unwrap(),
            Some(EventSystemTrialMode::EventDisabled)
        ));
        assert!(!matches!(
            resolve_event_system_trial_mode(true, Some(OsStr::new("production"))).unwrap(),
            Some(EventSystemTrialMode::EventDisabled)
        ));
        assert!(!matches!(
            resolve_event_system_trial_mode(true, None).unwrap(),
            Some(EventSystemTrialMode::EventDisabled)
        ));
    }

    #[test]
    fn env_reading_wrapper_defaults_to_none_when_unset() {
        // Exercises the real env-reading wrapper (not just the pure
        // resolver) so a future refactor cannot silently detach it from
        // `std::env::var_os`. Does not mutate the environment, so it needs
        // no serialization guard against other tests in this binary --
        // matching `benchmark_trial_gate_env_reading_wrapper_defaults_to_disabled`'s
        // own precedent.
        if std::env::var_os(MESH_LLM_EVENT_SYSTEM_TRIAL_MODE_ENV).is_none() {
            assert_eq!(event_system_trial_mode().unwrap(), None);
            assert!(!event_system_progress_diagnostic_bypass_enabled().unwrap());
        }
    }
}

#[cfg(test)]
mod benchmark_trial_gate_tests {
    use super::*;

    #[test]
    fn benchmark_trial_gate_defaults_to_disabled_when_unset() {
        assert!(!resolve_benchmark_tune_trial_gate(None).unwrap());
    }

    #[test]
    fn benchmark_trial_gate_accepts_one_as_enabled() {
        assert!(resolve_benchmark_tune_trial_gate(Some(OsStr::new("1"))).unwrap());
    }

    #[test]
    fn benchmark_trial_gate_accepts_zero_as_disabled() {
        assert!(!resolve_benchmark_tune_trial_gate(Some(OsStr::new("0"))).unwrap());
    }

    #[test]
    fn benchmark_trial_gate_rejects_invalid_value_as_hard_error() {
        let error = resolve_benchmark_tune_trial_gate(Some(OsStr::new("yes")))
            .expect_err("non 1/0 value must be a hard error, never a silent fallback");
        assert!(
            error
                .to_string()
                .contains(MESH_LLM_BENCHMARK_TUNE_TRIAL_ENV)
        );
    }

    #[test]
    fn benchmark_trial_gate_env_reading_wrapper_defaults_to_disabled() {
        // The real env-reading wrapper is exercised at all (not just the
        // pure resolver) so a future refactor cannot silently detach it
        // from `std::env::var_os(MESH_LLM_BENCHMARK_TUNE_TRIAL_ENV)`. This
        // does not mutate the environment, so it needs no serialization
        // guard against other tests in this binary.
        if std::env::var_os(MESH_LLM_BENCHMARK_TUNE_TRIAL_ENV).is_none() {
            assert!(!benchmark_tune_trial_enabled().unwrap());
        }
    }
}
