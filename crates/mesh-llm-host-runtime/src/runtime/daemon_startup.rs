// Copyright 2024 mesh-llm contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Daemon startup sequence for runtime initialization.
//!
//! This module handles the ordered startup of mesh components, mode resolution,
//! and failure policies during daemon initialization.

use crate::runtime::{RuntimeSurface, options::RuntimeOptions};
use mesh_llm_config::RuntimeMode;

/// Resolve effective runtime mode with priority:
/// client > shared endpoint > config > default serve
///
/// Sharing is an explicit per-invocation intent, so it overrides a persisted
/// `[runtime] mode` for this run only — nothing is written back. It resolves
/// to `OnDemand` rather than `Client`: the node must remain a full mesh
/// participant that claims the host role, while starting with no local
/// startup models.
pub(super) fn resolve_effective_mode(
    options: &RuntimeOptions,
    configured_mode: RuntimeMode,
) -> RuntimeMode {
    if options.client {
        RuntimeMode::Client
    } else if options.shared_endpoint.is_some() {
        RuntimeMode::OnDemand
    } else {
        configured_mode
    }
}

/// Check for conflicting flags and return error if found
pub(super) fn check_mode_conflicts(
    options: &RuntimeOptions,
    explicit_surface: Option<RuntimeSurface>,
    configured_mode: RuntimeMode,
) -> Result<(), String> {
    let has_explicit_model =
        !options.model.is_empty() || !options.gguf.is_empty() || options.mmproj.is_some();
    if options.client && has_explicit_model {
        return Err("client mode cannot be combined with --model, --gguf, or --mmproj".to_string());
    }
    if let Err(error) = options.validate_shared_endpoint_args() {
        return Err(error.to_string());
    }
    // An explicit sharing invocation deliberately overrides a persisted client
    // mode for this run; it is a serving intent, not a conflict.
    if options.shared_endpoint.is_some() {
        return Ok(());
    }
    if configured_mode == RuntimeMode::Client
        && (explicit_surface == Some(RuntimeSurface::Serve) || has_explicit_model)
    {
        return Err(
            "persisted runtime.mode is 'client', which cannot be overridden by serve or model \
             flags; change [runtime].mode to 'serve' or 'on_demand', or remove the conflicting \
             serve/model arguments"
                .to_string(),
        );
    }

    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::field_reassign_with_default,
    reason = "tests vary individual RuntimeOptions fields to keep each conflict scenario explicit"
)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_resolve_effective_mode_priority() {
        let options = RuntimeOptions::default();
        assert_eq!(
            resolve_effective_mode(&options, RuntimeMode::OnDemand),
            RuntimeMode::OnDemand
        );

        let mut client_options = RuntimeOptions::default();
        client_options.client = true;
        assert_eq!(
            resolve_effective_mode(&client_options, RuntimeMode::OnDemand),
            RuntimeMode::Client
        );
    }

    #[test]
    fn test_check_mode_conflicts_client_with_model() {
        let mut options = RuntimeOptions::default();
        options.client = true;
        options.model.push(PathBuf::from("test.gguf"));

        let result =
            check_mode_conflicts(&options, Some(RuntimeSurface::Client), RuntimeMode::Serve);

        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("client mode cannot be combined"));
    }

    #[test]
    fn persisted_client_rejects_explicit_serve_and_model_flags() {
        let options = RuntimeOptions::default();
        let error =
            check_mode_conflicts(&options, Some(RuntimeSurface::Serve), RuntimeMode::Client)
                .expect_err("explicit serve must not override persisted client mode");
        assert!(error.contains("change [runtime].mode"));

        let mut model_options = RuntimeOptions::default();
        model_options.model.push(PathBuf::from("test.gguf"));
        assert!(
            check_mode_conflicts(&model_options, None, RuntimeMode::Client).is_err(),
            "model flags must not override persisted client mode"
        );
    }

    fn sharing() -> RuntimeOptions {
        RuntimeOptions {
            shared_endpoint: Some("http://localhost:11434".to_string()),
            ..RuntimeOptions::default()
        }
    }

    #[test]
    fn sharing_overrides_persisted_client_mode_for_this_invocation() {
        assert_eq!(
            resolve_effective_mode(&sharing(), RuntimeMode::Client),
            RuntimeMode::OnDemand,
            "explicit endpoint sharing must not be demoted to client mode"
        );
    }

    #[test]
    fn sharing_overrides_persisted_serve_mode_without_startup_models() {
        assert_eq!(
            resolve_effective_mode(&sharing(), RuntimeMode::Serve),
            RuntimeMode::OnDemand
        );
    }

    #[test]
    fn explicit_client_still_wins_over_sharing_at_mode_resolution() {
        let mut options = sharing();
        options.client = true;
        assert_eq!(
            resolve_effective_mode(&options, RuntimeMode::Serve),
            RuntimeMode::Client
        );
    }

    #[test]
    fn sharing_does_not_conflict_with_persisted_client_mode() {
        check_mode_conflicts(&sharing(), Some(RuntimeSurface::Share), RuntimeMode::Client)
            .expect("sharing is an explicit serving intent, not a mode conflict");
    }

    #[test]
    fn sharing_with_a_local_model_is_rejected_at_the_mode_gate() {
        let mut options = sharing();
        options.model.push(PathBuf::from("test.gguf"));
        let error = check_mode_conflicts(&options, None, RuntimeMode::Serve)
            .expect_err("sharing must not accept a local model");
        assert!(error.contains("--model"));
    }

    #[test]
    fn sharing_combined_with_client_is_rejected() {
        let mut options = sharing();
        options.client = true;
        let error = check_mode_conflicts(&options, None, RuntimeMode::Serve)
            .expect_err("client cannot serve a shared endpoint to peers");
        assert!(error.contains("host role"));
    }

    #[test]
    fn normal_serve_mode_resolution_is_unchanged() {
        let options = RuntimeOptions::default();
        assert_eq!(
            resolve_effective_mode(&options, RuntimeMode::Serve),
            RuntimeMode::Serve
        );
        assert_eq!(
            resolve_effective_mode(&options, RuntimeMode::OnDemand),
            RuntimeMode::OnDemand
        );
    }
}
