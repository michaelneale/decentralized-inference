use super::*;
use anyhow::bail;

#[derive(Default)]
struct FakeRestoreRollback {
    cleanup_error: Option<&'static str>,
    reacquire_error: Option<&'static str>,
    position_error: Option<&'static str>,
    native_position: u64,
    token_count: u64,
    discarded: bool,
}

impl RestoreRollback for FakeRestoreRollback {
    fn drop_dirty_session(&mut self, _session_id: &str) -> Result<()> {
        match self.cleanup_error {
            Some(message) => bail!(message),
            None => Ok(()),
        }
    }

    fn reacquire_clean_session(&mut self, _session_id: &str) -> Result<()> {
        match self.reacquire_error {
            Some(message) => bail!(message),
            None => Ok(()),
        }
    }

    fn restored_native_position(&mut self, _session_id: &str) -> Result<u64> {
        match self.position_error {
            Some(message) => bail!(message),
            None => Ok(self.native_position),
        }
    }

    fn restored_token_count(&self, _session_id: &str) -> u64 {
        self.token_count
    }

    fn discard_dirty_session(&mut self, _session_id: &str) {
        self.discarded = true;
    }
}

#[test]
fn restore_rollback_returns_original_error_after_proving_a_clean_lane() {
    let mut runtime = FakeRestoreRollback::default();

    let error = rollback_restore_failure(
        &mut runtime,
        "lane-a",
        anyhow::anyhow!("injected import failure"),
    );

    assert_eq!(
        error.to_string(),
        "cache restore transaction rolled back for session lane-a"
    );
    assert!(format!("{error:#}").contains("injected import failure"));
    assert!(!runtime.discarded);
}

#[test]
fn restore_rollback_discards_when_cleanup_or_reacquire_fails() {
    for mut runtime in [
        FakeRestoreRollback {
            cleanup_error: Some("cleanup failed"),
            ..FakeRestoreRollback::default()
        },
        FakeRestoreRollback {
            reacquire_error: Some("reacquire failed"),
            ..FakeRestoreRollback::default()
        },
    ] {
        let error =
            rollback_restore_failure(&mut runtime, "lane-a", anyhow::anyhow!("restore failed"));
        assert!(format!("{error:#}").contains("restore failed"));
        assert!(runtime.discarded);
    }
}

#[test]
fn restore_rollback_discards_unverifiable_or_dirty_lanes() {
    for mut runtime in [
        FakeRestoreRollback {
            position_error: Some("position unavailable"),
            ..FakeRestoreRollback::default()
        },
        FakeRestoreRollback {
            native_position: 1,
            ..FakeRestoreRollback::default()
        },
        FakeRestoreRollback {
            token_count: 1,
            ..FakeRestoreRollback::default()
        },
    ] {
        let error =
            rollback_restore_failure(&mut runtime, "lane-a", anyhow::anyhow!("restore failed"));
        assert!(format!("{error:#}").contains("restore failed"));
        assert!(runtime.discarded);
    }
}
