use super::*;

/// A prefix restore can move an existing lane backwards to a shorter
/// common prefix. The tracked position must follow the imported native
/// state exactly; retaining the previous high-water mark submits the next
/// divergent token at the wrong position and makes llama_decode fail.
pub(super) fn record_restored_session_token_count(
    session_token_counts: &mut BTreeMap<String, u64>,
    session_id: &str,
    token_count: u64,
) {
    session_token_counts.insert(session_id.to_string(), token_count);
}

impl RuntimeState {
    pub fn export_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        let result = session.export_state(layer_start, layer_end);
        self.notify_export_outcome(&result);
        result
    }

    pub fn import_state(&mut self, session_id: &str, bytes: &[u8]) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        let result = session.import_state(layer_start, layer_end, bytes);
        self.notify_import_outcome(&result);
        result
    }

    pub fn import_state_for_token_count(
        &mut self,
        session_id: &str,
        bytes: &[u8],
        token_count: u64,
    ) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        let import_result =
            session.import_state_for_token_count(layer_start, layer_end, bytes, token_count);
        if import_result.is_ok() {
            record_restored_session_token_count(
                &mut self.session_token_counts,
                session_id,
                token_count,
            );
        }
        self.notify_import_outcome(&import_result);
        import_result
    }

    pub fn export_full_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        let result = session.export_full_state(layer_start, layer_end);
        self.notify_export_outcome(&result);
        result
    }

    pub fn import_full_state(&mut self, session_id: &str, bytes: &[u8]) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        let result = session.import_full_state(layer_start, layer_end, bytes);
        self.notify_import_outcome(&result);
        result
    }

    pub fn import_full_state_for_token_count(
        &mut self,
        session_id: &str,
        bytes: &[u8],
        token_count: u64,
    ) -> Result<()> {
        let layer_start = i32::try_from(self.model_layer_start())?;
        let layer_end = i32::try_from(self.model_layer_end())?;
        let session = self.session(session_id)?;
        let import_result =
            session.import_full_state_for_token_count(layer_start, layer_end, bytes, token_count);
        if import_result.is_ok() {
            record_restored_session_token_count(
                &mut self.session_token_counts,
                session_id,
                token_count,
            );
        }
        self.notify_import_outcome(&import_result);
        import_result
    }

    pub fn export_recurrent_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        let session = self.session(session_id)?;
        let result = session.export_recurrent_state();
        self.notify_export_outcome(&result);
        result
    }

    pub fn import_recurrent_state_for_token_count(
        &mut self,
        session_id: &str,
        bytes: &[u8],
        token_count: u64,
    ) -> Result<()> {
        let session = self.session(session_id)?;
        let import_result = session.import_recurrent_state_for_token_count(bytes, token_count);
        if import_result.is_ok() {
            record_restored_session_token_count(
                &mut self.session_token_counts,
                session_id,
                token_count,
            );
        }
        self.notify_import_outcome(&import_result);
        import_result
    }

    /// Reports the real result of any runtime-state export call (`export_state`,
    /// `export_full_state`, `export_recurrent_state`) to the attached observer.
    /// Never called before the native call returns, so success is only ever
    /// reported once native work has actually finished.
    fn notify_export_outcome<T>(&self, result: &Result<T>) {
        self.notify_session_lifecycle(if result.is_ok() {
            super::lifecycle::SessionLifecycleEvent::RuntimeStateExportCompleted
        } else {
            super::lifecycle::SessionLifecycleEvent::RuntimeStateExportFailed
        });
    }

    /// Reports the real result of the native import call only. Callers that
    /// also update `session_token_counts` bookkeeping (`record_restored_session_token_count`)
    /// do so BEFORE this notification and only on the `Ok` branch, so a
    /// reported `RuntimeStateImportCompleted` never precedes that bookkeeping
    /// -- consistent ordering with `import_kv_page`'s existing token-count
    /// update.
    fn notify_import_outcome<T>(&self, result: &Result<T>) {
        self.notify_session_lifecycle(if result.is_ok() {
            super::lifecycle::SessionLifecycleEvent::RuntimeStateImportCompleted
        } else {
            super::lifecycle::SessionLifecycleEvent::RuntimeStateImportFailed
        });
    }
}

#[cfg(test)]
#[path = "state_transfer/tests.rs"]
mod tests;
