//! Loads one native serving plugin and adapts its stable ABI to Skippy hooks.

mod plugin_dispatch;
#[cfg(test)]
mod test_support;
mod tokenizer_capability;

use std::{
    collections::HashMap,
    ffi::{c_char, c_void},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;
use mesh_native_serving_plugin_api as abi;
use plugin_dispatch::{PluginCommand, PluginDriver};
use skippy_server::frontend::{
    CompositeGenerationLifecycleIngress, GenerationAbort, GenerationCommit,
    GenerationLifecycleIngress, GenerationLifecycleObservation, GenerationReceipt,
    GenerationReceiptConfig, GenerationStart, LinearProposal, LinearProposalDiscardReason,
    LinearProposalDisposition, LinearProposalIngress, LinearProposalIngressConfig,
    LinearProposalQuery, LinearProposalReceipt, LinearProposalSourceResponse,
    OpaqueProposalDecisionId,
};
use skippy_server::serving_hooks::{ModelServingHooks, ModelServingHooksFactory};
use skippy_server::tokenizer::TokenizerCapability;
use tokenizer_capability::HostTokenizerCapability;

const ERROR_BUFFER_BYTES: usize = 2_048;
const MAX_NATIVE_PLUGIN_PROPOSAL_TOKENS: usize = 4_096;
const PROPOSAL_POLL_INTERVAL: Duration = Duration::from_micros(50);

const _: () = assert!(
    abi::MAX_TOKENIZER_INPUT_PIECES == skippy_tokenizer::MAX_TOKENIZE_PIECES,
    "ABI and tokenizer piece bounds must match",
);

/// Mesh-owned factory for one independently built native serving plugin.
#[derive(Clone)]
pub struct NativeServingPluginFactory {
    definition: Arc<LoadedDefinition>,
    config_path: PathBuf,
    state_directory: PathBuf,
    proposal_deadline: Duration,
}

impl NativeServingPluginFactory {
    /// Loads and validates a native plugin without activating model-specific state.
    pub fn load(
        library_path: &Path,
        config_path: PathBuf,
        state_directory: PathBuf,
        proposal_deadline: Duration,
    ) -> Result<Self> {
        validate_absolute_path("native serving plugin", library_path)?;
        validate_absolute_path("native serving plugin config", &config_path)?;
        validate_absolute_path("native serving plugin state", &state_directory)?;
        if !library_path.is_file() {
            bail!(
                "native serving plugin must be an existing file: {}",
                library_path.display()
            );
        }
        if !config_path.is_file() {
            bail!(
                "native serving plugin config must be an existing file: {}",
                config_path.display()
            );
        }
        if !state_directory.is_dir() {
            bail!(
                "native serving plugin state must be an existing directory: {}",
                state_directory.display()
            );
        }
        if proposal_deadline.is_zero() {
            bail!("native serving plugin proposal deadline must be greater than zero");
        }
        let definition = LoadedDefinition::load(library_path)?;
        Ok(Self {
            definition: Arc::new(definition),
            config_path,
            state_directory,
            proposal_deadline,
        })
    }
}

impl ModelServingHooksFactory for NativeServingPluginFactory {
    fn create(
        &self,
        tokenizer: TokenizerCapability,
        extra_generation_sink: Option<Arc<dyn GenerationLifecycleIngress>>,
    ) -> Result<ModelServingHooks> {
        let tokenizer_capability = HostTokenizerCapability::new(tokenizer)?;
        let identity = tokenizer_capability.tokenizer.identity();
        let context = abi::ActivationContext {
            struct_size: size_of::<abi::ActivationContext>(),
            model_id: abi::ByteSlice::from_bytes(identity.model_id.as_bytes()),
            source_model_sha256: abi::ByteSlice::from_bytes(
                identity.source_model_sha256.as_bytes(),
            ),
            tokenizer_id: abi::ByteSlice::from_bytes(identity.tokenizer_id.as_bytes()),
            tokenizer_capability: &tokenizer_capability.abi,
            config_path: path_slice(&self.config_path),
            state_directory: path_slice(&self.state_directory),
            proposal_deadline_ns: u64::try_from(self.proposal_deadline.as_nanos())
                .unwrap_or(u64::MAX),
            host_clock_context: std::ptr::null_mut(),
            monotonic_now_ns,
        };
        let mut activation = abi::PluginActivation {
            instance: std::ptr::null_mut(),
        };
        let status = unsafe { (self.definition.api().activate)(&context, &raw mut activation) };
        if status != abi::PluginStatus::OK {
            return Err(self.definition.status_error(
                activation.instance,
                "activate native serving plugin",
                status,
            ));
        }
        let instance = NonNull::new(activation.instance)
            .context("native serving plugin returned a null active instance")?;
        let deadline = self.proposal_deadline;
        let active = ActivePlugin {
            definition: Arc::clone(&self.definition),
            instance: Some(instance),
            _tokenizer_capability: Some(tokenizer_capability),
            proposal_token_buffer: Mutex::new(vec![0; MAX_NATIVE_PLUGIN_PROPOSAL_TOKENS]),
            committed_generated_tokens: Mutex::new(HashMap::new()),
        };
        let driver = Arc::new(PluginDriver::spawn(active)?);
        let native_lifecycle: Arc<dyn GenerationLifecycleIngress> =
            Arc::new(NativeLifecycleIngress {
                driver: Arc::clone(&driver),
            });
        let lifecycle: Arc<dyn GenerationLifecycleIngress> = match extra_generation_sink {
            Some(extra) => Arc::new(CompositeGenerationLifecycleIngress::new(vec![
                native_lifecycle,
                extra,
            ])),
            None => native_lifecycle,
        };
        let proposals: Arc<dyn LinearProposalIngress> = Arc::new(NativeProposalIngress { driver });
        Ok(ModelServingHooks::new(
            GenerationReceiptConfig::from_lifecycle_ingress(lifecycle),
            LinearProposalIngressConfig::new(
                proposals,
                deadline,
                MAX_NATIVE_PLUGIN_PROPOSAL_TOKENS,
            )?,
        ))
    }
}

struct LoadedDefinition {
    _library: Option<Library>,
    api: NonNull<abi::NativeServingPluginV2>,
    name: String,
}

// SAFETY: the ABI contract requires the static function table and loaded
// library to support concurrent host calls for the lifetime of the library.
unsafe impl Send for LoadedDefinition {}
// SAFETY: see the `Send` contract above; the table is immutable after load.
unsafe impl Sync for LoadedDefinition {}

impl LoadedDefinition {
    fn load(path: &Path) -> Result<Self> {
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("load native serving plugin {}", path.display()))?;
        let entry = unsafe {
            library.get::<abi::NativeServingPluginEntryV2>(abi::NATIVE_SERVING_PLUGIN_ENTRY_V2)
        }
        .with_context(|| {
            format!(
                "resolve native serving plugin entrypoint in {}",
                path.display()
            )
        })?;
        let api = NonNull::new(unsafe { entry() }.cast_mut())
            .context("native serving plugin entrypoint returned null")?;
        let name = validate_table(unsafe { api.as_ref() })?;
        Ok(Self {
            _library: Some(library),
            api,
            name,
        })
    }

    fn api(&self) -> &abi::NativeServingPluginV2 {
        unsafe { self.api.as_ref() }
    }

    fn status_error(
        &self,
        instance: abi::PluginInstance,
        action: &str,
        status: abi::PluginStatus,
    ) -> anyhow::Error {
        let detail = self.last_error(instance);
        anyhow!("{action} `{}` failed with {status:?}: {detail}", self.name)
    }

    fn last_error(&self, instance: abi::PluginInstance) -> String {
        let mut buffer = [0_u8; ERROR_BUFFER_BYTES];
        let written = unsafe {
            (self.api().last_error)(instance, buffer.as_mut_ptr().cast::<c_char>(), buffer.len())
        };
        let length = written.min(buffer.len());
        String::from_utf8_lossy(&buffer[..length]).into_owned()
    }
}

fn validate_table(table: &abi::NativeServingPluginV2) -> Result<String> {
    if table.abi_version != abi::NATIVE_SERVING_PLUGIN_ABI_V2 {
        bail!(
            "native serving plugin ABI {} is incompatible with host ABI {}",
            table.abi_version,
            abi::NATIVE_SERVING_PLUGIN_ABI_V2
        );
    }
    if table.struct_size != size_of::<abi::NativeServingPluginV2>() {
        bail!(
            "native serving plugin table size {} does not match host size {}",
            table.struct_size,
            size_of::<abi::NativeServingPluginV2>()
        );
    }
    let name = unsafe { read_utf8(table.plugin_name, "plugin name") }?;
    if name.trim().is_empty() {
        bail!("native serving plugin name must not be empty");
    }
    Ok(name)
}

struct ActivePlugin {
    definition: Arc<LoadedDefinition>,
    instance: Option<NonNull<c_void>>,
    _tokenizer_capability: Option<Arc<HostTokenizerCapability>>,
    proposal_token_buffer: Mutex<Vec<i32>>,
    committed_generated_tokens: Mutex<HashMap<(u64, u64), usize>>,
}

// SAFETY: activation succeeds only for plugins implementing the ABI's
// thread-safe instance contract. Mesh never mutates the opaque pointer.
unsafe impl Send for ActivePlugin {}
// SAFETY: see the `Send` contract above.
unsafe impl Sync for ActivePlugin {}

impl ActivePlugin {
    fn instance(&self) -> Result<abi::PluginInstance> {
        self.instance
            .map(NonNull::as_ptr)
            .context("native serving plugin is already shut down")
    }

    fn call_status(&self, action: &str, status: abi::PluginStatus) -> Result<()> {
        if status == abi::PluginStatus::OK {
            return Ok(());
        }
        Err(self.definition.status_error(
            self.instance.map_or(std::ptr::null_mut(), NonNull::as_ptr),
            action,
            status,
        ))
    }

    fn begin(&self, start: &GenerationStart) -> Result<()> {
        let agent_session_id = start.agent_session_id.as_deref().unwrap_or_default();
        let event = abi::GenerationStart {
            struct_size: size_of::<abi::GenerationStart>(),
            request_id: start.request_id,
            session_id: start.session_id,
            agent_session_id: abi::ByteSlice::from_bytes(agent_session_id.as_bytes()),
            prompt_token_ids: abi::TokenSlice::from_tokens(&start.prompt_token_ids),
        };
        let status = unsafe { (self.definition.api().begin_generation)(self.instance()?, &event) };
        self.call_status("begin generation", status)?;
        self.committed_generated_tokens
            .lock()
            .map_err(|_| anyhow!("native serving plugin commit state lock poisoned"))?
            .insert((start.request_id, start.session_id), 0);
        Ok(())
    }

    fn committed(&self, commit: &GenerationCommit) -> Result<()> {
        self.commit_tokens(
            commit.request_id,
            commit.session_id,
            commit.generated_token_count,
            &commit.token_ids,
        )
    }

    fn commit_tokens(
        &self,
        request_id: u64,
        session_id: u64,
        generated_token_count: usize,
        token_ids: &[i32],
    ) -> Result<()> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let mut committed_generated_tokens = self
            .committed_generated_tokens
            .lock()
            .map_err(|_| anyhow!("native serving plugin commit state lock poisoned"))?;
        self.commit_tokens_locked(
            &mut committed_generated_tokens,
            request_id,
            session_id,
            generated_token_count,
            token_ids,
        )
    }

    fn commit_tokens_locked(
        &self,
        committed_generated_tokens: &mut HashMap<(u64, u64), usize>,
        request_id: u64,
        session_id: u64,
        generated_token_count: usize,
        token_ids: &[i32],
    ) -> Result<()> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let key = (request_id, session_id);
        let prior_generated_token_count = committed_generated_tokens
            .get(&key)
            .copied()
            .context("native serving plugin commit has no active generation state")?;
        let expected_generated_token_count = prior_generated_token_count
            .checked_add(token_ids.len())
            .context("native serving plugin generated-token count overflow")?;
        if generated_token_count != expected_generated_token_count {
            bail!(
                "native serving plugin commit does not extend the tracked generated-token prefix"
            );
        }
        let event = abi::GenerationCommit {
            struct_size: size_of::<abi::GenerationCommit>(),
            request_id,
            session_id,
            generated_token_count: u64::try_from(generated_token_count)?,
            token_ids: abi::TokenSlice::from_tokens(token_ids),
        };
        let status = unsafe { (self.definition.api().commit_generation)(self.instance()?, &event) };
        self.call_status("commit generation", status)?;
        committed_generated_tokens.insert(key, generated_token_count);
        Ok(())
    }

    fn abort(&self, abort: &GenerationAbort) -> Result<()> {
        let event = abi::GenerationAbort {
            struct_size: size_of::<abi::GenerationAbort>(),
            request_id: abort.request_id,
            session_id: abort.session_id,
        };
        let status = unsafe { (self.definition.api().abort_generation)(self.instance()?, &event) };
        let result = self.call_status("abort generation", status);
        self.committed_generated_tokens
            .lock()
            .map_err(|_| anyhow!("native serving plugin commit state lock poisoned"))?
            .remove(&(abort.request_id, abort.session_id));
        result
    }

    fn finish(&self, receipt: &GenerationReceipt) -> Result<()> {
        let key = (receipt.request_id, receipt.session_id);
        self.commit_final_suffix(
            receipt.request_id,
            receipt.session_id,
            &receipt.generated_token_ids,
        )?;
        let request_to_first_token_us = receipt.request_to_first_token_us.unwrap_or_default();
        let event = abi::GenerationFinish {
            struct_size: size_of::<abi::GenerationFinish>(),
            request_id: receipt.request_id,
            session_id: receipt.session_id,
            prompt_token_count: u64::try_from(receipt.prompt_token_count)?,
            prompt_token_digest: receipt.prompt_token_digest,
            prompt_token_ids: abi::TokenSlice::from_tokens(&receipt.prompt_token_ids),
            generated_token_ids: abi::TokenSlice::from_tokens(&receipt.generated_token_ids),
            final_session_position: receipt.final_session_position,
            termination: convert_termination(receipt.termination),
            model_generation_elapsed_us: receipt.model_generation_elapsed_us,
            has_request_to_first_token: receipt.request_to_first_token_us.is_some(),
            request_to_first_token_us,
            request_to_token_emission_us: abi::U64Slice::from_values(
                &receipt.request_to_token_emission_us,
            ),
        };
        let status = unsafe { (self.definition.api().finish_generation)(self.instance()?, &event) };
        let result = self.call_status("finish generation", status);
        self.committed_generated_tokens
            .lock()
            .map_err(|_| anyhow!("native serving plugin commit state lock poisoned"))?
            .remove(&key);
        result
    }

    fn commit_final_suffix(
        &self,
        request_id: u64,
        session_id: u64,
        generated_token_ids: &[i32],
    ) -> Result<()> {
        let mut committed_generated_tokens = self
            .committed_generated_tokens
            .lock()
            .map_err(|_| anyhow!("native serving plugin commit state lock poisoned"))?;
        let key = (request_id, session_id);
        let committed = committed_generated_tokens
            .get(&key)
            .copied()
            .unwrap_or_default();
        if committed > generated_token_ids.len() {
            bail!("native serving plugin committed beyond the final generation receipt");
        }
        if committed < generated_token_ids.len() {
            self.commit_tokens_locked(
                &mut committed_generated_tokens,
                request_id,
                session_id,
                generated_token_ids.len(),
                &generated_token_ids[committed..],
            )?;
        }
        Ok(())
    }

    fn ensure_generated_token_count(
        &self,
        request_id: u64,
        session_id: u64,
        expected_generated_token_count: usize,
    ) -> Result<()> {
        let committed_generated_tokens = self
            .committed_generated_tokens
            .lock()
            .map_err(|_| anyhow!("native serving plugin commit state lock poisoned"))?;
        let Some(actual_generated_token_count) = committed_generated_tokens
            .get(&(request_id, session_id))
            .copied()
        else {
            bail!("native serving plugin proposal has no active generation state");
        };
        if actual_generated_token_count != expected_generated_token_count {
            bail!(
                "native serving plugin generation state is at {actual_generated_token_count} tokens, expected {expected_generated_token_count}"
            );
        }
        Ok(())
    }

    fn propose(&self, query: LinearProposalQuery) -> Result<Option<LinearProposal>> {
        let generated_token_count = query
            .committed_token_count
            .checked_sub(query.prompt_token_count)
            .context("native serving plugin proposal precedes its prompt boundary")?;
        self.ensure_generated_token_count(
            query.request_id,
            query.session_id,
            generated_token_count,
        )?;
        let event = abi::ProposalQuery {
            struct_size: size_of::<abi::ProposalQuery>(),
            request_id: query.request_id,
            session_id: query.session_id,
            prompt_token_count: u64::try_from(query.prompt_token_count)?,
            committed_token_count: u64::try_from(query.committed_token_count)?,
            decode_step: u64::try_from(query.decode_step)?,
            max_proposal_tokens: u64::try_from(query.max_proposal_tokens)?,
            absolute_deadline_ns: deadline_ns(query.deadline),
        };
        let mut operation = 0;
        let status = unsafe {
            (self.definition.api().start_proposal)(self.instance()?, &event, &raw mut operation)
        };
        self.call_status("start proposal", status)?;
        self.poll_until_deadline(operation, query.deadline, query.max_proposal_tokens)
    }

    fn poll_until_deadline(
        &self,
        operation: abi::ProposalOperation,
        deadline: Instant,
        max_proposal_tokens: usize,
    ) -> Result<Option<LinearProposal>> {
        let mut token_buffer = self
            .proposal_token_buffer
            .lock()
            .map_err(|_| anyhow!("native serving plugin proposal token buffer lock poisoned"))?;
        let token_capacity = max_proposal_tokens.min(token_buffer.len());
        self.poll_with_buffer(operation, deadline, &mut token_buffer[..token_capacity])
    }

    fn poll_with_buffer(
        &self,
        operation: abi::ProposalOperation,
        deadline: Instant,
        token_ids: &mut [i32],
    ) -> Result<Option<LinearProposal>> {
        let instance = self.instance()?;
        let mut decision_id = [0_u8; abi::MAX_DECISION_ID_BYTES];
        while Instant::now() < deadline {
            let mut output = abi::ProposalOutput {
                struct_size: size_of::<abi::ProposalOutput>(),
                decision_id: decision_id.as_mut_ptr(),
                decision_id_capacity: decision_id.len(),
                decision_id_length: 0,
                token_ids: token_ids.as_mut_ptr(),
                token_capacity: token_ids.len(),
                token_length: 0,
            };
            let status = unsafe {
                (self.definition.api().poll_proposal)(instance, operation, &raw mut output)
            };
            match status {
                abi::ProposalPollStatus::PENDING => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if !remaining.is_zero() {
                        thread::sleep(PROPOSAL_POLL_INTERVAL.min(remaining));
                    }
                }
                abi::ProposalPollStatus::ABSTAIN => return Ok(None),
                abi::ProposalPollStatus::FAILED => {
                    let error = self.definition.status_error(
                        instance,
                        "poll proposal",
                        abi::PluginStatus::INTERNAL_ERROR,
                    );
                    unsafe { (self.definition.api().cancel_proposal)(instance, operation) };
                    return Err(error);
                }
                abi::ProposalPollStatus::READY => {
                    let proposal = proposal_from_output(&decision_id, token_ids, &output);
                    if proposal.is_err() {
                        unsafe { (self.definition.api().cancel_proposal)(instance, operation) };
                    }
                    return proposal.map(Some);
                }
                unknown => {
                    unsafe { (self.definition.api().cancel_proposal)(instance, operation) };
                    bail!(
                        "native serving plugin returned unknown proposal poll status {}",
                        unknown.0
                    );
                }
            }
        }
        unsafe { (self.definition.api().cancel_proposal)(instance, operation) };
        Ok(None)
    }

    fn report(&self, receipt: &LinearProposalReceipt) -> Result<()> {
        self.ensure_generated_token_count(
            receipt.request_id,
            receipt.session_id,
            receipt.generated_token_count,
        )?;
        let event = abi::ProposalOutcome {
            struct_size: size_of::<abi::ProposalOutcome>(),
            decision_id: abi::ByteSlice::from_bytes(receipt.decision_id.as_bytes()),
            disposition: convert_disposition(receipt.disposition),
            proposal_token_count: u64::try_from(receipt.proposal_token_count)?,
            verification_rows: u64::try_from(receipt.verification_rows)?,
            accepted_proposal_tokens: u64::try_from(receipt.accepted_proposal_tokens)?,
            committed_tokens: abi::TokenSlice::from_tokens(&receipt.committed_tokens),
            verification_row_predictions: abi::TokenSlice::from_tokens(
                &receipt.verification_row_predictions,
            ),
            canonical_prediction_count: u64::try_from(receipt.canonical_prediction_count)?,
            has_correction_or_boundary_token: receipt.correction_or_boundary_token.is_some(),
            correction_or_boundary_token: receipt.correction_or_boundary_token.unwrap_or_default(),
            base_position: receipt.base_position,
            position_after_verification: receipt.position_after_verification,
            canonical_position: receipt.canonical_position,
            trimmed_rows: u64::try_from(receipt.trimmed_rows)?,
        };
        let status = unsafe { (self.definition.api().report_proposal)(self.instance()?, &event) };
        self.call_status("report proposal", status)
    }

    fn discard(&self, decision_id: &[u8], reason: LinearProposalDiscardReason) -> Result<()> {
        let event = abi::ProposalDiscard {
            struct_size: size_of::<abi::ProposalDiscard>(),
            decision_id: abi::ByteSlice::from_bytes(decision_id),
            reason: convert_discard_reason(reason),
        };
        let status = unsafe { (self.definition.api().discard_proposal)(self.instance()?, &event) };
        self.call_status("discard proposal", status)
    }

    fn shutdown(&mut self) -> Result<()> {
        let Some(instance) = self.instance.take() else {
            return Ok(());
        };
        let status = unsafe { (self.definition.api().shutdown)(instance.as_ptr()) };
        self.call_status("shutdown", status)
    }
}

impl Drop for ActivePlugin {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("native serving plugin shutdown failed: {error:#}");
        }
    }
}

struct NativeLifecycleIngress {
    driver: Arc<PluginDriver>,
}

impl GenerationLifecycleIngress for NativeLifecycleIngress {
    fn try_submit(&self, observation: GenerationLifecycleObservation) -> Result<()> {
        let command = match observation {
            GenerationLifecycleObservation::Started(start) => PluginCommand::Begin(start),
            GenerationLifecycleObservation::Committed(commit) => PluginCommand::Committed(commit),
            GenerationLifecycleObservation::Aborted(abort) => {
                return self.driver.enqueue_recovery(PluginCommand::Abort(abort));
            }
            GenerationLifecycleObservation::Completed(receipt) => PluginCommand::Finish(receipt),
            _ => return Ok(()),
        };
        self.driver.enqueue(command)
    }

    fn delivery_failures(&self) -> u64 {
        self.driver.lifecycle_delivery_failures()
    }

    fn receipt_unavailable(&self, unavailable: &GenerationAbort) {
        // Receipt construction can fail after generation has already
        // succeeded.  Use the recovery path so an earlier plugin callback
        // failure cannot strand the generation's active state.
        let _ = self
            .driver
            .enqueue_recovery(PluginCommand::Abort(*unavailable));
    }
}

struct NativeProposalIngress {
    driver: Arc<PluginDriver>,
}

impl LinearProposalIngress for NativeProposalIngress {
    fn propose(&self, query: LinearProposalQuery) -> Result<LinearProposalSourceResponse> {
        let response = self.driver.propose(query)?;
        // A failing plugin callback must not fail the generation, so a source
        // error degrades to an abstention for decode. The worker logs the
        // plugin's message and the `SourceError` outcome keeps the failure
        // distinguishable from a deliberate abstention in telemetry.
        let proposal = response.proposal.unwrap_or_default();
        Ok(LinearProposalSourceResponse::with_telemetry(
            proposal,
            response.telemetry,
        ))
    }

    fn report(&self, receipt: &LinearProposalReceipt) -> Result<()> {
        self.driver
            .enqueue_terminal(PluginCommand::ReportHandoff(receipt.clone()))
    }

    fn report_delivery_failures(&self) -> u64 {
        self.driver.report_delivery_failures()
    }

    fn discard(
        &self,
        decision_id: &OpaqueProposalDecisionId,
        reason: LinearProposalDiscardReason,
    ) -> Result<()> {
        self.driver.enqueue(PluginCommand::Discard(
            decision_id.as_bytes().to_vec(),
            reason,
        ))
    }
}

fn proposal_from_output(
    decision_id: &[u8; abi::MAX_DECISION_ID_BYTES],
    token_ids: &[i32],
    output: &abi::ProposalOutput,
) -> Result<LinearProposal> {
    if output.decision_id_length == 0 || output.decision_id_length > decision_id.len() {
        bail!(
            "native serving plugin returned invalid decision ID length {}",
            output.decision_id_length
        );
    }
    if output.token_length == 0 || output.token_length > token_ids.len() {
        bail!(
            "native serving plugin returned invalid proposal length {}",
            output.token_length
        );
    }
    let decision =
        OpaqueProposalDecisionId::new(decision_id[..output.decision_id_length].to_vec())?;
    Ok(LinearProposal::new(
        decision,
        token_ids[..output.token_length].to_vec(),
    ))
}

fn convert_termination(
    value: skippy_server::frontend::GenerationTermination,
) -> abi::GenerationTermination {
    match value {
        skippy_server::frontend::GenerationTermination::CallbackStop => {
            abi::GenerationTermination::CALLBACK_STOP
        }
        skippy_server::frontend::GenerationTermination::MaxTokens => {
            abi::GenerationTermination::MAX_TOKENS
        }
        skippy_server::frontend::GenerationTermination::Cancelled => {
            abi::GenerationTermination::CANCELLED
        }
        _ => abi::GenerationTermination::CANCELLED,
    }
}

fn convert_disposition(value: LinearProposalDisposition) -> abi::ProposalDisposition {
    match value {
        LinearProposalDisposition::FullAccept => abi::ProposalDisposition::FULL_ACCEPT,
        LinearProposalDisposition::FirstMismatch => abi::ProposalDisposition::FIRST_MISMATCH,
        LinearProposalDisposition::Stopped => abi::ProposalDisposition::STOPPED,
        _ => abi::ProposalDisposition::STOPPED,
    }
}

fn convert_discard_reason(value: LinearProposalDiscardReason) -> abi::ProposalDiscardReason {
    match value {
        LinearProposalDiscardReason::DeadlineExceeded => {
            abi::ProposalDiscardReason::DEADLINE_EXCEEDED
        }
        LinearProposalDiscardReason::InvalidTokenCount => {
            abi::ProposalDiscardReason::INVALID_TOKEN_COUNT
        }
        LinearProposalDiscardReason::InvalidTokenId => abi::ProposalDiscardReason::INVALID_TOKEN_ID,
        LinearProposalDiscardReason::PositionMismatch => {
            abi::ProposalDiscardReason::POSITION_MISMATCH
        }
        LinearProposalDiscardReason::ExecutionFailed => {
            abi::ProposalDiscardReason::EXECUTION_FAILED
        }
        _ => abi::ProposalDiscardReason::EXECUTION_FAILED,
    }
}

fn deadline_ns(deadline: Instant) -> u64 {
    let remaining = deadline.saturating_duration_since(Instant::now());
    unsafe { monotonic_now_ns(std::ptr::null_mut()) }
        .saturating_add(u64::try_from(remaining.as_nanos()).unwrap_or(u64::MAX))
}

unsafe extern "C" fn monotonic_now_ns(_context: *mut c_void) -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    let elapsed = ORIGIN.get_or_init(Instant::now).elapsed().as_nanos();
    u64::try_from(elapsed).unwrap_or(u64::MAX)
}

fn validate_absolute_path(label: &str, path: &Path) -> Result<()> {
    if !path.is_absolute() {
        bail!("{label} path must be absolute: {}", path.display());
    }
    Ok(())
}

fn path_slice(path: &Path) -> abi::ByteSlice {
    abi::ByteSlice::from_bytes(path.as_os_str().as_encoded_bytes())
}

unsafe fn read_utf8(slice: abi::ByteSlice, label: &str) -> Result<String> {
    if slice.pointer.is_null() && slice.length != 0 {
        bail!("native serving plugin {label} has a null pointer");
    }
    let bytes = if slice.length == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(slice.pointer, slice.length) }
    };
    std::str::from_utf8(bytes)
        .with_context(|| format!("native serving plugin {label} is not UTF-8"))
        .map(ToOwned::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::fake_table;

    #[test]
    fn output_validation_is_fail_closed() {
        let decision = [1_u8; abi::MAX_DECISION_ID_BYTES];
        let tokens = [7_i32; 8_192];
        let mut output = abi::ProposalOutput {
            struct_size: size_of::<abi::ProposalOutput>(),
            decision_id: std::ptr::null_mut(),
            decision_id_capacity: decision.len(),
            decision_id_length: 1,
            token_ids: std::ptr::null_mut(),
            token_capacity: tokens.len(),
            token_length: 1,
        };
        assert!(proposal_from_output(&decision, &tokens, &output).is_ok());
        output.decision_id_length = decision.len() + 1;
        assert!(proposal_from_output(&decision, &tokens, &output).is_err());
        output.decision_id_length = 1;
        output.token_length = 0;
        assert!(proposal_from_output(&decision, &tokens, &output).is_err());
    }

    #[test]
    fn absolute_deadline_uses_the_host_clock_epoch() {
        let before = unsafe { monotonic_now_ns(std::ptr::null_mut()) };
        let deadline = deadline_ns(Instant::now() + Duration::from_millis(5));
        let after = unsafe { monotonic_now_ns(std::ptr::null_mut()) };
        assert!(deadline >= before.saturating_add(1_000_000));
        assert!(deadline <= after.saturating_add(10_000_000));
    }

    #[test]
    fn table_validation_rejects_version_and_layout_mismatches() {
        let mut table = fake_table();
        assert_eq!(validate_table(&table).unwrap(), "test-serving-plugin");

        table.abi_version += 1;
        assert!(
            validate_table(&table)
                .unwrap_err()
                .to_string()
                .contains("incompatible")
        );
        table.abi_version = abi::NATIVE_SERVING_PLUGIN_ABI_V2;
        table.struct_size -= 1;
        assert!(
            validate_table(&table)
                .unwrap_err()
                .to_string()
                .contains("table size")
        );
    }

    #[test]
    fn proposal_rejects_missing_generation_state() {
        let (active, _) = test_support::fake_active_with_events(Duration::ZERO);
        let error = active
            .propose(LinearProposalQuery::new(
                1,
                2,
                1,
                1,
                0,
                8,
                Instant::now() + Duration::from_millis(100),
            ))
            .unwrap_err();
        assert!(error.to_string().contains("no active generation state"));
    }

    #[test]
    fn proposal_rejects_a_rewound_generation_position() {
        let (active, _) = test_support::fake_active_with_events(Duration::ZERO);
        active
            .begin(&GenerationStart {
                request_id: 1,
                session_id: 2,
                agent_session_id: None,
                prompt_token_ids: Arc::from([3]),
                frontend_request_id: None,
            })
            .unwrap();
        active
            .committed(&GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 2,
                token_ids: vec![4, 5].into_boxed_slice(),
            })
            .unwrap();

        let error = active
            .propose(LinearProposalQuery::new(
                1,
                2,
                1,
                2,
                1,
                8,
                Instant::now() + Duration::from_millis(100),
            ))
            .unwrap_err();
        assert!(error.to_string().contains("at 2 tokens, expected 1"));
    }

    #[test]
    fn proposal_lifecycle_sequence_does_not_duplicate_verified_tokens() {
        let (active, events) = test_support::fake_active_with_events(Duration::ZERO);
        active
            .begin(&GenerationStart {
                request_id: 1,
                session_id: 2,
                agent_session_id: None,
                prompt_token_ids: Arc::from([3]),
                frontend_request_id: None,
            })
            .unwrap();
        active
            .committed(&GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![4].into_boxed_slice(),
            })
            .unwrap();
        active
            .propose(LinearProposalQuery::new(
                1,
                2,
                1,
                2,
                1,
                8,
                Instant::now() + Duration::from_millis(100),
            ))
            .unwrap();
        active
            .committed(&GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 3,
                token_ids: vec![5, 6].into_boxed_slice(),
            })
            .unwrap();
        active
            .report(
                &LinearProposalReceipt::test_fixture_with_generated_token_count(
                    OpaqueProposalDecisionId::new(vec![7]).unwrap(),
                    3,
                ),
            )
            .unwrap();
        active
            .propose(LinearProposalQuery::new(
                1,
                2,
                1,
                4,
                3,
                8,
                Instant::now() + Duration::from_millis(100),
            ))
            .unwrap();

        assert_eq!(
            *events.lock().unwrap(),
            [
                "begin",
                "commit",
                "proposal",
                "commit",
                "report",
                "report_complete",
                "proposal"
            ]
        );
    }

    #[test]
    fn report_rejects_a_generated_token_count_mismatch() {
        let (active, _) = test_support::fake_active_with_events(Duration::ZERO);
        active
            .begin(&GenerationStart {
                request_id: 1,
                session_id: 2,
                agent_session_id: None,
                prompt_token_ids: Arc::from([3]),
                frontend_request_id: None,
            })
            .unwrap();
        active
            .committed(&GenerationCommit {
                request_id: 1,
                session_id: 2,
                generated_token_count: 1,
                token_ids: vec![4].into_boxed_slice(),
            })
            .unwrap();

        let error = active
            .report(
                &LinearProposalReceipt::test_fixture_with_generated_token_count(
                    OpaqueProposalDecisionId::new(vec![8]).unwrap(),
                    2,
                ),
            )
            .unwrap_err();
        assert!(error.to_string().contains("at 1 tokens, expected 2"));
    }
}
