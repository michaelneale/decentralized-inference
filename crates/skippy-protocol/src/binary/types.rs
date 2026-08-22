use std::io;

use super::invalid_data;

// v14 expands the fixed StageReplyStats payload from 23 to 27 i64 fields. Stage peers must be
// upgraded together so older readers reject the changed wire contract before decoding it.
pub const STAGE_STATE_VERSION: i32 = 14;
pub const MAX_STAGE_LOGIT_BIAS: usize = 256;
pub const MAX_STAGE_SAMPLERS: usize = 16;
pub const MAX_STAGE_DRY_SEQUENCE_BREAKERS: usize = 8;
pub const MAX_STAGE_SAMPLING_STRING_BYTES: usize = 64;
pub const MAX_STAGE_PREDICTED_TOKENS: usize = 262_144;
pub const MAX_STAGE_SIDEBAND_VALUES: usize = 1_048_576;
pub const MAX_STAGE_CHAT_SAMPLING_METADATA_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_STAGE_STATE_IMPORT_BYTES: usize = 512 * 1024 * 1024;
pub const MAX_STAGE_ACTIVATION_BYTES: usize = 512 * 1024 * 1024;
pub const MAX_STAGE_DECODED_ACTIVATION_BYTES: usize = 512 * 1024 * 1024;
pub const READY_MAGIC: i32 = 0x5352_4459; // "SRDY"
pub const LLAMA_TOKEN_NULL: i32 = -1;
pub const STAGE_STATE_HEADER_BYTES: usize = 9 * 4;
pub const STAGE_SAMPLING_CONFIG_BASE_BYTES: usize = 27 * 4;
pub const STAGE_LOGIT_BIAS_WIRE_BYTES: usize = 4 + 4;
pub const STAGE_WIRE_FIXED_HEADER_BYTES: usize = 5 * 4 + STAGE_STATE_HEADER_BYTES + 2 * 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum WireMessageKind {
    PrefillEmbd = 1,
    DecodeEmbd = 2,
    Stop = 3,
    PrefillFinalEmbd = 4,
    DecodeReplayEmbd = 5,
    DecodeReplayFinalEmbd = 6,
    StateImport = 7,
    DecodeReadout = 8,
    DecodeLightCtx = 9,
    VerifyWindow = 21,
    RetireVerifyWindow = 22,
    DiscardStaleWindows = 23,
    StateExport = 13,
    ConfigureGeneration = 14,
    ProbePrefill = 15,
    RestorePrefill = 16,
    TryRestorePrefill = 17,
    TryRestorePrefillDecode = 18,
    TrimSession = 19,
    PredictionReturnOpen = 20,
}

impl WireMessageKind {
    pub fn is_prefill(self) -> bool {
        matches!(self, Self::PrefillEmbd | Self::PrefillFinalEmbd)
    }

    pub fn is_decode_replay(self) -> bool {
        matches!(self, Self::DecodeReplayEmbd | Self::DecodeReplayFinalEmbd)
    }

    pub fn is_decode_light_context(self) -> bool {
        matches!(self, Self::DecodeLightCtx)
    }

    pub fn requires_predicted_reply(self) -> bool {
        matches!(
            self,
            Self::DecodeEmbd
                | Self::DecodeReadout
                | Self::DecodeLightCtx
                | Self::VerifyWindow
                | Self::PrefillFinalEmbd
                | Self::DecodeReplayFinalEmbd
        )
    }

    pub fn is_session_control(self) -> bool {
        matches!(self, Self::TrimSession)
    }

    pub fn is_verify_retirement(self) -> bool {
        matches!(self, Self::RetireVerifyWindow)
    }

    /// Control message invalidating a contiguous range of not-yet-executed
    /// verify windows after the driver detected divergence. Recorded at
    /// message-receive time so buffered stale windows are skipped instead of
    /// executed.
    pub fn is_stale_window_discard(self) -> bool {
        matches!(self, Self::DiscardStaleWindows)
    }

    pub fn is_generation_control(self) -> bool {
        matches!(self, Self::ConfigureGeneration)
    }

    pub fn is_prefix_cache_control(self) -> bool {
        matches!(
            self,
            Self::ProbePrefill
                | Self::RestorePrefill
                | Self::TryRestorePrefill
                | Self::TryRestorePrefillDecode
        )
    }

    pub fn is_activationless_prefix_cache_control(self) -> bool {
        matches!(
            self,
            Self::ProbePrefill | Self::RestorePrefill | Self::TryRestorePrefill
        )
    }
}

impl TryFrom<i32> for WireMessageKind {
    type Error = io::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::PrefillEmbd),
            2 => Ok(Self::DecodeEmbd),
            3 => Ok(Self::Stop),
            4 => Ok(Self::PrefillFinalEmbd),
            5 => Ok(Self::DecodeReplayEmbd),
            6 => Ok(Self::DecodeReplayFinalEmbd),
            7 => Ok(Self::StateImport),
            8 => Ok(Self::DecodeReadout),
            9 => Ok(Self::DecodeLightCtx),
            13 => Ok(Self::StateExport),
            14 => Ok(Self::ConfigureGeneration),
            15 => Ok(Self::ProbePrefill),
            16 => Ok(Self::RestorePrefill),
            17 => Ok(Self::TryRestorePrefill),
            18 => Ok(Self::TryRestorePrefillDecode),
            19 => Ok(Self::TrimSession),
            20 => Ok(Self::PredictionReturnOpen),
            21 => Ok(Self::VerifyWindow),
            22 => Ok(Self::RetireVerifyWindow),
            23 => Ok(Self::DiscardStaleWindows),
            _ => Err(invalid_data("unknown stage message kind")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum WireReplyKind {
    Ack = 1,
    PredictedToken = 2,
    PredictedTokens = 3,
}

impl TryFrom<i32> for WireReplyKind {
    type Error = io::Error;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Ack),
            2 => Ok(Self::PredictedToken),
            3 => Ok(Self::PredictedTokens),
            _ => Err(invalid_data("unknown stage reply kind")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum WireStagePhase {
    Prefill = 1,
    Decode = 2,
    DecodeReplay = 3,
    DecodeLight = 4,
}

pub mod state_flags {
    pub const FINAL_CHUNK: i32 = 1 << 0;
    pub const LIGHT_CONTEXT: i32 = 1 << 1;
    pub const SAMPLING: i32 = 1 << 3;
    pub const FULL_STATE: i32 = 1 << 4;
    pub const CHAT_SAMPLING_METADATA: i32 = 1 << 5;
    pub const RWKV7_V_FIRST_SIDEBAND: i32 = 1 << 6;
    pub const GEMMA3N_ALTUP_SIDEBAND: i32 = 1 << 7;
    pub const INKLING_MTP_EMBD_SIDEBAND: i32 = 1 << 8;
}

pub const ACTIVATION_FLAG_RWKV7_V_FIRST: u64 = 1 << 0;
pub const ACTIVATION_FLAG_GEMMA3N_ALTUP: u64 = 1 << 1;
pub const ACTIVATION_FLAG_INKLING_MTP_EMBD: u64 = 1 << 2;

pub fn activation_frame_flags_from_state_flags(flags: i32) -> u64 {
    let mut frame_flags = 0;
    if (flags & state_flags::RWKV7_V_FIRST_SIDEBAND) != 0 {
        frame_flags |= ACTIVATION_FLAG_RWKV7_V_FIRST;
    }
    if (flags & state_flags::GEMMA3N_ALTUP_SIDEBAND) != 0 {
        frame_flags |= ACTIVATION_FLAG_GEMMA3N_ALTUP;
    }
    if (flags & state_flags::INKLING_MTP_EMBD_SIDEBAND) != 0 {
        frame_flags |= ACTIVATION_FLAG_INKLING_MTP_EMBD;
    }
    frame_flags
}

pub fn activation_state_flags_from_frame_flags(flags: u64) -> i32 {
    let mut state = 0;
    if (flags & ACTIVATION_FLAG_RWKV7_V_FIRST) != 0 {
        state |= state_flags::RWKV7_V_FIRST_SIDEBAND;
    }
    if (flags & ACTIVATION_FLAG_GEMMA3N_ALTUP) != 0 {
        state |= state_flags::GEMMA3N_ALTUP_SIDEBAND;
    }
    if (flags & ACTIVATION_FLAG_INKLING_MTP_EMBD) != 0 {
        state |= state_flags::INKLING_MTP_EMBD_SIDEBAND;
    }
    state
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageStateHeader {
    pub version: i32,
    pub seq_id: i32,
    pub phase: i32,
    pub flags: i32,
    pub checkpoint_generation: i32,
    pub prompt_token_count: i32,
    pub decode_step: i32,
    pub current_token: i32,
    pub source_stage_index: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StageLogitBias {
    pub token_id: i32,
    pub bias: f32,
}

impl Default for StageLogitBias {
    fn default() -> Self {
        Self {
            token_id: 0,
            bias: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageSamplingConfig {
    pub flags: u32,
    pub seed: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub penalty_last_n: i32,
    pub logit_bias: Vec<StageLogitBias>,
    pub typical_p: f32,
    pub top_nsigma: f32,
    pub dynatemp_range: f32,
    pub dynatemp_exponent: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    pub dry_sequence_breakers: Vec<String>,
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    pub mirostat_mode: i32,
    pub mirostat_entropy: f32,
    pub mirostat_learning_rate: f32,
    pub samplers: Vec<String>,
    pub ignore_eos: bool,
}

pub mod sampling_flags {
    pub const ENABLED: u32 = 1 << 0;
    pub const IGNORE_EOS: u32 = 1 << 1;
}

impl Default for StageSamplingConfig {
    fn default() -> Self {
        Self {
            flags: 0,
            seed: 0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_penalty: 1.0,
            penalty_last_n: -1,
            logit_bias: Vec::new(),
            typical_p: 1.0,
            top_nsigma: -1.0,
            dynatemp_range: 0.0,
            dynatemp_exponent: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 64,
            dry_sequence_breakers: vec!["\n".into(), ":".into(), "\"".into(), "*".into()],
            xtc_probability: 0.0,
            xtc_threshold: 0.1,
            mirostat_mode: 0,
            mirostat_entropy: 5.0,
            mirostat_learning_rate: 0.1,
            samplers: vec![
                "penalties".into(),
                "dry".into(),
                "top_n_sigma".into(),
                "top_k".into(),
                "typical_p".into(),
                "top_p".into(),
                "min_p".into(),
                "xtc".into(),
                "temperature".into(),
            ],
            ignore_eos: false,
        }
    }
}

impl StageSamplingConfig {
    pub fn enabled(&self) -> bool {
        (self.flags & sampling_flags::ENABLED) != 0
    }
}

impl StageStateHeader {
    pub fn new(kind: WireMessageKind) -> Self {
        let mut header = Self {
            version: STAGE_STATE_VERSION,
            seq_id: 0,
            phase: expected_phase(kind) as i32,
            flags: 0,
            checkpoint_generation: 0,
            prompt_token_count: 0,
            decode_step: -1,
            current_token: LLAMA_TOKEN_NULL,
            source_stage_index: -1,
        };
        if matches!(
            kind,
            WireMessageKind::PrefillFinalEmbd | WireMessageKind::DecodeReplayFinalEmbd
        ) {
            header.flags |= state_flags::FINAL_CHUNK;
        }
        if kind.is_decode_light_context() {
            header.flags |= state_flags::LIGHT_CONTEXT;
        }
        header
    }

    pub fn matches_kind(self, kind: WireMessageKind) -> bool {
        if matches!(
            kind,
            WireMessageKind::StateImport | WireMessageKind::StateExport
        ) || kind.is_session_control()
            || kind.is_verify_retirement()
            || kind.is_stale_window_discard()
            || kind.is_generation_control()
        {
            return true;
        }
        if self.phase != expected_phase(kind) as i32 {
            return false;
        }
        let expected_final = matches!(
            kind,
            WireMessageKind::PrefillFinalEmbd | WireMessageKind::DecodeReplayFinalEmbd
        );
        let actual_final = (self.flags & state_flags::FINAL_CHUNK) != 0;
        let expected_light = kind.is_decode_light_context();
        let actual_light = (self.flags & state_flags::LIGHT_CONTEXT) != 0;
        expected_final == actual_final && expected_light == actual_light
    }
}

impl Default for StageStateHeader {
    fn default() -> Self {
        Self::new(WireMessageKind::PrefillEmbd)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageRequestEpoch {
    pub request_id: u64,
    pub session_id: u64,
    pub checkpoint_generation: i32,
    pub prompt_token_count: i32,
    pub decode_step: i32,
}

impl StageRequestEpoch {
    pub fn same_flow(self, other: Self) -> bool {
        self.request_id == other.request_id && self.session_id == other.session_id
    }

    /// Returns true when this epoch is strictly older than `current` within the
    /// same request/session flow.
    ///
    /// Epochs from different flows are never comparable. For matching flows,
    /// staleness uses lexicographic ordering of checkpoint generation, prompt
    /// token count, and decode step, so a newer checkpoint dominates prompt and
    /// decode progress, and prompt progress dominates decode progress.
    pub fn is_stale_for(self, current: Self) -> bool {
        self.same_flow(current)
            && (
                self.checkpoint_generation,
                self.prompt_token_count,
                self.decode_step,
            ) < (
                current.checkpoint_generation,
                current.prompt_token_count,
                current.decode_step,
            )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StageWireMessage {
    pub kind: WireMessageKind,
    pub pos_start: i32,
    pub token_count: i32,
    pub state: StageStateHeader,
    pub request_id: u64,
    pub session_id: u64,
    pub sampling: Option<StageSamplingConfig>,
    pub chat_sampling_metadata: Option<String>,
    pub tokens: Vec<i32>,
    pub positions: Vec<i32>,
    pub activation: Vec<u8>,
    pub raw_bytes: Vec<u8>,
}

impl StageWireMessage {
    /// The committed session position that must exist before this message runs.
    ///
    /// Stage-state v11 makes this absolute position authoritative: a worker whose
    /// speculative KV is ahead must rewind locally before executing the message.
    pub fn authoritative_session_position(&self) -> Option<u64> {
        if !matches!(
            self.kind,
            WireMessageKind::DecodeEmbd
                | WireMessageKind::DecodeReadout
                | WireMessageKind::DecodeLightCtx
                | WireMessageKind::VerifyWindow
        ) {
            return None;
        }
        u64::try_from(self.pos_start).ok()
    }

    pub fn verify_window_id(&self) -> Option<i32> {
        (self.kind == WireMessageKind::VerifyWindow).then_some(self.state.seq_id)
    }

    pub fn verify_window_base_position(&self) -> Option<i32> {
        (self.kind == WireMessageKind::VerifyWindow).then_some(self.pos_start)
    }

    pub fn verify_window_token_count(&self) -> Option<i32> {
        (self.kind == WireMessageKind::VerifyWindow).then_some(self.token_count)
    }

    pub fn estimated_wire_bytes(&self) -> usize {
        let sampling_bytes = self.sampling.as_ref().map_or(0, |sampling| {
            STAGE_SAMPLING_CONFIG_BASE_BYTES
                + sampling.logit_bias.len().min(MAX_STAGE_LOGIT_BIAS) * STAGE_LOGIT_BIAS_WIRE_BYTES
                + sampling_string_list_wire_bytes(
                    &sampling.dry_sequence_breakers,
                    MAX_STAGE_DRY_SEQUENCE_BREAKERS,
                )
                + sampling_string_list_wire_bytes(&sampling.samplers, MAX_STAGE_SAMPLERS)
        });
        let chat_metadata_bytes = self
            .chat_sampling_metadata
            .as_ref()
            .map_or(0, |metadata| std::mem::size_of::<u32>() + metadata.len());
        STAGE_WIRE_FIXED_HEADER_BYTES
            .saturating_add(sampling_bytes)
            .saturating_add(chat_metadata_bytes)
            .saturating_add(self.payload_wire_bytes())
    }

    fn payload_wire_bytes(&self) -> usize {
        if self.kind == WireMessageKind::StateImport {
            return self.raw_bytes.len();
        }
        self.tokens
            .len()
            .saturating_mul(std::mem::size_of::<i32>())
            .saturating_add(
                self.positions
                    .len()
                    .saturating_mul(std::mem::size_of::<i32>()),
            )
            .saturating_add(self.activation.len())
    }

    pub fn request_epoch(&self) -> StageRequestEpoch {
        StageRequestEpoch {
            request_id: self.request_id,
            session_id: self.session_id,
            checkpoint_generation: self.state.checkpoint_generation,
            prompt_token_count: self.state.prompt_token_count,
            decode_step: self.state.decode_step,
        }
    }

    pub fn stop() -> Self {
        Self::stop_with_identity(0, 0)
    }

    pub fn stop_with_identity(request_id: u64, session_id: u64) -> Self {
        Self {
            kind: WireMessageKind::Stop,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(WireMessageKind::Stop),
            request_id,
            session_id,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    pub fn configure_generation(
        request_id: u64,
        session_id: u64,
        prompt_token_count: i32,
        sampling: Option<StageSamplingConfig>,
        chat_sampling_metadata: Option<String>,
    ) -> Self {
        let mut state = StageStateHeader::new(WireMessageKind::ConfigureGeneration);
        state.prompt_token_count = prompt_token_count;
        Self {
            kind: WireMessageKind::ConfigureGeneration,
            pos_start: 0,
            token_count: 0,
            state,
            request_id,
            session_id,
            sampling,
            chat_sampling_metadata,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    pub fn activation_f32_payload(&self) -> io::Result<Vec<u8>> {
        if self.activation.is_empty() {
            return Ok(Vec::new());
        }
        if self.activation.len() > MAX_STAGE_DECODED_ACTIVATION_BYTES {
            return Err(invalid_data(
                "decoded activation payload byte count exceeds maximum",
            ));
        }
        Ok(self.activation.clone())
    }

    pub fn take_activation_f32_payload(&mut self) -> io::Result<Vec<u8>> {
        if self.activation.is_empty() {
            return Ok(Vec::new());
        }
        if self.activation.len() > MAX_STAGE_DECODED_ACTIVATION_BYTES {
            return Err(invalid_data(
                "decoded activation payload byte count exceeds maximum",
            ));
        }
        Ok(std::mem::take(&mut self.activation))
    }
}

fn sampling_string_list_wire_bytes(values: &[String], maximum_count: usize) -> usize {
    values
        .iter()
        .take(maximum_count)
        .map(|value| std::mem::size_of::<u32>() + value.len())
        .sum()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageReply {
    pub kind: WireReplyKind,
    pub predicted: i32,
    /// Target-model predictions only. Native MTP proposals are carried separately.
    pub predicted_tokens: Vec<i32>,
    pub native_mtp_draft: Option<StageNativeMtpDraft>,
    pub window: StageReplyWindow,
    pub stats: StageReplyStats,
}

/// A native MTP proposal associated with a stage reply.
///
/// This deliberately has its own reply field rather than sharing the target
/// prediction vector. Consumers must never mistake proposal metadata for a
/// target prediction while verifying a composite speculative window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageNativeMtpDraft {
    pub token_ids: Vec<i32>,
    pub proposal_compute_us: i64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageReplyWindow {
    pub window_id: i32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageReplyStats {
    pub kv_lookup_hits: i64,
    pub kv_lookup_misses: i64,
    pub kv_lookup_errors: i64,
    pub kv_imported_pages: i64,
    pub kv_imported_tokens: i64,
    pub kv_recorded_pages: i64,
    pub kv_recorded_bytes: i64,
    pub kv_hit_stage_mask: i64,
    pub kv_record_stage_mask: i64,
    pub verify_window_compute_us: i64,
    pub verify_window_forward_write_us: i64,
    pub verify_window_downstream_wait_us: i64,
    pub verify_window_total_us: i64,
    pub verify_window_stage_count: i64,
    pub verify_window_request_count: i64,
    pub verify_window_token_count: i64,
    pub verify_window_max_tokens: i64,
    pub prefill_edge_write_us_max: i64,
    pub prefill_edge_wait_us_max: i64,
    pub prefill_edge_total_us_max: i64,
    pub prefill_edge_stage_index: i64,
    pub prefill_edge_activation_bytes_max: i64,
    pub prefill_edge_observation_count: i64,
    pub prefill_compute_us_at_slowest_rate: i64,
    pub prefill_compute_stage_index: i64,
    pub prefill_compute_token_count_at_slowest_rate: i64,
    pub prefill_compute_observation_count: i64,
}

impl StageReplyStats {
    pub fn merge(&mut self, other: Self) {
        self.kv_lookup_hits += other.kv_lookup_hits;
        self.kv_lookup_misses += other.kv_lookup_misses;
        self.kv_lookup_errors += other.kv_lookup_errors;
        self.kv_imported_pages += other.kv_imported_pages;
        self.kv_imported_tokens += other.kv_imported_tokens;
        self.kv_recorded_pages += other.kv_recorded_pages;
        self.kv_recorded_bytes += other.kv_recorded_bytes;
        self.kv_hit_stage_mask |= other.kv_hit_stage_mask;
        self.kv_record_stage_mask |= other.kv_record_stage_mask;
        self.verify_window_compute_us += other.verify_window_compute_us;
        self.verify_window_forward_write_us += other.verify_window_forward_write_us;
        self.verify_window_downstream_wait_us += other.verify_window_downstream_wait_us;
        self.verify_window_total_us += other.verify_window_total_us;
        self.verify_window_stage_count += other.verify_window_stage_count;
        self.verify_window_request_count += other.verify_window_request_count;
        self.verify_window_token_count += other.verify_window_token_count;
        self.verify_window_max_tokens = self
            .verify_window_max_tokens
            .max(other.verify_window_max_tokens);
        self.prefill_edge_write_us_max = self
            .prefill_edge_write_us_max
            .max(other.prefill_edge_write_us_max);
        self.prefill_edge_wait_us_max = self
            .prefill_edge_wait_us_max
            .max(other.prefill_edge_wait_us_max);
        if other.prefill_edge_total_us_max > self.prefill_edge_total_us_max {
            self.prefill_edge_total_us_max = other.prefill_edge_total_us_max;
            self.prefill_edge_stage_index = other.prefill_edge_stage_index;
            self.prefill_edge_activation_bytes_max = other.prefill_edge_activation_bytes_max;
        }
        self.prefill_edge_observation_count += other.prefill_edge_observation_count;
        if better_prefill_calibration_sample(
            other.prefill_compute_us_at_slowest_rate,
            other.prefill_compute_token_count_at_slowest_rate,
            self.prefill_compute_us_at_slowest_rate,
            self.prefill_compute_token_count_at_slowest_rate,
        ) {
            self.prefill_compute_us_at_slowest_rate = other.prefill_compute_us_at_slowest_rate;
            self.prefill_compute_stage_index = other.prefill_compute_stage_index;
            self.prefill_compute_token_count_at_slowest_rate =
                other.prefill_compute_token_count_at_slowest_rate;
        }
        self.prefill_compute_observation_count += other.prefill_compute_observation_count;
    }

    pub fn observe_prefill_edge_transport(
        &mut self,
        stage_index: u32,
        write_us: i64,
        wait_us: i64,
        activation_bytes: usize,
    ) {
        let write_us = write_us.max(0);
        let wait_us = wait_us.max(0);
        let total_us = write_us.saturating_add(wait_us);
        self.prefill_edge_write_us_max = self.prefill_edge_write_us_max.max(write_us);
        self.prefill_edge_wait_us_max = self.prefill_edge_wait_us_max.max(wait_us);
        if total_us > self.prefill_edge_total_us_max {
            self.prefill_edge_total_us_max = total_us;
            self.prefill_edge_stage_index = i64::from(stage_index);
            self.prefill_edge_activation_bytes_max =
                i64::try_from(activation_bytes).unwrap_or(i64::MAX);
        }
        self.prefill_edge_observation_count = self.prefill_edge_observation_count.saturating_add(1);
    }

    pub fn observe_prefill_compute(
        &mut self,
        stage_index: u32,
        compute_us: i64,
        token_count: usize,
    ) {
        let compute_us = compute_us.max(0);
        let token_count = i64::try_from(token_count).unwrap_or(i64::MAX).max(1);
        if better_prefill_calibration_sample(
            compute_us,
            token_count,
            self.prefill_compute_us_at_slowest_rate,
            self.prefill_compute_token_count_at_slowest_rate,
        ) {
            self.prefill_compute_us_at_slowest_rate = compute_us;
            self.prefill_compute_stage_index = i64::from(stage_index);
            self.prefill_compute_token_count_at_slowest_rate = token_count;
        }
        self.prefill_compute_observation_count =
            self.prefill_compute_observation_count.saturating_add(1);
    }

    pub fn is_empty(self) -> bool {
        self.kv_lookup_hits == 0
            && self.kv_lookup_misses == 0
            && self.kv_lookup_errors == 0
            && self.kv_imported_pages == 0
            && self.kv_imported_tokens == 0
            && self.kv_recorded_pages == 0
            && self.kv_recorded_bytes == 0
            && self.kv_hit_stage_mask == 0
            && self.kv_record_stage_mask == 0
            && self.verify_window_compute_us == 0
            && self.verify_window_forward_write_us == 0
            && self.verify_window_downstream_wait_us == 0
            && self.verify_window_total_us == 0
            && self.verify_window_stage_count == 0
            && self.verify_window_request_count == 0
            && self.verify_window_token_count == 0
            && self.verify_window_max_tokens == 0
            && self.prefill_edge_observation_count == 0
            && self.prefill_compute_observation_count == 0
    }
}

fn better_prefill_calibration_sample(
    candidate_us: i64,
    candidate_tokens: i64,
    current_us: i64,
    current_tokens: i64,
) -> bool {
    if candidate_us <= 0 || candidate_tokens <= 0 {
        return false;
    }
    if current_us <= 0 || current_tokens <= 0 {
        return true;
    }
    // Compare stages at the largest representative chunk size. A short final
    // remainder pays fixed launch overhead and can look much slower per token
    // than the full chunks that the calibration is intended to bound.
    if candidate_tokens != current_tokens {
        return candidate_tokens > current_tokens;
    }
    i128::from(candidate_us) * i128::from(current_tokens)
        > i128::from(current_us) * i128::from(candidate_tokens)
}

fn expected_phase(kind: WireMessageKind) -> WireStagePhase {
    if kind.is_prefill()
        || matches!(
            kind,
            WireMessageKind::StateImport | WireMessageKind::StateExport
        )
        || kind.is_session_control()
        || kind.is_verify_retirement()
        || kind.is_generation_control()
        || kind.is_prefix_cache_control()
    {
        WireStagePhase::Prefill
    } else if kind.is_decode_replay() {
        WireStagePhase::DecodeReplay
    } else if kind.is_decode_light_context() {
        WireStagePhase::DecodeLight
    } else {
        WireStagePhase::Decode
    }
}
