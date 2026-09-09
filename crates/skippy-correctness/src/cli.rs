use std::{net::SocketAddr, path::PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "skippy-correctness")]
#[command(about = "Validate staged llama execution against full-model execution")]
pub struct Cli {
    #[command(subcommand)]
    pub command: CommandKind,
}

#[derive(Subcommand)]
pub enum CommandKind {
    SingleStep(SingleStepArgs),
    Chain(ChainArgs),
    SplitScan(SplitScanArgs),
    StateHandoff(StateHandoffArgs),
    SplitPrefixHit(SplitPrefixHitArgs),
    NativeMtpOpenAiAb(Box<NativeMtpOpenAiAbArgs>),
    GlmDsaStage0Trace(Box<GlmDsaStage0TraceArgs>),
    StageFaParity(StageFaParityArgs),
    KvPageGrowth(KvPageGrowthArgs),
}

#[derive(Args)]
pub struct KvPageGrowthArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    /// Tokens prefilled before the first export, standing in for an agent's
    /// system prefix.
    #[arg(long, default_value_t = 2048)]
    pub base_tokens: usize,
    /// Tokens appended per turn.
    #[arg(long, default_value_t = 512)]
    pub turn_tokens: usize,
    /// Turns appended after the base prefix.
    #[arg(long, default_value_t = 4)]
    pub turns: usize,
    /// Segment size the L3 store cuts at.
    #[arg(long, default_value_t = 8 * 1024 * 1024)]
    pub segment_bytes: u64,
    #[arg(long)]
    pub json: Option<PathBuf>,
}

#[derive(Args, Clone)]
pub struct RuntimeArgs {
    #[arg(long, alias = "model-path")]
    pub model: PathBuf,
    #[arg(
        long,
        help = "Model coordinate for local model paths, for example org/repo:Q4_K_M. If omitted, Hugging Face cache paths are resolved from cache provenance."
    )]
    pub model_id: Option<String>,
    #[arg(long)]
    pub stage_model: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "runtime-slice")]
    pub stage_load_mode: StageLoadMode,
    #[arg(long, default_value_t = 30)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 128)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: i32,
    #[arg(long)]
    pub n_batch: Option<u32>,
    #[arg(long)]
    pub n_ubatch: Option<u32>,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long = "flash-attn", value_enum, default_value = "auto")]
    pub flash_attn: FlashAttentionArg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum StageLoadMode {
    RuntimeSlice,
    ArtifactSlice,
    LayerPackage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum FlashAttentionArg {
    Auto,
    Disabled,
    Enabled,
}

#[derive(Args, Clone)]
pub struct ServerArgs {
    #[arg(long, default_value = "target/debug/skippy-server")]
    pub stage_server_bin: PathBuf,
    #[arg(long)]
    pub child_logs: bool,
    #[arg(long, default_value_t = 60)]
    pub startup_timeout_secs: u64,
    #[arg(long, default_value_t = 4)]
    pub max_inflight: usize,
}

#[derive(Args, Clone, Copy)]
pub struct NativeMtpArgs {
    #[arg(
        long,
        help = "Fail the correctness run unless the final stage returns a native MTP draft sideband"
    )]
    pub require_native_mtp_draft: bool,
}

#[derive(Args)]
pub struct OutputArgs {
    #[arg(long)]
    pub report_out: Option<PathBuf>,
}

#[derive(Args)]
pub struct SingleStepArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub native_mtp: NativeMtpArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value_t = 15)]
    pub split_layer: u32,
    #[arg(long, default_value = "127.0.0.1:19021")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Args)]
pub struct ChainArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub native_mtp: NativeMtpArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value = "10,20")]
    pub splits: String,
    #[arg(long, default_value = "127.0.0.1:19031")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19032")]
    pub stage2_bind_addr: SocketAddr,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Args)]
pub struct SplitScanArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub native_mtp: NativeMtpArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value = "1..30")]
    pub splits: String,
    #[arg(long, default_value = "127.0.0.1:19041")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Args)]
pub struct StateHandoffArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value = "127.0.0.1:19061")]
    pub source_bind_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19062")]
    pub restore_bind_addr: SocketAddr,
    #[arg(long, default_value_t = 2048)]
    pub activation_width: i32,
    #[arg(long, default_value_t = 0)]
    pub state_layer_start: u32,
    #[arg(long)]
    pub state_layer_end: Option<u32>,
    #[arg(long)]
    pub state_stage_index: Option<u32>,
    #[arg(long, value_enum, default_value = "full-state")]
    pub state_payload_kind: StatePayloadKind,
    #[arg(long)]
    pub prefix_token_count: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub cache_hit_repeats: usize,
    #[arg(long)]
    pub runtime_lane_count: Option<u32>,
    #[arg(long)]
    pub borrow_resident_hits: bool,
    #[arg(long)]
    pub cache_decoded_result_hits: bool,
    #[arg(long)]
    pub skip_suffix_prefill_check: bool,
    #[arg(long)]
    pub synthetic_input_activation: bool,
    #[arg(long)]
    pub binary_control: bool,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Args)]
pub struct SplitPrefixHitArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(
        long,
        default_value_t = 24,
        help = "Layer split between stage 0 (embeddings..split) and stage 1 (split..layer_end)"
    )]
    pub split_layer: u32,
    #[arg(long, default_value = "127.0.0.1:19290")]
    pub openai_bind_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19291")]
    pub stage0_bind_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19292")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long, default_value_t = 2048)]
    pub activation_width: i32,
    #[arg(
        long,
        default_value = " The expedition continued beyond the ridge where the maps ended."
    )]
    pub prompt_extension: String,
    #[arg(long, default_value_t = 12)]
    pub max_tokens: u32,
    #[arg(long, default_value_t = 120)]
    pub request_timeout_secs: u64,
    #[arg(long)]
    pub case_root: Option<PathBuf>,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Args)]
pub struct NativeMtpOpenAiAbArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value_t = 24)]
    pub split_layer: u32,
    #[arg(long, default_value = "127.0.0.1:19170")]
    pub openai_bind_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19171")]
    pub stage0_bind_addr: SocketAddr,
    #[arg(long)]
    pub stage0_endpoint_addr: Option<SocketAddr>,
    #[arg(long, default_value = "127.0.0.1:19172")]
    pub stage1_bind_addr: SocketAddr,
    #[arg(long)]
    pub stage1_endpoint_addr: Option<SocketAddr>,
    #[arg(long)]
    pub stage0_model: Option<PathBuf>,
    #[arg(long)]
    pub stage1_model: Option<PathBuf>,
    #[arg(long)]
    pub case_root: Option<PathBuf>,
    #[arg(long)]
    pub external_stage1: bool,
    #[arg(long)]
    pub stage1_ssh_host: Option<String>,
    #[arg(long)]
    pub stage1_remote_stage_server_bin: Option<String>,
    #[arg(long, default_value = "/tmp/skippy-native-mtp-openai-ab")]
    pub stage1_remote_root: String,
    #[arg(long)]
    pub stage1_remote_workdir: Option<String>,
    #[arg(long, default_value_t = 10)]
    pub batched_port_offset: u16,
    #[arg(long, default_value_t = 2048)]
    pub activation_width: i32,
    #[arg(long, default_value_t = 12)]
    pub max_tokens: u32,
    #[arg(long, default_value_t = 60)]
    pub request_timeout_secs: u64,
    #[arg(long)]
    pub allow_missing_batched_events: bool,
    #[arg(long)]
    pub allow_mismatch: bool,
}

#[derive(Args)]
pub struct GlmDsaStage0TraceArgs {
    #[command(flatten)]
    pub runtime: RuntimeArgs,
    #[command(flatten)]
    pub server: ServerArgs,
    #[command(flatten)]
    pub output: OutputArgs,
    #[arg(long, default_value = "target/debug/skippy-prompt")]
    pub prompt_bin: PathBuf,
    #[arg(long, default_value = "127.0.0.1:19285")]
    pub stage0_bind_addr: SocketAddr,
    #[arg(long, default_value = "127.0.0.1:19286")]
    pub fake_downstream_bind_addr: SocketAddr,
    #[arg(long, default_value_t = 2)]
    pub stage_layer_end: u32,
    #[arg(long, default_value_t = 6144)]
    pub activation_width: i32,
    #[arg(long, default_value_t = 128)]
    pub prefill_chunk_size: u32,
    #[arg(long, default_value_t = 1)]
    pub max_new_tokens: u32,
    #[arg(long, default_value_t = 16)]
    pub trace_values: u32,
    #[arg(long, default_value_t = 256)]
    pub trace_nodes: u32,
    #[arg(long, default_value = "kqv_out-0,dsa_sparse_attn-0,indexer_score-1")]
    pub trace_filter: String,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    pub trace_stats_max_bytes: u32,
    #[arg(
        long,
        help = "Allow fused/direct traced tensor digest mismatches. By default, matching trace points must be byte-identical."
    )]
    pub allow_trace_mismatch: bool,
    #[arg(long, default_value_t = 2e-4)]
    pub activation_atol: f32,
    #[arg(long, default_value_t = 5e-4)]
    pub activation_relative_rmse_tolerance: f64,
    #[arg(long)]
    pub case_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum StatePayloadKind {
    ResidentKv,
    FullState,
    RecurrentOnly,
    KvRecurrent,
}

#[derive(Args)]
pub struct StageFaParityArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long, default_value = "unsloth/inkling-GGUF:UD-Q2_K_XL")]
    pub model_id: String,
    #[arg(long, default_value_t = 0)]
    pub layer_start: u32,
    #[arg(long)]
    pub layer_end: u32,
    #[arg(long, default_value_t = 2048)]
    pub ctx_size: u32,
    #[arg(long, default_value_t = 99)]
    pub n_gpu_layers: i32,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long, default_value_t = 5e-3)]
    pub max_abs: f32,
    #[arg(long)]
    pub enabled_output: Option<PathBuf>,
    #[arg(long)]
    pub disabled_output: Option<PathBuf>,
}
