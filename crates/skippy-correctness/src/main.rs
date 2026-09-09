mod cli;
mod glm_dsa_trace;
mod native_mtp_openai;
mod report;
mod runner;
mod support;

use anyhow::Result;
use clap::Parser;

use crate::{
    cli::{Cli, CommandKind},
    glm_dsa_trace::glm_dsa_stage0_trace,
    native_mtp_openai::native_mtp_openai_ab,
    runner::{
        chain, kv_page_growth, single_step, split_prefix_hit, split_scan, stage_fa_parity,
        state_handoff,
    },
};

fn prepare_model_download_directories() {
    let prepared = match model_hf::prepare_download_directories() {
        Ok(prepared) => prepared,
        Err(error) => {
            eprintln!(
                "⚠ Unable to prepare model download directories: {error:#}. \
                 Model downloads may fail; set MESH_LLM_DATA_DIR to a writable directory."
            );
            return;
        }
    };
    for fallback in &prepared.fallbacks {
        eprintln!("⚠ {fallback}");
    }
    // SAFETY: runs before any Tokio runtime, process is single-threaded.
    unsafe { prepared.apply_to_process_environment() };
}

fn main() -> Result<()> {
    prepare_model_download_directories();
    match Cli::parse().command {
        CommandKind::SingleStep(args) => single_step(args),
        CommandKind::Chain(args) => chain(args),
        CommandKind::SplitScan(args) => split_scan(args),
        CommandKind::StateHandoff(args) => state_handoff(args),
        CommandKind::SplitPrefixHit(args) => split_prefix_hit(args),
        CommandKind::NativeMtpOpenAiAb(args) => native_mtp_openai_ab(*args),
        CommandKind::GlmDsaStage0Trace(args) => glm_dsa_stage0_trace(*args),
        CommandKind::StageFaParity(args) => stage_fa_parity(args),
        CommandKind::KvPageGrowth(args) => kv_page_growth(args),
    }
}
