use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_runtime::{
    FlashAttentionType, GenerationSignalWindow, MtpSource, RuntimeConfig, RuntimeLoadMode,
    SamplingConfig, StageModel, StageSession, TokenSignal, parse_cache_type,
};

use crate::cli::{FlashAttentionArg, MAX_VERIFY_WINDOW_WIDTH, VerifyWindowLocalArgs};

#[derive(Debug, Serialize)]
struct TimingStats {
    count: usize,
    total_us: u128,
    avg_us: f64,
    min_us: u128,
    p50_us: u128,
    p95_us: u128,
    max_us: u128,
}

#[derive(Debug, Serialize)]
struct TimingShape {
    first_half: TimingStats,
    second_half: TimingStats,
    second_half_avg_vs_first_half_avg: f64,
    first_sample_us: u128,
    last_sample_us: u128,
    samples_us: Vec<u128>,
}

#[derive(Debug, Serialize)]
struct VerifyWindowLocalReport {
    mode: &'static str,
    model_path: PathBuf,
    layer_end: u32,
    split_layer: Option<u32>,
    ctx_size: u32,
    n_gpu_layers: i32,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    cache_type_k: String,
    cache_type_v: String,
    flash_attn: &'static str,
    prompt_token_count: usize,
    verify_tokens: Vec<i32>,
    verify_widths: Vec<usize>,
    continuation_steps: usize,
    parity_checks: Vec<VerifyParityCheck>,
    warmup: usize,
    iterations: usize,
    sample_width: usize,
    batched: TimingStats,
    serial: TimingStats,
    split_inprocess: Option<SplitInprocessReport>,
    batched_avg_vs_serial_avg: f64,
    batched_token_per_sec: f64,
    serial_token_per_sec: f64,
    first_batched_prediction: Vec<i32>,
    first_serial_prediction: Vec<i32>,
}

#[derive(Debug, Serialize)]
struct VerifyParityCheck {
    width: usize,
    matched: bool,
    first_mismatch_position: Option<usize>,
    target_stream_matched: bool,
    continuation_matched: bool,
    continuation_steps: usize,
    native_position_matched: bool,
    full_state_matched: bool,
    serial_full_state_bytes: usize,
    batched_full_state_bytes: usize,
    token_signal_matched: bool,
    signal_window_matched: bool,
    serial_top_token: i32,
    serial_second_token: i32,
    serial_top2_margin: f32,
    batched_top_token: i32,
    batched_second_token: i32,
    batched_top2_margin: f32,
    batched_us: u128,
    serial_us: u128,
}

#[derive(Debug, Serialize)]
struct SplitInprocessReport {
    split_layer: u32,
    boundary_payload_bytes: usize,
    serial_boundary_payload_bytes: usize,
    total: TimingStats,
    stage0: TimingStats,
    stage1: TimingStats,
    serial_total: TimingStats,
    serial_stage0: TimingStats,
    serial_stage1: TimingStats,
    total_token_per_sec: f64,
    serial_total_token_per_sec: f64,
    total_avg_vs_full_batched_avg: f64,
    total_avg_vs_serial_total_avg: f64,
    diagnostics: SplitTimingDiagnostics,
    first_prediction: Vec<i32>,
    first_serial_prediction: Vec<i32>,
}

#[derive(Debug, Serialize)]
struct SplitTimingDiagnostics {
    batched_total: TimingShape,
    batched_stage0: TimingShape,
    batched_stage1: TimingShape,
    serial_total: TimingShape,
    serial_stage0: TimingShape,
    serial_stage1: TimingShape,
}

pub fn verify_window_local(args: VerifyWindowLocalArgs) -> Result<()> {
    validate_args(&args)?;
    let output = args.output.clone();
    let full = run_full_model_samples(&args)?;
    let split = match args.split_layer {
        Some(split_layer) => Some(run_split_inprocess_samples(
            &args,
            split_layer,
            &full.tokens,
            &full.verify_tokens,
            full.samples.batched_avg_us()?,
        )?),
        None => None,
    };
    let report = build_report(args, full, split)?;
    let parity_failure = report
        .parity_checks
        .iter()
        .find(|check| !check.matched)
        .map(format_parity_failure);
    let encoded = serde_json::to_vec_pretty(&report)?;

    if let Some(path) = output {
        fs::write(&path, &encoded)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(encoded)?);
    if let Some(failure) = parity_failure {
        bail!("{failure}");
    }
    Ok(())
}

fn validate_args(args: &VerifyWindowLocalArgs) -> Result<()> {
    if args.layer_end == 0 {
        bail!("layer_end must be greater than zero");
    }
    if args.iterations == 0 {
        bail!("iterations must be greater than zero");
    }
    validate_verify_widths(&args.verify_widths, args.sample_width)?;
    if args.continuation_steps == 0 {
        bail!("continuation_steps must be positive");
    }
    if let Some(split_layer) = args.split_layer
        && (split_layer == 0 || split_layer >= args.layer_end)
    {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    Ok(())
}

fn validate_verify_widths(verify_widths: &[usize], sample_width: usize) -> Result<()> {
    if verify_widths.is_empty() || verify_widths.contains(&0) {
        bail!("verify_widths must contain at least one positive width");
    }
    if verify_widths
        .iter()
        .any(|&width| width > MAX_VERIFY_WINDOW_WIDTH)
    {
        bail!(
            "verify_widths must not exceed the batch-invariant Metal ceiling of {MAX_VERIFY_WINDOW_WIDTH}"
        );
    }
    if verify_widths
        .iter()
        .enumerate()
        .any(|(index, width)| verify_widths[..index].contains(width))
    {
        bail!("verify_widths must not contain duplicates");
    }
    if sample_width == 0 || sample_width > MAX_VERIFY_WINDOW_WIDTH {
        bail!(
            "sample_width must be between 1 and the batch-invariant Metal ceiling of {MAX_VERIFY_WINDOW_WIDTH}"
        );
    }
    if !verify_widths.contains(&sample_width) {
        bail!(
            "sample_width must also appear in verify_widths so timed execution is parity-checked"
        );
    }
    Ok(())
}

fn full_runtime_config(args: &VerifyWindowLocalArgs) -> Result<RuntimeConfig> {
    Ok(RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        mmap: None,
        mlock: false,
        repack: false,
        op_offload: None,
        no_host_buffer: false,
        check_tensors: false,
        direct_io: false,
        main_gpu: None,
        split_mode: skippy_runtime::SplitMode::Auto,
        selected_backend_device: None,
        cache_type_k: parse_cache_type(&args.cache_type_k)?,
        cache_type_v: parse_cache_type(&args.cache_type_v)?,
        flash_attn_type: runtime_flash_attn(args.flash_attn),
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_runtime::GlmDsaPolicy::Auto,
        include_embeddings: true,
        include_output: true,
        mtp_source: MtpSource::Disabled,
        filter_tensors_on_load: false,
        resident_tensor_names: Vec::new(),
        checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    })
}

struct FullModelSamples {
    tokens: Vec<i32>,
    verify_tokens: Vec<i32>,
    parity_checks: Vec<VerifyParityCheck>,
    samples: SampleSet,
}

fn run_full_model_samples(args: &VerifyWindowLocalArgs) -> Result<FullModelSamples> {
    let config = full_runtime_config(args)?;
    let model = StageModel::open(&args.model_path, &config)
        .with_context(|| format!("failed to open {}", args.model_path.display()))?;
    let tokens = model
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    if tokens.is_empty() {
        bail!("prompt produced no tokens");
    }

    let mut session = model.create_session().context("failed to create session")?;
    session
        .prefill_chunked(&tokens)
        .context("failed to prefill prompt")?;
    let base_token_count = session.token_count();
    let target_plan = choose_target_plan(
        &mut session,
        base_token_count,
        &tokens,
        &args.prompt,
        &config,
        target_token_count(args)?,
    )?;
    let parity_checks = run_parity_checks(
        &mut session,
        base_token_count,
        args.layer_end,
        &target_plan,
        &args.verify_widths,
        args.continuation_steps,
    )?;
    let verify_tokens = target_plan.verify_tokens_for_width(sample_width(args))?;
    let samples = run_samples(
        &mut session,
        base_token_count,
        &verify_tokens,
        args.warmup,
        args.iterations,
    )?;
    Ok(FullModelSamples {
        tokens,
        verify_tokens,
        parity_checks,
        samples,
    })
}

struct VerifyTargetPlan {
    current: i32,
    targets: Vec<i32>,
}

enum ParityExecution {
    Serial,
    Batched,
}

struct ParitySide {
    prediction: Vec<i32>,
    continuation: Vec<i32>,
    position: u64,
    full_state: Vec<u8>,
    signal: TokenSignal,
    signal_window: GenerationSignalWindow,
    elapsed_us: u128,
}

struct ParityRun<'a> {
    base_token_count: u64,
    layer_end: i32,
    verify_tokens: &'a [i32],
    continuation_input: i32,
    width: usize,
    continuation_steps: usize,
}

impl VerifyTargetPlan {
    fn verify_tokens_for_width(&self, width: usize) -> Result<Vec<i32>> {
        if width == 0 {
            bail!("verify width must be greater than zero");
        }
        let required_targets = width.saturating_sub(1);
        if self.targets.len() < required_targets {
            bail!(
                "target stream has {} token(s), but width {width} requires {required_targets} target token(s)",
                self.targets.len(),
            );
        }
        let mut tokens = Vec::with_capacity(width);
        tokens.push(self.current);
        tokens.extend_from_slice(&self.targets[..width.saturating_sub(1)]);
        Ok(tokens)
    }

    fn expected_for_width(&self, width: usize) -> Result<&[i32]> {
        if self.targets.len() < width {
            bail!(
                "target stream has {} token(s), but width {width} requires {width}",
                self.targets.len()
            );
        }
        Ok(&self.targets[..width])
    }
}

fn choose_target_plan(
    session: &mut StageSession,
    base_token_count: u64,
    prompt_tokens: &[i32],
    prompt: &str,
    config: &RuntimeConfig,
    target_token_count: usize,
) -> Result<VerifyTargetPlan> {
    // Kernel parity needs an always-accepted target stream. Native MTP proposal
    // quality and acceptance remain covered by the dedicated MTP benchmarks.
    session
        .trim_session(base_token_count)
        .context("failed to trim session before choosing verify target stream")?;
    let current = *prompt_tokens
        .first()
        .context("prompt produced no token for verify-token seed")?;
    let mut input = current;
    let mut targets = Vec::with_capacity(target_token_count);
    for _ in 0..target_token_count {
        let (predicted, _native_mtp, _frame) = session
            .decode_step_frame_sampled_mtp(input, Some(&SamplingConfig::default()), None, 0, 1)
            .with_context(|| {
                format!(
                    "failed to extend greedy target stream from {} after prompt {:?}",
                    model_description(config),
                    prompt
                )
            })?;
        if predicted < 0 {
            bail!("target stream decode did not return a token");
        }
        targets.push(predicted);
        input = predicted;
    }
    session
        .trim_session(base_token_count)
        .context("failed to trim session after choosing verify target stream")?;
    Ok(VerifyTargetPlan { current, targets })
}

fn run_parity_checks(
    session: &mut StageSession,
    base_token_count: u64,
    layer_end: u32,
    target_plan: &VerifyTargetPlan,
    widths: &[usize],
    continuation_steps: usize,
) -> Result<Vec<VerifyParityCheck>> {
    let mut checks = Vec::with_capacity(widths.len());
    for &width in widths {
        checks.push(run_parity_check(
            session,
            base_token_count,
            layer_end,
            target_plan,
            width,
            continuation_steps,
        )?);
    }
    Ok(checks)
}

fn run_parity_check(
    session: &mut StageSession,
    base_token_count: u64,
    layer_end: u32,
    target_plan: &VerifyTargetPlan,
    width: usize,
    continuation_steps: usize,
) -> Result<VerifyParityCheck> {
    let verify_tokens = target_plan.verify_tokens_for_width(width)?;
    let expected = target_plan.expected_for_width(width)?;
    let continuation_expected = expected_continuation(target_plan, width, continuation_steps)?;
    let continuation_input = *target_plan
        .targets
        .get(width.saturating_sub(1))
        .with_context(|| format!("missing continuation input for width {width}"))?;
    let layer_end = i32::try_from(layer_end).context("layer_end exceeds i32")?;
    let run = ParityRun {
        base_token_count,
        layer_end,
        verify_tokens: &verify_tokens,
        continuation_input,
        width,
        continuation_steps,
    };
    let serial = collect_parity_side(session, &run, ParityExecution::Serial)?;
    let batched = collect_parity_side(session, &run, ParityExecution::Batched)?;
    build_parity_check(
        width,
        continuation_steps,
        expected,
        continuation_expected,
        &serial,
        &batched,
    )
}

fn collect_parity_side(
    session: &mut StageSession,
    run: &ParityRun<'_>,
    execution: ParityExecution,
) -> Result<ParitySide> {
    let label = match execution {
        ParityExecution::Serial => "serial",
        ParityExecution::Batched => "batched",
    };
    session
        .trim_session(run.base_token_count)
        .with_context(|| format!("failed to trim session before {label} parity check"))?;
    let start = Instant::now();
    let prediction = match execution {
        ParityExecution::Serial => serial_decode_expected(session, run.verify_tokens)?,
        ParityExecution::Batched => {
            session
                .verify_tokens_frame_sampled(
                    run.verify_tokens,
                    Some(&SamplingConfig::default()),
                    None,
                    0,
                    0,
                )
                .with_context(|| format!("batched width-{} VerifyWindow failed", run.width))?
                .0
        }
    };
    let elapsed_us = start.elapsed().as_micros();
    let position = session.token_count();
    let continuation = decode_continuation(
        session,
        run.continuation_input,
        run.width,
        run.continuation_steps,
    )?;
    let full_state = session
        .export_full_state(0, run.layer_end)
        .with_context(|| format!("failed to export {label} full state"))?;
    let signal = session.last_token_signal()?;
    let window_tokens =
        u32::try_from(run.width.saturating_add(1)).context("signal window width exceeds u32")?;
    let signal_window = session.signal_window(window_tokens)?;
    Ok(ParitySide {
        prediction,
        continuation,
        position,
        full_state,
        signal,
        signal_window,
        elapsed_us,
    })
}

fn build_parity_check(
    width: usize,
    continuation_steps: usize,
    expected: &[i32],
    continuation_expected: &[i32],
    serial: &ParitySide,
    batched: &ParitySide,
) -> Result<VerifyParityCheck> {
    let batched_prefix = prediction_prefix(&batched.prediction, width)?;
    let serial_prefix = prediction_prefix(&serial.prediction, width)?;
    let target_stream_matched = batched_prefix == expected && serial_prefix == expected;
    let continuation_matched = batched.continuation == continuation_expected
        && serial.continuation == continuation_expected;
    let native_position_matched = batched.position == serial.position;
    let full_state_matched = batched.full_state == serial.full_state;
    let token_signal_matched = batched.signal == serial.signal;
    let signal_window_matched = batched.signal_window == serial.signal_window;
    let first_mismatch_position = first_mismatch_position(batched_prefix, expected)
        .or_else(|| first_mismatch_position(serial_prefix, expected));
    // Signal windows summarize decode-call history, so one batched call and N serial
    // calls have different window shapes by construction. Keep this diagnostic
    // visible, but gate parity on canonical model state and token signals.
    let matched = target_stream_matched
        && continuation_matched
        && native_position_matched
        && full_state_matched
        && token_signal_matched;
    Ok(VerifyParityCheck {
        width,
        matched,
        first_mismatch_position,
        target_stream_matched,
        continuation_matched,
        continuation_steps,
        native_position_matched,
        full_state_matched,
        serial_full_state_bytes: serial.full_state.len(),
        batched_full_state_bytes: batched.full_state.len(),
        token_signal_matched,
        signal_window_matched,
        serial_top_token: serial.signal.top_token,
        serial_second_token: serial.signal.second_token,
        serial_top2_margin: serial.signal.margin,
        batched_top_token: batched.signal.top_token,
        batched_second_token: batched.signal.second_token,
        batched_top2_margin: batched.signal.margin,
        batched_us: batched.elapsed_us,
        serial_us: serial.elapsed_us,
    })
}

fn format_parity_failure(check: &VerifyParityCheck) -> String {
    format!(
        "verify parity failed at width {}: first_mismatch_position={:?} target_stream_matched={} continuation_matched={} native_position_matched={} full_state_matched={} serial_full_state_bytes={} batched_full_state_bytes={} token_signal_matched={} signal_window_matched={} serial_margin={} batched_margin={} serial_top={} batched_top={} batched_us={} serial_us={}",
        check.width,
        check.first_mismatch_position,
        check.target_stream_matched,
        check.continuation_matched,
        check.native_position_matched,
        check.full_state_matched,
        check.serial_full_state_bytes,
        check.batched_full_state_bytes,
        check.token_signal_matched,
        check.signal_window_matched,
        check.serial_top2_margin,
        check.batched_top2_margin,
        check.serial_top_token,
        check.batched_top_token,
        check.batched_us,
        check.serial_us
    )
}

fn serial_decode_expected(session: &mut StageSession, verify_tokens: &[i32]) -> Result<Vec<i32>> {
    let mut predicted_tokens = Vec::with_capacity(verify_tokens.len());
    for token_id in verify_tokens {
        let (predicted, _native_mtp, _frame) = session
            .decode_step_frame_sampled_mtp(*token_id, Some(&SamplingConfig::default()), None, 0, 1)
            .context("serial target-stream decode failed")?;
        if predicted < 0 {
            bail!("serial target-stream decode returned no token for input {token_id}");
        }
        predicted_tokens.push(predicted);
    }
    Ok(predicted_tokens)
}

fn expected_continuation(
    target_plan: &VerifyTargetPlan,
    width: usize,
    continuation_steps: usize,
) -> Result<&[i32]> {
    let end = width
        .checked_add(continuation_steps)
        .with_context(|| format!("continuation index overflow at width {width}"))?;
    target_plan.targets.get(width..end).with_context(|| {
        format!(
            "missing continuation targets for width {width}; target plan has {} token(s)",
            target_plan.targets.len()
        )
    })
}

fn decode_continuation(
    session: &mut StageSession,
    mut input: i32,
    width: usize,
    continuation_steps: usize,
) -> Result<Vec<i32>> {
    let mut predicted_tokens = Vec::with_capacity(continuation_steps);
    for step in 0..continuation_steps {
        let (predicted, _native_mtp, _frame) = session
            .decode_step_frame_sampled_mtp(input, Some(&SamplingConfig::default()), None, 0, 1)
            .with_context(|| format!("continuation decode failed at width {width}, step {step}"))?;
        if predicted < 0 {
            bail!("continuation decode returned no token at width {width}, step {step}");
        }
        predicted_tokens.push(predicted);
        input = predicted;
    }
    Ok(predicted_tokens)
}

fn prediction_prefix(prediction: &[i32], width: usize) -> Result<&[i32]> {
    if prediction.len() < width {
        bail!(
            "prediction has {} token(s), but width {width} requires {width}",
            prediction.len()
        );
    }
    Ok(&prediction[..width])
}

fn first_mismatch_position(left: &[i32], right: &[i32]) -> Option<usize> {
    left.iter()
        .zip(right.iter())
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

fn max_verify_width(args: &VerifyWindowLocalArgs) -> Result<usize> {
    args.verify_widths
        .iter()
        .copied()
        .max()
        .context("verify_widths must contain at least one width")
}

fn target_token_count(args: &VerifyWindowLocalArgs) -> Result<usize> {
    max_verify_width(args)?
        .max(args.sample_width)
        .checked_add(args.continuation_steps)
        .context("target token count overflow")
}

fn sample_width(args: &VerifyWindowLocalArgs) -> usize {
    args.sample_width
}

fn run_samples(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
    warmup: usize,
    iterations: usize,
) -> Result<SampleSet> {
    let total = warmup
        .checked_add(iterations)
        .context("sample count overflow")?;
    let mut batched = Vec::with_capacity(iterations);
    let mut serial = Vec::with_capacity(iterations);
    let mut first_batched_prediction = None;
    let mut first_serial_prediction = None;

    {
        let mut targets = SampleRecordTargets {
            batched: &mut batched,
            serial: &mut serial,
            first_batched_prediction: &mut first_batched_prediction,
            first_serial_prediction: &mut first_serial_prediction,
        };
        for index in 0..total {
            let record = index >= warmup;
            if index.is_multiple_of(2) {
                measure_batched_then_serial(
                    session,
                    base_token_count,
                    verify_tokens,
                    record,
                    &mut targets,
                )?;
            } else {
                measure_serial_then_batched(
                    session,
                    base_token_count,
                    verify_tokens,
                    record,
                    &mut targets,
                )?;
            }
        }
    }

    Ok(SampleSet {
        batched,
        serial,
        first_batched_prediction: first_batched_prediction.unwrap_or_default(),
        first_serial_prediction: first_serial_prediction.unwrap_or_default(),
    })
}

fn measure_batched_then_serial(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
    record: bool,
    targets: &mut SampleRecordTargets<'_>,
) -> Result<()> {
    let (batched_duration, batched_prediction) =
        measure_batched(session, base_token_count, verify_tokens)?;
    let (serial_duration, serial_prediction) =
        measure_serial(session, base_token_count, verify_tokens)?;
    record_sample(
        record,
        (batched_duration, batched_prediction),
        (serial_duration, serial_prediction),
        targets,
    );
    Ok(())
}

fn measure_serial_then_batched(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
    record: bool,
    targets: &mut SampleRecordTargets<'_>,
) -> Result<()> {
    let (serial_duration, serial_prediction) =
        measure_serial(session, base_token_count, verify_tokens)?;
    let (batched_duration, batched_prediction) =
        measure_batched(session, base_token_count, verify_tokens)?;
    record_sample(
        record,
        (batched_duration, batched_prediction),
        (serial_duration, serial_prediction),
        targets,
    );
    Ok(())
}

struct SampleRecordTargets<'a> {
    batched: &'a mut Vec<Duration>,
    serial: &'a mut Vec<Duration>,
    first_batched_prediction: &'a mut Option<Vec<i32>>,
    first_serial_prediction: &'a mut Option<Vec<i32>>,
}

fn record_sample(
    record: bool,
    batched_sample: (Duration, Vec<i32>),
    serial_sample: (Duration, Vec<i32>),
    targets: &mut SampleRecordTargets<'_>,
) {
    if !record {
        return;
    }
    targets.batched.push(batched_sample.0);
    targets.serial.push(serial_sample.0);
    targets
        .first_batched_prediction
        .get_or_insert(batched_sample.1);
    targets
        .first_serial_prediction
        .get_or_insert(serial_sample.1);
}

fn measure_batched(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
) -> Result<(Duration, Vec<i32>)> {
    session
        .trim_session(base_token_count)
        .context("failed to trim session before batched verify")?;
    let start = Instant::now();
    let prediction = session
        .verify_tokens_frame_sampled(verify_tokens, Some(&SamplingConfig::default()), None, 0, 0)
        .with_context(|| format!("batched width-{} VerifyWindow failed", verify_tokens.len()))?
        .0;
    Ok((start.elapsed(), prediction))
}

fn measure_serial(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
) -> Result<(Duration, Vec<i32>)> {
    session
        .trim_session(base_token_count)
        .context("failed to trim session before serial verify")?;
    let start = Instant::now();
    let prediction = serial_decode_mtp_n1(session, verify_tokens)?;
    Ok((start.elapsed(), prediction))
}

fn run_split_inprocess_samples(
    args: &VerifyWindowLocalArgs,
    split_layer: u32,
    tokens: &[i32],
    verify_tokens: &[i32],
    full_batched_avg_us: f64,
) -> Result<SplitInprocessReport> {
    let (stage0_config, stage1_config) = split_runtime_configs(args, split_layer)?;
    let stage0 = StageModel::open(&args.model_path, &stage0_config)
        .context("failed to open in-process split stage 0")?;
    let stage1 = StageModel::open(&args.model_path, &stage1_config)
        .context("failed to open in-process split stage 1")?;
    let mut session0 = stage0
        .create_session()
        .context("failed to create in-process split stage 0 session")?;
    let mut session1 = stage1
        .create_session()
        .context("failed to create in-process split stage 1 session")?;
    prefill_split_sessions(&mut session0, &mut session1, tokens)?;
    let base0 = session0.token_count();
    let base1 = session1.token_count();
    let samples = run_split_samples(
        &mut session0,
        &mut session1,
        base0,
        base1,
        verify_tokens,
        args.warmup,
        args.iterations,
    )?;
    split_report(
        split_layer,
        samples,
        full_batched_avg_us,
        verify_tokens.len(),
    )
}

fn split_runtime_configs(
    args: &VerifyWindowLocalArgs,
    split_layer: u32,
) -> Result<(RuntimeConfig, RuntimeConfig)> {
    let cache_type_k = parse_cache_type(&args.cache_type_k)?;
    let cache_type_v = parse_cache_type(&args.cache_type_v)?;
    let stage0 = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: split_layer,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        mmap: None,
        mlock: false,
        repack: false,
        op_offload: None,
        no_host_buffer: false,
        check_tensors: false,
        direct_io: false,
        main_gpu: None,
        split_mode: skippy_runtime::SplitMode::Auto,
        selected_backend_device: None,
        cache_type_k,
        cache_type_v,
        flash_attn_type: runtime_flash_attn(args.flash_attn),
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_runtime::GlmDsaPolicy::Auto,
        include_embeddings: true,
        include_output: false,
        mtp_source: MtpSource::Disabled,
        filter_tensors_on_load: true,
        resident_tensor_names: Vec::new(),
        checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };
    let stage1 = RuntimeConfig {
        stage_index: 1,
        layer_start: split_layer,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        mmap: None,
        mlock: false,
        repack: false,
        op_offload: None,
        no_host_buffer: false,
        check_tensors: false,
        direct_io: false,
        main_gpu: None,
        split_mode: skippy_runtime::SplitMode::Auto,
        selected_backend_device: None,
        cache_type_k,
        cache_type_v,
        flash_attn_type: runtime_flash_attn(args.flash_attn),
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        projector_use_gpu: None,
        media_marker: None,
        image_min_tokens: None,
        image_max_tokens: None,
        batch_max_tokens: None,
        glm_dsa_policy: skippy_runtime::GlmDsaPolicy::Auto,
        include_embeddings: false,
        include_output: true,
        mtp_source: MtpSource::Disabled,
        filter_tensors_on_load: true,
        resident_tensor_names: Vec::new(),
        checkpoint_quantization: skippy_runtime::CheckpointQuantization::Preserve,
        checkpoint_imatrix: None,
        checkpoint_imatrix_sha256: None,
        kv_offload: None,
        kv_unified: None,
        swa_full: None,
    };
    Ok((stage0, stage1))
}

fn prefill_split_sessions(
    session0: &mut StageSession,
    session1: &mut StageSession,
    tokens: &[i32],
) -> Result<()> {
    let (_stage0_prediction, boundary) = session0
        .prefill_chunk_frame_sampled(tokens, Some(&SamplingConfig::default()), None, 0)
        .context("in-process split stage 0 failed to prefill")?;
    if boundary.payload.is_empty() {
        bail!("in-process split stage 0 produced an empty prefill activation frame");
    }
    session1
        .prefill_chunk_frame_sampled(tokens, Some(&SamplingConfig::default()), Some(&boundary), 0)
        .context("in-process split stage 1 failed to prefill")?;
    Ok(())
}

fn run_split_samples(
    session0: &mut StageSession,
    session1: &mut StageSession,
    base0: u64,
    base1: u64,
    verify_tokens: &[i32],
    warmup: usize,
    iterations: usize,
) -> Result<SplitSampleSet> {
    let total = warmup
        .checked_add(iterations)
        .context("split sample count overflow")?;
    let mut total_samples = Vec::with_capacity(iterations);
    let mut stage0_samples = Vec::with_capacity(iterations);
    let mut stage1_samples = Vec::with_capacity(iterations);
    let mut serial_total_samples = Vec::with_capacity(iterations);
    let mut serial_stage0_samples = Vec::with_capacity(iterations);
    let mut serial_stage1_samples = Vec::with_capacity(iterations);
    let mut boundary_payload_bytes = 0usize;
    let mut serial_boundary_payload_bytes = 0usize;
    let mut first_prediction = None;
    let mut first_serial_prediction = None;

    for index in 0..total {
        let (batched, serial) = if index.is_multiple_of(2) {
            (
                measure_split_batched(session0, session1, base0, base1, verify_tokens)?,
                measure_split_serial(session0, session1, base0, base1, verify_tokens)?,
            )
        } else {
            let serial = measure_split_serial(session0, session1, base0, base1, verify_tokens)?;
            let batched = measure_split_batched(session0, session1, base0, base1, verify_tokens)?;
            (batched, serial)
        };
        if index >= warmup {
            total_samples.push(batched.total);
            stage0_samples.push(batched.stage0);
            stage1_samples.push(batched.stage1);
            serial_total_samples.push(serial.total);
            serial_stage0_samples.push(serial.stage0);
            serial_stage1_samples.push(serial.stage1);
            boundary_payload_bytes = batched.boundary_payload_bytes;
            serial_boundary_payload_bytes = serial.boundary_payload_bytes;
            first_prediction.get_or_insert(batched.prediction);
            first_serial_prediction.get_or_insert(serial.prediction);
        }
    }

    Ok(SplitSampleSet {
        total: total_samples,
        stage0: stage0_samples,
        stage1: stage1_samples,
        serial_total: serial_total_samples,
        serial_stage0: serial_stage0_samples,
        serial_stage1: serial_stage1_samples,
        boundary_payload_bytes,
        serial_boundary_payload_bytes,
        first_prediction: first_prediction.unwrap_or_default(),
        first_serial_prediction: first_serial_prediction.unwrap_or_default(),
    })
}

fn measure_split_batched(
    session0: &mut StageSession,
    session1: &mut StageSession,
    base0: u64,
    base1: u64,
    verify_tokens: &[i32],
) -> Result<SplitSample> {
    session0
        .trim_session(base0)
        .context("failed to trim split stage 0 before verify")?;
    session1
        .trim_session(base1)
        .context("failed to trim split stage 1 before verify")?;

    let total_start = Instant::now();
    let stage0_start = Instant::now();
    let (_stage0_prediction, _, boundary) = session0
        .verify_tokens_frame_sampled(verify_tokens, Some(&SamplingConfig::default()), None, 0, 0)
        .context("in-process split stage 0 VerifyWindow failed")?;
    let stage0 = stage0_start.elapsed();
    let boundary_payload_bytes = boundary.payload.len();
    if boundary_payload_bytes == 0 {
        bail!("in-process split stage 0 produced an empty VerifyWindow activation frame");
    }

    let stage1_start = Instant::now();
    let prediction = session1
        .verify_tokens_frame_sampled(
            verify_tokens,
            Some(&SamplingConfig::default()),
            Some(&boundary),
            0,
            0,
        )
        .context("in-process split stage 1 VerifyWindow failed")?
        .0;
    let stage1 = stage1_start.elapsed();
    Ok(SplitSample {
        total: total_start.elapsed(),
        stage0,
        stage1,
        boundary_payload_bytes,
        prediction,
    })
}

fn measure_split_serial(
    session0: &mut StageSession,
    session1: &mut StageSession,
    base0: u64,
    base1: u64,
    verify_tokens: &[i32],
) -> Result<SplitSample> {
    session0
        .trim_session(base0)
        .context("failed to trim split stage 0 before serial verify")?;
    session1
        .trim_session(base1)
        .context("failed to trim split stage 1 before serial verify")?;

    let total_start = Instant::now();
    let mut stage0_total = Duration::ZERO;
    let mut stage1_total = Duration::ZERO;
    let mut boundary_payload_bytes = 0usize;
    let mut prediction = Vec::with_capacity(verify_tokens.len() + 3);
    let mut last_draft = None;

    for token_id in verify_tokens {
        let stage0_start = Instant::now();
        let (_stage0_prediction, _stage0_draft, boundary) = session0
            .decode_step_frame_sampled_mtp(*token_id, Some(&SamplingConfig::default()), None, 0, 1)
            .context("in-process split stage 0 serial decode failed")?;
        stage0_total += stage0_start.elapsed();
        if boundary.payload.is_empty() {
            bail!("in-process split stage 0 produced an empty serial activation frame");
        }
        boundary_payload_bytes += boundary.payload.len();

        let stage1_start = Instant::now();
        let (predicted, native_mtp, _output) = session1
            .decode_step_frame_sampled_mtp(
                *token_id,
                Some(&SamplingConfig::default()),
                Some(&boundary),
                0,
                1,
            )
            .context("in-process split stage 1 serial decode failed")?;
        stage1_total += stage1_start.elapsed();
        if predicted >= 0 {
            prediction.push(predicted);
        }
        last_draft = native_mtp;
    }

    if let Some(draft) = last_draft {
        if let Some(token) = draft.token_ids.first().copied() {
            prediction.push(token);
        }
        prediction.push(i32::try_from(draft.proposal_compute_us.max(0)).unwrap_or(i32::MAX));
    }

    Ok(SplitSample {
        total: total_start.elapsed(),
        stage0: stage0_total,
        stage1: stage1_total,
        boundary_payload_bytes,
        prediction,
    })
}

fn serial_decode_mtp_n1(session: &mut StageSession, verify_tokens: &[i32]) -> Result<Vec<i32>> {
    let mut predicted_tokens = Vec::with_capacity(verify_tokens.len() + 3);
    let mut last_draft = None;
    for token_id in verify_tokens {
        let (predicted, native_mtp, _frame) = session
            .decode_step_frame_sampled_mtp(*token_id, Some(&SamplingConfig::default()), None, 0, 1)
            .context("serial native MTP n=1 decode failed")?;
        if predicted >= 0 {
            predicted_tokens.push(predicted);
        }
        last_draft = native_mtp;
    }
    if let Some(draft) = last_draft {
        if let Some(token) = draft.token_ids.first().copied() {
            predicted_tokens.push(token);
        }
        predicted_tokens.push(i32::try_from(draft.proposal_compute_us.max(0)).unwrap_or(i32::MAX));
    }
    Ok(predicted_tokens)
}

#[derive(Debug)]
struct SplitSample {
    total: Duration,
    stage0: Duration,
    stage1: Duration,
    boundary_payload_bytes: usize,
    prediction: Vec<i32>,
}

#[derive(Debug)]
struct SplitSampleSet {
    total: Vec<Duration>,
    stage0: Vec<Duration>,
    stage1: Vec<Duration>,
    serial_total: Vec<Duration>,
    serial_stage0: Vec<Duration>,
    serial_stage1: Vec<Duration>,
    boundary_payload_bytes: usize,
    serial_boundary_payload_bytes: usize,
    first_prediction: Vec<i32>,
    first_serial_prediction: Vec<i32>,
}

fn split_report(
    split_layer: u32,
    samples: SplitSampleSet,
    full_batched_avg_us: f64,
    sample_width: usize,
) -> Result<SplitInprocessReport> {
    let total = timing_stats(&samples.total)?;
    let serial_total = timing_stats(&samples.serial_total)?;
    let total_avg = total.avg_us;
    let serial_total_avg = serial_total.avg_us;
    Ok(SplitInprocessReport {
        split_layer,
        boundary_payload_bytes: samples.boundary_payload_bytes,
        serial_boundary_payload_bytes: samples.serial_boundary_payload_bytes,
        stage0: timing_stats(&samples.stage0)?,
        stage1: timing_stats(&samples.stage1)?,
        serial_stage0: timing_stats(&samples.serial_stage0)?,
        serial_stage1: timing_stats(&samples.serial_stage1)?,
        total,
        serial_total,
        total_token_per_sec: verified_tokens_per_sec(total_avg, sample_width),
        serial_total_token_per_sec: verified_tokens_per_sec(serial_total_avg, sample_width),
        total_avg_vs_full_batched_avg: total_avg / full_batched_avg_us,
        total_avg_vs_serial_total_avg: total_avg / serial_total_avg,
        diagnostics: split_timing_diagnostics(&samples)?,
        first_prediction: samples.first_prediction,
        first_serial_prediction: samples.first_serial_prediction,
    })
}

fn split_timing_diagnostics(samples: &SplitSampleSet) -> Result<SplitTimingDiagnostics> {
    Ok(SplitTimingDiagnostics {
        batched_total: timing_shape(&samples.total)?,
        batched_stage0: timing_shape(&samples.stage0)?,
        batched_stage1: timing_shape(&samples.stage1)?,
        serial_total: timing_shape(&samples.serial_total)?,
        serial_stage0: timing_shape(&samples.serial_stage0)?,
        serial_stage1: timing_shape(&samples.serial_stage1)?,
    })
}

#[derive(Debug)]
struct SampleSet {
    batched: Vec<Duration>,
    serial: Vec<Duration>,
    first_batched_prediction: Vec<i32>,
    first_serial_prediction: Vec<i32>,
}

impl SampleSet {
    fn batched_avg_us(&self) -> Result<f64> {
        Ok(timing_stats(&self.batched)?.avg_us)
    }
}

fn build_report(
    args: VerifyWindowLocalArgs,
    full: FullModelSamples,
    split_inprocess: Option<SplitInprocessReport>,
) -> Result<VerifyWindowLocalReport> {
    let sample_width = sample_width(&args);
    let batched = timing_stats(&full.samples.batched)?;
    let serial = timing_stats(&full.samples.serial)?;
    let batched_avg = batched.avg_us;
    let serial_avg = serial.avg_us;
    Ok(VerifyWindowLocalReport {
        mode: "verify-window-local",
        model_path: args.model_path,
        layer_end: args.layer_end,
        split_layer: args.split_layer,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        cache_type_k: args.cache_type_k,
        cache_type_v: args.cache_type_v,
        flash_attn: flash_attn_name(args.flash_attn),
        prompt_token_count: full.tokens.len(),
        verify_tokens: full.verify_tokens,
        verify_widths: args.verify_widths,
        continuation_steps: args.continuation_steps,
        parity_checks: full.parity_checks,
        warmup: args.warmup,
        iterations: args.iterations,
        sample_width,
        batched,
        serial,
        split_inprocess,
        batched_avg_vs_serial_avg: batched_avg / serial_avg,
        batched_token_per_sec: verified_tokens_per_sec(batched_avg, sample_width),
        serial_token_per_sec: verified_tokens_per_sec(serial_avg, sample_width),
        first_batched_prediction: full.samples.first_batched_prediction,
        first_serial_prediction: full.samples.first_serial_prediction,
    })
}

fn timing_stats(samples: &[Duration]) -> Result<TimingStats> {
    if samples.is_empty() {
        bail!("cannot summarize empty timing samples");
    }
    let mut micros = samples.iter().map(Duration::as_micros).collect::<Vec<_>>();
    micros.sort_unstable();
    let total_us = micros.iter().sum::<u128>();
    let avg_us = total_us as f64 / micros.len() as f64;
    Ok(TimingStats {
        count: micros.len(),
        total_us,
        avg_us,
        min_us: micros[0],
        p50_us: percentile(&micros, 0.50),
        p95_us: percentile(&micros, 0.95),
        max_us: *micros.last().context("missing max timing sample")?,
    })
}

fn timing_shape(samples: &[Duration]) -> Result<TimingShape> {
    if samples.is_empty() {
        bail!("cannot summarize empty timing shape");
    }
    let split_at = (samples.len() / 2).max(1);
    let (first_half, second_half) = samples.split_at(split_at);
    let second_half = if second_half.is_empty() {
        first_half
    } else {
        second_half
    };
    let first_half_stats = timing_stats(first_half)?;
    let second_half_stats = timing_stats(second_half)?;
    let samples_us = samples.iter().map(Duration::as_micros).collect::<Vec<_>>();
    Ok(TimingShape {
        second_half_avg_vs_first_half_avg: second_half_stats.avg_us / first_half_stats.avg_us,
        first_half: first_half_stats,
        second_half: second_half_stats,
        first_sample_us: samples_us[0],
        last_sample_us: *samples_us.last().context("missing timing shape sample")?,
        samples_us,
    })
}

fn percentile(sorted_micros: &[u128], percentile: f64) -> u128 {
    let last_index = sorted_micros.len().saturating_sub(1);
    let index = (last_index as f64 * percentile).round() as usize;
    sorted_micros[index.min(last_index)]
}

fn verified_tokens_per_sec(avg_us: f64, token_count: usize) -> f64 {
    token_count as f64 * 1_000_000.0 / avg_us
}

fn model_description(config: &RuntimeConfig) -> String {
    format!(
        "layers={}..{} ctx={} n_gpu_layers={}",
        config.layer_start, config.layer_end, config.ctx_size, config.n_gpu_layers
    )
}

fn runtime_flash_attn(value: FlashAttentionArg) -> FlashAttentionType {
    match value {
        FlashAttentionArg::Auto => FlashAttentionType::Auto,
        FlashAttentionArg::Disabled => FlashAttentionType::Disabled,
        FlashAttentionArg::Enabled => FlashAttentionType::Enabled,
    }
}

fn flash_attn_name(value: FlashAttentionArg) -> &'static str {
    match value {
        FlashAttentionArg::Auto => "auto",
        FlashAttentionArg::Disabled => "disabled",
        FlashAttentionArg::Enabled => "enabled",
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use skippy_runtime::{GenerationSignalWindow, TokenSignal};

    use super::{
        ParitySide, VerifyTargetPlan, build_parity_check, expected_continuation,
        first_mismatch_position, percentile, timing_shape, timing_stats, validate_verify_widths,
        verified_tokens_per_sec,
    };
    use crate::cli::MAX_VERIFY_WINDOW_WIDTH;

    #[test]
    fn timing_stats_sorts_and_summarizes_microseconds() {
        let stats = timing_stats(&[
            Duration::from_micros(30),
            Duration::from_micros(10),
            Duration::from_micros(20),
        ])
        .unwrap();

        assert_eq!(stats.count, 3);
        assert_eq!(stats.total_us, 60);
        assert_eq!(stats.min_us, 10);
        assert_eq!(stats.p50_us, 20);
        assert_eq!(stats.max_us, 30);
    }

    #[test]
    fn percentile_clamps_to_last_sample() {
        assert_eq!(percentile(&[10, 20, 30], 1.0), 30);
    }

    #[test]
    fn token_rate_uses_configured_verify_width() {
        assert_eq!(verified_tokens_per_sec(20_000.0, 4), 200.0);
    }

    #[test]
    fn verify_width_validation_rejects_duplicates_and_out_of_range_samples() {
        assert!(validate_verify_widths(&[1, 2, 4, 9, 16], 9).is_ok());
        assert!(validate_verify_widths(&[1, 2, 2], 2).is_err());
        assert!(validate_verify_widths(&[MAX_VERIFY_WINDOW_WIDTH + 1], 2).is_err());
        assert!(validate_verify_widths(&[2], MAX_VERIFY_WINDOW_WIDTH + 1).is_err());
        assert!(validate_verify_widths(&[1, 2, 4], 9).is_err());
    }

    #[test]
    fn timing_shape_reports_half_drift_in_sample_order() {
        let shape = timing_shape(&[
            Duration::from_micros(10),
            Duration::from_micros(20),
            Duration::from_micros(30),
            Duration::from_micros(50),
        ])
        .unwrap();

        assert_eq!(shape.first_sample_us, 10);
        assert_eq!(shape.last_sample_us, 50);
        assert_eq!(shape.samples_us, vec![10, 20, 30, 50]);
        assert_eq!(shape.first_half.avg_us, 15.0);
        assert_eq!(shape.second_half.avg_us, 40.0);
        assert_eq!(shape.second_half_avg_vs_first_half_avg, 40.0 / 15.0);
    }

    #[test]
    fn target_plan_builds_required_verification_widths() {
        let plan = VerifyTargetPlan {
            current: 10,
            targets: (20..=40).collect(),
        };

        for width in [1, 2, 4, 9] {
            let tokens = plan.verify_tokens_for_width(width).unwrap();
            let expected = plan.expected_for_width(width).unwrap();

            assert_eq!(tokens.len(), width);
            assert_eq!(expected.len(), width);
            assert_eq!(tokens[0], 10);
            assert_eq!(&tokens[1..], &expected[..width.saturating_sub(1)]);
        }
    }

    #[test]
    fn target_plan_selects_expected_continuation_after_width() {
        let plan = VerifyTargetPlan {
            current: 10,
            targets: (20..=40).collect(),
        };

        assert_eq!(expected_continuation(&plan, 4, 3).unwrap(), &[24, 25, 26]);
    }

    #[test]
    fn parity_aggregation_preserves_mismatch_details() {
        let serial = parity_side(vec![11, 12], vec![13], GenerationSignalWindow::default());
        let batched = parity_side(vec![11, 99], vec![13], GenerationSignalWindow::default());

        let check = build_parity_check(2, 1, &[11, 12], &[13], &serial, &batched).unwrap();

        assert!(!check.matched);
        assert_eq!(check.first_mismatch_position, Some(1));
        assert!(!check.target_stream_matched);
        assert!(check.continuation_matched);
    }

    #[test]
    fn signal_window_shape_is_advisory_for_batch_parity() {
        let serial = parity_side(
            vec![11, 12],
            vec![13],
            GenerationSignalWindow {
                token_count: 2,
                ..GenerationSignalWindow::default()
            },
        );
        let batched = parity_side(
            vec![11, 12],
            vec![13],
            GenerationSignalWindow {
                token_count: 1,
                ..GenerationSignalWindow::default()
            },
        );

        let check = build_parity_check(2, 1, &[11, 12], &[13], &serial, &batched).unwrap();

        assert!(check.matched);
        assert!(!check.signal_window_matched);
    }

    fn parity_side(
        prediction: Vec<i32>,
        continuation: Vec<i32>,
        signal_window: GenerationSignalWindow,
    ) -> ParitySide {
        ParitySide {
            prediction,
            continuation,
            position: 2,
            full_state: vec![1, 2, 3],
            signal: TokenSignal::default(),
            signal_window,
            elapsed_us: 1,
        }
    }

    #[test]
    fn mismatch_position_reports_value_and_length_differences() {
        assert_eq!(first_mismatch_position(&[1, 2, 3], &[1, 9, 3]), Some(1));
        assert_eq!(first_mismatch_position(&[1, 2], &[1, 2, 3]), Some(2));
        assert_eq!(first_mismatch_position(&[1, 2, 3], &[1, 2, 3]), None);
    }
}
