use std::env;
use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind, recv_ready, recv_reply,
    write_stage_message,
};

#[derive(Debug)]
struct Args {
    addr: String,
    requests: usize,
    concurrency: usize,
    activation_width: usize,
}

fn parse_args() -> Result<Args> {
    let mut addr = None;
    let mut requests = 64;
    let mut concurrency = 1;
    let mut activation_width = 6144;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .with_context(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--addr" => addr = Some(value),
            "--requests" => requests = value.parse().context("parse --requests")?,
            "--concurrency" => concurrency = value.parse().context("parse --concurrency")?,
            "--activation-width" => {
                activation_width = value.parse().context("parse --activation-width")?
            }
            _ => bail!("unknown argument {arg}"),
        }
    }
    Ok(Args {
        addr: addr.context("--addr is required")?,
        requests,
        concurrency: concurrency.max(1),
        activation_width,
    })
}

fn message(
    kind: WireMessageKind,
    request_id: u64,
    session_id: u64,
    pos_start: i32,
    token_ids: Vec<i32>,
    activation_width: usize,
) -> Result<StageWireMessage> {
    let mut state = StageStateHeader::new(kind);
    state.seq_id = 0;
    state.prompt_token_count = 1;
    state.decode_step = 0;
    state.current_token = token_ids.first().copied().unwrap_or(1);
    state.source_stage_index = 0;
    let token_count = i32::try_from(token_ids.len()).context("token count exceeds i32")?;
    let activation_width_i32 =
        i32::try_from(activation_width).context("activation width exceeds i32")?;
    let f32_payload = vec![
        0;
        token_ids
            .len()
            .checked_mul(activation_width)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .context("activation payload size overflow")?
    ];
    let activation = skippy_protocol::binary::encode_activation_payload_with_state_flags(
        state.activation_codec,
        token_count,
        activation_width_i32,
        &f32_payload,
        state.flags,
    )?;
    Ok(StageWireMessage {
        kind,
        pos_start,
        token_count,
        state,
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        activation,
        tokens: token_ids,
        positions: Vec::new(),
        raw_bytes: Vec::new(),
    })
}

fn run_request(addr: &str, index: usize, activation_width: usize) -> Result<serde_json::Value> {
    let started = Instant::now();
    let mut stream = TcpStream::connect(addr).with_context(|| format!("connect {addr}"))?;
    stream.set_nodelay(true).ok();
    stream
        .set_read_timeout(Some(Duration::from_secs(300)))
        .context("set read timeout")?;
    recv_ready(&mut stream).context("receive stage ready")?;
    let request_id = u64::try_from(index).unwrap_or(u64::MAX).saturating_add(1);
    let session_id = request_id;
    let decode = message(
        WireMessageKind::DecodeEmbd,
        request_id,
        session_id,
        0,
        vec![1],
        activation_width,
    )?;
    write_stage_message(&mut stream, &decode).context("write native-MTP decode")?;
    stream.flush().ok();
    let decode_reply = recv_reply(&mut stream).context("receive native-MTP decode")?;
    if decode_reply.kind != WireReplyKind::PredictedToken {
        bail!("decode returned {:?}", decode_reply.kind);
    }
    let draft = decode_reply
        .native_mtp_draft
        .as_ref()
        .and_then(|draft| draft.token_ids.first())
        .copied();
    let mut verified = None;
    let mut accepted = None;
    if let Some(draft_token) = draft {
        let mut verify = message(
            WireMessageKind::VerifyWindow,
            request_id,
            session_id,
            1,
            vec![decode_reply.predicted, draft_token],
            activation_width,
        )?;
        verify.state.seq_id = 1;
        write_stage_message(&mut stream, &verify).context("write native-MTP verify")?;
        stream.flush().ok();
        let verify_reply = recv_reply(&mut stream).context("receive native-MTP verify")?;
        if verify_reply.kind != WireReplyKind::PredictedTokens {
            bail!("verify returned {:?}", verify_reply.kind);
        }
        verified = verify_reply.predicted_tokens.first().copied();
        accepted = verified.map(|predicted| predicted == draft_token);
        let mut retire = message(
            WireMessageKind::RetireVerifyWindow,
            request_id,
            session_id,
            1,
            vec![0, 0],
            activation_width,
        )?;
        retire.tokens.clear();
        retire.activation.clear();
        retire.state.source_stage_index = -1;
        write_stage_message(&mut stream, &retire).context("retire native-MTP verify window")?;
        stream.flush().ok();
    }
    let stop = message(
        WireMessageKind::Stop,
        request_id,
        session_id,
        0,
        Vec::new(),
        activation_width,
    )?;
    write_stage_message(&mut stream, &stop).context("write stage stop")?;
    stream.flush().ok();
    let stop_reply = recv_reply(&mut stream).context("receive stage stop")?;
    if stop_reply.kind != WireReplyKind::Ack {
        bail!("stop returned {:?}", stop_reply.kind);
    }
    Ok(json!({
        "request_id": request_id,
        "elapsed_ms": started.elapsed().as_secs_f64() * 1_000.0,
        "predicted": decode_reply.predicted,
        "draft": draft,
        "verified": verified,
        "accepted": accepted,
    }))
}

fn main() -> Result<()> {
    let args = Arc::new(parse_args()?);
    let next = Arc::new(AtomicUsize::new(0));
    let results = Arc::new(Mutex::new(Vec::with_capacity(args.requests)));
    let makespan = Instant::now();
    let workers = (0..args.concurrency)
        .map(|_| {
            let args = Arc::clone(&args);
            let next = Arc::clone(&next);
            let results = Arc::clone(&results);
            thread::spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= args.requests {
                        break;
                    }
                    let result = run_request(&args.addr, index, args.activation_width)
                        .unwrap_or_else(
                            |error| json!({"request_id": index + 1, "error": format!("{error:#}")}),
                        );
                    results.lock().expect("result lock poisoned").push(result);
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().expect("benchmark worker panicked");
    }
    let makespan_ms = makespan.elapsed().as_secs_f64() * 1_000.0;
    let mut results = Arc::try_unwrap(results)
        .expect("result workers retained")
        .into_inner()
        .expect("result lock poisoned");
    results.sort_by_key(|row| row["request_id"].as_u64().unwrap_or(u64::MAX));
    let successful = results
        .iter()
        .filter(|row| row.get("error").is_none())
        .count();
    let drafted = results.iter().filter(|row| !row["draft"].is_null()).count();
    let accepted = results
        .iter()
        .filter(|row| row["accepted"].as_bool() == Some(true))
        .count();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "requests": args.requests,
            "concurrency": args.concurrency,
            "makespan_ms": makespan_ms,
            "throughput_rps": successful as f64 / (makespan_ms / 1_000.0),
            "successful": successful,
            "failed": args.requests.saturating_sub(successful),
            "drafted": drafted,
            "accepted": accepted,
            "acceptance_rate": if drafted == 0 { 0.0 } else { accepted as f64 / drafted as f64 },
            "per_request": results,
        }))?
    );
    Ok(())
}
