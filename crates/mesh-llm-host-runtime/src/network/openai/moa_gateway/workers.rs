use super::pool::{assemble_worker_pool, compute_actor_candidates};
use crate::inference::election;
use crate::mesh;
use mesh_mixture_of_agents as moa;

/// MoA's opinionated default: workers do not think unless the caller
/// explicitly asks for it. Workers are short-budget internal slots, not
/// user-facing reasoning steps. The fast worker's 256-token budget is
/// far too small to fit `<think>…</think>` + answer, and the reducer
/// doesn't want reasoning prose as candidate input.
///
/// The caller can still explicitly enable thinking (e.g. for
/// experimentation) via any of the recognised knobs — see
/// [`extract_enable_thinking_override`]. When no preference is
/// expressed, MoA picks for them: off (always `Some(false)`).
pub(super) fn effective_enable_thinking_for_moa(body: &serde_json::Value) -> Option<bool> {
    // Always off. Not a default — a policy.
    //
    // Reasoning is actively harmful inside MoA fan-out, and recorded traces
    // from 9 open-weight models make the failure mode concrete
    // (`evals/moa-openrouter/`):
    //
    // * Workers run on a short budget (the fast worker gets 256 tokens).
    //   With thinking on, qwen3-32b spent 408 reasoning tokens against a
    //   384-token cap and returned `finish_reason=length` with
    //   `content: null` — 1620 characters of reasoning and no answer. The
    //   worker contributed nothing but still cost a full inference.
    // * Across a recorded corpus, 15/140 responses came back truncated at
    //   the limit, concentrated in exactly the reasoning-capable models.
    // * The reducer synthesizes from worker answers; reasoning prose is bad
    //   candidate input regardless of budget.
    //
    // The previous shape honoured a caller's `reasoning_effort` /
    // `enable_thinking` override. That escape hatch only let callers ask for
    // the broken configuration, so it is gone: MoA decides this, not the
    // caller. Callers who want a reasoning model's thinking output should
    // address that model directly instead of going through `model=mesh`.
    //
    // We still parse the caller's preference so an ignored override is
    // visible in logs rather than silently dropped.
    if extract_enable_thinking_override(body) == Some(true) {
        tracing::info!(
            "moa: caller asked for reasoning, ignoring — MoA workers always run with thinking off"
        );
    }
    Some(false)
}

/// Pull the caller's "disable / enable thinking" preference out of an
/// inbound chat-completion or responses JSON body. Mirrors the same
/// shapes that `openai_frontend::common::normalize_reasoning_template_options`
/// recognises so MoA users get the same surface as direct callers.
///
/// Recognised inputs (any one is enough):
/// * `reasoning_effort: "none"` (off) or any non-`"none"` value (on)
/// * `reasoning: { enabled: false }` (off) / `{ enabled: true }` (on)
/// * `reasoning: { effort: "none" }` / `{ max_tokens: 0 }` (off)
/// * Any of `THINKING_BOOLEAN_ALIASES` as a top-level field with bool
/// * `thinking_budget: 0` (off)
/// * `chat_template_kwargs.enable_thinking` (or any alias) as bool
///
/// Returns `None` when the caller hasn't expressed a preference. The
/// MoA-specific policy layer in [`effective_enable_thinking_for_moa`]
/// turns that `None` into `Some(false)` so MoA workers default off.
fn extract_enable_thinking_override(body: &serde_json::Value) -> Option<bool> {
    let obj = body.as_object()?;
    let mut result: Option<bool> = None;

    // reasoning: { enabled, effort, max_tokens }
    if let Some(r) = obj.get("reasoning").and_then(|v| v.as_object()) {
        if r.get("enabled") == Some(&serde_json::Value::Bool(false))
            || r.get("effort").and_then(|v| v.as_str()) == Some("none")
            || r.get("max_tokens").and_then(|v| v.as_u64()) == Some(0)
        {
            result = Some(false);
        } else if r.get("enabled") == Some(&serde_json::Value::Bool(true))
            || r.get("effort").is_some()
            || r.get("max_tokens").is_some()
        {
            result = Some(true);
        }
    }

    // reasoning_effort: "none" / "low" / etc.
    if let Some(effort) = obj.get("reasoning_effort").and_then(|v| v.as_str()) {
        result = Some(effort != "none");
    }

    // Top-level boolean aliases (enable_thinking, enable_reasoning, etc.).
    for alias in openai_frontend::common::THINKING_BOOLEAN_ALIASES {
        if let Some(b) = obj.get(*alias).and_then(|v| v.as_bool()) {
            result = Some(b);
        }
    }

    if obj.get("thinking_budget").and_then(|v| v.as_u64()) == Some(0) {
        result = Some(false);
    }

    // chat_template_kwargs.{enable_thinking, ...}
    if let Some(kwargs) = obj.get("chat_template_kwargs").and_then(|v| v.as_object()) {
        for alias in openai_frontend::common::THINKING_BOOLEAN_ALIASES {
            if let Some(b) = kwargs.get(*alias).and_then(|v| v.as_bool()) {
                result = Some(b);
            }
        }
    }

    result
}

/// Build the admitted worker configuration, including the single-worker case.
///
/// Committee admission and deterministic rescue policy are decided by the
/// caller after this function has assembled the eligible workers.
pub(super) async fn build_moa_candidate_config(
    node: &mesh::Node,
    targets: Option<&election::ModelTargets>,
    required_tokens: Option<u32>,
    affinity: &crate::network::affinity::AffinityRouter,
) -> moa::GatewayConfig {
    let http = reqwest::Client::new();
    let (backends, models) =
        assemble_worker_pool(node, targets, required_tokens, &http, Some(affinity)).await;

    // Actor priority for the asymmetric tool path: best tool-caller first.
    // The actor is the one model that actually emits the tool call, so it must
    // be the best available tool-caller — a judgement the engine crate cannot
    // make because it can't see gossiped capabilities.
    let actor_candidates = compute_actor_candidates(node, &models).await;

    // Public meshes are a pathological availability case, not a trust case:
    // unknown peers, wider latency spread, more churn. Wait less for perfect.
    let patience = patience_profile(node.public_mesh);

    tracing::info!(
        required_tokens = ?required_tokens,
        "MoA config: {} workers (admitted): {:?}",
        models.len(),
        models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
    );

    moa::GatewayConfig {
        backends,
        models,
        // Bumped from 15s → 60s. 15s was tight for big-context interactive
        // turns: a large model with a 10–20k-token prompt and tool schema
        // (typical for agent harnesses like OpenCode/Goose) can need 20–30s
        // just to produce a first tool-call. Workers were getting killed
        // mid-inference and MoA reported `kind=early-exit` with the small
        // worker, never the strong one. 60s gives the strong worker room
        // to land without making the no-progress wait painful.
        worker_timeout: std::time::Duration::from_secs(60),
        // Per-attempt cap; hedged_reducer_call hedges across candidates so the
        // end-to-end wait is roughly reducer_timeout + a couple of hedge delays.
        reducer_timeout: std::time::Duration::from_secs(60),
        // Start a second reducer candidate after 5s if the first hasn't replied
        // (or sooner on outright failure). Cheap on the happy path, big win on
        // the cold-KV / stale-peer tail.
        hedge_delay: std::time::Duration::from_secs(5),
        // Chat-only grace: after this long since dispatch, if at least
        // one qualifying Answer is in hand we ship the highest-confidence
        // one. Tool turns bypass this entirely (consensus continues to
        // arbitrate tool proposals).
        //
        // 3 seconds is empirically good across the public mesh today.
        // Long enough that slow-but-good workers (studio MiniMax
        // landing at ~1s, mini Qwen3.5 at ~700ms) finish before the
        // timer; short enough that chat doesn't sit on a multi-second
        // ceiling on every turn. Lab data: median mesh_chat dropped
        // from ~6s (old default) to ~2s with this value, no quality
        // regression measured on factual / arithmetic / short-creative
        // prompts.
        //
        // The previous 6s was conservative because the original grace
        // logic only armed on a sole answer — it had to wait for a
        // second non-matching answer to arrive before becoming useless.
        // With the relaxed eligibility added in this change, the timer
        // is the dominant chat path, so a tighter default is the right
        // default.
        //
        // Tightened further on a public mesh — see `patience_profile`.
        first_answer_grace: patience.first_answer_grace,
        // Tier-gate patience: how long small-tier-only answers/consensus
        // are held when a big-tier strong worker (e.g. MiniMax) is still
        // running. 20s covers the strong worker's typical first-token
        // latency on agent-sized prompts over the public mesh without
        // approaching worker_timeout (60s). Hard-bounded: at expiry all
        // decision rules revert to ungated behavior. Same-tier pools are
        // unaffected, so "many small models lift each other" keeps its
        // current latency profile.
        strong_patience: patience.strong_patience,
        // Defaults to leaving each model's thinking behavior alone.
        // `try_handle_moa` overrides this from the inbound request body
        // when the caller has expressed a preference
        // (`reasoning_effort: "none"`, `enable_thinking: false`, etc.).
        enable_thinking: None,
        // Actor priority for tool turns / synthesis: best tool-caller first.
        // Computed below from gossiped `tool_use`, model size, and peer health.
        actor_candidates,
        // Gate advisory references on actor strength: they help a weak actor
        // and cost a strong one (evals/moa-openrouter/RESULTS.md).
        reference_policy: moa::ReferencePolicy::Auto,
        refinement_policy: Default::default(),
    }
}

/// How long a turn waits for better answers before shipping what it has.
struct PatienceProfile {
    first_answer_grace: std::time::Duration,
    strong_patience: std::time::Duration,
}

/// Timing profile for the turn, tightened on a public mesh.
///
/// A public mesh is a pathological *availability* case (unknown peers, wider
/// latency spread, more churn), not a trust case. Both knobs here are
/// "how long do we hold a usable answer hoping for a better one" — exactly the
/// wait that hurts most when the tail is long. The hard bounds are unchanged;
/// only the optional waiting shrinks, so quality paths still run when peers are
/// prompt.
fn patience_profile(public_mesh: bool) -> PatienceProfile {
    if public_mesh {
        PatienceProfile {
            // Ship a good answer sooner rather than wait out a long tail.
            first_answer_grace: std::time::Duration::from_millis(1500),
            // Still give a strong peer a real chance, but don't hold a usable
            // small-tier answer for 20s against an unknown remote worker.
            strong_patience: std::time::Duration::from_secs(8),
        }
    } else {
        PatienceProfile {
            // Widened 3s -> 10s: at 3s a fast small worker landed inside the
            // window while a larger peer was still generating, so grace armed
            // and (before the finalize fix) shipped the small answer, skipping
            // synthesis on ~every turn. 10s lets a normal committee complete
            // and synthesize; grace still bounds a genuinely stuck tail. Paired
            // with grace-finalizes-on-tool-turns-only, so even if it does fire
            // on an answer turn it synthesizes what arrived rather than shipping
            // one worker. See `evals/moa-openrouter/RESULTS.md`.
            first_answer_grace: std::time::Duration::from_secs(10),
            strong_patience: std::time::Duration::from_secs(20),
        }
    }
}

/// Keeps a physical clone reserved while any turn/backend reference survives.
/// Self-fill slots are pinned: failure must not retry onto another slot's clone.
pub(super) struct ReservedModelBackend {
    pub(super) inner: std::sync::Arc<dyn moa::ModelBackend>,
    pub(super) _reservation: crate::network::reservations::RoutingReservation,
}

#[async_trait::async_trait]
impl moa::ModelBackend for ReservedModelBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        max_tokens: u32,
        timeout: std::time::Duration,
        sampling: moa::SamplingParams,
    ) -> Result<serde_json::Value, String> {
        self.inner
            .chat_completion(model, messages, tools, max_tokens, timeout, sampling)
            .await
    }
}

/// Backend that calls a local model directly on its skippy HTTP port.
pub(super) struct LocalModelBackend {
    pub(super) port: u16,
    pub(super) http: reqwest::Client,
}

#[async_trait::async_trait]
impl moa::ModelBackend for LocalModelBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        max_tokens: u32,
        timeout: std::time::Duration,
        sampling: moa::SamplingParams,
    ) -> Result<serde_json::Value, String> {
        let url = format!("http://127.0.0.1:{}/v1/chat/completions", self.port);
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
            "mesh_hooks": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }
        moa::apply_enable_thinking(&mut body, sampling.enable_thinking);
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| format!("local:{} failed: {e}", self.port))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "HTTP {status}: {}",
                moa::truncate_chars(&text, 200)
            ));
        }
        resp.json::<serde_json::Value>()
            .await
            .map_err(|e| format!("parse: {e}"))
    }
}

/// Maximum number of physical replicas one logical MoA worker may try.
///
/// One primary plus one standby closes the single-peer robustness gap without
/// multiplying fan-out or letting retries consume the whole turn deadline.
pub(super) const MAX_REMOTE_REPLICAS_PER_WORKER: usize = 2;

/// Backend that calls a remote model over the QUIC tunnel. Replica order is the
/// origin-stable order produced by `Node::hosts_for_model`.
pub(super) struct RemoteModelBackend {
    pub(super) node: mesh::Node,
    pub(super) peer_ids: Vec<iroh::EndpointId>,
}

#[async_trait::async_trait]
impl moa::ModelBackend for RemoteModelBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[serde_json::Value],
        tools: Option<&serde_json::Value>,
        max_tokens: u32,
        timeout: std::time::Duration,
        sampling: moa::SamplingParams,
    ) -> Result<serde_json::Value, String> {
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
            "mesh_hooks": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }
        moa::apply_enable_thinking(&mut body, sampling.enable_thinking);
        let body_bytes = serde_json::to_vec(&body).map_err(|e| format!("serialize: {e}"))?;
        let http_request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             \r\n",
            body_bytes.len()
        );
        let mut raw = http_request.into_bytes();
        raw.extend_from_slice(&body_bytes);

        let total_replicas = self.peer_ids.len();
        try_with_replica_failover(
            &self.peer_ids,
            timeout,
            |index, peer_id, attempt_timeout| {
                let node = self.node.clone();
                let raw = raw.clone();
                async move {
                    let result = call_remote_replica(&node, peer_id, &raw, attempt_timeout).await;
                    if let Err(error) = &result {
                        if index + 1 < total_replicas && is_retryable_replica_error(error) {
                            tracing::warn!(
                                model,
                                peer = %peer_id.fmt_short(),
                                attempt = index + 1,
                                replicas = total_replicas,
                                "MoA remote replica failed; trying standby"
                            );
                        } else {
                            tracing::warn!(
                                model,
                                peer = %peer_id.fmt_short(),
                                attempt = index + 1,
                                replicas = total_replicas,
                                "MoA remote replica failed"
                            );
                        }
                    }
                    result
                }
            },
        )
        .await
    }
}

async fn try_with_replica_failover<T, F, Fut>(
    replicas: &[iroh::EndpointId],
    timeout: std::time::Duration,
    mut attempt: F,
) -> Result<T, String>
where
    F: FnMut(usize, iroh::EndpointId, std::time::Duration) -> Fut,
    Fut: std::future::Future<Output = Result<T, String>>,
{
    let deadline = tokio::time::Instant::now() + timeout;
    let mut last_error = "no eligible remote replicas".to_string();
    for (index, peer_id) in replicas.iter().copied().enumerate() {
        let Some(attempt_timeout) = replica_attempt_timeout(deadline) else {
            break;
        };
        match attempt(index, peer_id, attempt_timeout).await {
            Ok(value) => return Ok(value),
            Err(error) => {
                let retryable = is_retryable_replica_error(&error);
                last_error = error;
                if !retryable {
                    break;
                }
            }
        }
    }
    Err(last_error)
}

/// Whether a remote failure is safe to retry on a same-model standby.
fn is_retryable_replica_error(error: &str) -> bool {
    if !error.starts_with("HTTP ") {
        return true;
    }
    error
        .split_whitespace()
        .nth(1)
        .and_then(|status| status.trim_end_matches(':').parse::<u16>().ok())
        .is_none_or(|status| status == 0 || matches!(status, 408 | 429 | 500 | 502 | 503 | 504))
}

/// Return the remaining shared worker deadline for each replica attempt.
///
/// A connected-but-silent primary retains the full timeout it had before
/// failover support. Explicit transport, protocol, or retryable HTTP failures
/// can still hand the unused remainder to a standby without extending the
/// worker deadline.
fn replica_attempt_timeout(deadline: tokio::time::Instant) -> Option<std::time::Duration> {
    deadline.checked_duration_since(tokio::time::Instant::now())
}

async fn call_remote_replica(
    node: &mesh::Node,
    peer_id: iroh::EndpointId,
    raw: &[u8],
    timeout: std::time::Duration,
) -> Result<serde_json::Value, String> {
    tokio::time::timeout(timeout, async {
        let (mut send, mut recv) = node
            .open_http_tunnel(peer_id)
            .await
            .map_err(|e| format!("tunnel: {e}"))?;
        send.write_all(raw)
            .await
            .map_err(|e| format!("send: {e}"))?;
        send.finish().map_err(|e| format!("finish: {e}"))?;
        let response = recv
            .read_to_end(4 * 1024 * 1024)
            .await
            .map_err(|e| format!("recv: {e}"))?;
        parse_quic_http_response(&response)
    })
    .await
    .map_err(|_| format!("remote replica timeout after {}ms", timeout.as_millis()))?
}

fn parse_quic_http_response(response: &[u8]) -> Result<serde_json::Value, String> {
    let s = String::from_utf8_lossy(response);
    let header_end = s
        .find("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let status_line = s[..header_end].lines().next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if status != 200 {
        return Err(format!("HTTP {status}: {}", moa::truncate_chars(&s, 200)));
    }
    let body = &s[header_end + 4..];
    serde_json::from_str(body).map_err(|e| format!("parse: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_enable_thinking_override ────────────────────────────────
    //
    // Mirrors the shapes that `openai_frontend::common::normalize_reasoning_template_options`
    // accepts, so MoA users get the same surface as direct callers. If we
    // forget a shape, the model never gets told to stop thinking and the
    // fast worker burns its budget inside `<think>`.

    #[test]
    fn extract_no_knobs_returns_none() {
        let body = serde_json::json!({"model": "mesh", "messages": []});
        assert_eq!(extract_enable_thinking_override(&body), None);
    }

    #[test]
    fn extract_reasoning_effort_none_disables() {
        let body = serde_json::json!({"reasoning_effort": "none"});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_reasoning_effort_low_enables() {
        let body = serde_json::json!({"reasoning_effort": "low"});
        assert_eq!(extract_enable_thinking_override(&body), Some(true));
    }

    #[test]
    fn extract_reasoning_enabled_false_disables() {
        let body = serde_json::json!({"reasoning": {"enabled": false}});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_reasoning_max_tokens_zero_disables() {
        let body = serde_json::json!({"reasoning": {"max_tokens": 0}});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_top_level_enable_thinking_false() {
        let body = serde_json::json!({"enable_thinking": false});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_top_level_enable_thinking_alias() {
        // `use_thinking` is one of THINKING_BOOLEAN_ALIASES.
        let body = serde_json::json!({"use_thinking": false});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_thinking_budget_zero_disables() {
        let body = serde_json::json!({"thinking_budget": 0});
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_chat_template_kwargs_passes_through() {
        let body = serde_json::json!({
            "chat_template_kwargs": {"enable_thinking": false}
        });
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    #[test]
    fn extract_latest_wins_when_multiple_set() {
        // chat_template_kwargs is read last and so wins. Whatever ordering
        // we choose, picking ONE consistently is the contract.
        let body = serde_json::json!({
            "reasoning_effort": "low",                                  // enable
            "chat_template_kwargs": {"enable_thinking": false},         // disable
        });
        assert_eq!(extract_enable_thinking_override(&body), Some(false));
    }

    // ── MoA opinionated default ────────────────────────────────────────────────────
    //
    // For `model: "mesh"`, MoA does NOT let reasoning models think on
    // worker slots. The fast worker has a 256-token budget that doesn't
    // fit `<think>...</think>` + answer, and the reducer doesn't want
    // reasoning prose as candidate input. Callers can explicitly turn
    // reasoning back on, but the default is off.

    #[test]
    fn effective_default_is_no_thinking_when_caller_silent() {
        // No knobs in the body → MoA's opinion applies.
        let body = serde_json::json!({"model": "mesh", "messages": []});
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(false));
    }

    #[test]
    fn effective_respects_explicit_disable_from_caller() {
        let body = serde_json::json!({
            "reasoning_effort": "none",
            "model": "mesh",
        });
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(false));
    }

    #[test]
    fn effective_ignores_caller_request_to_enable_thinking() {
        // There is deliberately no escape hatch. Thinking-on is a broken
        // configuration for MoA fan-out: recorded traces show reasoning
        // models spending their entire worker budget inside `<think>` and
        // returning `finish_reason=length` with null content, contributing
        // nothing while still costing a full inference.
        //
        // The override is parsed (and logged) but never honoured, so a
        // caller asking for reasoning gets a working turn instead of a pool
        // of empty workers. Reasoning output should be requested from a
        // model directly, not through `model=mesh`.
        for body in [
            serde_json::json!({"reasoning_effort": "low", "model": "mesh"}),
            serde_json::json!({"reasoning_effort": "high", "model": "mesh"}),
            serde_json::json!({"enable_thinking": true, "model": "mesh"}),
            serde_json::json!({"reasoning": {"enabled": true}, "model": "mesh"}),
            serde_json::json!({"chat_template_kwargs": {"enable_thinking": true}}),
        ] {
            assert_eq!(
                effective_enable_thinking_for_moa(&body),
                Some(false),
                "MoA must force thinking off regardless of caller knobs: {body}"
            );
        }
    }

    #[test]
    fn effective_default_for_tool_calling_request_still_no_thinking() {
        // Agentic / tool turns get the same opinionated default.
        // The grace-bypass / consensus path in MoA already runs
        // differently for tool turns, but thinking is independent of
        // that and should still be off unless the caller insists.
        let body = serde_json::json!({
            "model": "mesh",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "x"}}],
        });
        assert_eq!(effective_enable_thinking_for_moa(&body), Some(false));
    }

    #[tokio::test(start_paused = true)]
    async fn replica_failover_preserves_order_and_succeeds_on_standby() {
        let replicas = vec![
            iroh::SecretKey::generate().public(),
            iroh::SecretKey::generate().public(),
        ];
        let calls = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = calls.clone();

        let result = try_with_replica_failover(
            &replicas,
            std::time::Duration::from_secs(10),
            move |index, peer, _| {
                let seen = seen.clone();
                async move {
                    seen.lock().unwrap().push(peer);
                    if index == 0 {
                        Err("tunnel: primary unavailable".to_string())
                    } else {
                        Ok("standby answer")
                    }
                }
            },
        )
        .await;

        assert_eq!(result, Ok("standby answer"));
        assert_eq!(*calls.lock().unwrap(), replicas);
    }

    #[tokio::test(start_paused = true)]
    async fn replica_failover_stops_after_success() {
        let replicas = vec![
            iroh::SecretKey::generate().public(),
            iroh::SecretKey::generate().public(),
        ];
        let calls = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let seen = calls.clone();

        let result = try_with_replica_failover(
            &replicas,
            std::time::Duration::from_secs(10),
            move |_, _, _| {
                let seen = seen.clone();
                async move {
                    *seen.lock().unwrap() += 1;
                    Ok::<_, String>("primary answer")
                }
            },
        )
        .await;

        assert_eq!(result, Ok("primary answer"));
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn replica_failover_preserves_primary_timeout_and_shares_deadline() {
        let replicas = vec![
            iroh::SecretKey::generate().public(),
            iroh::SecretKey::generate().public(),
        ];
        let budgets = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = budgets.clone();

        let _ = try_with_replica_failover(
            &replicas,
            std::time::Duration::from_secs(10),
            move |index, _, budget| {
                let seen = seen.clone();
                async move {
                    seen.lock().unwrap().push(budget);
                    if index == 0 {
                        tokio::time::sleep(std::time::Duration::from_secs(6)).await;
                    }
                    Err::<(), _>("tunnel: failed".to_string())
                }
            },
        )
        .await;

        let budgets = budgets.lock().unwrap();
        assert_eq!(budgets.len(), 2);
        assert_eq!(budgets[0], std::time::Duration::from_secs(10));
        assert_eq!(budgets[1], std::time::Duration::from_secs(4));
    }

    #[test]
    fn replica_failover_retries_only_transient_failures() {
        for error in [
            "tunnel: closed",
            "send: reset",
            "recv: reset",
            "parse: eof",
            "remote replica timeout after 100ms",
            "HTTP 0: malformed status line",
            "HTTP malformed: bad framing",
            "HTTP 429: busy",
            "HTTP 503: unavailable",
        ] {
            assert!(is_retryable_replica_error(error), "must retry {error}");
        }
        for error in [
            "HTTP 400: bad request",
            "HTTP 401: unauthorized",
            "HTTP 404: missing",
        ] {
            assert!(!is_retryable_replica_error(error), "must not retry {error}");
        }
    }

    #[tokio::test(start_paused = true)]
    async fn replica_failover_stops_after_non_retryable_response() {
        let replicas = vec![
            iroh::SecretKey::generate().public(),
            iroh::SecretKey::generate().public(),
        ];
        let calls = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        let seen = calls.clone();

        let result = try_with_replica_failover(
            &replicas,
            std::time::Duration::from_secs(10),
            move |_, _, _| {
                let seen = seen.clone();
                async move {
                    *seen.lock().unwrap() += 1;
                    Err::<(), _>("HTTP 400: bad request".to_string())
                }
            },
        )
        .await;

        assert_eq!(result, Err("HTTP 400: bad request".to_string()));
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    /// Public meshes are a pathological availability case: unknown peers,
    /// wider latency spread, more churn. Both patience knobs are "hold a
    /// usable answer hoping for a better one", which is exactly the wait that
    /// hurts when the tail is long — so they shrink, and only they.
    #[test]
    fn public_mesh_waits_less_for_a_better_answer() {
        let public = patience_profile(true);
        let private = patience_profile(false);
        assert!(
            public.first_answer_grace < private.first_answer_grace,
            "public mesh must ship a good answer sooner"
        );
        assert!(
            public.strong_patience < private.strong_patience,
            "public mesh must not hold a usable answer as long for a slow strong peer"
        );
    }

    /// Shrinking patience must not disable the quality paths entirely — a
    /// prompt strong peer should still get a chance to land.
    #[test]
    fn public_mesh_still_gives_strong_peers_a_chance() {
        let public = patience_profile(true);
        assert!(!public.first_answer_grace.is_zero());
        assert!(public.strong_patience >= std::time::Duration::from_secs(5));
    }

    /// Private-mesh timings are the tuned defaults and must not drift silently.
    /// Grace is 10s (widened from 3s): at 3s a fast worker landed inside the
    /// window while a larger peer was still generating, so grace armed and the
    /// committee never synthesized — measured 80/80 early-exit at capable
    /// scale. See `evals/moa-openrouter/RESULTS.md`.
    #[test]
    fn private_mesh_keeps_the_tuned_defaults() {
        let private = patience_profile(false);
        assert_eq!(
            private.first_answer_grace,
            std::time::Duration::from_secs(10)
        );
        assert_eq!(private.strong_patience, std::time::Duration::from_secs(20));
    }
}
