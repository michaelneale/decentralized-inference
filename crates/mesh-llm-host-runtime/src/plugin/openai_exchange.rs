//! Bridges the two real OpenAI-exchange dispatch paths (see
//! `docs/plugins/openai-exchange-lifecycle-design-note.md`, #1331 M1/M2) to
//! an out-of-process plugin over the existing `PluginMeshEvent::Channel`
//! transport, so a plugin sees one unified stream regardless of which
//! in-process Rust hook interface produced an event.

use std::sync::Arc;

use async_trait::async_trait;
use openai_frontend::{
    CapsuleMarker, ChatCompletionOutcome, ChatCompletionRequest, ChatCompletionResponse,
    ChatExchangeRoute, OpenAiHookPolicy,
};
use serde::Serialize;

use super::PluginManager;

/// The single mesh channel both dispatch paths publish to.
pub const OPENAI_EXCHANGE_CHANNEL: &str = "openai.exchange.v1";

/// Which real dispatch path produced an [`OpenAiExchangeEnvelope`] — the two
/// paths M1 found are disjoint and don't share a request type, so the
/// envelope carries this instead of assuming one shape fits both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiExchangeDispatchPath {
    /// `openai-frontend`'s typed `OpenAiHookPolicy`/`HookedOpenAiBackend` seam.
    TypedFrontend,
    /// The raw-proxy ingress (`network/openai/ingress.rs`), used for
    /// plugin-served models; never sees a typed `ChatCompletionRequest`.
    RawProxy,
    /// The raw-proxy ingress routes this exchange to a peer on the mesh
    /// rather than serving it locally (`route_missing_local_model`'s
    /// remote-mesh branch). This node is the requester/router, not the
    /// server, for the exchange this envelope describes — a downstream
    /// plugin must not treat it as the served-side event.
    RemoteMesh,
}

/// Which moment in an exchange's lifecycle an [`OpenAiExchangeEnvelope`]
/// reports — the same two moments [`OpenAiHookPolicy::on_effective_chat_completion`]
/// and [`OpenAiHookPolicy::on_chat_completion_terminal`] already observe for
/// path 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiExchangePhase {
    EffectiveRequest,
    Terminal,
}

/// Which side actually contributed a terminal event's `nonce` — the
/// authoritative signal for the same tri-state `capsule-emit-mesh`'s own
/// sidecar tracks as `client_nonce_source` (`client_supplied` /
/// `sidecar_generated_fallback`; the implicit third state is "no marker was
/// minted at all," carried by `nonce_source` itself being `None`). A
/// downstream plugin (M3) must use this field rather than sniffing the
/// `nonce`'s `fallback-` prefix, which stays only for human-readability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientNonceSource {
    ClientSupplied,
    SidecarGeneratedFallback,
}

/// The wire shape both dispatch paths publish on [`OPENAI_EXCHANGE_CHANNEL`].
/// Deliberately independent of `openai_frontend`'s typed request/response —
/// the raw-proxy path never has one — so one shape covers both paths without
/// either being forced into the other's type.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpenAiExchangeEnvelope {
    /// Stable per-exchange id, minted when the dispatch path admits the
    /// request. Shared by an exchange's `EffectiveRequest` and `Terminal`
    /// envelopes (and mirrored into the transport `correlation_id`), so a
    /// plugin can pair the two events for one exchange even when concurrent
    /// requests on the same model are in flight.
    pub exchange_id: String,
    pub dispatch_path: OpenAiExchangeDispatchPath,
    pub phase: OpenAiExchangePhase,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// Present only on a `Terminal` envelope carrying a rung-ladder response
    /// marker (see [`CapsuleMarker`]) — the `capsule_id` already written into
    /// the client's response as `X-Capsule-Id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capsule_id: Option<String>,
    /// The nonce the marker is correlated against, so a plugin observing
    /// this event knows what a later client ack must sign over.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
    /// Which side contributed `nonce` — see [`ClientNonceSource`]. `None`
    /// exactly when `nonce` is `None` (no marker minted).
    ///
    /// **Asymmetry across routing-node pairs:** when node A minted the
    /// fallback nonce (the client sent none), A reads its own
    /// `x-capsule-nonce-origin` header and reports
    /// `SidecarGeneratedFallback`. Node B strips that header deliberately
    /// (anti-smuggling, `request_parse.rs:582`) so it sees a
    /// well-formed nonce with no origin marker and reports `ClientSupplied`
    /// for the same nonce. Both are locally correct: A reports what it
    /// minted; B cannot trust the origin claim. A consumer joining both
    /// halves on the same nonce will observe two different `nonce_source`
    /// values — this is NOT a bug. Use the routing node's own envelope to
    /// judge whether the nonce was client-supplied or sidecar-generated;
    /// do not compare across nodes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce_source: Option<ClientNonceSource>,
}

impl OpenAiExchangeEnvelope {
    pub fn effective(
        exchange_id: impl Into<String>,
        dispatch_path: OpenAiExchangeDispatchPath,
        model: impl Into<String>,
    ) -> Self {
        Self {
            exchange_id: exchange_id.into(),
            dispatch_path,
            phase: OpenAiExchangePhase::EffectiveRequest,
            model: model.into(),
            status: None,
            capsule_id: None,
            nonce: None,
            nonce_source: None,
        }
    }

    pub fn terminal(
        exchange_id: impl Into<String>,
        dispatch_path: OpenAiExchangeDispatchPath,
        model: impl Into<String>,
        status: Option<u16>,
        marker: Option<CapsuleMarker>,
        nonce_source: Option<ClientNonceSource>,
    ) -> Self {
        Self {
            exchange_id: exchange_id.into(),
            dispatch_path,
            phase: OpenAiExchangePhase::Terminal,
            model: model.into(),
            status,
            capsule_id: marker.as_ref().map(|marker| marker.capsule_id.clone()),
            nonce: marker.as_ref().map(|marker| marker.nonce.clone()),
            nonce_source,
        }
    }

    /// Effective-request envelope for the `RemoteMesh` dispatch path,
    /// carrying the nonce this node is about to forward to the peer
    /// unchanged — so a plugin observing only the effective event already
    /// knows what a later client ack must sign over, rather than having to
    /// wait for the terminal event. `capsule_id` stays absent: this node
    /// mints nothing on this path.
    ///
    /// **`nonce_source` asymmetry:** when the routing node (node A) minted
    /// the fallback nonce, it reports `SidecarGeneratedFallback` here.
    /// The receiving peer (node B) strips the `x-capsule-nonce-origin`
    /// header (anti-smuggling) and therefore reports `ClientSupplied` for
    /// the same nonce on its own envelope. Both are locally correct; a
    /// consumer joining both envelopes will see two different `nonce_source`
    /// values for the same nonce — see the field-level doc on
    /// [`OpenAiExchangeEnvelope::nonce_source`] for the full explanation.
    pub fn effective_remote_mesh(
        exchange_id: impl Into<String>,
        model: impl Into<String>,
        nonce: Option<String>,
        nonce_source: Option<ClientNonceSource>,
    ) -> Self {
        Self {
            exchange_id: exchange_id.into(),
            dispatch_path: OpenAiExchangeDispatchPath::RemoteMesh,
            phase: OpenAiExchangePhase::EffectiveRequest,
            model: model.into(),
            status: None,
            capsule_id: None,
            nonce,
            nonce_source,
        }
    }

    /// Terminal envelope for the `RemoteMesh` dispatch path — a routing node
    /// observing (not serving) an exchange it forwarded to a peer.
    ///
    /// Unlike [`Self::terminal`]'s `marker`, which bundles a capsule_id this
    /// node minted together with the nonce that capsule is correlated
    /// against, a routing node mints nothing here: `nonce` is the same
    /// client-contributed value forwarded to the peer unchanged (present
    /// only when the request already carries a stabilized nonce). No peer
    /// response header is read back on this path, so `capsule_id` stays
    /// absent, same as the plugin-served terminal event.
    ///
    /// **`nonce_source` asymmetry:** same as [`Self::effective_remote_mesh`]
    /// — node A reports `SidecarGeneratedFallback` when it minted the nonce;
    /// node B strips the origin header (anti-smuggling) and reports
    /// `ClientSupplied` for the identical nonce. See
    /// [`OpenAiExchangeEnvelope::nonce_source`] for the full explanation.
    pub fn terminal_remote_mesh(
        exchange_id: impl Into<String>,
        model: impl Into<String>,
        status: Option<u16>,
        nonce: Option<String>,
        nonce_source: Option<ClientNonceSource>,
    ) -> Self {
        Self {
            exchange_id: exchange_id.into(),
            dispatch_path: OpenAiExchangeDispatchPath::RemoteMesh,
            phase: OpenAiExchangePhase::Terminal,
            model: model.into(),
            status,
            capsule_id: None,
            nonce,
            nonce_source,
        }
    }
}

/// Publishes [`OpenAiExchangeEnvelope`]s to whatever is subscribed on
/// [`OPENAI_EXCHANGE_CHANNEL`] — an out-of-process plugin in production, a
/// recording double in tests. Fire-and-forget by design, mirroring
/// [`OpenAiHookPolicy`]'s own observer methods: exchange delivery to a
/// plugin must never affect whether the client's own request succeeds.
#[async_trait]
pub trait OpenAiExchangeChannel: Send + Sync + 'static {
    async fn publish(&self, event: &OpenAiExchangeEnvelope);
}

#[async_trait]
impl OpenAiExchangeChannel for PluginManager {
    async fn publish(&self, event: &OpenAiExchangeEnvelope) {
        let body = match serde_json::to_vec(event) {
            Ok(body) => body,
            Err(error) => {
                tracing::warn!(%error, "failed to serialize openai exchange event");
                return;
            }
        };
        if let Err(error) = self
            .broadcast_channel_message(
                OPENAI_EXCHANGE_CHANNEL,
                "application/json",
                body,
                &event.exchange_id,
            )
            .await
        {
            tracing::warn!(%error, "failed to publish openai exchange event to plugins");
        }
    }
}

/// Bridges path 1 (`openai-frontend`'s typed hook seam) to
/// [`OpenAiExchangeChannel`], so an out-of-process plugin observes the same
/// effective-request/terminal events this crate's `MeshAutoHookPolicy`
/// already sees in-process. Compose alongside other [`OpenAiHookPolicy`]
/// implementors rather than in place of them — this bridge only observes and
/// mints capsule markers, it never mutates or denies a request.
pub struct OpenAiExchangeHookBridge {
    channel: Arc<dyn OpenAiExchangeChannel>,
}

impl OpenAiExchangeHookBridge {
    pub fn new(channel: Arc<dyn OpenAiExchangeChannel>) -> Self {
        Self { channel }
    }
}

#[async_trait]
impl OpenAiHookPolicy for OpenAiExchangeHookBridge {
    async fn on_effective_chat_completion(
        &self,
        _request: &ChatCompletionRequest,
        route: &ChatExchangeRoute,
    ) {
        self.channel
            .publish(&OpenAiExchangeEnvelope::effective(
                route.exchange_id.clone(),
                OpenAiExchangeDispatchPath::TypedFrontend,
                route.model.clone(),
            ))
            .await;
    }

    async fn on_chat_completion_terminal(
        &self,
        request: &ChatCompletionRequest,
        exchange_id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        let (status, marker): (Option<u16>, Option<CapsuleMarker>) = match outcome {
            ChatCompletionOutcome::Success { response } => {
                (Some(200), response.capsule_marker.clone())
            }
            ChatCompletionOutcome::Error { status, .. } => (Some(*status), None),
            ChatCompletionOutcome::Denied { status, .. } => (Some(*status), None),
            // `ChatCompletionOutcome::Cancelled` and any future variant:
            // no HTTP response was produced, so there's nothing to report
            // beyond a status-free terminal event.
            _ => (None, None),
        };
        // Recomputed from `request` rather than threaded through
        // `CapsuleMarker` (an `openai-frontend` public type this crate
        // doesn't own): both this and `capsule_marker_for_response` below
        // read the same `client_nonce` field, so they always agree on which
        // branch was taken.
        let nonce_source = marker.as_ref().map(|_| client_nonce_source(request));
        self.channel
            .publish(&OpenAiExchangeEnvelope::terminal(
                exchange_id,
                OpenAiExchangeDispatchPath::TypedFrontend,
                request.model.clone(),
                status,
                marker,
                nonce_source,
            ))
            .await;
    }

    /// Reference nonce sourcing for the rung-ladder response leg: a
    /// client-contributed `client_nonce` (landing in `request.extra` via
    /// `ChatCompletionRequest`'s `#[serde(flatten)]` bag, the same mechanism
    /// `mesh_hooks` already uses) wins; absent that, mint a fallback rather
    /// than silently mislabeling it as client-supplied — mirroring
    /// `capsule-emit-mesh`'s own `client_nonce_source` tri-state
    /// (`client_supplied` / `sidecar_generated_fallback`). The `fallback-`
    /// prefix stays for readability, but [`ClientNonceSource`] (see
    /// `on_chat_completion_terminal`) is the authoritative signal — a plugin
    /// must not infer sourcing by sniffing this string.
    async fn capsule_marker_for_response(
        &self,
        request: &ChatCompletionRequest,
        response: &ChatCompletionResponse,
    ) -> Option<CapsuleMarker> {
        let nonce = match client_nonce_source(request) {
            ClientNonceSource::ClientSupplied => request
                .extra
                .get("client_nonce")
                .and_then(|value| value.as_str())
                .expect("client_nonce_source() confirmed a client_nonce string is present")
                .to_string(),
            ClientNonceSource::SidecarGeneratedFallback => format!("fallback-{}", response.id),
        };
        Some(CapsuleMarker {
            capsule_id: format!("capsule-{}", response.id),
            nonce,
        })
    }
}

/// The single place that decides client-supplied vs. sidecar-minted, used by
/// both [`OpenAiExchangeHookBridge::capsule_marker_for_response`] (to choose
/// the nonce value) and [`OpenAiExchangeHookBridge::on_chat_completion_terminal`]
/// (to label it on the envelope) so the two can never disagree.
fn client_nonce_source(request: &ChatCompletionRequest) -> ClientNonceSource {
    if request
        .extra
        .get("client_nonce")
        .and_then(|value| value.as_str())
        .is_some()
    {
        ClientNonceSource::ClientSupplied
    } else {
        ClientNonceSource::SidecarGeneratedFallback
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use openai_frontend::{ChatCompletionOutcome, HookedOpenAiBackend, OpenAiBackend, Usage};

    use super::*;

    #[derive(Default)]
    struct RecordingChannel {
        events: Mutex<Vec<OpenAiExchangeEnvelope>>,
    }

    #[async_trait]
    impl OpenAiExchangeChannel for RecordingChannel {
        async fn publish(&self, event: &OpenAiExchangeEnvelope) {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    struct EchoBackend;

    #[async_trait]
    impl OpenAiBackend for EchoBackend {
        async fn models(&self) -> openai_frontend::OpenAiResult<Vec<openai_frontend::ModelObject>> {
            Ok(Vec::new())
        }

        async fn chat_completion(
            &self,
            request: ChatCompletionRequest,
        ) -> openai_frontend::OpenAiResult<ChatCompletionResponse> {
            Ok(ChatCompletionResponse::new(
                request.model,
                "ok",
                Usage::new(1, 1),
            ))
        }

        async fn chat_completion_stream(
            &self,
            _request: ChatCompletionRequest,
            _context: openai_frontend::OpenAiRequestContext,
        ) -> openai_frontend::OpenAiResult<openai_frontend::ChatCompletionStream> {
            Ok(Box::pin(futures_util::stream::empty()))
        }
    }

    fn chat_request(model: &str) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
    }

    /// Reference: a full request through `HookedOpenAiBackend` wired with
    /// this bridge publishes both the effective-request and terminal events
    /// on the typed-frontend path, and the terminal event carries the same
    /// capsule marker that (per the openai-frontend-crate tests) also became
    /// the client-visible `X-Capsule-Id` header — proving the plugin sees
    /// exactly what the client's response leg exposed, not a divergent copy.
    #[tokio::test]
    async fn typed_frontend_path_publishes_effective_and_terminal_with_capsule_marker() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = Arc::new(OpenAiExchangeHookBridge::new(channel.clone()));
        let hooked = HookedOpenAiBackend::new(Arc::new(EchoBackend), bridge);

        let response = hooked
            .chat_completion(chat_request("gpt-mesh"))
            .await
            .expect("backend call succeeds");

        let events = channel.events.lock().unwrap();
        assert_eq!(events.len(), 2, "one effective-request, one terminal");

        assert_eq!(
            events[0].dispatch_path,
            OpenAiExchangeDispatchPath::TypedFrontend
        );
        assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
        assert_eq!(events[0].model, "gpt-mesh");

        assert_eq!(events[1].phase, OpenAiExchangePhase::Terminal);
        assert_eq!(events[1].status, Some(200));
        assert!(!events[0].exchange_id.is_empty());
        assert_eq!(events[0].exchange_id, events[1].exchange_id);
        let capsule_id = events[1]
            .capsule_id
            .as_deref()
            .expect("terminal event carries the capsule id");
        assert_eq!(
            capsule_id,
            response
                .capsule_marker
                .as_ref()
                .expect("router-visible marker")
                .capsule_id
        );
    }

    #[tokio::test]
    async fn client_supplied_nonce_survives_into_the_terminal_event() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = Arc::new(OpenAiExchangeHookBridge::new(channel.clone()));
        let hooked = HookedOpenAiBackend::new(Arc::new(EchoBackend), bridge);

        let mut request = chat_request("gpt-mesh");
        request
            .extra
            .insert("client_nonce".to_string(), serde_json::json!("abc123"));

        hooked
            .chat_completion(request)
            .await
            .expect("backend call succeeds");

        let events = channel.events.lock().unwrap();
        assert_eq!(events[1].nonce.as_deref(), Some("abc123"));
        assert_eq!(
            events[1].nonce_source,
            Some(ClientNonceSource::ClientSupplied),
            "a plugin must be able to trust nonce_source over sniffing the nonce string"
        );
    }

    /// When the client contributes no nonce, the mint still labels it
    /// `sidecar_generated_fallback` via `nonce_source` — not just the
    /// human-readable `fallback-` prefix on the nonce string itself.
    #[tokio::test]
    async fn absent_client_nonce_is_labeled_sidecar_generated_fallback() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = Arc::new(OpenAiExchangeHookBridge::new(channel.clone()));
        let hooked = HookedOpenAiBackend::new(Arc::new(EchoBackend), bridge);

        hooked
            .chat_completion(chat_request("gpt-mesh"))
            .await
            .expect("backend call succeeds");

        let events = channel.events.lock().unwrap();
        assert!(
            events[1]
                .nonce
                .as_deref()
                .is_some_and(|n| n.starts_with("fallback-"))
        );
        assert_eq!(
            events[1].nonce_source,
            Some(ClientNonceSource::SidecarGeneratedFallback)
        );
    }

    /// A denial never reaches the backend, so there is no response to mint a
    /// marker from — the bridge's own terminal handling (not a stand-in) must
    /// publish a status-only event with no capsule id.
    #[tokio::test]
    async fn denied_outcome_publishes_terminal_without_a_capsule_marker() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = OpenAiExchangeHookBridge::new(channel.clone());
        let request = chat_request("gpt-mesh");
        let denial = ChatCompletionOutcome::Denied {
            status: 400,
            reason: "denied by policy",
        };

        bridge
            .on_chat_completion_terminal(&request, "exchange-1", &denial)
            .await;

        let events = channel.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].exchange_id, "exchange-1");
        assert_eq!(events[0].status, Some(400));
        assert!(events[0].capsule_id.is_none());
        assert!(events[0].nonce.is_none());
        assert!(events[0].nonce_source.is_none());
    }

    /// The exact scenario `TerminalGuard` (in `openai-frontend`) exists to
    /// close: the backend future never returns, so `HookedOpenAiBackend`
    /// reports `ChatCompletionOutcome::Cancelled` instead of nothing — this
    /// bridge must still publish a terminal event for it, with no status,
    /// capsule id, nonce, or nonce_source to report.
    #[tokio::test]
    async fn cancelled_outcome_publishes_a_status_free_terminal_event() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = OpenAiExchangeHookBridge::new(channel.clone());
        let request = chat_request("gpt-mesh");

        bridge
            .on_chat_completion_terminal(&request, "exchange-1", &ChatCompletionOutcome::Cancelled)
            .await;

        let events = channel.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].phase, OpenAiExchangePhase::Terminal);
        assert!(events[0].status.is_none());
        assert!(events[0].capsule_id.is_none());
        assert!(events[0].nonce.is_none());
        assert!(events[0].nonce_source.is_none());
    }

    struct DelayedBackend {
        delay: std::time::Duration,
    }

    #[async_trait]
    impl OpenAiBackend for DelayedBackend {
        async fn models(&self) -> openai_frontend::OpenAiResult<Vec<openai_frontend::ModelObject>> {
            Ok(Vec::new())
        }

        async fn chat_completion(
            &self,
            request: ChatCompletionRequest,
        ) -> openai_frontend::OpenAiResult<ChatCompletionResponse> {
            tokio::time::sleep(self.delay).await;
            Ok(ChatCompletionResponse::new(
                request.model,
                "ok",
                Usage::new(1, 1),
            ))
        }

        async fn chat_completion_stream(
            &self,
            _request: ChatCompletionRequest,
            _context: openai_frontend::OpenAiRequestContext,
        ) -> openai_frontend::OpenAiResult<openai_frontend::ChatCompletionStream> {
            Ok(Box::pin(futures_util::stream::empty()))
        }
    }

    /// Two concurrent exchanges on the same model must not be pairable by
    /// mere arrival order — the terminal event for the exchange with the
    /// shorter backend delay lands before the effective event of neither
    /// exchange lines up with it positionally. Only matching `exchange_id`
    /// correctly recovers each exchange's own effective/terminal pair.
    #[tokio::test(start_paused = true)]
    async fn concurrent_exchanges_on_the_same_model_pair_by_exchange_id_not_by_arrival_order() {
        let channel = Arc::new(RecordingChannel::default());
        let bridge = Arc::new(OpenAiExchangeHookBridge::new(channel.clone()));
        let slow = HookedOpenAiBackend::new(
            Arc::new(DelayedBackend {
                delay: std::time::Duration::from_millis(50),
            }),
            bridge.clone(),
        );
        let fast = HookedOpenAiBackend::new(
            Arc::new(DelayedBackend {
                delay: std::time::Duration::from_millis(1),
            }),
            bridge,
        );

        let (slow_result, fast_result) = tokio::join!(
            slow.chat_completion(chat_request("gpt-mesh")),
            fast.chat_completion(chat_request("gpt-mesh")),
        );
        slow_result.expect("slow exchange succeeds");
        fast_result.expect("fast exchange succeeds");

        let events = channel.events.lock().unwrap();
        assert_eq!(events.len(), 4, "two effective + two terminal events");
        assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
        assert_eq!(events[1].phase, OpenAiExchangePhase::EffectiveRequest);
        assert_eq!(events[2].phase, OpenAiExchangePhase::Terminal);
        assert_eq!(events[3].phase, OpenAiExchangePhase::Terminal);
        assert_ne!(
            events[0].exchange_id, events[1].exchange_id,
            "each exchange mints its own id"
        );

        // The fast exchange finishes first, so its terminal event (index 2)
        // is adjacent to the slow exchange's effective event (index 0) by
        // position — but it must still pair with the fast effective event
        // (index 1) by id, and the slow terminal (index 3) with the slow
        // effective (index 0).
        assert_eq!(
            events[2].exchange_id, events[1].exchange_id,
            "fast exchange's terminal event pairs with its own effective event"
        );
        assert_eq!(
            events[3].exchange_id, events[0].exchange_id,
            "slow exchange's terminal event pairs with its own effective event"
        );
    }

    // --- #1668 review round: RemoteMesh / RawProxy envelope shapes ---
    //
    // Shape tests only — these call envelope constructors directly and assert
    // on the fields they set. They do NOT invoke `route_missing_local_model`
    // or `try_route_plugin_model`, so they would still pass if the publish
    // calls inside those routing functions were deleted. Real end-to-end
    // publish coverage (including both envelopes being emitted and their
    // nonce_source values) lives in `ingress_tests::tests`.

    /// Verifies the envelope constructor shape for the RemoteMesh effective +
    /// terminal pair: both envelopes carry `RemoteMesh` dispatch path, the
    /// nonce and nonce_source are threaded onto both, and `capsule_id` is
    /// absent on both (this node mints nothing on the remote-mesh path).
    ///
    /// // Shape test only — does not invoke the routing function.
    /// // Real publish coverage is in ingress_tests::tests.
    #[tokio::test]
    async fn envelope_shape_remote_mesh_effective_and_terminal_carry_nonce_fields() {
        let channel = RecordingChannel::default();
        let nonce = Some("6d7d8d2e-3f4a-4b5c-8d9e-0a1b2c3d4e5f".to_string());
        let nonce_source = Some(ClientNonceSource::ClientSupplied);

        channel
            .publish(&OpenAiExchangeEnvelope::effective_remote_mesh(
                "exch-rm-1",
                "hermes-2-pro-mistral-7b",
                nonce.clone(),
                nonce_source,
            ))
            .await;
        channel
            .publish(&OpenAiExchangeEnvelope::terminal_remote_mesh(
                "exch-rm-1",
                "hermes-2-pro-mistral-7b",
                Some(200),
                nonce.clone(),
                nonce_source,
            ))
            .await;

        let events = channel.events.lock().unwrap();
        assert_eq!(events.len(), 2, "one effective-request, one terminal");

        assert_eq!(
            events[0].dispatch_path,
            OpenAiExchangeDispatchPath::RemoteMesh
        );
        assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
        assert_eq!(events[0].nonce, nonce);
        assert_eq!(events[0].nonce_source, nonce_source);
        assert!(events[0].capsule_id.is_none());

        assert_eq!(
            events[1].dispatch_path,
            OpenAiExchangeDispatchPath::RemoteMesh
        );
        assert_eq!(events[1].phase, OpenAiExchangePhase::Terminal);
        assert_eq!(events[1].exchange_id, events[0].exchange_id);
        assert_eq!(events[1].status, Some(200));
        assert_eq!(events[1].nonce, nonce);
        assert_eq!(events[1].nonce_source, nonce_source);
        assert!(events[1].capsule_id.is_none());
    }

    /// Verifies the envelope constructor shape for the RawProxy effective +
    /// terminal pair: both envelopes carry `RawProxy` dispatch path, and
    /// nonce/nonce_source/capsule_id are all absent (the raw-proxy path never
    /// runs through `openai-frontend`'s `OpenAiHookPolicy`, so no marker is
    /// minted).
    ///
    /// // Shape test only — does not invoke the routing function.
    /// // Real publish coverage is in ingress_tests::tests.
    #[tokio::test]
    async fn envelope_shape_raw_proxy_effective_and_terminal_have_no_marker() {
        let channel = RecordingChannel::default();

        channel
            .publish(&OpenAiExchangeEnvelope::effective(
                "exch-rp-1",
                OpenAiExchangeDispatchPath::RawProxy,
                "acme/plugin-model",
            ))
            .await;
        channel
            .publish(&OpenAiExchangeEnvelope::terminal(
                "exch-rp-1",
                OpenAiExchangeDispatchPath::RawProxy,
                "acme/plugin-model",
                Some(200),
                None,
                None,
            ))
            .await;

        let events = channel.events.lock().unwrap();
        assert_eq!(events.len(), 2, "one effective-request, one terminal");

        assert_eq!(
            events[0].dispatch_path,
            OpenAiExchangeDispatchPath::RawProxy
        );
        assert_eq!(events[0].phase, OpenAiExchangePhase::EffectiveRequest);
        assert!(events[0].nonce.is_none());
        assert!(events[0].capsule_id.is_none());

        assert_eq!(
            events[1].dispatch_path,
            OpenAiExchangeDispatchPath::RawProxy
        );
        assert_eq!(events[1].phase, OpenAiExchangePhase::Terminal);
        assert_eq!(events[1].exchange_id, events[0].exchange_id);
        assert_eq!(events[1].status, Some(200));
        assert!(events[1].nonce.is_none());
        assert!(events[1].nonce_source.is_none());
        assert!(events[1].capsule_id.is_none());
    }
}
