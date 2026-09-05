# openai-frontend

Reusable OpenAI-compatible HTTP frontend primitives for mesh and staged runtime
entry points.

This crate owns the public API shapes and route machinery that should not be
duplicated inside `skippy-server`. Stage server code should provide a thin
backend adapter that implements the frontend trait, while this crate handles
request/response JSON, OpenAI-style errors, `/v1/models`, chat completions, and
streaming Server-Sent Events framing. Mesh uses this as the single OpenAI
surface for embedded single-stage and stage-split serving.

Mesh-local compatibility wrappers should stay thin. Request normalization,
Responses API translation, stream-chunk parsing, stream usage conversion, and
upstream error mapping are owned here so the host binary and staged runtimes do
not grow separate OpenAI compatibility layers.

The compatibility target is the mesh OpenAI surface, not llama-server process
compatibility. Request fields such as tools, structured-output shape,
logprobs, and `/v1/responses` are parsed and normalized here; backend support
is advertised or rejected explicitly by the runtime adapter.

For the concrete benchy command and contract, see
[`docs/skippy/LLAMA_BENCHY.md`](../../docs/skippy/LLAMA_BENCHY.md).

## Supported API Surface

| Surface | Status | Notes |
|---|---|---|
| `GET /v1/models` | Supported | Returns backend-provided model objects with opaque ids such as `org/repo:Q4_K_M`. |
| `POST /v1/chat/completions` | Supported | Handles streaming and non-streaming response shapes. |
| `POST /v1/completions` | Supported | Handles streaming and non-streaming response shapes. |
| `POST /v1/responses` | Supported | Adapts OpenAI responses requests onto chat/completion backend calls and preserves response metadata where possible. |
| `GET /health` / `GET /healthz` | Supported | Lightweight liveness probes for hosts and CI smoke tests. |
| `GET /readyz` | Supported | Backend readiness probe that verifies model discovery through `OpenAiBackend::models`. |
| Server-Sent Events | Supported | Emits OpenAI-style JSON chunks and `[DONE]`. |
| `model` | Supported | Opaque exact-match id; Mesh-style refs such as `org/repo:Q4_K_M` pass through without frontend parsing. |
| `messages` | Supported | String and text-part content are parsed; stage backend applies the model chat template through llama.cpp `llama-common`. |
| `prompt` | Supported | String and string-array prompts are accepted; token prompts parse but are rejected until a backend can honor token IDs directly. |
| `max_tokens` / `max_completion_tokens` | Parsed | Backend decides enforcement. |
| `stop` | Parsed | Backend decides enforcement. |
| `n` / `best_of` | Parsed | Single-choice requests are accepted; multi-choice generation is rejected until the response/backend path supports it. |
| `temperature`, `top_p`, `seed` | Supported | Stage backend passes supported sampling controls to llama.cpp sampling. |
| penalties / `logit_bias` | Supported | Presence/frequency/repeat penalties and token-id logit bias are passed to llama.cpp. |
| `logprobs` / `top_logprobs` | Frontend-compatible | Parsed, validated, and preserved. Backend decides whether logits/probability data can be produced. |
| `tools` / `tool_choice` | Frontend-compatible | Parsed, validated, and preserved. Backend decides whether tool-call generation is available. |
| `response_format` | Frontend-compatible | Text and structured-output shapes are parsed and preserved. Backend decides whether constrained decoding is available. |
| OpenAI-style error envelope | Supported | Includes strict `type`, `param`, and `code` fields. |
| OpenAI-style HTTP fallbacks | Supported | Unknown routes, unsupported methods, invalid JSON, and oversized JSON return the shared error envelope. |
| Request body limit | Supported | Configurable via `OpenAiFrontendConfig`; defaults to 4 MiB. |
| Request IDs | Supported | Reuses only a valid UUID `x-request-id`; missing or invalid values are replaced with a generated UUID. Every response returns the canonical hyphenated UUID and the frontend emits a tracing event with method, URI, status, and request ID. |
| Client nonce | Supported | Accepts `x-capsule-client-nonce` only when it is exactly one valid UUIDv4; a missing, invalid, non-UUIDv4, or duplicated value is replaced with a freshly minted UUIDv4. When this frontend mints the value it stamps `x-capsule-nonce-origin: frontend`; a forwarded (client-supplied) nonce carries no origin marker, and any inbound `x-capsule-nonce-origin` is always stripped so a caller cannot forge it. Both headers are echoed on covered responses: the axum router and, via the host runtime's forwarding rebuild, the public `:9337` proxy JSON/SSE paths (including the pipeline/MoA strong-model path and remapped upstream error responses). Locally synthesized error responses (e.g. no-target `503`s and `/v1/models` listings) do not yet carry the headers; threading the request nonce onto those senders is a cross-cutting signature change tracked as a follow-up. The origin marker asserts only that *this* frontend minted the value, not that it is the original ingress for a remote-routed request. |
| Backend timeout | Supported | Configurable via `OpenAiFrontendConfig` or the `MESH_OPENAI_BACKEND_TIMEOUT_SECS` environment variable; defaults to 600 seconds (`0` disables it) and maps timeouts to OpenAI-shaped 504 errors. |
| Agent session header | Supported | Set `MESH_AGENT_SESSION_HEADER` to accept a trusted upstream header as the stable agent-session identity. |
| Mesh routing headers | Supported (remote-mesh routing only) | Implemented by the host runtime's public `:9337` proxy, not this crate, but documented here as part of the same header contract. `x-mesh-target: <EndpointId>` forces dispatch to exactly that peer if it currently advertises the requested model; otherwise the request fails closed with a `409` naming the mismatch — it is never silently rerouted to another peer or served locally. `x-mesh-exclude: <EndpointId>[,...]` removes one or more peers from the candidate set before selection (comma-separated within one header, or repeated). The routing node echoes the resolved peer back as `x-mesh-served-by: <EndpointId>` on the response only when `x-mesh-target` was used. Both request headers are no-ops when absent, and an unparseable value is a `400`, never a silently ignored one. |
| embeddings/rerank/infill/audio/vision | Out of scope | Not needed for staged text benchmark entrypoints. |

## Shape

```mermaid
flowchart TB
    C["OpenAI-compatible client<br/>chat, completions, responses"] --> R["openai-frontend<br/>Axum routes"]
    R --> Parse["request parsing<br/>validation<br/>normalization<br/>OpenAI errors"]
    Parse --> B["OpenAiBackend implementation"]
    B --> Local["embedded single-stage<br/>skippy runtime"]
    B --> Chain["mesh stage-0 route<br/>skippy-stage/2 chain"]
    Local --> Resp["OpenAI response JSON"]
    Chain --> Resp
    Resp --> R
    R --> C
```

The backend boundary is intentionally small:

```rust
#[async_trait]
pub trait OpenAiBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>>;
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse>;
    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream>;
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> OpenAiResult<CompletionResponse>;
    async fn completion_stream(
        &self,
        request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream>;
}
```

## Model Identity

`openai-frontend` treats model ids as opaque OpenAI-facing strings. For
Mesh-owned routing, the expected user-facing form is a Hugging Face coordinate
plus artifact selector, for example `org/repo:Q4_K_M`. The suffix is an artifact
selector, not a stage-server topology or serving-backend variant.

Backends should advertise the exact ids they accept through `/v1/models` and
perform exact string matching on requests. Resolution to a concrete Hugging Face
revision, GGUF file, split-shard distribution, local runtime, or staged binary
chain remains backend-owned.

## Backend Responsibility Matrix

| Area | Frontend responsibility | Backend responsibility | Status |
|---|---|---|---|
| `/v1/models` | Route shape and model-object serialization | Advertise exact full model refs accepted by the runtime | Supported |
| `/v1/chat/completions` | Parse common and advanced fields, stream/non-stream envelopes | Tokenization, sampling, stop handling, usage, feature execution | Supported with backend feature guards |
| `/v1/completions` | Prompt parsing and response envelopes | Token prompts, sampling, stop handling, usage | Supported with backend feature guards |
| `/v1/responses` | Translate request/response shapes onto the backend contract | Execute the resulting chat/completion request | Supported |
| HTTP operations | Health/readiness, fallbacks, payload limits, content-type handling | Model readiness and backend timeouts | Supported |
| Streaming | SSE chunks, `[DONE]`, cancellation context | Produce deltas, usage, and optional logprob/tool metadata | Supported with backend feature guards |
| Chat templates | Preserve OpenAI message shape | Apply model-aware chat templates through the skippy ABI | Backend-owned |
| Sampling | Parse OpenAI sampling fields | Apply supported llama sampling controls and reject unsupported knobs | Backend-owned |
| Stop handling | Preserve stop fields | Hold back streamed text enough to avoid leaking stop strings | Backend-owned |
| Context limits | Carry requested limits | Validate prompt plus generation against `ctx_size` | Backend-owned |
| Concurrency | Request timeout and cancellation | Runtime slots, batching, lane pools, and backpressure | Backend-owned |
| Cache behavior | Request IDs for correlation | Prefix/state cache policy and telemetry | Backend-owned |
| Errors | Shared OpenAI error envelope | Return precise unsupported/runtime errors | Supported |
| Usage accounting | Response field shape | Prompt/completion/total token counts | Backend-owned |
| Multi-choice generation | Parse `n` / `best_of` | Generate or reject multi-choice requests | Backend-owned |
| Logprobs | Parse and preserve request/response shape | Expose logits/probabilities | Frontend ready; backend-gated |
| Tools/function calling | Parse and preserve tool schemas and tool-call response shape | Generate tool calls | Frontend ready; backend-gated |
| JSON schema/grammar | Parse and preserve `response_format` | Constrained decoding | Frontend ready; backend-gated |
| Embeddings/rerank/infill/audio | Route fallback/error handling | Runtime implementation if reintroduced | Out of current scope |
| Metrics | Request IDs and tracing context | Stage/OpenAI telemetry emitted to `metrics-server` | Supported |

## Stage-Server Integration

`skippy-server` and mesh use this crate by implementing `OpenAiBackend` for a
small adapter:

- local text/runtime backend for single-stage smoke tests
- staged chain backend that connects to the first `serve-binary` endpoint

That keeps `serve-openai` and the embedded mesh path thin: parse or build the
runtime config, construct the backend, pass it to `openai_frontend::router`,
and serve the Axum app.
