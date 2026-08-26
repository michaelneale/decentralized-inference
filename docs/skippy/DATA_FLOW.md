# Stage Data Flow

This note describes the four-stage benchmark data path and the relative size of
each flow. The reference run is the mixed 192-prompt Qwen3.6 benchmark using
`f32` activation wire format, `--prefill-chunk-size 256`,
`--stage-max-inflight 2`, `--stage-reply-credit-limit 1`, `--n-gpu-layers -1`,
summary telemetry, and balanced splits `10,20,30`.

## Generation 2 Chained Flow

```mermaid
flowchart LR
    Client["OpenAI client / benchmark driver"] --> Mesh["mesh-llm<br/>openai-frontend + stage-0 route"]
    Mesh --> D["token IDs + control<br/>~183 KiB total corpus"]
    D --> S0["stage-0<br/>layers 0..10<br/>embeddings + first block"]

    S0 -->|"activation frames<br/>~364.7 MiB total<br/>max prefill frame ~2.0 MiB<br/>decode frame 8 KiB/token"| S1["stage-1<br/>layers 10..20"]

    S1 -->|"activation frames<br/>~364.7 MiB total<br/>max prefill frame ~2.0 MiB<br/>decode frame 8 KiB/token"| S2["stage-2<br/>layers 20..30"]

    S2 -->|"activation frames<br/>~364.7 MiB total<br/>max prefill frame ~2.0 MiB<br/>decode frame 8 KiB/token"| S3["stage-3<br/>layers 30..40<br/>output logits/token"]

    S3 -->|"small replies<br/>predicted token + ACKs"| S2
    S2 -->|"small replies<br/>ACKs / predicted token"| S1
    S1 -->|"small replies<br/>ACKs / predicted token"| S0
    S0 -->|"predicted tokens<br/>~6 KiB generated token IDs"| D
    D --> Mesh
    Mesh --> Client

    Mesh -.->|"run/topology metadata<br/>small"| M
    S0 -.->|"summary OTLP<br/>small"| M["metrics-server"]
    S1 -.->|"summary OTLP<br/>small"| M
    S2 -.->|"summary OTLP<br/>small"| M
    S3 -.->|"summary OTLP<br/>small"| M
```

Generation 2 returned prediction-bearing replies along the same chain as
activation forwarding. With four stages, each decode token crossed the three
activation links and then crossed three reply links before stage 0 could emit
the token. On a topology with a fixed 10 ms delay per inter-stage hop, the reply
chain alone makes the hot path six hops, or about 60 ms before compute.

## Generation 7 Direct Prediction Return and Verify Retirement

Stage protocol generation 7 is a compatibility-breaking change. A peer is stage
compatible only when it advertises both `skippy-stage/2` and
the complete `stage-generation-7` control/status/content-identity bundle.
Prediction-bearing messages return directly from the final/readout stage to the
driver-facing stage. Intermediate stages
continue to forward activations and may handle cold-path control acknowledgments,
but they are not part of the decode-token prediction return path.

```mermaid
flowchart LR
    Client["OpenAI client / benchmark driver"] --> Mesh["mesh-llm<br/>openai-frontend + stage-0 route"]
    Mesh --> D["token IDs + control<br/>~183 KiB total corpus"]
    D --> S0["stage-0<br/>layers 0..10<br/>embeddings + first block"]

    S0 -->|"activation frames<br/>~364.7 MiB total<br/>max prefill frame ~2.0 MiB<br/>decode frame 8 KiB/token"| S1["stage-1<br/>layers 10..20"]
    S1 -->|"activation frames<br/>~364.7 MiB total<br/>max prefill frame ~2.0 MiB<br/>decode frame 8 KiB/token"| S2["stage-2<br/>layers 20..30"]
    S2 -->|"activation frames<br/>~364.7 MiB total<br/>max prefill frame ~2.0 MiB<br/>decode frame 8 KiB/token"| S3["stage-3<br/>layers 30..40<br/>output logits/token"]

    S3 -->|"predicted token(s)<br/>direct return"| S0
    S0 -->|"predicted tokens<br/>~6 KiB generated token IDs"| D
    D --> Mesh
    Mesh --> Client

    S3 -.->|"control ACKs / stats<br/>cold path"| S2
    S2 -.->|"control ACKs / stats<br/>cold path"| S1
    S1 -.->|"control ACKs / stats<br/>cold path"| S0

    Mesh -.->|"run/topology metadata<br/>small"| M
    S0 -.->|"summary OTLP<br/>small"| M["metrics-server"]
    S1 -.->|"summary OTLP<br/>small"| M
    S2 -.->|"summary OTLP<br/>small"| M
    S3 -.->|"summary OTLP<br/>small"| M
```

With the same four stages and 10 ms inter-stage delay, the no-spec decode hot
path becomes `S0 -> S1 -> S2 -> S3 -> S0`: four hops, or about 40 ms before
compute. That removes two serialized reply hops from every generated token.

## Stale Verify-Window Discard

Generation 5 also adds the `DiscardStaleWindows` control frame (wire kind
23), which is what made the generation compatibility-breaking: a
generation-4 peer rejects the kind outright and drops the request
connection.

Run-ahead admission dispatches verify windows before their predecessors are
verified, so a rejection strands every window queued behind the divergence.
Those windows are already on the wire. The coordinator sends one discard
naming a contiguous `[min_window_id, max_window_id]` range for the request,
and it travels the same ordered path as the windows it cancels, so it is
read after them.

The handling rule differs by position in the chain, and this is the
non-obvious part:

- **Middle stages forward the frame and still execute the stale windows.**
  A stage that is not the final stage owns KV state its downstream neighbour
  depends on; skipping its forward pass would desynchronize the chain. It
  passes the discard along so the frame reaches the stage that can act on it.
- **The final/readout stage skips the named windows.** It is the only stage
  whose output is thrown away by a discard, so skipping there is what
  actually saves the work — the expensive readout and the direct prediction
  return back to stage 0.

The receiving stage records the range the moment it parses the frame, ahead
of executing the backlog queued in front of it, which is what lets the skip
take effect before the stale tail runs. See `INBOUND_LOOKAHEAD_BYTES` for
the case where a wide-frame backlog outruns that read-ahead and the tail
executes anyway.

## Relative Sizes

| Flow | Size |
| --- | ---: |
| Driver to stage-0 prompt tokens | ~177 KiB prefill token IDs |
| Driver/stage decode token IDs | ~6 KiB token IDs |
| stage-0 -> stage-1 activations | 364.7 MiB |
| stage-1 -> stage-2 activations | 364.7 MiB |
| stage-2 -> stage-3 activations | 364.7 MiB |
| Per-boundary activation total | ~365 MiB |
| Total activation traffic across 3 boundaries | ~1.09 GiB |
| Max prefill activation frame, `f32` chunk256 | ~2.0 MiB |
| Decode activation frame, `f32` | 8 KiB/token/boundary |
| Summary telemetry | Tiny compared with activations |

The activation frames dominate the data path. Prompt tokens, predicted-token
replies, ACKs, and summary telemetry are all small next to the activation
traffic.

```text
Driver tokens:          .
Stage replies:          .
Telemetry summary:      .
Each boundary activations:
████████████████████████████████████████
All 3 boundaries:
████████████████████████████████████████
████████████████████████████████████████
████████████████████████████████████████
```

## Optimization Implication

The next prefill optimization should attack activation traffic or activation
handling before spending time on token/control traffic. The current wire uses
raw little-endian `f32` so transport adds no quantization error and needs no
per-family dtype policy. Compression or quantization belongs in a separately
versioned protocol experiment with a model-agnostic safety contract.
