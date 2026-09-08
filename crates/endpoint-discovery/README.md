# endpoint-discovery

Detect OpenAI-compatible LLM servers already running on the local machine
(Ollama, LM Studio, LiteLLM, vLLM, TGI, llama.cpp, Lemonade Server).

```rust,no_run
# async fn example() {
for endpoint in endpoint_discovery::discover_local_endpoints().await {
    println!("{} -> {:?}", endpoint.describe(), endpoint.models);
}
# }
```

Probes go to loopback addresses only, never traverse a configured HTTP proxy,
and use a sub-second timeout. This crate does not read or write configuration —
it reports what is listening and lets the caller decide.

Used by the `mesh-llm plugins discover` command and re-exported by
`mesh-llm-sdk` under its `endpoint-discovery` feature.
