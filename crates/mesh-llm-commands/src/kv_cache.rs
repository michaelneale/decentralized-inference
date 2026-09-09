use std::io::{self, IsTerminal, Write};

use anyhow::{Context, Result, bail};
use mesh_llm_cli::KvCacheCommand;
use serde_json::{Value, json};

const KV_CACHE_HTTP_TIMEOUT_SECS: u64 = 60;

pub async fn dispatch_kv_cache_command(command: &KvCacheCommand) -> Result<()> {
    match command {
        KvCacheCommand::Status {
            endpoints,
            port,
            json,
        } => {
            let value = if endpoints.is_empty() {
                request(*port, reqwest::Method::GET, "/api/runtime/kv-cache", None).await?
            } else {
                remote_request(*port, endpoints, "status", None, None).await?
            };
            print_response(&value, *json)
        }
        KvCacheCommand::Prune {
            target,
            model_identity,
            yes,
            endpoints,
            port,
            json: json_output,
        } => {
            confirm_destructive("prune inactive disk prompt-cache entries", *yes)?;
            let target_bytes = target
                .as_deref()
                .map(mesh_llm_config::parse_iec_size)
                .transpose()
                .context("invalid --target")?;
            let body = json!({
                "target_bytes": target_bytes,
                "model_identity": model_identity,
            });
            let value = if endpoints.is_empty() {
                request(
                    *port,
                    reqwest::Method::POST,
                    "/api/runtime/kv-cache/prune",
                    Some(body),
                )
                .await?
            } else {
                remote_request(
                    *port,
                    endpoints,
                    "prune",
                    target_bytes,
                    model_identity.clone(),
                )
                .await?
            };
            print_response(&value, *json_output)
        }
        KvCacheCommand::Clear {
            model_identity,
            yes,
            endpoints,
            port,
            json: json_output,
        } => {
            confirm_destructive("clear inactive disk prompt-cache entries", *yes)?;
            let value = if endpoints.is_empty() {
                request(
                    *port,
                    reqwest::Method::DELETE,
                    "/api/runtime/kv-cache",
                    Some(json!({ "model_identity": model_identity })),
                )
                .await?
            } else {
                remote_request(*port, endpoints, "clear", None, model_identity.clone()).await?
            };
            print_response(&value, *json_output)
        }
    }
}

async fn remote_request(
    port: u16,
    endpoints: &[String],
    operation: &str,
    target_bytes: Option<u64>,
    model_identity: Option<String>,
) -> Result<Value> {
    request(
        port,
        reqwest::Method::POST,
        "/api/runtime/control/kv-cache",
        Some(json!({
            "endpoints": endpoints,
            "operation": operation,
            "target_bytes": target_bytes,
            "model_identity": model_identity,
        })),
    )
    .await
}

async fn request(
    port: u16,
    method: reqwest::Method,
    path: &str,
    body: Option<Value>,
) -> Result<Value> {
    let client = reqwest::Client::builder()
        // The runtime bounds a remote batch at 45 seconds and returns one
        // result per requested endpoint, including explicit timeout receipts.
        // Leave transport and JSON-decoding margin outside that server bound.
        .timeout(std::time::Duration::from_secs(KV_CACHE_HTTP_TIMEOUT_SECS))
        .build()?;
    let url = format!("http://127.0.0.1:{port}{path}");
    let mut request = client.request(method, &url);
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("request disk prompt-cache endpoint {url}"))?;
    let status = response.status();
    let value = response
        .json::<Value>()
        .await
        .with_context(|| format!("decode disk prompt-cache response from {url}"))?;
    if !status.is_success() {
        bail!(
            "disk prompt-cache request failed ({status}): {}",
            value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown error")
        );
    }
    Ok(value)
}

fn confirm_destructive(action: &str, yes: bool) -> Result<()> {
    if yes {
        return Ok(());
    }
    if !io::stdin().is_terminal() {
        bail!("{action} requires --yes when stdin is not interactive");
    }
    eprint!("{action}? [y/N] ");
    io::stderr().flush()?;
    let mut response = String::new();
    io::stdin().read_line(&mut response)?;
    if !matches!(response.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        bail!("operation cancelled");
    }
    Ok(())
}

fn print_response(value: &Value, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string(value)?);
        return Ok(());
    }
    if let Some(results) = value.get("results").and_then(Value::as_array) {
        for result in results {
            let node = result
                .get("target_node_id")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            if let Some(error) = result.get("error").filter(|value| !value.is_null()) {
                println!(
                    "Node {node}: error: {}",
                    error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                );
            } else if let Some(freed) = result.get("freed_bytes").and_then(Value::as_u64) {
                println!("Node {node}: freed {freed} bytes");
            } else {
                let state = result
                    .get("status")
                    .and_then(|status| status.get("effective"))
                    .and_then(|effective| effective.get("state"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                println!("Node {node}: {state}");
            }
        }
        return Ok(());
    }
    if let Some(freed) = value.get("freed_bytes").and_then(Value::as_u64) {
        println!("Freed {freed} bytes");
        return Ok(());
    }
    let configured = &value["configured"];
    let effective = &value["effective"];
    println!(
        "Disk prompt cache: {} ({})",
        effective["state"].as_str().unwrap_or("unknown"),
        configured["mode"].as_str().unwrap_or("unknown")
    );
    println!(
        "Root: {}",
        configured["directory"].as_str().unwrap_or("unknown")
    );
    if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
        println!(
            "Used: {} / {} bytes",
            usage["used_bytes"].as_u64().unwrap_or(0),
            usage["budget_bytes"].as_u64().unwrap_or(0)
        );
    }
    Ok(())
}
