//! Model download with resume support using reqwest.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

pub struct CatalogModel {
    pub name: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub size: &'static str,
    pub description: &'static str,
    /// If set, this model has a recommended draft model for speculative decoding.
    pub draft: Option<&'static str>,
}

pub const MODEL_CATALOG: &[CatalogModel] = &[
    // ── Tiny (≤3GB VRAM) ────────────────────────────────────────────
    CatalogModel {
        name: "Qwen3-4B-Q4_K_M",
        file: "Qwen3-4B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
        size: "2.5GB",
        description: "Qwen3 starter, thinking/non-thinking modes",
        draft: Some("Qwen3-0.6B-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen2.5-3B-Instruct-Q4_K_M",
        file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        size: "2.1GB",
        description: "Small & fast general chat",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Llama-3.2-3B-Instruct-Q4_K_M",
        file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        size: "2.0GB",
        description: "Meta Llama 3.2, goose default, good tool calling",
        draft: Some("Llama-3.2-1B-Instruct-Q4_K_M"),
    },
    // ── Small (6-8GB VRAM) ──────────────────────────────────────────
    CatalogModel {
        name: "Qwen3-8B-Q4_K_M",
        file: "Qwen3-8B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
        size: "5.0GB",
        description: "Qwen3 mid-tier, strong for its size",
        draft: Some("Qwen3-0.6B-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
        file: "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        size: "4.4GB",
        description: "Code generation & completion",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Gemma-3-12B-it-Q4_K_M",
        file: "Gemma-3-12B-it-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf",
        size: "7.3GB",
        description: "Google Gemma 3 12B, punches above weight",
        draft: Some("Gemma-3-1B-it-Q4_K_M"),
    },
    CatalogModel {
        name: "Hermes-2-Pro-Mistral-7B-Q4_K_M",
        file: "Hermes-2-Pro-Mistral-7B-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Hermes-2-Pro-Mistral-7B-GGUF/resolve/main/Hermes-2-Pro-Mistral-7B-Q4_K_M.gguf",
        size: "4.4GB",
        description: "Goose default, strong tool calling for agents",
        draft: None,
    },
    // ── Medium (11-17GB VRAM) ───────────────────────────────────────
    CatalogModel {
        name: "Qwen3-14B-Q4_K_M",
        file: "Qwen3-14B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf",
        size: "9.0GB",
        description: "Qwen3 strong chat, thinking modes",
        draft: Some("Qwen3-0.6B-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen2.5-14B-Instruct-Q4_K_M",
        file: "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        size: "9.0GB",
        description: "Solid general chat",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen2.5-Coder-14B-Instruct-Q4_K_M",
        file: "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        size: "9.0GB",
        description: "Strong code gen, fills gap between 7B and 32B",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M",
        file: "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        size: "9.0GB",
        description: "DeepSeek R1 reasoning distilled into Qwen 14B",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Devstral-Small-2505-Q4_K_M",
        file: "Devstral-Small-2505-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Devstral-Small-2505-GGUF/resolve/main/Devstral-Small-2505-Q4_K_M.gguf",
        size: "14.3GB",
        description: "Mistral agentic coding, tool use",
        draft: None,
    },
    CatalogModel {
        name: "Mistral-Small-3.1-24B-Instruct-Q4_K_M",
        file: "Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF/resolve/main/Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf",
        size: "14.3GB",
        description: "Mistral general chat, good tool calling",
        draft: None,
    },
    // ── Large (20-24GB VRAM) ────────────────────────────────────────
    CatalogModel {
        name: "GLM-4.7-Flash-Q4_K_M",
        file: "GLM-4.7-Flash-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
        size: "18GB",
        description: "MoE 30B/3B active, fast inference, tool calling",
        draft: None,
    },
    CatalogModel {
        name: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M",
        file: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        size: "18.6GB",
        description: "MoE agentic coding, tool use, 30B/3B active",
        draft: Some("Qwen3-0.6B-Q4_K_M"),
    },
    CatalogModel {
        name: "GLM-4-32B-0414-Q4_K_M",
        file: "GLM-4-32B-0414-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/GLM-4-32B-0414-GGUF/resolve/main/GLM-4-32B-0414-Q4_K_M.gguf",
        size: "19.7GB",
        description: "Strong 32B, good tool calling",
        draft: None,
    },
    CatalogModel {
        name: "Qwen3-32B-Q4_K_M",
        file: "Qwen3-32B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q4_K_M.gguf",
        size: "19.8GB",
        description: "Best Qwen3 dense, thinking/non-thinking modes",
        draft: Some("Qwen3-0.6B-Q4_K_M"),
    },
    CatalogModel {
        name: "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M",
        file: "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        size: "19.9GB",
        description: "DeepSeek R1 reasoning distilled into Qwen 32B",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen2.5-32B-Instruct-Q4_K_M",
        file: "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        size: "20GB",
        description: "Proven general chat",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen2.5-Coder-32B-Instruct-Q4_K_M",
        file: "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        size: "20GB",
        description: "Top-tier code gen, matches GPT-4o on code",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Llama-4-Scout-Q4_K_M",
        file: "Llama-4-Scout-4bit-Q4_K_M.gguf",
        url: "https://huggingface.co/glogwa68/Llama-4-scout-GGUF/resolve/main/Llama-4-Scout-4bit-Q4_K_M.gguf",
        size: "22.5GB",
        description: "MoE 109B/17B active, Meta latest, tool calling",
        draft: None,
    },
    CatalogModel {
        name: "Gemma-3-27B-it-Q4_K_M",
        file: "Gemma-3-27B-it-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/google_gemma-3-27b-it-GGUF/resolve/main/google_gemma-3-27b-it-Q4_K_M.gguf",
        size: "17GB",
        description: "Google Gemma 3 27B, strong reasoning",
        draft: Some("Gemma-3-1B-it-Q4_K_M"),
    },
    // ── XL (48-53GB VRAM, tensor split candidates) ──────────────────
    CatalogModel {
        name: "Llama-3.3-70B-Instruct-Q4_K_M",
        file: "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        size: "42.5GB",
        description: "Meta Llama 3.3 70B, strong all-around",
        draft: Some("Llama-3.2-1B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen2.5-72B-Instruct-Q4_K_M",
        file: "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF/resolve/main/Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        size: "47.4GB",
        description: "Flagship Qwen2.5, great tensor split showcase",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    // ── Draft models ────────────────────────────────────────────────
    CatalogModel {
        name: "Qwen2.5-0.5B-Instruct-Q4_K_M",
        file: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        size: "491MB",
        description: "Draft for Qwen2.5 and DeepSeek-R1-Distill models",
        draft: None,
    },
    CatalogModel {
        name: "Qwen3-0.6B-Q4_K_M",
        file: "Qwen3-0.6B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf",
        size: "397MB",
        description: "Draft for Qwen3 models",
        draft: None,
    },
    CatalogModel {
        name: "Llama-3.2-1B-Instruct-Q4_K_M",
        file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        size: "760MB",
        description: "Draft for Llama 3.x and Llama 4 models",
        draft: None,
    },
    CatalogModel {
        name: "Gemma-3-1B-it-Q4_K_M",
        file: "Gemma-3-1B-it-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf",
        size: "780MB",
        description: "Draft for Gemma 3 models",
        draft: None,
    },
];

/// Get the models directory (~/.models/)
pub fn models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".models")
}

/// Find a catalog model by name (case-insensitive partial match)
/// Parse a size string like "20GB", "4.4GB", "491MB" into GB as f64.
pub fn parse_size_gb(s: &str) -> f64 {
    let s = s.trim();
    if let Some(gb) = s.strip_suffix("GB") {
        gb.trim().parse().unwrap_or(0.0)
    } else if let Some(mb) = s.strip_suffix("MB") {
        mb.trim().parse::<f64>().unwrap_or(0.0) / 1000.0
    } else {
        0.0
    }
}

pub fn find_model(query: &str) -> Option<&'static CatalogModel> {
    let q = query.to_lowercase();
    MODEL_CATALOG.iter().find(|m| m.name.to_lowercase() == q)
        .or_else(|| MODEL_CATALOG.iter().find(|m| m.name.to_lowercase().contains(&q)))
}

/// Download a model to ~/.models/ with resume support.
/// Returns the path to the downloaded file.
pub async fn download_model(model: &CatalogModel) -> Result<PathBuf> {
    let dir = models_dir();
    tokio::fs::create_dir_all(&dir).await?;
    let dest = dir.join(model.file);

    if dest.exists() {
        let size = tokio::fs::metadata(&dest).await?.len();
        if size > 1_000_000 {
            eprintln!("✅ {} already exists ({:.1}GB)", model.file, size as f64 / 1e9);
            return Ok(dest);
        }
    }

    eprintln!("📥 Downloading {} ({})...", model.name, model.size);
    download_with_resume(&dest, model.url).await?;
    eprintln!("✅ Downloaded to {}", dest.display());
    Ok(dest)
}

/// Download any URL to a destination path with resume support.
pub async fn download_url(url: &str, dest: &Path) -> Result<()> {
    download_with_resume(dest, url).await
}

/// Download with resume support and retries using reqwest.
async fn download_with_resume(dest: &Path, url: &str) -> Result<()> {
    use tokio_stream::StreamExt;

    let tmp = dest.with_extension("gguf.part");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600)) // 1h overall timeout
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?;

    for attempt in 1..=5 {
        // Check how much we already have (for resume)
        let existing_bytes = if tmp.exists() {
            tokio::fs::metadata(&tmp).await?.len()
        } else {
            0
        };

        eprintln!("  attempt {attempt}/5{}...",
            if existing_bytes > 0 { format!(" (resuming from {:.1}MB)", existing_bytes as f64 / 1e6) } else { String::new() });

        let mut request = client.get(url);
        if existing_bytes > 0 {
            request = request.header("Range", format!("bytes={existing_bytes}-"));
        }

        let response = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  connection failed: {e}");
                if attempt < 5 {
                    eprintln!("  retrying in 3s...");
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                }
                continue;
            }
        };

        let status = response.status();
        if !status.is_success() && status != reqwest::StatusCode::PARTIAL_CONTENT {
            // If server doesn't support resume (416 Range Not Satisfiable), start fresh
            if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                let _ = tokio::fs::remove_file(&tmp).await;
                eprintln!("  server rejected resume, starting fresh...");
                continue;
            }
            anyhow::bail!("HTTP {status} downloading {url}");
        }

        // Total size from Content-Length (or Content-Range)
        let total_bytes = if status == reqwest::StatusCode::PARTIAL_CONTENT {
            // Content-Range: bytes 1234-5678/9999
            response.headers().get("content-range")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.rsplit('/').next())
                .and_then(|s| s.parse::<u64>().ok())
        } else {
            response.content_length().map(|cl| cl + existing_bytes)
        };

        // Open file for append (resume) or create
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&tmp)
            .await
            .context("Failed to open temp file")?;

        let mut stream = response.bytes_stream();
        let mut downloaded = existing_bytes;
        let mut last_progress = std::time::Instant::now();

        // Print initial progress
        print_progress(downloaded, total_bytes);

        loop {
            match stream.next().await {
                Some(Ok(chunk)) => {
                    file.write_all(&chunk).await.context("Failed to write chunk")?;
                    downloaded += chunk.len() as u64;

                    // Update progress every 500ms
                    if last_progress.elapsed() >= std::time::Duration::from_millis(500) {
                        print_progress(downloaded, total_bytes);
                        last_progress = std::time::Instant::now();
                    }
                }
                Some(Err(e)) => {
                    file.flush().await.ok();
                    eprint!("\r");
                    eprintln!("  download interrupted at {:.1}MB: {e}",
                        downloaded as f64 / 1e6);
                    if attempt < 5 {
                        eprintln!("  retrying in 3s (will resume)...");
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    }
                    break;
                }
                None => {
                    // Stream complete
                    file.flush().await?;
                    eprint!("\r");
                    print_progress(downloaded, total_bytes);
                    eprintln!();
                    tokio::fs::rename(&tmp, dest).await
                        .context("Failed to move downloaded file")?;
                    return Ok(());
                }
            }
        }
    }

    // Clean up partial on total failure
    let _ = tokio::fs::remove_file(&tmp).await;
    anyhow::bail!("Download failed after 5 attempts");
}

fn print_progress(downloaded: u64, total: Option<u64>) {
    if let Some(total) = total {
        let pct = (downloaded as f64 / total as f64) * 100.0;
        let downloaded_mb = downloaded as f64 / 1e6;
        let total_mb = total as f64 / 1e6;
        eprint!("\r  {:.1}/{:.1}MB ({:.1}%)", downloaded_mb, total_mb, pct);
    } else {
        eprint!("\r  {:.1}MB", downloaded as f64 / 1e6);
    }
}

/// List available models
pub fn list_models() {
    eprintln!("Available models:");
    eprintln!();
    for m in MODEL_CATALOG {
        let draft_info = if let Some(d) = m.draft {
            format!(" (draft: {})", d)
        } else {
            String::new()
        };
        eprintln!("  {:40} {:>6}  {}{}", m.name, m.size, m.description, draft_info);
    }
}
