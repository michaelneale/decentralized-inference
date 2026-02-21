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
    // --- Small (single machine) ---
    CatalogModel {
        name: "Qwen2.5-3B-Instruct-Q4_K_M",
        file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        size: "2GB",
        description: "Small & fast general chat",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
        file: "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        size: "4.7GB",
        description: "Code generation & completion",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen2.5-14B-Instruct-Q4_K_M",
        file: "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        size: "9GB",
        description: "Strong general chat",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    // --- Medium (1-2 machines, good for distributed) ---
    CatalogModel {
        name: "Qwen2.5-32B-Instruct-Q4_K_M",
        file: "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        size: "20GB",
        description: "Strong general chat, good for distributed",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen2.5-Coder-32B-Instruct-Q4_K_M",
        file: "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        size: "20GB",
        description: "Top-tier code generation, matches GPT-4o on code",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Qwen3-32B-Q4_K_M",
        file: "Qwen3-32B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q4_K_M.gguf",
        size: "20GB",
        description: "Latest Qwen3, thinking/non-thinking modes",
        draft: Some("Qwen3-0.6B-Q4_K_M"),
    },
    CatalogModel {
        name: "Gemma-3-27B-it-Q4_K_M",
        file: "Gemma-3-27B-it-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/google_gemma-3-27b-it-GGUF/resolve/main/google_gemma-3-27b-it-Q4_K_M.gguf",
        size: "17GB",
        description: "Google Gemma 3 27B, strong reasoning",
        draft: Some("Gemma-3-1B-it-Q4_K_M"),
    },
    CatalogModel {
        name: "GLM-4.7-Flash-Q4_K_M",
        file: "GLM-4.7-Flash-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
        size: "17GB",
        description: "General chat with reasoning (MoE, no draft available)",
        draft: None,
    },
    // --- Large (2-3 machines) ---
    CatalogModel {
        name: "Qwen2.5-72B-Instruct-Q4_K_M",
        file: "Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF/resolve/main/Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        size: "47GB",
        description: "Flagship Qwen2.5, needs 2+ machines",
        draft: Some("Qwen2.5-0.5B-Instruct-Q4_K_M"),
    },
    CatalogModel {
        name: "Llama-3.3-70B-Instruct-Q4_K_M",
        file: "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        size: "43GB",
        description: "Meta Llama 3.3 70B, strong all-around, needs 2+ machines",
        draft: Some("Llama-3.2-1B-Instruct-Q4_K_M"),
    },
    // --- Draft models ---
    CatalogModel {
        name: "Qwen2.5-0.5B-Instruct-Q4_K_M",
        file: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        size: "491MB",
        description: "Draft model for all Qwen2.5 models",
        draft: None,
    },
    CatalogModel {
        name: "Qwen3-0.6B-Q4_K_M",
        file: "Qwen3-0.6B-Q4_K_M.gguf",
        url: "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf",
        size: "397MB",
        description: "Draft model for Qwen3 models",
        draft: None,
    },
    CatalogModel {
        name: "Llama-3.2-1B-Instruct-Q4_K_M",
        file: "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        size: "760MB",
        description: "Draft model for Llama 3.x models",
        draft: None,
    },
    CatalogModel {
        name: "Gemma-3-1B-it-Q4_K_M",
        file: "Gemma-3-1B-it-Q4_K_M.gguf",
        url: "https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf",
        size: "780MB",
        description: "Draft model for Gemma 3 models",
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
