//! Model download with resume support.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

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
        // Check if it looks complete (at least some reasonable size)
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

/// Download with resume support, retrying up to 5 times.
async fn download_with_resume(dest: &Path, url: &str) -> Result<()> {
    let tmp = dest.with_extension("gguf.part");

    for attempt in 1..=5 {
        let args = vec![
            "-L".to_string(),
            "-C".to_string(), "-".to_string(),  // resume
            "-o".to_string(), tmp.to_string_lossy().to_string(),
            "--progress-bar".to_string(),
            "--retry".to_string(), "3".to_string(),
            "--retry-delay".to_string(), "5".to_string(),
            url.to_string(),
        ];

        eprintln!("  attempt {attempt}/5...");
        let status = tokio::process::Command::new("curl")
            .args(&args)
            .stdout(std::process::Stdio::inherit())
            .stderr(std::process::Stdio::inherit())
            .status()
            .await
            .context("Failed to run curl")?;

        if status.success() {
            tokio::fs::rename(&tmp, dest).await
                .context("Failed to move downloaded file")?;
            return Ok(());
        }

        if attempt < 5 {
            eprintln!("  download interrupted, retrying in 3s...");
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }
    }

    // Clean up partial
    let _ = tokio::fs::remove_file(&tmp).await;
    anyhow::bail!("Download failed after 5 attempts");
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
