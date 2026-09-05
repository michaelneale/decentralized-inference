//! Content identity for a served GGUF: SHA-256 over the file BYTES actually
//! loaded, as opposed to `model_identity::identity_hash_for`'s hash of a
//! *reference string* (repo/revision/file for a Hugging Face source, or
//! nothing at all for a local path -- see `ServedModelIdentity::identity_hash`).
//! A served-model NAME is not proof of served BYTES: a proxy or a
//! mis-deployed node can serve a different file under the same name. This
//! digest makes the served bytes a fact any peer can check against the file
//! on disk. It does not stop a host from reporting a digest for a file it did
//! not load: this is a self-reported value, so it surfaces an honest node's
//! stale or swapped file, not a host that lies about what it loaded.
//!
//! Cached by (path, size, mtime): hashing an 8GB GGUF costs real wall-clock
//! time, and must happen once per file, never once per request.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, UNIX_EPOCH};

/// (path, size in bytes, mtime as nanos since the epoch) -- the same recipe
/// `model-hf`'s local-GGUF synthetic ref already uses to detect "this exact
/// file state," just applied to a cache key instead of a name.
type CacheKey = (PathBuf, u64, u128);

fn cache() -> &'static Mutex<HashMap<CacheKey, String>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, String>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// SHA-256 of `path`'s bytes, lowercase hex. `None` when the file cannot be
/// stat'd or read -- an honest absent fact, never a fabricated value (never a
/// `0`-repeat placeholder). A second call for the same (path, size, mtime)
/// returns the cached digest without re-reading the file.
pub(crate) fn weights_digest_for_file(path: &Path) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    let size = metadata.len();
    let mtime_nanos = metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let key: CacheKey = (path.to_path_buf(), size, mtime_nanos);

    if let Some(digest) = cache()
        .lock()
        .expect("weights digest cache poisoned")
        .get(&key)
    {
        tracing::debug!(
            path = %path.display(),
            "weights_digest cache hit -- not re-hashed"
        );
        return Some(digest.clone());
    }

    let started = Instant::now();
    let digest = hash_file_bytes(path)?;
    tracing::info!(
        path = %path.display(),
        bytes = size,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "computed weights_digest for served GGUF (one-time cost for this file)"
    );
    cache()
        .lock()
        .expect("weights digest cache poisoned")
        .insert(key, digest.clone());
    Some(digest)
}

fn hash_file_bytes(path: &Path) -> Option<String> {
    let mut file = File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, contents: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "weights-digest-test-{}-{}",
            std::process::id(),
            name
        ));
        std::fs::create_dir_all(&dir).expect("mk temp dir");
        let path = dir.join("model.gguf");
        std::fs::write(&path, contents).expect("write temp file");
        path
    }

    /// The digest is a real SHA-256 of the bytes on disk -- recomputing it
    /// independently must agree exactly.
    #[test]
    fn digest_matches_independent_sha256_of_the_same_bytes() {
        let path = temp_file("matches", b"gguf-bytes-under-test");
        let digest = weights_digest_for_file(&path).expect("digest computed");

        let mut hasher = Sha256::new();
        hasher.update(b"gguf-bytes-under-test");
        let expected = hex::encode(hasher.finalize());

        assert_eq!(digest, expected);
        assert_eq!(digest.len(), 64);
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A different quantization/file swapped in under the same path AND the
    /// same size changes the digest as soon as mtime moves -- the digest
    /// records the swap (it does not, by itself, prove which bytes actually
    /// ran).
    #[test]
    fn swapping_the_file_contents_changes_the_digest() {
        let path = temp_file("swap", b"quant-a-bytes-000000");
        let before = weights_digest_for_file(&path).expect("first digest");

        // Same length, different bytes, and force mtime forward so the cache
        // key changes -- otherwise a same-second rewrite could alias the
        // prior (path, size, mtime) key, which is the documented limitation
        // of this cache, not the case under test here.
        std::fs::write(&path, b"quant-b-bytes-111111").expect("rewrite file");
        let future = std::time::SystemTime::now() + std::time::Duration::from_secs(2);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("reopen for mtime bump");
        file.set_modified(future).expect("bump mtime");

        let after = weights_digest_for_file(&path).expect("second digest");
        assert_ne!(before, after, "swapped bytes must change the digest");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A second call for the SAME (path, size, mtime) is a cache hit: the
    /// digest is identical, and (implicitly) the file is not re-read -- the
    /// case above proves the cache key can change; this proves an unchanged
    /// key does not silently drift.
    #[test]
    fn second_call_for_unchanged_file_returns_the_same_cached_digest() {
        let path = temp_file("cache-hit", b"stable-bytes");
        let first = weights_digest_for_file(&path).expect("first digest");
        let second = weights_digest_for_file(&path).expect("second digest");
        assert_eq!(first, second);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A file that does not exist -- or cannot be stat'd -- yields `None`,
    /// never a fabricated digest.
    #[test]
    fn unreadable_file_yields_none_never_a_fabricated_digest() {
        let path = std::env::temp_dir().join(format!(
            "weights-digest-test-missing-{}.gguf",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        assert!(weights_digest_for_file(&path).is_none());
    }
}
