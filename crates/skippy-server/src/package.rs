pub use skippy_runtime::package::is_hf_package_ref;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_hf_package_refs() {
        assert!(is_hf_package_ref("hf://Mesh-LLM/Qwen3.6-package"));
        assert!(!is_hf_package_ref("/tmp/package"));
    }
}
