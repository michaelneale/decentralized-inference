use std::env;
use std::path::PathBuf;

fn main() {
    let llama_build = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .join("llama.cpp/build");

    // Static library search paths
    for dir in [
        "src",
        "ggml/src",
        "ggml/src/ggml-metal",
        "ggml/src/ggml-cpu",
        "ggml/src/ggml-blas",
        "ggml/src/ggml-rpc",
    ] {
        let path = llama_build.join(dir);
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }

    // macOS frameworks (Metal backend)
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
