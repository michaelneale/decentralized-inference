fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_LINK_MODE");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_LINK_MODE");
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_BACKEND");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_BACKEND");
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=SKIPPY_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_AMDGPU_TARGETS");
    println!("cargo:rerun-if-env-changed=SKIPPY_AMDGPU_TARGETS");
    println!("cargo:rerun-if-env-changed=MESH_LLM_LLAMA_BUILD_ROOT");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_AUTO_BUILD");
    println!("cargo:rerun-if-env-changed=MESH_LLM_AUTO_BUILD_LLAMA");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=LLVMInstallDir");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    if std::env::var_os("CARGO_FEATURE_DYNAMIC_RUNTIME").is_some() {
        return;
    }

    let workspace_root = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"),
    )
    .join("../..");
    println!(
        "cargo:rerun-if-changed={}",
        workspace_root.join("scripts/build-llama.sh").display()
    );
    let target = std::env::var("TARGET").unwrap_or_default();
    let backend = std::env::var("LLAMA_STAGE_BACKEND")
        .or_else(|_| std::env::var("SKIPPY_LLAMA_BACKEND"))
        .unwrap_or_else(|_| default_backend(&target).to_string());
    let build_dir = std::env::var("LLAMA_STAGE_BUILD_DIR")
        .or_else(|_| std::env::var("SKIPPY_LLAMA_BUILD_DIR"))
        .map(std::path::PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                workspace_root.join(path)
            }
        })
        .unwrap_or_else(|_| default_build_dir(&workspace_root, &backend));
    let link_mode =
        std::env::var("LLAMA_STAGE_LINK_MODE").or_else(|_| std::env::var("SKIPPY_LLAMA_LINK_MODE"));
    if link_mode.as_deref() == Ok("dynamic") {
        if let Ok(lib_dir) =
            std::env::var("LLAMA_STAGE_LIB_DIR").or_else(|_| std::env::var("SKIPPY_LLAMA_LIB_DIR"))
        {
            println!("cargo:rustc-link-search=native={lib_dir}");
        }
        println!("cargo:rustc-link-lib=dylib=mtmd");
        println!("cargo:rustc-link-lib=dylib=llama-common");
        println!("cargo:rustc-link-lib=dylib=llama");
        if target.contains("linux") && backend == "cuda" {
            // Driverless builders use the toolkit's link-time driver stub.
            // Resolve the same library CMake selected without embedding its
            // directory in RPATH or copying it into the runtime package.
            let cmake_cache = build_dir.join("CMakeCache.txt");
            println!("cargo:rerun-if-changed={}", cmake_cache.display());
            link_linux_lib_from_cache(&cmake_cache, "CUDA_cuda_driver_LIBRARY", "cuda");
        }
        return;
    }
    ensure_static_native_ready(&workspace_root, &build_dir, &target, &backend);

    let search_dirs = [
        build_dir.join("tools/mtmd"),
        build_dir.join("vendor/hash"),
        build_dir.join("common"),
        build_dir.join("src"),
        build_dir.join("ggml/src"),
        build_dir.join("ggml/src/ggml-cpu"),
        build_dir.join("ggml/src/ggml-blas"),
        build_dir.join("ggml/src/ggml-cuda"),
        build_dir.join("ggml/src/ggml-hip"),
        build_dir.join("ggml/src/ggml-metal"),
        build_dir.join("ggml/src/ggml-vulkan"),
    ];

    for dir in search_dirs.iter().filter(|dir| dir.exists()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    let cmake_cache = build_dir.join("CMakeCache.txt");
    if cmake_cache.exists() {
        println!("cargo:rerun-if-changed={}", cmake_cache.display());
    }

    for (unix_archive, msvc_archive) in [
        ("src/libllama.a", "src/llama.lib"),
        ("tools/mtmd/libmtmd.a", "tools/mtmd/mtmd.lib"),
        (
            "vendor/hash/libvendor-hash.a",
            "vendor/hash/vendor-hash.lib",
        ),
        ("common/libllama-common.a", "common/llama-common.lib"),
        (
            "common/libllama-common-base.a",
            "common/llama-common-base.lib",
        ),
        ("ggml/src/libggml.a", "ggml/src/ggml.lib"),
        ("ggml/src/libggml-base.a", "ggml/src/ggml-base.lib"),
        (
            "ggml/src/ggml-cpu/libggml-cpu.a",
            "ggml/src/ggml-cpu/ggml-cpu.lib",
        ),
        ("ggml/src/libggml-cpu.a", "ggml/src/ggml-cpu.lib"),
        (
            "ggml/src/ggml-blas/libggml-blas.a",
            "ggml/src/ggml-blas/ggml-blas.lib",
        ),
        (
            "ggml/src/ggml-cuda/libggml-cuda.a",
            "ggml/src/ggml-cuda/ggml-cuda.lib",
        ),
        (
            "ggml/src/ggml-hip/libggml-hip.a",
            "ggml/src/ggml-hip/ggml-hip.lib",
        ),
        (
            "ggml/src/ggml-metal/libggml-metal.a",
            "ggml/src/ggml-metal/ggml-metal.lib",
        ),
        (
            "ggml/src/ggml-vulkan/libggml-vulkan.a",
            "ggml/src/ggml-vulkan/ggml-vulkan.lib",
        ),
    ] {
        for archive in [unix_archive, msvc_archive]
            .iter()
            .map(|path| build_dir.join(path))
            .filter(|archive| archive.exists())
        {
            println!("cargo:rerun-if-changed={}", archive.display());
        }
    }

    if static_archive_exists(&build_dir, "tools/mtmd/libmtmd.a", "tools/mtmd/mtmd.lib") {
        println!("cargo:rustc-link-lib=static=mtmd");
        // Upstream links mtmd against vendor::hash PRIVATE, so the archive is
        // not propagated to consumers. It must follow libmtmd.a on the link
        // line or hash_sha256_hex stays undefined. mtmd without vendor-hash is
        // not a degraded configuration to link anyway -- it is a guaranteed
        // undefined symbol, so fail here with the cause rather than emitting a
        // link line we know cannot resolve.
        if !static_archive_exists(
            &build_dir,
            "vendor/hash/libvendor-hash.a",
            "vendor/hash/vendor-hash.lib",
        ) {
            panic!(
                "libmtmd is present in {} but vendor/hash/libvendor-hash.a is not; \
                 mtmd requires vendor::hash for hash_sha256_hex. The llama.cpp \
                 build or the restored static ABI artifact is incomplete.",
                build_dir.display()
            );
        }
        println!("cargo:rustc-link-lib=static=vendor-hash");
    }
    println!("cargo:rustc-link-lib=static=llama-common");
    println!("cargo:rustc-link-lib=static=llama-common-base");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    let has_cuda = static_archive_exists(
        &build_dir,
        "ggml/src/ggml-cuda/libggml-cuda.a",
        "ggml/src/ggml-cuda/ggml-cuda.lib",
    );
    if has_cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    let has_hip = static_archive_exists(
        &build_dir,
        "ggml/src/ggml-hip/libggml-hip.a",
        "ggml/src/ggml-hip/ggml-hip.lib",
    );
    if has_hip {
        println!("cargo:rustc-link-lib=static=ggml-hip");
    }
    let has_vulkan = static_archive_exists(
        &build_dir,
        "ggml/src/ggml-vulkan/libggml-vulkan.a",
        "ggml/src/ggml-vulkan/ggml-vulkan.lib",
    );
    if has_vulkan {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
    }
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    if static_archive_exists(
        &build_dir,
        "ggml/src/ggml-blas/libggml-blas.a",
        "ggml/src/ggml-blas/ggml-blas.lib",
    ) {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    if static_archive_exists(
        &build_dir,
        "ggml/src/ggml-metal/libggml-metal.a",
        "ggml/src/ggml-metal/ggml-metal.lib",
    ) {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    println!("cargo:rustc-link-lib=static=ggml-base");

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        if static_archive_exists(
            &build_dir,
            "ggml/src/ggml-metal/libggml-metal.a",
            "ggml/src/ggml-metal/ggml-metal.lib",
        ) {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    } else if target.contains("android") {
        println!("cargo:rustc-link-lib=static=c++_static");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=log");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        for lib in linux_openmp_libs(&cmake_cache) {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
        if has_cuda {
            link_linux_cuda_libs(&cmake_cache);
        }
        if has_hip {
            link_linux_hip_libs();
        }
        if has_vulkan {
            println!("cargo:rustc-link-lib=dylib=vulkan");
        }
    } else if target.contains("windows") {
        link_windows_openmp_libs(&cmake_cache);
        if has_cuda {
            link_windows_cuda_libs(&cmake_cache);
        }
        if has_hip {
            link_windows_hip_libs();
        }
        if has_vulkan {
            link_windows_vulkan_libs();
        }
    }
}

fn default_build_dir(workspace_root: &std::path::Path, backend: &str) -> std::path::PathBuf {
    let script = workspace_root.join("scripts/build-llama.sh");
    let output = std::process::Command::new("bash")
        .arg(&script)
        .arg("--print-build-dir")
        .env("LLAMA_STAGE_BACKEND", backend)
        .env("LLAMA_STAGE_LINK_MODE", "static")
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "failed to ask {} for the canonical llama.cpp build directory: {error}",
                script.display()
            )
        });
    if !output.status.success() {
        panic!(
            "{} --print-build-dir failed: {}",
            script.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let path = String::from_utf8(output.stdout)
        .expect("build-llama.sh --print-build-dir returned non-UTF-8 output");
    let path = std::path::PathBuf::from(path.trim());
    if path.is_absolute() {
        path
    } else {
        workspace_root.join(path)
    }
}

fn default_backend(target: &str) -> &'static str {
    if target.contains("apple") {
        "metal"
    } else {
        "cpu"
    }
}

fn ensure_static_native_ready(
    workspace_root: &std::path::Path,
    build_dir: &std::path::Path,
    target: &str,
    backend: &str,
) {
    if required_static_archives_exist(build_dir) {
        return;
    }

    if !native_auto_build_enabled() {
        panic!(
            "patched llama.cpp ABI archives are missing from {}; run `just llama-build`, set LLAMA_STAGE_BUILD_DIR, or enable SKIPPY_LLAMA_AUTO_BUILD=1",
            build_dir.display()
        );
    }

    let prepare = workspace_root.join("scripts/prepare-llama.sh");
    let build = workspace_root.join("scripts/build-llama.sh");
    println!("cargo:rerun-if-changed={}", prepare.display());
    println!("cargo:rerun-if-changed={}", build.display());
    if !prepare.exists() || !build.exists() {
        panic!(
            "patched llama.cpp ABI archives are missing from {}, and mesh-llm build scripts were not found under {}; set LLAMA_STAGE_BUILD_DIR to a prepared native build",
            build_dir.display(),
            workspace_root.display()
        );
    }
    if target.contains("windows") {
        panic!(
            "patched llama.cpp ABI archives are missing from {}; automatic native preparation is not supported for Windows from build.rs yet",
            build_dir.display()
        );
    }

    eprintln!(
        "building patched llama.cpp ABI for mesh-llm SDK ({backend}) at {}",
        build_dir.display()
    );
    run_native_script(
        workspace_root,
        &prepare,
        ["pinned"].as_slice(),
        backend,
        &[
            ("LLAMA_WORKDIR", workspace_root.join(".deps/llama.cpp")),
            ("LLAMA_BUILD_DIR", build_dir.to_path_buf()),
            ("LLAMA_STAGE_BUILD_DIR", build_dir.to_path_buf()),
        ],
    );
    run_native_script(
        workspace_root,
        &build,
        [].as_slice(),
        backend,
        &[
            ("LLAMA_WORKDIR", workspace_root.join(".deps/llama.cpp")),
            ("LLAMA_BUILD_DIR", build_dir.to_path_buf()),
            ("LLAMA_STAGE_BUILD_DIR", build_dir.to_path_buf()),
        ],
    );

    if !required_static_archives_exist(build_dir) {
        panic!(
            "patched llama.cpp ABI build finished but required archives are still missing from {}",
            build_dir.display()
        );
    }
}

fn native_auto_build_enabled() -> bool {
    for key in ["SKIPPY_LLAMA_AUTO_BUILD", "MESH_LLM_AUTO_BUILD_LLAMA"] {
        if let Ok(value) = std::env::var(key) {
            return !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            );
        }
    }
    true
}

fn run_native_script(
    workspace_root: &std::path::Path,
    script: &std::path::Path,
    args: &[&str],
    backend: &str,
    paths: &[(&str, std::path::PathBuf)],
) {
    let mut command = std::process::Command::new("bash");
    command.current_dir(workspace_root).arg(script).args(args);
    for (key, value) in paths {
        command.env(key, value);
    }
    command.env("LLAMA_STAGE_BACKEND", backend);
    let status = command.status().unwrap_or_else(|error| {
        panic!("failed to run {}: {error}", script.display());
    });
    if !status.success() {
        panic!("{} failed with status {status}", script.display());
    }
}

fn required_static_archives_exist(build_dir: &std::path::Path) -> bool {
    [
        &["src/libllama.a", "src/llama.lib"][..],
        &["common/libllama-common.a", "common/llama-common.lib"],
        &[
            "common/libllama-common-base.a",
            "common/llama-common-base.lib",
        ],
        &["ggml/src/libggml.a", "ggml/src/ggml.lib"],
        &["ggml/src/libggml-base.a", "ggml/src/ggml-base.lib"],
        &[
            "ggml/src/libggml-cpu.a",
            "ggml/src/ggml-cpu.lib",
            "ggml/src/ggml-cpu/libggml-cpu.a",
            "ggml/src/ggml-cpu/ggml-cpu.lib",
        ],
    ]
    .iter()
    .all(|candidates| static_archive_exists_any(build_dir, candidates))
}

fn static_archive_exists_any(build_dir: &std::path::Path, candidates: &[&str]) -> bool {
    candidates
        .iter()
        .any(|candidate| build_dir.join(candidate).exists())
}

fn static_archive_exists(
    build_dir: &std::path::Path,
    unix_archive: &str,
    msvc_archive: &str,
) -> bool {
    build_dir.join(unix_archive).exists() || build_dir.join(msvc_archive).exists()
}

fn link_linux_cuda_libs(cmake_cache: &std::path::Path) {
    for (cache_key, lib) in [
        ("CUDA_cuda_driver_LIBRARY", "cuda"),
        ("CUDA_cudart_LIBRARY", "cudart"),
        ("CUDA_cublas_LIBRARY", "cublas"),
        ("CUDA_cublasLt_LIBRARY", "cublasLt"),
    ] {
        link_linux_lib_from_cache(cmake_cache, cache_key, lib);
    }
    // NCCL is conditionally linked by CMake when found on the system.
    // Check CMakeCache for NCCL_FOUND or NCCL_LIBRARY to detect this and extract the search path.
    if let Ok(contents) = std::fs::read_to_string(cmake_cache) {
        let mut nccl_found = cmake_cache_bool(&contents, "NCCL_FOUND");
        if let Some(nccl_path) = cmake_cache_value(&contents, "NCCL_LIBRARY")
            && !nccl_path.contains("NOTFOUND")
            && !nccl_path.contains("-NOTFOUND")
        {
            nccl_found = true;
            let path = std::path::PathBuf::from(&nccl_path);
            if let Some(parent) = path.parent()
                && parent.is_dir()
            {
                println!("cargo:rustc-link-search=native={}", parent.display());
            }
        }
        if nccl_found {
            println!("cargo:rustc-link-lib=dylib=nccl");
        }
    }
}

fn link_windows_cuda_libs(cmake_cache: &std::path::Path) {
    for path in windows_cuda_search_paths(cmake_cache) {
        if path.is_dir() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }
    for lib in ["cuda", "cudart", "cublas", "cublasLt"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn windows_cuda_search_paths(cmake_cache: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    if let Ok(cache) = std::fs::read_to_string(cmake_cache) {
        for key in [
            "CUDA_cuda_driver_LIBRARY",
            "CUDA_cudart_LIBRARY",
            "CUDA_cublas_LIBRARY",
            "CUDA_cublasLt_LIBRARY",
        ] {
            if let Some(value) = cmake_cache_value(&cache, key) {
                let path = std::path::PathBuf::from(value);
                if let Some(parent) = path.parent() {
                    push_unique_path(&mut paths, parent.to_path_buf());
                }
            }
        }
    }
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        push_unique_path(
            &mut paths,
            std::path::PathBuf::from(cuda_path).join("lib/x64"),
        );
    }
    paths
}

fn link_windows_hip_libs() {
    for env_name in ["ROCM_PATH", "HIP_PATH"] {
        if let Ok(root) = std::env::var(env_name) {
            for suffix in ["lib", "hip/lib"] {
                let path = std::path::PathBuf::from(&root).join(suffix);
                if path.is_dir() {
                    println!("cargo:rustc-link-search=native={}", path.display());
                }
            }
        }
    }
    for lib in ["amdhip64", "rocblas", "hipblas"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn link_windows_vulkan_libs() {
    if let Ok(vulkan_sdk) = std::env::var("VULKAN_SDK") {
        let lib_dir = std::path::PathBuf::from(vulkan_sdk).join("Lib");
        if lib_dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=vulkan-1");
}

fn link_windows_openmp_libs(cmake_cache: &std::path::Path) {
    let libs = openmp_libs(cmake_cache, "vcomp");
    if libs.is_empty() {
        return;
    }

    for path in windows_openmp_search_paths(cmake_cache, &libs) {
        if path.is_dir() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }

    for lib in libs {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn link_linux_hip_libs() {
    // Add ROCm library search paths
    for search_path in ["/opt/rocm/lib", "/opt/rocm/hip/lib"] {
        if std::path::Path::new(search_path).is_dir() {
            println!("cargo:rustc-link-search=native={search_path}");
        }
    }
    for lib in ["amdhip64", "rocblas", "hipblas"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    // RCCL (ROCm Collective Communications Library) provides the NCCL interface
    // for multi-GPU communication. Link it if available on the system.
    if std::path::Path::new("/opt/rocm/lib/librccl.so").exists() {
        println!("cargo:rustc-link-lib=dylib=rccl");
    }
}

fn push_unique_path(paths: &mut Vec<std::path::PathBuf>, path: std::path::PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn windows_openmp_search_paths(
    cmake_cache: &std::path::Path,
    libs: &[String],
) -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();
    if let Ok(cache) = std::fs::read_to_string(cmake_cache) {
        for lib in libs {
            for key in [
                format!("OpenMP_{lib}_LIBRARY"),
                format!("OpenMP_{lib}_LIBRARY_RELEASE"),
                format!("OpenMP_{lib}_LIBRARY_DEBUG"),
            ] {
                if let Some(value) = cmake_cache_value(&cache, &key) {
                    let path = std::path::PathBuf::from(value);
                    if let Some(parent) = path.parent() {
                        push_unique_path(&mut paths, parent.to_path_buf());
                    }
                }
            }
        }
    }

    for env_name in ["ROCM_PATH", "HIP_PATH", "LLVMInstallDir"] {
        if let Ok(root) = std::env::var(env_name) {
            for suffix in ["lib", "llvm/lib"] {
                push_unique_path(&mut paths, std::path::PathBuf::from(&root).join(suffix));
            }
        }
    }

    paths
}

fn link_linux_lib_from_cache(cmake_cache: &std::path::Path, cache_key: &str, lib: &str) {
    if let Ok(cache) = std::fs::read_to_string(cmake_cache)
        && let Some(path) = cmake_cache_value(&cache, cache_key)
    {
        let path = std::path::PathBuf::from(path);
        if path.exists()
            && let Some(parent) = path.parent()
        {
            println!("cargo:rustc-link-search=native={}", parent.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib={lib}");
}

fn linux_openmp_libs(cmake_cache: &std::path::Path) -> Vec<String> {
    openmp_libs(cmake_cache, "gomp")
}

fn openmp_libs(cmake_cache: &std::path::Path, fallback: &str) -> Vec<String> {
    let Ok(cache) = std::fs::read_to_string(cmake_cache) else {
        return Vec::new();
    };

    let mut libs = Vec::new();
    for key in ["OpenMP_C_LIB_NAMES", "OpenMP_CXX_LIB_NAMES"] {
        if let Some(value) = cmake_cache_value(&cache, key) {
            for lib in value.split(';') {
                let lib = lib.trim();
                if lib.is_empty() || lib == "NOTFOUND" || lib == "pthread" {
                    continue;
                }
                if !libs.iter().any(|existing| existing == lib) {
                    libs.push(lib.to_string());
                }
            }
        }
    }

    if libs.is_empty() && cmake_cache_bool(&cache, "GGML_OPENMP_ENABLED") {
        let fallback = if openmp_flags_reference_libomp(&cache) {
            "libomp"
        } else {
            fallback
        };
        libs.push(fallback.to_string());
    }

    libs
}

fn openmp_flags_reference_libomp(cache: &str) -> bool {
    ["OpenMP_C_FLAGS", "OpenMP_CXX_FLAGS"]
        .iter()
        .filter_map(|key| cmake_cache_value(cache, key))
        .any(|value| value.to_ascii_lowercase().contains("libomp"))
}

fn cmake_cache_value(cache: &str, key: &str) -> Option<String> {
    cache.lines().find_map(|line| {
        let (lhs, rhs) = line.split_once('=')?;
        let (name, _) = lhs.split_once(':')?;
        (name == key).then(|| rhs.to_string())
    })
}

fn cmake_cache_bool(cache: &str, key: &str) -> bool {
    cmake_cache_value(cache, key)
        .map(|value| matches!(value.as_str(), "ON" | "TRUE" | "1"))
        .unwrap_or(false)
}
