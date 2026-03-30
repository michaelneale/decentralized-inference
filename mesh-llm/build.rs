use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=ui/dist");
    watch_path(Path::new("proto"));
    compile_node_proto();
}

fn watch_path(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
    let Ok(meta) = fs::metadata(path) else {
        return;
    };
    if meta.is_dir() {
        let Ok(entries) = fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            watch_path(&entry.path());
        }
    }
}

fn compile_node_proto() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc");
    std::env::set_var("PROTOC", protoc);

    prost_build::Config::new()
        .compile_protos(&["proto/node.proto"], &["proto"])
        .expect("compile node proto");
}
