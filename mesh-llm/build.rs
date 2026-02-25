use std::fs;
use std::path::Path;

fn main() {
    watch_path(Path::new("ui/dist"));
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
