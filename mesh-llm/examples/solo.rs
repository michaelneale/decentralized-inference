/// Test all Engine modes: solo, connect, auto.
///
/// Usage:
///   cargo run --release --example solo -- ~/.models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
///   cargo run --release --example solo -- --multi ~/.models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
///   cargo run --release --example solo -- --connect http://localhost:9337
///   cargo run --release --example solo -- --auto ~/.models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf

use mesh_llm::Engine;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    let engine = match args.get(1).map(|s| s.as_str()) {
        Some("--multi") => {
            let path = args.get(2).expect("need model path");
            let engine = Engine::solo(path).expect("failed to load");
            run_multi_turn(&engine).await;
            return;
        }
        Some("--connect") => {
            let url = args.get(2).expect("need URL");
            let model = args.get(3).map(|s| s.as_str());
            Engine::connect(url, model).expect("failed to connect")
        }
        Some("--auto") => {
            let path = args.get(2).expect("need model path");
            Engine::auto(path).await.expect("failed")
        }
        Some(path) => Engine::solo(path).expect("failed to load"),
        None => {
            eprintln!("Usage:");
            eprintln!("  solo <model.gguf>              — in-process inference");
            eprintln!("  solo --multi <model.gguf>       — multi-turn test");
            eprintln!("  solo --connect <url> [model]    — remote server");
            eprintln!("  solo --auto <model.gguf>        — mesh if available, else solo");
            std::process::exit(1);
        }
    };

    // Streaming test
    eprintln!("\n--- Streaming response ---");
    let mut stream = engine.chat(&[("user", "Write a haiku about distributed computing.")]);
    while let Some(token) = stream.next().await {
        eprint!("{}", token);
    }
    eprintln!("\n--- Done ---");
}

async fn run_multi_turn(engine: &Engine) {
    eprintln!("\n--- Turn 1: setting name ---");
    let reply = engine
        .chat_blocking(&[("user", "My name is Alice. Just say OK.")])
        .await
        .unwrap();
    eprintln!("Assistant: {reply}");

    eprintln!("\n--- Turn 2: recalling name ---");
    let reply = engine
        .chat_blocking(&[
            ("user", "My name is Alice. Just say OK."),
            ("assistant", &reply),
            ("user", "What is my name? Reply with just the name."),
        ])
        .await
        .unwrap();
    eprintln!("Assistant: {reply}");
}
