use super::normalization::RuntimeSurface;

pub fn runtime_surface_help(surface: RuntimeSurface) -> String {
    match surface {
        RuntimeSurface::Share => concat!(
            "Share an already-running OpenAI-compatible HTTP server through Mesh.\n\n",
            "Usage: mesh-llm share <URL> [OPTIONS]\n\n",
            "Example: mesh-llm share http://localhost:11434\n\n",
            "      --join <TOKEN>          Join an existing mesh\n",
            "      --publish               Publish this mesh for discovery\n",
            "      --port <PORT>           Local OpenAI proxy port [default: 9337]\n",
            "      --console <PORT>        Management API port [default: 3131]\n\n",
            "Stays running until stopped. Ctrl-C stops sharing, not the upstream server.\n",
            "No model downloads, native inference runtime, plugin process, or config edits.\n",
            "HTTP upstreams only; HTTPS and upstream authentication are not supported yet.\n"
        ).to_string(),
        RuntimeSurface::Serve => concat!(
            "Serve local models and join or publish a mesh.\n\n",
            "Usage: mesh-llm serve [OPTIONS]\n\n",
            "Common serving options:\n",
            "      --model <MODEL>          Startup model to serve from the catalog, a path, or a Hugging Face ref\n",
            "      --gguf <GGUF>            Raw local GGUF file to serve directly\n",
            "      --mmproj <MMPROJ>        Multimodal projector for the primary served model\n",
            "      --quant <RECIPE>         Quantize SafeTensors weights while loading\n",
            "                               Use --help-advanced to list every valid recipe\n",
            "      --auto                   Auto-join the best discovered mesh\n",
            "      --join <JOIN>            Join a mesh via invite token\n",
            "      --publish                Publish this mesh for discovery\n",
            "      --local-model-only       Serve one local model without mesh networking or management APIs\n",
            "      --port <PORT>            OpenAI-compatible API port [default: 9337]\n",
            "      --console <CONSOLE>      Management console/API port [default: 3131]\n",
            "      --log-format <FORMAT>    Terminal output format [default: pretty]\n\n",
            "Bare `mesh-llm serve` loads startup models from ~/.mesh-llm/config.toml.\n",
            "Add [[models]] there or pass --model / --gguf explicitly.\n",
            "With --local-model-only, --model, --gguf, and --mmproj must be absolute, non-symlink paths.\n",
            "Run `mesh-llm --help-advanced` for the full runtime option surface.\n"
        )
        .to_string(),
        RuntimeSurface::Client => concat!(
            "Run as a client-only mesh node with no local model required.\n\n",
            "Usage: mesh-llm client [OPTIONS]\n\n",
            "Common client options:\n",
            "      --auto                   Auto-join the best discovered mesh\n",
            "      --discover [NAME]        Discover and join a mesh by name\n",
            "      --join <JOIN>            Join a mesh via invite token\n",
            "      --port <PORT>            Local OpenAI-compatible proxy port [default: 9337]\n",
            "      --console <CONSOLE>      Management console/API port [default: 3131]\n",
            "      --log-format <FORMAT>    Terminal output format [default: pretty]\n\n",
            "Run `mesh-llm --help-advanced` for the full runtime option surface.\n"
        )
        .to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_surface_help_describes_serving_options() {
        let help = runtime_surface_help(RuntimeSurface::Serve);

        assert!(help.contains("Usage: mesh-llm serve"));
        assert!(help.contains("--model"));
        assert!(help.contains("--gguf"));
        assert!(help.contains("--local-model-only"));
        assert!(help.contains("--quant"));
        assert!(help.contains("startup models"));
        assert!(!help.contains("Pool GPUs over the internet for LLM inference\n\nUsage: mesh-llm"));
    }

    #[test]
    fn client_surface_help_describes_client_options() {
        let help = runtime_surface_help(RuntimeSurface::Client);

        assert!(help.contains("Usage: mesh-llm client"));
        assert!(help.contains("--auto"));
        assert!(help.contains("--discover"));
        assert!(help.contains("client-only"));
        assert!(!help.contains("--model <MODEL>"));
    }
}
