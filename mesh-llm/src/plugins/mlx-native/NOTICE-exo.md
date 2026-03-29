This crate contains logic adapted from the exo project:

- Project: https://github.com/exo-explore/exo
- Copyright: Exo Technologies Ltd
- License: Apache License 2.0

At the time of porting, the main reference files were:

- `src/exo/worker/engines/mlx/utils_mlx.py`
- `src/exo/worker/engines/mlx/generator/generate.py`
- `src/exo/worker/engines/mlx/generator/batch_generate.py`

The adapted logic is limited to MLX prompt/token handling patterns such as:

- repairing unmatched thinking end tokens in encoded prompts
- token-aware decode/stream handling for Qwen-family models
- family-specific stop-token behavior

The upstream exo repository did not include a `NOTICE` file when this was
ported. This file records the attribution and provenance for the adapted
logic inside `mesh-llm-mlx`.
