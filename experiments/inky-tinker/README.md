# inky-tinker: Tinker LoRA on Inkling 2 Mini, served locally from a Q4 GGUF

Trains a LoRA on `thinkingmachines/Inkling-2-Mini` through Thinking Machines' Tinker API (free tier model
`thinkingmachines/Inkling-2-Mini:peft:65536:free`) from the bonsai-gym suite's JSONL rows, then converts the adapter
to a llama.cpp GGUF LoRA for unsloth's mainline-layout `Inkling-2-Mini-UD-Q4_K_M.gguf` on upstream llama.cpp.

| file | what it does |
|---|---|
| `tinker_env.py` | loads the API key from a one-line file into the process environment (never printed) |
| `tinker_sft.py` | SFT loop on Tinker: tml_v0 renderer at thinking effort 0, loss on the last assistant message, held-out loss, save_state every N steps, adapter download, greedy samples from base and adapter |
| `tinker_lora_to_gguf.py` | PEFT adapter (from `tinker_cookbook.weights.build_lora_adapter`) to a GGUF LoRA for `inkling2_loop`: attention 1:1, 128 experts stacked into 3D tensors, gate/up fused exactly as a rank-2r block-diagonal LoRA |
| `local_gen.py` | greedy completions from a local llama-server for pre-tokenized prompts (the renderer's exact tokens) |

## Runtime notes (2026-11-25)

- `tml-renderers` (needed by the Inkling renderer) ships Linux/macOS wheels only, so the Tinker side runs in WSL
  (`/root/inky-tinker/venv`: tinker 0.40.0, tinker-cookbook 0.7.1, tml-renderers 0.4.2, torch 2.15 CPU).
- `build_lora_adapter` reads base tensor names and shapes from safetensors headers; give it a directory of
  header-only sparse stub shards (HTTP range requests for the header, `truncate` to full size) instead of the
  19.5 GB checkpoint.
- Local serving needs upstream llama.cpp b12611 or later (inkling2_loop, per-expert NVFP4 scales in looped
  layers); the PrismML fork b12178 has no inkling2 architecture.
