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

## BDB results (Bitcoin DeFi Bench SFT rows, 2026-11-25)

Data: bonsai-gym suite `data/bdb/{train,val,eval}.jsonl` (450 / 36 / 36 rows; eval = seeds 7000-7003, 12 rows each
for bonding, dlc, netting). Scored with bitcoin-defi-bench `scripts/score_sft_replies.ts` (1 = accepted and within
1% of the teacher's value, 0.25 = accepted but worse, 0 = rejected or malformed). Greedy, thinking effort 0.

| arm | steps | held-out loss | BDB all | bonding | dlc | netting |
|---|---|---|---|---|---|---|
| base, 1,536-token budget | – | 0.400 | 0.000 | 0.000 | 0.000 | 0.000 |
| base, 6,144-token budget | – | 0.400 | 0.201 | 0.604 | 0.000 | 0.000 |
| pre-flight adapter (r16, batch 8, lr 2e-4) | 15 | 0.238 | 0.417 | 0.875 | 0.375 | 0.000 |
| full adapter (r16, batch 8, lr 2e-4, 3 epochs) | 168 | 0.218 | 0.646 | 1.000 | 0.938 | 0.000 |

- Held-out loss is flat after the first epoch (0.241 at step 14, 0.216-0.229 from step 28 on); the eval gain from
  15 to 168 steps is in dlc (0.375 to 0.938).
- The base model thinks through the budget: 21 of 36 replies hit 6,144 tokens; the adapter answers in a median
  144 tokens.
- Netting is 0 for every arm. The adapter emits a well-formed `settle` plan but fills every transfer with the
  same placeholder amount (e.g. 11111 sats), which the episode rejects as unbalanced: the format is learned, the
  net-balance arithmetic is not.
- The pre-flight GGUF LoRA on the local UD-Q4_K_M (llama.cpp b12686) scored the same as the Tinker-served
  adapter on the 13 eval rows run locally (0.827 vs 0.827, 7 of 13 replies byte-identical). The full adapter's
  GGUF (`inky-bdb-full-lora-f16.gguf`, 384 tensors, 1.27 GB) comes from the same converter.
- Tinker paths (states every 28 steps, sampler and external weights) are in `results/bdb-full/run.json`.
