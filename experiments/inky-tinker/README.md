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
| `squid_overlay.py` | folds the LoRA into overlay safetensors for squid5090 (no runtime LoRA there): BF16 attention merged, NVFP4 experts of layers 0-19 stochastically rounded on the checkpoint's own grid |
| `squid_gen.py`, `squid_arms.sh` | greedy completions from squid5090 `/v1/completions` for the rendered prompt text; base / attention-only / full arms in one WSL run |

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

## The full adapter on squid5090 (RTX 5080 Laptop, 2026-11-25)

squid5090 (the Zig + CUDA Inkling engine, 5080 port in `Documents\Codex\squid5090`, local commit 149a746 adds
`SQUID_OVERLAY`) loads the 19.5 GB checkpoint and has no LoRA path, so `squid_overlay.py` folds the adapter into
the weights it loads. Attention is BF16 and merges exactly (squid then quantizes it to NVFP4 at load, with fresh
scales). The routed experts of layers 0-19 are NVFP4 already: the delta is ~7% of half a grid step, so nearest
re-rounding keeps ~1% of it; stochastic rounding on the same grid keeps it in expectation (projection 0.94) with a
fixed noise 4-9x the delta (`results/bdb-full-squid/overlay_stats.json`). Layers 20-23 (MXFP8 experts, re-quantized
by squid at load) keep base experts. Overlays: `attn.safetensors` 1.2 GB, `experts.safetensors` 10.2 GB.

Greedy, MTP off, nvfp4, ctx 8192, same 36 rendered prompts (they re-encode to the Tinker tokens 36/36):

| arm | BDB all | bonding | dlc | netting | median reply tokens | decode tok/s |
|---|---|---|---|---|---|---|
| base | 0.000 | 0.000 | 0.000 | 0.000 | 1,536 (cap, mid-thought) | 97.6 |
| attention merged | 0.271 | 0.813 | 0.000 | 0.000 | 102 | 108.8 |
| attention + experts (full fold) | 0.646 | 1.000 | 0.938 | 0.000 | 144 | 105.4 |
| Tinker-served adapter (reference) | 0.646 | 1.000 | 0.938 | 0.000 | 144 | – |

The full fold matches the Tinker-served adapter on every family, with 14 of 36 replies byte-identical to Tinker's.
Attention alone turns off the long thinking and carries bonding; dlc needs the expert deltas.

Run: `SQUID_OVERLAY=<dir>/attn.safetensors,<dir>/experts.safetensors squid5090 serve <model> --mode nvfp4 ...`
(`squid_arms.sh` does all three arms under one vramorama lease).
