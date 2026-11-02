# Low-bit adapter audit (P0)

What our earlier "1-bit adapter" work actually did, where it lives, what the
"15–20% gap to QLoRA" figure is, and what carries over to Bonsai 2 27B.

## 1. The two prior lines of work

### A. `q1-adapters`: task LoRAs for a 1-bit 8B model

- Repo: `github.com/patrickdugan/q1-adapters`, local `OneDrive\Documents\GitHub\q1-adapters`, HEAD `5b7a575` (paper draft 2, 2026-10-19).
- Paper: *One gigabyte, many skills: task adapters on a 1-bit 8B model, scored on Prime Intellect environments* (`paper/q1_adapters.md`, PDF beside it). Adapters on Hugging Face: `AlephFunk/bonsai-8b-prime-env-adapters`.
- Base: PrismML **Bonsai 8B** `Q1_0` (1 sign bit + FP16 scale per 128 weights, Qwen3-8B dense architecture).
- **Training method: ordinary QLoRA, not a 1-bit method.** `harness/train_lora.py` loads `prism-ml/Bonsai-8B-unpacked` (revision `376f3815`, a bf16 expansion of the 1-bit weights) in bitsandbytes 4-bit NF4 with double quantization, attaches PEFT LoRA (rank 16, alpha 32, dropout 0.05, all 7 projections, 43.6M params), trains with `paged_adamw_8bit`, cosine schedule, gradient checkpointing, bf16 compute, then converts the adapter with upstream `convert_lora_to_gguf.py` (f16, 87 MB). The adapter is served unmerged on the `Q1_0` GGUF by the PrismML llama.cpp fork (`--lora-init-without-apply`, switched per request).
- Why it transfers: the unpacked checkpoint is an exact expansion of the 1-bit release, and earlier work (stego-bench) found byte-identical generations between the torch 4-bit path and the 1-bit server. No straight-through estimator, no quantization-aware training: the base is frozen and the adapter is full precision.
- Training cost, RTX 5080 Laptop: reverse-text 7 min (150 steps, 800 gold rows, effective batch 16, lr 2e-4, 3 epochs); the job "sat at the VRAM limit" (paper §3.2).

### B. `BitAgent/experiments/ternary-lora-5080`: LoRA through packed ternary weights

- Report `TERNARY_LORA_5080_RESULTS.md` (done 2026-10-15). Bulky files in `C:\Users\patri\bitnet-lora-5080`.
- Method: QVAC `llama-finetune-lora` on **Vulkan**, LoRA rank 8 (attention only) with gradients through **packed `TQ2_0` / `TQ1_0`** weights kept on the device; no 4-bit QLoRA and no simulated ternarisation. Needed a new Vulkan `OUT_PROD` kernel for `TQ1_0` (patch 0003) and three BitNet-2B4T converter fixes.
- Models: BitNet XL 1.3B, BitNet 2B4T, Falcon3-1.58bit 3B/7B/10B. 10.3B trained at 4.8 GB (ctx 128) and 9.7 GB (ctx 512, survived two GPU resets via checkpoint + auto-resume).
- Data: a synthetic rule-based instruction set; held-out PPL fell 25–81 % and greedy EM rose to 27–57 %. No real-task evaluation.
- Limits recorded there: no grad accumulation or checkpointing, F32 adapters, no gradient clipping, a 2 s WDDM TDR risk under the laptop's recurring 15 W power cap, and no support for PrismML's `Q2_0`/`PQ2_0`/`PTQ1_0` types.

## 2. The "15–20 % gap relative to QLoRA"

No experiment in either line compares a 1-bit-trained adapter against a QLoRA-trained adapter; in line A the adapters *are* QLoRA. The figure matches the **base-model gap between the 1-bit Bonsai 8B and a 4-bit Qwen3-8B** in the q1-adapters paper (n = 200 per env, non-thinking, `results/tables.md`):

| | reverse-text | gsm8k | medmcqa | ifeval | macro (4) | mean of the 3 knowledge tasks |
|---|---|---|---|---|---|---|
| Q1 Bonsai 8B base | 0.115 | 0.845 | 0.375 | 0.640 | 0.494 | 0.620 |
| Q1 + our QLoRA-trained adapter | 0.817 | 0.875 | 0.440 | 0.640 | 0.693 | 0.652 |
| Q4_K_M Qwen3-8B base (exact weights) | 0.208 | 0.940 | 0.560 | 0.685 | 0.598 | 0.728 |
| Q1 base relative to Q4 base | | −10.1 % | −33.0 % | −6.6 % | **−17.4 %** | **−14.9 %** |
| Q1 + adapter relative to Q4 base | | −6.9 % | −21.4 % | −6.6 % | +15.9 % | −10.5 % |

So the 15–20 % is a property of the 1-bit *base* against a 4-bit base of the same architecture, on the tasks the base already knows; the adapters narrow it to ~10 % on those tasks and reverse it on transformation skills. It is not a measurement of adapter quality versus QLoRA, and it says nothing about Bonsai 2 27B.

## 3. Checklist from the brief

| Item | Finding |
|---|---|
| Packed TQ1_0/TQ2_0, QVAC | Line B. QVAC Vulkan trainer, TQ2_0 prebuilt, TQ1_0 needs our patch. Llama/BitNet graphs only. |
| Frozen quantized weights + trainable adapters | Both lines. A: NF4-requantized bf16 expansion; B: packed ternary on device. |
| Straight-through estimators / QAT gradients | **Not used anywhere.** Both freeze the base; adapters are F32/bf16. This gym adds an optional `adapter_quant: ternary_ste` (absmean ternary + STE) as an untested arm (§NEXT_STEPS). |
| Vulkan vs CUDA | A: CUDA (bitsandbytes). B: Vulkan (QVAC). This gym: CUDA via PyTorch. |
| Export / merging / inference | A: GGUF LoRA, served unmerged (1-bit weights cannot absorb a merge; an active adapter roughly halves served throughput). B: GGUF LoRA, reload checked. This gym: safetensors + a GGUF LoRA writer (`src/bonsai_gym/adapter_io.py`). |
| Reusable code | A: `train_lora.py` data rules (chat template + completion + `<|im_end|>`, loss on completion), the reverse-text gold rows and 200 eval items, the LCS scorer. B: watchdog/checkpoint-resume ideas. HybridTRMLDT S0 (`research/s0-real-row-parity-20261031` @ `1688515`): bit-exact PTQ1_0 codec, GGUF reader, projection parity, logit parity and the 32K inference baseline. |

## 4. Smallest relevant reproduction

Line A's reverse-text adapter is the closest comparator, so the canary trains on **the same 800 gold rows** (736 train / 64 val) and is scored on **the same 200 eval items** with a scorer that reproduces all 400 stored q1-adapters scores exactly (`datasets/canary/reverse_text/manifest.json → scorer_check`). The 8B QLoRA reference curve is `q1-adapters/runs/train/reverse-text.json` (loss 5.16 at step 5, effective batch 16). Results: `docs/EXPERIMENT_LOG.md`.

## 5. Compatibility verdict for Bonsai 2 27B

Checkpoint: `prism-ml/Ternary-Bonsai-2-27B-gguf` @ `94dd51cd`, `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5,946,648,928 B, sha256 `53107f53…`), Apache-2.0, derived from Qwen3.8-27B (GGUF arch `qwen35`): 64 layers (48 gated-delta-net + 16 gated full attention), width 5120, FFN 17408, vocab 248,320, 262K context. All 401 matmul weights (including embeddings and LM head) are PTQ1_0 ternary (28 B per 128 weights) in a Hadamard-rotated input basis.

- **Line A does not transfer.** There is no unpacked Bonsai 2 27B (the repo ships `F16`, `Q2_0`, `PQ2_0`, `PTQ1_0`; an F16 load is 54 GB, beyond 32 GB RAM and 16 GB VRAM), and NF4-requantizing the ternary weights would not be the served model.
- **Line B does not transfer.** QVAC has no `qwen35` graph (no gated delta net), no PrismML quant types and no Hadamard folding.
- **What works: a PyTorch frozen-base forward that keeps the PTQ1_0 bytes resident on the GPU** (5.88 GB) and decodes slices on the fly in forward and backward, with LoRA in the unrotated basis so the adapter exports as an ordinary GGUF LoRA for the fork. Implemented in `src/bonsai_gym/` and checked against the PrismML runtime (`docs/ARCHITECTURE.md`, parity results in `docs/EXPERIMENT_LOG.md`).
