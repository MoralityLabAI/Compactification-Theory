# Architecture

## Frozen packed base (`src/bonsai_gym/model.py`, `packed.py`, `gdn.py`)

The base is `Ternary-Bonsai-2-27B-PTQ1_0.gguf` read in place (mmap). Every PTQ1_0 tensor is copied to the GPU as two tensors, `codes` (uint8, 26 bytes per 128 weights) and `scales` (fp16, one per 128 weights): 5.88 GB for all 403 packed tensors, including the 248,320 × 5120 embedding and LM head. Nothing dense the size of a layer is ever kept.

A projection computes, as the fork's `build_lora_mm` does:

```
y = W_rot · H(s ⊙ P x)  +  (α/r) · B A x
```

- `W_rot` is decoded on the fly, `chunk` rows at a time (default 64M weights ≈ 128 MB of bf16), by a lookup-table gather (`decode`: byte → five base-3 digits, then the fork's 16/8/2 staging order); the decoder matches `ptq1_codec` bit for bit (tests), which matches the fork's C decoder on every tensor of the file (S0 receipt).
- `H` is the normalized Sylvester Walsh–Hadamard transform in blocks of 1024, applied to activations as a matmul; `s` the explicit sign vector for the input width (5120, 6144, 17408) from `prism.hadamard.sign_values`; `P` the tiled→grouped GDN head permutation, `ssm_out` only (`prism.hadamard.gdn_v_grouped`). The rotated input is computed once per activation and shared by every projection that reads it (the fork's memo).
- The embedding row is restored as `s ⊙ H(z)` (the one inverse-transform table).
- Backward: a custom autograd function returns `dL/dz = Σ_chunks gout[:, chunk] · W_chunk`, decoding again rather than saving `W`. The packed tensors get no gradient.
- LoRA reads the primal (unpermuted, unrotated) input. That is the basis llama.cpp applies a GGUF LoRA in, so `adapter_io.write_gguf_lora` output loads into the fork unchanged.

Layer graph (mirrors `src/models/qwen35.cpp` at PrismML c96baf4, cache-free):

- 48 gated-delta-net layers: `attn_qkv` → causal depthwise conv (k = 4) → SiLU → q/k (16 heads × 128, L2-normalized) and v (48 heads × 128, tiled order, q/k repeated by tiling); decay `g = ssm_a · softplus(x W_α + dt_bias)`, `β = σ(x W_β)` (bf16, unrotated); chunked gated delta rule (64-token chunks, fp32; `gdn.py`, tested against the per-token recurrence); gated RMSNorm with `SiLU(attn_gate x)`; `ssm_out`.
- 16 full-attention layers (every 4th): `attn_q` gives query and output gate per head (24 × 256 each); q/k RMSNorm; NEOX RoPE on the first 64 of 256 dims, base 1e7 (the fork's IMROPE sections reduce to this for text positions); GQA SDPA (4 KV heads); `σ(gate) ⊙ attn`; `attn_output`.
- SwiGLU FFN 5120 → 17408 → 5120. RMSNorm weights already include the +1 (verified: means ≈ 1).

Inputs may be `[T]` or right-padded `[B, T]`; padding sits after the real tokens, so causal attention and the causal scan leave real positions unchanged.

Training memory: per-layer activation checkpointing (`torch.utils.checkpoint`, non-reentrant) keeps one `[B, T, 5120]` bf16 tensor per layer; the recompute of one layer and its decoded weight chunks are the only large transients. LoRA parameters are fp32 with AdamW; logits are computed only at target positions.

## Data (`src/bonsai_gym/data.py`)

Prompt = the GGUF chat template with `enable_thinking=False` (string-compared to the jinja template in tests); target = assistant text + `<|im_end|>`; loss on target tokens only — the q1-adapters `train_lora.py` contract, so 8B and 27B adapters train on identical strings. Tokenizer: Qwen3.5 `tokenizer.json`, identical to the GGUF vocabulary (248,070 real tokens + 250 pad entries) and merges.

## Scripts

| script | does | GPU |
|---|---|---|
| `scripts/llama_reference.py` | fork llama-server greedy continuations + llama-perplexity (base or `--lora`) | yes |
| `scripts/parity.py` | torch forward vs that reference: tokenizer ids, teacher-forced top-1, PPL | yes |
| `scripts/smoke.py` | first N layers, one LoRA forward/backward | yes |
| `scripts/canary.py` | config-driven LoRA training canary with eval, save/reload, GGUF export, resume | yes |
| `scripts/eval_llama.py` | reverse-text LCS on 200 env items via llama-server (q1-adapters sampling), paired bootstrap | yes |
| `scripts/prep_canary_data.py` | canary splits from q1-adapters, scorer check | no |
| `scripts/build_corpus.py` | meta-learning SFT corpus with manifest and hashes | no |

Every GPU job runs through `vramorama run --vram … --guard-power …` (shared card, one queued job of ours at a time).

## Gym (`src/bonsai_gym/gym.py`)

```
controller (LinUCB over task features)
  → task selection → specialist / adapter update (callable)
  → verification on a fresh dev sample (paired bootstrap; gain counts only if CI > 0)
  → regression guard on anchor tasks → optional second-format replication (PTQ1_0 vs PQ2_0)
  → keep or revert → ledger record → controller update (dev reward only)
  → every k rounds: confirmatory read from the sealed Vault (budgeted, logged, never a reward)
```

Controls: `Vault` (hash-pinned confirmatory items, read budget counted from the append-only ledger, labels never exported), `Thresholdout` for dev reuse, `contamination` (13-gram and exact-hash overlap between training rows and sealed items), `RegressionGuard`, a reject cost for reverted updates. The loop is tested on synthetic tasks only (`tests/test_gym.py`); no 27B gym round has run.

## Probes (`src/bonsai_gym/probes.py`)

Capture residual vectors for base (LoRA scale 0, same object) and adapted weights on identical ids; family-held-out ridge probes against a permutation null (correlational); steering along a probe direction against random directions (causal). Not run yet.
