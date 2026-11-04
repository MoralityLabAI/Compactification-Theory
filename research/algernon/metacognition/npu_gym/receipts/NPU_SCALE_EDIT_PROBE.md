# NPU weight-edit feasibility probe (2026-11-04)

Question: can the Ryzen AI NPU model be edited by patching the quantized weight file, without recompiling?

Model: `amd/Qwen3-4B-awq-quant-onnx-ryzenai-1.7-hybrid` served by lemonade (`ryzenai-llm` v1.7.0). Graph `model_jit.onnx` has 299 nodes (109 `com.ryzenai.MatMulNBits`, 72 `SLRN`, 36 `GQO`, 36 `SSMLP`); graph outputs are `logits` and `present.*` KV only (no hidden states). Weights are initializers in `model_jit.onnx.data` with explicit offsets: int4 `qweight`, fp16 `scales`, `qzeros` per projection (1305 initializers, 253 scale tensors).

Procedure: greedy decode of a fixed prompt (40 tokens); unload; zero the fp16 `model.layers.20.attn.o_proj.MatMulNBits.scales` (163,840 bytes at its offset), original bytes saved; reload; same prompt; restore bytes (verified identical); reload; same prompt.

Observation:
- Baseline: `...multiplied by 2 to get the next term. So,`
- Edited: `...Each number in the sequence is double the previous one, making it a geometric progression...` (different continuation, same first line)
- Restored: byte-identical to baseline. So the runtime reads the patched `model_jit.onnx.data` at load, and decoding is deterministic.
- Reload took ~28-30 s each time; `model_jit.bin` was not regenerated or invalidated.

Scope: one edit, one prompt, one layer. Not measured: effect size per scale perturbation, whether small relative scale changes (the regime a tinyLoRA delta lives in) change logits (the server exposes no logprobs), or transfer of a GPU-trained adapter to the quantized weights.
