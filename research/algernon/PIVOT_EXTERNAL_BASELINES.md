# Algernon pivot: build on reproducible external successes

**Decision 2026-11-03.** The 77 legacy LoopedTransformers checkpoint files (59 distinct SHA-256 hashes) and their unresolved training logs are **archival material**. Recovering the D-drive experiment logs is optional and explicitly **not a Phase A or B prerequisite**. Do not spend additional agent/GPU time searching for those checkpoints unless a future audit specifically asks for them.

## New critical path

Start with a working external reference, preserve its behavior on a fixed test roster, then make one independently measurable change.

| Priority | Implementation | Established reference | Our concrete first test |
| --- | --- | --- | --- |
| **P0: frozen loop** | [Training-Free Looped Transformers](https://github.com/L-z-Chen/Training-Free-Looped-Transformer), [paper](https://arxiv.org/abs/2605.23872) | Authors show damped middle-block recurrence on pretrained Qwen3 with reproducible code; reported Qwen3-4B-Instruct MMLU-Pro +2.64 percentage points | First reproduce baseline vs anchored damped K=6 wrapper on the author's supported Qwen3 architecture. Only after verification explore Qwen3.5, as its DeltaNet/attention internals differ |
| **P1: trained latent core** | [ReLIT](https://arxiv.org/abs/2608.08113) | Published lightweight recursive latent block on a frozen TinyLlama-1.1B, with reported logical reasoning gains | Locate verified code/weights and reproduce reported task accuracy. If unavailable, use [Retrofitted Recurrence](https://github.com/mcleish7/retrofitting-recurrence), which supplies public training/conversion/evaluation code |
| **P1: deterministic processor** | [Qwenputer](https://github.com/divagr18/qwenputer) | Compiled RV32 computer and DeltaNet RAM in Qwen3.5-4B; published reference checks | Repeat CPU reference suite and one minimal compiled arithmetic/state task, preserving exact trace receipts |
| **P1: sparse repair** | [Weight-Space Programming logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math) | Published reversible sparse p50c patch in Qwen3.5-0.8B | Finish independent frozen-patch evaluation from [HybridTRMLDT PR #5](https://github.com/MoralityLabAI/HybridTRMLDT/pull/5), compare support/norm-matched random control |
| **P2: joint learning computer** | Algernon | Speculative integration | Combine only after the first two components show causal benefit independently, with complete compute accounting |

## P0 proof of concept: preserve a working external Qwen3 model

**Replication protocol:**
1. Pin upstream Git commit and all environment/model revisions before inference. Inspect upstream licensing, versions and its test set, then run its exact baseline/recurrent settings.
2. Run a dry model-load and one prompt with K=0. On a supported Qwen3 model, validate that K=0 reproduces original logits/outputs to a recorded tolerance.
3. Evaluate the upstream's anchored damped recurrence with K=6, beta=0.5 and its documented cache handling; keep sample roster and prompt template identical.
4. Compare K=0, K=1, K=2, K=4, K=6 and K=8, pre-registering a single primary task and a total GPU/token limit. Distinguish loglikelihood prefill-only scores from generative decoding tests.
5. Check held-out gains, regression on base skills, KL to original model, latency, GPU-seconds and peak VRAM. Test naive unscaled block replay as a negative control.
6. Only after a positive reproduction, attempt a **small Qwen3.5-specific** recurrence adaptation, measuring DeltaNet-state/cache correctness explicitly.

**Causal extension:** freeze the successful loop operator; replace its contribution with a constant state, a shuffled state and a same-compute one-pass residual transform. If only the looped arm improves, investigate the responsible layer circuit with VPD.

## Recurrent graft follow-up

A trained ReLIT-style bridge, when verified, is the candidate to join Qwen with an iterative latent core. Train a zero-initialized, reversible bridge while the pretrained base stays frozen. Compare recurrent depths against an equally sized feedforward block. Evaluate new tasks by generator-disjoint split and lock candidate layer selection to calibration data. Keep the compiled exact kernel separate until training/recurrent gains are independently measured.

## Hard boundaries

- **Model-family boundary:** The Qwen3 training-free wrapper cannot simply be assumed correct on Qwen3.5 Gated DeltaNet. Reuse its research method, then write and validate a model-specific adapter.
- **Hardware boundary:** Published Qwenputer RV32 Qwen3.5-4B BF16 checkpoint (~18.7 GB) exceeds a 16 GB RTX 5080 in all-VRAM form. Plan offload/quantization and repeat exactness tests before deployment.
- **Scientific boundary:** A weight-space repair that rescues existing reasoning does not demonstrate algorithm grafting, while exact compiled arithmetic does not demonstrate learned reasoning.
- **Compute boundary:** Start with small frozen-roster smoke tests, an explicit spending cap, and preserved raw receipts. Avoid large exploratory training until a reference baseline is independently reproduced.

## What to archive

Keep `research/algernon/inventory/` and its hashes as historical provenance. The optional legacy retrospective may later recover training recipes and results from D drive. No present experiment depends on doing so.

## Exit conditions

**P0 pass:** pinned external source + model; same-device base/loop comparisons; recorded actual inference costs; primary accuracy result with paired uncertainty; absence of severe regression; reproducible command and retained records.

**P1 pass:** independently measured causal depth benefit from a trained latent module, or a documented negative result; exact computation validated separately.

**P2 pass:** a combined processor improving a held-out task over its best component at matched information and metered compute, with a state/weight ablation that removes the improvement.

Keep the original H-C, H-R, H-W and H-J hypothesis identifiers. This document updates execution priority without rewriting the registered scientific questions.
