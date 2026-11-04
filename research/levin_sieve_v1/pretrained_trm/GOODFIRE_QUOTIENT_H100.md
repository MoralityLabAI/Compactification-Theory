# Silico / Goodfire H100: one-shot neural quotient test

**Status: staged proposal, unallocated.** This specification does not claim a GPU run or resulting 9B measurements. It follows the exact CPU LSV-17 pilot in the adjacent directory.

## Paste as one experimental task

Run exactly one source-locked experiment on a reserved **H100 80GB**, using the complete frozen BF16 Qwen3.5-9B checkpoint. Pin model and tokenizer revisions, 9B architecture, host runtime, every activation hook and every prompt. Select the six layer IDs 4, 9, 14, 19, 24 and 29 only after verifying the actual architecture indexing; fail closed on mismatches. Use pre-existing SAEs only when their checkpoint and feature indexing are pinned and auditable. Treat SAE features as measurement instruments.

**Hard resource budget:** at most 2 H100 GPU-hours including loading, collecting, training small heads, evaluating, and writing receipts. Use a single allocated GPU and one scripted run, with all configuration in a frozen JSON manifest. Persist activation caches locally within the allocation, reusing one model load and one set of tokenized prompts. Minimize repeated orchestration prompt input. Stop on budget exhaustion and emit a partial-result receipt with explicit counts.

Freeze 512 development tasks and 512 held-out tasks before perturbation results are seen, grouped by task ID and family. Include independent ground-truth task success as well as behavioral logits for at least some tasks. Preserve exact prompt and continuation seeds. For each selected layer, compare matched-L2 perturbations using: (a) seeded centered random direction; (b) SAE-feature-aligned direction; (c) goal-history Jacobian-null direction, explicitly marked teacher-informed; and (d) terminal margin gradient-positive direction. Add a LayerNorm gauge control only if the actual hook exposes the correct pre-LayerNorm bus.

Run paired clean and edited continuations from identical prefixes. Record first-step distribution shift, bounded fixed-horizon token KL, delayed tool/action trace changes, independent correctness, SAE activation changes, layer ID, direction L2 norm and elapsed GPU time. Distinguish agreement of one token from agreement of a later action or proof certificate.

Fit a nonlinear low-dimensional quotient head of width 16 or 32 and a matched-input linear baseline, using only development split for fitting and hyperparameter selection. Reserve at least 256 entirely untouched tasks for a final blind comparison. Report held-out future-output prediction, task success, parameter counts, split-leakage checks, and paired task-clustered uncertainty intervals. Do not modify the 9B backbone weights.

**Gate for escalation:** repeated held-out nonlinear prediction gain with unchanged or better task success, causal perturbations showing large activation displacement and small delayed behavioral displacement, and a negative control where high task-relevant displacement is detected. Publish failures and zero-gain outcomes as first-class results.

Deliver a hash manifest, task split, anonymized pair index, activation-file index, interventions JSONL, summary JSON, one compact comparison chart, and a small independent CPU metric verifier. All deviations from the frozen plan must have an amendment log before their data is used.

This is an exploratory task-specific experiment. The general Unique Games nonlinear/linear separation theorem does not imply such a separation for pretrained transformer activations.
