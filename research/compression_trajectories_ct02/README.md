# CT-02: Compression trajectories and layer-reuse growth

**Status 2026-10-29:** self-contained synthetic CPU pilot implemented and executed at three seeds. This is a methods check, **not** an LLM or MoE replication. No model download, GPU access, BitAgent deployment, or ongoing worker is required. Other experiments and owners in this repository remain untouched.

## Research question

For a pretrained parameter state \(\theta_0\), a final architecture \(A_*\), and a recovery stream \(S\), does the schedule \(\gamma = ((A_i,\Pi_i, n_i))_{i=1}^k\) affect retained capabilities after the same exposure to examples? Here \(\Pi_i\) is the structural reduction or growth operator, and \(n_i\) is the number of optimizer steps at stage \(i\). The answer is an **empirical** property of the task, initialization, optimizer, and training protocol. Different training histories can reach functionally different states in the same final architecture.

Motivation: [SlimQwen](https://arxiv.org/abs/2605.08738) evaluates staged depth/width/expert reduction in **Qwen3-Next-80A3B**, followed by substantial continued pretraining. The unrelated October 29 [Qwen3.5 9B/27B layer-reuse claim](https://x.com/ChangJonathanC/status/2115568670772535424) reports unusually strong same-index weight similarity at repeat offsets; the image is third-party evidence, and the model-construction history remains independently unverified. The second finding motivates a separate toy *growth* arm. These are different model families and different experimental directions.

## Methods and fixed controls

The reference is an eight-block, width-64 residual MLP trained against a six-class nonlinear, phase-conditioned oracle. Outputs 0–1 belong to INSPECT, 2–3 to ACT, and 4–5 to VERIFY. Each arm uses the **same synthetic training examples, teacher, held-out sets, recovery mini-batch index stream, base LR, and optimizer reset at the stage boundary**. The final architecture is four blocks of width 32 with the same 6-way head. Every initial copy uses an explicit top-left width crop; depth selection is either uniformly spaced or calibration-importance-ranked. A fresh-initialization control shares the final architecture.

Arms: one-shot CE; one-shot CE+KD; depth-first CE+KD; width-first CE+KD; joint CE+KD; layer-saliency one-shot CE+KD; and fresh-initialized CE+KD. The KD loss uses \(0.5\,CE + 0.5\,T^2 KL(p_{teacher,T}\|p_{student,T})\), \(T=2\). Both one-shot and staged arms receive a 10%/90% stage split and fresh AdamW at the split. The phase gate operates **outside** the model; it rejects attempted out-of-phase actions deterministically, and the report distinguishes attempts from executed violations.

`--budget tokens`: equal recovery examples per arm, while intermediate models use more multiply-accumulate operations. `--budget flops`: approximately equal **3× forward-MAC proxy** through integer step rounding, which changes the number of examples seen. Both are required to examine a compute tradeoff. The proxy excludes input generation, evaluator, calibration saliency, and shared cached teacher predictions; it is **neither measured hardware FLOPs nor wall clock**. Teacher training cost is shared and reported separately. Final interpolation uses the same architecture and checks CE at \(t=0.25,0.5,0.75\), reporting a positive barrier above both endpoints. The result is a diagnostic, without neuron permutation alignment or a basin theorem.

A separate `--growth` experiment expands a trained four-block model to eight blocks, placing repeated copies of two middle layers in an `0,1,2,1,2,1,2,3` pattern. Variants are raw untied copies, copies with each reused middle block's residual gain divided by three, random inserts with the same scaled gains, and zero-residual identity inserts. Copies are independent trainable tensors. Identity insertion serves as an exact initial-function control. Growth arms share recovery step streams, target architecture, and teacher. This tests initialization mechanisms and cannot verify real Qwen3.5 weight ancestry.

## Run offline

Requires Python 3.10+ and PyTorch (existing root requirements suffice). Commands from the repository root:

```bash
python -m unittest discover -s research/compression_trajectories_ct02 -p 'test_*.py' -v
python -O -m unittest discover -s research/compression_trajectories_ct02 -p 'test_*.py' -v
python research/compression_trajectories_ct02/trajectory_lab.py --seeds 0,1,2 --teacher-steps 160 --student-steps 100 --batch 128 --growth --out /tmp/ct02_tokens.json
python research/compression_trajectories_ct02/trajectory_lab.py --seeds 0,1,2 --teacher-steps 160 --student-steps 100 --batch 128 --budget flops --out /tmp/ct02_flops.json
```

Runs emit schema `ct02.synth.v1` with code SHA-256, PyTorch version, each seed's source teacher score, per-arm recovery examples/MAC proxy, held-out and shifted accuracy, teacher KL, before/after deterministic gate violations, pairwise linear interpolation barriers, and growth initialization functional MSE. Seeds determine initial weights, synthetic samples, and mini-batch streams. Numerical equality across PyTorch installations and kernels is not guaranteed.

## Executed three-seed methods pilot (seeds 0,1,2)

Teacher: 160 steps; student: 100 steps; batch 128; same synthetic task throughout. Scores are **mean validation accuracy**, rounded to three decimals. JSON receipts in this folder are the source of record.

| Arm | Equal examples | MAC-proxy matched |
|---|---:|---:|
| One-shot CE | 0.877 | 0.877 |
| One-shot CE+KD | 0.868 | 0.868 |
| Depth-first CE+KD | 0.866 | 0.843 |
| Width-first CE+KD | 0.868 | 0.859 |
| Joint CE+KD | 0.868 | 0.847 |
| Saliency-selected one-shot CE+KD | 0.864 | 0.864 |
| Scratch CE+KD | 0.781 | 0.781 |

Teacher mean accuracy: 0.851. Progressive depth-first used ~229M training MAC proxy versus ~179M for one-shot in the equal-example experiment. Growth initial-function mean squared difference from the 4-block model: raw repeated blocks **20.15**, scaled repeats **0.0106**, scaled random insertions **2.14**, and identity insertions **0.0**. These numbers come from 3 seeds, a small smooth synthetic model, and short recovery training. There is **no evidence here of a universal progressive-pruning benefit**; the fresh-initialization gap is the clearest pilot signal. Expansion needs scaled residuals to preserve the original function approximately.

## Pre-register next benchmark before looking at results

1. Fix three architectures and an independent task-family split, teacher checkpoint, seeds (minimum 10 paired), optimization hyperparameters, and a single primary target capability. Do not tune on held-out families.
2. Compare the schedules under **both** example-matched and measured-device-compute-matched budgets; measure actual throughput, VRAM peak, and any extra saliency or distillation work. Include a larger pretrained transformer/LoRA path only after this synthetic experiment is stable.
3. Primary directional test: paired difference in held-out accuracy (progressive vs one-shot CE+KD) with a 95% paired bootstrap CI over seed-level outcomes. Register a minimum meaningful effect (e.g. +1.0 percentage point) and a multiplicity policy before data collection. Treat family holdouts as separate units where appropriate. The hypothesis fails its claimed useful-benefit gate if the interval does not support that minimum at fixed total compute.
4. Inspect action legality before and after the external gate, out-of-distribution accuracy, KL-to-teacher, and CE interpolation barriers. A gate failure is a code-contract failure regardless of model score. A low barrier is an observation about the tested parameter interpolation only.
5. For Qwen3.5 lineage, separately pin the official 9B and 27B checkpoint revisions and tensor hashes, compare layer-family-compatible same-index rows at offsets 0,+16,+32 with shuffled and nearest-neighbor/permutation controls, and report matched cosine distributions plus exact equality rates. Similarity is evidence of shared parameter ancestry; a training recipe requires further provenance.

**Ownership:** CT-02 lives exclusively under `research/compression_trajectories_ct02/` in Compactification-Theory. This package does not alter `HybridTRMLDT`, `EvalMath`, IC-01, RC-01, existing LoRA benchmarks, model weights, or production BitAgent control policies.
