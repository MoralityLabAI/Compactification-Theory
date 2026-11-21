# Inkling depth/precision bridge: gated recurrence, numerical error and goal-preserving quotients

Status: literature-derived source-only mathematics and an executable CPU bounds instrument (2026-11-21).
No neural checkpoint, proprietary data, GPU run, or claim of measured Jacobians. Keep all existing
NOR-37 and NOR-48 original records and verdicts unchanged.

## References inspected

- Thinking Machines, [Inkling 2 Mini release](https://thinkingmachines.ai/news/inkling-2-mini/), 2026-11-20.
- Thinking Machines, [End-to-End 4-Bit Training](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/).
- Chen et al., [Attention Residuals](https://arxiv.org/abs/2603.15031), 2026.
- Dehghani et al., [Universal Transformers](https://arxiv.org/abs/1807.03819), 2018.
- Yang et al., [Looped Transformers Are Better at Learning Learning Algorithms](https://arxiv.org/abs/2311.12424), 2024.
- Zhu et al., [Scaling Latent Reasoning via Looped Language Models / Ouro](https://arxiv.org/abs/2510.25741), 2025.
- Bae et al., [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524), 2025.
- Chen et al., [TetraJet-v2](https://arxiv.org/abs/2510.27527), 2025/2026.
- Panferov et al., [Quartet II](https://arxiv.org/abs/2601.22813), 2026.
- Goodfire, [Interpreting Inkling 2 Mini](https://www.goodfire.com/research/interpreting-inkling-2-mini/), 2026-11-20.

Inkling's 24 physical layers apply 12 fixed layers and 12 recurrent layers for each pass.
The fixed three-pass model therefore performs 48 layer applications per generated token.
The variable-loop research checkpoint trained pass counts 1–5 at ratio 1:2:6:2:1 and
post-trained at pass 3; pass 7/10 are extrapolation experiments. The 20 GB checkpoint
requires >=32 GB GPU/unified memory for native deployment. See its
[model card](https://thinkingmachines.ai/model-card/tml-d24-varloop-research/).

## Derivation: a fixed-gate precision envelope

Let a frozen-input recurrent map be

    z[t+1] = kappa[t] ⊙ p(x) + gamma[t] ⊙ M[t](z[t]),

and let a quantized implementation introduce local numerical deviation bounded by
epsilon[t] in infinity norm. If gamma and kappa are held constant with respect
to state and M[t] has infinity-norm Lipschitz constant m[t], then

    e[t+1] <= L[t] e[t] + epsilon[t],
    L[t] := ||gamma[t]||_infinity * m[t].

By induction,

    e[T] <= (prod_{j=0}^{T-1} L[j]) e[0]
            + sum_{j=0}^{T-1} epsilon[j] prod_{h=j+1}^{T-1} L[h].

The base case is the initial error. The induction step follows by inserting
the exact intermediate map between the quantized state transition and the
reference transition, then using the triangle inequality and Lipschitz bound.
This is an upper bound on numerical deviation, conditional on supplied L and epsilon.
It is not a guarantee of correct task output, attractor identity, or fidelity
to a private/unobserved reference state.

**Important assumption:** In Attention Residuals the depth-wise softmax and
cross-loop gate can depend on state. Their complete Jacobian contributes
terms involving derivatives of the gate, normalization and the reader.
The fixed-gate formula cannot certify Inkling by reading only gamma weights.
A native test must estimate/upper-bound the full local operator Jacobian.

For L < 1 and bounded epsilon, the numerical error approaches a finite
envelope. A perfectly converged wrong answer remains wrong. A serial pointer
chase can continue making verified progress through pass 10 even if its
successive states are far apart. Use the task verifier and algorithmic hop
state as distinct observable quantities.

## Four proposed falsifiers

1. **Recurrent placement**: compare middle-12 loop, final-12 loop and fully
   untied iso-application controls with matched weights, data and inference.
   Primary outcome is independently verified depth extrapolation, not NLL
   measured on the same training tasks.
2. **Quantization placement**: at the same file/activation budget, compare
   quantizing every loop with preserving cross-loop gates, block-boundary
   outlier coordinates and the coda at higher precision. Include randomly
   selected matched-number, matched-RMS protected coordinates.
3. **Residual access / goal quotient**: construct a state quotient q(z) and
   evaluate if the future output and next task-relevant operator are
   measurable from q(z) at a specified pass. Measure conditional future-output
   agreement on fresh root families, including serial and convergent tasks.
4. **Exit prediction**: calibrate a per-task depth-value predictor on ordinary
   development roots, then evaluate the fixed maximum, oracle hindsight,
   learned exit gate and random depth under *measured* compute budgets.
   Keep oracle as a diagnostic upper bound, not a deployable policy.

A meaningful precision certificate must report the reference precision,
software/hardware mode, source hashes, per-pass local error, measured or
certified full-state Jacobian bound, activation outlier statistics and
independently checked answer deltas. The native NVFP4+MXFP8 recipe and the
TetraJet/Quartet innovations concern training-specific numerics; our
source-only error bound does not validate equivalent benefits on an RTX 5080.

## Source-only local execution

    python -m unittest discover -s research/inkling_depth_precision -p 'test_bounds.py' -v

The instrument in bounds.py computes the stipulated envelope and analytical
token-layer applications. It does not substitute for measured CUDA throughput,
cache costs, state storage, energy, or correct-answer rate.

## Interfaces with neighboring projects

- LoopedTransformers: middle/final loop placement, coda lens, sustained
  input reinjection and per-step verified serial tasks.
- EvalMath: task-root matched depth comparisons, compute measured separately
  from layer-application proxies, selection-exposure receipts.
- BPD: component-specific interventions at block boundaries; calibrated,
  dose-matched native shams before mechanistic attribution.

The proposed mathematics is **derived source-only**, not a new verified neural
finding. Do not open or relabel sealed NOR or unrelated confirmation cohorts.
