# LSV-05: Pretrained recurrent checkpoint damage and rounding audit

**Date:** 2026-11-04. **Evidence:** reproducible pinned external checkpoint; independent held-out Sudoku tests; analytical bounds for *ideal real-arithmetic* evaluation of exactly represented FP32 and BF16-stored parameters. The experiment is on a third-party 44,001-parameter TRM-inspired Sudoku network, distinct from the original 7M-parameter Tiny Recursive Model. Its fixed checkpoint is separately trained by its upstream author.

## Locked source

- Upstream: [NagaSatyaPhanindraVallabhaneni/tiny-recursive-model](https://github.com/NagaSatyaPhanindraVallabhaneni/tiny-recursive-model), commit `f0406195fece4d0db29ced28f3a69d1993d95b1e`.
- Model: `models/trm.pt`, Git blob SHA1 `dfbe8ffbeb52f2c32f0c44fd326cd26c96272c54`, 181,147 bytes. The harness checks the entire Git blob preimage including the `blob {length}\0` prefix. It also checks SHA256 if pinned in `source_lock.json`, and always emits the computed SHA256 in the receipt.
- Architecture and Sudoku generation are minimal mirrors of the upstream MIT-licensed Python code, with its license copied in `LICENSE_UPSTREAM.txt`. Restricted `torch.load(weights_only=True)`, an exact key roster and `load_state_dict(strict=True)` prevent silent architecture mismatch.
- Source is unrelated to the local LSV-04 two-parameter trained proxy. It serves as the first external pretrained recurrence check and has no relation to a pretrained Qwen or pretrained LOTUS checkpoint.

## Test

```sh
python3 -m unittest discover -s research/levin_sieve_v1/pretrained_trm -p 'test_*.py' -v
python3 research/levin_sieve_v1/pretrained_trm/run_checkpoint.py --fetch \
  --out runs/lsv05/receipt.json
```

The network fetch occurs only when `--fetch` is explicitly passed; an already-downloaded checkpoint can be supplied with `--checkpoint`. The script audits hashes before deserializing the model.

A fixed set of 64 uniquely generated Sudoku puzzles (30 holes each) is disjoint in generator seed from the upstream training and validation roster. It compares clean 1/2/3/4 recurrences, input-to-stage-one perturbations at multiple amplitudes, one stale whole-state update, and BF16-stored weights cast back to FP32 for evaluation. Targets are used solely for scoring. Stage-one hidden-state damage is measured against an untouched clean trajectory and task-level correctness.

## Analytic bound

For each block, the token- and channel-mixing MLPs use matrix infinity norms, and GELU has a global derivative bound of `5/4`. LayerNorm on `d=48` features with `eps=1e-5` obeys:

\[
\|\operatorname{LN}_{\gamma,\beta}(u)-\operatorname{LN}_{\gamma,\beta}(v)\|_\infty
\le (2+\sqrt{48})\,\|\gamma\|_\infty/\sqrt{\varepsilon}\,\|u-v\|_\infty.
\]

The checker uses rational upward constants `ceil(sqrt(48))=7` and `ceil(1/sqrt(1e-5))=317`, hence a global coefficient `2853 * max(|gamma|)`. The normalized coordinate magnitude is bounded by `sqrt(48)`, so **every post-LN2 latent coordinate lies within**

\[
 |z_i| \le 7\|\gamma_{\mathrm{LN2}}\|_\infty+\|\beta_{\mathrm{LN2}}\|_\infty
\]

in exact real arithmetic. This absolute magnitude bound holds for arbitrary incoming perturbations. It says nothing on its own about task correctness or distance from the undamaged trajectory.

Weight tensors are read as exact rational values of their stored binary floating-point numbers. The BF16-stored version uses the rounded-to-BF16 values, converted back to FP32. Exact rational row sums and parameter differences propagate an upper bound on the output-logit change due to weight storage quantization through the four recurrent passes. The method also computes an upper bound on the block Lipschitz constant, and explicitly indicates when it fails the strict contraction criterion. **The bound excludes arithmetic roundoff inside PyTorch kernels, accumulation precision, GPU kernels, and numerical approximations to GELU.** Runtime quantization effects are measured separately.

This analytic envelope is intentionally conservative. A large bound means the global certificate is inconclusive, even if observed recovery occurs. A practical next gate is a region-validated local Jacobian or fixed-point certificate around a declared task set, with interval arithmetic and an invariant-set proof.

## Evidence and limitations

A successful CPU benchmark verifies the checkpoint's model and its behavior on this frozen roster. It does not establish globally convergent neural dynamics, performance on the original 7M TRM, faithful replication of upstream model accuracy, or numerical guarantees for arbitrary PyTorch backends. The Sudoku generator uses an extremely structured grid family; the independent test seed blocks prevent direct train/test seed reuse, while the broader source distribution remains similar.

## LSV-07: predictive goal-region enclosures

The [LSV-07 theorem and numerical scope](PREDICTIVE_GOAL_REGION.md) and [frozen protocol](PREDICTIVE_PROTOCOL.md) extend LSV-06's measured-output-radius check to an enclosure computed **before** an unknown disturbance runs. [Code](predictive_interval.py) propagates a specified stage-one hidden-state box through three complete frozen recurrent blocks and the digit readout; [benchmark](run_predictive.py) counts unchanged predicted labels, including stable wrong ones, under two damage footprints. A separate invariant-box test distinguishes finite-depth output stability from perpetual recurrent invariance. The interval enclosure depends on stated but unverified Torch FP64 numerical error assumptions, so its output is a **conditional** numerical certificate rather than a fully validated directed-rounding proof. Existing source/checkpoint hashes and 64-puzzle roster remain unchanged.
