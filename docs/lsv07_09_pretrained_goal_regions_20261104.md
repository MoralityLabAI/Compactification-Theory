# LSV-07–09: predictive goal-region findings, 2026-11-04

**Scope.** Reused the pinned [independently pretrained, 44,001-parameter, TRM-inspired Sudoku checkpoint](../research/levin_sieve_v1/pretrained_trm/README.md), Git source revision `f0406195fece4d0db29ced28f3a69d1993d95b1e`, checkpoint SHA256 `3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1`, and 64-puzzle roster SHA256 `eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611`. All damage is injected after the first step of a shared four-step recurrent block. Exactly three further passes are evaluated. This is not the canonical TRM, Qwen or LOTUS model.

## LSV-07: ordinary forward interval bounds

[Theorem and code](../research/levin_sieve_v1/pretrained_trm/PREDICTIVE_GOAL_REGION.md) / [CI run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41794358356) / [raw receipt artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41794358356/artifacts/13305246271).

A mathematically valid interval-inclusion theorem proves label preservation when the final logit lower bound exceeds all competing upper bounds. The practical FP64 implementation assumes unverified Torch erf/sqrt/BLAS error budgets. **The numerical results are conditional and do not establish a verified computer-level certificate.**

On the original 5,184 cell outputs, standard boxes certify 101 labels at radius zero and zero labels for every tested positive radius, starting at `1e-8`, for both unrestricted and one-cell hidden perturbations. The trained model's self-map/invariant-box test at radius 0.02 also fails its sufficient conditions. Thus the interval relaxation is too loose to substantiate practical predictive recovery.

Tests: 24/24 pass, with independent random forward checks. This is a negative *certification* result, not a demonstration that each prediction changes under small noise.

## LSV-08: centered sensitivity and LayerNorm Jacobian

[Proof and code](../research/levin_sieve_v1/pretrained_trm/CENTERED_REACHABILITY.md) / [CI run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41795288250) / [raw receipt artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41795288250/artifacts/13305081869).

The local LayerNorm result bounds each output difference from an exact clean trajectory via the variance lower envelope and the LayerNorm Jacobian formula. Composing componentwise radii with the MLPs and head produces a theoretically valid *conditional exact-center* certificate. Torch FP64 estimates the centers without validated directed-rounding error bounds.

At radius zero the method certifies all 5,184 clean digit predictions, including 5,094 correct cells and 90 wrong cells, with 20 of 64 puzzles both entirely certified and correct. At every tested positive radius, starting with `1e-8`, it certifies zero. The worst componentwise radius saturates at about 17 after three passes.

Tests: 30/30 pass. This follow-up was explicitly designed *after* inspecting the LSV-07 failure and is exploratory.

## LSV-09: goal-projected sensitivity

[Theorem, exact-rational oracle, and pretrained diagnostic](../research/levin_sieve_v1/pretrained_trm/GOAL_PROJECTED_THEOREMS.md) / [CI run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41796128892) / [raw gradient artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41796128892/artifacts/13305198169).

A rigorous inequality controls each pairwise final output-logit margin by the *weighted sum of uniform input derivative bounds on the entire damage box*. For affine recurrent maps the required row derivative is exactly `(w_a-w_b)^T A^k`; the Fraction-based checker passed 180 randomly generated systems against an independent exhaustive corner oracle. One exact example has hidden-state gain 20 while the goal prediction remains invariant for arbitrary input disturbances, because the output row annihilates the expanding mode.

On the real pretrained nonlinear model, the gradient **at the clean state** is measured with autograd. It is a tangent diagnostic; the current experiment has no certified derivative envelope for the whole box and makes **no finite-radius neural robustness claim**.

For each of the same 64 puzzles, select the first missing cell from the public puzzle input and use the top two clean predicted digits as the pair of interest. Median clean margin: **7.4374**. Median full-latent input gradient L1 norm: **89.0859**; median gradient L1 norm restricted to that cell's 48 channels: **10.7146**. These gradients and first-order radii are observations, not robustness certificates.

| Damage footprint | Amplitude | Cases with positive first-order tangent margin | Cases whose original highest-scoring class survived sign-gradient attack |
|---|---:|---:|---:|
| All hidden coordinates | 0.02 | 51 / 64 | 51 / 64 |
| All hidden coordinates | 0.10 | 31 / 64 | 17 / 64 |
| All hidden coordinates | 0.50 | 3 / 64 | 0 / 64 |
| First missing cell, 48 channels | 0.02 | 63 / 64 | 63 / 64 |
| First missing cell, 48 channels | 0.10 | 54 / 64 | 54 / 64 |
| First missing cell, 48 channels | 0.50 | 38 / 64 | 29 / 64 |

The disagreement at amplitude 0.10 for whole-state damage and 0.50 for the cell patch demonstrates why a clean-point derivative cannot substitute for a uniform regional bound. This table compares **64 selected cell predictions**, unlike the 5,184-cell counts in LSV-07/08. Equality of aggregate counts does not establish identical per-case behavior.

CI: 34/34 tests pass, and the independent checkpoint-gradient evaluator succeeds. The empirical model uses idealized FP64 future inference, with source weights originally stored in FP32.

## Research decision

The useful mathematics is now clearly the **goal-projected Jacobian/Hessian enclosure** problem, retaining cancellation across the recurrent transformation. A credible next benchmark must compute mathematically justified uniform bounds on each logit-margin derivative throughout an input region, with directed rounding and explicit GELU bounds; the certified radius must then be scored on fresh task groups without choosing favorable examples. Neural output robustness, hidden-state recovery, exact numerical verification, and economic inference savings remain separate questions.

No biological, metaphysical, self-consciousness, or pretrained canonical-TRM claim follows from these experiments.
