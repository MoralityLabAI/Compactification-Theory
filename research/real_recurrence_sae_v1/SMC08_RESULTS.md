# SMC-08: Measured recurrent Jacobians, exact global curvature and validated local SAE tubes

**November 15, 2026.** Source-locked full result of [SMC-08](SMC08_PROTOCOL.md), with mathematical companion [OLM-08](https://github.com/patrickdugan/MathCan/blob/research/olm08-validated-tube-20261115/research/orbit_lock/OLM08_VALIDATED_CAUSAL_TUBE.md). This is a follow-up on the **same** 16 heldout puzzle IDs already examined in SMC-07 and has an explanatory, rather than independently confirmatory, interpretation.

## Provenance and completed hosted executions

- Actual external pretrained 44,001-parameter TRM-inspired Sudoku recurrence, four shared iterations, 81 tokens × 48 channels. Upstream commit f0406195fece4d0db29ced28f3a69d1993d95b1e.
- Exact checked weight SHA-256: 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1.
- Existing 64-puzzle roster SHA-256: eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611.
- 32 train / 16 validation / 16 test puzzle IDs, all 81 tokens of a puzzle assigned to one cohort. Frozen model is evaluated as FP32 through stage one, then with its exact stored FP32 weights cast to FP64 for the three-step continuation.
- Original SMC-07 top-k SAE training protocol, seeds 71,89,113, 300 updates, best validation checkpoint. SAE fitting has no test-state access.
- [Numerical 64-puzzle CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43705306641) succeeded, with a full per-puzzle JSON artifact. [Full Arb local-bound CI](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43705858285) succeeded on the selected active SAE direction for seed 71.

The [permanently committed 16-puzzle clustered summary](SMC08_CLUSTERED_SUMMARY.json) preserves the bootstrap and gradient-alignment measurements. The original per-seed/per-puzzle intervention traces and the Arb interval enclosures remain downloadable through the linked hosted run artifacts.

## FP64 causal Jacobian geometry of goal logits

For each of 16 heldout puzzles, hold the other 80 stage-one tokens fixed. The numerical Jacobian is 10×48, from the selected first-blank goal token to ten logits after three further recurrent blocks. Compute the top right singular vector to obtain the maximally sensitive *local linear* direction. The SAE direction is its highest active decoder atom, unit-normalized in the original hidden-coordinate metric. Sampled inactive and isotropic directions use the same input norm.

Mean \(\|Ju\|_2/\sigma_{\max}(J)\), averaging three SAE seeds within each puzzle:

| Direction | Mean fraction of top singular slope | Descriptive 95% puzzle-bootstrap interval |
|---|---:|---:|
| Highest active SAE atom | 0.3750 | [0.3229,0.4313] |
| Sampled inactive SAE atom | 0.2812 | [0.2289,0.3397] |
| Isotropic vector | 0.1665 | [0.1405,0.1923] |
| Jacobian-optimal right singular vector | 1.0000 | [1.0000,1.0000] |

These FP64 autograd singular values are measured numerical derivatives, **not Arb-verified spectral enclosures**. The difference between active and control directions is consistent with greater alignment to the Jacobian's high-sensitivity directions; it does not by itself identify a symbolic feature.

## Actual finite-radius displacement versus first-order prediction

At three preregistered radii, compare the Euclidean norm of the ten-logit output displacement. The following differences average three SAE seeds for each heldout puzzle and bootstrap the **16 puzzle IDs** 20,000 times with replacement. Intervals are descriptive.

| Radius | Active minus control | Actual effect difference | Jacobian-predicted difference | Descriptive 95% interval | Positive puzzles |
|---:|---|---:|---:|---|---:|
| 0.02 | Isotropic | +0.01163 | +0.01169 | [+0.00661,+0.01710] | 15/16 |
| 0.02 | Inactive SAE atom | +0.00768 | +0.00774 | [+0.00315,+0.01292] | 12/16 |
| 0.02 | Jacobian-optimal direction | -0.02588 | -0.02585 | [-0.03565,-0.01789] | 0/16 |
| 0.2 | Isotropic | +0.11141 | +0.11689 | [+0.06170,+0.16497] | 15/16 |
| 0.2 | Inactive SAE atom | +0.07152 | +0.07745 | [+0.02623,+0.12344] | 11/16 |
| 0.2 | Jacobian-optimal direction | -0.26036 | -0.25851 | [-0.35795,-0.18083] | 0/16 |
| 0.8 | Isotropic | +0.38460 | +0.46755 | [+0.19921,+0.58926] | 15/16 |
| 0.8 | Inactive SAE atom | +0.21900 | +0.30978 | [+0.04498,+0.42299] | 10/16 |
| 0.8 | Jacobian-optimal direction | -1.01707 | -1.03405 | [-1.35335,-0.70333] | 0/16 |

For the active atom, mean \(\|\Delta F-rJu\|/\|\Delta F\|\) is **0.41%** at radius 0.02, **4.12%** at radius 0.2 and **16.69%** at radius 0.8. This is a measured finite-difference ratio, and is *not* a uniform Hessian upper bound. The selected Jacobian-optimal direction exceeds the active SAE direction in all 16 puzzle clusters for all three radii. At radius 0.02 the first-order model closely predicts the effect ranking.

## What the exact-real global certificate proves

The [rational source-weight checker](validated_causal_tube.py), following [OLM-08](https://github.com/patrickdugan/MathCan/blob/research/olm08-validated-tube-20261115/research/orbit_lock/OLM08_VALIDATED_CAUSAL_TUBE.md), uses integer-verified dyadic Frobenius upper bounds for each linear operator, established global GELU first/second derivative bounds and the LayerNorm Hessian inequality. Through 12 elementary stages it computes an exact rational outer bound

\[
\boxed{M_{\mathrm{global}}\approx10^{53.60649}}
\]

on the exact-real frozen three-step continuation Hessian operator norm. This bound applies at **every stage-one state and every input direction**, including the tested radii. It excludes FP64 arithmetic rounding in the empirical numerical evaluation.

Its finite-radius remainder envelope is \(M_{\mathrm{global}}r^2\|u\|^2/2\). It is **far too large to certify** which intervention yields a larger final-logit effect at radii 0.02, 0.2 or 0.8. A computed sound bound is recorded separately from its practical tightness.

## SMC-08B: validated local Arb Taylor remainder

The existing LSV-17 [signed-adjoint Arb engine](../levin_sieve_v1/goal_taylor_v1/README.md) supplies verified weight spectral bounds via Arb LDL, interval-clean forward states and adjoins, and per-layer local curvature estimates along a transported input-radius tube. The new [SMC-08B wrapper](smc08_local_arb.py) specializes it to the active decoder atom of a trained seed-71 SAE at the same stage-one goal-token site, measuring the clean fixed winner-minus-runner margin for all 16 heldout puzzles.

The full [Arb Actions receipt](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43705858285) verifies the following **exact-real signed margin-displacement certificates**, with 160-bit Arb precision:

| Intervention radius | Signed response certified / 16 | LayerNorm global epsilon fallback |
|---:|---:|---:|
| 1e-10 | **15** | 0 |
| 1e-8 | **3** | 0 |
| 1e-6 | **0** | 0 |
| 1e-4 | **0** | 0 |
| 0.2 | **0** | 16 |

These certificates apply to the exact-real model with exact binary stored weights and the exact observed FP32 stage-one anchor. They bound a **scalar signed margin** along the selected direction and do not provide an interval-verified 10×48 Jacobian singular-value decomposition. At radius 0.2 every validated local tube becomes too broad to distinguish the margin response after LayerNorm denominator fallback.

The post-hoc [26-step rational bisection](smc08_local_arb.py) has now completed in a separate passing [hosted run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43706361253). It searches the downward-closed sufficient sign-certificate condition up to radius 1e-6. The [per-puzzle exact certificates and floating summaries](SMC08_ARB_RADIUS_FRONTIERS.json) establish a nonzero certified lower radius for all sixteen puzzles, ranging from **4.9174e-11** to **1.2328e-8**, with median **3.3440e-9**. The upper bracket records where this particular certificate stops proving the sign, rather than where the true recurrent margin changes sign. [MathCan OLM-08C](https://github.com/patrickdugan/MathCan/blob/research/olm08-validated-tube-rebased-20261115/research/orbit_lock/OLM08_VALIDATED_CAUSAL_TUBE.md) proves monotonicity of the sufficient condition under nondecreasing local derivative envelopes.

The bisection is an exploratory post-hoc extension to the preregistered radius grid. It documents the current proof's quantitative reach and remains far below the model's empirically informative radius 0.02.

## Interpretation and next gates

The SMC-07 causal effect is reproducible in this more detailed audit, and its small-radius behavior is largely aligned with the ordinary output Jacobian. The learned active SAE atoms pick useful high-sensitivity directions without approaching the locally optimal singular direction. Their semantic interpretation, causal necessity for reasoning, and benefit to Sudoku accuracy remain unverified.

The mathematically substantive remaining problem is to obtain **tight, directional, source-locked curvature enclosures** at radii around 0.02–0.2 rather than the very small thresholds where the current Arb bounds certify a sign. A localized interval-JVP/HVP propagation or a validated low-dimensional intervention tube could potentially reduce amplification through off-direction channels. This will need its own soundness proof, roundoff boundary and independent evaluation.
