# SHAFT-COMP-15: Readout × statistic factorial observability study

**Preregistered:** 2026-11-11 before fitting or evaluating the new holdout seeds. This protocol is frozen on the `research/shaft-comp-05` branch of Compactification-Theory. Every outcome is synthetic, in memory. No distributed persistence, surveillance evasion, or model recovery mechanism is evaluated.

## Motivation

SHAFT-COMP-13 found rank-deficit ROC-AUC 0.504 on held-out seeds 303/404. SHAFT-COMP-14 improved to AUC 0.653 for a task-weighted *native-output directional* sensitivity on held-out seeds 505/606, against learned-readout rank-deficit AUC 0.481. Because the feature changed both the readout and the statistic, a 2×2 factorial study is needed to isolate those changes. The unweighted native directional score reached 0.655, motivating an **unweighted** directional primary metric.

## Sampling and eligibility

- Same frozen synthetic TaskMemoryGRU architecture and training recipe from SHAFT-COMP-06: 3,265 parameters; 24 hidden units; 16 four-bit truth-table goals; fixed train procedure `fit(seed)` and four-output predictive probe `train_probe(model,seed)`.
- Existing calibration/engineering seeds: **101, 202**. New held-out model seeds: **707, 808, 909, 1001**. Previous COMP-13/14 holdout seeds **303,404,505,606** are forbidden for choosing features or cutoffs in COMP-15.
- State start depths 12 and 40. For each model/depth, use 16 goals × 16 naturally reached pairs per goal = 256 pairs, total **2048 new held-out pairs**.
- Generate two independent state prefixes with the same goal per pair, as in COMP-14. Current native four-query binary outputs and four learned-probe bits must agree for eligibility. Report all exclusions and label prevalence.
- Diagnostic input sequence length **5**, one seeded random continuation *shared by each pair's two members*. The outcome is a separately seeded, shared **32-step** continuation with native four-query binary divergence as (Y=1). No diagnostic feature may use the outcome tokens or outcome labels.

## The 2×2 candidate scores

For model s ∈ {A,B}, let (Delta=h_A-h_B), (Phi_{s,0}=I), (Phi_{s,t+1}=J_{s,t}Phi_{s,t}), where the Jacobian is evaluated along that member's diagnostic trajectory.

Two readouts:

- **Learned:** (C^{L}_{s,t}=Winmathbb R^{4	imes24}), the four-output linear predictive probe weight, held fixed along the diagnostic trajectory.
- **Native:** (C^{N}_{s,t}=partial z_{s,t}/partial h_{s,t}), derivative of all four original model counterfactual logits with respect to the recurrent state, including their single query-step GRU transition and nonlinear readout.

For each readout (C), define (mathcal O^C_{s,5}) as the vertical stack of (C_{s,t}Phi_{s,t}) for t=0..5.

Two statistics, all readouts in continuous logit units:

1. **Rank deficit** (R_C=24-	frac12sum_{sin{A,B}}r_{10^{-3}}(mathcal O^C_{s,5})), numerical rank at *relative singular-value threshold 1e-3*. Higher deficit predicts more divergence, preserving the previous COMP-13 direction.
2. **Unweighted directional sensitivity** (D_C = [	frac12sum_s |mathcal O^C_{s,5}Delta|_2^2]^{1/2}). Higher score predicts more divergence. No boundary weighting.

This yields four prespecified cells: `learned_rank`, `learned_direction`, `native_rank`, `native_direction`. The registered primary contrast is **AUC(learned_direction) − AUC(learned_rank)** on pooled eligible new holdouts. This isolates the effect of directional information at a fixed learned readout. The native-direction-minus-native-rank comparison is secondary, as are readout changes at fixed statistic and the 2×2 interaction.

## Prespecified controls and statistics

- Extra baseline: raw Euclidean pair distance (|Delta|), higher presumed more divergence.
- Secondary, descriptive: normalize each directional score by (|Delta|) to test the contribution of state-difference magnitude, using 1e-9 denominator floor.
- Verify analytic native-output Jacobians against autograd for every model (<1e-5 maximum absolute difference); use the existing COMP-12 analytic recurrent-state Jacobian.
- Confirm no NaNs/Inf in features, no zero-length eligible groups, and that both statistics from the same readout use the *same exact* (mathcal O^C) and diagnostic trajectory. Keep float64 SVD for rank computation.
- Compute per-model × starting-depth ROC-AUC and pooled ROC-AUC, including sample counts and positive events. Constant/single-class AUC is explicitly undefined.
- Use **1000** label permutations within the model-seed × depth strata for a two-sided test of the primary AUC **difference**, centered on the stratified null mean. Use **1000** bootstrap draws resampling eligible pairs *within* each stratum for a 95% percentile interval on primary difference and the four ROC-AUC values. These intervals measure within-model variability; four model seeds remain the unit of model replication.
- Record seed-specific contrasts, and a leave-one-model-out comparison. The registered directional hypothesis is considered **supported in this synthetic benchmark** only if the paired bootstrap difference CI is wholly positive **and** at least 3 of 4 held-out models have positive seed-specific difference. Otherwise report negative/inconclusive as appropriate; do not flip metric orientation or adjust the threshold.
- Historical COMP-14 directional native AUC 0.6547 is not a valid benchmark on the new states; all comparisons must use the same COMP-15 held-out pairs.

## Reproducibility

Reproduce full raw JSON and an independently repeated run using CPU PyTorch 2.10.0+cpu and NumPy 2.3.5, `torch.use_deterministic_algorithms(True)`, three CPU threads, fixed per-seed RNG states, no web requests. Pin hashes of base models/probes and experiment receipts; publish a compact JSON summary in GitHub and full per-pair rows in a ZIP artifact. CI checks syntax, finite mathematical examples, exact Mann–Whitney AUC accounting, and Lean theorems, without retraining these models each PR.

Future transfer to LOOPSCOPE Q5 requires original hash-verified Q5 checkpoints, which are absent from the current accessible branch. Q5's F1/F8 parameter-gradient fragility and the proposed state-output observability measurements remain separate constructs.
