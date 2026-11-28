# SHAFT-COMP-17: History-aware behavioral probes on unseen dynamic task families

**Registered November 11, 2026, before training new models or inspecting outcomes.** This is a reproducible toy-model mechanistic-interpretability experiment on the open SHAFT-COMP branch. The entire model substrate remains intact. No external devices, stealth or persistence mechanism is involved.

## Registered hypotheses

SHAFT-COMP-16 failed to generalize a four-query instantaneous behavioral probe to two previous-token-dependent task families. COMP-17 tests whether a probe trained to predict a **two-step counterfactual behavioral signature** improves the prediction of long-horizon disagreement between naturally reached hidden states. Both the probe family and its score are fixed before evaluating new model seeds.

For any frozen hidden state h, define the present signature B1(h) as the four native binary outputs obtained by applying each possible one-token query q in {0,1,2,3} to h. Define B2(h) as the 16 final-step native binary outputs after each ordered query pair (q1,q2). The history-aware linear probe W2 (16 × 24) is trained to predict B2(h), while the baseline linear probe W1 (4 × 24) predicts B1(h). This is behavior-supervised self-distillation. Both are trained from frozen model activations, without task-ID labels as probe targets. The original base GRU sees the task ID in its initial embedding.

## Two new task families and fixed seeds

Both families are distinct from the previously tested lag_rotate and lag_match. They use previous input x_(t−1), with sentinel x_−1=0.

- **xor_lag**: y_t = bit(g, x_t XOR x_(t−1)).
- **parity_gate**: y_t = bit(g, x_t) XOR ((x_t XOR x_(t−1)) mod 2).

Four model seeds per family, two calibration and two heldout: xor_lag calibration **1301,1302** and heldout **1303,1304**; parity_gate calibration **1401,1402** and heldout **1403,1404**. Earlier SHAFT seeds are excluded.

Reuse the 3,265-parameter TaskMemoryGRU architecture, 24 hidden units and four input tokens. Train each base GRU for **1,200 Adam steps**, batch 128, length 12, learning rate 0.01, clip gradient norm at 2.0. Use the existing independent 48 contexts per goal at each depth (4,8,12) to train both probes by BCE and Adam, **400 steps**, learning rate 0.04. Both probes use the same frozen activations, and separate seeded random initializations. Report fresh length-12 and length-32 task accuracy for every base model; include all models irrespective of accuracy.

## Pair generation and eligibility

For each model and recurrent depth 12 or 40, sample 16 independent pairs per each of 16 task IDs (256 pairs/depth/model). Members of each pair share a synthetic task ID but receive independent random prefixes. Eligibility is the **same rule used in COMP-16**: equal current native four-query *binary* signatures and equal W1-predicted four-bit signatures. W2 is not used for eligibility, so its ability to distinguish these presently equivalent states can be evaluated.

Sample a five-token diagnostic continuation, shared within each pair. The separate shared 32-token outcome continuation is independently seeded and never accessed by either score. The outcome Y=1 indicates disagreement of native four-query binary signatures at the end of the 32 steps.

## Frozen scores and test

Along the diagnostic trajectory, use each state's GRUCell Jacobian cocycle Φ_t, and stack W1 Φ_t and W2 Φ_t for t=0,...,5. For Δ=hA−hB, score

S1 = sqrt((||O1_A Δ||²+||O1_B Δ||²)/2)
S2 = sqrt((||O2_A Δ||²+||O2_B Δ||²)/2).

Larger scores are registered as predicting divergence. The **primary contrast** is pooled heldout ROC-AUC(S2) − ROC-AUC(S1). Both metrics use the same 24-D states, pairs, diagnostic inputs and recurrent Jacobians. Secondary measurements: norm-normalized S1 and S2, Euclidean state distance, absolute 4-bit versus 16-bit probe agreement with their frozen-model teacher signatures, each family and model AUC, and eligibility.

**Calibrated operating points:** After training and evaluating *calibration seeds only* (1301,1302,1401,1402), choose once for S1 and S2 the 90th percentile of all eligible calibration negative scores, with NumPy quantile method "higher". Predict divergence only if S > threshold. Freeze both numeric thresholds and their source false-positive counts in a separate Git commit before any heldout seed fitting or evaluation. Apply unchanged to both heldout families, reporting TPR, FPR, and all denominators. Also report family-specific calibration thresholds as descriptive controls.

**Confirmatory AUC decision:** require the paired bootstrap 95% CI for pooled AUC(S2)-AUC(S1) to be wholly positive, positive family-pooled differences for both families, and positive model-pooled differences in at least three of four new heldout seeds. All criteria must hold for "supported". The *operating-point transfer* claim additionally requires the fixed S2 FPR ≤0.15 in each family.

**Inference:** exactly 1,000 bootstrap draws resampling eligible pairs within family×model-seed×state-depth strata and 1,000 within-stratum outcome-label permutations for a two-sided centered test of the AUC difference. Report per-model and per-family results, even if they disagree. Paired bootstrap intervals describe pair-level uncertainty conditional on four trained models, not across architectures or populations.

## Receipts and scientific scope

Record full raw per-pair output rows, source-model/probe hashes, fresh training accuracy, probe-teacher agreement at depths 12 and 40, calibration thresholds, outcome labels and independent rerun SHA-256 values. Publish exact Mann–Whitney AUC pair counts and a dependency-free audit. Keep the PyTorch workload opt-in for CI.

The experiment uses known synthetic task IDs only to generate data and evaluate base-model accuracy. It does not demonstrate discovery of an unknown latent goal, detection of covert misalignment, distributed persistence, or reconstruction of erased weights.
