# SHAFT-COMP-14: task-sensitive causal observability, preregistered protocol

**Registered on 2026-11-11 before numerical evaluation.** The primary outcome from SHAFT-COMP-13 was negative (rank-deficit AUC 0.5044), and its seeds 303/404 must be excluded from COMP-14's new holdout. All experiments use consented synthetic GRUs. There is no device persistence, hidden objective retrieval, or covert exfiltration.

## Hypothesis

The rank of a monitor's linearized observation stack discards the direction and magnitude of the difference between two actual hidden states, together with their proximity to task decision boundaries. A *task-conditioned directional Gramian* might predict delayed divergence between naturally reached states sharing the same current behavioral signature.

This hypothesis is **distinct** from Q5's registered F1/F8 gradient-fragility criterion. Q5 is retained as historical context, without implying a direct model-to-model comparison.

## Fixed design

- Same 3,265-parameter TaskMemoryGRU and frozen four-output probe code as COMP-13. Train on goal IDs 0..15 with fresh random task inputs.
- **Calibration model seeds:** 101 and 202; **new held-out model seeds:** 505 and 606. Seeds 303 and 404 are reserved as the completed COMP-13 test and excluded from this study.
- Recurrent state depths: 12 and 40. Sample 16 pairs per each of 16 goals, independently generated natural prefixes for each pair member, giving 256 pairs per seed/depth.
- Eligibility: both states share all four *native* current counterfactual binary outputs **and** all four thresholded learned-probe outputs. Eligibility and the outcome must be computed independently of the diagnostic inputs.
- **Diagnostic schedule:** one five-token random continuation shared by both pair members, seeded independently of the state prefixes and outcome inputs.
- **Outcome schedule:** independent common 32-token continuation. Label Y=1 if the two states' *native* four-query output signatures disagree after 32 recurrent steps. Do not supply any outcome tokens when computing scores.
- Exact paired state difference \(\Delta=h_A-h_B\). At diagnostic time \(t\), define the state-transition cocycles \(\Phi_{s,0}=I\), \(\Phi_{s,t+1}=J_{s,t}\Phi_{s,t}\) and the **native behavioral logit Jacobian** \(C_{s,t}=\partial z_{s,t}/\partial h_{s,t}\), where \(z\in\mathbb R^4\) are the actual GRU readout logits on four counterfactual query tokens (including the query step).
- Task boundary weight for each query logit is \(w(z)=1/(1+|z|)\). Its value is fixed here before fitting or testing.
- **Primary task score:** \(S_T = [ \tfrac12 \sum_{s\in\{A,B\}} \sum_{t=0}^{5} \|\operatorname{diag}(w(z_{s,t})) C_{s,t}\Phi_{s,t}\Delta\|_2^2 ]^{1/2}\). Larger score is predicted to mean higher probability of delayed behavioral divergence. This is the directionally evaluated, task-boundary-weighted finite-horizon observability Gramian, and is nonnegative. Compute using the actual four query logits and analytic GRU Jacobians.
- **Unweighted directional control:** same formula with \(w\equiv1\), AUC interpreted as positive direction.
- **COMP-13 matched baseline:** same horizon, state depths, diagnostic schedules and threshold \(10^{-3}\), with \(S_R=24-(r_A+r_B)/2\), where rank is from the learned *linear probe* W, evaluated at relative singular-value threshold \(10^{-3}\).
- **Other controls:** negative current native output-margin proxy, negative current probe margin, and Euclidean state distance. Keep their orientations fixed.
- Before evaluation, verify analytic native-output Jacobians against autograd on two states (maximum absolute difference < 1e-5), including the counterfactual query transition. Record this parity value.
- Primary analysis: pooled held-out ROC-AUC of S_T on eligible pairs from seeds 505 and 606, with no recalibration from held-out outcomes. Report baseline and task-score AUCs per seed and depth, plus pooled.
- Uncertainty: 1,000 within-seed/depth label permutations and 1,000 stratified pair bootstrap draws, 95% percentile interval for both S_T and its paired difference from the matched S_R AUC. These describe uncertainty within the two tested models only.
- AUC improvement over rank is measured without choosing an empirical threshold. Treat failure to exceed rank as a negative finding. Include calibration outcomes verbatim and no further feature search on held-out seeds.
- Preserve per-pair data, model hashes, environment versions, PR commit, all seeds and negative results. Full training is opt-in and light CI validates the frozen receipts using standard-library tests.

## Scope

This benchmark evaluates predictive sensitivity in a small recurrent neural network with known synthetic tasks. It cannot establish hidden misalignment, autonomous reconstruction of erased weights, or the persistence of an agent on distributed devices. Future transfer to Q5 needs independently verified original Q5 checkpoints and a distinct, predeclared projected-state measurement.
