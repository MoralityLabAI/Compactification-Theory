# SHAFT-COMP-17: History-aware counterfactual codes across two temporal task families

**Preregistered:** November 11, 2026, [fixed protocol](SHAFT_COMP_17_PROTOCOL.md), commit 938560c3cad9d2d8cf298f349311900a2510e0fe. The calibration thresholds were [committed separately](results/comp17_calibration_thresholds.json), commit 603eb2640f5f4f98c99630495249d005c9dd2729, **before fitting any held-out models**.

**Primary decision:** supported within this isolated synthetic benchmark. A 16-output learned probe trained on **two-step counterfactual behavioral signatures** achieved held-out ROC-AUC **0.660504** for delayed native behavior divergence, versus **0.557042** for the established four-output immediate-response probe. The paired AUC improvement was **+0.103462**, 95% within-model bootstrap interval **[+0.019129,+0.194882]**, stratified permutation p **0.023976**. Both task families and all four new model seeds had positive AUC differences.

The fixed-history-probe threshold retained FPR below 15% in both families, meeting the separately registered FPR transfer criterion. At the pooled operating point it detected **6 of 31** divergence events, with **20 false positives among 492 negatives**. Detection sensitivity remains limited.

## 1. Experiment and temporal tasks

Earlier SHAFT-COMP-16 found that a four-output probe trained on immediate native query signatures generalized poorly to sequence-dependent task families. COMP-17 tests new, independently trained small GRUs whose target output depends on the previous token. The two families are distinct from the earlier lag_rotate and lag_match tasks.

For current input x_t in {0,1,2,3}, previous input x_(t−1) (initially zero), and one of sixteen four-bit task tables g:

- **xor_lag:** y_t = bit(g, x_t XOR x_(t−1)).
- **parity_gate:** y_t = bit(g, x_t) XOR ((x_t XOR x_(t−1)) mod 2).

Both need temporal context during execution. For a hidden recurrent state h, let B1(h) give the native model's four one-query binary responses. Let B2(h) give the native model's *final* binary response after each of the sixteen ordered two-query continuations (q1,q2). The two-query signature is conditioned on a fresh intermediate query, so in the exact finite task specification it identifies g independently of the token that preceded q1. This makes it a meaningful test of **goal-related information under counterfactual history changes**.

The original base GRUs receive the explicit synthetic goal as an initial learned embedding. Their one-step 4-output and two-step 16-output probes receive *frozen activations* and are trained against the original GRU's own counterfactual responses, not direct task IDs. The trained models remain entirely intact.

The [exact 64-state reference](temporal_probe_reference.py) enumerates sixteen goals and four preceding-token states for each family. It finds **72** immediate-alias state pairs separated by two-query signatures in xor_lag, and **64** such pairs for parity_gate. The one-query and two-query signature maps both have sixteen distinct output patterns, although they partition the 64 physical states differently. [HistoryProbe.lean](lean/Shaft/HistoryProbe.lean) kernel-proves a smaller Boolean witness: two different goal/history states agree on every immediate query but can be distinguished by a two-step continuation. These are finite mathematical examples, distinct from trained neural results.

## 2. Fixed training, held-out families and evaluation endpoint

Eight independently initialized 3,265-parameter TaskMemoryGRUs have 24-dimensional hidden state and four input symbols. Each was trained for 1,200 Adam steps, batch 128, sequence length 12, learning rate 0.01, gradient clipping 2.0. The probes used independent training activations at recurrent depths 4, 8, and 12, with 48 contexts per goal at each depth; each linear probe trained for 400 steps with BCE and Adam at rate 0.04.

| Family | Calibration model seeds | Held-out model seeds |
|---|---|---|
| xor_lag | 1301, 1302 | 1303, 1304 |
| parity_gate | 1401, 1402 | 1403, 1404 |

All eight base models reached **100% fresh length-12 task accuracy**; their length-32 task accuracies ranged from approximately 97.2% to 99.4%, depending on the seed.

At recurrent depths 12 and 40, each model generated 16 goals × 16 same-goal state pairs, each with independent natural input prefixes. Eligibility required the two states to agree on their **immediate native four-query outputs** and on the original **four-output probe code**. The history-aware probe was not used to select eligible pairs.

A fresh *shared five-step diagnostic input sequence* was used to construct the recurrent-state Jacobian cocycles for both states. Independently sampled shared 32-step **outcome inputs** tested whether the original GRU's four native query-response bits diverged at the end. Outcome inputs and labels were withheld from feature construction.

For pair displacement Δ=h_A−h_B, compare the predeclared directional scores

\[
S_1^2=\tfrac12\sum_{s\in\{A,B\}}\sum_{t=0}^5\|W_1\Phi_{s,t}\Delta\|_2^2,\qquad
S_2^2=\tfrac12\sum_{s\in\{A,B\}}\sum_{t=0}^5\|W_2\Phi_{s,t}\Delta\|_2^2.
\]

W1 is the frozen 4×24 immediate behavioral probe; W2 is the frozen 16×24 two-step behavioral probe. Each uses the *same model state pairs and recurrent Jacobians*. Larger S1 or S2 was preregistered as predicting greater chance of delayed native-output divergence.

## 3. Registered discrimination results

| Split | Generated pairs | Eligible | Future divergences | Immediate probe AUC | Two-step probe AUC |
|---|---:|---:|---:|---:|---:|
| Calibration (four models) | 2,048 | 492 | 55 | 0.583982 | 0.732640 |
| **Held-out (four new models)** | **2,048** | **523** | **31** | **0.557042** | **0.660504** |

Held-out per family:

| Family | Eligible pairs | Divergences | Immediate AUC | Two-step AUC | Improvement |
|---|---:|---:|---:|---:|---:|
| xor_lag | 312 | 19 | 0.562781 | 0.617029 | +0.054248 |
| parity_gate | 211 | 12 | 0.556533 | 0.733250 | +0.176717 |

Each held-out seed individually had a positive AUC improvement:

| Seed | Eligible | Divergences | Improvement |
|---|---:|---:|---:|
| 1303 | 139 | 11 | +0.065341 |
| 1304 | 173 | 8 | +0.061363 |
| 1403 | 109 | 8 | +0.141089 |
| 1404 | 102 | 4 | +0.255102 |

The pooled paired bootstrap 95% interval for AUC(S2)−AUC(S1) was **[0.019129,0.194882]**, based on 1,000 stratified pair resamples within family×seed×depth cells. A 1,000-permutation test shuffling labels within the same strata and centering on its observed null mean (0.003247) gave two-sided p **0.023976**. These uncertainty estimates condition on **four fixed held-out GRUs**. With only **31 positive events**, the effect magnitude and cross-model generalization remain uncertain.

Normalization by initial pair distance gave held-out AUC **0.710923** for the two-step probe and **0.610477** for the immediate probe. Raw Euclidean pair distance alone gave 0.453908. The normalized results were prespecified secondary measurements and were not substituted for the primary outcome.

## 4. Fixed FPR operating point

The numeric thresholds were computed using only calibration negatives, with NumPy's 90th percentile using method="higher" and a **strictly greater-than** detection rule. Source threshold values and source false-positive numerators were committed before training the four held-out models.

| Probe | Frozen threshold | Calibration FPR | Held-out FPR | Held-out TPR |
|---|---:|---:|---:|---:|
| Immediate S1 | 12.010500 | 43/437 (9.84%) | 40/492 (8.13%) | 3/31 (9.68%) |
| Two-step S2 | 18.488289 | 43/437 (9.84%) | **20/492 (4.07%)** | **6/31 (19.35%)** |

For xor_lag, the frozen two-step threshold produced 10 false positives among 293 negatives (3.41%) and detected 4 of 19 divergences (21.05%). For parity_gate, it produced 10 false positives among 199 negatives (5.03%) and detected 2 of 12 divergences (16.67%).

Both family false-positive rates met the preregistered 15% maximum. These detection rates are low enough that a monitoring application would need additional signals and larger positive-event samples.

## 5. Post-hoc capacity-matched ablation

**This analysis was declared after the registered result.** The [separate amendment](SHAFT_COMP_17_POSTHOC.md) specifies a capacity-matched 16-output control. Instead of predicting sixteen two-query outcomes, the control trains a 16×24 linear readout on the *four immediate teacher query bits duplicated four times*. Training examples, optimizer, step count and output dimension match W2, with an independently seeded probe initialization.

The control reuses the original held-out model weights, state pairs, eligibility and outcome labels. Its results are **exploratory**:

| Learned readout | Held-out ROC-AUC |
|---|---:|
| Immediate four-output probe W1 | 0.557042 |
| Capacity-matched sixteen-output duplicate-immediate probe | **0.556517** |
| History-aware sixteen-output two-step probe W2 | **0.660504** |

The exploratory two-step-minus-duplicate AUC difference is **+0.103986**, with a paired within-model bootstrap interval **[0.012786,0.190357]** and permutation p **0.010989**. Both temporal task families retain a positive difference. The duplicate-immediate control closely matches the four-output baseline, suggesting that merely increasing output count and linear-probe parameters is insufficient to reproduce the observed gain.

Independent probe initialization and fit quality remain possible confounders, and the post-hoc confidence interval is not the registered confirmatory test.

## 6. Evidence and reproduction

- [Registered protocol](SHAFT_COMP_17_PROTOCOL.md) and the [calibration threshold freeze](results/comp17_calibration_thresholds.json), both preceding held-out evaluation.
- [Exact experiment runner](experiments/history_aware_transfer.py), which accepts separate calibration and held-out phases, and requires a previously frozen calibration JSON for held-out runs.
- [Compact result](results/comp17_summary.json), containing both splits, exact Mann–Whitney AUC pair counts, per-family results, seed signs, frozen operating points and full-receipt hashes.
- [Exact finite reference](temporal_probe_reference.py), [Lean witness](lean/Shaft/HistoryProbe.lean), and [pure-standard-library AUC/threshold audit](history_probe_audit.py).
- [Post-hoc control script](experiments/history_capacity_control.py), [amendment](SHAFT_COMP_17_POSTHOC.md) and [frozen post-hoc summary](results/comp17_posthoc_summary.json).

Run from repo root, with PyTorch 2.10.0+cpu and NumPy 2.3.5:

\`\`\`bash
python -m research.shaft_compactification.experiments.history_aware_transfer \
 --phase calibration --per-goal 16 --training-steps 1200 \
 --out /tmp/comp17_calibration.json
# Recompute a threshold receipt from the calibration results and verify it
# matches the committed, already-frozen threshold JSON.
python -m research.shaft_compactification.experiments.history_aware_transfer \
 --phase heldout --per-goal 16 --training-steps 1200 \
 --calibration-receipt research/shaft_compactification/results/comp17_calibration_thresholds.json \
 --permutations 1000 --bootstraps 1000 --out /tmp/comp17_heldout.json
python -m research.shaft_compactification.temporal_probe_reference
python -m research.shaft_compactification.history_probe_audit
\`\`\`

Both complete raw calibration and held-out experiments were independently run twice and produced **byte-identical JSON receipts** under the pinned local Python environment:

- Calibration SHA-256: \`f07496403ef26dff938b6d9c87030ac245ae0e784bcc098a1c57edce39540790\`
- Held-out SHA-256: \`aa1a2cd4c561af5828bb8559b8c0a8fde77123c5fa8c2a5f49dce86333791403\`

The separate full raw post-hoc result has SHA-256 \`1163d3b9044227650fac1d9eaeb77c0a46c67f7e8310dd71ed90dc6972d7bb0c\`. The full data and source snapshots are attached as an archival ZIP in the conversation; GitHub retains readable source, protocol, and compact frozen receipts. Lightweight CI runs the finite tests, audits summary receipts and compiles Lean 4.28 without retraining neural models.

## Scientific boundary

The result supports a narrow, falsifiable claim that **two-query counterfactual behavioral targets can improve a directional monitor under history-dependent task dynamics** in small synthetic GRUs. It does not prove that a learned 16-bit code is a causally sufficient, exact representation of all future behavior. The probes retain many real-valued fitted parameters, and the underlying networks still have their complete goal embeddings and weights.

The original Shaft motivation concerned whether objective-specific information might survive model deletion or spread across devices. No such mechanism, observation or experiment appears here. The next useful generalization test would move beyond the same GRU architecture and four-bit goal vocabulary, using unrelated trained task families and a registered monitoring-operating point.
