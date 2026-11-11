# SHAFT-COMP-19: Six-symbol replication and equal-capacity behavioral-probe controls

**Preregistered November 11, 2026, before fitting the COMP-19 models or reading their outcome labels.** This follows the COMP-18 six-symbol results. The entire test runs in memory on controlled synthetic task models; it does not involve devices, network access, covert deployment or weight recovery.

## Motivation and known limitations

COMP-18 compared an immediate V-output probe against a V²-output two-query probe across GRU/LSTM and vocabularies V=4/6. At V=6, the GRU holdout had only 3 positive events among 92 eligible pairs, while the LSTM had 32 events among 187. That makes per-cell AUC estimates unstable. Output dimensionality is also a confound: V² outputs provide more parameters than V outputs. This study increases samples and separates output capacity from history-specific supervision.

## Task-family and architecture factorial, fixed before training

Use V=6, 64 synthetic six-bit task IDs, the unchanged 24-dimensional recurrent hidden size, learned goal-state embeddings and native two-layer binary task readout.

* **lag_rotate**: previous-input rotated lookup, y_t = bit(g,(x_t + x_(t-1)) mod 6), with previous-input sentinel 0.
* **lag_match**: previous-input equality gate, y_t = bit(g,x_t) XOR 1[x_t = x_(t-1)], with previous-input sentinel 0.

GRUCell has a 24-dimensional state; LSTMCell includes both hidden and cell values for a 48-dimensional state. Base-model training is fixed at Adam 1,600 steps, batch 128, length 12, LR 0.01, gradient clip 2, using the COMP-18 task architecture.

**Model seeds, all newly assigned:**

| Cell | Calibration | Heldout |
|---|---:|---:|
| GRU / lag_rotate | 1901 | 1911 |
| LSTM / lag_rotate | 1902 | 1912 |
| GRU / lag_match | 1903 | 1913 |
| LSTM / lag_match | 1904 | 1914 |

Fit three separate linear probes to *the same* frozen recurrent activations at depths 4,8,12, 12 independent contexts per goal per depth (2,304 training contexts), with 400 Adam steps at LR 0.04, BCE loss and independently fixed random initialization:

1. **W1**, 6 outputs, trained to predict each of six native one-query output bits.
2. **Wdup**, 36 outputs, trained on the six immediate-output bits *repeated six times* in fixed query order. This is the equal-output-count capacity control.
3. **W2**, 36 outputs, trained on the 36 native two-query *final-step* output bits.

The Wdup target repeats the immediate signature and contains no new behavioral information. Wdup and W2 have the same parameter count (36×state dimension plus 36 biases), optimizer, activation data and number of training steps.

## Sampling and endpoint

Each model is evaluated at recurrent state depths 12 and 40. Sample 64 goals × 16 independent pairs per goal at each depth (1,024 pairs/depth/model, 2,048 per model). Each pair has the same goal but independently generated natural input prefixes.

Eligibility retains the COMP-18 rule: the two states must have identical *native six-query immediate binary signatures* and identical W1-predicted six-bit signatures. Wdup and W2 are **never used for eligibility**.

**Delayed divergence endpoint changed and frozen here:** Each eligible pair receives **two independently generated** shared 32-token future input continuations. On each continuation inspect native six-query signature disagreement at steps 8,16,32. Define Y=1 if any of these six future observations differs between the pair. This is a multi-future delayed-divergence event, distinct from the COMP-18 single step-32 outcome; this preregistered definition increases the number of informative positives while excluding trivially immediate 1- and 2-step events.

Features use a single independent five-token diagnostic continuation shared within each pair and never access either outcome sequence. The tangent displacement Δ is the difference between the states at the start. Use analytic JVP through the frozen GRU/LSTM cell and compute directional energy along diagnostic steps 0..5.

**Primary score, frozen:** S2 = sqrt[(||O2_A Δ||² + ||O2_B Δ||²)/(2×36)], where O2 stacks W2 Phi_t along diagnostic steps. **Control score:** Sdup is the identical formula with Wdup and identical 36-output normalization. Larger score always predicts Y=1. Additional registered controls: S1 from W1 (normalized by 6), norm-normalized S2/Sdup, and Euclidean state distance. No feature tuning after test labels.

## Calibration gate and statistical decisions

Execute all four calibration models first. Select the 90th percentile among **pooled calibration eligible negatives** separately for S2 and Sdup, with NumPy quantile method "higher" and strict greater-than prediction. Freeze numeric thresholds, source-negative counts and source model hashes in a separate Git commit **before any COMP-19 heldout model is fitted**. Also record per-cell calibration negative thresholds as secondary references.

**Primary test:** pooled heldout ROC-AUC(S2) − ROC-AUC(Sdup) over the four heldout models. Report exact positive/negative pair wins/ties, per-model/cell AUCs, exclusions, and source-to-heldout FPR/TPR at frozen thresholds.

Bootstrap eligible pairs 1,000 times within each architecture×task-family×starting-depth cell to obtain the paired 95% percentile interval for the AUC difference. Conduct a two-sided 1,000-permutation test, permuting labels within the same cells and centering on the stratified permutation null mean. Both analyses are conditional on the four fixed trained models.

Primary support requires (a) paired bootstrap lower bound > 0, and (b) at least 3 of 4 heldout model cells have positive S2−Sdup AUC difference. **Fixed-FPR transfer** is a separate criterion: S2 heldout FPR <=0.15 in each of the four cells, at the frozen pooled calibration threshold. Report failure even if AUC improves.

Measure base-model heldout task accuracy on fresh 12-token and 32-token sequences; retain weak models in the report. Probe-teacher bit agreement, exact model/probe SHA-256 hashes and independent replay checks are required. Use CPU PyTorch 2.10.0+cpu, NumPy 2.3.5, deterministic operations and 3 CPU threads. Full per-pair data and executable scripts are archived; light CI verifies syntax, finite references, frozen receipts and Lean statements without expensive retraining.

## Scope

All task IDs are explicitly supplied in synthetic base-model training, and the original model weights are intact. No conclusion about hidden goals in frontier agents, covert persistence, distributed deployment, or erased-weight resurrection follows from this test.
