# SHAFT-COMP-18: Cross-architecture and six-symbol history-probe transfer

**Registration:** November 11, 2026, before running COMP-18 calibration or held-out neural training.
**Prior evidence:** COMP-17 achieved a positive result in two four-token, GRU-only temporal families. This experiment tests an architecture × vocabulary change, with frozen inputs, scores, and success criteria.

## Task and models

Use a **previous-symbol rotated lookup** objective for alphabet size V in {4,6}. A goal g in {0,...,2^V-1} specifies the V binary outputs; on input x_t with preceding input p_t and p_0=0, return bit(g,(x_t+p_t) mod V). This is sequence-dependent for both vocabularies.

Two recurrent architectures:
1. GRUCell, recurrent state dimension 24, output hidden dimension 24.
2. LSTMCell, 24 hidden and 24 cell units, flattened recurrent state dimension 48. The current hidden h supplies the output head; the cell c remains part of causal state.

Both have learned initial state embeddings per goal, one-hot input of dimension V, and the same two-layer 24-wide tanh binary answer head. The parameter count differs across architectures and vocabulary sizes, so comparisons are within the matched architecture/vocabulary cell.

Train **each** model for 1,600 Adam steps, LR=0.01, batch=128, input sequence length=12, gradient clip=2.0, from fixed seeds. Train two linear probes, from independent naturally reached states at depths 4,8,12 (32 prefixes per goal per depth, or at most 2,304 total contexts per probe):

- Immediate W1: predict V native one-query binary responses.
- History W2: predict V^2 native *final* responses to all ordered two-query continuations.

Use identical probe training data, separate seeded initializations, 400 Adam steps at LR=0.04, binary cross-entropy. The probes see only frozen recurrent states as inputs and native model behavior as targets, with no task IDs in their supervision. Record held-out base-model accuracy at lengths 12 and 32 before interpreting the scientific outcome. Weakly trained models remain in the data, flagged.

Four architecture/vocab cells and fixed model seeds:

| Cell | Calibration seed | Held-out seeds |
|---|---|---|
| GRU / V=4 | 1811 | 1812,1813 |
| GRU / V=6 | 1821 | 1822,1823 |
| LSTM / V=4 | 1831 | 1832,1833 |
| LSTM / V=6 | 1841 | 1842,1843 |

Prior COMP-17 task/seed holdouts are excluded. Training seeds, target function, and model implementation are frozen by this registration.

## Paired-state test

For each model and starting recurrent depth 12 or 40, sample **256 natural same-goal pairs** with separately seeded input prefixes. Balance by goal: 16 goals × 16 pairs for V=4; 64 goals × 4 pairs for V=6. Eligible pairs must have identical native V-query binary responses **and** identical W1 predicted V-bit outputs at the start, matching the COMP-17 eligibility principle. W2 does not select eligibility.

Apply five diagnostic tokens common to both pair members, independent of state histories and outcome tokens. Compute forward-mode Jacobian-vector products of the recurrent transition along both trajectories, applied to the original state contrast delta=hA−hB. At each of t=0...5 monitor the linear probe response Wk Phi_t delta. Use the score **Sk = sqrt((sum_{state=A,B}sum_{t=0}^5 ||Wk Phi_t delta||²)/(2*output_dimension_k))**, with k=1 immediate and k=2 two-step. Dividing by output dimension standardizes comparison across V and V² probe outputs. Higher scores predict delayed divergence. The recurrent transition derivative can be computed analytically from GRU/LSTM gates; verify each against torch.func.jvp on a held-out state with tolerance <1e-5.

For the outcome, independently sample one shared 32-token continuation and compare the two native **V-query final** binary signatures after it. Y=1 when at least one bit differs. No diagnostic feature may access outcome tokens or labels.

## Calibration and frozen operating points

Run calibration seeds only first. Select a common global threshold for W1 score and for W2 score using the **90th percentile of pooled eligible calibration negatives** (NumPy quantile method='higher'), predicting Y=1 iff score is strictly greater. Record fixed numeric thresholds in a distinct Git commit **before fitting any held-out model**. Additionally choose per-cell calibration thresholds as *secondary* controls; the global thresholds are the registered transfer test.

The primary result is held-out ROC-AUC(W2 score) minus ROC-AUC(W1 score), across all four held-out cells. For uncertainty, compute exactly 1,000 label permutations and 1,000 stratified paired bootstraps within architecture×vocab×model-seed×depth, with a two-sided centered permutation test for the primary AUC difference. Also report per-cell and per-seed AUCs, positive counts, probe-teacher accuracy, and normalized score dispersion.

**Support criterion:** the pooled paired-bootstrap 95% interval for the AUC difference is wholly positive, at least 3/4 architecture/vocab cells have positive AUC differences, and at least 6/8 held-out model seeds have positive AUC differences. If any criterion fails, classify the primary claim as inconclusive or negative without changing directions or dropping poor models. For **fixed-FPR transfer**, separately require global W2 threshold held-out FPR <=0.15 in each of the four cells. Report this regardless of the primary AUC result.

## Reproducibility and limits

Use CPU PyTorch 2.10.0+cpu and NumPy 2.3.5, deterministic algorithms and fixed random generators. Record checkpoint hashes, per-pair outputs, exact Mann–Whitney AUC comparisons, binary output and probe calibration accuracy, all exclusions and four-cell operating-point results. Two independent reproducible full runs should agree byte-for-byte. Train scripts are opt-in; light CI checks source syntax, exact finite references and Lean statements.

This is an in-memory synthetic-model study. No distributed device work, real agent persistence, covert behavior, unknown latent malicious goal, or recovery of deleted model weights is attempted.
