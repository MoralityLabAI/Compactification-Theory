# SHAFT-COMP-16: Cross-family observability and fixed false-positive transfer

**Preregistered:** November 11, 2026, before running the new tasks, in [SHAFT_COMP_16_PROTOCOL.md](SHAFT_COMP_16_PROTOCOL.md), commit `83a436d3e301002491655a1d96af7bf1df88e4a7`.

**Decision:** Cross-family transfer **not supported** under the registered criteria. Directional observability performed at AUC **0.472189** on the history-rotated lookup task and **0.662162** on the history-match/XOR task. The original directional threshold produced **51.8% false positives** on history-rotated lookup. An aggregate improvement over rank was insufficient to satisfy the preregistered family-specific and bootstrap rules.

All experiments use 3,265-parameter synthetic GRUs with intact trained weights. The study provides no evidence for covert objectives, model-weight erasure survival, or devices that can autonomously restore agents.

## Task dynamics and exact history-dependence

Every earlier COMP-06–15 GRU was trained on a four-bit truth-table task in which the target at time t depends on the current token alone. COMP-16 uses the same four-token vocabulary and four-bit task IDs, with explicit dependence on the previous input (initial sentinel x[-1]=0):

- **lag_rotate:** y[t] = bit(g, (x[t]+x[t-1]) mod 4).
- **lag_match:** y[t] = bit(g, x[t]) XOR indicator(x[t]=x[t-1]).

The GRU must therefore track recent query history as well as the task identifier. The [pure Python reference](temporal_family_reference.py) enumerates all sixteen task IDs and four current queries. Under lag_rotate, 56/64 goal/current-query combinations yield different answers for different previous inputs; under lag_match, 64/64 do so. The [Lean proof](lean/Shaft/TemporalFamily.lean) shows that a goal-only code cannot decode all next-step answers in the one-bit match analogue.

Eight fresh models use the existing 24-hidden-unit TaskMemoryGRU and identical learned-behavioral-probe architecture, with 1,200 Adam updates, batch size 128, training length 12, and learning rate 0.01. The probe is fitted on native counterfactual query responses at recurrent depths 4/8/12, without receiving explicit task ID labels as targets.

## Registered evaluation

Calibration seeds: lag_rotate 1101/1102; lag_match 1201/1202. Held-out seeds: **1103/1104** for lag_rotate and **1203/1204** for lag_match. None appeared in earlier SHAFT tests.

Each model/depth (12 or 40) uses sixteen independently sampled pairs for each of sixteen goals. The two members share their task but have independent prefixes. Eligibility requires equality of all four present native response bits and all four learned-probe bits. The new features use an independent *common five-token diagnostic continuation* to construct each state's GRU Jacobian observation stack O_5. A separately seeded, common 32-token continuation supplies the binary divergence label, defined as native disagreement among the four counterfactual responses after that continuation.

At rank tolerance 10^-3, the frozen COMP-15 features are:

\[
D_L = \sqrt{\frac{\|O_A\Delta\|^2 + \|O_B\Delta\|^2}{2}}, \quad
R_L = 24-\frac{\operatorname{rank}(O_A)+\operatorname{rank}(O_B)}{2},
\qquad \Delta=h_A-h_B.
\]

Higher score is predicted to mean higher divergence probability. The source-calibration thresholds were determined **before** training the new models from the 759 original COMP-15 calibration negatives, as the NumPy 90th percentile (`method='higher'`) with a *strict greater-than* decision:

| Feature | Fixed source threshold | Source FPR |
|---|---:|---:|
| Learned directional | 6.854760490520719 | 75/759 = 9.88% |
| Learned rank deficit | 2.5 | 68/759 = 8.96% |

New-family calibration could supply a second threshold, but cannot change the predeclared source-transfer results.

## All eight GRUs learned the task

Every GRU scored **100%** on a fresh 12-step test. Accuracy over fresh 32-step sequences ranged from **98.10% to 99.55%**. The next-state task function was therefore learned for the intended horizon, with only modest degradation over the longer testing window. Exact per-seed test accuracies and model/probe SHA-256 hashes are in the [frozen numerical summary](results/comp16_summary.json).

## Held-out ROC-AUC

| Family | Eligible pairs | Divergences | Directional AUC | Rank AUC | Difference |
|---|---:|---:|---:|---:|---:|
| lag_rotate | 208 | 13 | **0.472189** | **0.494477** | **-0.022288** |
| lag_match | 163 | 15 | **0.662162** | **0.300676** | **+0.361486** |
| **Pooled** | **371** | **28** | **0.551853** | **0.429300** | **+0.122553** |

Across the four newly held-out model seeds, the learned directional-minus-rank AUC differences were +0.112655 (1103), -0.053448 (1104), +0.553125 (1203), and +0.233660 (1204).

The registered pooled AUC difference has a 1,000-resample, seed/family/depth-stratified bootstrap 95% interval **[-0.007698, +0.246804]**, which includes zero. The two-sided permutation test centered on the stratified null mean -0.028125 gave p=0.034965 with 1,000 permutations. The protocol required a strictly positive bootstrap lower bound and positive effects in both task families, as well as positive effects in at least three of four seeds. **Both the bootstrap and family requirements fail.** The preregistered cross-family prediction claim is therefore not supported.

Only 28 held-out divergences occurred. The outcome estimates are subject to considerable uncertainty across the two dynamic task families and four trained models. The permutation p-value and bootstrap interval answer different conditional questions; the decision follows the preregistered conjunction.

### Secondary scores

| Held-out score | lag_rotate | lag_match | Pooled |
|---|---:|---:|---:|
| Learned directional | 0.472189 | 0.662162 | 0.551853 |
| Learned rank | 0.494477 | 0.300676 | 0.429300 |
| **Native directional** | 0.635108 | 0.695946 | **0.667951** |
| Native rank | 0.478304 | 0.527027 | 0.500781 |
| Learned directional divided by pair distance | 0.408679 | 0.632883 | 0.484486 |
| Raw state distance | 0.507692 | 0.557658 | 0.527593 |

The native directional statistic has richer access to the model's true nonlinear answer function and remains more effective in both new families. It is an exploratory access-change control. The preregistered primary test used the learned readout and cannot be relabeled as a successful native-readout test.

## Fixed false-positive transfer

| Feature/family | False positives | Achieved FPR | True positives | Achieved TPR |
|---|---:|---:|---:|---:|
| Learned directional, lag_rotate | **101/195** | **51.79%** | **4/13** | 30.77% |
| Learned directional, lag_match | 17/148 | 11.49% | 4/15 | 26.67% |
| **Learned directional, pooled** | **118/343** | **34.40%** | **8/28** | 28.57% |
| Learned rank, lag_rotate | 3/195 | 1.54% | 0/13 | 0% |
| Learned rank, lag_match | 10/148 | 6.76% | 0/15 | 0% |

The preregistration required learned-direction FPR ≤15% in **each** new family. This fixed-source-threshold criterion clearly fails on lag_rotate.

The family-calibrated secondary operating point uses only the new family's own calibration negatives. For lag_rotate, the resulting threshold is **16.086047497375617**; its held-out FPR falls to 4/195 (2.05%) while its TPR becomes **0/13**. For lag_match, a threshold of 6.8429935369310835 yields FPR 17/148 (11.49%) and TPR 4/15 (26.67%). Family-specific calibration of the score distribution did not yield effective detection of the rotated family's observed divergences.

## Reproducibility and code

The full [family training/scoring runner](experiments/family_transfer.py) uses the existing COMP-15 observation-stack and AUC functions, with exactly the registered training seeds and tasks. Each family was trained and evaluated in a separate deterministic CPU execution; the [merger](experiments/merge_family_receipts.py) verifies the two disjoint seed sets and source thresholds, combines their complete rows and computes stratified bootstrap and permutation inference. A single invocation attempting all eight models exceeded the execution window; family-scoped runs completed successfully.

Each family execution was independently repeated byte for byte. The two merged 4,096-pair receipts were also byte-identical. PyTorch 2.10.0+cpu, NumPy 2.3.5, three CPU threads, and deterministic torch algorithms were used.

| Full local receipt | SHA-256 |
|---|---|
| lag_rotate | a9f41292ed5f9aea7fc8f1ec8c163d22456c1b8c13f23bc52522fe0feb4e8d23 |
| lag_match | 3947a2e0fd409f3684af4da8a025bead197e866f293a8c21360059dead42295c |
| merged (4,096 pairs) | **a3fa526da6c72e30f924af747b70197dd731bc6683347acd253f8290810a71a1** |

GitHub's compact receipt stores every model/probe hash, exact positive-negative Mann-Whitney strict-win and tie count, source and adapted operating thresholds, eligibility, all metrics, and the registered negative decision. [temporal_family_audit.py](temporal_family_audit.py) reconstructs the AUCs and FPR numerators from that frozen receipt using only Python's standard library. The independent finite history-dependent task reference and Lean statements compile in lightweight CI. Full GRU training is opt-in.

From the repository root:

```bash
python -m research.shaft_compactification.experiments.family_transfer \
  --family lag_rotate --per-goal 16 --training-steps 1200 --out /tmp/rotate.json
python -m research.shaft_compactification.experiments.family_transfer \
  --family lag_match --per-goal 16 --training-steps 1200 --out /tmp/match.json
python -m research.shaft_compactification.experiments.merge_family_receipts \
  --family-receipt /tmp/rotate.json --family-receipt /tmp/match.json \
  --out /tmp/shaft16_full.json
python -m research.shaft_compactification.temporal_family_reference
python -m research.shaft_compactification.temporal_family_audit
```

## Next falsifier

The key unresolved variable is **history-aware monitoring**. A fresh registered experiment can train a probe that explicitly represents previous-input state, then compare its directional observability with the current COMP-15 learned probe under new dynamic task families. Another test should include entirely different recurrent architectures and calibrated false-positive constraints.

The present negative result is preserved, including both failed preregistered decisions and the apparent secondary native-readout advantage.
