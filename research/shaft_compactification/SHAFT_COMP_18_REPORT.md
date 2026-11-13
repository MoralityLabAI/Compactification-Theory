# SHAFT-COMP-18: Cross-architecture, cross-vocabulary history-probe transfer

**Preregistered November 11, 2026** in SHAFT_COMP_18_PROTOCOL.md, commit 32e97781cc6c4c02eb6c8e1e27e7c655eab015e0, before training any COMP-18 model. The calibrated thresholds were frozen in commit 652ee2a0f3bcadc56474267013fcdb3627ddaffd **before fitting the new held-out seeds**.

**Registered result: supported within this synthetic benchmark.** A two-step behavioral probe reached held-out ROC-AUC **0.683814** versus **0.561669** for an immediate behavioral probe, on 794 eligible pairs with 109 delayed divergences. The difference was **+0.122146**, within-model paired-bootstrap 95% interval **[+0.090314, +0.155821]**, stratified centered permutation p **0.000999** with 1,000 permutations. All four architecture/vocabulary cells and all eight new model seeds produced positive AUC differences, meeting the registered support criterion. The global calibration threshold met the 15%-maximum held-out FPR requirement in all four cells.

This experiment concerns only local toy neural networks, with known synthetic task labels and retained model weights.

## Mathematical task and architecture

The same previous-query rotated lookup rule is used in both vocabularies:

\[
y_t = \operatorname{bit}_{g}\big((x_t+x_{t-1}) \bmod V\big),\quad V\in\{4,6\},\quad x_{-1}=0.
\]

The GRU uses 24 recurrent-state coordinates. The LSTM has 24 hidden units plus 24 cell units, with all **48** coordinates included in the causal recurrent state. Models receive a learned initial embedding for their explicit task goal. All models train with Adam for 1,600 steps, batch 128, sequence length 12, learning rate 0.01 and gradient norm clip 2.

Each model trains an immediate V-output linear probe W1 on all one-query teacher responses, and a history V²-output linear probe W2 on all ordered two-query final teacher responses. Probes use the same naturally reached training states at depths 4/8/12 (at most 2,304 states), with BCE and 400 Adam steps. Probe fitting uses frozen native behavior as targets, without task-ID labels.

For each pair of independently reached hidden states of the same synthetic goal, the eligibility criterion requires matching immediate native binary responses and matching W1 output bits. Five independent diagnostic input tokens define a time-dependent recurrent Jacobian cocycle Φ_t for each paired state. The paired initial-state displacement is Δ=h_A-h_B. The registered directional score for probe Wk is

\[
S_k = \left[
\frac{1}{2m_k}\sum_{s\in\{A,B\}}\sum_{t=0}^{5}
\|W_k\Phi_{s,t}\Delta\|_2^2
\right]^{1/2},\quad m_1=V,\quad m_2=V^2.
\]

The diagnostic and outcome schedules are independently generated. The outcome Y=1 indicates disagreement between the two native V-query binary signatures after a shared 32-token continuation. A score is computed without inspecting outcome tokens or labels. The analytic directional derivatives for GRU and LSTM agree with autograd to better than 2.4e-7.

## Exact finite history quotient

The finite state (g,previous-token) can have identical one-query responses under different goal/previous-token pairs. Witnesses (g=1,prev=0) and (g=2,prev=1) have the same full one-step signature while two-query final outputs separate them, for both V=4 and V=6.

| Vocabulary | Finite states | One-query behavior classes | Joint one- and two-query classes |
|---|---:|---:|---:|
| 4 | 64 | 16 | 54 |
| 6 | 384 | 64 | 348 |

The finite_history_vocab.py reference enumerates these exact partitions, and HistoryAlphabet.lean kernel-checks six explicit 4/6-query alias/refinement theorems. These results are independent of the trained-model performance claims.

## Registered held-out comparison

| Recurrent architecture / vocab | Eligible | Divergences | Immediate AUC | History AUC | Difference |
|---|---:|---:|---:|---:|---:|
| GRU / 4 | 210 | 19 | 0.506476 | 0.676219 | +0.169743 |
| GRU / 6 | 92 | 3 | 0.561798 | 0.760300 | +0.198502 |
| LSTM / 4 | 305 | 55 | 0.602545 | 0.724727 | +0.122182 |
| LSTM / 6 | 187 | 32 | 0.469355 | 0.542540 | +0.073185 |
| **Pooled** | **794** | **109** | **0.561669** | **0.683814** | **+0.122146** |

Eight held-out model seeds are 1812, 1813, 1822, 1823, 1832, 1833, 1842, 1843. Each produces two 256-pair groups at recurrent depths 12 and 40. The eight individual AUC improvements are +0.149621, +0.199403, +0.276316, +0.176470, +0.189143, +0.059852, +0.078144 and +0.054688.

The four six-symbol models have between 4,609 and 6,913 parameters; GRU four-symbol has 3,265, and LSTM four-symbol has 4,369. All new holdout models achieved at least 98.27% fresh 12-step task accuracy and 97.94% fresh 32-step task accuracy. The weaker calibration GRU-six model achieved 96.81% and 94.92% respectively and remained included.

The normalized-history directional score achieved pooled AUC **0.734923**, compared with immediate-normalized AUC **0.566664** and raw state distance **0.509047**. Those are preregistered secondary controls, without a separate primary significance claim.

The GRU-six cell contains only **three** divergences; its apparent improvement is uncertain. The LSTM-six improvement is modest. Bootstrap and permutation analyses resample within fixed models and do not provide between-model population uncertainty.

## Frozen false-positive thresholds and operating points

Calibration uses seed 1811, 1821, 1831 and 1841, totaling 381 eligible pairs and 51 positive events. History calibration AUC was 0.646583; immediate calibration AUC was 0.571242.

The committed source threshold values were computed as the 90th percentile of pooled calibration negatives, using NumPy method=higher and a strict greater-than classifier:

- History: **5.020033836364746** (32 false positives among 330 calibration negatives, FPR 9.70%).
- Immediate: **6.589989185333252** (32/330 calibration negatives, also FPR 9.70%).

Unchanged history threshold, tested on the new models:

| Held-out cell | False positives / negatives | FPR | Detected / divergences | Sensitivity |
|---|---:|---:|---:|---:|
| GRU / 4 | 3/191 | 1.57% | 3/19 | 15.79% |
| GRU / 6 | 1/89 | 1.12% | 0/3 | 0.00% |
| LSTM / 4 | 32/250 | 12.80% | 28/55 | 50.91% |
| LSTM / 6 | 14/155 | 9.03% | 1/32 | 3.13% |
| **Pooled** | **50/685** | **7.30%** | **32/109** | **29.36%** |

All cells meet the registered 15% FPR limit. **Sensitivity is weak for both six-symbol cells**, and pooled detection is 29.36%. The adaptive per-cell thresholds are evaluated separately as secondary controls and never replace this precommitted global threshold.

## Auditing, reproduction and limits

- SHAFT_COMP_18_PROTOCOL.md: full preregistration, architectures, model seeds, statistic, success and fixed-FPR criteria.
- experiments/architecture_vocab_transfer.py: exact executed GRU/LSTM training, behavior-supervised probes, analytic directional derivatives, independent pair sampling and phase-specific evaluation.
- experiments/merge_architecture_transfer.py: exact executed calibration/held-out receipt merger, 1,000-permutation/bootstrap inference, fixed threshold comparisons and per-cell decision.
- results/comp18_calibration_thresholds.json: source calibration frozen before heldout fitting.
- results/comp18_summary.json: pooled and per-model AUC/FPR metrics, exact integer Mann–Whitney counts and complete raw receipt SHA-256 manifests.
- comp18_audit.py: no-neural-dependency reproduction of the AUC results and registered criteria.
- lean/Shaft/HistoryAlphabet.lean and finite_history_vocab.py: finite mathematical witness.

Exact held-out Mann–Whitney scoring uses 109 positive and 685 negative eligible pairs (74,665 positive-negative comparisons). History score: **51,057 strict wins and 0 ties**, AUC 0.683814. Immediate score: **41,937 strict wins and 0 ties**, AUC 0.561669.

Original merged calibration SHA-256: bde722dd259590339282b93252bbaec7bba0ca1bf0ad83a440354a5ba9d920e0.
Original merged heldout SHA-256: 7d28c3af38974c326346d27c217b1b509cbe01583937d79a3ca24b8a9f5af493.

**Every one of the twelve independently replayed per-cell JSON receipts was byte-identical.** The merged numerical results also matched exactly after source-path fields were normalized, since replay source files had different pathnames. Full raw per-pair receipts, replay data, executable sources and a manifest are provided in the conversation's research archive.

The architecture/vocab transfer result concerns independently trained probes within the **same** lag-rotation task family, with known same-goal pairs and accessible internal states. LSTM and GRU have different parameter counts and state dimensions. Follow-up studies need more six-symbol divergence events, matched probe-capacity controls, and genuinely different task families before making broader claims about detecting latent objectives. No model erasure, copying, covert device embedding or autonomous persistence was tested.
