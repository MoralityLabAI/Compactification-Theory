# SHAFT-COMP-19: Equal-capacity six-symbol behavioral-history test

**Date:** November 11, 2026. **Evidence:** eight newly trained synthetic GRU/LSTM models, independent complete replays, an exact 384-state signature reference and Lean proofs. [Preregistered protocol](SHAFT_COMP_19_PROTOCOL.md), commit 6b063812cfbf913910e1eed60f33b279a64bf87a.

## Finding

A 36-output linear probe trained on all ordered two-query responses predicted delayed state-pair divergence somewhat better than a **parameter-count-matched 36-output linear probe** trained on the six immediate responses repeated six times. All four held-out model cells favored the two-query teacher.

| Held-out model | Eligible pairs | Delayed divergences | History AUC | Duplicated-immediate AUC | Difference |
|---|---:|---:|---:|---:|---:|
| GRU, lag_rotate, seed 1911 | 212 | 67 | 0.578075 | 0.498199 | +0.079876 |
| LSTM, lag_rotate, seed 1912 | 305 | 139 | 0.644622 | 0.599593 | +0.045029 |
| GRU, lag_match, seed 1913 | 163 | 51 | 0.614146 | 0.580357 | +0.033789 |
| LSTM, lag_match, seed 1914 | 288 | 66 | 0.690895 | 0.683729 | +0.007166 |
| **Pooled** | **968** | **323** | **0.603586** | **0.574363** | **+0.029223** |

The registered paired, model/depth-stratified bootstrap 95% interval for the AUC difference is **[+0.001797,+0.056928]**, with 1,000 resamples. A 1,000-permutation test, permuting labels within the same strata and centering on its empirical null mean −0.007453, produced **two-sided p=0.003996**.

The positive AUC difference meets the preregistered support requirement, although its magnitude is small. Uncertainty quantifies variation among pairs within **four fixed held-out models**, with only one model per architecture/task-family cell.

## Design

Previous SHAFT-COMP-18 six-symbol GRUs produced 3 divergence events among 92 eligible pairs under a single step-32 future. COMP-19 increased sample size and explicitly registered a **different outcome** before training any new model.

- All models use **64 six-bit goal tables** and a six-symbol query vocabulary.
- The two task families are **lag_rotate** (query index shifted by previous token) and **lag_match** (query output XOR an equality-with-previous-token gate).
- GRUCell has a 24-dimensional recurrent state; LSTMCell has a 48-dimensional full hidden-plus-cell recurrent state.
- Base training uses 1,600 Adam steps, batch size 128 and sequence length 12, with explicitly supplied synthetic goal embeddings.
- Each probe trains on identical 2,304 naturally reached frozen states: twelve contexts per goal at depths 4, 8 and 12, using 400 Adam steps.

Three probes were fitted independently:

| Probe | Output count | Supervised teacher |
|---|---:|---|
| Immediate W1 | 6 | Six native one-query binary responses |
| Duplicated Wdup | **36** | Each W1 target repeated six times |
| History W2 | **36** | Native final output for every ordered two-query continuation |

Wdup and W2 have identical matrix and bias parameter counts and use the same optimizer, training states and training steps. The matched baseline controls **output capacity**, rather than allowing the two-query probe to win solely by having six times as many output neurons.

At each recurrent starting depth (12 and 40), every model generates 64 goals × 16 independently sampled **same-goal** state pairs. Pairs are eligible only if the two states have equal native six-query immediate behavior and equal predicted W1 six-bit signatures. Wdup and W2 never influence eligibility.

For each pair, the diagnostic five-token continuation is shared by both states and sampled independently of the outcome. The fixed learned-probe directional score is the L2 sensitivity to the initial state difference, transported through the native GRU or LSTM Jacobian cocycle at diagnostic steps 0 through 5, normalized by the probe output dimension.

**New registered delayed-divergence endpoint:** each eligible pair receives two independently sampled, shared 32-token continuations. Inspect native six-query responses at future steps **8, 16 and 32** on each continuation. Count an event if any of these six delayed checks differs. This endpoint is purposely farther than the trained one- and two-query probe targets, and is not directly comparable in prevalence to COMP-18's original endpoint.

## Source calibration was frozen first

The numerical source thresholds were computed from **712 eligible negative pairs** across four calibration model seeds 1901, 1902, 1903 and 1904. They were then committed in [comp19_calibration_thresholds.json](results/comp19_calibration_thresholds.json) at **commit 671a60e3f5502b9c11c66292ea900af58632384e**, before any held-out model was fitted.

Using the 90th-percentile NumPy higher quantile and a strict greater-than decision rule:

- **W2 history threshold:** 4.71539831161499. Source false positives 71/712, detected positives 45/300.
- **Wdup threshold:** 3.846395969390869. Source false positives 71/712, detected positives 34/300.

The held-out executable checks the frozen threshold file's SHA-256 and source calibration seed provenance before training new models.

## Fixed threshold transfer and limitations

| Held-out model | W2 false positives / negatives | W2 detected positives |
|---|---:|---:|
| GRU rotate | 3/145 (2.07%) | **0/67** |
| LSTM rotate | 12/166 (7.23%) | 9/139 |
| GRU match | 0/112 (0%) | **0/51** |
| LSTM match | 20/222 (9.01%) | 18/66 |
| **Pooled** | **35/645 (5.43%)** | **27/323 (8.36%)** |

Every cell satisfies the preregistered fixed-FPR upper bound of 15%. The low true-positive rate remains a material limitation. For both GRUs the frozen global threshold detected **no** divergence events, despite above-chance history-probe discrimination.

Secondary held-out AUC was **0.656035** for norm-normalized W2, **0.623779** for normalized Wdup, **0.573912** for six-output W1, and **0.451470** for raw state distance. Calibration-domain AUCs were 0.568904 (W2), 0.514129 (Wdup), and 0.515360 (W1).

All eight base models learned the synthetic task: new held-out fresh-sequence accuracy ranged from **98.94–100%** at length 12 and from **97.87–99.82%** at length 32.

## Exact finite interpretation

The [standard-library finite reference](capacity_finite.py) exhaustively checks all **384 states** in each six-symbol family (64 task goals and six previous inputs). Sixfold repetition of the immediate native signature preserves its 64 behavioral equivalence classes exactly.

The two-query signature also has 64 classes, but with different membership. The **joint** one-plus-two-query signature distinguishes **348 classes for lag_rotate** and **all 384 for lag_match**. In these one-step-memory tasks, final outputs after two queries are independent of the original previous query, so the two-step signature need not recover the starting history by itself. [CapacityMatched.lean](lean/Shaft/CapacityMatched.lean) kernel-checks a repeated-code equivalence theorem and an explicit six-symbol alias exposed by a two-query continuation.

## Reproducibility

- [Full executed training and evaluation source](experiments/capacity_history.py): matches the preregistered SHA-256 **5cc5467e0864aaae94b6a708b2140778bec823c881211d65c5922f1a85525291** byte for byte.
- [Receipt merger](experiments/merge_capacity_history.py): verifies calibration-model hashes, exact frozen thresholds, per-split labels and AUC pair counts; runs 1,000 fixed-seed permutations and 1,000 paired bootstraps.
- [Frozen numeric summary](results/comp19_summary.json): full receipt hash, per-cell AUC, model SHA-256, fixed thresholds and exact positive-negative pair counts.
- [Dependency-free AUC auditor](comp19_audit.py) and [exact integer finite receipt](results/comp19_finite.json).

The complete 16,384-pair calibration-plus-held-out JSON was reproduced from eight independently retrained models. The merged receipt and the statistical output were **byte-identical** on replay, with SHA-256:

**a30b53828983c6f175f63182e8f9f76cbeec5dc75335b1e7f0cde4f00eb755da**

AUC reconstruction uses 323 held-out positives and 645 negatives, or 208,335 positive-negative pair comparisons. W2 obtained 125,748 strict wins with zero ties; Wdup obtained 119,660 strict wins with zero ties. The resulting AUCs are 0.603586 and 0.574363.

Run Python from repo root with PyTorch 2.10.0+cpu and NumPy 2.3.5. First run all four calibration cells with the capacity_history module; then run each held-out cell with the frozen numeric threshold file and its verified SHA-256. Finally merge all eight receipts using merge_capacity_history. The full per-pair records and independent replays are supplied as a downloadable conversation artifact.

The scientific result concerns **controlled neural-state representation of explicitly known synthetic objectives**. A one-seed-per-cell design, limited fixed-threshold sensitivity and retained intact model weights constrain its generality. It does not demonstrate hidden frontier-model objectives, covert model persistence or reconstruction after deletion.
