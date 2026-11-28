# SG-13: exact top-k strong branching and learned proof-obligation splits

Scope: synthetic, additive source-bound spectral routing, SG-12 integer certificates. All policies share the same best-bound live obligation, root multiplier vector, initial schedule and event accounting. Only the split component changes. The verifier and original default policy are unchanged. Frontier selection and adaptive stopping were held fixed, so this does not test learned obligation selection or learned stopping.

## Exact all-child bound evaluation

For each cached multiplier context, filter its ranked deltas to allowed components after removing required/excluded components. Let k be the residual cardinality and v the top-k sum plus base and forced contribution. If j is in the selected top k, forcing j leaves v unchanged; excluding j replaces delta_j with the (k+1)-st delta. Otherwise excluding j leaves v unchanged and forcing j replaces the k-th delta. Taking the minimum over multiplier contexts and the inherited upper cap gives the exact conditioned child bounds. The context's ranking is already cached; the all-candidate scoring work is O(Md) for M multiplier vectors and d components. These bounds guide branching only; SG-12 independently handles all actual child insertion and verification.

Policies: SG-12 analytic (first remaining component in the last multiplier vector's order); exact strong branching (maximize product of 1 plus incumbent-clipped bound reductions on each side); online pseudocost (per-component running normalized realized child reductions, prior based on single-vector margin); learned linear score and root-shuffled label control. The learned model has 12 coefficients and uses only the last vector's ranking and margins plus local features. Root-weighted ridge fits log-product of strong-branch gains on 618 obligations, 7,213 candidate labels from 24 separate training roots (23 contribute labels; one requires no branching). Training states sampled every fifth split, at most 32 per root. No model weights or native rollout, no teacher API. Fixed code and seeds before evaluation, no test-driven tuning.

## Reproduction

In an installed DataGoblin checkout with NumPy, from repo root:

```
PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient:training/rlt_gym/sg13_spectral_branching OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest training/rlt_gym/sg13_spectral_branching/test_spectral_branching.py
PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg13_spectral_branching/sg13.py
```

Two targeted tests passed: 628 exact child-bound comparisons against independent reference arithmetic, and 120 complete searches on 24 small tasks checked against the exact SG-03 oracle and independently replayed. All monotonicity checks passed. Full repository pytest/Ruff and Rust port remain outstanding. SG-12 dependency source blobs match their committed versions.

## Fresh 32-component campaign

24 independent evaluation roots, 8 per regime persistent/rotating/mixed, d=32, T=12, read budgets 7/8. Two randomized timing repetitions per root and arm. Compare 2,048-event cap and soft 15 ms process CPU deadline (same 2,048-event cap). CPU includes initialization, policy, verifier action and final receipt, excludes independent replay and offline fitting. Deadline checked between atomic events, all overshoots retained. Five arms, 480 full independent transcript replays, zero mismatches. Across policies and budgets verified intervals intersect on all 24 roots, with exact agreement on 23. The remaining root 1310001 (persistent) has interval [236,262]; all five fixed-event policies hit the cap with gap 26. All five fixed-event arms return identical objective and gap per root, though schedules and transcripts can differ.

### Event cap

| policy | mean events | mean CPU ms | mean policy CPU ms | certified roots | mean gap |
|---|---:|---:|---:|---:|---:|
| analytic |441.875|15.726730|1.187363|23/24|1.083333|
| strong |426.208333|40.634819|24.272364|23/24|1.083333|
| pseudo |424.5|27.200081|10.481967|23/24|1.083333|
| learned |448.458333|30.727684|14.019076|23/24|1.083333|
| shuffled |521.166667|34.842590|16.256515|23/24|1.083333|

### Soft 15 ms CPU deadline

| policy | mean gap | mean objective | mean events | certified runs /48 | mean CPU ms | max overshoot ms |
|---|---:|---:|---:|---:|---:|---:|
| analytic |9.854167|102.541667|342.625|29|12.586077|2.172802|
| strong |20.645833|92.458333|146|0|15.140217|0.979183|
| pseudo |19.270833|93.375|237.666667|4|15.079822|1.842216|
| learned |20.083333|92.916667|214.208333|3|15.079747|0.631651|
| shuffled |20.645833|92.916667|202.041667|3|15.153532|2.297201|

Analytic certifies 16/24 roots in at least one timed repetition and 13/24 in both. Every alternative has equal or worse mean gap on every root than analytic. The strong branching shortcut reduces theoretical work for candidate scoring but was not separately microbenchmarked versus repeated naive bound evaluation. Strong branching remains expensive compared with the original cheap rule. Pseudocost reduces mean event count 3.93% but takes 72.96% more mean CPU; learned uses 13.95% fewer events than shuffled but 1.49% more than analytic. Single-host measurements, two repeats, 24 independent test units, descriptive only. No true equal-cost claim for event cap. No hard preemptive time bound. Learning is not promoted; analytic remains default. Higher-dimensional certificates independently replayed structurally/arithmetic-wise; no exponential exhaustive oracle for d=32. No native BPD, model speedup or scientific confirmation.

## Evidence custody

Core blob 391cfe7354f5587a440c0b67268cd68f10324836; campaign e8aa867e26330123d55e34f09e823199a05c4667; tests edca7743525c258710dbcb16640e40b7f423ceaa. Raw results JSON SHA256 58cce1c0d65da1afae7432fb46db734b4dd435492ecdb5488a64075346555577. The downloadable raw archive contains the complete results, coefficients, root-level outcomes and independent replay dependencies. Generated timing runs are not committed. No GPU or paid inference. Next target: feasible-schedule recovery from relaxed component choices, with deterministic repair and local coordinate improvement; improve the incumbent rather than adding per-split policy overhead. A separate fresh-root campaign should measure both objective gain and proof gap at equal CPU budgets.

## Implementation custody

[DataGoblin campaign and tests](https://github.com/MoralityLabAI/DataGoblin/tree/68efd10f01035872cfe855418e7fae08ded9fa9a/training/rlt_gym/sg13_spectral_branching) · [Branching implementation](https://github.com/MoralityLabAI/DataGoblin/blob/68efd10f01035872cfe855418e7fae08ded9fa9a/src/datagoblin/training_gym/spectral_branching.py).

Archive `sg13_cpu_campaign.zip`, SHA256 `47594e2197b0906ca72ace9da25039bee9d475b0df99ea3ecf87620256efb382`. All four committed DataGoblin files match local hashes. Raw timing runs are supplied separately, not committed.
