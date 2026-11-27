# SG-14: primal recovery from dual certificates

Optional feasible initialization before unchanged SG-12 block search. All schedules are independently scored for legal row budgets. The existing initial schedule is retained unless the recovered schedule improves its exact objective.

## Mathematics

For a fixed row t, let L_j be the largest coordinate of another consumer of j, including the producer as fallback. The marginal cost of adding j in row t is lambda*b_j*max(0,x_t-L_j). Select the top k_t adjusted utilities; retain existing selected components on ties. Two Gauss-Seidel sweeps (reverse and forward) provide monotone row repair. Per-label visit masks allow O(d) marginal score computation per row plus sorting.

Dual rounding uses the exact SG-12 conditional relaxation for a single root multiplier vector. Traverse backward; each row chooses its top k adjusted components given the already feasible suffix. Recompute only one vector's prefix tables and rankings, O(Td) arithmetic plus row sorts. No intermediate schedule violates row budgets. This is a feasible heuristic, not a global exact primal solver.

Two-label repair fixes all other components and each row's combined occupancy of the pair. Four suffix states permit O(T) dynamic-program transitions by charging each label's price only at its last use. The implementation stores and copies traceback tuples, so O(T) transition arithmetic is not an O(T) total runtime or memory-traffic guarantee. Pair candidates are ranked by the value of swapping their complete histories; test at most twelve pairs, then perform a reverse row sweep. Exact within each two-label neighborhood, but no global optimality claim. The helper is intended for trusted callers supplying a valid integer root multiplier vector; the proof checker validates only the resulting legal incumbent, independently of the repair algorithm.

## Reproduction

From installed DataGoblin checkout, add SG-07 and SG-11 gym directories to PYTHONPATH. The source requires NumPy for fitting. The exposed fixture can be regenerated from the prior SG-13 task generator with seed 1310001, persistent regime, d=32, T=12, and k[i]=7+(1310001+i)%2. Its canonical json.dumps(q,indent=2) plus newline has Git blob d2dc6af18308c146a1bdff1d144a1fda171f74ad. Save as sg13_exposed.json in the working directory; it is also in the supplied archive.

    PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient python -m pytest training/rlt_gym/sg14_spectral_primal/test_spectral_primal.py
    PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg14_spectral_primal/sg14.py

Four targeted tests passed: 192 exhaustive row optima, 400 exhaustive pair optima, 100 full exact-oracle/replay checks, 64 single-vector rounding parity fixtures. Full repository tests and Ruff were unavailable. No Rust compilation. First full timing run hit the tool deadline after 21 of 24 evaluation roots; it did not persist complete receipts. Restarted from scratch using per-root progress serialization. Same seeds/methods; final results only, interrupted log retained. This is exposed synthetic development, not independent confirmation.

## Campaign

24 separate training roots (8 per regime), 24 evaluation roots (8 per regime), d=32, T=12, k=7/8. Fit five ridge predictors with 14 input features each to negative certified gap at a soft 15 ms process CPU deadline. Root-shuffled training target vectors preserve each method's unconditional outcomes. Select a single method before repair. The best training fixed policy is dual rounding (mean gap 2.542; dual+row ties). Seven arms, two randomized timing repetitions per budget and root, 672 evaluation replays, 120 training replays and five exposed-fixture replays. All certificates independently checked using the original SG-12 verifier. Cross-policy intervals intersect on all roots, exact on 22. Unresolved roots 1451005 [-35,-29] and 1451007 [-89,-87]. Deadline checked between atomic verifier events; setup, feature computation, repair, search and receipt included, replay and offline fitting excluded. No hard deadline; maximum overshoot 0.950 ms. No independent exhaustive oracle at d=32. All 24 roots were exposed in the interrupted attempt as far as it progressed (21 completed); final restart is a fresh run of the same planned cohort, not a new independent confirmation cohort.

### Soft 15 ms deadline

| method | initial gain | gap | mean CPU ms | certified runs /48 |
|---|---:|---:|---:|---:|
| base |0|4.729167|12.0359|31|
| row |5.833333|2.3125|12.1697|30|
| dual |11|1.3125|10.4817|32|
| dual_row |11.041667|1.291667|10.6540|32|
| pair |11.041667|1.354167|11.2058|32|
| learned |11|1.291667|10.6371|32|
| shuffled |11|1.333333|10.8011|32|

Dual improves initial objective in 19/24 roots; gap wins/ties/losses vs base 9/15/0, by mean over the two timing repetitions. Learned picks dual on 23 roots, row on one. Shuffled picks dual on 18, pair on six. No evidence of a useful learned-policy advantage over the cheap fixed dual arm. Pair contributes no extra initial objective gain after dual+row on these roots.

### 2048-event cap

| method | gap | mean events | mean CPU ms | certified roots /24 |
|---|---:|---:|---:|---:|
| base |1.125|672.875|19.7568|22|
| row |0.333333|672.833333|19.9116|22|
| dual |0.333333|595.333333|17.8713|22|
| dual_row |0.333333|590.5|18.1835|22|
| pair |0.333333|590.5|18.9414|22|
| learned |0.333333|595.333333|18.1623|22|
| shuffled |0.333333|595.333333|19.4321|22|

Exposed SG-13 root: base [236,262], row [249,262], dual/dual_row/pair [258,262] after 2048 events. Initial objectives equal the final lower endpoints. Two residual intervals in fresh cohort remain nonzero. Recommend fixed dual rounding as next deterministic baseline, without changing existing default. Single host, two timing repetitions; no hardware/model performance claim.

Source blobs primal aa320ea518d5b7daee801b2c0f5503e6523974d0; campaign 89c60742c55bdc2bfaa8190ac542010f1d34b423; tests d8359b24a91ec80302628bdfe73f62a2b5a12776. Full raw results SHA256 34922dffef7f7a750c9388eeb5c9fa01f5f4fa07f4a0d148c387c4ea83f00280. Raw generated timing runs supplied separately, not committed. Next: investigate exact dual-face recovery or coordinated multi-label repairs, then run a sealed new cohort.

## Implementation custody

[DataGoblin campaign and tests](https://github.com/MoralityLabAI/DataGoblin/tree/d7fe6f321ca994375e0fd9b0901955600db0a61a/training/rlt_gym/sg14_spectral_primal) · [Primal repair](https://github.com/MoralityLabAI/DataGoblin/blob/d7fe6f321ca994375e0fd9b0901955600db0a61a/src/datagoblin/training_gym/spectral_primal.py).

Complete archive `sg14_cpu_campaign.zip`, SHA256 `d2a183617154a45eac66de659900d83e7be99f53beac1dc32066c4cb8a40707e`. The archive includes executed sources, dependency sources, full receipts, exposed input, the interrupted log and SHA256 manifest. The three committed Python files match their local Git blob hashes. Archive integrity check passed. Generated timing runs are supplied separately.
