# SG-11: direct quotient generation and sparse certified routing

[DataGoblin source, benchmark and tests](https://github.com/MoralityLabAI/DataGoblin/tree/8db2c294a4e389d777fd3437554062249626a10a/training/rlt_gym/sg11_spectral_quotient) · [Sparse verifier](https://github.com/MoralityLabAI/DataGoblin/blob/8db2c294a4e389d777fd3437554062249626a10a/src/datagoblin/training_gym/spectral_quotient.py).

## Exact reduction

At state S, write R=A minus S. All actions in this class incur the same last-use price and have the same suffix union S union R. Maximizing utility among actions with a given R is equivalent to taking the best k-|R| members of S, with earliest original index ties. Enumerating only R costs Q(d,k,s)=sum_{r=max(0,k-s)}^{min(k,d-s)} binom(d-s,r) representatives. Terminal action is direct top-k adjusted utility. The original raw transition charge is retained alongside a separate quotient work counter (one unit for a terminal solve). Complete expansions are rejected before mutation if either cap would be exceeded. The sparse verifier supports d<=32,T<=16; no dense mask table or action list. Fixed bound table uses exact prefix contribution arithmetic, with separate accepted proposal overlay. Dynamic frontier and dominance state still grow with the work cap; no global RSS cap or frontier eviction.

## Verification and paired timing

Four targeted pytest tests passed; 87,480 quotient class comparisons, 5,120 state comparisons each for fixed bound, arbitrary dual bound and coordinate optimization, 40 interleaved original-verifier fixtures, high-d terminal/cap and malformed input checks. Initial pytest failure was an incorrect expected comparison-count threshold; fixed to the exact count, rerun passed. No full repository tests or Ruff run.

36 fresh development roots: 18 d8/T6, 9 d9/T7, 9 d10/T8. Equal three planted regimes. Best-bound priority, one zero-start reverse coordinate sweep per state, 24 proposals, 80 decisions, 2048 raw cap (sparse also 16384 class cap). Different allocation policy from prior SG-10 campaign, so only paired timings here are comparable. Three randomized repetitions, process CPU mean of root medians, includes init/receipt and excludes external oracle/replay. All 216 timed receipts independently checked by original SG-05 and compared against exact oracle. Both arms 35/36 certified, mean gap 1/3, identical receipts ignoring class counter, 14,054 raw actions. SG-10 batch 2.299314361 ms; SG-11 sparse 1.718166389 ms, 6,047 work units, 25.27% lower CPU.

## Larger tasks

24 separate roots with k=2 or 3, one run per arm. Raw cap 8192 plus class cap 8192 versus class cap 8192 alone, same proposal and decision caps. Different effective work budgets; not matched CPU. The 48 large receipts were replayed with the new implementation only. No independent high-d oracle or old-verifier replay. Zero gaps are certificates under the sparse verifier, with the mathematical argument and small-domain cross-check above.

| Dimension/visits | Roots | Raw cap certified | Class cap certified | Raw cap mean gap | Class cap mean gap | Raw cap mean CPU ms | Class cap mean CPU ms |
|---|---:|---:|---:|---:|---:|---:|---:|
|16/8|9|7|7|1.7778|1.8889|13.0459|15.2477|
|24/10|9|3|6|9.2222|5.6667|32.0396|35.5212|
|32/12|6|1|1|19.6667|19.5|42.2099|42.0638|

Peak live states 2107. More work can expose looser child bounds than the tightened parent; one d16 root has higher final gap. Inheriting a parent's admissible upper bound is a future improvement. Initial empty suffix still has the full combinatorial action space.

## Action generation microbenchmark

Five repetitions, median process CPU; compare exact dictionaries of raw and direct representatives. d20,k5 has 15,504 raw actions: s=0 direct 13.597811 vs raw 10.848722 ms; s=10 direct .412236 vs raw 9.685919; s=16 direct .014900 vs raw 9.323984; s=19 direct .006698 vs raw 8.971745. At d32,k8,s28 direct generated 16 representatives in .019138 ms; raw count 10,518,300 is theoretical and was not timed. Single host. No native model, GPU, or paid inference. Rust port pending. No memory timing or RSS measurement.

Evidence custody: executed blobs in DataGoblin commit 8db2c294a4e389d777fd3437554062249626a10a verified byte-identical (core cc52b97d2f532a1033380aefc5b7acdf810c87f0, campaign 0f7046e531abe666a79a8f9cafd2a7a6779fe586, tests a05e02a5cdf707a9ecda2b4604ce209cd938eb1f). Prior local dependency hashes match pinned source. Raw results SHA256 f5fc1beaa83811b194f6c5e4eb190ca9749debda03ac0bd23af487186b2099aa. Full archive sg11_cpu_campaign.zip SHA256 08e8e034b7463b8ee08308634d84d59ed6370493f15cc62e136afb4f49d093b6 supplied separately; generated raw runs not committed.
