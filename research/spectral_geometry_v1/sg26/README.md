# SG-26: coverage-aware certificate retention under a serialized byte budget

## Scope and policy

SG-25's convex certificate applies over a declared two-dimensional affine utility/price plane with fixed budgets, coordinates and bit prices. Every fresh vertex is independently verified using exact SG-15 integer arithmetic before insertion. A convex combination of vertices sharing a feasible schedule preserves a closed integer gap. The compact checker validates resident IDs, nonnegative integer weights summing to their denominator, exact point and rational multiplier interpolation, identical schedule and closed chord. Derived hits are not inserted. Immutable vertex records are scoped to one validated plane in trusted in-process Python code; this is not an untrusted execution/deserialization boundary.

Group vertices by their schedule. Eviction scores the loss in twice the hull area after removal of the candidate, then positive loss of squared segment length, then disappearance of the group, lexicographically. Ties use least recent accepted reference time then ID. Weighted multiplies each loss by the fixed acquisition-work proxy 1+4*LP calls. This is a heuristic for coverage and cost, not a learned model or guarantee of future query coverage. A geometrically redundant vertex can still affect rational representability; invalid interpolation falls back. LRU touches every referenced vertex on accepted hits. Newly inserted records may themselves be evicted. The 900-byte retained serialized budget counts the actual packed vertex payload, including ID, point, denominator, bound, objective, cost proxy, T schedule masks and T signed multiplier numerators: 36+12*T=180 bytes at T=12, five records maximum. The cap holds after each operation; transient insertion can reach six records. Python objects, duplicate packed storage, hull indexes, clocks, prediction history and audit logs are outside this limit. Total RSS not measured.

## Protocol

24 independent exposed synthetic trajectories, eight per persistent/rotating/mixed regime, d=32,T=12,k=7/8. 32 distinct integer queries per trajectory in [0,8]^2, half sparse utility only and half also affine global lambda. Within each group, half shuffled queries, half three corners followed by 29 shuffled interior points. Two randomized whole-trajectory timing repetitions per arm. Five policies share the stepwise fallback: recent multiplier and schedule, affine proposal from up to three prior endpoints, one reverse coordinate correction, SG-20 repair, chain/direct LP and capped 2048-event block search. No future query-order oracle. Initial common anchor construction excluded. Process CPU includes lookup, hull maintenance, independent final verification and block replay or compact hit acceptance, recovery, all solver calls, history and receipt assembly. Post-timing full diagnostic checks and separate replay excluded. Area_full uses identical area policy plus a redundant full task reconstruction and exact arithmetic check on hits. Independent units are trajectories, not queries or repetitions. One host; no hard deadline, sealed confirmation or native model claims.

Initial campaign stopped after 17 completed roots when the next anchor (2612001) had valid interval [39,43] instead of a closed certificate. Amended source starts this root with an empty verified cache, retaining its verified feasible schedule and valid bound for proposals. Earlier completed roots are retained (behavior unchanged for closed anchors); first revised attempt hit a 20-second execution timeout, and the same checkpoint was resumed with unchanged caps. No root or nonclosed result was excluded. All 62 block fallbacks and nonzero final gaps occur on this difficult root. Per repetition, step/area/weighted/area_full leave six unresolved queries (one gap 2, five gap 1); LRU leaves seven (one gap 2, six gap 1). Initial source and failed/resumed logs are archived.

| Arm | CPU ms/query | LP calls | Hits | Strict interior | Closed /1536 | Mean gap | Evictions | Wins vs step /24 | Wins vs LRU /24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step |5.855419|396|0|0|1524|0.009115|0|0|8|
| LRU |6.431226|386|430|334|1522|0.010417|906|16|0|
| area |5.424078|336|662|508|1524|0.009115|676|19|17|
| weighted |5.373223|336|662|508|1524|0.009115|676|20|18|
| area_full |5.581672|336|662|508|1524|0.009115|676|14|11|

Weighted vs LRU: 53.95% more hits, 12.95% fewer LP calls, 16.45% lower mean CPU. Weighted vs step: 15.15% fewer LP calls, 8.23% lower CPU. Paired trajectory mean saving vs LRU 1.058003 ms/query, median 0.151961, descriptive 10000-resample percentile bootstrap [0.265194,2.140170] (seed 26); vs step mean 0.482196, median 0.311529, interval [0.060630,0.878840]. Area vs step mean 0.431341, interval [-0.069131,0.857035]. The hard root dominates the mean. Weighted and area tie in total hits and LP calls, with differences per root; no robust evidence of superiority of weighting over area.

| Direction/order | Step ms | LRU ms | Area ms | Weighted ms | Weighted hits /384 |
|---|---:|---:|---:|---:|---:|
| utility, shuffled |0.829|0.764|0.654|0.680|166|
| utility, corners-first |0.801|0.390|0.390|0.399|306|
| mixed, shuffled |6.925|7.295|6.342|6.277|70|
| mixed, corners-first |14.866|17.275|14.310|14.137|120|

## Timed verification

All 1536 paired area/area_full non-timing receipts match. Across their same 662 hits, mean process CPU in the acceptance call is 24.815 us compact vs 511.961 us compact plus redundant full recheck (including task reconstruction), 20.63x ratio. Overall CPU saving 2.82%, paired mean 0.157594 ms, median 0.219683, bootstrap [-0.159728,0.443938]. The kernel benefit does not establish a robust end-to-end benefit of removing the recheck. All four memory arms use mean 836.016 retained payload bytes and peak 900. Keep geometry-aware policy optional; production default unchanged.

## Verification and reproduction

Four targeted tests passed: exact fractional negative certificate; area/segment coverage and payload cap; recency, tampering and atomic rejection; weighted loss; 300 small exact-oracle policy runs. Separate replay reconstructs 768 distinct tasks and verifies 24 anchors, 7680 final valid intervals (7618 closed), 12318 acquisition stages, 5844 candidate certificates, 2416 region hits, 62 block transcripts, 2934 evictions and payload accounting, 6144 paired interval-intersection checks and 1536 exact non-timing comparisons. Zero mismatches. No exhaustive oracle at d=32. Full repository pytest, Ruff, Lean and Rust not run (isolated source checkout). Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, CPU only.

Install DataGoblin and optional NumPy/SciPy/pytest. From repository root add the SG-07, SG-11, SG-15, SG-17, SG-20, SG-22, SG-25 and SG-26 gym directories to PYTHONPATH. From the extracted reproduction archive:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q test_sg26.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg26.py --limit 8
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg26.py --resume --limit 16
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg26.py --resume --limit 24
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_sg26.py

Do not mix checkpoints. Reexecution changes timings. Source Git blobs: memory 4a8f3d75b9e0c2b51aec5faf6606b65e66baf6c2; initial campaign 1ab789400ca05f31bb6be3bbd525a75b70db91dd (archive only); final campaign 17230d160ceb60183b6193472d5b42736ad7ad0f; tests b8c807f27442425b14b72fd1aba63b9c0cba5382; audit bd6e15a5ae6ee1ef2808acb56c859f23a94692ec. Raw results SHA256 1f62ef41d4a2bbe235a11d2388fa9ca1dc3292cbfaf0e6f9e1329a219e0bed43; audit SHA256 78201f57e450135f593ed9ea902f850c10067d38b3672dc7a9d95b21f3fdcb92. Complete archive sg26_cpu_campaign.zip has 41 files, SHA256 b640c3fa75c7599ad0110e00d81a98e14a32238c37ae317dbc8bcc8366c034b1. Archive integrity and manifest checked. Generated receipts supplied separately, not committed.

Next: evaluate query-density-weighted coverage instead of continuous hull area on fresh irregular-arrival trajectories, with bounded online statistics and matched serialized budgets. Separately investigate the exposed unresolved root through integer cuts or conditioned search. Keep the exposed status and independent confirmation separate.

## Implementation custody

[DataGoblin implementation, campaign, tests and independent replay](https://github.com/MoralityLabAI/DataGoblin/tree/5c1019b2cd10c7e21a16b8e4b31590dfa895042c/training/rlt_gym/sg26_spectral_retention). All four committed Python source hashes match executed files. The 41-file archive `sg26_cpu_campaign.zip`, SHA256 `b640c3fa75c7599ad0110e00d81a98e14a32238c37ae317dbc8bcc8366c034b1`, includes both campaign source versions, raw receipts, dependencies, failed/resumed logs, tests and audit. Archive integrity and manifest hashes checked. Generated timing runs supplied separately.
