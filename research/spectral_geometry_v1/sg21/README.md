# SG-21 / SG-21B: exact parametric certificate regions

Scope: finite additive source-bound routing, existing SG-20 task domain. For a ray q(s), s=0..H, keep dimensions and read budgets fixed and require integer affine utility and effective last-use price trajectories. The implementation permits at most one of bits, coordinates and lambda to vary simultaneously, along with utility changes. This sufficient restriction ensures prices are affine; endpoint validation and integral steps ensure all intermediate tasks meet the registered bounds. H<=10000. The whole direction and horizon are supplied in advance. No future-query oracle is claimed for arbitrary online edits.

For fixed feasible schedule A and fixed rational multiplier e/D, write scaled upper B(s)=e*k+sum_j max_m f_j(m,s). Each f_j is affine in s; B is convex and D*F(s,A) is affine. Hence B(s)-D*F(s,A) is convex and nonnegative. The set with gap numerator <D is an interval. If both endpoints a,b meet this criterion, the chord bound at each integer s in [a,b] is less than F(s,A)+1. Flooring proves the exact integer optimum, including negative objectives and fractional dual bounds. Two verified endpoint certificates suffice; no assertion that the fixed schedule or multiplier ceases to be optimal outside its certified interval. Starting from a certified anchor, evaluate the horizon and binary search the first failed step when necessary. Maximality is with respect to this fixed schedule, multiplier, and finite ray only. The bound evaluator updates only changed component terms; unchanged components retain their exact cached values. Chord interpolation needs O(1) scalar arithmetic, excluding receipt data, verification and ray preparation. Frozen dataclasses are used by trusted Python code, not a security boundary. Reconstruct the ray from validated endpoints and verify its endpoint proof before accepting any interpolated receipt.

SG-21 compares ordinary SG-20 differential acquisition with unrestricted region acquisition. SG-21B adds a gate fixed after SG-21 and before a new cohort: only plan after an ordinary update retains its certificate, when <=4 component terms change and at least two future steps remain. It consumes no separate first-step probe. Both arms lazily retain the anchor State during covered steps, then apply absolute replacements directly to the next uncovered step. All ordinary repairs and LP fallbacks are unchanged. No production default changed.

## Campaigns

Each cohort has 24 fresh exposed synthetic trajectories (eight per persistent/rotating/mixed regime), 32 components, 12 visits, alternating budgets 7/8 and 24 distinct integer steps per trajectory. Four directions, six roots each: one utility entry +/-1/step, four utility entries in distinct columns +/-1/step, one bit price +1/step, global lambda +1/step. Directions and horizon fixed before acquisition; two randomized whole-trajectory arm orders per root. Both arms receive the same prevalidated ray and initial certificate. Timing includes all interval planning, differential updates, recovery, solver calls and receipt construction; excludes common task/ray/anchor construction and independent post-timing replay. The first campaign showed a slowdown; the gated policy was then tested on distinct roots. No sealed confirmation cohort, no matched hard deadline, single-host process CPU measurements. Independent units are trajectory roots, not steps or repetitions.

| Cohort/arm | CPU ms/query | Certified /1152 | LP calls | Reused queries | Region probes | Regions | Nontrivial regions | Wins vs delta /24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SG-21 delta |2.568551|1152|362|0|0|0|0|0|
| SG-21 ray |2.711908|1152|362|764|1578|420|64|14|
| SG-21B delta |2.412177|1152|342|0|0|0|0|0|
| SG-21B ray |2.470292|1152|342|790|1464|394|64|19|
| SG-21B gated |2.383469|1152|342|726|198|64|60|19|

Primary unrestricted radius is slower by 5.58%, despite 764/1152 interval-covered runs. Follow-up gate removes 86.48% of region probes vs unrestricted, saves 1.19% CPU vs differential and 3.51% vs unrestricted. The three sparse modes (18 roots) have mean CPU ms 0.558837 delta, 0.514771 ray, 0.522304 gated; gated saves 6.54% vs delta, wins 16/18 roots. Global price mode needs an LP on every step and yields no interval reuse in either cohort. No decrease in LP calls vs differential. The gate's overall timing gain is small; preserve SG-20 default. Region evidence is useful for compacting batch certificates even if CPU cost does not improve.

| Edit mode | SG-21 delta/ray ms | SG-21B delta/ray/gated ms |
|---|---|---|
| one utility |0.900495 / 0.905062|0.443138 / 0.414398 / 0.404264|
| four utilities |1.053608 / 1.253484|0.944621 / 0.891212 / 0.923449|
| one bit price |0.342238 / 0.293570|0.288752 / 0.238701 / 0.239200|
| global price |7.977863 / 8.395515|7.972196 / 8.336854 / 7.966965|

Four targeted tests passed, covering 540 small ray positions (all 60 anchor tasks had closed certificates; exhaustive integer optima checked at covered positions), exact maximal prefixes, reanchoring, fractional negative floor, invalid nonaffine inputs and a gated translation fixture. Separate post-run audit reconstructs tasks independently by integer interpolation; all 1152 query signatures globally distinct across both cohorts. It verifies 48 anchors, 5760 final receipts, 8710 acquisition stages, 878 regions (1756 endpoint checks), 782 next-step maximality checks and 3456 paired output comparisons. All final gaps zero. All paired outputs agree on schedule, multiplier, denominator, objective, upper and gap. The 2280 reused query receipts independently pass full SG-15 bound checks. No exhaustive oracle at d=32. Full repository tests/Ruff, Lean and Rust not run. CPU only, no paid inference.

## Reproduction

Install DataGoblin and optional NumPy/SciPy/pytest. From repo root:

    export PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient:training/rlt_gym/sg15_spectral_face:training/rlt_gym/sg17_spectral_chain:training/rlt_gym/sg20_spectral_delta:training/rlt_gym/sg21_spectral_regions
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q training/rlt_gym/sg21_spectral_regions/test_sg21.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg21_spectral_regions/sg21.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg21_spectral_regions/sg21_gate.py
    python training/rlt_gym/sg21_spectral_regions/audit_sg21.py

Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0. Source Git blobs: ray 37a9d77f55197c07fd404a621183d05949e414ee; campaign 5596b7870502bc84e0420400658e6eb1bc0a4a7f; gated 613244fe8afeadb942826f4124b6c500db9bec5a; tests 0f2f7e68d8bce82a77463b937bad13df5b9b0729; audit b93fff8ee29f4ef8ea011f49ae6b895522cabb3b. Results SHA256: SG-21 1730cdfe528e5a152ebcf6740e7f23d3bf782734da4c3163f7c4e95e403c17b2; SG-21B 456d1c386c9a296ac89757b8383b9a51f8804657d280d780f351632937a2662c. Raw outputs supplied separately, not committed.

Next: allow multipliers to move affinely with the task. The same convexity argument applies to an affine multiplier path with fixed budgets, because the constant and each adjusted component objective are affine. Propose a cheap multiplier slope from prior certified endpoints, then independently verify a new endpoint and the fixed feasible schedule. Evaluate on fresh global-price trajectories, comparing against SG-20 and fixed-multiplier interval search. A pair of distinct endpoint schedules does not establish an interior schedule certificate.

## Implementation custody

[DataGoblin campaign, region implementation, tests and replay](https://github.com/MoralityLabAI/DataGoblin/tree/271faabc4b63334d09e91577d03cbb5c023d3eda/training/rlt_gym/sg21_spectral_regions). Complete archive `sg21_cpu_campaign.zip`, SHA256 `06402e8b904bdc8edc88284d9a79365842bc3c2cb0ad22124229679585788242`. Contains 34 files including executed sources and dependency copies, both raw result sets, test and audit logs, and SHA256 manifest. Archive integrity and every contained file hash checked. All five committed Python source hashes match the local executed files. Generated raw receipts and timings supplied separately.
