# SG-27: query-density-aware verified vertex retention

## Scope

Use the SG-26 independent acceptance path on a declared two-dimensional affine task plane, with fixed budgets, coordinates and bit prices. Each resident vertex is independently verified; hits require same-schedule convex interpolation with nonnegative rational weights and resident references. No change to the proof arithmetic or existing production default. The density policy records the current and last seven observed query positions. A five-point stencil assigns weight two to the center and one to each in-domain axial neighbor, accumulating weights across the bounded history. For each prospective eviction, measure lost weighted lattice coverage across the union of all schedule-specific convex hulls. The primary key is total lost sample weight, then the SG-26 unweighted area/segment/group loss, recency, ID. Overlapping schedule hulls count once. This is a causal heuristic; it uses neither future query order nor true future costs. Geometric coverage need not imply representable rational interpolation under the existing denominator cap. Every hit is independently accepted; failures/misses use existing warm acquisition. No learned model.

All cache arms have a common retained serialized budget of 964 bytes. The density arm reserves 64 bytes for eight signed int32 point pairs, leaving 900 bytes for vertices. All arms can retain five 180-byte vertices at T=12, since six exceed the budget. This is a logical packed-payload accounting limit, not a process RSS or Python object limit; deque, hulls, clocks, temporary stencil maps, temporary candidate insertion, prediction history and audit receipts are excluded. The density subclass temporarily admits one extra record using the unchanged SG-26 independent admission, then applies its own eviction. The retained budget holds after the operation.

## Pilot protocol

12 independent exposed synthetic trajectory roots (four per persistent/rotating/mixed regime), d=32,T=12,k=7/8, 32 distinct integer queries per root on [0,8]^2. Half sparse utilities only, half also affine global lambda. Within each regime, two stationary-cluster and two shifting-cluster roots, with one mixed-price and one utility-only root in each. Draw distinct points without replacement with weights (1+Manhattan distance to current center)^(-3); the shifting center moves halfway to the reflected location. Two randomized whole-trajectory repetitions per arm. Same stepwise warm fallback, prior-three-point multiplier proposals, one reverse coordinate sweep, repair, chain/direct LP and capped block search. Shared anchor/task creation outside timing. Timed process CPU includes retrieval, independent fresh or chord acceptance, eviction, recovery, solver and history/receipt assembly; excludes post-timing diagnostic rechecks and separate replay. Density history update is included on each lookup. No future query order given to policies.

The initial 2048-event attempt timed out before the first checkpoint; a second attempt was explicitly stopped before saving any complete roots. A diagnostic on the first root and five queries found gap one and repeated block searches. The measured pilot was then restarted from scratch with a uniform 256-event block cap, including shared anchors and all five policies, using the original 12 seeds and query orders. No completed root was dropped and no initial timing enters the results. This is an exposed pilot after an execution-budget change, not an independent confirmation. All 64 unresolved runs per arm occur on root 2710000, every gap one. No other root has a nonzero gap. Independent units are trajectories, not query repetitions. One host, descriptive; no hard wall deadline or native model claim.

| Policy | CPU ms/query | Region hits | LP calls | Closed /768 | Mean gap | Evictions | Wins vs LRU /12 | Peak charged bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step |7.988675|0|278|704|0.083333|0|4|0|
| LRU |7.800473|144|276|704|0.083333|472|0|900|
| area |7.559349|202|264|704|0.083333|414|8|900|
| weighted |7.994116|200|264|704|0.083333|416|7|900|
| density |7.693624|218|272|704|0.083333|398|6|964|

Density increases hits 51.39% vs LRU and 7.92% vs area, but increases numerical solves vs area (272 vs 264) and has 1.78% higher mean CPU. Density vs LRU saves 1.37% mean CPU, paired mean 0.106849 ms, median -0.017375, 10000-resample trajectory percentile bootstrap [-0.154107,0.387857] ms (seed 27). Area vs LRU saves 0.241124 ms, interval [-0.124860,0.607031]. Area vs step saves 0.429325 ms, interval [0.055377,0.824996]; this descriptive result includes the hard root. No evidence for promotion of density over area.

| Arrival | LRU hits | Area hits | Density hits | LRU CPU ms | Area CPU ms | Density CPU ms |
|---|---:|---:|---:|---:|---:|---:|
| Stable |82|90|98|12.671697|12.535277|12.563074|
| Shift |62|112|120|2.929250|2.583421|2.824173|

The stable cohort contains the hard root, so the between-cohort timing comparison cannot isolate the effect of arrival drift. Keep area as deterministic geometry-aware comparator. Density is optional.

## Verification and reproduction

Three targeted tests passed (1,200 exact hull membership comparisons using exhaustive simplex subsets, density stencil/budget/interior eviction fixture, 200 exact-oracle policy runs, 12 no-repeat trajectory checks). Separate replay independently reconstructs all 384 tasks, checks 12 anchors, 3,840 valid final intervals (3,520 closed), 7,506 stages, 3,514 candidates, 764 region hits, 320 query block transcripts, 1,700 evictions, 3,072 paired interval intersections. A second audit reconstructs all 398 density eviction decisions using exhaustive simplex membership, 3,840 charged-byte records, and all 321 capped block transcripts including the anchor. The first replay completed receipt checking but its density stage had a local variable-shadowing bug; fixed audit rerun passes. Campaign sources and timings unchanged. Both versions and logs archived. No exponential oracle at d=32. Full repository pytest, Ruff, Lean and Rust not run. Python and dependencies are inherited from SG-26; CPU only.

From an installed DataGoblin checkout with NumPy/SciPy/pytest, add the SG-07, SG-11, SG-15, SG-17, SG-20, SG-22, SG-25, SG-26 and SG-27 gym directories to PYTHONPATH. Or extract archive and run in its root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q test_sg27.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg27.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_sg27.py

`--limit` and `--resume` permit continuation of the same checkpoint only. Timing will change on rerun. Executed source blobs: original campaign f4b1c8af328a94114aed50ee7a72de2027209f7f; final campaign 3490ac7517cac1c8a284a0dad5310086e899c72f; tests f9dbc4f05c0e425b5021f1ca470c14913fe75a45; corrected audit bd11a139d9dfbe1484a6698ea2d19045cee8f598; initial audit fc8fa42b29ec5a36a90dd40bfd6d5c1cf76ca420. Raw results SHA256 d3f8ef5cb9698e14b877d3213de75f36188de80ad6e56eb0b8ecc31f0347c38e. Generated raw runs supplied separately, not committed.

Next: isolate the exposed nonclosed SG-26/27 roots and develop exact integer disjunctions (conditional label last-use bounds) to close residual integrality gaps. Distinguish proof certificates from numerical MILP proposals, and keep the hard exposed fixtures separate from new evaluation roots.

## Implementation custody

[DataGoblin implementation, campaign, tests and independent replay](https://github.com/MoralityLabAI/DataGoblin/tree/31256b90178bfd1ead55985840cfb22255f239b5/training/rlt_gym/sg27_spectral_density). All three Python files and the report match the local Git blob hashes. Complete 43-file archive `sg27_cpu_campaign.zip`, SHA256 `51874bd895f20d5192907f3b057a3a256bef789f07ac5ef2260fa12599050855`; corrected audit JSON SHA256 `3a83348afb2ddedb54404b329f06c603c56c95b113eb9ab5362c79591e87c0d1`. ZIP integrity and manifest hashes checked. Raw generated runs supplied separately.
