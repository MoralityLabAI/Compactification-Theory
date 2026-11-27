# SG-12: interruptible quotient search with monotone integer certificates

Scope: finite source-bound routing with additive utilities, fixed source coordinates and nonnegative assigned bit prices, d<=32 and T<=16. This is synthetic development, not a native model experiment. SG-11 remains available and unchanged.

## Bound and coverage

For a state with t visits remaining, suffix S, and current required/excluded outside components R,E, relax the cardinality constraints at visits 1..t-1 using integer multipliers. For component j, a_j is the sum of positive adjusted utilities at earlier visits (already activated); b_j is the best earlier last-use objective if not yet activated, including the option to remain unused. Both are computed by the SG-09 prefix recurrence. For the current visit, the absent contribution is a_j for j in S and b_j otherwise. The selected contribution is a_j+u_tj minus the last-use price if j is not in S. Let delta_j be selected minus absent. With the current read count k exact, the relaxed objective is the multiplier constant plus the sum of absent contributions, plus sum(delta_j for j in R), plus the largest k-|R| deltas outside R union E. This maximizes the relaxation subject to the current exact cardinality and forced choices; all legal completions are feasible in it. Minimize over the six fixed multiplier vectors and one reverse-coordinate-sweep root vector (valid integer proposals, not a claim of global dual optimality).

A partial union block splits on one undecided outside bit. Required and forbidden children are disjoint and exhaustive; infeasible blocks are discarded, and forced bits are normalized. When the union is resolved, maximize utility inside the already activated set, preserving component-index ties. Terminal t=1 uses exact constrained top-k. The complete suffix-state dominance rule is valid because greater accumulated value at equal (t,S) dominates every completion of the lesser state. New complete-state obligations are installed before old lower-value obligations become stale. Every child upper bound is capped by its parent; the incumbent can only improve. Heap removal only discards exhausted, dominated or certified-pruned obligations. Therefore upper and gap are nonincreasing at each event boundary and zero gap certifies the optimum. Work caps count events, not CPU time or all memory. There is no partial event or frontier eviction. Worst-case search is still exponential.

Independent replay reconstructs partitions with sets, directly enumerates each component's possible earlier last uses rather than using the prefix recurrence, independently scores the returned schedule, and recomputes every bound and gap. It shares input validation and the mathematical specification, not the engine's search or arithmetic implementation. The replay does not trust reported bounds or require the student to follow the default best-first branch policy. It is an independent software checker, not a formal proof or sandbox.

## Reproduction

From installed DataGoblin checkout:

```sh
export PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient:training/rlt_gym/sg12_spectral_blocks
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -s training/rlt_gym/sg12_spectral_blocks/test_spectral_blocks.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg12_spectral_blocks/sg12.py small
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg12_spectral_blocks/sg12.py large
```

Four targeted tests passed in 0.93 s: 10,933 feasible conditioned bounds against exhaustive completions and direct arithmetic, 12,292 partition-class checks, 96 capped or complete independent replay fixtures, and high-dimensional invalid-event/cap atomicity. Local Python compileall passed. Full repository pytest and Ruff not run. Python 3.13.5, CPU only. Rust unavailable. Dependencies match pinned SG-03/04/05/07/09/10/11 Git blobs.

## Fresh development cohort

36 small roots (18 d8/T6, 9 d9/T7, 9 d10/T8), 24 larger (9 d16/T8 with k=4/5, 9 d24/T10 with k=5/6, 6 d32/T12 with k=7/8), equal across persistent, rotating, mixed. Seeds 993000/994000/995000 and 997000/998000/999000 families are distinct from earlier studies. Same feasible initial heuristic. SG-11: 8192 quotient-class cap, 24 multiplier proposals, 80 decisions, no raw-action cap. SG-12: 256 or 2048 split/leaf/terminal events, six fixed vectors plus one root warm-start vector, no per-node dual proposals. These work budgets and bounds differ; the run is not a matched-work or matched-time comparison. One CPU timing per arm/root in randomized order, includes initialization and search, excludes reference optimum and replay. Original SG-11 replay 60/60; independent SG-12 replay 120/120; 12,398 total SG-12 replay events. Both SG-12 caps have identical prefix transcripts and nonincreasing gaps; every interval intersects the others. Small roots compared with exact SG-03 oracle; large roots have no separately solved exact DP optimum. Their zero gaps are independently replayed certificates under the stated integer bound and partition/dominance argument.

| Group | Arm | Certified | Mean gap | Mean process CPU ms | Charged work |
|---|---|---:|---:|---:|---:|
| small | SG-11 |35/36|0.277778|2.078451|7346 classes|
| small | blocks256 |36/36|0|0.867946|1025 events|
| small | blocks2048 |36/36|0|1.003041|1025 events|
| large | SG-11 |6/24|13.75|10.922316|50832 classes|
| large | blocks256 |17/24|3.75|4.560522|3555 events|
| large | blocks2048 |24/24|0|9.096158|6793 events|

SG-11 small mean exact regret 0.25; SG-12 both 0. At d16, 24, 32: SG-11 certified 5/9, 1/9, 0/6; SG-12 256 events 9/9, 8/9, 0/6; SG-12 2048 events 9/9, 9/9, 6/6. At d24/32 SG-11 cannot expand the root within its class cap (15/15 roots had no complete expansion, one certified from the root dual bound). SG-12 d32 uses 4658 events across 6 tasks; max 1226 on one task. Three d32 roots have 10,518,300 possible raw root actions; none enumerated all. Peak stored blocks 1052 at larger cap, includes stale blocks; not a byte or RSS measurement. The 2048-event small arm did identical work to the 256 arm; timing differences are measurement noise. No learned allocator, no native transformer, no GPU or paid API. Results are single-host development, no confirmation claim.

Executed sources Git blob hashes: spectral_blocks.py cb8cd3abdf866c100a85da56dad5619f0e3bd951; spectral_block_replay.py 8936a92cfbbfebbd7574f9ecf2645c37823c6f34; sg12.py 8ad41f64365f17429a479a6fb0296bd3991f6818; test_spectral_blocks.py 4ff28dd828aaa03a690d3f8d5c81af6651abcbf1. Raw JSON SHA256: small c915f359ea3c022ec0245529651734326b71129a7da46dccb30ea6bb426b1328; large a91be0a64ade5e5abdbfb6dd6eeb00a5856cbcfd20a8fe52f707e8db3989c30f. Full archive includes raw data, logs, sources and dependencies. Do not treat exposed roots as independent confirmation.

## Implementation custody

[DataGoblin code and reproducible campaign](https://github.com/MoralityLabAI/DataGoblin/tree/b2731209618193766502cf6821b718c403f80f6d/training/rlt_gym/sg12_spectral_blocks) · [Block engine](https://github.com/MoralityLabAI/DataGoblin/blob/b2731209618193766502cf6821b718c403f80f6d/src/datagoblin/training_gym/spectral_blocks.py) · [Independent replay](https://github.com/MoralityLabAI/DataGoblin/blob/b2731209618193766502cf6821b718c403f80f6d/src/datagoblin/training_gym/spectral_block_replay.py).

Full downloadable archive `sg12_cpu_campaign.zip`, SHA256 `76c3f5bf95c4598a453dd04de2d1aac753c498a22f5fe7d6feaa730e5a3cc1e8`. Generated runs are supplied separately and not committed. The five source and documentation blobs in the DataGoblin commit match the local files byte-for-byte.
