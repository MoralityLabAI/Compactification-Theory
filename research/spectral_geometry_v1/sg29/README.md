# SG-29: additive integer certificates with bounded row separators

## Construction and scope

For the finite SG-28 routing objective, fix rational multipliers eta=e/D and a legal incumbent with integer value L. Let M_j be the maximum scaled relaxed component-mask score and B=e*k+sum_j M_j. On a legal schedule A, the sum of nonnegative scaled regrets M_j-f_j(m_j) equals B-D*F(A). Any strictly improving schedule has total regret <=H=B-D*(L+1). If H<0, the original integer-floor bound closes. Otherwise independently maximize each component score conditioned on each read being absent or present. A condition whose minimum regret exceeds H is impossible in an improving schedule; therefore fix the complementary read. Remove fixed reads from row budgets and make an incidence graph of remaining free reads between components and rows. A residual budget outside the available range proves no improvement.

After removing a selected separator of zero, one or two row vertices, connected components partition the labels and all other rows. The local table records the minimum scaled regret for each possible vector of separator read counts, enforcing all local residual budgets. Enumerate each label's admissible masks and convolve their counts by min-plus DP. Since regrets are nonnegative, discard entries above H. Combine the group tables by min-plus convolution, enforcing every separator budget exactly. If a table exceeds the work budget, substitute zero regret for every separator profile; this is a relaxation, not a certified local optimum. Empty combined profile proves no improving schedule. Otherwise min regret R yields upper max(L,floor((B-R)/D)). If all group tables are exact and a combination exists, reconstruct its feasible schedule and independently score it. The proof covers all improving schedules even when the incumbent itself was removed by screening.

The replay checks the original incumbent, multipliers and scaled bound, recomputes screening using an explicit last-use formula, verifies disjoint label and local-row coverage and all free-read edges, then recomputes tables in reverse label order with independent mask enumeration and direct objective scoring. It checks all profiles and the combined upper bound. Missing tables only relax constraints. It uses the same DP/convolution structure, not a formally independent implementation of every helper. No numerical solver status is a proof. The proof is a factorized collection of local tables, not a general-purpose proof-DAG compiler or arbitrary-overlap solver.

Cap 100000 conservative table/convolution units, calculated from product of residual count ranges times number of candidate masks, plus an upper bound on separator convolution pairs. It excludes screening, separator enumeration, tuple/path copying, integer bit complexity, LP preparation and general Python overhead. Zero, one and two row separators are enumerated using a preflight proxy; factor work is linear in group count for fixed group sizes and fixed separator capacity, but screening and separator discovery need not be linear. Existing d<=32,T<=16 domain, no hard time or memory cap. Trusted research code, not adversarial execution boundary. Existing defaults unchanged.

## Executed comparison

51 tasks, two randomized arm orders per task. Common preparation once per task: SG-17 chain LP and SG-20 repair. Compare root integer floor, SG-28 tree without MILP at cap 63, and SG-29 with up to zero/one/two separator rows. Time is process CPU; total adds shared preparation, acquisition and one replay (tree internal replay excluded from its reported acquisition timing). Report mean across task-level two-run medians. No matched wall-clock cap; different work units. 12 independent triangle-cover tasks are exposed SG-28 regression cases. 12 new single-shared-row and nine new double-shared-row stress cases append rows with utility 10 for every label and budgets 1 and 2. Every group needs two labels; cross-group negative reads lose 110 utility, exceeding maximum possible activation-cost saving. Shared rows can use already activated labels. Exact optimum g*(30-2*c)+10*sum(1..b). For two bridges g<=4 to remain within T<=16. The 18 random tasks are separate fresh seeds 2910000..2910005, 2911000..2911005, 2912000..2912005, d=32,T=12. No sealed confirmation. Four checkpoint batches, same source, no tuning. Raw results retain every open interval.

| Workload | Root closed | Tree closed | Factor0 closed | Factor1 closed | Factor2 closed |
|---|---:|---:|---:|---:|---:|
| Independent |0/24|24/24|24/24|24/24|24/24|
| One bridge |0/24|20/24|12/24|24/24|24/24|
| Two bridges |0/18|10/18|6/18|6/18|18/18|
| Random |34/36|36/36|34/36|34/36|34/36|

| Workload | Tree total CPU ms | Factor matched separator ms | Tree mean proof B | Factor mean proof B |
|---|---:|---:|---:|---:|
| Independent |95.815339|6.758871|1613.17|497.33|
| One bridge |110.822|9.637|1640|552.25|
| Two bridges |128.422|13.265|2334|626.56|

The first 12 cases close with 42 factors instead of 288 tree nodes (first repetitions), mean serialized certificate 69.2% smaller and total CPU 14.18x faster. On retention cost 3 independent family g=2,3,4,5, tree nodes 7,15,31,63; factors 2,3,4,5; tree bytes 465,960,2001,4242 versus factor bytes 342,440,552,674. Matched separator factor work proxy 194*g (zero shared rows), 388*2*g (one shared row) and 2340*g (two shared rows), on these planted cases. The serialized size counts compact JSON of the replayable receipt including the final schedule and interval, with proof metadata (tree nodes/depth or factor work); factor proof also includes its initial schedule. No claim that a factor and a tree node have equal checking cost. The bridge tree control does not certify every case within its cap, so timing is not a matched-closure comparison. Random root 2910002 retains gap one under factor cap, while SG-28 closes in three nodes; root-only already closes the other 17. Use the cheap root check first, and preserve SG-28 fallback. Random mean total ms root 10.934, tree 14.055, factor0 11.533, factor1 11.530, factor2 12.061.

Separate exposed hard-root regression uses SG-28 frozen 66 input tasks. SG-27 uses original legal schedules and rational vectors; SG-26 has no saved original rational vector in that input file and uses new common preparation (as in SG-28 its task inputs were reconstructed from the generator). Factor2 closes SG-26 25/33, SG-27 0/33, improves no incumbent; mean gaps 1.303 and 1, acquisition CPU excluding new SG-26 preparation and replay 2.772 and 9.491 ms respectively. The method does not replace the successful SG-28 primal recovery and disjunction fallback on these coupled instances.

## Verification and reproduction

Four targeted tests passed: 1536 exhaustive conditional-mask screening checks, 864 tiny exact-oracle receipts at three caps and separator limits, planted separator cases, tampering and negative rational floor. Post-run replay checks 510 comparison receipts (388 closed; 306 factor and 204 tree/root) and 102 paired interval intersections; 66 additional exposed factor receipts (25 closed). All 576 valid. No claim of independent exponential oracle at d=32. Full repository pytest, Ruff, Lean and Rust not run; isolated dependency checkout, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, CPU only. No paid inference or native performance claim.

Install DataGoblin and optional NumPy/SciPy/pytest; from repository root add SG-07, SG-11, SG-15, SG-17, SG-20, SG-22, SG-25, SG-26, SG-28 and SG-29 gym directories to PYTHONPATH. The archive contains exact dependency copies. From its extracted root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q test_sg29.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg29.py --limit 12
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg29.py --resume --limit 24
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg29.py --resume --limit 33
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg29.py --resume --limit 51
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg29_regression.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_sg29.py

Raw results SHA256 ab2c201e0d2979e71cfc33f3d9ebfac14d27884ce6a5b34012391c382abd0d2c; regression e2a2814d096bd70b47045dfcaadfc700399b856c5b7fb27e458df162c906ecd2; audit 4e918edee454ca95a07c8327f2e5851117e2201ee4977060e94c477104c68966. Generated raw runs supplied separately, not committed.

Next: use conditional regret screening inside SG-28 branch nodes, identify independent residual components after each split, and memoize exact boundary-count tables under canonical subproblem signatures. This aims to shrink proof size on genuinely coupled hard roots rather than only on planted separated families.

## Implementation custody

[DataGoblin SG-29 implementation and independent replay](https://github.com/MoralityLabAI/DataGoblin/tree/0d125c7094f161816e7dadb6e4e04077981e081e/training/rlt_gym/sg29_spectral_factors). All five committed Python source files match the executed Git blob hashes. Archive `sg29_cpu_campaign.zip` contains 50 files, SHA256 `569a6f0a981bf48d2534a57459e597e520c141f21add701fb5a0c71682661e0b`; ZIP integrity and every manifest hash checked. Raw receipts supplied separately, not committed.
