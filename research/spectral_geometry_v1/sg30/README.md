# SG-30: conditional factorization and shared proof tables

## Construction

At a last-use domain box, use the fixed root rational vector to calculate each component's conditioned maximum and the regret threshold H=B-D*(L+1) for improvement over the current global incumbent L. Conditional per-read maxima (including impossible read conditions) yield mandatory and permitted bits. Repartition the residual read graph at each node, using up to two separator rows. Enumerate local masks consistent with the box and screened bits, and compute minimum regret profiles. If an individual table exceeds the remaining logical budget, replace it by the zero-regret relaxation over separator counts. A complete feasible combination can propose a legal schedule; partial tables only tighten an upper bound. A failed or weak factor attempt falls through to SG-28 conditional LP, integral-retention recovery and last-use branching. Retain the better of the checked bounds. Each split covers its parent. A factor leaf's recorded integer cutoff may be below the final incumbent; the proof shows either an upper bound on strictly improving schedules or that none exist. The final upper bound is the maximum of leaf bounds. No numerical solver status serves as a proof.

The canonical table signature includes local label IDs, last-use intervals, mandatory/permitted bits, scaled utilities/prices/maxima, local row IDs, active separator axes and residual counts. A separator axis that no label in the factor can read has identically zero free count, so remove it from the key and restore zero when convolving. Profiles are untruncated by H after screening; this permits reuse for different cutoffs when the screened local problem is exactly identical. Use a task-local 64-entry LRU for computed tables, plus exact-key proof table interning; no approximate/hash-only acceptance. The proof bank omits tables belonging solely to overwritten nodes. The replay reconstructs domain boxes, independently evaluates the conditioned last-use maxima using an explicit prefix sum, recomputes each referenced profile with alternate mask enumeration and reversed label order, verifies all exact keys and reconstructs the aggregate bound. It shares the partition and convolution helpers; it is not a fully independent formal checker.

Caps: 63 tree nodes and 100000 logical table/convolution units per acquisition, including unsuccessful factor attempts. Memo hits incur the same logical charge as misses, so the cap does not alter paired decisions. Computed-work diagnostic subtracts the estimated table cost on hits; it is not a count of CPU instructions. The work estimate excludes conditional screening, separator enumeration, LP solves, key hashing, path copying and other Python overhead. The final proof work can be smaller than the charged acquisition work. The field factor_leaves counts completed factor attempts, including those subsequently discarded, not surviving leaves. The 64-record computation cache does not bound total RAM; proof interning dictionaries, tables, transient profiles and verifier storage are outside this record cap. Trusted research implementation within the existing d<=32,T<=16 task domain, not an adversarial deserialization boundary. No production default changed.

## Campaign and results

Four cohorts, 105 task instances in total, 528 policy runs. Twelve exposed independent triangle problems from SG-28/29 form the forced-split control: fix initial branch (label 0, last use <=0 versus >=1) and compare memoized versus recomputed factors only, two randomized repetitions. Nine new three-shared-row triangle instances (2,3,4 groups, costs 1,2,3) extend SG-29 with three common rows of budgets 1,2,3. Exact optimum g*(30-2*c)+60. Eighteen fresh random tasks (six per utility regime, seeds 3010000..3010005, 3011000..3011005, 3012000..3012005, d=32,T=12,k=7/8). Both these cohorts have two randomized arm orders. The exposed cohort uses all 66 SG-28 frozen inputs in their recorded order, one randomized arm order per task: original SG-27 schedule/vector, regenerated SG-26 tasks with fresh common preparation where no original vector is saved. The exposed tasks are two related trajectories, not 66 independent roots. No sealed confirmation. The forced-split control was designed to isolate reuse; it is not a recommended acquisition route. The first exact-only smoke implementation did not produce cross-branch sharing until inactive separator axes were normalized; final source adds partial relaxations and exact projection. Smoke results excluded from timings.

All natural arms share the same SG-29 prepare (chain LP plus warm repair), except already recorded exposed inputs. SG-28 tree without MILP, static root factor attempt plus tree, dynamic factor attempts plus tree, and memoized dynamic. All use the same tree cap; the hybrid has additional table work. Timing is process CPU plus shared preparation and one replay, averaged across task-level repetition medians. SG-28's internal replay is excluded from its recorded acquisition CPU. Separate post-run audit excluded. No matched hard wall deadline. Single-host descriptive comparison.

| Cohort | Policy | Closed/runs | Mean total CPU ms | LP calls | Nodes | Mean receipt bytes |
|---|---|---:|---:|---:|---:|---:|
| Forced | dynamic |24/24|21.976614|24|72|634.25|
| Forced | memo |24/24|21.065540|24|72|611.25|
| Fresh | tree |36/36|10.320401|0|36|276.39|
| Fresh | static |36/36|10.426054|0|36|285.39|
| Fresh | dynamic |36/36|10.407444|0|36|285.39|
| Fresh | memo |36/36|10.414031|0|36|285.39|
| Three bridges | tree |10/18|138.456175|670|670|2442.44|
| Three bridges | static |10/18|170.329685|670|670|2451.44|
| Three bridges | dynamic |10/18|326.381518|660|670|2533.00|
| Three bridges | memo |10/18|328.511307|660|670|2533.00|
| Exposed | tree |66/66|21.769369|123|156|357.50|
| Exposed | static |66/66|28.121565|123|156|368.47|
| Exposed | dynamic |66/66|39.434740|123|156|368.48|
| Exposed | memo |66/66|40.505058|123|156|368.48|

All natural arms have identical closure counts. Three-bridge cohort leaves eight gap-one runs per policy; all other cohorts close. Forced control saves 60/218 table records, reduces logical 15072 units to 9312 computed units (38.22% saved), and reduces compact JSON receipt size from 634.25 to 611.25 bytes (3.63%). Mean CPU saving 4.15%, wins 9/12 paired tasks. Proof includes final schedule, interval, tree and bank; tree comparator has depth field and hybrids have work field. Natural three-bridge memo hits 984 but only 3552/926718 logical units avoided, final table count 38 for either hybrid. Exposed memo hits 1753 but only 7366/909939 units avoided, zero final table records. Memoization benefits genuine reused subproblems; redundant attempts are the dominant cost here. Do not promote dynamic factorization. Keep cheap root-floor, static factors when tractable, SG-28 conditional disjunction fallback. Next: cost-aware branch/LP-first admission and stronger separator relaxations on genuinely coupled cases. Investigate whether conditional regret screening can be cached separately from LP proposals.

## Verification and reproduction

Four targeted tests passed: 15360 exhaustive conditioned read-maximum comparisons across 64 small tasks (both prefix and explicit reference formula), 288 small exact-oracle policy receipts across three cap settings, forced split sharing and tampering, partial-cap and negative fractional floor. Post-run audit checks 528 valid receipts (496 closed), 3592 proof nodes, 452 unique table rechecks, 144 paired interval intersections and 144 exact memo/recompute policy comparisons. All pass. No exponential oracle for the d=32 tasks. Full repository pytest, Ruff, Lean and Rust not run (isolated source checkout). Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0. CPU only, no paid inference or native model claim.

Install DataGoblin and optional NumPy/SciPy/pytest; set PYTHONPATH to the earlier SG gym directories used by SG-29, plus SG-30. The archive contains the exact local dependency copies and frozen exposed inputs. From the extracted directory:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q test_sg30.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python campaign_sg30.py forced
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python campaign_sg30.py fresh
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python campaign_sg30.py bridge
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python campaign_sg30.py exposed --limit 33
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python campaign_sg30.py exposed --start 33
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_sg30.py

Do not mix checkpoints. Raw timing runs are supplied separately, not committed. Source hashes: sg30.py c391f04431d178e5c6644d5a6f913c612f3d773f; campaign 17adc7296ebb64f0358ca5284a16869616df07f4; tests 21dcd9158b793d67497796e1ac2df7614e9ef19f; audit bd475c458146bb6c1f9a15085f1735b33b98bc1c. Raw SHA256: forced 9da1aea727d923bf4d41ac9017a1098c83641ece2b5279f7c0dc15705ae0117b; fresh 6525d03668da24c24dbbc0ecb22f6a5463a7dc83a5e10157ef8bc5349a316116; bridge a5115cc2af3a08545c06ca3151b7b3f6e1b587d4c3039a35670a787da5bab0e0; exposed ded9d821a7bc2137599dc9892085a7d26f18bf489d29b5e90b0d5f3b6bc6d563.

## Implementation custody

[DataGoblin SG-30 implementation, campaign, tests and replay](https://github.com/MoralityLabAI/DataGoblin/tree/f0ee54ee84b87c5a4f70fcc05e9467a4d202bee8/training/rlt_gym/sg30_spectral_dynamic). All four committed Python source files and the report match the local Git blob hashes. Archive `sg30_cpu_campaign.zip` contains 57 files, SHA256 `d745e3cdc75d33cb131c6acbfc1dac38b1ea81057254e5cfbc6fce0d481e85e4`; audit JSON SHA256 `ba9a0633dae65f632118a8510a410d7bb00f456c1da6578b5a43cb0f769cb282`. ZIP integrity and every manifest hash checked. Includes the frozen exposed inputs and the final source, initial smoke source, dependency copies, results and logs. Raw receipts supplied separately.
