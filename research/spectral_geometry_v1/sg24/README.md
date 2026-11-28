# SG-24 / SG-24B: exact event-driven certificate boundaries

Finite additive source-bound routing, d<=32,T<=16, fixed row budgets and a validated SG-21 affine utility/effective-price ray with known horizon. The affine rational multiplier line and one fixed feasible schedule define nonnegative convex scaled gap g(s)=D(B(s)-F(s)). A certified anchor has g(0)<D. All integer positions strictly before the first crossing g(s)=D remain optimal by flooring, including negative objectives. Threshold zero locates the right edge of the initial exact-face plateau. A face exit need not invalidate a floor certificate. Maximality concerns the fixed schedule, multiplier line and horizon. Independent endpoint arithmetic remains SG-15. Trusted internal research helpers, not an adversarial execution boundary.

## Mathematics

For each component and relative position s, adjusted utility is z_i+s*dz_i. For last use l the relaxed value is z_l+s*dz_l+sum_{i<l}max(0,z_i+s*dz_i)-p_l-s*dp_l; unused contributes zero. At most T sign breakpoints and T+1 open intervals yield at most 1+T(T+1) genuine affine mask lines. The maxima agree at endpoints by continuity. Subtract the chosen mask's affine value from every line; the sum of per-label upper envelopes equals the scaled gap because row multipliers cancel on a feasible schedule. Sort slopes, keep largest intercepts, construct upper hulls by rational intersections, and sweep merged events until g reaches D. The last covered integer is ceil(crossing)-1 relative to the anchor. If both endpoint regrets of a label vanish, its nonnegative convex regret is zero throughout. The envelope planner tests the horizon first, skips such labels when it fails, and computes the last covered integer endpoint. O(d*T^2) candidate generation plus sorting and rational envelope arithmetic; no bit-complexity or universal runtime claim.

SG-24B supporting-line search computes each label's best last-use alternative with one positive-prefix scan at a rational position. Ties select maximum slope. Its aggregate active line a+b*s is a global lower bound. From a failing position g(s)>D, b>0 since the anchor is certified. Moving to (D-a)/b cannot go before the true first crossing, strictly decreases s and leaves the gap >=D. The previous active line cannot be used for another strict move; the envelope is finite. Equality proves the crossing. Threshold zero handles the initial exact-face plateau. A 128-probe cap falls back to the explicit envelope. Each support evaluation requires O(d*T) arithmetic, total work depends on visited lines. No floating arithmetic in the crossing calculation; all endpoint proofs independently checked.

## Campaign

Two distinct fresh exposed synthetic cohorts, each 24 independent trajectory roots (8 per persistent/rotating/mixed regime), d=32,T=12, alternating budgets 7/8. Half pure global lambda+1 per query, half also four +/-1 utility-column changes. SG-24 has 16 edited steps per trajectory; SG-24B extends the direction to 32 on fresh roots, adds supporting-line search, and runs in three planned eight-root checkpoint batches. Two randomized whole-trajectory repetitions. Corrected baseline uses SG-22 update/repair/secant/correction/fallback. Interval planners share gate: identical consecutive feasible schedules, at least two future positions, and successful next-step secant certificate. Then plan the maximal integer prefix by the assigned method. Covered queries use chord interpolation; rebuild preceding state at next uncovered query. Process CPU includes gate, state update, solver, planning, receipt; excludes common ray/anchor setup, independent audit and history update. Full endpoint probe counts exclude common gate evaluations; support scans are lighter arithmetic, not State.build. Independent units are trajectories. Single-host descriptive development, no matched hard deadline or sealed confirmation cohort. Existing production defaults unchanged.

### SG-24

| Arm | CPU ms/query | Planning us/query | LP calls | Gates | Reused | Regions | Full endpoint probes | Certified /768 | Wins vs correct /24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct |3.500758|0|218|0|0|0|0|768|0|
| binary |3.556756|166.300|218|288|308|116|378|768|10|
| events |3.630268|179.316|218|288|308|116|212|768|10|

116 paired region proofs agree apart from diagnostics; 96 finite crossings, 7470 generated distinct lines total, maximum span 12. Events reduce full endpoint probes 43.92% but take 2.07% more CPU than binary and 3.70% more than corrected. Paired trajectory mean savings vs corrected -0.129510 ms, descriptive bootstrap percentile interval [-0.248417,-0.028491] ms (10,000 resamples, seed 24). Pure global correct/binary/events 2.595864/2.638355/2.616987 ms, mixed 4.405652/4.475158/4.643550 ms.

### SG-24B

| Arm | CPU ms/query | Planning us/query | LP calls | Gates | Reused | Regions | Full endpoint probes | Support scans | Certified /1536 | Wins vs correct /24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| correct |1.509128|0|150|0|0|0|0|0|1536|0|
| binary |1.359329|90.647|150|252|1162|116|420|0|1536|19|
| events |1.321157|78.710|150|252|1162|116|188|0|1536|21|
| support |1.375891|82.539|150|252|1162|116|0|372|1536|19|

All three planners reproduce corresponding schedules, multiplier vectors, objectives and region endpoints. Events save 12.46% vs corrected and 2.81% vs binary; win 14/24 vs binary. Support saves 8.83% vs corrected, slower than both other planners in mean CPU. Zero envelope fallbacks, maximum seven support scans including final integer endpoint, 72 finite crossings per continuous method, maximum span 29. LP-call counts unchanged. Mean paired savings vs corrected: events 0.187971 ms, bootstrap [0.139069,0.232517]; support 0.133236 ms, [0.056162,0.203156] (10,000 resamples, seed 24). Pure global correct/binary/events/support 1.299873/1.164879/1.122336/1.157447 ms; mixed 1.718382/1.553779/1.519977/1.594335 ms. Cohorts differ in roots and horizon; cannot attribute differences to horizon alone. Retain stepwise corrected general baseline, envelope planning optional on known affine batches.

## Verification and reproduction

Seven targeted tests passed: 4640 exhaustive per-label envelope comparisons, 640 exhaustive aggregate crossings, 720 tiny ray positions checked against exact integer optima, 144 complete small oracle policy runs, negative fractional and integer-boundary fixtures, invalid/tampered endpoints, 1280 supporting-line comparisons (597 forced cap fallbacks), 80 support-region parity checks. Separate replay independently reconstructs all 1152 distinct query tasks and checks 48 anchors, 8448 final receipts, 9946 acquisition stages, 5958 candidate receipts, 580 regions, 1160 endpoints, 408 next-step maximality checks, 240 continuous rational crossings (explicit last-use gap equals one), 4102 interval-covered receipts, 6144 paired objective comparisons, 3840 planner-output comparisons. All pass, zero block fallback. No exponential oracle at d=32. Full repository tests, Ruff, Lean and Rust not run. Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0. CPU only; no GPU, paid inference or native model claims.

Install DataGoblin and optional NumPy/SciPy/pytest. From repository root set PYTHONPATH to SG-07, SG-11, SG-15, SG-17, SG-20, SG-21, SG-22, SG-23 and SG-24 gym directories. Or extract the archive and run from its root:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -s test_sg24.py test_sg24_support.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg24.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg24b.py --limit 8
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg24b.py --resume --limit 16
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python sg24b.py --resume --limit 24
    python audit_sg24.py

Do not mix checkpoints. Reexecution changes timings. Archive preserves actual sources, logs, results and dependency copies. Raw generated runs supplied separately, not committed. Executed Git blobs: events 0f93f635c26e255e5136841b4984501b73816f74; support 1e74993f0b6024bc000653e06ac07fc304c4d2f3; SG-24 20751260a234a15786aa30e1bbaf344c0da0a9cf; SG-24B ba858bdeb4fc5fb236885ef6c29ed252ca91f6a8; tests d152c6d535561221aafe7e767218f852e16d0811 and 407fc33325f0a1e45c086fe9171ed4dd8d5e5178; audit 2593f926b04420d42ce0ba965abd7be498cdf797. Raw results SHA256 976b3e9abb9676d10d865ea839101ea19338b6cbc0fca6e0ac3722101f17979b and 59d9488bdadaec50f16ff6549535eeab65780372ba28e9ef33fb122a1477f3aa. Archive sg24_cpu_campaign.zip SHA256 66d982277f9671fac028d68fae6b76a21f3e2935cb45178f8f70de5d995aa0d1. All 47 archived files present and integrity/manifest checks passed.

Next: investigate two-dimensional affine certificate regions with multiple simultaneous edit directions and online changes to direction. Test compact facet or endpoint proof representation and amortized region lookup; keep exact replay and compare against the stepwise corrected baseline on fresh direction-switching trajectories.

## Implementation custody

[DataGoblin executable campaigns, crossing implementations, tests and independent replay](https://github.com/MoralityLabAI/DataGoblin/tree/bd571cbce2f7c807ebd11c4f867aa6395661454b/training/rlt_gym/sg24_spectral_events). All seven Python files match the executed source blob hashes. Archive `sg24_cpu_campaign.zip` contains 47 files; SHA256 `66d982277f9671fac028d68fae6b76a21f3e2935cb45178f8f70de5d995aa0d1`. Generated timing runs are supplied separately.
