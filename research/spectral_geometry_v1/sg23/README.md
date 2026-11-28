# SG-23 / SG-23B: active dual-face multiplier velocities

Scope: finite additive source-bound routing, d<=32,T<=16, fixed dimensions and row budgets, affine utility and effective last-use price direction. At a primal/dual equality anchor, the component masks used by the feasible schedule are relaxed maximizers. For each component j, let M_j be the anchor-optimal masks and m_j the chosen one. With utility and price slopes du and dp, define D_j(m;v)=sum_{i in m}(du_ij-v_i)-dp_j(last(m)). The active-face condition is D_j(m_j;v)>=D_j(m;v) for all m in M_j. This is a polyhedron in v for a fixed task direction (a cone in joint task/multiplier directions). Equivalently the sum of nonnegative per-label directional regrets is zero. The condition is necessary and sufficient for the chosen mask to remain maximizing on some sufficiently short right interval, because inactive masks have strictly positive initial slack and there are finitely many masks. It does not certify a full integer step: inactive inequalities may become violated. The endpoint must be checked. Fractional-floor-only anchors use the existing fallback; they do not satisfy the exact-face premise.

The existing SG-15 face boxes (required a, possible b) represent every optimal mask. In a box, maximize directional utility by taking required terms plus positive parts on optional bits. The price slope is fixed by the box's last use. In the chosen box, optional bits already chosen require du-v>=0; omitted ones require du-v<=0. These sign constraints eliminate those epigraph variables in every alternative box for that label. Other optional bits use one shared p>=0,p>=du-v per label and visit. Remaining box inequalities compare the resulting affine expression to D_j(m_j;v). Duplicate rows are tightened; single-variable rows become bounds. All proposed directions undergo exact integer-scaled directional regret checking, then SG-20 endpoint construction and repair, and independent SG-15 verification. The numerical LP is only a proposal tool. Solver status or infeasibility of a particular proposal is not an optimality proof. The fixed-direction model uses differences of effective prices; it applies to an affine interpolation of these quantities, not arbitrary nonlinear parameter paths.

Three proposal strategies: clamp the rounded previous certified secant slope (or zero for the first step) to the single-variable bounds; clamp plus one reverse active-face coordinate sweep; clamp plus a compact numerical LP minimizing L1 distance to the clamped seed. For a sweep, compute best directional scores F0,F1 over anchor-optimal masks with a given bit excluded/included, excluding the velocity charge in F1. Masks forced to include the bit reduce the residual budget. The remaining thresholds F1-F0 yield the exact coordinate minimizer interval by the usual order-statistic argument. Select the closest point to the old velocity in that interval. The sweep is exact per coordinate, not a guarantee of global cone feasibility after one pass. SG-23B maintains each box's directional score and updates it by the changed required term or positive-part difference, giving O(T*N_boxes) sweep arithmetic after initialization, excluding sorting. Failed cone candidates are discarded and the original SG-22 corrected extrapolation/SG-20 fallback runs on the retained state. Accepted cone candidates must close the endpoint certificate after optional repair. The initial state and every final query are audited; all final endpoints use unchanged verifier. Existing production defaults unchanged.

## Campaigns

Two separate fresh exposed synthetic cohorts, each 24 independent trajectory roots (8 per persistent/rotating/mixed utility regime), d=32,T=12, alternating k=7/8, 16 queries per trajectory. Each has 12 pure global lambda+1 per step trajectories and 12 mixed trajectories with four further +/-1 utility changes per step. Two randomized whole-trajectory repetitions per arm. Common anchor and task construction outside timing. Process CPU includes patch, proposal construction, numerical cone LP if needed, endpoint recovery, fallback and receipt creation; excludes independent replay and history updates. The full-LP-every policy retains the previous schedule and forces a new LP. All full LP counts include chain and direct proposals; additional cone LP calls are shown separately. The first cohort compares five policies, the second compares the original and incremental face sweep with corrected extrapolation on distinct roots. Each run split into three planned eight-root batches using the same source and checkpoint. No timing tuning or sealed confirmation. Independent units are trajectories, not queries or repetitions.

### SG-23

| Policy | CPU ms/query | Full LP calls | Cone LP calls | Exact directions | Accepted after repair | Raw closed endpoints | Certified /768 | Wins vs corrected /24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| corrected |3.268933|198|0|0|0|0|768|0|
| clamp |3.721933|206|0|426|376|370|768|3|
| sweep |3.170044|158|0|608|502|490|768|10|
| cone LP |3.221190|134|200|668|532|514|768|8|
| LP every |7.251429|768|0|0|0|0|768|0|

Sweep reduces full LP count 20.20% and mean CPU 3.03%. Cone LP reduces full LP calls but makes 334 total numerical solves, more than the corrected control. Cone LP reported 22 infeasible solver returns; 52 calls in each cone arm found inconsistent simple bounds; no representation-cap exits. Average paired CPU saving of sweep 0.098889 ms, trajectory bootstrap percentile interval [-0.298120,0.537479] ms (10,000 resamples, seed 23). Pure global corrected/sweep 2.365341/2.183852 ms; mixed 4.172524/4.156236 ms. The compact cone often finds local directions that fail the finite endpoint; 1702 exact directions checked, 1374 close without repair, 1410 close after repair. Do not count a zero directional regret as a final certificate.

### SG-23B

| Policy | CPU ms/query | Full LP calls | Accepted cone endpoints | Certified /768 | Wins vs corrected /24 | Wins vs sweep /24 |
|---|---:|---:|---:|---:|---:|---:|
| corrected |2.667295|144|0|768|0|14|
| sweep |2.540044|112|550|768|10|0|
| fast sweep |2.478302|112|550|768|9|16|

Fast and original sweep agree on every non-timing field of all 768 paired query receipts. Fast reduces mean CPU 2.43% versus original sweep and 7.09% versus corrected, and full LP calls 22.22% versus corrected. Paired trajectory mean savings against corrected 0.188993 ms, bootstrap percentile interval [-0.258043,0.693232] ms (10,000 resamples, seed 23). Pure global corrected/sweep/fast 1.941553/1.936897/1.965309 ms; mixed 3.393038/3.143191/2.991296 ms. Both cohorts show reduced numerical full acquisitions but timing savings are inconsistent across roots. Keep SG-22 corrected extrapolation as the default comparator. No native model performance claim, no GPU or paid inference.

## Verification and reproduction

Four targeted tests passed: 64 exhaustive small exact-face anchors, 768 rational directional-gap and epigraph parity checks, 7872 coordinate grid comparisons, exact incremental-sweep equality, local-versus-finite obstruction, inconsistent-bound fixture, fractional-anchor rejection, changed-budget rejection, 288 complete small exact-oracle policy runs. Separate audit reconstructs all tasks from endpoints by integer interpolation, checks 768 distinct query signatures across the two cohorts, 48 anchors, 6144 final certificates, 13352 acquisition stages, 6884 candidates, 3018 directional proposals with 96576 independent per-label lexicographic last-use checks, 3018 raw endpoint receipts, 4608 paired objective comparisons, 768 full non-timing receipt comparisons. All pass; no block fallback. All 6144 final gaps zero. No exponential oracle at d=32. Full repository tests, Ruff, Lean and Rust not run. Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0. One host; numerical LP time limit is 0.1 s per cone solve, not a hard acquisition deadline.

Install DataGoblin and optional NumPy/SciPy/pytest; from the repo root:

    export PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient:training/rlt_gym/sg15_spectral_face:training/rlt_gym/sg17_spectral_chain:training/rlt_gym/sg20_spectral_delta:training/rlt_gym/sg21_spectral_regions:training/rlt_gym/sg22_spectral_moving:training/rlt_gym/sg23_spectral_velocity
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -s training/rlt_gym/sg23_spectral_velocity/test_sg23.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg23_spectral_velocity/sg23.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg23_spectral_velocity/sg23b.py
    python training/rlt_gym/sg23_spectral_velocity/audit_sg23.py

For the supplied archive, run directly from the extracted directory. To reproduce planned batches use --limit 8, --resume --limit 16, --resume --limit 24. Do not mix unrelated checkpoints. Source blob hashes: cone 5b505d2153ea9dbdd0e8aaff185e66ca45556ba6; fast 43cadecbe54dabc4b903862c5e138508b84c05e2; SG-23 71a0e8c66bc82ebe42b77aa8412d5772d85a63d9; SG-23B 12a901dd7adba101c38c82b72e59cdd0220e81f7; tests 18078e8369fb3e2805ee4cdd424c9c5f805df626; audit f68059c0456d4ce74687634fae74706126d09e6e. Raw result SHA256: SG-23 f3f45b8fdc20b18c35d65f7a42baab52ac6752ef2dc61ca43b3c5161dc549e77; SG-23B 8fbfbc9ae4815b07597b4232e4ace9fd4d78c95d27609c9b10a92994a16be052. Raw timing runs supplied separately, not committed.

Next: derive the first inactive-constraint crossing for a verified velocity and schedule. Each competing mask has affine slack a+s*b, so its earliest positive crossing when b<0 is a/(-b). Use last-use and adjusted-utility sign breakpoints to avoid exponential mask enumeration, check rational endpoint certificates, and compare event-driven continuation with the SG-22 baseline on fresh mixed-price trajectories. Do not infer finite-step feasibility from tangent membership alone.

## Implementation custody

[DataGoblin campaign, velocity solver, incremental sweep, tests and replay](https://github.com/MoralityLabAI/DataGoblin/tree/0dfefdb5573279b14b7ad4026b09a472339fe637/training/rlt_gym/sg23_spectral_velocity). Complete archive `sg23_cpu_campaign.zip`, SHA256 `6c412dff14b8d291e3b44cddf95bf9ae2592d55cbbfde7831c67aea639f22851`. Contains 43 files including both raw result sets, executed source and dependencies, six batch logs, tests, independent audit and SHA256 manifest. Archive integrity and every contained file hash checked. All six committed Python source hashes match executed files. Generated raw receipts and timings supplied separately.
