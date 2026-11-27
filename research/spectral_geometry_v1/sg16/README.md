# SG-16: adaptive certificate acquisition portfolio

The SG-15 rational dual-face verifier and the SG-12 block replay verifier establish bounds for the same immutable finite task. A portfolio stores a feasible incumbent and the smallest verified integer upper bound seen so far. Intersecting the intervals preserves validity, including for negative objectives (use mathematical floor for rational bounds). A block run may stop as soon as its incumbent reaches an earlier certified upper bound. The independent replay recomputes every rational bound, checks each block transcript, and reconstructs the full portfolio receipt. Face recovery status and timing fields are diagnostics; neither is trusted as a proof. The code is for trusted in-process callers, not a sandbox for executing arbitrary student code.

## Reproduction

Install DataGoblin and optional SciPy/NumPy. From the repo root add the SG-07, SG-11, SG-15 and SG-16 gym directories to PYTHONPATH:

    export PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient:training/rlt_gym/sg15_spectral_face:training/rlt_gym/sg16_spectral_acquisition
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q training/rlt_gym/sg16_spectral_acquisition/test_spectral_acquisition.py training/rlt_gym/sg15_spectral_face/test_spectral_face.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg16_spectral_acquisition/sg16.py

The script writes full results and per-root progress in the working directory. The downloadable archive contains the executed sources and all dependencies needed for an isolated replay. Seven targeted tests passed (three new and four existing SG-15 tests). The new tests cover 96 complete portfolio runs against the small exact oracle, 192 arbitrary rational proposals across 32 fixtures, monotone interval intersection, negative fractional flooring, tampering, invalid proposal rejection and stopping before a block event. Full repository tests, Ruff, Lean and Rust were not run.

## Experiment

24 training roots, 36 fresh exposed evaluation roots, evenly split across persistent/rotating/mixed, d=32,T=12,k=7/8. Each arm computes the same root vector and initial dual-rounded incumbent, and checks the exact root bound; certified tasks stop immediately. Four fixed escalation plans: LP-first; one further coordinate sweep and bounded face recovery then LP; up to four alternating sweeps with face recovery after each then LP; up to 256 block events then LP. Any remaining gap permits block fallback with at most 2048 total block events (block restarts count against that cap). A deterministic gap rule chooses one sweep when root gap<=8, otherwise LP. A learned selector fits four ridge regressors of 12 coefficients each on log(1+CPU_ms+100*gap) using one training timing per root and plan. Matched shuffled control permutes complete target rows. The learned policy picks a plan once from initial features, not a learned per-event stopping policy. The training-best fixed plan is LP-first. No evaluation feedback or tuning. Face node cap 512 per attempt, LP solver's own time limit 2 seconds. Numerical LP suggestions are independently verified; no LP status is a proof.

Two randomized repetitions per root/arm for fixed work and soft 30 ms process CPU deadline. The deadline is checked before stages and between block events; a single LP or sweep stage may overrun it. CPU includes setup, selection, repair, LP/face certificate checking, block search and final receipt; excludes independent portfolio/block replay and offline fitting. Every 1008 evaluation and 96 training receipt replay passed. All seven arms certified all 36 evaluation tasks on both repetitions in each budget mode. All cross-policy intervals intersect exactly. No exhaustive oracle at d=32. All LP calls returned status zero, and none of the LP-first, sweep-first or learned arms needed a block fallback after LP. One host, Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, CPU only. No native model or paid inference.

### Fixed work

| Policy | CPU ms (mean root median) | LP calls /72 | Certified /72 | Paired time wins vs LP /36 |
|---|---:|---:|---:|---:|
| LP-first |14.290918|68|72|0|
| One sweep |16.094955|62|72|4|
| Four sweeps |21.073284|48|72|10|
| Block-first |19.542210|56|72|7|
| Gap rule |15.023737|62|72|10|
| Learned |15.192669|52|72|16|
| Shuffled |17.510547|62|72|11|

### Soft 30 ms deadline

| Policy | CPU ms | Certified /72 | Certified within 30 ms /72 | Max overshoot ms |
|---|---:|---:|---:|---:|
| LP-first |14.890847|72|71|33.116727|
| One sweep |15.826637|72|72|0|
| Four sweeps |20.828502|72|70|2.075053|
| Block-first |20.165192|72|70|23.363608|
| Gap rule |15.311245|72|71|10.847631|
| Learned |15.278141|72|69|3.557558|
| Shuffled |17.091541|72|72|0|

Learned selects LP on 20 roots and four sweeps on 16. It saves 16/68 LP calls (23.53%) but takes 6.31% more mean CPU than LP-first at fixed work. No learned-policy promotion. One sweep and shuffled meet the deadline on all 72 timed runs, a descriptive observation on 36 independent roots rather than a deadline guarantee. The 33.12 ms LP-first overshoot illustrates why this is a soft cap. The previous 30 ms LP average is not a matched baseline; the new LP-first result is 14.29 ms. Preserve LP-first as the deterministic benchmark. Next: derive a smaller exact-proposal LP using only chain-prefix inequalities or warm-startable sparse dual optimization, then test end-to-end time at equal certificate quality on fresh roots. The code keeps the existing default unchanged.

Source blob hashes: core ecf03cdabe0bed584ece0a84b54d2f4a7a378a8a; campaign 3f43665928b37d395b8e51ee8b29951e448a820b; tests 77e7c623639a7c1c4b2e3f50f2ccc4afa0813fd4. Raw results SHA256 cc9651ce05df9836170ee55ff077f13fa917677e7a2d81f434d23789c5e83d94. Generated timing runs are supplied separately, not committed.

## Implementation custody

[DataGoblin campaign and tests](https://github.com/MoralityLabAI/DataGoblin/tree/32cdbbcf605a8d659bf9a42d7e1ba3f375cf45b4/training/rlt_gym/sg16_spectral_acquisition) · [Portfolio verifier](https://github.com/MoralityLabAI/DataGoblin/blob/32cdbbcf605a8d659bf9a42d7e1ba3f375cf45b4/src/datagoblin/training_gym/spectral_acquisition.py).

Complete archive `sg16_cpu_campaign.zip`, SHA256 `a95e3a696a38564a34af5593fd151cbb8edbd490ba78735ec846b5165f0f4df5`. Includes executed source, dependency sources, full receipts, timing log, test log and SHA256 manifest. All four committed DataGoblin files match the local Git blob hashes; archive integrity check passed. Generated timing data are supplied separately.
