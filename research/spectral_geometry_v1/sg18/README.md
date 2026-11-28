# SG-18: bounded multiplier memory for certificate acquisition

Synthetic additive source-bound routing; the SG-15 exact rational verifier and SG-17 chain-LP proposal generator are unchanged. Each task gets the same root multiplier and dual-rounded feasible incumbent. When the root certificate has nonzero gap, a policy may try at most two compatible resident multiplier vectors, then fall back to chain LP, direct LP and capped SG-12 block search. Every transfer is evaluated on the new task. The lower endpoint is the best independently scored legal schedule and the upper endpoint is the smallest checked rational bound, floored to an integer. A transfer-stage success means this combined interval closes before an LP call, not necessarily that the transferred vector attains the new optimum. A certificate from a previous task is never sufficient by itself.

Memory retains at most four immutable records, each containing the full task signature, rational multiplier vector and timestamp. Only results with zero independently checked dual-floor gap are inserted, with full-signature deduplication and recency eviction. Compatibility requires identical dimensions, budgets, coordinates, bit prices and lambda. Objective utilities may change. Memory capacity counts records, not bytes or total process RSS; no hard time cap. No cached schedules, basis warm starts, model inference or external retrieval services. The implementation is an optional research gym for trusted callers.

Policies: cold; full-signature identity-only (one probe); recency; nearest utility matrix (L1 distance); exact dual-bound ranking over resident vectors; learned ridge ranker with seven features; within-root shuffled-label control. For training, independently try each compatible resident candidate on a cold initialized task, label success by exact certificate closure, and advance the training cache with a cold acquisition. Fit ridge with penalty 2 on 620 candidate labels (104 positive) from 18 separate trajectory roots, six per regime. No evaluation feedback. The learned selector ranks candidates; it does not learn abstention. The root-shuffled control uses the same feature rows and architecture.

## Experiment and reproduction

24 evaluation trajectory roots (8 per persistent/rotating/mixed regime), three independent anchor utility matrices per trajectory, d=32,T=12,k=7/8. Shared structural task data within a trajectory. Four phases, three queries each in randomized anchor order: base, exact repeat, eight distinct utility entries perturbed by +/-1, and 48 distinct entries perturbed by +/-1..3. Both perturbed phases derive from the anchor, not from the preceding phase. Policy memory resets at each trajectory and repetition. Two randomized whole-trajectory arm orders per root. 288 logical queries per policy, 576 measured runs per policy, 4032 total. Independent statistical units are 24 trajectory roots, not individual queries or repetitions. All final intervals agree at the exact optimum. Every candidate bound and final receipt was checked again in a separate post-run audit (4032 final, 6712 candidate receipts); no LP failures or block fallback occurred. Three targeted pytest tests passed (336 small exact-oracle policy runs, immutable snapshots, compatibility, capacity and rejection of uncertified vectors). Full repository tests and Ruff not run; Lean and Rust not compiled.

Install DataGoblin and optional NumPy/SciPy/pytest. From the repo root:

    export PYTHONPATH=training/rlt_gym/sg07_spectral_cache:training/rlt_gym/sg11_spectral_quotient:training/rlt_gym/sg15_spectral_face:training/rlt_gym/sg17_spectral_chain:training/rlt_gym/sg18_spectral_transfer
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q training/rlt_gym/sg18_spectral_transfer/test_sg18.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python training/rlt_gym/sg18_spectral_transfer/sg18.py

CPU is process time, mean query time per trajectory averaged across roots, including initialization, retrieval, all attempted repairs and exact checks, LP/fallback, and memory insertion checks. Excludes independent audit after the timed interval and offline fitting. Single host, randomized repetitions, descriptive development experiment. The numerical solver remains a proposal tool; a successful status alone is never accepted as a proof. No paid inference or GPU.

| Arm | CPU ms | LP calls | Transfer closures | Probes | Certified /576 | Paired CPU wins /24 |
|---|---:|---:|---:|---:|---:|---:|
| cold |10.888219|550|0|0|576|0|
| identity |10.068876|412|138|138|576|20|
| recency |12.442940|450|100|948|576|2|
| nearest |9.266437|298|252|708|576|21|
| learned |9.837058|298|252|708|576|20|
| shuffled |10.795588|380|170|816|576|13|
| bound_rank |9.743133|298|252|708|576|21|

All arms have 26 root-certified runs. Nearest reduces LP calls 45.82% and CPU 14.90% relative to cold; learned ties its LP avoidance but costs more CPU. Nearest closes 138/138 eligible repeat queries, 94/136 mild queries, 20/138 large queries, and 0/138 base queries. Among changed-objective queries it avoids 114 LP calls. Phase CPU means (cold vs nearest, ms): base 10.854 vs 12.713, repeat 10.758 vs 4.278, mild 10.564 vs 7.393, large 11.376 vs 12.681. Counts include both repetitions. This workload explicitly contains repeat queries and holds structure fixed. No generalization claim to changes in prices, budgets, dimensions, or native model workloads. Recency and unrestricted probing can waste more time than cold acquisition. Recommend nearest as a candidate deterministic retrieval baseline, keep existing defaults unchanged. Next experiment: cheap perturbation- and gap-sensitive abstention, compare fixed thresholds and learned expected net CPU savings, with separate fresh roots and stronger no-repeat cohorts.

## Evidence custody

Executed source blobs sg18.py 8a4797f53176ca284a0c09afc0e77ec80684f310, test_sg18.py 2397e35b750c2cb613acd76d032f64784ec1a4ee. Raw results JSON SHA256 f1c05b5118c71d03d51317980412f7b779b3e23f4e102b08fd20ef3e6c1f4a87. Raw timing data and fitted coefficients are supplied separately, not committed.

## Implementation custody

[DataGoblin campaign and tests](https://github.com/MoralityLabAI/DataGoblin/tree/adfc1c231bdb2a582809912c9174bb4f4b099f21/training/rlt_gym/sg18_spectral_transfer). Complete archive `sg18_cpu_campaign.zip`, SHA256 `83313438c6ac4d418b04d617cc0505d04257fef53ffb02cc6f0718a27f1f9601`. Includes executed sources, dependency copies, full timing and certificate receipts, coefficients, test logs, post-run audit and SHA256 manifest. Archive integrity check passed. Both committed Python source blobs match the executed files. Raw timing data are supplied separately.
