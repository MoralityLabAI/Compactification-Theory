# SG-07: cached exact certificate arithmetic

[DataGoblin implementation and replay](https://github.com/MoralityLabAI/DataGoblin/tree/97455fdf843adb0cf083a2301789ce78677a3507/training/rlt_gym/sg07_spectral_cache) · [Verifier source](https://github.com/MoralityLabAI/DataGoblin/blob/97455fdf843adb0cf083a2301789ce78677a3507/src/datagoblin/training_gym/spectral_cache.py).

For fixed task, remaining depth t and multipliers eta, B(t,S;eta)=B(t,empty;eta)+sum_{j in S}(a_j-b_j), where a_j is the relaxed score with component already active and b_j the relaxed score if inactive. A table requires O(td) arithmetic, then each mask costs O(|S|) additions. A single-bit mask change changes exactly one contribution. The implemented evaluator sums the selected contributions; it does not yet maintain incremental Gray-code traversal. Two LRU caches retain tables and original fixed-multiplier scalar bounds; a separate overlay retains accepted proposals. Evicting arithmetic does not alter proof obligations. No frontier or dominance state is evicted. The original independent verifier checks the identical event format.

## Primary development campaign

60 fresh offline synthetic task roots, 20 per regime (persistent, rotating, mixed). 36 d8/T6, 12 d9/T7, 12 d10/T8. Three randomized timing repetitions per root. Five policies, identical initial schedule and work caps of 2048 transitions, 24 proposals, 80 decisions. Lazy reuse considers up to eight recent multiplier vectors at the same depth and uses one if the verified gain is at least half the current node gap rounded up, minimum one. Each accepted proposal follows the SG-05 event contract. Single-host process CPU includes initialization, exact arithmetic, search and receipt, excluding the external optimum, post-hoc transcript replay and separate memory measurements. All 900 receipts independently replayed, all solutions bounded by exact oracle, cache-only arms reproduce original full receipt on every root and repetition. Three targeted pytest tests passed: 7680 candidate bound comparisons, 7680 fixed bound comparisons, 32 interleaved search fixtures and invalid-input checks. Full repo tests and Ruff pending.

| Policy | Mean CPU ms (mean root median) | Certified /60 | Gap | Mean transitions | Coordinates | Mean traced peak bytes (9 roots) |
|---|---:|---:|---:|---:|---:|---:|
| original | 16.3258451 |59|0.25|444.6333|38.75|51141.33|
| cache128 |3.94396993|59|0.25|444.6333|38.75|58776|
| cache12 |4.2170357|59|0.25|444.6333|38.75|45994.67|
| lazy128 |3.6989779|59|0.25|444.5|26.5|59175.11|
| lazy12 |4.00981007|59|0.25|444.5|26.5|47225.78|

Every arm mean regret 0.20. Lazy arms reused 137 vectors across 402 candidate evaluations per campaign. Smaller cache-only evicted 1829 tables and 12323 scalars; peak entries 12 tables, 16 scalars; larger used up to 67 tables and 128 scalars. Proposal overlay maximum 24. Cache caps are entry counts, not total RSS bounds; memory measured via tracemalloc on nine preselected roots in untimed replay runs. Original versus lazy128 ratio of mean root medians is 4.41x; lazy128 wins 60/60 paired CPU medians, cache12 59/60. One host, offline development only.

## Separate post-hoc arithmetic/cache ablation

Same exposed 60 roots, another three randomized timing repetitions, 900 exact full-receipt comparisons, zero disagreements. Original 15.5607 ms; linear no-cache 17.3803; table-only 4.8671; scalar-only 10.1594; both caches 3.6621. Evidence supports table reuse as main source of measured savings on this workload. The separate ablation is not an independent confirmation cohort.

## Evidence custody

DataGoblin commit 97455fdf843adb0cf083a2301789ce78677a3507 preserves the byte-identical executed sources. Source blobs cache 409d7a6045193e57b3b4e89a9db342594ff6682c; campaign ee613be2880860dabb2b7d1bdd8ce94dac27cea1; tests 4237b4c10af9c126c3600accf8c240949271625a; ablation e67a1c2f6c76bcdafad891c837a48f2b7c67e24a. Original SG-03/04/05 source blobs verified against the SG-06 archive. Main raw results SHA256 550c1704a59a19caeae92b1659cfe4b7328d81849105dabe08ba7907a8d52644. Ablation raw results SHA256 dcccbcc6a0ddf2ce2f62a25c6645ea174f027d66d10bbba2b1574c7145e6f1d1. Full raw archive supplied with this session as sg07_cpu_campaign.zip, SHA256 b3211dba9ab5f2958cb9238d742470b0b528bb9926de5b56647ed3fcd27f9c40. Raw generated timing runs not committed. No GPU, API spending, native activation work, or general speed guarantee. Next: exact cost-aware selection of cache entries, incremental mask traversal, and a fresh root cohort at higher component count; retain independent replay.
