# SG-10: quotient and vectorized certified expansion

[DataGoblin implementation, executable and tests](https://github.com/MoralityLabAI/DataGoblin/tree/f5ca475b8fa493a10a70a0d51fab4a632a3ee8da/training/rlt_gym/sg10_spectral_batch) · [Optional verifier](https://github.com/MoralityLabAI/DataGoblin/blob/f5ca475b8fa493a10a70a0d51fab4a632a3ee8da/src/datagoblin/training_gym/spectral_batch.py).

For fixed suffix S, actions A with the same union S' have identical newly paid last-use price and remaining optimization problem. Within each group retain the highest local utility, with earliest original action on ties. This preserves complete-action search semantics and dominance state. Original transition accounting still charges every legal action. Skip already dominated groups before evaluating the remaining bound. Terminal expansion keeps earliest maximizing action. Vectorize union formation and batches of missing fixed-bank bounds with NumPy; group reduction, state updates, arbitrary proposed multipliers and certificate arithmetic outside the bank retain Python execution. Accepted-proposal overlay takes precedence. The snapshot batch cache may have a different eviction order, but recomputed values remain exact. Hybrid thresholds: 64 actions, 16 missing bounds, chosen before execution. Scope: existing validated task domain d,T<=10. Original scalar default unchanged.

## Fresh synthetic development campaign

60 new roots, 20 each persistent/rotating/mixed; 30 d8/T6, 15 d9/T7, 15 d10/T8. Same SG-07 lazy policy, split fixed bank, 12 dynamic tables, 16 scalar slots, initial incumbent and 2048 transition/24 proposal/80 decision caps. Three randomized timing repetitions per root; process CPU mean of root medians includes setup and receipt, excludes external oracle, independent replay and separate traced-memory runs. All 720 timed full receipts matched the scalar bank and independently replayed against SG-05. Exact oracle checked every interval. 60/60 certified, zero mean gap and regret. 47 required expansions. Five targeted pytest tests passed: 13,479 quotient comparisons against raw action enumeration; 46,080 repeated scalar/batch bounds with overlay lookup fixtures; 36 interleaved searches comparing full live and dominance state; ties, invalid state and cap atomicity; 4,096 extreme valid-range bounds. Artificial overlay fixture checks lookup precedence only, not admissibility. Full repository tests and Ruff unavailable in the isolated checkout.

| Implementation | Mean CPU ms | Wins vs scalar /60 | Expanding roots CPU ms | Mean traced peak bytes (9 roots) | Fixed bank queries |
|---|---:|---:|---:|---:|---:|
| scalar |2.784249083|0|3.243643426|57742.67|88965|
| dedup |2.585151367|49|2.993566809|61636.44|70456|
| batch |2.506502433|51|2.891035787|82920|69952|
| hybrid |2.509827067|48|2.893985511|82294.22|69952|

First repetition counts: every arm charges 25,716 actions (including 2,629 terminal). Quotient arms collapse 23,087 nonterminal actions to 11,539 union groups, skip 37 dominated groups and request 11,502 remaining bounds (including cached hits/overlay). Batch makes 277 vectorized batches covering 11,385 missing masks; hybrid 237/10,937. Fixed-bank query count means vector-mask evaluations, not total operations. All arms use 12,615 dynamic component steps plus 21,510 bank steps. Batch reduces mean CPU 9.98% vs scalar, 3.04% vs dedup and fixed-bank queries 21.37%. Tracemalloc measured nine predetermined roots separately; byte budgets and RSS are not matched. More allocation in NumPy arms. Single host, offline synthetic development, no native model/hardware traffic claim. Rust unimplemented.

## Evidence custody

Executed dependency blob hashes match prior pinned commits. New exact source hashes in DataGoblin commit f5ca475b8fa493a10a70a0d51fab4a632a3ee8da: spectral_batch cbbe0d1d68860864b5bc49be04591764db14737d; campaign ff6a3f414589af1555b2743ecb3e7be5f647e10a; tests 0259fb5673daf9626a933a7a3c0c3461f71dff5c. Raw results JSON SHA256 f914b9e118c89c9c5923c85f3e23ca5cd33c97e899c40f32fd90d714736b6b4c. Full archive sg10_cpu_campaign.zip SHA256 21755e8f2b18b2fe692453ff695744bf1f3065826a2c139f51635b7f139f3bde provided with the session. No generated timing runs committed. CPU only, no paid inference.

Next: generate union classes directly by choosing R=A minus S from the complement and filling the remaining k-|R| slots with highest-utility members of S, with exact original tie order. The number of classes is sum over feasible r of binom(d-|S|,r), potentially much smaller than binom(d,k). Retain the original transition charge and exact replay for comparisons, then investigate separately registered quotient-work accounting. Use fresh roots.
