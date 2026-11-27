# SG-08: learned retention of recomputable dual certificates

[DataGoblin implementation](https://github.com/MoralityLabAI/DataGoblin/tree/9036b0439d75c16896fac649b8679566291e242c/training/rlt_gym/sg08_spectral_retention) · [Optional retention session](https://github.com/MoralityLabAI/DataGoblin/blob/9036b0439d75c16896fac649b8679566291e242c/src/datagoblin/training_gym/spectral_retention.py).

Arithmetic cache retention cannot affect the exact value of a dual table. Use this separation to compare policies on the same request stream and require full proof-receipt parity. Policies have mandatory insertion and bounded resident metadata, with task-local immutable fixed-table keys; no historical frequency map. Learned rank estimates depth-weighted future references within 2C requests at capacity C. Eleven coefficients fitted by ridge regression then quantized to integer weights. All future labels come from 36 separate training roots (35,442 labels); shuffled control permutes labels within roots. Feature values depend only on observed resident history and current request. Fixed-first policy evicts nonfixed multiplier tables before fixed tables, LRU within class. Cost-weighted policy uses depth times resident frequency divided by age. Belady is a future-aware offline minimum-miss reference under mandatory insertion, not a weighted-work optimum.

48 fresh exposed evaluation roots, 24 d8/T6, 12 d9/T7, 12 d10/T8, three planted utility regimes. Scalar cache capacity 16; search caps and lazy proposal policy fixed to SG-07. 66,775 table requests. Offline replay at C=6,12,24; live timing at C=12, three repetitions with randomized order and original native LRU baseline. All 1008 complete event transcripts replayed through original SG-05 verifier, with receipt and simulator work parity; 48/48 certified optimal. Same full request sequences checked directly on 288 additional small fixture runs. 2916 tiny trace-cap combinations checked Belady against exhaustive offline cache DP. Three targeted pytest tests passed. Full repository tests and Ruff unavailable in this isolated checkout.

| Policy | C=6 misses | C=6 work | C=12 work | C=24 work | C=12 CPU ms |
|---|---:|---:|---:|---:|---:|
| LRU |3186|141733|74891|64096|4.706595|
| LFU |47651|1738911|1622917|441645|13.043341|
| Weighted |33288|1162029|473699|127582|7.533668|
| Fixed-first |2283|96457|77157|64843|4.747936|
| Learned |2648|113439|72997|65869|5.187982|
| Shuffled |45964|1680896|1523593|968979|26.193181|
| Belady |2134|88947|67621|62530|offline only|

Original native LRU C=12 mean root-median CPU 3.937916 ms. Learned work savings 19.96% at 6 slots, 2.53% at 12 slots; 2.77% more work at 24 slots. Fixed-first 31.94% savings at 6 slots. Frequency policies retain stale phase entries on this workload. All new generic policies slower in mean CPU than native LRU. Post-hoc C=6 timing ablation on same exposed roots, 576 receipt comparisons: native LRU 3.910774 ms; generic LRU 4.638782; fixed-first 4.555917; learned 5.024245. Fixed-first beats generic LRU at C=6 but not native implementation. No native-model conclusion or cross-host speed guarantee. Cache slots are not equal-byte budgets. Optional request recording uses unbounded trace memory and is disabled during live timing. Recommendation: preserve original LRU; test a specialized low-overhead fixed-vector bank or depth-phase replacement on fresh roots before promotion.

Evidence custody: all four executed new source blobs byte-identical to DataGoblin commit 9036b0439d75c16896fac649b8679566291e242c (core 09a47e6b508334991a46dda5b004dff8a0b5161e, campaign 2bd2281ad8bde7f33d2cdc64fc9d478dded1c807, tests e4f945a03dcc1c62488944f94b2eae357bdc1e2c, cap6 5fea26dd7673b015e2c03b1fc5f62c8d5e4aa41d). Raw JSON SHA256 99bca7bff03e58809155d5b9c1206c985618887a1cda188d216f9baa95638b7b and cap6 JSON 271cc6ba354111e346a432e39edbede0a2c7c49726f1083645575d7c324a601c. Complete archive sg08_cpu_campaign.zip SHA256 bc769e74c64d2da42986e71cc3254fa6ae7ab543963cec82014640f06293d694 provided with session; generated raw traces not committed. No GPU or paid inference.
