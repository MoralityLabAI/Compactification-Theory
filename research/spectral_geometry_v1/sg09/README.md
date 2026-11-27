# SG-09: prefix-shared spectral table compactification

[DataGoblin executable and tests](https://github.com/MoralityLabAI/DataGoblin/tree/2127efe4820606991409af068cff5681746bc0fa/training/rlt_gym/sg09_spectral_bank) · [Optional bank verifier](https://github.com/MoralityLabAI/DataGoblin/blob/2127efe4820606991409af068cff5681746bc0fa/src/datagoblin/training_gym/spectral_bank.py).

For each fixed multiplier vector, update the relaxed active and inactive component contributions by one exact prefix recurrence per visit. Append the base plus per-component differences for each unique prefix to a contiguous signed integer array. O(MTd) component arithmetic, M=6, instead of O(MT²d) to compute every prefix independently. Query uses O(popcount(mask)) additions. Split-mask variant precomputes all sums for each half, O(MT(2^floor(d/2)+2^ceil(d/2))) storage and preprocessing, then two integer lookups and one addition per fixed vector. Arbitrary proposed multipliers retain Python integer arithmetic and bounded SG-07 caches. Original proof state, overlay, and transcript are unchanged. Scope: internal fixed vectors from validated SG-03 tasks only.

## Fresh synthetic development campaign

60 new task roots, 20 each persistent, rotating, mixed; 30 d8/T6, 15 d9/T7, 15 d10/T8. Same SG-07 lazy multiplier policy, initial incumbent, work caps. Bank arms add fixed bank to 12 dynamic table slots and 16 scalar slots. LRU128 has 128 slots each. Three randomized timing repetitions per root, process CPU mean of root medians. Time includes initialization and final receipt; excludes independent exact oracle, transcript replay, separate traced memory and batch microbenchmarks.

All 720 full receipts identical across implementations and independently replayed through original SG-05 verifier. Exact oracle checked incumbent <= optimum <= upper. 58/60 certified optimal, mean gap 0.5, mean regret 0.216667. Three targeted pytest tests passed, including 38,400 exact fixed-vector comparisons, 24 interleaved search fixtures, subset-sum and integer-range controls. No full repository pytest/Ruff run.

| Implementation | CPU ms | Dynamic component steps | Bank component steps | Mean traced peak bytes (9 roots) |
|---|---:|---:|---:|---:|
| LRU12 |3.851877833|99667|0|32576.44|
| LRU128 |3.547936917|79807|0|45557.78|
| Prefix |3.177035650|13949|21510|41535.11|
| Split |2.898993367|13949|21510|56267.56|

Prefix wins 50/60 paired medians, split 48/60, versus LRU12. Split saves 24.74% mean CPU, prefix 17.52%. Combined counted bank and dynamic component steps 35,459 vs 99,667, 64.42% fewer; proxy excludes subset-sum construction, query additions and overhead. Mean payload 3,258.93 bytes prefix and 15,093.33 bytes split, maxima 4,312 and 25,088. Traced Python allocations from nine separate runs, not process RSS. Capacity is not matched by bytes.

## Separate batch microbenchmark

Nine full-depth batches of all 256/512/1024 masks, five repetitions; exact parity between NumPy int64 and scalar results. Mean root median 0.685204 ms scalar versus 0.019088667 ms NumPy (~35.9x). Preconstructed bank, masks, view, offsets; scalar path includes conversion to list and input checks, different output containers. NumPy is not used in timed search. This is a kernel microbenchmark, not end-to-end acceleration. Rust compiler unavailable in runtime; Rust port not implemented.

Evidence custody: exact executed source hashes verified in DataGoblin commit 2127efe4820606991409af068cff5681746bc0fa (bank e82feaffe3fcbc88ddf344509a4b84fce81a19a3; campaign e566336572d34f606354a38a55f7f6b840d5901b; tests 2b7a69f9177aeff9d09a300500ea5f373a9e5de2). Original dependency source blobs checked against prior commits. Raw results SHA256 102ed2e9fdc6e058e3280fb66531fc9d28f57915c1208773b160629dba6eff31. Archive sg09_cpu_campaign.zip SHA256 586f6dddbe20c480080e7bfc260bb30feb86ab9295b7b59d4231ad7a90c7eef4; raw timing runs supplied separately. CPU only, single host, synthetic development, no native model or hardware traffic claim. Original LRU remains default; optional compact prefix bank and split bank for different memory budgets.

Next: batch child-bound evaluation at complete search expansion, retain exact proposal overlay and independent replay, compare against scalar bank at equal byte budgets, then port the integer kernel to Rust and benchmark in a compiled environment. Avoid using exposed SG-09 roots as independent confirmation.
