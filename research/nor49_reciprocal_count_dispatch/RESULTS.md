# NOR-49: first timing-clean 48-source CPU benchmark

**Original engineering verdict: FAIL, 3/5 frozen gates passed.** The first prospectively registered CPU run [#44256305944](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44256305944) compared NOR45, NOR47, NOR48, and the frozen count-R>=2 NOR49 selector across 48 previously untimed historical source identities, with five interleaved repetitions and exact output/Merkle-root parity.

## Immutable evidence
- First original CPU JSON: [results/nor49_first_original_cpu.json](results/nor49_first_original_cpu.json), **84,469 bytes**, SHA-256 `c3f4a67b20df4504461c65168f6006041f2cc839aabdda054032a6f89bd24e4b`.
- First public-only source roster: [results/nor49_first_original_source_roster.json](results/nor49_first_original_source_roster.json), **13,081 bytes**, SHA-256 `40929d52abddc5a15e82ec8e8101fba265d39e28239172c7e9fcaca4f6f0d2ae`.
- Original [frozen receipt](results/nor49_first_receipt.json) and permanent [read-only SHA/gate auditor](original_archive_audit.py).
- First timing preregistration commit: [`26b70716`](https://github.com/MoralityLabAI/Compactification-Theory/commit/26b70716ec3d3830031919e593d3092cc31ae162), preceding all heldout timings. All 16 original NOR48 training IDs excluded.
- Original fixed median sourcewise NOR49/direct **0.9995949312×** versus target <0.75×: **FAIL**.
- Original fixed median sourcewise NOR49/NOR48 **0.9999505379×** versus target <0.95×: **FAIL**.
- Signed-program routing, full exact output/Merkle roots and source-first reentrant reproducibility gates: **PASS**.

## Original source group evidence

| Active signed R count | Sources | Median NOR49/direct |
| --- | ---: | ---: |
| 0 | 19 | 1.001912× |
| 1 | 15 | 1.000230× |
| 2 | 12 | 0.471252× |
| 3 | 2 | 0.398938× |

Each source's exact program operations, root, five arm samples and median are included in the unmodified original CPU JSON. Full session, authority and provider signing checks are exercised in separate tests; equally excluded common cryptographic/network costs are outside these arithmetic+root timings.

## Post-hoc workload-sum diagnostics (exploratory)

Source-median times summed over the 48 cases give NOR45 **2345.436 ms**, NOR48 **1883.189 ms**, and NOR49 **1712.580 ms**. NOR49/NOR45 is **0.730175×** and NOR49/NOR48 **0.909404×** on this workload-sum metric. A seeded 10,000-resample twelve-block bootstrap gave descriptive 95% intervals **[0.603, 0.874]** and **[0.832, 0.987]**, respectively. These are post-hoc analyses of the first heldout run. They preserve, and never replace, the **original 3/5 FAIL**.

## Next registered mechanism experiment

NOR-50 should compare NOR45 direct with a memoized-exact NOR45 executor that computes each distinct unary transform once, keeping all signed integers and independently computed Merkle roots identical. The NOR46/47 trace path already deduplicates transforms; NOR45 direct recomputes R once per occurrence. Register a completely new CPU timing panel and compare the original/direct-cached/GMP/dispatch arms before timing, with separate R=0, R=1 and R>=2 strata. This separates redundant inverse computation from the GMP acceleration claim. Program authenticity, trusted provider and full trace/transport assumptions remain outside the pure arithmetic comparator.
