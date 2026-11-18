# NOR-52: first source-locked exact support compression and GMP phase audit

**First original engineering verdict: PASS, all 5/5 prospectively registered gates met.** The [first hosted run #44275345207](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44275345207) completed 60 new SHA-disjoint source identities from the fixed 2×3×5×2 factorial panel. Its four exact integer arms produced 60/60 equal complete inverse vectors and independently recomputed full NOR45-format Merkle roots. Before any first timing, the separately dependent source-only job uploaded all 60 source IDs.

## Immutable first-run evidence

- The original CPU JSON was **166,277 bytes**, SHA-256 **83090c4d9647b543b8950025808f087676ffd8a6dee03709b729bd663304a99d**.
- The first public-only fixture was **1,474,292 bytes**, SHA-256 **d18b795053d81c16fc5316b490316e90fe6b373156a8ee8b295008a13df150f4**.
- Both original byte streams are retained inside [nor52_first_original_hosted.zip](results/nor52_first_original_hosted.zip). Its ZIP hash, original gates, source medians and phase ledger are in the [first receipt](results/nor52_first_receipt.json) and permanently checked by [original_archive_audit.py](original_archive_audit.py).
- First source-only artifact **14414901378** and first full CPU artifact **14413918829**. The [source-first run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44275345207) used the original registration commit **9526627e21b429e7fea1928b3ddfb0e1dadc5e74**, which predates all new sources/timings.
- Five source-seeded, same-run interleaved CPU repetitions per source and per arm on Ubuntu-24.04, CPython 3.12, gmpy2 2.3.2, GNU GMP 6.3.0. Shared signing/session/transport costs are excluded equally.

## All five original registered engineering gates

| Registered gate | First result | Verdict |
| --- | ---: | --- |
| Sparse m3073 (16 sources), compressed native/full native sourcewise median <0.50× | **0.163421×** | PASS |
| Same sparse m3073 cases, compressed GMP/full GMP sourcewise median <0.50× | **0.078663×** | PASS |
| Dense stride1 (12 sources), both compressed/full ratios <1.10× | **1.001166× native, 1.003118× GMP** | PASS |
| Full signed integer inverse/root parity and hostile source/witness controls | **60/60 and predata tests** | PASS |
| First source SHA, original caps, reentrancy, five repeats and independent phase-replay parity | **PASS** | PASS |

For sparse m3073, exact inverse support compression reduced the per-source median native cost by about **83.7%** and full GMP witness verification by about **92.1%**. The registered dense-control overheads remained close to one. At stride 4, m3073 compresses to 769 coefficients; stride 8 compresses to 385. Both paths nevertheless return all 3073 exact output coefficients, preserving original zeros, and rebuild the same complete Merkle root.

The independent full-length witness is validated in its **original full** coefficient count and original resource bounds, with strict built-in integer types and exact zero support in source and untrusted witness. The compressed-GMP path verifies full exact signed inverse identity on the reduced series and returns the original unchanged full witness. The algorithm does not use untrusted hints to bypass a source/support check.

## Whole-benchmark secondary matched-workload totals

| Arm | Sum of source-median wall times | Ratio to full native |
| --- | ---: | ---: |
| Full native single-R + full root | 8,659.482 ms | 1.000000× |
| **Compressed native + full root** | **4,171.394 ms** | **0.481714×** |
| Full GMP verification + full root | 35,096.269 ms | 4.052929× |
| Compressed GMP verification + full root | 8,004.903 ms | 0.924409× |

The 10-block bootstrap of the source-median workload-total ratio gave descriptive 95% intervals **[0.173, 0.735]** for compressed native/full native, **[0.353, 1.363]** for compressed GMP/full native. They are secondary descriptive statistics; the original five engineering gates remain the registered criterion.

## Untimed full NOR47 GMP phase breakdown

Separately replaying **one unchanged mathematical full convolution per source**, the independent phase ledger reproduced all exact signed products. Across all 60 first sources, packing the original full signed integers accounted for about **49.0%** of measured phase time; extracting and shifting low digits accounted for about **40.0%**; GMP multiplication itself accounted for about **8.3%**; signed cross-term correction used about **2.7%**. The parameter checks were negligible on this panel. These phase durations total about **35.50 seconds** over 60 diagnostic convolutions and are **outside** the four registered timed arms. They isolate measured costs in the implemented phase replay, not universal CPU or asymptotic arithmetic behavior.

The highest-precision stride4-b16 group was much slower in the original full GMP arm, with a 6.8897× median full GMP/full native ratio and a 2.7356× median compressed GMP/compressed native ratio. Low-precision sparse groups were closer in speed. Exact support compression alone does not prove that GMP is the preferred backend after native support compression. Sourcewise direct/compressed backend ratios and resource sizes are preserved in the original JSON.

## Follow-up and scope

The next engineering question is whether more efficient GMP input packing and extraction can materially improve its exact signed verifier. The largest phase costs suggest investigating a native limb pack/unpack interface or replacing repeated bit shifts, with exact no-carry bounds and independent parity maintained. Register a new source-only timing panel and a phase-matched control before claiming a kernel speedup. NOR52 remains an **algebra-only transparent exact witness** study on synthetic sources. It does not extend NOR45 signed program authorization to m3073, establish honest external provider truth, implement zero-knowledge proof, or include network/signature/transport time in these CPU ratios.
