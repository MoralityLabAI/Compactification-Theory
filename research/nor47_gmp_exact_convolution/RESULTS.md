# NOR-47: exact GMP acceleration of the full transparent signed-integer trace

**Original historical engineering result: PASS, both frozen speed thresholds met.** This work follows NOR46's correct yet slow non-succinct exact integer execution trace and keeps the same preselected **16 completed NOR42 source pairs**. It is a computational engineering benchmark, with **zero fresh NOR47 source identities** and no new AI model-evaluation claims.

## Original immutable CPU evidence

[First source-locked hosted CI #44240888113](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44240888113) completed both jobs successfully. The initial exactness suite passed all five groups, including random positive/negative **4,097-bit** integer polynomial products, signed inverse tampering and the historical 16-source signed program/session/root negative cases under both GMP backends.

| First original receipt | Source |
| --- | --- |
| Original exact 16-source NOR47 CPU JSON | [results/nor47_first_original_cpu.json](results/nor47_first_original_cpu.json) |
| Original immutable first receipt | [results/nor47_first_receipt.json](results/nor47_first_receipt.json) |
| First hosted benchmark artifact | [#14400450017](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44240888113/artifacts/14400450017) |
| Original CPU JSON size and SHA-256 | **34,928 bytes**, `809fd5d18cfb61e4e6d771c4e402b31e3fec99889774f4305ddfa3d6c793f530` |
| Historical NOR46 first trace SHA-256 | `d24bb2fec989a5cbe7422d3aae6e2e2779a9d780268f25f0f23ccf28f56db052` |
| Historical NOR46 first CPU SHA-256 | `4e9695f89aaa8acf42c09c6fe7aea9ee47ca0f86f74f6131d3e108e587454a64` |

The [one-shot archive publisher](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44241507297) committed the first benchmark raw JSON **byte-for-byte** before source-level posthoc interpretation. It preserved the original two engineering PASS booleans. The read-only [original_archive_audit.py](original_archive_audit.py) verifies original bytes, per-source medians and source identities; it never regenerates a benchmark.

## Same-run four-arm comparison

CPU was an **AMD EPYC 9V74 80-Core Processor**, CPython 3.12, `gmpy2 2.3.2`, **GMP 6.3.0**. Every historical source uses five independent wall-time repetitions per arm, with source/repeat-seeded interleaved arm order. All arms rebuild the complete output Merkle root. Transport, provider/session signing and authorization checks are excluded from every timed verification arm, following NOR46's original benchmark methodology; authorization/tampering correctness is tested separately.

| Same-host whole verifier + output root | Median across 16 source medians |
| --- | ---: |
| NOR45 direct exact native execution and Merkle root | **49.770 ms** |
| NOR46 Python packed exact full-trace verifier and root | **225.853 ms** |
| **NOR47 GMP packed exact full-trace verifier and root** | **28.506 ms** |
| NOR47 Python packing + GMP multiply full-trace verifier and root | **64.909 ms** |

The original **median of per-source ratios** is GMP/NOR45 **0.8104×** and GMP/NOR46 **0.1267×**, both below the frozen thresholds **1.0×** and **0.50×**. NOR47 GMP wins against NOR45 in **10/16** sources, and is over 2× faster than NOR46 Python in **12/16**. The hybrid Python-pack/GMP-multiply arm is slower than GMP-native packing, with source-median hybrid/NOR45 ratio **1.7711×**.

These are source-paired timing ratios on the **same NOR47 hosted runner**, not NOR47 timings divided by the older NOR46 run on a different machine. The sourcewise median ratio differs mathematically from the ratio of aggregate median milliseconds because different native programs have different costs.

## Length and reciprocal-transform strata

| Stratum | Source pairs | NOR45 median ms | NOR46 Python median ms | NOR47 GMP median ms | GMP/NOR45 median source ratio | GMP/NOR46 median source ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D512, future m=769 | 8 | 29.322 | 94.645 | 15.392 | **0.9815×** | **0.1630×** |
| D1024, future m=1537 | 8 | 169.266 | 831.586 | 95.072 | **0.6629×** | **0.1126×** |
| **Contains reciprocal R** | **12** | 93.150 | 470.249 | 55.477 | **0.4966×** | **0.1227×** |
| **No reciprocal R** | **4** | 2.908 | 5.173 | 5.155 | **1.8137×** | 1.0003× |

The acceleration is concentrated in reciprocal R traces. All 12 R-containing sources beat the 2×-faster-than-NOR46 goal, and **10 of those 12** also beat NOR45 direct execution. All four no-R sources are slower than NOR45, because the GMP convolution kernel is unused and the complete trace still carries validation overhead.

For R traces alone, median exact convolution kernel time is **463.56 ms** for original Python packing, **47.58 ms** for GMP packing/multiplication and **139.43 ms** for the Python-pack/GMP-multiply hybrid. This strongly implicates the original Python bigint arithmetic and packing as the largest NOR46 performance cost on these workloads.

The median complete transparent trace payload is **232,681 bytes**, with source-dependent output length and operation count. The first CPU process reported cumulative peak RSS **65,864 KiB**. That figure is a process-wide maximum and cannot isolate GMP memory usage from the full harness.

## Exactness and authenticated-root boundary

The GMP version uses the same signed no-carry identity as NOR46: for length m and offset `K=2^maxbits`, shift each signed coefficient into `[0,2K)`, choose digit base `L>4mK²`, multiply the packed integers once and subtract exact offset cross terms for every convolution coefficient. Each inverse R trace must satisfy the complete integer identity `F*R≡1 mod x^m`. Values are arbitrary signed Python/GMP big integers, with **no finite-field reduction** and the original coefficient/packed-bit resource caps.

The NOR47 historical suite verified exact parity of every original native program output, signed source/query/session/program/root and the negative controls under both GMP kernels, including the deliberately semantically false but properly authorized clean-program example. For performance measurement, a single-threaded benchmark-only context manager swaps the convolution implementation into the unchanged NOR46 full-trace verifier and restores it in a `finally` block. This patch is **not suitable for concurrent production verification**. A production backend-selection interface would require a separate security review.

An authentic NOR45 program-authority signature establishes source-bound program provenance under an independently pinned public key; executing that program faithfully does not establish semantic truth when the authorized program itself is wrong. This result preserves the original NOR43–46 caveats about public reproducible test keys, synthetic hidden worlds, computational assumptions for SHA-256/Ed25519 and a missing deployed ground-truth provider.

## Limits and next decision

The source cohort comprises **four historical replicate blocks of four sources**. The 10/16 overall source wins and four no-R losses make program mix important. Across those four groups, a descriptive t95 interval of the **replicate-mean log GMP/NOR45 ratio**, exponentiated back to ratio units, is approximately **0.444–1.143**. This interval includes 1.0, so the 0.8104× fixed-cohort source-median engineering PASS is not evidence that the speed advantage generalizes to an arbitrary distribution of native programs.

The next source-independent performance gym should choose the arithmetic backend **from the already signed authorized program**: use complete GMP trace verification for R-bearing programs and NOR45 direct recomputation for non-R programs. It must preserve all original source/session/root authorization checks and benchmark its dispatcher as a separately registered arm, avoiding changes to NOR47's first four-arm protocol or original timing receipt. An actual succinct proof design requires new work on exact integer limbs and independently checked proof-system soundness.
