# NOR-51: first exact reciprocal CPU crossover, source-first prospective result

**Original engineering result: FAIL, 2/5 registered gates passed.** The first prospectively registered hosted [run #44269058977](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44269058977) completed 48 SHA-frozen, previously unused public-only normalized signed integer series from a 2×3×4×2 source grid. All three arithmetic arms produced exactly matching inverses and independently recomputed complete Merkle roots.

## Immutable original first evidence
- [First original CPU JSON inside ZIP](results/nor51_first_original_hosted.zip), **86,314 raw bytes**, SHA256 `a0591f604ff0b2dfc1fa550ff87d22f62f9527b4ee96579674dee83618d7c9ea`.
- First public-only 48-source JSON inside same ZIP, **1,188,767 raw bytes**, SHA256 `0b0de5549de1e1815a40d56302cb01ca104e7ac8141f161bc8cab0b6501d2c93`. Its independent [source-only CI artifact #14411647008](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44269058977/artifacts/14411647008) was SHA-frozen **before first CPU timing**.
- First complete CPU [artifact #14411604927](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44269058977/artifacts/14411604927); first timing registration commit [`83fabe1c`](https://github.com/MoralityLabAI/Compactification-Theory/commit/83fabe1c34c2bedbba35faa928349f41b964115a), preceding source generation.
- Permanent [first source/CPU receipt](results/nor51_first_receipt.json), [read-only original archive audit](original_archive_audit.py) and CI. All five original PASS/FAIL flags retained.
- Exact runtime: ubuntu-24.04, CPython 3.12, pinned `gmpy2==2.3.2`, five interleaved repeats per source. Model and GNU MP version appear in first JSON.

## Five original registered engineering gates

| Registered gate | Original observation | Verdict |
| --- | ---: | --- |
| All 48 full signed exact inverses, roots, tampering controls | **48/48** | PASS |
| Longest m3073 median GMP witness verification/native <0.75× | **2.797903×** | **FAIL** |
| Longest m3073 median Newton generation+verification/native <0.95× | **7.439232×** | **FAIL** |
| 16-cell matched long/short GMP/native ratio slope <0.80× | **2.151121×** | **FAIL** |
| Upstream public-only 48-source SHA, reentrant caps and five repeats | **48/48** | PASS |

The arithmetic reference for each source is **one exact native reciprocal inversion**: already memoized by construction, so it avoids NOR45's repeated-R confound diagnosed in NOR50. The GMP verifier consumes a precomputed *untrusted* inverse and independently checks all coefficients of the signed identity `F(x)R(x)=1 mod x^m`. The Newton arm generates the entire inverse using exact doubled-degree products, then verifies it again with GMP. The full Merkle root is generated in **every timed arm**. Reference inverse production and witness JSON serialization costs are logged separately; common network/session/signing costs are outside this kernel.

## First source-median ratio by length

| Coefficient count m | Sources | GMP verification/native median | Newton generation+verification/native median |
| --- | ---: | ---: | ---: |
| 769 | 16 | **1.497853×** | 3.852123× |
| 1537 | 16 | **2.020282×** | 5.413167× |
| 3073 | 16 | **2.797903×** | 7.439232× |

These are medians of **sourcewise time ratios**, not ratios of group aggregate median milliseconds.

## Profile dependence and negative scaling

| Input source profile | Sources | Median GMP verification/native |
| --- | ---: | ---: |
| Stride 4, 2-bit coefficients | 12 | 0.614799× |
| Stride 1, 2-bit coefficients | 12 | 1.049914× |
| Stride 4, 8-bit coefficients | 12 | 2.899287× |
| Stride 4, 16-bit coefficients | 12 | **4.932078×** |

Rising coefficient width in the stride-four profiles strongly degrades this exact packed GMP implementation at tested lengths; the pooled longest-length result grows worse as m increases. The registered hypothesized crossover failed. Sourcewise raw medians, all five samples, exact witness text size, root and packed width are preserved in the first JSON. The timing pattern is consistent with packing/decoding and large-integer costs; it does not attribute those costs causally without phase-level profiling.

## Predeclared secondary matched-workload totals

| Arm | Sum of the 48 source-median CPU times | Ratio against native |
| --- | ---: | ---: |
| NOR45 single native R+root | **10,122.414 ms** | 1.000× |
| NOR47 GMP verifier+root | **33,689.510 ms** | **3.328209×** |
| Exact Newton producer + GMP verify + root | **86,021.070 ms** | **8.498079×** |

Eight matched (replicate,profile) block bootstrap descriptive 95% intervals are **[1.657,5.405]** for GMP/native and **[4.400,13.292]** for Newton+verify/native. These exploratory workload totals do not change the original 2/5 FAIL.

## Next exact arithmetic mechanism

The source profiles are exactly known, allowing stronger baselines. For the stride-4 subset, if `F(x)=H(x^4)`, then the exact inverse has form `F(x)^{-1}=H^{-1}(x^4)`. Thus the m3073 stride-4 inverse can be computed from a length-769 series before zero expansion, while the Merkle root remains full-length. A fresh registered sparse-aware native and GMP comparison would test this exact structure. Phase-level breakdown of GMP repeated large integer packing/unpacking versus multiplication should precede claims that multiplication itself caused the measured regression. The current single-R kernel has **no signed NOR45 program grammar at m3073**, is not a STARK/SNARK or zero-knowledge proof, and does not establish an honest external target provider.
