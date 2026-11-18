# NOR-53: first preregistered native GMP pack/unpack exact CPU result

**First original engineering result: PASS, all 5/5 registered gates.**

First [GitHub Actions run #44282552137](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44282552137) completed 60 SHA-disjoint source-only inputs with five interleaved four-arm exact signed CPU comparisons. All complete output integer vectors and original full-length independent Merkle roots matched, with previously verified adversarial source/witness controls. The [registration commit ae56f414](https://github.com/MoralityLabAI/Compactification-Theory/commit/ae56f414251a8485d127fc8f9899a24237870adf) precedes the first source-only job and the first CPU timing.

## Original immutable evidence

- Original first CPU JSON: **142,008 bytes**, SHA256 `b1e4f810624568af8b53794fa9e0a4890961f7f6b5f8dbd3d4c766e511974dee`.
- Original public-only 60-source fixture: **1,473,688 bytes**, SHA256 `b7e65cdacefced6fcf1084922fdaeaaa4da5af1cf9066681471e4be1cd99ad61`.
- Both original byte streams appear unchanged in [results/nor53_first_original_hosted.zip](results/nor53_first_original_hosted.zip). Independent [first receipt](results/nor53_first_receipt.json) pins the ZIP SHA and first gate values; the permanent [original_archive_audit.py](original_archive_audit.py) recalculates source medians, exact full roots and source hashes without retiming.
- First source-only artifact **14416469176** uploaded in the upstream job. First original full CPU artifact **14418133716**. Hosted Python 3.12, gmpy2 2.3.2, same-run five-repeat full-output/root arithmetic; shared session/signature/transport costs excluded equally.

## Original five frozen engineering gates

| Registered gate | Original first observation | Verdict |
| --- | ---: | --- |
| Median fast full GMP/original full GMP on 60 sources <0.70 | **0.143451x** | PASS |
| Eight m3073 high-bit sources fast full/original full <0.50 | **0.072464x** | PASS |
| All 48 sparse fast compressed/original compressed GMP <0.85 | **0.596853x** | PASS |
| All exact signed vectors/full Merkle roots and forged witness/resource controls | **60/60 + predata** | PASS |
| Source-first SHA, original caps, five interleaved repeats and unchanged NOR46 process-global verifier | **PASS** | PASS |

The first sourcewise-median full exact verification ratio corresponds to an **85.7% reduction in time** using GMP native pack/unpack. Among the eight longest, high-bit cases, the sourcewise-median ratio **0.072464x** represents a 92.8% reduction. On all sparse sources, adding the fast native digit packing/extraction on top of NOR52 support compression reduced complete GMP verifier time by about 40.3%.

## By-profile original fixed source medians

| Profile | Source count | Fast full / original full | Fast compressed / original compressed |
| --- | ---: | ---: | ---: |
| s1-b2 | 12 | 0.127573x | 0.127744x |
| s4-b2 | 12 | 0.193185x | 0.700231x |
| s8-b2 | 12 | 0.217764x | 0.902028x |
| s4-b8 | 12 | 0.125316x | 0.403306x |
| s4-b16 | 12 | 0.123974x | 0.374916x |

## Predeclared secondary matched-workload totals

| Complete exact verifier arm | Sum of 60 source-median CPU times | Ratio to original full GMP |
| --- | ---: | ---: |
| Original NOR47 full GMP | 84940.866 ms | 1.000000x |
| NOR53 fast full GMP | 6900.911 ms | 0.081244x |
| NOR52 original compressed GMP | 18952.972 ms | 0.223131x |
| NOR53 fast compressed GMP | 2586.891 ms | 0.030455x |

These are sums of individual source medians, a predeclared secondary throughput diagnostic; the registered engineering criterion uses sourcewise median ratios. Ten matched replicate/profile block bootstrap intervals remain inside the raw first JSON. No first performance gate is changed by these secondary totals.

## Mechanism and limitations

The original NOR47 exact carry-free signed Kronecker multiplication formula and full coefficient/packed-bit resource caps are unchanged. gmpy2.pack/unpack replace repeated Python-loop shifts and extraction while preserving exact signed cross-term correction. This experiment measures the implemented GMP packing API under one hosted Linux CPU, not an asymptotic theorem or a universal runtime guarantee.

The reference full exact inverse was independently generated outside every timed verifier arm, with production/serialization costs recorded separately in the original JSON. All four timed arms validate an untrusted full witness, return the same complete integer vector and rebuild its original full independent Merkle root. The compressed fast arm separately checks the original full witness and source support before shortening the convolution, including the original full resource caps.

This is a transparent, nonsuccinct exact algebra-only kernel experiment. It does not extend NOR45 signed native program grammar to m3073, authenticate an external provider, prove true-world honesty, establish zero knowledge, or include network/signature/transport costs in these CPU ratios.

## Next controlled frontier

Evaluate whether compressed native recurrence and fast native GMP verification cross over by precision/density using a new timing-clean panel. The existing NOR52 compressed-native baseline and the NOR53 fast-GMP source cohorts have different identities/hardware runs and should be compared only through a new same-run controlled benchmark. Add witness serialization/transport and full signed-program acceptance costs as separately reported integration work.
