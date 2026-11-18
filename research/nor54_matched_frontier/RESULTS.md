# NOR-54: first source-locked native vs fast GMP and local producer crossover

**Original first engineering result: FAIL (4/5 registered gates).**

The [first hosted run #44288506867](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44288506867) completed 60 new SHA-frozen public-only sources after registration commit [10e7ac3c](https://github.com/MoralityLabAI/Compactification-Theory/commit/10e7ac3c3fdefec89d90d7a98a947099db8af86e). All five timed arms produced exactly identical complete integer inverse vectors and independently recomputed full original NOR45 Merkle roots. The first source-only job uploaded the original 60 source IDs before any inverse timing.

## Immutable first evidence

- Original CPU JSON **177,615 raw bytes**, SHA256 `27f7926a53aee3bd25520a1c9e227041997b4bdb9a92a8876cd73af267f67ed1`.
- Original public-only source fixture **1,474,198 raw bytes**, SHA256 `9ac35fdee19a8b9a5b2d08c1ff890bddb3141b69232c4a2687d46195c37a627d`.
- Original source job artifact **14420037504** preceded dependent first CPU job artifact **14420281905**. Both exact original JSON byte streams are inside [source-controlled first ZIP](results/nor54_first_original_hosted.zip), with separately anchored ZIP hash and original gate flags in [first receipt](results/nor54_first_receipt.json) and permanent [read-only SHA/ratio auditor](original_archive_audit.py).
- CPU: Ubuntu 24.04, Python 3.12, gmpy2 2.3.2, five deterministic source-seeded interleaved repetitions per arm. The 168 original NOR51–53 SHA identities were excluded.

## Exactly five first registered engineering gates

| Frozen gate | Original result | Verdict |
| --- | ---: | --- |
| Sparse m3073, 16 cases: fast compressed GMP verifier / compressed native <0.85 | **0.740030x** | PASS |
| Same 16: local full witness produce + serialize/parse + fast verify / compressed native <1.50 | **1.711785x** | **FAIL** |
| Dense stride1 controls, native compressed/full and fast GMP compressed/full both <1.10 | **1.005203x**, **1.014643x** | PASS |
| All 60 exact output vectors/full Merkle roots and adversarial tampering controls | **60/60 plus predata** | PASS |
| Source-first SHA, 168 prior exclusions, five repeats, signed GMP caps, immutable archive | **PASS** | PASS |

## Verifier service economics

On the exact same sixteen sparse m3073 inputs, supplying the complete untrusted inverse witness made fast packed-GMP verification **26.0% faster** by median sourcewise ratio than recomputing that inverse with exact compressed native R. The registered full local computation arm includes fresh native compressed inverse generation, entire decimal witness JSON serialization and parsing, a separate exact fast GMP verification and the full Merkle root. Its **1.711785x** median sourcewise ratio is **71.2% slower** than compressed-native direct execution and fails the original <1.50 gate. This study provides evidence that cheap verification of a supplied exact witness need not imply cheaper complete local computation. Network transfer, provider signing/session and honesty costs remain external.

## Source profile dependence

| Profile | Sources | Fast compressed GMP verifier / compressed native | Local end-to-end / compressed native |
| --- | ---: | ---: | ---: |
| s1-b2 | 12 | 0.219309x | 1.196232x |
| s4-b2 | 12 | 0.568874x | 1.339254x |
| s8-b2 | 12 | 0.862039x | 1.330663x |
| s4-b8 | 12 | 0.986234x | 2.025315x |
| s4-b16 | 12 | 1.215004x | 2.657189x |

Across all source lengths and profiles, median sourcewise fast compressed GMP/compressed native was **0.782652x** and local end-to-end/compressed native **1.394233x**. These pooled ratios have different population weights from the **registered sixteen-case m3073 sparse primary**, whose local end-to-end gate FAILS. There is precision-dependent crossover: at stride4-b16, GMP verifier/compressed native rises to 1.215x by sourcewise median.

## Predeclared secondary matched-workload totals

| Whole-output and full-root timed arm | Sum of source-median wall times | Ratio to compressed native |
| --- | ---: | ---: |
| Original full native inverse | 8939.227 ms | 2.109367x |
| Support-compressed native inverse | 4237.871 ms | 1.000000x |
| Fast full GMP witness verification | 4410.642 ms | 1.040768x |
| Fast compressed GMP verification | 1533.661 ms | 0.361894x |
| Local compressed produce/JSON/fast verify | 6351.823 ms | 1.498824x |

The local producer+verifier workload-total ratio across all 60 source medians was **1.498824x**; this exploratory secondary aggregate does not override the original sparse-m3073 primary **FAIL**. Ten matched replicate/profile block resampling intervals are preserved in the first CPU JSON.

## Full-length output and independent cost ledgers

The independent NOR51 full native reference inverse was created once outside each source’s five-arm timing loop and used as the same untrusted witness for the two GMP verifier-only arms. Its production time, exact canonical decimal JSON size and serialization time were recorded separately. The end-to-end arm freshly generated its own compressed inverse on **each timed invocation**, serialized and parsed that full witness, checked original full source/witness bounds and exact signed identity through NOR53 fast GMP, then rebuilt one complete original Merkle root. The SHA identity of each source, original resource bounds, each five-sample arm timing and identical output root are in the first CPU JSON.

## Limits and next controlled study

These are synthetic algebra-only exact signed integer series. At m3073 this kernel is outside NOR45’s currently authenticated signed native program grammar. No remote provider identity, external honest target, zero-knowledge proof, succinctness, actual network latency or session signature costs are established. A next experiment should measure a separately authenticated producer-to-verifier channel and register a selector using only public signed source complexity, with root verification and adversarial provider replay preserved. Distinguish local production cost from a justified remote-verification use case.
