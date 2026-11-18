# NOR-50: memoization-controlled exact inverse CPU benchmark

**First original engineering verdict: FAIL (4/5 registered gates).** The source-registered [first hosted run #44265466762](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44265466762) completed 64 new source identities in 16 replicate blocks 200–215, with all four exact output vector and independent Merkle-root arms matching. Registration [a3b7a211](https://github.com/MoralityLabAI/Compactification-Theory/commit/a3b7a211e312235f8e9363612aff4173f50a3a4e) predates first source generation and timing. Source-only roster was uploaded and SHA-frozen before any CPU job.

## Immutable original first-run evidence
- The original CPU JSON is **118,129 bytes** with SHA256 `69344bc6be09246296a5d6c0ac2e22b58031da2ed303f67be69d49bd78cb7f2e`.
- The original 64-source public-only fixture is **9,428,144 bytes** with SHA256 `5a64a6d0ff9d9a4d74535443bc02b1d069cb6bc76b441da35acdb016dd1d9a93`.
- Both byte-identical files are retained in [results/nor50_first_original_hosted.zip](results/nor50_first_original_hosted.zip), whose independent ZIP SHA is captured in [results/nor50_first_receipt.json](results/nor50_first_receipt.json). The [original_archive_audit.py](original_archive_audit.py) independently recomputes full-source counts, source-level median timing ratios, gate booleans, and byte hashes.
- Original artifact ID **14409649467**; first pre-CPU roster artifact ID **14409798029**.
- Signed-program/session/provider, root tampering and source controls run in an unchanged separate predata suite; all original exact/root comparisons passed. The public signing keys are research fixtures.

## Five original registered gates
| Frozen gate | Original observed | Verdict |
| --- | ---: | --- |
| R>=2 memoized/direct sourcewise median <0.70 and n>=8 | 19 sources, **0.511636×** | PASS |
| R>=2 GMP/memoized sourcewise median <0.90 | **1.183263×** | **FAIL** |
| R<=1 memoized/direct sourcewise median <1.10 and n>=8 | 45 sources, **1.002072×** | PASS |
| Exact vectors/root parity and deduplication/signed controls | **64/64**, predata green | PASS |
| New disjoint source, upstream SHA, reentrancy, 5 repeats | **64 sources, 16 blocks** | PASS |

Signed active-R counts were 17 zero-R, 28 one-R, 16 two-R, and three three-R programs. For R>=2, the NOR45 memoized native executor was approximately 48.8% faster than the repeated-recomputation original. NOR47 GMP full-trace verification was **18.3% slower** than cached direct on the same sourcewise-median criterion. Part of NOR47–49's apparent GMP advantage therefore came from redundant inverse recomputation in the original NOR45 baseline. This is an engineering result on the registered finite synthetic source cohort.

## Predeclared secondary matched-workload totals
Summing the 64 per-source median complete arithmetic+Merkle runtimes:
| Exact backend | Sum of source-median runtimes | Ratio to NOR45 |
| --- | ---: | ---: |
| NOR45 original direct | 5548.319 ms | 1.000× |
| NOR50 memoized exact native | **4068.977 ms** | **0.733371×** |
| NOR49 count-R dispatcher | 4912.469 ms | 0.885398× |
| NOR47 full GMP trace | 7487.768 ms | 1.349556× |

The 8,192-resample 16-replicate-block descriptive bootstrap interval for the memoized/original matched-workload ratio was **[0.6580, 0.8261]**. These secondary workload totals illuminate a mixed program population and never replace the original registered **4/5 FAIL**. Trace production times and transparent payload bytes are recorded separately, outside each equally charged arithmetic+root arm.

## Next mechanism direction
Use the cached native executor as the direct-computation comparator in later exact-verifier studies. Test high-bit-width and longer series on a new precommitted cohort before claiming an asymptotic GMP advantage, and account for trace production/transfer, source-aware routing, and independent signed provider checks. The current transparent witness is nonsuccinct and neither honest-program authority nor external true-future correctness is established by these synthetic CPU results.
