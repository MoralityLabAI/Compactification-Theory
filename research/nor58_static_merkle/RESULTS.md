# NOR-58 first exact indexed static Merkle template evidence

**First prospectively frozen engineering result: PASS (all 5/5 original gates).**

The [first GitHub Actions run #44316658346](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44316658346) completed 60 new SHA-disjoint public-only sources, with all 468 original NOR51–57B source IDs excluded, and five interleaved exact signed full-output/full-root CPU repetitions of three matching-verifier arms per source. [Registration commit ddbabc02](https://github.com/MoralityLabAI/Compactification-Theory/commit/ddbabc02cf8702fe2da2c9962174ddcfcbed39a4) predates all NOR58 source/input and template timing.

## Immutable source-first evidence

- First original CPU JSON: **204,565 exact bytes**, SHA-256 244d84accfd2821af25a3d1d98c114964a217925fa457451d2c83c1d6793342b.
- First upstream public-only 60-source JSON: **1,473,865 exact bytes**, SHA-256 6b34e964bc170f3e8efcdfae23403fead030992cf1fc08640ed13c2e34c6b5b3.
- GitHub source artifact **#14431354312** was uploaded before the first dependent CPU artifact **#14433176882**.
- Both raw first byte streams are retained within [nor58_first_original_hosted.zip](results/nor58_first_original_hosted.zip), SHA and first Boolean gates pinned in [receipt](results/nor58_first_receipt.json), independently checked by [original_archive_audit.py](original_archive_audit.py).

## Exactly five registered first engineering gates

| Registered original gate | First observation | Verdict |
| --- | ---: | --- |
| 16 sparse m3073 sources, warm signed GMP verifier/full root vs original <0.93x | **0.802428x** | PASS |
| 60 source ratio of sums of source medians warm/original <0.95x | **0.911459x** | PASS |
| 12 dense stride1 controls, cold/original sourcewise median <1.20x | **1.032694x** | PASS |
| All original exact signed integer output vectors/full original NOR45 Merkle roots and malicious signing/source/witness controls | **60/60 plus source-free CI** | PASS |
| SHA frozen source roster, 468 historical exclusions, five repeats, original resource caps, immutable static geometry and full evidence | **PASS** | PASS |

The registered longest sparse sourcewise median warm/full signed root ratio corresponds to a **19.8% full-verifier CPU reduction**. Across all 60 first sources, the ratio of sums was **0.911459x**, an **8.9% total full-verifier CPU saving**. Every signed verifier uses identical signed compact-v2 bytes, pinned source, expected root, key, verifier nonce/sequence, original NOR47 full-sized resource bounds and NOR53 exact reduced signed GMP identity. The only timed algorithmic difference is the full original NOR45 Merkle-root construction.

## Original matched workload sums

| Whole authenticated exact output/full Merkle verifier | Sum of 60 source medians | Ratio to original |
| --- | ---: | ---: |
| Original NOR45 full Merkle | 2100.047 ms | 1.000000x |
| Warm indexed-zero/pad immutable template | **1914.107 ms** | **0.911459x** |
| Cold fresh template constructed per call | 2148.014 ms | 1.022841x |

Cold per-call template construction remains slightly slower over the complete workload, even though the registered dense cold-control overhead was only ~3.3%. Warm templates amortize geometry-only costs across repeated independently signed requests, with no reuse of dynamic hashes or completed roots. The ten-block descriptive uncertainty is in first CPU JSON; sourcewise and workload ratios are distinguished in registered scoring.

## First by-profile warm/cold sourcewise medians

| Fixed source profile | Sources | Warm / original | Cold / original |
| --- | ---: | ---: | ---: |
| s1-b2 | 12 | 0.980366x | 1.032694x |
| s4-b2 | 12 | 0.602699x | 1.058939x |
| s8-b2 | 12 | 0.445987x | 1.061040x |
| s4-b8 | 12 | 0.842906x | 1.028250x |
| s4-b16 | 12 | 0.918849x | 1.017482x |

## Immutable indexed-only templates and original root-only diagnostics

Each fixed (full m, trusted support stride) template precomputes only SHA256 of canonical index-specific full-zero leaves, canonical index-specific padding leaves and internal nodes with exclusively fixed descendants. It stores no full signed witness or output-dependent Merkle root. Every supported coefficient leaf is hashed fresh on each invocation; NOR45's exact original leaf, padding, internal-node domains and full index framing are retained.

On the 16 sparse m3073 first sources, the independently timed post-whole-arm root-only diagnostic gave median original full Merkle **10.094 ms** and median warm-template original root **3.922 ms**. This isolated root-only diagnostic is strictly secondary.

### Template construction by original trusted geometry

| Trusted geometry | One template build | Canonical static template bytes | Static node hashes |
| --- | ---: | ---: | ---: |
| m769-s1 | 0.916 ms | 18,124 | 502 |
| m769-s4 | 1.759 ms | 45,772 | 1,270 |
| m769-s8 | 2.048 ms | 56,140 | 1,558 |
| m1537-s1 | 1.834 ms | 36,520 | 1,013 |
| m1537-s4 | 3.522 ms | 91,816 | 2,549 |
| m1537-s8 | 4.058 ms | 112,552 | 3,125 |
| m3073-s1 | 3.514 ms | 73,348 | 2,036 |
| m3073-s4 | 7.311 ms | 183,940 | 5,108 |
| m3073-s8 | 8.496 ms | 225,412 | 6,260 |

The original CPU JSON contains exact source-specific template geometry digests, signed original roots, original full-vs-compressed resource caps, all five interleaved arm samples, one-time reference witness and fixture signer costs, warm amortized five-request diagnostics, and 4,096-resample ten-block descriptive bootstrap results. The first five gate decisions remain independent of these descriptive breakdowns.

## Security and formal limits

This is a synthetic exact integer formal-series verifier with publicly reproducible research-fixture Ed25519 signer seed. Original full source and every supplied complete full signed witness are checked for unit, exact integer types, original full NOR47 coefficient and packed caps, and all off-support zeros **before** using a static template. Exact signed inverse identity and original full Merkle root must match the independently trusted reference to advance the replay guard. The observed speedup is a legitimate changed root-computation kernel under shared signing semantics, outside the earlier NOR57B fixed two-backend routing-only ceiling; the two studies used different source panels and their raw CPU times must not be treated as directly paired.

The result proves neither genuine independent provider honesty nor secure deployed keys, off-host transport, Bitcoin consensus, zero knowledge or succinctness. m3073 remains outside the original NOR45 signed native-program grammar. Merkle-template extensional equivalence requires a separately checked Lean/MathCan theorem before being called machine-checked.
