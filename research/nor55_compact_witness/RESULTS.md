# NOR-55: first source-bound signed compact exact witness verifier

**Original first registered engineering result: FAIL (4/5 gates).**

The first [source-locked Actions run #44292743862](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44292743862) completed 60 fresh, 228 historical-excluded sources with four complete exact inverse/vector and independent NOR45 full-Merkle-root arms, five source-seeded interleaved repetitions. First registration [9f7fb42a](https://github.com/MoralityLabAI/Compactification-Theory/commit/9f7fb42a2e0a8648db63301bed381cb7f3aa0dc2) preceded any NOR55 source or CPU. A separate upstream source-only artifact was frozen before any witness/nonce/timing data.

## First immutable evidence

- Original first CPU JSON: **230,272 bytes**, SHA-256 `98c45ed555349fcfd860def1975d571d0e84565e6cad976d30941cae3384b10e`.
- Original public-only 60-source JSON: **1,474,567 bytes**, SHA-256 `d1e6b2f5f37edc37f4b996b42e10768b47b4c199ed274cf53643be34fddddd59`.
- Separate first two-process TCP loopback secondary JSON: **20,128 bytes**, SHA-256 `45d06c36990cd1cd005e6a0c8478ddca57f52afe1fef168c149e45787063e761`.
- All three original byte streams are preserved inside [results/nor55_first_original_hosted.zip](results/nor55_first_original_hosted.zip); ZIP SHA, original five gate values, source medians and secondary scope are pinned in the [receipt](results/nor55_first_receipt.json), checked by [permanent read-only SHA auditor](original_archive_audit.py).
- First source artifact **14421873595**, first CPU artifact **14421997454**, separate local TCP secondary artifact **14422382237**. Python 3.12, gmpy2 2.3.2, cryptography 46.0.4, Ubuntu 24.04. Test signing fixture key is deterministic and public, **insecure for deployment**.

## Exactly five frozen original first engineering gates

| Registered gate | Original measured result | Verdict |
| --- | ---: | --- |
| Authenticated compact verifier / compressed-native on 16 sparse m3073, <0.85 | **0.744645x** | PASS |
| Authenticated compact / authenticated decimal JSON on same 16, <0.85 | **0.851483x** | **FAIL** |
| Complete signed compact envelope bytes / original full decimal JSON payload bytes on 48 sparse, <0.55 | **0.412842x** | PASS |
| Exact complete signed integers/full Merkle roots and pinned-key/replay/malformed witness controls | **60/60 plus predata CI** | PASS |
| SHA-frozen 60-source upstream, 228 old IDs excluded, five repeats, resource caps, reentrancy, first evidence archive | **PASS** | PASS |

The failed registered codec runtime threshold was missed by about **0.001483x**, which remains an original **FAIL**. Raw source and all five per-arm timing samples are preserved for independent uncertainty analysis. The compact signed format used about **41.3%** of the original unsigned full-decimal JSON witness payload bytes on the registered 48-source sparse group, including the complete signature/header in the compact numerator. This is an approximately **58.7% byte reduction**. On the 16 longest sparse sources, authenticated compact verifier-only execution was about **25.5% faster** than compressed-native recomputation of the inverse, assuming the full witness is already supplied.

## Source profile dependence

| Profile | Sources | Signed compact/native | Signed compact/signed JSON | Compact signed wire/full JSON payload |
| --- | ---: | ---: | ---: | ---: |
| s1-b2 | 12 | 0.233668x | 0.891092x | 0.421277x |
| s4-b2 | 12 | 0.581816x | 0.964914x | 0.396903x |
| s8-b2 | 12 | 0.927397x | 0.941965x | 0.338376x |
| s4-b8 | 12 | 0.974856x | 0.849136x | 0.412914x |
| s4-b16 | 12 | 1.237419x | 0.755283x | 0.414199x |

## Secondary matched-workload total CPU times

| Complete output+full root timed arm | Sum of source medians | Ratio to native |
| --- | ---: | ---: |
| NOR52 compressed native inverse | 5346.427 ms | 1.000000x |
| NOR53 raw supplied inverse verification | 1892.126 ms | 0.353905x |
| NOR55 authenticated compact verification | 1985.250 ms | 0.371323x |
| NOR55 authenticated full JSON verification | 2510.328 ms | 0.469534x |

These are sums of first source-level medians, a registered **secondary** workload measure; they cannot convert the registered 0.851483x primary codec-speed FAIL to a PASS. Ten source-block descriptive bootstrap results are preserved in the original CPU JSON.

## Separate actual local TCP loopback measurement

After primary CPU completion, a second dependent CI job measured **16 sparse m3073 sources**, three each, with the producer and verifier running as separate Python processes connected through **real TCP 127.0.0.1 sockets on the same GitHub runner**. The producer freshly computes a complete compressed native inverse, creates canonical binary payload, signs its independently verifier-issued nonce/session/source/root context, sends the signed wire and receives a verified full-output SHA and original full Merkle root. Server process startup and issuance of nonces occur outside timed samples.

- Sourcewise median complete loopback/local compressed-native same-job ratio: **1.677223x**.
- 95th percentile of 48 request-response samples: **314.710 ms**.
- All **48/48** separate-process signed requests returned exact root and full vector digest. The request has a compact signed full witness; the response contains its exact result root and vector hash.
- The loopback source SHA and original primary CPU SHA are chained in its separate first JSON. The localhost transport latency is **secondary** and does not replace or alter the first five primary engineering gates.

## Security and applicability limits

The verifier checks a pinned **research-fixture** Ed25519 public key, canonical binary source-bound signed header/payload, full authenticated source-vector SHA, trusted original source record SHA, full Merkle-root premise, provider ID, session sequence and independently issued 256-bit nonce before marking a replay-guard session accepted. It rejects tampered signatures, alternate integer encodings, trailing bytes, invalid support, source-swapped data and stale nonces. All full original NOR47 coefficient and packing caps remain enforced before the NOR53 fast exact signed compressed inverse identity and full-root recomputation.

The public fixture signing private seed is included in repository code, so **this is deliberately not secure deployed signing**. Authentication proves integrity only conditional on a private signing key and independently trusted source/root/session; it does not prove the remote source publisher honest or an external future correct. The supplied exact witness is transparent and nonsuccinct. m3073 remains outside original NOR45 native signed-program grammar. Separate local TCP is not cross-machine production network latency or a Bitcoin/ZK proof.

## Follow-on mechanism

A new prospective study can select JSON versus compact payload codec from **authenticated public source complexity** and use source-frozen unseen cases to test whether that improves the signed-verification latency frontier. The original 0.851483x codec gate stays FAIL. An off-host benchmark with genuine independently managed keys and trusted source distribution would require its own security and network protocol registration. Verify source-bound signed arithmetic and sparse support lemmas in Lean before calling this math machine checked.
