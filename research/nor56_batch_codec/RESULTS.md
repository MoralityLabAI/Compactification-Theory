# NOR-56: first prospectively source-locked signed compact-v2 batch decoder

**Original engineering result: FAIL, 3/5 registered gates passed.**

The [first original run #44296895730](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44296895730) completed 60 new SHA-frozen public-only source identities, with 288 first NOR51–55 source IDs excluded. Five source-seeded interleaved repetitions of four complete integer-output/full-NOR45-Merkle-root arms agree exactly. [Registration commit b186ee74](https://github.com/MoralityLabAI/Compactification-Theory/commit/b186ee7478f27230478b099d7091b25817839464) predates every first NOR56 source and benchmark timing.

## Immutable original evidence

- First original CPU JSON: **400,679 bytes**, SHA-256 `2ad2c0e149ca9f2405f7883fbb768fc72f380a03d717c2b265883343a6d7429d`.
- First public-only source JSON: **1,473,496 bytes**, SHA-256 `f1326f4ac9cac314564c96eb82008177f30280b38ab8972eab696f51996b421c`. Uploaded in a separate upstream CI job before any inverse or first timing.
- First source artifact **14423891566** and first CPU artifact **14424252022**.
- Both exact original JSON byte streams are preserved in [nor56_first_original_hosted.zip](results/nor56_first_original_hosted.zip), with its complete ZIP SHA, original gate verdict, source medians and independent phase ledger in [nor56_first_receipt.json](results/nor56_first_receipt.json). [Permanent read-only auditor](original_archive_audit.py) recomputes raw file SHAs, all gate ratios and full-root/sample evidence without source regeneration.
- Runner Ubuntu 24.04, Python 3.12, gmpy2 2.3.2, cryptography 46.0.4. Full integer vectors, original signed caps, pinned test-fixture Ed25519 signature, canonical source/session/nonce and original full roots verified.

## Five preregistered first engineering gates

| Fixed gate | First observed result | Original verdict |
| --- | ---: | --- |
| All 16 sparse m3073: signed v2/v1 verification time <0.92 | **0.978214×** | **FAIL** |
| Same 16: signed v2/full JSON time <0.82 | **0.835501×** | **FAIL** |
| All 48 sparse: complete signed v2/v1 envelope bytes <1.05 | **0.999223×** | PASS |
| All full signed exact integer vectors/original Merkle roots, signed encoding, source, key and replay controls | **60/60 and source-free tests** | PASS |
| First source-only SHA, 288 historical exclusions, five repeats, original caps and reentrancy | **PASS** | PASS |

The sourcewise paired medians indicate v2's authenticated whole-operation verifier is about **2.18% faster** than v1 at m3073 in the 16 sparse cases. Signed v2/full JSON is **0.835501×**, which misses the original registered <0.82 gate. Complete signed v2 bytes are effectively equal to v1 on the 48 sparse cases, at **0.999223×**; both include the signature/header. These are finite original synthetic-source CPU measurements rather than an asymptotic theorem.

## Fixed by-profile sourcewise median ratios

| Source profile | Units | Signed v2/v1 | Signed v2/JSON | Signed v2/v1 wire bytes |
| --- | ---: | ---: | ---: | ---: |
| s1-b2 | 12 | 0.961228× | 0.858228× | 0.996666× |
| s4-b2 | 12 | 0.948732× | 0.924453× | 1.000000× |
| s8-b2 | 12 | 0.969016× | 0.921813× | 1.000000× |
| s4-b8 | 12 | 0.974742× | 0.830332× | 0.996914× |
| s4-b16 | 12 | 0.985342× | 0.748076× | 0.997838× |

## Secondary matched 60-source workload-total CPU

| Exact full-output/root arm | Sum of source medians | Ratio to signed v1 |
| --- | ---: | ---: |
| NOR52 direct compressed native inversion | 6912.131 ms | 2.717481× |
| NOR55 authenticated compact-v1 | 2543.580 ms | 1.000000× |
| NOR55 authenticated full JSON | 3250.265 ms | 1.277831× |
| NOR56 authenticated compact-v2 | 2494.365 ms | 0.980651× |

The totals above are predeclared **secondary sums of 60 source-median timings**. The first registered engineering score uses sourcewise ratios on the sixteen longest sparse inputs and the 48-source wire cohort. Ten matched (replicate,profile) block-bootstrap descriptive intervals remain in first CPU JSON; secondary totals cannot change original 3/5 FAIL.

## Separately measured untimed phase replay

After each source's first whole-arm timings, an independent signed verifier replay checked original root/vector parity while measuring canonical header/replay, Ed25519, payload decode, original support/size caps, exact signed GMP identity and complete full Merkle reconstruction. These are diagnostic original CPU phase observations and do not prove one phase caused the observed whole-arm difference.

| Authenticated format | Canonical decode phase | Exact GMP phase | Full Merkle phase |
| --- | ---: | ---: | ---: |
| Compact-v1 | 3.46% | 79.24% | 13.73% |
| Full decimal JSON | 23.30% | 62.06% | 10.67% |
| Batch-table v2 | 1.53% | 80.94% | 13.89% |

In this **untimed phase replay**, total decoder phase time for the v2 panel was about **56.5% lower** than v1's. Exact GMP and full-root reconstruction account for most measured remaining work, limiting whole-operation gains. The phase attribution is a descriptive replay under the same source mix, not a registered causal phase intervention.

## Boundary and next work

The complete source-bound v2 Ed25519 signed protocol uses a **public-by-construction research-test private key**, which is unsuitable for deployment. The test proves conditional source/session and message integrity given an independently pinned public key, plus an independent exact inverse identity and full Merkle reconstruction. It does not establish an honest remote provider, signed NOR45 native grammar at m3073, cross-host transport, succinct proof size or zero knowledge.

The next experiment should target the **GMP identity/root work** rather than further table parsing, with a clean same-run source-locked arithmetic change. A Lean kernel formalization of the sparse-support substitution and no-carry signed correction remains separate from this runtime result.
