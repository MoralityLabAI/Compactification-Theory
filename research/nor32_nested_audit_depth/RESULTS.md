# NOR-32: first preregistered nested audit-depth scaling result

**Registered scientific verdict: PASS on all five prospective gates.** The first protocol registration was committed at [44d8c506](https://github.com/MoralityLabAI/Compactification-Theory/commit/44d8c506de2b2079c167cb0ce5f3a2eb620028b5) before new source generation and OS-root sampling. The first initiated GitHub Actions run was [#44043154269](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44043154269), and its independent source-native/HMAC/greedy replay passed **23,592,960 exact fixed-batch decisions**, verifying the original calibration and evaluation roots, all 18,432 source-depth-attack openings and every registered source-level gate. The original first-run bytes are fixed by [results/nor32_first_original_receipt.json](results/nor32_first_original_receipt.json) and archived in [results/nor32_original_hosted.zip](results/nor32_original_hosted.zip).

## Primary result: the four-unseen-coordinate threshold disappears

The first 24 test sources were new, source-distinct RRC25 problems (replicates 26–29, observation D=512/1024, all three NOR-21 source families); no target values, secrets, changed coordinates or witness data entered their public-only fixture. Separate 256-bit OS calibration root C and evaluation root E produced two independent sets of 2,048 schedules per source/auditor; the four depths below are **nested prefixes of the SAME evaluation-root schedule list**.

For a root-informed attacker committed AFTER viewing a batch, existence of four completely NEVER-inspected positions is necessary and sufficient for a four-coefficient attack detected in **zero out of n audits**. Under the NOR-32 HMAC sampler, the following counts and means were independently reproduced from the first original schedule bytes:

| Nested batch depth n | m=769 sources with ≥4 never-inspected | Mean never inspected at m=769 | m=1537 sources with ≥4 never-inspected | Mean never inspected at m=1537 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 3/12 | 2.917 | **12/12** | 103.667 |
| 1,024 | 0/12 | 0 | **11/12** | 6.583 |
| 1,536 | 0/12 | 0 | **0/12** | 0.250 |
| 2,048 | 0/12 | 0 | **0/12** | 0 |

The preregistered m=1537 gate required at least ten feasible long-output sources at 512 and at most two at 2048; the observed transition **12/12 → 0/12** met it. The finer transition **11/12 at 1,024 → 0/12 at 1,536** was a secondary fixed-depth readout. Because the batches are nested, untouched counts decreased monotonically in every source and policy. The mathematical companion independently proves this finite property in Lean.

## Detection of independent-root versus leaked-root greedy attacks

The attacker optimization algorithm, 32 tie-restart seeds, source and auditor were the same in both conditions, and used exactly n training schedules for each depth. Independent-calibration attacks were committed BEFORE root E existed; leaked-root attacks were selected with knowledge of E and the actual first-n evaluation schedules and were separately committed afterward. One fixed radius-four change set was held constant over the entire n-schedule batch.

| Audits n | Hybrid independent calibration | Hybrid leaked root | Uniform16 independent | Uniform16 leaked root |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 3.1685% | 0.1218% | 6.1493% | 1.0162% |
| 1,024 | 3.2129% | 0.5772% | 6.2393% | 2.2753% |
| 1,536 | 3.1549% | 0.9614% | 6.0400% | 2.8876% |
| 2,048 | **3.2522%** | **1.1561%** | **6.0783%** | **3.1940%** |

Static trained-risk16 was entirely public and deterministic. Its greedy-calibrated and root-informed detection rates were both **0% at every depth**. Its full-batch untouched-coordinate count remained m−16.

The primary 2,048-audit hybrid difference was **+2.09605 percentage points** in favor of precommitted independent-calibration detection, positive on **24/24 fresh source pairs**, clearing the ≥20/24 and +0.5pp registered gate. The source-level median difference was **+2.10190 pp**, with source-specific paired differences from **+1.41296 pp to +3.27148 pp**. A descriptive **retrospective**, source-block bootstrap over 10,000 resamples (analysis seed 202611320117) gave a central 95% percentile interval of **+1.9149 to +2.2938 pp**; this analysis does not change the precommitted decision.

Increasing the *disclosed* batch depth forced the root-aware attacker to place changes where some audit checks would detect them, but its **per-challenge** detection remained lower than the independent-calibration attack rate at 2048. Larger batches reduce the opportunity to find four entirely unseen positions while still allowing a greedy adversary to favor rarely inspected coordinates. The ideal-uniform one-trial fixed blind detection floor (about 4.147% at m=769 and 2.079% at m=1537) applies only to positions fixed independently of the actual root, and individual finite samples can be below the ideal expectation.

## Source-level 2048-audit paired reductions

All values are percentage points (independent-calibration minus leaked-root hybrid detection).

| Replicate | D | Family | Gain |
| ---: | ---: | --- | ---: |
| 26 | 512 | dense | +1.837 |
| 26 | 512 | sparse | +2.100 |
| 26 | 512 | rational | +3.271 |
| 26 | 1024 | dense | +1.579 |
| 26 | 1024 | sparse | +1.425 |
| 26 | 1024 | rational | +1.796 |
| 27 | 512 | dense | +2.629 |
| 27 | 512 | sparse | +2.321 |
| 27 | 512 | rational | +2.289 |
| 27 | 1024 | dense | +1.773 |
| 27 | 1024 | sparse | +2.112 |
| 27 | 1024 | rational | +2.441 |
| 28 | 512 | dense | +2.301 |
| 28 | 512 | sparse | +2.345 |
| 28 | 512 | rational | +2.979 |
| 28 | 1024 | dense | +1.904 |
| 28 | 1024 | sparse | +1.863 |
| 28 | 1024 | rational | +1.413 |
| 29 | 512 | dense | +2.515 |
| 29 | 512 | sparse | +2.536 |
| 29 | 512 | rational | +1.660 |
| 29 | 1024 | dense | +1.555 |
| 29 | 1024 | sparse | +2.104 |
| 29 | 1024 | rational | +1.556 |

These are **24 independent source identities**, with repeated correlated attack restarts and nested audits within every source. The apparent precision of the per-challenge rates should not be interpreted as millions of independent source units.

## Exact finite mathematics and threat-model boundary

The [MathCan NOR-32 companion](https://github.com/patrickdugan/MathCan/pull/124) proves that any batch-fixed alteration evading a longer sequence also evades its prefix, that four-coordinate batch-zero-hit feasibility cannot reappear after more audit checks, and that absolute hit counts for an attack fixed across a batch cannot fall when scans are appended. A separate strict finite counterexample shows fractional hit rates **need not be monotone**: one hit in the first of two audits gives 100% at depth1 and 50% at depth2. The applied rates above happened to rise for the leaked-root greedy arm, an empirical finding under this fixed source/policy/optimizer grid.

Under *ideal independent uniform* risk8/reserve8 sampling, the expected number of never-inspected nonrisk coordinates is N(1−8/N)^n with N=m−8; the first observed long-output means 103.667, 6.583, 0.250 and 0 agree with its predicted decreasing scale. Markov gives Pr(at least four uninspected) ≤ E[uninspected]/4 under that ideal sampling assumption. No perfect unpredictability theorem about HMAC-SHA256, SHA commitment binding, OS entropy or GitHub Actions runner isolation follows from these calculations.

Every one of 9216 original source/depth/salt commitments was validated before sampling the separate evaluation root. All 9216 new postroot openings were valid with different commitments, and none of the attempted retargeted old openings was accepted. Replaying the +1 and +4294049777 nonzero exact coefficient changes yielded identical witness decisions on every one of **23,592,960** fixed-batch audits. All **147,456** schedule-informed per-trial attacks evaded, and their altered openings could not reuse the earlier commitments. Original public/native 16-template masks and true program continuations matched.

## Source-locked first-original evidence

All eight ORIGINAL byte streams were preserved without regeneration, with exact SHA256 recorded in [results/nor32_first_original_receipt.json](results/nor32_first_original_receipt.json). The staged CI job ordering and external OS entropy draws were verified by [Actions #44043154269](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44043154269). The [source-controlled original evidence ZIP](results/nor32_original_hosted.zip) contains the source-only fixture, calibration root and 2048 schedules, first 9216 precommit openings/manifest, evaluation root and 2048 schedules, 9216 postroot openings/manifest, and full detected/undetected decision masks.

The first original hashes are: fixture c0023b7523843bb61fc2ec408b61236757fd1cd250fb22158f360a873d29f6ef; calibration c56590ae0f00b34647c52ecea5afd6820b67096758b4e520f1187ca1d881a18e; precommit openings fa3249c47f17d23a395aeb430af48879fbdbba0342e9635004c5615556765e7b; precommit manifest c9734a14fbeab9b2005fce6e86a1beca3eb81d90628b6de0d2954261ff0d082b; evaluation 0fa8f37bcdec68e1872ae0eb48acbbce249f57a91ef9a6003bf58fa6e15af6de; postroot openings c0a0a241762bad54652b608332f226e921bac0ccdaa6cfb0a135a0a4cd11ca1d; postroot manifest 9f02d769a57b5f0ecd0c8355c0f85e56af605741456a6ce2eb52c13f1b8e3281; full original outcome 51f28b3028a783b6182c8549f69d85119be27f16c92632ebd786891f60ba5833.

A separate [original_archive_audit.py](original_archive_audit.py) checks the eight *actual raw bytestrings* and their scientific verdict without importing NOR-32. The first-run [receipt_audit.py](receipt_audit.py) independently replayed both OS-root HMAC schedule sequences using a sparse Fisher-Yates algorithm, source-native targets and every greedy attack using trial-ID sets. The original files are bound in the permanent repository so future CI can reject any substituted artifact.

## Concurrent independent root realization

A second PR-triggered evaluation on the **same 24 source identities but independently minted OS roots** completed its scientific grading: 12/12 long sources feasible at depth512 and 0/12 at depth2048, 24/24 positive 2048-audit hybrid source pairs, and a **+1.9570 pp** 2048 hybrid detection difference. It is a separate randomness realization and does not double the independent source count. The first initiated run above remains the registered primary record.
