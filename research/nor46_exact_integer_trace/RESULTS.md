# NOR-46: transparent exact-integer full-trace certificate, historical verification and CPU benchmark

**Engineering correctness PASS; verification-speed goal FAILED on the tested workloads.** NOR-46 is a historical-only implementation of the [design issue #243](https://github.com/MoralityLabAI/Compactification-Theory/issues/243), with **zero new NOR46 source identities or prospectively graded model outcomes**. It replaces NOR45's direct recomputation of the authorized program's reciprocal coefficients with a transparent **complete exact-integer execution transcript**. The certificate verifies arithmetic and the already independently signed source/session/output root, with full witness disclosure. It is neither succinct, zero-knowledge, STARK, nor zkVM.

## First archived CPU correctness receipt

The [original hosted test run #44236764485](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44236764485) passed all exact-integer, authorized-source/signature, Merkle-root, perturbation and dishonest-authority tests. The original immutable [16-source receipt](results/nor46_first_historical_16.json) is **19,583 bytes**, SHA256 `d24bb2fec989a5cbe7422d3aae6e2e2779a9d780268f25f0f23ccf28f56db052`, first uploaded as [artifact #14398140866](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44236764485/artifacts/14398140866), then committed byte-identically by [one-shot publisher #44237652833](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44237652833). [original_archive_audit.py](original_archive_audit.py) verifies its exact SHA and original 16-source control booleans.

The historical sources were selected **before examining NOR46 CPU timing**: the original NOR42 source-pair archive's replicate groups **100, 104, 108, 112**, both observed degrees D512 and D1024, and both dense and block-regime input families. Each group contains four paired source cells, yielding 16 archived source identities and 16 already fixed NOR45 sparse challenges. Their prior NOR45 first raw 64-source receipt is **112,573 bytes**, SHA256 `c4b1197b87b43c4cb4e32e0c2708c4a070742244d7cc05639e22353af78d5536`.

| Archived-original verification control | Result |
| --- | --- |
| Complete independent transform/output trace matches each authorized program and original signed Merkle root | **16/16** |
| Original NOR45 fixed k16 challenge decision reproduced | **16/16** |
| Original sample cases whose authorized program includes reciprocal `R` | **12/16** |
| Tampered trace output coefficient rejected | **16/16** |
| Dishonest output provider root rejected despite valid provider signature | **16/16** |
| Semantically dishonest but separately authorized clean program accepted as faithful computation | **16/16**, although all miss the designated hidden true future |
| New prospective NOR46 source records | **0** |

The last control matters: **a valid proof of a falsely authorized program remains semantically wrong**. The reference program signer is the trust root, not a theorem about external world truth. Signing keys and challenge nonces in this research harness are public reproducible TEST fixtures.

## Exact integer proof relation and arithmetic

The source-normalized unary transforms I/N/S are checked at **every output coordinate**. For a reciprocal R, the separate verifier checks `F_0=R_0=1` and **all** exact signed convolution equations `(F*R)_0=1`, `(F*R)_n=0` for `1≤n<m`. One large carry-free Kronecker substitution packs positive-shifted coefficient arrays into a Python unbounded integer. The shift width satisfies `2^width > 4 m K²` for `K=2^maxCoefficientBitLength`, keeping every shifted convolution digit strictly below the base. The verifier extracts each exact convolution coefficient and subtracts signed offset cross terms; **no field reduction or unchecked modular wraparound** is used. It then checks every original RRC25 delayed suffix, optional hidden `βX^(D+32)T(F)`, and complete output vector.

Finally, the verifier checks the *first signed proposal precommit*, independently pinned NOR45 program-authority signature, NOR44 source/nonce/session/query/context, separate provider signature and exact Merkle root **against the trace-validated output**. The prover may reveal a false trace, but the verifier rejects any trace that fails the normalized transform identities, output equation or authorized signed root.

[Independent Python verifier](trace_verifier.py) is separate from the [witness producer](trace_producer.py) and does not call NOR45's original execution recurrence. The original NOR45 reference interpreter is used only to *generate the prover's full trace* and as an independently benchmarked arithmetic baseline.

The [MathCan PR #146](https://github.com/patrickdugan/MathCan/pull/146), kernel-verified and merged at commit `9b2e9f5a258b2a0bfa31489333e1e5b9b86b7566`, proves uniqueness of a normalized exact-integer coefficient sequence under a strictly causal recurrence and conditional composition of verified full-trace output with signed authorized roots and verifier-owned read indices. It **does not formally prove** the concrete Python Kronecker bit-packing implementation or SHA-256/Ed25519 security. Those remain separate executable checks and cryptographic assumptions.

## Original exploratory CPU cost

The first hosted [benchmark run #44237967510](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44237967510) completed all three jobs successfully. Its raw 16-source CPU JSON is **11,884 bytes**, SHA256 `4e9695f89aaa8acf42c09c6fe7aea9ee47ca0f86f74f6131d3e108e587454a64`, first published as [artifact #14397639639](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44237967510/artifacts/14397639639). Hardware: **AMD EPYC 7763 64-Core Processor**, hosted runner with four logical CPUs, CPython **3.12.16**. Every timed arm is the **median of three repetitions per source**, aggregated across the 16 historical source pairs. The output includes Merkle root recomputation in both verifier arms; common source/session signature authentication and transport are excluded from both timing arms. The prover trace-construction median is reported separately.

| Original historical subset | Pairs | Median prover trace creation | NOR45 full execution + root median | NOR46 full trace check + root median | Median per-source NOR46/NOR45 ratio | Mean full trace payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D512, future length **769** | 8 | 34.58 ms | **38.23 ms** | **110.39 ms** | 1.88× | 137,100 B |
| D1024, future length **1537** | 8 | 145.15 ms | **224.54 ms** | **961.93 ms** | 3.62× | 395,923 B |
| **All 16** | **16** | **35.72 ms** | **64.40 ms** | **262.99 ms** | **3.28×** | **266,512 B** |

The ratio is the **median of the per-source runtime ratios**. It need not equal the ratio of aggregate medians, because source programs differ in cost. Twelve cases require R and have median NOR46/NOR45 ratio **4.05×**; the four without R have ratio **1.86×**. The observed proof bytes and verifier work are far larger than the output/query budget of a cryptographic succinct proof.

The benchmark reported **63,232 KiB** process-level cumulative peak RSS. That is a single process-wide maximum and cannot be attributed to the verifier or prover arm separately. CPU scheduling and allocation variability limit interpretation of absolute wall times across different hosted machines. These numbers provide a concrete negative performance result for **this Python implementation and these m=769/1537 workloads**, rather than an asymptotic impossibility theorem for fast transparent execution proofs.

## Architectural consequence

The full trace is correct and verifiable, but the BigInt digit-packing and handling cost outweigh direct NOR45 computation at tested lengths. Improving it requires a different kernel, e.g. Rust fast polynomial multiplication with exact signed-CRT coefficient bounds, a transparent AIR/STARK proving *all* native integer limbs and output commitments, or a zkVM with independently checked exact bigint semantics. Merkle commitments or spot checks alone cannot upgrade a probabilistic local trace sampler into a full-execution proof.

Any future proof system must retain the separately authorized program and session-bound source/query/proposal, exact signed output root, honest authority limitation and integer range/limb bound. Benchmark the actual **proof bytes, prover CPU/GPU cost, verifier CPU and memory**, alongside both old NOR45 full recomputation and the non-succinct NOR46 trace. A fresh registered experimental cohort or any claimed cryptographic proof soundness requires separate evidence.

**Scientific claim:** the protocol certifies faithful computation of a separately authorized synthetic native program under stated external trust assumptions. It does not prove that program is the actual hidden true world, nor supply secret production signing keys or a deployed oracle.
