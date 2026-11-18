# NOR-47: GMP exact signed convolution acceleration, historical-only engineering test

[Design issue #248](https://github.com/MoralityLabAI/Compactification-Theory/issues/248) records the new engineering question before any NOR47 benchmark, motivated by the merged [NOR46 first source-frozen CPU results](https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/nor46_exact_integer_trace/RESULTS.md). This is a **historical-only performance experiment** on the exact same 16 SHA-archived NOR42 source pairs, with **zero new prospectively generated NOR47 source identities**.

## Frozen input and measurement protocol

The source-cohort is NOR46's preselected original replicate blocks **100, 104, 108, 112**, each with D512/D1024 and dense/block-regime input families. The first NOR46 original 16-source trace receipt is 19,583 bytes SHA256 `d24bb2fec989a5cbe7422d3aae6e2e2779a9d780268f25f0f23ccf28f56db052`; the first 16-source CPU performance JSON is 11,884 bytes SHA256 `4e9695f89aaa8acf42c09c6fe7aea9ee47ca0f86f74f6131d3e108e587454a64`. Both are committed on `main` before this work.

All four arms run on the **same CPU machine with identical archived inputs**, signed original program/root expectations and full output hash:
1. NOR45 original exact full execution and independent Merkle root rebuild, as the baseline.
2. NOR46 original pure-Python exact full trace verification plus root rebuild.
3. NOR47 pure-GMP carry-free Kronecker signed convolution for every `R` transform inside full NOR46 trace verification plus root rebuild.
4. NOR47 hybrid Python-int packing followed by GMP multiplication and exact GMP digit extraction, again inside full NOR46 trace verification plus root rebuild.

Every arm reports the median of **five** repetitions for each source with seeded, interleaved arm order to limit order/warmup bias. Additionally report **R-convolution-only** medians for each of the 12 R-bearing source programs; mark these absent for the 4 non-R cases. Compare exact full traces, all coefficients and the **original signed NOR46 authorized program root** at every source. The full signed session/authority/provider checks are independently tested on the exact original NOR46 16-source fixture with both GMP backends but are excluded from timing **equally** with the unchanged NOR46 first CPU benchmark.

The performance claim is source-paired: primary median over 16 per-source ratios `NOR47 GMP verified trace+root / NOR45 direct execution+root <1.0`; secondary `NOR47 GMP / NOR46 pure-Python trace+root <0.50`. Failure of either threshold is an **original negative engineering result** and must be recorded unchanged. Report means and medians by D512/D1024 and R/non-R, full trace bytes, CPU model, Python version, GMP version and Linux process-wide peak RSS. Note the first benchmark receipt includes wall times and therefore will not be byte-identically regenerable on a different runner; archive its **first exact original bytes and SHA** without fabricating deterministic timing.

## Arithmetic integrity and limits

The unchanged NOR46 verifier accepts exactly normalized integral finite-series inputs; `R` must satisfy each exact signed equation `(F*R)_0=1`, `(F*R)_n=0` for every coefficient n. For arrays of length m, choose offset `K=2^maximumCoefficientBitLength` and power-of-two digit base `L>4mK²`. Pack shifted **nonnegative** integer coefficients into L-base integers, multiply the two packed integers once, and subtract the signed offset cross terms at each degree. This is the same carry-free arithmetic identity used and tested in NOR46. GMP performs exact **unbounded-integer** multiplication. It is not finite-field arithmetic and its correctness is not evidence of SHA-256 or Ed25519 security.

Preserve NOR46 resource caps `MAX_COEFFICIENT_BITS=32768` and `MAX_PACKED_BITS=120000000`. Reject unsupported modes, noncanonical coefficients, wrong-length vectors, boolean/float inputs, modified reciprocal coefficients and signed program/session/root tampering. Validate full exact signed convolution against a directly summed independent integer oracle for seeded random coefficient vectors, including **4,097-bit** values and boundary patterns.

The experiment uses a **benchmark-only, single-threaded global function substitution** to compare GMP backends within the unchanged outer NOR46 trace verifier. The override is restored in a `finally` clause and MUST NOT be used as a production concurrent verification API. A separately reviewed injection interface is required if any accelerated backend is deployed.

Pinned external dependency: `gmpy2==2.3.2` (GMP's version string is recorded by first CI). Private keys used by the historical signed-source test are already **public reproducible TEST fixtures**. This benchmark generates no new root authority and no real authenticated semantic truth. Original first failure or success is an engineering performance result, not a new model-level science gate.

The next stage could compare Rust/GMP/NTT full-integer proof kernels, limb-based transparent proofs, or a succinct program-execution argument. Their actual proof soundness and resource claims must be studied separately.
