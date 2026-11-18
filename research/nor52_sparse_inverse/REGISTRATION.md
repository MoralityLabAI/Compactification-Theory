# NOR-52 preregistration: exact sparse-support inversion and GMP phase accounting

**Frozen November 18, 2026 before NOR52 public input generation and timing.** The completed NOR51 original first 48-source engineering verdict was FAIL (2/5), first raw CPU SHA256 a0591f604ff0b2dfc1fa550ff87d22f62f9527b4ee96579674dee83618d7c9ea. All previous NOR50/51 raw evidence, criteria and verdicts remain immutable. NOR52 measures a new exact algorithm on new public inputs.

## Algebra and threat boundary

For normalized integer formal series F(x)=H(x^s), H(0)=1, uniqueness of a unit inverse in the truncated integer formal series ring gives F(x)^(-1)=H^(-1)(x^s) mod x^m. The reduced length L=floor((m-1)/s)+1. For m3073 and s4, L769. Verify the *complete* untrusted full-m witness: canonical builtin signed integer coefficients, matching length and leading unit, source and witness zero at every index not divisible by s, original full-coefficient and full-packed-integer resource caps. Then shorten both series to L, verify that the exact GMP signed no-carry product equals (1,0,...,0), return the **unmodified** full witness, and rebuild the original independent full-m NOR45-format Merkle root. A supplied stride is checked against every original source and witness coefficient. No global verifier mutation. Dense s1 remains valid and is benchmarked as overhead control.

The program is an **algebra-only single-R kernel**, outside NOR45 signed native-program grammar for m3073, and is not a deployed authenticated provider verifier, ZK or succinct proof. Keep original NOR49/50 session, signed-program/provider tests, and NOR51 exact tamper tests as pre-data controls. Input hashes establish research source identity, not authority honesty.

## First source-only 60-case panel

Use precisely two replicate blocks (0,1), three full lengths m=(769,1537,3073), five profiles s1-b2, s4-b2, s8-b2, s4-b8, s4-b16 (name means stride and coefficient magnitude bits), and two families signed_random and alternating. The 2×3×5×2 grid yields **60 source IDs**, comprising 48 sparse (stride 4/8) and 12 dense controls (stride 1). For each F, F[0]=1. At positive indices divisible by stride, choose independent seeded integer magnitude uniformly in [2^(bits-1),2^bits-1]. Signed_random chooses an independent fair sign for each active coefficient; alternating sets the sign to (-1)^(i/stride). All off-support source coefficients are exactly zero. The seed is random.Random(int.from_bytes(SHA256("NOR52/SOURCE/v1|rep|m|profile|family"),"big")); iterate rep, length, profile, family. The source hash is SHA256 over canonical JSON containing only the public grid coordinates and full input coefficient decimal strings. The *public-only* fixture has no inverse, private future, witness, Merkle root, measured time or GMP phase fields.

Register and upload the exact fixture SHA and 60 source identities in an **upstream source-only CI job before any first timing**. Check SHA-disjointness against all 48 first NOR51 source IDs (original NOR51 public source raw SHA 0b0de5549de1e1815a40d56302cb01ca104e7ac8141f161bc8cab0b6501d2c93) using its immutable original archived public-only fixture. New domain prevents replay of NOR50/51 source IDs. No retuning or replacement of identities after first timing. Resource failure is part of first original FAIL.

## Four registered five-repeat interleaved arms

Pin Ubuntu 24.04, CPython 3.12, gmpy2==2.3.2, cryptography==46.0.4 and OMP_NUM_THREADS=1. For each of 60 sources, interleave exactly five source/repetition-seeded runs per arm. Each arm independently computes the **same full exact output vector and full original Merkle root**:

1. native_full_R_root: NOR51 native_single_inverse on the full source, once, plus full root.
2. native_compressed_R_root: independently validate original source support, compress, compute one exact native unary inverse of length L, reexpand all zero coefficients to full m, plus full root.
3. gmp_full_verify_root: check a precomputed untrusted full inverse through unchanged NOR47 full_gmp exact signed convolution, then full root.
4. gmp_compressed_verify_root: validate original full source and untrusted witness type, lengths, unit, original full no-carry cap and off-support zeros; compressed exact GMP identity; return the **same full witness**, plus full root.

Generate the reference full inverse via original quadratic NOR45 **once outside all timed arms**, recording source-specific production time and canonical witness JSON size/serialization time. The source's public registered stride is validated by the compressed verifier, never inferred from witness or timing. Record five samples, source medians, output/root parity, source IDs, expanded output size, maximum bits and full/compressed packing bounds. Shared signing/session/network costs are excluded equally; existing signed controls run separately.

## Separately instrumented original GMP phase ledger

Outside every timed arm, execute one exact full NOR47-style convolution with independently recorded nanosecond durations for: input type/size parameter and bound checks; GMP packing of both inputs; GMP integer multiplication; low-product digit extraction and right shifts; signed cross-term correction. Check phase-produced exact output == original full_gmp(F,R) == (1,0,...,0). Preserve all five phase times, complete width/packed bits and total phase sum per source. This diagnostic is untimed relative to registered arms and cannot alter any first performance gate. Log but do not presume which phase caused NOR51's precision regression.

## The exact five frozen first engineering gates

1. On the **16 sparse m3073** cases (4 sparse profiles × 2 reps × 2 signs), the median of 16 sourcewise native_compressed/native_full runtime ratios is **strictly below 0.50×**.
2. On those same 16, the median of sourcewise gmp_compressed/gmp_full runtime ratios is **strictly below 0.50×**.
3. On all **12 dense s1** cases (3 lengths × 2 reps × 2 signs), both medians of sourcewise native_compressed/native_full and gmp_compressed/gmp_full are **strictly below 1.10×**. Both must hold.
4. All 60 full exact inverse vectors and independently computed complete Merkle roots match across arms; predata controls reject off-support source or witness mutation, forged stride, wrong length/unit/integer/sign, invalid witness and resource cap violations; unchanged signed-native predata controls pass.
5. Original upstream public-only 60-source SHA freeze precedes first CPU, source IDs distinct and NOR51-disjoint, all full and compressed caps hold, NOR46 globals remain untouched, all five interleaved repetitions recorded, independent untimed five-phase GMP ledger reconstructs the correct exact product, and immutable first CPU/source raw evidence plus original gates are archived with an independent SHA audit.

A registered first engineering PASS requires **all five gates**; any original FAIL remains byte-for-byte. A source exception or out-of-cap value is recorded as first FAIL, without changing the panel. Report by-length, stride/bitwidth, sign family and dense/sparse strata, 10-block (replicate×profile) descriptive bootstrap intervals and sums of source-median runtimes as **secondary** diagnostics. Neither secondary statistics nor phase timing can replace the five gates. All first timing must follow this registration and the source-only SHA freeze.

The finite-series proof follows by substituting y=x^s into H(y)G(y)=1 mod y^L: H(x^s)G(x^s)=1 mod x^(sL), with sL>=m, and uniqueness of the inverse modulo x^m. Independent small exact exhaustive Python controls verify the finite identity. A separately checked Lean kernel proof remains future work rather than a claim based on runtime tests.
