# NOR-56 prospective registration: signed compact-v2 batch length-table exact verifier

**Frozen 2026-11-18 before any NOR56 new public source generation, witness construction, or CPU timing.** NOR55's immutable first engineering verdict was FAIL (4/5); the original signed compact-v1/JSON verifier median ratio was 0.8514830908862547 against registered <0.85, raw CPU SHA256 `98c45ed555349fcfd860def1975d571d0e84565e6cad976d30941cae3384b10e`. NOR55's original source SHA was `d1e6b2f5f37edc37f4b996b42e10768b47b4c199ed274cf53643be34fddddd59`. All original negative verdicts and archives remain unmodified. NOR56 is a separate registered trial.

## Exact canonical batch-table v2 format

For normalized integer formal series F(x)=H(x^s), full length m, independent public stride s∈{1,4,8}, exact witness R is supported only on multiples of s. Let L=floor((m−1)/s)+1. The v2 payload is:
- The **distinct fixed bytes** `b"N56C2"`, followed by a canonical unsigned LEB128 coefficient count L (minimal, bounded by 65536).
- A **contiguous table of precisely L big-endian 16-bit unsigned entries**, parsed in one `struct.unpack_from(">LH", raw, offset)` call. The high bit records the sign, the low 15 bits record magnitude byte width in [0,4096]. Zero has sign=0, width=0. All nonzero magnitudes must be unsigned minimally encoded big-endian with no leading zero.
- Immediately following the table, concatenated exact integer magnitudes, no separators, no extra bytes and no alternate encodings. The declared widths must sum exactly to the remaining byte count, without exceeding an 8MB payload limit.

Each signed integer is a **builtin Python int**, magnitude <=32768 bits, with canonical positive or negative representation, no negative zero, no nonminimal sign bit or trailing data, no fractional/machine-number conversion. The leading reduced coefficient must equal 1. A decoded reduced witness reconstructs exactly the original **full m vector** by inserting zero coefficients. The original full NOR47 coefficient and packed-bit caps (32768, 120,000,000) are enforced before accepting the exact signed identity, including on s4/s8. The NOR53 fast compressed GMP verifier checks full source/witness support, exact inverse and original full NOR45 Merkle root. Pooled exact output must match NOR55 v1 and decimal JSON vectors.

## Version-bound source/session authentication

Leave the merged NOR55 `envelope.py` and all original evidence unchanged. NOR56 has a distinct explicit signed `compact-v2` format identifier, and a **distinct domain-separated Ed25519 research signature prefix**. Reuse NOR55's hardened header parser/length limits and trusted session context, with independent source SHA, source-vector digest, full m, authenticated public stride, expected original full output root, provider fixture identity/key fingerprint, session sequence, 256-bit verifier-issued nonce, and exact payload byte count/SHA. The v2 verifier compares the signed header against the independently pinned trusted context **before** Ed25519 verification or decode and commits replay state **only on exact algebra and full-root success**. Explicitly reject version downgrade, wrong signer, signer/cross-source/session swapping, forged stride/root, invalid Ed25519 signature, malformed length table, payload hash or count, positive/negative ambiguity, nonce replay or nonmonotone sequence. The deterministic NOR55 research fixture key is public in the repo and **not a production key**. Source/session authority and genuine external provider truth remain separate premises.

## New first source-only 60-case cohort and controls

Use the identical NOR55 2×3×5×2 public panel grammar under **new NOR56/SOURCE/v1** domain: replicate 0/1; full m769,1537,3073; profiles s1-b2,s4-b2,s8-b2,s4-b8,s4-b16 (stride and coefficient magnitude bits); sign families signed_random and alternating. F[0]=1; positive multiples of s have magnitude uniformly sampled in [2^(bits-1),2^bits-1], with separately seeded fair sign or alternating (-1)^(i/s); every other source coefficient is zero. Per cell use `random.Random(int.from_bytes(sha256("NOR56/SOURCE/v1|rep|m|profile|family"),"big"))`, iterate rep, m, profile, family in registered order and build source IDs from the canonical JSON of public grid coordinates and complete decimal coefficient strings. The first source-only fixture contains no inverse, witness, result root, signature, session or timing.

Exclude the **288 first historical SHA source identities** from NOR51 (48), NOR52/53/54/55 (60 each), reading their permanently SHA-anchored original ZIPs. In a separate upstream source-only CI job, freeze raw first public fixture and 60 new source IDs and publish SHA256 **before** any new inverse or witness byte size/timing calculation. No source replacement or threshold tuning if one fails.

## Exactly four timed full-root arms and independent ledgers

On one pinned Ubuntu24.04 CPython3.12 runner, gmpy2==2.3.2 and cryptography==46.0.4, OMP_NUM_THREADS=1, compare exactly five deterministic source/repetition-seeded interleaved whole-operation samples each, on all 60 identities:
1. NOR52 **compressed native inverse + full original root**, computing from original source alone.
2. NOR55 **signed compact-v1 verifier** of a supplied independently calculated witness, including full header/source/session replay, Ed25519 signature, v1 decode, NOR53 exact GMP check and full root.
3. NOR55 **signed full-decimal JSON verifier** with identical validation and full root.
4. NOR56 **signed compact-v2 verifier** with its format/domain binding, batch-table parse, full original witness/limits/identity and full root.

For each source, compute an **independent NOR51 full native reference witness** and original full Merkle root once outside timed arms. Before timing pre-build five unique verifier nonce/sequence signed envelopes for each of the three signed formats; record complete signed wire/payload byte lengths, source-specific witness-production/serialization time and signer encode/signing time separately. Keep independent replay guards per arm and five ordered accepted challenges. Every arm returns the **same complete exact builtin signed integer full vector and independently recomputed full Merkle root**, and check all values/roots before and during timing. No hint or witness data selects a kernel. Input parsing, network/transport and fixture signature generation excluded equally from verifier-only timings; signature verification is inside the authenticated arm. Provide output/wire medians by m, stride/bitwidth and sign family and complete raw five-sample traces.

## Exactly five prospectively frozen first engineering gates

1. **v2 speed over v1:** among all **16 sparse m3073 sources** (4 sparse profiles × two replicates × two signs), median of sourcewise `signed_v2/signed_v1` full verifier/root runtime ratios is strictly **<0.92x**.
2. **v2 versus signed JSON:** same **16 sparse m3073** units, median `signed_v2/signed_json` full verifier/root runtime ratio strictly **<0.82x**.
3. **v2 wire overhead:** across exactly **48 sparse sources**, median sourcewise **complete signed v2 envelope bytes / complete signed v1 envelope bytes <1.05x**, including the full Ed25519 signature and canonical header bytes in *both* amounts.
4. **Exactness and hostile protocol tests:** all **60/60 full exact integer output vectors and independent original NOR45-format Merkle roots** agree across four arms, and source-free adversarial tests reject malformed endianness/lengths, leading-zero magnitudes, negative zero, overcaps, version downgrade/mismatched signed format, wrong key/root/source/session, tampered bytes/signature, off-support witness or source, nonce replay/sequence, and wrong unit. NOR50 signed-native controls and NOR55 original first immutable audit must remain green.
5. **Timing/source/evidence integrity:** new first public-only 60-source SHA frozen upstream of first CPU and 288 old identities excluded; exactly five interleaved samples per arm and original NOR47 full coefficient/packed caps, no NOR46 process-global verifier mutation; first raw CPU/source evidence and **all five Boolean original gates** archived byte-identically with permanent read-only SHA/median audit. One-shot publisher must be retired after archival.

All five required for an original engineering PASS. Preserve a FAIL if any registered threshold/correctness/coverage gate fails; subsequent repair is a labeled follow-up. Never change the historical NOR55 0.851483x FAIL, replace sources, or adjust thresholds after heldout timing.

## Predeclared secondary diagnostics

After the first four-arm original CPU timings, an **untimed diagnostic** independently replays each format with stage timers isolating canonical header/authentication, payload table/JSON decode, original full support/cap checks, GMP verification and original Merkle root. These are descriptive only, cannot attribute the first runtime difference without controls, and cannot change first gates. Also report the sum of source-median whole-arm times and 4,096 ten-block (replicate,profile) bootstrap descriptive uncertainty. Any producer-inclusive or true off-host latency remains outside this study.

**Scope:** synthetic transparent exact integer inverse witnesses, m3073 outside NOR45 signed-native program grammar, public research signing key, no independently honest provider or deployed network. A future Lean kernel proof of sparse-support substitution and no-carry signed exact convolution is separate from Python/CI correctness.
