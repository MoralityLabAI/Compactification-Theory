# NOR-46: transparent exact-integer execution trace, historical-only prototype

[Research design issue #243](https://github.com/MoralityLabAI/Compactification-Theory/issues/243) proposes replacing NOR45's full independent CPU recomputation with a transparent execution proof. This branch prototypes a **deterministic, nonsuccinct proof-carrying exact-integer trace** under the already-authorized NOR45 program/session root. It uses only the original 64 NOR42 SHA-archived public source pairs and historical NOR45 source certificates. **No new NOR46 source identities, model evaluations or prospective scientific results.** Source and original first receipts are immutable.

## What the first construction proves

The prover supplies the exact arrays required by the signed program: one finite unary-transform output for each distinct member of `{base,u,v,hidden_transform}`, plus the complete result vector. Each transform is verified against the independently source-bound query. I/N/S are checked coefficient-by-coefficient. For R, the verifier checks `F_0=1`, `R_0=1` and the **entire exact integer convolution identity** `(F*R)[:m] = (1,0,...,0)` using a one-multiplication carry-free Kronecker substitution, with a formally stated integer size bound that prevents coefficient carry and field wraparound. This determines the unique inverse R coefficients in the ring of truncated normalized integral formal series.

For the native program `P=(base,K,Δ,c,u,v,β,T)`, the verifier checks every signed output coefficient against

`Y_n = B_n + c[n=K] + c U_{n-K-Δ}[n≥K+Δ] + c V_{n-K-2Δ}[n≥K+2Δ] + β T_{n-D-32}[β≠0∧n≥D+32]`.

The verifier independently reconstructs the full NOR43 Merkle root of that validated output and checks it against the **same root** signed by the NOR45 independent program authority and by the separately pinned NOR44 output provider. It verifies the signed model proposal precommit and source/session/query/nonce/key IDs before accepting the execution trace. The source input, program statement, output root and verifier challenge are **not chosen by the trace prover**. All nonzero transformations are checked as signed Python unbounded integers, not merely modulo a field.

This is a **transparent deterministic execution certificate with full trace**, rather than a STARK/zkVM proof, and its witness includes all output coefficients. It makes a narrower, legitimate claim: if the separately authorized program statement and authenticated output root are correctly validated, and all exact integer trace constraints pass, that root commits to the output of the authorized program. It provides no model correctness beyond the authorization authority's program selection and does not establish the underlying cryptographic signature/hash security.

## Carry-free exact convolution

For any two m-component signed integer vectors A and B, let `K=2^k` with `k=max_i bit_length(|A_i|,|B_i|)`, so `0≤A_i+K<2K` and the same for B. Choose a power-of-two digit base `L=2^w` with `L>4mK²`. Encode the two nonnegative vectors as `A'=Σ(A_i+K)L^i`, `B'=Σ(B_i+K)L^i`. Then every coefficient of `A'·B'` is strictly less than L, so no cross-digit carries occur. Extract digit n and subtract

`K·(Σ_{i=0}^n A_i + Σ_{j=0}^n B_j)+(n+1)K²`

to recover the **exact signed convolution coefficient** `Σ_{j=0}^n A_j B_{n-j}`. Checking every n from 0 through m−1 is an exact certificate of inverse-series arithmetic. The signer and verifier remain independently implemented. A malicious trace value fails even if a purported output-root Merkle proof would be internally self-consistent under a prover-selected key.

## Expected complexity and honest limitations

The full certificate is **not succinct**: it carries up to four m-element integer transform arrays plus m outputs. Verification must parse and range-check all integers, check every needed transformation, and rebuild the complete output Merkle tree. The R transform uses one large exact bigint multiplication rather than the original O(m²) Python nested-loop recurrence, but its packed bit size grows roughly as `m * (2·maxbit + log₂m)`. On m=769/1537, packing and unpacking can overwhelm the arithmetic savings. No claimed speedup is made before results from the hosted CPU benchmark. Memory, traceback byte size and wall time must be reported; a slowdown is a valid first engineering result.

This protocol is **not zero-knowledge**. The certificate reveals the full trace and output to the authorized verifier, which already needs access to private target coefficients under NOR45. Its cryptographic security is conditional on NOR43–45's SHA-256 and Ed25519 assumptions and externally pinned keys, while the verifier's exact integer arithmetic is deterministic.

## Threat model, negative controls and output

Test on the already archived NOR42 source pairs and original NOR45 signed statements. The first CPU smoke may use a fixed subset of 16 historical pairs to bound cost before expanding to 64; the selection rule must be recorded before reading timing outcomes. Include R appearances and both future lengths, independent witness reconstruction, valid source-root parity, a one-coordinate malformed trace, wrong inverse coefficient, wrong suffix amplitude, tampered source, nonce and program key, wrong output Merkle root, and a dishonest but signed original-program authority who certifies a semantically false clean future. The latter must **pass faithful authorized execution** and fail comparison against the separate hidden-world ground truth.

All first original smoke receipts (including negative performance) must be archived byte-identically with first SHA, with separate deterministic proof-status fields and separately timed CPU benchmarking. A future transparent STARK/zkVM or a truly low-verification-cost SNARK needs a new proof-system registration, exact signed big-integer range constraints, cryptographic soundness evidence, and a clearly scoped trusted key/authority boundary.
