# NOR-45: conditional program-execution trust boundary and historical evidence review

The first [NOR-45 historical 64-pair run #44232121765](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44232121765) passed its source-independent unit tests and already archived NOR42 source-pair recomputation. Its first 112,573-byte raw receipt was frozen at SHA-256 `c4b1197b87b43c4cb4e32e0c2708c4a070742244d7cc05639e22353af78d5536` before this review and is preserved **unchanged** at [results/nor45_first_archived_64_programs.json](results/nor45_first_archived_64_programs.json). New tests and code hardening do not revise those first results.

## What the new check establishes

NOR43 established source-bound Ed25519 and Merkle proof verification for output coefficients. NOR44 established a signed source/session-bound proposal receipt, reference certificate and exact challenge order. NOR45 adds a separately pinned **program-authority Ed25519 key**, distinct from the reference-output provider and verifier keys. The signed program certificate includes source ID, full query vector hash, D, m, NOR44 session nonce and first proposal receipt hash, provider/key ID, execution semantics version and an exact RRC25/hidden-suffix program.

The separate verifier authenticates the program statement and both NOR44 signatures, independently executes the complete finite integral formal series using its own I/N/R/S implementation, and rebuilds the **entire** NOR43 Merkle root with separate leaf/pad/internal-node framing. It refuses roots that differ from either the program authority's signed expected root or the source/session-bound provider root. Then the NOR44 verifier checks the original precommitted model proposal using sixteen precise challenge positions and the signed provider's inclusion proofs.

This is a proof of **consistency with the authorized program** conditional on signature security, source/query/session binding, correct independent interpreter, Merkle binding and trusted key provisioning. If the signer authorizes a different program from the one that describes actual external reality, this mechanism cannot recognize the semantic mistake.

## Threat-model negative controls

| Actor/failure | Original historical test outcome | Meaning |
| --- | --- | --- |
| Honest independent program authority and honest output provider | 64/64 hidden outputs and source-bound roots recomputed exactly | Execution fidelity relative to authorized program |
| Dishonest **output** provider signs wrong clean vector, while program authority certifies hidden program | **64/64** rejected | Root authentication alone no longer suffices if the certified program disagrees |
| Model owns a valid signature on its own clean-program certificate | **64/64** rejected by pinned authority key | Proposer cannot select its own authority |
| Dishonest but correctly pinned **program authority** signs clean program, output provider faithfully executes it | **64/64** accepted cryptographically while wrong about designated hidden truth | Authorization plus execution is separate from semantic truth |
| Public deterministic k16 test challenges against hidden output | 60/64 difference supports hit | Research fixture, no deployed entropy guarantee |

The authority is free to sign a wrong program. The certificate producer intentionally has no private mechanism to infer whether a given signed program is true; the verifier can check only its exact semantics. An output provider root may be signed by a correctly pinned key even when the signer is dishonest; independent recomputation rejects it if its output root disagrees with the program certificate. When program authority *itself* is dishonest, both layers can agree on a false world.

## Signed program and input commitments

The program certificate is bound to the NOR44 verifier's **first** signed proposal receipt digest and source/nonce/session context. Its full public query vector hash commits to all signed coefficients and their original indices. The program declares the RRC25 base transform, delay K, spacing Δ, nonzero amplitude and two suffix operations, and optionally a NOR42 late suffix with β≠0 at D+32. The independent interpreter enforces the original allowed operations `I,N,R,S`, delays `64,80,96,112`, spacings `24,32`, amplitudes `−2,−1,+1,+2`, normalized input F₀=1 and future length m=3D/2+1.

All public query and program encoding is strict in the independent verifier. Source SHA itself is assumed to have been independently bound to the original public inputs/demonstrations; the NOR45 verifier's program statement authenticates the exact **provided source identifier**. It cannot establish that the external source was authentic if the verifier itself accepts a forged or untrusted original public source identity.

## Post-first-evidence code correction

After the first source/grade receipt was SHA-locked, an audit of the historical-smoke helper found that `spec_from_public_proposal` validated every source's native spec using observation degree D512 because it derived the parameter from the header delay. Both D512 and D1024 fall within the same allowed RRC25 syntax, so this did **not change a generated program or any output** in the original run. The helper now receives `row["degree"]` explicitly, matching the source context that the independent verifier already checked. This is a correctness-hardening change recorded **after** the first run. The separate original-hash CI reruns the 64-pair smoke and requires the generated JSON to match the original raw SHA byte-for-byte.

## External assumptions and open questions

The reference code uses **publicly known deterministic Ed25519 test private keys** for program authority, output provider and verifier. NOR44 historical sessions use public reproducible nonce/challenge seeds. Normal NOR44 sessions have a CSPRNG path, but this test cannot assert online secrecy, global cross-process anti-equivocation or secure key provisioning. Original tests do not benchmark true-output computation cost, network transport, signature key rotation/revocation, memory limits or authenticated current time. Full independent RRC25 execution runs on the trusted verifier CPU and can require quadratic convolution for reciprocal transforms. It has not been replaced by any succinct proof, zkVM or actual on-chain verifier.

A future proof-of-execution experiment could replace full recomputation with a transparent certificate tied to an independently authorized program hash, query-input hash, source/session and signed target output root. A proof that *an authorized program was executed* remains conditional on the authority's choice of program. For real-world semantic truth, the authority must be accountable for how it determined the correct program or supported by independent external observations.
