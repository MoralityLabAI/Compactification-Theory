# NOR-43: Signed source-bound output commitments, first historical-only results

**Historical reference-verifier smoke PASS.** The original [two-stage CI run #44144269593](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44144269593) succeeded, validating the pinned-key Ed25519 reference manifest, domain-separated SHA-256 Merkle inclusion proofs, corrupt opening and context replay rejection, and 64 completed NOR42 paired-world sources. This is a **historical-only cryptographic workflow test**. It creates **zero prospective NOR43 source identities** and supplies no actual trusted target provider.

## Original SHA-locked evidence

| Item | Original value |
| --- | --- |
| Raw historical 64-row JSON | [results/nor43_archived_64_merkle.json](results/nor43_archived_64_merkle.json) |
| Original raw JSON bytes | **110,749** |
| Original raw SHA-256 | `f7ff23b0ba47eca3607233fe32b61114247bce02561f15ff1b67360950007145` |
| Original hosted workflow | [#44144269593](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44144269593) |
| First hosted artifact | [#14355867031](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44144269593/artifacts/14355867031) |
| Archive publisher | [#44144755841](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44144755841) |
| Original NOR42 first public source SHA-256 | `f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033` |
| Original NOR42 first graded SHA-256 | `bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae` |

The committed first raw JSON is checked by [original_archive_audit.py](original_archive_audit.py), with the exact original NOR42 source identities, per-source rational uniform16 hit probability, signed provider-key fingerprint and 16 selected audit coordinates. All 64 source records trace to previously registered and independently replayed NOR42 first results. The root and signature test uses a **public deterministic Ed25519 private-key fixture**, exclusively for reproducible proof and rejection checks.

## First historical smoke: 16 authenticated sparse openings

The source-public model first commits to its clean proposed full-output SHA. A distinct synthetic reference-provider role then signs the **hidden true-output vector root**. The verifier pins the expected provider public key externally and authenticates the source-bound signed manifest before checking 16 individual Merkle inclusion paths. A hidden-world discrepancy is detected exactly when any of the 16 checked positions intersects the clean-versus-hidden nonzero difference support.

| Archived input family | Source pairs | Exact uniform16 expected hit | One fixed challenge sample: detected | Mean serialized 16 openings |
| --- | ---: | ---: | ---: | ---: |
| Original dense input | 32 | **98.5584%** | **32/32 (100%)** | **13,181.5 bytes** |
| Block-regime input | 32 | **78.7614%** | **25/32 (78.125%)** | **12,931.3 bytes** |
| **Both input families** | **64** | **88.6599%** | **57/64 (89.0625%)** | **13,056.4 bytes** |

The per-source **exact** hit probability comes from `1−C(m−s,16)/C(m,16)` for the fixed hidden difference support size s and total output length m. The 57/64 observed count is one fixed publicly seeded research draw per source, so it measures a particular reproducible challenge sample. It is **not a fresh statistical study** or evidence of secret audit entropy.

The signed root's JSON manifest-plus-signature envelope was **569 bytes** for every source (including 64 signature bytes and one 32-byte binary Merkle root represented as 64 hex characters). Each source used 16 independently serialized Merkle paths, with mean depth **10.5** (future output lengths 769 and 1537, yielding path lengths 10 and 11). The signed envelope and the opening array together average **13,625.4 JSON bytes per 16-query audit**, excluding request transport and independent authorized key provisioning. Shared tree siblings are duplicated because the prototype uses separate per-leaf proofs. Batch multiproofs and actual verification-time distribution remain future engineering work.

## Post-evidence challenge-binding repair

The original historical smoke's trusted test harness returned exactly the sixteen requested openings. A subsequent independent review found that the first verifier API accepted *any* set of sixteen distinct valid openings under the authorized root without comparing their indices to the verifier's own independent challenge. A malicious prover could therefore substitute easy-index openings and evade a challenge that included a true discrepancy. The [security audit](SECURITY_REVIEW.md) documents this defect and its chronology.

The corrected [independent verifier](verifier_reference.py) requires keyword-only `challenged_indices` supplied from trusted verifier state and checks complete returned-set equality before accepting any opening. Adversarial regression tests pass a valid root with sixteen individually valid Merkle paths at the wrong coordinates and confirm that the verifier rejects the substitution. The original 110,749-byte first receipt and its 57/64 detection count remain byte-identical. The full historical smoke was rerun under the hardened API, and its original SHA is reproducible.

A separate negative control constructs an authorized signature over an *incorrect* target vector. Both signature and Merkle inclusion verify, while the inferred clean proposal remains wrong under a different hidden true world. This confirms the distinction between an authenticated reference **commitment** and externally established reference **truth**.

The [formal companion MathCan PR #143](https://github.com/patrickdugan/MathCan/pull/143) was merged at [`0d3fe32`](https://github.com/patrickdugan/MathCan/commit/0d3fe3208c3f3263a2d23de11dd623c5f1baa946). Lean proves exact sparse-hit equivalence under explicit root honesty, hash binding and opening soundness premises, and supplies the self-committed wrong-root counterexample. All theorems are kernel checked with no additional axioms. No cryptographic SHA-256 or Ed25519 security proof is claimed.

## Verified authorization boundary

The original [signed_witness.py](signed_witness.py) producer commits each exact signed integer coefficient with index and output length, using explicit domain-separated hashes for leaves, internal nodes and index-specific padded leaves. The [independent verifier_reference.py](verifier_reference.py) independently checks:

- The authorized reference provider's pinned public key and Ed25519 signature over a canonical manifest binding protocol version, provider ID, public source SHA-256, observed degree, output length and tree root.
- Every opening's coordinate index, canonical decimal integer encoding, Merkle path depth and sibling hashes. Tampered coefficient, sibling path, index, source identity, signed length, provider identity, version or signature is rejected.
- Duplicate or malformed k16 challenge-index lists, including opened values outside the authenticated output length.

**64/64 model-signed wrong-world self-roots were rejected** when the verifier pinned the intended reference-provider public key. The insecure arbitrary-public-key control accepted the model's internally consistent but wrong clean continuation, demonstrating why a signature alone does not establish external authorization.

An authorized provider could itself sign incorrect data, however. The reference-verification claim therefore requires **two external premises**: the public key genuinely belongs to an authorized independent provider, and that provider correctly computes the true target output. SHA-256 collision/second-preimage resistance and Ed25519 unforgeability are computational assumptions. Neither the Lean source-program argument nor the cryptographic smoke establishes these assumptions or deploys an authoritative true-future oracle.

## Information timing and interpretation

For both NOR42 paired true worlds, the public observation prefix through D and the full query input are identical. A model-only predictor therefore issues the same continuation for either world. NOR43 demonstrates the additional information available **if an independent reference provider computes and signs the actual hidden future** and the verifier retrieves its root after the model has committed its proposed continuation.

The reference Merkle opening checks the equality of a proposed coefficient to the authenticated *committed* coefficient. Since the committed vector's semantic correctness is an assumption about an external provider, authentication of the Merkle path has a narrower guarantee than proof of ground truth. A self-referential root commits only to the model's chosen numbers; it can verify perfectly while the model is wrong.

A domain-separated proposal hash is computed before true-root construction in the smoke's test chronology. The historical sample has no real-world hidden-world distribution, secret audit random draws or secure signing service. The deterministic test private key is publicly reproducible and **must never be deployed**.

## Reproducibility and next work

The first historical JSON and test-only source/grade provenance are permanently SHA-pinned. The archive and full 64-source smoke CI passed. Production-quality work needs external key registration and revocation, a secured signing environment, replay-protected signed sessions or epochs, an auditable provider truth computation, a challenge-order/commitment protocol and actual latency and multiproof measurements. Such work belongs in a new prospective NOR43 registration with source and threat-model gates frozen before generating new data.

This first historical smoke is complete under the scope of [NOR-43 design issue #237](https://github.com/MoralityLabAI/Compactification-Theory/issues/237). It remains conditional on the synthetic reference provider and the unchanged prior NOR42 paired sources.
