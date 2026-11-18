# NOR-43 post-evidence security audit: verifier-owned challenge binding

**Status: corrected and tested after the first historical-only receipt was frozen.** The original 64-source [NOR43 historical first CI #44144269593](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44144269593) produced the immutable 110,749-byte JSON, SHA-256 `f7ff23b0ba47eca3607233fe32b61114247bce02561f15ff1b67360950007145`. Its scoring harness generated the challenged 16 index positions and provided exactly those Merkle openings to the verifier, so its 57/64 observed support hits remain a valid description of **that trusted harness execution**.

## Found during independent security review

The first verifier API verified each individually returned opening against the source-bound signed root, and required 16 distinct indices, but lacked an **independent check that the 16 returned indices were the specific positions previously selected by the verifier**. A malicious prover with knowledge of the full committed target could submit perfectly valid openings at any 16 easy coordinates, avoiding a known discrepancy even when the verifier's genuine challenge included a disagreeing coordinate. Per-opening Merkle validity cannot enforce challenge participation on its own.

This is a **material challenge-binding defect in the initial API**. The original smoke itself used exact requested positions because the trusted test harness assembled the openings directly. That fact does not extend to untrusted prover-supplied opening lists.

## Post-original repair

The independent [verifier_reference.py](verifier_reference.py) now requires keyword-only `challenged_indices` from verifier-owned state. It checks the expected k, integer/range validity and uniqueness of the challenge indices, rejects duplicate or malformed returned openings, and requires **exact set equality** of returned and requested indices before any per-opening comparison.

The [test_signed_witness.py](test_signed_witness.py) suite includes the following adversarial counterexample: an honest provider signs a vector with a hidden suffix beginning at index 544; the independent verifier requests the set `{544,0,…,14}` containing a genuine model-vs-truth discrepancy; a dishonest prover responds with sixteen valid, signed-root-consistent openings for the easy prefix indices `{0,…,15}`. Each individual Merkle path verifies. The corrected verifier rejects the substitution on challenge-index mismatch. A correctly returned set containing 544 detects the hidden discrepancy. The full historical smoke now passes its independently generated challenge list to the verifier, and regenerates **exactly the same first original raw JSON SHA**, demonstrating no change to original reference data or science measurements.

The secure API's expected indices **must** come from independent trusted verifier state, never from the prover-supplied payload. The example tests allow proof order permutations, because the set of checked indices is authoritative; duplicate challenge indices or unchallenged openings fail.

## Important remaining boundary: honest provider

Another negative control signs a **false clean vector with the pinned authorized reference private key** when the actual target is the hidden future. The signature and all Merkle openings verify, and the checker accepts the clean proposal despite its semantic error. This is correct cryptographic behavior under a dishonest provider: the system authenticates what that key signed and cannot infer truth from a signature.

The reproducible test key is publicly available and can be used to sign any vector. Its pinning exercise demonstrates API mechanics only. An actual deployment would require an independently established authorization registry, securely generated and held signing keys, key revocation and an auditable output-computation path. Even an authorized secure key guarantees only *provenance of a commitment*, conditional on signature security, until the provider's correctness is established separately.

## Additional issues deferred to new registration

- **Temporal binding:** the original smoke computes an output proposal hash before constructing the provider root, but lacks a network or ledger protocol proving that the model committed before seeing the reference root. A future session-bound protocol needs a verifier-issued unpredictable nonce, submission receipt, context-bound proposal commitment and an ordering rule.
- **Source-binding of the proposal:** the first proposed-output hash binds length and values but omits source SHA, observation degree and session ID. The reference manifest includes source, degree and length; a future proposal commitment should include them too to defend against cross-context reuse.
- **Private challenge entropy:** original `CHALLENGE_STREAM` and keys are public deterministic fixtures. Real hidden audit indices must come from a secret or independently unpredictable draw committed before opening disclosure. The experiment's conditional uniform k16 probability assumes this.
- **Adversarial provider and key setup:** public test keys and a simulated honest provider do not test actual authorization, key compromise, root publication channels, truth discovery or authorized-provider dishonesty at scale.
- **Proof batching and resources:** each of 16 paths is serialized independently; no sibling deduplication or batched multiproof optimization was attempted. Full native output computation and certificate issuance costs are outside current benchmark scope.

The baseline historical report remains unchanged as an immutable source. This security review is an **explicit post-evidence implementation hardening**, and must not be counted as part of the original NOR43 first-run preregistration, the earlier NOR42 5/5 science outcome, or a new prospective NOR43 evaluation.
