# NOR-44 historical protocol integration: session-ordered sparse witness

This is an **archived-source-only engineering gym** for [NOR-44 design issue #239](https://github.com/MoralityLabAI/Compactification-Theory/issues/239). It reuses the 64 byte-frozen NOR42 public source pairs and NOR43's signed-coefficient Merkle producer and independent verifier. The original 64-pair source SHA is `f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033`; the NOR42 grade SHA is `bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae`; their first ZIP SHA is `e60e97fd933f660c2274cb4a473ede7a98968d24eaf0dd3ae45ace1e1176cd0d`.

NOR43's original 64-source historical smoke was **110,749 bytes**, SHA-256 `f7ff23b0ba47eca3607233fe32b61114247bce02561f15ff1b67360950007145`, and its first original findings remain unchanged. NOR43 demonstrated valid source-bound Merkle openings under a pinned Ed25519 provider key, but its proposal digest was not bound to a unique session. A later audit required the verifier to check precisely its own challenged indices, closing a substitution vulnerability. The independent [Lean NOR43 conditional root-soundness theorem](https://github.com/patrickdugan/MathCan/blob/main/MathCan/NOR43AuthenticatedRoot.lean) assumes an honest reference root and computational commitment binding.

## Protocol sequence

1. The verifier independently pins the reference-provider ID and Ed25519 public key, and creates a new **256-bit session nonce** with a CSPRNG. The session binds the public source SHA-256, observed degree D, future output length m, provider-key ID, verifier-key ID and a caller-supplied authenticated expiration epoch.
2. Before seeing the true reference root, the model commits to the hash of its entire proposed m-coefficient output, **domain-separated and bound to the session nonce, source SHA, D and m**. The verifier records this digest immutably and signs a proposal receipt using its separately pinned verifier key.
3. A reference provider independently verifies the signed proposal receipt, then signs a session-specific true-output Merkle root that binds **the original proposal receipt hash** and all context fields. The verifier checks this Ed25519 signature under its pinned provider key. The authorized provider may be dishonest; a signature on a false vector supplies no truth guarantee.
4. Only after the two signed commitments are frozen does the verifier sample **16 distinct positions** with a CSPRNG and retain those exact positions as trusted private session state. The prover returns exactly those 16 original Merkle openings.
5. The verifier accepts the proposed output reveal only if its hash equals the first proposal receipt, every returned proof opens under the authorized root, every opening matches the verifier-chosen indices, and the session is within its expiration epoch. The verifier signs a terminal, immutable transcript stating the sparse reject/accept result.

The state machine permits a single transition through `created → proposed → reference_authenticated → challenged → finalized`. Invalid attempts move it to `failed` so the proposer cannot adaptively retry. A signed proposal receipt is a transportable audit record; **durable external receipt storage and provider truth are separate premises**. An in-memory reference implementation cannot by itself enforce interprocess global uniqueness or prevent a compromised verifier signing key from equivocation.

## Entropy and test security boundary

Normal mode uses `secrets.token_bytes(32)` and `secrets.SystemRandom().sample` for session nonce and challenge coordinates. Deterministic injection of nonce or challenge is accepted **only in explicitly marked test mode**, whose signed session context records `test_only=true`. The historical 64-pair run uses fixed reproducible test nonces, public deterministic challenge seeds and the previously published public NOR43 test signing key. Its receipts are audit fixtures, **not evidence that deployed challenges were unpredictable**.

Both NOR43's reference key and the NOR44 verifier key used in tests are deterministically derived from public labels. The resulting public/private keys are for tests only and are unsuitable for production. The reference implementation makes the provider key and verifier key separate caller-supplied arguments, and requires an independently provisioned pinned provider verification key.

## Required attacks and controls

Challenge substitution with 16 individually valid Merkle proofs for easy, unrequested indices; a changed proposal after a valid root reveal; stale proposal receipts or signed roots replayed in a new session; tampered source, D, m, nonce, expiry, provider identity, provider key ID, protocol version or signature; duplicate challenge indices; a self-signed wrong model root using a model-selected key; and a fully valid but deliberately **dishonest authorized provider** that signs a false output vector. The last case should pass cryptographic checks while failing semantic truth, displaying the external trust assumption.

Report actual signed receipt/reference/final transcript bytes, root and k16 opening JSON payloads, and stage-specific verification costs. Keep timing out of immutable first source receipts, with explicitly labeled platform/version in any separate benchmark.

## Archive and formal scope

The first historical smoke uses **only the original NOR42 source fixture and grade archive**; no new NOR44 prospective source identities, hidden target worlds, new registered EVAL trials or scientific PASS/FAIL are generated. Preserve the first original smoke JSON and SHA regardless of its engineering test outcome. A new prospective NOR44 registration must specify nonces, key provisioning, entropy commitments, independent source units and science gates before generating genuinely new data.

A Lean companion should treat source/nonce/session/proposal binding, honest authorized root and correct challenge set as **explicit premises**, and prove that a signed precommitment cannot silently change its proposal and that exact authenticated openings reject iff the committed prediction differs from true output on the challenge set. Signature unforgeability, collision resistance, key custody and honest provider provenance remain assumptions.
