# NOR-43 historical-only signed coefficient commitment reference

This repository folder implements the **historical-only** prototype proposed in [NOR-43 issue #237](https://github.com/MoralityLabAI/Compactification-Theory/issues/237). It uses previously committed original NOR42 source-pair and grade bytes, without generating NOR43 source identities, new latent true worlds or prospective scored outcomes.

## Exact trust boundary

An externally authorized *reference publisher* signs a source-bound Merkle root of all m future-output coefficients. The model submits its public-prefix-derived proposed output **before that trusted root is made available to it**. The verifier pins the publisher's public key from an independently controlled channel, verifies the signed manifest (protocol version, source SHA, observation degree, output length m and root), and checks the k16 selected coefficient openings against the source-bound root.

Merkle inclusion proves consistency between a claimed coefficient and the signed root, **conditional on SHA-256 collision/second-preimage resistance**. Ed25519 signature verification establishes that the manifest was signed by the holder of the pinned verification key, **conditional on signature unforgeability and correct key provisioning**. Neither step proves that an authorized provider calculated the *true* coefficients faithfully. Honest provider data and key registration remain external premises.

The private key used in this prototype is a public reproducible TEST-ONLY key derived from a literal SHA-256 label; it cannot provide security against an observer. Tests assert the verification logic under a conceptual honest provider and an independently pinned expected public key. Production deployments require a separately generated private key, secure signing environment and trusted key distribution, which this research prototype deliberately does not implement.

For an m-coefficient integral formal-series future, a leaf commits to its coefficient index, m and a minimally encoded signed integer value. Internal nodes and padding use separate versioned SHA-256 domains. The tree pads to a power of two with canonical, index-specific empty leaves. An Ed25519 signature covers a deterministic JSON manifest containing the full public source SHA, observed degree, future coefficient count m, root hash, provider identifier and protocol name. Every opening includes an exact index, minimally framed integer and full sibling path.

The signed root is delivered to the verifier **after** the model's submitted proposal hash is frozen. A public root issued before proposal commitment can leak information about which paired world is true, undermining the original NOR42 common-prefix-only limitation.

## Controls

The test suite covers a valid honest reference commitment and k16 proofs; corrupted coefficient, sibling path and leaf index; incorrect source/degree/length/provider/version; signature substitution and model-owned self-signed roots; and the witness-free acceptor that accepts the clean proposal under a self-consistent model-supplied root. The verifier must reject a self-owned root signed with an attacker key when pinned to the actual reference provider.

The original NOR42 first source-pair fixture is 9,059,929 bytes SHA256 `f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033`. The first grade is 201,956 bytes SHA256 `bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae`; permanent first ZIP is 2,799,867 bytes SHA256 `e60e97fd933f660c2274cb4a473ede7a98968d24eaf0dd3ae45ace1e1176cd0d`. Original prospective NOR42 had 64 public pairs, two possible worlds each, all five registered science gates PASS and separate full replay green.

## Experimental interpretation

The historical-only smoke reconstructs the already graded NOR42 clean/hidden target vector from the archived fixtures, signs each hypothetical true world from a pinned test publisher, and verifies 16 sparse openings using the independent reference verifier. When its 16 positions hit the difference support, the clean public-model proposal is detected against a hidden-world reference; if not, the authenticated sparse readout accepts that limited transcript. The exact full-output uniform16 probability for any fixed hidden difference support remains `1-C(m-|S|,16)/C(m,16)`.

The smoke measures proof payload bytes, signature/root bytes and verification time on a fixed, declared CPU environment. It is not evidence that SHA-256 or Ed25519 are formally secure, that an actual provider will be honest, or that the source worlds correspond to deployed semantics. A separate NOR43 prospective registration would be required for new sources or model-evaluation claims.
