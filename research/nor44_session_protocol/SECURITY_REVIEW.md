# NOR-44 session protocol security review and evidence chronology

**Scope:** historical-only Python reference state machine, already sealed original NOR42 source pairs and original NOR43 signed-coefficient Merkle commitments. This review is a correctness and threat-model audit, **not** an independent proof of Ed25519/SHA-256 security or a deployed reference-provider guarantee. The [original first historical smoke #44228006426](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44228006426) passed and emitted 121,592 bytes SHA256 `b7f740b6916e651932dbd40ceee443bdd5fb40c4b7a7042830ef53f749bd1f83`; its byte-identical JSON is retained unchanged in [results/nor44_first_historical_64_sessions.json](results/nor44_first_historical_64_sessions.json).

## Threat model and premises

The verifier, with independently provisioned provider Ed25519 public key, owns session context and the nonce. A proposer observes the common public input and visible target-output prefixes and cannot see the externally authorized hidden true future or **unpredictable** challenge coordinates before freezing its proposal. The provider computes the correct true target outside the model and signs the exact source/nonce/epoch-bound root. The verifier trusts its own first receipt, challenge state, locally supplied epoch and separately pinned verifier key.

The protocol authenticates *which key signed the root* and *which indices were opened under it*, assuming signature unforgeability and hash binding. **Honest provider ground truth, key provenance, secure storage of receipts and nonces, true challenge secrecy and global session uniqueness are external premises.** A malicious authorized provider can sign a false target that passes cryptographic checks, as the first smoke explicitly demonstrates for all 64 pairs.

Normal mode issues a fresh 32-byte nonce using `secrets.token_bytes(32)` and 16 distinct challenge positions via `secrets.SystemRandom().sample`. All historical CI receipts use **public deterministic** test challenges, session nonces and test signing keys. Signed context marks `test_only=true`. This test path proves deterministic state/order and signature-enforcement behavior, **not online unpredictability**.

## Bound-message sequence

| Stage | Signing/authentication | Immutable binding | Security consequence |
| --- | --- | --- | --- |
| Session creation | Verifier key configured; provider key independently pinned | 256-bit nonce, source SHA, observed D, m, provider/key ID, verifier/key ID, expiry, version, test mode | Cross-source/session semantics |
| Proposal precommit | Verifier signs first digest receipt | Full proposed vector hash plus session context | Later output reveal must be the first proposed vector |
| Provider reference | Authorized provider verifies precommit receipt, then signs the root | Source/session context, first receipt SHA, proposal digest, Merkle root | Old reference and wrong-key self-root cannot be substituted |
| Challenge | Verifier signs requested indices and retains the exact set | 16 distinct positions, provider root envelope SHA, session | Prover cannot substitute valid easier coordinates |
| Terminal result | Verifier signs after checking digest, root and all 16 openings | First receipt, signed provider root, exact challenge, reject Boolean | Auditable single-use result under verifier honesty |

The signed terminal result is not itself a distributed ledger or durable receipt store. The reference verifier's Python memory state cannot enforce non-equivocation across independent processes. The verifier must retain original receipts and protect its signing key outside this process for deployment.

## Post-first-smoke fail-closed correction

**Chronology:** the first hosted 64-pair smoke was frozen and archived with the initial verifier implementation. Security follow-up after that first experiment noted that an invalid out-of-order method call raised `SessionError` but left the in-memory phase unchanged. A caller could then retry, which contradicted the documented terminal-failure design. The failure did not affect original 64 recorded successful/negative-control traces because none depended on invalid-stage retries. It nevertheless weakened the one-use interface.

The revised `VerifierSession._require` sets `phase='failed'` on an invalid stage or expired epoch. A malformed first proposal digest also invalidates the session. Tests now require that a second `precommit` attempt kills its session; a separate fresh session tests a changed post-root reveal, which also fails terminally. Subsequent calls remain rejected. **The original first 121,592-byte JSON and SHA are unchanged**. Hosted CI reruns both unit tests and 64 historical-source transcripts; identical raw bytes are required by the permanent `original_archive_audit.py` gate.

This is a protocol repair on a historical engineering gym, separately reported from any prospectively registered NOR44 scientific experiment. A new prospective source/cohort evaluation would require a first registration under new seeds and immutable registered thresholds.

## Negative controls and limitations

The first 64-pair run checked one valid provider-root session, one substitution attack on sixteen individually valid Merkle proofs that failed exact challenge membership, and one dishonest but authorized provider signing the wrong clean vector. All 64 challenge-substitution attempts were rejected, while all 64 dishonest reference vectors were **accepted cryptographically** and generated no rejection when compared with themselves. The latter is an intentional trust-boundary counterexample, not a failure of signature verification.

Synthetic unit tests also exercise signature and source-context tampering, root replays across nonces and output lengths, wrong reference key, premature challenge, invalid proposal reveal, expired epoch and duplicate coordinates. They exercise a normal CSPRNG path separately and do not draw prospective NOR44 source identities.

**Open requirements for production:** a mutually authenticated proposer identity and independent verifier receipt service; actual anti-equivocation durability; authenticated current epoch and nonce-collision handling; source-linked real-world true-output provider or a correctly scoped proof of execution of a separately authenticated true program; audited cryptographic implementation and key rotation; unpredictable challenge issuance held secret until both commitments; timeout and resource cap controls. A proof of faithful execution of a self-committed program still does not establish that the program describes real-world truth.

## Formal scope

The [MathCan NOR-44 proof](https://github.com/patrickdugan/MathCan/blob/main/MathCan/NOR44SessionOrdering.lean) checks the conditional phase ordering, first proposal binding, exact challenge-set equality and resulting equivalence between rejection and original proposal's true support intersection. Cryptographic assumptions, provider honesty, key authorization and entropy remain explicit hypotheses.
