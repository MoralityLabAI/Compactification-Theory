# NOR-44: historical-only signed session and postcommit challenge results

**Historical engineering result: PASS.** The first hosted [NOR-44 run #44228006426](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44228006426) completed both unit-security and 64-source signed-session jobs successfully. This is a **nonprospective, historical-only** integration of the completed NOR42 paired-world cohort and NOR43 signed coefficient Merkle reference. It created **zero new NOR44 prospective source identities**, and it does not assert a new registered scientific benchmark outcome.

The immutable [first raw session receipt](results/nor44_first_historical_64_sessions.json) contains **121,592 bytes**, SHA-256 `b7f740b6916e651932dbd40ceee443bdd5fb40c4b7a7042830ef53f749bd1f83`, originally produced and uploaded as [artifact 14393468272](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44228006426/artifacts/14393468272). Its [read-only verifier](original_archive_audit.py) pins the original SHA, per-source challenge and attack outcomes, provenance and measured payload lengths.

The 64 historical public-source pairs came from NOR42's SHA-locked original first source fixture, SHA-256 `f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033`. Their original grade SHA-256 is `bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae`; NOR43's preceding signed-root smoke SHA-256 is `f7ff23b0ba47eca3607233fe32b61114247bce02561f15ff1b67360950007145`. Every source retains the original 16-template accepted public RRC25 class, and each paired hidden truth includes a late suffix invisible in observed output prefixes.

## Authenticated session sequence

The [session_protocol.py](session_protocol.py) reference implementation creates an independent verifier-owned context containing a **256-bit nonce**, public source SHA-256, observation degree, future output length, reference provider/key ID, verifier/key ID and an expiration epoch. The proposer first hashes its entire integer-valued future output, binding that hash to this source/session context. The verifier signs and retains the **first proposal receipt**.

An authorized provider later signs a true-target Merkle root that incorporates the original signed receipt SHA and session context. Only then does the verifier issue **16 distinct, verifier-owned coordinates**, using `secrets.SystemRandom` in normal mode. The model reveals the proposed output and must provide valid Merkle proofs at **precisely those requested positions**. The verifier compares the revealed output digest against the original receipt and signs a terminal transcript after checking the proofs.

The verifier refuses out-of-order reference release, premature challenge, second proposal commitment and post-root proposal replacement. A failed opening set makes the session terminal. Strict source, nonce, output length, provider ID, key ID, protocol and epoch binding prevent cross-session/root/receipt replay under the verifier's state and pinned-key assumptions. The terminal result is signed by the independently configured verifier key.

The **historical research smoke explicitly uses test-only public deterministic nonces, challenges and private signing keys**. Its per-source transcript carries `test_only=true`. The unit tests separately exercise the normal runtime path using cryptographic nonce and challenge generation. No random-unpredictability claim applies to the immutable historical receipt, and external key custody and globally durable session state remain open work.

## First 64-pair outcomes

For each original NOR42 source pair, the harness reconstructed the already archived clean and hidden true futures. It ran one honest signed hidden-target reference, a second challenge-substitution negative control and a third **dishonest but authorized provider** control signing the wrong clean vector. These are three test transcripts per source; their observations are correlated and never counted as 192 independent sources.

| First historical smoke measure | Dense input family | Block-regime input family |
| --- | ---: | ---: |
| Original source pairs | 32 | 32 |
| Verifier-selected k16 challenge detects clean/hidden mismatch | **31/32 (96.875%)** | **27/32 (84.375%)** |
| Analytic uniform16 hit expectation, original NOR43/NOR42 supports | 98.5584% | 78.7614% |
| Mean Merkle depth | 10.5 | 10.5 |
| Mean signed verifier proposal receipt, JSON | 832 bytes | 832 bytes |
| Mean signed provider reference envelope, JSON | 988 bytes | 988 bytes |
| Mean signed challenge envelope, JSON | 897.0 bytes | 898.0 bytes |
| Mean signed terminal outcome envelope, JSON | 1,209.0 bytes | 1,210.1 bytes |
| Mean 16 original Merkle openings, JSON | 13,224.9 bytes | 13,083.9 bytes |

The overall fixed test-challenge hidden-world detection is **58/64 (90.625%)**. For comparison, the analytic equal-source uniform16 expectation on these already archived discrepancy supports is **88.6599%**. That difference describes one repeatable public test-challenge draw per source and has no independently hidden entropy premise.

Across the 64 pairs the source-averaged JSON payload for sixteen Merkle openings is **13,154.4 bytes**. The verifier receipt, signed provider reference, signed challenge and terminal transcript together average approximately **3,927.1 additional bytes**, giving a combined payload of about **17,081.5 bytes** per successful session, excluding transport framing, public key provisioning and the original full proposed output. The protocol handles `m=769` and `m=1537`, whose Merkle depths are 10 and 11 respectively. This version uses individual inclusion proofs with repeated sibling hashes; a source-bound Merkle multiproof could reduce payload size in a separately measured protocol.

## Adversarial controls

- **64/64 valid-proof challenge substitutions rejected.** The attacker returned sixteen cryptographically valid openings for unrequested easy coordinates. The session rejected them because their indices disagreed with the verifier's retained challenge set.
- **64/64 dishonest but authorized provider controls accepted cryptographically.** An externally pinned signing key certified the wrong clean vector. All signatures and Merkle inclusions passed and the clean proposal was accepted against the incorrect reference. This is an intentional counterexample to the claim that signature verification establishes semantic ground truth.
- The unsigned/attacker-signed model-root case is rejected against the independently pinned provider verification key. A valid but differently signed target root cannot rebind a prior session receipt.
- Tests reject tampered signature, context or digest, expired epochs, a new proposal after root publication, changed source/degree/length, premature challenge, duplicate coordinates and a challenge containing valid openings at substituted positions.

The model cannot silently replace the first proposal inside one trusted, monotonic session state under the assumed collision resistance of its SHA-256 digest and Ed25519 authenticity of the verifier-signed receipt. **Global replay prevention across different processes or durable storage is not implemented here**. The verifier must keep the first commitment and challenge secret in trusted state and provision keys independently of the proposer.

## Trust and reproducibility limitations

A reference signature proves that a root was authorized by the holder of the pinned provider key. Conditional Merkle binding proves an opened coefficient matches that root. The synthetic experiment's assumption that the authorized provider computed the **actual true target** is separate. A dishonest authorized provider passes cryptographic checks with a false vector, exactly as the negative controls demonstrate.

`NOR44/session-context/v1` (with separate signed receipt, reference, challenge and terminal domains) is a reference research protocol using public signing keys and reproducible public test challenges, with an optional normal-mode CSPRNG path. It does not deploy a verifier service, implement external trusted-program execution proofs, prove SHA-256 collision resistance or Ed25519 unforgeability, or cryptographically authenticate real-world semantic truth. CPU timing from NOR43 measured earlier one-shot root verification; a dedicated NOR44 end-to-end transport and concurrency benchmark should account for proposal hashing, signed receipt storage, provider root creation and the independent challenge flow.

## Formal next step

[NOR43's Lean authenticated-root theorem](https://github.com/patrickdugan/MathCan/blob/main/MathCan/NOR43AuthenticatedRoot.lean) captures the conditional root-binding and honest-publisher premises. NOR44 should add a formal **phase/nonce/proposal binding** statement and an exact challenge-index set lemma, keeping honesty and cryptographic security as explicit hypotheses.

The design for a new prospective source experiment remains [NOR-44 issue #239](https://github.com/MoralityLabAI/Compactification-Theory/issues/239). Historical-only first raw evidence and these measured session sizes are retained for independent audit and code review.

**Formal companion completed:** [MathCan PR #144](https://github.com/patrickdugan/MathCan/pull/144) was merged at [`a1339e9`](https://github.com/patrickdugan/MathCan/commit/a1339e9efcc49401154147e4e05ea9e59db1da91) after [Lean 4.35 kernel CI #44229015531](https://github.com/patrickdugan/MathCan/actions/runs/44229015531) passed. The proof checks session-state transitions, commitment uniqueness under a binding hash, exact challenge-set equality, and rejection iff the original committed proposal differs from the true output at a requested index, **conditional** on an honest signed root, authorized verifier, opening soundness and hash binding. Cryptographic security and actual provider truth remain separate premises.

**Security review:** [SECURITY_REVIEW.md](SECURITY_REVIEW.md) records a fail-closed post-evidence repair: out-of-order calls, invalid proposal digests and expired sessions now invalidate in-memory session state. The original first 121,592-byte raw receipt remains unchanged. The corrected harness independently reproduced the same SHA in CI.
