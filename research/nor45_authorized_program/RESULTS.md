# NOR-45: First historical program-authority and faithful output-root audit

**Historical-only engineering PASS** on **64 original NOR42 source pairs**. This is an implementation and trust-boundary study described in [NOR-45 issue #241](https://github.com/MoralityLabAI/Compactification-Theory/issues/241). It creates **zero prospective NOR45 source identities**, tests no newly registered model performance hypotheses, and does not claim an externally verified real-world ground truth.

The [first hosted NOR45 run #44232121765](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44232121765) completed both independent program/root unit tests and full 64-source historical reference sessions successfully. The first original evidence JSON is **112,573 bytes**, SHA-256 `c4b1197b87b43c4cb4e32e0c2708c4a070742244d7cc05639e22353af78d5536`, originally uploaded as [artifact 14394903606](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44232121765/artifacts/14394903606) and now preserved byte-for-byte in [results/nor45_first_archived_64_programs.json](results/nor45_first_archived_64_programs.json). The [original_archive_audit.py](original_archive_audit.py) rechecks its first SHA, all 64 original source IDs, fixed k16 challenge sets, independent authority controls and exact finite-support detection fractions.

## Information and chronological boundary

The model first freezes its source/session-bound hash of its proposed full output under a NOR44 verifier-signed receipt. Later, a **separate program authority** signs a source-bound executable program specification and its expected Merkle root. The statement binds the full query vector hash, original public source SHA, D, m, unique session nonce, first proposal receipt digest, provider key ID, program authority key ID, execution-semantics version, full original RRC25 six-parameter program and optional hidden late suffix `βX^(D+32)T(F)`.

The reference output provider independently signs its source/session-bound Merkle root. The NOR45 verifier checks the provider and program-authority signatures against **distinct pinned public keys**, evaluates the program using a **separate exact integer-series implementation**, independently rebuilds the Merkle root, and insists it match both signed claims. Only after that does the original NOR44 verifier issue an exact k16 read challenge and check the requested Merkle openings. The model cannot replace its original proposal in that one-use in-memory session.

The program authority's certification of the *correct program* is an **external trust premise**. SHA-256 binding, Ed25519 signature security, authorized key provisioning and true challenge unpredictability also remain assumptions. All original keys/nonces/challenges in this historical receipt are publicly reproducible TEST fixtures; none represent deployed credentials or hidden entropy.

## Original 64-source control results

| Historical original control | Observed |
| --- | ---: |
| Trusted test authority's hidden program independently recomputed exactly to authorized signed output root | **64/64** |
| Dishonest but authorized output provider signs false clean output root despite hidden program certificate; independent execution rejects | **64/64** |
| Model-owned self-signed false clean program attempts accepted under independently pinned authority key | **0/64** |
| Dishonest but authorized **program authority** signs false clean program and matching output root, passing signatures and faithful execution while disagreeing with designated hidden world | **64/64 pass cryptographically while semantically wrong** |
| Original first model proposal committed before certified program or hidden target release | **64/64** |
| Authorized hidden-root k16 challenge detects wrong clean proposal | **60/64** |

The 64 original NOR42 source identifiers were generated and SHA-frozen before NOR45 existed. The prior NOR42 first public-only source SHA was `f8bc6ef3db4fda78709f2b7cd1a6a09989dbc087bcfb963d00961cc23ce4f033` and first paired-world grade SHA was `bd57751d94ea49755888000675b0df8fce6490873683268b8f8625518701adae`. The hidden true program for each original source is reconstructed from that already independently verified synthetic source-grammar experiment.

The original model's clean continuation is an exact legal RRC25 program; the hidden world adds an unobserved suffix. A dishonest *output* provider cannot disguise a clean root as the execution result of an independently authorized hidden program. A dishonest but *authorized program authority* can certify the clean program itself, and the provider can correctly execute it: all cryptographic and deterministic computation checks pass while actual designated hidden world truth differs. That counterexample is the sharp limit of execution provenance.

## k16 sparse audit and exact expected coverage

The hidden discrepancy support S is fixed by the earlier source pair, with m=769 or 1537 true coefficients. The model's clean proposal is precommitted before any hidden program, root or challenge is disclosed. A first reproducible public test challenge selects sixteen distinct source-specific positions. It rejects iff its requested set intersects S.

| Original input family | Source pairs | First k16 hidden mismatch detections | Exact finite-support uniform16 hit expectation |
| --- | ---: | ---: | ---: |
| Historical dense | 32 | **32/32 (100%)** | 98.5584% |
| Block-regime inputs | 32 | **28/32 (87.5%)** | 78.7614% |
| **All 64** | **64** | **60/64 (93.75%)** | **88.6599%** |

This is **one deterministic test challenge per source**, with known public seeds. The difference from the exact conditional uniform expectation is sampling variation, not evidence of deployment secrecy. Over the 16 old replicate blocks, the descriptive t95 interval for the observed 60/64 rate is **[87.79%, 99.71%]**. The unit of analysis is a source pair; repeated provider controls from that pair are correlated.

## Original JSON payload accounting

These are actual canonical JSON byte counts, excluding transport framing, full model output reveal, provider key distribution, program computation, and session store costs.

| Component | Mean bytes across 64 historical source pairs |
| --- | ---: |
| Separately signed program-authority certificate | **1,195.2** |
| Verifier-signed original model proposal receipt | 833.0 |
| Provider-signed reference Merkle root | 989.0 |
| Sixteen individual Merkle coefficient openings | **13,065.9** |
| Signed terminal session result | 1,209.9 |
| Combined above five payloads | **17,293.0** |

The original NOR44 signed verifier challenge envelope adds another separately serialized payload, excluded from this five-component sum. No batched Merkle multiproof or succinct execution certificate is implemented. Verifier recomputation costs O(m²) in the worst case for the exact formal-series reciprocal `R`, and requires materializing the complete output vector. CPU timing, peak memory and full transport overhead require a separate explicitly benchmarked receipt.

## Formal verification

[MathCan PR #145](https://github.com/patrickdugan/MathCan/pull/145), merged at commit `17e9c92dbf48724786cdd6a2df2645b9ff7ac111`, kernel-verified five conditional NOR45 theorems with **empty axiom inventories**. They show that valid openings and exact sparse rejection are sound relative to the independently authorized program's execution, under explicit signature/root binding hypotheses. An incorrect output provider's root must differ under collision resistance. A dishonest authorized program can produce an internally consistent signed root while disagreeing with actual external truth.

The theorem proves the *logical consequence of assumed cryptographic authorization, hash binding and program execution*. It does not establish SHA-256 collision resistance, Ed25519 security, the correctness of a real-world program authority, or that the model's input captures an external semantic target.

## Next experiment

The engineering mechanism supports a prospective study of **program authorization provenance** or a proof of faithful execution replacing the full CPU recomputation. A transparent proof/zkVM could certify correct execution of a separately authorized program, with proof generation time, verifier bytes and cost on an actual CPU/GPU. **It would still not establish that the authorized program was semantically true.** Register separate sources, test keys, cost thresholds and hidden audit chronology before any new prospective evaluation.
