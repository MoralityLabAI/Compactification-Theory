# NOR-22: authenticated target commitments and the public-information boundary

**Preregistered 16 November 2026 before generating NOR-22 sources, opening future labels, or executing the prospective test.** Continues merged [NOR-21 #185](https://github.com/MoralityLabAI/Compactification-Theory/pull/185) and the exact observation-completeness mathematics in [MathCan #106](https://github.com/patrickdugan/MathCan/pull/106).

## Question

NOR-21 established an exact obstacle: two worlds can have identical public degree-512 or degree-1024 demonstrations and query INPUT, yet disagree on the hidden target OUTPUT. A deterministic public-only verifier issues the same output in both worlds and fails one. NOR-22 introduces **additional, explicitly trusted target-dependent information**, rather than claiming an output-only algebraic check can resolve that obstacle without a new source of truth. We test what small authenticated modular commitments can certify about an otherwise hidden future, alongside deliberately forged and colliding commitments.

The target commitment is an **exogenous input attested by an assumed trustworthy publisher**, who has read the true output. This is a new authorization/data channel and its value depends on the withheld target. The usual public-only policy sees neither a target commitment nor a true target. All source generation, commitments, and grading are accounted separately, so no result is described as pure public-only reasoning or general proof soundness.

## Frozen source distribution and alternate future worlds

- Native synthetic grammar: the original RRC25 sixteen U/V templates with four I/N/R/S unary transforms, four delays, two spacings and four amplitudes, exact integer formal series, original source and header operations. No retrained model.
- Public observation degrees **512 and 1024**; query INPUT degrees **768 and 1536**; two observed degree-D demonstrations per example.
- Source families: NOR-21's \`wide_dense_small\`, \`wide_sparse_small\`, and \`wide_rational_short\`, with the same small normalized source polynomial semantics. New independent source seed offset **94,322,000**: \`corpus.SEED + 94322000 + rep*1000003 + degree_index*100009 + family_index*10007\` for rep ∈ {0,1}. Pick the first unique public header with a nonempty exact accepted template set and one **uniquely predicted** query continuation among the accepted candidates, with a maximum of **100** sampling attempts. Any failure to generate a compatible source causes experiment failure.
- Exactly **12 distinct complete-public source inputs**, 6 per public observation degree and 4 per family. Each public input is paired with **four counterfactual target worlds**, for 48 source-world records. These worlds share the same source public observations and query INPUT, and differ **only** in the final coefficient of the hidden query OUTPUT. They must be analyzed as paired counterfactuals, not as 48 independent public source draws:
  - \`valid\`: original native generating-program future \(Y\)
  - \`unit_shift\`: \(Y + 1\cdot q^{Q}\)
  - \`p_shift\`: \(Y + p\cdot q^{Q}\)
  - \`product_shift\`: \(Y + pq\cdot q^{Q}\)
  where \(Q=768\) or \(1536\).
- Fix moduli **p=65537**, **q=65521**, and public polynomial evaluation point **r=3** (nonzero modulo both). The different symbol q used above in polynomial monomials is the formal variable, while the modulus q=65521 is a named integer. Both moduli are primes and coprime; this particular experiment does not require treating a hash as collision-free.
- For each of the four hidden-target worlds, the source publisher creates a **trusted pair of target-dependent commitments** \(H_p(Y)=\sum_i Y_i r^i\bmod p\) and \(H_q(Y)=\sum_i Y_i r^i\bmod q\), by exact Horner evaluation. These commitments are given separately from the public-only input; the target coefficient arrays and program identity remain sealed to the public policy. We explicitly record the commitment values and their assumed authenticity. This extra target-derived information is the *only* reason an otherwise invisible hidden-future shift can be detected.
- Public sources and source selection are fixed before grading, using only headers, demonstration compatibility and candidate predicted-query-output uniqueness. For each source the policy's public-only candidate and output are frozen before any target-dependent commitment comparison or sealed true-future correctness comparison. A program's public mask and publicly predicted output cannot depend on which of the four hidden worlds is later considered.

## Frozen policies

For each of the 48 source-world pairs, score the same publicly selected hypothetical future using these four policies, with identical underlying 16-template public checking:

1. **\`public_only\`**: return the unique publicly predicted output without seeing any target commitment. This is the NOR-21 reference.
2. **\`trusted_p\`**: return the candidate only when its \(H_p\) agrees with the **authentic target commitment** modulo p. Mismatch causes abstention. The policy is *allowed to read exactly one additional target-dependent modular value*.
3. **\`trusted_pq\`**: return only when **both** authenticated target modular commitments match. It sees two extra modular values and may abstain.
4. **\`forged_pq\`**: receives two unchecked commitments synthesized from the candidate's own predicted output, instead of the target publisher. This explicitly adversarial trust control cannot distinguish a hidden future override from the original world.

A mismatch with a trusted commitment is **definitive evidence of an incorrect proposed target**: exact equal output arrays always have equal fingerprints. Equality of these modular fingerprints is only a *necessary* condition for correctness. Modulus-sized and product-modulus-sized changes can collide. The latter are *preregistered attack constructions*, so the experiment does not imply cryptographic binding, external future truth access from a private source, or a sound unconditional certificate from a short modular digest.

## Registered expected arithmetic outcomes, without future-label fitting

For the terminal coefficient shifts under the fixed nonzero evaluation point, \(H_m(Y+\delta q^Q)-H_m(Y)=\delta r^Q\bmod m\). Because r=3 is coprime to both fixed moduli, a single trusted p commitment detects exactly those shifts **not divisible by p**; two trusted commitments detect those not divisible by **both p and q**. These are source-independent algebraic predictions.

We preregister the exact 48-world confusion matrix as a **deterministic test of implementation and declared assumptions**, rather than treating its artificially constructed rows as independent statistical evidence:

| Target world (12 distinct sources each) | public_only | trusted_p | trusted_pq | forged_pq |
| --- | --- | --- | --- | --- |
| valid | 12 correct | 12 correct | 12 correct | 12 correct |
| unit_shift | 12 false accepts | 12 abstentions | 12 abstentions | 12 false accepts |
| p_shift | 12 false accepts | 12 false accepts | 12 abstentions | 12 false accepts |
| product_shift | 12 false accepts | 12 false accepts | 12 false accepts | 12 false accepts |

A *different* observed matrix, failure of exact mask parity, a target-program leak, or source duplication is an experiment failure. If this registered table holds, it establishes correct implementation of the **additional trusted signal** and the explicit modular collision boundary, not superiority of learned reasoning or unconditional proof soundness.

## Evidence and formal mathematics

Report all 12 public digests, model-visible input hashes, exact public accepted masks, identities of the four counterfactual worlds by target-output SHA-256, the trusted target fingerprints (they are additional public information), the policy's accept/abstain result, sealed final correctness, false accepts, and checks for two-world public-data equality. Publish original complete JSON and a deterministic independent audit in the applied repository, with a source SHA-256 and no model training. The companion MathCan theorem should prove that a mismatched authentic fingerprint rejects a wrong output, that a self-commitment comparison is vacuous, and that two distinct hidden futures with matching public observations and matching modular fingerprints remain impossible to distinguish, including explicit p and pq shifts.

The first fixed experiment runs in Python 3.12 on GitHub Actions CPU with a **30-minute CI limit**, source-selection regeneration, and an independent post-selection grader. The verdict must report complete source-world counts, the number of distinct public sources, and the exact authenticated-information budget. No later seed or threshold choices are permitted without a dated protocol amendment before seeing new test data.

## Reproduction

    python -m unittest research.nor22_authenticated_commitment.test_experiment -v
    python research/nor22_authenticated_commitment/experiment.py \
      --check --out /tmp/nor22_full.json

This is a finite synthetic authorization and verifiability result, not a recommendation for deploying a 16-bit authentication code for security-sensitive applications.


## First hosted source-locked NOR-22 outcome

The preregistered first experiment completed in [Actions run 43816319904](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43816319904), after registering the source offset, moduli, evaluation point, policy semantics and exact expected matrix above.

All twelve newly generated complete-public inputs were distinct, with six degree-512 and six degree-1024 source cases and four cases from each of the three frozen source families. Each source was paired with four altered hidden-target worlds, giving 48 correlated source-world records. Every public mask had a uniquely predicted future in the original sixteen-program generating family. The target-dependent authenticated commitments are explicit additional inputs, while candidate selection was frozen before the private program and target were reconstructed for grading.

### Registered outcome matrix reproduced exactly

| Target world (12 each) | Public only | Trusted 65537 | Trusted 65537 and 65521 | Forged two-residue anchor |
| --- | --- | --- | --- | --- |
| Valid future | 12 correct | 12 correct | 12 correct | 12 correct |
| Final coefficient +1 | 12 false accepts | 12 abstain | 12 abstain | 12 false accepts |
| Final coefficient +65537 | 12 false accepts | 12 false accepts | 12 abstain | 12 false accepts |
| Final coefficient +4294049777 | 12 false accepts | 12 false accepts | 12 false accepts | 12 false accepts |

The exact sixteen-program public candidate and hypothetical future was **identical across all four hidden target worlds** for a given public source. Only the explicitly target-dependent trusted commitment changed. A forged "commitment" calculated from the proposed output itself did not alter the error rate. The two trusted moduli ruled out unit and p-sized target changes while still colliding on p·q-sized terminal changes. A trusted fingerprint is a necessary consistency check, with deterministic residual collisions.

The code audits **twelve distinct public sources and 192 policy decisions**; statistical significance from 48 or 192 independent examples would be invalid because each source has four controlled hidden worlds and every policy shares the same selected public prediction. This is an algebraically preregistered information-boundary experiment with exhaustive outcomes for its fixed counterfactual design.

### Frozen original evidence and independent audit

The source-only public fixture, **including the explicitly extra target-dependent modular commitments**, is 2,236,097 bytes, SHA-256:

    a325aa87214bbfcabe40c6f964e514ded5f16dd3a18752c09e2d99b907590a4d

The complete first-run sealed grade JSON is 128,050 bytes, SHA-256:

    c6e7eba604b43a113362505ded2a5581305ff5213afaff3fe0c726741b35795b

Both are permanently retained byte-for-byte as [results/nor22_original_hosted.zip](results/nor22_original_hosted.zip), with an immutable [first frozen receipt](results/nor22_first_frozen_receipt.json). The [standalone independent auditor](receipt_audit.py) rebuilds the original source programs from frozen seeds without importing the NOR-22 implementation, verifies complete public input hashes and original candidate membership, independently evaluates polynomial fingerprints through modular powers, checks all 96 authentic target values, and reconstructs all 192 policy choices and grades.

The first [independent receipt replay](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43817062027) passed on the original source data. The one-shot write-capable evidence publisher has been removed after a successful original-source archive commit. The permanent CI verifies archive SHA-256 before the original source-generation and prospective test jobs.

### Exact mathematics and trust scope

The companion [MathCan NOR-22](https://github.com/patrickdugan/MathCan/tree/research/nor22-modular-future-commitments-20261116/MathCan/NOR22ModularFutureCommitments.lean) proves authentic mismatches reject wrong predictions, self-signed commitments are vacuous, an observed signature collision preserves public indistinguishability, and an arbitrary-length polynomial terminal shift obeys exact evaluation change \(\Delta\cdot3^Q\). Both modular collisions persist at any finite degree.

This synthetic result **requires a trustworthy external commitment publisher to observe the true target**. That information was withheld from the old public-only verifier and is deliberately supplied in the new trusted policies. The experiment is not a way to recover unknown target truth from public demonstrations, and the two small moduli are not sufficient cryptographic protection against an adaptive adversary. A real commitment scheme and authorization trust boundary would need separate treatment.
