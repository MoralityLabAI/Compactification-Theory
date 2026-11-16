# NOR-17: risk-bounded public verifier under budgets 4, 8, and 16

**Registered before reading the fresh held-out outcomes, 16 November 2026.**

Continues NOR-16, merged [Compactification-Theory PR #170](https://github.com/MoralityLabAI/Compactification-Theory/pull/170), and the conditional public-output unanimity result in [MathCan PR #90](https://github.com/patrickdugan/MathCan/pull/90).

## Research question

The frozen original 1,000-parameter RRC25 recurrent ranker fails many cases in the late_source family because different public source examples can have the identical nine-by-eight model feature tensor. NOR-16 produced a stronger logical observation: all 16 public template checks, followed by exact hypothetical future-output agreement, provide a correct output **conditional on the planted program belonging to the enumerated generator family**.

A sixteen-check requirement may be reduced safely when extra *public* calculations rule out alternatives. NOR-17 tests a 160-coefficient **sound rejection filter** that rules out any candidate whose predicted demonstrations disagree with either observed demonstration before degree 160. A candidate rejecting at this prefix also rejects under full degree-192 verification. The certificate can then spend a budget of at most 4, 8, or 16 full exact public checks on remaining candidates, using their publicly simulated degree-256 query *input* predictions to decide when every still-compatible candidate has the same predicted output.

## Formal certificate rule

Let \(U\) denote the complete 16-candidate set, and let \(S\subseteq U\) be the subset surviving exact public prefix checks. Necessarily every full-demo-compatible candidate belongs to \(S\).

A certifier attempts full public verification within budget \(k\in\{4,8,16\}\). A failed exact check removes its candidate; an accepted check confirms a candidate program. Let \(R\subseteq S\) contain those not ruled out by a full-check rejection.

Return a proposed program \(j\) **only if** (i) that program has passed a full exact public check, and (ii) **every program in \(R\)** predicts the same complete hypothetical query output as \(j\). When verification produces two positively accepted programs with different predicted query outputs, the certifier abstains immediately.

Provided the true planted generator belongs to the sixteen candidate families and exact public template checks are sound, its program is in \(R\) at all times. If the certifier returns \(j\), the planted program and \(j\) predict the same future query output. This yields a zero-false-accept *theorem within the declared synthetic model family*, independently of statistical confidence or calibrated probability. More general theorem-prover soundness is beyond that premise.

## Locked source cohorts and policies

The original RRC25 checkpoint remains SHA-256:

\`54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b\`.

No model weight, task head, optimizer, training set, or development selector is changed. Only publicly computed inference features and public verification policies vary.

- **New four-cohort test seed:** \`corpus.SEED+93317000+rep*1000003+family_index*10007\`.
- **Primary:** 192 newly generated distinct complete-public late_source examples, 48 per replicate. They share one nine-row original GRU feature tensor.
- **Secondary:** 48 new signed and 48 rational examples, disjoint nonlate model feature hashes. All public demonstrations from earlier NOR11/12/13/14/15/16 are excluded.
- **Native greedy baselines:** fixed original score ordering; stop at the first full public-accepted candidate under budgets 4, 8, and 16.
- **Unscreened certificate controls:** simulate 16 hypothetical query outputs but skip public prefix screening; attempt to eliminate competing outputs via at most 4, 8, or 16 full public template checks.
- **Prefix-screened certificates:** apply all sixteen exact degree-160 public prefix comparisons, then use public predicted-output multiplicity among surviving programs to rank at most 4, 8, or 16 exact full checks. The returned program must satisfy the certificate rule.
- **Matching 4-check controls:** prefix filter followed by original GRU ordering, or a seeded query-hash-shuffled ordering of the same surviving programs. Both pay the same prefix work and query simulations as the main prefix-4 arm.

Four-check, eight-check and sixteen-check budgets measure **full degree-192 exact candidate verifications**. An additional prefix filter costs sixteen comparisons over two degree-160 demonstrations, with eight cached public unary transforms and thirty-two public prefix candidate compositions. All certificate arms also construct sixteen degree-256 hypothetical query outputs using four cached public unary source transforms. These costs appear separately in every arm's ledger. Baseline greedy arms do not use query continuations or prefix filtering. If implemented independently, arithmetic must be charged separately for every arm; the experimental runner may cache computations across arms for efficiency.

The source-only *retrospective* full-mask validation checks all sixteen full templates after every arm's candidate choice has been frozen, solely to audit the prefix-superset invariant. Its extra cost of sixteen public checks per case is recorded separately and is never used to choose a program or select a result.

The withheld degree-256 query *output* and private generating program identity are inaccessible to the policy functions. The original \`grade\` evaluator is invoked only after **all** proposals are frozen. Unit tests replace hidden future output and private program identity and require identical policy decisions.

## Registered success and statistical scope

Primary correctness measures:

- **Certificate coverage:** fraction of the 192 complete-public late_source cases on which a risk-controlled arm emits a certified program.
- **Safety:** future-correct and false accepts for each arm under the original degree-256 withheld target grader. Every certificate arm is expected to have *zero false accepts* under the exact-family assumption. Any nonzero result invalidates the implementation or its soundness premise.
- **Verifier work:** full paid checks, prefix degree-160 candidate comparisons and their operator arithmetic, query degree-256 candidate continuations, public unary transforms, and retrospective audit charges.
- **Budget response:** certificate coverage at budgets 4, 8 and 16, with monotonicity checks under the fixed ranking for each filter type.
- **Controls:** original first-public-accepted policy, query-only certificate without prefix filter, original-order prefix control and public-hash prefix control.

The **preregistered primary capability hypothesis** requires the prefix-4 certifier to achieve a strictly positive paired coverage gain over *both* the unscreened-4 certificate and the matched original-order prefix-4 certificate. Each two-sided exact paired sign p-value must be at most 0.025. The prefix-hash control, eight- and sixteen-check policies, and nonlate source families are secondary comparisons. Any positive result will need additional independent-seed replication for external generalization.

The 192 primary examples are distinct public cases, despite sharing a single original GRU feature tensor. Correlated generator structure may still violate exchangeability assumptions behind naïve case-level significance tests; paired p-values are descriptive until independently confirmed. The conditional zero-false-accept theorem is a *logical property* of the candidate-family and verifier interface, distinct from statistical reliability on systems outside that family.

## Reproduction

\`\`\`sh
python -m unittest research.nor17_prefix_certificates.test_experiment -v
python research/nor17_prefix_certificates/experiment.py --check \
  --repetitions 4 --late-per-repetition 48 \
  --signed-per-repetition 12 --rational-per-repetition 12 \
  --out /tmp/nor17_full.json
\`\`\`

The first completed hosted CI will upload full source/feature hashes, candidate masks, public query prediction hashes, individually paid check traces, all abstention reasons and sealed grader results. The independent receipt auditor will later freeze its byte-level JSON SHA-256 and recompute every causal comparison. No held-out numerical effect is claimed at protocol registration.
