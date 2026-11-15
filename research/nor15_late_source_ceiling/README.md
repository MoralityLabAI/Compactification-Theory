# NOR-15C: finite four-check information ceiling inside one late-source feature fibre

Protocol registered 15 November 2026, after completed and merged NOR-15A/B; **new independent source data** are generated only after this protocol and code are fixed. Complementary finite-resolution reconstruction theorem in MathCan NOR-15 PR #83.

NOR-15B's nonlate evaluation had zero false accepts under all eight arms. Its forty \`late_source\` examples shared one nine-by-eight model input and carried all false accepts (20 for the original GRU). A full observed demonstration contains information that the truncated nine-row neural feature tensor omits. The exact public verifier can use the longer demonstration to accept/reject the sixteen operator-pair templates, but the original ranker chooses its candidate order only from the compressed tensor.

## Precisely restricted policy class

One frozen feature hash induces one score vector and hence one shared total order of the 16 operator-pair templates. An episode checks that order until the first exact public accepted candidate, with a maximum of FOUR checks; then its acceptance is independently graded using withheld future coefficients. A deterministic agent using only the shared feature hash and a sequence of public rejection bits before the first acceptance also follows one fixed four-candidate order, since the only continuing history before acceptance consists of repeated rejections.

There are \(\mathrm{P}(16,4)=16\cdot15\cdot14\cdot13=\mathbf{43,680}\) ordered four-candidate policies. Exhaustively enumerate them over the selected cohort using a precomputed **public** sixteen-bit acceptance mask and a separately computed evaluation-only sixteen-bit withheld-future correctness mask for each source example. Report maximum future-correct, false accepts, paid exact verifier calls and abstentions under a deterministic future-correct-first tie-break. This is the **hindsight oracle upper bound for the restricted shared-order class**; it is selected using the test futures and cannot be described as a deployable learned policy.

Train selection uses exactly 80 previously unseen \`late_source\` cases under seed offset 93015000, with all original NOR-11/12/12B/13R/14/15A/15B *visible* source cases excluded. The best four-order on these 80 uses withheld future labels for **diagnostic selection**, and is frozen before the 160-case fresh TEST set is independently generated under seed offset 93115000. TEST computes the original GRU order, the DEV-selected four-order, the test-hindsight best fixed four-order, and a clairvoyant per-case full-sixteen oracle. The last two are upper bounds only. The test is distinct by full visible demos but shares the same model-visible input hash as DEV, as structural aliasing is the subject.

We also group TEST cases by their complete sixteen-bit public acceptance masks and count each group's maximum future-correct successes achievable by selecting one accepted candidate for that public mask. This intermediate oracle can use full-demonstration acceptance-mask information while remaining agnostic to private future within a group; it is still a hindsight benchmark, and assumes free access to the entire acceptance mask (16 public checks rather than four). It quantifies the price of compressing public demonstrations and stopping after four checks.

## Accounting, restrictions and expected conclusions

- Frozen original 1000-param GRU checkpoint SHA-256 \`54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b\`. No neural weights updated.
- 80 diagnostic DEV + 160 source-new TEST late_source cases, 240 distinct public-visible source fingerprints. All share the same nine-row model-input hash.
- 16 public observed-degree-192 exact candidate checks per case, separately charged **3,840 diagnostic public checks**. Sixteen withheld degree-256 future grades per case are diagnostic oracle computations; 3,840 grades must be recorded and excluded from deployable native inference resource accounting.
- Original and DEV-selected actual TEST policies each use at most FOUR paid exact public verifications per example.
- Primary reported outcomes are descriptive counts, ranked source-group representations, public-mask equivalence classes, and exact finite enumeration. No case-independent sign test is justified because every case shares one neural feature tensor and generator family.
- The test-hindsight optimal four-order is a *sample-conditional upper bound* on restricted policies on these exact 160 cases. It is not a population bound, an information-theoretic mutual-information claim, or an upper bound on policies that inspect fuller public inputs to adapt their order.
- The compiled Lean pair-collision theorem in MathCan #83 gives an exact abstract bound; this synthetic-source oracle enumeration is a separate finite operational result.

Run (after executable is committed):

    python -m unittest research.nor15_late_source_ceiling.test_ceiling -v
    python research/nor15_late_source_ceiling/ceiling.py --check \
       --dev-cases 80 --test-cases 160 --out /tmp/nor15c_full.json

Data-generating seeds, enumeration method and all separation criteria are frozen before the first new test scores are read.


## First independent executed result (15 November 2026)

[Green source-locked NOR-15C Actions](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43723763342), with archived run artifact ID **14165597500**, produced the original 205,705-byte result JSON at SHA-256 \`11b90206703a3e21c7163435de9f95d6c212cb320bf0a663bc86d0ce4981236e\`. The [immutable summary receipt](results/nor15c_frozen_receipt.json) pins the source manifests, input digest, exact policies and both oracle ceilings. The [independent finite-order auditor](receipt_audit.py) independently rebuilds the 43,680-order search on both cohorts from the received public and evaluation-only future masks.

All 80 DEV and 160 TEST source cases had unique full public-visible input hashes and one common model-visible feature hash. The original network's entire 16-candidate ordering is consequently identical across cases. Diagnostic public mask construction used **3,840 observed-demo template checks**, and independent withheld-future grading used **3,840** future checks. These exhaustively evaluated, future-labeled masks are analysis-only inputs to the hindsight oracles.

| Restricted policy | TEST future-correct /160 | TEST false accepts | Abstentions | Paid public checks |
| --- | ---: | ---: | ---: | ---: |
| Original trained GRU shared top four \`[3,12,0,7]\` | 30 | 77 | 53 | 387 |
| DEV-future-selected shared top four \`[4,15,3,11]\` | 34 | 94 | 32 | 394 |
| **TEST-hindsight best fixed four** \`[1,10,13,3]\` | **49** | 79 | 32 | 368 |

The last row is the exact optimum over all 43,680 fixed ordered-four policies **on this test cohort**, subject to the first-public-accepted stopping rule. Its candidate order was selected using withheld TEST futures. The DEV-selected order also used withheld DEV futures; it is a labeled diagnostic rather than a deployable train protocol.

The 160 TEST cases produced **26 distinct sixteen-bit public acceptance masks**. A separate hindsight oracle allowed a different choice for each complete public acceptance-mask group, without dividing by a four-check budget. It achieved **84/160** correct using future labels to select the group mapping. A per-case fully clairvoyant choice achieved 160/160. Forty-one cases had a \`0xffff\` acceptance mask: all sixteen templates passed observed demonstrations, yet the best single proposal for those 41 cases was future-correct in only **6**.

This is direct evidence of source-aliasing and incomplete demonstration identifiability for this finite synthetic generator. A richer decoder can use public information outside the nine-by-eight feature tensor, while public-mask classification still faces ambiguity when multiple future outcomes share one mask. The experiment cannot establish a population ceiling or positive conditional information without additional generative assumptions and independently certified identifiability arguments.

The original frozen RRC25 recurrent weights and task head were unchanged.
