# NOR-13R: independent basin-gated task readout and exact rank audit

**Protocol, 15 November 2026.** Independent continuation of NOR-12 and of the concurrently opened [416-parameter NOR-13 branch](https://github.com/MoralityLabAI/Compactification-Theory/pull/158). This research branch uses 400 allocated task-residual parameters, a different held-out seed family, and a binary-rational score-endpoint ranking audit. Its goal is to measure whether the previously trained basin coordinate has any marginal causal role in *native* RRC-25 proof-template selection.

## Locked trained model and proposed conditional mechanism

The original 1,000-parameter RRC-25 GRU is source-locked under SHA-256 54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b. Its 16-class task head and NOR-11 13-parameter basin head are frozen. The residual scorer is

\[
s_j(h,g)=b_j(h)+\lambda\bigl(u_j^\top z(h)+g\,v_j^\top z(h)+c_j\bigr),
\quad g(h)=2p_{\rm second}(h)-1,\quad j=1,\ldots,16.
\]

The standardized hidden vector \(z(h)\) has dimension 12. Each head has \(16\times25=400\) allocated parameter slots. Multiplication by the gate can change the task ranker's slopes as hidden states vary.

Four matched training architectures are tested: the frozen trained basin gate; the NOR-11 shuffled-label basin probe; a deterministic random scalar derived only from the public input SHA-256; and a constant-zero gate that leaves the interaction slots inactive. The original task ranker and both NOR-12 additive residual controls are evaluated on the same new test examples.

Each head uses the original 96 public training cases and trains 180 full-batch Adam steps with learning rate 0.023 and L2 penalty 0.075. The objective is the negative logarithm of softmax mass on all templates **exactly accepted by the two public degree-192 demonstrations**. Its 1536 full template checks per fitted model are a separately charged training oracle cost. Neither the private generating program nor withheld future coefficients contribute to the fitted scores.

The original 36-case development split freezes each head's scale from 0.25, 0.5, 1.0, 2.0. The global development choice includes the unchanged task readout. Native future correctness is used only for development scale selection and final held-out scoring.

## Source-independent replication

Four newly seeded cohorts of twelve examples per each of six source families total 288 cases. Every nonlate model-visible input is disjoint from the NOR-11/12 train, development, original test and four independent NOR-12B confirmation cohorts, and from the other NOR-13R cohorts. Late-source cases have unavoidable aliases because the synthetic generator may map distinct public-visible cases to identical nine-by-eight hidden-state features. The effective unique-input count is **241**. Both raw case and distinct-feature-cluster outcomes are recorded.

[First completed hosted CI run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43711277361) passed source and gradient tests and the complete four-cohort task evaluation:

| Readout | Future-correct /288 | False accepts | Paid inference checks | Paired gains / losses |
| --- | ---: | ---: | ---: | ---: |
| Original frozen GRU | 168 | 25 | 809 | 0 / 0 |
| Multiplicative real basin gate | 171 | 26 | 792 | 14 / 11 |
| Random public-feature gate | 171 | 30 | 796 | 22 / 19 |
| Shuffled-label basin gate | 170 | 26 | 795 | 17 / 15 |
| Constant-zero gate | 170 | 26 | 794 | 13 / 11 |
| Prior NOR-12 additive basin residual | 168 | 26 | 791 | 9 / 9 |
| Prior NOR-12 additive random residual | 169 | 25 | 794 | 13 / 12 |

The global development selector chose the random gate at scale 1.0. The basin-gated head selected scale 0.5 locally. The real gate gained three net future-correct cases while the matched random gate achieved the same 171/288. The unadjusted input-group sign-test value of approximately 0.049 for the basin arm is **exploratory**, with multiple model comparisons and source aliasing; it is insufficient to establish superiority of the basin feature.

## Causal gate ablation and exact endpoint order

The [runner](interaction.py) evaluates the same frozen basin-task matrix with its gate set to zero, replaced by a shuffled-basin probe or public-random scalar, sign-flipped, or permuted across unique input-feature groups. Each intervention changes only the scalar gate, leaving hidden states, trained task weights and the four paid candidate-check budget unchanged. The study reports these comparisons separately from the training advantage of residual task learning.

The exact score form for fixed input is \(s_j(g)=a_j+g\,d_j\) with \(g\in[-1,1]\). For each pair of the 16 classes, exact binary-rational endpoint comparisons at \(-1\) and \(+1\) certify whether relative order can vary in the interval. A stronger **ordered top-four** certificate checks the strict comparisons that matter to the original four-paid-check verifier. Lower-ranked templates may reorder with zero effect on the observed proof task. The analytic real-valued theorem has a [paired Lean finite-endpoint core](https://github.com/patrickdugan/MathCan/pull/74).

The rational calculations lift stored binary64 values to exact rational numbers, and therefore certify the mathematical affine score-lift. Full IEEE-754 evaluation semantics would require a separate outward floating-point rounding margin. The code flags this limit explicitly.

NOR-11 established strong prediction of the asymptotic recurrent mode; NOR-12 showed that additive basin-feature ablation left original test correctness unchanged. The richer interaction readout and exact endpoint tests now probe the marginal causal value of the basin coordinate for finite proof-template ranking.

The experiment concerns a trained synthetic algebraic GRU with a supplied exact header decoder. Its native task gains remain small and unconfirmed as basin-specific. No general Lean-reasoner or transformer performance claim follows.
