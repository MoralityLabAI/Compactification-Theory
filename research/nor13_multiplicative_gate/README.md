# NOR-13: a genuinely basin-conditioned task readout

**Protocol frozen November 15, 2026**, in an isolated branch of the original RRC-25 trained-GRU research. NOR-12's first held-out readout gave a two-case improvement from one aliased model-input cluster. A later 288-case confirmation found seven additional correct cases for both the basin-feature and random-feature residuals, and a controlled ablation of the basin scalar left the first 44/72 score unchanged.

NOR-13 tests a stricter hypothesis: *a recurrent basin should condition the weights applied to the task-relevant hidden vector*. The original GRU's twelve coordinates are preserved, with the frozen one-step task readout kept as an additive baseline.

## Readout architecture

Let \(h\in\mathbb R^{12}\) be the hidden state after the original nine public input rows, \(z(h)\in\mathbb R^{13}\) its NOR-11 train-normalized coordinates with bias, and \(q(h)\in[0,1]\) the frozen NOR-11 basin-head probability of the second zero-input attractor.

For each suffix-template class \(j\in\{0,\ldots,15\}\),

\[
s_j(h)=b_j(h)+\lambda\left[
  u_j^\top z(h)+(2q(h)-1)\,v_j^\top z(h)
\right].
\]

Here \(b_j\) is the original frozen RRC-25 task readout, and \((u_j,v_j)\in\mathbb R^{13}\times\mathbb R^{13}\) is the trainable task residual. The gate therefore changes both task slopes and class offsets. Each head has **16 × 26 = 416 allocated new parameters**. The original trained checkpoint and 13-parameter basin-head weights remain unchanged.

The genuine-basin-gate head is compared against four controls fitted with the same public label objective: a probe trained on shuffled basin labels over the same NOR-11 source cohort, a seeded random logistic projection of the normalized hidden vector, an all-zero gate with effectively 208 active parameters, and an additive-only basin scalar with effectively 224 active parameters. The two fully gated negative controls have the same 416 active parameter slots as the genuine gate.

## Supervision and native task safety

All arms train on the same **96 original NOR-11 training cases** selected from signed, sparse and product source families. The target is the set of *all* suffix templates accepted by two original public demonstrations through degree 192. The loss is the negative log softmax probability mass on the accepted public set, with L2 regularization. The training oracle performs **1,536 exact public template checks** independently of the inference budget.

The original generating program, withheld future query coefficients through degree 256, and test-family labels never enter the task head training objective. A held-out native future-correct outcome is computed only after the model ranks its sixteen candidates and the original exact verifier has checked at most **four** of them.

All residuals use **220 full-batch CPU Adam updates**, learning rate 0.023 and ridge coefficient 0.1. Development uses 36 previously frozen NOR-11 cases, selecting an independent amplitude for each arm from \(\lambda\in\{.125,.25,.5,1\}\). A global dev-selected policy also compares the original no-residual ranker and all gated heads. It maximizes future-correct minus 0.5 false accepts minus 0.015 paid checks; ties favor the original ranker.

## Source-split and independent evaluation

The first 72-case NOR-11/NOR-12 held-out cohort is reported as **retrospective**. The prospective evaluation uses **four new 72-case cohorts** with a new fixed random-seed offset 96110000. Each has twelve source cases per six families. Train/dev/test feature disjointness is checked, together with the previously used first test, and nonlate test tensors cannot duplicate across replications.

The late-source generator can produce many distinct algebraic cases with identical model-visible nine-by-eight tensors. This is kept as an explicit alias cluster, with case-level paired gains, per-replicate unique-feature counts and intervention ablation preserving identical inputs.

The predeclared causal ablations run the **same fitted basin interaction weights** at their frozen development-selected amplitude on each new test cohort, replacing the true basin gate with zero, the shuffled-label probe or a permutation across unique public-feature groups. The model is not refitted between ablations.

A positive basin-specific result requires that the *true gate* improve native future-correct outcomes relative to the best matched control and lose that advantage under genuine gate knockouts. Improvements that persist after knockout are attributable to residual supervised re-ranking rather than a causally needed basin coordinate.

## Scope

This is an empirical **task-conditioned readout**, with the original recurrent dynamics untouched. The frozen head's labels describe zero-input basin proximity at 720 steps and are accurate as measured in NOR-11. NOR-06/08 source-to-basin rational certificates apply to two particular public-input cases; they are not promoted to unseen NOR-13 cases by the learned head.

One held-out future-correct program means the original exact degree-192 public verifier accepted the proposal and the independent degree-256 grader agreed with the future. That result remains conditional on RRC-25's supplied exact public header decoder and its fixed operator vocabulary. Stronger claims about Lean proof search, AI safety, or global attractor semantics require separate experiments.

## Reproduce

From repository root, with Python 3.12 standard library:

    python -m unittest research.nor13_multiplicative_gate.test_gated_readout -v
    python research/nor13_multiplicative_gate/gated_readout.py --check --replicates 4 --per-family 12 --out /tmp/nor13_gated.json

The [NOR-13 hosted workflow](../../.github/workflows/nor13-multiplicative-task-gate.yml) runs independent source-lock and optimizer tests, then the full four-cohort matched control evaluation and gate knockouts. It uploads the complete fitted weights, test fingerprints, native task outcomes and ablation traces as a GitHub Actions artifact.
