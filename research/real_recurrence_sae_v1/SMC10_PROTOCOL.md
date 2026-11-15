# SMC-10: Uniform robust causal rankings around frozen SAE endpoints

Date: November 15, 2026. Builds on merged [SMC-09, 432 exact-real pointwise causal rankings](SMC09_RESULTS.md) and [MathCan OLM-10](https://github.com/patrickdugan/MathCan/blob/research/olm10-robust-endpoint-neighborhoods-20261115/research/orbit_lock/OLM10_ROBUST_ENDPOINT_NEIGHBORHOODS.md).

## Source lock and scientific question

SMC-09 certified strictly separated squared ten-logit effect intervals at fixed intervention endpoints for an independently pretrained 44,001-parameter TRM-inspired Sudoku model. It used exactly specified stage-one goal states and directions selected from fitted top-4 SAE decoder atoms or numerical controls. A fixed-endpoint theorem says nothing by itself about perturbing the hidden state or the direction.

This stage tests how far those comparisons extend to an **entire admitted neighborhood of goal-token states and intervention directions**, while freezing the other 80 token states and the teacher weights.

- Frozen checkpoint source commit: f0406195fece4d0db29ced28f3a69d1993d95b1e.
- Checked checkpoint SHA-256: 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1.
- Checked Sudoku roster SHA-256: eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611.
- Puzzle-disjoint 32 training, 16 validation, 16 previously inspected test puzzles, with full-token rows grouped by puzzle.
- Three SAE fit seeds 71,89,113, same 300-update top-4 ReLU SAE recipe and validation-only checkpoint selection as SMC-07–09.
- The goal is the first blank token of each heldout puzzle. The exact-real three-step recurrent function maps a 48-dimensional stage-one goal-token replacement into ten final goal logits.
- The nominal intervention directions are the highest active learned SAE atom, a seeded inactive atom, an isotropic vector and a numerical top-Jacobian singular direction. Their frozen FP64 dyadic values are normalized in exact-real Arb arithmetic.

## Robustness definition

For exact unit nominal directions u, v, one shared uncertain goal-token anchor h', and exact unit perturbed directions u',v', let

\[
\|h'-h\|_2\le\delta_h,\qquad
\|u'-u\|_2\le\delta_u,\qquad
\|v'-v\|_2\le\delta_v.
\]

The intervention radius r remains a fixed positive rational. The targeted result is that **every admissible choice** preserves the stated ordering of the ten-logit L2 effects measured from the same perturbed clean state h'.

A sound rectangular overapproximation is:

\[
B_0=[h\pm\delta_h]_\infty,\quad
B_u=[h+ru\pm(\delta_h+r\delta_u)]_\infty,\quad
B_v=[h+rv\pm(\delta_h+r\delta_v)]_\infty.
\]

The [OLM-10 finite interval theorem](https://github.com/patrickdugan/MathCan/blob/research/olm10-robust-endpoint-neighborhoods-20261115/research/orbit_lock/OLM10_ROBUST_ENDPOINT_NEIGHBORHOODS.md) shows that strict separation of the validated **squared-effect** intervals computed from these three boxes certifies effect dominance for the entire uncertainty class. The clean and perturbed evaluations can lose their common-anchor correlations, making the test conservative but sound.

The experiments choose a common \(\eta\) for all three L2 budgets. Thus every clean goal-token input is covered by per-channel half-width η, and every perturbed goal-token input by the larger half-width \((1+r)\eta\). These are coordinatewise outer boxes for the original Euclidean uncertainty balls, rather than a replacement for the actual L2 constraint.

## Fixed experiment and abstention criteria

Two heldout puzzle IDs, SAE seed 71, are used for the pilot. Full hosted replay uses **16 puzzle IDs × 3 SAE seeds × 2 radii × 5 budgets**, with:

- Exact intervention radii \(r\in\{1/50,1/5\}\), corresponding to 0.02 and 0.2.
- Common L2 uncertainty budgets \(\eta\in\{0,10^{-12},10^{-10},10^{-8},10^{-6}\}\).
- 160-bit validated Arb, with pretrained weights and stage-one anchors interpreted as exact stored binary input values.
- Four nominal directions and three active-versus-control comparisons at every puzzle/seed/radius/budget.

Each comparison reports **active greater**, **control greater**, or **inconclusive**. If the interval evaluation cannot provide a finite squared-effect enclosure, the recorded case abstains. Failure to prove a ranking at a larger budget is not taken as evidence of a true ranking reversal.

At \(\eta=0\) the 16-puzzle, three-seed comparison must reproduce SMC-09's source-locked pointwise counts (44/48 and 43/48 active-over-isotropic for radii 0.02 and 0.2, respectively; 32/48 active-over-inactive at each radius; 48/48 Jacobian-optimal-over-active). The run aborts if these checks do not match.

The [strict toy tests](test_smc10_robust_endpoint_arb.py) independently verify Arb rational-radius inflation, goal-token-only scope, equal-radius linear ranking, abstention under wide uncertainty and invalid parameter rejection.

## Evidence boundaries

This experiment verifies the **mathematical exact-real frozen recurrence** under bounded changes in the 48-channel goal token and the specific selected unit intervention directions. It does not establish uncertainty in the other 80 tokens, model-parameter changes, stable SAE atom selection after retraining, or a rounded FP64 hardware implementation. The existing weight-level derivative bounds are not substituted into this experiment, and no semantic interpretation of decoder atoms follows automatically from a robust causal sensitivity ranking.

This is a repeat follow-up on an already viewed test cohort and should not be presented as independent model generalization.

## Reproduction

Pinned CPU Python 3.11, torch==2.7.1+cpu, python-flint==0.9.0:

    python -m unittest discover -s research/real_recurrence_sae_v1 -p 'test_smc10_robust_endpoint_arb.py' -v
    python research/real_recurrence_sae_v1/smc10_robust_endpoint_arb.py --fetch --max-test-puzzles 2 --sae-seeds 71 --steps 300 --radii 0.02,0.2 --budgets 0,1e-12,1e-10,1e-8,1e-6 --out runs/smc10/pilot.json
    python research/real_recurrence_sae_v1/smc10_robust_endpoint_arb.py --fetch --max-test-puzzles 16 --sae-seeds 71,89,113 --steps 300 --radii 0.02,0.2 --budgets 0,1e-12,1e-10,1e-8,1e-6 --out runs/smc10/full.json

The [dedicated GitHub Actions workflow](../../.github/workflows/smc10-robust-endpoint.yml) runs both procedures and archives the full per-case interval receipts. Numerical coverage claims require inspection of a successful hosted source-locked run.
