# LSV-17: finite-horizon behavioral quotient discovery

**Date:** 2026-11-04. **Status:** reproducible CPU protocol with synthetic unit tests; real pretrained results require the separate source-locked run receipt. The experiment is distinct from the Unique Games linear-lift theorem.

## Mathematical question

For a recurrent map R and readout f, define the three-step goal-cell behavior signature

Y3(z) = (f(R(z))[cell], f(R^2(z))[cell], f(R^3(z))[cell]) in R^30.

The exact finite-horizon equivalence is z ~ z' if and only if Y3(z) = Y3(z'). Our measured approximate equivalence tracks per-horizon label flips, softmax total variation, KL divergence, and changes to a fixed clean winner-runner margin. Preserving only the immediate winner is a weaker condition.

For J3(z) = D Y3(z), an exact smooth 16-dimensional quotient around z must have rank J3(z) at most 16. The current gym measures three rows: the derivative of the clean top-two logit margin at each of three horizons with respect to the 48-dimensional goal-cell state. A perturbation in their joint kernel is neutral only to first order. If the Hessian norm of the chosen behavior signature along that segment is bounded by M, Taylor's integral formula bounds its deviation by M * ||delta||^2 / 2. The gym records Taylor residuals but supplies no valid upper bound on M.

## Fixed protocol

- Freeze the externally pretrained 44,001-parameter TRM-inspired Sudoku model from pretrained_trm/source_lock.json; restricted deserialization checks its complete source and checkpoint hashes. The first recurrence runs in original model FP32, with later stages and gradients in FP64, as in LSV-09/13.
- Use the existing 64 uniquely generated Sudoku cases. Target the first missing cell per puzzle. By puzzle index modulo 4, groups 0 and 1 train; group 2 validates; group 3 tests. Every perturbation of a puzzle remains in the same split.
- At the post-first-recurrence goal-cell token, inject centered Gaussian noise, first-horizon Jacobian-null noise, all-three-horizon Jacobian-null noise, terminal-margin-gradient noise, and a common shift. Match the L2 norms at frozen radii 0.2, 1.0, and 3.0; the code checks direction norms.
- The exact structural negative control uses a different intervention site, the *pre-LayerNorm2 bus* in the next recurrence. At the same L2 magnitude, compare per-token common shifts against a centered transverse perturbation. This is a FP64 measurement of the already-proved LSV-15 real-arithmetic identity.
- Record all three continuation horizons, clean and perturbed Sudoku correctness, fixed clean class margins, tangent predictions, finite differences, TV/KL and flip status. Avoid choosing perturbations using test outcomes.

## Supervised nonlinear quotient candidate

Train one linear 48-to-30 readout and one nonlinear 48-to-16-to-30 tanh readout from the altered goal-cell state to the *teacher-generated* three-horizon future logits. Use identical puzzle-disjoint splits; center and scale from training data only. AdamW, validation early stopping, trainable parameter count, test future-logit MSE and teacher-label agreement at each horizon are reported. Both models see identical data. Their exact parameters and bottleneck ranks differ and remain visible in receipts.

The nonlinear hidden layer supplies a candidate sixteen-coordinate task statistic. This does not demonstrate compressed inference until a quotient replaces recurrent computation or proves continuation-level semiconjugacy.

## Falsification gates

On held-out test puzzles, compare all-horizon Jacobian-null against matched random perturbations at every radius. Record early-stable/later-flipped examples. The higher-order residual shows where tangent neutrality fails at finite magnitude. Only consider a nonlinear readout gain if the final test logit MSE and future-label scores improve on a puzzle-disjoint split; preserve negative and null findings.

For a more defensible result, run the full 64 cases and additional precommitted perturbation seeds, with paired confidence intervals clustered by Sudoku puzzle. Source hash and roster hash must accompany every reported statistic.

The frozen Qwen3.5-9B H100 continuation is staged in GOODFIRE_QUOTIENT_H100.md. This CPU artifact does not consume GPU credits or make claims about 9B model activation structure.

## Reproduction

    python -m unittest discover -s research/levin_sieve_v1/pretrained_trm -p 'test_quotient_discovery.py' -v
    python research/levin_sieve_v1/pretrained_trm/quotient_discovery.py --fetch --max-cases 16 --out runs/lsv17/pilot.json
    python research/levin_sieve_v1/pretrained_trm/quotient_discovery.py --fetch --max-cases 64 --out runs/lsv17/full.json

Fetching is explicit and the source lock is mandatory. Each run writes one detailed machine-readable receipt, plus compact summary on stdout.
