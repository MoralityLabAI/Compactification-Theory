# NHB-01: executed finite monitor results

Date: 2026-11-14. Evidence: EXACT_FINITE_NOISY_CUBE_CONTROL and EXACT_FINITE_DECISION_MEMORY.

[Theory and source boundaries](THEORY.md) | [Exact reference](reference.py) | [Decision-memory certificate](decision_memory.py)

Verified GitHub Actions run: https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43554551602 . The code at da2a28a2b3f6286baedaa5e8fabcacba84396766 passed all 22 tests and produced both JSON artifacts.

## Source pairing

OpenAI Math #119 reports sharp contraction of binary information through independent symmetric noise on a uniform cube, including randomized binary channels at fixed initial information. #127 reports average sensitivity at most 8*d*sqrt(n) for degree-d polynomial threshold functions. The campaign's parameter-only information/error theorem is conditional on these general source statements. The selected exact finite controls and elementary probability deductions have independent checks.

## Frozen experiment

Each original state has independent fair sign coordinates. Each coordinate independently flips with probability 1/20 at every step. The target is a predefined Boolean function of the original state, independently flipped with probability 1/10. Only the final noisy state is available for decoding. Every finite state is enumerated; no sampled train/test split or fitted neural model is involved.

The table records the Bayes-optimal error given the entire terminal state. All underlying probabilities are exact rational numbers; displayed percentages are rounded.

| Target | One noisy step | Four steps | Eight steps |
|---|---:|---:|---:|
| One coordinate among seven | 14.0000% | 23.7560% | 32.7813% |
| Seven-bit majority | 17.5980% | 29.9119% | 37.7340% |
| Product of two three-bit majorities | 20.6049% | 37.3355% | 45.2997% |
| Seven-bit parity | 30.8681% | 47.9066% | 49.8904% |

The coordinate saturates the balanced-target error floor. For target error <=1/4, four steps are feasible and five are impossible for every balanced target under the registered terminal-only interface. The exact coordinate errors are 5939/25000 at four steps and 65951/250000 at five steps. The general error floor also has an elementary Fourier/L2 proof independent of the imported information theorem.

## Information at four steps

The full terminal state retains the following Shannon information about the randomized target:

| Target | Full-state information, bits | Best-decision bit information, bits |
|---|---:|---:|
| Coordinate | 0.209041 | 0.209041 |
| Majority7 | 0.151222 | 0.119789 |
| Product of majorities | 0.055693 | 0.046786 |
| Parity7 | 0.001265 | 0.001265 |
| Rare-alert conjunction | 0.002521 | 0.000000 |

The best-decision bit preserves full-state zero-one classification accuracy in every case. Its computation may require full state access and a posterior lookup table. With the rare target at this noise level, the best decision is constant even though the full state carries some target information.

Entropy values and the inverse-entropy source cap are numerical, with comparison tolerance 2e-11. The probabilities, Bayes errors, Fourier coefficients and state counts are exact Fractions. The information table is not labeled an exact real-arithmetic logarithm certificate.

## Minimum memory for arbitrary posterior-threshold decisions

For this finite population, a code that exactly answers every query of the form Pr(U=+1|Y)>=threshold needs one distinguishable state per distinct posterior probability. The source constructs the posterior-value code and midpoint queries that distinguish its levels.

| Target | Distinct posterior values | Minimum fixed-length bits for every threshold |
|---|---:|---:|
| Coordinate | 2 | 1 |
| Majority7 | 8 | 3 |
| Product of majorities | 6 | 3 |
| Parity7 | 2 | 1 |
| Rare-alert conjunction | 8 | 3 |

For majority7, one bit preserves the best fixed binary decision. The full threshold interface requires three output-state bits. All static table entries and encoder computation remain separate resources.

A first-order affine surrogate, built from the complete finite Fourier table, recovers majority7's best decision but collapses parity7 and the product of majorities to constant summaries. Their compressed errors become 1/2. This is an oracle finite-probe comparison, not a training result or a claim about generalization to natural activations.

## Stability without retained information

A separate rare-alert control uses no target flipping and a completely noisy terminal state, with crossover 1/2. The positive event has probability 1/128. The same Boolean readout agrees with its original output on exactly 8065/8192 of cases, approximately 98.4497%, despite zero mutual information. A constant-negative prediction has still higher accuracy, 127/128. The control exposes how target imbalance can inflate apparent stability.

## Sensitivity bound scope

For seven-bit majority the exact total influence is 35/16 and threshold degree is one; its real Fourier degree is seven. On these tiny cubes the general 8*d*sqrt(n) estimate can exceed the trivial influence bound n. The code uses exact influences for the finite calculations.

A separate 129-input majority calculation has exact influence approximately 9.079806. The #127 parameter bound is approximately 90.862534, below the trivial 129. This demonstrates a nonvacuous finite point and also the looseness of the general constant. Its influence is obtained from the exact binomial formula, not by enumerating 2^129 inputs.

## Verification and limits

The 22-test suite includes all 256 three-bit truth tables, an independent full-transition oracle, Walsh/Fourier reconstruction, coordinate sharpness, an exhaustive small randomized-channel grid, approximate-readout error bounds, and posterior-threshold separation. CI uploaded both source-hashed machine-readable reports.

The separate [MathCan PR #36](https://github.com/patrickdugan/MathCan/pull/36) checks finite noise recurrence and horizon-cutoff arithmetic plus a threshold/decoder separation lemma. Its proof scope excludes the original source theorems and the full probabilistic derivations.

This round evaluates complete synthetic Boolean models. Applying its horizon result to a recurrent checkpoint requires validating the uniform state law, independent symmetric corruption, binary target definition, and absence of retained source history. A cached original target can be stored perfectly in a single bit and changes the observation interface.
