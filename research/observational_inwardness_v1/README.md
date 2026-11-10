# OIC-01 | Observational inwardness, hidden topology, and probe cost

**Finite exact reference, 2026-11-10.** Inspired by M. Alan Kazlev's question
about intrinsic structure behind observable physics. Here the word inwardness
has a deliberately narrow meaning: inaccessible states and transition structure
relative to a declared sensor and action interface. The model makes no
consciousness, spiritual or cosmological inference.

A pair of deterministic models each has 2n states and identical externally
visible traces at every passive horizon. Their transition multigraphs have
different cycle decompositions and first Betti numbers. The declared probe
that couples one hidden phase to the visible readout distinguishes the
models in **exactly n action opportunities** in the worst case, even for
adaptive experimenters.

The separate triangle-context example demonstrates a nonzero first
cohomology class obstructing global truth assignments. It is logically
independent of hidden-state non-identifiability.

See [THEOREMS.md](THEOREMS.md) for definitions, proofs and limits.
[model.py](model.py) uses only Python standard-library integer arithmetic.
[results/reference.json](results/reference.json) records a deterministic
full-state enumeration for phase counts 3 through 8.

## Reproduce

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python research/observational_inwardness_v1/run.py --check
~~~

Existing adjacent work: [6GW-T01](../../6GW/THEORY.md),
[LSV-10](../levin_sieve_v1/pretrained_trm/GOAL_HORIZON.md),
[RC-01](../recurrent_circuits_v1/THEORY.md), and
[intervention-preserving compactification](../../note/intervention-preserving-compactification.md).
The new calculation concerns topology lost under a quotient and the worst-case
cost of restoring distinguishing power with an explicit intervention.

## Provenance

- M. Alan Kazlev, [The Lost Insight of Cosmology](https://cosmicevolution.substack.com/p/the-lost-insight-of-cosmology), 2026-11-10.
- Abramsky and Brandenburger, [The Sheaf-Theoretic Structure of Non-Locality and Contextuality](https://arxiv.org/abs/1102.0264), 2011.
- Classical nonminimal realization and Kalman observability are the mathematical backdrop; the general mechanisms are established work.

**Evidence scope:** written elementary proofs, exhaustive finite Python checks,
and pinned replay. There are no Lean kernel checks or trained-model observations.


## OIC-04: Bayesian observability curve

With equal priors on the two models and uniformly unknown hidden phase,
the optimal adaptive identification risk after T action opportunities is
1/2 for T=0 or 1; (n-T)/(2n) for 2 <= T <= n; and 0 thereafter.
Repeated probing is Bayes-optimal because it reveals the entire
hidden-zero indicator stream, which suffices to simulate any less
informative adaptive probe policy. The independently exhaustive
[Bayesian reference](bayes.py) and [tests](test_bayes.py) use exact
fractions. OIC-05 also shows that detection time can be arbitrarily
large while the passive quotient remains exactly two states.


## OIC-06: fault-tolerant audit channel

The new [OIC-T06 proof](THEOREMS.md#oic-t06-exact-errorerasure-robustness-law)
handles a monitor which receives the **single audit symbol**
b_t = probe_t * [hidden_phase_t = 0] per transition, with at most e
adversarial substitutions and r explicitly marked erasures per run.
The observer has no independent clean raw-state readout in this
scenario. For every n>=2, the exact minimax observation horizon is:

\[
T^*(n,e,r)=n(2e+r+1).
\]

This is a classical error-correcting code distance argument specialized
to the OIC pair, including an adaptive-probing impossibility proof.
[OIC-T07](THEOREMS.md#oic-t07-sharp-nonadaptive-trusted-probe-budget)
also proves a sharp active-probe count for fixed schedules.

- [robust.py](robust.py): exact codebooks, Hamming-distance oracle,
  adversarial collision witnesses, exhaustive schedule search
- [test_robust.py](test_robust.py): seven deterministic test cases
- [robust_run.py](robust_run.py): frozen receipt replay
- [results/robust_reference.json](results/robust_reference.json):
  25 finite boundary cases with n=2..6

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python research/observational_inwardness_v1/robust_run.py --check
~~~

**Interpretation boundary:** this is a synthetic audit channel with
declared integer corruption budgets, not an observed attack on a
transformer or a claim about cosmological consciousness.
