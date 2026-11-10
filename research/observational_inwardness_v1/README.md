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
