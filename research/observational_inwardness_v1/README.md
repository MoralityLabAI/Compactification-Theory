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


## OIC-08–10: raw-state monitoring and parity-sensitive robustness

A new [raw-state sensor reference](raw_sensor.py) measures the *state samples*
x_0,...,x_T themselves. With e arbitrary substitutions and r marked
erasures of raw state samples, the sharp fixed-schedule result for **even**
hidden-phase count n is

\[
T^*_{\rm raw}(n,e,r)=n+2e+r.
\]

Only n active interventions are required, even for larger e and r;
the remaining 2e+r time steps passively read a latched visible difference.
In contrast, the OIC-06 audit channel reports corrupted transition
*events*, so it has the different worst-case law
n(2e+r+1).

For odd n, the final visible state cannot universally separate the
hidden-model families for any fixed probe schedule. There are valid
whole-history probe schedules, however, and an independent finite
optimizer quantifies their cross-family distances. This is an
explicit parity effect tied to the declared raw-state interface.

Sensor error propagation is tracked directly:
\`delta(audit_t) = delta(raw_t) xor delta(raw_(t+1))\`.
Thus one interior raw sample error disturbs two adjacent audit
events, and direct raw-trace error correction can be much cheaper
than converting to a transition-audit stream.

- [OIC-T08–10 derivations](THEOREMS.md#oic-t08-sharp-robustness-law-for-corrupted-raw-state-observations-even-n)
- [Raw sensor code and exact codebook search](raw_sensor.py)
- [Raw sensor unit tests](test_raw_sensor.py)
- [Pinned finite raw sensor receipt](results/raw_reference.json)

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python research/observational_inwardness_v1/raw_run.py --check
~~~

**Model scope:** one finite synthetic latent state machine, nonadaptive
binary interventions, known raw-sample corruption budgets and exact
integer arithmetic. No neural, cosmological or consciousness claim.

## OIC-11 preregistered recurrent monitor test

The [OIC-11 protocol](EXPERIMENT.md) specifies exact, learned toy-recurrence and pretrained-recurrence tiers, phase-odd/even controls, raw/event sensor comparisons, matched compute and readout costs, and preregistered neural latch failure criteria. Only the finite exact tier has been implemented. Learned and pretrained results are prospective.


## OIC-12: first executed learned recurrence pilot

The [OIC-12 report](OIC12_REPORT.md) describes a CPU experiment on **60
trained GRUs**, paired across n=4,5,6, five seeds and two training splits.
Each model has 2,185 parameters and a 24-dimensional recurrent state.
The model is initialized from a *privileged Fourier encoding* of the true
hidden phase; the monitor sees only raw visible outputs.

When every initial phase was represented in training, the five-seed
exact latch-trajectory accuracy was 100% for n=4 and n=5 and
96.7% fixed / 100% rotating for n=6. With two phases withheld,
the corresponding full-sequence accuracies were **60%/50%** at n=4,
**60%/60%** at n=5 and **73.3%/73.3%** at n=6. For the
heldout split, trained-phase trajectory accuracy remained 100%;
unseen starting phases often failed. Model errors sometimes
*increased* apparent raw-trace code distance, demonstrating why
raw distance must be audited alongside transition fidelity.

- [Exact executed training source](learned_pilot.py), requiring
  PyTorch 2.10.0+cpu and NumPy 2.3.5 for pinned reproduction
- [60-model compressed rollout receipt](results/learned_pilot_full.json.gz.b64)
- [Human-readable metrics and hashes](results/learned_pilot_summary.json)
- [Independent standard-library receipt verifier](verify_learned.py)
  and [tampering tests](test_verify_learned.py)

The frozen receipt is checked in lightweight CI without installing
PyTorch. Retaining the ability to *retrain* the 60 models requires
an explicit CPU replay using the pinned dependencies, since the
archived receipt contains trajectories and training losses rather
than model weight checkpoints.

~~~sh
python research/observational_inwardness_v1/verify_learned.py --check
# Optional CPU reproduction with torch + numpy installed:
python research/observational_inwardness_v1/learned_pilot.py --n 4 --splits full --seeds 0 1 2 3 4 --out /tmp/oic12_n4_full.json
~~~

This experiment measures one synthetic supervised recurrent
model. The preregistered [OIC-11 protocol](EXPERIMENT.md) remains
intact; a pretrained looped-layer model study has not been run.
