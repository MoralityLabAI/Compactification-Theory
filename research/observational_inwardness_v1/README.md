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


## OIC-13: Exact-clock structural transfer control (executed)

[Report and conditional symmetry theorem](OIC13_REPORT.md). The follow-up
executed 120 additional CPU-trained models: 2-parameter reflection-tied
and 3-parameter oriented readouts, each trained against a **hard-coded
exact modular clock and XOR visible-state rule**. Both use the OIC-12
action distribution, phase splits, five seeds, optimizer steps, and
six frozen evaluation schedules. The architecture therefore supplies
a substantial inductive prior; the experiment is a structured
control rather than evidence that a generic RNN learned latent
phase dynamics.

At n=4,5,6, every reflection-tied model reproduced every phase's
entire trajectory on all six tested schedules, including unseen
initial phases. Every heldout-oriented fixed model mislabeled
the missing z=1 phase while maintaining oracle-level **interfamily
code distance**. This shows why monitoring studies should report
both distance and trajectory fidelity.

- [CPU training/replay source](clock_pilot.py)
- [Independent standard-library receipt verifier](verify_clock.py)
- [Tamper-detection and scope tests](test_verify_clock.py)
- [Pinned manifest (120 fits, 60 pairs)](results/oic13_summary.json)
- [Full compressed trajectories and losses](results/oic13_full.json.gz.b64)

~~~sh
python research/observational_inwardness_v1/verify_clock.py --check
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
# Full CPU retraining with PyTorch and NumPy:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/observational_inwardness_v1/clock_pilot.py --out /tmp/oic13_full.json
~~~

The next experiment should learn the modular state evolution while
retaining a verified symmetry constraint, then compare observable
futures on unseen phases at matched runtime compute. The fixed
OIC-13 clock deliberately makes this part of the task exact.


## OIC-14: learned cyclic transition and algebraic projection

**Executed CPU pilot, 2026-11-10:** [OIC-14 report](OIC14_REPORT.md) and
[theorems T14–T17](OIC14_THEORY.md). The phase update is now learned
from visible-state supervision. The system still receives its initial
phase encoding and uses a built-in XOR-visible schema and anchored
phase-zero detector. These are synthetic controlled dynamics.

Ninety paired runs produced 180 fitted models across phase counts 4–6,
five seeds, full/held-out initial-phase splits and three architectures:
local angle optimization (three scalars), likelihood-grid angle
initialization then optimization (three scalars, **extra 4n forward
candidate evaluations**), and an unconstrained 2×2 transition
(six scalars). Each model receives 350 optimizer steps with matched
batch and task distribution.

Held-out full-trajectory *latch-schedule* accuracy, pooling fixed and
rotating model families, was 25%/30%/33.3% for local optimization,
**100%/100%/100%** for the likelihood grid and
100%/90%/91.7% for the unconstrained matrix (n=4/5/6).
The grid results matched the oracle raw-code separation on that schedule
for all five seeds at every phase count. The architectures are matched
in optimizer steps, **not** total inference compute or parameter count.

The grid-trained continuous clocks can still drift during long
continuously probed rollouts. A post-training projection to the nearest
Z_n generator repairs the exact finite-state recurrence whenever
the projection selects the true group step and the learned readout
classifies every discrete phase correctly. Those conditions hold
for all 60 grid-trained fits and for 55/60 locally trained fits.
The conditional all-horizon theorem concerns the **projected integer
state machine**, not floating-point loop stability.

- [Neural CPU training and evaluation](learnable_clock.py)
- [Group-retraction implementation](projection.py)
- [Long-horizon drift replay](long_horizon.py) with [frozen drift receipt](results/oic14_long_horizon.json.gz.b64)
- [Independent standard-library receipt verifier](verify_receipt.py)
- [Tampering and group-action tests](test_verify_receipt.py)
- [Hash-pinned 180-fit summary](results/oic14_summary.json)
- [Compressed replayable trajectories](results/oic14_compact_full.json.gz.b64)
- [OIC-15 phase-inference protocol](OIC15_PROTOCOL.md), prospective

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python research/observational_inwardness_v1/verify_receipt.py
# Optional CPU training with torch and numpy:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/observational_inwardness_v1/learnable_clock.py --out /tmp/oic14_training.json
~~~

**OIC-15 executed:** [OIC15_REPORT.md](OIC15_REPORT.md) removes privileged initial-phase access and evaluates 30 trained generic and cyclic observers over held-out phases and phase counts. The [OIC-T18–T22 proofs](OIC15_THEORY.md) establish permanent phase ambiguity, exact family identification, a finite likelihood ceiling, and a raw sensor robustness margin. The archived [raw-history monitoring protocol](OIC15_PROTOCOL.md) remains as the preregistered design; [OIC16_PROTOCOL.md](OIC16_PROTOCOL.md) describes the next unknown-dynamics experiment. The full model checkpoints are preserved in `results/oic15_weights.json.gz.b64`, with `results/oic15_traces.json.gz.b64`, `results/oic15_analysis.json` and `results/oic15_manifest.json` for independent verification.


## OIC-16 executed: learned cyclic dynamics without initial phase access

[OIC16_REPORT.md](OIC16_REPORT.md) documents twenty CPU-trained 14-parameter
family-conditioned circulant belief filters on matched OIC-15 synthetic episodes.
The model learns five-offset stochastic transitions and a shared Fourier emission
using public action/state histories and supervised family labels; hidden initial
phase is absent from the inference interface.

- Five seeds, two training phase splits, clean/fault-augmented training;
  n=4,5,6 for training, held-out n=7,8 for registered transfer
- Clean n-active family identification: 100% in all 20 fits on n=4,...,8;
  balanced passive prior control: 50% classification
- Raw one-sample corruption at n=8: 58.3%/66.5% for clean
  full/heldout training and 84.9%/79.3% for augmented training
- Post-hoc deterministic cyclic projection: all 20 learned kernels select
  fixed shift zero and rotating shift ±1; exact raw-codebook equivalence
  gives 100% on every tested single error/erasure at even n up to 12
- Post-hoc n=9,...,32 stress identifies failure of the fixed-bandwidth
  Fourier emission as adjacent phases become indistinguishable

[OIC16_THEORY.md](OIC16_THEORY.md) proves phase-reflection gauge
indistinguishability, projected codebook equivalence, and a quadratic
Fourier phase-resolution bound. The original
[OIC16_PROTOCOL.md](OIC16_PROTOCOL.md) records the prospective experiment
conditions; [OIC18_PROTOCOL.md](OIC18_PROTOCOL.md) registers a separate
phase-resolution transfer experiment.

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic16*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic16*.py' -v
python research/observational_inwardness_v1/oic16_verify.py --check
python research/observational_inwardness_v1/oic16_scale.py --check
# To retrain (requires CPU PyTorch and NumPy):
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python research/observational_inwardness_v1/oic16_train.py --out /tmp/oic16_retrained.json
~~~

Receipts: [compressed learned weights and predictions](results/oic16_full.json.gz.b64),
[SHA-pinned summary](results/oic16_summary.json), and
[post-hoc scale reference](results/oic16_scaling.json).
The pure-Python replay recomputes all 600 clean panels, 100
phase-relabel controls and 1,080 raw fault-position panels from saved
weights. This finite synthetic result establishes neither a neural
mechanistic-interpretability certificate nor any conclusion about consciousness.
