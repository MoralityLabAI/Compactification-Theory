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

## OIC-18: n-scaled phase-localizing emissions, executed

The [OIC-18 report](OIC18_REPORT.md), [OIC-T26–T28 derivations](OIC18_THEORY.md), and [locked preregistration](OIC18_LOCK.md) extend the OIC-16 unknown-dynamics observer to n-dependent continuous phase emission. **60 new CPU-trained models** (five seeds, clean/fault-augmented sensors, full/heldout initial phases, three emissions) are compared with 20 frozen original Fourier fits, all on the matched 500-step × 128-example training budget.

On clean heldout-phase training, the n²-localized temperature and von Mises emissions achieve 100% balanced family classification at the previously unseen n=24 and n=64; the original Fourier model scores 59.6% and 52.2%, respectively. The equal-capacity third-harmonic control retains significant large-n errors. The learned stochastic filters can distinguish the *model family* while retaining an inaccurate or diffuse rotating generator, and fault-augmented training substantially improves the worst-case sensor-position sweeps while leaving a small number of substitution failures. Interpret class identification, phase reconstruction and sensor fault robustness separately.

Every raw-sample substitution and marked-erasure position was tested for even n=4,6,8,12,16,24,32,64, across both families and all initial phases: **1,610,400 fault episodes**. The standard-library [independent verifier](oic18_verify.py) reconstructs selected model posteriors from archived weights, audits all stored scores and receipts, and checks source hashes and gauge controls. Eight additional [unit tests](test_oic18_math.py) run with and without Python optimizations. Machine-readable [manifest](results/oic18_summary.json) indexes the four model and four fault archives and the frozen baseline receipt.

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic18*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic18*.py' -v
python research/observational_inwardness_v1/oic18_verify.py --check
# Requires PyTorch / NumPy, high CPU cost:
cd research/observational_inwardness_v1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python oic18_train.py --out results/oic18_retrain.json
~~~

## OIC-19: Exact observable-shift classes and predictive state recovery

The [executed OIC-19 experiment](OIC19_REPORT.md) contains 30 new fits and 10 frozen OIC-18 von Mises baselines. An observable raw-next-state predictive likelihood recovers the registered ±1 rotating shift in all five seeds and both training splits, with more than 99.7% mean correct-step kernel mass and 100% clean model-family and gauge-relative phase-MAP accuracy on unseen n=24,64. Stronger entropy regularization converges to a wrong ±2 shift while retaining family identification; a simulator-phase-supervised ceiling succeeds in all full-phase seeds and three of five held-out seeds.

[OIC-T26](OIC19_THEORY.md) proves that for uniform hidden initial phase, deterministic shifts have the same observable-history law exactly when they have the same gcd(n,d). Generator recovery is therefore relative to the restricted model class and the observational gauge.

The new raw-sensor fault receipt preserves **200,960** single-substitution and erasure episodes for the ten predictive fits. Mean held-out accuracy after one substituted raw sample is 73.07% at n=8, 99.69% at n=24, and 99.95% at n=64. Every tested single erasure is classified correctly; the exact codebook retains the worst-case correction guarantee.

- [Mathematical note](OIC19_THEORY.md)
- [Experiment report](OIC19_REPORT.md)
- [Trained weights, scores and pinned source hashes](results/oic19_summary.json)
- [Independent pure-Python verifier](oic19_verify.py)
- [Prospective OIC-20 fault-aware predictive protocol](OIC20_PROTOCOL.md)

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic19*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic19*.py' -v
python research/observational_inwardness_v1/oic19_verify.py
~~~


## OIC-20: Minimum labeled sensors for identifying hidden generators

[OIC-T27–T30](OIC20_THEORY.md) extend the OIC-T26 gcd-class result to a
new, explicitly resource-charged sensor interface. An active observation
reports separately labeled hits at registered marker positions in
Z_n. Two markers at 0 and k identify a modular shift by its gcd and
the relative hit-time lag when the markers share an orbit.
For prime power n=p^a, the exact minimum number of labeled sensors
needed to identify **every** shift is a+1 when p is odd, and a when p=2.

The [executable exact sensor search](oic20_markers.py) enumerates full
n-step observation distributions and proves minimum label counts by
exhaustion for n=2..16. Notably n=6 needs only markers (0,2), n=8
needs three markers, and n=16 needs four. Ten unit tests pass in both
Python modes; CI checks the [pinned deterministic receipt](results/oic20_reference.json).
This is a finite exact sensor-design result; it does not supply new
observations to the original one-marker OIC-19 trained models.

The [OIC-20 research report](OIC20_REPORT.md) records the full finite
enumeration. [OIC-21](OIC21_PROTOCOL.md) preregisters the next
cost-aware learned sensor-selection experiment.

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic20_markers.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic20_markers.py' -v
python research/observational_inwardness_v1/oic20_run.py --check
~~~


## OIC-21: Adaptive single-marker sensing, executed CPU pilot

The [OIC-21 report](OIC21_REPORT.md) records 10 trained **oracle-distilled** 145-parameter sensor-query policies, with five seeds each for full-phase and held-out-phase training at n=4–6. Every shift and hidden initial phase was evaluated at n=4,5,6,7,8,12,16,24,32 under a budget of n+2 one-bit marker queries.

The [OIC-T31–T35 mathematics](OIC21_THEORY.md) establishes the exact finite-horizon Bayes decision-tree recursion, a no-hit lower bound for reliable shift identification, a strict five-query adaptive advantage, and a counterexample separating **observation-law identifiability** from identification on one trajectory. Exact computations find full-identification budgets 2n−2 for n=2–6; the general claim is an open conjecture.

At unseen n=24, the learned held-out-phase policy averages **72.53% shift accuracy** using 26 returned sensor bits; the 64 randomized fixed policies average 28.69% using the same number of returned bits. The exact posterior enumerates all n² latent shift/phase candidates, and the learned component is only the sensor-address scorer. Sending the 26 adaptive sensor addresses also costs up to **130 control bits**, so this experiment does not establish a total communication saving.

All-position single-fault tests across n=4,6,8, five seeds and two splits contain **21,640 evaluated clean/flip/erasure episodes**. At n=8 the held-out controller attains 81.56% clean shift accuracy but only 48.94% with one adversarially substituted reported bit. Faults change later sensor choices. [OIC-22](OIC22_PROTOCOL.md) registered the subsequent fault-aware experiment, now executed and reported below.

- [Exact finite oracle](oic21_exact.py) and [policy/scoring reference](oic21_policy.py)
- [CPU training source](oic21_train.py) and [benchmark script](oic21_bench.py)
- [Online sensor corruption model](oic21_faults.py) and [fault batch runner](oic21_fault_bench.py)
- [Frozen source/weight and result hashes](results/oic21_manifest.json), [compressed ten-model receipts](results/oic21_fits.json.gz.b64), [clean benchmark](results/oic21_bench.json), and [fault panels](results/oic21_faults.json)
- [Independent standard-library integrity/replay verifier](oic21_verify.py)

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic21_*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic21_*.py' -v
python research/observational_inwardness_v1/oic21_verify.py --check
# Optional CPU PyTorch retraining of a single seed:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python research/observational_inwardness_v1/oic21_train.py --seed 0 --split heldout --out /tmp/oic21_heldout_0.json
~~~

**Evidence scope:** exact finite calculations and a supervised synthetic CPU policy pilot. The exact hypothesis belief and oracle action labels are privileged structural knowledge; the model does not infer unknown physical dynamics from scratch. Sensor faults are evaluated end-to-end under the declared likelihood model.


## OIC-22: fault-aware active sensing, executed negative pilot

[OIC22_REPORT.md](OIC22_REPORT.md) records ten **145-parameter** trained
sensor scorers (five seeds, full and held-out phase splits), trained with
500 Adam steps and batches of 96. The cyclic hypothesis transition table
remains supplied as structural knowledge. A two-mask posterior now accounts
exactly for zero or one sensor-report substitution. The policy uses a stable
four-decimal score tie break. The teacher combines exact joint Bayes planning
on the n=4,5 **optimal-policy trees** with frozen OIC-21 policy supervision
off-tree and at n=6; the additional teacher computation is disclosed.

Across the 21,640 registered n=4,6,8 population episodes, the held-out
policy obtains **48.26%** shift accuracy with one substitution at n=6
versus OIC-21's 43.75%, and **50.50%** at n=8 versus 48.94%.
Clean identification remains within the registered tolerance.
**The ten-percentage-point adversarial improvement gate fails.**

[OIC22_THEORY.md](OIC22_THEORY.md) gives exact one-flip, one-erasure
and mixed clean/flip Bayesian recursions on small finite systems,
with a new necessary lower bound for universal full-shift recovery:

\[
\boxed{Q\ge n(e+1)+r}
\]

for up to e substitutions and r marked erasures. Thus the previous
Q=n+2 benchmark cannot provide a worst-case one-flip certificate at n>=4.

- [OIC-22 exact Bellman solver](oic22_exact.py) and [finite replay](oic22_reference.py)
- [Two-mask Bayesian monitor](oic22_filter.py) and [CPU training](oic22_train.py)
- [Independent Python-only receipt verifier](oic22_verify.py)
- [Saved training weights and evaluation counts](results/oic22_fits.json.gz.b64), [aggregate scores and pinned hashes](results/oic22_summary.json)
- [Optional four-step receding planner](oic22_lookahead.py) and [pinned exploratory comparison](results/oic22_lookahead.json)
- [OIC23_PROTOCOL.md](OIC23_PROTOCOL.md): prospective certified query horizons starting at the necessary information lower bound

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic22*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic22*.py' -v
python research/observational_inwardness_v1/oic22_reference.py --check
python research/observational_inwardness_v1/oic22_verify.py --check
python research/observational_inwardness_v1/oic22_verify.py --deep
~~~

The source hashes and 90 panel totals have been checked by an independent
scalar implementation. The results are synthetic and establish no measured
transformer self-model, consciousness or cosmological mechanism.


## OIC-23: exact robust adaptive-query certificates

[OIC23_REPORT.md](OIC23_REPORT.md) and [OIC23_THEORY.md](OIC23_THEORY.md)
establish **complete finite** robust query thresholds for unknown shift and initial phase on
cycles n=2,3,4. For one substituted report the exact minimum queries are 6,8,10;
for one marked erasure they are 4,5,7; for one of each they are 8,9,12.
These are exhaustive finite results, with an open conjecture Q*(n)=2n+2
for one substitution at n>=5. The n=5 exploratory run hit a declared
300,000-node cap, so its status is inconclusive.

At n=4,Q=10, a verified adaptive policy always recovers the shift despite
one substituted report. Exhaustive enumeration of all 4^9=262144 fixed
schedules after phase-translation reduction proves that each has
minimum cross-shift Hamming distance <=2, insufficient to correct one
substitution. This gives a strict robust adaptive-versus-fixed separation
at equal ten-report budget, while adaptive addresses also require online
address communication.

- [Exact Boolean solver](oic23_exact.py) and
  [independent certificate generator/replayer](oic23_reference.py)
- [Seven finite tests](test_oic23_exact.py) for ordinary and optimized Python
- [Pinned manifest](results/oic23_reference.json) and
  [nine compressed query trees](results/oic23_certificates.json.gz.b64)
- [OIC-24 protocol](OIC24_PROTOCOL.md) for larger cycles and a stronger lower bound

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic23_exact.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic23_exact.py' -v
python research/observational_inwardness_v1/oic23_reference.py --check
~~~

The claims concern a synthetic exact finite sensor game. The source, oracle
and certificate do not establish any learned-model or physical cosmology result.


## OIC-24: phase-translation quotient and five-phase robust thresholds

The [OIC-24 note](OIC24_NOTE.md) proves that simultaneous translation of
hidden initial phases and sensor addresses preserves every admissible
report, including substitutions and marked erasures. A verified
symmetry-reduced Boolean oracle applies this equivalence to policy search.

At n=5, the **minimum** worst-case adaptive query budget with up to
one marked erasure is exactly nine. The ordinary unquotiented oracle
rules out eight queries, while the nine-query adaptive tree passes
independent replay on all 174 true-state/fault paths.

For one substituted report, complete symmetry-quotient searches rule out
all horizons through eleven queries, establishing Q*_flip(5)>=12.
The Q=12 attempt hit a declared 700,000-state cap; its feasibility is open.

- [Exact symmetry-aware oracle](oic24_sym.py)
- [Canonical certificate replay](oic24_run.py)
- [Five symmetry and integrity tests](test_oic24_sym.py)
- [Pinned reference](results/oic24_reference.json) and
  [nine-query erasure policy](results/oic24_n5_erasure_policy.json.gz.b64)

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic24_sym.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic24_sym.py' -v
python research/observational_inwardness_v1/oic24_run.py --check
~~~


## OIC-25: exact five-phase one-substitution threshold

The [OIC-25 proof and computer-assisted certificate](OIC25_THEORY.md)
resolves OIC-24's formerly capped five-phase search. With one addressed
binary report per step, unknown (shift, initial phase), and at most one
substituted report, **the exact finite adaptive horizon is Q*=14**.
A faster C++ Boolean game and a separately written Python dynamic
program agree that Q=13 is impossible; the C++ negative search was
also rerun without the new pairwise pruning rule.

At Q=14 both implementations construct the same complete 662-node
winning tree. OIC-23's independent verifier checks all 327
true-world/adversarial report paths. This refutes the earlier
conjectured Q*(n)=2n+2 pattern at n=5.

- [OIC-T44 pairwise residual-disagreement theorem and OIC-T45 exact finite result](OIC25_THEORY.md)
- [Executed search report](OIC25_REPORT.md)
- [Fast C++ search](oic25_fast.cpp) and [independent Python search](oic25_independent.py)
- [Pinned source, search and policy replay](oic25_reference.py) and [tests](test_oic25_exact.py)
- [Finite search summary](results/oic25_reference.json) and [compressed winning tree](results/oic25_n5_q14_tree.json.gz.b64)
- [OIC-26 protocol](OIC26_PROTOCOL.md) on general lower bounds and n=6

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic25_exact.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic25_exact.py' -v
python research/observational_inwardness_v1/oic25_reference.py --check
python research/observational_inwardness_v1/oic25_reference.py --cpp-check
~~~

The threshold is a computer-assisted finite-state theorem under the
declared deterministic sensor and adversarial fault rules. It does not
provide a trained-model, consciousness or physical-cosmology result.


## OIC-26: Exact six-phase one-erasure query threshold

The [OIC-26 executed report](OIC26_REPORT.md) resolves the
six-phase **one marked erasure** case with an exact minimum
of **eleven adaptive, one-bit marker queries**. An unquotiented
finite search rules out ten queries after examining 2,223,374 states;
the translation-quotient search independently rules out ten
after 370,580 states.

At eleven queries, the winning 753-node policy passes
OIC-23's independent checker and 331 true-world fault
replays. The empirical pattern Q*_erasure(n)=2n−1 now
holds for n=3,4,5,6. It is a conjecture for arbitrary n.

- [Executable exact reference](oic26_erasure.py)
- [Independent tests](test_oic26_erasure.py)
- [Pinned negative-search receipt](results/oic26_reference.json)
- [Complete eleven-query winning tree](results/oic26_n6_erasure_tree.json.gz.b64)

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic26_erasure.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic26_erasure.py' -v
python research/observational_inwardness_v1/oic26_erasure.py --check
python research/observational_inwardness_v1/oic26_erasure.py --recompute
~~~

The [OIC-26 protocol](OIC26_PROTOCOL.md) still governs the
open six-phase one-substitution problem. The erasure
extension is an exact synthetic result with no Lean kernel proof.


## OIC-29: Seven-phase search audit and fixed-sensor codes

The [OIC-29 report](OIC29_REPORT.md) records exact-state search caps for the unresolved seven-phase adaptive horizons Q15–19. Its independently checked fixed-sensor reference constructs a **22-query, three-marker** policy and a **27-query, two-marker** policy, each correcting any one substituted report bit. [OIC-T55–T57](OIC29_THEORY.md) prove the coding criterion and account for online addressing versus offline preprogramming. [OIC-30](OIC30_PROTOCOL.md) targets a proof-carrying adaptive threshold and fixed Q20/Q21 infeasibility certificates.

```sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic29_fixed.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic29_fixed.py' -v
python research/observational_inwardness_v1/oic29_fixed.py --check
```


## OIC-30: prime-cycle error-correcting sensor blocks

[OIC-30's executed exact report](OIC30_REPORT.md) supersedes the earlier
**22-query three-marker** fixed upper bound at seven hidden phases.
The new **21-query, three-marker** schedule is:

~~~text
0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2
~~~

Its minimum cross-shift Hamming distance is exactly **3**, as independently
checked across all 1,029 candidate pairs and 1,078 zero-or-one-substitution
report paths. The first-episode literal preprogramming budget is **72 bits**
(21 sensor responses, 9 marker-address bits and 42 schedule-description bits).
[OIC-T58–T61](OIC30_THEORY.md) prove a general prime-cycle block code,
a two-marker obstruction, a sharp query lower bound for fixed schedules
using fewer than all n labels, and an exact phase-residue occupancy reduction.
All 16,384 normalized two-marker fixed Q21 schedules are covered by an
analytic obstruction and an exhaustive finite check.

The **global** fixed Q20 case with access to all seven sensor addresses is
still unresolved, as are adaptive horizons Q15–19. The next
[OIC-31 protocol](OIC31_PROTOCOL.md) requests an independently checkable
unrestricted Q20 integer infeasibility certificate or verified positive
schedule, plus a proof-carrying adaptive Q19 search. The optional
[oic30_reduced_milp.py](oic30_reduced_milp.py) uses SciPy/HiGHS for
exploratory occupancy searches; it is excluded from lightweight CI.

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic30_prime_blocks.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic30_prime_blocks.py' -v
python research/observational_inwardness_v1/oic30_run.py --check
~~~


**Unrestricted Q20 audit:** the 49-variable occupancy MILP with all
seven addresses has now returned HiGHS INFEASIBLE; see
[the pinned solver status](results/oic30_q20_solver_status.json).
This remains solver-reported evidence until an independently checkable
integer-unsatisfiability certificate is produced.


## OIC-31: exact mixed substitution/erasure bounds with proper marker subsets

[OIC-T62/T63](OIC31_THEORY.md) extends the OIC-30 prime-cycle construction
to e adversarial substituted sensor reports and r marked erasures.
Any fixed schedule on n hidden phases which **omits at least one marker**
requires Q >= n(2e+r+1) queries. For a prime p with
2 <= 2e+r+1 < p, a block code that uses 2e+r+1 different labels,
each queried for p steps, attains this bound exactly.

A seven-phase example correcting one substitution plus one erasure needs
28 queries and four markers in this resource class. The
[executed report](OIC31_REPORT.md), [integer reference](oic31_mixed.py),
[six tests](test_oic31_mixed.py), and
[pinned receipt](results/oic31_reference.json) are lightweight and
dependency-free; [OIC-32](OIC32_PROTOCOL.md) specifies the remaining
unrestricted fixed Q20 certificate and adaptive Q19 tasks.

    python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic31_mixed.py' -v
    python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic31_mixed.py' -v
    python research/observational_inwardness_v1/oic31_run.py --check

**Scope:** finite synthetic sensing. Full seven-marker Q20 feasibility has
only solver-reported infeasibility, so global fixed minimum still requires
a checkable certificate. The exact adaptive seven-phase interval is 15–20.


## OIC-32: Global fixed-schedule optimum for prime cycles

[OIC-T64/T65](OIC32_THEORY.md) prove a **global** optimal fixed one-substitution query horizon of `Q=3p` for every prime `p>=5`, with all `p` sensor addresses permitted. This settles the seven-phase unrestricted Q20 feasibility gap left open in OIC-30/31: 20 queries are analytically impossible and the existing three-marker 21-query schedule attains the exact optimum. The proof uses the second factorial moment of affine-line query weights and independently handles repeated phase-residue measurements; it does **not** rely on HiGHS or an SMT UNSAT claim.

- [Proof and incidence identity](OIC32_THEORY.md)
- [Research report](OIC32_REPORT.md)
- [Standalone integer reference](oic32_global.py), [seven tests](test_oic32_global.py) and [pinned receipt](results/oic32_reference.json)
- [Next adaptive research protocol](OIC33_PROTOCOL.md): the n=7 adaptive minimum remains in the certified interval 15–20

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic32_global.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic32_global.py' -v
python research/observational_inwardness_v1/oic32_run.py --check
~~~


**OIC-T66 small-prime classification:** exhaustive fixed-schedule enumeration establishes Q=6 for p=2 and Q=8 for p=3. Combined with OIC-T64, the global prime-size rule is 6, 8, and 3p for p≥5. See [small-prime exact enumerator](oic32_small.py), [tests](test_oic32_small.py), and [pinned receipt](results/oic32_exceptional.json).


**OIC-T68/T69 global one-erasure theorem:** for every prime p≥2, the exact unrestricted fixed minimum is Q=2p queries with one marked erasure. For p≥5, an affine-plane line-pair moment contradiction rules out 2p−1 queries even when every marker is addressable. Exhaustive p=2,3 checks close the small cases. See [the proof](OIC32_THEORY.md), [pure-Python reference](oic32_erasure.py), [unit tests](test_oic32_erasure.py), and [pinned receipt](results/oic32_erasure.json).
