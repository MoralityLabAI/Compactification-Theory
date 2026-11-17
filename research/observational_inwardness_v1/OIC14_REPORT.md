# OIC-14: Learnable cyclic phase dynamics, optimizer basins and lattice locking

**Completed CPU pilot: November 10, 2026.** Ninety paired experiments, **180 trained models**, five seeds each, three phase counts, two training splits, three architectures and two target dynamical families. Full emitted traces and training parameters are in the hash-pinned compressed receipt. Independent standard-library CI reconstructs the oracle and checks all recorded sequences. This run concerns synthetic supervised recurrence with a privileged initial hidden-phase encoding.

## Question

OIC-12's generic GRU often failed on withheld initial phases. OIC-13's reflection-tied model generalized well while using a *hard-coded* exact clock. Here the clock transition itself is learnable. The primary question is whether exact SO(2) equivariance helps infer the hidden recurrence from observable outputs, and how the optimization strategy affects reliability.

## Experimental contract

For `n=4,5,6`, both target families use the same task `x' = x XOR 1 XOR (u AND [z=0])`. Fixed targets retain `z`, rotating targets increment `z mod n`. The learner receives the Fourier encoding of `z0`, known n, and the XOR-visible rule, and learns a phase update and a thresholded phase-zero event readout. The monitor receives emitted raw visible samples only.

- **Orthogonal local**: `h_(t+1)=R(omega)h_t`, train `omega`, readout gain and bias (3 parameters); initialize `omega=π/n` for both target families.
- **Orthogonal grid**: same three parameters, plus a **4n-candidate** initial likelihood scan on the first training batch, followed by all 350 Adam steps; the scan's angular candidates include the true group shifts but receive only visible-state supervision.
- **Unconstrained**: arbitrary learned `2x2` transition and the same readout (6 parameters); initialize at `R(π/n)`.

Matched conditions: 350 Adam steps, batch size 96, learning rate 0.014, training length `2n+6`, independent Bernoulli(0.7) active probes, matched five seeds and phase holdout `{1,n-2}`. Training uses teacher-forced `x_t`, while evaluation feeds each predicted binary `x_t` into the next visible update. The model update depends on the supplied initial phase and learned parameters; the visible XOR gate is given explicitly. Six fixed evaluation schedules are reused from OIC-12 and OIC-13.

The models have different parameter counts and the grid uses additional candidate inference. This is a **matched-gradient-step comparison**, not a compute-matched architecture comparison.

## Measured held-out latch results

The `latch` schedule uses `n` consecutive probes followed by `n` passive raw-state measurements. Whole-trajectory accuracy pools five seeds and all `n` initial phases across both target families.

| Architecture | n=4 | n=5 | n=6 |
|---|---:|---:|---:|
| Orthogonal local | 25.0% | 30.0% | 33.3% |
| Orthogonal grid | **100.0%** | **100.0%** | **100.0%** |
| Unconstrained | 100.0% | 90.0% | 91.7% |

Each orthogonal-grid held-out run matched the exact oracle's minimum cross-family Hamming distance: `5/5` at n=4, `2/2` at n=5 and `7/7` at n=6, in all five seeds. Every locally initialized model had minimum cross-family distance zero on that schedule. The unconstrained held-out distances were `5,5,5,5,5` (n=4), `2,2,2,2,1` (n=5), and `2,2,2,2,2` (n=6).

With full initial-phase training, orthogonal-grid and unconstrained models attained 100% on the registered latch schedule, while local angular optimization remained inaccurate. This makes optimization initialization a central confounder in any simple symmetry-versus-no-symmetry comparison.

## Out-of-window observations

A replay from the archived fitted parameters tests the first wrong *phase-event* prediction under **continuous active probing**, checking at most 600 transitions. This additional test uses float64 reconstruction of the stored parameters, rounded to eight decimal places. It reports a numerical stress test, not a floating-point formal certificate.

Among the 10 held-out orthogonal-grid fitted models per phase count (5 seeds x 2 target families), the earliest phase-event errors were at transitions 16 (n=4), 8 (n=5) and 6 (n=6). Some errors occur during the passive tail of the registered latch schedule, where the event detector's output is masked by `u=0`. Consequently, **100% latch accuracy does not entail accurate continuously probed dynamics at the same horizon**. For the held-out grid models, whole-sequence accuracy on the registered `always_probe` schedule was 100% (n=4), 90% (n=5) and 91.7% (n=6), pooling model families and seeds.

This is the expected failure mode of a continuously learned angle: small per-step phase error accumulates with recurrence depth. OIC-T16 derives a sufficient logit-margin bound for its detection.

## Finite-group post-training projection

For every learned orthogonal model we projected the trained angle onto the nearest generator of the known cyclic group, `q=round(n*omega/(2π)) mod n`. We independently checked whether the learned threshold readout classified all `n` discrete phases correctly, then checked the selected group step against the reference target.

| Model | Full-split fits satisfying both conditions | Held-out-split fits satisfying both |
|---|---:|---:|
| Orthogonal grid | 30/30 | **30/30** |
| Orthogonal local | 30/30 | **25/30** |

OIC-T15 proves that, when the conditions hold, the projected **finite-state** implementation exactly matches the target for every subsequent action word and every initial hidden phase, at any horizon. For the grid-trained model all 60 fitted systems satisfy the conditions; five held-out local fixed-phase models select the incorrect group step.

Projection uses known `n` and discrete phase-update arithmetic. The true target step is inspected only to score the result. This is a powerful structural prior and provides a candidate route to compact, auditable recurrent controllers. It does not show that the network independently discovered the cyclic group.

## Exact theoretical claims and verification

[OIC-T14–T17](OIC14_THEORY.md) prove: rotation equivariance and norm preservation, the group retraction sufficiency condition, a pre-projection angular drift bound, and fixed-point precision margins. The implementation includes an independent receipt validator, finite oracle reconstruction, parameter replay, deliberately altered-record rejection tests, and small exact group-action checks.

The code in `learnable_clock.py` and `projection.py` is SHA-256 pinned in `results/oic14_summary.json`. The compressed `results/oic14_compact_full.json.gz.b64` contains all 90 paired conditions, 180 fitted parameter records, and raw emitted traces for each of six evaluation schedules. Standard-library CI checks the saved trained outcomes without running PyTorch or reproducing the optimization procedure. Actual retraining requires the optional PyTorch and NumPy CPU environment and will depend on the pinned versions.

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python research/observational_inwardness_v1/verify_receipt.py
# Optional CPU retraining, 350 optimizer steps for each of 180 models:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/observational_inwardness_v1/learnable_clock.py --out /tmp/oic14_train.json
~~~

## Scope and follow-up

The training problem supplied the hidden phase at initialization, known `n`, a phase-zero event gate and the binary visible-state update. It tests phase-transition learning under those assumptions. A natural next pilot removes privileged phase initialization, supplies only a bounded observation history and tests whether the group structure can be inferred from active interventions. That problem needs identifiability controls before model training, since OIC-T01 proves passive non-identifiability.

No claim about consciousness, cosmological structure, phenomenal inwardness or a particular deployed AI architecture follows from these finite synthetic measurements.
