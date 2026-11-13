# OIC-12: held-out initial phases in a trained recurrent monitor

**Executed CPU pilot, November 10, 2026.** This experiment tests how the
exact hidden-cycle monitoring results of OIC-01–10 transfer to a small
trainable recurrent model. Its strongest observation is a failure of
**initial-phase transfer** under withheld starting conditions, even after
teacher-forced training loss becomes small.

## Method

Two separate GRU models are trained for each hidden-phase count n=4,5,6:
the fixed and rotating families of the exact finite oracle.
They share an architecture and training budget, with 2,185 trainable
parameters per model and hidden width 24. The initial hidden phase z0
is provided to the *model* via first and second Fourier harmonics,
with an additional visible x0 coordinate. This is substantial
privileged initial-state information; the *monitor* sees only emitted
visible x0,...,xT samples. The model is not learning hidden state
solely from public observations.

Each model receives action u_t and visible x_t, and predicts visible
x_(t+1). Training uses true x_t teacher forcing, while evaluation
feeds back each binary hard-thresholded prediction. Targets are
deterministic outputs of the OIC finite oracle. Training uses
350 Adam steps, batch size 96, learning rate 0.014, Bernoulli(0.70)
random active actions, and trajectory length 2n+6. On CPU, 5
fixed seeds (0,...,4) are used, matched across both model families.
No pretraining, RL, fine tuning, or external model calls occur.

The **full** training split includes all n initial phases.
The **heldout** training split excludes z0=1 and z0=n-2, while
still training on the other initial phases. Held-out phase inputs
have the prescribed Fourier encoding but were never used as starting
conditions for training. Each run evaluates all n initial phases
for initial public x0=0. Six separate action schedules are registered:
latch (1^n 0^n), long latch (1^n 0^(2n)), always probe,
never probe, alternating, and a fixed randomly placed n-probe
budget schedule over 2n steps. The train generator samples random
Bernoulli schedules rather than optimizing against these test words.

60 trained GRUs were executed (30 paired fixed/rotating comparisons).
PyTorch and NumPy versions, exact hyperparameters, all emitted traces,
training losses and the six evaluation words are included in the
compressed full receipt. The source-code SHA is recorded in the
manifest. CI verifies the **stored evidence**, while reproducing
the actual neural training requires CPU PyTorch and NumPy.

## Measured exact whole-sequence accuracy on the latch schedule

Scores are across 5 seeds x n initial phases for each family.
An exact sequence means *every* emitted x_t agrees with the
finite-state oracle during the n-probe plus n-passive window.

| n | train initial phases | fixed GRU | rotating GRU |
|---|---|---:|---:|
| 4 | all | 100.0% | 100.0% |
| 4 | two withheld | 60.0% | 50.0% |
| 5 | all | 100.0% | 100.0% |
| 5 | two withheld | 60.0% | 60.0% |
| 6 | all | 96.7% | 100.0% |
| 6 | two withheld | 73.3% | 73.3% |

The trained-phase accuracy of every **heldout** model was 100%
for the evaluated latch schedule. Withheld-phase accuracy was:

| n | fixed on unseen phases | rotating on unseen phases |
|---|---:|---:|
| 4 | 20.0% | 0.0% |
| 5 | 0.0% | 0.0% |
| 6 | 20.0% | 20.0% |

Consequently, the generalization deficit is concentrated on
unseen *initial* conditions, rather than all visible trajectories.

## Minimax codebook separability

Each row lists the minimum Hamming distance between any raw
visible-state trace produced by the two trained families,
minimized over every initial phase in each family. Larger distance
permits more bounded raw-state substitutions. The exact-oracle
reference on the same latch action word is d=n+1 for even n.

| n | training split | measured minimum interfamily distances (seeds 0–4) | exact oracle |
|---|---|---|---|
| 4 | full | 5, 5, 5, 5, 5 | 5 |
| 4 | heldout | 0, 0, 0, 0, 1 | 5 |
| 5 | full | 2, 2, 2, 2, 2 | 2 |
| 5 | heldout | 0, 0, 0, 0, 0 | 2 |
| 6 | full | 7, 7, 6, 7, 7 | 7 |
| 6 | heldout | 1, 1, 0, 0, 8 | 7 |

**Important interpretation:** The observed heldout n=6 seed-4
distance 8 exceeds the exact oracle's 7 because this model
deviates from the declared physical transition law. Spurious
model errors can produce a *larger* apparent code distance.
Distance by itself does not certify that a learned model
faithfully implements the target recurrence. Always report
both code distance and per-phase trace fidelity.

At n=4, four of five heldout paired models admit at least
one identical visible trajectory between the families.
At n=5, all five do. The n=6 heldout distances remain
highly variable across seeds; one exceeds the exact oracle.
These values are exact for the saved finite model outputs
and limited to the tested seeds, schedules, and architecture.

## Independent receipt validation

The repository includes:
- a replayable PyTorch training source, with the exact source
  SHA recorded in the receipt;
- the full CPU training receipt archived as deterministic gzip
  and base64 (30 pairs, all trajectories);
- a human-readable JSON manifest with the six-row summary;
- a pure-standard-library independent verifier that recomputes
  each oracle trace from the declared finite state equations,
  each stored success flag, error count, code distance,
  terminal overlap and aggregate score;
- lightweight CI tests including deliberate receipt tampering.

The verifier recomputes metrics from archived emitted trajectories.
It **does not** retrain or establish that the archived model weights
derive from the stated optimizer. For an additional assurance,
rerun the matching CPU training source and compare emitted traces
under the pinned PyTorch/NumPy versions.

## Meaning and follow-up

OIC-T08 establishes an exact sensor-identification result for a
specified dynamics. A learned transition approximation must earn
that property on *every relevant starting phase* if the worst-case
robustness guarantee is to transfer. This pilot demonstrates an
empirical barrier in the tested architecture. It supports a
follow-up with a **cyclic-equivariant phase initialization and
transition constraint**, matched training budget, and heldout
initial phase split. An equivariant construction would make
the role of structural priors measurable.

These findings are about supervised toy recurrent models.
There are no claims about introspection, subjective experience,
cosmological physical structure, or deployed frontier model behavior.
