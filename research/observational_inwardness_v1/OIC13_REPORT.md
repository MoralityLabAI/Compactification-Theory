# OIC-13: Structural phase-clock transfer under held-out initialization

**Executed CPU comparison, 2026-11-10.** Status: 120 trained, two- or three-parameter structural-control models, 60 matched fixed/rotating pairs, each evaluated over six schedule families and every initial hidden phase. The experiment uses the same n=4,5,6, five seeds, withheld-phase split, 350 optimizer steps, 96 examples per batch, learning rate 0.014 and Bernoulli(0.7) probe distribution as OIC-12. The architecture and computation differ substantially from the 2,185-parameter GRU.

## Question

The OIC-12 GRU fit the training initial phases but often failed on held-out initial phases. Does an explicit cyclic latent clock combined with a constrained readout recover the exact trajectory dynamics on those initial phases?

## Architecture and scope

The model receives initial hidden phase z0 and visible x0, as in OIC-12. It updates its internal clock **exactly** by z_(t+1)=z_t for the fixed condition and z_(t+1)=z_t+1 mod n for the rotating condition. The visible-state update is also built in: x_(t+1)=x_t XOR 1 XOR (u_t AND estimated_pulse_t).

The **only learned component** is the pulse detector at the distinguished phase z=0:

- Reflection-tied: `sigmoid(w_cos*cos(2πz/n)+bias)` (2 trainable scalars).
- Oriented: `sigmoid(w_cos*cos(2πz/n)+w_sin*sin(2πz/n)+bias)` (3 scalars).

Models train on teacher-forced visible outputs and generate binary autoregressive outputs during evaluation, hard-thresholding the event detector. This is a **strong architecture prior**, including exact phase transition, event origin, and XOR gate. It is a privileged-input control, not a demonstration that a generic recurrent network discovered an unobservable dynamical phase.

## Exact measured results

Five fixed-model seeds and five rotating-model seeds are trained for each architecture, split and phase count. The table gives exact **whole-sequence** accuracy when evaluating the `latch` schedule (n active probes, n passive observations), across all five seeds and all initial phases. The withheld columns evaluate only the two phases omitted as starting states during training.

| n | Split | Architecture | Fixed whole trace | Rotating whole trace | Fixed withheld phases | Rotating withheld phases |
|---:|---|---|---:|---:|---:|---:|
| 4 | Full | Reflection-tied | 100% | 100% | 100% | 100% |
| 4 | Heldout | Reflection-tied | 100% | 100% | **100%** | **100%** |
| 4 | Heldout | Oriented | 75% | 100% | 50% | 100% |
| 5 | Full | Reflection-tied | 100% | 100% | 100% | 100% |
| 5 | Heldout | Reflection-tied | 100% | 100% | **100%** | **100%** |
| 5 | Heldout | Oriented | 80% | 100% | 50% | 100% |
| 6 | Full | Reflection-tied | 100% | 100% | 100% | 100% |
| 6 | Heldout | Reflection-tied | 100% | 100% | **100%** | **100%** |
| 6 | Heldout | Oriented | 83.3% | 100% | 50% | 100% |

Every full-split oriented model also achieved 100% on the latch schedule. All reflection-tied models reproduced **every** trajectory on **all six** evaluation schedules, including passive and alternate schedules. Every heldout-oriented fixed model misclassified the initial phase z=1 while preserving the other fixed phases.

The oracle minimum interfamily Hamming distance on the latch schedule is 5 for n=4, 2 for n=5 and 7 for n=6. **Both** learned architecture variants matched these minimum distances across the five seeds, despite the oriented model's systematic transition errors. Thus code distance alone does not certify fidelity to the declared transition law.

For comparison, the 2,185-parameter OIC-12 GRU trained with withheld initial phases achieved fixed/rotating **unseen-phase** whole-trace accuracies of 20%/0% (n=4), 0%/0% (n=5) and 20%/20% (n=6). Its worst-case codebook distances at n=4 were 0,0,0,0,1, below the exact distance 5.

## OIC-T13a: conditional reflection-orbit transfer

Define the reflection R(z)=-z mod n and feature c(z)=cos(2πz/n). For any function g: [-1,1] -> Y, the readout g(c(z)) obeys g(c(R(z)))=g(c(z)) for every z. Proof: cos is even. Therefore any trained readout of **only** c cannot assign different event decisions to a phase and its mirror. If a training set includes a correct event observation at R(z), the same readout decision applies at z. This is a representation identity, independent of the optimizer.

The held-out pair is {1,n-2}. For n=5 and n=6, these are mirrors of phases n-1 and 2, respectively, both present in the training split. The readout equivalence gives correct fixed-family decisions at held-out phases whenever the corresponding trained decisions are correct.

For n=4 the held-out z=2 is its own reflection. The architecture's **linear cosine logit** supplies a further conditional result. Training includes phase 0 with cos=1 (positive event) and phase 3 with cos=0 (negative event). If the trained classifier correctly separates both, then w_cos+bias > 0 and bias < 0, implying w_cos > 0, and therefore `-w_cos+bias < 0` at the missing phase z=2 with cos=-1. Phase z=1 mirrors trained phase z=3. Hence classifying the two training phases correctly guarantees every held-out phase decision for n=4. The measured training fits satisfied these inequalities.

The exact rotating clock visits every modular phase within n steps from every starting state. The learned event readout is shared across time and initial phases. This allows initial-phase transfer in the rotating condition once the detector has fit all phase positions encountered during training. The finite result uses the stipulated exact internal clock.

## OIC-T13b: distance fidelity requires separate contracts

A classifier's minimum cross-family distance is a property of the **predicted** codebooks. It can equal the distance between the **oracle** codebooks even when one or more predicted trajectories disagree with their oracle counterpart. This experiment supplies finite constructive witnesses: all heldout-oriented runs have the correct latch code distance yet mispredict the fixed family on z=1. In OIC-12, a learned code distance can even exceed the oracle distance when the model generates spurious departures. Monitor evaluation must retain code distance, within-family trace correctness and class-wise worst-case errors as separate measurements.

## Reproduction and receipts

The [source](clock_pilot.py), [independent standard-library verifier](verify_clock.py), [test suite](test_verify_clock.py) and [compressed exact receipt](results/oic13_full.json.gz.b64) are committed together. The source SHA-256 and gzip/uncompressed payload SHA-256 are in [oic13_summary.json](results/oic13_summary.json). The reference verifies 120 saved fits, 60 paired runs and 12 aggregate rows. The independent code recomputes every oracle trajectory, trace-level success flag, total prediction error, cross-family Hamming distance and terminal overlap. It detects deliberate tampering with archived traces.

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_*.py' -v
python research/observational_inwardness_v1/verify_clock.py --check
# Optional CPU retraining, requires PyTorch and NumPy:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/observational_inwardness_v1/clock_pilot.py --out /tmp/oic13_full.json
~~~

The stored source was also spot-replayed at n=4, seed 0 for both heldout architecture variants and produced byte-for-byte equal run objects. This is not an independent model or a mechanistic interpretation of a pretrained system.

## Further experiment

Next, **remove the exact modular clock** from the inductive-bias controller, replacing it with a learnable norm-preserving rotation with an explicit equivariance regularizer. Use held-out initial phases, phase-label perturbations and held-out probe windows, then test whether the learned recurrence preserves phase orbits and OIC-T08's robust raw-code distance under matched compute. That asks a less privileged and more genuinely neural question.

**Evidence limits:** synthetic finite-state task, known latent phase supplied during initialization, exact transition schema hard-coded, 2/3 trainable scalars, five seeds and small n. The result makes no claim about subjective inwardness, physical cosmology or pretrained frontier models.
