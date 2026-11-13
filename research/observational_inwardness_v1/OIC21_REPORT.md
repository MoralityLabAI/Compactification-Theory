# OIC-21 | Active single-marker sensor policies: executed exact and CPU-trained experiments

**Date:** 10 November 2026. **Status:** exact finite Bayesian oracle and 10 CPU-trained oracle-distilled query scorers executed; independent source/receipt integrity checks and finite test suite. Neural results are confined to a synthetic cyclic simulator.

## What was tested

The hidden state is z_t=z_0+t d mod n with unknown shift d and initial phase z0. At each step an active controller chooses one label m_t and observes the bit [z_t=m_t]. Shift d is the supervised target; **no true z0 or d is given at inference**. The input n is known. Each scored query consumes one returned bit and requires an address command. The main policy budget is Q=n+2 returned bits.

The learned controller is deliberately **structure-assisted**: it receives an exact uniform belief over all n² candidate (d,z0) pairs, updates that belief deterministically, and learns only which marker address to query. It was supervised by action values from an exact finite-horizon Bayes oracle. Consequently its performance does not demonstrate independent learning of the hidden transition law. The OIC-19 generator-learning research remains a different experimental track.

### Training

- Five seeds 0..4 with two training conditions, full-phase and two initial phases withheld at n=4,5,6.
- 500 Adam steps, batch size 96, learning rate 0.01, with 120 synthetic training episodes per n in the teacher context generator. Episodes sample d and z0 from the registered training split; the teacher's Bayes prior still supports **all** phases. This latter structural privilege is disclosed.
- The scorer is a 7→16→1 tanh network with **145 trainable parameters**. Its seven features are computed for each candidate marker from the **exact** current shift/phase belief, including expected one-step MAP and entropy gain. The exact oracle supplies all optimal next-action labels, including ties. Training data include random, greedy and teacher-guided policy histories.
- Tested n=4,5,6 and unseen n=7,8,12,16,24,32, with all n² true (shift,phase) combinations in the exact population evaluation. Training initialized from seed-matched random parameters. Every test n uses Q=n+2 one-bit queries.
- 10 frozen model-weight receipts contain loss checkpoints, training-context counts, oracle costs and held-out results. Pure CPU PyTorch and NumPy, with exact structural inference in Python. Exact finite reference code uses the standard library.

## Clean shift identification at Q=n+2 queries

Accuracy is averaged over all n² hidden configurations for each model, then averaged over five learned policy seeds. Random fixed means use 64 distinct deterministic **precommitted** schedules per n; their marker assignments are independent of sensor reports. The fixed simultaneous OIC-20 marker controls use their reported minimum labeled sensor sets and n time steps, so they have **different output-bit budgets**.

| n | Query bits | 64 random fixed schedules | Greedy adaptive | Learned adaptive, full | Learned adaptive, heldout | OIC-20 simultaneous bits / one-episode accuracy |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 6 | 78.22% | 87.50% | 95.00% | 93.75% | 8 bits / 100% |
| 5 | 7 | 72.44% | 96.00% | 96.00% | 96.00% | 10 bits / 100% |
| 6 | 8 | 58.64% | 77.78% | 90.00% | 88.89% | 12 bits / 77.78% |
| 7 | 9 | 60.30% | 87.76% | 87.76% | 87.76% | 14 bits / 100% |
| 8 | 10 | 50.54% | 76.56% | 79.69% | 79.69% | 24 bits / 90.63% |
| 12 | 14 | 38.88% | 61.81% | 79.58% | 79.72% | 36 bits / 72.92% |
| 16 | 18 | 36.13% | 69.14% | 71.64% | 72.27% | 64 bits / 81.25% |
| 24 | 26 | 28.69% | 55.03% | 72.29% | 72.53% | 96 bits / 65.28% |
| 32 | 34 | 28.65% | 64.65% | 67.27% | 67.42% | 160 bits / 74.80% |

The exact Bayes-optimal adaptive Q=n+2 policy achieves 100% (n=4), 96% (n=5), and 94.44% (n=6). The trained scorer approaches it on n=4–6, while its OOD performance declines with the phase count and available bits. At five queries, exact adaptive-vs-fixed comparison gives 15/16 vs 13/16 at n=4, 19/25 vs 17/25 at n=5, and 22/36 vs 18/36 at n=6. The universal full-identification horizon conjecture Q*=2n−2 is verified only for n=2–6.

### Distribution identification versus trajectory identification

OIC-20's marker minimum asks whether **complete observation distributions** are distinct for different shifts. A single episode may still fail to determine the shift. For n=6, the minimum marker set (0,2) identifies all six laws but attains 28/36=77.78% single-run Bayes shift accuracy after six simultaneous readouts per marker. In particular, shifts 0 and 3 with initial phase 1 share the silent observation word. This distinction matters when assessing claims of "identifying" an internal generator.

### Address and sensor budget

The adaptive policy transmits Q sensor-report bits and, if addresses are transmitted at each step, Q ceil(log₂ n) additional address bits. At n=24 its principal evaluation therefore requires 26 sensor bits and up to 130 address bits, giving 156 transmitted bits, plus the policy's live exact n²-hypothesis belief state and weight storage. The fixed simultaneous OIC-20 control at n=24 uses 96 simultaneous sensor bits over 24 steps, with four preinstalled labeled positions (20 bits to specify their binary addresses once). Thus the adaptive method uses fewer *returned sensor bits* and a different communication and latency profile. The comparison does not establish a reduction in total communication cost.

### Sensor-fault panels: clean objective exposes an unprotected failure mode

For n=4,6,8, every shift, every initial phase, every position of one corrupted reported bit and one marked erasure was evaluated under a fixed 4% independent-flip likelihood parameter. Sensor positions were chosen *online from received history*, so a substituted bit could redirect subsequent actions. This parameter is an assumed likelihood, not the adversary's actual budget.

| n | Training | Clean online accuracy | One adversarial flip | One marked erasure |
|---:|---|---:|---:|---:|
| 4 | Full | 98.75% | 41.25% | 86.67% |
| 4 | Heldout | 93.75% | 40.83% | 86.46% |
| 6 | Full | 92.78% | 45.62% | 80.28% |
| 6 | Heldout | 92.22% | 43.75% | 80.63% |
| 8 | Full | 82.50% | 48.97% | 74.31% |
| 8 | Heldout | 81.56% | 48.94% | 74.38% |

These full-population finite tests quantify failures of the **complete controller-plus-Bayes-filter interface**. Many are consequences of sparse information and changed query paths. The OIC-08 fixed-schedule correction guarantee for an alternative sensor channel does not apply directly to the current moving one-marker protocol. The noisy evaluation did not train a new fault-aware controller.

## Exact finite mathematics

See [OIC21_THEORY.md](OIC21_THEORY.md). Bellman induction proves optimal adaptive Bayes shift identification for the declared observations, and a zero-response argument proves that fewer than n one-bit queries cannot guarantee recovery of d. Exact dynamic programming gives a strict advantage for adaptivity and the observed n=2–6 worst-case threshold Q=2n−2. A second elementary counterexample distinguishes law-level identification from a single-run codebook guarantee.

## Reproducibility and evidence boundaries

Files: `oic21_exact.py`, `oic21_policy.py`, `oic21_train.py`, `oic21_bench.py`, `oic21_faults.py`, `oic21_fault_bench.py`, and `oic21_verify.py`. The frozen `results/` folder contains ten model training receipts (compressed), exact finite baseline comparisons and 30 full adversarial fault panels with all positions and states. A standard-library verifier checks the frozen hashes, metadata, calculated summary rows and small exact oracles. Optional CPU Torch replay reconstructs learned policy outputs from the archived weights; the basic checksum verifier itself does not prove that weights originated in the specified optimizer run.

No real trained transformer, deployed sensor network, or actual biological/cosmological system was evaluated. The novel component lies in these concrete applications and exact finite experiment results within the existing mathematics of Bayesian sequential design.
