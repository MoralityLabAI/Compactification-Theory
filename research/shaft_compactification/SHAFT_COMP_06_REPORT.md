# SHAFT-COMP-06: recurrent observation depth and substrate-relative objective codes

**Date:** 2026-11-10. **Evidence:** (a) exact constructed finite-system quotient; (b) Lean-verified abstract statements, subject to CI; (c) two independently trained synthetic CPU GRUs. This work concerns consented local toy models. It provides no evidence that a large frontier AI has a hidden transferable objective representation or could survive a complete weight and state deletion.

## I. Behavioral quotient changes with the observer's reach

Let B be a computational substrate, g a latent task identifier, and F_B its recurrent dynamics. For a defender with a set of allowed input words W and observation function O_B, define the behavioral signature

```text
sig(g; B,W) = ( O_B( F_{B,w}(init_B(g)) ) : w in W )
```

and g ~ g' iff their signatures agree. The quotient `Goals / ~` contains C(B,W) behavior classes. For a fixed-length, exact objective code whose decoder may use B, at least `ceil(log2 C(B,W))` bits are necessary, and this bound is achieved by an indexed codebook when the codebook's storage is charged to B. The cost of B and of the index decoder is **excluded** from these conditional code bits.

When W grows, the partition can only refine. When the defender can observe additional substrates B, the joint signature concatenates observations and the equivalence becomes the intersection of per-substrate equivalences. Neither result implies neural accessibility of a hidden objective.

The exact fixture has 16 four-bit goals. Substrate 0 exposes bit 0 at step 1, then bits 1,2,3 in subsequent steps; substrate 2 exposes bits 2,3,0,1. The fixture generates:

| Horizon H | Classes, substrate 0 | Minimum bits | Classes, substrates 0+2 | Minimum bits |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 1 | 0 |
| 1 | 2 | 1 | 4 | 2 |
| 2 | 4 | 2 | 16 | 4 |
| 3 | 8 | 3 | 16 | 4 |
| 4 | 16 | 4 | 16 | 4 |

The finite fixture is executable with `python -m research.shaft_compactification.depth_quotient`, and its pinned receipt is [results/depth_quotient.json](results/depth_quotient.json).

## II. A sharp recurrent depth bound

Let F and G act on a common metric state space, with F L-Lipschitz on the visited region, `sup_h d(F(h),G(h)) <= delta`, and initial discrepancy `d(h_0,k_0) <= epsilon_0`. Write `h_{t+1}=F(h_t)` and `k_{t+1}=G(k_t)`. The triangle inequality gives

```text
e_{t+1} <= L e_t + delta
e_n <= L^n epsilon_0 + delta * sum_{j=0}^{n-1} L^j.
```

For L<1 the second term is bounded by `delta/(1-L)`; for L=1 it equals `n*delta`; for L>1 it can grow exponentially. A K-Lipschitz observable inherits a discrepancy at most `K*e_n`. In a multiclass classifier, a reference logit margin greater than `2*K*e_n` suffices to preserve the argmax decision. This is a *conditional* certificate: verifying a global L and delta for an actual neural model remains an open task.

A symbolic Nat-valued version is kernel-checked in [lean/Shaft/DepthTransfer.lean](lean/Shaft/DepthTransfer.lean): finite-horizon monotonicity, step-consumption, substrate-scope monotonicity, code separation, and recurrence error unrolling. The exact CPU scalar examples at depth 5, delta=1 have defects 1, 5, and 31 for gains L=0,1,2, respectively. The scalar example saturates the bound.

## III. Trained recurrent toy experiment

Source: [experiments/trained_recurrence.py](experiments/trained_recurrence.py).

A synthetic task g in {0,...,15} is a four-bit truth table: given token x in {0,1,2,3}, output bit x of g. Each separate GRU receives g only through its **initial** learned goal embedding. Subsequent tokens enter a recurrent GRUCell and output head without re-supplying g.

* Two independent CPU runs with seeds 101, 202; 3,265 parameters each, hidden dimension 24, 600 Adam steps, batch 128, learning rate 0.01, training sequences of length 12.
* Unseen evaluation sequences of lengths 1,12,32,64; batch 256 per seed and horizon. Goals were present in training; only the token sequences are held out.
* PyTorch 2.10.0+cpu, three threads, deterministic algorithms requested.
* Controls: shared-code model-native initialization, goal ID shifted by one, initial-state zero erasure, uncalibrated embedding transplanted from the other trained GRU, and independent per-coordinate embedding quantization at 2/4/8 bits.
* Mid-run intervention at time step 32 of a 64-step sequence: set hidden state to zero, or explicitly restore the native learned embedding indexed by the four-bit goal code. Nothing is written to external or networked storage.

### Main accuracy (%)

| Condition | Seed 101, 12 steps | Seed 202, 12 steps | Seed 101, 64 steps | Seed 202, 64 steps |
|---|---:|---:|---:|---:|
| Native semantic code | 100.0 | 100.0 | 92.7 | 92.1 |
| Initial state erased | 50.2 | 51.7 | 47.8 | 50.0 |
| Other model's raw hidden embedding | 51.6 | 52.8 | 46.9 | 52.7 |
| Two-bit latent coordinate quantization | 97.6 | 91.8 | 88.7 | 88.4 |
| Four-bit latent coordinate quantization | 100.0 | 100.0 | 92.4 | 91.9 |

### Final eight steps of a 64-step sequence (%)

| Intervention | Seed 101 | Seed 202 |
|---|---:|---:|
| Native, uninterrupted | 81.3 | 78.0 |
| Initial state erased | 49.1 | 49.2 |
| Mid-run state wipe at step 32 | 51.0 | 49.8 |
| Explicit native semantic-code reseed at step 32 | 95.7 | 96.4 |

The semantic 4-bit code selects the goal-specific initial state **inside each independently trained model**. The 24-dimensional raw initial embedding is model-specific; transplanting it without alignment performs near chance. Recovery in this fixture requires the receiving model and its learned embedding table to remain available.

The apparent compression ratio of a four-bit task ID relative to a 24x32-bit activation vector is conditional on the shared substrate. The 16x24 embedding table alone accounts for 12,288 raw fp32 parameter bits; the model contains additional recurrent and readout weights. This study measures neither an emerging clandestine objective nor unconditional compression of arbitrary models.

## IV. Reproduction and follow-up

From repository root:

```bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.depth_quotient
python -m research.shaft_compactification.experiments.trained_recurrence \
  --out /tmp/shaft-trained-recurrence.json
```

The trained receipt is [results/trained_recurrence.json](results/trained_recurrence.json). CPU floating-point results may vary with PyTorch and BLAS implementations; compare metrics with tolerance if running in a different environment. The lightweight CI tests the standard-library finite reference and Lean; it does not install PyTorch or retrain these models on every PR.

Next tests: learn a compact code from *unsupervised* recurrent state probes on a calibration split, hold out goal families, use representation alignment before transplant between independent substrates, and measure repair/error growth under recurrent state perturbations. A real model claim requires activation extraction, causal interventions, and resource accounting on a pinned checkpoint, with no contact with uncontrolled devices.
