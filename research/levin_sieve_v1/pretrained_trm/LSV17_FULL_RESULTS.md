# LSV-17: source-locked 64-case quotient geometry results

**Date:** November 4, 2026. **Status:** completed real pretrained CPU experiment. GitHub Actions run [41852747448](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41852747448) passed the source-locked pilot and complete 64-case jobs. Full raw receipt artifact ID [13335732498](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41852747448/artifacts/13335732498). No Goodfire or GPU allocation took place.

Source-locked external TRM-inspired Sudoku model: 44,001 parameters; checkpoint SHA256 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1. Frozen 64-puzzle roster SHA256 eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611. Eighty? No: the evaluation contains exactly 64 puzzle IDs, with 32 training, 16 validation, and 16 held-out test IDs. Each of the 16 test puzzles contributes 16 correlated clean/perturbation states, for 256 test samples in supervised probe fitting. The causal comparisons below have **16 independent puzzle clusters**.

## Observed future distribution displacement

The intervention changes only the selected first-missing goal-cell token after the first recurrent pass. Each random and all-three-horizon tangent-null perturbation has the same L2 norm. The table is the mean total variation of the future digit distribution after the third subsequent recurrence.

| L2 radius | Centered random | Joint Jacobian-null | Reduction of mean TV | Test puzzles with smaller TV |
|---|---:|---:|---:|---:|
| 0.2 | 0.00143078 | 0.000175872 | 87.7% | 16 of 16 |
| 1.0 | 0.00698386 | 0.000724493 | 89.6% | 16 of 16 |
| 3.0 | 0.02199430 | 0.00165825 | 92.5% | 14 of 16 |

The direction is constructed using the frozen model's goal-history gradients on the same puzzle. It is *teacher-informed* and incurs computational cost; this measured behavioral stability is not a claim of a learned low-cost production quotient. For the evaluated radii, all final-horizon winners remained unchanged under both random and joint-null perturbations, including cases whose clean output was wrong. One random radius-3.0 perturbation flipped the intermediate horizon. The clean final winner was Sudoku-correct on 14 of the 16 held-out puzzles.

The mean reduction can be influenced by high-displacement cases. The single largest random radius-3.0 case contributes about 74% of random TV over the 16 held-out cases, even though 14 of 16 cases individually favored the null perturbation. Report paired signs and puzzle-level distributions alongside means.

### Exploratory paired uncertainty

Downloaded the exact full raw receipt and used 20,000 bootstrap resamples of the **16 test puzzle IDs** with replacement, NumPy default_rng seed 1704. In each resample, recompute 1 - mean(null_all)/mean(random) at horizon 3. These are descriptive conditional intervals, not a claim of population generalization or an independent model-selection test.

| Radius | Observed mean-TV reduction | Percentile 95% puzzle-bootstrap interval |
|---|---:|---:|
| 0.2 | 87.7% | 79.2% to 98.9% |
| 1.0 | 89.6% | 81.6% to 97.4% |
| 3.0 | 92.5% | 83.5% to 93.4% |

No data or tuning decisions were selected from the bootstrap. Each radius uses the same 16 test puzzle IDs; intervals across radii are dependent.

## Pre-LayerNorm2 symmetry control

For per-token common-mode shifts injected at the separate pre-LayerNorm2 bus, maximum observed future-logit difference was 3.553e-15 in FP64. Matched-radius centered transverse perturbation reached 0.200446. This checks the expected LayerNorm symmetry in this frozen checkpoint; the exact real-arithmetic statement already appears in LSV-15. The intervention sites differ from those in the main table.

## Linear and nonlinear future-logit probes

Both probes train on the same 48-dimensional edited goal-cell state, predicting a concatenated 30-dimensional vector of teacher logits at horizons 1, 2 and 3. Training uses 32 puzzle clusters, validation 16, testing 16. Hyperparameters and group splitting follow the original LSV-17 protocol, including validation early stopping.

| Head | Parameters | Test logit MSE | Future teacher-label agreement at h1 / h2 / h3 |
|---|---:|---:|---|
| Linear 48 -> 30 | 1,470 | **3.09911** | 58.59% / 73.05% / 71.48% |
| Nonlinear 48 -> tanh(16) -> 30 | 1,294 | 3.26925 | 77.34% / 76.95% / 62.11% |

The nonlinear model's logit MSE is 5.49% higher on the full held-out roster, reversing the apparent 39% gain from the earlier 16-case pilot. It improves h1 and h2 teacher-label agreement while reducing h3 agreement. This is a **mixed and partly negative** outcome for the proposed learned nonlinear quotient. The model is small, the task distribution narrow, and the linear head is not rank-constrained to the nonlinear head's 16-dimensional bottleneck.

## What has and has not been shown

The experiment demonstrates that nulling three selected *local margin derivatives* can greatly reduce observed future distribution change on a finite held-out roster. It does not show that a 16-dimensional nonlinear quotient preserves the underlying recurrent dynamics, that a compressed student retains Sudoku capability, that the pointwise Jacobian nullspace is robust at arbitrary radius, or that the Unique Games large-field construction transfers to real activation spaces.

The next falsification should freeze two new Gaussian perturbation seeds and compare signs, paired TV and late flips, then evaluate the stronger semiconjugacy obligation of a learned macro-update, q(R(z)) approximately F(q(z)), on unseen states and action histories. Keep the Qwen H100 stage bounded and exploratory.

## Reproduce

    python research/levin_sieve_v1/pretrained_trm/quotient_discovery.py --fetch --max-cases 64 --out runs/lsv17/full.json

Raw reproducibility is anchored in the [Actions run](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41852747448), with source-locked JSON receipt artifact [13335732498](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41852747448/artifacts/13335732498).
