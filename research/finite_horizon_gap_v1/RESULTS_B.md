# FHC-01B: executed future-observation geometry pilot

Date: 2026-11-13. This is a **FROZEN_SYNTHETIC_RECURRENT_PILOT** using
the existing RC-01 affine GF(2) recurrence. It is a measured
finite-state/PCA experiment, not a trained neural checkpoint run.

## Reproduction

Source CLI: bench_future_metric.py. Exact primitive recurrence:
horizon_geometry.py, which directly imports the RC-01 Affine,
row_times_matrix, row_basis and coordinates implementations.
Random seed 940198, NumPy 2.1.3, 48 calibration states, 16
independent held-out state IDs, 120 possible held-out pairs.
All 64 six-bit states have a source-generated future trace.

At every T the finite future metric is the Euclidean distance
between T Boolean observations of the primitive six-bit looper.
Scale is fitted as inverse median positive pairwise code/source
ratio on calibration pairs only. The primary D=2.0 violation count
uses that frozen scale and the interval [1/sqrt(D),sqrt(D)].
The separately reported optimal D is the ratio of the largest to
the smallest nonzero held-out distance ratio. A collapsed pair
gives infinite optimal D (recorded as null in the JSON receipt).

Reproduce on Python 3.12 with NumPy 2.1.3:

    python -m unittest discover -s research/finite_horizon_gap_v1 -p test_horizon_geometry.py -v
    python research/finite_horizon_gap_v1/bench_future_metric.py --seed 940198 --horizons 1 2 4 6 8 16 32 63 --widths 2 4 6 8 16 32 --out /tmp/fhc01b_heldout.json

The JSON receipt is uploaded as a GitHub Actions artifact. The
original printed run can be inspected at
https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43405190674 .
Program source SHA-256 at that run:
3245bd638b4da90da06b46b13c79a905ff18dbdd9f395b203cfff63b2b01a5f0 .

## Exact finite rank frontier

| Horizon T | Minimal predictive binary bits | Exact isometric Euclidean dimension |
|---:|---:|---:|
| 1 | 1 | 1 |
| 2 | 2 | 2 |
| 4 | 4 | 4 |
| 6 | 6 | 6 |
| 8 | 6 | 8 |
| 16 | 6 | 16 |
| 32 | 6 | 32 |
| 63 | 6 | 63 |

The binary rank counts every possible exact semantic code, including
nonlinear ones, if the code holds a fixed B-bit word. The real
dimension is a lower bound for *exact Euclidean isometry* on all 64
source states. Its proof uses the 63 pairwise orthogonal nontrivial
Walsh characters. At T=63, all distinct source points have squared
Euclidean distance exactly 32 in future-trace space.

## Held-out geometry at T=63

| Encoder | Code dimension | Optimal held-out distortion D | Violating pairs at fixed calibrated D=2 |
|---|---:|---:|---:|
| Exact predictive quotient | 6 binary bits | 2.449 | 13/120 |
| PCA trained on 48 states | 6 real coordinates | infinity | 120/120 |
| Gaussian random projection | 6 real coordinates | 5.081 | 30/120 |
| PCA trained on 48 states | 32 real coordinates | infinity | 120/120 |
| Gaussian random projection | 32 real coordinates | 1.972 | 3/120 |
| Full future trace | 63 real coordinates | 1.000 | 0/120 |

PCA returns identical codes for all 16 unseen states, within floating
point error, for every tested dimension d=2,4,6,8,16,32. FHC-8 proves
this collapse for *any positive-variance PCA component fitted on any
proper subset of regular simplex vertices*.

Gaussian-32's optimal D=1.972 is a descriptive held-out rescaling
statistic. With the scale frozen on calibration data, three of the
120 distinct held-out pairs fail the D=2 condition. Both quantities
are retained because reselecting the scale on test pairs would
leak information.

At horizon T=32, the quotient has optimal distortion 2.291, versus
1.954 for Gaussian-16 and 2.116 for Gaussian-32. Both Gaussian-16
and Gaussian-32 still have calibrated D=2 failures (9 and 1,
respectively). PCA-32 is the full metric dimension there and
reproduces pairwise distances exactly.

## Geometric interpretation

The source recurrence is invertible and every length-63 trace
contains all six source bits. Predictive-memory sufficiency is a
lossless *finite-state decoder* contract. PCA projects a future
feature space and is fitted on a disjoint subset of states; its
failure is a generalization property of that fitted subspace.
Random projections preserve distances approximately by a
different mechanism.

The candidate bit encoder requires a basis description and a
future-bit decoder table. PCA and Gaussian projections require T
oracle observations per state to assemble their input feature
vectors, plus matrix parameters and a calibration scale. The JSON
resource ledger records these costs explicitly. A cheap encoder
from a native neural hidden state has not been established.

None of the finite geometry results constructs the OpenAI #098
compact doubling witness; an N-point equilateral family has
doubling constant at least N. Source #094 concerns existential
finite-set embeddings and does not promise that sample-trained
PCA will generalize.
