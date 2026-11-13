# FHC-01C: executed trained RRC-25 future-logit geometry pilot

Date: 2026-11-13. Status: **FROZEN_TRAINED_RRC25_DEVELOPMENT**.
Source-locked original recurrent checkpoint:
SHA-256 54f225f2a61c52fee5a740576eaab3bcbc5280a595fc1ac1ac115ec8ac26c41b.

[Experimental registration](PROTOCOL_RRC25.md) and
[source benchmark](bench_rrc25_future.py).
GitHub Actions receipt: https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43406537505

## Frozen native computation

The checkpoint is the existing RRC-25 1,000-parameter GRU, trained
to rank sixteen suffix-template pairs. The public source gate
bench25.checkpoints() verifies the committed JSON weight SHA-256,
training source hashes, step count and model structure before the
new benchmark can execute.

Using each case's first publicly inferred header, the benchmark
replays all nine original source feature rows through the portable
GRU. Its final 16 logits agree exactly with the existing portable
model25.logits implementation (maximum absolute difference 0.0).
No hidden true suffix label or case.future is consulted; a targeted
test mutates the entire case.future and confirms identical states
and outputs.

The analysis freezes the post-input 12-dimensional recurrent
hidden state h9 and feeds zero 8-vectors as artificial continuations
for horizons T=1,2,4,8,16. The observable G is the trained model's
16-dimensional logit head, so the future feature dimension is 16T.
The trained checkpoint's native task never asked for zero-input
continuations. This study measures **the resulting controlled
recurrent geometry**, not native future-task accuracy.

The original checkpoint was trained using PyTorch float32. The
portable reference converts stored JSON weights to Python floats
and uses double-precision scalar arithmetic, matching the source
portable inference exactly. The nominal storage ledger quotes
12*32=384 bits for float32 live hidden-state storage and
1000*32=32,000 bits of checkpoint weights; the actual Python
reference object and arithmetic costs are separate.

## Split and metric

The source roster produces 36 calibration cases, twelve from each
of the signed/sparse/product development families, and 48 held-out
cases, eight from each of signed/sparse/product/rational/squared/
late_source. No held-out case enters PCA fitting or scale fitting.
The benchmark retains full roster fingerprints, source hashes,
family labels and replayable worst-pair IDs.

For each horizon T define the future logit vector as concatenation
of the 16-dimensional native logit heads at the next T zero-input
transitions. The tested distance is Euclidean in this 16T-dimensional
space. One calibration scale per method is frozen from dev pairwise
ratios.

The held-out cohort contains 48 cases and 1128 distinct case pairs.
The table's "best D" is the descriptive optimal-rescaling distortion
computed on held-out pairs, while the violation count is the primary
calibration-only D=2 test.

## Measured future metric

| T | Native hidden 12D, best D | Native hidden, D=2 failures | PCA 12D, best D | PCA 12D, D=2 failures |
|---:|---:|---:|---:|---:|
| 1 | 7.820 | 193/1128 | 1.000 | 0/1128 |
| 2 | 9.527 | 210/1128 | 1.059 | 0/1128 |
| 4 | 13.832 | 309/1128 | 1.087 | 0/1128 |
| 8 | 30.087 | 441/1128 | 1.102 | 0/1128 |
| 16 | 39.528 | 839/1128 | 1.230 | 0/1128 |

At T=16, PCA with 8 coordinates has descriptive D=1.457,
with one of 1128 held-out pairs violating the calibration-scaled
D=2 rule. Gaussian-12 has descriptive D=2.515 and 13 calibrated
D=2 failures. The original hidden 12D code exactly determines the
future logit trajectory when paired with its original recurrent
transition and trained readout; its raw Euclidean metric has
substantial distortion relative to this particular future logit
metric.

A compact representation's *dynamic sufficiency* and the
distances implied by its unmodified coordinate norm are different
properties. The projection may better preserve the chosen metric
because it consumes the future logit vectors themselves.

## Experimental caveats

- PCA and Gaussian projections receive all T recurrent future
  logit vectors as their input, at a cost of T native recurrent
  transitions per case. The projections then store a matrix of
  16T*d real coefficients. The native hidden code has twelve
  coordinates and access to the already trained recurrent decoder.
- A run on fixed generated cases is a descriptive empirical test,
  with no global bi-Lipschitz certificate. It does not instantiate
  the #098 infinite-dimensional compact doubling witness.
- Results are sensitive to the definition and units of the
  observable G. The output is a 16-dimensional logit vector from
  an experimentally imposed zero-input continuation.
- All dev/test case measurements are retrospective relative to
  the previously trained RRC25 checkpoint. A new neural training
  cohort and frozen source-generation protocol are necessary
  before a prospective model-level claim.
- The code records every held-out pair's worst witness and
  family-level statistics in the workflow JSON artifact. The
  optimal test rescaling statistic is descriptive and cannot
  be used to select the calibration scale.

The source-lock and native replay tests passed in GitHub Actions,
including a six-test dedicated RRC25 FHC-01C suite. Existing RRC25
research decisions are unaffected.
