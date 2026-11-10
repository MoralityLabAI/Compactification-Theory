# RRC-20: calibrated expert mixtures on new algebraic source constructions

**Exploratory CPU pilot, November 10, 2026.** Three RRC-19 models and the
certified finite quotient were frozen. A 60-parameter gate was trained on
420 previously exposed development examples, then scored once on fresh
source-generator constructions with exact two-demonstration verification.
The source labels and query futures were hidden from each inference path.

## Primary results

Every budgeted method had up to **12 exact graph-class attempts** per case.
The two recurrent and one feedforward experts were all executed whenever
an ensemble proposed a list.

| Method | Known families, 144 | Four new families, 384 | Combined, 528 |
|---|---:|---:|---:|
| Adaptive 60-param gate | **131** | 301 | 432 |
| Constant 4-param gate | **131** | 302 | 433 |
| Uniform four-expert mixture | 126 | **318** | **444** |
| Shuffled-gate-label mixture | 120 | **318** | 438 |
| Frozen balanced GRU | **131** | 298 | 429 |
| Frozen skewed GRU | 128 | 300 | 428 |
| Frozen balanced feedforward | 116 | 290 | 406 |
| Certified class-size prior | 89 | 275 | 364 |
| Fixed twelve syntactic programs | 66 | 198 | 264 |
| Complete certified class search, larger budget | 144 | 384 | 528 |

Numbers are **future-correct degree-32 predictions**, after exact
verification on both public input/output prefixes through degree 12.
The adaptive gate improves in-familiar-family performance relative to
uniform weighting. Its primary generalization gate **fails**: the uniform
ensemble obtains 17 more correct programs on the four new families. The
simple constant-calibrated gate also exceeds the adaptive model by one
program on that split. A final gate trained on scrambled class targets
matches the uniform method's novel-family 318/384; its frozen base experts
retain their valid supervision, so this is a negative test of the *new
gate training* rather than an untrained-model comparison.

### Newly composed source families

| Unseen source generator /96 | Adaptive | Uniform | Balanced GRU | Skewed GRU | Class-mass prior |
|---|---:|---:|---:|---:|---:|
| Arithmetic-part product under `q -> -q` | 63 | **69** | 61 | 66 | **69** |
| Product times reciprocal of sparse polynomial | 78 | **81** | 77 | 70 | 66 |
| Signed unit series times arithmetic-part product | **89** | 88 | **89** | 88 | 72 |
| Cube of sparse unit series | 71 | **80** | 71 | 76 | 68 |

The aggregate deterioration is concentrated in signed-twist and cubic
inputs. The paired receipt includes exact exclusive-success counts: on
signed twists, the adaptive mixture wins five unique cases and uniform
wins eleven; on cubic sparse, adaptive-only is four and uniform-only is
thirteen. The remaining differences occur on rational-product (3 vs 6)
and mixed-signed-product (4 vs 3) instances. These are one-seed paired
counts, with no cross-seed uncertainty estimate.

One ordinary cubic-sparse example produces an **accepted 12-degree
program with an incorrect future** for the adaptive and constant gates,
while the uniform mixture extrapolates correctly on that example. This
is a finite-horizon alias rather than a verifier implementation error.
The full-class solver achieves 384/384 on all positive new instances
with its larger class-verification allowance.

## Calibration and method interpretation

The 420 calibration records are exactly the old RRC-19 development set,
with certified canonical program labels. All three base predictors
remain frozen. Full-batch FP64 Adam ran 500 updates for each gate.

| Gate | Learned FP64 parameters | Calibration canonical top-12 | Calibration mean NLL |
|---|---:|---:|---:|
| Adaptive | 60 | **381/420** | 2.0670 |
| Constant | 4 | 375/420 | 2.1351 |
| Shuffled gate labels, frozen pretrained experts | 60 | 332/420 | 2.8786 |

The calibration improvements are **in-sample**. RRC-19 development
results were already visible during this research series. Extra
calibration capacity can therefore promote source-dependent confidence
patterns that have limited validity under a new generator. No held-out
family is used for choosing a new weight, threshold or temperature after
the initial run.

## Verifier, cost and uncertainty

All twelve constructed inconsistent observed examples are rejected by
complete finite-grammar search and by every budgeted method. Six paired
future mutants share all visible information, with degree-15 future
changes. Every method issues identical candidate traces for both
members. Complete search accepts all twelve observed prefixes and
predicts six correctly, making six unavoidable false future
acceptances. The adaptive mixture accepts six of those prefixes and
predicts three futures correctly, while abstaining on the other six.

The adaptive ensemble performs approximately **112,608 forward dense
MACs per case**: two 53,216-MAC GRUs, one 6,120-MAC MLP and a 56-MAC
small gate. The constant and uniform ensembles cost 112,552 MACs;
using a single frozen GRU costs 53,216. Mean integer-loop proxy work
varies with the selected proposals and is recorded per family in the
receipt. On the new rational/product family it is **1,682.42** for the
adaptive gate versus **1,436.77** for the uniform mixture. On cubic
sparse it is **1,947.56** versus **1,678.90**. The extra calibration
therefore increases both neural inference and, on these examples,
verification work relative to a cheaper uniform selection of the same
three base networks.

The adaptive gate adds 3,840 bare bits in FP64 to the 598,880 bare
FP32 bits of its three frozen experts. It also relies on exact class
tables and the public algebraic interpreter. Calibration precomputes
three neural outputs for 420 examples, costing at least 47 million
forward MACs by the same proxy, followed by 500 batches of gate
optimization per trained arm. The inherited RRC-19 expert training
requires about 14.85 billion forward-only MACs across its four
original arms, and the RRC-18 class normalizer used 256 offline graph
normalizations. These are separate resource categories; all MAC and
integer-coefficient operation counts omit memory and runtime overhead.

## Reproduction and research decision

Eighteen new unit tests check exact source constructions, source/test
separation, class-label calibration, portable inference parity, the
literal two-demo query budget, source/checkpoint hashes, invalid
samples and collision invariance. The new standard-library benchmark
regenerates the complete JSON receipt. Independent retraining has to
reproduce the three stored gate JSON checkpoints byte-for-byte; no
model download or GPU job is required.

The frozen expert ensemble gains accuracy through diversity, while
family-specific calibration contributes little reliable improvement on
new algebraic source distributions. Future work should test proper
OOD detection or finite-degree program semantics under **new graph
operators**, with a separate development roster and calibrated cost
budget before any further model architecture changes.
