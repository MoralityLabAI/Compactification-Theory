# RRC-21: Known-only selective deferral to certified algebraic search

**Exploratory CPU result, November 10, 2026.** The three RRC-19 predictors,
RRC-18 exact polynomial quotient and the public integer verifier are frozen.
RRC-21 performs no neural training. It calibrates a source-shift detector
using **420 known-family reference** cases and an independent **180 known-family
calibration** cases, then makes explicit routing decisions on 480 fresh
positive series cases, 16 constructed negatives and 12 collision controls.

## Main exact future results

| Policy | Known /192 | Novel /288 | All positive /480 | Deferred cases | Max candidate checks |
|---|---:|---:|---:|---:|---:|
| Uniform three-model neural mixture | **169** | **205** | **374** | 0 | 12 |
| Certified size-ordered top-12 | 132 | 172 | 304 | 0 | 12 |
| Six neural then six new mass classes | 152 | 197 | 349 | 0 | 12 |
| **Selective strict-12** | 157 | 167 | 324 | 311 | **12** |
| **Selective full-symbolic deferral** | **181** | **276** | **457** | **311** | **33** |
| OOD-only full-symbolic deferral | 175 | 275 | 450 | 280 | 33 |
| Visible-hash random full deferral | 177 | 225 | 402 | 137 | 33 |
| **Complete certified search on every case** | **192** | **288** | **480** | all cases symbolic | **33** |

The **strict-12 criterion fails**: the selector loses 50 correct predictions
versus the full neural mixture because its mass-only fallback is too weak.
The expanded-budget policy improves from 374 to 457 correct futures by
spending more exact interpreter calls on deferred examples. The full
certified solver reaches 480, with zero neural inference, and remains the
reference for actual class-complete accuracy. Hence the gain supports
*OOD-conditioned compute allocation*, not neural superiority over the exact
symbolic compiler.

## Source-shift detection and ablations

The detector uses the squared standardized distance to its closest among
420 known-source reference feature vectors. The exact, nonrandomized
conformal rank p-value flags a case when `p <= 0.10`; a second gate
flags high normalized mixture entropy using the calibrated 80th percentile,
**0.9094310009903586**. Calibration defers 50/180: 17 are OOD flags.

On the test roster the OOD-only score has **AUROC 0.960892** between
192 known-source and 288 novel-source examples. It flags **19/192**
known and **261/288** novel examples. The complete two-gate policy
routes **48/192** known and **263/288** novel examples to its fallback.
Among test families:

| Novel generator | OOD flagged /48 | Full selective correct /48 | Neural top-12 correct /48 |
|---|---:|---:|---:|
| Rational reciprocal | 48 | 48 | 38 |
| Convolution square | 45 | 46 | 39 |
| Sign-twisted product | 48 | 48 | 29 |
| Cubic sparse | 48 | 48 | 38 |
| Degree-three lacunary | 24 | 38 | 32 |
| Large signed coefficients | 48 | 48 | 29 |

The old RRC-19 rational/square and RRC-20 twisted/cubic source *types*
were known during research design. Only lacunary and high-amplitude signed
sources are newly introduced type definitions here. This roster is a new
instance sample with prior family knowledge, not an independent blind
source-discovery benchmark. In lacunary sources, many early coefficients
vanish, weakening the detector and class model simultaneously.

The OOD-only full policy already recovers **450/480** with 280 deferrals;
adding the confidence threshold reaches **457/480** with 31 extra
deferrals. Since candidate budgets differ, the 402/480 visible-hash random
policy with 137 deferrals cannot serve as a matched-cost superiority test.
It is included to expose the cost and consequences of random fallback.

## Resource ledger on 480 normal positive cases

| Policy | Mean two-demo graph checks | Mean nominal exact integer work | Mean neural dense MACs | Mean extra OOD scalar proxy |
|---|---:|---:|---:|---:|
| Neural12 | 5.83 | 1,551 | 112,552 | 0 |
| Mass12 | 7.55 | 1,640 | 0 | 0 |
| Selective12 | 6.80 | 1,511 | 46,897 | 12,600 |
| SelectiveFull | 8.86 | 2,120 | 46,897 | 12,600 |
| OOD-only full | 8.28 | 2,012 | 46,897 | 12,600 |
| Complete certified search | 10.12 | 2,412 | 0 | 0 |

These are distinct proxies, with big-integer arithmetic, interpreter
implementation, Python work, sorting, and actual elapsed time omitted.
The OOD detector stores 420 vectors of ten scalar coordinates and
scans them for each routed case. It skips the **112,552** MAC neural
forward computation whenever a case is immediately OOD-flagged; otherwise
it invokes both frozen GRUs and the matched feedforward model. The
three frozen networks contain 18,715 FP32 parameters combined, and
were trained in earlier experiments. Exact polynomial compilation of
all 256 graph/topology pairs is another shared offline cost.

The complete solver uses about 1.26 additional two-demo queries per
positive case relative to SelectiveFull, with **zero** model MACs and
no reference-distance scan. Results do not establish any hardware-agnostic
superiority for learned deferral over this strong exact baseline.

## Exactness, negative controls and limits

All 16 selected inconsistent demonstrations are rejected by the
class-complete solver. All six base/future-mutant pairs share identical
visible data, candidate traces and routing choices; complete search
accepts their prefixes and gets one future wrong in each pair. The
plain Neural12 policy also has **one** additional incorrect future
prediction on a sparse signed source despite a valid degree-12 prefix
certificate. SelectiveFull instead returns a future-correct program on
that case. These are finite certificates and independent forecast tests,
not infinite algebraic identity proofs.

The conformal p-value's superuniform theorem assumes exchangeability
between an independently fixed known-source calibration sample and a
new known-source test sample. The evaluation's fixed per-family quotas,
prior revealed source constructions and small calibrated feature set
mean it should be interpreted as an *empirical source-shift test*, without
claiming a distribution-free conditional-error guarantee. Entropy
thresholding further changes routing, with no selective-risk guarantee.

## Reproducibility

- 19 new RRC-21 tests and 71 inherited RRC-17/18/19/20 tests pass locally.
- Frozen source- and checkpoint-hashed complete replay reproduces the
  saved JSON receipt byte-for-byte.
- Reproducing calibration from the known-only data runs purely on CPU,
  applies **zero** gradient updates, and reproduces the checkpoint exactly.
- The source, benchmark, mathematical note and CI workflow preserve the
  negative fixed-budget result and separate full-fallback query limits.

A next experimental gate should evaluate *resource-matched deferral* using
latency and integer-bit work on independent hardware, or learn a
calibration score that predicts **verified program acceptance** under
new primitive/topology shifts. The current detector recognizes familiar
source-feature differences within a fixed public graph language.
