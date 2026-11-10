# RRC-21: Split-calibrated OOD detection and selective class search

**Executed 2026-11-10.** Exploratory, one seeded roster with class labels
and public primitives inherited from RRC-18/19/20. The evaluation families
and thresholds were chosen during research development. There is no
externally preregistered multi-seed or new-primitive claim.

## Question

Can a known-only source-shift detector recognize when a frozen neural
class-ranking policy may be unreliable, enabling cheaper learned search on
familiar examples and exact symbolic deferral on unfamiliar examples?
Measure verification rates, OOD separation, error and abstention, and
resource costs. Distinguish a **hard 12-query budget** from a permitted
larger-budget complete symbolic fallback.

## Frozen training/calibration and test split

- Frozen base: RRC-19 balanced recurrent, skewed recurrent, and balanced
  feedforward checkpoints. The RRC-18 polynomial class compiler supplies
  25/24/33/24 finite-degree semantic classes, with all programs and
  operation implementations known.
- Independent reference: 140 examples per *known* source family, 420 total.
  The features depend only on source coefficients of the two public demos.
- Separate calibration: 60 examples per same three known source families,
  180 total. It selects the OOD rank cutoff \(\alpha=0.10\) and 80th
  percentile of uniform-mixture entropy; no hidden graph or future label
  is used to choose either threshold. These constants were fixed before
  the RRC-21 test run. There is **zero** model or calibration-gate training.
- New test cases: 64 examples per known family (signed, partition product,
  sparse signed), 48 per novel family (rational reciprocal, convolution
  square, sign-twisted product, cubic sparse, degree-three lacunary and
  amplified signed), for 480 normal positives. Test rosters are disjoint
  at the visible-demonstration level from all reference/calibration cases.
- Additional controls: 16 inconsistent observed-output pairs, selected
  only after exact finite-grammar rejection; six pairs of identical
  visible evidence whose withheld query outputs differ at degree 15.
  Source selection never consults the neural accuracy or future grade.

Two newer *source shapes* are lacunary_q3 and amplified_signed. Rational,
squared, twisted-product and cubic-sparse were introduced in earlier RRC
experiments and are **new instances only**, rather than genuinely blind
source constructions across the research sequence. The graph topology and
operator vocabulary are public and unchanged for every group.

## Policies, route and checks

- **Neural12:** average the three frozen distributions and the certified
  mass prior uniformly; independently execute the first 12 classes.
- **Mass12:** certified class-size representatives, first 12.
- **Hybrid6+6:** first six neural, followed by six unseen mass-prior
  representatives, removing duplicates.
- **Selective12:** first OOD rank test; if flagged, execute Mass12 without
  model inference. Otherwise compute full neural mixture entropy; use
  Mass12 above calibration threshold, Neural12 below it. Maximum 12.
- **SelectiveFull:** same routing. A deferred input executes the complete
  certified class list (at most 33 candidates); accepted native neural
  cases receive the same 12-check ranking.
- **OOD-onlyFull:** complete symbolic fallback only for OOD p≤0.1, with
  no entropy trigger, for an ablation of the calibration confidence gate.
- **RandomFull:** a deterministic visible-input hash chooses full search
  with probability equal to the previously recorded **27.78%** calibration
  deferral rate; it is a comparator with lower actual test deferral on
  this shifted distribution, and hence is **not cost-matched**.
- **FullCertified:** certified mass-ordered complete class search on every
  sample. Up to 33 checks; serves as the class-complete upper benchmark.

Routing never reads test labels, generator-family names, future targets or
unchecked graph executions. Every candidate is checked with the same
exact two-demo interpreter, and each check is metered. A program accepted
through degree 12 can still fail on the withheld degree-32 target.

## Gates and reporting

1. Construct reference/calibration using `dataset(include_test=False)`;
   test-source generators are unreachable in this mode. Audit source,
   roster and frozen checkpoint SHA-256 hashes and zero neural updates.
2. Test the exact conformal rank formula and independently measure
   known-family false-flag rate, OOD detection rate and AUROC. Any
   exchangeability assumption must be stated for a p-value guarantee.
3. Every strict arm stays at 12 unique candidate checks; full-fallback
   arms can make more checks. Every paid check executes both observed
   demonstrations, and the exact work ledger matches the actual trace.
4. Test finite negative rejection, source/future blindness, paired
   future-collision invariance, model-free OOD early exit, and full
   symbolic completeness on the 480 in-grammar positives.
5. Report results for the *strict* and *full* policies separately, with
   average integer work and forward MACs, plus OOD feature-scan work.
   Declare failures, including any selective strict-budget regression.
6. Reproduce an identical calibration JSON checkpoint without model
   training, replay every score and hashed trace, and keep old
   RRC-17/18/19/20 mathematical regressions green.

## Reproduce

From the Compactification-Theory root, standard-library Python 3.10+:

```sh
python -m unittest discover -s research/rrc_compactification -p 'test_select21.py' -v
python research/rrc_compactification/bench21.py --check
python research/rrc_compactification/train21.py --out /tmp/rrc21_calibration_replay.json
cmp /tmp/rrc21_calibration_replay.json research/rrc_compactification/results/rrc21_calibration.json
```

Earlier regressions:

```sh
for pattern in test_family20.py test_family19.py test_certify18.py test_quotient17.py; do
  python -m unittest discover -s research/rrc_compactification -p "$pattern" -v
done
```

The CPU verifier requires no Torch and does not download weights. The
historical JSON checkpoints must be present and source-hash matched. The
RRC-21 receipt contains source/checkpoint hashes, all per-family metrics,
exact-test trace hashes, deferral decisions and calibration provenance.
