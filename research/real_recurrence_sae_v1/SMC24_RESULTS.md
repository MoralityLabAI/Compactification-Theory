# SMC-24: frozen hidden-state gradient predictions fail to transfer across new Sudoku puzzle families

**Executed 17 November 2026. Registered program verdict: FAIL (0/4 independent-teacher SAE pairs).** This is the prospective follow-on to [SMC-23](SMC23_RESULTS.md), which failed the stronger claim that naturally highest-active SAE coordinates were uniformly beneficial to Sudoku solving. SMC-24 tests a different hypothesis: whether the gradient-mediated erasure effect is predictable from a *calibration-trained state-to-gradient map* on new Sudoku contexts. Both teachers were independently trained in SMC-13 and have similar weak task competence. Teacher weights and SAE snapshots remain frozen.

The registered protocol was the first branch commit f3a65a0, before any SMC-24 new puzzle data. The final [four-stage GitHub Actions run #44106692708](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44106692708) passed calibration-only model fitting, sealed test-label-free predictions, three-block FP64 native scoring, and an independent Python standard-library receipt replay. All scientific gates and cohort seeds remain those of the original protocol.

## Frozen problem and interpretable mathematical prediction

SMC-23 selected each receiving teacher's highest-positive SAE feature j and chose a physical minimal-norm active-coordinate kill delta with overshoot tau=.02, preserving the other three currently active encoder values. It compared the kill with a same-norm tangent sham s selected before task outcomes from an outcome-blind 33-candidate inactive-preactivation matching search. BPD-OUT03 shows the exact local task-gradient split:

\[
D = m(h-s)-m(h-\delta),\qquad
D_{\rm lin}=g(h)\cdot(\delta-s)
           =(z_j+\tau)\lambda_j-g_T\cdot s.
\]

Here m is the **correct Sudoku digit minus its strongest clean competing logit** after the original three FP64 recurrent blocks, g is the stage-one physical margin gradient, lambda is the active encoder dual coordinate, and g_T is the gradient component tangent to the active encoder constraints. In SMC-24, the **test target digit is never given to the gradient predictor**. An affine ridge model fitted on earlier calibrated contexts supplies a frozen hidden-state-dependent approximation to g.

The predictor uses the clean 48D stage-one state h, its original calibration mean and standard deviation, and an unpenalized intercept plus 48 ridge-penalized coefficients, with fixed lambda=32. Each of two teacher-specific 49×48 maps is fitted on the full 512 calibration goal gradients from the original 64 SMC-13 feature_calibration puzzles. A matched-capacity negative control permutes gradient targets across calibration puzzles within each of the eight visible-only selected goal slots and trains an otherwise identical ridge fit. Further baselines use a constant mean calibration gradient and an identically zero gradient. At each new test goal, all four predictions are the corresponding gradient estimate dotted with the **same frozen physical** delta-s.

On original calibration goals, B501 correctly predicted 235/512 (45.9%) and B701 predicted 259/512 (50.6%) missing digits. Their mean native correct-digit margin gradient norms were 4.2083 and 3.8715. This is a fixed-pair capability-matched comparison of two **previously independently trained models**, not new checkpoint training in this study. The gap to the original pretrained source A remains a limitation.

## Source locks and chronological data boundary

Original source artifactual SHA-256 values were verified before calibration:
- SMC-11C original source SAE: 7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087
- SMC-13A independently trained B501/B701 teachers and four SAE fits: 1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3
- SMC-13C conditioned correspondence calibration: 3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07

The four original source→B matched and clue-shuffled feature mapping hashes were reconstructed using old calibration, though those mappings never chose a target-native feature. The separate first CI stage trained the two ridge maps and uploaded its byte-identical calibration artifact **before** any fresh test input-only prediction. The second job generated two new noncanonical, uniquely solvable 64-puzzle cohorts using the preregistered complete-grid seeds 202634030000..202634030063 (30 holes) and 202634040000..202634040063 (34 holes), and verified disjointness from SMC-13/15/16/23. It archived **only public visible puzzle inputs, source FP32 hidden activations, task-blind SAE edit plans and four scalar effect predictions**. The target digits and native correct-digit effects were absent from that prediction archive.

In the third job the frozen prediction-only records were recovered, source state/physical edits independently checked, and the original teacher evaluated with **exactly three FP64 native recurrent blocks** from the archived FP32 stage-one anchor. Correct digit labels entered only here to choose the fixed clean rival and evaluate the task margin. The fourth job independently reconstructed the new Sudoku full grids, SHA-checked original model receipt identities and all staged files, verified each goal's source plan/physical dose/margin identity, and re-ran 2,000 whole-puzzle bootstrap draws with the registered seed and multiplicity threshold.

The predictor's original source code, calibration gradients and SMC-23 intervention design used frozen teacher checkpoints and never modified original model weights or SAE parameters. No ARG-03-N held-out cases were accessed.

## Pre-outcome engineering repairs, retained as failures

Two earlier source-only work attempts are explicitly preserved in [SMC24_PREOUTCOME_AMENDMENT.md](SMC24_PREOUTCOME_AMENDMENT.md):
1. [Run #44103770025](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44103770025) completed calibration and archived label-free predictions, but a cross-runner byte-exact Python dictionary comparison of two separately computed FP64 SAE plans failed **before any native task effect was scored**. The remedy was to archive complete canonical FP32 stage-one input states with the prediction, then score the originally archived physical kill/sham edits and verify them independently rather than reselecting them.
2. [Run #44105759040](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44105759040) passed that source-stage parity check, then its stricter-than-native cross-runner target covector/FP32 top-four comparison stopped before scoring. The remedy used the **pre-existing** SAE encoder relative recode tolerance 1e-4×(1+z_j) for that one FP32 comparison, while preserving FP64 constraints on other active covectors, same physical dose, frozen pretest selection and source original admission. Neither repair inspected a native held-out task effect or changed the scientific gate.

The final successful run found source stage-one maximum absolute cross-runner discrepancy between **9.54e-7 and 1.19e-6** across the two independent teachers and two new families, below the frozen 1e-4 admission bound. All source-self-patches and archived input/edit integrity checks passed. The separate audit verified the original source FP32 tensor bytes via standard-library IEEE754 packing.

## Primary eight-cell predictive result

The registered metric is the *per-puzzle mean absolute-error improvement* in true finite native D from calibration-trained ridge versus each fixed control; positive values mean ridge predicts better. The original 64 puzzle IDs are resampled 2,000 times, preserving all eight goal slots. Eligibility is frozen before native outcome and requires at least 60 admitted goals across 30 independent puzzles in each cell. All eight prospective B-teacher/SAE/family cells satisfied this coverage, with **209–299 admitted goals over 63–64 puzzles each**.

| Frozen teacher/SAE | Novel30: n eligible | Novel30 ridge MAE | Novel30 gain vs mean gradient | Novel34: n eligible | Novel34 ridge MAE | Novel34 gain vs mean gradient |
|---|---:|---:|---:|---:|---:|---:|
| B501 SAE71 | 213 | 1.6108 | +0.0065 | 209 | 1.6227 | -0.0491 |
| B501 SAE89 | 222 | 1.3733 | -0.0928 | 219 | 1.3390 | -0.0120 |
| B701 SAE71 | 294 | 1.4774 | -0.0264 | 299 | 1.6235 | -0.0488 |
| B701 SAE89 | 283 | 1.6046 | +0.0254 | 259 | 1.5278 | -0.0865 |

Two of the eight rows by family have positive *point estimates* versus the constant gradient. The learned predictor improves on the goal-slot-shuffled fitted ridge in four of eight cells, and on the zero-gradient baseline in only **one of eight** cells. These descriptions are descriptive, with two teacher seeds and shared puzzle sources.

The preregistered primary positive gate required a **99.375% Bonferroni-adjusted puzzle-bootstrap lower bound above zero** against all three baselines *and* better sign concordance than the constant-gradient predictor, for each teacher/SAE/family. All **24 of the 24** adjusted lower bounds across eight cells and three controls were negative. The eight lower bounds against the constant-gradient baseline alone were:

| Teacher/SAE | Novel30 adjusted lower | Novel34 adjusted lower |
|---|---:|---:|
| B501 SAE71 | -0.0935 | -0.1638 |
| B501 SAE89 | -0.1831 | -0.1184 |
| B701 SAE71 | -0.1212 | -0.1455 |
| B701 SAE89 | -0.0642 | -0.1962 |

For two cells, the **uncorrected 95% cluster intervals** for ridge-minus-constant MAE gain are entirely negative: B501 SAE89 novel30 [-0.1575,-0.0256], B701 SAE89 novel34 [-0.1608,-0.0100]. B501 SAE89 novel30 additionally has a completely negative 99.375% interval [-0.1831,-0.0031]. No favorable subset was chosen after evaluation.

Sign concordance also varied across families. For example, B501 SAE71 novel30 had 123/213 correct ridge effect signs versus 90/213 for the constant mean-gradient baseline; B701 SAE71 novel30 had 144/294 versus 161/294. Of the eight teacher/SAE/family cells, ridge's sign concordance was higher than constant in four and lower in four. The original combined gate is **FAIL for all four teacher/SAE pairs and for the overall program**.

## Why this result matters for BPD and compactification

The exact BPD-OUT03 decomposition specifies how a local task gradient controls the leading-order erasure response. SMC-24 asks whether a calibration-only *state-to-gradient learner* supplies enough of that gradient to predict actual frozen native task responses on distinct Sudoku puzzles. The negative result shows that **this simple 49-feature ridge map does not generalize the paired finite intervention effect under the registered hard controls**.

Two distinct sources of prediction error remain unresolved. A correct target-digit task gradient may depend on information that an affine state-to-gradient map trained on 64 calibration puzzles cannot learn, especially when teachers solve only around half the goals. Separately, the frozen erasure dose is a finite physical edit, so task-gradient linearization may have a substantial nonlinear remainder. The current experiment **did not run a held-out answer-conditioned exact gradient oracle**; it cannot decompose its error into gradient-model approximation and Taylor-remainder contributions.

From BPD-OUT03, if the true native task margin Hessian has uniform operator norm bound M along both edit segments and physical kill and sham have equal norm r with row/tangent orthogonality, the following *conditional* bound holds:

\[
|D-\widehat{g}\cdot(\delta-s)|
  \leq \sqrt{2}\,r\,\|g-\widehat{g}\|+Mr^2.
\]

This bound is a mathematical consequence of physical-row/tangent orthogonality and Taylor's theorem, while SMC-24 does **not** certify M for the trained teachers. Its purpose is to separate prospective gradient approximation quality from native finite-dose curvature, without asserting which term dominates the observed failure.

## Original receipts and reproducibility

- Source-registration commit: f3a65a0, [SMC24_PROTOCOL.md](SMC24_PROTOCOL.md). Both engineering amendments: [SMC24_PREOUTCOME_AMENDMENT.md](SMC24_PREOUTCOME_AMENDMENT.md).
- Model/predictor source: [smc24_gradient_predict.py](smc24_gradient_predict.py); immutable-edit native scorer: [smc24_native_score.py](smc24_native_score.py).
- Source-independent Torch and NumPy regression checks: [test_smc24_gradient_predict.py](test_smc24_gradient_predict.py), 8/8 passed.
- Python standard-library independent case replay: [audit_smc24.py](audit_smc24.py); 128 fresh full-grid solutions, 4,096 selected goal slots, 8 cell-level uncertainty/acceptance decisions, all replay PASS.
- Full four-stage CI: [Actions #44106692708](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44106692708), success.
- Calibration-only ridge artifact [#14339061692](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44106692708/artifacts/14339061692), raw JSON SHA256 4db29e312db3c8a7036323005ca7d6bdc8d823cad5668ac12f8451fb01df5d79.
- Label-free new-test predictions and archived full source state [#14338218622](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44106692708/artifacts/14338218622), raw JSON SHA256 5b526ea387dfa0ac51afb72ed8e287471caa1681edec927bbb0a45029107701d.
- Native FP64 task results [#14338583629](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44106692708/artifacts/14338583629), raw JSON SHA256 1af7aca9cb3bba21c6a6789cb335b97f8fb1628aea0f1ef98758e7cb6ef26a18.
- Independent checksum-locked math/statistics replay [#14338713043](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44106692708/artifacts/14338713043); its archived report verifies all three hashes above. ZIP SHA256 f60f0f97b154498ade5d30b5589dc0968ca51503ba3224f926ccc0843ddf07fc.

**Scientific continuation.** To decide which cause limits gradient-mediated prediction, a new registered source-independent experiment should first measure the heldout **answer-conditioned native task gradient oracle** on the same frozen teacher and task metrics, and score its linearization residual against the measured finite edit D. This is a *diagnostic* that is not label-free or deployable. It would separate a high-fidelity local differential from a bad calibration-only gradient learner and guide whether effort belongs on a nonlinear chart, a richer predictor, a smaller intervention dose or a capability-matched stronger teacher pair. Any new primary gate requires an independent puzzle cohort and a protocol committed before its measurement.
