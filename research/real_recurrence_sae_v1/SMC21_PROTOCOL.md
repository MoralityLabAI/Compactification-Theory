# SMC-21: Prospective calibration-only SAE erasure-residual transfer against a strong direct-gradient baseline

**Prospectively registered November 16, 2026.** This protocol is committed *alone*, before any new SMC-21 calibration gradients or fresh test states are computed. Stacked on original source-locked [SMC-20 PR #186](https://github.com/MoralityLabAI/Compactification-Theory/pull/186), whose 4/4 primary gate and **stronger prespecified direct-gradient-only control** remain unchanged. The original SMC-19 refit causal correspondence result remains 2/4. No heldout SMC-20 data is used to tune the SMC-21 correction.

## Task question

SMC-20's anchored native refit prediction \(\widehat E_B=E_A+g^\top(\delta_B-\delta_A)\) passed its original source-only and dose-matched-other controls on 4/4 frozen SAE refit pairs, but the simpler **direct clean task-gradient predictor** \(g^\top\delta_B\) had better pooled absolute error. The exact OLM-20F identity shows that anchoring adds the old SAE's finite residual \(R_A=E_A-g^\top\delta_A\), which may differ from the new SAE's finite residual \(R_B=E_B-g^\top\delta_B\).

SMC-21 asks whether a **single calibration-only scalar** mapping \(R_A\) to \(R_B\) improves *genuine fresh-cohort* native-erasure magnitude predictions over direct clean gradient, and over an **equal-budget other-active target SAE residual** calibration control.

## Immutable source and calibration sources

Reuse original B501 and B701 frozen 44,001-parameter, four-loop Sudoku teachers and old SAE71, plus four SMC-19 refitted SAEs: B501 SAE131, B501 SAE167, B701 SAE131 and B701 SAE167. No training of any model or SAE is permitted. Read and verify all original SMC-11C joint \`joint-full.json\` SHA \`7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087\`, original independent-teacher SMC-13A \`full.json\` SHA \`1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3\`, SMC-13C source assignment \`conditioned.json\` SHA \`3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07\`, and **all seven** original conditioned Hungarian source assignments. Exact SMC-19 four new SAE trained snapshots, their frozen feature matches, checkpoint byte hashes and original calibration sign orientations come only from its original [artifact #14199643904](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525/artifacts/14199643904). Frozen \`frozen.json\` SHA must equal \`612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b\`.

Four fixed within-teacher pairs in SMC-19 order: B501 SAE71→SAE131, B501 SAE71→SAE167, B701 SAE71→SAE131, B701 SAE71→SAE167.

The *only* calibration cohort is the **original 64-puzzle SMC-13 feature_calibration** dataset, disjoint from all SMC-21 test cases, SAE training/validation, and earlier SMC-20 fresh cohorts. No SMC-19 test or SMC-20 test state is permitted for fitting. Reconstruct the four frozen SMC-19 original pair/feature assignments and eligible masks on these 64 original puzzles before any target response, identical to SMC-19 original preoutcome criteria: source SAE71 code top-one strict positive; refit matched feature naturally positive among its top-four; a different naturally active target refit feature available. The other-active target SAE feature uses original SMC-15 seeded source-blind selection with \`pair_index\`, split literal \`feature_calibration\`, original calibration puzzle seed, slot and target matched ID. Its physical edit is norm-matched to the target new SAE matched edit.

For every calibration-eligible goal, compute the exact **clean FP64 task-margin gradient** \(g=\nabla_hm(h)\), where \(m\) is the *correct hidden Sudoku digit minus the strongest clean competing digit*, after three frozen FP64 recurrent continuation steps from the original FP32 stage-one state. The true digit enters after geometry selection. With the original single unmodified teacher, record the source SAE71 native edit \(\delta_A\), new refit matched native edit \(\delta_B\), and other-active equal-dose native edit \(\delta_O\). The original frozen SMC-19 calibration artifact already contains their measured native erasure drops \(E_A,E_B,E_O\); **use those original source-locked calibration causal values**, cross-check original source/refit feature IDs and eligibility and calculate the fresh gradient, without measuring a new target effect to choose the fit.

Define calibration finite residuals:
\[
R_A=E_A-g^\top\delta_A,\qquad
R_B=E_B-g^\top\delta_B,\qquad
R_O=E_O-g^\top\delta_O.
\]

## Frozen one-parameter ridge fit and hard counterfactual

For each of the four pairs, fit exactly two **univariate no-intercept** scalars by the same fixed algebraic rule, one from \(R_A\) to \(R_B\), and one from \(R_O\) to \(R_B\). On all calibration-eligible goals \(C\), for \(x\in\{R_A,R_O\}\),
\[
\boxed{\beta_x=\operatorname{clip}_{[-2,2]}
 \frac{\sum_{c\in C}x_cR_{B,c}}
      {1.05\sum_{c\in C}x_c^2}}.
\]

The constant 1.05 is a fixed 5% relative ridge shrinkage. If \(\sum x_c^2\le10^{-12}\), freeze \(\beta_x=0\) and flag *degenerate*; the pair fails the registered gate. Each fit is one scalar with the same target, labels, calibration episodes and parameter budget, no intercept, no hyperparameter search, no validation/seed tuning. Require **≥24** calibration-eligible goals from ≥20 of the original 64 puzzle IDs and finite coefficients. Freeze \(\beta_A,\beta_O\), every calibration-eligible record (gradient 48D, physical edits, native effects, residuals), original paired SMC-19 frozen-map hash and original four checkpoint SHA receipts into a **separate immutable calibration artifact** \`smc21/frozen/residual_calibration.json\`. Upload this artifact and verify its SHA **before generating any SMC-21 novel Sudoku state**. The source-trained/SAE refs are loaded again from the frozen artifact in independent pilot and full test jobs.

## New data, preserved eligibility and predictions

Only after calibration locks have been finalized and uploaded, generate two new complete-solution-disjoint exact-unique structurally noncanonical Sudoku cohorts (each 64 puzzles):
- \`smc21_novel_30\`, full-grid seed bases **202621030000..202621030063**, 30 dug holes.
- \`smc21_novel_34\`, bases **202621040000..202621040063**, 34 dug holes.

Use original SMC-13 randomized MRV complete-grid and exact uniqueness-preserving \`dig_unique(y, seed+1000000, holes)\`, full positive alternating-rectangle invariant, full 81-target validity. Every test puzzle and complete solution must be different from SMC-13 original train/calib/heldout and all 64-puzzle SMC-15/16/17/18/19/20 novel families. Preserve SHA-256 full new cohort fingerprints. A four-puzzle prefix is an engineering pilot, never a scientific pass. Eight blank positions per puzzle selected from visible input at index \(\lfloor j\,n_{\rm missing}/8\rfloor\) for \(j=0..7\), with the SMC-19 source/reference strict-top1 and target-matched-naturally-active plus other-control eligibility criteria. No goal selected based on true digit, target erasure response, prediction error or gradient magnitude.

Measure \(g\) on a clean FP64 three-step recurrent rollout with frozen correct-digit rival, and original source native erasure \(E_A\) and other-active target native erasure \(E_O\), **before** scoring target new SAE matched native erasure \(E_B\). On all eligible goal instances freeze these predictions:
\[
\widehat E_B^{\rm direct}=g^\top\delta_B,\quad
\widehat E_B^{\rm source\text{-}residual}
 =g^\top\delta_B+\beta_A\underbrace{(E_A-g^\top\delta_A)}_{R_A},
\]
\[
\widehat E_B^{\rm other\text{-}residual}
 =g^\top\delta_B+\beta_O\underbrace{(E_O-g^\top\delta_O)}_{R_O}.
\]

All three predictions use the **same clean gradient** and frozen receiving target edit. The two residual-corrected predictors each require exactly one observed other erasure and one calibration-only scalar, giving an equal-budget source-vs-other comparison. Record unshrunk unit anchor \(g^\top\delta_B+R_A\) as a non-gating SMC-20 historical reference and physical edit distance/cosine descriptively; never tune beta on heldout labels.

## Frozen uncertainty and primary gates

On the exact same eligible goals define \(\ell_D=|E_B-\widehat E_B^{\rm direct}|\), \(\ell_S=|E_B-\widehat E_B^{\rm source}|\), \(\ell_O=|E_B-\widehat E_B^{\rm other}|\). Primary paired improvements are **\(I_D=\ell_D-\ell_S\)** and **\(I_O=\ell_O-\ell_S\)**. Preserve all 64 puzzle IDs and eligible/ineligible slots. Use exactly **2,000** resamples of 64 puzzle IDs, clustering every eligible token from a sampled puzzle, with \`random.Random(202621130+100*pair_index+len(family))\`; empirical 95% percentile bounds at sorted indices **50,1949**. Invalid draws with zero eligible tokens mean bootstrap gate fails. Also report direct-vs-source and other-vs-source sign accuracy with common \(|x|\le10^{-8}\) zero convention, all per-puzzle effects, errors and nonlinearity ratios.

**Preregistered success per pair on BOTH fresh cohorts:**
1. Each calibration fit identified and not degenerate with ≥24 calibration-eligible goals in ≥20 original calibration puzzles. Test coverage ≥24 eligible goals in ≥20 new puzzle IDs, all 2,000 bootstrap draws valid.
2. Lower 95% paired bootstrap endpoint for **\(I_D>0\)** and separately for **\(I_O>0\)**, each strictly positive.
3. Source-calibrated pooled MAE ratio \(\sum\ell_S/\sum\ell_D\le0.95\), with \(\sum\ell_D>10^{-8}\).
4. Source-calibrated target-erasure **sign accuracy ≥ direct-gradient** sign accuracy on the identical eligible goals.

Report per family gate components, interval endpoints, calibration fit coefficients and all failures. If *all* fitted \(\beta_A\) are near zero, the registered experiment can fail; this is informative rather than a reason to retune. A positive result would add evidence for source finite-residual transfer *after* the direct gradient is available. Under full-rank inactive encoder features, it still would not identify intrinsic semantic SAE atoms or deployed Sudoku capability. The target digit remains an answer-conditioned evaluation input.

## Independent audit, applied mathematics and limits

Write an independent stdlib receipt checker that (a) verifies original three source JSON hashes, all seven SMC-13C conditioned Hungarian map SHA and four SMC-19 checkpoint file-byte SHA; (b) independently reconstructs all four calibration ridge coefficients from raw frozen \(R_A,R_B,R_O\) and the exact fixed denominator; (c) checks full new Sudoku roster/goal masks and heldout native decoder vectors reconstructed from the archived SAE decoder matrices and nonnegative clean codes; (d) computes clean gradient dot products, each prediction, all three absolute errors, signs, and original 16,000 clustered resamples; (e) checks every original registered decision. **Green CI is execution and evidence-integrity, separate from scientific success.**

Pure mathematics OLM-21 should prove that calibration-optimal squared residual transfer does **not** guarantee an improvement in heldout *absolute* error or sign accuracy; derive a quantitative shift bound between calibration and heldout quadratic risks for frozen \(\beta\) with bounded residual covariates; and give exact rational counterexamples for non-transferability even with equal source residual distributions. OLM-20's identity remains correct for every state; its ability to predict heldout benefit depends on stability of joint residual relations.

No new training, SAE refit, new mapping, new assistant-selected threshold or retrospective alteration to SMC-19/20 results is authorized. Keep SMC-21 and OLM-21 draft, stacked on their parents pending CI and audit.
