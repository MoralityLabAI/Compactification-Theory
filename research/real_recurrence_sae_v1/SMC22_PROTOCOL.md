# SMC-22: One-parameter LAD residual transfer against ridge, direct gradient and other-active LAD

**Prospective registration, November 16, 2026.** Commit this file *alone* before constructing any SMC-22 fresh heldout Sudoku state or scoring any new target erasure. This branch stacks on [SMC-21 PR #188](https://github.com/MoralityLabAI/Compactification-Theory/pull/188), whose strict **1/4** two-family calibrated-residual gate and earlier SMC-20 4/4 gate remain fixed historical evidence. The analysis below concerns a new loss function, new cohort seeds and distinct source-only fit receipts.

## Question and immutable sources

SMC-21 fit one fixed ridge scalar to source SAE native-erasure finite residuals on 64 original calibration puzzles and found pooled fresh MAE improvement, but only 1/4 receiving refits cleared its strict paired bootstrap and other-active comparator gates. The direct clean true-digit task gradient remains a strong baseline. This experiment asks whether fitting the **same one scalar for absolute error** better transfers residual information.

Reuse exactly the original B501/B701 recurrent Sudoku teacher weights, original SAE71 and independent refit SAE131/167 checkpoints, seven original SMC-13C position+visible-input-symbol Hungarian assignments, four frozen SMC-19 SAE71→refit feature assignments and original source-only SMC-21 calibration gradients and native erasure responses. No new training or checkpoint/fingerprint edits, no retuning of source calibration seeds or feature maps. Confirm the unchanged:
- SMC-11C original \`joint-full.json\` SHA \`7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087\`;
- SMC-13A independently trained teachers \`full.json\` SHA \`1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3\`;
- original SMC-13C conditioned \`conditioned.json\` SHA \`3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07\`;
- original four refits \`frozen.json\` SHA \`612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b\` from SMC-19 [run #43810248525 artifact #14199643904](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525/artifacts/14199643904);
- SMC-21 original 64-puzzle residual calibration \`residual_calibration.json\` SHA \`e7aa12a3da5b8eaa0132b5de45083ca1b2af5c38e556fef01e1f26d2b94f90af\` from [run #43822895405 artifact #14204423839](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43822895405/artifacts/14204423839).

An independent stdlib auditor rechecks all original SMC-19 checkpoint file bytes, old seven source feature SHA receipts and full 64-puzzle SMC-21 original residuals, their source-only calibration 48D gradient dot products and frozen ridge coefficients.

Four fixed comparisons: B501 SAE71→B501 SAE131, →B501 SAE167, B701 SAE71→B701 SAE131, →B701 SAE167. Reuse the SMC-21 source-only calibration eligibility mask in its exact original order. Its active-feature and different naturally active refit other-control selection was locked *before* original calibration native outcomes.

## Deterministic LAD/weighted median calibration

For each of the four pair calibrations, read every original eligible tuple \((R_A,R_B,R_O)\) from SMC-21's saved source-only calibration. \(R_A=E_A-g^\top \delta_A\), \(R_B=E_B-g^\top \delta_B\), and \(R_O=E_O-g^\top\delta_O\), all based on original same-teacher FP64 task gradients on the **original** 64 SMC-13 \`feature_calibration\` puzzles. No new source or target calibration inference is authorized.

For \(x\in\{R_A,R_O\}\), fit the **one-parameter, no-intercept empirical LAD**:
\[
\beta_x\in\arg\min_{\beta\in[-2,2]}\sum_{c\in C}|R_{B,c}-\beta x_c|.
\]
Use the following deterministic exact operational rule with original full FP64 stored numbers:
1. Exclude \(x_c=0\) from the ratio/weight list (their absolute losses contribute constants); \(|x_c|>0\) enters with \((r_c,w_c)=(R_{B,c}/x_c,|x_c|)\). Require all values finite and \(\sum_c|x_c|>10^{-12}\).
2. Sort by increasing ratio \(r_c\), then by original calibration puzzle index and slot index as a stable tie-break, **without any heldout score**.
3. Choose the *first* ratio whose cumulative weight is at least one half of the total weight, including equality. This is the **lower weighted median** in an interval of minimizers.
4. Freeze \(\beta_x=\operatorname{clip}_{[-2,2]}(r_{\rm median})\).
5. Record each ratio, weight, source index, sum of weights, objective value at the chosen fit and at SMC-21 original ridge beta, full original calibration eligibility counts and SHA of the resulting standalone \`lad_calibration.json\`.

Fit exactly two scalars per refit pair: source-LAD \(\beta_A^\mathrm{LAD}\) and other-active-LAD \(\beta_O^\mathrm{LAD}\), using **the identical** original target-calibration residuals and eligible goal masks. The frozen original SMC-21 source-ridge coefficient \(\beta_A^\mathrm{ridge}\) is an additional mandatory registered *strong* control; it has an equal one-parameter budget and was fitted before either SMC-21 or SMC-22 heldout outcomes. Require ≥24 calibration-eligible goals from ≥20 original calibration puzzles, both LAD scalars identified and no input/receipt tampering. Archive LAD calibration+independent audit **in a separate completed CI job before any new SMC-22 fresh test input or state is generated**. No validation hyperparameter search, tuned clipping, robustification threshold, extra intercept, source-seed selection or per-teacher success reweighting.

## Frozen new distribution

Construct two **fresh**, exact-unique, noncanonical 64-puzzle Sudoku cohorts with original SMC-13 randomized MRV full-grid generator and uniqueness-preserving dig seed=full-grid seed+1000000:
- \`smc22_novel_30\`: full solution seeds **202622030000..202622030063**, 30 holes;
- \`smc22_novel_34\`: full solution seeds **202622040000..202622040063**, 34 holes.

Verify completed Sudoku validity, exact unique-solution completion, positive alternating-rectangle invariant, exactly 30/34 blank positions, and no repeated puzzle inputs or complete solutions against **any original SMC-13, SMC-15, SMC-16, SMC-17, SMC-18, SMC-19, SMC-20, SMC-21** train/calibration/test family. Archive every puzzle and full solution with a SHA-256 manifest. Only the deterministic first four puzzles are an engineering pilot and cannot meet the registered full-size scientific gate.

Exactly eight goal tokens per puzzle come from the visible-blank positions at indices \(\lfloor j\,N_{\rm blanks}/8\rfloor\) for j=0..7, with the same SMC-19–21 preoutcome eligibility: source SAE71 strict positive top-one, refit mapped atom naturally positive in the target top-four, and a different naturally active other-active refit atom. Use the original frozen SMC-19 calibration Hungarian map and original SMC-15 SHA-seeded other-active choice; do not allow test outcomes or true digits to determine edit/mapping/eligibility.

## Four registered predictors and heldout erasure

At each eligible goal, the same frozen trained teacher yields the clean, *correct Sudoku digit minus highest clean non-target competitor* margin, with 3 FP64 recurrent continuation steps from original FP32 stage-one hidden state. The 48D **answer-conditioned** clean task gradient \(g=\nabla_hm(h)\) uses only the clean state and originally visible-mask-chosen goal, after eligibility lock. Edits are exactly original source SAE71 native physical contribution \(a\), matched new SAE contribution \(b\), and equal-physical-dose other-active target SAE direction \(o\).

Before measuring target matched-refit native erasure \(E_b\), freeze the clean-gradient direct, SMC-21 ridge, new source LAD and new other-active LAD predictions:
\[
D=g^\top b,\qquad
S_\mathrm{LAD}=D+\beta_A^\mathrm{LAD}(E_a-g^\top a),\quad
S_\mathrm{ridge}=D+\beta_A^\mathrm{ridge}(E_a-g^\top a),\quad
O_\mathrm{LAD}=D+\beta_O^\mathrm{LAD}(E_o-g^\top o).
\]
Only the target native refit erasure margin drop \(E_b=m(h)-m(h-b)\) is the withheld outcome. Both source correction and other-active LAD each observe one extra native intervention on the same eligible goals and use exactly one scalar coefficient; direct gradient needs none. All four use the same correct digit and frozen clean rival and gradient, and all means/errors use an identical eligible denominator. Record full 48D physical edit vectors and gradient, natural SAE coefficients, raw erasure drops, all 4 predictions/errors/signs, and all ineligible slots/reasons.

## Frozen statistical scoring and success

Per eligible target define absolute errors \(\ell_D=|E_b-D|\), \(\ell_R=|E_b-S_\mathrm{ridge}|\), \(\ell_S=|E_b-S_\mathrm{LAD}|\), \(\ell_O=|E_b-O_\mathrm{LAD}|\). The registered **paired gains** on the *same* eligible goals are \(G_D=\ell_D-\ell_S\), \(G_R=\ell_R-\ell_S\) and \(G_O=\ell_O-\ell_S\). Positive gain favors source-LAD. For each family × pair, preserve all 64 independent puzzle IDs and perform exactly **2,000** puzzle-cluster resamples (all eligible goals of every drawn puzzle), with Python \`random.Random(202622130+100*pair_index+len(family))\`. Use sorted indices **50 and 1949** for descriptive empirical 95% intervals and mark invalid any sample of all ineligible puzzles.

**Prospective source-LAD positive scientific gate per pair on BOTH new families** requires:
1. All calibration source/other-LAD fits and original ridge source coefficient identified, ≥24 original eligible calibration goals and ≥20 original calibration puzzles; ≥24 new eligible heldout goals in ≥20 new puzzles and all 2,000 resamples valid.
2. 95% lower paired bootstrap endpoints **strictly greater than zero for all three** \(G_D,G_R,G_O\).
3. Pooled source-LAD MAE / direct-gradient MAE ≤**0.95**, with \(\sum\ell_D>10^{-8}\).
4. Source-LAD native refit erasure sign accuracy ≥ direct clean-gradient sign accuracy under the unchanged \(|x|\le10^{-8}\) zero convention.

Every pair has two fresh families. Missing one gate fails the pair, even if source-LAD pooled MAE improves or all eight mean gains are positive. Preserve negative/low-coverage/null results and other-LAD dominance. Report exact same-goal MAE, all intervals, per-puzzle paired differences, calibration distributions and heldout goal/feature frequency. The four pair comparisons share just two frozen teachers and a related synthetic puzzle generator. The 4×2 count is **not** an independent model-population inference.

## Mathematics, audit and interpretation

Pure math OLM-22 should prove the deterministic weighted-median solution of a one-parameter no-intercept LAD objective under clipping; exhibit exact rational examples where LAD beats ridge in-sample but loses under an unseen shifted target law, and derive a coupling-based absolute-loss shift bound \(\left|\mathbb E_\nu|y-\beta x|-\mathbb E_\mu|y-\beta x|\right|\le\mathbb E_{\pi}(|y'-y|+2|x'-x|)\) for \(|\beta|\le2\), allowing a conditional test-transfer certificate if the original calibration LAD-over-ridge gap exceeds both distribution-shift bounds. A positive local decoder relationship would remain *operational* given the full-rank inactive encoder and SAE reparameterization obstructions in OLM-18–20.

An independent standard-library-only auditor will re-verify original 4 new SAE file SHA-256 and original 7 conditioned feature assignment hashes; original 64-calibration values and exact two LAD weighted medians and original ridge coefficients; new Sudoku uniqueness and source disjointness; SHA-seeded other-active choice and FP64 physical decoder edit reconstruction; every 48D clean task-gradient prediction and target outcome error; all 24,000 clustered resamples (3 paired gains × 8 conditions) and the original gate decisions. The auditor is **numerical receipt replay, not independent neural model execution**. CI execution success and preregistered scientific success are different. No causal native-feature identity or answer-free deployable Sudoku solver follows from a positive result.

SMC-19 2/4, SMC-20 4/4 and SMC-21 1/4 gates remain unchanged historical records. No existing branch merges, model/SAE refits, or new training allowed in this campaign.
