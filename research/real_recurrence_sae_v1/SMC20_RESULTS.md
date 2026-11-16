# SMC-20: Task-gradient transport predicts frozen SAE refit erasures on fresh Sudoku

**Executed November 16, 2026.** Original [SMC-20 prospective protocol commit \`eb4b6c4\`](https://github.com/MoralityLabAI/Compactification-Theory/commit/eb4b6c419fd741c4915d105235a9979426398710) precedes any SMC-20 fresh test state. This campaign follows source-locked [SMC-19 PR #184](https://github.com/MoralityLabAI/Compactification-Theory/pull/184), whose strict *SAE-native causal sign correspondence* test remains **2/4** pairs. **SMC-20 tests a different, predeclared task-margin prediction endpoint.**

The complete [hosted Actions run #43816907745](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43816907745) passed source-lock checks, 14 initial CPU regression and independent audit mutation tests, its four-puzzle engineering pilot, the full **four refit pairs × two fresh 64-puzzle families**, and the separate standard-library raw receipt replay. **All four receiving SAE refit pairs passed the SMC-20 registered prediction gate on both novel families, 8/8 conditions.** Source code head \`5814e5828fc22fe84ec7cb42cbc617ff3af1bcb8\` produced the original audited full evidence.

## Immutable experiments and heldout units

All trained model weights, four SMC-19 new SAE seeds (B501/B701 each with refits 131/167), original SAE71/89 and three base source JSON SHA-256s are unchanged. All seven original position/digit-conditioned SMC-13C Hungarian matches, four original calibration-only SMC-19 refit Hungarian matches, and four independent refit checkpoint *file bytes* were reverified against the [SMC-19 frozen weights and calibration archive #14199643904](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525/artifacts/14199643904), with frozen \`frozen.json\` SHA-256 \`612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b\`.

Two new **source-disjoint** noncanonical exact-unique Sudoku groups, each comprising 64 independently seeded puzzles, use full-grid seeds \`202620030000..202620030063\` (30 holes) and \`202620040000..202620040063\` (34 holes), with visible-mask-only selection of eight blank goal tokens per puzzle. Full complete solution and visible input rosters were regenerated and checked against every SMC-13/15/16/17/18/19 previous cohort and each other. Fresh manifest hashes: novel30 \`51a204aeba7bbc0f6a1e8dc75e827768b91ef88eddd559188222f610548d5557\`, novel34 \`b80b14b75afdebe792bd8cb668fb9d98d6496980ed2c3d4c147da4dddd5412fe\`.

Only goals with strict positive source SAE71 top-one and a *naturally active* calibration-matched new SAE feature plus another distinct active new SAE feature qualify. The exact source/refit physical decoder contributions and equal-dose other-active decoder control are fixed from clean hidden state and visible input before any target refit erasure. Ineligible goals and reason codes remain in full raw rows. The full test includes **1,352 eligible pair/family/goal observations** across 512 pair/puzzle-family cases and 4,096 selected goal records.

## Frozen prediction test and primary success

For the **same unmodified recurrent Sudoku teacher**, let \(\delta_A\) and \(\delta_B\) be the original SAE71 and new matched refit SAE native *physical* decoder edits at a naturally eligible goal. The reference causal margin drop \(E_A=m(h)-m(h-\delta_A)\) is observed before target scoring. The scalar \(m(h)\) is the correct Sudoku digit logit minus the strongest **clean** non-target competitor after three fixed FP64 recurrent steps, starting from the original FP32 stage-one hidden states. Obtain the full 48D FP64 clean-task gradient \(g=\nabla m(h)\) before scoring the new refit edit.

The **primary anchored-gradient prediction** is
\[
\hat E_B=E_A+g^\top(\delta_B-\delta_A),
\]
which predicts the heldout target new-refit native effect \(E_B\). The two registered comparisons are against (1) **transferring reference effect unchanged** \(\hat E_B^A=E_A\) and (2) the **same-dose other naturally active refit SAE decoder** surrogate \(\hat E_B^O=E_A+g^\top(\delta_O-\delta_A)\). A positive paired difference in absolute prediction error means the gradient correction is better on the exact same eligible goal.

Per pair **on both** fresh families, the frozen scientific gate requires ≥24 eligible goals from ≥20 distinct puzzles, 2,000 valid 64-puzzle-cluster bootstrap draws, positive lower 95% bootstrap endpoints for **both** mean paired MAE improvements, pooled \(\sum|\hat E_B-E_B|/\sum|E_A-E_B|\le0.90\), and anchored target-effect sign accuracy at least as high as the source-only sign accuracy. All eight conditions pass without changing any threshold or sample eligibility.

## Registered eight-condition results

Positive values in the improvement columns mean a reduction in absolute refit-native-effect prediction error. All 95% intervals are *descriptive 64-puzzle cluster bootstrap intervals*, with 2,000 fixed-seed resamples.

| Receiving refit SAE | Family | Eligible goals / puzzles | Mean MAE gain over source-only [95% CI] | Mean MAE gain over other-active [95% CI] | Anchored / source-only MAE | Registered gate |
|---|---|---:|---:|---:|---:|---|
| B501 SAE131 | novel30 | 172 / 63 | **+0.8439 [0.7054, 0.9876]** | +1.1290 [0.9469, 1.3352] | 0.2651 | **PASS** |
| B501 SAE131 | novel34 | 167 / 63 | **+0.9613 [0.8268, 1.1125]** | +1.4246 [1.1262, 1.7472] | 0.2412 | **PASS** |
| B501 SAE167 | novel30 | 218 / 64 | **+0.7756 [0.6814, 0.8672]** | +1.3550 [1.1495, 1.6055] | 0.2871 | **PASS** |
| B501 SAE167 | novel34 | 215 / 64 | **+0.7112 [0.6092, 0.8195]** | +1.1136 [0.8982, 1.3453] | 0.3109 | **PASS** |
| B701 SAE131 | novel30 | 149 / 60 | **+0.7685 [0.6232, 0.9432]** | +1.6718 [1.2911, 2.0578] | 0.2849 | **PASS** |
| B701 SAE131 | novel34 | 154 / 61 | **+0.7081 [0.5659, 0.8618]** | +1.3204 [1.0963, 1.5506] | 0.3011 | **PASS** |
| B701 SAE167 | novel30 | 137 / 60 | **+0.9529 [0.8020, 1.1138]** | +1.4683 [1.1790, 1.8040] | 0.2221 | **PASS** |
| B701 SAE167 | novel34 | 140 / 59 | **+0.9161 [0.7624, 1.0830]** | +1.2835 [1.0022, 1.6056] | 0.2353 | **PASS** |

The anchored-gradient absolute target-effect MAE ranges from **0.272 to 0.321**, while the original source-only MAE ranges from **1.013 to 1.267** across the eight conditions. The pooled source-normalized absolute error ratio is **0.26964** (73.04% less error in the pooled eligible set). The anchored predictor gets the finite native refit effect sign right on **1,217/1,352 (90.0%)** eligible goals, compared with **894/1,352 (66.1%)** for source-only.

These results remain a **same-teacher, answer-conditioned mechanistic diagnostic**. The task gradient is computed for the **true Sudoku digit** and clean rival, so this numerical prediction is **not an inference-time heuristic for solving new Sudoku puzzles without their answers**. Four comparisons also share two teacher weights and repeated synthetic puzzle-generating procedures. A positive score is evidence for predictive local task geometry, not proof of semantically intrinsic SAE features or an independently replicated population law.

## Prespecified descriptive direct clean-gradient control: an important limitation

The protocol also preserved the simpler direct clean-gradient predictor
\[
\hat E_B^{\rm direct}=g^\top\delta_B
\]
as a *descriptive* control, outside the preregistered scientific success gate. A separate numerical replay of every archived per-goal output found its **pooled MAE ratio 0.23504** relative to the original source-only MAE, smaller than the **0.26964** ratio for the anchored estimator. Direct-gradient prediction had lower observed MAE point estimates in **seven of eight** pair/family conditions, although the pairwise puzzle-cluster uncertainty for that retrospective contrast often spans zero. Its target-effect sign matched **1,274/1,352 (94.2%)** eligible goals, compared with **1,217/1,352 (90.0%)** for the anchored estimator.

This controls the interpretation of the registered 4/4 pass: **clean task gradients are highly predictive of refit physical decoder erasure**, and the additional source-native erasure anchor does **not** show general additional predictive value over the direct gradient on these heldout conditions. This is a prespecified descriptive comparison, evaluated with a **post-result secondary pooled analysis** that leaves all original gates unchanged.

The exact identity derived in the post-result [OLM-20F extension](https://github.com/patrickdugan/MathCan/pull/107) explains this:
\[
R(v)=E(v)-g^\top v,\quad
\hat E_B^{\rm anchored}=g^\top\delta_B+R(\delta_A),\quad
E_B=g^\top\delta_B+R(\delta_B).
\]
Anchoring is more accurate than direct gradient prediction exactly when
\(R(\delta_A)[R(\delta_A)-2R(\delta_B)]<0\). The independently observed source erasure helps only if its finite-nonlinearity residual approximates the refit's residual sufficiently closely. Exact-rational counterexamples in MathCan demonstrate where the source anchor helps or hurts.

## Source hashes and independently replayed evidence

- Original fully green [SMC-20 Actions run #43816907745](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43816907745), original experimental code SHA \`5814e5828fc22fe84ec7cb42cbc617ff3af1bcb8\`; engineering pilot, full four-pair run, and source-separated independent audit passed.
- Full \`full.json\` plus stdlib \`audit.json\`: [artifact #14202004711](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43816907745/artifacts/14202004711), ZIP SHA-256 \`cb68a796ab5b0213a283105876e502f3bdc5bc8cc40a67e494d9793d7f1bfaed\`.
- Exact original \`full.json\` bytes SHA-256 \`f828b370c96edccfdcb70d51d3ef58ef8d1b3d7c6516a2884024bf5ec1ce5932\`, **20,140,128 bytes**. Independent standard-library \`audit.json\` SHA-256 \`80555ab71fde1816e392092ba5bded268e61be815f513aeca25d9638a96d20ca\`.
- Immutable four newly trained SMC-19 SAE checkpoint file fingerprints and original calibration \`frozen.json\`: [source artifact #14199643904](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525/artifacts/14199643904), archive ZIP SHA \`43630f8b9e6e7fbd07f8c668dc607f8b53c3a6b78cd47ce43aefe33bc005cb9e\`, frozen source JSON SHA \`612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b\`.
- [Machine-readable full eight-condition summary](SMC20_SUMMARY.json) preserves both registered comparative uncertainty intervals, all four gate components, sign accuracies, physical decoder distance statistics, source/target direct-error diagnostics, original evidence hashes and cohort fingerprints.

The original independent audit is a **separate standard-library source**. It checks all original frozen checkpoint bytes and calibration maps, full visible-only goal positions and exact-unique Sudoku manifests, 48D decoder column reconstruction, matched-dose other-active SHA-seeded selection, clean gradient dot products, three formulas and error figures, target sign calculations, all **16,000 puzzle cluster bootstrap replicates**, and registered decisions. A **second separate archived-JSON review** also recomputed all eight intervals and four refit checkpoint SHA-256 hashes using independent code. Neither reviewer independently retrains an SAE or reruns the recurrent neural forward/gradient passes; the source experiment remains the verified numerical model execution.

## Mathematics and next prospective discriminator

[OLM-20 PR #107](https://github.com/patrickdugan/MathCan/pull/107) derives the exact difference-sensitive error bound
\[
|E_B-E_A-g^\top(\delta_B-\delta_A)|
\le \tfrac M2(\|\delta_A\|+\|\delta_B\|)\|\delta_B-\delta_A\|,
\]
under an independently verified full-triangle Hessian operator-norm ceiling M, along with a conditional strict target-sign certificate. It proves orthogonal-gauge covariance and the exact residual transfer criterion for anchoring vs direct task-gradient prediction. This source experiment **does not establish such a global M**, and the original bound is not a numerical certificate for its checkpoints.

The next preregistered campaign should prioritize **direct gradient versus calibrated finite-residual transfer**, selecting residual predictors only from a separate calibration cohort and testing on newly disjoint Sudoku states. Its primary result should compare against the strong direct-gradient predictor with equal answer-conditioning and compute limits, rather than relying on the weaker source-only/native-other controls. That test would determine whether the old SAE reference intervention adds information after the task gradient is available.
