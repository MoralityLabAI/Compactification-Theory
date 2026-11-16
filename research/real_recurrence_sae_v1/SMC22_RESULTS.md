# SMC-22: MAE-optimal source calibration fails to beat the original ridge consistently

**Executed November 16, 2026.** The original standalone [prospective protocol commit \`b208c1e\`](https://github.com/MoralityLabAI/Compactification-Theory/commit/b208c1e7a63ac3e0d3008564d25bded5a2f80f52) preceded any SMC-22 calibration computation or fresh heldout Sudoku generation. The complete [GitHub Actions run #43827985476](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43827985476) passed every stage: (1) exact original-source weighted-median LAD calibration and independent audit, (2) independent four-puzzle pilot and audit, (3) four SAE refits × two fresh 64-puzzle families and independent stdlib full audit.

**Preregistered scientific verdict: 0 of 4 receiving SAE refits passed the original two-family source-LAD gate.** One of eight individual refit/family conditions passed. The source-LAD numerical prediction nevertheless beat the direct clean-gradient baseline in mean MAE in all eight fresh conditions. Its advantage over the **original SMC-21 source ridge**, fitted to exactly the same original calibration source and frozen before the SMC-21 heldout outcomes, was small and failed the paired confidence requirement in seven of eight new conditions. Preserve all failures and historical SMC-19 2/4, SMC-20 4/4 and SMC-21 1/4 results.

## Immutable sources, strict calibration separation and target

Retain original SMC-13 B501/B701 frozen recurrent Sudoku teachers, their SAE71 sources and the four independently fitted SAE131/167 receiving dictionaries from [SMC-19 frozen artifact #14199643904](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43810248525/artifacts/14199643904). No original teacher or SAE is retrained. Reverify all original SMC-11C/SMC-13A/SMC-13C JSON SHA-256 receipts, seven original visible-position/digit-conditioned Hungarian assignments and four SMC-19 newly trained SAE checkpoint byte fingerprints. The frozen refit source \`frozen.json\` SHA is \`612b7bd0bd206043d0766b398e85dd3e6bf62e9ce3d8427ecdbf4cbd4c0db52b\`.

The **only** residual calibration is the original 64 SMC-13 \`feature_calibration\` Sudoku puzzle dataset recorded, including native physical SAE edits, 48D task gradients and measured original/target/other finite erasures, in SMC-21's original source-only [artifact #14204423839](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43822895405/artifacts/14204423839). Its immutable \`residual_calibration.json\` SHA is \`e7aa12a3da5b8eaa0132b5de45083ca1b2af5c38e556fef01e1f26d2b94f90af\`. No SMC-21 test outcomes or SMC-20 test states enter the SMC-22 calibration. The original SMC-21 calibration-ridge coefficients remain frozen, with zero additional parameters.

For the *same trained frozen teacher*, the heldout target is the native erasure effect of its matched refit SAE feature \(b\). Fix the correct Sudoku digit and highest **clean** non-target competing digit and continue the same post-first-loop hidden state for three FP64 recurrent blocks. Let \(E(v)=m(h)-m(h-v)\), \(g=\nabla_hm(h)\), and \(R(v)=E(v)-g^\top v\), with source SAE71 native physical contribution \(a\) and one other naturally active, equal-dose target SAE contribution \(o\). All feature mapping and activity masks are original SMC-19 preoutcome visible-state selections; the new target \(E(b)\) is scored after coefficients and prediction formulas are frozen.

The new source-LAD prediction is \(\hat E_b^{\rm LAD}=g^\top b+\beta_A^{\rm LAD}R(a)\). Its mandatory same-eligible-goal baselines are:
- **Strong direct gradient:** \(g^\top b\).
- **Original source ridge:** \(g^\top b+\beta_A^{\rm ridge}R(a)\), from SMC-21's original source-only calibration.
- **Equal-parameter, equal-one-additional-native-erasure other-active LAD:** \(g^\top b+\beta_O^{\rm LAD}R(o)\).

The receiving target edit, source native erasure, other-active erasure and correct-digit clean gradient are evaluated on the same original post-first-loop teacher. The source and other correction each fit **one** scalar on the **same** original calibration target and eligibility mask. The direct gradient requires no native erasure observation.

## Source-only deterministic weighted-median fits

Source-LAD and other-active-LAD solve \(\min_{\beta\in[-2,2]}\sum_c|R_B(c)-\beta R_X(c)|\) with the **lower weighted median** of ratios \(R_B/R_X\), weighted by \(|R_X|\). Exclude exact zero covariates, use source puzzle/goal index for ties, and clamp \(\beta\) into [-2,2]. The original complete set of sorted ratio/weight/puzzle/slot records, corresponding fitted objectives, old ridge objective and independent calibration audit are archived **before any new SMC-22 heldout puzzle is constructed**.

| New refit SAE | Original calibration goals / puzzles | New \(\beta_A^{\rm LAD}\) | Same-budget \(\beta_O^{\rm LAD}\) | Original ridge \(\beta_A^{\rm ridge}\) | LAD calibration error / ridge |
|---|---:|---:|---:|---:|---:|
| B501 SAE131 | 198 / 63 | 0.411250 | 0.097462 | 0.502533 | 34.9201 / 35.7187 |
| B501 SAE167 | 222 / 62 | 0.469490 | 0.129229 | 0.423139 | 50.9897 / 51.2765 |
| B701 SAE131 | 165 / 61 | 0.252806 | 0.261622 | 0.396573 | 35.0450 / 35.7730 |
| B701 SAE167 | 142 / 59 | 0.341813 | 0.128479 | 0.572915 | 35.0213 / 35.9712 |

All four source and four other-active fit coefficients are identified and nondegenerate. The LAD fit, by construction, improves calibration absolute error over the frozen regularized ridge fit on all four original calibration sets. The test asks whether this improvement generalizes when the *original ridge is itself a compulsory heldout control*.

## Registered heldout results: 0/4 pairs, 1/8 individual families

Two new noncanonical exact-unique Sudoku cohorts of 64 puzzles each use new full-grid seed bases **202622030000** (30 blanks) and **202622040000** (34 blanks). All complete solutions and visible input puzzles are distinct from every original SMC-13, SMC-15/16/17/18/19/20/21 test and original training/calibration group. Each puzzle has eight visible-mask-selected missing goal tokens; original matched feature must be naturally active and the other active refit target decoder must be available. Ineligible tokens remain visible with explicit abstention reasons.

The frozen success criterion for a receiving refit **on both fresh families** requires ≥24 eligible goals in ≥20 puzzles, two identified calibration LAD fits and original ridge, 2,000 valid puzzle-cluster draws, **positive 95% lower paired MAE improvement versus each of direct, ridge and other-LAD**, pooled LAD/direct MAE ratio ≤0.95, and source-LAD correct target-effect sign count no lower than direct. Each interval below resamples **all 64 puzzle IDs**, preserving each puzzle's selected goals.

| Receiving refit | Family | Eligible goals / puzzles | LAD gain vs direct gradient [95% CI] | LAD gain vs *original ridge* [95% CI] | LAD gain vs other-LAD [95% CI] | Registered gate |
|---|---|---:|---|---|---|---|
| B501 SAE131 | 30 holes | 173 / 62 | +0.03341 [0.00946, 0.05646] | +0.00768 [0.00068, 0.01521] | +0.02303 **[−0.00288, 0.05112]** | **FAIL** |
| B501 SAE131 | 34 holes | 165 / 61 | +0.05577 [0.02335, 0.08817] | +0.00312 **[−0.00494, 0.01137]** | +0.03574 [0.00288, 0.06933] | **FAIL** |
| B501 SAE167 | 30 holes | 207 / 64 | +0.06397 [0.03609, 0.09257] | −0.00009 **[−0.00346, 0.00300]** | +0.05380 [0.02658, 0.08117] | **FAIL** |
| B501 SAE167 | 34 holes | 176 / 61 | +0.09607 [0.06610, 0.12881] | +0.00416 [0.00028, 0.00818] | +0.08506 [0.05350, 0.11888] | **PASS** |
| B701 SAE131 | 30 holes | 172 / 62 | +0.03111 [0.01780, 0.04530] | −0.00113 **[−0.01145, 0.00820]** | +0.02848 [0.00430, 0.05493] | **FAIL** |
| B701 SAE131 | 34 holes | 147 / 62 | +0.03609 [0.01895, 0.05219] | −0.00097 **[−0.01210, 0.01050]** | +0.02380 **[−0.00515, 0.05725]** | **FAIL** |
| B701 SAE167 | 30 holes | 129 / 56 | +0.05107 [0.02877, 0.07667] | +0.00052 **[−0.01943, 0.01948]** | +0.04679 [0.01264, 0.08972] | **FAIL** |
| B701 SAE167 | 34 holes | 129 / 60 | +0.04052 [0.01704, 0.06546] | +0.00341 **[−0.01385, 0.02106]** | +0.03955 [0.00466, 0.08014] | **FAIL** |

All eight new cohorts pass the coverage and the registered pooled LAD/direct MAE ≤0.95 criterion. All eight have a **positive 95% lower bound for LAD vs the direct clean gradient**. The more demanding original ridge null blocks seven of eight family passes, often because mean differences are approximately zero; the other-LAD null blocks B501 SAE131 novel30 and B701 SAE131 novel34. The prespecified sign-nondegradation criterion also fails on B501 SAE131 novel30 (**161/173** correct LAD signs against **165/173** direct) and B701 SAE131 novel30 (**163/172** versus **165/172** direct).

Only B501 SAE167 in the new 34-hole cohort passes all conditions. Its 30-hole ridge lower CI crosses zero, causing the **original two-family pair verdict to fail**. The preregistration, fit coefficients and source-lock masks remain unchanged.

## What the pooled analysis says and does not establish

An independently written archived-number replay reproduced **all three intervals** in each of the eight new families, **24,000 seeded paired bootstrap estimates**, and the identical original 0/4 and 1/8 gates. On **1,298 eligible pair/family/goal observations** (out of 4,096 selected blank goals in 512 puzzle/family/pair units), the descriptive pooled target native-refit erasure MAE is:

| Frozen predictor | Pooled target native-erasure MAE | Sign-correct goals / 1,298 |
|---|---:|---:|
| Direct clean task gradient | 0.273883 | 1,225 |
| Original source 5%-ridge residual | 0.223902 | 1,216 |
| **New source weighted-median LAD residual** | **0.221802** | **1,228** |
| Equal-parameter other-active LAD residual | 0.264579 | 1,220 |

The source LAD reduces pooled MAE by **19.01% relative to direct**. Its pooled MAE improvement over the original ridge is **0.00210**, or **0.94% of ridge MAE**, and its signed effect predictions differ little in accuracy. A small pooled difference across correlated teacher/SAE refits cannot substitute for preregistered *per-pair, both-family* paired-bootstrap superiority.

The study shows why choosing a calibration loss that matches the heldout evaluation loss is *plausible yet insufficient* for a reliable causal-refit superiority claim. In-sample LAD calibration wins all original calibration fits, while direct gradient is beaten convincingly and *ridge* often remains indistinguishable on a fresh puzzle family. This is the intended discriminating result, including the complete negative decisions.

Both SMC-21 and SMC-22 use **the true Sudoku answer** when constructing the task-gradient logit margin. These are answer-conditioned mechanistic erasure analyses, not inference-time Sudoku solvers. Four SAE comparisons share only two recurrent teacher checkpoints and are related by synthetic Sudoku generators, so per-goal observations are not independent model population replications. Full-rank inactive SAE encoder representations and the exact OLM-19 reconstruction-equivalent opposite-erasure construction remain limits on an intrinsic causal feature identity claim.

## Evidence, source locks and audit scope

- Original successful [three-stage SMC-22 hosted Actions run #43827985476](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43827985476), result-producing source SHA **\`306c18b6c77612580aba8660cc013138955dee48\`**.
- [Frozen new LAD original-only calibration + independent standard-library audit artifact #14207241984](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43827985476/artifacts/14207241984); ZIP SHA-256 \`576052ca617d90dd8de93b443f40c1e70898bba710464a5729dccbacd505869b\`; \`lad_calibration.json\` SHA \`60566a0148274a6de379a4c84e12989cfce3effc1ce2b0baaabc7e6b1bb6084e\`, \`lad-audit.json\` SHA \`9c03521ab8a38d6a8b454716e687ca3549f2b48ce8ce3e847b81215e62892ef2\`.
- [Full new 128-puzzle/four-refit native erasure and independent audit artifact #14207338085](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43827985476/artifacts/14207338085); ZIP SHA-256 \`190d773e1db0060179410fa7b6c035b5160f0a9f2d7ca01c3f79202f619b908d\`; original \`full.json\` SHA \`6abe509046ba937bea63bdf682ed891d3eaf5d2505819882a3826327c5e11af2\` (26,349,444 bytes), \`audit.json\` SHA \`f8a275948ab1a252cb6f07e404aee881daad303d2b17cfa3a0792f1f970ec24e\`.
- Original frozen SMC-19 trained SAE and source calibration file bytes from artifacts #14199643904 and #14204423839, all original SHA hashes and calibrated original seven + four Hungarian assignments reverified. All 64 original calibration weighted ratio/weight/puzzle/slot records for each of eight new LAD fits preserved in the calibration artifact.
- [Machine-readable permanent registered result](SMC22_SUMMARY.json) includes all original input/fit/test SHA-256s, every failed/passed condition, new puzzle roster manifest SHA, three bootstrap intervals and calibrated source coefficients.
- A separate independent **standard-library** full auditor replays original SAE checkpoint file SHA-256s, original calibration/ridge response receipts, all eight weighted-median fits, 128 fresh Sudoku complete solutions and exact unique visible-goal masks, source-native/refit/other-active decoder contributions with original SHA-selected alternative, 48D clean gradients and four predictor response/error/sign scalars, all 24,000 original 64-puzzle clustered bootstrap scores and all registered gate decisions. A *second separately coded numerical review* also reconstructed every archived paired interval. These checks **do not independently rerun PyTorch trained teacher forward passes or clean FP64 autograd**.

## Mathematics and next hypothesis

[MathCan OLM-22 PR #110](https://github.com/patrickdugan/MathCan/pull/110) derives the exact clipped weighted-median one-parameter MAE optimizer and supplies exact Fraction witnesses where LAD beats ridge in calibration but **loses on a heldout target**, despite unchanged source residual covariates. It also proves a conditional absolute-risk transfer bound through coupling distance in the source/target joint residual distribution, with sharp factor two under the [-2,2] coefficient constraint. These are written mathematical proofs and exact-rational tests, not Lean-kernel-certified theorems.

A next prospective experiment could test **whether the original ridge and LAD slopes differ meaningfully between original calibration puzzle subpopulations** using only the original calibration set and a truly reserved new heldout group. Another test could prioritize answer-free task-gradient targets to assess whether the physical feature-effect predictor improves a practical reasoning controller. Neither proposal changes the original SMC-22 0/4 outcome.
