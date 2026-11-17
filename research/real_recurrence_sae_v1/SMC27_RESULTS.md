# SMC-27: Old-calibration-selected abstention for native SAE causal decisions

**18 November 2026. Registered scientific result: FAIL, 0/4 original frozen teacher seeds passed both new Sudoku families.** The source-locked [four-stage GitHub Actions run #44136434145](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44136434145) passed old-calibration-only threshold selection, target-free new puzzle prediction, original native FP64 scoring and independent Python standard-library audit. The scientific protocol [SMC27_PROTOCOL.md](SMC27_PROTOCOL.md) was committed FIRST at 6f769add, before any SMC-27 calibration outcome or new Sudoku source was generated. All original SMC-26 through SMC-23 verdicts and the sealed ARG-03-N 168-case split remain unchanged.

## Experimental question

The preceding [SMC-26](SMC26_RESULTS.md) prospective four-teacher sign-utility experiment found that a temperature-calibrated weighted median of nine candidate Sudoku causal effects gave positive point-estimate utility gains over a label-uniform median in all eight teacher/family cells, while its original 0.15-logit intervention decision threshold acted on too many goals and failed every corrected simultaneous utility comparison with always abstaining.

SMC-27 asks whether a threshold fitted from **old calibration-only, source-locked native causal intervention outcomes** improves new-puzzle offline action utility over (a) abstaining, (b) retaining the original fixed 0.15-threshold calibrated median, and (c) a uniform-nine median with the SAME fitted threshold. These three comparisons were predeclared together. A positive directional point estimate alone, or a less conservative confidence bound, does not replace them.

Every eligible goal freezes the **native source-teacher SAE71 highest positive Top4 coordinate**, its minimal-norm selective hidden-state erasure, and the original best of 33 outcome-blind, equal-L2-dose, inactive-preactivation-matched tangent shams. The predictor evaluates all nine hypothetical task-label digit effects through one clean and two +/-0.08 physical native three-block FP64 passes, without knowing the true heldout target digit. The correct digit enters only after the frozen predictor, feature plan, abstention threshold and binary FP32 source state have been uploaded as source-locked artifacts.

The result measures an OFFLINE decision utility, with original SMC-26 true-effect 0.15 neutral band and 0.10 action cost. It does not improve the teacher weights, modify a Sudoku solver, train new SAEs or establish global semantic feature identity.

## Frozen teachers and original SMC-26 provenance

SMC-27 uses exactly the original successfully completed SMC-26 model/SAE checkpoints of B501/B701/B811/B919, one Top4 SAE71 per independently initialized model, all with the same 44,001-parameter 48D recurrent Sudoku architecture and one shared original source train/validation puzzle roster. The original B811/B919 were trained in SMC-26 on fixed 600-step teacher and 300-step SAE schedules; none were retrained here. Four teacher initializations and two new Sudoku families are the source of model and problem variation.

Original source SHA256 checks in all model-bearing CI stages:
- Original SMC-26 science run 44130482978 training bundle: **408348eeecc300e7518ee7e8ea14d71caac29821aff39aaaa5ee819c794231f1**.
- Original SMC-26 calibration JSON: **145312c04648847c409b506aecdf1a3464af855d4ec8aa638eb0d604185512c9**.
- SMC-11C original source SAE receipt: 7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087.
- SMC-13A B501/B701 independent teacher+SAE receipt: 1fc69e15c24087f631806a89d55beaf519c2f6ab58ac7b8193cf8fdc36d240e3.
- SMC-13C conditioned correspondence receipt: 3b4637699d5af236af2d3b8aa5d04e024025b16e0d59395535892c104d472d07.

The original SMC-26 report had one bundle-SHA entry copied from a later QA retraining. This PR independently verified the original bundle from the original artifact and corrected that entry in [SMC26_RESULTS.md](SMC26_RESULTS.md). B811/B919 original teacher and SAE parameter fingerprints remain those preserved in the original SMC-26 raw calibration receipt, with no model substitution.

Original calibration accuracies were B501 235/512 (45.9%), B701 259/512 (50.6%), B811 235/512 (45.9%) and B919 213/512 (41.6%), all within SMC-27's registered 40–60% competence gate; all original class temperatures were T=2.

## Source-only threshold calibration

The **first** job used just the original SMC-13 64 feature_calibration puzzles, with eight input-mask-only blank-cell goals each. True digits and finite native intervention outcomes were used ONLY in this old calibration. For each teacher, nine prospectively frozen candidate thresholds were evaluated: 0.15, 0.30, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0. A threshold is eligible if 5–90% of physically admissible old goals produce a nonzero intervention action. Select the threshold with maximal old 512-slot average SMC-26 reward, breaking ties toward more abstention.

| Original teacher | Old eligible goals / 512 | Frozen threshold | Old selected action rate | Old mean utility per original slot | Gate |
|---|---:|---:|---:|---:|---|
| B501 | 202 | 0.30 | 77.72% | +0.09238 | Admitted |
| B701 | 320 | 0.50 | 68.44% | +0.06074 | Admitted |
| B811 | 313 | 0.30 | 77.64% | +0.09902 | Admitted |
| B919 | 275 | 0.30 | 73.82% | +0.16152 | Admitted |

All four selected policies passed the pre-outcome calibration admission. The complete nine-threshold utility grid, all source calibration goal records, actual old goal D values and selected-threshold arithmetic were saved in a separate SHA256-pinned artifact **before the new 256-puzzle source generation job**. The independent audit subsequently regenerated the original old feature_calibration puzzle positions and recomputed these four threshold optimizations from the original old cases.

## Fresh Sudoku evaluation and physical admission

Fresh novel30 source solution seeds: 202637030000 through 202637030127. Fresh novel34 seeds: 202637040000 through 202637040127, unique-dig seed always grid seed +1,000,000. Every case was verified uniquely solvable with strictly positive noncanonical rectangle count, and identical visible input and full target solutions were excluded from the 1,408 original/prior fresh source solutions.

| New source family | 128-puzzle full source roster SHA256 | Minimum noncanonical rectangles |
|---|---|---:|
| Novel30 | 35610bb984fc6b14b1cf6f58b74eff9f6672e9b5368d17f66a46af7342423a2b | 6 |
| Novel34 | a5529eef4d38048afb9f4211ebf793419107373d2453c5715b9fb557e044c044 | 6 |

With four teachers, eight visible-mask selected blank goals per puzzle and two cohorts, there were **8,192 originally selected teacher-goal slots**. The physical selected feature, erasure and equal-dose hard-sham pre-outcome gates admitted **4,260** (52.0%) interventions. Per-cell admission was 411–598 goals across 124–128 independent puzzle clusters, exceeding the original 200-goal/80-puzzle minimum everywhere.

The predictor archived its full original FP32 stage-one tensor per teacher and family in a separate binary file, with a source SHA and no correct-digit labels or native outcomes in the prediction JSON. Native scoring reran the original source first-stage state and found maximum absolute cross-runner deviations **1.1921×10⁻⁶**, well within the original 10⁻⁴ admission cutoff. Source feature recoding, clean/zero-self bitwise closure, matched physical norm, and three genuine FP64 recurrent blocks passed.

## Original eight-cell prospective result

The primary utility is the mean over all 128 independent puzzle IDs of the eight-original-slot within-puzzle utility. Inadmissible slots contribute zero under every policy. Nonneutral true effect correct intervention earns +1, incorrect earns -1 and abstention zero; true neutral abstention earns +1, action earns -1; each action pays 0.10. A goal's true effect sign is defined by D=m(sham)−m(kill) with the original fixed ±0.15-logit neutral band.

The registered primary comparisons are selected-threshold calibrated policy utility minus always abstain, minus original fixed-0.15 calibrated policy, and minus uniform-nine median with the SAME selected threshold. Every cell used **5,000 bootstrap resamples of all 128 puzzle clusters**, preserving the original eight slots, and a two-sided **99.79167% Bonferroni** interval over eight cells × three primary contrasts. A cell passes only when ALL THREE adjusted lower gains exceed +0.02, along with coverage and 5–90% heldout action-rate admission. The teacher must pass both novel families and the program at least three of four teachers.

| Teacher | Family | Admitted / puzzles | Selected action rate | Gain vs abstain [99.7917% CI] | Gain vs original 0.15 [99.7917% CI] | Gain vs uniform [99.7917% CI] |
|---|---|---|---:|---|---|---|
| B501 | 30-hole | 433 / 126 | 78.75% | +0.03994 [−0.01562,+0.09463] | +0.00400 [−0.02041,+0.03242] | +0.07461 [+0.02568,+0.12881] |
| B701 | 30-hole | 592 / 128 | 66.39% | +0.00947 [−0.05176,+0.06592] | +0.02178 [−0.01768,+0.06523] | +0.05977 [−0.01318,+0.13828] |
| B811 | 30-hole | 586 / 127 | 76.11% | +0.01699 [−0.05313,+0.08301] | +0.01445 [−0.01777,+0.05176] | +0.09961 [+0.00811,+0.18252] |
| B919 | 30-hole | 538 / 128 | 72.12% | +0.05098 [−0.00615,+0.10576] | +0.00645 [−0.02383,+0.03516] | +0.06934 [+0.00107,+0.14678] |
| B501 | 34-hole | 411 / 124 | 75.43% | −0.00488 [−0.05752,+0.04434] | −0.00039 [−0.02490,+0.02471] | +0.01064 [−0.06377,+0.08096] |
| B701 | 34-hole | 598 / 128 | 66.56% | +0.05391 [−0.00674,+0.11475] | +0.00410 [−0.03232,+0.04189] | +0.09268 [+0.01758,+0.16387] |
| B811 | 34-hole | 588 / 128 | 77.89% | +0.03242 [−0.03232,+0.09385] | +0.00918 [−0.01807,+0.03926] | +0.04785 [−0.03574,+0.13242] |
| B919 | 34-hole | 514 / 127 | 66.54% | +0.02227 [−0.03506,+0.08242] | +0.01768 [−0.01494,+0.05293] | +0.02959 [−0.03672,+0.09375] |

Every cell cleared the preregistered teacher capability, calibration admission, test action-rate and sample coverage requirements. The calibrated threshold reduced action rate from the older 0.15 rule, yet the **corrected simultaneous utility gate failed in all eight cells**. Against always abstaining, **0/8** adjusted lower utility bounds were positive; against original fixed 0.15, **0/8** were positive. Against uniform mixing at the matched selected threshold, **4/8** adjusted lower bounds were positive, and only **1/8** (B501 novel30) exceeded the required +0.02. Hence **no teacher passed both families and the registered SMC-27 verdict is FAIL, 0/4 teacher seeds**.

On the original eight outcome point estimates, selected-threshold policy utility is above the original fixed-threshold rule in 7/8 cells, with magnitude as small as 0.004 and at most 0.022 per original goal except for one negative 34-hole B501 cell. This is descriptive evidence that calibration-only abstention reduces costly actions without establishing reproducible utility dominance over both serious controls. It is too early to infer that four teachers learn a stable utility-optimal abstention mechanism.

## Integrity, limitations and source evidence

The first pre-data CI attempt #44136178234 failed at import discovery because the historical cohort-audit module was referenced under a wrong name. [SMC27_PREOUTCOME_AMENDMENT.md](SMC27_PREOUTCOME_AMENDMENT.md) records this source-only repair, which preceded the first SMC-27 calibration and all test data. In the successful four-stage [Actions #44136434145](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44136434145), eight independent CPU regression tests passed before old-only calibration; the target-free prediction and native scorer passed, followed by an independent Python standard-library audit.

The auditor independently verified the original old-calibration source puzzles and re-optimized all four selected thresholds, regenerated all 256 new Sudoku solutions and their 1,408-prior-source exclusion manifest, checked the original training bundle SHA, every frozen prediction action and physical dose, all native correct-digit identities, 4,260 admitted effects over 8,192 original slots, and **all 5,000 puzzle bootstrap draws in eight cells for all three primary comparisons**. It reported scientific FAIL with **audit PASS**. This is independent source and numerical receipt verification; the original teacher-forward function was not independently reimplemented.

All four teachers were independently initialized under a common small training source, so model-level generalization is limited; the two source families share the same Sudoku generator. The original native source uses a 48D recurrent model, four-positive-atom Top4 SAE feature selection and fixed horizon of three FP64 continuation blocks. Target feature erasure does not certify an intrinsic semantic coordinate or stable causal quotient. The offline reward prespecifies an action cost and neutral abstention benefit; the outcome is a model/harness result, not a claim about operational deployment or full-grid Sudoku solving.

Raw sources and SHA256:
- Original protocol first commit: 6f769add, [SMC27_PROTOCOL.md](SMC27_PROTOCOL.md).
- [Successful original science-producing CI #44136434145](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44136434145).
- [Old-only threshold calibration artifact #14352002934](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44136434145/artifacts/14352002934): ZIP SHA256 026578b2d4709de46b14d18fee13661a6ec7034fbd80eea085a6487fa3e23790; selected.json SHA256 7edea69ab35e83e0d29d9793c5dd4cf25756ae702a0c8133a9d9ffe736d42dac.
- [Frozen new 256-puzzle target-free predictions #14352098206](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44136434145/artifacts/14352098206): ZIP SHA256 748b56d9fb40dce10a2b0f76c38b2905fe85f1c922782ea2bfe9c74657d914d5, contains prediction.json plus source state1.pt with independently checked physical state source hash.
- [Native finite scoring #14351883821](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44136434145/artifacts/14351883821): ZIP SHA256 cb89ccbe9f0c95d94bfabfe7245022b5189a8e0c6c16738743e641326a54a408; scored.json SHA256 7b86a126c2f8b94517862d89a3634abe14f2162905604db4c30807d1d0b3ba7a.
- [Independent old/new source and utility audit #14352861792](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/44136434145/artifacts/14352861792): ZIP SHA256 352e1c965a7d9fcad30dbe98d7a7c013f9c3ae5c3e849ce9d20b0149a24bf131; audit.json SHA256 8668c6da741797caac044b16acfa35bea6289694a40ac91592c97aaaba67158f.

Future tests that change the abstention objective, calibration threshold grid, teacher architecture, or horizon must use another preregistered independent source cohort. Source-informed threshold fitting here already used all permitted original old calibration cases; retuning against the 256 new test outcomes would invalidate the prospective distinction.
