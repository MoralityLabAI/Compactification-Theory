# QO-16: Dense expected-gap Bellman distillation, frozen experiment results

**Post-outcome note, November 4, 2026.** The [independent QO-16 test contract](gap_student_frozen_test.py) was committed first (\`369e1ca\`), before fitting and outcome inspection. The [machine-readable source-bound receipt](../../results/gap_student/receipt.json) contains complete trained weights, validation histories, all 288 policy checkpoints, 96 full trajectory hashes, frozen supervised metrics and bootstrap comparisons. Source-independent reference checks verify exact QO-12 Bellman action values and original QO-07 certificate semantics.

## The new teacher gives dense, informative action labels

All labels are exact Fraction evaluations of QO-12's *two-step expected reduction* in the robust-minus-optimistic QO-07 certificate gap \`G=U-L\`, with Beta(1,1) predictive recourse. The training label is \`T(Q_G)=Q_G/(Q_G+1/64)\`, a fixed strictly increasing transform used for the unchanged QO-14 sigmoid-head models.

| Source family | Unaugmented examples | Nonzero Q_G | Unequal action values |
|---|---:|---:|---:|
| QO-14's original TRAIN rates/seeds | 288 | 287 | 274 (95.1%) |
| QO-14's original VALIDATION rates/seeds | 72 | 71 | 70 (97.2%) |
| Independently frozen QO-16 TEST | 288 | 287 | **283 (98.3%)** |

Training/validation additionally include one counterfactual cell-label swap per example, for 576 training and 144 validation samples; no held-out rates, seeds or labels entered fitting. Source training used 3,159 leaf-gap evaluations, validation 767, and frozen test 3,071. All parent confidence streams completed without invalid intersections. The QO-12 parity tests passed.

This fixes the *data-density* problem seen in QO-14 (8/505 informative held-out states) and QO-15 (0/720). Both QO-16 students received a genuinely action-rich objective.

## Held-out supervised performance

Both arms retain their original architecture and deterministic QO-14 Adam/BPTT implementation, 65-epoch cap and validation-only checkpoint choice.

| Held-out test metric | 26-param linear | 138-param tanh RNN |
|---|---:|---:|
| Validation-selected epoch | 21 | 63 |
| Normalized action-value MAE | 0.106184 | **0.100068** |
| Normalized predicted-margin MAE | **0.052479** | 0.054279 |
| Informative teacher-action disagreements | **66/283** | 75/283 |
| Conditional exact raw-progress action regret | **0.00032451** | 0.00037098 |
| Unconditional exact raw-progress action regret | **0.00031888** | 0.00036454 |
| False engagement predictions | 11/288 | 11/288 |
| Multiply–accumulates per inference | 24 | 240 |

The RNN fit the two-dimensional exact Q-value surface slightly better, yet selected the lower-value probe more often and incurred greater raw teacher-action regret. MOR-13's statewise margin-regret theorem remains sound; smaller mean Q error is not by itself enough to reduce action regret on this distribution.

## Independent 160-query matched policy evaluation

Eight disjoint rational rate pairs × three new policy seeds × four arms produced **96 complete trajectories, 15,360 Bernoulli draws and 288 checkpoint records**. Every queried per-cell stream prefix was independently replayed.

| Policy | Correct certificates at 80 | Correct certificates at 160 | Mean G at 160 | Paired mean G improvement vs bottleneck |
|---|---:|---:|---:|---:|
| QO-08 bottleneck | 4/24 | **12/24** | 0.22722251 | baseline |
| Exact QO-12 horizon-two teacher | 4/24 | 11/24 | **0.22608481** | **+0.00113770** |
| 26-param linear student | 5/24 | **12/24** | 0.22762698 | −0.00040447 |
| 138-param recurrent student | 5/24 | 10/24 | 0.22814249 | −0.00091998 |

For the prespecified 160-query primary endpoint, teacher-versus-baseline exact gap improvements were positive in 6 case/seed pairs, tied in 16 and negative in 2. The 2,000-resample, eight-rate-cluster bootstrap percentile interval was **[+0.00016814, +0.00220654]**. The linear student's interval was **[−0.00147713, +0.00063352]** (6 wins, 11 ties, 7 losses), and the RNN student's was **[−0.00318872, +0.00085605]** (5 wins, 15 ties, 4 losses). These intervals are descriptive with eight non-random test clusters and do not establish population-wide efficacy.

The exact teacher lowered the average robust confidence gap but earned one fewer full certificate than bottleneck at 160 queries. Minimizing the expected short-horizon gap does not imply maximizing 160-step certificate attainment.

### Compute and gating behavior

| Model/policy | Eligible gap decisions | Exact teacher evaluations | Student MACs | Explicit model-chosen probes |
|---|---:|---:|---:|---:|
| Exact teacher | 2,643 | 22,714 leaf-gap evaluations | 0 | 2,000 |
| Linear student | 2,617 | 0 | 62,808 | 2,617 |
| Recurrent student | 2,661 | 0 | 638,640 | 2,661 |

At this held-out policy distribution, both trained models exceeded the fixed normalized progress gate on every eligible inference. The exact teacher used bottleneck fallback 643 times on eligible low-gain states. Thus the newly learned policies remain poorly calibrated at their own *visited states*, a distribution different from the source schedules used for training.

### Confidence validity and audit exceptions

Across all 288 checkpoint records, **zero incorrect certificates** were observed. The single shared confidence-coverage miss occurred in \`qo16_frozen_pair_1\`, seed \`2521\`, and persisted at checkpoints 32, 80 and 160 under all four policies (12 affected records from the same case/seed family). The underlying rates are source-frozen and QO-07's original anytime guarantee permits low-probability misses; they have been retained, not removed or silently averaged away.

Three of the 24 exact-teacher rollout transcripts were byte-for-byte identical to bottleneck, while neither trained student had an identical full transcript. Costs count mathematical gap evaluations and deterministic MACs, not wall-clock energy or amortized training.

## Next mathematical target

QO-16 gives a clean result: a dense exact teaching signal exists, and the small RNN is able to regress it. The unresolved engineering problem is **calibrated action-margin learning under the policy's own evidence distribution**. Its lower Q MAE yielded worse decisions than the linear learner, and the recurrent rollout spent ten times as many MACs while earning two fewer certificates than baseline.

Register QO-17 as a distinct experiment before selecting new test results: compare a directly trained two-action margin head or pairwise ranking loss, a zero-progress engagement classifier, and a policy-occupancy-aware training mixture. Freeze an untouched test distribution and retain the full 160-query paired certificate endpoint, together with conditional action regret. Any occupancy samples generated using a teacher policy must come solely from allowed training seeds and rates.

QO-16 is evidence about known-addressable digital Bernoulli cells. It does not certify learned uniform approximation error, long-horizon optimality, hidden-state identification, neural-LLM transfer or biological efficacy.
