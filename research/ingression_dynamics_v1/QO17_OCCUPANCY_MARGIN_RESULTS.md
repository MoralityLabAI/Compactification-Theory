# QO-17: Source-frozen occupancy and margin/gate training results

**Post-outcome research note, November 4, 2026.** The independent [frozen test contract](qo17_frozen_contract.py) was committed first as \`be91b083df589477d3b6f9878488071a9ebecf46\`, before fitting. The [full canonical result receipt](../../results/qo17_occupancy/receipt.json) stores all model weights, train/validation histories, 576 policy checkpoints, 192 trajectory hashes, exact progress labels for training-family policy-visited ledgers, and held-out policy-visited calibration samples. A source-bound rerun regenerates every byte.

## Train/validation source occupancy

QO-16 already established a dense exact QO-12 expected-gap teacher: 283/288 informative held-out action labels. QO-17 probes whether observed *policy-state occupancy* and a margin/gate objective improve a tiny learned controller. Its training distribution combines the original QO-16 fixed-schedule set (with existing cell swaps) and source-only QO-08 bottleneck or exact QO-12 teacher policy-visited ledgers.

| Provenance | Training rows | Validation rows |
|---|---:|---:|
| Fixed-schedule QO-16 source including swaps | 576 | 144 |
| QO-17 eligible policy-visited evidence | 194 | 79 |
| Combined supervision | **770** | **223** |

Two training-only occupancy seeds per source family, two source policies and eight prespecified query checkpoints produced 194 labeled eligible train states. All 194 include the original rational QO-12 action values, public counts/hits/nested bounds, a source-policy identifier, rate-family index, source stream seed and query-history SHA-256. Training occupancy generation used **1,331 exact teacher action calls**, **12,563 exact policy-planning gap evaluations**, and **2,040 additional gap evaluations** to label recorded train snapshots. Teacher provenance and validation-source costs are separately recorded.

The four newly fitted architectures retain QO-14's fixed parameter count and input encoding. Both 26-param linear and 138-param fully trainable recurrent students were fitted on the same mixed data, once with unchanged QO-14 weighted MSE, once with the precommitted margin softplus and parameter-free gate BCE. Their validation-selected epochs were 34, 52, 17, and 57, respectively (linear MSE, RNN MSE, linear margin/gate, RNN margin/gate). Each took 50,050 source-fixed Adam updates. Two untouched QO-16 MSE weights provide historical fixed-only comparisons.

## Frozen supervised action decisions

The eight independently frozen QO-17 rate pairs supplied **260 unresolved supervised states**, of which **249/260 (95.8%)** had unequal exact QO-12 first-action progress. Their source families and random seeds were inaccessible to fitting and epoch selection.

| Model | Incorrect action on 249 informative states | Conditional exact raw-progress regret | Normalized Q-value MAE |
|---|---:|---:|---:|
| Frozen QO-16 linear MSE | 61 | 0.00031421 | 0.10815 |
| Frozen QO-16 recurrent MSE | 67 | 0.00035521 | 0.10458 |
| Occupancy linear MSE | 60 | 0.00035203 | 0.10790 |
| Occupancy recurrent MSE | 61 | 0.00036164 | 0.09844 |
| Occupancy linear margin/gate | **57** | 0.00030548 | 0.11740 |
| Occupancy recurrent margin/gate | 58 | **0.00028967** | **0.09535** |

The margin/gate recurrent model reduced conditional exact progress regret by **18.45%** relative to the unchanged QO-16 recurrent MSE baseline and about **19.9%** relative to the occupancy recurrent MSE ablation. Its action mistakes fell from 67 to 58 compared with the unchanged baseline. The reduction in normalized Q-MAE is smaller than the relative gain in action regret, consistent with the MOR-13 distinction between individual action-value approximation and correct ordering.

**Important negative finding:** all six students still had 22 false engagement predictions and zero missed engagements on this fixed-schedule frozen supervised distribution, despite the gate BCE in two new models. The trained gate therefore remains poorly calibrated near the fixed threshold.

## Independent 160-query policy evaluation

Eight new Bernoulli rate pairs × three independent policy seeds × **eight arms** × exactly 160 queries yielded **192 matched trajectories, 30,720 observed Bernoulli draws and 576 checkpoint records**. All cell-specific stream prefixes were replayed independently. Source-fixed checkpoints were 32, 80, 160 queries.

| Arm | Correct certificates at query 160 | Mean QO-07 G gap | Paired mean gap gain vs bottleneck |
|---|---:|---:|---:|
| QO-08 bottleneck | **10/24** | 0.23198684 | baseline |
| Exact QO-12 H2 gap oracle | **10/24** | **0.22990101** | **+0.00208583** |
| Frozen QO-16 linear MSE | **10/24** | 0.23074949 | +0.00123735 |
| Frozen QO-16 recurrent MSE | **10/24** | 0.23134913 | +0.00063771 |
| Occupancy linear MSE | **10/24** | 0.23245495 | −0.00046811 |
| Occupancy recurrent MSE | 9/24 | 0.24168979 | −0.00970295 |
| Occupancy linear margin/gate | **10/24** | 0.23219813 | −0.00021129 |
| Occupancy recurrent margin/gate | 9/24 | 0.23215078 | **−0.00016394** |

**Prespecified primary policy endpoint:** the recurrent occupancy margin/gate model's 160-query gap improvement was **−0.00016394**, with seven matched wins, 11 ties and six losses across 24 case-seed pairs. The source-fixed 2,000-resample rate-cluster bootstrap percentile interval was **[−0.00434062,+0.00256773]**. The outcome does not establish a policy-level improvement.

The exact QO-12 teacher achieved **+0.00208583** mean gap gain (nine wins, 13 ties, two losses), with descriptive eight-rate cluster percentile interval **[+0.00011956,+0.00447720]**. The 160-query certificate count matched bottleneck. Low-gain short-horizon progress is a proxy for the 160-step objective, not itself a certificate.

Post hoc error analysis reveals occupancy-only RNN MSE was particularly poor on frozen family \`qo17_independent_7\` (paired mean gap deterioration \`−0.062746\`), while the margin/gate variant recovered a positive \`+0.001404\` on that family. The margin/gate variant's largest remaining negative family was \`qo17_independent_1\` (\`−0.012851\`). Those case-specific observations are retrospective diagnostics and cannot be used to tune the frozen QO-17 test.

## Fresh policy-visited engagement calibration

After full-budget rollouts, independent replay of each student policy's query receipts reconstructed public evidence at six precommitted checkpoints. The exact QO-12 teacher labeled only ready, unresolved states, **after** action choices had been finalized.

| Student | Eligible visited states | Teacher-negative false engagements | Correct teacher-negative fallbacks | Missed true engagements | Engagement precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Frozen QO-16 linear | 120 | 14 | 0 | 0 | 0.8833 | 1.0000 |
| Frozen QO-16 RNN | 118 | 15 | 0 | 0 | 0.8729 | 1.0000 |
| Occupancy linear MSE | 120 | 13 | 0 | 0 | 0.8917 | 1.0000 |
| Occupancy RNN MSE | 122 | 13 | 0 | 0 | 0.8934 | 1.0000 |
| Occupancy linear margin/gate | 119 | 16 | 0 | 0 | 0.8655 | 1.0000 |
| Occupancy RNN margin/gate | 120 | **10** | **1** | 4 | **0.9130** | 0.9633 |

The primary recurrent model's visited-state confusion counts were \`TP=105, FP=10, TN=1, FN=4\`. Thus its false-engagement rate on the 11 teacher-negative visited states remained **90.9%**. Every arm visits a potentially different state distribution; cross-arm differences are descriptive, not interchangeable fixed-sample calibration estimates.

The QO-17 gate's failure modes suggest two separate interventions: direct estimation of the **teacher's threshold distance** with a selective fallback rule, and additional training on policy-visited near-gate states with sufficient independent support. The MOR-14 Lean theorem supplies the conditional pointwise guarantee: if each predicted Q error is ≤ ε and the teacher score is more than ε from the gate, the learned engage/fallback decision is preserved.

## Certificate and confidence accounting

There were **zero incorrect certificates** across all 576 checkpoint records. The single shared case-seed \`qo17_independent_1\`, seed \`3719\`, had confidence-coverage misses at checkpoints 32, 80 and 160 in all eight arms (24 affected records). All misses remain in the source-bound receipt, counted as **one shared matched case-seed event family**, rather than 24 independent events. This finite observation is compatible with QO-07's nonzero allowed confidence failure probability, under its original IID/addressability conditions.

QO-17 has established a positive supervised action-ranking result alongside a negative full-policy outcome. The next experimental question is whether a pointwise gate-margin audit and a policy-occupancy-aware threshold model can reduce false engagement without destroying the recurrent model's supervised action-regret improvement. Any follow-up needs new, independently precommitted test rates and seeds. These results are restricted to the two-cell digital Bernoulli model; they do not establish learned-model calibration guarantees or transfer to Qwen, BitAgent, neural hidden states, or biological systems.
