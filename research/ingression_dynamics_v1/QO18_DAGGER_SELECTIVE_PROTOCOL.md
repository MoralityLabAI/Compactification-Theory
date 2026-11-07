# QO-18: Student-visited DAgger aggregation and selective gate calibration

**Preregistered on November 5, 2026.** The independent test contract was committed **before** any QO-18 training or test code at \`8d420681163f1bb34b66fa946baf59e9cdc34179\`. The held-out QO-18 rates and seeds are disjoint from QO-14 through QO-17 testing, original training, and all new occupancy/calibration sources. The registered test generator is imported by \`qo18_dagger_pilot.full_report\` only after all training and validation threshold selection have finished.

## Scientific question

QO-17 learned to reduce conditional exact QO-12 expected-gap progress action regret by 18.45% against its unchanged recurrent baseline, yet had no positive improvement in 160-query certification-gap outcome. Its student-visited exact teacher-negative states had a 90.9% false-engagement rate. The QO-17 occupancy training set came from teacher and bottleneck policies. QO-18 samples the states actually visited by its evolving student policy and attempts to calibrate the engagement threshold without editing the QO-07 certificate.

Let \(\widetilde Q_a=T(Q_{G,a})\), where \(Q_G\) is the exact rational QO-12 two-step robust-minus-optimistic certificate gap improvement and \(T(q)=q/(q+1/64)\). The architecture is the unchanged 138-parameter, 8-hidden-unit two-token tanh RNN trained with the fixed QO-17 full-BPTT weighted MSE, margin-softplus and gate BCE. All teacher targets are generated from public QO-07 confidence evidence, with true rates used solely for synthetic Bernoulli draws.

The source-frozen teacher engagement threshold is \(\tau_T=0.05\). At deployment the model engages its predicted greedy action if \(\max_a\widehat Q_a\ge\tau_S\); otherwise it queries QO-08's existing bottleneck cell. Both exact and student controllers preserve the original readiness requirement (eight observations per cell) and confidence-certificate fallback.

## Causally separated two-round training

The initial data combines 576 original fixed-schedule QO-16 training rows and 194 original QO-17 training-family occupancy rows, with corresponding independent validation data (144 + 79). The source-locked original QO-17 recurrent margin/gate checkpoint, from its merged receipt, is the round-0 student.

Two new DAgger-like rounds each run the *current* student policy with its fixed 0.05 gate on QO-14 original TRAIN rate families and fresh independent source seeds; collect public evidence after query counts 24/40/56/72/96/128/160 while undecided and ready. Label the collected states using the exact QO-12 teacher after the policy has visited them. Refit a fresh 138-parameter RNN from the original QO-14 initialization and unchanged QO-17 loss on all original plus cumulative student occupancy states. Epoch selection uses the corresponding original VALIDATION rate families and separate source-only round seeds. Freeze the fitted round-1 checkpoint for the second sampling round, then train and freeze the round-2 checkpoint.

The hard-negative oversampling rule is source-fixed: from collected TRAIN-only student states, take the first at most 64 unique source histories with exact teacher \(\max T(Q_G)<0.05\), and append eight additional copies each to the current cumulative training set. Validation examples retain natural rates and counts. Unique identities, state histories, bounds, Q-values and exact teacher costs remain in the receipt.

Round-1 TRAIN seeds 5701/5711; round-2 5903/5917. Separate VALIDATION collection seeds 6101/6113 and 6301/6311. All state observations come from policy-independent per-cell Bernoulli stream prefixes, with source trace hashes checked against independent replay.

## Independent validation-only gate selection

After the two-round model has been fitted, collect fresh validation-policy-visited evidence using the original VALIDATION rate families and independent seeds 6503/6521, with source policies fixed at the round-2 student (0.05 gate) and QO-08 bottleneck. Collect ready and unresolved states at eight checkpoints 24/32/40/56/72/96/128/160. Exact QO-12 values are produced retrospectively and stored with the public state. Dependence within each source trajectory is explicitly preserved in cluster receipts.

Evaluate the seven source-frozen model thresholds 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, and 0.50. A candidate is feasible only with positive reference-negative and reference-positive denominators and **empirical false-engagement rate ≤10% and recall ≥80%**. Among feasible candidates, choose maximum recall, breaking ties toward the lower threshold. If no candidate meets the constraints, record \`feasible=false, threshold=null\`, and define the final calibrated policy to run QO-08 bottleneck on every query. Such safe fallback can produce a zero-valued paired policy improvement and is a valid negative outcome. A threshold that engages nobody is forbidden from being described as a positive result.

The explicit threshold choice uses *validation* evidence only. An independently frozen test set is never used to select, adjust or inspect thresholds.

## Independent frozen evaluation

Eight rate pairs with denominator 120, three supervised test seeds, and three separate rollout seeds; supervised checkpoints 24/40/64/96, all on source-frozen public schedules. Policy checkpoints 32/80/160. Six policy arms:

- QO-08 bottleneck
- Exact QO-12 H2 expected-gap teacher
- Frozen QO-17 recurrent margin/gate student with 0.05 threshold
- QO-18 round-1 recurrent student with unchanged 0.05 gate
- QO-18 round-2 recurrent student with unchanged 0.05 gate
- QO-18 round-2 recurrent student with validation-selected threshold, or bottleneck fallback if infeasible

This requires **144 full trajectories, 432 checkpoint records, 23,040 Bernoulli queries**. The primary endpoint is the paired mean QO-07 robust-minus-optimistic certification-gap reduction at 160 queries between calibrated round-2 student and bottleneck over 24 fixed case-seed pairs. Report exact rational deltas, wins/ties/losses, and the QO-16/QO-17 deterministic eight-rate-cluster bootstrap with 2,000 resamples and fixed seed 20261104. This interval is descriptive under eight nonrandom clusters and does not establish population-wide efficacy.

**Primary gate constraint:** independently frozen ready/undecided policy-visited evidence must show observed false engagement ≤10% and true-positive recall ≥80%, where both denominators are nonzero. Capture TP/FP/TN/FN and clustered provenance. Also evaluate a policy-independent frozen supervised distribution so each student's conditional exact teacher regret, margin error, Q MAE and engagement calibration can be compared on the same states, independent of its self-induced occupancy.

Additional metrics include certificates at all three budgets, false certificate events, confidence-coverage misses, true frozen TV error, training and exact planning costs, student inference MACs, teacher-label counts, source hashes and trajectory hashes. All experiments run the unchanged QO-07 nested confidence sequence as the sole certificate authority.

## Safety, statistical interpretation and acceptance

Report empirical confidence misses without deleting any seeds. Adjacent policy-visited states are dependent; raw state-level binomial error bounds lack the necessary IID premise. MOR-14's Lean theorem gives conditional gate robustness only when there is a valid *statewise* action-value error bound \(\varepsilon\). Training MAE is not a uniform guarantee.

Complete source-bound deterministic training and evaluation receipt, frozen-test split checks, replay tests and the existing original QO-14/16/17 BPTT gradient and QO-12 oracle-parity tests. Preserve all negative results; merge only after the final PR-head exact retraining \`--check\` and full morphospace CI pass. This is a tiny source-only digital benchmark, with no claims of Qwen, BitAgent, hidden-state or biological transfer.
