# QO-17: Occupancy-aware margin/gate certificate distillation

**Preregistered November 4, 2026.** The independent test contract was committed **before QO-17 fitting** as \`be91b083df589477d3b6f9878488071a9ebecf46\`. Its data families, seeds, policies, losses, budgets, action gate and primary outcome are fixed for this experiment. QO-16 source, exact QO-12 and original QO-07 confidence-sequence algorithms remain unchanged.

## Scope and causal split

The source task is two known, resettable Bernoulli cells, a single action, and one macro transient class. QO-07's nested, rational anytime confidence bounds are the only certificate authority. QO-12's exact two-step Beta(1,1) predictive expected gap-progress action values serve as teaching labels; predicted values are planning heuristics. Underlying source Bernoulli rates drive simulated draws only, never student inputs.

The eight independently frozen QO-17 rate pairs have denominator 80 and are disjoint from the QO-14 training/validation/test and QO-15/QO-16 test families. Test supervised seeds \`3503, 3517, 3529\` and policy seeds \`3701, 3719, 3733\` are disjoint from all previous family seeds and the newly fixed train/validation occupancy seeds. The frozen module is **first imported by full_report only after all four train_model fits have returned**; experiment unit tests may separately check the public constants. The experiment does not tune its architecture or policy using test metrics.

## Precommitted on-policy source mixture

Use the exact QO-16 train and validation rate families with outcome-independent fixed-schedule evidence at four checkpoints, including their existing cell-label swaps. Add public evidence from two source policies: QO-08 bottleneck and exact QO-12 horizon-two expected-gap selection with fixed fallback. Source-specific per-cell Bernoulli streams use training-only occupancy seeds \`3101,3109\` and validation-only occupancy seeds \`3301,3313\`. Source trajectories run 128 queries; capture eligible undecided ledgers at \`16,24,32,40,56,72,96,128\` observed queries.

The training dataset concatenates one original QO-16 source set (including cell-swap copies) and one captured eligible-policy set. For every visited sample, preserve rate-family index, source policy, seed, query time, public counts/hits/nested rational bounds, source-history hash, exact QO-12 per-action raw values, normalized targets and label-generation work. Preserve complete source-policy trajectory hashes and teacher oracle counts. The validation split uses its own existing rates and occupancy seeds. The original QO-16 fixed-only fitted weights are loaded unchanged as historical ablation baselines.

## Registered architecture/loss comparison

Use only the original 26-parameter linear sigmoid model and the original eight-hidden-unit 138-parameter tanh RNN. Their input tokens are exactly the six public QO-14 features per cell, and the RNN uses full two-step BPTT. Both architectures fit the same mixed training and validation datasets, with two loss functions:

The MSE ablation uses QO-14's source-fixed weighted two-head squared error and Adam \`fit\` verbatim. The margin-and-gate ablation retains that MSE and adds

\[
L_{\rm rank}=2|\Delta_T|\operatorname{softplus}
   \left(-\frac{\operatorname{sign}(\Delta_T)(p_0-p_1)}{0.10}\right),\quad
\Delta_T=T(Q_{G,0})-T(Q_{G,1}),
\]

\[
L_{\rm gate}=0.015\left[
 \operatorname{softplus}(z)-ez\right],
\qquad z=\frac{\max(p_0,p_1)-0.05}{0.025},
\qquad e=\mathbf1\{\max T(Q_G)\ge0.05\}.
\]

\[
L=L_{\rm QO14\,weighted\,MSE}+L_{\rm rank}+L_{\rm gate}.
\]

All coefficients are fixed in the initial frozen contract. A source-fixed subgradient assigns a tied predicted maximum to head 0; true Q-margin ties incur zero ranking penalty. Extra gradients propagate through the existing input and recurrent matrices, keeping parameter/MAC costs unchanged. Independent central finite differences must check all relevant head, input, recurrent and bias parameter classes. Adam settings and validation-only checkpoint selection are inherited unchanged from QO-14: up to 65 epochs, rate 0.012, clip norm 3, seed 20261104.

The experiment's model-policy arms are the four occupancy students \`linear_occupancy_mse\`, \`recurrent_occupancy_mse\`, \`linear_occupancy_margin_gate\` and \`recurrent_occupancy_margin_gate\`. Historical fixed-only QO-16 linear and RNN weights provide occupancy-free MSE baselines. QO-08 bottleneck and the exact QO-12 progress teacher complete the **eight** frozen policy arms. Every arm uses the original QO-16 readiness threshold (eight observations in each cell), confidence-certificate fallback and normalized progress gate 0.05.

## Held-out evaluation and audit

The independent supervised test has 8 source-frozen rate pairs × 3 supervised seeds × 3 public fixed schedules × 4 checkpoints \`24,40,64,96\`, with the original QO-07 evidence update. Primary supervised endpoint is **conditional exact raw QO-12 progress regret** among held-out unequal-value states, with denominator and action disagreements. Keep the normalized margin MAE and Q MAE, fixed-schedule gate confusion, and value-margin buckets.

For visited-state calibration, replay each frozen policy's entire observed Bernoulli transcript after the policy run. At \`24,32,48,80,112,160\` queries, collect eligible public evidence states and compute exact QO-12 progress labels independently of the controller. Report per-arm calibration counts TP/FP/TN/FN, precision, recall, false-engagement rate, missed-engagement rate and conditional action regret. This post-hoc teacher work must be charged separately from online inference and can never change selected actions.

The registered policy evaluation has 8 rate pairs × 3 policy seeds × **8 policy arms** × 160 queries = 192 complete trajectories, 576 checkpoints and 30,720 Bernoulli draws. Checkpoints are 32/80/160. Primary policy endpoint is the **paired mean reduction in robust-minus-optimistic QO-07 certification gap at query 160**, comparing the recurrent occupancy margin/gate student with bottleneck. All other arm comparisons are declared secondary. Use exact rational case-seed differences and a source-fixed 2,000-resample eight-rate-cluster percentile bootstrap; report wins/ties/losses and the CI as descriptive.

Also report complete certificate counts, incorrect certificates, confidence-coverage misses, true frozen TV error, online exact teacher calls and gap-evaluation counts, inference MACs, retrospective visited-state oracle costs, training cost and SHA-256 source/traj/test fingerprints.

A zero or negative primary improvement is a valid completed result. No QO-07 coverage claim is inferred from empirical audit counts, and the model has no certified uniform error bound or tested Qwen, BitAgent, hidden-state or biological transfer.

## Merge gate

The first CI run emits a complete deterministic result receipt, to be reviewed and committed unchanged. Rerun the source-bound \`--check\` on the final PR head, plus QO-17 unit tests, finite-difference derivatives, original QO-14 gradient regression tests and the complete morphospace audit. Merge only after final-head success, preserving any negative result in a post-outcome research note.
