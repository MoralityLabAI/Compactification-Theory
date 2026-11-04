# QO-15: Decision-boundary-enriched certificate student

**Preregistered 2026-11-04, before QO-15 fitting and held-out evaluation.** The [frozen test generator](certificate_boundary_test_contract.py) was committed first, at \`c6ac3a19511f36cfc14bd70503e9132d33ab0766\`. This study retains the QO-07 evidence and certificate rules, QO-13 horizon-two exact Beta-predictive Bellman values, QO-14 eight-hidden-unit tanh RNN and Adam trainer, and equal-query paired Bernoulli streams.

## Hypothesis and primary outcome

QO-14's 689 usable training evidence states had only eight unequal teacher-action values. The 505 held-out states likewise had eight informative decisions. Its 138-parameter RNN reduced mean absolute Q-error but equaled the linear student's action disagreement and regret; the horizon-two exact teacher also tied bottleneck at 6/18 final certificates. QO-15 asks whether enriching **causally reachable decision-frontier evidence** improves choosing the higher exact Bellman value under newly frozen, independent rates and seeds.

For a two-action state s let \`delta = Q(s,0)-Q(s,1)\` and \`pred_delta = Qhat(s,0)-Qhat(s,1)\`. Exact-teacher regret for the greedy student is \`abs(delta)\` on a wrong action and zero otherwise. Consequently, statewise \`regret <= abs(delta-pred_delta) <= abs(Q0-Qhat0)+abs(Q1-Qhat1)\`. These inequalities require no uniform-error assumption. They give a directly measurable regret surrogate; the existing MOR-12 uniform \`2 epsilon\` theorem remains valid independently.

**Primary supervised endpoint:** exact-teacher action regret **conditional on states with \`abs(delta)>0\`** in the independently frozen supervised test generator. Report the informative denominator, action disagreements, and paired baseline/student values. A zero-informative test is inconclusive and must remain in the receipt. Secondary metrics are unconditional mean regret, absolute Q-value error, margin MAE, and the fraction whose teacher action is informative.

**Primary policy endpoint:** number of correct certificates after exactly 80 queries on frozen scenarios, with matched cell-specific Bernoulli stream prefixes. Also report the 32/56 checkpoints, the bottleneck and horizon-two oracle policies, true frozen macro TV error, remaining certification gap, 0/1 false-certificate and coverage indicators, teacher calls, and student inference MACs.

## Frozen test contract

The frozen module fixes eight new ordered two-state rate pairs, three supervised seeds, three separate rollout seeds, the existing three outcome-independent sampling schedules, ten supervised checkpoints, and policy checkpoints 32/56/80. This source is imported only after training for evaluation. Its rate pairs and seed groups are disjoint from all QO-14 train/validation/test generators. Comparing to QO-14's previously studied test distribution would be a retrospective diagnostic, not the primary QO-15 result.

Policy arms: bottleneck, unchanged frozen QO-14 RNN weights, boundary-enriched QO-15 RNN, exact QO-13 horizon-two teacher. Both RNNs use readiness \`min(cell_count)>=8\`, a fixed \`max(predicted Q)>=0.05\` engagement gate, and identical bottleneck fallback. QO-14 weights are read unchanged from the checked-in QO-14 receipt; QO-15 starts at QO-14's original random initialization and architecture, then fits on the new training rows. All policies use the same QO-07 certificate authority and the same query budget. A model is never given the simulator's true Bernoulli rates.

## Training augmentation and ablation

Only QO-14's original **training** and **validation** rate families and independent seed sets supply augmentation parent ledgers. Reconstruct their observation streams with the source-fixed schedule and RNG algorithm. Record the parent ledgers at t in {32,40,48,56}. For each, enumerate all two cells and both outcomes for one step, then the same four possibilities after each valid one-step child. The resulting one- and two-step successor states are genuine QO-07 ledger histories; exclude empty confidence intersections, already-certified states, and states with fewer than eight samples in either cell.

Rank candidates using the **public** QO-07 distance of the optimistic or robust error bound to the fixed TV tolerance; select a deterministic cap of 96 unique candidates per training rate family and 64 per validation family. Only after selection compute the exact rational QO-13 pair. The public-state ranking is independent of exact action margins, teacher preferences, test labels, or private Bernoulli rates. The cap and fixed ordering also bound oracle compute. Deduplicate by exact evidence state, including nested confidence endpoints, counts and successes.

The enriched dataset consists of the source training examples, selected frontier candidates, their cell-swap copies (two-action permutation symmetry), and 24 extra copies of every informative frontier example and its swapped copy. Validation uses the analogous disjoint rate family, with eight extra copies of informative candidates. This is a deliberately simple exact-Q-target oversampling ablation. It uses the original QO-14 squared-Q training objective and gradient implementation, keeping the architecture and inference cost identical. A positive result isolates the effect of data distribution more cleanly than changing the loss, head and gate simultaneously.

Use the existing 65 epochs, learning rate 0.012, gradient clipping, Adam seed, and validation-only epoch selection. Archive every selected sample count, informative fraction, and result without searching the held-out test to tune anything. Report the imbalance and exact-label generation cost.

## Interpretation and stop rule

QO-14's exact H2 oracle selected an independent probe only 13 times in 963 eligible choices, and matched bottleneck's 6/18 certificate count. Because the oracle itself failed to beat the bottleneck there, improved student regret on QO-15's frozen distribution may leave the rollout certificate count unchanged. Such an outcome is informative: it redirects QO-16 toward dense expected-gap value targets (QO-11/QO-12) or longer-horizon acquisition, rather than ever larger Q-value networks.

QO-15 must keep any negative result, cannot alter the frozen generator after evaluation, and cannot describe the learned recurrent predictor as a certified error-bound model. Anytime confidence validity follows from QO-07 under its original addressability/IID conditions; the learned policy is a predictable choice of which cell to probe.
