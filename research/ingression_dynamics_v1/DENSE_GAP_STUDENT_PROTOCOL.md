# QO-16: Dense exact Bellman gap-progress distillation

**Preregistered November 4, 2026.** The independent test generator was committed **first** at \`369e1caf8cd458f9043658d15d0991d66b77bbf7\` and must remain unchanged after outcomes are observed. This study follows QO-14 and QO-15's negative decision-label results and uses QO-12's already specified dense expected certification-gap objective.

## Mathematical definition

For a public, addressable Bernoulli evidence state E, let \`G(E)=U(E)-L(E)\` be QO-11's robust-minus-optimistic certificate gap computed with QO-07's nested anytime confidence intervals. Let E^{a,y} be the exact hypothetical confidence update after probing cell a and observing outcome y. Define the per-first-action expected two-step gap progress:

\[
Q_G(E,a)=G(E) -
\sum_{y\in\{0,1\}}\Pr(y\mid E,a)\,
\min_b\sum_{z\in\{0,1\}}\Pr(z\mid E^{a,y},b)\,
G(E^{a,y;b,z}).
\]

The Bernoulli predictive probabilities are the QO-12 Beta(1,1) heuristic; the second action b can depend on the first bit y. An action with an invalid hypothetical confidence-set intersection is excluded from the QO-12 minimization and gets zero progress when reported as unavailable. A code-independent reference check compares max of both first-action Q_G values to the existing \`optimal_horizon_value(ledger,1,2)\` reference.

The QO-07 e-process and its original assumptions are the sole certificate authority. Predictions are learned planning values and never become confidence bounds. Every action is selected before its Bernoulli draw, with complete source-ordered replay.

### Monotone sigmoid-target transform

The raw exact gap-progress gains are frequently small fractions. To train a sigmoid student without adding new parameters, apply the fixed strictly increasing map

\[
T(q)=q/(q+1/64).
\]

Both actions in a state use the same map, preserving the teacher's action ordering exactly. The registered selection gate is \`max(T(q))>=1/20\`, with bottleneck fallback below it; a trained student's predicted score replaces \`T(q)\` in that gate. Both student architectures are those of QO-14: linear 26 parameters/24 multiply–accumulates, and an eight-hidden-neuron two-token tanh RNN, 138 parameters/240 multiply–accumulates, with full BPTT. The original Adam optimizer (65-epoch maximum, learning rate 0.012, clipping 3, seed 20261104) and validation-only best checkpoint selection are retained. The RNN has no extra trainable head.

For ties, the policy chooses by current widest interval, then fewest counts, then cell index, as the exact QO-12 optimizer does. All active policies require at least eight observations from each cell and use the same predictable bottleneck fallback when the current state has already been certified.

## Strictly separated training and evaluation

**Train/validation:** QO-14's eight original training rate pairs/three original seeds and three original validation pairs/two validation seeds. Only the source-original outcome-independent balanced, skew-low and skew-high sampling schedules generate parent histories. Source checkpoints 24/40/64/96. Every example carries both true exact rational Q_G labels and transformed sigmoid targets, with an additional cell-swapped symmetry copy. Both arms train on identical examples and choose epochs only by their own validation losses. No QO-14 test, QO-15 frozen test or QO-16 frozen test seeds/rates enter learning.

**Independently frozen test:** eight new rational rate pairs with denominator 100, three new supervised seeds, four supervised checkpoints 24/40/64/96. The test generator also commits three separate policy seeds, checkpoint queries 32/80/160, the gate, exact objective, tie-break, and cost-accounting rules. The frozen module is accessed to generate held-out labels only after both model fits have returned. Zero-informative supervised samples remain recorded as such.

**Policy arms:** QO-08 bottleneck, exact QO-12 two-step expected-gap teacher, 26-parameter linear student, 138-parameter recurrent student. Eight scenarios × three policy seeds × four arms × 160 queries yields 96 complete matched trajectories and 15,360 Bernoulli queries. Every per-cell stream prefix is replayed independently and checked.

## Prespecified metrics

Primary supervised: **conditional exact Q_G action regret** on held-out states with unequal exact Q_G values. Record its informative denominator, absolute raw regret, greedy action disagreements, normalized predicted margin MAE, normalized Q MAE, false/missed engagement and raw Q_G margin bands. The MOR-13 Lean theorem applies pointwise to the *normalized* Q values and corresponding predictions; the monotone transform also preserves the argmax of the raw values.

Primary policy: **paired robust-minus-optimistic certification-gap reduction over bottleneck at query 160**. For case and seed, improvement is \`G_bottleneck(E_160) - G_policy(E_160)\`, so positive is better. Report its exact rational average, wins/ties/losses and a deterministic 2,000-resample 95% percentile interval using rate pair as the resampling cluster. The bootstrap interval is descriptive with just eight clusters and no inferential performance guarantees.

Secondary: correct certificates and unresolved decisions at all three registered query budgets, incorrect certificates, observed confidence-coverage failures, exact teacher gap evaluations and candidate counts, student MACs and inference counts, true frozen TV error, full replay hashes, full source hashes and validation histories.

## Execution and outcome handling

Run unit tests, which check parity against QO-12, split isolation, preserved confidence updates, gradient checks inherited from QO-14, and short common-stream rollouts. A first complete QO-16 CI run emits the entire canonical receipt in base64; inspect it, commit it unchanged and rerun the current-head deterministic retraining \`--check\`. Only merge once both the QO-16 and morphospace audits pass on the final PR head.

Negative results, tied teacher Q values, confidence-coverage misses and budget exhaustion remain in the full receipt and results report. The benchmark remains a synthetic two-state, one-action system with no hidden state, neural language model, biological measurement or certified recurrent approximation error.
