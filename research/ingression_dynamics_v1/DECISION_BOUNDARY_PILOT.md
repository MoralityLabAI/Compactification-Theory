# QO-10: Decision-aware fractional probing on a fixed synthetic casebook

**Protocol fixed November 4, 2026 before the complete QO-10 experiment.** This addresses the negative transfer in [QO-09](FRACTIONAL_PROBE_PILOT.md). The companion [QI-04 mathematics](https://github.com/MoralityLabAI/EvalMath/pull/27) constructs candidate Bernoulli-rate tables that **actually cross the goal-visible quotient-feasibility threshold**.

All rate tables, policies, seeds and budgets are committed in [decision_boundary_probe.py](decision_boundary_probe.py) before the full report. This is a source-fixed, independently seeded synthetic study, with publicly readable cases after commit. It is not an independently secret/private benchmark or neural-model training result.

## Mathematical training-side contract

The micro-system has three resettable, named transient states and two actions, plus a separately observable absorbing goal. Every addressed state/action probe reveals an IID Bernoulli hit observation. The controller asks whether the best **one-transient-class** goal-preserving macro-model achieves worst-case projected one-step TV error at most 1/5.

Training baseline rates equal 1/10 at all six state/action cells; its minimum macro error is zero. Six predeclared counterfactuals change either one action-zero cell or two action-one cells to 9/10. Each changes the true macro minimum error to 2/5, so each crosses the actual decision boundary. The QI-04 exact fractional cover for these six alternatives has the certified primal weights

\[
 x=(1,\tfrac12,1,\tfrac12,1,\tfrac12),\qquad \sum x=9/2.
\]

A matching dual certificate proves optimality for this **structured baseline/alternative hypothesis family**. The weighted-fair scheduler queries the cell minimizing the ratio of its previous observation count to its fixed weight. It has access only to public interface dimensions and its probe-count ledger.

The rotated control transports the candidate masks and weights by half a revolution through cell indices, so action one receives higher priority with the same exact LP objective. The existing QO-09 fractional policy, optimized for an unrelated candidate family, is also included. Over the **complete** opposite-decision binary perturbation family, six single-cell changes force all six weights to be at least one: the minimum fractional cost is 6, giving a uniform allocation. Thus the nonuniform candidate is conditional on its narrower declared structural prior.

## Frozen synthetic casebook

| Scenario | True capacity label | Relationship to QI-04 alternatives |
|---|---|---|
| feasible_uniform_shift | Feasible | Nearby baseline probabilities |
| action0_singleton | Infeasible | Perturbation support in training family, shifted rates |
| action1_pair | Infeasible | Action-one pair support in training family, shifted rates |
| action1_singleton_out_of_family | Infeasible | An action-one singleton omitted from the structured prior |
| feasible_graded_rates | Feasible | Graded rates closer to feasibility threshold |

Every true rate is in the source and committed result. The test runner alone uses the rates to generate Bernoulli observations; the policy receives only past outcomes. The matched QO-08 cell-stream seed formula is preserved. The test uses **new seeds** 101,103,107,109,113,127 and cumulative query checkpoints 32,96,192.

The six policies are QO-08 round robin, widest interval and robust bottleneck; QO-09 fractional cover; QI-04 decision-boundary fractional cover; and the rotated QI-04 control. All use the same QO-07 anytime confidence procedure (alpha=1/20, ten outward rational bisection steps). Original QO-08 and QO-09 baseline runners are invoked unchanged.

Five scenarios × six seeds × six policies yield **180 complete trajectories**, **540 checkpoint records** and **34,560 Bernoulli queries**. Every trajectory has a deterministic SHA-256 hash of its query/response sequence and is independently replayed against the same cell-stream prefixes.

## Registered evaluation

At each checkpoint, report all three confidence outcomes: certified feasible, certified infeasible, and undetermined. Record the actual frozen-model TV error (audit only), robust prediction upper bound, optimistic optimum lower bound, per-cell query counts, partition regret, interval-coverage failure, and incorrect certificate. A false certificate on a valid confidence event triggers an exception.

Within each scenario/seed/budget, calculate baseline minus decision-aware error and baseline minus decision-aware robust upper bound. Positive means the decision-aware policy performed better. Negative differences and all unresolved results are retained. The rotated allocation tests whether the result is sensitive to which action is prioritized.

Full exact results, source SHA-256 commitments and 180 response-prefix hashes are stored at [results/decision_boundary_probe/receipt.json](../../results/decision_boundary_probe/receipt.json). The [verifier](../../scripts/verify_decision_boundary_probe.py) reconstructs every paired trajectory and checks the committed bytes. A dedicated read-only-token GitHub Actions workflow runs the entire verifier; the original morphospace workflow runs the unit tests.

**Interpretation boundary:** LP optimality certifies only the declared QI-04 alternative-family information *lower bound*. It makes no claim that the static policy is optimal for the unknown test-table family, or that this synthetic test predicts neural transfer. The anytime e-process probability guarantee still requires stationary IID Bernoulli streams for each separately addressable state/action cell, predictable probes and a fixed state universe.

A useful negative result is any worse average error, lower certificate rate, or strong sensitivity to the rotated prior. Such outcomes remain part of the accepted evidence. Additional cost models must charge planning CPU time, model description, decoder and any future neural inference cost before a compact-model efficiency claim.
