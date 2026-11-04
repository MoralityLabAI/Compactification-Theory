# QO-10: Completed decision-aware probe pilot and negative result

**Evidence date: November 4, 2026.** Complete result: [source-bound machine receipt](../../results/decision_boundary_probe/receipt.json). Reproduction: [QO-10 verifier](../../scripts/verify_decision_boundary_probe.py). Fixed protocol and hypothesis family: [DECISION_BOUNDARY_PILOT.md](DECISION_BOUNDARY_PILOT.md). The receipt was created from the full first pilot execution; this analysis was written **after** the outcomes and is labelled retrospective.

## Experiment completed

The QI-04 fractional-cover probe plan was derived from six candidate Bernoulli hit/hold rate tables, each independently demonstrated to cross **the same** one-class goal-quotient TV feasibility threshold used for evaluation. Its exact primal/dual objective is 9/2 for a structured counterfactual family; rotated action-index controls preserve the same LP optimum. The complete unrestricted opposite-decision family would instead require a uniform allocation.

Five predeclared synthetic scenarios, six new seeds and six unchanged/declared policies were evaluated at 32, 96 and 192 queries under exactly paired per-cell Bernoulli streams. The final result contains **180 trajectories, 540 checkpoints, 34,560 queries**, and 180 independently replayed SHA-256 response hashes. The QO-07 exact-rational time-uniform confidence ledger was used throughout.

At the final 192-query checkpoint, across all 30 case-seed combinations:

| Policy | Correctly certified / 30 | Undetermined / 30 | Mean true frozen one-step TV error |
|---|---:|---:|---:|
| QO-08 robust bottleneck | 16 | 14 | 0.3033 |
| QI-04 decision-aware static fractional | 2 | 28 | 0.3157 |
| QI-04 rotated-control static fractional | 2 | 28 | 0.3293 |
| QO-09 unrelated static fractional | 2 | 28 | 0.3274 |
| QO-08 round robin | 2 | 28 | 0.3180 |
| QO-08 widest confidence interval | 1 | 29 | 0.3052 |

The detailed results at 192 queries, expressed as number of correctly certified seeds out of six:

| Scenario | Bottleneck | Decision-aware | Rotated | Other static fractional | Round robin | Width |
|---|---:|---:|---:|---:|---:|---:|
| Feasible uniform shift | 1 | 1 | 0 | 0 | 1 | 1 |
| Action-zero singleton (near support) | 6 | 1 | 0 | 0 | 0 | 0 |
| Action-one pair (near support) | 4 | 0 | 1 | 1 | 0 | 0 |
| Action-one singleton (outside prior) | 5 | 0 | 1 | 1 | 1 | 0 |
| Feasible graded rates | 0 | 0 | 0 | 0 | 0 | 0 |

At 32 queries, all 30 case-seed combinations were unresolved by every policy. At 96 queries, bottleneck certified 2/30, while all other policies remained unresolved. At 192 queries, the static allocation continued to yield substantially fewer actionable certificates.

**Validity checks:** 0 false certificates, 0 confidence-interval coverage failures, full query-count equality at each checkpoint, and deterministic paired source replays. Every unresolved case remains in the result.

## Interpretation

The static fractional probe allocation is a **globally optimal solution of the declared candidate-family information lower-bound LP**. The actual task requires a three-valued, time-uniform certificate of quotient feasibility for partially observed rate tables. The study provides a concrete counterexample to a tempting claim that optimizing a valid static information lower bound for decision-changing alternatives is sufficient for competitive finite-budget certification.

The robust bottleneck rule actively tracks the current interval endpoints that prevent a feasibility/infeasibility certificate. It can concentrate queries on the rate cells that currently control the margin. The static QI-04 plan receives no observation-driven reallocation and uses only its initial structural prior; its prior also excludes the action-one singleton negative-control case. This supplies a **plausible mechanism** for the observed ranking, which the experiment does not isolate causally.

The one-class quotient and stationary hit/hold assumptions simplify the numerical problem. The pilot uses 30 paired case-seed combinations with deliberately chosen synthetic rates, which do not support a population-level generalization, statistical significance claim, or any conclusion about compact neural agents. It additionally omits measured CPU latency; the bottleneck heuristic invokes exact robust partition optimization during querying, while static weighted-fair scheduling performs cheaper arithmetic. Any future economic comparison must count that difference.

## Next theorem and experiment

A more relevant information design would condition on the **current confidence region** and solve a constrained min-cost query allocation against only those counterfactual models whose rates lie in that region and whose true quotient decision differs. This calls for an *adaptive optimization and a regret/control certificate*, including the possibility that a counterfactual family no longer contains all relevant alternatives. One candidate is

\[
 \min_{x\ge0}\sum_c w_c x_c
 \quad\text{subject to}\quad
 \sum_c x_c\,\operatorname{kl}(\widehat p_c,q_{jc})\ge
 \operatorname{kl}(1-\delta,\delta)
 \quad\text{for every still-plausible opposite-decision }j.
\]

The displayed constraint is a **candidate planning objective**. Fitting \(\widehat p\) and selecting \(j\) from earlier outcomes changes its statistical semantics, so the expected-count KL lower-bound theorem from QI-03 cannot simply be reused as a per-step confidence certificate. Continue to use the independent QO-07 anytime-valid intervals to authorize actual decisions. Independent holdout panel, sample-budget ledger, planner CPU cost and drift controls are needed before making performance claims.
