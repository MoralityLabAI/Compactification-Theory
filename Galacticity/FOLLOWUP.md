# GAL-01 E/G follow-up receipts

## Existing gym evidence
The successful DISTILL-11 hosted run at
https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/46073402431
printed its full aggregate report. distill11_aggregate.json preserves exact
fractions for 24 paired aggregate cells (3 splits, 2 caps, 4 arms).
The marginal tail arm uses 1.00955x less charged work on ordinary roots
and 1.02462x on shifted roots (same aggregate utility and risk).
Scarce roots have 1.05422x work ratio but differing utility/risk; this is
not an equal-semantics comparison. Both caps give the same tail ratios.
Marginal broad removes many expansions but has total work ratios near one,
including regression in the shifted 32768-cap cell. No aggregate cell
achieves 2x. Aggregates do not establish per-root dominance, and no
horizon sweep, cold-build separation, or tail certificate is supplied.
The row-level adapter needs the archived artifact and was not run against
its individual rows here.

## Learned routing pilot
Run node Galacticity/learned_router.mjs. Fit a four-weight logistic router
by full-batch gradient descent on 768 synthetic cases, then test 768
disjoint construction-cost cases. The public feature extraction uses the
two-point affine estimate (log horizon, log C, log d). Ground truth is
exact integer cost comparison. The learned router scores 729/768 with
39 unsafe early compactification choices; an exact affine-contract gate
scores 768/768. A shared-prefix non-affine adversary demonstrates why
a two-point observation cannot identify the tail. No DataGoblin or LLM
parameter update took place; the learned router is a standalone offline
baseline and training_admission remains false. The affine contract must
be independently established on actual workloads; the gate cannot infer
it from two data points. Synthetic test split is fixed, not a blind
independent external evaluation.

## Next experiment gate
Use public packets with opaque IDs and no oracle fields in the actual
Distill integration; add independent nonlinear controls, horizon grids,
source-family holdouts, cost-accounting and exact interval checkers.
Collect construction and marginal execution costs separately. Deploy
learned suggestions behind certified dominance or abstention, with
independent verification of the workload contract.
