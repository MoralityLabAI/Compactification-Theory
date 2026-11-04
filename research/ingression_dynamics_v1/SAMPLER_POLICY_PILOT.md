# QO-08: preregistered paired CPU sampler pilot

**Protocol frozen 2026-11-04 before observing the workflow's results.** This is a reproducible synthetic test of [QO-07](ANYTIME_ACTIVE_QUOTIENT.md). The addressable hit/hold states are invented digital systems. No neural model, biological experiment, API calls, or GPU work are involved.

## Fixed study design

Compare round-robin sampling, largest-confidence-width sampling, and robust quotient bottleneck sampling. Every controller receives the same 96-query budget and exactly the same **cell-indexed Bernoulli stream prefixes** under each public development seed 101 through 112. The simulator hides true rates from the query-selection algorithm. Each policy sees only its own earlier probe outcomes.

**Family A, two_goal_regimes:** four transient states with two-action rates

    (0,0), (0,1/4), (1,3/4), (1,1).

**Family B, three_action_nuisance:** four transient states with three-action rates

    (0,1/2,1/2), (0,1/2,1/2),
    (1,1/2,1/2), (1,1/2,1/2).

Both have an absorbing, separately labelled goal and a maximum of two transient macrostate classes. The confidence level is alpha=1/20, the queried macrostate-error tolerance is 1/4, and the rational interval oracle uses 12 bisection steps. Seeds, tasks, policies, outputs, and query budgets are fixed in [sampler_policy_pilot.py](sampler_policy_pilot.py).

## Predetermined results and interpretation

The primary **descriptive** endpoint is the mean candidate-minus-round-robin *certified robust macro-error upper bound*, paired by seed and reported separately for each family. Negative differences favor the candidate. Report exact mean certificate gaps and evaluator-only true frozen macro errors, as well as all coverage failures, false certificates, and certified feasible/infeasible outcomes.

The program aborts if a certificate is wrong on a seed where all confidence intervals contain the true rate table. Statistical coverage is a probability statement, so every seed remains in the aggregate even when its observed confidence interval fails to include the true rate. The goal is a fidelity audit and limited heuristic comparison, **not** a demonstration that metacognitive active sampling beats baselines across neural tasks.

The 12 seeds and two public synthetic task families constitute development material. Publication of code and seeds makes them unsuitable as a genuinely sealed benchmark. Any model-training experiment must use new evaluator-held task families and identity-verified state preparation, with paired budgets and no post-selection.

## Reproduction and source binding

The runner emits all per-seed records, paired deltas and SHA-256 hashes for the runner, QO-07 implementation, and protocol tests. The output path is create-or-compare: inconsistent existing bytes are rejected. The [tests](test_sampler_policy_pilot.py) check fixed protocol details, replay determinism, matched seed/policy keys, source hashes, and certificates on covered worlds.

The QO-08 GitHub Actions workflow runs the smoke test and the 12-seed finite pilot, then uploads the JSON result as a workflow artifact. No recurring worker is created.

    python -m unittest discover -s research/ingression_dynamics_v1 -p 'test_sampler_policy_pilot.py' -v
    python -m research.ingression_dynamics_v1.sampler_policy_pilot --out results/qo08_pilot.json
