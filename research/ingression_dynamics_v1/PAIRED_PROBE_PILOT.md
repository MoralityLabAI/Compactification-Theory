# QO-08: Paired active-query pilot with common Bernoulli streams

**Protocol fixed November 4, 2026, before the complete pilot was executed.** This is a CPU-only *synthetic* comparison of three existing QO-07 probe-selection policies. The paired outcome is descriptive. Six fixed digital seeds are not evidence of real neural learning, transfer, biological competency or general algorithmic superiority.

## Registered question and interface

Does the bottleneck-guided probe rule produce smaller certified worst-case macro-model error at the same number of Bernoulli queries compared with round-robin and largest-width rules?

The policies are:
- **round_robin:** the currently least-sampled addressed state/action cell
- **width:** the widest confidence interval, breaking ties by count and cell identifier
- **bottleneck:** a cell whose uncertainty contributes to the current robust quotient's maximum one-step TV error

The learner receives only a public named-state/action interface and its past observed hit/hold bits. The unknown rate table is used exclusively to generate Bernoulli outcomes and perform post-run scoring. The absorbing goal has its own macrostate.

The frozen casebook is in [paired_probe_pilot.py](paired_probe_pilot.py):

| Scenario | Transient states | Actions | Transient class cap | TV tolerance |
|---|---:|---:|---:|---:|
| two_cluster_feasible | 4 | 2 | 2 | 7/20 |
| separated_infeasible | 3 | 2 | 1 | 1/10 |
| weak_margin_feasible | 3 | 2 | 2 | 1/10 |

The six seeds are 11, 19, 23, 37, 41 and 53. The three fixed cumulative checkpoints are **24, 64 and 128 queries**. Every method runs continuously for 128 queries, using exactly the same confidence parameter alpha=1/20 and 10 exact-rational bisection steps per confidence interval endpoint. The public casebook contains every true probability as a Fraction, and the receipt records it.

## Common random stream and fairness

For scenario index c, seed s, microscopic state i, and action a, generate the independent Python pseudo-random stream using integer seed

    1000003*s + 10000019*c + 9907*i + 7919*a.

A query uses the **next unconsumed** uniform from that cell's stream, returning hit when it is less than the declared Bernoulli rate. With identical scenario and seed, all policies receive identical per-cell response prefixes even when they choose the cells at different times. An independent deterministic replay of every cell stream verifies all receipt sequences.

This is a paired simulation with a fixed Python 3.12 pseudorandom generator, not a physical test of iid Bernoulli randomness.

## Predeclared outcomes and audit

Every case/seed/policy/checkpoint records the complete three-valued decision, true optimum (audit only), robust upper bound, optimistic lower bound, actual frozen macro-model error, ex-post partition regret, maximum confidence interval width, queried-cell count, full state/action query counts and whether the true table is within all intervals.

Wrong certificates and missed confidence coverage are recorded. If a wrong certificate appears when the true rate table lies in all declared intervals, the verification raises an error. Results that remain undetermined are counted explicitly. No scenarios, seeds, policies or checkpoints are dropped because of poor performance.

Within each paired scenario/seed/budget, the reported bottleneck-minus-baseline comparisons use exact Fraction arithmetic. A positive difference in the **baseline minus bottleneck frozen error** means bottleneck was better. Negative values and ties are preserved. No significance test is asserted, and CPU/inference overhead is not treated as free for subsequent real-model comparisons.

## Reproduce

- [paired_probe_pilot.py](paired_probe_pilot.py): casebook, deterministic stream seeds, paired trajectories, exact summary and raw trial records
- [test_paired_probe_pilot.py](test_paired_probe_pilot.py): protocol, equal-budget, replay, outcome accounting and paired-difference checks
- [scripts/verify_paired_probe_pilot.py](../../scripts/verify_paired_probe_pilot.py): source-hashed, compare-or-fail receipt checker
- [machine-readable results](../../results/paired_probe_pilot/receipt.json): all predeclared cases, paired trials, aggregate results and SHA-256 source commitments

From the repo root:

    python -m unittest discover -s research/ingression_dynamics_v1 -p test_paired_probe_pilot.py -v
    PYTHONPATH=. python scripts/verify_paired_probe_pilot.py --check

GitHub CI runs the full existing morphospace suite followed by the same source-bound receipt check. The initial receipt is emitted into a CI log before being committed; it must match the final sources byte-for-byte before merge.

## Evidence boundary

The underlying QO-07 anytime confidence guarantee depends on iid Bernoulli streams for each fixed state/action cell, predictable query selection and the declared fixed state universe. The pilot measures finite pseudo-random realizations of that contract. It does not test trained agents, hidden-state POMDP identification, or new-neural-substrate transfer.

The next independent experiment should train a query controller only on a preregistered training distribution, then evaluate it on sealed problem families with equal query and inference budgets, source-counted evidence, and an untouched control panel.
