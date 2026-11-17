# OIC-33: Tight-pair multihypothesis pruning of the n=7 search

**Executed November 11, 2026.** An exact mandatory-marker restriction was
proved and added to the full 49-world, two-fault-layer adaptive sensor solver.
The 7-phase one-substitution optimum remains in the previously certified
interval 15..20.

## Exact reference benchmark

All rows below use the same affine gauge, original OIC-T44 pair cut,
heuristic 0, `n=7`, and explicit state cap 11,000,000. The new run uses
OIC-T70's mandatory-address intersections in addition to OIC-T44.

| Horizon | Original result | Original expanded states | New result | New expanded states |
|---:|---|---:|---|---:|
| 14 | UNSAT | 6,719,523 | UNSAT | 1,673,753 |
| 20 | SAT | 1,094,946 | SAT | 270,328 |

State-count savings are 75.09% and 75.31%, respectively. With the measured
local native build, Q14 peak resident memory decreased from approximately
385,484 KB to 96,768 KB; Q20 from approximately 69,888 KB to 31,360 KB.
Machine-specific runtimes varied with build and instrumentation. These are
finite algorithmic search improvements, not improvements in learned model
performance.

The Q20 policy contains 1,701 nodes and 782 shift-labeled leaves. The
independent original OIC-28 Python checker validates all public receipt
branches and all 1,029 true-world/fault-position paths. The new solver
produced the same canonical policy tree as the matched unmodified solver.

## Unresolved Q19 searches

- Earlier OIC-29 affine searches at Q15–19 reached resource caps.
- OIC-33 heuristic 0, Q19, state cap 40 million: timed out after 75 seconds,
  reaching no verdict (development snapshot).
- OIC-33 heuristic 1, Q19, state cap 50 million: reached the state cap
  after approximately 106.6 seconds and 2.9 GB peak RSS (development snapshot).
- OIC-33 heuristic 3, Q19, state cap 60 million: timed out after 130 seconds,
  reaching about 3.3 GB peak RSS without a verdict.

Those attempts are *inconclusive*. The pinned CI checks reproduce Q14 and
Q20 to verify the pruning rule and keep the unchanged 15..20 interval.

## Reproduction

Run from the repository root:

~~~sh
python -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic33_tight.py' -v
python -O -m unittest discover -s research/observational_inwardness_v1 -p 'test_oic33_tight.py' -v
python research/observational_inwardness_v1/oic33_reference.py --check
python research/observational_inwardness_v1/oic33_reference.py --cpp-check
~~~

`--check` uses standard-library Python to compare source hashes and replay
the complete pinned SAT tree. `--cpp-check` recompiles both the new and
existing OIC-28 solvers, performs complete Q14/Q20 searches, checks exact
status/state counts, and validates the generated trees. A cap or timeout
fails rather than being labeled infeasible.

The stronger OIC-T70 pruning is proved as a necessary condition. It is
independent of the oracle's learned policy or physical/cosmological claims.
