# OIC-26: General fault-correction bounds across cyclic phase sizes

**Prospective, 11 November 2026.** OIC-25 finds an exact five-phase one-substitution threshold of fourteen queries and refutes the proposed all-n rule Q*(n)=2n+2. This protocol isolates the role of modular phase coincidences and scales the exact solver while retaining independent certificate verification.

## Target

Study Q*_flip(n), the minimum number of adaptively addressed one-bit sensors required to identify any hidden shift d under unknown initial phase and at most one substituted report. Preserve the exact OIC-21 sensor contract. Start with n=6; use a 36-world (72-bit candidate-layer) implementation or a generic bitset. Never silently truncate candidate masks to 32 bits.

## Plan

First verify that a generic solver reproduces the exact n=2,3,4,5 thresholds (6,8,10,14) with its pairwise pruning disabled and enabled. Preregister memory and state limits and treat timeouts as inconclusive. Compare branch orderings, symmetry reductions and sound information lower bounds by recording both visited states and wall time. For successful policies, save complete deterministic query trees and run the independent OIC-23 trajectory/fault replay against every true world. For failed horizons, rerun at least one search without the OIC-T44 pairwise bound.

Derive a structural obstruction from the maximum number of temporal intersections between two modular trajectories of different shifts. Investigate whether the observed extra query requirement follows from the unit group, subgroups, or adversarial indistinguishable histories. Distinguish analytically proved general results from finite exhaustive outcomes.

## Sensor/resource controls

Account for one returned bit per query, ceil(log2 n) address bits, policy description, at least 2n² logical candidate bits under the single-substitution Boolean formulation, and exact-search memory. Adaptive policies can have very large decision trees and should be compared with fixed codebooks under identical online communication constraints.

## Acceptance

Report at least one conclusive n=6 exact threshold or a stronger analytic lower bound with a proof independently checked in Python. Any model-based claim must retain a complete adversarial history audit. Publication requires a pinned source hash, a runnable solver, a fault-path replay and a negative-search status that distinguishes exhaustive UNSAT from a resource-limit failure.
