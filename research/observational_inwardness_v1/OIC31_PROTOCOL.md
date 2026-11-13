# OIC-31: Proof-carrying unrestricted fixed Q20 and adaptive Q19

**Prospective, November 11, 2026.** OIC-30 proves a sharp 21-query, three-marker fixed code for one error on seven hidden phases and proves that 21 is the minimum for any fixed monitor omitting at least one of the seven address labels. The full seven-marker Q20 question remains open. The adaptive seven-phase interval remains 15–20.

## Exact fixed-schedule target

Study Q20 with access to all seven marker labels, keeping the same unknown shift/phase and one-substitution sensor model. Use OIC-T61 to represent the schedule as an occupancy matrix over seven phase/time residues with 49 integer variables, row totals `(3,3,3,3,3,3,2)` and 1,029 cross-shift linear distance constraints. Recover a concrete 20-step address word for any feasible integer solution and verify all 1,029 pairwise distances and 49·21 fault paths with a separately implemented integer checker.

For an infeasible outcome, require a **checkable integer-unsatisfiability certificate** or an independently verifiable exhaustive branch-and-bound trace. SciPy/HiGHS exit status alone is solver-reported evidence. Presolve and added symmetry-breaking constraints must have proof arguments, with all-seven-address schedules explicitly retained. The 21-query certificate is an always-valid constructive upper control.

## Adaptive exact target

At Q19, test OIC-28's verified 20-query policy as a branching-ordering reference. Reuse the 49-world, two-error-layer state encoding and the 42 affine gauges with exact back-transformations. Maintain OIC-T44 pairwise pruning. Any additional cut must have its own soundness proof and independent tests. Distinguish CAP, TIMEOUT, UNSAT and SAT in every archived receipt, including node and RAM limits. A winning tree needs complete independent replay over all 49 worlds and every possible fault location.

## Resources

For every policy record returned bits, online address bits, literal offline marker and schedule configuration, policy description size, peak live candidate bits, maximum query latency, peak RSS and measured wall time. A fixed address word may be described by a short generator, but compare it with other policies using an explicit, shared description-code convention.

## Reproducibility

Keep lightweight CI to pure-standard-library exact code, pinned witness receipt replay and two-marker exhaustiveness. Run expensive SAT/MILP/branch-search processes as optional separately budgeted jobs. Any constrained all-marker infeasibility proof must be independently machine-checkable before upgrading its status.


## Registered Q20 status update

The full-label 49-variable occupancy model has now returned SciPy/HiGHS
INFEASIBLE after 50.52 seconds, with all seven addresses permitted.
The previous 147-variable model reported the same result. These
are integer solver decisions; neither supplies a standalone
proof of unsatisfiability. The next milestone is a checkable
certificate and a second solver/checker methodology, rather
than repeated positive witness searches alone.
