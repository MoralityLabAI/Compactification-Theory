# OIC-28: Seven-phase fault-aware sensing and proof-grade solver crosschecks

**Prospective, November 11, 2026.** OIC-27 establishes the exact one-substitution n=6 adaptive minimum Q=15 and a strict adaptive advantage over fixed addressing. The next research stage tests n=7 with explicit computational and proof limitations.

## Registered exact task

Take n=7, one dynamically addressed binary marker per time step, hidden shift and initial phase uniform on Z7, and at most one substituted report. Search horizons Q=14,... with the full 49-world two-layer Boolean candidate state. Use at least 49 bits in **each** candidate layer (e.g. C++ `unsigned __int128` or a two-word bitset); never silently truncate to `uint64_t` when packing both masks. A controller succeeds only when every admissible corrupted report history identifies one shift.

Exploit affine label gauges with 7 translations and 6 multiplicative units, but formally prove their effect on any memo key and certificate extraction. Retain the pairwise obstruction of OIC-T44. Compare an independently implemented unquotiented or translation-only solver at smaller horizons with the affine one. Set and log explicit memory, wall-time and explored-state budgets; cap exhaustion is **INCONCLUSIVE**, with an exact negative bound established only by complete searches or an analytic proof.

## Evidence grade

For every successful horizon, archive an entire addressed decision tree, source commit hash and canonical SHA-256. Check it with an independently written Python verifier that enumerates each true (d,z0) and every legal online one-flip path. For a negative horizon, provide exhaustive state counts and source-hash pins. Repeat at least one negative check using a separate algorithm or a demonstrably sound different symmetry quotient.

## Further analytical goals

- Find a structural bound using the solution count of `(d-d')t = z'_0-z_0 (mod n)` and worst-case adversarial receipt overlap
- Determine whether the strict adaptive-versus-fixed separation strengthens for prime phase sizes
- Explain the nonmonotone observed substitution thresholds `n=5:14` and `n=6:15` through group-substructure or information-set branch complexity
- Design a proof-producing fixed-schedule SAT/SMT check to replace the solver-reported Q=20 integer-program infeasibility result

## Resource account

Report return bits, direct-address bits `Q ceil(log2 n)`, policy description bits, dynamic belief state storage, exact-search memory peak and measured CPU time. Compare sensor-addressing costs under preprogrammed versus online control. The target is an exact synthetic identifiability benchmark, with neural transfer and physical interpretations requiring separate evidence.
