# OIC-27: Exact robust shift recovery for a six-phase cyclic system

**November 11, 2026 · Completed exact finite search and independent tree audit.**

The open OIC-26 one-bit substitution problem for six hidden phases is resolved:

\[
\boxed{Q^*_{\rm adaptive,one\ flip}(6)=15.}
\]

The identical registered observation contract from OIC-21 through OIC-26 is retained: one addressed phase-marker sensor per clock step; one returned binary bit; hidden constant shift \(d\), hidden initial phase \(z_0\); and up to one substituted report that can redirect subsequent queries. The target is exact identification of \(d\) on every possible true world and legal fault path.

## Executed results

| Solver / checker | Q | Result | States / checked paths |
|---|---:|---|---:|
| Translation-quotient C++ solver | 14 | Complete UNSAT | 1,639,144 memo states |
| Affine-quotient C++ solver | 14 | Complete UNSAT | 813,921 memo states |
| Translation-quotient C++ solver | 15 | SAT | 214,548 memo states |
| Affine-quotient C++ solver | 15 | SAT | 210,059 memo states |
| Independent Python verifier, translation policy | 15 | PASS | 998 policy nodes, 477 legal fault paths |
| Independent Python verifier, affine policy | 15 | PASS | 994 policy nodes, 471 legal fault paths |

The two proof trees are distinct, and the verifier contains a separate world-by-world dynamic simulation and a public-history belief-set checker. The unpruned Q=14 C++ search and a separately attempted Python exhaustive search exceeded their execution limits and were explicitly **inconclusive**. The complete UNSAT results use the proved OIC-T44 pairwise obstruction; the new affine reduction is proved in OIC-T47.

The original n=6 one-marked-erasure minimum from OIC-26 remains 11 queries. One substitution is more expensive in this information game because its reported bit can be mistaken for a genuine marker hit and redirect the policy.

## Fixed policy comparison

An analytic even-cycle argument establishes that any six-phase fixed address schedule correcting one substitution needs at least 18 reports. An exact SciPy/HiGHS 0–1 feasibility search reports no 20-query schedule and finds a valid 21-query schedule. Its witness is exhaustively validated across all 540 distinct-shift candidate pairs. Thus the computed adaptive/fixed gap is 15 versus 21 queries, with an **independent analytic proof of strict separation** already at 15 versus at least 18.

The HiGHS infeasibility claim is solver-based; no independent integer-program unsatisfiability certificate is supplied. The adaptive 15-query minimum has stronger cross-implementation exhaustive evidence.

## Verification and provenance

- `oic27_fast.cpp` and `oic27_affine.cpp`: independently implemented 36-world Boolean decision-game solvers, with complete Q=14 searches and distinct Q=15 policy trees
- `oic27_verify.py`: independent truth-world fault-path simulator and belief-edge auditor; explicit validation remains active under `python -O`
- `oic27_reference.py`: SHA-256 pins for sources, canonical compressed certificates, manifest and resource ledger; Python-only deterministic replay
- `test_oic27_exact.py`: eight independent control and mutation tests, passing in normal and optimized Python
- `oic27_fixed_milp.py`: optional SciPy/HiGHS fixed policy feasibility search and exact integer witness verification
- `results/oic27_reference.json`, `results/oic27_n6_flip_policies.json.gz.b64`: frozen evidence for both policy trees

The full negative search is rerun in lightweight GitHub Actions with strict state caps. Any cap-exhaustion status fails CI rather than being treated as impossibility. The test suite checks affine gauge equivariance and a per-step exhaustive version of the pair-disagreement ceiling.

## Resource costs

Adaptive Q=15: 15 returned report bits plus 45 live address bits using a three-bit direct address code, totaling 60 online link bits, plus the 994–998-node policy description. The exact belief model stores two 36-bit candidate layers (72 logical bits), aside from clock and control state. A fixed Q=21 schedule uses 21 returned report bits with preprogrammed addresses and pays its schedule description separately.

**Scope.** A finite, synthetic, known-state-space active-sensing problem. The results provide a testable observation-theoretic control for recurrent interpretability; they do not establish latent dynamics in a trained model or metaphysical inwardness.
