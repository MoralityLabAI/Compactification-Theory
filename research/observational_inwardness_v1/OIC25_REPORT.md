# OIC-25 executed result: 14 queries needed for robust shift identification on Z5

**Date:** 11 November 2026. **Status:** exact finite, computer-assisted certificate and independent double implementation. No trained-model experiment.

## Question

Under an adaptive, one-address-per-step binary sensor, how many readings are needed to recover an arbitrary hidden shift \(d\in\mathbb Z_5\), with unknown starting phase, despite one arbitrarily substituted sensor report?

## Outcome

The minimum number of queries is **14**. A 14-query decision tree guarantees correct recovery of the shift for all hidden worlds and legal online faults. A full finite Bellman search rules out 13 queries.

| Horizon | Python symmetry + OIC-T44 | C++ symmetry + OIC-T44 | C++ symmetry, no pair pruning |
|---:|---|---|---|
| 11 | UNSAT, 35,542 states | UNSAT, 35,542 states | reference |
| 12 | UNSAT, 125,216 states | UNSAT, 125,217 states | UNSAT, 1,355,518 states |
| 13 | UNSAT, 379,887 states | UNSAT, 379,889 states | UNSAT, 2,597,699 states |
| 14 | SAT, 922 states | SAT, 926 states | not needed |

The slight state-count differences follow implementation details around unreachable empty branches; both implementations agree on every existence result. The C++ negative search at thirteen queries has an independently disabled pairwise pruning rule. Every search completed within its explicit resource cap.

## Independent positive certificate

- 662 nodes in the stored adaptive decision tree
- All 25 true shift/phase combinations tested
- 327 independently enumerated valid report/fault paths
- The same winning decision tree reconstructed by Python and C++
- Every tree leaf correctly identifies the shift

## What changed

The older OIC-24 attempt at twelve queries was capped at 700,000 Python states. An optimized C++ exact-search implementation and an independent Python enumerator with a *sound pairwise residual-disagreement bound* resolve the open case. The extra four queries beyond the earlier \(2n\) necessary bound are specific to the finite five-phase instance, and the previous conjectured \(2n+2\) rule fails at n=5.

The result assumes a known cycle size, exact deterministic cyclic evolution and an adversarial single-bit substitution. A policy with addresses predetermined in advance has different constraints; no fixed-schedule optimality claim is made at 14 queries.

## Repository evidence

- `OIC25_THEORY.md`: mathematical definitions, pairwise obstruction proof and finite exact threshold statement
- `oic25_fast.cpp`: fast exhaustive exact Boolean solver, independently switchable pairwise pruning
- `oic25_independent.py`: separately implemented Python solver and certificate construction
- `oic25_reference.py`: pinned-certificate integrity check, original OIC-23 policy-tree verifier and negative-horizon replay
- `test_oic25_exact.py`: finite fault, independent solver and tampering tests
- `results/oic25_reference.json`: exact search counts, source hashes and artifact hashes
- `results/oic25_n5_q14_tree.json.gz.b64`: complete executable 14-query winning tree

**Research direction:** the next step is a general lower-bound theory explaining why the five-phase cycle needs four queries above \(2n\), together with scalable exact search for n=6. This would test whether a number-theoretic pattern exists beyond the first five cyclic sizes.
