# OIC-29: seven-phase search audit and new fixed-schedule certificates

**11 November 2026. Executed exact finite reference and resource-limited searches.**

The principal seven-phase adaptive one-substitution identification interval established by OIC-28 remains

\[
15\le Q^*_{\mathrm{adaptive,flip}}(7)\le20.
\]

The earlier Q14 complete UNSAT run and Q20 complete winning tree were independently replayed in OIC-28. The Q15–19 interval is still unresolved. The searches below are logged to prevent resource-cap exits from being mistaken for mathematical impossibility.

## Search ledger

| Variant | Horizon | Result | States visited | Bound type |
|---|---:|---|---:|---|
| OIC-28 baseline affine, pairwise pruning | 14 | UNSAT | 6,719,523 | exhaustive certified in OIC-28 |
| OIC-28 baseline affine, pairwise pruning | 20 | SAT | 1,097,803 | verified policy in OIC-28 |
| OIC-29 affine, pairwise pruning, action heuristic 0 | 15 | CAP | 20,000,001 | inconclusive |
| OIC-29 affine, pairwise pruning, heuristic 0 | 19 | CAP | 12,000,001 | inconclusive |
| OIC-29 affine, pairwise pruning, heuristics 1,2,3 | 19 | CAP | 8,000,001 each | inconclusive |
| OIC-29 enlarged Q19 attempts | 19 | TIME LIMIT | unavailable | inconclusive |
| OIC-29 HiGHS fixed-address integer model | 21 | TIME LIMIT | solver did not return a primal witness | inconclusive |

The Q19 heuristic-0 12M-state run took 15.47 CPU-search seconds and reached approximately 705 MB peak resident memory. The three separate Q19 8M-state runs completed in approximately 9–10 seconds each and required about 446 MB resident memory. The Q15 20M search reached its cap after approximately 33.2 seconds. The aborted enlarged runs have no complete proof log. Recomputing the negative or positive Q14/Q20 results uses OIC-28's separately pinned source.

## New constructive results

A vectorized stochastic coordinate search proposed two fixed schedule words. Their search process is heuristic and contributes *no impossibility claim*. A separately implemented pure-integer reference then verified all 1,029 distinct-shift codeword pairs and all single-fault receipts for each word:

| Witness | Queries | Available marker labels | Minimum cross-shift Hamming distance | True-world/fault paths |
|---|---:|---:|---:|---:|
| Three-marker | **22** | {0,1,2} | **3** | **1,127** |
| Two-marker | **27** | {0,1} | **3** | **1,372** |

The 22-query word is a new exact upper bound on the fixed-schedule requirement. The 27-query word demonstrates that correction can be obtained with only two labeled marker positions. Both correct up to one substituted bit for every shift and initial phase. Their offline communication costs are analyzed in `OIC29_THEORY.md`.

The previously reported Q20 fixed-schedule HiGHS infeasibility is still awaiting an independent integer UNSAT proof. The Q21 solver returned a time limit after 20.1 seconds. Therefore the exact fixed minimum remains unresolved; an OIC-29 certificate now gives Q*_fixed(7)≤22.

## Files and status

- `oic29_fixed.py`, `test_oic29_fixed.py`, `results/oic29_fixed_reference.json` form a compact fully replayable constructive proof.
- `OIC29_THEORY.md` gives the code-distance proof and two online/offline address-accounting conventions.
- `OIC30_PROTOCOL.md` preregisters exact Q15–19 adaptive search improvements and independently certified fixed Q20/Q21 feasibility.

The new verifier passes six tests in normal Python and six in optimized Python. No theorem-level kernel proof or trained neural experiment is claimed.
