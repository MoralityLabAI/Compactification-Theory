# OIC-32: Global fixed-query optimum for prime cycles

**Executed 11 November 2026.** OIC-T64 proves analytically that every prime cycle `p>=5` needs **at least 3p fixed marker queries** to guarantee shift recovery with at most one substituted report. OIC-T58's three-block construction attains this lower bound. The theorem covers all `p` addresses, resolving the global Q20 sensor feasibility issue left open in OIC-30/31.

\[
\boxed{Q^*_{\mathrm{fixed,flip}}(p)=3p\quad(p\ge5\text{ prime}).}
\]

The seven-phase global minimum is thus exactly **21 queries**, with a three-address block schedule. The previous HiGHS Q20 infeasibility result becomes a cross-check, and the mathematical proof stands independently of solver certificates.

## Independent integer checks

| Prime p | Forbidden horizon 3p−1 | Moment upper if robust | Moment lower from query pairs | Contradiction gap | Attained at 3p |
|---:|---:|---:|---:|---:|---:|
| 5 | 14 | 68 | 78 | 10 | yes, distance 3 |
| 7 | 20 | 139 | 171 | 32 | yes, distance 3 |
| 11 | 32 | 356 | 465 | 109 | yes, distance 3 |
| 13 | 38 | 502 | 666 | 164 | yes, distance 3 |

The pure-Python reference calculates affine-line weights from raw query schedules, audits a multiset-correct pair-incidence identity, evaluates the algebraic contradiction and checks the full cross-shift codebooks of the attaining three-block words. Seven unit tests pass in normal Python and with optimization enabled.

## Resource and evidence account

- `oic32_global.py`: standard-library exact finite reference and symbolic-polynomial integer inequality checker
- `test_oic32_global.py`: seven deterministic tests, including repeated sensor queries and randomized negative controls
- `oic32_run.py`: deterministic pinned machine-readable replay
- `results/oic32_reference.json`: prime examples with both moment bounds and attained code distance
- `OIC32_THEORY.md`: uniform general theorem, with separate incidence identity

The proof does not assume that a marker address is missing. It is a stronger result than the earlier proper-marker bound. The adaptive n=7 query threshold remains within **15–20**, with its Q19 search still inconclusive after the registered caps. See `OIC33_PROTOCOL.md` for the next search.

## Small-prime exceptions (OIC-T66)

A separate exhaustive search normalizes the first sensor marker to zero by the exact global translation gauge. It enumerates all 16 length-five and 32 length-six schedules for `p=2`, and all 729 length-seven and 2,187 length-eight schedules for `p=3`.

The globally optimal fixed one-substitution query horizon for **every prime phase size** is consequently `6` at `p=2`, `8` at `p=3`, and `3p` for `p>=5`. Exact positive witnesses and negative exhaustive maxima are pinned in `results/oic32_exceptional.json`. The combined OIC-32 suite has **12 tests**, each passing with and without Python assertions enabled.

## Certified strict adaptive advantage

At p=7, the previously verified 20-query adaptive policy of OIC-28 now has a **mathematically certified strict query-count advantage** over every fixed policy: OIC-T64 proves the fixed minimum is 21. At p=5, the adaptive optimum 14 is one below the fixed optimum 15; at n=6, the adaptive optimum 15 is at least three below the existing fixed lower bound 18. These are returned-query comparisons; online sensor-address and policy-description costs are separately charged.

## Unrestricted one-erasure result (OIC-T68/T69)

A second affine-plane moment argument proves that **every prime** cycle has a globally optimal fixed single-marked-erasure horizon of **2p queries**, allowing the full marker alphabet. For p>=5, ruling out Q=2p−1 gives a pair-moment upper bound `3p(p−1)/2` and a strict lower bound `2(p−1)²`, separated by `(p−1)(p−4)/2`. Two distinct p-query marker blocks attain distance two. The small cases p=2 and p=3 are covered by complete normalized enumerations of Q=3 and Q=5 schedules.

The same standard-library module verifies p=5,7,11,13 and the two-block constructive code distance. Five additional unit tests bring the OIC-32 total to **17 tests per Python execution mode**. The pinned erasure receipt is `results/oic32_erasure.json`.
