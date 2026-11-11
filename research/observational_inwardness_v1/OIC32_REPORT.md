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
