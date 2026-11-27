# DISTILL-12: Amortized exact continuation quotient

Stacked on DISTILL-11. The incumbent and normalized quotient semantics are unchanged.

## Registered protocol

- Eight new calibration roots and twelve fresh roots in each of ordinary, shifted and scarce-credit cohorts. Sixty-four weighted worlds per root. Work caps 8,192 and 32,768; price 1/4096 utility per logical unit.
- Compare literal, scan marginal, bitset marginal, cached scan, cached bitset, and cached bitset with deterministic-experiment pruning, each under the frozen tail and broad gates. Include budget-rule baseline, late shortcut, forced-incumbent controls for the last two kernels, and calibration-selected arm (including abstention).
- Public model only. No persistent cache across simulated worlds or roots. Cache holds at most 32 normalized histograms, keyed by stage and exact mask, with no history-dependent prefix reuse. Charge setup before allocation and each set-bit construction, bin intersection, popcount/accumulation, gcd reduction, lookup, insert and eviction under a declared logical operation model. No uncharged post-exhaustion computation in the frontier. Actual process CPU time is descriptive.
- The bit planes are exact integer prior masses: h[j,k](S)=sum_b 2^b popcount(S & P[j,k] & W[b]). The normalized key and mass factor must match the scan kernel exactly. No float arithmetic.
- Prune only a deterministic screen/control in stages 1 or 2 whose observation leaves the compatible mask unchanged and whose extra cost leaves the saturated budget unchanged. The action changes only ignored flags and actual budget within the same quotient class. It pays extra experimental cost plus incumbent selection work; its exact value difference is nonpositive. Independent auditor checks every recorded rejection.
- Referee-only microbenchmark: up to eight incumbent-reachable states per stage (1,2,3), cold projector per signature versus four repeats of the same stream with one projector; report exact equality, work and measured CPU, no cross-world actor reuse or credit from microbenchmark.
- All accepted opportunity receipts retain independent utility/risk interval and world replay. All forced controls must preserve raw incumbent metrics. Calibration selects only positive mean priced gain with at least half the roots winning and no risk increase; evaluation never retunes.
- No paid API, native DataGoblin integration, weight updates or training admission. A negative cost result remains valid evidence. Cold setup may dominate the bitset kernel.
