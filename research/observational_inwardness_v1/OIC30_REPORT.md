# OIC-30: Three-marker 21-query code and conditional optimality

**Executed November 11, 2026. Evidence grade:** exact constructive fixed-address certificate, analytical prime-cycle code proof, and a fully enumerated two-label obstruction. The adaptive seven-phase minimum and unrestricted seven-marker fixed minimum remain open.

## New result

A simple schedule observes addresses 0, 1 and 2 for seven consecutive steps each. All 1,029 pairs of hidden worlds with distinct modular shifts have Hamming distance at least three, and 1,078 zero/single-substitution paths decode unambiguously. The minimum code distance equals three.

| Schedule | Query count | Distinct sensor labels | Literal first-episode link bits | Evidence |
|---|---:|---:|---:|---|
| OIC-28 adaptive | 20 | up to 7 | 80 online, plus policy storage | verified 1,697-node tree |
| **OIC-30 fixed** | **21** | **3** | **72** with offline preprogramming | analytical proof and full replay |
| OIC-29 fixed | 22 | 3 | 75 with offline preprogramming | exact replay |
| OIC-29 fixed | 27 | 2 | 60 with offline preprogramming | exact replay |

The fixed code uses 21 response bits and a three-label physical sensor set. Its latency is 21 readings; the adaptive code uses 20 readings with dynamic sensor addresses.

## Theorems

- **OIC-T58:** For prime `p` and `2≤k<p`, `k` different marker positions queried for `p` consecutive time steps each produce an exact minimum cross-shift Hamming distance of `k`.
- **OIC-T59:** At horizon `3p`, every fixed policy using at most two sensor labels fails to correct one substituted report, for any odd prime `p`.
- **OIC-T60:** For any cyclic size `n`, a fixed schedule using fewer than all `n` sensor labels must use at least `n(2e+1)` queries to correct up to `e` substituted reports. For seven phases and one flip, OIC-30 attains the sharp **21-query** lower bound in this resource class.
- **OIC-T61:** Fixed-schedule Hamming distances factor through a phase-residue occupancy matrix `c_{r,m}`, allowing an exact integer formulation with `n²` variables.

## Executed checks

The independently implemented exact integer checker enumerates all 1,029 cross-shift pairs, 1,078 no-fault / single-flip cases and all 16,384 normalized two-marker count configurations at 21 queries. Eleven unit tests pass in normal and optimized Python. The prime block construction is also checked at `p=3,5,7,11,13` with appropriate block counts.

The exploratory SciPy/HiGHS reduced MILP (49 integer variables) found a 21-query, seven-marker witness in 2.3 seconds using 506 solver nodes. It was superseded by the simple three-marker block schedule. The restricted three-marker MILP independently found a valid Q21 schedule in 0.04 seconds. A 20-query reduced MILP did not return a verdict in 12 seconds, and an earlier full-variable MILP did not return a verdict in 25 seconds. Both were **time-limit results**, providing no mathematical lower bound.

## Remaining problems

The unrestricted fixed-schedule horizon is currently between the prior **solver-reported** exclusion at Q20 and the verified Q21 upper bound. A proof-carrying all-marker Q20 infeasibility certificate would establish the exact fixed minimum. The adaptive seven-phase horizon remains between 15 and 20. The next registered protocol targets proof-producing Q20 occupancy feasibility or infeasibility, and adaptive Q19 with explicit resource caps.

No neural implementation, subjective-experience claim, or physical-cosmology result is inferred from these finite sensor codes.


## Extended unrestricted Q20 solver audit

After the initial twelve-second time-limit test, the **exact phase-residue
49-integer-variable** model was rerun for 55 seconds with all seven sensor
positions available. SciPy 1.17.0 / HiGHS returned **INFEASIBLE**
(status 2) after 50.52 seconds. The constraints include all 1,029
different-shift world pairs with minimum code distance 3, and the row totals
(3,3,3,3,3,3,2). The independent older 147-variable formulation
previously reported infeasibility too.

This strengthens the solver evidence for a globally minimal fixed
horizon of 21, but **a solver-reported integer infeasibility result is
not an independently checkable proof certificate**. The full result,
source SHA and runtime limit are pinned in
[results/oic30_q20_solver_status.json](results/oic30_q20_solver_status.json).
OIC-31 retains the exact proof-certificate gate.
