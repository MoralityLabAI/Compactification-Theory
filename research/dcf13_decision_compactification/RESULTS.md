# DCF-13 results

All numbers in this file come from receipts committed in this directory. Figures quoted in the earlier conversation, before the repository publication was interrupted, are marked *preliminary*.

## Recovery and reproduction of DCF-13A/B/C

The sources `gym.py`, `eliminate.py`, `hybrid.py` and the protocols were on the branch. All three hash locks verify against the committed LF blobs. The raw receipts were not on the branch, so the three runs were repeated (Python 3.12.10, NumPy 2.3.5, Numba 0.65.1, single thread; the README records Python 3.13.5): `receipt.json`, `followup_receipt.json`, `hybrid_receipt.json`, `REPRODUCTION_SHA256`.

| Quantity | Preliminary | Reproduced |
|---|---:|---:|
| 13A adaptive_full median speedup vs full | 15.08× | 13.69× |
| 13A joint gate (adaptive8) | 19/24 | 20/24 |
| 13B/13C exact-arm expected utility, coverage, regret | as README | identical to 6 decimals (deterministic) |
| 13C hybrid actions identical to full | 3,072/3,072 | 3,072/3,072 |
| 13C hybrid mean ms/128 | 0.676 | 0.556 |
| 13C hybrid median speedup vs full | 23.16× | 20.85× |
| 13C strict per-condition latency gate | 6/24 FAIL | 7/24 FAIL |
| 13B robust-bound coverage (ambiguous) | 1.4% | 1.4323% |

Every decision-level quantity reproduces exactly; only wall-clock timings move with the host.

## DCF-13D (fresh seeds 1301/1303/1307, protocol and calibration committed before the run)

### A. Hybrid by estimated execution cost (36 conditions: K ∈ {8,12,16} × 3 seeds × 4 regimes)

α = 0.3967, the ratio of enumeration to elimination seconds per work unit. It was calibrated on DCF-13B development fixtures and frozen before the fresh run.

| Gate | Result |
|---|---|
| A1: hybrid_cost exact everywhere | PASS (4,608 decisions identical to full, zero skips) |
| A2: hybrid_cost ≤ min(component_full, ve_full) in ≥ 27/36 | **FAIL, 4/36** |
| A3: symbolic work = executed work | PASS, 4,608/4,608 |

| Mean ms/128 | K=8 | K=12 | K=16 |
|---|---:|---:|---:|
| full enumeration | 3.40 | 44.0 | 1282 |
| component_full | 1.04 | 6.88 | 5.46 |
| ve_full | 1.13 | 1.37 | 2.49 |
| hybrid13c (threshold 4, cap 6) | 0.54 | 0.55 | 0.69 |
| hybrid_cost | 0.60 | 0.63 | 0.75 |

The cost-estimated router selects the right operator: it routes 65–98% of bridge episodes to elimination (98% at K=8, 92% at K=12, 65% at K=16) and ≥ 99% of K=16 stable/rewired/ambiguous episodes to enumeration. But computing the component work and the lower bound costs more than it saves. It beat the plain threshold hybrid in 1/36 conditions, and hybrid13c met the same latency criterion in 17/36. The full-enumeration median speedup is 153× (driven by K=16).

### B. Elimination orderings, budgeted abstention, fixed clusters (36 conditions, 4,608 episodes)

| Arm | EU | Regret | Coverage | Selective risk | Skip | Mean ops | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact (any ordering) | .4804 | 0 | .7448 | .0727 | 0 | 137.8 | .0671 | .0982 |
| VE min-fill, budget 32 entries | .4804 | 0 | .7448 | .0727 | 0 | 137.8 | .0671 | .0982 |
| VE min-fill, budget 16 entries | .4769 | .0034 | .7407 | .0726 | .0046 | 135.6 | .0674 | .0990 |
| VE min-fill, budget 8 entries | .2167 | .2637 | .3876 | .0637 | .4405 | 45.5 | .0623 | .1742 |
| fixed 4-blocks | .2403 | .2401 | .4260 | .0950 | 0 | — | .0711 | .1331 |

On these graphs (induced width 1.8–3.0), min-degree and min-fill produce identical widths in all 4,608 episodes. The degree-weighted min-fill variant is worse in 3 and never better. Every non-declined NumPy elimination posterior matches the Numba reference within 1e-9 (gate B1). The fixed-cluster approximation is badly calibrated in the decisions that matter: its selective risk of .095 exceeds the exact .073. Peak live table entries ≤ 100; tracemalloc peak ≤ 27 kB per 128 episodes.

### C. Unknown sensor law (`w ~ U[0,1]` shared per episode; 9 conditions, 1,152 episodes)

Reference: the posterior given the true `w` (privileged). Expected utility is computed under that posterior; realized utility is scored against the generated bit.

| Arm | EU | Realized | Regret | Coverage | Wrong / cond. | Selective risk | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle w (privileged) | .4371 | .4366 | 0 | .7014 | 6.8 | .0793 | .0670 | .1098 |
| **BMA exact (Bernstein)** | **.4055** | .4297 | .0316 | .6727 | 6.2 | .0756 | .0696 | .1149 |
| BMA 5-point grid | .4043 | .4115 | .0329 | .6806 | 6.9 | .0825 | .0774 | .1151 |
| plug-in w = 0.5 | .3908 | .4201 | .0464 | .6545 | 6.0 | .0745 | .0720 | .1185 |
| nominal XOR (w = 0) | .2268 | .2205 | .2104 | .6806 | 11.8 | .1359 | .1039 | .1588 |
| **credal, shared w, depth 6** | .2843 | .2934 | .1528 | **.3976** | 2.7 | .0596 | — | — |
| credal, shared w, depth 0 | .2798 | .2856 | .1573 | .3898 | 2.7 | .0610 | — | — |
| pointwise bounds (13B) | .0140 | .0165 | .4232 | .0165 | 0 | 0 | — | — |

| Gate | Result |
|---|---|
| C1: BMA = 16-point Gauss–Legendre over independent brute masses; oracle = brute | PASS (max Δ 5.6e-16, 3.3e-16; 36 audited episodes) |
| C2: credal soundness | PASS: zero actions with negative expected utility under the true `w` in all 1,152 episodes; minimum audited confidence over a 101-point grid .855 ≥ .8 |
| C3: BMA ≥ plug-in EU; credal coverage > pointwise in ≥ 8/9 | PASS: .4055 vs .3908 (BMA ahead in 7/9 conditions); coverage 9/9 |

Moving from per-factor to shared-parameter uncertainty raises certified coverage 24-fold, from 1.65% to 39.8%, and keeps zero negative-utility actions. Subdivision adds 0.8 points of coverage. The exact BMA closes 32% of the plug-in's regret gap to the oracle. Model uncertainty doubles to triples the relevant variable set (T4): enumerating the polynomials over that set costs about 0.3 s per 128 episodes in NumPy (tracemalloc peak 17 MB at K=16). The DCF-13B pointwise Numba bound costs 5–7 ms; the two implementations are not equally optimised.

## Phase-1 checklist status

| Item | Where | Status |
|---|---|---|
| Full exact enumeration | gym.py `predict(mode 0)` | done, reproduced |
| Query-directed component inference | gym.py `predict(mode 2)` | done, reproduced |
| Min-fill and weighted min-fill elimination | `fg.elimination_order`, 13D-B | done |
| Fixed-cluster approximation | `fixed4`, 13A and 13D-B | done |
| Budgeted component / elimination with abstention | `adaptive4/8`, `ve4/ve6`, 13D-B budgets | done |
| Hybrid by estimated cost | `hybrid_cost`, 13D-A | done; latency gate FAIL |
| BMA over uncertain observation operators | 13D-C | done |
| Conservative posterior-bound decisions | 13B pointwise, 13D-C credal | done |
| Independent exact verifiers | brute NumPy, Gauss–Legendre, cross-implementation | done |
| XOR, AND, majority, noisy unary, ambiguous semantics | all regimes | done |
