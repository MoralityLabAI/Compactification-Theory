# DCF-19 results: regime-level routing

Setup:
- Repository run, 2026-11-26, one CPU process.
- World, cost model and λ are unchanged from DCF-15.
- τ = 2 was selected on the DCF-15 development seeds and committed (`PRE_TEST_SHA256`) before the held-out traces were generated.
- Held-out: 6 conditions (ρ ∈ {.2, .4, .6} × noise {low, high}), 64 fresh episodes × 32 steps each, batch 32, seeds never used by DCF-15.

## Gates

| Gate | Result |
|---|---|
| R1: regime beats fixed BP in more held-out conditions than it loses | **FAIL: 1 win, 5 losses** |
| R2: regime within .001 of the better fixed operator in ≥ 4/6 | PASS, 5/6 |
| Trace evaluation equals online execution | PASS (`tests/test_regime19.py`, 4 settings, query by query) |

## Held-out outcomes (objective_total = EU − λ·(kernel + routing ms))

| Condition | fixed BP | fixed exact | **regime** | epoch oracle (privileged) | regime exact share |
|---|---:|---:|---:|---:|---:|
| ρ .2, low | .1629 | .1603 | .1629 | .1630 | 0 |
| ρ .2, high | .0680 | .0646 | .0679 | .0680 | 0 |
| ρ .4, low | .1640 | .1638 | .1630 | .1650 | .38 |
| ρ .4, high | .0490 | .0455 | .0489 | .0491 | 0 |
| ρ .6, low | .1631 | .1710 | **.1690** | .1713 | .50 |
| ρ .6, high | .0370 | .0352 | .0365 | .0370 | .12 |
| **mean** | .10732 | .10673 | **.10803** | .10889 | .17 |

| Mean over conditions | EU | Regret | Coverage | Selective risk | Kernel ms/query | Routing ms/query |
|---|---:|---:|---:|---:|---:|---:|
| fixed BP | .1082 | .0035 | .201 | .098 | .085 | 0 |
| fixed exact | .1116 | 0 | .214 | .100 | .489 | 0 |
| regime (τ = 2) | .1100 | .0017 | .206 | .097 | .190 | **.005** |
| epoch oracle | .1105 | .0012 | .207 | .099 | .157 | 0 |

Findings:
- **Routing cost is no longer the binding constraint.** One pass over pending factors per episode per epoch costs 0.005 ms per query, 100–140× less than DCF-15's per-query routers (0.50–0.68 ms). Every step still launches one kernel.
- **The per-condition gate fails anyway.** In three conditions the statistic never crosses τ, so regime and fixed BP make identical choices, and regime loses by its routing cost alone (about 5e-5 utility). It also loses narrowly where it switches to exact without the switch paying off: at ρ = .4 low it mixes, sending 38% of queries to exact, and lands below both fixed operators. Its one win is large: at ρ = .6 low it recovers 75% of the gap between fixed BP and fixed exact.
- **On the mean it is the best non-oracle policy measured on this mesh**: .10803 vs fixed BP .10732. That closes 45% of the gap to the privileged epoch oracle. Most of the gain comes from one condition, so this does not meet the predeclared promotion rule.
- **Correction to DCF-15.** DCF-15's offline tuning objective (`offline_objective` in `mesh15.py`) charged the exact kernel's 1.3 ms launch overhead per query. The simulator charges it once per batched step. That biased the tuned symbolic router toward never choosing exact. DCF-19 uses batch-level accounting for both tuning and evaluation. Fixed exact is the better fixed operator at ρ = .6 low and within .0002 of BP at ρ = .4 low. In the sweep below it beats BP at every batch size ≥ 32.

## Batch sweep (ρ = .4, low noise, 256 fresh episodes)

| B | fixed BP | fixed exact | regime | epoch oracle | regime exact share |
|---:|---:|---:|---:|---:|---:|
| 1 | .1750 | .1629 | .1714 | .1780 | .30 |
| 8 | .1751 | .1744 | .1743 | .1772 | .37 |
| 32 | .1751 | .1756 | .1746 | .1768 | .34 |
| 128 | .1751 | .1759 | .1749 | .1766 | .38 |
| 256 | .1751 | .1760 | .1750 | .1763 | .50 |

At B = 1 the exact kernel's launch overhead dominates (1.8 ms per query), and fixed BP wins clearly. From B = 32 up, batching amortises it and fixed exact overtakes BP. Within this condition the regime statistic does not separate the batch-epochs where exact pays from those where it does not. Regime trails the better fixed operator at every batch size, while the epoch oracle shows 0.001–0.003 of headroom.

## Verdict

Making the routing decision coarse fixes the cost problem DCF-10 to DCF-15 kept hitting, and keeps batches homogeneous. The structural statistic S is too blunt to pick the right epochs reliably, so the gate fails. The next step is a better per-epoch signal, such as the fraction of queries whose local answer sits near a decision threshold, or the pending correlation restricted to the query distribution. Evaluate it against this same epoch oracle.
