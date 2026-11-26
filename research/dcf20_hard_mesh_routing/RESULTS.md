# DCF-20 results: routing in a harder mesh world

All runs: CPU, one process each. DCF-15 mesh, operators and calibrated batch cost model; DCF-19 accounting.

## 1. World scan (world-selection seeds, ρ = .5, 64 episodes per world)

| noise | XOR share of cross sensors | triangle | aligned queries | decision-change rate (exact vs BP) | epoch headroom |
|---|---:|---|---|---:|---:|
| low | 1/3 | no | no | 2.6% | .00027 |
| low | 1/3 | no | yes | 9.1% | 0 |
| low | 1/3 | yes | yes | 10.0% | 0 |
| low | 1 | no | no | 5.7% | 0 |
| **low** | **1** | **no** | **yes** | **18.6%** | 0 |
| low | 1 | yes | yes | 17.1% | 0 |
| tight | 1/3 | yes | no | 1.9% | **.00050** (selected) |
| tight | 1 | yes | yes | 16.1% | 0 |

The table shows 8 of the 16 worlds; all 16 are in `world20.json`. Aligned XOR queries straddling a cross-unit sensor make BP's decision differ from exact up to 18.6% of the time. In every world with a decision-change rate above 3%, though, fixed exact is the best operator in nearly every batch-epoch. Epoch-level routing headroom there is at most .00006.

## 2. Gated DCF-20 (predeclared world and policy selection)

The predeclared criterion (largest epoch headroom) selected the tight/triangle/unaligned world, which is barely harder than DCF-15. Development selected the BP probe (δ = .3, φ = .02).

| Gate (6 fresh held-out conditions) | Result |
|---|---|
| W1: world materially harder (decision change ≥ 5% and headroom ≥ .01) | **FAIL**: 1.8%, .0005 |
| R1: beats the dev-strongest fixed operator (fixed exact) | **FAIL**: 0 wins, 6 losses |
| R2: beats the better fixed operator in ≥ 3/6 | **FAIL**: 0/6 |

Mean objective: fixed BP .2777, fixed exact .2835, selected probe .2829, epoch oracle .2841, query oracle .2859.

Deviation: `regime19`'s trace helper wrote the DCF-20 traces into `../dcf19_regime_routing`. The chained commit that was meant to freeze `selection20.json` therefore failed, and the held-out stage ran before that commit existed. The file itself was written before the held-out trace (23:06:03 vs 23:12:43). Re-deriving it from the dev trace gives the same policy and parameters. The traces now live here.

**Lesson.** Selecting a world by epoch headroom picks worlds where BP is already nearly right. Headroom for *epoch* routing is near zero at both ends: where BP is good, neither operator matters; where BP is bad, exact wins every epoch.

## 3. Exploratory λ sweep (not gated; `explore20.json`)

Setup: hardest world (low noise, all-XOR cross sensors, aligned queries), fresh seeds, ρ ∈ {.3, .5, .7}.
- BP decides differently from exact on 15–23% of queries. Fixed exact beats fixed BP by .11 in expected utility.
- Epoch headroom is 0 up to λ = 0.1. Per-query headroom grows with the compute price: .003 at λ = .01, .027 at .1, .077 at .3.
- Regime, probe and near-threshold escalation all lose to fixed exact by .02–.04 at λ ≤ .1.
- **95% of BP/exact decision changes (1,116 of 1,173) occur where BP abstains.** BP's independence read-out on aligned XOR queries collapses toward p ≈ .5, inside the abstention band and away from the thresholds, so near-threshold margins miss it.
- Escalating BP's abstentions to exact crossed over fixed exact between λ = .03 and .1 on these traces. That observation is the hypothesis behind DCF-20B.

## 4. Gated DCF-20B: escalate-on-abstention (fresh seeds 990000+, protocol committed before the run)

Policy: BP for every query in the step (one launch), then exact for the queries BP abstains on (second launch). It has no tuned parameters. The compute prices λ = 0.1 and 0.3 utility/ms were declared in `PROTOCOL20B.json` before the run.

| Gate | λ = 0.1 | λ = 0.3 |
|---|---|---|
| G1: beats the better fixed operator in > 3/6 conditions | **PASS, 6/6** | **PASS, 6/6** |

| ρ | λ = .1: escalate | fixed BP | fixed exact | λ = .3: escalate | fixed BP | fixed exact |
|---:|---:|---:|---:|---:|---:|---:|
| .2 | **.2800** | .2350 | .2626 | **.2380** | .2320 | .1851 |
| .3 | **.2968** | .2368 | .2808 | **.2366** | .2231 | .1860 |
| .4 | **.3302** | .2665 | .3151 | **.2647** | .2461 | .2133 |
| .5 | **.3508** | .2878 | .3356 | **.2821** | .2640 | .2286 |
| .6 | **.3473** | .2538 | .3440 | **.2525** | .2104 | .2330 |
| .7 | **.3691** | .2702 | .3687 | **.2729** | .2236 | .2566 |
| mean | **.3290** | .2584 | .3178 | **.2578** | .2332 | .2171 |

Decision-change rate on these seeds: 11–21%. Escalation sends 62.5% of queries to exact. Expected utility before cost: escalate .3647, fixed exact .3681, fixed BP .2709. The policy gives up .003 of utility and skips exact inference on the 37.5% of queries BP decides. Selective risk: .074 vs .069 for fixed exact.

Qualifications:
- The world was chosen as the hardest in the scan, and the policy idea came from the exploratory traces. The gate is a fresh-seed confirmation of a hypothesis formed on other seeds; nothing was tuned on the held-out data.
- At the DCF-15 price (λ = .01) the exploratory traces show this policy *losing* to fixed exact (.381 vs .384). The result holds only when exact compute is expensive relative to a utility unit.
- At ρ = .7, λ = .1 the margin is .0004.
- Kernel costs come from the DCF-15 calibrated model (exact-kernel fit R² .75), not from measured per-step time.

## Verdict

Adaptive routing earns its keep under three conditions together:
- the cheap operator fails on a recognisable subset of queries;
- that subset is flagged by the cheap operator's own output (here, BP abstaining);
- exact computation is expensive.

Escalate-on-abstention is batch-friendly, with at most two homogeneous launches per step, needs no learned component, and passes its gate 6/6 at both declared prices. Routing on structural statistics (DCF-19), on near-threshold margins (DCF-19B) or by per-epoch headroom (DCF-20) did not.
