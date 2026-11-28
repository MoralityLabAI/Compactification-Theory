# DCF-24 results: the frozen DCF-23 model on wide graphs

Setup:
- 88 fresh instances: dense n 16–28 (an unseen family), grids n 64–144 and sparse random n 48–128.
- All 88 are verified by two independent exact paths. Induced width ranges from 5 to 19 (median 12), against at most about 8 in DCF-23's training families.
- The three DCF-23 `mp_bp` seeds were loaded frozen; there was no retraining. The protocol was committed before any instance was generated.

## Gates

| Gate | Result |
|---|---|
| H1: frozen model beats BP in EU (lower 95% > 0), ≥ 2/3 seeds | **FAIL**: gains +.005 / +.008 / +.012, lower bounds −.017 / −.013 / −.003 |
| H2: same, on instances where exact costs more than BP + model (≥ 20 such) | **FAIL**: only 9 such instances (all dense); difference there exactly 0 |
| H3: "exact if within cap, else model" beats "exact if within cap, else BP" at caps 2^16 and 2^20 | **FAIL**: lower bounds −.017 to 0.000 |

## What the wide regime looks like

| | Value |
|---|---|
| BP posterior MSE vs reference | .0036 |
| frozen model posterior MSE (seeds) | .0040 / .0034 / .0039 |
| EU: exact operator (solver model) / BP / model (mean of seeds) | .594 / .596 / .605 |
| median counted ops: exact / BP | 147,744 / 45,034 |
| median model multiply-accumulates | 4,994,592 |
| instances where exact costs more than BP + model | 9 of 88 |

At cap 2^16, 39% of instances go to exact. The learned-fallback policy then costs 4.26M per query on average against 47k for the BP fallback, for a gain of +.005 to +.012 that is not significant.

## Interpretation

- **BP is nearly exact in this wide regime.** Dense, grid and sparse graphs with these sensor noise levels have weak enough loop correlations that BP's posterior error is lower than on DCF-16's held-out set (.0036 vs .0110). Acting on BP earns as much as acting on the exact solver-model posterior. Width makes exact inference expensive, but it does not make BP wrong.
- **DCF-23's gains came from specific BP failure modes,** namely cross-block parity read-outs on chains and loops in k-trees, not from width. Those failure modes are rare here.
- **The model is almost never the cheapest accurate option.** It is cheaper than exact on only 9 dense instances, where BP is already right. Elsewhere exact elimination is cheaper than the model's 5M multiply-accumulates, even at width 16.

## Verdict

The learned message-passing correction does not earn a place as the fallback for wide graphs in these families: there is nothing for it to correct, and it costs 30–100× more than BP. Learned correction would pay off in a regime that combines exact inference too expensive to run, BP materially wrong, and a model cheaper than exact. None of the verifiable families tested so far shows all three at once. Building such a regime needs graphs with strong frustrated loops at high width, and verifying it needs either exact inference at large compute or certified bounds. That is a separate, more expensive experiment.
