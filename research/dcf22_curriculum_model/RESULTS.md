# DCF-22 results: a compact recurrent model trained on the DCF-16 curriculum

Setup:
- CPU, PyTorch 2.5.1. Training used 3,600 records: DCF-16 train plus 3,000 fresh records from the same four families and sizes, with zero id overlap with val or held-out.
- The epoch was chosen on DCF-16 val (120 records). Held-out (200 records; six of its nine families unseen) was predicted once, after the weights were committed (`models22_weights.json`, `PRE_PREDICT_SHA256`).
- Models: a GRU over elimination-ordered factor tokens (3,175 params) and a pooled MLP over the same tokens (2,707 params, 15% fewer). Three seeds each. Heads: posterior, operator choice, log2 exact cost.
- Inputs never include an operator's posterior (`tests/test_model22.py` checks this).

## Gates (held-out)

| Gate | Result |
|---|---|
| G1: selector non-inferior to symbolic in EU (lower 95% bound > −.01) with fewer ops, ≥ 2/3 GRU seeds | **FAIL**: lower bounds −.108, −.041, −.071 |
| G2: GRU posterior MSE below the MLP's (mean and ≥ 2/3 paired seeds) | **FAIL**: mean .1368 vs .1373, but only 1 of 3 paired seeds |
| G3: acting on the GRU posterior beats BP at budget (lower 95% bound > 0), ≥ 2/3 seeds | **FAIL**: posterior EU ≈ 0 vs BP .229 in every seed |

## Selector (choice head acting on the chosen operator's stored answer)

| Predictor | Held-out EU | Mean operator ops | Matches label |
|---|---:|---:|---:|
| symbolic (EvalMath reference) | .352 | 8,009 | — |
| GRU seeds 0 / 1 / 2 | .301 / .353 / .331 | 1,921 / 2,323 / 1,932 | .62 / .67 / .68 |
| MLP seeds 0 / 1 / 2 | .355 / .339 / .360 | 2,031 / 1,922 / 2,046 | .70 / .71 / .71 |

The learned selectors use about a quarter of symbolic's operator ops. They get there by choosing cheaper exact operators and abstaining more, and lose 0 to .05 of utility; no seed is non-inferior at the .01 margin. Each prediction also costs about 34,700 multiply-accumulates (GRU) or 8,800 (MLP). That is more than the selector's operator ops; the units differ, and neither gate charges it.

## Posterior head (amortised inference)

| Split | constant p = .5 | GRU (mean of seeds) | MLP (mean of seeds) |
|---|---:|---:|---:|
| train | .1256 | .1148 | .1071 |
| val | .1264 | .1136 | .1021 |
| held-out | **.1361** | .1368 | .1373 |

MSE is measured against the verified reference posterior. In-distribution the models remove only 10–19% of the constant predictor's error. On held-out families both are slightly worse than the constant. Acting on the learned posterior commits to a decision on 0–11% of held-out queries, with selective risk .3–1.0, so its EU is about 0. Loopy BP, by comparison, earns .229.

The complexity head reaches R² .09–.38 for log2 exact cost on held-out (GRU). The symbolic count it is trying to learn is exact and free (DCF-13 T3, DCF-14).

## Verdict

At this scale, about 3,000 parameters and 3,600 training records, a compact model trained on the curriculum learns neither amortised parity posteriors nor a selector that matches the symbolic rule. Recurrence over elimination-ordered factors gives no advantage over pooling. What the model can learn cheaply (exact cost, which operator is affordable) is already computable exactly by symbolic analysis. What the symbolic rule cannot do (posteriors without running inference) the model does not learn. A model that should beat this baseline needs either orders of magnitude more data and capacity for amortised inference, or an architecture that executes elimination-like message passing rather than summarising factor tokens.

Deviation: after the weights were frozen, a swapped loop variable in `stage_predict` (class index vs cost estimate) crashed the first prediction run before any prediction was written. It was fixed in prediction plumbing only. The weights, protocol and inputs are unchanged.
