# DCF-23 results: learned message passing on the factor graph

Setup:
- CPU, PyTorch 2.5.1. Data and splits are DCF-22's: 3,600 training-family records for training, DCF-16 val (120) for epoch selection, DCF-16 held-out (200) scored once after the weights were committed (`models23_weights.json`, `PRE_PREDICT_SHA256`).
- Two variants, each with 12.7k parameters, 4 message-passing rounds and 3 seeds:
  - `mp_bp`: BP beliefs as inputs; output = BP's parity logit plus a learned correction;
  - `mp_pure`: graph and factor tables only, no BP input.
- BP is `fg.loopy_bp` with 50 iterations and damping .5. It is both an input and the baseline.

## Gates (held-out; all seeds pass)

| Gate | Result |
|---|---|
| M1: `mp_bp` posterior MSE below BP's, ≥ 2/3 seeds | **PASS, 3/3**: .0097 / .0102 / .0099 vs BP .0110 |
| M2: acting on `mp_bp` beats acting on BP in EU (lower 95% bound > 0), ≥ 2/3 seeds | **PASS, 3/3**: EU .407 / .408 / .414 vs BP .370; lower bounds .010 / .011 / .014 |
| M3: `mp_pure` posterior MSE below the constant 0.5 predictor's, ≥ 2/3 seeds | **PASS, 3/3**: .105 / .093 / .117 vs .136 |

## Where the gain comes from (`mp_bp`, mean of seeds)

| Family | Seen in training | BP MSE | mp_bp MSE | EU gain over BP |
|---|---|---:|---:|---:|
| chain (n = 48, 64) | family yes, size no | .0106 | .0072 | +.110 |
| ktree | no | .0019 | .0004 | +.111 |
| dense | no | .0014 | .0006 | +.068 |
| star | no | .0014 | .0011 | +.035 |
| weak_dense | no | .0218 | .0158 | +.031 |
| grid (n = 48, 64) | family yes, size no | .0034 | .0043 | 0 |
| sparse (n = 48, 64) | family yes, size no | .0015 | .0007 | 0 |
| hyper | no | .0091 | .0089 | 0 |
| xor_noiseless | no | .0391 | .0395 | 0 |

The correction transfers to four of six unseen families and to larger sizes of the training families. It does nothing on the higher-arity parity structures (hypergraphs, noise-free XOR systems). Those were absent from training: the training families have only pairwise multi-variable factors.

## Context

| | Held-out EU | Cost per query |
|---|---:|---:|
| exact posterior (unconstrained ceiling) | .446 | symbolic exact ops: mean 8,009 within budget (EvalMath symbolic) |
| **mp_bp (mean of seeds)** | **.410** | BP 29,798 counted ops + 2.89M model multiply-accumulates |
| BP (50 iterations) | .370 | 29,798 counted ops |
| EvalMath symbolic (budget-enforced) | .352 | 8,009 counted ops |
| mp_pure (mean of seeds) | −.031 | 2.89M multiply-accumulates |

- `mp_bp` closes about half of the BP-to-exact utility gap. It does this by learning where BP's independence read-out and loop errors mislead the decision.
- It is not a compute win on this curriculum. Exact inference on these records is cheaper than BP in counted operations (DCF-21), and the model adds about 100× BP's operation count in multiply-accumulates. The units differ, but the gap is two orders of magnitude.
- Neither `mp_bp` nor BP respects the records' compute budgets. Both run BP to 50 iterations regardless.
- `mp_pure` passes M3 but is badly overconfident: it commits to decisions it gets wrong, and its EU is below zero.

## Verdict

Learning on the factor graph works where DCF-22's token summaries did not. A message-passing correction to BP improves posteriors and decisions on held-out data in every seed, and transfers to most unseen graph families, which makes it the first learned component in the series to pass its gates. It answers the accuracy question, not the compute question. On this curriculum it costs far more than the exact inference it approximates, so it earns its place only where exact inference is genuinely infeasible (high width) and BP is the realistic alternative. That regime is not covered by DCF-16's verified held-out set, because verification requires exact inference.
