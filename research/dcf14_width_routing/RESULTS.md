# DCF-14 results: computational width as a routing signal

Repository run, 2026-11-25, CPU only, six worker processes (wall times are parallel-process measurements; counted operations are the primary cost). 200 development instances (chain, tree, grid, sparse; n = 8–32) and 324 held-out instances: star, dense, hyper, ktree, xor_noiseless and weak_dense at n = 8–64; chain/tree/grid/sparse at n = 48, 64; and chain/tree/star/ktree at n = 128, 256. The protocol, selectors and development receipts were committed before the held-out run (`PRE_TEST_SHA256`).

This run is separate from the DCF-14A–C pilots in `../dcf14_structural_width`. Those report conversation-archive results without committed source; this directory commits source, receipts and tests.

## Verification

288/324 held-out instances (and 200/200 dev) have a reference posterior confirmed by at least one independent exact path:
- broadcast VE only: 50
- cutset + broadcast VE: 129
- cutset + enumeration + broadcast VE: 109

Maximum disagreement is 3.3e-15. The 36 unverifiable instances are dense/weak_dense at n ≥ 32 and hyper at n = 64; they are scored by realized utility only. The library's 129 tests check every exact path against a naive full-joint product.

## H1: does induced width predict the cost of exact reasoning?

Single-feature log2-cost models were fit on dev and evaluated on held-out verified instances (n = 288).

| Feature | R² best-exact ops | Spearman | R² VE ops | Spearman |
|---|---:|---:|---:|---:|
| n (log2 hidden-state dimension of the full joint) | −1.32 | .33 | −1.54 | .37 |
| query-component size | −1.59 | .35 | −1.77 | .37 |
| **min-fill induced width** | **.36** | .51 | **.86** | .85 |
| degeneracy lower bound | .37 | .45 | .50 | .68 |
| loop-cutset size | .19 | **.59** | .34 | .87 |

**H1 passes its predeclared criterion**: width's R² exceeds both n and component size by ≥ 0.10, and its rank correlation is higher than both. The negative R² for n and component size is extrapolation failure: the dev-fitted slope predicts exponential cost for 256-variable chains, which actually cost a few thousand operations.

Qualifications, all from the same receipts:
- Within a single family, n predicts about as well as width (Spearman .86–.99 for both), because size drives width there. Width's advantage is across families.
- For the best available exact operator, cutset size ranks costs better than width (.59 vs .51) and width explains only 36% of the variance.
- **Failure case: algebraic structure.** On noise-free XOR systems the GF(2) operator decides exactly in a median of 90 operations, while min-fill elimination needs 4,523 at median width 7. Width prices the wrong algorithm.
- **Failure case: approximation error.** Width does not predict how wrong approximations are. Mean |p − p\*| for BP is .070 at width ≤ 4, .116 at width 5–10 and .043 at width > 10: the high-width families here (dense, weak_dense) have weak couplings where BP is accurate.
- **Failure case: decision value.** On weak_dense (all sensors at flip rate .45), every policy earns ≈ 0. Approximate operators lose utility (exact_else_mbe −.034) by acting on noise; abstention is optimal regardless of width.

## Algorithm selection under an explicit budget

Each instance is scored at budgets of 2^10, 2^14 and 2^18 counted operations, at penalties 4 and 9 (act at p ≥ .8 or ≥ .9). Approximate operators adapt to the budget: BP iteration cap, particle count, and the largest mini-bucket i-bound that fits.

| Policy (held-out, verified, 1,728 decisions) | Mean EU | Mean ops | Routing ms |
|---|---:|---:|---:|
| oracle (hindsight best, privileged) | .3853 | 1,862 | — |
| **exact-else-mini-bucket** (best hand policy on dev) | **.3448** | 9,403 | 9.9 |
| exact-else-BP | .3283 | 6,037 | 9.9 |
| learned MLP (340 parameters, 0.83 s training) | .3068 | 4,191 | 10.0 |
| exact-or-abstain (certified, never wrong in expectation) | .3017 | 2,570 | 9.9 |
| width rule (threshold tuned on dev) | .2703 | 13,340 | 9.9 |
| always BP | .2629 | 16,644 | 0 |
| learned tree (depth 4) | .2135 | 413 | 10.0 |
| random | .0886 | 32,249 | 9.9 |
| always particles with CI abstention | −.1666 | 76,548 | 0 |

**Selector gate S1 fails for both learned selectors.** Paired over 324 held-out conditions (family × n × budget × penalty), in expected utility:
- MLP vs exact-else-MBE: 17 wins, 86 losses, 221 ties
- tree vs exact-else-MBE: 46 wins, 152 losses

On the λ-objective the selectors were trained on (EU − 0.05·ops/budget; descriptive, not the gate), the MLP is level (85 vs 87) and the tree loses (129 vs 147). The learned selectors abstain more often than the oracle off-distribution: the tree abstains on 62% of held-out decisions vs the oracle's 52%. Training labels were dominated by "abstain" (683/1,200), because exact utility is 0 whenever the posterior is uncertain.

By family, exact-else-MBE leads on dense (.614 vs oracle .650) and ktree (.460 vs .498). Every policy that can reach the exact or GF(2) operator ties the oracle on xor_noiseless (.905). Grid and sparse at n = 48–64 show the largest oracle gap (.16 vs .32 and .22 vs .33): budgets cut off exact elimination there, and no fallback chooses well.

Structural routing overhead (min-fill ordering, cutset search, symbolic costs in Python) averages 9.9 ms per decision. That exceeds the operator time on most instances, which repeats the DCF-10–12 finding that routing bookkeeping costs as much as the inference it saves unless it is compiled.

## Verdict

Induced width is the right currency for elimination cost and a better cross-family predictor than state dimension or component size. It does not price algebraic shortcuts, approximation error or decision value. The strongest router is symbolic: exact whenever the symbolic cost fits the budget, otherwise mini-bucket at the largest i-bound that fits. Neither learned selector earns promotion.
