# MIT-11: depth–precision–compression frontier for certified search

Cross-link from EvalMath's Mathematical Instrument Transfer programme (MoralityLabAI/EvalMath, branch `research/mit01-instrument-transfer`, PR 74; code in `src/spiral_ln/evaluation_depth_precision.py`, runner `gyms/MathematicalInstrumentTransfer/depth_precision.py`, report `MIT09_11_REPORT.md`, paper `paper/certified_instrument_transfer.md`). Evidence: exact rational arithmetic on 36 open synthetic development roots; timings from one shared CPU are diagnostics only.

## Setting in compactification terms

A planner must pick a strategy under a budget of verified attempts and costly noisy instruments. Three compactifications act on its world model:

* **precision**: each likelihood column quantized to k bits (largest-remainder rounding, exact normalization);
* **quotient**: strategies with identical mass and signature are merged exactly (the dynamic signature quotient, lossless);
* **approximate merging**: classes capped at K by merging nearest signatures (lossy, error computed exactly).

A fourth resource is **search depth** d (receding horizon). The certificate plays the role a Lipschitz constant plays in `note/bits-depth-lipschitz.html`: total-variation perturbation of the world model maps to a bound on closure loss, via the coupling constant D = 1 − (1 − d_prior)·min Π(1 − e_j):

    V*_true − V_true(π) ≤ [UB_d − V_M(π)] + 2D,

with UB_d an information-relaxation upper bound at the search horizon.

## Findings (`frontier.csv`, 3 families × 96 configurations)

* **Bits buy certificates, not decisions.** On the certified frontier low precision never appears beyond the cheapest point, because D grows like the per-column rounding radius. On the actual-regret frontier 2 bits at depth 1 is competitive (palette family: mean regret 0.0075 at about 64 ms versus 0 at about 800 ms with full precision, depth 2).
* **Depth is the expensive axis.** Depth 2 tightens the certified gap modestly at about 7× planner time; certified regret reaches zero only with exact search at full precision.
* **Lossless compression barely moves the allocation.** The exact quotient cut planner states by 16% where signatures repeat (1,021 vs 1,212), and changed the matched-state-budget choice of (bits, depth) only at the tightest budget.
* **Lossy merging is uncertifiable here.** Caps of 6 or 3 classes lie only at the cheapest end of the actual-regret frontier; their certificates are near 1.
* **One bit can flip the decision.** 46 exact witnesses where k → k+1 bits changes the root action and lowers regret (EvalMath receipt seed palette 940001: 5 → 6 bits, regret 671/451632 → 0). Regret is not monotone in bits on the same root, and three roots gain nothing in decisions from any precision while their certificates keep improving.

Relation to this repository: the precision-bit obstructions of CGC-01 and the bits–depth–Lipschitz note concern approximation of functions; this experiment measures the same trade-off for a sequential decision problem where the quantity preserved is the optimal action, and shows that preserving the action needs far fewer bits than certifying it.
