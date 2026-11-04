# MONO-01 research note: conditional-leverage one-shot edits (2026-11-04)

Status: Phases A-D executed on CPU; Phase E (recurrent, Qwen) not started. Algebra, synthetic CPU evidence and simulator outcomes are kept separate below. The ChatGPT reference zip was not available on disk, so `monoshot.py` is an independent implementation of the workorder mathematics; its SHA-256 manifest could not be checked.

## 1. Mathematics (verified by `tests/test_monoshot.py`, 11 tests)

For a linear receiver `y = W x`, keys `K` (n x m), desired output corrections `V` (d x m) and reference second moment `C = lam I + X^T X / N`, the exact constraint `D K = V` with minimum disturbance `tr(D C D^T)` is `D* = V (K^T C^-1 K)^-1 K^T C^-1`, with cost `tr(V (K^T C^-1 K)^-1 V^T)` and rank at most m. Checked: constraint to 1e-10, any feasible perturbation costs more, the cost identity, the support-restricted edit, Woodbury `C^-1 B` against the dense solve, bf16 round-to-nearest-even against torch on 20,000 values, and the rank-one eigenvalue shift (target eigenvalue reached, other eigenvalues unchanged).

Conditional leverage: for one key and support `S`, the optimal disturbance is `||v||^2 / q(S)` with `q(S) = k_S^T C_SS^-1 k_S`, and adding column `j` raises `q` by `(k_j - C_jS C_SS^-1 k_S)^2 / (C_jj - C_jS C_SS^-1 C_Sj)` (tested against the difference of two `q` values).

Limitation: greedy on this gain is not guaranteed to be near-optimal. With `C = [[1,0,0],[0,1,.99],[0,.99,1]]`, `k = (1.5, 1, -1)` and budget 2, greedy takes {0,1} with q = 3.25 while {1,2} gives q = 200, a 61.5x lower preservation cost. Beam search (width 3) recovers {1,2}. On 25 random 9-column instances beam is never above exhaustive, and a beam as wide as the number of subsets equals exhaustive; narrower beams are not proven to match it.

Robust extension: `min_D max_e tr(D C_e D^T) s.t. D K = V` equals `max_w tr(V (K^T C_w^-1 K)^-1 V^T)` over the simplex (the cost is convex in D and linear in the mixture), with the closed-form edit as the inner solution. It is implemented by exponentiated-gradient ascent and checked against a 2001-point grid over the two-environment mixture (relative gap below 5e-3).

## 2. Policy Surgery reproduction (upstream 46f95e6; separate from our experiments)

See `receipts/POLICY_SURGERY_REPRODUCTION.md`. Offline `verify` and `reproduce` pass after two Windows environment fixes (LF checkout; one path-mapping patch). The base actor hash matches and 483 of 486 candidates rebuild bit-exactly. The Box2D simulator is not bit-reproducible here (0 of 381 replays identical, 63 differ in a discrete outcome, 5 in goal success), so every arm below is compared on this machine only. Reversed thrusters is an exact rank-one update of the augmented output layer (130 nonzero entries). The four accepted crosswind and weak-main policies have per-layer delta rank 1-3, no layer-2 change, and bf16 storage adds at most 1.3% relative error to their deltas.

## 3. Causal localization and analytic edits (`phase_c.py`)

Development seeds 50000-50063, selection seeds 60000-60063; approval and final seeds untouched; 13,413 simulator episodes, 60 s.

Observations:
- Counterfactual action-window patching (force one action for 8 steps, 5,828 replays, deterministic per seed) finds a return-improving window in 60 of the 64 failing crosswind development episodes; 29 patched episodes reach the landing goal.
- Edits built from the 65-d output-layer activation (`h2`, bias-augmented) at those windows, with `C` from nominal-dynamics rollouts, show **no task gain beyond noise** on the selection seeds. The base crosswind goal rate is about 3 of 64; the best arms reach 5 of 64, and the single best value (0.077) is a random-support arm. Larger `m` is worse (m = 4 and 8 give 0 goals and large nominal damage). Norm-matched random directions are destructive (cost 4.7-816, nominal goal 0-0.83).
- Preservation cost predicts damage before any edited evaluation. The Spearman correlation between `tr(D C D^T)` and the nominal action-flip rate on held-out nominal states is 0.96 over 57 edits (0.94 without random arms, 0.85 over the 12 analytic edits). Against nominal goal loss it is 0.76 (0.70 without random arms, 0.81 over analytic edits). The cost orders edits by harm; it does not make a harmless edit useful.
- Whitening by nominal `C` reduces nominal action flips relative to the `C = I` outer-product (gradient-like) edit at m >= 2 (0.30 vs 0.68 at m = 2). At m = 1 the `C = I` edit kept nominal goal at 0.97 with crosswind 0.06 (noise level).
- Greedy, beam and random column supports (s = 4, 8, 16; m = 1) did not separate on task outcome; greedy and beam had slightly lower predicted cost than random support at s <= 8.

## 4. Robust edit and quantization (`phase_d.py`)

- The minimax edit over nominal and crosswind covariances lowers the worst-case preservation cost by 66% at m = 1 (0.669 to 0.230) and 58% at m = 2 (2.31 to 0.978) relative to the nominal-only edit, at the price of higher nominal cost (0.128 to 0.212; 0.518 to 0.707). Crosswind goal rates are indistinguishable from noise (about 5/64 vs 5/64 at m = 1; 4/64 vs 0/64 at m = 2).
- bf16 merge versus factorized, actor output layer (65 or 130 targeted entries, all of which change in bf16): the relative error of the realized correction at the keys is 0.13-1.2% when merged into bf16 weights, about 0 with fp32 factors, and 0.04-1.1% with bf16 factors.
- Synthetic 1024 x 3584 layer with a 64-column support: 59,092 of 65,536 targeted entries change in a bf16 merge, merged-bf16 relative error is 1.7%, fp32 factors 5e-8, stored factor parameters 4,608. This is an analog of the workorder smoke run, not a reproduction of its random draws (it reported 23,784 changed entries and 0.00537 maximal error).

## 5. Answer to the main hypothesis, so far

Covariance geometry predicts the cost of a repair (damage to nominal behavior) before the first edited evaluation, with rank correlation 0.7-0.96. In this setting it did not predict a useful repair: causal localization found real corrective action windows, but a rank-1 to rank-8 linear output-layer edit built from them did not generalize to new seeds. A candidate failure mechanism (hypothesis, not isolated): the corrections are state-specific control decisions, so a global linear edit that flips them at the key also flips decisions at nearby nominal states, and the `h2` activations of failure and nominal states overlap heavily. Selection sets are small (64 episodes) and platform-specific.

## 6. Next experiment

1. Rerun the localization with 256 selection episodes and paired intervals.
2. Localize at layer 1 (input affine edits are what the accepted policies use) instead of the output layer.
3. Use margin-aware keys: only states whose logit margin is below a threshold, so the edit targets decision boundaries.
4. Then Phase E: recurrent unstable-mode edit, reproduction of the published Qwen3.5-0.8B edit, and only then the Qwen3.5-9B `mlp.down_proj` plan. No GPU time has been spent.
