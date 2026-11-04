# LSV-10: Goal-observable dimensions, intervention closure, and approximate quotient repair

**2026-11-04.** Mathematical basis: classical linear-system observability, invariant rowspaces, switched affine systems, and approximate semiconjugacy. This is a new *application and executable experiment* within the Levin-inspired discovery program, **not a claim of newly invented observability mathematics**. All exact theorems below have written proofs and an exact-rational executable checker; local nonlinear-model derivatives are exploratory measurements, not certified global robustness.

## T1. Minimal finite-horizon goal statistic

Take an `n`-dimensional real state `x`, a nonempty set of fully specified matrices `A_a` for available actions `a`, and a goal readout `y=Cx`. For a finite word `w=(a_1,...,a_k)`, use `A_w=A_(a_k)...A_(a_1)`. Let `O_H` be the vertically stacked rows of `C A_w` for every word of length at most `H`. Define

\[
 V_H=\operatorname{rowspan}(O_H),\qquad d_H=\dim V_H.
\]

**Theorem.** Among **linear** statistics `P x` that determine `C A_w x` for **every** initial state and every allowed word `|w|<=H`, the smallest possible output dimension is exactly `d_H`.

**Proof.** Let `P` be a matrix whose statistic determines the stated readouts, including when its decoder is nonlinear. When `P x=P x'`, the decoder must give identical goal histories; for `v=x-x'` this says `P v=0` implies `C A_w v=0` for all words, or `ker P subset ker O_H`. Linear algebra gives `row(O_H) subset row(P)` and thus `rank P>=d_H`. Conversely select a basis for `V_H` as rows of `P`; then every `C A_w` is a linear combination of rows of `P`, so the statistic suffices and has exactly `d_H` components. QED.

**Interpretation.** `d_H` is an operational *goal-horizon state dimension* for a declared action family and readout; it is measured in **real coordinates**, with no finite-bit memory claim. A different declared goal or an additional intervention may increase it. This formalizes a restricted notion of a growing cognitive light cone, inspired by Levin's multiscale analysis.

## T2. Smallest exact autonomous linear quotient for all horizons

Starting with `V_0=rowspan(C)`, iterate

\[
 V_{k+1}=V_k+\sum_{a} V_k A_a.
\]

This increasing chain stops after at most `n-rank(C)` strict refinements. If `C=0`, it is already constant. Let `P` have basis rows for the stable subspace `V_*`.

**Theorem.** For each action `a`, a unique induced matrix `B_a` exists with `P A_a=B_a P`, and a matrix `D` exists with `C=D P`. Every *surjective linear* exact goal-preserving quotient `Q` for all allowed actions has `rank(Q)>=rank(P)`. Affine shifts `F_a(x)=A_a x+b_a` induce `z'=B_a z+P b_a` without changing the minimal linear dimension.

**Proof.** At stabilization, every row `p` of `P` has `p A_a in V_*`, which uniquely expands in the independent rows of `P`; these coefficients form `B_a`. Because `row(C) subset V_*`, its rows expand as `D P`. If `Q A_a=G_a Q` and `C=H Q`, then repeated substitution yields `C A_w=H G_w Q` for all words. Its rowspace contains every `V_H`, hence `rank(Q)>=rank(P)`. The affine statement follows by multiplying `A_a x+b_a` by `P`. QED.

This is the standard observable quotient construction (Kalman-style observability extended to declared switching actions). It gives a **rank-minimal linear macrostate**, which may be a small fraction of the original latent state.

### Witnesses

- **Shift-register horizon:** A four-coordinate left-shift with readout of the first coordinate gives `d_H=(1,2,3,4,4,...)`. Each additional lookahead exposes one previously hidden coordinate until the four-state system is fully observable.
- **Output cancellation despite hidden gain:** `A=[[10,10],[-10,-10]]`, `C=[1,1]`. Hidden error may grow 20-fold under the ordinary sup norm. Yet `CA=0`, so `d_H=1` for every `H` and the induced macro transition is `B=0`.
- **Intervention-induced expansion:** Under `A0=diag(1/2,100)` with `C=[1,0]`, the second state is irrelevant to the goal, so `d_H=1`. Adding `A1=[[0,1],[0,0]]` makes that state influence the observed coordinate, and `d_1=2`. A repair guarantee must cover the declared perturbation/action interface.

## T3. Nonlinear approximate goal quotient and finite-time repair

Let `F_a:X->X` be nonlinear state updates on a forward-invariant domain, and `pi:X->Z` an explicitly declared goal projection. Suppose every allowed action has an approximate induced macro map `G_a:Z->Z` such that

\[
 \|\pi(F_a(x))-G_a(\pi(x))\|\leq\varepsilon
\]

**uniformly for every** reachable `x`, and all the `G_a` have Lipschitz constant at most `rho<1` in one common norm. For a fixed action sequence applied to both systems, write `z_{t+1}=G_{a_t}(z_t)` and `x_{t+1}=F_{a_t}(x_t)`. Then

\[
 \boxed{\|\pi(x_t)-z_t\|\leq\rho^t\|\pi(x_0)-z_0\|+\varepsilon\frac{1-\rho^t}{1-\rho}.}
\]

**Proof.** For `e_t=||pi(x_t)-z_t||`, triangle inequality yields `e_{t+1} <= epsilon+rho*e_t`. Iterate. QED. This is a familiar approximate-semiconjugacy error-transfer argument.

If a pairwise goal-logit gap `m_ab(z)` is `L_ab`-Lipschitz on the relevant macro domain, and its nominal gap exceeds `L_ab e_t`, its class ordering is preserved at that horizon. All competing pairs must pass to preserve an argmax. The code implements the exact-rational geometric error bound and strict margin check.

**Global condition.** A small measured defect along 64 sampled trajectories is not the required uniform `epsilon` on the entire domain. When transverse state can grow without bound, a tiny coupling from transverse state to the goal may destroy any finite global `epsilon`.

### Nonlinear exact goal quotient example

Let `pi(x)=x1+x2`, `s=pi(x)`, `v=x1-x2`, and define

\[
 F(x_1,x_2)=(s/4+100v^3,\ s/4-100v^3).
\]

Then `pi(F(x))=pi(x)/2` **for every real x**. The hidden transverse coordinate may grow explosively, while the goal sum contracts at rate `1/2`. The checker verifies the identity on rational witnesses. Adding even `0.01 v` to the first coordinate produces a macro defect proportional to the unbounded transverse state and invalidates a global constant-defect guarantee.

## T4. Exploratory goal-horizon tangent diagnostic on the frozen Sudoku checkpoint

Run `run_goal_horizon.py` on the same external 44,001-parameter frozen four-step recurrent Sudoku model, checkpoint SHA256 `3bab6189...47e1`, and the same 64 seeded puzzles as LSV-05 through LSV-09.

A first missing cell is selected from each *public puzzle*, and the final clean prediction determines the top-two logit pair. Freeze that pair. At the stage-one hidden state `h1`, differentiate its pairwise logit gap at remaining depths `k=0,1,2,3`, obtaining the four row gradients

\[
 \nabla_{h_1}g_k(h_1),\quad k=0,1,2,3.
\]

The singular values of the resulting `4 x 3888` matrix define a **local tangent goal-history rank**. A threshold of `1e-3` times the largest singular value is registered before checking the checkpoint. A matching identity-recurrence control gives rank one. Adjacent gradient cosine, coordinate norm, and gradient support by token quantify whether the local goal-relevant directions spread through the network over recurrence depth.

**Important limit:** for the nonlinear model these are point Jacobians at the clean trajectory, and their rank does **not** establish a minimal exact nonlinear quotient over a finite neighborhood. They are observational diagnostics. A nonlinear robust certificate still requires validated bounds over the full perturbation domain, including floating-point rounding if deployment arithmetic is part of the claim.

### Reproduction

```bash
python -m unittest discover -s research/levin_sieve_v1/pretrained_trm -p 'test_*.py' -v
python research/levin_sieve_v1/pretrained_trm/run_goal_horizon.py --fetch --out runs/lsv10/receipt.json
```

The CPU checkpoint runner loads hash-pinned third-party weights in `weights_only=True` mode. No model retraining or weight mutation occurs. The GitHub workflow retains the observed metrics as an artifact.

## Follow-up experiment

Fit low-dimensional *approximately* autonomous goal projections from calibration-only states, then test independent held-out trajectories and adversarial transverse perturbations. A viable result would require a **verified region-wide leakage bound** and retained Sudoku accuracy with compute measured against full recurrence. Training a projection solely for the validation trajectories can manufacture a false compression result; the goal interface and held-out groups must be frozen before selection.
