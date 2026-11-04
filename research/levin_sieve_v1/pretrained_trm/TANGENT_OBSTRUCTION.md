# LSV-11: A differentiable compactification obstruction from goal-history rank

**2026-11-04.** Written theorem and exact-rational mathematical witnesses, with a numerical application to the pinned pretrained recurrent Sudoku model. This is the chain rule and the classical observability-rank obstruction applied to nonlinear goal-history readouts. It is **not** an original general mathematical theorem, a certified floating-point singular-value bound, or evidence that a biological agent has a particular dimension of mind.

## Theorem TO-01: lower bound on smooth sufficient-state dimension

Let `U subset R^n` be open, and let `y=(y_0,...,y_H):U->R^m` be continuously differentiable, where its components are the goal observations of a controlled dynamical system under a declared action schedule. Suppose a continuously differentiable statistic `pi:U->R^d` and a continuously differentiable decoder `h` satisfy

\[
 y(x)=h(\pi(x)) \qquad\text{for every }x\in U.
\]

Then for every `x in U`,

\[
 \boxed{\operatorname{rank} Dy(x) \le \operatorname{rank}D\pi(x) \le d.}
\]

**Proof.** Apply the chain rule `Dy(x) = Dh(pi(x)) Dpi(x)`. The rank of a matrix product cannot exceed the rank of either factor. QED.

Thus a goal-history gradient matrix with rank `r` at **even one state** obstructs all differentiable exact goal-complete representations with fewer than `r` coordinates on any open neighborhood of that state. The observation horizon, readout, action schedule, and differentiability requirements are part of the theorem.

For a scalar goal observation per depth, the rows are `grad_x y_k(x)` for each `k`. This theorem applies to nonlinear networks even though their Jacobian changes along the trajectory. A local rank measurement is therefore conceptually useful without claiming the nonlinear model admits a global linear quotient.

A weaker class of encoders (discontinuous encodings, nondifferentiable readouts, finite-roster lookup) is outside the theorem. An approximate factorization requires its own quantitative error and regularity analysis.

## Exact polynomial witness

Let `x=(x1,x2,x3)` and choose the goals

\[
 y_0(x)=x_1,\qquad y_1(x)=x_1+x_2^2,\qquad
 y_2(x)=x_1+x_2^2+x_3.
\]

At `x=(0,1,0)`, the goal-history derivative is

\[
 Dy=\begin{pmatrix}1&0&0\\1&2&0\\1&2&1\end{pmatrix},
 \qquad\det Dy=2.
\]

Any exact `C^1` factorization through a goal macrostate must therefore have at least three real coordinates in a neighborhood of that point. `test_tangent_obstruction.py` checks this and a two-coordinate factorization by exact rational row reduction, then checks that the runtime SVD diagnostic distinguishes rank-one and rank-four controls.

## Application to the pretrained TRM-inspired Sudoku checkpoint

[LSV-10](GOAL_HORIZON.md) fixed the first missing cell from the public puzzle and its top-two final-predicted digit pair, then computed the gradients of that **same pairwise margin** at four successive latent stages, all differentiated with respect to the stage-one hidden state. The resulting matrix has shape `4 x 3888` per puzzle. A threshold of `1e-3` relative to the largest singular value was chosen before the external checkpoint evaluation.

The successful [GitHub Actions run 41812470503](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41812470503) reported:

- `64/64` puzzles had **numerical tangent-history rank four** at the registered relative singular-value threshold.
- The smallest-to-largest singular-value ratio ranged from `0.03010108` to `0.11851834`, median `0.06544428`; this is well above the `1e-3` reporting threshold.
- Median adjacent gradient cosine was `0.86504771`; the margin directions change across recurrent depth.
- The spatial gradient footprint (tokens above 1% of the maximum token gradient energy within the stage) had median counts `1, 41, 22, 28` across the four observed stages.
- The final goal-class prediction for the first missing cell was correct on `60/64` puzzles. This is a score for the selected cell, not the whole Sudoku grid.

**Conditional interpretation.** If those floating-point derivatives are accurate enough to establish the true Jacobian's rank, TO-01 rules out a *differentiable exact 1–3D goal-history statistic* around each tested stage-one state for this particular readout and fixed horizon. The evidence currently remains a **numerical rank diagnostic**, and its practical validity depends on a rigorous error bound for the computed singular values and on the mathematical model of the recurrent operators. Random-noise or finite-difference tests alone would be weaker than an error-certified smallest singular value.

The rank four is a lower bound on the representation needed to reproduce the **four scalar margin observations**. It gives no lower bound for reconstructing every Sudoku digit at every stage, and it does not measure the dimension of an agent's cognition. It also does not imply that four coordinates suffice in a neighborhood: that is a separate upper-bound construction problem.

## Next gate: numerical singular-value certification

At each pinned input, use validated interval arithmetic or a rigorous floating-point backward error analysis to bound the error in each computed gradient row, including PyTorch operator error. If `J_hat` is the observed 4-by-3888 goal-history Jacobian and `||J-J_hat||_2 <= delta_J` is **verified**, Weyl's inequality gives

\[
 \sigma_4(J)\ge \sigma_4(\widehat J)-\delta_J.
\]

A strictly positive right-hand side would upgrade the local rank-four claim to a certified analytic obstruction. Record full checkpoint, roster and runtime hashes, with calibration held apart from the final audit. For a **useful** model compression, construct a small approximate quotient and compare its forecast errors and total compute with a fixed-depth baseline; a tangent obstruction alone cannot establish an economic gain.
