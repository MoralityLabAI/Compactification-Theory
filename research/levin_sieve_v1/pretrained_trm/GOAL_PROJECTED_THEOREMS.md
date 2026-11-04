# LSV-09: Goal-projected gradients and the obstruction to state-norm certificates

**Date:** 2026-11-04. **Evidence:** written exact proofs, Fraction-based affine reference/oracle and a separately labelled exploratory first-order diagnostic for the pinned pretrained Sudoku recurrent model. The general goal-derivative inequality is classical analysis and is not claimed as a new general theorem. A *verified nonlinear Jacobian envelope* for the pretrained checkpoint has not yet been established.

## Theorem GP-01: task-projected finite perturbation bound

Let \(S\subseteq\mathbb R^d\) contain a closed axis-aligned box \(B=\{x:\lvert x_i-c_i\rvert\le r_i\}\), and let \(g_{ab}:S\to\mathbb R\) be continuously differentiable. In a recurrent digit classifier, \(g_{ab}(x)\) denotes the difference of the final logits of nominal winner \(a\) and competitor \(b\), after all remaining recurrent passes. Assume we have mathematically justified uniform coordinate derivative bounds \(M_{ab,i}\) on **every point of the entire box**:

\[
\sup_{x\in B}|\partial_i g_{ab}(x)|\le M_{ab,i}.
\]

Then for every \(x\in B\),

\[
\boxed{g_{ab}(x)\ge g_{ab}(c)-\sum_i M_{ab,i}r_i.}
\]

If the right-hand side is positive for all competing classes \(b\ne a\), the final output class is guaranteed to remain \(a\) for all admissible perturbations.

**Proof.** Set \(\delta=x-c\) and integrate \(\frac{d}{dt}g_{ab}(c+t\delta)=\nabla g_{ab}(c+t\delta)\cdot\delta\) from \(t=0\) to 1. The entire segment lies in \(B\), and \(|\partial_i g|\le M_i\), so the integral is bounded below by \(-\sum_i M_i|\delta_i|\ge-\sum_i M_i r_i\). QED.

**Essential condition:** the derivative envelopes must hold on the entire perturbation set. Point gradients from an automatic differentiation run at the clean state are *not* envelopes and do not prove positive robustness radii.

## Theorem GP-02: an exact certificate for affine recurrent systems

Suppose the recurrent map is affine, \(F(x)=Ax+d\), with a frozen output head \(\ell_s(z)=w_s^\top z+b_s\). After \(k\) steps,

\[
\nabla_x (\ell_a-\ell_b)(F^k(x))=(w_a-w_b)^\top A^k
\]

for every initial state. Therefore the exact worst-case reduction in the pairwise final score over the input box is

\[
D_{ab}=\sum_i\left|[(w_a-w_b)^\top A^k]_i\right|r_i.
\]

A sufficient and, for a fixed competitor, necessary condition for strict robust dominance is \(g_{ab}(c)>D_{ab}\). The necessity follows because the minimizing corner \(\delta_i=-r_i\operatorname{sign}(\partial_i g_{ab})\) achieves the displayed decrease. All calculations in [`goal_projected.py`](goal_projected.py) use exact rational arithmetic and independently test small instances against exhaustive box corners. This is a direct specialization of GP-01 and established linear systems theory.

## Exact witness: hidden gain and goal preservation

Let

\[
A=\begin{pmatrix}10&10\\-10&-10\end{pmatrix},\qquad
w_a=(1,1),\quad w_b=(0,0),\quad b_a-b_b=1.
\]

The largest absolute row sum of \(A\) is 20. Coordinatewise hidden error can increase up to 20 times in one step. Nevertheless \((w_a-w_b)^\top A=(0,0)\), and every output gap after one step equals exactly 1. The goal-projective certificate proves robust class preservation for *arbitrary finite hidden-state disturbances*. The proof uses cancellation in the readout direction which any absolute hidden-error bound loses.

The example demonstrates why a Michael-Levin-inspired **goal-preservation quotient** can be a more useful mathematical object than an invariant over every microscopic degree of freedom. It is a deliberately constructed linear example, not a neural model result.

## Exploratory pretrained diagnostic

[`run_goal_gradients.py`](run_goal_gradients.py) uses the same locked 44,001-parameter externally pretrained Sudoku model, stage-one FP32 state, and 64 frozen puzzles as LSV-05 through LSV-08. For each puzzle, it selects the first missing cell from the **public input puzzle**, then uses the top two clean predicted digits as the output-margin pair. Autograd calculates the clean-point gradient through the three remaining recurrent blocks. It reports the full and one-cell-patch gradient L1 norms, tangent radii \(m/\|\nabla g(c)\|_1\), and a diagnostic first-order sign-gradient perturbation at fixed radii. Correct Sudoku answers are used only in evaluation.

**Every calculated tangent radius is observational and uncertified.** A positive GP-01 certificate for the nonlinear checkpoint requires a validated numerical or analytic Jacobian envelope throughout the entire declared input region; this experiment currently measures gradients only at the nominal state and evaluates specific directional attacks. A further follow-up could bound Jacobian variation using Hessian enclosures with outward rounding and choose region size without held-out leakage.

## Research direction

Task-projected derivative bounds might preserve cancellation that vanishes under hidden-state norm bounds. The high-value applied-mathematics question is how to compute these **verified** margins cheaply at model scale, and whether the certificate cost plus any enabled inference savings beats fixed-depth inference on held-out tasks. Possible methods include sound Jacobian interval enclosures, CROWN-style linear relaxations and explicitly bounded second-order remainders.
