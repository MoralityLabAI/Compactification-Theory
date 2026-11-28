# MOLT-16: finite-dose decision feasibility with LayerNorm bias

Companion to [finite carry fibers](FINITE_CARRY_FIBERS.md) and
[LoopedTransformers #414](https://github.com/MoralityLabAI/LoopedTransformers/pull/414).
Handwritten real-arithmetic argument, no machine-checked theorem.

Fix a recipient h in R^n, orthonormal intervention chart Q, and radius R.
Let C be the centering projection. For each legal rival j to target class t,
write a_j=(W_t-W_j) elementwise multiplied by the final LayerNorm scale,
b_j=(W_t-W_j) applied to its learned bias. Put z=Ch, B=CQ and

    s(c)=sqrt(||z+B c||²/n+eps).

A nonnegative target-vs-rival logit contrast is equivalent to

    a_j^T B c >= -b_j s(c) - a_j^T z.

Orthogonally split z into its component z_parallel in range B and z_perp.
For all ||c||<=R, the triangle and reverse triangle inequalities give

    l = sqrt((||z_perp||²+max(0,||z_parallel||-R||B||_2)²)/n+eps)
    u = sqrt((||z_perp||²+(||z_parallel||+R||B||_2)²)/n+eps).

Then l<=s(c)<=u. Taking the minimum of -b_j l and -b_j u gives a
necessary affine constraint; taking the maximum gives a sufficient
constraint. Both retain the same radius ball. If the sufficient system
has a feasible point, the original nonlinear winner inequalities hold.
If the necessary system has no feasible point, the nonlinear system
has none. The gap between the two systems need not vanish.

For the necessary system A c>=v, any nonnegative lambda yields

    lambda^T v <= lambda^T A c <= R ||A^T lambda||.

The first inequality is componentwise nonnegative weighting, the
second Cauchy-Schwarz. Strict separation is a sufficient obstruction,
even if lambda came from an incomplete or inaccurate optimization,
provided the actual inequalities are established. The numerical
implementation supplies only conditional real-arithmetic witnesses:
singular values, projection and dual expressions lack directed rounding.
A verified strict actual winning margin is a primal witness. An
unsuccessful search supplies neither an obstruction nor a general
claim about optimality. Any numerical contradiction aborts the run.

The relaxation depends on R itself. A ratio lambda^T v/||A^T lambda||
bounds the minimum norm for that fixed affine relaxation, and should
not be extrapolated to an arbitrary new dose whose envelope differs.
The target is the immediate legal cut winner. It is not a condition on
the complete future decision trajectory; soft carried probabilities
with the same argmax can lead to different suffix outputs.

This supplies a budget-specific classification of repair outcomes
inside declared charts. It does not establish causal specificity of
BPD, an autonomous teacher improvement, or a hard logical certificate.
