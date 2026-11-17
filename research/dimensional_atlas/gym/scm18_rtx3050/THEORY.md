# Causal subspaces across recurrent visits

Let a tied block update `h[v+1] = F_theta(h[v])`. Sharing `theta` constrains the
parameters; it does not force the tangent map at different visited states to
be identical. For calibration-selected intervention directions `D` and input
kick `D u`, the exploratory Jacobian is

\[
J_v = \partial h_v / \partial u.
\]

The campaign uses symmetric differences at two registered step sizes. Its
columns are derivatives with respect to two/four selected coefficients, so
their measured rank is **at most two/four by construction**. A near-four
effective rank does not estimate the full hidden-state dimension. Principal
angles compare retained tangent ranges in the same raw residual coordinates.
Effective rank here is `exp(entropy(s / sum(s)))`, based on singular values,
not squared singular values. Second differences estimate directional curvature;
step-size and FP32 rounding sensitivity are retained in the receipt.

## Exact linear fibers

SCM-16's deliberately bottlenecked head is `W2 W1 h + b`. For an orthonormal
basis `Q` of `row(W1)`, let `P = Q Q^T`. Then

\[
W_1(I-P)=0,\qquad \Delta h_v=(I-P)U\,\mathrm{diag}(g_v(a_v))V^T h_v.
\]

Thus a final-visit fiber perturbation is invisible to this exact-real linear
head. An earlier perturbation can become visible after later nonlinear block
updates. A measured nonzero transported effect establishes a causal effect
on logits, not the identity of a mathematical intermediate variable. The
joint adapter additionally allows a head-readable sector. Its `U,V`, sparse
feature router and visit gates are shared across visits except for the gate
rows. It is a low-rank residual/parameterized intervention inspired by VPD,
not recovered Goodfire weight components.

## Nonlinear RMP readout

RMP's authentic lens is `embedding.weight @ LayerNorm(h)`. SCM-17 computes
calibration gradients of clean top-logit margins and freezes their retained
row span. Orthogonality gives a local first-order null for those retained
margins. It gives neither an exact global kernel nor silence for unretained
margins. If the Hessian is bounded on the entire perturbation segment, the
remaining change can be bounded at second order. This campaign does not
replace absent genuine RMP checkpoints with random models.

## Semantic mediation and compactification

A source program has three unary operators and a fixed composition topology.
The registered named variable is its first unary operator. Exact recipient/
donor counterfactuals keep source series, topology and the other two operators
fixed. A projected donor-state swap tests whether the representation carries
enough causal information to produce the donor's exact future. Define

\[
F_{causal} = N_{correct\ counterfactuals}/N_{valid\ tests}.
\]

Validity requires exact clean top-one correctness on both programs and distinct
oracle futures. Report the denominator, unconditional attempted-pair success,
full-state swap control, and retained dimension. Small or zero eligible counts
prevent a positive mechanism claim, even if raw intervention success is high.
The present diagnostic uses PCA interchange directions; compression accuracy
is additionally compared for SAE groups/reconstruction and head sectors.

## Certificates

SCM-12's rank-one response-funnel obstruction is conditional on a uniform
remainder bound and continuous factorization through an interval. SCM-13's
affine-disk routine uses exact rational geometry, but still requires exact-real
chart Jacobians and proven bounds over the full disk/circle. Feeding numerical
Jacobians and maximum errors from sixteen circle samples into these routines
tests only the conditional surrogate. A returned conditional braid result
cannot establish neural topology. SCM-09–14 interval and source-Hessian
certificates are the appropriate route to stronger claims. Here every neural
topology conclusion remains inconclusive without those premises.
