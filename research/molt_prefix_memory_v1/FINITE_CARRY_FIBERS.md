# Exact finite carry fibers and dose-constrained repair (MOLT-14)

This handwritten argument supplements [the typed carry quotient](TYPED_CARRY_QUOTIENT.md)
and the [MOLT-14 development experiment](https://github.com/MoralityLabAI/LoopedTransformers/pull/412).
It fixes the reached recipient, physical parameters, future operand word and
complete observer contract. No new population or Lean-certified claim.

Let h be the actual recipient residual and t=h+d its intervened value. For
a LayerNorm with learned scale gamma, learned bias eta and positive epsilon,
the legal logit contrasts take the form

    M(t) = A(t-mean(t)1)/sqrt(||t-mean(t)1||²/n+epsilon) + b,

where the rows of A are the legal unembedding differences multiplied
coordinatewise by gamma, and b is their action on eta.

Let C have rows A, the all-ones row and h^T. Write P for its orthogonal
row-space projector. Reflection t'=2Pt-t gives Ct'=Ct and ||t'||=||t||.
The means agree, so centered norms and LayerNorm denominators agree.
It follows directly that M(t')=M(t). The legal softmax carries agree and,
by the exact typed suffix factorization, every subsequent readout agrees.
Moreover

    ||t'-h||² = ||t'||² - 2 h^T t' + ||h||² = ||t-h||².

The reflection is distinct exactly when (I-P)t is nonzero. It preserves
the actual post-residual physical dose and endpoint norm; it need not
preserve current unrestricted logits or the original intervention's
signed linearization. A more general set of such matches is the sphere
with center Pt and radius ||(I-P)t|| in the nullspace of C. Requiring a
particular native intervention span can eliminate every distinct solution.

For an orthonormal support chart Q, local recovery of a donor margin
difference m is approximated by the constrained problem

    min ||J Q c - m||² subject to ||c||<=R,

where J is the reached-state derivative of M. A full-column-rank JQ
gives uniqueness of the unconstrained least-squares minimizer. It does
not imply exact finite-dose reachability. When Q has 16 columns and
JQ is invertible, the inverse function theorem provides local uniqueness
for a fixed nonlinear carry in that chart, not global injectivity.

The MOLT-14 instrument evaluates the nonlinear objective directly,
using the same finite solver budget and dose cap in native, random
and observer-derived orthonormal charts. A finite exact-carry sham
uses the ambient chart instead. Equal future predictions of that
sham and the target are required by the architecture, not evidence of
causal specificity. All claims concern the typed operation clock,
whose only suffix carry is the legal-state probability vector.
