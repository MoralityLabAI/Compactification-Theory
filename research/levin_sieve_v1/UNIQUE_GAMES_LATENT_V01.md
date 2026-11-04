# UG-LAT-01: quadratic latent observable as a nonlinear quotient

**Date:** 2026-11-04. **Status:** exact algebraic local block and exhaustive
small-field CPU checks. This work does not reconstruct the paper's
full recursive latent alphabet gadget or prove its large-rank
detection theorem.

## Pinned construction

Use the irreducible polynomial encodings of F2, F4, F8 (also F16
available). On X=B=F³ define

    Q(x1,x2,x3) = (x2*x3, x1*x3, x1*x2)
    C_blk(x,y) = y + Q(x)
    D(x,a) = Q(x+a)+Q(x)+Q(a).

For every nonzero v, and a=t*v,

    v dot D(x,a) = 0.

Set A=Fv, D_A={w:v dot w=0}, and
U_A={(a,Q(a)+w):a∈A,w∈D_A}. These are binary-linear
subspaces of dimension 3d. Along any such update,

    C_blk((x,y)+(a,Q(a)+w)) - C_blk(x,y) = w + D(x,a) ∈ D_A.

All plus/minus operations are the same in characteristic two.
For uniform nonzero updates from U_A and independent uniform aggregate,
the **one-block** change probability is

    Pr[C_blk(z+ζ) != C_blk(z)] = q(q+1)/(q²+q+1)
                               = 1 - 1/(q²+q+1).

The standalone oracle computes this probability by enumerating actual
updates and comparing outputs, independent of the closed-form expression.
Tests also check field identities, line-count q²+q+1, hyperplane size q²,
update-space closure, line additivity and polarization orthogonality.

## Connection to the recursive theorem

OpenAI's manuscript combines many copies, random linear isomorphisms,
and a recursive descent. Its theorem gives the output-change law

    p_t = (1-q^-3)(1-theta)^t, theta=1/(q²+q+1),

and separately proves that sufficiently high-rank linear families
detect the chosen noise with probability >=1/8. The function
`recursive_change_formula` checks only the displayed formula.
A full implementation would require the recursion, the lift
factorization, and a rank-survival audit with preselected parameters.
Nothing here asserts that the one-block construction already has
low nonlinear change probability.

The intended Compactification comparison is with LSV-15's exact
LayerNorm shift quotient: LSV-15 is invariant under common shifts;
this quadratic map is equivariant under output translations, and its
noise stability emerges through recursive composition.

## Reproduce

    python3 -m unittest discover -s research/levin_sieve_v1 -p 'test_unique_games_latent_v01.py' -v

No pretrained model, Goodfire credits, or cluster GPU time is needed.
Future neural tests should measure intervention success, output
stability, linear-probe detection, and measured activation rank on
the *same* data and distinguish them from mathematical guarantees.

Sources:
- https://github.com/openai/math/blob/main/preprints/The-Unique-Games-Theorem-September-23-2026/build/sections/02-latent.tex
- https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/levin_sieve_v1/pretrained_trm/GAUGE_QUOTIENT.md
