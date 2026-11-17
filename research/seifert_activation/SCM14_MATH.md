# SCM-14: Source-certified second-order recurrent response and affine-disk topology

**Scientific status:** A mathematical upper bound for the *ideal real-valued* RMP pre-normalized causal decoder, implemented with exact rational inequalities and source-validated clean-state center balls. The CPU floating/BLAS interval verifier has the previously declared arithmetic contract. The result is measured on synthetic frozen weights. It is not a certified physical FP32/CUDA execution, a trained activation result, or a 4D Seifert-surface invariant.

## The second-order source question

At a fixed injection visit, an intervention is indexed by a vector `u` in the closed unit disk `D²`. The source patch is `h_j <- h_j + amplitude * A u`, with an exact-real constant `amplitude`, frozen axes `A`, and named observed token projections `z_i(u)=C^T h_i(u)`. Define the first- and second-order spectral bounds

    J_i = sup_{u in D², ||v||<=1} ||D h_i(u)[v]||_2,
    H_i = sup_{u in D², ||v||,||w||<=1} ||D² h_i(u)[v,w]||_2.

The main output is a source-derived `H_i` for every visit and observed token. It is a bound on the **full closed disk**, not a finite-angle estimate. Because the patch is affine in `u`, the entry state has `H_j=0` and `J_j<= amplitude * ||A||_2`. Every causal prefix before token `j` retains `J_i=H_i=0` under the exact-real architecture.

## LayerNorm curvature

For `a=Px`, with `P=I-11ᵀ/d`, `s²=eps+||a||²/d`, and `LN(x)=Gamma*a/s+beta`, the exact derivatives for directions `v,w`, with `u=Pv`, `v'=Pw`, are

    D LN(x)[v] = Gamma*(u/s - a <a,u>/(d*s³))

and

    D² LN(x)[v,w] = Gamma*(-u<a,v'>/(d*s³)
                           -v'<a,u>/(d*s³)
                           -a<u,v'>/(d*s³)
                           +3*a<a,u><a,v'>/(d²*s⁵)).

Consequently,

    ||D LN|| <= ||Gamma||inf/s,
    ||D² LN|| <= ||Gamma||inf*(3A/(d*s³)+3A³/(d²*s⁵)), A=||a||.

The global bound `||D² LN|| <= 4 ||Gamma||inf/(sqrt(3d)*eps)` follows from `A²<=d*s²` and maximizing `A/(eps+A²/d)^(3/2)`. The checker takes the minimum of a validated source-local bound and the global one. A center `c` with proven state error radius `R` yields `Alo=max(0,||Pc||-R)` and `Ahi=||Pc||+R`, with outward rounding for all comparisons.

## Causal attention curvature

For a probability row `p=softmax(s)`, measured with the source score `L_inf` norm and output `L_1` norm,

    ||D p|| <= ||D s||inf,
    ||D² p|| <= ||D² s||inf + 2||D s||inf².

The first inequality follows from `E|a-Ea|<=sqrt(Var(a))<=||a||inf`. The Hessian formula is the centered product of the two score perturbations minus their covariance, plus the centered score Hessian; Cauchy-Schwarz bounds its `L_1` norm by two score products and one score-Hessian term.

For each attention head and query token `i`, let `Q,K,V` be validated maximum vector norms over the causal prefix, and let `q1,k1,v1,q2,k2,v2` bound their first and second derivatives. With `d_h` the head dimension,

    s1 <= (q1*K+Q*k1)/sqrt(d_h),
    s2 <= (q2*K+2*q1*k1+Q*k2)/sqrt(d_h),
    head1 <= v1 + V*s1,
    head2 <= v2 + 2*v1*s1 + V*(s2+2*s1²).

These bounds use the exact product rule for scores, the probability derivative inequalities, and the convex value-weighted output. A head's certified source-center Q/K/V norms plus `||W||*||D LN||*R` bound the entire disk; the minimum with the global LayerNorm-image cap remains sound. Prefix maxima are cached to avoid quadratic repetitions of rational arithmetic.

The projected multihead result uses `||W_out||_2 * sqrt(sum(head_bound_h²))`. A pre-norm residual block applies the product and chain rules at its middle state, then the second LayerNorm and MLP. For **exact GELU**,

    |GELU'(x)| <= 5/4,  |GELU''(x)| <= 1,

because `GELU''(x)=(2-x²)phi(x)` with standard Gaussian density `phi`. The final output derivative and Hessian bounds are propagated through all recurrent visits, reusing the same weight norms for tied blocks. A fixed FF-U embedding reinjection adds a constant and contributes zero to both derivatives.

## Affine-disk Taylor theorem

By the second-order integral remainder in Banach spaces,

    z_i(u) = z_i(0) + Dz_i(0)u + e_i(u),
    ||e_i(u)|| <= (1/2) ||C||_2 H_i ||u||².

For the unit intervention disk, `eta_i=||C||_2 H_i/2` is a sufficient uniform error. SCM-13 accepts exactly these `eta_i` when its affine anchor and Jacobian are independently justified. For a pair with affine minimum clearance `m_ij`, a collision-free straight homotopy follows if

    m_ij > (||C||_2/2)(H_i+H_j) + margin.

Where the affine pair has full rank and all its collision preimages are outside the disk, SCM-13 then proves the **entire pure braid is null**. In contrapositive form, if a nontrivial observed braid disagrees with that affine null prediction, at least one pair must violate the inequality, giving the curvature lower-bound obligation

    H_i + H_j >= 2*m_ij / ||C||_2.

This is a conditional mathematical implication, rather than an empirical bound inferred from a small set of angles. The usual critical radius calculation remains available after certified affine anchors and Jacobians are supplied.

## Source-level trust boundary

- Checkpoint and synthetic weights are binary floating coefficients interpreted as exact reals; spectral norms come from exact integer/Gram estimates.
- Clean-state centers and their source error radii come from SCM-10's validated interval/ball executor, whose CPU binary64/BLAS model is a declared execution assumption.
- The 2D patch amplitude is fixed before varying `u`. In the registered RMP experiment it is the *clean* token's RMS multiplied by a fixed fractional radius, and the RMS upper bound must itself be source-verified.
- SCM-14 supplies a proved **Hessian upper bound** under that contract. It does **not yet** produce a source-verified exact-real `z_i(0)` or `Dz_i(0)` serialization for trained models. A float64 autodiff Jacobian alone remains an exploratory estimate.
- `scm14_bridge.py` therefore labels every affine conclusion conditional on the missing anchor/Jacobian proof. The program refuses unvalidated numeric Hessian fields and retains explicit `trained_rmp_model_measured=false` and `float32_cuda_execution_certified=false` markers.

## Mathematical experiment scope

The full-shape deterministic synthetic source has 64 tokens, hidden width 128, 4 heads, 8 tied visits, beta 0.1, injection after visit zero at token 4, and readouts at tokens 16, 32, 48. It uses a frozen dyadic frequency table for reproducibility and an amplitude derived from the source token's certified RMS. At this tiny radius, source-local Q/K/V bounds keep the unit-disk Taylor remainder below 0.032 at visit eight; the global-image bound alone grows to approximately 3.56e49 and is inconclusive. Both are valid conservative upper bounds. Neither numerical result implies a trained-model topology effect.

The next mathematical milestone is an exact-real, source-certified Jacobian and projected anchor exporter to pair with these Hessian bounds. Independent physical floating-point error bounds would then be needed to lift ideal-real conclusions to actual activation captures.
