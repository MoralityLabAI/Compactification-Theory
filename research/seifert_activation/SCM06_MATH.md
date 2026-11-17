# SCM-06: A global derivative certificate for the RMP recurrence

**Status, 2026-11-13.** Exact stored-weight norm calculations, a real-arithmetic theorem about RMP's *pinned* decoder, a deterministic synthetic CPU reference, and executable tests. Actual `LOOP-T`/`FF-U` checkpoint artifacts are held outside this repository. The new certificate has **not** been applied to trained activations. Floating-point execution, `cos/sin` numerical errors, and continuous-time interpolation of machine outputs are separate proof obligations. Nothing here establishes smooth four-dimensional Seifert-surface isotopy.

## Model class and source contract

The pinned implementation is `MoralityLabAI/LoopedTransformers`, branch `rmp-v0-phase2`, with `research_gym/neural/looped_decoder.py` `ExplicitDecoderBlock`, its `ScheduleExecutor`, `research_gym/interp/models.py`, and `research_gym/interp/analysis.py`. Source blobs are checked before the standalone SCM-06 auditor loads a model. The RMP block uses one causal multihead attention layer after per-token pre-LayerNorm, followed by another pre-LayerNorm and an `Linear -> GELU(exact erf) -> Linear` MLP. Alpha and beta residual factors appear *twice*:

\[
 A(X)=\alpha X+\beta\operatorname{Att}(\operatorname{LN}_1(X)), \qquad
 B(X)=\alpha A(X)+\beta\operatorname{MLP}(\operatorname{LN}_2(A(X))).
\]

RMP's `AnalysisModel` optionally adds the **fixed embedding** after each visit, before the SCM cyclic injection. Adding a theta-independent tensor changes no derivative bound. Every bound is valid only for models in evaluation mode, this explicit architectural graph, and the exact stored coefficients interpreted as real numbers.

## Lemma 1: global LayerNorm bound

For one token of dimension `d`, let `P = I - 11^T/d` and `s(x)=sqrt(||Px||_2^2/d + epsilon)`, where `epsilon>0`. Ignoring learned affine parameters, `LN_0(x)=Px/s(x)` has Jacobian

\[
 D\operatorname{LN}_0(x) =\frac P{s(x)}-\frac{(Px)(Px)^\top}{d\,s(x)^3}.
\]

On the mean-zero subspace orthogonal to `Px` the eigenvalue is `1/s(x)`; on its radial direction it is `epsilon/s(x)^3`. Thus `||D LN_0||_2 <= 1/s(x) <= 1/sqrt(epsilon)`. For `LN(x)=gamma \odot LN_0(x) + bias`, the global Lipschitz factor and image norm bound are

\[
 \boxed{\ell_{\rm LN}=\frac{||\gamma||_\infty}{\sqrt{\epsilon}}}, \qquad
 \boxed{M_{\rm LN}=\sqrt d\,||\gamma||_\infty+||bias||_2}.
\]

These bounds hold for *every real input*, including zero-variance residual states. They exploit the strictly positive LayerNorm epsilon. They do not depend on an empirically estimated minimum variance.

## Lemma 2: attention is globally Lipschitz after LayerNorm

Let `T` be sequence length, `H` the number of heads and `d_h` the head dimension. For head `h`, let `q_h`, `k_h`, `v_h` be proven spectral upper bounds on the three linear projection matrices and `o` the output matrix bound. After LayerNorm, every token's feature vector has Euclidean norm at most `M=M_LN1`. Consequently the row-stacked key and value matrices satisfy

\[
 ||K_h||_2 \le \sqrt T\,M k_h,\qquad
 ||V_h||_2 \le \sqrt T\,M v_h.
\]

The spectral norm of the softmax Jacobian `diag(p)-pp^T` is at most `1/2` because its absolute row sum is `2p_i(1-p_i) <= 1/2`. A row-stochastic causal attention matrix satisfies `||P_h||_2 <= ||P_h||_F <= sqrt(T)`. Apply the product rule to `P_h V_h` and `Q_h K_h^T / sqrt(d_h)`; the `1/2` factor cancels the two Q/K derivative terms. For Frobenius perturbations, the bound is

\[
 \|D (P_h V_h)\| \le
  \left(\sqrt T\,v_h + \frac{T M^2}{\sqrt{d_h}}q_h k_h v_h\right)\ell_{\rm LN1}.
\]

Concatenating head outputs and applying the output projection gives

\[
 \boxed{
 L_{\rm att} = o\,\ell_{\rm LN1}\,
 \left[\sum_{h=1}^{H}\left(\sqrt T\,v_h+\frac{T M^2}{\sqrt{d_h}}q_h k_h v_h\right)^2\right]^{1/2}.}
\]

The formula bounds the entire attention operator in the Frobenius norm. It already includes derivatives through the Q and K matrices; no token-independent attention approximation is used. The causal mask only removes options from a softmax row and leaves these inequalities valid.

## Lemma 3: GELU and recurrent block

The exact erf GELU is `f(x)=x Phi(x)` with `f'(x)=Phi(x)+x phi(x)`. Since `0<=Phi<=1` and `sup_x |x phi(x)| = 1/sqrt(2 pi e)<1/4`, we have `|f'| <= 5/4`. Therefore for the RMP MLP,

\[
 L_{\rm mlp}\le\frac54 \|W_{down}\|_2\|W_{up}\|_2\ell_{\rm LN2},
 \qquad
 \boxed{L_{\rm block}\le(|\alpha|+|\beta| L_{\rm att})(|\alpha|+|\beta| L_{\rm mlp})}.
\]

For a tied `LOOP-T`, reuse the *same* block factor at each later visit. For `FF-U`, calculate a distinct factor for each physical module. The source checks refuse approximate GELU, non-pre-norm blocks, QKV bias, unknown dimensions, or unrecognized model structure.

## Theorem 1: model-derived angular Lipschitz bound

The cyclic injection after visit `v0` at token `j0` is

\[
 X_{v0}(\theta)=X_{v0}^{clean}+a e_{j0}\big(u\cos\theta+v\sin\theta\big)^T,
 \quad a=r\,\mathrm{RMS}(X_{v0}^{clean}[j0]),\quad A=[u\ v].
\]

For a frozen two-dimensional chart `C` and distinct observed tokens `i,j`, the relative projected curve is `q_ij(theta)=(X_i(theta)-X_j(theta)) C`. The injection derivative has Frobenius norm at most `a ||A||_2`. A difference of two token rows has linear operator norm `sqrt(2)` on the full residual matrix. Repeated application of the preceding lemmas yields, for every angle and each subsequent visit `v`,

\[
 \boxed{\|q'_{ij}(\theta)\|_2\le
  \sqrt2\,a\,\|C\|_2\|A\|_2
  \prod_{k=v0+1}^{v}L_{B_k}.}
\]

At visit `v0`, observed tokens excluding `j0` have zero derivative. **Causal-prefix theorem:** for any visit and pair with both observed token positions strictly below the injected position, the derivative vanishes exactly. Every operation is either token-local or attends only to prior tokens; its computation is therefore independent of later-token perturbations by induction over visits.

SCM-05 may consume the uniform upper bound, with exact interval length `2*pi/N < 44/(7*N)`. The resulting certificate is a theorem about the **real-valued model**. It remains conservative on the real model because all norms are global.

## Theorem 2: a small-radius null-braid disk

For any vector `s in R^2` with `||s|| <=1`, replace the unit-circle intervention by `a e_j (A s)^T`. This extends the boundary family over the entire intervention disk. The previous Lipschitz product also bounds displacement of each projected pair from its clean center by `D_ij,v`. Let `m_ij,v^clean` be the Euclidean distance between the two clean projected token positions. If

\[
 \boxed{m_{ij,v}^{clean}>D_{ij,v}\qquad\text{for all observed pairs,}}
\]

then the disk map avoids every projected collision, and its boundary **pure braid is null-homotopic** in the ordered configuration space. The CPU code checks this inequality with rational distance lower bounds **conditional on the recorded clean projections being exact or enclosed by a verified error interval**. It permits a direct falsifier: a reported nontrivial braid where the verified disk inequality holds would imply an inconsistency in the chart/injection, architecture, execution model, or observation.

## Exact arithmetic used by the instrument

For a stored `float32`/`float64` coefficient `w`, obtain its exact binary rational `w = n / 2^e`. Scale each weight matrix to a common dyadic denominator `2^E`. All row sums, column sums and squared Frobenius sums then become *integers*. Compute

\[
 \|W\|_2\le\min\left(\|W\|_F,\sqrt{\|W\|_1\|W\|_\infty}\right)
\]

using integer square roots rounded upward. Rational square roots in other inequalities are also rounded upward on an explicit decimal grid. This is conservative for the exact binary weights and avoids trusting `torch.linalg.svd` as a proof. A `float32` SVD itself sometimes rounds *above* the mathematically valid exact upper bound in tests; comparing to double-precision SVD removes that numerical artifact.

## What this result can and cannot certify

- **It proves a derivative bound for the smooth real-valued RMP mathematical graph**, provided the locked source and checkpoint are used. An audit containing derived factors is a stronger foundation than SCM-05's self-reported external gradient estimates.
- **It does not prove exact real-valued activation samples from the stored floating-point samples.** The geometry layer treats captured coordinates as exact rational samples, although they were produced by a rounded PyTorch forward. An independent forward-error envelope must enclose the real sample positions before a positive topology claim can be promoted from conditional to certified.
- **It does not prove continuity of the actual finite-precision PyTorch computation.** Rounding and transcendental implementation errors can be discontinuous functions of theta. Machine-level proof requires a forward-error envelope or sound interval arithmetic over the chosen checkpoint, plus explicit endpoint/injection-angle treatment.
- **It does not demonstrate any nontrivial neural braid.** Real trained RMP artifacts are unavailable to this execution host. For wide blocks and many visits, the global bound is often far too loose to establish positive later-visit isotopy results.
- **It is chart-relative.** Milnor triple linking and other link-level observables can be preserved if the finite-sampling conditions pass for the real model. Miller's 4D embedded-surface non-isotopy requires a separate surface construction and invariant.
