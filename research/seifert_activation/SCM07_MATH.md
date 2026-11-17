# SCM-07: triangular attention sensitivities and interval-local LayerNorm

**Scope.** Proved inequalities for the real-valued `ExplicitDecoderBlock` graph in pinned RMP source `rmp-v0-phase2`, with stored floating-point weights interpreted as exact dyadic rational numbers. No trained weights, externally certified clean-state enclosures, or real-activation roundoff intervals are present in this repository. Numerical Jacobian tests are diagnostics, not proofs. Pure braid topology is always relative to a selected 2D readout chart; no 4D Seifert-surface equivalence is asserted.

## 1. Why the SCM-06 global product is coarse

The previous bound on an entire `T × d` state replaced the causal attention triangle by a dense operator norm. It also used `||gamma||_inf/sqrt(epsilon)` for both LayerNorms at every visit. These choices are sound but needlessly multiply worst-case effects across tokens that cannot influence one another. A token-local recurrence propagates a *vector* of radii instead of one scalar.

Let `X ∈ R^{T×d}` and `Xbar` be the clean state entering a block. For every token `i` suppose `||X_i - Xbar_i|| ≤ r_i`. The same `r_i` can bound a selected directional derivative `||dX_i||` along a differentiable input path. All bounds below are for the Euclidean row norm, and preserve a zero if all relevant earlier input radii are zero.

## 2. Exact interval-local LayerNorm estimate

With `P = I - 11ᵀ/d`, `s(x)^2 = ||Px||²/d + epsilon`, and affine weights `gamma,bias`, LayerNorm's Jacobian norm obeys `||D LN(x)||₂ ≤ ||gamma||∞/s(x)`. Suppose an encoded, exact-dyadic center `c` differs from the true clean center by at most `e` and the perturbed token stays within `r` of that true center. Then

`||Px||₂ ≥ max(0, ||Pc||₂ − r − e)`.

It follows that

\[
  \boxed{\ell_{\rm LN}(c,r,e)\le
   \frac{\|\gamma\|_\infty}{\sqrt{\epsilon+
      \big[\max(0,\|Pc\|_2-r-e)\big]^2/d}}.}
\]

`local_jacobian.centered_norm_sq` computes `||Pc||²` as an exact rational over stored binary-float coordinates; lower square roots and upper reciprocals are rounded conservatively. The program takes the minimum with the global `SCM-06` norm. A center path captured in ordinary PyTorch with an asserted `e=0` does not itself establish a correct enclosure of the ideal real center. Such a local output is explicitly `conditional_on_external_ideal_center_enclosures`.

## 3. A causal attention-row theorem

RMP computes queries, keys and values from `Y=LN1(X)`. Fix output token `i`. Its attention is supported only at `j≤i`. For head `h`, write the exact stored-weight spectral *upper* bounds `q_h,k_h,v_h` for the corresponding projection matrices. Let `M` be the global Euclidean bound on any row of `Y`, as in SCM-06, and `d_h` the head dimension. Define

\[
 u_j = \ell_{1,j} r_j,\qquad U_i = \Big(\sum_{j=0}^i u_j^2\Big)^{1/2}.
\]

For `Q_i=W_qY_i`, `K≤i=(W_kY_j)_{j≤i}`, `V≤i=(W_vY_j)_{j≤i}`, the product rule gives

\[
 d(P_iV)=dP_i V + P_i\,dV,
 \quad P_i=\operatorname{softmax}(Q_iK_{\le i}^{\mathsf T}/\sqrt{d_h}).
\]

A probability row has `||P_i||₂≤1`, hence `||P_i dV||₂≤v_h U_i`. The softmax Jacobian has `||J||₂≤1/2`. Furthermore, `||K≤i||_F≤√(i+1) k_h M`, `||V≤i||_F≤√(i+1) v_h M`, `||dK≤i||_F≤ k_h U_i`, `||Q_i||₂≤q_h M`, and `||dQ_i||₂≤q_h u_i`. Combining them yields

\[
 \boxed{\|d\operatorname{Head}_{h,i}\|_2\le
 v_h U_i+
 \frac{q_h k_h v_h M^2}{2\sqrt{d_h}}
 \big((i+1)u_i+\sqrt{i+1}\,U_i\big).}
 \tag{SCM-07.1}
\]

This inequality also bounds finite differences along any segment contained in the assumed tokenwise LayerNorm tubes. To obtain the full attention bound, take the Euclidean sum of squared head bounds and multiply by the spectral upper bound for the output matrix. Denote the resulting bound `a_i`.

RMP's double residual factors and exact erf GELU give the tokenwise middle and output bounds

\[
 m_i = |\alpha|r_i+|\beta|a_i,\qquad
 r_i^{\rm next}\le \left(|\alpha|+|\beta|\tfrac54
       \|W_{\rm down}\|\|W_{\rm up}\|\ell_{2,i}\right)m_i.
 \tag{SCM-07.2}
\]

Here `ell_2,i` may use the interval-local LN bound with the clean *middle-state* center and radius `m_i`; if that center lacks a certified enclosure, the source-global LN bound is substituted. The code validates the block graph by calling the SCM-06 strict architecture checker and uses exact dyadic matrix norm bounds throughout.

## 4. Induction across tied and untied recurrences

The intervention occurs after expanded visit `v0`, injecting `a A (cosθ,sinθ)` at one token `j0`. Its angular derivative and maximum displacement from the clean center are each bounded at the intervention site by `a||A||₂`. Initialize `r_{j0}=a||A||₂` and every other radius at zero. At subsequent visits apply SCM-07.1 and SCM-07.2, using the appropriate tied or untied block.

For a frozen planar readout `C`, the pair curve `q_ij=(X_i-X_j)C` satisfies

\[
  \boxed{\|q'_{ij}(\theta)\|_2\le
         \|C\|_2(r_i+r_j).}
  \tag{SCM-07.3}
\]

The same upper bound controls displacement of the pair throughout the entire intervention disk. Any theta-independent RMP FF-U embedding reinjection cancels from the derivative and changes only the clean states used for local center estimates. If `i,j<j0`, all radii in their causal prefix are exactly zero by induction. For these positions, the observed braid is constant for the ideal RMP model, independently of the size of later perturbations.

## 5. Interaction with SCM-05

For `N` uniform angular intervals the exact length `2π/N` is strictly smaller than `44/(7N)`. SCM-07 outputs a per-visit uniform pair bound for SCM-05's exact rational polygon clearance test. A pass establishes topology between the **ideal continuous activation loop** and the polygon built from exact real samples, assuming all enclosures used by the bound are verified.

`scm07_audit.py` intentionally refuses to promote positive tests to an observed-link certificate because captured PyTorch coordinates and angle evaluation have unbounded roundoff in the current pipeline. Read-only source locking, trusted clean-state enclosures, and a verified forward-error envelope remain separate requirements. Algebraic source-global bounds are valid independently of captured centers, subject to the pinned architecture and stored-weight assumptions.

## 6. Falsifiers and limitations

- A tangent-vector JVP above the corresponding source-global bound in the same mathematical graph falsifies either the implementation or the analysis.
- The center-local bound must expand toward its global counterpart as `r+e` approaches the recorded center's mean-free norm.
- Any claimed downstream influence on strict causal-prefix tokens implies an invalid source/model assumption, a test harness bug, or a changed architecture.
- Local bounds can lose their advantage at later visits when their propagated tubes become too large. Reports must include every visit and every rejected inequality.
- The zero-disk criterion controls projected collision-free fillings of the intervention disk. It does not distinguish smoothly embedded Seifert surfaces in four dimensions or produce Khovanov cobordism maps.
