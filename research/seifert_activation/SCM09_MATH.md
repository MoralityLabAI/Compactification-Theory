# SCM-09: Verified-centered clean-state enclosures for recurrent activation topology

**Scope:** A source-level, ideal-real arithmetic result for the frozen RMP `ExplicitDecoderBlock` graph. The newly implemented algorithm uses exact integer/rational computations and outward dyadic rounding for affine arithmetic, analytic rational error bounds for sqrt, exp, Gaussian CDF and exact GELU, and interval propagation through attention and LayerNorm. It does **not** certify PyTorch floating-point inference, GPU sin/cos, or any trained `LOOP-T` / `FF-U` observation. All reference model weights are synthetic and fixed by a deterministic dyadic recipe.

## Why SCM-08's clean tubes grow

SCM-08 uses an input anchor `a_t X_0` and an upper bound on the complete attention/MLP image at each visit:

    r_next <= |alpha|^2 r_t + |alpha beta| A_global + |beta| H_global.

Even when the model is numerically close to its clean trajectory, the large conservative source-image terms accumulate. The algorithm remains valid, but its tube is insufficient for source-derived geometric clearance at late visits.

## New theorem: exact-real interval execution encloses the clean trajectory

Suppose `X_0` lies coordinatewise within certified interval boxes, the stored model weights and epsilon constants are interpreted exactly as their binary-rational values, and the computational graph is:

    N1 = LN1(X)
    q,k,v = W_qkv N1
    p_ij = exp(<q_i,k_j>/sqrt(d_h))/sum_{l<=i} exp(<q_i,k_l>/sqrt(d_h))
    A_i = W_o concat_h(sum_{j<=i} p_hij v_hj)
    M_i = alpha X_i + beta A_i
    N2_i = LN2(M_i)
    F_i = W_down GELU(W_up N2_i)
    X'_i = alpha M_i + beta F_i.

The following interval operators are inclusion-isotone for exact-real arguments and enclose the output of each node when fed containing intervals:

* Addition, subtraction, multiplication and reciprocal with rational endpoint minima/maxima, each rounded outward to a `2^-64` grid.
* `sqrt([l,u]) = [sqrt_lower(l),sqrt_upper(u)]` with integer-square-root rational endpoints.
* `exp(x)` obtained by reducing `|x|` by powers of two, summing 33 exact rational Taylor terms for the reduced `0<=|x|<=1/4`, bounding the positive remainder by a geometric tail, then squaring intervals or reciprocating as appropriate. Unrepresentably large proof requests are refused rather than evaluated heuristically.
* For the exact normal CDF `Phi(x)`, use the alternating series

      Phi(x)=1/2+1/sqrt(2 pi)*sum_{n>=0} (-1)^n x^(2n+1)/(2^n n! (2n+1)).

  Every used series truncates after its terms are decreasing; the first omitted absolute term bounds the remainder. The normalizing `pi` interval comes from the Machin/arctangent rational proof in SCM-08. For `|x|>6`, a conservative probability interval is used.
* `GELU(x)=x Phi(x)` at the interval midpoint, enlarged by `(5/4)*radius` on either side. This follows from `|GELU'(x)|=|Phi(x)+x phi(x)|<5/4` for every real x.
* LayerNorm uses interval enclosures for its shared mean, centered coordinates, positive variance, inverse square root of `variance+epsilon`, and affine gamma/beta.
* Softmax subtracts a **fixed rational** shift, then bounds each positive weight as `e_i/(e_i+sum_{j != i} e_j)` using independently enclosed numerators and denominators. Masked future tokens are excluded by the RMP causal graph.

Induction over this graph and the recurrent visit schedule proves that every exact-real clean pre-LayerNorm, middle-state and post-visit token vector lies in its returned coordinate boxes. Export to float64 records adds the *exact* center-rounding distance before producing Euclidean error radii.

For FF-U input reinjection, add the original certified input interval after each block, as specified by the RMP `AnalysisModel` runner.

This yields, for every visit `t` and token `i`, a certified bound

    ||X_ideal_real[t,i] - C[t,i]||_2 <= E[t,i].          (SCM-09.1)

`C` is a published numeric center, with all rounding errors included in `E`. Here `X_ideal_real` is the **defined ideal real computation**, not an IEEE-754 execution trace. SCM-09's synthetic tests compare it to an independent 90-decimal-digit implementation.

## Causal topology consequence

Use these enclosures at both pre- and middle-LayerNorm sites as the clean baseline for SCM-07's tokenwise perturbation-disk Jacobian bounds. The intervention norm is bounded using the **post-visit centered tube**:

    a_upper = r_source * max( (||C_post,j||_2 + E_post,j)/sqrt(d), float_binary(1e-6)).

For a frozen readout chart `Q`, observed token pair `(i,j)` has a proven clean separation lower bound

    s_ij,t = max(0, ||(C_t,i-C_t,j) Q||_lower
                     - ||Q||_upper*(E_t,i+E_t,j)).

Let `D_ij,t` bound the corresponding projected pair's perturbation throughout the disk, derived via SCM-07 from the certified clean tubes. If `s_ij,t > D_ij,t + margin` for every observed pair, the disk has no projected collisions. Therefore its continuous boundary pure braid is null-homotopic in the observed chart. This can prove a **null braid** for a particular frozen chart and task, and it never asserts intrinsic hidden-state topology or a four-dimensional Seifert-surface isotopy class.

## Engineering limits and refusal

The reference implementation is intentionally bounded to at most 16 tokens, 32 hidden dimensions and 8 visits by default. RMP's original 64-token, 128-hidden full cells require an optimized verified interval backend and their external checkpoints/dictionaries. The CLI accepts higher *explicit* caps but runtime and memory can grow rapidly, especially with long causal attention prefixes. Any unsupported architecture, positive score exponent above its verified domain, incompatible embedding scope, non-eval model, or missing provenance results in a refusal.

The source contract inherits SCM-08's frozen exact-dyadic positional-frequency table. The physical frequency-table computation on another CPU/GPU backend must be matched separately. Physical rounding of residual states, softmax, GEMMs, and recorded chart coordinates is not covered.

## Interpretation for compactification

For a fixed instrumented chart, the verified clean lower clearance and perturbation upper bound give a sufficient guarantee that the intervention boundary braid is trivial. It is stronger than simply assigning a topology label to sampled activations: it is a checkable source-derived statement about a continuous real function. The bit-depth result in SCM-05 can be combined with separately bounded output-coordinate roundoff after a trained capture has been checked.

The measured shrinkage of the clean tube in the SCM-09 CPU reference concerns synthetic exact-real programs; no claim of reasoning-specific braid structure, trained-model compression, parameter-bit information gain, or a Khovanov surface obstruction follows.
