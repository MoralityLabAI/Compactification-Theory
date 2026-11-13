# SCM-08: source-derived clean-state tubes and ideal null-braid certificates

**Date:** 2026-11-13. **Evidence:** elementary exact-real inequalities, rational enclosures of a specified idealized embedding function, and an executed CPU synthetic reference. This note is on top of SCM-07. No trained RMP checkpoint or D-shared artifact was available to this run. No claim about the actual rounded PyTorch computation or an observed, trained-model braid follows from this work.

## The outstanding problem

SCM-07's local LayerNorm constant is a rigorous theorem **if** a proposed clean-state center has a proven error radius. Its previous CPU reference used PyTorch float centers with synthetically asserted zero error. SCM-08 removes that assertion for a specified exact-real input model by constructing clean-state enclosures directly from the frozen weights, input tokens, and source architecture. An independently enclosed center need not equal the numerical hidden state recorded by PyTorch.

## 1. Exact-real input construction

The source model's input is

    X_0[pos] = LN_input(E[token[pos]]) + position(pos).

All stored embedding and LayerNorm weights are interpreted as exact binary rational coefficients. For a row x of dimension d, write `m=sum(x)/d`, `v=sum((x-m)^2)/d`, and `s=sqrt(v+eps)`. Integer square roots produce rational bounds `s_lo<=s<=s_hi`. Consequently

    1/s_hi <= 1/s <= 1/s_lo.

Multiplying this interval by `gamma_i(x_i-m)` and adding `beta_i`, with interval endpoints ordered according to the multiplier sign, gives an enclosure for every LayerNorm output coordinate. Implementation obtains a symmetric rational center and a guaranteed absolute coordinate error from the interval for `1/s`.

For positional encoding the idealized source contract takes the float32 frequency table `f_k` **as frozen exact dyadic constants**, and defines the corresponding real functions `sin(pos*f_k)` and `cos(pos*f_k)`. This is an explicit model definition and frequency-table SHA-256 is recorded. The table can be reconstructed on CPU or supplied directly. It is not an enclosure of the floating torch.sin/torch.cos output on another backend.

To enclose the real sine/cosine, use the exact Machin identity

    pi/4 = 4 atan(1/5) - atan(1/239)

with the alternating arctangent series, a next-term rational remainder, rational period reduction and a Taylor polynomial whose derivative remainder is bounded by `|x|^(N+1)/(N+1)!`. The final midpoint is rounded to a `10^-18` decimal grid and the lost precision is added outward to its error bound. All other center conversions to float64 include a separately computed exact rational-to-binary rounding radius. Hence `rmp_input_enclosure` returns `c_0,pos` and `e_0,pos` such that

    ||X_0,pos - c_0,pos||_2 <= e_0,pos                  (SCM-08.1)

for the stated ideal real embedding and stored-frequency model.

## 2. Source-only attention and MLP image bounds

For the pinned `ExplicitDecoderBlock`, let `Y=LN1(X)` and `M1` be SCM-06's unconditional row norm upper bound for `Y`. Head h outputs

    H_h,i = sum_{j <= i} p_h,ij (W_v,h Y_j)

where the causal softmax probabilities are nonnegative and sum to one. Thus

    ||H_h,i||_2 <= v_h M1

for `v_h >= ||W_v,h||_2`. Concatenate all heads and apply `W_o` to get

    ||Attn_i||_2 <= A = ||W_o||_upper * M1 * sqrt(sum_h v_h^2).    (SCM-08.2)

This holds for any input and for every attention-score configuration. It follows from a convexity bound and requires no numerical softmax evaluation.

At the second LayerNorm, let `M2` bound the normalized output row. The exact erf GELU satisfies `|GELU(z)| = |z Phi(z)| <= |z|` componentwise. With unbiased RMP MLP matrices,

    ||MLP(LN2(middle_i))||_2 <= H = ||W_down||_upper ||W_up||_upper M2.   (SCM-08.3)

Biases or altered normalization/activation graphs are refused. Matrix norm bounds use SCM-06's exact dyadic and integer-square-root implementation.

## 3. Anchored clean tubes through the recurrence

Write `c_0,i` for the exact input center before float64 export, and let `e_0,i` bound its error. For visit t suppose

    ||X_t,i - a_t c_0,i|| <= r_t,i.

RMP's pre-norm block has

    middle_i = alpha X_t,i + beta Attn_i,
    B(X_t)_i = alpha middle_i + beta MLP_i.

A valid middle-state center is `alpha*a_t*c_0,i`, with radius

    s_t,i = |alpha|r_t,i + |beta|A.                          (SCM-08.4)

For the block output, use center `alpha^2*a_t*c_0,i` and radius

    r_next,i = |alpha|^2 r_t,i + |alpha beta|A + |beta|H.    (SCM-08.5)

For RMP's fixed FF-U input reinjection, add `c_0,i` to the center and `e_0,i` to the radius after each block. Therefore

    a_next = alpha^2 a_t + 1,
    r_next,i <- r_next,i + e_0,i.

The code iterates exact rational coefficients and radii, then converts pre/middle centers to float64. Its exported center-error bound includes exact conversion loss. No clean activations, Jacobian samples or checkpoint-dependent float forward evaluations are used.

For SCM-07's perturbation disk, the clean-center error bounds above are used *in addition* to its perturbation radii. SCM-07 then derives local LayerNorm sensitivity bounds. The result is source-verified in the defined real model and may be too loose when a long recurrence accumulates large source-only tubes.

## 4. Source-derived injection RMS

RMP's intervention amplitude is `a = relative_radius * RMS(X_clean_after_visit[j0])`. If a clean post-visit center is `a_t*c_0,j0` with radius `r_t,j0`, triangle inequality gives

    RMS(X_clean_after_visit[j0]) <=
        (||a_t*c_0,j0|| + r_t,j0) / sqrt(d).              (SCM-08.6)

The implementation rounds both square roots upward using rational arithmetic. Its source-derived amplitude upper is substituted for the unverified float-RMS value in SCM-06/07. The operator norm of the intervention axes remains a separate exact dyadic upper bound.

## 5. A source-only null-braid test

Let C be the fixed linear readout chart. For observed distinct tokens i,j at visit t, the exact clean projected pair separation has the lower bound

    D_clean,ij,t >= ||(a_t*c_0,i - a_t*c_0,j) C||_lower
                     - ||C||_upper*(r_t,i+r_t,j).         (SCM-08.7)

Clamp the result below by zero. Every quantity on the right is computed from exact rational anchors, proven norm bounds and outward square-root rounding.

Let `U_ij,t` be SCM-07's independently derived upper bound on the pair's displacement over the *entire two-dimensional intervention disk*. If, for all pairs,

    D_clean,ij,t > U_ij,t + margin,                         (SCM-08.8)

then no pair collides in the disk for the specified **continuous real** model. The boundary loop is therefore null-homotopic in `Conf_n(R^2)` and its pure-braid class is trivial. The theorem requires no observed float samples. This is a sufficient null result, not a way to prove a nontrivial neural braid or a four-dimensional Seifert-surface obstruction.

## 6. Proof boundaries

- The constructed mathematical graph has exact stored floating-point weights, a separately frozen exact-dyadic frequency table, real-valued LayerNorm, attention softmax, GELU and sin/cos, and a fixed single-example input.
- A source verification step pins the RMP implementation blobs, while the checkpoint and dictionary hashes must match the capture before a real audit.
- The numerical frequency-table generation and execution hardware may change the table; the host must pin and compare its table SHA-256 to the construction used in the proof.
- This program encloses the ideal real graph defined above. It does **not** establish an enclosure for rounded PyTorch forward outputs or the recorded intervention angle, nor does it establish a verified 4D embedded Seifert surface.
- For a source with arbitrary learned modules, nonlinear approximate GELU, biases or carry, the theorem and implementation require a fresh derivation or explicit rejection. The source-only clean radius may become too large to improve the local LayerNorm factor.

## 7. Reproduce

    python -m unittest discover -s research/seifert_activation/tests -q
    python research/seifert_activation/scm08_reference.py --out /tmp/scm08_reference.json
    python research/seifert_activation/scm08_audit.py --help

`results/scm08_cpu_reference.json` is a fixed synthetic CPU reference. It contains no trained-model evidence.
