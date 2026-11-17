# SCM-11: Source-centered value-diameter bounds for recurrent intervention dynamics

**Evidence class:** Exact-real mathematical inequalities, source-validated Q/K/V and attention-probability reference intervals under SCM-10's declared IEEE binary64/BLAS assumptions, small independent 80-digit checks, and fixed synthetic 64×128×8 evidence. No trained RMP activation traces, float32/CUDA numerical certificates, intrinsic latent knots, or four-dimensional Seifert-surface obstructions are claimed.

## 1. The overestimate in earlier attention bounds

SCM-07 and SCM-10 use an operator-norm estimate in which changes to query and key scores can cost the entire magnitude of the value vector. That is pessimistic when value vectors share a large offset, or when the altered values lie at positions with low baseline attention weight. A softmax row is a probability vector, and its first-order change has zero total mass. Retaining that cancellation yields a stronger inequality.

Fix a causal prefix `0..i` of one attention head. Let `s0`, `p0=softmax(s0)`, and `v0_j` be the source-validated, *exact-real* Q/K/V reference at the SCM-10 certified center. Let `(sb,vb)` denote the unknown clean state and `(s,v)` the perturbed state. Assume established bounds

- `||sb-s0||_inf <= DB`, `||s-sb||_inf <= DS`;
- `||vb_j-v0_j||_2 <= eb_j`, `||v_j-vb_j||_2 <= dv_j`;
- every reference value lies within radius `R0` of **one common anchor** `a`.

The softmax Jacobian `diag(p)-pp^T` has induced norm `L_infty -> L_1` at most 1, so `||p-p0||_1 <= DB+DS` and `||p-pb||_1 <= DS`. The difference in the attention head decomposes as

    sum_j p_j v_j - sum_j pb_j vb_j
      = sum_j p_j (v_j-vb_j) + sum_j (p_j-pb_j) vb_j.

Since the difference in probability sums is zero, subtract the common anchor from every `vb_j` in the second term. Its norm is at most `(R0+max eb_j)*DS`. For the first term, the total-variation inequality for a bounded scalar variable gives

    sum_j p_j dv_j <= sum_j p0_j dv_j
                        + (DB+DS)*(max dv - min dv)/2.

Combining this with the trivial `max dv` upper bound yields

    HEAD <= min(max dv,
                sum_j p0_j dv_j + (DB+DS)*(max dv-min dv)/2)
            + (R0+max eb_j)*DS.                         (SCM-11.A)

The implementation replaces `p0_j` by rigorously calculated **upper intervals** and chooses the smaller of two certified anchor balls, one centered at zero and one centered at the first reference value. This is safe for arbitrary scores and weights, even when the value vectors vary wildly. If all values are identical and unchanged, score perturbations contribute exactly zero to the head variation. For a singleton causal prefix, this remains true regardless of query/key changes.

## 2. Score variation from source-certified Q/K norms

Let `eq_j` and `ek_j` denote the source-clean Q/K center errors and `dq_i,dk_j` the additional intervention response. With source Q/K norms `q0_i,k0_j` and head dimension `dh`, the score change between baseline and perturbation obeys

    DS <= max_{j<=i} [dq_i*(k0_j+ek_j) + (q0_i+eq_i)*dk_j + dq_i*dk_j] / sqrt(dh).

The baseline score error `DB` uses the analogous expression with `eq,ek` and the source norms. The Q/K/V bounds are derived by source-certified `LayerNorm`, matrix norm, and centered clean-state tube calculations. Per-position value error bounds are `||W_V|| * L_LN * radius`. Thus (SCM-11.A) covers the **entire two-dimensional perturbation disk**, rather than sampled angular gradients.

The four head bounds combine by Euclidean quadrature and a certified output-projection spectral norm. The middle residual response has upper bound `|alpha|*r_i + |beta|*d_att_i`. Source-validated middle centers and the local second LayerNorm bound then give the block response

    r'_i <= (|alpha| + |beta|*||W_down||*(5/4)*||W_up||*L2_i) * r_mid_i.

SCM-11 carries these bounds through tied and untied visits using exact rational arithmetic, with conservative decimal-grid upward rounding. For FF-U, input reinjection is fixed for the whole intervention, so it cancels in the perturbed-versus-clean difference. It changes the verified clean centers, which are read from the SCM-10 source tube.

## 3. Exact tokenwise LayerNorm gauge

For any channel-constant row `U_i=c_i 1`, LayerNorm obeys `LN(x_i+U_i)=LN(x_i)` in exact real arithmetic. For the pre-normalized RMP block

    middle(X) = alpha X + beta Attention(LN1(X)),
    B(X) = alpha middle(X) + beta MLP(LN2(middle(X))),

we therefore have the exact equivariance

    B(X+U) = B(X) + alpha^2 U.                   (SCM-11.B)

A shift at the injected token remains at that token through every later visit. The other token states and their final logits are independent of this shift. If one axis of a two-dimensional cyclic intervention is channel-constant, all observed tokens excluding the injected position depend only on the other axis' scalar coefficient. Hence the projected observation loop factors through an interval and is nullhomotopic **when it is collision-free for the entire interval**. The gauge checker reports that collision-free condition as a separate unverified premise. The theorem assumes the pinned pre-LN block and intervention-independent reinjection. PyTorch rounding is outside its scope.

## 4. What is certified and what is measured

The algebraic attention and gauge results are mathematical proofs. The `refined_response.py` implementation derives conditional source-ball and null-braid statements from the already verified SCM-10 clean tubes; it never treats observed PyTorch/CUDA activation coordinates as certified. The source Q/K/V intervals and probability envelopes use the SCM-10 CPU IEEE/BLAS contract. Stored float centers are **exact dyadic reference points**, with separately tracked distance to the ideal real clean states.

A source-null certificate requires every clean projected pair separation to exceed the bound on the full perturbation disk. It guarantees the loop in the **fixed two-dimensional readout chart** is nullhomotopic and its closure is the trivial pure braid. The existence of any nontrivial link, its Milnor invariant, and the topology of surfaces in the four-ball are separate experimental questions. SCM-11 retains SCM-05's sampled-curve and quantization certificates for those purposes.

The 64×128, four-head, eight-visit tied synthetic model at beta=0.1 serves as a deterministic full-size reference. An intervention at token 4 with RMS-relative radius 10^-6 and a fixed two-dimensional coordinate chart is analyzed at observed tokens 16,32,48. SCM-11's visit-eight maximum projected-pair displacement bound is about 0.04955; the SCM-07 inherited bound on the identical source is about 2.84e43. All eight source-only null-braid gates pass. These values compare *mathematical upper bounds*, not neural model accuracy or compression gains.

## 5. Limits, falsifiers, and next experiment

- SCM-10's declared CPU roundoff assumptions remain essential. The current backend is restricted to power-of-two hidden dimensions, which includes RMP's 128. Arbitrary LayerNorm hidden dimensions need a separately certified rational `1/d` implementation.
- Source-only attention-score reference intervals must enclose every score and normalization probability. Negative denominators, unsupported nonlinear ranges, nonfinite input data, and excessively large response budgets cause refusal.
- Baseline source receipts remain separate from the new intervention bound. Reconstructing model/checkpoint hashes and the calibration task is mandatory in `scm11_audit.py`.
- The full trained `LOOP-T` and `FF-U` checkpoints, D-shared dictionaries and `monitor_calibration` bundle are held outside this environment. Existing RMP G0 and frozen registrations are untouched.
- The real test must keep intervention axes, readout chart and clean-state anchors fixed across comparisons. Report the fraction of certified null loops, any eligible nontrivial loops, task accuracy changes, and resource use. Compare with shuffled and random charts at matched compute.

**References:** SCM-05 sampled isotopy criterion, SCM-09 rational interval centered verifier, SCM-10 full-shape causal ball proof, and the original RMP source `research_gym/neural/looped_decoder.py` / `research_gym/interp/analysis.py`.
