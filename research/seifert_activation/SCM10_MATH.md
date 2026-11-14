# SCM-10: A causal ball certificate at the full RMP decoder shape

**Proof scope:** Exact-real source function under a declared, conventional CPU IEEE-754 binary64/BLAS execution contract for the *verifier*. This source-level proof is conditional on arithmetic semantics that NumPy's public API does not formally attest. It **does not certify PyTorch CPU or GPU execution, BF16/TF32 tensor cores, recorded chart coordinates, or trained neural behavior**. We preserve the exact-rational SCM-09 backend as the independent small-shape reference.

## 1. Why direct coordinate intervals stopped scaling

SCM-09 encloses every scalar after each operation in an outward rational interval. At 64 tokens × 128 dimensions × eight visits, computing all dense interval matrix products is expensive, and independently widened coordinates lose correlations when reused in LayerNorm and attention. Our first SCM-10 coordinate-only implementation ran in seconds but became too wide to complete eight visits on the dense synthetic 128-dimensional graph. Those failures are part of the design evidence.

SCM-10 uses a **verified center + tokenwise Euclidean ball**. Source matrices remain exact binary rationals. A binary64 CPU computation supplies interval-enclosed *point evaluations* around each chosen center. The discrepancy between the source state and the chosen center propagates through analytic, causal Jacobian bounds. The two error channels are added after each visit.

## 2. Exact matrix-norm refinement

For a real matrix of stored dyadic parameters, scale all weights by a common power of two, obtaining an integer matrix `Z = 2^e W`. Then

    ||W||_2^2 = ||W W^T||_2 <= ||W W^T||_infinity

when the row dimension is smaller than the column dimension; otherwise use `W^T W`. Computing the integer Gram matrix and its maximum absolute row sum gives

    ||W||_2 <= 2^-e sqrt(max_i sum_j |(Z Z^T)_ij|).

The implementation rounds the final square root upward and takes the minimum of this exact Gram bound and the earlier SCM-06 source-exact bounds. It performs *no* floating-point matrix arithmetic for this norm certificate. An independently computed SVD serves only as a numerical diagnostic.

## 3. Short causal attention sensitivity

For a fixed token `i` at recurrence visit `v`, define `E_i >= ||X_i-C_i||_2`, where `C_i` is the declared exact binary64 center. The source LayerNorm admits the already-proved SCM-07 local Jacobian factor

    l_i = ||gamma||_inf / sqrt(eps + (max(0, ||P C_i||_2 - E_i)^2/d)).

`||P C_i||` is rounded downward and the output bound upward. Set `u_i=l_i E_i`, and let `u_* = max_{j<=i} u_j`. For each head, with source-exact bounds `q,k,v` on Q/K/V weight norms, the projected point-query/key/value upper bounds `Q_i^0,K_*^0,V_*^0` are obtained from interval-verified *point* evaluations of the source graph at the centers. Then

    Dq_i = q u_i, Dk_* = k u_*, Dv_* = v u_*,
    Ds_i = (Dq_i K_*^0 + Q_i^0 Dk_* + Dq_i Dk_*)/sqrt(head_dim).

The softmax Jacobian obeys the global operator estimate

    ||D softmax(s)[z]||_1 <= ||z||_infinity.

Indeed, if `p=softmax(s)`, the left side equals `E_p |z-E_p z|`, which is at most the standard deviation of `z` and hence at most `||z||_infinity`. The output of an attention head, a probability-weighted average of values, therefore varies by at most

    Dhead_i <= Dv_* + V_*^0 Ds_i.

Combining heads in Euclidean norm and applying the certified output-projection norm gives `Datt_i`. For RMP's `alpha,beta` residual graph,

    Dmid_i <= |alpha| E_i + |beta| Datt_i.

Use the interval-verified point-middle center and its point-roundoff radius to calculate the *second* local LayerNorm factor. Since exact GELU is globally 5/4-Lipschitz, the source block output error is bounded by

    Dpost_i <= (|alpha| + (5/4)|beta| ||Wdown|| ||Wup|| l2_i) Dmid_i.

Finally add the verified interval point-evaluation error. FF-U with embedding reinjection also adds the original certified input error at each visit. Induction on visits and causal token positions proves the returned per-token balls enclose the **ideal real-valued function**, subject to the explicit CPU verifier arithmetic contract.

## 4. Verifier floating-point proof contract

The CPU point evaluator uses ordinary IEEE binary64 elementwise `+,-,*,/`, `sqrt` and outward `nextafter`. For length-`n` sums/matrix products, the standard roundoff inequality uses `gamma_m = m u/(1-mu)`, `u=2^-53`. We inflate the budget to `gamma_(32(n+1))` to cover conventional product/reduction paths and the nonnegative-error summations that calculate their own bounds. Products and sums must use binary64 round-to-nearest, avoid overflow and unsupported unsafe reassociation, and preserve gradual underflow. Tensor cores and PyTorch GPU kernels are outside the contract. The runtime checks basic `float64`, `sqrt`, and `nextafter` behavior but **cannot prove the BLAS implementation respects this contract**. Hence public receipts are labeled `conditional_IEEE_roundoff_source_bound` rather than unconditional numerical-verification results.

The real `exp` for shifted attention scores is enclosed by a 32-term outward interval Taylor polynomial on `[-1/4,0]`, with an exact absolute remainder <= `2^-66`; scaling/squaring covers input scores in `[-64,0]`. The Gaussian CDF needed by exact GELU uses 60 Maclaurin terms on `[-4,4]` with a source-proven uniform remainder

    |R_N(x)| <= |x|^(2N+1) / (sqrt(2 pi) 2^N N! (2N+1)).

The prefactor uses the independently proved rational `pi` enclosure from SCM-08. Outside that range, the point GELU evaluator falls back to `0<=Phi<=1` and a global 5/4 derivative bound. Unsupported softmax-score domains and nonpositive denominators are rejected.

## 5. Compactification and topology interpretation

For a 64-token/128-hidden, eight-visit synthetic tied model with `beta=.1`, the full-shape verifier produces an upper error radius of about `7e-4` at visit eight. This is an **enclosure of the unperturbed clean state**. It is not evidence of a nontrivial braid. For a synthetic untied model with reinjection at the original RMP `beta=.25`, the corresponding value is about `7.3e-5`. A tied synthetic model at `beta=.25` has a much looser final bound of about 39, yielding an intentionally inconclusive late-visit clearance test.

An SCM-10 clean ball may be passed to SCM-07's causal perturbation-disk Jacobian module and SCM-05's projected-clearance check. Those downstream claims remain relative to a fixed observed 2D chart. Real `LOOP-T`/`FF-U` experiments require hash-verified external checkpoint and D-shared dictionary files. Physical floating-point inference and sampled chart reconstruction require independent bounds before claiming an observed braid, topology-preserving neural compression, or a Miller-type 4D surface effect.
