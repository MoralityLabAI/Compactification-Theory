# Infinite-flavor resolvent: an explicit Dirac theorem and rational certificates

**Paper:** Shu-Heng Shao and Nikita Sopenko, [arXiv:2611.05329v1](https://arxiv.org/abs/2611.05329), Eqs. (2)-(3), (10)-(11).  This note concerns the **one-particle free Dirac Hamiltonian** at fixed lattice momentum. The Schwinger model, anomalous many-body charges, and local von Neumann algebra statements remain separate questions.

## The infinite operator is well-defined

Fix real `k`. Work on `H = ℓ²(ℤ; ℂ)` with standard flavor basis `e_n`. Let

`(D u)_n = π(n + 1/2) u_n`, on the maximal domain
`Dom(D) = {u ∈ ℓ² : Σ_n (n+1/2)² |u_n|² < ∞}`.

Define the bounded perturbation `V(k)` by the exact coefficients of Eq. (3):

```
V_nm(k) = -(π/2) (-1)^n W_(m-n) *
          { cos(k),    m-n even;
            i sin(k),  m-n odd },
W_d = sinc(d/4)^2.
```

An explicit **uniform** bound is `||V(k)|| ≤ π` for all `k`:

* Separate even and odd flavors, `n=2a` and `n=2a+1`, identifying `ℓ²(ℤ)` with two copies of `ℓ²(ℤ)`. Let `E` be convolution by `W_(2r)` and `O` convolution by `W_(2r+1)` (with the appropriate shift). They commute, since both are bilateral convolution operators.
* The odd-square Basel sum `Σ_{j∈ℤ, j odd} j⁻² = π²/4`, and the elementary sine values at multiples of `π/4`, give `Σ_{r∈ℤ} W_(2r) = Σ_{r∈ℤ} W_(2r+1) = 2`. Since the coefficients are nonnegative, `||E||, ||O|| ≤ 2` by the `ℓ¹ * ℓ² → ℓ²` convolution inequality.
* In parity block form,

  `V(k) = (π/2) [[-cos(k) E, -i sin(k) O], [i sin(k) O*, cos(k) E]]`.

  Commutation of `E,O,O*` makes the off-diagonal blocks of `V²` vanish. Both diagonal blocks have norm at most `(π²/4) 4 (cos²(k)+sin²(k)) = π²`. `V=V*`; consequently `||V||² = ||V²|| ≤ π²`.

Thus `H(k) = D+V(k)` is self-adjoint on `Dom(D)` by the bounded self-adjoint perturbation theorem, and `(H-z)⁻¹` exists for each `z` with `Im(z) ≠ 0`, with `||(H-z)⁻¹|| ≤ 1/|Im(z)|`.

## Analytic full-projection theorem

Let `M=2p` and `P=P_M` project onto flavor indices `-p,...,p-1`; `Q=I-P`. Write `H_P = PHP` acting on `P H`, `H_Q = QHQ` acting on `Q H`, and `B=PHQ`. The finite cutoffs preserve the domain of `D`, so `H_Q` is self-adjoint on `Dom(D)∩QH`. Let `z=x+iy` with `y≠0` and set

`Δ_M(z) = π(p+1/2) - |x|`.

On the discarded tail, `|(D_n-z)| ≥ Δ_M(z)`. Provided `Δ_M(z)>π`, the Neumann series for `H_Q-z=(D_Q-z)(I+(D_Q-z)⁻¹ V_Q)` yields

`||(H_Q-z)⁻¹|| ≤ [Δ_M(z)-π]⁻¹`.

The Schur-complement identity is exact on the retained subspace:

`G_M - R_M = R_M B (H_Q-z)⁻¹ B* G_M`, where
`G_M=P(H-z)⁻¹P` and `R_M=(H_P-z)⁻¹`.

Taking operator norms and using `||B||≤π` and `||G_M||,||R_M||≤1/|y|` gives

\[
\boxed{\|P(H-z)^{-1}P-(H_P-z)^{-1}\|
\leq \frac{\pi^2}{|y|^2\left[\pi(M+1)/2-|x|-\pi\right]}.}
\]

The elementary bound `2/|y|` also applies, so the minimum is useful at small `M`. As `M→∞` at fixed `z`, the explicit upper bound tends to zero at rate `O(1/M)`. This is a **mathematical infinite-flavor operator-norm estimate**; `run.py` previously compared only two finite cutoffs. The theorem bounds *compressed* resolvents, with their distinct domains made explicit. It does not assert `||H-H_M||→0` for unbounded `H`.

## Tighter finite-core certificate with no floating-point trust

For exact trigonometry choose `k=π/2` and `z=iy` with positive integer `y`. Eq. (3) then simplifies to

\[
H_{nm}=\pi(n+\tfrac12)\delta_{nm}
-\begin{cases} 4i(-1)^n/[\pi(m-n)^2],&m-n\text{ odd},\\0,&m-n\text{ even}.\end{cases}
\]

Let `F: ℂ²→ℓ²(ℤ)` inject basis vectors into indices `n=-1,0`. Let `U:ℂ²→P_M ℓ²` be **any** finitely supported, rational approximate solution of `(H_M-iy)U≈F`; in practice `numpy.linalg.solve` proposes a candidate and each component is rounded to an exact integer multiple of `10⁻¹²`. The following residual inequality follows immediately from self-adjointness, for *any* such `U`:

\[
\|(H-iy)^{-1}F-J_M U\|_{\mathrm{op}}
\leq \frac{1}{y}\| (H-iy)J_M U-F\|_{\mathrm F}.
\]

`J_M` is extension by zero. The right-hand side is a bound against the **actual infinite operator**, and also bounds the corresponding compression to `F`.

`infinite_resolvent.py` verifies it using integer arithmetic alone:

1. Compute exact rational lower/upper enclosures for `π` from `π=16 atan(1/5)-4 atan(1/239)` and alternating Taylor-series bounds, then enclose `1/π` by positive reciprocal monotonicity.
2. Evaluate every residual component on indices `-N,...,N-1` with outward-rounded integer intervals. Sum squares of the maximum absolute real and imaginary endpoints, then take an upward integer square root. The candidate matrix is treated as exact rational input; numerical inversion is absent from validation.
3. For `|n|≥N`, use `|V_nm|≤4/[π(|n|-p)^2]` and `∑_{n=N}∞(n-p)⁻⁴≤1/[3(N-p-1)^3]`. For each core column `c`, let `A_c=Σ_{m∈P} (|Re U_mc|+|Im U_mc|)`. The discarded infinite tail has squared Frobenius norm at most

   ` [32/(3 π² (N-p-1)³)] * Σ_c A_c²`.

All quantities are enclosed with integers; the JSON receipt contains the outward-rounded scaled integers rather than depending on decimal printout. The proof relies on the standard arctangent identity, alternating series, bounded-perturbation self-adjointness, and the displayed arithmetic bounds. The certificate checker is exact-integer Python, **not a Lean kernel proof or a formally verified implementation of Python itself**.

## Protocol and scope

```
python -m unittest discover -s experiments/qft-lattice-map/tests -v
python experiments/qft-lattice-map/run_infinite_resolvent.py \
  --output experiments/qft-lattice-map/results/infinite_resolvent_certificate.json
python experiments/qft-lattice-map/run_infinite_resolvent.py \
  --verify experiments/qft-lattice-map/results/infinite_resolvent_certificate.json
```

The first command checks the original finite-Dirac reproduction plus independent matrix, exact-integer, and bound tests. The second generates untrusted candidate solutions and exact certificates. The third validates **the committed candidate bytes** without invoking a numerical solver. The proof and test runner require CPU only; `numpy` and `scipy` are already needed by the original experiment.

### Lean connection

The corresponding proof bridge in `patrickdugan/OntologicalProofs` is intended to kernel-check the norm propagation of the Schur identity and the residual criterion at the level of bounded complex-linear maps. Turning this into a Lean proof of the explicit `ℓ²(ℤ)` construction requires formalizing the parity convolution bound `||V||≤π`, the maximal unbounded diagonal domain, and the exact Fourier matrix. These remain explicit separate obligations.
