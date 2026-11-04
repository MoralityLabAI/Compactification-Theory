# LSV-07: Predictive finite-horizon goal regions for a pretrained recurrence

Date: 2026-11-04. **Evidence class:** written mathematics, local CPU unit tests, conditional FP64 interval implementation, and a checkpoint-locked CPU benchmark. This extends [LSV-05](README.md) and the post-hoc affine readout result [LSV-06](MARGIN_CERTIFICATE.md). This is **not yet a verified real-arithmetic numerical certificate** because the CPU backend's transcendental and BLAS error assumptions have not been independently verified. It does not certify FP32/GPU kernel rounding. The general interval-inclusion theorem is exact in real arithmetic.

## Mathematical problem

Let \(F:\mathbb R^{81\times48}\to\mathbb R^{81\times48}\) be the frozen shared recurrent operator, \(H(z)=Wz+b\) its per-cell affine digit logits, \(z_1\) the observed clean hidden tensor after the first pass, and \(B_0\) a preregistered perturbation set centered on \(z_1\). All admissible damage is injected once, after the first pass; three recurrent updates remain.

For a requested digit \(r\), the **goal region** is the open polyhedron

\[
G_r=\{z:\forall s\ne r,\ (W_r-W_s)\cdot z + (b_r-b_s)>0\}.
\]

A certifier must decide, before drawing or executing a damaging perturbation, whether every damaged trajectory is in the desired goal region after three passes.

## Theorem PG-01: finite-horizon output constancy from a reachable enclosure

Take any box \(B_0\) containing all declared damage to \(z_1\). Let \(\mathcal I_F\) be a **sound inclusion operator** satisfying \(F(B)\subseteq\mathcal I_F(B)\) for every box reached during the computation. Define \(B_{k+1}=\mathcal I_F(B_k)\), and let the interval extension of the final logit of class \(r\) be \([\ell_r^-,\ell_r^+]\). If

\[
\boxed{\ell_r^- > \max_{s\ne r}\ell_s^+},
\]

then \(H(F^3(z))\) has unique argmax \(r\) for every \(z\in B_0\).

**Proof.** By induction on \(k\), \(F^k(B_0)\subseteq B_k\). The interval extension encloses every logit at each point in \(B_3\). The strict inequality therefore places every such point inside the class-\(r\) goal region. QED.

A prediction certificate is label-preserving and can prove preservation of an *incorrect* answer. Report certified-correct digits separately, using labels only for evaluation.

## Theorem PG-02: perpetual goal invariance from a self-map

If an interval box \(B\) satisfies \(F(B)\subseteq B\) and is entirely contained in \(G_r\), then all its future recurrent iterates remain in \(G_r\), for any number of passes.

**Proof.** Invariance of \(B\) follows by induction from the self-map property. Since \(B\subseteq G_r\), every iterate satisfies the readout inequalities. QED.

A finite-horizon PG-01 certificate, even with an extremely high certified fraction, does **not** imply PG-02. In our code the invariant check is a separate output, and a failed self-map is recorded explicitly.

## Sound interval extensions for the actual recurrent block

The frozen upstream recurrence is

\[
T(z)=\mathrm{LN}_1\left(z+M_t(z)\right),\qquad
F(z)=\mathrm{LN}_2\left(T(z)+M_c(T(z))\right),
\]

where both \(M_t\) and \(M_c\) are affine–GELU–affine MLPs, the first mixing 81 token coordinates and the second mixing 48 channels. The same module is reused for all four steps.

**Affine:** For a box \([l,u]\), split the weight matrix into nonnegative and nonpositive parts. Componentwise exact affine extrema are \(W^+l+W^-u+b\) and \(W^+u+W^-l+b\).

**GELU:** The exact normal CDF \(\Phi\) is monotone, and \(\mathrm{GELU}(x)=x\Phi(x)\). An interval product of \([l,u]\) with \([\Phi(l),\Phi(u)]\) encloses the output. It may be tightened by intersecting with \(\mathrm{GELU}(c)\pm(5/4)r\), where \(c=(l+u)/2\) and \(r=(u-l)/2\), using the global \(5/4\) Lipschitz bound.

**LayerNorm:** For \(d\) channels and incoming intervals \([l_j,u_j]\), the mean lies in \([\bar l,\bar u]\), hence every centered coordinate lies in \(v_j\in[l_j-\bar u,u_j-\bar l]\). Bound \(v_j^2\) by its interval extrema; averaging gives an interval for variance. With \(\varepsilon_{LN}>0\), division by \(\sqrt{\mathrm{variance}+\varepsilon_{LN}}\) uses positive-denominator interval arithmetic. Affine gamma and beta follow. Independently, every normalized coordinate has magnitude at most \(\sqrt d\), giving \(|\mathrm{LN}_j|\le\sqrt d\,|\gamma_j|+|\beta_j|\) and a useful cap on otherwise exploding boxes.

Combining the operators with interval addition yields a sound enclosure of the **ideal real-valued** recurrent map, provided every primitive endpoint and CDF bound is itself sound. All of these results are established by elementary interval inclusion, the triangle inequality, and the LayerNorm variance identity. They are established principles of interval analysis rather than claims of novel general mathematics.

## Numerical soundness boundary

The file [`predictive_interval.py`](predictive_interval.py) evaluates the bounds with PyTorch **FP64**, enlarges affine endpoints by a 16-times standard dot-product summation guard, and pads nonlinear operations. It assumes an absolute Torch `erf`/normal-CDF error no larger than `2e-12`; the implementation also pads sqrt and normalization stages. That library-specific assumption has **not** been independently certified. Even with extensive successful sampling, the returned status remains `conditional_float64_interval`. A formal numeric certificate would require a verified interval backend with directed rounding, proven CDF bounds, and a model of the deployed FP32/GPU kernels.

The initial stage-one tensor is treated as exactly equal to the stored binary32 numbers. Its generation from input tokens and the first recurrent step is outside PG-01's damage-path proof. A fully end-to-end guarantee from raw Sudoku input must also enclose the first step and embedding arithmetic.

## Frozen benchmark

The original 64 Sudoku puzzles, 30 missing cells per puzzle, original checkpoint SHA256, and upstream source revision remain unchanged. Interventions are either (A) independent box damage to all 81×48 hidden entries or (B) damage restricted to all 48 channels of the **first missing cell**, which can be identified directly from the input puzzle. For each intervention footprint, test radii `0`, `1e-8`, `1e-6`, `1e-4`, `0.02`, `0.10`, and `0.50` without selecting a favorable radius after observing results.

Compute the certificate before running independent random damages (two pre-fixed seeds, 34 and 89), and record any contradiction as a hard failure. Test whether every certified label agrees with the independent, undamaged FP64 forward pass. Report the count of certified digits, certified correct digits, certified wrong digits, and completely certified correct puzzles, along with full interval widths and runtime. All observations have the same frozen model and evaluation roster. **This experiment cannot establish the formal error specification of PyTorch's transcendental kernels.**

A negative result at a practical perturbation radius identifies lost precision in interval relaxation, true vulnerability of the model, or both. Output constancy under damage can coexist with expanding hidden-state discrepancies. The pre-execution certificate is more demanding than a post-hoc certificate whose radius comes from the actual damaged final state.

## Next mathematical avenues

- **PG-03:** relational or zonotope bounds that preserve token/channel correlations through LayerNorm, with genuinely verified normal-CDF enclosures.
- **PG-04:** barrier functions or a contraction in a task-relative seminorm, rather than the full hidden-state infinity norm.
- **PG-05:** checkpoint- and dataset-frozen utility comparisons between a certificate-driven early-stop controller and fixed-depth baselines, including actual certificate-computation overhead.
