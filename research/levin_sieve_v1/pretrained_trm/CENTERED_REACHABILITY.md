# LSV-08: Centered, task-directed sensitivity bound

**Date:** 2026-11-04. **Evidence:** written mathematical inequality, exact upstream source/checkpoint lock, CPU control tests, post-LSV-07 exploratory experiment. The practical exploration was designed **after seeing the failed LSV-07 interval result** and is neither confirmatory nor preregistered. This method concerns the third-party TRM-inspired 44K Sudoku model, not the original TRM, Qwen or LOTUS.

## Motivation

Straight interval propagation in LSV-07 discards coordinate correlations. On the frozen 64-puzzle roster it certified 101/5184 output cells at *zero* perturbation and none at even `1e-8`. The present method conditions on the exact clean latent trajectory `c_t` and propagates only a guaranteed coordinatewise error vector `r_t`. With a *mathematically exact* reference trajectory, zero error stays exactly zero; a local Jacobian estimate can be dramatically tighter than a global Lipschitz constant.

## Theorem CS-01: local LayerNorm difference radius

Let `LN(x)=gamma ⊙ (x - mean(x))/sqrt(var(x)+eps) + beta` on \(d\) coordinates, with \(\varepsilon>0\). For a fixed clean vector \(c\), consider all deviations \(x=c+\delta\) satisfying \(|\delta_i|\le r_i\), \(r_i\ge0\). Define

\[
 v_i=c_i-\overline c,\quad \sigma_c=\sqrt{d^{-1}\sum_i v_i^2+\varepsilon},
 \quad R=\max_i r_i,\quad\bar r=d^{-1}\sum_i r_i,
\]
\[
\sigma_*=\max(\sqrt\varepsilon,\sigma_c-R),\qquad
V_i=|v_i|+r_i+\bar r,\qquad S=\sum_j V_j r_j.
\]

Then every allowed perturbation satisfies the *coordinatewise* bound

\[
\boxed{
|LN_i(c+\delta)-LN_i(c)|\le |\gamma_i|\left(
\frac{r_i+\bar r}{\sigma_*}+
\frac{V_iS}{d\sigma_*^3}\right).}
\tag{CS-01}
\]

The bound may be tightened by taking the minimum with \(2\sqrt d|\gamma_i|\), an unconditional range bound on the difference of two normalized LayerNorm outputs.

**Proof.** Let \(P=I-\mathbf 1\mathbf 1^\top/d\). Then \(\sigma(x)=\sqrt{\|Px\|_2^2/d+\varepsilon}\). By the reverse triangle inequality for the Euclidean norm of \((Px,\sqrt{d\varepsilon})\),
\[
|\sigma(c+t\delta)-\sigma(c)|\le t\|P\delta\|_2/\sqrt d\le tR
\]
for every \(t\in[0,1]\). Therefore \(\sigma\ge\sigma_*\), and \(|(P(c+t\delta))_i|\le V_i\). The Jacobian is
\[
\frac{\partial LN_i}{\partial x_j}=\gamma_i\left(
\frac{\mathbf 1_{i=j}-1/d}{\sigma(x)}-
\frac{(Px)_i(Px)_j}{d\sigma(x)^3}\right).
\]
Bound the product with \(\delta\) using \(|\delta_j|\le r_j\), sum the row, and integrate the Jacobian along the segment. The displayed inequality follows. The normalized coordinates have squared norm at most \(d\), which yields the separate difference bound. QED.

## Theorem CS-02: local error composition and goal-relative margin

For an affine map \(z\mapsto Wz+b\), propagate coordinatewise uncertainty by \(r' = |W|r\). For GELU, use the global derivative magnitude bound \(5/4\) to set \(r'=\frac54 r\). Residual sums add radii. Use CS-01 for both LayerNorm sites. Composing these rules over any finite number of frozen recurrent passes gives a coordinatewise radius containing the output of **every** admissible damaged trajectory about the exact nominal trajectory.

Let \(c\) be the nominal final state at one Sudoku cell, and \(a\) its unique highest-scoring class under an affine readout. For each rival \(b\), define

\[
 m_{ab}=(w_a-w_b)^\top c+(\beta_a-\beta_b),\quad
 D_{ab}=\sum_j |w_{aj}-w_{bj}|r_j.
\]

If \(m_{ab}>D_{ab}\) for every \(b\ne a\), then **every** admissible damaged latent state has the same output digit \(a\). Proof: the worst reduction of each affine pairwise margin is at most \(D_{ab}\) by the triangle inequality. This is the task-relative counterpart of the LSV-06 post-hoc theorem, with \(r\) computed before damage.

## Numeric implementation boundary

[`centered_reachability.py`](centered_reachability.py) implements the formulas. It computes the nominal center using PyTorch FP64 and uses ordinary FP64 matrix multiplication for nonnegative error propagation. It does **not** establish that these computed values enclose the mathematical exact-real centers and radii, nor does it certify BLAS, LayerNorm, GELU, square-root or floating-point rounding. Thus counts are labelled *conditional sensitivity certificates* pending a directed-rounding interval backend and a rigorous model of the nominal center. A zero perturbation can give 5184/5184 under the exact-center assumption without constituting a CPU/GPU execution guarantee.

For visibility, the checkpoint is loaded only after both its Git blob and SHA256 hashes match the original LSV-05 lock. The model and 64-puzzle roster are unchanged. The experiment compares whole-state damage with damage restricted to 48 hidden channels of the first missing Sudoku cell; radii equal the LSV-07 values. Independent random forward passes under seeds 34 and 89 check that no purportedly stable class flips in sampled FP64 execution, although sampling cannot establish worst-case validity.

## Interpretation

If LSV-08's positive predictive coverage exceeds LSV-07, the difference is due to retaining correlations to the exact nominal trajectory and a tighter LayerNorm sensitivity estimate. Both algorithms' results remain conditional on numeric implementation assumptions. Both can certify a wrong digit as **unchanged**. None of these results alone establishes learning, self-repair of global hidden state, perpetual region invariance, or safe model editing.

The next independent mathematical milestone is converting this conditional analytical envelope to a computer-checked numeric certificate, ideally with directed interval rounding and verified Gaussian CDF enclosures, then testing on new model/problem families that were not used to develop LSV-08.
