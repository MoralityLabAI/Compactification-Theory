# OIC-T26–T28: Scaling phase-localization on learned cyclic beliefs

**Mathematics:** elementary exact bounds on smooth periodic emissions, contingent on the declared cyclic structure. **Empirical evidence:** see OIC18_REPORT.md for the executed, seed-paired CPU sweep. The abstract statements do not establish trained-model reliability.

## Definition: normalized cyclic emission

Let the declared period be an integer \(n\ge 2\), and let \(\mathbb Z_n\) denote latent phase labels. A phase-dependent binary pulse probability is \(p_n(z)=\sigma(\ell_n(z))\). Consider the von Mises-shaped learned emission

\[
\ell_n(z)=a-\kappa_0\Bigl(\frac{n}{n_0}\Bigr)^2\left(1-\cos\frac{2\pi z}{n}\right),\qquad \kappa_0>0,\ n_0>0.
\]

The OIC-18 experiment sets \(n_0=5\) and learns \(a\) and the positive \(\kappa_0=\operatorname{softplus}(k)\). The \(n^2\) rule is an explicitly supplied scale prior. The five-shift stochastic transition kernels are separately learned.

## OIC-T26: Uniform positive nearest-neighbor logit gap

For all \(n\ge2\), nearest-neighbor phase logits obey

\[
\boxed{\frac{8\kappa_0}{n_0^2}\le\ell_n(0)-\ell_n(1)\le\frac{2\pi^2\kappa_0}{n_0^2}.}
\]

**Proof.** The identity \(1-\cos(2\pi/n)=2\sin^2(\pi/n)\), the bound \(\sin(\pi/n)\ge 2/n\) for \(n\ge2\), and \(1-\cos t\le t^2/2\) give

\[
\frac8{n^2}\le1-\cos(2\pi/n)\le\frac{2\pi^2}{n^2}.
\]

Multiply by \(\kappa_0 n^2/n_0^2\). Both bounds are independent of phase count. \(\square\)

With \(a\ge m>0\) and \(8\kappa_0/n_0^2\ge a+m\), the logit for the marked phase is at least \(m\), and that for every other phase is at most \(-m\). These are sufficient conditions for a uniform fixed classification margin. They may be conservative.

## OIC-T27: Uniform concentration of off-marker emission mass

Write \(d_n(z)=\min(z,n-z)\), and take \(z\ne0\). Then

\[
\ell_n(z)\le a-\frac{8\kappa_0}{n_0^2}d_n(z)^2
\]

and therefore

\[
\boxed{\sum_{z=1}^{n-1}p_n(z)
\le 2e^a\sum_{j=1}^{\infty}\exp\left(-\frac{8\kappa_0j^2}{n_0^2}\right).}
\]

**Proof.** Replace \(z\) by its cyclic distance \(d\in[1,\lfloor n/2\rfloor]\). The previous sine inequality at \(\pi d/n\in[0,\pi/2]\) gives \(1-\cos(2\pi d/n)\ge8d^2/n^2\). Since \(\sigma(v)\le e^v\) for all real \(v\), the contribution of either phase at cyclic distance \(d\) is bounded by \(e^{a-8\kappa_0d^2/n_0^2}\). There are at most two phases at a given distance. Summing yields the stated convergent Gaussian upper bound, uniform in \(n\). \(\square\)

For any *fixed integer* distance \(j\), the logarithm of the probability odds has the asymptotic form

\[
\lim_{n\to\infty}\ell_n(j)=a-\frac{2\pi^2\kappa_0}{n_0^2}j^2.
\]

Thus the effective emission width stays \(O(1)\) in *discrete phase slots*, even while the circle's total number of slots grows. This addresses the representational obstruction OIC-T25, which applied to Fourier coefficients independent of \(n\).

## OIC-T28: Sufficient gap for polynomial-temperature Fourier localization

Consider

\[
\ell_n(z)=a+\tau(n)\sum_{j=1}^{J}c_j\bigl(\cos(2\pi jz/n)-1\bigr),
\quad \tau(n)=1+\rho\Bigl(\bigl(n/n_0\bigr)^2-1\Bigr),
\]

where \(0<\rho\le1\), \(c_j\ge0\), and \(c_1>0\). Since \(\tau(n)\ge\rho n^2/n_0^2\), the marked-versus-neighbor gap satisfies

\[
\boxed{\ell_n(0)-\ell_n(1)\ge\frac{8\rho c_1}{n_0^2}.}
\]

**Proof.** Every term \(1-\cos(2\pi j/n)\) is nonnegative. Keep the \(j=1\) term and apply the lower bound from T26. \(\square\)

The actual OIC-18 implementation learns both \(c_1\) and \(c_2\) without a nonnegativity constraint, so this theorem is a **conditional sufficient criterion**, to be checked against fitted coefficients. A learned negative coefficient can invalidate the bound. The higher-harmonic equal-capacity control replaces \(j=2\) with \(j=3\) while keeping coefficients \(n\)-independent, and therefore continues to have an \(O(n^{-2})\) adjacent-phase logit gap under a fixed coefficient bound.

## Identifiability and resource scope

The oracle's hidden transition is still defined by a phase permutation. The learned observer carries **two** five-offset stochastic circulant kernels, not an exact generator, and stores \(4n\) floating belief-state coordinates per episode (two families by \(n\) phases by two visible bits). The \(n\)-dependent emission uses the input \(n\) and a preassigned functional family. At 32-bit precision this is \(128n\) live belief bits before counting the program, parameter storage, intermediate buffers, public probe/readout history, and actual work.

None of T26–T28 implies successful family identification after arbitrary noisy readings: the learned stochastic transition kernels, finite data, posterior normalization, and sensor likelihood all matter. A consistent permutation of hidden labels, including a non-dihedral permutation, conjugates the hidden transition and translates the emission marker without changing the public history law; such a relabeling is a gauge transformation. The permitted family-orientation reflection from OIC-T23 persists under the phase-localized emission.
