# OIC-16: Observable equivalence, projection, and phase-resolution limits

**Date:** November 10, 2026. **Evidence:** exact finite proofs, independently verified Python model receipts and CPU-trained toy systems. These are specialized applications of finite-state identifiability, group actions, minimum-distance decoding, and Fourier smoothness. No Lean kernel certificate or empirical claim about mental experience, physical cosmology, or frontier model internals.

## Declared system and interface

Let \(n\ge 2\). The true hidden state is \(z\in\mathbb Z_n\), the visible bit is \(x\in\mathbb Z_2\), and the family \(a\in\{0,1\}\) fixes the hidden transition:

\[
 z_{t+1}=z_t+a\pmod n,\qquad
 x_{t+1}=x_t\oplus1\oplus\bigl(u_t[z_t=0]\bigr).
\]

The experimenter receives the known \(n\), the public intervention word \(u\), and the raw visible readings \(x\). At inference it receives neither \(z_0\) nor \(a\). Initial hidden phase is uniform within each family; the family prior is balanced. Family labels are provided for supervised *training* only.

The learned observer carries \(2n\) family/phase hypotheses and a two-valued visible state. It learns two five-offset circulant transition kernels with offsets \(D=\{-2,-1,0,+1,+2\}\), a shared Fourier emission function, and an observation mismatch scale:

\[
 K_a=\sum_{d\in D}p_{a,d}S^d,\quad
 p_{a,d}=\operatorname{softmax}(\theta_{a,:})_d,
\]

\[
 g_n(z)=\sigma\Bigl(c_0+c_1\cos\frac{2\pi z}{n}
                       +c_2\cos\frac{4\pi z}{n}\Bigr).
\]

There are fourteen learned scalar parameters. Known structural priors include the number of model families, the circulant kernel form, the five allowed offsets, and a cyclic emission geometry. At test time \(n\) determines the size of the state hypothesis array.

## OIC-T23. Phase-reflection gauge invariance

Define \(R:\mathbb Z_n\to\mathbb Z_n\) by \(R(z)=-z\). Then

\[
R(z+1)=R(z)-1,
\quad [z=0]=[R(z)=0].
\]

Consequently, an initial phase \(z_0\) under forward rotation \(z\mapsto z+1\) and its reflection \(-z_0\) under reverse rotation \(z\mapsto z-1\) produce identical pulse indicators at all times, for *every externally selected intervention word*. They therefore produce identical raw visible traces. Since reflection permutes the uniform initial-phase prior, both generators have identical marginalized likelihoods over raw traces:

\[
\boxed{\mathcal L_{+1}(u,x)=\mathcal L_{-1}(u,x).}
\]

The orientation is unidentifiable under this interface. This is stronger than a numerical ambiguity: it is an exact isomorphism of the observational models. A phase-anchoring intervention, additional sensor, or non-reflection-invariant initial-phase prior may break it. QED.

## OIC-T24. Exact transfer after deterministic cyclic projection

For each learned kernel, choose its maximum-weight offset \(\widehat d_a\), resolving ties by a declared fixed rule. Select an emission marker \(\widehat m\in\mathbb Z_n\) by maximizing the learned emission score over the n phases, again with a fixed tie rule. Define the **projected** model

\[
 \widehat F_a(z)=z+\widehat d_a\pmod n,
 \qquad \widehat e(z)=[z=\widehat m].
\]

**Proposition.** If \(\widehat d_0=0\), \(\widehat d_1\in\{-1,+1\}\), and exactly one phase is selected as the emission marker, then the projected fixed and rotating families have *exactly the same sets of raw visible traces* as the target families for every \(n\ge2\), every action word \(u\), and every initial visible bit.

**Proof.** In the fixed model, the selected hidden phase remains constant; every initial phase either always generates the active pulse or never generates it. The resulting family of visible histories is independent of the marker's label. In the rotating model, translation \(z\mapsto z-\widehat m\) removes the marker offset. If the learned shift is \(+1\), the translated process is identical to the target rotation. If it is \(-1\), compose the translation with reflection \(z\mapsto-z\) and apply OIC-T23. Both transformations bijectively permute the set of initial phases. Hence the entire codebook, including corrupted-readout distances computed on its raw traces, is preserved. QED.

For **even** \(n\), the OIC-T08 minimum raw-codebook distance under \(n\) active probes followed by \(T-n\) passive transitions is \(T-n+1\). Thus the exact minimum-distance decoder using the projected codebook corrects up to \(e\) substitutions and \(r\) marked erasures whenever

\[
 \boxed{T\ge n+2e+r.}
\]

This inherits an existing theorem through an explicit hypothesis-class projection. A learned stochastic kernel on its own does not satisfy the guarantee. The inference-time decoder now uses hard-coded deterministic cyclic transitions and a one-hot emission, which are material additional structural assumptions.

## OIC-T25. A Fourier phase-resolution obstruction

Consider the OIC-16 emission logit

\[
\ell_n(z)=c_0+c_1\cos\frac{2\pi z}{n}
                 +c_2\cos\frac{4\pi z}{n}.
\]

Its zero-phase versus adjacent-phase separation is

\[
\Delta_n=\ell_n(0)-\ell_n(1)
=c_1\Bigl(1-\cos\frac{2\pi}{n}\Bigr)
+c_2\Bigl(1-\cos\frac{4\pi}{n}\Bigr).
\]

Using \(1-\cos t\le t^2/2\),

\[
\boxed{\lvert\Delta_n\rvert\le\frac{2\pi^2|c_1|+8\pi^2|c_2|}{n^2}.}
\]

Therefore with coefficients fixed across all phase counts, neighboring phase logits converge together at order \(n^{-2}\). If a classifier requires a fixed positive separation \(\ell_n(0)\ge m\) and \(\ell_n(1)\le -m\) for \(m>0\), then

\[
2\pi^2|c_1|+8\pi^2|c_2|\ge2mn^2.
\]

Thus coefficients of at least quadratic magnitude in n (or an explicitly n-dependent localization mechanism) are necessary to sustain such a margin for unbounded n. The smooth detector trained at n=4–6 has no such scaling rule. QED.

This is a precise representational bottleneck, not a lower bound for arbitrary neural emissions. Highly localized kernels, an n-dependent temperature, higher Fourier modes, discrete phase-code outputs, or extra physical observations change the hypothesis class.

## Statistical and proof scope

The completed CPU experiment evaluates n=4,5,6 in training and n=7,8 as the **registered** phase-count transfer. Values for n=9,10,12,16,20,24,32 are **post-hoc stress tests** and exploratory. The posterior reproducer checks model predictions from archived weights without torch. A successful fitted posterior supports claims about the tested synthetic population and training distribution; it does not certify exact physical dynamics, a unique latent coordinate system, or future performance under arbitrary sensor corruption.

Companion files: [OIC16_REPORT.md](OIC16_REPORT.md), [oic16_model.py](oic16_model.py), [oic16_projection.py](oic16_projection.py), [oic16_verify.py](oic16_verify.py), [oic16_scale.py](oic16_scale.py).
