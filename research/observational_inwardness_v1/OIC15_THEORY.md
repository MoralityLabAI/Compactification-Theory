# OIC-15: Partial observability, residual phase entropy, and sensor-likelihood margins

**Status:** finite elementary proofs, integer tests, and an executed CPU toy experiment. This note extends OIC-01–14 and specializes classical belief-state inference, active automata identification and error-correcting code distance to the already defined OIC state machine. It makes no claim of a new general observability theorem, a neural consciousness result, or a physical cosmology test.

## Interface and assumptions

For a known integer `n≥2` let `S=Z₂ × Zₙ`. A model family `a ∈ {0,1}` is fixed for an episode. With a binary intervention `u_t`, the hidden state and observable state evolve as

\[
z_{t+1}=z_t+a\pmod n,\qquad
x_{t+1}=x_t\oplus1\oplus\bigl(u_t\,[z_t=0]\bigr).
\]

The evaluator sees the complete intervention and raw-state histories and is told `n`. Its initial phase `z0` and model family `a` remain unknown. For the finite results use a uniform phase prior and balanced family prior when reporting Bayesian risks. In the synthetic training data, family labels are available **only as training targets**; no model receives the true phase as an input. Initial visible state `x0=0` is known and shared.

We distinguish two statistical sensor models. The *audit-event* filter derives \(b_t=1\oplus x_t\oplus x_{t+1}\) and treats audit mismatches as independent. The *raw-state* filter scores the observed raw state samples directly, including explicitly marked erasures. The ideal nearest-codeword decoder handles a separate **adversarial bounded-fault** model and has no learned parameters.

## OIC-T18. Dynamical-law identifiability with persistent phase entropy

After `n` consecutive probes `u_t=1`, each rotating phase yields a pulse at exactly one time, while the fixed family yields either zero pulses (`z0≠0`) or `n` pulses (`z0=0`). The family is therefore identified exactly for every possible initial hidden phase for `n≥2`.

Now condition on the fixed family and a nonzero starting phase `z0`. Its phase remains nonzero for all time. Consequently *every future action sequence* generates exactly the same visible trace for all `n−1` such phases. Under the uniform phase prior, the posterior after observing the all-zero pulse sequence is uniform on `n−1` phases, hence

\[
\boxed{H(Z_0\mid A=0,\text{all-zero probe response})=\log_2(n-1)\text{ bits}.}
\]

No additional observations through this interface can reduce that conditional uncertainty. Therefore complete model-family identification is compatible with permanent hidden-state non-identifiability. QED.

## OIC-T19. Exact Bayes-optimal clean family identification

For `T` active probes with uniform phase and equal family priors, the minimal binary classification error is

\[
R_T=\begin{cases}\frac12,&T=0,1,\\\frac{n-T}{2n},&2\le T\le n,\\0,&T\ge n.\end{cases}
\]

This restates OIC-T04 as the correct target for the zero-phase-input observer. Continuous probing observes the whole hidden-zero indicator stream and therefore simulates any alternative active/passive policy. The result follows by counting the overlap of the fixed and rotating model pulse-word distributions. In particular the balanced passive risk is `1/2`. A classifier showing reliable advantage on strictly passive traces under the declared balanced sampling contract indicates input leakage or a protocol change. QED.

## OIC-T20. Finite mismatch penalties have a phase-count ceiling

Consider the event-based cyclic likelihood filter with an arbitrary *finite* mismatch penalty \(\beta>0\). Give each phase equal weight and each mismatched audit event a multiplicative likelihood \(q=e^{-\beta}\). Suppose `T=n` probes produce the one-hot pulse sequence that the rotating family predicts for its true initial phase.

Among rotating hypotheses, one phase matches perfectly and `n−1` others differ in two positions, giving

\[
L_1=1+(n-1)q^2.
\]

Among fixed hypotheses, `n−1` nonzero phases predict no pulses (one mismatch each), and the zero phase predicts all ones (n−1 mismatches), yielding

\[
L_0=(n-1)q+q^{n-1}.
\]

The equal-prior filter chooses the correct rotating family exactly when

\[
\boxed{1-(n-1)q(1-q)-q^{n-1}>0.}
\]

For every fixed finite \(\beta\), there is a finite `n` beyond which this inequality fails because the negative term grows linearly in `n`. Thus a soft phase marginalizer with fixed mismatch penalty cannot preserve the ideal zero-error `n`-probe guarantee *uniformly over phase count*. This is a model-architecture statement, distinct from the perfect logical decoder's behavior. The exact finite test is implemented in `oic15_math.py`.

The ten learned event-filter penalties \(\beta\) span approximately 6.355 to 6.412. Evaluating the exact inequality yields a first one-hot misclassification between `n=578` and `n=611` across those fits. This is an **analytic extrapolation from stored parameters**, not a 600-phase neural rollout. The finite model's `n=7,8` transfer results remain valid within their tested scope. QED.

## OIC-T21. Sufficient raw-likelihood margin for bounded sensor faults

Consider two raw-state codebooks `C0,C1` for a fixed intervention word, each containing `n` possible trajectories. Let

\[
d_{\min}=\min_{a\in C_0,b\in C_1}d_H(a,b).
\]

Suppose the observed raw sequence has at most `e` substitutions and `r` explicitly marked erasures. The correct family's true candidate has Hamming cost on nonerased positions at most `e`. Every candidate from the incorrect family has cost at least `d_min−e−r` by the triangle inequality.

A raw-state cyclic filter assigns each candidate likelihood \(e^{-\beta d}\), with the same finite penalty \(\beta>0\) and uniform initial phase. The total correct-family likelihood is at least \(e^{-\beta e}\), while the incorrect-family sum is at most \(n e^{-\beta(d_{\min}-e-r)}\). Hence the sufficient strict inequality

\[
\boxed{\beta(d_{\min}-2e-r)>\log n}
\]

ensures the correct family receives higher posterior likelihood for **every** eligible initial phase and adversarial error placement. QED.

For even `n`, OIC-T08 supplies an intervention schedule consisting of `n` active probes and `2e+r` passive samples, with \(d_{\min}=2e+r+1\). At this sharp horizon, the sufficient condition becomes simply

\[
\boxed{\beta>\log n.}
\]

Every OIC-15 raw-filter fit has \(\beta\ge 5.85\), which exceeds \(\log 8\approx 2.079\), so the theorem predicts all recorded single-flip and single-erasure successes for the even phase counts `4,6,8`. It also identifies a clear *scale limit*: a fixed finite penalty does not justify universal extrapolation across unbounded `n`.

The exact nearest-codeword decoder follows the more general bound \(d_{\min}\ge 2e+r+1\) without the extra soft marginalization requirement.

## OIC-T22. Raw-state noise creates dependent audit errors

Let a raw reading error be \(\widetilde x_t=x_t\oplus f_t\). Derived pulse signals satisfy

\[
\boxed{\widetilde b_t\oplus b_t=f_t\oplus f_{t+1}.}
\]

A single interior corrupted raw reading changes two neighboring derived audit symbols. The event filter's independent-event noise assumption is therefore misspecified for raw-state substitutions. For the finite model and the registered adversarial faults, direct raw-likelihood filtering empirically recovers the sharp error-correction guarantee, while event-only likelihood filtering degrades. The observed error rates are experiment-specific; this identity explains the sensor-level dependency, and OIC-T21 supplies the robustness guarantee for the raw decoder. QED.

## Mathematical lineage

These results draw on established belief-state inference and automata identification. The OIC construction gives a narrow, exactly testable separation between *what a controller can identify* and *which latent states remain in the same observational fibre*.

- V. Krishnamurthy, *Partially Observed Markov Decision Processes: From Filtering to Controlled Sensing*, Cambridge University Press, 2016. [POMDP belief-state chapter](https://doi.org/10.1017/CBO9781316471104.010)
- Y. Chang, A. Garcia, Z. Wang, and L. Sun, *Structural Estimation of Partially Observable Markov Decision Processes*, IEEE Transactions on Automatic Control 68(8), 2023. [Paper](https://openreview.net/pdf?id=v2j4drBBBq)
- See also the repository's OIC-T01–17 finite derivations and [6GW-T01](../../6GW/THEORY.md) for intervention-preserving quotients.

**Evidence status:** proof arguments are written; seven elementary finite math tests pass. These statements have not been compiled with a Lean kernel. They concern explicit synthetic automata and a declared sensing interface.
