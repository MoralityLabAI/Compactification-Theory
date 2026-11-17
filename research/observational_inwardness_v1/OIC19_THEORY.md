# OIC-19: Observable generator gauge and predictive-state recovery

**Date:** 2026-11-10. **Evidence:** elementary exact proofs, exhaustive integer finite-state checks, and a separate CPU-trained toy experiment. This analysis describes a specific observable hidden-cycle interface; it makes no statement about cosmological consciousness, subjective experience, or pretrained transformer internal states. No Lean kernel certificate is claimed.

## Interface and model class

Let \(n\ge2\), with hidden phase \(z_t\in\mathbb Z_n\), known action \(u_t\in\{0,1\}\), and visible state \(x_t\in\{0,1\}\):

\[
z_{t+1}=z_t+d\pmod n,\qquad
x_{t+1}=x_t\oplus1\oplus (u_t[z_t=0]).
\]

The initial visible bit is known and initial hidden phase \(z_0\) is uniformly distributed over \(\mathbb Z_n\), independent of the action policy. The observer receives raw visible readings and the actions it selects. The shift \(d\) is an unknown, deterministic constant in this theorem. The previously studied two-family problem has \(d=0\) (fixed phase) and \(d=+1\) (rotating phase).

## OIC-T26. Complete observational equivalence of constant shifts

**Theorem.** For every \(n\ge2\), two shifts \(d,d'\in\mathbb Z\) induce identical distributions over visible histories under every fixed or adaptive action policy if and only if

\[
\boxed{\gcd(n,d)=\gcd(n,d').}
\]

**Proof.** Put \(g=\gcd(n,d)\) and \(L=n/g\). The permutation \(z\mapsto z+d\) has \(g\) disjoint cycles of length \(L\). Exactly one cycle contains the marked state zero, and it has \(L\) starting phases. Initial phases in the other cycles never emit a pulse. Since \(z_0\) is uniform, the probability that a pulse ever occurs under continuous probing is \(L/n=1/g\).

Among the \(L\) phases in the marked cycle, there is exactly one initial phase giving its first pulse at any specified offset \(0\le j<L\). Its probability is \(1/n\), and subsequent pulses recur every \(L\) transitions. Thus the complete unmasked pulse-stream distribution consists of each of these \(L\) periodic sequences with probability \(1/n\), plus the silent sequence with probability \(1-L/n\). It depends only on \(L\), equivalently on \(g\).

A controller selecting actions adaptively sees a deterministic function of the pulse stream and its previous choices. Identical pulse-stream distributions therefore induce identical visible-history distributions for every controller policy. This proves sufficiency.

Conversely, continuous probing for \(n\) transitions reveals whether any pulse occurred. Its probability is \(1/g\). If \(\gcd(n,d)\ne\gcd(n,d')\), these probabilities differ. The corresponding visible-history distributions are distinguishable. QED.

**Corollary 26A (unit-step gauge).** Every \(d\) coprime to \(n\) is observationally equivalent to \(+1\) and \(-1\). For prime \(n\), all nonzero shifts belong to one observational class. In particular, \(+1\) and \(+2\) on \(\mathbb Z_5\) cannot be distinguished by this interface, even with unlimited adaptive observations and exact noiseless sensing.

**Corollary 26B (limits of generator recovery).** With an equal prior on two shifts belonging to the same gcd class, the minimum Bayes error for identifying the shift is \(1/2\) under every policy and horizon. Learning an exact labeled generator requires additional assumptions or observations. Gauge-relative recovery is the natural identifiable target. A stochastic transition law requires separate analysis because the theorem assumes a fixed deterministic shift.

**Executable evidence.** `oic19_identifiability.py` enumerates all hidden initial phases using exact integer arithmetic and rational probabilities. `test_oic19_identifiability.py` checks every gcd class through \(n=12\) on many action words and independently checks pulse probabilities through \(n=29\). Six tests pass with and without Python assertions. These tests accompany the analytic proof and are not a substitute for a Lean certificate.

## The OIC-19 training objectives

The continuous learned model inherits the OIC-18 architecture: a 13-parameter von Mises-style emission with concentration proportional to \(n^2\), two five-offset row-stochastic circulant kernels, and a learnable raw-sample mismatch rate. It maintains \(4n\) FP32 belief cells for the two families, phases, and visible states. The phase count \(n\), action history, and raw sensor history are known at inference.

A labeled family likelihood \(\mathcal L_{\mathrm{family}}\) trains the two-family posterior. OIC-19 compares it with:

- A selected entropy penalty \(\lambda\sum_a H(K_a)\) on the learned shift kernels (without hidden-phase labels);
- A conditional observable predictive loss \(-\frac1T\sum_t\log P_\theta(x_{t+1}\mid x_{0:t},u_{0:t},a)\), using the *supervised family* label \(a\) and public trajectory only;
- A ceiling with an additional simulator phase-label loss, evaluated modulo the forward/reversed phase gauge, with nonzero fixed phases scored as a common observational fibre.

The predictive objective favors a model that explains observed transitions over a sequence of probe events. It supplies no direct simulator phase coordinates. Whether minimizing this objective recovers a generator representative is an empirical, architecture-relative claim. OIC-T26 shows why the underlying observation law alone cannot select a unique numerical step among all units of \(\mathbb Z_n\).

## Conditioning and sensor robustness

The observation-channel derivations OIC-T06 and OIC-T08 give the exact Hamming-distance correction criterion for an explicit deterministic codebook. A learned stochastic filter with fitted nonzero sensor noise has no automatic minimum-distance guarantee. OIC-19's raw-sensor fault audit therefore reports *all* initial phases and fault positions; even extremely high mean accuracy can leave an adversarial counterexample.

The OIC-19 results apply to five-offset kernels learned on phase counts \(n=4,5,6\), with evaluation through \(n=64\). The allowed shift set is a strong architectural prior: for some training phase counts, shifts of order smaller than \(n\) can be observationally distinguished from generators, which helps select the \(\pm1\) representative. OIC-T26 remains the general identifiability statement.
