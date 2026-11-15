# SMC-02: A Markov lift with permanently hidden correlation

Date: 2026-11-15. Supersedes the **scope** of the first SMC-01 toy experiment, while preserving the original shared-quantile experiment and its separate results. Pure mathematics: [MathCan OLM-02](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM02_COLLISION_SPECTRUM_LIFT.md).

## The stronger experimental contract

SMC-01 previously joined a lazy matching-switch kernel P with a Thorp-*inspired* permutation-sweep kernel Q by a shared quantile. Its dependence survived ten steps, but it lacked an algebraic invariant making it clear how a true graph switch chain and a true permutation Markov walk could remain perfectly synchronized.

SMC-02 introduces a **single** Markov chain on permutations whose exact quotient is a lazy two-edge matching-switch chain. This eliminates the snapshot-versus-trajectory gap of OLM-01:

* Reference matching M0={(0,1),(2,3),...,(2k-2,2k-1)}.
* At every step, independently either hold with probability h or right-multiply the current permutation by one uniformly sampled transposition of vertices from different M0 edges.
* Interpret the observed graph as Gt=Πt M0. For k=2, h=1/2, graph transition probabilities are 1/2 hold and 1/4 for each alternate matching.
* The permutation chain is symmetric and irreducible on S(2k), has a self-loop, and approaches the uniform law. The graph Markov quotient also approaches uniformity.
* The equality Gt=Πt M0 is preserved at every step, for every coin realization.

The source pair is OpenAI Math [#131, polynomial mixing of degree-preserving graph switches](https://github.com/openai/math/blob/main/lean/docs/131.md) and [#238, optimal-order Thorp permutation mixing](https://github.com/openai/math/blob/main/lean/docs/238.md). This finite permutation walk uses its own cross-pair transpositions, **not** the original Thorp shuffle, and does not inherit its quantitative theorem.

## Exact discriminators

Let pt be the law of Gt and qt the law of Πt. Define the positive class Jt=Law(Gt,Πt) and the precisely matched negative U_t=pt⊗qt. The negative uses **the same per-coordinate distributions** as the positive; it destroys only the pairing between graph and permutation labels.

The relational one-bit monitor evaluates

    locked(g, perm) = (g == relabel(perm, M0))

The positive class always passes. The negative passes with probability sum_g pt(g)^2, yielding the *exact optimal* advantage

\[
\operatorname{TV}(J_t,U_t)=1-\sum_g p_t(g)^2.
\]

Every graph-only or permutation-only classifier has zero population advantage between these classes by construction. At t=0, even the joint test has zero advantage because both are deterministic; the advantage increases as the graph marginal randomizes.

For four vertices and h=1/2,

\[
p_t=\left(\frac13+\frac23\,4^{-t},
\frac13-\frac13\,4^{-t},\frac13-\frac13\,4^{-t}\right),
\quad
\operatorname{TV}(p_t,U_G)=\frac23\,4^{-t},
\]

and the exact temporal relational advantage is

\[
\boxed{D_t=\frac23(1-16^{-t}).}
\]

## Exact CPU replay

All values below come from complete rational enumeration of the 24 permutation states and 3 graph states.

| Sweep t | Graph marginal TV | Permutation marginal TV | Joint TV to fixed uniform product | Relational TV to time-matched product |
|---:|---:|---:|---:|---:|
| 0 | 2/3 | 23/24 | 71/72 | 0 |
| 1 | 1/6 | 19/24 | 67/72 | 5/8 |
| 2 | 1/24 | 29/48 | 19/24 | 85/128 |
| 3 | 1/96 | 11/24 | 35/48 | 1365/2048 |
| 4 | 1/384 | 271/768 | 1559/2304 | 21845/32768 |
| 6 | 1/6144 | 2503/12288 | 2/3 | 5592405/8388608 |
| 8 | 1/98304 | 22771/196608 | 2/3 | 1431655765/2147483648 |
| 10 | 1/1572864 | 205903/3145728 | 2/3 | 366503875925/549755813888 |
| 12 | 1/25165824 | 1856971/50331648 | 2/3 | 93824992236885/140737488355328 |
| 16 | 1/6442450944 | 150614371/12884901888 | 2/3 | 6148914691236517205/9223372036854775808 |

The joint's distance from *fixed uniform independent equilibrium* equals exactly 2/3 at sweep 6 and remains there in this finite system. The **time-matched** relational TV increases monotonically to 2/3. These distances use different null distributions, hence their different trajectories.

The exact 24x24 permutation kernel satisfies the degree-five annihilating polynomial

\[
R(R-I)(R-\tfrac14I)(R-\tfrac12I)(R-\tfrac34I)=0.
\]

Trace moments for exponents 0..4 verify eigenvalue multiplicities 0:1, 1/4:8, 1/2:6, 3/4:8, 1:1. The second eigenvalue is 3/4, giving the rigorous bound

\[
\operatorname{TV}(q_t,U_{S_4})\le\frac{\sqrt{23}}2(3/4)^t.
\]

The same code independently verifies exact Markov projection and row-stationarity for six vertices: 720 permutation states and 15 matching states. At stationarity the relational advantage becomes 14/15. Only the small finite cases are enumerated.

## A stronger linear/bilinear monitoring target

MathCan OLM-02 proves a weighted Gram identity for an arbitrary deterministic relationship X=f(S). Under an independent-noise mixture of weight ε at the *same* marginals, all m-1 nonconstant population canonical-correlation singular values equal exactly 1-ε, even when the observed marginal p is nonuniform.

This translates into a precise two-head training probe:

* Extract an SAE feature-incidence matching G and an alignment/permutation state Π from a fixed recurrent layer.
* Pair true (G,Π) samples against cross-sample resampled pairs whose individual distributions are identical. A marginal-only classifier should achieve 1/2 balanced accuracy.
* Train and evaluate a relational probe on the compatibility event. At t, its exact population balanced accuracy on a 50/50 positive/negative split is 1/2 + D_t/2. The ideal t→∞ limit here is 5/6.
* With independent-noise contamination ε, the oracle advantage becomes (1-ε) D_t and the centered correlation spectrum flattens at 1-ε. Vary ε to calibrate robustness.
* Distinguish synthetic label/latent gauges from actual cross-layer model features, and split test families, seeds and alignments before model training.

The toy exact label-map detector is an **upper control**, not evidence that a learned SAE can discover such a map or that recurrent transformers empirically exhibit this coupling. The follow-up is to test that question on source-pinned model states.

## Reproduction

From the repository root:

    python -m unittest discover -s research/synchronized_mixing_v1 -p 'test_*.py' -v
    python research/synchronized_mixing_v1/transport_lift.py

The CI workflow runs both SMC-01 and SMC-02 tests and uploads deterministic JSON traces for both. The new SMC-02 test file contains five families: strong lumpability and stationary measures at 4/6 vertices; an unrelated nonuniform collision identity; the four-slot finite-time curve and optimal detector; exact polynomial/trace spectral verification; and the six-vertex extension.

MathCan holds the separately written pure-math proof and an independent exact-rational Gram oracle. This file records applied finite evidence and a registered path to recurrent-model falsification.


## Exact gauge intervention controls

Treat matching G and permutation Π as two coordinates in the same labeling gauge. A common relabeling by σ sends (G,Π) to (σG,σΠ) and preserves the relation G=ΠM0 with probability **1 at every sweep**. Swapping two vertex labels in the graph **alone**, with Π fixed, preserves the relation exactly when the chosen pair is already an edge of G. For a uniform matching on 2k vertices this occurs with probability 1/(2k-1).

| Matching size | Common relabel success | One-sided transposition at stationarity | One-sided independent full permutation at stationarity |
|---:|---:|---:|---:|
| 4 vertices | 1 | 1/3 | 1/3 |
| 6 vertices | 1 | 1/5 | 1/15 |

For k=2, selecting the base-matching pair (0,1) gives one-sided success 1/3+(2/3)4^-t; selecting (0,2), a base cross pair, gives 1/3-(1/3)4^-t. The common-frame control stays at 1. These interventions distinguish a gauge-consistent relational detector from one that memorizes labels.

The local exact CPU replay confirms the finite-time values and the general k=2,3 stationarity counts. The CI artifact now includes the intervention traces. The corresponding proof is [MathCan OLM-02E](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM02_COLLISION_SPECTRUM_LIFT.md).

## Why the TV plateau is permanent

The checked minimum mass of the permutation law at sweep 6 is **65/4096**, which exceeds 1/72, the uniform-product mass at every joint compatible atom. Hence every compatible atom has nonnegative signed discrepancy from the fixed uniform product. Their union is the optimal TV event and gives distance 2/3. The permutation kernel is doubly stochastic, so the minimum point mass can only increase at successive steps. Consequently the 2/3 plateau holds for every integer sweep t≥6, as proved in MathCan OLM-02F.

