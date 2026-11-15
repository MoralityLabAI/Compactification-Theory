# SMC-03: Learnable relational probes and exact capacity frontiers

Date: November 15, 2026. Continuation of [SMC-01](README.md) and [SMC-02](TRANSPORT_LIFT_RESULTS.md) with a **CPU-trained** experiment. Pure mathematics is in [MathCan OLM-03](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM03_FINITE_MEMORY_QUOTIENT.md).

The source pair remains OpenAI Math [#131](https://github.com/openai/math/blob/main/lean/docs/131.md), polynomial mixing of degree-preserving graph switches, and [#238](https://github.com/openai/math/blob/main/lean/docs/238.md), optimal logarithmic mixing of Thorp permutations. The synthetic lift is an independently specified transposition Markov chain with a matching-switch quotient; it is **not the original Thorp transition kernel**.

## 1. Registered question and exact labels

At six vertices, there are 720 permutations and 15 perfect matchings. A permutation S induces matching \(X=SM_0\).

**Positive class:** S uniform on the chosen split's permutation cohort, X its induced matching.

**Negative class:** draw S from the identical permutation cohort and draw X independently and uniformly among the 15 matchings. This null has the **same graph and permutation marginals** as the positive class, exactly within each train and test cohort.

Class priors are each 1/2 when measuring balanced accuracy. The negative process includes one compatible pair in 15, producing an irreducible classification overlap. The Bayes-optimal population balanced accuracy is

\[
A^*=\frac12+\frac12\left(1-\frac1{15}\right)=\frac{29}{30}\approx96.67\%.
\]

The exact witness checks \(X=SM_0\). A matching-only or permutation-only classifier has optimal accuracy 50%. A population-log-loss-minimizing additive logit of the two coordinate feature sets also stays at zero (its gradient there vanishes and its logistic risk is convex). This latter property concerns the log-loss optimum, **not** every possible additive 0-1 decision threshold.

## 2. Pinned holdout and input features

A SHA-256 sort of the 48 permutations in each matching's stabilizer fiber assigns 36 to training and **12 unseen permutations** to testing. The split is fixed independently of model training, contains each of the 15 matching classes in both cohorts, and has no overlapping permutation IDs.

* **Training:** 540 positive examples and the full 540×15=8,100 negative input combinations, sampled in equal-prior class minibatches.
* **Testing:** 180 positives and all 180×15=2,700 negatives, evaluated exhaustively.
* **Raw neural input:** 15 binary incidence bits (one for each of the 15 possible vertex edges) plus a flattened 6×6 permutation matrix (36 bits). Total 51 features.
* **Raw MLP:** \`51 -> 48 GELU -> 48 GELU -> 1\`. PyTorch AdamW with lr 0.008, weight decay 0.0001, 300 steps, batch of 256 positives and 256 negatives per step.
* **Oracle-assisted relational control:** computes the integer overlap of the observed matching with the reference matching *transported by the permutation*. A trainable one-input logistic readout learns a threshold using AdamW lr 0.08 and the same minibatch indices. The transported overlap is structural information supplied to the baseline; it is not learned autonomously by an SAE.
* **Compute:** Python 3.13.5, PyTorch 2.10.0+cpu, two CPU threads, no model download and no GPU. Fixed seeds 20261115, 20261116, 20261117.

## 3. Executed three-seed results

Saved machine-readable records: [smc03_cpu_3seed_results.json](smc03_cpu_3seed_results.json).

| Probe / seed | Held-out balanced accuracy | Gauge-transformed held-out accuracy | Prediction agreement under gauge relabel |
|---|---:|---:|---:|
| Raw MLP / 20261115 | 95.72% | 96.02% | 98.44% |
| Raw MLP / 20261116 | 93.91% | 96.15% | 96.35% |
| Raw MLP / 20261117 | 95.67% | 95.54% | 97.94% |
| Oracle-assisted overlap logistic / all three | 96.67% | exact compatibility control | 100% for the Boolean oracle |
| Marginal-only population optimum | 50.00% | 50.00% | N/A |
| Bayes-optimal relational classifier | 96.67% | 96.67% | 100% |

The raw MLP's mean held-out balanced accuracy is **95.10%**. It approaches the finite-support Bayes optimum while generalizing to previously unobserved permutation representatives *inside known matching orbits*. This is narrower than holding out entire matching classes, vertex counts, or new graph families.

For the gauge control, the same fixed vertex transposition (0,2) is applied simultaneously to graph labels and to permutation outputs. The exact compatibility predicate is unchanged. The learned raw MLP's **binary decisions** agree between original and transformed examples at the measured rates; this test is descriptive, since the transformed examples may map to permutation representatives included in training. It is not an independently held-out intervention cohort.

The findings provide evidence that a small synthetic neural network can approximate an algebraic joint-state relationship from raw finite inputs. They do not establish that pretrained transformers encode the same mode or that the network has learned a principled gauge-equivariant internal representation.

## 4. Exact memory frontier for an alignment representation

MathCan OLM-03 proves that a compressed m-state matching label with at most r discrete encoder outputs retains, at most,

\[
D_{\max}=1-\frac{(r-s)a^2+s(a+1)^2}{m^2},
\quad m=ar+s,\quad 0\le s<r.
\]

Balanced deterministic partitions attain the maximum, even against stochastic encoders. The corresponding Bayes balanced accuracy for detecting coupled versus independent pairs is \((1+D_{\max})/2\). Entire-matching recovery from the r-code representation is at most r/m.

With m=15:

| Discrete codes | Binary bits | Best joint TV | Best balanced accuracy |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 50.00% |
| 2 | 1 | 112/225 | 74.89% |
| 3 | 2 | 2/3 | 83.33% |
| 4 | 2 | 56/75 | 87.33% |
| 5 | 3 | 4/5 | 90.00% |
| 8 | 3 | 196/225 | 93.56% |
| 15 | 4 | 14/15 | 96.67% |

The task-relative quotient of a full permutation \(S\in S_{2k}\) to its induced matching \(SM_0\) is exactly \((2k-1)!!\) classes, corresponding to a stabilizer of size \(2^k k!\). Thus an exact six-vertex compatibility monitor needs only 4 bits for the matching quotient rather than 10 bits for a complete permutation ID.

The same MathCan note establishes a complementary **two-dimensional circle embedding** of matching identities. Its exact equality detector is bilinear, but its worst-case angular margin scales as \(m^{-2}\), requiring roughly \(2\log_2m\) fractional precision bits for robust fixed-point separation under the stated sufficient bound. A fixed-dimensional representation can therefore become precision-hungry as the number of matching classes grows.

## 5. Finite-horizon one-bit capacity inversion

In the four-vertex graph-switch chain, the full graph/permutation relational TV at time t is

\[
D_{\mathrm{full}}(t)=\frac23(1-16^{-t}),
\]

which strictly increases with t. With **one bit** of matching state, the optimal TV at time t is instead

\[
\boxed{D_{\mathrm{1bit}}(t)=\frac49(1+4^{-t}-2\cdot16^{-t}).}
\]

It increases from 0 at t=0 to **1/2 at t=1**, then decreases toward **4/9** at equilibrium. Thus the optimal one-bit balanced detector accuracy peaks at 75% after one switch, and approaches 13/18 ≈ 72.22% despite the underlying process continuing to mix.

The general finite-horizon capacity for nonuniform matching law p is a weighted partition collision problem,

\[
D_{\max}(p,r)=1-\min_h\sum_{z=1}^r\left(\sum_{x:h(x)=z}p_x\right)^2.
\]

For r=2, the optimizer is an exact subset sum as close to 1/2 as possible. MathCan [weighted_frontier.py](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/weighted_frontier.py) executes this certificate. At six vertices the first switch also attains the 1/2 one-bit dependence ceiling.

**Protocol implication:** compare joint probes at matched representation precision and loop depth. A low-bit bottleneck can alter the apparent information trend even while the full-information TV distance increases.

## 6. Reproduce and boundaries

Exact standard-library checks:

    python -m unittest discover -s research/synchronized_mixing_v1 -p 'test_*.py' -v
    python research/synchronized_mixing_v1/relational_probe.py

Trained CPU runs (requires \`torch==2.10.0\`):

    python research/synchronized_mixing_v1/relational_probe.py --train --seed 20261115 --steps 300
    python research/synchronized_mixing_v1/relational_probe.py --train --seed 20261116 --steps 300
    python research/synchronized_mixing_v1/relational_probe.py --train --seed 20261117 --steps 300

The exact CI workflow runs pure-Python tests, archives the cohort-control JSON, and retains the SMC-01/02 Markov traces. The heavier CPU model training was executed locally and its three-seed measurements are committed with full reproducibility parameters.

**Further gates:** held-out matching families and vertex counts; an actual learned SAE dictionary extracted from a source-pinned recurrent model; intervention transfer between hidden layers; a causal control proving probe reliance on discovered latent alignment rather than only synthetic input geometry. All of those remain untested.


## 7. Stricter SMC-03b: complete matching-identity holdout

The previous 540/180 permutation split tests generalization to new *representatives* within all 15 familiar matching orbits. To measure genuinely new matching IDs, we fixed a second split **before training**:

* 12 of 15 matching classes, with all 48 permutations in each class, form training (576 permutations).
* The remaining three matching classes, again all 48 representatives each, form testing (144 permutations).
* The negative class in each cohort independently samples the graph matching uniformly **within that cohort's matching classes**. Consequently graph and permutation marginals still match exactly between positive and negative labels within each split. Train and test have different graph supports by design.
* Raw 51-input MLP, structured one-scalar oracle-assisted readout, optimizer rates, minibatches, training steps and three seeds are unchanged from SMC-03.

The test population now has three equiprobable matching classes, so its exact Bayes-optimal balanced accuracy is \(1-1/(2\cdot3)=5/6\approx83.33\%\). The training Bayes ceiling is \(1-1/(2\cdot12)=23/24\approx95.83\%\).

| Seed | Raw MLP train | Raw MLP on unseen matching IDs | Oracle-assisted structured probe on unseen IDs |
|---|---:|---:|---:|
| 20261115 | 94.22% | **55.90%** | 83.33% |
| 20261116 | 95.78% | **51.62%** | 83.33% |
| 20261117 | 95.33% | **50.46%** | 83.33% |

Reproduce:

    python research/synchronized_mixing_v1/family_holdout.py --seed 20261115
    python research/synchronized_mixing_v1/family_holdout.py --seed 20261116
    python research/synchronized_mixing_v1/family_holdout.py --seed 20261117

[Fixed split and training code](family_holdout.py) | [Raw measurements](smc03_family_holdout_results.json) | [Exact split controls](test_family_holdout.py).

**Interpretation.** The same small raw network learned the relationship sufficiently to classify unseen representatives of *known* matching classes, yet failed to extrapolate it across entirely new matching classes. The structured overlap readout transported the reference matching and generalized perfectly at the population Bayes limit. The result identifies a concrete inductive-bias bottleneck for future compact recurrent-model and SAE experiments. The class-exclusive holdout remains within one synthetic six-vertex topology and is not evidence about unrelated graph types.

A useful next gate is to train an explicitly equivariant network, then compare it with the raw MLP using the exact same class-disjoint split and equal compute. The matchings and the model's permutation-index conventions should remain host-controlled to rule out information leakage through preprocessing.

