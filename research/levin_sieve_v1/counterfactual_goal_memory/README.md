# LSV-CGM-01: latent morphogenetic goals and network cuts

**Date:** 2026-11-07. **Evidence:** a seeded sample of three Levinverse transcripts, exact finite-state mathematics, rational min-cut calculations, complete nine-site enumeration, and CPU tests. **Scope:** a mathematical proxy. No model calibration on biological measurements, tumor data, or measured bioelectric voltages.

## Random source sample and research anchors

Sampled **three** transcript files from **228** eligible `.txt` files under [patrickdugan/levinverse](https://github.com/patrickdugan/levinverse/tree/main/transcripts/permitted). Eligible size: 12,000 through 120,000 bytes. Sort file paths, FNV-1a-mix the ASCII string `levinverse|2026-11-07|talk-math-3` starting at `0x1a2b3c4d`, then run 32-bit xorshift (`x ^= x<<13; x ^= x>>>17; x ^= x<<5`) Fisher–Yates on path indices, taking the first three. Initial shuffled RNG state: `0x76308a48`. Sample is over text files, not a deduplicated census of distinct talks. The chosen paths and catalog-declared hashes are in [sources.json](sources.json).

| Selected talk | Transcript evidence | Model connection |
|---|---|---|
| [Evolution and Intelligence: inversion and a positive feedback spiral](https://thoughtforms-life.aipodcast.ing/evolution-and-intelligence-inversion-and-a-positive-feedback-spiral-by-michael-levin/), publisher auto-transcribed, 2026-01-15 | Slide 9 at **10m28s**: goal attainment by different means; Slide 12 at **18m52s**: latent planarian pattern expressed after injury | Same anatomy can contain distinct future target memories |
| [Michael Levin: What Is Bioelectrical Intelligence? Technological Approach To Minds Everywhere](https://podcasters.spotify.com/pod/show/mindbodysolution/episodes/Michael-Levin-What-Is-Bioelectrical-Intelligence--Technological-Approach-To-Minds-Everywhere-e27bq96), local Whisper ASR, unverified | **~28m20s**: collective goal; **~31m**: bioelectric coupling; **~53m58s**: pattern reprogramming | Communication topology defines which sites can share a goal within a horizon |
| [Diverse Intelligence: Origins, Possibilities, and Diseases of Minds](https://thoughtforms-life.aipodcast.ing/diverse-intelligence-new-frontier-for-origins-possibilities-and-diseases-of-minds-by-m-levin/), publisher auto-transcribed, 2026-07-10 | Slide 27 at **37m38s**: competing tissue pattern signals; Slide 33 at **42m04s** and Slide 34 at **43m25s**: collective goal scale and disconnection | Threshold for a programmed tissue region to overcome neighbors; sham edge intervention |

The biology descriptions above are claims made in the sampled talks and subject to the transcript's stated quality. The mathematical objects and calculations below are independent research extensions. This work makes no claim that cancer or regeneration is explained by an Ising lattice.

## Finite tissue model

Let `G=(V,E)` be a finite graph of switchable sites with spins `s_i \in \{-1,+1\}`. Exterior normal tissue is clamped at `-1`. Each site `i` has `b_i` fixed exterior connections. Coupling strength is `J>0`. A nonempty programmed region `P` receives a field `h` toward `+1`, while sites outside `P` receive an anchoring field `a\ge0` toward `-1`. A flipped set `S` has relative energy

\[
\Delta E_h(S)=2J\left(|\partial_E S|+\sum_{i\in S}b_i\right)
+2a|S\setminus P|-2h|S\cap P|. \tag{1}
\]

A cut removes an existing internal coupling edge; it does not create an exterior ground edge. The memory bit `m\in\{0,1\}` controls the effective programmed field `h=m h_0`. An observed morphology is a global energy-minimizing spin pattern. This equilibrium assumption omits developmental kinetics and electrophysiology.

### T1. Exact critical field

For nonempty `P`,

\[
\boxed{ h_c(G)=\min_{S:S\cap P\ne\varnothing}
\frac{J\left(|\partial_E S|+\sum_{i\in S}b_i\right)+a|S\setminus P|}
{|S\cap P|}. }\tag{2}
\]

For `h<h_c`, the baseline is a global minimizer. For `h>h_c`, some nonbaseline state has lower energy. At equality a nonbaseline coalition ties the baseline. For the ground-clamped lattices here, the baseline is unique below threshold.

**Proof:** if `S\cap P` is empty, every contribution to (1) is nonnegative. Otherwise divide (1) by `2|S\cap P|` and compare `h` with the displayed ratio. Taking the finite minimum over all coalitions establishes the trichotomy. With grounded boundary edges every nonempty set has a strictly positive cost when `h<h_c`. QED.

Checking only the programmed patch gives the upper bound `h_c\le J(|\partial_E P|+\sum_{i\in P}b_i)/|P|`. A larger coalition can sometimes have a lower ratio, motivating a global optimizer.

### T2. Reduction to an exact rational min-cut

Put `z_i=1` for flipped sites and define

\[
w_i=Jb_i+a\mathbf1_{i\notin P}-h\mathbf1_{i\in P},
\qquad
\Delta E_h(S)/2=J|\partial_E S|+\sum_{i\in S}w_i. \tag{3}
\]

Give each internal undirected edge symmetric capacity `J`. Add `source\to i` capacity `-w_i` for `w_i<0`, or `i\to sink` capacity `w_i` for `w_i>0`. If `C=\sum_i\max(0,-w_i)`, every source-side set `S` has cut capacity `C+\Delta E_h(S)/2`. The max-flow/min-cut theorem therefore certifies the optimum energy as `2(\mathrm{mincut}-C)`. The implementation uses exact `Fraction` capacities and an Edmonds–Karp search. This is a classical submodular binary-energy reduction.

### T3. Effects of cutting communication

Deleting internal edges weakly decreases the boundary cost of every fixed coalition. Therefore

\[
h_c(G\setminus F)\le h_c(G). \tag{4}
\]

For any local synchronous message-passing process whose new state depends only on a vertex and its graph neighbors, after `T` rounds an output at vertex `v` can depend only on

\[
C_T(v)=\{u:\operatorname{dist}_G(u,v)\le T\}. \tag{5}
\]

**Proof:** the statement is immediate at `T=0`; replacing inputs by one-hop neighbors expands possible dependencies by at most one graph edge at each round. QED. This is an information-support bound, not a measure of consciousness or clinical pathology.

### T4. Counterfactual memory witness

If `h_c(G_{\rm intact})>h_0>h_c(G_{\rm cut})`, then memory states `m=0` and `m=1` share the baseline resting morphology on the intact graph. Following the specified cut, `m=1` has a strictly favorable alternative. The intervention makes otherwise latent target information observable from equilibrium morphology. Uniqueness of the new state needs a separate check.

## Verified exact experiment

The test graph is a `3\times3` four-neighbor lattice, exterior grounded, center programmed, `J=a=1` and `h_0=3`. The targeted cut removes center edges `(1,4)` and `(3,4)`. The sham removes equally many distant edges `(0,1)` and `(7,8)`.

| Intervention | `h_c` | `+1` ground-state sites at `h=3` | Relative optimum energy | `|C_1(4)|`, `|C_2(4)|` |
|---|---:|---|---:|---|
| intact | 4 | none | 0 | 5, 9 |
| two center edges cut | 2 | center only | -2 | 3, 6 |
| sham distant edges cut | 4 | none | 0 | 5, 9 |

With `m=0`, all three conditions retain the ordinary baseline. Each of the 512 states was checked by a direct Ising Hamiltonian. An independent rational max-flow solver agrees with exhaustive optimization. All subsets of `k=0,1,2,3,4` center-incident edge cuts satisfy `h_c=4-k` in this example. A nine-by-nine lattice was checked by min-cut without full-state enumeration.

The stored field of strength three remains latent in the intact lattice. Cutting two specified edges makes the center fate energetically favorable, while the sham remains normal. The numbers depend on deliberately chosen dimensionless parameters and are not predictions for planarian or mammalian tissues.

## How to reproduce

From repository root, with Python 3.10+ and no dependencies:

```bash
python3 -m unittest discover -s research/levin_sieve_v1/counterfactual_goal_memory -p 'test_territory.py' -v
python3 research/levin_sieve_v1/counterfactual_goal_memory/territory.py
```

[results.json](results.json) pins the reproducible nine-site receipt. Ten tests cover a fully independent energy oracle, all mask configurations, threshold ties, latent and expressed memory, graph cones, every center-cut subset, rational min-cut on a complete three-node graph topology family, the nine-by-nine case, and contract validation.

## Falsification and transfer

Measure coupling structure and perturbation-response trajectories before fitting this proxy to a biological system; pre-register topology cuts, sham cuts and held-out specimens. A discriminatory experiment would attempt to decode a latent goal from bioelectric signals while visible anatomy is held fixed, then predict distinct repair after controlled lesions. Distinct microphysical models may share the same resting morphology, so the intervention protocol matters. Future work should include kinetic barriers and multi-attractor transitions.

This experiment complements [LSV-01 asynchronous repair](../THEOREMS.md) and [QO-01 goal-visible quotient learning](../../ingression_dynamics_v1/GOAL_AWARE_RATE_DISTORTION.md). Its additional mathematical subject is an exact equilibrium reprogramming threshold with a min-cut certificate, not a new general theorem or measured biological mechanism.
