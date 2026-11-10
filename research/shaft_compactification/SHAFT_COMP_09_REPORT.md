# SHAFT-COMP-09: Noisy and incomplete anchor observations

**Date:** 2026-11-10. **Evidence class:** exhaustive finite combinatorics, five conditional Lean 4 theorems, and 1,024-trial-per-condition seeded simulation. All policies and interfaces are synthetic; the study contains no real-device operations, hidden model deployment, or evidence of frontier-model objective survival.

## Research question

The preceding SHAFT-COMP-08 benchmark identified the four-query semantic mapping between two trained toy agents using three *exact*, paired task responses. We now ask how many paired behavioral anchors suffice when the destination's responses are partially missing or corrupted, and whether the choice of anchor tasks matters.

The specified gauge family is the 48-element finite set G = S4 × C2: permutations of four query coordinates, with an optional global binary output inversion. Each task is a four-bit truth table. A paired anchor provides the four responses of a source task and the corresponding destination responses, with the task ID hidden from the gauge-recovery procedure. The source signature is **assumed exact** in SHAFT-COMP-09. Destination observations are independently erased with probability m (missing completely at random) and independently flipped, when observed, with probability p < 1/2.

This is a finite *channel coding and identifiability* problem conditional on known tasks and paired anchors. Its simulation involves no trained neural model; the previously measured trained-GRU test in SHAFT-COMP-08 motivates the abstraction.

## 1. Exact geometry and optimized anchor selection

Let t_g ∈ {0,1}^{4n} be the expected concatenated destination signature for gauge g across a fixed set of n anchors. The clean codeword distance from the true gauge g* to alternative g is

d_g = Hamming(t_g, t_{g*}),    d_min = min_{g≠g*} d_g.

Search all C(16,3)=560 triples of **distinct** task IDs. The maximum possible d_min for any three-anchor selection is exactly **4**, achieved by eight triples. The first is (3,5,6), and the full list is in [noisy_gauge_reference.json](results/noisy_gauge_reference.json).

| Anchor budget n | One-hot 1,2,4 cycle: d_min | Balanced 3,5,6,9,10,12 cycle: d_min |
|---:|---:|---:|
| 3 | 2 | 4 |
| 6 | 4 | 8 |
| 12 | 8 | 16 |
| 24 | 16 | 32 |

The six balanced IDs are the six weight-two four-bit truth tables. The sequence repeats when n exceeds its cycle length. The one-hot sequence repeats every three anchors. At these registered budgets, d_min grows as 2n/3 for one-hot and 4n/3 for balanced anchors. The balanced design doubles the minimum separation for this particular candidate family.

Both designs resolve the exact gauge with three clean, fully observed anchors. Increasing separation improves *robustness*, rather than changing noiseless identifiability.

## 2. Deterministic corruption certificate

Let O be the observed positions, E the subset of those positions flipped by observation noise, and let

d_obs = min_{g≠g*} |{i∈O : t_g(i)≠t_{g*}(i)}|,    r=|E|.

For Hamming loss ℓ over observed positions, the difference between the candidate and true losses is

ℓ(g) − ℓ(g*) = |O ∩ D_g| − 2|E ∩ D_g|.

Therefore every wrong gauge has strictly larger loss whenever

**d_obs > 2r.**

A conservative version using the clean minimum distance d_min and at most e erasures is d_min > e + 2r. These certificates are sufficient conditions; identification may succeed when they fail.

The new [NoisyGauge.lean](lean/Shaft/NoisyGauge.lean) formally establishes the strict-score uniqueness and its robust-missingness consequences **assuming the displayed loss-accounting inequalities**. Finite Hamming identity, enumeration of anchor-code distances and the probability bound below are verified by executable Python or mathematical derivation, not formalized inside Lean.

The frozen Monte Carlo receipt includes a posthoc sufficient-certificate count for every condition. It uses the true mapping to compute d_obs, so the certificate is an audit quantity, not an observer's deployable decision rule.

## 3. Conditional Chernoff bound

Fix any incorrect gauge with clean Hamming distance d_g. At a differing position, the score difference favors the true gauge by +1 when observed correctly, favors the wrong gauge by -1 when observed flipped, and contributes zero if missing.

Its moment-generating factor for t>0 is

E[exp(-tY)] = m + (1-m)[(1-p)e^{-t} + p e^t].

Choosing t = (1/2) ln((1-p)/p), and interpreting p=0 as a limit, yields

**Pr[ℓ(g) ≤ ℓ(g*)] ≤ α^{d_g},    α=m+2(1-m)√(p(1-p)).**

A union bound over the 47 alternatives gives

**Pr[any incorrect gauge ties or beats g*] ≤ min(1, Σ_{g≠g*} α^{d_g}) ≤ min(1,47 α^{d_min}).**

The result assumes independent erasures/flips across anchor-query observations, known clean source-anchor signatures, a fixed true gauge and the specified binary noise channel. It also assumes inference by observed-bit Hamming maximum likelihood, appropriate when p<1/2 and all observed bit errors have identical probability.

If d_min(n)≥δn and α∈(0,1), a sufficient sample budget for error at most η is

n ≥ ln(47/η)/(δ · [-ln α]).

For example, at p=0.1 and m=0.25, α=0.7. With η=0.01, the worst-case bound is satisfied at **n≥36** for the repeated one-hot design, versus **n≥18** for the balanced design. These are conservative sufficient budgets, derived from the group union bound. Empirical success can require fewer anchors.

## 4. Seeded finite experiments

[noisy_gauge.py](noisy_gauge.py) enumerates all 48 gauge codewords, applies independent random erasure/bit-flip noise to the destination's anchor observations, then chooses the gauge with minimum observed Hamming loss. Ties are broken lexicographically, with tie frequency reported separately. The source reference bits are exact.

- Two anchor designs: one-hot and balanced.
- Four budgets: 3, 6, 12, 24 paired task observations.
- Four target bit-flip rates: p = 0%, 5%, 10%, 20%.
- Three independently missing observation rates: m = 0%, 25%, 50%.
- Exactly 1,024 seed-locked trials per design/noise/missingness condition, evaluated at all four budgets.
- Complete 96-cell receipt includes exact geometry, success and unique-winner counts, posthoc sufficient certificates, posterior entropy, and the finite-distance Chernoff bound.
- There are 24 design/channel conditions and 98,304 total inference evaluations. This is a synthetic Monte Carlo result. An empirical 1,024/1,024 rate remains a finite sample outcome.

### Correct mapping recovered, out of 1,024 trials

| Channel | Anchors | One-hot | Balanced |
|---|---:|---:|---:|
| p=5%, m=25% | 3 | 801 | 857 |
|  | 6 | 970 | 1,015 |
|  | 12 | 1,020 | 1,024 |
|  | 24 | 1,024 | 1,024 |
| p=10%, m=25% | 3 | 646 | 707 |
|  | 6 | 855 | 960 |
|  | 12 | 993 | 1,021 |
|  | 24 | 1,023 | 1,024 |
| p=20%, m=50% | 3 | 216 | 241 |
|  | 6 | 427 | 470 |
|  | 12 | 680 | 753 |
|  | 24 | 918 | 969 |

At p=10%, m=25%, six anchors recover the true gauge in **83.50%** of one-hot trials (855/1024) and **93.75%** of balanced trials (960/1024). With 12 anchors the observed rates are 96.97% and 99.71%. The Chernoff bound on the probability of any wrong tie or victory at n=12 is 0.20717 for one-hot and 0.05308 for balanced; its conservatism is visible in the measured errors.

At p=20%, m=50%, even 24 anchors have residual ambiguity or errors. The balanced design's observed posterior gauge entropy is 0.243 bits, compared to 0.472 bits for one-hot. Rates from separate seeded trial streams should be treated as descriptive comparisons; replication across independent seeds and paired streams would give a stronger uncertainty assessment.

The experiment reports correct-guess rates, unique-correct rates and tie counts separately because forced lexicographic tie-breaking can otherwise exaggerate confidence.

## 5. Structured-noise counterexamples

The independence assumption carries substantial weight. [structured_missingness.py](structured_missingness.py) implements two exact finite controls:

1. **Fixed unseen coordinates.** If query coordinates 2 and 3 are censored on *every* anchor, two gauges remain indistinguishable at n=3,6,12,24 under both designs. Repeating more tasks leaves that symmetry intact.
2. **Consistent wrong-gauge relabeling.** A global transformation that makes observations perfectly consistent with some different gauge causes the Hamming selector to identify the transformed gauge uniquely. Increasing the number of identical-semantic anchors strengthens its confidence in that interpretation.

These are mathematical observation failures. They show that systematic ambiguity is governed by which distinctions the observer can access, and does not obey the bound derived for independent random noise. The negative controls use in-memory bitstrings, with no covert communication or interference with real monitors.

## 6. Interpretation and next experimental boundary

The key structural object is the **observer-relative quotient under a noisy channel**. As the measured query space becomes sufficiently separating, its ambiguity collapses; when observational symmetries persist, the quotient remains nontrivial even with unlimited repetitions.

For the original SHAFT/Astra control question, this supplies a testable research hypothesis: local behavioral monitors may have persistent symmetries that keep some objective-relevant distinctions inaccessible. Realistic claims require pinned trained models, independent activation capture, perturbation tests, held-out task families, and evaluations against monitors with differing observational privileges.

A valuable next experiment would fit an observer's confidence-calibrated gauge posterior from partially sampled Q5-style recurrent trajectories, then intervene on selected hidden states to test causal sufficiency. Use explicit disjoint task-family splits and monitor false-positive matching before claiming detectability or persistence.

## Reproduction

From repo root (Python standard library only):

\`\`\`bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.noisy_gauge \
  --trials 1024 --out /tmp/shaft09.json
python -m research.shaft_compactification.structured_missingness
\`\`\`

Light CI compares the simulated receipt to [results/noisy_gauge_reference.json](results/noisy_gauge_reference.json), checks both structured failure controls, and compiles the standalone Lean 4.28 package. The receipt contains no learned-model weights or private data.
