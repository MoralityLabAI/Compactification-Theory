# SMC-04: Unknown-gauge transport identification under class holdout and label noise

**November 15, 2026.** Fourth stage of [SMC-01/02/03](SMC03_LEARNED_PROBES.md). Independent [OLM-04 pure-math derivation](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM04_TRANSPORT_IDENTIFICATION.md).

## Question

SMC-03b found that a 4,897-parameter raw MLP essentially failed to generalize to three entirely unseen six-vertex matching identities. SMC-03c closed the synthetic gap using a tiny equivariant edge scorer supplied with the correct graph transport operation. This continuation makes both the **reference matching** and **output-frame alignment** hidden, then infers them from calibration pairs.

The motivating mathematical pair remains OpenAI Math [#131](https://github.com/openai/math/blob/main/lean/docs/131.md), polynomial mixing of degree-preserving graph switches, and [#238](https://github.com/openai/math/blob/main/lean/docs/238.md), optimal mixing of full permutations. The present six-vertex walk, inference procedure and probability bounds are separately defined finite objects. We do not apply either paper's asymptotic mixing constants.

## Model and scope of prior structure

The learner knows that each observed matching G arises from an input permutation S according to

\[
\boxed{G=\sigma S B,}
\]

for an unknown global frame permutation σ∈S₆ and an unknown reference matching B among the 15 perfect matchings.

The candidate family therefore contains 720 × 15 = **10,800** models. The learner receives only observed pairs (S,G). The hidden parameters are host-owned by the fixture and omitted from the inference functions. A single clean pair determines B uniquely for each candidate σ, leaving exactly 720 alignments to examine. Under corrupted pairs, the learner searches all 10,800 candidates and maximizes exact-match count. This is maximum-likelihood for the specified independent uniform replacement-noise model.

**What is supplied:** six-vertex permutation group, matching catalogue and group-action formula. **What is learned:** the specific hidden frame and reference matching. The experiment does not discover the algebraic hypothesis family from unstructured activations and does not train a neural alignment model.

## Holdout and error model

Three host-generated hidden models use fixed seeds 43, 77 and 91. For each, we split all 720 permutations into 576 calibration-compatible permutations whose *observed* graph belongs to 12 selected matching classes and 144 evaluation permutations whose observed graph belongs to the other three classes. Each class contains exactly 48 permutation representatives, so the split is completely matching-class-disjoint.

Each trial samples calibration permutations independently with replacement from its training cohort. Under corruption ε, the graph label is replaced with an independent uniform draw from all 15 matchings with probability ε. We test ε=0, 0.1 and 0.25.

Heldout positives use the true graph for all 144 withheld permutations. Negatives use the same permutation distribution and an independently sampled graph uniformly from the three heldout matching classes. The positive/negative marginal laws therefore match exactly. The Bayes-optimal balanced accuracy for the correct recovered relation is 5/6 (83.33%).

The training-cohort input support depends on the hidden model. This is an observational, truth-conditioned sampling protocol. A separate independent-input information lower bound must not be applied to it. The solver itself uses each input only as part of its observed (S,G) pair and does not consult the host secret or the cohort construction.

## Exact group-action identification certificate

All 10,800 hypotheses were exhaustively compared against the 720 possible input permutations. The independent algebraic risk proof is in [MathCan OLM-04](https://github.com/patrickdugan/MathCan/blob/research/olm01-orbit-lock-dual-mixing-20261115/research/orbit_lock/OLM04_TRANSPORT_IDENTIFICATION.md).

For a false model h, let α_h be its agreement probability on iid calibration inputs from the relevant cohort. Then

\[
\mathbb E[\text{false survivors after }N]=\sum_{h\ne h_*}\alpha_h^N.
\]

For the full uniform permutation domain, the maximum false agreement is 7/15. On the restricted 576-input calibration cohort it is 1/2. Exactly one hypothesis agrees with all calibration-cohort inputs.

| Clean calibration pairs | Expected false candidates | Upper bound on nonunique recovery |
|---:|---:|---:|
| 1 | 719 | 1 |
| 4 | ~1.485 | 1 |
| 6 | ~0.2094 | 0.2094 |
| 8 | ~0.04816 | 0.04816 |
| 10 | ~0.01182 | 0.01182 |
| 12 | ~0.002938 | 0.002938 |

Under noise ε, the exact likelihood uses scores equal to the number of matching predictions. For a false candidate with agreement α, a Chernoff argument yields pairwise error at most

\[
\left[1-(1-\alpha)
\left(\sqrt{1-\epsilon+\epsilon/15}-\sqrt{\epsilon/15}\right)^2\right]^N.
\]

Summing the individual finite candidate bounds at ε=1/4 gives a rigorous error upper bound of approximately **0.005219 after N=24** and **0.000351 after N=32** under the specified iid input/noise distribution.

These are failure-probability bounds for finite candidate identification, independent of the empirical success counts below.

## Executed CPU campaign: 180 trajectories per corruption rate

Each of the three true hidden models was evaluated using 60 independent, nested calibration trajectories. For a given trial, all calibration sizes are prefixes of the same randomized sequence. This makes the cleaner learning curve comparable across depths. Full numeric record: [smc04_calibration_180trial.json](smc04_calibration_180trial.json).

| Calibration examples N | Correct / 180, clean | Correct / 180, 10% corruption | Correct / 180, 25% corruption |
|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 5 | 5 | 5 |
| 3 | 87 | 57 | 37 |
| 4 | 146 | 107 | 58 |
| 5 | 170 | 148 | 91 |
| 6 | 175 | 164 | 118 |
| 8 | 178 | 176 | 150 |
| 10 | 180 | 180 | 166 |
| 12 | 180 | 180 | 174 |
| 16 | 180 | 180 | 179 |
| 24 | 180 | 180 | 180 |

All 180 clean trajectories identified the exact hidden frame and reference matching by ten calibration examples, and all 180 quarter-corrupted trajectories did so by 24. When the exact hidden model is selected, the heldout balanced accuracy is 5/6 on completely unseen graph matching identities. The empirical success counts are conditional on the three chosen fixture models and their secret-conditioned input cohorts; they are not claimed to characterize every possible hidden transformation without further sampling.

## Matched-marginal negative control

To check calibration leakage, retain the same 576-input train cohort but independently resample each calibration graph uniformly from the 12 train matching classes. This destroys the graph–permutation pairing while preserving the cohort's graph and permutation marginals.

Across 60 such trials with 12 calibration pairs, the learned candidate attains a mean heldout balanced accuracy of 659/1296 ≈ **50.85%**, near chance. The null distribution includes some individual high-scoring candidates by coincidence. This control supports the interpretation that the positive-pair relationship supplies the identification signal.

## Gauge metamorphic controls

For a simultaneous global relabeling of observed graph outputs by τ, the hidden frame changes from σ to τσ. Any exact-fit candidate (σ,B) maps bijectively to (τσ,B). Every maximum-consensus score is identical under the transformed data.

Regression tests check this relation both for exact version spaces and for corrupted calibration scores. The group action is treated as a known structural symmetry; recovering an equivalent frame under that symmetry is the precise inference contract.

## Reproduce, verification and next gates

Pure standard-library code, with no torch or downloads:

    python -m unittest discover -s research/synchronized_mixing_v1 -p 'test_identify_transport.py' -v
    python research/synchronized_mixing_v1/run_smc04.py --smoke --output smc04-smoke-report.json
    python research/synchronized_mixing_v1/run_smc04.py --output smc04-full-report.json

The full replay emits the exact all-hypothesis agreement histograms, seed-conditioned empirical results, Chernoff error guarantees, negative-control metrics, and SHA256 hashes of the inference and replay scripts. CI reruns the nine exact tests and a smaller deterministic campaign and archives its output. The 180-trial summary is committed alongside the source.

**Next experiment:** infer transport from sparse, noisy learned feature incidences where the permutation representation is itself uncertain. A structured soft-assignment or synchronization estimator can compete with this exact finite oracle on unseen matching classes and different vertex counts. To claim a mechanistic-interpretability finding, the eventual test must use source-pinned model activations and causal interventions, rather than synthetic graph labels.

**Boundaries:** The present result is exact finite-group parameter recovery with a known candidate family. It does not establish efficient recovery as k grows, automatic discovery of the right group, a trained neural alignment module, or enhanced reasoning ability in looped transformers.
