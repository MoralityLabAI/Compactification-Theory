# SMC-11: Exact SAE selection geometry and same-fit causal guarantees

**Completed November 15, 2026.** Continues the source-locked recurrent SAE investigations SMC-07 through SMC-10. The mathematical derivations are in [MathCan OLM-11](https://github.com/patrickdugan/MathCan/pull/79) and [OLM-11D](https://github.com/patrickdugan/MathCan/pull/82), both merged.

The full same-fit source-locked experiment passed in [GitHub Actions run #43721281348](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43721281348). Its [raw 16-puzzle/3-SAE-seed artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43721281348/artifacts/14164917026) includes 480 detailed exact-real causal contexts, 48 exact selection-cell records, an independent auditor receipt and three **weights-only frozen SAE state files**. The [permanently committed result summary](SMC11_CERTIFIED_SUMMARY.json) records their byte-level SHA-256 hashes.

## 1. Source and same-fit boundary

The original model is an independently pretrained 44,001-parameter TRM-inspired Sudoku network with four iterations of a shared recurrent block on 81 tokens and 48 hidden channels. Upstream commit: f0406195fece4d0db29ced28f3a69d1993d95b1e. Source weight SHA-256: 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1. Fixed roster SHA-256: eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611.

The same 64 Sudoku puzzles are grouped by puzzle ID into 32 train, 16 validation and **16 previously inspected heldout test puzzles**. All 81 tokens of a puzzle remain within its cohort.

An SAE trained with hard top-four ReLU encoding, architecture 48→96→48, uses stage-one training hidden states only, an unchanged 300-optimizer-update budget, validation checkpoint selection and frozen seeds 71, 89 and 113. Source teacher weights remain fixed.

**The critical provenance refinement:** SMC-11C fits each SAE once and uses the identical in-memory FP32 encoder, decoder, mean and scale for both the exact feature identity calculation and the Arb-validated causal comparison. Its weights-only snapshot is reloaded and compared tensor-for-tensor before publication. Earlier experiments using separately refitted SAEs can differ by small FP32 changes even when seeds match. The final combined certificate is based on this **single-source-fit** protocol.

## 2. Exact SAE selection-cell mathematics

For the physically scaled 48-dimensional hidden state h, the fitted encoder preactivation is affine,

\[
a_i(h)=b_i+\sum_{\ell=1}^{48} W_{i\ell}\frac{h_\ell-\mu_\ell}{\sigma_\ell}.
\]

The top-one and top-four feature identities remain unchanged inside their strict polyhedral selection cells. Their sharp Euclidean stability radii are the minimum distances to positive-to-zero and selected/unselected equal-score hyperplanes, calculated from **exact rational squared margins** using the precise stored binary tensor values.

Every goal-token case passed an exact-vs-Torch nominal identity cross-check and the strict-selection certificate.

| Certified physical L2 radius over 48 test puzzle/SAE pairs | Minimum | Conventional median | Maximum |
|---|---:|---:|---:|
| Highest positive feature identity | 0.03279965 | 0.79775546 | 2.74081752 |
| Entire top-four support | 0.01105420 | 0.15663992 | 0.54784895 |

All 48 source cases pass both strict selection-identity guards for every shared hidden-state L2 uncertainty budget used in SMC-10: zero through **10^-6**. Thus the causal bound's tightness, rather than uncertainty in reselected feature identity, limits the current combined certificate.

## 3. Stronger directional result: positive decoder rays

OLM-11D measures the first exact feature-switching boundary **along the chosen decoder intervention ray**, rather than in the worst possible hidden-state direction. For any affine score-margin constraint with initial margin g and negative slope c along raw direction d, the first squared crossing radius is

\[
r_{\mathrm{cross}}^2=\frac{g^2\|d\|_2^2}{c^2}.
\]

Its minimum over the relevant constraints gives the sharp first crossing; if every constraint is nondecreasing, the first positive-ray crossing radius is infinite.

The ray computation uses exactly the **FP64-normalized dyadic decoder direction frozen by SMC-09**, including the original FP32 decoder×scale multiplication and subsequent FP64 normalization. A unit test checks equality of all 48 direction components against the SMC-09 helper.

| Finite intervention amplitude | Positive decoder ray: highest feature unchanged | Positive decoder ray: full top-four unchanged | Negative decoder ray: highest feature unchanged |
|---:|---:|---:|---:|
| 0.02 | 48/48 | 48/48 | 48/48 |
| 0.2 | 48/48 | 47/48 | 45/48 |
| 0.8 | 48/48 | 43/48 | 34/48 |

**All 48 positive decoder rays have an infinite highest-feature stability radius.** Every fitted top neuron has nondecreasing positive score and nondecreasing advantage over every competitor along its own selected physical decoder ray, as evaluated by exact rational score slopes. This is a property of the fitted **affine SAE**, and it does not imply causal benefit, stable full top-four support or source-model output ordering at arbitrary intervention amplitudes.

The signed directional result and the worst-direction Euclidean ball result solve different problems. The causal proof below uses small **anchor-neighborhood balls** around a fixed intervention amplitude, not a proof about output effects over the entire infinite ray.

## 4. The dynamically reselected SAE retains the verified causal effect ordering

SMC-10C proved causal sensitivity rankings for fixed SAE decoder directions uniformly over shared L2 uncertainty balls around the goal-token state and reference directions. SMC-11C now also proves that the SAE **would dynamically select the same highest feature throughout those anchor balls** and therefore use exactly the same physical decoder direction.

For the seeded inactive-atom comparator, the stricter top-four support guard preserves its inactive-index complement. For the isotropic and Jacobian singular-vector controls, the comparison is to their originally **fixed** vectors. It does not claim the numerical top singular vector would remain the same if recomputed at a perturbed hidden state.

| Intervention radius | Shared anchor/direction L2 uncertainty | Active SAE > isotropic | Reverse ordering | Original causal inconclusive | Additional selection abstentions |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 10^-8 | 38/48 | 2/48 | 8/48 | 0 |
| 0.2 | 10^-8 | 43/48 | 4/48 | 1/48 | 0 |
| 0.2 | 10^-6 | 20/48 | 0/48 | 28/48 | 0 |

All results use 160-bit outward-rounded Arb local-Lipschitz enclosures of the exact-real frozen source recurrence, at the same predeclared goal-token site. The source model has other 80 token states fixed. This is not a claim about the rounded Torch implementation's exact hardware behavior.

## 5. Independent artifact audit

The [receipt auditor](audit_smc11_joint.py), executed after the hosted evaluation, rechecked every declared radius, source identity guard, policy composition decision and frozen snapshot SHA-256.

| Independent check | Verified |
|---|---:|
| Exact feature-selection goal-token cases | 48 |
| Arb causal goal-token contexts | 480 |
| Causal control decision labels | 1,440 |
| Frozen SAE snapshot files with verified SHA-256 | 3 |
| Selection guard inconsistencies or extra abstentions | 0 |
| Source/seed/puzzle/radius/budget provenance mismatches | 0 |

Raw JSON SHA-256: 7caafcb54e36f508b49a082892b689789a751f86b52d6d206e955e3782431087. Archive ZIP SHA-256: 1b62914ba7fecd91614eab36c1f799e2fc0d6d4e3279d0e1ac03a0ee5ee2a2fb. The auditor JSON and all three saved SAE tensor files are preserved in the same Actions artifact. A source-independent regression suite tests that modified model hashes, unsupported perturbation budgets and falsified causal labels are rejected.

## Scientific interpretation and next gate

The learned SAE's active decoder directions have a **certified feature-selection stability structure**, and the original source-locked causal comparisons now hold for the actual reselected-feature policy over the small tested hidden-state neighborhoods. The infinite self-selection ray is a property of the SAE's affine score geometry, not a proof of a semantic reasoning mode or capability gain.

The 16 Sudoku test puzzle IDs have been inspected repeatedly since SMC-07. This stage strengthens exact mathematical provenance and causal verification on those finite cases. Independent confirmation requires unseen puzzle families, independently trained recurrent checkpoints and causal hypotheses about what the learned features encode.

The next mathematically tractable refinement is to characterize the **polyhedral cone of all decoder directions that reinforce their own selected feature**, then certify a nonzero angular neighborhood around each selected direction. That may connect positive-ray identity invariance to uncertainty in the decoder itself, rather than only perturbations of its input hidden state.
