# SMC-11: Exact feature-identity stability in a pretrained recurrent model

**Registered November 15, 2026.** A mathematical follow-up to merged [SMC-09](SMC09_RESULTS.md) and [SMC-10](SMC10_RESULTS.md). Pure proof: [MathCan OLM-11](https://github.com/patrickdugan/MathCan/blob/research/olm11-sae-selection-cells-20261115/research/orbit_lock/OLM11_SELECTION_CELL_THEOREM.md).

## Question

SMC-10 certifies that an originally selected SAE decoder direction causes a greater or smaller output effect than chosen controls for **every** exact-real goal-token anchor and fixed direction within a specified small L2 uncertainty neighborhood. A previously unresolved assumption was whether the *SAE itself would select that same active atom* after changing the hidden state.

For the fitted 48→96 top-four ReLU SAE, the encoder scores are affine in the **physical, unstandardized 48-dimensional recurrent hidden state**:

\[
a_i(h)= b_i+\sum_{\ell=1}^{48} W_{i\ell}(h_\ell-\mu_\ell)/\sigma_\ell.
\]

Here the scale/mean are computed only from stage-one training puzzle states and then frozen, as are the learned encoder/decoder weights.

The highest positive atom determines the SMC-07–10 active decoder intervention direction. The full four-atom active set determines the deterministic inactive-feature sampler's available complement.

## Exact selection-cell theorem and computational certificate

For the unique highest positive preactivation t, define the exact top-one stability radius as the minimum distance to a score-zero hyperplane or to a hyperplane where its score equals another neuron's score. Similarly, for the four positive selected indices S, the exact top-four support stability radius is the minimum of their zero-crossing distances and all selected-versus-unselected equality crossings.

[MathCan OLM-11](https://github.com/patrickdugan/MathCan/blob/research/olm11-sae-selection-cells-20261115/research/orbit_lock/OLM11_SELECTION_CELL_THEOREM.md) proves these sharp Euclidean distances. This experiment uses its [exact Fraction oracle](selection_cells_exact.py), copied without modification into the standalone applied repository. Every stored fitted FP32 weight, physical hidden-state anchor, mean and scale is interpreted as its exact binary rational value.

Each squared boundary radius is therefore an **exact rational**, obtained by comparing positive score margins and exact squared Euclidean row-normal norms. An integer-square-root calculation also gives a proven dyadic radius bracket. An admissible closed L2 uncertainty budget \(\eta\) is certified by comparing \(\eta^2<\rho^2\) using rational arithmetic; the strict inequality is essential because a tie may appear at the boundary.

The actual Torch FP32 nominal top-one and top-four identities are cross-checked independently. Any exact-vs-FP32 identity disagreement or strict-tie failure receives an abstention label. The theorem concerns the exact-real fitted affine SAE; it does not certify arbitrary rounded FP32 inference arithmetic.

## Source lock and heldout evaluation

- Independently pretrained 44,001-parameter third-party TRM-inspired Sudoku recurrent model, checkpoint source commit f0406195fece4d0db29ced28f3a69d1993d95b1e.
- Exact checked pretrained weight SHA-256: 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1.
- The fixed 64-puzzle roster SHA-256: eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611.
- The unchanged 32 train / 16 validation / 16 previously inspected heldout puzzle split. Each puzzle's 81 tokens remain within one cohort.
- Seeds 71,89,113 each refit the exact SMC-07 top-four SAE with 300 optimizer updates and validation-only checkpoint selection.
- Compute exact top-one and top-four sharp squared radii for the **first missing goal token in each of the sixteen heldout puzzles**. That is 48 exact finite selection-cell audits.
- Separately compute a descriptive **FP64 approximation** of top-one and top-four stability-radius distributions over all 16×81=1296 test tokens per seed. These additional token-level values are an exploratory distribution, not an exact-rational proof for every token.

The compact train/validation/test data are the same as prior SMC-07–10, and these states have already been inspected. Thus SMC-11 is a mathematical audit and a stronger specification of the intervention policy, rather than an independent test of model generalization.

## Composing with the verified SMC-10C causal effects

[The original full SMC-10C Arb local-Lipschitz receipt](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43713063188/artifacts/14161327923) contains 480 source-locked contexts: 16 puzzles × 3 SAE seeds × 2 finite intervention radii × 5 common L2 uncertainty budgets. The full receipt SHA-256 for its JSON content is

    176078f22967482420e08a20aa602b7c45a87074e2ab2b600848f2214d9e20aa

The hosted SMC-11 CI downloads that exact artifact and validates the SHA-256 before running. It then joins each prior 48-puzzle/seed selection cell to its ten matching radius/budget cases by **exact seed and puzzle ID**.

- For a previously certified active-versus-isotropic comparison, if the top-one selection guard covers the declared anchor budget, the OLM-11/OLM-10 composition proves the result for **the actual dynamically reselected highest SAE decoder atom**, throughout the admitted hidden-state neighborhood. The isotropic control remains the frozen seeded direction.
- For the inactive-decoder-atom control, demand the stronger top-four support guard. Its deterministic selected-inactive complement is then unchanged, and the originally certified comparator continues to apply.
- For the original numerical Jacobian-optimal control, the top-one guard suffices **only for comparison against that fixed stored optimal direction**. It does not certify recomputing a new singular vector at the perturbed hidden state.
- If either the prior causal ordering is inconclusive, the nominal Torch identities disagree, or the exact selection guard fails, the composed certificate abstains rather than extrapolating beyond its premises.

The full result contains the **exact squared selection radii**, their dyadic lower brackets, selected feature identities and all 480 joined robust effect comparisons. This is stricter than merely observing that features often remain stable under random perturbation samples.

## Reproduction

CPU-only Python 3.11 and pinned PyTorch 2.7.1, with the original 181147-byte source-hashed checkpoint:

    python -m unittest discover -s research/real_recurrence_sae_v1 -p 'test_smc11_selection.py' -v
    python research/real_recurrence_sae_v1/smc11_selection.py --fetch --max-test-puzzles 2 --seeds 71 --steps 300 --out runs/smc11/pilot.json
    python research/real_recurrence_sae_v1/smc11_selection.py --fetch --max-test-puzzles 16 --seeds 71,89,113 --steps 300 --smc10-receipt path/to/lipschitz-full.json --out runs/smc11/full.json

The [dedicated hosted workflow](../../.github/workflows/smc11-selection.yml) downloads the preverified SMC-10C receipt from its source GitHub Actions run, checks that byte-level hash and then archives the entire exact selection/composition result. It cannot silently accept a newly generated or modified causal result.

**Evaluation boundary.** The result concerns the exact-real fitted SAE's discrete feature identity as a function of one physical goal-token hidden state, and the exact-real teacher's previously proved effect ordering. It does not verify feature semantics, accuracy improvement, robust retraining, weight perturbations or machine-level FP32/FP64 rounding throughout the uncertainty ball.

## SMC-11C: Single-fit provenance strengthens the causal composition

The original SMC-11 recipe joined a newly refitted SAE's selection-cell output to a **previously archived SMC-10C causal receipt**. Its SHA-256 lock prevents changing that receipt, but cannot prove the new SAE fit produces the same binary weights as the one used in SMC-10C. CPU environment differences can cause small optimizer variations despite fixed seeds.

The stronger [same-fit joint evaluator](smc11_joint_causal.py) resolves this by fitting the SAE **once per seed** and using its in-memory FP32 encoder, decoder, mean, and scale for both the exact-rational selection-cell proof and the Arb-validated local-Lipschitz causal comparison. The evaluator serializes the weights-only SAE model and frozen normalization, verifies a strict reload round trip, and archives a canonical SHA-256 fingerprint of the exact tensor bytes. Source teacher weights, 64-puzzle roster, goal-token sites, trained SAE hyperparameters and original causal intervention radii remain frozen.

The joint composition requires the OLM-11 positive top-one cell for the active SAE decoder policy, plus the top-four support cell when the selected comparator is a deterministically sampled inactive SAE feature. It keeps the original Jacobian-optimal and isotropic controls fixed. Every failure of a strict guard produces an abstention, and no heldout observation participates in SAE fitting.

The joint [CI workflow](../../.github/workflows/smc11-same-fit.yml) runs standalone tests plus a two-puzzle pilot and a complete 16-puzzle × three-fit evaluation, archiving the exact-rational feature radii, 160-bit Arb causal intervals and serialized SAE weights in the same artifact. For scientific claims about dynamically reselected SAE features, **the joint result is the preferred proof**, with the older artifact-to-artifact join retained for historical replication.

### Signed decoder-ray extension

The [exact directional extension](https://github.com/patrickdugan/MathCan/pull/82) separately asks whether the SAE would retain its top feature after the *finite intervention itself*. Its sharp positive/negative ray crossing radius is obtained by minimizing exact rational squared scores along the selected direction. The direction must match the **FP64 normalized dyadic vector frozen by the causal evaluator**. The SMC-09/10 intervention normalizes that vector again with Arb, preserving its exact-real ray. Using the unnormalized FP32 decoder×scale product would differ infinitesimally and cannot substitute for this exact provenance condition. A passing ray guard at radius 0.8 is a point-on-ray identity theorem, independent of the uniform η-ball robust causal certificate at fixed intervention radius.
