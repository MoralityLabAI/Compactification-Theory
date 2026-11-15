# SMC-10: Source-locked robust neighborhoods of recurrent SAE interventions

**November 15, 2026.** This is a continuation of [SMC-09](SMC09_RESULTS.md), whose 432 strict finite-endpoint comparisons are now extended to small **L2 neighborhoods of both the hidden-state anchor and the selected exact unit intervention directions**. Pure mathematical companion: [MathCan OLM-10](https://github.com/patrickdugan/MathCan/blob/main/research/orbit_lock/OLM10_ROBUST_ENDPOINT_NEIGHBORHOODS.md).

## Source lock and interpretation of the test

The source is the same externally pretrained, 44,001-parameter TRM-inspired Sudoku recurrence used in SMC-07–09, with source commit f0406195fece4d0db29ced28f3a69d1993d95b1e. Pretrained weight SHA-256: 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1. Fixed 64-puzzle roster SHA-256: eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611.

We reuse the same 32 training, 16 validation and **16 previously inspected heldout puzzle IDs**, with all puzzle tokens grouped. An unsupervised top-four SAE is refit from training stage-one hidden states for each seed 71, 89 and 113, using exactly 300 updates and validation-only checkpoint selection. The teacher weights remain unchanged.

For a fixed first-blank goal token, compare the output effect of the highest active SAE decoder direction with one inactive SAE direction, one seeded isotropic direction and the numerically Jacobian-optimal right singular direction. The selected numerical directions are fixed as exact-real, L2-unit-normalized vectors sourced from stored FP64 dyadic components. Both the common 48-channel stage-one anchor and the directions may vary within prescribed L2 distances \(\eta\). The other 80 tokens remain fixed. Intervention radii are exact rationals \(r=1/50\) and \(r=1/5\). The output effect is the L2 norm of the change in all ten final goal logits after three recurrent blocks.

The mathematical assertion for each successful case is **uniform effect ranking over every such admissible exact-real hidden state and intervention direction**, including shared anchor uncertainty. The learning procedure and the choice of directions are frozen before the robustness audit; it does not establish stability of which SAE atom would be selected by refitting.

## Two separately validated methods

**SMC-10A: Direct coordinate-box propagation.** The [Arb evaluator](smc10_robust_endpoint_arb.py) uses outer coordinate boxes of half-width \(\eta\) for the clean goal-token state and \((1+r)\eta\) for the corresponding perturbed token. It propagates them through the entire exact-real recurrence. Squaring centered interval values in LayerNorm is refined by the mathematical fact that exact squares are nonnegative. If interval dependency loss makes LayerNorm or an output effect impossible to enclose finitely, the result is labeled *inconclusive*, with the corresponding error included in the full receipt.

**SMC-10C: Endpoint-local Lipschitz transfer.** The [independent validated implementation](smc10_local_lipschitz.py) starts with SMC-09's exact-real interval enclosures for the three nominal output effects. It uses Arb-LDL verified spectral upper bounds for the pretrained weight tensors and the existing LSV-17 local LayerNorm radius transport to produce \(L_0,L_u,L_v\) as **valid Lipschitz coefficients on separately certified Euclidean balls** around the clean, active and control endpoints. OLM-10D then bounds the difference between the nominal output effect and the effect at any admitted uncertain input:

\[
E_u=L_0\eta + L_u(1+r)\eta,\qquad
\|F(h'+ru')-F(h')\|_2
\in[\max(0,\ell_u-E_u),\,b_u+E_u].
\]

The three clean/active/control endpoint enclosures share the original source anchor but may be widened independently by the proof method. Both approaches are conservative and can abstain. All experiments use 160-bit outward-rounded Arb arithmetic on the exactly represented source weights, and preserve the full proof-relevant per-case output.

## Source-locked full-run results

Each method completed a **16-puzzle × 3-seed × 2-radius × 5-uncertainty-budget** run: 480 contexts, with three comparisons each, for 1,440 categorical certificates or abstentions per method.

- [Full direct-box GitHub Actions run #43712195763](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43712195763), [complete raw JSON artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43712195763/artifacts/14161721672).
- [Full local-Lipschitz GitHub Actions run #43713063188](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43713063188), [complete raw JSON artifact](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43713063188/artifacts/14161327923).
- [Permanent machine-readable comparative summary](SMC10_CERTIFIED_SUMMARY.json) has counts for all three controls, all five budgets and both intervention radii, exact source hashes and ZIP/JSON SHA-256 evidence hashes.

### Active SAE versus isotropic control

Each cell below shows **strictly certified active-greater / control-greater / inconclusive**, among 48 combinations of puzzle and SAE seed.

| Intervention radius | Common anchor and direction L2 uncertainty | Direct Arb boxes | Verified local Lipschitz |
|---:|---:|---:|---:|
| 0.02 | 0 | 44 / 4 / 0 | 44 / 4 / 0 |
| 0.02 | 1e-12 | 0 / 0 / 48 | 44 / 4 / 0 |
| 0.02 | 1e-10 | 0 / 0 / 48 | 44 / 4 / 0 |
| 0.02 | 1e-8 | 0 / 0 / 48 | **38 / 2 / 8** |
| 0.02 | 1e-6 | 0 / 0 / 48 | 0 / 0 / 48 |
| 0.2 | 0 | 43 / 5 / 0 | 43 / 5 / 0 |
| 0.2 | 1e-12 | **22 / 0 / 26** | 43 / 5 / 0 |
| 0.2 | 1e-10 | 0 / 0 / 48 | 43 / 5 / 0 |
| 0.2 | 1e-8 | 0 / 0 / 48 | **43 / 4 / 1** |
| 0.2 | 1e-6 | 0 / 0 / 48 | **20 / 0 / 28** |

The direct boxes lose discriminating power rapidly. Local Lipschitz transfer certifies every previously known pointwise comparison at budget \(10^{-10}\) for both tested radii. At budget \(10^{-8}\), it retains 40/48 strict decisions at radius 0.02 and 47/48 at radius 0.2. At budget \(10^{-6}\) and radius 0.2, **twenty active-over-isotropic** comparisons remain certified.

Across all 480 matched contexts, an independent audit of the **two unmodified hosted result files** found **zero contradictory strict decisions** between methods. Both reproduce the complete zero-budget SMC-09 counts, including the inactive SAE and Jacobian-optimal controls; otherwise inconclusive cases are preserved as such. The local Lipschitz method also certifies the Jacobian-optimal effect greater than the active SAE on 48/48 at budget \(10^{-8}\) at both radii, and 40/48 at radius 0.2 with budget \(10^{-6}\).

## What is proved, and what remains

Each positive certificate is a theorem about the **exact-real frozen pretrained network** on an explicitly bounded input neighborhood. It has no dependency on empirical sampling of uncertain points, numerical maxima over sampled Hessians or the previous vacuous global OLM-08 curvature bound.

The independent uncertainty is confined to the **48 channels of one chosen goal token and the two selected unit intervention directions**. We do not certify uncertainty in the other tokens, pretrained weights, SAE retraining, new feature-selection identities or the machine-level FP64 implementation. The test roster was previously analyzed in SMC-07–09, making this a mathematical extension to the known finite cohort and **not fresh independent generalization evidence**. No semantic Sudoku reasoning feature or capability improvement is established by larger logit effects.

A useful next mathematical gate is to certify which exact channel-coordinate perturbations preserve an SAE active-atom identity and then compose that guard with the present local effect-ordering theorem. This would require a bound on the trained sparse encoder's feature-selection margin and a declared input-to-SAE code perturbation map, together with an independent puzzle-family evaluation.
