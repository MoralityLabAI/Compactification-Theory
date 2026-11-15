# SMC-09: Arb-certified finite endpoint causal ordering

**November 15, 2026.** This is a source-locked follow-up to merged SMC-07 and SMC-08. The mathematical companion is [MathCan OLM-09](https://github.com/patrickdugan/MathCan/blob/main/research/orbit_lock/OLM09_ENDPOINT_CAUSAL_DOMINANCE.md).

## Source and evidence

- Full successful 16-puzzle, three-seed run: [GitHub Actions #43708737779](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43708737779). [Download every per-puzzle interval enclosure](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43708737779/artifacts/14160105425).
- Source checkpoint: independently pretrained 44,001-parameter TRM-inspired Sudoku model, commit f0406195fece4d0db29ced28f3a69d1993d95b1e.
- Exact checked checkpoint SHA-256: 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1.
- Source-locked 64-puzzle roster SHA-256: eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611.
- Puzzle-disjoint SAE fitting: 32 train/16 validation/16 test, seeds 71,89,113, validation-frozen checkpoints after 300 training steps.
- The source stage-one goal-token state is an exact frozen binary FP32 anchor. All selected 48-vector decoder/Jacobian directions are fixed dyadic FP64 components and renormalized to exact unit L2 length under Arb enclosure.
- Intervention radii are exact rationals 1/50, 1/5 and 4/5. Each exact-real three-step continuation is evaluated with 160-bit outward-rounded Arb operations, including erf-GELU, LayerNorm, and the complete ten-goal-logit readout.

## Complete verified results

There were 16 puzzle IDs × 3 SAE seeds × 3 radii = 144 intervention contexts. Each context has three active-versus-control comparisons: **432 strict, validated squared-effect interval separations, with zero inconclusive cases**.

| Radius | Active SAE exceeds isotropic | Active exceeds sampled inactive atom | Jacobian-optimal exceeds active | Inconclusive |
|---:|---:|---:|---:|---:|
| 0.02 | **44/48** | **32/48** | **48/48** | 0 |
| 0.2 | **43/48** | **32/48** | **48/48** | 0 |
| 0.8 | **42/48** | **30/48** | **48/48** | 0 |

The largest width among all 432 certified squared-effect gap intervals was approximately **3.13×10^-35**. Every strict comparison uses the sign of the enclosing Arb interval. The results concern the exact-real frozen model at those fixed endpoints; they are mathematical statements about the selected inputs and directions.

For scale, the descriptive mean *L2-effect* differences reconstructed from the exact-real interval midpoints agree closely with the previous FP64 SMC-08 measurements:

| Radius | Active minus isotropic | Active minus inactive | Active minus Jacobian-optimal |
|---:|---:|---:|---:|
| 0.02 | +0.0116329800 | +0.0076845591 | -0.0258810396 |
| 0.2 | +0.1114052405 | +0.0715181669 | -0.2603627220 |
| 0.8 | +0.3845984655 | +0.2190001208 | -1.0170691928 |

The 48 repeated seed/puzzle cases per radius arise from **16 independent puzzle clusters**, already examined in SMC-07/08. At all three radii, 15 of 16 puzzle groups show active-over-isotropic ordering for at least two of the three SAE seeds. At radius 0.02, 13 puzzle groups show that ordering for all three seeds; at 0.2, 12 groups; and at 0.8, 11.

## The mathematical advance in certification scope

SMC-08's exact-real global Hessian ceiling, about 10^53.61, was too conservative to certify causal ordering at empirically relevant radii. The local Arb Taylor tube certified the *sign of one winner-versus-runner margin change* on 15/16 puzzles at radius 10^-10 and only 3/16 at 10^-8.

SMC-09 instead evaluates the **complete ten-logit output effect at two specific endpoints**:

\[
E_u(r)=\sum_{j=1}^{10}\left[F_j(h+ru)-F_j(h)\right]^2.
\]

Validated interval inclusions give E_u(r) in I_u and E_v(r) in I_v. The strict separation inf(I_u) > sup(I_v) proves that intervention u has the larger final-logit L2 effect at that exact radius, without requiring an upper bound on Hessian curvature throughout the intervening segment. The standalone [OLM-09 proof](https://github.com/patrickdugan/MathCan/blob/main/research/orbit_lock/OLM09_ENDPOINT_CAUSAL_DOMINANCE.md) includes a separate sufficient condition for uncertain anchors/directions when a Lipschitz bound is independently available.

## Scientific limits

The certificate concerns exact-real execution with exactly represented stored source parameters, a frozen FP32 stage-one anchor and fixed learned directions. It does not certify the FP64 Torch kernel's hardware rounding or the effect ordering in a continuous neighborhood of radii and inputs.

The Jacobian-leading direction beats the active SAE atom in every finite experiment, and the active atom beats seeded isotropic directions in most contexts. This demonstrates reproducible directional sensitivity, not an identifiable semantic Sudoku feature, a causal necessity for correct reasoning or a capability improvement. The same test puzzles have been inspected in prior campaigns; genuinely new generalization evidence requires an independent source-defined puzzle-family split.

Reproduction: see [SMC-09 protocol](SMC09_PROTOCOL.md), [validator](smc09_endpoint_arb.py), [regression tests](test_smc09_endpoint_arb.py), and [full Actions evidence](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43708737779).
