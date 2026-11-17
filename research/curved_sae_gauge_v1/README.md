# SAU-CURVE01: curved feature charts, gauge matching and causal transport

**Status:** six-test cross-repo CPU arithmetic smoke completed on 2026-11-17 (three tests here; three in LoopedTransformers' SAU-OBS01). The toy geometry is exact by construction. This is a *theoretical capacity witness*, not empirical recovery of a trained SAE or causal transfer between real teachers.

Motivation: [SauersML/Manifold-SAE](https://github.com/SauersML/Manifold-SAE) represents features by smooth curves/surfaces. Its [README](https://github.com/SauersML/Manifold-SAE/blob/main/README.md) describes GAM fitting, identification gauges and superposition separation. The public [report](https://github.com/SauersML/Manifold-SAE/blob/main/REPORT.md) distinguishes a K=1 real-model dose-calibration result from still-blocked K>=2 recovery gates. This code is an independent analytic construction; Manifold-SAE is AGPL-3.0 and **none of its source was copied**.

## Reproduction

```sh
python research/curved_sae_gauge_v1/curved_chart.py --self-test
python research/curved_sae_gauge_v1/curved_chart.py
python -m unittest discover -s research/curved_sae_gauge_v1 -p 'test_*.py' -v
```

The witness has one unit-circle latent, one held-out circle rotated relative to the calibration grid, and two physical model frames differing by a *fixed unknown rotation*. Source/target examples are paired at 64 calibration angles. A single 2D orthogonal gauge is learned there and frozen; 64 different angles are reserved for evaluation. A shuffled calibration correspondence is a negative control. A native linear readout is rotated with each physical frame; interventions follow local tangents and use the same physical radius.

Observed CPU numbers:

| Diagnostic | Matched | Control |
|---|---:|---:|
| Held-out physical coordinate MSE | 0.0 | 3.079709805747655 (shuffled assignment) |
| Held-out native tangent-effect MAE | 0.0 | 0.08937904177349443 (shuffled assignment) |
| Held-out residual energy | 0.0 (oracle curved chart) | 0.5000000000000001 (fixed rank-one linear projection) |

These numbers follow from the synthetic construction, including the covariantly rotated native readouts. An oracle coordinate is given to the curved decoder. The rank-one linear baseline uses one ambient direction, and therefore does **not** match the chart's nonlinear decoder complexity, fitting budget or learned encoder. It proves only a *restricted linear capacity obstruction*. The 64 phase points lie on one deterministic curve and are not independent model runs.

## Mathematical boundary: linear sparse coordinates vs chart coordinates

Let x(theta)=(cos(theta),sin(theta)) with theta uniform on the circle. Then

- E[x]=0 and Cov[x]=(1/2)I_2.
- For any rank-one orthogonal projector P, E||x-Px||_2^2 = 1/2.
- A 1D continuous curved chart g(theta)=(cos(theta),sin(theta)) has exactly zero residual with the true angle coordinate.
- For every nonzero linear read direction m, Pr[m dot x = 0]=0: the event consists of at most two points on the circle. Thus the **square linear expected-L0** cost is identically 2 for every invertible read basis. It cannot, by itself, identify a preferred coordinate frame for this data distribution.

The rank-one residual bound and flat full-rank expected-L0 objective are different statements. Neither implies an effective learned TopK SAE can represent the circle with one decoder atom, since that depends on architecture and nonlinear coordinate ownership. See [BPD boundary note](https://github.com/MoralityLabAI/BPD/blob/research/sauersml-linear-basis-boundary-20261117/mechinterp/bpd/theory/SAU_CURVED_BOUNDARY.md) for the relation to finite-hyperplane-mass filtrations.

## Prospective SMC-15 protocol

Use existing [SMC-14](../real_recurrence_sae_v1/SMC14_PROTOCOL.md) source-locked teachers and held-out Sudoku registration without unsealing or changing the SMC-14 primary scoring. Begin on *new* calibration/development teacher states and **fresh distinct puzzles**. Train a curved-chart dictionary against the existing SAE with controlled assignment sparsity, effective width, compute, exposure, parameter budget and independent restarts; evaluate both on untouched held-out families.

Record for every candidate: reconstruction EV on new teacher states, active atom count, curvature, local tangent singular values, calibrated/gauge-frozen cross-teacher correspondence, native output margin responses under matched physical interventions, and the number of puzzles giving an admissible intervention. Report errors within each puzzle cluster, not each token/atom. Reverse/phase gauge choices must be decided using calibration only.

**Primary falsifiers:** (i) the curved chart's improvement disappears under a size- and sparsity-matched nonlinear decoder; (ii) two near-coincident manifold tangent spaces become unidentifiable when both atoms coactivate; (iii) calibration-frame alignment fails on independent teachers; (iv) good reconstruction/coordinate agreement fails the signed native causal-response test; (v) output changes arise from the activation-capture runtime. Add [SAU-OBS01](https://github.com/MoralityLabAI/LoopedTransformers/tree/research/sauersml-observer-invariance-20261117/mechinterp/observer_footprint_cpu_v1) before measuring real activations.

This synthetic witness does not touch the SMC-14 source artifacts, results, or sealed tests. The native tangent metric is a constructed example, so it is not evidence for natural teacher-mechanism equivalence.
