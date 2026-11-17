# SAU-CURVE02: multi-atom tangent-frame identifiability

**Executed 2026-11-17**, Python 3.13/NumPy CPU; four deterministic tests passed. This is an independent mathematical/synthetic follow-on to the earlier [SAU-CURVE01](README.md) oracle-chart comparison, motivated by [SauersML/Manifold-SAE](https://github.com/SauersML/Manifold-SAE). No source from the AGPL-3.0 upstream repository has been copied.

## Exact construction

Two unit-circle feature curves lie in R^4. Feature 1 lives in plane (e1,e2). Feature 2 lives in a plane whose normal separation from feature 1 is φ: g2(θ) = (cos φ · cos θ, cos φ · sin θ, sin φ · cos θ, sin φ · sin θ). Its tangent is the corresponding derivative. The first feature's tangent is t1(θ)=(-sin θ,cos θ,0,0).

If both atoms are active, the physical Jacobian is J=[t1(θ1), t2(θ2)]. Since both columns have unit length and inner product q=cos(φ)cos(θ1−θ2), the singular values are sqrt(1+|q|) and sqrt(1−|q|). Hence

\`\`\`text
sigma_min(J) = sqrt(1 - |cos(phi) cos(theta1 - theta2)|).
inf_theta1,theta2 sigma_min(J) = sqrt(1 - cos(phi)), 0 <= phi <= 90 degrees.
\`\`\`

In a one-active-atom episode the single tangent has singular value 1. The study samples each active atom phase independently from Uniform[0,2π), then samples a coactivation Bernoulli variable; if both are active the two-column Jacobian is scored. This isolates *local inverse-conditioning geometry*. It is neither a fitted multi-atom SAE nor an experiment on a neural network.

## CPU sweep

Four independent seeds (103,131,167,211), 4,096 phase pairs per seed and condition. Angles φ = 0°, 5°, 15°, 30°, 60°, 90°. Coactivation coverage 0.25, 0.50, 1.00. The failure indicator is sigma_min < 0.10. Each source seed reuses its phase roster across angles and coverages. **There are 16,384 phase pairs per condition, not 18 independent experiments.**

| Plane tilt | Analytic worst-case sigma_min | Fraction below 0.10, 100% coactive | 5% quantile, 100% coactive |
|---|---:|---:|---:|
| 0° | 0.000000 | 0.088318 | 0.058914 |
| 5° | 0.061687 | 0.067871 | 0.085260 |
| 15° | 0.184592 | 0.000000 | 0.193487 |
| 30° | 0.366025 | 0.000000 | 0.370122 |
| 60° | 0.707107 | 0.000000 | 0.708337 |
| 90° | 1.000000 | 0.000000 | 1.000000 |

At full plane overlap (0°), the fraction below 0.10 varies with coactivation:

| Coactivation probability | Total coactive cases across 4 seeds | Fraction below 0.10 |
|---|---:|---:|
| 25% | 4,142 | 0.022278 |
| 50% | 8,176 | 0.043030 |
| 100% | 16,384 | 0.088318 |

The analytic worst-case threshold also gives a **condition-independent safe separation** for this particular model: at φ >= arccos(0.99) ≈ 8.11°, sigma_min >=0.1 for all coactive phases. The grid verifies this at 15° and above. At 0° the dangerous fraction is substantial when both atoms participate, even though either isolated atom has a perfectly conditioned coordinate.

## What the computation establishes

Both features can be smooth and locally valid while the *joint* inverse becomes ill-conditioned. An ambient additive edit near a collapsed two-atom tangent frame admits a large coordinate ambiguity. The local linear inverse error bound scales as 1/sigma_min(J). Selecting each feature separately during calibration can therefore conceal the near-singularity that appears when features coactivate. This is a direct numerical-geometry falsifier for naïve cross-atom identification claims.

The coactivation fraction and tilt are set by construction. We have not fitted a GAM or SAE, measured actual decoder incoherence, trained top-K baselines, or established real-model feature identity. The source program's multi-atom optimizer gates remain a separate empirical question.

## Reproduce

\`\`\`sh
python -m unittest discover -s research/curved_sae_gauge_v1 -p test_sau_curve02.py -v
python research/curved_sae_gauge_v1/sau_curve02.py > sau_curve02_full.json
\`\`\`

Four tests cover (1) analytic singular-value identity against direct 4×2 NumPy SVD at 120 independent parameter pairs, (2) zero/orthogonal plane endpoint cases, (3) worst-case analytic bound and >=15° safe-threshold gates, and (4) coactivation risk ordering.

## Next registered real-model comparison

Fit the physical SAE and a curved-chart candidate on the *same* new calibration set with matched active atoms, FLOPs, learned parameters and held-out examples. For every observed jointly active feature pair, report the actual native decoder's tangent-frame sigma_min and end-task effects of individually targeted interventions. Include an incoherence-regularized arm, overcomplete/topK matched controls, and actual multi-atom optimizer convergence status. Freeze correspondence and physical dose on calibration, evaluate native behavior with independent Sudoku families, and run the [SAU-OBS01 runtime parity gate](https://github.com/MoralityLabAI/LoopedTransformers/tree/main/mechinterp/observer_footprint_cpu_v1). Preserve every prior sealed test split and protocol.
