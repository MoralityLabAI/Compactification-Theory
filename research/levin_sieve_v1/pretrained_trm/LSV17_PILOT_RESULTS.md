# LSV-17: first source-locked pretrained pilot, 16 Sudoku cases

**Date:** 2026-11-04. **Evidence:** completed GitHub Actions run [41852276229](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41852276229), with successful source-lock verification and six synthetic unit controls. The full machine-readable, source-locked 16-case receipt was uploaded as the lsv17-source-locked-pilot artifact (GitHub artifact ID 13334668754, retention 60 days). Real checkpoint SHA-256 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1; 64-puzzle source roster fingerprint eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611.

## Split and result size

16 distinct puzzle cases were included, divided by puzzle ID into 8 train, 4 validation and 4 test. Every clean or edited state of one puzzle stays in the same group. Each of five intervention directions was tested at L2 radii 0.2, 1.0 and 3.0 across three future recurrent blocks. Four held-out puzzles produce the displayed TV comparisons. The probe's 64 held-out records consist of correlated perturbations from those same four puzzles, rather than 64 independent Sudoku tasks.

## Future output total-variation at horizon 3

| L2 radius | Matched random | Joint three-horizon Jacobian-null | Relative decrease |
|---|---:|---:|---:|
| 0.2 | 0.00319303 | 0.000662989 | 79.2% |
| 1.0 | 0.0179629 | 0.00259019 | 85.6% |
| 3.0 | 0.0673574 | 0.00490569 | 92.7% |

Both intervention types were applied at the same post-first-recurrence goal-token site, with equal Euclidean norms. Among four held-out puzzles, no final-horizon clean label changed under these random or joint-null perturbations; intermediate-horizon flips were seen for random at radius 3.0 (one of four). With four test puzzles, these are descriptive measurements without a meaningful population inference.

For the separate pre-LayerNorm2 bus control, common-mode shifts produced a maximum future logit difference of 3.553e-15 in float64, while a same-radius centered transverse direction produced maximum 0.1668. The exact mathematical gauge symmetry was previously established in LSV-15; this observation checks its numerical implementation and selected trajectory.

## Linear vs nonlinear distillation head

| Head | Trained parameters | Held-out future-logit MSE | Teacher digit agreement at horizons 1/2/3 |
|---|---:|---:|---|
| Linear 48 to 30 | 1,470 | 7.56534 | 50.0%, 51.56%, 50.0% |
| Nonlinear 48 to 16 to 30 (tanh) | 1,294 | 4.61543 | 25.0%, 46.88%, 46.88% |

The nonlinear head lowered raw held-out logit MSE by 39.0%, while its teacher-label agreement was lower at all three horizons. The nonlinear validation normalized MSE was also slightly higher (3.4357 vs 3.2221), so this pilot is not evidence of a reliably superior deployable task quotient. Test data were used only for this final comparison. Hyperparameters and the split were specified before the pilot run.

## Next audit

Run the identical locked protocol on all 64 Sudoku puzzles, giving 16 held-out puzzle clusters; inspect paired differences at the puzzle level, probe calibration/argmax agreement, and the curvature of the null-direction Taylor residual as L2 radius grows. Positive logit-MSE changes with lower teacher action accuracy should remain reported as a mixed finding. The task family and model are small and highly structured, and no H100 or Goodfire credits were used.

Source: https://github.com/MoralityLabAI/Compactification-Theory/blob/main/research/levin_sieve_v1/pretrained_trm/LSV17_FUTURE_QUOTIENT.md
