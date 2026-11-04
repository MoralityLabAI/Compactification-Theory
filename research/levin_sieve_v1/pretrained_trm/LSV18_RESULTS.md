# LSV-18: frozen learned semiconjugacy, 64-puzzle result

**November 4, 2026.** Source-locked CPU experiment, unit tests and both 16/64-puzzle CI jobs passed: https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41855844709 . Complete machine-readable result including per-puzzle scores is archived as the lsv18-locked-64-puzzle GitHub Actions artifact. No GPU/Goodfire credits were consumed.

Checkpoint SHA256: 3bab6189d4f25fe1e7a326a6b29ddc3141d21325f835ed1cd6409c0ac31447e1
Roster SHA256: eedcb21de54ab1e8fff70361fa2ddcedda342d79d788cf757108b6369c78a611

Exactly 64 puzzle IDs: 32 training, 16 validation, 16 test. Each puzzle generates nine correlated source states, so there are **16 independent held-out puzzle clusters and 144 test samples**. Every student has a 16-dimensional macrostate. The local observation has 48 coordinates from the selected goal token. The context observation adds global/row/column means to reach 192 coordinates.

## Exact research target and results

The source-locked frozen teacher evolves its 81x48 state with shared update R. Train encoder q, macro-transition F and decoder d to approximate q(R(h)) = F(q(h)) and future readouts d(F^k(q(h))). The student rollout applies F for three steps, without further teacher calls. Metrics include raw and scale-normalized closure defects, free-running teacher-logit prediction, label agreement and true Sudoku correctness.

| Student | Parameters | Normalized closure MSE | Held-out rollout logit MSE | Final teacher-label agreement | Final Sudoku correctness |
|---|---:|---:|---:|---:|---:|
| Local linear | 1,226 | 0.11454 | 2.09534 | 79.86% | 68.75% |
| Local nonlinear | 3,338 | 0.11741 | 2.17388 | 79.17% | 70.83% |
| Context linear | 3,530 | 0.21092 | 3.07541 | 92.36% | 81.25% |
| Context nonlinear | 7,946 | 0.32482 | 4.19068 | 79.86% | 68.75% |
| Stage-one persistence | No trained student | N/A | **1.89993** | 86.81% | 75.00% |
| Frozen teacher final | 44,001 | N/A | 0 | 100% (self) | **88.19%** |

Context linear has the strongest third-horizon teacher-label agreement among candidates, and improves 5.56 percentage points over persistence. Persistence retains the lowest logit MSE among the approximate predictors. All four candidate closure defects remain materially nonzero. The nonlinear models do not demonstrate better macrostate dynamics on this distribution.

## Paired puzzle-level uncertainty

A post hoc 20,000-resample paired bootstrap over the **16 puzzle IDs** with replacement (NumPy default_rng seed 1818) compared context-linear with persistence on the third future horizon:

| Outcome | Mean paired change | Descriptive 95% percentile interval | Wins / losses / ties by puzzle |
|---|---:|---:|---|
| Teacher-label agreement | +5.56 percentage points | -13.89 to +25.00 points | 3 / 2 / 11 |
| Sudoku correctness | +6.25 percentage points | -12.50 to +25.69 points | 3 / 1 / 12 |

The intervals include zero. Edited states of one puzzle are not independent observations. This additional evaluation reused the frozen trained models and altered no fitting choices.

## Local-only nonclosure, with a measurable lower bound

Construct an off-site perturbation that changes a different token while leaving the selected goal token exactly equal. The local goal observation and thus any deterministic q of that observation are identical for the clean and edited full hidden states.

On the 16 held-out puzzle IDs, all **64** remote perturbation pairs produced different future goal logits by greater than 1e-6. The largest future logit difference was 0.22057. Therefore an exact factorization of all the observed future logits through only the unchanged local token is impossible.

For each held-out puzzle group consisting of its clean state plus four remote edits, the best constant prediction within that common observation class has a mean-squared-logit-error floor at three future horizons of:

    0.000416671, 0.000351950, 0.000431974.

As fractions of overall held-out future-logit variance these are only 0.00583%, 0.00465%, and 0.00568%. Thus exact local factorization is ruled out by the finite witness, while practical approximate local prediction remains plausible.

The universal theorem that two states with the same observation and future-output gap D force a worst-case prediction error at least D/2 is an elementary norm inequality. Its Lean companion in OntologicalProofs is audited separately.

## Limits and next test

These are experimental measurements from an externally pretrained, 44,001-parameter TRM-inspired Sudoku network, not the official Tiny Recursive Model and not a Qwen checkpoint. Even the context view summarizes 192 numbers from an entire 3888-coordinate hidden state; obtaining those features still requires that underlying state. A 16-dimensional goal-conditioned statistic is not a complete replacement for the full neural state.

The next evaluation should freeze a broader independently generated Sudoku distribution, test stronger remote perturbations causing delayed output changes, and measure actual compute costs across all 81 goal cells. The current result does not establish a deployable compressed recurrent engine or the transfer of the Unique Games separation theorem to neural activations.

Reproduce with source lock and fixed train/eval scripts:

    python -m unittest discover -s research/levin_sieve_v1/pretrained_trm -p 'test_semiconjugacy_gym.py' -v
    python research/levin_sieve_v1/pretrained_trm/semiconjugacy_gym.py --fetch --max-cases 64 --max-epochs 200 --out runs/lsv18/full.json

Protocol: LSV18_SEMICONJUGACY.md. CI: https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/41855844709
