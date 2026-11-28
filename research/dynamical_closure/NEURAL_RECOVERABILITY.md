# Learned recurrent-state recoverability: CPU pilot

This is a small controlled follow-up to [TRANSFER_AND_GENERALIZATION.md](TRANSFER_AND_GENERALIZATION.md), **not** a theorem about neural networks or a pretrained-model experiment. The table of physical states and interventions is unchanged. The neural system is a 24-wide, two-head recurrent attention block: its action token attends causally to the previous recurrent state and itself, then updates that state with residual feed-forward layers. It predicts current task bit y after each action. Its recurrent state is the complete computational state for this model; no unobserved KV cache is carried across calls.

All eight states (y,b,c) occur in training, along with all five one-step actions. Of the 25 two-action words, two are held out entirely: `flip_b/reveal_b` and `flip_y/reveal_b`. Train on the other 23 two-action words, all intermediate task targets, and no held-out word. This is **composition generalization**, not new-state or new-action generalization. Two fixed seeds, 300 full-batch AdamW steps, BCE loss and no weight decay. CPU only; the image has CUDA-capable torch but **no accelerator was requested**. No pretrained model, model API, or external dataset.

We compare two objectives, both with the same architecture, seeds, cases and training steps:

- Task-only: minimize task BCE.
- Single-action state consistency: add weight 1 times MSE between the recurrent state after one action and the *detached* canonical encoding of that action's known successor microstate. This is supervised latent alignment to the known transition; it uses no held-out composition. It deliberately supplies a mechanistic inductive bias, rather than discovering state alignment from task labels alone.

Fit a linear least-squares probe with intercept to the recurrent hidden state at t=0 and after every trained single action. The probe labels are all three true state bits. Threshold its output at 1/2; the nested **discrete decoded codes** are (y,b,c), (y,b), y, and constant. The pilot risk is the population conditional-variance Bayes risk for the true task target after `reveal_b`, evaluated over all eight states, with tolerance 1/16. It does not measure performance of the network's own task readout or establish causal reliance on a probe direction. Code fitting uses all eight starting states, so apparent full-state decoding is not evidence of unseen-state generalization.

## Observations from initial runs

| Condition | Seed | Training final-task accuracy | Held-out `flip_b/reveal_b` | Held-out `flip_y/reveal_b` | Held-out decoded bits (two words × 24) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Task-only | 0 | 224/224 | 3/8 | 5/8 | 10/24, 14/24 |
| Task-only | 1 | 224/224 | 0/8 | 6/8 | 14/24, 15/24 |
| State-consistent | 0 | 224/224 | 8/8 | 8/8 | 24/24, 24/24 |
| State-consistent | 1 | 224/224 | 8/8 | 8/8 | 24/24, 24/24 |

Source outputs: task-only [exp:exp_01m6avgqvgf09ahq2arkmvxyyx] (`neural_results.json`); state-consistent [exp:exp_01m6avk3w5fct8we299hhambv5] (`neural_aligned.json`). Full-batch train accuracy is evaluated on the final label of each trained word; training also supervises intermediate outputs. Counts are exact finite-domain counts, not population estimates with confidence intervals.

Both seeds in both conditions decoded all eight starting states correctly. Therefore the *oracle* fixed-target risk along their decoded-code paths is (0, 0, 1/4, 1/4) for pilot `reveal_b` and the two held-out b-sensitive compositions: k*=2. The c-reveal risk is (0, 1/4, 1/4, 1/4): k*=1. **This risk calculation alone cannot explain the task-only model's held-out failures**: even its uncompressed state is decoded at t=0, but the trained recurrent update fails to compose, and held-out state-bit probe transfer also fails. For the state-consistent model the same code threshold becomes relevant to compression *after* the full model demonstrably generalizes on held-out compositions.

## Fitted abstract transitions, rather than oracle risks alone

Fit squared-loss transition tables from initial decoded codes to decoded
neural codes after each **single** action. Apply the tables sequentially to
the held-out words. The fitted table gets the neural successor code from the
probe, not from the microscopic transition map. This is an association test
on a small, fully covered state space; it is not a causal intervention on
hidden units. The uncompressed and task-only rows are separate from the
network's own task readout.

| Condition, seed | Word | Fitted MSE for decoded (y,b,c), (y,b), y, constant | Single-action bit decoding |
| --- | --- | --- | ---: |
| Task-only, 0 | flip_b/reveal_b | 1/4, 1/4, 1/4, 1/4 | 116/120 |
| Task-only, 1 | flip_b/reveal_b | 1/4, undefined, 1/4, 1/4 | 116/120 |
| State-consistent, 0 | flip_b/reveal_b | 0, 0, 1/4, 1/4 | 120/120 |
| State-consistent, 1 | flip_b/reveal_b | 0, 0, 1/4, 1/4 | 120/120 |
| State-consistent, 0 and 1 | flip_y/reveal_b | 0, 0, 1/4, 1/4 | 120/120 |

The undefined entry arises because an intermediate fractional code is not in
the fitted discrete transition table; it is **not** a high numerical loss.
In the state-consistent condition, the decoded (y,b) code is closed for the
b-sensitive compositions but **not** for the entire five-action alphabet:
reveal_c breaks its closure. Four nonclosed fibers reported for that code
count this out-of-family action. All eight starting states decode correctly in
both conditions, so the task-only uncompressed fitted-table error demonstrates
failure of its *successor* encoding, not lack of information at t=0. The
state-consistent model resolves that failure on these words, and only then
does discarding b cause irreducible and fitted MSE 1/4 at k=2.

The final fitted-code results, including the other task-only composition,
are in [exp:exp_01m6avnt3bewra6ddpp0wasy5j] (`aligned.json` and
`unconstrained.json`). That job reran both settings from the same code and
seeds, reproducing the earlier full-model counts. The probe-decoding threshold (1/2) was fixed before this final check.
The state-consistency condition was chosen **after** observing task-only
composition failure, so this comparison is exploratory, not a held-out
confirmatory estimate of the effect across architectures or tasks.

## Replay and limits

From the repository root, with CPU torch installed (e.g. `apy`):

```sh
PYTHONDONTWRITEBYTECODE=1 apy -m unittest discover -s research/dynamical_closure -p neural_test.py -v
PYTHONDONTWRITEBYTECODE=1 apy research/dynamical_closure/neural_recoverability.py --steps 300 --seeds 0 1 --consistency 0
PYTHONDONTWRITEBYTECODE=1 apy research/dynamical_closure/neural_recoverability.py --steps 300 --seeds 0 1 --consistency 1
```

The original jobs used torch 2.11.0+cu130 (CPU execution) in the pinned `job-core` image. Small differences across CPU libraries and BLAS implementations may affect model optimization and probe thresholds. The result has only two seeds and eight states, a probe fitted on all initial states and one-step trajectories, known transition labels in the aligned condition, and no intervention on neural hidden units. It establishes a reproducible failure mode and a targeted repair in this toy, not a generalization guarantee or a causal feature attribution for real models.
