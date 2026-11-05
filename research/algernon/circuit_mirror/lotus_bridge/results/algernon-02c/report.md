# Algernon-02C result: LOTUS's own recipe on the pointer task

Run 2026-11-04 21:59 to 2026-11-05 08:51 on the local RTX 5080, under
[EXPERIMENT.md, H-A2c](../../../EXPERIMENT.md#h-a2c-lotuss-own-recipe-on-the-pointer-task-loop-to-step-tying-registered-2026-11-04-before-any-training),
registered before any training. The per-step progress log is [progress.md](progress.md).

**Both registered decisions are negative.**

- **Loop-to-step tying matters: not shown.** Arm B (step k supervised at loop k) is worse than arm A (the published after-loop supervision) at depths 9-16: -0.099 [-0.133, -0.070].
- **Reusable transition: no.** Neither arm passes the 02B gate; both are classified `no_iterative_state_signal`. Arm A clears the behavioural `long_cycle` condition by a hair; arm B clears neither condition.

## Answer

LOTUS's training method, with the task in distribution, does put the pointer trace into the latent blocks, but only within the trained depth range:

- At stages 1-8 a linear readout of the loop-aligned view decodes the held-out state at 0.836 (arm A) and 0.885 (arm B). The published 3B checkpoint in 02B managed 0.404.
- Transition accuracy is 0.82-0.89 and the commuting defect 0.32-0.36. The decoders transfer across generator families in range, where 02B's did not.

Past the trained depth nothing carries over:

- At stages 9-16 the same views decode 0.516 (A) and 0.464 (B). The no-recurrence controls reach 0.54-0.57: shuffled stages for A, the prompt-only start token for B.
- Behaviourally, asked for d > 8 steps with d loops, all three models collapse to the trivially solvable `sink` family.
  - The explicit-CoT model writes 5-7 correct steps and stops.
  - Arm A gets exactly one step further: depth 9 is 1.00 on every held-out family.
  - Arm B gets none.

The latent loop learns a transition that is used for the depths it was trained on. It is not reused beyond them, and tying each loop to one step makes the failure worse, not better.

## Training

| model | epochs | final validation (512 cases, depths 1-8) | weights | process starts | power-cap pauses | out-of-memory pauses |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| stage-1 CoT | 8 | 1.000 | `1bc71aa4ce96` | 1 | 0 | 0 |
| arm A (after-loop) | 16 | 1.000 | `0be1a42ce3a2` | 9 | 5 | 2 (+1 that ended the first chain) |
| arm B (loop-tied) | 16 | 0.998 | `6279c30224ee` | 2 | 1 | 0 |

- Both arms pass the registered validity rule (at least 0.90). Per-epoch numbers are in [progress.md](progress.md) and `train_*.json`.
- Both arms dipped as the latent stages took over the last explicit steps: arm A to 0.859 at stage 7, arm B to 0.844 at its first stage-8 epoch. Both recovered to 1.000 on the 64-case checks.

## Behavioural depth curve (held-out `long_cycle` + `sink`, 48 cases; d steps asked)

| model | 1-8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 9-16 [95% CI] | `long_cycle` 9-16 | vs best trivial on `long_cycle` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage-1 CoT | 1.00 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 | 0.50 | 0.50 | 0.50 | 0.562 [0.43, 0.69] | 0.125 | -0.063 [-0.130, +0.000] |
| arm A | 1.00 | 1.00 | 0.50 | 0.50 | 1.00 | 0.50 | 0.52 | 0.50 | 0.52 | 0.630 [0.52, 0.74] | 0.260 | +0.073 [+0.005, +0.135] |
| arm B | 0.99 | 0.50 | 0.50 | 0.52 | 0.50 | 0.58 | 0.54 | 0.52 | 0.58 | 0.531 [0.39, 0.66] | 0.062 | -0.125 [-0.203, -0.052] |

- Latent models run max(8, d) blocks and loops; the CoT model decodes freely.
- `sink` is 1.00 at every depth for every model, because a trace that reaches the sink stays there. The 0.50 floor is therefore all `sink`.
- Every `long_cycle` map is one 6-cycle, so the answer at d = 12 is the start state. The CoT model and arm A score at 12 because what they output there is the start state.
- Arm A's margin over the best trivial baseline comes from depths 9 and 12 only. The best trivial baseline is the 02B train-majority-by-depth.
- Full per-prompt records are in [depth/](depth/).

## 02B instrument on the trained arms (unchanged; roster at 16 loops, c_thought 6)

Both captures passed every stop rule: all 9 parity checks bitwise, repeat captures bitwise, the drift recapture bitwise, and weights unchanged. Peak allocation was 0.83 GiB, at 0.24 s per case.

Held-out accuracy of the decoded state (48 cases):

| view or control | arm A 1-8 | arm A 9-16 | arm A `long_cycle` 9-16 | arm B 1-8 | arm B 9-16 | arm B `long_cycle` 9-16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| view:diagonal (primary for both, by calibration) | 0.836 | 0.516 | 0.234 | 0.885 | 0.464 | 0.146 |
| view:final | 0.755 | 0.573 | 0.203 | 0.586 | 0.445 | 0.177 |
| view:lastblock | 0.711 | 0.633 | 0.266 | 0.852 | 0.383 | 0.167 |
| view:pass0 | 0.581 | 0.318 | 0.177 | 0.549 | 0.156 | 0.182 |
| view:static | 0.573 | 0.521 | 0.125 | 0.609 | 0.573 | 0.146 |
| control:shuffled_stage | 0.513 | 0.539 | 0.172 | 0.526 | 0.534 | 0.208 |
| control:constant_state | 0.625 | 0.562 | 0.125 | 0.625 | 0.562 | 0.125 |
| control:recurrence_removed | 0.156 | 0.177 | 0.177 | 0.161 | 0.180 | 0.193 |
| baseline:max_indegree | 0.594 | 0.570 | 0.141 | 0.594 | 0.570 | 0.141 |

The gate tests:

| gate test | arm A | arm B | required |
| --- | --- | --- | --- |
| primary vs strongest no-recurrence control, 9-16 | -0.023 [-0.076, +0.021] vs shuffled_stage | -0.109 [-0.195, -0.031] vs static | >= 0.10, lower bound > 0 |
| primary vs pass0 on `long_cycle` | +0.057 [+0.026, +0.089] | -0.036 [-0.078, +0.000] | lower bound > 0 |
| primary vs best trivial on `long_cycle` (classification) | +0.047 [-0.026, +0.120] | -0.042 [-0.120, +0.031] | lower bound > 0 |
| classification | `no_iterative_state_signal` | `no_iterative_state_signal` | |

- Transition accuracy for the primary view, P(correct at t+1 | correct at t), is 0.815 for A and 0.892 for B. The commuting defect is 0.361 for A and 0.324 for B.
- Full per-family tables, input-encoding probes and native readouts are in [cap_armA/tables.md](cap_armA/tables.md) and [cap_armB/tables.md](cap_armB/tables.md).

LOTUS answers to the 16-step roster questions (152 cases):

| condition | arm A | arm A `long_cycle` | arm B | arm B `long_cycle` |
| --- | ---: | ---: | ---: | ---: |
| 8 loops (trained) | 124 | 9/24 | 88 | 0/24 |
| 16 loops | 85 | 1/24 | 94 | 4/24 |
| explicit decoding, no latents | 33 | 0/24 | 71 | 0/24 |

Giving arm A more loops than it trained with hurts it. With its trained 8 blocks and loops it answers 9 of 24 sixteen-step `long_cycle` questions; with 16 it answers 1. Chance is about 4.

## Limits

- **One seed, GPT-2 small, one task family** (functional graphs on 6 nodes) trained on `random_function` only. The intervals are case-bootstrap over 48 held-out cases and say nothing about seed variance.
- **The prompt states the step count.** For an unseen count the models have to extrapolate a number they only saw up to 8. The explicit-CoT model shows this directly: it stops after 5-7 steps. So the depth failure mixes "reuse the transition" with "read and track a new step count". The registered design does not separate them.
- **Latent blocks 9-16 sit at positions never trained.** Extra loops beyond 8 also refine every block again, which arm A did not see in training.
- **Implementation choices recorded in the training manifests:**
  - Each 128-row step was accumulated over 8-row micro-batches, each micro-batch's loss weighted by its share of the rows. Upstream's single-GPU loss is a token mean over the whole batch.
  - TF32 matmuls.
  - A torch memory cap at 0.60 of the card.
  - Arm A's out-of-memory pauses resumed from checkpoints at most 120 s old, so a few steps were rerun with fresh dropout masks. The CPU test shows a pause-free resume is bitwise exact.
- **Readout only, no patching.** In range the states are decodable; whether the model uses them causally is untested.

## Smallest next intervention (not run)

The step-count confound is common to all three models, so the next experiment should remove it.

1. Drop the count from the prompt. Run d loops for d steps over one fixed set of latent slots, so no new positions appear. Upstream's `fixed_latent_tokens` is the closest switch; per-loop supervision of fixed slots needs a small change to the intermediate loss.
2. Train at d = 1..8 and test 9..16 with the same depth curve and gate.

If a reusable transition exists in this recipe, that is where it can show. The loop count would be the only depth signal and the slots the only state.

## Files

| file | content |
| --- | --- |
| [progress.md](progress.md) | per-epoch training tables, depth curves and decisions as rendered during the run |
| `train_cot.json`, `train_armA.json`, `train_armB.json` | training manifests: data and code hashes, hyperparameters, per-epoch validation, pauses and faults, final weight hashes |
| [compare.json](compare.json) | the registered A-vs-B and trivial-baseline comparisons |
| `depth_*.json`, [depth/](depth/) | depth-curve summaries and full per-prompt records |
| `cap_armA/`, `cap_armB/` | the 02B instrument receipts per arm: compact manifest, metrics, per-case records, tables, parity, behaviour |
| `pointer_data.json` | data parameters and hashes (the 8 MB data files regenerate from `pointer_cot.py`) |

Checkpoints (about 1.5 GB each with optimizer state), captures and per-case parts stay outside git; their hashes are in the manifests.
