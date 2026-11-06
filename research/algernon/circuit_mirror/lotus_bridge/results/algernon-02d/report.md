# Algernon-02D result: loop count as the only depth signal, fixed latent slots

Run 2026-11-06 04:55 to 22:03 on the local RTX 5080, under
[EXPERIMENT.md, H-A2d](../../../EXPERIMENT.md#h-a2d-loop-count-as-the-only-depth-signal-one-fixed-set-of-latent-slots-registered-2026-11-06-before-any-training),
registered before any training. The per-step progress log is [progress.md](progress.md).

**Arm D1 implements a reusable transition; arm D0 did not train.**

- **Reusable transition, D1: yes.**
  - D1 passes the 02B gate (`recurrence_specific_state`): the primary `diagonal` view decodes 1.000 at stages 9-16, against 0.583 for the strongest no-recurrence control, +0.417 [+0.297, +0.534].
  - Its behavioural `long_cycle` accuracy over d = 9..16 beats the best trivial baseline by +0.812 [+0.745, +0.875].
- **Reusable transition, D0: no.** D0 is `training_failed` (validation 0.639, threshold 0.90). It fails the gate (`no_iterative_state_signal`), and its `long_cycle` behaviour is at the trivial baseline: +0.010 [-0.052, +0.078].
- **Per-loop supervision matters: not decided as registered.** D1 minus D0 on held-out behaviour over d = 9..16 is +0.401 [+0.286, +0.521], which clears the registered margin of 0.10. But D0 is `training_failed`. So the comparison is between a model that learned the task and one that did not learn it even at depths 1-8 in 16 epochs. It shows that per-loop supervision made the task learnable within this budget. It does not show what per-loop supervision does to depth generalization in two equally trained models.

## Answer

With the step count out of the question and one fixed set of six latent slots refined once per loop, GPT-2 trained with per-loop supervision at depths 1-8 runs the pointer transition for as many loops as it is given.

- **It answers every roster case correctly at every depth from 1 to 16:** 1,152 of 1,152, with exactly d loops for depth d. Nothing beyond 8 was ever trained.
  - The held-out `long_cycle` maps are single 6-cycles, so the answer moves on every loop. In 44 of the 72 roster cases the answer changes with d.
  - Only 12.5% of `long_cycle` answers at 9-16 equal the depth-8 answer, so a constant guess cannot score here.
- **The state after each loop is in the slots.**
  - A linear decoder trained on `random_function` stages 1-8 reads the held-out state at every stage 1-16 with accuracy 1.000. Transition accuracy is 1.000, the commuting defect is 0.000, and all 48 held-out traces are exact.
  - The model's own LM head, applied to the slots after loop k, spells out the k-th step `<<s->t>>` for every held-out case at every k from 9 to 16 (1.000, [readout_shift.json](cap_D1/readout_shift.json)).
  - Every no-recurrence control stays at chance on `long_cycle` (0.125-0.167 against 1.000).

02C's best arm, with the count in the prompt and a new block per step, scored 0.630 on the same held-out cases over 9-16. Its `long_cycle` score there, 0.260, was barely above the trivial baseline.

## Training

| arm | epochs | final validation (512 cases, depths 1-8) | validity | weights | process starts | power-cap pauses | out-of-memory pauses |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| D1 (per-loop weight 0.05) | 16 | 1.000 | valid | `80eda0824cff` | 9 | 7 | 0 |
| D0 (weight 0) | 16 | 0.639 | `training_failed` | `ec9650544ab6` | 2 | 0 | 0 |

- **Shared setup:**
  - Both arms start from the 02C stage-1 CoT weights (`1bc71aa4ce96`).
  - The patched upstream source has hash `ffd4e4ef6138`; the patch is one line, `block_start = 0` under the `fixed_slot_supervision` flag.
  - The data is the 02C pairs with the no-count question (train `2dbe58737c7a`, validation `c97512fbb461`).
- **D1** reached 1.000 on the 64-case checks at epoch 4 and stayed there.
- **D0** stayed between 0.55 and 0.78 on the 64-case checks and ended at 0.639 on 512. By depth, its final accuracies range from 0.48 to 0.70, with no trend in depth.
- **D0's run was interrupted once:** it was stopped at its epoch-4 checkpoint for a RAM upgrade and resumed at 17:19, 4 h 45 min later. The first step after the resume reproduced the loss and gradient norm logged before the stop exactly (0.205262 and 1.852647).
- Per-epoch numbers are in [progress.md](progress.md) and `train_D1.json` / `train_D0.json`.

## Behavioural depth curve (held-out `long_cycle` + `sink`, 48 cases; no count in the question, d loops)

| model | 1-8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 9-16 [95% CI] | `long_cycle` 9-16 | vs best trivial on `long_cycle` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 [1.00, 1.00] | 1.000 | +0.812 [+0.745, +0.875] |
| D0 | 0.65 | 0.56 | 0.60 | 0.54 | 0.50 | 0.71 | 0.67 | 0.60 | 0.60 | 0.599 [0.48, 0.71] | 0.198 | +0.010 [-0.052, +0.078] |
| 02C arm A (context) | 1.00 | 1.00 | 0.50 | 0.50 | 1.00 | 0.50 | 0.52 | 0.50 | 0.52 | 0.630 [0.52, 0.74] | 0.260 | +0.073 [+0.005, +0.135] |

- The best trivial baseline is the 02B train-majority-by-depth.
- `sink` is 1.00 at every depth for both 02D arms. D0's held-out score is therefore almost all `sink`.
- D1 is also 1.00 at every depth on the `permutation` calibration family.
- Full per-prompt records, calibration included, are in [depth/](depth/).

## 02B instrument on the trained arms (unchanged; roster at 16 loops, 6 fixed slots)

Both captures passed every stop rule: all 9 parity checks, the drift recapture bitwise and the weights unchanged. Peak allocation was 0.57 GiB.

- With one fixed block, the capture stores the same six slots at every stage index.
  - `diagonal` and `lastblock` both read the slots after k loops, and are identical here.
  - `final` reads them after all 16 loops.
  - `pass0` reads them before any loop.
- Decoders were selected on the `permutation` calibration family, as in 02B:
  - D1: linear, last latent, 4,614 parameters, calibration 1.000 at 1-8.
  - D0: linear, block mean, calibration 0.526.

Held-out accuracy of the decoded state (48 cases):

| view or control | D1 1-8 | D1 9-16 | D1 `long_cycle` 9-16 | D0 1-8 | D0 9-16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| view:diagonal (primary for both, by calibration) | 1.000 | 1.000 | 1.000 | 0.659 | 0.591 |
| view:final | 0.565 | 0.622 | 0.245 | | |
| view:pass0 | 0.555 | 0.500 | 0.125 | | 0.581 |
| view:static | 0.576 | 0.521 | 0.125 | | |
| control:shuffled_stage | 0.565 | 0.583 | 0.167 | | |
| control:constant_state | 0.625 | 0.562 | 0.125 | | |
| control:param_matched_feedforward | 0.497 | 0.396 | 0.125 | | |
| control:recurrence_removed | 0.336 | 0.294 | 0.130 | | |
| baseline:max_indegree | 0.594 | 0.570 | 0.141 | | |

D0's full table is in [cap_D0/tables.md](cap_D0/tables.md). The gate tests:

| gate test | D1 | D0 | required |
| --- | --- | --- | --- |
| primary vs strongest no-recurrence control, 9-16 | +0.417 [+0.297, +0.534] vs shuffled_stage | +0.010 [-0.008, +0.031] vs pass0 | >= 0.10, lower bound > 0 |
| primary vs pass0 on `long_cycle` | +0.875 [+0.875, +0.875] | +0.021 [-0.016, +0.063] | lower bound > 0 |
| primary vs best trivial on `long_cycle` (classification) | +0.812 [+0.745, +0.875] | -0.005 [-0.083, +0.068] | lower bound > 0 |
| classification | `recurrence_specific_state` | `no_iterative_state_signal` | |

LOTUS answers to the 16-step roster questions (152 cases):

| condition | D1 | D1 `long_cycle` | D0 | D0 `long_cycle` |
| --- | ---: | ---: | ---: | ---: |
| 16 loops | 152 | 24/24 | 89 | 5/24 |
| explicit decoding, no latents | 59 | 1/24 | 90 | 6/24 |

D0's loops add nothing over decoding with no latents at all.

### The native readout in `analysis.py` scores the wrong integer for this task

- `analysis.py` reads the first integer of each block's own readout as the state after step k. The supervised text for step k is `<<s->t>>`, whose first integer is s, the state after k-1 steps.
- On D1 that scoring gives 0.500 at 9-16: right on `sink`, where s = t, and wrong on every `long_cycle` step. Scored against the whole step, the same readout is 1.000 ([cap_D1/readout_shift.json](cap_D1/readout_shift.json)).
- D0's slots read as `###` and newlines, 0.000 either way.
- The registered `analysis.py` is unchanged. The re-scoring is `readout_shift.py`, a diagnostic added after the run and not part of any decision. 02C's native readout rows, also `<<s->t>>` steps, were scored the same way; no decision used them either.

## Limits

- **One seed, GPT-2 small, one task family:** functional graphs on 6 nodes, trained on `random_function` only. The intervals are case-bootstrap over 48 held-out cases (24 `long_cycle`, 24 `sink`) and say nothing about seed variance. `sink` is solvable from the prompt, so the informative evidence is the 24 `long_cycle` cases at 8 depths each.
- **Depth tested only to 16,** twice the trained depth. Where, or whether, the transition breaks beyond that is not measured.
- **Readable slots are partly by construction at 1-8.** D1's training loss asks the slots, read through the LM head, to spell step k after loop k. So readability at depths 1-8 is what was trained. The evidence is the extrapolation to 9-16 and the control margins there.
- **D0 did not train.** The registered comparison cannot separate "per-loop supervision is needed to learn the task" from "per-loop supervision changes extrapolation". Answer-only training might reach validity with more epochs or a different learning rate; that was not tried.
- **Readout only, no patching.** The state is decodable at every loop, and the answers follow it. Whether the answer is caused by that state, for example whether swapping the slots between cases at loop k redirects the answer, is untested.
- **Implementation choices recorded in the training manifests:** the same as 02C.
  - 8-row micro-batches accumulated to 128 rows.
  - TF32 matmuls.
  - A torch memory cap at 0.60 of the card.
  - D1's power-cap pauses and D0's manual stop resumed from checkpoints. The CPU test shows a resumed loop run is bitwise equal to a straight one.

## Smallest next steps (not run)

On the existing D1 weights, no training needed:

1. **Depth beyond 16.** Run the depth curve at d = 17..64 to find where, if anywhere, the transition degrades.
2. **Causal slot patching.** At loop k, replace case A's six slots with case B's from the same loop, then continue. If the slots carry the state, the answer should follow B's trajectory under A's map from that point on. This tests pi(F(h)) = T(pi(h)) as an intervention rather than a readout.

With D1 passing the gate, the H-A2b condition for Phase 4 (constraint-guided correction) is met for this model for the first time.

## Files

| file | content |
| --- | --- |
| [progress.md](progress.md) | per-epoch training tables, depth curves and decisions as rendered during the run |
| `train_D1.json`, `train_D0.json` | training manifests: data and code hashes, patched-source hash, hyperparameters, per-epoch validation, pauses, final weight hashes |
| [compare.json](compare.json) | the registered D1-vs-D0 and trivial-baseline comparisons (keys `loop_step_tying`, `b_minus_a_heldout_9_16` are the shared 02C names: b = D1, a = D0) |
| `depth_*.json`, [depth/](depth/) | depth-curve summaries and full per-prompt records |
| `cap_D1/`, `cap_D0/` | the 02B instrument receipts per arm: compact manifest, metrics, per-case records, tables, parity, behaviour, and the readout re-scoring |
| `pointer_data.json` | data parameters and hashes (the data files regenerate from `pointer_cot.py` and `pointer_loop.py`) |

Checkpoints (about 1.5 GB each with optimizer state), captures and per-case parts stay outside git; their hashes are in the manifests.
