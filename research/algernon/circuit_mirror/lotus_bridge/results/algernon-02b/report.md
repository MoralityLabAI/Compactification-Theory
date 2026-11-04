# Algernon-02B result: LOTUS latent recurrence on the 6x16 pointer roster

Run 2026-11-04 on the local RTX 5080, under the protocol registered in
[EXPERIMENT.md, H-A2b](../../../EXPERIMENT.md#h-a2b-block-aligned-capture-causal-controls-and-the-phase-4-gate-registered-2026-11-04-before-any-lotus-gpu-run)
before any GPU run. **Gate: failed. Classification: `no_iterative_state_signal`. Phase 4 (constraint-guided
correction) did not run.**

## Answer

On this task the published LOTUS 3B checkpoint does not implement a reusable computational transition, and
what can be decoded is prompt information, weakly:

- Recurrence lowers decodability. On the 48 held-out cases at stages 9-16, the calibration-selected primary
  view (`diagonal`: block k-1 after k passes) reads 0.185 [0.12, 0.26]. The same blocks before any refinement
  (`pass0`) read 0.276, and the start-latent token, which sees only the prompt (`static`), reads 0.341. The
  primary view is 0.156 below the strongest no-recurrence control, 95% interval [-0.279, -0.044].
- On `long_cycle`, the family that needs iteration, every view sits at chance (1/6 = 0.167): `diagonal`
  0.193, `pass0` 0.167, `static` 0.141. The gate's `long_cycle` test, `diagonal` minus `pass0`, is +0.026
  [-0.026, +0.073].
- The prompt representations carry the inputs. A linear readout of the start token on held-out cases recovers
  the start state at 0.96 (majority 0.19) and the first successor entry at 0.75 (0.19). Stage 1, one
  application of the map, decodes at 0.71 from `pass0` and `diagonal` alike, so it needs no loop.
- LOTUS's own answers are at chance: 24/152 final states correct with 16 loops, 15/152 with its trained 6
  loops, 10/152 as explicit CoT with no latents. Only 49% of its 16-loop answers are even a state id.

So the short answer is the second option, with a qualifier: the decodable content is task information already
in the prompt representation, and the latent loop adds none and erodes some. The checkpoint cannot do the task,
so there is no transition for a decoder to find.

## Run

| item | value |
| --- | --- |
| checkpoint | `yingfanbot/gsm-lotus-llama3b` @ `b392d2cb7aaa73475b93028221523c47f49f66a2`, bf16, 3,212,759,040 parameters |
| upstream | LOTUS `eb77e2f7909c5006f58ff0ad7cd6629b942caa9e`, `scripts/eval.py` unmodified |
| code | Compactification-Theory `0463686` (circuit_mirror clean at run start) |
| hardware | RTX 5080 Laptop GPU 16 GB, driver 577.05, Windows 11; torch 2.7.0+cu128, transformers 4.46.2, Python 3.12.10 |
| roster | `frozen_roster_6x16.json` (152 cases; train `random_function` 80, calibration `permutation` 24, held-out `long_cycle` 24 + `sink` 24), 16 loops x 25 latent tokens |
| GSM8K reproduction | 130/200 = 0.650, Wilson 95% [0.582, 0.713]; paper 0.7005 inside; floor 0.55 passed |
| parity (16 loops) | all 9 checks pass: hook installed, logits suppressed and hook removed all bitwise identical to upstream; readout ids equal upstream's intermediate argmax; lm_head on captured vectors argmax agreement 1.0 |
| repeat captures | smoke and calibration subset bitwise equal to the full roster; case 0 recaptured at the end bitwise equal; weight fingerprint unchanged (`ddae3a13...`); parity digests equal across the three process starts that recorded them (attempt 1's manifest was overwritten) |
| GPU memory | 5.98 GiB allocated after load; peak allocated 7.72 GiB (8,285,663,744 B), reserved 7.82 GiB; upstream eval peak 6.58 GB as eval.py reports it |
| GPU compute | full roster 236.9 s (1.56 s/case), counterfactual 66.0 s, 6-loop answers 72.8 s, explicit CoT 137.6 s, GSM8K 151.3 s; about 44.3 TFLOP and 6,891 tokens through the stack per captured case |
| CPU analysis | 120 s; linear decoders 18,438 parameters, MLP 197,062; symbolic stepwise model 222 |

## Depth by depth (all 48 held-out cases; accuracy of the decoded state)

| method | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 9-16 [95% CI] | defect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| view:diagonal (primary) | 0.71 | 0.29 | 0.38 | 0.40 | 0.44 | 0.42 | 0.38 | 0.23 | 0.25 | 0.21 | 0.17 | 0.21 | 0.15 | 0.12 | 0.21 | 0.17 | 0.185 [0.12, 0.26] | 0.754 |
| view:final | 0.38 | 0.38 | 0.33 | 0.23 | 0.19 | 0.38 | 0.21 | 0.23 | 0.27 | 0.21 | 0.25 | 0.29 | 0.25 | 0.23 | 0.17 | 0.19 | 0.232 [0.15, 0.31] | 0.771 |
| view:lastblock (A2a) | 0.31 | 0.33 | 0.33 | 0.35 | 0.44 | 0.56 | 0.46 | 0.23 | 0.31 | 0.27 | 0.21 | 0.25 | 0.15 | 0.21 | 0.23 | 0.17 | 0.224 [0.17, 0.28] | 0.759 |
| view:pass0 | 0.71 | 0.33 | 0.44 | 0.42 | 0.46 | 0.48 | 0.50 | 0.31 | 0.31 | 0.23 | 0.27 | 0.35 | 0.21 | 0.33 | 0.23 | 0.27 | 0.276 [0.20, 0.37] | 0.663 |
| view:static | 0.46 | 0.31 | 0.31 | 0.29 | 0.27 | 0.48 | 0.46 | 0.31 | 0.31 | 0.29 | 0.27 | 0.48 | 0.46 | 0.31 | 0.31 | 0.29 | 0.341 [0.23, 0.46] | 0.717 |
| control:constant_state | 0.71 | 0.48 | 0.42 | 0.46 | 0.46 | 0.48 | 0.71 | 0.48 | 0.42 | 0.46 | 0.46 | 0.48 | 0.71 | 0.48 | 0.42 | 0.46 | 0.484 [0.37, 0.61] | 0.565 |
| control:recurrence_removed | 0.62 | 0.38 | 0.31 | 0.31 | 0.27 | 0.29 | 0.21 | 0.19 | 0.15 | 0.19 | 0.23 | 0.21 | 0.23 | 0.17 | 0.15 | 0.17 | 0.185 [0.11, 0.27] | 0.809 |
| control:param_matched_feedforward | 0.33 | 0.15 | 0.19 | 0.12 | 0.17 | 0.38 | 0.29 | 0.12 | 0.19 | 0.15 | 0.17 | 0.40 | 0.31 | 0.17 | 0.19 | 0.15 | 0.214 [0.12, 0.31] | 0.840 |
| control:label_shuffled_probe | 0.12 | 0.21 | 0.21 | 0.21 | 0.19 | 0.19 | 0.15 | 0.27 | 0.17 | 0.12 | 0.19 | 0.06 | 0.21 | 0.15 | 0.19 | 0.19 | 0.159 [0.11, 0.21] | 0.850 |
| baseline:max_indegree | 0.62 | 0.62 | 0.50 | 0.56 | 0.65 | 0.54 | 0.62 | 0.62 | 0.50 | 0.56 | 0.65 | 0.54 | 0.62 | 0.62 | 0.50 | 0.56 | 0.570 [0.45, 0.70] | 0.492 |
| baseline:learned_stepwise_symbolic (toy, 222 params) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 [1.00, 1.00] | 0.000 |
| counterfactual:relabelled_truth | 0.71 | 0.35 | 0.25 | 0.31 | 0.31 | 0.40 | 0.19 | 0.27 | 0.23 | 0.21 | 0.23 | 0.25 | 0.23 | 0.10 | 0.15 | 0.15 | 0.193 [0.13, 0.27] | 0.776 |

Defect is the rate at which the decoded next state is not the true successor of the decoded current state.
Per-family tables, every control and the native readouts are in [tables.md](tables.md).

- **Transition.** For the primary view, P(correct at t+1 | correct at t) = 0.556, against 0.809 for the
  constant-state control. Exact-trace validity over all 16 steps is 0.021 (1 of 48 cases).
- **Family transfer fails inside the trained depth range.** Every selected decoder fits the training family
  perfectly at stages 1-8 (train accuracy 1.0, except `static` at 0.66). Its calibration-family accuracy at
  the same stages is 0.21-0.28. The failure is not only depth extrapolation.
- **`sink` is solved without iteration.** Every `sink` trace collapses onto the sink node, so repeating
  stage-1 features scores 0.833 and max in-degree scores 1.00. That is why the registered gate requires the
  `long_cycle` comparison.
- **The counterfactual roster cannot separate hypotheses.** The decoder's predictions match the relabelled
  truth at 0.193 and the parent labels at 0.180 for stages 9-16, both at chance. At stage 1 they follow the
  relabelling (0.71), consistent with stage 1 being read from the prompt.
- **LOTUS's own block readout is mostly not a state.** Decoded with its own lm_head, the first integer per
  block parses for 51% of blocks (`diagonal`) and is right 0.107 of the time at stages 9-16.
- **The registered A2a probe** (`probe.py` unchanged, last block per loop) scores 0.229 at step 16, Wilson
  [0.133, 0.365], with defect 0.730. Its own constant-stage control scores 0.250 at step 16 and 0.276 over
  9-16, against the probe's 0.208. A2a fails its controls too.

## Gate

Registered rule: pass only if the primary view beats the strongest no-recurrence control on all held-out
cases at stages 9-16 by at least 0.10 with a case-bootstrap interval above 0, and beats `pass0` on
`long_cycle` with an interval above 0.

| test | difference | 95% interval | required |
| --- | ---: | --- | --- |
| diagonal vs strongest no-recurrence control (`view:static`) | -0.156 | [-0.279, -0.044] | >= 0.10, lower bound > 0 |
| diagonal vs pass0, `long_cycle` | +0.026 | [-0.026, +0.073] | lower bound > 0 |
| diagonal vs best trivial baseline, `long_cycle` (classification only) | +0.005 | [-0.083, +0.089] | lower bound > 0 |

Neither the gate nor the `static_representation_decoding` condition holds, which gives
`no_iterative_state_signal`. Phase 4 is skipped as registered.

## Causal limits

- Everything here is readout. A failed linear or MLP decoder shows these decoders cannot recover the state.
  It does not show the state is absent in a code they cannot read. No activation patching was run, because
  with no decodable signal there is no variable to patch.
- The pointer task is far outside the checkpoint's GSM8K training, and the model's answers are at chance.
  The result is about this checkpoint on this task, not about LOTUS's training method.
- n = 48 held-out cases. Intervals on 9-16 accuracy are about +-0.07.
- The gate has been checked only on synthetic fixtures (the CPU tests pass a recurrent fixture and fail a
  static one). No real model has yet passed it, so the instrument still lacks a real-model positive control.

## Failures during the run

The card fell to a 15 W power cap after about 5-6 minutes of full load, three times. The first two attempts
had no checkpoints: vramorama's power guard killed them at 88/152 and 72/152 of the full capture, and Windows
logged a GPU reset (nvlddmkm 153) one second after the second kill. The runner was then made resumable:
per-case parts, a GSM8K receipt, and parity digests compared across restarts. It now reads the enforced power
limit between cases and exits 75 at a checkpoint (vramorama 0.14.0 `run --pause-exit`).

Attempt 3 paused once at `full:test/sink/19`. It resumed four minutes later with 147 of 152 full-roster cases
from parts, and finished. The GSM8K figure comes from attempt 2's eval run, reused after its sample and
predictions rescored. See [attempts.json](attempts.json).

## Smallest next intervention (not run)

There is nothing for constraint-guided correction to act on. The smallest step that could change the answer
makes the task learnable with LOTUS's own recipe, then reruns this instrument unchanged (same roster, decoders,
controls and gate):

1. **Put the task in distribution.** Run Stage-1 CoT fine-tuning on pointer traces written like GSM8K-Aug
   steps (one `<<f(s)=t>>` per step), from upstream's CoT init, then the LOTUS curriculum with
   `max_latent_stage` 16. Use a small `c_thought`: a pointer step is one or two tokens, and 25 was sized for
   arithmetic. Train on `random_function` depths 1-8 only.
   - On the 5080 this is a LoRA on the 1B CoT checkpoint (`yingfanbot/gsm-cot-llama1b`) in short resumable
     windows, using the Parts/PowerWatch pattern above.
   - A 3B full fine-tune does not fit in 16 GB.
2. **Tie loops to steps.** The audit found that the published configuration supervises block b with step b
   after the final loop (`intermediate_loss_after_loop: True`). A loop iteration is therefore a refinement of
   every block, not a step.
   - Arm B sets `intermediate_loss_after_loop: False`, so step k is supervised at loop k. The flag already
     exists upstream.
   - Arm A keeps the published setting at equal data and compute.
   - This is the variable that decides whether π(F(h)) = T(π(h)) can hold for one loop.
3. **Circuit Mirror constraint as a third arm only.** Add a commuting penalty on train-family depths 1-8
   through a decoder frozen before training, at equal compute. The exact verifier stays scoring-only on
   held-out cases.

Qwenputer's compiled RV32 transition meets the abstraction contract φ(F(x,u)) = δ(φ(x),u) by construction.
Running the same roster through it as a compiled program gives the positive control this instrument lacks:
a model it should pass. It is not an initialisation for LOTUS, because it exposes no LOTUS-compatible
hidden-state bus.

## Files

| file | content |
| --- | --- |
| [manifest.json](manifest.json) | compact run manifest: environment, source and checkpoint hashes, parity, per-stage memory and compute, hashes of the large outputs kept outside git |
| [metrics.json](metrics.json) | every view, control, baseline, selection trial, input-encoding probe, behaviour score and the gate |
| [records.jsonl](records.jsonl) | per-case, per-method decoded traces with verifier results |
| [tables.md](tables.md) | the per-family depth tables |
| [parity.json](parity.json) | full-depth hook parity and reference digests |
| [probe_registered.json](probe_registered.json) | the unchanged A2a probe on the legacy pooled features |
| [behaviour.jsonl](behaviour.jsonl) | LOTUS answers at 6 loops and as explicit CoT |
| [gsm8k.json](gsm8k.json), [gsm8k_preds.json](gsm8k_preds.json) | upstream eval receipt and predictions (local sample path scrubbed from the key) |
| [attempts.json](attempts.json) | the four process starts and what each reused |

The hidden-state captures stay outside git. `manifest.json` records the SHA-256 of `full_blocks.npz`,
`cf_blocks.npz`, `full.npz`, the smoke captures and the probe weights. The 1.5 GB of per-case parts are the
resume cache.
