# Algernon-02E result: D1's transition runs to 64 loops, and its slots carry the state

Run 2026-11-06 22:23 to 22:31 on the local RTX 5080, under
[EXPERIMENT.md, H-A2e](../../../EXPERIMENT.md#h-a2e-how-far-d1s-transition-runs-and-whether-its-slots-carry-the-state-registered-2026-11-06-before-any-run),
registered and pushed (`2c8589a`) before any run. The model is 02D's arm D1 (`80eda0824cff`), with no further training.

**The registered decision is positive: the slots carry the pointer state.**

- **Depth.** D1 answers all 432 prompts correctly, the 72 calibration and held-out roster cases at d = 16, 20, 24, 32, 48 and 64 loops. It holds to depth 64, eight times its trained depth, the largest depth tested.
- **Causal patching at d = 16.** Replace a recipient's six slot states after loop k with a donor's after loop k, and the answer becomes the recipient's map applied 16 - k times from the donor's state.
  - This "follow" prediction holds in 141 of 144 (recipient, k) pairs: 0.979, Wilson 95% [0.941, 0.993]. The registered bar is 0.80 with a lower bound above 0.5.
  - The answer stays the recipient's own in 3 (0.021; the bar is at most 0.10) and is never the donor's own answer.
  - Every self-patch reproduces the unpatched answer (144 of 144), so the instrument changes nothing by itself.

## Answer

For this model, the latent recurrence is a reusable transition, and its state is causal.

- **The transition.** After each loop, the six fixed slots hold the current pointer state. The next loop reads it and applies the successor map stated in the question.
- **It runs far past training.** It keeps running correctly 56 loops past its trained depth of 8. The slots' largest activation after the last loop is 197.55 at every tested depth, so nothing grows or decays.
- **The state is causal.** Move the state from another case into the slots, and the model continues from it under its own map. It does not continue from the state it would have reached itself, and it does not copy the donor's answer.

## Depth beyond 16

| family (cases) | 16 | 20 | 24 | 32 | 48 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `long_cycle` held-out (24) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `sink` held-out (24) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `permutation` calibration (24) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| largest slot activation after the last loop | 197.55 | 197.55 | 197.55 | 197.55 | 197.55 | 197.55 |

The truth is each map iterated d times from its start. Per-prompt answers and activations are in [depth.json](depth.json).

## Causal slot patching (d = 16)

- **Recipients:** the 48 cases with bijective maps (24 `permutation`, 24 `long_cycle`).
- **Donors:** for each recipient and each k in {4, 8, 12}, the donor is the first case in a seeded order for which three states are pairwise different: the "follow" prediction, the recipient's own answer, and the donor's own answer. Every pair found one.
- **Intervention:** the recipient's slot states after loop k are overwritten with the donor's from an unpatched 16-loop run. All other positions, the prompt and the map are the recipient's.

| outcome | k = 4 | k = 8 | k = 12 | all 144 [Wilson 95%] |
| --- | ---: | ---: | ---: | ---: |
| answer = "follow" | 47/48 | 47/48 | 47/48 | 0.979 [0.941, 0.993] |
| answer = recipient's own | 1/48 | 1/48 | 1/48 | 0.021 [0.007, 0.059] |
| answer = donor's own | 0/48 | 0/48 | 0/48 | 0.000 [0.000, 0.026] |
| self-patch = unpatched answer | 48/48 | 48/48 | 48/48 | 1.000 [0.974, 1.000] |

**The three misses share one feature** (post-hoc breakdown, not a registered test).

- Each recipient's start state maps to itself (state 1 -> 1), so its true trace never moves. In each miss the model kept answering 1.
  - `calibration/permutation/10` at k = 4;
  - `calibration/permutation/18` at k = 8 and k = 12.
- Four recipients have such a fixed-point start, and their 12 patched pairs follow the patch in 9.
- The other 132 pairs follow it in 132.

A plausible reading is that when the question's start is a fixed point, part of the answer comes from the prompt as well as the slots. This was not tested further.

Per-pair records, with the donor, the three predictions and the answer text, are in [patch.json](patch.json).

## Limits

- **One model and one seed:** GPT-2 small, 6-node maps, the 02D D1 weights. Nothing here says how often this training recipe produces such a model.
- **Depth and node count.** The depth test stops at 64 loops. Larger graphs, which change the question and the state space, were not tested.
- **Patching was done only at d = 16 and k in {4, 8, 12}**, on bijective recipient maps. On `sink` maps, different states merge, so "follow" and "own" often coincide.
- **The patch moves all six slots together.** Which slots, or which directions within them, carry the state is not resolved.
- **The decision is about the answer.** The slot readout after the patch was not scored.

## What this means for Phase 4 (constraint-guided correction)

The brief's precondition for Phase 4, a meaningful latent-state signal, is met. But D1 makes no errors up to depth 64, so a correction operator has nothing to correct in this regime.

A Phase 4 comparison at matched compute needs a regime where the plain recurrence fails. Two candidates, neither registered or run:

- **Slot noise.** Perturb the slot state at loop k at a scale where the plain recurrence fails about half the time, then compare recovery with and without a small corrector driven by the exact transition check.
- **The 02C arms beyond depth 8.** These fail there and use LOTUS's published block layout.

## Files

| file | content |
| --- | --- |
| [manifest.json](manifest.json) | code commit, weights and roster hashes, seed, environment, hardware, commands, receipt hashes |
| [depth.json](depth.json) | depth summary and all 432 per-prompt records |
| [patch.json](patch.json) | patching summary (the registered decision is `slots_carry_state`) and all 288 per-run records: 144 donor patches and 144 self-patches |
