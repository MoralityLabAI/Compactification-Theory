# Algernon-02 hypothesis and evidence gates

Registered as a follow-up architecture proposal on 2026-11-03, after the already-run CPU smoke. The exact CPU and toy neural pilot are **exploratory development**, with their results stored as receipts. Any later experiment requiring independent confirmation must freeze a *new* test roster, code and model revision before computing its scores.

## Exact statement

For a finite state space \(S\) and deterministic transition \(T:S\to S\), a neural latent operator \(F\) and decoder \(\pi:H\to S\) define the defect \(e(h)=\mathbf 1[\pi(F(h))\ne T(\pi(h))]\). If \(e\) vanishes on all states reachable from \(h\), induction yields agreement at every loop depth. Holding only on finite training traces has no such guarantee.

## H-A2 (proposed)

A trained, pretrained-LOTUS-based recurrent reasoning state achieves lower **held-out, depth-conditioned commuting defect** at steps 9–16 than a no-constraint-feedback LOTUS baseline at matched total inference compute and access to task information. Required baselines: same checkpoint and token budget, input-equivalent feedforward readout, shuffled feedback, exact solver upper bound, and randomly initialized adapter.

Primary metric: average per-step exact transition correctness across *independently generated* held-out graph families; secondary: whole-trace valid fraction, final-state accuracy, base-task regression and GPU-seconds. Use pairwise comparisons clustered by graph generator seed. A separate unconstrained exact solver is an upper bound, since the test specification contains the full deterministic transition.

## Compiler mirror requirements

1. The CIRCOM2 source and Python constraint builder must accept identical public graph/start inputs, and both must reject a witness when *any intermediate state* violates the transition, even if the final output happens to match.
2. Public inputs must be bound to user-supplied state and successor mapping. Booleanity and one-hot constraints must rule out out-of-range field representations.
3. The Circom compiler output, actual witness generator, and a tool such as `snarkjs wtns check` must be exercised before claiming a compiled R1CS or zk proof. Python JSON from the local builder is a reference constraint list with its own wire order.
4. Any neural inference-time use of constraint residuals must account for information revealed. On deterministic pointer chasing, the full residual can reveal the successor state, making an exact solver a mandatory comparator.

## Hardware and training boundaries

The CPU toy does not require models or GPU. LOTUS's own Llama-3.2-3B model is the first larger neural reference. Qwenputer is a separate 4B compiled-execution baseline and does not yet expose a LOTUS-compatible hidden-state bus. The proposed common IR has explicit inputs, state variables, transition relation, output, and a witness encoding; bridge/decoder training and Qwen deployment are separate future experiments.

## H-A2a: observational LOTUS hidden-state probe (2026-11-04, pre-GPU run)

The first model-based stage uses the **published, frozen** LOTUS 3B checkpoint as a feature generator. With 400 latent slots and sixteen refinement iterations, the extraction captures a representation before recurrence and after each recurrence. A linear readout trained solely on train-family states from depths 1–8 predicts a discrete pointer-machine state. The pooling method is selected on calibration-family depth 1–8. The held-out generator families and depths 9–16 are scored once after selection. Whole-trace R1CS-validity and the decoded commuting-square defect are evaluated without feeding any violation signal into the model.

The frozen roster, exact input prompt, train/test boundaries and controls are in [lotus_bridge/README.md](lotus_bridge/README.md), with source pin and code. The probe may show a correlation from reading graph/input features rather than an algorithmic recurrent computation; same-stage controls are included, and any strong causal claim requires an input-only model and equal-compute explicit-CoT baseline. This protocol extends the earlier toy CPU evidence and does not modify its published receipts. **Actual model inference unrun.**

## H-A2b: block-aligned capture, causal controls and the Phase 4 gate (registered 2026-11-04, before any LOTUS GPU run)

**Audit finding that changes the instrument.** Upstream `scripts/lotus.py` at `eb77e2f` refines *every* latent block on every loop, and the published 3B configuration (`args/gsm8k_lotus_llama3b.yaml`: `intermediate_loss_after_loop: True`, `flat_intermediate_supervision: False`, `max_latent_stage: 6`, `c_thought: 25`) supervises block *b* with CoT step *b* using the logits of the **final** loop. A reasoning step therefore lives in a block, not in a loop iteration. The A2a capture reads only the last block after each loop. A2b keeps that view (it is the registered A2a protocol, still run unchanged through `probe.py`) and adds the full `[pass 0..16, block 0..15]` grid, the start-latent token (which attends only to the prompt and is never re-injected) and upstream's own intermediate-logit argmax over the loop region.

**Views of pointer stage k (k = 1..16).** `diagonal`: block k-1 after k passes. `final`: block k-1 after 16 passes. `lastblock`: block 15 after k passes (A2a). `pass0`: block k-1 before any refinement. `static`: the start-latent token. Stage 0 is the public start, which the R1CS witness binds.

**Decoder protocol.** For every view the same search: pool (last latent of the block, block mean) x {multinomial logistic regression with L2 in {1e-4, 1e-3, 1e-2, 1e-1}, one-hidden-layer MLP width 64}, fitted on train-family (`random_function`) stages 1-8 only, selected on calibration-family (`permutation`) stages 1-8 only. Held-out families (`long_cycle`, `sink`) and stages 9-16 are scored once. The primary view is whichever of `diagonal` and `final` has the higher calibration accuracy.

**Controls, scored with identical access.** Constant-state (stage-1 features repeated), shuffled-stage (stages permuted per case), recurrence-removed (primary decoder on the same blocks at pass 0), random-direction (pass-0 block plus a random unit direction scaled to the norm of the recurrent update), random-probe (trained norms, random directions), label-shuffled probe, parameter-matched feedforward (MLP on start token + one-hot depth, parameter count matched to the linear decoder), wrong-map scoring (same predictions scored against another same-family map from the same start), and a frozen relabelled roster (`frozen_counterfactual_6x16.json`, sha `ddb5216f...`): each held-out map conjugated by a fixed derangement, so a readout that follows the prompt must follow the relabelling. Symbolic baselines: copy-start, train majority by depth, max in-degree, a learned 222-parameter bilinear one-step model on the explicit map iterated 16 times, and the exact solver. LOTUS's own answers are scored at 16 loops (from the captured run), at the trained 6 loops, and with no latent tokens; its own block readout text is parsed for a state number.

**Gate for Phase 4 (constraint-guided correction).** Pass only if, on all held-out cases at stages 9-16, the primary view beats the strongest no-recurrence control (`pass0`, `static`, parameter-matched feedforward, recurrence-removed, shuffled-stage, random-direction, label-shuffled) by at least 0.10 with a case-bootstrap 95% interval above 0, **and** beats `pass0` on `long_cycle` alone with an interval above 0 (`sink` is solvable from the prompt). Classification: gate passed -> `recurrence_specific_state`; otherwise, if the primary view beats the best trivial baseline on `long_cycle` or the strongest control on all held-out cases with an interval above 0 -> `static_representation_decoding`; otherwise `no_iterative_state_signal`. If the gate fails, Phase 4 does not run.

**Reproduction and stop rules.** Before any roster capture, upstream `scripts/eval.py`, unmodified, runs on a fixed 200-item sample of the GSM8K test split (`random.Random(20261104)`), 6 loops, `c_thought` 25, bf16; the job stops if accuracy is below 0.55 (paper: 0.7005). The job then requires bitwise-identical upstream outputs with the hook installed, with logits suppressed and after removal, readout ids equal to upstream's intermediate-logit argmax, two-case smoke, a 16-case calibration subset (first 4 per family), then the full roster. It stops on nonfinite states, allocated memory growing more than 512 MiB over the first case, peak allocation over 11 GiB (torch capped at 70% of the card), any difference between repeated captures of the same case, or any change in the weight fingerprint. Code: `lotus_bridge/gpu_run.py`, `lotus_bridge/analysis.py`.

**Result (2026-11-04, code `0463686`, checkpoint `b392d2cb`): gate failed, `no_iterative_state_signal`; Phase 4 not run.**

- Every stop rule passed: GSM8K 130/200 = 0.650 (paper inside the Wilson interval), all parity checks, bitwise repeat captures, drift check and unchanged weights.
- Primary view `diagonal` (chosen on calibration) on held-out stages 9-16: 0.185 [0.12, 0.26]. The strongest no-recurrence control, `static`, scores 0.341, so the primary view trails it by -0.156 [-0.279, -0.044].
- `long_cycle`, `diagonal` vs `pass0`: +0.026 [-0.026, +0.073], with every view near chance (1/6).
- LOTUS's own final-state answers are 24/152 at 16 loops, 15/152 at 6 loops and 10/152 as explicit CoT.
- The registered A2a probe scores 0.229 at step 16, below its own constant-stage control (0.250).

Full receipts and the smallest next intervention are in [lotus_bridge/results/algernon-02b/report.md](lotus_bridge/results/algernon-02b/report.md). The one protocol change made after registration was operational: the job became resumable across power-cap restarts. Resumed captures were checked bitwise against same-process repeats and across restarts.

## H-A2c: LOTUS's own recipe on the pointer task, loop-to-step tying (registered 2026-11-04, before any training)

**Question.** With the pointer task in distribution and LOTUS's own training method, does the latent loop learn a transition that a frozen decoder reads as `pi(F(h)) = T(pi(h))` and that keeps working past the trained depth? Does supervising step k at loop k (rather than after the final loop, as published) decide it?

**Why GPT-2.** It is upstream's smallest LOTUS recipe (`args/gsm8k_cot_gpt2.yaml`, `args/gsm8k_lotus_gpt2.yaml`). The 1B LoRA route named in the 02B report needs peft, and its looped backward pass does not fit the 5-minute windows the power-capped card allows. Base `openai-community/gpt2` @ `607a30d783dfa663caf39e06633721c8d4cfcd7e`, fp32 as upstream, LOTUS source `eb77e2f`. Training reuses upstream's `Lotus` class, `get_dataset`, `get_cot_latent_dataset`, `MyCollator` and single-GPU loss unchanged; only the loop around them (single GPU, step checkpoints, power-cap pause) is new (`lotus_bridge/train_lotus.py`).

**Data** (`lotus_bridge/pointer_cot.py`). 20,000 train and 512 validation cases from the `random_function` generator (n = 6, generator seed 20261105), depth k uniform in 1..8. Every (map, start) pair in `frozen_roster_6x16.json` and `frozen_counterfactual_6x16.json` is excluded, and train and validation are disjoint. Question: `roster.prompt_for` without its final newline, so upstream's tokenization equals the 02B capture input exactly. Steps: `<<s->t>>`, 6 GPT-2 tokens with the newline. Answer: the final state.

**Stage 1 (explicit CoT).** Upstream CoT objective; lr 1e-4, weight decay 0.01, batch 128, seed 0, 8 epochs. If validation answer accuracy at depths 1-8 is below 0.95, continue to at most 20 epochs. Report the epoch used.

**Stage 2 (LOTUS curriculum)** from the stage-1 weights. c_thought 6, max_latent_stage 8, epochs_per_stage 1, 16 epochs (stages 0..8, then stage 8). intermediate_loss_weight 0.05, lr 1e-4, weight decay 0.01, batch 128, gradient clip 1.0, no optimizer reset, seed 0. Two arms, identical except one flag:
- **A (published):** `intermediate_loss_after_loop: True`, so block b is supervised with step b from the final loop's logits.
- **B (loop-tied):** `intermediate_loss_after_loop: False`, so block k-1 is supervised with step k at loop k. This is the 02B `diagonal` view.
The commuting-penalty arm is not run in this pass.

**Training validity.** An arm counts only if its validation answer accuracy at depths 1-8, with its trained 8 latent blocks and 8 loops, is at least 0.90. Otherwise it is reported as `training_failed`.

**Evaluation**, the same for the stage-1 model and each arm:
1. **Behavioural depth curve** on the frozen roster's calibration and held-out cases. For d = 1..16, ask for d steps and score the state at trace[d]. Each latent model runs L = max(8, d) blocks and loops. The CoT model generates freely, up to 160 tokens.
2. **The 02B instrument, unchanged:** capture on the full frozen roster and counterfactual at 16 loops, with c_thought 6 and the same parity, repeat and drift checks; then `analysis.py` with the same views, decoder search, controls and Phase 4 gate. GSM8K reproduction is replaced by the validation check above.

**Decision rules.**
- *Loop-step tying matters* if arm B's held-out behavioural accuracy over d = 9..16 exceeds arm A's by at least 0.10, with a case-bootstrap 95% interval above 0.
- *Reusable transition* for an arm if it passes the 02B gate (`recurrence_specific_state`) and its held-out behavioural accuracy over d = 9..16 beats the best trivial baseline from the 02B table on `long_cycle`, with an interval above 0.
- Everything else is reported as measured. Each decision is made once, on these frozen sets. No arm is rerun with changed settings and reported as the same arm.

**Result (2026-11-05, code `443d93f`): both decisions negative.**

- Stage-1 CoT and both arms passed training validity: final validation 1.000, 1.000 and 0.998 on 512 cases.
- *Loop-step tying matters*: not shown. On held-out behaviour over d = 9..16, B minus A is -0.099 [-0.133, -0.070]. Arm A scores 0.630, arm B 0.531 and the CoT model 0.562; every model gets 1.00 at d = 1..8.
- *Reusable transition*: no for both arms. Both fail the 02B gate, `no_iterative_state_signal`.
  - The primary `diagonal` view decodes 0.836 (A) and 0.885 (B) at stages 1-8, but at 9-16 it scores 0.516 vs the shuffled-stage control's 0.539 (A) and 0.464 vs the start token's 0.573 (B).
  - On the behavioural `long_cycle` condition against the best trivial baseline, arm A is +0.073 [+0.005, +0.135] and arm B -0.125.
- Past d = 8 every model collapses to the `sink` family, except arm A at d = 9. The CoT model writes 5-7 correct steps and stops.

The one deviation from this registration is gradient accumulation: each 128-row step was summed over 8-row micro-batches, because a full stage-8 batch does not fit the card. TF32 matmuls and out-of-memory pauses that resumed from checkpoints are also recorded in the training manifests. Report: [lotus_bridge/results/algernon-02c/report.md](lotus_bridge/results/algernon-02c/report.md).

## H-A2d: loop count as the only depth signal, one fixed set of latent slots (registered 2026-11-06, before any training)

**Why.** In 02C the prompt stated the step count, and the latent models wrote each step into a new block at a new position. Past the trained count of 8, every model failed, the explicit-CoT model included; it wrote 5-7 correct steps and stopped. That failure mixes two things: reusing a transition, and reading a step count never seen in training. 02D removes the count and the new positions. The only depth signal is the number of loops, and the only state is one fixed set of latent slots, refined once per loop.

**Model and change.**
- GPT-2 small as in 02C, initialised from the 02C stage-1 CoT weights (`1bc71aa4ce96`), fp32, upstream LOTUS `eb77e2f`.
- Every input carries c_thought = 6 latent slots, whatever the depth (upstream's `fixed_latent_tokens`). A depth-k case runs k loops.
- One change to upstream's loss, applied at import time to a copy of `scripts/lotus.py` (the file on disk is unchanged; the patched source hash is recorded). In per-loop supervision, `block_start = loop_idx * c_thought` becomes `block_start = 0` when the model's `fixed_slot_supervision` flag is set, so loop k's logits over the same six slots are supervised with step k. Inference is untouched.

**Data.**
- The 02C pairs: the same 20,000 train and 512 validation maps, starts and traces, depth 1..8.
- Question: `pointer_loop.prompt_nocount`, which is `roster.prompt_for` with the step count removed: "Starting at state s, follow the successor once per loop. Which state is reached?"
- Steps `<<s->t>>`; answer: the final state. Batches hold a single depth (bucketed, shuffled, seed 0), because one batch runs one loop count.

**Training.** No curriculum: without a count in the prompt, explicit trailing steps would have no way to know when to stop, so every step is latent from the start. lr 1e-4, weight decay 0.01, 128 rows per step as 8-row micro-batches, gradient clip 1.0, 16 epochs, seed 0. Two arms, identical except the intermediate loss weight:
- **D1:** 0.05 (upstream's value), so loop k is supervised with step k.
- **D0:** 0.0, so only the answer after k loops is supervised.

**Training validity.** An arm counts only if its validation answer accuracy at depths 1-8, with loops = depth, is at least 0.90. Otherwise it is `training_failed`.

**Evaluation.**
1. **Behavioural depth curve** on the frozen roster's calibration and held-out cases: d = 1..16, the no-count question, 6 slots, d loops.
2. **The 02B instrument** on the frozen roster and counterfactual at 16 loops, with the no-count prompt and 6 slots; the same parity, repeat and drift checks.
   - The capture stores the one slot block for every stage, so `diagonal` and `lastblock` read the slots after k loops and `pass0` reads them before any loop.
   - `analysis.py` runs unchanged, with the same decoder search, controls and gate.

**Decision rules.**
- *Per-loop supervision matters:* D1's held-out behavioural accuracy over d = 9..16 exceeds D0's by at least 0.10, with a case-bootstrap 95% interval above 0.
- *Reusable transition* for an arm: it passes the 02B gate (`recurrence_specific_state`), and its `long_cycle` behavioural accuracy over d = 9..16 beats the best trivial 02B baseline with an interval above 0.
- Both are decided once, on these frozen sets. The 02C arm A (0.630 held-out over 9..16) is reported alongside as context, not as a test.

**Result (2026-11-06, code `5578566`): reusable transition shown for D1; D0 `training_failed`, so the supervision comparison is not decided as registered.**

- *Training validity*: D1 1.000 on 512 validation cases at depths 1-8 (valid); D0 0.639 (`training_failed`).
- *Reusable transition*:
  - D1: yes. It passes the 02B gate (`recurrence_specific_state`): `diagonal` decodes 1.000 at stages 9-16 against 0.583 for the shuffled-stage control, +0.417 [+0.297, +0.534], and on `long_cycle` it beats `pass0` by +0.875. Its behavioural `long_cycle` accuracy over d = 9..16 beats the best trivial baseline by +0.812 [+0.745, +0.875].
  - D0: no. It fails the gate (`no_iterative_state_signal`), and its `long_cycle` behaviour is +0.010 [-0.052, +0.078] over the trivial baseline.
- *Per-loop supervision matters*: D1 minus D0 on held-out behaviour over d = 9..16 is +0.401 [+0.286, +0.521]. That clears the margin, but D0 never learned the task in range, so the comparison measures learnability within 16 epochs, not depth generalization between equally trained arms. Reported, not decided.
- D1 answers all 1,152 roster cases correctly at every depth 1-16 with d loops; the 02C arm A context figure is 0.630 held-out over 9-16.

D0 was stopped at its epoch-4 checkpoint for a hardware upgrade and resumed; the first resumed step matched the logged loss and gradient norm exactly. `analysis.py` ran unchanged. Its native-readout row scores the first integer of `<<s->t>>` (the state before the step) as the state after it; the whole-step re-scoring is the post-hoc diagnostic `readout_shift.py` and enters no decision. Report: [lotus_bridge/results/algernon-02d/report.md](lotus_bridge/results/algernon-02d/report.md).

## H-A2e: how far D1's transition runs, and whether its slots carry the state (registered 2026-11-06, before any run)

**Why.**
- 02D's arm D1 answers every roster case at every depth 1-16.
- Phase 4 of the original brief (constraint-guided correction, compared with plain recurrence at matched compute) needs a regime where the plain recurrence fails. At depth 16 there is none.
- The 02B gate is a readout. It shows the state is decodable from the slots after every loop, not that the answer depends on it.

**Model.** The 02D D1 weights (`80eda0824cff`), fp32, no training. Code: `lotus_bridge/slot_patch.py`.

**1. Depth beyond 16.**
- Setup: the no-count question, one block of 6 slots and d loops, for d in {16, 20, 24, 32, 48, 64}, on the 72 calibration and held-out roster cases.
- The truth is the map iterated d times from the start.
- The largest absolute slot activation after the last loop is recorded per prompt.
- Reported: accuracy by family and depth, and "holds to depth D", the largest tested D with `long_cycle` accuracy at least 0.95 at every tested depth up to D. This part is descriptive; it has no pass/fail.

**2. Causal slot patching**, at d = 16 and k in {4, 8, 12}.
- **Recipients:** the 48 cases with bijective maps (24 `permutation` calibration, 24 `long_cycle` held-out). On these, different states stay different under the map.
- **Donors:** for each recipient A and each k, the donor B is the first case, in a seeded order over all 72 non-train cases (`random.Random(f"20261106:{A}:{k}")`), for which three states are pairwise different:
  - "follow": A's map iterated 16 - k times from B's state after k steps;
  - A's own answer;
  - B's own answer.
- **Intervention:** the hidden states of A's six slots after loop k are replaced by B's after loop k, both taken from unpatched 16-loop runs. A then runs on to 16 loops.
- **Instrument check:** a self-patch (A's own states after loop k) must reproduce A's unpatched answer in every pair; otherwise the patching result is void.
- **Decision, "the slots carry the pointer state":** answers equal "follow" in at least 0.80 of the (A, k) pairs, with a Wilson 95% lower bound above 0.5, and equal A's own answer in at most 0.10.
- **Reported alongside:** the rate equal to B's own answer, and the rates by k.

Phase 4 itself is not registered here. It needs a depth or input regime from part 1 where the plain recurrence fails.

**Result (2026-11-06, code `2c8589a`, weights `80eda0824cff`): the slots carry the pointer state.**

- *Depth*: all 432 prompts correct, the 72 calibration and held-out cases at d = 16, 20, 24, 32, 48 and 64, so the transition holds to depth 64. The largest slot activation after the last loop is 197.55 at every depth.
- *Patching at d = 16, k in {4, 8, 12}, 144 (recipient, k) pairs, every pair with a donor*:
  - answer = "follow" in 141, 0.979 [0.941, 0.993];
  - answer = the recipient's own in 3, 0.021;
  - answer = the donor's own in 0;
  - self-patch reproduces the unpatched answer in 144 of 144.
  - The registered decision, `slots_carry_state`, is positive.
- Post hoc: all three misses are recipients whose start state maps to itself. The 132 pairs without a fixed-point start follow in 132.

Phase 4 is still not run: D1 makes no errors up to depth 64, so a corrector has nothing to correct in this regime. Report: [lotus_bridge/results/algernon-02e/report.md](lotus_bridge/results/algernon-02e/report.md).

## H-A3 (Phase 4): constraint-guided correction of D1's slot state under noise (registered 2026-11-06, before any run)

**Why.** H-A2e showed that D1's six slots carry the pointer state and that its plain recurrence makes no errors up to 64 loops. Phase 4 of the brief compares plain recurrence with constraint-guided recurrence at matched compute. It needs a regime where the plain recurrence fails, so this experiment corrupts the slot state and asks whether an exact transition constraint, fed back into the loop, repairs it. The control arms get the same information without the constraint.

**Model.**
- F is the 02D D1 weights (`80eda0824cff`), frozen, fp32.
- pi is D1's own LM head on slot 2. That slot spells the state after the step (`<<s->t>>`), so pi takes the logits over the digit tokens 0-5 and normalises them; there is no trained probe.
- Code: `lotus_bridge/correct.py`.

**Corruption.** After loop j, Gaussian noise of scale sigma times the slots' RMS is added to the six slots. The noise is seeded per case, condition, loop and seed, and is identical across arms.

**Arms.** After every loop k, before the next loop reads the slots, each corrector arm adds G(x) to them. G is one linear map to 6 x 768, initialised at zero. Notation: e_k = p_{k-1} M is the map applied to the previous decoded distribution (p_0 is the question's start state), and p_k = pi(slots after loop k, after the noise).

| arm | G input x | what it has |
| --- | --- | --- |
| P | no corrector | plain recurrence |
| C | e_k - p_k | the soft residual of the trace constraint s_k = f(s_{k-1}) |
| I | [e_k, p_k] | the same two vectors unsubtracted: every bit of information C has, without the constraint's form |
| N | p_k | the decoded state only, no map |

- **Matched compute.** Every arm runs the same 16 loops of F. A corrector adds a 6-way readout and one small linear map per loop. C and N have 32,256 parameters and I has 59,904.
- **The violation signal and the gold successor.** C's residual exposes f of the previous decoded state, and I receives exactly that vector too.

**Training.**
- One G per corrector arm, all with the same recipe.
- Data: `random_function` maps on 6 nodes from `spec.case`, seed 20261107, excluding the frozen roster and counterfactual.
- Schedule: 600 steps of 8 cases, Adam at lr 1e-3. j is uniform in 1..7, and 25% of cases get no noise.
- Loss: cross-entropy of pi on the corrected slots after loop j and after loop j + 1 against the true states, backpropagated through F's loops into G only.
- This loss is the differentiable surrogate. The exact finite-field check is used only in evaluation.

**sigma\*.** Plain recurrence is run on 48 generated `random_function` cases (seed 20261108, not the roster) at d = 16 with j in {4, 8, 12}, for each sigma in {0.25, 0.5, 1, 2, 4, 8, 16}. sigma\* is the sigma whose plain answer accuracy is closest to 0.5, the smaller on a tie. It is fixed before training.

**Evaluation.**
- Cases: the frozen roster's 72 calibration and held-out cases, at d = 16.
- Conditions, every arm:
  - clean (no noise);
  - primary: sigma\* with j in {4, 8, 12} and 4 noise seeds, 864 runs per arm;
  - sigma\*/2 and 2 sigma\* with one seed each.
- Scored:
  - the answer;
  - whole-trace acceptance: the pi-decoded 16-step trace checked against the public machine by the exact BN254 R1CS checker in `spec.py`.

**Decision rules,** on the primary condition. Each difference is an arm's mean answer accuracy per case, differenced across arms, with a case-bootstrap 95% interval over the 72 cases.
- **R1 Constraint-guided correction repairs the state:** C - P >= 0.10, lower bound > 0.
- **R2 The constraint's form adds beyond its information:** C - I >= 0.05, lower bound > 0.
- **R3 Repair needs the map:** C - N >= 0.10, lower bound > 0.
- **Instrument:**
  - P must be correct on all 72 clean cases, with every clean pi-decoded trace accepted by the exact checker; otherwise the run is void.
  - A corrector arm under 0.97 clean accuracy is reported as damaging clean runs.
- **Reported alongside:** trace acceptance per arm, accuracy by j and by family, and the sigma\*/2 and 2 sigma\* conditions.

### H-A3 training diagnostic (recorded 2026-11-07, after training, before any evaluation record was read)

- **The correctors cannot reach the noise.** Measured on D1 on CPU, four calibration cases, loops 1-8:
  - clean slot RMS is 5.9-7.3, slot 2's norm is 166-207, and the digit rows of the LM head have norms 2.85-3.11;
  - the clean top-minus-median digit logit is 11.8-15.7;
  - at sigma\* = 16, the noise has a per-element std of about 105, a slot-2 norm of about 2,900, and about 300 std on each digit logit.
- **Trained G_C.**
  - Weights: |w| at most 0.223. Adam at lr 1e-3 for 600 steps bounds each weight's movement at 0.6.
  - Column norms over all six slots are 3.0-3.6, so the most it can move a digit logit at slot 2 is about 4, two orders of magnitude below the noise at loop j.
  - The noise at loop j wipes the previous decode. After that, e = p M no longer holds the right successor, so a correction at a later loop cannot recover the state either.
- **Training loss shows no trend.**
  - C: mean 13.63 over steps 1-100, 13.64 over the last 100.
  - I: 13.64 and 13.14.
  - The batches are identical across arms, and the same-step C - I gap is 0.28 on average.
- H-A3 is evaluated and reported as registered. H-A3b changes only the corrector's output scale.

**H-A3 Result (2026-11-07, code `f525c9a`, weights `80eda0824cff`, sigma\* = 16): R1, R2 and R3 not met; instrument valid.**

- *Primary condition* (864 runs per arm): P 0.494, C 0.487, I 0.505, N 0.500.
  - C - P = -0.007 [-0.028, +0.016]
  - C - I = -0.017 [-0.043, +0.008]
  - C - N = -0.013 [-0.031, +0.006]
- *Clean:* every arm answers 72 of 72. P's clean decoded traces are all accepted by the exact checker; I's clean trace acceptance is 0.958, with three rejected traces whose answers are still correct.
- *Trace acceptance under the primary condition:* P 0.073, C 0.104, I 0.101, N 0.014.
- *By family:* `sink` is 1.000 in every arm and condition; `long_cycle` is 0.13-0.16.
- By the diagnostic above, no corrector reaches the noise, so H-A3 does not test the constraint. H-A3b does.

Report: [lotus_bridge/results/algernon-03/report.md](lotus_bridge/results/algernon-03/report.md).

## H-A3b: H-A3 with the corrector's output in units of the noise (registered 2026-11-07, before H-A3's evaluation was read)

**Why.** By the diagnostic above, no arm in H-A3 can move pi, so H-A3 cannot test whether the constraint repairs the state. H-A3b asks the same question with correctors that can reach the noise.

**One change.** Every corrector's output is multiplied by a fixed kappa = sigma\* x r, applied at every loop:
- r is the mean slot RMS of clean D1 runs on the 48 calibration cases (seed 20261108), over loops 1-16;
- `correct.py scale` measures r before training and writes it to `scale.json`;
- a corrector weight of 1 then moves a slot element by about one noise std.

**Everything else is H-A3 unchanged:**
- F and pi;
- sigma\* = 16, reusing H-A3's calibration: same code, cases and seed;
- the arms and their inputs, the zero initialisation, and Adam at lr 1e-3 for 600 steps of 8;
- the training cases, loops and noise;
- the local loss;
- the evaluation conditions, the exact checker, the decision rules R1-R3 and the instrument.

The runs go to a new output directory.

**Predictions.**
- **C repairs: C - P >= 0.10 (R1).** At loop j, e = p_{j-1} M is the true successor, and C's residual is near zero on clean loops.
- **R2 is not met.**
  - I can represent C's map exactly (W_e = -W_p).
  - Writing e alone also suffices on clean loops, because there e = p.
- **N stays at P.** N sees only the post-noise decode.

**Why the outcome is not settled in advance.**
- A correction big enough to override the noise inflates the slots' scale. F's residual stream then carries the corrector's vector almost unchanged through the next loop, so the repaired state may stall rather than advance.
- I and N are nonzero on clean loops and can damage clean runs, which the 0.97 clean floor reports.
- If R1 fails here too, the conclusion is that, under this recipe, an additive linear corrector on D1's slots cannot use the constraint residual to repair a state wiped at sigma\*.

### H-A3b training diagnostic (recorded 2026-11-07, after C finished training, before any H-A3b evaluation)

- **Scale:** kappa = 110.78. The mean clean slot RMS is 6.92, over 768 loops of the 48 calibration cases.
- **C's training loss rose.** On batches identical to H-A3's, mean loss per 100 steps (steps 1-100, then each later hundred):

  | | 1-100 | 101-200 | 201-300 | 301-400 | 401-500 | 501-600 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | H-A3b C | 17.3 | 25.7 | 28.3 | 29.3 | 29.3 | 27.3 |
  | H-A3 C | 13.6 | 14.2 | 14.1 | 14.7 | 13.9 | 13.6 |

- **The trained G_C.** |w| is at most 0.084 and the column norms are 1.33-1.70. The bias has norm 1.79, which kappa turns into a constant push of norm about 198 added to the slots after every loop, whatever the input.
- **Post-hoc CPU check, six calibration cases, not the eval roster.** With C, pi's decoded trace collapses onto one digit on clean runs, for example `2222...` where the truth is `2052052...`. The generated answer is still right in three of the six.
  - On clean loops, C's input e - p is near zero, so the bias is its only active term.
  - The registered corrector therefore breaks the defining property of correction from a constraint residual: a satisfied constraint should produce no correction.
- H-A3b runs to completion and is reported as registered.

**H-A3b Result (2026-11-07):** R1, R2 and R3 not met. The instrument is valid, and C, I and N damage clean runs.
- Run details: code `29be916` (scale, train) and `1ce6afd` (eval; default `--corrector-bias on`, the same code path); weights `80eda0824cff`; sigma\* = 16; kappa = 110.78.

- *Primary condition* (864 runs per arm): P 0.494, C 0.527, I 0.528, N 0.501.
  - C - P = +0.032 [-0.028, +0.095]
  - C - I = -0.001 [-0.073, +0.073]
  - C - N = +0.025 [-0.084, +0.141]
- *Clean:* P 1.000 (every decoded trace accepted, records identical to H-A3's), C 0.556, I 0.556, N 0.681. All three corrector arms are below the 0.97 floor.
- *C's gain is noise-insensitivity, not repair.*
  - C's accuracy is 0.52-0.56 in every condition, clean included.
  - Its primary answer equals its own clean answer in 85.0% of runs (P 49.4%).
  - pi's final clean decode is the digit 2 in 69 of 72 runs, the collapse the diagnostic above predicted.
- *Trace acceptance under the primary condition:* P 0.073, C 0.044, I 0.227, N 0.399.
- *Exploratory, not registered: N cleans the decoded trace but breaks the answer text.*
  - N's trace acceptance beats P's: N - P = +0.326 [+0.287, +0.367] (primary) and +0.681 [+0.620, +0.741] (sigma\*/2), both case bootstrap.
  - 30% of N's generated answers are malformed, which caps its accuracy.
  - Where N's answer is a well-formed digit, it equals pi's final decode in 99.5% of primary runs.

Report: [lotus_bridge/results/algernon-03b/report.md](lotus_bridge/results/algernon-03b/report.md).

## H-A3c: H-A3b with bias-free correctors (registered 2026-11-07, before H-A3b's evaluation ran)

**Why.** The diagnostic above shows that H-A3b's correctors add a constant push of norm about 198 to the slots on every loop. For C, that makes the correction nonzero even when the constraint holds. H-A3c asks H-A3b's question with correctors that map zero input to zero output.

**One change.** Every corrector is a linear map with no bias term: C, I and N alike, with zero initialisation as before.
- C, I and N then have 6, 12 and 6 weights per output element: 27,648, 55,296 and 27,648 parameters in all.
- C's correction now vanishes when e = p, up to the softmax residue of a sharp decode.

**Everything else is H-A3b unchanged:**
- F and pi;
- sigma\* = 16;
- kappa = 110.78, reusing H-A3b's `scale.json`;
- the arms' inputs, Adam at lr 1e-3, 600 steps of 8;
- the training cases, loops and noise;
- the local loss;
- the evaluation conditions, the exact checker, R1-R3 and the instrument.

The runs go to a new output directory.

**Predictions.**
- **C does not damage clean runs.**
- **R1 C - P >= 0.10:** open.
  - The scale is now within reach: C's trained H-A3b weights, at column norms of about 1.5, already move slot 2 by roughly the noise's projection on the digit logits.
  - Whether F continues correctly from a corrected but inflated slot state is the open part.
- **R2 is not met.** I can represent C's map.
- **N stays at or below P.**

**Note (recorded 2026-11-07 after H-A3b's evaluation was read, while H-A3c trains; it changes nothing in H-A3c's procedure).**
- p and e are distributions, so N's and I's inputs always sum to 1. A bias therefore folds into their weights (W p + b = (W + b 1ᵀ) p), and removing it leaves their function class unchanged. Only their parametrisation, and so their training path, differs from H-A3b.
- C's input e - p sums to 0, so C is the only arm whose function class changes. A constant push is no longer expressible.
- "Maps zero to zero" therefore constrains C alone, because N's and I's inputs are never zero.

**H-A3c Result (2026-11-07):** R1, R2 and R3 not met. The instrument is valid. C no longer damages clean runs; I and N do.
- Run details: code `1ce6afd` (train and eval); weights `80eda0824cff`; sigma\* = 16; kappa = 110.78. The evaluation hit a CUDA `unknown error` at 23:02 and resumed from its per-run records.
- *Primary condition* (864 runs per arm): P 0.494, C 0.461, I 0.888, N 0.499.
  - C - P = -0.034 [-0.068, -0.000]
  - C - I = -0.427 [-0.531, -0.317]
  - C - N = -0.038 [-0.091, +0.010]
- *Clean:* P 1.000 (records identical to H-A3b's), C 1.000 (every clean trace accepted), I 0.889, N 0.722.
- *Predictions:* C does not damage clean runs (held). R1 was open: not met. R2 not met (held, reversed). N at or below P (held: N - P = +0.005).
- *Trace acceptance, primary:* P 0.073, C 0.156, I 0.999, N 0.075. I's decode is right at the noise loop in 0.999 of runs; C's in 0.207, P's in 0.199.
- *Post-hoc, not registered: the I result is confounded by the noise definition.*
  - CPU check on 12 generated cases, j = 4: I's corrections shrink F's slot RMS before the noise from 6.95 to 0.92 (N: 0.66). The noise is sigma times that RMS, so I's absolute noise is about 7 times smaller than P's and C's.
  - I's code for e = (previous decode) x M then outweighs F's own slot-2 signal on every loop: clean-loop margin +10 from F, +58 after I's correction. I runs the public map on its own decode.
  - C, zero on clean loops, faces the full noise. Its push moves the margin at loop j by +11 against a noise cost of about 240, and its training loss never fell.
  - H-A3b's biased correctors shrank the slot RMS too (C 2.70, I 1.74, N 0.49), which qualifies H-A3b's exploratory N result.

Report: [lotus_bridge/results/algernon-03c/report.md](lotus_bridge/results/algernon-03c/report.md).

## H-A3d: H-A3c with one fixed noise scale for every arm (registered 2026-11-07, after H-A3c's evaluation was read, before any H-A3d run)

**Why.** From H-A3 to H-A3c, the noise after loop j is sigma times the run's own slot RMS at loop j. H-A3c's post-hoc check found that arms correcting on every loop shrink F's slot RMS before the noise arrives, by up to 14 times. That applies to I and N, and to all of H-A3b's arms. The shrink shrinks their absolute noise with it, while C, which is zero on clean loops, faces the full noise. H-A3d compares the arms under the same absolute corruption.

**One change.** The noise std is sigma x r for every arm, case and loop, in training and in evaluation.
- r = 6.92 is the mean clean slot RMS in H-A3b's `scale.json`, so at sigma\* the std is 110.78 per slot element, equal to kappa.
- `correct.py --noise-ref fixed`.

**Everything else is H-A3c unchanged:**
- F and pi;
- sigma\* = 16, not recalibrated;
- kappa = 110.78;
- the bias-free correctors and the arms' inputs;
- the training recipe, batches and noise draws;
- the local loss;
- the evaluation conditions, the exact checker, R1-R3, the instrument and the 0.97 clean floor.

The runs go to a new output directory. P is evaluated again, because its noise scale changes from its own per-run RMS to r.

**Added secondary rule, about the control rather than C:**
- **R4 The map's information repairs at matched corruption:** I - P >= 0.10 on primary accuracy, with a case-bootstrap lower bound > 0, computed as R1-R3 are.

**Predictions.**
- **P's primary accuracy stays within 0.05 of 0.494.** Its own slot RMS at loop j is 6.4-7.0, close to r.
- **C stays at P, and R1 is not met.** C's slots keep F's scale, so its noise barely changes.
- **R4 is not met, and I's primary accuracy falls below 0.65.**
  - At the full noise, a margin shift of about +90 against a noise spread of about 240 rarely fixes the decode at loop j.
  - A wrong decode at loop j makes every later e wrong.
- **R2 and R3 are not met.**

**What it settles.**
- If R4 is met, the map's prediction written into the slots repairs the state at matched corruption. H-A3c's gap between I and C then reflects the residual's form rather than the noise interaction.
- If R4 is not met, H-A3c's I result is attributed to the noise-scale interaction, and no arm under this recipe repairs D1's slot state at sigma\*.
