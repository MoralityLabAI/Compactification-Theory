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

## H-A2c: LOTUS's own recipe on the pointer task, loop-to-step tying (registered 2026-11-05, before any training)

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
