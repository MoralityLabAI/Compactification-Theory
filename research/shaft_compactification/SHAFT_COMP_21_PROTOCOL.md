# SHAFT-COMP-21: Supervised temporal-state learning and the 90% capability gate

**Registered November 11, 2026 before training COMP-21 models or inspecting results.** Source: SHAFT-COMP-20 exact persistent-latch automata and its failed 24-unit GRU implementation. This study uses local synthetic tasks and fully retained weights.

## Question and causal intervention

In COMP-20, the finite automaton distinguishes a fixed task ID g, previous query p, and persistent toggle-latch l. A two-query behavioral signature separates temporal states that goal-only and one-query signatures merge. The eight COMP-20 trained GRUs did not learn the latch task reliably and therefore failed to support the probe claim.

COMP-21 intervenes on **training supervision** while holding architecture, training stream, optimizer, and evaluation fixed. A baseline arm learns output bits only. An auxiliary arm receives labels for the hidden post-step latch and previous-symbol register from the known synthetic automaton. The auxiliary heads are used only during training; the **native task output remains the same unchanged TaskMemoryGRU readout**.

## Frozen design

- State model: original 24-unit GRU with embedding for each of 64 six-bit task IDs, same two-layer 24-width tanh output head from COMP-20. Inputs are six one-hot tokens. Each model processes 12 tokens after the initial task embedding.
- Task families: repeat_toggle (trigger p=q) and successor_toggle (trigger q=(p+1) mod 6), with initial p=0, latch=0. The output at each step is goal bit indexed by the current token XOR the *pre-update* latch. The post-step latch is the cumulative trigger parity and previous-symbol target equals the current input.
- Arms: output_only (lambda_latch=0, lambda_previous=0) and aux_state (lambda_latch=1.0, lambda_previous=0.25). Each aux head reads the recurrent 24-unit state after the current token: latch uses a linear 24->1 BCE readout, previous-symbol uses a 24->6 CE readout. The loss is task BCE plus these weighted auxiliary losses. During evaluation, auxiliary heads are excluded from the primary model, and no oracle latch/previous bit is supplied at inference.
- Both arms train **3,200 Adam steps**, LR 0.01, batch 128, sequence length 12, clip norm 2.0, with identical per-seed random task/input streams. Initialize the shared GRU/embedding/output-head identically within each paired arm, using the same torch seed. Auxiliary heads receive independently fixed initialization without disturbing the common shared-model initialization. Model uses three CPU threads and deterministic torch algorithms.
- Model-seed assignments:
  - repeat_toggle: calibration seed 2201; held-out seeds 2202 and 2203.
  - successor_toggle: calibration seed 2211; held-out seeds 2212 and 2213.
  - Both arms use each assigned seed. All earlier COMP-20 seeds are excluded.
- **Base-task capability gate**, fixed now: at least 90.0% native task-bit accuracy on 1,024 fresh 12-token sequences, using a separate RNG seed. Record 32-token fresh accuracy as a non-gating secondary test. Include and report every trained model, including gate failures.

## Gate-first measurement protocol

Train the calibration seed for each family in both arms, report task accuracies and joint/auxiliary training losses. Freeze model hashes and all gate outcomes in a Git commit before training either family’s held-out seeds. No replacement seeds or optimization adjustments are permitted.

For a gated model only, train equal-output-count goal, immediate and two-query linear probes on the frozen GRU states. Use the COMP-20 recipe: 12 contexts per goal at recurrent depths 4/8/12, 400 Adam steps at LR .04, BCE on 36 targets. Goal control is six canonical goal bits repeated sixfold (privileged task supervision); duplicate-immediate control is the six native one-query bits repeated sixfold; the history probe learns the 36 native final-step responses to all ordered two-query continuations. The probes receive no oracle latch/previous state. The three output matrices and training procedure have identical parameter counts.

For a gated model only, score 64 goals × 8 same-goal paired independent natural prefixes at each state depth 12 and 40. Eligibility requires matching current native six-query binary outputs and matching the duplicate-immediate probe bits. Compare native behavior after two independently sampled shared 32-step continuations at checkpoints 8/16/32, as in COMP-20. Also compute the deterministic automaton’s delayed divergence label from tracked latch and previous input. The directional probe score for each probe is the same five-token tangent-cocycle energy as COMP-20, normalized by 36 output dimensions. The diagnostic input sequence is independent from both outcome continuations.

**Primary causal capability endpoint:** improvement of aux_state minus output_only in 12-token native task accuracy within the same model seed, reported by family, all seeds, and pooled. Successful learning means >=90% gate pass in at least one aux_state held-out seed per family and at least 3/4 total aux_state held-out seeds. Use paired comparison with baseline; do not claim success if the capability gate is missed.

**Conditional probe endpoint:** only if both probe types are evaluable on at least two gated held-out models with positive and negative divergence events, estimate native behavioral divergence ROC-AUC(history) versus ROC-AUC(goal) and ROC-AUC(duplicate) and separately assess oracle/native label agreement. Report all events, denominators, seed-specific AUCs. Treat the probe result as descriptive, conditional on the preregistered gate, and never promote a failed model’s probe AUC into a positive claim.

**Calibration thresholds and operating point:** if calibration models pass the capability gate, choose 90th percentile of negative directional history/goal/duplicate scores (NumPy method=higher; predict iff score > threshold) on their calibration pair data. Freeze thresholds in a *separate commit* before held-out training. If neither calibration aux model passes 90%, record an explicit null threshold and do not make an FPR transfer claim. The user’s COMP-20 frozen source thresholds remain unchanged as historical controls.

## Exact mathematical check

Provide an exact finite-state reference showing post-step latch and previous-token bookkeeping; demonstrate that the pair of state variables suffices with the goal to simulate all future outputs, while goal-only and one-step behavioral signatures do not separate the temporal states. Add Lean theorems for the finite update and for the conditional restoration theorem using an exact state code. Keep any learning-based empirical conclusion distinct from these finite proofs.

## Reproduction and limits

Pin task streams, base/probe/aux-head hashes, accuracy, complete per-pair raw receipts, and exact finite references. The complete experiment should replay deterministically on PyTorch 2.10.0+cpu. Light GitHub Actions runs standard-library tests, frozen receipt audit, and Lean 4.28 only, not expensive retraining.

The experiment is confined to known synthetic tasks in local models. It does not test an unknown malicious objective, real device behavior, covert replication, resistance to weight deletion, or transfer of a frontier model.
