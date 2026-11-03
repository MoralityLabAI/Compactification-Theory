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
