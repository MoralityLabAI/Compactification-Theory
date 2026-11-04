# LSV-DISC-01: preregistration for philosophy-guided mathematical discovery

**Draft protocol, 2026-11-04.** No comparative AI-agent runs have occurred. LSV-01's established small-gain mathematics is the seed positive control and is excluded from discovery outcomes.

## Question

Does a search prior inspired by Michael Levin's multiscale agency and goal-directed morphogenesis yield more independently verified and practically useful mathematical techniques per fixed inference-plus-experiment budget than a topic-matched neutral question generator?

## Paired arms

- **Levin-prior generator:** directed toward universal conditions for local-to-global competence, goal recovery after perturbation, preservation of goals under abstraction, and transfer across embodiments.
- **Neutral generator:** same model, tools, mathematical fields, background definitions, token budget and number of proposals; prompted to identify valuable unsolved questions in dynamical systems, numerical algorithms, distributed control and recurrent computation without the special agency/morphogenesis emphasis.
- Randomize matched runs and independently freeze prompts, model version, seeds, timestamps, candidate identifiers, sources, software environment and budget caps before scoring.
- Reviewers assessing correctness, novelty and real-world improvement must remain blind to generator arm.

## Primary endpoint

Verified useful original techniques per all-in inference/compute cost, requiring all of:

1. **Correctness:** formal proof checked by Lean or by independent qualified mathematical reviewers. Sampled rational examples alone receive a separate status.
2. **Novelty:** a prior-art search independently assesses whether the technique or theorem is a rediscovery, restatement or genuinely new advance. Ambiguous novelty is recorded as unresolved.
3. **Utility:** measured improvement under a frozen workload, against existing strong baselines and on a held-out task split. Account for inference, training, monitoring and certificate overhead, not just the inner-loop cost.

Track invalidated claims and failed candidate attempts, not just successes.

## Secondary endpoints and controls

Record correct-theorem rate, counterexample discovery rate, proof effort, actual runtime/latency/VRAM/communication, performance under perturbation, and transfer to an unrelated problem sharing the formal assumptions. Use confidence intervals for outcome differences and report matched total compute.

Negative controls should include: unstable cross-coupling, stable individual modes with destabilizing switches, hidden-goal quotients, starvation of one coordinate, stale reads, unbounded perturbations, and loose Jacobian estimates mistaken for globally verified bounds. A generator that proposes an unsupported theorem should be penalized after a valid counterexample.

## Planned deployment

Start with finite rational dynamical systems for inexpensive independent checks. Freeze a candidate set before running tests. After this stage, use a held-out recurrent neural model and a separate distributed-control benchmark. Prove or conservatively bound the derivative envelope over each declared invariant region, including intervention paths. All unsuccessful proof attempts and inadequate bounds stay in the audit ledger.

## Meaning of an eventual result

A positive result would support a problem-selection heuristic under the tested budgets and domains. A negative result would constrain that heuristic. The experiment does not infer consciousness or metaphysical claims from certificates for dynamical systems.
