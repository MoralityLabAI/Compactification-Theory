# Continual learning and epistemic control for Algernon

**Status: design cross-reference, 2026-11-03.** This document extends the research questions in [ARCHITECTURE.md](ARCHITECTURE.md) and [PROTOCOL.md](PROTOCOL.md). It records an *unrun candidate* experiment. The existing H-C, H-R, H-W, H-J registry and their gates remain unchanged.

## Origin and scientific scope

A user-supplied continual-learning video transcript (timecodes 0:00–25:15) motivates four practical requirements: selective retention (0:00–7:28), reconsideration of assumptions (9:43–12:23), choosing research directions under limited compute (14:50–18:21), and learning priorities from experience (18:30–22:44). The video is a conceptual prompt, not evidence that any proposed controller works or that a particular brain mechanism is necessary for AGI.

Algernon's current experiments establish separate machinery for exact state transitions (H-C), recurrent latent refinement (H-R), sparse receiver correction (H-W), and their possible interaction (H-J). Their gates can succeed without producing a system that chooses what to learn next. That capability needs its own falsifiable evaluation.

## Fourth clock: experiment time

The architecture already distinguishes token-time DeltaNet memory S_t, recurrent algorithm-time state z_k, and persistent sparse parameter edits. Add an *experiment-time* state (b_e, M_e, B_e): a set of beliefs with provenance and uncertainty, bounded episodic/replay memory, and remaining resource budget.

Candidate controller actions include selecting a probe or hypothesis, allocating refinement steps K, invoking an exact kernel, retrieving/replaying an experience, applying a separately approved parameter update, or halting an unproductive investigation. Keep exact-machine state, latent reasoning state, episodic memory, and model weights separately measurable.

~~~text
a_e ~ pi_psi(a | b_e, M_e, B_e, task_e)
o_e = run_probe(a_e)                 # observed evidence and measured resource use
b_(e+1) = update(b_e, o_e, provenance(o_e))
M_(e+1) = retain_or_replay(M_e, o_e, capacity)
B_(e+1) = B_e - measured_cost(a_e)
~~~

For a known simulator, a Bayesian filter can provide an *oracle comparator*. The controller need not assume oracle likelihoods are available in real tasks. Evidence from copies of one source must be treated as correlated; source-independent confirmations provide a different likelihood update. Quantify calibration and unwarranted updates.

Suggested evaluation objective (pre-register coefficients and budgets before fitting):

~~~text
J = held_out_task_utility
    - lambda_cost * total_measured_probe_and_inference_cost
    - lambda_forget * retained_skill_regression
    - lambda_error * unsupported_belief_or_weight_updates
~~~

Log predicted value of information, realized information gain (where simulator truth permits), decision cost, source dependency, and the final held-out result. A high predicted gain alone is insufficient.

## Candidate off-registry hypothesis H-M0: learned research pruning

**Claim to test:** Within a fixed evidence and compute budget, a trained compact controller chooses probes that yield better held-out generalization and lower cumulative regret than simple scheduling and prompt-only controls. Its choices should adapt when previously promising hypotheses lose support.

**First substrate:** [Pluto-Gym](../pluto_gym/README.md) already has a tiny CPU trainable student, bounded DRAM replay, explicit curriculum designer, deterministic synthetic environment, held-out rule combinations, and logged designer decisions. Reuse its manifest/receipt pattern and provider-neutral designer interface. Keep a new epistemic-control task module isolated from current Pluto-Gym training baselines.

**Proposed evaluation families:**

1. **Selective consolidation:** stable versus rapidly changing synthetic facts, noisy measurements, and clusters of copied testimony. Score posterior calibration, incorrect overwrite rate, retention accuracy, and adaptation lag.
2. **Budgeted experimental choice:** seeded hypothesis families with probes of unequal cost and variable informativeness. Give each arm the same observation and spend cap. Score held-out decision regret, realized information per total compute, and the ability to stop low-value probes.
3. **Framework revision:** hide some causal rule families from the initial hypothesis menu, then permit structured hypothesis proposals. Score genuinely new held-out predictions, independent falsification experiments, and subsequent transfer. Record whether an apparent proposal was available verbatim in the initial menu.

**Controls:** random probe schedule; fixed heuristic/greedy uncertainty schedule; Bayesian oracle with access to the simulator's true likelihoods (upper bound); prompt-only frontier or local-model selector with matched information access and metered inference; learned controller with shuffled feedback; learned controller with frozen policy; replay-only student. Separate the cost of gathering oracle evidence from cost of choosing an action.

**Metrics:** held-out accuracy/regret after each fixed spend interval, calibration (Brier/log loss where applicable), false-update rate, cumulative probe cost, wall time, tokens/FLOPs, replay bytes, retained prior-task accuracy, and run-to-run dispersion. Train/calibration/test must be split by generator family and seed, including independent final rule families. Confidence intervals cluster on task generator.

**Strong-claim failure:** learned selection provides no repeatable held-out gain over a cost-matched simple policy; its gains vanish with shuffled-feedback control; or improvement comes exclusively from unmetered external inference or test leakage. Report partial results under their narrower labels.

## Integration sequence

- **Now, CPU-first:** complete Algernon Phase A/B and run an isolated small H-M0 simulator on Pluto-Gym-style receipts. Register a new protocol/version before first evaluation, preserving the four original unrun hypotheses.
- **After H-R passes:** let a tiny controller choose K and receiver mode from calibration-derived uncertainty. Compare adaptive K against a fixed-K policy at equal *measured* compute; test with the gate disabled and the decision policy shuffled.
- **After H-J passes:** choose between a compiled exact operation, recurrent heuristic refinement, and another evidence probe, using typed receipts. Score tool accessibility parity, exact fidelity, and total processor cost.
- **Only after memory and adaptation checks:** evaluate persistent changes in replay state or low-rank adapters across task episodes. Distinguish updates to weights, bounded external memory, transient activations, and purely in-context reasoning.

## Related project boundaries

- [HybridTRMLDT recurrence taxonomy](https://github.com/MoralityLabAI/HybridTRMLDT/blob/master/docs/looped_transformer_trm_ldt_taxonomy.md) identifies the learned mode/schedule and projection operator as potential control points. Its current VPD role is instrumentation.
- [HybridTRMLDT fixed-edit replication PR #5](https://github.com/MoralityLabAI/HybridTRMLDT/pull/5) remains an independently measured H-W prerequisite. A frozen human- or agent-discovered patch is evidence for a repair method, with no implied on-device continual learning.
- The published Qwenputer construction is an exact execution substrate. Continued training, uncertainty updates, and autonomous experimental selection require their own receipts and intervention controls.

**Next coding-agent instruction:** keep the original Phase A/B handoff intact. As a separate CPU-only branch or follow-on PR, propose the smallest H-M0 simulator and scoring script using Pluto-Gym-compatible manifests. Report simulated observations with provenance, and leave all capability claims unrun until measured.
