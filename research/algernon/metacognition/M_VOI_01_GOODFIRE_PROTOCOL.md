# Algernon M-VOI-01: mechanistic value-of-computation experiment

**Status:** preregistration candidate, CPU-first, GPU/Silico unrun. 2026-11-04.
**Research question:** Does a compact model causally represent the *decision advantage of gathering another unit of evidence or running another reasoning step*, and can we calibrate that representation into a controller that beats cost-aware heuristics on genuinely held-out tasks?

## Evidence this proposal rests on

- H-M0 terminal-policy rereun on 1,000 worlds: expected-confidence-minus-cost utility 0.69584; information-gain baseline 0.69508; fixed heuristic 0.67952. Its probe cost was 2.845 versus 4.665 for IG. Subsequent independent stopping comparison ties a cost-aware greedy MAP heuristic.
- H-M1 rule grammar experiment: on its 1,000-world confirmation, learned policy utility 0.79158 versus myopic_expand 0.81497; difference -0.02340 with paired 95% interval [-0.03204,-0.01483]. H-M1 policy gets similar results under its shuffled-feedback intervention. H-M1 utility is **realized** weighted accuracy minus 0.028 times cost, whereas H-M0 utility uses **expected MAP probability** minus 0.004 times cost. Never pool these quantities.
- Qwen3-4B Hybrid ONNX/INT4 Ryzen NPU: 0-shot 0.81, weak/negative 1-4 shot response; serving is inference-only. A posterior/marginal prompt did not repair multi-digit formatting. This is a deployment diagnostic, not white-box mechinterp evidence.
- H-M2 claims/records remain unverified in this research tree; recover original artifacts before using numerical H-M2 claims for conclusions.

Source receipts:
- `research/algernon/metacognition/receipts/phase1_rerun_20261103/`
- `research/algernon/metacognition/npu_gym/receipts/`

## The mathematical object: advantage of computation

Let b denote a posterior over hypotheses, m the legal action mask, B the remaining budget, and z any sufficient externally supplied task specification. Define observable state x=(b,m,B,z).

Let `STOP` have terminal expected reward R(b,z). For a nonterminal probe action a, let its observation o have probability P(o|x,a), and its real cost be c(a). With a finite budget the optimal Bellman value is:

`V*(x) = max{ R(b,z), max_a [-lambda*c(a) + E_o V*(update(x,a,o))] }`.

Define continuation advantage:
`A*(x,a) = -lambda*c(a) + E_o V*(update(x,a,o)) - R(b,z)`.
Let `G*(x)=max_a A*(x,a)`; stop is optimal when G*(x) <= 0.

**Finite-horizon result (elementary).** Suppose all legal action values including STOP are estimated within epsilon of the exact optimal Q* at every reachable decision state. A greedy controller's conditional action regret is at most 2 epsilon, and its terminal expected regret over at most H decisions is at most 2H epsilon (sum the per-decision action gaps along its visited states). If the optimal action's margin over runner-up exceeds 2 epsilon, the estimated action must match it.

**Sufficiency limit.** Under the assumed known generative likelihoods, conditional independence, and specified utility, (posterior, available probes, remaining budget, public likelihood and cost specification) is a sufficient Markov state. No policy using only the same information can exceed the exact Bayes-optimal planner in expected terminal utility. A white-box controller's apparent extra information must be audited for privileged simulator access or leakage. This theorem says nothing about environments with unknown likelihoods or an incomplete belief state.

**Cost-adjusted information gain is not the same objective as terminal Bayes risk.** A probe can increase posterior entropy reduction while having low chance of changing the eventual decision. The minimum experiment is to predict G* or its action-level Q advantages, not only entropy.

## Critical confound in H-M1 to fix first

The exact H-M1 teacher accesses the **complete 12-rule generative prior**, while the learned policy's unexpanded features zero out the six unrevealed composition prior probabilities. This may create an unavoidable information mismatch. The policy does receive the public high-level q mixture and current primitive posterior, but not the unrevealed composition proportions.

Build TWO teachers:
1. **Privileged oracle:** original exact DP, all simulator priors and likelihoods.
2. **Information-matched planner:** integrate/marginalize hidden composition prior under the public hyperprior, using *only* information available to the student. Never replace missing hidden values with their simulator truth. Document the assumed hyperprior.

Compare each learned controller against information-matched DP and the two-step/myopic heuristic on the same action, likelihood, cost and stop contracts.

## CPU protocol before any new interpretability spend

1. Add deterministic Monte Carlo with seeds and distinct train/calibration/test **generator families**. Keep original H-M1 confirmation range untouched.
2. Construct labels for stop-vs-continue advantage G*, not just action IDs. Include cases with near-zero margins and explicit `STOP` advantage.
3. Fit a cheap calibrated classifier/regressor that reads externally observable sufficient state x. Score Brier, ECE, signed advantage error, chosen action regret and terminal utility.
4. Compare the existing policy, a monotone heuristic, calibrated VOI lookup, parameter-matched MLP, recurrent GRU with persistent *belief* state, plus a GRU with shuffled feedback. All should use identical legal action masks and cost accounting.
5. Evaluate at fixed budgets and report paired group-level CIs. Select architectures on new validation families and freeze parameters before looking at final unseen families.
6. Measure whether a recurrent hidden state adds information *conditional on the exact sufficient-state features*: conditional incremental accuracy or regret is the correct null. A recurrent benefit on deliberately incomplete observations is allowed, but should be called a belief-compression result.

Strong success condition: positive, paired, held-out terminal-utility difference against the best cost-aware heuristic under an information-matched interface, with improvement retained under seed and family shifts, and ablations showing controller dependence on actual observations.

## Goodfire/Silico mechanistic experiment: one hypothesis, one small canary

**Precondition:** validate a supported open-weight white-box model and activation access. This is a separate experiment from the quantized Ryzen NPU ONNX runtime. Pin full model revision, layer sites and tokenization. Prefer the Qwen3.5-0.8B target because it aligns with the planned compact graft; do not assume activations are interchangeable with the 4B NPU model.

Use Silico's current activation-harvesting, probing and causal-analysis capabilities. The deprecated 2024-era Goodfire feature-edit API is not an approved interface here.

For each matched task state, label:
- exact information-matched advantage G*(x), advantage sign, margin to runner-up;
- cost-adjusted action regret and terminal utility;
- whether another computation is expected to change the current optimal action;
- the valid action mask, actual probe costs, and belief-state entropy.

**Step A: representation test.** Collect activations from a small balanced roster of positive/negative stop-advantage cases and tricky near-boundary states. Train a linear/ridge probe for advantage sign and scalar G* on a train set. Test on held-out **generator families** with context/label-format controls and public posterior/feature baselines. Log layer-by-layer decodability and calibration. Only use a sparse dictionary or SAE if linear probes show a nontrivial signal worth dissecting.

**Step B: causality.** Freeze probe direction on calibration data. Run within-case activation swapping/ablation between cost-matched states where STOP optimality differs, then matched random-direction and norm controls. Check whether the intervention moves stopping probability and actual terminal utility in the predicted direction. Preserve the action mask and unchanged solver answers.

**Step C: graft relevance.** Fit the smallest controller that uses an extracted score to choose `PROBE`, `LDT_STEP`, `EXACT`, `REVISE`, or `STOP`; the exact host tool computes the result. Compare to the CPU information-matched planner and the best heuristic. Report tokens, RU, GPU time and extra verification cost in the utility denominator or as separate Pareto axes.

Stop if representation accuracy fails to beat an externally computed sufficient-state control, if causal interventions have no matched effect, or if action utility fails to improve. A good decodability score alone is not a success claim.

## RU budget gate

- **Before paid work:** freeze a one-page `run_request.json` containing model ID, layer set, max cases, max prompt/input tokens, max activations bytes, RU ceiling, expected completion outputs, and stop criteria; require explicit approval before spending.
- Prefer a **single small canary** to any open-ended agentic sweep. Suggested first canary: 64 independent task families with balanced margins, 2-4 layers, one activation extraction pass and one probe; budget figures must be priced using current Silico account rates rather than assumed percentages.
- Pass content-addressed references and small typed JSON receipts to the worker. The agent needs the relevant manifest and code path, not this complete conversation. Cache experiment context under an immutable manifest hash.
- No training/graft on the Ryzen NPU: use it as a frozen low-cost action selector only after the CPU policy is better than heuristic. Do not conflate prompt shots with weight learning.

## Pre-registered reporting

Mandatory: observed action utility, expected terminal utility, actual realized accuracy, paired deltas against the best heuristic, calibration, probe/step count, execution milliseconds, input/output tokens, activation sizes, RU consumption, retained abilities and intervention success rates.

Required provenance label: `CPU_deterministic`, `CPU_learned`, `NPU_quantized_inference`, `Silico_whitebox_observation`, or `Silico_whitebox_intervention`. Store an explicit `unrun` state until execution receipts exist.

**Do not publish an SAE or causal claim based only on its feature label.**
