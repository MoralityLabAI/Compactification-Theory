# BGI-01H: certified progress controller for model-recommended repairs

**Prespecified 2026-10-30 before H model inference.** This experiment follows the completed [BGI-01G PR #24](https://github.com/MoralityLabAI/Compactification-Theory/pull/24) and leaves its source, task scores, prompts, and model receipts unchanged.

## Why this experiment exists

G's Qwen2.5-1.5B policy repair generated **7/8** complete recoveries when given its current policy and missing-guard witness. Giving extra feedback about its previous edit **reduced** that to **3/8**. In the latter mode it repeatedly selected already restored guards. The full previous feedback trial is frozen in G, including one residual unauthorized-access counterexample from initial damage.

This experiment replaces the verbose model feedback with an explicit, bounded **choice of exactly one missing guard**. It measures model-directed repair separately from the trusted software fallback. The same finite Boolean invariant verifier, protected five-guard reference and hash-chained ledger are imported from [BGI-01F](../verified_repair/README.md).

## Two modes and exactly one distinct model suggestion per round

**Model-only:** The Qwen model gets a machine-produced ordered list of current missing protected guards and an untrusted peer note. It must produce precisely one JSON object with a chosen field name. If the selected field is missing, a fixed external executor sets it to true. Invalid, stale or no-op selections cause the round to stall. The verifier still rejects new policy weakening.

**Model plus independently certified fallback:** The exact same prompt and same model attempts that field selection. If it fails to make progress while any missing guard remains, the trusted controller selects the *first current missing guard* and restores it. Both the ineffective model proposal and the external fallback appear in the linked ledger with separate event hashes. Fallback completions are never counted as model-generated successes.

The experiment runs **three rounds** for each of the eight previously used BGI-01G synthetic fault states. Two added controls begin with four or five missing guards, exceeding the three-round recovery horizon. Thus: **10 scenarios × 2 modes × 3 rounds = 60 model callbacks**, with identical prompts allowed to share deterministic inference and the cache counts disclosed.

This is a deliberate *exploratory development study* on already inspected faults. There is no independent blinded model-evaluation set in this follow-up.

## Exact progress theorem inside this declared finite system

Let the integer potential be the number of false protected guards:

`V(p) = |{k in PROTECTED : p[k] = False}|`.

Under an authenticated, accurate missing-guard observation, availability of the external executor, and accepted verified changes only setting a false guard true:

`V(p_next) = max(0, V(p_current) - 1)`

for every fallback-enabled round. The model may select the repaired guard itself; otherwise the controller supplies one. Consequently every state with at most three missing guards reaches the full protected reference in three rounds, independently of the model's text. States with four or five missing guards cannot complete within three single-change rounds.

The **per-step** claim is checked exhaustively across all **32 policy configurations × 11 valid/adversarial proposed selections = 352 transitions**. It is an induction argument about these defined transitions, not a claim that Qwen acquired an internal repair drive. The imported F certificate separately verifies every 352 typed policy transitions against no-regression and access containment.

Every round logs its actual before/after potential, parsed selection, whether the model repaired a missing guard, any external substitution, both separate ledger events and their SHA-256 chains. The trusted actor identity and decision are supplied by the harness; the model cannot grant itself write authority by emitting text.

## Model and outcome measures

Pinned inference: `Qwen/Qwen2.5-1.5B-Instruct@775b11afaf83e0dc75bd5abaf90133e47b3ec082`, CPU float32 and deterministic greedy generation, max 80 new tokens for every model suggestion.

For each arm report full recovery among the eight in-budget states, missing-guard potential after three rounds, model-generated effective repairs, externally substituted repairs, redundant/stale/malformed selections and unauthorized-access witnesses remaining. For the two out-of-budget states, report how many protected flags remain after the strict horizon.

Model-only performance measures Qwen's ability to select genuinely missing guards. Model-plus-fallback performance measures an engineered **progress guarantee**, whose causal mechanism is partly deterministic external substitution. Compare model initiative separately.

The engineering checker guarantees preservation of the finite no-regression policy; safety beyond these five flags and the observed sixteen access conditions is outside its proof. The study does not use gradient training, human preferences or a persistent moral self-model. All model task outcomes are exploratory and require independent review before publication claims.

## Commands

From the repository root, Python standard library:

```bash
PYTHONPATH=research/bgi_attractors_v1/verified_repair:research/bgi_attractors_v1/factored_admission:research/bgi_attractors_v1/progress_repair python -m unittest discover -s research/bgi_attractors_v1/progress_repair/tests -v
```

For real CPU model inference, install the same pinned torch/transformers requirements as [BGI-01G](../factored_admission/README.md), then run:

```bash
python -u research/bgi_attractors_v1/progress_repair/run_model.py
```

GitHub Actions checks independent finite progress and no-regression oracles, then runs genuine 1.5B inference and uploads the entire JSON receipt, whose canonical SHA-256 digest is independently recomputed.
