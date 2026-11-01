# SIML-01 CPU reference runner (v0.2)

The implementation is a finite-state **fitted transition-table recurrent learner**. It learns deterministic successor-code tables from one-step supervised transitions, rolls its learned state through previously unseen multi-step words, and passes proposed state-code updates through a fresh-score promotion broker. Training data include privileged microscopic successor states. The proposer is currently a fixed heuristic derived from exact finite collision diagnostics; its experiment policy has not yet undergone meta-training. This tests the plumbing and the state-capacity mathematics; it is not evidence about a 27B model, neural generalization, or self-modification.

Files: `microlearner.py`, `promotion.py`, `test_runtime.py`. The earlier `gate_math.py` and `test_gate_math.py` remain in place.

## Run

From the repository root:

```sh
python -m unittest discover -s research/self_introspective_meta_learning_v1 -p 'test_*.py' -v
python -O -m unittest discover -s research/self_introspective_meta_learning_v1 -p 'test_*.py' -v
python research/self_introspective_meta_learning_v1/promotion.py --output /tmp/siml01-run --blocks 4096
```

Choose a **new output directory** for an independent demonstration. The directory contains `state.json`, append-only `events.jsonl` and a `broker_private/` folder with random seeds, intentionally excluded from version control. A single-writer broker is required. Untrusted proposer and broker must execute in distinct processes with separately enforced filesystem permissions in the next iteration; the v0.2 Python class does not provide that isolation.

`Broker.verify_chain()` verifies the event hash chain against the state snapshot. `Broker.verify_scoring()` additionally reruns every successful/inconclusive/rejected evaluation from the broker's retained private seed, checkpoint and comparator models, reproducing its means and confidence intervals. An error during an attempt consumes its global attempt index. Source and graph digests are checked on restart. A small batch can yield `insufficient_evidence`.

## Finite task graph and prediction contracts

| Task | Microscopic state | What is learned or evaluated | Expected exact accuracy |
|---|---|---|---|
| T0 | `(y,b,c)` | Present bit y | 1 initially |
| T1 | `(y,b,c)` | `flip_b -> reveal_b` or `flip_y -> reveal_b`, excluded from one-step fit | 1/2 with y-only state; 1 with `(y,b)` |
| T4 | `(a,b)` in `Z31²` | Future sum after shear `(a,a+b)` | 1/31 with present-sum-only code; 1 with joint `(s,a)` |

The model's learned one-step tables are maximum-count successor codes under uniform state training. A tie chooses the lexicographically smallest code. The state code is updated recurrently using the fitted table for each action. The T1 test uses compositions never fitted as such. The T4 update expands its code from 5 to 10 fixed bits and fits a one-step shear table. This is a *synthetic supervised information-acquisition experiment*: labels, state oracles and generator semantics are known and exhaustive.

On the configured graph distribution (`T0=.25`, `T1=.45`, `T4=.30`) the exact baseline mean accuracy is `.25*1+.45*.5+.30/31`, or about `.484677`. The fixed heuristic first expands the bit state to `(y,b)` and then adds a joint shear state `(s,a)`; the final population score is 1. These exact finite-domain values are separate from each fresh-batch estimate and its confidence bound.

## Conditional promotion calculation

Each registered attempt t receives `alpha_t=delta/[t(t+1)]` with `delta=1/20`, split equally across predeclared endpoints. The test creates 4096 independent **initial world** blocks; each block samples one world per task family. The aggregate paired block difference includes family weights and lies in [-1,1]. The `target` endpoint compares to the parent, and every protected family compares with its immutable anchor. The two-sided fixed-n Hoeffding radius for endpoint j is `sqrt(2*log(2/alpha_tj)/n)`. With default thresholds, aggregate gain must exceed .025, target gain must exceed .05 and retained-family degradation must be within .08. A confidence interval crossing a floor yields `insufficient_evidence`; an upper interval already below it yields `rejected`.

This is a conditional validity statement: no other learning process may access the private promotion data before the candidate is frozen. Fresh samples from the same eight/961-state spaces provide finite synthetic evidence; they do **not** test novel state semantics, distribution shift, or external task generalization. A model able to memorize the entire finite public domain can exploit that structure legitimately, so report the exact exhaustive population score and avoid calling this a neural-learning discovery.

## What the v0.2 result establishes

- Trained one-step transition tables compose on two held-out word types when the compressed state retains the hidden bit needed for the subsequent action.
- Expansion to the exact joint shear code repairs a predictable future-output loss in the 961-state fixture.
- The evaluator applies its paired bounds to a single frozen candidate, preserving stable competence anchors and failing closed on malformed proposals, source mismatch, tampered event records and corrupted replay seeds.
- A registered but unhelpful state enlargement leaves the incumbent unchanged. A regressive reduction after a promotion is rejected.

The next implementation step is **a neural recurrent student** with the same graph and measurements, followed by an actual 27B proposer adapter. Compare the introspection-guided proposer to fixed, random and behavioral-only policies on independent graph variants under matched full compute. FAW/FRC remain deferred.
