# SHAFT-COMP-22: Factorized temporal decoder and the effective-goal gauge

**Registration:** November 11, 2026, before the new decoder training and evaluation. See [preregistered protocol](SHAFT_COMP_22_PROTOCOL.md), commit `f539c28e0dc4409cc76741a48c5d39ba9531f9c0`.  
**Result:** An exact finite state-space reduction from 768 canonical triples to 384 distinct behavioral classes, together with a positive but sharply limited result for a privileged-label neural factorization on four previously studied, frozen synthetic GRUs. Every model retains its full weights and explicitly supervised task embedding.

## 1. Exact behavioral quotient

The persistent latch state is `(g,l,p)`, with a six-bit goal table `g`, parity latch `l∈{0,1}`, and remembered previous token `p∈{0,...,5}`. For query `q`, the system outputs the **pre-update** bit `(g >> q & 1) XOR l`, then updates `l ← l XOR T(p,q)` and `p ← q`. The registered families use `T(p,q)=[p=q]` and `T(p,q)=[q=(p+1) mod 6]`.

Define the **effective goal** `b = g XOR (63 if l else 0)`. The full observable computation factors through `(b,p)`:

```
output(q) = bit(b,q)
next_b    = b XOR (63 if T(p,q) else 0)
next_p    = q
```

The output after two fresh queries `(a,c)` is therefore exactly

\[
B_2(a,c)=\operatorname{bit}(b,c)\oplus T(p,a).
\]

Complementing the goal table while toggling the latch, `(g,l,p) ↦ (g XOR 63,l XOR 1,p)`, leaves `(b,p)` unchanged. The original canonical states occur in pairs under this symmetry.

[The exhaustive integer reference](factorized_latch_finite.py) verifies for **both** families:

- 768 canonical triples and exactly **384** distinct `(b,p)` behavioral codes.
- Exactly 384 distinct two-query signatures, so two queries distinguish every different effective behavioral code.
- Exact canonical and reduced reconstruction for all **768 × 36** possible two-query cases per family.
- The transition/output factorization for all 768 states and six possible next inputs, and the gauge witness across every future word of length 0–3. The transition-congruence identity extends this to arbitrarily long future words.
- Previous-token distinctions are invisible to the immediate monitor but separable by two-query tests.

[FactorizedLatch.lean](lean/Shaft/FactorizedLatch.lean) formalizes the exact output factorization, the transition congruence, and its finite-continuation consequences. This is an **identifiability result**: the canonical latch and uncomplemented goal table cannot each be uniquely recovered from behavior alone without an external gauge convention. The effective table and previous token define the minimal observable quotient for the two tested automata.

## 2. Frozen-model experiment

The experiment uses four COMP-21 `aux_state` models that had passed the **90% fresh 12-step task-accuracy gate** before COMP-22. They are the repeat-toggle models 2202/2203 and successor-toggle models 2212/2213. Their checkpoints are loaded from the previously archived COMP-21 receipts, and each model's stable SHA-256 digest is checked. Model weights and task training remain unchanged. This is a **new probe/evaluation-data study on previously examined model seeds**, not a new-model holdout.

For each frozen GRU, create the same **2,304 independent training states**, comprising 12 contexts per goal at each of recurrent depths 4, 8, and 12. Each new probe head is trained for 400 Adam steps at learning rate 0.04, with bitwise BCE or six-way previous-token CE. The common state representation has 24 dimensions.

Four decoder strategies are then evaluated on **32 fresh contexts per goal** at each of depths 12 and 40. Across the four models this yields **16,384 test states**, all from seeds disjoint from probe training and previous COMP-21 evaluation contexts.

| Decoder | Trainable readout parameters | Training targets | Reconstruction |
|---|---:|---|---|
| Direct36 | 900 | Native 36-bit two-query signature | Threshold 36 logits |
| Canonical factorization | 325 | Oracle 6-bit goal, oracle 1-bit latch, oracle 6-way previous token | Apply exact finite transition law |
| Behavioral factorization | 300 | Native 6-bit immediate signature, oracle 6-way previous token | Apply same finite law |
| Full finite oracle | 0 | Ground-truth `(g,l,p)` | Exact finite formula, used only for reference |

**Supervision matters.** The canonical and behavioral factorizations use *privileged automaton state labels*, at least for previous-token classification and additionally for goal/latch in canonical. The direct control is trained solely against native model behavior. Their parameter counts also differ. Therefore any advantage is attributed to the **combined structural decoder and richer supervision**, not a pure architecture comparison holding training information constant.

## 3. Main held-out results

### Depth 40, primary comparison

| Method | Native 36-bit micro accuracy | Native complete-signature accuracy | Complete matches / 8,192 |
|---|---:|---:|---:|
| Direct36 | 52.0894% | 0.0000% | 0 |
| Canonical factorization | **67.5581%** | **7.2754%** | **596** |
| Behavioral factorization | 52.4180% | 1.6724% | 137 |
| Full finite oracle | 95.6763% | 65.3564% | 5,354 |

The **registered primary contrast** is canonical minus direct complete-signature native accuracy at depth 40: **+0.072754**. A 1,000-draw, per-model-stratified bootstrap gives interval **[+0.067505,+0.078125]**. Each of the four model seeds has a positive difference:

| Model | Registered difference in complete native signatures |
|---|---:|
| Repeat 2202 | +0.014648 |
| Repeat 2203 | +0.098633 |
| Successor 2212 | +0.087402 |
| Successor 2213 | +0.090332 |

The registered support criterion is met *for this particular privileged supervised factorization on four previously studied models*.

### Depth 12, independent evaluation contexts

The same pattern occurs at depth 12: direct 36-bit micro-accuracy 52.3668% and zero complete signatures, canonical 67.6198% and 631/8192 (7.7026%) complete matches, behavioral 52.7293% and 140/8192 (1.7090%), and oracle 96.0941% with 5536/8192 (67.5781%) complete matches against native behavior.

### Which internal state information was available?

The new temporal heads decoded **latch and previous symbol at 100% accuracy** on these fresh states. The six canonical goal bits were individually correct **66.9617%** of the time at depth 40, while entire canonical goal tables were exact in **8.5083%** of states.

The gauge-invariant effective-table head, trained against the frozen model's own current responses, reached only **2.6123%** complete-table agreement with those current native responses. Its low accuracy accounts for the poor fully behavioral factorization result. The canonical decoder is stronger primarily because the privileged goal labels induce a more useful representation of task-specific bits.

The ideal finite automaton achieves **95.6763% native bit agreement** and **65.3564% complete signature agreement** at depth 40. These values bound what perfect automaton-state recovery alone can reproduce about the approximate neural model; a trained GRU can deviate from its ideal task even if it passed a fresh 12-step capability gate.

## 4. Interpretation

The exact finite theorem shows a causal sufficiency result for `(effective_goal, previous_token)`. The neural experiment establishes that the trained models expose their latch and previous-token registers to linear probes, while canonical six-bit goal retrieval remains difficult. A structural reconstruction using privileged goal/latch/previous supervision outperforms direct 36-bit probe training on the registered full-signature metric.

**This does not establish a purely behavioral, information-matched decoder advantage:** the gauge-invariant effective-table probe derived only from native behavioral supervision remains near chance for complete signatures. The distinction between accessible temporal registers and incomplete goal-table representations is an empirical outcome of the experiment, and indicates where to focus later model-agnostic work.

The model's initial goal IDs and full weights were always available for controlled training and evaluation. None of these results addresses unknown hidden objectives in frontier systems, restoring deleted parameters, or off-machine persistence.

## 5. Reproduction and audit

- [Executable neural experiment](experiments/factorized_decoder.py): deterministic frozen-checkpoint loading; verified original model digests; new probe fitting; independent state sampling; native versus automaton targets; 1,000 bootstrap samples; full per-state rows.
- [Finite exact reference](factorized_latch_finite.py) and [unit tests](tests/test_factorized_latch_finite.py).
- [Frozen compact summary](results/comp22_summary.json), preserving model/probe hashes, cell metrics, exact count numerators, bootstrap interval and raw JSON SHA-256.
- [Standard-library audit](comp22_audit.py), which recomputes the registered pooled exact and micro accuracies from unrounded counts.
- [Lean module](lean/Shaft/FactorizedLatch.lean), importing the prior causal sufficiency module.

Both full 16,384-state local JSON runs, under PyTorch 2.10.0+cpu and NumPy 2.3.5, were **byte-identical**, with SHA-256 `d76bd5c53fb01d9a8a03ef5995b2c45fec3c377380937aef0bc50c6f6d8e7808`.

Run the standard-library and optional neural tests from the repo root:

```bash
python -m unittest discover -s research/shaft_compactification/tests -v
python -m research.shaft_compactification.factorized_latch_finite
python -m research.shaft_compactification.comp22_audit
python -m research.shaft_compactification.experiments.factorized_decoder \
  --frozen-root /path/to/extracted/COMP21/results \
  --per-goal 32 --out /tmp/comp22_full.json
cd research/shaft_compactification/lean && lake build
```

The full COMP-21 frozen checkpoint archive is an input prerequisite for the optional neural experiment. Lightweight CI checks finite mathematics, source syntax, compact receipt arithmetic and the Lean kernel, without retraining previous models.
