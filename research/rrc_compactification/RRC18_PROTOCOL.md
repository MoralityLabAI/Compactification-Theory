# RRC-18: Exploratory finite-class certification and recurrent identification

**Executed 2026-11-10; CPU-only, one fixed training seed.** This protocol
records the implementation and its frozen result after the initial pilot.
Earlier RRC-11 through RRC-17 experiments and their outcomes were known
when designing the test. The classes and holdout generators are public;
this is not externally preregistered or an independently blind study.

## Primary questions

1. Can deterministic normalization prove that the RRC-17 probabilistic
   quotient classes agree coefficientwise on **every integer unit series
   through degree 12**?
2. Given only two raw input/output demonstrations and a public topology,
   can a compact recurrent model rank the certified classes more efficiently
   than a class-mass symbolic prior at the same 12 exact-verification budget?
3. Does any advantage survive a **source-generator shift** from signed
   integer coefficient series to arithmetic-part partition products, and
   what happens when recurrence is replaced by a parameter-matched MLP?

## Split and leakage restrictions

`learn18.roster(include_test=False)` constructs just the 3,600 training
and 400 development records: **900 train and 100 dev per topology**,
using fresh signed integer unit-series sources. All four topologies are
represented in training, so the heldout split tests new input *instances*
and a new source family, rather than an unseen topology. Each source graph
is drawn from the 64 syntax programs. Training labels are cheapest exact
polynomial-class representatives of the training source graph; labels are
computed with the algebraic teacher. The model's input is 13 positions of
raw signed-log coefficients on two demos plus the public topology ID.
It receives no candidate execution, graph label or query future at
inference time. Feedforward receives the same data flattened.

Test-only groups, generated after the training roster and deduplicated by
visible input except for deliberate pairs:

| Group | Cases | Conditions |
|---|---:|---|
| Signed holdout | 240 | 60 fresh cases per topology, training source type |
| Partition-product shift | 160 | 40 per topology from arithmetic part-set products |
| Inconsistent observed examples | 20 | mutation of one public demo, rejected by all 64 candidates |
| Future-collision pairs | 16 | eight pairs with identical visible inputs and a later output mutation |

The test roster is now exposed and must be treated as development context in
future iterations. All 20 negative examples are generated before scoring and
accepted only after a complete public-grammar rejection audit. No test
model accuracy is used for data construction or checkpoint selection.

## Frozen model and computation budgets

- GRU: input width 9, 32 hidden units, one pass over 13 positions,
  64-class head, **6,240 FP32 parameters**; live hidden state 32 FP32 values.
- Full-prefix feedforward: 56 inputs, 51 hidden tanh units,
  64-class head, **6,235 FP32 parameters** (0.08% difference).
- Shuffled-target GRU: identical architecture, labels permuted within
  each topology's supervised roster, preserving class frequency.
- 900 Adam steps per model, batch 128, learning rate 0.003,
  one PyTorch CPU thread, same minibatch indices, separate fixed seeds.
  All checkpoints are frozen before heldout scoring.
- One *single* ranked class proposal or up to 12 distinct proposals,
  with no interpreter previews. Each paid proposal is checked against
  **both** public degree-12 demonstrations, even when the first fails.
- Exact offline comparator: size-ordered certified class representatives,
  also with 12 full candidate checks. Other baselines: certified canonical
  class order, first 12 syntax programs, exhaustive classes and exhaustive
  64 syntax programs. Exhaustive arms have a larger, explicitly recorded
  budget and are only a bounded-grammar upper reference.

## Required gates

1. Sparse polynomial normalization must implement all four supplied
   graph topologies, including formal reciprocal and unit Hadamard join.
   Canonical class partitions must match the RRC-17 probe partitions,
   with 25/24/33/24 classes by topology.
2. Independent small-input evaluations must agree with the original
   integer interpreter across all 256 graph/topology pairs; alternative
   integer inputs must never change a certified class identity.
3. Train/dev/test visible signals may not overlap, except the declared
   eight paired future collisions. `include_test=False` must not call the
   partition-product generator or create negative/holdout cases.
4. The exact verifier must reject all 20 inconsistent public demos.
   It must accept every in-grammar positive case with exhaustive classes.
   Every accepted candidate must match both public demonstrations.
5. Every charged graph query must make exactly two independent public
   example executions. Recurrent inference must make **zero** graph
   execution calls. All trace lengths, cached uniqueness and work estimates
   must be checked. Never add FP MACs and integer loop units together.
6. Future-collision pairs must produce identical candidate traces, and
   exhaustive search must produce precisely eight false future acceptances.
7. A full source- and checkpoint-hashed receipt must regenerate exactly.
   Independently retraining three checkpoints must reproduce JSON bytes.
   19 dedicated unit tests plus RRC-16/RRC-17 regressions are the CPU gate.

A recurrent claim requires greater *verified* recovery than the
parameter-matched feedforward and the symbolic class-mass ordering,
across both held-out source groups and after compute accounting. A
signed-source-only advantage is narrower. A shuffled-label control that
approaches trained performance on the product shift is reported as a
failure of robust learned generalization.

## Reproduce

Python 3.10+ standard library is sufficient to validate frozen weights,
run portable inference and execute exact arithmetic. From repository root:

```bash
python -m unittest discover -s research/rrc_compactification -p 'test_certify18.py' -v
python research/rrc_compactification/bench18.py --check
python -m unittest discover -s research/rrc_compactification -p 'test_quotient17.py' -v
python -m unittest discover -s research/rrc_compactification -p 'test_topology16.py' -v
```

Optional CPU PyTorch independent retraining:

```bash
python research/rrc_compactification/train18.py --out-dir /tmp/rrc18-retrain
for kind in recurrent feedforward shuffled; do
  cmp /tmp/rrc18-retrain/rrc18_${kind}.json research/rrc_compactification/results/rrc18_${kind}.json
done
```

The receipt hashes all source modules, model bytes, class assignments,
roster identities and per-method traces. CI does not perform training or
fetch model weights. A source hash mismatch is a failed gate.
