# FC-05: a proof certificate on either side of a Horn entailment

The [FC-05 preregistration](PREREGISTRATION.md) was committed as
`e6ab5716e3316441adad131b087a50d1ca0f2734`, before creating
the dataset or running any FC-05 neural inference.

RefutationBench's earlier FC-04 system could issue a verified negative
countermodel, while its positive output was provisional. FC-05 adds
a **model-proposed action trace** for the positive side. A separate
source-only proof checker replays every inference, and a Lean 4.19.0
kernel run compiles the exact trace. No model failure is silently
repaired by a reference theorem or exhaustive solver.

## Frozen source set

FC-05 provides **384 new cases** across four previously unused graph
families: `split_relay`, `cross_gate`, `detour_join`,
`layered_cycle`.

| Held-out split | Depth | Total | Valid | Invalid |
| --- | --- | ---: | ---: | ---: |
| `test_same` | 2–4 | 192 | 96 | 96 |
| `test_deep` | 6–8 | 192 | 96 | 96 |

Each family supplies 24 valid and 24 invalid examples to each split.
Every case is compared against all 2,304 earlier FC-02, FC-03 and FC-04
cases by a conservative structural WL fingerprint to rule out repeated
whole-graph instances.

The independent truth-table checker establishes all finite entailment
labels and verifies 192 negative Boolean witnesses. The generated 192
valid theorems compile as explicit Lean proposition/implication
hypotheses and proof terms.

[First source CI after a test-harness correction](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43399156073)
passed 8 tests, verified all 384 cases, compiled 192 Lean proofs and
reported no extra axioms for the two endpoint theorem checks.
The first test run rejected an overlong reference trace that contained
legal steps after reaching the goal; the test was corrected to stop
at its first goal derivation. The independent checker continues to
reject unnecessary trailing model actions.

Pinned source commit for model evaluation:
`e2fd9f6a0bc7466475541a7ee2afd1320b4e0366`

Canonical source-test JSONL SHA-256:
`faced2fce3ecc176a86871211ec013c1a8deb1a83f3b5e89f9cb650c979dd722`

Generated reference Lean source SHA-256:
`2f370136c147ed2c66fe0e80a9416321f794a48bf0769d39d3c13db4aa78aec9`

## Independent checker

`proofs.py` verifies the given rule indices in their submitted order.
It requires every rule to be present in the original graph, every
antecedent to be known at the claimed time of application, every
consequent to be novel, and the requested goal to be derived within
ten actions. It rejects trailing model actions after the goal.
The Lean renderer accepts only a verified trace and reproduces
that same sequence of inference applications. At model evaluation,
each accepted generated theorem receives another Lean kernel
compilation, independent of the Python replay checker.

The proof checker is deliberately a strict replay verifier.
It never searches for missing steps. The exact forward solver is
available as an upper-bound control, outside neural proof generation.

From the repository root:

~~~sh
python -m unittest -v research.refutation_bench.tests.test_fc05
python -m research.refutation_bench.fc05.generator \
    --out /tmp/FC05.jsonl \
    --lean /tmp/FC05_reference.lean \
    --receipt /tmp/FC05_receipt.json
lean /tmp/FC05_reference.lean
~~~

## Neural training

The model training and test scripts are in
`MoralityLabAI/LoopedTransformers/experiments/refutation_transfer/`,
under the stacked FC-05 PR.

The 11,521-parameter shared LR-01 recurrent trunk is trained only
on the existing FC-02 train set, and an additional 4,705-parameter
rule-priority head is trained on **training-only** shortest-trace
labels. The FC-04 stopping threshold is calibrated on FC-02 dev.
The evaluator then allows up to ten legal rule applications. It
compares the learned head with uniformly random legal choices and
an immediate-goal-first policy under the same action budget.

Both model-generated positive and negative claims are checked
independently. A failed positive proposal becomes an abstention;
positive certificate success, computational cost, and any advantage
over the heuristic controls are all measured separately.

This is a synthetic propositional logic experiment. Any proof of a
human mathematical manuscript requires an independently verified
formal interpretation of the manuscript's claim and argument.
