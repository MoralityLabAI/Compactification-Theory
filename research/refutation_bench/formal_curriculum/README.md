# FC-02: certified Horn argument and counterargument curriculum

This deterministic curriculum has 1,536 argument graphs: **1,152 train, 192
development, and 192 deeper structural-OOD test**. Half are valid entailments;
the other half have verified Boolean countermodels. The verifier checks every
case using operational Horn forward closure and exhaustive truth tables, two
separately implemented algorithms.

The historical RH, P vs NP, Collatz, four-colour and Fermat manuscript cases
stay in the original seed/expansion and are **not training examples** here.
FC-02 is an exact propositional surrogate for recognizing missing premises,
reversed implication directions and unused proof dependencies.

## Split discipline

| Split | Count | Dependency depth | Exclusive graph families |
| --- | ---: | ---: | --- |
| Train | 1,152 | 2–4 | linear, sidechain, fanin |
| Development | 192 | 5–6 | ladder, diamond |
| Structural OOD | 192 | 7–8 | deepjoin, cycle |

Each split balances valid and refuted labels. Inputs receive randomized
proposition names and shuffled rule order. Structural WL fingerprints group
potentially equivalent tasks conservatively; no fingerprint repeats across
splits. WL is a heuristic and does not establish exact graph nonisomorphism.
The generated OOD test is held out from gradients, while remaining publicly
reproducible, so this is not a private blind challenge.

The forward model sees only facts, Horn rules, proposition nodes and goal
identity. Labels, countermodels, known fixed points, split names, families,
fingerprints and proof traces stay in the supervision/receipt fields.

## Independent checks

Each problem contains at most 12 atomic propositions and 20 implication
rules. The negative assignment is the least Horn model: every premise is true
and the goal is false. The independent propositional verifier inspects that
assignment and also exhaustively checks whether *any* countermodel exists.

All 768 positive examples generate theorem declarations with explicit
proposition/fact/implication parameters and concrete Lean proof terms. Their
compilation succeeds with pinned **Lean 4.19.0**. Printed axiom reports for the
first and last declarations say that those theorems depend on no axioms.

From the repository root:

~~~sh
python -m research.refutation_bench.formal_curriculum.generator \
  --out /tmp/formal_curriculum.jsonl \
  --lean /tmp/FormalCurriculum.lean \
  --receipt /tmp/formal_curriculum_receipt.json
python -m unittest -v research.refutation_bench.tests.test_formal_curriculum
lean /tmp/FormalCurriculum.lean
~~~

[Successful CI run 43240137061](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43240137061)
ran eight new tests, validated all 1,536 finite examples and compiled 768
positive theorems.

SHA-256 of canonical generated dataset:
8e86a41dd93ff162072799332ebbef4c7134951798eb4af7ebb4ab044bebd56f

SHA-256 of generated Lean file:
fe66347cd1a12f5e8b09d4b06ed1508cb12c0c466377250e9c830d9e271166c1

The compiler verifies the positive Horn arguments, while the finite checker
verifies negative witnesses. These results concern bounded propositional
logic only. They neither prove nor disprove the famous problems whose
manuscripts motivated RefutationBench.

## Recurrent student experiment

A [LoopedTransformers LR-01 student](https://github.com/MoralityLabAI/LoopedTransformers/tree/feat/lr01-refutationbench-ci/experiments/refutation_transfer)
uses the original tied TypedLoop cell to learn whether each goal is derivable
and to propose complete Boolean assignments for invalid entailments.

The independent countermodel checker either accepts its unmodified predicted
assignment or marks it unverified; no oracle repairs the proposal. One-pass
and four-pass policies share parameters but spend different compute per
example. The held-out graph-family test measures transfer beyond training
depths, with exact solver (100%) and constant-verdict (50%) controls.

Extensions under consideration include proof dependency scopes, true
first-fault localization, alternative derivations, disjunctions, and raw
manuscript audits. Those are separate benchmark tasks and require new
verification contracts.
