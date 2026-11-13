# FC-03: pre-registered proof-graph-family transfer

The [preregistration](PREREGISTRATION.md) was frozen in commit
`650bd2cc5f0fd628784f672248c3a3e362e9dcdf`, **before**
generating FC-03 test data, training FC-03 models or inspecting results.

This is a wholly synthetic, exactly checkable extension of RefutationBench.
It tests whether a shared LR-01 reasoning cell transfers across structural
Horn-logic families. The source paper audits remain separate.

## Frozen test collection

| Test split | Depth | Count | Valid arguments | Invalid arguments |
| --- | --- | ---: | ---: | ---: |
| `test_same` | 2–4 | 192 | 96 | 96 |
| `test_deep` | 6–8 | 192 | 96 | 96 |

The four held-out families, `fork_join`, `anchored_cycle`,
`alternate_routes`, and `guarded_ladder`, each contribute 48 tasks
to each split with 24 valid and 24 invalid. FC-02's 1,152 training
tasks use `linear`, `sidechain`, `fanin`. Its 192 development tasks
use `ladder`, `diamond`. No test-family training examples are admitted.

The dataset is public. It is preregistered but it is **not** a private
hidden benchmark or an untouched challenge set after results are viewed.
The generator rejects duplicate WL structural fingerprints against
the entire 1,536-example original corpus and against its own test cases.

## Ground truth

The source implementation constructs finite Horn-closure proofs and
countermodels. Every generated claim is checked by both forward chaining
and exhaustive Boolean truth tables. A valid argument yields a concrete
Lean 4.19.0 proof term under exactly the stated premises. A false entailment
yields a complete Boolean valuation satisfying all the premises with the
goal false. Negative certificates are verified using the original
RefutationBench exact checker.

[Source CI run 43352394607](https://github.com/MoralityLabAI/Compactification-Theory/actions/runs/43352394607)
passed all six tests, checked 384 arguments and their certificates,
and compiled all 192 valid Lean theorems. The first and last proof
terms reported no extra axioms.

Frozen source commit:
`c859f04cda1d142d4eb560148414b82d53f80705`

Test JSONL SHA-256:
`ddc04ec8553a24269b16b53e27d28b521bc2f928978bfd5ee8a9e5fc36463fdb`

Lean source SHA-256:
`7cc5ff644f9298608f44adb527f6606404f1eb6a2a39f0f7e5bc40565bea4476`

Run from the repository root:

~~~sh
python -m unittest -v research.refutation_bench.tests.test_fc03
python -m research.refutation_bench.fc03.generator \
  --out /tmp/fc03_test.jsonl \
  --lean /tmp/FC03.lean \
  --receipt /tmp/fc03_receipt.json
lean /tmp/FC03.lean
~~~

Training and evaluation are in the private
[LoopedTransformers FC-03 pull request](https://github.com/MoralityLabAI/LoopedTransformers/pull/23).
The train CI job receives the original FC-02 checkout only; the test
CI job downloads frozen weight artifacts and *then* receives the pinned
FC-03 generator. The primary comparison is four shared-weight passes
against four untied passes at the same width and operator budget.
A width-reduced untied control approximately matches parameter count.
The preregistration fixes seeds 17/29, 350 steps and the outcome measures.

The formal fragment can be solved perfectly by exact forward chaining.
A learner's higher accuracy is interesting as a study of recurrent
dynamics, with no claimed advantage over the symbolic solver.
